"""
server/simulator.py — Core discrete-event simulation engine.

Orchestrates the service graph, failure injection, metric evolution,
propagation, log generation, and trace generation into a coherent
per-tick simulation loop.

Fully deterministic: random.Random(seed) exclusively.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from server.failures import (
    FailureSpec,
    FailureType,
    apply_failure_to_metrics,
    make_failure_spec,
)
from server.graph import ServiceGraph, ServiceNode, generate_graph
from server.logs import generate_healthy_log, generate_log_message
from server.propagation import (
    CircuitBreaker,
    ServiceRuntimeState,
    propagate_failures,
)
from server.traces import generate_trace


# ---------------------------------------------------------------------------
# SLO targets
# ---------------------------------------------------------------------------

# Per-difficulty SLO thresholds: a service is "meeting SLO" if ALL conditions hold
SLO_TARGETS = {
    "easy":   {"max_error_rate": 0.05, "max_p99_ms": 500,  "max_cpu": 85, "max_memory": 90},
    "medium": {"max_error_rate": 0.05, "max_p99_ms": 1000, "max_cpu": 90, "max_memory": 90},
    "hard":   {"max_error_rate": 0.05, "max_p99_ms": 2000, "max_cpu": 95, "max_memory": 95},
}


def _service_meets_slo(state: ServiceRuntimeState, difficulty: str) -> bool:
    targets = SLO_TARGETS[difficulty]
    return (
        state.error_rate <= targets["max_error_rate"]
        and state.latency_p99_ms <= targets["max_p99_ms"]
        and state.cpu_pct <= targets["max_cpu"]
        and state.memory_pct <= targets["max_memory"]
    )


# ---------------------------------------------------------------------------
# Pending action effects (delayed remediation)
# ---------------------------------------------------------------------------

@dataclass
class PendingEffect:
    """A remediation action effect that resolves after a delay."""
    action_type: str
    target_service: str
    params: Dict[str, Any]
    resolve_tick: int   # Tick at which this effect takes place


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclass
class Simulator:
    """
    Core simulation engine.

    Usage:
        sim = Simulator()
        obs_data = sim.reset(seed=42, difficulty="easy")
        obs_data = sim.step(action_type="inspect_logs", params={"service_id": "order-service"})
    """

    # --- Graph and topology ---
    graph: Optional[ServiceGraph] = None
    difficulty: str = "easy"

    # --- Mutable per-service state ---
    services: Dict[str, ServiceRuntimeState] = field(default_factory=dict)

    # --- Failure injection ---
    failures: List[FailureSpec] = field(default_factory=list)
    failure_onset_tick: Dict[str, int] = field(default_factory=dict)  # service_id → tick failure started

    # --- Simulation state ---
    tick: int = 0
    max_steps: int = 10
    terminated: bool = False
    termination_reason: Optional[str] = None

    # --- Pending remediation effects ---
    pending_effects: List[PendingEffect] = field(default_factory=list)

    # --- Action history ---
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)

    # --- Deploy history ---
    deploys: List[Dict[str, Any]] = field(default_factory=list)

    # --- Diagnostic output (from inspect_* actions, consumed by observation builder) ---
    last_logs: Optional[str] = None
    last_metric_history: Optional[List[Dict[str, Any]]] = None
    last_traces: Optional[Dict[str, Any]] = None

    # --- Metric history per service (for inspect_metrics) ---
    metric_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # --- RNG ---
    rng: random.Random = field(default_factory=random.Random)

    # --- Remediation tracking ---
    remediated_services: Dict[str, int] = field(default_factory=dict)  # service_id → tick remediated

    def reset(
        self,
        seed: int,
        difficulty: str,
        failure_specs: Optional[List[FailureSpec]] = None,
    ) -> None:
        """Initialize a new episode. Call get_observation() after this."""
        self.rng = random.Random(seed)
        self.difficulty = difficulty
        self.tick = 0
        self.terminated = False
        self.termination_reason = None
        self.pending_effects = []
        self.actions_taken = []
        self.deploys = []
        self.last_logs = None
        self.last_metric_history = None
        self.last_traces = None
        self.metric_history = {}
        self.remediated_services = {}

        # Step budgets
        budgets = {"easy": 10, "medium": 20, "hard": 50}
        self.max_steps = budgets.get(difficulty, 10)

        # Generate graph
        self.graph = generate_graph(difficulty, self.rng)

        # Initialize runtime state for each service
        self.services = {}
        for node in self.graph.nodes:
            state = ServiceRuntimeState(
                service_id=node.id,
                arrival_rate=node.base_arrival_rate,
                service_time_local=node.base_service_time_local,
                thread_pool_size=node.thread_pool_size,
                replicas=node.default_replicas,
                version=node.default_version,
                timeout_ms=node.default_timeout_ms,
                retry_max=node.default_retry_max,
                retry_backoff=node.default_retry_backoff,
                pool_size=node.default_pool_size,
            )
            # Initialize circuit breakers for dependencies
            for dep_id in self.graph.adjacency.get(node.id, []):
                state.circuit_breakers[dep_id] = CircuitBreaker(
                    error_threshold=node.default_circuit_breaker_threshold,
                )
            self.services[state.service_id] = state
            self.metric_history[state.service_id] = []

        # Inject failures
        self.failures = failure_specs or []
        self.failure_onset_tick = {}
        for spec in self.failures:
            self.failure_onset_tick[spec.service_id] = 0
            svc = self.services.get(spec.service_id)
            if svc:
                svc.has_active_failure = True
                # Apply bad deploy version
                if spec.failure_type == FailureType.BAD_DEPLOY and spec.bad_version:
                    svc.previous_version = svc.version
                    svc.version = spec.bad_version
                    self.deploys.append({
                        "service": spec.service_id,
                        "version": spec.bad_version,
                        "ticks_ago": 0,
                    })

        # Run initial tick of failure evolution
        self._evolve_failures()
        self._run_propagation()
        self._record_metrics()

    def step(self, action_type: str, params: Dict[str, Any]) -> float:
        """
        Execute one agent action and advance the simulation by one tick.
        Returns the step reward (dense Δ-SLO shaping).
        """
        if self.terminated:
            return 0.0

        prev_slo = self.get_slo_score()

        # Clear diagnostic output from previous step
        self.last_logs = None
        self.last_metric_history = None
        self.last_traces = None

        # Process the action
        action_record = self._process_action(action_type, params)
        self.actions_taken.append(action_record)

        # Advance tick
        self.tick += 1

        # Resolve pending effects
        self._resolve_pending_effects()

        # Evolve failures (for non-remediated services)
        self._evolve_failures()

        # Run propagation
        self._run_propagation()

        # Record metric history
        self._record_metrics()

        # Update deploy ticks_ago
        for d in self.deploys:
            d["ticks_ago"] += 1

        # Compute reward
        new_slo = self.get_slo_score()
        reward = self._compute_reward(prev_slo, new_slo, action_type, action_record)

        # Check termination
        self._check_termination()

        return reward

    # -------------------------------------------------------------------
    # Action processing
    # -------------------------------------------------------------------

    def _process_action(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process an agent action. Returns an action record dict."""
        service_id = params.get("service_id")
        record = {
            "tick": self.tick,
            "action": action_type,
            "target": service_id,
            "success": False,
            "note": None,
        }

        if action_type == "noop":
            record["success"] = True
            record["note"] = "Waited and observed"
            return record

        if action_type == "inspect_logs":
            return self._do_inspect_logs(service_id, record)
        elif action_type == "inspect_metrics":
            return self._do_inspect_metrics(service_id, record)
        elif action_type == "inspect_traces":
            return self._do_inspect_traces(service_id, record)
        elif action_type == "restart_service":
            return self._do_restart(service_id, record)
        elif action_type == "rollback_service":
            return self._do_rollback(service_id, record)
        elif action_type == "scale_service":
            return self._do_scale(service_id, params, record)
        elif action_type == "tune_config":
            return self._do_tune_config(service_id, params, record)
        elif action_type == "clear_cache":
            return self._do_clear_cache(params, record)
        elif action_type == "rebalance_traffic":
            return self._do_rebalance_traffic(params, record)
        elif action_type == "pause_job":
            return self._do_pause_job(params, record)
        else:
            record["note"] = f"Unknown action type: {action_type}"
            return record

    def _do_inspect_logs(self, service_id: Optional[str], record: Dict) -> Dict:
        svc = self.services.get(service_id or "")
        if not svc:
            record["note"] = f"Service '{service_id}' not found"
            return record

        record["success"] = True
        # Generate log output based on service state
        logs_lines = []
        failure = self._get_failure_for_service(service_id)
        if failure and svc.error_rate > 0.01:
            dep = self._get_primary_dependency(service_id)
            for _ in range(self.rng.randint(3, 6)):
                logs_lines.append(generate_log_message(
                    failure.failure_type, service_id, self.rng,
                    dependency=dep,
                    error_rate=svc.error_rate,
                    memory_pct=svc.memory_pct,
                    p99_ms=svc.latency_p99_ms,
                    pool_pct=svc.connection_pool_usage_pct,
                    version=svc.version,
                    config_key=failure.broken_config_key or "unknown",
                    config_value=failure.broken_config_value or "unknown",
                    region=self.graph.node_map[service_id].region if self.graph and service_id in self.graph.node_map else "us-east-1",
                    throughput=svc.throughput_rps,
                ))
        elif svc.error_rate > 0.01:
            # Propagated errors — show upstream dependency issues
            dep = self._get_primary_dependency(service_id)
            logs_lines.append(f"WARN  {service_id} Elevated error rate: {svc.error_rate*100:.1f}%. Upstream dependency {dep} may be degraded.")
            logs_lines.append(f"ERROR {service_id} Request to {dep} failed: timeout after {svc.timeout_ms}ms. Retry 1/{svc.retry_max}.")
        else:
            logs_lines.append(generate_healthy_log(service_id, self.rng))

        self.last_logs = "\n".join(logs_lines)
        return record

    def _do_inspect_metrics(self, service_id: Optional[str], record: Dict) -> Dict:
        svc = self.services.get(service_id or "")
        if not svc:
            record["note"] = f"Service '{service_id}' not found"
            return record

        record["success"] = True
        self.last_metric_history = self.metric_history.get(service_id, [])[-10:]
        return record

    def _do_inspect_traces(self, service_id: Optional[str], record: Dict) -> Dict:
        svc = self.services.get(service_id or "")
        if not svc or not self.graph:
            record["note"] = f"Service '{service_id}' not found"
            return record

        record["success"] = True
        errors = {sid: s.error_rate for sid, s in self.services.items()}
        latencies = {sid: s.latency_p99_ms for sid, s in self.services.items()}
        self.last_traces = generate_trace(
            service_id, self.graph, errors, latencies, self.rng,
        )
        return record

    def _do_restart(self, service_id: Optional[str], record: Dict) -> Dict:
        svc = self.services.get(service_id or "")
        if not svc:
            record["note"] = f"Service '{service_id}' not found"
            return record

        failure = self._get_failure_for_service(service_id)
        # Restart fixes: CRASH, RESOURCE_LEAK (temporarily), CONFIG_STARTUP (if config was fixed)
        if failure and failure.failure_type in (FailureType.CRASH, FailureType.RESOURCE_LEAK):
            delay = self.rng.randint(1, 2)
            self.pending_effects.append(PendingEffect(
                action_type="restart_service",
                target_service=service_id,
                params={},
                resolve_tick=self.tick + delay,
            ))
            record["success"] = True
            record["note"] = f"Restarting {service_id}, effect in {delay} tick(s)"
        elif failure and failure.failure_type == FailureType.CONFIG_STARTUP:
            # Config startup: restart alone doesn't fix it (need tune_config first)
            record["success"] = True
            record["note"] = f"Restarted {service_id} but config error persists — fix config first"
        elif failure:
            # Restart gives temporary relief for other failures
            delay = self.rng.randint(1, 2)
            self.pending_effects.append(PendingEffect(
                action_type="restart_partial",
                target_service=service_id,
                params={},
                resolve_tick=self.tick + delay,
            ))
            record["success"] = True
            record["note"] = f"Restarting {service_id}, partial recovery expected in {delay} tick(s)"
        else:
            record["success"] = True
            record["note"] = f"{service_id} is healthy, restart had no effect"
        return record

    def _do_rollback(self, service_id: Optional[str], record: Dict) -> Dict:
        svc = self.services.get(service_id or "")
        if not svc:
            record["note"] = f"Service '{service_id}' not found"
            return record

        if not svc.previous_version:
            record["note"] = f"No previous version to rollback to for {service_id}"
            return record

        failure = self._get_failure_for_service(service_id)
        if failure and failure.failure_type == FailureType.BAD_DEPLOY:
            delay = self.rng.randint(2, 3)
            self.pending_effects.append(PendingEffect(
                action_type="rollback_service",
                target_service=service_id,
                params={"version": svc.previous_version},
                resolve_tick=self.tick + delay,
            ))
            record["success"] = True
            record["note"] = f"Rolling back {service_id} to {svc.previous_version}, effect in {delay} tick(s)"
        else:
            record["success"] = True
            record["note"] = f"Rollback queued for {service_id} but issue may not be deploy-related"
            delay = self.rng.randint(2, 3)
            self.pending_effects.append(PendingEffect(
                action_type="rollback_service",
                target_service=service_id,
                params={"version": svc.previous_version},
                resolve_tick=self.tick + delay,
            ))
        return record

    def _do_scale(self, service_id: Optional[str], params: Dict, record: Dict) -> Dict:
        svc = self.services.get(service_id or "")
        if not svc:
            record["note"] = f"Service '{service_id}' not found"
            return record

        target_replicas = params.get("replicas", svc.replicas + 1)
        node = self.graph.node_map.get(service_id) if self.graph else None
        max_r = node.max_replicas if node else 8
        target_replicas = max(1, min(target_replicas, max_r))

        delay = self.rng.randint(2, 4)
        self.pending_effects.append(PendingEffect(
            action_type="scale_service",
            target_service=service_id,
            params={"replicas": target_replicas},
            resolve_tick=self.tick + delay,
        ))
        record["success"] = True
        record["note"] = f"Scaling {service_id} to {target_replicas} replicas, effect in {delay} tick(s)"
        return record

    def _do_tune_config(self, service_id: Optional[str], params: Dict, record: Dict) -> Dict:
        svc = self.services.get(service_id or "")
        if not svc:
            record["note"] = f"Service '{service_id}' not found"
            return record

        key = params.get("key", "")
        value = params.get("value", "")
        record["success"] = True
        record["target"] = service_id

        failure = self._get_failure_for_service(service_id)
        if failure and failure.failure_type in (FailureType.CONFIG_STARTUP, FailureType.CONFIG_RUNTIME):
            if key == failure.broken_config_key:
                # Correct fix!
                self.pending_effects.append(PendingEffect(
                    action_type="tune_config_fix",
                    target_service=service_id,
                    params={"key": key, "value": value},
                    resolve_tick=self.tick + 1,
                ))
                record["note"] = f"Config key '{key}' updated on {service_id}. Fix takes effect next tick."
            else:
                record["note"] = f"Config key '{key}' updated on {service_id}, but this may not be the broken key."
        else:
            # General config tune (e.g., timeout, retry)
            self._apply_config_immediately(svc, key, value)
            record["note"] = f"Config '{key}'={value} applied to {service_id}"
        return record

    def _do_clear_cache(self, params: Dict, record: Dict) -> Dict:
        cache_name = params.get("cache_name") or params.get("service_id", "")
        record["target"] = cache_name

        if not self.graph or cache_name not in self.graph.cache_services:
            record["note"] = f"'{cache_name}' is not a cache service"
            return record

        failure = self._get_failure_for_service(cache_name)
        if failure and failure.failure_type == FailureType.CACHE_FAILURE:
            self.pending_effects.append(PendingEffect(
                action_type="clear_cache",
                target_service=cache_name,
                params={},
                resolve_tick=self.tick + 1,
            ))
            record["success"] = True
            record["note"] = f"Flushing cache {cache_name}, recovery in 1 tick"
        else:
            record["success"] = True
            record["note"] = f"Cache {cache_name} flushed (was not failing)"
        return record

    def _do_rebalance_traffic(self, params: Dict, record: Dict) -> Dict:
        from_region = params.get("from_region", "")
        to_region = params.get("to_region", "")
        pct = params.get("pct", 50)
        record["target"] = f"{from_region}->{to_region}"

        if not self.graph or not self.graph.has_multiple_regions:
            record["note"] = "Traffic rebalancing only available in multi-region (hard) mode"
            return record

        delay = self.rng.randint(2, 3)
        self.pending_effects.append(PendingEffect(
            action_type="rebalance_traffic",
            target_service="",
            params={"from_region": from_region, "to_region": to_region, "pct": pct},
            resolve_tick=self.tick + delay,
        ))
        record["success"] = True
        record["note"] = f"Shifting {pct}% traffic from {from_region} to {to_region}, effect in {delay} tick(s)"
        return record

    def _do_pause_job(self, params: Dict, record: Dict) -> Dict:
        job_name = params.get("job_name") or params.get("service_id", "")
        record["target"] = job_name

        if not self.graph or job_name not in self.graph.background_jobs:
            record["note"] = f"'{job_name}' is not a background job service"
            return record

        svc = self.services.get(job_name)
        if svc:
            svc.arrival_rate *= 0.3  # Reduce load significantly
            record["success"] = True
            record["note"] = f"Background job on {job_name} paused, load reduced"
        return record

    # -------------------------------------------------------------------
    # Effect resolution
    # -------------------------------------------------------------------

    def _resolve_pending_effects(self) -> None:
        """Resolve pending effects that have reached their tick."""
        still_pending = []
        for effect in self.pending_effects:
            if self.tick >= effect.resolve_tick:
                self._apply_effect(effect)
            else:
                still_pending.append(effect)
        self.pending_effects = still_pending

    def _apply_effect(self, effect: PendingEffect) -> None:
        svc = self.services.get(effect.target_service)

        if effect.action_type == "restart_service":
            # Full restart: clears crash/leak failures
            if svc:
                self._remediate_service(effect.target_service)
                svc.memory_pct = 30.0  # Reset memory (leak fix)

        elif effect.action_type == "restart_partial":
            # Partial: temporary relief
            if svc:
                svc.error_rate *= 0.5
                svc.memory_pct = max(30.0, svc.memory_pct * 0.7)

        elif effect.action_type == "rollback_service":
            if svc:
                version = effect.params.get("version", svc.previous_version)
                svc.version = version
                svc.previous_version = None
                self._remediate_service(effect.target_service)
                self.deploys.append({
                    "service": effect.target_service,
                    "version": version,
                    "ticks_ago": 0,
                })

        elif effect.action_type == "scale_service":
            if svc:
                svc.replicas = effect.params.get("replicas", svc.replicas)

        elif effect.action_type == "tune_config_fix":
            self._remediate_service(effect.target_service)
            # If config_startup, also need a restart — but we apply partial fix
            failure = self._get_failure_for_service(effect.target_service)
            if failure and failure.failure_type == FailureType.CONFIG_STARTUP:
                # Config fixed + implicit restart
                if svc:
                    svc.error_rate = 0.02  # Near-zero while restarting

        elif effect.action_type == "clear_cache":
            self._remediate_service(effect.target_service)

        elif effect.action_type == "rebalance_traffic":
            # Reduce arrival rate in from_region, increase in to_region
            from_region = effect.params.get("from_region", "")
            to_region = effect.params.get("to_region", "")
            pct = effect.params.get("pct", 50) / 100.0
            if self.graph:
                for node in self.graph.nodes:
                    s = self.services.get(node.id)
                    if not s:
                        continue
                    if node.region == from_region:
                        s.arrival_rate *= (1 - pct)
                    elif node.region == to_region:
                        s.arrival_rate *= (1 + pct * 0.5)  # Some traffic absorbed

    def _remediate_service(self, service_id: str) -> None:
        """Mark a service as remediated — stop failure evolution."""
        self.remediated_services[service_id] = self.tick
        svc = self.services.get(service_id)
        if svc:
            svc.has_active_failure = False
            svc.failure_ticks = 0

    def _apply_config_immediately(self, svc: ServiceRuntimeState, key: str, value: Any) -> None:
        """Apply a config change that takes effect immediately."""
        if key == "timeout_ms":
            svc.timeout_ms = int(value)
        elif key == "retry_max":
            svc.retry_max = int(value)
        elif key == "pool_size":
            svc.pool_size = int(value)
        elif key == "retry_backoff":
            svc.retry_backoff = bool(value)

    # -------------------------------------------------------------------
    # Failure evolution
    # -------------------------------------------------------------------

    def _evolve_failures(self) -> None:
        """Evolve all active failures by one tick."""
        for spec in self.failures:
            sid = spec.service_id
            if sid in self.remediated_services:
                # Remediated — gradually recover
                svc = self.services.get(sid)
                if svc:
                    svc.error_rate = max(0.0, svc.error_rate * 0.5)
                    svc.latency_p99_ms = max(50.0, svc.latency_p99_ms * 0.7)
                    svc.cpu_pct = max(10.0, svc.cpu_pct * 0.8)
                    svc.memory_pct = max(25.0, svc.memory_pct * 0.9)
                    svc.connection_pool_usage_pct = max(5.0, svc.connection_pool_usage_pct * 0.7)
                    svc.status = svc.compute_status()
                continue

            svc = self.services.get(sid)
            if not svc:
                continue

            onset = self.failure_onset_tick.get(sid, 0)
            ticks_since = self.tick - onset

            node = self.graph.node_map.get(sid) if self.graph else None
            base_p99 = 100.0
            base_cpu = 15.0
            base_memory = 30.0
            base_pool = 10.0

            error_rate, p99_ms, cpu_pct, memory_pct, pool_pct = apply_failure_to_metrics(
                spec, ticks_since,
                base_error_rate=0.0,
                base_p99_ms=base_p99,
                base_cpu=base_cpu,
                base_memory=base_memory,
                base_pool=base_pool,
                rng=self.rng,
            )

            svc.error_rate = error_rate
            svc.update_latency_percentiles(base_p99, p99_ms / base_p99, self.rng)
            svc.cpu_pct = cpu_pct
            svc.memory_pct = memory_pct
            svc.connection_pool_usage_pct = pool_pct
            svc.failure_ticks = ticks_since
            svc.status = svc.compute_status()

    def _run_propagation(self) -> None:
        """Run propagation engine to cascade failures through the graph."""
        if not self.graph:
            return

        edge_activation = {}
        for edge in self.graph.edges:
            edge_activation[(edge.source, edge.target)] = edge.activation_probability

        propagate_failures(
            self.services,
            self.graph.adjacency,
            self.graph.reverse_adjacency,
            edge_activation,
            self.rng,
            current_tick=self.tick,
        )

    # -------------------------------------------------------------------
    # Metric recording
    # -------------------------------------------------------------------

    def _record_metrics(self) -> None:
        """Record current metrics snapshot for all services."""
        for sid, svc in self.services.items():
            self.metric_history[sid].append({
                "tick": self.tick,
                "error_rate": round(svc.error_rate, 4),
                "latency_p99_ms": round(svc.latency_p99_ms, 1),
                "cpu_pct": round(svc.cpu_pct, 1),
                "memory_pct": round(svc.memory_pct, 1),
                "pool_pct": round(svc.connection_pool_usage_pct, 1),
                "throughput_rps": round(svc.throughput_rps, 1),
                "status": svc.status,
            })

    # -------------------------------------------------------------------
    # Reward computation
    # -------------------------------------------------------------------

    def _compute_reward(
        self, prev_slo: float, new_slo: float,
        action_type: str, record: Dict,
    ) -> float:
        """Dense Δ-SLO reward with action-type penalties."""
        # Base: delta SLO (positive = improvement)
        delta = new_slo - prev_slo
        reward = delta * 10.0  # Scale up for signal strength

        # Bonus for reaching full recovery
        if new_slo >= 1.0:
            reward += 5.0

        # Penalty for invalid/failed actions
        if not record.get("success", False):
            reward -= 0.5

        # Small penalty for non-diagnostic actions (encourage efficiency)
        if action_type not in ("inspect_logs", "inspect_metrics", "inspect_traces", "noop"):
            reward -= 0.1  # Small cost for remediation actions

        # Penalty for redundant noops when system is degraded
        if action_type == "noop" and new_slo < 0.9:
            reward -= 0.2

        return round(reward, 4)

    # -------------------------------------------------------------------
    # Termination
    # -------------------------------------------------------------------

    def _check_termination(self) -> None:
        """Check if the episode should end."""
        slo = self.get_slo_score()

        # Success: all SLOs met
        if slo >= 1.0:
            self.terminated = True
            self.termination_reason = "resolved"
            return

        # Timeout: exceeded step budget
        if self.tick >= self.max_steps:
            self.terminated = True
            self.termination_reason = "timeout"
            return

        # System collapse: all services down
        down_count = sum(1 for s in self.services.values() if s.status == "down")
        if down_count == len(self.services) and len(self.services) > 0:
            self.terminated = True
            self.termination_reason = "failed"

    # -------------------------------------------------------------------
    # Observation helpers
    # -------------------------------------------------------------------

    def get_slo_score(self) -> float:
        """Fraction of services meeting SLO targets."""
        if not self.services:
            return 0.0
        meeting = sum(1 for s in self.services.values() if _service_meets_slo(s, self.difficulty))
        return meeting / len(self.services)

    def get_observation_summary(self) -> str:
        """Generate a natural-language summary of the current state."""
        slo = self.get_slo_score()
        total = len(self.services)
        healthy = sum(1 for s in self.services.values() if s.status == "healthy")
        degraded = sum(1 for s in self.services.values() if s.status == "degraded")
        critical = sum(1 for s in self.services.values() if s.status == "critical")
        down = sum(1 for s in self.services.values() if s.status == "down")

        parts = []
        if down > 0:
            parts.append(f"{down} service(s) DOWN")
        if critical > 0:
            parts.append(f"{critical} CRITICAL")
        if degraded > 0:
            parts.append(f"{degraded} degraded")
        if healthy > 0:
            parts.append(f"{healthy} healthy")

        status_str = ", ".join(parts) if parts else "all nominal"
        return f"Tick {self.tick}/{self.max_steps}: SLO compliance {slo*100:.0f}% ({status_str}). {total} services total."

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Generate active alerts from current service states."""
        alerts = []
        for sid, svc in self.services.items():
            if svc.error_rate >= 0.50:
                alerts.append({
                    "severity": "critical",
                    "service": sid,
                    "type": "error_rate_high",
                    "message": f"{sid} error rate at {svc.error_rate*100:.0f}%",
                    "first_seen_tick": max(0, self.tick - svc.failure_ticks),
                })
            elif svc.error_rate >= 0.05:
                alerts.append({
                    "severity": "warning",
                    "service": sid,
                    "type": "error_rate_high",
                    "message": f"{sid} error rate elevated at {svc.error_rate*100:.1f}%",
                    "first_seen_tick": max(0, self.tick - svc.failure_ticks),
                })

            if svc.latency_p99_ms >= 5000:
                alerts.append({
                    "severity": "critical",
                    "service": sid,
                    "type": "latency_high",
                    "message": f"{sid} p99 latency {svc.latency_p99_ms:.0f}ms",
                    "first_seen_tick": max(0, self.tick - svc.failure_ticks),
                })
            elif svc.latency_p99_ms >= 1000:
                alerts.append({
                    "severity": "warning",
                    "service": sid,
                    "type": "latency_high",
                    "message": f"{sid} p99 latency elevated at {svc.latency_p99_ms:.0f}ms",
                    "first_seen_tick": max(0, self.tick - svc.failure_ticks),
                })

            if svc.status == "down":
                alerts.append({
                    "severity": "critical",
                    "service": sid,
                    "type": "service_down",
                    "message": f"{sid} is DOWN",
                    "first_seen_tick": max(0, self.tick - svc.failure_ticks),
                })

            if svc.memory_pct >= 90:
                alerts.append({
                    "severity": "warning",
                    "service": sid,
                    "type": "memory_high",
                    "message": f"{sid} memory at {svc.memory_pct:.0f}%",
                    "first_seen_tick": max(0, self.tick - svc.failure_ticks),
                })

            if svc.connection_pool_usage_pct >= 80:
                alerts.append({
                    "severity": "warning",
                    "service": sid,
                    "type": "connection_pool_saturated",
                    "message": f"{sid} connection pool at {svc.connection_pool_usage_pct:.0f}%",
                    "first_seen_tick": max(0, self.tick - svc.failure_ticks),
                })

            # Circuit breaker alerts
            for dep_id, breaker in svc.circuit_breakers.items():
                if breaker.state.value == "OPEN":
                    alerts.append({
                        "severity": "warning",
                        "service": sid,
                        "type": "circuit_breaker_open",
                        "message": f"{sid} circuit breaker OPEN for {dep_id}",
                        "first_seen_tick": max(0, self.tick - breaker.ticks_in_current_state),
                    })

        # Sort by severity (critical first)
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda a: severity_order.get(a["severity"], 9))
        return alerts

    def get_legal_actions(self) -> List[Dict[str, Any]]:
        """Return the set of currently legal actions with valid targets."""
        service_ids = list(self.services.keys())
        actions = [
            {"action_type": "noop", "valid_targets": []},
            {"action_type": "inspect_logs", "valid_targets": service_ids},
            {"action_type": "inspect_metrics", "valid_targets": service_ids},
            {"action_type": "inspect_traces", "valid_targets": service_ids},
            {"action_type": "restart_service", "valid_targets": service_ids},
        ]

        # Rollback: only services with previous versions
        rollback_targets = [sid for sid, s in self.services.items() if s.previous_version]
        if rollback_targets:
            actions.append({"action_type": "rollback_service", "valid_targets": rollback_targets})

        # Scale: all services
        actions.append({"action_type": "scale_service", "valid_targets": service_ids})

        # Tune config: all services
        actions.append({"action_type": "tune_config", "valid_targets": service_ids})

        # Clear cache: only cache services
        if self.graph and self.graph.cache_services:
            actions.append({"action_type": "clear_cache", "valid_targets": self.graph.cache_services})

        # Rebalance traffic: only in multi-region
        if self.graph and self.graph.has_multiple_regions:
            actions.append({
                "action_type": "rebalance_traffic",
                "valid_targets": self.graph.regions,
            })

        # Pause job: only background job services
        if self.graph and self.graph.background_jobs:
            actions.append({"action_type": "pause_job", "valid_targets": self.graph.background_jobs})

        return actions

    def get_service_observations(self) -> List[Dict[str, Any]]:
        """Build per-service observation dicts."""
        result = []
        for sid, svc in self.services.items():
            node = self.graph.node_map.get(sid) if self.graph else None
            deps = self.graph.adjacency.get(sid, []) if self.graph else []
            cb_states = {
                dep: breaker.state.value
                for dep, breaker in svc.circuit_breakers.items()
            }
            result.append({
                "id": sid,
                "layer": node.layer if node else "unknown",
                "status": svc.status,
                "error_rate": round(svc.error_rate, 4),
                "latency_p50_ms": round(svc.latency_p50_ms, 1),
                "latency_p95_ms": round(svc.latency_p95_ms, 1),
                "latency_p99_ms": round(svc.latency_p99_ms, 1),
                "throughput_rps": round(svc.throughput_rps, 1),
                "cpu_pct": round(svc.cpu_pct, 1),
                "memory_pct": round(svc.memory_pct, 1),
                "connection_pool_usage_pct": round(svc.connection_pool_usage_pct, 1),
                "replicas": svc.replicas,
                "version": svc.version,
                "previous_version": svc.previous_version,
                "depends_on": deps,
                "circuit_breakers": cb_states,
            })
        return result

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _get_failure_for_service(self, service_id: Optional[str]) -> Optional[FailureSpec]:
        if not service_id:
            return None
        for spec in self.failures:
            if spec.service_id == service_id and service_id not in self.remediated_services:
                return spec
        return None

    def _get_primary_dependency(self, service_id: Optional[str]) -> str:
        if not service_id or not self.graph:
            return "unknown"
        deps = self.graph.adjacency.get(service_id, [])
        return deps[0] if deps else "unknown"
