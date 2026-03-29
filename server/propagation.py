"""
server/propagation.py — Queueing-theory cascade engine.

Computes how failures propagate through the service dependency graph using:
- Little's Law: L = λ × S for thread pool saturation (ρ = L/T)
- Retry amplification: E[attempts] = (1 - p^(R+1)) / (1 - p)
- Per-hop dampening (~0.7 with circuit breakers) vs amplification (~1.2-1.8×)
- 1-2 tick propagation delay (not instant)
- Circuit breaker state machine: CLOSED → OPEN → HALF_OPEN → CLOSED

Sources: Google SRE Book, Netflix Hystrix, Docs/DataResearch.md Answer 3.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Circuit breaker state machine
# ---------------------------------------------------------------------------


class BreakerState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitBreaker:
    """Per-edge circuit breaker with rolling error window."""

    state: BreakerState = BreakerState.CLOSED

    # Config (tunable by agent via tune_config)
    error_threshold: float = 0.5      # Error rate to trip OPEN
    cooldown_ticks: int = 5           # Ticks to stay OPEN before half-open
    half_open_success_threshold: int = 3  # Successes needed to close

    # Runtime state
    ticks_in_current_state: int = 0
    error_window: List[float] = field(default_factory=list)
    window_size: int = 5
    half_open_successes: int = 0

    def record_error_rate(self, error_rate: float) -> None:
        """Record an error rate observation and potentially transition state."""
        self.error_window.append(error_rate)
        if len(self.error_window) > self.window_size:
            self.error_window = self.error_window[-self.window_size:]
        self.ticks_in_current_state += 1

    def tick(self, current_error_rate: float, rng: random.Random) -> BreakerState:
        """Advance the circuit breaker state machine by one tick."""
        self.record_error_rate(current_error_rate)
        avg_error = sum(self.error_window) / len(self.error_window) if self.error_window else 0.0

        if self.state == BreakerState.CLOSED:
            if avg_error >= self.error_threshold:
                self.state = BreakerState.OPEN
                self.ticks_in_current_state = 0
                self.half_open_successes = 0

        elif self.state == BreakerState.OPEN:
            if self.ticks_in_current_state >= self.cooldown_ticks:
                self.state = BreakerState.HALF_OPEN
                self.ticks_in_current_state = 0
                self.half_open_successes = 0

        elif self.state == BreakerState.HALF_OPEN:
            if current_error_rate < self.error_threshold * 0.5:
                self.half_open_successes += 1
                if self.half_open_successes >= self.half_open_success_threshold:
                    self.state = BreakerState.CLOSED
                    self.ticks_in_current_state = 0
                    self.error_window.clear()
            else:
                # Probe failed — go back to OPEN
                self.state = BreakerState.OPEN
                self.ticks_in_current_state = 0
                self.half_open_successes = 0

        return self.state

    @property
    def dampening_factor(self) -> float:
        """How much this breaker dampens downstream error propagation."""
        if self.state == BreakerState.OPEN:
            return 0.05   # Nearly all errors blocked (fail-fast)
        elif self.state == BreakerState.HALF_OPEN:
            return 0.3    # Some probe traffic gets through
        else:
            return 1.0    # No dampening


# ---------------------------------------------------------------------------
# Queueing theory functions
# ---------------------------------------------------------------------------


def compute_utilisation(
    arrival_rate: float,
    service_time: float,
    thread_pool_size: int,
) -> float:
    """
    Little's Law: L = λ × S (average items in system).
    Utilisation ρ = L / T where T is thread pool size.
    When ρ → 1.0, latency blows up nonlinearly (M/M/c queueing).
    """
    L = arrival_rate * service_time
    T = max(1, thread_pool_size)
    rho = L / T
    return min(rho, 1.0)  # Cap at 1.0 (saturated)


def compute_queueing_latency_multiplier(rho: float) -> float:
    """
    Approximate M/M/1 queueing delay multiplier.
    As ρ → 1, response time → ∞.
    Uses 1/(1-ρ) approximation with a cap to avoid infinity.
    """
    if rho >= 0.99:
        return 50.0   # ~50x baseline latency (effectively down)
    if rho >= 0.95:
        return 20.0   # ~20x
    if rho >= 0.90:
        return 10.0   # ~10x
    if rho >= 0.80:
        return 5.0    # ~5x
    if rho < 0.01:
        return 1.0    # No queueing
    return 1.0 / (1.0 - rho)


def compute_retry_amplification(
    failure_probability: float,
    max_retries: int,
) -> float:
    """
    Expected number of attempts with retries.
    E[attempts] = (1 - p^(R+1)) / (1 - p)
    where p = failure probability, R = max retries.
    """
    p = max(0.0, min(1.0, failure_probability))
    if p < 0.001:
        return 1.0  # No failures, no retries
    if p > 0.999:
        return float(max_retries + 1)  # Every attempt fails

    R = max(0, max_retries)
    return (1.0 - p ** (R + 1)) / (1.0 - p)


# ---------------------------------------------------------------------------
# Propagation engine
# ---------------------------------------------------------------------------


@dataclass
class ServiceRuntimeState:
    """Mutable runtime state for one service during simulation."""

    service_id: str

    # --- Current metrics (updated each tick) ---
    error_rate: float = 0.0
    latency_p50_ms: float = 20.0
    latency_p95_ms: float = 50.0
    latency_p99_ms: float = 100.0
    throughput_rps: float = 100.0
    cpu_pct: float = 15.0
    memory_pct: float = 30.0
    connection_pool_usage_pct: float = 10.0

    # --- Queueing model state ---
    arrival_rate: float = 100.0       # λ — requests/tick
    service_time_local: float = 0.05  # S_local — seconds per request
    thread_pool_size: int = 50        # T — max concurrent
    utilisation: float = 0.0          # ρ = L/T

    # --- Deployment ---
    replicas: int = 2
    version: str = "v1.0.0"
    previous_version: Optional[str] = None
    status: str = "healthy"  # healthy | degraded | critical | down

    # --- Config (tunable by agent) ---
    timeout_ms: int = 5000
    retry_max: int = 3
    retry_backoff: bool = False
    pool_size: int = 20

    # --- Circuit breakers (per-dependency) ---
    circuit_breakers: Dict[str, CircuitBreaker] = field(default_factory=dict)

    # --- Failure state ---
    has_active_failure: bool = False
    failure_ticks: int = 0
    propagation_error_rate: float = 0.0  # Error rate from upstream propagation

    def compute_status(self) -> str:
        """Derive health status from metrics."""
        if self.error_rate >= 0.90:
            return "down"
        elif self.error_rate >= 0.30 or self.latency_p99_ms >= 5000:
            return "critical"
        elif self.error_rate >= 0.05 or self.latency_p99_ms >= 1000:
            return "degraded"
        else:
            return "healthy"

    def update_latency_percentiles(self, base_p99: float, multiplier: float, rng: random.Random) -> None:
        """Update p50/p95/p99 from a base p99 and multiplier, with natural noise."""
        noise = rng.uniform(0.95, 1.05)
        self.latency_p99_ms = max(1.0, base_p99 * multiplier * noise)
        self.latency_p95_ms = self.latency_p99_ms * rng.uniform(0.60, 0.85)
        self.latency_p50_ms = self.latency_p95_ms * rng.uniform(0.30, 0.50)


def propagate_failures(
    services: Dict[str, ServiceRuntimeState],
    adjacency: Dict[str, List[str]],
    reverse_adjacency: Dict[str, List[str]],
    edge_activation: Dict[Tuple[str, str], float],
    rng: random.Random,
    propagation_delay: int = 1,
    current_tick: int = 0,
) -> None:
    """
    Propagate failure effects through the dependency graph for one tick.

    Each service that has errors causes downstream impact on its callers:
    1. Caller's arrival rate may spike (retries, cache miss stampede)
    2. Caller's service time increases (waiting on slow downstream)
    3. Caller's thread pool fills up (blocked threads)
    4. Circuit breakers may trip (dampening propagation)

    This modifies ServiceRuntimeState in-place.
    """
    # Process in reverse topological order: infra → business → edge
    # So downstream failures propagate to upstream callers
    for service_id, state in services.items():
        if state.error_rate < 0.01:
            continue  # Healthy — no propagation from this service

        # Who calls this service? (reverse edges = callers)
        callers = reverse_adjacency.get(service_id, [])

        for caller_id in callers:
            caller = services.get(caller_id)
            if caller is None:
                continue

            edge_key = (caller_id, service_id)
            activation_prob = edge_activation.get(edge_key, 1.0)

            # Is this edge active this tick?
            if rng.random() > activation_prob:
                continue  # Edge not active — this dependency not called

            # Get circuit breaker for this edge
            if service_id not in caller.circuit_breakers:
                caller.circuit_breakers[service_id] = CircuitBreaker()
            breaker = caller.circuit_breakers[service_id]

            # Update circuit breaker state
            breaker.tick(state.error_rate, rng)
            dampening = breaker.dampening_factor

            # --- Compute propagated impact ---

            # 1. Error propagation (dampened by circuit breaker)
            propagated_error = state.error_rate * dampening * rng.uniform(0.5, 0.9)
            caller.propagation_error_rate = max(
                caller.propagation_error_rate,
                propagated_error,
            )

            # 2. Retry amplification (increases arrival rate)
            if dampening > 0.1:  # Only retries if breaker isn't fully open
                retry_mult = compute_retry_amplification(
                    state.error_rate * dampening,
                    caller.retry_max,
                )
                caller.arrival_rate *= min(retry_mult, 3.0)  # Cap at 3x

            # 3. Latency propagation (waiting on slow downstream)
            if state.latency_p99_ms > 500 and dampening > 0.1:
                downstream_wait = state.latency_p99_ms * dampening * 0.001  # ms → seconds
                caller.service_time_local += downstream_wait * 0.5  # Partial impact

    # --- After propagation: update utilisation and derived metrics ---
    for service_id, state in services.items():
        # Recompute utilisation
        state.utilisation = compute_utilisation(
            state.arrival_rate / max(1, state.replicas),  # Per-replica arrival rate
            state.service_time_local,
            state.thread_pool_size,
        )

        # Apply queueing delay to latency
        q_mult = compute_queueing_latency_multiplier(state.utilisation)
        if q_mult > 1.1:
            base_p99 = 100.0  # Baseline p99 in ms
            state.update_latency_percentiles(base_p99, q_mult, rng)

        # Combine direct failure error rate with propagation error rate.
        # Services with no direct failure recover naturally when upstream heals.
        if state.has_active_failure:
            combined_error = max(state.error_rate, state.propagation_error_rate)
        else:
            combined_error = state.propagation_error_rate
        state.error_rate = min(1.0, combined_error)

        # Compute throughput (inverse of error rate, scaled by arrival)
        state.throughput_rps = state.arrival_rate * (1.0 - state.error_rate) / max(1, state.replicas)

        # Update status
        state.status = state.compute_status()

        # Reset per-tick propagation accumulator
        state.propagation_error_rate = 0.0
