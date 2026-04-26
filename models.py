"""
SevZero — Typed Pydantic models for Action, Observation, and State.

These are the public API contracts at the package root (OpenEnv requirement).
Every field is documented because the observation JSON must be self-explanatory
to any LLM evaluator without additional context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Sub-models: nested inside SevZeroObservation
# ---------------------------------------------------------------------------


class ServiceInfoModel(BaseModel):
    """
    All observable per-service metrics, ordered by SRE triage priority:
    symptoms first, traffic second, saturation third, context last.
    """

    # Identity
    id: str = Field(description="Service identifier, e.g. 'payment-service'")
    layer: str = Field(
        description="Service layer: 'edge' | 'domain' | 'infra' | 'cross-cutting'"
    )
    status: str = Field(
        description="Aggregate health: 'healthy' | 'degraded' | 'critical' | 'down'"
    )

    # --- Symptoms (error + latency) ---
    error_rate: float = Field(
        description="Fraction of requests failing this tick (0.0–1.0)"
    )
    latency_p50_ms: float = Field(description="Median request latency in milliseconds")
    latency_p95_ms: float = Field(description="95th-percentile latency in milliseconds")
    latency_p99_ms: float = Field(description="99th-percentile latency in milliseconds")

    # --- Traffic ---
    throughput_rps: float = Field(
        description="Successful requests served per tick"
    )

    # --- Saturation ---
    cpu_pct: float = Field(description="CPU utilisation 0–100")
    memory_pct: float = Field(description="Memory utilisation 0–100")
    connection_pool_usage_pct: float = Field(
        description="DB connection pool saturation 0–100; high = I/O bottleneck"
    )

    # --- Deployment context ---
    replicas: int = Field(description="Number of running replicas")
    version: str = Field(description="Currently deployed version tag")
    previous_version: Optional[str] = Field(
        default=None,
        description="Previous version available for rollback; null if never changed",
    )

    # --- Dependency graph ---
    depends_on: List[str] = Field(
        default_factory=list,
        description="Direct service dependencies (downstream calls)",
    )
    circuit_breakers: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-dependency circuit breaker state. "
            "Keys are dependency IDs; values are 'CLOSED' | 'OPEN' | 'HALF_OPEN'."
        ),
    )


class AlertInfo(BaseModel):
    """A structured active alert, ordered by severity."""

    severity: str = Field(description="'critical' | 'warning' | 'info'")
    service: str = Field(description="Service ID that triggered the alert")
    type: str = Field(
        description=(
            "Alert category: 'error_rate_high' | 'latency_high' | "
            "'circuit_breaker_open' | 'connection_pool_saturated' | "
            "'memory_high' | 'cpu_high' | 'service_down'"
        )
    )
    message: str = Field(description="Human-readable alert description with metric values")
    first_seen_tick: int = Field(description="Tick at which this alert first fired")


class DeployInfo(BaseModel):
    """A recent deployment event visible in the observation."""

    service: str = Field(description="Service that was deployed")
    version: str = Field(description="New version deployed")
    ticks_ago: int = Field(description="How many ticks ago the deploy happened")


class ActionRecord(BaseModel):
    """A previously taken action, shown in the observation for agent context."""

    tick: int = Field(description="Tick at which the action was executed")
    action: str = Field(description="Action type, e.g. 'restart_service'")
    target: Optional[str] = Field(default=None, description="Primary target service/resource")
    success: bool = Field(description="Whether the action completed successfully")
    note: Optional[str] = Field(
        default=None,
        description="Extra context, e.g. 'service already healthy' or error reason",
    )


class LegalAction(BaseModel):
    """One type of action the agent is currently allowed to take."""

    action_type: str = Field(
        description=(
            "One of: inspect_logs | inspect_metrics | inspect_traces | "
            "restart_service | rollback_service | scale_service | tune_config | "
            "clear_cache | rebalance_traffic | pause_job | request_approval | noop"
        )
    )
    valid_targets: List[str] = Field(
        description="Service IDs (or other resource names) this action can target right now"
    )


# ---------------------------------------------------------------------------
# Top-level OpenEnv models
# ---------------------------------------------------------------------------


class SevZeroAction(Action):
    """
    An action the agent takes in SevZero.

    Choose exactly one action_type and provide the required params for it:

      inspect_logs(service_id)         -> logs: str in next observation
      inspect_metrics(service_id)      -> metric_history in next observation
      inspect_traces(service_id)       -> traces in next observation
      restart_service(service_id)      -> restarts pod; 1-2 tick delay
      rollback_service(service_id)     -> reverts to previous_version; 2-3 tick delay
      scale_service(service_id, replicas=N)   -> adjusts replica count; 2-4 tick delay
      tune_config(service_id, key, value)     -> updates config param; 1 tick delay
      clear_cache(cache_name)          -> flushes cache; 1 tick delay
      rebalance_traffic(from_region, to_region, pct)  -> shifts traffic; 2-3 tick delay
      pause_job(job_name)              -> pauses background job; 1 tick delay
      request_approval(action_type, target, reason) -> asks manager for gating (oversight)
      noop()                           -> wait and observe; 0 ticks
    """

    action_type: str = Field(
        description=(
            "Which operation to perform. Must be one of the 11 action types. "
            "Must appear in legal_actions from the previous observation."
        )
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action parameters. Examples: "
            "{'service_id': 'payment-service'}, "
            "{'service_id': 'payment-service', 'replicas': 4}, "
            "{'service_id': 'payment-service', 'key': 'timeout_ms', 'value': 2000}"
        ),
    )


class SevZeroObservation(Observation):
    """
    Full observation returned by reset() and step().

    Fields are ordered by SRE triage priority: incident summary first,
    then per-service metrics, then alerts, then context, then agent state.

    The `done` and `reward` fields are inherited from Observation base.
    """

    # --- Episode context ---
    tick: int = Field(default=0, description="Current simulation tick (0-indexed)")
    episode_id: Optional[str] = Field(
        default=None, description="Unique ID for this episode"
    )
    task_id: str = Field(
        default="easy",
        description="Which task is running: 'easy' | 'medium' | 'hard'",
    )
    status: str = Field(
        default="playing",
        description=(
            "Episode status: 'playing' | 'resolved' (all SLOs met) | "
            "'failed' (system collapse) | 'timeout' (max steps exceeded)"
        ),
    )
    max_steps: int = Field(
        default=10, description="Step budget for this task (Easy=10, Medium=20, Hard=50)"
    )

    # --- Health summary ---
    global_slo_score: float = Field(
        default=0.0,
        description="Fraction of services currently meeting all SLO targets (0.0–1.0)",
    )
    observation_summary: str = Field(
        default="",
        description=(
            "One-sentence natural-language summary of the current situation. "
            "Read this first — it gives you the critical context for your next action."
        ),
    )

    # --- Per-service state ---
    services: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Full state for every service in the cluster. "
            "See ServiceInfoModel for field definitions."
        ),
    )
    cluster: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "When schema drift renames the envelope, the service list may appear "
            "under cluster.services; otherwise null."
        ),
    )
    schema_version: str = Field(
        default="v1",
        description="Observation schema tag; drift episodes use v1.2-drift when enabled.",
    )
    schema_changelog: List[str] = Field(
        default_factory=list,
        description="Plain-English list of active schema drift mutations, if any.",
    )

    # --- Active alerts ---
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active alerts sorted by severity (critical first). See AlertInfo.",
    )

    # --- Context ---
    recent_deploys: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Deployments in the last 10 ticks. Correlate with error onset.",
    )
    actions_taken: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Last 10 actions taken in this episode, for agent context.",
    )

    # --- Action space ---
    legal_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Exactly what actions are available right now with valid targets. "
            "Only use actions listed here. Invalid actions return a -0.5 penalty."
        ),
    )

    # --- Diagnostic output from inspect_* actions ---
    logs: Optional[str] = Field(
        default=None,
        description="Log output from the most recent inspect_logs action, if any.",
    )
    metric_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Per-tick metric history from the most recent inspect_metrics action.",
    )
    traces: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Distributed trace from the most recent inspect_traces action.",
    )
    oversight_policy: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="High-impact rules when oversight is enabled (read-only for the agent).",
    )
    pending_approvals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="In-flight or recent approval requests when oversight is enabled.",
    )


class SevZeroState(State):
    """
    Episode metadata returned by the state property.
    `episode_id` and `step_count` are inherited from State base.
    """

    task_id: str = Field(default="easy", description="Which task: 'easy' | 'medium' | 'hard'")
    seed: Optional[int] = Field(
        default=None, description="Seed used for this episode (for reproducibility)"
    )
    global_slo_score: float = Field(
        default=0.0, description="Current fraction of services meeting SLO targets"
    )
    terminated: bool = Field(
        default=False, description="Whether the episode has ended for any reason"
    )
    termination_reason: Optional[str] = Field(
        default=None,
        description="Why the episode ended: 'resolved' | 'failed' | 'timeout' | None",
    )
