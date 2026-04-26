"""
server/failures.py — 8 failure types with injection logic and metric evolution patterns.

Each failure type has:
  - A distinctive metric temporal shape (how metrics evolve per tick)
  - Config error subtypes (startup vs runtime)
  - Weighted distribution matching real-world incident data

Sources: Google SRE postmortems, Netflix Hystrix, AWS incident reports.
See Docs/DataResearch.md for full citation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------


class FailureType(str, Enum):
    CRASH = "crash"
    BAD_DEPLOY = "bad_deploy"
    CONFIG_STARTUP = "config_startup"   # Service can't boot
    CONFIG_RUNTIME = "config_runtime"   # Service runs but specific paths fail
    CASCADING_LATENCY = "cascading_latency"
    RESOURCE_LEAK = "resource_leak"
    DB_DEGRADATION = "db_degradation"
    CACHE_FAILURE = "cache_failure"
    NETWORK_ERROR = "network_error"


# Weighted distribution matching Google empirical incident data
# config=32%, deploy=25%, cascade=15%, crash=10%, leak=8%, DB=5%, cache=3%, network=2%
_FAILURE_WEIGHTS: Dict[FailureType, float] = {
    FailureType.CONFIG_STARTUP:    0.16,
    FailureType.CONFIG_RUNTIME:    0.16,
    FailureType.BAD_DEPLOY:        0.25,
    FailureType.CASCADING_LATENCY: 0.15,
    FailureType.CRASH:             0.10,
    FailureType.RESOURCE_LEAK:     0.08,
    FailureType.DB_DEGRADATION:    0.05,
    FailureType.CACHE_FAILURE:     0.03,
    FailureType.NETWORK_ERROR:     0.02,
}

# For multi-root incidents: avoid unlikely combinations
_INCOMPATIBLE_PAIRS = {
    (FailureType.NETWORK_ERROR, FailureType.NETWORK_ERROR),  # Two network errors is unrealistic
    (FailureType.CACHE_FAILURE, FailureType.CACHE_FAILURE),  # Two cache failures is unrealistic
}


@dataclass
class FailureSpec:
    """Describes a single injected failure and its evolution parameters."""

    service_id: str
    failure_type: FailureType

    # Error rates at various stages (used by metric evolution)
    base_error_rate: float = 0.0        # Healthy baseline
    peak_error_rate: float = 0.0        # At full failure
    onset_ticks: int = 1                # Ticks to reach peak (1=instant, 5=gradual)

    # Latency impact at peak
    latency_multiplier: float = 1.0     # How much p99 multiplies at peak

    # Resource impact at peak
    cpu_impact: float = 0.0             # CPU increase (0–1)
    memory_impact: float = 0.0          # Memory increase per tick (for leaks)
    pool_saturation: float = 0.0        # Connection pool impact

    # Config error subtype metadata
    broken_config_key: Optional[str] = None    # Which config key is wrong
    broken_config_value: Optional[str] = None  # What the wrong value is

    # Deployment metadata (for bad_deploy)
    bad_version: Optional[str] = None
    good_version: Optional[str] = None

    # Network error metadata
    affected_region: Optional[str] = None


# ---------------------------------------------------------------------------
# Failure selection
# ---------------------------------------------------------------------------


def select_failure_type(
    rng: random.Random,
    exclude: Optional[List[FailureType]] = None,
    weight_override: Optional[Dict[FailureType, float]] = None,
) -> FailureType:
    """Sample a failure type from the empirically-weighted distribution."""
    if weight_override:
        base: Dict[FailureType, float] = {
            f: weight_override.get(f, _FAILURE_WEIGHTS.get(f, 0.0))
            for f in _FAILURE_WEIGHTS
        }
    else:
        base = dict(_FAILURE_WEIGHTS)
    population = list(base.keys())
    weights = [max(1e-9, base[f]) for f in population]

    # Remove excluded types
    if exclude:
        filtered = [(f, w) for f, w in zip(population, weights) if f not in exclude]
        if filtered:
            population, weights = zip(*filtered)
            population, weights = list(population), list(weights)

    return rng.choices(population, weights=weights, k=1)[0]


def select_multi_root_failures(
    rng: random.Random, count: int = 2,
    weight_override: Optional[Dict[FailureType, float]] = None,
) -> List[FailureType]:
    """Select multiple failure types with incompatibility constraints."""
    selected: List[FailureType] = []
    for _ in range(count):
        exclude = selected[:]
        # Also exclude incompatible pairs
        for s in selected:
            for a, b in _INCOMPATIBLE_PAIRS:
                if s == a:
                    exclude.append(b)
                elif s == b:
                    exclude.append(a)
        ft = select_failure_type(
            rng, exclude=exclude, weight_override=weight_override,
        )
        selected.append(ft)
    return selected


# ---------------------------------------------------------------------------
# Failure specification factories
# ---------------------------------------------------------------------------


def make_crash_spec(service_id: str, rng: random.Random) -> FailureSpec:
    """Service Crash: sudden 5xx spike then drop (service is dead)."""
    return FailureSpec(
        service_id=service_id,
        failure_type=FailureType.CRASH,
        base_error_rate=0.0,
        peak_error_rate=rng.uniform(0.85, 1.0),
        onset_ticks=1,           # Instant
        latency_multiplier=0.1,  # Latency drops (fast fails, no waiting)
        cpu_impact=0.0,          # CPU near zero (process dead)
        memory_impact=0.0,
    )


def make_bad_deploy_spec(service_id: str, rng: random.Random) -> FailureSpec:
    """Bad Deployment: step-function error increase after version change."""
    return FailureSpec(
        service_id=service_id,
        failure_type=FailureType.BAD_DEPLOY,
        base_error_rate=0.0,
        peak_error_rate=rng.uniform(0.30, 0.70),
        onset_ticks=1,                # Step function — appears at deploy tick
        latency_multiplier=rng.uniform(1.5, 3.0),
        cpu_impact=rng.uniform(0.1, 0.3),
        memory_impact=rng.uniform(0.05, 0.15),
        bad_version="v" + str(rng.randint(2, 9)) + "." + str(rng.randint(0, 9)) + "." + str(rng.randint(1, 9)),
        good_version="v1.0.0",
    )


def make_config_startup_spec(service_id: str, rng: random.Random) -> FailureSpec:
    """Config Error (Startup): service can't boot — zero traffic, health checks fail."""
    config_keys = ["db_password", "db_host", "api_endpoint", "env_var", "config_file"]
    return FailureSpec(
        service_id=service_id,
        failure_type=FailureType.CONFIG_STARTUP,
        base_error_rate=0.0,
        peak_error_rate=1.0,          # 100% — service is completely down
        onset_ticks=1,
        latency_multiplier=0.0,       # No latency, no traffic
        cpu_impact=-0.9,              # CPU near zero (process exited immediately)
        memory_impact=-0.9,
        broken_config_key=rng.choice(config_keys),
        broken_config_value="WRONG_VALUE",
    )


def make_config_runtime_spec(service_id: str, rng: random.Random) -> FailureSpec:
    """Config Error (Runtime): service runs but specific code paths fail."""
    config_keys = ["api_endpoint", "feature_flag", "timeout_ms", "retry_max"]
    return FailureSpec(
        service_id=service_id,
        failure_type=FailureType.CONFIG_RUNTIME,
        base_error_rate=0.0,
        peak_error_rate=rng.uniform(0.20, 0.60),
        onset_ticks=1,
        latency_multiplier=rng.uniform(1.2, 2.0),
        cpu_impact=0.0,              # Normal resource usage
        memory_impact=0.0,
        broken_config_key=rng.choice(config_keys),
        broken_config_value="MISCONFIGURED",
    )


def make_cascading_latency_spec(service_id: str, rng: random.Random) -> FailureSpec:
    """
    Cascading Latency: gradual latency ramp → thread pool exhaustion.
    KEY signature: p99 ramps BEFORE errors appear. CPU rises from blocked threads.
    """
    return FailureSpec(
        service_id=service_id,
        failure_type=FailureType.CASCADING_LATENCY,
        base_error_rate=0.0,
        peak_error_rate=rng.uniform(0.40, 0.85),
        onset_ticks=rng.randint(3, 6),  # Gradual ramp
        latency_multiplier=rng.uniform(8.0, 20.0),
        cpu_impact=rng.uniform(0.30, 0.60),   # Rising CPU from blocked threads
        memory_impact=rng.uniform(0.10, 0.25),
    )


def make_resource_leak_spec(service_id: str, rng: random.Random) -> FailureSpec:
    """Resource Leak: steady memory/CPU climb; sawtooth pattern on restarts."""
    return FailureSpec(
        service_id=service_id,
        failure_type=FailureType.RESOURCE_LEAK,
        base_error_rate=0.0,
        peak_error_rate=rng.uniform(0.20, 0.50),
        onset_ticks=rng.randint(5, 10),  # Slow burn
        latency_multiplier=rng.uniform(2.0, 5.0),
        cpu_impact=0.05,              # Grows per tick (applied in evolution)
        memory_impact=0.06,           # LINEAR RAMP — key signature
    )


def make_db_degradation_spec(service_id: str, rng: random.Random) -> FailureSpec:
    """DB Degradation: rising DB latency, pool saturation, app CPU paradoxically LOW."""
    return FailureSpec(
        service_id=service_id,
        failure_type=FailureType.DB_DEGRADATION,
        base_error_rate=0.0,
        peak_error_rate=rng.uniform(0.30, 0.70),
        onset_ticks=rng.randint(2, 4),
        latency_multiplier=rng.uniform(5.0, 15.0),
        cpu_impact=-0.2,              # PARADOXICALLY LOW (waiting on I/O)
        memory_impact=0.05,
        pool_saturation=0.90,         # Connection pool hits 90%+
    )


def make_cache_failure_spec(service_id: str, rng: random.Random) -> FailureSpec:
    """Cache Failure: hit-rate cliff → backend QPS 10-50x spike → DB overload."""
    return FailureSpec(
        service_id=service_id,
        failure_type=FailureType.CACHE_FAILURE,
        base_error_rate=0.0,
        peak_error_rate=rng.uniform(0.20, 0.50),
        onset_ticks=1,               # CLIFF — simultaneous, not gradual
        latency_multiplier=rng.uniform(3.0, 8.0),
        cpu_impact=0.20,
        memory_impact=0.0,
    )


def make_network_error_spec(service_id: str, rng: random.Random, region: str = "us-east-1") -> FailureSpec:
    """Network/Routing Error: connection failures affecting all services to this region."""
    return FailureSpec(
        service_id=service_id,
        failure_type=FailureType.NETWORK_ERROR,
        base_error_rate=0.0,
        peak_error_rate=rng.uniform(0.80, 1.0),
        onset_ticks=1,               # Simultaneous, not hop-by-hop
        latency_multiplier=0.2,      # Timeout values — fixed high, then drop
        cpu_impact=-0.3,             # Low CPU (nothing getting through)
        memory_impact=0.0,
        affected_region=region,
    )


_SPEC_FACTORIES = {
    FailureType.CRASH:              make_crash_spec,
    FailureType.BAD_DEPLOY:         make_bad_deploy_spec,
    FailureType.CONFIG_STARTUP:     make_config_startup_spec,
    FailureType.CONFIG_RUNTIME:     make_config_runtime_spec,
    FailureType.CASCADING_LATENCY:  make_cascading_latency_spec,
    FailureType.RESOURCE_LEAK:      make_resource_leak_spec,
    FailureType.DB_DEGRADATION:     make_db_degradation_spec,
    FailureType.CACHE_FAILURE:      make_cache_failure_spec,
    FailureType.NETWORK_ERROR:      make_network_error_spec,
}


def make_failure_spec(
    service_id: str,
    failure_type: FailureType,
    rng: random.Random,
    **kwargs,
) -> FailureSpec:
    """Create a FailureSpec for the given service and failure type."""
    factory = _SPEC_FACTORIES[failure_type]
    return factory(service_id, rng, **kwargs)


# ---------------------------------------------------------------------------
# Metric evolution: per-type temporal shapes
# ---------------------------------------------------------------------------


def compute_failure_magnitude(spec: FailureSpec, ticks_since_failure: int) -> float:
    """
    Return a 0.0–1.0 magnitude factor for how fully the failure has manifested.
    - Instant failures (onset_ticks=1): full magnitude from tick 1
    - Gradual failures: linear ramp over onset_ticks
    - Resource leaks: continues growing after onset (handled separately)
    """
    if spec.onset_ticks <= 1:
        return 1.0
    return min(1.0, ticks_since_failure / spec.onset_ticks)


def apply_failure_to_metrics(
    spec: FailureSpec,
    ticks_since_failure: int,
    base_error_rate: float,
    base_p99_ms: float,
    base_cpu: float,
    base_memory: float,
    base_pool: float,
    rng: random.Random,
) -> Tuple[float, float, float, float, float]:
    """
    Apply failure evolution to metrics.
    Returns: (error_rate, p99_ms, cpu_pct, memory_pct, pool_pct)

    Each failure type produces a DISTINCTIVE temporal shape:
    - crash: instant spike → drop (service dead)
    - bad_deploy: step function up at deploy tick
    - config_startup: 100% error, zero traffic
    - config_runtime: partial errors on affected paths
    - cascading_latency: p99 ramps BEFORE errors (early warning)
    - resource_leak: memory linear ramp, sawtooth CPU
    - db_degradation: pool saturation, CPU paradoxically LOW
    - cache_failure: cliff drop simultaneous
    - network_error: cliff, then fixed-high timeout values
    """
    mag = compute_failure_magnitude(spec, ticks_since_failure)

    # Add natural stochastic variance (±5%) — Bernoulli trial model
    noise = rng.uniform(-0.03, 0.03)

    ft = spec.failure_type

    if ft == FailureType.CRASH:
        error_rate = spec.peak_error_rate * mag + noise
        p99_ms = base_p99_ms * 0.1 * mag + base_p99_ms * (1 - mag)  # Drops fast
        cpu_pct = max(0.0, base_cpu * (1 - 0.9 * mag))
        memory_pct = base_memory
        pool_pct = base_pool

    elif ft == FailureType.BAD_DEPLOY:
        error_rate = spec.peak_error_rate * mag + noise
        p99_ms = base_p99_ms * (1 + (spec.latency_multiplier - 1) * mag)
        cpu_pct = min(100.0, base_cpu * (1 + spec.cpu_impact * mag))
        memory_pct = min(100.0, base_memory * (1 + spec.memory_impact * mag))
        pool_pct = base_pool

    elif ft == FailureType.CONFIG_STARTUP:
        error_rate = 1.0                 # Always 100% — service won't start
        p99_ms = 0.0                     # No traffic = no latency
        cpu_pct = max(0.0, base_cpu * 0.02)   # Near zero
        memory_pct = max(0.0, base_memory * 0.02)
        pool_pct = 0.0

    elif ft == FailureType.CONFIG_RUNTIME:
        error_rate = spec.peak_error_rate * mag + noise
        p99_ms = base_p99_ms * (1 + (spec.latency_multiplier - 1) * mag)
        cpu_pct = base_cpu                # Normal — only specific paths fail
        memory_pct = base_memory
        pool_pct = base_pool

    elif ft == FailureType.CASCADING_LATENCY:
        # p99 ramps BEFORE errors — the key diagnostic signature
        latency_onset_fraction = min(1.0, ticks_since_failure / max(1, spec.onset_ticks - 1))
        error_onset_fraction = min(1.0, max(0.0, (ticks_since_failure - 1) / spec.onset_ticks))

        error_rate = spec.peak_error_rate * error_onset_fraction + noise
        p99_ms = base_p99_ms * (1 + (spec.latency_multiplier - 1) * latency_onset_fraction)
        cpu_pct = min(100.0, base_cpu * (1 + spec.cpu_impact * latency_onset_fraction))
        memory_pct = min(100.0, base_memory * (1 + spec.memory_impact * latency_onset_fraction))
        pool_pct = base_pool

    elif ft == FailureType.RESOURCE_LEAK:
        # Memory: LINEAR RAMP to limit (key signature)
        # CPU: Growing GC thrash
        leak_fraction = min(1.0, ticks_since_failure * 0.08)  # ~12 ticks to peak
        error_rate = spec.peak_error_rate * min(1.0, leak_fraction * 1.5) + noise
        p99_ms = base_p99_ms * (1 + (spec.latency_multiplier - 1) * leak_fraction)
        cpu_pct = min(100.0, base_cpu * (1 + leak_fraction * 0.8))     # GC pressure
        memory_pct = min(100.0, base_memory + leak_fraction * (100 - base_memory))
        pool_pct = base_pool

    elif ft == FailureType.DB_DEGRADATION:
        error_rate = spec.peak_error_rate * mag + noise
        p99_ms = base_p99_ms * (1 + (spec.latency_multiplier - 1) * mag)
        # CPU paradoxically LOW — waiting on I/O, not computing
        cpu_pct = max(5.0, base_cpu * (1 + spec.cpu_impact * mag))
        memory_pct = min(100.0, base_memory * (1 + spec.memory_impact * mag))
        pool_pct = min(100.0, base_pool + spec.pool_saturation * mag * 100)

    elif ft == FailureType.CACHE_FAILURE:
        # CLIFF: simultaneous, not gradual (onset_ticks=1)
        error_rate = spec.peak_error_rate * mag + noise
        p99_ms = base_p99_ms * (1 + (spec.latency_multiplier - 1) * mag)
        cpu_pct = min(100.0, base_cpu * (1 + spec.cpu_impact * mag))
        memory_pct = base_memory
        pool_pct = base_pool

    elif ft == FailureType.NETWORK_ERROR:
        # Cliff: all fails simultaneously; latency = timeout values then 0
        error_rate = spec.peak_error_rate * mag + noise
        # Latency spikes to timeout then drops (nothing gets through)
        p99_ms = base_p99_ms * 10.0 * max(0.1, 1 - ticks_since_failure * 0.3)
        cpu_pct = max(2.0, base_cpu * (1 + spec.cpu_impact * mag))
        memory_pct = base_memory
        pool_pct = base_pool

    else:
        error_rate = base_error_rate
        p99_ms = base_p99_ms
        cpu_pct = base_cpu
        memory_pct = base_memory
        pool_pct = base_pool

    return (
        max(0.0, min(1.0, error_rate)),
        max(1.0, p99_ms),
        max(0.0, min(100.0, cpu_pct)),
        max(0.0, min(100.0, memory_pct)),
        max(0.0, min(100.0, pool_pct)),
    )
