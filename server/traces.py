"""
server/traces.py — Distributed trace generation for inspect_traces action.

Generates realistic Jaeger/Zipkin-style trace trees showing request flow
through the service dependency graph. Healthy services show normal latencies;
failing services show errors, timeouts, and cascading delays.

Each trace is a tree of spans rooted at the inspected service.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from server.graph import ServiceGraph


def _make_span_id(rng: random.Random) -> str:
    return f"{rng.randint(0, 0xFFFFFFFF):08x}"


def _make_trace_id(rng: random.Random) -> str:
    return f"{rng.randint(0, 0xFFFFFFFFFFFFFFFF):016x}"


def generate_trace(
    service_id: str,
    graph: ServiceGraph,
    service_errors: Dict[str, float],
    service_latencies: Dict[str, float],
    rng: random.Random,
    max_depth: int = 4,
) -> Dict[str, Any]:
    """
    Generate a distributed trace tree rooted at service_id.

    Returns a dict with trace_id, root_span, and flat spans list.
    service_errors: service_id → error_rate (0.0–1.0)
    service_latencies: service_id → p99_ms
    """
    trace_id = _make_trace_id(rng)
    spans: List[Dict[str, Any]] = []

    def _build_span(
        svc_id: str,
        parent_span_id: Optional[str],
        depth: int,
        start_offset_ms: float,
    ) -> Dict[str, Any]:
        span_id = _make_span_id(rng)
        error_rate = service_errors.get(svc_id, 0.0)
        base_latency = service_latencies.get(svc_id, rng.uniform(5, 50))
        has_error = rng.random() < error_rate

        # Span duration: base latency + noise
        if has_error and error_rate > 0.8:
            # Fast fail or timeout
            duration_ms = rng.choice([
                rng.uniform(0.5, 5),       # Fast fail
                rng.uniform(3000, 10000),   # Timeout
            ])
        elif has_error:
            duration_ms = base_latency * rng.uniform(1.5, 5.0)
        else:
            duration_ms = base_latency * rng.uniform(0.3, 1.2)

        duration_ms = max(0.1, duration_ms)

        span = {
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "service": svc_id,
            "operation": _operation_name(svc_id, rng),
            "start_ms": round(start_offset_ms, 1),
            "duration_ms": round(duration_ms, 1),
            "status": "ERROR" if has_error else "OK",
            "tags": {},
        }

        if has_error:
            span["tags"]["error"] = True
            span["tags"]["error.message"] = _error_message(svc_id, error_rate, rng)

        node = graph.node_map.get(svc_id)
        if node:
            span["tags"]["service.layer"] = node.layer
            span["tags"]["service.region"] = node.region

        spans.append(span)

        # Recurse into downstream dependencies
        if depth < max_depth:
            deps = graph.adjacency.get(svc_id, [])
            child_offset = start_offset_ms + rng.uniform(0.1, 2.0)
            for dep_id in deps:
                # Check edge activation (probabilistic)
                edge = next(
                    (e for e in graph.edges if e.source == svc_id and e.target == dep_id),
                    None,
                )
                if edge and rng.random() > edge.activation_probability:
                    continue

                child_span = _build_span(dep_id, span_id, depth + 1, child_offset)
                child_offset += child_span["duration_ms"] + rng.uniform(0.1, 1.0)

        return span

    root_span = _build_span(service_id, None, 0, 0.0)

    # Compute total trace duration
    if spans:
        total_duration = max(s["start_ms"] + s["duration_ms"] for s in spans)
    else:
        total_duration = 0.0

    return {
        "trace_id": trace_id,
        "root_service": service_id,
        "span_count": len(spans),
        "total_duration_ms": round(total_duration, 1),
        "spans": spans,
    }


def _operation_name(service_id: str, rng: random.Random) -> str:
    """Generate a realistic operation name based on service type."""
    if "gateway" in service_id or "bff" in service_id:
        return rng.choice(["HTTP GET /api/v1/resource", "HTTP POST /api/v1/action", "HTTP GET /health"])
    if "auth" in service_id or "identity" in service_id or "session" in service_id:
        return rng.choice(["validateToken", "authenticate", "refreshSession"])
    if "postgres" in service_id:
        return rng.choice(["SELECT", "INSERT", "UPDATE", "pg_pool.checkout"])
    if "redis" in service_id:
        return rng.choice(["GET", "SET", "MGET", "EXPIRE"])
    if "kafka" in service_id:
        return rng.choice(["produce", "consume", "commitOffset"])
    if "elasticsearch" in service_id:
        return rng.choice(["search", "index", "bulk"])
    return rng.choice(["processRequest", "handleMessage", "execute"])


def _error_message(service_id: str, error_rate: float, rng: random.Random) -> str:
    """Generate a trace-level error message."""
    if error_rate > 0.8:
        return rng.choice([
            f"{service_id}: Connection refused",
            f"{service_id}: Service unavailable (HTTP 503)",
            f"{service_id}: Timeout after 5000ms",
        ])
    return rng.choice([
        f"{service_id}: Internal server error (HTTP 500)",
        f"{service_id}: Upstream dependency timeout",
        f"{service_id}: Rate limited (HTTP 429)",
        f"{service_id}: Bad gateway (HTTP 502)",
    ])
