"""
server/logs.py — Framework-specific log message templates per failure type.

Each failure type has 5-10 realistic log templates drawn from real frameworks:
Spring Boot, Node.js, FastAPI, Kubernetes, HikariCP, Redis, gRPC.

Templates use placeholders {service}, {dependency}, {value} etc. that are
filled at runtime with actual service/metric values.

Sources: Docs/DataResearch.md Answer 4 + Answer 11.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from server.failures import FailureType


# ---------------------------------------------------------------------------
# Log templates per failure type
# ---------------------------------------------------------------------------

_TEMPLATES: Dict[FailureType, List[str]] = {
    FailureType.CRASH: [
        "ERROR {service} OOMKilled: container exceeded memory limit ({memory_limit}Mi). Exit code 137. Pod restarting (backoff: {backoff}s)",
        "FATAL {service} Process exited with signal 9 (SIGKILL). Out of memory. Restart count: {restart_count}",
        "ERROR {service} CrashLoopBackOff: back-off restarting failed container. Last exit: OOMKilled",
        "CRIT {service} JVM heap exhausted: java.lang.OutOfMemoryError: Java heap space. Heap: {heap_used}Mi/{heap_max}Mi",
        "ERROR {service} Panic: runtime error: out of memory. goroutine stack overflow at allocateHeap()",
        "FATAL {service} Node process crashed: FATAL ERROR: CALL_AND_RETRY_LAST Allocation failed - JavaScript heap out of memory",
    ],

    FailureType.BAD_DEPLOY: [
        "ERROR {service} {version} NullPointerException: Cannot invoke \"{method}\" on null reference at {class}.process({class}.java:{line})",
        "ERROR {service} {version} TypeError: Cannot read properties of undefined (reading '{property}'). Stack: at {handler} ({file}:{line})",
        "ERROR {service} {version} Traceback (most recent call last):\\n  File \"{file}\", line {line}\\n    {code_line}\\nAttributeError: '{class}' object has no attribute '{attribute}'",
        "ERROR {service} {version} panic: interface conversion: interface {} is nil, not *{type}. goroutine {goroutine_id} [running]",
        "ERROR {service} {version} Unhandled rejection: ValidationError: \"{field}\" is required. Schema version mismatch between {version} and data format.",
        "WARN  {service} {version} Health check failing: /health returned 500. Error rate climbing: {error_rate}%",
    ],

    FailureType.CONFIG_STARTUP: [
        "FATAL {service} password authentication failed for user \"{db_user}\" on {dependency}:{port}. Connection refused.",
        "ERROR {service} Could not resolve placeholder '{config_key}' in value \"${{{config_key}}}\"",
        "FATAL {service} Configuration error: required key [{config_key}] not found in application.yml",
        "ERROR {service} Failed to bind to port {port}: EADDRINUSE. Another process is using this port.",
        "FATAL {service} SSL/TLS certificate error: certificate has expired. CN={dependency}. Valid until: {expiry}",
        "ERROR {service} Cannot connect to {dependency}: Connection refused. Retried {retry_count} times, giving up.",
    ],

    FailureType.CONFIG_RUNTIME: [
        "ERROR {service} Request to https://{config_value}/charge failed: ECONNREFUSED. Feature \"{feature_flag}\" enabled but endpoint misconfigured.",
        "WARN  {service} Fallback triggered for {dependency}: timeout after {timeout_ms}ms. Config key '{config_key}' may be incorrect.",
        "ERROR {service} Invalid JSON response from {dependency}: Unexpected token '<' at position 0. Endpoint returning HTML instead of API response.",
        "ERROR {service} Feature flag '{feature_flag}' enabled new code path but dependency '{dependency}' not configured. Returning 500 for {error_rate}% of /api/v2 requests.",
        "WARN  {service} Rate limit config mismatch: max_rps={config_value} but actual traffic is {throughput}rps. Dropping {error_rate}% of requests.",
    ],

    FailureType.CASCADING_LATENCY: [
        "WARN  {service} Thread pool self-saturation: {active}/{pool_size} worker threads active. Queue depth: {queue_depth}. Avg wait: {wait_ms}ms. "
        "This service is the bottleneck — scale or rebalance traffic away from this service.",
        "WARN  {service} Worker thread exhaustion: arrival rate {throughput}rps exceeds processing capacity. "
        "Active threads: {active}/{pool_size}. Queued: {queue_depth}. Fix: scale_service or rebalance_traffic.",
        "ERROR {service} Request queue overflow: {queue_depth} requests waiting for worker threads ({active}/{pool_size} busy). "
        "p99={p99_ms}ms. Root cause is this service's own capacity — restart to clear threads or scale to add capacity.",
        "WARN  {service} Internal latency spiral: p99={p99_ms}ms (baseline: {baseline_ms}ms). Thread pool utilisation critical. "
        "Retry amplification causing {throughput}rps effective load. This service needs to be restarted or scaled.",
        "CRIT  {service} Capacity overload: {active}/{pool_size} threads saturated, {queue_depth} requests pending. "
        "All downstream timeouts are a symptom of THIS service being overwhelmed. "
        "Run: restart_service or scale_service on {service}.",
    ],

    FailureType.RESOURCE_LEAK: [
        "WARN  {service} Memory usage {memory_pct}% ({memory_used}Mi/{memory_limit}Mi). GC overhead {gc_pct}%. Last full GC: {gc_pause}s pause. Allocation failure imminent.",
        "WARN  {service} File descriptor leak detected: open_fds={open_fds} (limit: {fd_limit}). Growing at {fd_rate}/min.",
        "WARN  {service} Goroutine leak: count={goroutine_count} (baseline: {baseline}). Growing linearly. Stack trace: {leak_source}",
        "ERROR {service} GC overhead limit exceeded: spending {gc_pct}% of time in GC. Heap: {memory_used}Mi/{memory_limit}Mi.",
        "WARN  {service} Connection leak to {dependency}: {active} connections checked out but not returned. Pool: {active}/{pool_size}.",
    ],

    FailureType.DB_DEGRADATION: [
        "ERROR {service} HikariPool-1 connection not available, request timed out after {timeout_ms}ms. Active: {active}/{pool_size}, Waiting: {waiting}.",
        "WARN  {service} Slow query detected: SELECT * FROM {table} WHERE ... took {query_ms}ms (threshold: {threshold_ms}ms). Lock contention on {table}.",
        "ERROR {service} Connection pool exhausted for {dependency}. Active: {active}/{pool_size}. Oldest connection age: {age_ms}ms.",
        "WARN  {service} Database replication lag: {lag_ms}ms on {dependency}. Read-after-write consistency violated.",
        "ERROR {service} Deadlock detected on {dependency}: Transaction {tx_id} waiting for lock held by {blocking_tx}. Auto-rolling back.",
        "WARN  {service} {dependency} CPU={db_cpu}% but app CPU={app_cpu}% (paradoxically low). Threads blocked on I/O wait.",
    ],

    FailureType.CACHE_FAILURE: [
        "WARN  {service} CLUSTERDOWN: {dependency} cluster is down. Hit rate dropped from {baseline_hit_rate}% to 0%. Backend QPS spiked {spike_factor}x.",
        "ERROR {service} Redis connection lost: {dependency} ECONNRESET. Failover in progress. Cache miss rate: 100%.",
        "WARN  {service} Cache stampede detected: {concurrent_misses} concurrent cache misses for key pattern '{key_pattern}'. Backend overloaded.",
        "ERROR {service} {dependency} READONLY: Redis replica cannot accept writes. Cluster rebalancing.",
        "WARN  {service} Cache eviction storm: {evicted} keys evicted in last {interval}s. Memory pressure on {dependency}.",
    ],

    FailureType.NETWORK_ERROR: [
        "ERROR {service} DNS resolution failed for {dependency}.{region}.internal: NXDOMAIN. 0/{endpoint_count} endpoints reachable.",
        "ERROR {service} TCP connection to {dependency}:{port} failed: ETIMEDOUT after {timeout_ms}ms. Network partition suspected.",
        "ERROR {service} TLS handshake failed with {dependency}: certificate verify failed (depth 0). CN mismatch or expired cert.",
        "CRIT  {service} All endpoints for {dependency} unreachable in region {region}. Last successful connection: {last_success} ago.",
        "ERROR {service} gRPC transport error: UNAVAILABLE: {dependency} DNS resolution failed for \"{dependency}.svc.cluster.local\"",
    ],
}


# ---------------------------------------------------------------------------
# Placeholder value generators
# ---------------------------------------------------------------------------


def _random_class_name(rng: random.Random) -> str:
    prefixes = ["Payment", "Order", "Auth", "Inventory", "Cart", "Billing", "Shipping"]
    suffixes = ["Service", "Handler", "Controller", "Processor", "Manager"]
    return rng.choice(prefixes) + rng.choice(suffixes)


def _random_method(rng: random.Random) -> str:
    return rng.choice(["process", "handle", "execute", "validate", "transform", "serialize", "getId", "getStatus"])


def _random_property(rng: random.Random) -> str:
    return rng.choice(["id", "status", "amount", "userId", "orderId", "timestamp", "payload", "response"])


def _fill_placeholders(
    template: str,
    service_id: str,
    rng: random.Random,
    dependency: str = "unknown",
    error_rate: float = 0.0,
    memory_pct: float = 50.0,
    p99_ms: float = 100.0,
    pool_pct: float = 10.0,
    version: str = "v1.0.0",
    config_key: str = "db_host",
    config_value: str = "wrong-endpoint.internal",
    region: str = "us-east-1",
    throughput: float = 100.0,
) -> str:
    """Fill placeholders in a log template with realistic values."""
    replacements = {
        "service": service_id,
        "dependency": dependency,
        "version": version,
        "error_rate": f"{error_rate * 100:.0f}",
        "memory_pct": f"{memory_pct:.0f}",
        "memory_used": f"{int(memory_pct * 20.48):.0f}",
        "memory_limit": "2048",
        "heap_used": f"{int(memory_pct * 10.24):.0f}",
        "heap_max": "1024",
        "p99_ms": f"{p99_ms:.0f}",
        "baseline_ms": f"{rng.randint(20, 80)}",
        "timeout_ms": f"{rng.choice([3000, 5000, 10000, 30000])}",
        "cooldown": f"{rng.randint(15, 60)}",
        "queued": f"{rng.randint(50, 500)}",
        "queue_depth": f"{rng.randint(100, 1000)}",
        "wait_ms": f"{rng.randint(500, 5000)}",
        "active": f"{rng.randint(15, 25)}",
        "pool_size": "20",
        "pending": f"{rng.randint(50, 200)}",
        "checkout_ms": f"{rng.randint(1000, 10000)}",
        "threshold_ms": "1000",
        "retry_count": f"{rng.randint(1, 5)}",
        "retry_max": "3",
        "backoff": f"{rng.choice([10, 15, 30, 60])}",
        "restart_count": f"{rng.randint(3, 15)}",
        "port": f"{rng.choice([5432, 6379, 8080, 9090, 3000])}",
        "db_user": rng.choice(["app_user", "service_account", "auth_user", "readonly"]),
        "config_key": config_key,
        "config_value": config_value,
        "feature_flag": rng.choice(["new_checkout_flow", "v2_api", "experimental_search", "dynamic_pricing"]),
        "region": region,
        "endpoint_count": f"{rng.randint(2, 5)}",
        "class": _random_class_name(rng),
        "method": _random_method(rng),
        "property": _random_property(rng),
        "attribute": _random_property(rng),
        "type": _random_class_name(rng),
        "handler": rng.choice(["processRequest", "handleEvent", "onMessage"]),
        "file": rng.choice(["app.py", "handler.js", "service.go", "controller.java"]),
        "line": f"{rng.randint(42, 350)}",
        "code_line": rng.choice(["result = response.data['items']", "return self.client.process(payload)"]),
        "field": rng.choice(["amount", "currency", "userId", "orderId"]),
        "goroutine_id": f"{rng.randint(100, 999)}",
        "table": rng.choice(["orders", "payments", "users", "inventory", "sessions"]),
        "query_ms": f"{rng.randint(5000, 30000)}",
        "tx_id": f"tx-{rng.randint(1000, 9999)}",
        "blocking_tx": f"tx-{rng.randint(1000, 9999)}",
        "lag_ms": f"{rng.randint(1000, 10000)}",
        "age_ms": f"{rng.randint(30000, 120000)}",
        "db_cpu": f"{rng.randint(5, 25)}",
        "app_cpu": f"{rng.randint(2, 15)}",
        "waiting": f"{rng.randint(50, 300)}",
        "baseline_hit_rate": f"{rng.uniform(95.0, 99.5):.1f}",
        "spike_factor": f"{rng.randint(10, 50)}",
        "concurrent_misses": f"{rng.randint(100, 1000)}",
        "key_pattern": rng.choice(["user:*", "product:*:price", "session:*", "inventory:*"]),
        "evicted": f"{rng.randint(10000, 100000)}",
        "interval": f"{rng.randint(10, 60)}",
        "gc_pct": f"{rng.randint(30, 70)}",
        "gc_pause": f"{rng.uniform(0.5, 3.0):.1f}",
        "open_fds": f"{rng.randint(800, 1024)}",
        "fd_limit": "1024",
        "fd_rate": f"{rng.randint(5, 20)}",
        "goroutine_count": f"{rng.randint(5000, 50000)}",
        "baseline": f"{rng.randint(50, 200)}",
        "leak_source": rng.choice(["http.ListenAndServe", "grpc.NewServer", "sql.Open"]),
        "hop_count": f"{rng.randint(2, 5)}",
        "remaining_ms": f"{rng.randint(-500, 10)}",
        "last_success": rng.choice(["45s", "2m30s", "5m12s"]),
        "throughput": f"{throughput:.0f}",
    }

    result = template
    for key, value in replacements.items():
        result = result.replace("{" + key + "}", str(value))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_log_message(
    failure_type: FailureType,
    service_id: str,
    rng: random.Random,
    dependency: str = "unknown",
    error_rate: float = 0.0,
    memory_pct: float = 50.0,
    p99_ms: float = 100.0,
    pool_pct: float = 10.0,
    version: str = "v1.0.0",
    config_key: str = "db_host",
    config_value: str = "wrong-endpoint.internal",
    region: str = "us-east-1",
    throughput: float = 100.0,
) -> str:
    """Generate a realistic log message for the given failure type and service."""
    templates = _TEMPLATES.get(failure_type, [])
    if not templates:
        return f"ERROR {service_id} Unknown failure condition detected."

    template = rng.choice(templates)
    return _fill_placeholders(
        template, service_id, rng,
        dependency=dependency,
        error_rate=error_rate,
        memory_pct=memory_pct,
        p99_ms=p99_ms,
        pool_pct=pool_pct,
        version=version,
        config_key=config_key,
        config_value=config_value,
        region=region,
        throughput=throughput,
    )


def generate_healthy_log(service_id: str, rng: random.Random) -> str:
    """Generate a log message for a healthy service being inspected."""
    templates = [
        f"INFO  {service_id} Health check passed. Status: UP. Response time: {rng.randint(2, 15)}ms.",
        f"INFO  {service_id} All endpoints healthy. Error rate: 0.0%. p99: {rng.randint(10, 50)}ms.",
        f"DEBUG {service_id} Metrics nominal. CPU: {rng.randint(5, 25)}%, Memory: {rng.randint(20, 45)}%, Connections: {rng.randint(2, 10)}/20.",
        f"INFO  {service_id} No anomalies detected in last 60s. request_count={rng.randint(500, 2000)}, error_count=0.",
    ]
    return rng.choice(templates)
