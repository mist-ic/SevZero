"""
server/graph.py — Service dependency graph generation.

Builds layered tree-like DAGs matching real production microservice topologies,
grounded in Alibaba trace analysis (depth ~3, 5% hotspot services, sparse edges).

Design principles:
- Services chosen from realistic role pools (not generic names)
- Layered: edge → identity → business → infra; edge → leaf dependencies
- Dependency edges are directed (A depends_on B = A calls B)
- ~5% of services are high-in-degree hotspots (shared cache, DB, auth)
- Sparse and tree-like; most nodes have in-degree 1
- Conditional edges have activation_probability < 1.0 (Easy: all 1.0)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Service role pools (realistic names, not generic)
# ---------------------------------------------------------------------------

_EDGE_POOL = [
    "api-gateway",
    "graphql-gateway",
    "bff-web",
    "bff-mobile",
    "cdn-edge",
]

_IDENTITY_POOL = [
    "auth-service",
    "identity-provider",
    "session-service",
    "oauth-service",
    "token-service",
]

_BUSINESS_POOL = [
    "order-service",
    "payment-service",
    "inventory-service",
    "catalog-service",
    "pricing-service",
    "cart-service",
    "checkout-service",
    "shipping-service",
    "recommendation-service",
    "search-service",
    "review-service",
    "subscription-service",
    "billing-service",
    "refund-service",
    "notification-service",
]

_INFRA_POOL = [
    "postgres-primary",
    "postgres-replica",
    "redis-cache",
    "redis-session",
    "kafka-broker",
    "elasticsearch",
    "object-storage",
    "config-service",
]

_CROSS_CUTTING_POOL = [
    "email-service",
    "sms-service",
    "metrics-collector",
    "fraud-service",
    "audit-service",
    "feature-flags",
    "rate-limiter",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ServiceNode:
    """A service node in the dependency graph."""

    id: str
    layer: str  # "edge" | "identity" | "business" | "infra" | "cross-cutting"

    # Queueing theory baseline parameters (modified by failures at runtime)
    base_arrival_rate: float = 100.0       # λ — requests/tick at baseline
    base_service_time_local: float = 0.05  # S_local — seconds per request (local work)
    thread_pool_size: int = 50             # T — max concurrent in-flight requests

    # Default config (tunable by agent)
    default_timeout_ms: int = 5000
    default_retry_max: int = 3
    default_retry_backoff: bool = False
    default_circuit_breaker_threshold: float = 0.5
    default_pool_size: int = 20

    # Deployment defaults
    default_replicas: int = 2
    default_version: str = "v1.0.0"

    # Whether this node is a "hotspot" (high in-degree shared infra)
    is_hotspot: bool = False

    # Whether this is a background-job node (can be pause_job target)
    has_background_job: bool = False

    # Whether this is a cache node (can be clear_cache target)
    is_cache: bool = False

    # Max replicas the agent can scale to
    max_replicas: int = 8

    # Region (for Hard mode multi-region topologies)
    region: str = "us-east-1"


@dataclass
class DependencyEdge:
    """A directed dependency edge: source depends on (calls) target."""

    source: str   # service that makes the call
    target: str   # service that receives the call

    # Fraction of ticks this edge is active (1.0 = always; 0.2 = ~20% of ticks)
    activation_probability: float = 1.0

    # Edge type for documentation
    edge_type: str = "sync"  # "sync" | "async" | "optional"


@dataclass
class ServiceGraph:
    """Complete service dependency graph for one episode."""

    nodes: List[ServiceNode] = field(default_factory=list)
    edges: List[DependencyEdge] = field(default_factory=list)

    # Derived lookup structures (populated after build)
    node_map: Dict[str, ServiceNode] = field(default_factory=dict)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)  # source → [targets]
    reverse_adjacency: Dict[str, List[str]] = field(default_factory=dict)  # target → [callers]

    # Metadata
    difficulty: str = "easy"
    has_multiple_regions: bool = False
    regions: List[str] = field(default_factory=lambda: ["us-east-1"])
    cache_services: List[str] = field(default_factory=list)
    background_jobs: List[str] = field(default_factory=list)

    def build_indices(self) -> None:
        """Build lookup maps after nodes/edges are populated."""
        self.node_map = {n.id: n for n in self.nodes}
        self.adjacency = {n.id: [] for n in self.nodes}
        self.reverse_adjacency = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            self.adjacency[edge.source].append(edge.target)
            self.reverse_adjacency[edge.target].append(edge.source)
        self.cache_services = [n.id for n in self.nodes if n.is_cache]
        self.background_jobs = [n.id for n in self.nodes if n.has_background_job]


# ---------------------------------------------------------------------------
# Graph generation functions
# ---------------------------------------------------------------------------


def _pick(pool: List[str], rng: random.Random, exclude: set) -> Optional[str]:
    """Pick a random name from pool not already in exclude set."""
    choices = [x for x in pool if x not in exclude]
    if not choices:
        return None
    return rng.choice(choices)


def _make_node(
    service_id: str,
    layer: str,
    is_hotspot: bool = False,
    is_cache: bool = False,
    has_background_job: bool = False,
    arrival_rate: float = 100.0,
    service_time: float = 0.05,
    thread_pool: int = 50,
) -> ServiceNode:
    """Create a ServiceNode with sensible per-layer defaults."""
    # Infra nodes handle more concurrency, edge nodes get more traffic
    if layer == "edge":
        arrival_rate = 500.0
        thread_pool = 100
    elif layer == "infra":
        arrival_rate = 200.0
        service_time = 0.02   # DBs are fast per-query
        thread_pool = 30
        if is_cache:
            service_time = 0.001
            thread_pool = 200

    return ServiceNode(
        id=service_id,
        layer=layer,
        base_arrival_rate=arrival_rate,
        base_service_time_local=service_time,
        thread_pool_size=thread_pool,
        is_hotspot=is_hotspot,
        is_cache=is_cache,
        has_background_job=has_background_job,
    )


def generate_easy_graph(rng: random.Random) -> ServiceGraph:
    """
    Easy: 3-5 services, linear chain.
    api-gateway → order-service → postgres-primary
    Agent must identify and fix one failing service in this simple topology.
    """
    graph = ServiceGraph(difficulty="easy")
    used: set = set()

    # Always have a gateway at the edge
    gateway_id = "api-gateway"
    used.add(gateway_id)

    # Pick 1-2 business services
    biz_count = rng.randint(1, 2)
    biz_nodes = []
    for _ in range(biz_count):
        svc = _pick(_BUSINESS_POOL, rng, used)
        if svc:
            used.add(svc)
            biz_nodes.append(svc)

    # Always have one DB at the leaf
    db_id = "postgres-primary"
    used.add(db_id)

    # Optionally add a cache
    add_cache = rng.random() > 0.4
    cache_id = "redis-cache" if add_cache else None
    if cache_id:
        used.add(cache_id)

    # Build nodes
    graph.nodes.append(_make_node(gateway_id, "edge"))
    for biz in biz_nodes:
        graph.nodes.append(_make_node(biz, "business"))
    graph.nodes.append(
        _make_node(db_id, "infra", is_hotspot=True, arrival_rate=200.0)
    )
    if cache_id:
        graph.nodes.append(
            _make_node(cache_id, "infra", is_hotspot=True, is_cache=True)
        )

    # Build linear dependency chain: gateway → biz[0] → biz[1]? → db
    chain = [gateway_id] + biz_nodes + [db_id]
    for i in range(len(chain) - 1):
        graph.edges.append(DependencyEdge(source=chain[i], target=chain[i + 1]))

    # If cache exists, business services call it (optional edge for realism)
    if cache_id and biz_nodes:
        for biz in biz_nodes:
            graph.edges.append(
                DependencyEdge(source=biz, target=cache_id, activation_probability=0.9)
            )

    graph.build_indices()
    return graph


def generate_medium_graph(rng: random.Random) -> ServiceGraph:
    """
    Medium: 8-15 services, branching DAG.
    gateway → auth + 3-4 domain services → shared DB + cache + kafka.
    Agent must trace through the graph to find a root cause that's upstream
    of the service showing the worst symptoms.
    """
    graph = ServiceGraph(difficulty="medium")
    used: set = set()

    # Edge layer: 1 gateway
    gateway_id = "api-gateway"
    used.add(gateway_id)
    graph.nodes.append(_make_node(gateway_id, "edge"))

    # Identity layer: auth (gateway always calls auth)
    auth_id = "auth-service"
    used.add(auth_id)
    graph.nodes.append(_make_node(auth_id, "identity"))
    graph.edges.append(DependencyEdge(source=gateway_id, target=auth_id))

    # Business layer: 4-6 domain services fanning out from gateway
    biz_count = rng.randint(4, 6)
    biz_nodes = []
    for _ in range(biz_count):
        svc = _pick(_BUSINESS_POOL, rng, used)
        if svc:
            used.add(svc)
            biz_nodes.append(svc)
            graph.nodes.append(_make_node(svc, "business"))
            graph.edges.append(DependencyEdge(source=gateway_id, target=svc))

    # Infra layer: shared DB + cache (hotspot nodes)
    db_id = "postgres-primary"
    cache_id = "redis-cache"
    used.update([db_id, cache_id])
    graph.nodes.append(_make_node(db_id, "infra", is_hotspot=True, arrival_rate=300.0))
    graph.nodes.append(_make_node(cache_id, "infra", is_hotspot=True, is_cache=True))

    # Business services call the shared DB and cache
    for biz in biz_nodes:
        graph.edges.append(DependencyEdge(source=biz, target=db_id))
        # Cache: most biz services call it, but with high-freq optional
        graph.edges.append(
            DependencyEdge(source=biz, target=cache_id, activation_probability=0.8)
        )

    # Optionally add kafka as an async edge (1-2 business services produce to it)
    if rng.random() > 0.4:
        kafka_id = "kafka-broker"
        used.add(kafka_id)
        graph.nodes.append(
            _make_node(kafka_id, "infra", has_background_job=True)
        )
        producers = rng.sample(biz_nodes, min(2, len(biz_nodes)))
        for p in producers:
            graph.edges.append(
                DependencyEdge(source=p, target=kafka_id, edge_type="async", activation_probability=0.6)
            )

    # Cross-cutting: add 1-2 optional services (fraud, notification) called by some biz
    cross_count = rng.randint(1, 2)
    for _ in range(cross_count):
        svc = _pick(_CROSS_CUTTING_POOL, rng, used)
        if svc and biz_nodes:
            used.add(svc)
            caller = rng.choice(biz_nodes)
            graph.nodes.append(_make_node(svc, "cross-cutting"))
            graph.edges.append(
                DependencyEdge(source=caller, target=svc, activation_probability=0.3)
            )

    graph.build_indices()
    return graph


def generate_hard_graph(rng: random.Random) -> ServiceGraph:
    """
    Hard: 15-30 services, complex multi-region DAG with hotspots,
    conditional edges, multiple infra tiers, and background jobs.
    Agent must manage a Sev-0 multi-root incident with conflicting mitigations.
    """
    graph = ServiceGraph(difficulty="hard", has_multiple_regions=True)
    graph.regions = ["us-east-1", "us-west-2"]
    used: set = set()

    all_biz_nodes: List[str] = []

    # Build per-region sub-graphs, then connect them
    for region in graph.regions:
        suffix = "-east" if "east" in region else "-west"

        # Edge: one gateway per region
        gw = f"api-gateway{suffix}"
        used.add(gw)
        node = _make_node(gw, "edge")
        node.region = region
        graph.nodes.append(node)

        # Identity: auth per region
        auth = f"auth-service{suffix}"
        used.add(auth)
        node = _make_node(auth, "identity")
        node.region = region
        graph.nodes.append(node)
        graph.edges.append(DependencyEdge(source=gw, target=auth))

        # Business: 4-6 services per region
        region_biz: List[str] = []
        for _ in range(rng.randint(4, 6)):
            svc_base = _pick(_BUSINESS_POOL, rng, used)
            if svc_base:
                svc = f"{svc_base}{suffix}"
                used.add(svc)
                region_biz.append(svc)
                node = _make_node(svc, "business")
                node.region = region
                graph.nodes.append(node)
                graph.edges.append(DependencyEdge(source=gw, target=svc))

        all_biz_nodes.extend(region_biz)

        # Infra: per-region replicas (postgres-replica is a hotspot)
        pg_replica = f"postgres-replica{suffix}"
        redis_svc = f"redis-cache{suffix}"
        used.update([pg_replica, redis_svc])
        node = _make_node(pg_replica, "infra", is_hotspot=True)
        node.region = region
        graph.nodes.append(node)
        node = _make_node(redis_svc, "infra", is_hotspot=True, is_cache=True)
        node.region = region
        graph.nodes.append(node)

        for biz in region_biz:
            graph.edges.append(DependencyEdge(source=biz, target=pg_replica))
            graph.edges.append(
                DependencyEdge(source=biz, target=redis_svc, activation_probability=0.85)
            )

    # Shared global infra (hotspots called by both regions)
    pg_primary = "postgres-primary"
    kafka = "kafka-broker"
    config_svc = "config-service"
    used.update([pg_primary, kafka, config_svc])

    graph.nodes.append(_make_node(pg_primary, "infra", is_hotspot=True, arrival_rate=500.0))
    graph.nodes.append(_make_node(kafka, "infra", has_background_job=True))
    graph.nodes.append(_make_node(config_svc, "infra", is_hotspot=True))

    # Replicas call primary (replication)
    for region in graph.regions:
        suffix = "-east" if "east" in region else "-west"
        graph.edges.append(
            DependencyEdge(source=f"postgres-replica{suffix}", target=pg_primary)
        )

    # Business services use kafka for async events and config-service for feature flags
    for biz in all_biz_nodes:
        if rng.random() > 0.5:
            graph.edges.append(
                DependencyEdge(source=biz, target=kafka, edge_type="async", activation_probability=0.5)
            )
        graph.edges.append(
            DependencyEdge(source=biz, target=config_svc, activation_probability=0.2)
        )

    # Cross-cutting services (low-freq optional edges)
    for _ in range(rng.randint(2, 3)):
        svc = _pick(_CROSS_CUTTING_POOL, rng, used)
        if svc and all_biz_nodes:
            used.add(svc)
            caller = rng.choice(all_biz_nodes)
            graph.nodes.append(_make_node(svc, "cross-cutting"))
            graph.edges.append(
                DependencyEdge(source=caller, target=svc, activation_probability=0.25)
            )

    graph.build_indices()
    return graph


def generate_graph(difficulty: str, rng: random.Random) -> ServiceGraph:
    """Generate a service dependency graph for the given difficulty level."""
    if difficulty == "easy":
        return generate_easy_graph(rng)
    elif difficulty == "medium":
        return generate_medium_graph(rng)
    elif difficulty == "hard":
        return generate_hard_graph(rng)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty!r}. Must be easy|medium|hard.")
