"""
server/scenarios.py — Procedural scenario generation from seed + difficulty.

Maps difficulty to graph topology, failure count, and failure placement.
Same seed + same difficulty = identical scenario every time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from server.failures import (
    FailureSpec,
    FailureType,
    make_failure_spec,
    select_failure_type,
    select_multi_root_failures,
)
from server.graph import ServiceGraph, generate_graph


@dataclass
class ScenarioConfig:
    """Complete scenario definition for one episode."""
    difficulty: str
    seed: int
    graph: ServiceGraph
    failure_specs: List[FailureSpec]
    max_steps: int
    description: str


# ---------------------------------------------------------------------------
# Task definitions (the 3 required tasks)
# ---------------------------------------------------------------------------

TASK_DEFINITIONS = [
    {
        "task_id": "easy",
        "name": "Single Service Outage",
        "difficulty": "easy",
        "description": (
            "A single service in a small linear microservice chain is experiencing failures. "
            "Diagnose the root cause and apply the correct remediation within 10 steps."
        ),
        "max_steps": 10,
        "num_failures": 1,
    },
    {
        "task_id": "medium",
        "name": "Cascading Failure",
        "difficulty": "medium",
        "description": (
            "A failure in a shared infrastructure service is cascading through a branching "
            "dependency graph. Trace the root cause upstream from symptomatic services and "
            "remediate within 20 steps."
        ),
        "max_steps": 20,
        "num_failures": 1,
    },
    {
        "task_id": "hard",
        "name": "Multi-Root Sev-0 Incident",
        "difficulty": "hard",
        "description": (
            "Multiple simultaneous failures across a multi-region microservice architecture. "
            "Failures may have conflicting mitigations. Triage, diagnose, and resolve all "
            "root causes within 50 steps."
        ),
        "max_steps": 50,
        "num_failures": 3,
    },
]


def get_task_definition(task_id: str) -> dict:
    """Get a task definition by ID."""
    for t in TASK_DEFINITIONS:
        if t["task_id"] == task_id:
            return t
    raise ValueError(f"Unknown task_id: {task_id!r}. Must be one of: easy, medium, hard")


# ---------------------------------------------------------------------------
# Failure placement logic
# ---------------------------------------------------------------------------


def _pick_failure_target(
    graph: ServiceGraph,
    failure_type: FailureType,
    rng: random.Random,
    exclude: set,
) -> Optional[str]:
    """Pick an appropriate service to inject this failure type into."""
    candidates = []

    for node in graph.nodes:
        if node.id in exclude:
            continue

        # Cache failures only on cache services
        if failure_type == FailureType.CACHE_FAILURE:
            if node.is_cache:
                candidates.append(node.id)
            continue

        # DB degradation on infra services (postgres, etc.)
        if failure_type == FailureType.DB_DEGRADATION:
            if node.layer == "infra" and "postgres" in node.id:
                candidates.append(node.id)
            continue

        # Network errors prefer non-edge services
        if failure_type == FailureType.NETWORK_ERROR:
            if node.layer != "edge":
                candidates.append(node.id)
            continue

        # Config errors on any non-edge service
        if failure_type in (FailureType.CONFIG_STARTUP, FailureType.CONFIG_RUNTIME):
            if node.layer != "edge":
                candidates.append(node.id)
            continue

        # Bad deploy on business or identity services
        if failure_type == FailureType.BAD_DEPLOY:
            if node.layer in ("business", "identity"):
                candidates.append(node.id)
            continue

        # Resource leak on business services
        if failure_type == FailureType.RESOURCE_LEAK:
            if node.layer in ("business", "identity"):
                candidates.append(node.id)
            continue

        # Crash on any non-edge service
        if failure_type == FailureType.CRASH:
            if node.layer != "edge":
                candidates.append(node.id)
            continue

        # Cascading latency: prefer hotspot infra or busy business
        if failure_type == FailureType.CASCADING_LATENCY:
            if node.is_hotspot or node.layer == "business":
                candidates.append(node.id)
            continue

    if not candidates:
        # Fallback: any non-edge service
        candidates = [n.id for n in graph.nodes if n.layer != "edge" and n.id not in exclude]

    if not candidates:
        return None

    return rng.choice(candidates)


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------


def generate_scenario(seed: int, task_id: str) -> ScenarioConfig:
    """
    Generate a complete scenario for the given task and seed.
    Deterministic: same seed + same task_id = identical scenario.
    """
    task = get_task_definition(task_id)
    rng = random.Random(seed)

    # Generate graph
    difficulty = task["difficulty"]
    graph = generate_graph(difficulty, rng)

    # Select and place failures
    num_failures = task["num_failures"]
    used_services: set = set()
    failure_specs: List[FailureSpec] = []

    if num_failures == 1:
        ft = select_failure_type(rng)
        target = _pick_failure_target(graph, ft, rng, used_services)
        if target:
            spec = make_failure_spec(target, ft, rng)
            failure_specs.append(spec)
            used_services.add(target)
    else:
        failure_types = select_multi_root_failures(rng, count=num_failures)
        for ft in failure_types:
            target = _pick_failure_target(graph, ft, rng, used_services)
            if target:
                spec = make_failure_spec(target, ft, rng)
                failure_specs.append(spec)
                used_services.add(target)

    return ScenarioConfig(
        difficulty=difficulty,
        seed=seed,
        graph=graph,
        failure_specs=failure_specs,
        max_steps=task["max_steps"],
        description=task["description"],
    )
