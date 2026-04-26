"""
server/schema_drift.py — Per-episode observation schema drift (hard but fair).

Applies 0–2 mutations from a fixed catalog, chosen deterministically from seed
and episode_id. New randomness only via random.Random derived from the seed
pipeline (not module-level random).
"""

from __future__ import annotations

import copy
import hashlib
import random
from typing import Any, Dict, List, Optional

# Fixed catalog indices (order is the application pipeline: renames -> nest -> envelope)
CATALOG = (
    "rename_latency_p99",
    "rename_cpu",
    "nest_service_metrics",
    "cluster_services",
)


def _episode_rng(seed: int, episode_id: str) -> random.Random:
    h = hashlib.sha256(
        f"schema_drift|{seed}|{episode_id or ''}".encode("utf-8")
    ).hexdigest()
    return random.Random(int(h[:16], 16))


def _rename_latency(services: List[Dict[str, Any]], changelog: List[str]) -> None:
    for s in services:
        if "latency_p99_ms" in s and "latency_ms_p99" not in s:
            s["latency_ms_p99"] = s.pop("latency_p99_ms")
    changelog.append("renamed: latency_p99_ms -> latency_ms_p99")


def _rename_cpu(services: List[Dict[str, Any]], changelog: List[str]) -> None:
    for s in services:
        if "cpu_pct" in s and "cpu_utilization" not in s:
            s["cpu_utilization"] = s.pop("cpu_pct")
    changelog.append("renamed: cpu_pct -> cpu_utilization")


def _nest_service_metrics(
    services: List[Dict[str, Any]], changelog: List[str],
) -> None:
    for s in services:
        metrics: Dict[str, Any] = {}
        for k in (
            "error_rate",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "latency_ms_p99",
        ):
            if k in s:
                metrics[k] = s.pop(k)
        if metrics:
            s["metrics"] = metrics
    changelog.append("nested: services[].metrics (error rate + latency fields)")


def _cluster_envelope(
    obs: Dict[str, Any], services: List[Dict[str, Any]], changelog: List[str],
) -> None:
    obs["cluster"] = {"services": services}
    obs["services"] = []
    changelog.append("envelope: services are under cluster.services")


def _choose_mutation_ids(rng: random.Random) -> List[int]:
    k = rng.randint(0, 2)
    if k == 0:
        return []
    ids = sorted(rng.sample(range(len(CATALOG)), k=k))
    return ids


def apply(
    obs: Dict[str, Any],
    *,
    seed: int,
    episode_id: Optional[str],
    enabled: bool = False,
) -> Dict[str, Any]:
    """
    Mutate a copy of the raw observation dict to simulate schema drift.

    When `enabled` is False, only sets `schema_changelog` (empty) and
    `schema_version` to the baseline.
    """
    out = copy.deepcopy(obs)
    if not enabled:
        out["schema_changelog"] = []
        out["schema_version"] = "v1"
        return out

    rng = _episode_rng(seed, episode_id or "")
    selected = set(_choose_mutation_ids(rng))
    changelog: List[str] = []

    services: List[Dict[str, Any]] = copy.deepcopy(out.get("services") or [])

    for mid in range(len(CATALOG)):
        if mid not in selected:
            continue
        name = CATALOG[mid]
        if name == "rename_latency_p99":
            _rename_latency(services, changelog)
        elif name == "rename_cpu":
            _rename_cpu(services, changelog)
        elif name == "nest_service_metrics":
            _nest_service_metrics(services, changelog)
        elif name == "cluster_services":
            _cluster_envelope(out, services, changelog)

    cluster_idx = CATALOG.index("cluster_services")
    if cluster_idx not in selected:
        out["services"] = services
        out["cluster"] = None
    out["schema_changelog"] = changelog
    out["schema_version"] = "v1.2-drift"
    return out
