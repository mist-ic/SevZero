"""Oversight / governance (OversightManager)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.oversight import OversightManager
from server.scenarios import generate_scenario
from server.simulator import Simulator


def _sim_hard():
    sc = generate_scenario(9, "hard")
    sim = Simulator()
    sim.reset(9, sc.difficulty, sc.failure_specs)
    return sim


def test_restart_postgres_requires_governance():
    sim = _sim_hard()
    om = OversightManager()
    om.on_reset(sim, True, 50)
    sid = "postgres-primary"
    if sid not in sim.services:
        sid = next((s for s in sim.services if "postgres" in s), None)
    if sid is None:
        return
    assert om.is_high_impact(sim, "restart_service", {"service_id": sid})
    sim.tick = 0
    assert om.should_block(sim, "restart_service", {"service_id": sid})


def test_request_then_grant_allows():
    sim = _sim_hard()
    om = OversightManager()
    om.on_reset(sim, True, 50)
    sid = "postgres-primary"
    if sid not in sim.services:
        sid = next((s for s in sim.services if "postgres" in s), None)
    if sid is None:
        return
    # Start tick 0: submit approval request for this restart
    sim.tick = 0
    om.on_request_approval(
        {
            "action_type": "restart_service",
            "target": sid,
            "reason": "need restart",
        },
        0,
    )
    # tick 1: manager grants
    sim.tick = 1
    om.on_tick_start(sim)
    assert not om.should_block(sim, "restart_service", {"service_id": sid})


def test_policy_surface():
    sim = _sim_hard()
    om = OversightManager()
    om.on_reset(sim, True, 50)
    assert any("postgres" in str(x).lower() for x in om.policy[0].values())


def test_rebalance_high_pct_is_high_impact():
    sim = _sim_hard()
    if not (sim.graph and sim.graph.has_multiple_regions):
        return
    om = OversightManager()
    om.on_reset(sim, True, 50)
    a, b = sim.graph.regions[0], sim.graph.regions[1]
    assert om.is_high_impact(
        sim, "rebalance_traffic", {"from_region": a, "to_region": b, "pct": 45},
    )
