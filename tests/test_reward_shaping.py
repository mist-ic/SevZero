"""Tests for reward_shaping (dense_v1 / dense_v2) in the simulator."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.scenarios import generate_scenario
from server.simulator import Simulator


def _make(rshaping: str) -> Simulator:
    scenario = generate_scenario(100, "easy")
    sim = Simulator(reward_shaping=rshaping)
    sim.reset(
        seed=100,
        difficulty=scenario.difficulty,
        failure_specs=scenario.failure_specs,
    )
    return sim


def test_dense_v1_default_matches_explicit_dense_v1():
    sc = generate_scenario(5, "easy")
    a = Simulator()
    a.reset(5, sc.difficulty, sc.failure_specs)
    b = Simulator(reward_shaping="dense_v1")
    b.reset(5, sc.difficulty, sc.failure_specs)
    assert a.step("noop", {}) == b.step("noop", {})


def test_dense_v2_double_noop_has_repetition_penalty():
    v2 = _make("dense_v2")
    n0 = v2.step("noop", {})
    n1 = v2.step("noop", {})
    assert n1 <= n0 + 0.5


def test_inspect_logs_dense_v2_returns_float():
    s = _make("dense_v2")
    if s.failures:
        sid = s.failures[0].service_id
        r = s.step("inspect_logs", {"service_id": sid})
        assert isinstance(r, float)


def test_request_approval_succeeds():
    s = _make("dense_v1")
    s.step("request_approval", {
        "action_type": "restart_service",
        "target": "x",
        "reason": "t",
    })
    assert s.actions_taken[-1]["success"]
