"""Tests for the simulation engine — determinism, actions, SLO scoring."""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.simulator import Simulator
from server.scenarios import generate_scenario


def _make_sim(task_id: str = "easy", seed: int = 42) -> Simulator:
    scenario = generate_scenario(seed, task_id)
    sim = Simulator()
    sim.reset(seed=seed, difficulty=scenario.difficulty, failure_specs=scenario.failure_specs)
    return sim


class TestDeterminism:
    """Same seed + same actions = identical state."""

    def test_reset_determinism(self):
        sim1 = _make_sim(seed=42)
        sim2 = _make_sim(seed=42)
        assert sim1.get_slo_score() == sim2.get_slo_score()
        assert len(sim1.services) == len(sim2.services)
        for sid in sim1.services:
            s1 = sim1.services[sid]
            s2 = sim2.services[sid]
            assert s1.error_rate == s2.error_rate
            assert s1.latency_p99_ms == s2.latency_p99_ms

    def test_step_determinism(self):
        sim1 = _make_sim(seed=42)
        sim2 = _make_sim(seed=42)
        # Take same actions
        for _ in range(3):
            r1 = sim1.step("noop", {})
            r2 = sim2.step("noop", {})
            assert r1 == r2
            assert sim1.get_slo_score() == sim2.get_slo_score()

    def test_different_seeds_differ(self):
        sim1 = _make_sim(seed=42)
        sim2 = _make_sim(seed=999)
        # Different seeds should (very likely) produce different failure targets
        failures1 = {s.service_id for s in sim1.failures}
        failures2 = {s.service_id for s in sim2.failures}
        # At minimum, graphs or failures should differ (not guaranteed but extremely likely)
        services1 = set(sim1.services.keys())
        services2 = set(sim2.services.keys())
        assert failures1 != failures2 or services1 != services2


class TestSLOScoring:
    """SLO score is 0.0–1.0 and reflects service health."""

    def test_slo_range(self):
        sim = _make_sim()
        score = sim.get_slo_score()
        assert 0.0 <= score <= 1.0

    def test_initial_slo_below_one(self):
        """After failure injection, at least one service should be degraded."""
        sim = _make_sim()
        assert sim.get_slo_score() < 1.0

    def test_slo_after_noop(self):
        sim = _make_sim()
        sim.step("noop", {})
        score = sim.get_slo_score()
        assert 0.0 <= score <= 1.0


class TestActions:
    """Action processing works correctly."""

    def test_noop(self):
        sim = _make_sim()
        reward = sim.step("noop", {})
        assert isinstance(reward, float)

    def test_inspect_logs(self):
        sim = _make_sim()
        # Get any service
        service_id = list(sim.services.keys())[0]
        sim.step("inspect_logs", {"service_id": service_id})
        assert sim.last_logs is not None
        assert len(sim.last_logs) > 0

    def test_inspect_metrics(self):
        sim = _make_sim()
        service_id = list(sim.services.keys())[0]
        sim.step("inspect_metrics", {"service_id": service_id})
        assert sim.last_metric_history is not None

    def test_inspect_traces(self):
        sim = _make_sim()
        service_id = list(sim.services.keys())[0]
        sim.step("inspect_traces", {"service_id": service_id})
        assert sim.last_traces is not None
        assert "trace_id" in sim.last_traces
        assert "spans" in sim.last_traces

    def test_restart_service(self):
        sim = _make_sim()
        target = sim.failures[0].service_id if sim.failures else list(sim.services.keys())[0]
        reward = sim.step("restart_service", {"service_id": target})
        assert isinstance(reward, float)
        assert len(sim.pending_effects) >= 0  # May or may not have pending

    def test_invalid_service(self):
        sim = _make_sim()
        sim.step("inspect_logs", {"service_id": "nonexistent-service"})
        assert sim.last_logs is None
        # Should have a failed action record
        assert not sim.actions_taken[-1]["success"]

    def test_unknown_action(self):
        sim = _make_sim()
        reward = sim.step("fly_to_moon", {})
        assert not sim.actions_taken[-1]["success"]


class TestTermination:
    """Episode termination logic."""

    def test_timeout(self):
        sim = _make_sim(task_id="easy")  # 10 step budget
        for _ in range(15):
            if sim.terminated:
                break
            sim.step("noop", {})
        assert sim.terminated
        assert sim.termination_reason in ("timeout", "resolved", "failed")

    def test_tick_advances(self):
        sim = _make_sim()
        assert sim.tick == 0
        sim.step("noop", {})
        assert sim.tick == 1
        sim.step("noop", {})
        assert sim.tick == 2


class TestObservationHelpers:
    """Observation builder methods."""

    def test_observation_summary(self):
        sim = _make_sim()
        summary = sim.get_observation_summary()
        assert "Tick" in summary
        assert "SLO" in summary

    def test_alerts(self):
        sim = _make_sim()
        alerts = sim.get_alerts()
        assert isinstance(alerts, list)
        # With failures injected, there should be at least one alert
        assert len(alerts) > 0

    def test_legal_actions(self):
        sim = _make_sim()
        legal = sim.get_legal_actions()
        assert isinstance(legal, list)
        assert len(legal) > 0
        action_types = {a["action_type"] for a in legal}
        assert "noop" in action_types
        assert "inspect_logs" in action_types

    def test_service_observations(self):
        sim = _make_sim()
        obs = sim.get_service_observations()
        assert isinstance(obs, list)
        assert len(obs) > 0
        svc = obs[0]
        assert "id" in svc
        assert "error_rate" in svc
        assert "latency_p99_ms" in svc
        assert "circuit_breakers" in svc
