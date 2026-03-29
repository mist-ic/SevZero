"""Tests for the deterministic grader."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.grader import grade_episode


class TestGraderBounds:
    """Score is always 0.0–1.0."""

    def test_perfect_score(self):
        result = grade_episode(
            final_slo_score=1.0,
            steps_taken=3,
            max_steps=10,
            actions_taken=[
                {"tick": 0, "action": "inspect_logs", "target": "svc", "success": True},
                {"tick": 1, "action": "restart_service", "target": "svc", "success": True},
            ],
            terminated=True,
            termination_reason="resolved",
        )
        assert 0.0 <= result.score <= 1.0
        assert result.score > 0.8  # Resolved quickly = high score

    def test_zero_score(self):
        result = grade_episode(
            final_slo_score=0.0,
            steps_taken=10,
            max_steps=10,
            actions_taken=[],
            terminated=True,
            termination_reason="timeout",
        )
        assert result.score == 0.0

    def test_partial_credit(self):
        result = grade_episode(
            final_slo_score=0.5,
            steps_taken=10,
            max_steps=10,
            actions_taken=[
                {"tick": i, "action": "noop", "success": True}
                for i in range(10)
            ],
            terminated=True,
            termination_reason="timeout",
        )
        assert 0.0 < result.score < 1.0

    def test_determinism(self):
        args = dict(
            final_slo_score=0.7,
            steps_taken=5,
            max_steps=20,
            actions_taken=[
                {"tick": 0, "action": "inspect_logs", "target": "svc", "success": True},
                {"tick": 1, "action": "restart_service", "target": "svc", "success": True},
            ],
            terminated=True,
            termination_reason="timeout",
        )
        r1 = grade_episode(**args)
        r2 = grade_episode(**args)
        assert r1.score == r2.score

    def test_resolved_bonus(self):
        """Resolved episodes should score higher than timed-out ones at same SLO."""
        resolved = grade_episode(
            final_slo_score=1.0,
            steps_taken=5,
            max_steps=10,
            actions_taken=[{"tick": i, "action": "restart_service", "target": "svc", "success": True} for i in range(5)],
            terminated=True,
            termination_reason="resolved",
        )
        timeout = grade_episode(
            final_slo_score=1.0,
            steps_taken=10,
            max_steps=10,
            actions_taken=[{"tick": i, "action": "noop", "success": True} for i in range(10)],
            terminated=True,
            termination_reason="timeout",
        )
        assert resolved.score > timeout.score
