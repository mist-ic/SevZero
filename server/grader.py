"""
server/grader.py — Deterministic grading for SevZero episodes.

Score formula:
    score = slo_recovery * 0.70 + action_efficiency * 0.15 + time_efficiency * 0.15

All inputs are derived from the episode state — fully deterministic.
Score is continuous 0.0–1.0 with partial credit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GradeResult:
    """Grading result with breakdown."""
    score: float
    slo_recovery: float
    action_efficiency: float
    time_efficiency: float
    details: Dict[str, Any]


def grade_episode(
    final_slo_score: float,
    steps_taken: int,
    max_steps: int,
    actions_taken: List[Dict[str, Any]],
    terminated: bool,
    termination_reason: Optional[str],
) -> GradeResult:
    """
    Grade a completed episode.

    Args:
        final_slo_score: fraction of services meeting SLO at episode end (0.0–1.0)
        steps_taken: number of steps the agent took
        max_steps: maximum allowed steps for this task
        actions_taken: list of action records
        terminated: whether the episode ended
        termination_reason: "resolved" | "timeout" | "failed" | None
    """
    # --- SLO recovery (70%) ---
    # Direct fraction of services recovered
    slo_recovery = final_slo_score

    # Bonus for full resolution
    if termination_reason == "resolved":
        slo_recovery = 1.0

    # --- Action efficiency (15%) ---
    # Penalize wasted actions (noops when degraded, failed actions, redundant inspects)
    total_actions = len(actions_taken)
    if total_actions == 0:
        action_efficiency = 0.0
    else:
        successful = sum(1 for a in actions_taken if a.get("success", False))
        remediation_actions = sum(
            1 for a in actions_taken
            if a.get("action") not in ("inspect_logs", "inspect_metrics", "inspect_traces", "noop")
            and a.get("success", False)
        )
        inspect_actions = sum(
            1 for a in actions_taken
            if a.get("action") in ("inspect_logs", "inspect_metrics", "inspect_traces")
        )

        # Good ratio: some inspection + targeted remediation
        success_rate = successful / total_actions
        # Penalize excessive inspections (>50% of budget is too much looking, not enough doing)
        inspect_penalty = max(0.0, (inspect_actions / total_actions) - 0.5) if total_actions > 0 else 0.0
        action_efficiency = max(0.0, success_rate - inspect_penalty)

    # --- Time efficiency (15%) ---
    # Faster resolution = higher score
    if max_steps == 0:
        time_efficiency = 0.0
    elif termination_reason == "resolved":
        # Resolved: reward faster resolution
        time_efficiency = max(0.1, 1.0 - (steps_taken / max_steps))
    else:
        # Not resolved: partial credit based on how close we got
        time_efficiency = final_slo_score * 0.3

    # --- Final score ---
    score = (
        slo_recovery * 0.70
        + action_efficiency * 0.15
        + time_efficiency * 0.15
    )
    score = max(0.0, min(1.0, round(score, 4)))

    return GradeResult(
        score=score,
        slo_recovery=round(slo_recovery, 4),
        action_efficiency=round(action_efficiency, 4),
        time_efficiency=round(time_efficiency, 4),
        details={
            "final_slo_score": round(final_slo_score, 4),
            "steps_taken": steps_taken,
            "max_steps": max_steps,
            "termination_reason": termination_reason,
            "total_actions": len(actions_taken),
        },
    )
