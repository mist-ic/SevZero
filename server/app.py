"""
server/app.py — FastAPI application wiring.

Uses OpenEnv SDK's create_app() for core endpoints (/reset, /step, /state, /ws, /health),
then adds custom routes for /tasks, /grader, and /baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from openenv.core.env_server import create_app
from pydantic import BaseModel

from models import SevZeroAction, SevZeroObservation
from server.environment import SevZeroEnvironment
from server.grader import grade_episode
from server.scenarios import TASK_DEFINITIONS


# Create the OpenEnv app (wires /reset, /step, /state, /ws, /health, /schema, /metadata)
app = create_app(
    SevZeroEnvironment,
    SevZeroAction,
    SevZeroObservation,
    env_name="sevzero",
)


# ---------------------------------------------------------------------------
# Custom routes
# ---------------------------------------------------------------------------


@app.get("/tasks")
async def list_tasks() -> List[Dict[str, Any]]:
    """Return the 3 task definitions (easy, medium, hard)."""
    return [
        {
            "task_id": t["task_id"],
            "name": t["name"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_steps": t["max_steps"],
        }
        for t in TASK_DEFINITIONS
    ]


class GraderRequest(BaseModel):
    final_slo_score: float
    steps_taken: int
    max_steps: int
    actions_taken: List[Dict[str, Any]]
    terminated: bool
    termination_reason: Optional[str] = None


@app.post("/grader")
async def grade(request: GraderRequest) -> Dict[str, Any]:
    """
    Deterministic grading endpoint.
    Accepts episode results and returns a score 0.0–1.0 with breakdown.
    """
    result = grade_episode(
        final_slo_score=request.final_slo_score,
        steps_taken=request.steps_taken,
        max_steps=request.max_steps,
        actions_taken=request.actions_taken,
        terminated=request.terminated,
        termination_reason=request.termination_reason,
    )
    return {
        "score": result.score,
        "slo_recovery": result.slo_recovery,
        "action_efficiency": result.action_efficiency,
        "time_efficiency": result.time_efficiency,
        "details": result.details,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
