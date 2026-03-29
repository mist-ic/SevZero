"""
server/app.py — FastAPI application wiring.

Uses OpenEnv SDK's create_app() for WebSocket and standard endpoints
(/ws, /health, /schema, /metadata), then adds our own HTTP routes for
/reset, /step, /state, /tasks, /grader that use a singleton environment.

The SDK's HTTP /reset and /step are stateless (new env per request),
which doesn't work for our multi-step episodes. The WebSocket path
(used by the actual hackathon evaluation) handles sessions correctly.
We override the HTTP paths for testing and inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from openenv.core.env_server import create_app
from openenv.core.env_server.serialization import serialize_observation
from pydantic import BaseModel

from models import SevZeroAction, SevZeroObservation
from server.environment import SevZeroEnvironment
from server.grader import grade_episode
from server.scenarios import TASK_DEFINITIONS

# Singleton environment for HTTP mode
_env = SevZeroEnvironment()

# Create the OpenEnv app (wires /ws, /health, /schema, /metadata, /mcp)
app = create_app(
    SevZeroEnvironment,
    SevZeroAction,
    SevZeroObservation,
    env_name="sevzero",
)


# ---------------------------------------------------------------------------
# Override HTTP endpoints with stateful versions
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_id: str = "easy"


class StepRequest(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = None


# Remove SDK's stateless routes and replace with ours
_routes_to_remove = {"/reset", "/step", "/state"}
app.routes[:] = [r for r in app.routes if getattr(r, "path", None) not in _routes_to_remove]


@app.post("/reset")
async def reset_env(request: ResetRequest) -> Dict[str, Any]:
    """Reset the environment and return initial observation."""
    obs = _env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task_id=request.task_id,
    )
    return serialize_observation(obs)


@app.post("/step")
async def step_env(request: StepRequest) -> Dict[str, Any]:
    """Execute an action and return the new observation."""
    action = SevZeroAction(**request.action)
    obs = _env.step(action, timeout_s=request.timeout_s)
    return serialize_observation(obs)


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """Return the current environment state."""
    state = _env.state
    return state.model_dump()


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
