"""
server/environment.py — SevZeroEnvironment: OpenEnv Environment subclass.

Bridges the OpenEnv SDK contract (reset/step/state) with the Simulator engine.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import SevZeroAction, SevZeroObservation, SevZeroState
from server import schema_drift
from server.grader import grade_episode
from server.scenarios import generate_scenario
from server.simulator import Simulator


class SevZeroEnvironment(Environment[SevZeroAction, SevZeroObservation, SevZeroState]):
    """
    SRE Incident Response Environment.

    The agent observes service metrics, alerts, and logs, then issues
    remediation commands to restore SLO compliance across a microservice cluster.
    """

    def __init__(self, enable_curriculum: bool = False) -> None:
        super().__init__()
        self._sim = Simulator()
        self._curriculum: Any = None
        self._enable_curriculum = enable_curriculum
        if enable_curriculum:
            from server.curriculum import Curriculum

            self._curriculum = Curriculum()
        self._episode_id: Optional[str] = None
        self._task_id: str = "easy"
        self._seed: Optional[int] = None
        self._step_count: int = 0
        self._enable_schema_drift: bool = False
        self._enable_oversight: bool = False
        self._oversight: Any = None
        self._curriculum_stash: Optional[dict] = None

    def close(self) -> None:
        # No-op: the SDK calls close() after every HTTP request, but we need
        # state to persist between reset() and step() calls in HTTP mode.
        # WebSocket sessions manage their own lifecycle.
        pass

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="sevzero",
            description=(
                "SRE Incident Response Environment — an autonomous on-call SRE "
                "managing a microservice cluster undergoing cascading failures"
            ),
            version="1.0.0",
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SevZeroObservation:
        if self._curriculum is not None and self._curriculum_stash is not None:
            s = self._curriculum_stash
            self._curriculum.on_episode_end(
                float(s.get("mean_score", 0.0)),
                bool(s.get("resolved", False)),
                list(s.get("failure_types", [])),
            )
            self._curriculum_stash = None

        self._episode_id = episode_id or str(uuid.uuid4())
        self._task_id = kwargs.get("task_id", "easy")
        self._seed = seed if seed is not None else 42
        self._step_count = 0
        self._enable_schema_drift = bool(kwargs.get("enable_schema_drift", False))
        self._enable_oversight = bool(kwargs.get("enable_oversight", False))
        if self._enable_oversight and self._oversight is None:
            from server.oversight import OversightManager

            self._oversight = OversightManager()
        elif not self._enable_oversight:
            self._oversight = None

        overrides: dict = {}
        if self._curriculum is not None:
            overrides = self._curriculum.next_scenario_overrides() or {}

        scenario = generate_scenario(
            self._seed, self._task_id, **overrides,
        )
        self._sim.reset(
            seed=self._seed,
            difficulty=scenario.difficulty,
            failure_specs=scenario.failure_specs,
            max_steps_override=scenario.max_steps,
        )
        if self._oversight is not None:
            self._oversight.on_reset(
                self._sim, enable=True, max_steps_override=scenario.max_steps,
            )

        return self._build_observation(reward=None, done=False)

    def step(
        self,
        action: SevZeroAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SevZeroObservation:
        self._step_count += 1
        t0 = int(self._sim.tick)

        if self._oversight is not None:
            self._oversight.on_tick_start(self._sim)
            o = self._oversight
            if o.should_block(self._sim, action.action_type, action.params):
                reward = self._sim.step(
                    action.action_type,
                    action.params,
                    prebuilt_record={
                        "action": action.action_type,
                        "target": self._sim.action_fingerprint(
                            action.action_type, action.params,
                        ),
                        "success": False,
                        "note": "oversight_required",
                    },
                    fixed_reward=-0.15,
                )
            else:
                reward = self._sim.step(action.action_type, action.params)
        else:
            reward = self._sim.step(action.action_type, action.params)

        if self._oversight is not None and action.action_type == "request_approval":
            self._oversight.on_request_approval(action.params, t0)

        done = self._sim.terminated
        if done and self._curriculum is not None:
            fts: List[str] = [
                f.failure_type.value for f in self._sim.failures
            ]
            g = grade_episode(
                final_slo_score=self._sim.get_slo_score(),
                steps_taken=self._step_count,
                max_steps=self._sim.max_steps,
                actions_taken=list(self._sim.actions_taken),
                terminated=done,
                termination_reason=self._sim.termination_reason,
            )
            self._curriculum_stash = {
                "mean_score": g.score,
                "resolved": (self._sim.termination_reason == "resolved"),
                "failure_types": fts,
            }

        return self._build_observation(reward=reward, done=done)

    @property
    def state(self) -> SevZeroState:
        return SevZeroState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            seed=self._seed,
            global_slo_score=self._sim.get_slo_score(),
            terminated=self._sim.terminated,
            termination_reason=self._sim.termination_reason,
        )

    def _build_observation(
        self, reward: Optional[float], done: bool,
    ) -> SevZeroObservation:
        sim = self._sim
        legal = sim.get_legal_actions(
            include_request_approval=bool(self._enable_oversight),
        )
        pol: list = list(self._oversight.policy) if self._oversight else []
        pend: list = (
            self._oversight.pending_approvals
            if self._oversight
            else []
        )
        ob: dict = {
            "done": done,
            "reward": reward,
            "tick": sim.tick,
            "episode_id": self._episode_id,
            "task_id": self._task_id,
            "status": sim.termination_reason or "playing",
            "max_steps": sim.max_steps,
            "global_slo_score": round(sim.get_slo_score(), 4),
            "observation_summary": sim.get_observation_summary(),
            "services": sim.get_service_observations(),
            "alerts": sim.get_alerts(),
            "recent_deploys": [d for d in sim.deploys if d["ticks_ago"] <= 10],
            "actions_taken": sim.actions_taken[-10:],
            "legal_actions": legal,
            "logs": sim.last_logs,
            "metric_history": sim.last_metric_history,
            "traces": sim.last_traces,
            "oversight_policy": pol,
            "pending_approvals": pend,
        }
        if self._seed is None or self._episode_id is None:
            raise RuntimeError("Episode context missing (seed, episode_id)")
        ob = schema_drift.apply(
            ob,
            seed=self._seed,
            episode_id=self._episode_id,
            enabled=self._enable_schema_drift,
        )
        return SevZeroObservation(**ob)
