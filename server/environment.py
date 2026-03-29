"""
server/environment.py — SevZeroEnvironment: OpenEnv Environment subclass.

Bridges the OpenEnv SDK contract (reset/step/state) with the Simulator engine.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import SevZeroAction, SevZeroObservation, SevZeroState
from server.scenarios import generate_scenario
from server.simulator import Simulator


class SevZeroEnvironment(Environment[SevZeroAction, SevZeroObservation, SevZeroState]):
    """
    SRE Incident Response Environment.

    The agent observes service metrics, alerts, and logs, then issues
    remediation commands to restore SLO compliance across a microservice cluster.
    """

    def __init__(self) -> None:
        super().__init__()
        self._sim = Simulator()
        self._episode_id: Optional[str] = None
        self._task_id: str = "easy"
        self._seed: Optional[int] = None
        self._step_count: int = 0

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
        self._episode_id = episode_id or str(uuid.uuid4())
        self._task_id = kwargs.get("task_id", "easy")
        self._seed = seed if seed is not None else 42
        self._step_count = 0

        # Generate scenario and reset simulator
        scenario = generate_scenario(self._seed, self._task_id)
        self._sim.reset(
            seed=self._seed,
            difficulty=scenario.difficulty,
            failure_specs=scenario.failure_specs,
        )

        return self._build_observation(reward=None, done=False)

    def step(
        self,
        action: SevZeroAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SevZeroObservation:
        self._step_count += 1

        reward = self._sim.step(action.action_type, action.params)
        done = self._sim.terminated

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
        return SevZeroObservation(
            done=done,
            reward=reward,
            # Episode context
            tick=sim.tick,
            episode_id=self._episode_id,
            task_id=self._task_id,
            status=sim.termination_reason or "playing",
            max_steps=sim.max_steps,
            # Health summary
            global_slo_score=round(sim.get_slo_score(), 4),
            observation_summary=sim.get_observation_summary(),
            # Per-service state
            services=sim.get_service_observations(),
            # Alerts
            alerts=sim.get_alerts(),
            # Context
            recent_deploys=[d for d in sim.deploys if d["ticks_ago"] <= 10],
            actions_taken=sim.actions_taken[-10:],
            # Action space
            legal_actions=sim.get_legal_actions(),
            # Diagnostics
            logs=sim.last_logs,
            metric_history=sim.last_metric_history,
            traces=sim.last_traces,
        )
