"""SevZero Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import SevZeroAction, SevZeroObservation


class SevZeroEnv(EnvClient[SevZeroAction, SevZeroObservation, State]):
    """
    Client for the SevZero SRE Incident Response Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling multi-step incident response episodes.

    Example:
        >>> with SevZeroEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset(task_id="easy", seed=42)
        ...     obs = result.observation
        ...     print(obs.global_slo_score)
        ...
        ...     action = SevZeroAction(
        ...         action_type="inspect_logs",
        ...         params={"service_id": "order-service"}
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.logs)

    Example with Docker:
        >>> client = SevZeroEnv.from_docker_image("sevzero-env:latest")
        >>> try:
        ...     result = client.reset(task_id="medium", seed=123)
        ...     action = SevZeroAction(action_type="noop", params={})
        ...     result = client.step(action)
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SevZeroAction) -> Dict:
        return {
            "action_type": action.action_type,
            "params": action.params,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SevZeroObservation]:
        obs_data = payload.get("observation", payload)
        observation = SevZeroObservation(**{
            k: v for k, v in obs_data.items()
            if k in SevZeroObservation.model_fields
        })
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
