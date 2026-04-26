"""
Async HTTP client for the SevZero OpenEnv server (stateful /reset, /step, /state, /grader).
Used by train_grpo rollout_func. Does not use root client.py (WebSocket); mirrors inference.py HTTP usage.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx

_DEFAULT_TIMEOUT = 120.0
_MAX_RETRIES = 5
_BACKOFF = 1.6


def _space_id_to_runtime_url(space_id: str) -> str:
    """HF Space 'org/name' -> https://org-name.hf.space (common runtime URL)."""
    space_id = space_id.strip()
    if space_id.startswith("http"):
        return space_id.rstrip("/")
    parts = space_id.split("/")
    if len(parts) == 2:
        org, name = parts[0], parts[1]
        # HF uses lowercase, slashes -> dashes in subdomains
        sub = f"{org}-{name}".replace("_", "-").lower()
        return f"https://{sub}.hf.space"
    raise ValueError(f"Invalid space_id (expected 'org/name' or URL): {space_id!r}")


def _backoff_delay(attempt: int) -> float:
    return min(30.0, _BACKOFF**attempt)


def _is_transient_status(code: int) -> bool:
    return code in (429, 500, 502, 503, 504)


class AsyncSevZeroEnvClient:
    """
    Minimal async env client: reset / step / state / grader.
    Pass base_url from SEVZERO_ENV_URL or from_hf_space().
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            base_url=self._base,
            headers=headers,
            timeout=timeout,
        )

    @classmethod
    def from_hf_space(
        cls,
        space_id: str,
        token: Optional[str] = None,
    ) -> "AsyncSevZeroEnvClient":
        """
        space_id: 'organization/space_name' (HF Space) or a full http(s) URL.
        For private Spaces, pass a read token with Space access.
        """
        return cls(_space_id_to_runtime_url(space_id), token=token or os.environ.get("HF_TOKEN"))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
    ) -> httpx.Response:
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                r = await self._client.request(method, path, json=json)
                if r.status_code < 400:
                    return r
                if _is_transient_status(r.status_code) and attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_backoff_delay(attempt + 1))
                    continue
                return r
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_err = e
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_backoff_delay(attempt + 1))
                    continue
                raise
        if last_err:
            raise last_err
        raise RuntimeError("request failed")

    async def reset(
        self,
        *,
        task_id: str = "hard",
        seed: int = 13,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"task_id": task_id, "seed": seed}
        if episode_id:
            body["episode_id"] = episode_id
        r = await self._request("POST", "/reset", json=body)
        r.raise_for_status()
        return r.json()

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = await self._request("POST", "/step", json={"action": action})
        r.raise_for_status()
        return r.json()

    async def get_state(self) -> Dict[str, Any]:
        r = await self._request("GET", "/state")
        r.raise_for_status()
        return r.json()

    async def grade_episode(
        self,
        *,
        final_slo_score: float,
        steps_taken: int,
        max_steps: int,
        actions_taken: List[Dict[str, Any]],
        terminated: bool,
        termination_reason: Optional[str],
    ) -> Dict[str, Any]:
        r = await self._request(
            "POST",
            "/grader",
            json={
                "final_slo_score": final_slo_score,
                "steps_taken": steps_taken,
                "max_steps": max_steps,
                "actions_taken": actions_taken,
                "terminated": terminated,
                "termination_reason": termination_reason,
            },
        )
        r.raise_for_status()
        return r.json()


def run_async(coro):
    """Run async coroutine from sync context (rollout_func)."""
    return asyncio.run(coro)
