"""
server/curriculum.py — Heuristic (Tier1) and optional LLM (Tier2) scenario overrides.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional

from server.failures import FailureType

LOG = logging.getLogger(__name__)
_tier2_once: bool = False

try:
    from dotenv import load_dotenv

    for _path in ("api.env", "hg.env"):
        load_dotenv(_path, override=False)
except ImportError:
    pass


def _llm_tier2_once(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Optional Gemini call. Returns None on any failure; logs once if missing key."""
    global _tier2_once
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        if not _tier2_once:
            LOG.info("curriculum Tier2: GEMINI_API_KEY not set, using Tier1")
            _tier2_once = True
        return None
    try:
        from google import genai  # type: ignore[import-not-found]
    except ImportError:
        if not _tier2_once:
            LOG.info("curriculum Tier2: google.genai not available, using Tier1")
            _tier2_once = True
        return None
    model_id = os.environ.get("GEMINI_MODEL_FLASH", "gemini-3-flash-preview")
    try:
        client = genai.Client(api_key=key)
        r = client.models.generate_content(
            model=model_id,
            contents=(
                "Return only JSON: failure_type_weights (map of failure type id string to "
                f"weight), min_failures (int), max_steps (int), rationale. Input: {json.dumps(summary)[:6000]}"
            ),
        )
        if not (r and getattr(r, "text", None)):
            return None
        data = json.loads(r.text)  # type: ignore[union-attr]
        w = data.get("failure_type_weights", {})
        if not isinstance(w, dict):
            return None
        return {
            "failure_type_weights": {str(a): float(b) for a, b in w.items()},
            "num_failures": int(data.get("min_failures", 1)),
            "max_steps": int(data.get("max_steps", 20)),
        }
    except Exception as e:  # noqa: BLE001
        if not _tier2_once:
            LOG.info("curriculum Tier2: API error, Tier1: %s", e)
            _tier2_once = True
        return None


class Curriculum:
    def __init__(self) -> None:
        # Last 10 episodes: failure type ids, whether resolved, grader / proxy score
        self._episodes: Deque[Dict[str, Any]] = deque(
            maxlen=10,
        )
        self._episode_idx: int = 0

    def on_episode_end(
        self,
        mean_score: float,
        resolved: bool,
        failure_types: List[str],
    ) -> None:
        self._episodes.append(
            {
                "failure_types": list(failure_types) or [FailureType.CRASH.value],
                "resolved": bool(resolved),
                "mean_score": float(mean_score),
            },
        )
        self._episode_idx += 1

    def next_scenario_overrides(self) -> Dict[str, Any]:
        n = self._episode_idx
        out: Dict[str, Any] = {}
        if self._episodes:
            by_type: Dict[str, int] = {}
            success_by: Dict[str, int] = {}
            for ep in self._episodes:
                for ft in ep["failure_types"]:
                    by_type[ft] = by_type.get(ft, 0) + 1
                    if ep["resolved"]:
                        success_by[ft] = success_by.get(ft, 0) + 1
            success_rate: Dict[str, float] = {}
            for t, c in by_type.items():
                success_rate[t] = success_by.get(t, 0) / max(1, c)
            if success_rate:
                worst = sorted(
                    success_rate.items(), key=lambda x: (x[1], -by_type[x[0]]),
                )
                w1, w2 = worst[0][0], (
                    worst[1][0] if len(worst) > 1 else worst[0][0]
                )
                wmap: Dict[str, float] = {f.value: 1.0 for f in FailureType}
                wmap[w1] = wmap.get(w1, 1.0) * 3.0
                wmap[w2] = wmap.get(w2, 1.0) * 2.0
                out["failure_type_weights"] = wmap
            means = [float(ep["mean_score"]) for ep in self._episodes]
            if means and (sum(means) / len(means)) > 0.85:
                out["bump_num_failures"] = 1
                out["max_steps_offset"] = -2
        if n > 0 and n % 10 == 0:
            t2 = _llm_tier2_once({"episodes": list(self._episodes)})
            if t2:
                return {**out, **t2}
        return out
