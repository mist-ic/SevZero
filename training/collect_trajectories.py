"""
Collect expert trajectories for SevZero SFT (Round 2).

Loads API keys from api.env and hg.env (gitignored). Does not log secrets.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

# Repo root: parent of training/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import (  # noqa: E402
    build_observation_prompt,
    parse_action,
)
from inference import SYSTEM_PROMPT as _BASE_SYSTEM  # noqa: E402

load_dotenv(REPO_ROOT / "api.env")
load_dotenv(REPO_ROOT / "hg.env")

# ---------------------------------------------------------------------------
# Config matrix (must match spec)
# ---------------------------------------------------------------------------

GEMINI_SEEDS = [
    42, 123, 7, 11, 23, 31, 47, 59, 67, 71, 83, 89, 97, 101, 109, 113, 127, 131, 137, 149
]
GPT_SEEDS = [
    42, 123, 7, 13, 17, 19, 29, 37, 41, 43, 53, 61, 73, 79, 83, 89, 97, 101, 103, 107
]
GROK_EXTRA_SEEDS = [13, 17, 19, 29, 37, 41, 43, 53, 61, 73]

# Combined pool for grok / kimi / deepseek (any from grok list + full Gemini list)
GROK_KIMI_POOL: List[int] = sorted(set(GEMINI_SEEDS) | set(GROK_EXTRA_SEEDS))

MODEL_GEMINI = "gemini-3.1-pro-preview"
MODEL_GPT = "gpt-5.4-pro"
MODEL_GROK = "grok-4.20-reasoning"
MODEL_KIMI = "kimi-k2.6"
MODEL_DEEPSEEK = "DeepSeek-V3.2"
ALL_CANON = {MODEL_GEMINI, MODEL_GPT, MODEL_GROK, MODEL_KIMI, MODEL_DEEPSEEK}


def _split_seeds(
    pool: List[int], counts: Tuple[int, int, int], offset: int
) -> List[Tuple[str, int]]:
    """Return list of (task_id, seed) in order easy, medium, hard."""
    c_e, c_m, c_h = counts
    n = len(pool)
    if n == 0:
        return []
    o = [pool[(i + offset) % n] for i in range(n)]
    out: List[Tuple[str, int]] = []
    i = 0
    for _ in range(c_e):
        out.append(("easy", o[i % len(o)]))
        i += 1
    for _ in range(c_m):
        out.append(("medium", o[i % len(o)]))
        i += 1
    for _ in range(c_h):
        out.append(("hard", o[i % len(o)]))
        i += 1
    return out


def plan_gemini(c_e: int, c_m: int, c_h: int) -> List[Tuple[str, str, int]]:
    return [
        (MODEL_GEMINI, t, s)
        for t, s in _split_seeds(GEMINI_SEEDS, (c_e, c_m, c_h), offset=0)
    ]


def plan_gpt(c_e: int, c_m: int, c_h: int) -> List[Tuple[str, str, int]]:
    return [
        (MODEL_GPT, t, s)
        for t, s in _split_seeds(GPT_SEEDS, (c_e, c_m, c_h), offset=0)
    ]


def plan_grok(c_e: int, c_m: int, c_h: int) -> List[Tuple[str, str, int]]:
    return [
        (MODEL_GROK, t, s)
        for t, s in _split_seeds(GROK_KIMI_POOL, (c_e, c_m, c_h), offset=0)
    ]


def plan_kimi(c_e: int, c_m: int, c_h: int) -> List[Tuple[str, str, int]]:
    return [
        (MODEL_KIMI, t, s)
        for t, s in _split_seeds(GROK_KIMI_POOL, (c_e, c_m, c_h), offset=7)
    ]


def plan_deepseek(c_e: int, c_m: int, c_h: int) -> List[Tuple[str, str, int]]:
    return [
        (MODEL_DEEPSEEK, t, s)
        for t, s in _split_seeds(GROK_KIMI_POOL, (c_e, c_m, c_h), offset=3)
    ]


def full_plan(c_e: int, c_m: int, c_h: int) -> List[Tuple[str, str, int]]:
    return (
        plan_gemini(c_e, c_m, c_h)
        + plan_gpt(c_e, c_m, c_h)
        + plan_grok(c_e, c_m, c_h)
        + plan_kimi(c_e, c_m, c_h)
        + plan_deepseek(c_e, c_m, c_h)
    )


# Rough USD cost tracking (tunable; for guardrail only)
@dataclass
class CostTracker:
    usd: float = 0.0
    budget: float = 5.0
    by_model: Dict[str, float] = field(default_factory=dict)
    per_model_max: float = 2.0

    def add(self, model: str, usd: float) -> None:
        self.usd += usd
        self.by_model[model] = self.by_model.get(model, 0.0) + usd
        m = self.by_model[model]
        cap = self.per_model_max
        if m > cap:
            raise RuntimeError(
                f"Model {model} exceeded ${cap:.2f} in estimated spend (${m:.2f}); stopping per cap."
            )
        if self.usd > self.budget:
            raise RuntimeError(
                f"Total estimated API spend ${self.usd:.2f} exceeded budget ${self.budget:.2f}."
            )


def _estimate_openai_style_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    # Conservative blended rate per 1K tokens (USD) — for guardrails only
    if "gemini" in model:
        p, c = 0.00125, 0.01
    elif "gpt" in model.lower() or "5.4" in model:
        p, c = 0.0025, 0.01
    else:
        p, c = 0.001, 0.006
    return (prompt_tokens * p + completion_tokens * c) / 1000.0


# ---------------------------------------------------------------------------
# Pydantic for Gemini structured action JSON
# ---------------------------------------------------------------------------


class AgentActionOut(BaseModel):
    action_type: str
    params: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Azure deployment self-heal
# ---------------------------------------------------------------------------


def _is_not_found(err: str) -> bool:
    s = (err or "").lower()
    return "deploymentnotfound" in s or "deployment" in s and "not found" in s


def list_azure_openai_deployments() -> List[str]:
    key = os.environ.get("AZURE_API_KEY", "")
    ep = (os.environ.get("AZURE_OPENAI_ENDPOINT", "") or "").rstrip("/")
    ver = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")
    if not key or not ep:
        return []
    url = f"{ep}/openai/deployments?api-version={ver}"
    try:
        r = httpx.get(url, headers={"api-key": key}, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        return [d.get("id", "") for d in data.get("value", []) if d.get("id")]
    except Exception:
        return []


def list_foundry_deployments() -> List[str]:
    """
    Best-effort: project endpoint may expose deployments; schema varies.
    """
    fe = (os.environ.get("AZURE_FOUNDRY_PROJECT_ENDPOINT", "") or "").rstrip("/")
    key = os.environ.get("AZURE_API_KEY", "")
    if not fe or not key:
        return []
    for suffix in ("/deployments", "/openai/models"):
        try:
            url = f"{fe}{suffix}"
            r = httpx.get(
                url, headers={"api-key": key}, params={"api-version": "2024-12-01-preview"}, timeout=30.0
            )
            if r.status_code != 200:
                continue
            data = r.json()
            if isinstance(data, list):
                return [str(x.get("id", x)) for x in data if isinstance(x, dict)]
            if "value" in data:
                return [d.get("id", "") for d in data.get("value", []) if d.get("id")]
        except Exception:
            continue
    return []


def pick_closest(name: str, options: List[str]) -> str:
    if not options:
        return name
    if name in options:
        return name
    ranked = difflib.get_close_matches(name, options, n=1, cutoff=0.2)
    if ranked:
        return ranked[0]
    return options[0]


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.gemini_client: Any = None
        self.azure_openai: Any = None
        self.azure_inf: Any = None
        if model == MODEL_GEMINI:
            from google import genai

            key = os.environ.get("GEMINI_API_KEY", "")
            if not key:
                raise ValueError("GEMINI_API_KEY missing for Gemini collection.")
            self.gemini_client = genai.Client(api_key=key)
        elif model == MODEL_GPT:
            if not all(
                os.environ.get(x)
                for x in (
                    "AZURE_API_KEY",
                    "AZURE_OPENAI_ENDPOINT",
                    "AZURE_API_VERSION",
                )
            ):
                raise ValueError("AZURE_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_API_VERSION required for gpt-5.4-pro.")
            self.azure_openai = AzureOpenAI(
                api_key=os.environ["AZURE_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.environ["AZURE_API_VERSION"],
            )
        else:
            if not all(os.environ.get(x) for x in ("AZURE_API_KEY", "AZURE_AI_INFERENCE_ENDPOINT")):
                raise ValueError("AZURE_API_KEY and AZURE_AI_INFERENCE_ENDPOINT required for inference models.")
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential

            self.azure_inf = ChatCompletionsClient(
                endpoint=os.environ["AZURE_AI_INFERENCE_ENDPOINT"],
                credential=AzureKeyCredential(os.environ["AZURE_API_KEY"]),
            )

    def _deployment_name(self) -> str:
        m = {MODEL_GPT: "AZURE_MODEL_GPT", MODEL_GROK: "AZURE_MODEL_GROK", MODEL_KIMI: "AZURE_MODEL_KIMI", MODEL_DEEPSEEK: "AZURE_MODEL_DEEPSEEK"}.get(self.model)
        if m:
            v = os.environ.get(m, "").strip()
            if v:
                return v
        return self.model

    def call(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, int, int]:
        """Return (raw_text, prompt_tokens, completion_tokens)."""
        p_tok, c_tok = 0, 0
        if self.gemini_client is not None:
            return self._call_gemini(messages, p_tok, c_tok)
        if self.azure_openai is not None:
            return self._call_azure_openai(messages, p_tok, c_tok)
        if self.azure_inf is not None:
            return self._call_azure_inference(messages, p_tok, c_tok)
        raise RuntimeError("No backend initialised")

    def _call_gemini(
        self, messages: List[Dict[str, str]], p0: int, c0: int
    ) -> Tuple[str, int, int]:
        from google.genai import types

        if not messages:
            return '{"action_type": "noop", "params": {}}', 0, 0
        system = messages[0]["content"] if messages[0]["role"] == "system" else _BASE_SYSTEM
        rest = messages[1:] if messages[0]["role"] == "system" else messages
        name = os.environ.get("GEMINI_MODEL_PRO", MODEL_GEMINI)
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=AgentActionOut,
            temperature=0.0,
            max_output_tokens=512,
        )
        # Build contents: alternating user / model for few-shot tail
        contents: List[Any] = []
        for m in rest:
            if m["role"] == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=m["content"])])
                )
            else:
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=m["content"])],
                    )
                )
        for attempt in range(3):
            try:
                resp = self.gemini_client.models.generate_content(
                    model=name, contents=contents, config=config
                )
                text = (resp.text or "").strip() if hasattr(resp, "text") else ""
                u = getattr(resp, "usage_metadata", None) or getattr(resp, "usage", None)
                pt = int(getattr(u, "prompt_token_count", None) or getattr(u, "prompt_tokens", 0) or 0) if u else 0
                ct = int(getattr(u, "candidates_token_count", None) or getattr(u, "completion_tokens", 0) or 0) if u else 0
                if not text and hasattr(resp, "candidates") and resp.candidates:
                    p0x = resp.candidates[0].content.parts[0] if resp.candidates[0].content.parts else None
                    text = getattr(p0x, "text", "") or ""
                return text, pt, ct
            except Exception:
                if attempt < 2:
                    time.sleep(1.0 + attempt)
                else:
                    return '{"action_type": "noop", "params": {}}', p0, c0

    def _call_azure_openai(
        self, messages: List[Dict[str, str]], p0: int, c0: int
    ) -> Tuple[str, int, int]:
        dep = self._deployment_name()
        for attempt in range(3):
            try:
                comp = self.azure_openai.chat.completions.create(
                    model=dep,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.0,
                    max_tokens=512,
                    timeout=90.0,
                )
                text = (comp.choices[0].message.content or "").strip()
                u = comp.usage
                pt = u.prompt_tokens if u else 0
                ct = u.completion_tokens if u else 0
                return text, pt, ct
            except Exception as e:
                err = str(e)
                if _is_not_found(err):
                    names = list_azure_openai_deployments()
                    if names:
                        dep = pick_closest(dep, names)
                if attempt == 2:
                    return '{"action_type": "noop", "params": {}}', p0, c0
                time.sleep(1.0 + attempt)
        return '{"action_type": "noop", "params": {}}', p0, c0

    def _call_azure_inference(
        self, messages: List[Dict[str, str]], p0: int, c0: int
    ) -> Tuple[str, int, int]:
        dep = self._deployment_name()
        for attempt in range(3):
            try:
                resp = self.azure_inf.complete(
                    model=dep,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0.0,
                    max_tokens=512,
                )
                ch = resp.choices[0].message
                text = (ch.content or "").strip() if ch else ""
                u = getattr(resp, "usage", None)
                pt = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
                ct = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
                return text, pt, ct
            except Exception as e:
                err = str(e)
                if _is_not_found(err) or "404" in err or "not found" in err.lower():
                    names = [n for n in list_foundry_deployments() + list_azure_openai_deployments() if n]
                    if names:
                        dep = pick_closest(dep, names)
                if attempt == 2:
                    return '{"action_type": "noop", "params": {}}', p0, c0
                time.sleep(1.0 + attempt)
        return '{"action_type": "noop", "params": {}}', p0, c0


# ---------------------------------------------------------------------------
# Episode (mirrors inference.run_episode; logs full trace)
# ---------------------------------------------------------------------------


def _memory_block(tried_actions: Dict[str, List[str]], resolved_services: List[str]) -> str:
    if not tried_actions and not resolved_services:
        return ""
    lines = ["## Episode Memory (do not repeat failed approaches)"]
    if resolved_services:
        lines.append(f"  Resolved: {', '.join(resolved_services)}")
    for act, targets in tried_actions.items():
        lines.append(f"  {act}: {'; '.join(targets)}")
    return "\n".join(lines)


def run_one_episode(
    llm: LLMClient,
    model_id: str,
    base: str,
    task_id: str,
    seed: int,
    cost: CostTracker,
) -> Dict[str, Any]:
    grade: Dict[str, Any] = {}
    with httpx.Client(timeout=60.0) as http:
        r = http.post(
            f"{base}/reset", json={"seed": seed, "task_id": task_id}
        )
        r.raise_for_status()
        resp_data = r.json()
        obs: Dict[str, Any] = dict(resp_data.get("observation", resp_data))
        max_steps = int(obs.get("max_steps", 10))
        done = bool(resp_data.get("done", False))
        conv: List[Dict[str, Any]] = []
        tried: Dict[str, List[str]] = {}
        resolved: List[str] = []
        steps_out: List[Dict[str, Any]] = []
        for step_num in range(1, max_steps + 1):
            if done:
                break
            obs_pre = copy.deepcopy(obs)
            user_msg = build_observation_prompt(obs_pre)
            conv.append({"role": "user", "content": user_msg})
            trimmed = conv[-6:]
            memory = _memory_block(tried, resolved)
            system_content = _BASE_SYSTEM + ("\n\n" + memory if memory else "")
            messages: List[Dict[str, str]] = (
                [{"role": "system", "content": system_content}] + trimmed
            )
            raw, pt, ct = llm.call(messages)
            cost.add(
                model_id, _estimate_openai_style_cost(model_id, pt, ct)
            )
            try:
                action = parse_action(raw)
            except Exception:
                action = {"action_type": "noop", "params": {}}
            if isinstance(action, dict) and "action_type" in action and model_id == MODEL_GEMINI:
                try:
                    a2 = (
                        json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
                        if "{" in raw
                        else None
                    )
                    if a2 and isinstance(a2, dict) and "action_type" in a2:
                        action = a2
                except Exception:
                    pass
            act_params = action.get("params", {}) or {}
            if "replicas" in act_params:
                try:
                    act_params["replicas"] = int(act_params["replicas"])
                except (ValueError, TypeError):
                    act_params["replicas"] = 2
            act_type = action.get("action_type", "noop")
            target = act_params.get("service_id") or act_params.get("cache_name") or act_params.get("from_region") or ""
            step_resp = http.post(
                f"{base}/step",
                json={"action": {"action_type": act_type, "params": act_params}},
            )
            sdata = step_resp.json() if step_resp.status_code == 200 else {}
            obs = dict(sdata.get("observation", sdata))
            done = bool(sdata.get("done", False))
            reward = float(
                obs.get("reward", sdata.get("reward", 0.0)) or 0.0
            )
            conv.append({"role": "assistant", "content": raw})
            if act_type not in (
                "inspect_logs",
                "inspect_metrics",
                "inspect_traces",
                "noop",
            ) and target:
                new_slo = obs.get("global_slo_score", 0.0)
                for svc in obs.get("services", []):
                    if svc.get("id") == target and svc.get("status") == "healthy":
                        if target not in resolved:
                            resolved.append(target)
                entry = f"{target} (slo={new_slo:.0%})"
                tried.setdefault(str(act_type), [])
                if entry not in tried[str(act_type)]:
                    tried[str(act_type)].append(entry)
            obs_ser = json.loads(
                json.dumps(
                    {k: v for k, v in obs_pre.items() if k != "reward"},
                    default=str,
                )
            )
            steps_out.append(
                {
                    "step": step_num,
                    "observation": obs_ser,
                    "prompt": user_msg,
                    "messages": messages,
                    "completion": raw,
                    "action": action,
                    "reward": reward,
                    "info": {k: v for k, v in sdata.items() if k not in ("observation",)},
                }
            )
        try:
            final_state = http.get(f"{base}/state").json()
        except Exception:
            final_state = {}
        try:
            grade = http.post(
                f"{base}/grader",
                json={
                    "final_slo_score": final_state.get("global_slo_score", 0.0),
                    "steps_taken": final_state.get("step_count", 0),
                    "max_steps": max_steps,
                    "actions_taken": obs.get("actions_taken", []),
                    "terminated": final_state.get("terminated", True),
                    "termination_reason": final_state.get("termination_reason"),
                },
            ).json()
        except Exception:
            grade = {}
    score = float(grade.get("score", 0.0) or 0.0)
    return {
        "model": model_id,
        "task_id": task_id,
        "seed": seed,
        "steps": steps_out,
        "grader": grade,
        "final_score": score,
        "max_steps": max_steps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _raw_path(model: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", model)
    d = REPO_ROOT / "training" / "data" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{safe}.jsonl"


def _wait_health(base: str, timeout: float = 45.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{base}/health", timeout=3.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.0)
    print(f"[collect] health check timeout for {base} — continuing", flush=True)


def start_server(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    pp = str(REPO_ROOT)
    env["PYTHONPATH"] = pp if not env.get("PYTHONPATH") else pp + os.pathsep + env["PYTHONPATH"]
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def parse_models(s: str) -> List[str]:
    return [m.strip() for m in s.split(",") if m.strip()]


def _plan_for_model(
    model: str, c_e: int, c_m: int, c_h: int
) -> List[Tuple[str, str, int]]:
    p = {
        MODEL_GEMINI: plan_gemini,
        MODEL_GPT: plan_gpt,
        MODEL_GROK: plan_grok,
        MODEL_KIMI: plan_kimi,
        MODEL_DEEPSEEK: plan_deepseek,
    }
    fn = p.get(model)
    if not fn:
        return []
    return fn(c_e, c_m, c_h)


def sanity_runs() -> List[Tuple[str, str, int]]:
    return [
        (MODEL_GEMINI, "easy", 42),
        (MODEL_GPT, "easy", 42),
        (MODEL_GROK, "easy", 13),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        type=str,
        default=",".join(sorted(ALL_CANON)),
        help="Comma-separated model ids (default: all)",
    )
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-start-server", action="store_true")
    ap.add_argument("--sanity-only", action="store_true", help="Run only 3 smoke episodes (gemini, gpt, grok easy).")
    ap.add_argument("--no-sanity", action="store_true", help="Skip pre-flight sanity runs.")
    ap.add_argument(
        "--budget-usd",
        type=float,
        default=5.0,
        help="Total estimated-spend cap (heuristic) across all models.",
    )
    ap.add_argument(
        "--per-model-budget-usd",
        type=float,
        default=0.0,
        help="Per-model cap (0 = auto: max(2, budget/num selected models)).",
    )
    ap.add_argument(
        "--episodes-easy",
        type=int,
        default=15,
        help="Number of easy-task episodes per model (default 15, Wave 1.5).",
    )
    ap.add_argument(
        "--episodes-medium",
        type=int,
        default=15,
        help="Number of medium-task episodes per model (default 15).",
    )
    ap.add_argument(
        "--episodes-hard",
        type=int,
        default=20,
        help="Number of hard-task episodes per model (default 20).",
    )
    args = ap.parse_args()
    want = set(parse_models(args.models))
    bad = want - ALL_CANON
    if bad:
        raise SystemExit(f"Unknown model(s): {bad}. Valid: {sorted(ALL_CANON)}")

    c_e, c_m, c_h = args.episodes_easy, args.episodes_medium, args.episodes_hard
    if min(c_e, c_m, c_h) < 0:
        raise SystemExit("--episodes-* must be non-negative.")
    if c_e + c_m + c_h == 0:
        raise SystemExit("At least one of --episodes-easy/medium/hard must be > 0.")

    _ = full_plan(c_e, c_m, c_h)  # exercise planner (raises if misconfigured)

    # Required keys
    for m in want:
        if m == MODEL_GEMINI and not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit("GEMINI_API_KEY missing (needed for gemini-3.1-pro-preview).")
        if m == MODEL_GPT and not all(
            os.environ.get(x) for x in ("AZURE_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_API_VERSION")
        ):
            raise SystemExit("Azure OpenAI env vars missing for gpt-5.4-pro.")
        if m in (MODEL_GROK, MODEL_KIMI, MODEL_DEEPSEEK) and not all(
            os.environ.get(x) for x in ("AZURE_API_KEY", "AZURE_AI_INFERENCE_ENDPOINT")
        ):
            raise SystemExit("Azure inference env missing for " + m)

    proc: Optional[subprocess.Popen] = None
    if not args.no_start_server:
        proc = start_server(args.port)
    base = f"http://127.0.0.1:{args.port}"
    _wait_health(base)
    n_m = max(1, len(want))
    per_cap = args.per_model_budget_usd
    if per_cap <= 0.0:
        per_cap = max(2.0, args.budget_usd / n_m)
    cost = CostTracker(budget=args.budget_usd, per_model_max=per_cap)
    # LLM clients (lazy)
    _clients: Dict[str, LLMClient] = {}
    def get_llm(mid: str) -> LLMClient:
        if mid not in _clients:
            _clients[mid] = LLMClient(mid)
        return _clients[mid]

    try:
        already: Set[Tuple[str, str, int]] = set()
        if args.sanity_only:
            final_list = [r for r in sanity_runs() if r[0] in want]
        else:
            if not args.no_sanity:
                for mid, task_id, seed in (r for r in sanity_runs() if r[0] in want):
                    print(f"[sanity] {mid} {task_id} seed={seed}", flush=True)
                    llm = get_llm(mid)
                    _ = run_one_episode(llm, mid, base, task_id, seed, cost)
                    already.add((mid, task_id, seed))
                print("[sanity] pre-flight ok", flush=True)
            final_list = []
            for m in want:
                for x in _plan_for_model(m, c_e, c_m, c_h):
                    if x in already:
                        continue
                    final_list.append(x)
        n_done = 0
        for mid, task_id, seed in final_list:
            print(f"[episode] {mid} {task_id} seed={seed}", flush=True)
            try:
                llm = get_llm(mid)
                ep = run_one_episode(llm, mid, base, task_id, seed, cost)
            except RuntimeError as e:
                print(f"[collect] Stopped: {e}", flush=True)
                break
            p = _raw_path(mid)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            n_done += 1
            print(
                f"  -> score={ep.get('final_score', 0):.4f} lines->{p.name} (total est ${cost.usd:.2f})",
                flush=True,
            )
        print(f"Done. Episodes written: {n_done}. Estimated spend: ${cost.usd:.2f}", flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
