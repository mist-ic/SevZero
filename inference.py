"""
Inference Script — SevZero Baseline Agent
==========================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

Recommended setup (free, no credit card):
    API_BASE_URL=https://api.groq.com/openai/v1
    MODEL_NAME=llama-3.3-70b-versatile
    HF_TOKEN=<your_groq_api_key>   # Free at console.groq.com
"""

import json
import os
import time
import textwrap
from typing import Any, Dict, List

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# Fallback providers tried in order if the primary hits rate limits or errors.
# Each uses the same HF_TOKEN env var as the API key — all are OpenAI-compatible.
_FALLBACK_PROVIDERS = [
    # Tier 1 fallback: same Groq key, lighter model (14,400 RPD free)
    {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "api_key": API_KEY,
    },
    # Tier 2 fallback: HuggingFace Inference Router
    {
        "base_url": "https://router.huggingface.co/v1",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "api_key": os.getenv("HF_INFERENCE_TOKEN") or API_KEY,
    },
]

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Site Reliability Engineer (SRE) responding to a production incident.
    You are managing a microservice cluster experiencing failures.

    Your goal: restore all services to healthy SLO compliance as efficiently as possible.

    Strategy:
    1. First, inspect logs of services showing the highest error rates or critical alerts
    2. Diagnose the root cause from log patterns:
       - OOMKilled/CrashLoopBackOff -> restart_service
       - NullPointerException/TypeError + recent deploy -> rollback_service
       - "password authentication failed"/"config not found" -> tune_config with the broken key
         (the logs will show: "Configuration diagnostic: key '<KEY>' has invalid value")
       - Thread pool exhaustion/timeout from downstream -> fix the downstream dependency first
       - Memory climbing linearly -> restart_service (resource leak)
       - HikariPool exhaustion/slow queries -> scale_service or restart_service on the DB
       - CLUSTERDOWN/cache miss -> clear_cache
       - DNS/network errors -> rebalance_traffic (if multi-region)
    3. Apply the correct remediation action
    4. Verify recovery with inspect_logs or inspect_metrics

    Respond with EXACTLY one JSON object — no explanation, no markdown, just raw JSON:
    {"action_type": "...", "params": {...}}

    Param rules (STRICT — single service only, never a list):
    - inspect_logs / inspect_metrics / inspect_traces / restart_service / rollback_service / scale_service:
        {"action_type": "X", "params": {"service_id": "order-service"}}
    - tune_config:
        {"action_type": "tune_config", "params": {"service_id": "order-service", "key": "api_endpoint", "value": "correct"}}
    - clear_cache:
        {"action_type": "clear_cache", "params": {"cache_name": "redis-cache"}}
    - noop:
        {"action_type": "noop", "params": {}}
""")


def _call_llm(
    messages: List[Dict[str, Any]],
    primary_client: OpenAI,
    primary_model: str,
) -> str:
    """Call LLM with automatic fallback on rate limit or error."""
    providers = [{"client": primary_client, "model": primary_model}] + [
        {
            "client": OpenAI(base_url=p["base_url"], api_key=p["api_key"]),
            "model": p["model"],
        }
        for p in _FALLBACK_PROVIDERS
    ]

    last_err = None
    for i, provider in enumerate(providers):
        try:
            completion = provider["client"].chat.completions.create(
                model=provider["model"],
                messages=messages,
                temperature=0.2,
                max_tokens=200,
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            is_rate_limit = any(x in str(e).lower() for x in ("429", "rate_limit", "quota", "credits", "402"))
            label = "fallback" if i > 0 else "primary"
            print(f"  [{label} {provider['model']}] error: {e}")
            if is_rate_limit and i < len(providers) - 1:
                time.sleep(3)
                continue
            if i < len(providers) - 1:
                continue
    print(f"  All providers failed. Last error: {last_err}")
    return '{"action_type": "noop", "params": {}}'


def build_observation_prompt(obs: Dict[str, Any]) -> str:
    """Build a concise prompt from the observation."""
    parts = [f"## Incident Status\n{obs.get('observation_summary', 'N/A')}"]

    # Alerts (most important)
    alerts = obs.get("alerts", [])
    if alerts:
        alert_lines = [f"  [{a['severity'].upper()}] {a['message']}" for a in alerts[:10]]
        parts.append("## Active Alerts\n" + "\n".join(alert_lines))

    # Service states (condensed — degraded only)
    services = obs.get("services", [])
    degraded = [s for s in services if s.get("status") in ("degraded", "critical", "down")]
    if degraded:
        svc_lines = [
            f"  {s['id']} [{s['status']}]: error={s['error_rate']:.1%}, "
            f"p99={s['latency_p99_ms']:.0f}ms, cpu={s['cpu_pct']:.0f}%, "
            f"mem={s['memory_pct']:.0f}%, pool={s['connection_pool_usage_pct']:.0f}%"
            for s in degraded
        ]
        parts.append("## Degraded Services\n" + "\n".join(svc_lines))

    # Recent deploys
    deploys = obs.get("recent_deploys", [])
    if deploys:
        dep_lines = [
            f"  {d['service']} -> {d['version']} ({d['ticks_ago']} ticks ago)"
            for d in deploys
        ]
        parts.append("## Recent Deploys\n" + "\n".join(dep_lines))

    # Actions taken
    actions = obs.get("actions_taken", [])
    if actions:
        act_lines = [
            f"  tick {a['tick']}: {a['action']}({a.get('target', '')}) -> {'OK' if a['success'] else 'FAIL'}"
            for a in actions[-5:]
        ]
        parts.append("## Recent Actions\n" + "\n".join(act_lines))

    # Logs (if available from inspect)
    logs = obs.get("logs")
    if logs:
        parts.append(f"## Logs\n{logs}")

    # Traces (if available)
    traces = obs.get("traces")
    if traces:
        error_spans = [s for s in traces.get("spans", []) if s.get("status") == "ERROR"]
        if error_spans:
            trace_lines = [
                f"  {s['service']}: {s.get('tags', {}).get('error.message', 'ERROR')} ({s['duration_ms']}ms)"
                for s in error_spans[:5]
            ]
            parts.append("## Trace Errors\n" + "\n".join(trace_lines))

    # Legal actions
    legal = obs.get("legal_actions", [])
    if legal:
        legal_strs = [f"  {la['action_type']}: targets={la['valid_targets'][:5]}" for la in legal]
        parts.append("## Available Actions\n" + "\n".join(legal_strs))

    return "\n\n".join(parts)


def parse_action(response_text: str) -> Dict[str, Any]:
    """Parse the model's JSON response into an action dict."""
    text = response_text.strip()

    # Strip markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Extract JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return {"action_type": "noop", "params": {}}


def run_episode(
    client: OpenAI,
    env_url: str,
    task_id: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run one episode using the OpenEnv HTTP API."""
    import httpx

    base = env_url.rstrip("/")

    # Reset
    reset_resp = httpx.post(
        f"{base}/reset",
        json={"seed": seed, "task_id": task_id},
        timeout=30.0,
    )
    resp_data = reset_resp.json()
    obs = resp_data.get("observation", resp_data)

    max_steps = obs.get("max_steps", 10)
    total_reward = 0.0
    done = resp_data.get("done", False)

    # Rolling conversation: system prompt + last 6 messages (3 turns).
    # Prevents context explosion on hard tasks (50 steps x ~800 tokens/step).
    conversation_history: List[Dict[str, Any]] = []

    for step_num in range(max_steps):
        if done:
            break

        user_msg = build_observation_prompt(obs)
        conversation_history.append({"role": "user", "content": user_msg})

        # Keep only last 6 messages (3 user+assistant turns) to bound context size
        trimmed = conversation_history[-6:]
        messages_to_send = [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed

        response_text = _call_llm(messages_to_send, client, MODEL_NAME)
        action = parse_action(response_text)
        conversation_history.append({"role": "assistant", "content": response_text})

        print(f"  Step {step_num + 1}: {action.get('action_type', 'noop')}({action.get('params', {})})")

        # Step the environment
        params = action.get("params", {})
        # Coerce replicas to int if model sends a string
        if "replicas" in params:
            try:
                params["replicas"] = int(params["replicas"])
            except (ValueError, TypeError):
                params["replicas"] = 2

        step_resp = httpx.post(
            f"{base}/step",
            json={"action": {
                "action_type": action.get("action_type", "noop"),
                "params": params,
            }},
            timeout=30.0,
        )
        try:
            resp_data = step_resp.json()
        except Exception:
            # Empty or non-JSON response (server error) — treat as noop
            resp_data = {}
        obs = resp_data.get("observation", resp_data)
        done = resp_data.get("done", False)
        reward = obs.get("reward") or resp_data.get("reward") or 0.0
        total_reward += reward if reward else 0.0

    # Final state + grade
    final_state = httpx.get(f"{base}/state", timeout=10.0).json()
    grade = httpx.post(
        f"{base}/grader",
        json={
            "final_slo_score": final_state.get("global_slo_score", 0.0),
            "steps_taken": final_state.get("step_count", 0),
            "max_steps": max_steps,
            "actions_taken": obs.get("actions_taken", []),
            "terminated": final_state.get("terminated", True),
            "termination_reason": final_state.get("termination_reason"),
        },
        timeout=10.0,
    ).json()

    return {
        "task_id": task_id,
        "seed": seed,
        "total_reward": total_reward,
        "score": grade.get("score", 0.0),
        "slo_recovery": grade.get("slo_recovery", 0.0),
        "action_efficiency": grade.get("action_efficiency", 0.0),
        "time_efficiency": grade.get("time_efficiency", 0.0),
        "steps_taken": final_state.get("step_count", 0),
        "termination_reason": final_state.get("termination_reason"),
    }


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_url = os.getenv("ENV_URL", "http://localhost:7860")

    tasks = ["easy", "medium", "hard"]
    seeds = [42, 123, 7]

    print("=" * 60)
    print("SevZero Baseline Inference")
    print("=" * 60)
    print(f"Model:       {MODEL_NAME}")
    print(f"API:         {API_BASE_URL}")
    print(f"Environment: {env_url}")
    print()

    results = []
    for task_id, seed in zip(tasks, seeds):
        print(f"--- Task: {task_id} (seed={seed}) ---")
        result = run_episode(client, env_url, task_id, seed)
        results.append(result)
        print(
            f"  Score: {result['score']:.4f} | SLO: {result['slo_recovery']:.4f} | "
            f"AE: {result['action_efficiency']:.4f} | TE: {result['time_efficiency']:.4f} | "
            f"Steps: {result['steps_taken']} | Outcome: {result['termination_reason']}"
        )
        print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r['task_id']:8s} score={r['score']:.4f}  slo={r['slo_recovery']:.4f}  steps={r['steps_taken']}")
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n  Average score: {avg_score:.4f}")


if __name__ == "__main__":
    main()
