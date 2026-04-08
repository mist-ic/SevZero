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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
ENV_NAME = "sevzero"

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Site Reliability Engineer (SRE) responding to a production incident.
    You are managing a microservice cluster experiencing failures.

    Your goal: restore all services to healthy SLO compliance as efficiently as possible.

    Strategy:
    1. First, inspect logs of services showing the highest error rates or critical alerts
    2. Diagnose the root cause from log patterns:
       - OOMKilled/CrashLoopBackOff -> restart_service
       - NullPointerException/TypeError + recent deploy -> rollback_service
       - "Configuration diagnostic: key '<KEY>'" -> tune_config with that exact key, value='correct'
       - Thread pool exhaustion on THIS service -> restart_service or scale_service on THIS service
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
    - rebalance_traffic:
        {"action_type": "rebalance_traffic", "params": {"from_region": "us-east-1", "to_region": "us-west-2"}}
    - noop:
        {"action_type": "noop", "params": {}}
""")

# ---------------------------------------------------------------------------
# Structured logging — required by hackathon evaluator
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Any = None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} "
        f"done={str(done).lower()} error={error}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------

_token_usage: Dict[str, int] = {"prompt": 0, "completion": 0}


def _track_usage(completion: Any) -> None:
    usage = getattr(completion, "usage", None)
    if not usage:
        return
    _token_usage["prompt"] += getattr(usage, "prompt_tokens", 0)
    _token_usage["completion"] += getattr(usage, "completion_tokens", 0)


# ---------------------------------------------------------------------------
# LLM call — standard OpenAI client, retry on transient errors
# ---------------------------------------------------------------------------


def _call_llm(messages: List[Dict[str, Any]], client: OpenAI) -> str:
    """Call the LLM with exponential backoff retry. Returns raw response text."""
    attempt = 0
    while True:
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
                max_tokens=1024,
            )
            _track_usage(completion)
            return completion.choices[0].message.content or ""
        except Exception as e:
            attempt += 1
            wait = min(10 * (2 ** (attempt - 1)), 60)
            print(f"  [attempt {attempt}] {MODEL_NAME} error: {e}", flush=True)
            print(f"  [retry] waiting {wait}s...", flush=True)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Observation → prompt
# ---------------------------------------------------------------------------


def build_observation_prompt(obs: Dict[str, Any]) -> str:
    parts = [f"## Incident Status\n{obs.get('observation_summary', 'N/A')}"]

    alerts = obs.get("alerts", [])
    if alerts:
        alert_lines = [f"  [{a['severity'].upper()}] {a['message']}" for a in alerts[:10]]
        parts.append("## Active Alerts\n" + "\n".join(alert_lines))

    services = obs.get("services", [])
    degraded = [s for s in services if s.get("status") in ("degraded", "critical", "down")]
    if degraded:
        # Identify root causes: services that have OPEN circuit breakers pointing at them
        # from callers, but do not themselves have OPEN outgoing breakers
        breaker_targets: set = set()
        for s in services:
            for dep, state in s.get("circuit_breakers", {}).items():
                if state == "OPEN":
                    breaker_targets.add(dep)

        svc_lines = []
        for s in degraded:
            sid = s["id"]
            own_open = any(v == "OPEN" for v in s.get("circuit_breakers", {}).values())
            is_root = sid in breaker_targets and not own_open
            label = " [ROOT CAUSE]" if is_root else " [propagation victim]" if sid not in breaker_targets else ""
            svc_lines.append(
                f"  {sid} [{s['status']}]{label}: error={s['error_rate']:.1%}, "
                f"p99={s['latency_p99_ms']:.0f}ms, cpu={s['cpu_pct']:.0f}%, "
                f"mem={s['memory_pct']:.0f}%"
            )
        parts.append("## Degraded Services\n" + "\n".join(svc_lines))

    deploys = obs.get("recent_deploys", [])
    if deploys:
        dep_lines = [f"  {d['service']} -> {d['version']} ({d['ticks_ago']} ticks ago)" for d in deploys]
        parts.append("## Recent Deploys\n" + "\n".join(dep_lines))

    actions = obs.get("actions_taken", [])
    if actions:
        act_lines = [
            f"  tick {a['tick']}: {a['action']}({a.get('target', '')}) -> {'OK' if a['success'] else 'FAIL'}"
            for a in actions[-5:]
        ]
        parts.append("## Recent Actions\n" + "\n".join(act_lines))

    logs = obs.get("logs")
    if logs:
        parts.append(f"## Logs\n{logs}")

    traces = obs.get("traces")
    if traces:
        error_spans = [s for s in traces.get("spans", []) if s.get("status") == "ERROR"]
        if error_spans:
            trace_lines = [
                f"  {s['service']}: {s.get('tags', {}).get('error.message', 'ERROR')} ({s['duration_ms']}ms)"
                for s in error_spans[:5]
            ]
            parts.append("## Trace Errors\n" + "\n".join(trace_lines))

    legal = obs.get("legal_actions", [])
    if legal:
        legal_strs = [f"  {la['action_type']}: targets={la['valid_targets'][:5]}" for la in legal]
        parts.append("## Available Actions\n" + "\n".join(legal_strs))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------


def parse_action(response_text: str) -> Dict[str, Any]:
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"action_type": "noop", "params": {}}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def _wait_for_server(base: str, max_wait: int = 90) -> None:
    """Poll /health until server is ready or timeout."""
    import httpx, time
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/health", timeout=5.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"Server at {base} not ready after {max_wait}s")


def run_episode(
    client: OpenAI,
    task_id: str,
    seed: int,
) -> Dict[str, Any]:
    import httpx

    base = ENV_URL.rstrip("/")

    # Wait for server to be ready (handles startup race condition)
    _wait_for_server(base)

    # Reset environment
    reset_resp = httpx.post(
        f"{base}/reset",
        json={"seed": seed, "task_id": task_id},
        timeout=30.0,
    )
    resp_data = reset_resp.json()
    obs = resp_data.get("observation", resp_data)

    max_steps = obs.get("max_steps", 10)
    done = resp_data.get("done", False)
    rewards: List[float] = []

    # Persistent episode memory — survives rolling context truncation
    conversation_history: List[Dict[str, Any]] = []
    tried_actions: Dict[str, List[str]] = {}
    resolved_services: List[str] = []

    def _build_memory() -> str:
        if not tried_actions and not resolved_services:
            return ""
        lines = ["## Episode Memory (do not repeat failed approaches)"]
        if resolved_services:
            lines.append(f"  Resolved: {', '.join(resolved_services)}")
        for act, targets in tried_actions.items():
            lines.append(f"  {act}: {'; '.join(targets)}")
        return "\n".join(lines)

    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    steps_taken = 0
    for step_num in range(1, max_steps + 1):
        if done:
            break

        user_msg = build_observation_prompt(obs)
        conversation_history.append({"role": "user", "content": user_msg})

        # Rolling window of last 6 messages + persistent memory in system prompt
        trimmed = conversation_history[-6:]
        memory = _build_memory()
        system_content = SYSTEM_PROMPT + ("\n\n" + memory if memory else "")
        messages_to_send = [{"role": "system", "content": system_content}] + trimmed

        response_text = _call_llm(messages_to_send, client)
        action = parse_action(response_text)
        conversation_history.append({"role": "assistant", "content": response_text})

        act_type = action.get("action_type", "noop")
        act_params = action.get("params", {})
        target = act_params.get("service_id") or act_params.get("cache_name") or act_params.get("from_region") or ""

        # Coerce replicas to int
        if "replicas" in act_params:
            try:
                act_params["replicas"] = int(act_params["replicas"])
            except (ValueError, TypeError):
                act_params["replicas"] = 2

        print(f"  Step {step_num}: {act_type}({act_params})", flush=True)

        try:
            step_resp = httpx.post(
                f"{base}/step",
                json={"action": {"action_type": act_type, "params": act_params}},
                timeout=30.0,
            )
            resp_data = step_resp.json()
        except Exception as e:
            print(f"  [step error] {e}", flush=True)
            resp_data = {}

        obs = resp_data.get("observation", resp_data)
        done = resp_data.get("done", False)
        reward = float(obs.get("reward") or resp_data.get("reward") or 0.0)
        rewards.append(reward)
        steps_taken = step_num

        log_step(step=step_num, action=act_type, reward=reward, done=done)

        # Update persistent memory
        if act_type not in ("inspect_logs", "inspect_metrics", "inspect_traces", "noop") and target:
            new_slo = obs.get("global_slo_score", 0.0)
            for svc in obs.get("services", []):
                if svc["id"] == target and svc["status"] == "healthy":
                    if target not in resolved_services:
                        resolved_services.append(target)
            entry = f"{target} (slo={new_slo:.0%})"
            tried_actions.setdefault(act_type, [])
            if entry not in tried_actions[act_type]:
                tried_actions[act_type].append(entry)

    # Grade the episode
    try:
        final_state = httpx.get(f"{base}/state", timeout=10.0).json()
    except Exception:
        final_state = {}
    try:
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
    except Exception:
        grade = {}

    score = grade.get("score", 0.0)
    outcome = final_state.get("termination_reason", "timeout")
    success = outcome == "resolved"

    log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "seed": seed,
        "score": score,
        "slo_recovery": grade.get("slo_recovery", 0.0),
        "action_efficiency": grade.get("action_efficiency", 0.0),
        "time_efficiency": grade.get("time_efficiency", 0.0),
        "steps_taken": final_state.get("step_count", 0),
        "termination_reason": outcome,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    all_tasks = {"easy": 42, "medium": 123, "hard": 7}
    task_filter = os.getenv("TASKS", "").strip()
    selected = [t.strip() for t in task_filter.split(",")] if task_filter else list(all_tasks)
    tasks = [(t, all_tasks[t]) for t in selected if t in all_tasks]

    print("=" * 60, flush=True)
    print("SevZero Baseline Inference", flush=True)
    print("=" * 60, flush=True)
    print(f"Model:       {MODEL_NAME}", flush=True)
    print(f"API:         {API_BASE_URL}", flush=True)
    print(f"Environment: {ENV_URL}", flush=True)
    print(flush=True)

    results = []
    for task_id, seed in tasks:
        print(f"--- Task: {task_id} (seed={seed}) ---", flush=True)
        result = run_episode(client, task_id, seed)
        results.append(result)
        print(
            f"  Score: {result['score']:.4f} | SLO: {result['slo_recovery']:.4f} | "
            f"AE: {result['action_efficiency']:.4f} | TE: {result['time_efficiency']:.4f} | "
            f"Steps: {result['steps_taken']} | Outcome: {result['termination_reason']}",
            flush=True,
        )
        print(flush=True)

    print("=" * 60, flush=True)
    print("Summary", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(f"  {r['task_id']:8s} score={r['score']:.4f}  slo={r['slo_recovery']:.4f}  steps={r['steps_taken']}", flush=True)
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n  Average score: {avg_score:.4f}", flush=True)

    # Save results
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "run_at": run_ts,
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "average_score": round(avg_score, 4),
        "results": results,
    }
    out_file = outputs_dir / f"baseline_{run_ts}.json"
    (outputs_dir / "baseline_latest.json").write_text(json.dumps(payload, indent=2))
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"\n  Results saved -> {out_file.name}", flush=True)

    total = _token_usage["prompt"] + _token_usage["completion"]
    print(f"\n  Token usage:", flush=True)
    print(f"    prompt:     {_token_usage['prompt']:,}", flush=True)
    print(f"    completion: {_token_usage['completion']:,}", flush=True)
    print(f"    total:      {total:,}", flush=True)


if __name__ == "__main__":
    main()
