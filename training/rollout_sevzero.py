"""
SevZero multi-turn rollout helpers for TRL GRPO (sync API for rollout_func).
Builds chat prompts from observations and parses one JSON action per turn.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple

SRE_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert Site Reliability Engineer (SRE) responding to a production incident.
    You are managing a microservice cluster experiencing failures.
    Your goal: restore all services to healthy SLO compliance as efficiently as possible.

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
"""
)


def build_observation_prompt(obs: Dict[str, Any]) -> str:
    """Port of inference.build_observation_prompt (observation dict from HTTP JSON)."""
    parts = [f"## Incident Status\n{obs.get('observation_summary', 'N/A')}"]
    alerts = obs.get("alerts") or []
    if alerts:
        alert_lines = [f"  [{a['severity'].upper()}] {a['message']}" for a in alerts[:10]]
        parts.append("## Active Alerts\n" + "\n".join(alert_lines))
    services = obs.get("services") or []
    degraded = [s for s in services if s.get("status") in ("degraded", "critical", "down")]
    if degraded:
        svc_lines = []
        for s in degraded:
            sid = s["id"]
            svc_lines.append(
                f"  {sid} [{s['status']}]: error={s['error_rate']:.1%}, "
                f"p99={s['latency_p99_ms']:.0f}ms, cpu={s['cpu_pct']:.0f}%, "
                f"mem={s['memory_pct']:.0f}%"
            )
        parts.append("## Degraded Services\n" + "\n".join(svc_lines))
    deploys = obs.get("recent_deploys") or []
    if deploys:
        dep_lines = [f"  {d['service']} -> {d['version']} ({d['ticks_ago']} ticks ago)" for d in deploys]
        parts.append("## Recent Deploys\n" + "\n".join(dep_lines))
    actions = obs.get("actions_taken") or []
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
        spans = (traces.get("spans") or []) if isinstance(traces, dict) else []
        error_spans = [s for s in spans if s.get("status") == "ERROR"]
        if error_spans:
            trace_lines = [
                f"  {s.get('service')}: {s.get('tags', {}).get('error.message', 'ERROR')}"
                for s in error_spans[:5]
            ]
            parts.append("## Trace Errors\n" + "\n".join(trace_lines))
    legal = obs.get("legal_actions") or []
    if legal:
        legal_strs = [f"  {la.get('action_type', '')}: targets={la.get('valid_targets', [])[:5]}" for la in legal]
        parts.append("## Available Actions\n" + "\n".join(legal_strs))
    return "\n\n".join(parts)


def parse_action(response_text: str) -> Dict[str, Any]:
    text = (response_text or "").strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"action_type": "noop", "params": {}}


def _normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    act_type = action.get("action_type", "noop")
    params = dict(action.get("params") or {})
    if "replicas" in params:
        try:
            params["replicas"] = int(params["replicas"])
        except (TypeError, ValueError):
            params["replicas"] = 2
    return {"action_type": act_type, "params": params}
