"""
server/oversight.py — Virtual SRE manager gating for high-impact actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class _Grant:
    key: str
    for_action: str
    for_target: str
    granted_at_tick: int
    expires_after_tick: int  # grant valid: granted_at <= tick < expires_after


def _is_identity_rollback(simulation: Any, service_id: str) -> bool:
    g = simulation.graph
    if not g or not service_id:
        return False
    node = g.node_map.get(service_id)
    return bool(node and node.layer == "identity")


def _needs_postgres_or_primary_restart(target: str) -> bool:
    t = (target or "").lower()
    return "postgres" in t or "primary" in t


def _approval_key(action_type: str, target: str) -> str:
    return f"{action_type}::{target}"


@dataclass
class OversightManager:
    """
    Policy + approval storage. Ticks are simulation ticks after each env step
    (matches Simulator.tick at the start of a step, before inner increment).
    """

    _grants: Dict[str, _Grant] = field(default_factory=dict)
    _policy: List[Dict[str, Any]] = field(default_factory=list)
    _pending: List[Dict[str, Any]] = field(default_factory=list)
    _request_tick: Dict[str, int] = field(default_factory=dict)
    _enabled: bool = False

    def on_reset(self, simulation: Any, enable: bool, max_steps_override: int) -> None:  # noqa: ARG002
        self._enabled = enable
        self._grants.clear()
        self._pending.clear()
        self._request_tick.clear()
        if not enable:
            self._policy = []
            return
        self._policy = [
            {
                "action_type": "restart_service",
                "target_pattern": "*postgres* or *primary*",
                "reason": "Restarts on database primaries are high-blast-radius",
            },
            {
                "action_type": "rebalance_traffic",
                "target_pattern": "pct >= 40",
                "reason": "Large traffic shifts are high-risk",
            },
            {
                "action_type": "rollback_service",
                "target_pattern": "identity layer services",
                "reason": "Auth/session rollbacks are customer-impacting",
            },
        ]

    @property
    def policy(self) -> List[Dict[str, Any]]:
        return self._policy

    @property
    def pending_approvals(self) -> List[Dict[str, Any]]:
        return list(self._pending)

    def is_high_impact(
        self, simulation: Any, action_type: str, params: Dict[str, Any],
    ) -> bool:
        if action_type == "restart_service":
            sid = str(params.get("service_id", ""))
            return _needs_postgres_or_primary_restart(sid)
        if action_type == "rebalance_traffic":
            try:
                p = int(params.get("pct", 50))
            except (TypeError, ValueError):
                p = 50
            return p >= 40
        if action_type == "rollback_service":
            sid = str(params.get("service_id", ""))
            return _is_identity_rollback(simulation, sid)
        return False

    def _prune(self, current_tick: int) -> None:
        dead: List[str] = []
        for k, g in self._grants.items():
            if current_tick >= g.expires_after_tick:
                dead.append(k)
        for k in dead:
            self._grants.pop(k, None)
        for p in self._pending:
            st = p.get("state", "")
            if st != "requested":
                continue
            t0 = int(p.get("submitted_at", 0))
            if current_tick - t0 > 3:
                p["state"] = "expired"

    def on_tick_start(self, simulation: Any) -> None:
        if not self._enabled:
            return
        t = int(simulation.tick)
        self._prune(t)
        new_pending: List[Dict[str, Any]] = []
        for p in self._pending:
            st = p.get("state", "")
            if st != "requested":
                new_pending.append(p)
                continue
            sub = int(p.get("submitted_at", t))
            if t < sub + 1:
                new_pending.append(p)
                continue
            a = str(p.get("action_type", ""))
            tgt = str(p.get("target", ""))
            k = _approval_key(a, tgt)
            self._grants[k] = _Grant(
                key=k, for_action=a, for_target=tgt,
                granted_at_tick=t, expires_after_tick=t + 3,
            )
            p2 = dict(p)
            p2["state"] = "granted"
            p2["granted_at"] = t
            new_pending.append(p2)
        self._pending = new_pending

    def has_valid_approval(
        self, action_type: str, target: str, current_tick: int,
    ) -> bool:
        k = _approval_key(action_type, target)
        g = self._grants.get(k)
        if not g:
            return False
        return g.granted_at_tick <= current_tick < g.expires_after_tick

    def should_block(
        self, simulation: Any, action_type: str, params: Dict[str, Any],
    ) -> bool:
        if not self._enabled or not self.is_high_impact(simulation, action_type, params):
            return False
        t = int(simulation.tick)
        target = self._target_for_approval(action_type, params)
        return not self.has_valid_approval(action_type, target, t)

    @staticmethod
    def _target_for_approval(action_type: str, params: Dict[str, Any]) -> str:
        if action_type == "rebalance_traffic":
            fr = str(params.get("from_region", "") or params.get("region", "") or "")
            to = str(params.get("to_region", "") or params.get("target", "") or "")
            return f"{fr}->{to}"
        return str(params.get("service_id", ""))

    def on_request_approval(
        self, params: Dict[str, Any], current_tick: int,
    ) -> None:
        a = str(params.get("action_type", ""))
        tgt = str(params.get("target", ""))
        k = _approval_key(a, tgt)
        self._pending.append({
            "action_type": a,
            "target": tgt,
            "reason": str(params.get("reason", "")),
            "state": "requested",
            "submitted_at": current_tick,
        })
        self._request_tick[k] = current_tick
