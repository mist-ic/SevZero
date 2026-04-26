"""
Build Llama-3.1-8B-Instruct SFT jsonl from raw trajectory jsonl (score ≥ 0.85).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import SYSTEM_PROMPT  # noqa: E402

load_dotenv(REPO_ROOT / "api.env")
load_dotenv(REPO_ROOT / "hg.env")

DATA_DIR = REPO_ROOT / "training" / "data"
RAW_GLOB = "raw/*.jsonl"
OUT_TRAIN = DATA_DIR / "sft_train.jsonl"
OUT_EVAL = DATA_DIR / "sft_eval.jsonl"
OUT_STATS = DATA_DIR / "build_stats.json"

MAX_OBS_TOKENS = 2048


def _get_tokenizer():
    import os

    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    name = "meta-llama/Llama-3.1-8B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(
            name, token=os.environ.get("HF_MAIN_TOKEN")
        )
        return tok
    except Exception:
        try:
            return AutoTokenizer.from_pretrained(
                "hf-internal-testing/llama-tokenizer"
            )
        except Exception:
            return None


def _count_tokens(toker, text: str) -> int:
    if toker is not None:
        return len(toker.encode(text, add_special_tokens=False))
    return max(1, len(text) // 4)


def _shrink_observation(obs: Dict[str, Any], toker, max_toks: int) -> str:
    """Serialize observation to JSON, shrink until user message fits max_toks (approximate)."""
    o = {k: v for k, v in obs.items() if k not in ("reward",)}
    order_drop = [
        "metric_history",
        "traces",
        "logs",
        "actions_taken",
        "recent_deploys",
    ]
    for _ in range(40):
        text = json.dumps(o, ensure_ascii=False, separators=(",", ":"), default=str)
        tcount = _count_tokens(toker, text)
        if tcount <= max_toks:
            return text
        shrunk = False
        for k in order_drop:
            if k in o and o[k]:
                o[k] = None
                if k == "actions_taken":
                    o[k] = []
                elif k in ("metric_history", "recent_deploys"):
                    o[k] = []
                shrunk = True
                break
        if shrunk:
            continue
        if "services" in o and isinstance(o["services"], list) and len(o["services"]) > 2:
            o["services"] = o["services"][: max(1, len(o["services"]) - 1)]
            continue
        if "alerts" in o and isinstance(o["alerts"], list) and len(o["alerts"]) > 1:
            o["alerts"] = o["alerts"][: max(0, len(o["alerts"]) - 1)]
            continue
        o["__truncated__"] = True
        break
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"), default=str)


def _episode_id(ep: Dict[str, Any]) -> str:
    return f"{ep.get('model', '')}|{ep.get('task_id', '')}|{ep.get('seed', 0)}"


def _assistant_action_json(action: Any) -> str:
    if not isinstance(action, dict):
        return json.dumps(
            {"action_type": "noop", "params": {}}, ensure_ascii=False
        )
    a = {
        "action_type": str(action.get("action_type", "noop")),
        "params": action.get("params") or {},
    }
    return json.dumps(a, ensure_ascii=False)


def _load_episodes_from_raw(raw_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(raw_dir.glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    return out


def build(
    min_score: float = 0.85,
) -> Dict[str, Any]:
    toker = _get_tokenizer()
    raw_dir = DATA_DIR / "raw"
    episodes = _load_episodes_from_raw(raw_dir)
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for ep in episodes:
        sc = float(ep.get("final_score", 0.0) or 0.0)
        if sc >= min_score and ep.get("steps"):
            kept.append(ep)
        else:
            dropped.append(ep)

    eids = [_episode_id(e) for e in kept]
    unique_eids = list(dict.fromkeys(eids))
    n_ep = len(unique_eids)
    rng = random.Random(42)
    rng.shuffle(unique_eids)
    if n_ep <= 1:
        n_eval = 0
    else:
        n_eval = max(1, n_ep // 10)
    eval_ids: Set[str] = set(unique_eids[:n_eval]) if n_eval else set()

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    max_prompt_toks = 0

    for ep in kept:
        eid = _episode_id(ep)
        is_eval = eid in eval_ids
        for st in ep.get("steps", []):
            obs = st.get("observation", {})
            if not isinstance(obs, dict):
                continue
            user_str = _shrink_observation(obs, toker, MAX_OBS_TOKENS)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_str},
                {
                    "role": "assistant",
                    "content": _assistant_action_json(st.get("action", {})),
                },
            ]
            if toker is not None:
                try:
                    plen = len(
                        toker.apply_chat_template(
                            messages, tokenize=True, add_generation_prompt=False
                        )
                    )
                except Exception:
                    plen = _count_tokens(
                        toker, SYSTEM_PROMPT + "\n" + user_str
                    )
            else:
                plen = _count_tokens(
                    None, SYSTEM_PROMPT + "\n" + user_str
                )
            max_prompt_toks = max(max_prompt_toks, plen)
            row = {
                "messages": messages,
                "meta": {
                    "episode_id": eid,
                    "model": ep.get("model"),
                    "task_id": ep.get("task_id"),
                    "seed": ep.get("seed"),
                    "step": st.get("step"),
                    "episode_score": ep.get("final_score"),
                },
            }
            if is_eval:
                eval_rows.append(row)
            else:
                train_rows.append(row)

    scores = [float(x.get("final_score", 0) or 0) for x in kept]
    mean_sc = sum(scores) / len(scores) if scores else 0.0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_TRAIN.open("w", encoding="utf-8") as ft:
        for r in train_rows:
            ft.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_EVAL.open("w", encoding="utf-8") as fe:
        for r in eval_rows:
            fe.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats: Dict[str, Any] = {
        "episodes_total_seen": len(episodes),
        "episodes_kept": len(kept),
        "episodes_dropped": len(dropped),
        "mean_episode_score_kept": round(mean_sc, 6),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "max_prompt_token_length": max_prompt_toks,
        "max_observation_user_token_budget": MAX_OBS_TOKENS,
        "min_score_filter": min_score,
    }
    with OUT_STATS.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=0.85)
    args = ap.parse_args()
    build(min_score=args.min_score)


if __name__ == "__main__":
    main()
