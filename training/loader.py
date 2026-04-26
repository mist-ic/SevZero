"""
Load SevZero SFT data for a trainer: local JSONL or the Hub Parquet copy.

The training config should set `max_seq_length` to at least
`max_prompt_token_length` from `build_stats.json` (plus max completion length).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "training" / "data"

try:
    from datasets import Dataset, DatasetDict, load_dataset
except ImportError as e:
    raise ImportError("Install `datasets` to use the loader.") from e


def load_local_jsonl(
    train_path: Optional[Path] = None,
    eval_path: Optional[Path] = None,
) -> DatasetDict:
    train_path = train_path or (DATA_DIR / "sft_train.jsonl")
    eval_path = eval_path or (DATA_DIR / "sft_eval.jsonl")
    train = load_dataset("json", data_files=str(train_path), split="train")
    if eval_path.is_file() and eval_path.stat().st_size > 0:
        ev = load_dataset("json", data_files=str(eval_path), split="train")
    else:
        ev = train.select([])
    return DatasetDict(train=train, eval=ev)


def load_from_hub(
    repo_id: str = "Mist-ic/sevzero-expert-trajectories",
    token: Optional[str] = None,
) -> DatasetDict:
    tok = token or os.environ.get("HF_MAIN_TOKEN")
    return load_dataset(repo_id, token=tok)  # type: ignore[return-value]


def read_build_stats() -> dict[str, Any]:
    p = DATA_DIR / "build_stats.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def recommended_max_seq_length(plus_completion: int = 1024) -> int:
    s = read_build_stats()
    m = int(s.get("max_prompt_token_length", 0) or 0)
    return m + plus_completion
