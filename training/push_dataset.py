"""
Upload SFT jsonl to Hugging Face (Mist-ic Main account) as a public dataset with Parquet.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / "api.env")
load_dotenv(REPO_ROOT / "hg.env")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "training" / "data"
STATS_PATH = DATA_DIR / "build_stats.json"


def _readme(stats: dict) -> str:
    return f"""# SevZero expert trajectories (SFT)

## Sources

- Synthetic expert rollouts from frontier models (Gemini 3.1 Pro, Azure OpenAI, Azure AI Inference)
  against the local OpenEnv `server.app` SevZero environment.

## Filtering

- Episodes with final grader `score` **≥** `{stats.get("min_score_filter", 0.85)}` are included.

## Schema

- Each example has a `messages` list (Llama-3.1-8B-Instruct–style SFT) and `meta` (episode / step provenance):
  - `system`: SRE on-call system prompt (same as `inference.SYSTEM_PROMPT` in the repo)
  - `user`: JSON-serialized observation (shrink to ≤ {stats.get("max_observation_user_token_budget", 2048)} tokens for the user part)
  - `assistant`: one JSON object `{{"action_type": "...", "params": {{...}}}}`

## Stats (from `build_stats.json` at publish time)

{json.dumps(stats, indent=2)}

## Parquet

- Splits `train` and `eval` are also pushed in Parquet for fast `datasets.load_dataset`.
"""


def _dataset_info(stats: dict) -> dict:
    return {
        "description": "SevZero SFT expert trajectories for Llama-3.1-8B-Instruct style chat training.",
        "version": "1.0.0",
        "license": "apache-2.0",
        "build": stats,
    }


def main() -> None:
    token = os.environ.get("HF_MAIN_TOKEN", "")
    if not token:
        raise SystemExit("HF_MAIN_TOKEN missing (set in api.env or hg.env).")
    user = (os.environ.get("HF_MAIN_USERNAME", "") or "").strip() or "Mist-ic"
    repo_id = f"{user}/sevzero-expert-trajectories"
    if not (DATA_DIR / "sft_train.jsonl").is_file():
        raise SystemExit(f"Missing {DATA_DIR / 'sft_train.jsonl'} — run build_dataset.py first.")
    stats: dict = {}
    if STATS_PATH.is_file():
        stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    readme = _readme(stats)
    info = _dataset_info(stats)
    (DATA_DIR / "DATASET_README_HF.md").write_text(readme, encoding="utf-8")
    (DATA_DIR / "dataset_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    for name in (
        "sft_train.jsonl",
        "sft_eval.jsonl",
        "build_stats.json",
        "dataset_info.json",
    ):
        p = DATA_DIR / name
        if p.is_file():
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Add SFT files and metadata",
            )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset README",
    )

    from datasets import DatasetDict, load_dataset

    train = load_dataset("json", data_files=str(DATA_DIR / "sft_train.jsonl"))["train"]
    evp = DATA_DIR / "sft_eval.jsonl"
    if evp.is_file() and evp.stat().st_size > 0:
        ev = load_dataset("json", data_files=str(evp))["train"]
    else:
        ev = train.select([])
    dd = DatasetDict(train=train, eval=ev)
    dd.push_to_hub(repo_id, private=False, token=token)

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(url, flush=True)


if __name__ == "__main__":
    main()
