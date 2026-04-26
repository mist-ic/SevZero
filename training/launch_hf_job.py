#!/usr/bin/env python3
"""
Submit a HuggingFace Job to run training/train_sft.py or training/train_grpo.py.
Uses huggingface_hub.run_job; prints job URL; appends training/runs.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.config_utils import try_load_env_files

try_load_env_files()


def _default_git_url() -> str:
    r = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
    )
    return (r.stdout or "").strip() if r.returncode == 0 else ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--account_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--script", type=str, choices=("sft", "grpo"), required=True)
    p.add_argument("--variant_name", type=str, default="run")
    p.add_argument("--hardware", type=str, default="l40sx1")
    p.add_argument(
        "--image",
        type=str,
        default="pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime",
    )
    p.add_argument("--git-url", type=str, default="")
    p.add_argument(
        "--env_vars",
        type=str,
        default="",
        help="KEY=val pairs comma-separated, e.g. SEVZERO_ENV_URL=https://x.hf.space,HF_MAIN_TOKEN=...",
    )
    a, rest = p.parse_known_args()
    if not a.account_token:
        raise SystemExit("Need HF_TOKEN or --account_token")
    git_url = a.git_url or _default_git_url()
    if not git_url:
        raise SystemExit("Set --git-url or configure git origin")
    ev = {k: v for k, v in [x.split("=", 1) for x in a.env_vars.split(",") if "=" in x]}
    if "SEVZERO_ENV_URL" not in ev and os.environ.get("SEVZERO_ENV_URL"):
        ev["SEVZERO_ENV_URL"] = os.environ["SEVZERO_ENV_URL"]

    which = f"training/train_{a.script}.py"
    extra = " ".join(rest)
    if a.script == "grpo":
        # April 2026 pin compatible with our rollout_func + trl.experimental.openenv API.
        #   - trl==1.2.0 (2026-04-17): rollout_func + generate_rollout_completions came in TRL 1.0.0 (PR #5122).
        #   - vllm==0.18.0: TRL 1.2.0 caps vLLM at 0.18.0 (PR #5547); 0.18.0 requires transformers<5.
        #   - transformers==4.57.0: in the intersection of trl>=4.56.2 and vllm<5,>=4.56.0.
        #   - Image: pytorch:2.10.0-cuda12.8-cudnn9-runtime (vllm 0.18 ships against torch 2.10).
        # Ubuntu 24.04 needs PIP_BREAK_SYSTEM_PACKAGES; no unsloth.
        deps = (
            "pip install 'trl==1.2.0' 'peft' 'transformers==4.57.0' "
            "'accelerate' 'bitsandbytes' 'datasets' 'huggingface_hub' 'httpx' 'python-dotenv' "
            "'vllm==0.18.0' 'trackio'"
        )
    else:
        # SFT also uses plain transformers + PEFT (no Unsloth import path).
        deps = (
            "pip install 'trl==1.2.0' 'peft' 'transformers==4.57.0' "
            "'accelerate' 'bitsandbytes' 'datasets' 'huggingface_hub' 'httpx' 'python-dotenv' "
            "'trackio'"
        )
    inner = (
        "set -euo pipefail && "
        "(command -v git >/dev/null 2>&1 || (apt-get update -y && apt-get install -y --no-install-recommends git ca-certificates)) && "
        f"git clone --depth 1 {git_url!r} /work/r && cd /work/r && "
        "export PIP_BREAK_SYSTEM_PACKAGES=1 && "
        "pip install -U pip && "
        f"{deps} && "
        f"python {which} --variant_name {a.variant_name!r} {extra}"
    )
    from huggingface_hub import run_job

    job = run_job(
        image=a.image,
        command=["bash", "-lc", inner],
        env=ev,
        secrets={"HF_TOKEN": a.account_token},
        flavor=a.hardware,
    )
    with (_REPO / "training" / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "account_token_tail": a.account_token[-4:] if len(a.account_token) > 4 else "",
                    "job_id": str(getattr(job, "id", job)),
                    "variant_name": a.variant_name,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )
    print(getattr(job, "url", f"https://huggingface.co/jobs/{getattr(job, 'id', job)}"), flush=True)


if __name__ == "__main__":
    main()
