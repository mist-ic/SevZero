#!/usr/bin/env python3
"""
GRPO on SevZero via TRL rollout_func + trl.experimental.openenv.generate_rollout_completions.
Verify API with Context7 before changing integration (rollout_func is required; environment_factory is deprecated).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.config_utils import try_load_env_files

try_load_env_files()

BASE_MODEL = os.environ.get(
    "SEVZERO_BASE_MODEL",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
)
METRICS_NAME = "metrics.jsonl"

# Pinned in README: trl, unsloth, vllm — orchestrator sets exact versions


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="./outputs/grpo")
    p.add_argument("--sft_adapter_repo", type=str, required=True, help="HF adapter repo (worker account)")
    p.add_argument("--env_url", type=str, default="", help="Override; else SEVZERO_ENV_URL")
    p.add_argument("--max_steps", type=int, default=350)
    p.add_argument("--lr", type=float, default=7e-6)
    p.add_argument("--K", type=int, default=4, dest="K", help="num_generations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--reward_shaping",
        type=str,
        default="dense_v1",
        choices=("dense_v1", "dense_v2", "sparse"),
    )
    p.add_argument("--enable_schema_drift", action="store_true")
    p.add_argument("--enable_curriculum", action="store_true")
    p.add_argument("--enable_oversight", action="store_true")
    p.add_argument(
        "--task_mix",
        type=str,
        default="hard",
        choices=("hard", "mixed", "curriculum"),
    )
    p.add_argument("--push_to_hub_repo", type=str, default="")
    p.add_argument("--variant_name", type=str, default="grpo")
    p.add_argument("--rollout_max_steps", type=int, default=0, help="0 = from env observation max_steps")
    return p.parse_args()


def _pick_task_id(args, idx: int, step: int) -> str:
    if args.task_mix == "hard":
        return "hard"
    if args.task_mix == "mixed":
        return ["easy", "medium", "hard"][idx % 3]
    # curriculum: escalate every ~50 steps
    if args.enable_curriculum:
        tier = min(2, step // 50)
        return ["easy", "medium", "hard"][tier]
    return "hard"


def _compute_episode_return(
    shaping: str,
    step_rewards: List[float],
    grader: Optional[Dict[str, Any]],
) -> float:
    if shaping == "sparse" and grader is not None:
        return float(grader.get("score", 0.0))
    if shaping == "dense_v2" and grader is not None:
        # Slightly weight terminal score
        s = sum(step_rewards) if step_rewards else 0.0
        return 0.7 * s + 0.3 * float(grader.get("score", 0.0))
    return float(sum(step_rewards)) if step_rewards else 0.0


def _build_default_dataset():
    from datasets import Dataset

    rows = []
    for i in range(64):
        text = (
            "You are the on-call SRE. Restore service health. "
            f"Incident session {i} — triage, diagnose root cause, remediate, verify."
        )
        rows.append({"text": text, "row_id": i})
    return Dataset.from_list(rows)


def _reward_from_env(completions, **kwargs):
    r = kwargs.get("env_reward")
    if r is None:
        return [0.0] * len(completions)
    return [float(x) for x in r]


def main() -> None:
    args = _parse_args()
    env_url = (args.env_url or os.environ.get("SEVZERO_ENV_URL", "")).rstrip("/")
    if not env_url:
        raise SystemExit("Set --env_url or SEVZERO_ENV_URL to the remote SevZero HTTP base URL")

    worker_token = os.environ.get("HF_TOKEN", "")
    main_token = os.environ.get("HF_MAIN_TOKEN", "")

    try:
        import trackio

        kwargs: dict = {"project": "sevzero-grpo", "space_id": "Mist-ic/sevzero-trackio"}
        if main_token:
            try:
                trackio.init(**kwargs, token=main_token)
            except TypeError:
                trackio.init(**kwargs)
        else:
            trackio.init(**kwargs)
    except Exception as e:
        print(f"trackio init skipped: {e}", flush=True)

    try:
        from unsloth import FastLanguageModel, PatchFastRL
    except ImportError as e:
        raise SystemExit(
            f"unsloth is required for GRPO on this path: {e}\n"
            "Install training extras, or on unsupported platforms set UNSLOTH_DISABLE=1 and extend train_grpo."
        ) from e

    PatchFastRL(algorithm="grpo", FastLanguageModel=FastLanguageModel)

    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer
    from trl.experimental.openenv import generate_rollout_completions

    from training.env_client import AsyncSevZeroEnvClient, run_async
    from training.rollout_sevzero import (
        SRE_SYSTEM_PROMPT,
        build_observation_prompt,
        parse_action,
    )

    max_seq = 4096
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, args.sft_adapter_repo, token=worker_token or None)
    # Optional env flags (future env upgrades) — no-op for baseline server
    if args.enable_schema_drift:
        os.environ["SEVZERO_SCHEMA_DRIFT"] = "1"
    if args.enable_oversight:
        os.environ["SEVZERO_OVERSIGHT"] = "1"

    metrics_path = Path(args.output_dir) / METRICS_NAME
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Capture trainer ref for step index in seeding
    _trainer_holder: List[Any] = [None]
    _global_episode: List[int] = [0]

    def rollout_func(prompts: List[str], trainer) -> Dict[str, List[Any]]:
        _trainer_holder[0] = trainer
        episode_prompt_ids: List[List[int]] = []
        episode_completion_ids: List[List[int]] = []
        episode_logprobs: List[List[float]] = []
        env_rewards: List[float] = []
        tkn = os.environ.get("HF_TOKEN", "")  # for private Space
        for batch_idx, prompt_text in enumerate(prompts):
            tr = _trainer_holder[0]
            state = getattr(tr, "state", None) if tr else None
            step = getattr(state, "global_step", 0) if state else 0
            _global_episode[0] += 1
            task_id = _pick_task_id(args, batch_idx, step)
            seed = 13 + (batch_idx * 997) + (step * 13) + _global_episode[0] + random.randint(0, 1_000_000) % 100_000

            async def _one_ep() -> tuple:
                client = AsyncSevZeroEnvClient(env_url, token=tkn or None)
                try:
                    p_ids: List[int] = []
                    c_ids: List[int] = []
                    lps: List[float] = []
                    step_rewards: List[float] = []
                    ro = await client.reset(task_id=task_id, seed=seed)
                    obs = ro.get("observation", ro)
                    done = ro.get("done", False)
                    grader: Optional[Dict[str, Any]] = None
                    user_prefix = f"{prompt_text}\n\n## Session\n"
                    for _t in range(args.rollout_max_steps or int(obs.get("max_steps", 20))):
                        if done:
                            break
                        user_msg = build_observation_prompt(obs)
                        messages = [
                            {"role": "system", "content": SRE_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prefix + user_msg},
                        ]
                        p_text = tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False,
                        )
                        out = generate_rollout_completions(tr, [p_text])[0]
                        p_ids.extend(out.get("prompt_ids", []))
                        c_ids.extend(out.get("completion_ids", []))
                        lps.extend(out.get("logprobs", []))
                        gen_ids = out.get("completion_ids", [])
                        raw = out.get("text")
                        if not raw and gen_ids:
                            raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        action = parse_action(raw or "")
                        step_payload = {
                            "action_type": str(action.get("action_type", "noop")),
                            "params": action.get("params") or {},
                        }
                        sr = await client.step({"action": step_payload})
                        obs = sr.get("observation", sr)
                        done = sr.get("done", False)
                        r = float(obs.get("reward", sr.get("reward", 0.0) or 0.0))
                        step_rewards.append(r)
                    st = await client.get_state()
                    max_st = int(obs.get("max_steps", 10))
                    try:
                        grader = await client.grade_episode(
                            final_slo_score=float(st.get("global_slo_score", 0.0)),
                            steps_taken=int(st.get("step_count", 0)),
                            max_steps=max_st,
                            actions_taken=list(obs.get("actions_taken", [])),
                            terminated=bool(st.get("terminated", True)),
                            termination_reason=st.get("termination_reason"),
                        )
                    except Exception:
                        grader = None
                    R = _compute_episode_return(args.reward_shaping, step_rewards, grader)
                    return p_ids, c_ids, lps, R
                finally:
                    await client.aclose()

            p_ids, c_ids, lps, r_ep = run_async(_one_ep())
            episode_prompt_ids.append(p_ids)
            episode_completion_ids.append(c_ids)
            episode_logprobs.append(lps)
            env_rewards.append(r_ep)
        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "env_reward": env_rewards,
        }

    grpo = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_completion_length=1024,
        num_train_epochs=1,
        max_steps=args.max_steps,
        num_generations=args.K,
        temperature=0.85,
        max_prompt_length=4096,
        beta=0.04,
        lr_scheduler_type="cosine",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.55,
        report_to="trackio",
        logging_steps=1,
        save_steps=100,
    )

    train_ds = _build_default_dataset()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo,
        train_dataset=train_ds,
        reward_funcs=[_reward_from_env],
        rollout_func=rollout_func,
    )

    from transformers import TrainerCallback

    class _MetricsJSONL(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            rec = {
                "step": state.global_step,
                "reward_mean": logs.get("rewards", logs.get("reward", None)),
                "reward_std": logs.get("reward_std", None),
                "kl": logs.get("kl", None),
                "entropy": logs.get("entropy", None),
                "grad_norm": logs.get("grad_norm", None),
                "loss": logs.get("loss", None),
                "frac_reward_zero_std": logs.get("frac_reward_zero", logs.get("frac_reward_zero_std", None)),
                "lr": logs.get("learning_rate", None),
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            print(json.dumps({"type": "grpo", **rec}, default=str), flush=True)

    trainer.add_callback(_MetricsJSONL())
    trainer.train()

    if args.push_to_hub_repo:
        model.push_to_hub(args.push_to_hub_repo, token=worker_token or None, private=True)
        tokenizer.push_to_hub(args.push_to_hub_repo, token=worker_token or None, private=True)


if __name__ == "__main__":
    main()
