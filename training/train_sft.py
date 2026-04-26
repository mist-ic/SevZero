#!/usr/bin/env python3
"""
SFT warmup: QLoRA on Mist-ic/sevzero-expert-trajectories (see training/data/HANDOFF.md).
Target TRL / Unsloth versions: see comments after `pip index` in training/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.config_utils import try_load_env_files

try_load_env_files()

# --- Pin guidance (orchestrator resolves exact pins): trl>=0.22, unsloth, bitsandbytes, peft, accelerate
# Use Unsloth's ungated mirror of Llama-3.1-8B-Instruct (same weights, 4-bit pre-quant) so worker
# accounts don't need to be approved for meta-llama/* gated repos.
BASE_MODEL = os.environ.get(
    "SEVZERO_BASE_MODEL",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
)
USE_4BIT = os.environ.get("SEVZERO_USE_4BIT", "0").lower() in ("1", "true", "yes")
DATASET_ID = "Mist-ic/sevzero-expert-trajectories"
DEFAULT_MAX_SEQ = 1024


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="./outputs/sft")
    p.add_argument("--max_steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push_to_hub_repo", type=str, default="", help="e.g. PhaseOfCode/sevzero-llama3-8b-sft")
    p.add_argument("--variant_name", type=str, default="default")
    p.add_argument("--max_seq_length", type=int, default=0, help="0 = read HANDOFF / 2048")
    return p.parse_args()


def _read_default_max_seq() -> int:
    handoff = _REPO / "training" / "data" / "HANDOFF.md"
    if not handoff.is_file():
        return DEFAULT_MAX_SEQ
    text = handoff.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "max_seq" in line.lower() and "`" in line:
            try:
                return int(line.split("`")[1])
            except (ValueError, IndexError):
                pass
    return DEFAULT_MAX_SEQ


def _format_row_to_text(row: dict, tokenizer) -> str:
    """Support 'text' column or OpenAI-style messages JSON."""
    if "text" in row and row["text"]:
        return str(row["text"])
    if "messages" in row and row["messages"]:
        msgs = row["messages"]
        if isinstance(msgs, str):
            import json as _j

            msgs = _j.loads(msgs)
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    raise ValueError("Dataset row must have 'text' or 'messages'")


def main() -> None:
    args = _parse_args()
    max_seq = args.max_seq_length or _read_default_max_seq()

    worker_token = os.environ.get("HF_TOKEN", "")
    main_token = os.environ.get("HF_MAIN_TOKEN", "")
    if not worker_token:
        print("warning: HF_TOKEN not set — Hub push and model download may fail.", flush=True)

    # Trackio with main account (read-only space) while training pushes use HF_TOKEN
    try:
        import trackio

        if main_token:
            os.environ.setdefault("HF_TOKEN", worker_token)
        trackio.init(
            project="sevzero-sft",
            space_id="Mist-ic/sevzero-trackio",
            **({"hf_token": main_token} if main_token else {}),
        )
    except Exception as e:
        print(f"trackio init skipped: {e}", flush=True)

    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset(DATASET_ID, split="train")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if USE_4BIT:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("[sft] 4-bit QLoRA mode (set SEVZERO_USE_4BIT=0 for bf16 LoRA)", flush=True)
    else:
        print(f"[sft] bf16 LoRA mode on {BASE_MODEL} (needs ~24 GB free; recommend 80 GB GPU)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def formatting_prompts(examples: dict) -> dict:
        texts = []
        n = len(next(iter(examples.values())))
        keys = list(examples.keys())
        for i in range(n):
            row = {k: (examples[k][i] if k in examples else None) for k in keys}
            texts.append(_format_row_to_text(row, tokenizer))
        return {"text": texts}

    cols = ds.column_names
    if "text" not in ds.column_names:
        if "messages" in ds.column_names:
            ds = ds.map(
                formatting_prompts,
                batched=True,
                remove_columns=[c for c in cols if c not in ("messages",)],
            )
        else:
            raise ValueError("Dataset must include a 'text' or 'messages' column")
    report_backends = ["trackio"]
    try:
        import trackio  # noqa: F401
    except ImportError:
        report_backends = ["none"]

    targs = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=int(os.environ.get("SEVZERO_PER_DEVICE_BS", "2")),
        gradient_accumulation_steps=int(os.environ.get("SEVZERO_GRAD_ACCUM", "8")),
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
        bf16=True,
        seed=args.seed,
        logging_steps=1,
        report_to=report_backends,
        save_total_limit=2,
        max_length=max_seq,  # TRL >=0.21 renamed max_seq_length -> max_length
        gradient_checkpointing=True,
    )

    from transformers import TrainerCallback

    class JsonStepLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            payload = {
                "type": "sft_step",
                "step": state.global_step,
                "loss": logs.get("loss"),
                "lr": logs.get("learning_rate"),
            }
            print(json.dumps(payload, default=str), flush=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=targs,
        train_dataset=ds,
        callbacks=[JsonStepLog()],
    )
    trainer.train()

    if args.push_to_hub_repo:
        print(json.dumps({"event": "push_to_hub", "repo": args.push_to_hub_repo}, default=str), flush=True)
        model.push_to_hub(
            args.push_to_hub_repo,
            token=worker_token or None,
            private=True,
        )
        tokenizer.push_to_hub(
            args.push_to_hub_repo,
            token=worker_token or None,
            private=True,
        )


if __name__ == "__main__":
    main()
