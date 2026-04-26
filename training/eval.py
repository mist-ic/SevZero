#!/usr/bin/env python3
"""
Eval: local HF adapters + Gemini (google-genai) + Azure OpenAI + Azure AI Inference.
Writes eval_results.csv; pushes Mist-ic/sevzero-eval-results with HF_MAIN_TOKEN. No Claude.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.config_utils import try_load_env_files
from training.rollout_sevzero import SRE_SYSTEM_PROMPT, build_observation_prompt, parse_action

try_load_env_files()

HELD_OUT = (13, 99, 777)
DEFAULT_TASKS = ("easy", "medium", "hard")
DATASET_HUB = "Mist-ic/sevzero-eval-results"

BUILTIN: Dict[str, str] = {
    "untrained-llama": "base:" + os.environ.get("SEVZERO_BASE_MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct"),
    "sft-primary": os.getenv("SFT_ADAPTER_PRIMARY", "PhaseOfCode/sevzero-llama3-8b-sft"),
    "sft-backup": os.getenv("SFT_ADAPTER_BACKUP", "NoahInOblivion/sevzero-llama3-8b-sft"),
    "sft-innovation": os.getenv("SFT_ADAPTER_INNOVATION", "NoxIsOblivion/sevzero-llama3-8b-sft"),
    "grpo-primary": os.getenv("GRPO_ADAPTER_PRIMARY", "PhaseOfCode/sevzero-llama3-8b-grpo-primary"),
    "grpo-stability": os.getenv("GRPO_ADAPTER_STABILITY", "NoahInOblivion/sevzero-llama3-8b-grpo-stability"),
    "grpo-innovation": os.getenv("GRPO_ADAPTER_INNOVATION", "NoxIsOblivion/sevzero-llama3-8b-grpo-innovation"),
}

AZURE_INF = {
    "grok-4.20-reasoning": "grok-2-latest",
    "kimi-k2.6": "kimi-k2-6-2025",
    "DeepSeek-V3.2": "DeepSeek-V3-2",
}


def run_episode(
    base: str, task: str, seed: int, answer: Callable[[str, str], str]
) -> Dict[str, Any]:
    import httpx

    with httpx.Client(base_url=base.rstrip("/"), timeout=120.0) as client:
        r = client.post("/reset", json={"task_id": task, "seed": seed})
        r.raise_for_status()
        ro = r.json()
        obs = ro.get("observation", ro)
        done = ro.get("done", False)
        user_pfx = f"You are the on-call SRE. task={task!r} seed={seed}.\n\n## Session\n"
        for _ in range(1 + int(obs.get("max_steps", 20))):
            if done:
                break
            user_block = user_pfx + build_observation_prompt(obs)
            text = answer(SRE_SYSTEM_PROMPT, user_block)
            act = parse_action(text)
            sr = client.post(
                "/step",
                json={"action": {"action_type": str(act.get("action_type", "noop")), "params": act.get("params") or {}}},
            )
            sr.raise_for_status()
            out = sr.json()
            obs = out.get("observation", out)
            done = out.get("done", False)
        stt = client.get("/state")
        stt.raise_for_status()
        fs = stt.json()
        g = client.post(
            "/grader",
            json={
                "final_slo_score": float(fs.get("global_slo_score", 0.0)),
                "steps_taken": int(fs.get("step_count", 0)),
                "max_steps": int((obs or {}).get("max_steps", 10)),
                "actions_taken": list((obs or {}).get("actions_taken", [])),
                "terminated": bool(fs.get("terminated", True)),
                "termination_reason": fs.get("termination_reason"),
            },
        )
        js: Dict[str, Any] = {}
        if g.status_code < 400:
            js = g.json()
    return {
        "score": float(js.get("score", 0.0)),
        "slo_recovery": float(js.get("slo_recovery", 0.0)),
        "action_efficiency": float(js.get("action_efficiency", 0.0)),
        "time_efficiency": float(js.get("time_efficiency", 0.0)),
        "steps_used": int(fs.get("step_count", 0)),
        "terminated": fs.get("terminated", True),
        "termination_reason": str(fs.get("termination_reason", "")),
    }


def load_llama_peft(adapter_id: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_id = os.environ.get("SEVZERO_BASE_MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    m = AutoModelForCausalLM.from_pretrained(
        base_id, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN")
    )
    if adapter_id:
        m = PeftModel.from_pretrained(m, adapter_id, token=os.environ.get("HF_TOKEN"))
    m.eval()
    return tok, m


def hf_answer(tok, mdl):
    import torch

    def answer(system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        p = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tok(p, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            o = mdl.generate(**inputs, max_new_tokens=256, do_sample=False)
        gen = o[0, inputs["input_ids"].shape[1] :]
        return tok.decode(gen, skip_special_tokens=True)

    return answer


def answer_gemini(system: str, user: str) -> str:
    from google import genai

    model = os.environ.get(
        "GEMINI_EVAL_MODEL",
        os.environ.get("GEMINI_MODEL_PRO", "gemini-3.1-pro-preview"),
    )
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = c.models.generate_content(model=model, contents=f"{system}\n\n{user}")
    return (r.text or "").strip()


def answer_azure_openai(system: str, user: str) -> str:
    from openai import OpenAI

    ep = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    c = OpenAI(
        api_key=os.environ.get("AZURE_API_KEY", ""),
        base_url=ep + "/openai/v1",
    )
    dep = os.environ.get("AZURE_GPT_DEPLOYMENT", "gpt-5.4-pro")
    r = c.chat.completions.create(
        model=dep,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=512,
    )
    return (r.choices[0].message.content or "").strip()


def answer_azure_inference(model_name: str, system: str, user: str) -> str:
    from azure.ai.inference import ChatCompletionsClient
    from azure.core.credentials import AzureKeyCredential

    ep = os.environ.get("AZURE_AI_INFERENCE_ENDPOINT", "").rstrip("/") + "/"
    c = ChatCompletionsClient(endpoint=ep, credential=AzureKeyCredential(os.environ.get("AZURE_API_KEY", "")))
    r = c.complete(
        model_name=model_name,
        messages=[{"role": "user", "content": f"{system}\n\n{user}"}],
    )
    return (r.choices[0].message.content or "").strip()


def pick_answer_fn(name: str) -> Callable[[str, str], str]:
    n = name.strip()
    if n in BUILTIN:
        spec = BUILTIN[n]
        aid = None if spec.startswith("base:") else spec
        tok, m = load_llama_peft(aid)
        return hf_answer(tok, m)
    if "/" in n and n.count("/") == 1 and not n.startswith("meta-llama/"):
        tok, m = load_llama_peft(n)
        return hf_answer(tok, m)
    if n.startswith("gemini"):
        return answer_gemini
    if "gpt" in n.lower() or n == "gpt-5.4-pro":
        return answer_azure_openai
    if n in AZURE_INF:
        mid = AZURE_INF[n]

        def _fn(s: str, u: str) -> str:
            return answer_azure_inference(mid, s, u)

        return _fn
    raise ValueError(f"Unknown model key: {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="untrained-llama")
    ap.add_argument("--out", type=str, default="eval_results.csv")
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in HELD_OUT))
    ap.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    a = ap.parse_args()

    base = (os.environ.get("SEVZERO_ENV_URL") or "").rstrip("/")
    if not base:
        raise SystemExit("SEVZERO_ENV_URL required")

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    seeds = [int(x) for x in a.seeds.split(",")]
    tasks = [t.strip() for t in a.tasks.split(",")]

    rows: List[Dict[str, Any]] = []
    for mname in models:
        try:
            answer = pick_answer_fn(mname)
        except ValueError as e:
            print(f"SKIP {mname}: {e}", flush=True)
            continue
        for task in tasks:
            for seed in seeds:
                r = run_episode(base, task, seed, answer)
                rows.append(
                    {
                        "model": mname,
                        "task": task,
                        "seed": seed,
                        **r,
                    }
                )
                print(rows[-1], flush=True)

    with Path(a.out).open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "task",
            "seed",
            "score",
            "slo_recovery",
            "action_efficiency",
            "time_efficiency",
            "steps_used",
            "terminated",
            "termination_reason",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    tok_m = os.environ.get("HF_MAIN_TOKEN", "")
    if not tok_m:
        print("HF_MAIN_TOKEN not set — skip Hub push", flush=True)
        return
    from datasets import Dataset

    ds = Dataset.from_list([dict(x) for x in rows])
    ds.push_to_hub(DATASET_HUB, token=tok_m, private=False)
    print(f"OK: pushed hf.co/datasets/{DATASET_HUB}", flush=True)


if __name__ == "__main__":
    main()
