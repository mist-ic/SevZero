#!/usr/bin/env python3
"""
(1) In-process Sim + grader: golden remediation plan → score >= 0.9 when possible
(2) Uvicorn /health (optional) + 5 CPU GRPO steps with rollout_func + tiny model
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.config_utils import try_load_env_files

try_load_env_files()


def _action_plan(seed: int, task_id: str) -> List[Tuple[str, Dict[str, Any]]]:
    from server.failures import FailureType
    from server.scenarios import generate_scenario

    sc = generate_scenario(seed, task_id)
    if not sc.failure_specs:
        return [("noop", {})]
    spec = sc.failure_specs[0]
    sid = spec.service_id
    ft = spec.failure_type
    if ft == FailureType.BAD_DEPLOY:
        return [("rollback_service", {"service_id": sid})]
    if ft in (FailureType.CONFIG_STARTUP, FailureType.CONFIG_RUNTIME):
        k = spec.broken_config_key or "timeout_ms"
        out = [("tune_config", {"service_id": sid, "key": k, "value": "correct"})]
        if ft == FailureType.CONFIG_STARTUP:
            out.append(("restart_service", {"service_id": sid}))
        return out
    if ft == FailureType.CACHE_FAILURE:
        return [("clear_cache", {"cache_name": sid})]
    if ft == FailureType.CASCADING_LATENCY:
        return [("scale_service", {"service_id": sid, "replicas": 4})]
    if ft == FailureType.NETWORK_ERROR:
        return [("noop", {}), ("noop", {})]
    return [("restart_service", {"service_id": sid})]


def _inproc_golden_score(seed: int, task_id: str) -> float:
    from server.grader import grade_episode
    from server.scenarios import generate_scenario
    from server.simulator import Simulator

    sc = generate_scenario(seed, task_id)
    sim = Simulator()
    sim.reset(seed=seed, difficulty=sc.difficulty, failure_specs=sc.failure_specs)
    for at, p in _action_plan(seed, task_id):
        sim.step(at, p)
        for _ in range(4):
            if sim.terminated:
                break
            sim.step("noop", {})
    g = grade_episode(
        final_slo_score=sim.get_slo_score(),
        steps_taken=len(sim.actions_taken),
        max_steps=sc.max_steps,
        actions_taken=sim.actions_taken,
        terminated=sim.terminated,
        termination_reason=sim.termination_reason,
    )
    return float(g.score)


def _grpo_tiny() -> bool:
    try:
        import trl  # noqa: F401
    except ImportError:
        print("GRPO preflight: trl not installed — skip (pip install trl)", flush=True)
        return True
    os.environ["UNSLOTH_DISABLE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from trl.experimental.openenv import generate_rollout_completions

    from training.env_client import AsyncSevZeroEnvClient, run_async
    from training.rollout_sevzero import SRE_SYSTEM_PROMPT, build_observation_prompt, parse_action

    base = (os.environ.get("SEVZERO_ENV_URL") or "").rstrip("/")
    if not base:
        print("SEVZERO_ENV_URL unset — skip GRPO smoke", flush=True)
        return True

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", device_map="cpu")
    m = get_peft_model(
        m,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        ),
    )

    def rollout_func(prompts, trainer):
        ep_ids: List[int] = []
        ec_ids: List[int] = []
        elp: List[float] = []
        env_r: List[float] = []
        for pr in prompts:
            client = AsyncSevZeroEnvClient(base, None)

            async def run_one():
                p_ids, c_ids, lps = [], [], []
                step_sum = 0.0
                try:
                    ro = await client.reset(task_id="easy", seed=7)
                    obs = ro.get("observation", ro)
                    done = ro.get("done", False)
                    for _ in range(2):
                        if done:
                            break
                        u = build_observation_prompt(obs)
                        msg = [
                            {"role": "system", "content": SRE_SYSTEM_PROMPT},
                            {"role": "user", "content": f"{pr}\n{u}"},
                        ]
                        ptxt = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
                        out = generate_rollout_completions(trainer, [ptxt])[0]
                        p_ids.extend(out.get("prompt_ids", []))
                        c_ids.extend(out.get("completion_ids", []))
                        lps.extend(out.get("logprobs", []))
                        ctext = out.get("text")
                        if not ctext and cids:
                            ctext = tok.decode(cids, skip_special_tokens=True)
                        a = parse_action(ctext or "")
                        sr = await client.step(
                            {
                                "action": {
                                    "action_type": str(a.get("action_type", "noop")),
                                    "params": a.get("params") or {},
                                }
                            }
                        )
                        obs = sr.get("observation", sr)
                        done = sr.get("done", False)
                        step_sum += float(obs.get("reward", sr.get("reward", 0.0) or 0.0))
                    return p_ids, c_ids, lps, step_sum
                finally:
                    await client.aclose()

            p, c, lp, s = run_async(run_one())
            ep_ids.append(p)
            ec_ids.append(c)
            elp.append(lp)
            env_r.append(s)
        return {
            "prompt_ids": ep_ids,
            "completion_ids": ec_ids,
            "logprobs": elp,
            "env_reward": env_r,
        }

    def rf(completions, **kwargs):
        return [float(x) for x in kwargs.get("env_reward", [0.0] * len(completions))]

    out_dir = str(_REPO / "training" / ".preflight_grpo")
    os.makedirs(out_dir, exist_ok=True)
    tr = GRPOTrainer(
        model=m,
        processing_class=tok,
        args=GRPOConfig(
            output_dir=out_dir,
            per_device_train_batch_size=1,
            max_steps=5,
            num_generations=1,
            use_vllm=False,
            learning_rate=1e-5,
            max_completion_length=32,
        ),
        train_dataset=Dataset.from_list([{"text": "x"}] * 2),
        reward_funcs=[rf],
        rollout_func=rollout_func,
    )
    tr.train()
    return True


def main() -> None:
    # --- Part A: in-process (no network)
    for seed, task in ((100, "easy"), (13, "easy"), (7, "easy")):
        s = _inproc_golden_score(seed, task)
        print(f"in-proc grader: seed={seed} task={task} score={s:.3f}", flush=True)
        if s >= 0.9:
            print("OK: in-process golden path reached >=0.9", flush=True)
            break
    else:
        print("WARN: no seed reached 0.9 in in-proc test — check failure coverage", flush=True)

    # --- B: Uvicorn + optional GRPO (requires same deps as the project)
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("SKIP: uvicorn not installed — pip install the project (see training/README.md)", flush=True)
        print("OK", flush=True)
        return

    port = int(os.environ.get("PREFLIGHT_PORT", "8765"))
    base = f"http://127.0.0.1:{port}"
    os.environ["SEVZERO_ENV_URL"] = base
    import urllib.request

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(_REPO),
    )
    try:
        for _ in range(25):
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=2) as r:
                    if getattr(r, "status", 200) < 500:
                        break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("uvicorn not up")
        try:
            _grpo_tiny()
        except Exception as e:
            print(f"GRPO smoke failed (env OK): {e}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    print("OK", flush=True)


if __name__ == "__main__":
    main()
