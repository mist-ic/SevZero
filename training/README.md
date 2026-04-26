# SevZero — training (Round 2)

One-liner per script:

- **`train_sft.py`**: SFT on `Mist-ic/sevzero-expert-trajectories` with QLoRA (Unsloth or PEFT fallback) → push adapter with `HF_TOKEN`.
- **`train_grpo.py`**: GRPO with `rollout_func` + remote env (`SEVZERO_ENV_URL`); vLLM colocate, Trackio `Mist-ic/sevzero-trackio`.
- **`eval.py`**: Compare HF adapters and frontier models; write `eval_results.csv`, push `Mist-ic/sevzero-eval-results` with `HF_MAIN_TOKEN`.
- **`preflight.py`**: In-process grader + tiny GRPO smoke (5 steps) on CPU; starts local uvicorn.
- **`launch_hf_job.py`**: `huggingface_hub.run_job` wrapper; `--hardware l40sx1` (verify with `hf jobs hardware`).

## Env files

Load with `python-dotenv` (auto-tried in `config_utils`):

- `hg.env` — `HF_TOKEN` (worker), `HF_MAIN_TOKEN` (Mist-ic, Trackio + eval dataset)
- `api.env` — `GEMINI_API_KEY`, `AZURE_*` for `eval.py`

| Variable | Role |
|----------|------|
| `HF_TOKEN` | Worker: train pushes, private adapter pulls |
| `HF_MAIN_TOKEN` | `Mist-ic`: Trackio + `sevzero-eval-results` only |
| `SEVZERO_ENV_URL` | HTTP base of SevZero Space/ server for GRPO + eval + preflight |
| `GEMINI_API_KEY` | Direct Gemini in eval |
| `AZURE_API_KEY` | Azure OpenAI + Azure AI Inference |
| `AZURE_OPENAI_ENDPOINT` | Deployment base for gpt-5.4-pro |
| `AZURE_AI_INFERENCE_ENDPOINT` | For grok / kimi / DeepSeek in eval |
| `AZURE_API_VERSION` | OpenAI client version header if needed |
| `GEMINI_EVAL_MODEL` | Optional override (default set in `eval.py`) |

## Local debug (from repo root)

```bash
# Install (pin versions in comments / orchestrator)
pip install -e ".[training]"

# SFT
python training/train_sft.py --output_dir ./out/sft --max_steps 10 --push_to_hub_repo "" --variant_name test

# GRPO (remote env required)
$env:SEVZERO_ENV_URL="https://<your-sevzero-space>.hf.space"
python training/train_grpo.py --sft_adapter_repo YOUR/adapters --max_steps 5 --output_dir ./out/grpo
```

## Wave 3 — three GRPO variants (see `playbook/00-orchestration.md`)

Primary (PhaseOfCode):

```bash
python training/train_grpo.py --sft_adapter_repo PhaseOfCode/sevzero-llama3-8b-sft --K 4 --lr 7e-6 --max_steps 350 --variant_name primary
```

Stability (NoahInOblivion):

```bash
python training/train_grpo.py --sft_adapter_repo NoahInOblivion/sevzero-llama3-8b-sft --K 8 --lr 5e-6 --max_steps 350 --variant_name stability
```

Innovation (NoxIsOblivion, env flags on):

```bash
python training/train_grpo.py --sft_adapter_repo NoxIsOblivion/sevzero-llama3-8b-sft --enable_schema_drift --enable_curriculum --K 4 --max_steps 350 --variant_name innovation
```

**HF Job (after merge + public git URL or bucket):**

```bash
$env:HF_TOKEN="<worker>"
$env:SEVZERO_ENV_URL="https://....hf.space"
python training/launch_hf_job.py --script grpo --variant_name primary -- --sft_adapter_repo YOUR/sevzero-llama3-8b-sft
```

**Dependency pins:** run `pip index versions trl openenv-core unsloth` and `python -c "import trl; print(trl.__version__)"` after install; pin in the orchestrator’s lock, not in this file.
