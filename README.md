# SevZero

**A self-evolving SRE war-room for training on-call AI agents.**

> At step fourteen, an untrained 8B model panicked and restarted the primary database, turning a minor latency spike into a regional outage. 300 steps later, it learned to throttle background jobs instead. This is SevZero.

In R1 we built the foundation; in R2 we turned it into a self-evolving SRE war-room: live curriculum pressure, schema drift, oversight for risky actions, and a training stack that shows up in reward curves, not just pull requests.

---

## Live artifacts (main hosting)

| | |
|:--|:--|
| **HF Space (environment)** | [`huggingface.co/spaces/mist-ic/sevzero-env`](https://huggingface.co/spaces/mist-ic/sevzero-env) |
| **HF Space (Trackio / metrics)** | [`huggingface.co/spaces/mist-ic/sevzero-trackio`](https://huggingface.co/spaces/mist-ic/sevzero-trackio) |
| **HF Model (8B GRPO adapter)** | [`huggingface.co/mist-ic/sevzero-llama3-8b-grpo`](https://huggingface.co/mist-ic/sevzero-llama3-8b-grpo) |
| **HF Dataset (SFT / trajectories)** | [`huggingface.co/datasets/mist-ic/sevzero-expert-trajectories`](https://huggingface.co/datasets/mist-ic/sevzero-expert-trajectories) |
| **Blog (HF)** | `__BLOG_URL__` |
| **Video** | `__VIDEO_URL__` |

---

## What’s new in R2

| Upgrade | What it does (one line) |
|--------|-------------------------|
| **Schema drift** | `inspect_metrics` / `inspect_logs` payloads and keys can change mid-episode; a change log keeps it fair. |
| **Oversight** | High-impact actions (e.g. primary DB, traffic drain) go through a virtual SRE manager: approve, deny, or ask for a safer plan. |
| **Adversarial curriculum** | As rolling reward crosses thresholds, the simulator adds failures, tightens the step budget, and scales topology difficulty. |
| **Fine-grained sub-rewards** | Dense step-wise signals so GRPO does not collapse into zero-advantage groups when SLO movement is small. |

---

## Architecture (conceptual)

```mermaid
flowchart LR
  subgraph Agent
    A[Policy LLM]
  end
  subgraph HTTP
    H[OpenEnv / FastAPI]
  end
  subgraph Environment
    S[Simulator + grader]
    C[Curriculum + adversary]
    O[Oversight / governance]
    D[Schema drift]
  end
  A <--> H
  H <--> S
  H <--> C
  H <--> O
  H <--> D
```

*Source: [`assets/architecture.md`](assets/architecture.md) (mermaid for editing).*

---

## Training pipeline

```mermaid
flowchart LR
  T[Collect expert trajectories\nGemini / Claude / GPT] --> F[SFT\nLlama-3.1-8B-Instruct + LoRA]
  F --> G[GRPO\nremote SevZero / TRL + vLLM]
  G --> M[Model + eval on held-out seeds]
```

*Source: [`assets/training_pipeline.md`](assets/training_pipeline.md).*

---

## Results

**Scores** (held-out eval seeds: **13, 99, 777** — not 42/123/7 from baseline). Replace `__FILL__` when eval lands.

| Task | Baseline 8B | SFT | GRPO | Frontier (Gemini-3.1-Pro) |
|------|------------|-----|------|----------------------------|
| Easy | `__FILL__` | `__FILL__` | `__FILL__` | 0.930 |
| Medium | `__FILL__` | `__FILL__` | `__FILL__` | 0.970 |
| Hard | `__FILL__` | `__FILL__` | `__FILL__` | 0.887 |
| **Mean** | `__FILL__` | `__FILL__` | `__FILL__` | **0.929** |

**Reward curve (GRPO)** — regenerate after each run:

```text
python assets/reward_curve.py <path_to_metrics.jsonl> [--baseline __FILL__]
```

![GRPO reward vs step](assets/reward_curve.png)

**Bar chart (Easy / Medium / Hard)** — from `eval_results.csv` (produced by `training/eval.py`):

```text
python assets/scores_bar.py path/to/eval_results.csv
```

![Scores by task and stage](assets/scores_bar.png)

**Before / after** episode behavior: [`assets/before_after.md`](assets/before_after.md).

---

## Theme and rubric mapping

| Criterion (weight) | How SevZero satisfies it |
|--------------------|--------------------------|
| Environment innovation (40%) | SRE sim + queueing cascades; R2: drift, oversight, curriculum, sub-reward density. |
| Storytelling (30%) | Autopsy hook, blog, short video, README, annotated plots. |
| Reward improvement (20%) | Logged GRPO `metrics.jsonl`, curve + bar + before/after traces. |
| Pipeline (10%) | SFT to GRPO, TRL `rollout_func`, scripts linked below. |
| *Themes* | World modeling (professional): multi-signal state; long-horizon: Hard tier; self-improvement: curriculum; multi-agent: oversight layer. |

---

## Reproducibility

**Install (local)**

```bash
git clone https://github.com/mist-ic/SevZero.git
cd SevZero
uv sync   # or: pip install -e .
```

**Run the environment**

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

**Docker (reset to clean env)**

```bash
docker build -t sevzero .
docker run --rm -p 7860:7860 sevzero
```

**OpenEnv check**

```bash
uv run openenv validate
uv run openenv validate --url http://localhost:7860
```

**Training entrypoints** (see repo `training/` after merge): `collect_trajectories.py`, `build_dataset.py`, `train_sft.py`, `train_grpo.py`, `eval.py`. Colab-friendly paths are documented in the training README inside that package.

**Regenerate story plots**

```bash
python assets/reward_curve.py training/outputs/grpo/metrics.jsonl
python assets/scores_bar.py training/outputs/eval_results.csv
```

---

## Cite

```bibtex
@software{sevzero2026,
  title = {SevZero: A Reinforcement Learning Environment for Site Reliability Engineering},
  author = {SevZero Team},
  year = {2026},
  url = {https://github.com/mist-ic/SevZero}
}
```

---

*Frontier ceiling (Gemini-3.1-Pro, 28-run aggregate): 0.929. Untrained 8B baseline for plots: `__FILL__` (see `metrics.jsonl` + zero-shot eval).*
