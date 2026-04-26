# SevZero: from simulator to a trainable SRE war-room (Round 2)

*HF blog draft — no inline hosted images; upload plots separately and replace the placeholders below.*

## The autopsy (hook)

At step fourteen, an untrained 8B model panicked and restarted the primary database, turning a minor latency spike into a regional outage. 300 steps later, it learned to throttle background jobs instead. This is SevZero.

That failure was not a toy bug hunt. In production, the damage lives in a few irreversible actions taken under pressure: wrong service restarted, change applied without a rollback plan, a primary store touched when a leaf service was the root cause. SevZero is built to make those mistakes *expensive* in simulation so policy learning can make them *rare* in policy.

In Round 1 we shipped a deterministic, OpenEnv-native incident simulator: queues, breakers, SLOs, and eight failure types with distinct log signatures. In Round 2 the product is not “more of the same environment.” It is a **self-evolving SRE war-room** — non-stationary observations, an oversight channel for the riskiest tool calls, a curriculum that tightens the incident as the agent’s rolling reward improves, and reward components dense enough for GRPO to see gradients instead of a flat line.

## The environment: what is novel

**Core:** partial observability, delayed effects, and propagation along a service DAG. The agent never sees a labeled root cause. It can only use the same surfaces a human on-call has—metrics, logs, traces—and the same *classes* of actions: `inspect_*` diagnostics, `restart_service`, `rollback_service`, `scale_service`, `tune_config`, `clear_cache`, `rebalance_traffic`, and a few more. That matters: failures propagate through a dependency graph; circuit breakers open and close with delay; a bad restart on an upstream can look like a downstream cache miss until you read the trace.

The scalar score is a blend of SLO recovery, action efficiency, and time under budget. The simulator is **deterministic for a given seed**—`random.Random(seed)` throughout—so a GRPO run that misbehaves is debuggable, and held-out eval seeds are true generalization over topology and failure mix, not replay of the same micro-incident in disguise.

**Round 2 upgrades (implementation-level):**

- **Schema drift** — a middleware path mutates the shapes and keys of `inspect_metrics` and `inspect_logs` responses while exposing a small change log in the observation. Rigid string parsing fails; semantic parsing survives. This tracks real production reality: your dashboards change version without your pager updating first.
- **Oversight** — a virtual SRE manager gates high-blast-radius actions (e.g. touching a primary data plane or draining a region at the wrong time). The model must learn *when* to request approval, not only *what* to type. That maps directly to the “weaker supervisor, stronger worker” story enterprises already run in shadow mode.
- **Adversarial curriculum (lite)** — as rolling performance crosses thresholds, the environment increases failure count, service count, and tightens the step budget. It is a performance-linked escalator, not a long table of hand-authored levels: the *distribution* of incidents shifts as the policy improves.
- **Fine-grained sub-rewards** — early GRPO runs hit a pattern we should own in public: the policy occasionally spammed `inspect_logs` to stay inside dense shaping and avoid committing to a fix. Tightening sub-reward structure—without hiding the real terminal SLO—restored non-zero group variance so GRPO had something to backpropagate.

## The training pipeline: SFT, then GRPO

**Collect:** 100–150 expert-style trajectories from frontier chat models, filtered to a minimum episode score (we used ≥ `__FILL__`).

**SFT:** LoRA on Llama-3.1-8B-Instruct to lock in valid function-call JSON, incident vocabulary, and a “read before you break glass” inductive bias. Approximate run: `__FILL__` steps, effective batch `__FILL__`, LR `1e-5` (see repository training config for the exact file).

**GRPO:** *K* completions per prompt, group-relative advantages, and rollouts that hit the *same* HTTP OpenEnv the judges can open from a Space. The trainer does not get a hand-wavy stub reward: the FastAPI app runs the full tick engine, the grader, and the R2 modules. In TRL, wire custom rollouts through `rollout_func`—`environment_factory` is the legacy path that breaks silent on recent releases.

**Infra in practice:** vLLM (or a compatible server) for fast multi-completion sampling, LoRA on attention and MLP blocks for 8B, cosine LR schedule, and a 30–45 minute *health* window where we watch entropy, KL, and the fraction of steps with near-zero advantage standard deviation. If the curve is flat, the bug is usually integration—not “RL doesn’t work.”

High-level config that matched the GPU hours we had: rank `__FILL__`, LR in the `7e-6`–`1e-5` band, *K* of `4` or `8`, temperature `0.85`, β `0.04`, 300–400 steps. The exact job JSON and dependency pins live next to `train_grpo.py` in the repository.

**Why GRPO, not DPO?** DPO needs a static preference set over pairs; the failure modes here are multi-turn and path-dependent. GRPO’s per-group normalization lets the same prompt explore multiple remediation strategies and learn from the one that actually moves SLO under delayed physics.

**Why 8B?** A 70B API can score near the 0.929 frontier on aggregate benchmarks, but the deployment story for a regulated network is a local policy with auditable weights. The hackathon ask is to show a believable *lift* on that 8B class, not to pretend 8B equals Gemini on every seed.

## Results

**What a judge should see in 10 seconds** — a line that starts near the *measured* untrained-8B floor, steps upward with visible slope changes, and approaches—but may not need to meet—the frontier at **0.929** (Gemini-3.1-Pro, aggregate of 28 reference runs on our protocol). A shaded band between the floor and the curve is the *learning delta* in points, not a decoration.

![GRPO mean reward vs step](path/to/reward_curve.png)

- **Frontier line:** **0.929** (reference aggregate above).
- **Pre-GRPO 8B floor:** `__FILL__` (measured zero-shot on held-out seeds **13, 99, 777** — we deliberately avoid 42/123/7 that appeared in early baselines).
- **Post-GRPO:** `__FILL__` at step `__FILL__` (from `metrics.jsonl`); learning delta `+__FILL__` points in the figure above. Inflection captions are drafted from `assets/reward_curve.py` heuristics and edited against the run log for the final asset.

**Per-tier bars** are more legible to humans than a single scalar. Easy should look boring (everyone is high); *Hard* is where a weak policy collapses. That is the column we expect improvement to show up first if anything does.

![Easy / medium / hard bars](path/to/scores_bar.png)

**Before/after** (same task and seed) is the human-readable twin of the curve: one JSONL line per step with action and observation text. The repository’s `assets/before_after.md` is the working template; the final post will include one medium and one hard excerpt once eval lands.

## Lessons and failure modes (honest)

- **Reward hacking (inspect loop):** a short run spiked by spamming `inspect_logs` to farm dense shaping without remediating. We addressed it with repetition-style penalties in the sub-reward terms and a stronger terminal SLO term so “busy work” could not outscore a resolved incident.
- **Zero-advantage batches:** if every completion in a group gets the same return, GRPO has nothing to differentiate. The fine-grained sub-rewards and curriculum variance exist partly to keep group standard deviation alive.
- **What still breaks:** `__FILL__` (e.g. multi-region + simultaneous independent root causes in the Hard tier) — the honest answer in Q&A is that this is the next curriculum axis, not a reason to hand-wave the current metrics.

## Reuse

- `pip install` / `uv sync` and Docker as in the GitHub `README.md`.
- OpenEnv schema and validation: the Space exposes the same routes evaluators expect.
- **Main Hub links (when live):** [`mist-ic/sevzero-env`](https://huggingface.co/spaces/mist-ic/sevzero-env) · [`mist-ic/sevzero-trackio`](https://huggingface.co/spaces/mist-ic/sevzero-trackio) · [`mist-ic/sevzero-llama3-8b-grpo`](https://huggingface.co/mist-ic/sevzero-llama3-8b-grpo) · [`mist-ic/sevzero-expert-trajectories`](https://huggingface.co/datasets/mist-ic/sevzero-expert-trajectories)

---

Thanks to the OpenEnv team, Hugging Face TRL, and Unsloth for the post-training stack this round actually shipped on.
