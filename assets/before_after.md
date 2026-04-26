# Before / after: episode traces

This is the matched replay artifact behind the final blog table. It is intentionally not a victory-lap trace: on the deterministic held-out seeds, `GRPO-primary` reproduced the same aggregate behavior as the untrained baseline. That negative-control result is the point. SevZero makes it possible to say "the adapter did not change the policy enough" with exact task/seed evidence instead of relying on a demo clip.

| | Untrained (baseline 8B) | GRPO-trained 8B |
|---|------------------------|-------------------|
| **Task / seed** | `hard` / `13` | `hard` / `13` |
| **Final score** | `0.6105` | `0.6105` |
| **Steps used** | `50` / `50` | `50` / `50` |
| **Termination** | `timeout` | `timeout` |

## Untrained: representative hard-tier timeout

The baseline 8B policy can produce valid tool-call syntax and survive the OpenEnv loop, but on the hard tier it does not recover SLO before the action budget expires.

1. `reset(task="hard", seed=13)` starts the same deterministic multi-fault incident used in held-out evaluation.
2. The policy completes the episode without a catastrophic invalid-action failure, but it consumes the full 50-step budget.
3. Final grader components: `slo_recovery=0.6000`, `action_efficiency=1.0000`, `time_efficiency=0.2700`.
4. Final score: `0.6105`, termination reason: `timeout`.

## GRPO-primary: matched scenario

`GRPO-primary` was evaluated on the same task and seed after 120 GRPO steps from the SFT adapter. On this replay, it reaches the same final outcome.

1. `reset(task="hard", seed=13)` starts an identical replay because the simulator is seeded end-to-end.
2. The adapter also consumes the full 50-step budget.
3. Final grader components match the baseline: `slo_recovery=0.6000`, `action_efficiency=1.0000`, `time_efficiency=0.2700`.
4. Final score: `0.6105`, termination reason: `timeout`.

## Why this still matters

The trace is useful because it prevents a common hackathon failure mode: showing a good-looking isolated action sequence while the reproducible held-out eval says something else. The training loop produced nonzero rewards, gradients, and KL movement, but on the deterministic replay seeds the adapter did not alter the evaluated action outcomes. That is the next training target, not a result to hide.

---

**Source rows:** `training/eval.py` schema (`model`, `task`, `seed`, `score`, `slo_recovery`, `action_efficiency`, `time_efficiency`, `steps_used`, `terminated`, `termination_reason`). Baseline row comes from `_diag/eval_done.log`; GRPO-primary uses the confirmed final held-out result mirrored into `Mist-ic/sevzero-eval-results`.
