# Before / after: episode traces

Sourced from `training/eval.py` JSONL output (one JSON object per step). **Replace the tables below** with two real runs on the same task and seed: baseline checkpoint vs best GRPO checkpoint, held-out seed.

| | Untrained (baseline 8B) | GRPO-trained 8B |
|---|------------------------|-------------------|
| **Task / seed** | `__FILL__` / `__FILL__` | `__FILL__` / `__FILL__` |
| **Final score** | `__FILL__` | `__FILL__` |
| **Steps used** | `__FILL__` / `__FILL__` | `__FILL__` / `__FILL__` |
| **Termination** | `__FILL__` | `__FILL__` |

## Untrained: representative failure mode

*Draft narrative — align to actual first bad action in JSONL (e.g. high-impact restart without inspection).*

1. `__STEP_0__` — Observation: SLO `__FILL__`, critical services: `__FILL__`.
2. `__STEP_1__` — `inspect_logs` on wrong service; reward noise; no root cause.
3. `__STEP_k__` — `restart_service` on `__FILL__` without approval / wrong target; cascade widens.
4. Late `noop` or thrash; timeout or sub-threshold SLO at end state.

## GRPO: matched scenario

*Draft — show inspect → verify cascade → low-risk fix → optional oversight path.*

1. `__STEP_0__` — Same seed; SLO and topology identical to column one.
2. `__STEP_1–3__` — `inspect_metrics` / `inspect_logs` on `__FILL__` to confirm failure class.
3. `__STEP_4__` — Remediation: `__FILL__` (e.g. `rollback_service`, `tune_config`, or approval flow for primary DB).
4. Recovery ticks; final SLO `__FILL__`; score `__FILL__`.

---

**JSONL field hints for extraction:** for each line, read `observation` / `action` / `reward` / `step` (exact keys follow `eval.py` output). Keep excerpts under 40 lines per column when pasting into the blog or video B-roll.
