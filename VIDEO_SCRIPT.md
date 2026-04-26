# SevZero R2 — video script (~110–130 s, under 2 min)

**On-screen text (0:00):** `SevZero` · `A self-evolving SRE war-room for on-call agents`

**0:00–0:15 — Autopsy hook**  
*Spoken (~55 words):*  
“At step fourteen, an untrained 8B model panicked and restarted the primary database, turning a minor latency spike into a regional outage. 300 steps later, it learned to throttle background jobs instead. This is SevZero — a trainable SRE environment where the mistakes are expensive so the policy can become safe.”

`[Brackets — visual: full-screen terminal or Space UI; one hard cut on “primary database” to a red SLO readout; no B-roll over the hook line.]`

**On-screen (0:12):** `R1: foundation` → `R2: self-evolving war-room`

---

**0:15–0:45 — What it is + four R2 upgrades**  
*Spoken (~100 words):*  
“In round one we built the foundation — a deterministic OpenEnv for cascading microservice failures with queueing-theory propagation. In round two we productized: schema drift in observability APIs so brittle parsers die and semantic readers live; a virtual SRE manager that must approve the highest-blast actions; a curriculum that makes incidents harder as your rolling reward improves; and sub-reward structure so GRPO sees real gradients, not mode collapse. Same HTTP surface the judges can hit from our Space — same seeds, stricter world.”

`[Brackets — visual: `assets/architecture.md` mermaid or exported diagram; four quick labels on screen matching drift / oversight / curriculum / sub-rewards. Pace: ~5–7 s per upgrade.]`

**On-screen (each ~4 s):** `Schema drift` · `Oversight` · `Adversarial curriculum` · `Fine-grained sub-rewards`

---

**0:45–1:10 — Training + evidence**  
*Spoken (~95 words):*  
“We collected expert runs from frontier models, SFT-warmed Llama-3.1-8B on LoRA, then ran GRPO through the live environment with group-relative advantages — not a static DPO pair dataset. The curve you care about is mean reward against training step: a floor for the untrained 8B, a ceiling at 0.929 from Gemini on our reference aggregate, and our run climbing in between. The shaded area is the learning delta in points. Inflections line up with inspect-then-act behavior instead of random restarts.”

`[Brackets — visual: `assets/reward_curve.png` full width; pointer or circle on shaded delta and two inflection callouts. Optional split: left half = one bad step trace, right half = trained trace — from `assets/before_after.md`.]`

**On-screen:** `SFT → GRPO` · `K rollouts / group` · `+Δ = __FILL__ pts` *(replace at H+15)*

---

**1:10–1:25 — Capstone + links**  
*Spoken (~60 words):*  
“This is now a reusable benchmark: environment on Hugging Face, Trackio for metrics, 8B adapter on the Hub, open training scripts, and a dataset of expert trajectories. Install with pip or pull the container — validate with OpenEnv — reproduce the curves. SevZero is the room where the next on-call model trains before it touches your graph.”

`[Brackets — visual: static end card with QR or URLs — `mist-ic/sevzero-env`, `mist-ic/sevzero-trackio`, `mist-ic/sevzero-llama3-8b-grpo`, `mist-ic/sevzero-expert-trajectories` — and GitHub.]*

**On-screen (end card):** `Space` · `Trackio` · `Model` · `Dataset` · `github.com/mist-ic/SevZero`

---

**Total:** ~320 words (comfort band 280–360); trim the middle paragraph by ~20 words if the VO runs long.

**Audio note:** one music bed allowed under VO at -18 dB; duck to silence on the autopsy first sentence if using music.
