#!/usr/bin/env python3
"""
Plot GRPO reward vs step from a metrics.jsonl (one JSON object per line).

Non-negotiable visual bar:
- Faint horizontal dashed: untrained 8B baseline (see --baseline).
- Faint horizontal dashed: frontier ceiling 0.929 (Gemini-3.1-Pro aggregate).
- High-contrast curve: reward mean vs step.
- Shaded region between baseline and the curve, labeled with +learning delta to final point.
- 2-3 inflection markers (slope/peak heuristics); edit captions in ORCHESTRATION when real data lands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Output layout: 1920x1080 at dpi=160
FIG_W_IN = 1920 / 160
FIG_H_IN = 1080 / 160
DPI = 160
OUT_PNG = Path(__file__).resolve().parent / "reward_curve.png"
FRONTIER = 0.929

# Default baseline: Consensus table "weak" aggregate until measured 8B zero-shot is available.
BASELINE_DEFAULT = 0.76

CURVE_COLOR = "#0b3d5c"
FILL_COLOR = "#1f77b4"
FRONTIER_STYLE = {"color": "#b0b0b0", "linestyle": "--", "linewidth": 1.5, "zorder": 1}
BASELINE_STYLE = {"color": "#a0a0a0", "linestyle": "--", "linewidth": 1.5, "zorder": 1}

INFLECTION_CAPTIONS = [
    "Step {step}: inspect-before-restart pattern emerges",
    "Step {step}: steeper SLO recovery segment",
    "Step {step}: policy stabilizes (advantage spread drops)",
]


def _parse_line(obj: dict, line_idx: int) -> tuple[int | None, float | None]:
    step = None
    for k in ("step", "global_step", "train/global_step", "current_step"):
        if k in obj and isinstance(obj[k], (int, float)):
            step = int(obj[k])
            break
    if step is None:
        step = line_idx

    r = None
    for k in (
        "reward_mean",
        "mean_reward",
        "rewards/mean",
        "eval_reward",
        "reward",
    ):
        v = obj.get(k)
        if isinstance(v, (int, float)):
            r = float(v)
            break
    if r is None and "log" in obj:
        # Some exporters nest metrics
        log = obj["log"]
        if isinstance(log, dict):
            for k in ("reward_mean", "mean_reward", "train/reward"):
                if k in log and isinstance(log[k], (int, float)):
                    r = float(log[k])
                    break
    return step, r


def load_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps_list: list[int] = []
    rewards: list[float] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            st, r = _parse_line(obj, i)
            if r is not None:
                steps_list.append(st if st is not None else i)
                rewards.append(r)
    if not rewards:
        raise SystemExit(
            f"No parseable reward fields in {path}. Expected keys like reward_mean, mean_reward, reward."
        )
    order = np.argsort(steps_list)
    s = np.array(steps_list, dtype=int)[order]
    y = np.array(rewards, dtype=float)[order]
    return s, y


def smooth_moving(y: np.ndarray, w: int) -> np.ndarray:
    if w < 2 or len(y) < w:
        return y.astype(float)
    k = np.ones(w, dtype=float) / w
    return np.convolve(y, k, mode="valid")


def inflection_step_indices(
    steps: np.ndarray, rewards: np.ndarray, n_max: int = 3, smooth_win: int = 7
) -> list[int]:
    """Return indices into `steps` for annotation (local max of smoothed d(reward)/d(step))."""
    if len(rewards) < 4:
        return []
    sm = smooth_moving(rewards, min(smooth_win, max(3, len(rewards) // 5)))
    if len(sm) < 3:
        return [len(steps) // 2]
    d = np.diff(sm)
    candidates: list[int] = []
    for j in range(1, len(d) - 1):
        if d[j] > d[j - 1] and d[j] > d[j + 1] and d[j] > 0:
            # map back to full index approx
            off = (len(rewards) - len(d) - 1) // 2
            idx = j + 1 + off
            idx = int(np.clip(idx, 0, len(steps) - 1))
            candidates.append((d[j], idx))
    candidates.sort(key=lambda t: t[0], reverse=True)
    out: list[int] = []
    for _, idx in candidates:
        if idx not in out:
            out.append(idx)
        if len(out) >= n_max:
            break
    if not out and len(steps) > 0:
        out = [len(steps) // 3, 2 * len(steps) // 3][: min(n_max, len(steps))]
    return out[:n_max]


def main() -> None:
    p = argparse.ArgumentParser(description="GRPO reward curve from metrics.jsonl")
    p.add_argument("metrics_jsonl", type=Path, help="Path to metrics.jsonl")
    p.add_argument(
        "-o", "--output", type=Path, default=OUT_PNG, help="Output PNG path"
    )
    p.add_argument(
        "--baseline",
        type=float,
        default=BASELINE_DEFAULT,
        help="Untrained 8B mean reward (replace with measured zero-shot; default 0.76 from weak-model table until filled).",
    )
    p.add_argument(
        "--frontier", type=float, default=FRONTIER, help="Frontier ceiling (default 0.929)"
    )
    p.add_argument(
        "--no-annotations", action="store_true", help="Skip inflection arrows (debug)"
    )
    args = p.parse_args()

    steps, rewards = load_metrics(args.metrics_jsonl)
    last_r = float(rewards[-1])
    delta = last_r - args.baseline

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor="white")

    ax.axhline(
        args.baseline, **BASELINE_STYLE, label=f"Untrained 8B baseline ({args.baseline:.3f})"
    )
    ax.axhline(
        args.frontier, **FRONTIER_STYLE, label=f"Frontier ceiling ({args.frontier:.3f})"
    )
    ax.plot(
        steps,
        rewards,
        color=CURVE_COLOR,
        linewidth=2.5,
        label="GRPO mean reward",
        zorder=3,
    )
    # Shade between baseline and curve (vertical band: improve area between min/max per x)
    y_low = np.minimum(rewards, args.baseline)
    y_high = np.maximum(rewards, args.baseline)
    ax.fill_between(
        steps,
        y_low,
        y_high,
        color=FILL_COLOR,
        alpha=0.22,
        zorder=2,
    )
    ax.text(
        0.02,
        0.12,
        f"learning delta: +{delta:.3f} pts\nto step {int(steps[-1])} reward {last_r:.3f}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#333333", alpha=0.95),
    )
    if not args.no_annotations and len(steps) > 0:
        idxs = inflection_step_indices(steps, rewards, n_max=3)
        for j, i in enumerate(idxs):
            if j >= len(INFLECTION_CAPTIONS):
                break
            sx = int(steps[i])
            sy = float(rewards[i])
            cap = INFLECTION_CAPTIONS[j].format(step=sx)
            ax.annotate(
                cap,
                xy=(sx, sy),
                xytext=(20, 20 + j * 18),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#222222", lw=1.2),
                fontsize=11,
            )

    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (mean)")
    ax.set_title("SevZero GRPO — reward vs step")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output} ({FIG_W_IN*DPI:.0f}x{FIG_H_IN*DPI:.0f} @ dpi={DPI})")


if __name__ == "__main__":
    main()
