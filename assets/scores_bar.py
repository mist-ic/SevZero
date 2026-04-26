#!/usr/bin/env python3
"""
Grouped bar chart: Easy / Medium / Hard for baseline, SFT, GRPO, frontier.

Expected CSV (header required), from training/eval.py or hand-built:

  task,baseline,sft,grpo,frontier
  easy,0.71,0.85,0.90,0.93
  medium,0.72,0.86,0.91,0.97
  hard,0.60,0.70,0.80,0.887

`task` values: easy, medium, hard (case-insensitive). Numeric columns 0-1.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DPI = 160
OUT_PNG = Path(__file__).resolve().parent / "scores_bar.png"
FIG_W_IN = 1920 / 160
FIG_H_IN = 1080 / 160

STAGES = ("baseline", "sft", "grpo", "frontier")
COLORS = ("#6c757d", "#17a2b8", "#0b3d5c", "#adb5bd")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit("Empty CSV")
        norm = {k.strip().lower(): k for k in r.fieldnames if k and k.strip()}
        for c in STAGES + ("task",):
            if c not in norm:
                raise SystemExit(
                    f"CSV must include columns: task, {', '.join(STAGES)}. Got: {list(r.fieldnames)}"
                )
        rows: list[dict[str, str]] = []
        for row in r:
            d = {k: (row.get(norm[k]) or "").strip() for k in (list(STAGES) + ["task"])}
            rows.append(d)
        return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("eval_results_csv", type=Path)
    p.add_argument("-o", "--output", type=Path, default=OUT_PNG)
    args = p.parse_args()

    raw = load_rows(args.eval_results_csv)
    order = ("easy", "medium", "hard")
    by_task: dict[str, dict[str, float]] = {}
    for row in raw:
        t = row.get("task", "").lower().strip()
        if t not in order:
            continue
        by_task[t] = {s: float(row[s]) for s in STAGES}
    for t in order:
        if t not in by_task:
            by_task[t] = {s: 0.0 for s in STAGES}

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor="white")

    x = np.arange(len(order))
    w = 0.18
    for i, stage in enumerate(STAGES):
        heights = [by_task[tt][stage] for tt in order]
        ax.bar(
            x + (i - 1.5) * w,
            heights,
            width=w,
            label=stage,
            color=COLORS[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in order])
    ax.set_ylabel("Mean score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("SevZero eval — by task and training stage (held-out seeds)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output} ({FIG_W_IN*DPI:.0f}x{FIG_H_IN*DPI:.0f} @ dpi={DPI})")


if __name__ == "__main__":
    main()
