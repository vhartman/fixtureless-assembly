"""Hybrid-arm figure: can we recover native's early drop while keeping the tree's final cost?

New file; new outputs (shortcut_v5_hybrid_*).  Reads *-bench5-hybrid.json (4 assembly
scenes, 3 seeds).  Arms: native, tree_simple, tree_rand, tree_cap, tree_warmbeta.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCENES = {
    "chair8": "Chair (4 robots)",
    "cross8-nb": "Cross (3 robots)",
    "car-mini-lim": "Car-mini (limit)",
    "car-mini-opt": "Car-mini (optimized)",
}
ARMS = {
    "native": ("native gain/Beta", "#3974b9", "-"),
    "tree_simple": ("tree simple (Beta fallback)", "#d5662f", "-"),
    "tree_rand": ("tree random fallback", "#2f8f56", "-."),
    "tree_cap": ("tree cap-40 seed", "#8a6db8", "--"),
    "tree_warmbeta": ("tree beta-warmup 50", "#c9a227", (0, (5, 3))),
}


def _step(curve, grid, initial):
    x = np.asarray([0.0, *curve["time"]], dtype=float)
    y = np.asarray([initial, *curve["cost"]], dtype=float)
    idx = np.searchsorted(x, grid, side="right") - 1
    return y[np.maximum(idx, 0)]


def make(log_x=False):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    for ax, (scene, label) in zip(axes.ravel(), SCENES.items()):
        with (HERE / f"{scene}-bench5-hybrid.json").open() as stream:
            data = json.load(stream)
        runs = data["runs"]
        horizon = min(float(run["curve"]["time"][-1]) for run in runs)
        if log_x:
            positive = [t for run in runs for t in run["curve"]["time"] if t > 0]
            lo = max(1e-3, min(positive))
            grid = np.geomspace(lo, horizon, 400)
            ax.set_xscale("log")
        else:
            grid = np.linspace(0.0, horizon, 400)
        for arm, (arm_label, color, ls) in ARMS.items():
            arm_runs = [run for run in runs if run["arm"] == arm]
            if not arm_runs:
                continue
            ys = np.asarray([
                _step(run["curve"], grid, run["cost_initial"]) for run in arm_runs
            ])
            ax.plot(grid, ys.mean(axis=0), color=color, ls=ls, lw=2.2, label=arm_label)
            ax.fill_between(grid, ys.min(axis=0), ys.max(axis=0), color=color, alpha=0.13)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("wall time (s)", fontsize=9)
        ax.set_ylabel("path cost", fontsize=9)
        ax.grid(alpha=0.25, which="both")
        ax.legend(frameon=False, fontsize=8)
    scale = "log x-axis" if log_x else "linear x-axis"
    fig.suptitle(
        f"Hybrid arms: recovering the native early drop — 3 seeds, 1,000 validations "
        f"({scale})\nline = mean, band = min\u2013max", fontsize=12)
    fig.tight_layout()
    suffix = "logx" if log_x else "linear"
    output = ROOT / f"shortcut_v5_hybrid_{suffix}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    make(False)
    make(True)
