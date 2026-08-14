"""Matched anytime plots for current gain/Beta versus replacement-aware tree prefix."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCENES = {"cross8-nb": "Cross (3 robots)", "chair8": "Chair (4 robots)"}
ARMS = {
    "current_gain_beta": ("Current gain/Beta", "#3974b9"),
    "persistent_tree": ("Replacement-aware tree + current fallback", "#d5662f"),
}


def _load(scene):
    path = HERE / f"{scene}-replacement-audit-persistent-vs-current.json"
    with path.open() as stream:
        return json.load(stream)


def _step(curve, grid, initial):
    x = np.asarray([0.0, *curve["time"]], dtype=float)
    y = np.asarray([initial, *curve["cost"]], dtype=float)
    idx = np.searchsorted(x, grid, side="right") - 1
    return y[np.maximum(idx, 0)]


def make(log_x=False):
    data = {scene: _load(scene) for scene in SCENES}
    fig, axes = plt.subplots(1, len(SCENES), figsize=(12.8, 4.7), squeeze=False)
    for ax, (scene, scene_label) in zip(axes[0], SCENES.items()):
        runs = data[scene]["runs"]
        horizon = min(float(run["curve"]["time"][-1]) for run in runs)
        if log_x:
            positive = [t for run in runs for t in run["curve"]["time"] if t > 0]
            lo = max(1e-3, min(positive))
            grid = np.geomspace(lo, horizon, 500)
        else:
            grid = np.linspace(0.0, horizon, 500)

        for arm, (label, color) in ARMS.items():
            arm_runs = [run for run in runs if run["arm"] == arm]
            ys = np.asarray([
                _step(run["curve"], grid, run["cost_initial"]) for run in arm_runs
            ])
            ax.plot(grid, ys.mean(axis=0), color=color, lw=2.2, label=label)
            ax.fill_between(grid, ys.min(axis=0), ys.max(axis=0), color=color, alpha=0.16)

        if log_x:
            ax.set_xscale("log")
        ax.set_title(scene_label)
        ax.set_xlabel("Shortcut compute time (s)")
        ax.set_ylabel("Path cost")
        ax.grid(alpha=0.25, which="both")
        ax.legend(frameon=False)

    scale = "log x-axis" if log_x else "linear x-axis"
    fig.suptitle(
        f"Replacement-aware coarse-to-fine prefix — 3 matched seeds, 1,000 validations ({scale})\n"
        "line = mean, band = min–max; native cost gate and collision checker unchanged",
        fontsize=12,
    )
    fig.tight_layout()
    suffix = "logx" if log_x else "linear"
    output = ROOT / f"shortcut_persistent_tree_anytime_{suffix}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    make(False)
    make(True)

