"""Upstream span vs tree figure across all 12 benchmark environments (full planner loop).

New file; new outputs (shortcut_v5_upstream_choice_*).  Reads live-persistent-v5-upstream-*.json.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = "shortcut-portfolio-v5-exact-gate"
ENVS = (
    "triple", "box", "welding", "handover", "hallway", "rearrangement",
    "box_two_arm", "eggcartons", "bottles", "pyramid", "mobile_wall_four",
    "spiral_tower_two", "chair8", "cross8-nb", "car-mini-lim", "car-mini-opt",
)
ENV_MAP = {
    "triple_waypoints": "triple", "box_stacking": "box", "welding": "welding",
    "2d_handover": "handover", "other_hallway": "hallway",
    "box_rearrangement": "rearrangement",
    "box_stacking_two_robots": "box_two_arm",
    "eggcartons": "eggcartons", "bottles": "bottles", "pyramid": "pyramid",
    "mobile_wall_four": "mobile_wall_four", "spiral_tower_two": "spiral_tower_two",
    "chair8": "chair8", "cross8-nb": "cross8-nb",
    "car-mini-lim": "car-mini-lim", "car-mini-opt": "car-mini-opt",
}
CHOICES = {
    "span": ("span (historical random)", "#3974b9", "-"),
    "tree": ("tree (structural prefix)", "#d5662f", "-"),
    "whole_random": ("whole-path random", "#8a6db8", "--"),
}


def _step(x, y, grid):
    idx = np.searchsorted(x, grid, side="right") - 1
    return y[np.maximum(idx, 0)]


def _load():
    by = {}
    files = sorted(glob.glob(str(HERE / "live-persistent-v5-upstream-*.json")))
    for filename in files:
        # assembly standalone (-a0-) curves are kept on disk but excluded from this
        # full-loop figure
        if "-a0-" in filename:
            continue
        try:
            with open(filename) as stream:
                data = json.load(stream)
        except Exception:
            continue
        cert = data.get("certificate", {})
        if data.get("protocol") != PROTOCOL or not cert.get("strict_full_edge"):
            continue
        env_key = ENV_MAP.get(data["env"].removeprefix("rai."))
        if env_key not in ENVS:
            continue
        t = np.asarray(data["full_curve"]["time"], dtype=float)
        c = np.asarray(data["full_curve"]["cost"], dtype=float)
        if not len(t) or not len(c):
            continue
        by.setdefault(env_key, {}).setdefault(data["policy"], []).append((t, c))
    return by


def make(log_x=False):
    data = _load()
    fig, axes = plt.subplots(4, 4, figsize=(17.5, 13.0))
    for ax, key in zip(axes.ravel(), ENVS):
        by_env = data.get(key, {})
        if not by_env:
            ax.axis("off")
            continue
        ax.set_title(key, fontsize=9)
        tmax = max(float(t[-1]) for runs in by_env.values() for t, _c in runs)
        if log_x:
            positive = [t for runs in by_env.values() for t, _c in runs for t in t if t > 0]
            lo = max(1e-3, min(positive)) if positive else 1e-3
            grid = np.geomspace(lo, tmax, 400)
            ax.set_xscale("log")
        else:
            grid = np.linspace(0.0, tmax, 400)
        for choice, (label, color, ls) in CHOICES.items():
            runs = by_env.get(choice)
            if not runs:
                continue
            curves = np.asarray([_step(t, c, grid) for t, c in runs])
            ax.plot(grid, curves.mean(axis=0), color=color, ls=ls, lw=2.2, label=label)
            if len(runs) > 1:
                ax.fill_between(grid, curves.min(axis=0), curves.max(axis=0),
                                color=color, alpha=0.14, linewidth=0)
        ax.set_xlabel("wall time (s)", fontsize=8)
        ax.set_ylabel("path cost", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, which="both")
        ax.legend(frameon=False, fontsize=7)
    handles = [
        plt.Line2D([0], [0], color=color, ls=ls, lw=2.2, label=label)
        for choice, (label, color, ls) in CHOICES.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.015))
    scale = "log x-axis" if log_x else "linear x-axis"
    fig.suptitle(
        f"Landed upstream: run_choice span vs tree vs whole-path random — FULL planner loop "
        f"everywhere — 3 seeds, {scale}\nline = mean, band = min\u2013max; every output "
        "carries the strict full-edge certificate", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    suffix = "logx" if log_x else "linear"
    output = ROOT / f"shortcut_v5_upstream_choice_{suffix}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    make(False)
    make(True)
