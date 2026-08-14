"""One-figure comparison: all mr_bench environments + all real assembly scenes.

New file.  mr_bench data = the v5 standalone phase (matched 1,000-validation final pass,
5 seeds); assembly data = the bench5 runs (same matched budget, 3 seeds).  Both x-axes are
shortcut wall seconds, so the two blocks are directly comparable.
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
ARMS = {
    "native": ("native gain/Beta", "#3974b9", "-"),
    "persistent_tree": ("persistent tree", "#d5662f", "-"),
    "tree_adaptive": ("tree adaptive stop", "#2f8f56", "-."),
    "tree_partial": ("tree partial prefix", "#8a6db8", "--"),
    "tree_warm": ("tree warm prefix", "#b05590", ":"),
}
MRBENCH = (
    "triple", "box", "welding", "handover", "hallway", "rearrangement", "box_two_arm",
)
ASSEMBLY = {
    "chair8": "Chair (4 robots)",
    "cross8-nb": "Cross (3 robots)",
    "car-mini-lim": "Car-mini (limit)",
    "car-mini-opt": "Car-mini (optimized)",
}
MRBENCH_MAP = {
    "triple_waypoints": "triple", "box_stacking": "box", "welding": "welding",
    "2d_handover": "handover", "other_hallway": "hallway",
    "box_rearrangement": "rearrangement",
    "box_stacking_two_robots": "box_two_arm",
}


def _step(x, y, grid):
    idx = np.searchsorted(x, grid, side="right") - 1
    return y[np.maximum(idx, 0)]


def _load_mrbench():
    by = {}
    files = sorted(glob.glob(str(HERE / "live-persistent-v5-standalone-*.json")))
    for filename in files:
        try:
            with open(filename) as stream:
                data = json.load(stream)
        except Exception:
            continue
        cert = data.get("certificate", {})
        if data.get("protocol") != PROTOCOL or not cert.get("strict_full_edge"):
            continue
        env_key = MRBENCH_MAP.get(data["env"].removeprefix("rai."))
        if env_key not in MRBENCH:
            continue
        call = data["patch"]["shortcut_calls"][-1]
        setup = float(call.get("trace_setup_offset_seconds", 0.0))
        t = np.concatenate((np.asarray([0.0]),
                            setup + np.asarray(call["curve"]["time"], dtype=float)))
        c = np.concatenate((
            np.asarray([float(call["input"]["cost"])]),
            np.asarray(call["curve"]["cost"], dtype=float),
        ))
        by.setdefault(env_key, {}).setdefault(data["policy"], []).append((t, c))
    return by


def _load_assembly():
    by = {}
    for scene in ASSEMBLY:
        with (HERE / f"{scene}-bench5.json").open() as stream:
            data = json.load(stream)
        for run in data["runs"]:
            t = np.concatenate((np.asarray([0.0]),
                                np.asarray(run["curve"]["time"], dtype=float)))
            c = np.concatenate((np.asarray([run["cost_initial"]]),
                                np.asarray(run["curve"]["cost"], dtype=float)))
            by.setdefault(scene, {}).setdefault(run["arm"], []).append((t, c))
    return by


def _draw(ax, title, by_env, logx):
    ax.set_title(title, fontsize=9)
    tmax = max(float(t[-1]) for runs in by_env.values() for t, _c in runs)
    if logx:
        positive = [t for runs in by_env.values() for t, _c in runs for t in t if t > 0]
        lo = max(1e-3, min(positive)) if positive else 1e-3
        grid = np.geomspace(lo, tmax, 400)
        ax.set_xscale("log")
    else:
        grid = np.linspace(0.0, tmax, 400)
    for arm, (label, color, ls) in ARMS.items():
        runs = by_env.get(arm)
        if not runs:
            continue
        curves = np.asarray([_step(t, c, grid) for t, c in runs])
        ax.plot(grid, curves.mean(axis=0), color=color, ls=ls, lw=2.0, label=label)
        if len(runs) > 1:
            ax.fill_between(grid, curves.min(axis=0), curves.max(axis=0),
                            color=color, alpha=0.13, linewidth=0)
    ax.set_xlabel("wall time (s)", fontsize=8)
    ax.set_ylabel("path cost", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, which="both")


def make(log_x=False):
    mrb = _load_mrbench()
    asm = _load_assembly()
    fig, axes = plt.subplots(3, 4, figsize=(17.5, 11.5))
    order = list(MRBENCH) + list(ASSEMBLY)
    for ax, key in zip(axes.ravel(), order):
        by_env = mrb.get(key) if key in mrb else asm.get(key, {})
        title = (ASSEMBLY[key] if key in ASSEMBLY else key)
        if by_env:
            _draw(ax, title, by_env, log_x)
        else:
            ax.axis("off")
    axes.ravel()[-1].axis("off")

    handles = [
        plt.Line2D([0], [0], color=color, ls=ls, lw=2.2, label=label)
        for arm, (label, color, ls) in ARMS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    scale = "log x-axis" if log_x else "linear x-axis"
    fig.suptitle(
        f"v5 shortcut portfolio — all mr_bench environments (5 seeds) and all assembly scenes "
        f"(3 seeds), matched 1,000-validation pass, {scale}\n"
        "line = mean, band = min\u2013max; left block = generic benchmark, right block = "
        "real assembly", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    suffix = "logx" if log_x else "linear"
    output = ROOT / f"shortcut_v5_all_{suffix}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    make(False)
    make(True)
