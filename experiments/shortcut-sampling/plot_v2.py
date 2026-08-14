"""Anytime-curve figures for the v5 shortcut round.

For each environment this plots the integrated cost-over-wall curve per policy, averaged over
the available seeds with a min-max band.  Two phases are supported: the standalone final pass
(uses the final shortcut call curve) and the full planner loop (uses the stitched full curve).

New file; the v4 plotting scripts are untouched.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROTOCOL = "shortcut-portfolio-v5-exact-gate"
POLICIES = (
    "native", "persistent_tree", "tree_v2", "tree_adaptive", "tree_partial",
    "tree_warm", "length_uniform", "length_tilted", "max_compatible",
    "max_bandit", "lazy_certify",
)
ENV_KEYS = ("triple", "box", "welding", "handover", "hallway", "rearrangement", "box_two_arm")
LABELS = {
    "native": "native gain/Beta",
    "persistent_tree": "persistent tree",
    "tree_v2": "tree + exact gate",
    "tree_adaptive": "tree adaptive stop",
    "tree_partial": "tree partial prefix",
    "tree_warm": "tree warm prefix",
    "length_uniform": "length uniform",
    "length_tilted": "length tilted",
    "max_compatible": "max compatible",
    "max_bandit": "max bandit",
    "lazy_certify": "lazy certify",
}
COLORS = plt.get_cmap("tab10").colors
STYLE = {
    "native": ("-", 1.8, 0),
    "persistent_tree": ("-", 1.2, 1),
    "tree_v2": ("--", 1.2, 2),
    "tree_adaptive": ("-.", 1.2, 3),
    "tree_partial": ("-", 1.0, 4),
    "tree_warm": (":", 1.0, 5),
    "length_uniform": ("-", 1.0, 6),
    "length_tilted": ("--", 1.0, 7),
    "max_compatible": ("-", 1.2, 8),
    "max_bandit": (":", 1.2, 8),
    "lazy_certify": ("--", 1.0, 9),
}


def _curves_for(data):
    call = data["patch"]["shortcut_calls"][-1]
    setup = float(call.get("trace_setup_offset_seconds", 0.0))
    t = np.concatenate((np.asarray([0.0]), setup + np.asarray(call["curve"]["time"], dtype=float)))
    c = np.concatenate((
        np.asarray([float(call["input"]["cost"])]),
        np.asarray(call["curve"]["cost"], dtype=float),
    ))
    return t, c


def _step(x, y, grid):
    idx = np.searchsorted(x, grid, side="right") - 1
    return y[np.maximum(idx, 0)]


def _load(phase):
    files = sorted(glob.glob(str(HERE / f"live-persistent-v5-{phase}-*.json")))
    by_env: dict[str, dict[str, list]] = {}
    for filename in files:
        try:
            with open(filename) as stream:
                data = json.load(stream)
        except Exception:
            continue
        cert = data.get("certificate", {})
        if (
            data.get("protocol") != PROTOCOL
            or not cert.get("strict_full_edge")
            or not data.get("planner", {}).get("returned_path")
        ):
            continue
        mapping = {
            "triple_waypoints": "triple", "box_stacking": "box", "welding": "welding",
            "2d_handover": "handover", "other_hallway": "hallway",
            "box_rearrangement": "rearrangement",
            "box_stacking_two_robots": "box_two_arm",
        }
        env_key = mapping.get(data["env"].removeprefix("rai."))
        if env_key not in ENV_KEYS:
            continue
        if not data["patch"]["shortcut_calls"]:
            continue
        if phase == "standalone":
            t, c = _curves_for(data)
        else:
            t = np.asarray(data["full_curve"]["time"], dtype=float)
            c = np.asarray(data["full_curve"]["cost"], dtype=float)
            if not len(t) or not len(c):
                continue
        by_env.setdefault(env_key, {}).setdefault(data["policy"], []).append((t, c))
    return by_env


def _draw(ax, env_key, by_env, phase, title_suffix="", logx=False):
    ax.set_title(f"{env_key}{title_suffix}", fontsize=9)
    tmax = 0.0
    tmin = np.inf
    for runs in by_env.values():
        for t, _c in runs:
            tmax = max(tmax, float(t[-1]))
            positive = t[t > 0]
            if len(positive):
                tmin = min(tmin, float(positive[0]))
    if logx:
        grid = np.logspace(np.log10(max(tmin, 1e-4)), np.log10(tmax), 300)
        ax.set_xscale("log")
    else:
        grid = np.linspace(0.0, tmax, 300)
    for policy in POLICIES:
        runs = by_env.get(policy)
        if not runs:
            continue
        curves = np.array([_step(t, c, grid) for t, c in runs])
        mean = curves.mean(axis=0)
        lo = curves.min(axis=0)
        hi = curves.max(axis=0)
        ls, lw, color_i = STYLE[policy]
        color = COLORS[color_i % len(COLORS)]
        ax.plot(grid, mean, ls, color=color, lw=lw, label=LABELS[policy])
        if len(runs) > 1:
            ax.fill_between(grid, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("path cost")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("standalone", "full"), default="standalone")
    args = parser.parse_args()
    by_env = _load(args.phase)
    present = [key for key in ENV_KEYS if by_env.get(key)]
    if not present:
        print("no certified results found")
        return
    panels_a = [key for key in ("triple", "box", "welding", "handover") if key in present]
    panels_b = [key for key in ("hallway", "rearrangement", "box_two_arm") if key in present]
    suffix = " (5-seed mean, min-max band)" if any(
        len(v) > 1 for env in by_env.values() for v in env.values()
    ) else " (single seed)"
    for tag, panels in (("a", panels_a), ("b", panels_b)):
        if not panels:
            continue
        for logx, scale_tag in ((False, ""), (True, "_logx")):
            fig, axes = plt.subplots(
                1, len(panels), figsize=(4.6 * len(panels), 3.6), squeeze=False,
            )
            for ax, env_key in zip(axes[0], panels):
                _draw(ax, env_key, by_env[env_key], args.phase,
                      title_suffix=suffix, logx=logx)
            out = HERE / f"shortcut_v5_{args.phase}_{tag}{scale_tag}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=160)
            plt.close(fig)
            print("wrote", out)


if __name__ == "__main__":
    main()
