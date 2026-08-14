"""Six-environment standalone and full-planner anytime plots."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENVS = {
    "triple": "Triple waypoints",
    "box": "Box stacking",
    "welding": "Welding",
    "handover": "2D handover",
    "hallway": "Hallway",
    "rearrangement": "Box rearrangement",
    "box_two_arm": "Box stacking (two arms)",
}
POLICIES = {
    "whole_path_random": ("Old whole-path random", "#777777"),
    "native": ("Current gain/Beta", "#3974b9"),
    "persistent_tree": ("Tree prefix + current fallback", "#d5662f"),
    "lazy_portfolio": ("Lazy Thompson portfolio", "#2a9d66"),
    "containment_portfolio": ("Containment-aware portfolio", "#8a4fa3"),
}


def _runs(phase, env, policy):
    out = []
    pattern = HERE / f"live-persistent-v4-{phase}-{env}-{policy}-e*-r*-s*.json"
    for filename in sorted(glob.glob(str(pattern))):
        try:
            with open(filename) as stream:
                data = json.load(stream)
        except Exception:
            continue
        if data.get("protocol") != "shortcut-portfolio-v4-preserve-input-edges":
            continue
        if not data.get("certificate", {}).get("strict_full_edge", False):
            continue
        if phase == "standalone":
            call = data["patch"]["shortcut_calls"][-1]
            setup = float(call.get("trace_setup_offset_seconds", 0.0))
            x = np.concatenate((
                np.asarray([0.0]),
                setup + np.asarray(call["curve"]["time"], dtype=float),
            ))
            y = np.concatenate((
                np.asarray([float(call["input"]["cost"])]),
                np.asarray(call["curve"]["cost"], dtype=float),
            ))
        else:
            x = np.asarray(data["full_curve"]["time"], dtype=float)
            y = np.asarray(data["full_curve"]["cost"], dtype=float)
        if len(x) and len(x) == len(y):
            out.append((x, y, data))
    return out


def _step(x, y, grid):
    idx = np.searchsorted(x, grid, side="right") - 1
    return y[np.maximum(idx, 0)]


def make(phase, log_x=False, allowed_run_ids=None):
    columns = 3
    rows = math.ceil(len(ENVS) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16.2, 4.5 * rows), squeeze=False)
    for ax, (env, title) in zip(axes.flat, ENVS.items()):
        by_policy = {policy: _runs(phase, env, policy) for policy in POLICIES}
        if allowed_run_ids is not None:
            by_policy = {
                policy: [
                    run for run in runs
                    if int(run[2]["run_id"]) in allowed_run_ids
                ]
                for policy, runs in by_policy.items()
            }
        common_runs = set.intersection(*(
            {int(run[2]["run_id"]) for run in runs} for runs in by_policy.values()
        ))
        by_policy = {
            policy: [run for run in runs if int(run[2]["run_id"]) in common_runs]
            for policy, runs in by_policy.items()
        }
        all_runs = [run for runs in by_policy.values() for run in runs]
        if not all_runs:
            ax.set_title(title + " (no common certified seeds)")
            continue
        start = max(float(run[0][0]) for run in all_runs) if phase == "full" else 0.0
        # Hold a completed run at its final incumbent instead of truncating every curve at the
        # fastest-finishing policy.
        finish = max(float(run[0][-1]) for run in all_runs)
        if log_x:
            positive = [float(v) for x, _, _ in all_runs for v in x if v > 0]
            start = max(start, min(positive), 1e-3)
            grid = np.geomspace(start, finish, 500)
        else:
            grid = np.linspace(start, finish, 500)

        for policy, (label, color) in POLICIES.items():
            runs = by_policy[policy]
            ys = np.asarray([_step(x, y, grid) for x, y, _ in runs])
            shown = f"{label} (n={len(runs)})"
            ax.plot(grid, ys.mean(axis=0), color=color, lw=2.0, label=shown)
            ax.fill_between(grid, ys.min(axis=0), ys.max(axis=0), color=color, alpha=0.16)
        if log_x:
            ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Full planner wall time (s)" if phase == "full"
                      else "Post-plan shortcut time (s)")
        ax.set_ylabel("Path cost")
        ax.grid(alpha=0.25, which="both")
        ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat[len(ENVS):]:
        ax.set_visible(False)

    phase_title = ("Full BiRRT* loop with internal shortcut-and-reinject"
                   if phase == "full" else "Standalone post-plan shortcut phase")
    scale = "log x-axis" if log_x else "linear x-axis"
    fig.suptitle(
        f"{phase_title} — fixed environments, matched certified runs ({scale})\n"
        "line = mean, band = min–max; 1,000 final validations",
        fontsize=12,
    )
    fig.tight_layout()
    suffix = "logx" if log_x else "linear"
    run_tag = f"{len(allowed_run_ids)}run" if allowed_run_ids is not None else "10run"
    output = ROOT / f"shortcut_portfolio_mrbench_{run_tag}_{phase}_{suffix}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("standalone", "full", "all"), default="all")
    parser.add_argument("--run-ids", help="optional comma-separated matched run IDs")
    args = parser.parse_args()
    allowed = (
        {int(value) for value in args.run_ids.split(",") if value}
        if args.run_ids else None
    )
    phases = ("standalone", "full") if args.phase == "all" else (args.phase,)
    for phase in phases:
        make(phase, False, allowed)
        make(phase, True, allowed)
