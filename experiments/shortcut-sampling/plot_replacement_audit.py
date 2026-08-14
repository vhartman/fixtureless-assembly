"""Plot shortcut survival and overwritten-work fractions from replacement audits."""

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
    "cross8-nb": "Cross (3 robots)",
    "chair8": "Chair (4 robots)",
}
COLORS = {"cross8-nb": "#3974b9", "chair8": "#d5662f"}


def load(scene):
    with (HERE / f"{scene}-replacement-audit.json").open() as stream:
        return json.load(stream)


def kaplan_meier(events):
    """Survival of an accepted edit retaining at least one configuration label."""
    durations = np.asarray([e["lifetime_checks"] for e in events], dtype=int)
    observed = np.asarray([e["fully_overwritten"] for e in events], dtype=bool)
    xs, ys = [0], [1.0]
    survival = 1.0
    for t in sorted(set(durations[observed])):
        at_risk = int(np.sum(durations >= t))
        deaths = int(np.sum((durations == t) & observed))
        if at_risk:
            survival *= 1.0 - deaths / at_risk
            xs.append(int(t))
            ys.append(float(survival))
    return xs, ys


def main():
    data = {scene: load(scene) for scene in SCENES}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    ax = axes[0]
    for scene, label in SCENES.items():
        events = [e for run in data[scene]["runs"] for e in run["audit"]["accepted"]]
        x, y = kaplan_meier(events)
        ax.step(x, y, where="post", lw=2.2, color=COLORS[scene], label=label)
    ax.set(xlabel="Collision checks since shortcut acceptance",
           ylabel="Probability edit retains any changed state",
           title="Accepted-shortcut survival (Kaplan–Meier)", ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    metrics = [
        ("Events", "fully_overwritten_event_fraction"),
        ("Immediate\ngain", "fully_overwritten_gain_fraction"),
        ("Successful\nconfig checks", "fully_overwritten_successful_config_fraction"),
        ("Successful\ncheck time", "fully_overwritten_successful_collision_wall_fraction"),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    for offset, (scene, label) in zip((-width / 2, width / 2), SCENES.items()):
        agg = data[scene]["aggregate"]["current_gain_beta"]
        vals = [100.0 * agg[key] for _, key in metrics]
        bars = ax.bar(x + offset, vals, width, color=COLORS[scene], label=label)
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)
    ax.set_xticks(x, [name for name, _ in metrics])
    ax.set(ylabel="Share attributable to fully overwritten edits (%)",
           title="Work and gain that do not survive the run", ylim=(0, 100))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "Current gain/Beta shortcutting — 5 seeds × 1,000 native collision validations",
        fontsize=12.5,
    )
    fig.tight_layout()
    output = ROOT / "shortcut_replacement_audit.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()

