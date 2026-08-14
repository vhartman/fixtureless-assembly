"""Ideation-round figure in the style of shortcut_sampling_axes_linear.png:
rows = candidate checks / wall time / configuration checks, columns = the four scenes
(chair8 = held-out, fewer arms), linear axes, min-max band over seeds.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SC = os.path.dirname(os.path.abspath(__file__))
OUT = "/home/valentin/git/postdoc/robotic-venv/assembly/shortcut_sampler_ideation.png"
SCENES = ["car-mini-lim", "car-mini-opt", "cross8-nb", "chair8"]
ARMS = ["baseline", "r2", "sweep_bisect", "sweep_bisect_loc", "gainprop"]
COL = {"baseline": "#444444", "r2": "#d62728", "sweep_bisect": "#17becf",
       "sweep_bisect_loc": "#2ca02c", "gainprop": "#9467bd"}
LBL = {"baseline": "baseline (landed per-run uniform)", "r2": "R2 Kronecker LDS",
       "sweep_bisect": "sweep + blind bisection", "sweep_bisect_loc": "sweep + located bisection",
       "gainprop": "gain-proportional runs"}
AXES = [("pchecks", "candidate collision checks"), ("time", "wall time (s)"),
        ("configs", "configuration checks")]

fig, axes = plt.subplots(3, len(SCENES), figsize=(20.5, 12.0))
for col, s in enumerate(SCENES):
    d = json.load(open(os.path.join(SC, f"{s}-bench3.json")))
    runs = d.get("runs") or d.get("main")
    by = {}
    for r in runs:
        by.setdefault(r["arm"], []).append(r)
    for row, (xkey, xlabel) in enumerate(AXES):
        ax = axes[row][col]
        lo = max(max(r["curve"][xkey][0] for r in rr) for rr in by.values())
        hi = min(min(r["curve"][xkey][-1] for r in rr) for rr in by.values())
        grid = np.linspace(0.0, hi, 240)
        for arm in ARMS:
            rr = by.get(arm)
            if not rr:
                continue
            ys = np.array([np.interp(grid, np.asarray(r["curve"][xkey], float),
                                     np.asarray(r["curve"]["cost"], float),
                                     left=float(r["cost_initial"])) for r in rr])
            ax.plot(grid, ys.mean(axis=0), color=COL[arm], lw=1.9,
                    label=LBL[arm] if (row == 0 and col == 0) else None)
            if len(rr) > 1 and ys.std(axis=0).max() > 1e-9:
                ax.fill_between(grid, ys.min(axis=0), ys.max(axis=0), color=COL[arm],
                                alpha=0.13, lw=0)
        ax.set_xlabel(xlabel)
        if col == 0:
            ax.set_ylabel("path cost")
        if row == 0:
            tag = "  [HELD OUT]" if s == "chair8" else ""
            ax.set_title(f"{s}{tag}  ({d['prepared']['n_states']} states, "
                         f"tol {d['tolerance']:g})", fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=10)
fig.suptitle("Sampler ideation round vs the LANDED baseline — 5 seeds, band = min-max, "
             "linear axes; chair8 was untouched until the final validation", fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 0.965])
fig.savefig(OUT, dpi=145)
print("wrote", OUT)
