#!/usr/bin/env python3
"""Overlay aggregate anytime convergence and search-comparison solver points.

The anytime curve is the geometric mean incumbent cost over *all* scenes at a common wall-time
budget. It therefore starts only once every scene has produced an incumbent. Comparison points
use the paired geometric means exported by ``run_paper_search_comparison.py``. A hollow marker
means that the corresponding solver failed on at least one scene; its mean is over the displayed
``n/N`` successful subset and is not directly comparable to the all-scene curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/assembly-mpl-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER_STYLE = ROOT / "paper_2.mplstyle"
MARKERS = ("o", "*", "X", "s", "^", "D")


def _geomean(values):
  values = [float(value) for value in values]
  if not values or any(value <= 0 for value in values):
    raise ValueError("geometric means require positive, non-empty values")
  return math.exp(sum(math.log(value) for value in values) / len(values))


def aggregate_anytime(runs):
  """Return a common-budget step curve over every supplied anytime run."""
  if not runs:
    raise ValueError("anytime input has no runs")
  for run in runs:
    if not run.get("events"):
      raise ValueError(f"{run.get('label', '<unnamed>')}: no incumbent events")
  start = max(float(run["events"][0]["time_s"]) for run in runs)
  end = max(float(run["final_time_s"]) for run in runs)
  times = {start, end}
  for run in runs:
    times.update(float(event["time_s"]) for event in run["events"]
                 if start <= float(event["time_s"]) <= end)

  curve = []
  for when in sorted(times):
    costs = []
    for run in runs:
      incumbent = None
      for event in run["events"]:
        if float(event["time_s"]) <= when:
          incumbent = float(event["cost"])
        else:
          break
      if incumbent is None:
        raise AssertionError("curve starts before every scene has an incumbent")
      costs.append(incumbent)
    point = {"time_s": when, "cost_geomean": _geomean(costs)}
    # Suppress times at which no aggregate value changed, except for the final extension.
    if not curve or not math.isclose(point["cost_geomean"], curve[-1]["cost_geomean"]):
      curve.append(point)
    elif math.isclose(when, end):
      curve.append(point)
  return curve


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--anytime", type=Path, required=True,
                      help="all-scenes JSON from plot_anytime_convergence.py")
  parser.add_argument("--comparison", type=Path, required=True,
                      help="comparison-time-cost-geomeans.json")
  parser.add_argument("--output-prefix", type=Path, required=True)
  parser.add_argument("--paper", action="store_true")
  args = parser.parse_args(argv)

  anytime = json.loads(args.anytime.read_text(encoding="utf-8"))["runs"]
  solvers = json.loads(args.comparison.read_text(encoding="utf-8"))
  curve = aggregate_anytime(anytime)
  total = len(anytime)
  args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

  payload = {"definition": "geometric mean incumbent cost over all scenes at common time",
             "scene_count": total, "curve": curve, "solvers": solvers}
  args.output_prefix.with_suffix(".json").write_text(
      json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  with args.output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=("time_s", "cost_geomean"))
    writer.writeheader()
    writer.writerows(curve)

  context = (plt.style.context(str(PAPER_STYLE))
             if args.paper and PAPER_STYLE.is_file() else plt.style.context("default"))
  with context:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.step([point["time_s"] for point in curve],
            [point["cost_geomean"] for point in curve], where="post",
            color="#111111", linewidth=3.0, label=f"Ours anytime ({total}/{total})", zorder=2)
    for index, solver in enumerate(solvers):
      x, y = solver.get("t_init_geomean_s"), solver.get("cost_geomean")
      if x is None or y is None:
        continue
      count = len(solver.get("instances", []))
      color = f"C{index}"
      complete = count == total
      ax.scatter(float(x), float(y), marker=MARKERS[index % len(MARKERS)], s=105,
                 facecolor=color if complete else "none", edgecolor=color, linewidth=2.0,
                 label=f"{solver['label']} ({count}/{total})", zorder=3)
    ax.set_xlabel(r"Search time [s]", fontsize=22)
    ax.set_ylabel(r"Geometric mean cost", fontsize=22)
    ax.tick_params(labelsize=18)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=2, fontsize=11, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), columnspacing=1.1, handletextpad=0.5)
    for suffix in ("pdf", "png"):
      fig.savefig(args.output_prefix.with_suffix(f".{suffix}"),
                  dpi=220 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

  print(f"[anytime-geomean] curve starts at {curve[0]['time_s']:.1f}s "
        f"with cost {curve[0]['cost_geomean']:.2f}, ends at {curve[-1]['time_s']:.1f}s "
        f"with cost {curve[-1]['cost_geomean']:.2f}")
  print(f"[anytime-geomean] plot -> {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
  main()
