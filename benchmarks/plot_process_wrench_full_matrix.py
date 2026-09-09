#!/usr/bin/env python3
"""Plot the full robot-count x part-count x process-wrench benchmark matrices."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/assembly-mpl-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from benchmarks import plot_cube_scaling as base


LAMBDAS = (0.0, 0.01, 0.1, 0.5, 1.0, 2.0)
COLORS = {
    0.0: "#9aa0a6",
    0.01: "#c7e1f2",
    0.1: "#8fc1df",
    0.5: "#5b9bd1",
    1.0: "#3568a8",
    2.0: "#173f73",
}


def _read_results(suite: Path):
  path = suite.resolve() / "results.json"
  if not path.is_file():
    raise ValueError(f"{path}: benchmark results do not exist")
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, list) or not value:
    raise ValueError(f"{path}: no benchmark samples")
  return value


def _aggregate(samples, *, physical: bool):
  grouped = {}
  for sample in samples:
    command = sample.get("planner_command")
    if not isinstance(command, list):
      raise ValueError(f"{sample.get('run_id', '<unknown>')}: missing planner command")
    robots = int(base._flag_value(command, "--num-robots"))
    parts = int(base._flag_value(command, "--parts"))
    lam = float(base._flag_value(command, "--insert-force")) / 10.0 if physical else None
    if physical and lam not in LAMBDAS:
      raise ValueError(f"unexpected insertion-force multiplier lambda={lam:g}")
    grouped.setdefault((robots, parts, lam), []).append(sample)

  rows = []
  for (robots, parts, lam), group in sorted(grouped.items(),
                                             key=lambda item: tuple(-1 if x is None else x
                                                                    for x in item[0])):
    successful = [x for x in group if x.get("success")]

    def mean(key):
      values = [float(x[key]) for x in successful
                if isinstance(x.get(key), (int, float)) and math.isfinite(float(x[key]))]
      return statistics.fmean(values) if values else None

    failed_elapsed = [float(x.get("wall_s", x.get("observed_wall_s", 0.0)))
                      for x in group if not x.get("success")]
    rows.append({
        "parts": parts,
        "robots": robots,
        "lambda": lam,
        "insert_force_N": None if lam is None else 10.0 * lam,
        "runs": len(group),
        "successes": len(successful),
        "failures": len(group) - len(successful),
        "timeouts": sum(bool(x.get("timed_out")) for x in group),
        "milestone_search_s_mean": mean("milestone_search_s"),
        "wall_s_mean": mean("wall_s"),
        "plan_actions_mean": mean("plan_actions"),
        "plan_cost_mean": mean("plan_cost"),
        "ik_bind_wall_s_mean": mean("ik_bind_wall_s"),
        "backward_search_s_mean": mean("backward_search_s"),
        "failed_elapsed_s_max": max(failed_elapsed) if failed_elapsed else None,
    })
  return rows


def _write_csv(path: Path, rows):
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def plot_pair(physical_suite: Path, no_wrench_suite: Path, output_dir: Path, *,
              output_stem: str, title: str, x_label: str,
              metric: str = "milestone_search_s_mean",
              y_label: str = "Time to initial witnessed plan (s)"):
  physical = _aggregate(_read_results(physical_suite), physical=True)
  topological = _aggregate(_read_results(no_wrench_suite), physical=False)
  robots = sorted({int(row["robots"]) for row in physical})
  if robots != [2, 3, 4]:
    raise ValueError(f"expected robot counts 2, 3, 4, got {robots}")
  output_dir.mkdir(parents=True, exist_ok=True)
  _write_csv(output_dir / f"{output_stem}.csv", physical)
  _write_csv(output_dir / f"{output_stem}-no-wrench.csv", topological)

  fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.4), sharey=True)
  for ax, robot_count in zip(axes, robots):
    controls = sorted((row for row in topological if row["robots"] == robot_count),
                      key=lambda row: row["parts"])
    ax.plot([row["parts"] for row in controls],
            [math.nan if row[metric] is None else row[metric] for row in controls],
            color="#111111", linestyle=":", marker="D", linewidth=2.0,
            markersize=4.2)
    for row in controls:
      if row["failures"] and row["failed_elapsed_s_max"]:
        ax.scatter(row["parts"], row["failed_elapsed_s_max"], color="#111111",
                   marker="x", s=45, linewidth=1.7, zorder=4)
    for lam in LAMBDAS:
      series = sorted((row for row in physical
                       if row["robots"] == robot_count and row["lambda"] == lam),
                      key=lambda row: row["parts"])
      ax.plot([row["parts"] for row in series],
              [math.nan if row[metric] is None else row[metric] for row in series],
              color=COLORS[lam], marker="o", linewidth=2.0, markersize=4.2)
      for row in series:
        if row["failures"] and row["failed_elapsed_s_max"]:
          ax.scatter(row["parts"], row["failed_elapsed_s_max"], color=COLORS[lam],
                     marker="x", s=45, linewidth=1.7, zorder=4)
    ax.set_title(f"{robot_count} robots")
    ax.set_xlabel(x_label)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xticks(sorted({int(row["parts"]) for row in physical}))
  axes[0].set_ylabel(y_label)
  handles = [Line2D([], [], color="#111111", linestyle=":", marker="D",
                    label="topological (no wrench)")]
  handles.extend(Line2D([], [], color=COLORS[lam], marker="o",
                        label=("gravity only" if lam == 0 else
                               f"gravity + lambda={lam:g} ({10*lam:g} N)"))
                 for lam in LAMBDAS)
  if any(row["failures"] for row in physical + topological):
    handles.append(Line2D([], [], color="#555555", marker="x", linestyle="None",
                          label="no solution within budget"))
  fig.suptitle(title)
  fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3,
             bbox_to_anchor=(0.5, -0.01))
  fig.tight_layout(rect=(0.0, 0.17, 1.0, 0.96))
  for suffix in ("png", "pdf"):
    fig.savefig(output_dir / f"{output_stem}.{suffix}",
                dpi=180 if suffix == "png" else None, bbox_inches="tight")
  plt.close(fig)
  return physical, topological


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("physical_suite", type=Path)
  parser.add_argument("no_wrench_suite", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--output-stem", required=True)
  parser.add_argument("--title", required=True)
  parser.add_argument("--x-label", default="Number of parts")
  parser.add_argument("--metric", choices=("time", "cost"), default="time")
  args = parser.parse_args(argv)
  metric = ("milestone_search_s_mean", "Time to initial witnessed plan (s)")
  if args.metric == "cost":
    metric = ("plan_cost_mean", "Initial witnessed plan cost")
  plot_pair(args.physical_suite, args.no_wrench_suite, args.output,
            output_stem=args.output_stem, title=args.title, x_label=args.x_label,
            metric=metric[0], y_label=metric[1])
  print(f"[process-wrench-matrix] plot -> "
        f"{args.output.resolve() / (args.output_stem + '.pdf')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
