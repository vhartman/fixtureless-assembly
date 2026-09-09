#!/usr/bin/env python3
"""Plot topological, gravity-only, and increasing insertion-wrench cube scaling."""

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


LAMBDAS = (0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0)
COLORS = {
    0.0: "#9aa0a6",
    0.001: "#d7e9f6",
    0.01: "#b5d5eb",
    0.1: "#87b9da",
    0.5: "#76a9dc",
    1.0: "#3568a8",
    2.0: "#173f73",
}


def _has_flag(command, flag):
  return any(str(token).split("=", 1)[0] == flag for token in command)


def summarize(results):
  grouped = {}
  for result in results:
    command = result.get("planner_command")
    if not isinstance(command, list):
      raise ValueError(f"{result.get('run_id', '<unknown>')}: missing planner_command")
    parts = int(base._flag_value(command, "--parts"))
    robots = int(base._flag_value(command, "--num-robots"))
    if robots != 2:
      raise ValueError(f"process-wrench experiment requires 2 robots, got {robots}")
    insertion_force = float(base._flag_value(command, "--insert-force"))
    lam = insertion_force / 10.0
    if lam not in LAMBDAS:
      raise ValueError(f"unexpected insertion force {insertion_force:g} N (lambda={lam:g})")
    gravity = not _has_flag(command, "--ignore-gravity")
    grouped.setdefault((gravity, lam, parts), []).append(result)

  rows = []
  for (gravity, lam, parts), samples in sorted(grouped.items()):
    successful = [x for x in samples if x.get("success")]

    def mean_metric(key):
      values = [float(x[key]) for x in successful
                if isinstance(x.get(key), (int, float)) and math.isfinite(float(x[key]))]
      return statistics.fmean(values) if values else None

    elapsed = [float(x.get("wall_s", x.get("observed_wall_s", 0.0)))
               for x in samples if not x.get("success")]
    rows.append({
        "parts": parts,
        "robots": 2,
        "gravity": "on" if gravity else "off",
        "lambda": lam,
        "insert_force_N": 10.0 * lam,
        "runs": len(samples),
        "successes": len(successful),
        "failures": len(samples) - len(successful),
        "timeouts": sum(bool(x.get("timed_out")) for x in samples),
        "milestone_search_s_mean": mean_metric(base.DEFAULT_METRIC),
        "wall_s_mean": mean_metric("wall_s"),
        "plan_actions_mean": mean_metric("plan_actions"),
        "plan_cost_mean": mean_metric("plan_cost"),
        "ik_bind_wall_s_mean": mean_metric("ik_bind_wall_s"),
        "backward_search_s_mean": mean_metric("backward_search_s"),
        "failed_elapsed_s_max": max(elapsed) if elapsed else None,
    })
  return rows


def _no_wrench_rows(suite: Path):
  path = suite.resolve() / "results.json"
  if not path.is_file():
    raise ValueError(f"{path}: no-wrench control results do not exist")
  results = json.loads(path.read_text(encoding="utf-8"))
  rows = base.summarize(results, metric=base.DEFAULT_METRIC)
  unexpected = {int(row["robots"]) for row in rows} - {2}
  if unexpected:
    raise ValueError(f"no-wrench control contains other robot counts: {sorted(unexpected)}")
  return rows


def plot_suite(suite: Path, *, no_wrench_suite: Path | None = None,
               additional_suites=(), output_stem="cube-process-wrench-scaling",
               title="Two-robot cube: topological to physical assembly loads",
               x_label="Number of cube parts"):
  suite = suite.resolve()
  results_path = suite / "results.json"
  if not results_path.is_file():
    raise ValueError(f"{results_path}: benchmark results do not exist")
  results = json.loads(results_path.read_text(encoding="utf-8"))
  for extra in additional_suites:
    path = Path(extra).resolve() / "results.json"
    if not path.is_file():
      raise ValueError(f"{path}: additional benchmark results do not exist")
    results.extend(json.loads(path.read_text(encoding="utf-8")))
  if not isinstance(results, list) or not results:
    raise ValueError(f"{results_path}: no completed benchmark samples")
  all_rows = summarize(results)
  # Historical proof-of-concept suites may contain the discarded gravity-off factorial rows.
  # The paper-facing progression always retains physical gravity and varies process load only.
  rows = [row for row in all_rows if row["gravity"] == "on"]
  if not rows:
    raise ValueError("suite contains no gravity-on process-wrench results")
  no_wrench = _no_wrench_rows(no_wrench_suite) if no_wrench_suite is not None else []

  csv_path = suite / f"{output_stem}.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  if no_wrench:
    control_path = suite / f"{output_stem}-no-wrench.csv"
    with control_path.open("w", newline="", encoding="utf-8") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(no_wrench[0]))
      writer.writeheader()
      writer.writerows(no_wrench)

  fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
  for lam in LAMBDAS:
    series = sorted((row for row in rows if row["lambda"] == lam),
                    key=lambda row: row["parts"])
    if not series:
      continue
    xs = [row["parts"] for row in series]
    ys = [math.nan if row["milestone_search_s_mean"] is None
          else float(row["milestone_search_s_mean"]) for row in series]
    label = "gravity only" if lam == 0.0 else f"gravity + lambda={lam:g} ({10*lam:g} N)"
    ax.plot(xs, ys, color=COLORS[lam], linestyle="-", marker="o", markersize=4.8,
            linewidth=2.1, label=label)
    for row in series:
      elapsed = row["failed_elapsed_s_max"]
      if row["failures"] and elapsed is not None and elapsed > 0:
        ax.scatter([row["parts"]], [elapsed], color=COLORS[lam], marker="x",
                   s=52, linewidth=1.8, zorder=4)

  if no_wrench:
    series = sorted(no_wrench, key=lambda row: int(row["parts"]))
    xs = [int(row["parts"]) for row in series]
    ys = [math.nan if row["milestone_search_s_mean"] is None
          else float(row["milestone_search_s_mean"]) for row in series]
    ax.plot(xs, ys, color="#111111", linestyle=":", marker="D", markersize=4.8,
            linewidth=2.2, label="no wrench (topological)")
    for row in series:
      elapsed = row.get("failed_elapsed_s_max")
      if row["failures"] and elapsed is not None and elapsed > 0:
        ax.scatter([row["parts"]], [elapsed], color="#111111", marker="x",
                   s=52, linewidth=1.8, zorder=4)

  ax.set_xlabel(x_label)
  ax.set_ylabel("Time to initial witnessed plan (s)")
  ax.set_xticks(sorted({int(row["parts"]) for row in rows}))
  ax.set_yscale("log")
  ax.grid(True, which="both", alpha=0.25)
  handles = [Line2D([], [], color=COLORS[x], marker="o", linewidth=2.2,
                    label=("gravity only" if x == 0.0
                           else f"gravity + lambda={x:g} ({10*x:g} N)"))
             for x in LAMBDAS]
  if no_wrench:
    handles.insert(0, Line2D([], [], color="#111", marker="D", linestyle=":",
                             label="topological (no wrench)"))
  if any(row["failures"] for row in rows + no_wrench):
    handles.append(Line2D([], [], color="#555", marker="x", linestyle="None",
                          label="no solution within budget"))
  ax.legend(handles=handles, frameon=False, title="Support and assembly load",
            loc="upper left")
  ax.set_title(title)
  for suffix in ("png", "pdf"):
    fig.savefig(suite / f"{output_stem}.{suffix}",
                dpi=180 if suffix == "png" else None)
  plt.close(fig)
  return rows


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("suite", type=Path)
  parser.add_argument("--no-wrench-suite", type=Path, default=None)
  parser.add_argument("--additional-suite", type=Path, action="append", default=[])
  parser.add_argument("--output-stem", default="cube-process-wrench-scaling")
  parser.add_argument("--title", default="Two-robot cube: topological to physical assembly loads")
  parser.add_argument("--x-label", default="Number of cube parts")
  args = parser.parse_args(argv)
  rows = plot_suite(args.suite, no_wrench_suite=args.no_wrench_suite,
                    additional_suites=args.additional_suite, output_stem=args.output_stem,
                    title=args.title, x_label=args.x_label)
  print(f"[process-wrench] {len(rows)} aggregate points -> "
        f"{args.suite.resolve() / (args.output_stem + '.pdf')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
