#!/usr/bin/env python3
"""Plot two-robot assembly scaling under uniform gripper-wrench envelope scales."""

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


SCALES = (1, 2, 5, 10, 100)
COLOR = "#3568a8"                 # established two-robot cube-scaling blue
HIGH_FRICTION_COLOR = "#8b4f9f"   # diagnostic overlay, visually separate from scale curves
ALPHA = {1: 1.0, 2: 0.82, 5: 0.64, 10: 0.46, 100: 0.28}


def summarize(results, *, scale_flag="--vacuum-scale"):
  grouped = {}
  for result in results:
    command = result.get("planner_command")
    if not isinstance(command, list):
      raise ValueError(f"{result.get('run_id', '<unknown>')}: missing planner_command")
    parts = int(base._flag_value(command, "--parts"))
    robots = int(base._flag_value(command, "--num-robots"))
    if robots != 2:
      raise ValueError(f"wrench-envelope experiment requires 2 robots, got {robots}")
    scale = float(base._flag_value(command, scale_flag))
    if scale not in SCALES:
      raise ValueError(f"unexpected gripper wrench scale {scale:g}")
    grouped.setdefault((scale, parts), []).append(result)

  rows = []
  for (scale, parts), samples in sorted(grouped.items()):
    values = []
    failed_elapsed = []
    for sample in samples:
      if sample.get("success"):
        value = sample.get(base.DEFAULT_METRIC)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
          raise ValueError(
              f"{sample.get('run_id')}: successful run lacks finite {base.DEFAULT_METRIC}")
        values.append(float(value))
      else:
        failed_elapsed.append(float(sample.get("wall_s",
                                                sample.get("observed_wall_s", 0.0))))
    rows.append({
        "parts": parts,
        "robots": 2,
        "wrench_scale": scale,
        "runs": len(samples),
        "successes": len(values),
        "failures": len(samples) - len(values),
        "timeouts": sum(bool(sample.get("timed_out")) for sample in samples),
        "milestone_search_s_mean": statistics.fmean(values) if values else None,
        "milestone_search_s_sd": statistics.pstdev(values) if len(values) > 1 else (
            0.0 if values else None),
        "failed_elapsed_s_max": max(failed_elapsed) if failed_elapsed else None,
    })
  return rows


def summarize_no_wrench(suite: Path):
  """Read the matched binary control without pretending it is a finite scale."""
  results_path = suite / "results.json"
  if not results_path.is_file():
    raise ValueError(f"{results_path}: no-wrench benchmark results do not exist")
  results = json.loads(results_path.read_text(encoding="utf-8"))
  rows = base.summarize(results, metric=base.DEFAULT_METRIC)
  unexpected = {int(row["robots"]) for row in rows} - {2}
  if unexpected:
    raise ValueError(f"no-wrench control contains non-two-robot results: {sorted(unexpected)}")
  return rows


def summarize_high_friction(suite: Path, *, scale_flag: str):
  """Read a matched finite-wrench suite and verify its table-friction setting."""
  results_path = suite / "results.json"
  if not results_path.is_file():
    raise ValueError(f"{results_path}: high-friction benchmark results do not exist")
  results = json.loads(results_path.read_text(encoding="utf-8"))
  mus = set()
  for result in results:
    command = result.get("planner_command")
    if not isinstance(command, list):
      raise ValueError(f"{result.get('run_id', '<unknown>')}: missing planner_command")
    mu = base._flag_value(command, "--table-friction")
    if mu is None:
      raise ValueError(f"{result.get('run_id')}: high-friction suite lacks --table-friction")
    mus.add(float(mu))
  if len(mus) != 1:
    raise ValueError(f"high-friction suite contains multiple coefficients: {sorted(mus)}")
  rows = summarize(results, scale_flag=scale_flag)
  scales = {float(row["wrench_scale"]) for row in rows}
  if scales != {100.0}:
    raise ValueError(f"high-friction diagnostic must contain only 100x, got {sorted(scales)}")
  return rows, next(iter(mus))


def plot_suite(suite: Path, *, scale_flag="--vacuum-scale",
               output_stem="cube-wrench-envelope-scaling",
               title="Two-robot cube scaling versus grasp wrench capability",
               x_label="Number of cube parts", envelope_label="Vacuum wrench envelope",
               no_wrench_suite: Path | None = None,
               high_friction_suite: Path | None = None):
  suite = suite.resolve()
  results_path = suite / "results.json"
  if not results_path.is_file():
    raise ValueError(f"{results_path}: benchmark results do not exist")
  results = json.loads(results_path.read_text(encoding="utf-8"))
  if not isinstance(results, list) or not results:
    raise ValueError(f"{results_path}: no completed benchmark samples")
  rows = summarize(results, scale_flag=scale_flag)
  no_wrench_rows = (summarize_no_wrench(no_wrench_suite.resolve())
                    if no_wrench_suite is not None else [])
  high_friction_rows, high_friction_mu = (
      summarize_high_friction(high_friction_suite.resolve(), scale_flag=scale_flag)
      if high_friction_suite is not None else ([], None))

  csv_path = suite / f"{output_stem}.csv"
  csv_rows = [{"wrench_setting": f"{row['wrench_scale']:g}x", "table_mu": "", **row}
              for row in rows]
  csv_rows += [{"wrench_setting": "disabled", "wrench_scale": "", "table_mu": "", **row}
               for row in no_wrench_rows]
  csv_rows += [{"wrench_setting": f"100x + table-mu-{high_friction_mu:g}",
                "table_mu": high_friction_mu, **row}
               for row in high_friction_rows]
  with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)

  fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
  # Draw the faint, permissive envelopes first so opaque restrictive curves stay readable.
  for scale in reversed(SCALES):
    series = sorted((row for row in rows if row["wrench_scale"] == scale),
                    key=lambda row: row["parts"])
    if not series:
      continue
    xs = [row["parts"] for row in series]
    ys = [math.nan if row["milestone_search_s_mean"] is None
          else float(row["milestone_search_s_mean"]) for row in series]
    ax.plot(xs, ys, color=COLOR, alpha=ALPHA[scale], marker="o", markersize=4.8,
            linewidth=2.15, label=f"{scale:g}x")
    for row in series:
      elapsed = row.get("failed_elapsed_s_max")
      if row["failures"] and elapsed is not None and elapsed > 0:
        ax.scatter([row["parts"]], [elapsed], color=COLOR, alpha=ALPHA[scale],
                   marker="x", s=52, linewidth=1.8, zorder=4)

  if no_wrench_rows:
    series = sorted(no_wrench_rows, key=lambda row: int(row["parts"]))
    xs = [int(row["parts"]) for row in series]
    ys = [math.nan if row["milestone_search_s_mean"] is None
          else float(row["milestone_search_s_mean"]) for row in series]
    ax.plot(xs, ys, color=COLOR, marker="s", markersize=4.8,
            linewidth=2.15, linestyle="--", label="no wrench")
    for row in series:
      elapsed = row.get("failed_elapsed_s_max")
      if row["failures"] and elapsed is not None and elapsed > 0:
        ax.scatter([row["parts"]], [elapsed], color=COLOR, marker="x",
                   s=52, linewidth=1.8, zorder=4)

  if high_friction_rows:
    series = sorted(high_friction_rows, key=lambda row: int(row["parts"]))
    xs = [int(row["parts"]) for row in series]
    ys = [math.nan if row["milestone_search_s_mean"] is None
          else float(row["milestone_search_s_mean"]) for row in series]
    label = f"100x + table μ={high_friction_mu:g}"
    ax.plot(xs, ys, color=HIGH_FRICTION_COLOR, marker="D", markersize=4.8,
            linewidth=2.15, linestyle="-.", label=label)
    for row in series:
      elapsed = row.get("failed_elapsed_s_max")
      if row["failures"] and elapsed is not None and elapsed > 0:
        ax.scatter([row["parts"]], [elapsed], color=HIGH_FRICTION_COLOR, marker="x",
                   s=52, linewidth=1.8, zorder=4)

  max_parts = max(int(row["parts"])
                  for row in rows + no_wrench_rows + high_friction_rows)
  ax.set_xlabel(x_label)
  ax.xaxis.set_label_coords(0.5, -0.10)
  ax.set_ylabel("Time to initial witnessed plan (s)")
  ax.set_xticks(range(1, max_parts + 1))
  ax.set_xlim(0.7, max_parts + 0.3)
  ax.set_yscale("log")
  ax.grid(True, which="both", alpha=0.25)
  handles = [Line2D([], [], color=COLOR, alpha=ALPHA[s], marker="o",
                    linewidth=2.15, label=f"{s:g}x") for s in SCALES]
  if no_wrench_rows:
    handles.append(Line2D([], [], color=COLOR, marker="s", linewidth=2.15,
                          linestyle="--", label="no wrench"))
  if high_friction_rows:
    handles.append(Line2D([], [], color=HIGH_FRICTION_COLOR, marker="D",
                          linewidth=2.15, linestyle="-.",
                          label=f"100x + table μ={high_friction_mu:g}"))
  if any(row["failures"] for row in rows + no_wrench_rows + high_friction_rows):
    handles.append(Line2D([], [], color="#555555", marker="x", linestyle="None",
                          label="no solution within budget"))
  ax.legend(handles=handles, frameon=False, title=envelope_label)
  ax.set_title(title)
  for suffix in ("png", "pdf"):
    fig.savefig(suite / f"{output_stem}.{suffix}",
                dpi=180 if suffix == "png" else None)
  plt.close(fig)
  return rows


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("suite", type=Path)
  parser.add_argument("--scale-flag", default="--vacuum-scale",
                      choices=("--vacuum-scale", "--two-finger-scale"))
  parser.add_argument("--output-stem", default="cube-wrench-envelope-scaling")
  parser.add_argument("--title", default="Two-robot cube scaling versus grasp wrench capability")
  parser.add_argument("--x-label", default="Number of cube parts")
  parser.add_argument("--envelope-label", default="Vacuum wrench envelope")
  parser.add_argument("--no-wrench-suite", type=Path, default=None,
                      help="matched two-robot suite with wrench checks disabled")
  parser.add_argument("--high-friction-suite", type=Path, default=None,
                      help="matched 100x suite with a fixed high --table-friction")
  args = parser.parse_args(argv)
  rows = plot_suite(args.suite, scale_flag=args.scale_flag,
                    output_stem=args.output_stem, title=args.title,
                    x_label=args.x_label, envelope_label=args.envelope_label,
                    no_wrench_suite=args.no_wrench_suite,
                    high_friction_suite=args.high_friction_suite)
  print(f"[wrench-envelope] {len(rows)} aggregate point(s) -> "
        f"{args.suite.resolve() / (args.output_stem + '.pdf')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
