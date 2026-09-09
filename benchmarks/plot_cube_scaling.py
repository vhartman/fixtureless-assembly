#!/usr/bin/env python3
"""Plot time to first witnessed plan for the cube scaling benchmark.

Only successful runs with the requested timing metric contribute to a line. A timeout or
failed run is drawn as an ``x`` at the elapsed budget and never treated as a solution time.
The companion CSV retains success counts and censoring information for every plotted point.
"""

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

# The plotting module is imported by the runner before its child-process environment exists.
# Keep matplotlib's cache out of a read-only home directory in batch/CI environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/assembly-mpl-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


DEFAULT_METRIC = "milestone_search_s"
METRIC_LABELS = {
    "milestone_search_s": "Time to initial witnessed plan (s)",
    "wall_s": "Search-process wall time (s)",
    "ik_bind_wall_s": "Exact IK / witness validation (s)",
    "backward_search_s": "Backward local-PDB search (s)",
    "residual_search_s": "Remaining milestone search (s)",
}
COMPONENT_METRICS = ("ik_bind_wall_s", "backward_search_s", "residual_search_s")
IK_PRIMITIVES = ("pick", "handover", "add_grasp", "place", "assemble")


def _flag_value(command, flag: str) -> str:
  """Return the last explicit value for a two-token command-line flag."""
  hits = [i for i, token in enumerate(command[:-1]) if token == flag]
  if not hits:
    raise ValueError(f"planner command is missing required {flag}")
  return command[hits[-1] + 1]


def summarize(results, metric=DEFAULT_METRIC):
  """Aggregate seeds without turning failed/censored samples into runtimes."""
  grouped = {}
  for result in results:
    command = result.get("planner_command")
    if not isinstance(command, list):
      raise ValueError(f"{result.get('run_id', '<unknown>')}: missing planner_command list")
    parts = int(_flag_value(command, "--parts"))
    robots = int(_flag_value(command, "--num-robots"))
    grouped.setdefault((robots, parts), []).append(result)

  rows = []
  for (robots, parts), samples in sorted(grouped.items()):
    values = []
    for sample in samples:
      if not sample.get("success"):
        continue
      value = sample.get(metric)
      if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(
            f"{sample.get('run_id', '<unknown>')}: successful run has no finite {metric}; "
            "refusing to substitute a different timing definition")
      values.append(float(value))
    failures = len(samples) - len(values)
    timeouts = sum(bool(sample.get("timed_out")) for sample in samples)
    failed_elapsed = [float(sample.get("wall_s", sample.get("observed_wall_s", 0.0)))
                      for sample in samples if not sample.get("success")]
    rows.append({
        "parts": parts,
        "robots": robots,
        "runs": len(samples),
        "successes": len(values),
        "failures": failures,
        "timeouts": timeouts,
        f"{metric}_mean": statistics.fmean(values) if values else None,
        f"{metric}_sd": statistics.pstdev(values) if len(values) > 1 else (
            0.0 if values else None),
        "failed_elapsed_s_max": max(failed_elapsed) if failed_elapsed else None,
    })
  return rows


def _write_csv(path: Path, rows) -> None:
  fields = list(rows[0]) if rows else ["parts", "robots", "runs", "successes"]
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def _per_part_rows(rows, metric: str):
  """Normalize an aggregate timing curve by the number of parts in each instance."""
  derived_metric = f"{metric}_per_part"
  normalized = []
  for original in rows:
    row = dict(original)
    parts = int(row["parts"])
    if parts <= 0:
      raise ValueError(f"cannot normalize a non-positive part count: {parts}")
    for suffix in ("mean", "sd"):
      source = row.get(f"{metric}_{suffix}")
      row[f"{derived_metric}_{suffix}"] = (
          float(source) / parts if source is not None else None)
    failed_elapsed = row.get("failed_elapsed_s_max")
    if failed_elapsed is not None:
      row["failed_elapsed_s_max"] = float(failed_elapsed) / parts
    normalized.append(row)
  return derived_metric, normalized


def _with_component_metrics(suite_dir: Path, results):
  """Backfill component timings from raw logs and derive the non-overlapping residual.

  Old suites predate ``backward_search_s`` in ``results.json``.  Their raw logs remain the
  source of truth, so plotting them does not require rerunning or rewriting the benchmark.
  """
  # Local import avoids a module cycle: search_runner imports this plotter in the dedicated
  # wrapper, while its stable log parser is exactly the definition these plots should use.
  from benchmarks import search_runner

  enriched = []
  for original in results:
    sample = dict(original)
    artifacts = sample.get("artifacts") or {}
    log_rel = artifacts.get("log")
    if log_rel:
      log_path = suite_dir / log_rel
      if log_path.is_file():
        parsed = search_runner.parse_log(log_path.read_text(encoding="utf-8", errors="replace"))
        keys = ("ik_bind_wall_s", "fill_pdb_reverse_s", "backward_build_s",
                "backward_search_s") + tuple(
                    f"ik_{kind}_{suffix}" for kind in IK_PRIMITIVES
                    for suffix in ("binds", "bind_fail", "bind_s"))
        for key in keys:
          if key in parsed:
            sample.setdefault(key, parsed[key])

    if sample.get("success"):
      total = sample.get("milestone_search_s")
      if not isinstance(total, (int, float)) or not math.isfinite(float(total)):
        raise ValueError(f"{sample.get('run_id')}: cannot decompose missing milestone time")
      # A one-part assembly performs neither a bind nor a local backward fill and therefore
      # prints no corresponding profiler line; its missing components are exactly zero.
      sample.setdefault("ik_bind_wall_s", 0.0)
      sample.setdefault("backward_search_s", 0.0)
      for kind in IK_PRIMITIVES:
        sample.setdefault(f"ik_{kind}_binds", 0)
        sample.setdefault(f"ik_{kind}_bind_fail", 0)
        sample.setdefault(f"ik_{kind}_bind_s", 0.0)
      ik = float(sample["ik_bind_wall_s"])
      backward = float(sample["backward_search_s"])
      residual = float(total) - ik - backward
      # Printed component counters are rounded independently.  Permit only that small error;
      # a materially negative residual indicates overlapping/incompatible timing definitions.
      if residual < -0.11:
        raise ValueError(
            f"{sample.get('run_id')}: timing components exceed milestone time by "
            f"{-residual:.3f}s")
      sample["residual_search_s"] = max(0.0, residual)
    enriched.append(sample)
  return enriched


def _draw_metric(ax, rows, metric, *, log_y=False, show_failures=False):
  colors = {2: "#3568a8", 3: "#dd7b1f", 4: "#3a923a"}
  mean_key, sd_key = f"{metric}_mean", f"{metric}_sd"
  robots_seen = sorted({int(row["robots"]) for row in rows})
  for robots in robots_seen:
    series = sorted((row for row in rows if row["robots"] == robots),
                    key=lambda row: row["parts"])
    xs = [row["parts"] for row in series]
    ys = [float(row[mean_key]) if row[mean_key] is not None else math.nan
          for row in series]
    line, = ax.plot(xs, ys, marker="o", linewidth=2.1, markersize=5,
                    color=colors[robots], label=f"{robots} robots")
    if any(row["successes"] > 1 for row in series):
      lower = [y - float(row[sd_key]) if math.isfinite(y) else math.nan
               for y, row in zip(ys, series)]
      upper = [y + float(row[sd_key]) if math.isfinite(y) else math.nan
               for y, row in zip(ys, series)]
      ax.fill_between(xs, lower, upper, color=line.get_color(), alpha=0.14, linewidth=0)
    if show_failures:
      for row in series:
        if row["failures"]:
          elapsed = row["failed_elapsed_s_max"]
          if elapsed is not None and elapsed > 0:
            ax.scatter([row["parts"]], [elapsed], marker="x", s=54, linewidth=1.8,
                       color=line.get_color(), zorder=4)

  ax.set_xlabel("Number of cube parts")
  ax.set_ylabel(METRIC_LABELS.get(metric, metric))
  ax.set_xticks(range(1, 13))
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.set_xlim(0.7, 12.3)
  ax.set_ylim(bottom=0 if not log_y else None)
  if log_y:
    ax.set_yscale("log")
  ax.grid(True, alpha=0.28)


def plot_components(suite_dir: Path, *, log_y=False):
  """Plot mutually exclusive successful-search timing components in one three-panel figure."""
  suite_dir = suite_dir.resolve()
  results_path = suite_dir / "results.json"
  results = json.loads(results_path.read_text(encoding="utf-8"))
  results = _with_component_metrics(suite_dir, results)
  component_rows = {metric: summarize(results, metric=metric)
                    for metric in COMPONENT_METRICS}

  combined = {}
  for metric, rows in component_rows.items():
    for row in rows:
      key = (row["robots"], row["parts"])
      dest = combined.setdefault(key, {
          k: row[k] for k in ("parts", "robots", "runs", "successes", "failures", "timeouts")
      })
      dest[f"{metric}_mean"] = row[f"{metric}_mean"]
      dest[f"{metric}_sd"] = row[f"{metric}_sd"]
  _write_csv(suite_dir / "cube-scaling-components.csv",
             [combined[key] for key in sorted(combined)])

  fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.3), constrained_layout=True, sharex=True)
  for ax, metric in zip(axes, COMPONENT_METRICS):
    _draw_metric(ax, component_rows[metric], metric, log_y=log_y)
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.035),
             ncol=3, frameon=False)
  fig.suptitle("Cube-part lattice search-time components (successful runs only)", y=1.09)
  fig.savefig(suite_dir / "cube-scaling-components.png", dpi=180, bbox_inches="tight")
  fig.savefig(suite_dir / "cube-scaling-components.pdf", bbox_inches="tight")
  plt.close(fig)
  return component_rows


def plot_primitive_attempts(suite_dir: Path, *, log_y=False):
  """Plot exact RobotIK.bind attempts, split by task primitive.

  These are validation attempts, including repeated open-witness effort, rather than the much
  larger number of symbolic successors generated before lazy validation.
  """
  suite_dir = suite_dir.resolve()
  results = json.loads((suite_dir / "results.json").read_text(encoding="utf-8"))
  results = _with_component_metrics(suite_dir, results)
  metrics = tuple(f"ik_{kind}_binds" for kind in IK_PRIMITIVES)
  rows_by_metric = {metric: summarize(results, metric=metric) for metric in metrics}

  combined = {}
  for kind, metric in zip(IK_PRIMITIVES, metrics):
    fail_metric = f"ik_{kind}_bind_fail"
    fail_rows = {(row["robots"], row["parts"]): row
                 for row in summarize(results, metric=fail_metric)}
    for row in rows_by_metric[metric]:
      key = (row["robots"], row["parts"])
      dest = combined.setdefault(key, {
          k: row[k] for k in ("parts", "robots", "runs", "successes", "failures", "timeouts")
      })
      dest[f"{kind}_binds_mean"] = row[f"{metric}_mean"]
      dest[f"{kind}_binds_sd"] = row[f"{metric}_sd"]
      fr = fail_rows[key]
      dest[f"{kind}_failed_mean"] = fr[f"{fail_metric}_mean"]
      dest[f"{kind}_failed_sd"] = fr[f"{fail_metric}_sd"]
  _write_csv(suite_dir / "cube-scaling-task-attempts.csv",
             [combined[key] for key in sorted(combined)])

  fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.0), constrained_layout=True, sharex=True)
  for ax, kind, metric in zip(axes.flat, IK_PRIMITIVES, metrics):
    _draw_metric(ax, rows_by_metric[metric], metric, log_y=log_y)
    ax.set_ylabel(f"Attempted {kind.replace('_', '-')} binds")
  axes.flat[-1].axis("off")
  handles, labels = axes.flat[0].get_legend_handles_labels()
  axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
  fig.suptitle("Exact task-edge validation attempts by primitive (successful runs only)")
  fig.savefig(suite_dir / "cube-scaling-task-attempts.png", dpi=180)
  fig.savefig(suite_dir / "cube-scaling-task-attempts.pdf")
  plt.close(fig)
  return rows_by_metric


def plot_suite(suite_dir: Path, *, metric=DEFAULT_METRIC, log_y=False):
  suite_dir = suite_dir.resolve()
  results_path = suite_dir / "results.json"
  if not results_path.is_file():
    raise ValueError(f"{results_path}: benchmark results do not exist")
  results = json.loads(results_path.read_text(encoding="utf-8"))
  if not isinstance(results, list) or not results:
    raise ValueError(f"{results_path}: no completed benchmark samples")
  results = _with_component_metrics(suite_dir, results)
  rows = summarize(results, metric=metric)
  robots_seen = sorted({int(row["robots"]) for row in rows})
  unexpected = set(robots_seen) - {2, 3, 4}
  if unexpected:
    raise ValueError(f"unexpected robot counts in cube scaling results: {sorted(unexpected)}")

  _write_csv(suite_dir / "cube-scaling.csv", rows)
  fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
  _draw_metric(ax, rows, metric, log_y=log_y, show_failures=True)
  handles, labels = ax.get_legend_handles_labels()
  if any(row["failures"] for row in rows):
    handles.append(Line2D([], [], color="#555555", marker="x", linestyle="None",
                          markersize=7, markeredgewidth=1.8))
    labels.append("no solution within run budget")
  ax.legend(handles, labels, frameon=False)
  ax.set_title("Cube-part lattice assembly scaling (all-vacuum UR10)")
  fig.savefig(suite_dir / "cube-scaling.png", dpi=180)
  fig.savefig(suite_dir / "cube-scaling.pdf")
  plt.close(fig)

  per_part_metric, per_part_rows = _per_part_rows(rows, metric)
  _write_csv(suite_dir / "cube-scaling-time-per-part.csv", per_part_rows)
  fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
  _draw_metric(ax, per_part_rows, per_part_metric, log_y=log_y, show_failures=True)
  metric_label = METRIC_LABELS.get(metric, metric).removesuffix(" (s)")
  ax.set_ylabel(f"{metric_label} (s/part)")
  handles, labels = ax.get_legend_handles_labels()
  if any(row["failures"] for row in per_part_rows):
    handles.append(Line2D([], [], color="#555555", marker="x", linestyle="None",
                          markersize=7, markeredgewidth=1.8))
    labels.append("no solution within run budget")
  ax.legend(handles, labels, frameon=False)
  ax.set_title("Average initial-solution search time per part")
  fig.savefig(suite_dir / "cube-scaling-time-per-part.png", dpi=180)
  fig.savefig(suite_dir / "cube-scaling-time-per-part.pdf")
  plt.close(fig)
  plot_components(suite_dir, log_y=log_y)
  plot_primitive_attempts(suite_dir, log_y=log_y)
  return rows


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("suite", type=Path,
                      help="completed/partial cube benchmark result directory")
  parser.add_argument("--metric", choices=tuple(METRIC_LABELS), default=DEFAULT_METRIC)
  parser.add_argument("--log-y", action="store_true",
                      help="use a logarithmic time axis (linear is the default)")
  args = parser.parse_args(argv)
  rows = plot_suite(args.suite, metric=args.metric, log_y=args.log_y)
  print(f"[cube-scaling] {len(rows)} aggregate point(s) -> "
        f"{args.suite.resolve() / 'cube-scaling.pdf'} and "
        f"{args.suite.resolve() / 'cube-scaling-time-per-part.pdf'} and "
        f"{args.suite.resolve() / 'cube-scaling-components.pdf'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
