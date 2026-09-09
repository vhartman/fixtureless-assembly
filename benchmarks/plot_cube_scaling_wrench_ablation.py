#!/usr/bin/env python3
"""Overlay cube scaling with wrench checks (solid) and without them (dashed)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
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


COLORS = {2: "#3568a8", 3: "#dd7b1f", 4: "#3a923a"}


def _rows(suite: Path, *, per_part: bool = False):
  results = json.loads((suite / "results.json").read_text(encoding="utf-8"))
  results = base._with_component_metrics(suite, results)
  rows = base.summarize(results, metric=base.DEFAULT_METRIC)
  if per_part:
    _metric, rows = base._per_part_rows(rows, base.DEFAULT_METRIC)
  return rows


def _draw(ax, rows, *, no_wrench: bool, per_part: bool):
  metric = (f"{base.DEFAULT_METRIC}_per_part" if per_part else base.DEFAULT_METRIC)
  for robots in sorted({int(row["robots"]) for row in rows}):
    series = sorted((row for row in rows if int(row["robots"]) == robots),
                    key=lambda row: int(row["parts"]))
    xs = [int(row["parts"]) for row in series]
    ys = [math.nan if row[f"{metric}_mean"] is None
          else float(row[f"{metric}_mean"]) for row in series]
    ax.plot(xs, ys, marker=("s" if no_wrench else "o"), markersize=4.5,
            linewidth=2.0, linestyle=("--" if no_wrench else "-"),
            color=COLORS[robots])
    for row in series:
      if row["failures"] and row.get("failed_elapsed_s_max"):
        ax.scatter([row["parts"]], [row["failed_elapsed_s_max"]], marker="x",
                   s=50, linewidth=1.7, color=COLORS[robots], zorder=4)


def _plot(wrench_suite: Path, no_wrench_suite: Path, output: Path, *, per_part=False):
  wr, nw = _rows(wrench_suite, per_part=per_part), _rows(no_wrench_suite,
                                                         per_part=per_part)
  fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
  _draw(ax, wr, no_wrench=False, per_part=per_part)
  _draw(ax, nw, no_wrench=True, per_part=per_part)
  ax.set_xlabel("Number of cube parts")
  ax.set_ylabel("Time to initial witnessed plan (s/part)" if per_part
                else "Time to initial witnessed plan (s)")
  ax.set_xticks(range(1, 13))
  ax.set_xlim(0.7, 12.3)
  ax.set_ylim(bottom=0)
  ax.grid(True, alpha=0.28)
  robot_handles = [Line2D([], [], color=COLORS[r], marker="o", label=f"{r} robots")
                   for r in (2, 3, 4)]
  physics_handles = [
      Line2D([], [], color="#333", linestyle="-", marker="o", label="wrench checks"),
      Line2D([], [], color="#333", linestyle="--", marker="s", label="no wrench"),
      Line2D([], [], color="#555", linestyle="None", marker="x",
             label="no solution within budget"),
  ]
  legend1 = ax.legend(handles=robot_handles, loc="upper left", frameon=False,
                      title="Robot count")
  ax.add_artist(legend1)
  ax.legend(handles=physics_handles, loc="upper right", frameon=False,
            title="Physics")
  ax.set_title("Cube assembly scaling: wrench-model ablation")
  output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output.with_suffix(".png"), dpi=180)
  fig.savefig(output.with_suffix(".pdf"))
  plt.close(fig)

  rows = []
  for physics, data in (("wrench", wr), ("no_wrench", nw)):
    rows.extend({"physics": physics, **row} for row in data)
  with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("wrench_suite", type=Path)
  parser.add_argument("no_wrench_suite", type=Path)
  parser.add_argument("--output-prefix", type=Path, required=True)
  args = parser.parse_args(argv)
  _plot(args.wrench_suite.resolve(), args.no_wrench_suite.resolve(),
        args.output_prefix.resolve())
  _plot(args.wrench_suite.resolve(), args.no_wrench_suite.resolve(),
        args.output_prefix.resolve().with_name(args.output_prefix.name + "-time-per-part"),
        per_part=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
