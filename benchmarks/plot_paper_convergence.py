#!/usr/bin/env python3
"""Make compact per-assembly anytime/search-baseline plots for the paper.

The black step curve is the across-seed mean validated incumbent cost at each time and its
band is one standard deviation.  Its first point is the mean initial-solution event from those
same anytime runs, with a horizontal one-standard-deviation time bar.  The common-time curve
then begins once every plotted run has a feasible incumbent, so no missing solution is silently
assigned a finite cost. Search variants contribute only their returned solution as a point with
a horizontal one-standard-deviation time bar; the plot does not pretend that these variants
expose an incumbent trajectory.
Marker shape is deliberately constant: planner identity is encoded only by color.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/assembly-mpl-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from benchmarks.plot_anytime_per_assembly_comparison import _case_from_anytime
from benchmarks.run_paper_search_comparison import CASE_LABEL, METHODS


PAPER_STYLE = ROOT / "paper_2.mplstyle"
DEFAULT_CASES = ("cross", "stool", "chair")
LEGEND_METHODS = ("ours", "optimistic", "flat-lazy", "milestones-counting",
                  "ours-exact-fills", "ours-fine-projection")
COLORS = {
    "ours": "#111111",
    "optimistic": "#0072B2",
    "flat-lazy": "#999999",
    "milestones-counting": "#009E73",
    "ours-exact-fills": "#D55E00",
    "ours-fine-projection": "#CC79A7",
}
SHORT_LABEL = {
    "ours": "Ours",
    "optimistic": "Optimistic",
    "flat-lazy": "Flat lazy",
    "milestones-counting": "Counting",
    "ours-exact-fills": "Exact fills",
    "ours-fine-projection": "Fine projection",
}


def _comparison_groups(path: Path) -> dict[tuple[str, str], dict]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
    raise ValueError(
        f"{path}: expected the aggregate summary.json containing a 'groups' list")
  return {(row["case"], row["variant"]): row for row in payload["groups"]}


def _aggregate_anytime(runs: list[dict]) -> dict:
  """Aggregate right-continuous incumbents without inventing pre-solution costs."""
  if not runs:
    raise ValueError("cannot aggregate an empty set of anytime runs")
  for run in runs:
    if not run.get("events"):
      raise ValueError(f"anytime run {run.get('label')!r} has no incumbent")
  initial_times = np.asarray([float(run["events"][0]["time_s"]) for run in runs])
  initial_costs = np.asarray([float(run["events"][0]["cost"]) for run in runs])
  # A cost is undefined before a run's first solution. Begin the timewise aggregate when every
  # seed contributes; the separate first point below summarizes the initial-event distribution.
  # averaging only the lucky early seeds would make the early curve optimistically biased.
  start = max(float(run["events"][0]["time_s"]) for run in runs)
  horizon = min(float(run["final_time_s"]) for run in runs)
  if horizon < start:
    raise ValueError(f"anytime horizons end before all runs have an incumbent: {start:g}s")
  times = sorted({start, horizon, *(
      float(event["time_s"])
      for run in runs for event in run["events"]
      if start <= float(event["time_s"]) <= horizon)})
  values = []
  for run in runs:
    event_times = np.asarray([float(e["time_s"]) for e in run["events"]])
    event_costs = np.asarray([float(e["cost"]) for e in run["events"]])
    index = np.searchsorted(event_times, np.asarray(times), side="right") - 1
    if np.any(index < 0):
      raise AssertionError("aggregate grid precedes a run's first incumbent")
    values.append(event_costs[index])
  matrix = np.asarray(values, dtype=float)
  mean = np.mean(matrix, axis=0)
  sd = (np.std(matrix, axis=0, ddof=1)
        if len(runs) > 1 else np.zeros(matrix.shape[1]))
  initial_time_mean = float(np.mean(initial_times))
  initial_cost_mean = float(np.mean(initial_costs))
  # Make the reported initial solution an actual vertex of the plotted curve.  Between this
  # event-summary point and ``start``, the line is only a visual connection; the shaded
  # timewise aggregate begins where every seed owns an incumbent.
  if initial_time_mean < times[0] - 1e-12:
    times = [initial_time_mean, *times]
    mean = np.concatenate(([initial_cost_mean], mean))
    sd = np.concatenate(([
        float(np.std(initial_costs, ddof=1)) if len(runs) > 1 else 0.0], sd))
  else:
    times[0] = initial_time_mean
    mean[0] = initial_cost_mean
    sd[0] = float(np.std(initial_costs, ddof=1)) if len(runs) > 1 else 0.0
  return {
      "times": np.asarray(times),
      "mean": mean,
      "sd": sd,
      "initial_time_mean": initial_time_mean,
      "initial_time_sd": (float(np.std(initial_times, ddof=1))
                           if len(runs) > 1 else 0.0),
      "initial_cost_mean": initial_cost_mean,
      "start_s": start,
      "horizon_s": horizon,
      "runs": len(runs),
  }


def _write_tikz_legend(path: Path) -> None:
  color_names = {
      "ours": "convergenceOurs",
      "optimistic": "convergenceOptimistic",
      "flat-lazy": "convergenceFlatLazy",
      "milestones-counting": "convergenceCounting",
      "ours-exact-fills": "convergenceExactFills",
      "ours-fine-projection": "convergenceFineProjection",
  }
  lines = [
      "% Standalone legend for the paper convergence figures.",
      "% Requires: \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}",
  ]
  for method in LEGEND_METHODS:
    lines.append(
        f"\\definecolor{{{color_names[method]}}}{{HTML}}{{{COLORS[method][1:].upper()}}}")
  lines.extend([
      "\\begin{tikzpicture}",
      "  \\begin{axis}[",
      "    width=\\columnwidth,",
      "    height=0.17\\columnwidth,",
      "    hide axis,",
      "    xmin=0, xmax=10, ymin=0, ymax=10,",
      "    legend style={",
      "      draw=none, anchor=center, legend cell align=left,",
      "      legend columns=3, at={(axis cs:5,5)}, font=\\scriptsize,",
      "      /tikz/every even column/.append style={column sep=0.45em}",
      "    }",
      "  ]",
  ])
  for method in LEGEND_METHODS:
    if method == "ours":
      image = (f"color={color_names[method]}, solid, line width=1.0pt, "
               "mark=*, mark size=1.35pt")
    else:
      image = (f"color={color_names[method]}, only marks, "
               "mark=*, mark size=1.35pt")
    lines.extend([
        f"    \\addlegendimage{{{image}}}",
        f"    \\addlegendentry{{{SHORT_LABEL[method]}}}",
    ])
  lines.extend(["  \\end{axis}", "\\end{tikzpicture}", ""])
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--anytime", required=True, type=Path,
                      help="all-scenes.json produced by plot_anytime_convergence.py")
  parser.add_argument("--comparison-summary", required=True, type=Path,
                      help="aggregate summary.json from the paper search comparison")
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES),
                      choices=tuple(CASE_LABEL))
  parser.add_argument("--paper", action="store_true",
                      help="use paper_2.mplstyle")
  parser.add_argument("--linear-time", action="store_true",
                      help="use a linear rather than logarithmic time axis")
  args = parser.parse_args(argv)

  anytime_payload = json.loads(args.anytime.read_text(encoding="utf-8"))
  anytime = {}
  for run in anytime_payload["runs"]:
    anytime.setdefault(_case_from_anytime(run), []).append(run)
  groups = _comparison_groups(args.comparison_summary)
  missing = [case for case in args.cases if case not in anytime]
  if missing:
    raise ValueError(f"anytime data is missing cases: {missing}")

  args.output_dir.mkdir(parents=True, exist_ok=True)
  style = (plt.style.context(str(PAPER_STYLE))
           if args.paper and PAPER_STYLE.is_file() else plt.style.context("default"))
  with style:
    for case in args.cases:
      aggregate = _aggregate_anytime(anytime[case])
      times = aggregate["times"]
      costs = aggregate["mean"]
      cost_sd = aggregate["sd"]
      horizon = aggregate["horizon_s"]

      # A near-square source panel is compact enough to use below column width.  Override
      # the 8-inch scaling style's typography proportionally instead of merely squeezing
      # its large labels into the shorter horizontal extent.
      with plt.rc_context({
          "font.size": 15,
          "axes.labelsize": 18,
          "xtick.labelsize": 15,
          "ytick.labelsize": 15,
          "lines.linewidth": 2.2,
      }):
        fig, ax = plt.subplots(figsize=(4.4, 3.05))
        ax.step(times, costs, where="post", color=COLORS["ours"],
                linewidth=2.2, zorder=2)
        if aggregate["runs"] > 1:
          ax.fill_between(times, costs - cost_sd, costs + cost_sd, step="post",
                          color=COLORS["ours"], alpha=0.16, linewidth=0, zorder=1)
        # The Ours marker belongs to the same sample as the curve.  Plotting the separate
        # comparison-run marker here can put it off the curve when either artifact, source
        # revision, or initial-search policy differs.
        ax.errorbar(aggregate["initial_time_mean"], aggregate["initial_cost_mean"],
                    xerr=aggregate["initial_time_sd"], fmt="o", markersize=7.0,
                    color=COLORS["ours"], markeredgewidth=0, elinewidth=1.8,
                    capsize=3.5, capthick=1.8, barsabove=True, zorder=5)
        for method in METHODS:
          if method == "ours":
            continue
          group = groups.get((case, method))
          if not group or not group.get("successes"):
            continue
          elapsed = group.get("search_pipeline_s_mean")
          elapsed_sd = group.get("search_pipeline_s_sd")
          cost = group.get("plan_cost_mean")
          if elapsed is None or cost is None:
            continue
          ax.errorbar(float(elapsed), float(cost),
                      xerr=(None if elapsed_sd is None else float(elapsed_sd)),
                      fmt="o", markersize=7.0, color=COLORS[method], markeredgewidth=0,
                      elinewidth=1.8, capsize=3.5, capthick=1.8,
                      barsabove=True, zorder=4)

        if args.linear_time:
          ax.set_xlim(0, horizon * 1.015)
        else:
          positive = [float(x) for x in times if x > 0]
          positive.extend(
              float(groups[(case, method)]["search_pipeline_s_mean"])
              for method in METHODS
              if groups.get((case, method), {}).get("search_pipeline_s_mean"))
          ax.set_xscale("log")
          ax.set_xlim(max(0.8, min(positive) * 0.72), horizon * 1.04)
        ax.set_xlabel(r"Search time [s]")
        ax.set_ylabel(r"Plan cost")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.grid(True, alpha=0.24, linewidth=0.7)
        # Titles belong in the paper caption; keeping them out saves vertical space.
        prefix = args.output_dir / f"{case}-convergence-paper"
        for suffix in ("pdf", "png"):
          fig.savefig(prefix.with_suffix(f".{suffix}"),
                      dpi=240 if suffix == "png" else None, bbox_inches="tight")
        plt.close(fig)
      print(f"[paper-convergence] {CASE_LABEL[case]} -> {prefix.with_suffix('.pdf')}")

  legend = args.output_dir / "convergence-legend.tex"
  _write_tikz_legend(legend)
  print(f"[paper-convergence] legend -> {legend}")


if __name__ == "__main__":
  main()
