#!/usr/bin/env python3
"""Make compact per-assembly anytime/search-baseline plots for the paper.

The black step curve is the sequence of validated incumbents produced by the anytime
planner.  Search variants contribute only their returned solution as a point; the plot
does not pretend that these variants expose an incumbent trajectory.  Marker shape is
deliberately constant: planner identity is encoded only by color.
"""

from __future__ import annotations

import argparse
import json
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
  anytime = {_case_from_anytime(run): run for run in anytime_payload["runs"]}
  groups = _comparison_groups(args.comparison_summary)
  missing = [case for case in args.cases if case not in anytime]
  if missing:
    raise ValueError(f"anytime data is missing cases: {missing}")

  args.output_dir.mkdir(parents=True, exist_ok=True)
  style = (plt.style.context(str(PAPER_STYLE))
           if args.paper and PAPER_STYLE.is_file() else plt.style.context("default"))
  with style:
    for case in args.cases:
      run = anytime[case]
      times = [float(event["time_s"]) for event in run["events"]]
      costs = [float(event["cost"]) for event in run["events"]]
      horizon = float(run["final_time_s"])
      if horizon > times[-1]:
        times.append(horizon)
        costs.append(costs[-1])

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
        for method in METHODS:
          group = groups.get((case, method))
          if not group or not group.get("successes"):
            continue
          elapsed = group.get("search_pipeline_s_mean")
          cost = group.get("plan_cost_mean")
          if elapsed is None or cost is None:
            continue
          ax.scatter(float(elapsed), float(cost), marker="o", s=46,
                     color=COLORS[method], edgecolors="none", zorder=3)

        if args.linear_time:
          ax.set_xlim(0, horizon * 1.015)
        else:
          positive = [x for x in times if x > 0]
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
