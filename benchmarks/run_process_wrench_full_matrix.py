#!/usr/bin/env python3
"""Run the full piecewise-Cube and Cross process-wrench matrices."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks import plot_process_wrench_full_matrix as plotter  # noqa: E402
from benchmarks import search_runner  # noqa: E402

MANIFEST_DIR = ROOT / "benchmarks" / "manifests"
EXPERIMENTS = {
    "cross": {
        "physical": MANIFEST_DIR / "cross_process_wrench_full_matrix.json",
        "control": MANIFEST_DIR / "cross_process_wrench_full_no_wrench.json",
        "optimistic_control": MANIFEST_DIR / "cross_process_wrench_full_optimistic_no_wrench.json",
        "stem": "cross-process-wrench-full-matrix",
        "title": "Cross: topological to physical assembly loads",
        "x_label": "Number of Cross parts",
    },
    "cube": {
        "physical": MANIFEST_DIR / "cube_piecewise_process_wrench_full_matrix.json",
        "control": MANIFEST_DIR / "cube_piecewise_process_wrench_full_no_wrench.json",
        "optimistic_control": MANIFEST_DIR / "cube_piecewise_process_wrench_full_optimistic_no_wrench.json",
        "stem": "cube-piecewise-process-wrench-full-matrix",
        "title": "Piecewise cube: topological to physical assembly loads",
        "x_label": "Number of cube parts",
    },
}


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--assembly", choices=("all", "cube", "cross"), default="all")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--repeats", type=int, default=1, metavar="N",
                      help="independent repetitions per matrix cell (default: 1)")
  parser.add_argument(
      "--seeds", type=int, nargs="+", default=None, metavar="SEED",
      help=("independent planner seeds, replacing the seed list in every matrix manifest; "
            "unlike --repeats, these change the randomized search instance"))
  parser.add_argument(
      "--jobs", type=int, default=1, metavar="N",
      help=("run up to N different part-count cases concurrently within each matrix "
            "(default: 1)"))
  parser.add_argument("--timeout", type=float, default=None, metavar="SEC")
  parser.add_argument("--only", default=None, metavar="REGEX")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--no-plot", action="store_true")
  parser.add_argument(
      "--optimistic-no-wrench", action="store_true",
      help="also run and plot the optimistic/Dogar-style solver under the matched topological "
           "(--no-wrench) model")
  parser.add_argument(
      "--paper", action="store_true",
      help="plot with paper_2.mplstyle and omit all titles and legends")
  args = parser.parse_args(argv)
  if args.repeats < 1:
    raise SystemExit("--repeats must be a positive integer")
  if args.jobs < 1:
    raise SystemExit("--jobs must be a positive integer")

  stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  output = (args.output or ROOT / "benchmarks" / "results"
            / f"process-wrench-full-matrix-{stamp}").resolve()
  selected = EXPERIMENTS if args.assembly == "all" else {
      args.assembly: EXPERIMENTS[args.assembly]}
  status = 0
  legend_has_failure = False

  def run_args(manifest, destination):
    values = [str(manifest), "--output", str(destination)]
    if args.resume:
      values.append("--resume")
    if args.repeats != 1:
      values += ["--repeats", str(args.repeats)]
    if args.seeds is not None:
      values += ["--seeds", *map(str, args.seeds)]
    if args.jobs != 1:
      values += ["--jobs", str(args.jobs)]
    if args.timeout is not None:
      values += ["--timeout", str(args.timeout)]
    if args.only is not None:
      values += ["--only", args.only]
    if args.dry_run:
      values.append("--dry-run")
    return values

  for name, spec in selected.items():
    physical_dir = output / name
    control_dir = output / f"{name}-no-wrench"
    optimistic_dir = output / f"{name}-optimistic-no-wrench"
    # Complete the cheap baseline before the finite-wrench matrix. This does not affect samples
    # (every one is a fresh process), and gives an interrupted overnight run a useful control
    # curve before --resume continues the more expensive cells.
    status = max(status, search_runner.main(run_args(spec["control"], control_dir)))
    if args.optimistic_no_wrench:
      status = max(status, search_runner.main(
          run_args(spec["optimistic_control"], optimistic_dir)))
    status = max(status, search_runner.main(run_args(spec["physical"], physical_dir)))
    if not args.dry_run and not args.no_plot:
      physical_rows, control_rows, optimistic_rows = plotter.plot_pair(
          physical_dir, control_dir, output,
          output_stem=spec["stem"], title=spec["title"], x_label=spec["x_label"],
          paper=args.paper,
          optimistic_no_wrench_suite=(optimistic_dir if args.optimistic_no_wrench else None))
      legend_has_failure = legend_has_failure or any(
          row["failures"] for row in physical_rows + control_rows + optimistic_rows)
      plotter.plot_pair(
          physical_dir, control_dir, output,
          output_stem=spec["stem"] + "-cost", title=spec["title"] + ": plan cost",
          x_label=spec["x_label"], metric="plan_cost_mean",
          y_label="Initial witnessed plan cost", paper=args.paper,
          optimistic_no_wrench_suite=(optimistic_dir if args.optimistic_no_wrench else None))
      print(f"[process-wrench-matrix] {name} plot -> "
            f"{output / (spec['stem'] + '.pdf')}")
  if not args.dry_run and not args.no_plot:
    legend_path = output / "process-wrench-scaling-legend.tex"
    plotter.write_tikz_legend(
        legend_path, include_optimistic=args.optimistic_no_wrench,
        include_failure=legend_has_failure)
    print(f"[process-wrench-matrix] shared TikZ legend -> {legend_path}")
  return status


if __name__ == "__main__":
  raise SystemExit(main())
