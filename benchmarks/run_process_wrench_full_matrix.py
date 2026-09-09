#!/usr/bin/env python3
"""Run the full piecewise-Cube and Cross process-wrench matrices sequentially."""

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
        "stem": "cross-process-wrench-full-matrix",
        "title": "Cross: topological to physical assembly loads",
        "x_label": "Number of Cross parts",
    },
    "cube": {
        "physical": MANIFEST_DIR / "cube_piecewise_process_wrench_full_matrix.json",
        "control": MANIFEST_DIR / "cube_piecewise_process_wrench_full_no_wrench.json",
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
  parser.add_argument("--timeout", type=float, default=None, metavar="SEC")
  parser.add_argument("--only", default=None, metavar="REGEX")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--no-plot", action="store_true")
  args = parser.parse_args(argv)
  if args.repeats < 1:
    raise SystemExit("--repeats must be a positive integer")

  stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  output = (args.output or ROOT / "benchmarks" / "results"
            / f"process-wrench-full-matrix-{stamp}").resolve()
  selected = EXPERIMENTS if args.assembly == "all" else {
      args.assembly: EXPERIMENTS[args.assembly]}
  status = 0

  def run_args(manifest, destination):
    values = [str(manifest), "--output", str(destination)]
    if args.resume:
      values.append("--resume")
    if args.repeats != 1:
      values += ["--repeats", str(args.repeats)]
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
    # Complete the cheap baseline before the finite-wrench matrix. This does not affect samples
    # (every one is a fresh process), and gives an interrupted overnight run a useful control
    # curve before --resume continues the more expensive cells.
    status = max(status, search_runner.main(run_args(spec["control"], control_dir)))
    status = max(status, search_runner.main(run_args(spec["physical"], physical_dir)))
    if not args.dry_run and not args.no_plot:
      plotter.plot_pair(
          physical_dir, control_dir, output,
          output_stem=spec["stem"], title=spec["title"], x_label=spec["x_label"])
      plotter.plot_pair(
          physical_dir, control_dir, output,
          output_stem=spec["stem"] + "-cost", title=spec["title"] + ": plan cost",
          x_label=spec["x_label"], metric="plan_cost_mean",
          y_label="Initial witnessed plan cost")
      print(f"[process-wrench-matrix] {name} plot -> "
            f"{output / (spec['stem'] + '.pdf')}")
  return status


if __name__ == "__main__":
  raise SystemExit(main())
