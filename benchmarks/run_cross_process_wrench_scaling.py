#!/usr/bin/env python3
"""Run topological, gravity-only, and increasing insertion-wrench Cross scaling."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks import plot_cube_process_wrench_gravity as plotter  # noqa: E402
from benchmarks import search_runner  # noqa: E402

MANIFEST = ROOT / "benchmarks" / "manifests" / "cross_process_wrench_scaling.json"
CONTROL = ROOT / "benchmarks" / "manifests" / "cross_process_wrench_no_wrench_control.json"


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--only", default=None, metavar="REGEX")
  parser.add_argument("--max-runs", type=int, default=None)
  parser.add_argument("--timeout", type=float, default=None, metavar="SEC")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--with-no-wrench", action="store_true")
  args = parser.parse_args(argv)

  stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  output = (args.output or ROOT / "benchmarks" / "results"
            / f"cross-process-wrench-scaling-2r-{stamp}").resolve()

  def common(manifest, destination):
    values = [str(manifest), "--output", str(destination)]
    if args.resume:
      values.append("--resume")
    if args.timeout is not None:
      values += ["--timeout", str(args.timeout)]
    if args.dry_run:
      values.append("--dry-run")
    return values

  runner_args = common(MANIFEST, output)
  if args.only is not None:
    runner_args += ["--only", args.only]
  if args.max_runs is not None:
    runner_args += ["--max-runs", str(args.max_runs)]
  status = search_runner.main(runner_args)
  control_output = output.with_name(output.name + "-no-wrench")
  if args.with_no_wrench:
    status = max(status, search_runner.main(common(CONTROL, control_output)))
  if not args.dry_run:
    plotter.plot_suite(
        output, no_wrench_suite=control_output if args.with_no_wrench else None,
        output_stem="cross-process-wrench-scaling",
        title="Two-robot Cross: topological to physical assembly loads",
        x_label="Number of Cross parts")
    print(f"[process-wrench] plot -> {output / 'cross-process-wrench-scaling.pdf'}")
  return status


if __name__ == "__main__":
  raise SystemExit(main())
