#!/usr/bin/env python3
"""Run the two-robot 1--7 Cross scaling experiment at five wrench-envelope scales."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks import plot_cube_wrench_envelope_scaling as plotter  # noqa: E402
from benchmarks import search_runner  # noqa: E402


MANIFEST = ROOT / "benchmarks" / "manifests" / "cross_wrench_envelope_scaling.json"


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--only", default=None, metavar="REGEX")
  parser.add_argument("--max-runs", type=int, default=None)
  parser.add_argument("--timeout", type=float, default=None, metavar="SEC")
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args(argv)

  stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  output = (args.output or ROOT / "benchmarks" / "results"
            / f"cross-wrench-envelope-scaling-2r-{stamp}").resolve()
  runner_args = [str(MANIFEST), "--output", str(output)]
  if args.resume:
    runner_args.append("--resume")
  if args.only is not None:
    runner_args += ["--only", args.only]
  if args.max_runs is not None:
    runner_args += ["--max-runs", str(args.max_runs)]
  if args.timeout is not None:
    runner_args += ["--timeout", str(args.timeout)]
  if args.dry_run:
    runner_args.append("--dry-run")

  status = search_runner.main(runner_args)
  if not args.dry_run:
    plotter.plot_suite(
        output, scale_flag="--two-finger-scale",
        output_stem="cross-wrench-envelope-scaling",
        title="Two-robot Cross scaling versus grasp wrench capability",
        x_label="Number of Cross parts", envelope_label="Two-finger wrench envelope")
    print(f"[wrench-envelope] plot -> {output / 'cross-wrench-envelope-scaling.pdf'}")
  return status


if __name__ == "__main__":
  raise SystemExit(main())
