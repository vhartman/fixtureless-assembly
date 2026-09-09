#!/usr/bin/env python3
"""Run topological, gravity-only, and increasing insertion-wrench cube scaling."""

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

MANIFEST = ROOT / "benchmarks" / "manifests" / "cube_process_wrench_gravity_scaling.json"
NO_WRENCH_MANIFEST = (ROOT / "benchmarks" / "manifests"
                      / "cube_process_wrench_no_wrench_control.json")
PIECEWISE_MANIFEST = (ROOT / "benchmarks" / "manifests"
                      / "cube_piecewise_process_wrench_scaling.json")
PIECEWISE_NO_WRENCH_MANIFEST = (ROOT / "benchmarks" / "manifests"
                                / "cube_piecewise_process_wrench_no_wrench_control.json")
# The 8-part lambda<=0.1 runs already take 7--8.5 minutes apiece. Keep the informative tested
# subset as the safe default; the three unmeasured 10-part tiny-load points require an explicit
# opt-in rather than unexpectedly turning a plot preview into a multi-hour run.
DEFAULT_SUBSET = (r"^(?:cube-(?:02|04|08)p__lambda-[^_]+|"
                  r"cube-10p__lambda-(?:0|0[.]5|1|2))__seed-")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--only", default=None, metavar="REGEX")
  parser.add_argument("--max-runs", type=int, default=None)
  parser.add_argument("--timeout", type=float, default=None, metavar="SEC")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--with-no-wrench", action="store_true",
                      help="also run one matched topological control per part count")
  parser.add_argument("--full-grid", action="store_true",
                      help="also run 10-part lambda 0.001/0.01/0.1 (potentially multi-hour)")
  parser.add_argument("--piecewise", action="store_true",
                      help="use the strict full cube sequence instead of plate modules")
  args = parser.parse_args(argv)

  stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  manifest = PIECEWISE_MANIFEST if args.piecewise else MANIFEST
  no_wrench_manifest = (PIECEWISE_NO_WRENCH_MANIFEST if args.piecewise
                        else NO_WRENCH_MANIFEST)
  experiment = ("cube-piecewise-process-wrench-scaling-2r" if args.piecewise
                else "cube-process-wrench-scaling-2r")
  output = (args.output or ROOT / "benchmarks" / "results"
            / f"{experiment}-{stamp}").resolve()
  runner_args = [str(manifest), "--output", str(output)]
  if args.resume:
    runner_args.append("--resume")
  if args.only is not None:
    runner_args += ["--only", args.only]
  elif not args.full_grid:
    runner_args += ["--only", DEFAULT_SUBSET]
  if args.max_runs is not None:
    runner_args += ["--max-runs", str(args.max_runs)]
  if args.timeout is not None:
    runner_args += ["--timeout", str(args.timeout)]
  if args.dry_run:
    runner_args.append("--dry-run")
  status = search_runner.main(runner_args)
  no_wrench_output = output.with_name(output.name + "-no-wrench")
  if args.with_no_wrench:
    control_args = [str(no_wrench_manifest), "--output", str(no_wrench_output)]
    if args.resume:
      control_args.append("--resume")
    if args.timeout is not None:
      control_args += ["--timeout", str(args.timeout)]
    if args.dry_run:
      control_args.append("--dry-run")
    status = max(status, search_runner.main(control_args))
  if not args.dry_run:
    output_stem = ("cube-piecewise-process-wrench-scaling" if args.piecewise
                   else "cube-process-wrench-scaling")
    title = ("Two-robot piecewise cube: topological to physical assembly loads"
             if args.piecewise else
             "Two-robot cube: topological to physical assembly loads")
    plotter.plot_suite(output,
                       no_wrench_suite=no_wrench_output if args.with_no_wrench else None,
                       output_stem=output_stem, title=title)
    print(f"[process-wrench] plot -> {output / (output_stem + '.pdf')}")
  return status


if __name__ == "__main__":
  raise SystemExit(main())
