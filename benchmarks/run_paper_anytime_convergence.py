#!/usr/bin/env python3
"""Run the paper anytime planner and emit cost-versus-time convergence curves.

Each assembly is one scheduling unit: ``--jobs N`` may run N assemblies concurrently, while
seeds for the same assembly remain sequential.  This keeps a single assembly from competing
with another copy of itself and matches the case-level parallelism of ``search_runner``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks import plot_anytime_convergence as plotter  # noqa: E402
from benchmarks import search_runner  # noqa: E402
from benchmarks.run_paper_search_comparison import CASES, CASE_LABEL, MANIFEST  # noqa: E402


def _case_selector(cases) -> str:
  alternatives = "|".join(re.escape(case) for case in cases)
  return rf"^(?:{alternatives})__ours__seed-"


def _write_plots(output: Path, cases, seeds) -> int:
  results_path = output / "results.json"
  if not results_path.is_file():
    raise FileNotFoundError(f"missing convergence results: {results_path}")
  records = json.loads(results_path.read_text(encoding="utf-8"))
  case_order = {case: i for i, case in enumerate(cases)}
  records = sorted(
      (row for row in records if row.get("case") in case_order
       and row.get("variant") == "ours"),
      key=lambda row: (case_order[row["case"]], row.get("seed", 0), row.get("repeat", 1)))
  runs = []
  multiple = len(seeds) > 1
  for row in records:
    relative_log = row.get("artifacts", {}).get("log")
    if not relative_log:
      continue
    label = CASE_LABEL[row["case"]]
    if multiple:
      label += f" (seed {row['seed']})"
    try:
      runs.append(plotter.parse_log(output / relative_log, label))
    except (OSError, ValueError) as exc:
      print(f"[anytime] skip {row.get('run_id')}: {exc}")
  if not runs:
    print("[anytime] no validated incumbent curves were produced")
    return 1
  plotter.write_outputs(
      runs, output / "all-scenes", title="Milestone-anytime convergence")
  return 0


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--output", type=Path, default=None,
      help="result directory (default: benchmarks/results/paper-search-anytime-TIMESTAMP)")
  parser.add_argument(
      "--cases", nargs="+", choices=CASES, default=list(CASES), metavar="CASE",
      help="assembly subset to run (default: all paper cases)")
  parser.add_argument(
      "--seeds", type=int, nargs="+", default=[11023], metavar="SEED",
      help="independent planner seeds (default: 11023)")
  parser.add_argument(
      "--jobs", type=int, default=1, metavar="N",
      help="run up to N different assembly cases concurrently (default: 1)")
  parser.add_argument(
      "--runtime", type=float, default=300.0, metavar="SEC",
      help="milestone-anytime optimization budget per run (default: 300)")
  parser.add_argument(
      "--timeout", type=float, default=None, metavar="SEC",
      help="whole-process timeout (default: runtime + 90 seconds)")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--no-plot", action="store_true")
  args = parser.parse_args(argv)
  if args.jobs < 1:
    parser.error("--jobs must be a positive integer")
  if args.runtime <= 0:
    parser.error("--runtime must be positive")
  timeout = args.timeout if args.timeout is not None else args.runtime + 90.0
  if timeout <= 0:
    parser.error("--timeout must be positive")

  stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  output = (args.output or ROOT / "benchmarks" / "results"
            / f"paper-search-anytime-{stamp}").resolve()
  runner = [
      str(MANIFEST), "--output", str(output),
      "--only", _case_selector(args.cases),
      "--seeds", *map(str, args.seeds),
      "--jobs", str(args.jobs),
      "--timeout", str(timeout),
      "--append-arg=--milestones-anytime", "--append-arg", str(args.runtime),
  ]
  if args.resume:
    runner.append("--resume")
  if args.dry_run:
    runner.append("--dry-run")
  status = search_runner.main(runner)
  if args.dry_run or args.no_plot:
    return status
  plot_status = _write_plots(output, args.cases, args.seeds)
  return max(status, plot_status)


if __name__ == "__main__":
  raise SystemExit(main())
