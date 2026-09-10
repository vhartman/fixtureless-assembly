#!/usr/bin/env python3
"""Run the paper search comparison and emit table-ready LaTeX/CSV artifacts.

The actual experiments are delegated to :mod:`benchmarks.search_runner`: every sample is a
fresh, sequential, time- and address-space-limited process. This wrapper fixes the paper
manifest, exposes convenient smoke/resume controls, and turns the resulting records into the
two tables used by the paper. Motion planning and witness post-processing are intentionally
excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks import search_runner  # noqa: E402


MANIFEST = ROOT / "benchmarks/manifests/paper_search_comparison.json"
METHODS = (
    "optimistic", "ours", "flat-lazy", "milestones-counting",
    "ours-exact-fills", "ours-exact-backwards",
)
CASES = (
    "cross", "stool", "chair", "fabrica-stool", "cooling", "duct",
    "gamepad", "beam", "fabrica-car",
)
CASE_LABEL = {
    "cross": "Cross", "stool": "Stool", "chair": "Chair",
    "fabrica-stool": "F. Stool", "cooling": "Cooling", "duct": "Duct",
    "gamepad": "Gamepad", "beam": "Beam", "fabrica-car": "F. Car",
}
METHOD_LABEL = {
    "optimistic": "Optimistic",
    "ours": "Ours",
    "flat-lazy": "Flat lazy",
    "milestones-counting": "Milestones + counting",
    "ours-exact-fills": "Ours, exact fills",
    "ours-exact-backwards": "Ours, exact backwards",
}


def _fmt(value, sd=None):
  if value is None or not math.isfinite(float(value)):
    return "--"
  if sd is not None and math.isfinite(float(sd)) and float(sd) > 0.05:
    return f"{float(value):.1f} $\\pm$ {float(sd):.1f}"
  return f"{float(value):.1f}"


def _geomean(values):
  values = [float(v) for v in values if v is not None and float(v) > 0]
  return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def _cell(group, metric, include_success=True):
  if group is None:
    return "--"
  runs, successes = int(group["runs"]), int(group["successes"])
  value, sd = group.get(f"{metric}_mean"), group.get(f"{metric}_sd")
  rendered = _fmt(value, sd)
  if include_success and successes != runs:
    if successes == 0 and int(group.get("out_of_memory", 0)) == runs:
      return "OOM"
    if successes == 0 and int(group.get("timeouts", 0)) == runs:
      return "timeout"
    if successes == 0 and int(group.get("no_plans", 0)) == runs:
      return "no plan"
    if successes == 0 and int(group.get("errors", 0)) == runs:
      return "error"
    return f"{successes}/{runs}; {rendered}"
  return rendered


def _comparison_tex(groups, metric, caption, label):
  lookup = {(g["case"], g["variant"]): g for g in groups}
  columns = " & ".join(CASE_LABEL[c] for c in CASES)
  lines = [
      "\\begin{table*}[t]", "  \\centering", f"  \\caption{{{caption}}}",
      f"  \\label{{{label}}}", "  \\resizebox{\\textwidth}{!}{%",
      "  \\begin{tabular}{l|" + "c" * len(CASES) + "|c}", "\\toprule",
      f"Solver & {columns} & Geo. mean \\\\", "\\midrule",
  ]
  for i, method in enumerate(METHODS):
    if i in (2, 4):
      lines.append("\\midrule")
    vals = [lookup.get((case, method), {}).get(f"{metric}_mean") for case in CASES]
    cells = [_cell(lookup.get((case, method)), metric) for case in CASES]
    lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells)
                 + f" & {_fmt(_geomean(vals))} \\\\")
  lines.extend(["\\bottomrule", "  \\end{tabular}%", "  }", "\\end{table*}", ""])
  return "\n".join(lines)


def _breakdown_tex(groups):
  lookup = {(g["case"], g["variant"]): g for g in groups}
  metrics = ("grasp_generation_s", "backward_graph_search_s", "backward_force_lp_s",
             "forward_search_s", "ik_bind_wall_s")
  lines = [
      "\\begin{table}[t]", "  \\centering",
      "  \\caption{Breakdown of search time for our method. Times are seconds.}",
      "  \\label{tab:search_time_spent}",
      "  \\resizebox{\\columnwidth}{!}{%", "  \\begin{tabular}{l|c|cc|cc}",
      "\\toprule",
      "& & \\multicolumn{2}{c|}{Backwards} & \\multicolumn{2}{c}{Forward} \\\\",
      "Object & $t_{\\text{grasp}}$ & $t_{\\text{bwd search}}$ & "
      "$t_{\\text{force LP}}$ & $t_{\\text{fwd search}}$ & $t_{\\text{IK}}$ \\\\",
      "\\midrule",
  ]
  for case in CASES:
    group = lookup.get((case, "ours"))
    lines.append(CASE_LABEL[case] + " & "
                 + " & ".join(_cell(group, metric, include_success=False)
                              for metric in metrics) + " \\\\")
  lines.extend(["\\bottomrule", "  \\end{tabular}%", "  }", "\\end{table}", ""])
  return "\n".join(lines)


def write_tables(output: Path):
  results_path = output / "results.json"
  if not results_path.exists():
    print(f"[paper-search] no results yet at {results_path}; tables not written")
    return
  results = json.loads(results_path.read_text(encoding="utf-8"))
  # Older result files predate the explicit OOM status. Preserve report-only usefulness by
  # classifying their raw log without mutating the source record.
  for result in results:
    if result.get("status") != "error" or result.get("out_of_memory") is not None:
      continue
    log = result.get("artifacts", {}).get("log")
    log_path = output / log if log else None
    if log_path is not None and log_path.exists() and "MemoryError" in log_path.read_text(
        encoding="utf-8", errors="replace"):
      result["out_of_memory"] = True
      result["status"] = "oom"
  groups = search_runner.aggregate(results)
  output.mkdir(parents=True, exist_ok=True)
  (output / "comparison-time.tex").write_text(
      _comparison_tex(
          groups, "search_pipeline_s",
          "Search time for the algorithmic comparison. Partial cells report solved/runs; "
          "the final column is the geometric mean over available instance means.",
          "tab:comparison"), encoding="utf-8")
  (output / "comparison-cost.tex").write_text(
      _comparison_tex(
          groups, "plan_cost", "Plan cost for the algorithmic comparison.",
          "tab:comparison_cost"), encoding="utf-8")
  (output / "search-time-breakdown.tex").write_text(_breakdown_tex(groups), encoding="utf-8")

  rows = []
  for group in groups:
    if group["case"] not in CASES or group["variant"] not in METHODS:
      continue
    rows.append({
        "object": group["case"], "solver": group["variant"],
        "runs": group["runs"], "successes": group["successes"],
        "timeouts": group["timeouts"], "out_of_memory": group["out_of_memory"],
        "errors": group["errors"], "no_plans": group["no_plans"],
        **{key: group.get(f"{key}_mean") for key in (
            "search_pipeline_s", "wall_s", "plan_cost", "max_rss_kb",
            "grasp_generation_s", "grasp_generation_calls", "grasp_backward_s",
            "grasp_backward_calls", "backward_search_s", "backward_graph_search_s",
            "backward_force_lp_s", "backward_force_lp_calls", "forward_search_s",
            "forward_force_lp_s", "forward_force_lp_calls", "ik_bind_wall_s",
            "ik_bind_calls", "ik_bind_fail",
        )},
    })
  with (output / "table-data.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["object", "solver"])
    writer.writeheader()
    writer.writerows(rows)
  (output / "table-data.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
  print(f"[paper-search] LaTeX/CSV tables -> {output}")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path,
                      default=ROOT / "benchmarks/results/paper-search-comparison")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--repeats", type=int, default=1)
  parser.add_argument("--only", default=None, metavar="REGEX")
  parser.add_argument("--max-runs", type=int, default=None)
  parser.add_argument("--timeout", type=float, default=None, metavar="SEC")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--report-only", action="store_true")
  args = parser.parse_args(argv)
  output = args.output.resolve()
  if not args.report_only:
    runner = [str(MANIFEST), "--output", str(output), "--repeats", str(args.repeats)]
    if args.resume:
      runner.append("--resume")
    if args.only:
      runner += ["--only", args.only]
    if args.max_runs is not None:
      runner += ["--max-runs", str(args.max_runs)]
    if args.timeout is not None:
      runner += ["--timeout", str(args.timeout)]
    if args.dry_run:
      runner.append("--dry-run")
    status = search_runner.main(runner)
    if args.dry_run:
      return status
  write_tables(output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
