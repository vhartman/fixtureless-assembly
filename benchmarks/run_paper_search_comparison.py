#!/usr/bin/env python3
"""Run the paper search comparison and emit table-ready LaTeX/CSV artifacts.

The actual experiments are delegated to :mod:`benchmarks.search_runner`: every sample is a
fresh, sequential, time- and address-space-limited process. This wrapper fixes the paper
manifest, exposes convenient smoke/resume controls, and turns the resulting records into the
paper tables and runtime--cost summary plot. By default, a second stage runs
first-solution-only motion planning for successful Ours plans; witness post-processing
remains excluded.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks import search_runner  # noqa: E402


MANIFEST = ROOT / "benchmarks/manifests/paper_search_comparison.json"
METHODS = (
    "optimistic", "ours", "flat-lazy", "milestones-counting",
    "ours-exact-fills", "ours-fine-projection",
)
CASES = (
    "cross", "stool", "chair", "beam", "plumbers-block", "fabrica-car",
    "gamepad", "cooling", "duct", "fabrica-stool",
)
CASE_LABEL = {
    "cross": "Cross", "stool": "Stool", "chair": "Chair",
    "beam": "F. Beam", "plumbers-block": "F. Plumber block",
    "fabrica-car": "F. Car", "gamepad": "F. Gamepad",
    "cooling": "F. Cooling Manifold", "duct": "F. Duct",
    "fabrica-stool": "F. Stool",
}
METHOD_LABEL = {
    "optimistic": "Optimistic",
    "ours": "Ours",
    "flat-lazy": "Flat lazy",
    "milestones-counting": "Milestones + counting",
    "ours-exact-fills": "Ours, exact fills",
    "ours-fine-projection": "Ours, fine projection",
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


def _best_by_case(groups, metric):
  """Smallest successful aggregate mean for each benchmark instance."""
  lookup = {(g["case"], g["variant"]): g for g in groups}
  best = {}
  for case in CASES:
    values = []
    for method in METHODS:
      value = lookup.get((case, method), {}).get(f"{metric}_mean")
      if value is not None and math.isfinite(float(value)):
        values.append(float(value))
    if values:
      best[case] = min(values)
  return best


def _numeric_source(value, sd, metric):
  """Return a raw number expression for parsing by an siunitx S column."""
  if metric == "plan_cost":
    # Action costs are integral.  Preserve that presentation even if an aggregate reader has
    # converted the stored value to float.
    return f"{float(value):.0f}"
  elif sd is not None and math.isfinite(float(sd)):
    return f"{float(value):.1f} +- {float(sd):.1f}"
  return f"{float(value):.1f}"


def _numeric_tex(value, sd, metric, best=False):
  """An S-column entry; leaving the number raw is required for decimal alignment."""
  number = _numeric_source(value, sd, metric)
  return f"\\bfseries {number}" if best else number


def _comparison_cell(group, metric, best_value=None):
  """Render an aligned numeric cell, or a centered explicit failure outcome."""
  if group is None:
    return "\\multicolumn{1}{c}{--}"
  runs, successes = int(group["runs"]), int(group["successes"])
  value, sd = group.get(f"{metric}_mean"), group.get(f"{metric}_sd")
  if successes == 0:
    outcomes = tuple(
        label for key, label in (("out_of_memory", "OOM"), ("timeouts", "timeout"),
                                 ("no_plans", "no plan"), ("errors", "error"))
        if int(group.get(key, 0)) > 0)
    return "\\multicolumn{1}{c}{" + "/".join(outcomes or ("error",)) + "}"
  if value is None or not math.isfinite(float(value)):
    return "\\multicolumn{1}{c}{--}"
  is_best = best_value is not None and math.isclose(
      float(value), float(best_value), rel_tol=1e-9, abs_tol=1e-9)
  rendered = _numeric_tex(value, sd, metric, best=is_best)
  if successes != runs:
    number = _numeric_source(value, sd, metric)
    weight = "\\bfseries " if is_best else ""
    return f"\\multicolumn{{1}}{{c}}{{{successes}/{runs}; {weight}\\num{{{number}}}}}"
  return rendered


def _aligned_time_cell(group, metric):
  """A centered, decimal-aligned time entry for the breakdown table."""
  if group is None:
    return "\\multicolumn{1}{c}{--}"
  value, sd = group.get(f"{metric}_mean"), group.get(f"{metric}_sd")
  if value is None or not math.isfinite(float(value)):
    return "\\multicolumn{1}{c}{--}"
  return _numeric_tex(value, sd, "time")


def _comparison_tex(groups):
  lookup = {(g["case"], g["variant"]): g for g in groups}
  columns = " & ".join(f"\\multicolumn{{1}}{{c}}{{{CASE_LABEL[c]}}}" for c in CASES)
  best_time = _best_by_case(groups, "search_pipeline_s")
  best_cost = _best_by_case(groups, "plan_cost")
  # Timeout suppression can leave some cells with fewer observations.  The largest cell count
  # is the requested experimental N; incomplete cells remain marked individually in the body.
  run_count = max((int(group["runs"]) for group in groups
                   if group["case"] in CASES and group["variant"] in METHODS), default=0)
  lines = [
      "% Requires: \\usepackage{booktabs,multirow,siunitx}",
      "\\begin{table*}[t]", "  \\centering",
      "  \\caption{Comparison of search times and costs for different algorithms. We report "
      f"the mean and standard deviation of {run_count} "
      f"{'run' if run_count == 1 else 'runs'}.}}",
      "  \\label{tab:comparison}", "  \\resizebox{\\textwidth}{!}{%",
      "  \\begin{tabular}{ll|" + "*{" + str(len(CASES)) + "}{"
      "S[table-format=3.1(1),separate-uncertainty=true,detect-weight=true,"
      "table-number-alignment=center,table-text-alignment=center]}" + "}",
      "\\toprule",
      f"Solver & Metric & {columns} \\\\", "\\midrule",
  ]
  for i, method in enumerate(METHODS):
    if i in (2, 4):
      lines.append("\\midrule")
    time_cells = [_comparison_cell(lookup.get((case, method)), "search_pipeline_s",
                                   best_time.get(case)) for case in CASES]
    cost_cells = [_comparison_cell(lookup.get((case, method)), "plan_cost",
                                   best_cost.get(case)) for case in CASES]
    lines.append(f"\\multirow{{2}}{{*}}{{{METHOD_LABEL[method]}}} & "
                 "$t_{\\mathrm{init}}$ & "
                 + " & ".join(time_cells) + " \\\\")
    lines.append(" & cost & " + " & ".join(cost_cells) + " \\\\")
  lines.extend(["\\bottomrule", "  \\end{tabular}%", "  }", "\\end{table*}", ""])
  return "\n".join(lines)


def _geomean_rows(groups):
  """Return paired runtime/cost geometric means for each method.

  A method's point uses only instances for which that method has both a runtime and a cost.
  Keeping the pairwise instance set explicit prevents a runtime from one subset and a cost from
  another subset from being combined into a visually plausible but meaningless point.
  """
  lookup = {(g["case"], g["variant"]): g for g in groups}
  rows = []
  for method in METHODS:
    paired = []
    for case in CASES:
      group = lookup.get((case, method), {})
      runtime, cost = group.get("search_pipeline_s_mean"), group.get("plan_cost_mean")
      if (runtime is not None and cost is not None and float(runtime) > 0
          and float(cost) > 0):
        paired.append((case, float(runtime), float(cost)))
    rows.append({
        "solver": method,
        "label": METHOD_LABEL[method],
        "t_init_geomean_s": _geomean([entry[1] for entry in paired]),
        "cost_geomean": _geomean([entry[2] for entry in paired]),
        "instances": [entry[0] for entry in paired],
    })
  return rows


def _comparison_plot(groups, output: Path):
  # Import lazily so report parsing/table generation remains usable without a plotting backend.
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  rows = _geomean_rows(groups)
  style = ROOT / "paper_2.mplstyle"
  context = plt.style.context(str(style)) if style.exists() else plt.style.context("default")
  markers = ("o", "*", "X", "s", "^", "D")
  offsets = ((-8, 10), (8, -10), (0, 0), (8, -5), (8, -5), (6, 10))
  alignments = (("right", "bottom"), ("left", "top"), ("left", "bottom"),
                ("left", "top"), ("left", "top"), ("left", "bottom"))
  with context:
    fig, ax = plt.subplots()
    missing = []
    for index, row in enumerate(rows):
      runtime, cost = row["t_init_geomean_s"], row["cost_geomean"]
      color = f"C{index}"
      if runtime is None or cost is None:
        missing.append(row["label"])
        continue
      ax.scatter(runtime, cost, color=color, marker=markers[index], s=90, zorder=3)
      horizontal, vertical = alignments[index]
      ax.annotate(row["label"], (runtime, cost), xytext=offsets[index],
                  textcoords="offset points", fontsize=14, ha=horizontal, va=vertical)
    ax.set_xlabel(r"Geometric mean $t_{\mathrm{init}}$ [s]", fontsize=26)
    ax.set_ylabel("Geometric mean plan cost", fontsize=26)
    ax.tick_params(labelsize=22)
    ax.margins(x=0.14, y=0.18)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    if missing:
      ax.text(0.02, 0.03, ", ".join(missing) + ": no successful run",
              transform=ax.transAxes, fontsize=13, color="0.35", ha="left", va="bottom")
    for suffix in ("pdf", "png"):
      fig.savefig(output / f"comparison-time-cost.{suffix}",
                  dpi=220 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
  return rows

def _breakdown_tex(groups):
  lookup = {(g["case"], g["variant"]): g for g in groups}
  metrics = ("grasp_generation_s", "backward_graph_search_s", "backward_force_lp_s",
             "forward_search_s", "ik_bind_wall_s", "mp_initial_solution_s")
  lines = [
      "% Requires: \\usepackage{booktabs,siunitx}",
      "\\begin{table}[t]", "  \\centering",
      "  \\caption{Breakdown of where time is spent in \\textsc{Wrap}.}",
      "  \\label{tab:search_time_spent}", "  \\resizebox{\\columnwidth}{!}{%",
      "  \\begin{tabular}{l|"
      "S[table-format=3.1(1),separate-uncertainty=true,table-number-alignment=center]|"
      "S[table-format=3.1(1),separate-uncertainty=true,table-number-alignment=center]"
      "S[table-format=3.1(1),separate-uncertainty=true,table-number-alignment=center]|"
      "S[table-format=3.1(1),separate-uncertainty=true,table-number-alignment=center]"
      "S[table-format=3.1(1),separate-uncertainty=true,table-number-alignment=center]|"
      "S[table-format=3.1(1),separate-uncertainty=true,table-number-alignment=center]}",
      "\\toprule",
      "& \\multicolumn{1}{c|}{} & \\multicolumn{2}{c|}{Backwards} & "
      "\\multicolumn{2}{c|}{Forward} & \\multicolumn{1}{c}{Motion} \\\\",
      "Object & \\multicolumn{1}{c|}{$t_{\\text{grasp}}$} & "
      "\\multicolumn{1}{c}{$t_{\\text{bwd search}}$} & "
      "\\multicolumn{1}{c|}{$t_{\\text{force LP}}$} & "
      "\\multicolumn{1}{c}{$t_{\\text{fwd search}}$} & "
      "\\multicolumn{1}{c|}{$t_{\\text{IK}}$} & "
      "\\multicolumn{1}{c}{$t_{\\text{MP init}}$} \\\\",
      "\\midrule",
  ]
  for case in CASES:
    group = lookup.get((case, "ours"))
    lines.append(CASE_LABEL[case] + " & "
                 + " & ".join(_aligned_time_cell(group, metric)
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
    if result.get("status") != "error" or result.get("out_of_memory"):
      continue
    log = result.get("artifacts", {}).get("log")
    log_path = output / log if log else None
    log_text = (log_path.read_text(encoding="utf-8", errors="replace")
                if log_path is not None and log_path.exists() else "")
    if search_runner._looks_like_out_of_memory(log_text):
      result["out_of_memory"] = True
      result["status"] = "oom"
  groups = search_runner.aggregate(results)
  output.mkdir(parents=True, exist_ok=True)
  # These two generated files were superseded by the paired-row table below. Remove stale
  # copies so a report-only refresh cannot leave three apparently current comparison tables.
  for legacy_name in ("comparison-time.tex", "comparison-cost.tex"):
    (output / legacy_name).unlink(missing_ok=True)
  (output / "comparison.tex").write_text(_comparison_tex(groups), encoding="utf-8")
  (output / "search-time-breakdown.tex").write_text(_breakdown_tex(groups), encoding="utf-8")
  geomeans = _comparison_plot(groups, output)
  (output / "comparison-time-cost-geomeans.json").write_text(
      json.dumps(geomeans, indent=2) + "\n", encoding="utf-8")
  with (output / "comparison-time-cost-geomeans.csv").open(
      "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=(
        "solver", "label", "t_init_geomean_s", "cost_geomean", "instances"))
    writer.writeheader()
    for row in geomeans:
      writer.writerow({**row, "instances": ",".join(row["instances"])})

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
            "ik_bind_calls", "ik_bind_fail", "mp_initial_solution_s",
        )},
    })
  with (output / "table-data.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["object", "solver"])
    writer.writeheader()
    writer.writerows(rows)
  (output / "table-data.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
  print(f"[paper-search] LaTeX/CSV tables and runtime-cost plot -> {output}")


def _resolve_plan_artifact(output: Path, result, merge_sources):
  """Resolve a plan in either a native or a report-only merged result directory."""
  relative = result.get("artifacts", {}).get("plan")
  if not relative:
    return None
  local = output / relative
  if local.is_file():
    return local.resolve()
  source = merge_sources.get(result.get("run_id"))
  if source:
    candidate = Path(source).parent / relative
    if candidate.is_file():
      return candidate.resolve()
  return None


def _motion_command(plan: Path, runtime_s: float, seed: int):
  # No shortcut: measure discovery of the first complete path, not post-solution trajectory
  # simplification. Explicit no-optimize overrides the standard profile's anytime RRT* default
  # without changing its collision policy.
  return [
      sys.executable, str(ROOT / "main.py"), "--config", str(plan),
      "--load-plan", str(plan), "--motion-plan", "--motion-profile", "standard",
      "--no-mp-optimize", "--mp-no-shortcut", "--mp-runtime", str(runtime_s),
      "--mp-seed", str(seed), "--quiet-rai", "--no-view",
  ]


def _motion_fingerprint(result, plan: Path, command):
  payload = {
      "command": command,
      "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
      "source_sha256": result.get("source_sha256"),
  }
  return hashlib.sha256(json.dumps(
      payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _run_motion_initial(result, output: Path, runtime_s: float, timeout_s: float,
                        memory_gb: float, merge_sources):
  run_id = result["run_id"]
  run_dir = output / "motion-initial" / run_id
  run_dir.mkdir(parents=True, exist_ok=True)
  log_path = run_dir / "stdout.log"
  result_path = run_dir / "result.json"
  plan = _resolve_plan_artifact(output, result, merge_sources)
  if plan is None:
    measured = {
        "status": "missing_plan", "success": False, "mp_initial_solution_s": None,
        "mp_initial_observed_wall_s": 0.0,
        "mp_initial_log": str(log_path.relative_to(output)),
    }
    search_runner._write_json(result_path, measured)
    return measured

  command = _motion_command(plan, runtime_s, int(result.get("seed", 0)))
  fingerprint = _motion_fingerprint(result, plan, command)
  if memory_gb > 0:
    if not search_runner._PRLIMIT.is_file():
      raise RuntimeError("--motion-memory-gb requires /usr/bin/prlimit")
    limit_bytes = int(memory_gb * 1024 ** 3)
    guarded = [str(search_runner._PRLIMIT), f"--as={limit_bytes}", "--", *command]
  else:
    guarded = command
  env = os.environ.copy()
  env.update({
      "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1",
      "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
      "NUMBA_CACHE_DIR": "/tmp/assembly-numba-cache",
      "MPLCONFIGDIR": "/tmp/assembly-mpl-cache",
  })
  t0 = time.perf_counter()
  timed_out = False
  with log_path.open("w", encoding="utf-8") as log:
    log.write(f"[benchmark-command] {shlex.join(command)}\n")
    log.flush()
    proc = subprocess.Popen(guarded, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
      returncode = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
      timed_out = True
      os.killpg(proc.pid, signal.SIGTERM)
      try:
        returncode = proc.wait(timeout=5)
      except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        returncode = proc.wait()
  wall_s = time.perf_counter() - t0
  text = log_path.read_text(encoding="utf-8", errors="replace")
  match = re.search(r"^\[mp-prof\] initial_solution_s=" + search_runner._NUM + r"\s",
                    text, re.MULTILINE)
  initial_s = float(match.group(1)) if match else None
  terminal = bool(re.search(r"\[motion-plan\].*reaches terminal=True", text))
  success = returncode == 0 and initial_s is not None and terminal
  status = ("success" if success else "timeout" if timed_out else
            "oom" if search_runner._looks_like_out_of_memory(text) else
            "no_plan" if returncode == 0 else "error")
  measured = {
      "status": status, "success": success, "timed_out": timed_out,
      "returncode": returncode, "mp_initial_solution_s": initial_s,
      "mp_initial_observed_wall_s": wall_s,
      "mp_initial_log": str(log_path.relative_to(output)),
      "run_fingerprint": fingerprint,
      "planner_command": command, "planner_command_shell": shlex.join(command),
  }
  search_runner._write_json(result_path, measured)
  return measured


def measure_initial_motion_plans(output: Path, runtime_s: float, timeout_s: float,
                                 memory_gb: float, resume=False,
                                 retry_failed_only=False, jobs=1):
  """Measure first-solution motion planning for successful Ours search rows only."""
  results_path = output / "results.json"
  if not results_path.is_file():
    raise FileNotFoundError(f"missing search results: {results_path}")
  results = json.loads(results_path.read_text(encoding="utf-8"))
  provenance_path = output / "merge-provenance.json"
  merge_sources = {}
  if provenance_path.is_file():
    merge_sources = json.loads(
        provenance_path.read_text(encoding="utf-8")).get("selected_from", {})
  selected = [
      result for result in results
      if result.get("variant") == "ours" and result.get("success")
      and (not retry_failed_only or result.get("mp_initial_status") != "success")
  ]
  print(f"[paper-search] measuring initial MP for {len(selected)} successful Ours run(s), "
        f"up to {jobs} case(s) concurrently")
  output_lock = threading.Lock()

  def run_group(group):
    for index, result in group:
      cached = output / "motion-initial" / result["run_id"] / "result.json"
      old = json.loads(cached.read_text(encoding="utf-8")) if cached.is_file() else None
      plan = _resolve_plan_artifact(output, result, merge_sources)
      expected = (_motion_fingerprint(
          result, plan, _motion_command(plan, runtime_s, int(result.get("seed", 0))))
                  if plan is not None else None)
      if resume and old is not None and old.get("run_fingerprint") == expected:
        measured = old
        action = "reused"
      else:
        measured = _run_motion_initial(
            result, output, runtime_s, timeout_s, memory_gb, merge_sources)
        action = "measured"
      with output_lock:
        result["mp_initial_status"] = measured["status"]
        result["mp_initial_solution_s"] = measured.get("mp_initial_solution_s")
        result["mp_initial_observed_wall_s"] = measured.get("mp_initial_observed_wall_s")
        result.setdefault("artifacts", {})["mp_initial_log"] = measured.get("mp_initial_log")
        # Checkpoint after every potentially expensive motion-planning run.
        search_runner._write_json(results_path, results)
      print(f"[paper-search] MP {index}/{len(selected)} {result['run_id']}: {action} "
            f"{measured['status']}, t_init={measured.get('mp_initial_solution_s')}")

  grouped = {}
  for index, result in enumerate(selected, 1):
    grouped.setdefault(result["case"], []).append((index, result))
  groups = list(grouped.values())
  if not groups:
    return
  if jobs == 1:
    for group in groups:
      run_group(group)
  else:
    with ThreadPoolExecutor(max_workers=min(jobs, len(groups)),
                            thread_name_prefix="motion-case") as pool:
      futures = [pool.submit(run_group, group) for group in groups]
      for future in as_completed(futures):
        future.result()


def merge_results(inputs, output: Path):
  """Merge report records while retaining their original per-run provenance.

  Later inputs replace duplicate run ids. This is useful when a corrected subset was rerun:
  it avoids rerunning unaffected cells, and records the source selected for every row.
  """
  merged, selected_from, sources = {}, {}, []
  for raw in inputs:
    source = raw / "results.json" if raw.is_dir() else raw
    if not source.is_file():
      raise FileNotFoundError(f"missing benchmark results: {source}")
    records = json.loads(source.read_text(encoding="utf-8"))
    sources.append({"path": str(source.resolve()), "records": len(records)})
    for record in records:
      run_id = record.get("run_id")
      if not run_id:
        raise ValueError(f"{source}: result without run_id")
      merged[run_id] = record
      selected_from[run_id] = str(source.resolve())
  ordered = sorted(merged.values(), key=lambda row: (
      CASES.index(row["case"]) if row.get("case") in CASES else len(CASES),
      METHODS.index(row["variant"]) if row.get("variant") in METHODS else len(METHODS),
      row.get("seed", 0), row.get("repeat", 1)))
  output.mkdir(parents=True, exist_ok=True)
  (output / "results.json").write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
  (output / "merge-provenance.json").write_text(json.dumps({
      "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
      "policy": "later input replaces duplicate run_id",
      "sources": sources,
      "unique_records": len(ordered),
      "selected_from": selected_from,
  }, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path,
                      default=ROOT / "benchmarks/results/paper-search-comparison")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument(
      "--seeds", type=int, nargs="+", default=None, metavar="SEED",
      help=("independent planner seeds, replacing the checked-in manifest seeds "
            "(default: 11023 11024 11025)"))
  parser.add_argument(
      "--repeats", type=int, default=1,
      help="repeat each selected seed unchanged to measure same-seed timing variation")
  parser.add_argument(
      "--jobs", type=int, default=1, metavar="N",
      help=("run up to N different assembly cases concurrently; each case retains its "
            "sequential solver/seed order (default: 1)"))
  parser.add_argument(
      "--continue-after-timeout", action="store_true",
      help=("run every requested seed even after an earlier seed times out for the same "
            "case/solver; default preserves the paper suite's timeout suppression"))
  parser.add_argument("--only", default=None, metavar="REGEX")
  parser.add_argument("--max-runs", type=int, default=None)
  parser.add_argument("--timeout", type=float, default=450.0, metavar="SEC",
                      help="per-search process cutoff (default: 450)")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--report-only", action="store_true")
  parser.add_argument(
      "--skip-motion-initial", action="store_true",
      help="skip the default first-solution motion-planning stage for successful Ours rows")
  parser.add_argument(
      "--retry-failed-motion", action="store_true",
      help=("do not rerun search; rerun initial motion planning only for successful Ours "
            "rows whose previous motion measurement was unsuccessful"))
  parser.add_argument(
      "--motion-runtime", type=float, default=300.0, metavar="SEC",
      help="BiRRT first-solution budget for each Ours plan (default: 300)")
  parser.add_argument(
      "--motion-timeout", type=float, default=420.0, metavar="SEC",
      help="whole-process guard including export/setup (default: 420)")
  parser.add_argument(
      "--motion-memory-gb", type=float, default=4.0, metavar="GB",
      help="per-process address-space guard for motion planning; 0 disables (default: 4)")
  parser.add_argument("--merge-from", action="append", type=Path, default=[], metavar="DIR",
                      help="merge DIR/results.json into OUTPUT; later duplicate run ids win")
  args = parser.parse_args(argv)
  if args.timeout <= 0:
    parser.error("--timeout must be positive")
  if args.jobs < 1:
    parser.error("--jobs must be a positive integer")
  if args.motion_runtime <= 0 or args.motion_timeout <= 0:
    parser.error("--motion-runtime and --motion-timeout must be positive")
  if args.motion_memory_gb < 0:
    parser.error("--motion-memory-gb must be non-negative")
  output = args.output.resolve()
  if args.retry_failed_motion:
    if args.merge_from or args.dry_run:
      parser.error("--retry-failed-motion cannot be combined with --merge-from or --dry-run")
    measure_initial_motion_plans(
        output, runtime_s=args.motion_runtime, timeout_s=args.motion_timeout,
        memory_gb=args.motion_memory_gb, resume=False, retry_failed_only=True,
        jobs=args.jobs)
    write_tables(output)
    return 0
  if args.merge_from:
    merge_results(args.merge_from, output)
    write_tables(output)
    return 0
  if not args.report_only:
    runner = [str(MANIFEST), "--output", str(output), "--repeats", str(args.repeats),
              "--shared-grasp-db", "--jobs", str(args.jobs)]
    if not args.continue_after_timeout:
      runner.append("--stop-cell-on-timeout")
    if args.seeds is not None:
      runner += ["--seeds", *map(str, args.seeds)]
    if args.resume:
      runner.append("--resume")
    if args.only:
      runner += ["--only", args.only]
    if args.max_runs is not None:
      runner += ["--max-runs", str(args.max_runs)]
    runner += ["--timeout", str(args.timeout)]
    if args.dry_run:
      runner.append("--dry-run")
    status = search_runner.main(runner)
    if args.dry_run:
      return status
    if not args.skip_motion_initial:
      measure_initial_motion_plans(
          output, runtime_s=args.motion_runtime, timeout_s=args.motion_timeout,
          memory_gb=args.motion_memory_gb, resume=args.resume,
          retry_failed_only=False, jobs=args.jobs)
  write_tables(output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
