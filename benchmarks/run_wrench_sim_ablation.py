#!/usr/bin/env python3
"""End-to-end wrench-planning ablation for Cross, Cube, and Chair.

For each scene and seed, this runner computes a fresh task plan and optimized motion path,
then executes that path in MuJoCo under the *same* physical process wrench.  The three planning
models are:

  topological  -- no static-equilibrium checks;
  gravity      -- static equilibrium under gravity, but zero assembly-process load;
  full         -- gravity and the nominal assembly-process load.

The gravity-only search must be solved with zero insertion force.  Before simulation, the
runner writes a separate execution-plan copy whose process force is restored to the common
physical value.  Its symbolic wrench flag is disabled only for replay-state reconstruction;
MuJoCo still applies gravity, grasp breakaway, and ``--apply-process-wrench``.  The original
planner artifact is never modified.

Motion planning is deliberately force-blind for every variant. It uses the same collision and
kinematic settings throughout, while only task planning changes its support model. This avoids
turning the experimental motion-planning wrench gate into a selective failure mode. The common
physical simulation remains the independent test of the resulting trajectories.

Task planning, optimized motion planning, and simulation run sequentially in separate,
thread-limited, memory-limited subprocesses. Simulation timing is selectable between the
certified lockstep baseline and the standard collision-audited phase retimer. Results are
checkpointed after every trial and distinguish task planning, motion planning, simulator
preflight, and physical-execution failures. ``table.tex`` reports end-to-end success over all
completed trials; ``summary.csv`` retains the full failure taxonomy and stage timings.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 8
DEFAULT_SEEDS = (11023, 11024, 11025, 11026, 11027)
RETRYABLE_INFRA_FAILURES = {
    "motion_planning", "motion_planning_timeout", "planning_timeout",
    "task_planning_timeout", "simulation_timeout",
    "simulator_preflight_or_crash",
}

SCENES = {
    "cross": {
        "label": "Cross",
        "gripper_kinds": ("two_finger",),
        "args": ["--assembly", "cross", "--parts", "7", "--num-robots", "4",
                 "--robot", "ur5_limited", "--scatter-spacing", "0.2"],
    },
    "cube": {
        "label": "Cube",
        "gripper_kinds": ("vacuum",),
        # Eight parts is the established executable Cube ablation (four vacuum URs).  The
        # 12-part family remains the scaling benchmark and is needlessly expensive here.
        "args": ["--assembly", "cube-plates-2x2x3", "--parts", "8",
                 "--num-robots", "4", "--robot", "ur5_limited", "--vacuum",
                 "--scatter-spacing", "0.2"],
    },
    "chair": {
        "label": "Chair",
        "gripper_kinds": ("vacuum", "two_finger"),
        "args": ["--assembly", "chair", "--parts", "8", "--num-robots", "4",
                 "--robot", "ur5_limited",
                 "--grippers", "vacuum,vacuum,two_finger,two_finger",
                 "--allow-curved-grasps", "--scatter-spacing", "0.2",
                 "--min-sep", "0.3"],
    },
}

VARIANTS = {
    "topological": {
        "label": "Topological",
        "args": ["--no-wrench"],
        "planning_force": None,
        "mp_wrench_gate": False,
    },
    "gravity": {
        "label": "Gravity only",
        "args": ["--wrench"],
        "planning_force": 0.0,
        "mp_wrench_gate": False,
    },
    "full": {
        "label": "Gravity + process wrench",
        "args": ["--wrench"],
        "planning_force": "physical",
        "mp_wrench_gate": False,
    },
}


def _slug(value):
  return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.") or "run"


def _write_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with tmp.open("w", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
  os.replace(tmp, path)


def _write_csv(path, rows):
  fields = [
      "scene", "variant", "seed", "status", "success", "failure_stage",
      "returncode", "planning_wall_s", "task_planning_wall_s",
      "motion_planning_wall_s", "sim_wall_s", "num_actions",
      "num_actions_ok_prefix", "num_objects_assembled", "num_objects_to_assemble",
      "first_failure_index", "first_failure_action", "path_length_rad",
      "wrench_window_s",
      "transport_peak_force_n", "transport_peak_torque_nm",
      "transport_peak_utilization", "transport_peak_sustained_utilization",
      "insertion_peak_force_n", "insertion_peak_torque_nm",
      "insertion_peak_utilization", "insertion_peak_sustained_utilization",
      "transport_two_finger_peak_sustained_utilization",
      "insertion_two_finger_peak_sustained_utilization",
      "transport_vacuum_peak_sustained_utilization",
      "insertion_vacuum_peak_sustained_utilization",
      "transport_table_peak_utilization", "insertion_table_peak_utilization",
      "transport_table_peak_sustained_utilization",
      "insertion_table_peak_sustained_utilization",
      "num_path_states", "plan", "execution_plan", "motion_path", "sim_report",
      "recording", "planning_log", "task_planning_log", "motion_planning_log", "sim_log",
  ]
  tmp = Path(str(path) + ".tmp")
  with tmp.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({key: row.get(key) for key in fields})
  os.replace(tmp, path)


def _guard(command, memory_gb):
  limiter = shutil.which("prlimit")
  if limiter is None:
    raise RuntimeError("GNU prlimit is required; refusing to run without the RAM guard")
  return [limiter, f"--as={int(memory_gb * 1024**3)}", "--", *command]


def _run(command, *, log, timeout, memory_gb, env, dry_run=False):
  guarded = _guard(command, memory_gb)
  log.parent.mkdir(parents=True, exist_ok=True)
  with log.open("w", encoding="utf-8") as stream:
    stream.write("[benchmark-command] " + shlex.join(guarded) + "\n")
    stream.flush()
    if dry_run:
      return 0, 0.0, "dry_run"
    start = time.monotonic()
    proc = subprocess.Popen(
        guarded, cwd=ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True)
    outcome = "completed"
    try:
      returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
      outcome = "timeout"
      os.killpg(proc.pid, signal.SIGTERM)
      try:
        proc.wait(timeout=10)
      except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
      returncode = 124
      stream.write(f"\n[runner] stage exceeded {timeout:g}s and its process group was killed\n")
    return returncode, time.monotonic() - start, outcome


def _motion_metrics(path):
  with Path(path).open(encoding="utf-8") as stream:
    payload = json.load(stream)
  states = payload.get("states", payload) if isinstance(payload, dict) else payload
  qs = [state["q"] if isinstance(state, dict) else state[0] for state in states]
  length = 0.0
  for before, after in zip(qs, qs[1:]):
    if len(before) != len(after):
      raise ValueError("motion-path joint-vector dimension changes between states")
    length += math.sqrt(sum((float(b) - float(a)) ** 2 for a, b in zip(before, after)))
  return length, len(qs)


def _num_actions(plan):
  with Path(plan).open(encoding="utf-8") as stream:
    return len(json.load(stream)["plan"])


def _assembly_targets(plan):
  with Path(plan).open(encoding="utf-8") as stream:
    actions = json.load(stream)["plan"]
  return {
      action[2] for action in actions
      if isinstance(action, list) and len(action) > 2 and action[0] in {"assemble", "screw"}
  }


def _execution_progress(plan, report):
  """Return prefix progress, stopping immediately before the first failed result."""
  assembly_kinds = {"assemble", "screw"}
  targets = _assembly_targets(plan)
  results = report.get("results", [])
  failure_index = report.get("first_failure_index")
  stop = failure_index if isinstance(failure_index, int) else len(results)
  prefix = results[:stop]
  successful_tasks = 0
  assembled = set()
  for result in prefix:
    action = result.get("action", [])
    if not result.get("ok") or not action or action[0] in {"path", "servo"}:
      continue
    successful_tasks += 1
    if action[0] in assembly_kinds and len(action) > 2:
      assembled.add(action[2])
  failure_action = None
  if isinstance(failure_index, int) and 0 <= failure_index < len(results):
    failure_action = results[failure_index].get("action")
  return {
      "num_actions_ok_prefix": successful_tasks,
      "num_objects_assembled": len(assembled),
      "num_objects_to_assemble": len(targets),
      "first_failure_index": failure_index,
      "first_failure_action": failure_action,
  }


def _wrench_metrics(report):
  """Flatten phase- and gripper-specific grasp loads into stable CSV columns."""
  block = report.get("grasp_wrench_metrics") or {}
  phases = block.get("phases") or {}
  out = {"wrench_window_s": block.get("window_s")}
  for phase in ("transport", "insertion"):
    metric = phases.get(phase) or {}
    out.update({
        f"{phase}_peak_force_n": metric.get("peak_force_n"),
        f"{phase}_peak_torque_nm": metric.get("peak_torque_nm"),
        f"{phase}_peak_utilization": metric.get("peak_utilization"),
        f"{phase}_peak_sustained_utilization":
            metric.get("peak_sustained_utilization"),
    })
  by_kind = block.get("by_gripper_kind") or {}
  for kind in ("two_finger", "vacuum"):
    kind_phases = by_kind.get(kind) or {}
    for phase in ("transport", "insertion"):
      metric = kind_phases.get(phase) or {}
      out[f"{phase}_{kind}_peak_sustained_utilization"] = \
          metric.get("peak_sustained_utilization")
  table = report.get("table_wrench_metrics") or {}
  out["table_mu"] = table.get("table_mu")
  for phase in ("transport", "insertion"):
    metric = (table.get("phases") or {}).get(phase) or {}
    out[f"{phase}_table_peak_utilization"] = metric.get("peak_utilization")
    out[f"{phase}_table_peak_sustained_utilization"] = \
        metric.get("peak_sustained_utilization")
  return out


def _execution_plan(source, destination, *, variant, physical_force):
  """Write a replay-only copy under the common physical disturbance.

  Gravity-only action sequences may deliberately violate the full process-wrench successor
  gate.  Reconstructing them with ``wrench=True`` and the restored force would reject them
  before MuJoCo gets to test them.  ``wrench=False`` changes only symbolic state replay here;
  the simulator still restores the saved grasp caps and physically applies both gravity and
  the explicitly requested process wrench.
  """
  with Path(source).open(encoding="utf-8") as stream:
    payload = json.load(stream)
  planned = dict(payload.get("config", {}))
  cfg = dict(planned)
  cfg["insert_force"] = float(physical_force)
  if variant == "gravity":
    cfg["wrench"] = False
  payload["config"] = cfg
  payload["benchmark_execution_override"] = {
      "purpose": "common-physics wrench-planning ablation",
      "planning_variant": variant,
      "planning_wrench": bool(planned.get("wrench", False)),
      "planning_insert_force": planned.get("insert_force"),
      "execution_state_replay_wrench": bool(cfg.get("wrench", False)),
      "execution_insert_force": float(physical_force),
      "note": ("wrench is disabled only while reconstructing the already-certified symbolic "
               "state sequence; MuJoCo gravity, grasp breakaway, and process loads remain on"),
  }
  _write_json(destination, payload)


def _aggregate(rows, seeds, scenes=None, variants=None):
  out = []
  scenes = list(SCENES) if scenes is None else list(scenes)
  variants = list(VARIANTS) if variants is None else list(variants)
  for scene in scenes:
    scene_spec = SCENES[scene]
    for variant in variants:
      variant_spec = VARIANTS[variant]
      cell = [r for r in rows if r["scene"] == scene and r["variant"] == variant]
      planned = [r for r in cell if r.get("num_actions") is not None]
      executed = [r for r in cell if r.get("num_objects_assembled") is not None]
      succeeded = sum(bool(r.get("success")) for r in cell)
      def mean(key, source=planned):
        vals = [float(r[key]) for r in source if r.get(key) is not None]
        return statistics.fmean(vals) if vals else None
      kind_util = {}
      for kind in ("two_finger", "vacuum"):
        for phase in ("transport", "insertion"):
          value = mean(f"{phase}_{kind}_peak_sustained_utilization", executed)
          # Backward-compatible recovery for old reports from homogeneous-tool scenes. Mixed
          # Chair data cannot honestly be split and remains blank until its simulations rerun.
          if value is None and scene_spec["gripper_kinds"] == (kind,):
            value = mean(f"{phase}_peak_sustained_utilization", executed)
          kind_util[f"{phase}_{kind}_peak_sustained_utilization_mean"] = value
      out.append({
          "scene": scene, "environment": scene_spec["label"], "variant": variant,
          "planning_model": variant_spec["label"], "trials": len(cell),
          "requested_trials": len(seeds), "plans_found": len(planned),
          "successes": succeeded,
          "success_percent": 100.0 * succeeded / len(cell) if cell else None,
          "actions_mean": mean("num_actions"),
          "path_length_rad_mean": mean("path_length_rad"),
          "actions_ok_prefix_mean": mean("num_actions_ok_prefix", executed),
          "objects_assembled_mean": mean("num_objects_assembled", executed),
          "objects_to_assemble_mean": mean("num_objects_to_assemble", executed),
          # Average per-run peaks, rather than pooling samples: otherwise a longer path would
          # receive more statistical weight merely because it contains more physics ticks.
          "transport_peak_force_n_mean": mean("transport_peak_force_n", executed),
          "transport_peak_torque_nm_mean": mean("transport_peak_torque_nm", executed),
          "transport_peak_sustained_utilization_mean":
              mean("transport_peak_sustained_utilization", executed),
          "insertion_peak_force_n_mean": mean("insertion_peak_force_n", executed),
          "insertion_peak_torque_nm_mean": mean("insertion_peak_torque_nm", executed),
          "insertion_peak_sustained_utilization_mean":
              mean("insertion_peak_sustained_utilization", executed),
          "insertion_table_peak_sustained_utilization_mean":
              mean("insertion_table_peak_sustained_utilization", executed),
          "insertion_table_peak_utilization_mean":
              mean("insertion_table_peak_utilization", executed),
          **kind_util,
          "task_planning_wall_s_mean": mean("task_planning_wall_s", cell),
          "motion_planning_wall_s_mean": mean("motion_planning_wall_s", cell),
          "simulation_wall_s_mean": mean("sim_wall_s", cell),
          "failure_stages": {
              stage: sum(r.get("failure_stage") == stage for r in cell)
              for stage in sorted({r.get("failure_stage") for r in cell
                                   if r.get("failure_stage")})
          },
      })
  return out


def _fmt_mean(value, digits):
  if value is None:
    return "--"
  return f"{value:.{digits}f}"


def _fmt_util(row, phase, kind):
  """Render utilization; parentheses mark a prefix censored by execution failure."""
  value = row.get(f"{phase}_{kind}_peak_sustained_utilization_mean")
  if value is None:
    return "--"
  number = f"{100.0 * float(value):.0f}"
  rendered = rf"$\mathbf{{{number}}}$" if float(value) > 1.0 else number
  if row.get("success_percent") is not None and float(row["success_percent"]) < 100.0:
    return rf"({rendered})"
  return rendered


def _fmt_task_progress(row):
  """Format mean successfully executed task prefix over mean planned tasks."""
  done = row.get("actions_ok_prefix_mean")
  total = row.get("actions_mean")
  if done is None or total is None:
    return "--"

  def compact(value):
    rounded = round(float(value))
    return str(rounded) if abs(float(value) - rounded) < 1e-9 else f"{float(value):.1f}"

  return f"{compact(done)}/{compact(total)}"


def _fmt_failures(row, repeats):
  """Compact first-failure summary for the narrow paper table."""
  counts = row.get("failure_stages") or {}
  if not counts:
    return "--"
  labels = {
      "dropped": "drop",
      "slid": "slide",
      "not_seated": "seat",
      "grasp_broke": "grasp",
      "physical_execution": "physical",
      "task_planning": "task",
      "task_planning_timeout": "task t/o",
      "motion_planning": "MP",
      "motion_planning_timeout": "MP t/o",
      "simulation_timeout": "sim t/o",
      "simulator_preflight_or_crash": "sim err.",
  }
  fields = []
  for stage, count in sorted(counts.items()):
    label = labels.get(stage, stage.replace("_", " "))
    fields.append(label if repeats == 1 else f"{label} ({count})")
  return ", ".join(fields)


def _write_tex(path, summary, repeats, scenes=None):
  scenes = list(SCENES) if scenes is None else list(scenes)
  nmodels = max((sum(row["scene"] == scene for row in summary) for scene in scenes), default=1)
  complete = all(row["trials"] == row["requested_trials"] for row in summary)
  if complete:
    caption = (r"\caption{End-to-end simulation success of plans computed with different support "
               rf"models. We report success and mean peak support utilization over {repeats} "
               + (r"run. " if repeats == 1 else r"runs. ")
               + r"Full includes gravity and the process wrench. Grasp utilization is the "
               + r"maximum 20\,ms-averaged load as a percentage of the directional limit, "
               + r"split by execution phase and gripper type; bold values exceed the limit. A dash "
               + r"denotes an absent gripper type. Tasks reports the mean successfully "
               + r"executed prefix over the mean number of planned tasks. Parenthesized "
               + r"utilizations are prefix-censored because execution did not complete.}")
  else:
    caption = (r"\caption{Interim end-to-end simulation results. Success is evaluated over "
               r"completed runs; unrun cells are shown as --. The load columns report the "
               r"maximum 20\,ms-averaged utilization as percentages of the directional "
               r"grasp limits, split by execution phase and gripper type.}")
  # Keep the result readable at the native width of one paper column.  Short
  # labels and tabular* avoid scaling the entire table with a resizebox.
  model_labels = {
      "Topological": "Topol.",
      "Gravity only": "Gravity",
      "Gravity + process wrench": "Full",
  }
  lines = [
      r"\begin{table}[t]",
      r"\centering",
      caption,
      r"\begingroup",
      r"\footnotesize",
      r"\setlength{\tabcolsep}{1.5pt}",
      r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrlccccc@{}}",
      r"\toprule",
      r"Env. & Model & Succ. & Tasks & Fail. & \multicolumn{2}{c}{Transport util.} "
      r"& \multicolumn{2}{c}{Insertion util.} \\",
      r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
      r" & & [\%] & & & 2F & Vac. & 2F & Vac. \\",
      r"\midrule",
  ]
  for si, scene in enumerate(scenes):
    spec = SCENES[scene]
    cells = [x for x in summary if x["scene"] == scene]
    for vi, row in enumerate(cells):
      env = rf"\multirow{{{nmodels}}}{{*}}{{{spec['label']}}}" if vi == 0 else ""
      lines.append(
          f"{env} & {model_labels.get(row['planning_model'], row['planning_model'])} & "
          f"{_fmt_mean(row['success_percent'], 0)} & "
          f"{_fmt_task_progress(row)} & "
          f"{_fmt_failures(row, repeats)} & "
          f"{_fmt_util(row, 'transport', 'two_finger')} & "
          f"{_fmt_util(row, 'transport', 'vacuum')} & "
          f"{_fmt_util(row, 'insertion', 'two_finger')} & "
          f"{_fmt_util(row, 'insertion', 'vacuum')} \\\\")
    if si + 1 < len(scenes):
      lines.append(r"\midrule")
  lines += [r"\bottomrule", r"\end{tabular*}", r"\endgroup",
            r"\label{tab:wrench_sim}", r"\end{table}"]
  Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path,
                      default=ROOT / "benchmarks/results/wrench-sim-ablation")
  parser.add_argument("--scenes", nargs="+", choices=tuple(SCENES), default=list(SCENES))
  parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS),
                      default=list(VARIANTS))
  parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
  parser.add_argument("--physical-force", type=float, default=10.0, metavar="N")
  parser.add_argument("--mp-runtime", type=float, default=100.0, metavar="SEC",
                      help=("BiRRT* optimization budget for the saved task plan "
                            "(default 100; Cross/Cube/Chair first solved within 30s in the "
                            "recorded benchmark runs)"))
  parser.add_argument("--task-timeout", type=float, default=300.0, metavar="SEC",
                      help="wall-time guard for task planning alone (default 300)")
  parser.add_argument("--motion-timeout", type=float, default=450.0, metavar="SEC",
                      help="wall-time guard around the MP consultation plus export (default 450)")
  parser.add_argument("--sim-timeout", type=float, default=600.0, metavar="SEC")
  parser.add_argument("--physics-step", type=float, choices=(0.002, 0.004), default=0.004,
                      metavar="SEC",
                      help="MuJoCo/control timestep passed to sim.py (wrench benchmark "
                           "default 0.004; pass 0.002 for the conservative replay baseline)")
  parser.add_argument("--execution-timing", choices=("lockstep", "phase"),
                      default="lockstep",
                      help="simulation timing policy (default: lockstep)")
  parser.add_argument("--memory-gb", type=float, default=4.0, metavar="GB")
  parser.add_argument(
      "--jobs", type=int, default=1, metavar="N",
      help=("run up to N seeds concurrently; each seed keeps its selected scene/model trials "
            "sequential and writes to an isolated worker directory (default: 1)"))
  parser.add_argument("--resume", action="store_true")
  parser.add_argument(
      "--reuse-task-plans", type=Path, metavar="RESULT_DIR",
      help=("reuse plan.json files from RESULT_DIR/runs/<scene>__<variant>__seed-<seed> "
            "and rerun only motion planning and simulation"))
  parser.add_argument(
      "--reuse-motion-plans", type=Path, metavar="RESULT_DIR",
      help=("reuse plan.json and mp.json from RESULT_DIR and rerun only simulation; useful "
            "when changing execution-only ablation settings or recorded metrics"))
  parser.add_argument(
      "--insertion-only-grasp-check", action=argparse.BooleanOptionalAction, default=False,
      help=("measure grasp utilization throughout execution, but allow overload to break a "
            "grasp only during the insertion interval; physical gravity remains enabled"))
  parser.add_argument(
      "--abort-on-action-failure", action=argparse.BooleanOptionalAction, default=True,
      help=("stop at the first failed action so loads from a corrupted downstream scene do "
            "not contaminate benchmark utilization (default: enabled)"))
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--record", choices=("none", "failures", "all"), default="none",
                      help="NPZ retention policy; recording is disabled by default")
  parser.add_argument(
      "--animate-grippers", action=argparse.BooleanOptionalAction, default=True,
      help=("animate and wait for physical two-finger closure (default: enabled; use "
            "--no-animate-grippers only for the historical fixed-open-hand ablation)"))
  args = parser.parse_args(argv)
  if not args.seeds:
    parser.error("--seeds cannot be empty")
  if args.jobs < 1:
    parser.error("--jobs must be a positive integer")
  if args.reuse_task_plans and args.reuse_motion_plans:
    parser.error("--reuse-task-plans and --reuse-motion-plans are mutually exclusive")
  if args.jobs > 1 and (args.reuse_task_plans or args.reuse_motion_plans):
    parser.error("parallel runs cannot currently reuse task or motion plans")
  if args.dry_run and args.reuse_motion_plans:
    parser.error("--dry-run cannot be combined with --reuse-motion-plans")
  if any(x <= 0 for x in (args.mp_runtime, args.task_timeout, args.motion_timeout,
                          args.sim_timeout, args.memory_gb)):
    parser.error("runtime, timeout, and memory limits must be positive")
  if args.physical_force < 0:
    parser.error("--physical-force must be non-negative")
  return args


def _sequential_main(args):
  out = args.output.resolve()
  runs_dir = out / "runs"
  runs_dir.mkdir(parents=True, exist_ok=True)
  lock_stream = (out / ".runner.lock").open("w", encoding="utf-8")
  try:
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError as exc:
    raise SystemExit(f"another ablation runner is already writing {out}") from exc
  env = os.environ.copy()
  env.update({
      "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1",
      "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
      "NUMBA_CACHE_DIR": "/tmp/assembly-numba-cache",
      "MPLCONFIGDIR": "/tmp/assembly-mpl-cache",
  })
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "scenes": args.scenes, "variants": args.variants, "seeds": args.seeds,
      "physical_insert_force_n": args.physical_force,
      "simulation_process_noise": "nominal", "search_profile": "standard",
      "motion_profile": "standard", "sim_profile": "standard",
      "physics_step_s": args.physics_step,
      "simulation_gravity": True,
      "motion_gravity_gate": {
          name: bool(spec["mp_wrench_gate"]) for name, spec in VARIANTS.items()},
      "execution_timing": args.execution_timing,
      "mp_runtime_s": args.mp_runtime, "task_timeout_s": args.task_timeout,
      "motion_timeout_s": args.motion_timeout,
      "sim_timeout_s": args.sim_timeout, "memory_gb": args.memory_gb,
      "record": args.record, "animate_grippers": args.animate_grippers,
      "reused_task_plans_from": (str(args.reuse_task_plans.resolve())
                                  if args.reuse_task_plans else None),
      "reused_motion_plans_from": (str(args.reuse_motion_plans.resolve())
                                    if args.reuse_motion_plans else None),
      "insertion_only_grasp_check": bool(args.insertion_only_grasp_check),
      "abort_on_action_failure": bool(args.abort_on_action_failure),
      "path_length_definition": "sum of Euclidean joint-space increments over saved MP states",
  }
  manifest_path = out / "manifest.json"
  summary_path = out / "summary.json"
  if args.resume and manifest_path.is_file():
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    semantic_keys = (
        "scenes", "variants", "seeds", "physical_insert_force_n",
        "simulation_process_noise", "search_profile", "motion_profile", "sim_profile",
        "physics_step_s",
        "simulation_gravity", "motion_gravity_gate", "execution_timing", "animate_grippers",
        "reused_task_plans_from", "reused_motion_plans_from",
        "insertion_only_grasp_check", "abort_on_action_failure",
    )
    mismatches = [key for key in semantic_keys if previous.get(key) != manifest.get(key)]
    if mismatches:
      raise SystemExit(
          "refusing to mix incompatible runs in one output directory; changed: "
          + ", ".join(mismatches) + ". Choose a fresh --output directory.")
  elif summary_path.is_file():
    raise SystemExit(f"{summary_path} already exists; use --resume or a fresh --output")
  _write_json(manifest_path, manifest)

  reused_motion_rows = {}
  if args.reuse_motion_plans:
    source_root = args.reuse_motion_plans.resolve()
    if source_root == out:
      raise SystemExit("--reuse-motion-plans must name a different result directory")
    source_summary = source_root / "summary.json"
    if not source_summary.is_file():
      raise SystemExit(f"reusable result summary not found: {source_summary}")
    source_payload = json.loads(source_summary.read_text(encoding="utf-8"))
    reused_motion_rows = {
        (row["scene"], row["variant"], int(row["seed"])): row
        for row in source_payload.get("runs", [])
    }

  existing = {}
  if args.resume and summary_path.is_file():
    for row in json.loads(summary_path.read_text(encoding="utf-8")).get("runs", []):
      if row.get("failure_stage") in RETRYABLE_INFRA_FAILURES:
        print(f"[wrench-sim] retrying incomplete infrastructure outcome: "
              f"{row['scene']}/{row['variant']}/seed-{row['seed']} "
              f"({row['failure_stage']})", flush=True)
        continue
      if row.get("num_objects_assembled") is None:
        plan_path = out / row["plan"]
        report_path = out / row["sim_report"]
        if plan_path.is_file():
          row.update(num_actions_ok_prefix=0, num_objects_assembled=0,
                     num_objects_to_assemble=len(_assembly_targets(plan_path)),
                     first_failure_index=None, first_failure_action=None)
          if report_path.is_file():
            try:
              report = json.loads(report_path.read_text(encoding="utf-8"))
              row.update(_execution_progress(plan_path, report))
            except Exception:
              pass
      # Schema-4 result directories can be resumed after re-running only their simulations:
      # always import a newly available metric block, even when progress was already present.
      report_path = out / row["sim_report"]
      if report_path.is_file():
        try:
          row.update(_wrench_metrics(json.loads(report_path.read_text(encoding="utf-8"))))
        except Exception:
          pass
      existing[(row["scene"], row["variant"], int(row["seed"]))] = row
  rows = []
  total = len(args.scenes) * len(args.variants) * len(args.seeds)
  ordinal = 0
  for scene in args.scenes:
    for variant in args.variants:
      for seed in args.seeds:
        ordinal += 1
        key = (scene, variant, seed)
        if key in existing:
          rows.append(existing[key])
          print(f"[wrench-sim] {ordinal}/{total} {scene}/{variant}/seed-{seed}: resumed",
                flush=True)
          continue
        tag = _slug(f"{scene}__{variant}__seed-{seed}")
        run_dir = runs_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        plan = run_dir / "plan.json"
        execution_plan = run_dir / "execution-plan.json"
        motion = run_dir / "mp.json"
        sim_report = run_dir / "sim-report.json"
        recording = run_dir / "replay.npz"
        task_planning_log = run_dir / "task-planning.log"
        motion_planning_log = run_dir / "motion-planning.log"
        sim_log = run_dir / "sim.log"
        # A retry must never accept an artifact left by an interrupted earlier attempt.
        for stale in (plan, execution_plan, motion, sim_report, recording):
          if stale.is_file():
            stale.unlink()

        v = VARIANTS[variant]
        pforce = args.physical_force if v["planning_force"] == "physical" \
            else (0.0 if v["planning_force"] == 0.0 else args.physical_force)
        task_cmd = [
            sys.executable, "-u", str(ROOT / "main.py"),
            "--solve", "--search-profile", "standard",
            *SCENES[scene]["args"], *v["args"],
            "--wrench-profile", "max_robotiq", "--insert-force", str(pforce),
            "--insertion-noise", "0", "--seed", str(seed),
            "--save", str(plan), "--quiet-rai", "--no-view",
        ]
        motion_cmd = [
            sys.executable, "-u", str(ROOT / "main.py"),
            "--config", str(plan), "--load-plan", str(plan),
            "--motion-plan", "--motion-profile", "standard",
            ("--mp-wrench-gate" if v["mp_wrench_gate"] else "--no-mp-wrench-gate"),
            "--mp-runtime", str(args.mp_runtime), "--mp-seed", str(seed),
            "--mp-save", str(motion), "--quiet-rai", "--no-view",
        ]
        reused_row = reused_motion_rows.get(key)
        if args.reuse_motion_plans:
          source_run = args.reuse_motion_plans.resolve() / "runs" / tag
          source_plan = source_run / "plan.json"
          source_motion = source_run / "mp.json"
          missing = [str(path) for path in (source_plan, source_motion) if not path.is_file()]
          if missing:
            raise SystemExit("reusable motion-plan artifact(s) not found: " + ", ".join(missing))
          if reused_row is None:
            raise SystemExit(f"reusable summary has no row for {scene}/{variant}/seed-{seed}")
          shutil.copy2(source_plan, plan)
          shutil.copy2(source_motion, motion)
          task_planning_log.write_text(
              f"[runner] reused task plan: {source_plan}\n", encoding="utf-8")
          motion_planning_log.write_text(
              f"[runner] reused motion plan: {source_motion}\n", encoding="utf-8")
          print(f"[wrench-sim] {ordinal}/{total} {scene}/{variant}/seed-{seed}: "
                "reused task and motion plans", flush=True)
          code = 0
          task_wall = float(reused_row.get("task_planning_wall_s") or 0.0)
          task_outcome = "completed"
        elif args.reuse_task_plans and not args.dry_run:
          source_plan = (args.reuse_task_plans.resolve() / "runs" / tag / "plan.json")
          if not source_plan.is_file():
            raise SystemExit(f"reusable task plan not found: {source_plan}")
          shutil.copy2(source_plan, plan)
          task_planning_log.write_text(
              f"[runner] reused task plan: {source_plan}\n", encoding="utf-8")
          print(f"[wrench-sim] {ordinal}/{total} {scene}/{variant}/seed-{seed}: "
                f"reused task plan", flush=True)
          code, task_wall, task_outcome = 0, 0.0, "completed"
        else:
          print(f"[wrench-sim] {ordinal}/{total} {scene}/{variant}/seed-{seed}: "
                "task planning", flush=True)
          code, task_wall, task_outcome = _run(
              task_cmd, log=task_planning_log, timeout=args.task_timeout,
              memory_gb=args.memory_gb, env=env, dry_run=args.dry_run)
        row = {
            "scene": scene, "variant": variant, "seed": seed, "status": task_outcome,
            "success": False, "failure_stage": None, "returncode": code,
            "planning_wall_s": round(task_wall, 6),
            "task_planning_wall_s": round(task_wall, 6),
            "motion_planning_wall_s": (
                float(reused_row.get("motion_planning_wall_s") or 0.0)
                if reused_row is not None else None),
            "sim_wall_s": None,
            "num_actions": None, "path_length_rad": None, "num_path_states": None,
            "num_actions_ok_prefix": None, "num_objects_assembled": None,
            "num_objects_to_assemble": None, "first_failure_index": None,
            "first_failure_action": None,
            "wrench_window_s": None,
            "transport_peak_force_n": None, "transport_peak_torque_nm": None,
            "transport_peak_utilization": None,
            "transport_peak_sustained_utilization": None,
            "insertion_peak_force_n": None, "insertion_peak_torque_nm": None,
            "insertion_peak_utilization": None,
            "insertion_peak_sustained_utilization": None,
            "transport_two_finger_peak_sustained_utilization": None,
            "insertion_two_finger_peak_sustained_utilization": None,
            "transport_vacuum_peak_sustained_utilization": None,
            "insertion_vacuum_peak_sustained_utilization": None,
            "transport_table_peak_sustained_utilization": None,
            "insertion_table_peak_sustained_utilization": None,
            "transport_table_peak_utilization": None,
            "insertion_table_peak_utilization": None,
            "table_mu": None,
            "plan": str(plan.relative_to(out)),
            "execution_plan": str(execution_plan.relative_to(out)),
            "motion_path": str(motion.relative_to(out)),
            "sim_report": str(sim_report.relative_to(out)), "recording": "",
            "planning_log": "",
            "task_planning_log": str(task_planning_log.relative_to(out)),
            "motion_planning_log": str(motion_planning_log.relative_to(out)),
            "sim_log": str(sim_log.relative_to(out)),
        }
        if args.dry_run:
          _run(motion_cmd, log=motion_planning_log, timeout=args.motion_timeout,
               memory_gb=args.memory_gb, env=env, dry_run=True)
          row.update(status="dry_run", failure_stage="dry_run")
        elif code != 0 or not plan.is_file():
          if plan.is_file():
            try:
              row["num_actions"] = _num_actions(plan)
              row["num_actions_ok_prefix"] = 0
              row["num_objects_assembled"] = 0
              row["num_objects_to_assemble"] = len(_assembly_targets(plan))
            except Exception:
              pass
          failed = "task_planning_timeout" if task_outcome == "timeout" else "task_planning"
          row.update(status="planning_failed",
                     failure_stage=failed)
        else:
          try:
            row["num_actions"] = _num_actions(plan)
            row["num_actions_ok_prefix"] = 0
            row["num_objects_assembled"] = 0
            row["num_objects_to_assemble"] = len(_assembly_targets(plan))
          except Exception as exc:
            row.update(status="artifact_invalid", failure_stage="artifact_parse")
            motion_planning_log.write_text(
                f"[runner] invalid task-plan artifact: {exc}\n", encoding="utf-8")
          else:
            if args.reuse_motion_plans:
              print(f"[wrench-sim] {ordinal}/{total} {scene}/{variant}/seed-{seed}: "
                    "motion planning reused", flush=True)
              mcode = 0
              motion_wall = float(reused_row.get("motion_planning_wall_s") or 0.0)
              motion_outcome = "completed"
            else:
              print(f"[wrench-sim] {ordinal}/{total} {scene}/{variant}/seed-{seed}: "
                    "motion planning", flush=True)
              mcode, motion_wall, motion_outcome = _run(
                  motion_cmd, log=motion_planning_log, timeout=args.motion_timeout,
                  memory_gb=args.memory_gb, env=env, dry_run=False)
            row["motion_planning_wall_s"] = round(motion_wall, 6)
            row["planning_wall_s"] = round(task_wall + motion_wall, 6)
            row["returncode"] = mcode
            if mcode != 0 or not motion.is_file():
              failed = ("motion_planning_timeout" if motion_outcome == "timeout"
                        else "motion_planning")
              row.update(status="planning_failed", failure_stage=failed)
            else:
              try:
                plen, nstates = _motion_metrics(motion)
                row["path_length_rad"] = round(plen, 9)
                row["num_path_states"] = nstates
                _execution_plan(plan, execution_plan, variant=variant,
                                physical_force=args.physical_force)
              except Exception as exc:
                row.update(status="artifact_invalid", failure_stage="artifact_parse")
                sim_log.write_text(
                    f"[runner] invalid planner artifact: {exc}\n", encoding="utf-8")
              else:
                sim_cmd = [
                    sys.executable, "-u", str(ROOT / "sim.py"), "run", str(execution_plan),
                    "--sim-profile", "standard", "--mp-path", str(motion),
                    "--physics-step", str(args.physics_step),
                    "--apply-process-wrench", "--process-noise", "nominal",
                    "--process-noise-seed", str(seed), "--report-json", str(sim_report),
                ]
                if args.execution_timing == "lockstep":
                  sim_cmd += ["--no-phase-schedule", "--no-phase-smooth",
                              "--no-phase-batch", "--no-spline"]
                if args.insertion_only_grasp_check:
                  sim_cmd += ["--insertion-only-grasp-check"]
                if args.abort_on_action_failure:
                  sim_cmd += ["--abort-on-action-failure"]
                if args.animate_grippers:
                  sim_cmd += ["--animate-grippers", "--wait-for-gripper"]
                if args.record != "none":
                  sim_cmd += ["--record", str(recording)]
                  if args.record == "failures":
                    sim_cmd += ["--record-on-failure"]
                print(f"[wrench-sim] {ordinal}/{total} {scene}/{variant}/seed-{seed}: sim",
                      flush=True)
                scode, sim_wall, sim_outcome = _run(
                    sim_cmd, log=sim_log, timeout=args.sim_timeout,
                    memory_gb=args.memory_gb, env=env, dry_run=False)
                row["sim_wall_s"] = round(sim_wall, 6)
                row["returncode"] = scode
                report = {}
                if sim_report.is_file():
                  try:
                    report = json.loads(sim_report.read_text(encoding="utf-8"))
                  except Exception:
                    pass
                if report:
                  row.update(_execution_progress(plan, report))
                  row.update(_wrench_metrics(report))
                success = bool(report.get("success", False))
                row["success"] = success
                if success:
                  row.update(status="success", failure_stage=None)
                elif sim_outcome == "timeout":
                  row.update(status="sim_failed", failure_stage="simulation_timeout")
                elif report:
                  row.update(status="sim_failed",
                             failure_stage=report.get("first_failure") or "physical_execution")
                else:
                  row.update(status="sim_failed",
                             failure_stage="simulator_preflight_or_crash")
                if recording.is_file():
                  keep = args.record == "all" or (args.record == "failures" and not success)
                  if keep:
                    row["recording"] = str(recording.relative_to(out))
                  else:
                    recording.unlink()

        rows.append(row)
        aggregate = _aggregate(rows, args.seeds, args.scenes, args.variants)
        _write_json(summary_path, {"schema_version": SCHEMA_VERSION,
                                   "manifest": manifest, "runs": rows,
                                   "aggregate": aggregate})
        _write_csv(out / "summary.csv", rows)
        _write_json(out / "aggregate.json", aggregate)
        _write_tex(out / "table.tex", aggregate, len(args.seeds), args.scenes)
        print(f"  -> {row['status']}"
              + (f" ({row['failure_stage']})" if row.get("failure_stage") else ""),
              flush=True)

  # Also refresh derived artifacts when every cell was resumed.  Previously the per-cell
  # `continue` path left table.tex stale after presentation-only changes to this runner.
  aggregate = _aggregate(rows, args.seeds, args.scenes, args.variants)
  _write_json(summary_path, {"schema_version": SCHEMA_VERSION,
                             "manifest": manifest, "runs": rows,
                             "aggregate": aggregate})
  _write_csv(out / "summary.csv", rows)
  _write_json(out / "aggregate.json", aggregate)
  _write_tex(out / "table.tex", aggregate, len(args.seeds), args.scenes)

  if args.dry_run:
    return 0
  return 0 if all(r.get("success") for r in rows) else 1


_ARTIFACT_FIELDS = (
    "plan", "execution_plan", "motion_path", "sim_report", "recording", "planning_log",
    "task_planning_log", "motion_planning_log", "sim_log",
)


def _worker_command(args, seed, output):
  """Recreate one single-seed invocation without relying on shell quoting or ambient CLI."""
  command = [
      sys.executable, "-u", str(Path(__file__).resolve()),
      "--output", str(output),
      "--scenes", *args.scenes,
      "--variants", *args.variants,
      "--seeds", str(seed),
      "--physical-force", str(args.physical_force),
      "--mp-runtime", str(args.mp_runtime),
      "--task-timeout", str(args.task_timeout),
      "--motion-timeout", str(args.motion_timeout),
      "--sim-timeout", str(args.sim_timeout),
      "--physics-step", str(args.physics_step),
      "--execution-timing", args.execution_timing,
      "--memory-gb", str(args.memory_gb),
      "--record", args.record,
      "--jobs", "1",
  ]
  if args.resume:
    command.append("--resume")
  if args.insertion_only_grasp_check:
    command.append("--insertion-only-grasp-check")
  if not args.abort_on_action_failure:
    command.append("--no-abort-on-action-failure")
  if not args.animate_grippers:
    command.append("--no-animate-grippers")
  if args.dry_run:
    command.append("--dry-run")
  return command


def _parallel_main(args):
  """Run independent seeds concurrently and merge their checkpointed result trees.

  Each worker uses the ordinary sequential runner and therefore retains the established
  subprocess, timeout, and address-space guards.  Worker artifacts stay isolated below
  ``.workers``; merged rows carry paths relative to the requested top-level result directory.
  """
  out = args.output.resolve()
  out.mkdir(parents=True, exist_ok=True)
  if (out / "summary.json").is_file() and not args.resume:
    raise SystemExit(f"{out / 'summary.json'} already exists; use --resume or a fresh --output")
  lock_stream = (out / ".runner.lock").open("w", encoding="utf-8")
  try:
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError as exc:
    raise SystemExit(f"another ablation runner is already writing {out}") from exc

  worker_root = out / ".workers"
  worker_root.mkdir(parents=True, exist_ok=True)

  def run_seed(seed):
    destination = worker_root / f"seed-{seed}"
    command = _worker_command(args, seed, destination)
    log = destination / "runner.log"
    destination.mkdir(parents=True, exist_ok=True)
    with log.open("a" if args.resume else "w", encoding="utf-8") as stream:
      stream.write("[parallel-command] " + shlex.join(command) + "\n")
      stream.flush()
      completed = subprocess.run(
          command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True)
    summary = destination / "summary.json"
    if not summary.is_file():
      return seed, completed.returncode, [], f"worker produced no summary: {summary}"
    try:
      worker_manifest = json.loads(
          (destination / "manifest.json").read_text(encoding="utf-8"))
      payload = json.loads(summary.read_text(encoding="utf-8"))
      worker_rows = payload["runs"]
    except Exception as exc:
      return seed, completed.returncode, [], f"invalid worker summary {summary}: {exc}"
    expected_manifest = {
        "scenes": args.scenes, "variants": args.variants, "seeds": [seed],
        "physical_insert_force_n": args.physical_force,
        "physics_step_s": args.physics_step, "execution_timing": args.execution_timing,
        "record": args.record, "animate_grippers": args.animate_grippers,
        "insertion_only_grasp_check": bool(args.insertion_only_grasp_check),
        "abort_on_action_failure": bool(args.abort_on_action_failure),
    }
    mismatches = [key for key, value in expected_manifest.items()
                  if worker_manifest.get(key) != value]
    if mismatches:
      return seed, completed.returncode, [], (
          "worker result has incompatible manifest fields: " + ", ".join(mismatches))
    expected = len(args.scenes) * len(args.variants)
    error = None if len(worker_rows) == expected else (
        f"worker returned {len(worker_rows)}/{expected} rows")
    prefix = Path(".workers") / f"seed-{seed}"
    for row in worker_rows:
      for field in _ARTIFACT_FIELDS:
        value = row.get(field)
        if value:
          row[field] = str(prefix / value)
    return seed, completed.returncode, worker_rows, error

  gathered = []
  errors = []
  workers = min(args.jobs, len(args.seeds))
  print(f"[wrench-sim] parallel seed workers: {workers}; seeds={args.seeds}", flush=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(run_seed, seed): seed for seed in args.seeds}
    for future in concurrent.futures.as_completed(futures):
      seed, returncode, worker_rows, error = future.result()
      gathered.extend(worker_rows)
      if error:
        errors.append(f"seed {seed}: {error}")
      print(f"[wrench-sim] seed {seed}: {len(worker_rows)} rows, runner rc={returncode}"
            + (f", ERROR: {error}" if error else ""), flush=True)

  # Keep paper output deterministic regardless of worker completion order.
  scene_rank = {name: i for i, name in enumerate(args.scenes)}
  variant_rank = {name: i for i, name in enumerate(args.variants)}
  seed_rank = {seed: i for i, seed in enumerate(args.seeds)}
  gathered.sort(key=lambda row: (
      scene_rank[row["scene"]], variant_rank[row["variant"]], seed_rank[int(row["seed"])]))
  if gathered:
    first_manifest = json.loads(
        (worker_root / f"seed-{args.seeds[0]}" / "manifest.json").read_text(
            encoding="utf-8"))
    first_manifest["seeds"] = list(args.seeds)
    first_manifest["jobs"] = int(args.jobs)
    first_manifest["parallel_layout"] = ".workers/seed-<seed>"
    _write_json(out / "manifest.json", first_manifest)
    aggregate = _aggregate(gathered, args.seeds, args.scenes, args.variants)
    _write_json(out / "summary.json", {
        "schema_version": SCHEMA_VERSION, "manifest": first_manifest,
        "runs": gathered, "aggregate": aggregate,
    })
    _write_csv(out / "summary.csv", gathered)
    _write_json(out / "aggregate.json", aggregate)
    _write_tex(out / "table.tex", aggregate, len(args.seeds), args.scenes)
  if errors:
    print("[wrench-sim] incomplete workers:\n  " + "\n  ".join(errors), flush=True)
  expected_total = len(args.scenes) * len(args.variants) * len(args.seeds)
  lock_stream.close()
  if args.dry_run:
    return 0 if len(gathered) == expected_total and not errors else 1
  return 0 if (len(gathered) == expected_total and not errors
               and all(row.get("success") for row in gathered)) else 1


def main(argv=None):
  args = _parse_args(argv)
  return _parallel_main(args) if args.jobs > 1 else _sequential_main(args)


if __name__ == "__main__":
  raise SystemExit(main())
