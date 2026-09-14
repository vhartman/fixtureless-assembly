#!/usr/bin/env python3
"""Manifest-driven benchmark runner for assembly search.

The solver is deliberately executed in a fresh subprocess for every sample. This prevents
RAI, wrench, and IK state from leaking between variants or seeds. With ``--shared-grasp-db``,
one explicitly fingerprinted grasp table is constructed outside timing and then used as the
same immutable problem input by every variant; no process-local cache is shared. The runner
adds only search-output controls (no viewer, no witness optimization, no motion planning),
retains the raw log and replayable plan, and emits machine-readable records.

See benchmarks/README.md for the manifest schema and example commands.
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
import platform
import re
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
_GNU_TIME = Path("/usr/bin/time")
_PRLIMIT = Path("/usr/bin/prlimit")
_SOURCE_GLOBS = (
    # Conservative by design: --resume must invalidate when a scheduler, feasibility gate,
    # calibrated IK model, assembly mesh, or robot collision model changes. The former short
    # hand list omitted open_witness.py and every .g file, so precisely those changes could
    # reuse stale paper results.
    "*.py", "benchmarks/*.py", "ik_artifacts/*.py", "part_meshes/*",
    "src/rai/**/*.g", "src/rai/**/*.urdf", "src/rai/**/collision/*.stl",
)
_RESERVED = {
    "--save", "--load-plan", "--refresh-witnesses", "--motion-plan", "--mp-save",
    "--mp-export-save", "--mp-export-load", "--view", "--no-view", "--viser",
    "--mp-viser", "--optimize-witnesses", "--witness-sets", "--fuse-cograsp",
    "--no-fuse-cograsp", "--seed", "--solve",
}
_NUM = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def _read_json(path: Path):
  with path.open(encoding="utf-8") as f:
    return json.load(f)


def _write_json(path: Path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with tmp.open("w", encoding="utf-8") as f:
    json.dump(value, f, indent=2, sort_keys=True)
    f.write("\n")
  os.replace(tmp, path)


def _slug(value: str) -> str:
  out = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
  return out or "run"


def _check_args(args, where):
  if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
    raise ValueError(f"{where}.args must be a list of strings")
  for token in args:
    flag = token.split("=", 1)[0]
    if flag in _RESERVED:
      raise ValueError(
          f"{where} contains reserved runner-controlled flag {flag!r}; put only search and "
          "scene settings in the manifest")


def validate_manifest(data):
  if not isinstance(data, dict):
    raise ValueError("manifest root must be a JSON object")
  if data.get("schema_version") != SCHEMA_VERSION:
    raise ValueError(
        f"manifest schema_version must be {SCHEMA_VERSION}, got "
        f"{data.get('schema_version')!r}")
  if not isinstance(data.get("name"), str) or not data["name"].strip():
    raise ValueError("manifest needs a non-empty string 'name'")
  _check_args(data.get("defaults", []), "defaults")
  seeds = data.get("seeds", [0])
  if not seeds or not isinstance(seeds, list) or not all(isinstance(x, int) for x in seeds):
    raise ValueError("seeds must be a non-empty list of integers")
  timeout = data.get("timeout_s", 600)
  if not isinstance(timeout, (int, float)) or timeout <= 0:
    raise ValueError("timeout_s must be positive")
  cases = data.get("cases")
  if not isinstance(cases, list) or not cases:
    raise ValueError("manifest needs a non-empty cases list")
  matrix = data.get("matrix")
  if matrix is not None:
    if "variants" in data:
      raise ValueError("a manifest may define either variants or matrix, not both")
    if not isinstance(matrix, dict):
      raise ValueError("matrix must be an object with non-empty rows and columns")
    matrix_rows = matrix.get("rows")
    matrix_columns = matrix.get("columns")
    if not isinstance(matrix_rows, list) or not matrix_rows:
      raise ValueError("matrix.rows must be a non-empty list")
    if not isinstance(matrix_columns, list) or not matrix_columns:
      raise ValueError("matrix.columns must be a non-empty list")
    named_groups = (("cases", cases), ("matrix.rows", matrix_rows),
                    ("matrix.columns", matrix_columns))
  else:
    variants = data.get("variants", [{"name": "default", "args": []}])
    if not isinstance(variants, list) or not variants:
      raise ValueError("variants must be a non-empty list")
    named_groups = (("cases", cases), ("variants", variants))
  for group_name, rows in named_groups:
    names = set()
    for i, row in enumerate(rows):
      if not isinstance(row, dict) or not isinstance(row.get("name"), str):
        raise ValueError(f"{group_name}[{i}] needs a string name")
      if row["name"] in names:
        raise ValueError(f"duplicate {group_name} name {row['name']!r}")
      names.add(row["name"])
      _check_args(row.get("args", []), f"{group_name}[{i}]")
      if "seeds" in row and (not isinstance(row["seeds"], list)
                              or not all(isinstance(x, int) for x in row["seeds"])):
        raise ValueError(f"{group_name}[{i}].seeds must be a list of integers")
      for limit_name in ("max_address_space_gb", "max_rss_gb"):
        if (limit_name in row
            and (not isinstance(row[limit_name], (int, float))
                 or row[limit_name] <= 0)):
          raise ValueError(f"{group_name}[{i}].{limit_name} must be positive")
  for limit_name in ("max_address_space_gb", "max_rss_gb"):
    if (limit_name in data
        and (not isinstance(data[limit_name], (int, float))
             or data[limit_name] <= 0)):
      raise ValueError(f"{limit_name} must be positive")
  env = data.get("env", {})
  if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                           for k, v in env.items()):
    raise ValueError("env must map strings to strings")
  return data


def expand_runs(data, repeats=1, seeds=None):
  """Return the deterministic case x variant x seed product.

  ``repeats`` duplicates each logical cell with the same planner seed.  The duplicate
  processes are intentionally independent, so aggregation can report timing variation
  without changing the solver's sampled problem or search semantics. ``seeds`` is an
  experiment-wide override: unlike ``repeats``, it selects independent planner seeds and
  takes precedence over manifest-, case-, and variant-local seed lists.
  """
  if not isinstance(repeats, int) or repeats < 1:
    raise ValueError("repeats must be a positive integer")
  if seeds is not None:
    if (not isinstance(seeds, (list, tuple)) or not seeds
        or not all(isinstance(seed, int) for seed in seeds)):
      raise ValueError("seeds override must be a non-empty list of integers")
    seeds = list(seeds)
  runs = []
  defaults = list(data.get("defaults", []))
  matrix = data.get("matrix")
  if matrix is not None:
    variants = []
    for row in matrix["rows"]:
      if not row.get("enabled", True):
        continue
      for column in matrix["columns"]:
        if not column.get("enabled", True):
          continue
        variants.append({
            "name": f"{row['name']}__{column['name']}",
            "args": list(row.get("args", [])) + list(column.get("args", [])),
            "method": row["name"],
            "physics": column["name"],
            "tags": sorted(set(row.get("tags", [])) | set(column.get("tags", []))),
            "timeout_s": column.get("timeout_s", row.get("timeout_s")),
            "max_address_space_gb": column.get(
                "max_address_space_gb", row.get("max_address_space_gb")),
            "max_rss_gb": column.get("max_rss_gb", row.get("max_rss_gb")),
        })
  else:
    variants = data.get("variants", [{"name": "default", "args": []}])
  for case in data["cases"]:
    if not case.get("enabled", True):
      continue
    case_seeds = seeds if seeds is not None else case.get("seeds", data.get("seeds", [0]))
    for variant in variants:
      if not variant.get("enabled", True):
        continue
      run_seeds = case_seeds if seeds is not None else variant.get("seeds", case_seeds)
      for seed in run_seeds:
        for repeat in range(1, repeats + 1):
          rid = f"{_slug(case['name'])}__{_slug(variant['name'])}__seed-{seed}"
          if repeats > 1:
            rid += f"__rep-{repeat:02d}"
          timeout = variant.get("timeout_s")
          if timeout is None:
            timeout = case.get("timeout_s", data.get("timeout_s", 600))
          memory_gb = variant.get("max_address_space_gb")
          if memory_gb is None:
            memory_gb = case.get("max_address_space_gb", data.get("max_address_space_gb"))
          rss_gb = variant.get("max_rss_gb")
          if rss_gb is None:
            rss_gb = case.get("max_rss_gb", data.get("max_rss_gb"))
          run = {
              "run_id": rid,
              "case": case["name"],
              "variant": variant["name"],
              "seed": seed,
              "repeat": repeat,
              "timeout_s": float(timeout),
              "max_address_space_gb": float(memory_gb) if memory_gb is not None else None,
              "max_rss_gb": float(rss_gb) if rss_gb is not None else None,
              "args": defaults + list(case.get("args", [])) + list(variant.get("args", [])),
              "tags": sorted(set(data.get("tags", [])) | set(case.get("tags", []))
                             | set(variant.get("tags", []))),
          }
          if matrix is not None:
            run.update(method=variant["method"], physics=variant["physics"])
          runs.append(run)
  ids = [r["run_id"] for r in runs]
  if len(ids) != len(set(ids)):
    raise ValueError("expanded run ids are not unique; check case/variant names and seeds")
  return runs


def _command_for(spec, plan_path: Path):
  return [
      sys.executable, str(ROOT / "main.py"), *spec["args"],
      "--seed", str(spec["seed"]), "--solve", "--no-view", "--quiet-rai",
      "--optimize-witnesses", "off", "--witness-sets", "0", "--no-fuse-cograsp",
      "--save", str(plan_path),
  ]


def _guarded_command(spec, command):
  memory_gb = spec.get("max_address_space_gb")
  if memory_gb is None:
    return command
  if not _PRLIMIT.is_file():
    raise RuntimeError("manifest requests max_address_space_gb but /usr/bin/prlimit is absent")
  limit_bytes = int(float(memory_gb) * 1024 ** 3)
  return [str(_PRLIMIT), f"--as={limit_bytes}", "--", *command]


def _looks_like_out_of_memory(text: str) -> bool:
  """Recognize Python and native allocation failures emitted by the solver stack."""
  return "MemoryError" in text or "bad_alloc" in text


def _process_tree_pids(root_pid: int):
  """Return the live Linux process tree rooted at ``root_pid``."""
  seen = set()
  pending = [root_pid]
  while pending:
    pid = pending.pop()
    if pid in seen:
      continue
    seen.add(pid)
    try:
      children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
      continue
    pending.extend(int(child) for child in children)
  return seen


def _process_tree_rss_kb(root_pid: int):
  """Best-effort sum of resident memory for a process and all descendants."""
  total = 0
  for pid in _process_tree_pids(root_pid):
    try:
      for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
          total += int(line.split()[1])
          break
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
      pass
  return total


def _terminate_process_group(proc, grace_s=5):
  """Terminate a benchmark's isolated process group and return its exit status."""
  try:
    os.killpg(proc.pid, signal.SIGTERM)
  except ProcessLookupError:
    return proc.wait()
  try:
    return proc.wait(timeout=grace_s)
  except subprocess.TimeoutExpired:
    try:
      os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
    return proc.wait()


def _wait_with_guards(proc, timeout_s: float, max_rss_gb=None, poll_s=0.05):
  """Wait while enforcing wall-time and resident memory on the whole process tree."""
  deadline = time.monotonic() + timeout_s
  rss_limit_kb = (None if max_rss_gb is None
                  else int(float(max_rss_gb) * 1024 ** 2))
  sampled_peak_kb = 0
  while True:
    returncode = proc.poll()
    if returncode is not None:
      sampled_peak_kb = max(sampled_peak_kb, _process_tree_rss_kb(proc.pid))
      return returncode, False, False, sampled_peak_kb
    rss_kb = _process_tree_rss_kb(proc.pid)
    sampled_peak_kb = max(sampled_peak_kb, rss_kb)
    if rss_limit_kb is not None and rss_kb > rss_limit_kb:
      return _terminate_process_group(proc), False, True, sampled_peak_kb
    remaining = deadline - time.monotonic()
    if remaining <= 0:
      return _terminate_process_group(proc), True, False, sampled_peak_kb
    time.sleep(min(poll_s, remaining))


def _logical_cell(spec):
  """Cell whose repeated trials estimate one case/algorithm combination."""
  return spec["case"], spec["variant"]


def _git(args):
  try:
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL, timeout=10, check=True)
    return p.stdout.strip()
  except Exception:
    return None


def _source_hash():
  h = hashlib.sha256()
  files = set()
  for pattern in _SOURCE_GLOBS:
    files.update(p for p in ROOT.glob(pattern) if p.is_file())
  included = []
  for p in sorted(files):
    rel = str(p.relative_to(ROOT))
    h.update(rel.encode()); h.update(b"\0"); h.update(p.read_bytes()); h.update(b"\0")
    included.append(rel)
  return h.hexdigest(), included


def provenance():
  source_sha, source_files = _source_hash()
  status = _git(["status", "--porcelain=v1"]) or ""
  return {
      "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git_commit": _git(["rev-parse", "HEAD"]),
      "git_branch": _git(["branch", "--show-current"]),
      "git_dirty": bool(status),
      "git_status": status.splitlines(),
      "source_sha256": source_sha,
      "source_files": source_files,
      "python": sys.version,
      "python_executable": sys.executable,
      "platform": platform.platform(),
      "machine": platform.machine(),
      "hostname": platform.node(),
      "cpu_count": os.cpu_count(),
  }


def parse_log(text: str):
  """Extract stable metrics while retaining the complete log as the source of truth."""
  out = {}

  def one(key, pattern, cast=float):
    m = re.search(pattern, text, re.MULTILINE)
    if m:
      out[key] = cast(m.group(1))

  m = re.search(r"\[sig-h\]\s+(\d+) coreachable states,\s+(\d+) pick facts, "
                r"built in " + _NUM + r"s", text)
  if m:
    out.update(backward_states=int(m.group(1)), backward_pick_facts=int(m.group(2)),
               backward_build_s=float(m.group(3)))
  # A lazy fill-local PDB prints its cumulative reverse-search time whenever a fill attempt
  # returns.  Widening/retry paths can print the same PDB more than once, so summing log lines
  # would double count it.  Within one milestone object, the largest cumulative value is the
  # best reconstructable total (and is exact for the ordinary no-repair scaling runs).
  fill_reverse = {}
  for fm in re.finditer(r"\[fill-pdb\s+([^:\]\s]+):[^\]]+\].*?\breverse=" +
                        _NUM + r"s", text):
    key, value = fm.group(1), float(fm.group(2))
    fill_reverse[key] = max(value, fill_reverse.get(key, 0.0))
  if fill_reverse:
    out["fill_pdb_reverse_s"] = sum(fill_reverse.values())
  if "backward_build_s" in out or "fill_pdb_reverse_s" in out:
    out["backward_search_s"] = (out.get("backward_build_s", 0.0)
                                + out.get("fill_pdb_reverse_s", 0.0))
  one("milestone_search_s", r"\[milestones\] complete: .*? in " + _NUM + r"s")
  one("milestone_repairs", r"\[milestones\] complete: .*?\((\d+) repairs\)", int)
  one("peak_rss_reported_mb", r"\[rss\] peak RSS after solve:\s*" + _NUM + r" MB")
  one("strict_collision_rejects", r"\[strict-collisions\] vetoed (\d+)", int)
  one("clearance_rejects", r"\[ik-clearance\] vetoed (\d+)", int)

  m = re.search(
      r"\[search-phase-prof\] pipeline=" + _NUM + r"s "
      r"grasp=" + _NUM + r"s/(\d+) "
      r"grasp_backward=" + _NUM + r"s/(\d+) "
      r"force_lp_backward=" + _NUM + r"s/(\d+) "
      r"force_lp_forward=" + _NUM + r"s/(\d+)", text)
  if m:
    out.update(
        search_pipeline_s=float(m.group(1)),
        grasp_generation_s=float(m.group(2)), grasp_generation_calls=int(m.group(3)),
        grasp_backward_s=float(m.group(4)), grasp_backward_calls=int(m.group(5)),
        backward_force_lp_s=float(m.group(6)), backward_force_lp_calls=int(m.group(7)),
        forward_force_lp_s=float(m.group(8)), forward_force_lp_calls=int(m.group(9)))

  m = re.search(
      r"\[ik-prof\] binds: (\d+) called, (\d+) cache hits .*?, (\d+) solved "
      r"\((\d+) ok / (\d+) failed\), bind wall " + _NUM + r"s", text)
  if m:
    keys = ("ik_bind_calls", "ik_bind_cache_hits", "ik_bind_solves", "ik_bind_success",
            "ik_bind_fail")
    out.update({k: int(v) for k, v in zip(keys, m.groups()[:5])})
    out["ik_bind_wall_s"] = float(m.group(6))
  for pm in re.finditer(
      r"^\[ik-prof\]\s+(pick|handover|add_grasp|place|assemble)\s+"
      r"(\d+)/(\d+)\s+" + _NUM +
      r"s;\s+full\s+(\d+)/\s*" + _NUM +
      r"s;\s+precheck\s+(\d+)/\s*" + _NUM + r"s",
      text, re.MULTILINE):
    kind = pm.group(1)
    out.update({
        f"ik_{kind}_binds": int(pm.group(2)),
        f"ik_{kind}_bind_fail": int(pm.group(3)),
        f"ik_{kind}_bind_s": float(pm.group(4)),
        f"ik_{kind}_nlp_solves": int(pm.group(5)),
        f"ik_{kind}_nlp_s": float(pm.group(6)),
        f"ik_{kind}_precheck_solves": int(pm.group(7)),
        f"ik_{kind}_precheck_s": float(pm.group(8)),
    })
  m = re.search(r"\[ik-prof\] bind outcome time: successes " + _NUM +
                r"s .*?, failures " + _NUM + r"s", text)
  if m:
    out.update(ik_bind_success_s=float(m.group(1)), ik_bind_fail_s=float(m.group(2)))
  m = re.search(r"\[ik-prof\] NLP solves: (\d+) in " + _NUM + r"s .*?"
                r"seeded (\d+)/" + _NUM + r"s, unseeded (\d+)/" + _NUM +
                r"s, restarts (\d+)/" + _NUM + r"s", text)
  if m:
    out.update(ik_nlp_solves=int(m.group(1)), ik_nlp_s=float(m.group(2)),
               ik_seeded_solves=int(m.group(3)), ik_seeded_s=float(m.group(4)),
               ik_unseeded_solves=int(m.group(5)), ik_unseeded_s=float(m.group(6)),
               ik_restart_solves=int(m.group(7)), ik_restart_s=float(m.group(8)))
  m = re.search(r"\[ik-prof\] reach precheck: (\d+) relaxed solves in " + _NUM +
                r"s \((\d+) rejects caught\)", text)
  if m:
    out.update(ik_precheck_solves=int(m.group(1)), ik_precheck_s=float(m.group(2)),
               ik_precheck_rejects=int(m.group(3)))
  m = re.search(r"\[ik-reuse\] positive certificates: (\d+) accepted / "
                r"(\d+) tried in " + _NUM + r"s", text)
  if m:
    out.update(ik_reuse_hits=int(m.group(1)), ik_reuse_trials=int(m.group(2)),
               ik_reuse_s=float(m.group(3)))
  m = re.search(r"\[ik-reuse\] negative diagnostic ONLY \(no pruning\): "
                r"(\d+) repeated exact-scene failures / " + _NUM + r"s; "
                r"(\d+) cross-scene same-core repeats / " + _NUM +
                r"s optimistic nogood upper bound", text)
  if m:
    out.update(ik_negative_exact_repeats=int(m.group(1)),
               ik_negative_exact_repeat_s=float(m.group(2)),
               ik_negative_core_repeats=int(m.group(3)),
               ik_negative_core_repeat_s=float(m.group(4)))
  m = re.search(r"\[ik-reuse\] physical-query invariant: (\d+) symbolic aliases "
                r"collapsed across (\d+) (?:candidate-touched )?physical queries", text)
  if m:
    out.update(ik_physical_aliases=int(m.group(1)), ik_physical_queries=int(m.group(2)))
  m = re.search(r"\[ik-reuse\] open pool across \d+ .*?: (\d+) physical queries probed, "
                r"(\d+) revisited; (\d+)/(\d+) probes continued prior effort", text)
  if m:
    out.update(ik_open_physical_queries=int(m.group(1)),
               ik_open_physical_revisited=int(m.group(2)),
               ik_open_continued_probes=int(m.group(3)),
               ik_open_total_probes=int(m.group(4)))

  # Fixed-world pick scaling: distinguish the finite robot/grasp target alphabet from
  # obstacle-scene variants and repeated optimizer effort on an already identical physical
  # query. Prefer the globally deduplicated end-of-run row. A timed-out run has no final
  # RobotIK report, so retain the sum of completed per-fill rows as a useful lower bound.
  m = re.search(
      r"^\[pick-prof\] (\d+) binds/(\d+) failed over (\d+) exact robot-grasp targets "
      r"and (\d+) physical table scenes; exact-query reprobes=(\d+)",
      text, re.MULTILINE)
  if m:
    out.update(pick_profile_binds=int(m.group(1)),
               pick_profile_failures=int(m.group(2)),
               pick_profile_targets=int(m.group(3)),
               pick_profile_table_scenes=int(m.group(4)),
               pick_profile_exact_reprobes=int(m.group(5)),
               pick_profile_complete=True)
  else:
    rows = list(re.finditer(
        r"^\[pick-prof [^\]]+\] binds=(\d+)/(\d+)f targets=(\d+) "
        r"table-scenes=(\d+) exact-query-reprobes=(\d+)", text, re.MULTILINE))
    if rows:
      totals = [sum(int(row.group(i)) for row in rows) for i in range(1, 6)]
      out.update(pick_profile_binds=totals[0], pick_profile_failures=totals[1],
                 pick_profile_targets=totals[2], pick_profile_table_scenes=totals[3],
                 pick_profile_exact_reprobes=totals[4], pick_profile_complete=False)

  out["fill_attempts"] = len(re.findall(r"^\[fill-prof ", text, re.MULTILINE))
  out["milestone_failed_fills"] = len(re.findall(
      r"^\[milestones\] fill-in .* failed", text, re.MULTILINE))
  out["milestone_windows_attempted"] = len(re.findall(
      r"^\[milestones\] (?:added )?WINDOW|^\[milestones\] window .* failed",
      text, re.MULTILINE))
  out["open_effort_exhaustions"] = len(re.findall(r"open-effort|:exhausted", text))
  out["policy_grasp_batches"] = len(re.findall(
      r"^\[fill-policy-widen .*?\] Bellman frontier adds one grasp batch",
      text, re.MULTILINE))
  out["policy_yaw_orbits"] = len(re.findall(
      r"^\[fill-policy-widen .*?\] Bellman frontier adds one rest/yaw orbit",
      text, re.MULTILINE))
  out["policy_failed_probes"] = len(re.findall(
      r"^\[fill-policy-widen .*?\] (?:grasp batch|rest/yaw orbit) did not validate",
      text, re.MULTILINE))
  sk_ok = re.findall(
      r"^\[fill-skeleton .*?\] VALIDATED \d+-action direct schema after (\d+) "
      r"candidate path\(s\), (\d+) failed IK edge\(s\), " + _NUM + r"s",
      text, re.MULTILINE)
  sk_fb = re.findall(
      r"^\[fill-skeleton .*?\] (?:assignment probe budget \d+ reached|family exhausted|"
      r"node/deadline budget).*?(\d+) failed IK edge\(s\), " + _NUM +
      r"s -> complete local fallback", text, re.MULTILINE)
  if sk_ok or sk_fb:
    out["skeleton_successful_fills"] = len(sk_ok)
    out["skeleton_fallback_fills"] = len(sk_fb)
    out["skeleton_candidates"] = sum(int(m[0]) for m in sk_ok)
    out["skeleton_failed_ik_edges"] = (
        sum(int(m[1]) for m in sk_ok) + sum(int(m[0]) for m in sk_fb))
    out["skeleton_probe_s"] = (
        sum(float(m[2]) for m in sk_ok) + sum(float(m[1]) for m in sk_fb))

  # Exclusive paper-table decomposition. Reverse-search log timers include any lazily
  # materialized grasp rows and equilibrium optimizer calls made during predecessor
  # expansion. Remove those nested measurements here, then assign the remaining pipeline
  # residual to forward discrete search/setup. The five reported columns therefore sum to
  # search_pipeline_s up to timer precision. A missing component remains absent rather than
  # being silently treated as measured zero.
  required = ("search_pipeline_s", "grasp_generation_s", "grasp_backward_s",
              "backward_search_s", "backward_force_lp_s", "ik_bind_wall_s")
  if all(k in out for k in required):
    out["backward_graph_search_s"] = max(
        0.0, out["backward_search_s"] - out["grasp_backward_s"]
        - out["backward_force_lp_s"])
    outside_backward_grasp = max(
        0.0, out["grasp_generation_s"] - out["grasp_backward_s"])
    out["forward_search_s"] = max(
        0.0, out["search_pipeline_s"] - out["backward_search_s"]
        - out["ik_bind_wall_s"] - outside_backward_grasp)
  return out


def parse_time_file(path: Path):
  out = {}
  if not path.exists():
    return out
  for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    if "=" not in line:
      continue
    key, value = line.split("=", 1)
    try:
      out[key] = float(value) if key == "wall_s" else int(value)
    except ValueError:
      pass
  return out


def parse_plan(path: Path):
  if not path.exists():
    return {}
  try:
    data = _read_json(path)
    plan = data.get("plan")
    if not isinstance(plan, list):
      return {"plan_parse_error": "missing plan list"}
    kinds = {}
    for action in plan:
      if action:
        kinds[action[0]] = kinds.get(action[0], 0) + 1
    witnesses = (data.get("witnesses") or {}).get("wit") or []
    return {
        "plan_actions": len(plan),
        "plan_cost": sum(2 if a and a[0] == "handover" else 1 for a in plan),
        "plan_action_kinds": kinds,
        "witness_count": sum(w is not None for w in witnesses),
        "witness_stage": (data.get("witnesses") or {}).get("stage"),
    }
  except Exception as exc:
    return {"plan_parse_error": f"{type(exc).__name__}: {exc}"}


def _fingerprint(spec, command, source_sha, grasp_db_key=None):
  payload = json.dumps({"spec": spec, "command": command, "source": source_sha,
                        "grasp_db_key": grasp_db_key},
                       sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(payload).hexdigest()


def run_one(spec, suite_dir: Path, suite_provenance, env_extra, timeout_override=None,
            grasp_database=None):
  run_dir = suite_dir / "runs" / spec["run_id"]
  run_dir.mkdir(parents=True, exist_ok=True)
  plan_path = run_dir / "plan.json"
  log_path = run_dir / "stdout.log"
  time_path = run_dir / "time.txt"
  result_path = run_dir / "result.json"
  # A non-resume invocation is a fresh sample. Never let a plan from an earlier failed or
  # interrupted invocation turn the new run into a false success.
  for stale in (plan_path, log_path, time_path, result_path):
    if stale.exists():
      stale.unlink()
  command = _command_for(spec, plan_path)
  guarded_command = _guarded_command(spec, command)
  grasp_db_key = None if grasp_database is None else grasp_database["cache_key"]
  fingerprint = _fingerprint(spec, command, suite_provenance["source_sha256"], grasp_db_key)
  timeout_s = float(timeout_override if timeout_override is not None else spec["timeout_s"])
  run_meta = {
      "schema_version": SCHEMA_VERSION, "run_fingerprint": fingerprint,
      "run_id": spec["run_id"], "case": spec["case"], "variant": spec["variant"],
      "seed": spec["seed"], "repeat": spec.get("repeat", 1), "tags": spec["tags"],
      "timeout_s": timeout_s,
      "max_address_space_gb": spec.get("max_address_space_gb"),
      "max_rss_gb": spec.get("max_rss_gb"),
      "planner_command": command, "planner_command_shell": shlex.join(command),
      "execution_command": guarded_command,
      "execution_command_shell": shlex.join(guarded_command),
      "source_sha256": suite_provenance["source_sha256"],
  }
  if grasp_database is not None:
    run_meta["grasp_database"] = {
        "path": str(Path(grasp_database["path"]).relative_to(suite_dir)),
        "cache_key": grasp_database["cache_key"],
        "build_s": grasp_database.get("build_s"),
        "shared_preprocessing": True,
    }
    run_meta["grasp_database_build_s"] = grasp_database.get("generation_s")
  if "method" in spec:
    run_meta.update(method=spec["method"], physics=spec["physics"])
  _write_json(run_dir / "run.json", run_meta)

  timed = [str(_GNU_TIME), "-f", "wall_s=%e\nmax_rss_kb=%M\nexit_status=%x",
           "-o", str(time_path), *guarded_command]
  env = os.environ.copy()
  env.update({"PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1",
              "ASSEMBLY_BENCH_PROFILE": "1",
              "NUMBA_CACHE_DIR": "/tmp/assembly-numba-cache",
              "MPLCONFIGDIR": "/tmp/assembly-mpl-cache"})
  env.update(env_extra)
  if grasp_database is not None:
    env.update({"ASSEMBLY_GRASP_DB_PATH": str(grasp_database["path"]),
                "ASSEMBLY_GRASP_DB_KEY": grasp_database["cache_key"]})
  started = dt.datetime.now(dt.timezone.utc)
  t0 = time.perf_counter()
  with log_path.open("w", encoding="utf-8") as log:
    log.write(f"[benchmark-command] {shlex.join(command)}\n")
    log.flush()
    proc = subprocess.Popen(timed, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    returncode, timed_out, rss_limit_hit, sampled_peak_kb = _wait_with_guards(
        proc, timeout_s, spec.get("max_rss_gb"))
  observed_wall = time.perf_counter() - t0
  if rss_limit_hit:
    with log_path.open("a", encoding="utf-8") as log:
      log.write(
          f"[benchmark-resource] process-tree RSS exceeded {spec['max_rss_gb']:g} GiB "
          f"(sampled peak {sampled_peak_kb / 1024:.1f} MiB); process group terminated\n")
  text = log_path.read_text(encoding="utf-8", errors="replace")
  time_metrics = parse_time_file(time_path)
  plan_metrics = parse_plan(plan_path)
  success = returncode == 0 and "plan_actions" in plan_metrics
  # Python allocations normally surface as ``MemoryError``.  Native allocations (notably
  # FCL/RAI containers) can instead terminate the process with an uncaught
  # ``std::bad_alloc`` and SIGABRT.  Both are the same resource-limit outcome for this
  # benchmark and must not be reported as planner errors.
  out_of_memory = (not timed_out and returncode != 0
                   and (rss_limit_hit or _looks_like_out_of_memory(text)))
  status = "success" if success else "timeout" if timed_out else (
      "oom" if out_of_memory else "no_plan" if returncode == 0 else "error")
  result = {
      **run_meta,
      "status": status,
      "success": success,
      "timed_out": timed_out,
      "out_of_memory": out_of_memory,
      "rss_limit_hit": rss_limit_hit,
      "sampled_process_tree_peak_rss_kb": sampled_peak_kb,
      "returncode": returncode,
      "started_utc": started.isoformat(),
      "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
      "observed_wall_s": observed_wall,
      **time_metrics,
      **parse_log(text),
      **plan_metrics,
      "artifacts": {
          "log": str(log_path.relative_to(suite_dir)),
          "plan": str(plan_path.relative_to(suite_dir)) if plan_path.exists() else None,
          "time": str(time_path.relative_to(suite_dir)) if time_path.exists() else None,
      },
  }
  # The solver timing deliberately excludes shared preprocessing. Keep its actual one-time
  # cost in the ordinary grasp column so the paper breakdown does not misleadingly show zero.
  if (grasp_database is not None
      and isinstance(grasp_database.get("generation_s"), (int, float))):
    result["grasp_generation_s"] = float(grasp_database["generation_s"])
  _write_json(result_path, result)
  return result


def _grasp_database_specs(runs, source_sha, suite_dir):
  """One immutable grasp input per physical case/seed, shared across solver variants."""
  groups = {}
  for spec in runs:
    groups.setdefault((spec["case"], spec["seed"]), []).append(spec)
  answer = {}
  for group, specs in groups.items():
    case, seed = group
    # Include every variant's resolved arguments in the fingerprint. This makes the artifact
    # a property of this exact comparison matrix, while deliberately excluding `repeat`.
    payload = {
        "schema": 1, "source": source_sha, "case": case, "seed": seed,
        "resolved_args": sorted({s["variant"]: s["args"] for s in specs}.items()),
    }
    key = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = suite_dir / "grasp-databases" / f"{_slug(case)}__seed-{seed}__{key[:12]}"
    entry = {"cache_key": key, "path": path, "representative": specs[0],
             "case": case, "seed": seed}
    answer[group] = entry
  return answer


def _prepare_grasp_database(entry, suite_dir, source_sha, env_extra):
  """Build the shared DB in a guarded subprocess outside every measured solver run."""
  path = Path(entry["path"])
  meta_path = path / "metadata.json"
  if meta_path.exists() and any(path.glob("*.json.gz")):
    meta = _read_json(meta_path) if meta_path.exists() else {}
    entry["build_s"] = meta.get("build_s", 0.0)
    entry["generation_s"] = meta.get("generation_s", entry["build_s"])
    entry["status"] = "reused"
    return entry
  path.mkdir(parents=True, exist_ok=True)
  spec = entry["representative"]
  log_path = path / "build.log"
  command = _command_for(spec, path / "unused-plan.json")
  guarded = _guarded_command(spec, command)
  env = os.environ.copy()
  env.update({"PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1",
              "ASSEMBLY_BENCH_PROFILE": "1",
              "ASSEMBLY_GRASP_DB_BUILD_ONLY": "1",
              "ASSEMBLY_GRASP_DB_PATH": str(path),
              "ASSEMBLY_GRASP_DB_KEY": entry["cache_key"],
              "NUMBA_CACHE_DIR": "/tmp/assembly-numba-cache",
              "MPLCONFIGDIR": "/tmp/assembly-mpl-cache"})
  env.update(env_extra)
  timeout_s = float(spec["timeout_s"])
  print(f"[grasp-db] build {entry['case']} seed {entry['seed']} -> {path.name}")
  t0 = time.perf_counter()
  with log_path.open("w", encoding="utf-8") as log:
    log.write(f"[grasp-database-command] {shlex.join(command)}\n")
    log.flush()
    proc = subprocess.Popen(guarded, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    returncode, timed_out, rss_limit_hit, sampled_peak_kb = _wait_with_guards(
        proc, timeout_s, spec.get("max_rss_gb"))
  elapsed = time.perf_counter() - t0
  if rss_limit_hit:
    with log_path.open("a", encoding="utf-8") as log:
      log.write(
          f"[benchmark-resource] process-tree RSS exceeded {spec['max_rss_gb']:g} GiB "
          f"(sampled peak {sampled_peak_kb / 1024:.1f} MiB); process group terminated\n")
  log_text = log_path.read_text(encoding="utf-8", errors="replace")
  match = re.search(r"\[grasp-db\] build-only complete: .* in " + _NUM + r"s", log_text)
  entry["build_s"] = elapsed
  entry["generation_s"] = float(match.group(1)) if match else elapsed
  entry["status"] = ("timeout" if timed_out else "oom" if rss_limit_hit else
                     "success" if returncode == 0 else "error")
  entry["rss_limit_hit"] = rss_limit_hit
  entry["sampled_process_tree_peak_rss_kb"] = sampled_peak_kb
  entry["log"] = log_path
  artifacts = tuple(path.glob("*.json.gz"))
  if entry["status"] != "success" or not artifacts:
    tail = log_text.splitlines()[-12:]
    raise RuntimeError(
        f"shared grasp database build failed for {entry['case']} seed {entry['seed']} "
        f"({entry['status']}, returncode={returncode}); see {log_path}\n" + "\n".join(tail))
  _write_json(meta_path, {
      "schema_version": 1, "cache_key": entry["cache_key"],
      "build_s": entry["build_s"], "generation_s": entry["generation_s"],
      "log": str(log_path.relative_to(suite_dir)),
  })
  size = sum(artifact.stat().st_size for artifact in artifacts)
  print(f"[grasp-db] ready in {elapsed:.2f}s ({size / 1024**2:.1f} MiB)")
  return entry


def _numeric_summary(rows, key):
  vals = [float(r[key]) for r in rows
          if isinstance(r.get(key), (int, float)) and math.isfinite(float(r[key]))]
  if not vals:
    return None, None
  return statistics.fmean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0


def aggregate(results):
  metrics = ("wall_s", "observed_wall_s", "max_rss_kb", "plan_actions", "plan_cost",
             "mp_initial_solution_s",
             "search_pipeline_s", "grasp_generation_s", "grasp_generation_calls",
             "grasp_backward_s", "grasp_backward_calls",
             "backward_graph_search_s", "backward_force_lp_s",
             "backward_force_lp_calls", "forward_search_s", "forward_force_lp_s",
             "forward_force_lp_calls",
             "backward_states", "backward_build_s", "fill_pdb_reverse_s",
             "backward_search_s", "milestone_search_s",
             "milestone_repairs", "ik_bind_calls", "ik_bind_fail", "ik_bind_wall_s",
             "ik_bind_success_s", "ik_bind_fail_s",
             "ik_nlp_solves", "ik_nlp_s", "ik_restart_solves",
             "ik_reuse_hits", "ik_reuse_trials", "ik_reuse_s",
             "ik_negative_exact_repeats", "ik_negative_exact_repeat_s",
             "ik_negative_core_repeats", "ik_negative_core_repeat_s",
             "ik_physical_aliases", "ik_physical_queries",
             "ik_open_physical_queries", "ik_open_physical_revisited",
             "ik_open_continued_probes", "ik_open_total_probes",
             "policy_grasp_batches", "policy_yaw_orbits", "policy_failed_probes",
             "skeleton_successful_fills", "skeleton_fallback_fills",
             "skeleton_candidates", "skeleton_failed_ik_edges", "skeleton_probe_s")
  primitive_metrics = tuple(
      f"ik_{kind}_{suffix}"
      for kind in ("pick", "handover", "add_grasp", "place", "assemble")
      for suffix in ("binds", "bind_fail", "bind_s", "nlp_solves", "nlp_s",
                     "precheck_solves", "precheck_s"))
  metrics += primitive_metrics
  groups = []
  keys = sorted({(r["case"], r["variant"]) for r in results})
  for case, variant in keys:
    rows = [r for r in results if r["case"] == case and r["variant"] == variant]
    row = {"case": case, "variant": variant, "runs": len(rows),
           "successes": sum(bool(r.get("success")) for r in rows),
           "timeouts": sum(bool(r.get("timed_out")) for r in rows),
           "out_of_memory": sum(
               bool(r.get("out_of_memory")) or r.get("status") == "oom" for r in rows),
           "errors": sum(r.get("status") == "error" for r in rows),
           "no_plans": sum(r.get("status") == "no_plan" for r in rows)}
    row["success_rate"] = row["successes"] / max(1, row["runs"])
    for metric in metrics:
      mean, sd = _numeric_summary(rows, metric)
      row[f"{metric}_mean"] = mean
      row[f"{metric}_sd"] = sd
    groups.append(row)
  return groups


def aggregate_matrix(results):
  """Attach explicit paper-matrix coordinates to the ordinary aggregate rows."""
  coords = {(r["case"], r["variant"]): (r.get("method"), r.get("physics"))
            for r in results if r.get("method") is not None}
  rows = []
  for group in aggregate(results):
    coord = coords.get((group["case"], group["variant"]))
    if coord is None:
      continue
    rows.append({**group, "method": coord[0], "physics": coord[1]})
  return rows


def _matrix_cell(row):
  if row is None:
    return "--"
  runs = int(row["runs"])
  successes = int(row["successes"])
  bits = [f"{successes}/{runs}"]
  wall = row.get("wall_s_mean")
  if wall is not None:
    wall_sd = row.get("wall_s_sd") or 0.0
    bits.append(f"{wall:.1f}+/-{wall_sd:.1f}s")
  rss = row.get("max_rss_kb_mean")
  if rss is not None:
    bits.append(f"{rss / 1024.0:.0f}MB")
  cost = row.get("plan_cost_mean")
  if cost is not None:
    bits.append(f"cost {cost:.1f}")
  binds, fails = row.get("ik_bind_calls_mean"), row.get("ik_bind_fail_mean")
  if binds is not None and fails is not None:
    bits.append(f"IK {binds:.0f}/{fails:.0f}")
  return " · ".join(bits)


def matrix_markdown(manifest, groups):
  """Return compact case-specific method x physics tables suitable for a paper notebook."""
  matrix = manifest["matrix"]
  methods = [r["name"] for r in matrix["rows"] if r.get("enabled", True)]
  physics = [c["name"] for c in matrix["columns"] if c.get("enabled", True)]
  lookup = {(g["case"], g["method"], g["physics"]): g for g in groups}
  lines = [f"# {manifest['name']}", "", manifest.get("description", ""), "",
           "Each cell is `successful/runs · wall mean+/-population sd · peak RSS · "
           "successful-plan cost · IK calls/failures`. Failed and timed-out runs remain in "
           "the success ratio and wall-time mean; cost is computed only from saved plans.", ""]
  for case in (c for c in manifest["cases"] if c.get("enabled", True)):
    lines.extend([f"## {case['name']}", "",
                  "| method | " + " | ".join(physics) + " |",
                  "|---|" + "---:|" * len(physics)])
    for method in methods:
      cells = [_matrix_cell(lookup.get((case["name"], method, p))) for p in physics]
      lines.append("| " + method.replace("|", "\\|") + " | " + " | ".join(cells) + " |")
    lines.append("")
  return "\n".join(lines)


def _write_csv(path: Path, rows):
  if not rows:
    return
  keys = sorted({k for row in rows for k in row if not isinstance(row[k], (dict, list))})
  with path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for row in rows:
      w.writerow({k: row.get(k) for k in keys})


def write_suite_outputs(suite_dir: Path, manifest, suite_provenance, results):
  groups = aggregate(results)
  summary = {
      "schema_version": SCHEMA_VERSION,
      "suite": manifest["name"],
      "description": manifest.get("description"),
      "provenance": suite_provenance,
      "total_runs": len(results),
      "successes": sum(bool(r.get("success")) for r in results),
      "timeouts": sum(bool(r.get("timed_out")) for r in results),
      "groups": groups,
  }
  _write_json(suite_dir / "results.json", results)
  _write_json(suite_dir / "summary.json", summary)
  _write_csv(suite_dir / "runs.csv", results)
  _write_csv(suite_dir / "summary.csv", groups)
  if manifest.get("matrix") is not None:
    matrix_groups = aggregate_matrix(results)
    _write_json(suite_dir / "matrix.json", {
        "schema_version": SCHEMA_VERSION,
        "suite": manifest["name"],
        "rows": manifest["matrix"]["rows"],
        "columns": manifest["matrix"]["columns"],
        "groups": matrix_groups,
    })
    _write_csv(suite_dir / "matrix.csv", matrix_groups)
    (suite_dir / "matrix.md").write_text(
        matrix_markdown(manifest, matrix_groups) + "\n", encoding="utf-8")


def build_parser():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("manifest", type=Path, help="JSON benchmark manifest")
  p.add_argument("--output", type=Path, default=None,
                 help="suite directory (default: benchmarks/results/NAME-TIMESTAMP)")
  p.add_argument("--resume", action="store_true",
                 help="reuse matching completed per-run result files")
  p.add_argument("--repeats", type=int, default=1, metavar="N",
                 help="independent repetitions per logical cell (default: 1; same planner seed)")
  p.add_argument(
      "--seeds", type=int, nargs="+", default=None, metavar="SEED",
      help=("replace every manifest/case/variant seed list with these independent planner "
            "seeds; unlike --repeats, this changes the randomized search instance"))
  p.add_argument("--only", default=None, metavar="REGEX",
                 help="run only expanded ids matching REGEX")
  p.add_argument("--max-runs", type=int, default=None,
                 help="truncate the selected run list (useful for smoke tests)")
  p.add_argument(
      "--jobs", type=int, default=1, metavar="N",
      help=("run up to N different cases concurrently; runs within one case remain "
            "sequential (default: 1)"))
  p.add_argument("--timeout", type=float, default=None, metavar="SEC",
                 help="override every manifest timeout")
  p.add_argument(
      "--stop-cell-on-timeout", action="store_true",
      help="after one trial times out, skip unrun seeds/repeats for that case/variant cell")
  p.add_argument(
      "--shared-grasp-db", action="store_true",
      help=("precompute one fingerprinted collision-filtered grasp database per case/seed "
            "outside timed search, then load that identical artifact in every variant"))
  p.add_argument("--append-arg", action="append", default=[], metavar="TOKEN",
                 help="append one planner argument token to every expanded run; repeat for "
                      "flags with values (use --append-arg=--flag for option-like tokens)")
  p.add_argument("--dry-run", action="store_true", help="print commands without executing")
  return p


def main(argv=None):
  args = build_parser().parse_args(argv)
  if not _GNU_TIME.is_file():
    raise SystemExit("/usr/bin/time is required for per-process peak-RSS measurements")
  manifest_path = args.manifest.resolve()
  manifest = validate_manifest(_read_json(manifest_path))
  if args.repeats < 1:
    raise SystemExit("--repeats must be a positive integer")
  if args.jobs < 1:
    raise SystemExit("--jobs must be a positive integer")
  runs = expand_runs(manifest, repeats=args.repeats, seeds=args.seeds)
  if args.append_arg:
    _check_args(args.append_arg, "--append-arg")
    runs = [{**run, "args": list(run["args"]) + list(args.append_arg)} for run in runs]
  if args.only:
    rx = re.compile(args.only)
    runs = [r for r in runs if rx.search(r["run_id"])]
  if args.max_runs is not None:
    runs = runs[:max(0, args.max_runs)]
  # Make a CLI timeout override part of the resolved run specification.  The fingerprint then
  # prevents --resume from reusing a short diagnostic timeout as though it were a full paper run.
  if args.timeout is not None:
    runs = [{**run, "timeout_s": float(args.timeout)} for run in runs]
  stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  suite_dir = (args.output or ROOT / "benchmarks" / "results"
               / f"{_slug(manifest['name'])}-{stamp}").resolve()
  # Capture git status before creating a default results directory inside the worktree; the
  # benchmark must not report itself as the source of a dirty tree.
  prov = provenance()
  prov["appended_args"] = list(args.append_arg)
  prov["seed_override"] = list(args.seeds) if args.seeds is not None else None
  prov["shared_grasp_database"] = bool(args.shared_grasp_db)
  prov["jobs"] = int(args.jobs)
  prov["manifest_path"] = str(manifest_path)
  prov["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
  suite_dir.mkdir(parents=True, exist_ok=True)
  _write_json(suite_dir / "manifest.json", manifest)
  _write_json(suite_dir / "provenance.json", prov)

  if not runs:
    raise SystemExit("no runs selected")
  mode = "sequential" if args.jobs == 1 else f"up to {args.jobs} cases in parallel"
  print(f"[bench] {manifest['name']}: {len(runs)} run(s), {mode} -> {suite_dir}")
  results_by_id = {}
  grasp_databases = (_grasp_database_specs(runs, prov["source_sha256"], suite_dir)
                     if args.shared_grasp_db else {})
  run_order = {spec["run_id"]: i for i, spec in enumerate(runs)}
  output_lock = threading.Lock()

  def current_results():
    return sorted(results_by_id.values(), key=lambda row: run_order[row["run_id"]])

  def checkpoint(result):
    with output_lock:
      results_by_id[result["run_id"]] = result
      write_suite_outputs(suite_dir, manifest, prov, current_results())

  def write_grasp_database_index():
    with output_lock:
      _write_json(suite_dir / "grasp-databases.json", {
          "schema_version": 1,
          "databases": [{k: (str(v.relative_to(suite_dir)) if isinstance(v, Path) else v)
                          for k, v in db.items() if k != "representative"}
                         for db in grasp_databases.values()],
      })

  def run_case(case_runs):
    # A case is the unit of concurrency. Keeping its variants/seeds ordered preserves both
    # stop-on-timeout semantics and deterministic reuse of its immutable grasp databases.
    timed_out_cells = set()
    for i, spec in case_runs:
      run_dir = suite_dir / "runs" / spec["run_id"]
      result_path = run_dir / "result.json"
      command = _command_for(spec, run_dir / "plan.json")
      grasp_database = grasp_databases.get((spec["case"], spec["seed"]))
      grasp_db_key = None if grasp_database is None else grasp_database["cache_key"]
      fingerprint = _fingerprint(spec, command, prov["source_sha256"], grasp_db_key)
      if args.resume and result_path.exists():
        old = _read_json(result_path)
        if old.get("run_fingerprint") == fingerprint:
          print(f"[bench {i}/{len(runs)}] reuse {spec['run_id']}: {old.get('status')}")
          checkpoint(old)
          if old.get("timed_out"):
            timed_out_cells.add(_logical_cell(spec))
          continue
      if args.stop_cell_on_timeout and _logical_cell(spec) in timed_out_cells:
        print(f"[bench {i}/{len(runs)}] skip {spec['run_id']}: "
              "an earlier trial in this case/variant cell timed out")
        continue
      print(f"[bench {i}/{len(runs)}] {spec['run_id']}")
      if spec.get("max_rss_gb") is not None:
        print(f"  [rss-guard] process tree limited to {spec['max_rss_gb']:g} GiB resident")
      if args.dry_run:
        if grasp_database is not None:
          print(f"  [grasp-db] {grasp_database['path']}")
        print("  " + shlex.join(_guarded_command(spec, command)))
        continue
      if grasp_database is not None and "status" not in grasp_database:
        _prepare_grasp_database(grasp_database, suite_dir, prov["source_sha256"],
                                manifest.get("env", {}))
        write_grasp_database_index()
      result = run_one(spec, suite_dir, prov, manifest.get("env", {}),
                       grasp_database=grasp_database)
      if result.get("timed_out"):
        timed_out_cells.add(_logical_cell(spec))
      wall = result.get("wall_s", result.get("observed_wall_s"))
      print(f"  -> {result['status']} in {wall:.2f}s"
            + (f", cost {result['plan_cost']}" if result.get("success") else ""))
      checkpoint(result)

  grouped = {}
  for i, spec in enumerate(runs, 1):
    grouped.setdefault(spec["case"], []).append((i, spec))
  case_groups = list(grouped.values())
  if args.jobs == 1 or args.dry_run:
    for group in case_groups:
      run_case(group)
  else:
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(case_groups)),
                            thread_name_prefix="benchmark-case") as pool:
      futures = [pool.submit(run_case, group) for group in case_groups]
      for future in as_completed(futures):
        future.result()
  if args.dry_run:
    return 0
  results = current_results()
  write_suite_outputs(suite_dir, manifest, prov, results)
  print(f"[bench] summary -> {suite_dir / 'summary.json'}")
  if manifest.get("matrix") is not None:
    print(f"[bench] paper matrix -> {suite_dir / 'matrix.md'}")
  return 0 if all(r.get("success") for r in results) else 1


if __name__ == "__main__":
  raise SystemExit(main())
