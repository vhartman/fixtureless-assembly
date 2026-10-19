#!/usr/bin/env python3
"""Sequential Monte Carlo runner for perturbed MuJoCo assembly execution.

Every trial is a fresh, memory-limited process. The simulator owns perturbation sampling and
physical outcome classification; this runner only expands a reproducible matrix, checkpoints
JSON/CSV results, and controls artifact retention.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def _slug(x):
  return re.sub(r"[^A-Za-z0-9._-]+", "-", str(x)).strip("-.") or "run"


def _write_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with tmp.open("w", encoding="utf-8") as f:
    json.dump(value, f, indent=2, sort_keys=True)
    f.write("\n")
  os.replace(tmp, path)


def _write_csv(path, rows):
  fields = ["trial", "seed", "xy_m", "yaw_deg", "observation", "process_noise",
            "status", "success", "returncode", "wall_s", "first_failure",
            "detail", "num_actions_ok", "num_actions", "recording", "log"]
  tmp = Path(str(path) + ".tmp")
  with tmp.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in rows:
      w.writerow({k: row.get(k) for k in fields})
  os.replace(tmp, path)


def build_command(args, trial, report, recording):
  cmd = [sys.executable, str(ROOT / "sim.py"), "run", str(args.plan.resolve()),
         "--mp-path", str(args.mp_path.resolve()),
         "--object-perturb-xy", str(trial["xy_m"]),
         "--object-perturb-yaw", str(trial["yaw_deg"]),
         "--object-perturb-seed", str(trial["seed"]),
         "--object-perturb-observation", trial["observation"],
         "--object-perturb-max-tries", str(args.max_placement_tries),
         "--object-grasp-capture", str(args.grasp_capture),
         "--object-grasp-capture-angle", str(args.grasp_capture_angle),
         "--process-noise-seed", str(trial["seed"]),
         "--report-json", str(report), "--record", str(recording)]
  if args.apply_process_wrench:
    cmd += ["--apply-process-wrench", "--process-noise", args.process_noise]
    if args.process_noise == "corner":
      cmd += ["--process-noise-corner", str(args.process_noise_corner)]
  if args.extra:
    extra = list(args.extra)
    if extra and extra[0] == "--":
      extra = extra[1:]
    cmd += extra
  if args.memory_gb:
    limiter = shutil.which("prlimit")
    if limiter is None:
      raise RuntimeError("memory guard requested but GNU prlimit is unavailable; refusing to "
                         "run trials without the advertised guard")
    cmd = [limiter, f"--as={int(args.memory_gb * 1024**3)}", "--"] + cmd
  return cmd


def _expand(args):
  observations = ["observed", "unobserved"] if args.observation == "both" \
      else [args.observation]
  out = []
  for xy in args.xy:
    for yaw in args.yaw_deg:
      for obs in observations:
        for seed in args.seeds:
          out.append({"seed": seed, "xy_m": xy, "yaw_deg": yaw,
                      "observation": obs, "process_noise": (
                          args.process_noise if args.apply_process_wrench else "disabled")})
  return out


def _parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--plan", type=Path, required=True)
  p.add_argument("--mp-path", type=Path, required=True)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
  p.add_argument("--xy", type=float, nargs="+", default=[0.005], metavar="M")
  p.add_argument("--yaw-deg", type=float, nargs="+", default=[3.0], metavar="DEG")
  p.add_argument("--observation", choices=("observed", "unobserved", "both"),
                 default="observed")
  p.add_argument("--apply-process-wrench", action="store_true")
  p.add_argument("--process-noise", choices=("nominal", "sample", "corner"),
                 default="sample")
  p.add_argument("--process-noise-corner", type=int, default=0, metavar="N")
  p.add_argument("--timeout", type=float, default=900.0, metavar="SEC")
  p.add_argument("--memory-gb", type=float, default=4.0, metavar="GB")
  p.add_argument("--max-placement-tries", type=int, default=200)
  p.add_argument("--grasp-capture", type=float, default=0.010, metavar="M")
  p.add_argument("--grasp-capture-angle", type=float, default=10.0, metavar="DEG")
  p.add_argument("--keep-recordings", choices=("failures", "all", "none"),
                 default="failures",
                 help="Default keeps every failure and one representative success per cell.")
  p.add_argument("--no-representative-success", action="store_true")
  p.add_argument("--dry-run", action="store_true")
  p.add_argument("extra", nargs=argparse.REMAINDER,
                 help="Extra sim.py flags after a literal -- (use only for controlled ablations).")
  a = p.parse_args(argv)
  if any(x < 0.0 for x in a.xy) or any(x < 0.0 for x in a.yaw_deg):
    p.error("placement bounds must be non-negative")
  if a.timeout <= 0.0 or a.memory_gb <= 0.0:
    p.error("timeout and memory-gb must be positive")
  if not a.plan.is_file() or not a.mp_path.is_file():
    p.error("--plan and --mp-path must exist")
  controlled = {"--mp-path", "--record", "--report-json",
                "--object-perturb-xy", "--object-perturb-yaw",
                "--object-perturb-seed", "--object-perturb-observation",
                "--object-perturb-max-tries", "--object-grasp-capture",
                "--object-grasp-capture-angle", "--apply-process-wrench",
                "--no-apply-process-wrench", "--process-noise",
                "--process-noise-seed", "--process-noise-corner"}
  for token in a.extra:
    if token.split("=", 1)[0] in controlled:
      p.error(f"{token.split('=', 1)[0]} is runner-controlled, not an extra sim argument")
  return a


def main(argv=None):
  args = _parse_args(argv)
  out = args.output.resolve()
  reports, recordings, logs = out / "reports", out / "recordings", out / "logs"
  for d in (reports, recordings, logs):
    d.mkdir(parents=True, exist_ok=True)
  trials = _expand(args)
  rows, kept_success = [], set()
  manifest = {
      "schema_version": SCHEMA_VERSION, "plan": str(args.plan.resolve()),
      "motion_path": str(args.mp_path.resolve()), "simulation_contract": "standard",
      "timeout_s": args.timeout, "memory_gb": args.memory_gb,
      "apply_process_wrench": args.apply_process_wrench,
      "process_noise": args.process_noise, "process_noise_corner": args.process_noise_corner,
      "max_placement_tries": args.max_placement_tries,
      "grasp_capture_m": args.grasp_capture,
      "grasp_capture_angle_deg": args.grasp_capture_angle,
      "keep_recordings": args.keep_recordings,
      "representative_success": not args.no_representative_success,
      "extra_sim_args": list(args.extra), "trials": trials,
  }
  _write_json(out / "manifest.json", manifest)
  print(f"[noisy-sim] {len(trials)} sequential fresh-process trial(s); "
        f"guard={args.memory_gb:g}GiB/{args.timeout:g}s")

  env = os.environ.copy()
  env.update({"PYTHONHASHSEED": "0", "NUMBA_CACHE_DIR": "/tmp/assembly-numba-cache",
              "MPLCONFIGDIR": "/tmp/assembly-mpl-cache", "PYTHONUNBUFFERED": "1"})
  for i, trial in enumerate(trials):
    tag = _slug(f"{i:04d}-xy{trial['xy_m']:g}-yaw{trial['yaw_deg']:g}-"
                f"{trial['observation']}-seed{trial['seed']}")
    report = reports / f"{tag}.json"
    recording = recordings / f"{tag}.npz"
    log = logs / f"{tag}.log"
    cmd = build_command(args, trial, report, recording)
    print(f"[noisy-sim] {i + 1}/{len(trials)} {tag}")
    if args.dry_run:
      print("  " + " ".join(repr(x) for x in cmd))
      continue
    t0 = time.monotonic()
    status, returncode, output = "process_error", None, ""
    try:
      proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
      returncode, output = proc.returncode, proc.stdout
      status = "completed" if report.is_file() else "preflight_or_process_failure"
    except subprocess.TimeoutExpired as e:
      status = "timeout"
      output = e.stdout or ""
      if isinstance(output, bytes):
        output = output.decode(errors="replace")
      output += "\n[runner] timeout\n"
    except OSError as e:
      status = "process_error"
      output = f"[runner] could not start trial: {e}\n"
    wall = time.monotonic() - t0
    log.write_text(output, encoding="utf-8")
    sim = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
    success = bool(sim.get("success", False))
    cell = (trial["xy_m"], trial["yaw_deg"], trial["observation"], trial["process_noise"])
    keep = (args.keep_recordings == "all" or
            (args.keep_recordings == "failures" and
             (not success or (not args.no_representative_success and
                              success and cell not in kept_success))))
    if success and keep:
      kept_success.add(cell)
    if recording.exists() and not keep:
      recording.unlink()
    row = dict(trial)
    _lines = [x.strip() for x in output.splitlines() if x.strip()]
    detail = "" if sim else (_lines[-1] if _lines else status)
    row.update({"trial": i, "status": status, "success": success,
                "returncode": returncode, "wall_s": round(wall, 6),
                "first_failure": sim.get("first_failure", status), "detail": detail,
                "num_actions_ok": sim.get("num_actions_ok"),
                "num_actions": sim.get("num_actions"),
                "recording": str(recording.relative_to(out)) if recording.exists() else "",
                "report": str(report.relative_to(out)) if report.exists() else "",
                "log": str(log.relative_to(out))})
    rows.append(row)
    _write_json(out / "summary.json", {"schema_version": SCHEMA_VERSION, "runs": rows})
    _write_csv(out / "summary.csv", rows)
    print(f"  {status}: {'PASS' if success else 'FAIL'} in {wall:.1f}s"
          + (f" ({sim.get('first_failure')})" if sim.get("first_failure") else ""))
  return 0 if all(r["success"] for r in rows) else (0 if args.dry_run else 1)


if __name__ == "__main__":
  raise SystemExit(main())
