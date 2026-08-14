"""Process-isolated matrix driver for ``run_live_persistent_v2.py`` (round 3).

Same fixed-environment protocol as ``run_live_persistent_matrix.py``: the environment is
constructed at its fixed benchmark seed, then NumPy/Python are reseeded with
``environment_seed + run_id`` for RRT and shortcutting.  ``PYTHONHASHSEED`` is fixed at the
environment seed.  Every child process caps its own address space at 6 GB.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_live_persistent_v2.py"
ENVS = {
    # (registry name, fixed construction seed from mr_bench/scripts/run_all_envs.py,
    #  short standalone planning time). Planner/shortcut seed is env_seed + run_id.
    "triple": ("rai.triple_waypoints", 2, 2.0),
    "box": ("rai.box_stacking", 2, 2.0),
    "welding": ("rai.welding", 2, 2.0),
    "handover": ("rai.2d_handover", 2, 5.0),
    "hallway": ("rai.other_hallway", 2, 2.0),
    "rearrangement": ("rai.box_rearrangement", 2, 2.0),
    "box_two_arm": ("rai.box_stacking_two_robots", 1, 2.0),
    "eggcartons": ("rai.eggcartons", 2, 2.0),
    "bottles": ("rai.bottles", 2, 2.0),
    "pyramid": ("rai.pyramid", 2, 2.0),
    "mobile_wall_four": ("rai.mobile_wall_four", 2, 2.0),
    "spiral_tower_two": ("rai.spiral_tower_two", 0, 2.0),
}
POLICIES = (
    "native", "persistent_tree", "tree_v2", "tree_adaptive", "tree_partial",
    "tree_warm", "tree_credit", "tree_simple", "tree_split2", "tree_rand",
    "tree_cap", "tree_warmbeta", "tree_mix", "tree_only", "length_uniform",
    "length_tilted", "max_compatible", "max_bandit", "lazy_certify",
)
PROTOCOL = "shortcut-portfolio-v5-exact-gate"


def valid_result(path, *, env_name, env_seed, policy, seed, run_id, phase, runtime, checks):
    try:
        data = json.load(path.open())
        cert = data.get("certificate", {})
        required = (
            "strict_full_edge", "valid_plan", "terminal_mode", "mode_runs_preserved",
            "endpoints_preserved", "trace_matches_output",
        )
        calls = data.get("patch", {}).get("shortcut_calls", [])
        return (
            bool(data.get("planner", {}).get("returned_path"))
            and data.get("protocol") == PROTOCOL
            and data.get("env") == env_name
            and data.get("policy") == policy
            and data.get("seed") == seed
            and data.get("env_seed") == env_seed
            and data.get("run_id") == run_id
            and bool(data.get("environment_fingerprint"))
            and bool(data.get("planner", {}).get("optimize")) == (phase == "full")
            and abs(float(data.get("planner", {}).get("runtime_seconds", -1)) - runtime) < 1e-9
            and all(bool(cert.get(key)) for key in required)
            and bool(calls)
            and int(calls[-1]["acceptance"]["max_iter"]) == checks
        )
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("standalone", "full"), required=True)
    parser.add_argument(
        "--runs", default="0,1,2,3,4,5,6,7,8,9",
        help="run ids; planner/shortcut seed is the environment seed plus run id",
    )
    parser.add_argument("--envs", default=",".join(ENVS))
    parser.add_argument("--checks", type=int, default=1000)
    parser.add_argument("--full-runtime", type=float, default=20.0)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_ids = [int(x) for x in args.runs.split(",") if x]
    env_keys = [x for x in args.envs.split(",") if x]
    policies = tuple(x for x in args.policies.split(",") if x)
    unknown = sorted(set(env_keys) - set(ENVS))
    if unknown:
        parser.error(f"unknown environments: {unknown}")
    unknown_policies = sorted(set(policies) - set(POLICIES))
    if unknown_policies:
        parser.error(f"unknown policies: {unknown_policies}")
    if not policies:
        parser.error("at least one policy is required")

    records = []
    for run_id in run_ids:
        shift = run_id % len(policies)
        policy_order = policies[shift:] + policies[:shift]
        for key in env_keys:
            env_name, env_seed, standalone_runtime = ENVS[key]
            seed = env_seed + run_id
            runtime = args.full_runtime if args.phase == "full" else standalone_runtime
            for policy in policy_order:
                output = HERE / (
                    f"live-persistent-v5-{args.phase}-{key}-{policy}"
                    f"-e{env_seed}-r{run_id}-s{seed}.json"
                )
                valid_existing = output.exists() and valid_result(
                    output, env_name=env_name, env_seed=env_seed, policy=policy,
                    seed=seed, run_id=run_id,
                    phase=args.phase, runtime=runtime, checks=args.checks,
                )
                if valid_existing and not args.force:
                    print("skip", output.name, flush=True)
                    records.append({
                        "env": key, "env_seed": env_seed, "policy": policy,
                        "seed": seed, "run_id": run_id, "runtime": runtime,
                        "output": str(output), "skipped": True, "returncode": 0,
                    })
                    continue
                if output.exists():
                    output.unlink()
                command = [
                    sys.executable, str(RUNNER), env_name,
                    "--policy", policy, "--seed", str(seed),
                    "--env-seed", str(env_seed), "--run-id", str(run_id),
                    "--runtime", str(runtime), "--checks", str(args.checks),
                    "--quiet", "--output", str(output),
                ]
                if args.phase == "full":
                    command.append("--optimize")
                started = time.perf_counter()
                completed = None
                for attempt in range(2):
                    completed = subprocess.run(
                        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        env={**os.environ, "NUMBA_CACHE_DIR": "/tmp/numba-cache",
                             "MPLCONFIGDIR": "/tmp/mplconfig",
                             "PYTHONHASHSEED": str(env_seed)},
                    )
                    if completed.returncode == 0:
                        break
                    print(
                        f"  retry {attempt + 1} after returncode {completed.returncode}",
                        flush=True,
                    )
                duration = time.perf_counter() - started
                record = {
                    "env": key, "env_seed": env_seed, "policy": policy,
                    "seed": seed, "run_id": run_id, "runtime": runtime,
                    "output": str(output), "returncode": completed.returncode,
                    "driver_wall_seconds": duration,
                    "tail": completed.stdout[-4000:],
                }
                records.append(record)
                status = "ok" if completed.returncode == 0 else "FAIL"
                print(
                    f"[{args.phase} {key} {policy} e{env_seed} r{run_id} s{seed}] "
                    f"{status} {duration:.1f}s", flush=True,
                )

    fingerprints = {}
    for record in records:
        if record["returncode"] != 0:
            continue
        with open(record["output"]) as stream:
            fingerprint = json.load(stream).get("environment_fingerprint")
        fingerprints.setdefault(record["env"], set()).add(fingerprint)
    mismatched_envs = {
        key: sorted(values) for key, values in fingerprints.items() if len(values) != 1
    }

    subset_suffix = "" if policies == POLICIES else "-" + "-".join(policies)
    summary = HERE / f"live-persistent-v5-{args.phase}{subset_suffix}-matrix.json"
    with summary.open("w") as stream:
        json.dump({"phase": args.phase, "run_ids": run_ids, "envs": env_keys,
                   "policies": list(policies), "checks": args.checks,
                   "environment_fingerprints": {
                       key: next(iter(values)) for key, values in fingerprints.items()
                       if len(values) == 1
                   },
                   "runs": records}, stream, indent=2)
    print("wrote", summary)
    failed = [record for record in records if record["returncode"] != 0]
    if failed:
        print(json.dumps(failed, indent=2))
        raise SystemExit(1)
    if mismatched_envs:
        print("environment fingerprints changed across runs:")
        print(json.dumps(mismatched_envs, indent=2))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
