"""Full-planner-loop span vs tree vs whole_random on the real assembly scenes.

Same protocol as run_upstream_choice.py (landed planner loop, every internal shortcut call
and the final pass forced to one run_choice, strict full-edge certificate) but the
environment is an assembly fixture rebuilt through benchenv.  One scene per process.

    python run_upstream_choice_assembly_full.py <scene> --choice span|tree|whole_random \
        --seed S --run-id R --runtime 60 --checks 1000 --output out.json

New file; the standalone assembly driver and every earlier experiment file are untouched.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import benchenv as B
from multi_robot_multi_goal_planning.planners import shortcutting as shortcutting_module
from multi_robot_multi_goal_planning.planners.planner_birrtstar import BidirectionalRRTstar
from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)
from run_live_persistent_v2 import _environment_fingerprint
from run_upstream_choice import ChoicePatch, _certificate


PROTOCOL = "shortcut-portfolio-v5-exact-gate"
CHOICES = ("span", "tree", "whole_random")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--choice", choices=CHOICES, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--runtime", type=float, default=60.0)
    parser.add_argument("--checks", type=int, default=1000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        resource.setrlimit(resource.RLIMIT_AS, (12 * 1024 ** 3, 12 * 1024 ** 3))
    except (ValueError, OSError):
        pass

    env, raw, meta = B.load_scene(args.scene)
    env_fingerprint = _environment_fingerprint(env)
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"
    config = BaseRRTConfig()
    config.distance_metric = "max_euclidean"
    config.stepsize = 1.0
    config.shortcutting = True
    config.with_mode_validation = False

    np.random.seed(args.seed)
    random.seed(args.seed)
    run_start = time.perf_counter()
    with ChoicePatch(args.choice) as patch:
        sink = open(os.devnull, "w") if args.quiet else None
        try:
            redirect = contextlib.redirect_stdout(sink) if sink else contextlib.nullcontext()
            with redirect:
                planner = BidirectionalRRTstar(env, config=config)
                planned_path, info = planner.plan(
                    ptc=RuntimeTerminationCondition(float(args.runtime)),
                    optimize=True,
                )
                if planned_path:
                    out, _ = shortcutting_module.robot_mode_shortcut(
                        env, planned_path, int(args.checks),
                        tolerance=float(env.collision_tolerance),
                        resolution=float(env.collision_resolution),
                    )
                else:
                    out = None
        finally:
            if sink:
                sink.close()
    total_wall = time.perf_counter() - run_start

    if out is None:
        result = {
            "protocol": PROTOCOL,
            "env": f"rai.{args.scene}", "policy": args.choice, "seed": args.seed,
            "env_seed": int(meta.get("mp_seed", 1)), "run_id": int(args.run_id),
            "environment_fingerprint": env_fingerprint,
            "planner": {"returned_path": False, "runtime_seconds": args.runtime,
                        "total_wall_seconds": total_wall},
            "patch": {"calls": patch.calls},
        }
        with open(args.output, "w") as stream:
            json.dump(result, stream, indent=2, default=float)
        raise RuntimeError(f"planner returned no path; wrote {args.output}")

    if not patch.calls:
        raise RuntimeError("planner produced a path but no shortcut call was observed")
    certificate = _certificate(env, out, patch.calls[-1])
    required = (
        "strict_full_edge", "valid_plan", "terminal_mode", "mode_runs_preserved",
        "endpoints_preserved", "trace_matches_output",
    )
    if not all(certificate[key] for key in required):
        raise RuntimeError(f"strict output invariant failed: {certificate}")

    info = dict(info or {})
    info_costs = [float(x) for x in info.get("costs", [])]
    info_times = [float(x) for x in info.get("times", [])]
    available = bool(info_times) and len(info_times) == len(info_costs)
    stitched_time = list(info_times) if available else []
    stitched_cost = list(info_costs) if available else []
    final_call = patch.calls[-1]
    final_time = final_call["curve"]["time"]
    if final_time:
        handoff = total_wall - final_call["wall_seconds"]
        if not stitched_time or handoff > stitched_time[-1]:
            stitched_time.append(handoff)
            stitched_cost.append(float(final_call["input"]["cost"]))
        stitched_time.extend(float(handoff + t) for t in final_time[1:])
        stitched_cost.extend(float(c) for c in final_call["curve"]["cost"][1:])

    result = {
        "protocol": PROTOCOL,
        "purpose": "landed mr_bench full loop with run_choice forced, on a real assembly "
                   "fixture (benchenv rebuild)",
        "env": f"rai.{args.scene}", "policy": args.choice, "seed": args.seed,
        "env_seed": int(meta.get("mp_seed", 1)), "run_id": int(args.run_id),
        "environment_fingerprint": env_fingerprint,
        "planner": {
            "runtime_seconds": float(args.runtime), "optimize": True,
            "returned_path": True, "total_wall_seconds": float(total_wall),
            "info_costs": info_costs, "info_times": info_times,
            "info_curve_available": available,
        },
        "patch": {"calls": patch.calls},
        "output": {"states": len(out),
                   "cost": float(final_call["curve"]["cost"][-1])},
        "certificate": certificate,
        "full_curve": {"time": stitched_time, "cost": stitched_cost},
    }
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2, default=float)
    print(json.dumps({
        "scene": args.scene, "choice": args.choice, "seed": args.seed,
        "calls": len(patch.calls), "final_cost": result["output"]["cost"],
        "wall": total_wall, "certificate": certificate,
    }, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
