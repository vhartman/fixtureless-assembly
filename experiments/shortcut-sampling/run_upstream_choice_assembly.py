"""Upstream span vs tree on the real assembly scenes, saved in the upstream-choice schema.

Loads one assembly fixture through benchenv, runs the LANDED robot_mode_shortcut with a
forced run_choice, and writes a JSON compatible with plot_upstream_choice.py (protocol tag,
policy = choice, strict_full_edge certificate, full_curve).  One scene per process.

    python run_upstream_choice_assembly.py <scene> --choice span|tree \
        --seed S --checks 1000 --output out.json

New file; no earlier experiment file is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import benchenv as B

PROTOCOL = "shortcut-portfolio-v5-exact-gate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--choice", choices=("span", "tree", "whole_random"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--checks", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    env, raw, meta = B.load_scene(args.scene)
    from multi_robot_multi_goal_planning.planners.shortcutting import robot_mode_shortcut
    from multi_robot_multi_goal_planning.problems.util import path_cost

    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.choice == "whole_random":
        from run_upstream_choice import _whole_path_random_shortcut
        out, (costs, times) = _whole_path_random_shortcut(
            env, raw, max_iter=args.checks,
            resolution=float(env.collision_resolution),
            tolerance=float(env.collision_tolerance),
        )
    else:
        out, (costs, times) = robot_mode_shortcut(
            env, raw, max_iter=args.checks,
            resolution=float(env.collision_resolution),
            tolerance=float(env.collision_tolerance),
            run_choice=args.choice,
        )

    prep = B.prepared_path(env, raw)
    start_err = float(np.max(np.abs(out[0].q.state() - prep[0].q.state())))
    end_err = float(np.max(np.abs(out[-1].q.state() - prep[-1].q.state())))
    omodes = [tuple(int(t) for t in np.atleast_1d(s.mode.task_ids)) for s in out]
    pmodes = [tuple(int(t) for t in np.atleast_1d(s.mode.task_ids)) for s in prep]
    strict = bool(env.is_path_collision_free(
        out, resolution=float(env.collision_resolution),
        tolerance=float(env.collision_tolerance), check_start_and_end=True,
        check_edges_in_order=True, binary_order=False,
    ))
    certificate = {
        "strict_full_edge": strict,
        "valid_plan": True,
        "terminal_mode": bool(env.is_terminal_mode(out[-1].mode)),
        "mode_runs_preserved": omodes == pmodes,
        "endpoints_preserved": start_err <= 1e-8 and end_err <= 1e-8,
        "endpoint_max_abs_error": max(start_err, end_err),
        "trace_matches_output": abs(float(costs[-1])
                                    - path_cost(out, env.batch_config_cost)) <= 1e-8,
    }
    required = ("strict_full_edge", "terminal_mode", "mode_runs_preserved",
                "endpoints_preserved", "trace_matches_output")
    if not all(certificate[key] for key in required):
        raise RuntimeError(f"strict output invariant failed: {certificate}")

    result = {
        "protocol": PROTOCOL,
        "purpose": "landed robot_mode_shortcut with forced run_choice on a real assembly "
                   "fixture (standalone matched pass)",
        "env": f"rai.{args.scene}",
        "policy": args.choice,
        "seed": int(args.seed),
        "env_seed": int(meta.get("mp_seed", 1)),
        "run_id": int(args.run_id),
        "planner": {"returned_path": True, "total_wall_seconds": float(times[-1])},
        "output": {"states": len(out), "cost": float(costs[-1])},
        "certificate": certificate,
        "full_curve": {
            "time": [float(t) for t in times],
            "cost": [float(c) for c in costs],
        },
    }
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2, default=float)
    print(f"[assembly] {args.scene} {args.choice} seed {args.seed}: "
          f"{costs[0]:.2f} -> {costs[-1]:.2f} strict={strict}")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
