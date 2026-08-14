"""Upstream-vs-upstream full-loop comparison.

Runs the LANDED mr_bench planner loop with every internal shortcut call and the final pass
forced to a single ``run_choice`` (``span`` = the historical random selector, ``tree`` = the
new structural prefix).  No sampler is monkeypatched: only the run_choice keyword of the
already-imported ``robot_mode_shortcut`` aliases is overridden, and everything else is the
unchanged landed code path.

Protocol matches the v5 full phase: environment built at ``env_seed``, planner and shortcut
seeded with ``seed``, 20 s BiRRT* with optimize, final 1,000-validation pass, strict
full-edge certificate on every output.

    python run_upstream_choice.py <env> --choice span|tree --seed S --env-seed E \
        --run-id R --runtime 20 --checks 1000 --output out.json

New file; the earlier experiment files are untouched.
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

from multi_robot_multi_goal_planning.planners import shortcutting as shortcutting_module
from multi_robot_multi_goal_planning.planners.planner_birrtstar import BidirectionalRRTstar
from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)
from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost

from run_live_persistent_v2 import (
    _environment_fingerprint, _mode_runs, _summary,
)


PROTOCOL = "shortcut-portfolio-v5-exact-gate"
CHOICES = ("span", "tree", "whole_random")


def _whole_path_random_shortcut(env, path, max_iter=1000, resolution=0.001,
                                tolerance=0.01, interpolation_resolution=0.5,
                                max_attempts=None, **_):
    """The historical pre-run-decomposition shortcut loop, reproduced locally.

    Two indices are uniform over the whole interpolated path, the robot is round-robin, and a
    pair whose task id differs across the interval is discarded by the gate.  Candidate
    construction, cost gate, collision checking and application follow the landed
    robot_mode_shortcut exactly (including the mode-switch doubling convention).
    """
    new_path = interpolate_path(path, interpolation_resolution)
    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]
    start_time = time.time()
    if max_attempts is None:
        max_attempts = max(2500, 5 * int(max_iter))
    cnt = 0
    iter_ = 0
    rr = 0
    n = len(new_path)
    n_robots = len(env.robots)
    while True:
        iter_ += 1
        if cnt >= max_iter or iter_ >= max_attempts:
            break
        i = int(np.random.randint(0, n))
        j = int(np.random.randint(0, n))
        if i > j:
            i, j = j, i
        if j - i < 2:
            continue
        robot = rr % n_robots
        rr += 1
        if new_path[i].mode.task_ids[robot] != new_path[j].mode.task_ids[robot]:
            continue

        q0 = new_path[i].q
        q1 = new_path[j].q
        q0_tmp = q0[robot] * 1
        diff = (q1[robot] * 1 - q0_tmp) / (j - i)
        path_element = []
        for k in range(j - i + 1):
            q = new_path[i + k].q.state() * 1.0
            r_cnt = 0
            for r in range(len(env.robots)):
                dim = env.robot_dims[env.robots[r]]
                if r == robot:
                    if k != 0 and i + k != j and new_path[i + k].mode != new_path[i + k - 1].mode:
                        q_interp = q0_tmp + diff * (k - 1)
                    else:
                        q_interp = q0_tmp + diff * k
                    q[r_cnt:r_cnt + dim] = q_interp
                r_cnt += dim
            path_element.append(State(q0.from_flat(q), new_path[i + k].mode))

        if path_cost(path_element, env.batch_config_cost) >= path_cost(
            new_path[i:j + 1], env.batch_config_cost
        ):
            continue

        cnt += 1
        if env.is_path_collision_free(
            path_element, resolution=resolution, tolerance=tolerance,
            check_start_and_end=False,
        ):
            for k in range(j - i + 1):
                new_path[i + k].q = path_element[k].q

        current_time = time.time()
        times.append(current_time - start_time)
        costs.append(path_cost(new_path, env.batch_config_cost))

    return new_path, [costs, times]


class ChoicePatch:
    """Force ``run_choice`` on every robot_mode_shortcut call in the process."""

    def __init__(self, choice):
        self.choice = choice
        self.original = shortcutting_module.robot_mode_shortcut
        self.calls = []
        self.restores = []

    def __enter__(self):
        original, choice, calls = self.original, self.choice, self.calls

        def patched(env, path, max_iter=1000, resolution=0.001, tolerance=0.01,
                    robot_choice="round_robin", interpolation_resolution=0.5,
                    max_attempts=None, run_choice="gain", **extra):
            t0 = time.perf_counter()
            input_summary = _summary(env, path)
            input_start = np.asarray(path[0].q.state(), dtype=float).copy()
            input_end = np.asarray(path[-1].q.state(), dtype=float).copy()
            if choice == "whole_random":
                out, trace = _whole_path_random_shortcut(
                    env, path, max_iter=int(max_iter), resolution=float(resolution),
                    tolerance=float(tolerance),
                    interpolation_resolution=float(interpolation_resolution),
                    max_attempts=max_attempts, **extra,
                )
            else:
                out, trace = original(
                    env, path, max_iter=int(max_iter), resolution=float(resolution),
                    tolerance=float(tolerance), robot_choice=robot_choice,
                    interpolation_resolution=float(interpolation_resolution),
                    max_attempts=max_attempts, run_choice=choice, **extra,
                )
            wall = time.perf_counter() - t0
            calls.append({
                "run_choice": choice, "max_iter": int(max_iter),
                "wall_seconds": float(wall),
                "input": input_summary,
                "input_start": input_start.tolist(),
                "input_end": input_end.tolist(),
                "output": _summary(env, out),
                "curve": {
                    "cost": [float(c) for c in trace[0]],
                    "time": [float(t) for t in trace[1]],
                },
            })
            return out, trace

        self.patched = patched
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith("multi_robot_multi_goal_planning"):
                continue
            if getattr(module, "robot_mode_shortcut", None) is original:
                setattr(module, "robot_mode_shortcut", patched)
                self.restores.append((module, original))
        if shortcutting_module.robot_mode_shortcut is original:
            shortcutting_module.robot_mode_shortcut = patched
            self.restores.append((shortcutting_module, original))
        return self

    def __exit__(self, exc_type, exc, tb):
        for module, original in reversed(self.restores):
            if getattr(module, "robot_mode_shortcut", None) is self.patched:
                setattr(module, "robot_mode_shortcut", original)
        return False


def _certificate(env, path, call):
    start = np.asarray(call["input_start"], dtype=float)
    end = np.asarray(call["input_end"], dtype=float)
    endpoint_error = max(
        float(np.max(np.abs(path[0].q.state() - start))),
        float(np.max(np.abs(path[-1].q.state() - end))),
    )
    strict = bool(env.is_path_collision_free(
        path, resolution=float(env.collision_resolution),
        tolerance=float(env.collision_tolerance), check_start_and_end=True,
        check_edges_in_order=True, binary_order=False,
    ))
    valid = bool(env.is_valid_plan(path)) if hasattr(env, "is_valid_plan") else True
    output = _summary(env, path)
    return {
        "strict_full_edge": strict,
        "valid_plan": valid,
        "terminal_mode": bool(env.is_terminal_mode(path[-1].mode)),
        "mode_runs_preserved": output["mode_runs"] == call["input"]["mode_runs"],
        "endpoints_preserved": endpoint_error <= 1e-8,
        "endpoint_max_abs_error": endpoint_error,
        "trace_matches_output": abs(output["cost"] - call["curve"]["cost"][-1]) <= 1e-8,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env")
    parser.add_argument("--choice", choices=CHOICES, required=True)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--env-seed", type=int)
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--runtime", type=float, default=20.0)
    parser.add_argument("--checks", type=int, default=1000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3, 6 * 1024 ** 3))
    except (ValueError, OSError):
        pass

    env_seed = args.seed if args.env_seed is None else int(args.env_seed)
    np.random.seed(env_seed)
    random.seed(env_seed)
    env = get_env_by_name(args.env)
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
            "env": args.env, "policy": args.choice, "seed": args.seed,
            "env_seed": env_seed, "run_id": int(args.run_id),
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
        "purpose": "landed mr_bench full loop with run_choice forced to span or tree; "
                   "no sampler monkeypatch",
        "env": args.env, "policy": args.choice, "seed": args.seed,
        "env_seed": env_seed, "run_id": int(args.run_id),
        "environment_fingerprint": env_fingerprint,
        "planner": {
            "runtime_seconds": float(args.runtime), "optimize": True,
            "returned_path": True, "total_wall_seconds": float(total_wall),
            "info_costs": info_costs, "info_times": info_times,
            "info_curve_available": available,
        },
        "patch": {"calls": patch.calls},
        "output": _summary(env, out),
        "certificate": certificate,
        "full_curve": {"time": stitched_time, "cost": stitched_cost},
    }
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2, default=float)
    print(json.dumps({
        "env": args.env, "choice": args.choice, "seed": args.seed,
        "calls": len(patch.calls), "final_cost": result["output"]["cost"],
        "wall": total_wall, "certificate": certificate,
    }, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
