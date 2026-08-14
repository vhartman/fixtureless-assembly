"""Diagnostic: find the first strict-certificate failing edge of the lazy arm on box seed 2."""
from __future__ import annotations

import os
import resource
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import lazy_certify as LZ
from multi_robot_multi_goal_planning.planners.planner_birrtstar import BidirectionalRRTstar
from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)
from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.util import path_cost


def main():
    try:
        resource.setrlimit(resource.RLIMIT_AS, (12 * 1024 ** 3, 12 * 1024 ** 3))
    except (ValueError, OSError):
        pass

    env_seed, seed = 2, 2
    np.random.seed(env_seed)
    env = get_env_by_name("rai.triple_waypoints")
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"
    config = BaseRRTConfig()
    config.distance_metric = "max_euclidean"
    config.stepsize = 1.0
    config.shortcutting = False
    config.with_mode_validation = False

    np.random.seed(seed)
    path = None
    for attempt in range(3):
        np.random.seed(seed + attempt)
        planner = BidirectionalRRTstar(env, config=config)
        path, _info = planner.plan(ptc=RuntimeTerminationCondition(6.0))
        if path:
            print(f"planner path on attempt {attempt}")
            break
        del planner
    assert path
    input_cost = float(path_cost(path, env.batch_config_cost))

    fail = {"q": None}
    inner_edge = env.is_edge_collision_free
    inner_config = env.is_collision_free

    def edge_capture(q1, *args, **kwargs):
        result = inner_edge(q1, *args, **kwargs)
        if not result:
            fail["q"] = q1
        return result

    def config_capture(q, *args, **kwargs):
        result = inner_config(q, *args, **kwargs)
        if not result:
            fail["q"] = q
        return result

    env.is_edge_collision_free = edge_capture
    env.is_collision_free = config_capture
    stats = {}
    debug = {}
    out, trace = LZ.shortcut_run_lazy(
        env, path, max_iter=1000, max_attempts=5000,
        resolution=float(env.collision_resolution),
        tolerance=float(env.collision_tolerance),
        interpolation_resolution=0.5, seed=seed, wave_size=20,
        fail_cell=fail, stats_out=stats, debug_out=debug,
    )
    env.is_edge_collision_free = inner_edge
    env.is_collision_free = inner_config

    print("input cost", input_cost, "final cost", float(path_cost(out, env.batch_config_cost)))
    print("stats:", {k: v for k, v in stats.items() if k in
                     ("accepted", "provisional_accepted", "wave_commits", "wave_failures",
                      "degraded_waves", "rejected_collision", "gate_eps_fallback")})

    # find the first failing edge under the strict certificate
    for i in range(len(out) - 1):
        if not env.is_collision_free(out[i].q, out[i].mode):
            from multi_robot_multi_goal_planning.problems.util import interpolate_path
            base = interpolate_path(path, 0.5)
            same_as_input = (
                i < len(base)
                and np.array_equal(np.asarray(out[i].q.state()), np.asarray(base[i].q.state()))
            )
            print("first failing STATE", i, "identical to unshortcut input:", bool(same_as_input))
            print("  same object as input state:", out[i].q is base[i].q)
            print("  input state value:", np.asarray(base[i].q.state()).tolist()[:6], "...")
            print("  final state value:", np.asarray(out[i].q.state()).tolist()[:6], "...")
            print("  input state collides too:",
                  [env.is_collision_free(base[i].q, base[i].mode) for _ in range(2)])
            return
    for i in range(len(out) - 1):
        free = env.is_edge_collision_free(
            out[i].q, out[i + 1].q, out[i].mode,
            resolution=float(env.collision_resolution),
            tolerance=float(env.collision_tolerance), include_endpoints=False,
        )
        if not free:
            from multi_robot_multi_goal_planning.problems.util import interpolate_path
            base = interpolate_path(path, 0.5)
            same_as_input = (
                i < len(base) - 1
                and np.array_equal(np.asarray(out[i].q.state()), np.asarray(base[i].q.state()))
                and np.array_equal(np.asarray(out[i + 1].q.state()), np.asarray(base[i + 1].q.state()))
            )
            print("first failing EDGE", i, "of", len(out) - 1)
            print("  mode", out[i].mode.task_ids)
            print("  identical to unshortcut input edge:", bool(same_as_input))
            print("  strict cert on the UNMODIFIED input path at this edge:",
                  env.is_edge_collision_free(
                      base[i].q, base[i + 1].q, base[i].mode,
                      resolution=float(env.collision_resolution),
                      tolerance=float(env.collision_tolerance), include_endpoints=False,
                  ))
            for _ in range(3):
                free2 = env.is_edge_collision_free(
                    out[i].q, out[i + 1].q, out[i].mode,
                    resolution=float(env.collision_resolution),
                    tolerance=float(env.collision_tolerance), include_endpoints=False,
                )
                print("  recheck ->", free2)
            return
    print("all edges free")


if __name__ == "__main__":
    main()
