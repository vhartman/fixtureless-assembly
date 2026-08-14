"""Verify the exact vectorized gate against the native construction gate on real scenes.

Builds one real rai environment, plans a short path, then compares:

* the table's total path cost against the real ``path_cost`` (the build-time self check),
* the table's gate verdict against the native construction gate for many random proposals,
* the table's full-run gains against ``ideation_samplers.full_run_gain``.

Exit code 0 only when every check passes.  Nothing is benchmarked here.
"""

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

import ideation_samplers as IS
from exact_gate import SegmentCostTable
from multi_robot_multi_goal_planning.planners.planner_birrtstar import BidirectionalRRTstar
from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)
from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost
from multi_robot_multi_goal_planning.planners.shortcutting import (
    constant_task_runs,
)


def native_gate(env, new_path, i, j, robots):
    """The unchanged native cost gate: True when the candidate strictly improves."""
    State, _, _, _, _ = IS._mrb()
    element = IS._proposal_candidate(env, new_path, i, j, robots, State)
    return path_cost(element, env.batch_config_cost) < path_cost(
        new_path[i : j + 1], env.batch_config_cost
    )


def main():
    try:
        resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3, 6 * 1024 ** 3))
    except (ValueError, OSError):
        pass

    env_seed = 2
    np.random.seed(env_seed)
    env = get_env_by_name("rai.box_stacking")
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"
    config = BaseRRTConfig()
    config.distance_metric = "max_euclidean"
    config.stepsize = 1.0
    config.shortcutting = False
    config.with_mode_validation = False

    path = None
    for attempt in range(8):
        np.random.seed(env_seed + attempt)
        planner = BidirectionalRRTstar(env, config=config)
        path, _info = planner.plan(ptc=RuntimeTerminationCondition(2.0))
        if path:
            print(f"planner returned a path on attempt {attempt}")
            break
    assert path, "planner returned no path in any attempt"
    new_path = interpolate_path(path, 0.5)

    table = SegmentCostTable(env, lambda: new_path)
    print("gate table ok:", table.ok)
    if not table.ok:
        print("FAIL: table self-check failed")
        raise SystemExit(1)

    real_total = float(path_cost(new_path, env.batch_config_cost))
    print(f"total cost real={real_total:.6f} table={table.total_cost():.6f}")

    runs = [constant_task_runs(new_path, r) for r in range(len(env.robots))]
    print("runs per robot:", [len(rs) for rs in runs])

    rng = np.random.default_rng(12345)
    disagreements = 0
    indeterminate = 0
    checked = 0
    for _ in range(3000):
        r = int(rng.integers(0, len(env.robots)))
        if not runs[r]:
            continue
        lo, hi = runs[r][int(rng.integers(0, len(runs[r])))]
        i = int(rng.integers(lo, hi + 1))
        j = int(rng.integers(lo, hi + 1))
        if i > j:
            i, j = j, i
        if j - i < 2:
            continue
        checked += 1
        verdict = table.decide([r], i, j)
        if verdict is None:
            # The loop falls back to the native construction gate here, so an indeterminate
            # verdict is not a decision error.
            indeterminate += 1
            continue
        native = native_gate(env, new_path, i, j, [r])
        if verdict != native:
            disagreements += 1
            print("DISAGREE", r, i, j, verdict, native, table.delta([r], i, j))

    print(f"checked={checked} indeterminate={indeterminate} disagreements={disagreements}")
    if disagreements:
        print("FAIL: gate disagrees with the native construction gate")
        raise SystemExit(1)

    # full-run gains vs ideation_samplers.full_run_gain
    State, _, path_cost_fun, _, _ = IS._mrb()
    max_rel_error = 0.0
    compared = 0
    for r, rs in enumerate(runs):
        for lo, hi in rs:
            native_gain = IS.full_run_gain(env, new_path, r, lo, hi, State, path_cost_fun)
            table_gain = table.delta([r], lo, hi)
            compared += 1
            rel = abs(native_gain - table_gain) / max(1.0, abs(native_gain))
            max_rel_error = max(max_rel_error, rel)
    print(f"full-run gains compared={compared} max relative error={max_rel_error:.3e}")
    if max_rel_error > 1e-6:
        print("FAIL: full-run gains differ from full_run_gain beyond tolerance")
        raise SystemExit(1)

    # multi-robot set gate (max_compatible) vs native
    disagreements = 0
    checked = 0
    for _ in range(1000):
        r = int(rng.integers(0, len(env.robots)))
        if not runs[r]:
            continue
        lo, hi = runs[r][int(rng.integers(0, len(runs[r])))]
        i = int(rng.integers(lo, hi + 1))
        j = int(rng.integers(lo, hi + 1))
        if i > j:
            i, j = j, i
        if j - i < 2:
            continue
        robots = [
            rr for rr in range(len(env.robots))
            if new_path[i].mode.task_ids[rr] == new_path[j].mode.task_ids[rr]
        ]
        checked += 1
        verdict = table.decide(robots, i, j)
        native = native_gate(env, new_path, i, j, robots)
        if verdict is None:
            continue  # the loop falls back to the native gate here
        if verdict != native:
            disagreements += 1
            print("DISAGREE multi", robots, i, j, verdict, native)
    print(f"multi-robot checked={checked} disagreements={disagreements}")
    if disagreements:
        print("FAIL: multi-robot gate disagrees with the native construction gate")
        raise SystemExit(1)

    print("PASS: gate matches the native construction gate on this scene")

    # Same-process decision equivalence: tree_v2 (exact gate loop) must reproduce
    # persistent_tree (native construction gate loop) exactly at a fixed seed.
    import samplers_v2 as V2

    path_a = interpolate_path(path, 0.5)
    np.random.seed(99)
    stats_a = {}
    out_a, _ = IS.shortcut_run(
        env, path_a, "persistent_tree", max_iter=300, max_attempts=1500,
        resolution=float(env.collision_resolution), tolerance=float(env.collision_tolerance),
        interpolation_resolution=0.5, seed=99, fail_cell=None, stats_out=stats_a,
    )
    np.random.seed(99)
    stats_b = {}
    out_b, _ = V2.shortcut_run_v2(
        env, path_a, "tree_v2", max_iter=300, max_attempts=1500,
        resolution=float(env.collision_resolution), tolerance=float(env.collision_tolerance),
        interpolation_resolution=0.5, seed=99, fail_cell=None, stats_out=stats_b,
    )
    cost_a = float(path_cost(out_a, env.batch_config_cost))
    cost_b = float(path_cost(out_b, env.batch_config_cost))
    same_states = (
        len(out_a) == len(out_b)
        and all(
            np.array_equal(np.asarray(sa.q.state()), np.asarray(sb.q.state()))
            for sa, sb in zip(out_a, out_b)
        )
    )
    print(
        f"tree_v2 vs persistent_tree same-process: cost {cost_a:.6f} vs {cost_b:.6f}, "
        f"accepted {stats_a.get('accepted')} vs {stats_b.get('accepted')}, "
        f"states_identical={same_states}"
    )
    if not same_states or abs(cost_a - cost_b) > 1e-9:
        print("FAIL: tree_v2 does not reproduce persistent_tree decisions")
        raise SystemExit(1)
    print("PASS: tree_v2 reproduces persistent_tree decisions exactly")


if __name__ == "__main__":
    main()
