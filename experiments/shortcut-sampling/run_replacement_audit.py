"""Measure how often accepted shortcuts are later overwritten, without changing mr_bench.

Usage:
    python run_replacement_audit.py <scene> [n_seeds] [arms_csv] [max_checks] [output_tag]

The default arm is the landed baseline reproduced by ``ideation_samplers``.  The output is
``<scene>-replacement-audit.json`` beside this file.  One scene is built per process, matching
the isolation rule in :mod:`benchenv`.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import benchenv as B
import ideation_samplers as IS


def _safe_ratio(a, b):
    return float(a / b) if b else 0.0


def _aggregate(runs):
    summaries = [r["audit"]["summary"] for r in runs]
    keys = [
        "accepted_events",
        "partly_or_fully_overwritten_events",
        "fully_overwritten_events",
        "accepted_gain_sum",
        "fully_overwritten_gain_sum",
        "successful_collision_wall_s",
        "fully_overwritten_successful_collision_wall_s",
        "successful_config_checks",
        "fully_overwritten_successful_config_checks",
        "exact_repeated_collision_checks",
        "exact_repeated_failed_collision_checks",
        "exact_repeat_collision_wall_s",
        "exact_repeat_config_checks",
        "audit_overhead_s",
    ]
    total = {key: sum(s[key] for s in summaries) for key in keys}
    total.update(
        fully_overwritten_event_fraction=_safe_ratio(
            total["fully_overwritten_events"], total["accepted_events"]
        ),
        fully_overwritten_gain_fraction=_safe_ratio(
            total["fully_overwritten_gain_sum"], total["accepted_gain_sum"]
        ),
        fully_overwritten_successful_collision_wall_fraction=_safe_ratio(
            total["fully_overwritten_successful_collision_wall_s"],
            total["successful_collision_wall_s"],
        ),
        fully_overwritten_successful_config_fraction=_safe_ratio(
            total["fully_overwritten_successful_config_checks"],
            total["successful_config_checks"],
        ),
    )
    lifetimes = [
        event["lifetime_checks"]
        for run in runs for event in run["audit"]["accepted"]
        if event["fully_overwritten"]
    ]
    lifetime_s = [
        event["lifetime_s"]
        for run in runs for event in run["audit"]["accepted"]
        if event["fully_overwritten"]
    ]
    total["fully_overwritten_lifetime_checks_median"] = (
        float(np.median(lifetimes)) if lifetimes else None
    )
    total["fully_overwritten_lifetime_s_median"] = (
        float(np.median(lifetime_s)) if lifetime_s else None
    )
    return total


def main():
    if len(sys.argv) < 2:
        print("scenes:", ", ".join(B.scene_names()))
        raise SystemExit(2)
    scene = sys.argv[1]
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    arms = sys.argv[3].split(",") if len(sys.argv) > 3 else ["baseline"]
    max_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    output_tag = sys.argv[5] if len(sys.argv) > 5 else ""
    max_attempts = 5 * max_iter

    env, raw, meta = B.load_scene(scene)
    from multi_robot_multi_goal_planning.problems.planning_env import State

    q_saved = [np.asarray(s.q.state(), dtype=float).copy() for s in raw]
    modes = [s.mode for s in raw]
    conf = type(raw[0].q)
    slices = raw[0].q._array_slice

    def fresh():
        return [State(conf(q.copy(), slices), mode) for q, mode in zip(q_saved, modes)]

    counter = B.count_collision_checks(env)

    # Capture the first failing coarse path edge when the environment exposes its q object.
    fail = {"q": None}
    inner_edge = env.is_edge_collision_free
    inner_cfg = env.is_collision_free

    def edge_capture(q1, *args, **kwargs):
        result = inner_edge(q1, *args, **kwargs)
        if not result:
            fail["q"] = q1
        return result

    def cfg_capture(q, *args, **kwargs):
        result = inner_cfg(q, *args, **kwargs)
        if not result:
            fail["q"] = q
        return result

    env.is_edge_collision_free = edge_capture
    env.is_collision_free = cfg_capture

    runs = []
    for seed in range(n_seeds):
        order = arms[seed % len(arms):] + arms[:seed % len(arms)]
        for arm in order:
            counter.reset()
            fail["q"] = None
            np.random.seed(seed)
            stats, audit = {}, {}
            t0 = time.perf_counter()
            out, (costs, native_times) = IS.shortcut_run(
                env,
                fresh(),
                arm,
                max_iter=max_iter,
                max_attempts=max_attempts,
                resolution=float(env.collision_resolution),
                tolerance=float(env.collision_tolerance),
                interpolation_resolution=0.5,
                seed=seed,
                fail_cell=fail,
                stats_out=stats,
                audit_out=audit,
                check_counter=counter,
            )
            wall = time.perf_counter() - t0
            bad = B.is_path_valid(env, out)
            checks = audit["checks"]
            assert len(checks) == len(costs) - 1
            cumulative_configs = np.cumsum([c["config_checks"] for c in checks])
            cumulative_edges = np.cumsum([c["edge_checks"] for c in checks])
            record = {
                "scene": scene,
                "arm": arm,
                "seed": seed,
                "max_iter": max_iter,
                "max_attempts": max_attempts,
                "wall_s": float(wall),
                "cost_initial": float(costs[0]),
                "cost_final": float(costs[-1]),
                "collision_checks": int(stats["collision_checks"]),
                "config_checks": int(counter.configs),
                "edge_checks": int(counter.edges),
                "accepted": int(stats["accepted"]),
                "rejected_collision": int(stats["rejected_collision"]),
                "cost_rejects": int(stats["cost_rejects"]),
                "collision_free_output": not bad,
                "colliding_output_states": bad,
                "audit": audit,
                "curve": {
                    "checks": list(range(1, len(checks) + 1)),
                    "configs": [int(v) for v in cumulative_configs],
                    "edges": [int(v) for v in cumulative_edges],
                    "time": [float(c["elapsed_s"]) for c in checks],
                    "native_time_with_audit": [float(t) for t in native_times[1:]],
                    "cost": [float(c) for c in costs[1:]],
                },
            }
            runs.append(record)
            summary = audit["summary"]
            print(
                f"[{scene} {arm} s{seed}] {costs[0]:.2f}->{costs[-1]:.2f} "
                f"checks={stats['collision_checks']} accepted={stats['accepted']} "
                f"full-overwrite={summary['fully_overwritten_events']}/"
                f"{summary['accepted_events']} "
                f"repeat-fails={summary['exact_repeated_failed_collision_checks']} "
                f"audit={1e3 * summary['audit_overhead_s']:.1f}ms",
                flush=True,
            )

    aggregate = {}
    for arm in arms:
        aggregate[arm] = _aggregate([run for run in runs if run["arm"] == arm])
    payload = {
        "scene": scene,
        "meta": {k: v for k, v in meta.items()
                 if isinstance(v, (int, float, str, bool, list, type(None)))},
        "n_seeds": n_seeds,
        "arms": arms,
        "max_iter": max_iter,
        "aggregate": aggregate,
        "runs": runs,
    }
    suffix = f"-{output_tag}" if output_tag else ""
    output = os.path.join(HERE, f"{scene}-replacement-audit{suffix}.json")
    with open(output, "w") as stream:
        json.dump(payload, stream, indent=1, default=float)
    print("aggregate:", json.dumps(aggregate, indent=1))
    print("wrote", output)


if __name__ == "__main__":
    main()
