"""Smoke test for the upstream run_choice="tree" patch on a real assembly scene.

Loads chair8 through benchenv, runs the UPSTREAM robot_mode_shortcut with run_choice="tree",
and checks the strict invariants the benchmark enforces (states/modes unchanged vs the
prepared path, endpoints bitwise, output collision-free, monotone cost trace).

New file; nothing in mr_bench is modified by this script.
"""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import benchenv as B

SCENE = sys.argv[1] if len(sys.argv) > 1 else "chair8"
MAX_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
RUN_CHOICE = sys.argv[3] if len(sys.argv) > 3 else "tree"
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0

env, raw, meta = B.load_scene(SCENE)
from multi_robot_multi_goal_planning.planners.shortcutting import robot_mode_shortcut

np.random.seed(SEED)
random.seed(SEED)
out, (costs, times) = robot_mode_shortcut(
    env, raw, max_iter=MAX_ITER,
    resolution=float(env.collision_resolution), tolerance=float(env.collision_tolerance),
    run_choice=RUN_CHOICE,
)

prep = B.prepared_path(env, raw)
inv = {}
inv["n_states_unchanged"] = (len(out) == len(prep))
omodes = [tuple(int(t) for t in np.atleast_1d(s.mode.task_ids)) for s in out]
pmodes = [tuple(int(t) for t in np.atleast_1d(s.mode.task_ids)) for s in prep]
inv["modes_unchanged"] = (omodes == pmodes)
inv["start_drift"] = float(np.linalg.norm(out[0].q.state() - prep[0].q.state()))
inv["end_drift"] = float(np.linalg.norm(out[-1].q.state() - prep[-1].q.state()))
inv["endpoints_within_1e9"] = bool(inv["start_drift"] < 1e-9 and inv["end_drift"] < 1e-9)
d = np.diff(np.asarray(costs))
inv["max_cost_increase"] = float(d.max()) if len(d) else 0.0
inv["cost_monotone"] = bool(inv["max_cost_increase"] <= 1e-9)
bad = B.is_path_valid(env, out)
inv["collision_free"] = (len(bad) == 0)

print(f"[smoke] {SCENE}: {costs[0]:.2f} -> {costs[-1]:.2f} "
      f"(strict checks done: {len(costs) - 1}) invariants: {inv}")
if not all(inv[k] for k in ("n_states_unchanged", "modes_unchanged",
                            "endpoints_within_1e9", "cost_monotone", "collision_free")):
    print("SMOKE FAILED")
    raise SystemExit(1)
print("SMOKE PASS")
