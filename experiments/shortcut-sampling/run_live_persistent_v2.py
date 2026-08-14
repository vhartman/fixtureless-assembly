"""Run the round-3 shortcut policies in the real generic mr_bench planner loop.

New file; ``run_live_persistent.py`` and every earlier experiment file are untouched.  Like
its predecessor it only monkeypatches already-imported ``robot_mode_shortcut`` aliases,
runs one generic BiRRT* process, restores the aliases, and writes a JSON trace.

Round-3 additions (protocol ``shortcut-portfolio-v5-exact-gate``):

* the exact vectorized segment-cost gate (``exact_gate.py``) inside ``samplers_v2``,
* adaptive stop, bounded tree prefix, warm prefix, cadence credit (``samplers_v2``),
* lazy certification with waves (``lazy_certify.py``),
* a default 6 GB address-space limit per process so a run dies cleanly instead of dragging
  the machine down (``--ram-limit-gb``).

``native`` and ``persistent_tree`` run the unchanged v4 code path, so they remain the
reference baselines under the same fixed-environment protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import resource
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import ideation_samplers as IS
import lazy_certify as LZ
import samplers_v2 as V2
from multi_robot_multi_goal_planning.planners import shortcutting as shortcutting_module
from multi_robot_multi_goal_planning.planners.planner_birrtstar import BidirectionalRRTstar
from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)
from multi_robot_multi_goal_planning.problems import get_env_by_name
from multi_robot_multi_goal_planning.problems.util import path_cost


PROTOCOL = "shortcut-portfolio-v5-exact-gate"
BASELINE_POLICIES = ("native", "persistent_tree")
V2_POLICIES = (
    "tree_v2", "tree_adaptive", "tree_partial", "tree_warm", "tree_credit",
    "tree_simple", "tree_split2", "tree_rand", "tree_cap", "tree_warmbeta",
    "tree_mix", "tree_only",
    "length_uniform", "length_tilted", "max_compatible", "max_bandit",
    "lazy_certify",
)
POLICIES = BASELINE_POLICIES + V2_POLICIES


class CheckCounter:
    """Observational instance wrappers; no extra collision query is issued."""

    def __init__(self, env):
        self.env = env
        self.configs = self.edges = self.paths = 0
        names = (
            "is_collision_free", "is_collision_free_np",
            "is_edge_collision_free", "is_path_collision_free",
        )
        self.original = {name: getattr(env, name) for name in names}
        env.is_collision_free = self._config(self.original["is_collision_free"])
        env.is_collision_free_np = self._config(self.original["is_collision_free_np"])
        env.is_edge_collision_free = self._call("edges", self.original["is_edge_collision_free"])
        env.is_path_collision_free = self._call("paths", self.original["is_path_collision_free"])

    def _config(self, fn):
        def wrapped(*args, **kwargs):
            self.configs += 1
            return fn(*args, **kwargs)
        return wrapped

    def _call(self, field, fn):
        def wrapped(*args, **kwargs):
            setattr(self, field, getattr(self, field) + 1)
            return fn(*args, **kwargs)
        return wrapped

    def restore(self):
        for name, fn in self.original.items():
            setattr(self.env, name, fn)


def _mode_runs(path):
    modes = [tuple(int(x) for x in np.atleast_1d(s.mode.task_ids)) for s in path]
    if not modes:
        return []
    out = [list(modes[0])]
    for mode in modes[1:]:
        if list(mode) != out[-1]:
            out.append(list(mode))
    return out


def _summary(env, path):
    return {
        "states": len(path),
        "cost": float(path_cost(path, env.batch_config_cost)),
        "mode_runs": _mode_runs(path),
    }


def _stable_value(value, depth=0):
    """Small deterministic serializer used only for an environment audit fingerprint."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": str(array.dtype), "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, (list, tuple)):
        return [_stable_value(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _stable_value(item, depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if hasattr(value, "state") and callable(value.state):
        return {
            "type": type(value).__qualname__,
            "state": _stable_value(np.asarray(value.state()), depth + 1),
        }
    if depth < 4 and hasattr(value, "__dict__"):
        return {
            "type": type(value).__qualname__,
            "fields": _stable_value(vars(value), depth + 1),
        }
    return {"type": type(value).__qualname__}


def _environment_fingerprint(env):
    """Hash scene geometry, start state, task goals, and robot metadata."""
    scene = None
    if hasattr(env, "C"):
        scene = {
            "joint_state": np.asarray(env.C.getJointState()),
            "frame_state": np.asarray(env.C.getFrameState()),
        }
    tasks = []
    for task in getattr(env, "tasks", []):
        tasks.append({
            "name": getattr(task, "name", None),
            "type": getattr(task, "type", None),
            "robots": getattr(task, "robots", None),
            "frames": getattr(task, "frames", None),
            "side_effect": getattr(task, "side_effect", None),
            "side_effect_data": getattr(task, "side_effect_data", None),
            "goal": getattr(task, "goal", None),
            "constraints": getattr(task, "constraints", None),
        })
    payload = _stable_value({
        "class": type(env).__qualname__,
        "robots": getattr(env, "robots", None),
        "robot_dims": getattr(env, "robot_dims", None),
        "limits": getattr(env, "limits", None),
        "start": getattr(env, "start_pos", None),
        "tasks": tasks,
        "scene": scene,
    })
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class LivePatchV2:
    policy: str
    seed: int
    time_origin: float
    calls: list[dict[str, Any]] = field(default_factory=list)
    patched_modules: list[str] = field(default_factory=list)
    restores: list[tuple[Any, Any]] = field(default_factory=list)
    portfolio_state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.original = shortcutting_module.robot_mode_shortcut
        self.original_remove_interpolated = shortcutting_module.remove_interpolated_nodes

    def _make(self):
        original, policy, seed0, calls = self.original, self.policy, self.seed, self.calls

        def patched(env, path, max_iter=1000, resolution=0.001, tolerance=0.01,
                    robot_choice="round_robin", interpolation_resolution=0.5,
                    max_attempts=None, run_choice="gain", **extra):
            if extra or run_choice != "gain" or float(interpolation_resolution) != 0.5:
                raise RuntimeError(
                    "shortcut experiment supports the current gain invocation only: "
                    f"run_choice={run_choice!r}, interpolation={interpolation_resolution!r}, "
                    f"extra={sorted(extra)}"
                )
            if max_attempts is None:
                max_attempts = max(2500, 5 * int(max_iter))

            call_start = time.perf_counter()
            input_summary = _summary(env, path)
            input_start = np.asarray(path[0].q.state(), dtype=float).copy()
            input_end = np.asarray(path[-1].q.state(), dtype=float).copy()
            counter = CheckCounter(env)
            # Shortcut proposals must not advance the planner's RNG stream.
            np_state = np.random.get_state()
            py_state = random.getstate()
            call_seed = seed0 + len(calls)
            np.random.seed(call_seed)
            random.seed(call_seed)
            stats = None
            try:
                if policy == "native":
                    # The landed mr_bench routine, unchanged; bit-for-bit baseline.
                    out, trace = original(
                        env, path, max_iter=int(max_iter), resolution=float(resolution),
                        tolerance=float(tolerance), robot_choice=robot_choice,
                        interpolation_resolution=float(interpolation_resolution),
                        max_attempts=int(max_attempts), run_choice=run_choice,
                    )
                else:
                    # Capture the coarse failing path edge already discovered by the native
                    # checker. This observer sits above the counter and is removed with it.
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
                    if policy == "persistent_tree":
                        out, trace = IS.shortcut_run(
                            env, path, policy, max_iter=int(max_iter),
                            max_attempts=int(max_attempts), resolution=float(resolution),
                            tolerance=float(tolerance),
                            interpolation_resolution=float(interpolation_resolution),
                            seed=call_seed, fail_cell=fail, stats_out=stats,
                            check_counter=counter, shared_state=self.portfolio_state,
                        )
                    elif policy == "lazy_certify":
                        out, trace = LZ.shortcut_run_lazy(
                            env, path, max_iter=int(max_iter),
                            max_attempts=int(max_attempts), resolution=float(resolution),
                            tolerance=float(tolerance),
                            interpolation_resolution=float(interpolation_resolution),
                            seed=call_seed, wave_size=LZ.WAVE_SIZE, fail_cell=fail,
                            stats_out=stats, check_counter=counter,
                            shared_state=self.portfolio_state,
                        )
                    else:
                        kwargs = dict(
                            arm=policy,
                            adaptive_stop=V2.ADAPTIVE_STOP_K
                            if policy == "tree_adaptive" else None,
                            checks_cap=V2.CREDIT_INTERNAL_CHECKS
                            if (policy == "tree_credit" and int(max_iter) <= 100)
                            else None,
                        )
                        out, trace = V2.shortcut_run_v2(
                            env, path, max_iter=int(max_iter),
                            max_attempts=int(max_attempts), resolution=float(resolution),
                            tolerance=float(tolerance),
                            interpolation_resolution=float(interpolation_resolution),
                            seed=call_seed, fail_cell=fail, stats_out=stats,
                            check_counter=counter, shared_state=self.portfolio_state,
                            **kwargs,
                        )
            finally:
                counter.restore()
                np.random.set_state(np_state)
                random.setstate(py_state)

            wall = time.perf_counter() - call_start
            trace_last = float(trace[1][-1]) if trace[1] else 0.0
            setup_offset = max(0.0, wall - trace_last)
            start_elapsed = call_start - self.time_origin
            integrated_time = [float(start_elapsed)]
            integrated_cost = [float(input_summary["cost"])]
            integrated_time.extend(float(start_elapsed + setup_offset + t) for t in trace[1])
            integrated_cost.extend(float(c) for c in trace[0])
            calls.append({
                "call_index": len(calls),
                "policy": policy,
                "input": input_summary,
                "output": _summary(env, out),
                "input_start": input_start.tolist(),
                "input_end": input_end.tolist(),
                "acceptance": {
                    "max_iter": int(max_iter), "max_attempts": int(max_attempts),
                    "resolution": float(resolution), "tolerance": float(tolerance),
                },
                "shortcut_start_elapsed_seconds": float(start_elapsed),
                "shortcut_wall_seconds": float(wall),
                "trace_setup_offset_seconds": float(setup_offset),
                "collision_checks": {
                    "configs": int(counter.configs), "edges": int(counter.edges),
                    "paths": int(counter.paths),
                },
                "curve": {
                    "cost": [float(c) for c in trace[0]],
                    "time": [float(t) for t in trace[1]],
                },
                "integrated_shortcut_curve": {
                    "cost": integrated_cost, "time": integrated_time,
                },
                "stats": stats,
            })
            return out, trace

        patched.__name__ = "robot_mode_shortcut"
        patched._shortcut_experiment_patch = True
        patched._shortcut_experiment_original = original
        return patched

    def __enter__(self):
        # Shared safety prerequisite: preserve every certified input edge.
        shortcutting_module.remove_interpolated_nodes = IS.preserve_input_nodes
        self.patched = self._make()
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith("multi_robot_multi_goal_planning"):
                continue
            if getattr(module, "robot_mode_shortcut", None) is self.original:
                setattr(module, "robot_mode_shortcut", self.patched)
                self.restores.append((module, self.original))
                self.patched_modules.append(name)
        if shortcutting_module.robot_mode_shortcut is not self.patched:
            shortcutting_module.robot_mode_shortcut = self.patched
            self.restores.append((shortcutting_module, self.original))
            self.patched_modules.append(shortcutting_module.__name__)
        self.patched_modules = sorted(set(self.patched_modules))
        return self

    def __exit__(self, exc_type, exc, tb):
        for module, original in reversed(self.restores):
            if getattr(module, "robot_mode_shortcut", None) is self.patched:
                setattr(module, "robot_mode_shortcut", original)
        shortcutting_module.remove_interpolated_nodes = self.original_remove_interpolated
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
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--env-seed", type=int,
        help="fixed environment-construction seed (defaults to --seed for old one-off usage)",
    )
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--runtime", type=float, default=2.0)
    parser.add_argument("--checks", type=int, default=1000)
    parser.add_argument("--stepsize", type=float, default=1.0)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ram-limit-gb", type=float, default=6.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Cap the address space so an experiment dies cleanly instead of dragging the machine down.
    limit = int(args.ram_limit_gb * 1024 ** 3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError) as exc:
        print(f"warning: could not set RAM limit: {exc}", flush=True)

    env_seed = args.seed if args.env_seed is None else int(args.env_seed)
    np.random.seed(env_seed)
    random.seed(env_seed)
    env = get_env_by_name(args.env)
    env_fingerprint = _environment_fingerprint(env)
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"
    config = BaseRRTConfig()
    config.distance_metric = "max_euclidean"
    config.stepsize = float(args.stepsize)
    config.shortcutting = bool(args.optimize)
    config.with_mode_validation = False

    np.random.seed(args.seed)
    random.seed(args.seed)
    run_start = time.perf_counter()
    with LivePatchV2(args.policy, args.seed, run_start) as patch:
        sink = open(os.devnull, "w") if args.quiet else None
        try:
            redirect = contextlib.redirect_stdout(sink) if sink else contextlib.nullcontext()
            with redirect:
                planner = BidirectionalRRTstar(env, config=config)
                planned_path, info = planner.plan(
                    ptc=RuntimeTerminationCondition(float(args.runtime)),
                    optimize=bool(args.optimize),
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
            "env": args.env, "policy": args.policy, "seed": args.seed,
            "env_seed": env_seed, "run_id": int(args.run_id),
            "environment_fingerprint": env_fingerprint,
            "planner": {"returned_path": False, "runtime_seconds": args.runtime,
                        "optimize": args.optimize, "total_wall_seconds": total_wall},
            "patch": {"patched_modules": patch.patched_modules,
                      "shortcut_calls": patch.calls},
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
    final_curve = final_call["integrated_shortcut_curve"]
    if final_curve["time"]:
        handoff = float(final_call["shortcut_start_elapsed_seconds"])
        if not stitched_time or handoff > stitched_time[-1]:
            stitched_time.append(handoff)
            stitched_cost.append(float(final_call["input"]["cost"]))
        stitched_time.extend(final_curve["time"][1:])
        stitched_cost.extend(final_curve["cost"][1:])

    result = {
        "protocol": PROTOCOL,
        "purpose": "generic mr_bench BiRRT* with assembly-side shortcut-policy monkeypatch "
                   "(round 3: exact vectorized gate, adaptive stop, lazy certification)",
        "env": args.env, "policy": args.policy, "seed": args.seed,
        "env_seed": env_seed, "run_id": int(args.run_id),
        "environment_fingerprint": env_fingerprint,
        "rng_protocol": {
            "environment_seed": env_seed,
            "planner_shortcut_seed": int(args.seed),
            "shortcut_calls_isolated_from_planner_rng": True,
        },
        "planner": {
            "runtime_seconds": float(args.runtime), "optimize": bool(args.optimize),
            "stepsize": float(args.stepsize), "returned_path": True,
            "total_wall_seconds": float(total_wall),
            "info_costs": info_costs, "info_times": info_times,
            "info_curve_available": available,
            "planner_path_cost_at_final_shortcut": float(final_call["input"]["cost"]),
        },
        "patch": {"patched_modules": patch.patched_modules,
                  "shortcut_calls": patch.calls,
                  "portfolio_state": patch.portfolio_state},
        "output": _summary(env, out),
        "certificate": certificate,
        "full_curve": {"time": stitched_time, "cost": stitched_cost},
    }
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2, default=float)
    print(json.dumps({
        "env": args.env, "policy": args.policy, "seed": args.seed,
        "optimize": args.optimize, "calls": len(patch.calls),
        "final_cost": result["output"]["cost"], "wall": total_wall,
        "certificate": certificate,
    }, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
