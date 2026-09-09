"""View a SAVED BiRRT motion plan in viser WITHOUT re-planning.

Usage: python mp_viser_saved.py <plan.json> <mp-path.json> [port]
       python mp_viser_saved.py <plan.json> <mp-path.json> [port] --mp-export export.json

Rebuilds the problem from the saved plan, restores mode-entry configurations from the saved
path, and plays it through mr_bench's display_path_viser. This is visual replay only; sim.py
keeps the stricter exact-signature requirement for physics execution.
"""
import argparse

import numpy as np

import assembly as A
import mp_export as MP
import replay
from plan_io import read_plan
from solver import RobotIK


def main():
    ap = argparse.ArgumentParser(description="View a saved MP path without resolving any IK")
    ap.add_argument("path", nargs="?", default="car-mini-3r-plan.json")
    ap.add_argument("mp_path", nargs="?", default="mp-car-mini-3r.json")
    ap.add_argument("port", nargs="?", type=int, default=8080)
    ap.add_argument("--mp-export", default=None,
                    help="standalone final export from --mp-export-save; new --mp-save files "
                         "embed this block and need no extra argument")
    args = ap.parse_args()
    path, mpf, port = args.path, args.mp_path, args.port

    cfg, plan = read_plan(path)
    plan = [tuple(a) for a in plan]
    C, asm = replay.build_scene(cfg)
    robots = A.get_robots(C)
    gk = A.wrench_kind_arg(C, robots) if cfg.get("wrench") else None

    rik = RobotIK(C, robots)
    rik.ik_seeding = bool(cfg.get("ik_seeding", True))
    export_path = args.mp_export or mpf
    saved = MP.load_export_bundle(
        export_path, plan, list(rik.C.getJointNames()), required=False)
    legacy = saved is None
    if legacy:
        # PRE-EXPORT mp file: the path predates the embedded final export.  Visual replay
        # can still use the SOLVED witnesses persisted in the plan file; only the pre-*
        # keyframes the exporter fused later are missing, and make_problem re-solves those
        # (IK only, no BiRRT).  A strictly zero-IK replay still needs a bundle: make one
        # once with --mp-export-save or pass --mp-export <bundle.json>.
        from plan_io import read_witnesses
        wit = read_witnesses(path, list(rik.C.getJointNames()), len(plan))
        if not wit:
            raise SystemExit(
                f"{path}: no persisted witnesses and {mpf} has no export block; "
                "nothing to view without solving")
        print(f"[viser] LEGACY visual replay: {export_path} has no embedded export; using "
              "the plan's solved witnesses and re-solving only missing pre-keyframes "
              "(no BiRRT). For a strict zero-IK replay, make an export bundle first "
              "(--mp-export-save).")
        pre, settings, witness_sets = {}, {}, None
        tasks = step_names = sequence_names = None
    else:
        wit, pre = saved["witnesses"], saved["pre_witnesses"]
        settings = saved["settings"]
        witness_sets = saved["witness_sets"]
        tasks = saved["tasks"]
        step_names = saved["step_names"]
        sequence_names = saved["sequence_names"]
    rik.pre_insertion_delta = float(settings.get("pre_insertion",
                                                 cfg.get("pre_insertion", 0.0)))
    rik.pre_pick_delta = float(settings.get("pre_pick", cfg.get("pre_pick", 0.0)))
    rik.post_pick_delta = float(settings.get("post_pick", cfg.get("post_pick", 0.0)))
    rik.pre_place_delta = float(settings.get("pre_place", cfg.get("pre_place", 0.0)))
    rik.pre_add_grasp_delta = float(settings.get("pre_add_grasp",
                                                 cfg.get("pre_add_grasp", 0.0)))
    rik.post_release_delta = float(settings.get("post_release",
                                                cfg.get("post_release", 0.0)))
    brace = bool(settings.get("bracing", cfg.get("bracing", True)))
    prob = MP.make_problem(
        C, asm, plan, wit, robots, rik, gk, bracing=brace,
        depgraph=bool(settings.get("depgraph", not cfg.get("mp_sequence", False))),
        collision_tolerance=float(settings.get("collision_tolerance",
                                               cfg.get("mp_collision_tolerance", 1e-4))),
        pre_witnesses=pre,
        robot_robot_margin=float(settings.get("robot_robot_margin",
                                              cfg.get("mp_robot_margin", 0.0))),
        robot_margin_ramp=float(settings.get("robot_margin_ramp",
                                             cfg.get("mp_margin_ramp", 0.0))),
        table_margin=float(settings.get("table_margin",
                                        cfg.get("mp_table_margin", 0.0))),
        table_float=float(settings.get("table_float", cfg.get("part_float", 0.002))),
        witness_sets=witness_sets, resolve_missing_pre=legacy,
        task_records=tasks, saved_step_names=step_names,
        saved_sequence_names=sequence_names)

    have_sig = MP.task_signature(prob)
    if not legacy and saved["task_signature"] is not None \
            and have_sig != saved["task_signature"]:
        raise SystemExit("saved final export no longer rebuilds the same task signature; "
                         "refusing to replay against changed task semantics")
    print(f"[viser] {len(prob.tasks)} tasks rebuilt "
          f"({'legacy: plan witnesses + IK-only pre-keyframes' if legacy else 'from persisted final export ' + export_path}); "
          "IK solves: 0" if not legacy else "")

    mp = MP.load_path(mpf)
    print(f"[viser] loaded {len(mp)} states from {mpf}")
    if not mp:
        raise SystemExit("saved motion path contains no states")

    # A saved mode contains only task ids.  Current mr_bench Mode objects also need
    # the configuration at which the mode was entered and a prev_mode chain.  The
    # viser viewer follows that chain to reconstruct grasp/release/assembly effects.
    want_sig = getattr(MP.load_path, "last_signature", None)
    if want_sig is not None and have_sig != want_sig:
        # Execution must reject this (sim.py does), but this tool is a read-only
        # viewer.  Older paths did not persist their task manifest, so after a
        # harmless task-robot-list change the exact old signature cannot be rebuilt.
        # Continue only after the ordered mode chain below proves compatible.
        print("[viser] WARNING: task signature changed since this path was saved "
              f"(path={want_sig}, rebuilt={have_sig}); visual replay will use the "
              "supplied persisted-export task effects. Physics execution should still "
              "refuse this pair.")
    live_modes = MP.mode_sequence(prob)
    mp, n_repaired = MP.repair_mode_chain(
        mp, live_modes,
        lambda ref, mode: MP._PathState(
            ref.q, MP._Mode([int(t) for t in np.atleast_1d(mode.task_ids)])))
    if n_repaired:
        print(f"[viser] restored {n_repaired} zero-motion mode(s) omitted by shortcutting")

    expected_ids = [tuple(int(t) for t in np.atleast_1d(m.task_ids)) for m in live_modes]
    saved_ids = []
    for s in mp:
        key = tuple(int(t) for t in np.atleast_1d(s.mode.task_ids))
        if not saved_ids or key != saved_ids[-1]:
            saved_ids.append(key)
    if saved_ids != expected_ids:
        raise SystemExit(
            "saved path mode chain does not match the rebuilt plan; refusing even visual "
            f"replay (saved {len(saved_ids)} modes, rebuilt {len(expected_ids)})")

    # load_path returns lightweight (_PathState, ndarray q) records; mr_bench's viewer
    # pipeline needs proper State objects whose .q is a Configuration (interpolate_path
    # calls config_dist -> type(q)._dist). Rebuild them with the live problem's
    # configuration layout so q[k] remains one robot's joints, not the whole flat vector.
    from multi_robot_multi_goal_planning.problems.planning_env import Mode as _MBMode
    from multi_robot_multi_goal_planning.problems.planning_env import State as _MBState
    q_template = prob.get_start_pos()
    q_shape = q_template.state().shape
    states = []
    current_ids = None
    current_mode = None
    previous_mode = None
    for s in mp:
        q_flat = np.asarray(s.q, dtype=float).ravel()
        if q_flat.shape != q_shape:
            raise SystemExit(
                f"saved configuration has shape {q_flat.shape}, expected {q_shape}")
        mode_ids = tuple(int(t) for t in np.atleast_1d(s.mode.task_ids))
        q = q_template.from_flat(q_flat)
        if mode_ids != current_ids:
            # Use the SAVED transition configuration, not the freshly rebuilt
            # goal witness.  This preserves the exact attachment pose of the path.
            current_mode = _MBMode(list(mode_ids), q)
            current_mode.prev_mode = previous_mode
            if previous_mode is not None:
                previous_mode.next_modes.append(current_mode)
            previous_mode = current_mode
            current_ids = mode_ids
        states.append(_MBState(q, current_mode))

    from multi_robot_multi_goal_planning.problems.util import interpolate_path
    disp = interpolate_path(states, 0.05, kind="euclidean")
    print(f"[viser] serving interpolated path ({len(disp)} states) on "
          f"http://localhost:{port} (Ctrl-C to stop)")
    prob.display_path_viser(disp, port=port, primitives_only=False)


if __name__ == "__main__":
    main()
