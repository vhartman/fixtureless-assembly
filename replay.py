"""Replay a saved assembly plan WITHOUT re-searching.

A plan exported by `python main.py --solve ... --save plan.json` bundles the discrete plan
plus everything needed to rebuild the exact scene. This script reconstructs that scene and
plays the path back:

    python replay.py plan.json                 # step through in the rai viewer (blocking)
    python replay.py plan.json --viser         # scrubbable web view (viser)
    python replay.py plan.json --real-ik       # replay saved arm witnesses (zero IK solves)
    python replay.py plan.json --no-view       # rebuild + validate the replay, no window

A plan saved from a --real-ik run replays with its persisted witnesses by default. Replay never
silently resolves missing IK; ``--resolve-ik`` is the explicit one-time diagnostic escape hatch.
"""
import argparse
import numpy as np

from assembly import *
from env import make_four_robot_env
from plan_io import read_plan, read_witnesses
from config import saved_cfg as _saved


_WRENCH_PARAMS0 = None


def _restore_in_place(dst, src):
  """Deep-restore dst to src WITHOUT rebinding any dict: other modules hold references to
  wrench.PARAMS and its sub-dicts, so replacing them would leave those readers on stale copies."""
  import copy
  for k in [k for k in dst if k not in src]:
    del dst[k]
  for k, v in src.items():
    if isinstance(v, dict) and isinstance(dst.get(k), dict):
      _restore_in_place(dst[k], v)
    else:
      dst[k] = copy.deepcopy(v)


def apply_wrench_params(cfg):
  import copy
  import wrench
  # RESET TO THE PRISTINE DEFAULTS FIRST. This function mutates wrench.PARAMS IN PLACE and
  # CUMULATIVELY, so back-to-back scenes in one process contaminated each other two ways:
  # `tf_close_torque` is applied only when non-None, so a previous scene's value survived into
  # one that wanted the default, and `vacuum_scale` MULTIPLIES the vacuum caps, so two scaled
  # scenes scaled them twice. Wrench feasibility decides which grasps survive, which decides
  # the successor set -- the symptom was a later scene failing to reproduce its own plan's
  # actions. Measured: a cross8 sim run in-process made car-mini's plan unreproducible.
  global _WRENCH_PARAMS0
  if _WRENCH_PARAMS0 is None:
    _WRENCH_PARAMS0 = copy.deepcopy(wrench.PARAMS)
  else:
    _restore_in_place(wrench.PARAMS, _WRENCH_PARAMS0)
  wrench.PARAMS["grasp_model"] = _saved(cfg, "grasp_model")
  tm = cfg.get("table_mu")                                  # friction-cone table model
  if tm is not None:
    wrench.PARAMS["table_model"] = "friction"
    wrench.PARAMS["table_mu"] = float(tm)
  wrench.PARAMS["check_inserter"] = _saved(cfg, "inserter_check")
  wrench.PARAMS["insert_force"]["_default"] = _saved(cfg, "insert_force")
  wrench.PARAMS["curved_grasp_torque_scale"] = _saved(cfg, "curved_torque_scale")
  wrench.PARAMS["insertion_uncertainty"] = (
      None if _saved(cfg, "insertion_noise") is None
      else {"lateral": _saved(cfg, "insertion_noise"), "offset": _saved(cfg, "insertion_offset")})
  wrench.PARAMS["robust_mode"] = _saved(cfg, "robust_mode")
  # Vacuum sealing is a successor gate, so replay must restore it before walking the saved
  # action list. Missing in an old plan means the compatibility default (0 N).
  wrench.PARAMS["vacuum_engage_force"] = float(_saved(cfg, "vacuum_engage_force"))
  # The engagement gate's scoped static friction is the same successor-gate scope.
  wrench.PARAMS["vacuum_engage_friction"] = float(
      cfg.get("vacuum_engage_friction") or 0.0)
  # weightless stability -- without this a plan searched under --ignore-gravity is
  # unreproducible here (its successor set differs). .get(): older files predate the key.
  wrench.PARAMS["ignore_gravity"] = bool(cfg.get("ignore_gravity"))
  gc = cfg.get("grasp_compliance")
  wrench.PARAMS["grasp_compliance"] = (
      None if gc is None else {"two_finger": 1.0, "vacuum": float(gc)})
  vs = cfg.get("vacuum_scale")                              # scale every vacuum force+torque cap
  if vs is not None:
    for _model in wrench.PARAMS["grasp"].values():
      vac = _model.get("vacuum")
      if vac:
        for _k in vac:
          if _k != "mu":
            vac[_k] = vac[_k] * float(vs)
  ts = cfg.get("two_finger_scale")                          # scale every two-finger force+torque cap
  if ts is not None:
    for _model in wrench.PARAMS["grasp"].values():
      tf = _model.get("two_finger")
      if tf:
        for _k in tf:
          if _k != "mu":
            tf[_k] = tf[_k] * float(ts)
  tfc = cfg.get("tf_close_torque")                          # two-finger closing-axis torque cap
  if tfc is not None:
    wrench.PARAMS["grasp"]["asymmetric"]["two_finger"]["T_close"] = float(tfc)


def build_scene(cfg):
  # reconstruct the exact environment the plan was solved in (same seed => same scatter/perturb)
  set_seed(cfg["seed"])
  # Closure-aware versioned sampling changes the ranked, bounded grasp database. Restore it BEFORE
  # make_assembly/install_sampled_scene_ctx so a saved grasp id cannot silently rebind (and so
  # the database-hash check below compares like with like).
  from solver import set_grasp_closure_check
  set_grasp_closure_check(bool(_saved(cfg, "grasp_closure_check", legacy=False)))
  # mixed EEs: cfg["grippers"] = per-robot CSV (see main --grippers); rolls follow the
  # two-finger whenever any two-finger arm exists, exactly like the solve.
  ks = [k.strip() for k in (cfg.get("grippers") or "").split(",") if k.strip()]
  gripper_list = None
  if ks:
    ks = (ks + [ks[-1]] * cfg["num_robots"])[:cfg["num_robots"]]
    gripper_list = [k == "vacuum" for k in ks]
  any_tf = (gripper_list is not None and not all(gripper_list)) or \
      (gripper_list is None and not _saved(cfg, "vacuum"))
  set_rolls_per_face(2 if any_tf else 1)
  set_dense_grasps(bool(_saved(cfg, "dense_grasps")) and any_tf)
  set_inhand_reorient(bool(_saved(cfg, "inhand_reorient")))    # plan may contain reorient edges
  from problem import set_staged_mates, set_allow_curved_grasps, set_plain_box
  set_staged_mates(bool(_saved(cfg, "staged_mates")))          # staged-mate invariant, as searched
  set_allow_curved_grasps(bool(_saved(cfg, "allow_curved_grasps")))  # grasp table must match the search
  from problem import set_grasp_translations
  set_grasp_translations(int(_saved(cfg, "grasp_translations")))  # grasp IDS index this DB:
  # a replay with a different count would read every saved grasp id against a different table.
  from problem import set_banned_actions
  set_banned_actions(())                                   # search-only; never active in replays
  from problem import set_kind_forbid
  set_kind_forbid(())                                      # search-only; never active in replays
  from problem import set_sampled_grasps
  set_sampled_grasps({"seed": int(_saved(cfg, "grasp_sampler_seed")),
                      "candidates": int(_saved(cfg, "grasp_sampler_candidates")),
                      "keep": int(_saved(cfg, "grasp_sampler_keep")),
                      "master_keep": (int(_saved(cfg, "grasp_widen_max"))
                                      if _saved(cfg, "grasp_local_widen", legacy=False)
                                      else int(_saved(cfg, "grasp_sampler_keep"))),
                      # LEGACY-FAITHFUL: a plan saved BEFORE the edge margin existed carries
                      # no key and was built with NO margin -- reconstructing it at today's
                      # DEFAULT would rebind every saved grasp id. Missing => 0.0, not 0.008.
                      "margin": float(_saved(cfg, "grasp_sampler_margin", legacy=0.0)),
                      # pad_support's default never changed, so a missing key means today's
                      # default (0.75) -- the historical `or 0.0` here was wrong.
                      "pad_support": float(_saved(cfg, "grasp_sampler_pad_support")),
                      # None = the legacy support metric; a value switches to the
                      # curvature-fair wrap metric (rebinds ids -> saved with the plan).
                      "wrap": _saved(cfg, "grasp_sampler_wrap")}
                     if _saved(cfg, "grasp_sampler") else None)  # sampled ids: same hazard class
  set_plain_box(bool(_saved(cfg, "plain_box")))                # cube collision geometry must match
  from problem import set_placement_yaw_prefix, set_placement_yaws, set_stable_rest_poses
  # PLACEMENT ids index the per-group table, so a replay must fill it the same way the
  # search did. Missing key => False, which is what every plan saved before the flag used.
  set_stable_rest_poses(bool(_saved(cfg, "stable_rest_poses")))
  set_placement_yaws(int(_saved(cfg, "placement_yaws")))
  set_placement_yaw_prefix(None)                            # scheduling never restricts replay
  from env import set_tight_fingers
  # LEGACY-FAITHFUL: tight fingers became the DEFAULT on 2026-07-29, so a plan saved before
  # the flag was persisted was certified with the FAT proxy. Missing => False, not True.
  set_tight_fingers(bool(_saved(cfg, "tight_fingers", legacy=False)))
  # -- set unconditionally (not under an `if`): a stale global from a previous scene in the same
  # process would rebuild this one with the wrong finger proxy (FRAGILITY 1.10).
  from problem import set_dominance, set_useful_grasps_enabled
  set_dominance(bool(cfg.get("dominance")))                # successor set must match the search
  set_useful_grasps_enabled(bool(cfg.get("useful_grasps")))  # ordering must match the search
  from problem import set_brace_resting
  set_brace_resting(bool(cfg.get("brace_resting")))         # action space must match the search
  # RESET THE REST OF THE SUCCESSOR-AFFECTING GLOBALS TOO, unconditionally. These are module
  # state in problem.py, so whatever ran in this process BEFORE us leaves them set -- a single
  # in-process main() for another assembly was enough to change the grasp table here and make
  # the plan's own actions unreproducible ("cannot reproduce action ... from the successor
  # set"). Absent keys fall back to the parser defaults, which is what a plan saved before
  # these were persisted was solved with.
  from problem import (set_brace_for_assembly_only, set_extra_place_cells, set_prefer_center,
                       set_anonymous_robots, set_lru_robots)
  set_brace_for_assembly_only(bool(_saved(cfg, "brace_for_assembly_only")))
  set_extra_place_cells(int(_saved(cfg, "place_cells")))
  set_prefer_center(bool(_saved(cfg, "prefer_center")))
  set_anonymous_robots(bool(_saved(cfg, "anon_robots")))
  set_lru_robots(bool(_saved(cfg, "lru_robots")))
  # and drop the cross-scene CACHES: they outlive a scene and are not keyed on every global
  # that feeds them, so a previous scene's entries would be served to this one.
  from problem import reset_wrench_cache, reset_grasp_pose_cache
  reset_wrench_cache()
  reset_grasp_pose_cache()
  from problem import clear_sampled_scene_ctx
  clear_sampled_scene_ctx()                 # a previous scene's filter must not leak into
  # make_assembly below (define() queries the grasp DB, which raises under a stale ctx).
  apply_wrench_params(cfg)
  if (_saved(cfg, "robot_types") or _saved(cfg, "robot")) == "fr3":
    from env import make_fr3_duo_env
    C = make_fr3_duo_env(num_duos=max(1, cfg["num_robots"] // 2))
  elif (_saved(cfg, "robot_types") or _saved(cfg, "robot")) == "openarm":
    from env import make_openarm_duo_env, make_openarm_floating_env
    C = (make_openarm_floating_env(num_robots=cfg["num_robots"])
         if _saved(cfg, "floating_ee") else
         make_openarm_duo_env(num_duos=max(1, cfg["num_robots"] // 2)))
  else:
    C = make_four_robot_env(num_robots=cfg["num_robots"], floating_ee=cfg["floating_ee"],
                            vacuum=gripper_list if gripper_list is not None else _saved(cfg, "vacuum"),
                            robot_types=_saved(cfg, "robot_types") or _saved(cfg, "robot"))
  assembly = make_assembly(cfg["assembly"], cfg["parts"])
  assembly.configure_screw_skills(
      angle=(None if _saved(cfg, "screw_angle") is None
             else np.deg2rad(float(_saved(cfg, "screw_angle")))),
      depth=_saved(cfg, "screw_depth"), torque=_saved(cfg, "screw_torque"),
      angular_speed=_saved(cfg, "screw_speed"), samples=_saved(cfg, "screw_samples"))
  # min_sep/grid/grid_gap MUST be passed too: they shape the initial scatter, so omitting them
  # rebuilt a different layout than the plan was solved in (FRAGILITY 1.8). Absent keys keep the
  # parser defaults, which is what a plan saved before they were persisted was solved with.
  C = add_disassembled_obj(C, assembly, perturb_start_pose=cfg["perturb_start_pose"],
                           spacing=cfg["scatter_spacing"],
                           random_init=_saved(cfg, "random_init"),
                           # LEGACY-FAITHFUL: the historical float was 0.015; the 0.002 default
                           # came later. Missing => 0.015, what pre-flag plans were solved with.
                           float_height=_saved(cfg, "part_float", legacy=0.015),
                           min_sep=_saved(cfg, "min_sep"), grid=_saved(cfg, "grid_init"),
                           grid_gap=_saved(cfg, "grid_gap"),
                           # The standard profile now centres the scatter lattice.  This is
                           # scene geometry, not merely a search preference: omitting it puts
                           # every part at a different world pose during replay.  Plans that
                           # predate the persisted field retain the historical left-anchored
                           # layout rather than inheriting today's profile default.
                           center_scatter=bool(_saved(cfg, "center_scatter", legacy=False)))
  if _saved(cfg, "grasp_sampler_scenes", legacy=False):
    # the scene-filtered DB is per PART INSTANCE and its ids are what the plan indexes: the
    # replay scene must install the same filter context or every saved grasp id misreads.
    from problem import install_sampled_scene_ctx, get_robots as _get_robots
    install_sampled_scene_ctx(C, assembly, robots=_get_robots(C))
  _expected_db = cfg.get("grasp_sampler_db_hash")
  if _expected_db:
    from problem import sampled_grasp_db_signature
    _actual_db = sampled_grasp_db_signature(assembly)
    if _actual_db != _expected_db:
      raise ValueError(
          "saved sampled-grasp database hash does not match this rebuild "
          f"({_expected_db} != {_actual_db}); refusing to reinterpret positional grasp ids")
  return C, assembly


def replay(path, viser=False, view=True, real_ik=False, viser_port=8080, serve=True,
           resolve_ik=False):
  """Rebuild the saved scene and replay the plan. Returns the ViserRecorder (or None) so a
  headless caller can inspect the recorded steps without serving."""
  cfg, plan = read_plan(path)
  C, assembly = build_scene(cfg)
  robots = get_robots(C)
  use_ik = bool(cfg.get("real_ik")) or real_ik
  print(f"[replay] {cfg['assembly']} ({cfg['parts']} parts), {cfg['num_robots']} robots, "
        f"{len(plan)} actions, {'persisted arm-keyframe' if use_ik else 'geometric'} replay")
  # recorder=None => the built-in BLOCKING rai viewer (the default interactive replay). For
  # --viser we record then serve; for --no-view (headless validation) we record into a
  # throwaway recorder so the replay runs without opening any window.
  recorder = None
  if viser or not view:
    from viser_view import ViserRecorder
    recorder = ViserRecorder()
  brace = bool(_saved(cfg, "bracing", legacy=False))         # match the search that made the plan
  if use_ik:
    rik = RobotIK(C, robots)
    rik.closure_check = bool(_saved(cfg, "grasp_closure_check", legacy=False))
    rik.pre_insertion_delta = float(_saved(cfg, "pre_insertion", legacy=0.0))  # pre-skill split (export/replay)
    # These funnels now default ON for fresh runs.  A missing key in an old plan means they
    # were OFF; silently adding split tasks would invalidate every saved task id.
    rik.pre_pick_delta = float(_saved(cfg, "pre_pick", legacy=0.0))
    rik.pre_place_delta = float(_saved(cfg, "pre_place", legacy=0.0))
    rik.pre_add_grasp_delta = float(_saved(cfg, "pre_add_grasp", legacy=0.0))
    rik.post_release_delta = float(_saved(cfg, "post_release", 0.0))
    if _saved(cfg, "witness_robot_collisions", legacy=False):  # replay IK sees the same scene
      rik.strip_inactive_collision = False
    _sc = _saved(cfg, "strict_collisions", legacy=None)      # same witness gate as the search
    if _sc is not None:
      rik.penetration_reject = float(_sc)
      rik.col_weight = 1e2
    # --always-contact: the search never disabled touched contacts; the replay must not either.
    rik.keep_contacts = bool(_saved(cfg, "always_contact", legacy=False))
    witnesses = read_witnesses(path, list(rik.C.getJointNames()), len(plan))
    needs_ik = {"pick", "place", "stow", "assemble", "add_grasp", "handover"}
    missing = ([] if witnesses is None else
               [i for i, (a, w) in enumerate(zip(plan, witnesses))
                if a[0] in needs_ik and w is None])
    if (witnesses is None or missing) and not resolve_ik:
      raise SystemExit(
          "real-arm replay has no compatible persisted witness"
          + (" block" if witnesses is None else f" for required plan step(s) {missing}")
          + "; refusing to resolve IK implicitly. Re-save the plan with witnesses, or pass "
            "--resolve-ik for an explicit one-time diagnostic solve.")
    if witnesses is None or missing:
      print("[replay] --resolve-ik explicitly requested: resolving per-action IK because the "
            "plan has missing/incompatible persisted witnesses.")
    else:
      print(f"[replay] {sum(w is not None for w in witnesses)} persisted witnesses loaded; "
            "per-action IK solving SKIPPED.")
    visualize_plan_ik(C, assembly, plan, robots=robots, robot_ik=rik,
                      recorder=recorder, bracing=brace, witnesses=witnesses)
  else:
    visualize_plan(C, assembly, plan, robots=robots, recorder=recorder, bracing=brace)
  if viser and serve:
    recorder.serve(port=viser_port)
  return recorder


def parse_args():
  p = argparse.ArgumentParser(description="Replay a saved assembly plan (see main.py --save).")
  p.add_argument("path", help="Plan JSON written by main.py --save.")
  p.add_argument("--viser", action="store_true", help="Scrubbable web view instead of the "
                 "blocking rai viewer.")
  p.add_argument("--viser-port", type=int, default=8080, help="Port for --viser (default 8080).")
  p.add_argument("--real-ik", action="store_true", help="Replay real-arm keyframes (a plan "
                 "saved from --real-ik already enables this). Persisted witnesses are required "
                 "and no IK is solved unless --resolve-ik is also passed.")
  p.add_argument("--resolve-ik", action="store_true",
                 help="Explicitly allow per-action IK solving when no persisted witnesses are "
                      "available. Replay otherwise fails instead of silently re-solving.")
  p.add_argument("--no-view", dest="view", action="store_false",
                 help="Rebuild + run the replay without opening a window (validation only).")
  return p.parse_args()


if __name__ == "__main__":
  a = parse_args()
  replay(a.path, viser=a.viser, view=a.view, real_ik=a.real_ik,
         viser_port=a.viser_port, resolve_ik=a.resolve_ik)
