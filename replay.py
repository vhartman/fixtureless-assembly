"""Replay a saved assembly plan WITHOUT re-searching.

A plan exported by `python main.py --solve ... --save plan.json` bundles the discrete plan
plus everything needed to rebuild the exact scene. This script reconstructs that scene and
plays the path back:

    python replay.py plan.json                 # step through in the rai viewer (blocking)
    python replay.py plan.json --viser         # scrubbable web view (viser)
    python replay.py plan.json --real-ik       # replay with arm IK (re-solves per action)
    python replay.py plan.json --no-view       # rebuild + validate the replay, no window

A plan saved from a --real-ik run replays with IK by default; pass --real-ik to force IK on a
geometric plan. IK replay re-runs the per-action IK (deterministic under the saved seed) --
only the expensive SEARCH is skipped, not the IK.
"""
import argparse

from assembly import *
from env import make_four_robot_env
from plan_io import read_plan


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
  wrench.PARAMS["grasp_model"] = cfg["grasp_model"]
  wrench.PARAMS["check_inserter"] = cfg["inserter_check"]
  wrench.PARAMS["insert_force"]["_default"] = cfg["insert_force"]
  wrench.PARAMS["curved_grasp_torque_scale"] = cfg["curved_torque_scale"]
  wrench.PARAMS["insertion_uncertainty"] = (
      None if cfg["insertion_noise"] is None
      else {"lateral": cfg["insertion_noise"], "offset": cfg["insertion_offset"]})
  wrench.PARAMS["robust_mode"] = cfg.get("robust_mode", "corners")
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
  tfc = cfg.get("tf_close_torque")                          # two-finger closing-axis torque cap
  if tfc is not None:
    wrench.PARAMS["grasp"]["asymmetric"]["two_finger"]["T_close"] = float(tfc)


def build_scene(cfg):
  # reconstruct the exact environment the plan was solved in (same seed => same scatter/perturb)
  set_seed(cfg["seed"])
  # mixed EEs: cfg["grippers"] = per-robot CSV (see main --grippers); rolls follow the
  # two-finger whenever any two-finger arm exists, exactly like the solve.
  ks = [k.strip() for k in (cfg.get("grippers") or "").split(",") if k.strip()]
  gripper_list = None
  if ks:
    ks = (ks + [ks[-1]] * cfg["num_robots"])[:cfg["num_robots"]]
    gripper_list = [k == "vacuum" for k in ks]
  any_tf = (gripper_list is not None and not all(gripper_list)) or \
      (gripper_list is None and not cfg["vacuum"])
  set_rolls_per_face(2 if any_tf else 1)
  set_dense_grasps(bool(cfg.get("dense_grasps")) and any_tf)
  set_inhand_reorient(bool(cfg.get("inhand_reorient")))    # plan may contain reorient edges
  from problem import set_staged_mates, set_allow_curved_grasps, set_plain_box
  set_staged_mates(bool(cfg.get("staged_mates")))          # staged-mate invariant, as searched
  set_allow_curved_grasps(bool(cfg.get("allow_curved_grasps")))  # grasp table must match the search
  from problem import set_grasp_translations
  set_grasp_translations(int(cfg.get("grasp_translations") or 0))  # grasp IDS index this DB:
  # a replay with a different count would read every saved grasp id against a different table.
  set_plain_box(bool(cfg.get("plain_box")))                # cube collision geometry must match
  from env import set_tight_fingers
  set_tight_fingers(bool(cfg.get("tight_fingers")))        # GRIPPER collision geometry likewise
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
  set_brace_for_assembly_only(bool(cfg.get("brace_for_assembly_only")))
  set_extra_place_cells(int(cfg.get("place_cells") or 0))
  set_prefer_center(bool(cfg.get("prefer_center")))
  set_anonymous_robots(bool(cfg.get("anon_robots")))
  set_lru_robots(bool(cfg.get("lru_robots")))
  # and drop the cross-scene CACHES: they outlive a scene and are not keyed on every global
  # that feeds them, so a previous scene's entries would be served to this one.
  from problem import reset_wrench_cache, reset_grasp_pose_cache
  reset_wrench_cache()
  reset_grasp_pose_cache()
  apply_wrench_params(cfg)
  C = make_four_robot_env(num_robots=cfg["num_robots"], floating_ee=cfg["floating_ee"],
                          vacuum=gripper_list if gripper_list is not None else cfg["vacuum"],
                          robot_types=cfg.get("robot_types") or cfg.get("robot", "ur10"))
  assembly = make_assembly(cfg["assembly"], cfg["parts"])
  # min_sep/grid/grid_gap MUST be passed too: they shape the initial scatter, so omitting them
  # rebuilt a different layout than the plan was solved in (FRAGILITY 1.8). Absent keys keep the
  # parser defaults, which is what a plan saved before they were persisted was solved with.
  C = add_disassembled_obj(C, assembly, perturb_start_pose=cfg["perturb_start_pose"],
                           spacing=cfg["scatter_spacing"],
                           random_init=cfg.get("random_init"),
                           float_height=cfg.get("part_float", 0.015),
                           min_sep=cfg.get("min_sep"), grid=cfg.get("grid_init"),
                           grid_gap=cfg.get("grid_gap", 0.1))
  return C, assembly


def replay(path, viser=False, view=True, real_ik=False, viser_port=8080, serve=True):
  """Rebuild the saved scene and replay the plan. Returns the ViserRecorder (or None) so a
  headless caller can inspect the recorded steps without serving."""
  cfg, plan = read_plan(path)
  C, assembly = build_scene(cfg)
  robots = get_robots(C)
  use_ik = bool(cfg.get("real_ik")) or real_ik
  print(f"[replay] {cfg['assembly']} ({cfg['parts']} parts), {cfg['num_robots']} robots, "
        f"{len(plan)} actions, {'IK' if use_ik else 'geometric'} replay")
  # recorder=None => the built-in BLOCKING rai viewer (the default interactive replay). For
  # --viser we record then serve; for --no-view (headless validation) we record into a
  # throwaway recorder so the replay runs without opening any window.
  recorder = None
  if viser or not view:
    from viser_view import ViserRecorder
    recorder = ViserRecorder()
  brace = bool(cfg.get("bracing"))                           # match the search that made the plan
  if use_ik:
    rik = RobotIK(C, robots)
    rik.pre_insertion_delta = float(cfg.get("pre_insertion") or 0.0)  # pre-skill split config (export/replay)
    rik.pre_pick_delta = float(cfg.get("pre_pick") or 0.0)
    rik.pre_place_delta = float(cfg.get("pre_place") or 0.0)
    rik.pre_add_grasp_delta = float(cfg.get("pre_add_grasp") or 0.0)
    if cfg.get("witness_robot_collisions"):                # replay IK sees the same scene
      rik.strip_inactive_collision = False
    if cfg.get("strict_collisions") is not None:           # same witness gate as the search
      rik.penetration_reject = float(cfg["strict_collisions"])
      rik.col_weight = 1e2
    visualize_plan_ik(C, assembly, plan, robots=robots, robot_ik=rik,
                      recorder=recorder, bracing=brace)
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
  p.add_argument("--real-ik", action="store_true", help="Replay with arm IK (re-runs per-action "
                 "IK; a plan saved from --real-ik already replays with IK).")
  p.add_argument("--no-view", dest="view", action="store_false",
                 help="Rebuild + run the replay without opening a window (validation only).")
  return p.parse_args()


if __name__ == "__main__":
  a = parse_args()
  replay(a.path, viser=a.viser, view=a.view, real_ik=a.real_ik, viser_port=a.viser_port)
