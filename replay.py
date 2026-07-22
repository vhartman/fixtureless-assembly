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


def apply_wrench_params(cfg):
  import wrench
  wrench.PARAMS["grasp_model"] = cfg["grasp_model"]
  wrench.PARAMS["check_inserter"] = cfg["inserter_check"]
  wrench.PARAMS["insert_force"]["_default"] = cfg["insert_force"]
  wrench.PARAMS["curved_grasp_torque_scale"] = cfg["curved_torque_scale"]
  wrench.PARAMS["insertion_uncertainty"] = (
      None if cfg["insertion_noise"] is None
      else {"lateral": cfg["insertion_noise"], "offset": cfg["insertion_offset"]})
  wrench.PARAMS["robust_mode"] = cfg.get("robust_mode", "corners")


def build_scene(cfg):
  # reconstruct the exact environment the plan was solved in (same seed => same scatter/perturb)
  set_seed(cfg["seed"])
  set_rolls_per_face(1 if cfg["vacuum"] else 2)
  set_dense_grasps(bool(cfg.get("dense_grasps")) and not cfg["vacuum"])
  set_inhand_reorient(bool(cfg.get("inhand_reorient")))    # plan may contain reorient edges
  from problem import set_staged_mates, set_allow_curved_grasps
  set_staged_mates(bool(cfg.get("staged_mates")))          # staged-mate invariant, as searched
  set_allow_curved_grasps(bool(cfg.get("allow_curved_grasps")))  # grasp table must match the search
  apply_wrench_params(cfg)
  C = make_four_robot_env(num_robots=cfg["num_robots"], floating_ee=cfg["floating_ee"],
                          vacuum=cfg["vacuum"],
                          robot_types=cfg.get("robot_types") or cfg.get("robot", "ur10"))
  assembly = make_assembly(cfg["assembly"], cfg["parts"])
  C = add_disassembled_obj(C, assembly, perturb_start_pose=cfg["perturb_start_pose"],
                           spacing=cfg["scatter_spacing"],
                           random_init=cfg.get("random_init"),
                           float_height=cfg.get("part_float", 0.015))
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
