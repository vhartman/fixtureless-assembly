"""Calibrated Husky participates in the ordinary assembly-scene pipeline."""

import numpy as np

import assembly as A
import analytic_ik
from env import is_husky_scene, make_husky_duo_env
from config import RunConfig
from problem import (add_disassembled_obj, get_initial_state, make_assembly, places,
                     set_extra_place_cells)
from replay import build_scene
from solver import RobotIK, _quat_to_R


def _matrix(frame):
  pose = np.asarray(frame.getPose(), dtype=float)
  T = np.eye(4)
  T[:3, 3] = pose[:3]
  T[:3, :3] = _quat_to_R(pose[3:])
  return T


def test_husky_exposes_two_normal_robot_subtrees_and_valid_home():
  C = make_husky_duo_env()

  assert A.get_robots(C) == ["a1", "a2"]
  assert is_husky_scene(C)
  table = C.getFrame("table")
  assert np.isclose(table.getPosition()[2] + table.getSize()[2] / 2.0, 0.54)
  assert len(C.getJointState()) == 12
  assert all(n.startswith(("a1_", "a2_")) for n in C.getJointNames())
  for r in A.get_robots(C):
    for suffix in ("ur_base_link", "ur_ee_link", "ur_gripper_center",
                   "ur_arm_ur5", f"husky_{'right' if r == 'a1' else 'left'}_arm"):
      assert C.getFrame(f"{r}_{suffix}") is not None

  lo, hi = (np.asarray(x, dtype=float) for x in C.getJointLimits())
  q = np.asarray(C.getJointState(), dtype=float)
  assert np.min(np.minimum(q - lo, hi - q)) >= 1e-3
  assert C.getCollisionsTotalPenetration() < 1e-6


def test_husky_peg_hole_is_built_by_the_shared_scenario_code():
  C = make_husky_duo_env()
  assembly = make_assembly("peg-hole", 2)
  C = add_disassembled_obj(C, assembly, float_height=0.002, center_scatter=True)

  assert all(C.getFrame(p) is not None for p in assembly.objects)
  assert C.getFrame("obj_0__shape0") is not None       # compound concave socket
  assert all(C.getFrame(p).getParent().name == "table" for p in assembly.objects)
  assert C.getCollisionsTotalPenetration() < 1e-6


def test_placement_cells_are_relative_to_the_offset_husky_table():
  """Cell zero means table centre, not world origin (which lies on the carrier body)."""
  C = make_husky_duo_env()
  assembly = make_assembly("peg-hole", 2)
  C = add_disassembled_obj(C, assembly, float_height=0.002, center_scatter=True)
  state = get_initial_state(C, assembly)
  part = "obj_0"
  state.parent[part] = ("gripper", "a1")
  state.grasp[part] = 0
  state.supports[part] = frozenset({("grasp", "a1", part, 0)})

  set_extra_place_cells(1)
  try:
    cell = next(ns for action, ns in places(state, assembly, ["a1", "a2"])
                if len(action) == 5 and action[-1] == 0)
  finally:
    set_extra_place_cells(0)

  np.testing.assert_allclose(cell.poses[part][:2], C.getFrame("table").getPosition()[:2])


def test_husky_robotik_uses_real_calibrated_roots_without_wrong_ssik_model():
  C = make_husky_duo_env()
  rik = RobotIK(C, A.get_robots(C))
  rik.ik_seeding = True

  assert rik.robot_arm == {"a1": "husky_right", "a2": "husky_left"}
  assert rik.robot_base == {"a1": "a1_ur_base_link", "a2": "a2_ur_base_link"}
  assert rik._analytic_reject_ok("a1")
  assert rik._analytic_reject_ok("a2")


def test_husky_side_specific_ssik_matches_real_rai_fk_and_inverse():
  C = make_husky_duo_env()
  joint_names = list(C.getJointNames())
  lo, hi = (np.asarray(x, dtype=float) for x in C.getJointLimits())
  rng = np.random.default_rng(90210)

  for robot, arm in (("a1", "husky_right"), ("a2", "husky_left")):
    idx = [i for i, n in enumerate(joint_names) if n.startswith(robot + "_")]
    limits = np.vstack([lo[idx], hi[idx]])
    cal = analytic_ik._CAL[arm]
    solver = analytic_ik._solver(arm)
    for _ in range(4):
      q = rng.uniform(limits[0] + 0.05, limits[1] - 0.05)
      full = np.asarray(C.getJointState(), dtype=float)
      full[idx] = q
      C.setJointState(full)
      target = np.linalg.inv(_matrix(C.getFrame(robot + "_ur_base_link"))) @ \
          _matrix(C.getFrame(robot + "_ur_gripper_center"))

      predicted = np.asarray(solver.fk(q), dtype=float) @ cal["tools"]["two_finger"]
      assert np.allclose(predicted, target, atol=2e-8)
      seeds = analytic_ik.seed_configs(
          arm, target, tool_kind="two_finger", joint_limits=limits)
      assert seeds
      assert min(np.max(np.abs(np.asarray(solver.fk(seed)) @
                               cal["tools"]["two_finger"] - target))
                 for seed in seeds) < 2e-8


def test_husky_saved_scenario_rebuilds_through_replay():
  cfg = RunConfig(assembly="peg-hole", parts=2, robot="husky", num_robots=2,
                  center_scatter=True, part_float=0.002).to_dict()
  C, assembly = build_scene(cfg)

  assert is_husky_scene(C)
  assert A.get_robots(C) == ["a1", "a2"]
  assert assembly.objects == ["obj_0", "obj_1"]
  assert C.getCollisionsTotalPenetration() < 1e-6
