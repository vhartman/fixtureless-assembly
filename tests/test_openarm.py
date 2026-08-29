"""Focused regressions for the OpenArm duo integration.

These deliberately avoid a full assembly search: they guard the model/search boundary that
made OpenArm grasps look feasible with the old Robotiq proxy and then fail in real-arm IK.
"""
from pathlib import Path

import numpy as np
import pytest
import robotic as ry

import assembly as A
import analytic_ik
import solver as S
import visualization as V
from env import (OPENARM_FLOATING_EE_PATH, is_openarm_scene, make_openarm_duo_env,
                 make_openarm_floating_env, robot_exit_configuration,
                 set_openarm_real_finger_radius_scale)


ROOT = Path(__file__).resolve().parents[1]


def _restricted_copy(C, rik, robot):
  Cr = ry.Config()
  Cr.addConfigurationCopy(C)
  rik._restrict_to_arm(Cr, robot)
  return Cr


def test_openarm_two_duos_start_collision_free_and_inside_physical_limits():
  C = make_openarm_duo_env(2)
  q = np.asarray(C.getJointState(), dtype=float)
  lo, hi = (np.asarray(x, dtype=float) for x in C.getJointLimits())
  assert len(q) == 28
  assert np.all(q >= lo) and np.all(q <= hi)
  C.computeCollisions()
  assert C.getCollisionsTotalPenetration() <= 1e-9


def test_openarm_wrist_limits_are_tightened_but_keep_home_and_exit_valid():
  C = make_openarm_duo_env(2)
  names = list(C.getJointNames())
  limits = np.asarray(C.getJointLimits(), dtype=float)
  home = np.asarray(C.getJointState(), dtype=float)
  exit_q = np.asarray(robot_exit_configuration(C, A.get_robots(C)), dtype=float)

  j6 = [i for i, name in enumerate(names) if name.endswith("_joint6")]
  j7 = [i for i, name in enumerate(names) if name.endswith("_joint7")]
  assert len(j6) == len(j7) == 4
  np.testing.assert_allclose(limits[:, j6], np.tile([[-1.8], [1.8]], (1, 4)))
  np.testing.assert_allclose(limits[:, j7], np.tile([[-1.85], [1.85]], (1, 4)))
  assert np.all(home >= limits[0]) and np.all(home <= limits[1])
  assert np.all(exit_q >= limits[0]) and np.all(exit_q <= limits[1])


def test_openarm_home_is_distinct_from_ik_exit_while_ur_style_default_is_identity():
  C = make_openarm_duo_env(2)
  robots = A.get_robots(C)
  home = np.asarray(C.getJointState(), dtype=float)
  exit_q = np.asarray(robot_exit_configuration(C, robots), dtype=float)
  assert not np.allclose(home, exit_q)

  names = list(C.getJointNames())
  for r in robots:
    j1 = next(i for i, n in enumerate(names) if n.startswith(r + "_openarm_")
              and n.endswith("_joint1"))
    assert home[j1] == 0.0
    assert exit_q[j1] == -3.0

  rik = S.RobotIK(C, robots)
  np.testing.assert_allclose(rik.home, home)
  np.testing.assert_allclose(rik.exit_configuration, exit_q)
  np.testing.assert_allclose(rik.C.getJointState(), exit_q)


def test_openarm_model_contract_rejects_the_old_home_as_manipulation_exit(monkeypatch):
  """Do not rediscover an obstructing idle-arm pose through days of failed assembly IK."""
  C = make_openarm_duo_env(2)
  home = np.asarray(C.getJointState(), dtype=float).copy()
  monkeypatch.setattr(S, "robot_exit_configuration",
                      lambda _C, _robots: home.copy())
  with pytest.raises(RuntimeError, match="OpenArm EXIT equals HOME"):
    S.RobotIK(C, A.get_robots(C))


def test_openarm_hand_collision_frames_are_derived_from_the_ee_subtree():
  """Unconventional ee_coll names must still receive intended hand-contact semantics."""
  C = make_openarm_duo_env(2)
  rik = S.RobotIK(C, A.get_robots(C))
  for robot in rik.robots:
    hand_collision = [n for n in C.getFrameNames()
                      if n.startswith(robot + "_") and "_ee_coll" in n]
    assert hand_collision
    assert all(n in rik.hand_frames and rik._is_hand_frame(n) for n in hand_collision)
    # The ancestry boundary must not swallow the last arm links into the hand exemption.
    arm_collision = next(n for n in C.getFrameNames()
                         if n.startswith(robot + "_") and "joint5_coll" in n)
    assert arm_collision not in rik.hand_frames
    assert not rik._is_hand_frame(arm_collision)


def test_openarm_mount_boxes_are_collision_geometry_but_visual_mesh_is_not():
  """The shared mount must constrain IK/MP without enabling the detailed display mesh."""
  C = make_openarm_duo_env(2)
  for k in range(2):
    assert C.getFrame(f"oa{k}_torso_base").info().get("contact") == 1
    assert C.getFrame(f"oa{k}_torso_column").info().get("contact") == 1
    assert C.getFrame(f"oa{k}_openarm_body_link0_0").info().get("contact") is None
    assert C.getFrame(f"oa{k}_torso_shoulder_plate") is None

  # RobotIK builds every per-action scene by copying this config.  Guard the copy boundary
  # explicitly: a future collision-stripping optimization may remove inactive arms, but not
  # the fixed workcell mount.
  assembly = A.make_assembly("car-mini", 1)
  C = A.add_disassembled_obj(C, assembly, grid=4, grid_gap=2.0)
  rik = S.RobotIK(C, A.get_robots(C))
  ik_scene = rik.scene_for(A.get_initial_state(C, assembly), assembly)
  for k in range(2):
    assert ik_scene.getFrame(f"oa{k}_torso_base").info().get("contact") == 1
    assert ik_scene.getFrame(f"oa{k}_torso_column").info().get("contact") == 1


def test_openarm_exact_ik_seeds_are_seven_dof_and_limit_filtered():
  C = make_openarm_duo_env(1)
  robots = A.get_robots(C)
  rik = S.RobotIK(C, robots)
  rik.ik_seeding = True
  assert [rik.robot_arm[r] for r in robots] == ["openarm_right", "openarm_left"]
  for robot in robots:
    Cr = _restricted_copy(C, rik, robot)
    target = np.asarray(Cr.getFrame(rik.gripper_frame[robot]).getPose(), dtype=float)
    # Materialize RAI's lazily updated selected-joint set, then snapshot its borrowed arrays.
    assert len(Cr.getJointNames()) == 7
    limits = np.array(Cr.getJointLimits(), dtype=float, copy=True)
    lo, hi = limits
    seeds = rik._analytic_seeds(Cr, robot, target, None)
    assert seeds, f"no exact seed for {robot}'s known-reachable home pose"
    assert all(np.asarray(q).shape == (7,) for q in seeds)
    assert all(np.all(np.asarray(q) >= lo) and np.all(np.asarray(q) <= hi) for q in seeds)


def test_ssik_neighborhood_is_bounded_guidance_when_exact_target_is_empty(monkeypatch):
  """Nearby SSIK samples must seed the original KOMO, never become an empty-set refuter."""
  rik = S.RobotIK.__new__(S.RobotIK)
  rik.ssik_seed_neighborhood = 0.01
  rik.ssik_neighborhood_cap = 3
  rik.joint_margin_reject = 0.02
  rik._cur_action = ("pick", "a1", "obj_0", 0)
  calls = []

  def fake_seeds(_C, _robot, target, _frame):
    calls.append(np.asarray(target, dtype=float).copy())
    if len(calls) == 1:
      return []                         # exact pose has no physical-limit branch
    return [np.full(7, calls[-1][0])]   # deterministic fake near-target branch

  monkeypatch.setattr(rik, "_analytic_seeds", fake_seeds)
  monkeypatch.setattr(rik, "_analytic_reject_ok", lambda _robot: True)
  seeds, exact_count = rik._analytic_seed_bundle(
      object(), "a1", np.array([0., 0., 0., 1., 0., 0., 0.]), None)
  assert exact_count == 0                # near branches do not rewrite exact coverage
  assert len(seeds) == 3                 # explicit compute bound
  assert len(calls) == 13                # exact + two radii * six axial directions


def test_openarm_exact_seeds_compose_into_full_four_arm_configuration():
  C = make_openarm_duo_env(2)
  robots = A.get_robots(C)
  rik = S.RobotIK(C, robots)
  names = C.getJointNames()
  q0 = np.asarray(C.getJointState(), dtype=float)
  arm_seeds = {}
  for k, robot in enumerate((robots[0], robots[-1])):
    idx = [i for i, n in enumerate(names) if n.startswith(robot + "_")]
    arm_seeds[robot] = [q0[idx] + (k + 1) * 1e-3]
  seeds = rik._compose_full_seeds(C, arm_seeds)
  assert len(seeds) == 1 and seeds[0].shape == q0.shape
  for k, robot in enumerate((robots[0], robots[-1])):
    idx = [i for i, n in enumerate(names) if n.startswith(robot + "_")]
    np.testing.assert_allclose(seeds[0][idx], q0[idx] + (k + 1) * 1e-3)


def test_openarm_search_context_uses_native_pinch_proxy(monkeypatch):
  C = make_openarm_duo_env(1)
  robots = A.get_robots(C)
  seen = {}

  class Checker:
    def __init__(self, _assembly, _robots, gripper_path=None, **_kwargs):
      seen["checker"] = gripper_path
      self.fact_gripper_path = None

  monkeypatch.setattr(S, "CollisionChecker", Checker)
  monkeypatch.setattr(S, "precompute_grasps",
                      lambda _assembly, gripper_path=None: seen.setdefault("precompute", gripper_path) or {})
  checker, _table = S._search_context(C, object(), robots)
  assert Path(seen["checker"]).name == "openarm_pinch_floating.g"
  assert Path(checker.fact_gripper_path).name == "openarm_pinch_floating.g"
  assert Path(seen["precompute"]).name == "openarm_pinch_floating.g"


def test_openarm_no_real_ik_replay_uses_native_pinch_proxy():
  """The schematic replay must select OpenArm without relying on a RobotIK object."""
  C = make_openarm_duo_env(1)
  assert Path(V._scene_floating_ee_path(C)).name == "openarm_pinch_floating.g"


def test_openarm_floating_env_keeps_native_pinch_and_removes_fixed_arms():
  C = make_openarm_floating_env(4)
  names = set(C.getFrameNames())
  robots = A.get_robots(C)
  assert robots == ["a1", "a2", "a3", "a4"]
  assert is_openarm_scene(C)
  assert all(f"{r}_openarm_floating_marker" in names for r in robots)
  assert all(f"{r}_openarm_ee_coll" in names for r in robots)
  assert not any("_openarm_left_base_link" in n or "_openarm_right_base_link" in n
                 for n in names)
  assert len(C.getJointState()) == 6 * len(robots)
  rik = S.RobotIK(C, robots)
  assert all(rik.robot_arm[r] is None and rik.base_link[r] is None for r in robots)
  assert Path(V._scene_floating_ee_path(C)).name == "openarm_pinch_floating.g"

  # The actual scene search context must also retain the OpenArm proxy, rather than
  # classifying this arm-less scene as a generic Robotiq setup.
  assembly = A.make_assembly("car-mini", 1)
  C = A.add_disassembled_obj(C, assembly, grid=4, grid_gap=2.0)
  checker, _ = S._search_context(C, assembly, robots)
  assert Path(checker.gripper_path).name == "openarm_pinch_floating.g"


def test_openarm_cheap_scene_tracks_real_workcell_table():
  """Initial pick geometry must use OpenArm's raised, compact table—not make_world()."""
  C = make_openarm_duo_env(1)
  assembly = A.make_assembly("car-mini", 5)
  C = A.add_disassembled_obj(C, assembly, grid=4, grid_gap=2.0)
  robots = A.get_robots(C)
  checker, _table = S._search_context(C, assembly, robots)
  cheap = checker._scene_for(A.get_initial_state(C, assembly), assembly)

  for name in ("floor", "table"):
    np.testing.assert_allclose(cheap.getFrame(name).getPose(), C.getFrame(name).getPose())
    np.testing.assert_allclose(cheap.getFrame(name).getSize(), C.getFrame(name).getSize())
  for part in assembly.objects:
    np.testing.assert_allclose(cheap.getFrame(part).getPose(), C.getFrame(part).getPose())


def test_openarm_collision_capsule_policy_is_explicit():
  for side in ("left", "right"):
    text = (ROOT / "src" / "rai" / "openarm" / f"{side}_openarm.g").read_text()
    # User policy: the double-commented guesses stay disabled; the single candidates are
    # active and are what both real-arm IK and the floating pinch proxy approximate.
    assert f"##{side}_joint4_coll" in text
    assert f"##{side}_joint6_coll" in text
    assert f"\n{side}_joint5_coll" in text
    assert f"\n{side}_ee_coll" in text
    assert f"\n{side}_finger1_coll" in text
    assert f"\n{side}_finger2_coll" in text


def test_openarm_real_finger_radius_scale_is_real_only_and_not_compounded():
  try:
    set_openarm_real_finger_radius_scale(0.5)
    C = make_openarm_duo_env(1)
    assert C.getFrame("a1_right_finger1_coll").getSize()[1] == 0.0075
    assert C.getFrame("a2_left_finger2_coll2").getSize()[1] == 0.0075

    # The separate floating grasp/closure proxy deliberately remains nominal for this
    # diagnostic knob, preserving sampled grasp ids in fixed-plan A/B comparisons.
    D = make_openarm_floating_env(1)
    assert D.getFrame("a1_openarm_finger1_coll").getSize()[1] == 0.015
  finally:
    set_openarm_real_finger_radius_scale(1.0)


def test_openarm_floating_proxy_tracks_real_tool_frame_and_open_collision_union():
  """The cheap grasp scene must not drift from a newly calibrated real pinch hand."""
  C = make_openarm_duo_env(1)
  D = ry.Config()
  D.addFile(OPENARM_FLOATING_EE_PATH)

  def relative_pose(child, parent):
    return S._pose_compose(S._pose_invert(parent.getPose()), child.getPose())

  flange = C.getFrame("a1_openarm_right_ee_base_link")
  proxy_flange = D.getFrame("ur_ee_link")
  np.testing.assert_allclose(
      relative_pose(C.getFrame("a1_gripper_center"), flange),
      relative_pose(D.getFrame("gripper_center"), proxy_flange), atol=1e-9)

  for suffix in ("ee_coll", "ee_coll1", "ee_coll2"):
    real = C.getFrame("a1_right_" + suffix)
    proxy = D.getFrame("openarm_" + suffix)
    np.testing.assert_allclose(relative_pose(real, flange),
                               relative_pose(proxy, proxy_flange), atol=1e-9)
    np.testing.assert_allclose(real.getSize(), proxy.getSize(), atol=1e-12)

  # Finger joints introduce a calibrated 30um offset. Compare the collision *union* in the
  # flange frame, independent of left/right finger labels (the left arm swaps the y labels).
  real_names = ("finger1_coll", "finger1_coll2", "finger2_coll1", "finger2_coll2")
  proxy_names = ("finger1_coll", "finger1_coll2", "finger2_coll", "finger2_coll2")
  real_xyz = sorted(tuple(np.round(relative_pose(C.getFrame("a1_right_" + n), flange)[:3], 4))
                    for n in real_names)
  proxy_xyz = sorted(tuple(np.round(relative_pose(D.getFrame("openarm_" + n), proxy_flange)[:3], 4))
                     for n in proxy_names)
  assert real_xyz == proxy_xyz


def test_openarm_ssik_tool_calibration_tracks_real_gripper_center():
  """SSIK solves to the flange; its separate flange-to-tool transform must follow RAI."""
  C = make_openarm_duo_env(1)

  def relative_pose(child, parent):
    return S._pose_compose(S._pose_invert(parent.getPose()), child.getPose())

  def matrix(pose):
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = S._quat_to_R(pose[3:])
    return T

  for prefix, side in (("a1", "right"), ("a2", "left")):
    flange = C.getFrame(f"{prefix}_openarm_{side}_ee_base_link")
    tool = C.getFrame(f"{prefix}_gripper_center")
    actual = matrix(relative_pose(tool, flange))
    expected = analytic_ik._CAL[f"openarm_{side}"]["tools"]["two_finger"]
    np.testing.assert_allclose(expected, actual, atol=1e-9)
