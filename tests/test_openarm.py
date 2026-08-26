"""Focused regressions for the OpenArm duo integration.

These deliberately avoid a full assembly search: they guard the model/search boundary that
made OpenArm grasps look feasible with the old Robotiq proxy and then fail in real-arm IK.
"""
from pathlib import Path

import numpy as np
import robotic as ry

import assembly as A
import solver as S
from env import OPENARM_FLOATING_EE_PATH, make_openarm_duo_env


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
