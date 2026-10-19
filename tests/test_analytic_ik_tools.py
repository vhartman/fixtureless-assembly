"""FK-level regression checks for the arm/tool-specific ssik calibration.

These tests intentionally use the real RAI models.  A branch is useful as a KOMO seed only
when it reaches the selected tool centre, not merely the arm flange or another gripper model.
"""
import numpy as np
import pytest

import assembly_planner.analytic_ik as analytic_ik
from assembly_planner.env import make_four_robot_env
from assembly_planner.problem import _pose_compose, _pose_invert
from assembly_planner.solver import _quat_to_R


def _target_in_base(C, frame):
  base = np.asarray(C.getFrame("a1_ur_base_link").getPose(), dtype=float)
  ee = np.asarray(C.getFrame(frame).getPose(), dtype=float)
  rel = _pose_compose(_pose_invert(base), ee)
  T = np.eye(4)
  T[:3, 3] = rel[:3]
  T[:3, :3] = _quat_to_R(rel[3:])
  return T, ee


def _pose_error(a, b):
  dp = float(np.linalg.norm(a[:3] - b[:3]))
  dq = min(1.0, abs(float(np.dot(a[3:], b[3:]))))
  return dp, 2.0 * float(np.arccos(dq))


@pytest.mark.parametrize("arm", ["ur5", "ur10"])
@pytest.mark.parametrize("tool_kind", ["two_finger", "vacuum"])
def test_ssik_seeds_reproduce_the_selected_tool_pose(arm, tool_kind):
  if not analytic_ik.available(arm, tool_kind):
    pytest.skip("optional ssik dependency unavailable")
  cal = analytic_ik._CAL[arm]
  U = analytic_ik._solver(arm)
  rng = np.random.default_rng(1947 if arm == "ur5" else 1948)
  for _ in range(4):
    q = -2.0 + 4.0 * rng.random(6)
    q_ssik = cal["sign"] * q + cal["offset"]
    T = np.asarray(U.fk(q_ssik), dtype=float) @ cal["tools"][tool_kind]
    seeds = analytic_ik.seed_configs(arm, T, tool_kind=tool_kind)
    assert seeds
    errors = []
    for seed in seeds:
      T2 = np.asarray(U.fk(cal["sign"] * seed + cal["offset"]), dtype=float)
      T2 = T2 @ cal["tools"][tool_kind]
      dp = np.linalg.norm(T2[:3, 3] - T[:3, 3])
      da = np.linalg.norm(T2[:3, :3] - T[:3, :3])
      errors.append((dp, da))
    assert min(errors) < (1e-7, 2e-6)


@pytest.mark.parametrize("arm", ["ur5", "ur10"])
def test_vacuum_tool_transform_matches_the_actual_rai_frame(arm):
  """Tie the calibration to RAI, independently of ssik's inverse/forward consistency."""
  rel = {}
  for tool_kind, frame in (("two_finger", "a1_ur_gripper_center"),
                           ("vacuum", "a1_ur_vacuum")):
    C = make_four_robot_env(num_robots=1, vacuum=(tool_kind == "vacuum"), robot_types=arm)
    ee = np.asarray(C.getFrame("a1_ur_ee_link").getPose(), dtype=float)
    tip = np.asarray(C.getFrame(frame).getPose(), dtype=float)
    rel[tool_kind] = _pose_compose(_pose_invert(ee), tip)

  def matrix(pose):
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = _quat_to_R(pose[3:])
    return T

  cal = analytic_ik._CAL[arm]["tools"]
  expected = cal["two_finger"] @ np.linalg.inv(matrix(rel["two_finger"])) \
      @ matrix(rel["vacuum"])
  assert np.allclose(cal["vacuum"], expected, atol=2e-6)


def test_vacuum_calibration_is_not_the_two_finger_transform():
  """The historical bug was API-level: omitting the tool silently selected Robotiq."""
  for arm in ("ur5", "ur10"):
    assert not np.allclose(
        analytic_ik._CAL[arm]["tools"]["vacuum"],
        analytic_ik._CAL[arm]["tools"]["two_finger"],
    )
