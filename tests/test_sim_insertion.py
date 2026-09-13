"""Pure geometry contracts of the parent-relative insertion controller."""

import numpy as np
import pytest

import sim


def test_mate_error_is_expressed_in_rotating_parent_frame():
  # Parent is yawed +90deg: its local +x insertion axis is world +y.
  s = np.sqrt(0.5)
  parent = np.array([0.4, -0.2, 0.3, s, 0.0, 0.0, s])
  target = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  desired = sim._pose_compose(parent, target)
  child = desired.copy()
  child[:3] -= np.array([0.0, 0.03, 0.01])

  e = sim._mate_pose_error(parent, child, target, [1.0, 0.0, 0.0])
  assert e["axial"] == pytest.approx(0.03)
  assert e["lateral"] == pytest.approx(0.01)
  assert e["position"] == pytest.approx(np.hypot(0.03, 0.01))
  assert e["angle"] == pytest.approx(0.0)


def test_mate_error_includes_relative_orientation():
  parent = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  child = np.array([0.0, 0.0, 0.0, np.cos(0.1), 0.0, 0.0, np.sin(0.1)])
  target = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  e = sim._mate_pose_error(parent, child, target, [0.0, 0.0, 1.0])
  assert e["position"] == pytest.approx(0.0)
  assert e["angle"] == pytest.approx(0.2)


def test_mate_error_quotients_continuous_cylinder_roll():
  identity = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  target = identity.copy()
  axial_roll = identity.copy()
  axial_roll[3:] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
  e = sim._mate_pose_error(identity, axial_roll, target, [1.0, 0.0, 0.0],
                           intrinsic_axis=[0.0, 0.0, 1.0])
  assert e["angle"] == pytest.approx(0.0)
  assert np.linalg.norm(e["rotation_vec"]) == pytest.approx(0.0)

  tilt = identity.copy()
  tilt[3:] = [np.cos(0.15), np.sin(0.15), 0.0, 0.0]
  e = sim._mate_pose_error(identity, tilt, target, [1.0, 0.0, 0.0],
                           intrinsic_axis=[0.0, 0.0, 1.0])
  assert e["angle"] == pytest.approx(0.3)
  assert np.linalg.norm(e["rotation_vec"]) == pytest.approx(0.3)
