"""Small invariants for the opt-in canonical assembly-pose SSIK initializer."""

import numpy as np
import pytest
import robotic as ry

import solver as S


def test_bias_pose_is_the_exact_rigid_pose_encoded_by_each_soft_bias():
  """The experimental initializer must not invent a second orientation convention."""
  for z_dir, x_dir in S._ASSEMBLE_BIAS_ORIENTATIONS:
    pose = S._assemble_bias_pose((z_dir, x_dir))
    R = S._quat_to_R(pose[3:])
    assert np.allclose(pose[:3], S._ASSEMBLE_POS_BIAS_TARGET)
    assert np.allclose(R[:, 2], z_dir)
    assert np.allclose(R[:, 0], x_dir)
    assert np.allclose(R.T @ R, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_bias_pose_accepts_a_projected_position_without_changing_orientation():
  bias = S._ASSEMBLE_BIAS_ORIENTATIONS[3]
  pose = S._assemble_bias_pose(bias, [0.2, -0.3, 0.7])
  R = S._quat_to_R(pose[3:])
  assert np.allclose(pose[:3], [0.2, -0.3, 0.7])
  assert np.allclose(R[:, 2], bias[0])
  assert np.allclose(R[:, 0], bias[1])


def test_assembly_position_bias_tracks_the_physical_table_centre():
  C = ry.Config()
  C.addFrame("table").setPosition([0.82, -0.13, 0.40]).setShape(
      ry.ST.box, [0.64, 1.10, 0.08, 0.0])
  assert np.allclose(S._assemble_position_bias_target(C), [0.82, -0.13, 0.74])


def test_exact_and_projected_ablation_cannot_run_together():
  """The projection contains the exact-centre sample, so combining flags is ambiguous."""
  cfg = __import__("config").RunConfig(
      assemble_ssik_pose_seeds=True, assemble_ssik_pose_projection=True)
  with pytest.raises(ValueError, match="alternative ablations"):
    __import__("main").run(cfg)


def test_assembly_ik_ablation_policies_cannot_run_together():
  cfg = __import__("config").RunConfig(
      assemble_unbiased_first=True, assemble_ssik_pose_seeds=True)
  with pytest.raises(ValueError, match="alternative ablations"):
    __import__("main").run(cfg)
  cfg = __import__("config").RunConfig(
      assemble_unbiased_first=True, assemble_basin_racing=True)
  with pytest.raises(ValueError, match="alternative ablations"):
    __import__("main").run(cfg)
