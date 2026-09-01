"""Small invariants for the opt-in canonical assembly-pose SSIK initializer."""

import numpy as np
import pytest

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


def test_assembly_pose_ssik_initializer_is_opt_in():
  off = __import__("main").parse_args([])
  on = __import__("main").parse_args(["--assemble-ssik-pose-seeds"])
  projected = __import__("main").parse_args(["--assemble-ssik-pose-projection"])
  reordered = __import__("main").parse_args(["--assemble-unbiased-first"])
  racing = __import__("main").parse_args(["--assemble-basin-racing"])
  handover_racing = __import__("main").parse_args(["--handover-basin-racing"])
  assert off.assemble_ssik_pose_seeds is False
  assert off.assemble_ssik_pose_projection is False
  assert off.assemble_unbiased_first is False
  assert off.assemble_basin_racing is False
  assert on.assemble_ssik_pose_seeds is True
  assert on.assemble_ssik_pose_projection is False
  assert projected.assemble_ssik_pose_projection is True
  assert reordered.assemble_unbiased_first is True
  assert racing.assemble_basin_racing is True
  assert racing.assemble_race_probe_evals == 24
  assert racing.assemble_race_promote == 3
  assert handover_racing.handover_basin_racing is True
  assert handover_racing.handover_race_probe_evals == 24
  assert handover_racing.handover_race_starts == 6
  assert handover_racing.handover_race_promote == 3


def test_bias_pose_accepts_a_projected_position_without_changing_orientation():
  bias = S._ASSEMBLE_BIAS_ORIENTATIONS[3]
  pose = S._assemble_bias_pose(bias, [0.2, -0.3, 0.7])
  R = S._quat_to_R(pose[3:])
  assert np.allclose(pose[:3], [0.2, -0.3, 0.7])
  assert np.allclose(R[:, 2], bias[0])
  assert np.allclose(R[:, 0], bias[1])


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
