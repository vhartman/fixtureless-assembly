"""--grasp-sampler: the mesh-sampled second grasp-DB path. Pins the contract the rest of
the pipeline relies on: OFF = byte-identical to the enumerator; ON = deterministic across
cache resets (private RNG, never the global stream); the three parallel per-gid views stay
aligned; poses obey the load-bearing frame conventions (local z = approach INTO the part,
unit quats, antipodal widths within the finger aperture)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import problem as P
from config import RunConfig
import main


CUBE = ("box", (0.05, 0.05, 0.05, 0.005))
CYL = ("cylinder", (0.12, 0.02))
MODE = {"seed": 3, "candidates": 64, "keep": 8}


@pytest.fixture(autouse=True)
def _clean_sampler_state():
  yield
  P.set_sampled_grasps(None)
  P.reset_grasp_pose_cache()


def test_flag_off_is_byte_identical():
  P.set_sampled_grasps(None)
  P.reset_grasp_pose_cache()
  base = P.part_grasp_poses(CUBE)
  base_s = P.part_grasp_surfaces(CUBE)
  base_m = P.part_grasp_face_min_dims(CUBE)
  P.set_sampled_grasps(MODE)                               # on, then OFF again
  P.set_sampled_grasps(None)
  P.reset_grasp_pose_cache()
  again = P.part_grasp_poses(CUBE)
  assert len(base) == len(again)
  for a, b in zip(base, again):
    np.testing.assert_array_equal(a, b)
  assert base_s == P.part_grasp_surfaces(CUBE)
  assert base_m == P.part_grasp_face_min_dims(CUBE)


def test_sampled_deterministic_across_cache_reset():
  P.set_sampled_grasps(MODE)
  first = [np.array(p) for p in P.part_grasp_poses(CUBE)]
  P.reset_grasp_pose_cache()                               # fresh compute, same seed
  second = P.part_grasp_poses(CUBE)
  assert len(first) == len(second) == MODE["keep"]
  for a, b in zip(first, second):
    np.testing.assert_array_equal(a, b)


def test_parallel_views_aligned_and_labels_sane():
  P.set_sampled_grasps(MODE)
  for spec in (CUBE, CYL):
    poses = P.part_grasp_poses(spec)
    surfs = P.part_grasp_surfaces(spec)
    mins = P.part_grasp_face_min_dims(spec)
    assert len(poses) == len(surfs) == len(mins)
    assert all(s in ("flat", "curved") for s in surfs)
    assert all(m >= 0.0 for m in mins)
  # a cube has no curved surface anywhere
  assert set(P.part_grasp_surfaces(CUBE)) == {"flat"}


def test_frame_conventions():
  P.set_sampled_grasps(MODE)
  h = P._shape_half_extents(CUBE)
  for pose in P.part_grasp_poses(CUBE):
    pos, q = np.asarray(pose[:3]), np.asarray(pose[3:])
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    z = P._quat_to_R(q) @ np.array([0.0, 0.0, 1.0])        # approach axis
    # approach points INTO the part: stepping along z from the tool centre must DECREASE
    # the distance to the part centre (the asymmetric wrench model's sign convention).
    assert np.linalg.norm(pos + 0.01 * z) < np.linalg.norm(pos)
    # the tool centre stands OFF the surface, outside the box
    assert np.max(np.abs(pos) - h) > 0.0


def test_antipodal_width_within_aperture():
  P.set_sampled_grasps({"seed": 1, "candidates": 128, "keep": 12})
  big = ("box", (0.2, 0.05, 0.05, 0.005))                  # 200mm > aperture along x
  poses = P.part_grasp_poses(big)
  mins = P.part_grasp_face_min_dims(big)
  n_pinch = 0
  for pose, mind in zip(poses, mins):
    if mind != 0.0:
      continue                                             # suction row: no width constraint
    n_pinch += 1
    y = P._quat_to_R(np.asarray(pose[3:])) @ np.array([0.0, 1.0, 0.0])   # closing axis
    width = 2.0 * float(np.abs(y) @ P._shape_half_extents(big))
    assert width <= P._MAX_GRASP_WIDTH + 1e-6, f"pinch across {width*1000:.0f}mm > aperture"
  assert n_pinch > 0, "no antipodal rows sampled at all"


def test_cylinder_pinches_include_both_axial_approaches():
  """A peg held for insertion needs the palm on the free end, not through the socket.

  Antipodal contacts fix the closing axis but leave two valid approach directions.  The
  sampled DB must retain both signs, just as the face enumerator retains two pinch rolls.
  """
  P.set_sampled_grasps({"seed": 0, "candidates": 200, "keep": 12,
                        "margin": 0.008, "pad_support": 0.75})
  poses = P.part_grasp_poses(CYL)
  meta = P.sampled_grasp_meta(CYL)
  axial = [(P._quat_to_R(np.asarray(q[3:])) @ np.array([0.0, 0.0, 1.0]))[2]
           for q, m in zip(poses, meta) if m["family"] == "pinch"]
  assert any(z > 0.8 for z in axial), axial
  assert any(z < -0.8 for z in axial), axial


def test_cli_defaults_and_guard():
  assert RunConfig().grasp_sampler is False
  a = main.parse_args([])
  assert a.grasp_sampler is False and a.grasp_sampler_keep == 12
  with pytest.raises(SystemExit):
    main.run(RunConfig(grasp_sampler=True, dense_grasps=True, solve=False, view=False))
