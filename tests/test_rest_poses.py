"""TRUE stable rest orientations (`problem.stable_rest_poses`) -- the enumerator half of the
deferred "change 2".

These are NOT yet wired into the search: a placement id is still an index into the six
_FACE_DOWN_QUATS and it lives inside State.place, so swapping the enumerator rebinds every
saved plan and witness. The tests pin the computation so the integration can be done later
against a known-good enumerator.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assembly_planner.problem as P


def _down_axis(quat):
  """Which PART-frame axis points down (-z world) in this rest pose, as (index, sign)."""
  R = P._quat_to_R(np.asarray(quat, dtype=float))
  col = R.T @ np.array([0.0, 0.0, -1.0])                   # world -z in part coordinates
  i = int(np.argmax(np.abs(col)))
  return i, float(np.sign(col[i]))


def test_cube_has_six_equally_likely_rests():
  r = P.rest_pose_report(("box", [0.05, 0.05, 0.05, 0.005]))
  assert r["n"] == 6 and r["equal_probs"]
  assert not r["continuum"], "six equal rests are six FACES, not a rolling continuum"
  assert not r["neutral"]


def test_probability_follows_the_footprint():
  """A flat plate rests on its two large faces far more often than on its four edges. The
  current six-face table cannot express this ranking at all -- every face is equally good."""
  poses = P.stable_rest_poses(("box", [0.08, 0.08, 0.012, 0.002]))
  probs = sorted((p for _q, p in poses), reverse=True)
  assert len(poses) == 6
  assert probs[0] == pytest.approx(probs[1], rel=1e-3)     # the two large faces
  assert probs[1] > 5.0 * probs[2], f"large faces not dominant: {probs}"


def test_cylinder_caps_and_the_rolling_continuum():
  """A squat cylinder has exactly two rests (its caps). A long thin one is dominated by the
  BARREL, which is a continuum: trimesh samples it at many roll angles with equal
  probability. Reporting that as a set of distinct placements would be wrong."""
  squat = P.rest_pose_report(("cylinder", [0.03, 0.025]))
  assert squat["n"] == 2 and squat["equal_probs"] and not squat["continuum"]
  for q, _p in P.stable_rest_poses(("cylinder", [0.03, 0.025])):
    assert _down_axis(q)[0] == 2                           # a cap is down: part z is vertical
  leg = P.rest_pose_report(("cylinder", [0.30, 0.015]))
  assert leg["continuum"], f"a long leg should roll: {leg}"


def test_sphere_is_neutral_not_empty():
  """A sphere has NO discrete rest -- it rests in every orientation. The heuristics assume a
  canonical rest face exists and send h to infinity when it does not, so this case must be
  answered explicitly rather than discovered."""
  r = P.rest_pose_report(("sphere", [0.03]))
  assert r["neutral"] and r["n"] == 0


def test_chamfered_mesh_rests_on_faces_not_chamfers():
  """The chamfer facets are too small to catch the centre of mass, so a chamfered cube still
  rests on its six faces. This is the case the six-face table happens to get right -- and it
  is worth pinning, because the mesh path must not invent extra rests."""
  r = P.rest_pose_report(("mesh", {"file": "part_meshes/chamfered_box.stl"}))
  assert r["n"] == 6 and r["equal_probs"]


def test_compound_rests_include_the_plinth():
  spec = ("compound", [
      {"shape": ("box", [0.05, 0.05, 0.03, 0.005]), "pos": [0.0, 0.0, -0.015]},
      {"shape": ("cylinder", [0.03, 0.018]), "pos": [0.0, 0.0, 0.015]}])
  poses = P.stable_rest_poses(spec)
  assert len(poses) >= 2
  assert any(_down_axis(q) == (2, 1.0) for q, _p in poses), "must rest on the box underside"


# --------------------------------------------------------------------------- groups
def test_group_rests_differ_from_the_root_part():
  """A group is not its root part: the assembled car rests on its WHEELS. The current model
  takes the reference PART's faces, which is why a group can end up with no usable rest."""
  a = P.make_assembly("car-mini", 5)
  group = P.group_stable_rest_poses(a, tuple(a.objects))
  body = P.stable_rest_poses(a.part_shape(a.base))
  assert group, "the assembled car must have a rest pose"
  assert len(group) != len(body) or not np.allclose(group[0][0], body[0][0]), \
      "group rests must be computed from the group, not from its root part"


def test_group_uses_mass_weighted_centre_of_mass():
  """Masses differ per part, so a volume centroid is the wrong pivot. Making one part much
  heavier must change the ranking of the group's rests."""
  a = P.make_assembly("car-mini", 5)
  base = dict(a.masses)
  before = [p for _q, p in P.group_stable_rest_poses(a, tuple(a.objects))]
  heavy = a.objects[1]
  a.masses[heavy] = base[heavy] * 200.0
  after = [p for _q, p in P.group_stable_rest_poses(a, tuple(a.objects))]
  a.masses.update(base)
  assert before != after, "mass had no effect -- the centre of mass is not mass weighted"
