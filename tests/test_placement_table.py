"""The per-group PLACEMENT TABLE and the flagged migration to computed rest poses.

`placement_table` is the one place that decides what a placement id means. It is filled
either from the six axis-aligned face-down rotations (default, unchanged) or from the
shape's true stable rest orientations (`--stable-rest-poses`). A face direction IS a
placement pose, so the two are the same kind of object and every consumer downstream --
places(), rest heights, the labeled heuristics, the wrench gates -- is untouched.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import problem as P


@pytest.fixture(autouse=True)
def _legacy_by_default():
  P.set_stable_rest_poses(False)
  yield
  P.set_stable_rest_poses(False)                           # module state: never leak


ASSEMBLIES = [("cross", 7), ("car-mini", 5), ("chair", 8)]


@pytest.mark.parametrize("kind,n", ASSEMBLIES)
def test_legacy_table_is_byte_identical_to_the_old_rule(kind, n):
  """The default path must not move at all -- saved plans and witnesses index this table."""
  a = P.make_assembly(kind, n)
  groups = [(o,) for o in a.objects] + [tuple(a.objects)]
  for grp in groups:
    ref = P.reference_part(set(grp))
    for face in range(len(P._FACE_DOWN_QUATS)):
      assert np.allclose(P.placement_quat(a, grp, ref, face),
                         P._legacy_placement_quat(a, ref, face), atol=1e-12)


def test_identity_id_is_reserved_in_both_modes():
  """Id 5 means 'as anchored' everywhere. `_group_rest_pose` and `_group_rest_lift`
  special-case it, and it is the default of several helpers, so the computed table must
  route its own poses around it rather than renumbering."""
  ident = np.array([1.0, 0.0, 0.0, 0.0])
  for flag in (False, True):
    P.set_stable_rest_poses(flag)
    a = P.make_assembly("car-mini", 5)
    for grp in [("obj_1",), tuple(a.objects)]:
      table = P.placement_table(a, grp, P.reference_part(set(grp)))
      assert P._IDENTITY_FACE in table
      assert np.allclose(table[P._IDENTITY_FACE], ident, atol=1e-12)


def test_computed_table_drops_impossible_wheel_placements():
  """A wheel is a disc: it rests on its two caps, not on six axis-aligned faces. The legacy
  table offers all six and leans on the stability filter; the computed table never proposes
  them in the first place."""
  P.set_stable_rest_poses(False)
  legacy = P.make_assembly("car-mini", 5)
  n_legacy = len(P.rest_faces(legacy, ("obj_1",)))
  P.set_stable_rest_poses(True)
  computed = P.make_assembly("car-mini", 5)
  ids = P.rest_faces(computed, ("obj_1",))
  assert n_legacy == 6
  assert len(ids) == 3, f"expected two caps + 'as anchored', got {ids}"
  table = P.placement_table(computed, ("obj_1",), "obj_1")
  spec = computed.part_shape("obj_1")
  # the wheel is a cylinder ROTATED inside the part frame (its axis is not part-z), so the
  # cap test must be against the shape's own axis. Re-deriving it as "part z" reads as a
  # failure that is not there.
  axis = P._quat_to_R(P._shape_quat(spec)) @ np.array([0.0, 0.0, 1.0])
  for i in ids:
    if i == P._IDENTITY_FACE:
      continue
    down = P._quat_to_R(table[i]).T @ np.array([0.0, 0.0, -1.0])   # world -z, in part frame
    assert abs(float(down @ axis)) > 0.99, "a cap rest puts the cylinder AXIS vertical"


def test_continuum_is_thinned_but_keeps_every_tier():
  """A 30cm leg returns a rolling CONTINUUM (24 equal-probability barrel rests). Keeping all
  of them would multiply the search's branching factor for no new behaviour; thinning by
  probability alone would drop the rare-but-essential cap rests. Tiers keep both."""
  poses = P.stable_rest_poses(("cylinder", [0.30, 0.015]))
  assert len(poses) > 12
  thinned = P._thin_rest_poses(poses, ("leg",))
  assert len(thinned) <= P._STABLE_REST_MAX
  tiers_in = {round(p, 4) for _q, p in poses}
  tiers_out = {round(p, 4) for _q, p in thinned}
  assert tiers_in == tiers_out, "a whole probability tier was lost"


def test_a_cube_keeps_all_six_faces_under_the_flag():
  """The flag must not LOSE placements a cube already had: its six rests are one
  equal-probability tier, and the per-tier cap is set so none of them is dropped.

  There are six ids, not seven: for an UNROTATED part the identity rotation IS one of the
  six face-down rests, and `_placement_id` maps it onto the reserved 'as anchored' id. The
  two meanings coincide there, which is exactly right."""
  P.set_stable_rest_poses(True)
  a = P.make_assembly("cross", 7)
  table = P.placement_table(a, ("obj_0",), "obj_0")
  downs = {tuple(np.round(P._quat_to_R(q).T @ np.array([0.0, 0.0, -1.0]), 6))
           for q in table.values()}
  assert len(downs) == 6, f"expected six distinct down-directions, got {len(downs)}"
  assert P._IDENTITY_FACE in table


@pytest.mark.parametrize("flag", [False, True])
def test_a_placement_id_survives_its_group_growing(flag):
  """A placement id names a ROTATION, not a slot in a per-group list.

  When a part is mated onto a resting group the group's ORIENTATION does not change -- only
  the set of orientations it could newly rest in does. So the id it was placed at must still
  resolve, to the same rotation, after the group grows. A per-group index does not: it
  silently means a different rotation (measured on car-mini before `_placement_id`: id 8,
  assigned when the group was one shape, resolved against a two-part group with eight ids,
  and the search died mid-run)."""
  P.set_stable_rest_poses(flag)
  a = P.make_assembly("car-mini", 5)
  base = a.base
  solo = P.placement_table(a, (base,), base)
  for i, q in solo.items():
    for grown in ((base, "obj_1"), tuple(a.objects)):
      assert np.allclose(P.placement_quat(a, grown, base, i), q, atol=1e-12), (
          f"placement id {i} changed meaning when the group grew to {grown}")


def test_a_grown_group_gets_its_own_rests():
  """...and the grown group still offers ITS own stable poses, which is the whole point of
  computing them: the assembled car rests differently from its bare body. Shared rotations
  keep the SAME global id, which is what makes both statements true at once."""
  P.set_stable_rest_poses(True)
  a = P.make_assembly("car-mini", 5)
  solo = P.placement_table(a, (a.base,), a.base)
  whole = P.placement_table(a, tuple(a.objects), a.base)
  assert set(whole) != set(solo), "the assembled group must have its own rests"
  shared = set(whole) & set(solo)
  assert shared, "rotations common to both must share their id"
  for i in shared:
    assert np.allclose(whole[i], solo[i], atol=1e-12)


def test_ids_are_deterministic_across_processes():
  """Ids go into saved plans, so the same rotation must give the same id in every run. They
  are derived from the rounded rotation itself, not from discovery order."""
  q = np.array([0.5, 0.5, 0.5, 0.5])
  assert P._placement_id(q) == P._placement_id(q.copy())
  assert P._placement_id(q) == P._placement_id(-q), "q and -q are the same rotation"
  assert P._placement_id(np.array([1.0, 0.0, 0.0, 0.0])) == P._IDENTITY_FACE
