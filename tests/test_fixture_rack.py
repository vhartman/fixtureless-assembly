"""Structural regression tests for the scalable fixture-rack benchmark family."""

import numpy as np

import problem as P


def test_fixture_rack_scales_and_truncates_by_station():
  for n in range(1, 14):
    a = P.make_assembly("fixture-rack", n)
    assert len(a.objects) == n
    assert a.base == "obj_0"
    assert P.assembly_max_parts("fixture-rack") == 13
    assert set(a.part_shapes) == set(a.objects)


def test_fixture_rack_has_branching_precedence_and_two_mate_directions():
  a = P.make_assembly("fixture-rack", 9)                  # rail + four complete stations
  posts = ["obj_1", "obj_3", "obj_5", "obj_7"]
  brackets = ["obj_2", "obj_4", "obj_6", "obj_8"]

  for post in posts:
    parent, _ = a.relative_pose(post)
    assert parent == a.base
    assert np.allclose(a.mate_axis(post), [0.0, 0.0, -1.0])

  for station, bracket in enumerate(brackets):
    post = posts[station]
    parent, off = a.relative_pose(bracket)
    assert parent == post
    assert a.dependencies_of(bracket) == [post]
    assert abs(off[1]) > abs(off[0]) + abs(off[2])
    expected = [0.0, -1.0 if station % 2 == 0 else 1.0, 0.0]
    assert np.allclose(a.mate_axis(bracket), expected)

  # Independent post branches are what create assembly-order choices; this is not a chain.
  assert all(not a.dependencies_of(post) for post in posts)


def test_fixture_rack_geometry_is_heterogeneous_and_nonoverlapping_by_station():
  a = P.make_assembly("fixture-rack", 13)
  rail = np.asarray(a.part_shape("obj_0")[1][:3], dtype=float)
  post = np.asarray(a.part_shape("obj_1")[1][:3], dtype=float)
  bracket = np.asarray(a.part_shape("obj_2")[1][:3], dtype=float)
  assert not np.allclose(rail, post)
  assert not np.allclose(post, bracket)
  assert rail[0] > rail[1] and post[2] > post[0] and bracket[1] > bracket[0]

  xs = [a.position_in_base(f"obj_{i}")[0] for i in range(1, 13, 2)]
  assert np.allclose(np.diff(xs), a.station_pitch)
  assert min(np.diff(xs)) > post[0]


def test_fixture_rack_alternates_bracket_sides():
  a = P.make_assembly("fixture-rack", 13)
  ys = [a.position_in_base(f"obj_{i}")[1] for i in range(2, 13, 2)]
  assert all(y != 0.0 for y in ys)
  assert [int(np.sign(y)) for y in ys] == [1, -1, 1, -1, 1, -1]


def test_fixture_rack_mates_have_explicit_collision_clearance():
  a = P.make_assembly("fixture-rack", 5)
  rail = np.asarray(a.part_shape("obj_0")[1][:3], dtype=float)
  post = np.asarray(a.part_shape("obj_1")[1][:3], dtype=float)
  bracket = np.asarray(a.part_shape("obj_2")[1][:3], dtype=float)

  _, post_off = a.relative_pose("obj_1")
  post_gap = post_off[2] - 0.5 * (rail[2] + post[2])
  assert np.isclose(post_gap, a.mate_clearance)

  _, bracket_off = a.relative_pose("obj_2")
  bracket_gap = abs(bracket_off[1]) - 0.5 * (post[1] + bracket[1])
  assert np.isclose(bracket_gap, a.mate_clearance)


def test_fixture_rack_mass_matches_its_small_wood_polymer_scale():
  a = P.make_assembly("fixture-rack", 5)
  assert np.isclose(a.mass("obj_1"), 0.03375)
  assert np.isclose(a.mass("obj_2"), 0.0225)
  assert sum(a.mass(p) for p in a.objects) < 0.25


def test_fixture_rack_10p_large_pitch_scatter_is_collision_free_and_reachable():
  a = P.make_assembly("fixture-rack", 10)
  C = P.add_disassembled_obj(P.make_world(), a, spacing=0.2, float_height=0.002)
  pos = {p: np.asarray(C.getFrame(p).getPosition(), dtype=float) for p in a.objects}

  assert C.getCollisionFree()
  assert max(abs(q[0]) for q in pos.values()) <= 0.5 + 1e-9
  assert np.isclose(pos["obj_0"][1], -0.15)
  # Six columns at 0.2m: obj_6 starts the second row instead of running to x=0.7.
  assert np.isclose(pos["obj_6"][0], -0.5)
  assert np.isclose(pos["obj_6"][1], 0.2)
