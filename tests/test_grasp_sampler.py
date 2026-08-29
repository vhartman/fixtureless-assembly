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
import solver as S
from config import RunConfig
import main


CUBE = ("box", (0.05, 0.05, 0.05, 0.005))
CYL = ("cylinder", (0.12, 0.02))
MODE = {"seed": 3, "candidates": 64, "keep": 8}


@pytest.fixture(autouse=True)
def _clean_sampler_state():
  yield
  S.set_grasp_closure_check(False)
  P.clear_sampled_scene_ctx()
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


def test_wrap_metric_keeps_peg_side_pinches():
  """A peg lying on a table is picked from the side: fingers close ACROSS the diameter
  (closing axis perpendicular to the peg axis) and the palm approaches radially. The
  sampler's roll search only lays the pads ALONG the axis, so without the flag no radial
  side pinch exists at all; the legacy lift+depth metric would drop it anyway (the pad
  bends over the barrel, a ~12 mm sagitta over its 20 mm half-length, far past the 5 mm
  legacy budget). --grasp-sampler-wrap W adds the wrap roll AND scores it curvature-fair
  (surface drop relative to the nearest contact <= W), so at W=0.015 the side pinch passes
  the 0.75 support gate, and the wrap flag reserves part of the pinch quota for the side
  approach so FPS cannot race it out."""
  gates = {"seed": 0, "candidates": 200, "keep": 12,
           "margin": 0.008, "pad_support": 0.75}
  peg = ("cylinder", (0.15, 0.0125))

  def _side_radial(poses, meta):
    out = []
    for q, m in zip(poses, meta):
      if m["family"] != "pinch":
        continue
      R = P._quat_to_R(np.asarray(q[3:]))
      y = R @ np.array([0.0, 1.0, 0.0])                    # closing axis, part frame
      z = R @ np.array([0.0, 0.0, 1.0])                    # approach axis, part frame
      if abs(float(y[2])) < 0.5 and abs(float(z[2])) < 0.5:   # across the diameter, from
        out.append((float(z[2]), float(m["pad_support"])))    # the side (radial approach)
    return out

  P.set_sampled_grasps(dict(gates, wrap=None))             # legacy: no wrap roll emitted
  legacy = _side_radial(P.part_grasp_poses(peg), P.sampled_grasp_meta(peg))
  assert legacy == [], legacy                              # documents the pre-flag hole

  P.set_sampled_grasps(dict(gates, wrap=0.015))
  poses = P.part_grasp_poses(peg)
  meta = P.sampled_grasp_meta(peg)
  side = _side_radial(poses, meta)
  assert len(side) >= 2, side                       # the wrap-roll quota (keep 12 -> 2 slots)
  assert all(s >= 0.75 for _nz, s in side), side    # admitted above the gate, no fallback
  # the gate still holds for every retained pinch (no silent fallback under the wrap flag)
  assert all(m["pad_support"] >= 0.75
             for q, m in zip(poses, meta) if m["family"] == "pinch")


def test_cli_defaults_and_guard():
  assert RunConfig().grasp_sampler is False
  assert RunConfig().grasp_sampler_scenes is False
  a = main.parse_args([])
  assert a.grasp_sampler is False and a.grasp_sampler_keep == 20
  assert a.grasp_sampler_version == 2
  with pytest.raises(SystemExit):
    main.run(RunConfig(grasp_sampler=True, dense_grasps=True, solve=False, view=False))
  with pytest.raises(SystemExit):
    main.run(RunConfig(grasp_sampler_scenes=True, solve=False, view=False))


def test_v2_hybrid_contains_conventional_grasps_and_refines_by_prefix():
  """Primitive V2 DBs keep every conventional grasp and increasing K appends ids."""
  P.set_rolls_per_face(2)
  P.set_sampled_grasps(None)
  conventional = [np.asarray(q) for q in P.part_grasp_poses(CUBE)]
  mode = {"version": 2, "seed": 3, "candidates": 96, "keep": 20,
          "margin": 0.008, "pad_support": 0.75, "wrap": 0.015}
  P.set_sampled_grasps(mode)
  q20 = [np.asarray(q).copy() for q in P.part_grasp_poses(CUBE)]
  m20 = P.sampled_grasp_meta(CUBE)
  canonical = [q for q, m in zip(q20, m20) if m["origin"] == "canonical"]
  assert len(q20) == 20 and len(canonical) == len(conventional) == 12
  assert all(np.array_equal(q, c) for q, c in zip(canonical, conventional))

  P.set_sampled_grasps(dict(mode, keep=24))
  q24 = P.part_grasp_poses(CUBE)
  assert len(q24) == 24
  assert all(np.array_equal(a, b) for a, b in zip(q20, q24))


def test_closure_ranking_prioritizes_certified_canonical_rows_without_deleting_fallbacks():
  """A cuboid's conventional prefix must not bypass closure-aware sampler ranking."""
  entries = []
  for i, coverage in enumerate((0, 2, 1, 0)):
    # Distinct valid pose rows; only the metadata coverage matters to the tier ordering.
    row = (np.array([0.01 * i, 0.0, 0.06, 1.0, 0.0, 0.0, 0.0]),
           "flat", 0.05, np.zeros(6), float("inf"), 1.0, "canonical", False)
    roles = tuple(("pick", "assemble")[:coverage])
    entries.append((row, {"closure_by_kind": {"two_finger": roles}}))
  ranked = P._closure_ranked_canonical(entries)
  coverages = [sum(len(v) for v in e[1]["closure_by_kind"].values()) for e in ranked]
  assert coverages == [2, 1, 0, 0]
  assert {id(e[0]) for e in ranked} == {id(e[0]) for e in entries}


def test_v2_scene_masks_are_per_gripper_kind():
  """Mixed fleets must not classify every explicit pose through whichever EE slot is first."""
  import env
  from problem import add_disassembled_obj, get_robots, make_assembly
  P.set_rolls_per_face(2)
  P.set_sampled_grasps({"version": 2, "seed": 0, "candidates": 64, "keep": 20,
                        "margin": 0.008, "pad_support": 0.75, "wrap": 0.015})
  C = env.make_four_robot_env(num_robots=2, floating_ee=False, vacuum=[True, False])
  asm = make_assembly("cross", 3)
  C = add_disassembled_obj(C, asm, spacing=0.2)
  robots = get_robots(C)
  P.install_sampled_scene_ctx(C, asm, robots=robots)
  spec = asm.part_shape("obj_1")
  poses = P.part_grasp_poses(spec, part="obj_1")
  meta = P.sampled_grasp_meta(spec, part="obj_1")
  assert all(set(m["scene_by_kind"]) == {"vacuum", "two_finger"} for m in meta)
  ctx = P._SAMPLED_SCENE_CTX
  reps = ctx["kind_robot"]
  for q, m in zip(poses, meta):
    for kind, roles in m["scene_by_kind"].items():
      for role in roles:
        state = ctx["s0"] if role == "pick" else ctx["s_full"]
        assert ctx["checker"].pose_collision_free(
            state, "obj_1", q, asm, robot=reps[kind])


def test_openarm_car_wheel_sampler_generates_strict_table_clear_grasp():
  """A flat 12mm wheel needs a small approach retraction for the 15mm OpenArm fingers.

  The source pose remains useful away from the table, but the PICK-labelled database member
  must be a distinct generated pose that passes the strict full closure certificate—not the
  historical 5mm open-hand tolerance.
  """
  import env
  from problem import add_disassembled_obj, get_robots, make_assembly

  P.set_rolls_per_face(2)
  P.set_sampled_grasps({"version": 2, "seed": 0, "candidates": 96, "keep": 16,
                        "margin": 0.008, "pad_support": 0.75, "wrap": 0.015})
  S.set_grasp_closure_check(True)
  C = env.make_openarm_duo_env(2)
  asm = make_assembly("car-mini", 5)
  C = add_disassembled_obj(C, asm, grid=4, grid_gap=2.0, float_height=0.002)
  robots = get_robots(C)
  P.install_sampled_scene_ctx(C, asm, robots=robots)

  poses = P.part_grasp_poses(asm.part_shape("obj_1"), part="obj_1")
  meta = P.sampled_grasp_meta(asm.part_shape("obj_1"), part="obj_1")
  refined = [(q, m) for q, m in zip(poses, meta)
             if m.get("refinement", (None,))[0] == "approach-retract"
             and "pick" in m.get("scene_by_kind", {}).get("two_finger", ())]
  assert refined, "no strictly table-clear generated wheel pick"

  ctx = P._SAMPLED_SCENE_CTX
  checker, robot, s0 = ctx["checker"], ctx["kind_robot"]["two_finger"], ctx["s0"]
  for pose, md in refined:
    assert 0.0015 < md["refinement"][3] < 0.010
    assert checker.pose_collision_free(s0, "obj_1", pose, asm, robot=robot)
    cert = checker.pose_grasp_closure_result(s0, "obj_1", pose, asm, robot)
    assert cert is not None and cert.feasible


def test_v2_canonical_epoch_reproduces_legacy_grasp_table():
  """Staged epoch zero must be the old primitive action space, not merely similar poses."""
  from problem import make_assembly
  P.set_rolls_per_face(2)
  asm = make_assembly("cross", 2)
  P.set_sampled_grasps(None)
  legacy = P.precompute_grasps(asm)
  P.set_sampled_grasps({"version": 2, "seed": 0, "candidates": 32, "keep": 20,
                        "margin": 0.008, "pad_support": 0.75, "wrap": 0.015})
  P.set_sampled_canonical_only(True)
  staged = P.precompute_grasps(asm)
  P.set_sampled_canonical_only(False)
  full = P.precompute_grasps(asm)
  assert set(staged) == set(legacy)
  for key in legacy:
    assert [(p, g) for p, g, _q in staged[key]] == [(p, g) for p, g, _q in legacy[key]]
    for (_p, _g, a), (_p2, _g2, b) in zip(staged[key], legacy[key]):
      np.testing.assert_array_equal(a, b)
  assert any(len(full[k]) > len(staged[k]) for k in staged)


def test_wrap_emits_side_roll_for_long_boxes():
  """A long flat box (the cuboid peg) needs the SIDE pinch: fingers across the short
  dimension, palm approaching perpendicular to the long axis. Without the complementary
  roll the sampler only emits pads-along-the-long-axis pinches, whose palm sits at the peg
  END and collides with the seat during insertion -- the cuboid-peg failure the user hit."""
  gates = {"seed": 0, "candidates": 200, "keep": 12, "margin": 0.008, "pad_support": 0.5}
  peg = ("box", (0.02, 0.02, 0.11, 0.0))                  # long axis = part z, like the legs

  def _side(poses, meta):
    out = []
    for q, m in zip(poses, meta):
      if m["family"] != "pinch":
        continue
      R = P._quat_to_R(np.asarray(q[3:]))
      y = R @ np.array([0.0, 1.0, 0.0])                   # closing axis, part frame
      z = R @ np.array([0.0, 0.0, 1.0])                   # approach axis, part frame
      if abs(float(y[2])) < 0.5 and abs(float(z[2])) < 0.5:   # across the short dimension,
        out.append(float(m["pad_support"]))                   # approach from the side
    return out

  P.set_sampled_grasps(dict(gates, wrap=None))            # legacy: no complementary roll
  assert _side(P.part_grasp_poses(peg), P.sampled_grasp_meta(peg)) == []

  P.set_sampled_grasps(dict(gates, wrap=0.1))
  side = _side(P.part_grasp_poses(peg), P.sampled_grasp_meta(peg))
  assert side, "no side pinch of the long box with --grasp-sampler-wrap 0.1"
  assert all(s >= 0.5 for s in side)                      # admitted above the support gate


def test_scene_aware_sets_free_pick_asm():
  """--grasp-sampler-scenes: the DB per PART INSTANCE is split into FREE / PICK (floating-EE
  clear at the initial pose) / ASSEMBLE (clear at the seated pose) sets with per-set quotas.
  Every tagged row must re-pass its own set's check, and identical parts get their OWN DB."""
  import env
  from problem import add_disassembled_obj, get_robots, make_assembly
  P.set_sampled_grasps({"seed": 0, "candidates": 64, "keep": 12,
                        "margin": 0.008, "pad_support": 0.75})
  C = env.make_four_robot_env(num_robots=1, floating_ee=False, vacuum=False)
  asm = make_assembly("cross", 3)
  C = add_disassembled_obj(C, asm, spacing=0.2)
  P.install_sampled_scene_ctx(C, asm, robots=get_robots(C))
  spec = asm.part_shape("obj_1")
  poses = P.part_grasp_poses(spec, part="obj_1")
  meta = P.sampled_grasp_meta(spec, part="obj_1")
  assert len(poses) == len(meta) == 12
  assert all("set" in m for m in meta)
  sets = {m["set"] for m in meta}
  assert sets <= {"free", "pick", "asm"}
  assert "pick" in sets and "free" in sets            # a cube on the table has both pools
  ctx = P._SAMPLED_SCENE_CTX
  for pose, m in zip(poses, meta):
    state = ctx["s0"] if m["set"] == "pick" else ctx["s_full"] if m["set"] == "asm" else None
    if state is not None:                             # tagged rows re-pass their own gate
      assert ctx["checker"].pose_collision_free(state, "obj_1", pose, asm)
  # identical shape, different instance: its own DB (same raw cloud, own filter)
  spec2 = asm.part_shape("obj_2")
  assert spec2[0] == "box" and spec2[1] == spec[1]
  poses2 = P.part_grasp_poses(spec2, part="obj_2")
  assert len(poses2) == 12
  assert any(not np.array_equal(a, b) for a, b in zip(poses2, poses))   # own filter per instance
  # a part-less query under the scene ctx must fail loudly, never silently unfiltered
  with pytest.raises(ValueError):
    P.part_grasp_poses(spec)


def test_scene_aware_off_is_byte_identical():
  P.set_sampled_grasps({"seed": 0, "candidates": 64, "keep": 12})
  before = [np.array(p) for p in P.part_grasp_poses(CUBE)]
  P.reset_grasp_pose_cache()                          # rebuild without any scene context
  after = P.part_grasp_poses(CUBE)
  assert len(before) == len(after)
  for a, b in zip(before, after):
    np.testing.assert_array_equal(a, b)
