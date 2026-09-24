"""Compound part shapes: ("compound", [{"shape": spec, "pos": [x,y,z]}, ...]).

ONE rigid part built from several primitives/meshes, realized as __shape{i} children.
Validated end to end on compound-demo (cube base + box-with-boss + cube cap; mixed
grippers, --grasp-sampler, --grasp-eq-weight 100 --grasp-accuracy 0.001
--mate-accuracy 0.0005): search -> witnesses -> BiRRT -> MuJoCo, 6/6. These tests pin
the fast pieces.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import problem as P

SPEC = ("compound", [
    {"shape": ("box", [0.05, 0.05, 0.03, 0.005]), "pos": [0.0, 0.0, -0.015]},
    {"shape": ("cylinder", [0.03, 0.018]), "pos": [0.0, 0.0, 0.015]},
])
ASYM = ("compound", [  # deliberately NOT symmetric about the origin
    {"shape": ("box", [0.04, 0.04, 0.02, 0.005]), "pos": [0.0, 0.0, 0.03]},
])


@pytest.fixture(autouse=True)
def _no_sampler_leak():
  yield
  P.set_sampled_grasps(None)


def test_half_extents_are_the_symmetric_cover():
  assert np.allclose(P._shape_half_extents(SPEC), [0.025, 0.025, 0.03], atol=1e-9)
  assert P._shape_half_extent_along(SPEC, np.array([0.0, 0.0, 1.0])) == pytest.approx(0.03)


def test_half_z_is_exact_per_child_even_asymmetric():
  # child centre at +0.03 with half 0.01 -> lowest point at +0.02 -> DOWNWARD extent -0.02
  assert P._shape_half_z(ASYM, np.eye(3)) == pytest.approx(-0.02)
  R_flip = P._quat_to_R(np.array([0.0, 1.0, 0.0, 0.0]))    # upside down: extent +0.04
  assert P._shape_half_z(ASYM, R_flip) == pytest.approx(0.04)


def test_support_points_union():
  pose = np.array([0.0, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0])
  pts = P._part_support_points_world(pose, SPEC)
  assert len(pts) == 8 + 16                                # box corners + cylinder rims
  assert min(x[2] for x in pts) == pytest.approx(0.0, abs=1e-9)
  assert max(x[2] for x in pts) == pytest.approx(0.06, abs=1e-9)


def test_rai_children_and_contact_toggle():
  import robotic as ry
  C = ry.Config()
  P.add_part_frame(C, "cp", SPEC)
  assert P.part_shape_frames(C, "cp") == ["cp__shape0", "cp__shape1"]
  P.set_part_contact(C, "cp", 0)                           # must not raise; toggles BOTH
  P.set_part_contact(C, "cp", 1)


def test_enumerator_refuses_compound_without_sampler():
  P.set_sampled_grasps(None)
  with pytest.raises(ValueError, match="grasp-sampler"):
    P.part_grasp_poses(SPEC)


def test_sampled_views_over_the_union():
  P.set_sampled_grasps({"seed": 0, "candidates": 200, "keep": 12})
  poses, surfs, mins = P._sampled_grasp_views(SPEC, 0.01)
  assert len(poses) == 12
  assert "flat" in surfs                                   # box faces / boss cap


def test_spec_key_distinguishes_compounds():
  other = ("compound", [
      {"shape": ("box", [0.05, 0.05, 0.03, 0.005]), "pos": [0.0, 0.0, -0.015]},
      {"shape": ("cylinder", [0.03, 0.012]), "pos": [0.0, 0.0, 0.015]},
  ])
  assert P._spec_key(SPEC) != P._spec_key(other)
  assert P._spec_key(SPEC) == P._spec_key(("compound", list(SPEC[1])))


def test_no_nested_compounds():
  bad = ("compound", [{"shape": SPEC, "pos": [0, 0, 0]}])
  with pytest.raises(ValueError, match="nested"):
    P._compound_children(bad)


def test_sim_geoms_split_mass_by_volume():
  import sim as S
  xml = S._compound_geoms_xml(SPEC, 0.8, 1.0, 3, "1 0 0 1")
  # Rounded boxes are represented by several overlapping primitives.  Pin the physical
  # invariant (one total part mass, distributed by child volume), not an obsolete exact geom
  # count from before that collision-model improvement.
  assert xml.count("<geom") >= 2
  import re
  masses = [float(m) for m in re.findall(r'mass="([0-9.e-]+)"', xml)]
  assert sum(masses) == pytest.approx(1.0)
  v_box = 0.05 * 0.05 * 0.03
  v_cyl = np.pi * 0.018 ** 2 * 0.03
  assert sum(masses[:-1]) == pytest.approx(v_box / (v_box + v_cyl))
  assert masses[-1] == pytest.approx(v_cyl / (v_box + v_cyl))


def test_peg_hole_compound_preserves_the_open_pocket():
  """The seated peg occupies the socket's concavity without penetrating any convex child."""
  asm = P.make_assembly("peg-hole", 2)
  assert asm.part_shape("obj_0")[0] == "compound"
  assert len(P._compound_children(asm.part_shape("obj_0"))) == 5
  C = P.make_subassembly_scene(asm, tuple(asm.objects))
  C.computeCollisions()
  assert C.getCollisionFree(), C.getCollisions()


def test_peg_hole_mujoco_contains_five_socket_geoms_and_the_peg():
  import sim as S
  asm = P.make_assembly("peg-hole", 2)
  poses = {
      "obj_0": np.array([0.0, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]),
      "obj_1": np.array([0.0, 0.0, 0.038, 1.0, 0.0, 0.0, 0.0]),
  }
  state = P.State(parent={"obj_0": ("table",), "obj_1": ("part", "obj_0")},
                  supports={"obj_0": frozenset({("table",)}),
                            "obj_1": frozenset({("mate", "obj_0")})},
                  poses=poses)
  xml = S.build_mjcf(asm, state, ["a1"])
  # Five convex pieces make the socket; the peg remains one cylinder geom.
  socket_body = xml.split('<body name="obj_0"', 1)[1].split("</body>", 1)[0]
  peg_body = xml.split('<body name="obj_1"', 1)[1].split("</body>", 1)[0]
  assert socket_body.count("<geom") == 5
  assert peg_body.count("<geom") == 1 and 'type="cylinder"' in peg_body


def test_peg_stool_has_three_real_collision_free_sockets():
  asm = P.make_assembly("peg-stool", 4)
  assert asm.part_shape("obj_0")[0] == "compound"
  # Plain seat plate + four convex walls for each of three blind sockets: no vacuum pedestal.
  assert len(P._compound_children(asm.part_shape("obj_0"))) == 13
  assert all(np.allclose(asm.mate_axes[f"obj_{i}"], [0.0, 0.0, 1.0])
             for i in range(1, 4))
  C = P.make_subassembly_scene(asm, tuple(asm.objects))
  C.computeCollisions()
  assert C.getCollisionFree(), C.getCollisions()


def test_peg_stool_cuboid_square_legs_fit_the_real_sockets():
  asm = P.make_assembly("peg-stool-cuboid", 4)
  assert asm.part_shape("obj_0")[0] == "compound"
  assert len(P._compound_children(asm.part_shape("obj_0"))) == 13
  for i in range(1, 4):
    kind, prm = asm.part_shape(f"obj_{i}")[:2]
    assert kind == "box" and prm[2] > prm[0]              # a long square leg, not a cube
    assert prm[0] == prm[1]
    sq = asm.scatter_orientations.get(f"obj_{i}")         # initial pose: LYING on the table
    assert sq is not None
    leg_axis_world = P._quat_to_R(np.asarray(sq, dtype=float)) @ np.array([0.0, 0.0, 1.0])
    assert abs(float(leg_axis_world[2])) < 0.5            # long axis horizontal, not standing
  assert all(np.allclose(asm.mate_axes[f"obj_{i}"], [0.0, 0.0, 1.0])
             for i in range(1, 4))
  C = P.make_subassembly_scene(asm, tuple(asm.objects))
  C.computeCollisions()
  assert C.getCollisionFree(), C.getCollisions()          # square pegs clear the socket corners


def test_peg_stool_cuboid_4_legs():
  asm = P.make_assembly("peg-stool-cuboid-4", 5)
  assert len(asm.objects) == 5
  # Plain seat plate + four convex walls for each of FOUR blind sockets.
  assert len(P._compound_children(asm.part_shape("obj_0"))) == 17
  for i in range(1, 5):
    kind, prm = asm.part_shape(f"obj_{i}")[:2]
    assert kind == "box" and prm[0] == prm[1] and prm[2] > prm[0]
    assert asm.scatter_orientations.get(f"obj_{i}") is not None   # lying flat
  assert all(np.allclose(asm.mate_axes[f"obj_{i}"], [0.0, 0.0, 1.0])
             for i in range(1, 5))
  C = P.make_subassembly_scene(asm, tuple(asm.objects))
  C.computeCollisions()
  assert C.getCollisionFree(), C.getCollisions()


def test_peg_stool_4_legs_round():
  asm = P.make_assembly("peg-stool-4", 5)
  assert len(asm.objects) == 5
  assert len(P._compound_children(asm.part_shape("obj_0"))) == 17
  for i in range(1, 5):
    assert asm.part_shape(f"obj_{i}")[0] == "cylinder"
  C = P.make_subassembly_scene(asm, tuple(asm.objects))
  C.computeCollisions()
  assert C.getCollisionFree(), C.getCollisions()


def test_peg_stool_tight_clearance_kinds():
  import pytest as _pytest
  # round peg: radius = hole/2 - c
  asm = P.make_assembly("peg-stool-tight", 4)
  kind, prm = asm.part_shape("obj_1")[:2]
  assert kind == "cylinder"
  assert abs(prm[1] - (asm.socket_hole / 2.0 - asm.peg_clearance)) < 1e-9
  # square peg: side = hole - c*sqrt(2), so the CORNER gap is exactly the kind's c
  asm2 = P.make_assembly("peg-stool-cuboid-tight", 4)
  kind2, prm2 = asm2.part_shape("obj_1")[:2]
  assert kind2 == "box"
  assert abs(prm2[0] - (asm2.socket_hole - asm2.peg_clearance * np.sqrt(2.0))) < 1e-9
  assert abs((asm2.socket_hole - prm2[0]) / np.sqrt(2.0) - asm2.peg_clearance) < 1e-9
  # the tight stools still assemble collision-free
  for a in (asm, asm2):
    C = P.make_subassembly_scene(a, tuple(a.objects))
    C.computeCollisions()
    assert C.getCollisionFree(), C.getCollisions()
  # four-leg tight kind
  a4 = P.make_assembly("peg-stool-cuboid-4-tight", 5)
  assert len(a4.objects) == 5 and a4.part_shape("obj_1")[0] == "box"
  C4 = P.make_subassembly_scene(a4, tuple(a4.objects))
  C4.computeCollisions()
  assert C4.getCollisionFree(), C4.getCollisions()
  # an impossible square clearance raises at build time
  with _pytest.raises(AssertionError):
    P.PegStoolAssembly(leg_kind="box", peg_clearance=0.022)


def test_peg_stool_cuboid_rejects_oversized_square_legs():
  # side >= hole: the peg's DIAGONAL no longer clears the hole's diagonal
  import pytest as _pytest
  with _pytest.raises(AssertionError):
    P.PegStoolAssembly(leg_kind="box", leg_side=0.031)



def test_peg_stool_mujoco_keeps_the_socket_concavities():
  import sim as S
  asm = P.make_assembly("peg-stool", 4)
  poses = {}
  for part in asm.objects:
    pos = asm.position_in_base(part)
    poses[part] = np.r_[pos, [1.0, 0.0, 0.0, 0.0]]
  state = P.State(parent={"obj_0": ("table",),
                          **{f"obj_{i}": ("part", "obj_0") for i in range(1, 4)}},
                  supports={"obj_0": frozenset({("table",)}),
                            **{f"obj_{i}": frozenset({("mate", "obj_0")})
                               for i in range(1, 4)}},
                  poses=poses)
  xml = S.build_mjcf(asm, state, ["a1"])
  seat_body = xml.split('<body name="obj_0"', 1)[1].split("</body>", 1)[0]
  assert seat_body.count("<geom") == 13
  assert all(xml.split(f'<body name="obj_{i}"', 1)[1].split("</body>", 1)[0]
             .count('type="cylinder"') == 1 for i in range(1, 4))
