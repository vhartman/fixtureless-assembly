"""Chamfered peg-hole / peg-stool kinds: 45-degree frustum insertion tips.

The chamfer is a peg-side lead-in: the insertion end tapers by 4mm over 4mm, realized
as an inline convex mesh ({"verts": ...}) child on a compound peg/leg. The socket and
every clearance stay unchanged; the tip only widens the alignment tolerance of a real
insertion (a slightly misaligned descent slides along the 45-degree facet instead of
catching the square rim). These tests pin the geometry, the inline-mesh plumbing, the
rest-face equivalence, and the MuJoCo emission.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import problem as P


@pytest.fixture(autouse=True)
def _no_sampler_leak():
  yield
  P.set_sampled_grasps(None)


def _children(spec):
  return P._compound_children(spec)


def test_chamfer_tip_round_preserves_length_and_tapers():
  body, tip = P._chamfer_tip(("cylinder", [0.055, 0.012]), 0.004, -1.0)
  assert body[0][0] == "cylinder"
  assert body[0][1][1] == pytest.approx(0.012)
  assert body[0][1][0] == pytest.approx(0.051)              # length - chamfer
  assert np.allclose(body[1], [0.0, 0.0, 0.002])
  assert tip[0][0] == "mesh"
  V = np.asarray(tip[0][1]["verts"])
  assert V.shape == (24, 3)
  # inner (base) ring at -c/2 with the FULL radius; outer (end) ring at +c/2 reduced by c.
  # end=-1: the tip sits at the -z end of the part frame.
  assert np.allclose(tip[1], [0.0, 0.0, -0.0255])
  zs = np.unique(np.round(V[:, 2], 9))
  assert np.allclose(zs, [-0.002, 0.002])
  # end=-1: the FULL-section base ring sits on the inner (+z) side, the reduced end
  # ring on the outer (-z) side.
  base_rad = np.linalg.norm(V[V[:, 2] > 0.0][:, :2], axis=1)
  end_rad = np.linalg.norm(V[V[:, 2] < 0.0][:, :2], axis=1)
  assert np.allclose(base_rad, 0.012)
  assert np.allclose(end_rad, 0.008)
  # total extent preserved
  spec = ("compound", [{"shape": body[0], "pos": list(body[1])},
                       {"shape": tip[0], "pos": list(tip[1])}])
  assert np.allclose(P._shape_half_extents(spec),
                     P._shape_half_extents(("cylinder", [0.055, 0.012])))


def test_chamfer_tip_square_preserves_length_and_tapers():
  body, tip = P._chamfer_tip(("box", [0.02, 0.02, 0.11, 0.0]), 0.004, 1.0)
  assert body[0] == ("box", [0.02, 0.02, 0.106, 0.0])
  assert np.allclose(body[1], [0.0, 0.0, -0.002])
  V = np.asarray(tip[0][1]["verts"])
  assert V.shape == (8, 3)
  zs = np.unique(np.round(V[:, 2], 9))
  assert np.allclose(zs, [-0.002, 0.002])                  # inner square full, outer reduced
  assert max(abs(V[V[:, 2] < 0.0][:, 0])) == pytest.approx(0.010)   # base half-side
  assert max(abs(V[V[:, 2] > 0.0][:, 0])) == pytest.approx(0.006)   # end half-side (s-2c)/2
  assert np.allclose(tip[1], [0.0, 0.0, 0.053])
  with pytest.raises(AssertionError):
    P._chamfer_tip(("box", [0.02, 0.02, 0.11, 0.0]), 0.011, 1.0)   # consumes the section
  with pytest.raises(AssertionError):
    P._chamfer_tip(("cylinder", [0.055, 0.012]), 0.012, 1.0)       # c >= radius


def test_inline_verts_mesh_plumbing():
  V = np.array([[-0.02, -0.02, -0.02], [0.02, -0.02, -0.02], [0.02, 0.02, -0.02],
                [-0.02, 0.02, -0.02], [-0.02, -0.02, 0.02], [0.02, -0.02, 0.02],
                [0.02, 0.02, 0.02], [-0.02, 0.02, 0.02]], dtype=float)
  prm = {"verts": V}
  assert P._mesh_key(prm)[0] == "verts"
  assert P._mesh_key({"verts": V.copy()}) == P._mesh_key(prm)      # content-addressed
  loaded = P._load_mesh(prm)
  assert loaded["hull"].shape == (8, 3)
  assert np.allclose(loaded["half"], [0.02, 0.02, 0.02])           # centred by the loader
  import sim as S
  xml = S._geom_xml(("mesh", prm), 0.8)
  assert 'type="mesh"' in xml
  # the MuJoCo asset generator writes the hull OBJ for the inline verts
  name, path = S._part_mesh_asset(prm)
  assert os.path.exists(path)


def test_peg_hole_chamfer_kind():
  asm = P.make_assembly("peg-hole-chamfer", 2)
  spec = asm.part_shape("obj_1")
  assert spec[0] == "compound" and len(_children(spec)) == 2
  body, tip = _children(spec)
  assert body[0][0] == "cylinder" and tip[0][0] == "mesh"
  # same envelope as the plain peg -> same scatter height / rest geometry
  plain = P.make_assembly("peg-hole", 2)
  assert plain.part_shape("obj_1")[0] == "cylinder"               # default kind unchanged
  assert np.allclose(P._shape_half_extents(spec),
                     P._shape_half_extents(plain.part_shape("obj_1")))
  # the socket keeps its five convex pieces plus four 45-degree entry-funnel prisms
  seat = _children(asm.part_shape("obj_0"))
  assert len(seat) == 9
  assert sum(1 for sub, _ in seat if sub[0] == "mesh") == 4
  C = P.make_subassembly_scene(asm, tuple(asm.objects))
  C.computeCollisions()
  assert C.getCollisionFree(), C.getCollisions()


def test_peg_hole_chamfer_mujoco_geoms():
  import sim as S
  asm = P.make_assembly("peg-hole-chamfer", 2)
  poses = {
      "obj_0": np.array([0.0, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]),
      "obj_1": np.array([0.0, 0.0, 0.038, 1.0, 0.0, 0.0, 0.0]),
  }
  state = P.State(parent={"obj_0": ("table",), "obj_1": ("part", "obj_0")},
                  supports={"obj_0": frozenset({("table",)}),
                            "obj_1": frozenset({("mate", "obj_0")})},
                  poses=poses)
  xml = S.build_mjcf(asm, state, ["a1"])
  socket_body = xml.split('<body name="obj_0"', 1)[1].split("</body>", 1)[0]
  peg_body = xml.split('<body name="obj_1"', 1)[1].split("</body>", 1)[0]
  assert socket_body.count("<geom") == 9                 # 5 pieces + 4 entry-funnel prisms
  assert socket_body.count('type="mesh"') == 4
  assert peg_body.count("<geom") == 2                         # body cylinder + frustum tip
  assert peg_body.count('type="mesh"') == 1 and 'type="cylinder"' in peg_body


def test_peg_stool_chamfer_kinds():
  # plain seat (1 plate + 4 walls per socket) + 4 entry-funnel prisms per socket
  for kind, max_parts, seat_children in (("peg-stool-chamfer", 4, 25),
                                         ("peg-stool-4-chamfer", 5, 33),
                                         ("peg-stool-cuboid-chamfer", 4, 25),
                                         ("peg-stool-cuboid-4-chamfer", 5, 33)):
    asm = P.make_assembly(kind, max_parts)
    seat = _children(asm.part_shape("obj_0"))
    assert len(seat) == seat_children
    assert sum(1 for sub, _ in seat if sub[0] == "mesh") == 4 * (max_parts - 1)
    for i in range(1, max_parts):
      spec = asm.part_shape(f"obj_{i}")
      assert spec[0] == "compound" and len(_children(spec)) == 2
      body, tip = _children(spec)
      assert body[0][0] == asm.leg_kind and tip[0][0] == "mesh"
      assert np.allclose(asm.mate_axes[f"obj_{i}"], [0.0, 0.0, 1.0])
    C = P.make_subassembly_scene(asm, tuple(asm.objects))
    C.computeCollisions()
    assert C.getCollisionFree(), C.getCollisions()


def test_peg_stool_tight_chamfer_keeps_the_clearances():
  asm = P.make_assembly("peg-stool-tight-chamfer", 4)
  body, _ = _children(asm.part_shape("obj_1"))
  assert body[0][0] == "cylinder"
  assert abs(body[0][1][1] - (asm.socket_hole / 2.0 - asm.peg_clearance)) < 1e-9
  asm2 = P.make_assembly("peg-stool-cuboid-tight-chamfer", 4)
  body2, _ = _children(asm2.part_shape("obj_1"))
  assert body2[0][0] == "box"
  assert abs(body2[0][1][0] - (asm2.socket_hole - asm2.peg_clearance * np.sqrt(2.0))) < 1e-9
  for a in (asm, asm2):
    C = P.make_subassembly_scene(a, tuple(a.objects))
    C.computeCollisions()
    assert C.getCollisionFree(), C.getCollisions()


def test_hole_chamfer_geometry():
  # peg-hole numbers: h=0.016, wall=0.024, rim at +0.005, c=0.006, opening UP
  kids = P._hole_chamfer_children(0.016, 0.024, 0.04, 0.005, 0.006, +1.0, 0.08, 0.032)
  assert len(kids) == 4
  for spec, pos in kids:
    assert spec[0] == "mesh"
    V = np.asarray(spec[1]["verts"]) + pos
    # each prism stays inside the wall footprint, across in [h, h+wall]
    assert np.all((V[:, :2].min(axis=0) >= -0.040 - 1e-9) &
                  (V[:, :2].max(axis=0) <= 0.040 + 1e-9))
    # only the chamfer band z in [z_rim - c, z_rim] is occupied
    assert V[:, 2].min() == pytest.approx(-0.001)
    assert V[:, 2].max() == pytest.approx(0.005)
  # the +x wall prism: inner facet from (h, z_rim - c) to (h + c, z_rim) -- a 45-degree
  # widening; the mouth half-width at the rim plane is exactly h + c.
  spec, pos = kids[0]
  assert np.allclose(pos, [0.028, 0.0, 0.002])               # across centre h+wall/2
  V = np.asarray(spec[1]["verts"]) + pos
  assert np.allclose(V[:, 0].min(), 0.016)                   # facet bottom edge at h
  top = V[np.isclose(V[:, 2], 0.005)]
  assert np.allclose(np.sort(np.unique(top[:, 0])), [0.022, 0.040])  # h+c and h+wall
  assert any(abs(v[0] - 0.016) < 1e-9 and abs(v[2] + 0.001) < 1e-9 and
             abs(abs(v[1]) - 0.04) < 1e-9 for v in V)       # the A corner (h, z_rim-c) exists
  # the pocket is never narrowed below the original h: nothing sits across < h
  assert V[:, 0].min() == pytest.approx(0.016)
  # the -x wall prism is the MIRROR: facet from (-h, z_rim - c) to (-(h+c), z_rim) -- the
  # guide always faces the pocket, never the wall's outer face.
  spec2, pos2 = kids[1]
  assert np.allclose(pos2, [-0.028, 0.0, 0.002])
  V2 = np.asarray(spec2[1]["verts"]) + pos2
  assert np.allclose(V2[:, 0].max(), -0.016)                # inner face at -h
  assert np.allclose(V2[:, 0].min(), -0.040)                # outer face at -(h+wall)
  top2 = V2[np.isclose(V2[:, 2], 0.005)]
  assert np.allclose(np.sort(np.unique(top2[:, 0])), [-0.040, -0.022])  # -(h+wall), -(h+c)
  # the -y wall prism is mirrored the same way (across -> part y)
  spec3, pos3 = kids[3]
  assert np.allclose(pos3, [0.0, -0.028, 0.002])
  V3 = np.asarray(spec3[1]["verts"]) + pos3
  assert np.allclose(V3[:, 1].max(), -0.016)
  assert np.allclose(V3[:, 1].min(), -0.040)
  # impossible chamfers are rejected
  with pytest.raises(AssertionError):
    P._hole_chamfer_children(0.016, 0.024, 0.04, 0.005, 0.03, +1.0, 0.08, 0.032)


def test_hole_chamfer_stool_opens_down():
  kids = P._hole_chamfer_children(0.015, 0.005, 0.025, -0.034, 0.004, -1.0, 0.04, 0.030)
  assert len(kids) == 4
  spec, pos = kids[0]
  V = np.asarray(spec[1]["verts"]) + pos
  # rim at the BOTTOM: the funnel spans z in [z_rim, z_rim + c], widest at z_rim
  assert V[:, 2].min() == pytest.approx(-0.034)
  assert V[:, 2].max() == pytest.approx(-0.030)
  bottom = V[np.isclose(V[:, 2], -0.034)]
  assert np.allclose(np.sort(np.unique(bottom[:, 0])), [0.019, 0.02])  # h+c, h+wall
  assert V[:, 0].min() == pytest.approx(0.015)               # original half-hole h
  # the -x wall prism is the MIRROR: the facet faces the pocket from below
  spec2, pos2 = kids[1]
  V2 = np.asarray(spec2[1]["verts"]) + pos2
  assert np.allclose(V2[:, 0].max(), -0.015)                # inner face -h
  assert np.allclose(V2[:, 0].min(), -0.020)                # outer face -(h+wall)
  bottom2 = V2[np.isclose(V2[:, 2], -0.034)]
  assert np.allclose(np.sort(np.unique(bottom2[:, 0])), [-0.020, -0.019])  # -(h+wall), -(h+c)


def test_chamfered_round_leg_keeps_the_neutral_side_rest():
  # a lone chamfered round leg is still cylinder-like for rest purposes: the lying-side
  # placement (neutral rest) must remain available exactly as for the plain leg.
  plain = P.make_assembly("peg-stool", 4)
  chamf = P.make_assembly("peg-stool-chamfer", 4)
  assert set(P.rest_faces(chamf, ("obj_1",))) == set(P.rest_faces(plain, ("obj_1",)))
