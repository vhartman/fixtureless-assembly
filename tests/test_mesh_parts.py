"""Mesh part shapes: ("mesh", {"file": ..., "scale": ...}[, quat]).

v1 contract: a mesh part COLLIDES AS ITS CONVEX HULL (rai and MuJoCo), the mesh is
re-centered on its AABB centre at load, and grasps REQUIRE --grasp-sampler (the 6-face
enumerator's ids are cube-face semantics). Validated end to end on the mesh-demo assembly:
search -> witnesses -> BiRRT -> MuJoCo, 6/6. These tests pin the fast pieces.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import problem as P

MESH = ("mesh", {"file": "part_meshes/chamfered_box.stl"})
_have_mesh = os.path.exists(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "part_meshes/chamfered_box.stl"))
pytestmark = pytest.mark.skipif(not _have_mesh, reason="demo mesh not present")


@pytest.fixture(autouse=True)
def _no_sampler_leak():
  yield
  P.set_sampled_grasps(None)                               # module state: never leak


def test_half_extents_and_lever():
  assert np.allclose(P._shape_half_extents(MESH), [0.025, 0.025, 0.025], atol=1e-6)
  assert abs(P._shape_half_extent_along(MESH, [0, 0, 1.0]) - 0.025) < 1e-6
  a = P.make_assembly("mesh-demo", 3)
  assert a.part_shape("obj_1")[0] == "mesh"
  assert abs(a.grasp_lever("obj_1") - 0.025) < 1e-6


def test_support_points_are_hull_vertices():
  pose = np.array([0.0, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0])
  pts = P._part_support_points_world(pose, MESH)
  assert len(pts) == 24                                    # the chamfered cube's hull
  assert min(x[2] for x in pts) == pytest.approx(0.0, abs=1e-6)


def test_rai_frame_is_convex_mesh():
  import robotic as ry
  C = ry.Config()
  P.add_part_frame(C, "m", MESH)
  assert C.getFrame("m").info()["shape"] == "mesh"


def test_enumerator_refuses_mesh_without_sampler():
  P.set_sampled_grasps(None)
  with pytest.raises(ValueError, match="grasp-sampler"):
    P.part_grasp_poses(MESH)


def test_sampled_views_classify_flats():
  P.set_sampled_grasps({"seed": 0, "candidates": 200, "keep": 12})
  poses, surfs, mins = P._sampled_grasp_views(MESH, 0.01)
  assert len(poses) == 12
  assert "flat" in surfs                                   # the big octagonal faces
  assert max(mins) == pytest.approx(0.05, abs=1e-3)        # their in-plane extent


def test_sampler_filter_rejects_buried_fingers():
  """The regression that killed motion planning: an off-center sampled FLAT grasp parks an
  open finger capsule 5-10mm inside the part; the DB filter must drop it (a clean grasp
  grazes <=0.5mm). Cubes show it too -- this is not mesh-specific."""
  P.set_sampled_grasps({"seed": 0, "candidates": 200, "keep": 12})
  asm = P.make_assembly("mesh-demo", 3)
  labeled = P.compute_cube_grasps(
      asm, ("obj_1",), return_labeled=True,
      gripper_path=os.path.join(os.path.dirname(os.path.abspath(P.__file__)),
                                "src/rai/ur10/ur10_two_finger_floating.g"))
  assert labeled, "some sampled grasps must survive the filter"
  import robotic as ry
  base = P.make_subassembly_scene(asm, ("obj_1",))
  base.addFile(os.path.join(os.path.dirname(os.path.abspath(P.__file__)),
                            "src/rai/ur10/ur10_two_finger_floating.g"), namePrefix="ee_")
  from env import maybe_tighten_fingers
  maybe_tighten_fingers(base, "ee_")
  _, brt, qc = P._gripper_rest_transforms(base, "ee_")
  for _obj, _gid, grasp in labeled:
    C = ry.Config()
    C.addConfigurationCopy(base)
    bw = P._ee_root_pose(np.asarray(grasp, dtype=float), brt, qc)
    r = C.getFrame("ee_ur_base")
    r.setPosition(bw[:3])
    r.setQuaternion(bw[3:])
    C.computeCollisions()
    worst = min([d for a, b, d in C.getCollisions()
                 if (a.startswith("ee_") or b.startswith("ee_"))
                 and ("obj_1" in (a, b) or P.part_shape_frame(C, "obj_1") in (a, b))],
                default=0.0)
    assert worst > -1.5e-3, f"gid {_gid} buried {-worst*1000:.1f}mm survived the filter"


# --------------------------------------------------------------------- edge clearance
def test_feature_edges_are_creases_not_tessellation():
  """A 48-section cylinder steps 7.5 deg per barrel quad. Those are NOT features -- only
  the two cap rims are. Getting this wrong (a 5 deg rule) made the margin gate read
  0.97mm of clearance ANYWHERE on the barrel and silently do nothing on curved shapes."""
  cyl = P._spec_mesh(("cylinder", [0.30, 0.015]))
  A, _B = P._mesh_feature_edges(cyl)
  assert len(A) == 96, f"expected 2x48 cap-rim edges, got {len(A)}"
  mid = P._edge_clearance(cyl, np.array([0.015, 0.0, 0.0]))[0]
  assert mid == pytest.approx(0.15, abs=1e-6)              # half the length, to a rim
  # a box: exactly its 12 edges; a sphere: none at all
  assert len(P._mesh_feature_edges(P._spec_mesh(("box", [0.05] * 3)))[0]) == 12
  assert len(P._mesh_feature_edges(P._spec_mesh(("sphere", [0.03])))[0]) == 0
  assert np.isinf(P._edge_clearance(P._spec_mesh(("sphere", [0.03])),
                                    np.array([0.03, 0.0, 0.0]))[0])


@pytest.mark.parametrize("spec", [
    ("box", [0.05, 0.05, 0.05, 0.005]),
    ("cylinder", [0.30, 0.015]),
    ("mesh", {"file": "part_meshes/chamfered_box.stl"}),
])
def test_margin_keeps_contacts_off_the_edges(spec):
  """Checked against the values the GATE recorded, not against a contact re-derived from the
  pose: `pose + offset*z` recovers the contact only for a SUCTION candidate. For a pinch it
  gives the approach-side surface point, which is a different place on the part and is under
  no obligation to clear the edges."""
  P.set_sampled_grasps({"seed": 0, "candidates": 200, "keep": 12, "margin": 0.008})
  meta = P.sampled_grasp_meta(spec)
  assert meta and min(m["edge_clearance"] for m in meta) >= 0.008 - 1e-9
  # spot-check the recorded value against the geometry for the family where the contact IS
  # recoverable, so the recording itself cannot drift unnoticed.
  poses, _s, _m = P._sampled_grasp_views(spec, 0.01)
  mesh = P._spec_mesh(spec)
  for pose, md in zip(poses, meta):
    if md["family"] != "suction":
      continue
    contact = pose[:3] + 0.01 * P._quat_to_R(pose[3:])[:, 2]
    assert P._edge_clearance(mesh, contact)[0] == pytest.approx(md["edge_clearance"], abs=1e-6)


def test_margin_falls_back_to_most_interior_when_unhonourable():
  """A shape too small for the margin must still yield `keep` grasps -- the most interior
  ones -- rather than raising or silently accepting corner grasps."""
  tiny = ("box", [0.01, 0.01, 0.01, 0.001])                # 10mm cube vs an 8mm margin
  P.set_sampled_grasps({"seed": 0, "candidates": 200, "keep": 12, "margin": 0.008})
  poses, _s, _m = P._sampled_grasp_views(tiny, 0.01)
  assert len(poses) == 12


# ------------------------------------------------------------------- pad (contact patch)
def test_pad_support_is_full_mid_face_and_partial_at_the_rim():
  """The measure itself: a cup in the middle of a 5cm face is fully backed; the same cup
  centred on the face RIM has about half its area over nothing."""
  mesh = P._spec_mesh(("box", [0.05, 0.05, 0.05, 0.005]))
  n = np.array([0.0, 0.0, 1.0])
  u, v = P._plane_axes(n)
  cup = P._pad_samples_disc(0.0125)
  mid = P._pad_support(mesh, np.array([0.0, 0.0, 0.025]), n, u, v, cup)
  rim = P._pad_support(mesh, np.array([0.025, 0.0, 0.025]), n, u, v, cup)
  assert mid == pytest.approx(1.0)
  assert 0.3 < rim < 0.7, f"a cup on the rim should be about half supported, got {rim}"


@pytest.mark.parametrize("spec", [
    ("box", [0.05, 0.05, 0.05, 0.005]),
    ("mesh", {"file": "part_meshes/chamfered_box.stl"}),
])
def test_kept_grasps_honour_the_pad_gate(spec):
  P.set_sampled_grasps({"seed": 0, "candidates": 200, "keep": 12,
                        "margin": 0.008, "pad_support": 0.75})
  meta = P.sampled_grasp_meta(spec)
  assert len(meta) == 12
  assert min(m["pad_support"] for m in meta) >= 0.75 - 1e-9
  assert min(m["edge_clearance"] for m in meta) >= 0.008 - 1e-9


def test_pad_gate_falls_back_per_family_on_a_thin_leg():
  """A 15mm-radius barrel cannot seal a 25mm cup anywhere, but it CAN back a finger pad laid
  along the axis. The fallback is therefore per FAMILY: the suction block keeps its
  best-supported candidates below the gate, the pinch block honours it, and the part still
  gets its full complement of grasps. Thin legs are exactly the chair case, so starving them
  would be the worst outcome."""
  # The current hybrid DB reserves its conventional source rows before the sampled families;
  # expose enough of the immutable master to exercise the sampled family fallback itself.
  P.set_sampled_grasps({"seed": 0, "candidates": 200, "keep": 64,
                        "margin": 0.008, "pad_support": 0.75})
  meta = P.sampled_grasp_meta(("cylinder", [0.30, 0.015]))
  assert meta, "the leg must not be starved of grasps"
  pinch = [m["pad_support"] for m in meta if m["family"] == "pinch"]
  suction = [m["pad_support"] for m in meta if m["family"] == "suction"]
  assert pinch and suction, "both families must still be represented"
  # a rigid 20mm-wide pad cannot conform to a 30mm barrel: its edges stand ~4mm off, past
  # the pad-compliance allowance, so a leg pinch tops out around 0.6 and the fallback keeps
  # the best of them. The cup does better -- it seals on the flat caps.
  assert max(pinch) < 0.75, f"a flat pad should not fully back a barrel: {pinch}"
  assert max(suction) >= 0.75, f"a cup should seal on the caps: {suction}"


def test_pinch_roll_lays_the_pads_along_the_part():
  """THE improvement behind the pad criterion: the roll about the closing axis used to be
  fixed with an arbitrary reference vector, so on a cylinder the 40mm pad ran AROUND the
  barrel (0.20 support) instead of along it. The roll is now chosen from the geometry."""
  m = P._spec_mesh(("cylinder", [0.30, 0.015]))
  p1 = np.array([0.015, 0.0, 0.02])
  p2 = np.array([-0.015, 0.0, 0.02])
  y = (p2 - p1) / np.linalg.norm(p2 - p1)
  u0, v0 = P._plane_axes(y)
  _across, along = P._best_pad_axes(m, p1, p2, y, u0, v0)
  axis = np.array([0.0, 0.0, 1.0])                         # the cylinder's own axis
  assert abs(float(along @ axis)) > 0.98, (
      f"the pad's long axis should lie along the cylinder, got {along}")
