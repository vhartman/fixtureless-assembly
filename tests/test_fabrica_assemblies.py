"""Fabrica CAD imports: frame conversion, reviewed semantics, and backend parity."""

import os

import mujoco
import numpy as np

import assembly_planner.problem as P
import sim as S
from assembly_planner.env import make_world


FABRICA_MODELS = (
    ("fabrica-car", 5),
    ("fabrica-beam", 5),
    ("fabrica-cooling-manifold", 7),
    ("fabrica-duct", 8),
    ("fabrica-gamepad", 6),
    ("fabrica-plumbers-block", 5),
    ("fabrica-circular-stool", 9),
)


def _collision_child_mesh(shape):
  """Trimesh equivalent of one convex collision child (in its child frame)."""
  import trimesh
  if shape[0] == "mesh":
    return P._load_mesh(shape[1])["hull_mesh"]
  if shape[0] == "box":
    return trimesh.creation.box(extents=np.asarray(shape[1][:3], dtype=float))
  if shape[0] == "cylinder":
    return trimesh.creation.cylinder(
        radius=float(shape[1][1]), height=float(shape[1][0]), sections=32)
  raise AssertionError(f"unsupported collision child {shape[0]!r}")


def _assembled_scene(kind, n_parts):
  asm = P.make_assembly(kind, n_parts)
  C = P.add_disassembled_obj(
      make_world(), asm, center_scatter=True, float_height=0.002)
  table = C.getFrame("table")
  table_top = table.getPosition()[2] + table.getSize()[2] / 2.0
  rel = {p: asm.position_in_base(p) for p in asm.objects}
  zmin = min(x[2] for p in asm.objects for x in P._part_support_points_world(
      np.r_[rel[p], [1.0, 0.0, 0.0, 0.0]], asm.part_shape(p)))
  for p in asm.objects:
    C.getFrame(p).setPosition(
        [rel[p][0], rel[p][1], table_top - zmin + rel[p][2]])
    C.getFrame(p).setQuaternion([1.0, 0.0, 0.0, 0.0])
  return asm, C


def test_fabrica_source_mapping_and_dependencies():
  car = P.make_assembly("fabrica-car", 5)
  assert car.base == "obj_0"
  assert [os.path.basename(car.visual_meshes[p]) for p in car.objects] == [
      "1.obj", "0.obj", "3.obj", "4.obj", "5.obj"]
  assert car.assembly_dependencies == {
      "obj_1": ["obj_0"], "obj_2": ["obj_0"],
      "obj_3": ["obj_0"], "obj_4": ["obj_0"]}
  assert all("2.obj" not in path for path in car.visual_meshes.values())
  assert np.allclose(car.position_in_base("obj_1"), [-.05606, .0225, -.02157], atol=2e-5)
  assert np.allclose(car.mate_axis("obj_1"), [0, -1, 0])
  assert np.allclose(car.mate_axis("obj_3"), [0, 1, 0])
  assert set(car.mate_symmetry_quats) == set(car.objects[1:])
  assert all(len(car.mate_quaternions(p)) == 1 for p in car.objects[1:])
  car.configure_mate_symmetry(True)
  assert all(len(car.mate_quaternions(p)) == 2 for p in car.objects[1:])
  assert all(np.allclose(car.mate_quaternions(p)[1], [0, 0, 1, 0])
             for p in car.objects[1:])
  # A wheel is one rounded tread plus one coaxial locating peg.  Hundreds of tetrahedral
  # cells for this surface of revolution make every MP predicate needlessly expensive.
  for source in ("0", "3", "4", "5"):
    collision = car._manifest["parts"][source]["collision"]
    assert collision["method"] == "primitive-inner"
    assert collision["hull_count"] == 2
    assert collision["settings"]["primitive"] == "two_coaxial_cylinders"
    assert collision["settings"]["mate_translation_envelope_m"] >= .001

  beam = P.make_assembly("fabrica-beam", 5)
  assert [os.path.basename(beam.visual_meshes[p]) for p in beam.objects] == [
      "6.obj", "2.obj", "0.obj", "3.obj", "1.obj"]
  assert beam.assembly_dependencies == {
      "obj_1": ["obj_0"], "obj_2": ["obj_1"],
      "obj_3": ["obj_0"], "obj_4": ["obj_3"]}
  assert all(np.allclose(beam.mate_axis(p), [0, 0, -1]) for p in beam.objects[1:])
  # Manufactured mortised beams do not need hundreds of generic convex cells.  The exact
  # CAD remains the visual/grasp surface; the reviewed proxy retains every insertion hole.
  assert [len(P._compound_children(beam.part_shape(p)))
          for p in beam.objects] == [7, 3, 10, 3, 10]

  expected_sources = {
      "fabrica-cooling-manifold": ["1", "0", "4", "2", "3", "5", "6"],
      "fabrica-duct": ["0", "1", "2", "3", "4", "5", "6", "7"],
      "fabrica-gamepad": ["3", "1", "2", "4", "0", "5"],
      "fabrica-plumbers-block": ["2", "3", "0", "1", "4"],
      "fabrica-circular-stool": ["2", "1", "0", "3", "6", "5", "8", "7", "4"],
  }
  for kind, sources in expected_sources.items():
    asm = P.make_assembly(kind, len(sources))
    assert [os.path.splitext(os.path.basename(asm.visual_meshes[p]))[0]
            for p in asm.objects] == sources
    assert set(asm.assembly_dependencies) == set(asm.objects[1:])
    # Every prefix is a connected rooted subtree, which is the contract used by --parts.
    assert all(asm.assembly_dependencies[p][0] in asm.objects[:i]
               for i, p in enumerate(asm.objects[1:], 1))

  manifold = P.make_assembly("fabrica-cooling-manifold", 7)
  # Grasp generation uses the authored surface rather than artificial internal ACD seams.
  assert manifold.grasp_shape("obj_3")[0] == "mesh"
  assert manifold.part_shape("obj_3")[0] == "compound"


def test_gamepad_square_pins_have_opt_in_quarter_turn_mate_symmetry():
  asm = P.make_assembly("fabrica-gamepad", 6)
  # Reviewed source order is (3,1,2,4,0,5): the last two objects are the two
  # geometrically/functionally fourfold square pins.
  assert set(asm.mate_symmetry_quats) == {"obj_4", "obj_5"}
  for part in ("obj_4", "obj_5"):
    assert len(asm.mate_quaternions(part)) == 1           # opt-in remains compatible
  asm.configure_mate_symmetry(True)
  for part in ("obj_4", "obj_5"):
    quats = asm.mate_quaternions(part)
    assert len(quats) == 4
    for k, q in enumerate(quats):
      expected = np.array([np.cos(k * np.pi / 4.0), 0.0, 0.0,
                           np.sin(k * np.pi / 4.0)])
      assert np.allclose(q, expected)


def test_circular_stool_pegs_and_caps_have_authored_axial_mate_symmetry():
  asm = P.make_assembly("fabrica-circular-stool", 9)
  # Reviewed source order is (2,1,0,3,6,5,8,7,4): odd obj ids are the four
  # hexagonal long pegs, while the following even ids are their square caps.
  pegs = ("obj_1", "obj_3", "obj_5", "obj_7")
  caps = ("obj_2", "obj_4", "obj_6", "obj_8")
  assert set(asm.mate_symmetry_quats) == set(pegs + caps)
  assert all(len(asm.mate_quaternions(part)) == 1 for part in pegs + caps)
  asm.configure_mate_symmetry(True)
  for part in pegs:
    quats = asm.mate_quaternions(part)
    assert len(quats) == 6
    for k, q in enumerate(quats):
      assert np.allclose(q, [np.cos(k * np.pi / 6.0), 0.0, 0.0,
                             np.sin(k * np.pi / 6.0)])
  for part in caps:
    quats = asm.mate_quaternions(part)
    assert len(quats) == 4
    for k, q in enumerate(quats):
      assert np.allclose(q, [np.cos(k * np.pi / 4.0), 0.0, 0.0,
                             np.sin(k * np.pi / 4.0)])


def test_cooling_manifold_hex_fasteners_have_sixfold_mate_symmetry():
  asm = P.make_assembly("fabrica-cooling-manifold", 7)
  # Reviewed source order is (1,0,4,2,3,5,6): only the final four objects are the
  # exact hexagonal fasteners.
  fasteners = ("obj_3", "obj_4", "obj_5", "obj_6")
  assert set(asm.mate_symmetry_quats) == set(fasteners)
  assert all(len(asm.mate_quaternions(part)) == 1 for part in fasteners)
  asm.configure_mate_symmetry(True)
  for part in fasteners:
    quats = asm.mate_quaternions(part)
    assert len(quats) == 6
    for k, q in enumerate(quats):
      assert np.allclose(q, [np.cos(k * np.pi / 6.0), 0.0, 0.0,
                             np.sin(k * np.pi / 6.0)])


def test_duct_fasteners_are_not_infinite_strength_handles():
  asm = P.make_assembly("fabrica-duct", 8)
  # obj_1 is the load-bearing cover.  The remaining six children are small
  # fasteners whose authored finite mates must not carry the complete duct.
  assert set(asm.mate_wrench_limits) == {
      "obj_2", "obj_3", "obj_4", "obj_5", "obj_6", "obj_7"}
  for limit in asm.mate_wrench_limits.values():
    assert np.allclose(limit["force"], [0.25, 0.25, 0.25])
    assert np.allclose(limit["torque"], [0.005, 0.005, 0.005])
  assert sum(limit["force"][2] for limit in asm.mate_wrench_limits.values()) \
      < 9.81 * sum(asm.masses.values())


def test_fabrica_scattered_collision_and_mate_openings():
  for kind, n_parts in FABRICA_MODELS:
    asm = P.make_assembly(kind, n_parts)
    C = P.add_disassembled_obj(
        make_world(), asm, center_scatter=True, float_height=0.002)
    C.computeCollisions()
    assert all(d >= -1e-6 for _a, _b, d in C.getCollisions())

    # The insertion centre must remain outside every convex component of its parent's
    # collider. A whole-part convex hull fails this immediately by filling the bore/socket.
    for part in asm.objects[1:]:
      parent = asm.assembly_dependencies[part][0]
      point = asm.position_in_base(part) - asm.position_in_base(parent)
      parent_shape = asm.part_shape(parent)
      assert parent_shape[0] == "compound"
      imported_inside = any(_collision_child_mesh(sub).contains([point - pos])[0]
                            for sub, pos in P._compound_children(parent_shape))
      import trimesh
      exact = trimesh.load(os.path.join(P.HERE, asm.visual_meshes[parent]),
                           force="mesh", process=True)
      if exact.is_watertight:
        # Preserve the authored topology at the mate centre: holes remain holes, while a
        # cap intentionally centred within the end of an upright remains material.
        # Inner approximations may omit a little material, but may never fill a
        # point which the authored CAD leaves empty. Full sweeps are checked below.
        assert not imported_inside or bool(exact.contains([point])[0]), (kind, part)
      elif kind == "fabrica-circular-stool" and parent == asm.base:
        # This source OBJ has duplicated/non-manifold decorative shells, so trimesh cannot
        # assign an exact inside sign. The four leg axes are nevertheless authored openings.
        assert not imported_inside, (kind, part)


def test_fabrica_collision_decomposition_retains_authored_extents():
  """Pegs/pins are collision, not renderer-only decoration or trimmed final-state caps."""
  for kind, n_parts in FABRICA_MODELS:
    asm = P.make_assembly(kind, n_parts)
    for part in asm.objects:
      visual = P._visual_mesh_data(asm.visual_meshes[part])[0]
      visual_half = (visual.max(axis=0) - visual.min(axis=0)) / 2.0
      source = os.path.splitext(os.path.basename(asm.visual_meshes[part]))[0]
      collision = asm._manifest["parts"][source]["collision"]
      settings = collision.get("settings", {})
      if settings.get("explicit_approximation", False):
        # User-approved primitive proxies can intentionally omit decorative radial
        # geometry (for example the cooling-manifold peg's threaded/nut profile).
        # They must still retain the complete insertion axis; otherwise the old
        # renderer-only peg regression could return unnoticed.
        axis = settings["axis"]
        if isinstance(axis, int):
          visual_axial_half = visual_half[axis]
          collision_axial_half = P._shape_half_extents(asm.part_shape(part))[axis]
        else:
          direction = np.asarray(axis, dtype=float)
          direction /= np.linalg.norm(direction)
          exact_projection = visual @ direction
          visual_axial_half = (exact_projection.max() - exact_projection.min()) / 2
          points = []
          for shape, offset in P._compound_children(asm.part_shape(part)):
            points.append(_collision_child_mesh(shape).vertices + offset)
          collision_projection = np.concatenate(points) @ direction
          collision_axial_half = (collision_projection.max()
                                  - collision_projection.min()) / 2
        assert np.isclose(collision_axial_half, visual_axial_half, atol=2e-3), (kind, part)
        assert settings.get("approximation_reason"), (kind, part)
        continue
      # Contained VHACD and explicit manufacturing clearances may retreat an authored
      # extremum slightly.  The circular-stool legs declare a 2.5mm radial fit retreat; add
      # 0.5mm for the contained decomposition's own boundary loss.  This still catches the
      # old failure mode where an entire peg/stem was discarded, without demanding a
      # collision over-approximation.
      extent_atol = 3e-3 if settings.get("axial_fit_trim") else 2e-3
      assert np.allclose(P._shape_half_extents(asm.part_shape(part)), visual_half,
                         atol=extent_atol), (kind, part)
  stool = P.make_assembly("fabrica-circular-stool", 9)
  assert stool.part_shape(stool.base)[0] == "compound"    # never regress to a round proxy
  duct = P.make_assembly("fabrica-duct", 8)
  assert len(duct.part_shape(duct.base)[1]) > 1            # elbow/shell, not one enclosing box


def test_fabrica_semantic_insert_proxies_are_local_and_small():
  """Primitive replacements preserve heads/bodies and reduce only male mating features."""
  expected = {
      "fabrica-beam": ({"2", "3"}, 3),
      "fabrica-duct": ({"2", "3", "4", "5", "6", "7"}, 2),
      "fabrica-gamepad": ({"0", "5"}, 2),
      "fabrica-plumbers-block": ({"0", "1", "4"}, None),
  }
  for kind, (sources, count) in expected.items():
    asm = P.make_assembly(kind, next(n for name, n in FABRICA_MODELS if name == kind))
    for source in sources:
      collision = asm._manifest["parts"][source]["collision"]
      settings = collision["settings"]
      assert collision["method"] == "primitive-approved"
      if count is not None:
        assert collision["hull_count"] == count
      assert settings["manufacturing_clearance_m"]["radial"] > 0
      assert settings["clearance_scope"].startswith("male insertion feature only")


def test_cooling_manifold_keeps_socket_rings_and_screw_heads():
  asm = P.make_assembly("fabrica-cooling-manifold", 7)
  base_source = os.path.splitext(os.path.basename(asm.visual_meshes[asm.base]))[0]
  rings = asm._manifest["parts"][base_source]["collision"]["settings"][
      "semantic_socket_rings"]
  assert rings["count"] == 4
  assert rings["sections_each"] >= 8
  for part in asm.objects:
    source = os.path.splitext(os.path.basename(asm.visual_meshes[part]))[0]
    settings = asm._manifest["parts"][source]["collision"].get("settings", {})
    if settings.get("primitive") == "central_peg_cylinder_plus_hex_head":
      assert len(asm.part_shape(part)[1]) == 2
      assert settings["head_radius_m"] > settings["stem_radius_m"]


def test_cooling_manifold_loose_hex_fasteners_rest_on_a_face():
  """The authored scatter pose must not roll before its planned pick.

  The primitive proxy has a hexagonal head.  Laying its axis down without the matching
  30-degree roll balances the head on a vertex; MuJoCo then changes the grasp datum even
  though symbolic planning quite reasonably treats a resting table pose as stationary.
  """
  asm = P.make_assembly("fabrica-cooling-manifold", 7)
  C = P.add_disassembled_obj(
      make_world(), asm, center_scatter=True, spacing=.18, float_height=.015)
  state = P.get_initial_state(C, asm)
  model = mujoco.MjModel.from_xml_string(S.build_mjcf(asm, state, []))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  initial = S._state_part_poses(state, asm)
  for _ in range(1000):
    mujoco.mj_step(model, data)
  # obj_3..obj_6 are source fasteners 2,3,5,6 in the reviewed source order.
  for part in asm.objects[3:]:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
    got = np.r_[data.xpos[body], data.xquat[body]]
    assert np.linalg.norm(got[:2] - initial[part][:2]) < 1e-3
    assert S._part_orientation_error(asm, part, got[3:], initial[part][3:]) < np.deg2rad(2)


def test_circular_stool_fixture_free_legs_start_upright_and_stable():
  """Removing Fabrica's stock fixture must not leave a Robotiq-unpickable table pose."""
  asm = P.make_assembly("fabrica-circular-stool", 9)
  for part in ("obj_1", "obj_3", "obj_5", "obj_7"):
    np.testing.assert_allclose(asm.scatter_orientations[part], [1.0, 0.0, 0.0, 0.0])
  C = P.add_disassembled_obj(
      make_world(), asm, center_scatter=True, spacing=.18, float_height=.002)
  state = P.get_initial_state(C, asm)
  model = mujoco.MjModel.from_xml_string(
      S.build_mjcf(asm, state, [], object_damping=.001))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  initial = S._state_part_poses(state, asm)
  for _ in range(5000):
    mujoco.mj_step(model, data)
  for part in ("obj_1", "obj_3", "obj_5", "obj_7"):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
    got = np.r_[data.xpos[body], data.xquat[body]]
    assert np.linalg.norm(got[:2] - initial[part][:2]) < 1e-3
    assert S._part_orientation_error(asm, part, got[3:], initial[part][3:]) < np.deg2rad(2)


def test_object_damping_is_explicit_and_opt_in():
  asm = P.make_assembly("fabrica-cooling-manifold", 3)
  C = P.add_disassembled_obj(
      make_world(), asm, center_scatter=True, spacing=.18, float_height=.002)
  state = P.get_initial_state(C, asm)
  plain = S.build_mjcf(asm, state, [])
  damped = S.build_mjcf(asm, state, [], object_damping=.001)
  assert plain.count("<freejoint/>") == len(asm.objects)
  assert 'type="free" damping="0.001"' not in plain
  assert damped.count('type="free" damping="0.001"') == len(asm.objects)


def test_fabrica_exact_visuals_are_noncolliding_in_mujoco():
  for kind, n_parts in FABRICA_MODELS:
    asm = P.make_assembly(kind, n_parts)
    C = P.add_disassembled_obj(
        make_world(), asm, center_scatter=True, float_height=0.002)
    model = mujoco.MjModel.from_xml_string(
        S.build_mjcf(asm, P.get_initial_state(C, asm), []))
    assert model.nmesh > n_parts       # exact visuals plus named convex collision components
    for p in asm.objects:
      bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p)
      gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, p + "__visual")
      assert gid >= 0
      assert model.geom_contype[gid] == model.geom_conaffinity[gid] == 0
      assert model.geom_bodyid[gid] == bid
      collision = [g for g in range(model.ngeom)
                   if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                   .startswith(p + "__collision")]
      assert collision
      assert all(model.geom_bodyid[g] == bid for g in collision)
      assert all(model.geom_group[g] == S.GROUP_COLL for g in collision)
      assert model.geom_group[gid] == S.GROUP_VISUAL


def test_fabrica_insertion_sweeps():
  """A clear bore centre is insufficient: the entire mover must fit along the path."""
  from tools.validation.audit_fabrica import audit
  for name in ("car", "beam", "duct", "cooling-manifold", "gamepad", "plumbers-block",
               "circular-stool"):
    for row in audit(name, step=.0025):
      assert row["rai_bad_samples"] == 0, (name, row)
      assert row["mujoco_bad_samples"] == 0, (name, row)


def test_car_mates_cover_the_witness_residual_envelope():
  """The 1mm IK mate gate must not create a collision after its exemption is removed."""
  from tools.validation.audit_fabrica import audit
  for row in audit("car", step=.005, tolerance=1e-4, robustness=.001):
    assert row["robust_bad"] == 0, row


def test_duct_cover_starts_on_a_stable_face():
  """Do not scatter the tilted CAD cover on its edge and let it fall during execution."""
  asm = P.make_assembly("fabrica-duct", 8)
  q = asm.scatter_orientations["obj_1"]
  normal = P._quat_to_R(q)[2]
  rests = P.stable_rest_poses(asm.grasp_shape("obj_1"), min_prob=0.0)
  assert any(np.dot(normal, P._quat_to_R(rest[0])[2]) > 1 - 1e-6 for rest in rests)


def test_manifold_fin_envelopes_leave_connectors_open():
  asm = P.make_assembly("fabrica-cooling-manifold", 7)
  metadata = asm._manifest["parts"]
  boxes = metadata["1"]["collision"]["approximation_boxes"]
  assert boxes
  base_center = np.array(metadata["1"]["center_cm"]) * .01
  for source in ("0", "4"):
    center = np.array(metadata[source]["center_cm"]) * .01 - base_center
    half = np.array(metadata[source]["extents_cm"]) * .005
    for box in boxes:
      lo, hi = np.array(box["lower"]), np.array(box["upper"])
      # Keep the entire connector's XY footprint clear, not just its centre line.
      overlap = np.minimum(hi[:2], center[:2] + half[:2]) - np.maximum(
          lo[:2], center[:2] - half[:2])
      assert np.any(overlap <= 0), (source, box)
