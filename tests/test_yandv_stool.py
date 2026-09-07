"""Y&V stool: simplified collision plus exact non-colliding visuals in both backends."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import problem as P


def test_yandv_rai_visuals_and_lying_legs():
  from env import make_world

  asm = P.make_assembly("real-stool", 5)
  C = P.add_disassembled_obj(make_world(), asm, spacing=0.3, float_height=0.002)
  lying = np.array([np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0])
  for i, p in enumerate(asm.objects):
    visual = C.getFrame(p + "__visual")
    assert visual is not None and visual.info()["shape"] == "mesh"
    assert visual.info().get("contact") in (None, 0)
    assert len(np.asarray(visual.getMeshTriangles())) > 0
    if i:
      assert np.allclose(C.getFrame(p).getQuaternion(), lying)
  # Contact output can contain positive-distance proximity pairs; no penetration is present.
  C.computeCollisions()
  assert all(d >= -1e-6 for _a, _b, d in C.getCollisions())


def test_yandv_mujoco_uses_exact_visual_meshes_but_box_contacts():
  import mujoco
  import sim as S
  from env import make_world

  asm = P.make_assembly("real-stool", 5)
  C = P.add_disassembled_obj(make_world(), asm, spacing=0.3, float_height=0.002)
  xml = S.build_mjcf(asm, P.get_initial_state(C, asm), [])
  model = mujoco.MjModel.from_xml_string(xml)
  assert model.nmesh == 5                         # seat plus four handed leg meshes
  for p in asm.objects:
    vid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, p + "__visual")
    assert vid >= 0
    assert model.geom_type[vid] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_contype[vid] == model.geom_conaffinity[vid] == 0
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p)
    # A contact-disabled render mesh is not automatically inertia-disabled in MuJoCo.
    # Keep the CAD shell from silently replacing the assembly's authored physical mass.
    assert model.body_mass[body] == pytest.approx(asm.masses[p])
    physical = [i for i in np.flatnonzero(model.geom_bodyid == body)
                if model.geom_contype[i] != 0]
    assert physical and all(model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX for i in physical)


def test_real_stool_cad_alignment_and_legacy_alias():
  asm = P.make_assembly("real-stool", 5)
  legacy = P.make_assembly("yandv-stool", 5)
  assert asm.objects == legacy.objects
  # The source CAD puts a leg top 7.25mm above the main slab centre. Our collision mate
  # stops 0.5mm below the blind floor, hence 7.0mm (0.25mm from the CAD transform).
  for p in asm.objects[1:]:
    top = asm.assembly_relative_poses[p][1][2] + asm.leg_length / 2.0
    assert np.isclose(top, 0.007)
  children = P._compound_children(asm.part_shape("obj_0"))
  assert any(np.allclose(sub[1][:3], [0.170, 0.010, 0.040]) for sub, _ in children)


def test_real_stool_discrete_mate_symmetry_is_explicit_and_opt_in():
  asm = P.make_assembly("real-stool", 5)
  identity = np.array([1.0, 0.0, 0.0, 0.0])
  for part in asm.objects[1:]:
    assert len(asm.mate_symmetry_quats[part]) == 2       # authored representation exists
    assert len(asm.mate_quaternions(part)) == 1          # historical behavior until enabled
    assert np.allclose(asm.mate_quaternions(part)[0], identity)
  asm.configure_mate_symmetry(True)
  for part in asm.objects[1:]:
    quats = asm.mate_quaternions(part)
    assert len(quats) == 2
    # Two distinct rotations modulo quaternion's q == -q double cover.
    assert all(abs(float(np.dot(quats[i], quats[j]))) < 1.0 - 1e-8
               for i in range(2) for j in range(i))
    assert np.allclose(asm.closest_mate_quaternion(part, -quats[1]), quats[1])


def test_assemble_bind_tries_symmetry_members_inside_one_symbolic_action():
  import solver as S

  asm = P.make_assembly("real-stool", 2)
  asm.configure_mate_symmetry(True)
  rik = object.__new__(S.RobotIK)
  tried = []

  def one(action, s0, s1, assembly, seed, **kw):
    q = np.asarray(kw["mate_symmetry_quat"])
    tried.append(q.copy())
    return (len(tried) == 2), (np.array([42.0]) if len(tried) == 2 else None)

  rik._ik_assemble_single = one
  action = ("assemble", "a1", "obj_1", "obj_0")
  ok, witness = rik._ik_assemble(action, object(), object(), asm, None)
  assert ok and np.array_equal(witness, [42.0])
  assert len(tried) == 2
  assert all(np.allclose(a, b) for a, b in zip(tried, asm.mate_quaternions("obj_1")))


def test_mujoco_mate_weld_keeps_the_achieved_discrete_member():
  """The mode switch must not erase the member selected by the IK witness."""
  from types import SimpleNamespace
  import sim as S

  asm = P.make_assembly("real-stool", 2)
  asm.configure_mate_symmetry(True)
  q180 = asm.mate_quaternions("obj_1")[1]

  scene = object.__new__(S.SimScene)
  scene.assembly = asm
  scene.active_grasps = {}
  scene._body_id = lambda name: {"obj_0": 0, "obj_1": 1}[name]
  scene._eq_id = lambda name: 0
  scene.model = SimpleNamespace(eq_data=np.zeros((1, 11), dtype=float))
  scene.data = SimpleNamespace(
      xquat=np.asarray([[1.0, 0.0, 0.0, 0.0], q180], dtype=float),
      eq_active=np.zeros(1, dtype=np.int32))

  scene.weld_mate("obj_1")
  assert scene.data.eq_active[0] == 1
  assert np.allclose(scene.model.eq_data[0, 6:10], q180)


def test_yandv_normal_mujoco_view_enables_visual_group():
  import sim as S

  class Viewer:
    class Opt:
      geomgroup = np.zeros(6, dtype=int)
    opt = Opt()

  v = Viewer()
  S._default_geomgroups(v, True)
  assert v.opt.geomgroup[S.GROUP_VISUAL] == 1
  assert v.opt.geomgroup[S.GROUP_COLL] == 0
