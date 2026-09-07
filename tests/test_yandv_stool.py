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
