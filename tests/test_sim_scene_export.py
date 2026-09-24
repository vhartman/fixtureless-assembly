import json

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")
trimesh = pytest.importorskip("trimesh")

import sim


def test_recording_mesh_assets_relocate_by_content_keyed_basename(tmp_path, monkeypatch):
  cache = tmp_path / "sim_meshes"
  cache.mkdir()
  local = cache / "arm__m0123456789.obj"
  local.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
  monkeypatch.setattr(sim, "_MESH_CACHE_DIR", str(cache))
  old = "/another/computer/project/sim_meshes/arm__m0123456789.obj"
  xml = f'<mujoco><asset><mesh name="arm" file="{old}"/></asset></mujoco>'

  relocated = sim._relocate_recording_mesh_assets(xml)

  assert old not in relocated
  assert f'file="{local.resolve()}"' in relocated


def test_recording_mesh_assets_do_not_guess_for_non_cache_paths(tmp_path, monkeypatch):
  cache = tmp_path / "sim_meshes"
  cache.mkdir()
  (cache / "visual.obj").write_text("not the source asset", encoding="utf-8")
  monkeypatch.setattr(sim, "_MESH_CACHE_DIR", str(cache))
  old = "/another/computer/project/source_meshes/visual.obj"
  xml = f'<mujoco><asset><mesh name="visual" file="{old}"/></asset></mujoco>'

  assert sim._relocate_recording_mesh_assets(xml) == xml


def test_replay_exports_exact_frame_as_visual_only_self_contained_glb(tmp_path):
  xml = """
  <mujoco>
    <worldbody>
      <geom name="fixed_visual" type="box" size=".1 .1 .1"
            rgba="1 0 0 1" group="0"/>
      <geom name="collision_proxy" type="sphere" size="2" group="2"/>
      <body name="moving" pos="0 0 .3">
        <joint type="slide" axis="1 0 0"/>
        <geom name="moving_visual" type="box" size=".1 .1 .1"
              rgba="0 0 1 1" group="3"/>
      </body>
    </worldbody>
  </mujoco>
  """
  recording = tmp_path / "recording.npz"
  np.savez_compressed(
      recording, xml=xml, qpos=np.asarray([[0.0], [1.0]]),
      marks=json.dumps([]), results=json.dumps([]))
  output = tmp_path / "frame-one.glb"

  got = sim.replay_recording(
      str(recording), frame=1, export_scene=str(output), export_only=True)

  assert got == output
  scene = trimesh.load(output, force="scene")
  assert len(scene.graph.nodes_geometry) == 2  # group-2 collision proxy was excluded
  assert scene.bounds[1, 0] == pytest.approx(1.1, abs=1e-5)
  assert output.read_bytes()[:4] == b"glTF"




def test_scene_export_does_not_apply_compiled_mesh_offset_twice(tmp_path):
  # MuJoCo recentres an imported mesh and rotates it to principal axes.  The compensating
  # transform is already folded into geom_pos/geom_quat; geom_xpos/xmat therefore maps the
  # compiled mesh vertices directly to world.  This deliberately off-centre tetrahedron
  # catches the tempting but wrong extra mesh_pos/mesh_quat multiplication.
  obj = b"""\
v 5 0 0
v 7 0 0
v 5 2 0
v 5 0 2
f 1 2 3
f 1 4 2
f 1 3 4
f 2 4 3
"""
  xml = """
  <mujoco>
    <asset><mesh name="offcentre" file="offcentre.obj"/></asset>
    <worldbody>
      <body pos="1 2 3">
        <geom type="mesh" mesh="offcentre" pos=".1 .2 .3"
              rgba=".2 .4 .8 1" group="3"/>
      </body>
    </worldbody>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml, assets={"offcentre.obj": obj})
  data = mujoco.MjData(model)
  output = tmp_path / "offcentre.glb"
  sim.export_visual_scene_glb(model, data, output)

  scene = trimesh.load(output, force="scene")
  # Raw MuJoCo world bounds are x=[6.1,8.1], y=[2.2,4.2], z=[3.3,5.3].  The exporter maps
  # (x,y,z) -> (x,z,-y) for glTF's Y-up convention.
  np.testing.assert_allclose(
      scene.bounds, [[6.1, 3.3, -4.2], [8.1, 5.3, -2.2]], atol=1e-5)
