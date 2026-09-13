import json

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")
trimesh = pytest.importorskip("trimesh")

import sim


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


def test_scene_export_requires_glb_suffix(tmp_path):
  model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
  data = mujoco.MjData(model)
  with pytest.raises(ValueError, match="requires a .glb"):
    sim.export_visual_scene_glb(model, data, tmp_path / "scene.obj")
