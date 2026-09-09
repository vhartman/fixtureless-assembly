import json

import numpy as np
import pytest

import export_open_loop as E
import sim


def _recording(path):
  xml = """
  <mujoco><worldbody>
    <body name="a1_link"><joint name="a1_joint" type="hinge"/><geom type="sphere" size=".01"/></body>
    <body name="a2_link"><joint name="a2_joint" type="hinge"/><geom type="sphere" size=".01"/></body>
  </worldbody></mujoco>
  """
  qpos = np.asarray([[0.00, 0.00], [0.01, -0.02], [0.04, -0.08], [0.09, -0.18]])
  events = [{"event": "close_start", "robot": "a2", "part": "obj_1",
             "physics_tick": 3, "time_s": 0.3, "frame": 2},
            {"event": "open_start", "robot": "a1", "part": "obj_0",
             "physics_tick": 4, "time_s": 0.4, "frame": 3}]
  np.savez_compressed(
      path, xml=xml, qpos=qpos, robots=np.asarray(["a1", "a2"]),
      physics_tick=np.arange(1, 5),
      controller_mode=np.asarray(["trajectory", "settle", "skill_servo", "trajectory"]),
      gripper_events=json.dumps(events), marks=json.dumps([]), success=np.asarray(True),
      metadata=json.dumps({"dt": 0.1, "record_every": 1}))


def test_export_open_loop_contains_robot_derivatives_and_annotations(tmp_path):
  source, destination = tmp_path / "recording.npz", tmp_path / "plan.npz"
  _recording(source)
  E.export_recording(source, destination)
  with np.load(destination, allow_pickle=False) as z:
    assert str(z["schema"]) == E.SCHEMA
    assert list(z["joint_names"]) == ["a1_joint", "a2_joint"]
    assert z["q"].shape == z["qd"].shape == z["qdd"].shape == (4, 2)
    assert np.allclose(z["time"], [0.0, 0.1, 0.2, 0.3])
    assert list(z["controller_active"]) == [False, True, True, False]
    close = z["gripper_close_start"]
    assert close.shape == (4, 2) and close[2, 1] and close.sum() == 1
    attached = z["attached_objects"]
    assert attached.shape == (4, 2)
    assert list(attached[0]) == ["", ""]
    assert list(attached[2]) == ["", "obj_1"]
    assert json.loads(str(z["robot_slices"])) == {"a1": [0, 1], "a2": [1, 2]}
    assert json.loads(str(z["metadata"]))["source_signal"] == "simulated_joint_qpos"


def test_export_open_loop_json_has_the_same_typed_channels(tmp_path):
  source, destination = tmp_path / "recording.npz", tmp_path / "plan.json"
  _recording(source)
  E.export_recording(source, destination, sample_dt=0.05, grasp_wait=0.1)
  payload = json.loads(destination.read_text())
  assert payload["schema"] == E.JSON_SCHEMA
  assert payload["dt"] == 0.05 and payload["grasp_wait"] == 0.1
  assert payload["robots"] == ["a1", "a2"]
  assert payload["joint_names"] == ["a1_joint", "a2_joint"]
  assert payload["robot_slices"] == {"a1": [0, 1], "a2": [1, 2]}
  samples = payload["samples"]
  assert len(samples) == 9
  assert [row["time"] for row in samples] == pytest.approx(np.arange(9) * 0.05)
  assert all(set(row) == {"q", "qd", "qdd", "time", "opening_gripper",
                          "closing_gripper", "servo_controller", "attached_objects"}
             for row in samples)
  assert np.asarray([row["q"] for row in samples]).shape == (9, 2)
  assert np.asarray([row["qd"] for row in samples]).shape == (9, 2)
  assert np.asarray([row["qdd"] for row in samples]).shape == (9, 2)
  assert samples[4]["closing_gripper"] == ["a2"]
  assert samples[5]["closing_gripper"] == ["a2"]
  assert samples[8]["opening_gripper"] == ["a1"]
  assert samples[0]["attached_objects"] == {"a1": None, "a2": None}
  assert samples[4]["attached_objects"] == {"a1": None, "a2": "obj_1"}
  # The arm is held at the close configuration for the requested 100ms before proceeding.
  assert np.allclose(samples[4]["q"], samples[5]["q"])
  assert np.allclose(samples[5]["q"], samples[6]["q"])


def test_controller_mode_marks_only_feedback_hooks_active():
  assert sim._record_controller_mode((np.zeros(2),)) == "trajectory"
  assert sim._record_controller_mode(("path",)) == "trajectory"
  assert sim._record_controller_mode(("servo",)) == "skill_servo"
  assert sim._record_controller_mode(("settle",)) == "settle"
  assert sim._record_controller_mode(("guard",)) == "wrench_guard"


def test_export_refuses_failed_simulation_without_explicit_override(tmp_path):
  source, destination = tmp_path / "recording.npz", tmp_path / "plan.npz"
  _recording(source)
  with np.load(source, allow_pickle=False) as z:
    payload = {name: z[name] for name in z.files}
  payload["success"] = np.asarray(False)
  np.savez_compressed(source, **payload)

  with pytest.raises(ValueError, match="source simulation reports failure"):
    E.export_recording(source, destination)
  E.export_recording(source, destination, allow_failed=True)
  assert destination.exists()


def test_legacy_close_only_recording_recovers_open_commands(tmp_path):
  source, destination = tmp_path / "recording.npz", tmp_path / "plan.json"
  _recording(source)
  with np.load(source, allow_pickle=False) as z:
    payload = {name: z[name] for name in z.files}
  # This matches recordings from the interval where logical closes were recorded but opens
  # were accidentally guarded by --animate-grippers.
  payload["gripper_events"] = np.asarray(json.dumps([
      {"event": "close_start", "robot": "a1", "part": "obj_0", "frame": 1},
      {"event": "close_start", "robot": "a2", "part": "obj_1", "frame": 2},
  ]))
  payload["marks"] = np.asarray(json.dumps([
      [1, "('pick', 'a1', 'obj_0', 0)"],
      [2, "('handover', 'a2', 'obj_0', 1)"],
      [3, "('place', 'a2', 'obj_0', 0)"],
  ]))
  np.savez_compressed(source, **payload)

  E.export_recording(source, destination, sample_dt=0.1, grasp_wait=0.0)
  result = json.loads(destination.read_text())
  samples = result["samples"]
  assert samples[1]["closing_gripper"] == ["a1"]
  assert samples[2]["closing_gripper"] == ["a2"]
  assert samples[2]["opening_gripper"] == ["a1"]
  assert samples[3]["opening_gripper"] == ["a2"]
