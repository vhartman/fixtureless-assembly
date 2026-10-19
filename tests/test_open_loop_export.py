import json

import numpy as np
import pytest

from tools.exporters import export_open_loop as E
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
      {"event": "close_start", "robot": "a2", "part": "obj_0", "frame": 2},
  ]))
  payload["marks"] = np.asarray(json.dumps([
      [1, "('pick', 'a1', 'obj_0', 0)"],
      [2, "('handover', 'a2', 'obj_0', 1)"],
      [3, "('place', 'a2', 'obj_0', 0)"],
  ]))
  np.savez_compressed(source, **payload)

  E.export_recording(source, destination, sample_dt=0.1, grasp_wait=0.1)
  result = json.loads(destination.read_text())
  samples = result["samples"]
  assert samples[1]["closing_gripper"] == ["a1"]
  assert samples[3]["closing_gripper"] == ["a2"]
  assert samples[3]["opening_gripper"] == []
  assert samples[4]["closing_gripper"] == []
  assert samples[4]["opening_gripper"] == ["a1"]
  assert samples[5]["opening_gripper"] == ["a2"]


def test_handover_closes_receiver_before_opening_giver_and_keeps_overlap(tmp_path):
  source, destination = tmp_path / "recording.npz", tmp_path / "plan.json"
  _recording(source)
  with np.load(source, allow_pickle=False) as z:
    payload = {name: z[name] for name in z.files}
  # a1 first holds obj_0; at the next logical instant a2 receives it. The source side effects
  # are atomic, exactly as sim.py records them.
  payload["gripper_events"] = np.asarray(json.dumps([
      {"event": "close_start", "robot": "a1", "part": "obj_0", "physics_tick": 2},
      {"event": "close_start", "robot": "a2", "part": "obj_0", "physics_tick": 3},
      {"event": "open_start", "robot": "a1", "part": "obj_0", "physics_tick": 3},
  ]))
  np.savez_compressed(source, **payload)

  E.export_recording(source, destination, sample_dt=0.05, grasp_wait=0.1)
  samples = json.loads(destination.read_text())["samples"]
  # First close dwell shifts source t=.2 to output t=.3; the handover dwell is [.3,.4).
  overlap = [row for row in samples if 0.3 - 1e-9 <= row["time"] < 0.4 - 1e-9]
  assert overlap and all(row["closing_gripper"] == ["a2"] for row in overlap)
  assert all(row["opening_gripper"] == [] for row in overlap)
  assert all(row["attached_objects"] == {"a1": "obj_0", "a2": "obj_0"}
             for row in overlap)
  release = next(row for row in samples if row["time"] == pytest.approx(0.4))
  assert release["opening_gripper"] == ["a1"]
  assert release["closing_gripper"] == []
  assert release["attached_objects"] == {"a1": None, "a2": "obj_0"}
  assert not any(row["opening_gripper"] and row["closing_gripper"] for row in samples)


def test_zero_grasp_wait_is_rejected_for_a_handover(tmp_path):
  source, destination = tmp_path / "recording.npz", tmp_path / "plan.json"
  _recording(source)
  with np.load(source, allow_pickle=False) as z:
    payload = {name: z[name] for name in z.files}
  payload["gripper_events"] = np.asarray(json.dumps([
      {"event": "close_start", "robot": "a2", "part": "obj_0", "physics_tick": 3},
      {"event": "open_start", "robot": "a1", "part": "obj_0", "physics_tick": 3},
  ]))
  np.savez_compressed(source, **payload)
  with pytest.raises(ValueError, match="handover requires positive --grasp-wait"):
    E.export_recording(source, destination, sample_dt=0.05, grasp_wait=0.0)


def test_json_auto_preserves_source_robot_local_wait_without_global_time_warp(tmp_path):
  source, destination = tmp_path / "recording.npz", tmp_path / "plan.json"
  _recording(source)
  with np.load(source, allow_pickle=False) as z:
    payload = {name: z[name] for name in z.files}
  payload["metadata"] = np.asarray(json.dumps({
      "dt": 0.1, "record_every": 1, "wait_for_gripper": True,
      "phase_schedule": True, "gripper_motion_time": 0.1}))
  payload["gripper_events"] = np.asarray(json.dumps([
      {"event": "close_start", "robot": "a2", "part": "obj_1",
       "physics_tick": 2, "attachment_physics_tick": 3},
      {"event": "open_start", "robot": "a1", "part": "obj_0",
       "physics_tick": 4},
  ]))
  np.savez_compressed(source, **payload)

  E.export_recording(source, destination, sample_dt=0.05)
  result = json.loads(destination.read_text())
  samples = result["samples"]
  assert result["grasp_wait_mode"] == "preserve"
  assert result["grasp_wait"] == 0.1
  assert len(samples) == 7                                  # source 0.3s, no extra global dwell
  assert any(row["closing_gripper"] == ["a2"] for row in samples)
  # a1 changes while a2 closes: preserving the certified source timing must not freeze it.
  closing = [row for row in samples if row["closing_gripper"]]
  assert np.ptp([row["q"][0] for row in closing]) > 0.0
  # Ownership begins at the recorded attachment time, not at close command start.
  assert next(row for row in samples if row["time"] == pytest.approx(0.1))[
      "attached_objects"]["a2"] is None
  assert next(row for row in samples if row["time"] == pytest.approx(0.2))[
      "attached_objects"]["a2"] == "obj_1"
