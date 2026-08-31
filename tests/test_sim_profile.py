"""Named sim reproduction profiles are thin, reversible CLI defaults."""
import json
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import sim
import wrench


def test_standard_sim_profile_is_reversible_by_explicit_flags():
  a = sim.parse_args([
      "run", "plan.json", "--sim-profile", "standard",
      "--no-actuated", "--no-skill-servo", "--no-arm-arm-contacts",
      "--no-spline", "--no-phase-schedule", "--no-phase-smooth", "--no-phase-batch",
      "--no-servo-torque-guard", "--no-path-preflight", "--no-force-limits",
      "--no-require-mp-path", "--no-require-success",
      "--corridor", "--joint-rate", "0.5", "--mass-scale", "0.25",
  ])
  assert not a.actuated and not a.skill_servo and not a.arm_arm_contacts
  assert not a.spline and not a.phase_schedule and not a.phase_smooth and not a.phase_batch
  assert a.servo_torque_guard is None and not a.path_preflight and not a.force_limits
  assert not a.require_mp_path and not a.require_success
  assert a.corridor and a.joint_rate == 0.5 and a.mass_scale == 0.25


def test_standard_profile_refuses_an_implicit_motion_plan_before_scene_build():
  a = sim.parse_args(["run", "plan.json", "--sim-profile", "standard"])
  with pytest.raises(SystemExit, match="require-mp-path"):
    sim._cli_run(a)


def test_saved_wrench_model_is_restored_before_sim_replay():
  old = copy.deepcopy(wrench.PARAMS)
  try:
    sim._restore_saved_wrench_config({
        "grasp_model": "asymmetric", "tf_close_torque": 0.57,
        "insert_force": 12.0, "insertion_noise": 0.003,
        "insertion_offset": 0.004, "ignore_gravity": True,
    })
    assert wrench.PARAMS["grasp_model"] == "asymmetric"
    assert wrench.PARAMS["grasp"]["asymmetric"]["two_finger"]["T_close"] == 0.57
    assert wrench.PARAMS["insert_force"]["_default"] == 12.0
    assert wrench.PARAMS["insertion_uncertainty"] == {
        "lateral": 0.003, "offset": 0.004}
    assert wrench.PARAMS["ignore_gravity"] is True
  finally:
    wrench.PARAMS.clear()
    wrench.PARAMS.update(old)


def test_recording_embeds_machine_readable_verdict_and_profile(tmp_path):
  scene = SimpleNamespace(
      xml="<mujoco/>", robots=["a1"], assembly=SimpleNamespace(objects=["obj_0"]))
  rec = {"qpos": [[0.0]], "mpos": [], "mquat": [], "forces": [[0.0]], "marks": []}
  report = sim.SimReport()
  report.results.append(sim.ActionResult(("pick", "a1", "obj_0", 0), True))
  report.final_pose_err = {"obj_0": (0.001, 0.002)}
  path = tmp_path / "run.npz"
  sim._save_recording(path, scene, rec, report,
                      metadata={"profile": "standard", "joint_rate": 1.0})
  with np.load(path) as data:
    assert bool(data["success"])
    assert json.loads(str(data["metadata"]))["profile"] == "standard"
    assert json.loads(str(data["final_pose_err"]))["obj_0"] == [0.001, 0.002]
