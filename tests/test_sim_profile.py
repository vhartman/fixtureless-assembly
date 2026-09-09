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
      "--no-servo-pick",
      "--no-servo-place-seat",
      "--no-servo-torque-guard", "--no-path-preflight", "--no-force-limits",
      "--no-require-mp-path", "--no-require-success",
      "--corridor", "--joint-rate", "0.5", "--mass-scale", "0.25",
  ])
  assert not a.actuated and not a.skill_servo and not a.arm_arm_contacts and not a.servo_pick
  assert not a.servo_place_seat
  assert not a.spline and not a.phase_schedule and not a.phase_smooth and not a.phase_batch
  assert a.servo_torque_guard is None and not a.path_preflight and not a.force_limits
  assert not a.require_mp_path and not a.require_success
  assert a.corridor and a.joint_rate == 0.5 and a.mass_scale == 0.25


def test_standard_profile_refuses_an_implicit_motion_plan_before_scene_build():
  a = sim.parse_args(["run", "plan.json", "--sim-profile", "standard"])
  assert a.servo_place_seat
  with pytest.raises(SystemExit, match="require-mp-path"):
    sim._cli_run(a)


def test_process_wrench_application_is_explicit_and_reversible():
  assert not sim.parse_args(["run", "plan.json"]).apply_process_wrench
  assert sim.parse_args(
      ["run", "plan.json", "--apply-process-wrench"]).apply_process_wrench
  assert not sim.parse_args([
      "run", "plan.json", "--apply-process-wrench",
      "--no-apply-process-wrench"]).apply_process_wrench


def test_process_noise_cli_is_explicit_and_replayable():
  a = sim.parse_args([
      "run", "plan.json", "--apply-process-wrench", "--process-noise", "sample",
      "--process-noise-seed", "17", "--process-engagement-distance", "0.006",
      "--process-wrench-ramp", "0.08"])
  assert a.process_noise == "sample" and a.process_noise_seed == 17
  assert a.process_engagement_distance == pytest.approx(0.006)
  assert a.process_wrench_ramp == pytest.approx(0.08)


def test_process_noise_realizations_cover_exact_box_corners_and_seeded_samples():
  old = copy.deepcopy(wrench.PARAMS)
  plan = [("assemble", "a1", "obj_1", "obj_0"),
          ("release", "a1", "obj_1"),
          ("assemble", "a2", "obj_2", "obj_0")]
  try:
    wrench.PARAMS["insertion_uncertainty"] = {"lateral": 0.2, "offset": 0.004}
    c0 = sim._process_realizations(plan, mode="corner", corner=0)
    c15 = sim._process_realizations(plan, mode="corner", corner=15)
    assert c0[0]["coefficients"] == [-1.0, -1.0, -1.0, -1.0]
    assert c15[0]["coefficients"] == [1.0, 1.0, 1.0, 1.0]
    assert set(c0) == {0, 2}
    s1 = sim._process_realizations(plan, mode="sample", seed=23)
    s2 = sim._process_realizations(plan, mode="sample", seed=23)
    assert s1 == s2
    assert s1[0] != s1[2]                                  # one fixed draw per action
  finally:
    wrench.PARAMS.clear()
    wrench.PARAMS.update(old)


def test_low_table_friction_scales_all_mujoco_contact_coefficients():
  assert sim._contact_friction(0.5, table=True) == "0.5 0.005 0.002"
  assert sim._contact_friction(0.02, table=True) == "0.02 0.0002 8e-05"
  assert sim._contact_friction(0.0, table=True) == "0 0 0"


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


def test_saved_wrench_cap_scales_are_idempotent_not_squared():
  old = copy.deepcopy(wrench.PARAMS)
  try:
    base_tf = old["grasp"]["asymmetric"]["two_finger"]["T_close"]
    base_vac = old["grasp"]["asymmetric"]["vacuum"]["T_axis"]
    cfg = {"grasp_model": "asymmetric", "two_finger_scale": 4.0,
           "vacuum_scale": 2.0}
    sim._restore_saved_wrench_config(cfg)
    sim._restore_saved_wrench_config(cfg)                    # build_scene + sim restoration
    assert wrench.PARAMS["grasp"]["asymmetric"]["two_finger"]["T_close"] == \
        pytest.approx(4.0 * base_tf)
    assert wrench.PARAMS["grasp"]["asymmetric"]["vacuum"]["T_axis"] == \
        pytest.approx(2.0 * base_vac)
  finally:
    wrench.PARAMS.clear()
    wrench.PARAMS.update(old)


def test_max_robotiq_selects_axis_resolved_breakaway_unless_overridden():
  assert sim.parse_args(["run", "plan.json"]).directional_caps is None
  assert sim._resolve_directional_caps({"wrench_profile": "max_robotiq"}, None)
  assert not sim._resolve_directional_caps({"wrench_profile": "max_robotiq"}, False)
  assert sim._resolve_directional_caps({}, True)
  assert not sim._resolve_directional_caps({}, None)


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
