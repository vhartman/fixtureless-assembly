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


def test_kinematic_arm_reasserts_visual_gripper_mimic_joints(monkeypatch):
  """Kinematic parent resets must not leave presentation-only child hinges free-running."""
  scene = sim.SimScene.__new__(sim.SimScene)
  scene.animate_grippers = True
  scene.actuated = False
  scene.robots = ["a1"]
  scene._rai_joints = []
  scene._qadr = {}
  scene._vadr = {}
  scene._cmd_q = None
  scene._phys = 0
  scene._gripper_model = {"a1": "robotiq"}
  scene._gripper_aperture = {"a1": 0.021}                 # half of the 42mm stroke
  scene._gripper_qadr = {"a1": [0, 1]}
  scene._gripper_vadr = {"a1": [0, 1]}
  scene._gripper_ctrl = {"a1": [0, 1]}
  scene._gripper_gain = {"a1": [1.0, -1.0]}
  scene._gripper_angle = {"a1": [0.8, 0.8]}
  scene.data = SimpleNamespace(
      qpos=np.asarray([0.71, 0.33]), qvel=np.asarray([4.0, -3.0]), ctrl=np.zeros(2))
  scene.model = object()
  monkeypatch.setattr(sim.mujoco, "mj_forward", lambda _model, _data: None)

  scene.set_arm_q(np.empty(0))

  assert np.allclose(scene.data.qpos, [0.4, -0.4])
  assert np.allclose(scene.data.qvel, 0.0)
  assert np.allclose(scene.data.ctrl, [0.4, -0.4])


def test_standard_profile_refuses_an_implicit_motion_plan_before_scene_build():
  a = sim.parse_args(["run", "plan.json", "--sim-profile", "standard"])
  assert a.servo_place_seat
  assert not a.spline
  assert a.phase_schedule and a.phase_smooth and a.phase_batch
  with pytest.raises(SystemExit, match="require-mp-path"):
    sim._cli_run(a)


def test_exact_grasp_target_uses_authored_relation_not_witness_fk(monkeypatch):
  part_pose = np.asarray([0.4, -0.2, 0.7, 1.0, 0.0, 0.0, 0.0])
  grasp = np.asarray([0.03, 0.01, -0.02, 1.0, 0.0, 0.0, 0.0])
  # Representative robot-specific tool-frame calibration: a 90 degree roll.
  correction = np.asarray([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
  monkeypatch.setattr(sim, "grasp_pose", lambda _a, _p, _g: grasp)
  robot_ik = SimpleNamespace(q_correction={"a1": correction})

  got = sim._exact_grasp_anchor_target(
      object(), robot_ik, "a1", "obj_0", 7, part_pose)
  expected = sim._pose_compose(
      part_pose, np.r_[grasp[:3], sim._quat_mul(grasp[3:], correction)])

  assert np.allclose(got, expected)


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


def test_scalar_vacuum_cap_includes_friction_limited_lateral_force():
  old = copy.deepcopy(wrench.PARAMS)
  try:
    wrench.PARAMS["grasp_model"] = "asymmetric"
    wrench.PARAMS["grasp"]["asymmetric"]["vacuum"] = {
        "F_push": 40.0, "F_pull": 25.0, "mu": 0.15,
        "T_axis": 0.6, "T_lat": 0.8}
    force, torque = sim._grasp_caps("vacuum")
    assert force == pytest.approx(3.75)
    assert torque == pytest.approx(0.6)
  finally:
    wrench.PARAMS.clear()
    wrench.PARAMS.update(old)


def test_grasp_wrench_metric_requires_a_contiguous_full_window():
  scene = sim.SimScene.__new__(sim.SimScene)
  scene._wrench_metric_window = 3
  scene._wrench_metric_history = {}
  scene._wrench_metrics_by_kind = {}
  scene.gripper_kinds = {"a1": "two_finger"}
  scene._wrench_metrics = {
      phase: {
          "samples": 0,
          "peak_force_n": 0.0,
          "peak_torque_nm": 0.0,
          "peak_utilization": 0.0,
          "peak_sustained_force_n": None,
          "peak_sustained_torque_nm": None,
          "peak_sustained_utilization": None,
      }
      for phase in ("transport", "insertion")
  }
  for tick, utilization in enumerate((0.3, 0.6, 0.9)):
    scene.step_i = tick
    scene._observe_grasp_wrench(
        "transport", "a1", "obj_0", utilization, 10.0 * utilization, utilization)
  got = scene.grasp_wrench_metrics()
  assert got["phases"]["transport"]["peak_utilization"] == pytest.approx(0.9)
  assert got["phases"]["transport"]["peak_sustained_utilization"] == pytest.approx(0.6)
  assert got["by_gripper_kind"]["two_finger"]["transport"][
      "peak_sustained_utilization"] == pytest.approx(0.6)

  # A gap starts a new interval rather than joining physically unrelated samples.
  scene.step_i = 10
  scene._observe_grasp_wrench("transport", "a1", "obj_0", 3.0, 30.0, 3.0)
  got = scene.grasp_wrench_metrics()
  assert got["phases"]["transport"]["peak_utilization"] == pytest.approx(3.0)
  assert got["phases"]["transport"]["peak_sustained_utilization"] == pytest.approx(0.6)


def test_grasp_wrench_incremental_window_recovers_after_infinite_sample():
  scene = sim.SimScene.__new__(sim.SimScene)
  scene._wrench_metric_window = 2
  scene._wrench_metric_history = {}
  scene._wrench_metrics_by_kind = {}
  scene.gripper_kinds = {"a1": "vacuum"}
  scene._wrench_metrics = {
      phase: {
          "samples": 0, "peak_force_n": 0.0, "peak_torque_nm": 0.0,
          "peak_utilization": 0.0, "peak_sustained_force_n": None,
          "peak_sustained_torque_nm": None, "peak_sustained_utilization": None,
      }
      for phase in ("transport", "insertion")
  }
  for tick, utilization in enumerate((float("inf"), 1.0, 1.0)):
    scene.step_i = tick
    scene._observe_grasp_wrench(
        "transport", "a1", "obj_0", utilization, 1.0, 1.0)
  _hist, _last, total = scene._wrench_metric_history[("transport", "a1", "obj_0")]
  assert np.allclose(total, [2.0, 2.0, 2.0])


def test_max_robotiq_selects_axis_resolved_breakaway_unless_overridden():
  assert sim.parse_args(["run", "plan.json"]).directional_caps is None
  assert sim._resolve_directional_caps({"wrench_profile": "max_robotiq"}, None)
  assert not sim._resolve_directional_caps({"wrench_profile": "max_robotiq"}, False)
  assert sim._resolve_directional_caps({}, True)
  assert not sim._resolve_directional_caps({}, None)


def test_measured_profile_selects_axis_resolved_breakaway():
  assert sim._resolve_directional_caps({"wrench_profile": "measured"}, None)


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
