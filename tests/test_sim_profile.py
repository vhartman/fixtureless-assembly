"""Named sim reproduction profiles are thin, reversible CLI defaults."""
import sim


def test_no_sim_profile_preserves_historical_cli_defaults():
  a = sim.parse_args(["run", "plan.json"])
  assert a.sim_profile is None
  assert a.v2 and a.actuated and a.skill_servo and a.spline
  assert a.arm_arm_contacts is None
  assert not a.phase_schedule and not a.phase_smooth and not a.phase_batch
  assert a.servo_torque_guard is None
  assert a.corridor


def test_standard_sim_profile_pins_reproduction_contract():
  a = sim.parse_args(["run", "plan.json", "--sim-profile", "standard"])
  assert a.v2 and a.actuated and a.skill_servo and a.arm_arm_contacts and a.spline
  assert a.phase_schedule and a.phase_smooth and a.phase_batch
  assert a.phase_servo_time == 0.15 and a.phase_smooth_time == 0.12
  assert a.phase_batch_min_motion == 0.08 and a.phase_batch_max_gap == 0.35
  assert a.phase_audit_resolution == 0.01
  assert a.settle_tol == 0.002 and a.settle_max == 1.0
  assert a.servo_force == 6.0 and a.servo_max_dev == 0.010
  assert a.servo_torque_guard == 0.6 and a.servo_tol == 0.002
  assert a.force_limits and a.corner_limits
  assert a.path_preflight and a.path_preflight_tolerance == 1e-4
  assert a.path_preflight_resolution == 0.01
  assert a.joint_rate == 1.0 and a.mass_scale == 1.0 and a.coll_scale == 1.0
  assert not a.corridor and not a.corridor_online and not a.pace
  assert not a.phase_intent and not a.c2_spline and not a.limit_rate


def test_standard_sim_profile_is_reversible_by_explicit_flags():
  a = sim.parse_args([
      "run", "plan.json", "--sim-profile", "standard",
      "--no-v2", "--no-actuated", "--no-skill-servo", "--no-arm-arm-contacts",
      "--no-spline", "--no-phase-schedule", "--no-phase-smooth", "--no-phase-batch",
      "--no-servo-torque-guard", "--no-path-preflight", "--no-force-limits",
      "--corridor", "--joint-rate", "0.5", "--mass-scale", "0.25",
  ])
  assert not a.v2 and not a.actuated and not a.skill_servo and not a.arm_arm_contacts
  assert not a.spline and not a.phase_schedule and not a.phase_smooth and not a.phase_batch
  assert a.servo_torque_guard is None and not a.path_preflight and not a.force_limits
  assert a.corridor and a.joint_rate == 0.5 and a.mass_scale == 0.25
