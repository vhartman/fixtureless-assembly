"""END-TO-END SIM OUTCOMES -- the regression net the unit tests do not provide.

WHY THIS FILE EXISTS. On 2026-07-26 a bare `NameError` killed EVERY `sim.py run`
invocation while all 126 unit tests stayed green, because nothing in the suite ever built a
SimScene. The same blind spot hid a `--mass-scale` StopIteration and a missing
`<compiler angle="radian"/>` that made MuJoCo read rai's joint ranges as degrees. All three were
found by running the sim by hand.

WHAT THESE ASSERT, AND WHY NOT MILLIMETRES. A final part-pose error is almost always a CASCADE:
one mate welds to a wrong datum and everything after inherits the displacement, so the headline
mm number describes the wreckage rather than the fault (measured: a 33.7mm first miss became a
1894mm final error). These tests assert the FIRST FAILURE's IDENTITY -- which action, which
failure kind -- plus the ok/total count. That is the thing that stays stable when the physics is
re-tuned, and the thing that actually changes when something breaks.

The scene/path artifacts are untracked working files; every test SKIPS if its pair is absent
rather than failing, so a fresh checkout stays green.
"""
import os
import re
import sys
import json

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _have(*files):
  return all(os.path.exists(os.path.join(REPO, f)) for f in files)


def _has_final_export(path):
  """Physics replay is meaningful only for a modern path with its exact task/keyframe bundle."""
  try:
    with open(os.path.join(REPO, path)) as f:
      data = json.load(f)
    return isinstance(data, dict) and isinstance(data.get("mp_export"), dict)
  except (OSError, ValueError):
    return False


def _run(capsys, argv):
  """Run a sim CLI invocation and return its printed output."""
  import sim as S
  cwd = os.getcwd()
  os.chdir(REPO)
  try:
    S._cli_run(S._build_cli().parse_args(argv))
  finally:
    os.chdir(cwd)
  return capsys.readouterr().out


def _summary(out):
  """(n_ok, n_total, first_failure_line or None) from a run's printed summary."""
  m = re.search(r"(\d+)/(\d+) actions ok", out)
  assert m, f"no summary line in output:\n{out[-2000:]}"
  first = None
  lines = out.splitlines()
  for i, ln in enumerate(lines):
    if "FIRST FAILURE" in ln:
      first = lines[i + 1].strip() if i + 1 < len(lines) else ""
      break
  return int(m.group(1)), int(m.group(2)), first


# The old local car path is retained as data, but it predates the exact task/keyframe export.
# Never turn an outcome regression into a fresh IK solve: skip it until the path is regenerated.
CAR = ["run", "car-mini-new.json", "--mp-path", "mp-car-mini-new.json"]


@pytest.mark.skipif(not (_have("car-mini-new.json", "mp-car-mini-new.json") and
                         _has_final_export("mp-car-mini-new.json")),
                    reason="modern car-mini path with embedded final export not present")
def test_car_mini_actuated_executes_cleanly(capsys):
  """The compliant-arm reference case. Catches anything that breaks --actuated end to end:
  the arm model, the servos, the velocity feedforward, the settle gate, the weld wrench.
  (--smooth is gone with the --prioritized branch: one preprocessing mode -> one executor.)"""
  # pin the ORIGINAL configuration: actuated drive alone (2026-07-29 made the full
  # corridor/servo stack the CLI default; this test isolates the plain actuated path).
  out = _run(capsys, CAR + ["--actuated", "--no-corridor", "--no-skill-servo",
                            "--no-force-limits"])
  n_ok, n_tot, first = _summary(out)
  assert first is None, f"expected a clean run, first failure was: {first}"
  assert n_ok == n_tot, f"{n_ok}/{n_tot}"


@pytest.mark.skipif(not (_have("car-mini-new.json", "mp-car-mini-new.json") and
                         _has_final_export("mp-car-mini-new.json")),
                    reason="modern car-mini path with embedded final export not present")
def test_car_mini_mocap_corridor_executes_cleanly_and_verifies(capsys):
  """The DEFAULT path plus corridor retiming. Also pins the corridor's own verification: with
  the legitimate-contact filter in place car-mini's schedule is genuinely collision-free, so a
  regression that reintroduces false positives (a gripper vs the part it HOLDS, a seated mate)
  shows up here rather than as mysterious refinement passes."""
  # pin the ORIGINAL configuration: mocap + bare corridor (no actuated/servo/spline/
  # rate-clamp/cross-only, which are all CLI defaults since 2026-07-29).
  out = _run(capsys, CAR + ["--corridor", "--force-limits", "--inertia-factor", "1",
                            "--no-actuated", "--no-skill-servo", "--no-spline",
                            "--no-corridor-rate-clamp", "--no-corridor-cross-only"])
  n_ok, n_tot, first = _summary(out)
  assert first is None, f"expected a clean run, first failure was: {first}"
  assert n_ok == n_tot, f"{n_ok}/{n_tot}"
  assert "certified collision-free" in out, "corridor verification did not run or did not pass"


@pytest.mark.skipif(not _have("car-mini-new.json", "mp-car-mini-new.json"),
                    reason="legacy car-mini plan/path pair not present")
def test_a_legacy_path_without_final_export_is_refused(capsys):
  """A bare-list path has neither exact tasks nor final keyframes. Replay must not improvise
  replacements by re-solving IK; regeneration is safer than a plausible-looking wrong run."""
  with pytest.raises(SystemExit) as exc:
    _run(capsys, CAR)
  out = capsys.readouterr().out + str(exc.value)
  assert "has no final post-fusion export block" in out
  assert "Refusing to re-solve IK" in out
  assert "actions ok" not in out, "legacy path executed despite lacking its exact export"


@pytest.mark.skipif(not (_have("cross8.json", "mp.json") and _has_final_export("mp.json")),
                    reason="modern cross8 path with embedded final export not present")
def test_cross8_warns_that_its_assembles_have_no_funnel(capsys):
  """cross8 was planned with pre_insertion=0, so every assemble drives the held part straight
  to a pose already in contact and it arrives LATERALLY -- 0.213mm of penetration while the
  target was still at rest, then ~800 steps of shoving and a weld fired 32mm off. The warning
  is what turns an hour of probing into a line of output, so it must keep firing."""
  out = _run(capsys, ["run", "cross8.json", "--mp-path", "mp.json",
                      "--no-corridor", "--no-actuated", "--no-skill-servo",
                      "--no-force-limits"])
  assert "NO INSERTION FUNNEL" in out
  m = re.search(r"NO INSERTION FUNNEL on (\d+)/(\d+) assemble", out)
  assert m and m.group(1) == m.group(2), "expected EVERY cross8 assemble to be unfunnelled"


def test_main_refuses_to_silently_do_nothing():
  """`main.py --motion-plan --save x` without --solve/--load-plan used to walk to the end and
  return None in 0.5s -- indistinguishable from a crash or a very fast solve."""
  import main
  cfg_args = ["--assembly", "cross", "--parts", "3", "--num-robots", "2",
              "--motion-plan", "--save", "/tmp/should-not-be-written.json", "--no-view"]
  with pytest.raises(SystemExit) as e:
    main.main(cfg_args)
  assert "need a plan" in str(e.value)


def test_group_root_handover_recovers_descendant_grasp_id():
  """mr_bench attaches the fixed group root, but the gripper can touch a mated descendant."""
  import sim as S

  class _State:
    supports = {
        "obj_0": frozenset(),
        "obj_3": frozenset({("mate", "obj_0"), ("grasp", "a2", "obj_3", 6)}),
    }

    @staticmethod
    def subassemblies():
      return [frozenset({"obj_0", "obj_3"})]

  assert S._held_grasp_id(_State(), "obj_0", robot="a2") == ("obj_3", 6)


def test_car_mini_wheel_rims_rest_on_sim_table():
  """A rotated cylinder must use its real support extent in both the planner and MuJoCo.

  This catches the car-specific failure where the four flat scatter wheels appeared inside
  the table, as well as collision/visual proxy offsets that make the wheel float above it.
  """
  import numpy as np
  import mujoco

  import assembly as A
  from env import make_four_robot_env
  import sim as S

  asm = A.make_assembly("car-mini", 5)
  C = make_four_robot_env(num_robots=1, floating_ee=True, vacuum=False)
  A.add_disassembled_obj(C, asm, spacing=0.2, float_height=0.002)
  state = A.get_initial_state(C, asm)
  model = mujoco.MjModel.from_xml_string(S.build_mjcf(asm, state, []))
  data = mujoco.MjData(model)
  for _ in range(100):
    mujoco.mj_step(model, data)

  table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
  table_top = float(data.geom_xpos[table_id, 2] + model.geom_size[table_id, 2])
  for part in ("obj_1", "obj_2", "obj_3", "obj_4"):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
    geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
    assert len(geom_ids) == 1, f"{part} must be one physical/visual cylinder"
    geom_id = int(geom_ids[0])
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
    axis = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)[:, 2]
    radius, half_length = model.geom_size[geom_id, :2]
    half_z = abs(float(axis[2])) * half_length + np.hypot(axis[0], axis[1]) * radius
    rim_bottom = float(data.geom_xpos[geom_id, 2] - half_z)
    assert abs(rim_bottom - table_top) < 5e-4, (part, rim_bottom, table_top)


def test_chair_round_parts_do_not_roll_out_of_their_scatter_cells():
  """The bench must give loose horizontal cylinders finite rolling resistance.

  MuJoCo ignores a geom's third friction coefficient unless the contact has ``condim=6``.
  The omission let numerical contact noise roll Chair legs/posts centimetres before their
  scheduled picks. This checks actual free-body dynamics rather than the XML spelling.
  """
  import numpy as np
  import mujoco

  import assembly as A
  from env import make_four_robot_env
  import sim as S

  asm = A.make_assembly("chair", 8)
  C = make_four_robot_env(num_robots=1, floating_ee=True, vacuum=False)
  A.add_disassembled_obj(C, asm, spacing=0.2, float_height=0.002)
  state = A.get_initial_state(C, asm)
  model = mujoco.MjModel.from_xml_string(S.build_mjcf(asm, state, []))
  data = mujoco.MjData(model)

  table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
  assert model.geom_condim[table_id] == 6
  starts = {}
  for part in ("obj_1", "obj_2", "obj_3", "obj_4", "obj_6", "obj_7"):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
    qadr = model.jnt_qposadr[model.body_jntadr[body_id]]
    starts[part] = (qadr, data.qpos[qadr:qadr + 2].copy())

  # Longer than the approach to the first pick in the phase-scheduled Chair replay.
  for _ in range(int(6.4 / model.opt.timestep)):
    mujoco.mj_step(model, data)

  for part, (qadr, xy0) in starts.items():
    drift = float(np.linalg.norm(data.qpos[qadr:qadr + 2] - xy0))
    assert drift < 1e-3, (part, drift)


def test_topple_error_respects_primitive_rotational_symmetry():
  """Rolling a round peg is not toppling it; tilting its axis still is."""
  import numpy as np

  import assembly as A
  import sim as S

  asm = A.make_assembly("peg-stool", 4)
  identity = np.array([1.0, 0.0, 0.0, 0.0])
  axial_roll = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
  axis_tilt = np.array([np.cos(np.pi / 12), np.sin(np.pi / 12), 0.0, 0.0])

  assert S._part_orientation_error(asm, "obj_1", identity, axial_roll) < 1e-12
  assert S._part_orientation_error(asm, "obj_1", identity, axis_tilt) == pytest.approx(
      np.pi / 6)
  # The compound seat has no declared rotational symmetry.
  assert S._part_orientation_error(asm, "obj_0", identity, axial_roll) == pytest.approx(
      np.pi / 2)


def test_topple_error_respects_enabled_discrete_mate_symmetry():
  """A concrete symmetry member retained by the mate weld is not a toppled part."""
  import numpy as np

  import assembly as A
  import sim as S

  asm = A.make_assembly("real-stool", 5)
  identity = np.array([1.0, 0.0, 0.0, 0.0])
  halfturn = np.array([0.0, 0.0, 0.0, 1.0])
  assert S._part_orientation_error(asm, "obj_1", halfturn, identity) == pytest.approx(np.pi)
  asm.configure_mate_symmetry(True)
  assert S._part_orientation_error(asm, "obj_1", halfturn, identity) < 1e-12
