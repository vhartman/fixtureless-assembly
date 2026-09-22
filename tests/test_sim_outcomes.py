"""Behavioral checks for simulator outcomes and generated MuJoCo models."""

import pytest


def test_contact_handover_commits_only_after_receiver_acquisition():
  """A failed receiver close must leave the physical giver as the payload support."""
  import sim as S

  scene = object.__new__(S.SimScene)
  scene.contact_grasps = True
  events = []

  def attach(_receiver, _child, **kwargs):
    events.append(("acquire", kwargs["preserve_existing_grasps"]))
    return False

  scene.attach = attach
  scene.release = lambda robot: events.append(("release", robot))
  assert not scene.contact_handover("a2_tool", "obj_0", ["a1"])
  assert events == [("acquire", True)]

  events.clear()
  scene.attach = lambda _receiver, _child, **kwargs: (
      events.append(("acquire", kwargs["preserve_existing_grasps"])) or True)
  assert scene.contact_handover("a2_tool", "obj_0", ["a1"])
  assert events == [("acquire", True), ("release", "a1")]

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


def test_mujoco_rounded_box_does_not_fill_rai_rounded_corners():
  """The sim collision proxy must represent the ssBox used by planning, not its AABB.

  A small probe near a sharp-box corner is outside the radius-5mm rounded corner, while a
  probe at the centre of a planar face must still make contact. This is the geometry mismatch
  that previously let RAI certify a cube re-pick which bulldozed the assembly in MuJoCo.
  """
  import mujoco
  import sim as S

  rounded = S._rounded_box_geoms_xml(
      ("box", [0.055, 0.055, 0.020, 0.005]), 0.5, 0.1, 0,
      "0.8 0.2 0.2 1", "rounded")
  assert rounded is not None and rounded.count("<geom") == 15

  def contacts_at(pos):
    xml = f"""
    <mujoco><option gravity="0 0 0"/><worldbody>
      {rounded}
      <body pos="{pos[0]} {pos[1]} {pos[2]}"><freejoint/>
        <geom name="probe" type="sphere" size="0.0005" mass="0.001"/>
      </body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data.ncon

  # At x=y=27mm the probe is 6.36mm from the corner-arc centre (22.5,22.5),
  # hence clear of the 5mm arc. A sharp 27.5mm-half-extent box would collide.
  assert contacts_at((0.027, 0.027, 0.0)) == 0
  assert contacts_at((0.0273, 0.0, 0.0)) > 0


def test_grasp_and_mate_welds_have_distinct_default_compliance():
  """A planned rigid subassembly must not become a soft serial chain in physics."""
  import mujoco
  import assembly as A
  from env import make_four_robot_env
  import sim as S

  asm = A.make_assembly("peg-hole", 2)
  C = make_four_robot_env(num_robots=1, floating_ee=True, vacuum=False)
  A.add_disassembled_obj(C, asm, spacing=0.2, float_height=0.002)
  state = A.get_initial_state(C, asm)

  def timeconsts(**kwargs):
    model = mujoco.MjModel.from_xml_string(
        S.build_mjcf(asm, state, ["a1"], **kwargs))
    grasp = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_a1_obj_1")
    mate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "mate_obj_1")
    return float(model.eq_solref[grasp, 0]), float(model.eq_solref[mate, 0])

  assert timeconsts() == pytest.approx((S.WELD_TIMECONST, S.MATE_WELD_TIMECONST))
  # The old option remains a shared override; the new option can independently restore the
  # rigid mate model in experiments that deliberately soften the physical grasp.
  assert timeconsts(weld_timeconst=0.006) == pytest.approx((0.006, 0.006))
  assert timeconsts(weld_timeconst=0.006,
                    mate_weld_timeconst=0.003) == pytest.approx((0.006, 0.003))


def test_contact_grasps_preserve_vacuum_welds_in_mixed_workcell():
  """Contact execution is per tool: suction welds remain, parallel-jaw welds disappear."""
  import mujoco
  import assembly as A
  from env import make_four_robot_env
  import sim as S

  asm = A.make_assembly("peg-hole", 2)
  C = make_four_robot_env(num_robots=1, floating_ee=True, vacuum=False)
  A.add_disassembled_obj(C, asm, spacing=0.2, float_height=0.002)
  state = A.get_initial_state(C, asm)
  model = mujoco.MjModel.from_xml_string(S.build_mjcf(
      asm, state, ["a1", "a2"], contact_grasps=True,
      gripper_kinds={"a1": "vacuum", "a2": "two_finger"}))

  assert mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_a1_obj_1") >= 0
  assert mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_a2_obj_1") == -1


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

  # Lockstep can leave a late post untouched for minutes. Thirty seconds is long enough to
  # expose the old steady creep (about 1.5mm here) without making this regression expensive.
  for _ in range(int(30.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)

  for part, (qadr, xy0) in starts.items():
    drift = float(np.linalg.norm(data.qpos[qadr:qadr + 2] - xy0))
    assert drift < 5e-4, (part, drift)


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
