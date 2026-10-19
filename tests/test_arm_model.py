"""The articulated MuJoCo arm generated from rai must agree with rai's own FK.

This is the guarantee the whole articulated-sim plan rests on: the planner expresses every
witness, strict-collision gate and seat tolerance in rai's joint space, so if MuJoCo's
kinematics drift even slightly, the same q means a different end-effector pose in the two and
witnesses land somewhere else -- silently. Cheap to check, catastrophic to get wrong, so it is
pinned here rather than left to `tools/validation/fk_compare.py` being run by hand.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

import assembly_planner.arm_model as arm_model
from tools.validation import fk_compare
from assembly_planner.env import (color_gripper, make_four_robot_env, make_fr3_duo_env, make_husky_duo_env,
                 make_openarm_duo_env)
import robotic as ry
import assembly_planner.assembly as A


def _env(robot, n=1):
  return make_four_robot_env(num_robots=n, floating_ee=False, vacuum=False, robot_types=robot)


@pytest.mark.parametrize("robot", ["ur10", "ur5"])
def test_generated_model_matches_rai_fk(robot):
  """Same script, different robot, no code change -- the model is READ from rai, so a new
  arm needs no porting. Tolerance is far tighter than anything the planner cares about."""
  C = _env(robot)
  rows, (wp, wa, wf) = fk_compare.compare(C, A.get_robots(C), n=12, seed=0)
  assert len(rows) > 20, "suspiciously few frames compared"
  assert wp < 1e-6, f"{robot}: {wf} position error {wp * 1000:.6f} mm"
  assert wa < 1e-6, f"{robot}: {wf} orientation error {np.degrees(wa):.6f} deg"


def test_joints_are_exactly_rais_joints():
  """Every generated hinge must be a real rai joint, and every rai joint of that arm must be
  generated -- a dropped joint would still pass FK at q=0 and fail everywhere else."""
  C = _env("ur10")
  prefixes = A.get_robots(C)
  _xml, jnames = arm_model.build_arms_mjcf(C, prefixes)
  rai = [j for j in C.getJointNames() if any(j.startswith(p) for p in prefixes)]
  assert sorted(jnames) == sorted(rai)


@pytest.mark.parametrize("robot", ["ur10", "ur5", "openarm"])
def test_articulated_gripper_hinges_are_sim_only_and_actuated(robot):
  """Finger export uses real hinges/actuators without entering the planner arm vector."""
  C = _env(robot) if robot != "openarm" else make_openarm_duo_env()
  prefix = A.get_robots(C)[0]
  assets = []
  body, arm_joints = arm_model.build_arm_mjcf(
      C, prefix, meshes=True, assets=assets, animate_grippers=True, armature=1.0,
      link_mass=4.0, link_inertia=0.4, static_mass=1e-4)
  acts, _ = arm_model.gripper_actuator_mjcf(C, [prefix])
  xml = (f'<mujoco><compiler angle="radian"/><option gravity="0 0 0"/>'
         f'<asset>{"".join(assets)}</asset><worldbody>{body}'
         f'</worldbody><actuator>{acts}</actuator></mujoco>')
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  sim_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                for j in range(model.njnt)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(
                    "sim_grip__")]
  assert len(sim_joints) >= 2
  assert all(j not in arm_joints for j in sim_joints)
  assert sorted(arm_joints) == sorted(j for j in C.getJointNames() if j.startswith(prefix))
  assert len(sim_joints) == (2 if robot == "openarm" else 6)
  fixed_geoms = []
  for gi in range(model.ngeom):
    body_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gi])) or ""
    if "sim_grip_fixed__" in body_name:
      assert model.geom_contype[gi] != 0
      assert model.body_mass[int(model.geom_bodyid[gi])] == pytest.approx(1e-4)
      fixed_geoms.append(gi)
  assert fixed_geoms, "animated export dropped the planner's fixed finger collision proxies"
  fixed_before = [(np.asarray(data.geom_xpos[g], dtype=float).copy(),
                   np.asarray(data.geom_xmat[g], dtype=float).copy()) for g in fixed_geoms]
  before = {}
  for name in sim_joints:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0
    body_id = int(model.jnt_bodyid[jid])
    before[name] = np.asarray(data.xmat[body_id], dtype=float).copy()
    gain, max_angle = model.jnt_user[jid, :2]
    data.qpos[int(model.jnt_qposadr[jid])] = 0.5 * gain * max_angle
  mujoco.mj_forward(model, data)
  changed = 0
  for name in sim_joints:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    body_id = int(model.jnt_bodyid[jid])
    assert abs(float(data.qpos[int(model.jnt_qposadr[jid])])) > 1e-3
    changed += np.linalg.norm(np.asarray(data.xmat[body_id]) - before[name]) > 1e-3
  # Robotiq's counter-rotating inner links deliberately keep their WORLD orientation; the
  # driven knuckles (and both OpenArm fingers) must still rotate.
  assert changed >= 2
  for g, (pos, mat) in zip(fixed_geoms, fixed_before):
    assert np.allclose(data.geom_xpos[g], pos, atol=1e-10)
    assert np.allclose(data.geom_xmat[g], mat, atol=1e-10)

  # Reset and reach the same half-closed state through MuJoCo's actuators, rather than by
  # qpos assignment. This catches a cosmetic-joint regression even if its XML still loads.
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  for name in sim_joints:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    gain, max_angle = model.jnt_user[jid, :2]
    data.ctrl[aid] = 0.5 * gain * max_angle
  for _ in range(250):
    mujoco.mj_step(model, data)
  for name in sim_joints:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    gain, max_angle = model.jnt_user[jid, :2]
    assert abs(float(data.qpos[int(model.jnt_qposadr[jid])]) - 0.5 * gain * max_angle) < 0.02


def test_physical_robotiq_uses_one_bounded_actuator_and_mimic_constraints():
  """A contact pinch is one coupled mechanism, not six fighting position servos."""
  C = _env("ur5")
  prefix = A.get_robots(C)[0]
  specs = arm_model.gripper_sim_joints(C, prefix)
  assert len(specs) == 6

  acts, names = arm_model.gripper_actuator_mjcf(C, [prefix], coupled=True)
  equalities = arm_model.gripper_equality_mjcf(C, [prefix])

  assert len(names) == 1
  assert acts.count("<position ") == 1
  assert equalities.count("<joint ") == len(specs) - 1
  assert f'joint2="sim_grip__{next(iter(specs))}"' in equalities

  capped, _ = arm_model.gripper_actuator_mjcf(
      C, [prefix], kp=100.0, coupled=True, max_effort=10.0)
  assert 'forcerange="-10 10"' in capped
  assert 'kp="100"' in capped


def test_tree_is_read_at_zero_not_at_home():
  """A rai joint frame's relative transform contains the CURRENT joint value, so the tree
  must be read at q=0. If it were read at home, moving the scene would change the model."""
  C = _env("ur10")
  xml_home, _ = arm_model.build_arm_mjcf(C, A.get_robots(C)[0])
  q = np.asarray(C.getJointState(), float)
  C.setJointState(q + 0.3)                                   # perturb, then regenerate
  xml_moved, _ = arm_model.build_arm_mjcf(C, A.get_robots(C)[0])
  C.setJointState(q)
  assert xml_home == xml_moved, "generated model depends on the scene's current pose"


def test_model_loads_and_has_one_body_per_frame():
  C = _env("ur10")
  prefix = A.get_robots(C)[0]
  xml, _ = arm_model.standalone_model(C, [prefix])
  m = mujoco.MjModel.from_xml_string(xml)
  frames = [n for n in C.getFrameNames() if n.startswith(prefix)]
  for f in frames:                                           # every rai frame is addressable
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) >= 0, f"{f} missing from MJCF"


def test_per_arm_contact_masks_exclude_self_and_include_other_arm():
  C = _env("ur10", n=2)
  prefixes = A.get_robots(C)
  # part/world bit is 1; arm bits are 2 and 4. Each arm accepts bit 1 and the OTHER arm bit.
  xml, _ = arm_model.standalone_model(
      C, prefixes, collide_fn=arm_model.collides_like_sim,
      contact_masks=[(2, 1 | 4), (4, 1 | 2)])
  m = mujoco.MjModel.from_xml_string(xml)
  seen = {p: 0 for p in prefixes}
  for gi in range(m.ngeom):
    if not m.geom_contype[gi]:
      continue
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi]))
    owner = next(p for p in prefixes if body.startswith(p))
    expected = (2, 5) if owner == prefixes[0] else (4, 3)
    assert (int(m.geom_contype[gi]), int(m.geom_conaffinity[gi])) == expected
    seen[owner] += 1
  assert all(seen.values())


def test_fr3_static_carrier_is_discoverable_and_exportable():
  """The shared TMR carrier is outside both ``aN_`` arm subtrees but must be in sim XML."""
  C = make_fr3_duo_env()
  assert arm_model.fr3_base_prefixes(C) == ["fr30_"]
  assets = []
  body, joints = arm_model.build_arm_mjcf(
      C, "fr30_", collide=False, meshes=True, assets=assets, frame_colors=True,
      static_mass=1e-4)
  xml = f'''<mujoco><asset>{"".join(assets)}</asset><worldbody>{body}</worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  assert joints == []
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr30_base_link") >= 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr30_franka_spine") >= 0


def test_openarm_duo_fk_and_static_torso_export():
  """Seven-axis arms use the generic exporter; their separately-prefixed torso must join it."""
  C = make_openarm_duo_env()
  robots = A.get_robots(C)
  rows, (wp, wa, wf) = fk_compare.compare(C, robots, n=4, seed=0)
  assert len(rows) > 20
  assert wp < 1e-6, f"OpenArm: {wf} position error {wp * 1000:.6f} mm"
  assert wa < 1e-6, f"OpenArm: {wf} orientation error {np.degrees(wa):.6f} deg"
  assert arm_model.openarm_base_prefixes(C) == ["oa0_"]
  assert arm_model.carrier_base_prefixes(C) == ["oa0_"]
  assets = []
  body, joints = arm_model.build_arm_mjcf(
      C, "oa0_", collide=False, meshes=True, assets=assets, frame_colors=True,
      static_mass=1e-4)
  xml = f'''<mujoco><asset>{"".join(assets)}</asset><worldbody>{body}</worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  assert joints == []
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                           "oa0_openarm_body_link0_0") >= 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "oa0_torso_base") >= 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "oa0_torso_column") >= 0


def test_husky_duo_fk_and_static_carrier_export():
  """Calibrated embedded arms use generic FK; the carrier is exported separately."""
  C = make_husky_duo_env()
  robots = A.get_robots(C)
  rows, (wp, wa, wf) = fk_compare.compare(C, robots, n=4, seed=0)
  assert len(rows) > 20
  assert wp < 1e-6, f"Husky: {wf} position error {wp * 1000:.6f} mm"
  assert wa < 1e-6, f"Husky: {wf} orientation error {np.degrees(wa):.6f} deg"
  assert arm_model.husky_base_prefixes(C) == ["husky0_"]
  assert arm_model.carrier_base_prefixes(C) == ["husky0_"]

  assets = []
  body, joints = arm_model.build_arm_mjcf(
      C, "husky0_", collide=False, meshes=True, assets=assets, frame_colors=True,
      static_mass=1e-4)
  xml = f'''<mujoco><asset>{"".join(assets)}</asset><worldbody>{body}</worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  assert joints == []
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "husky0_base_link") >= 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                           "husky0_dual_arm_bulkhead_link") >= 0


def test_gripper_coloring_does_not_create_empty_meshes_on_joint_helpers():
  """Tinting names containing ``finger`` must leave shapeless joint origins shapeless.

  RAI otherwise turns ``setColor`` on such a frame into a zero-vertex mesh, which later makes
  FCL abort during motion planning instead of returning a collision result.
  """
  C = ry.Config()
  C.addFile("src/rai/robotiq/robotiq.g", namePrefix="x_")
  helper = C.getFrame("x_finger_joint_origin")
  assert "shape" not in helper.info()
  color_gripper(C, "x_", [0.9, 0.1, 0.1, 0.3])
  assert "shape" not in helper.info()
