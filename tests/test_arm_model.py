"""The articulated MuJoCo arm generated from rai must agree with rai's own FK.

This is the guarantee the whole articulated-sim plan rests on: the planner expresses every
witness, strict-collision gate and seat tolerance in rai's joint space, so if MuJoCo's
kinematics drift even slightly, the same q means a different end-effector pose in the two and
witnesses land somewhere else -- silently. Cheap to check, catastrophic to get wrong, so it is
pinned here rather than left to fk_compare.py being run by hand.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

import arm_model
import fk_compare
from env import make_four_robot_env, make_fr3_duo_env, make_openarm_duo_env
import assembly as A


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
