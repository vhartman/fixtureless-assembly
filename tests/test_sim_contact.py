"""Behavioral guards for the separate physical-contact executor."""

from types import SimpleNamespace

import numpy as np

import pytest

import sim
import sim_contact


def _scene(monkeypatch):
  scene = sim_contact.ContactScene.__new__(sim_contact.ContactScene)
  scene.contact_grasps = True
  scene.active_grasps = {"a1": (-1, "obj_1"), "a2": (-1, "obj_0")}
  scene._physical_contact_target = {"a1": "obj_1", "a2": "obj_0"}
  scene._hold_info = {"a1": {"grasp_part": "obj_1"},
                      "a2": {"grasp_part": "obj_0"}}
  scene._contact_grasp_misses = {"a1": 2, "a2": 1}
  scene._contact_grasp_slips = {"a1": 0, "a2": 0}
  scene._contact_grasp_ref = {"a1": np.array([0, 0, 0, 1, 0, 0, 0]),
                              "a2": np.array([0, 0, 0, 1, 0, 0, 0])}
  scene._contact_relative_pose = lambda *_: np.array([0, 0, 0, 1, 0, 0, 0])
  scene._holdex = {}
  scene.model = SimpleNamespace(geom_contype=np.zeros(0, dtype=int),
                                geom_conaffinity=np.zeros(0, dtype=int))
  scene.data = object()
  scene.assembly = SimpleNamespace(objects={"obj_0", "obj_1"})
  monkeypatch.setattr(sim_contact.mujoco, "mj_forward", lambda *_: None)
  return scene

def test_mate_preserves_live_contact_holder(monkeypatch):
  scene = _scene(monkeypatch)
  # OpenArm includes moving pads in the historical wrist-exemption set.
  # Mating must not disable those load-bearing physical contacts.
  scene._holdex = {"a1": [0, 1]}
  scene.model.geom_contype = np.array([2, 2])
  scene.model.geom_conaffinity = np.array([1, 1])
  def old_weld(self, part, *, project=True):
    assert not project, 'physical mate must not project the part out of its hand'
    for robot, (_, held) in list(self.active_grasps.items()):
      if held == part:
        self.release(robot)
  monkeypatch.setattr(sim.SimScene, "weld_mate", old_weld)
  scene.weld_mate("obj_1")
  assert scene.active_grasps == {"a1": (-1, "obj_1"), "a2": (-1, "obj_0")}
  assert scene._physical_contact_target["a1"] == "obj_1"
  assert scene._contact_grasp_misses["a1"] == 2
  assert scene._contact_grasp_slips["a1"] == 0
  assert np.array_equal(scene._contact_grasp_ref["a1"], [0, 0, 0, 1, 0, 0, 0])
  assert np.array_equal(scene.model.geom_contype, [2, 2])
  assert np.array_equal(scene.model.geom_conaffinity, [1, 1])

def test_release_reroot_does_not_regrasp_continuing_holder(monkeypatch):
  scene = _scene(monkeypatch)
  scene.expect_state = SimpleNamespace(supports={
      "obj_0": (("grasp", "a2", "obj_0", 0),),
      "obj_1": (("part", "obj_0"),)})
  def no_grasp(*_a, **_kw):
    raise AssertionError("release re-root must not acquire a new grasp")
  monkeypatch.setattr(sim.SimScene, "attach", no_grasp)
  assert scene.attach("a2_gripper", "obj_0")
  assert set(scene.active_grasps) == {"a2"}

def test_mate_does_not_detach_physical_grasp_first(monkeypatch):
  scene = _scene(monkeypatch)
  scene.expect_state = SimpleNamespace(supports={
      "obj_0": (("grasp", "a2", "obj_0", 0),),
      "obj_1": (("part", "obj_0"), ("grasp", "a1", "obj_1", 0))})
  monkeypatch.setattr(scene, "_detach", lambda *_: (_ for _ in ()).throw(
      AssertionError("a mate must not first release the physical hand")))
  calls = []
  monkeypatch.setattr(scene, "weld_mate", lambda part: calls.append(part))
  assert scene.attach("obj_0", "obj_1")
  assert calls == ["obj_1"]
  assert set(scene.active_grasps) == {"a1", "a2"}




def test_scope_rejects_uncertified_contacts_and_vacuum():
  args = SimpleNamespace(mp_path="path.json", execution_timing="phase",
                         actuated=True, render_kinematic=False, physics_step=0.002)
  with pytest.raises(SystemExit, match="grasp-physics-check"):
    sim_contact._check_scope({"grippers": "two_finger,two_finger"}, args)
  with pytest.raises(SystemExit, match="all two-finger"):
    sim_contact._check_scope({"grippers": "vacuum,two_finger",
                        "grasp_physics_check": True}, args)


def test_handover_acquires_receiver_before_releasing_giver(monkeypatch):
  scene = _scene(monkeypatch)
  scene.active_grasps = {"a1": (-1, "obj_1")}
  scene._physical_contact_target = {"a1": "obj_1"}
  scene.expect_state = SimpleNamespace(supports={
      "obj_1": (("grasp", "a2", "obj_1", 4),)})
  events = []
  def acquire(self, _parent, _child, grasp_relpose=None,
              preserve_existing_grasps=False):
    assert preserve_existing_grasps and "a1" in self.active_grasps
    self.active_grasps["a2"] = (-1, "obj_1")
    self._physical_contact_target["a2"] = "obj_1"
    events.append("receiver-contact")
    return True
  monkeypatch.setattr(sim.SimScene, "attach", acquire)
  original_release = scene.release
  def release(robot):
    if robot in scene.active_grasps:
      events.append("release-" + robot)
    original_release(robot)
  monkeypatch.setattr(scene, "release", release)
  assert scene.contact_handover("a2_gripper", "obj_1", ["a1"])
  assert events == ["receiver-contact", "release-a1"]
  assert set(scene.active_grasps) == {"a2"}


def test_contact_drift_uses_measured_relative_pose_and_quaternion_sign():
  scene = sim.SimScene.__new__(sim.SimScene)
  scene._contact_grasp_ref = {"a1": np.array([0., 0., 0., 1., 0., 0., 0.])}
  scene._contact_relative_pose = lambda *_: np.array([0.002, 0., 0., -1., 0., 0., 0.])
  shift, angle = scene._contact_grasp_drift("a1", "obj_0")
  assert shift == pytest.approx(0.002)
  assert angle == pytest.approx(0.0)
  scene._contact_relative_pose = lambda *_: np.array([0.012, 0., 0., 1., 0., 0., 0.])
  shift, _angle = scene._contact_grasp_drift("a1", "obj_0")
  assert shift > sim.CONTACT_SLIP_SHIFT


def test_admittance_excludes_own_physical_fingers_but_sees_released_hand(monkeypatch):
  scene = sim.SimScene.__new__(sim.SimScene)
  scene.contact_grasps = True
  scene.active_grasps = {"a1": (-1, "obj_1", 0.0, 0.0)}
  scene.grasp_anchor = {"a1": "a1_gripper_center"}
  scene._body_id = lambda _name: 2
  scene._active_mate_component = lambda part: frozenset((part,))
  scene._part_owner_by_body = [None, "obj_1", None, None, None]
  scene._phys = 0
  scene.model = SimpleNamespace(
      nbody=5, body_parentid=np.array([0, 0, 0, 2, 0]),
      geom_bodyid=np.array([3, 4, 1]))
  contact = lambda geom: SimpleNamespace(
      geom1=geom, geom2=2, frame=np.eye(3))
  scene.data = SimpleNamespace(
      eq_active=np.zeros(0, dtype=int), ncon=2,
      contact=[contact(0), contact(1)])
  names = {0: "world", 1: "obj_1", 2: "a1_gripper_center",
           3: "a1_finger", 4: "a2_finger"}
  monkeypatch.setattr(sim.mujoco, "mj_id2name",
                      lambda _model, _kind, body: names[int(body)])
  def contact_force(_model, _data, index, out):
    out[:] = 0.0
    out[0] = 10.0 if index == 0 else 4.0
  monkeypatch.setattr(sim.mujoco, "mj_contactForce", contact_force)

  assert scene.net_contact_force("obj_1")[0] == pytest.approx(4.0)
  scene.active_grasps.clear()
  # The same contact from a released hand is now an external collision.
  assert scene.net_contact_force("obj_1")[0] == pytest.approx(14.0)
