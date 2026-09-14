"""Regression tests for live rigid-component contact bookkeeping."""

from types import SimpleNamespace

import numpy as np

import sim as S


def _scene(active):
  scene = S.SimScene.__new__(S.SimScene)
  # obj_0--obj_1--obj_2 is a potential chain; obj_3 is a separate branch.
  scene._mate_eqs = {
      "obj_1": (0, "obj_0"),
      "obj_2": (1, "obj_1"),
      "obj_3": (2, "obj_0"),
  }
  scene.data = SimpleNamespace(eq_active=np.asarray(active, dtype=np.int32))
  return scene


def test_active_mate_component_is_transitive_and_live():
  scene = _scene([1, 1, 0])

  assert scene._active_mate_component("obj_0") == {"obj_0", "obj_1", "obj_2"}
  assert scene._active_mate_component("obj_2") == {"obj_0", "obj_1", "obj_2"}
  assert scene._active_mate_component("obj_3") == {"obj_3"}

  # A broken/deactivated intermediate mate immediately splits the live component.
  scene.data.eq_active[0] = 0
  assert scene._active_mate_component("obj_0") == {"obj_0"}
  assert scene._active_mate_component("obj_2") == {"obj_1", "obj_2"}


def test_cargo_clearance_uses_lowest_geom_of_live_mate_component():
  scene = _scene([1, 0, 0])
  scene.active_grasps = {"a1": (7, "obj_1", 1.0, 1.0)}
  # geom 0 belongs to the directly picked obj_1 and is already clear; geom 1 belongs to its
  # mated parent obj_0 and is still only 1mm above the 0.5m table.
  scene.model = SimpleNamespace(
      ngeom=2,
      geom_bodyid=np.asarray([1, 2]),
      geom_size=np.asarray([[0.01, 0.01, 0.01], [0.01, 0.01, 0.02]]))
  scene.data.geom_xmat = np.tile(np.eye(3).reshape(1, 9), (2, 1))
  scene.data.geom_xpos = np.asarray([[0.0, 0.0, 0.53], [0.0, 0.0, 0.521]])
  scene._part_owner_by_body = [None, "obj_1", "obj_0"]

  servo = S._SkillServo.__new__(S._SkillServo)
  servo.scene = scene
  servo._ttop = 0.5
  servo._part_geoms = {}

  assert servo._cargo_clearance("a1") == 0.0010000000000000009


def test_pick_configuration_correction_ends_at_certified_post_pick():
  scene = S.SimScene.__new__(S.SimScene)
  scene._rai_joints = ["a1_j0", "a1_j1"]
  scene._rai_hinge = [True, True]
  scene._qadr = {"a1_j0": 0, "a1_j1": 1}
  scene.data = SimpleNamespace(qpos=np.asarray([0.2, -0.1]))
  scene.active_grasps = {"a1": (0, "obj_0", 1.0, 1.0)}

  servo = S._SkillServo.__new__(S._SkillServo)
  servo.scene = scene
  servo.qidx = {"a1": [0, 1]}
  servo.capture_qoff = {}
  servo.capture_clearance = {}
  servo.capture_hold = set()
  servo.capture_override_q = {}
  servo.pick_clearance = 0.004
  servo.off = {}
  servo.decay = 0.1
  servo._cargo_clearance = lambda _robot: None
  servo.inherit_pick_configuration(np.zeros(2), "a1")
  servo.capture_override_q["a1"] = np.asarray([0.3, -0.2])
  assert np.allclose(servo.adjust(np.zeros(2)), [0.3, -0.2])
  servo.finish_pick_retreat("a1")

  first = servo.adjust(np.zeros(2))
  second = servo.adjust(np.zeros(2))
  assert np.allclose(first, [0.0, 0.0])
  assert np.allclose(second, [0.0, 0.0])
  assert "a1" not in servo.capture_qoff


def test_pick_configuration_override_owns_retreat_then_persists():
  scene = S.SimScene.__new__(S.SimScene)
  scene._rai_joints = ["a1_j0"]
  scene._rai_hinge = [True]
  scene._qadr = {"a1_j0": 0}
  scene.data = SimpleNamespace(qpos=np.asarray([0.2]))
  scene.active_grasps = {}

  servo = S._SkillServo.__new__(S._SkillServo)
  servo.scene = scene
  servo.qidx = {"a1": [0]}
  servo.capture_qoff = {}
  servo.capture_clearance = {}
  servo.capture_hold = set()
  servo.capture_override_q = {}
  servo.pick_clearance = 0.004
  servo.off = {}
  servo.decay = 0.1
  servo.inherit_pick_configuration(
      np.zeros(1), "a1", position_correction=0.015)
  servo.capture_override_q["a1"] = np.asarray([0.3])

  # Before attachment/post-pick handoff, the skill-owned lifted posture is authoritative.
  assert np.allclose(servo.adjust(np.zeros(1)), [0.3])
  scene.active_grasps["a1"] = (0, "obj_0", 1.0, 1.0)
  servo.finish_pick_retreat("a1")
  assert np.allclose(servo.adjust(np.zeros(1)), [0.0])
  assert np.allclose(servo.adjust(np.zeros(1)), [0.0])


def test_equality_row_cache_tracks_realized_not_pending_active_set():
  """A same-cardinality handover must not relabel the previous step's efc rows."""
  eq_type = int(S.mujoco.mjtConstraint.mjCNSTR_EQUALITY)
  old_active = np.asarray([1, 0, 1], dtype=np.uint8)
  scene = S.SimScene.__new__(S.SimScene)
  scene.data = SimpleNamespace(
      eq_active=old_active.copy(), ne=12,
      efc_type=np.full(12, eq_type, dtype=int),
      efc_id=np.repeat([0, 2], 6))
  scene._efc_active = old_active.tobytes()
  scene._eq_cache = (None, None, None)
  scene.verify_efc_cache = False
  scene.step_i = 0

  _, ids = scene._eq_index()
  assert np.array_equal(ids, np.repeat([0, 2], 6))

  # Requested handover: old equality 0 off, equality 1 on, same total row count. Until the
  # next mj_step the efc arrays still belong to the OLD set and must retain their labels.
  new_active = np.asarray([0, 1, 1], dtype=np.uint8)
  scene.data.eq_active[:] = new_active
  _, ids = scene._eq_index()
  assert np.array_equal(ids, np.repeat([0, 2], 6))

  # What step() does immediately after MuJoCo rebuilds the constraints.
  scene.data.efc_id[:] = np.repeat([1, 2], 6)
  scene._efc_active = new_active.tobytes()
  _, ids = scene._eq_index()
  assert np.array_equal(ids, np.repeat([1, 2], 6))


def test_grasp_weld_freezes_achieved_capture_pose_not_ideal_target():
  """Closing a grasp must not turn the bounded capture residual into weld preload."""
  scene = S.SimScene.__new__(S.SimScene)
  scene.grasp_anchor = {"a1": "a1_anchor"}
  scene.gripper_kinds = {"a1": "vacuum"}
  scene.active_grasps = {}
  scene._hold_info = {}
  scene._holdex = {}
  scene._eq_id = lambda _name: 7
  scene._body_id = lambda name: {"a1_anchor": 2, "obj_0": 3}[name]
  activated = []
  scene._activate_weld = lambda eq, b1, b2, relative_pose=None: activated.append(
      (eq, b1, b2, relative_pose))

  # This is the ideal contacted-part_T_anchor relation used by the capture controller.  The
  # actual bodies can legitimately remain up to GRASP_ALIGN_TOL away when the hand closes.
  ideal = np.asarray([0.01, -0.02, 0.03, 1.0, 0.0, 0.0, 0.0])
  scene.grasp("a1", "obj_0", grasp_relpose=ideal)

  assert activated == [(7, 2, 3, None)]
  assert scene.active_grasps["a1"][:2] == (7, "obj_0")
