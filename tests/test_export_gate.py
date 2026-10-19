"""The final motion-export gate covers every IK skill, not only assembly seats."""
import numpy as np

import assembly_planner.mp_export as mp_export
import assembly_planner.solver as solver


class _Scene:
  def __init__(self):
    self.q = np.zeros(1)

  def setJointState(self, q):
    self.q = np.asarray(q, dtype=float).ravel()

  def getCollisionsTotalPenetration(self):
    # Positive q encodes penetration in this deliberately tiny fake scene.
    return max(0.0, float(self.q[0]))


class _RobotIK:
  penetration_reject = 1e-4
  keep_contacts = True

  def scene_for(self, _state, _assembly):
    return _Scene()


def _run(monkeypatch, shipped, baseline):
  monkeypatch.setattr(solver, "_replay_states", lambda *a, **k: [object(), object()])
  plan = [("handover", "a1", "obj_0", 13)]
  return solver.validate_export_witnesses(
      None, None, plan, [np.array([shipped])], ["a1", "a2"], _RobotIK(),
      grasp_table={}, baseline_witnesses=[np.array([baseline])], verbose=False)


def test_dirty_handover_is_restored_from_clean_search_witness(monkeypatch):
  witnesses, bad = _run(monkeypatch, 1.22e-3, 0.0)
  assert bad == []
  np.testing.assert_allclose(witnesses[0], [0.0])


def test_dirty_handover_with_dirty_baseline_fails_closed(monkeypatch):
  witnesses, bad = _run(monkeypatch, 1.22e-3, 0.8e-3)
  np.testing.assert_allclose(witnesses[0], [1.22e-3])
  assert len(bad) == 1
  assert bad[0][:2] == (0, ("handover", "a1", "obj_0", 13))
  assert bad[0][2] == 1.22e-3


class _Q:
  def __init__(self, parts):
    self.parts = [np.asarray(x, dtype=float) for x in parts]

  def robot_state(self, i):
    return self.parts[i]

  @classmethod
  def from_list(cls, parts):
    return cls(parts)


class _Mode:
  def __init__(self, task_ids):
    self.task_ids = list(task_ids)
    self.next_modes = []


class _Goal:
  def sample(self, _mode):
    return np.array([0.2])


class _Task:
  robots = ["a1"]
  goal = _Goal()
  name = "pick"


class _EmptyCollisionScene:
  def getCollisions(self):
    return []


class _ChainEnv:
  def __init__(self):
    self.robots = ["a1"]
    self.robot_dims = {"a1": 1}
    self.start_mode = _Mode([0])
    self.C = _EmptyCollisionScene()

  def get_start_pos(self):
    return _Q([[0.0]])

  def is_terminal_mode(self, mode):
    return mode.task_ids == [1]

  def get_valid_next_task_combinations(self, _mode):
    return [[1]]

  def get_active_task(self, _mode, _ids):
    return _Task()

  def is_collision_free(self, _q, _mode):
    return True

  def get_next_modes(self, _q, mode):
    child = _Mode([1])
    mode.next_modes.append(child)  # same mutating contract as mr_bench
    return [child]


def test_nominal_chain_validation_does_not_prepopulate_planner_frontier():
  env = _ChainEnv()
  existing = _Mode([-1])
  env.start_mode.next_modes.append(existing)

  assert mp_export.validate_nominal_keyframes(env, tolerance=0.0, verbose=False) == []
  assert env.start_mode.next_modes == [existing]
