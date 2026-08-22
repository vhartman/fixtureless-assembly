"""Shared search/witness hard-clearance pair classification."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import solver as S


class _Frame:
  def __init__(self, name, parent=None):
    self.name = name
    self._parent = parent

  def getParent(self):
    return self._parent


class _Config:
  def __init__(self, frames):
    self.frames = frames

  def getFrame(self, name):
    return self.frames[name]


def _rik():
  rik = object.__new__(S.RobotIK)
  rik.robots = ["a1", "a2"]
  rik.gripper_frame = {"a1": "a1_ur_vacuum", "a2": "a2_ur_vacuum"}
  rik._cur_action = None
  return rik


def test_gate_covers_inter_robot_self_and_unrelated_hand_part_pairs():
  rik = _rik()
  C = _Config({})
  assert rik._clearance_pair_reason(C, "a1_ur_coll8", "a2_ur_coll5") == "inter-robot"
  assert rik._clearance_pair_reason(C, "a1_ur_coll8", "a1_ur_coll5") == "self"
  assert rik._clearance_pair_reason(C, "a1_ur_vacuum", "obj_7") == "arm-part"
  # Internal hand mechanics are intended; rai already excludes adjacent arm-link pairs.
  assert rik._clearance_pair_reason(C, "a1_ur_palm", "a1_ur_finger") is None


def test_only_the_current_grasp_target_or_own_attached_payload_is_exempt():
  rik = _rik()
  a1g = _Frame("a1_ur_vacuum")
  own = _Frame("obj_0", a1g)
  other = _Frame("obj_1")
  C = _Config({"obj_0": own, "obj_1": other})

  rik._cur_action = ("pick", "a1", "obj_1", 4)
  assert rik._clearance_pair_reason(C, "a1_ur_vacuum", "obj_1__shape") is None
  assert rik._clearance_pair_reason(C, "a1_ur_vacuum", "obj_0") is None
  assert rik._clearance_pair_reason(C, "a2_ur_vacuum", "obj_1") == "arm-part"
  assert rik._clearance_pair_reason(C, "a1_ur_vacuum", "obj_9") == "arm-part"


def test_ik_skips_a_seed_when_komo_exposes_a_wider_internal_q():
  """rai may retain a stable-joint qIndex hole after reparenting a released assembly.

  Calling initWithConstant with the six named arm values can abort in C++; the ordinary KOMO
  initializer is still valid and its result must pass the unchanged penetration/clearance
  gates.
  """
  rik = object.__new__(S.RobotIK)
  rik._cur_action = ("pick", "a1", "obj_0", 0)
  rik.seed_init_incompatible = set()
  rik.seed_init_skips = 0
  rik.harvest_branches = None
  rik.seed_restart_cap = 2
  rik.restarts_full = 12
  rik._open_probe_attempt = None
  rik.prof = {
      "calls_with_seeds": 0, "calls_without_seeds": 0, "wins": {},
      "seeded_solves": 0, "seeded_time": 0.0,
      "unseeded_solves": 0, "unseeded_time": 0.0,
      "restart_solves": 0, "restart_time": 0.0,
      "getpath_calls": 0, "post_calls": 0, "post_time": 0.0,
  }
  rik._pen_ok = lambda _C, _q: True
  rik._clearance_ok = lambda _C, _q: True

  class _Result:
    def dict(self):
      return {"feasible": True}

  class _NLP:
    def solve(self):
      return _Result()

  class _Komo:
    def getPath(self):
      return np.zeros((1, 7))

    def initWithConstant(self, _q):
      raise AssertionError("shape mismatch must be detected before calling rai")

  class _C:
    def getJointNames(self):
      return [f"j{i}" for i in range(6)]

  rik._nlp = lambda _komo: _NLP()
  ok, q = rik._solve_ik(_Komo(), seeds=[np.zeros(6)], C=_C())
  assert ok
  assert q.shape == (7,)
  assert rik.seed_init_skips == 1
