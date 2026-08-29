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


class _JointConfig:
  def __init__(self):
    self.q = np.zeros(2)

  def getJointState(self):
    return self.q.copy()

  def setJointState(self, q):
    self.q = np.asarray(q, dtype=float).copy()

  def getJointLimits(self):
    return np.array([-1.0, -2.0]), np.array([1.0, 2.0])

  def getJointNames(self):
    return ["a1_joint1", "a1_joint2"]

  def getCollisions(self, *_args):
    return []


class _FourJointConfig(_JointConfig):
  def __init__(self):
    self.q = np.zeros(4)

  def getJointLimits(self):
    return np.array([-1.0] * 4), np.array([1.0] * 4)

  def getJointNames(self):
    return ["a1_joint1", "a1_joint2", "a2_joint1", "a2_joint2"]


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


def test_assemble_actor_hand_mate_target_is_margin_exempt_but_other_pairs_are_not():
  rik = _rik()
  target = _Frame("obj_0")
  C = _Config({"obj_0": target, "obj_1": _Frame("obj_1", target)})
  rik._cur_action = ("assemble", "a1", "obj_1", "obj_0")
  # The inserted-part hand is necessarily close to the part it is mating against. Strict
  # collision remains a separate gate; only the 5mm free-space margin is inapplicable here.
  assert rik._clearance_pair_reason(C, "a1_ur_finger", "obj_0__shape") is None
  assert rik._clearance_pair_reason(C, "obj_0", "a1_ur_palm") is None
  # OpenArm exports its rigid palm/base capsules as ee_coll{,1,2}, without a palm/gripper
  # token. They obey the same intended-margin contract as the finger capsules.
  assert rik._clearance_pair_reason(C, "a1_left_ee_coll1", "obj_0") is None
  assert rik._clearance_pair_reason(C, "obj_0", "a1_right_ee_coll2") is None
  assert rik._clearance_pair_reason(C, "a1_ur_finger", "obj_1__shape") is None
  # Do not broaden this to arm links, another robot, clutter, or an unrelated part.
  assert rik._clearance_pair_reason(C, "a1_ur_coll8", "obj_0") == "arm-part"
  assert rik._clearance_pair_reason(C, "a2_ur_finger", "obj_0") == "arm-part"
  assert rik._clearance_pair_reason(C, "a1_ur_finger", "obj_9") == "arm-part"


def test_coupled_task_hands_use_strict_collision_not_free_space_margin():
  rik = _rik()
  a1g, a2g = _Frame("a1_ur_vacuum"), _Frame("a2_ur_vacuum")
  C = _Config({"obj_0": _Frame("obj_0", a2g), "obj_1": _Frame("obj_1", a1g)})
  rik._cur_action = ("assemble", "a1", "obj_1", "obj_0")
  assert rik._clearance_pair_reason(C, "a1_ur_finger", "a2_ur_finger") is None
  # A bystander is not part of the rendezvous and retains the ordinary robot margin.
  rik.robots.append("a3")
  rik.gripper_frame["a3"] = "a3_ur_vacuum"
  assert rik._clearance_pair_reason(C, "a1_ur_finger", "a3_ur_finger") == "inter-robot"

  # Outside a coupled endpoint, even the same pair retains the free-space margin.
  rik._cur_action = ("pick", "a1", "obj_1", 0)
  assert rik._clearance_pair_reason(C, "a1_ur_finger", "a2_ur_finger") == "inter-robot"


def test_free_pose_handover_payload_must_clear_the_table():
  rik = _rik()
  a1g = _Frame("a1_ur_vacuum")
  C = _Config({"obj_0": _Frame("obj_0", a1g)})

  rik._cur_action = ("handover", "a2", "obj_0", 3)
  assert rik._clearance_pair_reason(C, "obj_0__shape", "table") == "payload-table"
  # Pick/place table contact is intentional and must not inherit the transport buffer.
  rik._cur_action = ("pick", "a2", "obj_0", 3)
  assert rik._clearance_pair_reason(C, "obj_0__shape", "table") is None


def test_joint_limit_margin_is_part_of_the_same_search_time_gate():
  rik = _rik()
  rik._cur_action = ("assemble", "a1", "obj_0", "obj_1", 0)
  rik.clearance_reject = None
  rik.joint_margin_reject = 0.02
  rik.margin_rejects = 0
  rik.margin_reject_pairs = {}
  rik.last_margin_reject = None
  C = _JointConfig()
  assert rik._clearance_ok(C, [0.0, 0.0])
  assert not rik._clearance_ok(C, [0.0, 1.999999])
  assert rik.last_margin_reject[3] == "joint-limit"


def test_joint_limit_margin_does_not_filter_an_ordinary_pick():
  rik = _rik()
  rik._cur_action = ("pick", "a1", "obj_0", 0)
  rik.clearance_reject = None
  rik.joint_margin_reject = 0.02
  rik.margin_rejects = 0
  rik.margin_reject_pairs = {}
  rik.last_margin_reject = None
  assert rik._clearance_ok(_JointConfig(), [0.0, 1.999999])
  assert rik.margin_rejects == 0


def test_joint_limit_margin_ignores_arm_less_floating_pose_coordinates():
  rik = _rik()
  rik.robot_arm = {"a1": None, "a2": None}
  rik._cur_action = ("assemble", "a1", "obj_0", "obj_1")
  rik.clearance_reject = None
  rik.joint_margin_reject = 0.02
  rik.margin_rejects = 0
  rik.margin_reject_pairs = {}
  rik.last_margin_reject = None
  assert rik._clearance_ok(_JointConfig(), [0.0, 1.999999])
  assert rik.margin_rejects == 0


def test_joint_limit_margin_ignores_nonparticipant_arm_at_home_stop():
  rik = _rik()
  rik._cur_action = ("assemble", "a1", "obj_0", "obj_1")
  rik.clearance_reject = None
  rik.joint_margin_reject = 0.02
  rik.margin_rejects = 0
  rik.margin_reject_pairs = {}
  rik.last_margin_reject = None
  # a1 is comfortably interior; uninvolved a2 sits exactly at its upper limit.
  assert rik._clearance_ok(_FourJointConfig(), [0.0, 0.0, 0.0, 1.0])
  # The same boundary is still rejected when it belongs to the actor.
  assert not rik._clearance_ok(_FourJointConfig(), [0.0, 1.0, 0.0, 0.0])
  assert rik.last_margin_reject[0] == "a1_joint2"


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


def test_ik_feeds_a_failed_margin_certificate_back_into_komo():
  """A nominally feasible optimum is refined, not merely rejected and randomized."""
  rik = object.__new__(S.RobotIK)
  rik._cur_action = ("pick", "a1", "obj_0", 0)
  rik.seed_init_incompatible = set()
  rik.seed_init_skips = 0
  rik.harvest_branches = None
  rik.seed_restart_cap = 2
  rik.restarts_full = 0
  rik._open_probe_attempt = None
  rik.clearance_reject = 0.005
  rik.joint_margin_reject = 0.02
  rik.col_weight = 10.0
  rik.last_margin_reject = None
  rik.prof = {
      "calls_with_seeds": 0, "calls_without_seeds": 0, "wins": {},
      "seeded_solves": 0, "seeded_time": 0.0,
      "unseeded_solves": 0, "unseeded_time": 0.0,
      "restart_solves": 0, "restart_time": 0.0,
      "getpath_calls": 0, "post_calls": 0, "post_time": 0.0,
  }
  rik._pen_ok = lambda _C, _q: True
  clearance_calls = []

  def clearance(_C, _q):
    clearance_calls.append(1)
    if len(clearance_calls) == 1:
      rik.last_margin_reject = ("a1_link", "obj_9", 0.001, "arm-part")
      return False
    return True

  rik._clearance_ok = clearance

  class _Result:
    def dict(self):
      return {"feasible": True}

  class _NLP:
    def solve(self):
      return _Result()

  class _Komo:
    def __init__(self):
      self.objectives = []
      self.warm_starts = []

    def getPath(self):
      return np.zeros((1, 2))

    def addObjective(self, *args):
      self.objectives.append(args)

    def initWithConstant(self, q):
      self.warm_starts.append(np.asarray(q).copy())

  komo = _Komo()
  rik._nlp = lambda _komo: _NLP()
  ok, q = rik._solve_ik(komo, restarts=0, C=object())
  assert ok
  assert q.shape == (2,)
  assert len(clearance_calls) == 2
  assert len(komo.objectives) == 1
  assert len(komo.warm_starts) == 1
  assert rik.prof["margin_refine_solves"] == 1
