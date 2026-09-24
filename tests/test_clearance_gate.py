"""Shared search/witness hard-clearance pair classification."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import solver as S
import robotic as ry


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


def _workcell_scene(x=0.0, y=0.0):
  C = ry.Config()
  C.addFrame("table").setShape(ry.ST.box, [1.0, 0.6, 0.05]).setPosition([0, 0, 0])
  hand = C.addFrame("gripper").setPosition([x, y, 0.2])
  C.addFrame("obj_0").setParent(hand).setRelativePosition([0, 0, 0])
  C.addFrame("obj_1").setPosition([10.0, 10.0, 0.2])       # unheld: deliberately ignored
  return C


def test_object_over_table_uses_local_table_footprint_and_only_held_parts():
  rik = object.__new__(S.RobotIK)
  rik.object_over_table = True
  rik.object_over_table_margin = 0.05
  rik.gripper_frame = {"a1": "gripper"}
  rik.object_frames = ("obj_0", "obj_1")
  assert rik._object_over_table_violation(_workcell_scene(0.44, 0.0)) is None
  violation = rik._object_over_table_violation(_workcell_scene(0.46, 0.0))
  assert violation[0] == "obj_0"
  assert np.isclose(violation[1], 0.01)


def test_object_over_table_checks_one_reference_datum_per_held_component_and_allows_overhang():
  rik = object.__new__(S.RobotIK)
  rik.object_over_table = True
  rik.object_over_table_margin = -0.1
  rik.gripper_frame = {"a1": "gripper"}
  rik.object_frames = ("obj_0", "obj_1")
  C = ry.Config()
  C.addFrame("table").setShape(ry.ST.box, [1.0, 0.6, 0.05])
  hand = C.addFrame("gripper").setPosition([0.58, 0.0, 0.2])
  root = C.addFrame("obj_0").setParent(hand).setRelativePosition([0, 0, 0])
  # A constituent datum may extend past the reference bound; it is not a second independent
  # workcell constraint on the same rigid payload.
  C.addFrame("obj_1").setParent(root).setRelativePosition([0.3, 0, 0])
  assert rik._held_object_frames(C) == ("obj_0",)
  assert rik._object_over_table_violation(C) is None
  hand.setPosition([0.61, 0.0, 0.2])
  violation = rik._object_over_table_violation(C)
  assert violation[0] == "obj_0"
  assert np.isclose(violation[1], 0.01)


def test_object_over_table_ignores_payloads_of_inactive_exit_arms():
  rik = object.__new__(S.RobotIK)
  rik.object_over_table = True
  rik.object_over_table_margin = 0.0
  rik.robots = ["a1", "a2"]
  rik.gripper_frame = {"a1": "a1_gripper", "a2": "a2_gripper"}
  rik.object_frames = ("obj_0", "obj_1")
  C = ry.Config()
  C.addFrame("table").setShape(ry.ST.box, [1.0, 0.6, 0.05])
  hand1 = C.addFrame("a1_gripper").setPosition([0.8, 0.0, 0.2])
  C.addFrame("a2_gripper").setPosition([0.0, 0.0, 0.2])
  C.addFrame("obj_0").setParent(hand1).setRelativePosition([0, 0, 0])
  C.addFrame("obj_1").setPosition([0.0, 0.0, 0.05])

  # obj_0 is outside only because uninvolved a1 is parked at the isolated IK EXIT pose.
  rik._cur_action = ("pick", "a2", "obj_1", 0)
  assert rik._object_over_table_violation(C) is None

  # Once the skill touches obj_0, its current holder is a participant and the same pose is
  # correctly rejected.
  rik._cur_action = ("handover", "a2", "obj_0", 0)
  violation = rik._object_over_table_violation(C)
  assert violation[0] == "obj_0"
  assert np.isclose(violation[1], 0.3)


def test_object_over_table_adds_four_hard_bounds_per_held_part():
  rik = object.__new__(S.RobotIK)
  rik.object_over_table = True
  rik.object_over_table_margin = 0.0
  rik.gripper_frame = {"a1": "gripper"}
  rik.object_frames = ("obj_0", "obj_1")
  C = _workcell_scene()
  komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=False)
  # This is also an API-shape regression: RAI validates the 1x3 scale rows at insertion.
  rik._add_object_over_table_objectives(komo, C)
  assert len(komo.info_objectiveNames()) == 4


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


@pytest.mark.parametrize("penetration", [False, True])
def test_ik_feeds_a_failed_margin_certificate_back_into_komo(penetration):
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
  rik._mate_exempt = None
  pen_calls = []
  def pen(_C, _q):
    pen_calls.append(1)
    if penetration and len(pen_calls) == 1:
      rik.last_strict_reject = (4e-5, [("a1_link", "table", -4e-5)])
      return False
    return True
  rik._pen_ok = pen
  clearance_calls = []

  def clearance(_C, _q):
    clearance_calls.append(1)
    if not penetration and len(clearance_calls) == 1:
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
  assert len(clearance_calls) == (1 if penetration else 2)
  assert len(pen_calls) == 2
  assert len(komo.objectives) == 1
  assert len(komo.warm_starts) == 1
  assert rik.prof["margin_refine_solves"] == 1


def test_handover_racing_continues_the_best_shallow_random_start():
  """A shallow miss only ranks starts; the promoted full solve still owns acceptance."""
  rik = object.__new__(S.RobotIK)
  rik._cur_action = ("handover", "a2", "obj_0", 3)
  rik._cur_query_key = (rik._cur_action, ("physical",))
  rik.seed_init_incompatible = set()
  rik.seed_init_skips = 0
  rik.harvest_branches = None
  rik.seed_restart_cap = 2
  rik.restarts_full = 1
  rik._open_probe_attempt = 0
  rik._open_probe_budget = 2
  rik._open_probe_random_only = False
  rik._handover_race_cache = {}
  rik.handover_race_probe_evals = 24
  rik.handover_race_starts = 3
  rik.handover_race_promote = 2
  rik.prof = {
      "calls_with_seeds": 0, "calls_without_seeds": 0, "wins": {},
      "seeded_solves": 0, "seeded_time": 0.0,
      "unseeded_solves": 0, "unseeded_time": 0.0,
      "restart_solves": 0, "restart_time": 0.0,
      "getpath_calls": 0, "post_calls": 0, "post_time": 0.0,
  }
  rik._pen_ok = lambda _C, _q: True
  rik._clearance_ok = lambda _C, _q: True

  # ordinary fail; three shallow probes (middle is best); promoted continuation succeeds.
  results = iter([
      {"feasible": False, "eq": 9.0, "ineq": 1.0, "sos": 0.0, "evals": 100},
      {"feasible": False, "eq": 3.0, "ineq": 0.2, "sos": 0.0, "evals": 24},
      {"feasible": False, "eq": 1.0, "ineq": 0.1, "sos": 0.0, "evals": 24},
      {"feasible": False, "eq": 2.0, "ineq": 0.3, "sos": 0.0, "evals": 24},
      {"feasible": True, "eq": 0.0, "ineq": 0.0, "sos": 0.0, "evals": 40},
  ])

  class _Result:
    def __init__(self, value):
      self.value = value

    def dict(self):
      return self.value

  class _NLP:
    def solve(self):
      return _Result(next(results))

  class _Komo:
    def __init__(self):
      self.q = np.zeros(1)
      self.random_id = 0
      self.continued = []

    def getPath(self):
      return self.q.reshape(1, -1)

    def initRandom(self):
      self.random_id += 1
      self.q = np.asarray([float(self.random_id)])

    def initWithConstant(self, q):
      self.q = np.asarray(q, dtype=float).copy()
      self.continued.append(float(self.q[0]))

  komo = _Komo()
  rik._nlp = lambda _komo, stop_evals=None: _NLP()
  ok, q = rik._solve_ik(komo, restarts=1, C=object(), race_random=True)
  assert ok
  assert np.allclose(q, [2.0])
  assert komo.continued == [2.0]
  assert rik.prof["handover_race_stats"]["probes"] == 3
  assert rik.prof["handover_race_stats"]["promoted"] == 1
  assert rik.prof["handover_race_stats"]["successes"] == 1
