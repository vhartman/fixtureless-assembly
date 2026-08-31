"""Canonical physical-query IK memoization."""

from types import MethodType

import numpy as np

from solver import RobotIK


class _State:
  def __init__(self, *, unrelated_grasp=0, giver_grasp=2, marker=None):
    self.marker = marker
    self.parent = {
        "obj_0": ("gripper", "a1"),
        "obj_1": ("part", "obj_0"),
        "obj_2": ("gripper", "a3"),
    }
    self.grasp = {"obj_0": giver_grasp, "obj_2": unrelated_grasp}
    self.place = {}
    self.poses = {}
    self.supports = {
        "obj_0": frozenset({("grasp", "a1", "obj_0", giver_grasp)}),
        "obj_1": frozenset(),
        "obj_2": frozenset({("grasp", "a3", "obj_2", unrelated_grasp)}),
    }

  def subassemblies(self):
    return [frozenset({"obj_0", "obj_1"}), frozenset({"obj_2"})]

  def bodies(self):
    return list(self.parent)


def _bare_rik():
  rik = RobotIK.__new__(RobotIK)
  rik.robots = ["a1", "a2", "a3", "a4"]
  rik.positive_ik_reuse = True
  rik._physical_query_legacy = {}
  rik._ik_cache = {}
  rik._opt_pull = None
  rik._ctx_seeds = None
  rik._last_win = ""
  rik.last_strict_reject = None
  rik.last_margin_reject = None
  rik.last_closure_reject = None
  rik.ana_stats = {}
  rik.prof = {
      "bind_calls": 0, "bind_cache_hits": 0, "bind_solves": 0,
      "bind_success": 0, "bind_fail": 0, "bind_time": 0.0,
      "bind_success_time": 0.0, "bind_fail_time": 0.0,
      "physical_query_collapses": 0, "by_action": {},
  }
  return rik


def test_handover_query_is_local_to_its_payload_robots_and_grasps():
  rik = _bare_rik()
  action = ("handover", "a2", "obj_1", 7)
  a = _State(unrelated_grasp=0)
  b = _State(unrelated_grasp=9)
  c = _State(unrelated_grasp=0, giver_grasp=3)

  # An unrelated held object is not part of this handover program. The giver grasp is.
  assert rik._physical_query_sig(action, a, a, object()) == \
      rik._physical_query_sig(action, b, b, object())
  assert rik._physical_query_sig(action, a, a, object()) != \
      rik._physical_query_sig(action, c, c, object())


def test_query_key_includes_successor_target_and_counts_only_symbolic_aliases():
  rik = _bare_rik()
  rik._scene_sig = MethodType(lambda self, s: ("legacy", id(s)), rik)
  rik._physical_query_sig = MethodType(
      lambda self, _act, before, after, _a: (before.marker, after.marker), rik)
  action = ("place", "a1", "obj_0", 0)

  a0, a1 = _State(marker="pre"), _State(marker="landing-a")
  b0, b1 = _State(marker="pre"), _State(marker="landing-b")
  assert rik._query_key(action, a0, a1, object()) != rik._query_key(action, b0, b1, object())

  c0, c1 = _State(marker="pre"), _State(marker="landing-a")
  assert rik._query_key(action, a0, a1, object()) == rik._query_key(
      action, c0, c1, object())
  assert rik.prof["physical_query_collapses"] == 1


def test_identical_physical_query_directly_reuses_witness_without_recheck_or_ik():
  rik = _bare_rik()
  action = ("handover", "a2", "obj_1", 7)
  q = np.array([1.0, 2.0, 3.0, 4.0])
  rik._query_key = MethodType(lambda self, *_args: ("same-physical-query",), rik)
  calls = [0]

  def solve(self, *_args, **_kwargs):
    calls[0] += 1
    return True, q.copy()

  rik._ik_add_grasp = MethodType(solve, rik)
  first = rik.bind(action, _State(), _State(), object())
  second = rik.bind(action, _State(unrelated_grasp=9), _State(), object())

  assert calls[0] == 1
  assert first[0] and second[0]
  np.testing.assert_array_equal(second[1], q)
  assert rik.prof["bind_cache_hits"] == 1


def test_failed_bind_time_is_accounted_separately():
  rik = _bare_rik()
  rik._query_key = MethodType(lambda self, *_args: ("q",), rik)
  rik._ik_add_grasp = MethodType(lambda self, *_args, **_kwargs: (False, None), rik)
  action = ("handover", "a2", "obj_1", 7)

  assert rik.bind(action, _State(), _State(), object()) == (False, None)
  assert rik.prof["bind_fail_time"] > 0.0
  assert rik.prof["bind_success_time"] == 0.0
