"""Replay must be a reader: opening a saved plan may not resolve IK implicitly."""
import numpy as np
import pytest

import replay as R


class _C:
  def getJointNames(self):
    return ["a1_j0", "a1_j1"]


class _IK:
  def __init__(self, C, robots):
    self.C = C
    self.pre_insertion_delta = self.pre_pick_delta = 0.0
    self.pre_place_delta = self.pre_add_grasp_delta = 0.0
    self.strip_inactive_collision = True
    self.penetration_reject = None
    self.col_weight = 0.0


def _patch_scene(monkeypatch, witnesses):
  cfg = {"assembly": "cross", "parts": 1, "num_robots": 1, "real_ik": True,
         "bracing": False, "pre_insertion": 0.0}
  plan = [("pick", "a1", "obj_0", 0)]
  monkeypatch.setattr(R, "read_plan", lambda _p: (cfg, plan))
  monkeypatch.setattr(R, "build_scene", lambda _cfg: (_C(), object()))
  monkeypatch.setattr(R, "get_robots", lambda _C: ["a1"])
  monkeypatch.setattr(R, "RobotIK", _IK)
  monkeypatch.setattr(R, "read_witnesses", lambda *_a: witnesses)
  return plan


def test_replay_passes_persisted_witnesses_to_visualizer(monkeypatch):
  witnesses = [np.array([0.1, 0.2])]
  _patch_scene(monkeypatch, witnesses)
  seen = {}
  monkeypatch.setattr(
      R, "visualize_plan_ik",
      lambda *_a, **kw: seen.setdefault("witnesses", kw.get("witnesses")))
  R.replay("plan.json", view=True)
  assert seen["witnesses"] is witnesses


def test_replay_refuses_missing_witness_without_explicit_resolve(monkeypatch):
  _patch_scene(monkeypatch, None)
  monkeypatch.setattr(R, "visualize_plan_ik",
                      lambda *_a, **_kw: pytest.fail("visualizer must not be reached"))
  with pytest.raises(SystemExit, match="refusing to resolve IK implicitly"):
    R.replay("plan.json", view=True)


def test_replay_restores_vacuum_engagement_successor_gate():
  import wrench

  R.apply_wrench_params({"vacuum_engage_force": 7.5})
  assert wrench.PARAMS["vacuum_engage_force"] == 7.5
  R.apply_wrench_params({})
  assert wrench.PARAMS["vacuum_engage_force"] == 0.0
