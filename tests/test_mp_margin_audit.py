"""Structured diagnostics for our mr_bench collision-predicate margin wrapper."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mp_export as MP


class _Frame:
  def __init__(self, name, pos=(0.0, 0.0, 0.0), parent=None):
    self.name = name
    self._pos = np.asarray(pos, dtype=float)
    self._parent = parent

  def getPosition(self):
    return self._pos

  def getParent(self):
    return self._parent


class _Config:
  def __init__(self):
    self.frames = {n: _Frame(n) for n in ("a1_link", "a2_link", "a1_ee", "a2_ee")}
    self.collisions = [("a1_link", "a2_link", 0.02)]
    self.q = np.zeros(2)

  def getFrame(self, name):
    return self.frames[name]

  def getJointState(self):
    return self.q.copy()

  def setJointState(self, q):
    self.q = np.asarray(q, dtype=float).copy()

  def getCollisions(self, _margin):
    return list(self.collisions)


class _Task:
  name = "next_transit"
  robots = []


class _Mode:
  task_ids = [7, 9]


class _Env:
  def __init__(self):
    self.C = _Config()
    self.base_ok = True
    self.task = _Task()

  def get_valid_next_task_combinations(self, _mode):
    return [0]

  def get_active_task(self, _mode, _combo):
    return self.task

  def is_collision_free(self, *_args, **_kwargs):
    return self.base_ok

  def is_collision_free_np(self, *_args, **_kwargs):
    return self.base_ok

  def is_edge_collision_free(self, *_args, **_kwargs):
    return self.base_ok


def test_margin_wrapper_records_and_prints_concrete_gate_reason(capsys):
  env = _Env()
  MP._install_robot_margin(
      env, 0.1, ["a1", "a2"], {"a1": "a1_ee", "a2": "a2_ee"}, ramp=0.1)

  assert not env.is_collision_free_np(np.zeros(2), _Mode())
  stats = env.margin_rejection_stats
  assert stats["base_rejects"] == 0
  assert stats["gate_rejects"] == 1
  rec = next(iter(stats["records"].values()))
  assert rec["gate"] == "robot-full-margin"
  assert rec["mode"] == (7, 9)
  assert rec["frames"] == ("a1_link", "a2_link")
  assert rec["observed"] == 0.02
  assert rec["required"] == 0.1

  env.report_margin_rejections([7, 9])
  out = capsys.readouterr().out
  assert "our added margin gates rejected 1/1" in out
  assert "frontier mode [7, 9]" in out
  assert "gate=robot-full-margin" in out
  assert "a1_link~a2_link" in out
  assert "d=20.00mm required=100.00mm" in out


def test_margin_audit_distinguishes_base_rejection_from_our_gate(capsys):
  env = _Env()
  env.base_ok = False
  env.C.collisions = [("a1_link", "a2_link", -0.012)]
  MP._install_robot_margin(
      env, 0.1, ["a1", "a2"], {"a1": "a1_ee", "a2": "a2_ee"}, ramp=0.1)

  assert not env.is_collision_free_np(np.zeros(2), _Mode())
  assert not env.is_edge_collision_free(None, None, _Mode())
  assert env.margin_rejection_stats["base_rejects"] == 1
  assert env.margin_rejection_stats["gate_rejects"] == 0
  env.report_margin_rejections([7, 9])
  out = capsys.readouterr().out
  assert "base predicate rejected 1" in out
  assert "our wrapper vetoed no base-valid sample" in out
  assert "concrete collision pairs from 1 sampled base rejection" in out
  assert "min_penetration=-12.00mm" in out
  assert "failed edges at frontier [7, 9]: 1" in out
