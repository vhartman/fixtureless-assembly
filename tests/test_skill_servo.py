"""Focused tests for the execution-time grasp-wrench guard.

These deliberately avoid MuJoCo: the guard must compare the exact measured weld wrench
against the exact breakaway caps before any Jacobian or plant dynamics become relevant.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sim as S


class _GuardScene:
  _rai_joints = ["a1_j0"]
  _vadr = {"a1_j0": 0}

  def __init__(self, force, torque, contact=(8.0, 0.0, 0.0)):
    # (equality id, part, force cap, torque cap): the same tuple SimScene's break monitor uses.
    self.active_grasps = {"a1": (7, "obj", 25.0, 0.4)}
    self.force = float(force)
    self.torque = float(torque)
    self.contact = np.asarray(contact, dtype=float)

  def net_contact_force(self, _part):
    return self.contact

  def _weld_wrench(self, _eq):
    return self.force, self.torque


@pytest.mark.parametrize("force,torque", [
    (15.01, 0.0),      # 60% of the 25N force cap
    (0.0, 0.241),      # 60% of the 0.4Nm torque cap
])
def test_grasp_wrench_guard_holds_on_either_normalized_cap(force, torque):
  servo = S._SkillServo(_GuardScene(force, torque), 0.005, force=6.0,
                        torque_guard=0.6)
  servo.update()

  assert servo.pressing(), "wrench guard must become a hard schedule hold"
  assert servo.wrench_holds == 1
  assert servo._hot_by["a1"] > 4.0 * servo.f0


def test_grasp_wrench_guard_stays_clear_below_both_caps():
  servo = S._SkillServo(_GuardScene(14.9, 0.23, contact=(1.0, 0.0, 0.0)), 0.005, force=6.0,
                        torque_guard=0.6)
  servo.update()

  assert not servo.pressing()
  assert servo.wrench_holds == 0


def test_gravity_only_wrench_does_not_deadlock_transport():
  servo = S._SkillServo(_GuardScene(20.0, 0.3, contact=(0.0, 0.0, 0.0)), 0.005,
                        force=6.0, torque_guard=0.6)
  servo.update()

  assert not servo.pressing()
  assert servo.wrench_holds == 0
