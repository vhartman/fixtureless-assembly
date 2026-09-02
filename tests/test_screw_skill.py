"""Single-grasp screw mate: data contract, helical geometry, wrench, persistence."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mp_export as MP
import problem as P
import solver as S


def test_peg_hole_screw_is_one_atomic_mate_with_typed_skill():
  asm = P.make_assembly("peg-hole-screw", 2)
  skill = asm.screw_skill("obj_1")
  assert isinstance(skill, P.ScrewSkill)
  assert skill.single_grasp and skill.angle == pytest.approx(np.pi)
  assert skill.depth == pytest.approx(0.020)
  # No task-planning regrasp schema was introduced: it is the same one-edge assembly DAG.
  assert asm.relative_pose("obj_1")[0] == "obj_0"
  assert asm.assembly_dependencies == {}


def test_screw_overrides_are_validated_and_robot_independent():
  asm = P.make_assembly("peg-hole-screw", 2)
  asm.configure_screw_skills(angle=-np.pi / 2, depth=0.012, torque=0.18,
                             angular_speed=0.4, samples=9)
  got = asm.screw_skill("obj_1")
  assert got == P.ScrewSkill(-np.pi / 2, 0.012, 0.18, 0.4, 9)
  with pytest.raises(ValueError):
    asm.configure_screw_skills(samples=1)
  # Ordinary mates remain ordinary under the same generic API.
  assert P.make_assembly("peg-hole", 2).screw_skill("obj_1") is None


def test_screw_entry_is_backed_off_and_counter_rotated():
  asm = P.make_assembly("peg-hole-screw", 2)
  skill = asm.screw_skill("obj_1")
  final = np.asarray(asm.relative_pose("obj_1")[1], dtype=float)
  axis = asm.mate_axis("obj_1")
  entry = S._backoff_mate(final, axis, skill.depth)
  assert np.allclose(entry, final - skill.depth * axis)
  q = S._axis_angle_quat(axis, -skill.angle)
  assert np.linalg.norm(q) == pytest.approx(1.0)
  # Half-turn around -z (with the signed counter-rotation) has zero scalar quaternion part.
  assert abs(q[0]) < 1e-9


def test_group_equilibrium_receives_pure_torque_without_fake_lever(monkeypatch):
  captured = {}
  monkeypatch.setattr(
      P, "_grounding_caps",
      lambda *_args, **_kwargs: (np.zeros(3), np.zeros(6), np.eye(6),
                                 np.full(6, np.inf), [(0.0, None)] * 6,
                                 (b"g", b"u")))

  def solve(_G, _ub, w_ext, **_kwargs):
    captured["w"] = np.asarray(w_ext).copy()
    return True

  monkeypatch.setattr(P.wrench, "equilibrium_feasible_matrices", solve)
  P._WRENCH_CACHE.clear()
  tau = np.array([0.1, -0.2, 0.3])
  assert P._group_stable(object(), object(), {"obj_0"}, None,
                         applied=[(np.zeros(3), np.array([99.0, 99.0, 99.0]), tau)])
  np.testing.assert_allclose(captured["w"][:3], 0.0)
  np.testing.assert_allclose(captured["w"][3:], tau)


def test_task_signature_and_export_persist_screw_certificate():
  skill_data = {"kind": "screw", "angle": np.pi, "depth": 0.02,
                "torque": 0.25, "angular_speed": 0.75, "samples": 2,
                "single_grasp": True, "contract": "single_grasp",
                "part": "obj_1", "parent": "obj_0",
                "joint_path": [[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]]}
  task = SimpleNamespace(name="assemble", type="place", robots=["a1"],
                         frames=["obj_0", "obj_1"],
                         goal=SimpleNamespace(goal=np.array([0.2, 1.2])),
                         side_effect=None, side_effect_data=None,
                         skill_data=skill_data)
  rec = MP._task_record(task)
  restored = MP._task_from_record(rec)
  assert restored.skill_data == skill_data
  p0 = SimpleNamespace(tasks=[task])
  p1 = SimpleNamespace(tasks=[restored])
  assert MP.task_signature(p0) == MP.task_signature(p1)
  restored.skill_data["angle"] = np.pi / 2
  assert MP.task_signature(p0) != MP.task_signature(p1)
