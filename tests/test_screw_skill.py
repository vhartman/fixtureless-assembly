"""Single-grasp screw mate: data contract, helical geometry, wrench, persistence."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assembly_planner.mp_export as MP
import assembly_planner.problem as P
import sim
import assembly_planner.solver as S
import assembly_planner.wrench as W


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


def test_four_leg_screw_stool_has_four_atomic_single_grasp_mates():
  asm = P.make_assembly("peg-stool-4-screw", 5)
  assert asm.objects == ["obj_0", "obj_1", "obj_2", "obj_3", "obj_4"]
  assert asm.masses["obj_0"] == pytest.approx(0.50)
  assert sorted(abs(asm.relative_pose(leg)[1][1]) for leg in asm.objects[1:]) == \
      pytest.approx([0.0425] * 4)
  for leg in asm.objects[1:]:
    assert asm.relative_pose(leg)[0] == "obj_0"
    assert asm.dependencies_of(leg) == ["obj_0"]
    assert asm.screw_skill(leg) == P.ScrewSkill(
        np.pi, 0.020, 0.05, 0.75, 16, single_grasp=True)
    children = P._compound_children(asm.part_shapes[leg])
    shaft = next((shape for shape, _ in children if shape[0] == "cylinder"), None)
    assert shaft is not None and 2.0 * shaft[1][1] < asm.socket_hole
    assert not any(shape[0] == "box" for shape, _ in children)
    marker = asm.visual_markers[leg][0]
    # The eccentric marker stays at the exposed foot, outside the top screw-depth band.
    assert marker["pos"][2] + marker["shape"][1][2] / 2.0 < \
        asm.leg_length / 2.0 - 0.020
    assert asm.masses[leg] == pytest.approx(0.08)
    leg_top = asm.relative_pose(leg)[1][2] + asm.leg_length / 2.0
    assert leg_top == pytest.approx(
        -asm.seat_thickness / 2.0 - (asm.socket_depth - asm.screw_skill(leg).depth))
  assert len(asm.mate_skills) == 4


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


def test_quaternion_mate_error_uses_shortest_rotation():
  identity = np.array([1.0, 0.0, 0.0, 0.0])
  ten_deg = S._axis_angle_quat([0.0, 0.0, 1.0], np.deg2rad(10.0))
  assert S._quat_angle_deg(identity, ten_deg) == pytest.approx(10.0)
  # q and -q encode the same orientation.
  assert S._quat_angle_deg(identity, -identity) == pytest.approx(0.0)


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


def test_sim_process_wrench_matches_planner_force_and_screw_couple():
  asm = P.make_assembly("peg-hole-screw", 2)
  parent_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  part_pose = np.concatenate([
      np.asarray(asm.relative_pose("obj_1")[1], dtype=float),
      np.array([1.0, 0.0, 0.0, 0.0])])
  loads = sim._process_wrench_pair(
      asm, "obj_1", "obj_0", part_pose, parent_pose)
  fp, tp, xp = loads["obj_1"]
  ft, tt, xt = loads["obj_0"]
  axis = np.asarray(asm.mate_axis("obj_1"), dtype=float)
  axis /= np.linalg.norm(axis)
  skill = asm.screw_skill("obj_1")

  np.testing.assert_allclose(ft, W.insert_force("obj_1") * axis)
  np.testing.assert_allclose(fp, -ft)
  np.testing.assert_allclose(tt, -np.sign(skill.angle) * skill.torque * axis)
  np.testing.assert_allclose(tp, -tt)
  np.testing.assert_allclose(xp, xt)
  # The declared pair is internal to the two-part system: zero resultant and zero net moment.
  np.testing.assert_allclose(fp + ft, 0.0, atol=1e-12)
  np.testing.assert_allclose(tp + np.cross(xp, fp) + tt + np.cross(xt, ft),
                             0.0, atol=1e-12)


def test_sim_process_noise_is_fixed_mate_frame_load_not_white_noise():
  asm = P.make_assembly("peg-hole-screw", 2)
  parent_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  part_pose = np.concatenate([
      np.asarray(asm.relative_pose("obj_1")[1], dtype=float),
      np.array([1.0, 0.0, 0.0, 0.0])])
  realization = {"lateral": 0.2, "offset": 0.004,
                 "coefficients": [1.0, -1.0, -1.0, 1.0]}
  loads = sim._process_wrench_pair(
      asm, "obj_1", "obj_0", part_pose, parent_pose, realization)
  fp, tp, xp = loads["obj_1"]
  ft, tt, xt = loads["obj_0"]

  np.testing.assert_allclose(fp, -ft)
  np.testing.assert_allclose(tp, -tt)
  np.testing.assert_allclose(xp, xt)
  # Axial load stays nominal; each requested transverse corner has 0.2 F magnitude.
  axis = np.asarray(asm.mate_axis("obj_1"), dtype=float)
  axis /= np.linalg.norm(axis)
  assert np.dot(ft, axis) == pytest.approx(W.insert_force("obj_1"))
  transverse = ft - np.dot(ft, axis) * axis
  assert np.linalg.norm(transverse) == pytest.approx(
      np.sqrt(2.0) * 0.2 * W.insert_force("obj_1"))
  np.testing.assert_allclose(fp + ft, 0.0, atol=1e-12)
  np.testing.assert_allclose(tp + np.cross(xp, fp) + tt + np.cross(xt, ft),
                             0.0, atol=1e-12)
