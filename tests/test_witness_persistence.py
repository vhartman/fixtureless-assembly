"""Witness persistence: the plan JSON's optional `witnesses` block round-trips exactly and
is REFUSED whenever it cannot be valid for the consumer's scene/plan (joint order, length).
The expensive end-to-end reuse (solve -> save -> load-plan/sim reuse) is exercised by the
smoke pipeline, not here."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plan_io import read_plan, read_witnesses, witness_block, write_plan
import mp_export as MP


JOINTS = ["a1_j0", "a1_j1", "a2_j0", "a2_j1"]
PLAN = [("pick", "a1", "obj_1", 0), ("assemble", "a1", "obj_1", "obj_0"),
        ("release", "a1", "obj_1")]


def _write(tmp_path, witnesses):
  p = tmp_path / "plan.json"
  write_plan(str(p), {"assembly": "cross"}, PLAN, witnesses=witnesses)
  return str(p)


def test_round_trip_exact(tmp_path):
  wit = [np.array([0.1, -0.2, 0.3, 4.0]), None, np.array([0.0, 1.5, -2.5, 0.25])]
  p = _write(tmp_path, witness_block(JOINTS, wit))
  cfg, plan = read_plan(p)
  assert plan == PLAN and cfg["assembly"] == "cross"     # plan/config unaffected by the block
  out = read_witnesses(p, JOINTS, len(PLAN))
  assert out is not None and len(out) == 3
  assert out[1] is None
  np.testing.assert_array_equal(out[0], wit[0])
  np.testing.assert_array_equal(out[2], wit[2])


def test_file_without_block_reads_as_none(tmp_path):
  p = _write(tmp_path, None)
  assert read_witnesses(p, JOINTS, len(PLAN)) is None


def test_joint_order_mismatch_is_refused(tmp_path, capsys):
  wit = [np.zeros(4), None, np.zeros(4)]
  p = _write(tmp_path, witness_block(JOINTS, wit))
  assert read_witnesses(p, list(reversed(JOINTS)), len(PLAN)) is None
  assert "joint order differs" in capsys.readouterr().out


def test_plan_length_mismatch_is_refused(tmp_path, capsys):
  wit = [np.zeros(4), None, np.zeros(4)]
  p = _write(tmp_path, witness_block(JOINTS, wit))
  assert read_witnesses(p, JOINTS, 5) is None
  assert "witness entries vs 5 plan actions" in capsys.readouterr().out


def test_final_motion_export_round_trips_post_fusion_and_pre_keyframes(tmp_path):
  tasks = [SimpleNamespace(name="pick", type="pick", robots=["a1"],
                           frames=["tool", "obj"], goal=SimpleNamespace(goal=np.zeros(2)),
                           side_effect=None, side_effect_data=None)]
  problem = SimpleNamespace(tasks=tasks, step_names=[["pick"], [], []],
                            sequence_names=["pick"])
  wit = [np.full(4, 1.0), np.full(4, 2.0), None]
  pre = {0: np.full(4, 0.5), 1: None}
  sets = {0: [np.full(4, 1.0), np.full(4, 1.5)]}
  bundle = MP.make_export_bundle(
      problem, PLAN, JOINTS, wit, pre, witness_sets=sets,
      settings={"pre_insertion": 0.05, "collision_tolerance": 1e-4})
  path = tmp_path / "export.json"
  MP.save_export_bundle(str(path), bundle)
  out = MP.load_export_bundle(str(path), PLAN, JOINTS)

  np.testing.assert_array_equal(out["witnesses"][1], wit[1])
  np.testing.assert_array_equal(out["pre_witnesses"][0], pre[0])
  assert out["pre_witnesses"][1] is None
  np.testing.assert_array_equal(out["witness_sets"][0][1], sets[0][1])
  assert out["settings"]["pre_insertion"] == 0.05
  assert out["task_signature"] == MP.task_signature(problem)
  restored = SimpleNamespace(tasks=[MP._task_from_record(r) for r in out["tasks"]])
  assert MP.task_signature(restored) == MP.task_signature(problem)


def test_final_motion_export_load_refuses_plan_or_joint_drift(tmp_path):
  tasks = [SimpleNamespace(name="terminal", type="goto", robots=["a1"], frames=None,
                           goal=SimpleNamespace(goal=np.zeros(2)), side_effect=None,
                           side_effect_data=None)]
  problem = SimpleNamespace(tasks=tasks, step_names=[[], [], []],
                            sequence_names=["terminal"])
  bundle = MP.make_export_bundle(problem, PLAN, JOINTS, [None] * len(PLAN), {})
  path = tmp_path / "export.json"
  MP.save_export_bundle(str(path), bundle)
  with pytest.raises(ValueError, match="different discrete plan"):
    MP.load_export_bundle(str(path), PLAN[:-1], JOINTS)
  with pytest.raises(ValueError, match="joint order differs"):
    MP.load_export_bundle(str(path), PLAN, list(reversed(JOINTS)))


def test_legacy_path_without_final_export_is_not_silently_rebuilt(tmp_path):
  import json
  path = tmp_path / "legacy-mp.json"
  path.write_text(json.dumps({"states": [], "task_signature": "abc"}))
  with pytest.raises(ValueError, match="refusing to rebuild witnesses"):
    MP.load_export_bundle(str(path), PLAN, JOINTS)


def test_saved_motion_path_embeds_the_final_export(tmp_path):
  task = SimpleNamespace(name="terminal", type="goto", robots=["a1"], frames=None,
                         goal=SimpleNamespace(goal=np.zeros(2)), side_effect=None,
                         side_effect_data=None)
  problem = SimpleNamespace(tasks=[task], step_names=[[], [], []],
                            sequence_names=["terminal"])
  bundle = MP.make_export_bundle(problem, PLAN, JOINTS, [None] * len(PLAN), {})

  class _State:
    def to_dict(self):
      return {"q": [0.0] * 4, "mode": [0]}

  path = tmp_path / "mp.json"
  MP.save_path(str(path), [_State()], problem=problem, mp_export_bundle=bundle)
  out = MP.load_export_bundle(str(path), PLAN, JOINTS)
  assert out["task_signature"] == MP.task_signature(problem)
  assert out["sequence_names"] == ["terminal"]
