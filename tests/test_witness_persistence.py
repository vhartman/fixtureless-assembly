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

from plan_io import missing_required_witnesses, read_plan, read_pre_witnesses, \
    read_skill_paths, read_witnesses, read_witness_stage, require_complete_witnesses, \
    witness_block, write_plan
import mp_export as MP
import solver


JOINTS = ["a1_j0", "a1_j1", "a2_j0", "a2_j1"]
PLAN = [("pick", "a1", "obj_1", 0), ("assemble", "a1", "obj_1", "obj_0"),
        ("release", "a1", "obj_1")]


def _write(tmp_path, witnesses):
  p = tmp_path / "plan.json"
  write_plan(str(p), {"assembly": "cross"}, PLAN, witnesses=witnesses)
  return str(p)


def test_write_plan_creates_requested_parent_directory(tmp_path):
  p = tmp_path / "new-run" / "nested" / "plan.json"
  write_plan(str(p), {"assembly": "cross"}, PLAN)
  assert p.is_file()
  assert read_plan(str(p)) == ({"assembly": "cross"}, PLAN)


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


def test_witness_stage_distinguishes_early_search_save_from_completed_optimizer(tmp_path):
  wit = [np.zeros(4), np.ones(4), None]
  p = _write(tmp_path, witness_block(JOINTS, wit, stage="search"))
  assert read_witness_stage(p) == "search"
  p = _write(tmp_path, witness_block(JOINTS, wit, stage="optimized"))
  assert read_witness_stage(p) == "optimized"


def test_path_valued_skill_certificate_round_trips_and_checks_endpoint(tmp_path):
  wit = [np.zeros(4), np.ones(4), None]
  skill = {1: [np.zeros(4), np.full(4, 0.5), np.ones(4)]}
  p = _write(tmp_path, witness_block(JOINTS, wit, stage="optimized", skill_paths=skill))
  out = read_skill_paths(p, JOINTS, len(PLAN), wit)
  assert list(out) == [1]
  np.testing.assert_array_equal(out[1][-1], wit[1])

  bad = {1: [np.zeros(4), np.full(4, 2.0)]}
  p = _write(tmp_path, witness_block(JOINTS, wit, skill_paths=bad))
  with pytest.raises(ValueError, match="does not end"):
    read_skill_paths(p, JOINTS, len(PLAN), wit)


def test_pre_insertion_certificate_round_trips_and_fails_closed(tmp_path):
  wit = [np.zeros(4), np.ones(4), None]
  pre = {1: np.array([0.9, 0.8, 0.7, 0.6])}
  p = _write(tmp_path, witness_block(JOINTS, wit, pre_witnesses=pre))
  out = read_pre_witnesses(p, JOINTS, len(PLAN), wit)
  assert list(out) == [1]
  np.testing.assert_array_equal(out[1], pre[1])

  bad = {1: np.ones(3)}
  p = _write(tmp_path, witness_block(JOINTS, wit, pre_witnesses=bad))
  with pytest.raises(ValueError, match="wrong-length pre-witness"):
    read_pre_witnesses(p, JOINTS, len(PLAN), wit)


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


def test_required_ik_witnesses_fail_closed_but_release_null_is_valid():
  wit = [np.zeros(4), None, None]
  assert missing_required_witnesses(PLAN, wit) == [1]
  with pytest.raises(ValueError, match=r"1:\('assemble'"):
    require_complete_witnesses(PLAN, wit, who="test artifact")
  wit[1] = np.ones(4)
  assert require_complete_witnesses(PLAN, wit) is wit


def test_rewritten_plan_reuses_only_same_action_and_scene_certificate(monkeypatch):
  """Release reordering must never treat an old witness index as a certificate.

  The same action can appear at a different predecessor scene after a release is hoisted.  The
  persistence helper may reuse the source witness only when RobotIK's ordinary cache key agrees;
  otherwise it must issue a fresh bind.
  """
  source = [("pick", "a1", "obj_0", 0)]
  candidate = [("pick", "a1", "obj_0", 0)]

  class _State:
    def __init__(self, sig):
      self.sig = sig

  source_states = [_State("source"), _State("after")]
  candidate_states = [_State("source"), _State("after")]

  def replay(_C, _assembly, plan, *_args, **_kwargs):
    return source_states if plan is source else candidate_states

  monkeypatch.setattr(solver, "_replay_states", replay)

  class _IK:
    _ik_cache = {}

    @staticmethod
    def _scene_sig(state):
      return state.sig

    def __init__(self):
      self.bind_calls = 0

    def bind(self, *_args):
      self.bind_calls += 1
      return True, np.array([9.0])

  ik = _IK()
  source_witness = np.array([1.0])
  out_same = solver.collect_plan_witnesses(
      None, None, candidate, ["a1"], ik, grasp_table={}, resolve_missing=True,
      verbose=False, reuse_from_plan=source, reuse_from_witnesses=[source_witness])

  # An unchanged predecessor scene is genuinely the same certificate even though this is a
  # different plan object/indexing context.
  assert ik.bind_calls == 0
  np.testing.assert_array_equal(out_same[0], source_witness)

  candidate_states[0] = _State("candidate")
  out = solver.collect_plan_witnesses(
      None, None, candidate, ["a1"], ik, grasp_table={}, resolve_missing=True,
      verbose=False, reuse_from_plan=source, reuse_from_witnesses=[source_witness])

  # The action text is identical, but the predecessor signature is not: the old vector must
  # not leak positionally into the candidate.  A fresh exact bind supplies the new certificate.
  assert ik.bind_calls == 1
  np.testing.assert_array_equal(out[0], [9.0])


def test_release_hoist_rolls_back_to_source_witnesses_on_failed_recertification(
    monkeypatch, capsys):
  source = [("pick", "a1", "obj_0", 0),
            ("assemble", "a1", "obj_0", "obj_1"),
            ("release", "a1", "obj_0")]
  candidate = [("pick", "a1", "obj_0", 0),
               ("release", "a1", "obj_0"),
               ("assemble", "a1", "obj_0", "obj_1")]
  source_witnesses = [np.array([1.0]), np.array([2.0]), None]
  monkeypatch.setattr(solver, "hoist_releases", lambda *_args, **_kwargs: candidate)

  seen = {}

  def recertify(*args, **kwargs):
    seen.update(kwargs)
    assert list(args[2]) == candidate
    # The moved assembly edge has no exact certificate.  This is the normal IK failure mode
    # that used to bubble out of require_complete_witnesses and discard the whole plan.
    return [np.array([1.0]), None, None]

  monkeypatch.setattr(solver, "collect_plan_witnesses", recertify)
  final_plan, final_witnesses, applied = solver.hoist_releases_certified(
      None, None, source, source_witnesses, ["a1"], object(), verbose=True)

  assert not applied
  assert list(final_plan) == source
  np.testing.assert_array_equal(final_witnesses[0], source_witnesses[0])
  np.testing.assert_array_equal(final_witnesses[1], source_witnesses[1])
  assert final_witnesses[2] is None
  assert seen["resolve_missing"]
  assert seen["reuse_from_plan"] is source
  assert "rollback" in capsys.readouterr().out


def test_release_hoist_commits_only_after_complete_recertification(monkeypatch):
  source = [("pick", "a1", "obj_0", 0),
            ("assemble", "a1", "obj_0", "obj_1"),
            ("release", "a1", "obj_0")]
  candidate = [("pick", "a1", "obj_0", 0),
               ("release", "a1", "obj_0"),
               ("assemble", "a1", "obj_0", "obj_1")]
  source_witnesses = [np.array([1.0]), np.array([2.0]), None]
  candidate_witnesses = [np.array([1.0]), None, np.array([3.0])]
  monkeypatch.setattr(solver, "hoist_releases", lambda *_args, **_kwargs: candidate)
  monkeypatch.setattr(solver, "collect_plan_witnesses",
                      lambda *_args, **_kwargs: candidate_witnesses)

  final_plan, final_witnesses, applied = solver.hoist_releases_certified(
      None, None, source, source_witnesses, ["a1"], object(), verbose=False)

  assert applied and list(final_plan) == candidate
  np.testing.assert_array_equal(final_witnesses[2], [3.0])


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
