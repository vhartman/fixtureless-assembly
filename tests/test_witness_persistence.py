"""Witness persistence: the plan JSON's optional `witnesses` block round-trips exactly and
is REFUSED whenever it cannot be valid for the consumer's scene/plan (joint order, length).
The expensive end-to-end reuse (solve -> save -> load-plan/sim reuse) is exercised by the
smoke pipeline, not here."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plan_io import read_plan, read_witnesses, witness_block, write_plan


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
