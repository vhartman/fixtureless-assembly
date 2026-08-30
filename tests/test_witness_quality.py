import json
import numpy as np

from benchmarks.witness_quality import analyze, markdown


def test_witness_quality_counts_bounded_motion_hidden_by_modulo(tmp_path):
  p = tmp_path / "plan.json"
  p.write_text(json.dumps({
      "config": {"witness_branch_metric": "motion", "witness_branch_budget": 0},
      "plan": [["pick", "a1", "obj", 0], ["release", "a1", "obj"],
               ["place", "a1", "obj", 0]],
      "witnesses": {"stage": "optimized", "joints": ["a1_j1", "a1_j2"],
                    "wit": [[-3.47, 0.0], None, [0.54, 0.25]]},
  }))
  row = analyze(p, threshold=3.0)
  assert row["witness_edges"] == 1
  assert row["large_coordinate_transitions"] == 1
  assert row["modulo_hidden_transitions"] == 1
  assert np.isclose(row["chain_l2_sum_rad"], np.hypot(4.01, 0.25))
  assert "motion / 0" in markdown([row])
