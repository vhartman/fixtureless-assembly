import json

import pytest

from benchmarks import search_runner as B


def _manifest():
  return {
      "schema_version": 1,
      "name": "unit",
      "timeout_s": 12,
      "seeds": [3, 5],
      "defaults": ["--weight", "inf"],
      "cases": [
          {"name": "a", "args": ["--assembly", "cross", "--parts", "3"]},
          {"name": "b", "args": ["--assembly", "cube", "--parts", "4"], "seeds": [7]},
      ],
      "variants": [
          {"name": "legacy", "args": ["--no-open-witness"]},
          {"name": "open", "args": ["--open-witness"]},
      ],
  }


def _matrix_manifest():
  return {
      "schema_version": 1,
      "name": "matrix-unit",
      "timeout_s": 12,
      "max_address_space_gb": 6,
      "seeds": [3],
      "defaults": ["--weight", "1"],
      "cases": [{"name": "cross", "args": ["--assembly", "cross", "--parts", "3"]}],
      "matrix": {
          "rows": [
              {"name": "counting", "args": ["--no-fill-local-h"]},
              {"name": "reverse", "args": ["--fill-local-h"]},
          ],
          "columns": [
              {"name": "geometry", "args": ["--no-wrench"]},
              {"name": "noisy", "args": ["--wrench", "--insertion-noise", "0.05"]},
          ],
      },
  }


def test_manifest_expansion_is_deterministic_and_case_seed_can_override():
  m = B.validate_manifest(_manifest())
  runs = B.expand_runs(m)
  assert [r["run_id"] for r in runs] == [
      "a__legacy__seed-3", "a__legacy__seed-5",
      "a__open__seed-3", "a__open__seed-5",
      "b__legacy__seed-7", "b__open__seed-7",
  ]
  assert runs[0]["args"] == ["--weight", "inf", "--assembly", "cross", "--parts", "3",
                              "--no-open-witness"]


def test_matrix_expansion_is_row_by_column_and_keeps_coordinates_and_guard():
  runs = B.expand_runs(B.validate_manifest(_matrix_manifest()))
  assert [r["run_id"] for r in runs] == [
      "cross__counting__geometry__seed-3",
      "cross__counting__noisy__seed-3",
      "cross__reverse__geometry__seed-3",
      "cross__reverse__noisy__seed-3",
  ]
  assert runs[-1]["method"] == "reverse" and runs[-1]["physics"] == "noisy"
  assert runs[-1]["max_address_space_gb"] == 6.0
  assert runs[-1]["args"] == [
      "--weight", "1", "--assembly", "cross", "--parts", "3",
      "--fill-local-h", "--wrench", "--insertion-noise", "0.05",
  ]
  guarded = B._guarded_command(runs[-1], ["planner", "--flag"])
  assert guarded == ["/usr/bin/prlimit", "--as=6442450944", "--",
                     "planner", "--flag"]


def test_manifest_rejects_ambiguous_variant_and_matrix_products():
  m = _matrix_manifest()
  m["variants"] = [{"name": "also", "args": []}]
  with pytest.raises(ValueError, match="either variants or matrix"):
    B.validate_manifest(m)


@pytest.mark.parametrize("name", ["search_smoke.json", "fixture_scaling.json",
                                   "fixture_abstract_scaling.json", "core_assemblies.json",
                                   "local_fill_reviewer.json",
                                   "fixture_support_diagnostic.json",
                                   "paper_search_matrix.json",
                                   "paper_search_stress_4r.json",
                                   "search_finalists_4r.json",
                                   "search_manipulation_scope_ablation.json"])
def test_repository_manifests_validate_and_expand(name):
  path = B.ROOT / "benchmarks" / "manifests" / name
  data = B.validate_manifest(json.loads(path.read_text()))
  assert B.expand_runs(data)


def test_runner_rejects_flags_that_would_escape_or_pollute_search_measurement():
  m = _manifest()
  m["defaults"] += ["--save", "somewhere.json"]
  with pytest.raises(ValueError, match="reserved"):
    B.validate_manifest(m)


def test_log_parser_extracts_backward_and_ik_work():
  log = """
[sig-h] 45509 coreachable states, 812 pick facts, built in 12.50s
[fill-prof obj_1:ok-open] binds=6 failed=1 bind_time=2.2s
[fill-skeleton obj_1] VALIDATED 4-action direct schema after 3 candidate path(s), 2 failed IK edge(s), 1.25s
[fill-skeleton obj_2] assignment probe budget 12 reached, 9 failed IK edge(s), 2.50s -> complete local fallback
[milestones] fill-in for obj_2 failed (1.2s) -> blacklisted (1 repairs)
[milestones] complete: 19 actions in 49.12s (2 repairs)
[strict-collisions] vetoed 4 dirty KOMO witnesses (> 0.0001m total penetration).
[ik-clearance] vetoed 3 IK candidate(s) below the shared 5.0mm search/witness margin.
[ik-prof] binds: 432 called, 7 cache hits (2%), 425 solved (158 ok / 267 failed), bind wall 272.2s
[ik-prof] NLP solves: 5655 in 244.1s (seeded 869/31.9s, unseeded 1587/77.3s, restarts 3199/134.9s)
[ik-prof] reach precheck: 402 relaxed solves in 96.5s (179 rejects caught); post-solve gates 513x in 0.6s; getPath calls 513
[rss] peak RSS after solve: 1712 MB
"""
  got = B.parse_log(log)
  assert got["backward_states"] == 45509
  assert got["backward_build_s"] == 12.5
  assert got["milestone_search_s"] == 49.12
  assert got["milestone_repairs"] == 2
  assert got["ik_bind_calls"] == 432 and got["ik_bind_fail"] == 267
  assert got["ik_nlp_solves"] == 5655 and got["ik_restart_solves"] == 3199
  assert got["ik_precheck_rejects"] == 179
  assert got["strict_collision_rejects"] == 4
  assert got["clearance_rejects"] == 3
  assert got["peak_rss_reported_mb"] == 1712
  assert got["skeleton_successful_fills"] == 1
  assert got["skeleton_fallback_fills"] == 1
  assert got["skeleton_candidates"] == 3
  assert got["skeleton_failed_ik_edges"] == 11
  assert got["skeleton_probe_s"] == 3.75


def test_plan_parser_reports_cost_kinds_and_witness_coverage(tmp_path):
  path = tmp_path / "plan.json"
  path.write_text(json.dumps({
      "plan": [["pick", "a1", "obj_1", 0],
               ["handover", "a1", "a2", "obj_1", 0],
               ["release", "a1", "obj_1"]],
      "witnesses": {"stage": "search", "wit": [[0.0], [1.0], None]},
  }))
  got = B.parse_plan(path)
  assert got["plan_actions"] == 3
  assert got["plan_cost"] == 4
  assert got["plan_action_kinds"] == {"pick": 1, "handover": 1, "release": 1}
  assert got["witness_count"] == 2 and got["witness_stage"] == "search"


def test_aggregate_keeps_failures_in_success_rate_but_not_numeric_means():
  rows = [
      {"case": "a", "variant": "x", "success": True, "timed_out": False,
       "wall_s": 2.0, "plan_cost": 10, "skeleton_successful_fills": 3,
       "skeleton_fallback_fills": 1, "skeleton_probe_s": 0.5},
      {"case": "a", "variant": "x", "success": False, "timed_out": True,
       "wall_s": 4.0},
  ]
  got = B.aggregate(rows)[0]
  assert got["runs"] == 2 and got["successes"] == 1 and got["success_rate"] == 0.5
  assert got["wall_s_mean"] == 3.0
  assert got["plan_cost_mean"] == 10.0
  assert got["skeleton_successful_fills_mean"] == 3.0
  assert got["skeleton_fallback_fills_mean"] == 1.0
  assert got["skeleton_probe_s_mean"] == 0.5


def test_matrix_markdown_reports_method_by_physics_cells():
  manifest = _matrix_manifest()
  results = [
      {"case": "cross", "variant": "counting__geometry", "method": "counting",
       "physics": "geometry", "success": True, "timed_out": False,
       "wall_s": 2.0, "max_rss_kb": 102400, "plan_cost": 7,
       "ik_bind_calls": 10, "ik_bind_fail": 2},
      {"case": "cross", "variant": "counting__noisy", "method": "counting",
       "physics": "noisy", "success": False, "timed_out": True, "wall_s": 12.0},
  ]
  groups = B.aggregate_matrix(results)
  text = B.matrix_markdown(manifest, groups)
  assert "| method | geometry | noisy |" in text
  assert "| counting | 1/1 · 2.0+/-0.0s · 100MB · cost 7.0 · IK 10/2 | " in text
  assert "0/1 · 12.0+/-0.0s" in text
  assert "| reverse | -- | -- |" in text


def test_matrix_suite_writes_markdown_csv_and_json(tmp_path):
  manifest = _matrix_manifest()
  results = [{
      "case": "cross", "variant": "counting__geometry", "method": "counting",
      "physics": "geometry", "success": True, "timed_out": False,
      "wall_s": 2.0, "max_rss_kb": 102400, "plan_cost": 7,
  }]
  B.write_suite_outputs(tmp_path, manifest, {}, results)
  assert (tmp_path / "matrix.md").is_file()
  assert (tmp_path / "matrix.csv").is_file()
  data = json.loads((tmp_path / "matrix.json").read_text())
  assert data["groups"][0]["method"] == "counting"
  assert data["groups"][0]["physics"] == "geometry"
