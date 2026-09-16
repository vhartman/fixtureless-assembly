import json
import re
import threading

import numpy as np
import pytest

from benchmarks import plot_cube_scaling as C
from benchmarks import plot_process_wrench_full_matrix as P
from benchmarks import search_runner as B
from benchmarks import run_wrench_sim_ablation as WS
from benchmarks import run_paper_search_comparison as PS
from benchmarks import run_paper_anytime_convergence as AC
from benchmarks import run_process_wrench_full_matrix as PWM
import problem


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


def test_wrench_sim_ablation_does_not_confound_task_model_with_mp_wrench_gate():
  assert all(not variant["mp_wrench_gate"] for variant in WS.VARIANTS.values())


def test_wrench_sim_ablation_flattens_phase_load_metrics():
  got = WS._wrench_metrics({
      "grasp_wrench_metrics": {
          "window_s": 0.02,
          "phases": {
              "transport": {
                  "peak_force_n": 12.0, "peak_torque_nm": 0.4,
                  "peak_utilization": 1.2, "peak_sustained_utilization": 0.8,
              },
              "insertion": {
                  "peak_force_n": 20.0, "peak_torque_nm": 0.9,
                  "peak_utilization": 1.5, "peak_sustained_utilization": 1.1,
              },
          },
          "by_gripper_kind": {
              "two_finger": {
                  "transport": {"peak_sustained_utilization": 0.7},
                  "insertion": {"peak_sustained_utilization": 1.3},
              },
              "vacuum": {
                  "transport": {"peak_sustained_utilization": 0.4},
                  "insertion": {"peak_sustained_utilization": 0.9},
              },
          },
      },
      "table_wrench_metrics": {
          "table_mu": 0.5,
          "phases": {
              "transport": {"peak_utilization": 0.3,
                            "peak_sustained_utilization": 0.2},
              "insertion": {"peak_utilization": 1.6,
                            "peak_sustained_utilization": 1.4},
          },
      },
  })
  assert got["wrench_window_s"] == pytest.approx(0.02)
  assert got["transport_peak_sustained_utilization"] == pytest.approx(0.8)
  assert got["insertion_peak_force_n"] == pytest.approx(20.0)
  assert got["transport_two_finger_peak_sustained_utilization"] == pytest.approx(0.7)
  assert got["insertion_vacuum_peak_sustained_utilization"] == pytest.approx(0.9)
  assert got["table_mu"] == pytest.approx(0.5)
  assert got["insertion_table_peak_utilization"] == pytest.approx(1.6)
  assert got["insertion_table_peak_sustained_utilization"] == pytest.approx(1.4)


def test_wrench_sim_table_formats_phase_and_tool_utilization_as_percent():
  row = {
      "transport_two_finger_peak_sustained_utilization_mean": 0.73,
      "insertion_two_finger_peak_sustained_utilization_mean": 1.24,
      "insertion_table_peak_utilization_mean": 0.81,
  }
  assert WS._fmt_util(row, "transport", "two_finger") == "73"
  assert WS._fmt_util(row, "insertion", "two_finger") == r"$\mathbf{124}$"
  assert WS._fmt_util(row, "transport", "vacuum") == "--"
  failed = dict(row, success_percent=0)
  assert WS._fmt_util(failed, "transport", "two_finger") == "(73)"
  assert WS._fmt_util(failed, "insertion", "two_finger") == \
      r"($\mathbf{124}$)"
  assert WS._fmt_task_progress({"actions_ok_prefix_mean": 17.0,
                                "actions_mean": 22.0}) == "17/22"
  assert WS._fmt_task_progress({"actions_ok_prefix_mean": 2.5,
                                "actions_mean": 4.5}) == "2.5/4.5"
  assert WS._fmt_task_progress({}) == "--"


def test_paper_runner_automatically_measures_first_motion_solution(tmp_path, monkeypatch):
  calls = []
  runner_calls = []
  monkeypatch.setattr(PS.search_runner, "main", lambda args: runner_calls.append(args) or 0)
  monkeypatch.setattr(
      PS, "measure_initial_motion_plans",
      lambda output, **kwargs: calls.append((output, kwargs)))
  monkeypatch.setattr(PS, "write_tables", lambda _output: None)
  output = tmp_path / "paper"
  assert PS.main(["--output", str(output), "--motion-runtime", "12"]) == 0
  assert len(calls) == 1 and calls[0][0] == output.resolve()
  assert calls[0][1]["runtime_s"] == 12.0
  assert not calls[0][1]["resume"] and not calls[0][1]["retry_failed_only"]
  assert "--shared-grasp-db" in runner_calls[0]


def test_benchmark_source_fingerprint_includes_nested_meshes_and_requirements():
  _digest, files = B._source_hash()
  assert "requirements.txt" in files
  assert any(path.startswith("part_meshes/fabrica/") for path in files)


def test_full_scaling_wrapper_forwards_independent_seeds_and_jobs(tmp_path, monkeypatch):
  calls = []
  monkeypatch.setattr(PWM.search_runner, "main", lambda args: calls.append(args) or 0)
  assert PWM.main([
      "--output", str(tmp_path / "scaling"), "--assembly", "cross",
      "--seeds", "3", "5", "7", "--jobs", "2", "--timeout", "500",
      "--dry-run", "--no-plot",
  ]) == 0
  assert len(calls) == 2  # topological control and finite-wrench matrix
  for call in calls:
    assert call[call.index("--seeds") + 1:call.index("--jobs")] == ["3", "5", "7"]
    assert call[call.index("--jobs") + 1] == "2"
    assert call[call.index("--timeout") + 1] == "500.0"


def test_anytime_wrapper_selects_case_subset_and_forwards_jobs(tmp_path, monkeypatch):
  calls = []
  monkeypatch.setattr(AC.search_runner, "main", lambda args: calls.append(args) or 0)
  assert AC.main([
      "--output", str(tmp_path / "anytime"), "--cases", "cross", "chair",
      "--seeds", "11", "13", "--jobs", "2", "--runtime", "150", "--dry-run",
  ]) == 0
  assert len(calls) == 1
  call = calls[0]
  assert call[call.index("--jobs") + 1] == "2"
  assert call[call.index("--timeout") + 1] == "240.0"
  assert call[call.index("--seeds") + 1:call.index("--jobs")] == ["11", "13"]
  selector = call[call.index("--only") + 1]
  assert re.search(selector, "cross__ours__seed-11")
  assert re.search(selector, "chair__ours__seed-13")
  assert not re.search(selector, "stool__ours__seed-11")
  assert "--append-arg=--milestones-anytime" in call
  assert call[call.index("--append-arg") + 1] == "150.0"


def test_grasp_database_roundtrip_preserves_lazy_rows_and_kind_masks(tmp_path):
  rows = {
      ("obj_0",): [("obj_0", 3, np.arange(7, dtype=float))],
      ("obj_0", "obj_1"): [("obj_1", 4, np.zeros(7))],
  }
  table = problem.LazyGraspTable(rows, lambda key: rows[key])
  table.kinds_of = {
      (("obj_0",), "obj_0", 3): frozenset({"two_finger"}),
      (("obj_0", "obj_1"), "obj_1", 4): frozenset({"vacuum"}),
  }
  table.robot_kinds = {"a1": "two_finger", "a2": "vacuum"}
  path = tmp_path / "grasps.json.gz"
  assert problem.save_grasp_table_cache(path, table, "unit") == {"keys": 2, "rows": 2}
  loaded, stats = problem.load_grasp_table_cache(path, "unit")
  assert isinstance(loaded, problem.LazyGraspTable)
  assert loaded.materialized == 0 and stats == {"keys": 2, "rows": 2}
  assert np.array_equal(loaded[("obj_0",)][0][2], np.arange(7, dtype=float))
  assert loaded.usable(("obj_0",), "obj_0", 3, "a1")
  assert not loaded.usable(("obj_0",), "obj_0", 3, "a2")
  with pytest.raises(ValueError, match="fingerprint mismatch"):
    problem.load_grasp_table_cache(path, "stale")


def test_paper_motion_command_is_first_solution_only(tmp_path):
  command = PS._motion_command(tmp_path / "plan.json", 30.0, 7)
  assert "--no-mp-optimize" in command
  assert "--mp-no-shortcut" in command
  assert command[command.index("--mp-runtime") + 1] == "30.0"


def test_paper_breakdown_includes_aggregated_initial_motion_time():
  groups = B.aggregate([{
      "case": "cross", "variant": "ours", "success": True,
      "timed_out": False, "out_of_memory": False, "status": "success",
      "mp_initial_solution_s": 4.25,
  }])
  assert groups[0]["mp_initial_solution_s_mean"] == pytest.approx(4.25)
  tex = PS._breakdown_tex(groups)
  assert "t_{\\text{MP init}}" in tex
  assert "4.2" in tex


def test_paper_tables_use_requested_order_caption_and_explicit_uncertainty():
  records = []
  for case in PS.CASES:
    for seed in (3, 5):
      records.append({
          "case": case, "variant": "ours", "seed": seed,
          "success": True, "timed_out": False, "out_of_memory": False,
          "status": "success", "search_pipeline_s": 1.0, "plan_cost": 4,
          "grasp_generation_s": 1.0, "backward_graph_search_s": 1.0,
          "backward_force_lp_s": 1.0, "forward_search_s": 1.0,
          "ik_bind_wall_s": 1.0, "mp_initial_solution_s": 1.0,
      })
  groups = B.aggregate(records)
  comparison = PS._comparison_tex(groups)
  breakdown = PS._breakdown_tex(groups)
  assert "mean and standard deviation of 2 runs" in comparison
  assert "1.00 +- 0.00" in comparison
  assert "4 +-" not in comparison
  assert comparison.index("F. Beam") < comparison.index("F. Plumber block")
  assert comparison.index("F. Plumber block") < comparison.index("F. Car")
  assert comparison.index("F. Duct") < comparison.index("F. Stool")
  assert "Breakdown of where time is spent in \\textsc{Wrap}." in breakdown
  assert "1.00 +- 0.00" in breakdown


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


def test_resident_memory_limit_expands_without_becoming_an_address_space_limit():
  manifest = _matrix_manifest()
  manifest.pop("max_address_space_gb")
  manifest["max_rss_gb"] = 4
  runs = B.expand_runs(B.validate_manifest(manifest))
  assert all(run["max_rss_gb"] == 4.0 for run in runs)
  assert all(run["max_address_space_gb"] is None for run in runs)
  assert B._guarded_command(runs[0], ["planner"]) == ["planner"]


def test_resident_memory_watchdog_marks_process_tree_limit(monkeypatch):
  class FakeProcess:
    pid = 123

    @staticmethod
    def poll():
      return None

  monkeypatch.setattr(B, "_process_tree_rss_kb", lambda _pid: 2048)
  monkeypatch.setattr(B, "_terminate_process_group", lambda _proc: -15)
  got = B._wait_with_guards(FakeProcess(), timeout_s=10, max_rss_gb=0.001)
  assert got == (-15, False, True, 2048)


def test_repeated_expansion_keeps_logical_cell_and_adds_stable_repeat_ids():
  runs = B.expand_runs(B.validate_manifest(_matrix_manifest()), repeats=2)
  assert len(runs) == 8
  assert runs[0]["run_id"].endswith("__rep-01")
  assert runs[1]["run_id"].endswith("__rep-02")
  assert runs[0]["seed"] == runs[1]["seed"] == 3
  assert runs[0]["repeat"] == 1 and runs[1]["repeat"] == 2
  with pytest.raises(ValueError, match="positive integer"):
    B.expand_runs(B.validate_manifest(_matrix_manifest()), repeats=0)


def test_seed_override_replaces_manifest_case_and_variant_seeds():
  manifest = _manifest()
  manifest["seeds"] = [1]
  manifest["cases"][0]["seeds"] = [2]
  manifest["variants"][0]["seeds"] = [3]
  runs = B.expand_runs(B.validate_manifest(manifest), seeds=[41, 42])
  assert {run["seed"] for run in runs} == {41, 42}
  assert len(runs) == 2 * len(manifest["cases"]) * len(manifest["variants"])


def test_paper_manifest_uses_portable_plain_fabrica_configs():
  from config import RunConfig
  from main import parse_args

  manifest = json.loads(PS.MANIFEST.read_text(encoding="utf-8"))
  fabrica = [case for case in manifest["cases"]
             if "--search-profile" in case["args"]
             and case["args"][case["args"].index("--search-profile") + 1] == "fabrica"]
  assert len(fabrica) == 7
  for case in fabrica:
    args = case["args"]
    config_token = args[args.index("--config") + 1]
    config_path = B.ROOT / config_token
    assert not config_token.startswith("/")
    assert config_path.is_file()
    assert "experiments" not in config_path.parts
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert "plan" not in raw and "witnesses" not in raw
    cfg = RunConfig.from_file(config_path)
    assert cfg.assembly.startswith("fabrica-")

    # A historical saved plan used as --config silently supplied legacy replay defaults.
    # These benchmark configs must instead inherit the current profile/default semantics.
    resolved = parse_args([*manifest["defaults"], *args, "--seed", "4242"])
    assert resolved.seed == 4242
    assert resolved.grasp_sampler_db_hash is None
    assert resolved.mate_wrench_limits
    assert resolved.post_pick == pytest.approx(0.05)


def test_runner_can_stop_remaining_cell_trials_after_first_timeout(tmp_path, monkeypatch):
  manifest = _manifest()
  manifest["cases"] = manifest["cases"][:1]
  manifest["variants"] = manifest["variants"][:1]
  path = tmp_path / "manifest.json"
  path.write_text(json.dumps(manifest), encoding="utf-8")
  calls = []

  def fake_run_one(spec, *_args, **_kwargs):
    calls.append(spec["run_id"])
    return {
        "run_id": spec["run_id"], "case": spec["case"], "variant": spec["variant"],
        "seed": spec["seed"], "repeat": spec["repeat"], "tags": spec["tags"],
        "status": "timeout", "success": False, "timed_out": True, "wall_s": 12.0,
    }

  monkeypatch.setattr(B, "run_one", fake_run_one)
  monkeypatch.setattr(B, "provenance", lambda: {"source_sha256": "unit-source"})
  status = B.main([
      str(path), "--output", str(tmp_path / "results"), "--repeats", "2",
      "--stop-cell-on-timeout",
  ])
  assert status == 1
  assert calls == ["a__legacy__seed-3__rep-01"]


def test_runner_jobs_parallelizes_cases_but_serializes_each_case(tmp_path, monkeypatch):
  manifest = _manifest()
  path = tmp_path / "manifest.json"
  path.write_text(json.dumps(manifest), encoding="utf-8")
  lock = threading.Lock()
  active_cases = set()
  first_cases = set()
  overlap = []
  first_case_gate = threading.Barrier(2)

  def fake_run_one(spec, *_args, **_kwargs):
    with lock:
      assert spec["case"] not in active_cases
      active_cases.add(spec["case"])
      overlap.append(len(active_cases))
      first = spec["case"] not in first_cases
      first_cases.add(spec["case"])
    if first:
      first_case_gate.wait(timeout=2)
    with lock:
      active_cases.remove(spec["case"])
    return {
        "run_id": spec["run_id"], "case": spec["case"], "variant": spec["variant"],
        "seed": spec["seed"], "repeat": spec["repeat"], "tags": spec["tags"],
        "status": "success", "success": True, "timed_out": False, "wall_s": 0.1,
        "plan_cost": 1,
    }

  monkeypatch.setattr(B, "run_one", fake_run_one)
  monkeypatch.setattr(B, "provenance", lambda: {"source_sha256": "unit-source"})
  status = B.main([str(path), "--output", str(tmp_path / "results"), "--jobs", "2"])
  assert status == 0
  assert max(overlap) == 2


def test_process_wrench_plot_can_use_plan_cost(tmp_path):
  def sample(parts, robots, force, cost):
    return {"planner_command": ["python", "main.py", "--parts", str(parts),
                                "--num-robots", str(robots), "--insert-force", str(force)],
            "success": True, "plan_cost": cost, "milestone_search_s": 1.0}
  physical = [sample(2, 2, 0, 3), sample(2, 2, 10, 4), sample(2, 2, 20, 5),
              sample(2, 3, 0, 3), sample(2, 3, 10, 4), sample(2, 3, 20, 5),
              sample(2, 4, 0, 3), sample(2, 4, 10, 4), sample(2, 4, 20, 5)]
  # The real plot requires all six lambda columns; exercise the metric path through the
  # aggregator directly so this remains a small, deterministic unit test.
  rows = P._aggregate(physical, physical=True)
  assert rows[0]["plan_cost_mean"] == 3.0


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
                                   "paper_search_comparison.json",
                                   "search_finalists_4r.json",
                                   "search_manipulation_scope_ablation.json",
                                   "cube_scaling.json", "cube_scaling_piecewise.json",
                                   "cube_wrench_envelope_scaling.json",
                                   "cube_wrench_x100_high_friction.json",
                                   "cross_wrench_envelope_scaling.json",
                                   "cross_wrench_x100_high_friction.json",
                                   "cross_no_wrench_scaling.json"])
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
[fill-pdb obj_1:open-effort] settled=20 discovered=30 reverse=0.40s pick=2 wrench=3 exhausted=False
[fill-pdb obj_1:ok-open] settled=30 discovered=35 reverse=0.75s pick=2 wrench=4 exhausted=False
[fill-pdb obj_2:ok-open] settled=40 discovered=45 reverse=1.25s pick=3 wrench=5 exhausted=False
[fill-prof obj_1:ok-open] binds=6 failed=1 bind_time=2.2s
[fill-skeleton obj_1] VALIDATED 4-action direct schema after 3 candidate path(s), 2 failed IK edge(s), 1.25s
[fill-skeleton obj_2] assignment probe budget 12 reached, 9 failed IK edge(s), 2.50s -> complete local fallback
[milestones] fill-in for obj_2 failed (1.2s) -> blacklisted (1 repairs)
[milestones] complete: 19 actions in 49.12s (2 repairs)
[strict-collisions] vetoed 4 dirty KOMO witnesses (> 0.0001m total penetration).
[ik-clearance] vetoed 3 IK candidate(s) below the shared 5.0mm search/witness margin.
[ik-prof] binds: 432 called, 7 cache hits (2%), 425 solved (158 ok / 267 failed), bind wall 272.2s
[ik-prof] primitive totals (binds/failed/bind-wall; full-NLP count/time; precheck-NLP count/time):
[ik-prof]   pick       300/250    200.00s; full 4000/180.00s; precheck    0/  0.00s
[ik-prof]   handover    80/10      12.00s; full  120/ 10.00s; precheck   80/  1.00s
[ik-prof]   add_grasp   12/2        3.00s; full   20/  2.50s; precheck   12/  0.20s
[ik-prof]   place       30/5        5.00s; full   50/  4.00s; precheck    0/  0.00s
[ik-prof]   assemble    10/0        1.00s; full   12/  0.80s; precheck    2/  0.10s
[ik-prof] bind outcome time: successes 31.7s (12%), failures 240.5s (88%)
[ik-prof] NLP solves: 5655 in 244.1s (seeded 869/31.9s, unseeded 1587/77.3s, restarts 3199/134.9s)
[ik-prof] reach precheck: 402 relaxed solves in 96.5s (179 rejects caught); post-solve gates 513x in 0.6s; getPath calls 513
[ik-reuse] positive certificates: 44 accepted / 47 tried in 0.13s; library=143 vectors over 143 kinematic cores.
[ik-reuse] negative diagnostic ONLY (no pruning): 3 repeated exact-scene failures / 0.4s; 428 cross-scene same-core repeats / 16.6s optimistic nogood upper bound.
[ik-reuse] physical-query invariant: 37 symbolic aliases collapsed across 211 physical queries; exact witnesses and open failure effort are shared under the skill-local equivalence.
[search-phase-prof] pipeline=300.000000s grasp=8.000000s/12 grasp_backward=2.000000s/3 force_lp_backward=3.000000s/90 force_lp_forward=1.500000s/45
[pick-prof obj_1:ok-open] binds=120/100f targets=30 table-scenes=40 exact-query-reprobes=80; robots: a1=60/50f,15t/20s, a2=60/50f,15t/20s
[pick-prof] 300 binds/250 failed over 70 exact robot-grasp targets and 95 physical table scenes; exact-query reprobes=205; by robot: a1=150/125f,35 targets/48 scenes, a2=150/125f,35 targets/47 scenes
[fill-policy-widen obj_2] Bellman frontier adds one grasp batch (loose, context=('obj_2',), part=obj_2, class=1) ids [20, 21]; 1 demand batch(es) remain. One bounded forward-IK epoch; activated columns remain available to later policy branches.
[fill-policy-widen obj_2] grasp batch did not validate; retaining its finite column and advancing to the next Bellman demand.
[fill-policy-widen obj_2] Bellman frontier adds one rest/yaw orbit (group=('obj_0',), class=2); activated orbits and raw columns remain available to later policy branches. One bounded forward-IK epoch.
[rss] peak RSS after solve: 1712 MB
"""
  got = B.parse_log(log)
  assert got["backward_states"] == 45509
  assert got["backward_build_s"] == 12.5
  assert got["fill_pdb_reverse_s"] == 2.0       # max(obj_1 retries) + obj_2
  assert got["backward_search_s"] == 14.5
  assert got["milestone_search_s"] == 49.12
  assert got["milestone_repairs"] == 2
  assert got["ik_bind_calls"] == 432 and got["ik_bind_fail"] == 267
  assert got["ik_bind_success_s"] == 31.7 and got["ik_bind_fail_s"] == 240.5
  assert got["ik_pick_binds"] == 300 and got["ik_pick_bind_fail"] == 250
  assert got["ik_handover_binds"] == 80 and got["ik_handover_nlp_solves"] == 120
  assert got["ik_add_grasp_binds"] == 12 and got["ik_add_grasp_precheck_solves"] == 12
  assert got["ik_assemble_bind_s"] == 1.0
  assert got["ik_nlp_solves"] == 5655 and got["ik_restart_solves"] == 3199
  assert got["ik_precheck_rejects"] == 179
  assert got["ik_reuse_hits"] == 44 and got["ik_reuse_trials"] == 47
  assert got["ik_reuse_s"] == 0.13
  assert got["ik_negative_exact_repeats"] == 3
  assert got["ik_negative_exact_repeat_s"] == 0.4
  assert got["ik_negative_core_repeats"] == 428
  assert got["ik_negative_core_repeat_s"] == 16.6
  assert got["ik_physical_aliases"] == 37 and got["ik_physical_queries"] == 211
  assert got["search_pipeline_s"] == 300.0
  assert got["grasp_generation_s"] == 8.0 and got["grasp_generation_calls"] == 12
  assert got["grasp_backward_s"] == 2.0 and got["grasp_backward_calls"] == 3
  assert got["backward_force_lp_s"] == 3.0 and got["backward_force_lp_calls"] == 90
  assert got["forward_force_lp_s"] == 1.5 and got["forward_force_lp_calls"] == 45
  assert got["backward_graph_search_s"] == 9.5
  assert abs(got["forward_search_s"] - 7.3) < 1e-9
  assert got["pick_profile_binds"] == 300
  assert got["pick_profile_failures"] == 250
  assert got["pick_profile_targets"] == 70
  assert got["pick_profile_table_scenes"] == 95
  assert got["pick_profile_exact_reprobes"] == 205
  assert got["pick_profile_complete"] is True
  assert got["strict_collision_rejects"] == 4
  assert got["clearance_rejects"] == 3
  assert got["peak_rss_reported_mb"] == 1712
  assert got["policy_grasp_batches"] == 1
  assert got["policy_yaw_orbits"] == 1
  assert got["policy_failed_probes"] == 1
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


def test_log_parser_sums_completed_pick_profiles_for_timeout():
  got = B.parse_log("""
[pick-prof obj_1:ok-open] binds=20/15f targets=8 table-scenes=10 exact-query-reprobes=10; robots: a1=10/7f,4t/5s
[pick-prof obj_2:ok-open] binds=30/24f targets=12 table-scenes=15 exact-query-reprobes=15; robots: a1=15/12f,6t/8s
[open-witness obj_3] epoch 1 ended on the node budget 20000 (not exhaustion)
""")
  assert got["pick_profile_binds"] == 50
  assert got["pick_profile_failures"] == 39
  assert got["pick_profile_targets"] == 20
  assert got["pick_profile_table_scenes"] == 25
  assert got["pick_profile_exact_reprobes"] == 25
  assert got["pick_profile_complete"] is False


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


def test_aggregate_preserves_resource_failure_kinds():
  rows = [
      {"case": "a", "variant": "x", "status": "oom", "success": False,
       "timed_out": False, "out_of_memory": True},
      {"case": "a", "variant": "x", "status": "timeout", "success": False,
       "timed_out": True, "out_of_memory": False},
  ]
  got = B.aggregate(rows)[0]
  assert got["out_of_memory"] == 1
  assert got["timeouts"] == 1
  assert got["errors"] == 0 and got["no_plans"] == 0


def test_native_bad_alloc_is_classified_as_oom():
  """The native collision stack can report allocation exhaustion via std::bad_alloc."""
  assert B._looks_like_out_of_memory("terminate called: std::bad_alloc")
  assert B._looks_like_out_of_memory("Traceback\nMemoryError")
  assert not B._looks_like_out_of_memory("ordinary planner failure")


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


def test_cube_scaling_summary_does_not_turn_timeout_into_solution_time():
  def sample(parts, robots, success, **extra):
    return {
        "run_id": f"p{parts}-r{robots}",
        "planner_command": ["python", "main.py", "--parts", str(parts),
                            "--num-robots", str(robots)],
        "success": success,
        **extra,
    }

  rows = C.summarize([
      sample(2, 3, True, milestone_search_s=4.0),
      sample(2, 3, True, milestone_search_s=6.0),
      sample(3, 3, False, timed_out=True, wall_s=90.0),
  ])
  assert rows[0]["milestone_search_s_mean"] == 5.0
  assert rows[0]["milestone_search_s_sd"] == 1.0
  assert rows[1]["milestone_search_s_mean"] is None
  assert rows[1]["failures"] == 1 and rows[1]["timeouts"] == 1


def test_cube_scaling_per_part_normalizes_times_and_censored_markers():
  metric, rows = C._per_part_rows([{
      "parts": 4,
      "milestone_search_s_mean": 10.0,
      "milestone_search_s_sd": 2.0,
      "failed_elapsed_s_max": 20.0,
  }], "milestone_search_s")
  assert metric == "milestone_search_s_per_part"
  assert rows[0]["milestone_search_s_per_part_mean"] == 2.5
  assert rows[0]["milestone_search_s_per_part_sd"] == 0.5
  assert rows[0]["failed_elapsed_s_max"] == 5.0


def test_cube_scaling_component_residual_comes_from_raw_log(tmp_path):
  log = tmp_path / "run.log"
  log.write_text("""
[fill-pdb obj_1:open-effort] settled=2 discovered=3 reverse=0.50s pick=1 wrench=1 exhausted=False
[fill-pdb obj_1:ok-open] settled=3 discovered=3 reverse=0.75s pick=1 wrench=1 exhausted=False
[fill-pdb obj_2:ok-open] settled=4 discovered=5 reverse=1.25s pick=1 wrench=1 exhausted=False
[milestones] complete: 10 actions in 10.00s (0 repairs)
[ik-prof] binds: 8 called, 0 cache hits (0%), 8 solved (2 ok / 6 failed), bind wall 6.0s
""")
  result = {
      "run_id": "p3-r2", "success": True, "milestone_search_s": 10.0,
      "planner_command": ["python", "main.py", "--parts", "3", "--num-robots", "2"],
      "artifacts": {"log": "run.log"},
  }
  got = C._with_component_metrics(tmp_path, [result])[0]
  assert got["ik_bind_wall_s"] == 6.0
  assert got["backward_search_s"] == 2.0
  assert got["residual_search_s"] == 2.0
