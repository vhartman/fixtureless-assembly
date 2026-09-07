"""RunConfig <-> CLI parity, persistence, profile precedence, and --config layering."""
import pytest

import main
from config import RunConfig


def _parser_defaults():
  import argparse
  p = main.build_parser()
  # SUPPRESS-default actions are auxiliary switches sharing a primary flag's dest
  # (e.g. --no-strict-collisions); the primary action carries the real default.
  return {a.dest: a.default for a in p._actions
          if a.dest not in ("help", "config", "search_profile", "motion_profile")
          and a.default is not argparse.SUPPRESS}


def test_defaults_match_parser():
  """Every RunConfig default equals the argparse default for the same dest, and the field SETS
  are identical -- so the mirror can never silently drift from the real parser."""
  pd = _parser_defaults()
  cd = RunConfig().to_dict()
  assert set(cd) == set(pd), \
      f"field mismatch: only-RunConfig {set(cd)-set(pd)}, only-parser {set(pd)-set(cd)}"
  for k in pd:
    assert cd[k] == pd[k], f"default drift on {k}: RunConfig={cd[k]!r} parser={pd[k]!r}"


def test_default_for_unknown_key_raises():
  with pytest.raises(KeyError):
    RunConfig.default_for("not_a_flag")


def test_saved_cfg_saved_wins_missing_default_legacy_override():
  from config import saved_cfg
  assert saved_cfg({"seed": 7}, "seed") == 7
  assert saved_cfg({}, "parts") == RunConfig.default_for("parts")
  assert saved_cfg({}, "part_float", legacy=0.015) == 0.015


def test_namespace_roundtrip():
  """A parsed namespace survives from_namespace -> to_namespace unchanged (all fields but the
  meta `config`)."""
  ns = main.parse_args([])
  ns2 = RunConfig.from_namespace(ns).to_namespace()
  a, b = vars(ns), vars(ns2)
  for k in (set(a) | set(b)) - {"config"}:
    assert a.get(k) == b.get(k), f"roundtrip changed {k}: {a.get(k)!r} -> {b.get(k)!r}"


def test_dict_roundtrip_and_unknown_key():
  rc = RunConfig(assembly="chair", parts=8, milestones=True, insert_force=25.0)
  assert RunConfig.from_dict(rc.to_dict()) == rc
  with pytest.raises(ValueError):
    RunConfig.from_dict({"assembly": "chair", "not_a_flag": 1})


def test_file_roundtrip(tmp_path):
  rc = RunConfig(assembly="car", parts=5, num_robots=3, wrench=True, vacuum_scale=0.1)
  p = tmp_path / "run.json"
  rc.to_file(str(p))
  assert RunConfig.from_file(str(p)) == rc


def test_config_cli_layering(tmp_path):
  """--config FILE is the BASE; an explicit CLI flag overrides it; unspecified flags take the
  file value; anything the file omits keeps the built-in default."""
  RunConfig(assembly="chair", parts=8, num_robots=3, seed=7).to_file(str(tmp_path / "c.json"))
  a = main.parse_args(["--config", str(tmp_path / "c.json")])
  assert (a.assembly, a.parts, a.num_robots, a.seed) == ("chair", 8, 3, 7)   # file only
  a = main.parse_args(["--config", str(tmp_path / "c.json"), "--seed", "1", "--parts", "5"])
  assert (a.assembly, a.parts, a.num_robots, a.seed) == ("chair", 5, 3, 1)   # CLI wins


def test_current_sampler_marker_is_migrated_and_obsolete_markers_fail():
  cfg = RunConfig.from_dict({"grasp_sampler": True, "grasp_sampler_version": 4,
                             "mixed_wrench_h": True})
  assert cfg.grasp_sampler and "grasp_sampler_version" not in cfg.to_dict()
  assert "mixed_wrench_h" not in cfg.to_dict()
  with pytest.raises(ValueError, match="versions 1-3 are no longer supported"):
    RunConfig.from_dict({"grasp_sampler": True, "grasp_sampler_version": 3})


def test_open_profile_is_a_compatibility_alias_and_fast_open_only_caps_seeds():
  alias = main.parse_args(["--search-profile", "open"])
  standard = main.parse_args(["--search-profile", "standard"])
  assert {k: getattr(alias, k) for k in main._STANDARD_SEARCH_PROFILE} == {
      k: getattr(standard, k)
      for k in main._STANDARD_SEARCH_PROFILE}
  fast = main.parse_args(["--search-profile", "fast-open"])
  differences = {k for k in main._STANDARD_SEARCH_PROFILE
                 if getattr(fast, k) != getattr(standard, k)}
  assert differences == {"analytic_seed_cap"}
  assert standard.analytic_seed_cap is None and fast.analytic_seed_cap is not None


def test_optimistic_dispatch_uses_the_shared_local_fill_settings(monkeypatch):
  seen = {}

  def fake(*_args, **kwargs):
    seen.update(kwargs)
    return []

  monkeypatch.setattr(main, "solve_assembly_problem_optimistic", fake)
  cfg = RunConfig(optimistic=True, lazy_fills=True, optimistic_fallback=False,
                  optimistic_chain_h=True, wrench_aware_h=False)
  assert main._dispatch_solve(cfg, None, None, [], None) == []
  assert seen["lazy_fills"] is True
  assert seen["fallback"] is False
  assert seen["chain_h"] is True
  assert seen["wrench_aware_h"] is False

  assert main.parse_args(["--lazy-fills"]).lazy_fills is True
  assert main.parse_args(["--no-lazy-fills"]).lazy_fills is False
  assert main.parse_args(["--optimistic-chain-h"]).optimistic_chain_h is True
  assert main.parse_args(["--no-optimistic-chain-h"]).optimistic_chain_h is False


def test_structural_milestone_oracle_cli_and_dispatch(monkeypatch):
  seen = {}

  def fake(*_args, **kwargs):
    seen.update(kwargs)
    return []

  monkeypatch.setattr(main, "solve_assembly_problem_milestones", fake)
  cfg = RunConfig(milestones=True, milestone_oracle="structural",
                  fill_local_h=True, fill_exact=True, fill_greedy=True,
                  fill_incumbent_first=True,
                  fill_policy_first=True,
                  fill_policy_guide=True,
                  fill_skeleton_first=True,
                  fill_scope_widening=True, milestone_boundaries=3,
                  wrench_aware_h=False)
  assert main._dispatch_solve(cfg, None, None, [], None) == []
  assert seen["milestone_oracle"] == "structural"
  assert seen["fill_local_h"] is True and seen["fill_exact"] is True
  assert seen["fill_greedy"] is True
  assert seen["fill_incumbent_first"] is True
  assert seen["fill_policy_first"] is True
  assert seen["fill_policy_guide"] is True
  assert seen["fill_skeleton_first"] is True
  assert seen["fill_scope_widening"] is True
  assert seen["milestone_boundaries"] == 3
  assert seen["wrench_aware_h"] is False
  assert main.parse_args(["--milestone-oracle", "structural"]).milestone_oracle == "structural"
  assert main.parse_args(["--milestone-oracle", "chain"]).milestone_oracle == "chain"
  assert main.parse_args(["--fill-local-h", "--fill-exact"]).fill_local_h is True
  assert main.parse_args([
      "--fill-target-support-quotient"
  ]).fill_target_support_quotient is True
  assert main.parse_args(["--fill-local-h", "--fill-exact"]).fill_exact is True
  assert main.parse_args(["--fill-greedy"]).fill_greedy is True
  assert main.parse_args(["--no-fill-greedy"]).fill_greedy is False
  assert main.parse_args(["--fill-incumbent-first"]).fill_incumbent_first is True
  assert main.parse_args(["--fill-skeleton-first"]).fill_skeleton_first is True
  assert main.parse_args(["--fill-policy-first"]).fill_policy_first is True
  assert main.parse_args(["--no-fill-policy-first"]).fill_policy_first is False
  assert main.parse_args(["--fill-policy-guide"]).fill_policy_guide is True
  assert main.parse_args(["--no-fill-policy-guide"]).fill_policy_guide is False
  assert main.parse_args(["--fill-scope-widening"]).fill_scope_widening is True
  assert main.parse_args(["--no-fill-scope-widening"]).fill_scope_widening is False
  assert main.parse_args(["--fill-touch-all"]).fill_touch_all is True
  assert main.parse_args(["--no-fill-touch-all"]).fill_touch_all is False
  assert main.parse_args(["--no-wrench-aware-h"]).wrench_aware_h is False
  assert main.parse_args(["--lazy-edge-selector", "defer-failed"]).lazy_edge_selector == \
      "defer-failed"
  assert main.parse_args([
      "--diagnose-plan-rank", "known.json"
  ]).diagnose_plan_rank == "known.json"


def test_incumbent_first_requires_exact_lazy_milestone_fills():
  with pytest.raises(SystemExit, match="requires --fill-local-h"):
    main._validate_solver_scoped_flags(RunConfig(
        milestones=True, fill_incumbent_first=True, fill_exact=False))
  with pytest.raises(SystemExit, match="requires --fill-local-h"):
    main._validate_solver_scoped_flags(RunConfig(
        milestones=True, fill_local_h=True, fill_incumbent_first=True,
        fill_exact=True, lazy_fills=False))
  assert main._validate_solver_scoped_flags(RunConfig(
      milestones=True, fill_local_h=True, fill_incumbent_first=True,
      fill_exact=True, lazy_fills=True)) == "milestones"


def test_fill_greedy_is_a_lazy_milestone_policy():
  with pytest.raises(SystemExit, match="implemented only by milestone local fills"):
    main._validate_solver_scoped_flags(RunConfig(
        milestones=False, fill_greedy=True, lazy_fills=True))
  with pytest.raises(SystemExit, match="requires --lazy-fills"):
    main._validate_solver_scoped_flags(RunConfig(
        milestones=True, fill_greedy=True, lazy_fills=False))
  assert main._validate_solver_scoped_flags(RunConfig(
      milestones=True, fill_greedy=True, lazy_fills=True)) == "milestones"


def test_target_support_quotient_is_a_no_wrench_local_pdb_ablation():
  with pytest.raises(SystemExit, match="requires --fill-local-h"):
    main._validate_solver_scoped_flags(RunConfig(
        milestones=True, fill_target_support_quotient=True, fill_local_h=False,
        wrench=False))
  with pytest.raises(SystemExit, match="requires --fill-local-h"):
    main._validate_solver_scoped_flags(RunConfig(
        milestones=True, fill_target_support_quotient=True, fill_local_h=True,
        wrench=True))
  assert main._validate_solver_scoped_flags(RunConfig(
      milestones=True, fill_target_support_quotient=True, fill_local_h=True,
      wrench=False, geo_facts=False)) == "milestones"


def test_standard_uses_greedy_local_fills_without_changing_global_weight():
  cfg = main.parse_args(["--search-profile", "standard"])
  assert cfg.fill_local_h is True
  assert cfg.fill_greedy is True
  assert cfg.fill_exact is False
  assert cfg.weight == 1.0


def test_milestone_anytime_receives_greedy_incumbent_policy(monkeypatch):
  seen = {}

  def fake(*_args, **kwargs):
    seen.update(kwargs)
    return []

  monkeypatch.setattr(main, "solve_assembly_problem_milestones_anytime", fake)
  cfg = RunConfig(milestones_anytime=30.0, milestones=True, fill_local_h=True,
                  fill_greedy=True, fill_exact=False, lazy_fills=True, weight=1.0,
                  placement_yaws=4, placement_yaw_widening=True)
  assert main._validate_solver_scoped_flags(cfg) == "milestones-anytime"
  assert main._dispatch_solve(cfg, None, None, [], None) == []
  assert seen["fill_greedy"] is True
  assert seen["fill_exact"] is False
  assert seen["weight"] == 1.0
  assert seen["placement_yaw_widening"] is True


def test_solver_scoped_policy_flags_fail_closed_in_other_engines():
  # `--milestones` defaults to true, but dispatch gives explicit solver families priority.
  # The validation must follow that same precedence instead of treating the dormant default as
  # evidence that a milestone-only schedule will run.
  lazy = RunConfig(milestones=True, lazy=True, fill_policy_guide=True,
                   fill_local_h=True)
  assert main._selected_solver_engine(lazy) == "lazy"
  with pytest.raises(SystemExit, match="fill-policy-guide.*selected engine is lazy"):
    main._validate_solver_scoped_flags(lazy)

  optimistic = RunConfig(milestones=True, optimistic=True, grasp_local_widen=True)
  with pytest.raises(SystemExit, match="grasp-local-widen.*selected engine is optimistic"):
    main._validate_solver_scoped_flags(optimistic)

  exact = RunConfig(milestones=False, fill_policy_first=True, fill_local_h=True)
  with pytest.raises(SystemExit, match="fill-policy-first.*selected engine is exact"):
    main._validate_solver_scoped_flags(exact)

  ordinary = RunConfig(milestones=True, fill_policy_guide=True, fill_local_h=True,
                       grasp_local_widen=True)
  assert main._validate_solver_scoped_flags(ordinary) == "milestones"


def test_saved_plan_persists_resolved_search_semantics(tmp_path):
  """A named-profile result must remain reproducible after the profile itself evolves."""
  cfg = RunConfig.from_namespace(main.parse_args(["--search-profile", "standard"]))
  out = tmp_path / "plan.json"
  main.save_plan(str(out), cfg, [])
  loaded = RunConfig.from_file(str(out))
  presentation_only = {"quiet_rai", "view"}
  assert {k: v for k, v in loaded.to_dict().items() if k not in presentation_only} == {
      k: v for k, v in cfg.to_dict().items() if k not in presentation_only}


def test_saved_plan_persists_milestone_anytime_budget(tmp_path):
  cfg = RunConfig.from_namespace(main.parse_args([
      "--search-profile", "standard", "--milestones-anytime", "300",
  ]))
  out = tmp_path / "anytime-plan.json"
  main.save_plan(str(out), cfg, [])
  loaded = RunConfig.from_file(str(out))
  assert loaded.milestones_anytime == 300.0
  assert loaded.fill_greedy is True
  assert loaded.weight == 1.0


def test_saved_plan_persists_target_support_quotient(tmp_path):
  cfg = RunConfig.from_namespace(main.parse_args([
      "--search-profile", "standard", "--no-wrench",
      "--fill-target-support-quotient",
  ]))
  out = tmp_path / "target-support-quotient-plan.json"
  main.save_plan(str(out), cfg, [])
  loaded = RunConfig.from_file(str(out))
  assert loaded.fill_target_support_quotient is True
  assert loaded.wrench is False


def test_every_nonpresentation_standard_setting_is_persisted():
  """A profile change must not silently create a replay/re-solve semantic gap."""
  presentation_only = {"quiet_rai", "view"}
  assert set(main._STANDARD_SEARCH_PROFILE) - presentation_only <= \
      set(main._SAVE_CONFIG_FIELDS)


def test_orientation_robust_wrench_cli_and_plan_roundtrip(tmp_path):
  cfg = RunConfig.from_namespace(main.parse_args(["--orientation-robust-wrench"]))
  assert cfg.orientation_robust_wrench is True
  out = tmp_path / "orientation-robust-plan.json"
  main.save_plan(str(out), cfg, [])
  assert RunConfig.from_file(str(out)).orientation_robust_wrench is True


def test_max_robotiq_wrench_profile_cli_and_plan_roundtrip(tmp_path):
  cfg = RunConfig.from_namespace(main.parse_args([
      "--search-profile", "standard", "--wrench-profile", "max_robotiq",
  ]))
  assert cfg.wrench_profile == "max_robotiq"
  assert cfg.grasp_model == "asymmetric"
  out = tmp_path / "max-robotiq-plan.json"
  main.save_plan(str(out), cfg, [])
  assert RunConfig.from_file(str(out)).wrench_profile == "max_robotiq"


def test_open_plan_motion_loop_cli_and_config_roundtrip():
  a = main.parse_args([
      "--mp-open-plans", "--mp-plan-candidates", "5",
      "--mp-plan-epochs", "6", "--mp-plan-initial-runtime", "2.5",
  ])
  assert a.mp_open_plans
  assert a.mp_plan_candidates == 5 and a.mp_plan_epochs == 6
  assert a.mp_plan_initial_runtime == 2.5
  cfg = RunConfig.from_namespace(a)
  assert RunConfig.from_dict(cfg.to_dict()) == cfg


def test_open_plan_loop_suspends_then_fairly_reopens(tmp_path, monkeypatch):
  """Two failed small pulls must precede either plan's second, larger pull."""
  import json
  import subprocess
  import plan_io
  import solver

  plan_a = [("pick", "a1", "obj_1", 0)]
  plan_b = [("pick", "a2", "obj_1", 1)]
  session_caps = []

  def candidate_session(cfg, *_args, **_kwargs):
    session_caps.append(cfg.component_grasp_keep)
    # First exhaust the standard restricted component alphabet, then enumerate from its
    # exact full-union completion. The fair MP behavior below is otherwise unchanged.
    return iter(()) if len(session_caps) == 1 else iter((plan_a, plan_b))

  monkeypatch.setattr(main, "_milestone_candidate_session", candidate_session)
  import problem
  installed_component_caps = []
  monkeypatch.setattr(problem, "set_component_grasp_keep",
                      installed_component_caps.append)
  monkeypatch.setattr(solver, "hoist_releases", lambda *_args, **_kwargs: list(_args[2]))
  monkeypatch.setattr(solver, "collect_plan_witnesses", lambda *_args, **_kwargs: {})
  monkeypatch.setattr(solver, "collect_plan_pre_witnesses", lambda *_args, **_kwargs: {})
  monkeypatch.setattr(plan_io, "require_complete_witnesses", lambda *_args, **_kwargs: None)
  monkeypatch.setattr(plan_io, "witness_block", lambda *_args, **_kwargs: {})

  def fake_save(path, _cfg, plan, witnesses=None):
    with open(path, "w") as f:
      json.dump({"config": {}, "plan": [list(a) for a in plan],
                 "witnesses": witnesses}, f)

  monkeypatch.setattr(main, "save_plan", fake_save)

  calls = []

  def fake_run(cmd, **_kwargs):
    cmd = list(cmd)
    calls.append(cmd)
    mp_path = cmd[cmd.index("--mp-save") + 1]
    if "--mp-export-save" in cmd:
      export_path = cmd[cmd.index("--mp-export-save") + 1]
      with open(export_path, "w") as f:
        f.write("{}")
    # candidate A's reopened pull succeeds; both initial pulls fail.
    if len(calls) == 3:
      with open(mp_path, "w") as f:
        f.write("[]")
    return subprocess.CompletedProcess(cmd, 0)

  monkeypatch.setattr(subprocess, "run", fake_run)

  class DummyConfig:
    def getJointNames(self):
      return []

  class DummyIK:
    C = DummyConfig()

  cfg = RunConfig(
      solve=True, motion_plan=True, milestones=True, real_ik=True,
      wrench=False,
      save=str(tmp_path / "selected-plan.json"),
      mp_save=str(tmp_path / "selected-mp.json"),
      mp_export_save=str(tmp_path / "selected-export.json"),
      mp_open_plans=True, mp_plan_candidates=2, mp_plan_epochs=2,
      mp_plan_initial_runtime=1.0, mp_runtime=2.0, mp_seed=17,
      component_grasp_keep=32, grasp_sampler=True,
      hoist_releases=False, view=False, quiet_rai=True,
  )
  result = main._run_open_plan_loop(cfg, object(), object(), ["a1", "a2"], DummyIK())
  assert result["plan_id"]
  assert session_caps == [32, 0]
  assert installed_component_caps == [0]
  assert len(calls) == 3
  seeds = [int(c[c.index("--mp-seed") + 1]) for c in calls]
  assert seeds == [17, 18, 100020]
  runtimes = [float(c[c.index("--mp-runtime") + 1]) for c in calls]
  assert runtimes == [1.0, 1.0, 2.0]
  assert (tmp_path / "selected-plan.json").exists()
  assert (tmp_path / "selected-mp.json").exists()
  manifests = list(tmp_path.glob("selected-mp-open/*/manifest.json"))
  assert len(manifests) == 1
  manifest = json.loads(manifests[0].read_text())
  assert manifest["status"] == "success" and manifest["selected_candidate"] == 1
  assert [a["status"] for a in manifest["candidates"][0]["attempts"]] == [
      "no_path", "success"]
  assert [a["status"] for a in manifest["candidates"][1]["attempts"]] == ["no_path"]


def test_motion_profile_is_reversible_by_explicit_cli():
  a = main.parse_args([
      "--motion-profile", "standard", "--mp-collision-tolerance", "0.002",
      "--mp-optimize", "--mp-runtime", "12", "--mp-stall-timeout", "3",
      "--mp-shortcut-iters", "17", "--mp-table-margin", "0.004",
      "--no-fuse-cograsp", "--no-fuse-pre-skill",
  ])
  assert a.mp_collision_tolerance == 0.002 and a.mp_optimize
  assert a.mp_runtime == 12.0 and a.mp_stall_timeout == 3.0
  assert a.mp_shortcut_iters == 17 and a.mp_table_margin == 0.004
  assert not a.fuse_cograsp and not a.fuse_pre_skill


def test_profile_values_are_reversible_by_explicit_cli():
  a = main.parse_args([
      "--search-profile", "fast-open", "--weight", "5", "--no-open-witness",
      "--analytic-seed-cap", "9", "--witness-margin", "0.002", "--view",
      "--no-quiet-rai", "--no-grasp-sampler", "--no-grasp-sampler-scenes",
      "--no-grasp-refine", "--no-grasp-widen", "--component-grasp-keep", "0",
  ])
  assert a.weight == 5.0 and not a.open_witness and a.analytic_seed_cap == 9
  assert a.witness_margin == 0.002 and a.view and not a.quiet_rai
  assert (not a.grasp_sampler and not a.grasp_sampler_scenes and not a.grasp_refine
          and not a.grasp_widen)
  assert a.component_grasp_keep == 0


def test_profile_wins_over_config_and_individual_cli_wins_over_both(tmp_path):
  RunConfig(weight=7.0, open_witness=False, witness_margin=0.003).to_file(
      str(tmp_path / "c.json"))
  a = main.parse_args([
      "--search-profile", "fast-open", "--config", str(tmp_path / "c.json")])
  assert a.weight == 1.0 and a.open_witness and a.witness_margin == 0.005
  assert a.analytic_seed_cap == 3
  b = main.parse_args([
      "--search-profile", "standard", "--config", str(tmp_path / "c.json"),
      "--weight", "9", "--open-witness"])
  assert b.weight == 9.0 and b.open_witness


def test_loaded_plan_is_not_hoisted_twice_without_explicit_request():
  """A saved plan already contains post-passes; its saved true flag is provenance only."""
  fresh = RunConfig(hoist_releases=True)
  loaded = RunConfig(load_plan="plan.json", hoist_releases=True)
  assert main._should_hoist_plan(fresh, set())
  assert not main._should_hoist_plan(loaded, set())
  assert main._should_hoist_plan(loaded, {"hoist_releases"})
  loaded.hoist_releases = False
  assert not main._should_hoist_plan(loaded, {"hoist_releases"})


def test_compact_placement_yaw_prefix_is_fresh_search_only(monkeypatch):
  """A saved non-prefix placement ID must remain visible to every plan consumer."""
  import problem

  seen = []
  monkeypatch.setattr(problem, "set_placement_yaw_prefix", seen.append)

  fresh = RunConfig(solve=True, placement_yaws=4, placement_yaw_widening=True)
  assert main._set_search_placement_yaw_prefix(fresh, active=True) == 1
  assert seen[-1] == 1

  # --load-plan wins even if --solve is also present: neither spelling may reactivate the
  # compact search prefix while reconstructing witnesses or a motion export.
  for solve in (False, True):
    loaded = RunConfig(solve=solve, load_plan="plan.json", placement_yaws=4,
                       placement_yaw_widening=True)
    assert main._set_search_placement_yaw_prefix(loaded, active=True) is None
    assert seen[-1] is None

  assert main._set_search_placement_yaw_prefix(fresh, active=False) is None
  assert seen[-1] is None


def test_explicit_release_hoist_can_reuse_the_source_witness_certificate():
  """Hoisting is a post-pass, not a property of the source IK scene.

  The transaction itself re-certifies changed candidate edges.  A different requested hoist
  setting must therefore not discard the valid persisted certificate for the unmodified source
  plan before that transaction gets a chance to roll back to it.
  """
  saved_run = RunConfig(hoist_releases=False)
  saved = {k: getattr(saved_run, k) for k in main._WITNESS_FIELDS}
  assert main._witnesses_baseline_compatible(saved, RunConfig(hoist_releases=True))


def test_optimizer_change_keeps_search_witness_as_baseline_but_reoptimizes():
  saved_run = RunConfig(optimize_witnesses=None, witness_branch_budget=0)
  saved = {k: getattr(saved_run, k) for k in main._WITNESS_FIELDS}
  current = RunConfig(optimize_witnesses="floor", witness_branch_budget=5)
  assert main._witnesses_baseline_compatible(saved, current)
  assert not main._witnesses_reusable(saved, current)


def test_feasibility_setting_change_rejects_saved_witness_baseline():
  saved_run = RunConfig(strict_collisions=1e-4)
  saved = {k: getattr(saved_run, k) for k in main._WITNESS_FIELDS}
  current = RunConfig(strict_collisions=1e-6)
  assert not main._witnesses_baseline_compatible(saved, current)


def test_placement_yaws_are_parsed_persisted_and_invalidate_witnesses(tmp_path):
  import json

  cfg = RunConfig.from_namespace(main.parse_args(["--placement-yaws", "8"]))
  assert cfg.placement_yaws == 8
  out = tmp_path / "yaw-plan.json"
  main.save_plan(str(out), cfg, [])
  assert json.loads(out.read_text())["config"]["placement_yaws"] == 8
  assert RunConfig.from_file(str(out)).placement_yaws == 8

  saved_run = RunConfig(placement_yaws=1)
  saved = {k: getattr(saved_run, k) for k in main._WITNESS_FIELDS}
  assert not main._witnesses_baseline_compatible(saved, RunConfig(placement_yaws=8))


def test_mp_export_clearance_settings_default_to_saved_artifact():
  """Zero-IK export replay is exact unless this invocation explicitly retunes planning."""
  cfg = RunConfig(mp_collision_tolerance=0.7, mp_robot_margin=0.6,
                  mp_margin_ramp=0.5, mp_table_margin=0.4, part_float=0.03)
  saved = {"collision_tolerance": 1e-4, "robot_robot_margin": 0.1,
           "robot_margin_ramp": 0.1, "table_margin": 0.02, "table_float": 0.007}
  values, sources = main._merge_mp_export_clearance_settings(cfg, saved, explicit=set())
  assert values == saved
  assert set(sources.values()) == {"export"}


def test_explicit_cli_overrides_mp_export_clearance_settings_independently():
  """A named CLI knob wins without accidentally replacing the other saved defaults."""
  cfg = RunConfig(mp_collision_tolerance=0.2, mp_robot_margin=0.0,
                  mp_margin_ramp=0.03, mp_table_margin=0.04, part_float=0.012)
  saved = {"collision_tolerance": 1e-4, "robot_robot_margin": 0.1,
           "robot_margin_ramp": 0.1, "table_margin": 0.02, "table_float": 0.007}
  values, sources = main._merge_mp_export_clearance_settings(
      cfg, saved, explicit={"mp_collision_tolerance", "mp_robot_margin", "mp_table_margin"})
  assert values == {"collision_tolerance": 0.2, "robot_robot_margin": 0.0,
                    "robot_margin_ramp": 0.1, "table_margin": 0.04,
                    "table_float": 0.012}
  assert sources == {"collision_tolerance": "CLI", "robot_robot_margin": "CLI",
                     "robot_margin_ramp": "export", "table_margin": "CLI",
                     "table_float": "CLI/scene"}


def test_explicit_zero_table_margin_also_zeros_derived_table_float():
  cfg = RunConfig(mp_table_margin=0.0, part_float=0.012)
  values, sources = main._merge_mp_export_clearance_settings(
      cfg, {"table_margin": 0.02, "table_float": 0.007},
      explicit={"mp_table_margin", "mp_margin_ramp"})
  assert values["table_margin"] == 0.0
  assert values["table_float"] == 0.0
  assert values["robot_margin_ramp"] == cfg.mp_margin_ramp
  assert sources["table_float"] == "CLI/scene"


def test_run_library_entry():
  """main.run(RunConfig) is the argparse-free library entry (main() is now a thin wrapper over
  it). A no-solve config builds the scene and returns without any CLI/viewer involvement."""
  main.run(RunConfig(assembly="cross", parts=3, num_robots=2, solve=False, view=False))


def test_grasp_lever_is_a_short_half_extent():
  """The wrench model takes moments about the part CENTROID, so the lever is a surface
  half-extent -- always short. Regression guard: a gripper-to-COM distance (~326mm) once
  slipped in here and its weight moment alone exceeded the grasp torque cap."""
  import problem as P
  a = P.make_assembly("cross", 5)
  for part in a.objects:
    lev = a.grasp_lever(part)
    assert 0.0 < lev < 0.35, f"{part} lever {lev} is not a plausible grasp moment arm"
    kind, p = a.part_shape(part)[0], a.part_shape(part)[1]
    if kind in ("box", "ssBox"):
      assert abs(lev - 0.5 * max(p[:3])) < 1e-9


def test_group_lever_adds_only_the_centroid_offset():
  import numpy as np
  import problem as P
  a = P.make_assembly("cross", 5)
  parts = sorted(a.objects)[:2]
  root = parts[0]
  at = {parts[0]: np.zeros(3), parts[1]: np.array([0.4, 0.0, 0.0])}
  solo, m1, _ = a.group_lever([root], root, at.get)
  pair, m2, com = a.group_lever(parts, root, at.get)
  assert abs(solo - a.grasp_lever(root)) < 1e-9        # a single part: just its half-extent
  assert pair > solo                                    # a group: plus the centroid offset
  assert m2 > m1                                        # and the mass is the whole group's
  assert 0.0 < com[0] <= 0.4

def test_isolated_pick_preflight_does_not_require_fixed_assembly_root():
  """The root may remain on the table while incoming parts are acquired."""
  asm = type("Assembly", (), {"base": "obj_0"})()
  report = {"obj_0": {"ok": False}, "obj_1": {"ok": True},
            "obj_2": {"ok": False}}
  assert main._unpickable_required_parts(asm, report) == ["obj_2"]
