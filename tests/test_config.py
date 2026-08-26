"""RunConfig <-> CLI parity, persistence, and --config layering. Guards the packaging refactor:
RunConfig must mirror the argparse parser EXACTLY (drift test), and the no-config CLI path must
stay byte-identical to before (only the new `config` field is added)."""
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


def test_default_for_matches_field_defaults():
  """RunConfig.default_for is THE source of defaults for saved-config rebuilds: it must agree
  with the live field defaults, or replay/sim fallbacks silently drift."""
  rc = RunConfig()
  for k, v in rc.to_dict().items():
    assert RunConfig.default_for(k) == v, f"default_for drift on {k}"


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


def test_historical_sampler_config_defaults_to_v1(tmp_path):
  """A pre-version positional sampled DB must never be silently rebuilt as hybrid V2."""
  p = tmp_path / "old-sampler.json"
  p.write_text('{"grasp_sampler": true, "grasp_sampler_keep": 12}')
  cfg = RunConfig.from_file(str(p))
  assert cfg.grasp_sampler and cfg.grasp_sampler_version == 1


def test_standard_search_profile_resolves_robust_batch_defaults():
  a = main.parse_args(["--search-profile", "standard"])
  assert a.weight == 1.0
  assert a.milestones and a.sig_collapsed_h and a.sig_components == "press"
  assert a.milestone_oracle == "structural"
  assert a.fill_local_h and a.fill_exact and not a.fill_skeleton_first
  assert a.fill_scope_widening
  assert not a.fill_touch_all
  assert a.milestone_boundaries == 4
  assert a.real_ik and a.wrench and a.wrench_aware_h
  assert a.grasp_model == "asymmetric"
  assert a.optimize_witnesses == "floor"
  assert a.witness_margin == 0.005
  assert a.witness_branch_budget == 5 and a.witness_sets == 4
  assert a.tight_fingers and a.strict_collisions == 1e-6
  assert a.open_witness and a.open_witness_batch == 7 and a.open_witness_epochs == 2
  assert a.analytic_seed_cap is None
  assert a.grasp_sampler and a.grasp_sampler_version == 2
  assert a.grasp_sampler_scenes and a.grasp_sampler_keep == 20
  assert a.grasp_sampler_candidates == 96
  assert a.grasp_refine
  assert a.grasp_sampler_wrap == 0.015
  assert a.quiet_rai and not a.view


def test_open_profile_is_a_compatibility_alias_and_fast_open_only_caps_seeds():
  a = main.parse_args(["--search-profile", "open"])
  standard = main.parse_args(["--search-profile", "standard"])
  assert a.open_witness and a.open_witness_batch == 7 and a.open_witness_epochs == 2
  assert a.weight == 1.0 and a.fill_local_h and a.fill_exact
  assert {k: getattr(a, k) for k in main._STANDARD_SEARCH_PROFILE} == {
      k: getattr(standard, k)
      for k in main._STANDARD_SEARCH_PROFILE}
  assert a.analytic_seed_cap is None
  b = main.parse_args(["--search-profile", "fast-open"])
  assert b.open_witness and b.analytic_seed_cap == 3


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
                  fill_local_h=True, fill_exact=True, fill_skeleton_first=True,
                  fill_scope_widening=True, milestone_boundaries=3,
                  wrench_aware_h=False)
  assert main._dispatch_solve(cfg, None, None, [], None) == []
  assert seen["milestone_oracle"] == "structural"
  assert seen["fill_local_h"] is True and seen["fill_exact"] is True
  assert seen["fill_skeleton_first"] is True
  assert seen["fill_scope_widening"] is True
  assert seen["milestone_boundaries"] == 3
  assert seen["wrench_aware_h"] is False
  assert main.parse_args(["--milestone-oracle", "structural"]).milestone_oracle == "structural"
  assert main.parse_args(["--milestone-oracle", "chain"]).milestone_oracle == "chain"
  assert main.parse_args(["--fill-local-h", "--fill-exact"]).fill_local_h is True
  assert main.parse_args(["--fill-local-h", "--fill-exact"]).fill_exact is True
  assert main.parse_args(["--fill-skeleton-first"]).fill_skeleton_first is True
  assert main.parse_args(["--fill-scope-widening"]).fill_scope_widening is True
  assert main.parse_args(["--no-fill-scope-widening"]).fill_scope_widening is False
  assert main.parse_args(["--fill-touch-all"]).fill_touch_all is True
  assert main.parse_args(["--no-fill-touch-all"]).fill_touch_all is False
  assert main.parse_args(["--mixed-wrench-h"]).mixed_wrench_h is True
  assert main.parse_args(["--no-wrench-aware-h"]).wrench_aware_h is False


def test_saved_plan_persists_resolved_search_semantics(tmp_path):
  """A named-profile result must remain reproducible after the profile itself evolves."""
  import json

  a = main.parse_args(["--search-profile", "standard"])
  cfg = RunConfig.from_namespace(a)
  out = tmp_path / "plan.json"
  main.save_plan(str(out), cfg, [])
  saved = json.loads(out.read_text())["config"]
  for key in ("weight", "milestones", "goal_on_table", "milestone_oracle", "fill_local_h", "fill_exact",
              "milestone_boundaries", "fill_scope_widening", "open_witness",
              "open_witness_batch", "open_witness_epochs", "analytic_seed_cap",
              "fail_order", "empty_seed_refutes"):
    assert saved[key] == getattr(cfg, key)
  loaded = RunConfig.from_file(str(out))
  assert loaded.milestone_boundaries == 4
  assert loaded.milestone_oracle == "structural" and loaded.open_witness


def test_standard_motion_profile_pins_the_motion_contract():
  a = main.parse_args(["--motion-profile", "standard"])
  assert (a.pre_pick, a.pre_place, a.pre_add_grasp,
          a.pre_insertion, a.post_release) == (0.05,) * 5
  assert a.fuse_cograsp and a.fuse_pre_skill
  assert a.mp_sequence and a.mp_optimize
  assert a.mp_runtime == 300.0 and a.mp_stall_timeout is None
  assert a.mp_plan_candidates == 4 and a.mp_plan_epochs == 6
  assert a.mp_plan_initial_runtime == 15.0
  assert not a.mp_no_shortcut and a.mp_shortcut_iters == 100
  assert a.mp_shortcut_strategy == "native" and a.mp_shortcut_res is None
  assert a.mp_collision_tolerance == 1e-4
  assert a.mp_robot_margin == 0.1 and a.mp_margin_ramp == 0.1
  assert a.mp_table_margin == 0.0 and a.mp_cograsp
  assert not a.motion_plan                         # profiles configure; verbs remain explicit


def test_saved_plan_persists_resolved_motion_profile_semantics(tmp_path):
  import json

  cfg = RunConfig.from_namespace(main.parse_args(["--motion-profile", "standard"]))
  out = tmp_path / "motion-profile-plan.json"
  main.save_plan(str(out), cfg, [])
  saved = json.loads(out.read_text())["config"]
  for key in ("fuse_cograsp", "fuse_pre_skill", "mp_sequence", "mp_optimize", "mp_runtime",
              "mp_stall_timeout", "mp_no_shortcut", "mp_shortcut_iters",
              "mp_shortcut_strategy", "mp_shortcut_res", "mp_shortcut_pair_checks",
              "mp_collision_tolerance", "mp_robot_margin", "mp_margin_ramp",
              "mp_table_margin", "mp_cograsp", "mp_plan_candidates", "mp_plan_epochs",
              "mp_plan_initial_runtime"):
    assert saved[key] == getattr(cfg, key)
  loaded = RunConfig.from_file(str(out))
  assert loaded.mp_optimize and loaded.mp_runtime == 300.0
  assert loaded.mp_stall_timeout is None
  assert loaded.mp_plan_candidates == 4 and loaded.mp_plan_epochs == 6


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
  monkeypatch.setattr(main, "_milestone_candidate_session",
                      lambda *_args, **_kwargs: iter((plan_a, plan_b)))
  monkeypatch.setattr(solver, "hoist_releases", lambda *_args, **_kwargs: list(_args[2]))
  monkeypatch.setattr(solver, "collect_plan_witnesses", lambda *_args, **_kwargs: {})
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
      hoist_releases=False, view=False, quiet_rai=True,
  )
  result = main._run_open_plan_loop(cfg, object(), object(), ["a1", "a2"], DummyIK())
  assert result["plan_id"]
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
      "--no-grasp-refine",
  ])
  assert a.weight == 5.0 and not a.open_witness and a.analytic_seed_cap == 9
  assert a.witness_margin == 0.002 and a.view and not a.quiet_rai
  assert not a.grasp_sampler and not a.grasp_sampler_scenes and not a.grasp_refine


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


def test_motion_profile_overrides_loaded_mp_defaults(tmp_path):
  RunConfig(mp_collision_tolerance=0.01, mp_robot_margin=0.0,
            mp_margin_ramp=0.0, mp_runtime=3.0, mp_optimize=True).to_file(
                str(tmp_path / "c.json"))
  a = main.parse_args([
      "--config", str(tmp_path / "c.json"), "--motion-profile", "standard"])
  assert a.mp_collision_tolerance == 1e-4
  assert a.mp_robot_margin == 0.1 and a.mp_margin_ramp == 0.1
  assert a.mp_runtime == 300.0 and a.mp_optimize
  assert a.mp_stall_timeout is None
  assert a.mp_plan_candidates == 4 and a.mp_plan_epochs == 6


def test_no_config_path_unchanged():
  """Without --config, parse_args is a single pass and every original dest keeps its built-in
  default -- only the inert config/profile meta fields are added."""
  a = vars(main.parse_args([]))
  for k, v in _parser_defaults().items():
    assert a[k] == v, f"{k} changed on the no-config path: {a[k]!r} != {v!r}"
  assert a.get("config") is None


def test_loaded_plan_is_not_hoisted_twice_without_explicit_request():
  """A saved plan already contains post-passes; its saved true flag is provenance only."""
  fresh = RunConfig(hoist_releases=True)
  loaded = RunConfig(load_plan="plan.json", hoist_releases=True)
  assert main._should_hoist_plan(fresh, set())
  assert not main._should_hoist_plan(loaded, set())
  assert main._should_hoist_plan(loaded, {"hoist_releases"})
  loaded.hoist_releases = False
  assert not main._should_hoist_plan(loaded, {"hoist_releases"})


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
