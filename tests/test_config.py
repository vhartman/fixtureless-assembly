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
          if a.dest not in ("help", "config") and a.default is not argparse.SUPPRESS}


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


def test_no_config_path_unchanged():
  """Without --config, parse_args is a single pass and every original dest keeps its built-in
  default -- `config` is the only added namespace field."""
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
