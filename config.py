"""RunConfig: the single configuration object for a planner run.

Mirrors the CLI argument namespace EXACTLY (same field names, same defaults) so the CLI, JSON
config files, and a library caller all produce the same object. It is a CARRIER: it runs
nothing itself and leaves main()/the solver untouched -- `to_namespace()` yields the plain
argparse.Namespace that main() already consumes, and `from_namespace()` goes the other way.

Defaults deliberately live in TWO places (here and the argparse parser) to avoid a risky
rewrite of the 115-flag parser; tests/test_config.py::test_defaults_match_parser asserts the
two never drift, so this stays a safe mirror rather than a second source of truth that rots.

Usage:
    from config import RunConfig
    import main
    cfg = RunConfig(assembly="chair", parts=8, num_robots=3, milestones=True, solve=True)
    main.run(cfg)                                 # run it (library entry -- no argparse)
    cfg.to_file("myrun.json")                     # persist the full config
    cfg2 = RunConfig.from_file("myrun.json")      # reload

CLI:  python main.py --config myrun.json --seed 3    (file is the base; --seed 3 overrides).
"""
import argparse
import json
from dataclasses import dataclass, asdict, fields
from typing import Optional


@dataclass
class RunConfig:
  # NOTE: field names + defaults MUST match main.build_parser() exactly (drift-tested).
  # `config` (the file pointer) is intentionally NOT a field -- a config file has no self-ref.

  # -- scene & problem --
  assembly: str = "cross"
  parts: int = 7
  num_robots: int = 4
  floating_ee: bool = False
  vacuum: bool = False
  grippers: Optional[str] = None
  robot: str = "ur10"
  robot_types: Optional[str] = None
  perturb_start_pose: bool = False
  plain_box: bool = False
  part_float: float = 0.015
  random_init: Optional[float] = None
  scatter_spacing: float = 1.0 / 7.0
  grid_init: Optional[int] = None
  grid_gap: float = 0.1
  min_sep: Optional[float] = None
  check_init: bool = False
  seed: int = 0
  goal_on_table: bool = True

  # -- solver / engine --
  solve: bool = False
  optimistic: bool = False
  milestones: bool = False
  lazy: bool = False
  anytime: Optional[float] = None
  milestones_anytime: Optional[float] = None
  ladder: Optional[float] = None
  ladder_lns: Optional[float] = None
  bounded: bool = False
  dfbnb: bool = False
  pea: bool = False
  beam: Optional[int] = None
  iter_bound: bool = False
  weight: float = 1.0

  # -- search tuning --
  canonical: bool = False
  labeled_h: bool = False
  collapsed_h: bool = False
  grasp_collapsed_h: bool = False
  sig_collapsed_h: bool = False
  sig_components: Optional[str] = None
  corridor_h: bool = False
  wrench_aware_h: bool = True
  geo_facts: bool = False
  por: bool = False
  pathmax: bool = False
  part_order_bias: bool = False
  dominance: bool = False
  useful_grasps: bool = False
  anon_robots: bool = False
  lru_robots: bool = False
  place_cells: int = 0
  prefer_center: bool = False
  fill_window: int = 2
  lazy_fills: bool = True
  lns_segments: bool = False
  handovers: bool = True
  allow_place: bool = True
  inhand_reorient: bool = False
  declutter: bool = False

  # -- physics / wrench model --
  wrench: bool = False
  grasp_model: str = "isotropic"
  vacuum_scale: Optional[float] = None
  tf_close_torque: Optional[float] = None
  curved_torque_scale: Optional[float] = None
  grasp_compliance: Optional[float] = None
  insertion_noise: Optional[float] = None
  insertion_offset: float = 0.0
  insert_force: float = 10.0
  ignore_gravity: bool = False
  robust_mode: str = "corners"
  inserter_check: bool = True
  bracing: bool = False
  brace_resting: bool = False
  brace_for_assembly_only: bool = False
  staged_mates: bool = False
  mixed_wrench_h: bool = False
  allow_curved_grasps: bool = False
  dense_grasps: bool = False

  # -- IK & feasibility --
  real_ik: bool = False
  ik_seeding: bool = True
  ik_collision_gate: float = 0.015
  full_ik: bool = False
  strict_collisions: Optional[float] = None

  # -- witness optimization --
  optimize_witnesses: Optional[str] = None
  witness_clearance: float = 0.0
  witness_robot_collisions: bool = False
  fuse_cograsp: bool = False

  # -- skills / pre-insertion splits --
  skill: float = 0.0
  pre_insertion: float = 0.0
  pre_pick: float = 0.0
  pre_place: float = 0.0
  pre_add_grasp: float = 0.0
  fuse_pre_skill: bool = False

  # -- motion-plan export (mr_bench) --
  motion_plan: bool = False
  mp_sequence: bool = False
  mp_optimize: bool = False
  mp_runtime: float = 60.0
  mp_no_shortcut: bool = False
  mp_shortcut_iters: int = 1000
  mp_shortcut_res: Optional[float] = None
  mp_clearance: bool = False
  mp_cograsp: bool = False
  mp_save: Optional[str] = None
  mp_collision_tolerance: float = 0.01
  mp_viser: bool = False
  mp_collision_view: bool = False
  mp_modes: bool = False

  # -- visualization --
  view: bool = True
  disp_final: bool = False
  show_init: bool = False
  viz_grasps: bool = False
  viz_rest_poses: bool = False
  mate_sequence_viz: bool = False
  show_successors: bool = False
  viser: bool = False
  viser_port: int = 8080
  view_cheap_checks: bool = False

  # -- diagnostics & IK tests --
  test_pick_ik: Optional[str] = None
  test_assemble_ik: Optional[str] = None
  test_assemble_held_ik: Optional[str] = None
  test_place_ik: Optional[str] = None
  search_stats: Optional[float] = None
  log_ik_failures: Optional[float] = None
  fill_touch_all: bool = False

  # -- output & persistence --
  utilization: bool = False
  hoist_releases: bool = False
  save: Optional[str] = None

  # ---- conversions ----
  @classmethod
  def from_namespace(cls, ns):
    """Build from an argparse.Namespace, ignoring extra keys (e.g. the meta `config`)."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in vars(ns).items() if k in known})

  def to_namespace(self):
    """The plain argparse.Namespace main() consumes. `config` is added as None so the
    namespace is a superset-compatible stand-in for a parsed one."""
    return argparse.Namespace(config=None, **asdict(self))

  def to_dict(self):
    return asdict(self)

  @classmethod
  def from_dict(cls, d):
    """Build from a plain dict; rejects unknown keys (typos in a hand-written file) but
    tolerates a stray `config` key."""
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known - {"config"}
    if unknown:
      raise ValueError(f"RunConfig: unknown key(s) {sorted(unknown)}")
    return cls(**{k: v for k, v in d.items() if k in known})

  # ---- persistence (JSON; yaml not a project dep). NB: named to_file/from_file, not
  # save/load, because `save` is itself a config FIELD (the --save CLI dest) and a method of
  # the same name would shadow it. ----
  def to_file(self, path):
    with open(path, "w") as f:
      json.dump(self.to_dict(), f, indent=2, sort_keys=True)
      f.write("\n")

  @classmethod
  def from_file(cls, path):
    with open(path) as f:
      return cls.from_dict(json.load(f))
