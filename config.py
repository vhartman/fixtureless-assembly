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
import dataclasses
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
  part_float: float = 0.002
  random_init: Optional[float] = None
  scatter_spacing: float = 1.0 / 7.0
  center_scatter: bool = False
  grid_init: Optional[int] = None
  grid_gap: float = 0.1
  min_sep: Optional[float] = None
  check_init: bool = False
  seed: int = 0
  goal_on_table: bool = True

  # -- solver / engine --
  solve: bool = False
  optimistic: bool = False
  optimistic_fallback: bool = True
  optimistic_chain_h: bool = False
  milestones: bool = True
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
  sig_collapsed_h: bool = True
  sig_components: Optional[str] = "press"
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
  milestone_oracle: str = "global"
  fill_local_h: bool = False
  fill_exact: bool = False
  fill_policy_first: bool = False
  fill_policy_guide: bool = False
  fill_skeleton_first: bool = False
  milestone_boundaries: int = 1
  lazy_fills: bool = True
  open_witness: bool = False
  open_witness_epochs: int = 2
  open_witness_batch: int = 7
  lazy_edge_selector: str = "forward"
  lns_segments: bool = False
  handovers: bool = True
  allow_place: bool = True
  inhand_reorient: bool = False
  declutter: bool = False

  # -- physics / wrench model --
  wrench: bool = True
  grasp_model: str = "asymmetric"
  table_mu: Optional[float] = None
  vacuum_scale: Optional[float] = None
  two_finger_scale: Optional[float] = None
  vacuum_engage_force: float = 0.0
  vacuum_engage_friction: Optional[float] = None
  tf_close_torque: Optional[float] = None
  curved_torque_scale: Optional[float] = None
  grasp_compliance: Optional[float] = None
  insertion_noise: Optional[float] = None
  insertion_offset: float = 0.0
  insert_force: float = 10.0
  ignore_gravity: bool = False
  robust_mode: str = "corners"
  inserter_check: bool = True
  bracing: bool = True
  brace_resting: bool = False
  brace_for_assembly_only: bool = False
  vacuum_no_assemble: bool = False
  staged_mates: bool = False
  mixed_wrench_h: bool = False
  allow_curved_grasps: bool = False
  dense_grasps: bool = False

  # -- IK & feasibility --
  real_ik: bool = True
  ik_seeding: bool = True
  ik_positive_reuse: bool = False
  ik_restarts: int = 12
  precheck_restarts: int = 12
  fail_order: bool = False
  analytic_seed_cap: Optional[int] = None
  ssik_seed_neighborhood: float = 0.0
  ssik_neighborhood_cap: int = 6
  empty_seed_refutes: bool = False
  ik_collision_gate: float = 0.015
  full_ik: bool = False
  strict_collisions: Optional[float] = 1e-6
  strict_mate_collisions: bool = False
  always_contact: bool = True
  ik_accuracy: Optional[float] = None
  mate_accuracy: Optional[float] = None
  grasp_accuracy: Optional[float] = None
  grasp_eq_weight: Optional[float] = None
  mate_eq_weight: Optional[float] = 100.0
  place_eq_weight: Optional[float] = 100.0
  grasp_translations: int = 0
  grasp_sampler: bool = False
  grasp_sampler_keep: int = 20
  grasp_sampler_candidates: int = 200
  grasp_sampler_seed: int = 0
  grasp_sampler_version: int = 2
  grasp_sampler_db_hash: Optional[str] = None
  grasp_refine: bool = False
  grasp_widen: bool = False
  grasp_widen_batch: int = 8
  grasp_widen_max: int = 96
  grasp_local_widen: bool = False
  grasp_local_caps: Optional[dict] = None
  grasp_sampler_margin: float = 0.008
  grasp_sampler_pad_support: float = 0.75
  grasp_sampler_wrap: Optional[float] = None
  grasp_sampler_scenes: bool = False
  stable_rest_poses: bool = False
  placement_yaws: int = 1
  tight_fingers: bool = True
  openarm_real_finger_radius_scale: float = 1.0
  grasp_closure_check: bool = False
  quiet_rai: bool = False

  # -- witness optimization --
  optimize_witnesses: Optional[str] = "floor"
  witness_clearance: float = 0.0
  witness_draw: Optional[int] = None
  witness_sets: int = 4
  witness_set_spread: float = 0.05
  ban_actions: Optional[str] = None
  witness_margin: Optional[float] = 0.005
  witness_joint_margin: Optional[float] = None
  witness_branch_budget: int = 5
  witness_branch_metric: str = "sign"
  witness_reconfig_thresh: float = 1.5707963267948966
  witness_robot_collisions: bool = True
  fuse_cograsp: bool = True

  # -- skills / pre-insertion splits --
  skill: float = 0.0
  pre_insertion: float = 0.05
  pre_pick: float = 0.05
  pre_place: float = 0.05
  pre_add_grasp: float = 0.05
  post_release: float = 0.05
  fuse_pre_skill: Optional[bool] = None   # None = auto (ON when a pre-insertion/
                                          # pre-add-grasp split is present)

  # -- motion-plan export (mr_bench) --
  motion_plan: bool = False
  mp_sequence: bool = True
  mp_optimize: bool = True
  mp_runtime: float = 30.0
  mp_stall_timeout: Optional[float] = None
  mp_no_shortcut: bool = False
  mp_shortcut_iters: int = 1000
  mp_shortcut_res: Optional[float] = None
  mp_shortcut_strategy: str = "native"
  mp_shortcut_pair_checks: int = 200
  mp_clearance: bool = False
  mp_cograsp: bool = False
  mp_save: Optional[str] = None
  mp_export_save: Optional[str] = None
  mp_export_load: Optional[str] = None
  mp_collision_tolerance: float = 1e-4
  mp_robot_margin: float = 0.1
  mp_margin_ramp: float = 0.1
  mp_table_margin: float = 0.0
  mp_seed: Optional[int] = None
  mp_fake_clock: bool = False
  mp_open_plans: bool = False
  mp_plan_candidates: int = 4
  mp_plan_epochs: int = 6
  mp_plan_initial_runtime: float = 15.0
  mp_viser: bool = False
  mp_collision_view: bool = False
  mp_modes: bool = False

  # -- visualization --
  view: bool = True
  disp_final: bool = False
  show_init: bool = False
  viz_grasps: bool = False
  viz_initial_grasps: Optional[str] = None
  viz_grasp_closure: Optional[str] = None
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
  diagnose_plan_rank: Optional[str] = None
  fill_scope_widening: bool = False
  fill_touch_all: bool = False
  placement_yaw_widening: bool = False

  # -- output & persistence --
  utilization: bool = False
  hoist_releases: bool = True
  load_plan: Optional[str] = None
  refresh_witnesses: bool = False
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

  @classmethod
  def default_for(cls, key):
    """The built-in default of one field -- the SINGLE source of defaults for saved-plan
    rebuilds. replay.py/sim.py resolve missing config keys through this (via saved_cfg)
    instead of hand-copied literals, so the parser, RunConfig, and the rebuild paths can
    never drift apart."""
    for f in fields(cls):
      if f.name == key:
        return None if f.default is dataclasses.MISSING else f.default
    raise KeyError(f"RunConfig has no field {key!r}")

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
      d = json.load(f)
    # A saved PLAN file (plan_io: {"config": {...}, "plan": [...]}) carries its config as a
    # nested block. Accept it directly so `--config plan.json --load-plan plan.json` (the
    # documented re-export idiom) works without hand-extracting the block.
    is_saved_plan = isinstance(d.get("config"), dict) and "plan" in d
    if is_saved_plan:
      d = d["config"]
      # New robustness defaults change search feasibility and exported task structure. A plan
      # predating a field must replay with the historical OFF value, never acquire new tasks or
      # a stronger IK gate merely because today's default changed.
      legacy = {"witness_margin": None, "witness_sets": 0,
                "pre_pick": 0.0, "pre_place": 0.0, "pre_add_grasp": 0.0,
                "post_release": 0.0}
      d = dict(d)
      # Persistence-only provenance, deliberately not a RunConfig/CLI field.  It certifies
      # the clearance gate that was active DURING SEARCH (as opposed to the historical
      # witness-only --witness-margin).  main.py reads it from the raw plan before loading;
      # RunConfig must ignore it so a saved plan remains usable as --config.
      d.pop("_search_witness_margin", None)
      for key, value in legacy.items():
        d.setdefault(key, value)
    # Before sampler V2 existed, every JSON carrying `grasp_sampler: true` meant the old
    # replacement database. Treat an absent version as V1 for both saved plans and standalone
    # historical config files; otherwise positional grasp ids silently change meaning.
    if d.get("grasp_sampler") and "grasp_sampler_version" not in d:
      d = dict(d)
      d["grasp_sampler_version"] = 1
    return cls.from_dict(d)


def saved_cfg(cfg, key, legacy=None):
  """Read one entry of a SAVED plan's config block. The saved value wins whenever the key is
  present; a MISSING key means the plan predates that flag, and the value is the RunConfig
  default (via RunConfig.default_for) -- the one place defaults live. `legacy` overrides that
  ONLY for keys whose solve-time default differed from today's default (each call site names
  the historical value and why), so old plans replay against the geometry/action space they
  were actually solved in, not today's defaults."""
  if key in cfg:
    return cfg[key]
  return RunConfig.default_for(key) if legacy is None else legacy
