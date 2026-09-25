import argparse
import os as _os
import sys as _sys
import time as _time

# BEFORE rai loads: RAI_QUIET=1 drops rai's libccd/pairCollision spam at the fd (see rai_quiet).
# Env var as well as a flag so it can be exported once instead of typed on every run.
if _os.environ.get("RAI_QUIET", "").strip() not in ("", "0", "false", "no"):
  import rai_quiet as _rq
  _rq.install()

import numpy as np

from env import *
from assembly import *
from config import RunConfig

# Named CLI profiles are deliberately a thin layer over the ordinary flags.  RunConfig stays
# the fully-resolved configuration (there is no profile magic in the solver/library API), and
# every value below can be overridden by an explicit CLI flag. Selecting a profile is itself
# an explicit request, so its named fields override a loaded config while unnamed scene fields
# still come from that file.
_STANDARD_SEARCH_PROFILE = {
    # The local PDB is used greedily for time-to-first validated assembly.  The global
    # weight remains one: --milestones-anytime can therefore hand the incumbent to its
    # separate exact improvement phase without inheriting the local feasibility ordering.
    "weight": 1.0,
    "milestones": True,
    "milestone_oracle": "structural",
    "fill_local_h": True,
    "fill_exact": False,
    "fill_greedy": True,
    # Experimental CEGAR precursor: a strict shortest-PDB policy is often spurious under
    # fixed-base IK (OpenArm Car-mini needs an abstract-cost +5 presentation detour).  Keep
    # the probe available, but do not make a failed strict concretization part of standard
    # until abstract counterexamples can be repaired instead of merely falling through.
    "fill_policy_first": False,
    # Post-failure reverse-policy column selection is an ablation until it has a broader
    # benchmark win.  The ordinary compact fill itself stays byte-for-byte unchanged.
    "fill_policy_guide": False,
    "fill_skeleton_first": False,
    "fill_action_widening": False,
    "milestone_boundaries": 4,
    "fill_scope_widening": True,
    "fill_touch_all": False,
    # The historical scatter is a fixed x=-0.5 one-row line. That made an otherwise valid
    # benchmark depend on arm length (opposite car-mini parts had 0 exact OpenArm branches)
    # rather than the declared density. The compact centred lattice changes only initialization.
    "center_scatter": True,
    "sig_collapsed_h": True,
    "sig_components": "press",
    # Raw-grasp guidance complementary to the uniformly projected local PDB: prefer grasps
    # that are both present at the mate geometry and can transmit this part's insertion load.
    # It is ordering plus an admissible forced-regrasp cost only; no action is removed.
    "useful_grasps": True,
    "real_ik": True,
    "wrench": True,
    "grasp_model": "asymmetric",
    "optimize_witnesses": "floor",
    "witness_margin": 0.005,
    # A collision-free endpoint exactly on a hardware stop has no open neighbourhood for a
    # sampling planner. OpenArm car-mini hit joint 7 with 3e-9 rad slack at its four-arm seat.
    "witness_joint_margin": 0.02,
    # Preserve the established clearance-floor epsilon-constraint, but do not buy clearance
    # with additional physical reconfigurations. Count actual bounded-joint travel rather than
    # the old UR-only elbow/wrist sign label (which missed OpenArm shoulder flips).
    "witness_branch_budget": 0,
    "witness_branch_metric": "motion",
    "witness_sets": 4,
    "tight_fingers": True,
    "strict_collisions": 1e-6,
    # Search witnesses must expose the same scene downstream planning validates. These fixed
    # distinct silent-invalid-plan classes: idle-arm intersections, hidden touched-part
    # contact, and millimetre-scale place/mate residuals.
    "witness_robot_collisions": True,
    "always_contact": True,
    "mate_eq_weight": 100.0,
    "place_eq_weight": 100.0,
    "ik_seeding": True,
    # Canonicalize physically identical pre/post IK programs. Exact witnesses and stochastic
    # effort cursors are shared; different obstacle scenes never exchange a verdict.
    "ik_positive_reuse": True,
    # Search/action semantics used by the current hands-off assembly artifacts.
    "bracing": True,
    "handovers": True,
    "allow_place": True,
    "goal_on_table": True,
    # Authored mate alternatives are exact physical symmetries, not heuristic poses or
    # additional symbolic actions. Assemblies without a declared set are unchanged; those
    # with one let the same assemble edge choose whichever equivalent realization is easiest
    # for the arms.
    "mate_symmetry": True,
    "lazy_fills": True,
    "hoist_releases": True,
    # Shared planner/simulator table datum. Keep this small: every millimetre is certified
    # free space that the simulator turns into raised table geometry.
    "part_float": 0.002,
    # A pre-insertion endpoint pair is not yet a skill certificate: the held gripper may
    # intersect the socket between two individually clear endpoints. Sample that short local
    # continuation at the same 0.002-rad resolution used by certified MP shortcuts.
    "skill_funnel_collision_res": 0.002,
    # A finite IK miss remains reopenable instead of becoming a false infeasibility verdict.
    # Two epochs are the measured operational budget, not a continuous completeness proof.
    "open_witness": True,
    "open_witness_batch": 7,
    "open_witness_epochs": 2,
    # LazySP edge selection only. Keep the measured prefix-first baseline in standard:
    # suffix-first reduced validation count on UR5 but made OpenArm's costly mate bundles
    # dominate (5537 vs 3490 binds on Car-mini obj_1). Alternatives remain explicit ablations.
    "lazy_edge_selector": "forward",
    "analytic_seed_cap": None,
    # The scene-reserved hybrid sampler keeps a conventional/sample alphabet and reserves a
    # scene-positive PICK/MATE/component-support prefix before the ordinary hybrid order.
    # The intermediate-component roles matter: a grasp may clear the current rigid component
    # but collide with a future part, so singleton/final-product labels alone can discard the
    # only useful transport column. Exact state gates stay authoritative.
    "grasp_sampler": True,
    # Keep ordinary scenes cheap. A failed complete fill monotonically exposes more of the
    # immutable master database, so K=20 is an initial alphabet rather than a pruning decision.
    # Imported CAD scenes can opt into the dense fixed alphabet through `fabrica`.
    "grasp_sampler_keep": 20,
    "grasp_sampler_candidates": 96,
    # The finite master includes the nested 45-degree pinch rolls; search still starts
    # with the small active prefix and exposes further IDs only when needed.
    "grasp_sampler_roll_refine": True,
    "grasp_sampler_scenes": True,
    # A fixed-open hand is not a grasp model. Every two-finger edge must carry a bilateral
    # virtual-close certificate and an object-dependent aperture into the saved artifact.
    "grasp_closure_check": True,
    # Multi-part components inherit candidate contacts from all constituent parts. Build
    # those tables only when first queried and expose a stable constituent-balanced prefix.
    # Exhaustion monotonically reopens the exact historical full union, so this is an anytime
    # ordering layer rather than an infeasibility claim.
    "component_grasp_keep": 32,
    # The compliant two-finger pads can wrap ordinary curved surfaces. Besides admitting
    # those contacts, this emits the complementary side-approach roll that tight insertions
    # often need. Vacuum curvature remains governed separately by --allow-curved-grasps.
    "grasp_sampler_wrap": 0.015,
    # A horizontal table leaves world-z placement yaw physically free.  Keep four useful
    # quarter-turn representatives in the finite action space, but expose them per failed fill
    # (1 -> 2 -> 4) so ordinary problems pay exactly the historical N=1 search cost.
    "placement_yaws": 4,
    "placement_yaw_widening": True,
    # Full hybrid grasps are the robust default. On centred OpenArm car-mini, compact-only
    # obj_2 exhausted 6280 effort slots; the full DB solved the same fill in 13 slots/0.8s.
    # The optional staged policy remains available as --grasp-refine for ablation/anytime use.
    "grasp_refine": False,
    "grasp_widen": True,
    "grasp_widen_batch": 8,
    "grasp_widen_max": 96,
    # A standard search is a batch artifact-producing run.  Both remain explicitly
    # reversible on the CLI via --view and --no-quiet-rai.
    "quiet_rai": True,
    "view": False,
}

_SEARCH_PROFILES = {
    "standard": dict(_STANDARD_SEARCH_PROFILE),
    # Imported CAD models need denser contact coverage. Keep that expense out of ordinary
    # scenes and make the different finite action alphabet explicit in saved provenance.
    "fabrica": {
        **_STANDARD_SEARCH_PROFILE,
        "grasp_sampler_keep": 64,
        # Preserve its historical K=64 cost by default, but a single
        # --grasp-widen-max N now opts into a larger fixed master.
        "grasp_widen_max": 64,
    },
}


# The planner still contains research ablations and diagnostics that are useful through a
# RunConfig or an old reproducibility command, but they are not supported top-level workflow
# choices.  Keeping all of them in ``--help`` had grown the public CLI to more than 2,200 lines
# and made obsolete experiments look equivalent to the two maintained profiles.  They remain
# parseable for artifact compatibility; new commands should use the compact public surface or a
# checked-in config/benchmark manifest.  Delete the underlying branches only together with their
# dedicated regression data, not opportunistically while cleaning the interface.
_INTERNAL_MAIN_CLI = {
    # One-off scene/model diagnostics.
    "robot_types", "openarm_real_finger_radius_scale", "perturb_start_pose", "plain_box",
    "check_init", "test_pick_ik", "test_assemble_ik", "test_assemble_held_ik",
    "test_place_ik", "show_init", "disp_final", "mate_sequence_viz", "show_successors",
    "view_cheap_checks", "diagnose_plan_rank", "log_ik_failures", "search_stats",
    # Superseded global search engines and heuristic experiments.  The submitted comparison
    # rows (optimistic, flat lazy, milestones/counting, exact fill, fine projection) stay public.
    "canonical", "optimistic_fallback", "optimistic_chain_h", "full_backward",
    "full_backward_max_states", "full_backward_max_edges", "bounded", "anytime", "ladder",
    "ladder_lns", "dfbnb", "pea", "beam", "iter_bound", "labeled_h", "collapsed_h",
    "grasp_collapsed_h", "corridor_h", "geo_facts", "por", "pathmax", "part_order_bias",
    "dominance", "anon_robots", "lru_robots", "declutter", "lns_segments",
    "lns_grasp_transition_h", "fill_incumbent_first", "fill_policy_first",
    "fill_policy_guide", "fill_skeleton_first", "fill_action_widening",
    # Standard-profile implementation details. They stay available to benchmark manifests,
    # but a normal run selects the coherent bundle instead of reconstructing it flag by flag.
    "placement_yaw_widening", "place_cells", "prefer_center", "wrench_aware_h",
    "lazy_fills", "open_witness", "open_witness_epochs", "open_witness_batch",
    "open_witness_positive_order",
    "lazy_edge_selector", "fill_window", "lns_window", "milestone_oracle",
    "fill_local_h", "fill_raw_grasp_h", "fill_target_support_quotient", "fill_exact",
    "fill_greedy", "milestone_boundaries", "ik_accuracy", "tight_fingers",
    "place_eq_weight", "quiet_rai", "mate_eq_weight", "mate_accuracy",
    "mate_angular_accuracy", "grasp_eq_weight", "grasp_accuracy", "witness_branch_metric",
    "useful_grasps", "fill_scope_widening", "fill_touch_all", "ban_actions",
    "sig_collapsed_h", "sig_components",
    # IK/grasp search experiments that are not selected by standard or fabrica.
    "assemble_ssik_pose_seeds", "assemble_ssik_pose_projection", "assemble_unbiased_first",
    "assemble_basin_racing", "assemble_race_probe_evals", "assemble_race_promote",
    "handover_basin_racing", "handover_race_probe_evals", "handover_race_starts",
    "handover_race_promote", "ik_restarts", "precheck_restarts", "fail_order",
    "empty_seed_refutes", "ssik_seed_neighborhood", "ssik_neighborhood_cap", "full_ik",
    "dense_grasps", "grasp_refine", "grasp_local_widen", "grasp_sampler_strict",
    "grasp_sampler_seed", "ik_collision_gate", "grasp_translations",
    "ik_positive_reuse", "analytic_seed_cap", "grasp_sampler", "grasp_sampler_keep",
    "grasp_widen", "grasp_widen_batch", "grasp_widen_max", "component_grasp_keep",
    "grasp_sampler_candidates", "grasp_sampler_full_probe", "grasp_sampler_roll_refine",
    "grasp_sampler_pad_support", "grasp_sampler_wrap",
    "grasp_sampler_scenes", "always_contact",
    # Narrow physics/witness ablations and manual optimizer controls.
    "orientation_robust_wrench", "curved_torque_scale", "insertion_offset", "robust_mode",
    "inserter_check", "brace_resting", "brace_for_assembly_only", "vacuum_no_assemble",
    "staged_mates", "strict_mate_collisions", "witness_draw", "witness_reconfig_thresh",
    "inhand_reorient", "utilization", "witness_clearance", "witness_set_spread",
    "witness_joint_margin", "witness_branch_budget", "witness_robot_collisions",
    # Artifact forensics and viewers.  Normal motion export is --motion-plan/--mp-save;
    # normal inspection is replay.py or --mp-viser.
    "mp_clearance", "mp_cograsp", "mp_export_save", "mp_export_load", "mp_modes",
    "mp_fake_clock", "viz_grasps", "viz_grasp_closure", "viz_rest_poses",
    "viz_rest_poses_grid", "mp_stall_timeout", "mp_open_plans", "mp_plan_candidates",
    "mp_plan_epochs", "mp_plan_initial_runtime", "mp_shortcut_iters",
    "mp_shortcut_strategy", "mp_shortcut_pair_checks", "mp_shortcut_res",
    "mp_collision_view", "mp_robot_margin", "mp_margin_ramp", "mp_margin_mode",
    "mp_table_margin", "mp_wrench_gate",
}


def _hide_internal_cli(parser, dests):
  """Remove internal/reproducibility switches from normal argparse help and usage."""
  found = {action.dest for action in parser._actions}
  missing = set(dests) - found
  if missing:
    raise AssertionError(f"unknown internal CLI destinations: {sorted(missing)}")
  for action in parser._actions:
    if action.dest in dests:
      action.help = argparse.SUPPRESS

_STANDARD_MOTION_PROFILE = {
    # Complete both sides of every contact skill and preserve the coupled endpoints. Tight
    # socket scenes may explicitly disable individual auxiliary funnels after the profile.
    "pre_pick": 0.05,
    "post_pick": 0.05,
    "pre_place": 0.05,
    "pre_add_grasp": 0.05,
    "pre_insertion": 0.05,
    "post_release": 0.05,
    "fuse_cograsp": True,
    "fuse_pre_skill": True,
    # The simulator consumes an ordered mode chain. A production artifact is always optimized:
    # after the first BiRRT solution, RRT* keeps improving it for the rest of the consultation
    # budget, then the certified shortcut pass runs. The 300 s cap covers the historically hard
    # Chair transit; no 30 s frontier-stall shortcut may silently truncate that budget.
    "mp_sequence": True,
    "mp_optimize": True,
    "mp_runtime": 300.0,
    "mp_stall_timeout": None,
    # Fair complete-plan loop: four alternatives match the search's retained milestone
    # boundaries. 15,30,60,120,240,300 s gives every suspended plan both quick probes and the
    # full historically sufficient consultation without one plan monopolizing the beginning.
    "mp_plan_candidates": 4,
    "mp_plan_epochs": 6,
    "mp_plan_initial_runtime": 15.0,
    "mp_shortcut": True,
    "mp_shortcut_iters": 100,
    "mp_shortcut_strategy": "native",
    # OpenArm's thin finger/object passages exposed a real swept collision inside a 0.01-rad
    # edge (the endpoint and neighbouring coarse samples were free).  Use one resolution for
    # BiRRT edges, accepted shortcuts, and simulator preflight; a plan cannot become invalid
    # merely because the next stage samples the same edge more finely.
    "mp_shortcut_res": 0.002,
    # The portable certificate is collision-free motion plus hardware joint limits.  A uniform
    # positive inter-robot shell is not generally feasible: even though the task-participant
    # ramp reaches zero at its anchors, 10cm can disconnect otherwise broad transit spaces in
    # compact workcells (Fabrica Beam), so the robot shell remains zero.  The 2mm carried-part
    # table margin is different: it exactly covers the standard search ``part_float`` datum
    # between the planner's lower table and the physical MuJoCo surface.  The 5cm endpoint ramp
    # preserves contact-feasible picks/places while preventing a nominally certified transit
    # from grazing the real table during a payload reorientation.
    "mp_collision_tolerance": 1e-4,
    "mp_robot_margin": 0.0,
    "mp_margin_ramp": 0.05,
    "mp_margin_mode": "linear",
    "mp_table_margin": 0.002,
    "mp_wrench_gate": False,
    "mp_cograsp": True,
    "quiet_rai": True,
    "view": False,
}

_EXECUTION_MOTION_PROFILE = {
    **_STANDARD_MOTION_PROFILE,
    # Explicit robustness profile: demand a 5cm inter-robot shell in unconstrained transit,
    # but remove it in a 5cm neighbourhood of either segment keyframe, where handovers,
    # co-grasps, and compact assembly postures can require deliberate proximity.
    "mp_robot_margin": 0.05,
    "mp_margin_ramp": 0.05,
    "mp_margin_mode": "binary",
}

_MOTION_PROFILES = {
    "standard": dict(_STANDARD_MOTION_PROFILE),
    "execution": dict(_EXECUTION_MOTION_PROFILE),
}

def _rss_mb():
  # A1/A6 instrumentation probe: peak RSS of THIS process in MB (ru_maxrss is KiB on Linux).
  import resource
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _unpickable_required_parts(assembly, report):
  """Non-root parts lacking the necessary isolated acquisition certificate."""
  return [p for p, result in report.items()
          if p != assembly.base and not result["ok"]]


def _sampled_scene_coverage(assembly, checker, robots, prefix="[grasp-scenes]"):
  """Report the finite sampler's pick/seat coverage without running IK.

  This diagnostic is also emitted while the benchmark's immutable shared grasp table is
  built. It makes an accidentally impoverished database visible *before* every algorithm
  spends its search budget on it. ``both=0`` is deliberately only a warning: a regrasp or
  handover may legitimately connect disjoint pick- and assembly-clear grasp sets.
  """
  from problem import sampled_grasp_meta, sampled_grasp_selection_diagnostics
  kinds = sorted(set(checker._robot_kind.get(r, r) for r in robots))
  report = {}
  profile_totals = {"raw": 0.0, "master": 0.0, "materialize": 0.0}
  for part in assembly.objects:
    meta = sampled_grasp_meta(assembly.grasp_shape(part), part=part)
    part_report = {}
    fields = []
    for kind in kinds:
      pick = {i for i, row in enumerate(meta)
              if "pick" in (row.get("scene_by_kind", {}).get(kind, ()) if row else ())}
      seat = {i for i, row in enumerate(meta)
              if "assemble" in (row.get("scene_by_kind", {}).get(kind, ()) if row else ())}
      support = {i for i, row in enumerate(meta)
                 if any(r == "support" or r.startswith("component:")
                        for r in (row.get("scene_by_kind", {}).get(kind, ()) if row else ()))}
      values = {"pick": len(pick), "assemble": len(seat), "support": len(support),
                "both": len(pick & seat)}
      part_report[kind] = values
      fields.append(f"{kind}: pick={values['pick']} mate={values['assemble']} "
                    f"support={values['support']} "
                    f"both={values['both']}")
    report[part] = part_report
    print(f"{prefix} {part}: " + ", ".join(fields))
    audit = sampled_grasp_selection_diagnostics(part)
    if audit is not None:
      mode = ("adaptive-full" if audit.get("expanded_full") else
              "compact" if audit["full_probe"] else "capped")
      print(f"{prefix} {part}: selector raw={audit['raw']} quality={audit['quality']} "
            f"probe={audit['probe']} evaluated={audit['evaluated']} "
            f"selected={audit['selected']} ({mode} probe), table-route="
            f"{','.join(f'{k}:g{a}->g{b}@{role}' for k, role, a, b in audit.get('table_routes', ())) or '-'}, "
            f"handover-route={','.join(f'{ka}/{kb}:g{a}->g{b}' for ka, kb, a, b in audit.get('handover_routes', ())) or '-'}")
      rejects = audit.get("acquisition_rejects", {})
      if any(rejects.values()):
        detail = ", ".join(
            f"{kind}:" + "/".join(f"{reason}={count}" for reason, count in reasons.items())
            for kind, reasons in rejects.items() if reasons)
        print(f"{prefix} {part}: isolated closure rejects: {detail}")
      physics_rejects = audit.get("physics_rejects", {})
      if any(physics_rejects.values()):
        detail = ", ".join(
            f"{kind}:" + "/".join(f"{reason}={count}" for reason, count in reasons.items())
            for kind, reasons in physics_rejects.items() if reasons)
        print(f"{prefix} {part}: isolated physics rejects: {detail}")
      for kind, gap_mm in audit.get("mate_gap_best_mm", {}).items():
        if gap_mm is not None:
          print(f"{prefix} {part}: best physical mate gap ({kind})={gap_mm:.2f}mm")
      lost = []
      for kind, available in audit["evaluated_coverage"].items():
        retained = audit["selected_coverage"].get(kind, {})
        for role in ("pick", "assemble", "support", "both"):
          if available.get(role, 0) and not retained.get(role, 0):
            lost.append(f"{kind}:{role}")
      if lost:
        print(f"{prefix} WARNING {part}: selection removed all candidates for available "
              f"certificate(s): {', '.join(lost)}")
      prof = audit.get("profile")
      if prof:
        profile_totals["raw"] += prof["raw_seconds"]
        profile_totals["master"] += prof["master_seconds"]
        profile_totals["materialize"] += prof["materialize_seconds"]
        checks = ", ".join(
            f"{name}={entry['seconds']:.2f}s/{entry['calls']}x"
            for name, entry in sorted(prof["checks"].items(),
                                      key=lambda item: -item[1]["seconds"])) or "none"
        print(f"[grasp-prof] {part}: raw={prof['raw_seconds']:.2f}s "
              f"({prof['raw_source']}), master={prof['master_seconds']:.2f}s "
              f"[checks: {checks}; other={prof['master_other_seconds']:.2f}s], "
              f"materialize={prof['materialize_seconds']:.2f}s")
    if part != assembly.base and all(values["both"] == 0
                                     for values in part_report.values()):
      print(f"{prefix} WARNING {part}: no grasp is clear in both the initial pick and "
            "mandatory mate scenes; any solution requires a regrasp or handover.")
  if any(profile_totals.values()):
    print(f"[grasp-prof] total: raw={profile_totals['raw']:.2f}s, "
          f"master={profile_totals['master']:.2f}s, "
          f"materialize={profile_totals['materialize']:.2f}s")
  return report

def build_parser():
  parser = argparse.ArgumentParser(
      description="Integrated multi-robot assembly task and motion planning."
  )
  parser.add_argument(
      "--config", metavar="FILE", default=None,
      help="Load a RunConfig JSON file as the BASE configuration; any explicit command-line "
           "flag overrides the file's value (CLI wins). Write one with config.RunConfig.save "
           "or by hand. Without --config nothing changes (byte-identical default path).",
  )
  parser.add_argument("--artifact-contract", default=None, help=argparse.SUPPRESS)
  parser.add_argument(
      "--search-profile", choices=tuple(_SEARCH_PROFILES), default=None, metavar="NAME",
      help="Named fresh-search defaults: standard = robust 5mm witness contract + greedy "
           "structural milestone fills with a uniform local PDB + non-pruning useful-grasp "
           "guidance + bounded fair open witnesses; fabrica = standard with a fixed dense "
           "64-grasp alphabet for imported CAD models. A profile is opt-in. Precedence is "
           "explicit flag > profile > --config file > built-ins, and "
           "--parts/grippers/forces/layout/seed/save remain scene-specific.",
  )
  parser.add_argument(
      "--motion-profile", choices=tuple(_MOTION_PROFILES), default=None, metavar="NAME",
      help="Named motion-export/planning defaults. standard pins skill funnels/fusion, "
           "sequence export, 1e-4 collision tolerance, and zero auxiliary margin; execution "
           "is otherwise identical but uses a 5cm robot margin with binary 5cm endpoint "
           "relaxation. Both use RRT* "
           "optimization over a full 300s budget (no frontier-stall cutoff), a 4-plan/6-tier "
           "open-loop budget, and 100 certified shortcuts. It does not imply "
           "--motion-plan or guess --mp-seed/output paths. Precedence matches search profiles.",
  )
  g_scene   = parser.add_argument_group("scene & problem", "assembly, parts, robots, grippers, initial layout, goal")
  g_solver  = parser.add_argument_group("solver / engine", "which search algorithm runs + suboptimality weight")
  g_search  = parser.add_argument_group("search tuning", "heuristics, ordering, pruning, fill windows, action toggles")
  g_phys    = parser.add_argument_group("physics / wrench model", "static-equilibrium stability, grasp caps, bracing, staged mates")
  g_ik      = parser.add_argument_group("IK & feasibility", "real-robot IK in the search feasibility loop")
  g_witness = parser.add_argument_group("witness optimization", "post-solve IK-keyframe smoothing (discrete plan fixed)")
  g_skills  = parser.add_argument_group("skills / pre-insertion splits", "export & replay funnel splits (search unchanged)")
  g_mp      = parser.add_argument_group("motion-plan export (mr_bench)", "hand the plan to the multi-robot motion planner")
  g_viz     = parser.add_argument_group("visualization", "viewers, grasp/rest-pose catalogs, schematic views")
  g_diag    = parser.add_argument_group("diagnostics & IK tests", "one-off IK probes, in-search stats, debug knobs")
  g_output  = parser.add_argument_group("output & persistence", "metrics, plan post-passes, save/replay")

  g_scene.add_argument(
      "--num-robots", type=int, default=4,
      help="Number of robots to spawn (max 6). Default: 4",
  )
  g_scene.add_argument(
      "--floating-ee", action=argparse.BooleanOptionalAction, default=False,
      help="Use free-floating end-effectors instead of the full arm.",
  )
  g_scene.add_argument(
      "--vacuum", action="store_true",
      help="Use the vacuum end-effector instead of the two-finger gripper.",
  )
  g_scene.add_argument(
      "--grippers", default=None, metavar="CSV",
      help="MIXED END EFFECTORS: per-robot gripper kinds, comma-separated (vacuum|two_finger), "
           "e.g. 'vacuum,two_finger,two_finger' for a1..a3. Fewer entries than robots: the "
           "last one repeats. Overrides --vacuum. Each grasp is kind-mask filtered (a robot "
           "only takes grasps its own gripper can stand at, collision-checked per kind); the "
           "wrench model prices each grasp with its holder's capability; the labeled "
           "heuristics stay admissible (pick facts OR over kinds; wrench facts fall back to "
           "topological under mixed). Incompatible with --anon-robots (identity matters).",
  )
  g_scene.add_argument(
      "--robot", choices=["ur10", "ur10_limited", "ur5", "ur5_limited", "fr3", "openarm", "husky"], default="ur10",
      help="Robot arm for all robots: ur10 (default), ur10_limited (ur10 with a tighter "
           "shoulder-lift/elbow workspace), ur5, ur5_limited (ur5 with the same "
           "restricted workspace), fr3 -- TWO static Franka FR3 arms with robotiq "
           "two-finger grippers on the table's -y edge (--num-robots is forced to 2; "
           "EXPERIMENTAL: no analytic IK yet, validation runs the precheck+KOMO path). "
           "openarm -- static OpenArm v2 humanoid duo(s) with pinch grippers "
           "(fingers deactivated open; --num-robots 2 or 4). With --floating-ee this keeps "
           "the native OpenArm pinch/tool geometry but replaces each arm by six free joints; "
           "or husky -- the calibrated fixed Husky workcell with its two embedded UR5e arms "
           "and Robotiq grippers (--num-robots must be 2; no calibrated SSIK yet). "
           "the arm choice matters for the real arm (no --floating-ee "
           "/ --real-ik) -- note the ur5 has a much shorter reach (~0.85 m vs ~1.3 m).",
  )
  g_scene.add_argument(
      "--robot-types", default=None,
      help="Advanced: comma-free per-slot override of the robot type (defaults to --robot for "
           "every slot). Rarely needed.",
  )
  g_scene.add_argument(
      "--openarm-real-finger-radius-scale", type=float, default=1.0, metavar="S",
      help="DIAGNOSTIC for --robot openarm: multiply only the REAL arm's 15mm finger-"
           "collision capsule radii by S in (0,1]. The floating grasp/closure proxy stays "
           "nominal, so fixed-plan A/B tests keep identical grasp ids. Not enabled by any "
           "profile; if a smaller value is accepted as model calibration, update the real "
           "and floating geometries together instead.",
  )
  g_scene.add_argument(
      "--perturb-start-pose", action="store_true",
      help="Randomly perturb the object start poses (kept collision free).",
  )
  g_scene.add_argument(
      "--part-float", type=float, default=0.002, metavar="M",
      help="z offset of the scattered parts above the table top (metres, default 0.002 since "
           "2026-07-29; 0.015 was the historical float). Propagates through placement anchors, "
           "so every later set-down keeps the same hover. The sim RAISES its table by this "
           "much to make parts start resting, so every mm here is planner-certified free "
           "space that is solid table in the sim (FRAGILITY 2.6) -- keep it small.",
  )
  g_scene.add_argument(
      "--random-init", type=float, nargs="?", const=0.45, default=None, metavar="HALF",
      help="FULLY randomize the initial part positions: uniform x/y in the central square "
           "[-HALF, +HALF]^2 between the robots (default 0.45 when given bare) plus a random "
           "yaw, rejection-sampled collision-free. Seeded via --seed (reproducible). "
           "Overrides the scatter line; --perturb-start-pose is ignored with it.",
  )
  g_scene.add_argument(
      "--scatter-spacing", type=float, default=1.0 / 7.0, metavar="M",
      help="x-pitch of the scatter line in metres (default 1/7 ~ 0.143, the historical "
           "layout). For analysing how much search/IK cost is scene density.",
  )
  g_scene.add_argument(
      "--center-scatter", action=argparse.BooleanOptionalAction, default=False,
      help="Arrange the ordinary scatter as a compact, centred, roughly-square lattice at "
           "the requested spacing instead of the historical fixed x=-0.5 one-row anchor. "
           "This preserves density and part order while avoiding embodiment-dependent "
           "reachability failures for short-arm robots. Enabled by --search-profile standard "
           "and persisted with saved plans.",
  )
  g_scene.add_argument(
      "--grid-init", type=int, nargs="?", const=2, default=None, metavar="COLS",
      help="Initial-state layout: pack the parts in a grid COLS wide (default 2 when given "
           "bare), centred on the table, instead of the scatter line. (cross --parts 7 "
           "--grid-init 2 -> a 2x3 block + 1.) The ASSEMBLY is unchanged; this only sets "
           "where parts START. Difficulty is set by --grid-gap: a tight grid crowds the "
           "arms so reaching an inner cube collides with its neighbours -> the planner must "
           "RELOCATE cubes to clear access. Overrides the scatter line; --random-init / "
           "--perturb-start-pose are ignored with it.",
  )
  g_scene.add_argument(
      "--stable-rest-poses", action=argparse.BooleanOptionalAction, default=False,
      help="Fill the per-group PLACEMENT TABLE from the shape's true stable rest "
           "orientations instead of the six axis-aligned face-down rotations. The two are "
           "the same kind of object -- a face direction IS a placement pose -- so this is a "
           "drop-in swap of the enumerator: id 5 still means 'as anchored', and every "
           "consumer (places, rest heights, the labeled heuristics, the wrench gates) is "
           "unchanged. A part then rests the way its geometry actually allows: a chamfered "
           "or mesh part on real facets, a group on the facets of the ASSEMBLED hull with "
           "the mass-weighted centre of mass. PLACEMENT ids rebind, so a plan saved with "
           "the flag does not replay without it (saved in the plan config). Default off.",
  )
  g_scene.add_argument(
      "--placement-yaws", type=int, default=1, metavar="N",
      help="Offer N uniformly spaced WORLD-Z rotations of every stable table placement "
           "(1..24; default 1 preserves the historical action space exactly). Table yaw "
           "is physically free, so every copy inherits the same support-stability verdict, "
           "but it can expose a different fixed-arm IK solution. Rotations have distinct "
           "placement ids and are saved with the plan. Branching grows by up to N; 4 or 8 "
           "is a useful diagnostic before 12.",
  )
  g_search.add_argument(
      "--placement-yaw-widening", action=argparse.BooleanOptionalAction, default=False,
      help="With --placement-yaws N>1 and the ordinary milestone solver, expose the finite "
           "yaw lattice per failed LOCAL fill in coarse-to-fine prefixes (1,2,4,...,N) "
           "before widening to all-part manipulation. Successful IK witnesses are reused; "
           "the final prefix is the same complete action space as eager N. Off keeps eager "
           "placement enumeration.",
  )
  g_scene.add_argument(
      "--plain-box", action="store_true",
      help="Cube parts use rai's SHARP box (ST.box) instead of the default sphere-swept ssBox. "
           "ssBox inflates every face by the corner radius r, so two flush-mated cubes overlap "
           "by 2r and a gripper grazing a face meets the inflated surface -- a geometry A/B "
           "test for spurious motion-planning collisions. Applied to solve, export, and replay "
           "(saved in the plan config).",
  )
  g_scene.add_argument(
      "--grid-gap", type=float, default=0.1, metavar="FRAC",
      help="With --grid-init: inter-cube gap as a FRACTION of cube size (grid pitch = "
           "cube_size*(1+FRAC), default 0.1). Small (~0.1) = shoulder-to-shoulder (a side-"
           "grasp gripper can only reach protruding cubes; best with --vacuum top-pick). "
           "Large (~1.0) = a cube-wide gap between parts (all reachable, little relocation). "
           "The knob that trades off how much relocation the layout forces.",
  )
  g_viz.add_argument(
      "--view", action=argparse.BooleanOptionalAction, default=True,
      help="Open the interactive viewer (default on; --no-view for batch runs).",
  )
  g_viz.add_argument(
      "--disp-final", dest="disp_final", action="store_true",
      help="Show the final desired result.",
  )
  g_viz.add_argument(
      "--show-init", action="store_true",
      help="Open a blocking viewer on the INITIAL scene (after scatter/--random-init) before "
           "anything else runs -- close the window to continue (e.g. into --solve). For "
           "checking layouts, e.g. what --random-init sampled.",
  )
  g_viz.add_argument(
      "--viz-grasps", action="store_true",
      help="GRASP CATALOG in viser (add --viser): one scrubbable step per part showing every "
           "candidate grasp as a floating EE (green=flat / orange=curved surface; with "
           "--wrench, green=mate-viable / grey=not). No search. Prints per-part counts.",
  )
  g_viz.add_argument(
      "--viz-initial-grasps", nargs="?", const="all", default=None, metavar="PART",
      help="INITIAL-SCENE CHEAP-GATE CATALOG in viser (add --viser): one scrubbable step "
           "per grasp that passes both the floating-EE pick collision check and virtual jaw "
           "closure, with the actual table, clutter and gripper geometry at the inferred "
           "closed aperture. Optionally name one PART; omitting PART shows all. SSIK and "
           "KOMO are not run.",
  )
  g_viz.add_argument(
      "--viz-grasp-closure", metavar="ROBOT:PART:GRASP", default=None,
      help="Visualize the virtual two-finger closure certificate for one grasp in the "
           "INITIAL scene. Records the fixed-open receiver through its inferred bilateral-"
           "contact aperture, including per-step target distances and any sweep blocker. "
           "Use with --viser for a scrubbable web view. No search.",
  )
  g_viz.add_argument(
      "--viz-rest-poses", action="store_true",
      help="REST-POSE CATALOG in viser (add --viser): each group (singletons, connect "
           "modules, full assembly) laid out on the table in every stable rest pose, one "
           "step per (group,face). A group with NO steps is an unparkable subassembly. No search.",
  )
  g_viz.add_argument(
      "--viz-rest-poses-grid", action="store_true",
      help="GRID REST-POSE CATALOG in viser (add --viser): one step per singleton/module/"
           "assembly group, with all of that group's stable placement orientations arranged "
           "side by side on the table. Labels map grid slots to the exact placement ids used "
           "by search. No search.",
  )
  g_viz.add_argument(
      "--mate-sequence-viz", action="store_true",
      help="SCHEMATIC mate-sequence view (no search): walk the DAG's assembly order, and at "
           "each mate draw the contact interface point(s) (green crosses), the insertion "
           "PRESS the seating must exert (orange), and the MOMENT that press makes about the "
           "joined part (blue -- nonzero when the interface is off the press axis). Add "
           "--viser for the scrubbable web view; prints per-mate press + peak moment. A way "
           "to sanity-check interface points and joint loads before ever solving.",
  )
  g_viz.add_argument(
      "--show-successors", action="store_true",
      help="Visualize a few successor states of the initial search state, then exit.",
  )
  g_scene.add_argument(
      "--parts", type=int, default=7,
      help="Number of parts in the assembly (clamped per type). Default: 7.",
  )
  g_scene.add_argument(
      "--assembly",
      choices=["cross", "cube", "arch", "arch-seq", "cube-plates", "cube-plates-2x2x3",
               "cube-sequence-2x2x3",
               "cuboid-tower", "fixture-rack", "car", "car-mini", "fabrica-car", "fabrica-beam",
               "fabrica-cooling-manifold", "fabrica-duct", "fabrica-gamepad",
               "fabrica-plumbers-block", "fabrica-circular-stool",
               "slot-pin", "peg-hole", "peg-hole-screw", "peg-stool-4-screw", "peg-hole-chamfer",
               "peg-stool",
               "peg-stool-4", "peg-stool-cuboid", "peg-stool-cuboid-4",
               "peg-stool-tight", "peg-stool-4-tight",
               "peg-stool-cuboid-tight", "peg-stool-cuboid-4-tight",
               "peg-stool-chamfer", "peg-stool-4-chamfer",
               "peg-stool-cuboid-chamfer", "peg-stool-cuboid-4-chamfer",
               "peg-stool-tight-chamfer", "peg-stool-4-tight-chamfer",
               "peg-stool-cuboid-tight-chamfer", "peg-stool-cuboid-4-tight-chamfer",
               "mesh-demo", "compound-demo", "real-stool", "real-bench",
               "yandv-stool", "yandv-bench", "chair"],
      default="cross",
      help="Assembly to build: cross (cube cross, 7), cube (2x2x2 grid, 8), "
           "arch (two pillars + lintel, 7 -- known-flawed star model), "
           "arch-seq (arch shape built as a sequential chain, 7), "
           "cube-plates (2x2x2 cube as two stacked 2x2 plates, faithful DAG), "
           "cube-plates-2x2x3 (three stacked 2x2 plates, 12 cubes -- use --parts 12), "
           "cube-sequence-2x2x3 (the same 12-cell lattice assembled in one strict, "
           "face-adjacent piece-by-piece sequence), "
           "cuboid-tower (jenga-style: 2 laid cuboids per level, levels rotated 90deg, up to 8), "
           "fixture-rack (scalable base rail + repeated top-down posts and lateral brackets, "
           "up to 13 parts), "
           "car (cuboid body + 4 sideways cylinder wheels, 5), "
           "car-mini (car scaled to w=0.03: wheels fit ACROSS the two-finger gap), "
           "fabrica-car (five-part Fabrica CAD car; source fixture part 2 excluded), "
           "fabrica-beam, fabrica-cooling-manifold, fabrica-duct, fabrica-gamepad, "
           "fabrica-plumbers-block, and fabrica-circular-stool (exact Fabrica visuals plus "
           "reviewed collision proxies and proximity-tree mate annotations), "
           "slot-pin (base + cantilever bar + fastening pin -- the --staged-mates testbed), "
           "peg-hole (compound blind socket + top-down cylindrical peg, 2 -- concavity smoke "
           "test; use --grasp-sampler), peg-stool (compound seat with three real downward "
           "sockets + cylindrical legs, 4; use --grasp-sampler and a vacuum gripper), "
           "peg-stool-4 (the same stool with FOUR cylindrical legs, 5), "
           "peg-stool-cuboid (the peg-stool with SQUARE cuboid legs, 4), "
           "peg-stool-cuboid-4 (the four-legged stool with SQUARE cuboid legs, 5), and "
           "the four -tight kinds (same four stools, peg size derived from the socket hole "
           "so the nominal clearance is 1mm instead of ~5mm; a square peg's CORNER gap is "
           "the binding clearance). peg-hole-chamfer: the peg-hole peg with a 4mm 45-degree "
           "CHAMFERED insertion tip (an inline convex mesh), and the eight -chamfer stool "
           "kinds (same stools/clearances, every leg's insertion end carries the same "
           "chamfer) -- a slightly misaligned insertion slides into the pocket instead of "
           "catching the rim. The -chamfer kinds ALSO bevel every socket mouth with a "
           "45-degree ENTRY chamfer of its own (peg-hole 6mm, stool 4mm): each hole rim "
           "widens into a funnel, so tip and mouth facets guide the insertion together. "
           "All peg-stool variants: "
           "use --grasp-sampler and a vacuum "
           "gripper; the clearance is a constructor parameter (PegStoolAssembly peg_clearance). "
           "real-stool (Y&V CAD 4-hole 170mm seat + 4 handed legs, 5; faithful compound "
           "blind pockets and rear upstand) and real-bench (the 2-hole 85mm variant, 3). "
           "The legacy yandv-stool/yandv-bench names are aliases. "
           "Default: cross.",
  )
  g_solver.add_argument(
      "--solve", action="store_true",
      help="Run the A* assembly search and print the plan.",
  )
  g_search.add_argument(
      "--canonical", action="store_true",
      help="Break robot symmetry in the search (lowest-index free robot). TEST-ONLY: "
           "unsound once robots have fixed bases -- emits a warning.",
  )
  g_solver.add_argument(
      "--weight", type=float, default=1.0,
      help="Weighted-A* factor f=g+w*h: 1=optimal, >1 faster & bounded-suboptimal, "
           "inf=greedy best-first. Use a large value (e.g. 5 or inf) for fast solutions.",
  )
  g_diag.add_argument(
      "--check-init", action="store_true",
      help="Validate the INITIAL scene before anything else: report whether it is collision-"
           "free and whether every part is graspable by at least one robot (a part no arm can "
           "pick -- e.g. a cube buried in a tight --grid-init block -- makes the instance "
           "unsolvable). Uses the cheap floating-EE check, or real-arm IK with --real-ik. "
           "Prints a report and continues (combine with --solve to check then solve).",
  )
  g_diag.add_argument(
      "--test-pick-ik", nargs="?", const="a1:obj_1:0", default=None,
      metavar="ROBOT:PART:GRASP",
      help="Pop up the real-robot pick IK for a (robot, part, grasp) and exit, e.g. "
           "--test-pick-ik a1:obj_1:0 (default a1:obj_1:0).",
  )
  g_diag.add_argument(
      "--test-assemble-ik", nargs="?", const="a1:obj_1:0", default=None,
      metavar="ROBOT:PART:GRASP",
      help="Pop up the real-robot assemble IK: robot holds PART at GRASP and mates it "
           "onto its manual parent, then exit, e.g. --test-assemble-ik a1:obj_1:0.",
  )
  g_diag.add_argument(
      "--test-assemble-held-ik", nargs="?", const="a1:obj_1:0:a2:1", default=None,
      metavar="ROBOT:PART:GRASP:TARGET_ROBOT:ONTO_GRASP",
      help="Like --test-assemble-ik but the target (manual parent) is HELD by TARGET_ROBOT "
           "at ONTO_GRASP, so both cubes are in the air and both arms co-optimize. "
           "E.g. --test-assemble-held-ik a1:obj_1:0:a2:1.",
  )
  g_diag.add_argument(
      "--test-place-ik", nargs="?", const="a1:obj_1:0:1:5", default=None,
      metavar="ROBOT:PART:GRASP:N_ASSEMBLED[:FACE]",
      help="Pop up the real-robot place IK: with N_ASSEMBLED parts already mated, ROBOT "
           "picks the subassembly by PART at GRASP and sets it on the table resting on "
           "placement id FACE (0-5 in the legacy table, 5=upright default; augmented/stable "
           "tables also use persisted quaternion-derived ids), then exit. "
           "E.g. --test-place-ik a1:obj_1:0:1:0.",
  )
  g_ik.add_argument(
      "--real-ik", action=argparse.BooleanOptionalAction, default=True,
      help="Run the search with real-robot IK in the feasibility loop (RobotIK), and "
           "replay the plan with the actual arm poses instead of floating-EE stand-ins. "
           "Default ON (2026-07-29 defaults change); --no-real-ik restores floating EEs.",
  )
  g_ik.add_argument(
      "--ik-seeding", action=argparse.BooleanOptionalAction, default=True,
      help="With --real-ik: use analytic closed-form IK (ssik) on every fixed-target skill "
           "(pick/place/assemble-onto-table, incl. braced): exact per-arm reachability rejects "
           "(replacing the relaxed precheck), collision-probed branch combos, and warm starts "
           "for the KOMO. Calibrated arm/tool pairs include ur5/ur10 with two-finger or "
           "vacuum EEs; others keep the precheck.",
  )
  g_ik.add_argument(
      "--assemble-ssik-pose-seeds", action=argparse.BooleanOptionalAction, default=False,
      help="EXPERIMENTAL held-target assembly initializer: instantiate the existing six "
           "canonical soft biases as exact poses at their existing centre target, run SSIK "
           "for every participating robot, collision-rank the complete configurations, and "
           "warm-start the matching KOMO biases from the best candidates. Candidate-pose "
           "failure only removes that seed; it never rejects the continuous assembly action.",
  )
  g_ik.add_argument(
      "--assemble-ssik-pose-projection", action=argparse.BooleanOptionalAction, default=False,
      help="EXPERIMENTAL held-target assembly initializer: for every canonical orientation, "
           "project the existing preferred centre toward the target subassembly's currently "
           "realized position on a five-point line, closest-to-centre first. The first pose "
           "with SSIK branches for every participating robot seeds the matching KOMO. This "
           "is an ablation beside --assemble-ssik-pose-seeds and never rejects an action.",
  )
  g_ik.add_argument(
      "--assemble-unbiased-first", action=argparse.BooleanOptionalAction, default=False,
      help="EXPERIMENTAL ordering-only ablation for a held-target assemble: try the neutral "
           "continuous KOMO before the six canonical soft-orientation basins, then retain "
           "the same fresh-random fallback. No basin is removed and total eventual effort "
           "is unchanged; only a successful neutral solve can end the sweep earlier.",
  )
  g_ik.add_argument(
      "--assemble-basin-racing", action=argparse.BooleanOptionalAction, default=False,
      help="EXPERIMENTAL collision-aware optimizer racing for held-target assembly. Try the "
           "neutral KOMO first; if it fails, give all six canonical basins a bounded real-KOMO "
           "probe, rank by their actual constraint residuals, and fully continue only a "
           "bounded number per open-witness visit. Unpromoted basins remain pending and are "
           "eventually widened; the historical policy is unchanged when this is off.",
  )
  g_ik.add_argument(
      "--assemble-race-probe-evals", type=int, default=24, metavar="N",
      help="NLP evaluation cap for each shallow canonical-basin probe under "
           "--assemble-basin-racing (default 24, selected below the measured ~100-evaluation "
           "cost of the repeatedly failing basins).",
  )
  g_ik.add_argument(
      "--assemble-race-promote", type=int, default=3, metavar="N",
      help="Number of shallow-ranked assembly basins continued to a full solve per open "
           "witness visit (default 3, so the standard two epochs cover all six).",
  )
  g_ik.add_argument(
      "--handover-basin-racing", action=argparse.BooleanOptionalAction, default=False,
      help="EXPERIMENTAL collision-aware optimizer racing for handover. Preserve the ordinary "
           "full solve; after it fails, shallow-probe fresh random initializations, "
           "rank their actual constraint residuals, and fully continue only the best few per "
           "open-witness visit. Unpromoted starts remain pending and never refute the action.",
  )
  g_ik.add_argument(
      "--handover-race-probe-evals", type=int, default=24, metavar="N",
      help="NLP evaluation cap for each shallow random handover probe under "
           "--handover-basin-racing (default 24).",
  )
  g_ik.add_argument(
      "--handover-race-starts", type=int, default=6, metavar="N",
      help="Number of fresh handover initializations in one cached racing generation "
           "(default 6). Exhausted generations widen to another fresh set on later visits.",
  )
  g_ik.add_argument(
      "--handover-race-promote", type=int, default=3, metavar="N",
      help="Maximum shallow-ranked handover starts continued to full convergence per visit "
           "(default 3; the visit's ordinary restart budget may lower this further).",
  )
  g_ik.add_argument(
      "--ik-restarts", type=int, default=12, metavar="N",
      help="With --real-ik: random-restart budget of each FULL collision-aware KOMO solve "
           "when no analytic seeds warm-start it (seeded solves already cap at 2). MEASURED "
           "(peg-stool chamfer): restarts deliver 25/202 full-solve successes and nothing "
           "wins at attempt >=7. Lower to trade stochastic completeness for time -- a missed "
           "witness blacklists the edge (the fill's other candidates still run).",
  )
  g_ik.add_argument(
      "--ik-positive-reuse", action=argparse.BooleanOptionalAction, default=False,
      help="Canonicalize skill-local physical IK programs across symbolic aliases (table "
           "scene for picks/table-supported skills; payload robot/grasp assignments for "
           "handovers; participating payloads for assembly; destination scene for place). "
           "Their exact witness and open-witness effort cursor are shared directly, including "
           "which deterministic NLP/bias slots were already attempted. Enabled by "
           "--search-profile standard; "
           "--no-ik-positive-reuse is the matched ablation.",
  )
  g_ik.add_argument(
      "--precheck-restarts", type=int, default=12, metavar="N",
      help="Restart budget of the RELAXED reach-precheck solve. Its verdict is residual-gated "
           "EVIDENCE only (borderline failures fall through to the full solve, which decides), "
           "and MEASURED: all precheck successes came from the FIRST relaxed attempt -- the "
           "default 12 burns restarts on hopeless calls for nothing.",
  )
  g_ik.add_argument(
      "--fail-order", action=argparse.BooleanOptionalAction, default=False,
      help="With --real-ik: failure-informed candidate ORDERING. Collects per-candidate "
           "outcome evidence (assemble-bias slots first) and sweeps candidates never-tried-"
           "first / ascending fail rate. Ordering ONLY -- every candidate stays reachable in "
           "its exhaustive sweep, so completeness is untouched; equal-f tie-breaks change. "
           "MEASURED motivation: assemble bias re-probes burned 70s/instance (obj_3 x56, "
           "obj_4 x37), and the winning bias repeats across milestones.",
  )
  g_ik.add_argument(
      "--empty-seed-refutes", action=argparse.BooleanOptionalAction, default=False,
      help="With --real-ik: treat an EMPTY calibrated analytic (ssik) seed set as a sound "
           "unreachability proof and reject the bind WITHOUT relaxed precheck or KOMO. "
           "Sound for the rai models because q1..q5 are roll-independent and wrist_3 spans "
           ">= 2pi (so no other vacuum roll helps either); OFF keeps the legacy tolerant-"
           "KOMO chance on exactly-unreachable poses. MEASURED motivation: empty-set binds "
           "otherwise pay precheck + full-solve cost before failing anyway.",
  )
  g_ik.add_argument(
      "--analytic-seed-cap", type=int, default=None, metavar="K",
      help="With --real-ik and a calibrated arm: warm-start KOMO with only the K "
           "best-ranked exact branches (shallowest collision first) instead of all of "
           "them. Cuts the seeded-solve cost on binds whose branch set is large and "
           "mostly colliding; None = legacy behaviour (all branches).",
  )
  g_ik.add_argument(
      "--ssik-seed-neighborhood", type=float, default=0.0, metavar="M",
      help="With --real-ik and calibrated SSIK: when the exact target has no physical-limit "
           "branch, or all exact branches lie inside the required witness joint margin, "
           "also seed KOMO from deterministic Cartesian neighbours up to M metres away. "
           "These are initialization hints only: KOMO still solves and validates the original "
           "target, and an empty neighbourhood never rejects an action. 0 disables.",
  )
  g_ik.add_argument(
      "--ssik-neighborhood-cap", type=int, default=6, metavar="K",
      help="Maximum number of additional near-target SSIK configurations supplied to one "
           "KOMO bind by --ssik-seed-neighborhood (default 6).",
  )
  g_phys.add_argument(
      "--dense-grasps", action="store_true",
      help="TWO-FINGER only: add the +-45deg WRIST-TILTED variant of every face grasp -- the "
           "pads keep the SAME flat face pinch (rotation is about the finger-closing axis "
           "through the tool centre), only the wrist/approach swings into the diagonal "
           "'between the faces'. 3x the grasp candidates (more IK/table work) in exchange "
           "for approaches that clear obstacles a face-aligned wrist hits. Ignored with "
           "--vacuum (rotationally symmetric).",
  )
  g_ik.add_argument(
      "--grasp-sampler", action=argparse.BooleanOptionalAction, default=False,
      help="SECOND grasp-DB code path (default off = byte-identical): tessellate each part "
           "shape into a MESH (trimesh) and SAMPLE candidate grasps off its surface -- a "
           "suction block (surface point + inward normal, canonical roll) and an antipodal "
           "pinch block (opposing surface pairs within the finger aperture) -- then build a "
           "bounded scene-reserved, diversity-ordered database. The precursor to a "
           "proper mesh grasp generator: everything downstream (EE collision filter, wrench "
           "pricing, IK, sim, export) consumes the sampled DB unchanged via grasp ids. "
           "Incompatible with --dense-grasps/--grasp-translations. Deterministic: a private "
           "RNG seeded from (--grasp-sampler-seed, shape), never the global stream.",
  )
  g_ik.add_argument(
      "--grasp-sampler-keep", type=int, default=20, metavar="N",
      help="--grasp-sampler: initial TOTAL grasps kept per part (default 20). The current "
           "scene-reserved hybrid reserves available per-kind pick, mandatory-mate, and "
           "intermediate/final rigid-component representatives before a stable conventional-"
           "plus-sampled tail. Mesh/compound parts use the whole budget.",
  )
  g_ik.add_argument("--grasp-sampler-db-hash", default=None, help=argparse.SUPPRESS)
  g_ik.add_argument(
      "--grasp-refine", action=argparse.BooleanOptionalAction, default=False,
      help="With sampled grasps, search primitives first with the exact conventional "
           "grasp prefix (mesh/compound parts still use samples), then retry with sampled "
           "primitive extras only if that compact search finds no plan. Off exposes the full "
           "configured hybrid prefix immediately. This is an explicit ablation; standard uses "
           "the full initial prefix plus failure-driven --grasp-widen.",
  )
  g_ik.add_argument(
      "--grasp-widen", action=argparse.BooleanOptionalAction, default=False,
      help="If search returns no plan with the initial sampled-grasp prefix, retry with "
           "monotonically larger stable prefixes. A finite IK/search miss never deletes an "
           "old grasp; successful witnesses remain reusable.",
  )
  g_ik.add_argument(
      "--grasp-widen-batch", type=int, default=8, metavar="N",
      help="Number of additional grasp ids exposed per failure-driven widening epoch "
           "(default 8).",
  )
  g_ik.add_argument(
      "--grasp-widen-max", type=int, default=96, metavar="N",
      help="Maximum finite grasp master per part under --grasp-widen (default 96). "
           "The master is generated/classified once before search; widening reveals "
           "stable prefixes of it. Raise N to probe and retain more raw candidates.",
  )
  g_ik.add_argument(
      "--grasp-local-widen", action=argparse.BooleanOptionalAction, default=False,
      help="Build one finite master database up to --grasp-widen-max, but expose "
           "only --grasp-sampler-keep ids per part to a milestone fill. If the fill "
           "stalls, enlarge the incoming part's prefix by --grasp-widen-batch before "
           "opening the combinatorial bracing/all-part recovery spaces. If that complete "
           "compact milestone session exhausts, the same iterator runs one eager pass with "
           "the full finite master, so this is lazy column generation rather than a "
           "grasp-pruning rule. Milestone solver only. If both widening schedules resolve "
           "true (for example, standard profile + explicit local widening), this local "
           "schedule supersedes the whole-search --grasp-widen schedule and that resolved "
           "choice is logged and saved.",
  )
  g_ik.add_argument("--grasp-local-caps", default=None, help=argparse.SUPPRESS)
  g_ik.add_argument(
      "--component-grasp-keep", type=int, default=0, metavar="N",
      help="Component-level grasp-column generation (built-in default 0; standard profile "
           "uses 32). Single parts retain their full "
           "sampled database; each multi-part rigid component exposes at most N collision-"
           "free rows from a stable round-robin merge of its constituent databases. If the "
           "restricted search returns no plan, one automatic retry restores the exact historical "
           "full union, so this orders finite search effort rather than pruning grasps. "
           "Zero restores the historical eager full union.",
  )
  g_ik.add_argument(
      "--grasp-sampler-candidates", type=int, default=200, metavar="M",
      help="--grasp-sampler: raw surface samples per shape before downsampling (default 200).",
  )
  g_ik.add_argument(
      "--grasp-sampler-full-probe", action=argparse.BooleanOptionalAction, default=False,
      help="EXPERIMENTAL: first scene-classify the historical 32-row primitive / 64-row "
           "mesh probe, then classify the complete raw cloud only for an incoming part whose "
           "compact pool has no direct pick-to-mate bridge. This reveals candidates lost "
           "before role-aware selection without charging every easy part for the dense pool.",
  )
  g_ik.add_argument(
      "--grasp-sampler-roll-refine", action=argparse.BooleanOptionalAction, default=False,
      help="Append the four 45-degree midpoint rolls between the sampled "
           "pinch generator's existing nested 0/90/180/270-degree approach rolls. This "
           "targets handover/co-grasp compatibility without changing surface contact pairs; "
           "the existing four rows retain their order.",
  )
  g_ik.add_argument(
      "--grasp-sampler-pad-support", type=float, default=0.75, metavar="F",
      help="--grasp-sampler: least fraction of the CONTACT PATCH that must lie on the part. "
           "The patch mirrors the collision proxies: the two-finger pad (20x40mm, robotiq_"
           "tight.g's capsule) at BOTH pads of a pinch, and the vacuum cup (the seal "
           "diameter) for a suction candidate. --grasp-sampler-margin keeps the contact "
           "POINT off a crease; this keeps the pad AREA on the part, which is what decides "
           "whether a pinch holds or pivots and whether a cup can seal. A shape too small "
           "for the patch (a thin leg) keeps its best-supported candidates instead. "
           "Default 0.75; 0 disables. Saved with the plan -- it rebinds grasp ids.",
  )
  g_ik.add_argument(
      "--grasp-sampler-wrap", type=float, default=None, metavar="W",
      help="--grasp-sampler: CURVATURE-FAIR pad-support metric (opt-in; off = the legacy "
           "metric, byte-identical). A pad sample counts as supported when its DEFORMATION "
           "DISTANCE (signed distance to the surface) RELATIVE TO THE NEAREST CONTACT is "
           "<= W metres -- the pad's wrap budget. The legacy metric lifts the whole pad "
           "along the centre normal and caps each ray at lift+depth: it scores a real side "
           "pinch of a peg (a compliant pad wrapping a 12.5mm-radius barrel, ~3mm sagitta "
           "over the 20mm pad) as ~0.3 and, worse, can never see the wrap roll (pads across "
           "the circumference, palm approaching from the side) because its rigid rays miss "
           "the barrel. e.g. 0.006 "
           "admits the axis-aligned pads on the peg; 0.012 also admits the side-approach "
           "wrap roll (~10mm bend). Flat faces (deformation 0) gate exactly as before. The "
           "flag also EMITS the complementary roll (pads ACROSS the part, approach from the "
           "side) for every surface -- for a flat long box (the cuboid peg) that is the only "
           "pinch whose palm does not sit at the peg END and collide with the seat during "
           "insertion. Saved with the plan -- it rebinds grasp ids.",
  )
  g_ik.add_argument(
      "--grasp-sampler-scenes", action=argparse.BooleanOptionalAction, default=False,
      help="--grasp-sampler: SCENE-AWARE sampling (opt-in). Every raw candidate is tagged "
           "with floating-EE verdicts at the INITIAL pick, mandatory mate, and each authored "
           "intermediate/final rigid-component context. The keep budget reserves available "
           "representatives for those roles before the diversity tail, rather than hoping a "
           "uniform cloud retains the grasps needed to carry a partial assembly. The DB "
           "becomes per PART INSTANCE (identical legs filter differently). These are ranking "
           "labels, not feasibility shortcuts; exact action gates remain authoritative. "
           "Deterministic; saved with the plan -- it rebinds grasp ids.",
  )
  g_ik.add_argument(
      "--grasp-sampler-margin", type=float, default=0.008, metavar="M",
      help="--grasp-sampler: minimum clearance from a sampled contact to the nearest "
           "FEATURE EDGE of the shape (dihedral > 5 deg: a box's 12 edges, a cylinder's cap "
           "rims, a chamfer's borders -- never the tessellation's internal edges). Keeps "
           "pads and suction cups INSIDE their face instead of hanging over a corner, which "
           "is also what the analytic patch reasoning (surface class, seal diameter) "
           "assumes. A shape too small to honour it keeps its most-interior candidates "
           "instead. Default 0.008; 0 disables. Saved with the plan -- it rebinds grasp ids.",
  )
  g_ik.add_argument(
      "--grasp-sampler-strict", action=argparse.BooleanOptionalAction, default=False,
      help="--grasp-sampler: treat --grasp-sampler-margin and --grasp-sampler-pad-support "
           "as hard quality gates. Below-gate sampled rows are omitted instead of being "
           "used as a last-resort fallback; if no candidate remains, isolated grasp preflight "
           "fails loudly. Canonical face grasps are unchanged. Off preserves the widening "
           "fallback behavior. Saved with the plan -- it rebinds grasp ids.",
  )
  g_ik.add_argument(
      "--grasp-sampler-seed", type=int, default=0, metavar="S",
      help="--grasp-sampler: sampler RNG seed (saved with the plan; grasp ids only replay "
           "under the same seed/counts).",
  )
  g_ik.add_argument(
      "--ik-collision-gate", type=float, default=0.015, metavar="M",
      help="With --ik-seeding: penetration depth (metres) beyond which an exact analytic branch "
           "counts as DEEPLY colliding; a bind is rejected pre-KOMO only if EVERY branch combo "
           "is deep (arm-arm pairs use 2x the gate). Default 0.015 -- measured: the full solve "
           "rescues grazes up to ~0.010, never >=0.014. 0 = ANY penetration rejects "
           "(aggressive: loses grazing rescues). Negative = off (probe/ranking only).",
  )
  g_ik.add_argument(
      "--full-ik", action="store_true",
      help="DEBUG: disable the single-active-arm pick-IK optimisations (restrict-to-arm + "
           "strip inactive-robot collision). Uses the full multi-arm KOMO instead. Slower; "
           "for checking the optimisations against the reference.",
  )
  g_solver.add_argument(
      "--optimistic", action="store_true",
      help="Use the optimistic (Dogar-style) per-part search -- add one part at a time via "
           "its own small complete sub-search (reorient/hold/pick/mate/set-down), with exact "
           "fallback -- instead of the exact search. Much faster on the cases that blow up.",
  )
  g_solver.add_argument(
      "--optimistic-fallback", action=argparse.BooleanOptionalAction, default=True,
      help="With --optimistic, fall back to the global exact search when a local part fill "
           "fails (default on). Use --no-optimistic-fallback on instances whose global "
           "backward abstraction is already known to exceed its state guard; the local "
           "failure is then reported directly instead of starting the intractable path.",
  )
  g_solver.add_argument(
      "--optimistic-chain-h", action=argparse.BooleanOptionalAction, default=False,
      help="With --optimistic, build one PROJECTED signature heuristic over only the fixed "
           "assembly-order prefix chain. It retains future rest-face, holder, grasp-class, "
           "bracing and wrench guidance while allowing only one next loose part, avoiding "
           "the global order-ideal/loose-part cross product. Experimental and off by default.",
  )
  g_solver.add_argument(
      "--milestones", action=argparse.BooleanOptionalAction, default=True,
      help="Default ON (2026-07-29). MILESTONE planner: the labeled abstraction's backward (assembly-by-disassembly) "
           "solve picks the next part to add (wrench-informed ORDER, grasps assumed "
           "reachable); each milestone is filled in by a small per-part search "
           "(pick/handover/brace/mate/set-down); a failed fill-in blacklists that milestone "
           "and reroutes. Scales like --optimistic but with a SEARCHED part order, bracing "
           "support, and repair instead of an exact-search fallback.",
  )
  g_search.add_argument(
      "--place-cells", type=int, nargs="?", const=4, default=0, metavar="N",
      help="EXPERIMENTAL: offer up to N extra TRANSLATED placement anchors per (group, face) "
           "near the table centre (default 4 when given bare), in addition to the classic "
           "scatter-anchor placement. Motivation: parts at the scatter extremes get set down "
           "where half the arms cannot reach; central cells make braced work reachable. Adds "
           "branching (~N x places).",
  )
  g_search.add_argument(
      "--por", action="store_true",
      help="EXPERIMENTAL partial-order reduction in the exact search: among INDEPENDENT "
           "successor actions (different robots, disjoint parts) only the canonically sorted "
           "interleaving is expanded. KNOWN UNSOUND in general (feasibility is order-coupled "
           "through geometry) -- meant for spread scenes (--min-sep) with floating EEs where "
           "independence actually holds. Prints when active.",
  )
  g_scene.add_argument(
      "--min-sep", type=float, default=None, metavar="M",
      help="With --random-init: enforce a pairwise x/y separation of at least M metres "
           "between parts (e.g. 0.2), so part geometries/graspability provably cannot "
           "interact -- the --por independence assumption.",
  )
  g_search.add_argument(
      "--prefer-center", action="store_true",
      help="With --place-cells: offer the CELL placements (table centre first) BEFORE the "
           "classic scatter-anchor placement -- at equal f the FIFO tie-break then tries "
           "central placements first. Pure successor ordering, no pruning.",
  )
  g_diag.add_argument(
      "--search-stats", type=float, nargs="?", const=10.0, default=None, metavar="SEC",
      help="Print periodic in-search statistics every SEC seconds (default 10 when given "
           "bare): expanded/claimed/frontier sizes and per-action-kind edge-check + failure "
           "counts; the same per-kind summary is printed with the final result. Exact "
           "search only.",
  )
  g_solver.add_argument(
      "--bounded", action="store_true",
      help="INCUMBENT-BOUNDED search: the milestone planner produces a fast incumbent plan, "
           "then the exact search runs with the incumbent's cost as a sound prune bound "
           "(g+h >= U dies; optimal prefixes always survive). Result <= min(incumbent, "
           "weight x optimal); with --weight 1 the phase-2 result is optimal. Combines the "
           "decomposed solver's speed with the exact search's guarantee.",
  )
  g_solver.add_argument(
      "--lazy", action="store_true",
      help="LAZY-VALIDATION exact search (LazySP-style): search the discrete graph "
           "optimistically (no per-edge cheap-collision/IK), then validate only the CANDIDATE "
           "PLAN's edges; a failed edge is blacklisted and the search reruns (heuristic built "
           "once, caches warm). Same completeness and plan validity as the exact search; "
           "wins when most eagerly-checked edges never end up in a plan.",
  )
  g_solver.add_argument(
      "--full-backward", action="store_true",
      help="Exact reference solver over the FULL concrete symbolic State (robot ids, raw "
           "grasp ids, support contacts, placements, and parent tree are retained). It "
           "materializes the cheap-feasible reachable graph, then runs multi-source reverse "
           "Dijkstra from every goal. Failed real-IK edges are removed and the reverse solve "
           "is repeated. This is deliberately distinct from the projected backward PDB.",
  )
  g_solver.add_argument(
      "--full-backward-max-states", type=int, default=200000, metavar="N",
      help="Hard state guard for --full-backward (default 200000). The solver fails loudly "
           "at the guard and never falls back to a projected or forward search.",
  )
  g_solver.add_argument(
      "--full-backward-max-edges", type=int, default=2000000, metavar="N",
      help="Hard stored-edge guard for --full-backward (default 2000000).",
  )
  g_phys.add_argument(
      "--brace-for-assembly-only", action="store_true",
      help="EXPERIMENTAL (MEASURED HARMFUL, off by default): assume a brace (2nd grasp via "
           "add_grasp) is only needed to stabilise an ASSEMBLE, so a braced state (>=2 grippers "
           "on one group) may only assemble/release. Markovian and correct, BUT it backfires in "
           "the forward search -- braced states reached before the part is positioned become "
           "near-dead-ends the search thrashes on (car-mini fill branching exploded 20-50x). "
           "The domain fact is right; the clean form is an atomic add_grasp+assemble MACRO, not "
           "this filter. Kept for experimentation; prints when active.",
  )
  g_search.add_argument(
      "--declutter", action="store_true",
      help="With --optimistic: allow each per-part step to RELOCATE blocking loose parts "
           "(stow them aside on demand) instead of only touching that part and the assembly. "
           "Removes the 'nothing else is in the way' assumption.",
  )
  g_phys.add_argument(
      "--wrench", action=argparse.BooleanOptionalAction, default=True,
      help="Turn on the static-equilibrium (stability) model: support-gate assemble/release "
           "with the insertion reaction, so sideways mates on a resting assembly need a hold "
           "(mating becomes perpendicular unless braced). Default ON (2026-07-29); "
           "--no-wrench = topological baseline.",
  )
  g_phys.add_argument(
      "--wrench-profile", choices=["max_robotiq", "measured"], default=None, metavar="NAME",
      help="Named gripper-capability envelope. max_robotiq installs the Robotiq 2F-85 "
           "section 6.2.4 absolute limits (50N on each force axis, 5Nm about local x/y, "
           "3Nm about local z/approach). This is a hardware maximum, NOT a measured slip "
           "calibration or safety-rated working envelope and affects two-finger grasps only. "
           "measured installs the paper's axis-resolved two-finger working envelope and its "
           "conservative vacuum envelope. "
           "explicit --two-finger-scale/--tf-close-torque are applied afterwards.",
  )
  g_phys.add_argument(
      "--grasp-model", choices=["isotropic", "directional", "asymmetric"],
      default="asymmetric",
      help="Grasp wrench-capability model (with --wrench): isotropic = the original "
           "direction-agnostic box; directional = weak-shear vacuum / weak-torque "
           "two-finger in the grasp frame; asymmetric = directional + push/pull asymmetry "
           "along the approach axis (pressing into the palm/cup is structural, pull-off is "
           "the weak suction/friction path; the default since 2026-07-29).",
  )
  g_phys.add_argument(
      "--table-friction", type=float, default=None, metavar="MU", dest="table_mu",
      help="With --wrench: model the TABLE support as a FRICTION cone (lateral capacity "
           "mu x normal per support point) instead of the FRICTIONLESS unilateral default. "
           "A table-rest target can then resist the lateral shear --insertion-noise adds "
           "(the unilateral default rejects ANY lateral load, so rest-on-table assembly "
           "with noise on is impossible). Default off (unilateral, byte-identical).",
  )
  g_phys.add_argument(
      "--vacuum-scale", type=float, default=None, metavar="FACTOR",
      help="With --wrench: scale ALL of the VACUUM's capability by this factor (every force "
           "AND torque cap -- push, pull, lateral, couples -- across whichever grasp model "
           "is active), leaving the friction ratio mu and the two-finger untouched. e.g. 0.1 "
           "= a cup a tenth as capable everywhere: still picks a light part but holds almost "
           "nothing against gravity or a press. Default off (factor 1).",
  )
  g_phys.add_argument(
      "--two-finger-scale", type=float, default=None, metavar="FACTOR",
      help="With --wrench: scale ALL of the TWO-FINGER's capability by this factor (every "
           "force AND torque cap -- push, pull, lateral, couples -- across whichever grasp "
           "model is active), leaving the friction ratio mu and the vacuum untouched. e.g. "
           "3.0 = strong fingers that hold the seat edge-grasps the lever arm rejects. "
           "Default off (factor 1).",
  )
  g_phys.add_argument(
      "--vacuum-engage-force", type=float, default=0.0, metavar="N",
      help="With --wrench: transient force a vacuum cup applies ALONG its approach axis while "
           "seating/sealing on the surface. Before pick/add_grasp/handover adds the new "
           "contact, the table and EXISTING holders must balance this point load at the cup. "
           "It is not kept as a permanent load after attachment: the cup/suction preload is "
           "then internal and the normal holding envelope belongs to --grasp-model. "
           "0 disables (compatibility default); try 5N.",
  )
  g_phys.add_argument(
      "--vacuum-engage-friction", type=float, default=None, metavar="MU",
      help="With --wrench and --vacuum-engage-force: give the TABLE support a STATIC friction "
           "cone (mu) inside the ENGAGEMENT gate only. The sealing push of a near-vertical "
           "approach has a small lateral component a real table holds by static friction, "
           "which the global frictionless model rejects at ANY magnitude. Every other gate "
           "(insertion, release, rest-on-table) keeps the global table model; a held part's "
           "gripper contacts are untouched. e.g. 0.3 admits tilts up to ~17 deg (arctan mu), "
           "so the ~2deg top-down grasps pass and the sideways ones stay rejected. Default off.",
  )
  g_phys.add_argument(
      "--tf-close-torque", type=float, default=None, metavar="NM",
      help="With --wrench --grasp-model asymmetric: override the two-finger's torque cap "
           "ABOUT THE CLOSING AXIS (the line joining the two finger contacts) -- the axis a "
           "pinch resists worst, since the collinear contacts form no couple there and the "
           "part rotates out of the jaws. Default (unset) keeps it equal to T_lat (0.4 Nm); "
           "a small value (e.g. 0.05) models a grip that cannot resist out-of-hand twist.",
  )
  g_search.add_argument(
      "--handovers", action=argparse.BooleanOptionalAction, default=True,
      help="Allow add_grasp/handover primitives (regrasp-in-air). On by default; disabling "
           "them gives a leaner reference search with no second-grasp branching.",
  )
  g_search.add_argument(
      "--allow-place", "--place", dest="allow_place",
      action=argparse.BooleanOptionalAction, default=True,
      help="Allow setting a held assembly back on the table. Disabling it makes the search "
           "hold the assembly and "
           "regrasp in the air (handovers become mandatory -- the Dogar hold-and-work "
           "regime). Meaningful for the exact search; the optimistic search needs to set down.",
  )
  g_solver.add_argument(
      "--anytime", type=float, nargs="?", const=60.0, default=None, metavar="SECONDS",
      help="Anytime mode (exact search): restarting weighted A* -- a fast suboptimal plan "
           "first, then keep improving it (and possibly prove optimality) until the time "
           "budget runs out. Optional value = budget in seconds (default 60).",
  )
  g_search.add_argument(
      "--no-wrench-aware-h", dest="wrench_aware_h", action="store_false",
      help="With --wrench: build the heuristic (h_abs or --labeled-h) WITHOUT the wrench "
           "facts, for comparison. Default: wrench-aware whenever --wrench is on.",
  )
  g_phys.add_argument(
      "--bracing", action=argparse.BooleanOptionalAction, default=True,
      help="Enable the add_grasp (co-grasp) primitive: a free robot adds a SECOND gripper to "
           "an already-held assembly and KEEPS it (unlike handover, which swaps). Needed for "
           "offset HORIZONTAL mates (e.g. car wheels), where one grasp cannot supply the "
           "yaw-cancelling couple so two braces are required. Default ON since 2026-07-29 "
           "(--no-bracing removes the branching). "
           "Exact search only (solve/anytime); the labeled-h abstraction models it when on.",
  )
  g_search.add_argument(
      "--labeled-h", action="store_true",
      help="EXACT search only: use the grasp+face-LABELED disassembly abstraction as the "
           "heuristic (strictly tighter than the default abstraction; one-time build of "
           "seconds). Finite weights only (like the default abstraction).",
  )
  g_search.add_argument(
      "--anon-robots", action="store_true",
      help="LATE BINDING: search with a robot-anonymous state key -- robot-permutation "
           "states merge, so adding (idle) robots does not blow up the search. Sound for "
           "interchangeable floating grippers; with --real-ik treat the plan as a candidate.",
  )
  g_phys.add_argument(
      "--insert-force", type=float, default=10.0, metavar="N",
      help="Insertion (mating) reaction magnitude in newtons (with --wrench). Default 10. "
           "Sets what binds: e.g. two-finger caps are shear 15 / pull 25 / push 100, so at "
           "10 N held side-mates are free, at 30 N only press-aligned grasps survive.",
  )
  g_phys.add_argument(
      "--ignore-gravity", action="store_true",
      help="With --wrench: WEIGHTLESS parts -- drop the gravity load from every stability "
           "check, so a hold must only resist the INSERTION reaction (and other applied "
           "loads), not part weight. Isolates insertion/bracing feasibility from gravity.",
  )
  g_phys.add_argument(
      "--orientation-robust-wrench", action="store_true",
      help="With --wrench: every freely held component must balance gravity for EVERY rigid-"
           "body orientation, using an exact continuous contact-zonotope containment test "
           "(no SO(3) sampling and no IK). Table-resting components retain their discrete "
           "rest orientation. Conservative by design and opt-in; may force extra bracing.",
  )
  g_phys.add_argument(
      "--curved-torque-scale", type=float, default=None, metavar="S",
      help="With --wrench: derate the TORQUE (couple) a CURVED grasp (cylinder side / sphere) "
           "can resist to S x that of a flat hold (force caps unchanged). E.g. 0.25. Models "
           "'a wheel grip pivots in the jaws'; off by default (flat and curved treated alike).",
  )
  g_phys.add_argument(
      "--grasp-compliance", type=float, nargs="?", const=3.5, default=None, metavar="VACUUM",
      help="GRASP COMPLIANCE (with --wrench and --insertion-noise): inflate the insertion-"
           "uncertainty box by this factor (default 3.5) whenever a VACUUM grasp holds a body "
           "in the press's load path -- the cup deflects under contact forces, so a vacuum-"
           "held target/inserter wanders more while seating. Picks/carries are unaffected; "
           "avoiding the vacuum for tight assembly becomes EMERGENT (it may still do loose "
           "mates), not a hardcoded prohibition. Two-finger stays 1.0. Default off.",
  )
  g_phys.add_argument(
      "--insertion-noise", type=float, nargs="?", const=0.2, default=None, metavar="LATERAL",
      help="With --wrench: model MISALIGNED seating as a worst-case uncertainty BOX -- an "
           "angular tilt gives a shear up to LATERAL x insert_force transverse to the mate "
           "axis (default 0.2 when given bare), and the state must be stable at EVERY corner "
           "of the box (exact, by convexity -- no sampling). Adds an unavoidable shear/torque "
           "so a weak grasp holding far from the mate fails. Off by default.",
  )
  g_phys.add_argument(
      "--insertion-offset", type=float, default=0.0, metavar="M",
      help="With --insertion-noise: add the second (positional) box axis -- the axial press "
           "also lands off-CENTRE by up to M metres (a transverse couple = insert_force x M). "
           "0 (default) => a 4-corner lateral-only box; >0 => a 16-corner box.",
  )
  g_phys.add_argument(
      "--robust-mode", choices=("corners", "affine"), default="corners",
      help="How the insertion-uncertainty box is checked. 'corners' (default): EXACT -- every box "
           "corner must be balanceable, contacts free to rechoose their force split per "
           "realization. 'affine': one fixed affine force-response policy (an open-loop / "
           "fixed-compliance grasp) covers the LINEARIZED box -- a single LP, stricter per "
           "axis, but sound only without a positional offset; with BOTH --insertion-noise and "
           "--insertion-offset the planner falls back to 'corners' (an affine policy cannot "
           "express the tilt-x-shift cross couple).",
  )
  g_phys.add_argument(
      "--no-inserter-check", dest="inserter_check", action="store_false",
      help="With --wrench: skip the inserter-side check (whether the pressing arm's own "
           "grasp can transmit the seating reaction) -- equivalent to giving that grasp "
           "unbounded capability. The target-side stability check stays on.",
  )
  g_diag.add_argument(
      "--log-ik-failures", type=float, nargs="?", const=60.0, default=None,
      metavar="SECONDS",
      help="With --real-ik: record FAILED IK binds of the search (the skill's scene posed "
           "at the best failed attempt, labelled with action + solver residuals) and serve "
           "them in a viser browser after the solve (before the plan replay). Collection "
           "stops after SECONDS (default 60, measured from the first failure) or 300 "
           "entries, whichever first.",
  )
  g_diag.add_argument(
      "--diagnose-plan-rank", default=None, metavar="PLAN.json",
      help="Replay a known feasible symbolic plan under the current successor model and, "
           "inside each matching milestone fill, report when every demonstrated state is "
           "popped by the actual optimistic local A*: g/h/f, pop rank, lower/equal-f "
           "states ahead of it, and the demonstrated successor's local rank. IK is not "
           "re-solved by the diagnostic. The ordinary search still runs; this is an "
           "explicit, potentially expensive search-order diagnostic.",
  )
  g_search.add_argument(
      "--collapsed-h", action="store_true",
      help="Use the IDENTITY-COLLAPSED labeled disassembly heuristic (loose held parts "
           "tracked WITHOUT grasp ids). Same admissibility/completeness guarantees as "
           "--labeled-h, weaker guidance, but the build scales in #robots (the fine "
           "version explodes at 4). Mutually exclusive with --labeled-h.",
  )
  g_search.add_argument(
      "--part-order-bias", action="store_true",
      help="Plateau TIE-BREAK: among frontier nodes with equal f (and h), expand the one "
           "whose action touches the LOWEST part index first -- i.e. prefer building in the "
           "canonical part order. Pure ordering: the w-suboptimality bound and completeness "
           "are untouched; it only decides which equally-promising node to try first, which "
           "is what matters on the abstraction's large equal-h plateaus.",
  )
  g_search.add_argument(
      "--grasp-collapsed-h", action="store_true",
      help="--collapsed-h PLUS the dense-grasp LABEL collapse: wrist-tilt variants fold onto "
           "their base grasp id in the abstraction (facts OR'd over variants -- a further "
           "relaxation, still admissible). THE heuristic to pair with --dense-grasps (raw "
           "labels explode the build ~3x per robot; measured 24x on cross-4/3R). Without "
           "--dense-grasps it behaves exactly like --collapsed-h.",
  )
  g_solver.add_argument(
      "--ladder-lns", type=float, nargs="?", const=900.0, default=None, metavar="SEC",
      help="THE LADDER v2: k=1 milestones (lazy fills) -> LNS sweeps over the incumbent "
           "(segment re-fills of width 1,2 bounded by old segment cost, cascade to "
           "reconvergence -- gradual strict improvements) -> RWA*-under-bound proof rung "
           "(w=5 passes: finds tighten U fast, any EXHAUSTED pass is the optimality "
           "proof; exhaustion is weight-neutral). k=2 milestone rung runs only as a "
           "feasibility fallback. --ladder is retained unchanged as the baseline.",
  )
  g_solver.add_argument(
      "--ladder", type=float, nargs="?", const=900.0, default=None, metavar="SEC",
      help="THE LADDER: unified anytime solver -- one commitment axis (fill window k), "
           "one monotone incumbent bound U. Rungs: k=1 milestones (lazy fills, bound "
           "propagated into fills), k=2 windows, then the k=n limit = bounded exact at "
           "WEIGHT 1 (PEA*) whose first goal pop is optimal-under-U and whose exhaustion "
           "PROVES U optimal. Budget split ~35/20/45%%. Default 900s.",
  )
  g_solver.add_argument(
      "--milestones-anytime", type=float, nargs="?", const=300.0, default=None,
      metavar="SEC",
      help="Greedy milestone incumbent under a hard deadline (default 300s), then bounded "
           "large-neighborhood improvement: re-fill consecutive windows through the full "
           "milestone horizon by default (--lns-window K caps it). Each window runs greedy "
           "then bounded weight-one search under its current cost, accepts a strict "
           "improvement, and restarts the sweep. PEA-greedy net if "
           "milestones itself was cut. With --open-witness, stochastic IK failures are "
           "fairly reopened until the milestone-phase deadline instead of inheriting its "
           "finite operational epoch cap. Always retains the best validated plan. This LNS "
           "schedule is not a global optimality proof unless it reaches the admissible lower "
           "bound; use the separate ladder proof rung when a certificate is required.",
  )
  g_search.add_argument(
      "--lazy-fills", action=argparse.BooleanOptionalAction, default=True,
      help="LAZY-VALIDATED milestone/optimistic local fills (default on): fills search with the "
           "cheap gate only and run real IK on candidate subplan edges (LazySP hybrid, "
           "IK failures blacklisted + rerun) -- the measured fix for module fills "
           "spending minutes on failed IK binds at pop. --no-lazy-fills restores eager IK "
           "validation at expansion for ablations.",
  )
  g_search.add_argument(
      "--open-witness", action=argparse.BooleanOptionalAction, default=False,
      help="Fair witness policy for lazy milestone fills: give each blocking symbolic edge "
           "one small batch, temporarily suspend that edge, and explore alternatives before "
           "revisiting it in the next fairness epoch. Physically equivalent aliases share "
           "the effort cursor and therefore use fresh rather than repeated attempt slots. "
           "Failed probes remain OPEN and are not stored in "
           "the legacy permanent IK/edge blacklist. Enabled by --search-profile standard; "
           "the no-profile built-in remains off for historical command compatibility.",
  )
  g_search.add_argument(
      "--open-witness-epochs", type=int, default=2, metavar="N",
      help="With --open-witness: maximum reopen/probe epochs per milestone fill (default 2). "
           "This is an effort limit, not an infeasibility claim. 0 means unlimited and is "
           "accepted only when the calling solver supplies a finite deadline.",
  )
  g_search.add_argument(
      "--open-witness-batch", type=int, default=7, metavar="N",
      help="With --open-witness: maximum bounded IK batch (default 7, must be >=1). "
           "Ordinary skills escalate through min(N,2), min(N,4), then N attempts. Assemble "
           "keeps one atomic N=7 structural bundle (six orientation biases + one unbiased "
           "initialization) because splitting it was slower. A failed incumbent is abandoned "
           "immediately; later edges are not speculatively probed.",
  )
  g_search.add_argument(
      "--open-witness-positive-order", action=argparse.BooleanOptionalAction, default=False,
      help="With --open-witness: use SUCCESSFUL solve-scoped physical evidence only to order "
           "equal-cost successors: exact certified edges first, then compatible co-grasp/"
           "handover sets, then grasps previously reached by the same robot. Pure stable "
           "ordering; unseen ties retain the original order and failures never prune or "
           "penalize an action. Experimental and disabled by the named profiles.",
  )
  g_search.add_argument(
      "--lazy-edge-selector", choices=("forward", "terminal-first", "reverse", "fail-first",
                                        "defer-failed"),
      default="forward", metavar="POLICY",
      help="IK-edge selection inside an optimistic lazy-fill candidate: forward preserves "
           "the historical prefix-first order; terminal-first certifies the candidate's "
           "last concrete assemble mode, then validates its prefix forward; reverse validates "
           "the whole path suffix-first; "
           "fail-first validates previously failed exact (action, scene) queries first and "
           "otherwise falls back to reverse; defer-failed scans the whole candidate and gives "
           "each physical query at most one failed effort batch per fairness epoch, temporarily "
           "deferring later symbolic aliases until the next epoch. These policies change only "
           "validation scheduling, never costs or failure semantics. Standard keeps measured "
           "forward order.",
  )
  g_search.add_argument(
      "--lns-segments", action="store_true",
      help="LNS neighborhood unit = SETTLED segments instead of the default OBJECT unit "
           "(k = consecutive mated parts). Baseline/ablation; segments coarsen unevenly "
           "under hold-through re-fills.",
  )
  g_search.add_argument(
      "--fill-window", type=int, default=2, metavar="K",
      help="Milestone fill window escalation limit (default 2): on width-1 failure, try "
           "JOINT fills of 2..K parts (escalating only as each smaller width exhausts) "
           "before backjumping -- covers interleaved-glue plans, the milestone "
           "commitment's completeness gap; K=n would be the flat exact search. "
           "1 disables windows.",
  )
  g_search.add_argument(
      "--lns-window", type=int, default=None, metavar="K",
      help="Maximum consecutive-milestone neighborhood for --milestones-anytime. "
           "The default is the full incumbent horizon. This is independent of "
           "--fill-window, which controls joint-fill recovery while finding the initial "
           "milestone plan.",
  )
  g_search.add_argument(
      "--lns-grasp-transition-h", action="store_true",
      help="Experimental --milestones-anytime heuristic: in each loose part's grasp graph, "
           "retain a handover edge only if two available floating grippers can coexist at "
           "concrete representatives of its grasp classes. Forward search and named profiles "
           "are unchanged.",
  )
  g_search.add_argument(
      "--milestone-oracle", choices=("global", "chain", "structural"), default="global",
      help="Milestone-ordering ablation (flag default global; --search-profile standard "
           "selects structural). 'global' builds the backward "
           "signature database and uses its cost-to-go plus grasp/brace hints. "
           "'chain' builds the smaller signature projection over assembly_order only "
           "(sequence assemblies; detached modules are rejected). "
           "'structural' skips that database completely: dependency/module eligibility "
           "and canonical part order choose milestones. Its fill heuristic is selected "
           "independently (--fill-local-h in the standard profile). Backtracking, adaptive "
           "fill windows, lazy IK, and final settling are unchanged.",
  )
  g_search.add_argument(
      "--fill-local-h", action=argparse.BooleanOptionalAction, default=False,
      help="Guide every one-part fill with a small backward pattern database whose goal is "
           "that fill's own settled endpoint. Every grasp role uses the same fact-signature "
           "projection while the abstraction tracks "
           "support face/holders, free tool capacity and wrench facts, but ignores robot "
           "joint configurations. Independent of sequence/DAG/module structure. Standard "
           "on; --no-fill-local-h is the counting-only guidance ablation.",
  )
  g_search.add_argument(
      "--fill-raw-grasp-h", action=argparse.BooleanOptionalAction, default=False,
      help="Local-PDB projection ablation: keep raw grasp ids in G and L instead of "
           "mapping them to fact-signature classes. Milestones, concrete greedy fills, "
           "the action space, and IK validation remain unchanged. Requires --fill-local-h.",
  )
  g_search.add_argument(
      "--fill-target-support-quotient", action=argparse.BooleanOptionalAction, default=False,
      help="Experimental no-wrench local-PDB quotient: replace grasp identities on the "
           "already-built target component by anonymous holder counts, preserving tool-kind "
           "masks and keeping mover grasps labeled. This removes target-size growth that is "
           "irrelevant when wrench and geometric facts are disabled. Requires "
           "--fill-local-h --no-wrench; not enabled by a profile.",
  )
  g_search.add_argument(
      "--fill-exact", action=argparse.BooleanOptionalAction, default=False,
      help="Run one-part concrete fill searches at weight 1, irrespective of the global "
           "--weight. Also removes the ordinary 200-plan lazy-repair cutoff; node/deadline "
           "limits still apply. IK remains lazy and fully validated. Enabled together with "
           "--fill-local-h for exact-fill ablations.",
  )
  g_search.add_argument(
      "--fill-greedy", action=argparse.BooleanOptionalAction, default=False,
      help="Order one-part lazy fills by h only (greedy best-first), independently of the "
           "global --weight. Validation still repairs the finite frontier until a solution, "
           "node/deadline limit, or exhaustion; it is not a pruning rule. This is the "
           "time-to-first-plan policy used by --search-profile standard. Exact improvement "
           "belongs to --milestones-anytime.",
  )
  g_search.add_argument(
      "--fill-incumbent-first", action=argparse.BooleanOptionalAction, default=False,
      help="Experimental anytime ordering for exact lazy milestone fills: first find a fully "
           "validated incumbent with greedy best-first search, then run the usual weight-one "
           "fill search with g+h bounded by the incumbent cost. Witness successes and "
           "open-failure effort are reused. Requires --fill-local-h, --fill-exact and "
           "--lazy-fills; not yet enabled by a profile.",
  )
  g_search.add_argument(
      "--fill-policy-first", action=argparse.BooleanOptionalAction, default=False,
      help="Before an unrestricted local fill, refine the shortest paths of its exact "
           "goal-conditioned PDB using ordinary concrete successors and the ordinary "
           "cheap/IK/open-witness gates. Only quotient-changing Bellman-optimal abstract "
           "transitions are admitted; there is no hand-written action or length bound. A "
           "failed/spurious abstract policy falls through to the complete local fill. "
           "Requires --fill-local-h. Experimental and not enabled by a profile.",
  )
  g_search.add_argument(
      "--fill-policy-guide", action=argparse.BooleanOptionalAction, default=False,
      help="After the normal compact attempt (and, with --grasp-local-widen, one numeric "
           "local widening) fails, use a separate full-master reverse PDB for a non-pruning "
           "additive rescue lift. Existing prefixes remain active; a breadth-first traversal "
           "of all settled Bellman-optimal branches schedules one bounded stable-ID batch per "
           "hidden grasp-signature fibre (mover or support), then each requested stable-rest "
           "yaw orbit. Failed probes retain prior coordinates and advance to another branch; "
           "normal forward IK remains authoritative and the ordinary grasp/yaw/scope recovery "
           "schedule remains available. Requires "
           "--fill-local-h; "
           "experimental and not enabled by a profile. It never performs backward expensive "
           "IK validation: the configured lazy validation order remains in force. Requires "
           "--milestones, --fill-local-h, and either --grasp-local-widen or "
           "--placement-yaw-widening.",
  )
  g_search.add_argument(
      "--fill-skeleton-first", action=argparse.BooleanOptionalAction, default=False,
      help="EXPERIMENTAL reviewer baseline: before each ordinary local fill, probe a "
           "small direct manipulation family (one mover pick, one target lift, one handover, "
           "one target brace, one target placement). Candidate paths use the same wrench, "
           "cheap-collision and lazy real-IK gates. At most three role assignments per robot "
           "are tried before falling through explicitly to the existing complete local graph; "
           "the probe never declares infeasibility.",
  )
  g_search.add_argument(
      "--fill-action-widening", action=argparse.BooleanOptionalAction, default=False,
      help="EXPERIMENTAL non-pruning action-family schedule for each local fill. Search the "
           "same concrete graph first with direct transport only, then with handovers, and "
           "finally with the configured placement/bracing actions. Every candidate uses the "
           "ordinary wrench, collision and real-IK gates; restricted-family failure always "
           "falls through to the existing complete fill and shared IK evidence is retained.",
  )
  g_search.add_argument(
      "--milestone-boundaries", type=int, default=1, metavar="K",
      help="Retain up to K distinct settled concrete endpoints per milestone transition "
           "(default 1; standard profile 4). Alternatives are generated lazily only when a "
           "later suffix fails: the rejected endpoint is excluded and the same exact fill is "
           "resumed/re-solved for its next-cheapest boundary. Applies recursively inside "
           "detached modules. K=1 is the historical claim-once ablation.",
  )
  g_search.add_argument(
      "--corridor-h", action="store_true",
      help="With --bounded: cap the phase-2 heuristic's backward Dijkstra at the "
           "incumbent's cost U (beyond-cap states never pushed; miss => h=inf, sound "
           "under the bound). MEASURED INEFFECTIVE: the abstract spaces are WIDE at "
           "shallow depth (R=4 cross still >2M states at cap 35; 2x2x3 table unchanged "
           "at 204k), and the pruning is redundant with cost_bound + exact h values. "
           "Harmless; kept for depth-heavy instances. The real version of the idea needs "
           "a TWO-SIDED corridor (forward+backward abstract dist <= U) to cut width.",
  )
  g_search.add_argument(
      "--pathmax", action="store_true",
      help="PATHMAX consistency repair in the exact engines: h(child) >= h(parent) - "
           "edge cost (triangle inequality => still admissible; all guarantees "
           "unchanged). MEASURED NEUTRAL: pathmax only RAISES h after too-fast drops -- "
           "it cannot touch the h-STALL steps, which are the actual f-bumps (sanities "
           "bit-identical; cube-plates-2x2x3 unchanged). Kept as a safe no-cost option.",
  )
  g_solver.add_argument(
      "--pea", action="store_true",
      help="PARTIAL-EXPANSION A* engine: identical guarantees and ordering to the exact "
           "search, but the open list holds only each node's current f-BAND -- successors "
           "above it are regenerated later instead of stored. Frontier memory / ~branching "
           "factor, for long horizons where the plain open list OOMs (compute-for-memory "
           "trade; succ/h are cached). Composes with --bounded (phase-2 engine).",
  )
  g_solver.add_argument(
      "--iter-bound", action="store_true",
      help="ITERATIVE COST-BOUND DEEPENING: run exact passes at cost_bound = h(s0)+1, +2, "
           "... until a plan appears -- the FIRST plan found is PROVEN OPTIMAL (each "
           "exhausted pass is a lower-bound proof; the inner --weight only orders "
           "exploration inside each pass). Per-pass memory = that bound's sublevel set, "
           "the minimum any optimality proof needs. Combine with --pea for the smallest "
           "footprint. Integer action costs assumed (step 1).",
  )
  g_solver.add_argument(
      "--beam", type=int, default=None, metavar="N",
      help="BEAM SEARCH engine (width N): keeps <= N states per depth level (memory-"
           "bounded). MEASURED WORSE THAN GREEDY on the long-horizon benchmark "
           "(cube-plates-2x2x3: greedy cost 33/63 exp vs beam-8 cost 43 at w=inf, 59 at "
           "w=5, beam-1 cost 116 -- greedy's GLOBAL frontier backs up where a beam must "
           "commit, and its frontier stays tiny anyway). Kept as the LAST-RESORT "
           "incumbent fallback in --bounded (fires only if greedy also fails) and for "
           "experimentation.",
  )
  g_solver.add_argument(
      "--dfbnb", action="store_true",
      help="DEPTH-FIRST BRANCH-AND-BOUND certificate engine (implies --bounded): linear "
           "memory + LRU transposition table, anytime. MEASURED SLOWER than the bounded "
           "A* it replaces (cube-plates-2x2x3, same bound 33: A* PROVES in 192s floating "
           "/ ~1000s real-ik; dfbnb exhausts its node budget unfinished at 350s / times "
           "out at 1500s -- DFS re-enters the interleaving lattice that A*'s claim-at-pop "
           "kills globally). Use only if the A* open list truly cannot fit memory.",
  )
  g_scene.add_argument(
      "--goal-on-table", action=argparse.BooleanOptionalAction, default=True,
      help="GOAL: require the finished assembly to REST ON THE TABLE and be released "
           "(hands-off), not merely complete-in-air. Default goal is complete (held "
           "counts). Extensible to a specific target pose later (set_goal(pose=...)).",
  )
  g_witness.add_argument(
      "--optimize-witnesses", nargs="?", const="joint", default="floor",
      choices=["joint", "ee", "dp-joint", "dp-ee", "floor", "off"], metavar="MODE",
      help="POST-optimize the plan's IK witnesses (discrete plan FIXED): re-solve each IK "
           "action pulled toward the ACTIVE arm's previous configuration, minimizing "
           "per-step movement in 'joint' space (default) or 'ee' space (position + "
           "weighted orientation). dp-joint/dp-ee: LADDER-DP -- harvest multiple feasible "
           "candidates per step (assemble orientation biases, fresh re-solves, the greedy "
           "pass) and pick the globally optimal combination by per-robot Viterbi "
           "'floor': CLEARANCE-FLOOR ladder -- clearance becomes a per-step FLOOR (a hard "
           "filter at alpha * the best clearance available at that step) and among the poses "
           "that clear it the chain minimising IK BRANCH CROSSINGS wins. Unlike the weighted "
           "dp-* modes the two objectives cannot trade; the floor is raised as high as is FREE "
           "(costs no extra crossings) and the whole front is printed. Multi-arm steps "
           "(handover/add_grasp/braced assemble) are JOINT nodes, clearance sees the other arms "
           "at their carried configs rather than at home, and tool roll is treated as a "
           "symmetry rather than a branch. "
           "(DP >= greedy >= original by construction). Feasibility-safe (search "
           "witnesses kept on any failure); idle arms stay home-pinned as in the search. "
           "Requires --real-ik. Default 'floor' since 2026-07-29; pass 'off' to disable.",
  )
  g_witness.add_argument(
      "--witness-clearance", type=float, default=0.0, metavar="W",
      help="Per-step CLEARANCE weight for --optimize-witnesses (0=off, ~1.0 works). "
           "Among candidate witnesses, prefer configs that keep the active arm clear of "
           "the OTHER robots and the TABLE (parts excluded -- intended contact). Score = "
           "SUM over arm-link pairs of min(gap, 0.4m) via C.eval(FS.distance) (sum not "
           "min: the gripper is pinned near the object; the cap ignores already-far "
           "pairs). Traded against movement by this weight. Best with the dp-* modes.",
  )
  g_witness.add_argument(
      "--witness-sets", type=int, default=4, metavar="K",
      help="WITNESS SETS: export up to K same-grasp IK-branch alternatives per task (member 0 "
           "= the canonical witness) as mr_bench GoalSets, so the motion planner gets several "
           "diverse keyframes per transition instead of one. Alternatives come from the "
           "witfloor candidate pool, pass the same gates as the canonical witness, keep every "
           "grasped part at the canonical world pose (handover/add_grasp diversity = arm "
           "branches only), and are selected greedy-max-min on wrist-joint spread. Default 4; "
           "0/1 = OFF. Requires --optimize-witnesses floor.")
  g_witness.add_argument(
      "--witness-set-spread", type=float, default=0.05, metavar="RAD",
      help="--witness-sets: minimum wrist-joint distance between accepted members (rad); "
           "selection stops below it.")
  g_witness.add_argument(
      "--witness-draw", type=int, default=None, metavar="K",
      help="DIVERSITY draw for the witness ladder (bandit outer loop): seed K thins the "
           "witfloor candidate ladder with a private RNG before the DP, so different draws "
           "ship locally-good but globally DIFFERENT witness chains (covers grasp-side/"
           "homotopy families that cost-ranking alone never picks). None = deterministic "
           "ladder (byte-identical).",
  )
  g_witness.add_argument(
      "--witness-margin", type=float, default=0.005, metavar="M",
      help="SHARED SEARCH+WITNESS hard clearance: reject every IK whose posed scene has a "
           "non-intended arm/part, inter-robot, or non-adjacent self-collision pair closer "
           "than M, plus payload/table clearance at free-pose handover/add_grasp skills. "
           "Only the acting hand/target and each holder's own payload are exempt (plus other "
           "part/part and table pairs). At coupled endpoints, task-participant hands and the "
           "acting hand versus the rigid mate-target group retain strict collision but not "
           "the free-space margin. The SAME gate runs during search and every "
           "later witness/fusion/funnel solve, so post-processing is never stricter than "
           "search. Default 0.005m; 0 disables.",
  )
  g_witness.add_argument(
      "--witness-joint-margin", type=float, default=None, metavar="RAD",
      help="SHARED SEARCH+WITNESS joint-limit interior margin for coupled handover, add_grasp, "
           "and assemble endpoints. Reject a selected joint closer than RAD to a hardware "
           "limit, so the motion planner receives an open local neighbourhood instead of a "
           "zero-measure boundary goal. Ordinary pick/place IK is unchanged. The standard "
           "profile uses 0.02 rad; unset/0 disables. Arm-less --floating-ee pose coordinates "
           "are not hardware joints and are exempt.",
  )
  g_witness.add_argument(
      "--object-over-table", action=argparse.BooleanOptionalAction, default=False,
      help="Keep the reference datum of each held rigid component moved by the current skill "
           "inside the tabletop's local x/y "
           "footprint at every real-IK witness. The same bounds are hard KOMO inequalities "
           "and a post-solve certificate; table-supported parts and payloads on uninvolved "
           "arms parked at the isolated IK EXIT pose are unchanged. This is an "
           "endpoint/workcell constraint, not a constraint on every motion-planning sample.",
  )
  g_witness.add_argument(
      "--object-over-table-margin", type=float, default=0.0, metavar="M",
      help="Signed --object-over-table boundary margin in metres (default 0): positive "
           "values inset the table footprint; negative values permit bounded overhang. "
           "The bound applies to each held component's reference datum, not its complete "
           "collision geometry.",
  )
  g_witness.add_argument(
      "--witness-branch-budget", type=int, default=5, metavar="N",
      help="--optimize-witnesses floor: permit N extra IK BRANCH CROSSINGS in exchange for a "
           "higher clearance floor. 0 takes only the clearance that is FREE -- the "
           "highest floor achieving the same crossing count as no floor at all. Default 5 "
           "since 2026-07-29.",
  )
  g_search.add_argument(
      "--ik-accuracy", type=float, default=None, metavar="TOL",
      help="NLP stopTolerance for every IK solve. rai's DEFAULT IS 0.01 -- 10mm -- so an "
           "eq-constrained mate is declared solved while MILLIMETRES out (measured: every "
           "cross-12 mate ~4.5mm short along its own insertion axis). mr_bench bakes that error "
           "in at attach, so the exported assembly disagrees with the scene the witness was "
           "solved in and the acting arm collides with an already-mated part. 1e-3 gave 0.012mm "
           "residuals on car-mini (100x better) for ~3x solve time; TIGHTER IS WORSE (1e-5/1e-6 "
           "regress -- the solver hits its eval caps before the augmented-Lagrangian schedule "
           "finishes).",
  )
  g_search.add_argument(
      "--grasp-translations", type=int, default=0, metavar="N",
      help="Add N TRANSLATED variants per base grasp: the same pinch SLID along the directions "
           "the gripper is open in (the tool's local x and z -- across the grasped face and "
           "deeper/shallower). Sliding along the CLOSING axis would open the pinch, so it is "
           "excluded. Travel is bounded by the part's half-extent minus the finger-pad half-"
           "length (20mm, robotiq_tight.g) so the pad still overlaps the part; a part too small "
           "to slide on keeps the centred grasp. Densifies the grasp set for re-orientations "
           "the six face-centred grasps cannot reach. Appended AFTER the existing ids.",
  )
  g_search.add_argument(
      "--tight-fingers", action=argparse.BooleanOptionalAction, default=True,
      help="Default ON (2026-07-29). Shrink the finger COLLISION proxy from robotiq.g's capsule (40mm long, 20mm RADIUS "
           "= 80mm end to end) to robotiq_tight.g's (10mm radius), shifted 8mm outward so the "
           "gripping surface stays put. The visible robotiq pads carry contact:0 -- this proxy "
           "is what every grasp and collision check actually uses, and it reaches 32.9mm past "
           "the tool centre, so it pokes out the far side of small parts and is what forces "
           "--part-float. Applied to the real scene, the cheap-EE scenes and the grasp "
           "precompute alike (they must agree). Changes which grasps survive the DB filter.",
  )
  g_search.add_argument(
      "--grasp-closure-check", action=argparse.BooleanOptionalAction, default=True,
      help="Default ON. Collision-only virtual close certificate for two-finger "
           "pick/handover/add_grasp "
           "and for retained closed hands after assembly. "
           "Finger joints stay DISABLED: a temporary proxy sweeps from the fixed-open model "
           "to the object-dependent bilateral-contact aperture and rejects a sweep through "
           "another gripper, part, table, or arm. Vacuum is unchanged. Saved with plans; "
           "--no-grasp-closure-check reproduces legacy fixed-open semantics.",
  )
  g_search.add_argument(
      "--grasp-physics-check", action=argparse.BooleanOptionalAction, default=False,
      help="Experimental sampled two-finger grasp certificate: in an isolated MuJoCo world, "
           "close the exported gripper's actual articulated finger joints on each candidate "
           "and require bilateral contact plus retention under gravity and six axial test "
           "loads. Scene collision/IK "
           "remain separate, and accepted grasps all retain the same configured analytic "
           "wrench envelope (no per-grasp force fitting). Requires --grasp-sampler and "
           "--grasp-sampler-scenes. Saved with plans because it changes the grasp DB.",
  )
  g_search.add_argument(
      "--grasp-physics-force", type=float, default=10.0, metavar="N",
      help="Magnitude of each isolated grasp-qualification load in +/- tool x/y/z "
           "(default: 10N). Gravity remains active during every rollout.",
  )
  g_search.add_argument(
      "--place-eq-weight", type=float, default=100.0, metavar="W",
      help="Weight on a PLACE's object-pose EQUALITY constraints (default 1e2 since "
           "2026-07-29; rai's own default is 1e1) -- the twin of "
           "--mate-eq-weight, which only reaches _ik_assemble. A place sets the DATUM every "
           "later mate is measured against, so its residual propagates: MEASURED car-mini "
           "(seed 11019) a body placed 3.91mm high while every wheel mated onto it at "
           "0.01-0.23mm, i.e. perfectly seated on a body in the wrong place. NOTE it cannot "
           "buy accuracy the geometry does not allow -- where the exact pose needs the arm "
           "inside the table, a higher weight turns the compromise into a hard IK failure "
           "(arguably the honest outcome; pair with a looser --strict-collisions). Made the "
           "default after FRAGILITY 2.10: a 2.43mm-low place datum sank a mated part below "
           "the table for every later mode and walled BiRRT's transition sampler.",
  )
  g_search.add_argument(
      "--quiet-rai", action=argparse.BooleanOptionalAction, default=False,
      help="Drop rai's libccd/pairCollision WARNING spam ('called MPR penetration for non "
           "intersecting meshes', 'but GJK says intersection'). rai prints those from C++ "
           "straight to the file descriptor, so a Python redirect cannot see them -- this "
           "splices a filter onto fd 1/2. Matches the SOURCE FILE, never the word WARNING, so "
           "real diagnostics survive. Equivalent: export RAI_QUIET=1. A profile-enabled "
           "value can be undone with --no-quiet-rai (unless the environment variable already "
           "installed the process-wide filter before argument parsing).",
  )
  g_search.add_argument(
      "--mate-eq-weight", type=float, default=100.0, metavar="W",
      help="Weight on the assemble's positionRel/quaternionRel EQUALITY constraints (default "
           "1e2 since 2026-07-29; rai's own is 1e1). rai's augmented-Lagrangian schedule trades the eq residual against the other "
           "objectives, so at 1e1 the mate settles millimetres short and rai still reports "
           "feasible. MEASURED car-mini: 1e2 gives 1.337mm -> 0.026mm AND is slightly faster; "
           "1e3 is no better; 1e4 starts failing solves. Asking for accuracy up front beats "
           "rejecting after the fact (--mate-accuracy cost ~90x search time).",
  )
  g_search.add_argument(
      "--mate-accuracy", type=float, default=None, metavar="M",
      help="ACCEPTANCE CRITERION for an assemble: reject a witness whose achieved "
           "positionRel(part, onto) misses the mate by more than M and let the solver try "
           "another bias/restart. rai calls an eq constraint 'feasible' while it is still "
           "MILLIMETRES out and --ik-accuracy does NOT change that (measured: the same 4.5mm "
           "mate at stopTolerance 1e-2 and 1e-3), so this is the only way to refuse an "
           "under-seated mate. mr_bench bakes the miss in at attach, which is what makes the "
           "exported assembly disagree with ours and the acting arm hit an already-mated part.",
  )
  g_search.add_argument(
      "--mate-angular-accuracy", type=float, default=None, metavar="DEG",
      help="ACCEPTANCE CRITERION for an assemble's quaternionRel, in degrees. This is the "
           "rotational counterpart of --mate-accuracy: reject a witness whose achieved "
           "relative orientation misses the authored mate by more than DEG. Particularly "
           "important for long, tight insertion features, where a small angular residual "
           "becomes contact after the achieved attachment transform is accumulated.",
  )
  g_search.add_argument(
      "--grasp-eq-weight", type=float, default=None, metavar="W",
      help="Weight of the NEW-grasp tool-pose eq in pick/add_grasp IK (rai default 1e1). "
           "Raise to 100 (like --mate-eq-weight) so the solve LANDS ON the grasp-DB pose "
           "instead of stopping millimetres off when collision costs push back -- the "
           "attach freezes the achieved relation, see --grasp-accuracy.",
  )
  g_search.add_argument(
      "--grasp-accuracy", type=float, default=None, metavar="M",
      help="ACCEPTANCE CRITERION for a pick/add_grasp: reject a witness whose achieved "
           "tool-in-part pose misses the grasp-DB pose by more than M and let bind try the "
           "next candidate. mr_bench's attach and the sim's weld FREEZE the ACHIEVED "
           "relation while later witnesses assume the DB relation, so slack here becomes a "
           "rigid part-in-hand error (measured ~5mm on sampled grasps; with a HELD mate "
           "base the two picks' residuals differ and the difference lands in the seat, "
           "which mr_bench cannot exempt).",
  )
  g_search.add_argument(
      "--strict-mate-collisions", action="store_true",
      help="Gate assemble witnesses on collisions the moving part makes with everything OTHER "
           "than its mate target. The IK must disable that group's contact to converge on a "
           "flush mate, which also hides part-vs-arm and part-vs-part -- so the search accepts "
           "witnesses the motion planner rejects at the same tolerance. Needs "
           "--strict-collisions for the threshold.",
  )
  g_search.add_argument(
      "--witness-branch-metric", choices=("sign", "motion"), default="sign",
      help="--optimize-witnesses floor: how a BRANCH CROSSING is counted. 'sign' (default) "
           "labels by (elbow, wrist_2) sign -- 4 of the arm's 8 analytic branches, blind to the "
           "SHOULDER flip (measured: two same-label candidates 2.414 rad apart on pan). "
           "'motion' instead counts joints that must travel > --witness-reconfig-thresh along "
           "their shortest path, catching shoulder/elbow/wrist flips alike.",
  )
  g_search.add_argument(
      "--witness-reconfig-thresh", type=float, default=float(np.pi / 2.0), metavar="RAD",
      help="Per-joint travel above which --witness-branch-metric motion calls it a crossing "
           "(default pi/2).",
  )
  g_search.add_argument(
      "--inhand-reorient", action="store_true",
      help="Enable the IN-HAND REORIENT macro action: a robot rotates its held LONE part to "
           "any other grasp WITHOUT place+pick (kills table-regrasp cycles). V1 assumptions: "
           "full in-hand connectivity (any grasp -> any grasp), unchanged grip capability "
           "during the transition, lone solo-held parts only. The labeled/collapsed "
           "heuristics MODEL this edge (inverse-reorient, cost 1) -> h stays admissible; "
           "bounded/exact optimality proofs remain valid.",
  )
  g_phys.add_argument(
      "--brace-resting", action="store_true",
      help="With --bracing: allow add_grasp on a TABLE-RESTING assembly (steadying it "
           "against a press), not only on held ones. Physically routine; without it a "
           "released brace on a resting group is IRREVERSIBLE, which is exactly why "
           "single milestone fills fail where windows (that never release) succeed on "
           "press-heavy configs. Default off.",
  )
  g_search.add_argument(
      "--useful-grasps", action=argparse.BooleanOptionalAction, default=False,
      help="SPEC 6.2 (reduced): compute each part's MATE-VIABLE grasp set U[p] (collision-"
           "free at the op x press-transmitting per kind, wrench-aware) and (a) ORDER every "
           "pick/handover candidate U-first -- the transit gradient -- and (b) price a held "
           "non-U grasp at +1 in the milestone fill h (the forced regrasp, admissible). "
           "Ordering + pricing only; nothing is removed -- completeness untouched. Enabled "
           "by --search-profile standard; --no-useful-grasps is the matched ablation.",
  )
  g_search.add_argument(
      "--fill-scope-widening", action=argparse.BooleanOptionalAction, default=False,
      help="After every target/current-assembly fill phase is exhausted, retry the same "
           "grasp/bracing widening with manipulation of any loose part. Standard on: a local-"
           "scope failure is not treated as milestone infeasibility until this completeness "
           "widening also fails. --no-fill-scope-widening is the locality ablation.",
  )
  g_search.add_argument(
      "--fill-touch-all", action=argparse.BooleanOptionalAction, default=False,
      help="Diagnostic: start every fill with all-part manipulation instead of first trying "
           "the cheap local scope. Unrelated future parts may move but may not assemble, and "
           "must be settled at the endpoint. Standard off; ordinary completeness widening is "
           "controlled by --fill-scope-widening.",
  )
  g_search.add_argument(
      "--dominance", action=argparse.BooleanOptionalAction, default=False,
      help="DOMINANCE pruning (spec 8.3, 'same node modulo free robots'): among currently-"
           "FREE arms of the same gripper kind, only the lowest-index one acts (idle twins "
           "are permutation-symmetric). Kills the x2-x3 idle-twin branching of pick/brace/"
           "handover transits. SOUND for floating EEs (identical arms); with REAL fixed-base "
           "arms the twins sit at different table sides and the pruning may cost plans "
           "(a warning is printed). Default off.",
  )
  g_phys.add_argument(
      "--allow-curved-grasps", action="store_true",
      help="Let a VACUUM keep CURVED-surface grasps (normally dropped -- a suction cup cannot "
           "seal on a cylinder/sphere side, so only the flat end-caps survive). Idealised knob "
           "to test whether curved-side grasps make thin cylinder parts (e.g. the chair legs) "
           "graspable by the vacuum. No effect on the two-finger (it keeps curved grasps "
           "already). Must be set before the grasp table is built (it is). Default off.",
  )
  g_phys.add_argument(
      "--staged-mates", action="store_true",
      help="Enable STAGED/PARTIAL mates (A11): a mate listed in the assembly's staged_mates "
           "({part: fastener}) is only a SLOT (lateral walls + seat, no couples, no extraction "
           "resistance) until its fastener is mated into the same group -- so an inserted-but-"
           "unpinned part falls out unless a hand stays on it or the assembly rests slot-"
           "opening-up. Enforced as a successor filter (every action gated); the labeled/"
           "collapsed heuristics model the same invariant -> h stays admissible, and the "
           "hold-until-fastened behavior is DISCOVERED, not scripted. No new actions; the "
           "fastener's own assemble upgrades the mate to a weld implicitly. Assemblies "
           "without staged_mates (all but slot-pin) are unaffected. Default off (byte-"
           "identical behavior).",
  )
  g_search.add_argument(
      "--mate-symmetry", action=argparse.BooleanOptionalAction, default=False,
      help="Enable finite, assembly-authored sets of functionally equivalent mate "
           "orientations. One symbolic assemble edge tries the discrete relative "
           "quaternions in IK; the successful realization is retained by its witness. "
           "Currently authored for the real-stool/yandv-stool legs (half turns), the "
           "Fabrica Gamepad's square pins (quarter turns), and the Fabrica Circular Stool's "
           "hexagonal pegs/square caps (six/four axial orientations). "
           "Default off for exact compatibility.",
  )
  g_phys.add_argument(
      "--mate-wrench-limits", action=argparse.BooleanOptionalAction, default=True,
      help="Honor assembly-authored finite wrench-transmission limits at already-seated "
           "mates. Ideal internal mates are contracted; each finite interface gets an "
           "equal-and-opposite bounded wrench in a per-rigid-node equilibrium LP. Default "
           "on, but assemblies without finite caps use the historical one-body LP exactly. "
           "Use --no-mate-wrench-limits for the ideal-weld baseline/ablation.",
  )
  g_search.add_argument(
      "--lru-robots", action="store_true",
      help="Order robots LEAST-RECENTLY-USED first in successor generation (utilization "
           "knob): the greedy/milestone searches take the first feasible successor, so the "
           "fixed roster order makes a1 win every tie and idles the rest. LRU alternation "
           "spreads actions across arms -- with the depgraph export that IS parallelism "
           "(different robots on independent parts stay unordered for the motion planner). "
           "Deterministic (no RNG, stable ties); pure successor ordering, search space "
           "unchanged. May cost a few extra actions (forced alternation).",
  )
  g_skills.add_argument(
      "--screw-angle", type=float, default=None, metavar="DEG",
      help="Override the signed rotation angle (degrees) of every screw mate authored by the "
           "assembly. The default comes from the assembly (peg-hole-screw: 180deg). This "
           "refines one atomic assemble action; it never adds task-level regrasp actions.",
  )
  g_skills.add_argument(
      "--screw-depth", type=float, default=None, metavar="M",
      help="Override screw axial travel in metres (peg-hole-screw default 0.020).",
  )
  g_skills.add_argument(
      "--screw-torque", type=float, default=None, metavar="NM",
      help="Override the pure fastening couple used by target and inserter wrench checks.",
  )
  g_skills.add_argument(
      "--screw-speed", type=float, default=None, metavar="RAD_S",
      help="Override nominal angular speed stored in the execution certificate.",
  )
  g_skills.add_argument(
      "--screw-samples", type=int, default=None, metavar="N",
      help="Override the number of warm-started IK intervals used to certify the continuous "
           "single-grasp helix (minimum 2; default 16). Intermediate samples are execution "
           "data, not symbolic actions.",
  )
  g_skills.add_argument(
      "--pre-insertion", type=float, nargs="?", const=0.05, default=0.05, metavar="DELTA",
      help="PRE-INSERTION funnel: split every "
           "`assemble` into a GOTO to the funnel-mouth pose T_pre (the mate witness backed off "
           "DELTA metres along the mate axis, default 0.05) followed by the REAL mate at the "
           "seated witness. The motion planner then connects the free-space approach to T_pre "
           "and only the short T_pre->seated seating motion is contact-rich -- on the mr_bench "
           "SKILLS branch that second task becomes the insertion skill. Search accepts a seated "
           "witness only if that same branch realizes the full DELTA; there is no shorter "
           "fallback. The mate pose, wrench gate, and symbolic action graph are untouched. "
           "Default 0.05 since 2026-07-29 (FRAGILITY 2.2: "
           "without it the held part arrives laterally, already in contact); "
           "0 = single un-split assemble.",
  )
  g_skills.add_argument(
      "--skill-funnel-collision-res", type=float, default=None, metavar="RAD",
      help="SEARCH/WITNESS certificate for each assemble funnel: sample the full held-object "
           "joint interpolation from the pre-insertion keyframe to the seated witness at "
           "most RAD per joint, rejecting any non-intended collision before motion planning. "
           "The standard search profile uses 0.002rad; unset/0 keeps endpoint-only behavior.",
  )
  g_skills.add_argument(
      "--pre-pick", type=float, nargs="?", const=0.05, default=0.05, metavar="DELTA",
      help="Like --pre-insertion, for PICK (EXPORT/REPLAY only): split each pick into a GOTO to "
           "the pre-grasp pose (gripper retracted DELTA m along its -approach axis, default "
           "0.05) + the real grasp at the pick witness. Default 0.05; 0 disables.",
  )
  g_skills.add_argument(
      "--post-pick", type=float, nargs="?", const=0.05, default=0.05, metavar="DELTA",
      help="Complete PICK's departure side: after the grasp attaches, add a GOTO to the "
           "table support normal (+world z) for table picks, with the payload attached; "
           "other contexts retain the paired -approach-axis endpoint. The motion planner "
           "therefore certifies the held-object retreat instead of the simulator injecting "
           "a transient lift that the nominal reference undoes. Default 0.05; 0 disables.",
  )
  g_skills.add_argument(
      "--pre-place", type=float, nargs="?", const=0.05, default=0.05, metavar="DELTA",
      help="Like --pre-insertion, for PLACE (EXPORT/REPLAY only): split each place into a GOTO "
           "to the pre-place pose (object lifted DELTA m in +z off its rest pose, default 0.05) "
           "+ the real set-down at the place witness. Default 0.05; 0 disables.",
  )
  g_skills.add_argument(
      "--pre-add-grasp", type=float, nargs="?", const=0.05, default=0.05, metavar="DELTA",
      help="Like --pre-pick, for ADD_GRASP and HANDOVER (EXPORT/REPLAY only): split each "
           "incoming-hand acquisition into a GOTO to the pre-grasp (gripper retracted DELTA "
           "m along its -approach axis, default 0.05) + the real grasp/ownership switch. "
           "Default 0.05; 0 disables.",
  )
  g_skills.add_argument(
      "--post-release", type=float, nargs="?", const=0.05, default=0.05, metavar="DELTA",
      help="POST-RELEASE departure funnel (EXPORT/REPLAY only): after the detach task, add a "
           "GOTO that retracts the released gripper DELTA m along its -approach axis while "
           "the endpoint IK pins every other arm at the release keyframe. BiRRT remains "
           "unconstrained between endpoints. Default 0.05; 0 disables.",
  )
  g_skills.add_argument(
      "--fuse-pre-skill", action=argparse.BooleanOptionalAction, default=None,
      help="With --motion-plan and any --pre-* split: BEFORE the export, re-solve every pre-skill "
           "funnel keyframe with each arm that is NOT supposed to move PINNED (hard eq) to the "
           "skill witness, instead of letting the exporter solve it with only a weight-1.0 pull. "
           "The skill constraints are RELATIVE, so today the optimiser pays for the backoff by "
           "moving the cheaper body: measured on car-mini, the TARGET assembly slides 16.8-48.8mm "
           "between the funnel mouth and the seat while the inserted part covers 6.1-35.8mm of "
           "the nominal 50mm -- i.e. the exported funnel is not an approach, and inside a "
           "--fuse-cograsp span it re-introduces exactly the holder drift fusion removed. The "
           "skill keyframe is never re-solved (so fused pairs survive). ALL-OR-NOTHING per "
           "split: any residual/penetration/no-improvement verdict keeps today's keyframe. The "
           "exported task structure is identical either way.",
  )
  g_output.add_argument(
      "--hoist-releases", action=argparse.BooleanOptionalAction, default=True,
      help="Default ON (2026-07-29). POST-PASS after --solve (before witness optimization): bubble every `release` as "
           "EARLY in the plan as legal -- applicable per the real successor generator (wrench "
           "stability: a needed hold cannot release early), cheap-geometry-checked, and only "
           "where the swapped pair provably commutes (identical state after). Frees arms "
           "sooner (shorter robot chains in the schedule / depgraph, less idle-arm clutter "
           "for motion planning). Plan length and final state unchanged.",
  )
  g_output.add_argument(
      "--utilization", action="store_true",
      help="After --solve, report the plan's SCHEDULED-MAKESPAN utilization: list-schedule "
           "the actions on their resource partial order (robots + touched parts -- the same "
           "order the depgraph export uses) and print makespan vs serial time, speedup, and "
           "per-robot busy%%. Durations = witness joint-distances when --optimize-witnesses "
           "ran, else 1/action. Report only; no behavior change.",
  )
  g_witness.add_argument(
      "--witness-robot-collisions", action=argparse.BooleanOptionalAction, default=True,
      help="Default ON (2026-07-29). During --optimize-witnesses, keep the OTHER robots' collision geometry ON (the "
           "search strips it on solo steps for speed). Re-solved witnesses then AVOID the idle "
           "arms instead of passing through them -- important before motion planning, where "
           "mr_bench validates every arm. Note idle arms are at the manipulation EXIT pose "
           "in the witness scene.",
  )
  g_witness.add_argument(
      "--fuse-cograsp", action=argparse.BooleanOptionalAction, default=True,
      help="Default ON (2026-07-29). With --motion-plan: AFTER witness optimization and BEFORE the export, re-solve "
           "each (add_grasp, assemble) witness PAIR as ONE 2-phase KOMO instead of two "
           "independent single-keyframe IK solves. With --pre-insertion, add_grasp and the "
           "inserter share the funnel-mouth keyframe; pre-switch support changes and releases "
           "reuse the adjacent coupled endpoint instead of introducing another transport "
           "keyframe. The ordinary motion planner remains free to detour between endpoints; "
           "no controller or interior path constraint is injected here. The shared MP task "
           "goals are always exported; this flag "
           "controls the additional witness re-solve, which requires one kinematic mode. This avoids "
           "transporting a closed co-grasp between unrelated IK branches. Per pair it "
           "is ALL-OR-NOTHING: any failed/violated/not-improved solve keeps BOTH originals. "
           "Pair with --mp-cograsp --mp-sequence to measure the effect.",
  )
  g_ik.add_argument(
      "--strict-collisions", type=float, nargs="?", const=1e-4, default=1e-6, metavar="EPS",
      help="With --real-ik: REJECT any KOMO witness whose skill scene has total penetration "
           "> EPS metres (default 1e-6 since 2026-07-29; 1e-4 when given bare; "
           "--no-strict-collisions disables the gate) -- the collision objective is a SOFT "
           "inequality, so the plain solve accepts mm-scale 'rescued grazes' (measured: wrist "
           "capsule 4.9mm into the table at a pick witness) that a physics sim or a "
           "0-tolerance motion-plan export then pays for. Dirty solutions count as infeasible "
           "and the restart loop keeps searching for a clean basin; the collision weight is "
           "also raised 1e1 -> 1e2 so the solver steers clean instead of leaning on the "
           "slack. The gate only sees what the skill scene contains: combine with "
           "--witness-robot-collisions to make stripped idle arms visible. Slower solves; "
           "prints the number of vetoed witnesses at the end.",
  )
  g_ik.add_argument(
      "--no-strict-collisions", dest="strict_collisions", action="store_const", const=None,
      default=argparse.SUPPRESS,
      help="Disable the strict witness-penetration gate (restores the pre-2026-07-29 default).",
  )
  g_ik.add_argument(
      "--always-contact", action=argparse.BooleanOptionalAction, default=True,
      help="Default ON (2026-08-17): NEVER disable a touched part's/group's contact in the "
           "skill IK. A witness is then accepted only if the whole scene clears the "
           "--strict-collisions gate -- grasp and mate contact included; an overlapping plan "
           "fails its IK and the search re-runs (blacklist/repair). Grasps whose finger proxy "
           "geometrically overlaps the part fail too, which is the point: only genuinely "
           "contact-clean witnesses pass. --no-always-contact restores the historical behavior "
           "(the touched group's contact is switched off for the solve, which also hides "
           "arm-vs-part and part-vs-part for that group).",
  )
  g_mp.add_argument(
      "--mp-clearance", action="store_true",
      help="With --motion-plan: BEFORE running BiRRT, print a per-task-goal collision report "
           "(mr_bench's own is_collision_free at each keyframe -- exactly what BiRRT checks): "
           "which action, total penetration, and the offending frame pairs with signed "
           "distance. Names the culprit (arm-arm graze / part-vs-table / part attached to a "
           "gripper it should have released) so 'looks fine in viser but the planner rejects' "
           "becomes concrete.",
  )
  g_mp.add_argument(
      "--mp-cograsp", action="store_true",
      help="With --motion-plan: AFTER planning, FK the solved path and report how far each "
           "pair of co-grasping tool centres drifts from its keyframe relative pose. A brace "
           "exported as a bare goto is unconstrained BETWEEN keyframes, so two grippers welded "
           "to one object can move independently; this measures it. SEQUENCE problems only -- "
           "pair it with --mp-sequence or it prints 'skipping'.",
  )
  g_mp.add_argument(
      "--mp-save", metavar="FILE", default=None,
      help="With --motion-plan: save the solved BiRRT PATH to FILE (JSON: per-state joint "
           "vectors + mode task ids). Load it later WITHOUT re-planning via "
           "`python3 sim.py run <plan.json> --mp-path FILE` -- pair it with --save "
           "<plan.json> from the SAME run (the problem is rebuilt from that discrete plan).",
  )
  g_mp.add_argument(
      "--mp-collision-tolerance", type=float, default=1e-4, metavar="M",
      help="With --motion-plan: the exported problem's collision tolerance in metres of "
           "TOTAL allowed penetration at validated configs (mr_bench semantics; default "
           "1e-4 since 2026-07-29, 0.01 was the historical value). Works with the "
           "--strict-collisions default so the plan the sim executes "
           "contains no tolerated penetration beyond noise.",
  )
  g_mp.add_argument(
      "--motion-plan", action="store_true",
      help="After --solve, EXPORT the plan (+ witnesses) to the mr_bench motion planner "
           "(mp_export) and run Bidirectional RRT* to a collision-free multi-robot path. "
           "Reuses THIS run's scene/witness flags (--assembly/--parts/--num-robots/--robot/"
           "--optimize-witnesses/--witness-clearance/--vacuum/--goal-on-table). Needs "
           "--real-ik (for the IK witnesses); if --optimize-witnesses is not given, RAW "
           "search witnesses are used. See the --mp-* flags for planner options.",
  )
  g_mp.add_argument(
      "--mp-export-save", metavar="FILE", default=None,
      help="Persist the FINAL motion-planning export (post-fusion skill witnesses, every "
           "pre-skill keyframe, witness sets, export settings and task signature). Written "
           "before --mp-modes/planning, so it also works when no motion path is found.",
  )
  g_mp.add_argument(
      "--mp-export-load", metavar="FILE", default=None,
      help="Load a final motion-planning export and skip ALL witness/fusion/pre-skill IK "
           "solves. The saved plan and joint order must match exactly; missing keyframes or "
           "a changed task signature are fatal instead of being silently re-solved. Saved "
           "collision/clearance settings are defaults; explicitly supplied --mp-collision-"
           "tolerance/--mp-robot-margin/--mp-margin-ramp/--mp-table-margin/--mp-wrench-gate "
           "values override "
           "them without changing the persisted keyframes.",
  )
  g_mp.add_argument(
      "--mp-sequence", action=argparse.BooleanOptionalAction, default=True,
      help="With --motion-plan: export a FIXED-SEQUENCE problem (total order) instead of a "
           "DEPENDENCY GRAPH (partial order, planner may reorder independent actions). "
           "Default ON since 2026-07-29 (the sim executor pipeline assumes a sequence); "
           "--no-mp-sequence restores the depgraph export.",
  )
  g_mp.add_argument(
      "--mp-optimize", action=argparse.BooleanOptionalAction, default=True,
      help="With --motion-plan: RRT* COST optimization until --mp-runtime. Default ON since "
           "2026-07-29. --no-mp-optimize is a diagnostic first-solution timing mode and may "
           "not be combined with --mp-save: exported production paths are always optimized. "
           "Uses the tree-extension shim.",
  )
  g_search.add_argument(
      "--ban-actions", type=str, default=None, metavar="CSV",
      help="SEARCH-scoped action bans: comma-separated 'kind:robot:part:arg' patterns "
           "('*' wildcards a field), e.g. 'handover:a1:obj_0:*'. The successor generator "
           "never yields matching actions -- the bandit outer loop's discrete escalation "
           "when a hold's acquisition locks an unconnectable homotopy. Never applied to "
           "replays (saved plans cannot contain banned actions).",
  )
  g_search.add_argument(
      "--vacuum-no-assemble", action="store_true",
      help="SEARCH-scoped CAPABILITY ban: no assemble may involve a vacuum arm -- as the "
           "acting robot, the holder of the inserted part or of the target, or a brace "
           "contact. The vacuum may pick and hand over only (the role split: the vacuum "
           "picks the flat seat, a two-finger receives the handover and does the mates). "
           "Needs --wrench (the per-robot kind map only flows under it). Never applied to "
           "replays.",
  )
  g_mp.add_argument(
      "--mp-stall-timeout", type=float, default=None, metavar="SEC",
      help="With --motion-plan: stop BiRRT early once the MODE FRONTIER has not grown for "
           "SEC seconds (a structured stuck signal for outer loops -- the per-mode attempt "
           "budget). run_birrt reports frontier depth + deepest task_ids either way. "
           "None = run the full --mp-runtime.",
  )
  g_mp.add_argument(
      "--mp-open-plans", action=argparse.BooleanOptionalAction, default=False,
      help="With --solve --motion-plan --milestones: fairly interleave complete assembly-plan "
           "enumeration with finite motion-planner pulls. A failed pull suspends that complete "
           "plan, tries another, then reopens suspended plans at increasing runtime tiers. "
           "No MP timeout becomes an action/boundary blacklist. Production loop outputs are "
           "always RRT*-optimized (--mp-optimize required).",
  )
  g_mp.add_argument(
      "--mp-plan-candidates", type=int, default=4, metavar="N",
      help="--mp-open-plans: number of distinct complete assembly plans to probe before only "
           "reopening the suspended pool (default 4).",
  )
  g_mp.add_argument(
      "--mp-plan-epochs", type=int, default=6, metavar="N",
      help="--mp-open-plans: fair MP effort tiers per candidate (default 6: with the default "
           "15s initial pull and standard 300s cap this is 15,30,60,120,240,300s).",
  )
  g_mp.add_argument(
      "--mp-plan-initial-runtime", type=float, default=15.0, metavar="SEC",
      help="--mp-open-plans: first BiRRT pull per complete plan (default 15s). Later tiers "
           "double this up to --mp-runtime.",
  )
  g_mp.add_argument(
      "--mp-runtime", type=float, default=30.0, metavar="SEC",
      help="With --motion-plan: BiRRT search budget in seconds (flag default 30 since 2026-07-29; the "
           "standard motion profile uses 300). "
           "Bounds ONLY the tree search, NOT shortcutting.",
  )
  g_mp.add_argument(
      "--mp-shortcut", action=argparse.BooleanOptionalAction, default=True,
      help="With --motion-plan: run the shortcutting post-process (default).",
  )
  g_mp.add_argument(
      "--mp-shortcut-iters", type=int, default=1000, metavar="N",
      help="With --motion-plan: shortcut iteration cap (default 1000; NOT time-bounded).",
  )
  g_mp.add_argument(
      "--mp-shortcut-strategy", choices=("native", "pair_then_gain"), default="native",
      help="With --motion-plan: native mr_bench shortcutter (default), or the local experimental "
           "pair-then-gain strategy. The latter first validates compatible two-arm shortcuts, "
           "then uses gain-guided single-arm cleanup; it never patches mr_bench.",
  )
  g_mp.add_argument(
      "--mp-shortcut-pair-checks", type=int, default=200, metavar="N",
      help="With --mp-shortcut-strategy pair_then_gain: native collision validations reserved "
           "for the initial pair phase (default 200).",
  )
  g_mp.add_argument(
      "--mp-shortcut-res", type=float, default=None, metavar="R",
      help="With --motion-plan: shortcut collision resolution (default: env collision_resolution, "
           "~10x faster than mr_bench's own 0.001 for identical quality).",
  )
  g_mp.add_argument(
      "--mp-viser", action="store_true",
      help="With --motion-plan: serve the found PATH in mr_bench's own viser viewer "
           "(display_path_viser: step slider + playback). Blocks. Port = --viser-port.",
  )
  g_mp.add_argument(
      "--mp-collision-view", action="store_true",
      help="With --mp-viser: show only PRIMITIVE (collision) shapes -- capsules/boxes/"
           "cylinders/ssBox -- and hide the file-loaded visual meshes "
           "(display_path_viser primitives_only). See exactly what the planner collides.",
  )
  g_mp.add_argument(
      "--mp-robot-margin", type=float, default=0.1, metavar="M",
      help="Plan with M metres of clearance between DIFFERENT ROBOTS (0 = off; default 0.1 "
           "since 2026-07-29). The planner "
           "certifies the COMMANDED path, but the executed one deviates by the controller's "
           "tracking error, so a path validated at zero clearance collides when run (measured "
           "on cross-new: p90 2.9mm, max 33.1mm at the EE). Classified by KINEMATIC ROOT, so a "
           "carried part counts as its holder -- gripper-to-payload, part-to-table and mating "
           "contacts keep the plain penetration budget because they are INTENDED.",
  )
  g_mp.add_argument(
      "--mp-margin-ramp", type=float, default=0.1, metavar="M",
      help="With --mp-robot-margin: relax the margin near BOTH endpoints of a segment, measured "
           "in metres of END-EFFECTOR displacement (0 = drop it outright for the task's own "
           "robot pair; "
           "default 0.1 since 2026-07-29). "
           "A handover needs its two arms close but not close to the third, so the margin is "
           "dropped only for the participating pair near its anchors. Keyed on EE "
           "DISTANCE, not path position: a sampling planner's predicate must be a pure function "
           "of the state, so 'the last 10%% of the path' is not expressible -- distance-to-goal "
           "is, and means the same thing.",
  )
  g_mp.add_argument(
      "--mp-margin-mode", choices=("linear", "binary"), default="linear",
      help="How --mp-margin-ramp relaxes robot/table margins near BOTH segment anchors. "
           "linear scales the requirement continuously from zero to full margin; binary "
           "requires zero auxiliary margin while the relevant EE is within the ramp distance "
           "of either anchor and the full margin outside it. The base collision predicate and "
           "joint-limit gate remain active in either mode.",
  )
  g_mp.add_argument(
      "--mp-seed", type=int, default=None, metavar="N",
      help="Seed BiRRT + shortcutting (mr_bench draws from the GLOBAL numpy/stdlib RNGs). "
           "Unseeded, the same command yields paths whose failure CLASSES differ run to run "
           "(measured: one draw grazes the table at 57N, the next has no table contact), so "
           "any A/B without this is noise. Default None = historical unseeded behaviour. "
           "Saved with the plan; sim.py reuses it unless overridden. With the real clock a "
           "seeded run is only APPROXIMATELY reproducible -- add --mp-fake-clock for bit-exact "
           "determinism.",
  )
  g_mp.add_argument(
      "--mp-fake-clock", action=argparse.BooleanOptionalAction, default=False,
      help="With --motion-plan (and --mp-seed): replace the wall clock with a deterministic "
           "fake clock (+1ms per consultation) so the run is BIT-EXACT reproducible -- in "
           "exchange --mp-runtime becomes a CONSULTATION budget, NOT wall seconds, so the "
           "tree phase can run much longer than the budget. Off by default: --mp-seed alone "
           "seeds the draw but keeps the real clock and true wall-second termination.",
  )
  g_mp.add_argument(
      "--mp-table-margin", type=float, default=0.0, metavar="M",
      help="Clearance a CARRIED part must keep to the table mid-transit (metres). The transit "
           "graze is the top sim killer: BiRRT+shortcut skims the table at a 0.1mm collision "
           "tolerance, mm-level tracking error then drives the held part INTO it and the "
           "contact snaps the grasp (measured obj_0~world 40-120N vs a 25N cap, the FIRST "
           "failure in every config tried). Relaxed over --mp-margin-ramp metres of EE travel "
           "at BOTH segment ends -- pick and set-down still touch. Resting parts and the arms "
           "themselves keep zero table margin. Use WITH --mp-margin-ramp (else the mode after "
           "every pick starts in violation).",
  )
  g_mp.add_argument(
      "--mp-wrench-gate", action=argparse.BooleanOptionalAction, default=False,
      help="Also require quasi-static GRAVITY equilibrium at every motion-planner feasibility "
           "sample. This reuses the task planner's grasp capacities and warm LP backend, but "
           "does not apply insertion/screw forces during unrelated free-space transit. Off by "
           "default so --no-wrench plans remain a genuinely topological baseline; requires "
           "--wrench when enabled.",
  )
  g_mp.add_argument(
      "--mp-modes", action="store_true",
      help="With --motion-plan: instead of planning, step through the exported problem's MODES "
           "(each posed at its task goal keyframe) in mr_bench's own viser viewer "
           "(scripts/show_problems.py::visualize_modes). Good for debugging the export / "
           "spotting per-mode collisions. Blocks. Port = --viser-port.",
  )
  g_search.add_argument(
      "--sig-collapsed-h", action=argparse.BooleanOptionalAction, default=True,
      help="Default ON (2026-07-29). SIGNATURE grasp collapse + LABELED loose parts: grasp labels are equivalence "
           "classes under the fact signature (press capability, pick-face vector, contact "
           "patch, solo-hold, brace-side), and loose parts keep their class label -- the "
           "fine labeled-h's plateau-killing pick/press discrimination at collapsed-h "
           "scale. Facts OR over class members (relaxation -> admissible).",
  )
  g_search.add_argument(
      "--sig-components", type=str, default="press", metavar="CSV",
      help="Ablation for --sig-collapsed-h: comma-separated subset of signature "
           "components (press,pickface,contact,solohold,braceside). Default 'press' since "
           "2026-07-29 (the load-bearing component; full signatures blow up at scale -- "
           "pass 'all' for every component).",
  )
  g_search.add_argument(
      "--geo-facts", action="store_true",
      help="Gate the labeled abstraction's backward Dijkstra with FLOATING-GRIPPER "
           "geometry: a labeled state (assembly at a rest face + its grasps/braces) whose "
           "grippers cannot physically coexist (gripper-vs-assembly, gripper-vs-gripper, "
           "gripper-vs-table) relays nothing backward. Sound and admissible, but MEASURED "
           "MOSTLY HARMFUL as a speed feature: the h build pays 10-100s of collision "
           "facts, the exact/lazy searches barely visit the killed labels, and on "
           "car-mini the tightened h wrecks the w=5 greedy corridor (2.5k -> 80k "
           "expansions; same dead-end-conversion effect as --brace-for-assembly-only). "
           "Real value measured: milestone plan QUALITY (car-mini 31 -> 25 actions, "
           "noisy cross 36 -> 32) and it lets the noise-0.05 cross exact search "
           "terminate. Kept for experimentation; prints when active.",
  )
  g_viz.add_argument(
      "--view-cheap-checks", action="store_true",
      help="Step through every (uncached) cheap collision check in a blocking C.view(True) "
           "window -- verdict, triggering action, and penetrating pairs in the overlay. "
           "Both FREE and COLLIDING results are shown; each distinct scene appears once. "
           "Use with small instances: the search blocks on every window.",
  )
  g_viz.add_argument(
      "--viser", action="store_true",
      help="Display via a viser web viewer (http://localhost:<port>) instead of the rai "
           "window: the whole replay is recorded and served with a step slider / playback "
           "controls (no per-step blocking windows).",
  )
  g_viz.add_argument(
      "--viser-port", type=int, default=8080,
      help="Port for the viser server (with --viser). Default: 8080.",
  )
  g_scene.add_argument(
      "--seed", type=int, default=0,
      help="Root RNG seed for the scene and search. It also seeds query-local KOMO IK unless "
           "--ik-seed is given. Default: 0.",
  )
  g_scene.add_argument(
      "--ik-seed", type=int, default=None,
      help="Optional independent root for deterministic per-physical-query KOMO restart "
           "streams. This permits IK robustness sweeps without changing the scene, grasp "
           "database, or discrete search seed. Default: use --seed.",
  )
  g_output.add_argument(
      "--load-plan", metavar="FILE", default=None,
      help="Read the ACTIONS from a saved plan JSON and skip the search, then run the normal "
           "post-passes and export on them (--optimize-witnesses, --fuse-cograsp, "
           "--motion-plan). Use this to apply a witness pass to a plan you already trust. "
           "Modern files persist the resolved search settings, but re-solving can still choose "
           "a different finite/anytime result; --load-plan keeps the certified action list "
           "fixed. Pair it with --config <the same file> so the scene matches.",
  )
  g_output.add_argument(
      "--grasp-physics-cache-plan", metavar="FILE", default=None,
      help="With --solve --grasp-physics-check, reuse FILE's complete a-priori handover "
           "certificate only when its force, physical hand models, and exact finite grasp "
           "database signature match. This does not load FILE's task plan or witnesses.",
  )
  g_output.add_argument(
      "--refresh-witnesses", action="store_true",
      help="With --load-plan: IGNORE the witnesses persisted in the plan file and re-run the "
           "witness pass. Default: reuse them whenever every witness-shaping setting matches "
           "the saved config (making the motion-plan stage a pure function of the file).",
  )
  g_output.add_argument(
      "--save", metavar="PATH", default=None,
      help="After --solve, write the plan + the scene config (assembly, parts, robots, seed, "
           "wrench settings, ...) to PATH as JSON, so it can be replayed later without "
           "re-searching via the standalone `python replay.py PATH` script.",
  )
  _hide_internal_cli(parser, _INTERNAL_MAIN_CLI)
  return parser


def _scan_flags(parser, argv):
  # Pre-parse scan of the raw argv: returns (duplicates, explicit_dests).
  # Argparse silently keeps the LAST occurrence of a repeated flag, so a repeated --seed
  # (or any other flag) is a footgun the parsed result cannot reveal -- scan the tokens
  # instead. Exact for this CLI: no short options, and no flag's VALUE starts with '-'.
  # (A '--' token ends the scan, argparse-style.)
  toks = list(_sys.argv[1:]) if argv is None else list(argv)
  dups = {}
  explicit = set()
  for tok in toks:
    if tok == "--":
      break
    if not tok.startswith("-"):
      continue
    name = tok.split("=", 1)[0]
    act = parser._option_string_actions.get(name)
    if act is None:
      continue
    dest = act.dest
    explicit.add(dest)
    if dest in dups:
      dups[dest][1] += 1
    else:
      dups[dest] = [name, 1]
  return dups, explicit


def _warn_duplicate_flags(parser, argv):
  dups, _ = _scan_flags(parser, argv)
  for dest, (name, n) in dups.items():
    if n > 1:
      print(f"[flag] !!! '{name}' appears {n} times: argparse keeps only the LAST value, so "
            f"every earlier occurrence is silently dead"
            + (" (use --mp-seed for a separate motion-plan seed)" if dest == "seed" else "")
            + ".")


def parse_args(argv=None):
  # Layer --config and named profile values as parser defaults, then parse the raw argv last:
  # explicit flag > explicitly selected profile > config file > built-in default. The first
  # parse only discovers the config/profile meta options. With none of them this remains the
  # historical one-pass path.
  parser = build_parser()
  _warn_duplicate_flags(parser, argv)
  args = parser.parse_args(argv)
  if getattr(args, "config", None):
    from config import RunConfig
    parser.set_defaults(**RunConfig.from_file(args.config).to_dict())
  if getattr(args, "search_profile", None):
    parser.set_defaults(**_SEARCH_PROFILES[args.search_profile])
  if getattr(args, "motion_profile", None):
    parser.set_defaults(**_MOTION_PROFILES[args.motion_profile])
  if (getattr(args, "search_profile", None) or getattr(args, "motion_profile", None)
      or getattr(args, "config", None)):
    args = parser.parse_args(argv)                       # layered defaults + CLI overrides
  return args


# fields persisted by --save so replay.py can reconstruct the EXACT scene (see plan_io.py).
_SAVE_CONFIG_FIELDS = (
    "assembly", "parts", "num_robots", "vacuum", "floating_ee", "robot", "robot_types",
    "scatter_spacing", "center_scatter", "perturb_start_pose", "seed", "ik_seed", "wrench",
    "wrench_profile", "grasp_model", "insert_force",
    "curved_torque_scale", "insertion_noise", "insertion_offset", "inserter_check", "real_ik",
    "bracing", "robust_mode", "dense_grasps", "random_init", "part_float", "inhand_reorient",
    "pre_insertion", "skill_funnel_collision_res", "pre_pick", "post_pick", "pre_place", "pre_add_grasp",
    "post_release", "staged_mates", "mate_symmetry", "mate_wrench_limits",
    "screw_angle", "screw_depth", "screw_torque", "screw_speed", "screw_samples",
    "allow_curved_grasps", "grippers", "grasp_compliance",
    "vacuum_scale", "two_finger_scale", "vacuum_engage_force",
    "vacuum_engage_friction", "tf_close_torque", "dominance",
    "table_mu",
    "useful_grasps", "brace_resting",
    "ignore_gravity", "orientation_robust_wrench",      # gravity semantics change the
    # successor set and must survive replay/config round-trips
    # weightless contains releases/lateral mates that are UNSTABLE under gravity, so a rebuild
    # without it cannot reproduce the actions (user-hit: sim StopIteration on a cross-12 plan)
    "strict_collisions", "witness_robot_collisions", "plain_box",
    "stable_rest_poses", "placement_yaws",              # placement ids index the per-group
    # placement table, so this changes the ACTION SPACE, not only the geometry
    # SUCCESSOR-AFFECTING and previously UNSAVED: a plan solved with any of these replays
    # against a DIFFERENT successor set than the search that made it, and the first symptom is
    # optimize_witnesses*/replay failing with "cannot reproduce action ... from the successor
    # set". min_sep and place_cells change the place cells, the rest change the action space.
    "strict_mate_collisions", "always_contact", "ik_accuracy", "mate_accuracy",
    "mate_angular_accuracy", "grasp_accuracy",
    "mate_eq_weight", "grasp_eq_weight",
    "place_eq_weight",                                   # a place is the DATUM later mates are
    # measured against, so its residual is not local (measured: body 3.91mm high, wheels then
    # mated onto it at 0.01-0.23mm -- perfectly seated on a body in the wrong place)
    "grasp_translations",                                # grasp ids index the DB
    "grasp_sampler", "grasp_sampler_keep",               # --grasp-sampler: sampled ids only
    "grasp_sampler_candidates", "grasp_sampler_full_probe", "grasp_sampler_roll_refine",
    "grasp_sampler_seed",                                # replay under identical settings
    "grasp_sampler_db_hash",                              # positional DB content identity
    "artifact_contract",                                  # collision/grasp/closure identity
    "grasp_refine", "grasp_widen", "grasp_widen_batch", "grasp_widen_max",
    "grasp_local_widen", "grasp_local_caps", "component_grasp_keep",
    # search provenance; replay reconstructs the final widened prefix recorded in `keep`
    "grasp_sampler_margin", "grasp_sampler_pad_support",
    "grasp_sampler_wrap",                                 # curvature-fair support metric (None = legacy)
    "grasp_sampler_strict",                               # reject below-gate fallback rows
    "grasp_sampler_scenes",                              # scene-aware set split (rebinds ids)
    "tight_fingers",                                     # changes what the grasp DB filters out
    "openarm_real_finger_radius_scale",                  # diagnostic real-arm IK geometry
    "grasp_closure_check",                               # virtual close changes grasp-edge feasibility
    "grasp_physics_check", "grasp_physics_force",      # exact finger physics changes grasp DB
    "ik_seeding",                                        # off => witness IK fails on re-picks/
    # reorienting places, and mp_export then exports the arm's PREVIOUS config as the goal
    "pre_pick", "post_pick", "pre_place", "pre_add_grasp", "post_release", # each SPLITS an action into tasks:
    # unsaved, a replay/sim rebuild has fewer tasks than the --mp-path it loads, and the path
    # indexes tasks by integer id. pre_insertion was already saved; these three were not.
    "witness_margin", "witness_joint_margin", "object_over_table",
    "object_over_table_margin", "witness_draw", # robust-keyframe/workcell gates + ladder
    "witness_sets", "witness_set_spread",                # multi-witness goal sets
    # draw: both change which witnesses ship, so replays/reuse must match
    "ban_actions",                                       # shapes the PLAN (provenance +
    # hoist/witness walks must see the same successor filter the search used)
    # which witnesses ship, so replays/reuse must match
    "optimize_witnesses", "witness_branch_budget",       # WHICH witness optimizer ran: a
    "witness_branch_metric", "witness_reconfig_thresh",  # different one yields a different
    # None-pattern, and build_tasks SKIPS a task whose witness is None -- so the rebuilt mode
    # chain stops matching the saved --mp-path ("mode alignment failed: N unmatched").
    "hoist_releases",                                    # requested release-hoist policy. A real-IK
    # rewrite is transactional: failed re-certification preserves the saved source action order.
    "witness_clearance", "handovers", "allow_place",
    "fuse_cograsp", "fuse_pre_skill",                   # motion-only witness/keyframe passes
    "mp_sequence",                                       # depgraph vs sequence = a different
    # task GRAPH; the sim rebuilt SEQUENCE unconditionally while main defaults to depgraph
    "mp_optimize", "mp_runtime", "mp_stall_timeout",   # RRT* consultation contract
    "mp_shortcut", "mp_shortcut_iters", "mp_shortcut_res",
    "mp_shortcut_strategy", "mp_shortcut_pair_checks", # certified path post-processing
    "mp_collision_tolerance",                            # base predicate certificate
    "mp_robot_margin", "mp_margin_ramp", "mp_margin_mode", # clearance certificate
    "mp_table_margin",                                   # carried-part-vs-table transit clearance
    "mp_wrench_gate",                                    # gravity equilibrium along MP path
    "mp_cograsp",                                        # closed-chain path diagnostic
    "mp_seed",                                           # reproducible BiRRT draw (None = unseeded)
    "mp_fake_clock",                                     # fake deterministic clock: runtime becomes
    "mp_open_plans", "mp_plan_candidates", "mp_plan_epochs",
    "mp_plan_initial_runtime",                           # outer MP fairness provenance
    # a CONSULTATION budget (bit-exact seeded replay)
    "min_sep", "place_cells", "prefer_center", "brace_for_assembly_only", "lru_robots",
    "anon_robots", "declutter", "por",
    # LAYOUT: these feed add_disassembled_obj, so without them the replay/sim rebuilds a
    # DIFFERENT initial scatter than the plan was solved in (FRAGILITY 1.8).
    "grid_init", "grid_gap",
    # SEARCH PROVENANCE. A saved plan is replayable from its certificates without these, but
    # `--config PLAN.json --solve` must also reproduce the search that selected it. Persist the
    # resolved values rather than only a named-profile label: profiles may evolve, whereas a
    # paper artifact must keep the semantics it was actually solved under.
    "weight", "milestones", "milestones_anytime", "full_backward",
    "full_backward_max_states", "full_backward_max_edges", "goal_on_table",
    "milestone_oracle",
    "fill_local_h", "fill_raw_grasp_h", "fill_target_support_quotient", "fill_exact",
    "fill_greedy", "fill_incumbent_first",
    "fill_policy_first", "fill_policy_guide", "fill_skeleton_first",
    "fill_action_widening", "milestone_boundaries",
    "fill_scope_widening", "fill_touch_all", "placement_yaw_widening",
    "fill_window", "lns_window", "lns_grasp_transition_h", "lazy_fills", "open_witness", "open_witness_batch",
    "open_witness_epochs", "open_witness_positive_order", "lazy_edge_selector",
    "analytic_seed_cap", "ssik_seed_neighborhood",
    "ssik_neighborhood_cap", "fail_order", "empty_seed_refutes",
    "ik_positive_reuse", "assemble_ssik_pose_seeds", "assemble_ssik_pose_projection",
    "assemble_unbiased_first", "assemble_basin_racing", "assemble_race_probe_evals",
    "assemble_race_promote", "handover_basin_racing", "handover_race_probe_evals",
    "handover_race_starts", "handover_race_promote",
    "wrench_aware_h", "sig_collapsed_h", "sig_components",
)


def save_plan(path, cfg, plan, witnesses=None, *, C=None, assembly=None, robots=None):
  from plan_io import write_plan, _motion_plan_digest
  config = {k: getattr(cfg, k) for k in _SAVE_CONFIG_FIELDS}
  # A plan can only be replay-gated as strictly as the SEARCH that selected it.  This marker
  # separates new plans (where --witness-margin is installed before search) from old files
  # where the same config key described a witness-only post-pass.  Preserve the source
  # certificate across --load-plan + --save; never accidentally upgrade an old plan merely
  # because it was copied through a newer executable.
  _cert = getattr(cfg, "_search_witness_margin", None)
  if _cert is None:
    _cert = (cfg.witness_margin if cfg.solve and cfg.real_ik
             and cfg.witness_margin is not None else 0.0)
  config["_search_witness_margin"] = float(_cert)
  grasp_physics = None
  if cfg.grasp_physics_check:
    from problem import _GRASP_PHYSICS_CTX
    if _GRASP_PHYSICS_CTX is None:
      raise RuntimeError("cannot save a physics-certified plan without its grasp context")
    grasp_physics = _GRASP_PHYSICS_CTX.export_handovers()
    if grasp_physics is None:
      raise RuntimeError(
          "cannot save a physics-certified plan before the a-priori handover table exists")
  motion_data = getattr(cfg, "_loaded_motion_data", None)
  digest = _motion_plan_digest(plan)
  if (motion_data is not None and
      (motion_data.get("plan_sha256") != digest or
       motion_data.get("grasp_db_sha256") != cfg.grasp_sampler_db_hash)):
    motion_data = None
  if (motion_data is None and cfg.real_ik and plan and C is not None
      and assembly is not None and robots is not None):
    cache = getattr(cfg, "_motion_data_cache", None)
    if cache is None:
      cache = cfg._motion_data_cache = {}
    cache_key = (digest, cfg.grasp_sampler_db_hash)
    motion_data = cache.get(cache_key)
    if motion_data is None:
      from plan_io import build_motion_data
      motion_data = build_motion_data(
          C, assembly, plan, robots,
          wrench_kind_arg(C, robots) if cfg.wrench else None,
          handovers=cfg.handovers, allow_place=cfg.allow_place,
          bracing=cfg.bracing, grasp_db_hash=cfg.grasp_sampler_db_hash)
      cache[cache_key] = motion_data
  write_plan(path, config, plan, witnesses=witnesses, grasp_physics=grasp_physics,
             motion_data=motion_data)
  print(f"[save] wrote {len(plan)}-action plan + scene config"
        + (" + witnesses" if witnesses is not None else "")
        + (" + handover certificates" if grasp_physics is not None else "")
        + f" to {path}")


# Witness REUSE guard: a persisted witness block is only valid if every setting that shaped
# it still holds. All of these are in _SAVE_CONFIG_FIELDS, so files that carry witnesses also
# carry the values; a key absent from an (older) saved config is treated as a mismatch.
_WITNESS_FIELDS = (
    "optimize_witnesses", "witness_branch_budget", "witness_branch_metric",
    "witness_reconfig_thresh", "witness_clearance", "witness_robot_collisions",
    "strict_collisions", "mate_eq_weight", "place_eq_weight", "mate_accuracy",
    "mate_angular_accuracy",
    "grasp_accuracy", "grasp_eq_weight",
    "ik_accuracy", "ik_seeding", "assemble_ssik_pose_seeds",
    "assemble_ssik_pose_projection", "assemble_unbiased_first", "assemble_basin_racing",
    "assemble_race_probe_evals", "assemble_race_promote", "handover_basin_racing",
    "handover_race_probe_evals", "handover_race_starts", "handover_race_promote",
    "ssik_seed_neighborhood", "ssik_neighborhood_cap",
    "seed", "ik_seed", "bracing", "handovers", "allow_place", "mate_symmetry",
    "tight_fingers", "openarm_real_finger_radius_scale", "part_float", "witness_margin",
    "witness_joint_margin", "object_over_table", "object_over_table_margin", "witness_draw",
    "witness_sets", "witness_set_spread", "pre_insertion", "skill_funnel_collision_res",
    "grasp_sampler", "grasp_sampler_keep", "grasp_sampler_candidates",
    "grasp_sampler_full_probe", "grasp_sampler_roll_refine", "grasp_sampler_seed",
    "grasp_sampler_db_hash", "grasp_closure_check",
    "grasp_sampler_margin", "grasp_sampler_pad_support", "grasp_sampler_wrap",
    "grasp_sampler_strict", "grasp_sampler_scenes", "placement_yaws")


def _witnesses_reusable(saved_cfg, cfg):
  # a key MISSING from the saved config means the plan predates that flag -- it was
  # produced with the flag at its DEFAULT, so compare against the RunConfig default
  # (the single source of defaults) instead of declaring a mismatch (a bare `k not in
  # saved_cfg` silently re-solved the witnesses of EVERY older plan file each time a new
  # witness-affecting flag landed).
  from config import RunConfig
  # These two defaults became non-zero after saved witness blocks already existed.  Missing
  # keys therefore mean the historical OFF values, not today's fresh-run defaults.
  legacy = {"witness_margin": None, "witness_sets": 0,
            "grasp_sampler_db_hash": None}
  bad = [k for k in _WITNESS_FIELDS
         if saved_cfg.get(k, legacy.get(k, RunConfig.default_for(k))) != getattr(cfg, k, None)]
  if bad:
    print("[witness-reuse] settings differ from the saved witnesses on: "
          + ", ".join(bad) + " -- re-solving instead of reusing.")
  return not bad


# These settings choose/rank EXTRA candidates but do not affect whether the persisted search
# certificate itself remains a valid baseline.  In particular, changing `off` -> `floor` must
# run floor optimization ON TOP OF the saved witness, not discard it and stochastically redo IK.
_WITNESS_OPTIMIZER_FIELDS = {
    "optimize_witnesses", "witness_branch_budget", "witness_branch_metric",
    "witness_reconfig_thresh", "witness_clearance", "witness_draw",
    "witness_sets", "witness_set_spread",
}


def _witnesses_baseline_compatible(saved_cfg, cfg):
  """Whether saved feasible keyframes are safe inputs under the current scene/IK gates."""
  from config import RunConfig
  legacy = {"witness_margin": None, "witness_sets": 0,
            "grasp_sampler_db_hash": None}
  fields = [k for k in _WITNESS_FIELDS if k not in _WITNESS_OPTIMIZER_FIELDS]
  bad = [k for k in fields
         if saved_cfg.get(k, legacy.get(k, RunConfig.default_for(k))) != getattr(cfg, k, None)]
  if bad:
    print("[witness-baseline] settings affecting witness feasibility differ on: "
          + ", ".join(bad) + ".")
  return not bad


def _selected_solver_engine(cfg):
  """Return the one search engine selected by `_dispatch_solve`'s precedence.

  Several newer flags refine the ordinary milestone *iterator*, rather than the common action
  model.  Keeping this small precedence mirror beside dispatch lets the CLI reject an otherwise
  plausible spelling which a higher-priority solver would silently ignore.  It deliberately
  describes only public engine families, not individual implementation branches.
  """
  if cfg.ladder_lns is not None:
    return "ladder-lns"
  if cfg.ladder is not None:
    return "ladder"
  if cfg.milestones_anytime is not None:
    return "milestones-anytime"
  if cfg.bounded or cfg.dfbnb:
    return "bounded"
  if cfg.full_backward:
    return "full-backward"
  if cfg.lazy:
    return "lazy"
  if cfg.optimistic:
    return "optimistic"
  if cfg.anytime is not None:
    return "anytime"
  return "milestones" if cfg.milestones else "exact"


def _validate_solver_scoped_flags(cfg):
  """Fail closed for flags implemented by a specific search schedule.

  The core finite grasp/placement action space is shared by all engines.  In contrast, local
  columns, yaw bands, and the reverse-policy rescue are *ordering schedules* implemented only
  by the ordinary milestone iterator.  Accepting those flags under a higher-priority solver
  would make an experiment look configured while doing nothing.
  """
  engine = _selected_solver_engine(cfg)
  if cfg.full_backward_max_states < 1:
    raise SystemExit("--full-backward-max-states must be >= 1")
  if cfg.full_backward_max_edges < 1:
    raise SystemExit("--full-backward-max-edges must be >= 1")
  if cfg.full_backward and (cfg.canonical or cfg.anon_robots or cfg.dominance or cfg.por):
    raise SystemExit("--full-backward retains the full labeled State; disable "
                     "--canonical, --anon-robots, --dominance, and --por")
  if cfg.lns_window is not None:
    if engine != "milestones-anytime":
      raise SystemExit("--lns-window requires --milestones-anytime; "
                       f"selected engine is {engine}")
    if cfg.lns_window < 1:
      raise SystemExit("--lns-window must be >= 1 (omit it for the full horizon)")
  if cfg.lns_grasp_transition_h:
    if engine != "milestones-anytime":
      raise SystemExit("--lns-grasp-transition-h requires --milestones-anytime; "
                       f"selected engine is {engine}")
    if not cfg.sig_collapsed_h:
      raise SystemExit("--lns-grasp-transition-h currently requires --sig-collapsed-h")
  if cfg.grasp_local_widen and engine != "milestones":
    raise SystemExit("--grasp-local-widen requires the ordinary --milestones solver; "
                     f"selected engine is {engine}")
  if cfg.placement_yaw_widening and engine not in (
      "milestones", "milestones-anytime", "full-backward"):
    raise SystemExit("--placement-yaw-widening is currently scheduled only by milestone "
                     f"solvers; selected engine is {engine}")
  if cfg.fill_greedy and engine != "full-backward":
    if engine not in ("milestones", "milestones-anytime"):
      raise SystemExit("--fill-greedy is implemented only by milestone local fills; "
                       f"selected engine is {engine}")
    if not cfg.lazy_fills:
      raise SystemExit("--fill-greedy requires --lazy-fills")
  if cfg.fill_incumbent_first:
    if engine not in ("milestones", "milestones-anytime"):
      raise SystemExit("--fill-incumbent-first is implemented only by milestone local fills; "
                       f"selected engine is {engine}")
    if not cfg.fill_local_h or not cfg.fill_exact or not cfg.lazy_fills:
      raise SystemExit("--fill-incumbent-first requires --fill-local-h, --fill-exact, "
                       "and --lazy-fills")
  if cfg.fill_raw_grasp_h:
    if engine not in ("milestones", "milestones-anytime"):
      raise SystemExit("--fill-raw-grasp-h is implemented only by milestone local fills; "
                       f"selected engine is {engine}")
    if not cfg.fill_local_h:
      raise SystemExit("--fill-raw-grasp-h requires --fill-local-h")
    if cfg.fill_target_support_quotient:
      raise SystemExit("--fill-raw-grasp-h and --fill-target-support-quotient are "
                       "mutually exclusive")
  if cfg.fill_policy_first:
    if not cfg.fill_local_h:
      raise SystemExit("--fill-policy-first requires --fill-local-h")
    if engine not in ("milestones", "milestones-anytime", "optimistic"):
      raise SystemExit("--fill-policy-first is implemented only by milestone/optimistic "
                       f"local fills; selected engine is {engine}")
  if cfg.fill_target_support_quotient:
    if engine not in ("milestones", "milestones-anytime"):
      raise SystemExit("--fill-target-support-quotient is implemented only by milestone "
                       f"local fills; selected engine is {engine}")
    if not cfg.fill_local_h or cfg.wrench or cfg.geo_facts:
      raise SystemExit("--fill-target-support-quotient requires --fill-local-h, "
                       "--no-wrench, and --no-geo-facts")
  if cfg.fill_policy_guide:
    if engine != "milestones":
      raise SystemExit("--fill-policy-guide is implemented only by the ordinary "
                       f"--milestones solver; selected engine is {engine}")
    if not cfg.fill_local_h:
      raise SystemExit("--fill-policy-guide requires --fill-local-h")
    if not (cfg.grasp_local_widen or cfg.placement_yaw_widening):
      raise SystemExit("--fill-policy-guide requires --grasp-local-widen or "
                       "--placement-yaw-widening; otherwise it has no fibre to widen")
  if cfg.fill_action_widening and engine not in ("milestones", "milestones-anytime"):
    raise SystemExit("--fill-action-widening is implemented only by milestone local fills; "
                     f"selected engine is {engine}")
  return engine


def _dispatch_solve(cfg, C, assembly, robots, robot_ik):
  # Select the search engine from the cfg flags and run it; returns the plan (or None).
  # PURE dispatch: no global-state changes here (all set-up ran before this is called), so
  # the engine choice reads as one flat if/elif instead of burying run() under it.
  if cfg.ladder_lns is not None:
    plan = solve_assembly_problem_ladder(
        C, assembly, robots=robots, time_budget=cfg.ladder_lns, robot_ik=robot_ik,
        weight=cfg.weight, wrench=cfg.wrench, wrench_aware_h=cfg.wrench_aware_h,
        handovers=cfg.handovers,
        allow_place=cfg.allow_place, bracing=cfg.bracing,
        geo_facts=cfg.geo_facts, sig_collapsed_h=cfg.sig_collapsed_h,
        labeled_h=cfg.labeled_h, collapsed_h=cfg.collapsed_h,
        grasp_collapsed_h=cfg.grasp_collapsed_h, lns=True, rwa_proof=False,
        lns_object=not cfg.lns_segments)
  elif cfg.ladder is not None:
    plan = solve_assembly_problem_ladder(
        C, assembly, robots=robots, time_budget=cfg.ladder, robot_ik=robot_ik,
        weight=cfg.weight, wrench=cfg.wrench, wrench_aware_h=cfg.wrench_aware_h,
        handovers=cfg.handovers,
        allow_place=cfg.allow_place, bracing=cfg.bracing,
        geo_facts=cfg.geo_facts, sig_collapsed_h=cfg.sig_collapsed_h,
        labeled_h=cfg.labeled_h, collapsed_h=cfg.collapsed_h,
        grasp_collapsed_h=cfg.grasp_collapsed_h)
  elif cfg.milestones_anytime is not None:
    plan = solve_assembly_problem_milestones_anytime(
        C, assembly, robots=robots, time_budget=cfg.milestones_anytime,
        robot_ik=robot_ik, weight=cfg.weight, wrench=cfg.wrench,
        wrench_aware_h=cfg.wrench_aware_h,
        handovers=cfg.handovers, allow_place=cfg.allow_place,
        bracing=cfg.bracing, geo_facts=cfg.geo_facts,
        sig_collapsed_h=cfg.sig_collapsed_h, labeled_h=cfg.labeled_h,
        collapsed_h=cfg.collapsed_h, grasp_collapsed_h=cfg.grasp_collapsed_h,
        fill_window=cfg.fill_window, milestone_oracle=cfg.milestone_oracle,
        lns_window=cfg.lns_window,
        lns_grasp_transition_h=cfg.lns_grasp_transition_h,
        fill_local_h=cfg.fill_local_h, fill_exact=cfg.fill_exact,
        fill_raw_grasp_h=cfg.fill_raw_grasp_h,
        fill_target_support_quotient=cfg.fill_target_support_quotient,
        fill_greedy=cfg.fill_greedy,
        fill_incumbent_first=cfg.fill_incumbent_first,
        fill_policy_first=cfg.fill_policy_first,
        fill_policy_guide=cfg.fill_policy_guide,
        fill_skeleton_first=cfg.fill_skeleton_first,
        fill_action_widening=cfg.fill_action_widening,
        fill_scope_widening=cfg.fill_scope_widening,
        milestone_boundaries=cfg.milestone_boundaries,
        placement_yaw_widening=cfg.placement_yaw_widening)
  elif cfg.bounded or cfg.dfbnb:
    plan = solve_assembly_problem_bounded(
        C, assembly, robots=robots, robot_ik=robot_ik, weight=cfg.weight,
        wrench=cfg.wrench, wrench_aware_h=cfg.wrench_aware_h,
        handovers=cfg.handovers, allow_place=cfg.allow_place,
        bracing=cfg.bracing, labeled_h=cfg.labeled_h, collapsed_h=cfg.collapsed_h,
        grasp_collapsed_h=cfg.grasp_collapsed_h or not (cfg.labeled_h or cfg.collapsed_h),
        stats_every=cfg.search_stats, geo_facts=cfg.geo_facts,
        sig_collapsed_h=cfg.sig_collapsed_h, beam_width=cfg.beam,
        dfbnb=cfg.dfbnb, pea=cfg.pea, pathmax=cfg.pathmax,
        corridor_h=cfg.corridor_h)
  elif cfg.full_backward:
    plan = solve_assembly_problem_full_backward(
        C, assembly, robots=robots, robot_ik=robot_ik,
        wrench=cfg.wrench, handovers=cfg.handovers,
        allow_place=cfg.allow_place, bracing=cfg.bracing,
        max_states=cfg.full_backward_max_states,
        max_edges=cfg.full_backward_max_edges)
  elif cfg.lazy:
    plan = solve_assembly_problem_lazy(
        C, assembly, robots=robots, weight=cfg.weight, robot_ik=robot_ik,
        wrench=cfg.wrench, handovers=cfg.handovers, allow_place=cfg.allow_place,
        labeled_h=cfg.labeled_h, collapsed_h=cfg.collapsed_h,
        grasp_collapsed_h=cfg.grasp_collapsed_h,
        wrench_aware_h=cfg.wrench_aware_h, bracing=cfg.bracing,
        geo_facts=cfg.geo_facts, sig_collapsed_h=cfg.sig_collapsed_h)
  elif cfg.optimistic and cfg.declutter:
    plan = solve_assembly_problem_optimistic_declutter(
        C, assembly, robots=robots, robot_ik=robot_ik, weight=cfg.weight,
        wrench=cfg.wrench, handovers=cfg.handovers, allow_place=cfg.allow_place)
  elif cfg.optimistic:
    plan = solve_assembly_problem_optimistic(C, assembly, robots=robots,
                                             robot_ik=robot_ik, weight=cfg.weight,
                                             wrench=cfg.wrench,
                                             wrench_aware_h=cfg.wrench_aware_h,
                                             handovers=cfg.handovers,
                                             allow_place=cfg.allow_place,
                                             bracing=cfg.bracing,
                                             fallback=cfg.optimistic_fallback,
                                             lazy_fills=cfg.lazy_fills,
                                             fill_local_h=cfg.fill_local_h,
                                             fill_exact=cfg.fill_exact,
                                             fill_policy_first=cfg.fill_policy_first,
                                             fill_policy_guide=cfg.fill_policy_guide,
                                             fill_skeleton_first=cfg.fill_skeleton_first,
                                             chain_h=cfg.optimistic_chain_h,
                                             grasp_collapsed_h=cfg.grasp_collapsed_h,
                                             collapsed_h=cfg.collapsed_h,
                                             labeled_h=cfg.labeled_h,
                                             sig_collapsed_h=cfg.sig_collapsed_h)
  elif cfg.anytime is not None:
    plan, _cost, _opt = solve_assembly_problem_anytime(
        C, assembly, robots=robots, time_budget=cfg.anytime, robot_ik=robot_ik,
        wrench=cfg.wrench, handovers=cfg.handovers, allow_place=cfg.allow_place,
        anonymous=cfg.anon_robots, labeled_h=cfg.labeled_h,
        collapsed_h=cfg.collapsed_h, grasp_collapsed_h=cfg.grasp_collapsed_h,
        wrench_aware_h=cfg.wrench_aware_h, bracing=cfg.bracing,
        part_order_bias=cfg.part_order_bias,
        sig_collapsed_h=cfg.sig_collapsed_h)
  elif cfg.milestones:
    # milestones is the DEFAULT solver (2026-07-29), so it sits LAST among the named ones:
    # any explicitly-requested solver above (--optimistic, --anytime, --lazy, ...) must win
    # over a default. --no-milestones falls through to the exact search below.
    plan = solve_assembly_problem_milestones(
        C, assembly, robots=robots, robot_ik=robot_ik, weight=cfg.weight,
        wrench=cfg.wrench, wrench_aware_h=cfg.wrench_aware_h,
        handovers=cfg.handovers, allow_place=cfg.allow_place,
        bracing=cfg.bracing, geo_facts=cfg.geo_facts,
        sig_collapsed_h=cfg.sig_collapsed_h, fill_window=cfg.fill_window,
        lazy_fills=cfg.lazy_fills, milestone_oracle=cfg.milestone_oracle,
        fill_local_h=cfg.fill_local_h, fill_exact=cfg.fill_exact,
        fill_raw_grasp_h=cfg.fill_raw_grasp_h,
        fill_target_support_quotient=cfg.fill_target_support_quotient,
        fill_greedy=cfg.fill_greedy,
        fill_incumbent_first=cfg.fill_incumbent_first,
        fill_policy_first=cfg.fill_policy_first,
        fill_policy_guide=cfg.fill_policy_guide,
        fill_skeleton_first=cfg.fill_skeleton_first,
        fill_action_widening=cfg.fill_action_widening,
        fill_scope_widening=cfg.fill_scope_widening,
        milestone_boundaries=cfg.milestone_boundaries,
        grasp_local_widen=cfg.grasp_local_widen,
        grasp_initial_keep=cfg.grasp_sampler_keep,
        grasp_widen_batch=cfg.grasp_widen_batch,
        grasp_widen_max=cfg.grasp_widen_max,
        placement_yaw_widening=cfg.placement_yaw_widening)
  else:
    plan = solve_assembly_problem(C, assembly, robots=robots, canonical=cfg.canonical,
                                  weight=cfg.weight, robot_ik=robot_ik, wrench=cfg.wrench,
                                  handovers=cfg.handovers, allow_place=cfg.allow_place,
                                  anonymous=cfg.anon_robots, labeled_h=cfg.labeled_h,
                                  collapsed_h=cfg.collapsed_h,
                                  grasp_collapsed_h=cfg.grasp_collapsed_h,
                                  wrench_aware_h=cfg.wrench_aware_h, bracing=cfg.bracing,
                                  part_order_bias=cfg.part_order_bias,
                                  stats_every=cfg.search_stats, por=cfg.por,
                                  geo_facts=cfg.geo_facts,
                                  sig_collapsed_h=cfg.sig_collapsed_h,
                                  beam_width=cfg.beam, pea=cfg.pea,
                                  iter_bound=cfg.iter_bound,
                                  pathmax=cfg.pathmax)
  return plan


def _milestone_candidate_session(cfg, C, assembly, robots, robot_ik):
  """The standard milestone search as a resumable stream of complete plans."""
  from solver import iter_assembly_problem_milestones
  return iter_assembly_problem_milestones(
      C, assembly, robots=robots, robot_ik=robot_ik, weight=cfg.weight,
      wrench=cfg.wrench, wrench_aware_h=cfg.wrench_aware_h,
      handovers=cfg.handovers, allow_place=cfg.allow_place,
      bracing=cfg.bracing, geo_facts=cfg.geo_facts,
      sig_collapsed_h=cfg.sig_collapsed_h, fill_window=cfg.fill_window,
      lazy_fills=cfg.lazy_fills, milestone_oracle=cfg.milestone_oracle,
      fill_local_h=cfg.fill_local_h, fill_exact=cfg.fill_exact,
      fill_raw_grasp_h=cfg.fill_raw_grasp_h,
      fill_target_support_quotient=cfg.fill_target_support_quotient,
      fill_greedy=cfg.fill_greedy,
      fill_incumbent_first=cfg.fill_incumbent_first,
      fill_policy_first=cfg.fill_policy_first,
      fill_policy_guide=cfg.fill_policy_guide,
      fill_skeleton_first=cfg.fill_skeleton_first,
      fill_action_widening=cfg.fill_action_widening,
      fill_scope_widening=cfg.fill_scope_widening,
      milestone_boundaries=cfg.milestone_boundaries,
      grasp_local_widen=cfg.grasp_local_widen,
      grasp_initial_keep=cfg.grasp_sampler_keep,
      grasp_widen_batch=cfg.grasp_widen_batch,
      grasp_widen_max=cfg.grasp_widen_max,
      placement_yaw_widening=cfg.placement_yaw_widening)


def _run_open_plan_loop(cfg, C, assembly, robots, robot_ik, explicit=None):
  """Fairly interleave resumable assembly-plan enumeration and finite MP pulls.

  Motion planning runs in isolated, memory-guarded child processes.  This is important beyond
  fault containment: rebuilding/exporting a candidate resets several global RNG/model knobs;
  doing it in the search process would corrupt the paused milestone session we are preserving.
  """
  import hashlib
  import json
  import shutil
  import subprocess
  import time
  from solver import collect_plan_witnesses, hoist_releases_certified
  from plan_io import require_complete_witnesses, witness_block

  if not cfg.motion_plan or not cfg.solve or cfg.load_plan:
    raise SystemExit("--mp-open-plans requires a fresh --solve --motion-plan run")
  if not cfg.milestones or any((cfg.optimistic, cfg.lazy, cfg.anytime is not None,
                                cfg.milestones_anytime is not None, cfg.ladder is not None,
                                cfg.ladder_lns is not None, cfg.bounded, cfg.dfbnb)):
    raise SystemExit("--mp-open-plans currently requires the ordinary milestone solver")
  if robot_ik is None:
    raise SystemExit("--mp-open-plans requires --real-ik")
  if not cfg.save or not cfg.mp_save:
    raise SystemExit("--mp-open-plans requires --save PLAN.json and --mp-save MP.json so the "
                     "winning pair can be promoted atomically")
  if cfg.mp_plan_candidates < 1 or cfg.mp_plan_epochs < 1:
    raise SystemExit("--mp-plan-candidates/--mp-plan-epochs must be >= 1")
  if cfg.mp_plan_initial_runtime <= 0.0 or cfg.mp_runtime <= 0.0:
    raise SystemExit("--mp-plan-initial-runtime/--mp-runtime must be positive")
  if not cfg.mp_optimize:
    raise SystemExit("--mp-open-plans requires --mp-optimize: a promoted production path must "
                     "spend the remainder of its successful pull on RRT* optimization. Use the "
                     "ordinary one-shot planner for a first-solution-only diagnostic.")

  # Bound the search process too, not only its MP children. This is a soft RLIMIT_AS installed
  # before milestone expansion; it prevents a hard Chair/fixture candidate from taking the
  # laptop with it while still leaving enough headroom for the measured ~3.9 GiB Chair export.
  import resource
  parent_limit = int(_os.environ.get("MP_OPEN_PARENT_AS", str(6 * 1024**3)))
  old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
  effective_parent_limit = min(parent_limit, old_hard) if old_hard >= 0 else parent_limit
  parent_limit_changed = old_soft < 0 or old_soft > effective_parent_limit
  if parent_limit_changed:
    resource.setrlimit(resource.RLIMIT_AS, (effective_parent_limit, old_hard))

  def restore_parent_limit():
    if parent_limit_changed:
      resource.setrlimit(resource.RLIMIT_AS, (old_soft, old_hard))

  stem, _ext = _os.path.splitext(_os.path.abspath(cfg.mp_save))
  run_tag = time.strftime("%Y%m%d-%H%M%S") + f"-{_os.getpid()}"
  work = f"{stem}-open/{run_tag}"
  _os.makedirs(work, exist_ok=False)
  manifest_path = _os.path.join(work, "manifest.json")
  base_seed = int(cfg.mp_seed if cfg.mp_seed is not None else cfg.seed)
  cap = float(cfg.mp_runtime)
  first = min(cap, float(cfg.mp_plan_initial_runtime))
  runtimes = [min(cap, first * (2 ** e)) for e in range(cfg.mp_plan_epochs)]
  records = []
  manifest = {
      "version": 1, "status": "running", "workdir": work,
      "policy": "complete-plan suspension + fair MP reopening",
      "candidate_limit": int(cfg.mp_plan_candidates), "runtimes": runtimes,
      "base_seed": base_seed, "candidates": records,
      "parent_memory_limit_bytes": effective_parent_limit,
      "grasp_refinement": bool(cfg.grasp_sampler and cfg.grasp_refine),
      "grasp_widening": bool(cfg.grasp_sampler and cfg.grasp_widen),
      "grasp_local_widening": bool(cfg.grasp_local_widen),
      "component_grasp_keep": int(cfg.component_grasp_keep),
  }
  seen_plan_ids = set()

  def write_manifest():
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
      json.dump(manifest, f, indent=2, sort_keys=True)
      f.write("\n")
    _os.replace(tmp, manifest_path)

  def plan_id(plan):
    raw = json.dumps([list(a) for a in plan], separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

  def prepare(raw_plan, index):
    plan = list(raw_plan)
    # The resumable milestone iterator can cross from its compact K->K+B session into the
    # finite-master completion session between candidate pulls.  Persist the cap that made
    # THIS candidate, not the command's initial K, before writing a replay/MP artifact.
    if cfg.grasp_local_widen and hasattr(assembly, "_grasp_local_caps"):
      cfg.grasp_local_caps = dict(assembly._grasp_local_caps)
    # Certify the candidate emitted by the paused search before touching its action order.
    # The transactional hoist below can then safely fall back to this exact source artifact.
    witnesses = collect_plan_witnesses(
        C, assembly, plan, robots, robot_ik,
        gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
        handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing,
        resolve_missing=True)
    require_complete_witnesses(plan, witnesses, who="open-plan source certificate")
    if cfg.hoist_releases:
      plan, witnesses, _hoist_applied = hoist_releases_certified(
          C, assembly, plan, witnesses, robots, robot_ik,
          gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
          handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing)
    pid = plan_id(plan)
    if pid in seen_plan_ids:
      print(f"[mp-open] duplicate complete plan {pid} skipped while widening search.")
      return None
    seen_plan_ids.add(pid)
    prefix = _os.path.join(work, f"candidate-{index:03d}-{pid}")
    pfile, efile = prefix + "-plan.json", prefix + "-export.json"
    require_complete_witnesses(plan, witnesses, who="open-plan search certificate")
    from solver import collect_plan_pre_witnesses
    pre_witnesses = collect_plan_pre_witnesses(
        C, assembly, plan, witnesses, robots, robot_ik,
        gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
        handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing,
        resolve_missing=True)
    required_pre = [j for j, a in enumerate(plan)
                    if a and a[0] == "assemble" and cfg.pre_insertion > 0.0
                    and assembly.screw_skill(a[2]) is None]
    missing_pre = [j for j in required_pre if j not in pre_witnesses]
    if missing_pre:
      raise ValueError("open-plan candidate lacks certified pre-insertion endpoints at "
                       + ", ".join(map(str, missing_pre)))
    save_plan(pfile, cfg, plan,
              witnesses=witness_block(list(robot_ik.C.getJointNames()), witnesses,
                                      stage="search", pre_witnesses=pre_witnesses),
              C=C, assembly=assembly, robots=robots)
    rec = {"index": index, "plan_id": pid, "actions": len(plan),
           "plan_path": pfile, "export_path": efile, "attempts": [],
           "status": "suspended"}
    records.append(rec)
    write_manifest()
    return rec

  def flag(name, value):
    return f"--{name}" if value else f"--no-{name}"

  def pull(rec, tier):
    runtime = runtimes[tier]
    seed = base_seed + rec["index"] - 1 + tier * 100003
    mpfile = (_os.path.splitext(rec["plan_path"])[0]
              + f"-tier{tier:02d}-seed{seed}-mp.json")
    cmd = [_sys.executable, _os.path.abspath(__file__),
           "--config", rec["plan_path"], "--load-plan", rec["plan_path"],
           "--motion-plan", flag("mp-sequence", cfg.mp_sequence),
           flag("mp-optimize", cfg.mp_optimize),
           "--mp-runtime", f"{runtime:g}", "--mp-seed", str(seed),
           "--mp-shortcut-iters", str(cfg.mp_shortcut_iters),
           "--mp-shortcut-strategy", cfg.mp_shortcut_strategy,
           "--mp-shortcut-pair-checks", str(cfg.mp_shortcut_pair_checks),
           "--mp-collision-tolerance", f"{cfg.mp_collision_tolerance:g}",
           "--mp-robot-margin", f"{cfg.mp_robot_margin:g}",
           "--mp-margin-ramp", f"{cfg.mp_margin_ramp:g}",
           "--mp-margin-mode", cfg.mp_margin_mode,
           "--mp-table-margin", f"{cfg.mp_table_margin:g}",
           flag("mp-wrench-gate", cfg.mp_wrench_gate),
           "--mp-save", mpfile, "--quiet-rai", "--no-view"]
    if getattr(cfg, "_motion_profile", None):
      cmd.extend(["--motion-profile", cfg._motion_profile])
    if cfg.mp_stall_timeout is not None:
      cmd.extend(["--mp-stall-timeout", f"{min(runtime, cfg.mp_stall_timeout):g}"])
    if not cfg.mp_shortcut:
      cmd.append("--no-mp-shortcut")
    if cfg.mp_shortcut_res is not None:
      cmd.extend(["--mp-shortcut-res", f"{cfg.mp_shortcut_res:g}"])
    if cfg.mp_fake_clock:
      cmd.append("--mp-fake-clock")
    # A killed/crashed first child may not have reached export despite being recorded as a
    # failed pull. Rebuild in that case; never feed a missing artifact to every later tier.
    if tier == 0 or not _os.path.exists(rec["export_path"]):
      cmd.extend(["--mp-export-save", rec["export_path"], "--save", rec["plan_path"]])
    else:
      cmd.extend(["--mp-export-load", rec["export_path"]])

    # Keep a failed candidate planner from taking the laptop with it.  Override only through
    # an explicit environment variable. Chair's current 4-robot witness-set export measured
    # ~3.87 GiB RSS, so 4 GiB is not viable once RRT* allocates its next tree; retain a bounded
    # 6 GiB default instead.
    limit = int(_os.environ.get("MP_OPEN_CHILD_AS", str(6 * 1024**3)))
    prlimit = shutil.which("prlimit")
    guarded = ([prlimit, f"--as={limit}", "--"] + cmd) if prlimit else cmd
    att = {"tier": tier, "runtime": runtime, "seed": seed, "mp_path": mpfile,
           "status": "running", "memory_limit_bytes": limit if prlimit else None}
    rec["attempts"].append(att)
    write_manifest()
    print(f"[mp-open] candidate {rec['index']}/{cfg.mp_plan_candidates} "
          f"{rec['plan_id']} tier {tier + 1}/{cfg.mp_plan_epochs}: "
          f"runtime={runtime:g}s seed={seed} (failed pull => suspend, never blacklist)",
          flush=True)
    try:
      cp = subprocess.run(guarded, cwd=_os.path.dirname(_os.path.abspath(__file__)),
                          timeout=max(600.0, runtime + 600.0))
      att["returncode"] = int(cp.returncode)
      has_path = _os.path.exists(mpfile) and _os.path.getsize(mpfile) > 0
      success = cp.returncode == 0 and has_path
      att["status"] = ("success" if success else
                       ("child_error" if cp.returncode != 0 else "no_path"))
    except subprocess.TimeoutExpired:
      success = False
      att["status"] = "child_timeout"
    write_manifest()
    return success, mpfile

  def promote(rec, mpfile):
    # Tier-0 certification intentionally re-saves the candidate with optimized witnesses, but
    # that child is invoked with its *pull* runtime (e.g. 15 s). Restore the parent's full
    # scheduling contract before promotion. The manifest remains the source of truth for the
    # selected attempt's exact tier/runtime/seed; the plan config reproduces the outer loop.
    with open(rec["plan_path"]) as f:
      selected_doc = json.load(f)
    selected_cfg = selected_doc.setdefault("config", {})
    for key in ("fuse_cograsp", "fuse_pre_skill", "mp_sequence", "mp_optimize",
                "mp_runtime", "mp_stall_timeout", "mp_shortcut", "mp_shortcut_iters",
                "mp_shortcut_res", "mp_shortcut_strategy", "mp_shortcut_pair_checks",
                "mp_collision_tolerance", "mp_robot_margin", "mp_margin_ramp",
                "mp_margin_mode",
                "mp_table_margin", "mp_cograsp", "mp_seed", "mp_open_plans",
                "mp_plan_candidates", "mp_plan_epochs", "mp_plan_initial_runtime"):
      selected_cfg[key] = getattr(cfg, key)
    selected_tmp = rec["plan_path"] + ".outer-config.tmp"
    with open(selected_tmp, "w") as f:
      json.dump(selected_doc, f, indent=2)
      f.write("\n")
    _os.replace(selected_tmp, rec["plan_path"])
    for src, dst in ((rec["plan_path"], cfg.save), (mpfile, cfg.mp_save),
                     (rec["export_path"], cfg.mp_export_save)):
      if dst:
        _os.makedirs(_os.path.dirname(_os.path.abspath(dst)), exist_ok=True)
        shutil.copy2(src, dst)
    rec["status"] = "selected"
    manifest.update({"status": "success", "selected_plan_id": rec["plan_id"],
                     "selected_candidate": rec["index"], "selected_mp_path": cfg.mp_save,
                     "selected_attempt": dict(rec["attempts"][-1])})
    write_manifest()
    print(f"[mp-open] SELECTED candidate {rec['index']} {rec['plan_id']}; promoted plan -> "
          f"{cfg.save}, motion path -> {cfg.mp_save}. Manifest: {manifest_path}")
    restore_parent_limit()
    return {"plan": cfg.save, "motion_path": cfg.mp_save,
            "manifest": manifest_path, "plan_id": rec["plan_id"]}

  session = _milestone_candidate_session(cfg, C, assembly, robots, robot_ik)
  widened_grasps = False
  widened_component_grasps = False
  staged_grasps = bool(cfg.grasp_sampler and cfg.grasp_refine)
  widen_keep = int(cfg.grasp_sampler_keep)
  widen_hash = cfg.grasp_sampler_db_hash
  # Epoch 0: move on quickly -- one small pull for each newly generated complete plan. If
  # the conventional primitive-grasp prefix is exhausted, preserve the ordinary standard
  # profile's staged refinement: reopen with the full hybrid sampled DB and skip duplicates.
  while len(records) < cfg.mp_plan_candidates:
    try:
      raw = next(session)
    except StopIteration:
      if staged_grasps and not widened_grasps:
        from problem import set_sampled_canonical_only
        widened_grasps = True
        set_sampled_canonical_only(False)
        manifest["grasp_refinement_widened"] = True
        write_manifest()
        print("[grasp-refine/mp-open] compact complete-plan stream exhausted -> reopening "
              "search with the full hybrid grasp DB; duplicate plans are skipped.")
        session = _milestone_candidate_session(cfg, C, assembly, robots, robot_ik)
        continue
      if cfg.component_grasp_keep and not widened_component_grasps:
        from problem import set_component_grasp_keep
        previous_keep = int(cfg.component_grasp_keep)
        widened_component_grasps = True
        set_component_grasp_keep(0)
        # Persist the alphabet which produces any later candidate. Stable grasp ids mean all
        # exact witnesses discovered under the restricted prefix remain reusable.
        cfg.component_grasp_keep = 0
        manifest["component_grasp_completion"] = {
            "from_keep": previous_keep, "to_keep": 0,
            "meaning": "exact full constituent-grasp union",
        }
        write_manifest()
        print(f"[component-grasps/mp-open] restricted K={previous_keep} complete-plan "
              "stream exhausted -> reopening with the exact full constituent-grasp union; "
              "suspended plans and witness evidence remain valid.")
        session = _milestone_candidate_session(cfg, C, assembly, robots, robot_ik)
        continue
      if (cfg.grasp_widen and cfg.grasp_sampler
          and widen_keep < int(cfg.grasp_widen_max)):
        if cfg.grasp_widen_batch < 1:
          raise ValueError("--grasp-widen-batch must be >= 1")
        from problem import set_sampled_grasp_keep, sampled_grasp_db_signature
        requested = min(int(cfg.grasp_widen_max), widen_keep + int(cfg.grasp_widen_batch))
        set_sampled_grasp_keep(requested)
        cfg.grasp_sampler_keep = requested
        new_hash = sampled_grasp_db_signature(assembly)
        if new_hash == widen_hash:
          set_sampled_grasp_keep(widen_keep)
          cfg.grasp_sampler_keep = widen_keep
          cfg.grasp_sampler_db_hash = widen_hash
          print(f"[grasp-widen/mp-open] finite evaluated master pool exhausted at "
                f"effective prefix {widen_keep}; requested {requested} added no candidate.")
        else:
          print(f"[grasp-widen/mp-open] complete-plan stream exhausted at prefix "
                f"{widen_keep} -> exposing stable ids [{widen_keep}, {requested}) and "
                "reopening assembly search. Existing plans remain suspended, not rejected.")
          widen_keep, widen_hash = requested, new_hash
          cfg.grasp_sampler_db_hash = new_hash
          manifest.setdefault("grasp_widen_epochs", []).append(
              {"keep": requested, "db_hash": new_hash})
          write_manifest()
          session = _milestone_candidate_session(cfg, C, assembly, robots, robot_ik)
          continue
      print(f"[mp-open] assembly-plan session exhausted after {len(records)} candidate(s).")
      break
    i = len(records) + 1
    rec = prepare(raw, i)
    if rec is None:
      continue
    ok, mpfile = pull(rec, 0)
    if ok:
      return promote(rec, mpfile)
    att = rec["attempts"][-1]
    rc = f", returncode={att['returncode']}" if "returncode" in att else ""
    print(f"[mp-open] candidate {i} suspended after tier 1: {att['status']}{rc}; "
          "requesting another complete plan.")

  # Later epochs: fairly reopen every suspended plan before increasing any one again.
  for tier in range(1, cfg.mp_plan_epochs):
    for rec in records:
      if rec["status"] == "selected":
        continue
      print(f"[mp-open] reopening candidate {rec['index']} at effort tier {tier + 1}.")
      ok, mpfile = pull(rec, tier)
      if ok:
        return promote(rec, mpfile)
      att = rec["attempts"][-1]
      rc = f", returncode={att['returncode']}" if "returncode" in att else ""
      print(f"[mp-open] candidate {rec['index']} remains suspended after tier {tier + 1}: "
            f"{att['status']}{rc}.")

  manifest["status"] = "exhausted_finite_budget"
  write_manifest()
  restore_parent_limit()
  raise SystemExit(
      f"[mp-open] no motion path within the configured FAIR finite effort. Nothing was "
      f"declared infeasible or blacklisted; {len(records)} suspended plan(s) and all evidence "
      f"remain in {manifest_path}. Increase --mp-plan-epochs/--mp-runtime to reopen them.")


def _should_hoist_plan(cfg, explicit=None):
  """Apply the release-hoist pass to fresh solves, or when explicitly requested on load."""
  return bool(cfg.hoist_releases and
              (not cfg.load_plan or (explicit and "hoist_releases" in explicit)))


def _merge_mp_export_clearance_settings(cfg, saved, explicit=None):
  """Layer motion-planner clearance knobs for ``--mp-export-load``.

  A final export is a zero-IK replay artifact, so its exact tasks/keyframes and structural
  settings remain authoritative.  Collision/clearance values are planner tuning knobs,
  however: the export supplies reproducible defaults, while an option explicitly present on
  THIS command line must win (the same CLI > file rule used by ``--config``).  Previously the
  assignments in the load branch unconditionally restored the export and silently made
  e.g. ``--mp-collision-tolerance 0.1`` ineffective.

  Returns ``(values, sources)`` where sources are suitable for the effective-settings log.
  ``explicit=None`` is the library-call case and deliberately keeps the saved artifact exact.
  """
  explicit = set(explicit or ())
  specs = (
      ("collision_tolerance", "mp_collision_tolerance", float),
      ("robot_robot_margin", "mp_robot_margin", float),
      ("robot_margin_ramp", "mp_margin_ramp", float),
      ("robot_margin_mode", "mp_margin_mode", str),
      ("table_margin", "mp_table_margin", float),
      ("wrench_gate", "mp_wrench_gate", bool),
  )
  values, sources = {}, {}
  for saved_key, cfg_key, cast in specs:
    cli_value = cast(getattr(cfg, cfg_key))
    if cfg_key in explicit:
      values[saved_key], sources[saved_key] = cli_value, "CLI"
    else:
      values[saved_key] = cast(saved.get(saved_key, cli_value))
      sources[saved_key] = "export"

  # table_float is not a CLI flag of its own.  It is the part/table start clearance paired
  # with --mp-table-margin: preserve the saved value unless that margin was explicitly
  # retuned, in which case derive it from this run's reconstructed scene.
  if "mp_table_margin" in explicit:
    values["table_float"] = float(cfg.part_float) if values["table_margin"] else 0.0
    sources["table_float"] = "CLI/scene"
  else:
    fallback = float(cfg.part_float) if values["table_margin"] else 0.0
    values["table_float"] = float(saved.get("table_float", fallback))
    sources["table_float"] = "export"
  return values, sources


def _set_search_placement_yaw_prefix(cfg, active):
  """Install the compact yaw prefix only while a fresh search is running.

  Placement IDs are persisted plan data.  Load-plan, replay, visualization and every
  post-processing pass must therefore see the complete configured table, even when the saved
  search used failure-driven yaw widening.  ``cfg.solve`` alone is insufficient because the
  CLI permits ``--solve --load-plan`` and load-plan deliberately wins without searching.
  """
  prefix = (1 if active and cfg.solve and not cfg.load_plan
            and cfg.placement_yaw_widening else None)
  from problem import set_placement_yaw_prefix
  set_placement_yaw_prefix(prefix)
  return prefix


def run(cfg, explicit=None):
  from problem import set_frozen_plan_grasps
  set_frozen_plan_grasps(None)  # a fresh search in this process must never inherit a loaded plan
  cfg._loaded_motion_data = None
  cfg._motion_data_cache = {}
  # cfg: RunConfig -- the single typed configuration object for the whole run. Every setting
  # below reads a cfg field; the thin main() wrapper builds cfg from the parsed CLI args.
  # explicit: set of dests the caller put on the command line (main()'s argv scan); None for
  # library callers. The dead-flag warnings below fire only for explicit flags.
  if cfg.assemble_ssik_pose_seeds and cfg.assemble_ssik_pose_projection:
    raise ValueError("--assemble-ssik-pose-seeds and --assemble-ssik-pose-projection "
                     "are alternative ablations; enable only one")
  _assembly_ik_ablations = sum(bool(x) for x in (
      cfg.assemble_unbiased_first, cfg.assemble_basin_racing,
      cfg.assemble_ssik_pose_seeds, cfg.assemble_ssik_pose_projection))
  if _assembly_ik_ablations > 1:
    raise ValueError("the neutral-first, basin-racing, and assembly-pose SSIK policies are "
                     "alternative ablations; enable only one")
  if cfg.assemble_race_probe_evals < 1 or cfg.assemble_race_promote < 1:
    raise ValueError("--assemble-race-probe-evals and --assemble-race-promote must be >= 1")
  if (cfg.handover_race_probe_evals < 1 or cfg.handover_race_starts < 1
      or cfg.handover_race_promote < 1):
    raise ValueError("--handover-race-probe-evals, --handover-race-starts, and "
                     "--handover-race-promote must be >= 1")
  if cfg.optimize_witnesses == "off":          # 'floor' is the default; 'off' is the escape
    cfg.optimize_witnesses = None
  explicit_set = set(explicit or ())
  if cfg.placement_yaw_widening and cfg.placement_yaws <= 1:
    if "placement_yaws" in explicit_set and "placement_yaw_widening" not in explicit_set:
      # A scalar family size of one has no widening to schedule. Let the natural one-flag
      # standard-profile ablation work; an explicitly contradictory pair still fails below.
      cfg.placement_yaw_widening = False
      print("[flag] explicit --placement-yaws 1 disables profile-owned placement-yaw "
            "widening (historical action-space ablation).")
    else:
      raise SystemExit("--placement-yaw-widening requires --placement-yaws N with N > 1")
  _profile = getattr(cfg, "_search_profile", None)
  if _profile:
    print(f"[profile] search '{_profile}' resolved: weight={cfg.weight:g}, "
          f"milestone-oracle={cfg.milestone_oracle}, "
          f"local-PDB={'on' if cfg.fill_local_h else 'off'}, "
          f"fill-order={'greedy->exact' if cfg.fill_incumbent_first else ('greedy' if cfg.fill_greedy else ('exact' if cfg.fill_exact else f'weighted({cfg.weight:g})'))}, "
          f"fill-scope={'all' if cfg.fill_touch_all else ('local->all' if cfg.fill_scope_widening else 'local-only')}, "
          f"milestone-boundaries={cfg.milestone_boundaries}, "
          f"witness-margin={float(cfg.witness_margin or 0.0)*1000:g}mm, "
          f"open-witness={'on' if cfg.open_witness else 'off'}"
          + (f" batch={cfg.open_witness_batch} epochs={cfg.open_witness_epochs}"
             f" selector={cfg.lazy_edge_selector}"
             if cfg.open_witness else "")
          + (f", placements=1->{cfg.placement_yaws} yaw bands"
             if cfg.placement_yaw_widening else f", placements={cfg.placement_yaws} yaw(s) eager")
          + f", analytic-seed-cap={cfg.analytic_seed_cap or 'all'}. "
          "Explicit individual flags remain authoritative.")
  _mp_profile = getattr(cfg, "_motion_profile", None)
  if _mp_profile:
    # mr_bench lives outside this repository and Numba cannot derive a writable source-local
    # cache locator for it in common venv installs.  A standard profile must not require two
    # undocumented exports merely to reach the first planner sample. Respect caller-provided
    # locations; otherwise install process-local, reusable caches under /tmp before the late
    # planner/matplotlib imports below.
    _cache_defaults = {
        "NUMBA_CACHE_DIR": "/tmp/assembly-numba-cache",
        "MPLCONFIGDIR": "/tmp/assembly-mpl-cache",
    }
    _cache_installed = []
    for _env_name, _cache_path in _cache_defaults.items():
      if not _os.environ.get(_env_name):
        _os.makedirs(_cache_path, exist_ok=True)
        _os.environ[_env_name] = _cache_path
        _cache_installed.append(f"{_env_name}={_cache_path}")
    if _cache_installed:
      print("[profile] writable caches installed: " + ", ".join(_cache_installed))
    _stall = "none" if cfg.mp_stall_timeout is None else f"{cfg.mp_stall_timeout:g}s"
    print(f"[profile] motion '{_mp_profile}' resolved: collision-tolerance="
          f"{cfg.mp_collision_tolerance:g}m, robot-margin={cfg.mp_robot_margin:g}m, "
          f"ramp={cfg.mp_margin_ramp:g}m/{cfg.mp_margin_mode}, "
          f"table-margin={cfg.mp_table_margin:g}m, "
          f"runtime={cfg.mp_runtime:g}s, stall={_stall}, "
          f"RRT*-optimize={'on' if cfg.mp_optimize else 'off'}, "
          f"shortcuts={cfg.mp_shortcut_iters}. Explicit individual flags remain authoritative.")
  set_seed(cfg.seed)                           # reproducible numpy + rai (KOMO) RNG
  # MIXED END EFFECTORS (--grippers): per-robot vacuum/two_finger list; overrides --vacuum.
  gripper_list = None
  if cfg.grippers:
    ks = [k.strip() for k in cfg.grippers.split(",") if k.strip()]
    bad = [k for k in ks if k not in ("vacuum", "two_finger")]
    if bad:
      raise SystemExit(f"--grippers: unknown kind(s) {bad} (vacuum|two_finger)")
    ks = (ks + [ks[-1]] * cfg.num_robots)[:cfg.num_robots]   # last entry repeats
    gripper_list = [k == "vacuum" for k in ks]
    if len(set(ks)) > 1:
      if cfg.anon_robots:
        raise SystemExit("--grippers (mixed) is incompatible with --anon-robots: "
                         "robot identity decides which gripper stands at a grasp.")
      print(f"[flag] --grippers ACTIVE (MIXED EEs): {', '.join(f'a{i+1}={k}' for i, k in enumerate(ks))}. "
            "Grasps are kind-mask filtered per robot; wrench prices each grasp with its "
            "holder's capability; labeled-h pick facts OR over kinds (admissible).")
    else:
      cfg.vacuum = ks[0] == "vacuum"                      # uniform via --grippers = plain path

  # grasp rolls are a property of the gripper: the vacuum is symmetric about its suction
  # axis (1 grasp per face); the two-finger has two in-plane orientations (90 deg apart).
  # --dense-grasps additionally enables the +-45deg WRIST-TILTED variants (same face pinch,
  # approach between the faces) -- see problem.set_dense_grasps. MIXED scenes take the
  # two-finger's 2 rolls whenever any two-finger arm exists (the vacuum ignores rolls).
  any_two_finger = (gripper_list is not None and not all(gripper_list)) or not cfg.vacuum
  set_rolls_per_face(2 if any_two_finger else 1)
  set_dense_grasps(cfg.dense_grasps and any_two_finger)
  from problem import (set_sampled_grasps, clear_sampled_scene_ctx,
                       set_grasp_physics_check,
                       set_component_grasp_keep)
  if cfg.grasp_sampler_scenes and not cfg.grasp_sampler:
    raise SystemExit("--grasp-sampler-scenes requires --grasp-sampler")
  if cfg.grasp_sampler_full_probe and not cfg.grasp_sampler:
    raise SystemExit("--grasp-sampler-full-probe requires --grasp-sampler")
  if cfg.grasp_sampler_roll_refine and not cfg.grasp_sampler:
    raise SystemExit("--grasp-sampler-roll-refine requires --grasp-sampler")
  if cfg.grasp_physics_check and not cfg.grasp_sampler:
    raise SystemExit("--grasp-physics-check requires --grasp-sampler")
  if cfg.grasp_physics_cache_plan and not cfg.grasp_physics_check:
    raise SystemExit("--grasp-physics-cache-plan requires --grasp-physics-check")
  if cfg.grasp_physics_check and not cfg.grasp_sampler_scenes:
    raise SystemExit("--grasp-physics-check currently requires --grasp-sampler-scenes so "
                     "the finite DB can reserve physics-certified pick/assembly grasps")
  if cfg.grasp_local_widen and cfg.grasp_widen:
    # Named standard currently carries the older whole-search retry policy. An explicit
    # local schedule replaces it cleanly instead of forcing profile users to remember a
    # second negating flag; the resolved/saved config records the actual policy.
    print("[flag] --grasp-local-widen supersedes --grasp-widen; disabling the "
          "whole-search restart schedule.")
    cfg.grasp_widen = False
  # These are search-schedule flags. A replay/export-only invocation may retain their saved
  # provenance while never selecting a search engine at all, so validate them only when this
  # process is actually asked to solve.
  if cfg.solve and not cfg.load_plan:
    _validate_solver_scoped_flags(cfg)
  if cfg.grasp_widen:
    if not cfg.grasp_sampler:
      raise SystemExit("--grasp-widen requires --grasp-sampler")
    if cfg.grasp_widen_batch < 1:
      raise SystemExit("--grasp-widen-batch must be >= 1")
    if cfg.grasp_widen_max < cfg.grasp_sampler_keep:
      raise SystemExit("--grasp-widen-max must be >= --grasp-sampler-keep")
  if cfg.grasp_local_widen:
    if not cfg.grasp_sampler:
      raise SystemExit("--grasp-local-widen requires --grasp-sampler")
    if cfg.grasp_widen_batch < 1:
      raise SystemExit("--grasp-widen-batch must be >= 1")
    if cfg.grasp_widen_max < cfg.grasp_sampler_keep:
      raise SystemExit("--grasp-widen-max must be >= --grasp-sampler-keep")
  if cfg.component_grasp_keep < 0:
    raise SystemExit("--component-grasp-keep must be >= 0")
  if cfg.component_grasp_keep and not cfg.grasp_sampler:
    raise SystemExit("--component-grasp-keep requires --grasp-sampler")
  set_component_grasp_keep(cfg.component_grasp_keep)
  if cfg.component_grasp_keep:
    print(f"[flag] --component-grasp-keep {cfg.component_grasp_keep} ACTIVE: singleton "
          "databases stay complete; every multi-part rigid component uses a stable "
          "round-robin pool of at most this many whole-component-clear grasps. A no-plan "
          "return automatically widens once to the exact historical full union.")
  # a previous in-process run's scene filter must not leak into THIS run's assembly
  # definition (define() queries the grasp DB, which raises under a stale part-less ctx).
  clear_sampled_scene_ctx()
  if cfg.grasp_physics_force < 0.0:
    raise SystemExit("--grasp-physics-force must be non-negative")
  set_grasp_physics_check(cfg.grasp_physics_check, cfg.grasp_physics_force)
  if cfg.grasp_sampler:
    if cfg.dense_grasps or cfg.grasp_translations:
      # dense/translations are FACE-ENUMERATOR variant families and several id-arithmetic
      # sites assume their block layout (g % nb collapses, nb-capped fill phases) -- with
      # sampled ids those silently corrupt the grasp identity space.
      raise SystemExit("--grasp-sampler is incompatible with --dense-grasps / "
                       "--grasp-translations (face-enumerator variant families).")
    set_sampled_grasps({"seed": cfg.grasp_sampler_seed,
                        "candidates": cfg.grasp_sampler_candidates,
                        "full_probe": cfg.grasp_sampler_full_probe,
                        "roll_refine": cfg.grasp_sampler_roll_refine,
                        "keep": cfg.grasp_sampler_keep,
                        "master_keep": (cfg.grasp_widen_max if (cfg.grasp_widen or cfg.grasp_local_widen)
                                        else cfg.grasp_sampler_keep),
                        "fixed_master": bool(cfg.grasp_widen and not cfg.grasp_local_widen),
                        "margin": cfg.grasp_sampler_margin,
                        "pad_support": cfg.grasp_sampler_pad_support,
                        "wrap": cfg.grasp_sampler_wrap,
                        "strict": cfg.grasp_sampler_strict})
    print("[flag] --grasp-sampler ACTIVE: SCENE-RESERVED HYBRID conventional + mesh-sampled"
          + f" grasp DB ({cfg.grasp_sampler_candidates} raw candidates -> "
          + (f"{cfg.grasp_sampler_keep} initially active, finite master requested up to "
             f"{cfg.grasp_widen_max} per part"
             if (cfg.grasp_widen or cfg.grasp_local_widen) else
             f"{cfg.grasp_sampler_keep} total kept per part")
          + f"; suction + antipodal blocks; seed "
          f"{cfg.grasp_sampler_seed}; {'strict quality gates' if cfg.grasp_sampler_strict else 'fallback quality rows allowed'}). Grasp ids index the SAMPLED list -- plans/witnesses "
          f"only replay under the same sampler settings (saved with the plan).")
    if cfg.grasp_sampler_full_probe:
      print("[flag] --grasp-sampler-full-probe ACTIVE: adaptively classifying the complete "
            "raw cloud when the compact role pool has no direct pick-to-mate bridge. This "
            "changes the positional grasp database and may make preprocessing slower.")
    if cfg.grasp_sampler_roll_refine:
      print("[flag] --grasp-sampler-roll-refine ACTIVE: appending the nested 45-degree "
            "pinch-roll lattice for transfer/co-grasp compatibility.")
    if cfg.grasp_widen:
      print(f"[flag] --grasp-widen ACTIVE: generate one scene-ranked finite master up to "
            f"K={cfg.grasp_widen_max}; start with K={cfg.grasp_sampler_keep}; after a "
            f"no-plan return expose {cfg.grasp_widen_batch} stable IDs per epoch. "
            "Finite misses remain open; the final effective K/hash are saved.")
    if cfg.grasp_local_widen:
      print(f"[flag] --grasp-local-widen ACTIVE: one stable finite master is built up to "
            f"K={cfg.grasp_widen_max}; concrete fills start with K={cfg.grasp_sampler_keep} "
            f"per part and expose {cfg.grasp_widen_batch} more incoming-part columns after "
            "a local failure. If the compact resumable session exhausts, one eager "
            "finite-master session follows. IK misses remain open and grasp IDs never change.")
  else:
    set_sampled_grasps(None)
  from problem import set_placement_yaws, set_stable_rest_poses
  set_stable_rest_poses(bool(getattr(cfg, "stable_rest_poses", False)))
  set_placement_yaws(cfg.placement_yaws)
  # The compact prefix is a search schedule, never part of the scene/plan identity.  Keep the
  # full table visible while constructing the scene and for all no-search consumers; the
  # search branch below activates prefix 1 inside its own try/finally scope.
  _set_search_placement_yaw_prefix(cfg, active=False)
  if getattr(cfg, "stable_rest_poses", False):
    print("[flag] --stable-rest-poses ACTIVE: placements come from the shape's TRUE stable "
          "rest orientations (convex hull + mass-weighted centre of mass), not the six "
          "axis-aligned faces. Placement id 5 still means 'as anchored'. Grasp ids are "
          "unaffected; PLACEMENT ids rebind, so plans do not cross the flag.")
  if cfg.placement_yaws != 1:
    print(f"[flag] --placement-yaws {cfg.placement_yaws} ACTIVE: each stable table rest is "
          "augmented with a uniform world-z yaw lattice. Support stability is shared, but "
          "every yaw has its own replay-safe placement id and IK witness; place branching "
          f"grows by up to {cfg.placement_yaws}x.")
  if cfg.placement_yaw_widening:
    print("[flag] --placement-yaw-widening ACTIVE: each milestone fill starts with the "
          "historical yaw only, opens the finite lattice coarse-to-fine on local failure, "
          "then admits all-part manipulation. The complete configured lattice remains the "
          "final action-space net.")
  if cfg.plain_box:
    from problem import set_plain_box
    set_plain_box(True)
    print("[flag] --plain-box ACTIVE: cube parts use sharp ST.box (no ssBox corner-radius "
          "inflation), for solve + export + replay.")
  if cfg.grasp_translations:
    from problem import set_grasp_translations
    set_grasp_translations(cfg.grasp_translations)
    print(f"[flag] --grasp-translations {cfg.grasp_translations}: each base grasp also offered "
          f"SLID along the gripper's open directions (bounded so the pads stay on the part).")
  if cfg.quiet_rai:
    import rai_quiet
    rai_quiet.install()                                  # no-op if RAI_QUIET already installed it
    print("[flag] --quiet-rai ACTIVE: rai's libccd/pairCollision spam is dropped at fd 1/2 "
          "(matched on the C++ source file, never on 'WARNING', so real diagnostics survive).")
  from env import set_tight_fingers
  set_tight_fingers(bool(cfg.tight_fingers))
  if cfg.tight_fingers:
    if cfg.robot == "openarm":
      print("[flag] --tight-fingers requested: OpenArm keeps its native calibrated pinch "
            "capsules (the Robotiq-only tightening transform does not apply).")
    else:
      print("[flag] --tight-fingers ACTIVE: finger COLLISION capsules shrunk to "
            "robotiq_tight.g's radius (20mm -> 10mm, shifted 8mm so the grip surface holds). "
            "The fat proxy reaches 32.9mm past the tool centre -- it is what pokes out the far "
            "side of small parts and what forces --part-float.")
  from env import set_openarm_real_finger_radius_scale
  set_openarm_real_finger_radius_scale(cfg.openarm_real_finger_radius_scale)
  if cfg.openarm_real_finger_radius_scale != 1.0:
    if cfg.robot != "openarm":
      raise ValueError("--openarm-real-finger-radius-scale only applies to --robot openarm")
    print(f"[diag] --openarm-real-finger-radius-scale "
          f"{cfg.openarm_real_finger_radius_scale:g}: REAL OpenArm finger capsule radius "
          f"15mm -> {15.0 * cfg.openarm_real_finger_radius_scale:g}mm; sampled grasps and "
          "the floating closure proxy remain nominal for a controlled A/B test.")
  from solver import set_grasp_closure_check
  _closure_active = bool(cfg.grasp_closure_check or cfg.viz_initial_grasps is not None)
  set_grasp_closure_check(_closure_active)
  if cfg.grasp_closure_check:
    print("[flag] --grasp-closure-check ACTIVE: two-finger pick/add_grasp use a "
          "collision-only virtual jaw sweep; Robotiq handovers rotate the articulated "
          "finger linkage in a disposable RAI scene. Planning joints remain disabled.")
  elif cfg.viz_initial_grasps is not None:
    print("[initial-grasp-viz] closure enabled during scene-aware DB construction so the "
          "catalog uses the same candidate IDs/ranking as a closure-enabled search.")
  set_brace_for_assembly_only(cfg.brace_for_assembly_only)
  set_extra_place_cells(cfg.place_cells)
  set_prefer_center(cfg.prefer_center)
  if cfg.place_cells:
    print(f"[flag] --place-cells ACTIVE: {cfg.place_cells} extra translated placement "
          f"anchors per (group, face)"
          + (", CENTRE-FIRST ordering" if cfg.prefer_center else "") + ".")
  if cfg.brace_for_assembly_only:
    print("[flag] --brace-for-assembly-only ACTIVE: a braced state (>=2 grippers on a group) "
          "may only assemble/release. Prunes brace branching; can miss plans that carry a "
          "brace across unrelated work (experimental).")
  if cfg.lru_robots:
    from problem import set_lru_robots
    set_lru_robots(True)
    print("[flag] --lru-robots ACTIVE: successor robots ordered least-recently-used first "
          "(deterministic tie-break; spreads actions across arms).")
  if cfg.brace_resting:
    from problem import set_brace_resting
    set_brace_resting(True)
    print("[flag] --brace-resting ACTIVE: resting assemblies may be braced (add_grasp), "
          "matching the abstraction's resting+braced states.")
  # Reset both switches on every in-process run. A standard-profile run must not leak this
  # ordering into a later --no-useful-grasps ablation in the same benchmark worker.
  from problem import set_useful_grasps_enabled, set_useful_grasps
  set_useful_grasps_enabled(cfg.useful_grasps)
  if not cfg.useful_grasps:
    set_useful_grasps(None)
  if cfg.useful_grasps:
    print("[flag] --useful-grasps ACTIVE: mate-viable grasp sets ordered first + fill-h "
          "regrasp pricing (spec 6.2).")
  from solver import set_fill_touch_all
  set_fill_touch_all(cfg.fill_touch_all)                 # reset for repeated in-process runs
  if cfg.fill_scope_widening:
    print("[flag] --fill-scope-widening ACTIVE: a failed local-scope fill is retried with "
          "all-part manipulation before the milestone is rejected.")
  if cfg.fill_action_widening:
    print("[flag] --fill-action-widening ACTIVE: local fills search nested concrete "
          "action alphabets (direct transport -> handover -> configured placement/bracing); "
          "restricted failures only widen and never declare infeasibility.")
  if cfg.fill_touch_all:
    print("[flag] --fill-touch-all ACTIVE: fills skip the cheap local-scope pass and may "
          "manipulate any part immediately; only their requested join may assemble and "
          "temporarily moved blockers must be settled. Local search may be much larger.")
  if cfg.lazy_edge_selector == "terminal-first" and not cfg.open_witness:
    raise SystemExit("--lazy-edge-selector terminal-first requires --open-witness")
  if cfg.dominance:
    from problem import set_dominance
    set_dominance(True)
    warn = "" if cfg.floating_ee else " WARNING: real fixed-base arms are not exact twins -- may prune plans."
    print(f"[flag] --dominance ACTIVE: one free arm per gripper kind acts (idle-twin symmetry).{warn}")
  if cfg.allow_curved_grasps:
    from problem import set_allow_curved_grasps
    set_allow_curved_grasps(True)
    print("[flag] --allow-curved-grasps ACTIVE: vacuum keeps curved-surface grasps (idealised "
          "suction on cylinders/spheres).")
  if cfg.staged_mates:
    from problem import set_staged_mates
    set_staged_mates(True)
    print("[flag] --staged-mates ACTIVE: staged mates are SLOTS until their fastener lands "
          "(unheld unfastened part falls out unless the slot faces up); successor filter + "
          "labeled-h gate, admissible. Assemblies without staged_mates are unaffected.")
  if cfg.inhand_reorient:
    from problem import set_inhand_reorient
    set_inhand_reorient(True)
    print("[flag] --inhand-reorient ACTIVE: held lone parts may rotate to any other grasp "
          "in hand (v1: full connectivity, unchanged grip capability, solo-held lone parts). "
          "Labeled/collapsed h model the inverse-reorient edge (cost 1) -> h stays admissible.")
  import wrench
  wrench.PARAMS["grasp_model"] = cfg.grasp_model  # capability model (used when --wrench)
  wrench.apply_wrench_profile(cfg.wrench_profile)
  if cfg.wrench_profile is not None:
    _tf = wrench.PARAMS["grasp"][cfg.grasp_model]["two_finger"]
    _vac = wrench.PARAMS["grasp"][cfg.grasp_model]["vacuum"]
    print(f"[flag] --wrench-profile {cfg.wrench_profile} ACTIVE: gripper envelope installed "
          f"for {cfg.grasp_model}: two-finger={_tf}, vacuum={_vac}. Explicit cap overrides "
          "are applied afterwards.")
  if cfg.table_mu is not None:
    wrench.PARAMS["table_model"] = "friction"
    wrench.PARAMS["table_mu"] = float(cfg.table_mu)
    print(f"[flag] --table-friction ACTIVE: table support is a friction cone (mu {cfg.table_mu:g}); "
          "the frictionless unilateral default rejects ANY lateral load on a table-rest "
          "target (e.g. every --insertion-noise corner).")
  if cfg.vacuum_engage_force < 0.0:
    raise ValueError("--vacuum-engage-force must be non-negative")
  wrench.PARAMS["vacuum_engage_force"] = float(cfg.vacuum_engage_force)
  if cfg.wrench and cfg.vacuum_engage_force > 0.0:
    print(f"[flag] --vacuum-engage-force {cfg.vacuum_engage_force:g}N ACTIVE: every vacuum "
          "pick/add_grasp/handover must have its pre-existing supports balance the cup's "
          "transient approach-axis sealing load.")
  wrench.PARAMS["vacuum_engage_friction"] = float(cfg.vacuum_engage_friction or 0.0)
  if cfg.wrench and cfg.vacuum_engage_friction:
    print(f"[flag] --vacuum-engage-friction {cfg.vacuum_engage_friction:g} ACTIVE: inside "
          "the ENGAGEMENT gate only, the table gets a static friction cone (arctan(mu) "
          f"~= {float(np.degrees(np.arctan(cfg.vacuum_engage_friction))):.0f} deg admissible "
          "tilt); every other gate keeps the global table model.")
  if cfg.vacuum_scale is not None:
    vs = float(cfg.vacuum_scale)                          # scale EVERY vacuum force+torque cap
    for _model in wrench.PARAMS["grasp"].values():         # all models; mu (a ratio) is left as-is
      vac = _model.get("vacuum")
      if vac:
        for _k in vac:
          if _k != "mu":
            vac[_k] = vac[_k] * vs
    print(f"[flag] --vacuum-scale ACTIVE: vacuum capability x{vs:g} (all forces + torques; "
          "two-finger and mu unchanged).")
  if cfg.two_finger_scale is not None:
    ts = float(cfg.two_finger_scale)                     # scale EVERY two-finger force+torque cap
    for _model in wrench.PARAMS["grasp"].values():        # all models; mu (a ratio) is left as-is
      tf = _model.get("two_finger")
      if tf:
        for _k in tf:
          if _k != "mu":
            tf[_k] = tf[_k] * ts
    print(f"[flag] --two-finger-scale ACTIVE: two-finger capability x{ts:g} "
          "(all forces + torques; vacuum and mu unchanged).")
  if cfg.tf_close_torque is not None:
    _tf_asym = wrench.PARAMS["grasp"]["asymmetric"]["two_finger"]
    _tf_asym["T_close"] = float(cfg.tf_close_torque)
    if "T_t2" in _tf_asym:  # axis-resolved named profiles must not shadow the explicit flag
      _tf_asym["T_t2"] = float(cfg.tf_close_torque)
    print(f"[flag] --tf-close-torque ACTIVE: two-finger closing-axis torque cap = "
          f"{cfg.tf_close_torque:g} Nm. Models a pinch that rotates out of hand.")
  if cfg.grasp_compliance is not None:
    wrench.PARAMS["grasp_compliance"] = {"two_finger": 1.0, "vacuum": float(cfg.grasp_compliance)}
    print(f"[flag] --grasp-compliance ACTIVE: vacuum-held load paths inflate the insertion-"
          f"uncertainty box x{cfg.grasp_compliance:g} (two-finger 1.0). Effective with "
          "--wrench + --insertion-noise; picks/carries unaffected.")
  wrench.PARAMS["check_inserter"] = cfg.inserter_check
  wrench.PARAMS["insert_force"]["_default"] = cfg.insert_force
  wrench.PARAMS["curved_grasp_torque_scale"] = cfg.curved_torque_scale
  wrench.PARAMS["insertion_uncertainty"] = (
      None if cfg.insertion_noise is None
      else {"lateral": cfg.insertion_noise, "offset": cfg.insertion_offset})
  wrench.PARAMS["robust_mode"] = cfg.robust_mode
  wrench.PARAMS["ignore_gravity"] = cfg.ignore_gravity
  wrench.PARAMS["orientation_robust_gravity"] = cfg.orientation_robust_wrench
  if cfg.wrench and cfg.orientation_robust_wrench:
    print("[flag] --orientation-robust-wrench ACTIVE: every freely held component must "
          "balance gravity for the complete continuous orientation sphere; exact cached "
          "contact-zonotope containment, no orientation samples or IK. Table-resting "
          "components keep their discrete rest orientation.")
  from problem import set_banned_actions
  set_banned_actions([x.strip() for x in cfg.ban_actions.split(",") if x.strip()]
                     if cfg.ban_actions else ())
  if cfg.ban_actions:
    print(f"[flag] --ban-actions ACTIVE: {cfg.ban_actions} (successor generator filters "
          f"matching actions; search only)")
  if cfg.vacuum_no_assemble:
    from problem import set_kind_forbid
    set_kind_forbid(("vacuum:assemble",))
    print("[flag] --vacuum-no-assemble ACTIVE: no assemble may involve a vacuum arm "
          "(actor, inserted-part/target holder, or brace) -- the vacuum may pick and hand "
          "over only. This explicit tool-role restriction remains active with --no-wrench; "
          "that ablation disables equilibrium/capacity checks, not the declared action set.")
  if cfg.ignore_gravity:
    print("[flag] --ignore-gravity ACTIVE: parts weightless in the wrench check "
          "(only insertion/applied loads must be resisted).")
  set_goal(on_table=cfg.goal_on_table)
  if cfg.goal_on_table:
    print("[flag] --goal-on-table ACTIVE: the finished assembly must rest on the table "
          "and be released (hands-off).")

  if cfg.robot == "fr3":
    num_duos = max(1, cfg.num_robots // 2)
    if cfg.num_robots != 2 * num_duos:
      print(f"[flag] fr3 duos: --num-robots {cfg.num_robots} -> {2 * num_duos} "
            f"({num_duos} duo(s))")
    C = make_fr3_duo_env(num_duos=num_duos)
  elif cfg.robot == "openarm":
    if cfg.floating_ee:
      C = make_openarm_floating_env(num_robots=cfg.num_robots)
      print(f"[flag] --robot openarm --floating-ee ACTIVE: {cfg.num_robots} free 6-DOF "
            "native OpenArm pinch hand(s); no fixed arms, reach limits, or SSIK.")
    else:
      num_duos = max(1, cfg.num_robots // 2)
      if cfg.num_robots != 2 * num_duos:
        print(f"[flag] openarm duos: --num-robots {cfg.num_robots} -> {2 * num_duos} "
              f"({num_duos} duo(s))")
      C = make_openarm_duo_env(num_duos=num_duos)
  elif cfg.robot == "husky":
    if cfg.num_robots != 2:
      raise SystemExit("--robot husky contains exactly two calibrated arms; use --num-robots 2")
    if cfg.floating_ee:
      raise SystemExit("--robot husky does not yet have a calibrated --floating-ee model")
    if cfg.vacuum or (gripper_list is not None and any(gripper_list)):
      raise SystemExit("--robot husky currently has two fixed Robotiq grippers; do not request vacuum")
    C = make_husky_duo_env()
  else:
    C = make_four_robot_env(
        num_robots=cfg.num_robots,
        floating_ee=cfg.floating_ee,
        vacuum=gripper_list if gripper_list is not None else cfg.vacuum,
        robot_types=cfg.robot_types or cfg.robot,   # --robot for all; --robot-types overrides
    )

  from problem import assembly_max_parts
  _maxp = assembly_max_parts(cfg.assembly)
  if cfg.parts > _maxp:
    if explicit and "parts" in explicit:
      print(f"[flag] !!! --parts {cfg.parts} exceeds {cfg.assembly}'s maximum of {_maxp}: "
            f"clamped to {_maxp} -- this run builds {_maxp} parts.")
    else:
      print(f"[flag] NOTE: --assembly {cfg.assembly} has at most {_maxp} parts; the default "
            f"--parts {cfg.parts} is clamped to {_maxp}.")
  assembly = make_assembly(cfg.assembly, cfg.parts)
  from problem import set_mate_wrench_limits
  set_mate_wrench_limits(cfg.mate_wrench_limits)
  if assembly.mate_wrench_limits:
    if cfg.mate_wrench_limits:
      print("[flag] --mate-wrench-limits ACTIVE: finite internal mate boxes enforced for "
            + ", ".join(sorted(assembly.mate_wrench_limits, key=lambda p: int(p.split('_')[1])))
            + "; ideal mate edges remain rigidly contracted.")
      if cfg.wrench and cfg.orientation_robust_wrench:
        raise SystemExit("--orientation-robust-wrench is not yet supported with finite "
                         "--mate-wrench-limits; use the default fixed-orientation model")
      if cfg.wrench and cfg.robust_mode == "affine" and cfg.insertion_noise is not None:
        raise SystemExit("--robust-mode affine is not yet supported with finite "
                         "--mate-wrench-limits; use --robust-mode corners")
    else:
      print("[flag] --no-mate-wrench-limits ACTIVE: authored finite mate caps ignored; "
            "using the historical ideal-weld baseline.")
  assembly.configure_mate_symmetry(cfg.mate_symmetry)
  if cfg.mate_symmetry:
    _sets = {p: len(qs) for p, qs in assembly.mate_symmetry_quats.items()}
    print("[flag] --mate-symmetry ACTIVE: one symbolic assemble edge tries each authored "
          f"discrete relative orientation in IK; sets={_sets or '{}'}.")
  assembly.configure_screw_skills(
      angle=(None if cfg.screw_angle is None else np.deg2rad(cfg.screw_angle)),
      depth=cfg.screw_depth, torque=cfg.screw_torque,
      angular_speed=cfg.screw_speed, samples=cfg.screw_samples)
  if assembly.mate_skills:
    for _part, _skill_spec in assembly.mate_skills.items():
      print(f"[skill] {_part} screw: angle={np.degrees(_skill_spec.angle):g}deg, "
            f"depth={_skill_spec.depth*1000:g}mm, torque={_skill_spec.torque:g}Nm, "
            f"speed={_skill_spec.angular_speed:g}rad/s, samples={_skill_spec.samples}; "
            "single grasp, atomic assemble action")

  if cfg.ladder_lns is not None:
    print(f"[flag] --ladder-lns ACTIVE: budget {cfg.ladder_lns:g}s (k=1 -> LNS sweeps "
          f"-> RWA*-under-bound proof).")
  if cfg.ladder is not None:
    print(f"[flag] --ladder ACTIVE: budget {cfg.ladder:g}s (k=1 -> k=2 -> w=1 proof "
          f"rung, monotone incumbent bound).")
  if cfg.milestones_anytime is not None:
    print(f"[flag] --milestones-anytime ACTIVE: budget {cfg.milestones_anytime:g}s "
          f"(milestone incumbent -> k=1.."
          f"{cfg.lns_window if cfg.lns_window is not None else 'full horizon'} LNS; "
          "each window greedy -> bounded w=1; "
          "PEA-greedy feasibility net only if the incumbent phase is cut).")
  if cfg.lns_grasp_transition_h:
    print("[flag] --lns-grasp-transition-h ACTIVE: LNS reverse loose-grasp handover edges "
          "require isolated floating-gripper coexistence (experimental; profiles unchanged).")
  if cfg.corridor_h:
    print("[flag] --corridor-h ACTIVE: phase-2 heuristic built only up to the incumbent "
          "bound (misses prune); requires --bounded.")
  if cfg.pathmax:
    print("[flag] --pathmax ACTIVE: h consistency repair (h_child >= h_parent - cost); "
          "admissible, fewer f-bumps/frontier sweeps.")
  if cfg.pea:
    print("[flag] --pea ACTIVE: partial-expansion A* engine (banded open list; same "
          "guarantees, frontier memory / ~branching factor).")
  if cfg.iter_bound:
    print("[flag] --iter-bound ACTIVE: iterative cost-bound deepening -- first plan found "
          "is proven optimal; --weight only orders exploration within passes.")
  if cfg.beam is not None:
    print(f"[flag] --beam ACTIVE: beam search, width {cfg.beam} (memory-bounded, no "
          f"quality bound{'; incumbent fallback for --bounded' if cfg.bounded or cfg.dfbnb else ''}).")
  if cfg.dfbnb:
    print("[flag] --dfbnb ACTIVE: depth-first branch-and-bound certificate phase "
          "(implies --bounded).")
  if cfg.sig_collapsed_h:
    if cfg.sig_components and cfg.sig_components != "all":
      set_sig_components([c.strip() for c in cfg.sig_components.split(",") if c.strip()])
    print("[flag] --sig-collapsed-h ACTIVE: signature grasp-equivalence classes + labeled "
          "loose parts (components: "
          + (cfg.sig_components if cfg.sig_components and cfg.sig_components != "all"
             else "press,pickface,contact,solohold,braceside") + ").")
  if cfg.geo_facts:
    print("[flag] --geo-facts ACTIVE: labeled-abstraction backward relay gated on "
          "floating-gripper coexistence (gripper-vs-assembly/gripper/table); sound "
          "necessary condition, h stays admissible.")
  if cfg.por:
    print("[flag] --por ACTIVE: partial-order reduction over independent successor actions "
          "(EXPERIMENTAL, unsound when part/arm geometries interact -- use --min-sep).")
  if cfg.grid_init is not None:
    print(f"[flag] --grid-init {cfg.grid_init} ACTIVE: parts packed shoulder-to-shoulder in a "
          f"{cfg.grid_init}-wide grid (relocation stress); scatter line / --random-init ignored.")
  if (explicit and "scatter_spacing" in explicit
      and (cfg.random_init is not None or cfg.grid_init is not None)):
    print(f"[flag] NOTE: --scatter-spacing {cfg.scatter_spacing:g} has NO EFFECT here: "
          f"{'--random-init' if cfg.random_init is not None else '--grid-init'} overrides the "
          f"scatter line.")
  if cfg.center_scatter and cfg.grid_init is None and cfg.random_init is None:
    print("[flag] --center-scatter ACTIVE: parts use a compact centred lattice; spacing and "
          "part order are unchanged, layout span grows as sqrt(part count).")
  C = add_disassembled_obj(C, assembly, perturb_start_pose=cfg.perturb_start_pose,
                           spacing=cfg.scatter_spacing, random_init=cfg.random_init,
                           float_height=cfg.part_float, min_sep=cfg.min_sep,
                           grid=cfg.grid_init, grid_gap=cfg.grid_gap,
                           center_scatter=cfg.center_scatter)
  if cfg.vacuum_no_assemble:
    # Role restrictions describe the physical tool roster, not whether the wrench LP is on.
    # Install that roster only after the actual scene exists; successor calls deliberately
    # keep gripper_kind=None in a --no-wrench ablation so all statics/capacity checks stay off.
    from problem import set_kind_forbid
    _role_robots = get_robots(C)
    set_kind_forbid(("vacuum:assemble",), robot_gripper_kinds(C, _role_robots))

  if cfg.grasp_sampler_scenes:
    from problem import install_sampled_scene_ctx
    install_sampled_scene_ctx(C, assembly, robots=get_robots(C))
    print("[flag] --grasp-sampler-scenes ACTIVE: every candidate is tagged PER GRIPPER KIND "
          "for PICK, the mandatory MATE endpoint, and intermediate/final rigid-component "
          "SUPPORT contexts; the bounded DB reserves diverse proposals for those roles. "
          "Exact state cheap/IK gates remain authoritative. Grasp ids are per PART INSTANCE.")
    if _closure_active:
      print("[grasp-sampler-scenes] strict closure generation ACTIVE: a two-finger source "
            "with unequal jaw contact is recentered from measured first-contact travel; a "
            "finger/table overlap may emit a minimum approach-retracted candidate. Every "
            "generated pose is retained only after open-hand + full closure re-certification. "
            "Clutter/other-hand collisions are never projected.")
  if cfg.grasp_physics_check:
    from problem import install_grasp_physics_ctx
    install_grasp_physics_ctx(C, assembly, robots=get_robots(C))
    print("[flag] --grasp-physics-check ACTIVE: every sampled two-finger candidate is "
          "closed in an isolated MuJoCo model using the exported gripper's actual finger "
          f"joints; bilateral acquisition plus gravity and six {cfg.grasp_physics_force:g}N "
          "tool-frame load directions are required. Results are "
          "cached per pose. The configured wrench profile remains one global envelope for "
          "all accepted grasps; no empirical per-grasp limits are fitted.")
    _physics_source = (cfg.grasp_physics_cache_plan or
                       (cfg.load_plan if not cfg.refresh_witnesses else None))
    if _physics_source:
      from plan_io import read_grasp_physics
      from problem import _GRASP_PHYSICS_CTX
      _saved_handover_physics = read_grasp_physics(_physics_source)
      if _saved_handover_physics is not None:
        _GRASP_PHYSICS_CTX.import_handovers(_saved_handover_physics)
        print(f"[grasp-physics/handover] restored a-priori certificate from "
              f"{_physics_source}; finite database signature checked on use")
      elif cfg.grasp_physics_cache_plan:
        raise ValueError(
            f"--grasp-physics-cache-plan {_physics_source} has no saved certificate")
      else:
        print("[grasp-physics/handover] legacy plan has no persisted certificate; one "
              "a-priori table will be built now and saved on the next plan export")
        # Upgrade before witness replay/search/postprocessing begins.  In particular, do not
        # let the first handover successor query trigger simulation as a hidden search cost.
        from solver import _search_context
        _search_context(C, assembly, get_robots(C))
  # A fixed saved plan carries its exact selected grasps and state chain. The contact-model
  # contract below still checks the live scene, but there is no reason to regenerate all
  # unused candidates merely to interpret positional grasp ids during motion export.
  if cfg.load_plan and cfg.motion_plan and not cfg.refresh_witnesses:
    from plan_io import read_motion_data, read_plan as _read_motion_plan
    _md_cfg, _md_plan = _read_motion_plan(cfg.load_plan)
    if (_witnesses_baseline_compatible(_md_cfg, cfg) and
        (cfg.optimize_witnesses is None or _witnesses_reusable(_md_cfg, cfg))):
      cfg._loaded_motion_data = read_motion_data(
          cfg.load_plan, _md_plan, grasp_db_hash=cfg.grasp_sampler_db_hash)
  if cfg.grasp_sampler:
    _grasp_profile_on = _os.environ.get("ASSEMBLY_GRASP_PROFILE", "").lower() in (
        "1", "true", "yes")
    _grasp_signature_t0 = _time.perf_counter() if _grasp_profile_on else None
    if cfg._loaded_motion_data is not None:
      _db_hash = cfg.grasp_sampler_db_hash
      print("[grasp-db] fixed-plan selected grasps loaded; full candidate DB rebuild SKIPPED")
    else:
      from problem import sampled_grasp_db_signature
      _db_hash = sampled_grasp_db_signature(assembly)
    if _grasp_profile_on:
      print(f"[grasp-prof] signature/master access="
            f"{_time.perf_counter() - _grasp_signature_t0:.2f}s")
    if cfg.grasp_sampler_db_hash is not None and cfg.grasp_sampler_db_hash != _db_hash:
      if not cfg.load_plan:
        # A saved plan is also a convenient, complete scene/configuration file for
        # a fresh solve.  Its old DB hash identifies the plan's grasp indices; it
        # must not prevent regenerating those indices when no plan is loaded.
        print("[grasp-db-regenerate] sampled grasp database differs from the config "
              f"({cfg.grasp_sampler_db_hash} != {_db_hash}); fresh --solve records "
              "the current database identity.", flush=True)
      elif cfg.refresh_witnesses:
        print("[grasp-db-refresh] sampled grasp database differs from the saved plan "
              f"({cfg.grasp_sampler_db_hash} != {_db_hash}). Explicit --refresh-witnesses "
              "treats its actions as a symbolic skeleton: every successor and every IK "
              "witness will be recertified under today's database; persisted witnesses are "
              "not reused.", flush=True)
      else:
        raise SystemExit(
            "sampled grasp database hash differs from the saved plan "
            f"({cfg.grasp_sampler_db_hash} != {_db_hash}); regenerate the plan, or explicitly "
            "pass --refresh-witnesses to recertify its "
            "symbolic skeleton -- refusing to silently rebind grasp ids")
    cfg.grasp_sampler_db_hash = _db_hash

  # Bind every newly saved task plan to the concrete contact model, grasp database and
  # virtual-close semantics that gave its grasp ids meaning. A fresh --solve may use an old
  # plan as a convenient config and deliberately upgrades the contract; --load-plan is a
  # certificate consumer and must either match exactly or explicitly recertify.
  from artifact_contract import build_artifact_contract, validate_artifact_contract
  _saved_contract = cfg.artifact_contract
  _actual_contract = build_artifact_contract(
      C, grasp_db_hash=cfg.grasp_sampler_db_hash,
      closure_enabled=bool(cfg.grasp_closure_check))
  if cfg.load_plan:
    if cfg.refresh_witnesses:
      if _saved_contract != _actual_contract:
        print("[artifact-contract] explicit --refresh-witnesses: recertifying the fixed "
              "symbolic plan under the current collision/grasp/closure contract.")
    else:
      validate_artifact_contract(
          _saved_contract, _actual_contract, who=f"--load-plan {cfg.load_plan}")
  elif _saved_contract is not None and _saved_contract != _actual_contract:
    print("[artifact-contract] fresh --solve replaces the config's stale model contract.")
  cfg.artifact_contract = _actual_contract
  if cfg._loaded_motion_data is not None:
    from plan_io import motion_data_grasps
    set_frozen_plan_grasps(motion_data_grasps(cfg._loaded_motion_data))

  # Internal benchmark pre-pass: construct the finite, scene/gripper-filtered grasp database
  # once, before any solver is timed.  The public runner supplies both an explicit artifact
  # path and a source/config fingerprint; _search_context refuses stale artifacts.  Keeping
  # this out of argparse prevents a benchmark lifecycle control from becoming a planner knob.
  if _os.environ.get("ASSEMBLY_GRASP_DB_BUILD_ONLY", "").strip().lower() in (
      "1", "true", "yes"):
    if not _os.environ.get("ASSEMBLY_GRASP_DB_PATH"):
      raise SystemExit("ASSEMBLY_GRASP_DB_BUILD_ONLY requires ASSEMBLY_GRASP_DB_PATH")
    from solver import _search_context
    _grasp_db_t0 = _time.perf_counter()
    _checker, _table = _search_context(C, assembly, get_robots(C))
    _grasp_db_s = _time.perf_counter() - _grasp_db_t0
    _rows = sum(len(rows) for rows in _table.values())
    if cfg.grasp_sampler_scenes:
      _sampled_scene_coverage(
          assembly, _checker, get_robots(C), prefix="[grasp-db-coverage]")
    print(f"[grasp-db] sampled signature: {cfg.grasp_sampler_db_hash}")
    print(f"[grasp-db] build-only complete: {len(_table)} component rows / {_rows} grasps "
          f"in {_grasp_db_s:.6f}s")
    if _os.environ.get("ASSEMBLY_GRASP_RAW_CACHE_DIR"):
      from problem import sampled_grasp_raw_cache_stats
      _raw = sampled_grasp_raw_cache_stats()
      print("[grasp-raw-cache] "
            f"memory_hits={_raw['memory_hits']} disk_hits={_raw['disk_hits']} "
            f"misses={_raw['misses']} load={_raw['load_s']:.3f}s "
            f"build={_raw['build_s']:.3f}s write={_raw['write_s']:.3f}s")
    return

  if cfg.check_init:
    chk_robots = get_robots(C)
    chk_ik = RobotIK(C, chk_robots) if cfg.real_ik else None
    check_initial_scene(C, assembly, robots=chk_robots, robot_ik=chk_ik)


  if cfg.show_init:
    C.view(True, "initial state (close to continue)")

  # per-skill IK test popups run on THE ACTUAL CONFIGURED SCENE built above (robot type,
  # scatter/--random-init layout, assembly). Note the popups solve real-arm IK: with
  # --floating-ee the "arm" is the free-floating gripper itself.
  if cfg.test_pick_ik is not None:
    robot, part, grasp = cfg.test_pick_ik.split(":")
    visualize_pick_ik(part=part, grasp_id=int(grasp), robot=robot, C=C, assembly=assembly,
                      view=cfg.view)
    return

  if cfg.test_assemble_ik is not None:
    robot, part, grasp = cfg.test_assemble_ik.split(":")
    visualize_assemble_ik(part=part, grasp_id=int(grasp), robot=robot, C=C,
                          assembly=assembly)
    return

  if cfg.test_place_ik is not None:
    fields = cfg.test_place_ik.split(":")
    robot, part, grasp, n_asm = fields[:4]
    face = int(fields[4]) if len(fields) > 4 else 5
    visualize_place_ik(part=part, grasp_id=int(grasp), robot=robot,
                       n_assembled=int(n_asm), face=face, C=C, assembly=assembly)
    return

  if cfg.test_assemble_held_ik is not None:
    robot, part, grasp, target_robot, onto_grasp = cfg.test_assemble_held_ik.split(":")
    visualize_assemble_ik(part=part, grasp_id=int(grasp), robot=robot,
                          target_robot=target_robot, onto_grasp=int(onto_grasp),
                          C=C, assembly=assembly)
    return

  if cfg.disp_final:
    display_final_assembly(C, assembly, [0, 0, 0.5])

  if cfg.viz_initial_grasps is not None:
    chosen = None if cfg.viz_initial_grasps == "all" else cfg.viz_initial_grasps
    if chosen is not None and chosen not in assembly.objects:
      raise SystemExit(f"--viz-initial-grasps: unknown part {chosen!r}; choose from "
                       f"{list(assembly.objects)}")
    from solver import _search_context
    checker, grasp_table = _search_context(C, assembly, get_robots(C))
    rec = None
    if cfg.viser:
      from viser_view import ViserRecorder
      rec = ViserRecorder()
    visualize_initial_grasps_viser(
        C, assembly, checker, grasp_table, get_robots(C), recorder=rec,
        view=cfg.view, part=chosen)
    if rec is not None:
      rec.serve(port=cfg.viser_port)
    elif not cfg.view:
      print("[initial-grasp-viz] add --viser (recommended) or --view to display candidates")
    return

  if cfg.viz_grasp_closure is not None:
    try:
      robot, part, gid_s = cfg.viz_grasp_closure.split(":")
      gid = int(gid_s)
    except Exception:
      raise SystemExit("--viz-grasp-closure expects ROBOT:PART:GRASP, e.g. a1:obj_3:0")
    robots = get_robots(C)
    if robot not in robots:
      raise SystemExit(f"--viz-grasp-closure: unknown robot {robot!r}; choose from {robots}")
    if part not in assembly.objects:
      raise SystemExit(f"--viz-grasp-closure: unknown part {part!r}; choose from "
                       f"{list(assembly.objects)}")
    # Build exactly the floating-EE scene used by the cheap action gate. The pose-valued
    # sampler and id-valued search paths both reduce to this same receiver/part geometry.
    from solver import _search_context
    from problem import get_initial_state
    from gripper_closure import closure_sweep_configs
    checker, _table = _search_context(C, assembly, robots)
    s0 = get_initial_state(C, assembly)
    try:
      Cc, owners = checker._scene_for(
          s0, assembly, extra_grasps=[(robot, part, gid)], return_assignments=True)
    except Exception as exc:
      raise SystemExit(f"--viz-grasp-closure: cannot build {robot}:{part}:{gid}: {exc}")
    if robot not in owners:
      raise SystemExit(f"--viz-grasp-closure: no floating receiver slot for {robot}")
    prefix, _held_part = owners[robot]
    result, frames = closure_sweep_configs(Cc, prefix, part, steps=21)
    print(f"[closure-viz] {robot}:{part}:g{gid}: feasible={result.feasible} "
          f"model={result.model} aperture={result.aperture} reason={result.reason} "
          f"detail={result.detail}")
    if cfg.viser:
      from viser_view import ViserRecorder
      rec = ViserRecorder()
      for Ci, label in frames:
        rec.record(Ci, label)
      print(f"[closure-viz] serving {len(frames)} sweep samples on "
            f"http://localhost:{cfg.viser_port}")
      rec.serve(port=cfg.viser_port)
    elif cfg.view:
      for Ci, label in frames:
        Ci.view(True, label + " (close for next sample)")
    else:
      print("[closure-viz] add --viser (recommended) or --view to display the sweep")
    return

  if cfg.viz_grasps or cfg.viz_rest_poses or cfg.viz_rest_poses_grid:
    rec = None
    if cfg.viser:
      from viser_view import ViserRecorder
      rec = ViserRecorder()
    if cfg.viz_grasps:
      U = None
      if cfg.wrench:
        from solver import _search_context     # local import ONLY of names not module-global
        _chk, _gt = _search_context(C, assembly, get_robots(C))  # -- shadowing wrench_kind_arg
        U = compute_useful_grasps(assembly, _gt, wrench_kind_arg(C, get_robots(C)))  # here made
        #                                                        # it function-local everywhere
      visualize_grasps_viser(C, assembly, recorder=rec, view=cfg.view, useful=U)
    if cfg.viz_rest_poses:
      visualize_rest_poses_viser(C, assembly, recorder=rec, view=cfg.view)
    if cfg.viz_rest_poses_grid:
      visualize_rest_poses_grid_viser(C, assembly, recorder=rec, view=cfg.view)
    if rec is not None:
      rec.serve(port=cfg.viser_port)
    return

  if cfg.mate_sequence_viz:
    # schematic DAG mate-sequence: interface points + per-mate insertion wrench, NO search.
    rec = None
    if cfg.viser:
      from viser_view import ViserRecorder
      rec = ViserRecorder()
    visualize_mate_sequence(C, assembly, recorder=rec, view=cfg.view)
    if rec is not None:
      rec.serve(port=cfg.viser_port)
    return

  if cfg.show_successors:
    visualize_successor_states(C, assembly)
    return

  # A RUN THAT SILENTLY IGNORES WHAT YOU ASKED FOR MUST SAY SO. Every flag below CONSUMES a
  # plan, so without --solve or --load-plan it is dead: run() used to walk to the end and return
  # None in 0.5s, indistinguishable from a crash or a very fast solve (cost me ~20min). Checking
  # "did you ask for a plan consumer" rather than "did you ask for anything" keeps the legitimate
  # viewer/diagnostic modes (and `view` defaults True, so the broad check was vacuous).
  # (optimize_witnesses/fuse_cograsp/hoist_releases/fuse_pre_skill left this list on
  # 2026-07-29: they are DEFAULTS now, so "present without a plan" is no longer a request
  # being silently dropped -- only explicitly-asked-for outputs still warrant the error.)
  _NEEDS_PLAN = ("motion_plan", "save", "mp_save", "mp_export_save", "mp_export_load",
                 "utilization")
  if not (cfg.solve or cfg.load_plan):
    _asked = [f"--{a.replace('_', '-')}" for a in _NEEDS_PLAN if getattr(cfg, a, None)]
    if _asked:
      raise SystemExit(
          f"main.py: {', '.join(_asked)} need a plan, but neither --solve nor --load-plan was "
          f"given -- nothing would happen.\n"
          f"  --solve            run the search\n"
          f"  --load-plan FILE   re-run the post-passes/export on a saved plan (no search)")
  if (cfg.mp_export_save or cfg.mp_export_load) and not cfg.motion_plan:
    raise SystemExit("main.py: --mp-export-save/--mp-export-load require --motion-plan")
  if cfg.solve or cfg.load_plan:
    robots = get_robots(C)
    # The retired task-conditioned sampler used to construct a RobotIK instance while
    # ranking grasp candidates and reuse it here.  Scene-reserved sampling is deliberately
    # floating-EE-only, so search now owns the single real-arm IK object unconditionally.
    robot_ik = RobotIK(C, robots) if cfg.real_ik else None
    if (robot_ik is not None and not np.allclose(
        robot_ik.home, robot_ik.exit_configuration, atol=1e-12, rtol=0.0)):
      print("[robot] distinct HOME/EXIT poses ACTIVE: the physical trajectory starts/ends "
            "at HOME; inactive arms park at EXIT during manipulation, and motion planning "
            "connects HOME directly to each arm's first useful task.")
    from solver import set_fill_reference_diagnostic
    set_fill_reference_diagnostic()                    # reset across repeated library runs
    if cfg.diagnose_plan_rank is not None:
      if not cfg.solve or cfg.load_plan or not cfg.milestones:
        raise SystemExit("--diagnose-plan-rank requires --solve with the milestone solver "
                         "(and cannot be combined with --load-plan)")
      from plan_io import read_plan as _read_rank_plan
      _rank_cfg, _rank_actions = _read_rank_plan(cfg.diagnose_plan_rank)
      set_fill_reference_diagnostic(_rank_actions, cfg.diagnose_plan_rank)
      print(f"[fill-rank] ACTIVE: {len(_rank_actions)} demonstrated actions loaded "
            f"from {cfg.diagnose_plan_rank}; diagnostic queues run without IK and do not "
            "alter the ordinary search.")
    if robot_ik is not None:
      robot_ik.seed_root = int(cfg.seed if cfg.ik_seed is None else cfg.ik_seed)
      print(f"[ik-seed] deterministic physical-query streams use root "
            f"{robot_ik.seed_root} ({'--seed' if cfg.ik_seed is None else '--ik-seed'}); "
            "open-witness effort tiers receive disjoint derived streams.")
      robot_ik.closure_check = bool(cfg.grasp_closure_check)
      robot_ik.restarts_full = cfg.ik_restarts
      robot_ik.restarts_precheck = cfg.precheck_restarts
      robot_ik.fail_order = cfg.fail_order
      robot_ik.positive_ik_reuse = bool(cfg.ik_positive_reuse)
      robot_ik.analytic_seed_cap = cfg.analytic_seed_cap
      robot_ik.assemble_ssik_pose_seeds = bool(cfg.assemble_ssik_pose_seeds)
      robot_ik.assemble_ssik_pose_projection = bool(cfg.assemble_ssik_pose_projection)
      robot_ik.assemble_unbiased_first = bool(cfg.assemble_unbiased_first)
      robot_ik.assemble_basin_racing = bool(cfg.assemble_basin_racing)
      robot_ik.assemble_race_probe_evals = int(cfg.assemble_race_probe_evals)
      robot_ik.assemble_race_promote = int(cfg.assemble_race_promote)
      robot_ik.handover_basin_racing = bool(cfg.handover_basin_racing)
      robot_ik.handover_race_probe_evals = int(cfg.handover_race_probe_evals)
      robot_ik.handover_race_starts = int(cfg.handover_race_starts)
      robot_ik.handover_race_promote = int(cfg.handover_race_promote)
      robot_ik.ssik_seed_neighborhood = float(cfg.ssik_seed_neighborhood)
      robot_ik.ssik_neighborhood_cap = int(cfg.ssik_neighborhood_cap)
      robot_ik.empty_seed_refutes = cfg.empty_seed_refutes
      if cfg.assemble_ssik_pose_seeds:
        print("[flag] --assemble-ssik-pose-seeds ACTIVE: held-target canonical assembly "
              "poses are projected through every participating arm's SSIK and complete "
              "collision-ranked configurations seed the matching KOMO biases; finite pose "
              "misses remain non-rejecting.")
      if cfg.assemble_ssik_pose_projection:
        print("[flag] --assemble-ssik-pose-projection ACTIVE: each canonical held-target "
              "orientation searches a five-point line from the existing preferred centre "
              "toward the currently realized target pose; the closest jointly SSIK-reachable "
              "pose seeds KOMO and all finite misses remain non-rejecting.")
      if cfg.assemble_unbiased_first:
        print("[flag] --assemble-unbiased-first ACTIVE: held-target assembly tries the "
              "neutral continuous KOMO before the same six canonical soft-orientation "
              "basins; failures still widen through every basin and fresh random effort.")
      if cfg.assemble_basin_racing:
        print(f"[flag] --assemble-basin-racing ACTIVE: neutral continuous KOMO first; "
              f"on failure all canonical basins receive {cfg.assemble_race_probe_evals} "
              f"collision-aware NLP evaluations, then {cfg.assemble_race_promote} best "
              "unpromoted basin(s) are fully continued per visit. Finite misses remain "
              "pending and widening eventually covers every basin.")
      if cfg.handover_basin_racing:
        print(f"[flag] --handover-basin-racing ACTIVE: preserve the ordinary full handover "
              f"solve; on failure {cfg.handover_race_starts} fresh random starts receive "
              f"{cfg.handover_race_probe_evals} collision-aware NLP evaluations, then up to "
              f"{cfg.handover_race_promote} best unpromoted start(s) are fully continued per "
              "visit. Partial starts are cached by physical query; finite misses remain open.")
      if cfg.ik_positive_reuse:
        print("[flag] --ik-positive-reuse ACTIVE: identical skill-local physical queries "
              "(table scene or participating payload/grasps, as appropriate) share their "
              "exact witness and open-witness effort cursor; failures remain open.")
      if cfg.ssik_seed_neighborhood < 0.0:
        raise SystemExit("main.py: --ssik-seed-neighborhood must be >= 0")
      if cfg.ssik_neighborhood_cap < 1:
        raise SystemExit("main.py: --ssik-neighborhood-cap must be >= 1")
      if cfg.ssik_seed_neighborhood > 0.0:
        print(f"[flag] --ssik-seed-neighborhood {cfg.ssik_seed_neighborhood:g}m ACTIVE: "
              f"up to {cfg.ssik_neighborhood_cap} deterministic near-target SSIK branches "
              "warm-start boundary/empty-exact KOMO binds; hints only, never rejection.")
      robot_ik.open_witness = cfg.open_witness
      robot_ik.open_witness_epochs = cfg.open_witness_epochs
      robot_ik.open_witness_batch = cfg.open_witness_batch
      robot_ik.open_witness_positive_order = bool(cfg.open_witness_positive_order)
      robot_ik.lazy_edge_selector = cfg.lazy_edge_selector
      if cfg.open_witness:
        if cfg.open_witness_batch < 1:
          raise SystemExit("main.py: --open-witness-batch must be >= 1")
        print(f"[flag] --open-witness ACTIVE: failed lazy-fill IK probes stay open; "
              f"ordinary effort escalates {min(cfg.open_witness_batch, 2)} -> "
              f"{min(cfg.open_witness_batch, 4)} -> {cfg.open_witness_batch}, assemble uses "
              f"an atomic {cfg.open_witness_batch}-slot structural batch; up to "
              f"{cfg.open_witness_epochs or 'unlimited'} epoch(s). Candidate IK selector "
              f"is {cfg.lazy_edge_selector}; its first failed edge abandons the incumbent "
              "immediately. "
              "Legacy bind and fill paths remain unchanged when this flag is off.")
        if cfg.open_witness_positive_order:
          print("[flag] --open-witness-positive-order ACTIVE: exact certified edges, "
                "compatible co-grasps, and robot-reachable grasps order equal-cost "
                "successors. Failures do not score or prune; unseen ties retain the original "
                "order.")
    if robot_ik is not None and cfg.full_ik:
      robot_ik.single_arm_ik = False
      robot_ik.strip_inactive_collision = False
      robot_ik.reach_precheck = False
    if robot_ik is not None and cfg.ik_seeding:
      robot_ik.ik_seeding = True
      robot_ik.collision_reject_depth = (
          None if cfg.ik_collision_gate < 0 else cfg.ik_collision_gate)
    if robot_ik is not None and cfg.witness_robot_collisions:
      # keep the OTHER robots' collision ON in the SEARCH's IK too (not just the post-opt pass):
      # the search's witnesses then avoid the idle arms from the start. (optimize_witnesses gets
      # robot_collisions too; with this already set, that toggle is a consistent no-op.)
      robot_ik.strip_inactive_collision = False
      print("[flag] --witness-robot-collisions ACTIVE: other robots' collision kept ON in IK "
            "(search + witness opt); idle arms are at EXIT in the IK scene.")
    if cfg.witness_margin is not None and robot_ik is not None:
      # ONE clearance contract, installed BEFORE search. The old implementation enabled this
      # only after a discrete plan existed, so witness optimization could reject every IK the
      # search itself had accepted -- exactly the contract violation this flag is meant to
      # prevent. Search, persisted witnesses, fusion and pre/post-skill solves now all call
      # RobotIK._clearance_ok with the same value and pair classification.
      robot_ik.clearance_reject = float(cfg.witness_margin)
      print(f"[flag] --witness-margin {cfg.witness_margin*1000:.0f}mm ACTIVE IN SEARCH + "
            "WITNESS PASSES: every accepted IK clears the same hard non-intended-pair gate; "
            "KOMO receives lazy pair-specific margin cuts whenever its nominal optimum "
            "fails that certificate "
            "(arm/part, inter-robot, non-adjacent self-collision, and free-pose "
            "payload/table); coupled task hands/mate-target groups use strict collision "
            "without this free-space buffer. Later passes may rank or diversify but cannot "
            "strengthen search feasibility.")
    if cfg.witness_joint_margin is not None and robot_ik is not None:
      robot_ik.joint_margin_reject = float(cfg.witness_joint_margin)
      if not any(v is not None for v in robot_ik.robot_arm.values()):
        print(f"[flag] --witness-joint-margin {cfg.witness_joint_margin:g}rad requested but "
              "NOT APPLIED: --floating-ee coordinates are pose parameters, not hardware "
              "joints.")
      else:
        print(f"[flag] --witness-joint-margin {cfg.witness_joint_margin:g}rad ACTIVE IN SEARCH "
              "+ WITNESS PASSES for coupled handover/add_grasp/assemble endpoints: IK keyframes "
              "on/near a hardware stop are rejected before they become zero-neighbourhood "
              "motion-planning goals; ordinary picks/places are unchanged.")
    if not np.isfinite(cfg.object_over_table_margin):
      raise SystemExit("main.py: --object-over-table-margin must be finite")
    if cfg.object_over_table and robot_ik is not None:
      robot_ik.object_over_table = True
      robot_ik.object_over_table_margin = float(cfg.object_over_table_margin)
      # Fail before a long search if a custom scene has no usable tabletop geometry.
      robot_ik._table_workcell_half_extents(robot_ik.C)
      margin_desc = (f" inset by {cfg.object_over_table_margin * 1000:g}mm"
                     if cfg.object_over_table_margin > 0.0 else
                     f" expanded by {-cfg.object_over_table_margin * 1000:g}mm"
                     if cfg.object_over_table_margin < 0.0 else "")
      print("[flag] --object-over-table ACTIVE IN SEARCH + WITNESS PASSES: every held "
            f"component moved by the current skill stays inside the table-local x/y footprint"
            f"{margin_desc}; KOMO receives "
            "the bounds up front and the accepted endpoint is checked again. This does not "
            "constrain inactive EXIT-pose payloads, arm links, or the continuous "
            "motion-planning path.")
    if robot_ik is not None and cfg.mate_eq_weight is not None:
      robot_ik.mate_eq_weight = float(cfg.mate_eq_weight)
      print(f"[flag] --mate-eq-weight {cfg.mate_eq_weight:g}: assemble mate eq constraints "
            f"weighted up from the 1e1 default, so the solve SEATS the part instead of "
            f"stopping millimetres short.")
    if robot_ik is not None and cfg.place_eq_weight is not None:
      robot_ik.place_eq_weight = float(cfg.place_eq_weight)
      print(f"[flag] --place-eq-weight {cfg.place_eq_weight:g}: PLACE object-pose eq "
            f"constraints weighted up from the 1e1 default. A place is the DATUM later mates "
            f"are measured against, so its residual is not local.")
    if robot_ik is not None and getattr(cfg, "grasp_eq_weight", None) is not None:
      robot_ik.grasp_eq_weight = float(cfg.grasp_eq_weight)
      print(f"[flag] --grasp-eq-weight {cfg.grasp_eq_weight:g}: pick/add_grasp tool-pose eq "
            f"weighted up from the 1e1 default, so the solve LANDS ON the grasp-DB pose.")
    if robot_ik is not None and getattr(cfg, "grasp_accuracy", None) is not None:
      robot_ik.grasp_pose_reject = float(cfg.grasp_accuracy)
      print(f"[flag] --grasp-accuracy {cfg.grasp_accuracy:g}m: pick/add_grasp witnesses "
            f"REJECTED when the achieved tool pose misses the grasp-DB pose by more "
            f"(the attach freezes the ACHIEVED relation).")
    if robot_ik is not None and cfg.mate_accuracy is not None:
      robot_ik.mate_residual_reject = float(cfg.mate_accuracy)
      print(f"[flag] --mate-accuracy {cfg.mate_accuracy:g}m: assemble witnesses REJECTED when "
            f"the achieved mate misses by more than that (rai's own feasibility verdict does "
            f"not enforce it).")
    if robot_ik is not None and cfg.mate_angular_accuracy is not None:
      robot_ik.mate_angular_residual_reject = float(cfg.mate_angular_accuracy)
      print(f"[flag] --mate-angular-accuracy {cfg.mate_angular_accuracy:g}deg: assemble "
            f"witnesses REJECTED when quaternionRel misses the authored mate by more than "
            f"that.")
    # RESIDUAL BUDGET. Both witness gates bound an error that ends up as PENETRATION in the
    # exported problem, and motion planning rejects a goal that penetrates more than its own
    # tolerance -- as "no path" at that mode, with nothing pointing at the cause.
    #   * a mate that stops d short overlaps its parent by ~d (flush parts touch at 0);
    #   * a grasp that misses the DB pose by d is FROZEN at the achieved relation by the
    #     attach, while later witnesses assume the DB relation, so d lands in the next seat.
    # Measured on compound-demo: grasp residuals 89-236um under a 1000um gate produced a
    # 110um seat penetration against a 100um tolerance -> no path. The knobs are a BUDGET.
    _mp_tol = float(getattr(cfg, "mp_collision_tolerance", 0.0) or 0.0)
    for _name, _val in (("--mate-accuracy", getattr(cfg, "mate_accuracy", None)),
                        ("--grasp-accuracy", getattr(cfg, "grasp_accuracy", None))):
      if _val is not None and _mp_tol > 0.0 and float(_val) >= _mp_tol:
        print(f"[flag] !!! {_name} {float(_val):g} is NOT tighter than "
              f"--mp-collision-tolerance {_mp_tol:g}: an accepted witness may penetrate more "
              f"than motion planning allows, which surfaces as 'no path' at that mode. Use "
              f"{_name} < {_mp_tol:g}.")
    if robot_ik is not None and cfg.ik_accuracy is not None:
      robot_ik.nlp_stop_tolerance = float(cfg.ik_accuracy)
      print(f"[flag] --ik-accuracy {cfg.ik_accuracy:g}: NLP stopTolerance tightened from rai's "
            f"default 0.01 (=10mm, which lets a mate settle millimetres short of its target).")
    if robot_ik is not None and cfg.strict_mate_collisions:
      robot_ik.mate_pair_gate = True
      print("[flag] --strict-mate-collisions ACTIVE: at an assemble the moving group's contact "
            "is disabled for the SOLVE (the mate is flush by construction) but the witness is "
            "GATED on every pair except the intended moving<->target one. Without it the strict "
            "gate is blind to part-vs-arm / part-vs-part at the mate -- which is where the "
            "motion planner disagrees with the search at the SAME tolerance.")
    if robot_ik is not None and cfg.strict_collisions is not None:
      robot_ik.penetration_reject = float(cfg.strict_collisions)
      robot_ik.col_weight = 1e2                          # steer clean, don't lean on the slack
      print(f"[flag] --strict-collisions ACTIVE: KOMO witnesses rejected above "
            f"{cfg.strict_collisions:g}m total penetration (collision weight 1e1 -> 1e2)."
            + ("" if cfg.witness_robot_collisions else
               " NOTE: idle arms are STRIPPED from skill scenes (add "
               "--witness-robot-collisions to gate arm-arm penetration too)."))
    if robot_ik is not None and cfg.always_contact:
      robot_ik.keep_contacts = True
      print("[flag] --always-contact ACTIVE (default): no skill IK disables a touched part's "
            "contact -- every witness must clear the strict-collisions gate with ALL contacts "
            "on (grasp and mate contact included); overlapping plans fail their IK and are "
            "re-searched. --no-always-contact restores the contact exclusions.")
    if robot_ik is not None and cfg.pre_insertion > 0.0:
      robot_ik.pre_insertion_delta = cfg.pre_insertion
      print(f"[flag] --pre-insertion ACTIVE: MP export splits each assemble into a goto to the "
            f"funnel mouth ({cfg.pre_insertion:g}m back along the mate axis) + the real mate at "
            "the seated witness. This is a hard boundary: the full configured separation must "
            "be feasible; it is never shortened toward the seated pose. With "
            "--skill-funnel-collision-res, search also certifies the short held-gripper edge "
            "between them.")
    if robot_ik is not None and cfg.skill_funnel_collision_res is not None:
      if cfg.skill_funnel_collision_res < 0.0:
        raise SystemExit("--skill-funnel-collision-res must be nonnegative")
      robot_ik.skill_funnel_collision_res = float(cfg.skill_funnel_collision_res)
      if cfg.skill_funnel_collision_res > 0.0:
        print(f"[flag] --skill-funnel-collision-res {cfg.skill_funnel_collision_res:g}rad "
              "ACTIVE: assemble IK is accepted only when its exact held-gripper "
              "pre-insertion-to-seat edge passes the strict collision gate.")
    if robot_ik is not None and cfg.pre_pick > 0.0:
      robot_ik.pre_pick_delta = cfg.pre_pick
      print(f"[flag] --pre-pick ACTIVE: MP export splits each pick into a goto to the pre-grasp "
            f"({cfg.pre_pick:g}m gripper retract) + the real grasp. Search/plan unchanged.")
    if robot_ik is not None and cfg.post_pick > 0.0:
      if cfg.pre_pick <= 0.0:
        if explicit and "post_pick" in explicit:
          raise SystemExit("--post-pick currently requires --pre-pick so export emits the "
                           "complete approach/contact/departure task split")
        print("[flag] --post-pick disabled because --pre-pick is 0 (no certified standoff to "
              "complete the pick funnel).")
        cfg.post_pick = 0.0
    if robot_ik is not None and cfg.post_pick > 0.0:
      robot_ik.post_pick_delta = cfg.post_pick
      print(f"[flag] --post-pick ACTIVE: MP export appends a held-object retreat "
            f"({cfg.post_pick:g}m along +world-z for table picks; otherwise the paired "
            "-approach endpoint) after each grasp. "
            "Search/plan unchanged.")
    if robot_ik is not None and cfg.pre_place > 0.0:
      robot_ik.pre_place_delta = cfg.pre_place
      print(f"[flag] --pre-place ACTIVE: MP export splits each place into a goto to the pre-place "
            f"({cfg.pre_place:g}m lift) + the real set-down. Search/plan unchanged.")
    if robot_ik is not None and cfg.pre_add_grasp > 0.0:
      robot_ik.pre_add_grasp_delta = cfg.pre_add_grasp
      print(f"[flag] --pre-add-grasp ACTIVE: MP export splits each add_grasp/handover into "
            f"a goto to the pre-grasp ({cfg.pre_add_grasp:g}m gripper retract) + the real "
            "grasp/ownership switch. Search/plan unchanged.")
    if robot_ik is not None and cfg.post_release > 0.0:
      robot_ik.post_release_delta = cfg.post_release
      print(f"[flag] --post-release ACTIVE: MP export appends a departure GOTO after each "
            f"detach ({cfg.post_release:g}m gripper retract); its endpoint IK holds every "
            "other arm at the release keyframe, while the MP path interior stays "
            "unconstrained. Search/plan unchanged.")
    if cfg.view_cheap_checks:
      set_cheap_check_debug(True)
    fail_rec = None
    if cfg.log_ik_failures is not None and robot_ik is not None:
      from viser_view import ViserRecorder
      fail_rec = ViserRecorder()
      robot_ik.fail_recorder = fail_rec
      robot_ik.fail_log_seconds = cfg.log_ik_failures
      robot_ik.fail_log_stop = True                # abort the search when the window closes
    _bench_profile = (_os.environ.get("ASSEMBLY_BENCH_PROFILE", "").strip().lower()
                      not in ("", "0", "false", "no"))
    _bench_search_t0 = None
    if _bench_profile and not cfg.load_plan:
      import problem as _bench_problem
      import wrench as _bench_wrench
      _bench_problem.reset_grasp_timing()
      _bench_wrench.reset_wrench_timing()
      _bench_search_t0 = _time.perf_counter()
    if cfg.load_plan:
      # RE-RUN THE POST-PASSES ON A PLAN YOU ALREADY TRUST -- no search.
      # Modern saved plans persist the resolved solver settings as reproducibility provenance,
      # but re-running those settings is still the wrong way to apply a witness pass
      # (--optimize-witnesses, --fuse-cograsp) or re-export a motion path: a finite/anytime
      # search may return a different discrete plan. Loading keeps the certified action list
      # fixed and re-runs only what was requested. The scene still comes from --config/CLI, so
      # it must MATCH the saved plan.
      from plan_io import read_plan as _read_saved_plan
      _saved_cfg, plan = _read_saved_plan(cfg.load_plan)
      # Keep the original search certificate if this plan is written out again.
      cfg._search_witness_margin = float(_saved_cfg.get("_search_witness_margin", 0.0))
      plan = [tuple(a) for a in plan]
      print(f"[load-plan] {len(plan)} actions read from {cfg.load_plan}; search SKIPPED "
            f"(post-passes and export run on the plan as saved).")
      # The hard margin is now a SEARCH feasibility criterion. Strengthening it while search
      # is explicitly skipped would recreate the old contract violation: post-processing could
      # reject a plan under a gate its search never saw. A weaker/equal replay is safe; a
      # stronger one requires a fresh solve so the discrete plan is certified too.
      # Old files have no certificate: their --witness-margin, if present, was installed
      # only after search and therefore certifies 0m here.
      _old_m = float(_saved_cfg.get("_search_witness_margin", 0.0))
      _new_m = 0.0 if cfg.witness_margin is None else float(cfg.witness_margin)
      # A final export contains all keyframes/tasks and --mp-export-load performs no IK at
      # all, so no new gate is applied in that exact-replay path.
      if (not (cfg.motion_plan and cfg.mp_export_load)
          and not cfg.refresh_witnesses and _new_m > _old_m + 1e-12):
        raise ValueError(
            f"--witness-margin {_new_m:g}m is stricter than the {_old_m:g}m gate used by "
            f"the search saved in {cfg.load_plan}. Because --load-plan skips search, applying "
            "it only to witnesses is forbidden. Re-run with --solve under the new margin, or "
            "keep the saved margin for this plan.")
      for _k in ("assembly", "parts", "num_robots", "seed"):
        if _saved_cfg.get(_k) is not None and _saved_cfg.get(_k) != getattr(cfg, _k, None):
          print(f"[load-plan] !!! scene MISMATCH on '{_k}': saved {_saved_cfg.get(_k)!r} vs "
                f"current {getattr(cfg, _k, None)!r} -- witnesses will be solved in the WRONG "
                f"scene. Pass --config {cfg.load_plan} (its config block) to match.")
    else:
      # Necessary-condition preflight on the CURRENT grasp prefix. Without lazy widening this
      # is the complete configured DB and can abort immediately. Under --grasp-widen an empty
      # prefix is only a request for the next prefix; the same sound abort runs again after the
      # finite master pool is exhausted.
      from solver import _search_context, isolated_graspability_report
      _grasp_profile_on = _os.environ.get("ASSEMBLY_GRASP_PROFILE", "").lower() in (
          "1", "true", "yes")
      _pre_t0 = _time.perf_counter() if _grasp_profile_on else None
      _pre_checker, _pre_table = _search_context(C, assembly, robots)
      _table_seconds = _time.perf_counter() - _pre_t0 if _grasp_profile_on else 0.0
      _pre = isolated_graspability_report(
          C, assembly, robots, _pre_checker, _pre_table,
          require_closure=bool(cfg.grasp_closure_check), robot_ik=robot_ik, verbose=True)
      if _grasp_profile_on:
        print(f"[grasp-prof] preflight: search-table={_table_seconds:.2f}s, "
              f"isolated-pick/reach={_time.perf_counter() - _pre_t0 - _table_seconds:.2f}s")
      if cfg.grasp_sampler_scenes:
        # The two marginal counts printed by the sampler are not enough to diagnose a
        # transport funnel: a database can contain many pick-clear rows and many
        # assembly-clear rows while their intersection is empty.  Report the exact finite
        # overlap per end-effector kind.  This is cached metadata only; it performs no IK and
        # has no effect on candidate ordering or feasibility.
        _sampled_scene_coverage(assembly, _pre_checker, robots)
      if hasattr(_pre_table, "materialized"):
        print(f"[component-grasps/preflight] materialized {_pre_table.materialized}/"
              f"{len(_pre_table)} component tables ({_pre_table.materialized_rows} rows).")
      if cfg.grasp_local_widen:
        for _p, _result in _pre.items():
          _accepted = {gid for _kind, _robot, gid in _result["accepted"]}
          _active = sum(gid < int(cfg.grasp_sampler_keep) for gid in _accepted)
          print(f"[grasp-funnel {_p}] isolated-pick active/master="
                f"{_active}/{len(_accepted)} (K={cfg.grasp_sampler_keep}); closure and "
                "open-hand geometry are already included in this count.")
      # The assembly root may remain at its initial table pose while every incoming
      # part is assembled onto it; valid plans routinely do exactly this. Therefore
      # isolated pickability of the root is not a necessary condition. Treating it
      # as one caused pointless full grasp-widening epochs (and could ultimately
      # reject a solvable task) for broad bases such as Fabrica's cooling manifold.
      # Every non-root object must cross an assemble edge and hence must first be
      # acquired, so the necessary-condition abort remains sound for those parts.
      _unpickable = _unpickable_required_parts(assembly, _pre)
      if _unpickable and not cfg.grasp_widen:
        layer = "open-hand collision + virtual closure" if cfg.grasp_closure_check \
            else "open-hand collision"
        raise SystemExit(
            f"[grasp-preflight] ABORT: no isolated feasible pick for {_unpickable} under "
            f"{layer}. Other parts were removed, so search/IK cannot repair this. Inspect "
            f"with --viz-initial-grasps {_unpickable[0]} --viser, increase the grasp "
            "database, or correct the gripper/part geometry.")
      if _unpickable and cfg.grasp_widen:
        print(f"[grasp-preflight] current {cfg.grasp_sampler_keep}-grasp prefix cannot pick "
              f"{_unpickable} in isolation; deferring the sound abort while the finite "
              "sampled-master pool is widened.")
      try:
        _set_search_placement_yaw_prefix(cfg, active=True)
        print(f"[rss] peak RSS before solve: {_rss_mb():.0f} MB")   # A1/A6 instrumentation
        _staged_grasps = bool(cfg.grasp_sampler and cfg.grasp_refine)
        _staged_dispatch = bool(_staged_grasps and not cfg.mp_open_plans)
        _original_bracing = bool(cfg.bracing)
        _original_scope_widening = bool(cfg.fill_scope_widening)
        if _staged_dispatch:
          from problem import set_sampled_canonical_only
          set_sampled_canonical_only(True)
          # Monotone action-space widening. The former nesting did the most expensive thing
          # first: exhaust compact grasps through unrestricted bracing/all-part recovery, and
          # only THEN expose sampled grasps. OpenArm car-mini spent minutes in a symbolic
          # recovery whose target body had no viable compact route. Grow the three independent
          # alphabets in branching-factor order instead; the final epoch is exactly the old
          # complete action space, so no plan is removed.
          cfg.bracing = False
          cfg.fill_scope_widening = False
          print("[grasp-refine] epoch 0: conventional primitive grasps + local unbraced "
                "manipulation. This is the smallest nested action space; failures do not "
                "refute later epochs.")
        if cfg.mp_open_plans:
          print("[flag] --mp-open-plans ACTIVE: complete assembly plans are generated from "
                "one resumable milestone session; finite MP failures suspend a plan, probe "
                "another, then fairly reopen every suspended plan at higher effort. No MP "
                "timeout becomes a symbolic or boundary no-good.")
          return _run_open_plan_loop(cfg, C, assembly, robots, robot_ik, explicit=explicit)
        plan = _dispatch_solve(cfg, C, assembly, robots, robot_ik)
        if plan is None and _staged_dispatch:
          set_sampled_canonical_only(False)
          print("[grasp-refine] epoch 0 exhausted -> epoch 1: full hybrid grasp DB, still "
                "local and unbraced (stable ids/witness cache retained).")
          plan = _dispatch_solve(cfg, C, assembly, robots, robot_ik)
        if plan is None and _staged_dispatch and _original_bracing:
          cfg.bracing = True
          print("[grasp-refine] epoch 1 exhausted -> epoch 2: add bracing, retain full grasps "
                "and local manipulation scope.")
          plan = _dispatch_solve(cfg, C, assembly, robots, robot_ik)
        if plan is None and _staged_dispatch and _original_scope_widening:
          cfg.bracing = _original_bracing
          cfg.fill_scope_widening = True
          print("[grasp-refine] local epochs exhausted -> final completeness epoch: full "
                "grasps + configured bracing + all-part scope widening.")
          plan = _dispatch_solve(cfg, C, assembly, robots, robot_ik)
        if plan is None and cfg.component_grasp_keep:
          from problem import set_component_grasp_keep
          _component_keep = int(cfg.component_grasp_keep)
          set_component_grasp_keep(0)
          # Save/replay must rebuild the alphabet which actually produced the plan. The
          # requested compact value remains in the command and raw benchmark provenance;
          # zero here records that this completion attempt used the historical full table.
          cfg.component_grasp_keep = 0
          print(f"[component-grasps] restricted K={_component_keep} search returned no plan -> "
                "completion retry with the exact full constituent-grasp union. Existing "
                "witness cache entries remain valid because grasp ids never changed.")
          plan = _dispatch_solve(cfg, C, assembly, robots, robot_ik)
        if plan is None and cfg.grasp_widen:
          from problem import (set_sampled_canonical_only, set_sampled_grasp_keep,
                               sampled_grasp_db_signature)
          if not cfg.grasp_sampler:
            raise ValueError("--grasp-widen requires --grasp-sampler")
          if cfg.grasp_widen_batch < 1:
            raise ValueError("--grasp-widen-batch must be >= 1")
          if cfg.grasp_widen_max < cfg.grasp_sampler_keep:
            raise ValueError("--grasp-widen-max must be >= --grasp-sampler-keep")
          set_sampled_canonical_only(False)
          cfg.bracing = _original_bracing
          cfg.fill_scope_widening = _original_scope_widening
          prev_keep = int(cfg.grasp_sampler_keep)
          prev_hash = str(cfg.grasp_sampler_db_hash)
          epoch = 0
          while plan is None and prev_keep < int(cfg.grasp_widen_max):
            requested = min(int(cfg.grasp_widen_max), prev_keep + int(cfg.grasp_widen_batch))
            set_sampled_grasp_keep(requested)
            cfg.grasp_sampler_keep = requested
            new_hash = sampled_grasp_db_signature(assembly)
            if new_hash == prev_hash:
              # The evaluated master pool can be shorter than the user-facing upper bound
              # (e.g. primitive canonical rows + the fixed scene-probe pool). Restore the
              # last effective prefix so saved provenance says exactly what was searched.
              set_sampled_grasp_keep(prev_keep)
              cfg.grasp_sampler_keep = prev_keep
              cfg.grasp_sampler_db_hash = prev_hash
              print(f"[grasp-widen] finite evaluated master pool exhausted at effective "
                    f"prefix {prev_keep}; requested {requested} added no candidate. No "
                    "geometric infeasibility is inferred from the finite search budget.")
              break
            cfg.grasp_sampler_db_hash = new_hash
            epoch += 1
            print(f"[grasp-widen] epoch {epoch}: search returned no plan with prefix "
                  f"{prev_keep}; exposing stable ids [{prev_keep}, {requested}) and retrying "
                  "the same action/IK semantics. Prior ids and successful witnesses remain "
                  "valid; failures are not blacklisted.")
            plan = _dispatch_solve(cfg, C, assembly, robots, robot_ik)
            prev_keep, prev_hash = requested, new_hash
          if plan is None:
            from problem import sampled_grasp_selection_diagnostics
            _coverage = sampled_grasp_selection_diagnostics()
            _unprobed = [p for p, row in _coverage.items()
                         if row["probe"] < row["raw"]]
            _unselected = [p for p, row in _coverage.items()
                           if row["selected"] >= int(cfg.grasp_widen_max)
                           and row["evaluated"] > row["selected"]]
            _next = ("Increase --grasp-widen-max to expose more of the existing raw pool"
                     if (_unprobed or _unselected) else
                     "The generated pool is exhausted; increase --grasp-sampler-candidates "
                     "to sample more contact points")
            print(f"[grasp-widen] no plan within the finite sampled master "
                  f"(active K={cfg.grasp_sampler_keep}, configured max "
                  f"K={cfg.grasp_widen_max}; parts with more raw/selected grasps: "
                  f"{sorted(set(_unprobed + _unselected)) or 'none'}). "
                  "This is not a proof of geometric "
                  f"infeasibility. {_next}.")
            _final_checker, _final_table = _search_context(C, assembly, robots)
            _final_pre = isolated_graspability_report(
                C, assembly, robots, _final_checker, _final_table,
                require_closure=bool(cfg.grasp_closure_check), robot_ik=robot_ik,
                verbose=True)
            _final_unpickable = [p for p, result in _final_pre.items() if not result["ok"]]
            if _final_unpickable:
              raise SystemExit(
                  f"[grasp-preflight] ABORT after grasp-master exhaustion: no isolated "
                  f"feasible pick for {_final_unpickable}. Search, IK effort, and clutter "
                  "cannot repair this intrinsic gripper/object/table failure; inspect with "
                  f"--viz-initial-grasps {_final_unpickable[0]} --viser or correct the "
                  "gripper/part geometry.")
      except IKFailureLogDone:
        print(f"[ik-failures] window/cap reached -- search stopped for debugging "
              f"({len(fail_rec.steps)} failures collected).")
        plan = None
      except MemoryError:
        # Do not turn resource exhaustion into an apparent geometric infeasibility or a
        # silent heuristic fallback.  In local/milestone search the immediately preceding
        # [fill-prof ...] line identifies the skill boundary that caused the expansion.
        print("[search-failure] process memory exhausted during search; no infeasibility "
              "certificate was produced.", flush=True)
        print("[search-failure] inspect the last [fill-prof ...] block. For a bounded "
              "diagnostic rerun use --log-ik-failures 60; if sampled grasps are active, "
              "try --grasp-refine before increasing the node or memory limit.", flush=True)
        raise
      except SystemError as exc:
        # Some native RAI/FCL allocation paths return a null C-API result without installing
        # a Python exception after the address-space guard is reached. CPython exposes that
        # as this otherwise opaque SystemError. Do not label arbitrary SystemErrors as OOM,
        # but make this exact native-allocation signature as explicit as MemoryError above.
        if str(exc) != "error return without exception set":
          raise
        print("[search-failure] native allocation failed under the process memory guard; "
              "no infeasibility certificate was produced.", flush=True)
        print("[search-failure] the live local-fill frontier did not fit. Reduce the sampled "
              "grasp alphabet or use a compact/regenerating frontier before raising the "
              "memory limit.", flush=True)
        raise MemoryError("native search allocation failed under memory guard") from exc
      finally:
        if '_original_bracing' in locals():
          cfg.bracing = _original_bracing
          cfg.fill_scope_widening = _original_scope_widening
        if cfg.grasp_sampler:
          from problem import set_sampled_canonical_only
          set_sampled_canonical_only(False)               # post-passes/replay see full DB
        _set_search_placement_yaw_prefix(cfg, active=False)  # saved plan/post-passes see all ids
    if cfg.grasp_local_widen and hasattr(assembly, "_grasp_local_caps"):
      cfg.grasp_local_caps = dict(assembly._grasp_local_caps)
      used = ", ".join(f"{p}:{k}" for p, k in sorted(cfg.grasp_local_caps.items()))
      print(f"[grasp-columns] final active prefixes: {used} (finite master hash persisted).")
    if robot_ik is not None and cfg.strict_collisions is not None:
      print(f"[strict-collisions] vetoed {robot_ik.strict_rejects} dirty KOMO witnesses "
            f"(> {cfg.strict_collisions:g}m total penetration).")
    if robot_ik is not None and robot_ik.clearance_reject is not None:
      print(f"[ik-clearance] vetoed {robot_ik.margin_rejects} IK candidate(s) below the "
            f"shared {robot_ik.clearance_reject*1000:.1f}mm search/witness margin.")
      rows = sorted(robot_ik.margin_reject_pairs.items(), key=lambda kv: kv[1], reverse=True)
      for (reason, fa, fb), count in rows[:5]:
        print(f"  [ik-clearance] {reason:11s} {fa} ~ {fb}: {count} veto(es)")
    if robot_ik is not None and getattr(robot_ik, "object_over_table", False):
      print(f"[ik-workcell] vetoed {robot_ik.workspace_rejects} IK candidate(s) whose held "
            "part datum left the table footprint.")
    if robot_ik is not None and cfg.mate_accuracy is not None:
      if getattr(cfg, "grasp_accuracy", None) is not None:
        print(f"[grasp-accuracy] vetoed {robot_ik.grasp_rejects} off-pose grasps "
              f"(> {cfg.grasp_accuracy:g}m tool residual).")
      print(f"[mate-accuracy] vetoed {robot_ik.mate_rejects} under-seated mates "
            f"(> {cfg.mate_accuracy:g}m residual).")
    if robot_ik is not None and cfg.mate_angular_accuracy is not None:
      print(f"[mate-angular-accuracy] vetoed {robot_ik.mate_angular_rejects} rotated mates "
            f"(> {cfg.mate_angular_accuracy:g}deg residual).")
    if robot_ik is not None:
      # A1 INSTRUMENTATION: where the real-IK time went + a peak-RSS checkpoint (A6).
      robot_ik.prof_report()
      print(f"[rss] peak RSS after solve attempt: {_rss_mb():.0f} MB")
    if _bench_search_t0 is not None:
      import problem as _bench_problem
      import wrench as _bench_wrench
      _gp = _bench_problem.grasp_timing_snapshot()
      _wp = _bench_wrench.wrench_timing_snapshot()
      print(f"[search-phase-prof] pipeline={_time.perf_counter() - _bench_search_t0:.6f}s "
            f"grasp={_gp['s']:.6f}s/{_gp['calls']} "
            f"grasp_backward={_gp['backward_s']:.6f}s/{_gp['backward_calls']} "
            f"force_lp_backward={_wp['backward_s']:.6f}s/{_wp['backward_calls']} "
            f"force_lp_forward={_wp['forward_s']:.6f}s/{_wp['forward_calls']}")
    if fail_rec is not None and fail_rec.steps:
      print(f"[ik-failures] {len(fail_rec.steps)} failed binds recorded -- serving in "
            f"viser (Stop in the GUI to continue to the plan replay).")
      fail_rec.serve(port=cfg.viser_port)
    if plan is not None:
      print(f"plan ({len(plan)} actions):")
      for step in plan:
        print("  ", step)
      # SEARCH CERTIFICATE PERSISTENCE. The exact witness that made each selected IK edge pass
      # is the baseline for every later stage. Never ask a new process to rediscover it. A
      # fresh plan is collected from the positive search cache/open pool. A release-hoist is a
      # transaction *after* this source certificate exists: it can commit only if every changed
      # predecessor scene is re-certified, otherwise the original plan/witnesses are retained.
      # A loaded plan is read-only unless the user explicitly asks for --refresh-witnesses.
      from plan_io import (read_grasp_closures, read_pre_witnesses, read_skill_paths, read_witnesses,
                           read_witness_stage,
                           require_complete_witnesses, witness_block)
      joint_names = list(robot_ik.C.getJointNames()) if robot_ik is not None else []
      baseline_witnesses = None
      baseline_pre_witnesses = None
      grasp_closures = {}
      skill_paths = None
      _saved_witness_stage = None
      _need_witnesses = bool(robot_ik is not None and not (
          cfg.motion_plan and cfg.mp_export_load))
      if _need_witnesses and cfg.load_plan:
        # --refresh-witnesses is an explicit request to recertify the fixed discrete plan
        # against TODAY'S geometry.  Do not read a complete persisted block first and then
        # retain it merely because it has no missing entries: that silently reused witnesses
        # produced before model fixes (the OpenArm mount used to be contact-disabled), in
        # direct contradiction to the CLI contract that refresh means IGNORE.  With refresh,
        # every IK action is rebuilt below; without it the normal fail-closed reuse guards
        # remain unchanged.
        if cfg.refresh_witnesses:
          print("[witness-refresh] ignoring the persisted witness block and recertifying every "
                "IK action against the current scene/model.")
        elif _witnesses_baseline_compatible(_saved_cfg, cfg):
          baseline_witnesses = read_witnesses(cfg.load_plan, joint_names, len(plan))
          _saved_witness_stage = read_witness_stage(cfg.load_plan)
          if baseline_witnesses is not None:
            skill_paths = read_skill_paths(
                cfg.load_plan, joint_names, len(plan), baseline_witnesses)
            baseline_pre_witnesses = read_pre_witnesses(
                cfg.load_plan, joint_names, len(plan), baseline_witnesses,
                require_full_assemble=bool(
                    cfg.pre_insertion > 0.0 and any(
                        a and a[0] == "assemble" and assembly.screw_skill(a[2]) is None
                        for a in plan)))
        if baseline_witnesses is None and cfg.refresh_witnesses:
          from solver import collect_plan_witnesses
          baseline_witnesses = collect_plan_witnesses(
              C, assembly, plan, robots, robot_ik,
              gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
              handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing,
              resolve_missing=True)
        if baseline_witnesses is None and (
            cfg.optimize_witnesses is not None or cfg.motion_plan or cfg.save is not None):
          raise ValueError(
              f"{cfg.load_plan} has no compatible complete search-witness certificate. "
              "Refusing to stochastically redo IK during replay/post-processing. Re-run the "
              "original solve once with this version, or explicitly pass --refresh-witnesses "
              "to certify this legacy plan.")
      elif _need_witnesses and not cfg.load_plan:
        from solver import collect_plan_witnesses
        baseline_witnesses = collect_plan_witnesses(
            C, assembly, plan, robots, robot_ik,
            gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
            handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing,
            resolve_missing=True)
      if baseline_witnesses is not None:
        require_complete_witnesses(plan, baseline_witnesses,
                                   who="source search-witness persistence")
        if cfg.load_plan and cfg.refresh_witnesses:
          # Explicit refresh has just rebuilt every IK-bearing edge under today's complete
          # feasibility gate. It is therefore a genuine certificate upgrade, unlike an
          # ordinary loaded-plan post-pass (which remains forbidden above). Persist it so
          # subsequent motion planning/replay can reuse the witnesses without another solve.
          cfg._search_witness_margin = (0.0 if cfg.witness_margin is None
                                        else float(cfg.witness_margin))
          print(f"[witness-refresh] fixed plan recertified at "
                f"{1000.0 * cfg._search_witness_margin:.1f}mm witness margin.")

      # A saved plan already contains every post-pass that ran before save_plan. Reapplying
      # the non-idempotent hoist on --load-plan can move releases a SECOND time, so the motion
      # path is then for a different action order than the JSON which sim.py reloads. The
      # saved true value documents how the plan was made; it is not a request to mutate it on
      # every load. An explicit CLI --hoist-releases remains the opt-in way to apply the pass
      # to an older/unhoisted file.
      _hoist_applied = False
      if _should_hoist_plan(cfg, explicit):
        gk = wrench_kind_arg(C, robots) if cfg.wrench else None
        if robot_ik is None:
          # There is no keyframe certificate in a floating-EE / symbolic-only solve. Retain the
          # original cheap-geometric hoist behaviour in that mode.
          from solver import hoist_releases
          plan = hoist_releases(C, assembly, plan, robots, gripper_kind=gk,
                                handovers=cfg.handovers, allow_place=cfg.allow_place,
                                bracing=cfg.bracing)
        elif baseline_witnesses is None:
          # A final loaded MP export is a zero-IK artifact for this exact plan; changing action
          # order would invalidate its task signature and there is no source certificate to
          # transactionally preserve here.
          print("[hoist] skipped: no source witness block is available to recertify the "
                "rewritten plan; retaining the original plan.")
        else:
          from solver import hoist_releases_certified
          plan, baseline_witnesses, _hoist_applied = hoist_releases_certified(
              C, assembly, plan, baseline_witnesses, robots, robot_ik,
              gripper_kind=gk, handovers=cfg.handovers, allow_place=cfg.allow_place,
              bracing=cfg.bracing)
          if _hoist_applied:
            # Persisted optimized witnesses/sets were aligned to the source ordering.  The
            # newly certified candidate is a search-stage baseline; any requested optimizer
            # must run again rather than treating the old final stage as positionally reusable.
            _saved_witness_stage = None

      if cfg.grasp_closure_check:
        if cfg.load_plan and not cfg.refresh_witnesses and not _hoist_applied:
          grasp_closures = read_grasp_closures(cfg.load_plan, plan, required=True)
        else:
          from solver import collect_plan_grasp_closures
          grasp_closures = collect_plan_grasp_closures(
              C, assembly, plan, robots,
              gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
              handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing)
        print(f"[closure-persist] {len(grasp_closures)} two-finger grasp aperture "
              "certificate(s) aligned to the final plan.")

      if baseline_witnesses is not None:
        require_complete_witnesses(plan, baseline_witnesses,
                                   who="search-witness persistence")
        _required_pre = [i for i, a in enumerate(plan)
                         if a and a[0] == "assemble" and cfg.pre_insertion > 0.0
                         and assembly.screw_skill(a[2]) is None]
        if not cfg.load_plan or cfg.refresh_witnesses or _hoist_applied:
          from solver import collect_plan_pre_witnesses
          baseline_pre_witnesses = collect_plan_pre_witnesses(
              C, assembly, plan, baseline_witnesses, robots, robot_ik,
              gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
              handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing,
              resolve_missing=True)
        _missing_pre = [i for i in _required_pre
                        if i not in (baseline_pre_witnesses or {})]
        if _missing_pre:
          detail = ", ".join(f"{i}:{tuple(plan[i])}" for i in _missing_pre[:8])
          raise ValueError(
              "search witness certificate is missing its certified pre-insertion endpoint "
              f"at {detail}. Refusing export-time IK reconstruction; rerun/refresh the "
              "search with this version.")
        print(f"[witness-persist] {sum(w is not None for w in baseline_witnesses)} exact "
              f"search witnesses and {len(baseline_pre_witnesses or {})} insertion mouths "
              "aligned to the final plan; later stages need not redo IK.")
      if cfg.save is not None:
        save_plan(cfg.save, cfg, plan,
                  witnesses=(witness_block(
                      joint_names, baseline_witnesses, stage="search",
                      pre_witnesses=baseline_pre_witnesses,
                      grasp_closures=grasp_closures)
                             if baseline_witnesses is not None else None),
                  C=C, assembly=assembly, robots=robots)
      opt_witnesses = baseline_witnesses
      opt_pre_witnesses = baseline_pre_witnesses
      witness_alts = None                                    # --witness-sets alternatives
      optimization_candidates = None                         # witness optimizer's ladder
      diagnostic_candidates = None                           # broad display-only IK branches
      _reuse_final = False
      if (plan is not None and cfg.optimize_witnesses is not None
          and not (cfg.motion_plan and cfg.mp_export_load)):
        if robot_ik is None:
          print("[witopt] --optimize-witnesses needs --real-ik; skipping.")
        else:
          # WITNESS PERSISTENCE: a loaded plan file may carry the optimized keyframes, making
          # this stage a pure function of the file -- no witfloor re-solve, no possibility of
          # witness drift between stages. Reused only when every witness-shaping setting
          # matches the saved config; --refresh-witnesses forces the re-solve.
          _reuse_final = bool(cfg.load_plan and not cfg.refresh_witnesses
                              and _witnesses_reusable(_saved_cfg, cfg)
                              and baseline_witnesses is not None
                              and _saved_witness_stage == "optimized")
          if _reuse_final:
            from plan_io import (read_diagnostic_candidates,
                                 read_optimization_candidates, read_witness_sets)
            if opt_witnesses is not None:
              print(f"[witness-reuse] {sum(w is not None for w in opt_witnesses)} persisted "
                    f"witnesses loaded from {cfg.load_plan}; witness optimization SKIPPED "
                    "(--refresh-witnesses to re-solve).")
              if cfg.witness_sets and cfg.witness_sets >= 2:
                witness_alts = read_witness_sets(cfg.load_plan,
                                                 list(robot_ik.C.getJointNames()), len(plan))
                if witness_alts:
                  print(f"[witsets-reuse] {len(witness_alts)} persisted witness set(s) "
                        f"loaded from {cfg.load_plan}.")
              optimization_candidates = read_optimization_candidates(
                  cfg.load_plan, list(robot_ik.C.getJointNames()), len(plan))
              if optimization_candidates:
                print(f"[witopt-candidates] {len(optimization_candidates)} persisted "
                      f"optimizer candidate pool(s) loaded from {cfg.load_plan}.")
              diagnostic_candidates = read_diagnostic_candidates(
                  cfg.load_plan, list(robot_ik.C.getJointNames()), len(plan))
              if diagnostic_candidates:
                print(f"[witdiag-candidates] {len(diagnostic_candidates)} persisted broad "
                      f"diagnostic candidate pool(s) loaded from {cfg.load_plan}.")
          if not _reuse_final and cfg.optimize_witnesses == "floor":
            from witness_floor import optimize_witnesses_floor
            # floor mode CONSTRAINS clearance (a per-step normalised floor) rather than
            # weighting it, so there is nothing for --witness-clearance to scale. Say so
            # instead of accepting the flag and ignoring it.
            if cfg.witness_clearance:
              print(f"[witfloor] NOTE: --witness-clearance {cfg.witness_clearance} is IGNORED "
                    "in floor mode (the floor is a constraint, not a weight); use "
                    "--witness-branch-budget to buy a higher floor.")
            if cfg.witness_sets and cfg.witness_sets >= 2:
              print(f"[witsets] --witness-sets {cfg.witness_sets} ACTIVE: exporting up to "
                    f"{cfg.witness_sets} same-grasp IK-branch alternatives per task as "
                    f"GoalSets (spread >= {cfg.witness_set_spread:g} rad); OFF = "
                    "byte-identical export.")
            opt_witnesses, _rep = optimize_witnesses_floor(
                C, assembly, plan, robots, robot_ik,
                gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
                handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing,
                branch_budget=cfg.witness_branch_budget,
                branch_metric=cfg.witness_branch_metric,
                reconfig_thresh=cfg.witness_reconfig_thresh,
                draw_seed=cfg.witness_draw,
                witness_sets=cfg.witness_sets, set_spread=cfg.witness_set_spread,
                baseline_witnesses=baseline_witnesses)
            witness_alts = _rep.get("wit_alts") or None
            optimization_candidates = _rep.get("optimization_candidates") or None
            diagnostic_candidates = _rep.get("diagnostic_candidates") or None
          elif not _reuse_final:
            opt_witnesses, _rep = optimize_witnesses(
                C, assembly, plan, robots, robot_ik, cost=cfg.optimize_witnesses,
                gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
                handovers=cfg.handovers, allow_place=cfg.allow_place,
                bracing=cfg.bracing, clear_weight=cfg.witness_clearance,
                robot_collisions=cfg.witness_robot_collisions,
                baseline_witnesses=baseline_witnesses)
            optimization_candidates = _rep.get("optimization_candidates") or None
      _has_path_skill = bool(plan and any(
          a and a[0] == "assemble" and assembly.screw_skill(a[2]) is not None
          for a in plan))
      if _has_path_skill and opt_witnesses is not None:
        _required_skill_steps = [i for i, a in enumerate(plan)
                                 if a and a[0] == "assemble"
                                 and assembly.screw_skill(a[2]) is not None]
        _paths_match = bool(skill_paths) and all(
            i in skill_paths for i in _required_skill_steps)
        if _paths_match:
          for _i in _required_skill_steps:
            _path = skill_paths[_i]
            if (_i >= len(opt_witnesses) or opt_witnesses[_i] is None or not _path
                or not np.allclose(_path[-1], opt_witnesses[_i], atol=1e-9, rtol=0.0)):
              _paths_match = False
              break
        if not _paths_match:
          if cfg.load_plan and not cfg.refresh_witnesses and _reuse_final:
            raise ValueError(
                f"{cfg.load_plan} has endpoint witnesses for an atomic screw skill but no "
                "matching path certificate. Refusing to redo skill IK implicitly; re-run "
                "the original solve once with this version.")
          from solver import certify_plan_skill_paths
          opt_witnesses, skill_paths = certify_plan_skill_paths(
              C, assembly, plan, opt_witnesses, robots, robot_ik,
              gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
              handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing,
              fallback_witnesses=baseline_witnesses)
          # A fallback may have restored the search endpoint, invalidating a set whose member
          # zero was the optimized endpoint.  Drop only that step's optional alternatives.
          if witness_alts:
            witness_alts = {i: ms for i, ms in witness_alts.items()
                            if np.allclose(ms[0], opt_witnesses[i], atol=1e-9, rtol=0.0)}
          if optimization_candidates:
            optimization_candidates = {
                i: ms for i, ms in optimization_candidates.items()
                if np.allclose(ms[0], opt_witnesses[i], atol=1e-9, rtol=0.0)}
          if diagnostic_candidates:
            diagnostic_candidates = {
                i: ms for i, ms in diagnostic_candidates.items()
                if np.allclose(ms[0], opt_witnesses[i], atol=1e-9, rtol=0.0)}
      # Search-stage plans already loaded their matching pre_wit block above.  Re-solving
      # those mouths is both unnecessary and non-reproducible: a fresh optimizer can land in
      # another basin and make an otherwise complete saved certificate fail during replay.
      # Only a pass that actually CHANGED endpoint witnesses needs matching mouths rebuilt.
      _endpoints_postprocessed = bool(cfg.optimize_witnesses is not None or _has_path_skill)
      if (opt_witnesses is not None and _required_pre and not _reuse_final
          and _endpoints_postprocessed):
        from solver import collect_plan_pre_witnesses
        opt_pre_witnesses = collect_plan_pre_witnesses(
            C, assembly, plan, opt_witnesses, robots, robot_ik,
            gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
            handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing,
            resolve_missing=True)
        _missing_pre = [i for i in _required_pre if i not in opt_pre_witnesses]
        if _missing_pre:
          raise ValueError(
              "optimized witness lacks a matching certified pre-insertion endpoint at "
              + ", ".join(map(str, _missing_pre)) + "; retaining/re-solving it later would "
              "change the IK certificate")
      if (cfg.save is not None and plan is not None and opt_witnesses is not None
          and (cfg.optimize_witnesses is not None or _has_path_skill)):
        # RE-save with the optimized witnesses attached (the first save above is deliberately
        # EARLY so a witness-pass crash cannot lose the plan). With this block downstream
        # stages (main --load-plan --motion-plan, sim.py run) reuse the exact keyframes
        # instead of re-solving them.
        require_complete_witnesses(plan, opt_witnesses,
                                   who="optimized witness export")
        save_plan(cfg.save, cfg, plan,
                  witnesses=witness_block(joint_names, opt_witnesses,
                                          alts=witness_alts,
                                          optimization_candidates=optimization_candidates,
                                          diagnostic_candidates=diagnostic_candidates,
                                          stage=("optimized" if cfg.optimize_witnesses is not None
                                                 else "search"),
                                          skill_paths=skill_paths,
                                          pre_witnesses=opt_pre_witnesses,
                                          grasp_closures=grasp_closures),
                  C=C, assembly=assembly, robots=robots)
      if plan is not None and cfg.utilization:
        from solver import utilization_report
        utilization_report(C, assembly, plan, robots, witnesses=opt_witnesses,
                           gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
                           handovers=cfg.handovers, allow_place=cfg.allow_place,
                           bracing=cfg.bracing)
      if (plan is not None and cfg.fuse_cograsp and not cfg.motion_plan
          and any(a and a[0] == "add_grasp" for a in plan)):
        # the fusion pass only ever ran inside the --motion-plan branch, so without it the
        # flag is inert -- and the REPLAY then shows the unfused pair, whose brace arm sits in
        # a different IK branch at add_grasp than at assemble. That looks like a broken pose.
        # (fuse_cograsp defaults ON since 2026-07-29, so only warn when the plan actually
        # CONTAINS a cograsp -- otherwise the note is perpetual noise on plain solves.)
        print("[flag] --fuse-cograsp has NO EFFECT without --motion-plan: it rewrites the "
              "MOTION-PLAN witnesses only. The replay/--viser shows the UNFUSED witnesses, "
              "where a braced arm may change IK branch between add_grasp and assemble.")
      if plan is not None and cfg.motion_plan:
        if robot_ik is None:
          print("[motion-plan] needs --real-ik (for the IK witnesses); skipping.")
        else:
          if cfg.mp_save is not None and not cfg.mp_optimize:
            raise SystemExit(
                "refusing to export an unoptimized motion path: --no-mp-optimize is only "
                "available for in-memory first-solution diagnostics. Remove it (the default "
                "and every named motion profile enable RRT* optimization), or omit --mp-save.")
          if cfg.mp_wrench_gate and not cfg.wrench:
            raise ValueError(
                "--mp-wrench-gate requires --wrench: a --no-wrench plan is the topological "
                "baseline and has no wrench-certified symbolic state. Omit the MP gate for "
                "that baseline, or re-run task planning with --wrench.")
          import mp_export as MP
          from plan_io import _motion_plan_digest, motion_data_states
          _motion_data = (cfg._loaded_motion_data or
                          getattr(cfg, "_motion_data_cache", {}).get(
                              (_motion_plan_digest(plan), cfg.grasp_sampler_db_hash)))
          _motion_states = None
          if _motion_data is not None:
            from problem import get_initial_state
            get_initial_state(C, assembly)  # restore anchor metadata, without grasp generation
            _motion_states = motion_data_states(_motion_data, assembly)
            print(f"[motion-data] {len(_motion_states)} saved symbolic states and "
                  f"{len(_motion_data['selected_grasps'])} selected grasps reused")
          gk = wrench_kind_arg(C, robots) if cfg.wrench else None
          _loaded_export = None
          _export_bracing = cfg.bracing
          _export_depgraph = not cfg.mp_sequence
          _export_collision_tolerance = cfg.mp_collision_tolerance
          _export_robot_margin = cfg.mp_robot_margin
          _export_margin_ramp = cfg.mp_margin_ramp
          _export_margin_mode = cfg.mp_margin_mode
          _export_table_margin = cfg.mp_table_margin
          _export_table_float = cfg.part_float if cfg.mp_table_margin else 0.0
          _export_wrench_gate = cfg.mp_wrench_gate
          # A physics-certified two-finger database is executed with the articulated Robotiq
          # pads, not only the historical capsule proxies.  Keep those exact open-pad boxes
          # in the motion collision model as well; this changes geometry, so persist it in the
          # final export rather than re-inferring it during simulation.
          _export_physical_fingers = bool(cfg.grasp_physics_check)
          mp_pre = dict(opt_pre_witnesses or {})
          _pre_pin_robots = {}
          _cograsp_pre = {}
          if cfg.mp_export_load:
            _loaded_export = MP.load_export_bundle(
                cfg.mp_export_load, plan, list(robot_ik.C.getJointNames()))
            mp_wit = _loaded_export["witnesses"]
            mp_pre = _loaded_export["pre_witnesses"]
            witness_alts = _loaded_export["witness_sets"]
            _es = _loaded_export["settings"]
            _export_bracing = bool(_es.get("bracing", _export_bracing))
            _export_depgraph = bool(_es.get("depgraph", _export_depgraph))
            _clearance, _clearance_source = _merge_mp_export_clearance_settings(
                cfg, _es, explicit)
            _export_collision_tolerance = _clearance["collision_tolerance"]
            _export_robot_margin = _clearance["robot_robot_margin"]
            _export_margin_ramp = _clearance["robot_margin_ramp"]
            _export_margin_mode = _clearance["robot_margin_mode"]
            _export_table_margin = _clearance["table_margin"]
            _export_table_float = _clearance["table_float"]
            _export_wrench_gate = _clearance["wrench_gate"]
            _export_physical_fingers = bool(
                _es.get("physical_finger_collisions", _export_physical_fingers))
            robot_ik.pre_insertion_delta = float(
                _es.get("pre_insertion", robot_ik.pre_insertion_delta))
            robot_ik.pre_pick_delta = float(_es.get("pre_pick", robot_ik.pre_pick_delta))
            robot_ik.post_pick_delta = float(_es.get("post_pick", robot_ik.post_pick_delta))
            robot_ik.pre_place_delta = float(_es.get("pre_place", robot_ik.pre_place_delta))
            robot_ik.pre_add_grasp_delta = float(
                _es.get("pre_add_grasp", robot_ik.pre_add_grasp_delta))
            robot_ik.post_release_delta = float(
                _es.get("post_release", robot_ik.post_release_delta))
            print(f"[mp-export-reuse] final post-fusion export loaded from "
                  f"{cfg.mp_export_load}: {sum(w is not None for w in mp_wit)} skill and "
                  f"{len(mp_pre)} pre-skill keyframes; ALL witness IK/fusion solves SKIPPED.")
            print("[mp-export-settings] effective "
                  f"collision_tolerance={_export_collision_tolerance:g} "
                  f"[{_clearance_source['collision_tolerance']}], "
                  f"robot_margin={_export_robot_margin:g} "
                  f"[{_clearance_source['robot_robot_margin']}], "
                  f"margin_ramp={_export_margin_ramp:g} "
                  f"[{_clearance_source['robot_margin_ramp']}], "
                  f"margin_mode={_export_margin_mode} "
                  f"[{_clearance_source['robot_margin_mode']}], "
                  f"table_margin={_export_table_margin:g} "
                  f"[{_clearance_source['table_margin']}], "
                  f"table_float={_export_table_float:g} "
                  f"[{_clearance_source['table_float']}], "
                  f"wrench_gate={_export_wrench_gate} "
                  f"[{_clearance_source['wrench_gate']}].")
          else:
            # witnesses: the optimized set if --optimize-witnesses was given, else RAW
            # (sweeps=0). This is the ONE-TIME export path; replay loads its final artifact.
            mp_wit = opt_witnesses
            if mp_wit is None:
              mp_wit, _ = optimize_witnesses(
                  C, assembly, plan, robots, robot_ik, cost="ee", sweeps=0,
                  gripper_kind=gk, handovers=cfg.handovers, allow_place=cfg.allow_place,
                  bracing=cfg.bracing, verbose=False,
                  robot_collisions=cfg.witness_robot_collisions,
                  baseline_witnesses=baseline_witnesses)
            if cfg.fuse_cograsp:
              from solver import fuse_cograsp_witnesses
              print("[flag] --fuse-cograsp ACTIVE: re-solving each (add_grasp, assemble) "
                    "witness pair as ONE 2-phase KOMO. add_grasp shares the assembly's "
                    "funnel-mouth rendezvous at the keyframe level; the ordinary motion "
                    "planner remains free to detour between endpoints.")
              mp_wit, _frep = fuse_cograsp_witnesses(
                  C, assembly, plan, mp_wit, robots, robot_ik, gripper_kind=gk,
                  handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing,
                  states=_motion_states)
              _cograsp_pre = dict(_frep.get("pre_witnesses") or {})
              if _cograsp_pre:
                mp_pre.update(_cograsp_pre)
                print(f"[cograsp-chain] using {len(_cograsp_pre)} fused add_grasp/pre-assemble "
                      "rendezvous keyframe(s).")
          require_complete_witnesses(plan, mp_wit,
                                     who="motion-plan witness export")
          # LAST GATE: re-check EVERY IK keyframe we are about to SHIP. Every earlier gate ran
          # at SOLVE time, and the object that leaves here is not necessarily the object that
          # was solved -- optimize_witnesses* emits a splice and fuse_cograsp re-solves pairs.
          # This used to inspect only assemble, allowing a handover to put its freely-held
          # subassembly through the table and fail later as an unexplained "no path".
          if (robot_ik.penetration_reject is not None
              or getattr(robot_ik, "object_over_table", False)):
            from solver import validate_export_witnesses
            _pre_gate_input_wit = list(mp_wit)
            mp_wit, _bad = validate_export_witnesses(
                C, assembly, plan, mp_wit, robots, robot_ik, gripper_kind=gk,
                handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing,
                baseline_witnesses=(baseline_witnesses if _loaded_export is None else mp_wit),
                states=_motion_states)
            # A collision-gate rollback changes the seated half of the certificate. Restore
            # the matching baseline mouth with it; never retain a mouth tied to the rejected
            # optimized/fused endpoint.
            if _loaded_export is None:
              for _i in _required_pre:
                if (not np.allclose(_pre_gate_input_wit[_i], mp_wit[_i],
                                    atol=1e-9, rtol=0.0)):
                  if (_i in (baseline_pre_witnesses or {})
                      and np.allclose(mp_wit[_i], baseline_witnesses[_i],
                                      atol=1e-9, rtol=0.0)):
                    mp_pre[_i] = np.asarray(baseline_pre_witnesses[_i], dtype=float).copy()
                    print(f"  [export-gate] step {_i}: restored its paired search-certified "
                          "pre-insertion mouth as well")
                  else:
                    mp_pre.pop(_i, None)
            if _bad:
              print(f"[export-gate] FAILURE: {len(_bad)} IK keyframe(s) exceed the search "
                    f"collision gate; motion planning was NOT started:")
              for _i, _act, _p, _, _why in _bad:
                # Clearance reasons are (frame_a, frame_b, distance, class); the
                # object-over-table gate reports (part, excess, table_xy).  Keep this
                # diagnostic fail-closed without assuming the reason schemas coincide.
                if (_why is not None and len(_why) == 3
                    and isinstance(_why[2], (tuple, list))):
                  print(f"    step {_i} {_act[:4]}: {_why[0]} datum is "
                        f"{_why[1] * 1000:.1f}mm outside the table footprint at "
                        f"xy=({_why[2][0]:.3f}, {_why[2][1]:.3f})m")
                elif _why is not None and len(_why) >= 4 and _why[3] == "joint-limit":
                  print(f"    step {_i} {_act[:4]}: {_why[0]} has only "
                        f"{_why[2]:.6f}rad limit slack "
                        f"(gate {robot_ik.joint_margin_reject:g}rad)")
                elif _why is not None and len(_why) >= 4:
                  print(f"    step {_i} {_act[:4]}: {_why[3]} pair {_why[0]} ~ {_why[1]} "
                        f"has {_why[2] * 1000:.2f}mm clearance "
                        f"(gate {robot_ik.clearance_reject * 1000:.1f}mm)")
                else:
                  print(f"    step {_i} {_act[:4]}: {_p * 1000:.2f}mm penetration "
                        f"(gate {robot_ik.penetration_reject * 1000:.3f}mm)")
              _hint = ("Regenerate the final motion export with this version; the loaded "
                       "artifact is invalid." if _loaded_export is not None else
                       "The persisted search witness is also invalid; retry the search so "
                       "the action receives a clean witness.")
              raise ValueError(f"motion export contains invalid IK keyframes. {_hint}")
            else:
              print("[export-gate] every exported IK keyframe is collision-free "
                    "(only intended skill contacts exempt).")
          # AFTER fusing: this pass never re-solves a SKILL witness, so a fused pair survives
          # untouched -- it only replaces the pre-* GOTO keyframes the exporter would solve.
          # ``mp_pre`` already contains the exact mouths certified with the selected seated
          # witnesses.  Do not discard them and ask an export process to redo IK.
          # DEFAULT ON where the pin can actually bite. It cannot for `pick` or a solo
          # `place` (_ik_pick/_ik_place already restrict the scene to the acting arm, so
          # pin_applied == 0 and the pass rejects) -- that is 6/14 splits on car-mini and 16/22
          # on cross8 of pure cost. It CAN bite for assemble/add_grasp, which is exactly where
          # the defect is: without it the target assembly slides 16.8-48.8mm between the funnel
          # mouth and the seat while the inserted part covers 6.1-35.8mm of a nominal 50mm.
          # Now safe to default because a REJECTED split falls back to the exporter's own solve
          # (mp_export._SOLVE_IT) instead of degrading to a zero-length funnel.
          _pin_default = (cfg.pre_insertion or 0.0) > 0.0 or (cfg.pre_add_grasp or 0.0) > 0.0
          if _loaded_export is None and cfg.fuse_pre_skill is None:
            cfg.fuse_pre_skill = _pin_default
            if _pin_default:
              print("[sim] --fuse-pre-skill defaulted ON (a --pre-insertion/--pre-add-grasp "
                    "split is present, which is where the pin bites). Disable with "
                    "--no-fuse-pre-skill.")
          if _loaded_export is None and cfg.fuse_pre_skill:
            from solver import fuse_pre_skill_witnesses
            print("[flag] --fuse-pre-skill ACTIVE: each pre-* GOTO keyframe is re-solved with "
                  "every arm that does NOT move with the skill PINNED (hard eq) to the skill "
                  "witness, then gated on actor dq <= 0.35 rad, idle-arm dq <= 0.02 rad, "
                  "stationary body <= 2.0mm, funnel residual <= 1mm/1deg and penetration -- "
                  "each of those 'or no worse than today's', plus a STRICT improvement in the "
                  "approach-axis deviation. A rejected split keeps today's keyframe; the task "
                  "structure never changes.")
            mp_pre, _prep = fuse_pre_skill_witnesses(
                C, assembly, plan, mp_wit, robots, robot_ik, gripper_kind=gk,
                handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing,
                certified_pre_witnesses=mp_pre, states=_motion_states)
            _pre_pin_robots = dict(_prep.get("pin_robots") or {})
            if _pre_pin_robots:
              print(f"[pre] {len(_pre_pin_robots)} accepted funnel solve(s) used idle-arm IK "
                    "pins. They remain transition hints; idle arms are not task participants.")
          if _loaded_export is None:
            _missing_mp_pre = [i for i in _required_pre if i not in (mp_pre or {})]
            if _missing_mp_pre:
              raise ValueError(
                  "final motion witness lacks its paired certified pre-insertion endpoint at "
                  + ", ".join(map(str, _missing_mp_pre))
                  + "; refusing to reconstruct IK during export")
          # The generic pre-skill pass knows how to pin an ordinary funnel, but the cograsp
          # pass solves the stronger three-arm rendezvous. Its accepted keyframes therefore
          # win for the paired assemble indices.
          if _export_collision_tolerance != 0.01:
            print(f"[flag] --mp-collision-tolerance={_export_collision_tolerance:g}: exported "
                  f"problem validates configs at this total-penetration budget.")
          if cfg.mp_seed is not None and not cfg.mp_fake_clock:
            print(f"[flag] NOTE: --mp-seed {cfg.mp_seed}: BiRRT draws are seeded, but "
                  f"termination uses the REAL wall clock, so runs are only approximately "
                  f"reproducible under machine jitter. Add --mp-fake-clock for bit-exact "
                  f"determinism.")
          if cfg.mp_fake_clock:
            print(f"[flag] NOTE: --mp-fake-clock ACTIVE: BiRRT swaps the wall clock for a "
                  f"deterministic fake clock, so --mp-runtime {cfg.mp_runtime:g} is a "
                  f"CONSULTATION budget, NOT wall seconds -- the tree phase can run much "
                  f"longer than {cfg.mp_runtime:g} s.")
          _resolved_pre = dict(mp_pre or {})
          prob = MP.make_problem(C, assembly, plan, mp_wit, robots, robot_ik, gk,
                                 collision_tolerance=_export_collision_tolerance,
                                 bracing=_export_bracing, depgraph=_export_depgraph,
                                 pre_witnesses=mp_pre,
                                 robot_robot_margin=_export_robot_margin,
                                 robot_margin_ramp=_export_margin_ramp,
                                 robot_margin_mode=_export_margin_mode,
                                 table_margin=_export_table_margin,
                                 table_float=_export_table_float,
                                 wrench_gate=_export_wrench_gate,
                                 witness_sets=witness_alts,
                                 resolved_pre_witnesses=_resolved_pre,
                                 resolve_missing_pre=(_loaded_export is None),
                                 task_records=(None if _loaded_export is None else
                                               _loaded_export["tasks"]),
                                 saved_step_names=(None if _loaded_export is None else
                                                   _loaded_export["step_names"]),
                                 saved_sequence_names=(None if _loaded_export is None else
                                                       _loaded_export["sequence_names"]),
                                 pre_pin_robots=_pre_pin_robots,
                                 skill_paths=skill_paths,
                                 physical_finger_collisions=_export_physical_fingers,
                                 motion_states=_motion_states)
          print(f"[motion-plan] {'DEPGRAPH' if _export_depgraph else 'SEQUENCE'} problem, "
                f"{len(prob.tasks)} tasks")
          if _loaded_export is not None:
            _want_sig = _loaded_export.get("task_signature")
            _have_sig = MP.task_signature(prob)
            if _want_sig is not None and _have_sig != _want_sig:
              raise ValueError(
                  f"final motion export task signature changed: saved {_want_sig}, rebuilt "
                  f"{_have_sig}; refusing to replay against different task semantics")
          # The isolated witness gate above cannot see error accumulated by mr_bench's real
          # attach semantics. Walk the canonical exported chain once before saving or planning:
          # every idle arm retains its preceding goal, and every pick/mate/handover side effect
          # is applied. A bad endpoint is not a narrow passage and no RRT budget can repair it.
          _chain_bad = MP.validate_nominal_keyframes(
              prob, tolerance=_export_collision_tolerance, verbose=True)
          if _chain_bad:
            _i, _nm, _p, _pairs = _chain_bad[0]
            raise ValueError(
                f"exported attachment chain has {len(_chain_bad)} collision-invalid "
                f"keyframe(s); first is task {_i} {_nm} ({_p * 1000:.2f}mm). "
                "The motion planner was not started. Retry the search/witness set; increasing "
                "BiRRT runtime or collision tolerance does not repair an invalid endpoint.")
          print("[export-chain-gate] canonical accumulated attachment chain is collision-free.")
          _export_settings = {
              "bracing": bool(_export_bracing), "depgraph": bool(_export_depgraph),
              "collision_tolerance": float(_export_collision_tolerance),
              "robot_robot_margin": float(_export_robot_margin or 0.0),
              "robot_margin_ramp": float(_export_margin_ramp or 0.0),
              "robot_margin_mode": str(_export_margin_mode),
              "table_margin": float(_export_table_margin or 0.0),
              "table_float": float(_export_table_float or 0.0),
              "wrench_gate": bool(_export_wrench_gate),
              "physical_finger_collisions": bool(_export_physical_fingers),
              "pre_insertion": float(robot_ik.pre_insertion_delta),
              "pre_pick": float(robot_ik.pre_pick_delta),
              "post_pick": float(robot_ik.post_pick_delta),
              "pre_place": float(robot_ik.pre_place_delta),
              "pre_add_grasp": float(robot_ik.pre_add_grasp_delta),
              "post_release": float(robot_ik.post_release_delta),
          }
          _mp_export_bundle = MP.make_export_bundle(
              prob, plan, list(robot_ik.C.getJointNames()), mp_wit, _resolved_pre,
              witness_sets=witness_alts, settings=_export_settings)
          if cfg.mp_export_save:
            MP.save_export_bundle(cfg.mp_export_save, _mp_export_bundle)
            print(f"[mp-export] saved final zero-IK replay artifact -> {cfg.mp_export_save}")
          if cfg.mp_clearance:
            # pass OUR per-step scene so the report can name parts that sit elsewhere here
            from solver import _replay_states as _rs, _search_context as _sc
            try:
              _ck2, _gt2 = _sc(C, assembly, robots)
              _st = _rs(C, assembly, plan, robots, gk, cfg.handovers, cfg.allow_place,
                        cfg.bracing, _gt2, "mp-clearance")

              def _ref(i, _st=_st):
                if mp_wit[i] is None:
                  return None
                Cx = robot_ik.scene_for(_st[i], assembly)
                Cx.setJointState(np.asarray(mp_wit[i], dtype=float).ravel())
                return Cx
            except Exception as _e:
              print(f"[clearance] scene-drift compare unavailable: {type(_e).__name__}")
              _ref = None
            MP.report_problem_clearance(prob, plan=plan, ref_scene=_ref,
                                        problem_assembly=assembly)
          if cfg.mp_modes:
            import sys as _modes_sys
            MP._ensure_mrbench()
            _modes_sys.path.insert(
                0, _os.path.join(_os.path.dirname(MP._MRBENCH_SRC), "scripts"))
            from show_problems import visualize_modes
            # mr_bench's generic mode viewer hard-codes ``primitives_only=True`` and does
            # not forward a port.  That silently hides every file-backed shape, including
            # legitimate mesh COLLIDERS (Fabrica Beam's two uprights), so a complete mode
            # appears to have missing parts.  Keep mr_bench unmodified, but intercept its
            # final display call and apply this CLI's normal viewer policy.  Primitive-only
            # rendering remains available explicitly via --mp-collision-view.
            _display_modes = prob.display_path_viser

            def _display_modes_with_cli(paths, *args, **kwargs):
              kwargs["port"] = cfg.viser_port
              kwargs["primitives_only"] = bool(cfg.mp_collision_view)
              return _display_modes(paths, *args, **kwargs)

            prob.display_path_viser = _display_modes_with_cli
            _geom = "primitive collision shapes only" if cfg.mp_collision_view \
                else "complete visual and collision geometry"
            print(f"[motion-plan] serving MODES on http://localhost:{cfg.viser_port} "
                  f"({_geom})")
            visualize_modes(prob, use_viser=True)
            return
          # One edge certificate, end to end.  mr_bench's RAI environment defaults to a
          # coarse 0.01-rad interpolation step, while standard shortcutting and simulation
          # preflight use 0.002 rad.  A narrow collision can therefore be accepted by BiRRT
          # and rejected only after an expensive successful run (Fabrica Beam: 1.46mm at
          # alpha=.989 of the approach-to-funnel edge).  The configured certified shortcut
          # resolution is also the planner's edge resolution; endpoint tolerances remain
          # unchanged.  This is a sampling-resolution consistency fix, not a stricter gate.
          if cfg.mp_shortcut_res is not None:
            _old_res = float(getattr(prob, "collision_resolution", cfg.mp_shortcut_res))
            prob.collision_resolution = float(cfg.mp_shortcut_res)
            if abs(_old_res - prob.collision_resolution) > 1e-15:
              print(f"[motion-plan] unified edge resolution: {_old_res:g} -> "
                    f"{prob.collision_resolution:g}rad (BiRRT, shortcut, sim preflight)")
          mp_path, _ = MP.run_birrt(
              prob, runtime=cfg.mp_runtime, shortcut=cfg.mp_shortcut,
              shortcut_iters=cfg.mp_shortcut_iters, shortcut_resolution=cfg.mp_shortcut_res,
              shortcut_strategy=cfg.mp_shortcut_strategy,
              shortcut_pair_checks=cfg.mp_shortcut_pair_checks,
              optimize=cfg.mp_optimize, seed=cfg.mp_seed,
              stall_timeout=cfg.mp_stall_timeout, fake_clock=cfg.mp_fake_clock)
          if not mp_path and cfg.mp_stall_timeout is not None:
            # STRUCTURED stuck report for outer loops: name the frontier tasks (their names
            # embed the plan step), so a bandit can compute ban candidates without parsing
            # any debug output.
            _mpinfo = _  # run_birrt's info dict
            _tids = (_mpinfo or {}).get("frontier_task_ids") or []
            _names = [prob.tasks[t].name for t in _tids if 0 <= t < len(prob.tasks)]
            print(f"[motion-plan] STALLED frontier task_ids {_tids} names {_names}")
          if mp_path:
            print(f"[motion-plan] path {len(mp_path)} states, "
                  f"reaches terminal={prob.is_terminal_mode(mp_path[-1].mode)}")
            if cfg.mp_cograsp:
              print("[flag] --mp-cograsp ACTIVE: checking co-grasp rigidity along the path.")
              MP.report_cograsp_drift(C, prob, mp_path, robot_ik)
            if cfg.mp_save is not None:
              if not cfg.mp_optimize:                       # defense below CLI validation
                raise RuntimeError("internal error: attempted to save an unoptimized path")
              MP.save_path(cfg.mp_save, mp_path, problem=prob,
                           mp_export_bundle=_mp_export_bundle, optimized=True)
              print(f"[motion-plan] saved path -> {cfg.mp_save} (load in sim with "
                    f"`sim.py run <plan.json> --mp-path {cfg.mp_save}`)")
            if cfg.mp_viser:
              from multi_robot_multi_goal_planning.problems.util import interpolate_path
              disp = interpolate_path(mp_path, 0.05, kind="euclidean")
              print(f"[motion-plan] serving interpolated path ({len(disp)} states) on "
                    f"http://localhost:{cfg.viser_port}"
                    + (" (collision shapes only)" if cfg.mp_collision_view else ""))
              prob.display_path_viser(disp, port=cfg.viser_port,
                                      primitives_only=cfg.mp_collision_view)
          else:
            print("[motion-plan] NO path found")
      # --motion-plan owns the display (via --mp-viser); don't ALSO fire the search-replay
      # viz, which would open the BLOCKING rai viewer (cfg.view defaults True).
      if (cfg.view or cfg.viser) and not cfg.motion_plan:
        recorder = None
        if cfg.viser:
          from viser_view import ViserRecorder
          recorder = ViserRecorder()
        # wrench arrows are computed whenever the capability model is on; a viser GUI
        # checkbox ("Show wrenches") toggles their display per step.
        wgk = wrench_kind_arg(C, robots) if cfg.wrench else None
        if cfg.real_ik:
          visualize_plan_ik(C, assembly, plan, robots=robots, robot_ik=robot_ik,
                            recorder=recorder, bracing=cfg.bracing,
                            witnesses=opt_witnesses, wrench_gk=wgk)
        else:
          visualize_plan(C, assembly, plan, recorder=recorder, bracing=cfg.bracing,
                         wrench_gk=wgk)
        if recorder is not None:
          recorder.serve(port=cfg.viser_port)
    return

  if cfg.viser:
    from viser_view import ViserRecorder
    recorder = ViserRecorder()
    recorder.record(C, "scene")
    recorder.serve(port=cfg.viser_port)
  elif cfg.view:
    C.view(True)

def main(argv=None, args=None):
  # CLI / entry wrapper: build the typed RunConfig from parsed CLI args (or an override
  # namespace a caller passed) and hand it to run(). Library callers may instead build a
  # RunConfig directly and call run(cfg) -- no argparse, no namespace round-trip.
  explicit = None
  if args is None:
    args = parse_args(argv)
    explicit = _scan_flags(build_parser(), argv)[1]
    # A motion profile is an explicit request to retune these fields, including when loading
    # an embedded final export. Mark its members explicit so the artifact-layer merge does not
    # silently reclaim collision/margin values merely because they came from a bundle rather
    # than individually spelled flags.
    if getattr(args, "motion_profile", None):
      explicit.update(_MOTION_PROFILES[args.motion_profile])
  cfg = RunConfig.from_namespace(args)
  # The label is diagnostic provenance only; RunConfig contains the resolved values and stays
  # usable without the CLI/profile layer.
  if getattr(args, "search_profile", None):
    cfg._search_profile = args.search_profile
  if getattr(args, "motion_profile", None):
    cfg._motion_profile = args.motion_profile
  return run(cfg, explicit=explicit)


if __name__ == "__main__":
  main()
