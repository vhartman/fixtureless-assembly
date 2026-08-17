import argparse
import os as _os
import sys as _sys

# BEFORE rai loads: RAI_QUIET=1 drops rai's libccd/pairCollision spam at the fd (see rai_quiet).
# Env var as well as a flag so it can be exported once instead of typed on every run.
if _os.environ.get("RAI_QUIET", "").strip() not in ("", "0", "false", "no"):
  import rai_quiet as _rq
  _rq.install()

import numpy as np

from env import *
from assembly import *
from config import RunConfig

def build_parser():
  parser = argparse.ArgumentParser(
      description="Spawn a multi-robot UR10 environment."
  )
  parser.add_argument(
      "--config", metavar="FILE", default=None,
      help="Load a RunConfig JSON file as the BASE configuration; any explicit command-line "
           "flag overrides the file's value (CLI wins). Write one with config.RunConfig.save "
           "or by hand. Without --config nothing changes (byte-identical default path).",
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
      "--floating-ee", action="store_true",
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
      "--robot", choices=["ur10", "ur10_limited", "ur5", "ur5_limited"], default="ur10",
      help="Robot arm for all robots: ur10 (default), ur10_limited (ur10 with a tighter "
           "shoulder-lift/elbow workspace), ur5, or ur5_limited (ur5 with the same "
           "restricted workspace). With --floating-ee the arm is "
           "irrelevant (a generic floating gripper); it matters for the real arm (no --floating-ee "
           "/ --real-ik) -- note the ur5 has a much shorter reach (~0.85 m vs ~1.3 m).",
  )
  g_scene.add_argument(
      "--robot-types", default=None,
      help="Advanced: comma-free per-slot override of the robot type (defaults to --robot for "
           "every slot). Rarely needed.",
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
      "--no-view", dest="view", action="store_false",
      help="Do not open the interactive viewer.",
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
      "--viz-rest-poses", action="store_true",
      help="REST-POSE CATALOG in viser (add --viser): each group (singletons, connect "
           "modules, full assembly) laid out on the table in every stable rest pose, one "
           "step per (group,face). A group with NO steps is an unparkable subassembly. No search.",
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
               "cuboid-tower", "car", "car-mini", "slot-pin", "peg-hole", "peg-stool",
               "peg-stool-4", "peg-stool-cuboid", "peg-stool-cuboid-4",
               "peg-stool-tight", "peg-stool-4-tight",
               "peg-stool-cuboid-tight", "peg-stool-cuboid-4-tight",
               "mesh-demo", "compound-demo", "chair"],
      default="cross",
      help="Assembly to build: cross (cube cross, 7), cube (2x2x2 grid, 8), "
           "arch (two pillars + lintel, 7 -- known-flawed star model), "
           "arch-seq (arch shape built as a sequential chain, 7), "
           "cube-plates (2x2x2 cube as two stacked 2x2 plates, faithful DAG), "
           "cube-plates-2x2x3 (three stacked 2x2 plates, 12 cubes -- use --parts 12), "
           "cuboid-tower (jenga-style: 2 laid cuboids per level, levels rotated 90deg, up to 8), "
           "car (cuboid body + 4 sideways cylinder wheels, 5), "
           "car-mini (car scaled to w=0.03: wheels fit ACROSS the two-finger gap), "
           "slot-pin (base + cantilever bar + fastening pin -- the --staged-mates testbed), "
           "peg-hole (compound blind socket + top-down cylindrical peg, 2 -- concavity smoke "
           "test; use --grasp-sampler), peg-stool (compound seat with three real downward "
           "sockets + cylindrical legs, 4; use --grasp-sampler and a vacuum gripper), "
           "peg-stool-4 (the same stool with FOUR cylindrical legs, 5), "
           "peg-stool-cuboid (the peg-stool with SQUARE cuboid legs, 4), "
           "peg-stool-cuboid-4 (the four-legged stool with SQUARE cuboid legs, 5), and "
           "the four -tight kinds (same four stools, peg size derived from the socket hole "
           "so the nominal clearance is 1mm instead of ~5mm; a square peg's CORNER gap is "
           "the binding clearance). All stool variants: use --grasp-sampler and a vacuum "
           "gripper; the clearance is a constructor parameter (PegStoolAssembly peg_clearance). "
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
           "placement FACE (0-5, 5=upright default), then exit. "
           "E.g. --test-place-ik a1:obj_1:0:1:0.",
  )
  g_ik.add_argument(
      "--real-ik", action=argparse.BooleanOptionalAction, default=True,
      help="Run the search with real-robot IK in the feasibility loop (RobotIK), and "
           "replay the plan with the actual arm poses instead of floating-EE stand-ins. "
           "Default ON (2026-07-29 defaults change); --no-real-ik restores floating EEs.",
  )
  g_ik.add_argument(
      "--no-ik-seeding", action="store_false", dest="ik_seeding",
      help="With --real-ik: use analytic closed-form IK (ssik) on every fixed-target skill "
           "(pick/place/assemble-onto-table, incl. braced): exact per-arm reachability rejects "
           "(replacing the relaxed precheck), collision-probed branch combos, and warm starts "
           "for the KOMO. Calibrated arms only (ur5/ur10 two-finger); others keep the precheck.",
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
           "pinch block (opposing surface pairs within the finger aperture) -- then "
           "farthest-point DOWNSAMPLE to --grasp-sampler-keep per shape. The precursor to a "
           "proper mesh grasp generator: everything downstream (EE collision filter, wrench "
           "pricing, IK, sim, export) consumes the sampled DB unchanged via grasp ids. "
           "Incompatible with --dense-grasps/--grasp-translations. Deterministic: a private "
           "RNG seeded from (--grasp-sampler-seed, shape), never the global stream.",
  )
  g_ik.add_argument(
      "--grasp-sampler-keep", type=int, default=12, metavar="N",
      help="--grasp-sampler: grasps kept per part shape after downsampling (default 12 -- "
           "matches the enumerator's 6 faces x 2 rolls search load). Split ~half suction-"
           "style, ~half antipodal.",
  )
  g_ik.add_argument(
      "--grasp-sampler-candidates", type=int, default=200, metavar="M",
      help="--grasp-sampler: raw surface samples per shape before downsampling (default 200).",
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
      "--grasp-sampler-scenes", action="store_true",
      help="--grasp-sampler: SCENE-AWARE sampling (opt-in). Every raw candidate is tagged "
           "with two floating-EE verdicts -- feasible at the part's INITIAL pose (the PICK "
           "set) and feasible at the part's ASSEMBLED pose (the ASSEMBLE set) -- and the "
           "keep budget splits into per-set quotas, so the DB provably contains a pick-able "
           "grasp, an assemble-able grasp and free-space (handover) grasps for every part "
           "instead of hoping the uniform cloud covers all three. The DB becomes per PART "
           "INSTANCE (identical legs filter differently). Deterministic; saved with the "
           "plan -- it rebinds grasp ids.",
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
      "--no-handovers", dest="handovers", action="store_false",
      help="Disable the add_grasp/handover primitives (regrasp-in-air). On by default; "
           "turning it off gives a leaner reference search (no second-grasp branching).",
  )
  g_search.add_argument(
      "--no-place", dest="allow_place", action="store_false",
      help="Forbid setting a held assembly back on the table: the search must hold it and "
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
      help="THE SINGLE RECIPE: adaptive-width milestones (DAG modules + width-2 window "
           "fills on width-1 failure) under a hard deadline (default 300s), then bounded "
           "improvement (PEA*, memory-safe) with the remaining budget; PEA-greedy net if "
           "milestones itself was cut. Always returns the best plan found within SEC; "
           "never OOMs; result <= min(milestone cost, weight x optimal) when the "
           "improvement pass completes. Use --weight inf for greedy fills (fastest).",
  )
  g_search.add_argument(
      "--no-lazy-fills", action="store_false", dest="lazy_fills",
      help="Disable LAZY-VALIDATED milestone fills (default on): fills search with the "
           "cheap gate only and run real IK on candidate subplan edges (LazySP hybrid, "
           "IK failures blacklisted + rerun) -- the measured fix for module fills "
           "spending minutes on failed IK binds at pop.",
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
      "--no-goal-on-table", action="store_false", dest="goal_on_table",
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
           "than M. Only the acting hand/target and each holder's own payload are exempt "
           "(plus part/part and table pairs). The SAME gate runs during search and every "
           "later witness/fusion/funnel solve, so post-processing is never stricter than "
           "search. Default 0.005m; 0 disables.",
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
      "--quiet-rai", action="store_true",
      help="Drop rai's libccd/pairCollision WARNING spam ('called MPR penetration for non "
           "intersecting meshes', 'but GJK says intersection'). rai prints those from C++ "
           "straight to the file descriptor, so a Python redirect cannot see them -- this "
           "splices a filter onto fd 1/2. Matches the SOURCE FILE, never the word WARNING, so "
           "real diagnostics survive. Equivalent: export RAI_QUIET=1.",
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
      "--useful-grasps", action="store_true",
      help="SPEC 6.2 (reduced): compute each part's MATE-VIABLE grasp set U[p] (collision-"
           "free at the op x press-transmitting per kind, wrench-aware) and (a) ORDER every "
           "pick/handover candidate U-first -- the transit gradient -- and (b) price a held "
           "non-U grasp at +1 in the milestone fill h (the forced regrasp, admissible). "
           "Ordering + pricing only; nothing is removed -- completeness untouched. Default off.",
  )
  g_diag.add_argument(
      "--fill-touch-all", action="store_true",
      help="DEBUG: let a milestone fill act on ANY loose part (move a blocking scatter part "
           "aside), not just the ideal+the one being added. Big fill-search blow-up; use only "
           "to test whether the ideal-only restriction is what makes a fill fail.",
  )
  g_phys.add_argument(
      "--mixed-wrench-h", action="store_true",
      help="With mixed --grippers + --wrench: build the PER-KIND wrench heuristic (facts + "
           "signature components evaluate each grasp's kind mask -- admissible, force-aware). "
           "PRECISE but SLOW (class split + kind LPs: ~35k->340k abstract states, ~2s->75s+ "
           "build on chair-8) and measured NOT to help the pathological weak-vacuum config. "
           "Default OFF = the fast topological fallback (wrench facts skipped under mixed). "
           "Turn on only when the search backtracks on force-feasibility the h can't see.",
  )
  g_search.add_argument(
      "--dominance", action="store_true",
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
      "--skill", type=float, nargs="?", const=0.05, default=0.0, metavar="DELTA",
      help="Shorthand: set ALL approach/departure funnels (--pre-pick/--pre-place/"
           "--pre-add-grasp/--pre-insertion/--post-release) to DELTA metres (default 0.05). "
           "An explicit per-skill flag overrides this for that skill. The individual funnels "
           "already default to 0.05; this is mainly a convenient common retune.",
  )
  g_skills.add_argument(
      "--pre-insertion", type=float, nargs="?", const=0.05, default=0.05, metavar="DELTA",
      help="PRE-INSERTION funnel (EXPORT/REPLAY only; the SEARCH is unchanged): split every "
           "`assemble` into a GOTO to the funnel-mouth pose T_pre (the mate witness backed off "
           "DELTA metres along the mate axis, default 0.05) followed by the REAL mate at the "
           "seated witness. The motion planner then connects the free-space approach to T_pre "
           "and only the short T_pre->seated seating motion is contact-rich -- on the mr_bench "
           "SKILLS branch that second task becomes the insertion skill. The mate pose, wrench "
           "gate, and plan are all untouched. Default 0.05 since 2026-07-29 (FRAGILITY 2.2: "
           "without it the held part arrives laterally, already in contact); "
           "0 = single un-split assemble.",
  )
  g_skills.add_argument(
      "--pre-pick", type=float, nargs="?", const=0.05, default=0.05, metavar="DELTA",
      help="Like --pre-insertion, for PICK (EXPORT/REPLAY only): split each pick into a GOTO to "
           "the pre-grasp pose (gripper retracted DELTA m along its -approach axis, default "
           "0.05) + the real grasp at the pick witness. Default 0.05; 0 disables.",
  )
  g_skills.add_argument(
      "--pre-place", type=float, nargs="?", const=0.05, default=0.05, metavar="DELTA",
      help="Like --pre-insertion, for PLACE (EXPORT/REPLAY only): split each place into a GOTO "
           "to the pre-place pose (object lifted DELTA m in +z off its rest pose, default 0.05) "
           "+ the real set-down at the place witness. Default 0.05; 0 disables.",
  )
  g_skills.add_argument(
      "--pre-add-grasp", type=float, nargs="?", const=0.05, default=0.05, metavar="DELTA",
      help="Like --pre-pick, for ADD_GRASP (the brace/handover reach; EXPORT/REPLAY only): split "
           "each add_grasp into a GOTO to the pre-grasp (gripper retracted DELTA m along its "
           "-approach axis, default 0.05) + the real brace reach. Default 0.05; 0 disables.",
  )
  g_skills.add_argument(
      "--post-release", type=float, nargs="?", const=0.05, default=0.05, metavar="DELTA",
      help="POST-RELEASE departure funnel (EXPORT/REPLAY only): after the detach task, add a "
           "GOTO that retracts the released gripper DELTA m along its -approach axis while "
           "the endpoint IK pins every other arm at the release keyframe. BiRRT remains "
           "unconstrained between endpoints. Default 0.05; 0 disables.",
  )
  g_skills.add_argument(
      "--fuse-pre-skill", action="store_true", default=None,
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
  g_skills.add_argument(
      "--no-fuse-pre-skill", dest="fuse_pre_skill", action="store_false", default=None,
      help="Force the pre-skill pin OFF (it defaults ON when a --pre-insertion/--pre-add-grasp "
           "split is present).",
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
           "mr_bench validates every arm. Note idle arms are at HOME in the witness scene.",
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
           "`python3 sim.py run <plan.json> --v2 --mp-path FILE` -- pair it with --save "
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
           "tolerance/--mp-robot-margin/--mp-margin-ramp/--mp-table-margin values override "
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
           "2026-07-29; --no-mp-optimize stops at the first solution. Improves path cost; "
           "uses the tree-extension shim.",
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
      "--mp-runtime", type=float, default=30.0, metavar="SEC",
      help="With --motion-plan: BiRRT search budget in seconds (default 30 since 2026-07-29). "
           "Bounds ONLY the tree search, NOT shortcutting.",
  )
  g_mp.add_argument(
      "--mp-no-shortcut", action="store_true",
      help="With --motion-plan: skip the shortcutting post-process (fastest to a raw path).",
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
      help="With --mp-robot-margin: relax the margin between the TASK'S OWN robots over the last "
           "M metres of END-EFFECTOR travel to their goal (0 = drop it outright for that pair; "
           "default 0.1 since 2026-07-29). "
           "A handover needs its two arms close but not close to the third, so the margin is "
           "dropped only for the participating pair, and only as they approach. Keyed on EE "
           "DISTANCE, not path position: a sampling planner's predicate must be a pure function "
           "of the state, so 'the last 10%% of the path' is not expressible -- distance-to-goal "
           "is, and means the same thing.",
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
      help="RNG seed for numpy + rai (KOMO IK), for reproducible runs. Default: 0.",
  )
  g_output.add_argument(
      "--load-plan", metavar="FILE", default=None,
      help="Read the ACTIONS from a saved plan JSON and skip the search, then run the normal "
           "post-passes and export on them (--optimize-witnesses, --fuse-cograsp, "
           "--motion-plan). Use this to apply a witness pass to a plan you already trust: a "
           "saved config does NOT carry the solver settings, so re-solving with --config gives "
           "a DIFFERENT plan. Pair it with --config <the same file> so the scene matches.",
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
  # Two-pass layering for --config: parse once to see if a config FILE was named; if so, load
  # it as the parser's DEFAULTS and re-parse so unspecified flags take the file's value while
  # any explicit command-line flag still wins (CLI > file > built-in default). Without --config
  # this is exactly `build_parser().parse_args(argv)` -- one pass, byte-identical to before.
  parser = build_parser()
  _warn_duplicate_flags(parser, argv)
  args = parser.parse_args(argv)
  if getattr(args, "config", None):
    from config import RunConfig
    parser.set_defaults(**RunConfig.from_file(args.config).to_dict())
    args = parser.parse_args(argv)                          # file defaults + CLI overrides
  return args


# fields persisted by --save so replay.py can reconstruct the EXACT scene (see plan_io.py).
_SAVE_CONFIG_FIELDS = (
    "assembly", "parts", "num_robots", "vacuum", "floating_ee", "robot", "robot_types",
    "scatter_spacing", "perturb_start_pose", "seed", "wrench", "grasp_model", "insert_force",
    "curved_torque_scale", "insertion_noise", "insertion_offset", "inserter_check", "real_ik",
    "bracing", "robust_mode", "dense_grasps", "random_init", "part_float", "inhand_reorient",
    "pre_insertion", "pre_pick", "pre_place", "pre_add_grasp", "post_release", "staged_mates",
    "allow_curved_grasps", "grippers", "grasp_compliance",
    "vacuum_scale", "two_finger_scale", "vacuum_engage_force",
    "vacuum_engage_friction", "tf_close_torque", "dominance",
    "table_mu",
    "useful_grasps", "brace_resting",
    "ignore_gravity",                                    # weightless stability: a plan searched
    # weightless contains releases/lateral mates that are UNSTABLE under gravity, so a rebuild
    # without it cannot reproduce the actions (user-hit: sim StopIteration on a cross-12 plan)
    "mixed_wrench_h", "strict_collisions", "witness_robot_collisions", "plain_box",
    "stable_rest_poses",                                 # placement ids index the per-group
    # placement table, so this changes the ACTION SPACE, not only the geometry
    # SUCCESSOR-AFFECTING and previously UNSAVED: a plan solved with any of these replays
    # against a DIFFERENT successor set than the search that made it, and the first symptom is
    # optimize_witnesses*/replay failing with "cannot reproduce action ... from the successor
    # set". min_sep and place_cells change the place cells, the rest change the action space.
    "strict_mate_collisions", "always_contact", "ik_accuracy", "mate_accuracy", "grasp_accuracy",
    "mate_eq_weight", "grasp_eq_weight",
    "place_eq_weight",                                   # a place is the DATUM later mates are
    # measured against, so its residual is not local (measured: body 3.91mm high, wheels then
    # mated onto it at 0.01-0.23mm -- perfectly seated on a body in the wrong place)
    "grasp_translations",                                # grasp ids index the DB
    "grasp_sampler", "grasp_sampler_keep",               # --grasp-sampler: sampled ids only
    "grasp_sampler_candidates", "grasp_sampler_seed",    # replay under identical settings
    "grasp_sampler_margin", "grasp_sampler_pad_support",
    "grasp_sampler_wrap",                                 # curvature-fair support metric (None = legacy)
    "grasp_sampler_scenes",                              # scene-aware set split (rebinds ids)
    "tight_fingers",                                     # changes what the grasp DB filters out
    "ik_seeding",                                        # off => witness IK fails on re-picks/
    # reorienting places, and mp_export then exports the arm's PREVIOUS config as the goal
    "pre_pick", "pre_place", "pre_add_grasp", "post_release", # each SPLITS an action into tasks:
    # unsaved, a replay/sim rebuild has fewer tasks than the --mp-path it loads, and the path
    # indexes tasks by integer id. pre_insertion was already saved; these three were not.
    "witness_margin", "witness_draw",                    # robust-keyframe gate + ladder
    "witness_sets", "witness_set_spread",                # multi-witness goal sets
    # draw: both change which witnesses ship, so replays/reuse must match
    "ban_actions",                                       # shapes the PLAN (provenance +
    # hoist/witness walks must see the same successor filter the search used)
    # which witnesses ship, so replays/reuse must match
    "optimize_witnesses", "witness_branch_budget",       # WHICH witness optimizer ran: a
    "witness_branch_metric", "witness_reconfig_thresh",  # different one yields a different
    # None-pattern, and build_tasks SKIPS a task whose witness is None -- so the rebuilt mode
    # chain stops matching the saved --mp-path ("mode alignment failed: N unmatched").
    "hoist_releases",                                    # rewrites the PLAN (before --save, so
    # the saved actions are already hoisted) -- recorded so a rebuild can tell what it is replaying
    "witness_clearance", "handovers", "allow_place",
    "mp_sequence",                                       # depgraph vs sequence = a different
    # task GRAPH; the sim rebuilt SEQUENCE unconditionally while main defaults to depgraph
    "mp_robot_margin", "mp_margin_ramp",                 # clearance the path was planned with
    "mp_table_margin",                                   # carried-part-vs-table transit clearance
    "mp_seed",                                           # reproducible BiRRT draw (None = unseeded)
    "mp_fake_clock",                                     # fake deterministic clock: runtime becomes
    # a CONSULTATION budget (bit-exact seeded replay)
    "min_sep", "place_cells", "prefer_center", "brace_for_assembly_only", "lru_robots",
    "anon_robots", "declutter", "por",
    # LAYOUT: these feed add_disassembled_obj, so without them the replay/sim rebuilds a
    # DIFFERENT initial scatter than the plan was solved in (FRAGILITY 1.8).
    "grid_init", "grid_gap",
)


def save_plan(path, cfg, plan, witnesses=None):
  from plan_io import write_plan
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
  write_plan(path, config, plan, witnesses=witnesses)
  print(f"[save] wrote {len(plan)}-action plan + scene config"
        + (" + witnesses" if witnesses is not None else "") + f" to {path}")


# Witness REUSE guard: a persisted witness block is only valid if every setting that shaped
# it still holds. All of these are in _SAVE_CONFIG_FIELDS, so files that carry witnesses also
# carry the values; a key absent from an (older) saved config is treated as a mismatch.
_WITNESS_FIELDS = (
    "optimize_witnesses", "witness_branch_budget", "witness_branch_metric",
    "witness_reconfig_thresh", "witness_clearance", "witness_robot_collisions",
    "strict_collisions", "mate_eq_weight", "place_eq_weight", "mate_accuracy",
    "grasp_accuracy", "grasp_eq_weight",
    "ik_accuracy", "ik_seeding", "seed", "bracing", "handovers", "allow_place",
    "tight_fingers", "part_float", "hoist_releases", "witness_margin", "witness_draw",
    "witness_sets", "witness_set_spread",
    "grasp_sampler", "grasp_sampler_keep", "grasp_sampler_candidates", "grasp_sampler_seed",
    "grasp_sampler_margin", "grasp_sampler_pad_support", "grasp_sampler_wrap",
    "grasp_sampler_scenes")


def _witnesses_reusable(saved_cfg, cfg):
  # a key MISSING from the saved config means the plan predates that flag -- it was
  # produced with the flag at its DEFAULT, so compare against the RunConfig default
  # (the single source of defaults) instead of declaring a mismatch (a bare `k not in
  # saved_cfg` silently re-solved the witnesses of EVERY older plan file each time a new
  # witness-affecting flag landed).
  from config import RunConfig
  # These two defaults became non-zero after saved witness blocks already existed.  Missing
  # keys therefore mean the historical OFF values, not today's fresh-run defaults.
  legacy = {"witness_margin": None, "witness_sets": 0}
  bad = [k for k in _WITNESS_FIELDS
         if saved_cfg.get(k, legacy.get(k, RunConfig.default_for(k))) != getattr(cfg, k, None)]
  if bad:
    print("[witness-reuse] settings differ from the saved witnesses on: "
          + ", ".join(bad) + " -- re-solving instead of reusing.")
  return not bad

def _dispatch_solve(cfg, C, assembly, robots, robot_ik):
  # Select the search engine from the cfg flags and run it; returns the plan (or None).
  # PURE dispatch: no global-state changes here (all set-up ran before this is called), so
  # the engine choice reads as one flat if/elif instead of burying run() under it.
  if cfg.ladder_lns is not None:
    plan = solve_assembly_problem_ladder(
        C, assembly, robots=robots, time_budget=cfg.ladder_lns, robot_ik=robot_ik,
        weight=cfg.weight, wrench=cfg.wrench, handovers=cfg.handovers,
        allow_place=cfg.allow_place, bracing=cfg.bracing,
        geo_facts=cfg.geo_facts, sig_collapsed_h=cfg.sig_collapsed_h,
        labeled_h=cfg.labeled_h, collapsed_h=cfg.collapsed_h,
        grasp_collapsed_h=cfg.grasp_collapsed_h, lns=True, rwa_proof=False,
        lns_object=not cfg.lns_segments)
  elif cfg.ladder is not None:
    plan = solve_assembly_problem_ladder(
        C, assembly, robots=robots, time_budget=cfg.ladder, robot_ik=robot_ik,
        weight=cfg.weight, wrench=cfg.wrench, handovers=cfg.handovers,
        allow_place=cfg.allow_place, bracing=cfg.bracing,
        geo_facts=cfg.geo_facts, sig_collapsed_h=cfg.sig_collapsed_h,
        labeled_h=cfg.labeled_h, collapsed_h=cfg.collapsed_h,
        grasp_collapsed_h=cfg.grasp_collapsed_h)
  elif cfg.milestones_anytime is not None:
    plan = solve_assembly_problem_milestones_anytime(
        C, assembly, robots=robots, time_budget=cfg.milestones_anytime,
        robot_ik=robot_ik, weight=cfg.weight, wrench=cfg.wrench,
        handovers=cfg.handovers, allow_place=cfg.allow_place,
        bracing=cfg.bracing, geo_facts=cfg.geo_facts,
        sig_collapsed_h=cfg.sig_collapsed_h, labeled_h=cfg.labeled_h,
        collapsed_h=cfg.collapsed_h, grasp_collapsed_h=cfg.grasp_collapsed_h,
        fill_window=cfg.fill_window)
  elif cfg.bounded or cfg.dfbnb:
    plan = solve_assembly_problem_bounded(
        C, assembly, robots=robots, robot_ik=robot_ik, weight=cfg.weight,
        wrench=cfg.wrench, handovers=cfg.handovers, allow_place=cfg.allow_place,
        bracing=cfg.bracing, labeled_h=cfg.labeled_h, collapsed_h=cfg.collapsed_h,
        grasp_collapsed_h=cfg.grasp_collapsed_h or not (cfg.labeled_h or cfg.collapsed_h),
        stats_every=cfg.search_stats, geo_facts=cfg.geo_facts,
        sig_collapsed_h=cfg.sig_collapsed_h, beam_width=cfg.beam,
        dfbnb=cfg.dfbnb, pea=cfg.pea, pathmax=cfg.pathmax,
        corridor_h=cfg.corridor_h)
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
                                             wrench=cfg.wrench, handovers=cfg.handovers,
                                             allow_place=cfg.allow_place,
                                             bracing=cfg.bracing,
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
        wrench=cfg.wrench, handovers=cfg.handovers, allow_place=cfg.allow_place,
        bracing=cfg.bracing, geo_facts=cfg.geo_facts,
        sig_collapsed_h=cfg.sig_collapsed_h, fill_window=cfg.fill_window,
        lazy_fills=cfg.lazy_fills)
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
  four assignments in the load branch unconditionally restored the export and silently made
  e.g. ``--mp-collision-tolerance 0.1`` ineffective.

  Returns ``(values, sources)`` where sources are suitable for the effective-settings log.
  ``explicit=None`` is the library-call case and deliberately keeps the saved artifact exact.
  """
  explicit = set(explicit or ())
  specs = (
      ("collision_tolerance", "mp_collision_tolerance", float),
      ("robot_robot_margin", "mp_robot_margin", float),
      ("robot_margin_ramp", "mp_margin_ramp", float),
      ("table_margin", "mp_table_margin", float),
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


def run(cfg, explicit=None):
  # cfg: RunConfig -- the single typed configuration object for the whole run. Every setting
  # below reads a cfg field; the thin main() wrapper builds cfg from the parsed CLI args.
  # explicit: set of dests the caller put on the command line (main()'s argv scan); None for
  # library callers. The dead-flag warnings below fire only for explicit flags.
  if cfg.optimize_witnesses == "off":          # 'floor' is the default; 'off' is the escape
    cfg.optimize_witnesses = None
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
  from problem import set_sampled_grasps, clear_sampled_scene_ctx
  if cfg.grasp_sampler_scenes and not cfg.grasp_sampler:
    raise SystemExit("--grasp-sampler-scenes requires --grasp-sampler")
  # a previous in-process run's scene filter must not leak into THIS run's assembly
  # definition (define() queries the grasp DB, which raises under a stale part-less ctx).
  clear_sampled_scene_ctx()
  if cfg.grasp_sampler:
    if cfg.dense_grasps or cfg.grasp_translations:
      # dense/translations are FACE-ENUMERATOR variant families and several id-arithmetic
      # sites assume their block layout (g % nb collapses, nb-capped fill phases) -- with
      # sampled ids those silently corrupt the grasp identity space.
      raise SystemExit("--grasp-sampler is incompatible with --dense-grasps / "
                       "--grasp-translations (face-enumerator variant families).")
    set_sampled_grasps({"seed": cfg.grasp_sampler_seed,
                        "candidates": cfg.grasp_sampler_candidates,
                        "keep": cfg.grasp_sampler_keep,
                        "margin": cfg.grasp_sampler_margin,
                        "pad_support": cfg.grasp_sampler_pad_support,
                        "wrap": cfg.grasp_sampler_wrap})
    print(f"[flag] --grasp-sampler ACTIVE: mesh-sampled grasp DB (trimesh surface sampling; "
          f"{cfg.grasp_sampler_candidates} candidates -> {cfg.grasp_sampler_keep} kept per "
          f"shape via farthest-point downsampling; suction + antipodal blocks; seed "
          f"{cfg.grasp_sampler_seed}). Grasp ids index the SAMPLED list -- plans/witnesses "
          f"only replay under the same sampler settings (saved with the plan).")
  else:
    set_sampled_grasps(None)
  if getattr(cfg, "stable_rest_poses", False):
    from problem import set_stable_rest_poses
    set_stable_rest_poses(True)
    print("[flag] --stable-rest-poses ACTIVE: placements come from the shape's TRUE stable "
          "rest orientations (convex hull + mass-weighted centre of mass), not the six "
          "axis-aligned faces. Placement id 5 still means 'as anchored'. Grasp ids are "
          "unaffected; PLACEMENT ids rebind, so plans do not cross the flag.")
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
  if cfg.tight_fingers:
    from env import set_tight_fingers
    set_tight_fingers(True)
    print("[flag] --tight-fingers ACTIVE: finger COLLISION capsules shrunk to "
          "robotiq_tight.g's radius (20mm -> 10mm, shifted 8mm so the grip surface holds). "
          "The fat proxy reaches 32.9mm past the tool centre -- it is what pokes out the far "
          "side of small parts and what forces --part-float.")
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
  if cfg.useful_grasps:
    from problem import set_useful_grasps_enabled
    set_useful_grasps_enabled(True)
    print("[flag] --useful-grasps ACTIVE: mate-viable grasp sets ordered first + fill-h "
          "regrasp pricing (spec 6.2).")
  if cfg.fill_touch_all:
    from solver import set_fill_touch_all
    set_fill_touch_all(True)
    print("[flag] --fill-touch-all: fills may move any loose part (debug; slow).")
  if cfg.mixed_wrench_h:
    from solver import set_mixed_wrench_h
    set_mixed_wrench_h(True)
    print("[flag] --mixed-wrench-h ACTIVE: per-kind wrench heuristic (precise, slower build).")
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
    wrench.PARAMS["grasp"]["asymmetric"]["two_finger"]["T_close"] = float(cfg.tf_close_torque)
    print(f"[flag] --tf-close-torque ACTIVE: two-finger closing-axis torque cap = "
          f"{cfg.tf_close_torque:g} Nm (was 0.4). Models a pinch that rotates out of hand.")
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
          "over only."
          + ("" if cfg.wrench else " NOTE: needs --wrench (the per-robot kind map only "
             "flows under it); without it the forbid has NO effect."))
  if cfg.ignore_gravity:
    print("[flag] --ignore-gravity ACTIVE: parts weightless in the wrench check "
          "(only insertion/applied loads must be resisted).")
  set_goal(on_table=cfg.goal_on_table)
  if cfg.goal_on_table:
    print("[flag] --goal-on-table ACTIVE: the finished assembly must rest on the table "
          "and be released (hands-off).")

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

  if cfg.ladder_lns is not None:
    print(f"[flag] --ladder-lns ACTIVE: budget {cfg.ladder_lns:g}s (k=1 -> LNS sweeps "
          f"-> RWA*-under-bound proof).")
  if cfg.ladder is not None:
    print(f"[flag] --ladder ACTIVE: budget {cfg.ladder:g}s (k=1 -> k=2 -> w=1 proof "
          f"rung, monotone incumbent bound).")
  if cfg.milestones_anytime is not None:
    print(f"[flag] --milestones-anytime ACTIVE: budget {cfg.milestones_anytime:g}s "
          f"(adaptive-width milestones -> bounded improvement -> PEA-greedy net).")
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
  C = add_disassembled_obj(C, assembly, perturb_start_pose=cfg.perturb_start_pose,
                           spacing=cfg.scatter_spacing, random_init=cfg.random_init,
                           float_height=cfg.part_float, min_sep=cfg.min_sep,
                           grid=cfg.grid_init, grid_gap=cfg.grid_gap)

  if cfg.grasp_sampler_scenes:
    from problem import install_sampled_scene_ctx
    install_sampled_scene_ctx(C, assembly, robots=get_robots(C))
    print("[flag] --grasp-sampler-scenes ACTIVE: every sampled candidate is tagged with "
          "floating-EE verdicts at its INITIAL pose (PICK set) and ASSEMBLED pose "
          "(ASSEMBLE set); the keep budget splits into per-set quotas, so the DB provably "
          "contains pick-able, assemble-able and free-space (handover) grasps per part. "
          "Grasp ids are per PART INSTANCE.")

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
    visualize_pick_ik(part=part, grasp_id=int(grasp), robot=robot, C=C, assembly=assembly)
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

  if cfg.viz_grasps or cfg.viz_rest_poses:
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
    robot_ik = RobotIK(C, robots) if cfg.real_ik else None
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
            "(search + witness opt); idle arms are at HOME in the IK scene.")
    if cfg.witness_margin is not None and robot_ik is not None:
      # ONE clearance contract, installed BEFORE search. The old implementation enabled this
      # only after a discrete plan existed, so witness optimization could reject every IK the
      # search itself had accepted -- exactly the contract violation this flag is meant to
      # prevent. Search, persisted witnesses, fusion and pre/post-skill solves now all call
      # RobotIK._clearance_ok with the same value and pair classification.
      robot_ik.clearance_reject = float(cfg.witness_margin)
      print(f"[flag] --witness-margin {cfg.witness_margin*1000:.0f}mm ACTIVE IN SEARCH + "
            "WITNESS PASSES: every accepted IK clears the same hard non-intended-pair gate "
            "(arm/part, inter-robot and non-adjacent self-collision); later passes may rank "
            "or diversify but cannot strengthen search feasibility.")
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
    if cfg.skill > 0.0:
      # shorthand: set every funnel not explicitly overridden on this command. Checking the
      # CLI presence rather than `<= 0` matters now that funnels default ON.
      for _f in ("pre_insertion", "pre_pick", "pre_place", "pre_add_grasp", "post_release"):
        if not explicit or _f not in explicit:
          setattr(cfg, _f, cfg.skill)
      print(f"[flag] --skill {cfg.skill:g} ACTIVE: enabling all pre-skill splits (pick, place, "
            "add_grasp, assemble) as skills.")
    if robot_ik is not None and cfg.pre_insertion > 0.0:
      robot_ik.pre_insertion_delta = cfg.pre_insertion
      print(f"[flag] --pre-insertion ACTIVE: MP export splits each assemble into a goto to the "
            f"funnel mouth ({cfg.pre_insertion:g}m back along the mate axis) + the real mate at "
            "the seated witness. Search/plan/wrench unchanged.")
    if robot_ik is not None and cfg.pre_pick > 0.0:
      robot_ik.pre_pick_delta = cfg.pre_pick
      print(f"[flag] --pre-pick ACTIVE: MP export splits each pick into a goto to the pre-grasp "
            f"({cfg.pre_pick:g}m gripper retract) + the real grasp. Search/plan unchanged.")
    if robot_ik is not None and cfg.pre_place > 0.0:
      robot_ik.pre_place_delta = cfg.pre_place
      print(f"[flag] --pre-place ACTIVE: MP export splits each place into a goto to the pre-place "
            f"({cfg.pre_place:g}m lift) + the real set-down. Search/plan unchanged.")
    if robot_ik is not None and cfg.pre_add_grasp > 0.0:
      robot_ik.pre_add_grasp_delta = cfg.pre_add_grasp
      print(f"[flag] --pre-add-grasp ACTIVE: MP export splits each add_grasp into a goto to the "
            f"pre-grasp ({cfg.pre_add_grasp:g}m gripper retract) + the real brace reach. "
            "Search/plan unchanged.")
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
    if cfg.load_plan:
      # RE-RUN THE POST-PASSES ON A PLAN YOU ALREADY TRUST -- no search.
      # A saved plan's `config` block does NOT persist the solver settings (no weight, no
      # heuristic, no lazy, no anytime budget, no --min-sep), so `--config saved.json` does not
      # reproduce the run that wrote it: MEASURED on car-mini, a re-solve with guessed flags gave
      # a 29-action plan that executes 19-21/29 where the saved 25-action plan runs 17/17. That
      # makes a fresh solve the wrong way to apply a WITNESS pass (--optimize-witnesses,
      # --fuse-cograsp) or to re-export a --motion-plan path: it silently changes the plan the
      # result is about. Loading the actions keeps the discrete plan fixed and re-runs only what
      # was asked for. The scene still comes from --config/CLI, so it must MATCH the saved plan.
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
      if not (cfg.motion_plan and cfg.mp_export_load) and _new_m > _old_m + 1e-12:
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
      try:
        plan = _dispatch_solve(cfg, C, assembly, robots, robot_ik)
      except IKFailureLogDone:
        print(f"[ik-failures] window/cap reached -- search stopped for debugging "
              f"({len(fail_rec.steps)} failures collected).")
        plan = None
    if robot_ik is not None and cfg.strict_collisions is not None:
      print(f"[strict-collisions] vetoed {robot_ik.strict_rejects} dirty KOMO witnesses "
            f"(> {cfg.strict_collisions:g}m total penetration).")
    if robot_ik is not None and robot_ik.clearance_reject is not None:
      print(f"[ik-clearance] vetoed {robot_ik.margin_rejects} IK candidate(s) below the "
            f"shared {robot_ik.clearance_reject*1000:.1f}mm search/witness margin.")
      rows = sorted(robot_ik.margin_reject_pairs.items(), key=lambda kv: kv[1], reverse=True)
      for (reason, fa, fb), count in rows[:5]:
        print(f"  [ik-clearance] {reason:11s} {fa} ~ {fb}: {count} veto(es)")
    if robot_ik is not None and cfg.mate_accuracy is not None:
      if getattr(cfg, "grasp_accuracy", None) is not None:
        print(f"[grasp-accuracy] vetoed {robot_ik.grasp_rejects} off-pose grasps "
              f"(> {cfg.grasp_accuracy:g}m tool residual).")
      print(f"[mate-accuracy] vetoed {robot_ik.mate_rejects} under-seated mates "
            f"(> {cfg.mate_accuracy:g}m residual).")
    if fail_rec is not None and fail_rec.steps:
      print(f"[ik-failures] {len(fail_rec.steps)} failed binds recorded -- serving in "
            f"viser (Stop in the GUI to continue to the plan replay).")
      fail_rec.serve(port=cfg.viser_port)
    if plan is not None:
      print(f"plan ({len(plan)} actions):")
      for step in plan:
        print("  ", step)
      # A saved plan already contains every post-pass that ran before save_plan. Reapplying
      # the non-idempotent hoist on --load-plan can move releases a SECOND time, so the motion
      # path is then for a different action order than the JSON which sim.py reloads. The
      # saved true value documents how the plan was made; it is not a request to mutate it on
      # every load. An explicit CLI --hoist-releases remains the opt-in way to apply the pass
      # to an older/unhoisted file.
      if _should_hoist_plan(cfg, explicit):
        # BEFORE save/witnesses/utilization/motion-plan, so all downstream consumers see the
        # hoisted plan (witnesses are solved for the hoisted states).
        from solver import hoist_releases
        plan = hoist_releases(C, assembly, plan, robots,
                              gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
                              handovers=cfg.handovers, allow_place=cfg.allow_place,
                              bracing=cfg.bracing)
      if cfg.save is not None:
        save_plan(cfg.save, cfg, plan)
      opt_witnesses = None
      witness_alts = None                                    # --witness-sets alternatives
      if (plan is not None and cfg.optimize_witnesses is not None
          and not (cfg.motion_plan and cfg.mp_export_load)):
        if robot_ik is None:
          print("[witopt] --optimize-witnesses needs --real-ik; skipping.")
        else:
          # WITNESS PERSISTENCE: a loaded plan file may carry the optimized keyframes, making
          # this stage a pure function of the file -- no witfloor re-solve, no possibility of
          # witness drift between stages. Reused only when every witness-shaping setting
          # matches the saved config; --refresh-witnesses forces the re-solve.
          if cfg.load_plan and not cfg.refresh_witnesses and _witnesses_reusable(_saved_cfg, cfg):
            from plan_io import read_witnesses, read_witness_sets
            opt_witnesses = read_witnesses(cfg.load_plan,
                                           list(robot_ik.C.getJointNames()), len(plan))
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
          if opt_witnesses is None and cfg.optimize_witnesses == "floor":
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
                witness_sets=cfg.witness_sets, set_spread=cfg.witness_set_spread)
            witness_alts = _rep.get("wit_alts") or None
          elif opt_witnesses is None:
            opt_witnesses, _rep = optimize_witnesses(
                C, assembly, plan, robots, robot_ik, cost=cfg.optimize_witnesses,
                gripper_kind=wrench_kind_arg(C, robots) if cfg.wrench else None,
                handovers=cfg.handovers, allow_place=cfg.allow_place,
                bracing=cfg.bracing, clear_weight=cfg.witness_clearance,
                robot_collisions=cfg.witness_robot_collisions)
      if cfg.save is not None and plan is not None and opt_witnesses is not None:
        # RE-save with the optimized witnesses attached (the first save above is deliberately
        # EARLY so a witness-pass crash cannot lose the plan). With this block downstream
        # stages (main --load-plan --motion-plan, sim.py run) reuse the exact keyframes
        # instead of re-solving them.
        from plan_io import witness_block
        save_plan(cfg.save, cfg, plan,
                  witnesses=witness_block(list(robot_ik.C.getJointNames()), opt_witnesses,
                                          alts=witness_alts))
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
          import mp_export as MP
          gk = wrench_kind_arg(C, robots) if cfg.wrench else None
          _loaded_export = None
          _export_bracing = cfg.bracing
          _export_depgraph = not cfg.mp_sequence
          _export_collision_tolerance = cfg.mp_collision_tolerance
          _export_robot_margin = cfg.mp_robot_margin
          _export_margin_ramp = cfg.mp_margin_ramp
          _export_table_margin = cfg.mp_table_margin
          _export_table_float = cfg.part_float if cfg.mp_table_margin else 0.0
          mp_pre = None
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
            _export_table_margin = _clearance["table_margin"]
            _export_table_float = _clearance["table_float"]
            robot_ik.pre_insertion_delta = float(
                _es.get("pre_insertion", robot_ik.pre_insertion_delta))
            robot_ik.pre_pick_delta = float(_es.get("pre_pick", robot_ik.pre_pick_delta))
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
                  f"table_margin={_export_table_margin:g} "
                  f"[{_clearance_source['table_margin']}], "
                  f"table_float={_export_table_float:g} "
                  f"[{_clearance_source['table_float']}].")
          else:
            # witnesses: the optimized set if --optimize-witnesses was given, else RAW
            # (sweeps=0). This is the ONE-TIME export path; replay loads its final artifact.
            mp_wit = opt_witnesses
            if mp_wit is None:
              mp_wit, _ = optimize_witnesses(
                  C, assembly, plan, robots, robot_ik, cost="ee", sweeps=0,
                  gripper_kind=gk, handovers=cfg.handovers, allow_place=cfg.allow_place,
                  bracing=cfg.bracing, verbose=False,
                  robot_collisions=cfg.witness_robot_collisions)
            if cfg.fuse_cograsp:
              from solver import fuse_cograsp_witnesses
              print("[flag] --fuse-cograsp ACTIVE: re-solving each (add_grasp, assemble) "
                    "witness pair as ONE 2-phase KOMO. add_grasp shares the assembly's "
                    "funnel-mouth rendezvous at the keyframe level; the ordinary motion "
                    "planner remains free to detour between endpoints.")
              mp_wit, _frep = fuse_cograsp_witnesses(
                  C, assembly, plan, mp_wit, robots, robot_ik, gripper_kind=gk,
                  handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing)
              _cograsp_pre = dict(_frep.get("pre_witnesses") or {})
          # LAST GATE: re-check the witnesses we are about to SHIP. Every earlier gate ran at
          # SOLVE time, and the object that leaves here is not the object that was solved --
          # optimize_witnesses* emits a splice and fuse_cograsp re-solves the pair again. A
          # dirty mate keyframe must fail HERE, named, not surface as an unexplained motion-
          # planner rejection three stages later.
          if (_loaded_export is None and cfg.strict_mate_collisions
              and robot_ik.penetration_reject is not None):
            from solver import validate_export_witnesses
            mp_wit, _bad = validate_export_witnesses(
                C, assembly, plan, mp_wit, robots, robot_ik, gripper_kind=gk,
                handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing)
            if _bad:
              print(f"[export-gate] {len(_bad)} assemble witness(es) STILL penetrate after "
                    f"falling back to the search's own witness -- the motion planner will "
                    f"reject these keyframes:")
              for _i, _act, _p, _ in _bad:
                print(f"    step {_i} {_act[:4]}: {_p * 1000:.2f}mm "
                      f"(gate {robot_ik.penetration_reject * 1000:.3f}mm)")
            else:
              print("[export-gate] every exported assemble keyframe is collision-free "
                    "(only the intended mate contact exempt).")
          # AFTER fusing: this pass never re-solves a SKILL witness, so a fused pair survives
          # untouched -- it only replaces the pre-* GOTO keyframes the exporter would solve.
          if _loaded_export is None:
            mp_pre = None
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
                handovers=cfg.handovers, allow_place=cfg.allow_place, bracing=cfg.bracing)
          # The generic pre-skill pass knows how to pin an ordinary funnel, but the cograsp
          # pass solves the stronger three-arm rendezvous. Its accepted keyframes therefore
          # win for the paired assemble indices.
          if _loaded_export is None and _cograsp_pre:
            mp_pre = dict(mp_pre or {})
            mp_pre.update(_cograsp_pre)
            print(f"[cograsp-chain] using {len(_cograsp_pre)} fused add_grasp/pre-assemble "
                  "rendezvous keyframe(s).")
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
                                 table_margin=_export_table_margin,
                                 table_float=_export_table_float,
                                 witness_sets=witness_alts,
                                 resolved_pre_witnesses=_resolved_pre,
                                 resolve_missing_pre=(_loaded_export is None),
                                 task_records=(None if _loaded_export is None else
                                               _loaded_export["tasks"]),
                                 saved_step_names=(None if _loaded_export is None else
                                                   _loaded_export["step_names"]),
                                 saved_sequence_names=(None if _loaded_export is None else
                                                       _loaded_export["sequence_names"]))
          print(f"[motion-plan] {'DEPGRAPH' if _export_depgraph else 'SEQUENCE'} problem, "
                f"{len(prob.tasks)} tasks")
          if _loaded_export is not None:
            _want_sig = _loaded_export.get("task_signature")
            _have_sig = MP.task_signature(prob)
            if _want_sig is not None and _have_sig != _want_sig:
              raise ValueError(
                  f"final motion export task signature changed: saved {_want_sig}, rebuilt "
                  f"{_have_sig}; refusing to replay against different task semantics")
          _export_settings = {
              "bracing": bool(_export_bracing), "depgraph": bool(_export_depgraph),
              "collision_tolerance": float(_export_collision_tolerance),
              "robot_robot_margin": float(_export_robot_margin or 0.0),
              "robot_margin_ramp": float(_export_margin_ramp or 0.0),
              "table_margin": float(_export_table_margin or 0.0),
              "table_float": float(_export_table_float or 0.0),
              "pre_insertion": float(robot_ik.pre_insertion_delta),
              "pre_pick": float(robot_ik.pre_pick_delta),
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
            import os as _os, sys as _sys
            MP._ensure_mrbench()
            _sys.path.insert(0, _os.path.join(_os.path.dirname(MP._MRBENCH_SRC), "scripts"))
            from show_problems import visualize_modes
            print(f"[motion-plan] serving MODES on http://localhost:{cfg.viser_port}")
            visualize_modes(prob, use_viser=True)
            return
          mp_path, _ = MP.run_birrt(
              prob, runtime=cfg.mp_runtime, shortcut=not cfg.mp_no_shortcut,
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
              MP.save_path(cfg.mp_save, mp_path, problem=prob,
                           mp_export_bundle=_mp_export_bundle)
              print(f"[motion-plan] saved path -> {cfg.mp_save} (load in sim with "
                    f"`sim.py run <plan.json> --v2 --mp-path {cfg.mp_save}`)")
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
  return run(RunConfig.from_namespace(args), explicit=explicit)


if __name__ == "__main__":
  main()
