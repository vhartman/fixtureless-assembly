import argparse

import numpy as np

from env import *
from assembly import *

def parse_args():
  parser = argparse.ArgumentParser(
      description="Spawn a multi-robot UR10 environment."
  )
  parser.add_argument(
      "--num-robots", type=int, default=4,
      help="Number of robots to spawn (max 6). Default: 4",
  )
  parser.add_argument(
      "--floating-ee", action="store_true",
      help="Use free-floating end-effectors instead of the full arm.",
  )
  parser.add_argument(
      "--vacuum", action="store_true",
      help="Use the vacuum end-effector instead of the two-finger gripper.",
  )
  parser.add_argument(
      "--robot", choices=["ur10", "ur10_limited", "ur5"], default="ur10",
      help="Robot arm for all robots: ur10 (default), ur10_limited (ur10 with a tighter "
           "shoulder-lift/elbow workspace), or ur5. With --floating-ee the arm is "
           "irrelevant (a generic floating gripper); it matters for the real arm (no --floating-ee "
           "/ --real-ik) -- note the ur5 has a much shorter reach (~0.85 m vs ~1.3 m).",
  )
  parser.add_argument(
      "--robot-types", default=None,
      help="Advanced: comma-free per-slot override of the robot type (defaults to --robot for "
           "every slot). Rarely needed.",
  )
  parser.add_argument(
      "--perturb-start-pose", action="store_true",
      help="Randomly perturb the object start poses (kept collision free).",
  )
  parser.add_argument(
      "--part-float", type=float, default=0.015, metavar="M",
      help="z offset of the scattered parts above the table top (metres, default 0.015 -- "
           "the historical float). Propagates through placement anchors, so every later "
           "set-down keeps the same hover.",
  )
  parser.add_argument(
      "--random-init", type=float, nargs="?", const=0.45, default=None, metavar="HALF",
      help="FULLY randomize the initial part positions: uniform x/y in the central square "
           "[-HALF, +HALF]^2 between the robots (default 0.45 when given bare) plus a random "
           "yaw, rejection-sampled collision-free. Seeded via --seed (reproducible). "
           "Overrides the scatter line; --perturb-start-pose is ignored with it.",
  )
  parser.add_argument(
      "--scatter-spacing", type=float, default=1.0 / 7.0, metavar="M",
      help="x-pitch of the scatter line in metres (default 1/7 ~ 0.143, the historical "
           "layout). For analysing how much search/IK cost is scene density.",
  )
  parser.add_argument(
      "--no-view", dest="view", action="store_false",
      help="Do not open the interactive viewer.",
  )
  parser.add_argument(
      "--disp-final", dest="disp_final", action="store_true",
      help="Show the final desired result.",
  )
  parser.add_argument(
      "--show-init", action="store_true",
      help="Open a blocking viewer on the INITIAL scene (after scatter/--random-init) before "
           "anything else runs -- close the window to continue (e.g. into --solve). For "
           "checking layouts, e.g. what --random-init sampled.",
  )
  parser.add_argument(
      "--show-successors", action="store_true",
      help="Visualize a few successor states of the initial search state, then exit.",
  )
  parser.add_argument(
      "--parts", type=int, default=7,
      help="Number of parts in the assembly (clamped per type). Default: 7.",
  )
  parser.add_argument(
      "--assembly",
      choices=["cross", "cube", "arch", "arch-seq", "cube-plates", "cube-plates-2x2x3",
               "cuboid-tower", "car", "car-mini"],
      default="cross",
      help="Assembly to build: cross (cube cross, 7), cube (2x2x2 grid, 8), "
           "arch (two pillars + lintel, 7 -- known-flawed star model), "
           "arch-seq (arch shape built as a sequential chain, 7), "
           "cube-plates (2x2x2 cube as two stacked 2x2 plates, faithful DAG), "
           "cube-plates-2x2x3 (three stacked 2x2 plates, 12 cubes -- use --parts 12), "
           "cuboid-tower (jenga-style: 2 laid cuboids per level, levels rotated 90deg, up to 8), "
           "car (cuboid body + 4 sideways cylinder wheels, 5), "
           "car-mini (car scaled to w=0.03: wheels fit ACROSS the two-finger gap). "
           "Default: cross.",
  )
  parser.add_argument(
      "--solve", action="store_true",
      help="Run the A* assembly search and print the plan.",
  )
  parser.add_argument(
      "--canonical", action="store_true",
      help="Break robot symmetry in the search (lowest-index free robot). TEST-ONLY: "
           "unsound once robots have fixed bases -- emits a warning.",
  )
  parser.add_argument(
      "--weight", type=float, default=1.0,
      help="Weighted-A* factor f=g+w*h: 1=optimal, >1 faster & bounded-suboptimal, "
           "inf=greedy best-first. Use a large value (e.g. 5 or inf) for fast solutions.",
  )
  parser.add_argument(
      "--test-pick-ik", nargs="?", const="a1:obj_1:0", default=None,
      metavar="ROBOT:PART:GRASP",
      help="Pop up the real-robot pick IK for a (robot, part, grasp) and exit, e.g. "
           "--test-pick-ik a1:obj_1:0 (default a1:obj_1:0).",
  )
  parser.add_argument(
      "--test-assemble-ik", nargs="?", const="a1:obj_1:0", default=None,
      metavar="ROBOT:PART:GRASP",
      help="Pop up the real-robot assemble IK: robot holds PART at GRASP and mates it "
           "onto its manual parent, then exit, e.g. --test-assemble-ik a1:obj_1:0.",
  )
  parser.add_argument(
      "--test-assemble-held-ik", nargs="?", const="a1:obj_1:0:a2:1", default=None,
      metavar="ROBOT:PART:GRASP:TARGET_ROBOT:ONTO_GRASP",
      help="Like --test-assemble-ik but the target (manual parent) is HELD by TARGET_ROBOT "
           "at ONTO_GRASP, so both cubes are in the air and both arms co-optimize. "
           "E.g. --test-assemble-held-ik a1:obj_1:0:a2:1.",
  )
  parser.add_argument(
      "--test-place-ik", nargs="?", const="a1:obj_1:0:1:5", default=None,
      metavar="ROBOT:PART:GRASP:N_ASSEMBLED[:FACE]",
      help="Pop up the real-robot place IK: with N_ASSEMBLED parts already mated, ROBOT "
           "picks the subassembly by PART at GRASP and sets it on the table resting on "
           "placement FACE (0-5, 5=upright default), then exit. "
           "E.g. --test-place-ik a1:obj_1:0:1:0.",
  )
  parser.add_argument(
      "--real-ik", action="store_true",
      help="Run the search with real-robot IK in the feasibility loop (RobotIK), and "
           "replay the plan with the actual arm poses instead of floating-EE stand-ins.",
  )
  parser.add_argument(
      "--no-ik-seeding", action="store_false", dest="ik_seeding",
      help="With --real-ik: use analytic closed-form IK (ssik) on every fixed-target skill "
           "(pick/place/assemble-onto-table, incl. braced): exact per-arm reachability rejects "
           "(replacing the relaxed precheck), collision-probed branch combos, and warm starts "
           "for the KOMO. Calibrated arms only (ur5/ur10 two-finger); others keep the precheck.",
  )
  parser.add_argument(
      "--dense-grasps", action="store_true",
      help="TWO-FINGER only: add the +-45deg WRIST-TILTED variant of every face grasp -- the "
           "pads keep the SAME flat face pinch (rotation is about the finger-closing axis "
           "through the tool centre), only the wrist/approach swings into the diagonal "
           "'between the faces'. 3x the grasp candidates (more IK/table work) in exchange "
           "for approaches that clear obstacles a face-aligned wrist hits. Ignored with "
           "--vacuum (rotationally symmetric).",
  )
  parser.add_argument(
      "--ik-collision-gate", type=float, default=0.015, metavar="M",
      help="With --ik-seeding: penetration depth (metres) beyond which an exact analytic branch "
           "counts as DEEPLY colliding; a bind is rejected pre-KOMO only if EVERY branch combo "
           "is deep (arm-arm pairs use 2x the gate). Default 0.015 -- measured: the full solve "
           "rescues grazes up to ~0.010, never >=0.014. 0 = ANY penetration rejects "
           "(aggressive: loses grazing rescues). Negative = off (probe/ranking only).",
  )
  parser.add_argument(
      "--full-ik", action="store_true",
      help="DEBUG: disable the single-active-arm pick-IK optimisations (restrict-to-arm + "
           "strip inactive-robot collision). Uses the full multi-arm KOMO instead. Slower; "
           "for checking the optimisations against the reference.",
  )
  parser.add_argument(
      "--optimistic", action="store_true",
      help="Use the optimistic (Dogar-style) per-part search -- add one part at a time via "
           "its own small complete sub-search (reorient/hold/pick/mate/set-down), with exact "
           "fallback -- instead of the exact search. Much faster on the cases that blow up.",
  )
  parser.add_argument(
      "--milestones", action="store_true",
      help="MILESTONE planner: the labeled abstraction's backward (assembly-by-disassembly) "
           "solve picks the next part to add (wrench-informed ORDER, grasps assumed "
           "reachable); each milestone is filled in by a small per-part search "
           "(pick/handover/brace/mate/set-down); a failed fill-in blacklists that milestone "
           "and reroutes. Scales like --optimistic but with a SEARCHED part order, bracing "
           "support, and repair instead of an exact-search fallback.",
  )
  parser.add_argument(
      "--place-cells", type=int, nargs="?", const=4, default=0, metavar="N",
      help="EXPERIMENTAL: offer up to N extra TRANSLATED placement anchors per (group, face) "
           "near the table centre (default 4 when given bare), in addition to the classic "
           "scatter-anchor placement. Motivation: parts at the scatter extremes get set down "
           "where half the arms cannot reach; central cells make braced work reachable. Adds "
           "branching (~N x places).",
  )
  parser.add_argument(
      "--por", action="store_true",
      help="EXPERIMENTAL partial-order reduction in the exact search: among INDEPENDENT "
           "successor actions (different robots, disjoint parts) only the canonically sorted "
           "interleaving is expanded. KNOWN UNSOUND in general (feasibility is order-coupled "
           "through geometry) -- meant for spread scenes (--min-sep) with floating EEs where "
           "independence actually holds. Prints when active.",
  )
  parser.add_argument(
      "--min-sep", type=float, default=None, metavar="M",
      help="With --random-init: enforce a pairwise x/y separation of at least M metres "
           "between parts (e.g. 0.2), so part geometries/graspability provably cannot "
           "interact -- the --por independence assumption.",
  )
  parser.add_argument(
      "--prefer-center", action="store_true",
      help="With --place-cells: offer the CELL placements (table centre first) BEFORE the "
           "classic scatter-anchor placement -- at equal f the FIFO tie-break then tries "
           "central placements first. Pure successor ordering, no pruning.",
  )
  parser.add_argument(
      "--search-stats", type=float, nargs="?", const=10.0, default=None, metavar="SEC",
      help="Print periodic in-search statistics every SEC seconds (default 10 when given "
           "bare): expanded/claimed/frontier sizes and per-action-kind edge-check + failure "
           "counts; the same per-kind summary is printed with the final result. Exact "
           "search only.",
  )
  parser.add_argument(
      "--bounded", action="store_true",
      help="INCUMBENT-BOUNDED search: the milestone planner produces a fast incumbent plan, "
           "then the exact search runs with the incumbent's cost as a sound prune bound "
           "(g+h >= U dies; optimal prefixes always survive). Result <= min(incumbent, "
           "weight x optimal); with --weight 1 the phase-2 result is optimal. Combines the "
           "decomposed solver's speed with the exact search's guarantee.",
  )
  parser.add_argument(
      "--lazy", action="store_true",
      help="LAZY-VALIDATION exact search (LazySP-style): search the discrete graph "
           "optimistically (no per-edge cheap-collision/IK), then validate only the CANDIDATE "
           "PLAN's edges; a failed edge is blacklisted and the search reruns (heuristic built "
           "once, caches warm). Same completeness and plan validity as the exact search; "
           "wins when most eagerly-checked edges never end up in a plan.",
  )
  parser.add_argument(
      "--brace-for-assembly-only", action="store_true",
      help="EXPERIMENTAL (MEASURED HARMFUL, off by default): assume a brace (2nd grasp via "
           "add_grasp) is only needed to stabilise an ASSEMBLE, so a braced state (>=2 grippers "
           "on one group) may only assemble/release. Markovian and correct, BUT it backfires in "
           "the forward search -- braced states reached before the part is positioned become "
           "near-dead-ends the search thrashes on (car-mini fill branching exploded 20-50x). "
           "The domain fact is right; the clean form is an atomic add_grasp+assemble MACRO, not "
           "this filter. Kept for experimentation; prints when active.",
  )
  parser.add_argument(
      "--declutter", action="store_true",
      help="With --optimistic: allow each per-part step to RELOCATE blocking loose parts "
           "(stow them aside on demand) instead of only touching that part and the assembly. "
           "Removes the 'nothing else is in the way' assumption.",
  )
  parser.add_argument(
      "--wrench", action="store_true",
      help="Turn on the static-equilibrium (stability) model: support-gate assemble/release "
           "with the insertion reaction, so sideways mates on a resting assembly need a hold "
           "(mating becomes perpendicular unless braced). Default off (topological baseline).",
  )
  parser.add_argument(
      "--grasp-model", choices=["isotropic", "directional", "asymmetric"],
      default="isotropic",
      help="Grasp wrench-capability model (with --wrench): isotropic = the original "
           "direction-agnostic box (default); directional = weak-shear vacuum / weak-torque "
           "two-finger in the grasp frame; asymmetric = directional + push/pull asymmetry "
           "along the approach axis (pressing into the palm/cup is structural, pull-off is "
           "the weak suction/friction path).",
  )
  parser.add_argument(
      "--no-handovers", dest="handovers", action="store_false",
      help="Disable the add_grasp/handover primitives (regrasp-in-air). On by default; "
           "turning it off gives a leaner reference search (no second-grasp branching).",
  )
  parser.add_argument(
      "--no-place", dest="allow_place", action="store_false",
      help="Forbid setting a held assembly back on the table: the search must hold it and "
           "regrasp in the air (handovers become mandatory -- the Dogar hold-and-work "
           "regime). Meaningful for the exact search; the optimistic search needs to set down.",
  )
  parser.add_argument(
      "--anytime", type=float, nargs="?", const=60.0, default=None, metavar="SECONDS",
      help="Anytime mode (exact search): restarting weighted A* -- a fast suboptimal plan "
           "first, then keep improving it (and possibly prove optimality) until the time "
           "budget runs out. Optional value = budget in seconds (default 60).",
  )
  parser.add_argument(
      "--no-wrench-aware-h", dest="wrench_aware_h", action="store_false",
      help="With --wrench: build the heuristic (h_abs or --labeled-h) WITHOUT the wrench "
           "facts, for comparison. Default: wrench-aware whenever --wrench is on.",
  )
  parser.add_argument(
      "--bracing", action="store_true",
      help="Enable the add_grasp (co-grasp) primitive: a free robot adds a SECOND gripper to "
           "an already-held assembly and KEEPS it (unlike handover, which swaps). Needed for "
           "offset HORIZONTAL mates (e.g. car wheels), where one grasp cannot supply the "
           "yaw-cancelling couple so two braces are required. Off by default (adds branching). "
           "Exact search only (solve/anytime); the labeled-h abstraction models it when on.",
  )
  parser.add_argument(
      "--labeled-h", action="store_true",
      help="EXACT search only: use the grasp+face-LABELED disassembly abstraction as the "
           "heuristic (strictly tighter than the default abstraction; one-time build of "
           "seconds). Finite weights only (like the default abstraction).",
  )
  parser.add_argument(
      "--anon-robots", action="store_true",
      help="LATE BINDING: search with a robot-anonymous state key -- robot-permutation "
           "states merge, so adding (idle) robots does not blow up the search. Sound for "
           "interchangeable floating grippers; with --real-ik treat the plan as a candidate.",
  )
  parser.add_argument(
      "--insert-force", type=float, default=10.0, metavar="N",
      help="Insertion (mating) reaction magnitude in newtons (with --wrench). Default 10. "
           "Sets what binds: e.g. two-finger caps are shear 15 / pull 25 / push 100, so at "
           "10 N held side-mates are free, at 30 N only press-aligned grasps survive.",
  )
  parser.add_argument(
      "--ignore-gravity", action="store_true",
      help="With --wrench: WEIGHTLESS parts -- drop the gravity load from every stability "
           "check, so a hold must only resist the INSERTION reaction (and other applied "
           "loads), not part weight. Isolates insertion/bracing feasibility from gravity.",
  )
  parser.add_argument(
      "--curved-torque-scale", type=float, default=None, metavar="S",
      help="With --wrench: derate the TORQUE (couple) a CURVED grasp (cylinder side / sphere) "
           "can resist to S x that of a flat hold (force caps unchanged). E.g. 0.25. Models "
           "'a wheel grip pivots in the jaws'; off by default (flat and curved treated alike).",
  )
  parser.add_argument(
      "--insertion-noise", type=float, nargs="?", const=0.2, default=None, metavar="LATERAL",
      help="With --wrench: model MISALIGNED seating as a worst-case uncertainty BOX -- an "
           "angular tilt gives a shear up to LATERAL x insert_force transverse to the mate "
           "axis (default 0.2 when given bare), and the state must be stable at EVERY corner "
           "of the box (exact, by convexity -- no sampling). Adds an unavoidable shear/torque "
           "so a weak grasp holding far from the mate fails. Off by default.",
  )
  parser.add_argument(
      "--insertion-offset", type=float, default=0.0, metavar="M",
      help="With --insertion-noise: add the second (positional) box axis -- the axial press "
           "also lands off-CENTRE by up to M metres (a transverse couple = insert_force x M). "
           "0 (default) => a 4-corner lateral-only box; >0 => a 16-corner box.",
  )
  parser.add_argument(
      "--robust-mode", choices=("corners", "affine"), default="corners",
      help="How the insertion-uncertainty box is checked. 'corners' (default): EXACT -- every box "
           "corner must be balanceable, contacts free to rechoose their force split per "
           "realization. 'affine': CONSERVATIVE -- one fixed affine force-response policy (an "
           "open-loop / fixed-compliance grasp that cannot sense the seating error and "
           "re-optimize) must cover the whole box; a single LP, stricter, still sound.",
  )
  parser.add_argument(
      "--no-inserter-check", dest="inserter_check", action="store_false",
      help="With --wrench: skip the inserter-side check (whether the pressing arm's own "
           "grasp can transmit the seating reaction) -- equivalent to giving that grasp "
           "unbounded capability. The target-side stability check stays on.",
  )
  parser.add_argument(
      "--log-ik-failures", type=float, nargs="?", const=60.0, default=None,
      metavar="SECONDS",
      help="With --real-ik: record FAILED IK binds of the search (the skill's scene posed "
           "at the best failed attempt, labelled with action + solver residuals) and serve "
           "them in a viser browser after the solve (before the plan replay). Collection "
           "stops after SECONDS (default 60, measured from the first failure) or 300 "
           "entries, whichever first.",
  )
  parser.add_argument(
      "--collapsed-h", action="store_true",
      help="Use the IDENTITY-COLLAPSED labeled disassembly heuristic (loose held parts "
           "tracked WITHOUT grasp ids). Same admissibility/completeness guarantees as "
           "--labeled-h, weaker guidance, but the build scales in #robots (the fine "
           "version explodes at 4). Mutually exclusive with --labeled-h.",
  )
  parser.add_argument(
      "--part-order-bias", action="store_true",
      help="Plateau TIE-BREAK: among frontier nodes with equal f (and h), expand the one "
           "whose action touches the LOWEST part index first -- i.e. prefer building in the "
           "canonical part order. Pure ordering: the w-suboptimality bound and completeness "
           "are untouched; it only decides which equally-promising node to try first, which "
           "is what matters on the abstraction's large equal-h plateaus.",
  )
  parser.add_argument(
      "--grasp-collapsed-h", action="store_true",
      help="--collapsed-h PLUS the dense-grasp LABEL collapse: wrist-tilt variants fold onto "
           "their base grasp id in the abstraction (facts OR'd over variants -- a further "
           "relaxation, still admissible). THE heuristic to pair with --dense-grasps (raw "
           "labels explode the build ~3x per robot; measured 24x on cross-4/3R). Without "
           "--dense-grasps it behaves exactly like --collapsed-h.",
  )
  parser.add_argument(
      "--ladder-lns", type=float, nargs="?", const=900.0, default=None, metavar="SEC",
      help="THE LADDER v2: k=1 milestones (lazy fills) -> LNS sweeps over the incumbent "
           "(segment re-fills of width 1,2 bounded by old segment cost, cascade to "
           "reconvergence -- gradual strict improvements) -> RWA*-under-bound proof rung "
           "(w=5 passes: finds tighten U fast, any EXHAUSTED pass is the optimality "
           "proof; exhaustion is weight-neutral). k=2 milestone rung runs only as a "
           "feasibility fallback. --ladder is retained unchanged as the baseline.",
  )
  parser.add_argument(
      "--ladder", type=float, nargs="?", const=900.0, default=None, metavar="SEC",
      help="THE LADDER: unified anytime solver -- one commitment axis (fill window k), "
           "one monotone incumbent bound U. Rungs: k=1 milestones (lazy fills, bound "
           "propagated into fills), k=2 windows, then the k=n limit = bounded exact at "
           "WEIGHT 1 (PEA*) whose first goal pop is optimal-under-U and whose exhaustion "
           "PROVES U optimal. Budget split ~35/20/45%%. Default 900s.",
  )
  parser.add_argument(
      "--milestones-anytime", type=float, nargs="?", const=300.0, default=None,
      metavar="SEC",
      help="THE SINGLE RECIPE: adaptive-width milestones (DAG modules + width-2 window "
           "fills on width-1 failure) under a hard deadline (default 300s), then bounded "
           "improvement (PEA*, memory-safe) with the remaining budget; PEA-greedy net if "
           "milestones itself was cut. Always returns the best plan found within SEC; "
           "never OOMs; result <= min(milestone cost, weight x optimal) when the "
           "improvement pass completes. Use --weight inf for greedy fills (fastest).",
  )
  parser.add_argument(
      "--no-lazy-fills", action="store_false", dest="lazy_fills",
      help="Disable LAZY-VALIDATED milestone fills (default on): fills search with the "
           "cheap gate only and run real IK on candidate subplan edges (LazySP hybrid, "
           "IK failures blacklisted + rerun) -- the measured fix for module fills "
           "spending minutes on failed IK binds at pop.",
  )
  parser.add_argument(
      "--lns-segments", action="store_true",
      help="LNS neighborhood unit = SETTLED segments instead of the default OBJECT unit "
           "(k = consecutive mated parts). Baseline/ablation; segments coarsen unevenly "
           "under hold-through re-fills.",
  )
  parser.add_argument(
      "--fill-window", type=int, default=2, metavar="K",
      help="Milestone fill window escalation limit (default 2): on width-1 failure, try "
           "JOINT fills of 2..K parts (escalating only as each smaller width exhausts) "
           "before backjumping -- covers interleaved-glue plans, the milestone "
           "commitment's completeness gap; K=n would be the flat exact search. "
           "1 disables windows.",
  )
  parser.add_argument(
      "--corridor-h", action="store_true",
      help="With --bounded: cap the phase-2 heuristic's backward Dijkstra at the "
           "incumbent's cost U (beyond-cap states never pushed; miss => h=inf, sound "
           "under the bound). MEASURED INEFFECTIVE: the abstract spaces are WIDE at "
           "shallow depth (R=4 cross still >2M states at cap 35; 2x2x3 table unchanged "
           "at 204k), and the pruning is redundant with cost_bound + exact h values. "
           "Harmless; kept for depth-heavy instances. The real version of the idea needs "
           "a TWO-SIDED corridor (forward+backward abstract dist <= U) to cut width.",
  )
  parser.add_argument(
      "--pathmax", action="store_true",
      help="PATHMAX consistency repair in the exact engines: h(child) >= h(parent) - "
           "edge cost (triangle inequality => still admissible; all guarantees "
           "unchanged). MEASURED NEUTRAL: pathmax only RAISES h after too-fast drops -- "
           "it cannot touch the h-STALL steps, which are the actual f-bumps (sanities "
           "bit-identical; cube-plates-2x2x3 unchanged). Kept as a safe no-cost option.",
  )
  parser.add_argument(
      "--pea", action="store_true",
      help="PARTIAL-EXPANSION A* engine: identical guarantees and ordering to the exact "
           "search, but the open list holds only each node's current f-BAND -- successors "
           "above it are regenerated later instead of stored. Frontier memory / ~branching "
           "factor, for long horizons where the plain open list OOMs (compute-for-memory "
           "trade; succ/h are cached). Composes with --bounded (phase-2 engine).",
  )
  parser.add_argument(
      "--iter-bound", action="store_true",
      help="ITERATIVE COST-BOUND DEEPENING: run exact passes at cost_bound = h(s0)+1, +2, "
           "... until a plan appears -- the FIRST plan found is PROVEN OPTIMAL (each "
           "exhausted pass is a lower-bound proof; the inner --weight only orders "
           "exploration inside each pass). Per-pass memory = that bound's sublevel set, "
           "the minimum any optimality proof needs. Combine with --pea for the smallest "
           "footprint. Integer action costs assumed (step 1).",
  )
  parser.add_argument(
      "--beam", type=int, default=None, metavar="N",
      help="BEAM SEARCH engine (width N): keeps <= N states per depth level (memory-"
           "bounded). MEASURED WORSE THAN GREEDY on the long-horizon benchmark "
           "(cube-plates-2x2x3: greedy cost 33/63 exp vs beam-8 cost 43 at w=inf, 59 at "
           "w=5, beam-1 cost 116 -- greedy's GLOBAL frontier backs up where a beam must "
           "commit, and its frontier stays tiny anyway). Kept as the LAST-RESORT "
           "incumbent fallback in --bounded (fires only if greedy also fails) and for "
           "experimentation.",
  )
  parser.add_argument(
      "--dfbnb", action="store_true",
      help="DEPTH-FIRST BRANCH-AND-BOUND certificate engine (implies --bounded): linear "
           "memory + LRU transposition table, anytime. MEASURED SLOWER than the bounded "
           "A* it replaces (cube-plates-2x2x3, same bound 33: A* PROVES in 192s floating "
           "/ ~1000s real-ik; dfbnb exhausts its node budget unfinished at 350s / times "
           "out at 1500s -- DFS re-enters the interleaving lattice that A*'s claim-at-pop "
           "kills globally). Use only if the A* open list truly cannot fit memory.",
  )
  parser.add_argument(
      "--no-goal-on-table", action="store_false", dest="goal_on_table",
      help="GOAL: require the finished assembly to REST ON THE TABLE and be released "
           "(hands-off), not merely complete-in-air. Default goal is complete (held "
           "counts). Extensible to a specific target pose later (set_goal(pose=...)).",
  )
  parser.add_argument(
      "--optimize-witnesses", nargs="?", const="joint", default=None,
      choices=["joint", "ee", "dp-joint", "dp-ee"], metavar="MODE",
      help="POST-optimize the plan's IK witnesses (discrete plan FIXED): re-solve each IK "
           "action pulled toward the ACTIVE arm's previous configuration, minimizing "
           "per-step movement in 'joint' space (default) or 'ee' space (position + "
           "weighted orientation). dp-joint/dp-ee: LADDER-DP -- harvest multiple feasible "
           "candidates per step (assemble orientation biases, fresh re-solves, the greedy "
           "pass) and pick the globally optimal combination by per-robot Viterbi "
           "(DP >= greedy >= original by construction). Feasibility-safe (search "
           "witnesses kept on any failure); idle arms stay home-pinned as in the search. "
           "Requires --real-ik.",
  )
  parser.add_argument(
      "--witness-clearance", type=float, default=0.0, metavar="W",
      help="Per-step CLEARANCE weight for --optimize-witnesses (0=off, ~1.0 works). "
           "Among candidate witnesses, prefer configs that keep the active arm clear of "
           "the OTHER robots and the TABLE (parts excluded -- intended contact). Score = "
           "SUM over arm-link pairs of min(gap, 0.4m) via C.eval(FS.distance) (sum not "
           "min: the gripper is pinned near the object; the cap ignores already-far "
           "pairs). Traded against movement by this weight. Best with the dp-* modes.",
  )
  parser.add_argument(
      "--lru-robots", action="store_true",
      help="Order robots LEAST-RECENTLY-USED first in successor generation (utilization "
           "knob): the greedy/milestone searches take the first feasible successor, so the "
           "fixed roster order makes a1 win every tie and idles the rest. LRU alternation "
           "spreads actions across arms -- with the depgraph export that IS parallelism "
           "(different robots on independent parts stay unordered for the motion planner). "
           "Deterministic (no RNG, stable ties); pure successor ordering, search space "
           "unchanged. May cost a few extra actions (forced alternation).",
  )
  parser.add_argument(
      "--hoist-releases", action="store_true",
      help="POST-PASS after --solve (before witness optimization): bubble every `release` as "
           "EARLY in the plan as legal -- applicable per the real successor generator (wrench "
           "stability: a needed hold cannot release early), cheap-geometry-checked, and only "
           "where the swapped pair provably commutes (identical state after). Frees arms "
           "sooner (shorter robot chains in the schedule / depgraph, less idle-arm clutter "
           "for motion planning). Plan length and final state unchanged.",
  )
  parser.add_argument(
      "--utilization", action="store_true",
      help="After --solve, report the plan's SCHEDULED-MAKESPAN utilization: list-schedule "
           "the actions on their resource partial order (robots + touched parts -- the same "
           "order the depgraph export uses) and print makespan vs serial time, speedup, and "
           "per-robot busy%%. Durations = witness joint-distances when --optimize-witnesses "
           "ran, else 1/action. Report only; no behavior change.",
  )
  parser.add_argument(
      "--witness-robot-collisions", action="store_true",
      help="During --optimize-witnesses, keep the OTHER robots' collision geometry ON (the "
           "search strips it on solo steps for speed). Re-solved witnesses then AVOID the idle "
           "arms instead of passing through them -- important before motion planning, where "
           "mr_bench validates every arm. Note idle arms are at HOME in the witness scene.",
  )
  parser.add_argument(
      "--motion-plan", action="store_true",
      help="After --solve, EXPORT the plan (+ witnesses) to the mr_bench motion planner "
           "(mp_export) and run Bidirectional RRT* to a collision-free multi-robot path. "
           "Reuses THIS run's scene/witness flags (--assembly/--parts/--num-robots/--robot/"
           "--optimize-witnesses/--witness-clearance/--vacuum/--goal-on-table). Needs "
           "--real-ik (for the IK witnesses); if --optimize-witnesses is not given, RAW "
           "search witnesses are used. See the --mp-* flags for planner options.",
  )
  parser.add_argument(
      "--mp-sequence", action="store_true",
      help="With --motion-plan: export a FIXED-SEQUENCE problem (total order) instead of the "
           "default DEPENDENCY GRAPH (partial order, planner may reorder independent actions).",
  )
  parser.add_argument(
      "--mp-optimize", action="store_true",
      help="With --motion-plan: RRT* COST optimization until --mp-runtime (default: stop at "
           "the first solution). Improves path cost; uses the tree-extension shim.",
  )
  parser.add_argument(
      "--mp-runtime", type=float, default=60.0, metavar="SEC",
      help="With --motion-plan: BiRRT search budget in seconds (default: 60). Bounds ONLY the "
           "tree search, NOT shortcutting.",
  )
  parser.add_argument(
      "--mp-no-shortcut", action="store_true",
      help="With --motion-plan: skip the shortcutting post-process (fastest to a raw path).",
  )
  parser.add_argument(
      "--mp-shortcut-iters", type=int, default=1000, metavar="N",
      help="With --motion-plan: shortcut iteration cap (default 1000; NOT time-bounded).",
  )
  parser.add_argument(
      "--mp-shortcut-res", type=float, default=None, metavar="R",
      help="With --motion-plan: shortcut collision resolution (default: env collision_resolution, "
           "~10x faster than mr_bench's own 0.001 for identical quality).",
  )
  parser.add_argument(
      "--mp-viser", action="store_true",
      help="With --motion-plan: serve the found PATH in mr_bench's own viser viewer "
           "(display_path_viser: step slider + playback). Blocks. Port = --viser-port.",
  )
  parser.add_argument(
      "--mp-collision-view", action="store_true",
      help="With --mp-viser: show only PRIMITIVE (collision) shapes -- capsules/boxes/"
           "cylinders/ssBox -- and hide the file-loaded visual meshes "
           "(display_path_viser primitives_only). See exactly what the planner collides.",
  )
  parser.add_argument(
      "--mp-modes", action="store_true",
      help="With --motion-plan: instead of planning, step through the exported problem's MODES "
           "(each posed at its task goal keyframe) in mr_bench's own viser viewer "
           "(scripts/show_problems.py::visualize_modes). Good for debugging the export / "
           "spotting per-mode collisions. Blocks. Port = --viser-port.",
  )
  parser.add_argument(
      "--sig-collapsed-h", action="store_true",
      help="SIGNATURE grasp collapse + LABELED loose parts: grasp labels are equivalence "
           "classes under the fact signature (press capability, pick-face vector, contact "
           "patch, solo-hold, brace-side), and loose parts keep their class label -- the "
           "fine labeled-h's plateau-killing pick/press discrimination at collapsed-h "
           "scale. Facts OR over class members (relaxation -> admissible).",
  )
  parser.add_argument(
      "--sig-components", type=str, default=None, metavar="CSV",
      help="Ablation for --sig-collapsed-h: comma-separated subset of signature "
           "components (press,pickface,contact,solohold,braceside). Default: all.",
  )
  parser.add_argument(
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
  parser.add_argument(
      "--view-cheap-checks", action="store_true",
      help="Step through every (uncached) cheap collision check in a blocking C.view(True) "
           "window -- verdict, triggering action, and penetrating pairs in the overlay. "
           "Both FREE and COLLIDING results are shown; each distinct scene appears once. "
           "Use with small instances: the search blocks on every window.",
  )
  parser.add_argument(
      "--viser", action="store_true",
      help="Display via a viser web viewer (http://localhost:<port>) instead of the rai "
           "window: the whole replay is recorded and served with a step slider / playback "
           "controls (no per-step blocking windows).",
  )
  parser.add_argument(
      "--viser-port", type=int, default=8080,
      help="Port for the viser server (with --viser). Default: 8080.",
  )
  parser.add_argument(
      "--seed", type=int, default=0,
      help="RNG seed for numpy + rai (KOMO IK), for reproducible runs. Default: 0.",
  )
  parser.add_argument(
      "--save", metavar="PATH", default=None,
      help="After --solve, write the plan + the scene config (assembly, parts, robots, seed, "
           "wrench settings, ...) to PATH as JSON, so it can be replayed later without "
           "re-searching via the standalone `python replay.py PATH` script.",
  )
  return parser.parse_args()


# fields persisted by --save so replay.py can reconstruct the EXACT scene (see plan_io.py).
_SAVE_CONFIG_FIELDS = (
    "assembly", "parts", "num_robots", "vacuum", "floating_ee", "robot", "robot_types",
    "scatter_spacing", "perturb_start_pose", "seed", "wrench", "grasp_model", "insert_force",
    "curved_torque_scale", "insertion_noise", "insertion_offset", "inserter_check", "real_ik",
    "bracing", "robust_mode", "dense_grasps", "random_init", "part_float",
)


def save_plan(path, args, plan):
  from plan_io import write_plan
  config = {k: getattr(args, k) for k in _SAVE_CONFIG_FIELDS}
  write_plan(path, config, plan)
  print(f"[save] wrote {len(plan)}-action plan + scene config to {path}")

def main():
  args = parse_args()
  set_seed(args.seed)                           # reproducible numpy + rai (KOMO) RNG
  # grasp rolls are a property of the gripper: the vacuum is symmetric about its suction
  # axis (1 grasp per face); the two-finger has two in-plane orientations (90 deg apart).
  # --dense-grasps additionally enables the +-45deg WRIST-TILTED variants (same face pinch,
  # approach between the faces) -- see problem.set_dense_grasps.
  set_rolls_per_face(1 if args.vacuum else 2)
  set_dense_grasps(args.dense_grasps and not args.vacuum)
  set_brace_for_assembly_only(args.brace_for_assembly_only)
  set_extra_place_cells(args.place_cells)
  set_prefer_center(args.prefer_center)
  if args.place_cells:
    print(f"[flag] --place-cells ACTIVE: {args.place_cells} extra translated placement "
          f"anchors per (group, face)"
          + (", CENTRE-FIRST ordering" if args.prefer_center else "") + ".")
  if args.brace_for_assembly_only:
    print("[flag] --brace-for-assembly-only ACTIVE: a braced state (>=2 grippers on a group) "
          "may only assemble/release. Prunes brace branching; can miss plans that carry a "
          "brace across unrelated work (experimental).")
  if args.lru_robots:
    from problem import set_lru_robots
    set_lru_robots(True)
    print("[flag] --lru-robots ACTIVE: successor robots ordered least-recently-used first "
          "(deterministic tie-break; spreads actions across arms).")
  import wrench
  wrench.PARAMS["grasp_model"] = args.grasp_model  # capability model (used when --wrench)
  wrench.PARAMS["check_inserter"] = args.inserter_check
  wrench.PARAMS["insert_force"]["_default"] = args.insert_force
  wrench.PARAMS["curved_grasp_torque_scale"] = args.curved_torque_scale
  wrench.PARAMS["insertion_uncertainty"] = (
      None if args.insertion_noise is None
      else {"lateral": args.insertion_noise, "offset": args.insertion_offset})
  wrench.PARAMS["robust_mode"] = args.robust_mode
  wrench.PARAMS["ignore_gravity"] = args.ignore_gravity
  if args.ignore_gravity:
    print("[flag] --ignore-gravity ACTIVE: parts weightless in the wrench check "
          "(only insertion/applied loads must be resisted).")
  set_goal(on_table=args.goal_on_table)
  if args.goal_on_table:
    print("[flag] --goal-on-table ACTIVE: the finished assembly must rest on the table "
          "and be released (hands-off).")

  C = make_four_robot_env(
      num_robots=args.num_robots,
      floating_ee=args.floating_ee,
      vacuum=args.vacuum,
      robot_types=args.robot_types or args.robot,   # --robot for all; --robot-types overrides
  )

  assembly = make_assembly(args.assembly, args.parts)

  if args.ladder_lns is not None:
    print(f"[flag] --ladder-lns ACTIVE: budget {args.ladder_lns:g}s (k=1 -> LNS sweeps "
          f"-> RWA*-under-bound proof).")
  if args.ladder is not None:
    print(f"[flag] --ladder ACTIVE: budget {args.ladder:g}s (k=1 -> k=2 -> w=1 proof "
          f"rung, monotone incumbent bound).")
  if args.milestones_anytime is not None:
    print(f"[flag] --milestones-anytime ACTIVE: budget {args.milestones_anytime:g}s "
          f"(adaptive-width milestones -> bounded improvement -> PEA-greedy net).")
  if args.corridor_h:
    print("[flag] --corridor-h ACTIVE: phase-2 heuristic built only up to the incumbent "
          "bound (misses prune); requires --bounded.")
  if args.pathmax:
    print("[flag] --pathmax ACTIVE: h consistency repair (h_child >= h_parent - cost); "
          "admissible, fewer f-bumps/frontier sweeps.")
  if args.pea:
    print("[flag] --pea ACTIVE: partial-expansion A* engine (banded open list; same "
          "guarantees, frontier memory / ~branching factor).")
  if args.iter_bound:
    print("[flag] --iter-bound ACTIVE: iterative cost-bound deepening -- first plan found "
          "is proven optimal; --weight only orders exploration within passes.")
  if args.beam is not None:
    print(f"[flag] --beam ACTIVE: beam search, width {args.beam} (memory-bounded, no "
          f"quality bound{'; incumbent fallback for --bounded' if args.bounded or args.dfbnb else ''}).")
  if args.dfbnb:
    print("[flag] --dfbnb ACTIVE: depth-first branch-and-bound certificate phase "
          "(implies --bounded).")
  if args.sig_collapsed_h:
    if args.sig_components:
      set_sig_components([c.strip() for c in args.sig_components.split(",") if c.strip()])
    print("[flag] --sig-collapsed-h ACTIVE: signature grasp-equivalence classes + labeled "
          "loose parts (components: "
          + (args.sig_components or "press,pickface,contact,solohold,braceside") + ").")
  if args.geo_facts:
    print("[flag] --geo-facts ACTIVE: labeled-abstraction backward relay gated on "
          "floating-gripper coexistence (gripper-vs-assembly/gripper/table); sound "
          "necessary condition, h stays admissible.")
  if args.por:
    print("[flag] --por ACTIVE: partial-order reduction over independent successor actions "
          "(EXPERIMENTAL, unsound when part/arm geometries interact -- use --min-sep).")
  C = add_disassembled_obj(C, assembly, perturb_start_pose=args.perturb_start_pose,
                           spacing=args.scatter_spacing, random_init=args.random_init,
                           float_height=args.part_float, min_sep=args.min_sep)


  if args.show_init:
    C.view(True, "initial state (close to continue)")

  # per-skill IK test popups run on THE ACTUAL CONFIGURED SCENE built above (robot type,
  # scatter/--random-init layout, assembly). Note the popups solve real-arm IK: with
  # --floating-ee the "arm" is the free-floating gripper itself.
  if args.test_pick_ik is not None:
    robot, part, grasp = args.test_pick_ik.split(":")
    visualize_pick_ik(part=part, grasp_id=int(grasp), robot=robot, C=C, assembly=assembly)
    return

  if args.test_assemble_ik is not None:
    robot, part, grasp = args.test_assemble_ik.split(":")
    visualize_assemble_ik(part=part, grasp_id=int(grasp), robot=robot, C=C,
                          assembly=assembly)
    return

  if args.test_place_ik is not None:
    fields = args.test_place_ik.split(":")
    robot, part, grasp, n_asm = fields[:4]
    face = int(fields[4]) if len(fields) > 4 else 5
    visualize_place_ik(part=part, grasp_id=int(grasp), robot=robot,
                       n_assembled=int(n_asm), face=face, C=C, assembly=assembly)
    return

  if args.test_assemble_held_ik is not None:
    robot, part, grasp, target_robot, onto_grasp = args.test_assemble_held_ik.split(":")
    visualize_assemble_ik(part=part, grasp_id=int(grasp), robot=robot,
                          target_robot=target_robot, onto_grasp=int(onto_grasp),
                          C=C, assembly=assembly)
    return

  if args.disp_final:
    display_final_assembly(C, assembly, [0, 0, 0.5])
    
  if args.show_successors:
    visualize_successor_states(C, assembly)
    return

  if args.solve:
    robots = get_robots(C)
    robot_ik = RobotIK(C, robots) if args.real_ik else None
    if robot_ik is not None and args.full_ik:
      robot_ik.single_arm_ik = False
      robot_ik.strip_inactive_collision = False
      robot_ik.reach_precheck = False
    if robot_ik is not None and args.ik_seeding:
      robot_ik.ik_seeding = True
      robot_ik.collision_reject_depth = (
          None if args.ik_collision_gate < 0 else args.ik_collision_gate)
    if robot_ik is not None and args.witness_robot_collisions:
      # keep the OTHER robots' collision ON in the SEARCH's IK too (not just the post-opt pass):
      # the search's witnesses then avoid the idle arms from the start. (optimize_witnesses gets
      # robot_collisions too; with this already set, that toggle is a consistent no-op.)
      robot_ik.strip_inactive_collision = False
      print("[flag] --witness-robot-collisions ACTIVE: other robots' collision kept ON in IK "
            "(search + witness opt); idle arms are at HOME in the IK scene.")
    if args.view_cheap_checks:
      set_cheap_check_debug(True)
    fail_rec = None
    if args.log_ik_failures is not None and robot_ik is not None:
      from viser_view import ViserRecorder
      fail_rec = ViserRecorder()
      robot_ik.fail_recorder = fail_rec
      robot_ik.fail_log_seconds = args.log_ik_failures
      robot_ik.fail_log_stop = True                # abort the search when the window closes
    try:
      if args.ladder_lns is not None:
        plan = solve_assembly_problem_ladder(
            C, assembly, robots=robots, time_budget=args.ladder_lns, robot_ik=robot_ik,
            weight=args.weight, wrench=args.wrench, handovers=args.handovers,
            allow_place=args.allow_place, bracing=args.bracing,
            geo_facts=args.geo_facts, sig_collapsed_h=args.sig_collapsed_h,
            labeled_h=args.labeled_h, collapsed_h=args.collapsed_h,
            grasp_collapsed_h=args.grasp_collapsed_h, lns=True, rwa_proof=False,
            lns_object=not args.lns_segments)
      elif args.ladder is not None:
        plan = solve_assembly_problem_ladder(
            C, assembly, robots=robots, time_budget=args.ladder, robot_ik=robot_ik,
            weight=args.weight, wrench=args.wrench, handovers=args.handovers,
            allow_place=args.allow_place, bracing=args.bracing,
            geo_facts=args.geo_facts, sig_collapsed_h=args.sig_collapsed_h,
            labeled_h=args.labeled_h, collapsed_h=args.collapsed_h,
            grasp_collapsed_h=args.grasp_collapsed_h)
      elif args.milestones_anytime is not None:
        plan = solve_assembly_problem_milestones_anytime(
            C, assembly, robots=robots, time_budget=args.milestones_anytime,
            robot_ik=robot_ik, weight=args.weight, wrench=args.wrench,
            handovers=args.handovers, allow_place=args.allow_place,
            bracing=args.bracing, geo_facts=args.geo_facts,
            sig_collapsed_h=args.sig_collapsed_h, labeled_h=args.labeled_h,
            collapsed_h=args.collapsed_h, grasp_collapsed_h=args.grasp_collapsed_h,
            fill_window=args.fill_window)
      elif args.bounded or args.dfbnb:
        plan = solve_assembly_problem_bounded(
            C, assembly, robots=robots, robot_ik=robot_ik, weight=args.weight,
            wrench=args.wrench, handovers=args.handovers, allow_place=args.allow_place,
            bracing=args.bracing, labeled_h=args.labeled_h, collapsed_h=args.collapsed_h,
            grasp_collapsed_h=args.grasp_collapsed_h or not (args.labeled_h or args.collapsed_h),
            stats_every=args.search_stats, geo_facts=args.geo_facts,
            sig_collapsed_h=args.sig_collapsed_h, beam_width=args.beam,
            dfbnb=args.dfbnb, pea=args.pea, pathmax=args.pathmax,
            corridor_h=args.corridor_h)
      elif args.lazy:
        plan = solve_assembly_problem_lazy(
            C, assembly, robots=robots, weight=args.weight, robot_ik=robot_ik,
            wrench=args.wrench, handovers=args.handovers, allow_place=args.allow_place,
            labeled_h=args.labeled_h, collapsed_h=args.collapsed_h,
            grasp_collapsed_h=args.grasp_collapsed_h,
            wrench_aware_h=args.wrench_aware_h, bracing=args.bracing,
            geo_facts=args.geo_facts, sig_collapsed_h=args.sig_collapsed_h)
      elif args.milestones:
        plan = solve_assembly_problem_milestones(
            C, assembly, robots=robots, robot_ik=robot_ik, weight=args.weight,
            wrench=args.wrench, handovers=args.handovers, allow_place=args.allow_place,
            bracing=args.bracing, geo_facts=args.geo_facts,
            sig_collapsed_h=args.sig_collapsed_h, fill_window=args.fill_window,
            lazy_fills=args.lazy_fills)
      elif args.optimistic and args.declutter:
        plan = solve_assembly_problem_optimistic_declutter(
            C, assembly, robots=robots, robot_ik=robot_ik, weight=args.weight,
            wrench=args.wrench, handovers=args.handovers, allow_place=args.allow_place)
      elif args.optimistic:
        plan = solve_assembly_problem_optimistic(C, assembly, robots=robots,
                                                 robot_ik=robot_ik, weight=args.weight,
                                                 wrench=args.wrench, handovers=args.handovers,
                                                 allow_place=args.allow_place,
                                                 bracing=args.bracing,
                                                 grasp_collapsed_h=args.grasp_collapsed_h,
                                                 collapsed_h=args.collapsed_h,
                                                 labeled_h=args.labeled_h,
                                                 sig_collapsed_h=args.sig_collapsed_h)
      elif args.anytime is not None:
        plan, _cost, _opt = solve_assembly_problem_anytime(
            C, assembly, robots=robots, time_budget=args.anytime, robot_ik=robot_ik,
            wrench=args.wrench, handovers=args.handovers, allow_place=args.allow_place,
            anonymous=args.anon_robots, labeled_h=args.labeled_h,
            collapsed_h=args.collapsed_h, grasp_collapsed_h=args.grasp_collapsed_h,
            wrench_aware_h=args.wrench_aware_h, bracing=args.bracing,
            part_order_bias=args.part_order_bias,
            sig_collapsed_h=args.sig_collapsed_h)
      else:
        plan = solve_assembly_problem(C, assembly, robots=robots, canonical=args.canonical,
                                      weight=args.weight, robot_ik=robot_ik, wrench=args.wrench,
                                      handovers=args.handovers, allow_place=args.allow_place,
                                      anonymous=args.anon_robots, labeled_h=args.labeled_h,
                                      collapsed_h=args.collapsed_h,
                                      grasp_collapsed_h=args.grasp_collapsed_h,
                                      wrench_aware_h=args.wrench_aware_h, bracing=args.bracing,
                                      part_order_bias=args.part_order_bias,
                                      stats_every=args.search_stats, por=args.por,
                                      geo_facts=args.geo_facts,
                                      sig_collapsed_h=args.sig_collapsed_h,
                                      beam_width=args.beam, pea=args.pea,
                                      iter_bound=args.iter_bound,
                                      pathmax=args.pathmax)
    except IKFailureLogDone:
      print(f"[ik-failures] window/cap reached -- search stopped for debugging "
            f"({len(fail_rec.steps)} failures collected).")
      plan = None
    if fail_rec is not None and fail_rec.steps:
      print(f"[ik-failures] {len(fail_rec.steps)} failed binds recorded -- serving in "
            f"viser (Stop in the GUI to continue to the plan replay).")
      fail_rec.serve(port=args.viser_port)
    if plan is not None:
      print(f"plan ({len(plan)} actions):")
      for step in plan:
        print("  ", step)
      if args.hoist_releases:
        # BEFORE save/witnesses/utilization/motion-plan, so all downstream consumers see the
        # hoisted plan (witnesses are solved for the hoisted states).
        from solver import hoist_releases
        plan = hoist_releases(C, assembly, plan, robots,
                              gripper_kind=gripper_kind(C) if args.wrench else None,
                              handovers=args.handovers, allow_place=args.allow_place,
                              bracing=args.bracing)
      if args.save is not None:
        save_plan(args.save, args, plan)
      opt_witnesses = None
      if plan is not None and args.optimize_witnesses is not None:
        if robot_ik is None:
          print("[witopt] --optimize-witnesses needs --real-ik; skipping.")
        else:
          opt_witnesses, _rep = optimize_witnesses(
              C, assembly, plan, robots, robot_ik, cost=args.optimize_witnesses,
              gripper_kind=gripper_kind(C) if args.wrench else None,
              handovers=args.handovers, allow_place=args.allow_place,
              bracing=args.bracing, clear_weight=args.witness_clearance,
              robot_collisions=args.witness_robot_collisions)
      if plan is not None and args.utilization:
        from solver import utilization_report
        utilization_report(C, assembly, plan, robots, witnesses=opt_witnesses,
                           gripper_kind=gripper_kind(C) if args.wrench else None,
                           handovers=args.handovers, allow_place=args.allow_place,
                           bracing=args.bracing)
      if plan is not None and args.motion_plan:
        if robot_ik is None:
          print("[motion-plan] needs --real-ik (for the IK witnesses); skipping.")
        else:
          import mp_export as MP
          gk = gripper_kind(C) if args.wrench else None
          # witnesses: the optimized set if --optimize-witnesses was given, else RAW (sweeps=0)
          mp_wit = opt_witnesses
          if mp_wit is None:
            mp_wit, _ = optimize_witnesses(
                C, assembly, plan, robots, robot_ik, cost="ee", sweeps=0,
                gripper_kind=gk, handovers=args.handovers, allow_place=args.allow_place,
                bracing=args.bracing, verbose=False,
                robot_collisions=args.witness_robot_collisions)
          prob = MP.make_problem(C, assembly, plan, mp_wit, robots, robot_ik, gk,
                                 bracing=args.bracing, depgraph=not args.mp_sequence)
          print(f"[motion-plan] {'SEQUENCE' if args.mp_sequence else 'DEPGRAPH'} problem, "
                f"{len(prob.tasks)} tasks")
          if args.mp_modes:
            import os as _os, sys as _sys
            MP._ensure_mrbench()
            _sys.path.insert(0, _os.path.join(_os.path.dirname(MP._MRBENCH_SRC), "scripts"))
            from show_problems import visualize_modes
            print(f"[motion-plan] serving MODES on http://localhost:{args.viser_port}")
            visualize_modes(prob, use_viser=True)
            return
          mp_path, _ = MP.run_birrt(
              prob, runtime=args.mp_runtime, shortcut=not args.mp_no_shortcut,
              shortcut_iters=args.mp_shortcut_iters, shortcut_resolution=args.mp_shortcut_res,
              optimize=args.mp_optimize)
          if mp_path:
            print(f"[motion-plan] path {len(mp_path)} states, "
                  f"reaches terminal={prob.is_terminal_mode(mp_path[-1].mode)}")
            if args.mp_viser:
              from multi_robot_multi_goal_planning.problems.util import interpolate_path
              disp = interpolate_path(mp_path, 0.05, kind="euclidean")
              print(f"[motion-plan] serving interpolated path ({len(disp)} states) on "
                    f"http://localhost:{args.viser_port}"
                    + (" (collision shapes only)" if args.mp_collision_view else ""))
              prob.display_path_viser(disp, port=args.viser_port,
                                      primitives_only=args.mp_collision_view)
          else:
            print("[motion-plan] NO path found")
      # --motion-plan owns the display (via --mp-viser); don't ALSO fire the search-replay
      # viz, which would open the BLOCKING rai viewer (args.view defaults True).
      if (args.view or args.viser) and not args.motion_plan:
        recorder = None
        if args.viser:
          from viser_view import ViserRecorder
          recorder = ViserRecorder()
        if args.real_ik:
          visualize_plan_ik(C, assembly, plan, robots=robots, robot_ik=robot_ik,
                            recorder=recorder, bracing=args.bracing,
                            witnesses=opt_witnesses)
        else:
          visualize_plan(C, assembly, plan, recorder=recorder, bracing=args.bracing)
        if recorder is not None:
          recorder.serve(port=args.viser_port)
    return

  if args.viser:
    from viser_view import ViserRecorder
    recorder = ViserRecorder()
    recorder.record(C, "scene")
    recorder.serve(port=args.viser_port)
  elif args.view:
    C.view(True)

if __name__ == "__main__":
  main()