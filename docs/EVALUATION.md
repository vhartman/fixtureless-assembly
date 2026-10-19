# Evaluation & TODO — fixtureless assembly paper

Consolidated status, baselines, ablations, Reviewer-2 attack list, and open work. Companion
to `PIPELINE.md` (how to run) and `assembly_planner_spec.md` §8 (deferred design). Honest
about what is DONE vs ASSUMED vs BROKEN. Last synced 2026-07-25.

## Thesis
Fixtureless multi-robot assembly: no jigs, so **static equilibrium of every intermediate**
is the planning constraint, and holds/braces/handovers/regrasps must be *reasoned about*, not
scripted. Contribution split: (a) the **fixtureless statics model** (per-mate wrench LP,
insertion reactions, bracing) and (b) the **search** (milestones + LNS, anytime→optimal).

## Claims → evidence (what each needs)
| Claim | Have | Missing / status |
|---|---|---|
| Fixtureless needs static-equilibrium reasoning | wrench LP; emergent brace/handover/hold-through | **a validated discriminator** — see ⚠ below |
| Emergent maneuvers from primitive support edits | plans exhibit them | count/qualitative study across the grid |
| Milestones+LNS: fast anytime, optimal on small | ladder, convergence curves | cost-vs-time plots vs baselines, gap table |
| Heterogeneous EEs handled admissibly | mixed-EE machinery, tests | ablation: uniform-EE planner fails/degrades |

## ⚠ The validation leg — current state (READ THIS)
- **Sim harness exists and works** (`sim.py`, MuJoCo): real-arm motion-plan execution;
  save/load motion plans; per-action failure taxonomy; scrub-able
  replay; visual UR meshes + collision capsules toggle. See `[[sim-validation-harness]]`.
- **BUT the earlier "clean discriminator" (cross-5 wrench-on succeeds / off topples) was an
  ARTIFACT** of a continuing-holder sim bug (now fixed). With the bug fixed, cross-5's
  topological plan is physically valid → **cross-5 does NOT discriminate**. A *real*
  discriminator (topological plan genuinely infeasible: single-holder TORQUE limit binding —
  car wheels / side-mates / heavier held groups) is **still TODO**. This is the single most
  important missing result. Don't quote the old cross-5 numbers.
- Physics-sim validation study (success-rate grid, wrench-on vs off) = **not yet run** (blocked
  partly on the discriminator above + the BiRRT-at-1e-6 sampling issue).

## Baselines
- **Dogar (ICRA'15)** — reduce via `--optimistic` (our own Dogar-style per-part solver, shared
  IK/collision/grasps → no "weak reimplementation" attack). Run 3 ways: optimistic+wrench-off
  (self-supporting assumption → returns statically-invalid plans on side/underside mates);
  optimistic+wrench-on; milestones+LNS. Reduction argument in spec §10.
- **Hartmann et al.** — that's you; position in related work, no head-to-head needed.
- **LGP (Toussaint)** — you judged it fails via optimization on the wrench-gated action set;
  state that explicitly (rai reviewers WILL ask "why not LGP").
- **Exact wA\*** (ours) — completeness/optimality reference on small instances.
- **PDDLStream-style TAMP** — optional "did you try the standard tool" checkbox on topological
  variants.

## Ablations (ranked by paper weight)
1. **Wrench tiers**: off / isotropic / directional / asymmetric / +noise / +compliance →
   plan differences (brace count, grasp-near-mate, EE assignment) + downstream sim validity.
2. **Bracing/handover off** → feasibility flips (car wheels / chair posts infeasible). Binary, cheap.
3. **Heuristic family**: h_abs / labeled / collapsed / sig-collapsed — build vs nodes vs
   quality, incl. the R=4 scaling story `[[labeled-h-r4-scaling]]`.
4. **Anytime anatomy**: milestones / +LNS / +proof rung — cost-vs-time with the proven-optimal line.
5. **Mixed vs uniform EEs**, incl. weak-vacuum stress.
6. **Scaling**: parts 3→12, robots 2→4.

## Reviewer-2 attack list (pre-empt each)
1. *"LP validity: quasi-static, point contacts, hand-set caps, no dynamics."* → sim-validation
   study + caps-sensitivity ablation (conclusions robust to ±2× params). **Force GENERATION is
   unmodeled** in BOTH planner and sim (`F_push`=100 N placeholder; kinematic mocap arms) —
   consistent, safe for light cubes / 10–20 N inserts; disclose it, tie to press-fits.
3. *"Toy geometry — all boxes/cylinders."* → the new objects (barbell, spool-shaft, thru-axle
   TODOs in `assembly_planner/problem.py`) help but are still primitives; either land one mesh-ish object or scope
   the claim to primitive-shape assemblies (model is shape-generic: patch/cone machinery).
4. *"7–8 parts is small."* → 12-part cube-plates + scaling curve; say combinatorial assembly,
   not motion, is the bottleneck.
5. *"Multi-robot but you minimize action count, not makespan."* → `[[makespan-utilization]]`:
   add makespan-aware acceptance or declare makespan out of scope IN the problem statement.
6. *"Dogar baseline is your reimplementation."* → reduction (§10) + shared infra; state it.
7. *"Milestone commitment loses completeness."* → windows + completeness-up-to-commitment +
   exact-fallback numbers.
8. *"Open-loop execution; uncertainty?"* → insertion-noise robust boxes at plan time +
   replanning as future work (spec §8.9, `[[sim-in-the-loop-transition]]`).
9. *"Assembly DAG + mate poses are given."* → standard assumption; cite CAD-derived manuals.
10. *"Sim uses capsules, not real collision geometry."* → known; see next-steps #1.

## Open next steps (your list + the rest)
1. **Real collision meshes** (your #1). Sim currently collides the planner's **capsules** (so
   sim contact == planner contract); visual meshes are visual-only. For true execution the
   planner + sim must collide real geometry. Open sub-question from `[[sim-validation-harness]]`:
   capsules are NOT an outer bound of the meshes → decide (a) fatten vendored `ur10.g` capsules
   to cover the meshes, (b) planner collision = meshes, or (c) accept capsule-level validation.
2. **Feasible-velocity paths** (your #2). Proper KINODYNAMIC planning is the real fix; INTERIM
   already in sim = execution-side **trapezoidal time-parameterization** (v≈0 at task goals +
   dwell) so attaches are quasi-static — geometry untouched, "what a jerk-limited tracking
   controller would do." Tunables at top of `sim.py` (`JOINT_RATE`, `RAMP_DIST`, `DWELL_T`).
   Postprocessing-to-feasible (TOPP-RA-style retiming under limits) is the lighter middle path.
3. **A real discriminator** (see ⚠) — the gating result for the whole validation leg.
4. **Force GENERATION check** (config-dependent `F_push` from the Jacobian / max EE force) —
   needed once press-fits (axle/spool) land; must go into BOTH planner cap AND sim (force-
   limited press actuator, not rigid mocap) to stay consistent.
5. **Zero-tolerance export** works (`--strict-collisions` + `--mp-collision-tolerance 1e-6`,
   `[[zero-collision-tolerance]]`) but **BiRRT transition-sampling fails at 1e-6** on larger
   instances (you're debugging; `--mp-clearance` reports the offending pairs).
6. **Non-primitive object** for the geometry-toyness rebuttal (deferred mesh grasps / off-axis
   rest poses `[[rest-pose-offaxis-change2]]`).
7. **Hardware demo** — the money shot for "fixtureless": a side/underside-press mate that the
   topological plan visibly fails and the braced plan succeeds; + the mixed-EE division of labor.

## One-line "am I missing something?" summary
Search + objects + sim harness are in good shape. The paper's **critical path** is: (a) find a
**real discriminator**, (b) run the **wrench-on-vs-off sim success grid**, (c) at least one
**hardware demo**. Real collision meshes and feasible-velocity retiming are needed for genuine
*execution* but not strictly for the *validation claim*; force-generation and press-fits are a
coherent *next paper / extension*, not a gap in this one — as long as you disclose them.
