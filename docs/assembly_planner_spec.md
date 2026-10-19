# Multi-Robot Grasp and Manipulation Planning for Sequential Assembly
## Design Document / Proof-of-Concept Specification — rev. 2

**Status:** design draft, intended as (a) a spec a coding agent can implement directly, and (b) a first skeleton for a paper.

**Changes in rev. 2:** (i) intermediate assembly states need not be self-supporting (A11) — per-mate, possibly unilateral wrench models, state-dependent external wrench, forced-hold intervals; (ii) motion constraints during forced-hold transit (safe orientation region Ω, supported-along-path check in L4, corridor term in the backward filter); (iii) action set redesigned around the **fuse-vs-reify principle** — fused `pick`/`place` shorten the horizon; unfused variants retained for interleavable maneuvers.

**Lineage:** generalizes Dogar, Spielberg, Baker, Rus, *Multi-Robot Grasp Planning for Sequential Assembly Operations* (ICRA 2015). Their CSP formulation is recovered as the degenerate case where regrasps are always feasible, robots are homogeneous and mobile, and the only decision is transfer-vs-regrasp per operation boundary. We drop those assumptions and replace the outer CSP with an explicit search over scene-graph states; small constraint problems survive as the *inner* per-node binding step.

---

## 1. Problem statement

**Given:**
- A set of rigid parts with known geometry.
- An assembly **dependency graph** `D`: nodes are assembly operations `op = (inputs, output, pose, process_wrench, self_supporting)`; edges are precedence. Part poses within each operation's output are fully specified — the *what* and *in-which-order* of assembly is input; we plan the *how*. `self_supporting: bool | until(op')` declares whether the operation's output can support itself without robot contact (see §5.2).
- A fixed roster of robots `R`, possibly heterogeneous (fixed or mobile bases, different kinematics, two-finger grippers with different capabilities). No robot identity symmetry is assumed.
- Environment geometry (table(s), obstacles).

**Find:** a minimum-cost sequence of manipulation actions that completes all operations in `D`, such that every state — and every point *along* every motion (§5.3) — is quasistatically supported, and every transition is kinematically and geometrically feasible.

**Cost:** counted on **gripper open/close events + transports**, weighted (regrasp-like maneuvers cost more; transfers are the absence of events and cost ~nothing). This event-based accounting is deliberately encoding-invariant: fused and decomposed action sequences describing the same physical maneuver cost the same, so the search never prefers one encoding over the other for spurious reasons. Makespan is explicitly *not* the objective (parallelization is post-processing, §8.6).

## 2. Assumptions (PoC scope)

| # | Assumption | Consequence / where it is relaxed |
|---|---|---|
| A1 | Quasistatic manipulation throughout; inertial wrenches handled by "move slowly" (velocity cap), not modeled | Time-parameterization under wrench constraints (TOPP-style) → §8.7 |
| A2 | Two-finger grippers only | Dexterous / in-hand → §8.6 |
| A3 | Finite, pregenerated grasp set per part (part frame) | Iterative widening → §8.4 |
| A4 | Finite parking poses: per part, a discrete set of (stable face, table slot) | Same widening mechanism |
| A5 | Assembly sequence & poses given by `D` | Joint sequencing+grasping search out of scope |
| A6 | Regrasps are **NOT** assumed free — a grasp/support may be removed only if the remaining support set suffices | Core departure from ICRA'15 |
| A7 | Conservative wrench sufficiency at search time; exact force balance deferred to validation — **except** per-mate capabilities, which must be modeled (possibly unilateral) already at search time (§5.2) | §5, §6.5 |
| A8 | Contact-rich phases executed by skills with declared funnels; planner reaches pre-contact poses only | §7 |
| A9 | Deterministic skills above a reliability threshold | Expected-cost / belief planning → §8.6 |
| A10 | One witness per search node, seeded from parent | Multi-witness / joint refinement → §8.5 |
| A11 | **Intermediate assembly states need not be self-supporting.** Nodes are required to be *supported*, not free-standing; grasps and push-contacts are first-class supports. An assembly may pass through states (e.g. pushed-through-but-unpinned) that are only stable while held | Truly dynamic intermediates (part must keep moving) are out of scope; encapsulate as composite skills if needed (§7) |

## 3. State: scene graph + witness

A search node has a **discrete identity** (hashable, duplicate detection) and a **continuous payload** (the witness — NOT part of identity).

```python
@dataclass(frozen=True)   # identity — hashable
class NodeId:
    parent:   Map[Body, Body]        # kinematic tree: part -> (table | gripper(r) | part)
    grasp:    Map[Body, GraspId]     # if parent is a gripper
    place:    Map[Body, PlacementId] # if parent is table: (stable_face, parking_slot)
    supports: Map[Body, FrozenSet[ContactId]]  # support MULTIGRAPH, distinct from parent
                                     # ContactId ∈ {grasp-by-(r,g), table-face, mate(op)}
    done_ops: FrozenSet[OpId]

@dataclass                # payload — mutable, excluded from hashing
class Node:
    id:       NodeId
    witness:  Map[Robot, Configuration]  # one consistent kinematic realization
    g_cost:   float
```

**Design notes:**
- `parent` is a tree (kinematics); `supports` is set-valued (statics). They must be separate: a part can have one parent and several supporters (held + braced; mated + held). "May contact c be removed from p?" is answered from `supports`, never from `parent`. This is what makes fixtureless and non-self-supporting-intermediate reasoning possible.
- **Mates are supports with their own (possibly partial) wrench capability.** A mate edge `mate(op)` enters `supports` when `op` completes, but its capability may be unilateral (e.g. a part slid through a hole resists lateral load, not axial back-out). "Mated" therefore does NOT imply "supported" — sufficiency is always evaluated over the full set (§5.2).
- Two nodes with equal `NodeId` are the same node even with different witnesses. The witness enforces **cross-node consistency**: robots not acted upon by an edge keep their exact configuration from the parent witness. Per-node independent feasibility solves are optimistic and forbidden (they imply phantom motions of stationary holding robots).
- `supports` being part of identity means "held" vs "held-and-braced" vs "mated-and-held" are different nodes — they admit different actions and must not alias.

## 4. Actions (search edges)

**Fuse-vs-reify principle.** An action is one atomic edge whenever no other robot could usefully act during its intermediate states; those intermediates are *checked inside `feasible()`* but not reified as nodes. Where another robot's interleaved action is meaningful (handovers, bracing), unfused primitives expose the intermediate state. This keeps the horizon short for common maneuvers without losing expressiveness: every maneuver expressible with fully split primitives remains expressible; the emergent ones are just no longer the default-priced path.

```python
def successors(s: Node):
    # ---- fused, common-case primitives (1 edge each) ----
    yield from picks(s)        # pick(r,p,g): grasp AND lift. parent table->gripper,
                               #   face support removed. g restricted by §6.2 filter.
    yield from places(s)       # place(r,p,slot): set down AND open. parent gripper->table,
                               #   grasp support removed.
                               #   feasible ⟺ supported(p, {face}, STATIC) — i.e. the old
                               #   release-gating, now internal to the fused edge.
    # ---- unfused variants (footholds for interleaving / bracing) ----
    yield from add_grasps(s)   # add_grasp(r,p,g): grasp WITHOUT lifting/reparenting;
                               #   adds a grasp support edge (brace / handover foothold)
    yield from releases(s)     # release(r,p): open WITHOUT having set down; support-gated:
                               #   feasible ⟺ supported(p, supports \ {this grasp}, STATIC).
                               #   If the released grasp was the parent, reparent to a
                               #   remaining supporter (table face or other gripper).
    yield from set_downs(s)    # set_down(r,p,slot): touch table WITHOUT opening;
                               #   adds face support, keeps grasp, parent stays gripper
    # ---- assembly ----
    yield from assembles(s)    # op ∈ ready_ops(s): all inputs held in op-compatible grasps
                               #   at pre-insertion poses (§7); rewrites parts into one
                               #   sub-assembly; adds mate(op) support edge with its
                               #   declared (possibly unilateral) capability
# emergent maneuvers (not primitives):
#   table regrasp   = place ; pick                        (2 edges)
#   handover        = add_grasp(r2) ; release(r1)         (2 edges — intermediate two-gripper
#                                                          state is what makes it stability-legal)
#   table handover  = set_down(r1) ; add_grasp(r2) ; release(r1) ; ...
#   bracing         = add_grasp held across assemble(s), released after
#   transit escort  = add_grasp before a forced-hold move, release after (§5.3)
```

`ready_ops(s)` = ops whose predecessors in `D` are in `done_ops` (weakest use of `D`; strong uses in §6).

Optional macro-edges (`handover(r1→r2,p)`, `table_regrasp(r,p,g→g')`) may be *added alongside* primitives to shorten the horizon further — §8.2. Costs are event-based (§1), so macro vs. primitive encodings of the same maneuver cost identically.

## 5. Feasibility

### 5.1 Layered edge check — cheap first

```python
def feasible(action, s, s2) -> bool:
    # L0  discrete filters (~free): grasp usefulness incl. forced-hold intersections (§6.2),
    #     robot budget (§6.4), part-symmetry-canonical grasp ids
    # L1  reachability lookup (~free): inverse/forward reachability map contains the
    #     required world gripper pose for that robot                     [EXT-1]
    # L2  stability, conservative — over the FULL support set, at s2 AND at the checked
    #     intermediates of fused actions (§4), using situation-dependent external wrench:
    if not all(supported(p, s2.supports[p], situation(action, p), W, s2.id)
               for p in parts(s2)):
        return False
    # L3  kinematic binding: incremental NLP/IK for ONLY the robots moved by `action`,
    #     all others pinned at s.witness; collision-checked against full scene.
    w = bind_witness(s2, seed=s.witness, moved=action.robots)           # [EXT-2]
    if w is None:
        return False        # solver "infeasible" may be a local minimum → §8.5
    s2.witness = w
    return True
# L4 (motion planning between witnesses — incl. supported-along-path on forced-hold
#     segments, §5.3) and L5 (exact force balance) are DEFERRED to lazy validation (§6.5).
```

### 5.2 Wrench model

```python
# offline:
W[p, g]        = grasp_wrench_capability(p, g)         # friction cone / gripper limits
W[p, face]     = contact_wrench_capability(p, face)    # table faces
W[mate(op)]    = mate_wrench_capability(op)            # PER-MATE, possibly unilateral /
                                                       # cone-like (resists lateral, not
                                                       # axial pull-out). Load-bearing at
                                                       # search time (A7): "mated" must NOT
                                                       # be shortcut to "supported".
def external_wrench(p, situation, node_id):
    # situation = STATIC        -> gravity  (+ PRELOAD from active mates: a mate may exert
    #                              a reaction wrench — spring/clip — until a later mate
    #                              neutralizes it; hence node_id-dependence)
    # situation = ASSEMBLE(op)  -> the above + op.process_wrench incl. lateral skill forces (§7)
    ...

def supported(p, contacts, situation, W, node_id) -> bool:
    # conservative sufficiency of the contact set vs. external wrench.
    # PoC rule may be crude for grasps/faces (stable face OR force-closure grasp OR
    # ≥2 contacts) but must use the real per-mate capability for mates.
    return wrench_balance(sum_capabilities(W, contacts),
                          external_wrench(p, situation, node_id))
```

**Non-self-supporting intermediates work with no new machinery (A11):** after `assemble(op_push)` with `self_supporting=until(op_pin)`, `supports = {mate(op_push), grasp(r,g)}` passes L2 only *because the grasp is in the set*; `release`/`place` are automatically infeasible until `mate(op_pin)` closes the balance. The robot holding through both operations **emerges** — the planner is simply never offered a legal release in between.

### 5.3 Motion constraints on forced-hold transit

During a **forced-hold interval** (any segment where a part's support set includes an insufficient/unilateral mate, i.e. between `op` and `op'` for `self_supporting=until(op')`), stability is a **pointwise constraint along the path**, not just at endpoints: gravity in the part frame depends on the assembly's world orientation, and a unilateral mate breaks orientation symmetry (tilt hole-down → part slides out even though both endpoints were supported).

- **Safe orientation region** `Ω(p, support_set) ⊂ SO(3)`: orientations where the support set balances gravity(+preload). Precomputed offline from the mate cone and the carry grasp capability — grasp-dependent: a stronger grasp enlarges Ω, possibly to all of SO(3), dissolving the constraint.
- **L4 on forced-hold segments** = motion planning with the additional constraint `orientation(t) ∈ Ω ∀t`. PoC implementation: plan normally, **reject-check** `supported(...)` at collision-check resolution along the path; failures route per §6.5. Proper constrained planner (projection / CBiRRT-style) → §8.7. Inertial component: velocity cap under A1; TOPP-with-wrench-constraints → §8.7.
- **Failure repair enabled by primitives:** if no supported path exists for the current support set, a **transit escort** (`add_grasp` before the move, `release` after) enlarges Ω for the duration — an existing composition, selected because the constrained planner fails without it.

## 6. Search

### 6.1 Main loop (A*, anytime-ready)

```python
def plan(D, parts, robots):
    G  = pregenerate_grasps(parts)                 # A3, symmetry-quotiented
    W  = precompute_wrench_models(parts, G)        # §5.2, incl. per-mate capabilities
    OM = precompute_safe_orientation_regions(W)    # §5.3, per (part, support set)
    U  = backward_useful_grasps(D, G, W, OM)       # §6.2
    RM = reachability_maps(robots)                 # L1 tables            [EXT-1]

    start = initial_node(parts)                    # all parts parked, supports={face}
    open_pq, best = PQ([(h(start), start)]), {start.id: 0.0}
    blacklist = FailureStore()                     # §6.5, with reasons

    while open_pq:
        s = open_pq.pop()
        if s.id.done_ops == all_ops(D):
            plan = reconstruct(s)
            ok, failure = validate_lazily(plan)    # §6.5: L4 (+path support) & L5
            if ok: return plan
            blacklist.add(failure); continue       # search state kept alive → resumes
        for action, s2 in successors(s):           # pre-filtered by U, budget, canon. order
            if blacklist.forbids(action, s.id): continue
            if not feasible(action, s, s2): continue
            c = best[s.id] + cost(action)          # event-based costs, §1
            if c < best.get(s2.id, INF) and not dominated(s2, best):     # §8.3
                best[s2.id] = c
                open_pq.push((c + h(s2), s2))
    return INFEASIBLE
```

### 6.2 Backward useful-grasp filter (heir of ICRA'15 transfer constraints)

Preprocessing over `D` in reverse topological order:

```python
U[p, op] = { g ∈ G[p] :  collision_free_at(op, p, g)        # not blocked by op's other parts
                     and supported(p, {g} ∪ env_supports(op), ASSEMBLE(op), W, ·) }
```

- `picks(s)` offers only grasps in `U[p, next_op(p)]`, plus grasps one regrasp away from such.
- **Preferential chaining:** non-empty `∩ᵢ U[p, opᵢ]` over p's route through `D` → a zero-regrasp carry exists; prefer those grasps. Empty → ≥1 regrasp on p is provably forced (admissible +1 pick in `h`; directs grasp-widening).
- **Mandatory chaining on forced-hold intervals:** `self_supporting=until(op')` makes the interval `[op, op']` a forced-hold segment. The carry grasp MUST come from `U[·,op] ∩ U[·,op']` **and** its `Ω(·, {mate, grasp})` must contain an **orientation corridor** connecting the assembly's orientation at `op` to that at `op'` (a connectivity query on the precomputed Ω, not a planning problem). Empty → known at preprocessing time that the plan needs a mid-interval handover (legal: support never drops below sufficiency during add_grasp-then-release) or a brace; `h` prices it in.
- Soundness: the filter is pairwise-optimistic → it may only *remove* certainly-useless grasps, never certify good ones. Search still verifies; completeness preserved.

### 6.3 Heuristic (admissible)

```python
def h(s):
    return sum(cost(ASSEMBLE)
               + sum(cost(PICK) for p in inputs(op)
                     if not held_in_grasp_from(U[p, op], s))
               for op in ops_not_done(s, D))
    # sharpening: + forced-regrasp / forced-handover counts from §6.2 emptiness results
    # alternative lower bound: critical-path length in D (take max of the two)
```

### 6.4 Structural pruning from D

- **Canonical ordering:** among incomparable ready ops (parallel in `D`) whose manipulations touch disjoint robots+parts, expand only one interleaving (lowest op index first). Unaffected by fixed bases.
- **Robot budgeting:** prune states where `free(s) + releasable(s) < max holders+bracers+escorts needed among remaining ops` — counting check, no geometry. (`releasable` respects support gating; escorts from §5.3 count.)

### 6.5 Lazy validation + blacklist with reasons

On each candidate plan: (L4) motion-plan between consecutive witnesses — terminal segments before an assemble target the **pre-insertion pose**, never the mated pose; forced-hold segments additionally reject-checked for path support (§5.3); (L5) exact force balance on support-critical states/transitions. Failure reasons route repairs:

| Failure | Repair |
|---|---|
| no collision-free motion between witnesses | resample/optimize witnesses (§8.5) — NOT edge blacklist |
| **no supported path (forced-hold segment)** | try carry grasp with larger Ω → else insert transit escort (`add_grasp`/`release` around the move) |
| exact wrench fails where L2 passed | blacklist that (support-set, situation) atom; tighten `supported` locally |
| skill precondition unreachable | blacklist edge; if grasp-caused, widen G[p] (§8.4) |

Search state (`open/best/blacklist`) persists across validation rounds — resume, don't restart (full incrementality → §8.3). Distinguishing collision-failures from support-failures in the motion planner's report is required plumbing from day one.

## 7. Skill interface (contact-rich phases)

Every edge is "precondition on (NodeId, witness) → scene-graph rewrite"; contact-rich actions make the contract explicit. The **funnel spec** is the planner–controller contract, needed per skill before implementation:

```python
@dataclass
class SkillSpec:                      # e.g. peg-in-hole insertion for op
    entry_region:   PoseRegion        # pre-insertion relative-pose tolerance (funnel mouth)
    worst_wrench:   Wrench            # incl. LATERAL search/wiggle forces — feeds L2/L5
    effects:        GraphRewrite      # mate edge(s) added with their W[mate] capability;
                                      # parts merged; may set/clear preload terms (§5.2)
    success_rate:   float             # admit iff ≥ threshold (A9)
    cost:           float
```

Consequences: witnesses for `assemble` bind at `entry_region` poses (the planner never solves IK for the mated, most-cluttered configuration — the skill owns the last centimeter); `supported()` during an assemble uses `worst_wrench`, which is what makes the planner *discover bracing*. A **composite skill** (e.g. push-through+pin as one edge adding both mates) is the sanctioned encapsulation for intermediates that violate A1/A11 (truly dynamic, or requiring continuous non-prehensile force beyond what's modeled); its funnel declares multi-robot preconditions.

## 8. Future work / optimizations — with their slots

**8.1 Offline IK & collision tables at operation states.** `D` fixes all part poses at every op → precompute `IK_table[r,p,g,op]` and pairwise config-collision per op; `bind_witness` at assemble states becomes lookup + tiny discrete matching. Slot: [EXT-1]/[EXT-2]. *Highest payoff; degenerates gracefully into ICRA'15's sampled domains.*

**8.2 Search algorithmics.** Weighted/anytime A* (`f = g + w·h`, decay w, reuse tree — restores ICRA'15's anytime property); partial expansion (k best successors per action type, re-open for the rest); macro-edges alongside primitives (§4). Slots: `plan` loop, `successors`.

**8.3 Dominance + incremental replanning.** `dominated()`: same NodeId-modulo-free-robots with superset of free robots at ≤ cost dominates. LPA*/D*-Lite-style subtree invalidation on blacklist events. Slots: marked in §6.1.

**8.4 Iterative widening.** Small G[p] (3–5 grasps) and few parking slots initially; widen the specific domain implicated by failure reasons (empty successor sets, §6.2 emptiness — incl. empty Ω-corridors, skill-precondition failures). Slot: `pregenerate_grasps` + §6.5 router.

**8.5 Witness quality & joint refinement.** (a) Post-`bind_witness` local optimization: clearance/manipulability via nullspace steps. (b) Pre-motion-planning **joint NLP over all plan keyframes** (consistency constraints included), initialized from stored witnesses — LGP-style global bindings with the search's early, attributable failures. (c) Multi-witness / constraint back-propagation if resample-on-failure proves insufficient. Slots: [EXT-2]; between `reconstruct()` and `validate_lazily()`.

**8.6 Beyond PoC assumptions.** Dexterous hands: per-(part,hand) **in-hand connectivity graph**; primitive `inhand_reorient(r,p,g→g')` with *transition* wrench capability and required external support (extrinsic dexterity = transient support edge — machinery exists); node identity unchanged. Dynamic skills: expected-cost planning over `success_rate`. Makespan: schedule the sequential plan along `D`'s parallel structure. Free sequencing: out of scope. Non-prehensile **push contacts** as a further `ContactId` type (unilateral force along pusher normal; pusher robot committed like a holder) if continuous-push intermediates are needed without full skill encapsulation.

**8.7 Constrained transit, properly.** Replace §5.3's reject-check with a constrained motion planner (projection/CBiRRT) on `orientation ∈ Ω`; time-parameterize forced-hold trajectories under contact-wrench constraints (TOPP-RA-style; the "waiter's problem") to lift the velocity-cap crudeness of A1. Slot: L4 implementation in §6.5.

**8.8 Supporting meshes (fixtures / jigs / cradles as first-class supports).** Generalize the support vocabulary beyond {grasp, flat table face, mate}: a *supporting mesh* is static environment geometry (v-block, cradle, vise, dedicated jig — or a table with arbitrary relief) that can hold parts/subassemblies in poses the flat table cannot. Per layer:
- **State:** new `ContactId` type `rest(mesh, pose_id)`; `place` generalizes from `(stable_face, slot)` to `(support_mesh, stable_pose_id, slot)` — A4's finite parking poses enumerate per (part-or-group, mesh).
- **Statics (§5.2):** `W[p, rest(mesh, pose)] = contact_wrench_capability(patch(p, mesh, pose))` — the table's unilateral friction-cone machinery on the mesh-contact patch. A cradle's capability may be *bilateral in some directions* (lateral pockets), which is exactly what makes tilted / off-axis parking stable.
- **Preprocessing:** stable-pose enumeration on meshes (extends the off-axis rest-pose plan): poses whose contact patch balances gravity, stored as placement ids; widened on demand (§8.4).
- **Actions: none new.** `place`/`set_down`/`pick` accept the richer placement ids — a jig is "a table slot with a richer capability." All support-gating logic (§5.2) applies unchanged.
- **Payoffs:** (i) *fixture escort* — a cradle whose capability closes the balance substitutes for a robot holder/brace during forced-hold intervals (§5.3), freeing arms (utilization); (ii) mates that require a held-or-jigged base orientation become assembles INTO the fixture; (iii) staging — parts pre-oriented at pick-friendly poses, reducing regrasps (interacts with in-hand reorient, §8.6).
- **Scope note:** non-mate part-ON-part resting (stacking without a mate) is the same mechanism with a *movable* support mesh, but needs support-dependency tracking (moving the lower part invalidates the rest) — keep as a separate extension. *(Superseded by §8.9: with sim-in-the-loop the sim IS the support model and no bespoke rest-dependency relation is needed.)*

**8.9 Simulation-in-the-loop transition (deterministic global effects) — for non-prehensile / settling actions.** Today the transition function is *analytic and LOCAL*: `successors` both enumerates actions and computes each one's effect on only the manipulated subassembly — the **effect-containment** assumption (§2). Some actions genuinely violate containment — a **push** may shove neighbours; a **drop onto a stack** may topple or bridge — yet remain **deterministic**. For those actions only, replace the analytic effect with a **deterministic forward simulator**: `state' = sim(state, action)`, free to move every part at once. This keeps determinism and the classical **sequence-planning** paradigm intact — same (state, action) → same next state, the plan is still a linear action list. It is explicitly **NOT** a move to policies / FOND / MDP / POMDP (that would turn the solution into a contingent tree and break every downstream consumer — replay, MP export, witness opt, schedule); the *only* assumption dropped is effect *containment*, not determinism.
- **Search:** unchanged. `successors` splits into (a) *enumerate candidate actions* and (b) `state' = sim(state, action)`. A*, the heuristics, anytime, lazy validation all stand — the swap is behind the successor boundary.
- **Two cost centres (where the real work is):**
  1. *Continuous → symbolic re-abstraction + canonicalisation.* The sim returns arbitrary continuous poses for possibly-many parts; infer the symbolic state back (which parts now rest on / contact which — rai contact queries; `get_initial_state` already does the table-only version of scene→state) and then **snap poses to the discrete anchor set** (exactly as `place_cells` / the `_initial_anchor` bijection do) so the state stays finite, hashable, and reproducible. Skip the snap and the state space silently goes continuous → the search stops terminating. This is the crux.
  2. *Deterministic, reproducible sim + per-edge cost.* States are hashed and re-expanded, so `sim` must be bit-reproducible (the `PYTHONHASHSEED=0` discipline, now for physics — notoriously touchy). A sim call per edge is far heavier than the analytic model → **gate the sim to the actions that need it** (push, stack-settling); keep pick/place/assemble on the analytic path.
- **Subsumes two other slots.** (i) The analytic **push contact** `ContactId` of §8.6 is no longer needed — a multi-body shove is just whatever `sim` produces. (ii) The **stacking** extension (§8.8 scope note) needs no bespoke `rest(below)` support relation or support-dependency DAG — *the sim IS the support model*: drop a block, let it settle, read where it lands (topple / bridge / multi-support all emerge). This is why stacking is best **deferred until sim-in-the-loop exists** rather than hand-rolled now (avoids the tree→DAG, non-rigid-coupling modelling escalation).
- **Execution / replanning:** the real world diverges from the nominal sim → **plan on the nominal, re-read the true state, replan** (determinise-and-replan). That is the accepted home for real-world non-determinism; the *planner* itself stays deterministic. Bounded within-action uncertainty that does *not* need a sim is still handled in-paradigm by the robust-mode / insertion-noise worst-case-over-a-box machinery (§5.2) — no branching.
- **Scope now:** design only, not scheduled. Interim cheap options if either is needed before the sim lands: a **containment-preserving push** (analytic, swept path required collision-free of all but the pushed part — §8.6) and **single-support stacks** (each part rests on ≤1 other, so the kinematic tree and the "can't move the bottom while occupied" rule keep the rigid-parent approximation sound).

## 9. PoC milestones (for the coding agent)

1. **M1:** Scene graph + NodeId hashing + the 6 primitives (§4) + crude `supported()` *with per-mate capabilities*; 2 fixed-base robots, 3 parts, 2 ops, hand-authored grasps; BFS end-to-end. *Verifies: emergence of handover/regrasp; support-gated place/release; fused-action horizon (regrasp = 2 edges).*
2. **M2:** A* + `h()` + backward filter U (incl. one forced-hold interval with mandatory chaining) + robot budgeting + canonical ordering. *Verifies: pruning correctness (plans match M1, faster); forced carry emerges from an unpinned mate.*
3. **M3:** Real geometry (Pinocchio+Coal or PyBullet): `bind_witness` with parent seeding, L1 reachability tables. *Verifies: witness consistency (stationary robots don't move).*
4. **M4:** Lazy validation with reason-routed repairs; RRT-connect between witnesses; reject-check path support on a forced-hold segment (a case where a naive path tilts out of Ω and the escort repair fires). *Verifies: §5.3 end-to-end.*
5. **M5:** Skill interface with stubbed insertion funnel; bracing emerges under nonzero `worst_wrench`. *Fixtureless novelty demo.*

## 10. Relation to ICRA'15 (paper-positioning paragraph)

Setting A6→free, A11→all-states-self-supporting, homogeneous mobile robots, and restricting actions to {assemble + implicit transfer/regrasp per boundary} collapses this framework to Dogar et al.: nodes lose `supports`, the search over boundaries reduces to their transfer-constraint imposition, `bind_witness` per operation is exactly their per-operation CSP with seeding, and §6.2's filter is their transfer constraint moved into preprocessing. Contributions over that baseline: (i) conditionally-feasible regrasps via explicit support reasoning (fixtureless); (ii) non-self-supporting intermediate assemblies via per-mate wrench models and forced-hold intervals, including motion-level support constraints (Ω) absent from prior grasp-planning formulations; (iii) heterogeneous fixed-base teams via identity-bearing values; (iv) emergent bracing/handover/escort from primitive support edits under the fuse-vs-reify principle; (v) skill-funnel integration making process wrenches first-class in grasp selection.
