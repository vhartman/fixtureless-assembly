# Structure-First Milestone Search with Local Reverse Abstractions

This document describes the search selected by `--search-profile standard` as of 2026-08-25.
It is presented twice: Part I is an intuitive explanation, while Part II is an
implementation-faithful methods description suitable as the basis of a paper section.

The standard search is:

```text
--weight 1
--milestones --milestone-oracle structural
--fill-local-h --fill-exact --no-fill-skeleton-first
--open-witness --open-witness-batch 7 --open-witness-epochs 2
```

The profile additionally enables the established physical and geometric contract: real IK,
wrench checking, asymmetric grasps, uniform `press` signatures, bracing, handovers, placement,
hybrid scene-aware grasp sampling, collision/clearance gates, witness persistence, and the
hands-off on-table goal. Those feasibility settings are described in the repository README; this
document focuses on the discrete and continuous search policy.

The central claim is deliberately limited. The method searches a supplied assembly dependency
graph through settled milestones, uses a small reverse abstraction to guide each concrete join,
and fairly reallocates finite IK effort among competing candidate paths. It supports dependency
graphs and detached modules and chronologically backtracks over high-level choices. It is not a
globally complete planner over every concrete boundary realization.

---

## Part I: intuitive explanation

### 1. The basic decomposition

A flat assembly search must choose part order, robot assignments, grasps, placements, braces,
handovers, and robot configurations simultaneously. The cross-product is large, while real-arm
IK is much more expensive than generating a discrete manipulation action.

The standard planner separates three levels:

1. **Dependency milestones:** choose a dependency-valid part or detached module to add next.
2. **Local manipulation search:** find the concrete pick, carry, handover, brace, insertion,
   release, and placement actions that realize that one addition.
3. **Continuous validation:** spend collision-aware IK effort only on edges of promising local
   paths and retain the successful joint configurations.

```text
supplied sequence or dependency graph
                  |
                  v
   choose a dependency-valid part/module
                  |
                  v
 build a small reverse PDB for this one join
                  |
                  v
 weight-one concrete local search over raw grasps
                  |
                  v
 lazily validate candidate-path IK with fair reopening
          | success                     | local failure
          v                             v
 retain settled checkpoint       try another milestone
          |                             |
          +---------- suffix failure ---+
                         |
                         v
             chronological backtracking
```

The outer planner is therefore structure-first, not a second attempt to infer a part order that
the assembly model already provides. Dependencies decide what is legal. The search is still free
to choose among simultaneously eligible branches and to revisit an earlier choice when its
suffix fails.

### 2. Structural milestones

A checkpoint records:

- the concrete support, grasp, placement, and mate state;
- the dependency-closed set of parts already joined to the main assembly;
- the validated plan prefix; and
- the high-level decision that produced the checkpoint.

With `--milestone-oracle structural`, no global order-ideal database is built. Eligible parts are
ordered canonically after dependency filtering. This is intentionally simple: the supplied task
structure chooses the assembly opportunities, while the local PDB handles manipulation guidance.

For each eligible part, the planner attempts a local fill. A successful fill produces a new
settled checkpoint. A local failure blacklists the pair

```text
(current assembled set, attempted part or module)
```

and tries another eligible choice. If a later checkpoint has no continuation, the planner pops
it, blacklists the decision that led there, and resumes at its parent. This is chronological
backtracking, or backjumping with milestone-level conflict records.

Milestones do not force the assembly to be placed on the table after every join. A local endpoint
may be on a stable table face or held by at most one robot. This permits intentional hold-through
plans while preventing braces and loose carried parts from accumulating indefinitely.

### 3. The local reverse pattern database

For a candidate join, let `B` be the current target component, `M` the incoming part or already
built module, and `q` the interface part. Instead of solving backward over the complete assembly,
the planner constructs only the local transition

```text
B  ->  B union M.
```

The abstract state remembers:

```text
(local assembled component, holder/brace grasp labels,
 table face or held-in-air, held incoming parts)
```

It omits joint configurations and normally omits robot identities, but retains total and
per-tool gripper capacity. It also retains the enabled support, table-pick, insertion-load, and
grasp-capability facts.

The database is computed backward from every abstract state representing a valid local endpoint.
Because handovers cost two and ordinary actions cost one, the implementation uses multi-source
Dijkstra. Construction is resumable: reverse search advances only until the abstract state
requested by forward A* is settled, then pauses with its queue and cached facts intact.

The local PDB is a heuristic only. A missing local-PDB value is treated as unknown rather than as
a concrete infeasibility certificate. The concrete search retains its counting lower bound and
continues. A 200,000-state PDB guard fails explicitly instead of silently selecting another
heuristic.

### 4. Uniform grasp projection

Raw grasp IDs are too numerous for a useful reverse state space, but collapsing all grasps into
one label loses the difference between a grasp that can transmit an insertion force and one that
cannot.

Every raw grasp is therefore mapped through the same fact-signature equivalence relation. The
available signature components are:

- `press`: whether the grasp can transmit the part's insertion load;
- `pickface`: the stable table faces from which the grasp is geometrically pickable;
- `contact`: contact location and closing-axis class;
- `solohold`: whether it can hold the relevant assembly alone; and
- `braceside`: whether its closing direction supplies a useful brace direction.

The standard uses `--sig-components press`. In mixed-tool scenes, the usable gripper-kind mask is
always included, even if it was not named as a signature component.

There is no special treatment for the active, interface, or final grasp. Every grasp in every
role uses the same projection. Facts for a projected class are existentially lifted over its raw
members. This may make the abstraction optimistic, but it cannot delete a concrete raw-grasp
route. The forward search still enumerates and validates the raw grasp IDs.

### 5. Concrete local fill search

The local forward graph uses the ordinary actions:

```text
pick, place, assemble, release, add_grasp, handover,
and inhand_reorient when enabled.
```

The first pass may manipulate the incoming component and the target component. Releasing an
inherited grasp is also permitted so a holder retained by a previous milestone cannot strand a
robot. If every local-scope phase fails, the standard repeats the fill with all-part manipulation:
future loose parts may be picked, relocated, handed over, reoriented, regrasped, placed, and
released. They may not be assembled early, and the endpoint must settle every temporarily moved
blocker. This preserves the outer dependency bookkeeping while preventing local-scope failure
from being mistaken for milestone infeasibility.

The local goal requires:

1. the requested mate or module join is complete;
2. no unrelated loose component remains in a gripper; and
3. the component being built is resting or has at most one holder.

The heuristic is the maximum of a cheap local counting bound and the finite local-PDB value. The
`--fill-exact` flag sets the concrete candidate search to weight one and removes the historical
200-candidate LazySP repair cutoff. Within a fixed action phase, search continues until it finds a
validated path, exhausts the finite frontier, or reaches a node/deadline guard.

The selected structural standard has **no grasp or brace action hints**. Its widening order is:

1. local parts, all insertion grasps, without bracing;
2. local parts, all insertion grasps and all braces;
3. all parts, all insertion grasps, without bracing; and
4. all parts, all insertion grasps and all braces.

The optional `global` and `chain` milestone oracles can prepend narrower phases using their
relaxed-model insertion-grasp and brace hints. Those modes always retain the unrestricted phases
above. The all-part passes are entered only after the cheap local graph exhausts. This distinction
matters for ablations: disabling hints in the selected structural method
would be a no-op, whereas disabling `--fill-local-h` would remove the reverse heuristic itself and
is a different comparison. Since the planner returns the first successful phase, `fill-exact`
does not imply global optimality over optional phased unions or over the full assembly problem.

### 6. Lazy IK and open witnesses

Searching the entire local graph with nonlinear IK at every expansion is wasteful. The planner
first searches using the sound cheap gates, obtains a candidate path, and then validates that
path's IK edges.

Under the standard open-witness policy, a stochastic IK failure means

```text
not found with this effort yet
```

rather than

```text
this edge is geometrically impossible.
```

During one fairness epoch:

- the first unresolved edge on a candidate path receives a bounded IK batch;
- success stores the exact joint vector and validation continues;
- failure suspends the edge for the remainder of the epoch;
- the candidate suffix is not probed after the first blocking failure; and
- local search immediately explores another candidate path.

At the next epoch, suspended edges reopen with greater effort. With the standard two epochs,
ordinary skills receive two fresh attempts on their first visit and four on their second.
Assembly uses an atomic seven-slot bundle per visit: six canonical orientation biases and one
unbiased initialization. The configured batch value of seven is the cap used by later ordinary
visits, although a two-epoch standard fill normally reaches only the 2-then-4 tiers.

Successful witnesses persist across local fills when their scene-action signature matches. Failed
queries are not entered into the legacy permanent IK cache. If both epochs finish without a
validated fill, however, the local fill returns failure to the milestone layer. The milestone
transition is then blacklisted even though its unresolved IK edges were not proven infeasible.
This is an operational effort policy, not a decision procedure for continuous feasibility.

### 7. Dependency graphs, Chair, and backtracking

For an ordinary dependency graph, a part becomes eligible when its mate closure is already
present. Independent branches may be attempted in different orders through chronological
backtracking.

Chair additionally contains a detached back/post module. The milestone solver recognizes its
connect root and:

1. builds the required module parts as a separate component;
2. permits that module to remain held between internal fills;
3. performs the module-to-chair join as one composite milestone; and
4. resumes the outer dependency search with the complete module added.

If that composite milestone fails, or if its resulting suffix dead-ends, the outer checkpoint
stack is popped and another eligible high-level choice is attempted. The selected method therefore
does support Chair's dependency graph and does backtrack.

It is nevertheless not globally complete for Chair or for a general dependency graph:

- only one concrete settled boundary state is retained for a successful milestone;
- a later failure does not enumerate every alternate grasp, holder, placement, or robot
  realization of that same successful milestone;
- the internal order of parts within a detached module is currently fixed;
- open-witness effort stops after two epochs;
- the default joint-fill width is two; and
- node, repair, candidate-window, and PDB-size guards are finite.

The local PDB does not create these completeness gaps: it is guidance and does not remove raw
actions. The gaps come from milestone commitment, finite continuous effort, and finite resource
limits.

### 8. Why this is the standard

The standard was selected by a seven-method screen followed by a three-seed finalist comparison
on Cross-7, mixed-tool Chair-8, and Fixture-rack-9, all with four robots. The physics columns were
geometry-only and a wrench-aware 5 N insertion with 5% lateral uncertainty.

Three-seed means are shown as wall time / plan cost:

| Scene | Counting + open | Local reverse exact + open |
|---|---:|---:|
| Cross-7 geometry | 7.8 s / 28 | **5.4 s / 28** |
| Cross-7 noisy 5 N | 12.0 s / 28 | **6.6 s / 28** |
| Chair-8 geometry | **22.7 s / 35** | 26.4 s / **31** |
| Chair-8 noisy 5 N | **26.9 s / 31** | 30.7 s / 31 |
| Fixture-9 geometry | **38.1 s / 24** | 46.6 s / 24 |
| Fixture-9 noisy 5 N | 125.9 s / 40 | **75.7 s / 40** |

Summed over the six equally weighted cells, the selected method took 191.4 s versus 233.3 s for
counting plus open witnesses, an 18% reduction. On the difficult noisy Fixture, peak RSS fell
from approximately 4.22 GiB to 0.91 GiB. Weight 100 was rejected as the default because it
repeatedly returned worse plans and sometimes increased runtime; direct-skeleton probing was a
useful baseline but degraded strongly on wrench-heavy Chair and Fixture.

The result is a deliberately uniform policy rather than a scene classifier. Its known overhead is
the construction of local reverse facts on easy geometric fills. Its benefit is avoiding much
larger IK and memory growth when force-constrained branching becomes difficult.

---

## Part II: paper-style methods description

### 1. Problem formulation

Let an assembly instance contain parts `P`, robots `R`, a designated base `p0`, a mate dependency
graph, stable placements, a grasp database, and optionally a quasi-static wrench model. A concrete
state `s` records kinematic parents, mate/table/grasp supports, component placements, and robot
grasp ownership.

The action set is

```text
A = {pick, place, assemble, release, add_grasp, handover, inhand_reorient}.
```

Ordinary actions have unit cost and handover has cost two because it represents a grasp addition
followed by a release. A feasible transition satisfies discrete preconditions, wrench/support
constraints, cheap geometric gates, and a real-arm IK predicate that returns a joint-space
witness. The goal contains all required mates and satisfies the configured support condition,
which is hands-off and on-table in the standard profile.

### 2. High-level structural search

The milestone search maintains a stack

```text
(s_k, I_k, n_k, delta_k),
```

where `s_k` is a concrete settled checkpoint, `I_k` is its assembled ideal, `n_k` is the plan
prefix length, and `delta_k` is the milestone decision that produced it.

A non-module part `p` is eligible when its mate closure is contained in `I_k`. A connect-root
module is eligible when all dependencies external to the module are contained in `I_k`. Under the
structural oracle, candidates are sorted by deterministic part index after eligibility filtering;
no global order-ideal heuristic is constructed.

#### Algorithm 1: standard milestone search

```text
STANDARD-MILESTONES(instance, width K, repair budget B)
    stack <- [(initial concrete state, {base}, 0, none)]
    blacklist <- empty set
    plan <- empty sequence

    while stack is not empty and budgets remain:
        (s, I, n, via) <- top(stack)

        if I contains every part:
            if s is not hands-off and on-table:
                attempt a final settle-and-release fill
                if it fails: pop; blacklist via; continue
            return plan prefix through n plus final fill

        Q <- dependency-valid parts/modules not blacklisted at I
        sort Q canonically

        for q in Q:
            if q is a connect root:
                result <- BUILD-AND-JOIN-MODULE(s, I, q)
            else:
                result <- FILL-ONE(s, I, q)

            if result succeeds:
                truncate plan to n; append result actions
                push the resulting settled checkpoint
                continue the outer loop

            blacklist (I, q)

        if no singleton succeeded:
            try dependency-valid joint windows of widths 2 through K
            push the first successful joint endpoint

        if no window succeeded:
            pop the checkpoint
            blacklist the parent decision via

    return failure within the configured resources
```

The stack implements depth-first chronological backtracking. Conflicts are keyed by assembled set
and next milestone, rather than by the full concrete checkpoint. This coarse key is economical
but is also a source of incompleteness.

### 3. Goal-conditioned local abstraction

For target component `B`, incoming component `M`, and interface part `q`, the pattern database is
restricted to the two assembly ideals

```text
B and B union M.
```

An abstract state is

```text
x = (I, G, phi, L),
```

where `I` is the local ideal, `G` is the tuple of holder/brace labels, `phi` is a stable table
face or the held-in-air symbol, and `L` is the tuple of held incoming parts and grasp labels.
Robot identities and configurations are omitted; total and per-kind gripper capacities remain.

#### 3.1 Fact-signature labels

For part `p`, raw grasp `g` is mapped to a label by

```text
sigma(p,g) = (tool mask, enabled fact components).
```

The enabled components may encode insertion-load transmission, table-face pickability, contact
geometry, solo holding, and brace direction. The standard component set is `press`, augmented
unconditionally by the tool-kind mask in mixed fleets.

For a projected class `gamma`, a fact is existentially lifted:

```text
F(p,gamma) = OR over raw g in gamma of F(p,g).
```

Consequently, label collapse may add abstract behavior but does not delete the projection of a
raw transition. The same map is used for interface, holder, brace, and loose-part grasps.

#### 3.2 Reverse transitions and facts

The reverse graph contains inverse assemble, pick, lift, place, release, add-grasp, handover,
in-hand reorientation, and loose-placement transitions. It uses necessary abstract facts for:

- dependency and mate legality;
- stable rest faces;
- table-aware floating-gripper pickability;
- gripper-kind capacity and grasp usability;
- release and holding stability;
- insertion-force transmission and target reaction support; and
- optional gripper coexistence geometry.

All abstract fill-goal states are inserted at distance zero. Multi-source Dijkstra computes
distance `d(x)` because action costs are nonnegative and handover costs two. The queue and fact
caches persist between queries; forward search advances reverse Dijkstra only until its queried
abstract state is settled.

The concrete fill heuristic is

```text
h(s) = max(h_count(s), d(pi(s)))
```

when the PDB query is finite. A missing value contributes no pruning and leaves `h_count` in
place.

#### Proposition 1: relaxation admissibility

Assume every concrete local transition projects to an abstract transition of no greater cost and
every enabled fact gate is necessary for its concrete counterpart. Then the local reverse
distance lower-bounds the optimal local concrete cost:

```text
d(pi(s)) <= J_local*(s).
```

To see this, project any feasible concrete suffix into the abstract graph. Each projected edge
has no greater cost, while anonymous robots, omitted configurations, and existential class facts
can only add alternatives. The abstract shortest path can therefore be no more expensive than
the projected concrete suffix.

This proposition describes the abstraction. It does not establish global optimality of the
implemented fill driver because phased action-set widening, finite search guards, and bounded
stochastic validation remain outside the idealized shortest-path statement.

### 4. Concrete fill and lazy continuous validation

For one milestone, the concrete successor function is restricted to the target and incoming
components, except for release of inherited holders. Its goal is the requested join with no
unrelated loose-held part and no more than one holder on the settling component.

Within each widening phase, weight-one A* searches the raw discrete state graph using the local
heuristic and cheap feasibility gates. The lazy validator evaluates expensive IK only on the
current candidate path.

#### Algorithm 2: bounded fair open-witness fill

```text
OPEN-FILL(start, local goal, maximum epochs E)
    preserve solve-wide successful witness pool

    for epoch = 1 through E:
        suspended <- cheap-refuted edges plus empty open set
        engine <- weight-one local A* excluding suspended edges

        while engine has a candidate path:
            for each edge e on the path in order:
                if e has a successful cached witness: continue
                validate e with its next effort batch

                if validation succeeds:
                    retain the exact joint vector
                else:
                    suspend e for this epoch
                    delete e from the current search tree
                    restart candidate extraction immediately

            if every edge is validated:
                return path and endpoint

    return unresolved failure to the milestone layer
```

Ordinary skills receive fresh effort tiers of 2, 4, and then at most `B`; assembly receives an
atomic `B=7` structural bundle on every visit. The standard uses `E=2`, so ordinary edges normally
receive the first two tiers. A zero epoch limit means unlimited reopening but is accepted only
with a finite solver deadline.

Cheap refutations remain permanently dead because they are treated as sound. Stochastic IK
failures remain open in the witness pool but become a failed local milestone after the epoch
budget. This separation avoids claiming that finite optimization failure proves infeasibility.

### 5. Modules, width, and repair

A connect-root module is built as a detached component. Module-internal fills use the connect root
as their settling reference, so the module may remain carried rather than being forced onto the
table. After its currently fixed internal order is complete, a final local fill joins the module
to the main assembly.

When every singleton candidate at an outer checkpoint fails, the planner tries joint windows up
to `--fill-window K`, with standard `K=2`. A joint window permits manipulation actions for two
nominal milestones to interleave. This is a limited relaxation of the settled-boundary
commitment, not a complete widening schedule.

On suffix failure, the checkpoint stack is popped and its parent decision is blacklisted. This
does search different dependency-valid outer orders. It does not retain multiple endpoints for a
successful milestone, nor does it revisit a blacklisted `(I,q)` with a different boundary
realization.

### 6. Witness certificates

Every validated IK action returns a joint vector. Successful open-witness queries retain that
vector, not merely a Boolean verdict. After release hoisting, the vectors are aligned with the
final action sequence; any edge whose predecessor scene changed is certified again.

The saved plan records one search or optimized witness per IK action. Missing required witnesses
cause export to fail. Optional witness optimization may improve a configuration but retains the
search witness as a feasible fallback. Motion planning and replay therefore consume persisted
certificates instead of rerunning stochastic IK.

### 7. Guarantees and limitations

The current implementation supports the following claims:

- Every returned plan passes the enabled discrete, wrench, collision/clearance, and IK gates.
- Successful IK actions carry persisted joint-space witnesses.
- Under Proposition 1's assumptions, the uniform local-PDB value is an admissible local
  abstraction heuristic.
- Uniform grasp projection does not remove raw concrete actions.
- The outer solver respects the supplied dependencies, handles connect modules, and
  chronologically backtracks among high-level milestone choices.
- Weight-one search and removal of the 200-candidate cutoff give exhaustive local repair within a
  fixed phase, provided its finite graph, edge oracle, and implementation guards are exhausted.
- Before a singleton fill is blacklisted, the standard exhausts both the local action scope and
  the all-part manipulation scope across its unbraced/braced phases. Thus locality alone is no
  longer a reason to reject the milestone; finite continuous effort and the remaining guards
  still prevent this from being an unconditional infeasibility proof.

The following are not valid unconditional claims:

- The complete milestone planner is not globally complete.
- `fill-exact` does not make the assembly plan globally optimal, and phased widening means it is
  not necessarily optimal over the union of local phases.
- Two open-witness epochs are not an IK infeasibility proof.
- Chair support and backtracking do not imply enumeration of every Chair boundary state or every
  module-internal order.
- Width two cannot express all manipulation interleavings across three or more milestones.
- Finite node, repair, window, PDB, and wall-time guards can terminate a feasible instance.
- A nonlinear IK call may exceed a cooperative deadline before returning.

Even with unlimited open-witness epochs, the one-boundary milestone commitment and fixed
module-internal order remain. A plausible probabilistic-completeness path would require fair
unbounded IK effort, multiple or nondominated concrete endpoints per milestone, search over module
internal orders, unbounded width/resource limits, and a reopening concrete graph search. That is
future work, not a property of the current standard.

### 8. Optional extensions and ablations

The following mechanisms remain available but are not part of the selected standard search:

- `--milestone-oracle global`: build a full signature-collapsed order-ideal database and use it
  for outer part ordering and fill hints;
- `--milestone-oracle chain`: restrict that global abstraction to a declared sequence; detached
  modules are rejected;
- counting-only fills: disable `--fill-local-h`;
- terminal IK failure: disable `--open-witness`;
- weighted local search: disable `--fill-exact` and select `--weight W`;
- direct skill skeletons: enable `--fill-skeleton-first`; and
- `--milestones-anytime T`: use milestones for an incumbent and spend reserved time on a bounded
  full-state partial-expansion improvement/fallback search.

The direct-skeleton probe searches a bounded family of ordinary manipulation paths—direct carry
and insert, target hold, one handover, one retained brace, and one target placement—before falling
back to the ordinary local graph. It is a baseline and diagnostic, not the standard.

### 9. Standard command and flag mapping

The ordinary fresh-search command is:

```bash
python3 main.py --solve --search-profile standard \
    --assembly ASSEMBLY --parts N --num-robots R --robot ROBOT \
    --seed SEED --save experiments/PLAN.json
```

Scene-specific grippers, insertion force/noise, layout, and special grasp compliance remain
explicit arguments.

| Standard flag | Function |
|---|---|
| `--milestones` | settled-checkpoint decomposition with chronological repair |
| `--milestone-oracle structural` | dependency/module eligibility without a global PDB |
| `--fill-local-h` | resumable goal-conditioned reverse PDB for each join |
| `--fill-exact` | weight-one local candidate search and no 200-repair cutoff |
| `--no-fill-skeleton-first` | use the general local graph directly |
| `--fill-scope-widening` | after local exhaustion, retry with all-part manipulation |
| `--sig-components press` | force-transmission grasp signature, plus mandatory tool mask |
| `--open-witness` | suspend and fairly reopen finite stochastic IK failures |
| `--open-witness-batch 7` | maximum ordinary tier and atomic assembly bundle |
| `--open-witness-epochs 2` | measured finite operational effort per local fill |
| `--fill-window 2` | built-in joint-fill widening limit |

`--search-profile open` is retained as a compatibility alias for `standard`.
`--search-profile fast-open` changes only the analytic warm-start cap and remains an aggressive
ablation.

### 10. Experimental reporting

A paper evaluation should report four layers separately:

1. **Structure:** dependency width, module count, chosen order, failed milestones, windows, and
   backtracks.
2. **Local abstraction:** PDB states settled/discovered, reverse time, fact counts, and raw grasps
   versus signature classes.
3. **Continuous validation:** IK calls, effort slots, failures by gate and skill, successful
   witnesses, and IK wall time.
4. **End-to-end outcome:** first-plan time, action cost, peak memory, witness completeness, motion
   planning success, and simulation or real execution.

The principal ablations should be counting versus local PDB, terminal versus open IK failure,
weight one versus weighted search, geometry-only versus wrench/noisy insertion, and one versus
multiple retained boundary states. The last comparison targets the primary remaining algorithmic
limitation rather than adding another part-order heuristic.

There is no “without hints” row for the selected standard: `--milestone-oracle structural`
constructs no global oracle, so it supplies no insertion-grasp or brace hints in the first place.
`--no-fill-local-h` is instead a much larger counting-versus-reverse-heuristic ablation and should
not be mislabeled as hint removal.

The manipulation-locality ablation keeps reverse-PDB guidance, weight-one exact fill search, and
open witnesses fixed. It compares the standard narrow-to-broad policy
(`--fill-scope-widening --no-fill-touch-all`) against an eager all-part policy
(`--no-fill-scope-widening --fill-touch-all`). Both ultimately expose the same broad concrete
action graph; only the scheduling differs. The broad graph permits picking, placing, handing over,
reorienting, regrasping, or releasing any future part. It still forbids assembling a part other
than the requested join, keeping the concrete state aligned with milestone bookkeeping, and
requires moved blockers to be settled at the endpoint.

The reproducible manifests are:

```text
benchmarks/manifests/paper_search_matrix.json
benchmarks/manifests/paper_search_stress_4r.json
benchmarks/manifests/search_finalists_4r.json
benchmarks/manifests/search_manipulation_scope_ablation.json
```

The honest summary is: the method is a structure-aware, locally abstraction-guided, lazily
validated milestone planner with chronological backtracking and persisted witnesses. It is robust
on the tested sequence and dependency-graph assemblies, including Chair's detached module, but it
is not yet a globally complete search over alternative concrete milestone boundaries.
