# Structure-First Milestone Search with Local Reverse Abstractions

This document describes the search selected by `--search-profile standard` as of 2026-09-01.
It is presented twice: Part I is an intuitive explanation, while Part II is an
implementation-faithful methods description suitable as the basis of a paper section.

The standard search is:

```text
--weight 1
--milestones --milestone-oracle structural
--fill-local-h --fill-greedy --no-fill-exact --no-fill-skeleton-first
--milestone-boundaries 4 --fill-scope-widening
--placement-yaws 4 --placement-yaw-widening
--open-witness --open-witness-batch 7 --open-witness-epochs 2
--lazy-edge-selector forward
```

The profile additionally enables the established physical and geometric contract: real IK,
wrench checking, asymmetric grasps, uniform `press` signatures, bracing, handovers, placement,
the scene-reserved hybrid grasp database (deterministic representatives for available
per-kind initial-pick/assemble scene facts before the diversity-ordered hybrid tail), a compact
centred initial scatter,
collision/clearance gates, witness persistence, and the
hands-off on-table goal. Those feasibility settings are described in the repository README; this
document focuses on the discrete and continuous search policy.

The central claim is deliberately limited. The method searches a supplied assembly dependency
graph through settled milestones, uses a small reverse abstraction to guide each concrete join,
and fairly reallocates finite IK effort among competing candidate paths. It supports dependency
graphs and detached modules and chronologically backtracks over high-level choices. It lazily
considers up to four distinct settled endpoints for each concrete milestone transition. This
removes the former claim-once boundary failure mode, but a finite boundary budget is still not
global completeness.

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
 greedy best-first local search over raw grasps
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
       reject this exact endpoint; retry lazily
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
settled checkpoint. A local failure blacklists the concrete transition

```text
(exact parent state, current assembled set, attempted part or module)
```

and tries another eligible choice. If a later checkpoint has no continuation, the planner pops
it but rejects only the exact settled endpoint that led there. It then reruns the same exact
local fill while excluding that endpoint, which produces the next-cheapest distinct symbolic
boundary if one exists. Only after four distinct endpoints fail does the standard profile
blacklist that parent transition. The parent state is part of the no-good key: a suffix action
that failed from one holder/rest-face state is not incorrectly suppressed after a different
boundary is selected. This is chronological backtracking with bounded, lazily generated
milestone alternatives.

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

The heuristic is the maximum of a cheap local counting bound and the finite local-PDB value.
The standard `--fill-greedy` policy orders the concrete queue by this heuristic alone (`f=h`).
It changes no successor, cheap gate, IK gate, or widening phase. It also removes the historical
200-candidate LazySP repair cutoff: within a fixed action phase, validation repairs continue until
a path succeeds, the finite frontier is exhausted, or a node/deadline guard is reached.

The experimental `--fill-incumbent-first` schedule runs this in two orders without changing the
graph or gates: greedy best-first (`f=h`) supplies a validated incumbent, then weight-one A*
searches with `g+h` no larger than that incumbent. Positive IK certificates and open-failure
effort are shared. This two-pass policy remains an experimental comparison for finding the first
milestone plan; the standard uses a single greedy pass there. The separate
`--milestones-anytime` phase owns cost improvement through greedy-then-weight-one neighborhoods
that grow to the complete milestone horizon by default.

The selected structural standard has no global-oracle grasp or brace phase restrictions. Table
yaw is a finite parameter family: every stable rest has the four quarter-turn copies, while the
local PDB quotients them to one abstract rest class because its represented support/wrench facts
are invariant to world-z rotation. Concrete IK is not quotiented. Its widening order is:

1. local parts, all insertion grasps, without bracing, first at one yaw and then at nested
   two- and four-yaw prefixes;
2. local parts, all insertion grasps and all braces, at the complete four-yaw lattice;
3. all parts, all insertion grasps, without bracing, at the complete lattice; and
4. all parts, all insertion grasps and all braces, at the complete lattice.

This is finite iterative broadening rather than eager branching. Quaternion-derived placement
IDs and successful IK witnesses remain stable across levels. The final phase union is exactly the
eager four-yaw action set, so scheduling cannot remove a discretized placement; it can only delay
its evaluation. The finite four-angle lattice is not completeness over continuous table yaw.

The optional `global` and `chain` milestone oracles can prepend narrower phases using their
relaxed-model insertion-grasp and brace hints. Those modes always retain the unrestricted phases
above. Independently, the standard enables `--useful-grasps`. It computes the raw incoming-part
grasps that both exist at the mate geometry and transmit that part's insertion load, orders those
grasps first for picks and handovers, and adds the admissible cost of the forced regrasp when the
part is held by another grasp. It never removes a grasp; `--no-useful-grasps` is therefore a clean
guidance ablation. Joint target feasibility remains in the local PDB because a rest face and a set
of holders are wrench-feasible together, not generally as independent per-grasp facts. Disabling
`--fill-local-h` removes that reverse heuristic itself and is a different comparison. The all-part
passes are entered only after the cheap local graph exhausts. Since the planner returns the first
successful phase, even `fill-exact` does not imply global optimality over optional phased unions
or over the full assembly problem.

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

- unresolved edges are selected by the configured lazy edge selector;
- the standard retains execution order; experimental selectors can instead test the final
  concrete assembly endpoint first, work backward, or prioritize exact scene-action queries
  with prior failed effort; `defer-failed` retains forward order but filters all aliases of a
  failed physical query until the next epoch;
- success stores the exact joint vector and validation continues;
- failure suspends that symbolic edge for the remainder of the epoch; a physically equivalent
  alias shares its attempt cursor but may receive genuinely fresh stochastic effort;
- no other edge of an invalid incumbent is probed after the first blocking failure; and
- local search immediately explores another candidate path.

This is LazySP edge selection, not backward state-space search. The complete symbolic path and
all predecessor states already exist before validation, while each IK witness certifies one
endpoint in its exact support scene. The validation calls may therefore be reordered without
changing successors or plan costs. Standard `forward` is the measured baseline.
`terminal-first` validates the last unresolved `assemble` endpoint, then resumes forward order;
it is a concrete terminal-mode test using the candidate's already known pre-assembly state, not
a second backward transition system. `reverse` is a static suffix-first ablation, and
`fail-first` promotes only failures of the exact
`(action, scene signature)` query before falling back to reverse. It deliberately does not use
the broader action-level failure statistics that can conflate different fixed-base/support
scenes. Reverse/fail-first reduced UR5 Car Mini's validation count but not its wall time, and was
substantially worse on OpenArm because seven-slot assembly bundles are much more expensive than
many prefix refutations. `defer-failed` is different from those ordering rules: after the first
finite batch fails, the incremental search's dynamic edge-family filter excludes every symbolic
edge with that physical-query key for the remainder of the epoch. Existing tree aliases are
invalidated together; queued and regenerated aliases are rejected before they can enter the
tree. The filter is discarded at the epoch boundary, preserving the open-witness effort ladder.
A future learned selector should rank estimated failure probability
divided by expected validation cost rather than failure evidence alone.

All five selectors preserve the same finite symbolic graph and exact witness keys. They differ
only in which unresolved edge of the current complete candidate receives the next finite IK
batch. In particular, terminal-first failure is not a grasp blacklist or an infeasibility proof;
the edge reopens in the next fairness epoch, and other concrete terminal modes remain available.
Physical-query equivalence shares successful certificates and the next-attempt cursor, not a
negative verdict: a different symbolic history may therefore invest fresh random/bias effort in
the same query without replaying already-spent deterministic slots.

> **BIG TODO — replace alias-driven retries with an explicit retry/diversify scheduler.**
>
> The standard `forward` policy can revisit one physical query immediately when a different
> symbolic edge aliases it. This is useful when the next optimizer basin succeeds, but makes
> effort allocation depend accidentally on symbolic alias multiplicity. The experimental
> `defer-failed` policy removes that dependence by allowing one failed batch per physical query
> per epoch, but Chair demonstrates that hard exclusion is too coarse: its final module grew
> from 1,289 to 4,383 expansions and from 14 to 48 binds because promising second-tier retries
> were postponed behind many distinct alternatives. The replacement must operate on physical
> query identities, preserve successful certificates and fresh-attempt cursors, remain fair in
> the limit, and softly balance another effort tier against a new physical realization. It must
> not turn finite nonlinear-optimizer failure into a permanent no-good. Evaluate it on the
> matched Cube, OpenArm Cross/Car-mini, and mixed-gripper Chair suite; Chair is the mandatory
> retry-heavy counterexample.

A focal-search variant was evaluated and not retained. It kept the exact local PDB as the anchor
but selected minimum `h` among nodes within `1.5 f_min`. On UR5-limited Cross-7 this swept a large
equal-progress plateau, reached the node/memory guard before any IK, and was strictly less robust
than the ordinary weight-one anchor. This negative result is why the method exposes no focal
weight or scene-specific focal score.

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

- at most four distinct symbolic settled boundaries are generated for a transition in the
  standard profile, rather than an unbounded or nondominated frontier;
- symbolic boundary diversity covers alternate grasps, holders, placements, and robot ownership,
  while alternate joint configurations for the same symbolic state remain the separate
  `--witness-sets` mechanism;
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
STANDARD-MILESTONES(instance, width W, boundary budget K, repair budget B)
    stack <- [(initial concrete state, {base}, 0, none)]
    blacklist <- empty set
    rejected_endpoints <- map from concrete transition to a set of States
    plan <- empty sequence

    while stack is not empty and budgets remain:
        (s, I, n, via) <- top(stack)

        if I contains every part:
            if s is not hands-off and on-table:
                attempt a final settle-and-release fill
                if it fails: pop; reject the exact endpoint in via; continue
            return plan prefix through n plus final fill

        Q <- dependency-valid parts/modules whose (s,I,q) transition is not exhausted
        sort Q canonically

        for q in Q:
            D <- (s, I, q)
            if q is a connect root:
                result <- BUILD-AND-JOIN-MODULE(s, I, q, exclude=rejected_endpoints[D])
            else:
                result <- FILL-ONE(s, I, q, exclude=rejected_endpoints[D])

            if result succeeds:
                truncate plan to n; append result actions
                push the resulting settled checkpoint
                continue the outer loop

            blacklist D because its remaining endpoint set is exhausted

        if no singleton succeeded:
            try dependency-valid joint windows of widths 2 through W
            push the first successful joint endpoint

        if no window succeeded:
            pop the checkpoint
            add via.endpoint to rejected_endpoints[via.transition]
            blacklist via.transition only when K endpoints have been rejected

    return failure within the configured resources
```

The stack implements depth-first chronological backtracking. Both the parent and rejected child
are exact symbolic `State` values. Consequently a failed continuation from one boundary does not
poison the same nominal next part after a different boundary is selected. Endpoint alternatives
are lazy: a successful run computes one endpoint per transition; endpoint two is requested only
after endpoint one's suffix fails. Detached modules apply the same recursion at each internal
part and at the final join.

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

Within each widening phase, greedy best-first search orders the raw discrete state graph using the
local heuristic and cheap feasibility gates. The lazy validator evaluates expensive IK only on
the current candidate path. Greedy ordering is used only to obtain the first validated milestone
plan; it is not reused as the exact improvement order in the anytime solver.

#### Algorithm 2: bounded fair open-witness fill

```text
OPEN-FILL(start, local goal, maximum epochs E)
    preserve solve-wide successful witness pool

    for epoch = 1 through E:
        suspended <- cheap-refuted edges plus empty open set
        deferred-queries <- empty set
        engine <- greedy local best-first search excluding suspended edges and every
                  symbolic alias whose physical key is in deferred-queries

        while engine has a candidate path:
            E <- unresolved IK edges of the path
            order E using the selected policy (standard: forward)
            for each edge e in E:
                validate e with its next effort batch

                if validation succeeds:
                    retain the exact joint vector
                else:
                    if selector is defer-failed:
                        add physical-key(e) to deferred-queries
                        invalidate all matching tree aliases
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

On suffix failure, the checkpoint stack is popped and only its exact endpoint is rejected. The
same local search is then rerun with that state excluded from its goal set. The standard requests
up to four distinct endpoints before blacklisting the concrete `(s,I,q)` transition. This policy
also recurses through fixed-order detached-module fills. It is bounded diversity, not an
unbounded Pareto frontier.

### 6. Witness certificates

Every validated IK action returns a joint vector. Successful open-witness queries retain that
vector, not merely a Boolean verdict. After release hoisting, the vectors are aligned with the
final action sequence; any edge whose predecessor scene changed is certified again.

The saved plan records one search or optimized witness per IK action. Missing required witnesses
cause export to fail. Optional witness optimization may improve a configuration but retains the
search witness as a feasible fallback. Motion planning and replay therefore consume persisted
certificates instead of rerunning stochastic IK.

The standard witness optimizer retains the clearance-floor epsilon-constraint used by earlier
artifacts. For each step it scores the finite certified candidate set by collision clearance.
A dynamic program then minimizes joint-space continuity cost across the complete keyframe chain.
Let `x_0` be the minimum number of large joint transitions attainable without an additional
clearance floor. The optimizer chooses the highest tested normalized clearance fraction whose
chain has at most `x_0 + B` transitions; standard sets `B=0`. Thus clearance may not purchase a
discretionary branch crossing, while the 5 mm search certificate remains the only absolute hard
clearance gate.

The motion metric uses actual bounded-joint displacement. It neither wraps a difference modulo
2 pi nor removes a robot-specific wrist coordinate. For redundant arms, the candidate generator
computes the pose-task Jacobian at a certified keyframe, takes its SVD null space, samples
deterministic joint-limit-aware continuation seeds, and projects each seed back onto the exact
tool-pose/collision manifold. Only candidates which pass the same collision, clearance,
joint-margin, and pose-residual gates join the dynamic program or exported goal sets. This is
candidate generation, not a relaxation of feasibility. The optimizer can remove a large motion
only when the certified set actually contains a continuous alternative; unavoidable transitions
remain visible in its report.

### 7. Guarantees and limitations

The current implementation supports the following claims:

- Every returned plan passes the enabled discrete, wrench, collision/clearance, and IK gates.
- Successful IK actions carry persisted joint-space witnesses.
- Under Proposition 1's assumptions, the uniform local-PDB value is an admissible local
  abstraction heuristic.
- Uniform grasp projection does not remove raw concrete actions.
- The outer solver respects the supplied dependencies, handles connect modules, and
  chronologically backtracks among high-level milestone choices and up to the configured number
  of exact settled endpoints per concrete transition.
- Greedy queue ordering and removal of the 200-candidate cutoff give exhaustive local repair
  within a fixed phase, provided its finite graph, edge oracle, and implementation guards are
  exhausted. This is an exhaustion/coverage statement, not a least-cost statement.
- Before a singleton fill is blacklisted, the standard exhausts the complete four-yaw placement
  lattice in both the local action scope and the all-part manipulation scope across its
  unbraced/braced phases. Thus neither locality nor a skipped configured yaw is a reason to
  reject the milestone; finite continuous effort and the remaining guards still prevent this
  from being an unconditional infeasibility proof.

The following are not valid unconditional claims:

- The complete milestone planner is not globally complete.
- Greedy standard fills do not minimize local or global action cost. `fill-exact` also does not
  make the assembly plan globally optimal, and phased widening means it is not necessarily
  optimal over the union of local phases.
- Two open-witness epochs are not an IK infeasibility proof.
- Chair support and backtracking do not imply enumeration beyond the finite boundary budget or
  over every module-internal order.
- Width two cannot express all manipulation interleavings across three or more milestones.
- Finite node, repair, window, PDB, and wall-time guards can terminate a feasible instance.
- A nonlinear IK call may exceed a cooperative deadline before returning.

Even with unlimited open-witness epochs, the finite boundary budget and fixed module-internal
order remain. A plausible probabilistic-completeness path would require fair unbounded IK effort,
unbounded or nondominated concrete endpoints per milestone, search over module internal orders,
unbounded width/resource limits, and a reopening concrete graph search. That is future work, not
a property of the current standard.

Grasp widening removes one narrower source of false finality: failure at the initial K=20 no
longer means every generated grasp was considered. The sampler has a deterministic K-independent
scene-reserved master order: its head prioritizes available per-kind pick, mandatory-mate, and
intermediate/final component-support scene facts, then its tail retains the hybrid
canonical/sampled ordering. Epochs therefore expose nested
prefixes and preserve action identity. The configured K=96 bound and the sampler's finite scene-probe pool still make
this finite coverage, not continuous-grasp completeness; the log says when the master hash stops
changing. Combined with finite IK/node budgets, a final no-plan result remains “not found under
the reported budgets.”

### 8. Optional extensions and ablations

The following mechanisms remain available but are not part of the selected standard search:

- `--milestone-oracle global`: build a full signature-collapsed order-ideal database and use it
  for outer part ordering and fill hints;
- `--milestone-oracle chain`: restrict that global abstraction to a declared sequence; detached
  modules are rejected;
- counting-only fills: disable `--fill-local-h`;
- terminal IK failure: disable `--open-witness`;
- weight-one local A*: use `--fill-exact --no-fill-greedy`;
- inherited weighted local search: use `--no-fill-exact --no-fill-greedy --weight W`;
- direct skill skeletons: enable `--fill-skeleton-first`; and
- `--milestones-anytime T`: use milestones for an incumbent, then repeatedly re-fill consecutive
  milestone neighborhoods through the full horizon by default (`--lns-window K` caps it).
  `--fill-window` remains the separate initial-plan recovery width. A neighborhood exposes
  exactly the parts touched by the destroyed incumbent
  actions (including both sides of an assembly); all other parts are frozen. Its refill uses the
  ordinary heap-based A* engine with a strict incumbent cost bound. Exact heuristic and cheap-edge
  queries are memoized, and successor/physical-collision caches admit full keys only on their
  second access. If candidate IK rejects an edge, the refill hands off once to the existing
  incremental LazySP engine; subsequent edge deletions repair the affected subtree. Thus there is
  no repair graph is allocated for a pass that never reaches an IK refutation. Each window first
  runs greedy best-first search below the incumbent span cost, then weight-one A* below the greedy
  cost if one was found (otherwise below the incumbent cost). Positive IK certificates and
  accumulated open-witness effort persist between passes. Completed-pass allocator arenas are
  released before the next consultation. Strict improvements replace the window and restart the
  sweep. A finite deadline is shared fairly over the remaining windows; greedy gets at most half
  of its window slice so exact cannot be starved, and unused greedy time flows to exact. Every non-final exact
  splice additionally gets a lazy reverse PDB rooted at the
  incumbent's projected target and restricted to the destroyed ideal chain. It reuses the local
  fill's signature quotient and is expanded only below the strict incumbent cost. Forward A* uses
  the maximum of that direct distance and the global-landmark difference. Reverse exhaustion can
  therefore certify a window unimprovable without concrete forward expansion; hitting the PDB
  state guard merely falls back to the landmark heuristic. The final window retains the ordinary
  set-valued assembly goal. Per-window logs report scope, bound, endpoint-PDB work, outcome,
  forward-search work, and IK failures.
  When
  open-witness validation is on, this explicit finite-deadline mode removes the ordinary
  two-epoch cap until the milestone-phase deadline and restores it before LNS. The LNS layer is
  incumbent-preserving. Exhausting its default final `k=n` pass proves the incumbent optimal in
  the finite incumbent-induced scope, but not over action families omitted by the surrounding
  milestone/widening schedule; reaching the global admissible lower bound remains a direct global
  certificate.

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
| `--fill-greedy` | heuristic-only local candidate order and exhaustive repair to frontier/budget |
| `--no-fill-exact` | do not run a redundant local weight-one confirmation pass |
| `--fill-incumbent-first` | greedy validated incumbent, then incumbent-bounded weight-one local search (experimental) |
| `--no-fill-policy-guide` | standard keeps optional Bellman-frontier widening off. With ordinary `--milestones`, `--fill-local-h`, and a local grasp or yaw fibre, the guide runs only after compact and normal numeric `K -> K+B` attempts fail; it then breadth-first schedules bounded loose/support grasp batches and stable-rest yaw orbits across the settled optimal policy DAG before returning to ordinary widening. |
| `--no-fill-skeleton-first` | use the general local graph directly |
| `--milestone-boundaries 4` | lazily retry distinct exact settled endpoints after suffix failure |
| `--fill-scope-widening` | after local exhaustion, retry with all-part manipulation |
| `--placement-yaws 4` | finite quarter-turn representatives of the free horizontal-table yaw |
| `--placement-yaw-widening` | expose 1, then 2, then 4 yaws per failed local fill; eager N=4 remains an ablation |
| `--sig-components press` | force-transmission grasp signature, plus mandatory tool mask |
| `--useful-grasps` | prefer raw mate-and-press-viable incoming grasps; price forced regrasp by one |
| `--grasp-sampler` + `--grasp-sampler-scenes` | immutable scene-reserved hybrid master: available per-kind pick/assemble representatives precede the diversity-ordered tail; exact IK and wrench remain authoritative |
| `--grasp-widen` | after a no-plan return, expose the next stable scene-reserved grasp prefix and retry; no finite miss becomes a grasp no-good |
| `--grasp-widen-batch 8` | additional IDs exposed per widening epoch |
| `--grasp-widen-max 96` | finite upper request; stop earlier when the evaluated master hash no longer changes |
| `--open-witness` | suspend and fairly reopen finite stochastic IK failures |
| `--open-witness-batch 7` | maximum ordinary tier and atomic assembly bundle |
| `--open-witness-epochs 2` | measured finite operational effort per local fill |
| `--lazy-edge-selector forward` | execution order; terminal-first/reverse/fail-first/defer-failed are ablations |
| `--fill-window 2` | built-in joint-fill widening limit |
| omitted `--lns-window` | anytime LNS neighborhoods grow through the full milestone horizon |

`--grasp-local-widen` is the experimental lazy-column alternative to the standard whole-search
`--grasp-widen`. It builds one finite immutable sampler master. Its first resumable milestone
session retains the cheap historical schedule: compact prefixes, at most one incoming-part
`K -> K+B` expansion for a concrete fill, then ordinary milestone repair and backtracking. If
that compact session returns without a deadline, the iterator schedules one fresh eager
finite-master session with every part cap at the evaluated master limit. This later session is
shared by ordinary solves and resumable motion-plan candidate generation, so a compact-prefix
miss alone cannot suppress the configured finite master. It does not remove the existing finite
node, repair, boundary, window, PDB, or IK-effort limits, and therefore is not a global
completeness claim. The local PDB stays on the compact projection, so later columns are
“unknown” rather than eagerly inflating every reverse state space. This is finite effort
scheduling, not grasp pruning.

`--fill-policy-guide` is an optional refinement of that local-column schedule, not a second
backward validator. It is available only to ordinary `--milestones` with `--fill-local-h` and at
least one finite fibre to schedule: `--grasp-local-widen` for raw grasp columns and/or
`--placement-yaw-widening` for yaw orbits. The separate outer `--grasp-widen` restart is not its
raw-column interface. The normal compact local PDB and concrete fill run first with their active
prefixes, followed, when local grasp columns are enabled, by the usual single numerical
incoming-part `K -> K+B` expansion.

Only after that baseline stalls does the guide query a reverse PDB over the relevant finite
master. It breadth-first traverses all settled Bellman-optimal successor branches, subject to a
finite optional traversal guard, and emits their *atomic* finite-coordinate demands in minimum
abstract-depth order. A grasp atom is one signature fibre: `loose` for a free held label, or
`support` for a label on the settling component, for any relevant part. `support` deliberately
does not name a concrete action; that contact may be inherited, arise through `add_grasp`, or be
reached by a handover. A yaw atom is one stable-rest class encountered on a held-to-rest
transition.

Each distinct grasp fibre receives one bounded stable-master-ID batch during the rescue pass.
After a failed forward probe, that batch remains active and the next Bellman demand is added;
later raw members remain for ordinary numerical and finite-master widening. Yaw demands are then
added one rest orbit at a time while retaining both earlier yaw classes and policy-selected raw
columns. The active set therefore grows monotonically and may satisfy a route requiring multiple
missing coordinates, but it never eagerly opens the full raw-grasp/yaw Cartesian product.

This is deliberately a scheduling interface rather than a second geometric model. The reverse
PDB performs no IK or KOMO. The guide applies no analytic/SSIK score, coverage test, or extra
policy-specific feasibility gate: raw variants are ordered by their stable master IDs, and normal
cheap checks plus all continuous validation remain lazy forward checks. Every newly exposed
batch/orbit gets one bounded forward probe (one open-witness epoch when enabled). Once the finite
Bellman demand frontier is exhausted, the standard generic grasp/yaw/scope widening schedule
resumes. Search returns to its compact yaw prefix after the rescue pass, while a public yielded
candidate temporarily exposes the full finite yaw table so replay, witness collection, and
`--mp-open-plans` can resolve its placement ids.

The frontier contains all equal-cost Bellman branches, not a hand-selected relaxed-cost slack.
Consequently, it removes tie-breaking fragility but does not reach a higher-cost abstract detour
when every shortest relaxed route is IK-spurious. In that case generic widening remains the net.
A future counterexample-guided version could temporarily remove an exhausted scene-specific
fibre and recompute the reverse policy, but bounded stochastic IK failure is not a sound permanent
abstract no-good and is not treated as one here.

If that probe fails, the later eager finite-master session still exposes every configured finite
grasp column after the compact resumable session exhausts. The policy therefore only changes the
order in which finite coordinates receive effort; it neither prunes a concrete action nor turns a
backward abstraction result into a feasibility or infeasibility certificate. Its finite-alphabet
coverage remains qualified by the existing node, repair, endpoint-boundary/window, deadline, and
finite witness/IK limits, so it is not a global-completeness theorem. The selected `standard`
profile keeps this experimental option disabled. A matched guarded Car Mini/Cross/Chair suite
changed neither plan, cost, IK work, nor peak memory because every compact fill succeeded and the
guide therefore never activated. Forced-miss regressions cover alternative optimal branches,
monotone cross-branch and grasp-plus-yaw accumulation, replay-safe cap publication, and generic
fallback; a natural benchmark demonstrating repeated post-compact activations is still needed
before treating this ordering as a performance result.

Bounded fair open-witness scheduling is part of `--search-profile standard`. The analytic
warm-start cap remains an explicit aggressive ablation rather than another named profile.

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
greedy versus weight-one local search, geometry-only versus wrench/noisy insertion, and
`--milestone-boundaries 1` versus `4`. The last comparison measures the bounded boundary repair
introduced after the original claim-once implementation.

There is no global-oracle-hint ablation for the selected standard: `--milestone-oracle structural`
constructs no global oracle. The narrower raw incoming-grasp guidance is ablated with
`--no-useful-grasps`. `--no-fill-local-h` is instead a much larger
counting-versus-reverse-heuristic ablation and should not be mislabeled as hint removal.

The manipulation-locality ablation keeps reverse-PDB guidance, greedy fill ordering, and
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
on the tested sequence and dependency-graph assemblies, including Chair's detached module. It now
reopens exact symbolic milestone boundaries lazily, but its finite boundary/IK/resource budgets
remain deliberately weaker than global or probabilistic completeness.
