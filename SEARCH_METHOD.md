# Abstraction-Guided Milestone Search for Multi-Robot Assembly

This document describes the search implemented by the current `--milestones` and
`--milestones-anytime` paths. It is deliberately presented twice. Part I gives an intuitive
account of what the planner is doing and why. Part II gives an implementation-faithful methods
description suitable as the basis of a paper section.

The description reflects the code as of 2026-08-22. In particular, it includes signature-
collapsed backward search, wrench and geometry facts, adaptive-width milestone fills, lazy and
open IK validation, chronological repair, DAG modules, the anytime wrapper, and exact witness
persistence.

## Part I: intuitive explanation

### The central idea: plan backward, execute forward

Assembly planning is difficult because a forward planner must make many choices before it knows
whether they were good:

- which part to assemble next;
- which robot should carry it;
- which grasp should be used;
- whether the growing assembly should remain on the table or be held;
- whether another robot must brace it;
- whether a handover or temporary placement is needed; and
- whether all of those choices have collision-free robot IK.

A flat search mixes all of these decisions. Its frontier becomes the cross-product of part order,
grasp, robot assignment, placement face, support state, and manipulation history.

The milestone planner separates the problem into two questions:

1. **What should the assembly look like at the next clean checkpoint?**
2. **How do the robots get from the current checkpoint to that one?**

The first question is answered by a small model solved backward from the finished assembly. The
second is answered forward, using the real manipulation actions, collision checks, wrench checks,
and robot IK.

The backward model can be understood as taking apart an idealized finished product. Starting at
the goal, it asks:

- which part could have been the last one inserted?
- which grasp could have carried it?
- did the target need to be held or braced?
- could the remaining support resist gravity and the insertion force?
- could the relevant grippers fit around the assembly?
- could the part have been picked from a stable table face, or would it need a regrasp?

Solving all such disassembly possibilities once produces a distance-to-go table. The forward
planner then uses that table as an assembly manual: among the currently legal next parts, prefer
the one whose completion leaves the cheapest abstract route to the goal.

```text
assembly + grasp models
          |
          v
  grasp signatures and cached feasibility facts
          |
          v
 backward disassembly Dijkstra  ---->  distance-to-go table
                                          |
current concrete checkpoint  ------------+
          |
          v
 rank next parts -> forward local fill -> validate IK witnesses
                           | success              | failure
                           v                      v
                    new checkpoint       try another part / backtrack
                           |
                           v
                  optional global anytime improvement
```

### What the backward state remembers

The abstract backward state is

```text
(assembled parts, grasps on the main assembly, table face or held-in-air, held loose parts)
```

It intentionally forgets exact arm configurations and usually forgets robot identities. Those
details are expensive and are checked later by the forward fills. It retains the information that
changes the qualitative feasibility of future assembly:

- which dependencies are already assembled;
- how many grippers are occupied;
- where the main assembly is supported;
- which kinds of grasps are holding or bracing it; and
- which kinds of grasps are carrying loose parts.

This is a relaxation: if a real manipulation is possible, its projection should also be possible
in the abstract model. The abstract distance is therefore a lower bound on the real remaining
action cost. Conversely, if the fully constructed abstract model cannot reach the goal from a
projected state, then the real state cannot reach it either under the modeled actions and facts.

### Why grasp signatures are needed

Keeping every raw grasp ID makes the backward table too large. Forgetting grasps entirely makes it
too weak: a grasp that can pick a part from the table may be unable to transmit its insertion
force, while another grasp has the opposite properties.

The current compromise groups grasps into **fact signatures**. Two grasps are interchangeable in
the backward model when they agree on the selected facts. The available components are:

- `press`: can the grasp transmit the part's insertion load?
- `pickface`: from which stable table faces can the gripper reach the grasp without colliding
  with the table or part?
- `contact`: what contact point and closing-axis class does the grasp use?
- `solohold`: can this grasp support the complete assembly alone in the air?
- `braceside`: on which side of the assembly can the grasp act as a useful brace?

Intuitively, the signature is a short list of a grasp's “qualifications,” not its geometric
identity. `press` asks whether it can do the insertion job; `pickface` asks whether it can be
acquired from the part's present pose; `contact` retains where and along which axis it acts;
`solohold` asks whether it can be the only carrier; and `braceside` asks which insertion loads it
can oppose. Grasps with the same selected qualifications share one backward label, while the
forward search still chooses and validates an actual raw grasp.

The default is `--sig-components press`, because force transmission is the most useful
discriminator in the tested wrench-heavy assemblies. With mixed gripper types, the gripper-kind
usability mask is always part of the signature. The expensive per-kind wrench facts—including
mixed-gripper `press` and `solohold` evaluations—are enabled by `--mixed-wrench-h`; without that
flag the mixed-gripper abstraction uses its faster topological fallback.

When a class contains several raw grasps, a fact is accepted if any member realizes it. This OR
rule is important: collapsing labels may make the abstraction optimistic, but cannot make it more
restrictive than the raw-grasp abstraction.

### What happens in the forward pass

Suppose the current checkpoint contains the base plus some assembled parts. The planner lists all
precedence-addable parts and sorts them using the backward distance after hypothetically adding
each part and settling the grown assembly.

For the best candidate, a local fill search is run. It may:

- pick the new part;
- pick or continue holding the target assembly;
- hand over or reorient the new part;
- add a brace;
- insert the part;
- release unnecessary grasps; and
- place a loose part or the growing assembly when placement is enabled.

The fill ends at a clean checkpoint: the requested part is assembled, no unrelated loose part is
left in a gripper, and the relevant assembly is either on the table or held by at most one robot.
Milestones therefore do **not** require placement to be disabled and do not force the assembly to
stop on the table after every insertion. A hold-through solution and a set-down solution are both
available.

The fill first tries narrow hints from the abstraction, but those hints are never trusted as a
complete description of concrete reachability. The widening order is approximately:

1. hinted insertion grasps, unbraced;
2. all insertion grasps, unbraced;
3. hinted insertion and brace grasps;
4. all grasps, with bracing.

Thus a useful abstract hint saves work, while a wrong hint is followed by a full local phase.

### Lazy IK and open witnesses

Robot IK is much more expensive than discrete successor generation. A fill therefore searches
the discrete graph first and spends IK primarily on edges of a candidate path.

The established lazy path treats a failed IK query as a permanent edge refutation within the
solve. It incrementally repairs the local search tree after deleting failed edges.

`--open-witness` changes the interpretation of a failed stochastic IK attempt. Failure means
“not found with this effort yet,” rather than “this edge is impossible.” Each blocking edge gets
a bounded batch:

- ordinary skills receive 2 attempts, then 4, then up to `--open-witness-batch` on later visits;
- assembly receives one atomic structural batch, normally six orientation-biased solves plus one
  unbiased solve;
- a failure suspends the edge for the rest of the fairness epoch;
- the invalid candidate plan is abandoned immediately, without probing its suffix; and
- a later epoch reopens suspended edges with additional effort.

Successful witnesses are retained exactly. Once a final discrete plan is selected, its successful
joint vectors are aligned with the final action order and serialized. Witness optimization may
improve them, but may not erase them. Replay therefore does not have to rediscover a stochastic IK
solution.

The current epoch limit is still an effort limit. If it is reached, the local fill reports failure
to the milestone layer even though the individual IK query remains logically open. Consequently,
the current open-witness policy is an incomplete resource-bounded search policy, not an
infeasibility proof.

### What repair means

If a candidate part cannot be filled, the high-level decision

```text
(current assembled set, chosen next part)
```

is blacklisted. The planner tries the next part suggested by the backward table. If every choice
at the current checkpoint fails, it pops the previous checkpoint, blames the decision that led to
the dead end, and tries another branch. This is chronological backtracking with conflict records at
milestone granularity.

This is much cheaper than flooding the entire concrete manipulation graph, but the conflict is
coarse. Two concrete checkpoints can share the same assembled set while differing in holder,
placement, or robot configuration. Blacklisting only by assembled set and next part assumes those
differences do not rescue the transition. That is one reason the milestone algorithm is not a
globally complete search at its default width.

When every one-part fill fails, the planner can widen the checkpoint. With the default
`--fill-window 2`, it jointly searches for two new parts before backtracking. Such a joint fill can
express manipulation glue that must overlap across two nominal assembly steps. In the conceptual
limit where the window contains every remaining part, the decomposition becomes the flat search;
the current implementation normally stops at width two and also caps the number of windows tried.

Assemblies with explicit module joins receive an additional decomposition: a detached module is
built internally and then joined to the main component as one milestone.

### The anytime version

`--milestones-anytime T` uses the milestone plan as a fast incumbent and then spends remaining
time trying to improve it in the full concrete state space.

The budget is used as follows:

1. Reserve `min(60 s, 0.25 T)` for a fallback or improvement phase.
2. Run adaptive-width milestones until the earlier milestone deadline.
3. If milestones found a plan of cost `U`, run memory-reduced partial-expansion weighted search
   while pruning every state with `g + h >= U`. Only a strictly cheaper plan can replace the
   incumbent.
4. If milestones did not find a plan and more than five seconds remain, run a greedy
   partial-expansion search in the full concrete state space as a feasibility net.
5. Return the best validated plan found so far.

This prioritizes “find something reliable first, improve it second.” The incumbent cannot be lost
when the improvement pass times out.

The repository also contains `--anytime`, a different algorithm: restarting weighted search with
weights `infinity, 5, 2, 1.5, 1`. It does not use milestone decomposition. The method described in
this document is the milestone-specific anytime wrapper.

## Part II: paper-style methods description

### 1. Problem formulation

Let an assembly instance contain parts \(P\), robots \(R\), a designated base part
\(p_0\), a mate dependency graph, stable placement faces, a grasp database, and optionally a
quasi-static wrench model. A discrete state \(s\in\mathcal S\) records:

- the kinematic parent of every part;
- mate, table, and grasp support contacts;
- the table placement face and anchor of every resting component; and
- robot-to-part grasp ownership.

The action set is

\[
\mathcal A=\{\texttt{pick},\texttt{place},\texttt{assemble},\texttt{release},
\texttt{add\_grasp},\texttt{handover},\texttt{inhand\_reorient}\},
\]

with `set_down` and `stow` in configurations that enable them. All actions have unit cost except
`handover`, which has cost two because it represents a grasp addition followed by release:

\[
c(a)=\begin{cases}2,&a=\texttt{handover},\\1,&\text{otherwise.}\end{cases}
\]

A transition must satisfy the discrete preconditions, support and wrench gates, cheap geometric
collision gates, and—when real IK is enabled—a robot-level feasibility predicate returning a
configuration witness \(q_a\). The objective is to find a feasible action sequence from the
initial scattered state to a goal in which all required mates are present and the requested goal
support conditions, such as on-table and hands-off, hold.

### 2. Signature-collapsed disassembly abstraction

We define a projection \(\pi:\mathcal S\rightarrow\bar{\mathcal S}\). An abstract state is

\[
\bar s=(I,G,\phi,L),
\]

where:

- \(I\subseteq P\) is the dependency-closed, base-containing assembled component;
- \(G\) is a tuple of grasp labels currently supporting or bracing \(I\);
- \(\phi\in\Phi(I)\cup\{H\}\) is a stable table face or the held-in-air symbol \(H\); and
- \(L\) is the tuple of held loose parts together with their grasp labels.

Robot identities are removed, while total and per-tool-kind gripper capacities are retained. In
mixed-tool scenes, each grasp label has a tool-kind mask. The optional mixed-wrench model adds
the more expensive per-tool wrench facts; it is not required for the capacity constraints.

#### 2.1 Grasp-label construction

For each part \(p\), raw grasps \(g\in\mathcal G_p\) are partitioned by a signature

\[
\sigma(p,g)=\big(\sigma_{\mathrm{tool}},\sigma_{\mathrm{press}},
\sigma_{\mathrm{pick}},\sigma_{\mathrm{contact}},
\sigma_{\mathrm{hold}},\sigma_{\mathrm{brace}}\big),
\]

restricted to the enabled components. The components are:

| Component | Meaning | Primary role in backward search |
|---|---|---|
| tool mask | gripper kinds able to use the grasp | prevents vacuum-only and finger-only grasps from being merged |
| `press` | feasibility of transmitting the part's own insertion wrench | separates pickable grasps from insertion-capable grasps |
| `pickface` | table-aware pickability vector over stable rest faces and tool types | prices reorientation and rejects finger-through-table picks |
| `contact` | quantized contact center and sign-symmetric closing axis | preserves force lever and brace-contact class while merging irrelevant wrist tilts |
| `solohold` | ability to hold the completed assembly alone | identifies viable terminal carriers and handover recipients |
| `braceside` | closing-axis alignment with mate directions | distinguishes grasps that can react forces on different sides |

For a class \([g]\), every context-dependent fact is evaluated by existential lifting,

\[
F([g])=\bigvee_{g'\in[g]}F(g').
\]

This can introduce abstract transitions that no single raw grasp realizes consistently across all
contexts, but it cannot delete the projection of a real transition. Signature collapse is thus an
additional relaxation.

#### 2.2 Cached abstract facts

The backward model uses five fact families.

1. **Dependency and mate facts.** A part can be removed backward only if it is maximal in the
   current ideal. A forward on-table mate must be compatible with the selected rest face.
2. **Contextual grasp validity.** A grasp must occur in the precomputed grasp table for the
   relevant subassembly. Removing parts is treated monotonically as removing obstacles.
3. **Table-aware pick facts.** For each subassembly, rest face, grasp, and tool geometry, a
   floating end effector is placed at the grasp and checked against the table and subassembly.
4. **Wrench facts.** Crafted states are evaluated with the same equilibrium LP used by forward
   support gates. These test ordinary holding, release stability, insertion-force transmission by
   the moving part, and reaction-force support by the target and its braces.
5. **Optional geometric coexistence facts.** With `--geo-facts`, a labeled state relays backward
   only if all abstract grippers can coexist around the assembly and table. Class-level facts use
   an existential assignment of raw variants.

The abstraction also mirrors staged-mate support constraints, stable face sets, bracing, in-hand
reorientation, and tool-kind capacity when those features are active.

### 3. Backward shortest-path construction

All abstract states satisfying the concrete goal projection are inserted into a multi-source
priority queue at distance zero. The table is expanded by inverse actions using Dijkstra's
algorithm because handovers cost two.

The principal inverse transitions are:

| Forward action | Backward predecessor operation | Cost and gate |
|---|---|---|
| assemble part \(p\) | remove maximal \(p\) from \(I\); move its insertion grasp from \(G\) to \(L\) | 1; dependency, rest-face, target-support, insertion and tool-budget facts |
| pick loose part | remove its labeled entry from \(L\) | 1; pickable from at least one stable face |
| lift main assembly | change \(H\) with one holder to a stable face with no holder | 1; table-aware pick fact |
| place main assembly | change a stable face to \(H\) and add the placing holder | 1; free gripper capacity |
| release | add backward the grasp that existed before release | 1; the forward post-state must remain wrench-stable |
| add brace | remove one grasp from \(G\) while retaining at least one holder | 1; predecessor remains support-feasible |
| handover | swap the sole assembly or loose-part grasp label | 2; transient gripper capacity |
| in-hand reorient | swap the loose-part grasp label | 1; only when the skill is enabled |
| place loose part | add a held loose-part label | 1; free gripper capacity |

Only states co-reachable from an abstract goal are generated. Let \(d(\bar s)\) be the resulting
distance. The heuristic used by concrete search is

\[
h(s)=\max\{h_{\mathrm{count}}(s),d(\pi(s))\},
\]

where \(h_{\mathrm{count}}\) charges one remaining assembly action for every unmated part and
one pick for every such part not already held.

#### Proposition 1: admissibility of the abstraction

Assume that every concrete transition has an abstract counterpart of no greater cost and that
all enabled fact gates are necessary conditions for the corresponding concrete transition. Then

\[
d(\pi(s))\leq J^*(s)
\]

for every concrete state \(s\), where \(J^*\) is the optimal remaining concrete action cost.

**Argument.** Project any feasible concrete suffix into the abstract graph. By assumption, each
concrete edge maps to an abstract edge of no larger cost, so the projected suffix upper-bounds the
abstract shortest-path distance. Signature OR-lifting, anonymous robots, existential rest faces,
and omitted arm configurations only add possibilities and therefore preserve the inequality.

An absent table entry is consequently a sound dead-end certificate only when the backward table
was built without an incompatible distance cap or state-limit failure and the fact gates satisfy
the necessity assumption.

### 4. Abstraction-guided milestone search

The forward search maintains a stack of settled checkpoints

\[
(s_k,I_k,n_k,\delta_k),
\]

where \(s_k\) is the concrete state, \(I_k\) is its assembled ideal, \(n_k\) is the current
plan length, and \(\delta_k\) is the high-level decision that reached it.

For a precedence-addable part \(p\), the high-level oracle is

\[
O(I,p)=\min_{\phi\in\Phi(I\cup\{p\})}
d(I\cup\{p\},\emptyset,\phi,\emptyset).
\]

Candidates are tried in ascending \((O(I,p),\mathrm{partIndex}(p))\) order. The empty grasp sets
in this oracle represent an idealized settled checkpoint; they are used for ranking and do not
constrain the concrete fill endpoint.

#### Algorithm 1: milestone search

```text
BACKWARD-MILESTONE-SEARCH(instance, width K, repair budget B)
    D <- backward Dijkstra table over signature-collapsed states
    stack <- [(initial state, {base}, plan length 0, no parent decision)]
    blacklist <- empty set
    plan <- empty sequence

    while stack is not empty and budget remains:
        (s, I, n, via) <- top(stack)

        if I contains all parts:
            if s is not a goal:
                try a final place/release fill
                if it fails: pop and blacklist via; continue
            return validated plan

        Q <- precedence-addable parts not blacklisted at I
        sort Q by the backward oracle O(I, p)

        for p in Q:
            subplan <- FILL-ONE(s, I, p, D)
            if subplan succeeds:
                truncate plan to n; append subplan
                push its settled endpoint and enlarged ideal
                continue outer loop
            blacklist (I, p)

        if every one-part fill failed and K >= 2:
            for widths k = 2, ..., K:
                enumerate precedence-valid k-part windows
                try one joint local fill for each window
                on success push the enlarged checkpoint and continue outer loop

        pop the dead checkpoint
        blacklist the decision via that created it

    return failure
```

#### 4.1 Local fill search

`FILL-ONE` searches only actions touching \(I\cup\{p\}\), except that release of an inherited
holder is always permitted. Its goal requires:

1. \(p\) is mated;
2. no unrelated loose component remains held; and
3. the component being built has at most one holder.

The local counting heuristic is

\[
h_{\mathrm{fill}}(s)=h_p(s)+N_{\mathrm{loose}}(s)+
\max(0,N_{\mathrm{holders}}(s)-1),
\]

where \(h_p\) is 0 when \(p\) is mated, 1 when it is held in a mate-capable grasp, and 2 when it
still requires a pick or a forced regrasp. For ordinary non-module fills, the full-assembly
abstraction is shifted by the expected post-milestone distance,

\[
\tilde h_D(s)=d(\pi(s))-O(I,p),
\]

and joined with the local heuristic to provide a gradient toward this fill rather than merely
toward the final assembly. This shifted term is an ordering device inside the decomposed fill;
the code explicitly does not rely on it for a global optimality guarantee.

Cached backward wrench facts additionally provide candidate insertion grasps and brace contacts.
The fill searches hinted unbraced, full unbraced, hinted braced, and full braced phases in that
order, subject to cases where the relaxed wrench model proves an unbraced insertion unavailable.

#### 4.2 Lazy continuous validation

Let \(\hat{\mathcal G}\) be the local discrete graph after cheap geometric filtering and let
\(V(e)\) be expensive IK validation. The established fill repeatedly finds a shortest weighted
candidate path in \(\hat{\mathcal G}\), evaluates \(V\) only along that path, deletes failed
edges, and incrementally repairs the affected search-tree subtree. Successful and failed queries
are shared across fills through scene-signature caches.

The open-witness variant replaces terminal stochastic failure with a solve-scoped effort state
\((n_e,v_e)\) per scene-action query. During each fairness epoch, an unresolved edge receives its
next bounded effort batch and is suspended after failure. The discrete planner then seeks an
alternative candidate. At the next epoch, suspended edges reopen. Successful queries store both
the verdict and exact configuration witness.

This is related to lazy shortest-path search, but its resource-bounded IK oracle is not a binary
edge oracle: `unknown after this effort` is distinct from `geometrically refuted`. The current
implementation preserves that distinction in the IK pool but collapses an exhausted local epoch
budget to a failed milestone fill at the high level.

#### 4.3 Modules and adaptive-width fills

For a connect-root part, the planner first builds its dependency module as a detached component,
allowing it to remain held between module-internal fills, and then performs the joining fill.

If all one-part fills at a checkpoint fail, width-\(k\) search jointly adds a precedence-valid
tuple of \(k\) parts. Increasing \(k\) weakens the decomposition commitment by permitting their
manipulation actions to interleave. The default \(K=2\) addresses common two-part couplings;
\(K=|P\setminus I|\) is conceptually the undecomposed limit, subject to implementation budgets.

#### 4.4 Chronological repair

Failures produce conflicts keyed by \((I,p)\), or \((I,W)\) for a joint window. When a node has
no remaining candidate, the stack is popped and the parent decision is blacklisted. This repair
is coarse but inexpensive: it changes assembly order rather than exploring all manipulation
interleavings below a failed commitment.

### 5. Anytime milestone search

Given total nominal budget \(T\), define

\[
T_r=\min(60,0.25T),\qquad T_m=T-T_r.
\]

The first phase runs milestone search until \(T_m\). If it returns an incumbent \(\xi\) with
cost \(U\), the remaining time is passed to partial-expansion weighted graph search over the full
concrete state space with the strict improvement bound

\[
g(s)+h(s)<U.
\]

Partial expansion retains only the current \(f\)-band of an expanded state's successors and
requeues the parent at the smallest deferred band. It trades successor regeneration for a smaller
OPEN list. If the improvement search returns a plan of cost below \(U\), it becomes the new
incumbent; otherwise the milestone plan is returned unchanged.

If milestone search fails or reaches its deadline without an incumbent, the reserved budget is
used for greedy partial-expansion search in the full concrete graph, provided more than five
seconds remain.

#### Algorithm 2: anytime wrapper

```text
ANYTIME-MILESTONES(instance, total time T, weight w)
    reserve <- min(60 seconds, 0.25 T)
    incumbent <- MILESTONES(deadline = now + T - reserve)

    if incumbent is absent:
        if more than 5 seconds remain:
            return PEA-GREEDY(full concrete graph, final deadline)
        return failure

    U <- cost(incumbent)
    if more than 10 seconds remain:
        candidate <- PEA-WEIGHTED-A*(full graph,
                                     prune when g + h >= U,
                                     final deadline)
        if candidate exists and cost(candidate) < U:
            incumbent <- candidate

    return incumbent
```

The timeout is cooperative: deadlines are checked between search operations, not by interrupting
an individual nonlinear IK solve.

### 6. Feasibility certificates and witness persistence

Every IK-validated action produces a full joint-space witness. Successful open-witness queries
retain the exact vector in addition to the boolean verdict. After optional release hoisting, the
planner aligns cached witnesses against the final state-action trajectory. Any affected edge whose
predecessor scene changed is certified once before serialization.

The plan JSON records one witness entry per action and labels the block as either `search` or
`optimized`. A required IK action with a missing witness causes export to fail. Subsequent witness
optimization always includes the search witness as an immutable feasible fallback. Thus search,
optimization, motion planning, and replay consume the same continuous certificates rather than
independently sampling IK.

### 7. Guarantees and current limitations

The following statements are supported by the present construction:

- Returned plans are checked against the enabled discrete, wrench, cheap-collision, and IK gates.
- Under the relaxation assumptions in Proposition 1, the backward distance is admissible.
- Signature collapse preserves admissibility because class facts are existentially lifted.
- A complete, uncapped backward table can soundly identify abstract dead ends.
- Incumbent pruning by the unweighted test \(g+h\geq U\) is sound when \(h\) is admissible.
- The anytime wrapper always retains its incumbent when a later search times out.

The following should **not** currently be stated as unconditional theorems:

- Default milestone search is not globally complete. It commits to settled checkpoints, uses a
  finite fill width and repair budget, and stores coarse milestone conflicts.
- A bounded open-witness failure is not an IK infeasibility proof.
- “Hard” wall-clock compliance is not guaranteed across one long nonlinear solve.
- The current weighted A*/PEA* engines claim a concrete state on its first feasible pop and do not
  reopen it for a lower-cost path. The implementation also documents inconsistency in the labeled
  heuristic. Standard weighted-suboptimality and weight-one optimality claims therefore require
  either a consistency proof, a reopening implementation, or an independent argument specific to
  these state transitions. `--pathmax` repairs local heuristic drops but is not by itself a
  general replacement for reopenings.
- Partial expansion substantially reduces frontier memory but cannot literally guarantee that an
  arbitrary instance will never exhaust memory.

These qualifications affect quality and completeness claims, not the feasibility of a returned
and fully witnessed plan.

### 8. Current command mapping

| Command | Role |
|---|---|
| `--milestones` | default abstraction-guided milestone search |
| `--milestones-anytime T` | milestone incumbent plus bounded PEA improvement/fallback |
| `--sig-collapsed-h` | signature-class backward abstraction; default on |
| `--sig-components press` | default signature ablation |
| `--mixed-wrench-h` | expensive per-tool wrench facts for mixed grippers |
| `--geo-facts` | optional floating-gripper coexistence facts |
| `--fill-window K` | maximum adaptive milestone width; default 2 |
| `--no-lazy-fills` | disable candidate-path-only IK validation |
| `--open-witness` | non-terminal, effort-tiered stochastic IK validation |
| `--open-witness-batch B` | maximum effort batch; default 7 |
| `--open-witness-epochs E` | fairness epochs per fill; default 2 |
| `--bounded` | milestone incumbent followed by an un-timed incumbent-bounded global search |
| `--anytime T` | separate restarting weighted global search, without milestone decomposition |

### 9. Suggested experimental reporting

For a paper evaluation, search quality should be separated into four layers:

1. **Backward construction:** build time, number of co-reachable abstract states, number of fact
   queries, raw grasps versus signature classes, and initial abstract distance.
2. **Milestone decisions:** chosen part order, failed one-part fills, window escalations, module
   fills, backtracks, and conflict count.
3. **Continuous validation:** IK effort slots, successful witnesses, open failures, time per skill,
   and the fraction of candidate paths invalidated at each edge position.
4. **Anytime behavior:** first-solution time and cost, incumbent cost over time, PEA frontier size,
   improvement-pass outcome, and final certificate completeness.

Ablations should independently vary the backward representation (`press`, all signature
components, raw labels, and identity collapse), milestone width, lazy versus eager IK, terminal
versus open witness failure, and the bounded global improvement phase. This separates better
high-level ordering from cheaper continuous validation and from global cost refinement.
