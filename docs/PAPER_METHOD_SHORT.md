# Search Method — one-page condensed version (for a ~6-page paper)

Companion to SEARCH_METHOD.md (full description). This is the cut-down methods core;
related work and evaluation live elsewhere in the paper.

---

## Method

We consider multi-robot assembly: parts with a mate dependency graph must be joined by
2–4 arms carrying heterogeneous grippers, under grasps from a precomputed database,
static support/wrench constraints, geometric collision gates, and per-action robot-level
feasibility (IK with a joint-space *witness*). The output is a fully witnessed action plan.

**Goal-conditioned reverse abstraction with signature collapse.** The supplied dependency
graph selects which join may be attempted; we do not rediscover a nominal part sequence. For
each join, a query-driven abstract predecessor graph retains the assembled set, support/rest
status, held loose parts, and grasp facts, while dropping robot identity and continuous
configuration. Raw grasps are uniformly projected into fact signatures encoding the enabled
pick, placement, tool, and wrench-support predicates. Reverse Dijkstra gives an admissible
local cost-to-join under the stated relaxation assumptions. Missing or unfinished table entries
remain unknown unless the finite abstract graph was actually exhausted.

**Abstraction-guided milestone construction.** Forward search proceeds through settled
checkpoints. Dependencies and a canonical tie order choose among currently legal parts or
detached modules. A greedy concrete *fill* search realizes the selected join using the raw
pick/place/assemble/handover/brace actions and the local reverse distance. It first restricts
manipulation to relevant components, then widens to all-part manipulation before rejecting the
milestone. The free world-z yaw of a table placement is represented by four quarter turns but
introduced by finite iterative broadening: one yaw first, then two, then four, with stable action
IDs and witness reuse. The reverse abstraction quotients this yaw orbit because its represented
support/wrench facts are invariant, while concrete fixed-arm IK sees every copy. Joint windows,
detached-module construction, multiple exact endpoint boundaries, and chronological repair
recover from decomposition choices. These are finite operational bounds, not a global
completeness proof.

**Open-witness validation.** Continuous feasibility inside fills follows LazySP: candidate
plans pay only cheap geometric gates; expensive IK runs on incumbent edges only, and
refuted edges are repaired incrementally. Unlike classical lazy validation we treat
stochastic IK failure as *non-terminal*: each blocking edge receives bounded effort batches
(analytic seeds where calibrated, relaxed reach probes, then escalating restarts) and is
suspended rather than blacklisted; exhausted alternatives reopen it in a later fairness
epoch with escalated effort. Every success stores the exact joint vector, so plans carry
machine-checkable witnesses that downstream stages must preserve.

**Anytime improvement.** Under an explicit deadline, the milestone phase removes the ordinary
finite IK-epoch cap and dovetails fresh witness effort until its reserved deadline. A found plan
is retained while consecutive milestone neighborhoods grow from one mate through the complete
incumbent horizon. Each non-certified window first runs greedy best-first search below the old
span cost, then weight-one A* below the greedy replacement cost (or the old cost when greedy
finds none). A strict improvement is spliced into the incumbent and restarts the sweep. Positive
IK witnesses and accumulated open-witness effort persist between passes. Exhausting a pass is an
exact result for its currently exposed finite graph; finite stochastic IK failures, resource
limits, and outer milestone/action-space broadening still preclude an unconditional global
optimality claim.

**Direct motion validation.** The optional executable-plan driver consumes complete plans from
one resumable milestone session. Each fresh plan receives a short optimized motion-planning
consultation. A miss suspends the whole plan—it never blacklists an action—then enumeration
continues. After the fresh-candidate budget, all suspended plans are revisited fairly at increasing
consultation budgets. This direct-validation driver and milestone-anytime cost improvement are
currently separate schedules; the implementation does not yet motion-validate every LNS update.

## Guarantees and scope

Abstract local distances are admissible under stated relaxation assumptions; uniform signature
projection never removes a concrete grasp. Returned plans satisfy all enabled discrete, wrench,
collision and IK gates with exact persisted witnesses. Weight-one search is exact inside each
fully explored finite anytime window; the standard initial fill remains greedy for time to first
solution. The complete system is not globally complete because endpoint diversity, module order,
resource guards, outer action-space broadening, and ordinary IK effort are bounded.

## Relation to existing work (positioning, to be expanded)

Assembly-by-disassembly planning and pattern databases motivate our local reverse guidance;
geometric and semantic abstractions motivate fact-based collapse; iterative broadening and
focused conditional-sampler TAMP motivate delayed placement parameterization; LazySP and lazy
constraint evaluation ground the fill validation; our contribution beyond these is
the combination of structure-first multi-robot fills, wrench- and tool-aware *signature*
collapse, and a *fair, resumable* effort model for stochastic continuous validation.
