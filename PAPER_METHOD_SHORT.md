# Search Method — one-page condensed version (for a ~6-page paper)

Companion to SEARCH_METHOD.md (full description). This is the cut-down methods core;
related work and evaluation live elsewhere in the paper.

---

## Method

We consider multi-robot assembly: parts with a mate dependency graph must be joined by
2–4 arms carrying heterogeneous grippers, under grasps from a precomputed database,
static support/wrench constraints, geometric collision gates, and per-action robot-level
feasibility (IK with a joint-space *witness*). The output is a fully witnessed action plan.

**Backward abstraction with signature collapse.** We solve an assembly-by-disassembly
relaxation exactly, backward from the goal. An abstract state is *(I, G, φ, L)*: assembled
parts *I*, grasp labels supporting *I*, rest face φ or held-in-air, and held loose parts
*L*. Robot identities are dropped (only arm counts and tool-kind capacities remain), and
raw grasps are partitioned into **fact signatures** σ(p,g) — insertion-force transmission,
table-aware pickability, contact class, solo-hold capability, brace side — with every
context-dependent fact existentially lifted over class members. A backward Dijkstra over
the resulting table yields an admissible distance-to-go d(π(s)); absent entries are sound
dead-end certificates when the table is uncapped.

**Abstraction-guided milestone construction.** Forward search proceeds through settled
checkpoints. At each checkpoint, precedence-addable parts are ranked by d after the
hypothetical addition; the chosen part is added by a small complete *fill* search over
pick/place/assemble/handover/brace actions, ending at ≤1 holder and nothing loose.
Decomposition failures are recovered by widening to joint k-part fills, detached module
construction, and chronological repair (blacklisting the committed decision). Fill-level
heuristics join the counting bound with the shifted abstract distance d(π(s)) − O(I,p).

**Open-witness validation.** Continuous feasibility inside fills follows LazySP: candidate
plans pay only cheap geometric gates; expensive IK runs on incumbent edges only, and
refuted edges are repaired incrementally. Unlike classical lazy validation we treat
stochastic IK failure as *non-terminal*: each blocking edge receives bounded effort batches
(analytic seeds where calibrated, relaxed reach probes, then escalating restarts) and is
suspended rather than blacklisted; exhausted alternatives reopen it in a later fairness
epoch with escalated effort. Every success stores the exact joint vector, so plans carry
machine-checkable witnesses that downstream stages must preserve.

**Anytime improvement.** The milestone plan seeds an incumbent-bounded partial-expansion
weighted A* over the full concrete space (prune g+h ≥ U); timeouts can never lose the
incumbent.

## Guarantees and scope

Abstract distances are admissible under stated fact-necessity assumptions; signature
collapse preserves this by existential lifting. Returned plans satisfy all enabled
discrete, wrench, collision and IK gates with exact witnesses. The method is complete up
to its explicit resource caps (fill width, repair budget, effort epochs); weighted search
claims suboptimality-bounded, not optimal, results absent heuristic consistency.

## Relation to existing work (positioning, to be expanded)

Assembly-by-disassembly planning and goal-set guidance give our backward structure;
geometric/semantic abstractions and pattern databases motivate fact-based collapse; LazySP
and lazy constraint evaluation ground the fill validation; our contribution beyond these is
the combination of wrench- and tool-aware *signature* collapse with a *fair, resumable*
effort model for stochastic continuous validation inside decomposed multi-robot fills.
