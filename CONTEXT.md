# CONTEXT — fixtureless multi-robot assembly

This folder builds and evaluates the fixtureless assembly paper stack: a search-based planner
for multi-robot sequential assembly with no jigs, where static equilibrium of every
intermediate state is the planning constraint.

## The task we solve

- Input: rigid parts, an assembly dependency graph `D` (operations with poses and process
  wrenches), a roster of possibly heterogeneous robots, environment geometry.
- Output: a minimum-cost sequence of manipulation actions (picks, places, assembles,
  handovers, braces) that completes `D`, such that every state — and every point along every
  motion — is quasistatically supported and kinematically feasible.
- Core idea: regrasps are NOT free, and mates are supports with possibly unilateral wrench
  capability. Hold / brace / handover decisions are reasoned about, not scripted.
- Design document: `assembly_planner_spec.md` (scene-graph state + witness, actions under the
  fuse-vs-reify principle, wrench LP at search time).

## Ubiquitous language

- **dependency graph `D`** — nodes are assembly ops `(inputs, output, pose, process_wrench,
  self_supporting)`; edges are precedence. The WHAT and the order are input; we plan the HOW.
- **parent vs supports** — `parent` is a kinematic tree (part -> table | gripper | part);
  `supports` is a set-valued statics multigraph (grasp, table face, mate). Different roles:
  whether a contact may be removed is answered from `supports`, never from `parent`.
- **mate** — a completed operation edge; a support with possibly partial/unilateral wrench
  capability. Mated does NOT imply supported.
- **witness** — one consistent kinematic realization per search node (excluded from identity).
- **fuse-vs-reify** — fused `pick`/`place` actions for the common cases; unfused variants
  expose intermediate states for handovers and bracing.
- **milestones / ladder / LNS** — the search-solver menu in `assembly.py` / `main.py`.
- **wrench LP** — conservative wrench sufficiency (`wrench.py`); `wrench=True` vs `False` is
  the paper discriminator pair.
- **zero-tolerance recipe** — `--strict-collisions` + `--witness-robot-collisions` +
  `--pre-insertion 0.05` + `--mp-collision-tolerance 1e-6` (`PIPELINE.md` §4).
- **discriminator** — a scene where the topological plan is physically infeasible and the
  wrench-aware plan succeeds. The single most important missing result (`EVALUATION.md`).
- **floating vs real arms** — floating-gripper keyframes (v1 sim, validates the statics) vs
  FK-synced arms following the BiRRT path (v2 sim).

## Pipeline

solve (`main.py`, `assembly.py`, `solver.py`) → motion-plan (mr_bench as an unmodified module
via `mp_export.py`, BiRRT) → validate (MuJoCo, `sim.py`). Commands: `PIPELINE.md`.

## Conventions

- `PYTHONHASHSEED=0` for anything you compare. Never run two heavy solves concurrently.
  Always `--no-view` in scripted runs.
- mr_bench is used unmodified; its issues live in `mr_bench_integration_notes.md`.
- A BiRRT export is irreproducible by design (wall-clock termination, `set()` iteration,
  numba RNG). Treat a working `mp` path as a preserved artifact; never regenerate
  (`FRAGILITY.md` §1.2).
- New experiment files only; keep experiment records under `experiments/`.
- Use ASD-STE100 Simplified Technical English.

## Doc map

- `PIPELINE.md` — how to run each stage.
- `EVALUATION.md` — status, baselines, open work. Read before claiming any result.
- `FRAGILITY.md` — cross-stage contract failures and known traps.
- `assembly_planner_spec.md` — the design document (paper skeleton).
- `mr_bench_integration_notes.md` — mr_bench-side bugs and fixes.
- `experiments/shortcut-sampling/README-v5.md` — the shortcutting experiment round record.

## Current state

The statics machinery, the search (milestones + LNS), and the sim harness work. The missing
validation leg is a real discriminator scene (topological plan infeasible, wrench plan
succeeds); the old cross-5 discriminator was an artifact of a sim bug and must not be quoted.
Motion plans are kept as artifacts because fresh exports are not reproducible.
