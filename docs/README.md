# Documentation index

The root [README](../README.md) gives the short project overview. The documents here contain
the detailed research record and operational guidance.

## Start here

- [PIPELINE.md](PIPELINE.md) - commands for task search, motion export, simulation, and replay.
- [EVALUATION.md](EVALUATION.md) - current evidence, limitations, and open paper work.
- [REPRODUCTION.md](REPRODUCTION.md) - historical experiment and reproduction record. Use it
  to audit older results; use the root README and pipeline guide for maintained commands.

## Method and paper

- [PAPER_METHOD_SHORT.md](PAPER_METHOD_SHORT.md) - short methods text for the paper.
- [SEARCH_METHOD.md](SEARCH_METHOD.md) - complete search formulation and flag mapping.
- [PHASE_SCHEDULING.md](PHASE_SCHEDULING.md) - collision-certified execution scheduling.
- [PAPER_PLAN.md](PAPER_PLAN.md) - paper claim, experiment plan, and proposed structure.

## Implementation reference

- [assembly_planner_spec.md](assembly_planner_spec.md) - planner design and proof-of-concept
  specification.
- [FRAGILITY.md](FRAGILITY.md) - cross-stage contracts and known failure modes.
- [mr_bench_integration_notes.md](mr_bench_integration_notes.md) - motion-planner integration
  issues.
- [SIMV2.md](SIMV2.md) - execution-program simulator notes.
- [SIM_CONTACT.md](SIM_CONTACT.md) - opt-in physical-contact experiment.

Maintained benchmark commands stay in [benchmarks/README.md](../benchmarks/README.md). Detailed
historical results and superseded experiments are retained in
[BENCHMARK_HISTORY.md](BENCHMARK_HISTORY.md); they are not the current usage guide.
Experiment-local notes stay with their artifacts under `experiments/`.
