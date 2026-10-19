# Fixtureless multi-robot assembly planner

This repository contains the research code for fixtureless multi-robot assembly. It searches
for sequences of picks, places, handovers, braces, and assembly actions while it checks
kinematics, collision clearance, and quasistatic support. It can then export real-arm motion
paths through [`mr_bench`](https://github.com/vhartman/multirobot-pathplanning-benchmark) and
validate them in MuJoCo.

> **Research status:** This is an active prototype, not a general-purpose robotics library.
> Read the [evaluation status](docs/EVALUATION.md) before you use a result as paper evidence.

## Pipeline

The main data flow is:

```text
assembly_planner/problem.py -> main.py -> saved plan and witnesses
  -> assembly_planner/mp_export.py -> sim.py
```

The planner represents kinematic attachment and physical support separately. Completed mates
can therefore be partial or unilateral supports, and the search must plan the holds and braces
that keep each intermediate assembly stable.

## Quick start

Use Python 3.12 and create the repository environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run a small geometric planning example:

```bash
PYTHONHASHSEED=0 python3 main.py \
  --assembly cross --parts 5 --num-robots 2 \
  --weight 3 --solve --no-view
```

Run the test suite:

```bash
PYTHONHASHSEED=0 python3 -m pytest tests/ -q
```

Motion planning also needs the
[`multirobot-pathplanning-benchmark`](https://github.com/vhartman/multirobot-pathplanning-benchmark)
source tree. Its default location is the sibling checkout `../mr_bench/src`. Set `MRBENCH_SRC`
when you keep it elsewhere:

```bash
export MRBENCH_SRC=/absolute/path/to/mr_bench/src
```

The full solve, export, simulation, and replay commands are in the
[pipeline guide](docs/PIPELINE.md). Use `python3 main.py --help` for the complete option list.

## Repository map

| Path | Purpose |
| --- | --- |
| `main.py` | Main command-line entry point and run configuration |
| `assembly_planner/` | Assembly models, search, witnesses, statics, and motion-planning export |
| `sim.py`, `execution/` | MuJoCo validation and execution programs |
| `benchmarks/` | Reproducible runners, manifests, plots, and benchmark notes |
| `tests/` | Maintained regression tests |
| `experiments/` | Saved run artifacts and focused experiment records |
| `part_meshes/`, `sim_meshes/`, `src/rai/` | Assembly and robot geometry |
| `project_page/` | Paper project-page draft and replay viewer |
| `docs/` | Method, evaluation, reproduction, and implementation writeups |
| `tools/` | Calibration, diagnostics, export, and visualization commands |

Generated plans, recordings, and result tables belong in `experiments/`, `sim_recordings/`,
or `benchmarks/results/`, not in the repository root.

## Documentation

- [Documentation index](docs/README.md) - guide to the maintained and archival writeups.
- [Pipeline guide](docs/PIPELINE.md) - solve, export, simulate, and replay.
- [Search method](docs/SEARCH_METHOD.md) - full search description and scope.
- [Short paper method](docs/PAPER_METHOD_SHORT.md) - condensed methods text.
- [Evaluation status](docs/EVALUATION.md) - available evidence and open validation work.
- [Full reproduction record](docs/REPRODUCTION.md) - detailed commands, ablations, and
  historical experiment notes formerly kept in this README.
- [Known pipeline failure modes](docs/FRAGILITY.md) - reproducibility and stage-contract risks.

For comparable runs, set `PYTHONHASHSEED=0`, do not run two heavy solves at the same time, and
use `--no-view` in scripts. Preserve a successful motion-plan artifact: motion planning uses
wall-clock limits and is not fully reproducible from a task plan alone.
