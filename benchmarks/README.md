# Benchmarks

This directory contains the maintained benchmark runners, checked-in experiment definitions,
and plotting programs used to evaluate the planner. Run all commands from the repository root.

The benchmark code writes generated plans, logs, tables, and plots below
`benchmarks/results/`. Generated results are not source files and should not be committed unless
they are selected paper artifacts.

For the full development history and superseded experiments, see the
[benchmark experiment history](../docs/BENCHMARK_HISTORY.md). The maintained end-to-end command
is in the [root README](../README.md#verified-car-mini-pipeline), with stage details in the
[pipeline guide](../docs/PIPELINE.md).

## Reproducibility rules

- Use `PYTHONHASHSEED=0` for comparable runs.
- Run reported timing experiments sequentially with `--jobs 1`.
- Use a fresh output directory, or use `--resume` only with the same source and configuration.
- Keep task plans and motion paths from the same run together.
- Treat timeouts, failed searches, and failed execution as results; do not remove them from the
  success denominator.
- Record the checked-in manifest and source revision with every reported table.

The runners execute individual samples in fresh subprocesses and retain the resolved command,
raw log, resource measurements, and machine-readable result. This prevents mutable RAI, wrench,
grasp, and IK state from leaking between experimental cells.

## Quick check

Preview one checked-in paper-search command without running a solve:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/paper_search_comparison.json \
    --only 'cross.*ours' --max-runs 1 --dry-run
```

Run one bounded search sample:

```bash
PYTHONHASHSEED=0 python3 benchmarks/search_runner.py \
    benchmarks/manifests/paper_search_comparison.json \
    --only 'cross.*ours' --max-runs 1 --timeout 450 \
    --output /tmp/assembly-search-smoke
```

The command exits nonzero if the selected search fails, times out, or crashes. Partial logs and
structured results remain in the output directory.

## Paper search comparison

The main search comparison uses the checked-in
`benchmarks/manifests/paper_search_comparison.json` definition:

```bash
PYTHONHASHSEED=0 python3 benchmarks/run_paper_search_comparison.py \
    --output benchmarks/results/paper-search-comparison
```

The default run uses three fixed seeds and a 450-second search limit. Successful planner rows
also receive a first-solution motion-planning check. Use `--skip-motion-initial` only when the
measurement is explicitly search-only.

Useful controls:

- `--only REGEX` selects expanded run IDs without editing the manifest.
- `--max-runs N` creates a bounded smoke run.
- `--resume` continues an interrupted suite after checking run and source fingerprints.
- `--report-only` regenerates tables from existing structured results.
- `--dry-run` prints every command without executing it.

## Wrench-planning execution ablation

This runner compares topological, gravity-only, and full process-wrench task planning while
executing every result under the same physical process load:

```bash
PYTHONHASHSEED=0 python3 benchmarks/run_wrench_sim_ablation.py \
    --output benchmarks/results/wrench-sim-ablation \
    --seeds 11023 11024 11025
```

Task planning, motion planning, and MuJoCo execution run as separate guarded stages. For a
development check, select one scene, model, and seed and avoid recordings:

```bash
PYTHONHASHSEED=0 python3 benchmarks/run_wrench_sim_ablation.py \
    --scenes cross --variants full --seeds 11023 \
    --record none --output /tmp/wrench-sim-smoke
```

## Imported CAD validation

`run_fabrica_smoke.py` applies geometry audit, search, motion planning, and simulation as
separate stages to the checked-in Fabrica scene descriptions:

```bash
PYTHONHASHSEED=0 python3 benchmarks/run_fabrica_smoke.py \
    --models car --stages audit search \
    --output /tmp/fabrica-smoke
```

Run `python3 benchmarks/run_fabrica_smoke.py --help` before a full suite; motion planning and
simulation are intentionally expensive.

## Generic manifest runner

`search_runner.py` expands a JSON manifest as `case x variant x seed`. A minimal manifest is:

```json
{
  "schema_version": 1,
  "name": "experiment-name",
  "description": "hypothesis tested by this suite",
  "timeout_s": 450,
  "seeds": [11023, 11024, 11025],
  "defaults": ["--search-profile", "standard"],
  "cases": [
    {"name": "cross", "args": ["--assembly", "cross", "--parts", "7"]}
  ],
  "variants": [
    {"name": "method", "args": ["--open-witness"]}
  ]
}
```

Cases and variants can override seeds, timeout, tags, or set `"enabled": false`. Shared
arguments precede case and variant arguments. Do not specify the same planner option more than
once: the planner reports duplicates because argparse would otherwise keep only the last value.

## Output contract

A search suite contains:

```text
manifest.json             exact manifest snapshot
provenance.json           source hash and host information
results.json              completed per-run records
runs.csv                  flat per-run metrics
summary.{json,csv}        grouped success and resource metrics
runs/<run-id>/
    run.json              resolved command and fingerprint
    stdout.log            complete planner output
    time.txt              wall time, peak RSS, and exit status
    plan.json              replayable plan for a successful search
    result.json            parsed metrics and artifact paths
```

The raw log is authoritative. Missing parser data remains null rather than becoming zero.
Resume reuses a run only when its resolved specification and source fingerprint match.

## Programs

| Program | Purpose |
| --- | --- |
| `search_runner.py` | Isolated manifest-driven search experiments |
| `run_paper_search_comparison.py` | Paper search matrix and table generation |
| `run_paper_anytime_convergence.py` | Anytime convergence experiments |
| `run_wrench_sim_ablation.py` | Task plan through physical execution ablation |
| `run_fabrica_smoke.py` | Imported-CAD audit and pipeline checks |
| `run_noisy_sim.py` | Seeded placement and process-load robustness trials |
| `witness_quality.py` | Compare persisted witness-chain continuity |
| `plot_*.py` | Regenerate figures from structured results |

Use each program's `--help` as the authoritative option reference.
