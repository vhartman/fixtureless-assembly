# Search benchmarks

## Wrench-planning execution ablation

`run_wrench_sim_ablation.py` runs the complete task-planning, optimized motion-planning, and
MuJoCo execution pipeline for Cross, the eight-part vacuum Cube, and Chair. It compares plans
computed with no equilibrium model, gravity only, and gravity plus the nominal process wrench.
All executions receive the same 10 N nominal process load, so this is a planning-model ablation
rather than an accidental change to the simulated disturbance:

```bash
python3 benchmarks/run_wrench_sim_ablation.py \
    --output benchmarks/results/wrench-sim-ablation-lockstep \
    --seeds 11023 11024 11025 11026 11027
```

To isolate process-wrench failures while retaining physical gravity, measure transport loads
but permit grasp breakage only during insertion. Existing task and motion plans can be reused;
this reruns simulation rather than silently mixing the new execution semantics into old reports:

```bash
python3 benchmarks/run_wrench_sim_ablation.py \
    --output benchmarks/results/wrench-sim-insertion-break \
    --seeds 11023 11024 \
    --reuse-motion-plans benchmarks/results/wrench-sim-ablation-lockstep \
    --insertion-only-grasp-check
```

The benchmark stops at the first failed action by default. This does not turn an overload into
a failure outside the selected breakage interval; it only prevents missed seats, broken grasps,
or other already-failed actions from contaminating utilization with downstream motion in a
corrupted scene. Use `--no-abort-on-action-failure` only for interactive cascade diagnostics.

Runs are sequential fresh processes with thread, address-space, and wall-time guards. Task
planning and motion planning are separate subprocesses and therefore have unambiguous timings
and failure labels. Defaults are a 300 s task-planning wall limit, a 300 s RRT* optimization
consultation inside a 450 s motion-stage wall limit, and a 600 s simulation wall limit; override
them with `--task-timeout`, `--mp-runtime`, `--motion-timeout`, and `--sim-timeout`.

Simulation explicitly disables the standard profile's spline and phase scheduler and executes
the certified synchronized/lockstep reference. Actuated arms, skill servos, arm--arm contacts,
strict path preflight, wrench guards, and the common physical process wrench remain enabled.
Motion planning is force-blind for every planning model: its collision and kinematic settings
are identical, leaving MuJoCo as the independent physical test rather than selectively rejecting
the gravity/full rows with the experimental MP wrench gate.

Progress is checkpointed in `summary.json`/`summary.csv`; `--resume` skips completed cells. `table.tex`
contains the paper table, while every failed cell records whether task planning, motion planning,
simulator preflight, or physical execution failed. The table reports success and the mean
per-run peak of the 20 ms averaged normalized grasp-wrench utilization. Two-finger and vacuum
loads are reported separately within the transport and insertion column groups, in percent of
the corresponding directional limit; values above 100% are bold. The insertion group also
reports physical table-friction utilization as `sum(|f_t|) / (table_mu * sum(f_n))`, from
MuJoCo's part--table contacts. The table uses a two-row header. The JSON and CSV retain the unsplit aggregate
for compatibility, plus per-tool utilization, instantaneous utilization, raw force/torque peaks,
path length, successful task-action prefix, failure index, failing action, and separate task,
motion, and simulation wall times.
The `planning_wall_s` field is the task-plus-motion sum. Useful
development subsets are, for
example, `--scenes cross --variants gravity full --seeds 11023` and `--dry-run`.

For execution robustness rather than search timing, `run_noisy_sim.py` expands seeded initial
table-placement and process-wrench trials. It runs every sample sequentially in a fresh,
memory-limited process and writes checkpointed JSON/CSV plus replayable failure recordings:

```bash
python3 benchmarks/run_noisy_sim.py \
    --plan experiments/PLAN.json --mp-path experiments/MP.json \
    --output benchmarks/results/noisy-sim \
    --seeds 0 1 2 3 4 --xy 0.005 --yaw-deg 3 --observation both \
    --apply-process-wrench --process-noise sample
```

The simulator, not the runner, samples the collision-free scene and judges physical capture;
there is no nominal fallback when a perturbation cannot be sampled. See the main README's
simulation section for the observed/unobserved controller semantics.

## Search timing

`search_runner.py` executes search experiments from a JSON manifest. Every sample runs in a
fresh Python process and all samples run sequentially. This is intentional: RAI, grasp, wrench,
and IK caches must not leak across variants, and the main repository explicitly discourages two
concurrent search or motion-planning jobs.

The runner controls only experiment hygiene and search-only output:

- `PYTHONHASHSEED=0` and the requested planner seed;
- no interactive viewer;
- no witness optimization or witness-set harvesting;
- no cograsp postprocessing or motion planning; and
- one replayable plan JSON and raw log per successful run.

Exact successful search-edge witnesses are still persisted. Consequently, success means that the
run produced a replayable action plan with one exact IK certificate for every required edge, not
merely an action list printed to stdout. It is deliberately not a globally combined robot-motion
certificate: inactive arms are parked independently in search scenes, and the later witness/export
and motion-planning stages compose consecutive multi-robot configurations and certify their paths.

The matched physical-query-deferral ablation uses the established OpenArm Car Mini, OpenArm
Cross, and mixed-gripper Chair scenes:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/lazy_physical_deferral.json \
    --output benchmarks/results/lazy-physical-deferral
```

It compares `--lazy-edge-selector forward` and `defer-failed` with every other standard-profile
setting fixed. Deferral remains an ablation: it helps Cross and the high-alias cube cases, is
neutral on Car Mini, and redirects Chair into a larger symbolic frontier.

### Paper search comparison

`run_paper_search_comparison.py` runs the search matrix used by the comparison and timing-
breakdown tables. Every sample is a fresh sequential process, defaults to three fixed seeds, has
a 4 GiB process-tree resident-memory limit, and has a 450 s wall-clock limit. It disables witness reranking and
motion planning but retains the exact per-edge IK
certificates found by lazy search. To avoid repeatedly spending the full budget on an
intractable cell, the first timeout for a case/solver pair suppresses its remaining unrun seeds
and repetitions. The timeout itself remains in the aggregate and generated table.

On another Linux machine, a repository checkout plus the environment documented in the root
README is sufficient for search. The default initial-motion-planning stage additionally needs
the `mr_bench` checkout selected by `MRBENCH_SRC`. The runner uses GNU `/usr/bin/time` for peak
RSS and a dependency-free `/proc` watchdog for the process-tree resident-memory guard. This is
deliberately not `RLIMIT_AS`: BLAS and robotics libraries reserve machine-dependent virtual
address ranges which do not represent committed physical memory. No Fabrica source checkout and no
generated file under `experiments/` is required: the converted meshes and plain benchmark scene
configs are repository inputs.

Before timing search, the wrapper builds one fingerprinted, collision-filtered grasp database
per case and seed. Every solver variant loads that identical immutable artifact in its fresh
process. The one-time generation duration is retained as `grasp_generation_s` for the breakdown,
while `search_pipeline_s` compares the solvers given the common database. The files and build
logs live under `OUTPUT/grasp-databases/`; source/config changes produce a new fingerprint rather
than silently reusing stale rows.

The comparison uses the finite `max_robotiq` two-finger hardware envelope and disables the
experimental action-family widening. Native Cross/Stool/Chair cases select `standard`: K=20
followed by monotone K+8 grasp widening up to 96 after a failed complete fill. Imported CAD cases
select `fabrica`: all standard search semantics with a fixed K=64 alphabet and no grasp-prefix
widening. Every solver row for a given case therefore sees the same profile and action alphabet;
the case-dependent profile choice is explicit in the manifest and recorded in each command.
The Fabrica scene descriptions are small checked-in files under `benchmarks/configs/fabrica/`.
The paper runner never depends on ignored generated plans under `experiments/`; those files may
contain legacy defaults and are output artifacts rather than portable benchmark inputs.

```bash
python3 benchmarks/run_paper_search_comparison.py \
    --output benchmarks/results/paper-search-comparison
```

The checked-in paper manifest selects seeds `11023 11024 11025`. Override them explicitly for
an independent-seed experiment; `--repeats` instead reruns each selected seed unchanged and is
only useful for measuring same-instance timing/numerical variation:

```bash
python3 benchmarks/run_paper_search_comparison.py \
    --seeds 11023 11024 11025 11026 11027 \
    --continue-after-timeout \
    --output benchmarks/results/paper-search-comparison-5seed
```

Without `--continue-after-timeout`, the wrapper retains its budget-protection policy: once one
seed times out for a case/solver cell, later requested seeds for that cell are skipped.

Interrupted experiments can be continued with `--resume`. A one-seed Cross verification is:

```bash
python3 benchmarks/run_paper_search_comparison.py \
    --only 'cross__.*__seed-11023' --timeout 180 \
    --output benchmarks/results/paper-search-comparison-cross-smoke
```

The six matched rows are:

- **Optimistic:** the optimistic/Dogar-style lazy search, without the exact fallback;
- **Ours:** the standard milestone solver with greedy concrete fills and the lazy projected
  reverse heuristic;
- **Flat lazy:** the same exact lazy validation in one unrestricted global search, without
  milestones or the projected fill heuristic;
- **Milestones + counting:** the same milestone decomposition, replacing the reverse heuristic
  by the counting heuristic;
- **Ours, exact fills:** weight-one A* ordering for each concrete milestone fill (this is
  “exact” for the finite searched fill graph, not a claim that finite numerical IK restarts prove
  geometric infeasibility); and
- **Ours, fine projection:** the standard method with the same local reverse Dijkstra search,
  but with the more informative `press,pickface,solohold,braceside` grasp projection instead of
  the standard `press` projection. This isolates the abstraction-resolution tradeoff without
  changing the forward search, IK validation, or milestone decomposition.

Outputs include raw logs and plans, `results.json`, aggregate JSON/CSV, the combined table-ready
`comparison.tex`, `search-time-breakdown.tex`, and a flat `table-data.csv`/`.json`. The paired
geometric means are also written to `comparison-time-cost-geomeans.csv`/`.json` and plotted in
`comparison-time-cost.pdf`/`.png`; a method without any successful run is shown in the legend but
cannot be assigned a runtime--cost point. The reported search pipeline begins before grasp
preflight and ends when the initial witnessed plan is found. `comparison.tex` requires
`booktabs`, `multirow`, and `siunitx`; it uses the latter to align numerical cells at their
decimal points. Its exclusive breakdown is
grasp generation + reverse graph expansion + reverse force-LP solves + forward-search overhead +
exact IK. Forward-search overhead includes its cheap checks and forward wrench LPs. The CSV/JSON
also retain the inclusive reverse timer, nested reverse-grasp time, optimizer call counts, and
forward-LP time so the exclusive decomposition can be audited. Solver-call timing covers both
the persistent HiGHS backend and SciPy fallback. OOM,
timeout, and no-plan outcomes remain explicit rather than being replaced by their resource cap.

After the search matrix, the runner automatically takes every successful **Ours** plan through
the standard motion export and BiRRT with `--no-mp-optimize --mp-no-shortcut`. The resulting
$t_{\mathrm{MP\,init}}$ is the planner's internally recorded time to its first complete path,
so scene reconstruction and post-solution optimization are excluded. Motion planning uses a
300 s planner budget, 420 s whole-process guard, and 4 GiB address-space guard by default;
these can be changed with `--motion-runtime`, `--motion-timeout`, and `--motion-memory-gb`.
`--resume` also reuses completed motion samples. Use `--skip-motion-initial` only for
search-only diagnostics.

For the compact paper convergence panels, combine the anytime incumbents with the aggregate
comparison results as follows:

```bash
python3 benchmarks/plot_paper_convergence.py \\
    --anytime benchmarks/results/paper-search-anytime-300/all-scenes.json \\
    --comparison-summary benchmarks/results/paper-search-comparison/summary.json \\
    --output-dir benchmarks/results/paper-convergence \\
    --cases cross stool chair --paper
```

This writes one title-free PDF/PNG per assembly and a common `convergence-legend.tex`. All
solver points use the same circular marker; color alone identifies the solver. The black Ours
curve is the validated incumbent trajectory, whereas the comparison algorithms contribute only
their final returned solution. Time is logarithmic by default so both early solutions and the
full anytime horizon remain legible; pass `--linear-time` for a linear axis.

The no-wrench target-support quotient ablation compares the ordinary labeled local PDB with a
version that represents holders on the already-built component only by count and tool-kind mask;
the incoming component's grasp stays labeled:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/no_wrench_target_support_quotient.json \
    --output benchmarks/results/no-wrench-target-support-quotient
```

This is deliberately not part of `standard`: it is defined only for `--no-wrench` and
`--no-geo-facts`. Concrete successor generation and IK validation are unchanged.

## Cube scaling (1--12 parts, 2--4 robots)

The dedicated wrapper runs the full 12-by-3 matrix sequentially and then writes both PDF and PNG
plots:

```bash
python3 benchmarks/run_cube_scaling.py \
    --output benchmarks/results/cube-scaling
```

Resume the same directory after sleep or interruption with `--resume`. Each child process is
capped at 6 GiB and 900 seconds by the manifest; timed-out samples are checkpointed, the suite
continues, and each timeout remains an explicit censored `x` in the plot rather than being
substituted for a solution time. The outputs are `cube-scaling.pdf`, `cube-scaling.png`, and
the corresponding inspectable `cube-scaling.csv`. The companion
`cube-scaling-time-per-part.pdf`/`.png` and CSV divide that same initial-solution time by the
instance's number of parts; this is a normalization of total search time, not a marginal time for
adding the final part. The same command also writes
`cube-scaling-components.pdf`/`.png`, a three-panel plot of exact IK validation, backward
local-PDB search, and the remaining milestone search time, plus
`cube-scaling-components.csv`. The residual is milestone completion time minus the first two
measured components; timeout samples are omitted from component curves because a partial run is
not a completed component measurement.

For timing statistics, repeat every logical cell in fresh processes with `--repeats N` (the
planner seed is kept fixed, so this measures machine/optimizer variation without changing the
benchmark instance):

```bash
python3 benchmarks/run_process_wrench_full_matrix.py \
    --output benchmarks/results/process-wrench-full-matrix-repeated \
    --repeats 3 --optimistic-no-wrench
```

The aggregate CSV/JSON reports include the mean and population standard deviation over repeats.
The same runner also writes `*-cost.png`/`*-cost.pdf`, using the identical robot-count panels,
and wrench colors but plotting initial witnessed plan cost on a linear scale instead of runtime.
With plotting enabled it additionally writes `process-wrench-scaling-legend.tex`, a standalone
four-column PGFPlots legend using the exact line, marker, and hexadecimal color definitions from
the figures. It requires `pgfplots` with compatibility level 1.18.
`cube-scaling-task-attempts.pdf`/`.png` additionally splits exact lazy-validation attempts into
pick, handover, add-grasp, place, and assemble primitives; its CSV includes both attempts and
failures. These are calls reaching `RobotIK.bind`, including repeated open-witness effort—not all
symbolic successors generated by the search.

This benchmark uses `cube-plates-2x2x3`, the existing 12-cell cube-part lattice, because the
ordinary `cube` scenario is specifically a 2x2x2 lattice and therefore stops at eight parts. All
runs use the current `standard` search profile, the established UR10 cube embodiment, and vacuum
grippers on every arm. The plotted default is the solver-reported milestone completion time: the
time at which the first complete, exactly witnessed task plan exists. It excludes process startup
and serialization, and the runner disables later witness optimization and motion planning.

A strict sequence control with the same final 2x2x3 geometry is set up separately. It never builds
detached plates: every cube mates face-to-face with its immediate predecessor. Preview it now and
run it later with:

```bash
python3 benchmarks/run_cube_scaling.py --piecewise --dry-run
python3 benchmarks/run_cube_scaling.py --piecewise \
    --output benchmarks/results/cube-piecewise-scaling
```

For a command preview or a small end-to-end check:

```bash
python3 benchmarks/run_cube_scaling.py --dry-run
python3 benchmarks/run_cube_scaling.py \
    --only '^cube-01p' --output /tmp/cube-scaling-smoke
```

For the paired no-wrench ablation at parts 4/8/10 and robot counts 2/4, use the expanded run-id
filter below. `--no-wrench` creates a provenance-captured requested manifest rather than modifying
the repository manifest:

```bash
python3 benchmarks/run_cube_scaling.py --no-wrench \
    --only '^(cube-04p|cube-08p|cube-10p)__(2-robots|4-robots)__seed-' \
    --output benchmarks/results/cube-scaling-no-wrench-subset
```

The same wrapper exposes the two local-fill projection/successor ablations without duplicating a
manifest. `--fill-target-support-quotient` is valid only together with `--no-wrench`: it removes
the labels of holders already on the target component from the *local reverse PDB*, while leaving
the mover labelled and leaving every concrete successor and IK check unchanged. `--fill-scope`
selects the concrete successor schedule: `standard` is the default local-first, then all-part
completeness widening; `eager-all` starts broad; and `local-only` disables that final widening.
For example:

```bash
python3 benchmarks/run_cube_scaling.py --no-wrench --fill-target-support-quotient \
    --output benchmarks/results/cube-scaling-no-wrench-target-support-quotient

python3 benchmarks/run_cube_scaling.py --no-wrench --fill-scope eager-all \
    --only '^(cube-06p|cube-10p)__(2-robots|4-robots)__seed-' \
    --timeout 180 --output benchmarks/results/cube-scaling-no-wrench-scope-eager-all
```

For the strict piece-by-piece version of that same subset:

```bash
python3 benchmarks/run_cube_scaling.py --piecewise --no-wrench \
    --only '^(cube-04p|cube-08p|cube-10p)__(2-robots|4-robots)__seed-' \
    --output benchmarks/results/cube-piecewise-scaling-no-wrench-subset
```

The standard profile's component-grasp policy bounds each multi-part rigid body's active grasp
table without truncating singleton acquisition grasps. Component K=32 and singleton K=64 are
separate alphabets: the smaller component cap avoids multiplying contacts by every constituent,
and a no-plan result automatically retries the old full component union:

```bash
python3 benchmarks/run_cube_scaling.py --no-wrench --component-grasp-keep 32 \
    --output benchmarks/results/cube-scaling-component-grasps-k32-no-wrench-lazy
```

On the one-seed dependency-cube subset, this retained the same plan costs and reduced 10-part
wall time from 60.33 to 33.78 seconds (2 robots) and 63.67 to 35.19 seconds (4 robots).
Milestone-search time fell from 10.54 to 6.31 seconds and 13.45 to 7.61 seconds, respectively.
The strict piecewise control improved more modestly: 10-part wall time changed from 21.92 to
17.24 seconds and 20.65 to 17.99 seconds.  This supports the narrow claim that repeated large
component alphabets are a real module-DAG cost, not the stronger claim that they explain all
wrench-constrained scaling.

The complete 1--12 by 2--4 no-wrench matrix is stored at that path.  With lazy table
construction, 12-part wall time is 18.57/20.06/21.80 seconds and witnessed-search time is
2.34/3.94/5.33 seconds for 2/3/4 robots, versus 124.56/118.58/111.38 and
22.03/22.92/23.91 seconds for the uncapped eager matched run.  The instrumented 2-robot run
materialized 12/133 tables (157 rows) during isolated preflight and 23/133 (475 rows) during
search.  All 36 compact runs solved directly; none used the full-union completion retry.

The cap-only matched wrench-on 4/6-part subset (before lazy table construction) confirms that
distinction.  At 6 parts, K=32 reduced
wall time from 44.16 to 30.83 seconds (2 robots) and 317.69 to 259.43 seconds (4 robots), while
the 4-part points changed only from 23.24 to 21.85 and 155.98 to 141.89 seconds.  Most of the
remaining 4-robot time is exact wrench/IK validation, not component-table construction.

The matched standard-profile promotion regression is reproducible with:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/component_grasp_standard_regression.json \
    --output benchmarks/results/component-grasp-standard-regression
```

On seed 11023, K=32 matched Car Mini and Chair plan cost and improved Cross cost (40 to 38).
Wall time changed from 46.30 to 36.53 seconds on OpenArm Car Mini, 109.00 to 77.15 seconds on
OpenArm Cross, and 76.18 to 28.38 seconds on mixed-tool Chair. No run needed full-union
completion; all three saved compact plans reloaded with zero IK calls.

In the one-seed run, every point required 0.46--3.74 seconds of milestone search, versus
4.77--172.68 seconds for the corresponding wrench-enabled deferred subset. Although the edge
selector differs in that earlier subset, the action traces explain the qualitative gap: without
wrench constraints most joins reduce to direct pick/assemble/release chains, no grasp or yaw
widening is triggered, and only 12--56 IK binds are attempted. Wrench feasibility, not raw part
count alone, creates the support/regrasp alternatives whose branching interacts with the initial
20-grasp prefix.

The same matched subset can evaluate physical-query deferral on the module and strict-sequence
families without changing either checked-in manifest:

```bash
python3 benchmarks/run_cube_scaling.py --lazy-edge-selector defer-failed \
    --only '^(cube-04p|cube-08p|cube-10p)__(2-robots|4-robots)__seed-' \
    --output benchmarks/results/cube-scaling-defer-failed-subset

python3 benchmarks/run_cube_scaling.py --piecewise --lazy-edge-selector defer-failed \
    --only '^(cube-04p|cube-08p|cube-10p)__(2-robots|4-robots)__seed-' \
    --output benchmarks/results/cube-piecewise-scaling-defer-failed-subset
```

To regenerate a plot from an existing partial or complete result directory without rerunning
search:

```bash
python3 benchmarks/plot_cube_scaling.py benchmarks/results/cube-scaling
```

The manifest defaults to one sample per point. For repeated trials, supply the seeds explicitly;
the same plotter then draws the per-point mean and a population-standard-deviation band:

```bash
python3 benchmarks/run_cube_scaling.py \
    --seeds 11023 11024 11025 \
    --output benchmarks/results/cube-scaling-3-seeds
```

### Cube scaling versus wrench capability

The two-robot wrench-envelope experiment holds the standard all-vacuum UR10 problem fixed and
multiplies every vacuum force and torque capability by 1, 2, 5, 10, and 100. It is not a
no-wrench approximation: each line retains all wrench checks, with only the admissible envelope
changed. Run or resume its 12-by-5 matrix with:

```bash
python3 benchmarks/run_cube_wrench_envelope_scaling.py \
    --output benchmarks/results/cube-wrench-envelope-scaling-2r
python3 benchmarks/run_cube_wrench_envelope_scaling.py \
    --resume --output benchmarks/results/cube-wrench-envelope-scaling-2r
```

The resulting PDF/PNG uses a logarithmic time axis and the established two-robot blue, with
decreasing opacity for increasingly permissive envelopes. Its CSV retains failures/timeouts as
censored observations rather than replacing them with solution times.

The complementary insertion-load experiment keeps the vacuum envelope and physical gravity fixed,
then compares topological support, finite-contact support under weight alone, and increasingly
demanding assembly forces. `lambda in {0, 0.001, 0.01, 0.1, 0.5, 1, 2}` multiplies the standard
10 N nominal insertion force, so the concrete values are 0, 0.01, 0.1, 1, 5, 10, and 20 N;
lambda zero is gravity-only. The standard profile has insertion uncertainty off, so this proof
of concept varies one process-wrench
magnitude rather than silently changing its direction distribution. `--with-no-wrench` adds one
independent topological control per part count (not one duplicate per lambda):

```bash
python3 benchmarks/run_cube_process_wrench_scaling.py --with-no-wrench \
    --output benchmarks/results/cube-process-wrench-scaling-2r
python3 benchmarks/run_cube_process_wrench_scaling.py --with-no-wrench --resume \
    --output benchmarks/results/cube-process-wrench-scaling-2r
```

Use `--piecewise` for the matched structural ablation in which
`cube-sequence-2x2x3` supplies one complete part order instead of the plate-module dependency
graph. No contact, load, robot, seed, or search-profile setting changes:

```bash
python3 benchmarks/run_cube_process_wrench_scaling.py --piecewise --with-no-wrench \
    --output benchmarks/results/cube-piecewise-process-wrench-scaling-2r
python3 benchmarks/run_cube_process_wrench_scaling.py --piecewise --with-no-wrench \
    --full-grid --resume \
    --output benchmarks/results/cube-piecewise-process-wrench-scaling-2r
```

The default cube subset evaluates every lambda at 2, 4, and 8 parts, but only lambda
`{0, 0.5, 1, 2}` at 10 parts. The first 8-part tiny-load run took several minutes, so the three
unmeasured 10-part tiny-load points are explicit with `--full-grid`, not an accidental default.

The plot uses a black dotted line for topological support, grey for gravity-only finite contacts,
and progressively darker blue for increasing insertion load. Its CSV also retains plan action
count, plan cost, IK time, backward search time, failures, and timeouts.

For the overnight paper matrix, use the strict piecewise cube at every prefix from 2 through 12
parts and the Cross at every prefix from 2 through 7. Both use 2, 3, and 4 UR10 robots and
`lambda={0, 0.01, 0.1, 0.5, 1, 2}`; cube arms are all vacuum and Cross arms are all
two-finger. Gravity remains enabled for every finite-wrench cell. A separate no-wrench manifest
adds exactly one topological control per `(assembly, parts, robots)` tuple:

```bash
python3 benchmarks/run_process_wrench_full_matrix.py \
    --output benchmarks/results/process-wrench-full-matrix
```

This is 357 sequential, fresh-process, single-seed searches in total: 231 cube and 126 Cross.
Each child retains the 6 GiB and 900 s guards. Resume the same directory after interruption:

Add `--optimistic-no-wrench` to run a second, matched topological control with the
optimistic/Dogar-style solver. The plots then show both the standard milestone solver's dotted
black no-wrench curve and the optimistic solver's dashed red no-wrench curve. This adds 51
single-seed samples; together with `--repeats 3`, the complete experiment contains 1224 fresh
processes. The optimistic run disables its exact fallback, local reverse heuristic, and yaw
widening, matching the paper comparison rather than silently reverting to our method.

```bash
python3 benchmarks/run_process_wrench_full_matrix.py --resume \
    --output benchmarks/results/process-wrench-full-matrix
```

To regenerate the matrix figures with the repository paper style, without titles or legends,
pass `--paper`. This affects plotting only; it does not change any benchmark sample:

```bash
python3 benchmarks/run_process_wrench_full_matrix.py --resume --paper \
    --output benchmarks/results/process-wrench-full-matrix
```

Use `--assembly cube` or `--assembly cross` to run only one family. The wrapper produces one
three-panel figure per family (2/3/4 robots), raw logs and plans for every successful cell, and
CSV files that retain failures as censored observations. It performs search only: witness
optimization and motion planning remain disabled by the benchmark runner.

The matched two-finger Cross subset uses 2, 4, and 7 parts with the same lambda grid, gravity,
robot count, seed, timeout, and resource guard:

```bash
python3 benchmarks/run_cross_process_wrench_scaling.py --with-no-wrench \
    --output benchmarks/results/cross-process-wrench-scaling-2r
```

The matched directional-gripper control replaces the vacuum-only cube by the Cross, uses two
UR10 robots with explicitly two-finger-only grippers, and truncates the assembly from one through
seven parts:

```bash
python3 benchmarks/run_cross_wrench_envelope_scaling.py \
    --output benchmarks/results/cross-wrench-envelope-scaling-2r
```

Here `--two-finger-scale` uniformly multiplies every two-finger force and torque limit. The five
scales, seed, search profile, resource guards, plotting convention, and timing definition are
otherwise identical to the cube experiment.

Both figures can include a matched dashed no-wrench limit. The controls are separate suites so
they cannot be confused with a finite multiplier or duplicated once per scale:

```bash
python3 benchmarks/run_cube_scaling.py --no-wrench \
    --only '^(cube-0[1-9]p|cube-10p)__2-robots__seed-' \
    --output benchmarks/results/cube-wrench-envelope-no-wrench-2r

python3 benchmarks/search_runner.py \
    benchmarks/manifests/cross_no_wrench_scaling.json \
    --output benchmarks/results/cross-wrench-envelope-no-wrench-2r
```

Pass the resulting directory to the envelope plotter with `--no-wrench-suite`. The dashed line
uses the same two-robot blue and remains explicitly labelled “no wrench”; it is not called an
infinite wrench multiplier because disabling the wrench model also removes its computation and
state distinctions.

To diagnose whether missing tangential support at the table causes the remaining gap, run the
matched 100x-envelope cases with an intentionally large table coefficient, `mu=10`:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/cube_wrench_x100_high_friction.json \
    --output benchmarks/results/cube-wrench-x100-mu10-2r

python3 benchmarks/search_runner.py \
    benchmarks/manifests/cross_wrench_x100_high_friction.json \
    --output benchmarks/results/cross-wrench-x100-mu10-2r
```

Supply these directories to the corresponding plot command with `--high-friction-suite`. The
purple dash-dot curve is a deliberately nonphysical stress test, not a calibrated friction
setting; it isolates the effect of adding a near-sticking table contact while retaining wrench
checks.

For the matched robot-embodiment result on the scattered five-part Car Mini, run:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/paper_car_mini_embodiment_scatter.json \
    --output benchmarks/results/paper-car-mini-embodiment-scatter
```

This uses the current `standard` search profile for both four-robot cases, changes only
`ur5_limited` versus `openarm`, and applies a 900 s / 6 GiB guard to each fresh sequential
process. The common `--scatter-spacing 0.2` initialization uses standard's persisted compact
centred lattice for both embodiments; it is not replaced by an OpenArm-specific grid.

The guarded seed-11023 run in
`benchmarks/results/paper-car-mini-embodiment-scatter-centered-standard` gave:

| robot | status | wall time | actions / cost | peak RSS |
|---|---|---:|---:|---:|
| UR5-limited | success | 33.98 s | 21 / 26 | 374 MiB |
| OpenArm | success | 64.62 s | 20 / 26 | 420 MiB |

The former standard used a fixed x=-0.5 one-row scatter. Analytic reach preflight showed the
car body had zero exact within-limit OpenArm branches for every arm; symbolic widening could not
repair an initially unreachable loose part. After centring the layout, conventional-only grasp
staging still spent 6,280 effort slots on `obj_2`, whereas the full hybrid DB solved it in 13.
Standard now uses the compact centred layout and full hybrid DB. The current OpenArm run needed
480 binds (405 failed, 14.2 s bind wall), reported 41.21 s milestone search time, and completed
without repairs; UR5 remains within the same order of runtime.

## Quick smoke test

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/search_smoke.json \
    --output /tmp/assembly-search-smoke
```

Preview every expanded command without running it:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/fixture_scaling.json --dry-run
```

Run or resume the scalable real-IK fixture benchmark:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/fixture_scaling.json \
    --output benchmarks/results/fixture-rack-scaling

python3 benchmarks/search_runner.py \
    benchmarks/manifests/fixture_scaling.json \
    --output benchmarks/results/fixture-rack-scaling --resume
```

That suite stops at nine parts because the growing independent-station partial order is already
a substantial backward-search stressor. The companion suite isolates abstraction and milestone
growth without paying robot IK and continues through all 13 parts:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/fixture_abstract_scaling.json \
    --output benchmarks/results/fixture-rack-abstract-scaling
```

The corresponding repeated comparison on Car-mini, Cross, Chair, and Peg stool is:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/core_assemblies.json \
    --output benchmarks/results/core-assemblies
```

This is deliberately a long suite: two witness policies, five seeds, and four assemblies make
40 fresh real-IK searches. Use `--only`, `--max-runs`, and `/tmp` while developing; reserve the
repository result path for the frozen paper configuration.

The finite placement-yaw scheduling ablation compares the historical one-yaw action space with
the standard four-yaw per-fill widening on the same four representative assemblies:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/placement_yaw_widening.json \
    --output benchmarks/results/placement-yaw-widening
```

Every run is sequential and guarded at 6 GiB. The runner disables post-search witness
optimization and motion planning in both columns, so the reported time isolates witnessed
search. Chair uses its validated seed 1; the other cases use seed 11023.

The post-audit `widened-n4` regression is stored in
`benchmarks/results/placement-yaw-widening-fixed-current`. It uses the corrected V4/V2 raw-cloud
contract and full-yaw terminal settle, while keeping the optional policy guide off:

| scene | search | process | actions / cost | peak RSS |
|---|---:|---:|---:|---:|
| OpenArm Car Mini-5 | 26.06 s | 52.41 s | 21 / 26 | 342 MiB |
| OpenArm Cross-7 | 121.81 s | 142.36 s | 30 / 40 | 612 MiB |
| mixed-tool Chair-8 | 141.67 s | 195.11 s | 34 / 43 | 1,898 MiB |

All three returned without milestone repair or yaw widening inside a part fill. Cross did use
the corrected terminal path: its final hands-off placement has augmented id `5580554`, outside
the legacy six-face table. A `--config PLAN --load-plan PLAN` reconstruction consumed that
action with zero IK calls, confirming that the compact search prefix no longer leaks into plan
consumers. The process column includes deterministic grasp-scene construction and the
transactional release-hoist post-pass; Chair's hoist candidate failed re-certification and was
correctly rolled back, so its process time is not a pure search measurement.

The reverse-policy widening ablation compares the same local finite-column action space with and
without the optional Bellman-frontier rescue pass:

```bash
PYTHONMALLOC=malloc MALLOC_ARENA_MAX=1 \
python3 benchmarks/search_runner.py \
    benchmarks/manifests/reverse_policy_widening.json \
    --output benchmarks/results/reverse-policy-widening
```

Its seed-11023/Chair-seed-1 audit is stored in
`benchmarks/results/reverse-policy-widening-current`. All six runs succeeded with identical
per-scene plans, costs, IK call/failure counts, and peak memory:

| scene | local columns search | Bellman frontier search | actions / cost | IK calls / failures | frontier activations |
|---|---:|---:|---:|---:|---:|
| OpenArm Car Mini-5 | 36.87 s | 28.58 s | 21 / 26 | 212 / 148 | 0 |
| OpenArm Cross-7 | 128.42 s | 160.27 s | 30 / 40 | 1250 / 971 | 0 |
| mixed-tool Chair-8 | 167.57 s | 195.05 s | 34 / 43 | 679 / 458 | 0 |

Because every compact fill succeeded, the optional full-master policy PDB and widening frontier
were never queried. These wall-time differences track continuous-solver/system variance and do
not establish a speedup or slowdown. The suite is a non-interference regression; the forced-miss
unit cases exercise actual multi-branch policy activation. The benchmark parser records
`policy_grasp_batches`, `policy_yaw_orbits`, and `policy_failed_probes` for a future natural
post-compact benchmark.

The reviewer-facing local-fill ablation holds the supplied dependency structure and every
feasibility gate fixed while comparing counting guidance, a bounded direct-skill probe, and the
query-driven reverse PDB:

```bash
prlimit --as=6442450944 -- \
python3 benchmarks/search_runner.py \
    benchmarks/manifests/local_fill_reviewer.json \
    --output benchmarks/results/local-fill-reviewer
```

It contains Car-mini, Cross-7, mixed-tool Chair-8, and the independent-branch fixture rack, with
three seeds each. The direct probe tries at most `3 * number_of_robots` candidate role
assignments per join and then invokes the unchanged complete local graph. Its failure is never an
infeasibility result. This is intentionally a simpler baseline, not another completeness claim.

## Paper method x physics matrix

`paper_search_matrix.json` is the compact paper comparison. It crosses seven method rows with two
physics columns on Car-mini, Cross, mixed-tool Chair, and the unseen fixture rack:

| row | meaning |
|---|---|
| `milestones-counting` | established structural milestones and ordinary counting-guided fills |
| `milestones-open-witness` | counting milestones with bounded fair reopening of stochastic IK failures |
| `reverse-local-exact` | current query-driven reverse abstraction and weight-one local A* |
| `reverse-local-exact-open` | the same exact discrete ordering with bounded fair reopening of stochastic IK failures |
| `reverse-local-w100` | the same uniformly projected abstraction with weighted local A* (`w=100`) |
| `reverse-local-w100-open` | the same weighted ordering with bounded fair reopening of stochastic IK failures |
| `direct-first` | bounded direct manipulation schemas, then the unchanged counting graph |

| column | meaning |
|---|---|
| `geometry-only` | `--no-wrench`: no load, gravity-stability, or grasp-capacity checks |
| `noisy-5N` | full wrench checks, 5 N axial insertion, and worst-case lateral load up to 5% |

Everything else—assembly dependencies, grasp model, real IK, collision gates, milestone policy,
and witness persistence—is held fixed. `direct-first` retains the complete local fallback; its
schema failure is therefore overhead/diagnostic data, not an infeasibility declaration.

Open-witness scheduling is crossed with both reverse-local orderings because it is an orthogonal
choice: the reverse PDB orders the discrete frontier, whereas open witness determines whether a
failed stochastic IK query is permanently blacklisted or revisited with bounded increasing
effort. In an `exact-open` row, *exact* refers only to weight-one discrete ordering; the finite
open-witness schedule is not an IK infeasibility proof.

Every reverse-local row now projects **all** grasp labels by the same fact signature. Earlier
experiments that kept only the active/end grasp raw were an unprincipled role-specific exception
and are not part of the paper matrix.

Run the complete 4 assemblies x 7 methods x 2 physics settings x 3 seeds matrix sequentially:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/paper_search_matrix.json \
    --output benchmarks/results/paper-search-matrix
```

The manifest applies a 6 GiB address-space limit to **each child search** using `prlimit`; a bad
variant cannot consume the laptop unchecked. Chair has a 900 s per-run timeout and the other
cases 600 s. The suite is intentionally an overnight experiment. It is resumable without mixing
source versions:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/paper_search_matrix.json \
    --output benchmarks/results/paper-search-matrix --resume
```

A guarded one-seed Car-mini smoke run after adopting uniform projection produced the following
results. These diagnose the matrix contract; they are not substitutes for the three-seed paper
statistics.

| method | geometry-only | noisy-5N |
|---|---:|---:|
| counting | 14.4 s / cost 18 | 11.7 s / cost 18 |
| counting + open | 5.9 s / cost 18 | 5.4 s / cost 18 |
| reverse exact | 8.9 s / cost 18 | 6.2 s / cost 18 |
| reverse exact + open | 6.8 s / cost 18 | 4.4 s / cost 18 |
| reverse w100 | 37.6 s / cost 26 | 4.8 s / cost 18 |
| reverse w100 + open | 8.3 s / cost 18 | 4.0 s / cost 18 |
| direct-first | 16.1 s / cost 18 | 12.6 s / cost 18 |

The useful signal is not that open witnesses always win—one seed cannot establish that—but that
the weighted geometry failure mode is an IK-allocation problem: fair reopening changes it from
420/246 IK calls/failures and a cost-26 plan to 162/118 and a cost-18 plan. The matched multi-seed
rows test whether that effect is repeatable.

`paper_search_stress_4r.json` applies the same seven rows and two physics columns at one seed to
Cross-7, mixed-tool Chair-8, and Fixture-rack-9 with four robots each. It is the deliberately
larger fast stress screen before spending three seeds on the paper matrix:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/paper_search_stress_4r.json \
    --output benchmarks/results/paper-search-stress-4r
```

After that screen, `search_finalists_4r.json` spends two additional seeds only on the two
principled finalists, counting+open and reverse-exact+open:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/search_finalists_4r.json \
    --output benchmarks/results/search-finalists-4r
```

## Manipulation-locality ablation

The selected structural standard has **no global grasp/brace phase restrictions**. Its local
reverse PDB supplies heuristic values, not restricted action suggestions. Consequently, a
global “no-hints” row would be identical to the standard. The independent raw incoming-grasp
ordering/regrasp signal is cleanly ablated with `--no-useful-grasps`; it never prunes an action.
`--no-fill-local-h` is the separate and much larger counting-versus-reverse-heuristic comparison
in the paper matrix; it is not hint removal.

`search_manipulation_scope_ablation.json` holds the reverse PDB, exact weight-one local search,
open witnesses, and all feasibility gates fixed. It compares eager all-part fills against the
narrow-to-broad scope widening under that historical exact-search control. Both policies
ultimately permit the same manipulation actions; only their scheduling differs. Because the
current standard is greedy, this manifest is an exact-policy scope ablation rather than a literal
expansion of today's named profile; use the profile runner for the operational baseline.

The standard first exhausts every local-scope grasp/bracing phase. Only then may it pick, place,
hand over, reorient, add a grasp to, or release a future part. It may not assemble that part early,
and every temporarily held blocker must be settled at the local endpoint. This keeps the concrete
state consistent with the supplied dependency/milestone search while preventing a restricted
local failure from becoming a false infeasibility conclusion. The eager ablation uses
`--fill-touch-all` to skip the cheap first pass.

Run the guarded one-seed, four-robot matrix (3 scenes x 2 rows x 2 physics columns):

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/search_manipulation_scope_ablation.json \
    --output benchmarks/results/search-manipulation-scope-ablation
```

For the cheapest initial check, select Cross geometry only:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/search_manipulation_scope_ablation.json \
    --only 'cross-7p-4r.*geometry-only' \
    --output /tmp/search-manipulation-scope-cross
```

The guarded seed-11023 matrix produced:

| scene / physics | eager all parts | local-to-all widening |
|---|---:|---:|
| Cross-7 geometry | 20.1 s / 2114 MB / cost 28 | **5.5 s / 218 MB / cost 28** |
| Cross-7 noisy 5 N | 16.4 s / 1460 MB / cost 28 | **7.8 s / 294 MB / cost 28** |
| Chair-8 geometry | 43.3 s / 1460 MB / cost 31 | **39.1 s / 168 MB / cost 31** |
| Chair-8 noisy 5 N | **failed**, 57.0 s / 4477 MB | **40.9 s / 388 MB / cost 31** |
| Fixture-9 geometry | 76.2 s / 160 MB / cost 24 | **37.0 s / 150 MB / cost 24** |
| Fixture-9 noisy 5 N | **memory exhausted**, 55.2 s / 4488 MB | **62.5 s / 913 MB / cost 40** |

The Chair eager/noisy child ended in a low-level error under the address-space guard; Fixture
eager/noisy reported `MemoryError` explicitly. Neither is an infeasibility certificate. Widening
succeeded in every cell and preserved the successful eager plan cost. This is why the standard
starts locally and exposes the same broad graph only after local exhaustion.

The seed-11023 full screen completed all 42 cells. It ruled out `w=100` as the general default:
although occasionally fast, it raised plan cost from 28 to 36 on Cross geometry, from 31--35 to
37--39 on Chair, and from 24/40 to 30/48 on Fixture geometry/noisy physics. Direct-first was a
reasonable geometry baseline but took 65.7 s on noisy Chair and 143.1 s on noisy Fixture.

Combining the screen with the two finalist seeds gives the following three-seed means. Each of
the 36 finalist runs succeeded; entries are wall time / plan cost.

| scene / physics | counting + open | reverse exact + open |
|---|---:|---:|
| Cross-7 geometry | 7.8 s / 28 | **5.4 s / 28** |
| Cross-7 noisy 5 N | 12.0 s / 28 | **6.6 s / 28** |
| Chair-8 geometry | **22.7 s / 35** | 26.4 s / **31** |
| Chair-8 noisy 5 N | **26.9 s / 31** | 30.7 s / 31 |
| Fixture-9 geometry | **38.1 s / 24** | 46.6 s / 24 |
| Fixture-9 noisy 5 N | 125.9 s / 40 | **75.7 s / 40** |

Summed over the six equally weighted cells, reverse-exact+open takes 191.4 s versus 233.3 s for
counting+open, an 18% reduction. More importantly for the branching wrench stress, its noisy
Fixture peak RSS is about 0.91 GiB instead of 4.22 GiB. This established the value of uniform
reverse-local guidance and bounded open witnesses. A later matched comparison found that using
the same heuristic greedily is the clearer time-to-first-plan policy, while local
greedy-then-exact confirmation was not a consistent win. The recommended operational search is:

```text
--weight 1 --milestones --milestone-oracle structural
--fill-local-h --fill-greedy --no-fill-exact
--open-witness --open-witness-batch 7 --open-witness-epochs 2
```

This is a deliberately simple universal choice, not a scene classifier. Global weight remains
one for the separate exact-search ablations; `--milestones-anytime` now improves its incumbent
with bounded one- and two-milestone LNS neighborhoods. The local `--fill-greedy` flag changes
only milestone-fill queue ordering. The
`--fill-exact --no-fill-greedy` row remains the weight-one local-search ablation. Two open-witness
epochs remain a bounded operational policy, not a continuous infeasibility proof; an
unlimited/deadline-bounded open schedule is the separate asymptotic experiment.

For a quick runner/contract check, execute the geometry-only Cross rows at one seed:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/paper_search_matrix.json \
    --only 'cross-7p-4r.*geometry-only.*seed-11023' \
    --output /tmp/paper-search-matrix-smoke
```

In addition to the ordinary raw artifacts and long-form summaries, matrix manifests write:

```text
matrix.md    compact method x physics tables, one per assembly
matrix.csv   long-form aggregate rows for plotting/statistics
matrix.json  coordinates, descriptions, and all aggregate metrics
```

Each Markdown cell contains success count, mean/population-standard-deviation wall time, mean
peak RSS, successful-plan cost, and IK calls/failures. Failed/time-limited samples remain visible
in success rate and runtime rather than disappearing from the average.

During development, select a subset without editing the manifest:

```bash
python3 benchmarks/search_runner.py \
    benchmarks/manifests/fixture_scaling.json \
    --only 'rack-(3p|5p).*open' --max-runs 2 \
    --output /tmp/fixture-subset
```

The normal runner exit status is zero only when every selected run succeeds. A failed search,
timeout, or solver crash therefore also fails a CI or shell pipeline, while its partial artifacts
and aggregate rows remain available for diagnosis.

## Manifest schema

```json
{
  "schema_version": 1,
  "name": "experiment-name",
  "description": "what hypothesis this suite tests",
  "timeout_s": 900,
  "seeds": [0, 1, 2, 3, 4],
  "env": {"OPTIONAL_VARIABLE": "value"},
  "defaults": ["--flags", "shared", "by", "every", "run"],
  "cases": [
    {"name": "instance-a", "args": ["--assembly", "cross", "--parts", "7"]}
  ],
  "variants": [
    {"name": "method-a", "args": ["--open-witness"]}
  ]
}
```

Runs are the deterministic Cartesian product `case x variant x seed`. A case or variant may
provide its own `seeds`, `timeout_s`, `tags`, or `"enabled": false`. Argument order is shared
defaults, case arguments, then variant arguments. Avoid assigning the same argparse destination
twice: the planner deliberately reports duplicate flags because argparse would silently retain
only the last value.

The runner reserves flags that could invalidate the comparison or write elsewhere, including
`--save`, `--seed`, viewers, witness postprocessing, and motion planning.

## Output contract

Each suite directory contains:

```text
manifest.json             exact manifest snapshot
provenance.json           git/source hash, dirty files, Python, host, CPU count
results.json              all completed per-run records
runs.csv                  flat per-run metrics
summary.json              aggregate groups and provenance
summary.csv               mean/std and success rate by case and variant
matrix.{md,csv,json}      method x physics views (matrix manifests only)
runs/<run-id>/
    run.json              resolved command and fingerprint
    stdout.log            complete planner output
    time.txt              GNU-time wall/RSS/exit metrics
    plan.json              replayable successful plan, when found
    result.json            parsed structured metrics and artifact paths
```

The source hash covers the planner/search/model files directly, including uncommitted edits. A
`--resume` result is reused only when both its resolved run specification and source hash match.

Parsed metrics currently include:

- end-to-end wall time and per-process peak RSS;
- plan actions, action cost, kind histogram, and persisted witness count;
- backward-state count, pick facts, and construction time;
- milestone solve time, repairs, failed fills, and width-window activity;
- IK binds, failures, cache hits, wall time, and gate rejection counts; and
- seeded, unseeded, restart, and precheck NLP work; and
- direct-skeleton successes/fallbacks, candidate paths, failed IK edges, and probe time.

The raw log remains authoritative. A parser miss leaves a metric null rather than inventing a
zero.

## Persisted-witness continuity diagnostic

Witness optimization is intentionally outside the search runner: it is a post-pass and should
not contaminate the search-time matrix. Compare optimized plan artifacts separately with:

```bash
python3 benchmarks/witness_quality.py \
    experiments/OLD-PLAN.json experiments/NEW-PLAN.json \
    --json /tmp/witness-ablation.json
```

The tool reports the serialized keyframe chain's total/max joint displacement, coordinate
changes above a specified threshold, and transitions which an old modulo-2pi metric would have
hidden. It does not interpolate, collision-check, or claim motion feasibility; the motion planner
and path preflight remain authoritative. For a paper ablation, hold the search plan, seed,
candidate count, and clearance-floor formulation fixed, then report these witness metrics
together with motion-planning success/path cost. A one-seed three-part OpenArm Cross development
check reduced serialized chain L2 travel only from 33.716 to 33.677 rad and removed no large
coordinate transition. That validates the implementation but is not evidence that redundant-IK
harvesting alone cures the visible full-Cross rotations.

For the fixture-rack failure analysis, `fixture_support_diagnostic.json` holds the direct schema,
scene, real-IK, collision, and clearance contract fixed and varies only `--wrench`. This is a
mechanism diagnostic, not a performance ablation: `support-off` asks whether the join is
geometrically realizable, while `support-on` additionally requires a holder/brace that can react
the frictionless-table lateral press.

## Fixture-rack family

`--assembly fixture-rack --parts N` is a scalable family with up to 13 parts:

- one long base rail;
- repeated independent upright posts inserted top-down; and
- one cantilever bracket pressed laterally onto each post.

Parts use heterogeneous cuboid aspect ratios, bracket directions alternate, and every bracket
depends on its own post while stations remain mutually unordered. This creates a growing partial
order instead of the cube family's chain. Under the frictionless table wrench model, the lateral
bracket press also requires the planner to hold or brace the target rather than letting the table
provide fictitious shear support.

Useful scales are 3, 5, 7, 9, 11, and 13 parts, corresponding to one through six complete
post/bracket stations. Even `--parts` values deliberately leave the last station without its
bracket and can be used as truncation controls.

The independent branches are intentionally not cheap. In the initial seed-0 wrench/geometric
check, the 9-part instance built 72,063 co-reachable backward states in 37.6 s and completed a
40-action milestone plan in 97.3 s. This is why the real-IK witness-policy matrix stops at nine
parts and the 11/13-part cases are treated as abstraction-scaling or timeout data rather than
routine smoke tests.

### Ten parts with four mixed grippers

The 10-part, four-robot mixed vacuum/two-finger case crosses a different boundary. The tight
signature heuristic retains the grasp class of every loose held part; vacuum/finger usability
splits even conventional cuboid grasps into distinct classes, and the global backward database
hits its explicit two-million-state guard before forward search. This is reported as an
actionable failure and does not silently select a weaker heuristic. Use:

```text
--optimistic --no-optimistic-fallback --grasp-refine --lazy-fills
```

`--optimistic` replaces the global database with dependency-ordered local fills.
`--no-optimistic-fallback` prevents a failed local fill from re-entering the already-known
intractable global path. `--grasp-refine` tries the conventional primitive prefix before exposing
sampled primitive extras. Lazy local validation is on by default through `--lazy-fills`: the
local search proposes a cheap-feasible path, validates only its IK edges, and incrementally
repairs around refutations. `--no-lazy-fills` restores the eager ablation.

Measured on fixture-rack-10, seed 11023, four UR5s with
`vacuum,two_finger,two_finger,vacuum`: eager optimistic required 2,850 IK binds and 641.5 s of
search; lazy optimistic required 126 binds and 220.2 s. The complete guarded run, including
sampled-scene setup and exact search-witness persistence, took 412.5 s and peaked at 1.91 GiB
RSS. The initial layout also matters: the fixture rail now occupies its own scatter lane, and a
0.2 m pitch wraps after six columns, so the 10-part scene is collision-free and remains inside
the historical `[-0.5,+0.5]` reachable x strip.

### Grasp-prefix widening ablation

The former runtime-oriented recipe starts with the V4 scene-reserved K=20 hybrid prefix and enables
`--grasp-widen --grasp-widen-batch 8 --grasp-widen-max 96`. V4 is a deterministic reordering
of the same raw candidate master as V2: it reserves available pick/assemble scene roles, then
uses the literal V2 tail. Eager K=32 was rejected on centred
OpenArm Car Mini: 89.2 s / 752 binds / 658 failures versus K=20's 41.2 s / 480 / 405, while
initial wheel feasibility increased only 5 to 6. That implementation kept the K=20 baseline
(41.9 s OpenArm, 19.9 s UR5-limited in the post-change confirmations) and widened only after no
plan was returned. A forced-prefix test banning pick IDs 0--19 failed in 1.47 s at K=20, then
solved in 0.13 s at K=28 with grasp 23. The saved plan records K=28 and its content hash. The
current standard uses fixed K=64 instead; these results retain widening as a measured ablation.

V3 task-certificate coverage is retained as an ablation rather than silently selected: despite
soft reach and exact single-grasp insertion-LP facts, it kept only four mate-useful grasps on the
difficult wheels versus V2's seven and took 66.7 s on OpenArm Car Mini. This is evidence against
replacing useful grasp redundancy with one-representative-per-fact coverage.

The follow-up `--grasp-local-widen` ablation moves that refinement inside milestone fills while
keeping stable IDs. Its positive forced-prefix case widens only the failing part (K=20 to 28,
then grasp 23 succeeds). It is not the default: OpenArm Car Mini K=16 produced exactly the
control plan and 212 IK solves but measured 27.1 s search / 324 MiB peak solve RSS, versus
25.6 s / 319 MiB for the global-widen control. The initially implemented policy of exhausting
all prefixes at one boundary was rejected after a K=8 stress run spent 24.8 + 20.4 + 18.0 s on
successive `obj_2` prefixes. The retained policy allows one local growth, yields to milestone
backtracking, and uses one full-master pass only as the final finite completeness net.
The revised K=8 run still reached the 300 s guard in unrestricted bracing; its new funnel
diagnostic reported `obj_2` isolated-pick active/master = 1/1, correctly indicating that more
raw pick columns were not the missing resource.

## Candidate follow-up synthetic families

The benchmark suite should eventually cover mechanisms rather than accumulate arbitrary shapes.
The following families would add distinct planner stressors:

| Family | Physical analogy | Search mechanism isolated | Current prerequisite |
|---|---|---|---|
| repeated staged clamps | rail-mounted angle brackets retained by pins | partial mates, hold-until-fastened, repeated staged support | generalize `slot-pin` into stations |
| detachable rack modules | post/bracket subassemblies built away from the base and then mounted | module milestones, join ordering, width-2 coupling | reuse `connect_parts` |
| shaft and bearing stack | spacers/gears threaded onto a shaft captured between supports | axial insertion, in-hand reorientation, capture ordering | represent the final second-support closure honestly |
| panel with tool-specific inserts | broad vacuum-held panel plus finger-inserted bosses/fasteners | heterogeneous tool capacity and per-kind wrench signatures | measured vacuum/finger parameters |
| tree truss | branching struts attached to a grounded spine | large partial-order width without repeated identical geometry | boxes/cylinders already sufficient |

The staged-clamp family is the best next implementation: it uses mechanisms already present in
the domain, is physically interpretable, and tests whether backward facts understand that a
partially seated bracket must remain supported until its fastener arrives. The shaft family is
more scientifically interesting but should wait until loop-closing/capture semantics are explicit;
encoding it as a convenient tree would repeat the known problem with the old arch benchmark.
