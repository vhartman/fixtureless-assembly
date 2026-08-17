# Assembly planner

This repository searches for multi-robot assembly plans, exports real-arm motion paths with
`mr_bench`, and validates the paths in MuJoCo. See [PIPELINE.md](PIPELINE.md) for the general
workflow and the other solver modes.

## Setup

Use Python 3.12 and the repository environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Motion planning also needs the `mr_bench` source tree. The default location is the sibling
checkout `../mr_bench/src`. Set `MRBENCH_SRC` if it is elsewhere:

```bash
export MRBENCH_SRC=/absolute/path/to/mr_bench/src
```

Use writable caches and a fixed Python hash seed for scripted runs:

```bash
export PYTHONHASHSEED=0
export NUMBA_CACHE_DIR=/tmp/assembly-numba-cache
export MPLCONFIGDIR=/tmp/assembly-mpl-cache
mkdir -p experiments
```

Do not run two search or motion-planning jobs at the same time. Viewer flags are blocking, so
use `--no-view` for search and motion-planning scripts.

## Robust keyframes: current contract

The current defaults make the robustness settings explicit in every fresh search:

- `--witness-margin 0.005` is one hard 5 mm non-intended-pair gate used by the discrete
  search and by every later IK/fusion/funnel solve. A post-pass never applies a stricter gate
  than the search that admitted the action. Saved plans record `_search_witness_margin`;
  loading a plan with a larger requested margin is refused. Older plans without that marker
  are certified at 0 m and must be re-searched, or replayed from an exact `--mp-export-load`
  artifact without resolving IK.
- `--witness-sets 4` keeps up to four same-grasp IK branches. Pre-skill and post-release goals
  are conditional pairs: entering member A can only exit through member A, never jump to a
  different IK branch.
- `--pre-pick 0.05 --pre-place 0.05 --pre-add-grasp 0.05 --pre-insertion 0.05
  --post-release 0.05` complete both sides of the contact skills. Release departure tries a
  deterministic 100/80/60/40/20/10% distance ladder under the same gate; if none works, the
  exported zero-length fallback is reported explicitly.
- `add_grasp -> pre_assemble` shares one all-robot rendezvous keyframe, and adjacent release
  events reuse the assembly endpoint. This removes a pointless extra transport goal. It does
  **not** constrain BiRRT's path interior or run a controller in the planning loop; detours
  during insertion are allowed. `--mp-cograsp` is therefore a diagnostic of interior
  closed-chain drift, not a requirement that ordinary BiRRT project onto that manifold.
- `--optimize-witnesses floor`, `--wrench`, `--grasp-model asymmetric`,
  `--sig-collapsed-h`, and `--sig-components press` remain enabled for the runs below. The
  fresh search writes the floor-optimized witnesses and branch sets; a later `--load-plan`
  reuses them. `--fuse-cograsp` may still re-solve eligible paired keyframes before the final
  export.

For reproducible experiments, spell those defaults out in the search command:

```bash
python3 main.py --solve \
    --assembly ASSEMBLY --parts N --num-robots R --weight inf \
    --wrench --grasp-model asymmetric \
    --sig-collapsed-h --sig-components press \
    --witness-margin 0.005 --witness-sets 4 \
    --pre-pick 0.05 --pre-place 0.05 --pre-add-grasp 0.05 \
    --pre-insertion 0.05 --post-release 0.05 \
    --quiet-rai --no-view --save experiments/PLAN.json
```

Generate motion from that persisted plan, then retry BiRRT from the exact final export without
ever resolving witnesses again:

```bash
python3 main.py \
    --config experiments/PLAN.json --load-plan experiments/PLAN.json \
    --motion-plan --fuse-cograsp --fuse-pre-skill \
    --mp-sequence --no-mp-optimize --mp-runtime 90 --mp-stall-timeout 30 \
    --mp-seed 11023 --mp-shortcut-iters 100 \
    --mp-collision-tolerance 1e-4 --mp-robot-margin 0.1 --mp-margin-ramp 0.1 \
    --mp-cograsp --quiet-rai --no-view \
    --mp-export-save experiments/FINAL_EXPORT.json --mp-save experiments/MP.json

python3 main.py \
    --config experiments/PLAN.json --load-plan experiments/PLAN.json \
    --motion-plan --mp-export-load experiments/FINAL_EXPORT.json \
    --mp-sequence --no-mp-optimize --mp-runtime 90 --mp-stall-timeout 30 \
    --mp-seed 11023 --mp-shortcut-iters 100 --mp-cograsp \
    --quiet-rai --no-view --mp-save experiments/MP_RETRY.json
```

### 2026-08-14 robust evaluation

All four fresh searches used wrench checking, asymmetric grasps, collapsed-h `press`, floor
witness optimization, four-branch witness sets, and the shared 5 mm search/IK gate. All saved
paths reached their terminal mode at `--mp-collision-tolerance 1e-4`.

The assembly-specific search arguments were:

```text
Car:   --assembly car-mini --parts 5 --num-robots 3 --robot ur10_limited
       --seed 11023 --min-sep 0.3 --insert-force 20 --insertion-noise 0.01
       --tf-close-torque 0.5 --ignore-gravity --random-init
Cross: --assembly cross --parts 7 --num-robots 4 --robot ur10 --seed 0
Cube:  --assembly cube --parts 8 --num-robots 4 --robot ur10 --vacuum
       --seed 11023 --check-init
Chair: --assembly chair --parts 8 --num-robots 4 --robot ur10_limited --seed 1
       --grippers vacuum,vacuum,two_finger,two_finger --allow-curved-grasps
       --scatter-spacing 0.2 --min-sep 0.3
```

| assembly | search | MP result | co-grasp diagnostic | artifacts |
|---|---:|---:|---:|---|
| Car-mini, 3 limited UR10 | 25 actions, 12 witness sets | 44 modes, 522 serialized states | 16/16 pairs drift internally; 4/4 eligible keyframe pairs fused | `car-mini-3r-robust-seed11023-*` |
| Cross-7, 4 UR10 | 28 actions, 16 witness sets | 51 modes, 760 states | 0/3 pairs drift | `cross7-4r-robust-seed0-*` |
| Cube-8, 4 vacuum UR10 | 40 actions, 26 witness sets | 70 modes, 995 states | 3/4 pairs drift | `cube8-4r-vacuum-robust-seed11023-*` |
| Chair-8, mixed UR10 | 38 actions, 21 witness sets | 64 modes, 903 states | 1/4 pairs drift | `chair8-4r-mixed-robust-seed1-*` |

Chair requires the established mixed fleet and curved vacuum contacts:

```bash
--grippers vacuum,vacuum,two_finger,two_finger --allow-curved-grasps \
--scatter-spacing 0.2 --min-sep 0.3
```

The Cube rollout used the requested simulation flags and completed 40/40 actions:

```bash
python3 sim.py run \
    experiments/cube8-4r-vacuum-robust-seed11023-plan.json --v2 \
    --mp-path experiments/mp-cube8-4r-vacuum-robust-seed11023.json \
    --actuated --corridor --corridor-rate-clamp --coll-scale 0.8 \
    --spline --limit-rate \
    --record experiments/cube8-4r-vacuum-robust-seed11023-actuated.npz

python3 sim.py replay \
    experiments/cube8-4r-vacuum-robust-seed11023-actuated.npz
```

Strict preflight checked 35,008 samples and the scheduled corridor verified 5,232 samples with
no collision. The NPZ has 34,791 time samples: the motion planner did not return 34k states.
The 995-state saved path was compacted to 322 geometric waypoints, splined to 2,017 waypoints,
then rate/force/corridor retiming produced the recording samples. `--coll-scale 0.8` remains a
debug geometry override, so this rollout is not a certificate for the unscaled robot geometry.

The dated artifact discussions later in this README describe older plans and are retained for
forensics. The robust artifacts and commands above are the current reproducible results.

## Peg-in-hole examples

`peg-hole` is the minimal two-part concavity check. `peg-stool` is the useful follow-up: its
seat is a plain plate containing three blind square sockets, and three round legs are inserted
24.5 mm into those sockets. There is no raised vacuum pedestal/pad on top of the seat. The seat
is one 13-child compound rigid body (plate plus 12 socket walls), so use the sampled grasp
database. The stool needs a mixed fleet: the vacuum gripper holds the broad seat while a
two-finger gripper handles the cylindrical legs.

Milestones do **not** require placement to be disabled. `--milestones` only selects the search
algorithm; placement successors remain enabled by default unless `--no-place` is explicitly
passed. The successful stool plan below uses both milestones and placement.

`--vacuum-engage-force N` covers the load that exists while a cup approaches and seals. With
`--wrench`, every vacuum `pick`, `add_grasp`, and fused `handover` first asks whether the
supports which exist *before* the new grasp can balance an `N`-newton push along the cup's
approach axis at the contact point. The new cup is deliberately absent from that LP, so it
cannot balance its own engagement force. This is separate from the steady attached-grasp
capability: after sealing, the cup preload is an internal balanced load and must not be added
permanently to gravity/insertion wrenches. Default `0` preserves old plans; the stool run uses
5 N. Two-finger grasps are unaffected.

The reproducible wrench-enabled stool search is:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py --solve \
    --assembly peg-stool --parts 4 --num-robots 2 --robot ur10_limited \
    --grippers vacuum,two_finger --weight inf --beam 16 --milestones \
    --scatter-spacing 0.2 --sig-collapsed-h --sig-components press --real-ik \
    --wrench --vacuum-engage-force 5 --insert-force 20 --grasp-model asymmetric \
    --grasp-sampler --grasp-sampler-keep 12 --allow-curved-grasps \
    --bracing --seed 11023 --optimize-witnesses floor \
    --witness-margin 0.005 --witness-branch-budget 5 \
    --fuse-cograsp --tight-fingers --strict-collisions 1e-4 \
    --save experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-plan.json \
    --quiet-rai --no-view
```

This produced 12 actions, nine persisted IK-bearing witnesses, two witness sets, and three
milestone repairs. It uses the wrench model, asymmetric grasps, the 5 N vacuum engagement load,
20 N insertion load, collapsed-h `press` signature, milestones, and placement. The geometry
therefore does not need the earlier no-wrench/free-floating fallback. `--witness-margin 0.005`
is the shared hard search/IK clearance gate; floor witness optimization maximizes additional
clearance above it. `--strict-collisions 1e-4` is the numerical penetration contract repeated
during motion planning.

For this assembly, keep the 5 cm funnel on insertion but disable the auxiliary pick/place/
add-grasp/release funnels. The all-default-funnel problem contained an unsampleable auxiliary
transition, while removing the insertion funnel made the simulated gripper hit the collars too
hard. This command builds the exact final export and found a complete optimized path:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-plan.json \
    --load-plan experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-plan.json \
    --motion-plan --pre-pick 0 --pre-place 0 --pre-add-grasp 0 \
    --pre-insertion 0.05 --post-release 0 \
    --mp-export-save experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-final-export.json \
    --mp-runtime 90 --mp-shortcut-iters 100 --mp-seed 11023 \
    --mp-collision-tolerance 1e-4 --mp-robot-margin 0.1 --mp-margin-ramp 0.1 \
    --mp-save experiments/mp-peg-stool-2r-mixed-wrench-flat-engage5-seed11023-limits.json \
    --quiet-rai --no-view
```

The successful MP file has 116 states and embeds the exact post-fusion export: nine skill and
three pre-skill keyframes. Replay performs zero IK solves. Motion-planning predicates now reject
joint-limit violations as well as collisions/margins; this fixed a sampled shoulder detour that
was outside the UR10 range even though both task endpoints were legal. To retry only motion
planning, replace `--mp-export-save` with
`--mp-export-load experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-final-export.json`;
do not refresh witnesses.

Run and replay the actuated simulation with:

```bash
python3 sim.py run \
    experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-plan.json \
    --v2 \
    --mp-path experiments/mp-peg-stool-2r-mixed-wrench-flat-engage5-seed11023-limits.json \
    --actuated --corridor --corridor-rate-clamp --coll-scale 0.8 \
    --spline --limit-rate --no-force-limits --servo-torque-guard 0.7 \
    --record experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-actuated.npz

python3 sim.py replay \
    experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-actuated.npz
```

The recording completed all 12 actions in 4,136 frames; its worst final position residual was
2.0 mm. Strict preflight checked 3,392 samples and reported both collision tolerance and joint
limits OK. `--servo-torque-guard 0.7` was needed for the long-lever support-switch approach:
it engaged for 167 ticks where contact force was below the 6 N force trigger but grasp torque
was approaching its cap. `--no-force-limits` disables only the predictive speed retimer: it
assigns a table-supported assembly's entire load to one gripper and can incorrectly reduce a
segment to zero speed. MuJoCo contact forces and physical grasp-break limits remain active, and
the search still uses `--wrench`. `--coll-scale 0.8` is the requested debug-only arm collision
shrink, so this rollout is not a clearance certificate for unscaled collision geometry.

## Car-mini with three robots

The saved recording
`experiments/car-mini-3r-asymmetric-seed11023-default-float-actuated.npz` completed all 21
physical actions with seed 11023. Replay it with step 4 below. The current motion export also
couples each `add_grasp -> pre-assemble -> assemble -> release(s)` chain; this changes task IDs,
so do not combine the older `...default-float.json` MP path with the current source. Export the
current `...coupled.json` path with step 2.

The run uses limited UR10 arms, wrench checks, asymmetric two-finger grasps, a 20 N insertion
force, and 0.01 m insertion uncertainty.

Current defaults already enable milestones, real IK, wrench checking, asymmetric grasps,
bracing, strict collisions, tight fingers, floor witness optimization, co-grasp fusion,
release hoisting, pre-insertion, sequential optimized motion planning, 1e-4 MP collision
tolerance, 0.1 m robot margin, 1,000 shortcut attempts, collapsed-h signature search, and the
`press` signature component. The commands below do not repeat most of those defaults; pass
`--sig-collapsed-h --sig-components press` explicitly when a saved experiment needs to make
the search signature unambiguous.

### 1. Search and save the discrete plan

```bash
python3 main.py --solve \
    --assembly car-mini --parts 5 --num-robots 3 --weight inf \
    --robot ur10_limited --seed 11023 --min-sep 0.3 \
    --insert-force 20 --insertion-noise 0.01 --tf-close-torque 0.5 \
    --ignore-gravity --random-init --quiet-rai --no-view \
    --save experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json
```

### 2. Export and save the motion path

Always load the config and plan from the same JSON. Keep the saved plan and MP path together.

```bash
python3 main.py \
    --config experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json \
    --load-plan experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json \
    --motion-plan --mp-runtime 60 --mp-seed 11023 --mp-shortcut-iters 100 \
    --quiet-rai --no-view \
    --mp-export-save experiments/car-mini-3r-asymmetric-seed11023-final-export.json \
    --mp-save experiments/mp-car-mini-3r-asymmetric-seed11023-coupled.json
```

`--mp-export-save` is written before planning starts, even if BiRRT finds no path. It contains
the final post-fusion skill witnesses, every pre-skill keyframe, exact task goals/effects, task
order, export settings and signature. A successful current `--mp-save` also embeds this same
block in the path file. Replayers treat missing data as an error; they never fill it by solving
IK unless an explicit refresh flag is given.

Once that final export exists, regenerate ONLY the motion path (zero witness/pre-skill/fusion
IK solves) with:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json \
    --load-plan experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json \
    --motion-plan \
    --mp-export-load experiments/car-mini-3r-asymmetric-seed11023-final-export.json \
    --mp-sequence --mp-optimize --mp-runtime 60 --mp-seed 11023 \
    --mp-shortcut-iters 100 --mp-collision-tolerance 1e-4 \
    --mp-robot-margin 0.1 --mp-margin-ramp 0.1 --mp-cograsp \
    --quiet-rai --no-view \
    --mp-save experiments/mp-car-mini-3r-asymmetric-seed11023-final.json
```

That older artifact produced 152 states and embedded its final export. The current robust
Car-mini result is the 522-state artifact in the evaluation table above. Its endpoint
keyframes are coupled, while `--mp-cograsp` intentionally reports the ordinary planner's
unconstrained interior detours.

On `--mp-export-load`, the artifact supplies the default collision/clearance settings, but an
option explicitly present on the current command line wins: `--mp-collision-tolerance`,
`--mp-robot-margin`, `--mp-margin-ramp`, and `--mp-table-margin` can therefore be retuned without
resolving any IK. The `[mp-export-settings]` line prints every effective value and labels its
source as `export`, `CLI`, or `CLI/scene`. Exact task/keyframe and structural settings remain
owned by the final export. Merely receiving a value through `--config` is not an explicit
override; name the option on this command when you intend to change the export's setting.

Do not add `--hoist-releases` to this load command. A saved plan already contains that
post-pass. The loader now avoids applying it twice unless the flag is explicitly requested.

Structural co-grasp coupling is automatic and always enabled; `--fuse-cograsp` and
`--pre-insertion 0.05` are also current defaults. For a paired chain, all assembly participants
first meet at one shared funnel-mouth keyframe when there is no support switch. If the target is
placed between `add_grasp` and `assemble`, the pre-switch holders instead share the true place
keyframe; released holders may then retract. These are endpoint equalities only: the regular
motion planner may detour between them. `--fuse-cograsp` is the additional same-kinematic-mode
witness re-solve. The current robust Car-mini evaluation fused 4/4 pairs and reached all 44
modes; its closed-chain drift diagnostic is reported in the table above.

### 2a. Replay the saved motion path without planning or physics

```bash
python3 mp_viser_saved.py \
    experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json \
    experiments/mp-car-mini-3r-asymmetric-seed11023-coupled.json \
    8080 \
    --mp-export experiments/car-mini-3r-asymmetric-seed11023-final-export.json
```

Open `http://localhost:8080`. This read-only viewer checks the complete ordered mode chain and
uses each saved transition configuration to reconstruct attachments. The extra `--mp-export` is
needed only because this particular path predates embedded final exports; newly written MP paths
need only the original three arguments. The saved coupled path can
print a task-signature warning because it predates the deterministic holder-order fix; its old
signature was reproduced exactly and differs only in equivalent robot-list ordering. A new
step-2 export has the stable signature and does not warn. Do not waive a signature mismatch for
physics execution: rerun step 2 before `sim.py run`.

### 3. Run the requested MuJoCo validation

```bash
python3 sim.py run \
    experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json \
    --v2 \
    --mp-path experiments/mp-car-mini-3r-asymmetric-seed11023-coupled.json \
    --actuated --corridor --corridor-rate-clamp \
    --coll-scale 0.8 --spline --limit-rate \
    --record experiments/car-mini-3r-asymmetric-seed11023-default-float-actuated.npz
```

For a fresh physics validation, require the final line to report
`21/21 actions ok; plan SUCCEEDED`. `--coll-scale 0.8` is a debug override that shrinks arm
collision primitives. The resulting simulation clearance does not exactly match the planner
certificate.

### 4. Replay the recording

```bash
python3 sim.py replay \
    experiments/car-mini-3r-asymmetric-seed11023-default-float-actuated.npz
```

## Car-mini-specific pitfalls

- `car-mini` has five parts. Do not use the old `--parts 12` value. The CLI clamps it to five,
  but `--parts 5` records the actual intent.
- Use `--num-robots 3` for this replay. The old command requested four robots.
- Do not use the old `--part-float 0.02` workaround. The current 0.002 m default is part of the
  validated plan. The 20 mm value produced table contacts and broken grasps in MuJoCo.
- `--scatter-spacing` has no effect with `--random-init`.
- Remove duplicate flags. In particular, the last repeated `--seed` silently wins. The CLI now
  warns about duplicates.
- `--viser`, `--mp-viser`, `--show-init`, and `--mp-collision-view` open blocking viewers. They
  are not part of a headless production run.
- Each wheel is one native MuJoCo cylinder. Its collision and visual geometry are now the same,
  and a regression test checks that the settled wheel rim is on the table.

Run the focused regression checks with:

```bash
python3 -m pytest -q \
    tests/test_config.py \
    tests/test_cograsp.py \
    tests/test_pre_skill.py \
    tests/test_sim_outcomes.py::test_car_mini_wheel_rims_rest_on_sim_table
```

## Cube-8 with four vacuum grippers

The 2026-08-14 run used wrench checking, the asymmetric grasp model, collapsed-h search, and
the `press` signature component. These options are current defaults, but they are written
explicitly here so a future default change cannot silently alter the experiment:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py --solve \
    --assembly cube --parts 8 --num-robots 4 --vacuum --weight inf \
    --wrench --grasp-model asymmetric \
    --sig-collapsed-h --sig-components press \
    --seed 11023 --check-init --quiet-rai --no-view \
    --save experiments/cube8-4r-vacuum-seed11023-plan.json
```

Search completed with 40 actions. It found 89,312 co-reachable signature states and persisted
33 witnesses. Generate the final zero-IK export first; it is saved before BiRRT starts, even
when the strict planning attempt times out:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config experiments/cube8-4r-vacuum-seed11023-plan.json \
    --load-plan experiments/cube8-4r-vacuum-seed11023-plan.json \
    --motion-plan --mp-runtime 60 --mp-seed 11023 --mp-shortcut-iters 100 \
    --mp-cograsp --quiet-rai --no-view \
    --mp-export-save experiments/cube8-4r-vacuum-seed11023-final-export.json \
    --mp-save experiments/mp-cube8-4r-vacuum-seed11023-final.json
```

The strict 1e-4 collision-tolerance attempt stopped at 36/48 modes. The terminal path used for
the simulation was generated from the saved export with an explicit 0.01 m tolerance and no
shortcut expansion:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config experiments/cube8-4r-vacuum-seed11023-plan.json \
    --load-plan experiments/cube8-4r-vacuum-seed11023-plan.json \
    --motion-plan \
    --mp-export-load experiments/cube8-4r-vacuum-seed11023-final-export.json \
    --mp-runtime 120 --mp-seed 11023 --no-mp-optimize --mp-no-shortcut \
    --mp-collision-tolerance 0.01 --mp-cograsp --quiet-rai --no-view \
    --mp-save experiments/mp-cube8-4r-vacuum-seed11023-raw.json
```

This produced a terminal 167-state path. The co-grasp audit reports one non-rigid span after
action 16 (`a2+a4`, 3089 mm / 178.9 degrees of relative drift). It is a handover/place-created
co-grasp rather than an `add_grasp` chain, so the current add-grasp/assemble coupling does not
cover it. The path is not a valid production artifact until both this rigidity defect and its
strict base-path collisions are removed.

The old relaxed rollout was recorded with:

```bash
PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 sim.py run \
    experiments/cube8-4r-vacuum-seed11023-plan.json \
    --v2 --mp-path experiments/mp-cube8-4r-vacuum-seed11023-raw.json \
    --actuated --corridor --corridor-rate-clamp --corridor-refine 0 \
    --coll-scale 0.8 --spline --limit-rate --mp-cograsp --no-path-preflight \
    --record experiments/cube8-4r-vacuum-seed11023-actuated.npz
```

This is retained as a historical/debug artifact, not a current validation result.
`--no-path-preflight` is an explicit unsafe bypass and `--coll-scale 0.8` changes the robot
geometry. Verification saw eight accepted base-path violations, including up to 2.04 mm of
`a2_ur_coll8`/table penetration. The rollout happened to complete all 40/40 actions with a
worst final part error of 0.6 mm / 0.1 degrees. Replay its 33,120 recorded frames with:

```bash
python3 sim.py replay experiments/cube8-4r-vacuum-seed11023-actuated.npz
```

Current simulation replay defaults to a strict base-path preflight before spline/corridor
construction. It samples every edge at 0.01 rad, enforces joint limits, and uses a collision
tolerance of 1e-4 m even when the saved path was planned with a looser value. The compacted
shortcut path preserves its 405.23-rad geometry while reducing 757 per-stretch serialized
samples to 246 geometric waypoints. It is therefore the path to use once it passes validation:

```bash
python3 sim.py run \
    experiments/cube8-4r-vacuum-seed11023-plan.json \
    --v2 --mp-path experiments/mp-cube8-4r-vacuum-seed11023-final.json \
    --actuated --corridor --corridor-rate-clamp --spline --limit-rate --mp-cograsp
```

The current Cube shortcut artifact is correctly rejected in about five seconds, before spline
work: stretch 19, task mode `[19, 21, 27]`, has 0.117 mm of
`a2_ur_coll8`/`a2_ur_coll5` penetration. The raw path is rejected even earlier at stretch 1
with 0.173 mm of `a3_ur_coll8`/`a3_ur_coll5` penetration. Regenerate a strict path rather than
disabling the audit or tuning the corridor. `--path-preflight-tolerance` and
`--path-preflight-resolution` exist for explicit diagnostics; `--no-path-preflight` is for
historical ablations only.

## Cross and Chair coupled-keyframe evaluation

These are keyframe + motion-planning runs only; no MuJoCo simulation was run. `srp-cross.json`
and `chair8-r9-plan.json` were selected because they are the only saved Cross/Chair plans in
this checkout containing `add_grasp` actions.

Cross (7 parts, 4 robots):

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config srp-cross.json --load-plan srp-cross.json \
    --motion-plan --fuse-cograsp --fuse-pre-skill --pre-insertion 0.05 \
    --mp-sequence --mp-optimize --mp-runtime 60 --mp-seed 11023 \
    --mp-shortcut-iters 100 --mp-collision-tolerance 1e-4 \
    --mp-robot-margin 0.1 --mp-margin-ramp 0.1 --mp-cograsp \
    --quiet-rai --no-view \
    --mp-export-save experiments/cross7-final-export.json \
    --mp-save experiments/mp-cross7-coupled.json
```

This plan contains `add_grasp -> place(target) -> assemble -> release -> release`. The place
changes the kinematic parent from a gripper to the table, so the two-phase witness fuser reports
`0/1` (expected). The first attempted shared goal incorrectly took former holder `a1` from the
post-place assemble witness, where it is at home: mode `[27, 27, 32, 27]` consequently left
`obj_0` 922 mm from its table pose. That 193-state path is invalid and has been renamed to
`experiments/mp-cross7-coupled.invalid-air-mode27.json` so it cannot be mistaken for the current
output.

The corrected exporter uses `a1`'s real place slice, keeps persistent brace `a4` on the assembly
funnel branch, and lets inserter `a2` approach only after `a1` releases. The target pose error is
0.004 mm and the exact exported-keyframe audit reports 0/33 colliding goals. A corrected
60-second BiRRT run reached 29 modes (deepest task IDs `[32, 28, 32, 28]`) but did not finish, so
there is currently no valid Cross MP path to replay. Inspect the corrected exported keyframes
without running the planner using:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config srp-cross.json --load-plan srp-cross.json \
    --motion-plan --mp-export-load experiments/cross7-final-export.json \
    --mp-modes --mp-runtime 0 \
    --quiet-rai --no-view --viser-port 8081
```

Retry motion planning from those precomputed keyframes, without resolving witnesses, using:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config srp-cross.json --load-plan srp-cross.json \
    --motion-plan --mp-export-load experiments/cross7-final-export.json \
    --mp-sequence --no-mp-optimize --mp-runtime 300 --mp-seed 11023 \
    --mp-shortcut-iters 100 --mp-collision-tolerance 1e-4 \
    --mp-robot-margin 0.1 --mp-margin-ramp 0.1 --mp-cograsp \
    --quiet-rai --no-view --mp-save experiments/mp-cross7-final.json
```

`--no-mp-optimize` makes this stop at the first complete path; the certified shortcut pass still
runs afterward. A 2026-08-14 attempt was stopped after it again remained at 29/33 modes with
deepest task IDs `[32, 28, 32, 28]`. That is not a proof of infeasibility, but it localizes the
failure to the final transit search rather than witness IK or the corrected coupled keyframes.

Planner termination now prints a structured collision-predicate audit. `[mp/margin]` separates
mr_bench's base collision rejections from our robot/table-margin wrapper and names every wrapper
gate that fired; `[mp/base]` samples concrete penetrating pairs; `[mp/edge]` reports the dominant
first collision on failed edges at the frontier. On Cross, the wrapper rejected only 1/50,556
base-valid frontier checks in a 10-second run. The dominant failed-edge pair was the just-released
`a1` finger against placed `obj_5` (291/1,403 failed frontier edges); the direct transition also
puts carried `obj_6` 39.4 mm into an `a4` link. The wall is therefore the coordinated `a1`
retraction plus `a2` approach around bracing `a4`, not `--mp-robot-margin`. Set
`MP_MARGIN_DEBUG=1` only when an individual line for every wrapper veto is needed; the default
aggregate is substantially more readable.

Chair (8 parts, 4 robots):

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config chair8-r9-plan.json --load-plan chair8-r9-plan.json \
    --motion-plan --fuse-cograsp --fuse-pre-skill --pre-insertion 0.05 \
    --mp-sequence --mp-optimize --mp-runtime 60 --mp-seed 11023 \
    --mp-shortcut-iters 100 --mp-collision-tolerance 1e-4 \
    --mp-robot-margin 0.1 --mp-margin-ramp 0.1 --mp-cograsp \
    --quiet-rai --no-view \
    --mp-export-save experiments/chair8-r9-final-export.json \
    --mp-save experiments/mp-chair8-r9-coupled.json
```

Chair built its 49-task keyframe problem, but the 60-second BiRRT run did not find a complete
path (41 modes reached; deepest task IDs `[43, 40, 40, 48]`), so no MP file was written. Its two
`add_grasp` actions are after the final assembly and support the final placement; this saved plan
therefore has no eligible add-grasp/assemble pair and does not exercise the new chain coupling.

Inspect Chair's saved action keyframes in the scrubbable replay viewer:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 replay.py \
    chair8-r9-plan.json --viser --viser-port 8082
```

Or inspect the exact 49 exported mr_bench modes (including pre-insertion keyframes and task
validity checks) without planning:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config chair8-r9-plan.json --load-plan chair8-r9-plan.json \
    --motion-plan --mp-export-load experiments/chair8-r9-final-export.json \
    --mp-modes --mp-runtime 0 \
    --quiet-rai --no-view --viser-port 8083
```

This is a true replay: it reports `[mp-export-reuse] ... ALL witness IK/fusion solves SKIPPED`,
loads the exact persisted 49-task manifest, and refuses a plan/joint/signature mismatch. A bare
`--motion-plan --mp-modes` is the one-time export/debug route and may solve missing final
keyframes; `--mp-export-load` is the zero-solve viewer route.

The Chair ending has three robots co-grasping `obj_0`; after `a2` releases, `a1` must remain
the kinematic holder for the saved `place a1` action. This re-root selection is now canonical
by part and robot-roster order. Older source revisions leaked `frozenset` iteration order here
and could fail at that place action under an unlucky ambient `PYTHONHASHSEED`. The explicit
environment prefix above remains recommended for reproducibility of the rest of the pipeline.

Flags used in both evaluations:

- `--config FILE` restores the saved scene and run configuration; `--load-plan FILE` reuses its
  discrete actions and witnesses without searching again.
- `--motion-plan` builds and runs the mr_bench problem. `--mp-sequence` fixes the saved action
  order; `--mp-optimize` keeps improving RRT* cost until `--mp-runtime 60` expires.
- `--fuse-cograsp` runs the extra same-mode two-keyframe co-grasp IK pass.
  `--fuse-pre-skill` pins idle arms while re-solving funnel keyframes, and
  `--pre-insertion 0.05` creates a 5 cm funnel-mouth keyframe before each mate.
- `--mp-seed 11023` seeds planner and shortcut draws. The wall-clock cutoff still makes this
  approximately, not bit-exactly, reproducible; use `--mp-fake-clock` when bit-exact planner
  draws matter.
- `--mp-shortcut-iters 100` caps the non-time-bounded shortcut pass.
  `--mp-collision-tolerance 1e-4` allows at most 0.1 mm total penetration at validated states.
- `--mp-robot-margin 0.1 --mp-margin-ramp 0.1` requests 10 cm inter-robot clearance in transit,
  relaxed over the final 10 cm for robots participating in the same task.
- `--mp-cograsp` reports path-level relative gripper drift (with `--mp-sequence`).
  `--mp-save FILE` writes a path only when planning succeeds.
- `--mp-export-save FILE` writes the final task/keyframe artifact whether planning succeeds or
  not. `--mp-export-load FILE` reconstructs from it with zero IK/fusion solves. Current
  `--mp-save` files embed this artifact automatically; `--mp-export` on `mp_viser_saved.py` is
  only for older path files.
- `--quiet-rai --no-view` suppresses known RAI collision-library noise and prevents blocking
  viewers. The three environment assignments fix hash order and put caches in writable paths.
