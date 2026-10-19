# Pipeline instructions — solve → motion-plan → simulate → replay

Use `PYTHONHASHSEED=0` for comparable task searches. Motion planning is wall-clock bounded and
is not fully deterministic, so retain every successful path used for a reported result. Never
run two heavy solves concurrently. Always use `--no-view` in scripted runs.

For the shortest maintained end-to-end example, use the verified four-arm Car Mini commands in
the [root README](../README.md#verified-car-mini-pipeline). The sections below are a reference
for individual stages and less common options.

## 1. Solve a plan (CLI)

```bash
# geometric baseline (floating grippers, topological)
PYTHONHASHSEED=0 python3 main.py --assembly cross --parts 5 --num-robots 2 \
    --weight 3 --solve --no-view

# the paper stack: wrench + milestones
PYTHONHASHSEED=0 python3 main.py --assembly chair --parts 8 --num-robots 3 \
    --milestones --wrench --grasp-model asymmetric --solve --no-view

# real arms + the ZERO-TOLERANCE recipe (clean witnesses; see §4)
PYTHONHASHSEED=0 python3 main.py --assembly cross --parts 3 --num-robots 2 \
    --robot ur10 --vacuum --real-ik --pre-insertion 0.05 \
    --strict-collisions --witness-robot-collisions \
    --weight 3 --solve --no-view
```

Useful around a solve:
- `--check-init` — validate the initial scene (collision-free + every part graspable) before solving.
- `--viz-grasps` / `--viz-rest-poses` / `--mate-sequence-viz` (+ `--viser`) — catalogs, no search.
- `--save plan.json` → `python3 replay.py plan.json --viser` — replay a saved plan from its
  persisted witnesses. Missing real-arm witnesses are fatal unless `--resolve-ik` is explicitly
  requested; opening a viewer never silently runs IK.
- `--config file.json` — RunConfig file as base config, CLI flags override. Library use:
  `from assembly_planner.config import RunConfig; import main;`
  `main.run(RunConfig(assembly="cross", parts=5, solve=True, view=False))`.
- Solver menu: default exact wA\* (`--weight`), `--milestones`, `--lazy`, `--ladder[-lns]`,
  `--bounded`, `--anytime` — see `--help` (grouped).

## 2. Motion planning (mr_bench export)

```bash
PYTHONHASHSEED=0 python3 main.py --assembly cross --parts 3 --num-robots 2 --robot ur10 \
    --vacuum --real-ik --pre-insertion 0.05 --strict-collisions --witness-robot-collisions \
    --weight 3 --solve --motion-plan --mp-collision-tolerance 1e-6 \
    --save plan.json --mp-export-save mp-export.json --mp-save mp.json --no-view
```
- `--mp-sequence` = fixed total order (the current default); `--no-mp-sequence` exports the
  depgraph partial order.
- `--mp-viser` / `--mp-modes` serve the path / per-mode keyframes (BLOCKING viewers).
- **Persist the export boundary**: `--mp-export-save mp-export.json` writes the FINAL
  post-fusion skill witnesses, every pre-skill keyframe, exact task goals/effects/order,
  settings and signature before planning begins. It is written even when no path is found.
  `--mp-save mp.json` embeds the same block in every newly saved path.
- **Load it in sim without re-planning or IK**: `sim.py run plan.json --mp-path mp.json`.
  Both files must be from the same run. A path without an embedded final export is refused;
  `--refresh-witnesses` is the explicit legacy escape hatch, not a replay default.
- **View a saved motion path without re-planning or physics**:
  `python3 tools/visualization/mp_viser_saved.py
  plan.json mp.json 8080`, then open `http://localhost:8080`. The viewer checks the ordered
  mode chain and reconstructs attachments from the saved transition configurations. For an
  older path, first make one standalone final export and add `--mp-export mp-export.json`.
- **View exported modes with zero solves, even if planning failed**:
  `python3 main.py --config plan.json --load-plan plan.json --motion-plan
  --mp-export-load mp-export.json --mp-modes --no-view`. A bare `--mp-modes` is an exporter
  invocation and may solve; `--mp-export-load` is the replay invocation and cannot.
- `--mp-collision-tolerance` = total penetration budget mr_bench accepts at validated configs
  (current default 1e-4; 0.01 was the historical slack, while 1e-6 is the zero-tolerance target
  and needs §4's clean witnesses).
- With `--fuse-cograsp` (default), an `add_grasp` used by the following assembly is one atomic
  keyframe chain. Without a support switch, all participants meet at the pre-insertion keyframe.
  With an intervening place, pre-switch holders meet at the true place pose, released holders
  may retract, and only persistent holders stay fixed while the inserter approaches and seats.
  Assembly participants stay pinned through consecutive releases. The task-level construction is
  always enabled; `--fuse-cograsp` controls the additional two-keyframe witness re-solve, which
  applies only when both endpoints share one kinematic mode. This joint-goal construction is
  required because the current BiRRT does not project samples onto a closed-chain relative-pose
  constraint manifold. Use `--mp-cograsp --mp-sequence` to report path-level gripper drift; a
  coupled run should report zero violations. README's Cross/Chair section records the exact
  evaluation commands and every flag used.

## 3. Physics-sim validation (MuJoCo, `sim.py`)

**sim does NOT solve — solve with `main.py --save`, then execute that plan in sim.**

```bash
# 1. solve + save (any solver flags; here real-arm zero-tolerance)
PYTHONHASHSEED=0 python3 main.py --assembly cross --parts 5 --num-robots 2 --robot ur10 \
    --vacuum --real-ik --pre-insertion 0.05 --strict-collisions --witness-robot-collisions \
    --weight 3 --solve --save plan.json --no-view

# 2. execute the persisted real-arm motion plan with the fail-closed reproduction profile
PYTHONHASHSEED=0 python3 sim.py run plan.json \
    --mp-path mp.json --record out.npz
# then look at it:
python3 sim.py replay out.npz [speed]
```
`sim.py run` reuses `replay.build_scene` (the exact same reconstruction the rai replay uses),
so there is a single source of truth for the scene and zero solver duplication. `--gui` for a
live window; `--meshes` is the PARKED experimental link-mesh collision (convex-hull approx is
wrong). Note: milestone/geometric/wrench plans all work — the state trajectory is re-derived
from the saved actions with the saved successor flags.

The production executor is `execute_motion_plan(...)`; the obsolete floating-gripper
keyframe executor and its CLI mode have been removed. `SimScene` still supports mocap-only
fixtures internally for focused contact-model unit tests.

Arm geometry: planner's collision capsules by default (sim contact == planner contract);
`_arm_shape_frames(..., meshes=True)` switches to hulled link meshes — PARKED until the
capsule-vs-mesh coverage question is settled (capsules are not an outer bound; see memory).

**The paper comparison** is the same config solved `wrench=True` vs `wrench=False`, both
executed in sim: wrench-on succeeds, wrench-off fails in the LP-predicted mode.

## 4. The zero-tolerance recipe (what makes 1e-6 export work)

1. `--strict-collisions [1e-4]` — KOMO witnesses rejected above EPS total penetration
   (restarts search clean basins; collision weight 1e1→1e2).
2. `--witness-robot-collisions` — idle arms visible to the gate (else arm-arm hides).
3. `--pre-insertion 0.05` — the funnel; without it BiRRT legally bulldozes seat approaches.
4. `--mp-collision-tolerance 1e-6`.
Known open: multi-arm goals can still conflict via the witness-idle-arm-at-HOME vs
mr_bench-entry-config gap (cross-7 transition sampling; being debugged mr_bench-side).

## 5. Look at things

```bash
python3 sim.py replay sim_recordings/<file>.npz [speed]   # MuJoCo viewer, exact kinematic
                                                          # playback + action marks (BLOCKS)
```
Scrub to a configuration and press `E` (or the **export GLB** button) to save the currently
shown scene beside the recording. The GLB is self-contained, uses metres, and contains the
visible parts, robot meshes/grippers, floor, and table—but not group-2 collision proxies. For
a reproducible headless snapshot at an exact frame:

```bash
python3 sim.py replay sim_recordings/<file>.npz \
  --frame 1234 --export-scene render/frame-1234.glb --export-only
```

Current recordings: `cross7_wrench_sim.npz` (whole cross, 25/25), `wrenchON/OFF_cross5.npz`
(the discriminator pair), `mp_cross3_strict.npz` (BiRRT path + arms, 9/9 at 1e-6).
Use `--gui` for a live view. Recordings carry per-robot grasp wrenches:
`np.load(f)["forces"]`.

## 6. Tests

```bash
PYTHONHASHSEED=0 python3 -m pytest tests/ -q
```
