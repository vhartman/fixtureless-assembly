# Pipeline instructions — solve → validate → motion-plan → simulate → look at it

Everything below is deterministic under `PYTHONHASHSEED=0` (use it for anything you intend
to compare). Never run two heavy solves concurrently. Always `--no-view` in scripted runs.

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
  `from config import RunConfig; import main; main.run(RunConfig(assembly="cross", parts=5, solve=True, view=False))`.
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
- **Load it in sim without re-planning or IK**: `sim.py run plan.json --v2 --mp-path mp.json`.
  Both files must be from the same run. A path without an embedded final export is refused;
  `--refresh-witnesses` is the explicit legacy escape hatch, not a replay default.
- **View a saved motion path without re-planning or physics**: `python3 mp_viser_saved.py
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

# 2a. execute in physics -- v1 floating-gripper keyframes (validates the STATICS)
PYTHONHASHSEED=0 python3 sim.py run plan.json --record out.npz
# 2b. execute with REAL ARMS following an mr_bench motion plan (rebuilds MP for the saved
#     discrete plan; needs a plan saved from a --real-ik / non-floating run)
PYTHONHASHSEED=0 python3 sim.py run plan.json --v2 --mp-tolerance 1e-6 --record out.npz
# then look at it:
python3 sim.py replay out.npz [speed]
```
`sim.py run` reuses `replay.build_scene` (the exact same reconstruction the rai replay uses),
so there is a single source of truth for the scene and zero solver duplication. `--gui` for a
live window; `--meshes` is the PARKED experimental link-mesh collision (convex-hull approx is
wrong). Note: milestone/geometric/wrench plans all work — the state trajectory is re-derived
from the saved actions with the saved successor flags.

Prefer a throwaway python driver instead? The two entry points are `SimExecutor(asm,
gripper_kinds).execute(plan, states, record=)` (v1) and `execute_motion_plan(C, asm, plan,
states, problem, mp_path, ...)` (v2); snippets below.

**v1 — keyframe executor** (floating gripper pucks; validates the STATICS claims):
```python
import assembly as A, wrench as W
from env import make_four_robot_env
from sim import SimExecutor
A.set_seed(0); A.set_rolls_per_face(1)
W.PARAMS["grasp_model"] = "asymmetric"; W.PARAMS["insert_force"]["_default"] = 10.0
asm = A.make_assembly("cross", 7)
C = make_four_robot_env(num_robots=3, floating_ee=True, vacuum=True)
A.add_disassembled_obj(C, asm, spacing=0.2)
robots = A.get_robots(C)
plan = A.solve_assembly_problem_milestones(C, asm, robots=robots, weight=5.0, wrench=True, verbose=False)
# milestone plans are plain lists -> re-derive the state trajectory:
gk = A.gripper_kind(C); _, gt = A._search_context(C, asm, robots)
states = [A.get_initial_state(C, asm)]
for act in plan:
    states.append(next(ns for a, ns in A.successors(states[-1], asm, robots, gt, gripper_kind=gk) if a == act))
rep = SimExecutor(asm, gripper_kinds={r: "vacuum" for r in robots}).execute(
    plan, states, record="sim_recordings/myrun.npz")
print(rep.summary())      # per-action taxonomy: dropped/slid/toppled/grasp_broke/not_seated
```
(Exact-search plans carry `plan.states` directly — skip the re-derivation.)

**v2 — motion-plan executor** (FK-synced arms follow the BiRRT path; SEQUENCE only):
```python
from solver import RobotIK, optimize_witnesses
import mp_export as MP
from sim import execute_motion_plan
rik = RobotIK(C, robots); rik.pre_insertion_delta = 0.05
rik.strip_inactive_collision = False; rik.penetration_reject = 1e-4; rik.col_weight = 1e2
plan = A.solve_assembly_problem(C, asm, robots=robots, weight=3.0, robot_ik=rik, verbose=False)
wit, _ = optimize_witnesses(C, asm, plan, robots, rik, cost="ee", sweeps=0, robot_collisions=True, verbose=False)
prob = MP.make_problem(C, asm, plan, wit, robots, rik, None, bracing=False, depgraph=False,
                       collision_tolerance=1e-6)
path, _ = MP.run_birrt(prob, runtime=60.0, verbose=False)
rep = execute_motion_plan(C, asm, plan, plan.states, prob, path,
                          gripper_kinds={r: "vacuum" for r in robots}, record="out.npz")
```
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
Current recordings: `cross7_wrench_sim.npz` (whole cross, 25/25), `wrenchON/OFF_cross5.npz`
(the discriminator pair), `mp_cross3_strict.npz` (BiRRT path + arms, 9/9 at 1e-6).
Live view instead: `SimExecutor(..., gui=True)` / `execute_motion_plan(..., ...)` with
`gui` via SimScene. Recordings carry per-robot grasp wrenches: `np.load(f)["forces"]`.

## 6. Tests

```bash
PYTHONHASHSEED=0 python3 -m pytest tests/ -q     # 43 tests, ~70s
```
