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

Current-audit note (2026-08-21): the car artifact in this table predates the strict
interpolated joint-limit preflight. It is now rejected at stretch 4 because
`a2_ur_wrist_2_joint=0.286711` exceeds its `0.28319` upper limit by 3.52 mrad. Regenerate its
motion path before using it for simulation/controller evaluation; `--no-path-preflight` only
turns it into an unsafe diagnostic and produced a large actuator tracking error in testing.

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

### Experimental C2 geometric smoothing

`--c2-spline` is a separate, opt-in geometry postprocessor. It replaces the default C1
Catmull--Rom stage for that run; it does not modify phase scheduling, phase-rate smoothing,
controllers, or event execution. Each mode-local free-motion stretch is fitted in composite
joint space by a clamped cubic B-spline. Its two task-witness endpoints are hard equalities,
while ordinary motion-planner waypoints are soft observations. The curve may therefore leave
and shorten the saved polyline instead of merely retiming it.

Safety is fail-closed. The emitted samples and the linear interpolation between every adjacent
sample are checked at `--c2-resolution` (default 0.01 rad maximum per-joint step) with the
mode's complete motion-planner collision predicate and the robot joint limits. Accepted fits
also obey `--c2-max-deviation` (default 0.20 rad in composite configuration norm). The fitter
tries progressively more faithful curves; if none certifies, that entire stretch stays on the
original preflight-certified polyline. Stretches containing a rigid co-hold or handover are
always protected because collision checking alone cannot enforce their closed-chain relative
tool transform. C2 continuity is consequently claimed within accepted free-motion stretches,
not across mode switches or protected/fallback stretches.

The implementation is isolated in `c2_smoothing.py` and passes ordinary sampled waypoints to
the existing postprocessing pipeline. The established version remains the default. A current
car-mini evaluation command is:

```bash
python3 sim.py run experiments/car-mini-ur10-new.json --v2 \
    --mp-path experiments/car-mini-ur-10-new-mp.json \
    --phase-schedule --phase-smooth --phase-batch --c2-spline --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/car-mini-ur10-new-c2-phase-batched.npz

python3 sim.py replay experiments/car-mini-ur10-new-c2-phase-batched.npz
```

Useful experimental controls are `--c2-control-ratio` (fewer controls smooth more),
`--c2-smooth-weight`, `--c2-samples-per-seg`, `--c2-max-deviation`, and
`--c2-resolution`. Defaults should be treated as a conservative starting point; rejection does
not require per-assembly tuning because local fallback is automatic.

### Visual-quality phase scheduling in simulation

A paper-style formulation, pseudocode, certification argument, and current limitations are in
[PHASE_SCHEDULING.md](PHASE_SCHEDULING.md).

Use `--phase-schedule` for the offline coordination-space scheduler. It keeps each robot on its
already planned joint path and chooses only its monotone phase along that path. The objective is
a calm reference—preferred cruise motion, fewer crawl/stop episodes, and ordinary-speed
asynchronous arrival at the final pose—not minimum makespan. Mode order, acting/holding robot
rendezvous, and monotonicity remain hard constraints.

Every proposed phase step is checked with the rebuilt motion planner's complete predicate,
including the saved robot-margin gates and the strict `--path-preflight-tolerance`. The complete
schedule is then checked again at bounded joint-path variation and on both sides of every mode
switch. The default `--phase-audit-resolution 0.01` rad matches the motion planner's edge scale;
`--phase-audit-resolution 0` restores the old check at every 2 ms controller sample. A
coordination deadlock falls back to the synchronized path, which is audited too; an invalid
fallback aborts rather than being simulated. Runtime tick-level collision gating is not the
planner. `--phase-schedule` also defaults physical arm-arm contacts on as a backstop. The
MuJoCo masks use one bit per robot, so cross-robot contacts are enabled without the old bug that
also enabled enormous same-arm self-contact forces.

Phase search runs first on a 10 ms coordination grid, then interpolates back to the 2 ms
controller rate. Acceptance uses the configuration-resolution audit above. A counterexample
automatically retries the phase search at 4 ms and 2 ms; all grids are exact multiples of the
controller tick. Audit samples are selected using accumulated maximum-joint variation, not
endpoint distance, so an out-and-back motion cannot be skipped. First/last samples and every
event pose are unconditional. This avoids rechecking long stationary runs without changing the
resolution contract used by motion planning.

Phase speed is measured in each robot's own joint-space arc, rather than the composite arc of
all robots. A robot therefore moves at the requested visual cruise whenever it is actually
moving, and spans where only another robot moves are skipped as zero-duration bookkeeping.
Every configuration produced by those skips remains subject to the same collision predicate
and final configuration-resolution audit.

`--phase-smooth` applies a second, time-domain postprocessor to the resulting per-robot phase
rates. It does **not** fit another joint-space spline: that would cut geometric corners and
invalidate the motion path. Each robot's shared coordination phase is first converted to that
robot's intrinsic accumulated joint-space arc. This is essential when a robot is stationary
over a long span of the composite path: smoothing the composite coordinate would erase the
scheduler's per-arm speed normalization and can amplify physical acceleration dramatically.
The intrinsic non-negative increments are Gaussian-smoothed over `--phase-smooth-time` (default
0.12 s), then each fire-to-fire interval receives a least-change non-negative projection onto
its exact displacement. The complete shared phase vector remains pinned at every mode switch;
actor stops, intentional sharp-corner slowdowns, and rigid co-hold/collision-coupling intervals
remain hard constraints. A per-robot guard rejects any interval that would exceed that robot's
certified source peak rate. The complete result is then
audited at 0.01 rad joint-path resolution in both pre- and post-switch modes;
collision-sensitive intervals revert locally
to their original certified samples. Thus smoothing can reduce acceleration and jerk but can
never become a reason to execute an unaudited shortcut. A collision-free proposal is also
rejected if either acceleration or jerk RMS becomes more than 2% worse than its source.
`--phase-smooth` is strict: if no independent phase trace exists, if the quality guard rejects
the proposal, or if collision repair cannot certify it, the run aborts before physics and no
recording is written. To execute a raw or synchronized schedule, omit `--phase-smooth`
explicitly.

`--phase-batch` is the final, geometry-independent cleanup for short move-stop-move patterns.
It runs after `--phase-smooth`, so it works identically with the default Catmull--Rom geometry
and with `--c2-spline`. For each robot it finds adjacent motion islands separated by at most
`--phase-batch-max-gap` (default 0.35 s). If either island contains less than
`--phase-batch-min-motion` (default 0.08 rad joint-space arc), it tries to pack that afterthought
next to the larger motion as one near-cruise burst. It deliberately leaves one clean stationary
period rather than spreading the same displacement into a long crawl.

An arm's own task stop, a co-grasp/handover equality interval, the global event-passage rule,
and the source peak joint rate remain hard constraints. Crossing another robot's mode switch is
allowed only if the candidate is valid in both the pre- and post-switch worlds. Each local
proposal is checked, followed by the complete configuration-resolution audit. A late counterexample restores only
the connected offending merge windows. The log reports total motion islands, small islands,
and internal slow gaps before and after; acceleration/deceleration at the two outer ends of one
continuous motion are not counted as internal gaps. Mandatory task stops are therefore not
misreported as removable visual defects.

`--phase-intent` is a separate experimental alternative to `--phase-batch` for the stronger
visual objective “move decisively, then be clearly idle.” It approximately minimizes the number
of control samples on which any robot is moving, with fixed duration and event times. For each
robot, its own task events partition time into robot-owned intervals. The complete intrinsic
joint-space path arc in each eligible interval is packed into one delayed motion with 150 ms
quintic velocity ramps and a peak path rate no greater than `--phase-intent-rate` (default
1.5 rad/s). Using intrinsic rather than shared composite arc removes bookkeeping stops where only
another robot moves. Several temporal placements are tried.

Only the robot's own events are rest anchors. A foreign mode switch is a passage deadline and a
pre/post-mode collision constraint, so the robot can cross it at nonzero velocity. True rigid
co-hold/handover spans stay byte-exact. Scheduler-created collision couplings are not physical
equalities: intent is allowed to explore them because its candidates are collision-screened and
the complete result receives the 0.01 rad pre/post-switch audit. Monotonicity and every event
witness remain exact.

Pre-skill GOTO tasks are geometric passage waypoints, not rest events: they change no attachment
and execute no skill hook, so their old task timestamp is not a hard phase deadline. However, an
own-event interval containing such a waypoint is a contact-skill funnel: the already continuous
phase-smoothed timing is retained there because it also encodes coordinated holder timing and the
terminal low-rate approach expected by the seat servo. `--phase-intent` therefore compacts free
transport and does not independently repack insertion/grasp funnels. For each eligible interval, several
continuous durations are considered. A candidate may not increase either total all-robot idle or
the squared lengths of its idle runs; among the survivors the shortest motion is placed latest.
A second coordinate-refinement pass shifts earlier robots later after the other robots' compact
motions are known. Thus maximizing individual idle cannot create a longer scene in which nothing
moves.

This objective is intentionally different from minimum makespan and minimum jerk. In practice,
the ramped bursts can improve derivatives too, but that is measured rather than assumed.
`--phase-intent` is fail-closed: if no certified motion-support reduction survives, simulation
does not start. It is mutually exclusive with `--phase-batch`, but normally follows
`--phase-smooth`. Compact timing explores farther from the synchronized diagonal, so a narrow
coordination corridor naturally rejects many placements and limits the achievable idle gain.

Current saved-path validation commands (no IK or motion replanning) are:

```bash
python3 sim.py run experiments/car-mini-ur10-new.json --v2 \
    --mp-path experiments/car-mini-ur-10-new-mp.json \
    --phase-schedule --phase-smooth --phase-intent --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/car-mini-ur10-new-phase-intent-rad-audit.npz

python3 sim.py run experiments/chair-no-vacuum-assemble.json --v2 \
    --mp-path experiments/mp-chair-no-vacuum.json \
    --phase-schedule --phase-smooth --phase-intent --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/chair-phase-intent-rad-audit.npz
```

The car completed 19/19 actions in 4,423 replay frames; the chair completed 34/34 in
9,172 frames. The car audit retained its known 10 ms-grid collision counterexample and accepted
the 4 ms retry. Final-audit collision calls fell from 34,840 to 3,798 on car and from 72,635 to
9,271 on chair. Use `python3 sim.py replay FILE.npz` to inspect either recording.

`--phase-servo-time SEC` is the fixed nominal pre-fire reservation used while forming the
offline phase schedule (default 0.15 s per settle or seat operation, matching the measured
car-mini mean rather than its 1 s worst-case cap). It is a guess made before execution, never
an EMA or per-tick phase correction. At the event, convergence is checked only for its acting
robots and holders; unrelated robots are intentionally moving and their tracking residual must
not create a global stop. The old offline velocity-funnel/wrench loop is not applied underneath
a phase schedule: it used the minimum factor as a second global clock and froze every robot
when one placed part touched the table. The actor still gets its bounded compliant offset,
event-owned settle/seat hooks, physical arm contacts, and the grasp breakaway monitor; the
other independently scheduled arms keep flowing. A genuine event-servo overrun can still hold
at that event, but no longer turns every contact tick into another global timeout.

The actuated settle gate now defaults to `--settle-tol 0.002`. Car-mini's loaded a3 servo has
a repeatable approximately 1.9 mrad steady-state residual; the former 1 mrad threshold waited
the complete `--settle-max 1.0` s without reducing it. Two mrad is about 2 mm at a one-metre
reach and matches the terminal seat-servo scale, while remaining far below the transient errors
that the gate is intended to keep out of a newly activated weld. More generally, a residual
below twice the requested tolerance that improves by less than one quarter of that tolerance
over 0.1 s is recognized as the finite-gain servo's steady-state plateau; large or still-moving
errors retain the normal gate and one-second safety bound.

Current car-mini validation (saved plan/path only; no IK or motion replanning):

```bash
python3 sim.py run experiments/car-mini-ur10-new.json --v2 \
    --mp-path experiments/car-mini-ur-10-new-mp.json \
    --phase-schedule --phase-smooth --phase-batch --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/car-mini-ur10-new-phase-batched.npz

python3 sim.py replay experiments/car-mini-ur10-new-phase-batched.npz
```

This completed 19/19 actions. With the established C1 geometry, batching kept 52 certified
merges and reduced motion islands from 207 to 155, sub-0.08-rad islands from 127 to 73, and
internal slow gaps from 700 to 620. The
phase reference has 37,976 control samples plus 34
post-switch poses, all accepted by the exact final audit; recording every eighth physics tick
produced 4,778 NPZ frames. The duration is printed only for context. Motion-quality reporting
measures each robot from its first motion through its own final arrival and reports the
harmless terminal idle tail
separately, so an early-finishing arm is not encouraged to creep merely to end in synchrony.
Car-mini's tight coordination and rigid-coupling windows leave relatively little freedom for
postprocessing: reference acceleration/jerk RMS fell by 1.5% and the recorded physical values
by approximately 0.8%, with the physical peak joint-space rate unchanged.

With `--c2-spline`, the current run kept 26/28 proposed merges after two local audit
restorations: motion islands fell 144 to 118, sub-0.08-rad islands 76 to 49, and internal slow
gaps 373 to 332. It also completed 19/19 actions. Use the earlier C2 command to reproduce or
replay `experiments/car-mini-ur10-new-c2-phase-batched.npz`.

Current peg-stool chamfer evaluation uses the supplied saved plan and path directly. Both the
established C1 geometry and opt-in C2 geometry now produce strict smoothed rollouts:

```bash
# Established C1 geometry
python3 sim.py run experiments/peg-stool-4r-new-plan.json --v2 \
    --mp-path experiments/mp-peg-stool-chamfer.json \
    --phase-schedule --phase-smooth --phase-batch --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/peg-stool-4r-new-phase-smoothed.npz

# C2 geometry: strict, collision-audited intrinsic-arc smoothing
python3 sim.py run experiments/peg-stool-4r-new-plan.json --v2 \
    --mp-path experiments/mp-peg-stool-chamfer.json \
    --phase-schedule --phase-smooth --phase-batch --c2-spline --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/peg-stool-4r-new-c2-phase-smoothed.npz

python3 sim.py replay experiments/peg-stool-4r-new-phase-smoothed.npz
python3 sim.py replay experiments/peg-stool-4r-new-c2-phase-smoothed.npz
```

For C1, the 45,285-sample asynchronous reference and smoothed result pass the exact 2 ms
collision audit. Intrinsic-arc smoothing reduces joint acceleration RMS by 26.7% and jerk RMS
by 26.8%. Batching keeps 23/23 certified merges: motion islands fall 160 to 137,
sub-0.08-rad islands 78 to 53, and internal slow gaps 459 to 435. Simulation completes 21/21
actions and records 5,735 frames.

An experimental intentional-motion rollout replaces `--phase-batch` with `--phase-intent`:

```bash
python3 sim.py run experiments/peg-stool-4r-new-plan.json --v2 \
    --mp-path experiments/mp-peg-stool-chamfer.json \
    --phase-schedule --phase-smooth --phase-intent --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/peg-stool-4r-new-phase-intent-v6.npz

python3 sim.py replay experiments/peg-stool-4r-new-phase-intent-v6.npz
```

At the default 1.5 rad/s intent rate, moving support falls from 67,590 to 62,701 robot-samples
(idle fraction 59.1% to 62.1%), motion islands fall 171 to 121, and internal slow gaps fall 512
to 363. Conditional moving speed rises from 0.68 to 0.73 rad/s. All-robot idle falls from 9.9 s
to 7.8 s and its longest interval from 1.4 s to 1.2 s. The 150 ms quintic ramps reduce
acceleration RMS by another 16.0% and jerk RMS by 16.7% relative to the already smoothed input.
The actuated recording peaks around 1.72 rad/s instead of the former approximately 2.4 rad/s.
All 13 accepted robot-owned intervals and the complete trajectory pass the exact audit; four
intervals retain certified smoothed timing after no alternative can be certified, 19 intersect a
true rigid span, and the coordinate-refinement pass shifts three intervals later. The wider
foreign-stop diagnostic falls from 11 to 9. Simulation completes 21/21 actions and records 5,281
frames.

For C2, 14 stretches fit, 15 closed-chain stretches remain protected, and 3 unsafe fits retain
their certified polylines. Its 45,575-sample asynchronous reference and smoothed result also
pass the exact audit. Smoothing reduces acceleration RMS by 20.4% and jerk RMS by 20.5%.
Batching keeps 17/17 certified merges, reducing motion islands 142 to 125, sub-0.08-rad islands
63 to 44, and internal slow gaps 282 to 259. Simulation completes 21/21 actions and records
5,772 frames.

The two earlier runs have been retained only for diagnosing the visible stop-and-go behavior and
renamed so they cannot be mistaken for smoothed results:

```bash
python3 sim.py replay experiments/peg-stool-4r-new-unsmoothed-diagnostic.npz
python3 sim.py replay experiments/peg-stool-4r-new-c2-unsmoothed-diagnostic.npz
```

Those diagnostic recordings predate the strict contract and the intrinsic-arc fix. The C1 file
used the synchronized fallback. The C2 file smoothed the composite arc, failed the quality guard,
then applied short-motion batching to the raw schedule. Neither is phase-smoothed.

Chair validates the same defaults on a longer mixed-EE UR5 plan (the historical filenames say
"no-vacuum", but the saved config contains two vacuum and two two-finger grippers):

```bash
python3 sim.py run experiments/chair-no-vacuum-assemble.json --v2 \
    --mp-path experiments/mp-chair-no-vacuum.json \
    --phase-schedule --phase-smooth --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/chair-no-vacuum-phase-smooth.npz

python3 sim.py replay experiments/chair-no-vacuum-phase-smooth.npz
```

This completed 34/34 actions. The first 10 ms phase search succeeded, and all 79,060 expanded
2 ms samples plus 60 post-switch poses passed the exact audit. The recording has 9,932 frames.
Three small finite-gain residuals plateaued at up to 3.3 mrad; plateau detection reduced total
settling from 3.002 s (three full timeouts) to 0.306 s while preserving the successful outcome.
In the previously conspicuous `obj_1` handover-to-table-place interval, the repeated global
force-guard freeze is gone: the longest all-robot stop fell from roughly 9.5 seconds to 0.53
seconds, while the whole interval now takes 7.41 seconds and still completes successfully.
The phase smoother required no collision-driven interval fallback on Chair. It reduced reference
acceleration/jerk RMS by 14.5%/13.4% and recorded physical acceleration/jerk RMS by
13.5%/16.0%, without increasing the recorded peak joint-space rate or changing the placement
pause above.

`--limit-rate` is deliberately ignored by this scheduler: it keeps the visual cruise at
`--joint-rate` (1 rad/s by default) rather than racing the UR joint limits. The existing
`--force-limits` tables are computed and reported but are not yet consumed by the phase
scheduler; the runtime normalized wrench guard and break monitor remain active. This is the
next safety-model integration, not a hidden claim.

The older `--corridor-online --pace` artifacts are retained only for debugging. Their earlier
“19/19” result was not a collision-free validation: arm-arm physics was masked, and the sparse
three-tick preventive gate could advance unchecked from an already colliding state. In the
recorded paced car run, two vacuum robots overlap by roughly 9 cm near frame 3544. Do not use
that command or NPZ as the current baseline.

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
limits OK. That dated recording used the former torque-only meaning of
`--servo-torque-guard 0.7`: it engaged for 167 ticks below the 6 N force trigger. The current
guard is contact-conditioned as documented above, because a total weld wrench also contains
the static gravity reaction and must not freeze transport indefinitely. `--no-force-limits`
disables only the predictive speed retimer: it
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

## Vacuum SS-IK and open witness search

An intuitive and paper-style description of the current search pipeline—including the
signature-collapsed backward pass, fact components, milestone fills and repair, open witness
validation, and `--milestones-anytime`—is in [SEARCH_METHOD.md](SEARCH_METHOD.md).
The proposed paper scope, evaluation matrix, simulation-realism work, and real-robot experiment
plan are collected in [PAPER_PLAN.md](PAPER_PLAN.md).

Analytic IK seeding is on by default with `--real-ik`. It is calibrated separately for the
UR5/UR10 two-finger and vacuum tool frames; vacuum uses the grasp target's canonical aligned EE
roll. `--no-ik-seeding` disables all SS-IK seeding and restores the nonlinear reach precheck.
An empty vacuum solution set is not treated as a hard impossibility: the relaxed/full KOMO path
still decides borderline targets.

`--open-witness` is an experimental, isolated replacement for the permanent-failure behavior of
lazy width-1 milestone fills. The first unresolved edge immediately invalidates the incumbent:
later edges of that plan are not speculatively probed. That edge is suspended for the rest of
the fairness epoch and reopened in the next epoch. Successes are cached; stochastic failures
are not. Ordinary skills escalate through 2, 4, then `--open-witness-batch N` attempts. Assemble
keeps one atomic N=7 structural batch: all six canonical orientation biases plus one unbiased
initialization in a shared scene. Splitting that small bundle across epochs measured worse on
peg-stool because the promising assembly did not reopen before greedy search found many other
plans. `--open-witness-epochs N`
(default 2) limits reopening effort per fill and is not an infeasibility certificate. The flag is
off by default and does not alter the legacy `bind()`/lazy-fill path. A batch shares scene and
reach-precheck setup.

Every fresh `--real-ik --save` now writes the **exact successful search witness** for every IK
action, even with `--optimize-witnesses off`. Open search retains the returned joint vector, not
just a successful boolean, and aligns those vectors to the final action list before writing JSON.
Release hoisting runs first; if it changes an IK predecessor scene, that changed edge is
certified once as part of the fresh solve. An IK action with a null witness makes export fail
immediately.

Witness optimization is layered on top of this immutable feasibility baseline. Changing, for
example, `off` to `floor` or changing the branch budget requests extra candidates but does not
discard the saved search solution. Failed candidate harvesting falls back to the exact baseline.
A normal `--load-plan`, `replay.py`, or sim replay never silently resolves missing IK.
`--refresh-witnesses` is the explicit one-time escape hatch for a legacy plan that predates this
persistence format; new files do not need it.

Example search only (no motion planning):

```bash
python3 main.py --solve \
  --assembly peg-stool-cuboid-4-tight-chamfer --parts 5 --num-robots 4 \
  --robot ur5_limited --grippers vacuum,two_finger,two_finger,vacuum \
  --weight 100 --milestones --sig-collapsed-h --sig-components press \
  --real-ik --open-witness --open-witness-batch 7 --open-witness-epochs 1 \
  --grasp-sampler --grasp-sampler-keep 20 --grasp-sampler-scenes \
  --grasp-sampler-wrap 0.1 --grasp-sampler-pad-support 0.02 \
  --wrench --grasp-model asymmetric --vacuum-no-assemble \
  --vacuum-engage-force 5 --two-finger-scale 10000 --grasp-compliance \
  --insert-force 0.1 --insertion-noise 0 --allow-curved-grasps \
  --scatter-spacing 0.2 --seed 11023 --witness-margin 1e-4 \
  --strict-collisions 1e-4 --optimize-witnesses off --witness-sets 0 \
  --no-fuse-cograsp --quiet-rai --no-view \
  --save experiments/bench-peg-stool-open-skill-tiered-plan.json
```

Historical V1 measurement on this command (before immediate replanning): true batched OPEN
reproduced the legacy
19-action plan in 38.3 s at 0.838 GB peak RSS. The otherwise identical legacy path took
54.0 s at 0.840 GB: search is 29% faster, bind wall falls from 25.2 to 9.1 s, and NLP solves
from 1,123 to 563. The two-part vacuum cube also improves from 9.7 s legacy to 6.6 s open.
As a second control, legacy with its ordinary restart budget reduced to six and reach-precheck
restarts disabled still took 66.0 s (31.9 s bind wall, 1,321 NLP solves) and selected a longer
20-action plan. Thus the measured gain is not merely a larger hidden per-edge effort budget.
The earlier one-bind-per-attempt prototype was much worse because it rebuilt the scene and
reran reach-precheck for every attempt; batching is the performance-critical distinction. V1
also kept validating the remainder of an incumbent after its first failure. That speculative
failure harvesting helped peg-stool but violated ordinary lazy-validation semantics, so the
current version stops immediately and is reported separately below.

To add optimized, replayable witness sets without rerunning search or baseline IK:

```bash
python3 main.py \
  --config experiments/bench-peg-stool-open-persisted-plan.json \
  --load-plan experiments/bench-peg-stool-open-persisted-plan.json \
  --optimize-witnesses floor --witness-sets 4 --witness-branch-budget 5 \
  --save experiments/bench-peg-stool-open-persisted-floor-plan.json \
  --quiet-rai --no-view

python3 replay.py experiments/bench-peg-stool-open-persisted-floor-plan.json --no-view
```

For an older plan without a `witnesses` block, repair it deliberately once with
`--refresh-witnesses --save NEW.json`. Without that flag, post-processing fails closed instead
of gambling on a fresh IK draw.

The corrected vacuum SS-IK itself does help on the hard case: the legacy search above took
54.0 s and 0.84 GB with analytic seeding versus 143.6 s and 1.70 GB under
`--no-ik-seeding`. The trivial two-part vacuum cube went the other way (9.7 s seeded versus
6.6 s unseeded), because collision-limited failures can spend more on exact branches than the
small nonlinear problem costs. The hard-case reduction is large enough to keep the calibrated
vacuum path enabled by default.

### Cross-assembly open-witness comparison

The original full-plan-prevalidation comparison was repeated on the robust Car-mini, Cross, and Chair
configurations on 2026-08-22. Both sides loaded the same saved configuration and disabled only
post-search work (`--optimize-witnesses off --witness-sets 0 --no-fuse-cograsp`); motion
planning was not run. The OPEN side additionally used
`--open-witness --open-witness-batch 7 --open-witness-epochs 2`.

| assembly | legacy search | V1 batched OPEN | change | NLP solves | bind wall | peak RSS | actions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Car-mini robust, 3 UR10 limited | 34.33 s | 26.42 s | **-23.0%** | 1,384 -> 893 | 23.1 -> 15.6 s | 504 -> 488 MB | 26 -> 26 |
| Cross-7 robust, 4 UR10 | 10.38 s | 10.09 s | -2.8% | 320 -> 264 | 2.0 -> 1.4 s | 737 -> 737 MB | 27 -> 28 |
| Chair-8 robust, mixed UR10 | 70.86 s | 61.57 s | **-13.1%** | 2,751 -> 1,340 | 24.7 -> 15.6 s | 1,716 -> 1,700 MB | 33 -> 33 |

These are single paired measurements, not statistical confidence intervals. Cross is already
cheap and dominated by discrete search, so reducing its IK work does not materially improve the
end-to-end time; its greedy (`--weight inf`) OPEN run also chose one extra action. Every real
fill in all three OPEN runs completed in epoch one. These runs therefore validate the batching
speedup but do not exercise real-instance reopening; the fairness/reopening behavior has a unit
regression test.

Car-mini and Cross were subsequently floor-optimized outside the timed region and replayed from
18 and 22 persisted witnesses respectively, with per-action IK solving skipped. Chair's OPEN
and legacy searches both found the same 33-action structure, but independently regenerated
floor witnesses expose the same existing final co-hold problem: at the keyframe before step 32,
`a1_ur_coll7` penetrates `obj_5/obj_6` by 17.65 mm total. Chair's search timing is useful as a
matched performance comparison, but neither regenerated Chair artifact is claimed as a clean
replay. The identical failure on both paths localizes it outside the OPEN policy.

Current immediate-replan results use the same maximum batch but ordinary-skill tiers 2 -> 4 ->
7. Assemble retains its atomic seven-slot bundle. These are again individual runs:

| assembly | legacy | current OPEN | change | NLP solves | actions | replay |
|---|---:|---:|---:|---:|---:|---|
| Car-mini robust | 34.33 s | **23.25 s** | **-32.3%** | 1,384 -> 254 | 26 -> 25 | clean, 17 persisted witnesses |
| Cross-7 robust | 10.38 s | 15.76 s | +51.8% | 320 -> 261 | 27 -> 27 | clean, 21 persisted witnesses |
| Peg-stool chamfer | 54.0 s | **49.12 s** | -9.0% | 1,123 -> 538 | 19 -> 17 | old artifact predates exact persistence |

Car-mini confirms that speculative tail validation can be very expensive. Peg-stool shows its
opposite accidental benefit: one invalid incumbent often contained several independently doomed
edges, and V1 harvested all of those refutations before replanning. Immediate replanning explores
more sibling plans instead. Recovering that pruning without touching an already-invalid plan
requires a better first-edge selector or the deferred persistent pending-edge scheduler. Cross is
small and discrete-search dominated; its current single timing is not evidence of an IK
regression because NLP work still falls, but it is not an end-to-end performance win.

The old 17-action peg-stool artifact demonstrates the persistence bug: search found every edge,
but its plan-only JSON discarded those vectors and a separate floor pass failed to rediscover
step 2. It remains intentionally incomplete and should not be used for motion planning. A fresh
end-to-end run after the fix produced a 20-action plan with all 16 required search witnesses;
floor optimization retained all 16, and a second-process load of the floor artifact performed
zero bind/NLP calls:

- `experiments/bench-peg-stool-open-persisted-plan.json`
- `experiments/bench-peg-stool-open-persisted-floor-plan.json`

Car-mini and Cross earlier optimized artifacts are:

- `experiments/bench-car-mini-open-skill-tiered-witness-plan.json`
- `experiments/bench-cross7-open-skill-tiered-witness-plan.json`

The benchmark form is:

```bash
# Replace CONFIG and TAG with one of:
#   experiments/car-mini-3r-robust-seed11023-plan.json  / car-mini
#   experiments/cross7-4r-robust-seed0-plan.json        / cross7
#   experiments/chair8-4r-mixed-robust-seed1-plan.json  / chair8
python3 main.py --config CONFIG --solve \
  --no-open-witness --optimize-witnesses off --witness-sets 0 \
  --no-fuse-cograsp --quiet-rai --no-view \
  --save experiments/bench-TAG-legacy-plan.json

python3 main.py --config CONFIG --solve \
  --open-witness --open-witness-batch 7 --open-witness-epochs 2 \
  --optimize-witnesses off --witness-sets 0 --no-fuse-cograsp \
  --quiet-rai --no-view --save experiments/bench-TAG-open-b7e2-plan.json
```
