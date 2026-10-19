# Execution-program simulator (`simv2.py`)

`simv2.py` is a new, opt-in executor. It does not call the legacy
`sim.execute_motion_plan`, retiming pipeline, or IK. It compiles an explicit
program from a saved task plan (`motion_data`) and motion path (`mp_export`):
path segments alternate with named skill events carrying participant robots,
attachment frames, relative targets, and explicit pre/post support modes. A mismatch
in plan identity, task signature, joint order, or mode/event chain fails before
physics starts. The saved motion plan remains the geometric existence proof.

The program JSON is checksummed and can be executed directly with `run-program`;
its plan path still needs to resolve so the exact scene and symbolic-state
certificate can be restored. The first execution mode is a deliberately simple
lockstep reference. The MuJoCo plant still has dynamic parts, contact, grasp/mate attachment and
breakaway monitoring. Arm paths advance together at a bounded joint rate, using
one continuous path-progress profile across each saved free-motion polyline rather
than stopping at every motion-planner waypoint. Participating arms still settle
at skill barriers. Gripper closure, grasp acquisition,
insertion seating, release, and post-event support checks complete from measured
state, not a reserved timer. Nominal process loads are applied over the final
entry-to-seat segment by default. An actuated arm plant is opt-in; the default
uses kinematic articulated arms for a reproducible event-contract baseline.
A failed event aborts and still writes a replay and report when requested.

```bash
python3 simv2.py compile experiments/simv2/cross2-plan.json \
  --mp-path experiments/simv2/cross2-mp.json \
  --program-json /tmp/cross2-program.json

python3 simv2.py run-program /tmp/cross2-program.json \
  --record /tmp/cross2-v2.npz --report-json /tmp/cross2-v2.json
python3 sim.py replay /tmp/cross2-v2.npz 2
```

Use `--actuated` to test the bounded PD plant. `--contact-grasps` is an
experimental two-finger option; the earlier standalone contact experiment is
now `sim_contact.py` and documented in [SIM_CONTACT.md](SIM_CONTACT.md).

At every assemble event, a one-shot preflight projects the collidable tool to
the seated target using the *measured* tool-to-part transform. It reports the
minimum gap to the receiver in `mate_preflight`, including a separate moving-pad
gap. This endpoint check does not certify the loaded insertion trajectory. Ideal
attachment mode reports it diagnostically; physical-contact mode requires at
least 1 mm of exact-mate closed-hand clearance by default. The same gate is
applied when generating physics-certified grasps. `--mate-grasp-clearance M`
overrides only the execution preflight; old plans with a blocked hand now fail
there instead of being driven through the receiver.

The actuated seat skill has two geometric phases: align lateral position and
orientation at an axial standoff, then advance along the mating axis. Each tick
uses the measured parent pose and the tool-to-part transform captured at skill
entry to derive a tool-pose target. The receiver holder may cooperate through
the same relative-pose task. This avoids feeding contact-induced part oscillations
directly into the incoming arm's Jacobian. The resolved-rate command is projected
onto joint-rate, acceleration, tracking-error, and local tool/receiver-clearance
bounds (1 mm target reserve). If these bounds prevent convergence, the skill
fails rather than relaxing mate or grasp checks. Every simv2 mode
now requires a measured relative mate error below 2 mm; the legacy sim.py
retains its historical 10 mm capture band. The contact transition
activates the mate equality without projecting the part away from its hand
or resetting the hand-to-part slip reference. A continuing inserter holds
its achieved posture while the assembly is stationary on the table or
supported by another robot; it follows the measured assembly only when that
supporting robot is commanded to move. This prevents uncommanded support
motion from being amplified by the follower. The finite skill budget allows
the authored funnel to be traversed at guarded speed.

This is the reference executor, not the eventual online asynchronous
coordinator. It intentionally does not claim independent phase progress or a
collision-certified connector after a large skill correction. The bounded,
monitored handoff after a small correction is not such a certificate. Larger
corrections require an online safety filter over observed robot/object state.
Keeping this baseline
separate makes the future phase controller replace only path progress, not task
semantics or the MuJoCo backend. The older `sim.py` remains available for
reproducing paper results.

## Validation and current limits

Earlier, the actuated physical-contact executor completed Cross-4 (11/11
actions) and Husky peg-in-hole (3/3) with the saved 10 N insertion load.
Those old plans used the permissive mate preflight and are not valid references
for the stricter exact-mate contract: their selected closed hands overlap the
receiver at the authored mate pose. The independently saved Husky bench plan
completed 8/8 with actuated arms and ideal attachments under its 10 N load.

A freshly searched and motion-planned Husky peg-in-hole plan completes 3/3
physical-contact actions with the 10 N process load
(`experiments/simv2/peg-hole-contact-exact-seat2.npz`). A fresh Cross-4 plan
clears both exact-mate preflights and completes 12/15 actions. Its next
handover fails because the giver's physical fingers block the receiver's:
the offline handover-pair certificate does not currently model both hands
in MuJoCo. These are execution checks, not a general success guarantee.

The complete 16-action Fabrica Car now succeeds with actuated arms, ideal
grasp attachments, and its saved 10 N process load. This does not yet establish
physical finger-contact grasping for the full Car. In the scatter checks below,
physical-contact Cross mates first entered the 10 mm capture band in 1.34–1.79 s,
Car mates in 1.49–2.19 s, and Husky bench mates in 1.03–2.09 s. Longer event
durations can be caused by concurrent robot motion. These are simulated skill
times, not wall-clock runtimes.

```bash
python3 simv2.py run experiments/simv2/peg-hole-contact-exact-plan.json \
  --mp-path experiments/simv2/peg-hole-contact-exact-mp.json \
  --actuated --contact-grasps \
  --record experiments/simv2/peg-hole-contact-exact-seat2.npz
python3 simv2.py run experiments/project-page-replays-current/husky-plan.json \
  --mp-path experiments/project-page-replays-current/husky-mp.json \
  --actuated --record experiments/simv2/husky-bench-v2.npz
python3 simv2.py run experiments/project-page-replays-current/fabrica-car-fullprobe-plan.json \
  --mp-path experiments/project-page-replays-current/fabrica-car-fullprobe-mp.json \
  --actuated --record experiments/simv2/fabrica-car-seated-hold.npz \
  --report-json experiments/simv2/fabrica-car-seated-hold-report.json
python3 sim.py replay experiments/simv2/fabrica-car-seated-hold.npz 8
```

The remaining full-assembly cases fail closed:

- Chair-8 stops after action 3: the vacuum holder exceeds its 3.75 N lateral
  limit during transport, even at 0.2 rad/s; no external contact is present.
- Full OpenArm Cross reaches action 10 with physical pinches, then misses the
  next seat by 2.2 mm/5.3 degrees. With ideal attachments it seats, but the
  inserter is 0.68 rad (about 5 cm/32 degrees at the EE) from the saved path,
  beyond the executor's certified local rejoin range.

`simv2` no longer treats a brief pause in axial progress as a jam. It keeps the
measured force/breakaway guard and finite seating budget; the legacy `sim.py`
retry policy is unchanged. Loaded Cross-4 and Husky bench were rerun after
this change (11/11 and 8/8 actions, respectively).

Full OpenArm testing exposed a separate physical-contact bug: moving finger
pads were disabled by the old wrist exemption after acquisition and mating.
They now remain collidable; Cross-4 and peg-in-hole still pass.

Current action support is pick, assemble, place, release, add-grasp and
handover. In-hand reorientation and screw skills fail closed. The reference
coordinator bounds joint speed along each saved path edge and scalar progress
acceleration across each whole free-motion segment. A sharp geometric corner can
still change joint-velocity direction abruptly; geometric smoothing would need its
own collision certificate. The coordinator does not certify independent phase
deviations. Small measured skill-to-path discrepancies
use a bounded connector monitored for physical failures; a large discrepancy
fails. Collision certification of that connector and an online phase safety
filter are later steps.

## Scatter robustness check

Run the guarded, resumable two-seed matrix with fresh task search, motion
planning, and execution:

```bash
python3 experiments/simv2/run_scatter_robustness.py \
  --scenes cross fabrica-car real-stool --seeds 11024 11025
```

This randomizes each initial part in a 0.2 m half-width box and randomizes
yaw. Stages have 6 GiB memory limits and explicit timeouts. `real-stool` is
the saved three-part Husky `real-bench` scene. Cross uses physical finger
contacts; Car and Husky use ideal grasp attachments (all use actuated arms).
The run writes `plan.json`, `mp.json`, `replay.npz`, `report.json`, and stage
logs per seed under `experiments/simv2/scatter-robustness/`. After a controller
change, `--retry-sim` reuses successful search and motion artifacts.

In the 2026-09-24 matrix, Cross seed 11024 succeeded (14/14 actions);
seed 11025 failed in execution and was later rerun successfully with the
bounded local handoff below. Both ideal-attachment Husky seeds succeeded (8/8),
and ideal-attachment Fabrica Car seed 11025 succeeded (18/18). Car seed 11024
hit the 450 s task-search cap before motion planning; that is a search
robustness result, not a simulator outcome.

## Physical-contact follow-up (2026-09-25)

`--physical-contact` on the scatter runner requests fresh isolated six-direction
grasp certificates during search and `--contact-grasps` in simv2. The runner now
keeps the visual meshes in replay NPZs. Physical finger contacts remain active
through handover and mating; a grasp is acquired only with two loaded pads.
Transport loss is measured as sustained tool-to-part pose drift (10 mm or 10 deg
for 20 ms). A one-sided contact-force reading is logged after 100 ms but does not
itself break a geometrically retained grasp: MuJoCo can show a 0.06 mm unloaded
pad gap while the payload has not moved. This is an execution-loss threshold,
not the isolated grasp certificate\u2019s stricter 3 mm quality threshold.

Current successful visual replays:

```bash
python3 sim.py replay experiments/simv2/scatter-robustness/cross__seed-11024/replay-final-contact.npz 8
python3 sim.py replay experiments/simv2/scatter-physical-cross-current/cross__seed-11025/replay-local-handoff.npz 8
python3 sim.py replay experiments/simv2/scatter-physical-cross-third/cross__seed-11026/replay.npz 8
python3 sim.py replay experiments/simv2/scatter-physical-peg-hole/peg-hole__seed-11024/replay-force-boundary.npz 8
python3 sim.py replay experiments/simv2/physical-contact-husky/force10-current-contact.npz 8
python3 sim.py replay experiments/simv2/cross4-handover-articulated-contact.npz 8
```

For Robotiq handovers, the a-priori RAI certificate now rotates the six finger
hinges using the same mimic map as MuJoCo. A translated-pad check had accepted
an `obj_3` handover whose receiving pads hit the giver's fingers despite the
receiver reaching its planned pose within 0.6 mm. The joint-space certificate
rejects that pair; the fresh 3-UR5, 4-part Cross above completes 16/16 actions
with actuated arms and physical contacts. The OpenArm floating collision proxy
does not attach its finger capsules to its hinges yet, so it retains the older
proxy certificate. The first Cross search certified 4,222 pairs in 39.7 s; a
subsequent solve with `--grasp-physics-cache-plan <saved-plan.json>` reuses the
signature-checked pair table instead of repeating this all-pairs work.

All three Cross replays use freshly searched, physics-certified grasps
and motion plans with independently scattered initial parts (0.2 m
half-width). Seeds 11024 and 11025 finish 14/14 actions, including contact-only
handovers; seed 11026 finishes its different 13-action plan. Their embedded
MuJoCo scenes contain 51 visible mesh geoms. The scattered Husky peg-in-hole
finishes 3/3 under its 10 N insertion load. The unscattered Husky bench
finishes 8/8 with physical contacts and the 10 N load using a saved
six-direction 10 N grasp certificate. All five replays retain visible meshes.

Two contact-execution corrections were needed. The admittance force estimate
now treats a holder's own finger contacts as internal to the grasped system;
released hands remain external contacts. A skill may also end at a different
joint configuration from the saved path despite a nearby tool pose. The local
rejoin permits up to 0.30 rad only when every participant's tool is within
20 mm and 5 degrees of its path waypoint, then runs the same monitored
connector. This connector is not yet independently collision-certified.

Fabrica Car remains open for physical contacts: two selected grasps put a
moving finger pad 8.6 and 8.3 mm inside the receiver at the first mate.
The seated-pose preflight rejects them. A third grasp search reached its
450 s cap while recomputing a-priori handover-pair certificates, before motion
planning. This is a grasp/receiver-geometry and preprocessing issue, not
evidence that a slower insertion controller would help. A separate failing
Cross plan showed an open released finger colliding with the carried assembly
later in free motion; the planner's disabled finger joints miss that geometry.
Neither failing plan is counted among the successful replays.
