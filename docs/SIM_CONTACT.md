# Physical-contact execution experiment (`sim_contact.py`)

`sim_contact.py` is an opt-in execution entry point, not a replacement for the planner or
for `sim.py replay`. It uses the **saved** task plan and motion path, the existing
MuJoCo robot model and skill servos, and no new IK or motion planning. Its small
new rule is that a mating event does not discard a finger contact that remains in
the saved symbolic state. The contact is monitored until an explicit release or
transfer. Re-rooting a planner attachment onto an already holding robot does not
command another grasp. A failed action stops execution instead of producing a
cascade of downstream failures. `sim.py` itself is unchanged by this experiment.

This separates three concerns:

1. The motion path is a geometric existence certificate, not a clock.
2. By default, the existing collision-audited independent phase scheduler gives
   each robot a monotone clock along its saved path. It reserves nominal skill
   time; unrelated arms may move during another arm's gripper or skill wait.
   Runtime controllers can take longer, but may advance only within the
   certified phase slack. Lockstep is available as a diagnostic baseline.
3. Pick, handover and release use the articulated finger contacts and measured
   bilateral pad forces. Insertion receives the saved nominal process wrench.
   Part--part mates still use a capture-checked ideal datum weld.

This is intentionally **not** MPCC, online motion replanning, or fully
contact-only peg dynamics. An offline collision certificate cannot prove that
the physics-tracking error is collision-free; the run reports physical failures.
A future MPCC could replace only the phase-clock policy, leaving the contact and
skill event contracts in place.

## Reproduce the first tests

Use the repo environment and `PYTHONHASHSEED=0`. The current initial scope is
all two-finger arms (tested here on UR5), 2 ms physics, actuated arms, a saved
motion plan from the same task plan, and an a-priori
`--grasp-physics-check` certificate. The contact experiment defaults
to phase timing, 0.5 rad/s joint cruise, actual finger contact, nominal process
wrench, and stopping at the first failed action. These can be overridden where
the CLI offers an explicit flag; `--execution-timing lockstep` is for diagnosis.

```bash
python3 sim_contact.py run experiments/simv2/cross4-3r-plan.json \
  --mp-path experiments/simv2/cross4-3r-mp.json \
  --record experiments/simv2/cross4-3r-phase.npz \
  --report-json experiments/simv2/cross4-3r-phase-report.json
python3 sim.py replay experiments/simv2/cross4-3r-phase.npz 2
```

The fresh Cross-4 plan was generated with:

```bash
python3 main.py --solve --search-profile standard --assembly cross \
  --parts 4 --num-robots 3 --robot ur5_limited \
  --grippers two_finger,two_finger,two_finger \
  --grasp-physics-check --wrench --seed 11023 \
  --save experiments/simv2/cross4-3r-plan.json --quiet-rai --no-view
python3 main.py --config experiments/simv2/cross4-3r-plan.json \
  --load-plan experiments/simv2/cross4-3r-plan.json \
  --motion-plan --motion-profile standard --mp-runtime 30 \
  --mp-save experiments/simv2/cross4-3r-mp.json --quiet-rai --no-view
```

The 11-action Cross-4 run succeeded with three nominal 10 N insertions and
a co-grasp span. The phase schedule was collision-audited at 0.002 rad path
resolution. Its estimated physics duration was 17.9% shorter than synchronized
timing; its complete wall time was about 83 s on the test machine. A separate
Cross-2 smoke test succeeded 3/3 actions: phase execution took about 15 s wall
versus about 36 s lockstep. These are two tests, not a robustness benchmark or
a claim that v2's MuJoCo loop is faster than `sim.py` with the *same* phase mode.

## Boundaries to keep explicit

- Finger grasps are contact-driven, but a successful mate becomes an ideal
  part--part equality after a capture check. This models a rigid snap/fastener,
  not forces through an unfinished insertion.
- Vacuum and mixed hands are not enabled in this first contact entry point. The
  original `sim.py` contact mode can still run mixed experiments.
- The 2 ms physics step is deliberate: the existing pad contact calibration
  creeps at 4 ms because MuJoCo softens the contact under its refsafe limit.
- The phase scheduler still has intentional stops at event barriers and
  co-grasp locks. The controller does not invent collision-safe asynchronous
  motion if the saved paths have no certified slack.
- Keeping the physical hand across the mate removes a bookkeeping error, but
  can expose a genuine slip if the saved post-mate arm path pulls away before
  the release. Such a run should fail visibly, not silently become a weld hold.

Next validation should include a longer Cross with handovers, a pegged Chair,
and repeated seeds. Only after those pass should the contact event contract be
considered for the main simulator.
