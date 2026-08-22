# Collision-certified phase scheduling for multi-robot assembly

## Method overview

We assume that a task-and-motion planner has already produced a collision-free multi-robot
joint path together with an ordered sequence of discrete assembly events. The geometric path is
therefore fixed. Our problem is not to find new robot motions, but to replace the path's single
shared clock with one monotone phase variable per robot. This permits robots to move
asynchronously while preserving their individual geometric paths, the assembly-mode sequence,
and the configurations at which grasps, releases, placements, and mates occur.

The method is deliberately biased toward visually regular motion rather than minimum
makespan. A moving robot should travel near a preferred joint-space speed, slow only near a
corner or an event that it participates in, and stop only for a physical or topological reason.
Makespan reduction is a consequence of allowing independent progress, not the optimization
objective.

## Problem formulation

Let the saved composite path be

\[
q(s) = [q_1(s),\ldots,q_R(s)], \qquad s\in[0,S],
\]

where \(R\) is the number of robots and \(s\) is arc length in the complete joint space. The
path is partitioned by event barriers

\[
0 < b_1 < \cdots < b_K = S.
\]

Barrier \(b_k\) terminates one motion-mode stretch and is associated with a set of task events
such as attaching a part to a gripper, mating two parts, or releasing a grasp. The original
execution follows the diagonal of the coordination space: every robot uses the same phase
\(s(t)\).

We instead search for a monotone coordination-space path

\[
\sigma(t)=[s_1(t),\ldots,s_R(t)]\in[0,S]^R,
\qquad \dot{s}_r(t)\geq 0,
\]

and execute the mixed configuration

\[
Q(\sigma)=[q_1(s_1),\ldots,q_R(s_R)].
\]

Thus, no robot leaves its planned joint path. Only the relative timing of the robots changes.
The diagonal \(s_1=\cdots=s_R\) remains a known feasible fallback.

The collision predicate is mode dependent. Let \(m_k\) denote the attachment topology before
event \(k\). A candidate phase vector is admissible only if

\[
\mathcal C(Q(\sigma),m_k)=\mathrm{free}.
\]

In the implementation, \(\mathcal C\) is the same complete predicate used by the motion
planner. It includes robot--robot, robot--part, and environment collisions, the configured
penetration tolerance, and additional saved clearance gates such as the robot-margin wrapper.

## Event semantics

Each event has an actor set \(A_k\). This contains not only the robot named by the symbolic
action, but also every robot holding a part touched by that action. For example, when a wheel is
mated to a car body, both the wheel robot and any robot currently holding the body must be at
the rendezvous configuration.

An actor may not pass its next unfinished event:

\[
s_r \leq \min\{b_j\mid j\geq k,\ r\in A_j\}.
\]

Robots that do not participate in the pending event may cross its scalar barrier and continue
under the old attachment mode. Event \(k\) fires only when

\[
s_r=b_k\quad\forall r\in A_k,
\qquad\text{and}\qquad
\min_r s_r\geq b_k.
\]

The first condition creates the required physical rendezvous. The second is a global-passage
condition: no lagging robot may remain in the pre-event portion of its lane when the world is
changed to the post-event attachment topology. Without this condition, a newly placed object
can appear in front of a lagging robot in a portion of coordination space that was certified
only before the placement.

Actors remain stationary for a short pre-event ceremony and post-event attachment dwell.
Non-actors continue moving. A fixed nominal servo duration is reserved before the event; the
current default is 0.15 s for settling and an additional 0.15 s for a seat/place servo when
applicable. These durations are feed-forward assumptions made before execution, not estimates
adapted from earlier runtime measurements.

## Per-robot velocity parameterization

Using the composite arc directly as a clock makes a robot crawl whenever its contribution to a
composite segment is small. For composite segment \(j\), define

\[
\ell_j=\|q^{j+1}-q^j\|_2,\qquad
\ell_{rj}=\|q_r^{j+1}-q_r^j\|_2,\qquad
\alpha_{rj}=\ell_{rj}/\ell_j.
\]

For a preferred realized joint speed \(v\), the phase rate of robot \(r\) is

\[
\dot{s}_r = \frac{v}{\alpha_{rj}}\rho_r(s_r),
\]

where \(\rho_r\in[\rho_{\min},1]\) is a ramp factor. Consequently,
\(\|\dot q_r\|\simeq v\rho_r\), independently of how many other robots move in the same
composite segment. If \(\alpha_{rj}=0\), the segment contains no physical motion for robot
\(r\) and is skipped as zero-duration phase bookkeeping. The configuration after a skip is
still collision checked.

The ramp factor decelerates an actor near its event barrier and accelerates it afterward. A
non-actor does not stop merely because it crosses another robot's event. It is slowed only when
the saved path has a genuine joint-space corner. If \(u^-\) and \(u^+\) are the normalized
incoming and outgoing directions for that robot, the crossing-speed floor is proportional to

\[
\operatorname{clip}\left(
\frac{\delta_v}{\|u^+-u^-\|},\rho_{\min},1
\right).
\]

Straight crossings therefore retain cruise speed, whereas sharp corners are traversed more
slowly. In the current implementation, \(v=1\) rad/s, the ramp distance is 0.2 rad,
\(\rho_{\min}=0.04\), and \(\delta_v=0.2\).

## Coordination-space search

The scheduler incrementally constructs \(\sigma(t)\) on a fixed time grid. At each tick it:

1. orders independent robots by increasing phase, with a deterministic priority tie-break;
2. proposes one forward step for each robot, capped by its next actor event and any active
   coupling window;
3. accepts the step only if the complete collision predicate accepts the resulting mixed
   configuration in the current mode;
4. advances any explicitly coupled robot components together; and
5. if no individual move succeeds, tries one collision-checked simultaneous step of all
   otherwise movable robots.

The laggard-first order discourages unnecessary phase spread, but it is only a search heuristic.
Safety comes from the collision predicate and the final configuration-resolution audit. A blocked robot holds its
phase while other robots are allowed to move and clear the conflict.

In pseudocode:

```text
sigma <- (0, ..., 0); event <- 1
repeat
    construct active coupling components
    for robot r in laggard-first order do
        proposal <- forward_step(r, sigma[r])
        if collision_free(mix(sigma, r, proposal), mode[event]) then
            sigma[r] <- proposal
    for fused component G do
        proposal <- common_forward_step(G)
        if collision_free(mix_component(sigma, G, proposal), mode[event]) then
            sigma[G] <- proposal
    if no phase advanced then
        try one collision-checked simultaneous step
    append Q(sigma)
    if actors are at the barrier and all robots have passed it then
        reserve the event ceremony, fire the event, and update the mode
until all events fired and sigma = (S, ..., S)
```

## Counterexample-guided local coupling

Purely prioritized progress can enter a coordination-space wedge: each robot's individual
forward step is in collision even though the original diagonal formation can pass. Rather than
running an online escape controller, the scheduler uses a counterexample-guided restart.

The collision predicate reports which robot owns each rejected geometry or margin gate. After
250 no-progress ticks, the scheduler selects a blocked robot and the most phase-separated robot
in its reported blocker set. It then introduces a local equality constraint for that pair over
an arc window

\[
s_i=s_j,\qquad s_i,s_j\in[a,b].
\]

The window begins 0.5 rad before the earlier phase and ends at the first event barrier after
the later phase. Both robots may move independently before \(a\); the first to reach \(a\)
waits for the other, and they traverse the window on the known diagonal formation. Until both
have arrived, **both** phases are capped at \(a\): the laggard may otherwise skip a zero-motion
composite span and overshoot the rendezvous in one per-arm-rate step. That bug caused the C1
peg-stool scheduler to deadlock at event 14 even though its synchronized diagonal was valid.
Coupling
windows that share a robot form a larger fused component. Windows remain disjoint unless they
overlap, preventing an isolated conflict near the end of an assembly from forcing lockstep over
the entire path.

After adding a coupling window, scheduling restarts from the beginning. Previously evaluated
collision verdicts are retained, so later attempts mainly evaluate configurations introduced by
the new coupling. If local coordination cannot be found within the restart budget, the method
uses the synchronized path as a conservative fallback rather than executing a partial schedule.

## Collision-audited phase-rate smoothing

The greedy schedule is safe but changes an individual robot's rate discretely whenever a
candidate is blocked or released. We therefore apply an optional postprocessor to the phase
rates. This is intentionally not another joint-space spline: the latter could cut across the
fixed geometric path. The shared coordination coordinate (s_r) is not a physical distance for
robot (r): it also advances through composite-path segments on which that robot is stationary.
We therefore map it to the robot's intrinsic cumulative joint-space arc (u_r(s_r)), and let

\[
\Delta u_{r,t}=u_r(s_r(t+\Delta t))-u_r(s_r(t))\geq0
\]

denote the raw intrinsic increments. We convolve these increments with a positive Gaussian
kernel and project each fire-to-fire interval onto the non-negative simplex with its original
total intrinsic displacement. This Euclidean projection is the smallest L2 correction to the
filtered rates; the former multiplicative normalization could impose a different scale factor
on each side of a mode switch. The result is mapped monotonically back to (s_r). Event anchors
are then restored exactly, so the complete shared phase vector remains pinned at every switch.
A short zero-rate neighbourhood is imposed only for robots in the event's actor set; crossing
robots are not forced to stop.

Rigid co-hold intervals and equality windows introduced by counterexample-guided coupling are
kept at their original samples. Source rates are also retained over the scheduler's slowdown
envelope around a crossing robot's sharp mode-boundary corner. After reconstruction on the
original per-robot paths, each
robot's largest discrete joint-space step is compared with its source schedule. Any interval
that would exceed the source peak by more than 2% is restored for that robot. This guard is
important: fixed event times can otherwise compress redistributed motion into a short burst,
which appears smoother in the phase signal but excites the physical actuator.

Collision freedom alone is not sufficient evidence that a smoothing proposal is useful. The
aggregate joint acceleration and jerk RMS are therefore compared against the source schedule;
if either regresses by more than 2%, the run aborts before physics and short-motion batching.
This fail-closed branch was exercised by the old peg-stool composite-phase smoother, whose
collision-free proposal increased acceleration by 8.9% and jerk by 15.6%. Intrinsic-arc
smoothing fixes the cause rather than weakening the gate: on the same artifact it reduces the
two metrics by 20.4% and 20.5%, respectively.

Finally, the smoothed trajectory receives the configuration-resolution collision audit described below.
All fire-to-fire intervals containing a counterexample are restored from the certified source,
and the result is audited again. If a rejection remains on an already restored interval, or
repair does not converge, the run aborts before physics. The certified source remains a local
repair within the smoothing algorithm, but is never a whole-trajectory execution fallback when
`--phase-smooth` was requested. Likewise, a synchronized scheduler fallback has no independent
phase trace and causes a strict failure. Because pinned and locally restored boundaries remain,
the result is not claimed to be a globally C2 spline; acceleration and jerk reductions are
measured explicitly instead.

## Short-motion batching

Gaussian rate smoothing addresses high-frequency changes but intentionally pins the complete
phase vector at every mode switch. It can therefore leave a short motion island separated from
a larger motion by a brief stop. As a final optional stage, we detect contiguous motion islands
from the actual joint reference. For adjacent islands \(I_1,I_2\), a batching proposal is made
when

\[
\min\bigl(L(I_1),L(I_2)\bigr)<L_{\min},\qquad
T_{\mathrm{gap}}<T_{\max},
\]

with defaults \(L_{\min}=0.08\) rad and \(T_{\max}=0.35\) s. The combined phase displacement
is packed into one contiguous, smoothly ramped burst. Left-packed, right-packed, and centred
placements are considered; attaching the smaller island to the larger one is preferred. This
keeps motion near its prior speed and creates one longer stationary interval, rather than
turning the gap into low-speed creep.

The window endpoints and total duration remain fixed. A proposal may not cross an event owned
by that robot or a rigid coupling interval, may not increase the robot's source peak step, and
may not introduce a new slow interval bracketed by fast motion. A foreign event may be crossed
only when every robot still satisfies the barrier-passage inequality at its fire sample. The
candidate configuration is checked in both the pre-event and post-event modes. The final dense
audit is repeated; a missed counterexample restores the connected offending merge windows and
audits again. Consequently `--phase-batch` is common to the default and C2 geometric paths and
may retain its preceding trajectory as a local or global batching fallback. When
`--phase-smooth` is also requested, that predecessor is already the certified smoothed
trajectory; batching can never resurrect the raw phase schedule rejected by the strict
smoothing contract.

## Intentional-motion support minimization

`--phase-intent` is an experimental alternative to short-motion batching. Its visual objective
is fast, purposeful motion separated by unambiguous idle periods. For fixed trajectory duration
and event times, the idealized objective is

\[
\min \sum_{r,t}\mathbf{1}\!\left[\|q_r(t+\Delta t)-q_r(t)\|/\Delta t
\geq v_{\mathrm{stop}}\right],
\]

subject to monotone path progress, every event-phase equality, actor stops, rigid couplings, a
peak path rate, and collision freedom in the active mode. This is non-convex even before adding
collisions. The current implementation uses a conservative greedy approximation. Each robot's
own task events partition time into robot-owned intervals. Its shared progress is converted to
intrinsic joint-space path arc, thereby deleting composite-path flats in which only other robots
move. The interval's fixed intrinsic displacement is packed into the shortest motion admitted by
the 1.5 rad/s default intent rate, with 150 ms quintic velocity ramps. Multiple continuous
durations are considered; the shortest one that does not worsen total or long all-robot idle is
placed as late as event deadlines and collision geometry permit.

Only events owned by that robot impose rest. At a foreign switch the robot must have passed the
global barrier and its configuration must be valid in both pre- and post-switch modes, but its
velocity is free and normally nonzero. True co-hold and handover spans remain byte-exact.
Scheduler-created collision couplings only certify the preceding trace and are deliberately
reopened: the intent proposal is collision-screened on a coarse time grid including every switch,
then the complete trajectory is audited at 2 ms. A late counterexample restores its connected
robot-owned interval and repeats the audit.

A pre-skill GOTO changes neither attachments nor controller state, so it remains a geometric
waypoint but is neither a rest anchor nor a hard timing deadline. A second coordinate-refinement
pass revisits early robots after all other compact motions are available. The visual objective is
lexicographic: never worsen all-robot idle, minimize continuous moving support, then maximize
lateness. This prevents independent idle maximization from stacking every robot's wait at once.

This deliberately explores farther from the synchronized diagonal than Gaussian smoothing or
local batching. A narrow coordination corridor therefore manifests as rejected placements, not
an unsafe execution. On C1 peg-stool, 13 accepted robot-owned intervals raise idle fraction from
59.1% to 62.1%, increase conditional moving rate from 0.68 to 0.73 rad/s, reduce islands from 171
to 121, and reduce internal slow gaps from 512 to 363. All-robot idle improves from 9.9 to 7.8 s;
acceleration and jerk RMS improve by a further 16.0% and 16.7% relative to the smoothed input.
The remaining boundaries are own physical task stops, true rigid spans, or intervals whose
alternative timing could not be certified. If no strict support reduction survives,
`--phase-intent` aborts before physics.

## Certification and multi-rate construction

Certification is performed in several stages:

1. **Base-path preflight.** The saved motion path is interpolated at 0.01 rad resolution and
   checked for joint-limit violations and collisions. Retiming is not allowed to conceal an
   invalid geometric path.
2. **Path preprocessing.** Redundant serialized edge samples are removed within each mode.
   The default optional Catmull--Rom smoother preserves every event witness and is accepted per
   span only after collision checking. An independent experimental `--c2-spline` alternative
   fits ordinary waypoints softly with mode-local cubic B-splines, pins event endpoints, keeps
   closed-chain stretches unchanged, and accepts a complete stretch only after dense collision
   and joint-limit certification. Either stage emits ordinary waypoints; all phase-planning
   stages below are identical.
3. **Phase search.** Every accepted coordination step is checked with the planner predicate.
   The deployed configuration uses an effectively zero collision-cache cell, so distinct
   candidates receive distinct checks.
4. **Rate smoothing and batching.** Optional Gaussian phase-rate smoothing and short-motion
   batching produce proposals under event, coupling, monotonicity, and peak-rate constraints.
   Collision-sensitive intervals or merge windows revert locally.
5. **Configuration-resolution audit.** Search begins on a 10 ms grid and the result is
   interpolated to the 2 ms controller grid. Audit samples are selected so accumulated
   L-infinity joint-path variation between checks is at most 0.01 rad, matching the motion
   planner's edge resolution. Accumulated variation rather than endpoint distance retains
   backtracking excursions. The first and last samples and every event configuration are
   mandatory; event configurations are tested in both pre- and post-event modes. A failure
   triggers a new search at 4 ms and then 2 ms. Set `--phase-audit-resolution 0` for the old
   every-controller-tick audit.
6. **Fallback audit.** If all asynchronous searches fail, the synchronized reference is built
   and subjected to the same configuration-resolution audit. If this also fails, simulation
   is aborted.

This is a sampled configuration-space certificate at the planner's 0.01 rad edge scale, not a
formal proof over the continuous interpolation between samples. The planner's clearance gates
and conservative robot geometry are engineering margins. The 2 ms trajectory remains the
controller reference; collision checks are configuration-driven rather than time-driven.

## Physics execution

The accepted schedule is a reference for actuated MuJoCo robots. Position servos use velocity
feed-forward and gravity compensation. At a mode switch, measured convergence is checked only
for the event's actors and holders; a non-acting robot's legitimate tracking residual does not
create a global stop. Small finite-gain residual plateaus are accepted only below twice the
requested settle tolerance and after negligible improvement over 0.1 s.

Contact-rich terminal skills add a bounded task-space admittance offset, guarded part-mating
servo, and force-terminated table-placement behavior. Crucially, phase-scheduled execution does
not run the older
minimum-over-arms reactive clock underneath the offline schedule. That clock caused all robots
to stop whenever one placement activated its force guard. Contact now causes local compliant
yield while the independently scheduled references continue. Cross-arm MuJoCo contacts and
physical grasp/mate breakaway limits remain active as execution-time backstops.

The geometric certificate applies to the nominal reference. Admittance may deliberately move
the executed robot by up to its configured bound (currently 10 mm), so this compliance layer is
not claimed to preserve the exact reference certificate. It is instead protected by physical
contact response and breakaway monitoring. Incorporating the bounded compliance tube directly
into the geometric certificate is future work. Likewise, predictive wrench-derived speed and
acceleration limits are currently reported but not consumed by the phase scheduler.

## Current validation

Using only saved plans and motion paths, with no IK or motion replanning:

- **Car-mini:** 19/19 actions and 4,423 recorded frames. The 10 ms candidate still exposed the
  known arm/object counterexample under the 0.01 rad audit; the 4 ms retry passed. The schedule
  audit selected 3,764/35,134 controller samples plus all 34 post-switch modes: 3,798 predicate
  calls in 0.41 s, versus about 34,840 calls/1.83 s for the old dense audit.
- **Chair:** 34/34 actions and 9,172 recorded frames. The 10 ms search passed. The schedule
  audit selected 9,211/72,575 controller samples plus all 60 post-switch modes: 9,271 predicate
  calls in 1.25 s, versus about 72,635 calls/7.86 s for the old dense audit.
