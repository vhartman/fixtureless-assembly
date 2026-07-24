"""Time-parameterize a collision-free geometric path into an executable trajectory.

PORTABLE BY DESIGN (pure numpy; no rai / MuJoCo / scene imports) so the same code
can run on the mr_bench side later. The contract is exactly:

    input :  a DAG/sequence  +  a collision-free path (waypoints per mode-stretch)
    output:  a trajectory q(t), sampled at dt, that is at REST (v=0) at every
             stretch boundary where a task-event (grasp / mate / release) fires,
             optionally acceleration-limited.

Geometry is NEVER touched -- we only assign *time* along the given waypoints. This
is the "execution-side retiming" leg: proper kinodynamic MP is the real fix, but
retiming a certified collision-free path is a lighter, path-preserving stand-in and
is where an APEX-MR-style asynchronous coordinator would eventually plug in (it
would emit one such stretch-decomposition per robot instead of one composite one).

Two layers:
  * PURE RETIMING PRIMITIVE -- `retime_stretch`: one stretch of waypoints -> configs
    sampled every dt, ramping from ~rest to cruise and back so the final sample sits
    on the goal with v~=0. `build_trajectory` stitches stretches + dwell + settle into
    a single `TimedTrajectory` (the portable output object).
  * PATH DECOMPOSITION -- `split_stretches`: a mode-tagged path + accessor callables
    (so it stays free of any mr_bench / rai type) -> [(waypoints, done_task_ids)].

The sim currently STREAMS `retime_stretch` (so physics stepping/attaches interleave);
`build_trajectory` produces the same samples as a standalone object for callers that
want the trajectory as data (mr_bench, tests, offline inspection).
"""
from dataclasses import dataclass, field

import numpy as np

# Library defaults for the interim trapezoidal profile. The sim passes its own
# module-level tunables explicitly at the call site (single active source of truth);
# these are the standalone/testing defaults and mirror sim.py's values.
DEFAULT_JOINT_RATE = 1.0        # rad/s : cruise speed along the joint-space arc length
DEFAULT_RAMP_DIST = 0.2         # rad   : ramp to/from ~rest over this much arc at each end
DEFAULT_RAMP_MIN = 0.04         # frac  : floor on the ramp (never exactly 0 -> guaranteed arrival)


def arc_length(qs):
  """Per-segment joint-space lengths and their total for a waypoint list."""
  segs = [float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))
          for a, b in zip(qs, qs[1:])]
  return segs, float(sum(segs))


def config_at(qs, csum, segs, s):
  """Linear-interpolate the waypoints at arc length `s` (0..total)."""
  i = min(int(np.searchsorted(csum, s, side="right")) - 1, len(segs) - 1)
  if segs[i] < 1e-12:
    return np.asarray(qs[i + 1], float)
  a = (s - csum[i]) / segs[i]
  return (1 - a) * np.asarray(qs[i], float) + a * np.asarray(qs[i + 1], float)


def retime_stretch(qs, dt, joint_rate=DEFAULT_JOINT_RATE, ramp_dist=DEFAULT_RAMP_DIST,
                   ramp_min=DEFAULT_RAMP_MIN, accel=None):
  """Trapezoidal time-parameterization of ONE collision-free stretch of waypoints.

  Yields configs sampled every `dt`, arc-length-following the waypoints: velocity
  ramps up from ~rest, cruises at `joint_rate`, and ramps back down over the last
  `ramp_dist` rad so the FINAL yielded sample lands on the goal config with v~=0
  (that boundary is where the task-event fires). Path geometry is untouched -- this
  is exactly what a jerk-limited tracking controller would do on hardware.

  `accel` (rad/s^2, optional): if given, use a proper acceleration-LIMITED velocity
  profile over the arc instead of the position-triggered ramp (see `_accel_profile`).
  Default None reproduces the interim behavior byte-for-byte.
  """
  qs = [np.asarray(q, float) for q in qs]
  if len(qs) < 2:
    return
  segs, D = arc_length(qs)
  if D < 1e-9:
    return
  csum = np.concatenate([[0.0], np.cumsum(segs)])
  if accel is not None:
    for s in _accel_profile(D, dt, joint_rate, accel):
      yield config_at(qs, csum, segs, s)
    return
  s = 0.0
  while s < D:
    ramp = min(1.0, (min(s, D - s) + 1e-6) / ramp_dist) if ramp_dist > 0 else 1.0
    s = min(D, s + joint_rate * max(ramp_min, ramp) * dt)
    yield config_at(qs, csum, segs, s)


def _accel_profile(D, dt, v_max, a_max):
  """Arc-length samples for an acceleration-limited trapezoid over [0, D].

  Symmetric ramp: accelerate at a_max to min(v_max, sqrt(a_max*D)) (triangular if the
  stretch is too short to reach v_max), cruise, decelerate to rest at D. v=0 at both
  ends by construction. Opt-in (accel!=None); NOT the default profile.
  """
  v_peak = min(v_max, float(np.sqrt(a_max * D)))
  t_ramp = v_peak / a_max
  d_ramp = 0.5 * a_max * t_ramp ** 2
  d_cruise = max(0.0, D - 2 * d_ramp)
  t_cruise = d_cruise / v_peak if v_peak > 1e-12 else 0.0
  T = 2 * t_ramp + t_cruise
  t = 0.0
  while t < T:
    t = min(T, t + dt)
    if t < t_ramp:
      s = 0.5 * a_max * t ** 2
    elif t < t_ramp + t_cruise:
      s = d_ramp + v_peak * (t - t_ramp)
    else:
      td = t - t_ramp - t_cruise
      s = d_ramp + d_cruise + v_peak * td - 0.5 * a_max * td ** 2
    yield min(D, s)


@dataclass
class TimedTrajectory:
  """A retimed, executable trajectory as portable data.

  q      : (M, dof) configs sampled `dt` apart.
  phase  : (M,) per-sample label -- "path" | "dwell" | "settle".
  fires  : list of (sample_index, [task_id, ...]) -- events fire AT that index, which
           by construction sits on a stretch goal at rest (after the pre-fire dwell).
  """
  dt: float
  q: np.ndarray
  phase: list = field(default_factory=list)
  fires: list = field(default_factory=list)

  def __len__(self):
    return len(self.q)


def build_trajectory(stretches, dt, joint_rate=DEFAULT_JOINT_RATE, ramp_dist=DEFAULT_RAMP_DIST,
                     ramp_min=DEFAULT_RAMP_MIN, dwell=0.0, settle=0.0, accel=None):
  """Stitch per-mode stretches into one `TimedTrajectory` with v=0 (+ optional dwell)
  at every boundary where events fire, and a final settle hold.

  stretches: [(waypoints, [done_task_id, ...]), ...] as from `split_stretches`.
  The fire index for stretch k is placed AFTER its pre-fire dwell, on the goal config
  at rest -- mirroring the sim's drive -> dwell -> fire -> dwell loop.
  """
  q, phase, fires = [], [], []
  n_dwell = int(round(dwell / dt)) if dt > 0 else 0
  goal = None
  for qs, events in stretches:
    for c in retime_stretch(qs, dt, joint_rate, ramp_dist, ramp_min, accel=accel):
      q.append(c)
      phase.append("path")
    goal = np.asarray(qs[-1], float) if len(qs) else goal
    for _ in range(n_dwell):                                 # settle to v~=0 before relinking
      q.append(goal)
      phase.append("dwell")
    fires.append((len(q) - 1, list(events)))                 # fire at rest, on the goal
    for _ in range(n_dwell):                                 # let the new attach take hold
      q.append(goal)
      phase.append("dwell")
  for _ in range(int(round(settle / dt)) if dt > 0 else 0):
    q.append(goal)
    phase.append("settle")
  arr = np.asarray(q, float) if q else np.zeros((0, 0))
  return TimedTrajectory(dt=dt, q=arr, phase=phase, fires=fires)


def split_stretches(states, q_of, mode_of, taskids_of):
  """Decompose a mode-tagged path into per-mode stretches at mode transitions.

  Kept free of mr_bench / rai types via accessor callables:
    q_of(state)      -> the full joint config for that path state (array-like)
    mode_of(state)   -> a mode handle, compared with != to detect a transition
    taskids_of(mode) -> iterable of task ids owned by that mode

  Returns [(waypoints, done_task_ids)], where a transition closes the current stretch
  and its `done_task_ids` are the tasks present in the OLD mode but not the NEW one
  (the events that complete at that boundary). The next stretch STARTS at the shared
  goal config (a duplicated transition waypoint), so consecutive stretches are joined.
  """
  states = list(states)
  if not states:
    return []
  mode = mode_of(states[0])
  cur = [np.asarray(q_of(states[0]), float)]
  out = []
  for st in states[1:]:
    m = mode_of(st)
    q = np.asarray(q_of(st), float)
    if m != mode:
      newset = {int(t) for t in taskids_of(m)}
      done = [int(t) for t in taskids_of(mode) if int(t) not in newset]
      out.append((cur, sorted(set(done))))
      cur = [cur[-1]]                                         # new stretch starts AT the goal
      mode = m
    cur.append(q)
  out.append((cur, sorted({int(t) for t in taskids_of(mode)})))
  return out
