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
DEFAULT_CORNER_DV = 0.2         # tolerated |Δ unit-direction|*speed when an arm CROSSES a barrier
                                # corner (the C0 path bends there); caps crossing speed by sharpness
                                # so a held part is not yanked. Larger => faster crossings.


def arc_length(qs):
  """Per-segment joint-space lengths and their total for a waypoint list."""
  segs = [float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))
          for a, b in zip(qs, qs[1:])]
  return segs, float(sum(segs))


def _seg_index(csum, segs, s):
  return min(int(np.searchsorted(csum, s, side="right")) - 1, len(segs) - 1)


def config_at(qs, csum, segs, s):
  """Linear-interpolate the waypoints at arc length `s` (0..total)."""
  i = _seg_index(csum, segs, s)
  if segs[i] < 1e-12:
    return np.asarray(qs[i + 1], float)
  a = (s - csum[i]) / segs[i]
  return (1 - a) * np.asarray(qs[i], float) + a * np.asarray(qs[i + 1], float)


def arm_rate_boost(qs, arm_dims, max_boost=None):
  """Per-segment rate multiplier that makes the BUSIEST arm travel at `joint_rate`.

  `joint_rate` is naturally applied to the COMPOSITE arc length, but the composite mixes
  every arm's joints: with k arms moving together each one's own joints advance at only
  |dq_arm|/|dq_composite| of the nominal rate (measured on car-mini: 0.61/0.51/0.41 --
  arms crawling at half their intended speed for a purely parametric reason). The boost
  1 / max_r(|dq_arm_r| / |dq|) restores the intended per-arm semantics and never exceeds
  it, since some arm always ends up exactly at `joint_rate`.

  SAFE WITHOUT ANY COLLISION CHECK when applied to a COMMON clock: it rescales time only,
  so the sequence of composite configurations -- and hence the planner's collision
  certificate -- is completely unchanged. (Per-ARM independent rates are a different
  thing and do need the oracle; see plan_schedule's `collide`.) The boost is bounded by
  sqrt(#arms), so it cannot run away; `max_boost` clamps it further if given.

  Returns an array aligned with the segments of `qs` (multiplier >= 1)."""
  qs = [np.asarray(q, float) for q in qs]
  d = np.diff(np.asarray(qs, float), axis=0)
  tot = np.linalg.norm(d, axis=1)
  share = np.zeros(len(d))
  for dims in arm_dims:
    share = np.maximum(share, np.linalg.norm(d[:, np.asarray(dims, int)], axis=1))
  boost = np.ones(len(d))
  ok = (tot > 1e-12) & (share > 1e-12)
  boost[ok] = tot[ok] / share[ok]
  cap = max_boost if max_boost is not None else float(np.sqrt(max(len(arm_dims), 1)))
  return np.clip(boost, 1.0, cap)


def joint_limit_boost(qs, vmax, joint_rate=DEFAULT_JOINT_RATE, max_boost=None):
  """Per-segment rate multiplier that puts the BINDING JOINT exactly on its velocity limit.

  `arm_rate_boost` normalises to a single nominal `joint_rate` on the composite arc; this uses
  the robot's REAL per-joint limits (rai reports them: UR10 = 2.16 rad/s pan/lift, 3.15 elbow,
  3.20 wrists, against a nominal 1.0). Along a segment a joint advances at
  (|dq_j| / |dq|) * rate, so the rate that saturates exactly one joint is

      rate = 1 / max_j( |dq_j| / (|dq| * vmax_j) )

  and the multiplier returned is that over `joint_rate`. Exactly one joint ends up at its limit
  and none exceed it, which is the "at least one robot at its velocity limit" semantics.

  SAFE WITHOUT ANY COLLISION CHECK on a COMMON clock, for the same reason as arm_rate_boost: it
  rescales TIME only, so the sequence of composite configurations -- and hence the planner's
  certificate -- is unchanged. It does NOT bound the resulting grasp wrenches or the tracking
  error, both of which grow with speed: compose with the --force-limits per-segment vmax and
  re-measure the deviation envelope before trusting the faster schedule.

  Returns an array aligned with the segments of `qs` (multiplier >= 0)."""
  qs = [np.asarray(q, float) for q in qs]
  d = np.diff(np.asarray(qs, float), axis=0)
  tot = np.linalg.norm(d, axis=1)
  v = np.asarray(vmax, dtype=float).ravel()
  v = np.where(np.isfinite(v) & (v > 0), v, np.inf)          # unknown limit -> never binds
  with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.max(np.abs(d) / v[None, :], axis=1)           # per segment: max_j |dq_j|/vmax_j
  boost = np.ones(len(d))
  ok = (tot > 1e-12) & (ratio > 1e-12)
  boost[ok] = (tot[ok] / ratio[ok]) / float(joint_rate)      # rate_seg / joint_rate
  if max_boost is not None:
    boost = np.minimum(boost, float(max_boost))
  return np.maximum(boost, 0.0)


def retime_stretch(qs, dt, joint_rate=DEFAULT_JOINT_RATE, ramp_dist=DEFAULT_RAMP_DIST,
                   ramp_min=DEFAULT_RAMP_MIN, accel=None, arm_dims=None, vmax=None):
  """Trapezoidal time-parameterization of ONE collision-free stretch of waypoints.

  Yields configs sampled every `dt`, arc-length-following the waypoints: velocity
  ramps up from ~rest, cruises at `joint_rate`, and ramps back down over the last
  `ramp_dist` rad so the FINAL yielded sample lands on the goal config with v~=0
  (that boundary is where the task-event fires). Path geometry is untouched -- this
  is exactly what a jerk-limited tracking controller would do on hardware.

  `accel` (rad/s^2, optional): if given, use a proper acceleration-LIMITED velocity
  profile over the arc instead of the position-triggered ramp (see `_accel_profile`).
  Default None reproduces the interim behavior byte-for-byte.

  `arm_dims` (optional): apply `arm_rate_boost` so the busiest ARM runs at `joint_rate`
  instead of the composite arc doing so. Time-only rescaling of a common clock -- the
  geometric path, and thus the collision certificate, is unchanged. None = off.
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
  # `vmax` (--limit-rate) takes precedence over `arm_dims` (--arm-rate): both are per-segment
  # rate multipliers on a common clock, but the per-JOINT one is strictly the better-founded
  # -- it saturates a real limit instead of a nominal constant.
  if vmax is not None:
    boost = joint_limit_boost(qs, vmax, joint_rate=joint_rate)
  elif arm_dims is not None:
    boost = arm_rate_boost(qs, arm_dims)
  else:
    boost = None
  s = 0.0
  while s < D:
    ramp = min(1.0, (min(s, D - s) + 1e-6) / ramp_dist) if ramp_dist > 0 else 1.0
    rate = joint_rate if boost is None else joint_rate * boost[_seg_index(csum, segs, s)]
    s = min(D, s + rate * max(ramp_min, ramp) * dt)
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
  progress: object = None       # (M, n_arms) per-arm arc position, when the producer tracks it
  couplings: list = field(default_factory=list)  # local (arms, s0, s1) equality windows, if any

  def __len__(self):
    return len(self.q)


def upsample_trajectory(traj, dt, q0=None):
  """Densify a ``TimedTrajectory`` by an integer factor without changing its timing/events.

  Coordination-space scheduling does not need to run at the 2ms physics-control rate: its
  exact predicate already uses the motion planner's ~0.01rad edge scale.  Plan coarsely, then
  use this helper to restore the controller rate and audit EVERY dense sample.  If that audit
  rejects an interpolated state, the caller retries at a finer planning rate or falls back;
  densification itself is therefore never treated as a collision certificate.

  ``q0`` is the configuration at t=0.  ``traj.q[0]`` is conventionally the configuration
  after the first source tick, so supplying q0 preserves that first interval as well.
  """
  dt = float(dt)
  if dt <= 0.0 or traj.dt <= 0.0:
    raise ValueError("trajectory sample times must be positive")
  ratio = float(traj.dt) / dt
  factor = int(round(ratio))
  if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1e-9):
    raise ValueError(f"source dt {traj.dt:g} is not an integer multiple of target dt {dt:g}")
  if factor == 1:
    return traj
  src = np.asarray(traj.q, dtype=float)
  if not len(src):
    return TimedTrajectory(dt=dt, q=src.copy(), phase=[], fires=[], progress=None,
                           couplings=list(traj.couplings))
  prev = np.asarray(src[0] if q0 is None else q0, dtype=float)
  out, phase = [], []
  pout = [] if traj.progress is not None else None
  pprev = (np.zeros(traj.progress.shape[1], dtype=float) if traj.progress is not None
           else None)
  for i, goal in enumerate(src):
    pg = traj.progress[i] if traj.progress is not None else None
    for j in range(1, factor + 1):
      a = j / factor
      out.append(prev + a * (goal - prev))
      phase.append(traj.phase[i] if i < len(traj.phase) else "path")
      if pout is not None:
        pout.append(pprev + a * (pg - pprev))
    prev = goal
    if pout is not None:
      pprev = pg
  fires = [((int(i) + 1) * factor - 1, list(ev)) for i, ev in traj.fires]
  return TimedTrajectory(dt=dt, q=np.asarray(out, dtype=float), phase=phase, fires=fires,
                         progress=(np.asarray(pout, dtype=float) if pout is not None else None),
                         couplings=list(traj.couplings))


def build_trajectory(stretches, dt, joint_rate=DEFAULT_JOINT_RATE, ramp_dist=DEFAULT_RAMP_DIST,
                     ramp_min=DEFAULT_RAMP_MIN, dwell=0.0, settle=0.0, accel=None,
                     arm_dims=None, vmax=None, event_durations=None):
  """Stitch per-mode stretches into one `TimedTrajectory` with v=0 (+ optional dwell)
  at every boundary where events fire, and a final settle hold.

  stretches: [(waypoints, [done_task_id, ...]), ...] as from `split_stretches`.
  The fire index for stretch k is placed AFTER its pre-fire dwell, on the goal config
  at rest -- mirroring the sim's drive -> dwell -> fire -> dwell loop.
  """
  q, phase, fires = [], [], []
  n_dwell = int(round(dwell / dt)) if dt > 0 else 0
  if event_durations is None:
    n_event = [0] * len(stretches)
  else:
    _event = np.asarray(event_durations, dtype=float).reshape(-1)
    if len(_event) != len(stretches):
      raise ValueError(
          f"event_durations has {len(_event)} entries for {len(stretches)} events")
    if np.any(_event < 0.0):
      raise ValueError("event_durations must be non-negative")
    n_event = (np.rint(_event / dt).astype(int).tolist() if dt > 0
               else [0] * len(stretches))
  goal = None
  for k, (qs, events) in enumerate(stretches):
    for c in retime_stretch(qs, dt, joint_rate, ramp_dist, ramp_min, accel=accel,
                            arm_dims=arm_dims, vmax=vmax):
      q.append(c)
      phase.append("path")
    goal = np.asarray(qs[-1], float) if len(qs) else goal
    for _ in range(n_dwell + n_event[k]):                    # nominal servo + settle before fire
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


def _fill_shape2(n_steps, ramp_in, ramp_out):
  """Normalized increments (sum = 1) that ramp from rest only where asked. `ramp_in`/
  `ramp_out` = 0 means "enter/leave this span at full speed" -- used to flow through a
  barrier an arm merely crosses, while still coming to rest at its own events."""
  n_steps = max(int(n_steps), 1)
  w = np.ones(n_steps)
  ri = int(max(0, min(ramp_in, n_steps // 2)))
  ro = int(max(0, min(ramp_out, n_steps // 2)))
  if ri:
    w[:ri] = (np.arange(ri) + 1.0) / (ri + 1.0)
  if ro:
    w[-ro:] = ((np.arange(ro) + 1.0) / (ro + 1.0))[::-1]
  tot = float(w.sum())
  return w / tot if tot > 0 else w


def _fill_shape(n_steps, ramp_steps):
  """Normalized per-step increments (sum = 1) shaped as ramp-up / flat / ramp-down in TIME.
  Used to spread a fixed arc over a fixed duration: continuous motion, zero velocity at both
  ends, no rush-then-wait."""
  n_steps = max(int(n_steps), 1)
  rs = int(max(1, min(ramp_steps, n_steps // 2)))
  w = np.ones(n_steps)
  ramp = (np.arange(rs) + 1.0) / (rs + 1.0)
  w[:rs] = ramp
  w[-rs:] = ramp[::-1]
  tot = float(w.sum())
  return w / tot if tot > 0 else w


def spread_between_events(traj, arm_dims, bar_s, acting, dt, joint_rate,
                          ramp_dist=DEFAULT_RAMP_DIST, P=None, csum=None, segs=None,
                          collide=None, fired_of=None):
  """Remove RUSH-THEN-WAIT: re-time each arm to spread its motion CONTINUOUSLY over the
  whole interval between its own consecutive events, instead of sprinting to the next
  barrier and parking there until the event fires.

  This is the "un-split" of the retiming: an arm's profile is laid across a whole
  event-to-event span rather than rebuilt inside every per-mode stretch, so it decelerates
  only where it genuinely must be at rest -- at its OWN events. Event sample indices are
  taken from `traj` and are NOT changed, so the firing schedule, its order, and every
  arm's arrival time stay exactly as scheduled; only the path *between* arrivals is
  re-paced (slower and continuous instead of fast-then-idle). An arm that was already at
  full speed for the whole interval cannot be spread and is left untouched.

  If `collide` is given it is checked per sample (with `fired_of(i)` giving the events
  fired by sample i); any interval that would violate keeps its original timing, so this
  can never introduce a conflict the schedule did not already accept.

  Returns a new (M, dof) config array."""
  q = np.array(traj.q, dtype=float, copy=True)
  prog = traj.progress
  if prog is None or len(q) < 3:
    return q
  M, n = len(q), len(arm_dims)
  ramp_steps = int(round((ramp_dist / max(joint_rate, 1e-9)) / max(dt, 1e-12)))
  for r in range(n):
    anchors = [(0, float(prog[0, r]))]
    for k, (idx, _) in enumerate(traj.fires):
      if k < len(acting) and r in acting[k] and 0 < idx < M:
        anchors.append((int(idx), float(bar_s[k])))
    anchors.append((M - 1, float(prog[-1, r])))
    for (i0, s0), (i1, s1) in zip(anchors, anchors[1:]):
      n_steps, L = i1 - i0, s1 - s0
      if n_steps < 3 or L <= 1e-9:
        continue
      if L / (n_steps * dt) > joint_rate * 0.98:
        continue                                             # already at full speed: nothing to spread
      inc = _fill_shape(n_steps, ramp_steps) * L
      s_new = s0 + np.cumsum(inc)
      block = np.array([config_at(P, csum, segs, sv)[arm_dims[r]] for sv in s_new])
      if collide is not None:
        bad = False
        for m in range(n_steps):
          cand = q[i0 + 1 + m].copy()
          cand[arm_dims[r]] = block[m]
          res = collide(cand, fired_of(i0 + 1 + m) if fired_of else 0)
          if res or res == set():
            bad = True
            break
        if bad:
          continue                                           # keep the scheduled timing here
      q[i0 + 1:i1 + 1, arm_dims[r]] = block
  return q


def motion_report(q, dt, arm_dims, joint_rate, crawl=0.5, stopped=0.05):
  """Per-arm MOTION QUALITY of a retimed trajectory -- the metric that matters for
  execution: do the arms move smoothly, or do they keep crawling/stalling?

  For each arm's ACTIVE interval (first motion through its own final arrival): the fraction
  of samples spent STOPPED (<stopped x cruise), CRAWLING
  (stopped..crawl x cruise) and running near cruise, plus the number of separate SLOW
  EPISODES (contiguous below-crawl runs bracketed by near-cruise motion) -- an arm that
  decelerates 20 times over a run reads as "randomly slowing down" no matter what the
  makespan says. Stops at an arm's own task goals are legitimate; the episode count is
  meaningful against that baseline, so compare prioritized vs lockstep rather than
  reading it absolutely. Returns [{arm, stopped, crawling, cruising, episodes, mean}].

  The terminal idle tail is reported separately.  It is not a visual defect: terminal
  completion is intentionally asynchronous, so a robot that reaches home at a reasonable
  speed may remain there while another finishes rather than creeping to manufacture a
  synchronized final instant.
  """
  q = np.asarray(q, float)
  out = []
  if len(q) < 2:
    return out
  for r, dims in enumerate(arm_dims):
    dims = np.asarray(dims, int)
    v = np.linalg.norm(np.diff(q[:, dims], axis=0), axis=1) / max(dt, 1e-12)
    frac_all = v / max(joint_rate, 1e-12)
    moving = np.flatnonzero(frac_all >= stopped)
    if len(moving):
      lo, hi = int(moving[0]), int(moving[-1]) + 1
      frac = frac_all[lo:hi]
      terminal_idle = max(0, len(frac_all) - hi) * dt
    else:
      frac = frac_all
      terminal_idle = len(frac_all) * dt
    is_stop = frac < stopped
    is_slow = frac < crawl
    eps_n, prev_fast = 0, True
    for slow in is_slow:
      if slow and prev_fast:
        eps_n += 1
      prev_fast = not slow
    out.append({"arm": r,
                "stopped": float(np.mean(is_stop)),
                "crawling": float(np.mean(is_slow & ~is_stop)),
                "cruising": float(np.mean(~is_slow)),
                "episodes": eps_n,
                "mean": float(np.mean(frac)),
                "terminal_idle": float(terminal_idle)})
  return out


def format_motion_report(rows, label=""):
  """One compact line per arm from `motion_report` (see it for the metric definitions)."""
  head = f"[retiming] motion quality{' ' + label if label else ''}:"
  body = ["  arm%d: cruise %4.0f%%  crawl %4.0f%%  stopped %4.0f%%  "
          "slow-episodes %3d  mean %.2f x cruise  terminal-idle %.1fs"
          % (d["arm"], 100 * d["cruising"], 100 * d["crawling"], 100 * d["stopped"],
             d["episodes"], d["mean"], d["terminal_idle"]) for d in rows]
  return "\n".join([head] + body)


def _composite(stretches):
  """Reconstruct the single composite waypoint array + the barrier waypoint indices
  from per-mode stretches (consecutive stretches share their boundary config)."""
  P = [np.asarray(q, float) for q in stretches[0][0]]
  bidx = [len(P) - 1]                                        # end-of-stretch waypoint indices
  for wp, _ in stretches[1:]:
    P.extend(np.asarray(q, float) for q in wp[1:])           # skip the duplicated boundary config
    bidx.append(len(P) - 1)
  return np.asarray(P, float), bidx


def _coupling_topology(couplings, s, n, D, eps=1e-6):
  """Return rendezvous caps and fused components for live pairwise coupling windows.

  Before both members reach a window's entry ``a``, *both* are capped at ``a``. Capping only
  the member whose partner has not arrived lets the laggard overshoot the rendezvous in one
  per-arm-rate step (especially when it skips a zero-motion composite span). The pair then
  remains unequal inside a window that requires equality and can deadlock despite the certified
  diagonal being feasible.
  """
  entry = {}
  adj = {r: set() for r in range(n)}
  for G, wins in couplings.items():
    i, j = sorted(G)
    live = next(((a, b) for a, b in wins if min(s[i], s[j]) < b - eps), None)
    if live is None:
      continue
    a, _ = live
    ia, ja = s[i] >= a - eps, s[j] >= a - eps
    if ia and ja:
      adj[i].add(j)
      adj[j].add(i)
    else:
      # Rendezvous semantics: neither participant may cross the entry before the other. The
      # already-arrived member waits there and the laggard lands there exactly on this tick.
      entry[i] = min(entry.get(i, D), a)
      entry[j] = min(entry.get(j, D), a)
  comps, seen = [], set()
  for r in range(n):
    if r in seen or not adj[r]:
      continue
    comp, todo = set(), [r]
    while todo:
      x = todo.pop()
      if x in comp:
        continue
      comp.add(x)
      todo.extend(adj[x] - comp)
    seen |= comp
    comps.append(sorted(comp))
  return entry, comps


def plan_schedule(stretches, arm_dims, acting, dt,
                  joint_rate=DEFAULT_JOINT_RATE, ramp_dist=DEFAULT_RAMP_DIST,
                  ramp_min=DEFAULT_RAMP_MIN, corner_dv=DEFAULT_CORNER_DV,
                  priority=None, collide=None, dwell=0.0, collide_ds=0.01,
                  settle=0.0, max_steps=None, stall_steps=None, rate_boost=False,
                  event_durations=None, per_arm_rate=False):
  """PRIORITIZED, EVENT-DRIVEN retiming: arms desync so only the arm(s) ACTING at a
  barrier ramp to rest there; uninvolved arms CROSS the barrier without stopping.
  Events fire in DAG/sequence order, each once its acting arm(s) have arrived at rest
  (and the previous event fired). Portable: pure numpy, geometry never touched.

    stretches : [(composite waypoints, done_task_ids)] from `split_stretches`.
    arm_dims  : list of column-index arrays into the full config, one per arm (the
                partition of the joints; arm r's config columns are full_q[arm_dims[r]]).
    acting    : list aligned with `stretches` -- acting[k] = set of ARM INDICES that
                must be at rest at barrier k (the arms performing that barrier's event,
                INCLUDING holders of any part the event touches).
    priority  : tie-break order for arms at EQUAL progress (default = arm index).
                Arms are otherwise processed LAGGARD-FIRST each tick (ascending s) --
                see the backpressure rule under `collide`.
    collide   : callable(full_q, fired) -- collision verdict for that composite config
                given the world AFTER `fired` events (attachments are a function of
                events fired, not of arm positions). Return False/None if free; if
                colliding, return True OR the set of ARM INDICES the mover is in
                conflict with (richer = better coordination). EVERY arm's candidate
                step is checked; a blocked arm HOLDS -- pass-through conflicts clear
                when the blocking arm moves on, world-evolution conflicts (a part
                still sitting in the lane) when the responsible event fires. Genuine
                deadlock (nothing can advance) raises RuntimeError -- never silent
                truncation. None => no avoidance (desync blindly; the physics sim is
                the judge).
    dwell     : seconds each event's acting arms REST at their barrier before the fire
                (held part settles) and again after it (weld takes hold). Non-acting
                arms keep moving throughout -- this replaces the composite executor's
                all-arms dwell freeze.
    event_durations: optional seconds reserved BEFORE each event in addition to `dwell`.
                This is the nominal duration of the event's servo/seat operation, not a
                measured runtime correction.  Reserving it in the offline schedule lets
                crossing arms use that time without a reactive re-timing loop.  A runtime
                overrun may still hold at the event; it never changes this planned profile.
    per_arm_rate: interpret `joint_rate` in each arm's OWN joint-space arc instead of in the
                composite all-arm arc. A phase span on which an arm is physically stationary
                is bookkeeping and is skipped immediately; a moving arm realizes approximately
                `joint_rate` regardless of how many other arms move in that composite segment.
                Every resulting desynchronized configuration still goes through `collide`.
    collide_ds: arc-length cell (rad) for caching collide verdicts -- one query per
                (mover, arm-cells, fired) cell, not per dt step. Verdicts are evaluated
                with EVERY arm at its forward cell edge (clamped to its cap), so a safe
                verdict covers all real positions until some cell index changes -- no
                stale-cache drift into collision (measured: 0.02-cells with mover-only
                quantization let two arms walk 10mm into each other).

  Returns a TimedTrajectory of full-config samples (arms at desynced progress) with
  `fires` marking, at the sample where each event completes, its done_task_ids. When
  `stretches` has a single arm or a single mode this reduces to the trapezoid of
  `build_trajectory` (no crossing to exploit)."""
  if not stretches:
    return TimedTrajectory(dt=dt, q=np.zeros((0, 0)))
  P, bidx = _composite(stretches)
  dof = P.shape[1]
  segs, D = arc_length(list(P))
  csum = np.concatenate([[0.0], np.cumsum(segs)]) if segs else np.array([0.0])
  n = len(arm_dims)
  order = list(priority) if priority is not None else list(range(n))
  bar_s = [csum[bi] for bi in bidx]                          # barrier positions on the arc
  K = len(stretches)
  done_ids = [sorted({int(t) for t in st[1]}) for st in stretches]
  act = [set(int(a) for a in acting[k]) for k in range(K)]

  # per-arm SPEED-LIMIT POINTS: (arc position, floor speed fraction). Start/end and each acting
  # barrier are near-stops (floor=ramp_min). A CROSSED barrier is a velocity-discontinuity corner
  # of the C0 path (per-mode segments stitched), so crossing it at speed yanks a held part; we
  # floor the crossing speed by corner sharpness (straight => 1, hard turn => ~ramp_min).
  def corner_floor(r, k):
    if r in act[k] or corner_dv is None or corner_dv <= 0:
      return ramp_min                                        # acting here (or disabled): near-stop
    bi = bidx[k]
    if bi <= 0 or bi >= len(P) - 1:
      return ramp_min
    din = P[bi][arm_dims[r]] - P[bi - 1][arm_dims[r]]
    dout = P[bi + 1][arm_dims[r]] - P[bi][arm_dims[r]]
    ni, no = float(np.linalg.norm(din)), float(np.linalg.norm(dout))
    if ni < 1e-9 or no < 1e-9:
      return 1.0                                             # stationary one side: not a corner
    dv = float(np.linalg.norm(dout / no - din / ni))         # |Δ unit direction|, in [0, 2]
    return float(np.clip(corner_dv / max(dv, 1e-9), ramp_min, 1.0))

  slim = []
  for r in range(n):
    d = {0.0: ramp_min, D: ramp_min}
    for k in range(K):
      d[bar_s[k]] = min(d.get(bar_s[k], 1.0), corner_floor(r, k))
    slim.append(sorted(d.items()))

  def config_full(s):
    return config_at(P, csum, segs, s)

  def assemble(svec, override=None):
    fq = np.empty(dof)
    for r in range(n):
      sr = override[1] if (override is not None and r == override[0]) else svec[r]
      fq[arm_dims[r]] = config_full(sr)[arm_dims[r]]
    return fq

  # rate boost so the busiest arm runs at joint_rate (see arm_rate_boost). Indexed by the
  # COMPOSITE arc, i.e. the same multiplier a common clock would get -- a time-only
  # rescaling, no new configurations, so it needs no extra collision reasoning of its own.
  _boost = arm_rate_boost(list(P), arm_dims) if rate_boost else None

  # Per-arm share of every COMPOSITE segment.  A 1rad/s composite clock makes an arm whose
  # share is 0.1 crawl at 0.1rad/s, and worse, makes a completely stationary arm spend real
  # seconds traversing invisible phase before an event may fire (chair place: 9.5s with every
  # physical joint stopped). Independent phase planning can normalize this safely because its
  # candidates are collision checked. Zero-share runs are skipped; they change no geometry.
  _d = np.diff(P, axis=0)
  _tot = np.linalg.norm(_d, axis=1)
  _share = np.ones((n, max(len(_d), 1)))
  for _r in range(n):
    _sr = np.linalg.norm(_d[:, np.asarray(arm_dims[_r], int)], axis=1)
    _ok = _tot > 1e-12
    _share[_r][_ok] = _sr[_ok] / _tot[_ok]

  def rate_at(r, s):
    i = _seg_index(csum, segs, s)
    base = joint_rate if _boost is None else joint_rate * _boost[i]
    if not per_arm_rate:
      return base
    sh = _share[r][i]
    return np.inf if sh <= 1e-9 else base / sh

  def ramp_factor(r, s):
    if ramp_dist <= 0:
      return 1.0
    pts = slim[r]
    pos = [p for p, _ in pts]
    i = int(np.searchsorted(pos, s, side="right"))
    pprev, fprev = pts[i - 1] if i > 0 else (0.0, ramp_min)
    pnext, fnext = pts[i] if i < len(pts) else (D, ramp_min)
    up = max(fprev, min(1.0, (s - pprev + 1e-6) / ramp_dist))      # ramp up from prev floor
    down = max(fnext, min(1.0, (pnext - s + 1e-6) / ramp_dist))    # ramp down to next floor
    return min(up, down)

  def cap_of(r, fired):
    for k in range(fired, K):                                # r may not pass its next unfired mate
      if r in act[k]:
        return bar_s[k]
    return D

  def step_to(r, s, cap):
    """One phase tick, skipping physically idle bookkeeping spans.

    Moving steps stop at composite waypoint boundaries. This keeps the local share/rate exact
    and prevents a high normalized phase rate from jumping across a corner. Idle spans may be
    crossed in one call because arm r's configuration is bit-identical throughout them.
    """
    cur = float(s)
    if per_arm_rate:
      while cur < cap - 1e-12:
        i = _seg_index(csum, segs, cur)
        if _share[r][i] > 1e-9:
          break
        nxt = min(float(csum[i + 1]), float(cap))
        if nxt <= cur + 1e-12:
          break
        cur = nxt
    if cur >= cap - 1e-12:
      return min(cur, cap)
    i = _seg_index(csum, segs, cur)
    nxt = min(float(csum[i + 1]), float(cap)) if per_arm_rate else float(cap)
    return min(nxt, cur + rate_at(r, cur) * ramp_factor(r, cur) * dt)

  if max_steps is None:                                      # generous: n arms each may wait out
    full = D / max(joint_rate * dt, 1e-9)                    # the others (10x margin for ramps/holds)
    max_steps = int(10 * n * full) + 10000
  if stall_steps is None:
    stall_steps = max(int(5.0 / dt), 1000) if dt > 0 else 1000
  n_dwell = int(round(dwell / dt)) if dt > 0 else 0
  if event_durations is None:
    n_event = np.zeros(K, dtype=int)
  else:
    _event = np.asarray(event_durations, dtype=float).reshape(-1)
    if len(_event) != K:
      raise ValueError(f"event_durations has {len(_event)} entries for {K} events")
    if np.any(_event < 0.0):
      raise ValueError("event_durations must be non-negative")
    n_event = np.rint(_event / dt).astype(int) if dt > 0 else np.zeros(K, dtype=int)
  eps = 1e-6
  import os as _os
  _trace = bool(_os.environ.get("RETIMING_TRACE"))

  # CEGAR-STYLE COORDINATION (replaces an online wedge-escape state machine that was
  # measured into livelocks): run the greedy desync schedule; when it WEDGES (a mutual
  # coordination-space pocket: every step was individually free, now no forward move is),
  # infer a PAIRWISE COUPLING -- "arms i,j are FUSED over arc span [a, b]" -- and RESTART
  # with it. FUSED means IDENTICAL progress: members wait at `a` for each other, then move
  # as one group along the composite diagonal (the certified formation, zero intra-group
  # offset -- a mere proximity band was measured insufficient: at a 3-arm formation even a
  # 0.05-rad residual spread wedged, since any single move shrinks one pairwise offset but
  # grows another). Tight zones thereby run certified lockstep exactly where evidence
  # demands it; free zones keep full desync; motion stays monotone; and the worst case
  # (all arms fused over the full span) IS the certified composite -- so the restart loop
  # terminates with a valid schedule. Restarts are cheap: the verdict cache is keyed on
  # positions+fired, so replays are cache hits.
  WEDGE_TICKS = 250                                          # no-progress ticks => wedged
  MAX_RESTARTS = 60   # disjoint windows => more, smaller couplings than one merged span
  REWIND = 0.5                                               # couple from this far before wedge
  couplings = {}                                             # frozenset{i,j} -> [a, b]
  cache = {}                                                 # (mover, cells, fired) -> verdict

  def _verdict(res, r):
    if not res and res != set():                             # falsy non-set => free
      return None
    if isinstance(res, (set, frozenset, list, tuple)):       # blocker arm indices (may be empty)
      return frozenset(int(j) for j in res) - {r}
    return frozenset()                                       # bare truthy (incl. np.bool_)

  class _Wedge(Exception):
    def __init__(self, r, j, sr, sj):
      self.r, self.j, self.sr, self.sj = r, j, sr, sj

  def run_attempt():
    s = [0.0] * n
    fired = 0
    q, phase, fires, prog = [], [], [], []
    steps = stall = wedge = 0
    ready_since = None                                       # step when act[fired] all arrived
    hold_until = [-1] * n                                    # post-fire rest, per acting arm

    def pair_topology():
      """Live couplings -> (entry_cap per arm, fused components).
      Each pair owns a list of DISJOINT windows; the first one the pair has not both
      passed is the live one. A member below `a` runs free but is CAPPED AT `a` until its
      partner arrives; once both are >= a they are FUSED and move only as a group
      (identical progress). Fused pairs sharing an arm merge into one component."""
      return _coupling_topology(couplings, s, n, D, eps)

    def blocked(r, s_new, cap):
      """None if arm r may take this step; else the (possibly empty) set of arm indices
      blocking it -- empty = static world (waiting on `fired` clears it). TWO TIERS:
      (1) CACHED conservative verdict with every arm at its forward cell edge (clamped to
      its cap) -- a free verdict covers all real positions until some cell index changes,
      so no stale-cache drift into collision (measured 10mm on car-mini with mover-only
      quantization). (2) If tier 1 says blocked, an EXACT uncached check of the true
      candidate -- accepted steps are verified fresh, so the ~2-cell inflation margin
      cannot wedge an arm at a tight-corridor frontier (measured: corridor test)."""
      if collide is None:
        return None
      ds = max(collide_ds, 1e-9)
      cell = int(s_new / ds)
      key = (r, cell, tuple(int(sv / ds) for sv in s), fired)
      v = cache.get(key)
      if v is None:
        s_chk = [min((int(s[j] / ds) + 1) * ds, cap_of(j, fired), D) for j in range(n)]
        s_chk[r] = min((cell + 1) * ds, cap, D)
        v = _verdict(collide(assemble(s_chk), fired), r)
        cache[key] = (v,)
      else:
        v = v[0]
      if v is None:
        return None                                          # tier 1 free: sound, cached
      return _verdict(collide(assemble(s, override=(r, s_new)), fired), r)

    while True:
      if steps >= max_steps:
        raise RuntimeError(
            f"[retiming] schedule exceeded max_steps={max_steps} (fired {fired}/{K}, "
            f"progress {[round(x, 2) for x in s]} of {D:.2f})")
      steps += 1
      progressed = False
      entry, comps = pair_topology()
      fused = {r for c in comps for r in c}
      # INDIVIDUAL movers, LAGGARD-FIRST (ascending s, `priority` breaks ties). A blocked
      # arm HOLDS: pass-through conflicts clear when the blocking arm moves on, static-
      # world conflicts when the responsible event fires, mutual pockets -> _Wedge/restart.
      seq = sorted(order, key=lambda r: (s[r], order.index(r)))
      for r in seq:
        if r in fused or s[r] >= D - eps or steps <= hold_until[r]:
          continue
        cap = min(cap_of(r, fired), entry.get(r, D))         # event cap AND coupling entry
        if s[r] >= cap - eps:
          continue                                           # parked: barrier or span entry
        s_new = step_to(r, s[r], cap)
        if s_new > s[r] + eps * dt and blocked(r, s_new, cap) is not None:
          continue                                           # yield: hold, retry next tick
        progressed = progressed or s_new > s[r]
        s[r] = s_new
      # FUSED groups: identical progress along the composite diagonal (the certified
      # formation -- intra-group needs no oracle; ONE exact check vs the outside arms).
      for comp in comps:
        if any(steps <= hold_until[r] for r in comp):
          continue                                           # a member rests through a dwell
        cap = min(min(cap_of(r, fired), entry.get(r, D)) for r in comp)
        base = max(s[r] for r in comp)                       # equal by construction; be safe
        g_new = min(step_to(r, base, cap) for r in comp)
        if g_new <= base + eps * dt:
          continue                                           # parked at a barrier inside span
        s_try = list(s)
        for r in comp:
          s_try[r] = g_new
        if _verdict(collide(assemble(s_try), fired), -1) is None:
          for r in comp:
            progressed = progressed or g_new > s[r]
            s[r] = g_new
      if not progressed and collide is not None:
        # COUPLED-STEP ESCAPE: singles all blocked, but an (un-fused) tight formation may
        # only be traversable by moving TOGETHER. One exact-checked joint step; often
        # avoids a wedge->restart round-trip.
        s_cand, any_move = list(s), False
        for r in range(n):
          if r in fused or s[r] >= D - eps or steps <= hold_until[r]:
            continue
          cap = min(cap_of(r, fired), entry.get(r, D))
          if s[r] >= cap - eps:
            continue
          s_cand[r] = step_to(r, s[r], cap)
          any_move = any_move or s_cand[r] > s[r] + eps * dt
        if any_move and _verdict(collide(assemble(s_cand), fired), -1) is None:
          s = s_cand
          progressed = True
      q.append(assemble(s))
      prog.append(list(s))                                   # per-arm arc position (diagnostics)
      phase.append("path")
      # FIRE GATE: acting arms at rest on the barrier AND every arm PAST the barrier arc.
      # The global-passage condition keeps the certificate valid for laggards: an event
      # that fires while an arm is still behind its arc imposes the post-event world on a
      # lane certified pre-event -- a placement can then block that lane FOREVER (measured
      # on cross8: fired=32 with arm0 at 259<bar, silently static-blocked). With the gate,
      # the minimum-progress arm always sees exactly the composite's world at its position
      # (certified => it always advances), and ahead-arms only wait on blocks that CLEAR.
      # Crossing arms never stop for this -- only the fire waits for them to pass.
      if fired < K and all(s[r] >= bar_s[fired] - eps for r in act[fired]) \
          and min(s) >= bar_s[fired] - eps:
        if ready_since is None:
          ready_since = steps                                # pre-fire dwell: settle at rest
        # The servo duration is feed-forward schedule data.  It is deliberately not learned
        # or adjusted from this execution: normal runs therefore keep the same phase profile,
        # while an exceptional servo overrun is handled later by the executor's safety hold.
        if steps - ready_since >= n_dwell + n_event[fired]:
          fires.append((len(q) - 1, done_ids[fired]))        # event completes, at rest, in order
          for r in act[fired]:
            hold_until[r] = steps + n_dwell                  # post-fire dwell: weld takes hold
          fired += 1
          ready_since = None
          progressed = True
          if _trace:
            print(f"[retiming trace] t={steps} FIRE {fired - 1} "
                  f"s={[round(x, 2) for x in s]}")
      else:
        ready_since = None
      if fired >= K and all(sr >= D - eps for sr in s):
        return q, phase, fires, prog
      waiting = ready_since is not None or any(h > steps for h in hold_until)
      if progressed or waiting:
        wedge = stall = 0
        continue
      wedge += 1
      stall += 1
      if wedge > WEDGE_TICKS and collide is not None:
        # WEDGED: find a blocked arm with a concrete arm blocker -> couple that pair.
        # Couple with the MOST DESYNCHRONIZED blocker: a blocker already fused/synced with
        # r cannot be the real conflict (measured: closest-in-s looped 14 restarts widening
        # a 0-2 coupling while the true blocker was arm1, off-formation).
        for r in sorted(range(n), key=lambda r: s[r]):
          if r in fused or s[r] >= D - eps or steps <= hold_until[r]:
            continue
          cap = min(cap_of(r, fired), entry.get(r, D))
          if s[r] >= cap - eps:
            continue
          s_new = step_to(r, s[r], cap)
          who = blocked(r, s_new, cap)
          if who is not None and _trace:
            print(f"[retiming trace] wedge scan: arm{r}@{s[r]:.3f} blocked by "
                  f"{sorted(who) if who else 'STATIC'} pairs="
                  f"{getattr(collide, 'last_pairs', '?')} fired={fired} "
                  f"s={[round(x, 2) for x in s]}")
          if who:
            j = max(who, key=lambda j: abs(s[j] - s[r]))
            raise _Wedge(r, j, s[r], s[j])
        for comp in comps:                                   # a fused GROUP held by outside
          cap = min(min(cap_of(r, fired), entry.get(r, D)) for r in comp)
          base = max(s[r] for r in comp)
          g_new = min(step_to(r, base, cap) for r in comp)
          if g_new <= base + eps * dt:
            continue
          s_try = list(s)
          for r in comp:
            s_try[r] = g_new
          who = _verdict(collide(assemble(s_try), fired), -1)
          if _trace:
            print(f"[retiming trace] wedge scan: GROUP {comp}@{base:.3f} g_new={g_new:.3f} "
                  f"who={who if who is not None else 'FREE'} pairs="
                  f"{getattr(collide, 'last_pairs', '?')} fired={fired} "
                  f"s={[round(x, 2) for x in s]}")
          who = (who or frozenset()) - set(comp)
          if who:
            r = comp[0]
            j = max(who, key=lambda j: abs(s[j] - s[r]))
            raise _Wedge(r, j, base, s[j])
        wedge = 0                                            # only static blocks: wait/stall
      if stall > stall_steps:
        state = ", ".join(f"arm{r}@{s[r]:.3f}" for r in range(n))
        raise RuntimeError(
            f"[retiming] coordination DEADLOCK: no progress for {stall_steps} steps, "
            f"fired {fired}/{K}, D={D:.2f}, {state}, couplings="
            f"{ {tuple(sorted(G)): [[round(a, 1), round(b, 1)] for a, b in ws] for G, ws in couplings.items()} }")

  q = phase = fires = prog = None
  for attempt in range(MAX_RESTARTS + 1):
    try:
      q, phase, fires, prog = run_attempt()
      break
    except _Wedge as w:
      G = frozenset((w.r, w.j))
      lo, hi = min(w.sr, w.sj), max(w.sr, w.sj)
      new_a = max(0.0, lo - REWIND)
      new_b = min([b for b in bar_s if b > hi - eps] or [D])
      # DISJOINT WINDOWS, not one merged span per pair: a conflict at s=104 must not be
      # unioned with an unrelated one at s=7 (measured on car-mini -- that merge grew a
      # 5.6-wide window into 98 wide and fused the arms over 88% of the arc, i.e. plain
      # lockstep). Only widen a window the new conflict actually touches.
      wins = couplings.setdefault(G, [])
      hit = next((k for k, (a, b) in enumerate(wins)
                  if new_a <= b + REWIND and a - REWIND <= new_b), None)
      if hit is None:
        wins.append([new_a, new_b])
      else:
        a, b = wins[hit]
        ext_b = min([bb for bb in bar_s if bb > b + eps] or [D])
        na, nb = max(0.0, min(a, new_a - REWIND)), max(b, new_b, ext_b)
        if (na, nb) == (a, b):
          raise RuntimeError(
              f"[retiming] coupling arms {sorted(G)} over [{na:.2f},{nb:.2f}] did not "
              f"resolve the wedge at s=({w.sr:.2f},{w.sj:.2f}) -- the saved path cannot "
              f"be desynchronized here; re-plan or run without --prioritized.")
        wins[hit] = [na, nb]
      wins.sort()
      merged = []                                            # keep the windows disjoint
      for wv in wins:
        if merged and wv[0] <= merged[-1][1] + eps:
          merged[-1][1] = max(merged[-1][1], wv[1])
        else:
          merged.append(list(wv))
      couplings[G] = merged
      if _trace:
        cov = sum(b - a for a, b in merged)
        print(f"[retiming trace] RESTART {attempt + 1}: wedge arm{w.r}@{w.sr:.2f} vs "
              f"arm{w.j}@{w.sj:.2f} -> couple {sorted(G)} windows "
              f"{[[round(a, 1), round(b, 1)] for a, b in merged]} "
              f"({100 * cov / max(D, 1e-9):.0f}% of arc)")
  else:
    raise RuntimeError(f"[retiming] no wedge-free schedule after {MAX_RESTARTS} restarts; "
                       f"couplings={couplings} -- re-plan or run without --prioritized.")
  s_final = [D] * n
  for _ in range(int(round(settle / dt)) if dt > 0 else 0):
    q.append(q[-1] if q else assemble(s_final))
    phase.append("settle")
  _coupling_rows = [((int(i), int(j)), float(a), float(b))
                    for G, wins in couplings.items() for i, j in [sorted(G)] for a, b in wins]
  return TimedTrajectory(dt=dt, q=np.asarray(q, float) if q else np.zeros((0, dof)),
                         phase=phase, fires=fires,
                         progress=np.asarray(prog, float) if prog else None,
                         couplings=_coupling_rows)


def phase_intervals(traj):
  """Inclusive sample intervals separated by fire configurations.

  Every interval shares its boundary fire sample with its neighbour. A phase smoother pins
  those samples exactly, so replacing one interval by the certified source cannot introduce a
  position discontinuity.
  """
  n = len(traj.q)
  if n == 0:
    return []
  pts = [0] + [int(i) for i, _ in traj.fires] + [n - 1]
  pts = sorted(set(min(max(i, 0), n - 1) for i in pts))
  return [(a, b) for a, b in zip(pts[:-1], pts[1:]) if b >= a]


def phase_interval_at(traj, sample):
  """Index of the fire-to-fire interval containing ``sample``."""
  iv = phase_intervals(traj)
  if not iv:
    return 0
  sample = int(sample)
  return next((k for k, (a, b) in enumerate(iv) if a <= sample <= b), len(iv) - 1)


def trajectory_smoothness(q, dt, arm_dims):
  """RMS joint acceleration and jerk, averaged over arms (diagnostic, not a constraint)."""
  q = np.asarray(q, dtype=float)
  if len(q) < 4:
    return {"accel_rms": 0.0, "jerk_rms": 0.0}
  aa, jj = [], []
  for dims in arm_dims:
    x = q[:, np.asarray(dims, dtype=int)]
    v = np.diff(x, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    aa.append(float(np.sqrt(np.mean(a * a))))
    jj.append(float(np.sqrt(np.mean(j * j))))
  return {"accel_rms": float(np.mean(aa)), "jerk_rms": float(np.mean(jj))}


def trajectory_smoothness_by_arm(q, dt, arm_dims):
  """The same derivative metrics as ``trajectory_smoothness``, without arm averaging."""
  q = np.asarray(q, dtype=float)
  if len(q) < 4:
    return [{"accel_rms": 0.0, "jerk_rms": 0.0} for _ in arm_dims]
  out = []
  for dims in arm_dims:
    x = q[:, np.asarray(dims, dtype=int)]
    v = np.diff(x, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    out.append({"accel_rms": float(np.sqrt(np.mean(a * a))),
                "jerk_rms": float(np.sqrt(np.mean(j * j)))})
  return out


def smoothness_not_worse(before, after, tolerance=0.02):
  """Whether a smoothing proposal keeps both acceleration and jerk within tolerance."""
  tolerance = max(float(tolerance), 0.0)
  return all(float(after[k]) <= (1.0 + tolerance) * max(float(before[k]), 1e-12)
             for k in ("accel_rms", "jerk_rms"))


def _project_nonnegative_sum(values, target):
  """Euclidean projection of a vector onto ``x >= 0, sum(x) == target``.

  Unlike multiplying an entire fire interval by ``target / sum(values)``, this adds the
  smallest possible correction in L2. Adjacent intervals therefore do not acquire arbitrary
  scale-factor jumps at their shared mode switch. This is the simplex projection of
  Duchi et al.; the small vectors here make the direct sort preferable to an iterative solver.
  """
  v = np.asarray(values, dtype=float)
  z = max(float(target), 0.0)
  if not len(v):
    return v.copy()
  if z <= 1e-15:
    return np.zeros_like(v)
  u = np.sort(v)[::-1]
  cssv = np.cumsum(u) - z
  idx = np.arange(1, len(u) + 1, dtype=float)
  good = u - cssv / idx > 0.0
  if not np.any(good):
    return np.full_like(v, z / len(v))
  rho = int(np.flatnonzero(good)[-1])
  theta = cssv[rho] / float(rho + 1)
  out = np.maximum(v - theta, 0.0)
  # Make the equality exact enough for event anchors even after a long float64 reduction.
  residual = z - float(out.sum())
  if abs(residual) > 1e-14 * max(1.0, z):
    j = int(np.argmax(out))
    out[j] = max(0.0, out[j] + residual)
  return out


def motion_islands(q, dt, arm_dims, stopped_rate=0.05):
  """Contiguous per-arm motion bursts, measured on the actual joint reference.

  A returned row is ``{"arm", "start", "end", "duration", "arc"}``, where ``start`` and
  ``end`` are sample indices and ``arc`` is accumulated joint-space motion.  Tiny bursts
  separated by stopped samples are the visually objectionable move-stop-move pattern that
  aggregate stopped/crawl percentages do not localize.
  """
  q = np.asarray(q, dtype=float)
  if len(q) < 2:
    return []
  out = []
  threshold = max(float(stopped_rate), 0.0) * max(float(dt), 0.0)
  for r, dims in enumerate(arm_dims):
    step = np.linalg.norm(np.diff(q[:, np.asarray(dims, dtype=int)], axis=0), axis=1)
    moving = step >= threshold
    start = None
    for j, yes in enumerate(moving):
      if yes and start is None:
        start = j
      if start is not None and (not yes or j == len(moving) - 1):
        stop = j - 1 if not yes else j                     # inclusive transition index
        out.append({"arm": r, "start": start, "end": stop + 1,
                    "duration": (stop - start + 1) * float(dt),
                    "arc": float(step[start:stop + 1].sum())})
        start = None
  return out


def internal_slow_episodes(q, dt, arm_dims, stopped_rate=0.05):
  """Count sub-cruise runs bracketed by fast motion, excluding outer acceleration ramps."""
  q = np.asarray(q, dtype=float)
  total = 0
  for dims in arm_dims:
    v = np.linalg.norm(np.diff(q[:, np.asarray(dims, dtype=int)], axis=0), axis=1) / dt
    slow = v < 10.0 * stopped_rate                           # 0.5 x nominal cruise
    start = None
    for i, yes in enumerate(slow):
      if yes and start is None:
        start = i
      if start is not None and (not yes or i == len(slow) - 1):
        end = i - 1 if not yes else i
        if np.any(~slow[:start]) and np.any(~slow[end + 1:]):
          total += 1
        start = None
  return total


def motion_support(q, dt, arm_dims, stopped_rate=0.05):
  """Aggregate moving support and conditional speed for a fixed-duration trajectory.

  Maximizing idle time is equivalent to minimizing ``moving_samples`` when duration and path
  displacement are fixed. ``mean_moving_rate`` distinguishes a compact intentional burst from
  motion merely reclassified below the stop threshold.
  """
  q = np.asarray(q, dtype=float)
  total = max(len(q) - 1, 0) * len(arm_dims)
  moving = 0
  moving_rates = []
  threshold = max(float(stopped_rate), 0.0)
  if len(q) > 1:
    for d in arm_dims:
      rate = np.linalg.norm(np.diff(q[:, np.asarray(d, dtype=int)], axis=0), axis=1) \
          / max(float(dt), 1e-12)
      mask = rate >= threshold
      moving += int(np.count_nonzero(mask))
      moving_rates.extend(rate[mask].tolist())
  return {"moving_samples": moving,
          "idle_fraction": 1.0 - moving / max(total, 1),
          "mean_moving_rate": float(np.mean(moving_rates)) if moving_rates else 0.0}


def batch_phase_motions(traj, path, arm_dims, fire_actors, bar_s, locked=(),
                        min_motion=0.08, max_gap=0.35, stopped_rate=0.05,
                        collision_free=None, peak_growth=1.02,
                        maximize_idle=False, intent_ramp_time=0.08,
                        intent_rate=2.0):
  """Merge short move-stop-move islands into a single smooth motion.

  This is a conservative final proposal after phase scheduling/smoothing.  Only adjacent
  motion islands of one arm are considered, at least one of which must be smaller than
  ``min_motion`` radians of accumulated joint motion and whose stopped gap is no longer than
  ``max_gap`` seconds.  Their fixed total phase displacement is spread over the combined
  window into one contiguous near-cruise burst with smooth acceleration/deceleration. The
  short island is preferentially moved next to the larger one, leaving one longer stationary
  period instead of crawling across the old gap. Thus the samples before and after the window,
  trajectory duration, event indices, and final configuration do not change.

  With ``maximize_idle``, every adjacent pair is eligible regardless of size/gap and its
  displacement is packed into the shortest raised-ramp burst that respects the source peak
  joint rate. This is a non-convex temporal-support objective: it intentionally prefers fast
  motion plus clean idle over spreading motion across the available time. Several temporal
  placements are collision-tested because the shortest burst can lie near the corridor edge.

  A merge never crosses that arm's own event, or a rigid/collision coupling window. Foreign
  events may be crossed: the global passage inequality ``phase >= barrier`` is checked at
  every fire, and ``collision_free(q, sample)`` can audit both the pre- and post-event modes.
  The candidate is also rejected if it grows that arm's largest source joint step.  Returns a
  new trajectory and a report; the source is always a valid whole-trajectory fallback.
  """
  if traj.progress is None:
    raise ValueError("motion batching requires per-arm progress")
  if min_motion <= 0.0 or max_gap <= 0.0:
    raise ValueError("motion batching thresholds must be positive")
  src_progress = np.asarray(traj.progress, dtype=float)
  progress = src_progress.copy()
  q = np.asarray(traj.q, dtype=float).copy()
  P = np.asarray(path, dtype=float)
  dims = [np.asarray(d, dtype=int) for d in arm_dims]
  if len(q) != len(progress) or progress.ndim != 2 or progress.shape[1] != len(dims):
    raise ValueError("motion batching progress shape does not match trajectory/arms")
  segs, D = arc_length(list(P))
  csum = np.concatenate([[0.0], np.cumsum(segs)]) if segs else np.array([0.0])
  fires = [(int(i), list(ev)) for i, ev in traj.fires]
  actors = [set(int(r) for r in a) for a in fire_actors]
  bars = [float(x) for x in bar_s]
  source_peak = [float(np.linalg.norm(np.diff(q[:, d], axis=0), axis=1).max())
                 if len(q) > 1 else 0.0 for d in dims]

  # Convert phase-space coupling windows to protected sample masks. Equality windows must be
  # byte-exact; merely preserving the endpoints can still alter closed-chain relative speed.
  protected = np.zeros((len(q), len(dims)), dtype=bool)
  protected_phase = {r: [] for r in range(len(dims))}
  for row in locked:
    if len(row) == 3:
      (i, j), lo, hi = row
    else:
      i, j, lo, hi = row
    for r in (int(i), int(j)):
      protected_phase[r].append((float(lo), float(hi)))
      protected[:, r] |= ((progress[:, r] >= float(lo) - 1e-9)
                          & (progress[:, r] <= float(hi) + 1e-9))

  report = {"before": len(motion_islands(q, traj.dt, dims, stopped_rate)),
            "after": 0, "merged": 0, "collision_rejected": 0,
            "rate_rejected": 0, "event_rejected": 0, "noeffect_rejected": 0,
            "quality_rejected": 0, "small_before": 0, "small_after": 0,
            "internal_slow_before": 0, "internal_slow_after": 0,
            "accepted_windows": [],
            "support_before": motion_support(q, traj.dt, dims, stopped_rate),
            "support_after": None,
            "smoothness_before": trajectory_smoothness(q, traj.dt, dims),
            "smoothness_after": None}

  def arm_runs(r):
    return [x for x in motion_islands(q, traj.dt, [dims[r]], stopped_rate)]

  for r in range(len(dims)):
    # Every accepted merge reduces the run count, so restarting the scan terminates.
    while True:
      runs = arm_runs(r)
      report["small_before"] += 0                           # populated globally below
      accepted = False
      for left, right in zip(runs, runs[1:]):
        gap = (right["start"] - left["end"]) * traj.dt
        if gap <= 0.0 or (not maximize_idle and gap > max_gap):
          continue
        if not maximize_idle and min(left["arc"], right["arc"]) >= min_motion:
          continue
        lo, hi = int(left["start"]), int(right["end"])
        if hi - lo < 3:
          continue
        # Own event stops and equality spans are semantic, not cosmetic stutters.
        own_fire = any(lo < fi < hi and k < len(actors) and r in actors[k]
                       for k, (fi, _) in enumerate(fires))
        crosses_lock = any(progress[hi, r] >= a - 1e-9 and progress[lo, r] <= b + 1e-9
                           for a, b in protected_phase[r])
        if own_fire or crosses_lock or np.any(protected[lo:hi + 1, r]):
          report["event_rejected"] += 1
          continue
        delta = float(progress[hi, r] - progress[lo, r])
        if delta <= 1e-12:
          report["noeffect_rejected"] += 1
          continue
        steps = hi - lo
        local_step = np.linalg.norm(np.diff(q[lo:hi + 1, dims[r]], axis=0), axis=1)
        old_active = int(np.count_nonzero(
            local_step >= max(stopped_rate, 0.0) * traj.dt))
        active_steps = max(3, old_active)
        taper_time = intent_ramp_time if maximize_idle else 0.04
        taper_steps = max(1, min(int(round(taper_time / traj.dt)), steps // 4))
        if maximize_idle:
          # Find the shortest feasible support by bisection. For a fixed monotone geometric
          # path, peak step decreases with duration; the final exact check below remains the
          # authority in case waypoint discretization makes that approximation imperfect.
          cap = max(peak_growth * source_peak[r], float(intent_rate) * traj.dt) + 1e-10

          def _peak_for(move_steps):
            ramp = max(1, min(taper_steps, move_steps // 3))
            inc = _fill_shape(move_steps, ramp) * delta
            ps = np.r_[progress[lo, r], progress[lo, r] + np.cumsum(inc)]
            qa = np.asarray([config_at(list(P), csum, segs, float(s))[dims[r]] for s in ps])
            return float(np.linalg.norm(np.diff(qa, axis=0), axis=1).max())

          low, high = 3, steps
          if _peak_for(high) > cap:
            report["rate_rejected"] += 1
            continue
          while low < high:
            mid = (low + high) // 2
            if _peak_for(mid) <= cap:
              high = mid
            else:
              low = mid + 1
          first_n = low
          durations = sorted(set([first_n,
                                  min(steps, int(np.ceil(1.15 * first_n))),
                                  min(steps, int(np.ceil(1.35 * first_n)))]))
        else:
          first_n = min(steps, active_steps + 2 * taper_steps)
          durations = sorted(set([first_n, min(steps, (first_n + steps) // 2), steps]))
        pair_accepted = False
        for move_steps in durations:
          free_steps = steps - move_steps
          # Prefer absorbing the small afterthought into the large motion. The opposite side
          # and a centred placement are tried if an event passage or collision needs them.
          offsets = ([0, free_steps, free_steps // 2] if right["arc"] < left["arc"]
                     else [free_steps, 0, free_steps // 2])
          if maximize_idle and free_steps > 0:
            offsets += np.linspace(0, free_steps, min(9, free_steps + 1)).round().astype(int).tolist()
          for offset in dict.fromkeys(offsets):
            ramp = max(1, min(taper_steps, move_steps // 3))
            inc = _fill_shape(move_steps, ramp) * delta
            pseg = np.full(steps + 1, progress[lo, r], dtype=float)
            pseg[offset + 1:offset + move_steps + 1] = progress[lo, r] + np.cumsum(inc)
            pseg[offset + move_steps + 1:] = progress[hi, r]
            pcol = progress[:, r].copy()
            pcol[lo:hi + 1] = pseg

            # Non-actors may move through a foreign switch, but every robot must already
            # have passed that barrier when its post-event world becomes active.
            event_ok = all(not (lo <= fi <= hi and k < len(bars)
                                and pcol[fi] < bars[k] - 1e-7)
                           for k, (fi, _) in enumerate(fires))
            if not event_ok:
              report["event_rejected"] += 1
              continue
            qarm = np.asarray([config_at(list(P), csum, segs, float(s))[dims[r]]
                               for s in pcol[lo:hi + 1]])
            new_peak = float(np.linalg.norm(np.diff(qarm, axis=0), axis=1).max())
            rate_cap = (max(peak_growth * source_peak[r], float(intent_rate) * traj.dt)
                        if maximize_idle else peak_growth * source_peak[r])
            if new_peak > rate_cap + 1e-10:
              report["rate_rejected"] += 1
              continue
            if maximize_idle:
              new_active = int(np.count_nonzero(
                  np.linalg.norm(np.diff(qarm, axis=0), axis=1)
                  >= max(stopped_rate, 0.0) * traj.dt))
              if new_active >= old_active:
                report["noeffect_rejected"] += 1
                continue
            candidate = q.copy()
            candidate[lo:hi + 1, dims[r]] = qarm
            if collision_free is not None and any(
                not bool(collision_free(candidate[i], i)) for i in range(lo, hi + 1)):
              report["collision_rejected"] += 1
              continue
            new_count = len(motion_islands(candidate, traj.dt, [dims[r]], stopped_rate))
            if new_count >= len(runs):
              report["noeffect_rejected"] += 1
              continue
            # Packing must not trade a literal stop for a comb of sub-cruise hesitations.
            # Ignore the natural acceleration/deceleration at the outer ends of a burst and
            # reject only NEW slow runs bracketed by fast motion.
            old_slow = internal_slow_episodes(
                q, traj.dt, [dims[r]], stopped_rate)
            new_slow = internal_slow_episodes(
                candidate, traj.dt, [dims[r]], stopped_rate)
            if new_slow > old_slow:
              report["quality_rejected"] += 1
              continue
            q, progress[:, r] = candidate, pcol
            report["merged"] += 1
            report["accepted_windows"].append((r, lo, hi))
            accepted = pair_accepted = True
            break
          if pair_accepted:
            break
        if pair_accepted:
          break
      if not accepted:
        break

  before_rows = motion_islands(traj.q, traj.dt, dims, stopped_rate)
  after_rows = motion_islands(q, traj.dt, dims, stopped_rate)
  report["before"], report["after"] = len(before_rows), len(after_rows)
  report["small_before"] = sum(x["arc"] < min_motion for x in before_rows)
  report["small_after"] = sum(x["arc"] < min_motion for x in after_rows)
  report["internal_slow_before"] = internal_slow_episodes(
      traj.q, traj.dt, dims, stopped_rate)
  report["internal_slow_after"] = internal_slow_episodes(
      q, traj.dt, dims, stopped_rate)
  report["support_after"] = motion_support(q, traj.dt, dims, stopped_rate)
  report["smoothness_after"] = trajectory_smoothness(q, traj.dt, dims)
  out = TimedTrajectory(dt=traj.dt, q=q, phase=list(traj.phase), fires=fires,
                        progress=progress, couplings=list(traj.couplings))
  return out, report


def foreign_switch_stops(q, dt, arm_dims, fires, fire_actors, stopped_rate=0.05,
                         context=0.2):
  """Count stop-go transitions centred on another robot's event.

  A foreign switch is counted only when the robot has meaningful motion within ``context`` on
  both sides but at least one transition immediately adjacent to the switch is stopped. Purely
  idle robots are therefore not mislabeled.
  """
  q = np.asarray(q, dtype=float)
  if len(q) < 3:
    return 0
  w = max(1, int(round(float(context) / max(float(dt), 1e-12))))
  threshold = max(float(stopped_rate), 0.0)
  total = 0
  for k, (fi, _) in enumerate(fires):
    fi = int(fi)
    actors = set(fire_actors[k]) if k < len(fire_actors) else set()
    if fi <= 0 or fi >= len(q) - 1:
      continue
    for r, d in enumerate(arm_dims):
      if r in actors:
        continue
      rate = np.linalg.norm(np.diff(q[:, np.asarray(d, dtype=int)], axis=0), axis=1) \
          / max(float(dt), 1e-12)
      left = rate[max(0, fi - w):fi]
      right = rate[fi:min(len(rate), fi + w)]
      if (len(left) and len(right) and np.max(left) >= threshold
          and np.max(right) >= threshold
          and min(rate[fi - 1], rate[fi]) < threshold):
        total += 1
  return total


def intentional_phase_motions(traj, path, arm_dims, fire_actors, bar_s, locked=(),
                              stopped_rate=0.05, collision_free=None,
                              intent_rate=2.0, ramp_time=0.08, stop_time=0.02):
  """Schedule one delayed continuous motion between each robot's OWN events.

  Foreign mode switches are not anchors: a burst may cross them with nonzero velocity provided
  the global barrier has been passed and the configuration is collision-free in both modes.
  Each own-event-to-own-event interval is packed as one shortest rate-bounded ramped burst and
  placed as late as possible; earlier placements are tried when event passage or the collision
  corridor requires them. Rigid/collision-coupled intervals remain byte-exact.
  """
  if traj.progress is None:
    raise ValueError("intentional phase scheduling requires per-arm progress")
  if intent_rate <= 0.0 or ramp_time <= 0.0 or stop_time < 0.0:
    raise ValueError("intent rate/ramp must be positive and stop time non-negative")
  progress = np.asarray(traj.progress, dtype=float).copy()
  q = np.asarray(traj.q, dtype=float).copy()
  P = np.asarray(path, dtype=float)
  dims = [np.asarray(d, dtype=int) for d in arm_dims]
  fires = [(int(i), list(ev)) for i, ev in traj.fires]
  actors = [set(int(r) for r in a) for a in fire_actors]
  bars = [float(x) for x in bar_s]
  segs, _ = arc_length(list(P))
  csum = np.concatenate([[0.0], np.cumsum(segs)]) if segs else np.array([0.0])
  protected = np.zeros((len(q), len(dims)), dtype=bool)
  for row in locked:
    if len(row) == 3:
      (i, j), lo, hi = row
    else:
      i, j, lo, hi = row
    for r in (int(i), int(j)):
      protected[:, r] |= ((progress[:, r] >= float(lo) - 1e-9)
                          & (progress[:, r] <= float(hi) + 1e-9))

  report = {"accepted": 0, "unresolved": [], "protected": 0,
            "collision_rejected": 0, "event_rejected": 0, "rate_rejected": 0,
            "noeffect_rejected": 0, "accepted_windows": [],
            "support_before": motion_support(q, traj.dt, dims, stopped_rate),
            "smoothness_before": trajectory_smoothness(q, traj.dt, dims),
            "islands_before": len(motion_islands(q, traj.dt, dims, stopped_rate)),
            "slow_before": internal_slow_episodes(q, traj.dt, dims, stopped_rate),
            "foreign_stops_before": foreign_switch_stops(
                q, traj.dt, dims, fires, actors, stopped_rate)}
  stop_steps = max(0, int(round(stop_time / traj.dt)))
  ramp_steps = max(1, int(round(ramp_time / traj.dt)))
  threshold_step = max(stopped_rate, 0.0) * traj.dt

  for r in range(len(dims)):
    anchors = [(0, None)]
    anchors.extend((fi, k) for k, (fi, _) in enumerate(fires)
                   if k < len(actors) and r in actors[k] and 0 < fi < len(q) - 1)
    anchors.append((len(q) - 1, None))
    anchors = sorted(dict.fromkeys(anchors))
    for (a, ka), (b, kb) in zip(anchors, anchors[1:]):
      if b - a < 4:
        continue
      s0, s1 = float(progress[a, r]), float(progress[b, r])
      if s1 - s0 <= 1e-10:
        continue
      old_qarm = q[a:b + 1, dims[r]]
      old_step = np.linalg.norm(np.diff(old_qarm, axis=0), axis=1)
      old_active = int(np.count_nonzero(old_step >= threshold_step))
      old_islands = len(motion_islands(old_qarm, traj.dt, [np.arange(len(dims[r]))],
                                       stopped_rate))
      if old_active == 0:
        continue
      if np.any(protected[a:b + 1, r]):
        report["protected"] += 1
        continue
      left = a + (stop_steps if ka is not None else 0)
      right = b - (stop_steps if kb is not None else 0)
      available = right - left
      if available < 3:
        report["unresolved"].append((r, a, b, "no room after own-event stops"))
        continue
      cap = float(intent_rate) * traj.dt + 1e-10

      def _packed(move_steps, offset):
        ramp = max(1, min(ramp_steps, move_steps // 3))
        inc = _fill_shape(move_steps, ramp) * (s1 - s0)
        pseg = np.full(b - a + 1, s0, dtype=float)
        start = (left - a) + int(offset)
        pseg[start + 1:start + move_steps + 1] = s0 + np.cumsum(inc)
        pseg[start + move_steps + 1:] = s1
        qa = np.asarray([config_at(list(P), csum, segs, float(s))[dims[r]] for s in pseg])
        peak = float(np.linalg.norm(np.diff(qa, axis=0), axis=1).max())
        return pseg, qa, peak

      low, high = 3, available
      if _packed(high, 0)[2] > cap:
        report["rate_rejected"] += 1
        report["unresolved"].append((r, a, b, "intent rate infeasible"))
        continue
      while low < high:
        mid = (low + high) // 2
        if _packed(mid, 0)[2] <= cap:
          high = mid
        else:
          low = mid + 1
      move_steps = low
      free = available - move_steps
      # Latest feasible start is the visual objective. A descending grid supplies corridor
      # alternatives without reintroducing a reactive online controller.
      offsets = np.linspace(free, 0, min(9, free + 1)).round().astype(int).tolist()
      accepted = False
      for offset in dict.fromkeys(offsets):
        pseg, qarm, peak = _packed(move_steps, offset)
        if peak > cap:
          report["rate_rejected"] += 1
          continue
        pcol = progress[:, r].copy()
        pcol[a:b + 1] = pseg
        if any(a <= fi <= b and k < len(bars) and pcol[fi] < bars[k] - 1e-7
               for k, (fi, _) in enumerate(fires)):
          report["event_rejected"] += 1
          continue
        candidate = q.copy()
        candidate[a:b + 1, dims[r]] = qarm
        if collision_free is not None:
          # Match plan_schedule's multi-rate contract: screen candidates on an exact 10ms
          # grid (plus every mode switch), then let the caller's mandatory 2ms final audit
          # certify or restore the complete interval. Running the full scene predicate at
          # every 2ms sample for every rejected placement made this stage needlessly minutes
          # slower without strengthening final acceptance.
          stride = max(1, int(round(0.01 / traj.dt)))
          samples = set(range(a, b + 1, stride))
          samples.update(fi for fi, _ in fires if a <= fi <= b)
          samples.update((a, b))
          if any(not bool(collision_free(candidate[i], i)) for i in sorted(samples)):
            report["collision_rejected"] += 1
            continue
        new_active = int(np.count_nonzero(
            np.linalg.norm(np.diff(qarm, axis=0), axis=1) >= threshold_step))
        new_islands = len(motion_islands(
            qarm, traj.dt, [np.arange(len(dims[r]))], stopped_rate))
        if new_islands != 1 or (new_active >= old_active and old_islands <= 1):
          report["noeffect_rejected"] += 1
          continue
        q, progress[:, r] = candidate, pcol
        report["accepted"] += 1
        report["accepted_windows"].append((r, a, b))
        accepted = True
        break
      if not accepted:
        report["unresolved"].append((r, a, b, "no collision/event-feasible placement"))

  report["support_after"] = motion_support(q, traj.dt, dims, stopped_rate)
  report["smoothness_after"] = trajectory_smoothness(q, traj.dt, dims)
  report["islands_after"] = len(motion_islands(q, traj.dt, dims, stopped_rate))
  report["slow_after"] = internal_slow_episodes(q, traj.dt, dims, stopped_rate)
  report["foreign_stops_after"] = foreign_switch_stops(
      q, traj.dt, dims, fires, actors, stopped_rate)
  out = TimedTrajectory(dt=traj.dt, q=q, phase=list(traj.phase), fires=fires,
                        progress=progress, couplings=list(traj.couplings))
  return out, report


def smooth_phase_trajectory(traj, path, arm_dims, fire_actors, window=0.12,
                            stop_time=0.02, iterations=3, locked=(),
                            restore_intervals=(), barrier_indices=(),
                            ramp_dist=DEFAULT_RAMP_DIST,
                            corner_dv=DEFAULT_CORNER_DV):
  """Smooth per-arm PHASE RATES while preserving the geometric paths and event samples.

  The raw prioritized scheduler changes rate one tick at a time. We convolve its non-negative
  phase increments with a Gaussian kernel, project each fire-to-fire interval back onto its
  exact phase displacement, and retain a short zero-rate neighbourhood for event actors.
  Positivity gives monotone phase; interval projection pins every complete phase vector at
  every mode switch. Configurations are rebuilt by evaluating each robot's original path at
  its new phase -- no joint-space corner cutting.

  ``locked`` contains ``(i, j, s0, s1)`` rigid/collision coupling windows. For now the
  fire-to-fire intervals intersecting such a window are left byte-for-byte unchanged for the
  involved arms. ``restore_intervals`` similarly restores complete intervals after a dense
  collision counterexample. A final per-arm check restores any interval in which smoothing
  would exceed that robot's largest certified source step by more than two percent; fixed
  event times must not compress a nicer-looking phase curve into an actuator-hostile speed spike.

  ``barrier_indices`` identifies the mode-boundary waypoints used by ``plan_schedule``. A
  crossing (non-acting) arm may have a genuine joint-space tangent discontinuity there. The
  scheduler deliberately reduces its phase rate over ``ramp_dist`` around such a corner; those
  source increments are hard constraints here. Without this constraint a Gaussian filter can
  erase the corner slowdown, producing a smoother phase signal but *larger* joint acceleration
  and jerk (the peg-stool failure that motivated this guard).
  The caller MUST collision-audit the result; this routine is a smoothness proposal, not a
  certificate.

  Returns ``(trajectory, report)``.
  """
  if traj.progress is None:
    raise ValueError("phase smoothing requires per-arm progress")
  if window <= 0.0 or traj.dt <= 0.0:
    raise ValueError("phase smoothing window and sample time must be positive")
  src = np.asarray(traj.progress, dtype=float)
  P = np.asarray(path, dtype=float)
  dims = [np.asarray(d, dtype=int) for d in arm_dims]
  if src.ndim != 2 or src.shape[1] != len(dims) or len(src) != len(traj.q):
    raise ValueError("phase progress shape does not match trajectory/arms")
  if len(P) < 1:
    raise ValueError("phase smoothing requires a non-empty geometric path")
  n, nr = src.shape
  if n < 3:
    return traj, {"restored": 0, "intervals": 0,
                  "before": trajectory_smoothness(traj.q, traj.dt, dims),
                  "after": trajectory_smoothness(traj.q, traj.dt, dims)}

  segs, D = arc_length(list(P))
  csum = np.concatenate([[0.0], np.cumsum(segs)]) if segs else np.array([0.0])
  # Smooth each robot in its OWN accumulated joint-space arc, not the composite path arc.
  # ``plan_schedule(per_arm_rate=True)`` already compensates for the local share
  # |dq_arm|/|dq_composite| so visible arm speed stays roughly constant. Filtering composite
  # phase would erase that compensation, especially across spans where this arm is stationary
  # (peg-stool arm3: phase looked 0.1% smoother while joint jerk grew 555%).
  arm_geom = []
  coord_src = np.zeros_like(src)
  phase_x = np.vstack([np.zeros((1, nr)), src])
  if segs:
    seg_arr = np.asarray(segs, dtype=float)
    for r in range(nr):
      AP = np.asarray(P[:, dims[r]], dtype=float)
      aseg = np.linalg.norm(np.diff(AP, axis=0), axis=1)
      acsum = np.concatenate([[0.0], np.cumsum(aseg)])
      sval = np.clip(src[:, r], 0.0, D)
      idx = np.searchsorted(csum, sval, side="right") - 1
      idx = np.clip(idx, 0, len(seg_arr) - 1)
      alpha = np.zeros(len(sval), dtype=float)
      nz = seg_arr[idx] > 1e-12
      alpha[nz] = (sval[nz] - csum[idx[nz]]) / seg_arr[idx[nz]]
      coord_src[:, r] = acsum[idx] + np.clip(alpha, 0.0, 1.0) * aseg[idx]
      coord_src[sval >= D - 1e-10, r] = acsum[-1]
      arm_geom.append((AP, list(aseg), acsum, float(acsum[-1])))
  else:
    for r in range(nr):
      arm_geom.append((np.asarray(P[:, dims[r]], dtype=float), [], np.array([0.0]), 0.0))
  # x includes the t=0 coordinate before traj.q[0], hence increments d[k] land on sample k.
  x = np.vstack([np.zeros((1, nr)), coord_src])
  d0 = np.maximum(np.diff(x, axis=0), 0.0)
  fire_x = [int(i) + 1 for i, _ in traj.fires]
  bounds = sorted(set([0] + fire_x + [n]))
  inc_intervals = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
  sample_intervals = phase_intervals(traj)
  restore = {int(k) for k in restore_intervals}

  # A rigidly co-held formation cannot be independently filtered. Preserve the source phase
  # for those two arms over every event interval that intersects its coupling window.
  arm_restore = set()
  for row in locked:
    if len(row) == 3:                                       # ((i,j), a, b)
      (i, j), lo, hi = row
    else:
      i, j, lo, hi = row
    i, j, lo, hi = int(i), int(j), float(lo), float(hi)
    for k, (a, b) in enumerate(sample_intervals):
      for r in (i, j):
        vals = src[a:b + 1, r]
        if len(vals) and float(vals.max()) >= lo - 1e-9 and float(vals.min()) <= hi + 1e-9:
          arm_restore.add((r, k))

  radius = max(2, int(round(0.5 * window / traj.dt)))
  z = np.arange(-radius, radius + 1, dtype=float)
  sigma = max(radius / 2.0, 1.0)
  kernel = np.exp(-0.5 * (z / sigma) ** 2)
  kernel /= kernel.sum()
  weights = np.ones((n, nr), dtype=float)
  # All robots start/end at rest. Only event actors stop at internal fire points; crossing
  # robots retain a smooth, nonzero derivative through somebody else's mode switch.
  centres = [(0, set(range(nr))), (n, set(range(nr)))]
  centres += [(fx, set(fire_actors[k]) if k < len(fire_actors) else set())
              for k, fx in enumerate(fire_x)]
  half_stop = max(1, int(round(0.5 * stop_time / traj.dt)))
  taper = max(1, radius)
  mids = np.arange(n, dtype=float) + 0.5
  for centre, actors in centres:
    dist = np.abs(mids - float(centre))
    w = np.ones(n, dtype=float)
    w[dist <= half_stop] = 0.0
    band = (dist > half_stop) & (dist < half_stop + taper)
    u = (dist[band] - half_stop) / taper
    w[band] = u * u * (3.0 - 2.0 * u)                       # smoothstep 0 -> 1
    for r in actors:
      weights[:, int(r)] = np.minimum(weights[:, int(r)], w)

  # Preserve the scheduler's intentional crossing-corner slowdowns. ``d0[k]`` advances from
  # x[k] to x[k+1], so its midpoint phase tells us whether that increment lies inside the same
  # +/-ramp_dist envelope used by plan_schedule.ramp_factor. Actors already stop at the event;
  # only crossing arms need this tangent-discontinuity guard.
  corner_fixed = np.zeros((n, nr), dtype=bool)
  bis = [int(i) for i in barrier_indices]
  if bis:
    mids_phase = 0.5 * (phase_x[:-1] + phase_x[1:])
    for k, bi in enumerate(bis):
      if bi <= 0 or bi >= len(P) - 1:
        continue
      actors = set(fire_actors[k]) if k < len(fire_actors) else set()
      for r in range(nr):
        if r in actors:
          continue
        din = P[bi, dims[r]] - P[bi - 1, dims[r]]
        dout = P[bi + 1, dims[r]] - P[bi, dims[r]]
        ni, no = float(np.linalg.norm(din)), float(np.linalg.norm(dout))
        if ni < 1e-9 or no < 1e-9:
          continue
        turn = float(np.linalg.norm(dout / no - din / ni))
        if corner_dv is None or corner_dv <= 0.0 or turn <= float(corner_dv):
          continue
        sb = float(csum[bi])
        corner_fixed[:, r] |= np.abs(mids_phase[:, r] - sb) <= max(float(ramp_dist), 0.0) + 1e-9

  cur = d0.copy()
  for _ in range(max(1, int(iterations))):
    proposal = np.empty_like(cur)
    for r in range(nr):
      pad = np.pad(cur[:, r], (radius, radius), mode="edge")
      proposal[:, r] = np.convolve(pad, kernel, mode="same")[radius:radius + n]
    proposal *= weights
    proposal = np.maximum(proposal, 0.0)
    for k, (a, b) in enumerate(inc_intervals):
      for r in range(nr):
        if k in restore or (r, k) in arm_restore:
          proposal[a:b, r] = d0[a:b, r]
          continue
        target = float(np.sum(d0[a:b, r]))
        fixed = corner_fixed[a:b, r]
        proposal[a:b, r][fixed] = d0[a:b, r][fixed]
        free = ~fixed
        fixed_total = float(np.sum(d0[a:b, r][fixed]))
        free_target = max(0.0, target - fixed_total)
        if target <= 1e-12:
          proposal[a:b, r] = 0.0
        elif not np.any(free):
          proposal[a:b, r][free] = d0[a:b, r][free]
        else:
          hard_zero = weights[a:b, r] <= 1e-15
          adjustable = free & ~hard_zero
          if not np.any(adjustable):
            proposal[a:b, r] = d0[a:b, r]
          else:
            proposal[a:b, r][hard_zero & free] = 0.0
            proposal[a:b, r][adjustable] = _project_nonnegative_sum(
                proposal[a:b, r][adjustable], free_target)
        proposal[a:b, r][fixed] = d0[a:b, r][fixed]
    cur = proposal

  out_coord = np.cumsum(cur, axis=0)
  # Exact restoration matters for rigid co-grasps: even harmless roundoff would turn equality
  # into a tiny relative motion once differentiated by a stiff controller.
  for k, (a, b) in enumerate(sample_intervals):
    if k in restore:
      out_coord[a:b + 1] = coord_src[a:b + 1]
    else:
      for r in range(nr):
        if (r, k) in arm_restore:
          out_coord[a:b + 1, r] = coord_src[a:b + 1, r]
  # Remove roundoff at every event anchor explicitly.
  for bx in bounds[1:]:
    out_coord[bx - 1] = x[bx]
  out_coord = np.maximum.accumulate(out_coord, axis=0)
  for r, (_, _, _, arm_D) in enumerate(arm_geom):
    out_coord[:, r] = np.minimum(out_coord[:, r], arm_D)
  # Clipping can change a source endpoint by a few ulps. Re-apply certified intervals last so
  # the promised exact rigid-coupling fallback is genuinely exact.
  for k, (a, b) in enumerate(sample_intervals):
    if k in restore:
      out_coord[a:b + 1] = coord_src[a:b + 1]
    else:
      for r in range(nr):
        if (r, k) in arm_restore:
          out_coord[a:b + 1, r] = coord_src[a:b + 1, r]

  # Map intrinsic arm arc back to the shared composite coordinate. Flat portions of this map
  # correspond to a stationary arm and are geometrically equivalent; choosing the last point
  # in a flat mirrors plan_schedule's immediate skip over zero-share bookkeeping spans.
  def _shared_progress(coord):
    progress = np.zeros_like(coord)
    for r, (_, _, acsum, arm_D) in enumerate(arm_geom):
      if arm_D <= 1e-12:
        progress[:, r] = src[:, r]
        continue
      rev_unique, rev_first = np.unique(acsum[::-1], return_index=True)
      last_idx = len(acsum) - 1 - rev_first
      order = np.argsort(rev_unique)
      uu = rev_unique[order]
      ss = csum[last_idx[order]]
      progress[:, r] = np.interp(coord[:, r], uu, ss)
    for bx in bounds[1:]:
      progress[bx - 1] = phase_x[bx]
    progress = np.maximum.accumulate(progress, axis=0)
    # Inverting intrinsic arc through a flat/roundoff-prone map is geometrically exact but not
    # necessarily byte-identical. Reapply every semantic raw region in the shared coordinate,
    # which is what downstream coupling and batching code consumes.
    for k, (a, b) in enumerate(sample_intervals):
      if k in restore:
        progress[a:b + 1] = src[a:b + 1]
      else:
        for r in range(nr):
          if (r, k) in arm_restore:
            progress[a:b + 1, r] = src[a:b + 1, r]
    for r in range(nr):
      fixed_samples = np.flatnonzero(corner_fixed[:, r])
      if len(fixed_samples):
        exact = np.unique(np.r_[fixed_samples, np.maximum(fixed_samples - 1, 0)])
        progress[exact, r] = src[exact, r]
    for bx in bounds[1:]:
      progress[bx - 1] = phase_x[bx]
    return progress

  def _rebuild(coord):
    oq = np.asarray(traj.q, dtype=float).copy()              # preserve any non-arm columns
    for kk in range(n):
      for rr in range(nr):
        AP, aseg, acsum, arm_D = arm_geom[rr]
        if arm_D > 1e-12:
          oq[kk, dims[rr]] = config_at(
              list(AP), acsum, aseg, float(coord[kk, rr]))
    raw_q = np.asarray(traj.q, dtype=float)
    for k, (a, b) in enumerate(sample_intervals):
      if k in restore:
        oq[a:b + 1] = raw_q[a:b + 1]
      else:
        for rr in range(nr):
          if (rr, k) in arm_restore:
            oq[a:b + 1, dims[rr]] = raw_q[a:b + 1, dims[rr]]
    return oq

  out_progress = _shared_progress(out_coord)
  out_q = _rebuild(out_coord)
  rate_restore = set()
  src_q = np.asarray(traj.q, dtype=float)
  source_peak = []
  for r in range(nr):
    _steps = np.linalg.norm(np.diff(src_q[:, dims[r]], axis=0), axis=1)
    source_peak.append(float(_steps.max()) if len(_steps) else 0.0)
  for k, (a, b) in enumerate(sample_intervals):
    if b <= a:
      continue
    for r in range(nr):
      new_step = np.linalg.norm(np.diff(out_q[a:b + 1, dims[r]], axis=0), axis=1)
      new_max = float(new_step.max()) if len(new_step) else 0.0
      if new_max > 1.02 * source_peak[r] + 1e-10:
        out_coord[a:b + 1, r] = coord_src[a:b + 1, r]
        rate_restore.add((r, k))
  if rate_restore:
    out_progress = _shared_progress(out_coord)
    out_q = _rebuild(out_coord)
  out = TimedTrajectory(dt=traj.dt, q=out_q, phase=list(traj.phase),
                        fires=[(int(i), list(ev)) for i, ev in traj.fires],
                        progress=out_progress, couplings=list(traj.couplings))
  return out, {"restored": len(restore), "intervals": len(inc_intervals),
               "locked_arm_intervals": len(arm_restore),
               "corner_fixed_increments": int(np.count_nonzero(corner_fixed)),
               "rate_limited_arm_intervals": len(rate_restore),
               "before": trajectory_smoothness(traj.q, traj.dt, dims),
               "after": trajectory_smoothness(out.q, out.dt, dims),
               "by_arm_before": trajectory_smoothness_by_arm(traj.q, traj.dt, dims),
               "by_arm_after": trajectory_smoothness_by_arm(out.q, out.dt, dims),
               "phase_before": trajectory_smoothness(
                   coord_src, traj.dt, [np.array([r]) for r in range(nr)]),
               "phase_after": trajectory_smoothness(
                   out_coord, traj.dt, [np.array([r]) for r in range(nr)]),
               "progress_max_delta": float(np.max(np.abs(out_progress - src)))}


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


def compact_polyline(points, atol=1e-9, rtol=1e-10):
  """Remove redundant samples without changing a joint-space polyline.

  mr_bench's shortcut pass serializes every interpolation sample along each accepted straight
  edge.  Those samples are useful while checking the edge, but they are not geometric knots.
  Passing all of them to ``spline_path`` turns one shortcut edge into dozens of independent
  cubic spans and makes certification scale with serialization density rather than path shape.

  A subsequence is collapsed only when (a) every point lies on its endpoint segment and (b)
  its accumulated arc length equals the endpoint chord.  The second condition deliberately
  preserves collinear backtracking.  Returned points are references to original samples, so
  exact task witnesses are never numerically reconstructed.
  """
  P = [np.asarray(p, float) for p in points]
  if len(P) <= 2:
    return P

  keep = {0, len(P) - 1}
  stack = [(0, len(P) - 1)]
  while stack:
    lo, hi = stack.pop()
    if hi <= lo + 1:
      continue
    a, b = P[lo], P[hi]
    d = b - a
    dd = float(d @ d)
    seg = [float(np.linalg.norm(P[k + 1] - P[k])) for k in range(lo, hi)]
    arc = float(sum(seg))
    chord = float(np.sqrt(dd))
    eps = float(atol) + float(rtol) * max(1.0, arc)

    max_dev, split = -1.0, lo + 1
    for k in range(lo + 1, hi):
      if dd <= 1e-30:
        dev = float(np.linalg.norm(P[k] - a))
      else:
        t = float(np.clip((P[k] - a) @ d / dd, 0.0, 1.0))
        dev = float(np.linalg.norm(P[k] - (a + t * d)))
      if dev > max_dev:
        max_dev, split = dev, k

    if max_dev <= eps and arc - chord <= eps:
      continue                                             # exact monotone interpolation run
    if max_dev <= eps:                                     # collinear but doubles back
      cumulative = np.cumsum(seg)
      split = lo + 1 + int(np.argmin(np.abs(cumulative[:-1] - 0.5 * arc)))
    keep.add(split)
    stack.append((lo, split))
    stack.append((split, hi))
  return [P[k] for k in sorted(keep)]


def compact_stretches(stretches, atol=1e-9, rtol=1e-10):
  """Compact every mode stretch independently, preserving all barrier endpoints."""
  out, before, after = [], 0, 0
  arc_before = arc_after = 0.0
  for waypoints, done in stretches:
    w = [np.asarray(q, float) for q in waypoints]
    c = compact_polyline(w, atol=atol, rtol=rtol)
    before += len(w)
    after += len(c)
    arc_before += sum(float(np.linalg.norm(b - a)) for a, b in zip(w, w[1:]))
    arc_after += sum(float(np.linalg.norm(b - a)) for a, b in zip(c, c[1:]))
    out.append((c, list(done)))
  return out, {"before": before, "after": after, "removed": before - after,
               "arc_before": arc_before, "arc_after": arc_after}


def iter_stretch_samples(stretches, resolution=0.01):
  """Yield dense, mode-local samples as ``(n, stretch, edge, alpha, q)``.

  Resolution follows mr_bench's edge checker: maximum absolute joint motion, not composite
  Euclidean arc.  Each stretch emits its first endpoint even when it duplicates the previous
  stretch's last endpoint, because the attachment mode changed at that exact configuration.
  """
  resolution = float(resolution)
  if resolution <= 0.0:
    raise ValueError("path audit resolution must be positive")
  n = 0
  for k, (waypoints, _done) in enumerate(stretches):
    w = [np.asarray(q, float) for q in waypoints]
    if not w:
      continue
    yield n, k, 0, 0.0, w[0]
    n += 1
    for edge, (a, b) in enumerate(zip(w, w[1:])):
      steps = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / resolution)))
      for j in range(1, steps + 1):
        alpha = j / steps
        yield n, k, edge, alpha, a + alpha * (b - a)
        n += 1


def audit_path(stretches, modes, free_of, limits=None, resolution=0.01,
               joint_tolerance=1e-9):
  """Strictly audit a mode-tagged base path before any smoothing or retiming.

  ``free_of(q, mode)`` may return a bool or ``(bool, detail)``.  The first failure is returned
  as structured data so callers can name the exact mode, edge, interpolation fraction, joint,
  or collision pair.  A successful report contains the number of checked samples.
  """
  if len(stretches) != len(modes):
    raise ValueError(f"path audit has {len(stretches)} stretches but {len(modes)} modes")
  lim = None
  if limits is not None:
    lo, hi = (np.asarray(x, float) for x in limits)
    lim = (lo, hi)
  checked = 0
  for sample, stretch, edge, alpha, q in iter_stretch_samples(stretches, resolution):
    checked += 1
    if lim is not None:
      lo, hi = lim
      if q.shape != lo.shape or q.shape != hi.shape:
        raise ValueError(
            f"path audit joint vector has shape {q.shape}, limits are {lo.shape}/{hi.shape}")
      bad = np.flatnonzero((q < lo - joint_tolerance) | (q > hi + joint_tolerance))
      if len(bad):
        return {"ok": False, "kind": "joint_limits", "samples": checked,
                "sample": sample, "stretch": stretch, "edge": edge, "alpha": alpha,
                "q": q, "joint_indices": bad.tolist(), "lower": lo[bad], "upper": hi[bad]}
    verdict = free_of(q, modes[stretch])
    if isinstance(verdict, tuple):
      ok, detail = verdict
    else:
      ok, detail = verdict, None
    if not bool(ok):
      return {"ok": False, "kind": "collision", "samples": checked,
              "sample": sample, "stretch": stretch, "edge": edge, "alpha": alpha,
              "q": q, "detail": detail}
  return {"ok": True, "samples": checked}


# =============================================================================================
# CORRIDOR RETIMING  --  the principled replacement for greedy desync + wedge recovery.
#
# THE PROBLEM. A composite path P(s) is certified collision-free only ON THE DIAGONAL
# (every arm at the same s). Retiming arms independently visits (s_0..s_{n-1}) combinations
# nobody ever checked -- that, and nothing else, is what makes per-arm retiming hard.
#
# THE STRUCTURE THAT MAKES IT TRACTABLE. Collisions are PAIRWISE: arm i vs arm j depends only
# on (s_i, s_j), and arm i vs the static world only on s_i. So the unknown region never has to
# be explored in n dimensions -- it decomposes into per-arm and per-pair bounds.
#
# STAGE A (`certify_corridor`) measures, ONCE, how far arms may drift apart before anything
# touches: a certified DESYNC CORRIDOR. Inside it, any timing at all is collision-free BY
# CONSTRUCTION -- no per-step oracle calls, no wedges, no restarts, no backtracking. Where the
# corridor is wide the arms are free; where it collapses to zero they are forced to move
# together, which is exactly how "these arms must stay at the same progress here" (a handover,
# a co-grasp, a tight formation) expresses itself -- as geometry, not as an inferred failure.
# `locked` lets known couplings be injected directly when we start deriving them from the plan.
#
# STAGE B (`corridor_schedule`) then plans progress inside that corridor, with an arm coming to
# rest ONLY at its own events -- never at a barrier it merely crosses.
#
# STAGE C (folded into Stage B) times it so that a robot is always at its velocity limit: each
# arm advances at joint_rate/|dq_arm/ds|, i.e. its own joints run at the limit whenever it is
# not held back, so the laggard -- who is never held back -- is always saturated.
# =============================================================================================
def certify_corridor(n, D, world_free, pair_free, ds_grid=1.0, ds_lead=0.25, lead_max=4.0,
                     locked=None, progress=None):
  """STAGE A: certify how far arms may desync, as a function of arc position.

    world_free(i, s, d) -> bool : arm i at s+d, every other arm at s, world state taken from
                                  s; True iff nothing involving arm i touches the STATIC world
                                  (table, resting parts) that did not already touch on the
                                  diagonal.
    pair_free(i, j, s, d) -> bool : arm i at s+d, arm j at s (others at s); True iff arms i
                                  and j (with whatever they carry) do not touch each other.

  Both are probed OUTWARD from d=0 in steps of `ds_lead`, stopping at the first violation, so
  the result is "every lead up to L is free" rather than "L happens to be free".

  `locked`: optional [(i, j, a, b), ...] arc spans where a pair is known to be rigidly coupled
  (handover, co-grasp) -- their corridor is forced to 0 there regardless of geometry.

  Returns a corridor dict consumed by `corridor_schedule`."""
  grid = np.arange(0.0, D + ds_grid, ds_grid)
  G, steps = len(grid), max(1, int(round(lead_max / ds_lead)))

  def _scan(fn):
    out = 0.0
    for t in range(1, steps + 1):
      d = t * ds_lead
      if not fn(d):
        break
      out = d
    return out
  # NB: leads past the end of the path are not "blocked", they are meaningless -- probes must
  # CLAMP s+d to D rather than fail, else the corridor collapses to 0 near the goal and the
  # schedule deadlocks there (measured).

  Lw = np.zeros((n, G))
  for i in range(n):
    for g, s in enumerate(grid):
      Lw[i, g] = _scan(lambda d, i=i, s=s: world_free(i, s, d))
    if progress:
      progress(f"    world corridor arm{i}: mean {Lw[i].mean():.2f} rad")
  Lp = {}
  for i in range(n):
    for j in range(i + 1, n):
      A = np.zeros((G, 2))                                   # [g,0]=i may lead j; [g,1]=j lead i
      for g, s in enumerate(grid):
        A[g, 0] = _scan(lambda d, i=i, j=j, s=s: pair_free(i, j, s, d))
        A[g, 1] = _scan(lambda d, i=i, j=j, s=s: pair_free(j, i, s, d))
      Lp[(i, j)] = A
      if progress:
        progress(f"    pair corridor {i}-{j}: mean {A.mean():.2f} rad, "
                 f"zero on {100.0 * np.mean(A <= 1e-9):.0f}% of the arc")
  cor = {"grid": grid, "ds": float(ds_grid), "Lw": Lw, "Lp": Lp, "D": float(D), "n": n,
         "ds_lead": float(ds_lead), "lead_max": float(lead_max)}
  for (i, j, a, b) in (locked or []):                        # known rigid couplings: no drift
    k = (min(i, j), max(i, j))
    if k in Lp:
      m = (grid >= a - 1e-9) & (grid <= b + 1e-9)
      Lp[k][m, :] = 0.0
  return cor


def corridor_width_report(cor):
  """Human-readable summary: how much desync freedom the scene actually has."""
  lines = [f"[corridor] certified over D={cor['D']:.1f} rad "
           f"({len(cor['grid'])} samples every {cor['ds']:g} rad, leads probed to "
           f"{cor['lead_max']:g} rad in {cor['ds_lead']:g} steps)"]
  Lw = cor["Lw"]
  for i in range(cor["n"]):
    lines.append(f"  arm{i} vs world : mean {Lw[i].mean():5.2f}  median {np.median(Lw[i]):5.2f} "
                 f" zero on {100.0 * np.mean(Lw[i] <= 1e-9):3.0f}% of arc")
  for (i, j), A in sorted(cor["Lp"].items()):
    lines.append(f"  arm{i}-arm{j}    : mean {A.mean():5.2f}  median {np.median(A):5.2f} "
                 f" zero on {100.0 * np.mean(A <= 1e-9):3.0f}% of arc "
                 f" (locked where 0 => must move together)")
  return "\n".join(lines)


def _cor_at(cor, s):
  return int(min(max(0, int(round(s / cor["ds"]))), len(cor["grid"]) - 1))


def brakeable(vmax, amax, ds):
  """Backward-sweep a speed limit so it can actually be DECELERATED into.

  A raw wrench-derived limit can drop sharply (a tight curve) faster than the acceleration
  bound allows an arm to slow down, which would let it enter the tight span too fast. One
  backward pass -- v[g] <= sqrt(v[g+1]^2 + 2*a*ds) -- makes the profile reachable, the same
  trick TOPP uses. `vmax`/`amax` are (n, G) arrays over an evenly spaced arc grid."""
  v = np.array(vmax, float, copy=True)
  a = np.asarray(amax, float)
  for r in range(v.shape[0]):
    for g in range(v.shape[1] - 2, -1, -1):
      v[r, g] = min(v[r, g], float(np.sqrt(v[r, g + 1] ** 2 + 2.0 * max(a[r, g], 0.0) * ds)))
  return v


def corridor_schedule(P, arm_dims, bar_s, acting, cor, dt, joint_rate=DEFAULT_JOINT_RATE,
                      ramp_dist=DEFAULT_RAMP_DIST, ramp_min=DEFAULT_RAMP_MIN, dwell=0.0,
                      settle=0.0, spread=True, max_steps=None, vmax=None, amax=None,
                      limit_ds=None, rate_clamp=False):
  """STAGES B+C: plan per-arm progress inside the certified corridor, then time it so a
  robot is always at its limit.

  Each arm advances as fast as ITS OWN joints allow (joint_rate / |dq_arm/ds| -- stage C),
  and comes to rest ONLY at barriers where it acts; barriers it merely crosses impose
  nothing. The only couplings are (a) its own next unfired event and (b) the corridor, which
  is a pure array lookup -- no collision checks here, because Stage A already certified every
  timing the corridor permits. The laggard is never corridor-capped, so progress (and hence
  termination) is guaranteed, and every event still fires in order with its arms at rest.

  `spread=True` adds the second pass: motion is re-paced continuously across the interval
  between an arm's own events (removing rush-then-wait) and re-clamped to the corridor, so
  arrival times are unchanged.

  `vmax`/`amax` ((n, G) arrays, arc units/s and /s^2) are extra per-arm speed and
  acceleration limits -- e.g. wrench limits, so an arm carrying a heavy group cannot
  accelerate hard enough to overload its own grasp. Only the arms that are actually loaded
  slow down, and arrivals simply move later. They are sampled on the corridor grid unless
  `limit_ds` gives their own (finer) spacing: a wrench limit needs the resolution of the
  geometry it came from, and a corner bound is a per-waypoint event that a 1-rad corridor
  cell would smear over a whole radian of arc.

  The spread pass does NOT respect them -- it can be locally 1.9x over the limit (measured);
  see the KNOWN HOLE in _corridor_spread."""
  P = np.asarray(P, float)
  segs, D = arc_length(list(P))
  csum = np.concatenate([[0.0], np.cumsum(segs)]) if segs else np.array([0.0])
  _segs_arr = np.asarray(segs, float)

  def _fcfg(x):
    # config_at, minus the per-call `list(P)` materialisation (a 330-row list build per call,
    # ~85k calls per schedule = ~75% of a whole sim run's wall time, profiled). Identical
    # arithmetic on the same float64 rows, so the output is BIT-IDENTICAL (probe-verified).
    i = min(int(np.searchsorted(csum, x, side="right")) - 1, len(_segs_arr) - 1)
    if not len(_segs_arr) or _segs_arr[i] < 1e-12:
      return P[i + 1]
    a = (x - csum[i]) / _segs_arr[i]
    return (1 - a) * P[i] + a * P[i + 1]
  n, K = len(arm_dims), len(bar_s)
  act = [set(int(a) for a in acting[k]) for k in range(K)]
  eps = 1e-6

  d = np.diff(P, axis=0)                                     # STAGE C: per-arm arc share
  tot = np.linalg.norm(d, axis=1)
  share = np.ones((n, max(len(d), 1)))
  for r in range(n):
    sr = np.linalg.norm(d[:, np.asarray(arm_dims[r], int)], axis=1)
    ok = tot > 1e-12
    share[r][ok] = np.maximum(sr[ok] / tot[ok], 1e-3)        # floor: idle arm must not fly

  rest = []                                                  # own events only -- NOT every barrier
  for r in range(n):
    rest.append(sorted({0.0, D} | {bar_s[k] for k in range(K) if r in act[k]}))

  def _lim_at(arr, s):                                       # index into vmax/amax at arc s
    if limit_ds:                                             # own grid (see `limit_ds`)
      return int(min(max(round(s / limit_ds), 0), np.shape(arr)[1] - 1))
    return min(_cor_at(cor, s), np.shape(arr)[1] - 1)

  floor_hits = {}

  def vlim(r, s):
    """Effective speed cap. THE FORCE CAP WINS OVER THE NEVER-STALL FLOOR.

    The floor (ramp_min * joint_rate = 0.04 rad/s by default) exists so an arm always arrives,
    but it used to be applied with max(), which SILENTLY EXCEEDED a tighter force limit --
    measured on car-mini, --force-limits asked for 0.02 rad/s on arm1 and 0.00 on arm2, both
    below the floor, so the certified wrench bound was simply ignored. Honour the cap instead
    and keep only an epsilon for termination; a cap that low is a real statement by the LP
    ('this waypoint cannot be traversed safely at any useful speed') and is reported rather
    than papered over."""
    cap = float(vmax[r][_lim_at(vmax, s)])
    fl = ramp_min * joint_rate
    if cap >= fl:
      return cap
    floor_hits[r] = floor_hits.get(r, 0) + 1
    return max(cap, 1e-4)                                    # epsilon: guarantee arrival only

  def rate_of(r, s):                                         # arm r at ITS joint limit
    return joint_rate / share[r][_seg_index(csum, segs, s)]

  def ramp_of(r, s):
    if ramp_dist <= 0:
      return 1.0
    pts = rest[r]
    i = int(np.searchsorted(pts, s, side="right"))
    prev = pts[i - 1] if i > 0 else 0.0
    nxt = pts[i] if i < len(pts) else D
    return max(ramp_min, min(1.0, (min(s - prev, nxt - s) + 1e-6) / ramp_dist))

  def cap_event(r, fired):
    """No arm may pass the PENDING barrier -- not just the ones acting there.

    Beyond a barrier the path is written in the POST-event attachment topology: after a mate,
    the part is welded to its target and the inserting arm has let go, so that arm's onward
    motion is "retreat empty-handed". An arm allowed to run ahead executes that retreat while
    still gripping the part and tears it off its seat (measured on car-mini: a 175.9N grasp
    break with a geometrically clean corridor -- no collision checker can see this, it is a
    semantics violation, the same class as a rigid co-grasp coupling).

    This is NOT a stop: the arm may arrive at the barrier at full speed and continue the
    moment the event fires. Only its OWN events put a zero-velocity point in its profile
    (see `rest`), and the spread pass paces arrivals so it flows through instead of waiting.
    It also keeps every arm inside one mode window, which is what makes the corridor lookup
    (certified with the world state taken from the lagging position) sound."""
    return bar_s[fired] if fired < K else D

  def cap_corridor(r, sv):
    """How far arm r may be AHEAD of the others -- certified, pure lookup, no oracle.
    The reference is the others' minimum, NOT the global minimum: the laggard is by
    definition ahead of nobody, so it must never cap itself (that deadlocked instantly)."""
    others = [sv[j] for j in range(n) if j != r]
    if not others:
      return D
    lim = D
    for j in range(n):
      if j == r:
        continue
      g = _cor_at(cor, min(sv[r], sv[j]))
      A = cor["Lp"].get((min(r, j), max(r, j)))
      if A is not None:
        lim = min(lim, sv[j] + (A[g, 0] if r < j else A[g, 1]))
    om = min(others)
    return min(lim, om + cor["Lw"][r][_cor_at(cor, om)])     # vs the static world

  if max_steps is None:
    max_steps = int(40 * n * D / max(joint_rate * dt, 1e-9)) + 10000
  s = [0.0] * n
  vel = [0.0] * n                                            # arc speed, for the accel limit
  fired, steps = 0, 0
  ready = None
  hold = [-1] * n
  n_dwell = int(round(dwell / dt)) if dt > 0 else 0
  q, phase, fires, prog = [], [], [], []
  while True:
    if steps >= max_steps:
      raise RuntimeError(f"[corridor] exceeded max_steps={max_steps} at fired {fired}/{K}, "
                         f"progress {[round(x, 2) for x in s]} of {D:.2f}")
    steps += 1
    # ADVANCE-THEN-PROJECT: propose every arm's step first, then pull leaders back into the
    # corridor. Testing each arm against the OTHERS' CURRENT positions would freeze a
    # zero-width corridor solidly (each arm capped at the other's present s), when in fact
    # arms may advance TOGETHER there -- a rigid pair keeps its offset, which is legal.
    cand = list(s)
    for r in range(n):
      if s[r] >= D - eps or steps <= hold[r]:
        vel[r] = 0.0
        continue
      lim = min(cap_event(r, fired), D)
      if s[r] < lim - eps:
        want = rate_of(r, s[r]) * ramp_of(r, s[r])            # stage C + own-event ramps
        if vmax is not None:                                  # e.g. wrench speed limit
          want = min(want, vlim(r, s[r]))
        if amax is not None:                                  # accel-limited slew
          am = max(float(amax[r][_lim_at(amax, s[r])]), 1e-9)
          want = float(np.clip(want, vel[r] - am * dt, vel[r] + am * dt))
        vel[r] = want
        cand[r] = min(lim, s[r] + want * dt)
        if rate_clamp and cand[r] > s[r] + eps:
          # RATE BUDGET vs REALISED MOTION. rate_of divides joint_rate by the arm's arc SHARE
          # in the segment it is currently IN, but a step can LAND in a segment where that arm
          # carries far more of the arc -- and then the joint speed actually executed exceeds
          # the budget. MEASURED on cross-new: 1.98 rad/s commanded against joint_rate 1.0.
          # Same off-by-one-cell class as the force-limit grid note in sim.py. A stiff arm
          # absorbs the overshoot exactly; a compliant one turns it into grasp load. So bound
          # the step by the motion it ACTUALLY produces rather than by the segment it started in.
          d0 = _fcfg(s[r])[arm_dims[r]]
          budget = joint_rate * dt
          for _ in range(4):                                 # geometric backoff, converges fast
            d1 = _fcfg(cand[r])[arm_dims[r]]
            moved = float(np.abs(d1 - d0).max())
            if moved <= budget * (1.0 + 1e-6) or cand[r] <= s[r] + eps:
              break
            cand[r] = s[r] + (cand[r] - s[r]) * max(budget / moved, 0.25)
          vel[r] = (cand[r] - s[r]) / dt
    for _ in range(2):                                       # project (steps are tiny; converges)
      for r in range(n):
        cand[r] = min(cand[r], max(s[r], cap_corridor(r, cand)))
    s = cand
    fq = np.empty(P.shape[1])
    for r in range(n):
      fq[arm_dims[r]] = _fcfg(s[r])[arm_dims[r]]
    q.append(fq)
    prog.append(list(s))
    phase.append("path")
    if fired < K and all(s[r] >= bar_s[fired] - eps for r in act[fired]) \
            and min(s) >= bar_s[fired] - eps:
      if ready is None:
        ready = steps
      if steps - ready >= n_dwell:
        fires.append((len(q) - 1, fired))
        for r in act[fired]:
          hold[r] = steps + n_dwell
        fired += 1
        ready = None
    else:
      ready = None
    if fired >= K and all(x >= D - eps for x in s):
      break
  prog = np.asarray(prog, float)
  if spread:
    prog = _corridor_spread(prog, bar_s, act, fires, cor, dt, D,
                            ramp_steps=int(round((ramp_dist / max(joint_rate, 1e-9)) / max(dt, 1e-12))))
    q = []
    for m in range(len(prog)):
      fq = np.empty(P.shape[1])
      for r in range(n):
        fq[arm_dims[r]] = _fcfg(prog[m, r])[arm_dims[r]]
      q.append(fq)
  for _ in range(int(round(settle / dt)) if dt > 0 else 0):
    q.append(q[-1])
    phase.append("settle")
  while len(phase) < len(q):
    phase.append("path")
  _t = TimedTrajectory(dt=dt, q=np.asarray(q, float), phase=phase[:len(q)],
                         fires=[(i, [k]) for i, k in fires], progress=prog)
  # how often the wrench cap was tighter than the never-stall floor (see vlim)
  _t.force_floor = dict(floor_hits)
  return _t


class CorridorStepper:
  """ONLINE Stage B/C: the SAME advance rule as `corridor_schedule`'s forward pass, exposed
  one tick at a time so the EXECUTOR can drive it with runtime information.

  Why online: the offline schedule must bake every slowdown in as worst-case pessimism (a
  static near-table cap slowed whole low carries, and barrier coupling turned even per-arm
  caps into global stalls). Here a slowdown exists only WHILE its trigger holds -- the caller
  passes per-arm rate factors each tick (proximity pacing, force guard) and the corridor keeps
  the timing certified: same conservative lookups as offline (`cap_corridor` against the
  OTHERS' minimum, world lead from the laggard's cell, floor/round grid indexing -- never
  interpolated), same no-pass-the-pending-barrier rule, same laggard-never-capped progress
  guarantee. The coarse certificate is honoured, not out-run: any state this stepper visits
  is inside the same certified set the offline scheduler used.

  NOT inherited from offline: `_corridor_spread` (its KNOWN HOLE exceeds wrench vmax at
  corners). Optional feed-forward pacing does use the future, but only through deterministic
  rollouts at a mode boundary: it selects fixed group rates for the whole stretch. There is
  no per-tick phase-error feedback, and runtime safety slowdowns remain explicit overrides.

  Interpolation is served from a PRECOMPUTED path cache: the offline pass paid ~2.4ms of
  Python per 5ms sample (config_at per arm per tick, x4 more in the rate-clamp backoff) --
  that was ~75% of a whole sim run's wall time. Vectorized lerp makes the per-tick cost
  microseconds, which is what makes running this inside the control loop viable at all.
  """

  def __init__(self, P, arm_dims, bar_s, acting, cor, dt, joint_rate=DEFAULT_JOINT_RATE,
               ramp_dist=DEFAULT_RAMP_DIST, ramp_min=DEFAULT_RAMP_MIN, dwell=0.0,
               vmax=None, amax=None, limit_ds=None, rate_clamp=False, fire_all=True,
               pace=False, settle_nom=0.0, trace=False):
    # fire_all=False: FIRE-FLOW gate -- an event fires as soon as its ACTING arms rest at the
    # barrier, without waiting for laggard arms to arrive. Laggards are still capped at the
    # pending barrier (no-pass), so they use the settle/servo window to arrive, never to cross.
    # pace=True: FEED-FORWARD ARRIVAL PACING. At the start of each mode stretch, roll the
    # deterministic corridor dynamics forward a few times and choose ONE rate factor per
    # corridor-coupled arm group. The factors stay fixed until the mode fires: ordinary
    # execution is therefore a planned phase profile, not a remaining-distance feedback law.
    # Acting groups reach the fire together; crossing groups reserve the caller-supplied upper
    # bound on pre-fire servo time. If the servo finishes early the barrier opens early and
    # they simply continue; if it consumes the budget they reach the barrier without waiting.
    self.P = np.asarray(P, float)
    self.segs, self.D = arc_length(list(self.P))
    self.csum = np.concatenate([[0.0], np.cumsum(self.segs)]) if self.segs else np.array([0.0])
    self._segs_arr = np.asarray(self.segs, float)
    self.arm_dims = [np.asarray(a, int) for a in arm_dims]
    self.n, self.K = len(arm_dims), len(bar_s)
    self.bar_s = list(bar_s)
    self.act = [set(int(a) for a in acting[k]) for k in range(self.K)]
    self.cor, self.dt = cor, float(dt)
    self.fire_all = bool(fire_all)
    self.pace = bool(pace)
    _sn = np.asarray(settle_nom, dtype=float)
    self.settle_nom = (float(_sn) if _sn.ndim == 0 else _sn.reshape(-1).copy())
    self._pace_tgt = None                                   # per-arm target arrival times (s)
    self._pace_factor = np.ones(self.n)                     # fixed within a mode stretch
    self._pace_k = -1                                       # barrier the profile belongs to
    self._pace_replans = 0
    self._in_preview = False
    self._pf_sum = np.zeros(self.n)
    self._pf_n = 0
    self._why = np.zeros((self.n, 8), dtype=np.int64)       # stop-cause histogram
    self._why_k = np.zeros((self.K, self.n), dtype=np.int64)  # at-cap ticks per barrier
    self._trace = [] if trace else None                     # (s, pace_f, rf, fired) per tick
    self.joint_rate, self.ramp_dist, self.ramp_min = joint_rate, ramp_dist, ramp_min
    self.vmax, self.amax, self.limit_ds, self.rate_clamp = vmax, amax, limit_ds, rate_clamp
    self.n_dwell = int(round(dwell / dt)) if dt > 0 else 0
    d = np.diff(self.P, axis=0)
    tot = np.linalg.norm(d, axis=1)
    self.share = np.ones((self.n, max(len(d), 1)))
    for r in range(self.n):
      sr = np.linalg.norm(d[:, self.arm_dims[r]], axis=1)
      ok = tot > 1e-12
      self.share[r][ok] = np.maximum(sr[ok] / tot[ok], 1e-3)
    self.rest = [sorted({0.0, self.D} | {bar_s[k] for k in range(self.K) if r in self.act[k]})
                 for r in range(self.n)]
    self.s = [0.0] * self.n
    self.vel = [0.0] * self.n
    self.fired, self.steps, self.ready = 0, 0, None
    self.hold = [-1] * self.n
    self.floor_hits = {}
    self.eps = 1e-6

  def _pace_groups(self, barrier):
    """Connected components that have to share one phase-rate factor on this stretch.

    A zero-width pairwise corridor anywhere between the current phase and ``barrier`` is a
    rigid coordination span. Giving its members independent arrival clocks merely makes the
    faster member bang into the corridor cap. Treating the component as one scheduling body
    removes that avoidable stop while leaving the per-tick corridor projection authoritative.
    """
    parent = list(range(self.n))

    def find(x):
      while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
      return x

    lo = min(self.s) if self.s else 0.0
    grid = np.asarray(self.cor["grid"], dtype=float)
    g0, g1 = _cor_at(self.cor, lo), _cor_at(self.cor, barrier)
    mask = np.zeros(len(grid), dtype=bool)
    mask[min(g0, g1):max(g0, g1) + 1] = True
    for i in range(self.n):
      for j in range(i + 1, self.n):
        A = self.cor["Lp"].get((i, j))
        if A is not None and np.any(np.max(A[mask], axis=1) <= self.eps):
          parent[find(i)] = find(j)
    groups = {}
    for r in range(self.n):
      groups.setdefault(find(r), []).append(r)
    return list(groups.values())

  def _pace_rollout(self, factors, barrier, max_ticks=60000):
    """Predict arrival ticks under fixed factors without changing live stepper state."""
    saved = (list(self.s), list(self.vel), self.fired, self.steps, self.ready,
             list(self.hold), dict(self.floor_hits), self._why.copy(), self._why_k.copy(),
             self.pace, self._in_preview)
    self.pace = False
    self._in_preview = True
    arrived = np.where(np.asarray(self.s) >= barrier - self.eps, 0.0, np.inf)
    for tick in range(1, max_ticks + 1):
      self.tick(_pace_override=factors)
      now = np.asarray(self.s) >= barrier - self.eps
      arrived[np.isinf(arrived) & now] = tick
      if np.isfinite(arrived).all():
        break
    (self.s, self.vel, self.fired, self.steps, self.ready, self.hold,
     self.floor_hits, self._why, self._why_k, self.pace, self._in_preview) = saved
    return arrived

  def _prepare_pace(self):
    """Build one fixed phase-rate schedule for the pending mode stretch.

    This deliberately does no tick-by-tick position correction. A natural rollout supplies
    feasible arrival times under the same ramps, wrench limits, acceleration limits and
    corridor projection as execution. A small fixed-point iteration then slows each coupled
    group so acting groups meet at one fire time and crossing groups meet at the bounded end
    of the pre-fire ceremony. The resulting factors are held until ``confirm_fire``.
    """
    if self.fired >= self.K:
      self._pace_factor = np.ones(self.n)
      self._pace_tgt = np.zeros(self.n)
      self._pace_k = self.fired
      return
    barrier = min(self.bar_s[self.fired], self.D)
    groups = self._pace_groups(barrier)
    natural = self._pace_rollout(np.ones(self.n), barrier)
    if not np.isfinite(natural).all():
      # The normal stepper will surface a real deadlock. Pacing must not invent a second,
      # harder-to-diagnose failure mode merely because its preview did not terminate.
      self._pace_factor = np.ones(self.n)
      self._pace_tgt = np.full(self.n, np.inf)
      self._pace_k = self.fired
      return

    settle = (float(self.settle_nom[min(self.fired, len(self.settle_nom) - 1)])
              if isinstance(self.settle_nom, np.ndarray) else float(self.settle_nom))
    ceremony = max(0.0, self.n_dwell * self.dt + settle)
    act = self.act[self.fired]
    natural_s = natural * self.dt
    act_t = float(np.max(natural_s[list(act)])) if act else 0.0
    crossing = [r for r in range(self.n) if r not in act]
    cross_t = float(np.max(natural_s[crossing])) if crossing else 0.0
    fire_t = max(act_t, cross_t - ceremony, self.dt)
    end_t = fire_t + ceremony
    targets = np.asarray([fire_t if r in act else end_t for r in range(self.n)])
    factors = np.ones(self.n)

    # A component containing an acting arm must arrive at fire time; a rigidly coupled
    # crossing arm cannot continue during the ceremony, which is a physical coordination
    # constraint rather than scheduler-induced waiting.
    group_targets = []
    for members in groups:
      group_targets.append(fire_t if any(r in act for r in members) else end_t)

    for _ in range(6):
      arrival = self._pace_rollout(factors, barrier) * self.dt
      if not np.isfinite(arrival).all():
        factors[:] = 1.0
        break
      changed = False
      for members, target in zip(groups, group_targets):
        actual = float(np.max(arrival[members]))
        if target <= self.dt:
          continue
        old = float(min(factors[r] for r in members))
        new = float(np.clip(old * actual / target, 0.02, 1.0))
        if abs(new - old) > 2e-3:
          changed = True
        for r in members:
          factors[r] = new
      if not changed:
        break

    # Report the target belonging to the entire coupled component, not merely to its
    # individual members. This keeps trace/diagnostic output honest at rigid spans.
    for members, target in zip(groups, group_targets):
      for r in members:
        targets[r] = target
    self._pace_factor = factors
    self._pace_tgt = targets
    self._pace_k = self.fired
    self._pace_replans += 1

  # -- fast path interpolation (vectorised config_at) ------------------------------------
  def _cfg(self, s):
    i = int(np.searchsorted(self.csum, s, side="right")) - 1
    i = min(max(i, 0), len(self._segs_arr) - 1) if len(self._segs_arr) else 0
    if not len(self._segs_arr) or self._segs_arr[i] < 1e-12:
      return self.P[min(i + 1, len(self.P) - 1)]
    a = (s - self.csum[i]) / self._segs_arr[i]
    return (1 - a) * self.P[i] + a * self.P[i + 1]

  def q_now(self):
    fq = np.empty(self.P.shape[1])
    for r in range(self.n):
      fq[self.arm_dims[r]] = self._cfg(self.s[r])[self.arm_dims[r]]
    return fq

  # -- the exact offline rules, one tick at a time ---------------------------------------
  def _lim_at(self, arr, s):
    if self.limit_ds:
      return int(min(max(round(s / self.limit_ds), 0), np.shape(arr)[1] - 1))
    return min(_cor_at(self.cor, s), np.shape(arr)[1] - 1)

  def _vlim(self, r, s):
    cap = float(self.vmax[r][self._lim_at(self.vmax, s)])
    fl = self.ramp_min * self.joint_rate
    if cap >= fl:
      return cap
    self.floor_hits[r] = self.floor_hits.get(r, 0) + 1
    return max(cap, 1e-4)

  def _ramp_of(self, r, s):
    if self.ramp_dist <= 0:
      return 1.0
    pts = self.rest[r]
    i = int(np.searchsorted(pts, s, side="right"))
    prev = pts[i - 1] if i > 0 else 0.0
    nxt = pts[i] if i < len(pts) else self.D
    return max(self.ramp_min, min(1.0, (min(s - prev, nxt - s) + 1e-6) / self.ramp_dist))

  def _cap_corridor(self, r, sv):
    lim = self.D
    others = [sv[j] for j in range(self.n) if j != r]
    if not others:
      return lim
    for j in range(self.n):
      if j == r:
        continue
      g = _cor_at(self.cor, min(sv[r], sv[j]))
      A = self.cor["Lp"].get((min(r, j), max(r, j)))
      if A is not None:
        lim = min(lim, sv[j] + (A[g, 0] if r < j else A[g, 1]))
    om = min(others)
    return min(lim, om + self.cor["Lw"][r][_cor_at(self.cor, om)])

  def lock_pair(self, i, j, lo, hi):
    """RUNTIME CEGAR: zero the corridor lead table for pair (i, j) over the arc window
    [lo, hi] -- the same local lock the offline refinement used on schedule-VERIFY
    counterexamples. The per-tick caps then force the pair together through the window, so
    a desync violation discovered by the oracle becomes a PERMANENT corridor constraint
    instead of a leaky one-tick hold: the composite reference can never re-enter the region
    (the diagonal itself is clear there -- measured 191mm -- so forcing the pair together
    resolves the overlap)."""
    _A = self.cor["Lp"].get((min(i, j), max(i, j)))
    if _A is None:
      return
    _g = np.asarray(self.cor["grid"], dtype=float)
    _msk = (_g >= lo) & (_g <= hi)
    if np.any(_A[_msk] > 1e-9):
      _A[_msk, :] = 0.0

  def tick(self, rate_factor=None, free_of=None, _pace_override=None):
    """Advance one dt. `rate_factor`: per-arm multiplier from RUNTIME triggers (1.0 = free;
    1/6 = proximity-paced; 0.0 = guard hold). Returns (q_ref, fire_ready) where fire_ready is
    the barrier index whose event should fire NOW (caller settles/seats/fires, then MUST call
    confirm_fire()), or None. `free_of`: optional callable fq -> False or a set of arm indices;
    if the CANDIDATE config of this tick is not free, the offending arms hold for THIS tick
    only -- a preventive gate, so a violation is never commanded (the old reactive check ran
    AFTER the drive and responded with 0.25s freezes + permanent corridor locks, which were
    exactly the standstills and the physics collision)."""
    self.steps += 1
    n, D, dt, eps = self.n, self.D, self.dt, self.eps
    cand = list(self.s)
    # One factor vector for this complete mode stretch. `_pace_override` is used only by the
    # deterministic rollout in `_prepare_pace`; it makes preview and execution share exactly
    # the same acceleration/rate ordering without abusing the runtime safety-factor channel.
    pace_f = _pace_override
    if pace_f is None and self.pace and self.fired < self.K:
      if self._pace_k != self.fired:
        self._prepare_pace()
      pace_f = self._pace_factor
    if pace_f is not None and not self._in_preview:
      self._pf_sum += pace_f
      self._pf_n += 1
    for r in range(n):
      if self.s[r] >= D - eps or self.steps <= self.hold[r]:
        self.vel[r] = 0.0
        continue
      lim = min(self.bar_s[self.fired] if self.fired < self.K else D, D)
      if self.s[r] < lim - eps:
        base = (self.joint_rate / self.share[r][_seg_index(self.csum, self.segs, self.s[r])]
                ) * self._ramp_of(r, self.s[r])
        want = base
        if self.vmax is not None:
          want = min(want, self._vlim(r, self.s[r]))
        if pace_f is not None and pace_f[r] < 1.0:
          # Planned pacing is a nominal rate, so it goes THROUGH the acceleration limiter.
          # Runtime safety factors below remain hard instantaneous caps. The old order put
          # both after amax, creating a velocity step whenever the feedback factor changed.
          want = min(want, base * float(pace_f[r]))
        if self.amax is not None:
          am = max(float(self.amax[r][self._lim_at(self.amax, self.s[r])]), 1e-9)
          want = float(np.clip(want, self.vel[r] - am * dt, self.vel[r] + am * dt))
        if rate_factor is not None and rate_factor[r] < 1.0:
          # RUNTIME SAFETY FACTORS ARE HARD CAPS, applied AFTER the accel clip. Routed through
          # amax (as a plain rate multiplier) the wrench decel limit turned a proximity trigger
          # into a gentle ramp-down -- the arm kept descending at speed through the whole
          # near-zone and grazed at 65N (measured, first online run). The offline pacing was
          # clean precisely because freezing the reference is INSTANT; match that: a factor
          # bounds the final rate this very tick, factor 0 = hold now.
          want = min(want, base * float(rate_factor[r]))
        self.vel[r] = want
        cand[r] = min(lim, self.s[r] + want * dt)
        if self.rate_clamp and cand[r] > self.s[r] + eps:
          d0 = self._cfg(self.s[r])[self.arm_dims[r]]
          budget = self.joint_rate * dt
          for _ in range(4):
            d1 = self._cfg(cand[r])[self.arm_dims[r]]
            moved = float(np.abs(d1 - d0).max())
            if moved <= budget * (1.0 + 1e-6) or cand[r] <= self.s[r] + eps:
              break
            cand[r] = self.s[r] + (cand[r] - self.s[r]) * max(budget / moved, 0.25)
          self.vel[r] = (cand[r] - self.s[r]) / dt
    _cand_pre = list(cand)
    for _ in range(2):
      for r in range(n):
        cand[r] = min(cand[r], max(self.s[r], self._cap_corridor(r, cand)))
    if free_of is not None:
      # PREVENTIVE COLLISION GATE: judge the candidate config BEFORE it is commanded; the
      # offending arms hold for this one tick (their stepper state stays consistent with what
      # is actually driven). The star-shaped certificate is unsound for simultaneous multi-arm
      # desync, and the old post-drive check let up to 10 ticks of penetration through before
      # freezing everyone for 0.25s -- the collision and the standstills the user saw.
      _fq = np.empty(self.P.shape[1])
      for _r in range(n):
        _fq[self.arm_dims[_r]] = self._cfg(cand[_r])[self.arm_dims[_r]]
      _who = free_of(_fq)
      if _who:
        for _r in _who:
          if cand[_r] > self.s[_r]:
            cand[_r] = self.s[_r]
            self.vel[_r] = 0.0
    # STOP-CAUSE HISTOGRAM (diagnostics): why did each arm not advance this tick?
    # 0 at the pending-barrier cap (no-pass gate), 1 post-fire hold, 2 pace factor < 0.05,
    # 3 servo rate factor 0 (guard), 4 path end, 5 moved, 6 corridor cap, 7 crawl.
    for r in range(n):
      if self.s[r] >= D - eps:
        self._why[r, 4] += 1
      elif self.steps <= self.hold[r]:
        self._why[r, 1] += 1
      elif self.fired < self.K and self.s[r] >= min(self.bar_s[self.fired], D) - eps:
        self._why[r, 0] += 1
        self._why_k[self.fired, r] += 1
      elif pace_f is not None and pace_f[r] < 0.05:
        self._why[r, 2] += 1
      elif rate_factor is not None and rate_factor[r] <= 0.0:
        self._why[r, 3] += 1
      elif _cand_pre[r] > self.s[r] + eps and cand[r] <= self.s[r] + eps:
        self._why[r, 6] += 1
      elif cand[r] > self.s[r] + eps:
        self._why[r, 5] += 1
      else:
        self._why[r, 7] += 1
    if self._trace is not None and not self._in_preview:
      self._trace.append((np.asarray(self.s, dtype=float).copy(),
                          np.asarray(pace_f, dtype=float).copy()
                          if pace_f is not None else np.ones(n),
                          np.asarray(rate_factor, dtype=float).copy()
                          if rate_factor is not None else np.ones(n),
                          self.fired))
    self.s = cand
    fire_ready = None
    if self.fired < self.K \
        and all(self.s[r] >= self.bar_s[self.fired] - eps
                or (rate_factor is not None and rate_factor[r] <= 0.0
                    and self.s[r] >= self.bar_s[self.fired] - 0.01)
                for r in self.act[self.fired]) \
        and (not self.fire_all or min(self.s) >= self.bar_s[self.fired] - eps):
      if self.ready is None:
        self.ready = self.steps
      if self.steps - self.ready >= self.n_dwell:
        fire_ready = self.fired
    else:
      self.ready = None
    return self.q_now(), fire_ready

  def confirm_fire(self, defer_pace_replan=False):
    old_factor = self._pace_factor.copy()
    for r in self.act[self.fired]:
      self.hold[r] = self.steps + self.n_dwell
    self.fired += 1
    self.ready = None
    if defer_pace_replan and self.pace:
      # Keep the just-executed constant rates through post-fire cleanup. The caller invokes
      # replan_pace() once cleanup actually completes, so its unknown duration does not make
      # us predict and then repeatedly correct the next stretch while some arms are frozen.
      self._pace_factor = old_factor
      self._pace_k = self.fired

  def replan_pace(self):
    """Invalidate the feed-forward profile; the next tick plans once from the live state."""
    self._pace_k = -1

  def done(self):
    return self.fired >= self.K and all(x >= self.D - self.eps for x in self.s)


def _corridor_spread(prog, bar_s, act, fires, cor, dt, D, ramp_steps=100):
  """Second pass: pace every arm so it ARRIVES AT EACH BARRIER exactly when that barrier's
  event fires, moving continuously in between -- instead of sprinting there and waiting.

  Anchors are ALL barriers (not just the arm's own): since no arm may pass the pending
  barrier, every arm has a fixed arrival there, and pacing to it is what turns "rush, stop,
  wait" into one smooth slower motion. The arm still ramps to rest at its OWN events; at a
  barrier it merely crosses, both ends of the span are full-speed, so it flows through.
  Timings of the anchors are untouched, so the firing schedule is unchanged.

  KNOWN HOLE (measured, unfixed): this pass does NOT respect corridor_schedule's `vmax`. It
  redistributes arc WITHIN an interval, so it can be locally faster than the forward pass was
  at the same arc -- measured on car-mini at 1.9x the wrench speed limit, at the very corner
  where the grasp then broke. Two repairs were tried and BOTH lost more than they gained,
  because this pass is also what keeps loaded arms off velocity steps: falling back to the
  forward profile on a violating interval restores its sprint-STOP-restart shape (cross8
  22/49 actions vs 39/40); water-filling the interval against the cap respects the speed but
  not the ACCELERATION, so it enters and leaves each capped corner in one step (car-mini
  13/20 vs 17/17). Doing this properly means an accel-aware re-pacing (a small TOPP) here,
  not a clamp."""
  prog = np.array(prog, float, copy=True)
  M, n = prog.shape
  for r in range(n):
    anchors = [(0, prog[0, r], True)]                        # (sample, arc, rest_here)
    for idx, k in fires:
      if 0 < idx < M - 1:
        anchors.append((int(idx), float(bar_s[k]), r in act[k]))
    anchors.append((M - 1, prog[-1, r], True))
    for (i0, s0, rest0), (i1, s1, rest1) in zip(anchors, anchors[1:]):
      if i1 - i0 < 3 or s1 - s0 <= 1e-9:
        continue
      w = _fill_shape2(i1 - i0, ramp_steps if rest0 else 0, ramp_steps if rest1 else 0)
      prog[i0 + 1:i1 + 1, r] = s0 + np.cumsum(w) * (s1 - s0)
  for m in range(M):                                         # re-clamp into the corridor
    for r in range(n):
      lim = D
      for j in range(n):
        if j == r:
          continue
        g = _cor_at(cor, min(prog[m, r], prog[m, j]))
        A = cor["Lp"].get((min(r, j), max(r, j)))
        if A is not None:
          lim = min(lim, prog[m, j] + (A[g, 0] if r < j else A[g, 1]))
      lim = min(lim, prog[m].min() + cor["Lw"][r][_cor_at(cor, prog[m].min())])
      prog[m, r] = min(prog[m, r], lim)
    if m:                                                    # keep monotone after clamping
      prog[m] = np.maximum(prog[m], prog[m - 1])
  return prog


# =============================================================================================
# C1 PATH SMOOTHING (the --spline variant of corridor retiming)
#
# The saved path is piecewise linear: C0. Every waypoint is a velocity DISCONTINUITY, which is
# why a curvature-based wrench limit collapses there (a kink differentiates to near-infinite
# p'') and why crossing arms had to be throttled at barriers. Fitting a C1 curve removes the
# kinks outright, so smoothness stops being something the time-parameterization has to fake.
#
# CATMULL-ROM rather than a fitted B-spline, deliberately: it is C1 AND INTERPOLATES every
# waypoint exactly. That matters here because the barrier waypoints ARE the task witnesses --
# a mate must happen at the config the IK certified, so a spline that merely approximates the
# waypoints would move the seat. Catmull-Rom changes the geometry only BETWEEN waypoints.
#
# The deviation still has to be certified: `spline_path` returns the densified path and the
# caller checks it with the same oracle, shrinking `tension` (0 = straight back to the
# original polyline) wherever a span fails -- so worst case degrades to the certified input.
# =============================================================================================
def catmull_rom(P, per_seg=8, tension=0.5, pin=None):
  """Densify waypoints `P` into a C1 curve through ALL of them (Catmull-Rom).

  `per_seg`: samples generated per original segment. `tension` 0.5 = uniform Catmull-Rom,
  0 = the original polyline (no deviation). `pin`: optional set of waypoint indices whose
  ADJACENT spans stay straight -- use it to keep an exact linear approach into/out of a
  witness (an insertion must arrive along the mate axis, not on a curve).

  Returns (Q, index_map) where Q is the densified path and index_map[i] is the index in Q of
  original waypoint i, so barrier bookkeeping can be carried over unchanged."""
  P = [np.asarray(p, float) for p in P]
  n = len(P)
  if n < 2:
    return np.asarray(P, float), [0] if n else []
  pin = set(pin or ())
  Q, imap = [P[0]], [0]
  for i in range(n - 1):
    p0 = P[i - 1] if i > 0 else P[i]
    p1, p2 = P[i], P[i + 1]
    p3 = P[i + 2] if i + 2 < n else P[i + 1]
    straight = (i in pin) or (i + 1 in pin) or tension <= 0.0
    m1 = np.zeros_like(p1) if straight else tension * (p2 - p0)
    m2 = np.zeros_like(p1) if straight else tension * (p3 - p1)
    for k in range(1, per_seg + 1):
      t = k / per_seg
      if straight:
        Q.append((1 - t) * p1 + t * p2)
      else:                                                  # cubic Hermite = C1 at knots
        t2, t3 = t * t, t * t * t
        Q.append((2 * t3 - 3 * t2 + 1) * p1 + (t3 - 2 * t2 + t) * m1
                 + (-2 * t3 + 3 * t2) * p2 + (t3 - t2) * m2)
    imap.append(len(Q) - 1)
  return np.asarray(Q, float), imap


def path_deviation(Q, imap, P):
  """Max distance (per joint-space norm) from the smoothed curve to the original polyline
  segment it replaces -- the number that has to be justified to a collision checker."""
  P = [np.asarray(p, float) for p in P]
  worst = 0.0
  for i in range(len(imap) - 1):
    a, b = P[i], P[i + 1]
    d = b - a
    dd = float(d @ d)
    for j in range(imap[i], imap[i + 1] + 1):
      t = 0.0 if dd < 1e-12 else float(np.clip((Q[j] - a) @ d / dd, 0.0, 1.0))
      worst = max(worst, float(np.linalg.norm(Q[j] - (a + t * d))))
  return worst


def spline_path(P, bar_idx, per_seg=8, tension=0.5, free_of=None, pin_barriers=True,
                shrink=(0.5, 0.25, 0.0)):
  """C1-smooth a path and CERTIFY the deviation, per span, shrinking where it fails.

  `bar_idx`: waypoint indices that are barriers (task witnesses). With `pin_barriers` their
  adjacent spans stay straight, so every event still happens exactly on its witness and the
  approach/retreat direction is preserved.
  `free_of(Q, j0, j1, i)`: called per smoothed span with the densified path, its index
  range, and the ORIGINAL span index `i` (so the caller can pick the right mode / world
  state); return True if that span is collision-free. None = accept unchecked (unsafe).

  Each span is tried at `tension`, then at each value in `shrink` until it certifies; 0.0 is
  the original straight segment, so the result is never worse than the input path. Returns
  (Q, index_map, report)."""
  P = [np.asarray(p, float) for p in P]
  pin = set(bar_idx) if pin_barriers else set()
  Q, imap = catmull_rom(P, per_seg=per_seg, tension=tension, pin=pin)
  if free_of is None:
    return Q, imap, {"checked": 0, "shrunk": 0, "straightened": 0,
                     "deviation": path_deviation(Q, imap, P)}
  checked = shrunk = straight = 0
  for i in range(len(imap) - 1):
    if i in pin or (i + 1) in pin:
      continue                                               # pinned: already straight
    checked += 1
    if free_of(Q, imap[i], imap[i + 1], i):
      continue
    for t_try in shrink:                                     # pull back toward the polyline
      Qi, _ = catmull_rom(P[max(i - 1, 0):i + 3], per_seg=per_seg,
                          tension=t_try, pin={0, 1, 2, 3} if t_try <= 0 else ())
      # rebuild just this span at the reduced tension
      sub, _ = catmull_rom([P[i - 1] if i > 0 else P[i], P[i], P[i + 1],
                            P[i + 2] if i + 2 < len(P) else P[i + 1]],
                           per_seg=per_seg, tension=t_try)
      lo, hi = imap[i], imap[i + 1]
      span = sub[per_seg:2 * per_seg + 1] if i > 0 else sub[:per_seg + 1]
      if len(span) == hi - lo + 1:
        Q[lo:hi + 1] = span
      if free_of(Q, lo, hi, i):
        shrunk += 1
        if t_try <= 0:
          straight += 1
        break
  return Q, imap, {"checked": checked, "shrunk": shrunk, "straightened": straight,
                   "deviation": path_deviation(Q, imap, P)}
