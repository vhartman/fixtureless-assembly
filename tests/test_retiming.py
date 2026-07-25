"""Pin the extracted retiming math to the pre-refactor sim behavior + the portable
trajectory contract (input = sequence/DAG + collision-free path, output = q(t) with
v=0 at goals). Pure numpy -- no sim/MuJoCo/rai needed."""
import numpy as np
import pytest

import retiming


def _old_drive_stretch(qs, dt, joint_rate, ramp_dist, ramp_min):
  """Verbatim copy of the inline profile that lived in sim.execute_motion_plan,
  so a drift in retime_stretch is caught immediately."""
  qs = [np.asarray(q, float) for q in qs]
  segs = [float(np.linalg.norm(b - a)) for a, b in zip(qs, qs[1:])]
  D = float(sum(segs))
  out = []
  if D < 1e-9:
    return out
  csum = np.concatenate([[0.0], np.cumsum(segs)])

  def config_at(s):
    i = min(int(np.searchsorted(csum, s, side="right")) - 1, len(segs) - 1)
    if segs[i] < 1e-12:
      return qs[i + 1]
    a = (s - csum[i]) / segs[i]
    return (1 - a) * qs[i] + a * qs[i + 1]

  s = 0.0
  while s < D:
    ramp = min(1.0, (min(s, D - s) + 1e-6) / ramp_dist) if ramp_dist > 0 else 1.0
    s = min(D, s + joint_rate * max(ramp_min, ramp) * dt)
    out.append(config_at(s))
  return out


def _old_split(states, full_q):
  """Verbatim copy of the inline stretch decomposition (mode handles = st.mode)."""
  mode = states[0].mode
  stretches = []
  cur = [full_q(states[0].q)]
  for st in states[1:]:
    q = full_q(st.q)
    if st.mode != mode:
      done = [int(t) for t in np.atleast_1d(mode.task_ids)
              if t not in np.atleast_1d(st.mode.task_ids)]
      stretches.append((cur, sorted(set(done))))
      cur = [cur[-1]]
      mode = st.mode
    cur.append(q)
  stretches.append((cur, sorted(set(int(t) for t in np.atleast_1d(mode.task_ids)))))
  return stretches


DT, RATE, RD, RM = 0.002, 1.0, 0.2, 0.04

STRETCHES = [
    [np.zeros(6), np.array([0.5, 0, 0, 0, 0, 0]), np.array([0.5, 0.3, 0, 0, 0, 0])],
    [np.zeros(3), np.array([1.7, -0.4, 0.2])],                # long single segment
    [np.array([1.0, 1.0]), np.array([1.0, 1.0])],             # degenerate (D=0)
]


@pytest.mark.parametrize("qs", STRETCHES)
def test_retime_stretch_matches_old(qs):
  new = list(retiming.retime_stretch(qs, DT, RATE, RD, RM))
  old = _old_drive_stretch(qs, DT, RATE, RD, RM)
  assert len(new) == len(old)
  for a, b in zip(new, old):
    assert np.allclose(a, b)


def test_final_sample_lands_on_goal_at_rest():
  qs = STRETCHES[0]
  samples = list(retiming.retime_stretch(qs, DT, RATE, RD, RM))
  assert np.allclose(samples[-1], qs[-1])                    # arrives exactly at the goal
  step = np.linalg.norm(samples[-1] - samples[-2])
  assert step < RATE * DT + 1e-9                             # v ramped down -> tiny final step


def test_short_stretch_stays_below_cruise():
  # a stretch shorter than 2*ramp_dist never reaches full JOINT_RATE (all-ramp triangle)
  qs = [np.zeros(2), np.array([0.05, 0.0])]
  steps = [np.linalg.norm(b - a) for a, b in
           zip(list(retiming.retime_stretch(qs, DT, RATE, RD, RM))[:-1],
               list(retiming.retime_stretch(qs, DT, RATE, RD, RM))[1:])]
  assert max(steps) < RATE * DT                              # capped by the ramp, not cruise


def test_accel_profile_zero_velocity_endpoints():
  qs = STRETCHES[0]
  samples = list(retiming.retime_stretch(qs, DT, RATE, RD, RM, accel=2.0))
  assert np.allclose(samples[-1], qs[-1])
  # both ends ramp: first and last inter-sample steps are small vs the mid-trajectory step
  d = [np.linalg.norm(b - a) for a, b in zip(samples, samples[1:])]
  assert d[0] < max(d) and d[-1] < max(d)


class _Mode:
  def __init__(self, ids):
    self.task_ids = np.asarray(ids)


class _St:
  def __init__(self, q, mode):
    self.q = np.asarray(q, float)
    self.mode = mode


def test_split_stretches_matches_old():
  m0, m1, m2 = _Mode([0]), _Mode([0, 1]), _Mode([0, 1, 2])
  path = [_St([0, 0], m0), _St([1, 0], m0),
          _St([1, 0], m1), _St([1, 1], m1),          # transition -> task 0 not done here
          _St([1, 1], m2), _St([2, 2], m2)]
  full_q = lambda q: np.asarray(q, float)
  new = retiming.split_stretches(path, q_of=lambda s: full_q(s.q),
                                 mode_of=lambda s: s.mode,
                                 taskids_of=lambda m: np.atleast_1d(m.task_ids))
  old = _old_split(path, full_q)
  assert len(new) == len(old)
  for (nq, nd), (oq, od) in zip(new, old):
    assert nd == od
    assert len(nq) == len(oq) and all(np.allclose(a, b) for a, b in zip(nq, oq))


def _two_arm_stretches():
  # 2 arms (1 dof each): arm0 moves 0->1->2 (mate at each barrier), arm1 moves 0->3->6.
  # barrier0 done by task 10 (arm0 acting), barrier1 done by task 11 (arm0 acting).
  s0 = ([np.array([0.0, 0.0]), np.array([1.0, 3.0])], [10])
  s1 = ([np.array([1.0, 3.0]), np.array([2.0, 6.0])], [11])
  arm_dims = [np.array([0]), np.array([1])]
  acting = [{0}, {0}]                                        # only arm0 rests at both barriers
  return [s0, s1], arm_dims, acting


def test_schedule_only_acting_arm_rests_others_cross():
  stretches, arm_dims, acting = _two_arm_stretches()
  traj = retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM)
  q = traj.q
  # arm0 (acting at both barriers) must pass through v~=0 at its barrier configs (x=1)
  x0 = q[:, 0]
  near1 = np.where(np.abs(x0 - 1.0) < 1e-3)[0]
  assert len(near1) > 1                                      # dwelt/ramped around the mate config
  # arm1 (never acting) should NOT stall at x=3 (its barrier config) -- it crosses smoothly
  x1 = q[:, 1]
  near3 = np.where(np.abs(x1 - 3.0) < 1e-3)[0]
  assert len(near3) <= 2                                     # at most the sample straddling it
  # both events fired, in order, and end at the goal
  assert [e for _, e in traj.fires] == [[10], [11]]
  assert np.allclose(q[-1], [2.0, 6.0])


def test_schedule_fires_in_dag_order_at_rest():
  stretches, arm_dims, acting = _two_arm_stretches()
  traj = retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM)
  for k, (idx, events) in enumerate(traj.fires):
    # at each fire, the acting arm (arm0) sits on its barrier config at rest
    expected_x = 1.0 if k == 0 else 2.0
    assert abs(traj.q[idx, 0] - expected_x) < 1e-3
    if idx > 0:
      assert np.linalg.norm(traj.q[idx] - traj.q[idx - 1]) < RATE * DT + 1e-6


def test_schedule_single_arm_matches_build_trajectory():
  # one arm, both barriers acting -> no crossing freedom -> same samples as the composite trapezoid
  st = [([np.array([0.0]), np.array([1.0])], [10]),
        ([np.array([1.0]), np.array([2.0])], [11])]
  arm_dims = [np.array([0])]
  sched = retiming.plan_schedule(st, arm_dims, [{0}, {0}], DT, RATE, RD, RM)
  comp = retiming.build_trajectory(st, DT, RATE, RD, RM)
  comp_path = np.array([q for q, ph in zip(comp.q, comp.phase) if ph == "path"])
  assert np.allclose(sched.q, comp_path)


def test_schedule_priority_lower_arm_yields():
  # Shared 1-D lane: a "gate" forbids arm1 in (1.5,2.5) WHILE arm0 hasn't cleared (x0<3).
  # Yielding suffices: arm1 waits below the gate until arm0 passes 3, then proceeds.
  st = [([np.array([0.0, 0.0]), np.array([4.0, 4.0])], [10])]
  arm_dims = [np.array([0]), np.array([1])]                  # arm0=x0, arm1=x1, move together
  gate = lambda fq, fired=0: 1.5 < fq[1] < 2.5 and fq[0] < 3.0
  traj = retiming.plan_schedule(st, arm_dims, [{0}], DT, RATE, RD, RM,
                                priority=[0, 1], collide=gate)
  assert not any(gate(fq) for fq in traj.q)                  # arm1 never entered the gate early
  assert (traj.q[:, 0] - traj.q[:, 1]).max() > 1.0           # arm0 pulled well ahead (arm1 held)
  assert np.allclose(traj.q[-1], [4.0, 4.0])                 # both still reach their goals
  assert [e for _, e in traj.fires] == [[10]]


def test_schedule_no_collide_is_pure_desync():
  # without a collide predicate the two arms desync freely (no waiting) -- baseline
  stretches, arm_dims, acting = _two_arm_stretches()
  a = retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM)
  b = retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM,
                             priority=[0, 1], collide=lambda fq, fired: False)
  assert a.q.shape == b.q.shape and np.allclose(a.q, b.q)    # a no-op collide changes nothing


def test_schedule_world_evolution_wait():
  # An AHEAD arm blocked by a part that MOVES at event 0: collide depends on `fired`.
  # arm1 crosses barrier 0's arc without stopping but may not pass x1=4 (beyond the
  # barrier -- certified paths only block ahead-arms this way; the fire gate guarantees
  # laggard lanes match the certificate) until event 0 fired. Wait, then cross, finish.
  stretches, arm_dims, acting = _two_arm_stretches()         # barrier0 acted by arm0 only
  blocked_lane = lambda fq, fired: fired < 1 and fq[1] > 4.0
  traj = retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM,
                                collide=blocked_lane)
  i0 = traj.fires[0][0]                                      # sample where event 0 fired
  assert traj.q[:i0 + 1, 1].max() <= 4.0 + 1e-6              # arm1 held out of the lane before
  assert traj.q[i0, 1] > 3.0 + 1e-6                          # ...but DID cross the barrier arc
  assert np.allclose(traj.q[-1], [2.0, 6.0])                 # and still reached its goal after
  assert [e for _, e in traj.fires] == [[10], [11]]


def test_schedule_deadlock_raises():
  # arm1 permanently blocked AND acting at the barrier -> the event can never fire.
  # This must be a LOUD RuntimeError (never silent truncation or an infinite loop).
  stretches, arm_dims, _ = _two_arm_stretches()
  acting = [{0, 1}, {0, 1}]                                  # arm1 must reach barrier... but:
  wall = lambda fq, fired: fq[1] > 1.0                       # ...it can never pass x1=1
  with pytest.raises(RuntimeError, match="DEADLOCK"):
    retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM,
                           collide=wall, stall_steps=200)


def test_schedule_corridor_formation_holds():
  # Tight-formation corridor: the pair must stay within 0.5 offset (like two capsule arms
  # tracking the certified diagonal). arm0 rests 1s at its event while arm1 crosses -- the
  # corridor stops arm1 at the bound (leader waits for laggard), then both proceed.
  st = [([np.array([0.0, 0.0]), np.array([2.0, 2.0])], [10]),
        ([np.array([2.0, 2.0]), np.array([6.0, 6.0])], [11])]
  arm_dims = [np.array([0]), np.array([1])]
  corridor = lambda fq, fired: {0, 1} if fq[1] - fq[0] > 0.5 else False
  traj = retiming.plan_schedule(st, arm_dims, [{0}, {0, 1}], DT, RATE, RD, RM,
                                collide=corridor, dwell=1.0)
  off = traj.q[:, 1] - traj.q[:, 0]
  assert off.max() <= 0.55                                   # formation held (cell-edge slack)
  assert off.max() > 0.3                                     # ...but real desync DID happen
  assert np.allclose(traj.q[-1], [6.0, 6.0])                 # both finished
  assert [e for _, e in traj.fires] == [[10], [11]]


def test_arm_rate_boost_values():
  # 2 arms of 1 dof. Both move equally -> each gets 1/sqrt(2) of composite -> boost sqrt(2).
  qs = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 1.0])]
  b = retiming.arm_rate_boost(qs, [np.array([0]), np.array([1])])
  assert np.isclose(b[0], np.sqrt(2), atol=1e-6)             # both moving: full dilution
  assert np.isclose(b[1], 1.0, atol=1e-6)                    # one arm moving: no boost
  assert (b <= np.sqrt(2) + 1e-9).all()                      # bounded by sqrt(#arms)


def test_arm_rate_boost_same_path_faster():
  # THE SAFETY PROPERTY: boosting a COMMON clock rescales time only -- the geometric path
  # is unchanged, so the collision certificate carries over. Verify the boosted samples all
  # lie on the unboosted path (same curve), and that it takes fewer samples.
  qs = [np.zeros(2), np.array([1.0, 1.0]), np.array([2.0, 2.0])]
  dims = [np.array([0]), np.array([1])]
  plain = np.array(list(retiming.retime_stretch(qs, DT, RATE, RD, RM)))
  fast = np.array(list(retiming.retime_stretch(qs, DT, RATE, RD, RM, arm_dims=dims)))
  assert len(fast) < len(plain)                              # genuinely faster
  assert np.allclose(fast[:, 0], fast[:, 1])                 # still on the x0==x1 path
  assert np.allclose(fast[-1], qs[-1])                       # still lands on the goal
  # every boosted sample lies on the segment set of the original path (max deviation ~0)
  assert float(np.abs(fast[:, 0] - fast[:, 1]).max()) < 1e-9


def test_schedule_rate_boost_preserves_events():
  stretches, arm_dims, acting = _two_arm_stretches()
  a = retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM)
  b = retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM, rate_boost=True)
  assert len(b.q) < len(a.q)                                 # faster
  assert [e for _, e in b.fires] == [e for _, e in a.fires]  # same events, same order
  assert np.allclose(b.q[-1], a.q[-1])                       # same final config


def test_spread_removes_rush_then_wait():
  # arm1 has NO events (it never acts), so under the schedule it sprints to the end and
  # sits. --smooth must spread its motion over the whole run: no long stall, same endpoints.
  stretches, arm_dims, _ = _two_arm_stretches()
  acting = [{0}, {0}]                                        # only arm0 acts, at both barriers
  P, bidx = retiming._composite(stretches)
  segs, D = retiming.arc_length(list(P))
  csum = np.concatenate([[0.0], np.cumsum(segs)])
  bar = [csum[b] for b in bidx]
  traj = retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM)
  v_before = np.abs(np.diff(traj.q[:, 1])) / DT
  q2 = retiming.spread_between_events(traj, arm_dims, bar, acting, DT, RATE,
                                      ramp_dist=RD, P=P, csum=csum, segs=segs)
  v_after = np.abs(np.diff(q2[:, 1])) / DT
  assert np.allclose(q2[0], traj.q[0]) and np.allclose(q2[-1], traj.q[-1])   # same endpoints
  assert np.mean(v_after < 0.05 * RATE) < 0.25 * np.mean(v_before < 0.05 * RATE)  # far less idling
  assert (np.diff(q2[:, 1]) > -1e-9).all()                                   # still monotone
  # arrivals are preserved: at every fire the acting arm sits on its barrier config at rest
  for k, (idx, _) in enumerate(traj.fires):
    if 0 in acting[k]:
      assert np.allclose(q2[idx, 0], traj.q[idx, 0], atol=1e-6)
      assert abs(q2[idx, 0] - q2[max(idx - 1, 0), 0]) < RATE * DT + 1e-9


def test_spread_respects_collide_veto():
  # a collide() that rejects everything must leave the trajectory exactly as scheduled
  stretches, arm_dims, _ = _two_arm_stretches()
  acting = [{0}, {0}]
  P, bidx = retiming._composite(stretches)
  segs, _ = retiming.arc_length(list(P))
  csum = np.concatenate([[0.0], np.cumsum(segs)])
  bar = [csum[b] for b in bidx]
  traj = retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM)
  q2 = retiming.spread_between_events(traj, arm_dims, bar, acting, DT, RATE, ramp_dist=RD,
                                      P=P, csum=csum, segs=segs,
                                      collide=lambda fq, fired: True, fired_of=lambda i: 0)
  assert np.allclose(q2, traj.q)                             # vetoed -> unchanged


def test_schedule_pocket_coupling_restart():
  # Coordination-space POCKET whose diagonal IS free (like a BiRRT-certified path):
  # greedy desync wedges (arm1 held at x1=2.7 by V1 while arm0 dwells; arm0's resume then
  # blocks on V2 -> mutual block, coupled step also blocked). The CEGAR restart must infer
  # a sync band for the pair; the coupled re-run tracks the free diagonal, stays MONOTONE
  # (no reversal), and finishes.
  st = [([np.array([0.0, 0.0]), np.array([2.0, 2.0])], [10]),
        ([np.array([2.0, 2.0]), np.array([6.0, 6.0])], [11])]
  arm_dims = [np.array([0]), np.array([1])]
  def pocket(fq, fired):
    x0, x1 = fq[0], fq[1]
    if 2.7 < x1 < 3.5 and 1.8 < x0 < 2.35:                   # V1: arm1 held while arm0 behind
      return {0, 1}
    if 2.2 < x0 < 2.4 and 2.6 < x1 < 2.8:                    # V2: arm0's resume blocked: pocket
      return {0, 1}
    return False
  traj = retiming.plan_schedule(st, arm_dims, [{0}, {1}], DT, RATE, RD, RM,
                                collide=pocket, dwell=1.0)
  assert np.allclose(traj.q[-1], [6.0, 6.0])                 # finished despite the pocket
  assert (np.diff(traj.q[:, 0]) > -1e-9).all()               # monotone: no reversal needed
  assert (np.diff(traj.q[:, 1]) > -1e-9).all()
  assert not any(pocket(fq, 0) for fq in traj.q)             # never sampled inside a violation
  assert [e for _, e in traj.fires] == [[10], [11]]


def test_schedule_dwell_rests_acting_arm_only():
  # dwell > 0: the acting arm rests at its barrier around the fire; the crossing arm
  # keeps moving through the same window (no all-arms freeze).
  stretches, arm_dims, acting = _two_arm_stretches()
  traj = retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM, dwell=0.05)
  i0 = traj.fires[0][0]
  w = int(0.05 / DT)                                         # one dwell window of samples
  a0 = traj.q[i0 - w:i0 + w, 0]                              # acting arm around the fire
  a1 = traj.q[i0 - w:i0 + w, 1]                              # crossing arm around the fire
  assert np.ptp(a0) < 1e-9                                   # at rest through pre+post dwell
  assert np.ptp(a1) > 0.01                                   # crossing arm kept moving


def test_build_trajectory_fires_at_rest_and_concats_path():
  m0, m1 = _Mode([0]), _Mode([0, 1])
  path = [_St([0, 0], m0), _St([0.5, 0], m0), _St([0.5, 0], m1), _St([0.5, 0.4], m1)]
  stretches = retiming.split_stretches(path, q_of=lambda s: s.q,
                                       mode_of=lambda s: s.mode,
                                       taskids_of=lambda m: np.atleast_1d(m.task_ids))
  traj = retiming.build_trajectory(stretches, DT, RATE, RD, RM, dwell=0.01, settle=0.02)
  # streamed path samples equal the trajectory's "path"-phase samples, in order
  streamed = [c for qs, _ in stretches for c in retiming.retime_stretch(qs, DT, RATE, RD, RM)]
  path_samples = [q for q, ph in zip(traj.q, traj.phase) if ph == "path"]
  assert len(streamed) == len(path_samples)
  # every fire index sits on a dwell sample (at rest) and matches a stretch goal
  goals = [qs[-1] for qs, _ in stretches]
  for k, (idx, events) in enumerate(traj.fires):
    assert traj.phase[idx] == "dwell"
    assert np.allclose(traj.q[idx], goals[k])
    assert events == stretches[k][1]


# ---- corridor retiming (stages A/B/C) --------------------------------------------------
def test_certify_corridor_measures_known_width():
  # synthetic geometry: arms may lead each other by at most 1.0 rad; world allows 2.0
  cor = retiming.certify_corridor(
      2, 10.0,
      world_free=lambda i, s, d: abs(d) <= 2.0,
      pair_free=lambda i, j, s, d: abs(d) <= 1.0,
      ds_grid=1.0, ds_lead=0.25, lead_max=3.0)
  assert np.allclose(cor["Lp"][(0, 1)][:, 0], 1.0)           # measured the pair bound
  assert np.allclose(cor["Lw"][0], 2.0)                      # and the world bound
  assert "arm0-arm1" in retiming.corridor_width_report(cor)


def test_certify_corridor_locked_span_is_zero():
  cor = retiming.certify_corridor(
      2, 10.0, world_free=lambda i, s, d: True, pair_free=lambda i, j, s, d: True,
      ds_grid=1.0, ds_lead=0.5, lead_max=2.0, locked=[(0, 1, 3.0, 6.0)])
  g = cor["grid"]
  assert (cor["Lp"][(0, 1)][(g >= 3) & (g <= 6)] == 0).all()  # rigid span: no drift allowed
  assert (cor["Lp"][(0, 1)][g < 3] > 0).all()                 # elsewhere still free


def test_corridor_schedule_uninvolved_arm_does_not_stop():
  # arm1 acts at NEITHER barrier -> under corridor retiming it must cross both without
  # coming to rest, while arm0 (acting at both) does stop.
  P = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0])]
  arm_dims = [np.array([0]), np.array([1])]
  segs, D = retiming.arc_length(P)
  csum = np.concatenate([[0.0], np.cumsum(segs)])
  bar = [csum[1], csum[2]]
  cor = retiming.certify_corridor(2, D, lambda i, s, d: True, lambda i, j, s, d: True,
                                  ds_grid=0.25, ds_lead=0.25, lead_max=3.0)
  traj = retiming.corridor_schedule(P, arm_dims, bar, [{0}, {0}], cor, DT, RATE, RD, RM)
  v0 = np.abs(np.diff(traj.q[:, 0])) / DT
  v1 = np.abs(np.diff(traj.q[:, 1])) / DT
  i0 = traj.fires[0][0]                                      # first event
  assert v0[max(i0 - 2, 0)] < 0.5 * RATE                     # acting arm at rest for its event
  assert v1[max(i0 - 2, 0)] > 0.5 * RATE                     # crossing arm sails through
  assert np.mean(v1 < 0.05 * RATE) < 0.05                    # and barely idles at all
  assert np.allclose(traj.q[-1], P[-1])                      # both still finish
  assert (np.diff(traj.q[:, 1]) > -1e-9).all()               # monotone


def test_corridor_schedule_zero_corridor_forces_lockstep():
  # a zero-width corridor = "these arms must move together": progress stays equal throughout
  P = [np.array([0.0, 0.0]), np.array([2.0, 2.0])]
  arm_dims = [np.array([0]), np.array([1])]
  segs, D = retiming.arc_length(P)
  cor = retiming.certify_corridor(2, D, lambda i, s, d: True, lambda i, j, s, d: False,
                                  ds_grid=0.5, ds_lead=0.25, lead_max=2.0)
  traj = retiming.corridor_schedule(P, arm_dims, [D], [{0}], cor, DT, RATE, RD, RM,
                                    spread=False)
  assert np.abs(traj.progress[:, 0] - traj.progress[:, 1]).max() < 0.05   # moved as one
  assert np.allclose(traj.q[-1], P[-1])


def _flat_corridor(n, D, ds=0.25):
  return retiming.certify_corridor(n, D, lambda i, s, d: True, lambda i, j, s, d: True,
                                   ds_grid=ds, ds_lead=ds, lead_max=3.0)


def test_brakeable_makes_limit_reachable():
  # a speed limit that drops off a cliff must be pulled down BEFORE the cliff so an arm
  # can decelerate into it (v^2 <= v_next^2 + 2*a*ds)
  v = np.array([[5.0, 5.0, 5.0, 0.5]])
  a = np.full((1, 4), 1.0)
  out = retiming.brakeable(v, a, 1.0)
  assert out[0, 3] == 0.5                                    # the cliff itself is unchanged
  assert out[0, 2] < 5.0 and out[0, 1] < 5.0                 # approach is braked in advance
  assert out[0, 2] <= np.sqrt(0.5 ** 2 + 2 * 1.0) + 1e-9     # and by exactly the accel bound


def test_corridor_schedule_respects_speed_limit():
  # a tight vmax on arm0 only: arm0 obeys it, arm1 is untouched, both still finish
  P = [np.array([0.0, 0.0]), np.array([3.0, 3.0])]
  arm_dims = [np.array([0]), np.array([1])]
  segs, D = retiming.arc_length(P)
  cor = _flat_corridor(2, D)
  G = len(cor["grid"])
  vmax = np.full((2, G), 1e6)
  vmax[0] = 0.2                                              # arc units/s, well under cruise
  amax = np.full((2, G), 1e6)
  traj = retiming.corridor_schedule(P, arm_dims, [D], [{0}], cor, DT, RATE, RD, RM,
                                    vmax=vmax, amax=amax, spread=False)
  v0 = np.abs(np.diff(traj.q[:, 0])) / DT
  share = 1.0 / np.sqrt(2.0)                                 # arm0's share of the composite arc
  assert v0.max() <= 0.2 * share + 1e-3                      # obeyed, in ITS joint units
  assert np.allclose(traj.q[-1], P[-1])                      # still arrives


def test_corridor_schedule_accel_limit_bounds_jerk_in():
  # amax must stop the arm from jumping to speed in one step
  P = [np.array([0.0]), np.array([3.0])]
  arm_dims = [np.array([0])]
  segs, D = retiming.arc_length(P)
  cor = _flat_corridor(1, D)
  G = len(cor["grid"])
  amax = np.full((1, G), 0.5)                                # rad/s^2
  traj = retiming.corridor_schedule(P, arm_dims, [D], [{0}], cor, DT, RATE, RD, RM,
                                    vmax=np.full((1, G), 1e6), amax=amax, spread=False)
  v = np.diff(traj.q[:, 0]) / DT
  # interior only: the final step is truncated by landing exactly on the goal, which is an
  # arrival clamp rather than an acceleration the arm actually commanded
  assert np.abs(np.diff(v[:-1])).max() <= 0.5 * DT + 1e-6    # per-step accel bound honoured
  assert np.allclose(traj.q[-1], P[-1])


# ---- C1 spline smoothing ---------------------------------------------------------------
def test_catmull_rom_interpolates_waypoints_and_is_c1():
  P = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 1.0])]
  Q, imap = retiming.catmull_rom(P, per_seg=10)
  for i, j in enumerate(imap):
    assert np.allclose(Q[j], P[i])                           # passes through every waypoint
  d = np.diff(Q, axis=0)
  u = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
  turn = np.linalg.norm(np.diff(u, axis=0), axis=1).max()    # C1 => no direction jump
  du = np.diff(np.array([np.array([1.0, 0.0]), np.array([0.0, 1.0])]), axis=0)
  assert turn < 0.5                                          # vs ~1.41 for the raw polyline
  assert float(np.linalg.norm(du)) > 1.0                     # (the polyline really does jump)


def test_catmull_rom_tension_zero_is_the_polyline():
  P = [np.zeros(2), np.array([1.0, 0.0]), np.array([1.0, 1.0])]
  Q, imap = retiming.catmull_rom(P, per_seg=5, tension=0.0)
  assert retiming.path_deviation(Q, imap, P) < 1e-12         # exactly the original geometry


def test_catmull_rom_pin_keeps_witness_approach_straight():
  P = [np.zeros(2), np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 1.0])]
  Q, imap = retiming.catmull_rom(P, per_seg=6, pin={2})      # waypoint 2 is a witness
  seg = Q[imap[1]:imap[2] + 1]                               # its approach span
  a, b = P[1], P[2]
  dev = max(float(np.linalg.norm(p - (a + np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)),
                                                  0, 1) * (b - a)))) for p in seg)
  assert dev < 1e-12                                         # approach stayed exactly linear


def test_spline_path_shrinks_where_it_would_collide():
  # forbid any deviation in the middle span: it must be straightened back, others smoothed
  P = [np.zeros(2), np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 1.0]),
       np.array([2.0, 2.0])]
  calls = {"n": 0}
  def free_of(Q, j0, j1, i):
    calls["n"] += 1
    mid = Q[(j0 + j1) // 2]
    return not (0.9 < mid[0] < 1.6 and 0.4 < mid[1] < 1.1)   # a keep-out in the middle
  Q, imap, rep = retiming.spline_path(P, bar_idx=[0, len(P) - 1], per_seg=8,
                                      free_of=free_of)
  assert calls["n"] > 0 and rep["checked"] > 0
  for i, j in enumerate(imap):
    assert np.allclose(Q[j], P[i])                           # waypoints still exact
  assert rep["deviation"] < 0.6                              # bounded, and reported
