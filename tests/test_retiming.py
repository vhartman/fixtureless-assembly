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


def test_compact_stretches_removes_dense_shortcut_samples_but_keeps_barriers():
  # Two serialized straight shortcut edges in different modes. Their shared task witness must
  # remain an exact endpoint of BOTH independently compacted stretches.
  a = [np.array([x, 0.0]) for x in np.linspace(0.0, 1.0, 21)]
  b = [np.array([1.0, y]) for y in np.linspace(0.0, 2.0, 31)]
  out, rep = retiming.compact_stretches([(a, [3]), (b, [7])])
  assert [len(w) for w, _ in out] == [2, 2]
  assert out[0][1] == [3] and out[1][1] == [7]
  assert np.array_equal(out[0][0][-1], a[-1])
  assert np.array_equal(out[1][0][0], b[0])
  assert np.array_equal(out[0][0][-1], out[1][0][0])
  assert rep["removed"] == 48
  assert abs(rep["arc_after"] - rep["arc_before"]) < 1e-12


def test_compact_polyline_preserves_corners_and_collinear_backtracking():
  corner = [np.array([0.0, 0.0]), np.array([0.5, 0.0]), np.array([1.0, 0.0]),
            np.array([1.0, 1.0])]
  compact = retiming.compact_polyline(corner)
  assert len(compact) == 3
  assert np.array_equal(compact[1], [1.0, 0.0])              # real corner survives

  backtrack = [np.array([0.0]), np.array([1.0]), np.array([0.25]), np.array([2.0])]
  compact = retiming.compact_polyline(backtrack)
  assert len(compact) > 2                                   # same line, different path geometry
  before = sum(np.linalg.norm(b - a) for a, b in zip(backtrack, backtrack[1:]))
  after = sum(np.linalg.norm(b - a) for a, b in zip(compact, compact[1:]))
  assert after == pytest.approx(before)


def test_compose_owned_path_preserves_both_polylines_and_endpoints():
  # Joint 0 is controller-owned. Joint 1 has an off-diagonal planner corner which must not
  # be shortcut when the denser skill path replaces joint 0.
  base = [np.array([0.0, 0.0]), np.array([0.4, 2.0]), np.array([1.0, 1.0])]
  skill = [np.array([0.0, 99.0]), np.array([0.2, 99.0]),
           np.array([0.7, 99.0]), np.array([1.0, 99.0])]
  got = retiming.compose_owned_path(base, skill, [0])
  np.testing.assert_allclose(got[0], [0.0, 0.0])
  np.testing.assert_allclose(got[-1], [1.0, 1.0])
  # Every owned skill knot survives at uniform skill phase.
  np.testing.assert_allclose([got[0][0], got[1][0], got[3][0], got[-1][0]],
                             [0.0, 0.2, 0.7, 1.0])
  # The base corner survives at its normalized composite arc phase.
  assert any(np.allclose(q[1], 2.0) for q in got)
  assert all(q[1] != 99.0 for q in got)


def test_audit_path_checks_interpolated_configs_in_their_exact_mode():
  modes = [object(), object()]
  stretches = [([np.array([0.0]), np.array([1.0])], [1]),
               ([np.array([1.0]), np.array([2.0])], [2])]
  seen = []
  def free(q, mode):
    seen.append((float(q[0]), mode))
    bad = mode is modes[0] and 0.49 <= q[0] <= 0.51
    return (not bad), ({"pair": "arm~table"} if bad else None)
  rep = retiming.audit_path(stretches, modes, free, limits=([-3.0], [3.0]),
                            resolution=0.1)
  assert not rep["ok"] and rep["kind"] == "collision"
  assert rep["stretch"] == 0 and rep["alpha"] == pytest.approx(0.5)
  assert rep["detail"] == {"pair": "arm~table"}
  assert all(mode is modes[0] for _, mode in seen)            # failed before next mode


def test_audit_path_reports_joint_limit_before_collision_callback():
  calls = {"n": 0}
  def free(_q, _mode):
    calls["n"] += 1
    return True
  rep = retiming.audit_path(
      [([np.array([0.0, 0.0]), np.array([0.0, 1.2])], [1])], [object()], free,
      limits=([-1.0, -1.0], [1.0, 1.0]), resolution=0.1)
  assert not rep["ok"] and rep["kind"] == "joint_limits"
  assert rep["joint_indices"] == [1]
  assert rep["q"][1] > 1.0
  assert calls["n"] == rep["samples"] - 1


def test_motion_audit_indices_use_joint_variation_and_pin_events():
  q = np.array([[0.000], [0.002], [0.004], [0.006], [0.008], [0.010],
                [0.012], [0.012], [0.012], [0.014]])
  got = retiming.motion_audit_indices(q, fires=[(2, [1]), (8, [2])], resolution=0.01)
  assert got.tolist() == [0, 2, 8, 9]
  assert {0, 2, 8, 9}.issubset(set(got))
  for a, b in zip(got, got[1:]):
    variation = np.abs(np.diff(q[a:b + 1], axis=0)).max(axis=1).sum()
    assert variation <= 0.01 + 1e-12


def test_motion_audit_indices_dense_compatibility_and_backtracking():
  q = np.array([[0.0], [0.004], [0.008], [0.004], [0.0]])
  assert retiming.motion_audit_indices(q, resolution=0).tolist() == list(range(len(q)))
  got = retiming.motion_audit_indices(q, resolution=0.01)
  assert got.tolist() == [0, 2, 4]  # accumulated variation, not endpoint displacement


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


def test_per_stretch_rate_slows_only_selected_skill_interval():
  st = [([np.array([0.0]), np.array([1.0])], [10]),
        ([np.array([1.0]), np.array([2.0])], [11])]
  fast = retiming.build_trajectory(st, DT, RATE, RD, RM)
  mixed = retiming.build_trajectory(st, DT, RATE, RD, RM,
                                    stretch_rates=[RATE, RATE / 2.0])
  assert mixed.fires[0][0] == fast.fires[0][0]
  assert mixed.fires[1][0] > fast.fires[1][0]

  dims = [np.array([0])]
  sched_fast = retiming.plan_schedule(st, dims, [{0}, {0}], DT, RATE, RD, RM)
  sched_mixed = retiming.plan_schedule(
      st, dims, [{0}, {0}], DT, RATE, RD, RM,
      stretch_rates=[RATE, RATE / 2.0])
  assert sched_mixed.fires[0][0] == sched_fast.fires[0][0]
  assert sched_mixed.fires[1][0] > sched_fast.fires[1][0]


def test_schedule_respects_external_phase_rate_profile():
  st = [([np.array([0.0]), np.array([1.0])], [10])]
  dims = [np.array([0])]
  free = retiming.plan_schedule(st, dims, [{0}], DT, RATE, RD, RM)
  limited = retiming.plan_schedule(
      st, dims, [{0}], DT, RATE, RD, RM,
      rate_limits=np.full((1, 21), 0.25), rate_limit_ds=0.05)
  assert len(limited.q) > len(free.q)
  speed = np.abs(np.diff(limited.q[:, 0])) / DT
  assert float(speed.max()) <= 0.25 + 1e-9


def test_schedule_rate_profile_requires_matching_arm_rows():
  st = [([np.array([0.0]), np.array([1.0])], [10])]
  with pytest.raises(ValueError, match="rate_limits must have shape"):
    retiming.plan_schedule(
        st, [np.array([0])], [{0}], DT, RATE, RD, RM,
        rate_limits=np.ones((2, 3)), rate_limit_ds=0.5)


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


def test_schedule_structural_lock_keeps_payload_on_certified_diagonal():
  # A carrying arm cannot be allowed to run ahead of another arm through the same symbolic
  # state: that would move its payload through a configuration absent from the certified
  # composite path.  Structural locks express this attachment invariant without needing a
  # collision counterexample first.
  st = [([np.array([0.0, 0.0]), np.array([4.0, 8.0])], [10])]
  dims = [np.array([0]), np.array([1])]
  free = retiming.plan_schedule(st, dims, [{0}], DT, RATE, RD, RM,
                                per_arm_rate=True)
  locked = retiming.plan_schedule(st, dims, [{0}], DT, RATE, RD, RM,
                                  per_arm_rate=True,
                                  locked=[(0, 1, 0.0, np.hypot(4.0, 8.0))])
  assert np.abs(free.q[:, 0] / 4.0 - free.q[:, 1] / 8.0).max() > 0.05
  assert np.allclose(locked.q[:, 0] / 4.0, locked.q[:, 1] / 8.0)
  assert [e for _, e in locked.fires] == [[10]]


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


def test_schedule_boolean_pocket_recovers_certified_diagonal():
  # Some collision backends expose only a boolean, so the ordinary wedge scan cannot name
  # the blocking arm.  A phase-lead deadlock must still restart toward the known-safe
  # synchronized diagonal instead of failing solely because blocker attribution is absent.
  st = [([np.array([0.0, 0.0]), np.array([2.0, 2.0])], [10]),
        ([np.array([2.0, 2.0]), np.array([6.0, 6.0])], [11])]
  dims = [np.array([0]), np.array([1])]
  def pocket(fq, fired):
    x0, x1 = fq
    return ((2.7 < x1 < 3.5 and 1.8 < x0 < 2.35)
            or (2.2 < x0 < 2.4 and 2.6 < x1 < 2.8))
  traj = retiming.plan_schedule(
      st, dims, [{0}, {1}], DT, RATE, RD, RM,
      collide=pocket, dwell=1.0, stall_steps=300)
  assert np.allclose(traj.q[-1], [6.0, 6.0])
  assert not any(pocket(fq, 0) for fq in traj.q)


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


def test_schedule_nominal_event_duration_is_feed_forward_slack():
  # The planned servo budget delays the fire while the acting arm remains parked.  It does
  # not freeze an unrelated arm: that arm consumes useful path during the nominal ceremony.
  stretches, arm_dims, acting = _two_arm_stretches()
  base = retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM,
                                dwell=0.0, event_durations=[0.0, 0.0])
  budget = 0.20
  timed = retiming.plan_schedule(stretches, arm_dims, acting, DT, RATE, RD, RM,
                                 dwell=0.0, event_durations=[budget, 0.0])
  i0, j0 = base.fires[0][0], timed.fires[0][0]
  assert j0 - i0 == round(budget / DT)
  assert np.ptp(timed.q[j0 - round(budget / DT):j0 + 1, 0]) < 1e-9
  assert np.ptp(timed.q[j0 - round(budget / DT):j0 + 1, 1]) > 0.01


def test_schedule_post_event_duration_holds_only_actor():
  stretches, arm_dims, acting = _two_arm_stretches()
  budget = 0.05
  base = retiming.plan_schedule(
      stretches, arm_dims, acting, DT, RATE, RD, RM, dwell=0.0,
      event_durations=[0.0, 0.0], post_event_durations=[0.0, 0.0])
  timed = retiming.plan_schedule(
      stretches, arm_dims, acting, DT, RATE, RD, RM, dwell=0.0,
      event_durations=[0.0, 0.0], post_event_durations=[budget, 0.0])
  fire = timed.fires[0][0]
  w = int(round(budget / DT))
  # Opening after the event parks its actor, but a foreign arm consumes useful path.
  assert np.ptp(timed.q[fire:fire + w + 1, 0]) < 1e-9
  assert np.ptp(timed.q[fire:fire + w + 1, 1]) > 0.01
  assert len(timed.q) >= len(base.q)


def test_upsample_trajectory_preserves_duration_events_and_endpoints():
  src = retiming.TimedTrajectory(
      dt=0.01, q=np.asarray([[1.0, 2.0], [2.0, 4.0]]),
      phase=["path", "dwell"], fires=[(1, [7])],
      progress=np.asarray([[1.0, 1.0], [2.0, 2.0]]))
  dense = retiming.upsample_trajectory(src, 0.002, q0=np.zeros(2))
  assert len(dense.q) == 10
  assert np.allclose(dense.q[4], src.q[0])
  assert np.allclose(dense.q[-1], src.q[-1])
  assert dense.fires == [(9, [7])]
  assert dense.phase[:5] == ["path"] * 5
  assert dense.phase[5:] == ["dwell"] * 5
  assert np.allclose(dense.progress[4], src.progress[0])
  assert np.allclose(dense.progress[-1], src.progress[-1])


def test_per_arm_rate_skips_invisible_stationary_phase():
  # Composite arc = arm0 moves, then arm1 moves. With a shared 1rad/s phase clock this takes
  # roughly 2s. Independent phases should overlap the two real 1rad motions; neither arm must
  # spend another second "moving" through the segment where its own q is constant.
  st = [([np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0])], [10])]
  dims = [np.array([0]), np.array([1])]
  shared = retiming.plan_schedule(st, dims, [{0, 1}], DT, RATE, RD, RM)
  own = retiming.plan_schedule(st, dims, [{0, 1}], DT, RATE, RD, RM,
                               per_arm_rate=True)
  assert len(own.q) < 0.7 * len(shared.q)
  assert np.allclose(own.q[-1], [1.0, 1.0])
  assert [e for _, e in own.fires] == [[10]]


def _stuttering_phase_trajectory():
  """Two-arm diagonal path with the right event anchors but an ugly 0/2x rate pattern."""
  P = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
  segs, D = retiming.arc_length(list(P))
  csum = np.concatenate([[0.0], np.cumsum(segs)])
  n, fire = 401, 200
  prog = np.zeros((n, 2))
  for r in range(2):
    d = np.where(np.arange(n) % 8 < 4, 1.0, 0.05)
    if r == 0:                                               # actor rests around its fire
      d[fire - 8:fire + 9] = 0.0
    for a, b, total in ((0, fire + 1, D / 2), (fire + 1, n, D / 2)):
      d[a:b] *= total / d[a:b].sum()
    prog[:, r] = np.cumsum(d)
  q = np.zeros((n, 2))
  for k in range(n):
    for r in range(2):
      q[k, r] = retiming.config_at(list(P), csum, segs, prog[k, r])[r]
  return P, retiming.TimedTrajectory(
      dt=DT, q=q, phase=["path"] * n, fires=[(fire, [7])], progress=prog)


def test_phase_smoothing_is_monotone_pins_events_and_reduces_jerk():
  P, src = _stuttering_phase_trajectory()
  dims = [np.array([0]), np.array([1])]
  out, rep = retiming.smooth_phase_trajectory(
      src, P, dims, [{0}], window=0.08, stop_time=0.016)

  assert np.all(np.diff(out.progress, axis=0) >= -1e-12)
  assert np.allclose(out.progress[200], src.progress[200])   # complete mode-switch vector pinned
  assert np.allclose(out.progress[-1], src.progress[-1])
  assert out.fires == src.fires
  assert rep["after"]["jerk_rms"] < 0.5 * rep["before"]["jerk_rms"]
  for r in range(2):
    raw_peak = np.abs(np.diff(src.q[:, r])).max()
    new_peak = np.abs(np.diff(out.q[:, r])).max()
    assert new_peak <= 1.02 * raw_peak + 1e-10             # smoothing never creates a burst
  # The actor has an actual rest neighbourhood, while the crossing arm is not forced to stop.
  assert np.max(np.diff(out.progress[197:204, 0], axis=0)) < 1e-10
  assert np.max(np.diff(out.progress[194:207, 1], axis=0)) > 1e-5


def test_phase_smoothing_preserves_asymmetric_physical_event_waits():
  stretches, arm_dims, acting = _two_arm_stretches()
  wait = 0.05
  raw = retiming.plan_schedule(
      stretches, arm_dims, acting, DT, RATE, RD, RM, dwell=0.0,
      event_durations=[wait, 0.0], post_event_durations=[wait, 0.0])
  path, _ = retiming._composite(stretches)
  smooth, _ = retiming.smooth_phase_trajectory(
      raw, path, arm_dims, acting, window=0.04,
      event_stop_times=[(wait, wait), (0.0, 0.0)])
  fire = smooth.fires[0][0]
  w = int(round(wait / DT))
  assert np.ptp(smooth.q[fire - w:fire + w + 1, 0]) < 1e-9
  assert np.ptp(smooth.q[fire - w:fire + w + 1, 1]) > 0.01


def test_smoothness_quality_guard_rejects_material_regression():
  before = {"accel_rms": 10.0, "jerk_rms": 100.0}
  assert retiming.smoothness_not_worse(
      before, {"accel_rms": 10.1, "jerk_rms": 99.0})
  assert not retiming.smoothness_not_worse(
      before, {"accel_rms": 10.3, "jerk_rms": 99.0})
  assert not retiming.smoothness_not_worse(
      before, {"accel_rms": 9.0, "jerk_rms": 103.0})


def test_nonnegative_sum_projection_is_exact_and_least_change():
  v = np.array([0.0, 0.2, 0.8, 0.3])
  out = retiming._project_nonnegative_sum(v, 0.7)
  assert np.all(out >= 0.0)
  assert np.isclose(out.sum(), 0.7, atol=1e-14)
  # KKT: every positive component receives the same additive correction; clamped entries
  # cannot be made negative. Multiplicative normalization does not have this property.
  active = out > 1e-12
  assert np.ptp(out[active] - v[active]) < 1e-12


def test_coupling_rendezvous_caps_both_members_until_both_arrive():
  couplings = {frozenset((0, 1)): [[1.0, 2.0]]}

  entry, comps = retiming._coupling_topology(couplings, [1.0, 0.4, 3.0], 3, 4.0)
  assert entry == {0: 1.0, 1: 1.0}
  assert comps == []

  entry, comps = retiming._coupling_topology(couplings, [1.0, 1.0, 3.0], 3, 4.0)
  assert entry == {}
  assert comps == [[0, 1]]


def test_phase_smoothing_preserves_rigid_coupling_intervals_exactly():
  P, src = _stuttering_phase_trajectory()
  dims = [np.array([0]), np.array([1])]
  _, D = retiming.arc_length(list(P))
  out, rep = retiming.smooth_phase_trajectory(
      src, P, dims, [{0}], window=0.08, locked=[(0, 1, 0.0, D)])

  assert np.array_equal(out.progress, src.progress)
  assert np.array_equal(out.q, src.q)
  assert rep["locked_arm_intervals"] == 4                   # 2 arms x 2 fire intervals


def test_phase_smoothing_preserves_crossing_arm_corner_rate_envelope():
  # Arm 0 owns the event. Arm 1 crosses a 90-degree path corner at its barrier phase with
  # an intentional scheduler slowdown. Smoothing may change rates elsewhere but must not
  # erase any source increment inside the corner ramp envelope.
  P = np.array([[0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 1.0]])
  dims = [np.array([0]), np.array([1, 2])]
  _, D = retiming.arc_length(list(P))
  n, fire = 401, 200
  inc = np.full((n, 2), D / n)
  inc[fire - 20:fire + 20, 1] *= 0.25
  inc[:fire, 1] *= (D / 2) / inc[:fire, 1].sum()
  inc[fire:, 1] *= (D / 2) / inc[fire:, 1].sum()
  inc[:fire, 0] *= (D / 2) / inc[:fire, 0].sum()
  inc[fire:, 0] *= (D / 2) / inc[fire:, 0].sum()
  prog = np.cumsum(inc, axis=0)
  segs, _ = retiming.arc_length(list(P))
  csum = np.concatenate([[0.0], np.cumsum(segs)])
  q = np.zeros((n, 3))
  for k in range(n):
    for r in range(2):
      q[k, dims[r]] = retiming.config_at(list(P), csum, segs, prog[k, r])[dims[r]]
  src = retiming.TimedTrajectory(
      dt=DT, q=q, phase=["path"] * n, fires=[(fire - 1, [7])], progress=prog)
  out, rep = retiming.smooth_phase_trajectory(
      src, P, dims, [{0}], window=0.08, barrier_indices=[1], ramp_dist=0.2,
      corner_dv=0.2)

  x = np.vstack([np.zeros((1, 2)), src.progress])
  mid = 0.5 * (x[:-1, 1] + x[1:, 1])
  fixed = np.abs(mid - csum[1]) <= 0.2 + 1e-9
  assert np.any(fixed)
  assert np.array_equal(np.diff(np.r_[0.0, out.progress[:, 1]])[fixed],
                        np.diff(np.r_[0.0, src.progress[:, 1]])[fixed])
  assert rep["corner_fixed_increments"] == int(np.count_nonzero(fixed))


def _small_motion_island_trajectory():
  # A long useful motion, 0.10s stop, then a visibly pointless 0.04rad afterthought.
  n, dt = 80, 0.01
  inc = np.zeros(n)
  inc[2:27] = 0.01
  inc[37:41] = 0.01
  p = np.cumsum(inc)
  q = p[:, None].copy()
  return np.array([[0.0], [p[-1]]]), retiming.TimedTrajectory(
      dt=dt, q=q, phase=["path"] * n, fires=[(32, [7])], progress=p[:, None])


def test_phase_batch_merges_small_island_across_foreign_event():
  P, src = _small_motion_island_trajectory()
  out, rep = retiming.batch_phase_motions(
      src, P, [np.array([0])], [set()], [0.10], min_motion=0.06, max_gap=0.2,
      collision_free=lambda q, i: True)
  assert rep["merged"] == 1
  assert rep["after"] == rep["before"] - 1
  assert rep["small_after"] < rep["small_before"]
  assert rep["internal_slow_after"] < rep["internal_slow_before"]
  assert np.all(np.diff(out.progress[:, 0]) >= -1e-12)
  assert out.progress[32, 0] >= 0.10                        # global event passage retained
  assert np.allclose(out.q[-1], src.q[-1])
  assert out.fires == src.fires


def test_phase_batch_never_crosses_own_event_or_accepted_collision():
  P, src = _small_motion_island_trajectory()
  own, own_rep = retiming.batch_phase_motions(
      src, P, [np.array([0])], [{0}], [0.10], min_motion=0.06, max_gap=0.2,
      collision_free=lambda q, i: True)
  assert own_rep["merged"] == 0 and own_rep["event_rejected"] > 0
  assert np.array_equal(own.q, src.q)

  def reject_bridge(q, i):
    return not (27 <= i <= 36 and q[0] > src.q[i, 0] + 1e-9)
  blocked, block_rep = retiming.batch_phase_motions(
      src, P, [np.array([0])], [set()], [0.10], min_motion=0.06, max_gap=0.2,
      collision_free=reject_bridge)
  assert block_rep["merged"] == 0 and block_rep["collision_rejected"] > 0
  assert np.array_equal(blocked.q, src.q)


def test_phase_intent_increases_idle_support_at_explicit_faster_rate():
  P, src = _small_motion_island_trajectory()
  out, rep = retiming.batch_phase_motions(
      src, P, [np.array([0])], [set()], [0.10],
      min_motion=float("inf"), max_gap=float("inf"),
      collision_free=lambda q, i: True, maximize_idle=True,
      intent_ramp_time=0.02, intent_rate=2.0)

  assert rep["merged"] == 1
  assert rep["support_after"]["moving_samples"] < rep["support_before"]["moving_samples"]
  assert rep["support_after"]["idle_fraction"] > rep["support_before"]["idle_fraction"]
  assert rep["support_after"]["mean_moving_rate"] > rep["support_before"]["mean_moving_rate"]
  assert np.linalg.norm(np.diff(out.q, axis=0), axis=1).max() <= 2.0 * src.dt + 1e-9
  assert np.all(np.diff(out.progress[:, 0]) >= -1e-12)
  assert np.allclose(out.q[-1], src.q[-1])


def test_phase_intent_preserves_rigid_coupling_window():
  P, src = _small_motion_island_trajectory()
  out, rep = retiming.batch_phase_motions(
      src, P, [np.array([0])], [set()], [0.10],
      locked=[(0, 0, 0.0, float(src.progress[-1, 0]))],
      min_motion=float("inf"), max_gap=float("inf"),
      collision_free=lambda q, i: True, maximize_idle=True,
      intent_rate=2.0)
  assert rep["merged"] == 0
  assert np.array_equal(out.q, src.q)
  assert np.array_equal(out.progress, src.progress)


def test_phase_intent_uses_one_delayed_motion_across_foreign_switch():
  dt, n = 0.01, 101
  inc = np.zeros(n)
  inc[10:50] = 0.01                                        # first hesitant motion
  inc[65:90] = 0.016                                       # resumes after foreign switch
  progress = np.cumsum(inc)[:, None]
  P = np.array([[0.0], [float(progress[-1, 0])]])
  src = retiming.TimedTrajectory(
      dt=dt, q=progress.copy(), phase=["path"] * n,
      fires=[(60, [1]), (90, [2])], progress=progress.copy())
  acting = [{1}, {0}]                                      # first event is physical but foreign
  assert retiming.foreign_switch_stops(
      src.q, dt, [np.array([0])], src.fires, acting, stopped_rate=0.05) == 1

  out, rep = retiming.intentional_phase_motions(
      src, P, [np.array([0])], acting, [0.4, float(progress[90, 0])],
      collision_free=lambda q, i: True, intent_rate=2.0, ramp_time=0.08,
      stop_time=0.02)

  assert rep["accepted"] == 1
  assert rep["unresolved"] == []
  assert rep["islands_before"] == 2 and rep["islands_after"] == 1
  assert rep["foreign_stops_before"] == 1 and rep["foreign_stops_after"] == 0
  # Joining the two motions is allowed to use slightly more individual support when that is
  # what prevents a new all-robot dead interval; visual continuity is the primary objective.
  assert rep["global_idle_after"]["samples"] <= rep["global_idle_before"]["samples"]
  assert out.progress[60, 0] >= 0.4 - 1e-9                # foreign event passage
  assert np.allclose(out.progress[90, 0], src.progress[90, 0])
  assert np.allclose(out.q[88:91, 0], out.q[90, 0])       # own event still has a rest plateau


def test_phase_intent_removes_shared_path_flats_from_one_robot_motion():
  dt, n = 0.01, 201
  # The middle composite segment belongs only to another robot. In shared progress arm0's
  # two motions are separated by that bookkeeping flat; in its intrinsic arc they are one.
  P = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [1.0, 1.0]])
  p = np.zeros(n)
  p[10:51] = np.linspace(0.0, 0.5, 41)
  p[51:101] = np.linspace(0.5, 1.5, 50)
  p[101:130] = 1.5
  p[130:171] = np.linspace(1.5, 2.0, 41)
  p[171:] = 2.0
  segs, _ = retiming.arc_length(list(P))
  csum = np.r_[0.0, np.cumsum(segs)]
  q = np.asarray([retiming.config_at(list(P), csum, segs, s) for s in p])
  src = retiming.TimedTrajectory(
      dt=dt, q=q, phase=["path"] * n,
      fires=[(110, [1]), (180, [2])], progress=p[:, None])

  out, rep = retiming.intentional_phase_motions(
      src, P, [np.array([0])], [{1}, {0}], [1.0, 2.0],
      collision_free=lambda q, i: True, intent_rate=2.0, ramp_time=0.08,
      stop_time=0.02)

  assert rep["accepted"] == 1
  assert rep["islands_before"] == 2 and rep["islands_after"] == 1
  assert len(retiming.motion_islands(out.q, dt, [np.array([0])], 0.05)) == 1
  assert out.progress[110, 0] >= 1.0 - 1e-9             # foreign barrier still passed
  assert np.allclose(out.q[178:181, 0], out.q[180, 0])   # own event remains a stop


def test_phase_intent_places_compact_motion_to_cover_global_idle_first():
  dt, n = 0.01, 101
  P = np.array([[0.0, 0.0], [1.0, 1.0]])
  D = np.sqrt(2.0)
  p0 = np.linspace(0.0, D, n)
  p1 = np.zeros(n)
  p1[60:81] = np.linspace(0.0, D, 21)
  p1[81:] = D
  progress = np.c_[p0, p1]
  q = progress / D
  src = retiming.TimedTrajectory(
      dt=dt, q=q, phase=["path"] * n, fires=[], progress=progress.copy())

  out, rep = retiming.intentional_phase_motions(
      src, P, [np.array([0]), np.array([1])], [], [],
      locked=[(1, 1, 0.0, D)], collision_free=lambda q, i: True,
      intent_rate=2.0, ramp_time=0.08, stop_time=0.02)

  moving0 = np.linalg.norm(np.diff(out.q[:, [0]], axis=0), axis=1) >= 0.05 * dt
  # Latest-first would overlap arm1's 60:80 motion and leave the beginning empty. Coverage
  # scheduling instead uses that empty beginning; compact support itself is unchanged.
  assert np.flatnonzero(moving0)[0] < 5
  assert rep["global_idle_after"]["quadratic_samples"] \
      < retiming.global_idle_stats(
          np.c_[np.r_[np.zeros(42), np.linspace(0, 1, 59)], q[:, 1]],
          dt, [np.array([0]), np.array([1])], 0.05)["quadratic_samples"]


def test_phase_smoothing_can_restore_one_collision_sensitive_interval_exactly():
  P, src = _stuttering_phase_trajectory()
  dims = [np.array([0]), np.array([1])]
  out, rep = retiming.smooth_phase_trajectory(
      src, P, dims, [{0}], window=0.08, restore_intervals={0})
  a, b = retiming.phase_intervals(src)[0]

  assert np.array_equal(out.progress[a:b + 1], src.progress[a:b + 1])
  assert np.array_equal(out.q[a:b + 1], src.q[a:b + 1])
  assert rep["restored"] == 1


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


def test_online_pace_is_fixed_per_mode_and_flows_crossing_arm_through_fire():
  # arm0 has four times as much joint travel as arm1 and acts at the first barrier. A greedy
  # arm1 reaches the no-pass cap early and parks. Feed-forward pacing must instead choose one
  # fixed factor for the whole stretch, leave arm1 short of the cap when arm0 fires, and use
  # the predicted ceremony window to finish the crossing motion.
  P = [np.array([0.0, 0.0]), np.array([1.0, 0.25]), np.array([2.0, 0.5])]
  dims = [np.array([0]), np.array([1])]
  segs, D = retiming.arc_length(P)
  bar = [segs[0], D]
  cor = _flat_corridor(2, D, ds=0.05)
  stepper = retiming.CorridorStepper(
      P, dims, bar, [{0}, {0}], cor, DT, joint_rate=RATE, ramp_dist=RD, ramp_min=RM,
      fire_all=False, pace=True, settle_nom=0.4, trace=True)

  fire = None
  for _ in range(10000):
    _, fire = stepper.tick()
    if fire is not None:
      break
  assert fire == 0
  assert stepper.s[0] >= bar[0] - 1e-6                       # actor is at its witness
  assert stepper.s[1] < bar[0] - 1e-3                        # crossing arm did not rush and park
  pf = np.asarray([row[1] for row in stepper._trace])
  assert np.ptp(pf[:, 0]) < 1e-12 and np.ptp(pf[:, 1]) < 1e-12  # no feedback chasing

  # The real servo finishes much earlier than its 0.4s bound. Opening the mode early must not
  # require an ETA correction: keep the same factor through cleanup and cross continuously.
  q1 = []
  for _ in range(int(round(0.1 / DT))):
    q, _ = stepper.tick(rate_factor=np.array([0.0, 1.0]))    # actor rests during ceremony
    q1.append(q[1])
  assert np.ptp(q1) > 0.01                                  # nonactor keeps moving
  assert np.ptp(np.diff(q1)) < 1e-10                         # at one constant planned velocity
  assert stepper.s[1] < bar[0] - 1e-3                        # reserve was not exhausted
  old_factor = stepper._pace_factor.copy()
  stepper.confirm_fire(defer_pace_replan=True)               # actual attachment opens mode
  q2 = []
  for _ in range(20):
    q, _ = stepper.tick()
    q2.append(q[1])
  assert np.allclose(stepper._pace_factor, old_factor)       # no correction during cleanup
  assert np.min(np.diff(q2)) > 0.0                           # crosses without a stop
  stepper.replan_pace()
  stepper.tick()
  assert stepper._pace_replans == 2                          # one event-triggered replan only


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
