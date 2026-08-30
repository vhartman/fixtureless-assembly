"""--witness-sets: persistence round trip, flag-off byte-identity, and the selection metric.
The heavy end-to-end (witfloor harvest + GoalSet export) is validated on real scenes; these
tests pin the pure-python contracts."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plan_io import witness_block, write_plan, read_witnesses, read_witness_sets


JOINTS = [f"j{i}" for i in range(6)]


def _w(seed):
  return np.arange(6, dtype=float) + seed


def test_witness_block_without_alts_is_unchanged(tmp_path=None):
  blk = witness_block(JOINTS, [_w(0), None, _w(2)])
  assert set(blk.keys()) == {"joints", "wit"}          # no new key when alts is None/empty
  blk2 = witness_block(JOINTS, [_w(0), None, _w(2)], alts=None)
  assert blk == blk2
  assert witness_block(JOINTS, [_w(0)], alts={}) == witness_block(JOINTS, [_w(0)])


def test_round_trip_and_member0_guard(tmp_path):
  wit = [_w(0), None, _w(2)]
  alts = {0: [_w(0), _w(0) + 0.5], 2: [_w(2), _w(2) + 1.0, _w(2) - 1.0]}
  p = tmp_path / "plan.json"
  write_plan(str(p), {"assembly": "t"}, [("pick", "a1", "obj_0", 0)] * 3,
             witnesses=witness_block(JOINTS, wit, alts=alts))
  got = read_witness_sets(str(p), JOINTS, 3)
  assert set(got.keys()) == {0, 2}
  assert len(got[2]) == 3
  assert np.allclose(got[0][0], wit[0])
  # wrong joint order -> refused entirely
  assert read_witness_sets(str(p), list(reversed(JOINTS)), 3) is None
  # member 0 drifted from the canonical witness (canonical rewritten later) -> that set drops
  alts_bad = {0: [_w(0) + 9.0, _w(0) + 0.5], 2: alts[2]}
  write_plan(str(p), {"assembly": "t"}, [("pick", "a1", "obj_0", 0)] * 3,
             witnesses=witness_block(JOINTS, wit, alts=alts_bad))
  got = read_witness_sets(str(p), JOINTS, 3)
  assert set(got.keys()) == {2}


def test_old_files_without_alts_load_as_none(tmp_path):
  p = tmp_path / "plan.json"
  write_plan(str(p), {"assembly": "t"}, [("pick", "a1", "obj_0", 0)],
             witnesses=witness_block(JOINTS, [_w(0)]))
  assert read_witness_sets(str(p), JOINTS, 1) is None
  assert read_witnesses(str(p), JOINTS, 1) is not None    # plain witnesses unaffected


def test_greedy_maxmin_wrist_selection():
  # mirror of the witfloor selection: greedy farthest-point on the last-3-joint distance,
  # stopping below the spread threshold
  last3 = [3, 4, 5]

  def dist(a, b):
    return float(np.linalg.norm(a[last3] - b[last3]))

  base = _w(0)
  pool = [base + np.r_[0, 0, 0, d, 0, 0] for d in (0.01, 0.6, 1.2, 0.62)]
  members = [base.copy()]
  K, spread = 3, 0.05
  pool = list(pool)
  while len(members) < K and pool:
    dmin = [min(dist(w, m) for m in members) for w in pool]
    k = int(np.argmax(dmin))
    if dmin[k] < spread:
      break
    members.append(pool.pop(k))
  assert len(members) == 3
  # farthest first (1.2), then the one maximizing min-distance to BOTH (0.6, min-dist 0.6)
  assert abs(members[1][3] - base[3] - 1.2) < 1e-9
  assert abs(members[2][3] - base[3] - 0.6) < 1e-9


# ---- ACTIVE branch-seed construction (solver.branch_seeds / wrap_to_limits) ----------------
# the generator behind the sets: pure functions, testable without a scene. UR10-like limits:
# most joints +-3.2832 (range just over 2*pi -> every pi-offset has an in-limits
# representative), shoulder-lift only [-3.2832, 0.2832] (range < 2*pi -> targets either fit
# or the transform is dropped).
def _solver_mod():
  import solver
  return solver


UR10_LO = np.array([-3.28319, -3.28319, -2.68319, -3.28319, -3.28319, -3.28319])
UR10_HI = np.array([3.28319, 0.28319, 2.68319, 3.28319, 3.28319, 3.28319])


def test_wrap_to_limits_picks_nearest_in_limits():
  S = _solver_mod()
  # 4.0 is out of [-3.28, 3.28]; its 2*pi-shifted representative -2.283 is in and nearest
  assert abs(S.wrap_to_limits(4.0, -3.28319, 3.28319, 0.0) - (4.0 - 2 * np.pi)) < 1e-9
  # in-limits value near ref stays put
  assert abs(S.wrap_to_limits(1.0, -3.28319, 3.28319, 1.2) - 1.0) < 1e-9
  # no representative fits a narrow range -> None (drop, never clamp)
  assert S.wrap_to_limits(2.0, -1.0, 1.0, 0.0) is None
  # narrow range, shifted representative fits: 5.0 - 2*pi = -1.283
  got = S.wrap_to_limits(5.0, -1.5, 1.0, 0.0)
  assert got is not None and abs(got - (5.0 - 2 * np.pi)) < 1e-9


def test_branch_seeds_all_within_limits_and_deterministic():
  S = _solver_mod()
  q = np.array([0.5, -1.2, 1.4, -0.6, 1.1, 0.4])
  exact, seeds = S.branch_seeds(q, UR10_LO, UR10_HI)
  again = S.branch_seeds(q, UR10_LO, UR10_HI)
  assert all(np.array_equal(a, b) for a, b in zip(exact, again[0]))
  assert all(np.array_equal(a, b) for a, b in zip(seeds, again[1]))
  for w in list(exact) + list(seeds):
    assert np.all(w >= UR10_LO - 1e-9) and np.all(w <= UR10_HI + 1e-9)
  # UR10 ranges are barely over 2*pi -> no +-2*pi single-joint turn fits from an interior q
  assert exact == []
  # the wrist flip must be present: q4 negated, q3 and q5 offset by pi (some representative)
  def _is_wrist_flip(w):
    return (abs(w[4] + q[4]) < 1e-9
            and min(abs(w[3] - q[3] - np.pi), abs(w[3] - q[3] + np.pi)) < 1e-9
            and min(abs(w[5] - q[5] - np.pi), abs(w[5] - q[5] + np.pi)) < 1e-9
            and np.allclose(w[:3], q[:3]))
  assert any(_is_wrist_flip(w) for w in seeds)
  # the elbow flip too: q2 negated, q1 compensated by q2
  def _is_elbow_flip(w):
    return (abs(w[2] + q[2]) < 1e-9 and abs(w[1] - (q[1] + q[2])) < 1e-9
            and abs(w[0] - q[0]) < 1e-9 and abs(w[4] - q[4]) < 1e-9)
  assert any(_is_elbow_flip(w) for w in seeds)


def test_branch_seeds_exact_turns_when_range_allows():
  S = _solver_mod()
  wide_lo, wide_hi = np.full(6, -2 * np.pi - 0.1), np.full(6, 2 * np.pi + 0.1)
  q = np.array([0.0, -1.0, 1.0, 0.05, -0.05, 0.0])
  exact, _seeds = S.branch_seeds(q, wide_lo, wide_hi)
  # wrist joints near 0 in a +-(2*pi+eps) range: BOTH turn directions fit for all three
  assert len(exact) == 6
  for w in exact:
    d = w - q
    j = int(np.argmax(np.abs(d)))
    assert abs(abs(d[j]) - 2 * np.pi) < 1e-9      # exactly one joint, one whole turn
    d[j] = 0.0
    assert np.allclose(d, 0.0)
    assert j in (3, 4, 5)


def test_branch_seeds_drops_unrepresentable_transforms():
  S = _solver_mod()
  # shoulder-lift range < 2*pi: an elbow flip pushing q1 out of range must be DROPPED.
  # q1+q2 = -3.28 - 0.8 = -4.08, no 2*pi representative in [-3.28, 0.28] (-4.08+2*pi=2.2 > 0.28)
  q = np.array([0.5, -3.28, -0.8, -0.6, 1.1, 0.4])
  _exact, seeds = S.branch_seeds(q, UR10_LO, UR10_HI)
  for w in seeds:
    assert np.all(w >= UR10_LO - 1e-9) and np.all(w <= UR10_HI + 1e-9)
  assert not any(abs(w[2] + q[2]) < 1e-9 and abs(w[0] - q[0]) < 1e-9
                 and abs(w[4] - q[4]) < 1e-9 and abs(w[3] - (q[3] + q[2])) < 1e-9
                 for w in seeds)                   # the plain elbow flip is gone


def test_branch_seeds_degenerate_elbow_not_duplicated():
  S = _solver_mod()
  # q2 = 0 -> the elbow flip is the identity; it must not emit a duplicate of q
  q = np.array([0.5, -1.2, 0.0, -0.6, 1.1, 0.4])
  _exact, seeds = S.branch_seeds(q, UR10_LO, UR10_HI)
  assert not any(np.linalg.norm(w - q) < 1e-6 for w in seeds)
  # and no two seeds coincide
  for a in range(len(seeds)):
    for b in range(a + 1, len(seeds)):
      assert np.linalg.norm(seeds[a] - seeds[b]) > 1e-6


def test_branch_seeds_bad_shapes_are_refused():
  S = _solver_mod()
  assert S.branch_seeds(np.zeros(5), UR10_LO, UR10_HI) == ([], [])
  assert S.branch_seeds(np.zeros(6), UR10_LO[:5], UR10_HI) == ([], [])


def test_redundancy_seeds_follow_nullspace_and_respect_limits():
  S = _solver_mod()
  q = np.zeros(7)
  lo, hi = np.full(7, -1.0), np.full(7, 1.0)
  # Six independent task rows leave exactly joint 7 as the local self-motion direction.
  J = np.c_[np.eye(6), np.zeros(6)]
  seeds = S.redundancy_seeds(q, lo, hi, J, max_seeds=6)
  again = S.redundancy_seeds(q, lo, hi, J, max_seeds=6)
  assert len(seeds) == 6
  assert all(np.array_equal(a, b) for a, b in zip(seeds, again))
  assert all(np.all(x >= lo) and np.all(x <= hi) for x in seeds)
  assert all(np.allclose(x[:6], 0.0) and abs(x[6]) > 0.0 for x in seeds)
  # A full-rank 6R pose task has no redundant continuation direction.
  assert S.redundancy_seeds(np.zeros(6), np.full(6, -1), np.full(6, 1),
                            np.eye(6)) == []


def test_bounded_motion_crossing_does_not_invent_a_two_pi_wrap():
  S = _solver_mod()
  # These bounded configurations are 4rad apart. The legacy branch-label metric reduces that
  # modulo 2*pi; the motion-quality metric must count what the planner physically traverses.
  a, b = np.array([-3.47]), np.array([0.54])
  assert S.reconfig_crossings(a, b, thresh=3.0) == 0
  assert S.bounded_reconfig_crossings(a, b, thresh=3.0) == 1
