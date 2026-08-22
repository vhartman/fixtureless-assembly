import numpy as np

import c2_smoothing as c2


def _corner():
  return [np.array([0.0, 0.0]), np.array([1.0, 0.0]),
          np.array([1.0, 1.0]), np.array([2.0, 1.0]), np.array([2.0, 2.0])]


def test_fit_is_c2_and_endpoints_are_exact():
  P = _corner()
  spline, _, _ = c2.fit_c2(P, control_ratio=1.0, smooth_weight=0.01)
  assert np.array_equal(spline(0.0), P[0])
  assert np.array_equal(spline(1.0), P[-1])
  # A single cubic BSpline object is C2 at every simple interior knot.  Verify both one-sided
  # second derivatives numerically at every knot rather than merely inspecting samples.
  knots = np.unique(spline.t[(spline.k + 1):-(spline.k + 1)])
  d2 = spline.derivative(2)
  for u in knots:
    assert np.linalg.norm(d2(u - 1e-8) - d2(u + 1e-8)) < 1e-4


def test_stretch_witnesses_exact_and_interior_waypoints_may_move():
  P = _corner()
  out, reports, total = c2.smooth_stretches(
      [(P, [4])], lambda q, k: True, per_segment=6, max_deviation=2.0)
  Q = np.asarray(out[0][0])
  assert np.array_equal(Q[0], P[0])
  assert np.array_equal(Q[-1], P[-1])
  assert reports[0].status == "fitted" and total["fitted"] == 1
  # The fit is not constrained to visit every ordinary MP waypoint.
  assert min(np.linalg.norm(Q - P[1], axis=1)) > 1e-4


def test_collision_rejection_falls_back_to_certified_polyline():
  P = _corner()
  def free(q, _stretch):
    # All curves that cut the first corner enter x>0.7,y>0; the raw horizontal/vertical
    # polyline is the conservative fallback and is assumed certified by the caller.
    return not (0.70 < q[0] < 0.99 and 0.01 < q[1] < 0.30)
  out, reports, total = c2.smooth_stretches(
      [(P, [])], free, per_segment=12, max_deviation=2.0)
  assert reports[0].status == "fallback" and total["fallback"] == 1
  assert np.array_equal(np.asarray(out[0][0]), np.asarray(P))


def test_joint_limit_rejection_and_protected_stretches_are_unchanged():
  P = _corner()
  stretches = [(P, [1]), (P, [2])]
  out, reports, total = c2.smooth_stretches(
      stretches, lambda q, k: True,
      limits=(np.array([-10.0, -10.0]), np.array([10.0, 0.2])),
      protected={1}, max_deviation=2.0)
  assert reports[0].status == "fallback"
  assert reports[1].status == "protected"
  assert np.array_equal(np.asarray(out[1][0]), np.asarray(P))
  assert total["protected"] == 1
