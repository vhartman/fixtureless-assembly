"""Independent, collision-audited C2 geometry postprocessing.

This module deliberately knows nothing about simulation, phase scheduling, modes, robots, or
RAI.  Its input is a sequence of mode-local polylines and a collision predicate supplied by
the caller.  Ordinary planner waypoints are soft observations of a cubic B-spline; stretch
endpoints (task witnesses) are exact.  Unsafe fits fall back to the original certified
polyline, one complete stretch at a time.

The returned arrays are samples *of* the C2 curve.  Downstream code may continue to consume a
plain waypoint path, which keeps this experimental geometry stage independent from the stable
retiming and controller pipeline.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass
class StretchReport:
  index: int
  status: str
  input_points: int
  output_points: int
  control_points: int = 0
  attempts: int = 0
  checked_samples: int = 0
  max_deviation: float = 0.0
  reason: str = ""


def _deduplicate(points, atol=1e-12):
  out = []
  for p in points:
    p = np.asarray(p, float)
    if not out or float(np.linalg.norm(p - out[-1])) > atol:
      out.append(p)
  return out


def chord_parameter(points):
  """Strictly increasing chord-length parameter in [0, 1]."""
  P = np.asarray(_deduplicate(points), float)
  if len(P) <= 1:
    return P, np.zeros(len(P))
  d = np.linalg.norm(np.diff(P, axis=0), axis=1)
  total = float(d.sum())
  if total <= 1e-15:
    return P[:1], np.zeros(1)
  return P, np.concatenate([[0.0], np.cumsum(d) / total])


def _knots(n_control, degree=3):
  interior = n_control - degree - 1
  mid = np.linspace(0.0, 1.0, interior + 2)[1:-1] if interior else np.empty(0)
  return np.concatenate([np.zeros(degree + 1), mid, np.ones(degree + 1)])


def _basis(u, knots, n_control, degree=3, derivative=0):
  spl = BSpline(knots, np.eye(n_control), degree, extrapolate=False)
  if derivative:
    spl = spl.derivative(derivative)
  return np.asarray(spl(np.asarray(u, float)), float)


def fit_c2(points, control_ratio=0.55, smooth_weight=0.03, n_control=None):
  """Fit a clamped cubic B-spline with exact endpoints and soft interior waypoints.

  The least-squares objective is waypoint error plus ``smooth_weight`` times squared second
  differences of the control polygon.  Returns ``(spline, parameter, control_points)``.
  """
  P, u = chord_parameter(points)
  if len(P) < 2:
    raise ValueError("a spline stretch needs at least two distinct points")
  degree = 3
  if n_control is None:
    n_control = max(4, int(np.ceil(len(P) * float(control_ratio))))
  n_control = max(4, int(n_control))
  knots = _knots(n_control, degree)
  B = _basis(u, knots, n_control, degree)

  # A clamped B-spline evaluates to its first/last control point at the endpoints.  Eliminate
  # those two variables so witness equality is algebraic, not a large-weight approximation.
  fixed = B[:, [0]] * P[0] + B[:, [-1]] * P[-1]
  A = B[:, 1:-1]
  Y = P - fixed
  if n_control > 2 and smooth_weight > 0.0:
    D2 = np.diff(np.eye(n_control), n=2, axis=0)
    root = np.sqrt(float(smooth_weight) * max(len(P), 1))
    A = np.vstack([A, root * D2[:, 1:-1]])
    rhs_pen = -root * (D2[:, [0]] * P[0] + D2[:, [-1]] * P[-1])
    Y = np.vstack([Y, rhs_pen])
  interior = np.linalg.lstsq(A, Y, rcond=None)[0]
  control = np.vstack([P[0], interior, P[-1]])
  return BSpline(knots, control, degree, extrapolate=False), u, control


def sample_spline(spline, segments, per_segment=8):
  n = max(1, int(segments) * max(1, int(per_segment)))
  u = np.linspace(0.0, 1.0, n + 1)
  return np.asarray(spline(u), float), u


def _polyline_at(P, u_src, u):
  out = np.empty((len(u), P.shape[1]))
  for j in range(P.shape[1]):
    out[:, j] = np.interp(u, u_src, P[:, j])
  return out


def _certify(Q, free_of, stretch, limits, resolution, joint_tolerance):
  """Audit the emitted polyline, including interpolation between its curve samples."""
  lo = hi = None
  if limits is not None:
    lo, hi = (np.asarray(v, float) for v in limits)
  checked = 0
  # The start witness belongs to this stretch's post-switch mode.  The base preflight checks
  # it too, but retaining the check here makes this module's certificate self-contained.
  q0 = np.asarray(Q[0], float)
  if lo is not None:
    bad = np.flatnonzero((q0 < lo - joint_tolerance) | (q0 > hi + joint_tolerance))
    if len(bad):
      return False, 1, f"joint limit at joint {int(bad[0])}"
  verdict = free_of(q0, stretch)
  checked += 1
  ok = bool(verdict[0]) if isinstance(verdict, tuple) else bool(verdict)
  if not ok:
    return False, checked, "collision at stretch start"
  for edge, (a, b) in enumerate(zip(Q, Q[1:])):
    steps = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / resolution)))
    for j in range(1, steps + 1):
      q = a + (b - a) * (j / steps)
      checked += 1
      if lo is not None:
        bad = np.flatnonzero((q < lo - joint_tolerance) | (q > hi + joint_tolerance))
        if len(bad):
          return False, checked, f"joint limit at joint {int(bad[0])}"
      verdict = free_of(q, stretch)
      ok = bool(verdict[0]) if isinstance(verdict, tuple) else bool(verdict)
      if not ok:
        return False, checked, f"collision on emitted edge {edge} at {j / steps:.3f}"
  return True, checked, ""


def smooth_stretches(stretches, free_of, limits=None, protected=(), per_segment=8,
                     control_ratio=0.55, smooth_weight=0.03, max_deviation=0.20,
                     resolution=0.01, joint_tolerance=1e-9):
  """Fit and certify independent mode-local C2 curves.

  ``stretches`` has the portable ``[(waypoints, event_ids), ...]`` shape used by retiming.
  ``free_of(q, stretch_index)`` supplies the exact mode-aware collision oracle.  Endpoint
  witnesses are copied bit-for-bit.  ``protected`` stretch indices are returned unchanged.

  Three increasingly faithful fits are attempted.  If all fail, the already-certified input
  polyline is retained.  This makes the operation fail-closed and fully optional.
  """
  if resolution <= 0:
    raise ValueError("C2 certification resolution must be positive")
  if per_segment < 1:
    raise ValueError("C2 samples per segment must be positive")
  protected = {int(k) for k in protected}
  out, reports = [], []
  totals = {"fitted": 0, "protected": 0, "fallback": 0, "checked_samples": 0,
            "max_deviation": 0.0, "input_points": 0, "output_points": 0}

  for k, (waypoints, events) in enumerate(stretches):
    raw = [np.asarray(q, float) for q in waypoints]
    totals["input_points"] += len(raw)
    if k in protected or len(_deduplicate(raw)) < 3:
      status = "protected" if k in protected else "linear"
      Q = np.asarray(raw, float)
      reports.append(StretchReport(k, status, len(raw), len(Q)))
      totals["protected"] += int(k in protected)
      totals["output_points"] += len(Q)
      out.append(([q for q in Q], list(events)))
      continue

    P, u_src = chord_parameter(raw)
    schedules = [
        (control_ratio, smooth_weight, None),
        ((1.0 + control_ratio) * 0.5, smooth_weight * 0.2, None),
        (1.0, 0.0, max(4, len(P))),
    ]
    accepted = None
    reason, checked_total, worst_dev, used_ctrl = "", 0, 0.0, 0
    for attempt, (ratio, weight, nctrl) in enumerate(schedules, 1):
      spline, _, control = fit_c2(P, control_ratio=ratio, smooth_weight=weight,
                                  n_control=nctrl)
      Q, uq = sample_spline(spline, len(P) - 1, per_segment=per_segment)
      Q[0], Q[-1] = raw[0].copy(), raw[-1].copy()           # exact witness bits
      baseline = _polyline_at(P, u_src, uq)
      deviation = float(np.linalg.norm(Q - baseline, axis=1).max())
      worst_dev = max(worst_dev, deviation)
      if max_deviation is not None and deviation > float(max_deviation):
        reason = f"deviation {deviation:.3g} > {float(max_deviation):.3g} rad"
        continue
      ok, checked, reason = _certify(Q, free_of, k, limits, float(resolution),
                                     float(joint_tolerance))
      checked_total += checked
      if ok:
        accepted, used_ctrl = Q, len(control)
        reports.append(StretchReport(k, "fitted", len(raw), len(Q), len(control),
                                     attempt, checked_total, deviation, ""))
        totals["fitted"] += 1
        totals["max_deviation"] = max(totals["max_deviation"], deviation)
        break

    if accepted is None:
      accepted = np.asarray(raw, float)
      reports.append(StretchReport(k, "fallback", len(raw), len(accepted), used_ctrl,
                                   len(schedules), checked_total, worst_dev, reason))
      totals["fallback"] += 1
    totals["checked_samples"] += checked_total
    totals["output_points"] += len(accepted)
    out.append(([q for q in accepted], list(events)))
  return out, reports, totals
