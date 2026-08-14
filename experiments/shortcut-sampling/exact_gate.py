"""Exact, vectorized segment-cost gate for the supported mr_bench cost functions.

The native shortcut loop builds the whole candidate segment with Python loops and then calls
``path_cost`` twice to decide whether the candidate improves cost.  For cost functions whose
per-edge value depends only on the per-robot distances of that edge, a straight-line candidate
gives every shortened robot one constant per-edge distance, so the exact old-minus-new segment
cost is a vectorized array expression.  The gate reproduces the native DECISION exactly and is
used only to skip candidate construction for gate-rejected proposals.  Accepted edits still go
through the unchanged native acceptance path, and every reported output still passes the strict
certificate.

Soundness notes
---------------
* The mode-switch ``k-1`` doubling correction of the candidate construction is encoded as
  per-edge multipliers ``m`` in {0, 1, 2}.
* A build-time self check compares the table's total path cost against the real ``path_cost``.
  On mismatch the table marks itself unusable and callers fall back to the native gate.
* Borderline deltas (``|delta|`` at or below ``eps * scale``) are reported as indeterminate;
  the caller must then use the native construction gate, so a last-ulp floating point
  difference can never flip a decision.
* The table is rebuilt (O(path length) numpy work) after every accepted edit, the same order
  of work the landed gain selector already performs with ``full_run_gain``.

Nothing in mr_bench is modified.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# The default smoothing weight of ``batch_config_cost`` for the max+sum reduction.
_W = 0.01

# Deltas at or below this relative magnitude are indeterminate; the caller falls back to the
# native construction gate.
_EPS = 1e-9


class SegmentCostTable:
    """Per-edge cost algebra for one concrete path.

    ``get_path`` returns the current (mutable) path; the table rebuilds its arrays whenever it
    is marked dirty, so callers only need to call ``mark_dirty`` after an accepted edit.
    """

    def __init__(self, env, get_path: Callable[[], List]):
        self.env = env
        self.get_path = get_path
        self.metric = str(getattr(env, "cost_metric", "euclidean"))
        self.reduction = str(getattr(env, "cost_reduction", "max"))
        self.n_robots = int(len(getattr(env, "robots", ())))
        self.ok = False
        self.dirty = True
        self._states = None
        self._modes = None
        self._slices = None
        self._dists = None
        self._switch = None
        if self.metric not in ("euclidean", "max", "max_euclidean"):
            return
        if self.reduction not in ("max", "sum"):
            return
        self.ensure()

    # ------------------------------------------------------------------ build
    def ensure(self) -> bool:
        """Rebuild from the current path when dirty. Returns ``self.ok``."""
        if not self.dirty:
            return self.ok
        self.ok = False
        try:
            self._build()
        except Exception:
            self.ok = False
        self.dirty = False
        return self.ok

    def mark_dirty(self) -> None:
        self.dirty = True

    def _build(self) -> None:
        from multi_robot_multi_goal_planning.problems.util import path_cost

        path = self.get_path()
        states = np.array([s.q.state() for s in path], dtype=np.float64)
        n = len(path)
        self._states = states
        # The candidate construction compares mode OBJECTS with ``!=``, and the Mode class
        # defines no __eq__, so a switch is an identity change, not a task-id change.
        self._modes = [s.mode for s in path]
        self._slices = [
            (int(s), int(e)) for s, e in path[0].q._array_slice
        ]
        n_edges = max(0, n - 1)
        dists = np.zeros((self.n_robots, n_edges), dtype=np.float64)
        for r, (s, e) in enumerate(self._slices):
            diff = np.diff(states[:, s:e], axis=0)
            if self.metric == "euclidean":
                dists[r] = np.sqrt(np.sum(diff * diff, axis=1))
            else:  # max / max_euclidean
                dists[r] = np.max(np.abs(diff), axis=1)
        self._dists = dists
        self._switch = np.zeros(n, dtype=np.int64)
        for k in range(1, n):
            if self._modes[k] is not self._modes[k - 1]:
                self._switch[k] = 1

        total_table = self._edge_cost(self._dists).sum()
        total_real = float(path_cost(path, self.env.batch_config_cost))
        if not np.isclose(total_table, total_real, rtol=1e-7, atol=1e-9):
            self.ok = False
            return
        self.ok = True

    def _edge_mult(self, i: int, j: int) -> np.ndarray:
        """Per-edge multiplier of a constant per-edge candidate distance over [i, j].

        The candidate construction sets the interpolation offset of an INTERIOR mode-switch
        state to ``k-1``.  The per-edge delta of a shortened robot is therefore
        ``(offset(k+1) - offset(k)) * d`` with ``offset(k) = k - c(k)`` and
        ``c(k) = [i < k < j and mode object changed at k]``, which gives
        ``m(k) = 1 - c(k+1) + c(k)`` in {0, 1, 2}.  A switch exactly at the query endpoints
        i or j receives no correction, which the ``k > i`` / ``k+1 < j`` guards encode.
        """
        span = j - i
        k = np.arange(i, j)
        s = self._switch
        m = np.ones(span, dtype=np.float64)
        m = m + np.where(k > i, s[k], 0)
        m = m - np.where(k + 1 < j, s[k + 1], 0)
        return m

    # ------------------------------------------------------------------ algebra
    def _edge_cost(self, dists: np.ndarray) -> np.ndarray:
        """Per-edge cost array for a (n_robots, n_edges) distance matrix."""
        if self.reduction == "max":
            return dists.max(axis=0) + _W * dists.sum(axis=0)
        return dists.sum(axis=0)  # reduction == "sum"

    def _robot_dist(self, r: int, i: int, j: int) -> float:
        s, e = self._slices[r]
        a = self._states[i, s:e]
        b = self._states[j, s:e]
        if self.metric == "euclidean":
            return float(np.linalg.norm(a - b))
        return float(np.max(np.abs(a - b)))

    def _query(self, robots: Sequence[int], i: int, j: int):
        """Return (delta, old_cost, new_cost) for straightening ``robots`` over [i, j]."""
        span = j - i
        slice_ = slice(i, j)
        old = self._dists[:, slice_]
        old_total = float(self._edge_cost(old).sum())
        new = old.copy()
        m = self._edge_mult(i, j)
        for r in robots:
            d_r = self._robot_dist(r, i, j) / span
            new[r] = d_r * m
        new_total = float(self._edge_cost(new).sum())
        return old_total - new_total, old_total, new_total

    # ------------------------------------------------------------------ public
    def old_cost(self, i: int, j: int) -> float:
        self.ensure()
        return float(self._edge_cost(self._dists[:, i:j]).sum())

    def delta(self, robots: Sequence[int], i: int, j: int) -> float:
        """Exact old-minus-new composite cost of straightening ``robots`` over [i, j].

        Positive means the candidate improves cost, matching the native gate
        ``candidate_segment_cost < old_segment_cost``.
        """
        self.ensure()
        delta, _, _ = self._query(robots, i, j)
        return delta

    def decide(self, robots: Sequence[int], i: int, j: int) -> Optional[bool]:
        """True/False gate verdict, or None when indeterminate (caller must fall back)."""
        if not self.ensure():
            return None
        delta, old_total, _ = self._query(robots, i, j)
        scale = max(1.0, abs(old_total))
        if abs(delta) <= _EPS * scale:
            return None
        return delta > 0.0

    def full_run_gains(self, runs_per_robot: Sequence[Sequence[Tuple[int, int]]]) -> Dict:
        """Exact full-run straight-line gains, keyed by ``(robot, (lo, hi))``."""
        self.ensure()
        out: Dict = {}
        for r, runs in enumerate(runs_per_robot):
            for lo, hi in runs:
                out[(r, (int(lo), int(hi)))] = self.delta([r], lo, hi)
        return out

    def total_cost(self) -> float:
        self.ensure()
        return float(self._edge_cost(self._dists).sum())
