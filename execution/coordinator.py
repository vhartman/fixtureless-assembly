"""Path-progress policies for an execution program.

Only the synchronized reference policy is implemented. An asynchronous policy
must provide a collision-certified admissible next phase for the observed plant
state; it cannot merely reuse a nominal offline timing trace.
"""
from __future__ import annotations

import numpy as np

from assembly_planner.retiming import _accel_profile


class LockstepCoordinator:
    """Advance continuously over one saved free-motion segment.

    Assign one scalar clock to the entire polyline, not one stop/start profile
    to every motion-planner sample. The exact saved geometric path is preserved,
    including its skill-boundary endpoint. Joint and optional tool speed are
    bounded on every edge. Acceleration is bounded in scalar path progress;
    a sharp C0 geometric corner can still change joint velocity abruptly and
    requires separately certified geometric smoothing to remove.
    """
    def __init__(self, dt: float, joint_rate: float, joint_accel: float = 2.0):
        if dt <= 0 or joint_rate <= 0 or joint_accel <= 0:
            raise ValueError('dt, joint rate and acceleration must be positive')
        self.dt = float(dt)
        self.joint_rate = float(joint_rate)
        self.joint_accel = float(joint_accel)

    def samples(self, waypoints, *, cartesian_positions=None, cartesian_rate=None):
        points = np.asarray(waypoints, dtype=float)
        if points.ndim != 2 or len(points) == 0 or not np.all(np.isfinite(points)):
            raise ValueError('a motion segment needs finite joint waypoints of equal size')
        if len(points) == 1:
            yield points[0]
            return
        if (cartesian_positions is None) != (cartesian_rate is None):
            raise ValueError('Cartesian waypoints and rate must be supplied together')
        edges = np.diff(points, axis=0)
        lengths = np.max(np.abs(edges), axis=1)
        if cartesian_positions is not None:
            positions = np.asarray(cartesian_positions, dtype=float)
            if (cartesian_rate <= 0 or positions.shape != (len(points), 3)
                    or not np.all(np.isfinite(positions))):
                raise ValueError('invalid Cartesian path or rate')
            tool_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            # Effective arc length makes either joint or Cartesian velocity
            # binding locally, without slowing unrelated edges to the tightest
            # approach constraint.
            lengths = np.maximum(lengths, self.joint_rate * tool_lengths / cartesian_rate)
        distance = float(np.sum(lengths))
        if distance <= 1e-12:
            yield points[-1]
            return
        cumulative = np.r_[0.0, np.cumsum(lengths)]
        for arc in _accel_profile(distance, self.dt, self.joint_rate, self.joint_accel):
            if arc >= distance - 1e-12:
                yield points[-1]
                continue
            edge = min(int(np.searchsorted(cumulative, arc, side='right')) - 1,
                       len(edges) - 1)
            width = float(lengths[edge])
            alpha = 1.0 if width <= 1e-12 else (arc - cumulative[edge]) / width
            yield points[edge] + alpha * edges[edge]
