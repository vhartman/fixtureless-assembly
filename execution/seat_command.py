"""One local command projection for the actuated insertion skill."""
from __future__ import annotations

import numpy as np

from execution.seat_clearance import project_box_halfspace


class SeatCommandGovernor:
    """Project resolved-rate IK onto command and clearance constraints.

    The per-tick problem has a box and at most one linearized clearance
    halfspace. Its solution is a scalar-dual projection, not a generic QP solve.
    """

    def __init__(self, dt, *, rate=10.0, acceleration=300.0, tracking=0.03):
        self.dt = float(dt)
        self.rate = float(rate)
        self.acceleration = float(acceleration)
        self.tracking = float(tracking)
        self.previous_velocity = None
        self.infeasible_ticks = 0
        self.clearance_ticks = 0

    def step(self, desired, reference, measured, *, constraint=None):
        desired = np.asarray(desired, dtype=float)
        reference = np.asarray(reference, dtype=float)
        measured = np.asarray(measured, dtype=float)
        previous = (np.zeros_like(desired) if self.previous_velocity is None
                    else self.previous_velocity)
        rate_step = self.rate * self.dt
        accel_step = self.acceleration * self.dt * self.dt
        lo = np.maximum.reduce((-np.full_like(desired, rate_step),
                                previous * self.dt - accel_step,
                                measured - self.tracking - reference))
        hi = np.minimum.reduce((np.full_like(desired, rate_step),
                                previous * self.dt + accel_step,
                                measured + self.tracking - reference))
        conflict = lo > hi
        if np.any(conflict):
            self.infeasible_ticks += 1
            dynamic_lo = np.maximum(-rate_step, previous * self.dt - accel_step)
            dynamic_hi = np.minimum(rate_step, previous * self.dt + accel_step)
            recovery = np.clip(measured - reference, dynamic_lo, dynamic_hi)
            lo[conflict] = recovery[conflict]
            hi[conflict] = recovery[conflict]
        if constraint is None:
            result = np.clip(desired, lo, hi)
        else:
            row, lower = constraint
            result, active, infeasible = project_box_halfspace(
                desired, row, lower, rate_step, box_lo=lo, box_hi=hi)
            self.clearance_ticks += int(active)
            self.infeasible_ticks += int(infeasible)
        self.previous_velocity = result / self.dt
        return result
