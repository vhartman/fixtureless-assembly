"""Online signed-distance guard for an insertion servo.

Projects the commanded joint increment onto a local gripper/receiver clearance
constraint. It leaves the nominal pose task and mate acceptance test unchanged.
"""
from __future__ import annotations

import mujoco
import numpy as np

from execution.grasp_preflight import (
    _active_collision_geom, _descends_from, _pair_enabled, _tool_root)


def project_box_halfspace(desired, row, lower, bound, *, box_lo=None,
                          box_hi=None):
    """Closest bounded increment satisfying one local clearance halfspace.

    Return the safest box corner when the linearized constraint is infeasible.
    The scalar dual is monotone, so bisection solves this tiny convex QP.
    """
    desired = np.asarray(desired, dtype=float)
    row = np.asarray(row, dtype=float)
    lo = np.full_like(desired, -bound)
    hi = np.full_like(desired, bound)
    if box_lo is not None:
        lo = np.maximum(lo, box_lo)
    if box_hi is not None:
        hi = np.minimum(hi, box_hi)
    result = np.clip(desired, lo, hi)
    if float(row @ result) >= lower:
        return result, False, False
    extreme = np.where(row >= 0, hi, lo)
    if float(row @ extreme) < lower - 1e-10:
        return extreme, True, True
    left, right = 0.0, 1.0
    while float(row @ np.clip(desired + right * row, lo, hi)) < lower:
        right *= 2.0
    for _ in range(28):
        mid = 0.5 * (left + right)
        if float(row @ np.clip(desired + mid * row, lo, hi)) < lower:
            left = mid
        else:
            right = mid
    return np.clip(desired + right * row, lo, hi), True, False


class SeatClearanceFilter:
    def __init__(self, scene, servo, holder, parent, *, margin=0.001,
                 activation=0.012, gain=40.0):
        self.scene, self.servo, self.holder = scene, servo, holder
        self.margin, self.activation, self.gain = margin, activation, gain
        model = scene.model
        root = _tool_root(model, scene._body_id(scene.grasp_anchor[holder]))
        component = set(scene._active_mate_component(parent))
        self.hands = [g for g in range(model.ngeom)
                      if _active_collision_geom(model, g)
                      and _descends_from(model, model.geom_bodyid[g], root)]
        self.receivers = [g for g in range(model.ngeom)
                          if _active_collision_geom(model, g)
                          and scene._part_owner_by_body[int(model.geom_bodyid[g])]
                          in component]
        self.minimum = float("inf")
        self.constrained_ticks = 0

    def _jac(self, body, point, robot):
        model, data = self.scene.model, self.scene.data
        self.infeasible_ticks = 0
        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jp, jr, point, body)
        return jp[:, self.servo.dofs[robot]]

    def __call__(self, q_control, dq, controlled, *, reference=None,
                 measured=None, tracking=None):
        box_lo = box_hi = None
        if tracking is not None:
            if reference is None or measured is None:
                raise ValueError('tracking tube requires measured and reference joints')
            box_lo = np.asarray(measured) - tracking - np.asarray(reference)
            box_hi = np.asarray(measured) + tracking - np.asarray(reference)
        constraint = self.constraint(q_control, controlled)
        if constraint is None:
            return np.clip(dq, np.maximum(-0.02, box_lo) if box_lo is not None else -0.02,
                           np.minimum(0.02, box_hi) if box_hi is not None else 0.02)
        row, lower = constraint
        result, constrained, infeasible = project_box_halfspace(
            dq, row, lower, 0.02, box_lo=box_lo, box_hi=box_hi)
        if constrained:
            self.constrained_ticks += 1
        if infeasible:
            self.infeasible_ticks += 1
        return result

    def constraint(self, q_control, controlled):
        model, data = self.scene.model, self.scene.data
        closest = (self.activation, None, None, None)
        for hand in self.hands:
            xh, rh = data.geom_xpos[hand], model.geom_rbound[hand]
            for receiver in self.receivers:
                if not _pair_enabled(model, hand, receiver):
                    continue
                if (np.linalg.norm(xh - data.geom_xpos[receiver]) - rh
                        - model.geom_rbound[receiver] >= closest[0]):
                    continue
                points = np.empty(6)
                gap = float(mujoco.mj_geomDistance(
                    model, data, hand, receiver, self.activation, points))
                # A few convex mesh pairs return zero with centimetre-separated
                # witnesses: no valid signed-distance derivative exists there.
                if abs(np.linalg.norm(points[:3] - points[3:]) - abs(gap)) > 1e-4:
                    continue
                if gap < closest[0]:
                    closest = (gap, hand, receiver, points)
        gap, hand, receiver, points = closest
        if hand is None:
            return None
        self.minimum = min(self.minimum, gap)
        sep = points[:3] - points[3:]
        nsep = float(np.linalg.norm(sep))
        if nsep < 1e-8:
            return None
        normal = sep / nsep * (-1.0 if gap < 0 else 1.0)
        row = np.zeros(len(q_control))
        offset = 0
        for robot in controlled:
            n = len(self.servo.qidx[robot])
            if robot == self.holder:
                jac = self._jac(int(model.geom_bodyid[hand]), points[:3], robot)
                row[offset:offset + n] = normal @ jac
            else:
                # The receiver is a free body welded to its holder. Differentiate
                # its contact point through that holder's grasp anchor.
                anchor = self.scene._body_id(self.scene.grasp_anchor[robot])
                jac = self._jac(anchor, points[3:], robot)
                row[offset:offset + n] = -(normal @ jac)
            offset += n
        strength = float(row @ row)
        if strength < 1e-10:
            return None
        # Discrete control barrier: d_dot >= -gain (d - margin).
        lower = -self.gain * (gap - self.margin) * self.servo.dt
        return row, lower
