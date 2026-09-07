"""Collision-only certificate for a parallel-jaw gripper closing on a grasp.

The assembly configuration deliberately keeps finger joints inactive.  This module does
*not* change that contract: it creates short-lived copies of an already posed collision
scene, translates only the receiver's finger collision proxies through a calibrated virtual
stroke, and asks RAI for contacts.  It is therefore a feasibility certificate, not another
configuration-space dimension, controller, or dynamics simulation.

The current collision proxies are intentionally simple capsules.  The certificate mirrors
that fidelity: it verifies that both proxy fingers can reach the target at one common
aperture and that their sweep does not intersect any non-target geometry.  The caller
supplies a scene in which every relevant part/gripper has already been posed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import robotic as ry


# Bump whenever the calibrated proxy semantics change. It keys the cheap-check cache; this
# first version intentionally does not alter replay semantics or persisted witness data.
VERSION = 1

# These are physical/proxy calibration constants, not search-tuning knobs.  Both current
# models expose an approximately 80--115 mm open jaw gap and have about 40 mm inward travel
# per finger before their capsule proxies meet.  The contact solve below determines the actual
# object-dependent aperture; the cap only detects objects outside the mechanism's range.
_MAX_INWARD = {
    "robotiq": 0.042,
    "openarm": 0.045,
}

_CONTACT_EPS = 7.5e-4       # numerical/mesh contact tolerance (metres)
_MISALIGN_PENETRATION = 1.5e-1
_HARD_PENETRATION = 1.0e-4
_SCAN_STEPS = 7             # coarse brackets; bisection refines each contact afterwards
_BISECTION_STEPS = 4

# The aperture depends only on the gripper/object relative geometry, not on the surrounding
# scene. Keep its inexpensive geometric result across the cheap floating gate and the achieved
# witness gate; each caller still performs its own environment sweep.
_APERTURE_CACHE: Dict[Tuple, ClosureResult] = {}


@dataclass(frozen=True)
class ClosureResult:
    feasible: bool
    aperture: Optional[float] = None
    reason: Optional[str] = None
    detail: Optional[Tuple[str, str, float]] = None
    model: Optional[str] = None


def _quat_to_R(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _target_frame(name: str, target: str) -> bool:
    # Compound collision children use obj_N__shape/... while the logical target is obj_N.
    name = str(name)
    return name == target or name.startswith(target + "__")


def _target_collision_frames(C: ry.Config, target: str):
    """Return only frames that actually carry the target's collision geometry.

    Compound parts use a shapeless logical ``obj_N`` parent and one or more
    ``obj_N__shape*`` children. Setting contact on that logical parent makes RAI hand an
    empty mesh to FCL (``collision object with no vertices``), so it must never be enabled.
    """
    names = [str(n) for n in C.getFrameNames()]
    children = [n for n in names if n.startswith(target + "__shape")]
    return children or ([target] if target in names else [])


def _finger_groups(C: ry.Config, prefix: str):
    """Return the two collision-proxy groups and their model, or a supported no-op vacuum.

    We intentionally identify only the explicit collision capsules, not visual finger meshes
    or inactive mechanism links.  This exactly follows the geometry the existing cheap gate
    and IK collision objective use.
    """
    names = list(C.getFrameNames())
    ours = [n for n in names if str(n).startswith(prefix)]
    if any("vacuum" in str(n) for n in ours):
        return "vacuum", (), None

    left, right = [], []
    for n in ours:
        s = str(n)
        # Robotiq: ...finger1 / ...finger2.  OpenArm: ...finger1_coll[2],
        # ...finger2_coll[12].  Do not include the visual/kinematic fingers.
        if s.endswith("finger1") or "finger1_coll" in s:
            left.append(s)
        elif s.endswith("finger2") or "finger2_coll" in s:
            right.append(s)
    if not left or not right:
        return None, (), None
    model = "openarm" if any("openarm" in n for n in ours) else "robotiq"
    centers = [n for n in ours if str(n).endswith("gripper_center")]
    if not centers:
        return None, (), None
    return model, (tuple(left), tuple(right)), centers[0]


def _copy(C: ry.Config) -> ry.Config:
    out = ry.Config()
    out.addConfigurationCopy(C)
    return out


def _aperture_key(C: ry.Config, model: str, groups, target: str):
    """Pose-invariant geometry signature for a cached bilateral-contact aperture."""
    tf = C.getFrame(target)
    if tf is None:
        return None
    Rt = _quat_to_R(tf.getQuaternion())
    pt = np.asarray(tf.getPosition(), dtype=float)
    rel = []
    for group in groups:
        for name in group:
            f = C.getFrame(name)
            rel.extend(np.round(Rt.T @ (np.asarray(f.getPosition(), dtype=float) - pt), 4))
            rel.extend(np.round((Rt.T @ _quat_to_R(f.getQuaternion())).ravel(), 4))
    shapes = []
    for name in C.getFrameNames():
        n = str(name)
        if n == target or n.startswith(target + "__shape"):
            f = C.getFrame(n)
            try:
                shapes.append((n[len(target):], str(f.getShapeType()),
                               tuple(np.round(f.getSize(), 5))))
            except Exception:
                shapes.append((n[len(target):], "mesh", ()))
    return (model, tuple(shapes), tuple(np.asarray(rel, dtype=float)))


def _finger_setup(C: ry.Config, prefix: str):
    model, groups, center_name = _finger_groups(C, prefix)
    if model == "vacuum":
        return model, groups, center_name, (), (), None
    if model is None:
        return None, (), None, (), (), None
    center = C.getFrame(center_name)
    axis = _quat_to_R(center.getQuaternion())[:, 1]       # repository-wide pinch axis = tool +y
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    positions = {}
    signs = []
    cp = np.asarray(center.getPosition(), dtype=float)
    for group in groups:
        ps = []
        for n in group:
            p = np.asarray(C.getFrame(n).getPosition(), dtype=float)
            positions[n] = p
            ps.append(float(np.dot(p - cp, axis)))
        signs.append(float(np.mean(ps)))
    # A malformed calibration where both finger proxies lie on the same side must never
    # silently pass as a vacuum hand.
    if signs[0] * signs[1] >= -1e-8:
        return None, (), None, (), (), None
    return model, groups, center_name, axis, positions, tuple(1.0 if s > 0 else -1.0 for s in signs)


def _move_to_shift(C: ry.Config, positions: Dict[str, np.ndarray], axis: np.ndarray,
                   groups: Tuple[Tuple[str, ...], Tuple[str, ...]], signs: Tuple[float, float],
                   shift: float):
    """Repose one private work scene; callers own the copy and source C is never mutated."""
    for group, sign in zip(groups, signs):
        for n in group:
            C.getFrame(n).setPosition(positions[n] - sign * float(shift) * axis)
    # The explicit round trip refreshes FCL's broad phase after writes to jointless child
    # frames.  Reusing this one work scene makes the certificate ~an order of magnitude cheaper
    # than copying the full floating scene for every aperture sample.
    C.setJointState(C.getJointState())


def _contacts(C: ry.Config, groups: Tuple[Tuple[str, ...], Tuple[str, ...]], target: str):
    """(per-side target distance, first non-target penetration)."""
    # A positive query margin gives us a smooth-ish signed-distance trace before first touch;
    # RAI returns all pairs inside the margin, including non-penetrating near contacts.
    pairs = list(C.getCollisions(0.01))
    group_of = {n: i for i, g in enumerate(groups) for n in g}
    d_target = [float("inf"), float("inf")]
    blocked = None
    for a, b, d0 in pairs:
        a, b, d = str(a), str(b), float(d0)
        ia, ib = group_of.get(a), group_of.get(b)
        if ia is None and ib is None:
            continue
        # Internal proxy overlaps are irrelevant; the underlying RAI model already excludes
        # them, but this also keeps a copied visual helper from poisoning the certificate.
        if ia is not None and ib is not None:
            continue
        i = ia if ia is not None else ib
        other = b if ia is not None else a
        if _target_frame(other, target):
            d_target[i] = min(d_target[i], d)
        elif d < -_HARD_PENETRATION:
            if blocked is None or d < blocked[2]:
                blocked = ((a, b, d))
    return tuple(d_target), blocked


def _contact_aperture(C: ry.Config, prefix: str, target: str):
    """Infer common inward travel from target contacts alone, plus reusable proxy metadata."""
    model, groups, _center, axis, positions, signs = _finger_setup(C, prefix)
    if model == "vacuum":
        return ClosureResult(True, aperture=None, model=model), None
    if model is None:
        return ClosureResult(False, reason="unsupported-gripper", model=None), None

    key = _aperture_key(C, model, groups, target)
    cached = _APERTURE_CACHE.get(key) if key is not None else None
    if cached is not None:
        return cached, (model, groups, axis, positions, signs)

    max_inward = _MAX_INWARD[model]
    work = _copy(C)
    # --no-always-contact disables the target in the KOMO scene so the ordinary objective can
    # converge.  Closing feasibility still needs to *measure* that target, and this disposable
    # work scene can force it back on without changing the solve scene.
    for n in _target_collision_frames(work, target):
        work.getFrame(n).setContact(1)

    def evaluate(shift):
        _move_to_shift(work, positions, axis, groups, signs, shift)
        return _contacts(work, groups, target)[0]

    samples = np.linspace(0.0, max_inward, _SCAN_STEPS + 1)
    trace = []
    for shift in samples:
        ds = evaluate(float(shift))
        trace.append((float(shift), ds))

    hits = []
    for side in range(2):
        hit = next((j for j, (_s, ds) in enumerate(trace) if ds[side] <= _CONTACT_EPS), None)
        if hit is None:
            res = ClosureResult(False, reason="aperture-out-of-range", model=model)
            if key is not None:
                _APERTURE_CACHE[key] = res
            return res, None
        if hit == 0:
            hits.append(0.0)
            continue
        lo, hi = trace[hit - 1][0], trace[hit][0]
        # Refine the first contact of this individual finger.  This is not a simulated close;
        # it only avoids treating a 6--7mm scan bin as acceptable asymmetric penetration.
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            ds = evaluate(mid)
            if ds[side] <= _CONTACT_EPS:
                hi = mid
            else:
                lo = mid
        hits.append(hi)

    # One mechanically coupled aperture must serve both sides.  If the first contact occurs
    # far earlier on one side, closing to the later contact would drive that finger through the
    # fixed object; do not pretend a disabled joint can individually adapt around it.
    closing = max(hits)
    ds = evaluate(closing)
    if any(d > _CONTACT_EPS for d in ds):
        res = ClosureResult(False, reason="no-bilateral-contact", model=model)
        if key is not None:
            _APERTURE_CACHE[key] = res
        return res, None
    if any(d < -_MISALIGN_PENETRATION for d in ds):
        res = ClosureResult(False, reason="miscentred-contact", model=model)
        if key is not None:
            _APERTURE_CACHE[key] = res
        return res, None

    # ``closing`` is per-finger inward travel.  It is deliberately metadata, not a robot DOF.
    res = ClosureResult(True, aperture=float(closing), model=model)
    if key is not None:
        _APERTURE_CACHE[key] = res
    return res, (model, groups, axis, positions, signs)


def contact_aperture(C: ry.Config, receiver_prefix: str, target: str) -> ClosureResult:
    """Return the object-dependent terminal aperture without checking the environment sweep.

    This is the geometry-only half of :func:`certify`.  Simulation uses it to render a planned
    grasp at the same object-dependent closure while leaving the certified collision proxies
    fixed.  The source configuration is never mutated.
    """
    result, _meta = _contact_aperture(C, receiver_prefix, target)
    return result


def certify(C: ry.Config, receiver_prefix: str, target: str) -> ClosureResult:
    """Certify a virtual parallel-jaw closure in an already posed collision scene.

    Existing holders remain their normal fixed-open collision geometry. This is conservative:
    the receiver must clear their larger open envelope too.

    TODO(gripper-closure): build a second floating-EE collision model with actual finger
    joints and replay each existing holder at its object-dependent aperture. That resolves the
    self-collision exclusion of a held object exactly, without touching the deliberately
    joint-disabled search/IK/simulation models.
    """
    res, meta = _contact_aperture(C, receiver_prefix, target)
    if not res.feasible:
        return res
    if meta is None:                                      # vacuum receiver
        return res
    model, groups, axis, positions, signs = meta

    # One private scene covers the complete receiver sweep. Existing holders intentionally
    # retain their fixed-open collision proxies; see the articulated-shadow TODO above.
    work = _copy(C)
    # Single-arm pick IK temporarily strips idle robots' collision geometry. Re-enable just
    # their finger proxies here: an incoming close must still be checked against every other
    # gripper, without changing the main IK scene or paying for full inactive-arm contacts.
    for n in work.getFrameNames():
        s = str(n)
        if s.endswith("finger1") or s.endswith("finger2") or "finger1_coll" in s or "finger2_coll" in s:
            work.getFrame(n).setContact(1)
    for n in _target_collision_frames(work, target):
        work.getFrame(n).setContact(1)

    # The contact aperture was solved without environmental blockers.  Now certify every point
    # of the actual receiver sweep against holders, the rest of the assembly, table and arms.
    samples = np.linspace(0.0, float(res.aperture or 0.0), _SCAN_STEPS + 1)
    for shift in samples:
        _move_to_shift(work, positions, axis, groups, signs, float(shift))
        _ds, blocked = _contacts(work, groups, target)
        if blocked is not None:
            return ClosureResult(False, reason="sweep-collision", detail=blocked, model=model)
    return res


def closure_sweep_configs(C: ry.Config, receiver_prefix: str, target: str, steps: int = 21):
    """Return scrubbable copies of the exact virtual-close sweep used by :func:`certify`.

    This is a diagnostic only: it never mutates ``C`` and it does not participate in a search
    verdict. Labels expose both target-finger signed distances and the first environmental
    blocker, making a visually plausible but numerically rejected close inspectable.
    """
    result, meta = _contact_aperture(C, receiver_prefix, target)
    if meta is None:                                      # vacuum or unsupported/aperture failure
        return result, [(_copy(C), f"closure unavailable: {result.reason or result.model}")]
    model, groups, axis, positions, signs = meta
    work = _copy(C)
    for n in work.getFrameNames():
        s = str(n)
        if (s.endswith("finger1") or s.endswith("finger2")
                or "finger1_coll" in s or "finger2_coll" in s):
            work.getFrame(s).setContact(1)
    for n in _target_collision_frames(work, target):
        work.getFrame(n).setContact(1)

    aperture = float(result.aperture or 0.0)
    samples = np.linspace(0.0, aperture, max(2, int(steps)))
    frames = []
    for i, shift in enumerate(samples):
        _move_to_shift(work, positions, axis, groups, signs, float(shift))
        distances, blocker = _contacts(work, groups, target)
        ds = ", ".join("inf" if not np.isfinite(d) else f"{1000*d:+.2f}mm"
                       for d in distances)
        status = "clear" if blocker is None else (
            f"BLOCKED {blocker[0]} ~ {blocker[1]} {1000*blocker[2]:+.2f}mm")
        label = (f"virtual close {i + 1}/{len(samples)}: inward={1000*shift:.2f}mm / "
                 f"{1000*aperture:.2f}mm; finger-target=[{ds}]; {status}")
        frames.append((_copy(work), label))
    # The aperture solve can succeed while the environmental sweep fails. Return the same
    # final certificate as search, while retaining every sample for visual diagnosis.
    return certify(C, receiver_prefix, target), frames
