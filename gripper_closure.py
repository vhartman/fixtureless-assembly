"""Collision-only certificate for a parallel-jaw gripper closing on a grasp.

The assembly configuration deliberately keeps finger joints inactive. This module does
not change that contract: it creates disposable posed scenes and asks RAI for contacts.
Ordinary pick checks translate pad proxies through a calibrated virtual stroke. Robotiq
handover checks instead rotate the full finger linkage through the same mimic map as
execution, so the two hands are checked in their actual closed geometry. The current
OpenArm floating proxy has no hinges and retains the translated-pad certificate. Neither
changes the planner's configuration space or simulates dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import robotic as ry


# Bump whenever the calibrated proxy semantics change. It keys the cheap-check cache and the
# sampled grasp database provenance, so a saved grasp id cannot silently acquire new meaning.
VERSION = 9

# These are physical/proxy calibration constants, not search-tuning knobs.  Both current
# models expose an approximately 80--115 mm open jaw gap and have about 40 mm inward travel
# per finger before their capsule proxies meet.  The contact solve below determines the actual
# object-dependent aperture; the cap only detects objects outside the mechanism's range.
_MAX_INWARD = {
    "robotiq": 0.042,
    "openarm": 0.045,
}

_CONTACT_EPS = 7.5e-4       # numerical/mesh contact tolerance (metres)
# At the common mechanically-coupled aperture, the jaw that touched first may penetrate the
# target only by this small collision-query allowance.  The old value was 0.15m, effectively
# disabling the bilateral-centering certificate and letting a jaw close centimetres through an
# off-centre part while waiting for the opposite jaw to make contact.
_MISALIGN_PENETRATION = 1.5e-3
_HARD_PENETRATION = 1.0e-4
_SCAN_STEPS = 7             # coarse brackets; bisection refines each contact afterwards
_BISECTION_STEPS = 4

# The aperture depends only on the gripper/object relative geometry, not on the surrounding
# scene. Keep its inexpensive geometric result across the cheap floating gate and the achieved
# witness gate; each caller still performs its own environment sweep.
_APERTURE_CACHE: Dict[Tuple, ClosureResult] = {}
_APERTURE_CACHE_LIMIT = 4096

# Disposable closure scenes are copied thousands of times while constructing a sampled grasp
# database, but their frame *schema* is identical.  Discovering the two finger groups by
# repeatedly scanning every frame was consequently a surprising preprocessing hotspot.  Key the
# structural result by the complete ordered frame-name tuple: unlike a Config id this is safe
# across short-lived copies and cannot alias configurations with different geometry names.
_FINGER_GROUP_CACHE: Dict[Tuple[str, Tuple[str, ...]], Tuple] = {}


@dataclass(frozen=True)
class ClosureResult:
    feasible: bool
    aperture: Optional[float] = None
    reason: Optional[str] = None
    detail: Optional[Tuple[str, str, float]] = None
    model: Optional[str] = None
    # Signed translation of the tool centre along its local +y/closing axis which equalises
    # the two first-contact travels.  It is reported only for a miscentred bilateral grasp;
    # callers may generate that corrected pose and must then certify it from scratch.
    centering_offset: Optional[float] = None


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
    names = tuple(str(n) for n in C.getFrameNames())
    key = (str(prefix), names)
    cached = _FINGER_GROUP_CACHE.get(key)
    if cached is not None:
        return cached

    ours = [n for n in names if n.startswith(prefix)]
    if any("vacuum" in n for n in ours):
        result = ("vacuum", (), None)
        _FINGER_GROUP_CACHE[key] = result
        return result

    left, right = [], []
    robotiq_left_pad, robotiq_right_pad = [], []
    for n in ours:
        s = n
        # Robotiq: prefer the manufacturer's flat inner-pad boxes whenever they are present.
        # The historical finger1/finger2 capsules use the opposite left/right naming from the
        # imported linkage (finger1 is parented to right_inner_finger_joint).  Mixing capsules
        # and pads in one side therefore averaged geometry from *opposite* jaws and could
        # certify a grasp that the articulated viewer showed 1--3 cm off centre.  Pads are the
        # actual pinch surfaces; capsules remain only the fallback for stripped models.
        # OpenArm: ...finger1_coll[2], ...finger2_coll[12].  The pads are normally visual
        # only; this disposable certificate enables them as collision geometry.  Including
        # them is what makes simultaneous handover closure agree with MuJoCo instead of
        # certifying two capsule proxies whose actual pad boxes overlap.
        if "left_inner_finger_pad" in s:
            robotiq_left_pad.append(s)
        elif "right_inner_finger_pad" in s:
            robotiq_right_pad.append(s)
        elif s.endswith("finger1") or "finger1_coll" in s:
            left.append(s)
        elif s.endswith("finger2") or "finger2_coll" in s:
            right.append(s)
    if robotiq_left_pad and robotiq_right_pad:
        left, right = robotiq_left_pad, robotiq_right_pad
    if not left or not right:
        result = (None, (), None)
        _FINGER_GROUP_CACHE[key] = result
        return result
    model = "openarm" if any("openarm" in n for n in ours) else "robotiq"
    centers = [n for n in ours if n.endswith("gripper_center")]
    if not centers:
        result = (None, (), None)
        _FINGER_GROUP_CACHE[key] = result
        return result
    result = (model, (tuple(left), tuple(right)), centers[0])
    _FINGER_GROUP_CACHE[key] = result
    return result


def finger_sweep_radius_bound(C: ry.Config, prefix: str) -> Optional[float]:
    """Conservative tool-centred sphere containing every finger proxy during closure.

    This is only a broad-phase *non-intersection* certificate. Unknown shape kinds return
    None so the caller falls back to the exact closure test. Full rather than half primitive
    dimensions cover RAI capsule and rounded-box size conventions.
    """
    model, groups, center_name = _finger_groups(C, prefix)
    if model not in _MAX_INWARD or center_name is None:
        return None
    centre = np.asarray(C.getFrame(center_name).getPosition(), dtype=float)
    bound = 0.0
    for group in groups:
        for name in group:
            frame = C.getFrame(name)
            shape = str(frame.getShapeType()).lower()
            size = np.asarray(frame.getSize(), dtype=float).ravel()
            if not np.all(np.isfinite(size)) or np.any(size < 0.0):
                return None
            if "capsule" in shape and size.size >= 2:
                local_radius = float(np.sum(size))
            elif "box" in shape and size.size >= 3:
                local_radius = float(np.sum(size))
            else:
                return None
            distance = float(np.linalg.norm(
                np.asarray(frame.getPosition(), dtype=float) - centre))
            bound = max(bound, distance + local_radius + _MAX_INWARD[model])
    return bound + 1e-3  # geometric/numeric cushion; false negatives are not permitted


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
    def geometry(name):
        f = C.getFrame(name)
        shape = str(f.getShapeType())
        mesh = None
        if "mesh" in shape.lower():
            # Equal bounding sizes do not imply equal convex collision geometry.
            mesh = hashlib.blake2b(np.asarray(f.getMeshPoints(), dtype=float).tobytes(),
                                   digest_size=16).digest()
        size = f.getSize()
        return (shape, () if size is None else tuple(np.asarray(size, dtype=float).ravel()), mesh,
                tuple(np.round(Rt.T @ (np.asarray(f.getPosition()) - pt), 12)),
                tuple(np.round((Rt.T @ _quat_to_R(f.getQuaternion())).ravel(), 12)))

    fingers = tuple(tuple(geometry(name) for name in group) for group in groups)
    shapes = []
    for name in C.getFrameNames():
        n = str(name)
        if n == target or n.startswith(target + "__shape"):
            shapes.append((n[len(target):], geometry(n)))
    return (model, tuple(shapes), fingers)


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


def _contacts(C: ry.Config, groups: Tuple[Tuple[str, ...], Tuple[str, ...]], target: str,
              *, capture_pairs=None):
    """(per-side target distance, first non-target penetration)."""
    # A positive query margin gives us a smooth-ish signed-distance trace before first touch;
    # RAI returns all pairs inside the margin, including non-penetrating near contacts.
    pairs = list(C.getCollisions(0.01))
    if capture_pairs is not None:
        capture_pairs.extend(pairs)
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
    # Queries at distinct IK iterates need not live for the whole multi-milestone search.
    # FIFO eviction changes only reuse, never a certificate or the optimization result.
    if key is not None and len(_APERTURE_CACHE) >= _APERTURE_CACHE_LIMIT:
        _APERTURE_CACHE.pop(next(iter(_APERTURE_CACHE)))

    max_inward = _MAX_INWARD[model]
    work = _copy(C)
    # --no-always-contact disables the target in the KOMO scene so the ordinary objective can
    # converge.  Closing feasibility still needs to *measure* that target, and this disposable
    # work scene can force it back on without changing the solve scene.
    for n in _target_collision_frames(work, target):
        work.getFrame(n).setContact(1)
    for group in groups:
        for n in group:
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
        # ``ds`` is evaluated at the later of the two individual first contacts.  A deep
        # negative value on the earlier side is therefore exactly the error we care about:
        # the fixed grasp transform is not centred between a pair of coupled jaws.  Rejecting
        # it here handles sampled/canonical candidates cheaply and is repeated on the achieved
        # IK witness, so optimizer residual cannot reintroduce the same defect.
        side = int(np.argmin(ds))
        res = ClosureResult(False, reason="miscentred-contact",
                            detail=(f"finger-side-{side}", target, float(ds[side])),
                            model=model,
                            centering_offset=0.5 * float(hits[0] - hits[1]))
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


def certify(C: ry.Config, receiver_prefix: str, target: str,
            existing_closures=(), *, closing=True) -> ClosureResult:
    """Certify a virtual parallel-jaw closure in an already posed collision scene.

    ``existing_closures`` contains ``(prefix, contacted_part)`` pairs for hands already
    holding the component.  Their proxies are first moved to their own object-dependent
    apertures.  This is essential for handovers: two individually valid pinches can have
    intersecting closed fingers even though their larger *open* proxy envelopes happen not to
    intersect.  The search model keeps finger joints disabled; this disposable scene remains
    a deterministic geometric certificate rather than another configuration-space dimension.
    """
    res, meta = _contact_aperture(C, receiver_prefix, target)
    if not res.feasible:
        return res
    if meta is None:                                      # vacuum receiver
        return res
    model, groups, axis, positions, signs = meta

    # One private scene covers the existing closed holders plus the complete receiver sweep.
    work = _copy(C)
    # Single-arm pick IK temporarily strips idle robots' collision geometry. Re-enable just
    # their finger proxies here: an incoming close must still be checked against every other
    # gripper, without changing the main IK scene or paying for full inactive-arm contacts.
    for n in work.getFrameNames():
        s = str(n)
        if (s.endswith("finger1") or s.endswith("finger2")
                or "finger1_coll" in s or "finger2_coll" in s
                or "inner_finger_pad" in s):
            work.getFrame(n).setContact(1)
    for n in _target_collision_frames(work, target):
        work.getFrame(n).setContact(1)

    # Pose every existing parallel-jaw holder at the same terminal aperture used to certify
    # its own grasp. A malformed/stale holder is left at its conservative open geometry: the
    # receiver still has to clear it, and no new action is rejected merely because a historic
    # state predates this gate.
    for holder_prefix, holder_target in existing_closures:
        if holder_prefix == receiver_prefix:
            continue
        holder_res, holder_meta = _contact_aperture(work, holder_prefix, holder_target)
        if not holder_res.feasible or holder_meta is None:
            continue
        _hm, holder_groups, holder_axis, holder_positions, holder_signs = holder_meta
        _move_to_shift(work, holder_positions, holder_axis, holder_groups, holder_signs,
                       float(holder_res.aperture or 0.0))

    # Acquisition checks the whole close. An already-held assembly endpoint does not reopen
    # the fingers: only its closed geometry must clear the newly assembled neighbours.
    samples = (np.linspace(0.0, float(res.aperture or 0.0), _SCAN_STEPS + 1)
               if closing else [float(res.aperture or 0.0)])
    for shift in samples:
        _move_to_shift(work, positions, axis, groups, signs, float(shift))
        _ds, blocked = _contacts(work, groups, target)
        if blocked is not None:
            return ClosureResult(False, reason="sweep-collision", detail=blocked, model=model)
    return res


def certify_handover(C: ry.Config, giver_prefix: str, giver_target: str,
                     receiver_prefix: str, receiver_target: str) -> ClosureResult:
    """Certify a handover with the same articulated finger linkage used in MuJoCo.

    Search keeps finger joints disabled, so this disposable RAI scene rotates their frames
    through the manufacturer's mimic map. A translated pad/capsule approximation misses the
    giver-finger blockage that a rotating receiving pad encounters during a real close.
    """
    from arm_model import gripper_sim_joints
    from problem import _quat_mul

    work = _copy(C)
    hands = []
    for prefix, target in ((giver_prefix, giver_target),
                           (receiver_prefix, receiver_target)):
        setup = _finger_setup(work, prefix)
        model, pads = setup[:2]
        joints = gripper_sim_joints(work, prefix)
        if model is None or model == "vacuum":
            return ClosureResult(False, reason="unsupported-articulated-gripper", model=model)
        def pad_moves_with_joint(name):
            parent = work.getFrame(name).getParent()
            while parent is not None:
                if parent.name in joints:
                    return True
                parent = parent.getParent()
            return False

        if not joints or not all(any(pad_moves_with_joint(n) for n in group)
                                 for group in pads):
            # OpenArm's floating EE declares finger hinges for its visual meshes, but its
            # collision capsules are still rigid siblings of those hinges. Rotating them
            # would not check physical closure. Preserve the existing proxy certificate
            # until the duplicate RAI model attaches collision fingers to its hinges.
            return certify(C, receiver_prefix, receiver_target,
                           existing_closures=((giver_prefix, giver_target),))
        bases = {name: np.asarray(work.getFrame(name).getRelativeQuaternion(), dtype=float)
                 for name in joints}
        names = set()
        for name in work.getFrameNames():
            name = str(name)
            if not name.startswith(prefix):
                continue
            info = work.getFrame(name).info()
            if (name in pads[0] or name in pads[1] or
                    (info.get("contact", 0) and not info.get("visual", False))):
                names.add(name)
                work.getFrame(name).setContact(1)
        for name in _target_collision_frames(work, target):
            work.getFrame(name).setContact(1)
        hands.append((prefix, target, model, pads, joints, bases, names))

    def pose(hand, angle):
        for name, (axis, gain, _max_angle, _side) in hand[4].items():
            half = float(gain) * float(angle) / 2.0
            delta = np.r_[np.cos(half), np.asarray(axis, dtype=float) * np.sin(half)]
            work.getFrame(name).setRelativeQuaternion(_quat_mul(hand[5][name], delta))
        work.setJointState(work.getJointState())

    def contact_angle(hand):
        key_body = _aperture_key(work, hand[2] + "/articulated", hand[3], hand[1])
        key = None if key_body is None else ("joint", key_body)
        cached = _APERTURE_CACHE.get(key) if key is not None else None
        if cached is not None:
            return cached
        limit = min(float(spec[2]) for spec in hand[4].values())
        samples = np.linspace(0.0, limit, _SCAN_STEPS + 1)
        trace = []
        for q in samples:
            pose(hand, q)
            trace.append(_contacts(work, hand[3], hand[1])[0])
        hits = []
        for side in range(2):
            hit = next((i for i, ds in enumerate(trace) if ds[side] <= _CONTACT_EPS), None)
            if hit is None:
                result = ClosureResult(False, reason="aperture-out-of-range", model=hand[2])
                break
            lo = hi = 0.0 if hit == 0 else float(samples[hit])
            if hit:
                lo = float(samples[hit - 1])
                for _ in range(_BISECTION_STEPS):
                    mid = (lo + hi) / 2.0
                    pose(hand, mid)
                    ds, _blocked = _contacts(work, hand[3], hand[1])
                    if ds[side] <= _CONTACT_EPS:
                        hi = mid
                    else:
                        lo = mid
            hits.append(hi)
        else:
            angle = max(hits)
            pose(hand, angle)
            ds, _blocked = _contacts(work, hand[3], hand[1])
            if any(d > _CONTACT_EPS for d in ds):
                result = ClosureResult(False, reason="no-bilateral-contact", model=hand[2])
            elif any(d < -_MISALIGN_PENETRATION for d in ds):
                result = ClosureResult(False, reason="miscentred-contact", model=hand[2],
                                       detail=("finger-side", hand[1], float(min(ds))))
            else:
                result = ClosureResult(True, aperture=angle, model=hand[2])
        pose(hand, 0.0)
        if key is not None:
            if len(_APERTURE_CACHE) >= _APERTURE_CACHE_LIMIT:
                _APERTURE_CACHE.pop(next(iter(_APERTURE_CACHE)))
            _APERTURE_CACHE[key] = result
        return result

    giver, receiver = hands
    giver_result = contact_angle(giver)
    if not giver_result.feasible:
        return giver_result
    receiver_result = contact_angle(receiver)
    if not receiver_result.feasible:
        return receiver_result
    pose(giver, giver_result.aperture)
    steps = max(_SCAN_STEPS, int(np.ceil(receiver_result.aperture / 0.05)))
    for angle in np.linspace(0.0, receiver_result.aperture, steps + 1):
        pose(receiver, angle)
        pairs = []
        _ds, blocked = _contacts(work, receiver[3], receiver[1], capture_pairs=pairs)
        if blocked is not None:
            return ClosureResult(False, reason="sweep-collision", detail=blocked,
                                 model=receiver[2])
        for a, b, depth in pairs:
            a, b = str(a), str(b)
            if ((a in giver[6] and b in receiver[6]) or
                    (b in giver[6] and a in receiver[6])) and depth < -_HARD_PENETRATION:
                return ClosureResult(False, reason="handover-hand-collision",
                                     detail=(a, b, float(depth)), model=receiver[2])
    return receiver_result


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
                or "finger1_coll" in s or "finger2_coll" in s
                or "inner_finger_pad" in s):
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
