"""One-shot geometric certificate for an achieved grasp at a mate target.

This projects the *current* tool geometry with the measured tool-to-part
transform. It deliberately does not mutate the live MuJoCo state or assert
that a collision-free endpoint implies a collision-free insertion trajectory.
"""
from __future__ import annotations

import mujoco
import numpy as np

import sim


_TOOL_TOKENS = ("gripper", "robotiq", "finger", "palm", "vacuum", "ee_coll")


def _body_name(model, body):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or ""


def _tool_root(model, anchor):
    """Highest hand-named ancestor; includes fixed and articulated finger siblings."""
    root = int(anchor)
    while True:
        parent = int(model.body_parentid[root])
        if parent <= 0 or not any(t in _body_name(model, parent).lower()
                                  for t in _TOOL_TOKENS):
            return root
        root = parent


def _descends_from(model, body, root):
    body = int(body)
    while body > 0:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return False


def _active_collision_geom(model, geom):
    return (int(model.geom_contype[geom]) != 0
            or int(model.geom_conaffinity[geom]) != 0)


def _pair_enabled(model, a, b):
    return bool((int(model.geom_contype[a]) & int(model.geom_conaffinity[b]))
                or (int(model.geom_contype[b]) & int(model.geom_conaffinity[a])))


def mate_gripper_clearance(scene, robot, parent, child, child_target,
                           *, max_distance=0.1):
    """Return closest tool/receiver gap if the achieved grasp seats at child_target.

    The tool is rigidly projected from its measured pose. Receiving bodies
    remain at their *measured* poses; this is the collision question the
    controller will face now, rather than the nominal witness geometry.
    """
    model, data = scene.model, scene.data
    anchor = scene.grasp_anchor[robot]
    root = _tool_root(model, scene._body_id(anchor))
    component = set(scene._active_mate_component(parent))
    hands = [g for g in range(model.ngeom)
             if _active_collision_geom(model, g)
             and _descends_from(model, model.geom_bodyid[g], root)]
    receivers = [g for g in range(model.ngeom)
                 if _active_collision_geom(model, g)
                 and scene._part_owner_by_body[int(model.geom_bodyid[g])] in component]
    if not hands or not receivers:
        raise ValueError(f"{robot}/{parent}: no collision geometry for mate preflight")

    anchor_now = scene.part_pose(anchor)
    child_now = scene.part_pose(child)
    achieved = sim._pose_compose(sim._pose_invert(anchor_now), child_now)
    anchor_goal = sim._pose_compose(
        np.asarray(child_target, dtype=float), sim._pose_invert(achieved))
    delta = sim._pose_compose(anchor_goal, sim._pose_invert(anchor_now))
    rotation = sim._quat_to_R(delta[3:])
    translation = delta[:3]

    # mj_geomDistance uses geom_xpos/xmat. A scratch MjData keeps this pure:
    # no arm, part, contact, equality or control state is changed in the plant.
    query = mujoco.MjData(model)
    query.geom_xpos[:] = data.geom_xpos
    query.geom_xmat[:] = data.geom_xmat
    for g in hands:
        query.geom_xpos[g] = rotation @ data.geom_xpos[g] + translation
        query.geom_xmat[g] = (rotation @ data.geom_xmat[g].reshape(3, 3)).ravel()

    nearest = float(max_distance)
    pair = None
    pad_nearest = float(max_distance)
    pad_geoms = {int(g) for side in getattr(scene, '_gripper_contact_geoms', {}).get(robot, ())
                 for g in side}
    witness = np.empty(6)
    for hand in hands:
        for receiver in receivers:
            if not _pair_enabled(model, hand, receiver):
                continue
            gap = float(mujoco.mj_geomDistance(
                model, query, hand, receiver, max_distance, witness))
            # MuJoCo can return a zero sentinel for an unsupported convex-mesh
            # pair while its witness points are centimetres apart. This is
            # not a usable collision distance.
            if abs(np.linalg.norm(witness[:3] - witness[3:]) - abs(gap)) > 1e-4:
                continue
            if hand in pad_geoms:
                pad_nearest = min(pad_nearest, gap)
            if gap < nearest:
                nearest = gap
                pair = (str(_body_name(model, model.geom_bodyid[hand])),
                        str(_body_name(model, model.geom_bodyid[receiver])))
    return {"clearance_m": nearest, "pair": pair,
            "pad_clearance_m": pad_nearest,
            "hand_geoms": len(hands), "receiver_geoms": len(receivers)}
