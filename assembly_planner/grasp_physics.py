"""Isolated, deterministic MuJoCo certificates for sampled parallel-jaw grasps.

This module deliberately answers a smaller question than the wrench LP: can the authored
object-to-tool pose be acquired and retained by the *actual articulated fingers*?  It does
not estimate a grasp-specific wrench envelope.  Accepted grasps continue to use the one
global envelope selected in :mod:`wrench`, which keeps the task planner's contact model
finite, reproducible, and independent of simulator identification noise.

The checker is scene-independent by construction.  Other parts, the table, and non-pad arm
geometry are present in the compiled model (so the exact exported linkage is reused) but are
disabled for this isolated contact experiment.  Scene clearance remains the responsibility
of the existing cheap floating-EE gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import os
import re
import time
from typing import Dict

import numpy as np


# A physical mate needs room for the closed hand at the exact datum.
MATE_CLEARANCE_MIN = 0.001
HANDOVER_CERTIFICATE_CONTRACT = "handover-physics-v2"

@dataclass(frozen=True)
class GraspPhysicsResult:
  feasible: bool
  reason: str
  left_force_n: float = 0.0
  right_force_n: float = 0.0
  acquisition_shift_m: float = 0.0
  retention_shift_m: float = 0.0
  retention_angle_deg: float = 0.0
  load_direction: str = ""

  def metadata(self):
    out = asdict(self)
    # Stable, compact saved-plan diagnostics.  These values do not define grasp identity;
    # the Boolean certificate does.
    for key in ("left_force_n", "right_force_n"):
      out[key] = round(float(out[key]), 3)
    for key in ("acquisition_shift_m", "retention_shift_m"):
      out[key] = round(float(out[key]), 6)
    out["retention_angle_deg"] = round(float(out["retention_angle_deg"]), 3)
    return out


def _quat_angle(a, b):
  dot = min(1.0, abs(float(np.dot(np.asarray(a, float), np.asarray(b, float)))))
  return 2.0 * math.acos(dot)


class GraspPhysicsChecker:
  """One compiled isolated grasp world per representative two-finger robot.

  Compilation is lazy because most runs never enable the experimental gate.  A single model
  is reset and reused for every pose and every part in the assembly; candidate checks therefore
  pay only MuJoCo stepping, not XML compilation.
  """

  CLOSE_TIME = 0.30
  SETTLE_TIME = 0.08
  RETAIN_TIME = 0.15
  # The physical hand may remove a small sampling/discretisation error while closing, but the
  # certificate must not turn a visibly off-centre authored pose into a valid grasp by dragging
  # the payload through a finger.  Use the same 3 mm geometric tolerance as retention below;
  # the simulator's looser 10 mm capture radius is an execution recovery bound, not a grasp-
  # quality criterion.
  MAX_ACQUISITION_SHIFT = 0.003
  MAX_RETENTION_SHIFT = 0.003
  MAX_RETENTION_ANGLE = np.deg2rad(5.0)

  LOAD_DIRECTIONS = (
      ("+x", np.array([1.0, 0.0, 0.0])),
      ("-x", np.array([-1.0, 0.0, 0.0])),
      ("+y", np.array([0.0, 1.0, 0.0])),
      ("-y", np.array([0.0, -1.0, 0.0])),
      ("+z", np.array([0.0, 0.0, 1.0])),
      ("-z", np.array([0.0, 0.0, -1.0])),
  )

  def __init__(self, C, assembly, state, robot, force_n=10.0):
    import mujoco
    import assembly_planner.problem as problem
    import sim

    self.mujoco = mujoco
    self.problem = problem
    self.sim = sim
    self.C = C
    self.assembly = assembly
    self.state = state
    self.robot = robot
    self.force_n = float(force_n)
    self.q0 = np.asarray(C.getJointState(), dtype=float).copy()

    arms = sim._arm_shape_frames(C, [robot], visual=False)
    frame_names = set(C.getFrameNames())
    semantic = next((n for n in (f"{robot}_ur_gripper_center", f"{robot}_ur_vacuum")
                     if n in frame_names), None)
    if semantic is None:
      raise ValueError(f"{robot}: no semantic tool-centre frame for physics grasp check")
    self.anchor = semantic

    def fk_poses():
      poses = {entry[1]: np.asarray(C.getFrame(entry[1]).getPose(), dtype=float)
               for entry in arms}
      poses[semantic] = np.asarray(C.getFrame(semantic).getPose(), dtype=float)
      return poses

    self.scene = sim.SimScene(
        assembly, state, [robot], gripper_kinds={robot: "two_finger"},
        arms=arms, arm_poses=fk_poses(), ee_anchor={robot: semantic},
        articulated_C=C, actuated=True,
        planner_table_top=sim.planner_table_top(C),
        table_geometry=sim.planner_table_geometry(C),
        animate_grippers=True, contact_grasps=True)
    if not all(self.scene._gripper_contact_geoms.get(robot, ((), ()) )):
      raise ValueError(f"{robot}: exact articulated gripper has no bilateral pad geometry")

    # Cache the model's immutable contact masks.  Each query enables exactly the target part
    # and both physical pads; all scene-dependent collision checking remains outside here.
    self._all_contype = np.asarray(self.scene.model.geom_contype, dtype=int).copy()
    self._all_conaffinity = np.asarray(self.scene.model.geom_conaffinity, dtype=int).copy()
    self._cache: Dict[tuple, GraspPhysicsResult] = {}
    self._acquired = {}  # physically closed hand states for geometric mate queries
    self._mate_clearance = {}

    # q_correction is part of the exact RobotIK/tool convention.  Omitting it would test a
    # visually plausible but different grasp on robots whose imported EE axes differ.
    _centre, _base_rel, self.q_correction = problem._gripper_rest_transforms(C, robot + "_")

  def _pose(self, body):
    bid = self.scene._body_id(body)
    return np.r_[self.scene.data.xpos[bid], self.scene.data.xquat[bid]]

  def _relative(self, parent, child):
    return self.problem._pose_compose(
        self.problem._pose_invert(self._pose(parent)), self._pose(child))

  def _enable_target_contacts(self, parts):
    model = self.scene.model
    parts = set(parts)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    pads = {int(g) for side in self.scene._gripper_contact_geoms[self.robot] for g in side}
    for gid in pads:
      model.geom_contype[gid] = 2
      model.geom_conaffinity[gid] = 1
    for gid in range(model.ngeom):
      body = int(model.geom_bodyid[gid])
      if self.scene._part_owner_by_body[body] in parts and self._all_contype[gid] != 0:
        model.geom_contype[gid] = 1
        model.geom_conaffinity[gid] = 2

  def _drive(self, ticks, target, start=0.0):
    scene, mujoco = self.scene, self.mujoco
    ticks = max(1, int(ticks))
    for k in range(ticks):
      u = (k + 1) / ticks
      smooth = u * u * (3.0 - 2.0 * u)
      scene._set_gripper_target(self.robot, start + smooth * (target - start))
      scene.command_arm_q(self.q0)
      scene.data.qfrc_applied[:] = 0.0
      scene.gravity_compensate()
      mujoco.mj_step(scene.model, scene.data)

  def _snapshot(self):
    """The acquired grasp state reused by the six independent load rollouts."""
    data = self.scene.data
    return {
        "time": float(data.time),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "act": data.act.copy(),
        "ctrl": data.ctrl.copy(),
        "eq_active": data.eq_active.copy(),
        "qacc_warmstart": data.qacc_warmstart.copy(),
    }

  def _restore(self, snapshot):
    data = self.scene.data
    data.time = snapshot["time"]
    for name in ("qpos", "qvel", "act", "ctrl", "eq_active", "qacc_warmstart"):
      getattr(data, name)[:] = snapshot[name]
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    self.mujoco.mj_forward(self.scene.model, data)

  def check(self, part, part_T_tool):
    """Close on one part-relative tool pose and retain it under ordinary gravity."""
    return self.check_component((part,), part, part_T_tool)

  def check_component(self, group, contacted_part, ref_T_tool):
    """Close on a grasp of an assembled rigid component and qualify six-axis retention."""
    group = tuple(group)
    ref = group[0]
    pose_key = tuple(np.round(np.asarray(ref_T_tool, dtype=float), 7))
    key = ("component", group, contacted_part, pose_key)
    hit = self._cache.get(key)
    if hit is not None:
      return hit

    scene, mujoco, problem = self.scene, self.mujoco, self.problem
    model, data = scene.model, scene.data
    mujoco.mj_resetData(model, data)
    self._enable_target_contacts(group)
    scene._physical_contact_target.clear()
    scene._physical_contact_target[self.robot] = contacted_part
    model.opt.gravity[:] = 0.0
    scene.set_arm_q(self.q0)
    scene.set_gripper_aperture(self.robot, 0.0)

    anchor_pose = self._pose(self.anchor)
    raw = np.asarray(ref_T_tool, dtype=float)
    exact = np.r_[raw[:3], problem._quat_mul(raw[3:], self.q_correction)]
    world_T_ref = problem._pose_compose(anchor_pose, problem._pose_invert(exact))
    ref_pos = self.assembly.position_in_base(ref)
    for part in group:
      offset = self.assembly.position_in_base(part) - ref_pos
      scene._set_free_part_pose(part, problem._pose_compose(
          world_T_ref, np.r_[offset, 1.0, 0.0, 0.0, 0.0]))
    mujoco.mj_forward(model, data)
    for part in group:
      rel = self.assembly.relative_pose(part)
      if rel is not None and rel[0] in group:
        scene.weld_mate(part)
    mujoco.mj_forward(model, data)
    authored_rel = self._relative(self.anchor, ref)

    physical_max = 0.045 if scene._gripper_model.get(self.robot) == "openarm" else 0.042
    self._drive(round(self.CLOSE_TIME / model.opt.timestep), physical_max)
    self._drive(round(self.SETTLE_TIME / model.opt.timestep), physical_max,
                start=physical_max)
    ok, forces = scene.contact_grasp_status(self.robot, contacted_part)
    acquired_rel = self._relative(self.anchor, ref)
    acquire_shift = float(np.linalg.norm(acquired_rel[:3] - authored_rel[:3]))
    if not ok:
      result = GraspPhysicsResult(False, "no-bilateral-contact", float(forces[0]),
                                  float(forces[1]), acquire_shift)
      self._cache[key] = result
      return result
    if acquire_shift > self.MAX_ACQUISITION_SHIFT:
      result = GraspPhysicsResult(False, "large-acquisition-correction", float(forces[0]),
                                  float(forces[1]), acquire_shift)
      self._cache[key] = result
      return result

    # Acquisition is evaluated without gravity so an unsupported free object cannot leave the
    # hand before both pads arrive.  Starting from that identical acquired state, qualify the
    # grasp against six tool-frame force directions under ordinary gravity.  The force is a
    # fixed global test load--we do not infer a per-grasp wrench envelope from the outcome.
    acquired = self._snapshot()
    self._acquired[key] = acquired
    sustain = max(1, round(self.sim.BREAK_SUSTAIN / model.opt.timestep))
    peak_forces = np.asarray(forces, float)
    worst_shift = worst_angle = 0.0
    worst_direction = ""
    reason, feasible = "ok", True
    ticks = max(1, round(self.RETAIN_TIME / model.opt.timestep))
    ramp_ticks = max(1, min(ticks, round(0.05 / model.opt.timestep)))
    anchor_R = problem._quat_to_R(anchor_pose[3:])
    masses = np.asarray([self.assembly.masses.get(
        p, self.assembly.DEFAULT_MASS) for p in group], dtype=float)
    masses /= max(float(np.sum(masses)), 1e-12)
    directions = self.LOAD_DIRECTIONS if self.force_n > 0.0 else (("gravity", np.zeros(3)),)
    for label, local_axis in directions:
      self._restore(acquired)
      model.opt.gravity[:] = (0.0, 0.0, -self.sim.GRAVITY)
      lost = 0
      for tick in range(ticks):
        scene._set_gripper_target(self.robot, physical_max)
        scene.command_arm_q(self.q0)
        scene.data.qfrc_applied[:] = 0.0
        scene.data.xfrc_applied[:] = 0.0
        scene.gravity_compensate()
        u = min(1.0, float(tick + 1) / ramp_ticks)
        smooth = u * u * (3.0 - 2.0 * u)
        total = smooth * self.force_n * (anchor_R @ local_axis)
        for part, share in zip(group, masses):
          scene.data.xfrc_applied[scene._body_id(part), :3] = share * total
        mujoco.mj_step(model, data)
        contact, now = scene.contact_grasp_status(self.robot, contacted_part)
        peak_forces = np.maximum(peak_forces, now)
        lost = 0 if contact else lost + 1
        if lost >= sustain:
          break
      final_rel = self._relative(self.anchor, ref)
      shift = float(np.linalg.norm(final_rel[:3] - acquired_rel[:3]))
      angle = _quat_angle(final_rel[3:], acquired_rel[3:])
      if shift > worst_shift or angle > worst_angle:
        worst_direction = label
      worst_shift = max(worst_shift, shift)
      worst_angle = max(worst_angle, angle)
      if lost >= sustain:
        reason, feasible, worst_direction = f"lost-contact-under-load:{label}", False, label
        break
      if shift > self.MAX_RETENTION_SHIFT:
        reason, feasible, worst_direction = f"slipped-under-load:{label}", False, label
        break
      if angle > self.MAX_RETENTION_ANGLE:
        reason, feasible, worst_direction = f"rotated-under-load:{label}", False, label
        break
    result = GraspPhysicsResult(
        feasible, reason, float(peak_forces[0]), float(peak_forces[1]), acquire_shift,
        worst_shift, float(np.rad2deg(worst_angle)), worst_direction)
    self._cache[key] = result
    return result

  def mate_clearance(self, child, part_T_tool, receiver_parts, mate_quat=None):
    """Exact closed-hand gap at one mate, or best over all authored alternatives.

    The hand is at its simulated bilateral acquisition aperture, not an open or translated
    proxy. Only tool-versus-receiver geometry is queried; the intended child/tool contact
    and child/receiver mating contact are excluded. A nonnegative result certifies that
    some discrete mate orientation has room for this grasp.
    """
    from assembly_planner.grasp_preflight import (_active_collision_geom, _descends_from,
                                                   _pair_enabled, _tool_root)
    receiver_parts = tuple(sorted(set(receiver_parts)))
    rel = self.assembly.relative_pose(child)
    if rel is None or rel[0] not in receiver_parts:
      raise ValueError(f'{child}: mate receiver must contain its authored parent')
    pose = np.asarray(part_T_tool, dtype=float)
    key = ('component', (child,), child, tuple(np.round(pose, 7)))
    chosen = (None if mate_quat is None else
              tuple(np.round(np.asarray(mate_quat, dtype=float), 7)))
    cache_key = (key, receiver_parts, chosen)
    if cache_key in self._mate_clearance:
      return self._mate_clearance[cache_key]
    if not self.check(child, pose).feasible:
      self._mate_clearance[cache_key] = -float('inf')
      return -float('inf')

    scene, model, data = self.scene, self.scene.model, self.scene.data
    self._restore(self._acquired[key])
    model.geom_contype[:] = self._all_contype
    model.geom_conaffinity[:] = self._all_conaffinity
    root = _tool_root(model, scene._body_id(self.anchor))
    hands = [g for g in range(model.ngeom) if _active_collision_geom(model, g)
             and _descends_from(model, model.geom_bodyid[g], root)]
    receivers = [g for g in range(model.ngeom) if _active_collision_geom(model, g)
                 and scene._part_owner_by_body[int(model.geom_bodyid[g])] in receiver_parts]
    if not hands or not receivers:
      raise ValueError(f'{child}: articulated mate check lacks collision geometry')
    child_pose = self._pose(child)
    parent, offset = rel
    best = -float('inf')
    witness = np.empty(6)
    quats = (self.assembly.mate_quaternions(child) if mate_quat is None
             else (np.asarray(mate_quat, dtype=float),))
    for quat in quats:
      parent_T_child = np.r_[np.asarray(offset, float), np.asarray(quat, float)]
      parent_pose = self.problem._pose_compose(
          child_pose, self.problem._pose_invert(parent_T_child))
      for part in receiver_parts:
        delta = self.assembly.position_in_base(part) - self.assembly.position_in_base(parent)
        world_pose = self.problem._pose_compose(
            parent_pose, np.r_[delta, 1.0, 0.0, 0.0, 0.0])
        scene._set_free_part_pose(part, world_pose)
      self.mujoco.mj_forward(model, data)
      nearest = 0.1
      for hand in hands:
        for receiver in receivers:
          if not _pair_enabled(model, hand, receiver):
            continue
          gap = float(self.mujoco.mj_geomDistance(
              model, data, hand, receiver, 0.1, witness))
          # Unsupported convex-mesh pairs can return zero with separated witness points.
          if abs(np.linalg.norm(witness[:3] - witness[3:]) - abs(gap)) > 1e-4:
            continue
          nearest = min(nearest, gap)
      best = max(best, nearest)
    self._mate_clearance[cache_key] = best
    return best

  def check_component_fixed_acquisition(self, group, contacted_part, ref_T_tool):
    """Whether this hand closes bilaterally while the component is held fixed.

    Ordinary picks may centre a free payload between the pads during closure.  A handover
    receiver cannot rely on that motion because the giver is still supporting the component;
    it must establish contact at the authored rendezvous before the transaction releases the
    giver.  Clamp the component root to that rendezvous during the close/settle rollout and
    require loaded contact on both pads.  Retention after release is still certified by the
    ordinary :meth:`check_component` six-load rollout.
    """
    group = tuple(group)
    ref = group[0]
    pose_key = tuple(np.round(np.asarray(ref_T_tool, dtype=float), 7))

    key = ("fixed-acquisition", group, contacted_part, pose_key)
    hit = self._cache.get(key)
    if hit is not None:
      return hit

    scene, mujoco, problem = self.scene, self.mujoco, self.problem
    model, data = scene.model, scene.data
    mujoco.mj_resetData(model, data)
    self._enable_target_contacts(group)
    scene._physical_contact_target.clear()
    scene._physical_contact_target[self.robot] = contacted_part
    model.opt.gravity[:] = 0.0
    scene.set_arm_q(self.q0)
    scene.set_gripper_aperture(self.robot, 0.0)

    anchor_pose = self._pose(self.anchor)
    raw = np.asarray(ref_T_tool, dtype=float)
    exact = np.r_[raw[:3], problem._quat_mul(raw[3:], self.q_correction)]
    world_T_ref = problem._pose_compose(anchor_pose, problem._pose_invert(exact))
    ref_pos = self.assembly.position_in_base(ref)
    for part in group:
      offset = self.assembly.position_in_base(part) - ref_pos
      scene._set_free_part_pose(part, problem._pose_compose(
          world_T_ref, np.r_[offset, 1.0, 0.0, 0.0, 0.0]))
    mujoco.mj_forward(model, data)
    for part in group:
      rel = self.assembly.relative_pose(part)
      if rel is not None and rel[0] in group:
        scene.weld_mate(part)
    mujoco.mj_forward(model, data)
    authored_rel = self._relative(self.anchor, ref)

    # The assembled component has one free root; its welded descendants follow it.  Reassert
    # that root immediately before each physics step, exactly modelling a stiff live giver
    # while preserving real receiver-pad contact dynamics during the step itself.
    root_body = scene._body_id(ref)
    j0, nj = int(model.body_jntadr[root_body]), int(model.body_jntnum[root_body])
    free_joint = next((j for j in range(j0, j0 + nj)
                       if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)), -1)
    if free_joint < 0:
      result = GraspPhysicsResult(False, "component-root-not-free")
      self._cache[key] = result
      return result
    qadr, vadr = int(model.jnt_qposadr[free_joint]), int(model.jnt_dofadr[free_joint])
    held_q = data.qpos[qadr:qadr + 7].copy()

    physical_max = 0.045 if scene._gripper_model.get(self.robot) == "openarm" else 0.042
    phases = ((round(self.CLOSE_TIME / model.opt.timestep), 0.0, physical_max),
              (round(self.SETTLE_TIME / model.opt.timestep), physical_max, physical_max))
    for ticks, start, target in phases:
      ticks = max(1, int(ticks))
      for tick in range(ticks):
        data.qpos[qadr:qadr + 7] = held_q
        data.qvel[vadr:vadr + 6] = 0.0
        u = float(tick + 1) / ticks
        smooth = u * u * (3.0 - 2.0 * u)
        scene._set_gripper_target(self.robot, start + smooth * (target - start))
        scene.command_arm_q(self.q0)
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        scene.gravity_compensate()
        mujoco.mj_step(model, data)

    ok, forces = scene.contact_grasp_status(self.robot, contacted_part)
    acquired_rel = self._relative(self.anchor, ref)
    shift = float(np.linalg.norm(acquired_rel[:3] - authored_rel[:3]))
    result = GraspPhysicsResult(
        bool(ok), "ok" if ok else "no-bilateral-contact-with-fixed-giver",
        float(forces[0]), float(forces[1]), shift)
    self._cache[key] = result
    return result


class GraspPhysicsContext:
  """Lazy checker registry keyed by representative two-finger robot."""

  def __init__(self, C, assembly, state, robots, kind_robot, force_n=10.0,
               gripper_paths=None):
    self.C, self.assembly, self.state = C, assembly, state
    self.robots = tuple(robots)
    self.kind_robot = dict(kind_robot)
    self.force_n = float(force_n)
    self.gripper_paths = dict(gripper_paths or {})
    self._checkers = {}
    self._handover_compat = {}
    self._handover_models = None
    self._handover_geometry = {}
    # A handover certificate is an OFFLINE property of one finite grasp database and hand
    # model roster.  Keep the completed table as a first-class artifact: repeated local fills,
    # witness collection, and motion export must never rerun (or even re-enumerate) physics.
    self._handover_signature = None

  @staticmethod
  def _model_key(path):
    """Portable identity for the hand and its transitive RAI model includes."""
    import assembly_planner.env as env
    path = os.path.realpath(path)
    h = hashlib.sha256()
    seen = set()

    def add_file(filename):
      filename = os.path.realpath(filename)
      if filename in seen:
        return
      seen.add(filename)
      with open(filename, "rb") as f:
        body = f.read()
      h.update(body)
      for child in re.findall(rb"(?m)^\s*Include:\s*<([^>]+)>", body):
        add_file(os.path.join(os.path.dirname(filename), os.fsdecode(child)))

    add_file(path)
    # These process-wide calibration switches change collision geometry after the .g
    # files are parsed. They must therefore be part of a persisted pair certificate too.
    h.update((f"tight={int(env._TIGHT_FINGERS)}\0"
              f"openarm-finger-scale={env._OPENARM_REAL_FINGER_RADIUS_SCALE:.17g}\0").encode())
    return f"{os.path.basename(path)}:{h.hexdigest()}"

  def _handover_table_signature(self, table, models):
    """Identity of every finite grasp row covered by an eager handover certificate."""
    import assembly_planner.gripper_closure as gripper_closure
    h = hashlib.sha256()
    h.update((f"{HANDOVER_CERTIFICATE_CONTRACT}\0"
              f"closure={gripper_closure.VERSION}\0force={self.force_n:.17g}\0").encode())
    for robot, model in sorted(models.items()):
      h.update(f"robot={robot}\0model={model}\0".encode())
    for key in sorted(table.keys(), key=repr):
      key = tuple(key)
      h.update(repr(key).encode() + b"\0")
      for part, gid, pose in sorted(table[key], key=lambda row: (str(row[0]), int(row[1]))):
        h.update(str(part).encode() + b"\0" + str(int(gid)).encode() + b"\0")
        h.update(np.asarray(pose, dtype=np.float64).ravel().tobytes())
        kinds = getattr(table, "kinds_of", None)
        if kinds is not None:
          h.update(repr(tuple(sorted(kinds.get((key, part, gid), ())))).encode())
    return h.hexdigest()

  def export_handovers(self):
    """Compact, portable certificate payload for a saved task plan.

    Only accepted pairs are stored.  Absence means rejection because ``signature`` certifies
    that the complete finite pair universe was enumerated.
    """
    if self._handover_signature is None or self._handover_models is None:
      return None
    accepted = []
    for (group, ma, ga, mb, gb), feasible in sorted(
        self._handover_compat.items(), key=lambda item: repr(item[0])):
      if feasible:
        accepted.append([list(group), ma, ga[0], int(ga[1]),
                         mb, gb[0], int(gb[1])])
    return {
        "contract": HANDOVER_CERTIFICATE_CONTRACT,
        "force_n": self.force_n,
        "signature": self._handover_signature,
        "models": dict(self._handover_models),
        "accepted": accepted,
    }

  def import_handovers(self, payload):
    """Install an a-priori certificate; its signature is checked against the DB on use."""
    if not payload:
      return False
    if payload.get("contract") != HANDOVER_CERTIFICATE_CONTRACT:
      raise ValueError("saved handover-physics certificate uses an obsolete collision "
                       "contract; rerun search to recompute handover pairs")
    if abs(float(payload.get("force_n", -1.0)) - self.force_n) > 1e-12:
      raise ValueError("saved handover-physics force differs from this run")
    models = {str(r): str(m) for r, m in (payload.get("models") or {}).items()}
    # The robot names and model contents are part of the certificate.  Refuse a stale model
    # before search rather than quietly recomputing physics inside a local fill.
    current = {
        r: self._model_key(path) for r, path in self.gripper_paths.items()
        if path is not None and "vacuum" not in os.path.basename(path)
    }
    if models != current:
      raise ValueError("saved handover-physics hand models differ from this scene")
    compat = {}
    for row in payload.get("accepted") or ():
      if len(row) != 7:
        raise ValueError("malformed saved handover-physics pair")
      group, ma, pa, ga, mb, pb, gb = row
      compat[(tuple(group), str(ma), (str(pa), int(ga)),
              str(mb), (str(pb), int(gb)))] = True
    self._handover_models = models
    self._handover_compat = compat
    self._handover_signature = str(payload.get("signature") or "")
    if not self._handover_signature:
      raise ValueError("saved handover-physics certificate has no database signature")
    return True

  def check(self, part, pose, kind="two_finger"):
    if kind != "two_finger":
      return GraspPhysicsResult(True, "not-applicable")
    robot = self.kind_robot.get(kind)
    if robot is None:
      return GraspPhysicsResult(False, "no-two-finger-robot")
    checker = self._checkers.get(robot)
    if checker is None:
      checker = self._checkers[robot] = GraspPhysicsChecker(
          self.C, self.assembly, self.state, robot, force_n=self.force_n)
    return checker.check(part, pose)

  def supports_component(self, group, robot, contacted_part, grasp_id):
    """Qualify a lone physical hand carrying this complete rigid component."""
    import assembly_planner.problem as problem
    group = tuple(sorted(group))
    ref = group[0]
    pose = problem.grasp_pose(self.assembly, contacted_part, grasp_id)
    delta = (self.assembly.position_in_base(contacted_part)
             - self.assembly.position_in_base(ref))
    ref_T_tool = problem._pose_compose(
        np.r_[delta, 1.0, 0.0, 0.0, 0.0], pose)
    return self._checker_for_robot(robot).check_component(
        group, contacted_part, ref_T_tool)

  def mate_clearance(self, child, pose, receiver_parts, kind='two_finger',
                     mate_quat=None):
    if kind != 'two_finger':
      return 0.1
    robot = self.kind_robot.get(kind)
    if robot is None:
      raise ValueError('no articulated two-finger model for mate clearance')
    return self._checker_for_robot(robot).mate_clearance(
        child, pose, receiver_parts, mate_quat)

  def _checker_for_robot(self, robot):
    checker = self._checkers.get(robot)
    if checker is None:
      checker = self._checkers[robot] = GraspPhysicsChecker(
          self.C, self.assembly, self.state, robot, force_n=self.force_n)
    return checker

  def precompute_handovers(self, table):
    """Eagerly certify every two-finger co-grasp represented by ``table``.

    A deterministic RAI simultaneous-hand closing-sweep certificate runs first. Only
    survivors enter MuJoCo for two separate unary retention checks and a receiver-on-fixed-
    component acquisition check. MuJoCo does not simulate both hands together here; the RAI
    pair sweep owns the inter-hand collision guarantee. Search sees only the resulting table.
    """
    import assembly_planner.problem as problem

    two_finger = [r for r in self.robots
                  if self.gripper_paths.get(r) is not None
                  and "vacuum" not in os.path.basename(self.gripper_paths[r])]
    models = {r: self._model_key(self.gripper_paths[r]) for r in two_finger}
    signature = self._handover_table_signature(table, models)

    # A loaded shared grasp DB may already carry the complete table. Adopt it into this
    # process-local context so every later independently rebuilt GraspTable is O(1).
    table_signature = getattr(table, "handover_signature", None)
    if (getattr(table, "handover_physics", None) is not None
        and table_signature == signature
        and getattr(table, "handover_models", None) == models):
      self._handover_compat = table.handover_physics
      self._handover_models = models
      self._handover_signature = signature
      print("[grasp-physics/handover] using a-priori certificate from grasp database")
      return table

    # This is the common in-process path: search, witness postprocessing, and MP export all
    # rebuild their views of the same finite DB.  Do not rescan thousands of certified pairs.
    if self._handover_signature == signature and self._handover_models == models:
      table.handover_physics = self._handover_compat
      table.handover_models = models
      table.handover_signature = signature
      print("[grasp-physics/handover] using a-priori in-memory certificate")
      return table

    # A genuinely different finite DB invalidates the complete-universe interpretation of
    # absent pairs.  Start one new a-priori certification, before search begins.
    self._handover_compat = {}
    self._handover_models = models
    self._handover_signature = None
    table.handover_physics = self._handover_compat
    table.handover_models = self._handover_models
    if len(two_finger) < 2:
      self._handover_signature = signature
      table.handover_signature = signature
      print("[grasp-physics/handover] skipped: fewer than two articulated two-finger hands")
      return table

    # One representative ordered robot pair for each ordered physical model pair.
    reps = {}
    for a in two_finger:
      for b in two_finger:
        if a == b:
          continue
        ma, mb = table.handover_models[a], table.handover_models[b]
        reps.setdefault((ma, mb), (a, b))
    stats = {"pairs": 0, "cached": 0, "unary_reject": 0, "geometric_reject": 0,
             "physics_reject": 0, "ok": 0, "reasons": {}}
    start = time.perf_counter()
    for key in table.keys():
      rows = list(table[key])                               # explicit eager materialisation
      unary_ok = {}
      for part, gid, _pose in rows:
        local_pose = problem.grasp_pose(self.assembly, part, int(gid))
        unary_ok[(part, int(gid))] = bool(
            self.check(part, local_pose, "two_finger").feasible)
      for (model_a, model_b), (robot_a, robot_b) in reps.items():
        path_a, path_b = self.gripper_paths[robot_a], self.gripper_paths[robot_b]
        for ia, row_a in enumerate(rows):
          part_a, gid_a, _pose_a = row_a
          if (getattr(table, "kinds_of", None) is not None
              and "two_finger" not in table.kinds_of.get((key, part_a, gid_a), ())):
            continue
          for ib in range(len(rows)):
            row_b = rows[ib]
            part_b, gid_b, _pose_b = row_b
            if (part_a, gid_a) == (part_b, gid_b):
              continue
            if (getattr(table, "kinds_of", None) is not None
                and "two_finger" not in table.kinds_of.get((key, part_b, gid_b), ())):
              continue
            ga, gb, ma, mb = ((part_a, int(gid_a)), (part_b, int(gid_b)),
                              model_a, model_b)
            ra, rb = row_a, row_b
            out_key = (problem._subasm_key(key), ma, ga, mb, gb)
            if out_key in table.handover_physics:
              stats["cached"] += 1
              continue
            stats["pairs"] += 1
            if not (unary_ok.get(ga, False) and unary_ok.get(gb, False)):
              table.handover_physics[out_key] = False
              stats["unary_reject"] += 1
              continue
            geometry_key = (problem._subasm_key(key), model_a, model_b)
            geometry = self._handover_geometry.get(geometry_key)
            if geometry is None:
              geometry = self._handover_geometry[geometry_key] = \
                  HandoverGeometryChecker(self.assembly, key, path_a, path_b)
            if not geometry.check(ra, rb):
              table.handover_physics[out_key] = False
              stats["geometric_reject"] += 1
              continue
            # Only geometric survivors pay for dynamics. Each component/grasp/model result
            # is cached by its one-hand checker, so the quadratic pair table does not imply
            # quadratic MuJoCo rollouts.
            result_a = self._checker_for_robot(robot_a).check_component(
                key, ra[0], ra[2])
            checker_b = self._checker_for_robot(robot_b)
            result_b = checker_b.check_component(
                key, rb[0], rb[2])
            fixed_b = checker_b.check_component_fixed_acquisition(
                key, rb[0], rb[2])
            result = (result_a if not result_a.feasible else
                      result_b if not result_b.feasible else fixed_b)
            feasible = bool(result_a.feasible and result_b.feasible and fixed_b.feasible)
            table.handover_physics[out_key] = feasible
            stats["ok" if feasible else "physics_reject"] += 1
            if not feasible:
              stats["reasons"][result.reason] = stats["reasons"].get(result.reason, 0) + 1
              if stats["physics_reject"] == 1:
                print("[grasp-physics/handover] first physics rejection: "
                      f"component={key}, giver={(ra[0], ra[1])}, "
                      f"receiver={(rb[0], rb[1])}, reason={result.reason}, "
                      f"finger_forces=({result.left_force_n:.3f},"
                      f"{result.right_force_n:.3f})N", flush=True)
      elapsed = time.perf_counter() - start
      print(f"[grasp-physics/handover] component {key}: {stats['pairs']} pairs, "
            f"{stats['cached']} cached, "
            f"{stats['unary_reject']} unary rejects, "
            f"{stats['geometric_reject']} geometric rejects, "
            f"{stats['physics_reject']} physics rejects, {stats['ok']} accepted "
            f"({elapsed:.1f}s)", flush=True)
    reasons = "" if not stats["reasons"] else f"; physics reasons={stats['reasons']}"
    print(f"[grasp-physics/handover] eager table complete: {stats['pairs']} new / "
          f"{stats['cached']} cached pairs in {time.perf_counter() - start:.1f}s; search "
          f"performs cached lookups only{reasons}.")
    self._handover_signature = signature
    table.handover_signature = signature
    return table


class HandoverGeometryChecker:
  """Preloaded articulated-finger RAI filter for one component and hand-model pair."""

  def __init__(self, assembly, group, path_a, path_b):
    import assembly_planner.problem as problem
    from assembly_planner.env import maybe_tighten_fingers

    self.problem = problem
    self.prefix_a, self.prefix_b = "handoverA_", "handoverB_"
    self.template = problem.make_subassembly_scene(assembly, group)
    self.template.addFile(path_a, namePrefix=self.prefix_a)
    self.template.addFile(path_b, namePrefix=self.prefix_b)
    maybe_tighten_fingers(self.template, self.prefix_a)
    maybe_tighten_fingers(self.template, self.prefix_b)
    self.tool_a = problem._gripper_rest_transforms(self.template, self.prefix_a)[1:]
    self.tool_b = problem._gripper_rest_transforms(self.template, self.prefix_b)[1:]

  def check(self, giver, receiver):
    import assembly_planner.gripper_closure as gripper_closure
    import robotic as ry

    C = ry.Config()
    C.addConfigurationCopy(self.template)
    for prefix, row, (base_rel, correction) in (
        (self.prefix_a, giver, self.tool_a),
        (self.prefix_b, receiver, self.tool_b)):
      root_pose = self.problem._ee_root_pose(row[2], base_rel, correction)
      root = C.getFrame(prefix + "ur_base")
      root.setPosition(root_pose[:3])
      root.setQuaternion(root_pose[3:])
    C.setJointState(C.getJointState())                     # refresh FCL broad phase
    result = gripper_closure.certify_handover(
        C, self.prefix_a, giver[0], self.prefix_b, receiver[0])
    return bool(result.feasible)
