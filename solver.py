"""Feasibility layers (cheap floating-EE CollisionChecker, real-robot RobotIK) and the
(weighted) A* search with the abstraction heuristic. Depends on the problem definition."""

import os

import robotic as ry
import numpy as np

from env import *
from problem import *
import problem as _problem
# pull in problem's private (single-underscore) names too, which `import *` skips
globals().update({k: v for k, v in vars(_problem).items() if not k.startswith("__")})


# weight of the soft orientation bias on a held subassembly in the assemble IK (see
# RobotIK._ik_assemble). LOW relative to the mate eq weight (1e1) so it only nudges the
# free-orientation DOF toward a canonical pose without fighting the hard mate constraint.
_ASSEMBLE_ORIENT_BIAS_W = 1e0

# slight position bias on a held subassembly in the assemble IK: nudge its centre body
# toward the middle of the workspace (a bit above the table). LOW weight vs the mate eq
# (1e1), so it only breaks the free-position DOF of a HELD assembly; inert on the table.
_ASSEMBLE_POS_BIAS_TARGET = (0.0, 0.0, 0.5)
_ASSEMBLE_POS_BIAS_W = 1e0

# the canonical orientations the assemble IK sweeps (structured restarts): the 6 ways a
# principal axis of the assembly's centre body points along a world axis. Each entry is
# (z_dir, x_dir) -- where the body's local +z / +x should point -- and the pair is
# orthogonal so it pins a full orientation. Swept in order; first feasible wins.
_ASSEMBLE_BIAS_ORIENTATIONS = [
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),    # z up (upright)
    ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),   # z down
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),    # z -> +x
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),   # z -> -x
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),    # z -> +y
    ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),   # z -> -y
]


def _park_pos(i):
  # a distinct far-away spot per gripper slot, so parked (unused) grippers neither touch
  # the scene nor each other.
  return [1000.0 + 10.0 * i, 1000.0, 1000.0]


class CollisionChecker:
  """Cheap geometric feasibility checker -- the CONSERVATIVE LOWER BOUND. It is its own
  purpose-built floating-EE scene (world + cube parts + one free-floating gripper per
  robot), NOT the real-robot env with arms removed. Built once; each query copies the
  template and just REPOSITIONS parts/grippers (no re-parsing the gripper .g). A fresh
  copy per query is required because rai caches the FCL broadphase after the first query.

  Catches part/table, part/part, gripper/gripper, gripper/object and gripper-into-the-
  assembly collisions. It is a sound pre-filter for the real-robot check (a gripper
  collision implies an arm collision), but it does NOT see the arms or reachability --
  that is the RobotIK (L3) layer."""

  def __init__(self, assembly, robots, gripper_path=None):
    self.gripper_path = gripper_path or os.path.join(os.path.dirname(__file__), DEFAULT_FLOATING_EE)
    self.cube_size = assembly.cube_size

    template = make_world()                                # floor + table, same as the real env
    cs, cr = assembly.cube_size, assembly.cube_radius
    for obj in assembly.objects:                           # the parts, as cubes (posed per query)
      template.addFrame(obj).setShape(ry.ST.ssBox, [cs, cs, cs, cr]).setContact(1)

    # one free-floating gripper slot per robot (at most one active grasp per robot),
    # pre-loaded and parked far away so unused slots touch nothing.
    self.slots = [f"eeSLOT{i}_" for i in range(len(robots))]
    for prefix in self.slots:
      template.addFile(self.gripper_path, namePrefix=prefix)
    _, self.base_rel_tool, self.q_correction = _gripper_rest_transforms(template, self.slots[0]) \
        if self.slots else (None, None, None)
    for i, prefix in enumerate(self.slots):
      template.getFrame(prefix + "ur_base").setPosition(_park_pos(i))
    self.template = template

  def _scene_for(self, state, assembly, extra_grasps=()):
    # build the floating-EE collision config for a state: parts posed per the state's scene
    # graph, a floating EE at every active grasp of the state, plus `extra_grasps`
    # [(gpart, grasp_id), ...] as additional EEs. Returns the ry.Config.
    #
    # SOUNDNESS: a held-in-air group's world pose is FREE (the IK optimizes it), so only
    # its pose-INVARIANT geometry may reject: the group plus its attached grippers as one
    # rigid body (does the new part fit? does a brace block a mate face?). Its incidental
    # relation to the table / other groups at some synthetic pose must NOT reject (the IK
    # can move it away). So held groups are parked far away -- internal collisions are
    # preserved (rigid), external ones vanish (conservative). On-table groups keep their
    # REAL poses (their pose is fixed), so table / neighbour collisions there do reject.
    C_col = ry.Config()
    C_col.addConfigurationCopy(self.template)

    part_poses = _state_part_poses(state, assembly, lift=0.0)
    for k, group in enumerate(state.subassemblies()):
      held = any(c[0] == "grasp" for p in group for c in state.supports.get(p, ()))
      on_table = any(("table",) in state.supports.get(p, ()) for p in group)
      if held and not on_table:                      # pose is free -> park it (own offset
        off = np.array([0.0, -1000.0 - 50.0 * k, 5.0])   # per group; clear of the EE park)
        for p in group:
          part_poses[p] = np.concatenate([part_poses[p][:3] + off, part_poses[p][3:]])
    for p, pose in part_poses.items():
      f = C_col.getFrame(p)
      f.setPosition(pose[:3])
      f.setQuaternion(pose[3:])

    grasps = [(c[2], c[3]) for p in state.bodies()
              for c in state.supports.get(p, ()) if c[0] == "grasp"]
    grasps += list(extra_grasps)
    for prefix, (gpart, grasp_id) in zip(self.slots, grasps):   # move a slot onto each grasp
      grasp_world = _pose_compose(np.asarray(part_poses[gpart], dtype=float),
                                  cube_grasp_pose(grasp_id, self.cube_size))
      base_world = _ee_root_pose(grasp_world, self.base_rel_tool, self.q_correction)
      root = C_col.getFrame(prefix + "ur_base")
      root.setPosition(base_world[:3])
      root.setQuaternion(base_world[3:])
    # any remaining slots stay parked (from the template) -> out of the way.
    return C_col

  def _check(self, state, assembly, extra_grasps=()):
    # tolerate grazing contacts (see _CHEAP_PENETRATION_TOL) instead of binary-exact
    # getCollisionFree -- the cheap layer must not be stricter than the (soft) IK.
    C_col = self._scene_for(state, assembly, extra_grasps)
    return all(d >= -_CHEAP_PENETRATION_TOL for _a, _b, d in C_col.getCollisions())

  def penetrating_pairs(self, state, assembly, extra_grasps=()):
    # the collision pairs that make `_check` reject: (frame_a, frame_b, depth) with
    # depth < -tol. Frame names are part names (obj_*) or gripper-slot frames (eeSLOT*_...).
    # Used for BLOCKER ATTRIBUTION in the decluttering search -- which loose part is in the
    # way -- so it can be relocated rather than the whole step abandoned.
    C_col = self._scene_for(state, assembly, extra_grasps)
    return [(a, b, d) for a, b, d in C_col.getCollisions()
            if d < -_CHEAP_PENETRATION_TOL]

  def collision_free(self, state, assembly):
    return self._check(state, assembly)

  def pick_collision_free(self, s0, gpart, grasp_id, assembly):
    # cheap gate for PICK edges: place a floating EE at the candidate grasp with the part
    # at its CURRENT (pre-pick, on-table) pose, in the actual environment. This is what
    # the precomputed grasp table does NOT see -- it checks a grasp only against its own
    # subassembly (no table, no neighbours), so e.g. a two-finger side grasp whose lower
    # finger dips below the cube (into the table) passes precompute yet can never be
    # picked. Rejecting it here costs ~2ms instead of a full (restarted) IK solve.
    return self._check(s0, assembly, extra_grasps=[(gpart, grasp_id)])


class RobotIK:
  """L3 real-robot feasibility -- the expensive, exact layer, run only AFTER the cheap
  CollisionChecker passes. For a state it solves a JOINT IK / optimization:

    - robots HOLDING a subassembly: their arm config AND the held subassembly's world
      pose are solved together, so the carry / assemble pose is optimized (not fixed);
    - robots NOT holding anything: pinned at their home config;
    - the whole thing must be collision-free (real arms + grippers + parts + table).

  Returns (feasible, witness) where witness is a joint state (or None). Stateless per node
  (free robots deterministically at home); `seed` may warm-start from the parent.

  Implemented per skill: pick (6-DOF EE-at-grasp) and assemble (cube-mate constraint with
  structured orientation restarts + centre bias) are done; place is still a stub.
  """

  def __init__(self, C, robots, debug_view=False):
    self.robots = list(robots)
    self.debug_view = debug_view                      # pop up each IK solution (visual check)
    self.C = ry.Config()
    self.C.addConfigurationCopy(C)                    # real arms + grippers + parts + world
    self.home = self.C.getJointState()                # home config = the env's default pose

    # tool-center frame + approach-axis correction per robot, from the SAME helper the
    # floating EE uses -- so gripper-frame handling lives in one place. The correction is
    # identity for the two-finger EE (approaches along +z) and a rotation for the vacuum
    # (approaches along +x); it aligns the grasp's +z approach with the EE's real axis.
    self.gripper_frame = {}
    self.q_correction = {}
    self.symmetric = {}                                # roll-free about the approach axis?
    for r in self.robots:
      tool_center, _base_rel, q_corr = _gripper_rest_transforms(self.C, r + "_")
      self.gripper_frame[r] = tool_center
      self.q_correction[r] = q_corr
      # the vacuum is rotationally symmetric about its suction axis -> roll is a free DOF,
      # so we constrain only position + approach axis (5-DOF). the two-finger's fingers
      # break that symmetry -> full pose (6-DOF).
      self.symmetric[r] = "vacuum" in tool_center

  def holders(self, state):
    # {robot: (grasped_part, grasp_id)} -- who holds what, at which grasp, in this state.
    return {c[1]: (c[2], c[3]) for p in state.bodies()
            for c in state.supports.get(p, ()) if c[0] == "grasp"}

  def ee_target(self, robot, grasp_id, assembly):
    # the gripper's target pose relative to its grasped part: the generated grasp,
    # approach-corrected for THIS robot's EE.
    grasp = cube_grasp_pose(grasp_id, assembly.cube_size)
    return np.concatenate([grasp[:3], _quat_mul(grasp[3:], self.q_correction[robot])])

  def _add_grasp_maintenance(self, komo, state, assembly):
    # GRASP MAINTENANCE: every grasp that scene_for does NOT enforce kinematically (i.e.
    # every BRACE: a holder of a table-parented group, or a secondary holder of a held
    # group) becomes an explicit objective -- the gripper pinned at its grasp, the arm a
    # co-optimized DOF. Without this, brace arms sit at HOME in the IK scene (obstacles in
    # the wrong place) and a moved group slides out of its braces' grippers.
    grp_of = {p: g for g in state.subassemblies() for p in g}
    for r, (pt, gid) in self.holders(state).items():
      g = grp_of[pt]
      root = _group_root(state, g)
      if pt == root and state.parent.get(root) == ("gripper", r):
        continue                            # kinematic holder: enforced by the attach
      tgt = self.ee_target(r, gid, assembly)
      komo.addObjective([1, 1], ry.FS.positionRel, [self.gripper_frame[r], pt],
                        ry.OT.eq, [1e1], tgt[:3])
      komo.addObjective([1, 1], ry.FS.quaternionRel, [self.gripper_frame[r], pt],
                        ry.OT.eq, [1e1], tgt[3:])

  # ------------------------------------------------------------------------------------
  # scene prep -- build a real-robot config whose kinematics reflect a state:
  #   * on-table subassemblies posed at their table pose (fixed obstacles),
  #   * each held subassembly rigidly built and attached to its (primary) holder's gripper
  #     at the grasp transform, so it moves with that arm,
  #   * all robots at home (the per-skill IK then frees only the ACTIVE arms).
  # ------------------------------------------------------------------------------------
  def scene_for(self, state, assembly):
    C = ry.Config()
    C.addConfigurationCopy(self.C)                        # home; loose parts at initial poses
    holders = self.holders(state)
    for group in state.subassemblies():
      members = sorted(group, key=_part_index)
      grp_holders = [(r, pt, gid) for r, (pt, gid) in holders.items() if pt in group]
      # the KINEMATIC parent decides: a gripper-parented group is attached to that holder;
      # a TABLE-parented group is posed fixed on the table EVEN WHEN BRACED (the braces are
      # grasp-maintenance objectives in the per-skill IK, not an attach -- attaching a
      # braced-on-table group to a brace arm would let the IK carry the resting assembly
      # through the air).
      root = _group_root(state, group)
      prim = next((h for h in grp_holders
                   if state.parent.get(h[1]) == ("gripper", h[0]) and h[1] == root), None)
      if prim is not None:
        r, gpart, gid = prim
        self._attach_subassembly(C, members, gpart, r, gid, assembly)
      else:
        # anchor the on-table group at the pose the STATE records for its reference part.
        # For every existing search that equals the initial env pose (poses are never
        # rewritten), so this is behaviour-preserving; the decluttering search rewrites it
        # when it stows a part aside, and the IK must then see that part at its stow cell.
        anchor = state.poses.get(members[0])
        self._pose_on_table(C, members, assembly, _placement_face(state, group),
                            anchor=None if anchor is None else np.asarray(anchor))
    return C

  def _pose_on_table(self, C, members, assembly, face=_IDENTITY_FACE, anchor=None):
    base = members[0]
    if anchor is None:
      anchor = np.asarray(self.C.getFrame(base).getPose())  # its initial table pose
    # table rest rule (same as _state_part_poses / _ik_place): rotate per the placement
    # face and lift so the group's lowest part rests on the table.
    base_pose = _group_rest_pose(assembly, members, base, anchor, face)
    base_off = assembly.position_in_base(base)
    for p in members:
      rel = assembly.position_in_base(p) - base_off
      member = _pose_compose(base_pose, np.concatenate([rel, [1.0, 0.0, 0.0, 0.0]]))
      f = C.getFrame(p)
      f.setPosition(member[:3])
      f.setQuaternion(member[3:])

  def _attach_subassembly(self, C, members, gpart, robot, grasp_id, assembly):
    # place the grasped part at gripper o inv(ee_target), attach it to the gripper, then mate
    # the rest of the subassembly rigidly onto it. ee_target is the APPROACH-CORRECTED grasp
    # (grasp o q_correction) -- the SAME transform the pick IK constrains the gripper to, so
    # the held pose here is consistent with how the part was grasped. (q_correction is
    # identity for the two-finger but a 90 deg rotation for the vacuum, so using the raw
    # grasp here would seat the cube rotated 90 deg off the gripper.)
    gripper = self.gripper_frame[robot]
    grasp = cube_grasp_pose(grasp_id, assembly.cube_size)
    ee_target = np.concatenate([grasp[:3], _quat_mul(grasp[3:], self.q_correction[robot])])
    gp_world = _pose_compose(np.asarray(C.getFrame(gripper).getPose()), _pose_invert(ee_target))
    C.getFrame(gpart).setPosition(gp_world[:3])
    C.getFrame(gpart).setQuaternion(gp_world[3:])
    C.attach(gripper, gpart)
    gp_off = assembly.position_in_base(gpart)
    for p in members:
      if p == gpart:
        continue
      # offset of p from the grasped part, in the grasped part's OWN (axis-aligned) frame;
      # compose it onto gp_world so it rotates WITH the grasp (adding it in world coords
      # would misplace the member whenever the grasp rotates the part -- e.g. the vacuum).
      rel = assembly.position_in_base(p) - gp_off
      member = _pose_compose(gp_world, np.concatenate([rel, [1.0, 0.0, 0.0, 0.0]]))
      f = C.getFrame(p)
      f.setPosition(member[:3])
      f.setQuaternion(member[3:])
      C.attach(gpart, p)

  # ------------------------------------------------------------------------------------
  # per-skill IK. bind() dispatches on the action; each skill sets up its scene and its
  # own objective, then solves. Returns (feasible, witness or None).
  # ------------------------------------------------------------------------------------
  def bind(self, action, s0, s1, assembly, seed=None):
    kind = action[0]
    if kind == "pick":
      return self._ik_pick(action, s0, s1, assembly, seed)
    if kind == "place":
      return self._ik_place(action, s0, s1, assembly, seed)
    if kind == "assemble":
      return self._ik_assemble(action, s0, s1, assembly, seed)
    if kind in ("add_grasp", "handover"):
      # both are a 2nd arm reaching a grasp while the current holder presents the group; the
      # handover's release adds no reach. Action layout (kind, robot, gpart, grasp) matches.
      return self._ik_add_grasp(action, s0, s1, assembly, seed)
    return True, None                                     # release: no new reach

  def _holders_of_group(self, state, group):
    # robots grasping any part of `group`
    return [r for r, (pt, _gid) in self.holders(state).items() if pt in group]

  def _solve_ik(self, komo, restarts=12):
    # solve a skill KOMO, retrying from fresh random inits until one converges. KOMO seeds
    # its guess with a little noise, so a borderline-reachable target flips feasible/
    # infeasible run-to-run; restarts make bind reliable -- a genuinely feasible action
    # returns a witness (instead of sometimes None, which would corrupt search + replay).
    # Returns (feasible, witness joint config or None).
    r = ry.NLP_Solver(komo.nlp(), verbose=0).solve().dict()
    if r["feasible"]:
      return True, komo.getPath()[-1]
    for _ in range(restarts):
      komo.initRandom()
      r = ry.NLP_Solver(komo.nlp(), verbose=0).solve().dict()
      if r["feasible"]:
        return True, komo.getPath()[-1]
    return False, None

  def _ik_pick(self, action, s0, s1, assembly, seed):
    # simple IK: ONE robot (the picker) reaches a fixed grasp pose on an on-table part.
    # Every other robot -- free or holding something else -- stays at home (an obstacle).
    _, robot, part, grasp_id = action
    C = self.scene_for(s0, assembly)                      # part sits on the table (fixed target)
    grasp = cube_grasp_pose(grasp_id, assembly.cube_size)  # gripper pose relative to `part`
    # align the grasp's +z approach with THIS EE's real approach axis (identity for the
    # two-finger, a rotation for the vacuum) -- so the tool-center is oriented correctly.
    ee_target = np.concatenate([grasp[:3], _quat_mul(grasp[3:], self.q_correction[robot])])

    robot_ee_name = self.gripper_frame[robot]
    part_name = part

    # the gripper is SUPPOSED to touch the part it grasps (two-finger palm/fingers close
    # around it), so exclude the target part from collision -- otherwise the collision
    # objective forbids the grasp itself.
    C.getFrame(part_name).setContact(0)

    komo = ry.KOMO(
        C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=True
    )
    # no scene-penetration: without this the arm reaches the grasp by driving its links
    # THROUGH the table and the other cubes -- the tool center looks right but the pose is
    # physically bogus, and grasps that are actually unreachable wrongly pass. (the target
    # part is excluded above, since the gripper is supposed to touch it.)
    komo.addObjective(
        [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0]
    )

    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 1, 1e-1)

    # 6-DOF: gripper tool-center must sit at the (approach-corrected) grasp on the part:
    #   pose(ee) relative to pose(part) == ee_target.
    komo.addObjective(
        [1, 1], ry.FS.positionRel, [robot_ee_name, part_name], ry.OT.eq, [1e1], ee_target[:3]
    )
    komo.addObjective(
        [1, 1], ry.FS.quaternionRel, [robot_ee_name, part_name], ry.OT.eq, [1e1], ee_target[3:]
    )
    self._add_grasp_maintenance(komo, s0, assembly)       # brace arms posed at their grasps

    feasible, witness = self._solve_ik(komo)

    if self.debug_view:
        print("pick feasible:", feasible)
        komo.view(True, f"pick IK: {robot} -> {part} (grasp {grasp_id})")

    return feasible, witness

  def _ik_add_grasp(self, action, s0, s1, assembly, seed):
    # a SECOND robot reaches a grasp on an ALREADY-HELD subassembly (the handover / brace
    # primitive). Like a pick, but the target is held (posed by its current holder) rather
    # than resting on the table: scene_for attaches the group to its kinematic holder, that
    # arm may co-optimise (control-cost-anchored) to present a face, and the new arm reaches
    # the grasp. Existing braces stay pinned via grasp maintenance.
    _, robot, gpart, grasp_id = action
    C = self.scene_for(s0, assembly)                      # group held at its current holder(s)
    grasp = cube_grasp_pose(grasp_id, assembly.cube_size)
    ee_target = np.concatenate([grasp[:3], _quat_mul(grasp[3:], self.q_correction[robot])])
    C.getFrame(gpart).setContact(0)                       # the new gripper is meant to touch gpart

    komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0])
    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 1, 1e-1)
    komo.addObjective([1, 1], ry.FS.positionRel, [self.gripper_frame[robot], gpart],
                      ry.OT.eq, [1e1], ee_target[:3])
    komo.addObjective([1, 1], ry.FS.quaternionRel, [self.gripper_frame[robot], gpart],
                      ry.OT.eq, [1e1], ee_target[3:])
    self._add_grasp_maintenance(komo, s0, assembly)       # existing braces stay at their grasps

    feasible, witness = self._solve_ik(komo)

    if self.debug_view:
        print("add_grasp feasible:", feasible)
        komo.view(True, f"add_grasp IK: {robot} -> {gpart} (grasp {grasp_id})")

    return feasible, witness

  def _ik_place(self, action, s0, s1, assembly, seed):
    # ONE robot (the placer) sets its held subassembly onto the table, resting on the
    # placement FACE the action names. The target is the pose the DISCRETE model will
    # assume afterwards (scene_for/_pose_on_table's rest pose anchored at the lowest-index
    # part's initial table x/y) -- the IK must land the group exactly there.
    _, robot, root, face = action
    C = self.scene_for(s0, assembly)                      # group attached to robot's gripper
    group = next(g for g in s0.subassemblies() if root in g)
    ref = min(group, key=_part_index)                     # scene_for's on-table reference
    # target the pose the RESULT state records for the group (its landing x/y): the initial
    # env pose for an ordinary place, but the stow cell when the decluttering search relocates
    # a blocker. Falls back to the env pose when the state carries no pose payload.
    anchor = s1.poses.get(ref)
    anchor = np.asarray(anchor) if anchor is not None else np.asarray(self.C.getFrame(ref).getPose())
    target = _group_rest_pose(assembly, group, ref, anchor, face)

    komo = ry.KOMO(
        C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=True
    )
    # resting on the table is intended contact (0 penetration satisfies the ineq); real
    # penetrations (through the table / into neighbours) still reject.
    komo.addObjective(
        [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0]
    )
    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 1, 1e-1)

    # object-pose constraint (like assemble, not an EE pose): the group's reference part
    # must reach its canonical table pose, axis-aligned.
    komo.addObjective([1, 1], ry.FS.position, [ref], ry.OT.eq, [1e1], target[:3])
    komo.addObjective([1, 1], ry.FS.quaternion, [ref], ry.OT.eq, [1e1], target[3:])
    self._add_grasp_maintenance(komo, s0, assembly)       # braces ride along / stay pinned

    feasible, witness = self._solve_ik(komo)

    if self.debug_view:
        print("place feasible:", feasible)
        komo.view(True, f"place IK: {robot} sets down {root} (ref {ref})")

    return feasible, witness

  def _assemble_komo(self, C, part_name, onto, rel_pos, rel_quat, bias_axes):
    # build the assemble KOMO for one orientation bias. Shared by every step of the
    # orientation sweep (only the soft bias objective changes).
    komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=True)
    # no scene-penetration: keep the arm(s) from driving links through the table / cubes.
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0])
    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 1, 1e-1)
    # cube-pose constraint (NOT an EE pose): pose(part) relative to pose(onto) == the mate.
    #   positionRel([A, B]) / quaternionRel([A, B]) = pose of A expressed in B's frame.
    komo.addObjective([1, 1], ry.FS.positionRel, [part_name, onto], ry.OT.eq, [1e1], rel_pos)
    komo.addObjective([1, 1], ry.FS.quaternionRel, [part_name, onto], ry.OT.eq, [1e1], rel_quat)
    # slight position bias: pull the centre body toward the middle of the workspace. LOW
    # weight vs the mate eq -- only breaks the free-position DOF of a HELD assembly (inert on
    # the table). Keeps held assemblies centred instead of drifting to the reachable fringe.
    komo.addObjective(
        [1, 1], ry.FS.position, [onto], ry.OT.sos, [_ASSEMBLE_POS_BIAS_W],
        list(_ASSEMBLE_POS_BIAS_TARGET),
    )
    # soft orientation bias on the centre body (`onto`): nudge its local +z / +x axes toward
    # the given world directions. LOW weight vs the mate eq (1e1), so it only breaks the
    # free-orientation DOF of a HELD assembly; inert when `onto` rests on the table (fixed).
    # Axis alignment is sign-free -- no quaternion double cover.
    if bias_axes is not None:
      z_dir, x_dir = bias_axes
      komo.addObjective([1, 1], ry.FS.vectorZ, [onto], ry.OT.sos, [_ASSEMBLE_ORIENT_BIAS_W], list(z_dir))
      komo.addObjective([1, 1], ry.FS.vectorX, [onto], ry.OT.sos, [_ASSEMBLE_ORIENT_BIAS_W], list(x_dir))
    return komo

  def _ik_assemble(self, action, s0, s1, assembly, seed, bias_axes=None):
    # solve for the added CUBE's pose: the added part must reach its mate pose relative to
    # the target subassembly. Active = the adder + whoever holds the target (if held) -- so
    # the assembly pose is co-optimized. If the target is on the table it is a fixed target
    # and only the adder is active (all others held near home by the control cost).
    #
    # STRUCTURED RESTARTS: instead of random inits, sweep the canonical assembly
    # orientations (`_ASSEMBLE_BIAS_ORIENTATIONS`) -- each biases the held assembly toward a
    # different sensible pose -- and accept the first that reaches the mate collision-free.
    # If none do, fall back to a few unbiased random restarts. When the target rests on the
    # table its orientation is fixed, so the first bias already succeeds (sweep is cheap).
    # Pass an explicit `bias_axes` to force one orientation (skips the sweep).
    _, robot, part, onto = action
    C = self.scene_for(s0, assembly)                      # `part` is attached to robot's gripper
    part_name = part
    # mate pose of `part` in `onto`'s frame: a pure OFFSET (parts stay axis-aligned, so the
    # relative orientation is identity). relative_pose returns (parent, position_3vec).
    rel_pos = np.asarray(assembly.relative_pose(part)[1], dtype=float)
    rel_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # the mate seats `part` flush against `onto`, so their contact is intended -- exclude the
    # moving part from collision (otherwise the mate itself reads as a penetration). The
    # cheap floating-EE check already covers this part-vs-scene geometry.
    C.getFrame(part_name).setContact(0)

    biases = [bias_axes] if bias_axes is not None else _ASSEMBLE_BIAS_ORIENTATIONS
    komo = None
    feasible, witness = False, None
    for bias in biases:                                   # deterministic orientation sweep
      komo = self._assemble_komo(C, part_name, onto, rel_pos, rel_quat, bias)
      self._add_grasp_maintenance(komo, s0, assembly)     # braces pinned at their grasps
      feasible, witness = self._solve_ik(komo, restarts=0)  # one biased solve per orientation
      if feasible:
        break
    if not feasible:                                      # fallback: unbiased random restarts
      komo = self._assemble_komo(C, part_name, onto, rel_pos, rel_quat, None)
      self._add_grasp_maintenance(komo, s0, assembly)
      feasible, witness = self._solve_ik(komo, restarts=6)

    if self.debug_view:
        print("assemble feasible:", feasible)
        komo.view(True, f"assemble IK: {robot} mates {part} -> {onto}")

    return feasible, witness


# actions that move a part into a possibly-colliding pose (so they need the CHEAP geometry
# check on the RESULT). release does not move any part, so its geometry is unchanged.
_GEOMETRIC_ACTIONS = {"assemble", "place", "set_down"}

# actions that place a NEW gripper at a grasp (checked cheaply like a pick: the EE at the
# candidate grasp alongside the state's existing EEs -- catches gripper/gripper for a 2nd
# grasp and gripper/neighbour for a pick).
_GRASP_ACTIONS = {"pick", "add_grasp", "handover"}

# actions that require a robot to REACH a new grasp / pose (so they need the L3 IK check).
# release only opens a gripper, so it needs no new reach.
_IK_ACTIONS = {"pick", "assemble", "place", "set_down", "add_grasp", "handover"}


def cheap_feasible(action, s0, s1, assembly, checker):
  # L-cheap: floating-EE collision -- a SOUND lower bound (gripper collides => arm
  # collides). Two variants:
  #   * actions that move a part into contact (assemble/place/set_down): check the
  #     RESULTING state s1;
  #   * grasp actions (pick / add_grasp / handover): check the candidate grasp -- the EE at
  #     that grasp WITH the state's existing EEs present (so a 2nd grasp is tested against the
  #     holder's gripper, and a pick against its neighbours). The precomputed grasp table only
  #     sees the subassembly itself, so this is where env/holder collisions are caught.
  if action[0] in _GEOMETRIC_ACTIONS and not checker.collision_free(s1, assembly):
    return False
  if action[0] in _GRASP_ACTIONS and not checker.pick_collision_free(
      s0, action[2], action[3], assembly):
    return False
  return True


def feasible(action, s0, s1, assembly, checker, robot_ik=None):
  # Layered feasibility, cheap first (design doc 5.1):
  #   L-cheap  : floating-EE collision (see cheap_feasible).
  #   L3       : real robots -- reachability + arm collision via joint IK (RobotIK.bind).
  #              Skipped entirely when robot_ik is None (current behaviour).
  #   L2 wrench: still stubbed True.
  if not cheap_feasible(action, s0, s1, assembly, checker):
    return False
  if robot_ik is not None and action[0] in _IK_ACTIONS:
    ok, _witness = robot_ik.bind(action, s0, s1, assembly)
    if not ok:
      return False
  return True


def _reconstruct_plan(came_from, goal):
  # walk parent links back to the start; return the list of actions start -> goal
  actions = []
  s = goal
  while came_from[s][0] is not None:
    prev, action = came_from[s]
    actions.append(action)
    s = prev
  return list(reversed(actions))


# Event-based action costs (design doc §1): grabs/releases and transports cost; a
# transfer that avoids re-gripping is the cheap path. Uniform for now; refine later.
_ACTION_COST = {
    "pick": 1.0, "place": 1.0, "assemble": 1.0, "release": 1.0,
    "add_grasp": 1.0, "set_down": 1.0,
    "stow": 1.0,                  # relocate a blocker aside (decluttering search only)
    "handover": 2.0,             # fused regrasp-in-air = a close + an open (== add_grasp;release)
}


def action_cost(action):
  return _ACTION_COST[action[0]]


def heuristic(state, assembly):
  """Admissible lower bound on the remaining cost. Each part not yet joined must still
  be (a) assembled -- one `assemble` -- and (b) brought into a gripper -- one `pick` --
  unless it is already held. Releases, regrasps and base-lifts are ignored (they only
  add cost), so this never overestimates."""
  h = 0.0
  for p in assembly.objects:
    if p == assembly.base:
      continue                                # the base is never a mate op
    if any(c[0] == "mate" for c in state.supports.get(p, ())):
      continue                                # already assembled (has a mate contact;
                                              # robust to re-rooting, unlike parent edges)
    h += _ACTION_COST["assemble"]
    held = any(c[0] == "grasp" for c in state.supports.get(p, ()))
    if not held:
      h += _ACTION_COST["pick"]
  return h


class AbstractionHeuristic:
  """Admissible abstraction heuristic (pattern-database style): solve a RELAXED model of
  the problem exactly, once, by backward BFS; use its goal distances as h. The cheap
  geometric layer informs the abstraction (on_table_mateable), so the heuristic prices in
  forced base-lifts, braces, releases and gripper occupancy -- the plateau killers -- with
  no hand-written penalties.

  Abstract state: (ideal, held, holders, on_table)
    ideal    : frozenset of assembled parts (the main component, incl. the base),
    held     : frozenset of LOOSE (unassembled) parts currently in a gripper,
    holders  : # grippers on the main component (holds and braces are the same thing),
    on_table : main component resting on the table?
  Robot identity and grasp choice are dropped (relaxation: only ADDS options). Abstract
  actions mirror the real primitives at unit cost, gated the same way (support-gated
  release, dependency-gated assemble, geometry via on_table_mateable -- which only forbids
  what the cheap layer soundly rejects). Every real action maps to an abstract action, so
  abstract goal distance <= real remaining cost: ADMISSIBLE. Combine with max()."""

  def __init__(self, assembly, n_robots, gripper_kind=None):
    self.assembly = assembly
    self.R = n_robots
    objs = list(assembly.objects)
    base = assembly.base
    # abstraction drops the placement ORIENTATION, so the on-table-mate fact must be the
    # most permissive over faces (exists-a-face): only then is every real action abstract-
    # feasible (relaxation -> admissibility preserved).
    on_table_ok = {p: any(_on_table_mateable_cached(assembly, p, f)
                          for f in range(len(_FACE_DOWN_QUATS)))
                   for p in objs if p != base}

    # WRENCH-AWARE gate: with the stability model on, an on-table mate also needs to be
    # wrench-stable (rest-mateable at SOME face). Where it is NOT, the abstract assemble may
    # only happen with the assembly HELD (off the table) -- so the abstraction prices in the
    # forced pick/hold (and its backward search amortises one hold across consecutive air-
    # mates). Exists-a-face is more permissive than the real specific-face rule, so the bound
    # stays admissible. gripper_kind None => topological baseline (mate_ok = on_table_ok).
    rest_mateable = {}
    if gripper_kind is not None:
      rest_mateable = {k: bool(v)
                       for k, v in precompute_stable_faces(assembly, gripper_kind).items()}

    def mate_ok(ideal, p):
      if gripper_kind is None:
        return on_table_ok[p]                               # geometric only (baseline)
      return rest_mateable.get((_subasm_key(ideal), p), False)

    def allowed_next(ideal):
      return [p for p in objs if p not in ideal and _mate_closure(assembly, p) <= ideal]

    # enumerate the dependency-closed ideals (the assembly's possible "done" sets)
    ideals = {frozenset({base})}
    frontier = [frozenset({base})]
    while frontier:
      cur = frontier.pop()
      for p in allowed_next(cur):
        grown = cur | {p}
        if grown not in ideals:
          ideals.add(grown)
          frontier.append(grown)

    # all valid abstract states: gripper budget respected, support invariant holds
    states = []
    for ideal in ideals:
      loose = [p for p in objs if p not in ideal]
      for held_mask in range(1 << len(loose)):
        held = frozenset(p for i, p in enumerate(loose) if held_mask >> i & 1)
        for holders in range(self.R - len(held) + 1):
          for on_table in (False, True):
            if holders == 0 and not on_table:
              continue                                    # unsupported -> unreachable
            states.append((ideal, held, holders, on_table))

    def succs(st):
      ideal, held, holders, on_table = st
      free = self.R - len(held) - holders
      out = []
      if free >= 1:
        for p in objs:                                    # pick a loose on-table part
          if p not in ideal and p not in held:
            out.append((ideal, held | {p}, holders, on_table))
        if on_table and holders == 0:                     # pick (lift) the assembly
          out.append((ideal, held, 1, False))
      for p in held:                                      # assemble p (gripper -> brace)
        if _mate_closure(assembly, p) <= ideal and (mate_ok(ideal, p) if on_table else True):
          out.append((ideal | {p}, held - {p}, holders + 1, on_table))
      if holders >= 1 and (on_table or holders >= 2):     # support-gated release
        out.append((ideal, held, holders - 1, on_table))
      # place: set a held thing back on the table and open (must mirror the real `places`,
      # or h would overestimate plans that use place-regrasps -> inadmissible).
      for p in held:                                      # place a held loose part
        out.append((ideal, held - {p}, holders, on_table))
      if holders >= 1:                                    # place the assembly (root holder)
        out.append((ideal, held, holders - 1, True))
      return out

    # backward uniform-cost = multi-source BFS from the goal states over reversed edges
    all_parts = frozenset(objs)
    from collections import deque
    preds = {}
    for st in states:
      for ns in succs(st):
        preds.setdefault(ns, []).append(st)
    self.dist = {}
    queue = deque()
    for st in states:
      if st[0] == all_parts:                              # goal: everything assembled
        self.dist[st] = 0.0
        queue.append(st)
    while queue:
      st = queue.popleft()
      for pr in preds.get(st, ()):
        if pr not in self.dist:
          self.dist[pr] = self.dist[st] + 1.0
          queue.append(pr)
    self.n_states = len(states)

  def abstract(self, state):
    # map a concrete search State onto the abstraction
    main = next(g for g in state.subassemblies() if self.assembly.base in g)
    ideal = frozenset(main)
    holders = sum(1 for p in main for c in state.supports.get(p, ()) if c[0] == "grasp")
    on_table = any(("table",) in state.supports.get(p, ()) for p in main)
    held = frozenset(p for p in state.parent if p not in main
                     and any(c[0] == "grasp" for c in state.supports.get(p, ())))
    return (ideal, held, min(holders, self.R), on_table)

  def value(self, state):
    return self.dist.get(self.abstract(state), 0.0)


def _astar(start, *, goal_fn, succ_fn, cost_fn, h_fn, cheap_fn=None,
           edge_check_fn=None, weight=1.0, max_nodes=200000, verbose=True,
           tag="A*", t0=None, cost_bound=None, deadline=None):
  """Generic LAZY weighted-A* engine, shared by every search variant (exact + optimistic).

  The domain enters only through callbacks -- this function knows nothing about assemblies:
    goal_fn(state)                 -> bool
    succ_fn(state)                 -> iterable of (action, next_state)
    cost_fn(action)                -> float (>= 0)
    h_fn(state)                    -> float (>= 0); admissible => weight=1 is optimal
    cheap_fn(prev, action, next)   -> bool : EAGER cheap filter at PUSH (None = none)
    edge_check_fn(prev, act, next) -> bool : EXPENSIVE check at POP, lazily (None = none)

  Priority f = h (greedy, weight=inf) or g + weight*h. A state is CLAIMED by the first
  popped edge that passes edge_check_fn; duplicates are left in the heap so a state pruned
  via one (infeasible) edge stays reachable via its others.

  cost_bound: prune every node with g + h >= cost_bound (h admissible => sound: no plan
  cheaper than the bound is lost). The anytime driver passes the incumbent's cost so each
  restart only looks for strictly better plans. deadline: perf_counter time to give up at.
  Returns the action list, or None (max_nodes / deadline / bound-exhausted)."""
  import heapq
  import itertools
  import math
  import time

  if t0 is None:
    t0 = time.perf_counter()

  def f_value(g, hval):
    return hval if math.isinf(weight) else g + weight * hval

  if goal_fn(start):
    return []
  tie = itertools.count()                                 # stable tie-break (shallower g first)
  claimed = {}
  came_from = {start: (None, None)}
  h0 = h_fn(start)
  pq = [(f_value(0.0, h0), h0, 0.0, next(tie), start, None, None)]
  optimal = weight == 1.0
  expanded = 0
  edge_checks = 0
  while pq:
    _f, _h, g, _, s, prev, action = heapq.heappop(pq)
    if s in claimed:
      continue                                            # already claimed via a cheaper/earlier edge
    if cost_bound is not None and g + _h >= cost_bound:
      continue                                            # cannot beat the incumbent (h admissible)
    if deadline is not None and time.perf_counter() > deadline:
      if verbose:
        print(f"[{tag}] deadline reached after {expanded} expansions")
      return None
    if action is not None and edge_check_fn is not None:
      edge_checks += 1
      if not edge_check_fn(prev, action, s):
        continue                                          # infeasible edge -- other queued edges remain
    claimed[s] = g
    if action is not None:
      came_from[s] = (prev, action)

    if goal_fn(s):
      plan = _reconstruct_plan(came_from, s)
      if verbose:
        bound = ("optimal" if optimal else "greedy, no bound" if math.isinf(weight)
                 else f"cost <= {weight:g}x optimal")
        print(f"[{tag}] found a {len(plan)}-action plan (cost {g:g}, {bound}); "
              f"expanded {expanded} nodes, {len(claimed)} states claimed, "
              f"{edge_checks} edge checks in {time.perf_counter() - t0:.2f}s")
      return plan

    expanded += 1
    if expanded > max_nodes:
      if verbose:
        print(f"[{tag}] gave up after {max_nodes} expansions ({len(claimed)} states "
              f"claimed) in {time.perf_counter() - t0:.2f}s")
      return None

    for act, ns in succ_fn(s):
      if ns in claimed:
        continue
      if cheap_fn is not None and not cheap_fn(s, act, ns):
        continue                                          # eager cheap filter (prunes heap growth)
      ng = g + cost_fn(act)
      hns = h_fn(ns)
      if cost_bound is not None and ng + hns >= cost_bound:
        continue                                          # sound prune vs the incumbent
      heapq.heappush(pq, (f_value(ng, hns), hns, ng, next(tie), ns, s, act))

  if verbose:
    print(f"[{tag}] exhausted {len(claimed)} states with no plan "
          f"in {time.perf_counter() - t0:.2f}s")
  return None


def _search_context(C, assembly, robots):
  """The shared, built-once machinery a domain search needs: the cheap collision checker
  and the precomputed grasp table, both keyed to the real gripper (auto-detected)."""
  # the cheap checker's floating EE must MATCH the real gripper: a vacuum stand-in is much
  # slimmer than the two-finger, so it would let grasps through that the real palm/fingers
  # rule out (neighbouring cubes, finger-under-the-cube-into-the-table).
  vacuum = any(n.endswith("_ur_vacuum") for n in C.getFrameNames())
  ee_path = os.path.join(os.path.dirname(__file__), UR10_PATHS[(True, vacuum)])
  checker = CollisionChecker(assembly, robots, gripper_path=ee_path)
  grasp_table = precompute_grasps(assembly, gripper_path=ee_path)
  return checker, grasp_table


def gripper_kind(C):
  # which gripper the scene uses, as the key into wrench.PARAMS['grasp'].
  return "vacuum" if any(n.endswith("_ur_vacuum") for n in C.getFrameNames()) \
      else "two_finger"


def solve_assembly_problem(C, assembly, robots=None, max_nodes=200000, verbose=True,
                           use_heuristic=True, canonical=False, weight=1.0, robot_ik=None,
                           wrench=False, wrench_aware_h=True, handovers=True,
                           allow_place=True, cost_bound=None, deadline=None,
                           anonymous=False):
  """EXACT (weighted) A* over scene-graph states from the initial scene to a completed
  assembly -- the reference baseline. Priority is f = g + weight * h.
    - weight = 1.0 : ordinary A*, returns a minimum-cost plan.
    - weight > 1.0 : greedier and faster, plan cost bounded by weight * optimal.
    - weight = inf : greedy best-first (ignore g), fastest, no quality bound.
  use_heuristic=False forces h=0 (uniform-cost / BFS). Returns the list of actions, or
  None if not found within `max_nodes` expansions.

  wrench=True turns on the static-equilibrium stability model (support-gating on assemble/
  release, incl. the insertion reaction); default False keeps the topological baseline.
  wrench_aware_h makes the abstraction heuristic price in wrench-forced holds (set False to
  keep the old topological heuristic for comparison under the same wrench gates).

  canonical=True enables ROBOT symmetry breaking in `picks` (lowest-index free robot).
  TEST-ONLY -- unsound once robots have fixed bases. See the warning below."""
  import math
  import time
  import warnings

  t0 = time.perf_counter()
  if canonical:
    warnings.warn(
        "solve_assembly_problem(canonical=True): breaking ROBOT symmetry (lowest-index "
        "free robot only). Sound while grippers are interchangeable floating EEs; UNSOUND "
        "once robots have fixed bases / different reachability -- the search may then miss "
        "feasible plans. Disable canonical before introducing those.",
        stacklevel=2,
    )
  if robots is None:
    robots = get_robots(C)
  gk = gripper_kind(C) if wrench else None                # None => topological baseline

  # abstraction heuristic: exact backward solve of a relaxed model. max() with the simple
  # counting heuristic keeps admissibility. Used only for FINITE weights: A*/wA* consume h
  # as a lower BOUND (tightness collapses the frontier); pure greedy consumes it as a
  # GRADIENT and h_abs's honest plateaus make greedy sweep equivalent-variant shells. Under
  # wrench the abstraction is made stability-aware (prices in forced holds) unless disabled.
  h_abs = AbstractionHeuristic(assembly, len(robots),
                               gripper_kind=gk if wrench_aware_h else None) \
      if use_heuristic and not math.isinf(weight) else None

  def h_fn(state):
    if not use_heuristic:
      return 0.0
    hv = heuristic(state, assembly)
    return max(hv, h_abs.value(state)) if h_abs is not None else hv

  checker, grasp_table = _search_context(C, assembly, robots)
  if gk is not None:
    reset_wrench_cache()

  def succ_fn(s):
    return successors(s, assembly, robots, grasp_table=grasp_table, canonical=canonical,
                      gripper_kind=gk, handovers=handovers, allow_place=allow_place)

  def cheap_fn(prev, act, ns):
    return cheap_feasible(act, prev, ns, assembly, checker)

  def edge_check_fn(prev, act, ns):
    if act[0] not in _IK_ACTIONS:
      return True
    ok, _witness = robot_ik.bind(act, prev, ns, assembly)
    return ok

  # LATE BINDING (anonymous=True): search with a robot-anonymous state key -- states that
  # differ only in WHICH robot holds what merge, so robot-permutation redundancy vanishes and
  # extra idle robots add no states. The returned plan still names concrete robots (the merged
  # class's representative assignment). Sound for interchangeable (floating) grippers, like
  # `canonical`; with fixed-base arms + robot_ik a merged assignment may hide a reachability
  # difference -- use for the fast first answer, then verify/repair with concrete IK.
  if anonymous and robot_ik is not None and verbose:
    warnings.warn("anonymous=True with robot_ik: merged robot assignments may hide "
                  "per-robot reachability differences; treat the plan as a candidate.")
  tag = ("UCS" if not use_heuristic else "greedy" if math.isinf(weight)
         else "A*" if weight == 1.0 else f"wA*(w={weight:g})")
  set_anonymous_robots(anonymous)
  try:
    return _astar(get_initial_state(C, assembly),
                  goal_fn=lambda s: s.is_goal(assembly),
                  succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn, cheap_fn=cheap_fn,
                  edge_check_fn=edge_check_fn if robot_ik is not None else None,
                  weight=weight, max_nodes=max_nodes, verbose=verbose, tag=tag, t0=t0,
                  cost_bound=cost_bound, deadline=deadline)
  finally:
    set_anonymous_robots(False)


def solve_assembly_problem_anytime(C, assembly, robots=None, time_budget=60.0,
                                   weights=(float("inf"), 5.0, 2.0, 1.5, 1.0),
                                   verbose=True, **kw):
  """ANYTIME planner: restarting weighted A* (RWA*). Runs the exact search with a DECREASING
  weight schedule -- the first (greedy / high-weight) pass returns a suboptimal plan FAST;
  every later pass searches only for STRICTLY BETTER plans (the incumbent's cost is a sound
  prune bound, h admissible) and either improves the incumbent or proves a tighter bound,
  until the weight-1 pass ends (incumbent = optimal) or the time budget runs out. Returns
  (best_plan, cost, optimal_proved). Accepts every solve_assembly_problem kwarg."""
  import math
  import time

  t0 = time.perf_counter()
  deadline = t0 + time_budget
  if robots is None:
    robots = get_robots(C)
  best, best_cost = None, float("inf")
  proved = False
  for w in weights:
    if time.perf_counter() > deadline:
      break
    plan = solve_assembly_problem(
        C, assembly, robots=robots, weight=w, verbose=verbose,
        cost_bound=best_cost if best is not None else None, deadline=deadline, **kw)
    if plan is not None:
      cost = sum(action_cost(a) for a in plan)
      if cost < best_cost:
        best, best_cost = plan, cost
        if verbose:
          print(f"[anytime] incumbent improved: cost {cost:g} (w={w:g}, "
                f"{time.perf_counter() - t0:.1f}s)")
    elif best is not None and w == 1.0 and time.perf_counter() <= deadline:
      # the w=1 restart exhausted under the incumbent bound WITHOUT timing out: nothing
      # strictly better exists -> the incumbent is optimal.
      proved = True
      if verbose:
        print(f"[anytime] incumbent proved optimal (cost {best_cost:g})")
  if verbose:
    print(f"[anytime] done: cost {best_cost:g}, optimal={proved}, "
          f"{time.perf_counter() - t0:.1f}s total")
  return best, best_cost, proved


# ===========================================================================
# Optimistic (Dogar-style) planner: per-assemble decomposition
# ===========================================================================
#
# "Optimistically plan the assemble steps, figure out the in-between later." Rather than
# one search over the full choreography of ALL mates (whose blowup is the cross-product of
# every mate's reorientations/holds/grasps), we plan each assemble step as its OWN small,
# complete search that adds exactly one part -- reorient / hold-by-any-part / pick / mate /
# set back down -- starting and ending with the assembly resting on the table so the steps
# compose. Optimistic because it assumes the assembly can always be returned to the table
# between mates (it can -- set it down; wrench/stability is deferred). Complete for
# everything the exact search does, because each sub-search IS an exact search over the
# real primitives, just scoped to one part; only the CROSS-PRODUCT between mates is dropped.


def _num_loose_held(state, assembly):
  # grasps on parts NOT in the main (base) subassembly -- i.e. loose parts held in a gripper.
  main = next((g for g in state.subassemblies() if assembly.base in g), frozenset())
  return sum(1 for p in state.bodies() if p not in main
             for c in state.supports.get(p, ()) if c[0] == "grasp")


def _assembly_holders(state, assembly):
  # how many grippers hold the main (base) subassembly (kinematic holder + braces).
  main = next((g for g in state.subassemblies() if assembly.base in g), frozenset())
  return sum(1 for p in main for c in state.supports.get(p, ()) if c[0] == "grasp")


def _part_mated(state, p):
  return any(c[0] == "mate" for c in state.supports.get(p, ()))


def _add_one_part(start, p, ideal, assembly, robots, checker, grasp_table, robot_ik,
                  weight, max_nodes, verbose, gripper_kind=None, handovers=True,
                  allow_place=True):
  # a small COMPLETE search that adds part `p` to the assembly `ideal` and leaves it SETTLED:
  # p mated, no loose part left in a gripper, and the assembly ending held by at most one
  # carrier OR resting on the table -- WHICHEVER the search finds. It forces neither a
  # set-down nor a hold; holding-through and setting-down are just options (allow_place=False
  # drops the set-down option, so the assembly naturally stays held). It may only touch `p`
  # and the assembly (parts in `ideal`). Returns (subplan, end_state), gripper_kind => wrench.
  relevant = set(ideal) | {p}

  def goal_fn(s):
    # p joined, nothing loose left in hand, and the assembly down to <=1 holder (a clean
    # rest point for the next step -- prevents inserter grasps / braces piling up).
    return (_part_mated(s, p) and _num_loose_held(s, assembly) == 0
            and _assembly_holders(s, assembly) <= 1)

  def h_fn(s):
    left = 0 if _part_mated(s, p) else 2                  # pick + mate remaining
    return left + _num_loose_held(s, assembly) + max(0, _assembly_holders(s, assembly) - 1)

  def succ_fn(s):
    for a, ns in successors(s, assembly, robots, grasp_table=grasp_table,
                            gripper_kind=gripper_kind, handovers=handovers,
                            allow_place=allow_place):
      if a[2] in relevant:                                # a[2] is the part every action acts on
        yield a, ns

  def cheap_fn(prev, act, ns):
    return cheap_feasible(act, prev, ns, assembly, checker)

  def edge_check_fn(prev, act, ns):
    if act[0] not in _IK_ACTIONS:
      return True
    ok, _w = robot_ik.bind(act, prev, ns, assembly)
    return ok

  subplan = _astar(start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn,
                   cheap_fn=cheap_fn,
                   edge_check_fn=edge_check_fn if robot_ik is not None else None,
                   weight=weight, max_nodes=max_nodes, verbose=False,
                   tag=f"add {p}")
  if subplan is None:
    return None, None
  s = start
  for a in subplan:
    s = next(ns for act, ns in successors(s, assembly, robots, gripper_kind=gripper_kind,
                                          handovers=handovers, allow_place=allow_place)
             if act == a)
  return subplan, s


def solve_assembly_problem_optimistic(C, assembly, robots=None, verbose=True,
                                      robot_ik=None, fallback=True, weight=float("inf"),
                                      max_nodes=20000, wrench=False, handovers=True,
                                      allow_place=True, **exact_kw):
  """OPTIMISTIC (Dogar-style) planner (see the section header): add the parts one at a time,
  each via its own small complete search that reorients/holds/picks/mates/sets-down, chained
  through the on-table assembly. This drops the exact search's cross-product blowup while
  staying complete. Falls back to the exact search if any single-part step is unsolvable
  within `max_nodes`. wrench=True turns on the stability gates. allow_place=False (--no-place)
  chains HELD states instead (hold-and-work with handovers). Returns the action list, or None."""
  import time

  t0 = time.perf_counter()
  if robots is None:
    robots = get_robots(C)
  checker, grasp_table = _search_context(C, assembly, robots)
  gk = gripper_kind(C) if wrench else None
  if gk is not None:
    reset_wrench_cache()

  s = get_initial_state(C, assembly)
  plan = []
  ideal = {assembly.base}
  for p in assembly_order(assembly):
    if p == assembly.base:
      continue
    sub, s2 = _add_one_part(s, p, ideal, assembly, robots, checker, grasp_table, robot_ik,
                            weight, max_nodes, verbose, gripper_kind=gk, handovers=handovers,
                            allow_place=allow_place)
    if sub is None:
      if verbose:
        print(f"[optimistic] could not add {p} in isolation -> falling back to exact")
      if not fallback:
        return None
      return solve_assembly_problem(C, assembly, robots=robots, verbose=verbose,
                                    robot_ik=robot_ik, weight=weight, max_nodes=max_nodes,
                                    wrench=wrench, handovers=handovers,
                                    allow_place=allow_place, **exact_kw)
    plan += sub
    s = s2
    ideal.add(p)
    if verbose:
      print(f"[optimistic] added {p} in {len(sub)} actions "
            f"({time.perf_counter() - t0:.1f}s, {len(plan)} total)")

  if verbose:
    print(f"[optimistic] complete: {len(plan)} actions in {time.perf_counter() - t0:.2f}s")
  return plan


# ===========================================================================
# Optimistic + DECLUTTER: lazy targeted relocation of blocking loose parts
# ===========================================================================
#
# The per-part search above may only touch the part being added and the already-built
# assembly -- it can never move a still-loose part out of the way. That is sound only
# because of the exact fallback: when a loose part physically blocks the pick or the mate,
# the plain optimistic search abandons the step and the whole problem re-solves exactly.
#
# This variant removes that assumption. When a single-part step fails, it ATTRIBUTES the
# failure to the specific loose parts that collided (from the cheap checker's penetration
# pairs), admits ONLY those as movable, and re-plans -- the sub-search may now also STOW a
# blocker (pick it and set it down in a dedicated clear cell out of the way) before adding
# the part. This is the Dogar move: clear exactly the obstacles you actually hit, lazily,
# so the branching stays bounded (a plain "may move anything" search would reintroduce the
# cross-product blowup). It resolves the geometric blocking case without the exact
# fallback; blockers only an ARM sweeps into (invisible to the cheap floating-EE layer)
# are not attributed and still fall back.


# how far (metres, +y) a stowed part is shifted from its scatter pose into the clear back
# row. The scatter is a single line at y=0 (add_disassembled_obj), so +y is empty; each
# part keeps its own x, so distinct parts' stow cells never overlap.
_STOW_Y_OFFSET = 0.4


def _stow_pose(pose):
  # a loose part's stow pose: its current pose shifted into the clear back row (keep x/z and
  # orientation, offset y). Deterministic per part -> its stow cell is unique and fixed.
  pose = np.asarray(pose, dtype=float).copy()
  pose[1] += _STOW_Y_OFFSET
  return pose


def _is_stowed(state, q):
  pl = state.place.get(q)
  return pl is not None and pl[0] == "stow"


def _is_held(state, q):
  return any(c[0] == "grasp" for c in state.supports.get(q, ()))


def _stow_moves(state, movable):
  # successor edges that MOVE A BLOCKER out of the way: a robot holding a loose single part
  # flagged `movable` sets it down at its dedicated stow cell (clear back row) instead of its
  # scatter spot. Like `places` but the part lands aside (state.poses updated) so it stops
  # blocking. Single cube -> one upright placement (no face branching). Emitted only for
  # admitted blockers, so the branching this adds is tiny.
  for group in state.subassemblies():
    if len(group) != 1:
      continue
    (q,) = tuple(group)
    if q not in movable:
      continue
    if state.parent.get(q, (None,))[0] != "gripper":
      continue                                            # must be held to set it down
    r = state.parent[q][1]
    ns = _copy_edit_support(state)
    ns.supports[q] = frozenset(c for c in ns.supports[q]
                               if not (c[0] == "grasp" and c[1] == r))
    ns.grasp.pop(q, None)
    ns.parent[q] = ("table",)
    ns.place[q] = ("stow", _part_index(q), _IDENTITY_FACE)
    ns.poses[q] = _stow_pose(state.poses[q])              # <-- moved aside
    ns.supports[q] = ns.supports[q] | {("table",)}
    yield ("stow", r, q, _IDENTITY_FACE), ns


def _declutter_edges(state, assembly, robots, grasp_table, movable, gripper_kind=None,
                     handovers=True, allow_place=True):
  # the full edge set of the declutter sub-search: the ordinary successors plus the
  # stow-a-blocker moves. Shared by the sub-search and the plan-realisation walk so both
  # agree on what a "stow" edge produces. (Stowing a loose blocker aside is always allowed --
  # allow_place=False only forbids setting the ASSEMBLY down.)
  yield from successors(state, assembly, robots, grasp_table=grasp_table,
                        gripper_kind=gripper_kind, handovers=handovers,
                        allow_place=allow_place)
  yield from _stow_moves(state, movable)


def _declutter_blockers(pairs, relevant, objects, slots):
  # from the checker's penetration pairs, the LOOSE parts (an object not in `relevant`) that
  # penetrate a RELEVANT body -- a relevant part or a gripper slot. These are the obstacles
  # worth relocating. loose/loose pairs (shouldn't occur in a valid scatter) and
  # relevant/relevant pairs (the action's own intended contact) are ignored.
  objset = set(objects)

  def is_slot(n):
    return any(n.startswith(s) for s in slots)

  blockers = set()
  for a, b, _d in pairs:
    for x, y in ((a, b), (b, a)):
      if x in objset and x not in relevant and ((y in relevant) or is_slot(y)):
        blockers.add(x)
  return blockers


def _add_one_part_declutter(start, p, ideal, assembly, robots, checker, grasp_table,
                            robot_ik, weight, max_nodes, verbose, gripper_kind=None,
                            handovers=True, allow_place=True):
  # add part `p` (as in _add_one_part) but allowed to RELOCATE blockers: plan with no
  # movable parts; if the step is infeasible, attribute the collision to the loose parts in
  # the way, admit them as movable, and re-plan (they can now be stowed aside). Repeat until
  # the step succeeds or no new blocker is found. Returns (subplan, end_state) or (None, None).
  # gripper_kind (non-None) turns on the wrench stability gates.
  base_relevant = set(ideal) | {p}
  objects = set(assembly.objects)
  movable = set()

  def goal_fn(s):
    # same SETTLE goal as _add_one_part (p joined, nothing loose in hand, assembly <=1
    # holder) -- blockers must end stowed (not held), so _num_loose_held == 0 covers them too.
    return (_part_mated(s, p) and _num_loose_held(s, assembly) == 0
            and _assembly_holders(s, assembly) <= 1)

  def make_h(movable):
    # remaining cost: pick+mate for p (2 until mated), the assembly down to <=1 holder, any
    # non-blocker loose part released, plus a "distance to stowed" for every admitted blocker
    # -- 2 if it still sits in the way, 1 if in a gripper, 0 once stowed. That last term is
    # the gradient that sends a greedy search straight to relocating the blockers. With no
    # blockers it collapses to the plain per-part settle h.
    def h_fn(s):
      d = sum(0 if _is_stowed(s, q) else 1 if _is_held(s, q) else 2 for q in movable)
      loose = sum(1 for q, cs in s.supports.items() for c in cs
                  if c[0] == "grasp" and q not in movable) - _assembly_holders(s, assembly)
      return ((0 if _part_mated(s, p) else 2) + max(0, _assembly_holders(s, assembly) - 1)
              + max(0, loose) + d)
    return h_fn

  while True:
    relevant = base_relevant | movable
    hit = set()

    def succ_fn(s):
      for a, ns in _declutter_edges(s, assembly, robots, grasp_table, movable,
                                    gripper_kind=gripper_kind, handovers=handovers,
                                    allow_place=allow_place):
        if a[0] == "stow" or a[2] in relevant:            # touch p, the assembly, or a blocker
          yield a, ns

    def cheap_fn(prev, act, ns):
      if act[0] == "stow":
        ok = checker.collision_free(ns, assembly)         # stow lands aside -> check the result
      else:
        ok = cheap_feasible(act, prev, ns, assembly, checker)
      if not ok and (act[2] == p or act[2] in ideal):     # a GOAL-RELEVANT action was blocked
        if act[0] == "pick":
          pairs = checker.penetrating_pairs(prev, assembly,
                                            extra_grasps=[(act[2], act[3])])
        else:
          pairs = checker.penetrating_pairs(ns, assembly)
        hit.update(_declutter_blockers(pairs, relevant, objects, checker.slots))
      return ok

    def edge_check_fn(prev, act, ns):
      if act[0] not in _IK_ACTIONS and act[0] != "stow":
        return True
      a = ("place",) + tuple(act[1:]) if act[0] == "stow" else act   # stow is a place aside
      ok, _w = robot_ik.bind(a, prev, ns, assembly)
      return ok

    subplan = _astar(start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost,
                     h_fn=make_h(movable), cheap_fn=cheap_fn,
                     edge_check_fn=edge_check_fn if robot_ik is not None else None,
                     weight=weight, max_nodes=max_nodes, verbose=False,
                     tag=f"add {p}")
    if subplan is not None:
      s = start
      for a in subplan:
        s = next(ns for act, ns in _declutter_edges(s, assembly, robots, grasp_table, movable,
                                                    gripper_kind=gripper_kind,
                                                    handovers=handovers,
                                                    allow_place=allow_place)
                 if act == a)
      return subplan, s

    new = hit - movable
    if not new:
      return None, None                                   # nothing left to relocate -> give up
    movable |= new
    if verbose:
      print(f"    [declutter] {p}: blocked by {sorted(new)} -> relocating and re-planning")


def solve_assembly_problem_optimistic_declutter(
    C, assembly, robots=None, verbose=True, robot_ik=None, fallback=True,
    weight=float("inf"), max_nodes=20000, wrench=False, handovers=True, allow_place=True,
    **exact_kw):
  """OPTIMISTIC planner WITH DECLUTTERING (see the section header): like
  solve_assembly_problem_optimistic, but each single-part step may relocate blocking loose
  parts (stow them aside) instead of abandoning the step. Removes the "only touch this part
  and the assembly" assumption. Falls back to the exact search only if a step is infeasible
  for a reason relocation cannot fix. wrench=True turns on the stability gates. Returns the
  action list, or None."""
  import time

  t0 = time.perf_counter()
  if robots is None:
    robots = get_robots(C)
  checker, grasp_table = _search_context(C, assembly, robots)
  gk = gripper_kind(C) if wrench else None
  if gk is not None:
    reset_wrench_cache()

  s = get_initial_state(C, assembly)
  plan = []
  ideal = {assembly.base}
  for p in assembly_order(assembly):
    if p == assembly.base:
      continue
    sub, s2 = _add_one_part_declutter(s, p, ideal, assembly, robots, checker, grasp_table,
                                      robot_ik, weight, max_nodes, verbose, gripper_kind=gk,
                                      handovers=handovers, allow_place=allow_place)
    if sub is None:
      if verbose:
        print(f"[declutter] could not add {p} even with relocation -> falling back to exact")
      if not fallback:
        return None
      return solve_assembly_problem(C, assembly, robots=robots, verbose=verbose,
                                    robot_ik=robot_ik, weight=weight, max_nodes=max_nodes,
                                    wrench=wrench, handovers=handovers,
                                    allow_place=allow_place, **exact_kw)
    plan += sub
    s = s2
    ideal.add(p)
    if verbose:
      print(f"[declutter] added {p} in {len(sub)} actions "
            f"({time.perf_counter() - t0:.1f}s, {len(plan)} total)")

  if verbose:
    print(f"[declutter] complete: {len(plan)} actions in {time.perf_counter() - t0:.2f}s")
  return plan
