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

    template = make_world()                                # floor + table, same as the real env
    for obj in assembly.objects:                           # the parts (posed per query)
      add_part_frame(template, obj, assembly.part_shape(obj))

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
    self._verdicts = {}                                    # discrete scene signature -> bool

  def _sig(self, state, extra_grasps):
    # DISCRETE signature of everything `_check` looks at, for verdict memoization. Per
    # subassembly: on-table groups contribute members + placement (face/slot/stow) + the
    # ref part's anchor pose (rounded; catches stowed/perturbed anchors) + their grasps;
    # held-in-air groups are PARKED far away in the check, so only their pose-invariant
    # internals matter (members + grasps). Robot identities are irrelevant to geometry
    # (the EE pose comes from the grasp), so they are omitted -- states differing only in
    # WHO holds what share one verdict. The same state is pushed/checked from many parents,
    # so hit rates are high.
    parts = []
    for group in state.subassemblies():
      members = tuple(sorted(group))
      grasps = tuple(sorted((c[2], c[3]) for p in group
                            for c in state.supports.get(p, ()) if c[0] == "grasp"))
      on_table = any(("table",) in state.supports.get(p, ()) for p in group)
      if on_table:
        ref = reference_part(group)
        anchor = tuple(np.round(np.asarray(state.poses[ref], dtype=float), 4))
        parts.append(("T", members, state.place.get(ref), anchor, grasps))
      else:
        parts.append(("H", members, grasps))
    return (tuple(sorted(parts)), tuple(sorted(extra_grasps)))

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
                                  grasp_pose(assembly, gpart, grasp_id))
      base_world = _ee_root_pose(grasp_world, self.base_rel_tool, self.q_correction)
      root = C_col.getFrame(prefix + "ur_base")
      root.setPosition(base_world[:3])
      root.setQuaternion(base_world[3:])
    # any remaining slots stay parked (from the template) -> out of the way.
    return C_col

  def _check(self, state, assembly, extra_grasps=(), note=""):
    # tolerate grazing contacts (see _CHEAP_PENETRATION_TOL) instead of binary-exact
    # getCollisionFree -- the cheap layer must not be stricter than the (soft) IK.
    # VERDICT-MEMOIZED on the discrete scene signature (see _sig): building the rai scene
    # + collision query costs ~2ms and dominated whole-search profiles (~90%).
    key = self._sig(state, tuple(map(tuple, extra_grasps)))
    hit = self._verdicts.get(key)
    if hit is not None:
      return hit
    C_col = self._scene_for(state, assembly, extra_grasps)
    ok = all(d >= -_CHEAP_PENETRATION_TOL or _wrist_table_pair(a, b)
             for a, b, d in C_col.getCollisions())
    if _CHEAP_DEBUG_VIEW:
      pairs = [(a, b, d) for a, b, d in C_col.getCollisions()
               if d < -_CHEAP_PENETRATION_TOL and not _wrist_table_pair(a, b)]
      msg = (f"cheap check: {'FREE' if ok else 'COLLIDING'}\n{note}\n"
             + "\n".join(f"  {a} x {b}  depth {d:.4f}" for a, b, d in pairs))
      C_col.view(True, msg)
    self._verdicts[key] = ok
    return ok

  def penetrating_pairs(self, state, assembly, extra_grasps=()):
    # the collision pairs that make `_check` reject: (frame_a, frame_b, depth) with
    # depth < -tol. Frame names are part names (obj_*) or gripper-slot frames (eeSLOT*_...).
    # Used for BLOCKER ATTRIBUTION in the decluttering search -- which loose part is in the
    # way -- so it can be relocated rather than the whole step abandoned.
    C_col = self._scene_for(state, assembly, extra_grasps)
    return [(a, b, d) for a, b, d in C_col.getCollisions()
            if d < -_CHEAP_PENETRATION_TOL and not _wrist_table_pair(a, b)]

  def collision_free(self, state, assembly, note=""):
    return self._check(state, assembly, note=note)

  def pick_collision_free(self, s0, gpart, grasp_id, assembly, note=""):
    # cheap gate for PICK edges: place a floating EE at the candidate grasp with the part
    # at its CURRENT (pre-pick, on-table) pose, in the actual environment. This is what
    # the precomputed grasp table does NOT see -- it checks a grasp only against its own
    # subassembly (no table, no neighbours), so e.g. a two-finger side grasp whose lower
    # finger dips below the cube (into the table) passes precompute yet can never be
    # picked. Rejecting it here costs ~2ms instead of a full (restarted) IK solve.
    return self._check(s0, assembly, extra_grasps=[(gpart, grasp_id)], note=note)


def _wrist_table_pair(a, b):
  # wrist-capsule-vs-TABLE grazes are the soft-IK-margin class the wrist radius was shaved
  # for -- yet two-finger SIDE grasps still put the wrist within ~10mm of the table, and
  # the real (soft) arm IK ACCEPTS those (measured: 8/24 edges of a verified plan were
  # wrongly rejected on exactly this pair). The wrist stand-in exists to catch wrist-vs-
  # CLUTTER; table-margin reachability is not soundly checkable at the EE level, so this
  # pair never rejects.
  return ("table" in (a, b)) and (a.endswith("ur_wrist_coll") or b.endswith("ur_wrist_coll"))


_CHEAP_DEBUG_VIEW = False


def set_cheap_check_debug(on=True):
  # step-through debugging of the cheap layer: every UNCACHED CollisionChecker query pops
  # C.view(True, ...) with verdict + triggering action + penetrating pairs. First-run cache
  # semantics mean each distinct scene appears exactly once. Use with tiny instances --
  # every check blocks on the viewer window.
  global _CHEAP_DEBUG_VIEW
  _CHEAP_DEBUG_VIEW = bool(on)


class IKFailureLogDone(Exception):
  # raised by RobotIK._log_failure (only when fail_log_stop is set) once the failure-log
  # window or cap is reached: in debug mode the run's PURPOSE is the failure browser, so
  # the search aborts instead of grinding on with logging silenced.
  pass


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
    self.robot_base = {}                               # subtree root, for single-arm IK
    self.robot_arm = {}                                # 'ur5' | 'ur10' -- picks the analytic solver
    self.base_link = {}                                # URDF base frame (ssik's reference)
    self.ik_seeding = False                            # warm-start KOMO with analytic (ssik) IK
    names = set(self.C.getFrameNames())
    for r in self.robots:
      tool_center, _base_rel, q_corr = _gripper_rest_transforms(self.C, r + "_")
      self.gripper_frame[r] = tool_center
      self.q_correction[r] = q_corr
      # the vacuum is rotationally symmetric about its suction axis -> roll is a free DOF,
      # so we constrain only position + approach axis (5-DOF). the two-finger's fingers
      # break that symmetry -> full pose (6-DOF).
      self.symmetric[r] = "vacuum" in tool_center
      self.robot_base[r] = r + "_ur_base"              # UR base frame; parented to the table
      self.robot_arm[r] = "ur5" if (r + "_ur_arm_ur5") in names else "ur10"
      self.base_link[r] = r + "_ur_base_link"          # ssik solves in base_link's frame
    # SINGLE-ACTIVE-ARM IK optimisations (pick/place move only one arm). Both default ON;
    # flip either OFF for debugging (they change the IK scene, not the plan semantics):
    #   single_arm_ik           : restrict the KOMO to the active arm's joints (others frozen)
    #   strip_inactive_collision: drop the other arms' collision geometry (fewer contact pairs)
    self.single_arm_ik = True
    self.strip_inactive_collision = True
    # REACH PRECHECK: before the full (collision-aware) solve, solve the SAME problem WITHOUT
    # the collision objective. Collisions only SHRINK the feasible set, so a TRULY infeasible
    # relaxation implies infeasibility -- but the relaxed solver is stochastic-incomplete, so
    # "relaxed solve failed" is only evidence, not proof (measured: 4/97 binds where 13
    # relaxed attempts missed a solution the FULL solve found -- the collision gradient can
    # guide the solver out of a bad basin). Hence the RESIDUAL GATE: reject instantly only
    # when the relaxed solve failed AND missed by a lot (eq > precheck_eq_min: a clear
    # can't-reach, measured false-rejects sat at eq 0.8-2.5 vs true catches median 5.6);
    # borderline relaxed failures fall through to the full solve. No warm-starting.
    self.reach_precheck = True
    self.precheck_eq_min = 3.0                            # residual gate (see above)
    self.precheck_rejects = 0                             # rejects caught by the precheck
    self._ik_cache = {}                                   # (action, scene sig) -> (ok, witness)
    self.fail_recorder = None                             # viser recorder for failed binds
    self.fail_log_cap = 300
    self.fail_log_seconds = 60.0                          # window from the FIRST failure
    self.fail_log_stop = False                            # abort the search when done?
    self._fail_log_t0 = None

  def holders(self, state):
    # {robot: (grasped_part, grasp_id)} -- who holds what, at which grasp, in this state.
    return {c[1]: (c[2], c[3]) for p in state.bodies()
            for c in state.supports.get(p, ()) if c[0] == "grasp"}

  def ee_target(self, robot, part, grasp_id, assembly):
    # the gripper's target pose relative to its grasped part: the part's generated grasp,
    # approach-corrected for THIS robot's EE.
    grasp = grasp_pose(assembly, part, grasp_id)
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
      tgt = self.ee_target(r, pt, gid, assembly)
      komo.addObjective([1, 1], ry.FS.positionRel, [self.gripper_frame[r], pt],
                        ry.OT.eq, [1e1], tgt[:3])
      komo.addObjective([1, 1], ry.FS.quaternionRel, [self.gripper_frame[r], pt],
                        ry.OT.eq, [1e1], tgt[3:])

  def _disable_inactive_collision(self, C, active_robot):
    # drop the collision geometry of every robot EXCEPT `active_robot` (setContact(0) on their
    # arm frames). Collision evaluation over all pairs is the pick-IK bottleneck (not DOF
    # count), so removing the inactive arms' links is the real speed lever -- sound where they
    # cannot block the active arm (verified 0 verdict change on cross). Held PARTS keep their
    # collision (only the arm links are dropped).
    for r in self.robots:
      if r == active_robot:
        continue
      pre = r + "_"
      for nm in C.getFrameNames():
        if nm.startswith(pre):
          C.getFrame(nm).setContact(0)

  def _restrict_to_arm(self, C, robot):
    # SINGLE-ARM IK: restrict the config's active joints to just `robot`'s subtree, so the
    # KOMO only optimizes that one arm and every other robot is a FROZEN obstacle (at home).
    # Cuts the IK from R*6 DOFs to 6 -> much faster; valid where exactly one arm moves (pick).
    # Returns the FULL joint-name list so the restricted solution can be re-expanded to the
    # full witness vector the replay expects (see _expand_witness). NOTE: brace grasp-
    # maintenance must be SKIPPED under this restriction (a frozen brace arm cannot satisfy
    # its eq objective -> spurious infeasibility).
    full_names = C.getJointNames()
    C.selectJointsBySubtree(C.getFrame(self.robot_base[robot]))
    return full_names

  def _expand_witness(self, C_restricted, w, full_names):
    # map a single-arm solution `w` (ordered like the restricted config's joints) back to the
    # full joint vector the replay/_write_robot_dofs expect. Non-active joints keep home
    # (harmless: replay only reads the moved arm's DOFs out of the witness).
    if w is None:
      return None
    idx = {nm: i for i, nm in enumerate(full_names)}
    full = np.array(self.home, dtype=float).copy()
    for j, nm in enumerate(C_restricted.getJointNames()):
      full[idx[nm]] = w[j]
    return full

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
    grasp = grasp_pose(assembly, gpart, grasp_id)
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
    if kind == "release":
      return True, None                                   # no new reach
    # VERDICT + WITNESS CACHE on a discrete scene signature: the search re-probes the SAME
    # (action, scene) from many parent states, and a FAILED solve costs ~135ms (initial +
    # 12 restarts) vs ~5ms for a success -- measured 81 re-probed pick failures per 40
    # expansions on cross7-R3. Unlike the cheap checker's signature this one KEEPS robot
    # identities (fixed bases: who reaches matters) and the grasp contacts' robots. Caching
    # also makes the stochastic restarts deterministic within a solve (same query -> same
    # verdict), which the search prefers anyway.
    key = (action, self._scene_sig(s0))
    hit = self._ik_cache.get(key)
    if hit is not None:
      return hit
    if kind == "pick":
      res = self._ik_pick(action, s0, s1, assembly, seed)
    elif kind == "place":
      res = self._ik_place(action, s0, s1, assembly, seed)
    elif kind == "assemble":
      res = self._ik_assemble(action, s0, s1, assembly, seed)
    else:                                                 # add_grasp / handover: 2nd arm
      res = self._ik_add_grasp(action, s0, s1, assembly, seed)
    self._ik_cache[key] = res
    return res

  def _scene_sig(self, state):
    # discrete signature of everything the per-skill IK scenes depend on: per subassembly
    # its members, table placement + rounded anchor (or held-flag), and grasp contacts WITH
    # robot identities. Mirrors CollisionChecker._sig but keeps robots (fixed bases).
    parts = []
    for group in state.subassemblies():
      members = tuple(sorted(group))
      grasps = tuple(sorted((c[1], c[2], c[3]) for p in group
                            for c in state.supports.get(p, ()) if c[0] == "grasp"))
      on_table = any(("table",) in state.supports.get(p, ()) for p in group)
      if on_table:
        ref = reference_part(group)
        anchor = tuple(np.round(np.asarray(state.poses[ref], dtype=float), 4))
        parts.append(("T", members, state.place.get(ref), anchor, grasps))
      else:
        parts.append(("H", members, grasps))
    return tuple(sorted(parts))

  def _holders_of_group(self, state, group):
    # robots grasping any part of `group`
    return [r for r, (pt, _gid) in self.holders(state).items() if pt in group]

  def _solve_ik(self, komo, restarts=12, seeds=None):
    # solve a skill KOMO, retrying from fresh inits until one converges. KOMO seeds its guess
    # with a little noise, so a borderline-reachable target flips feasible/infeasible run-to-
    # run; restarts make bind reliable -- a genuinely feasible action returns a witness (instead
    # of sometimes None, which would corrupt search + replay). Returns (feasible, witness joint
    # config or None). `seeds` (analytic ssik solutions, exact for the pose) are tried FIRST as
    # warm starts -- helps CONVERGENCE-limited reaches; note it does NOT help collision-limited
    # ones (an exact analytic pose may collide, whereas the KOMO's tolerant pose eq can settle on
    # a nearby collision-free pose the discrete IK solutions don't cover), hence the fallback.
    for seed in (seeds or []):
      komo.initWithConstant(np.asarray(seed, dtype=float))
      r = ry.NLP_Solver(komo.nlp(), verbose=0).solve().dict()
      if r["feasible"]:
        self._last_res = r
        return True, komo.getPath()[-1]
    r = ry.NLP_Solver(komo.nlp(), verbose=0).solve().dict()
    if r["feasible"]:
      self._last_res = r
      return True, komo.getPath()[-1]
    for _ in range(restarts):
      komo.initRandom()
      r = ry.NLP_Solver(komo.nlp(), verbose=0).solve().dict()
      if r["feasible"]:
        self._last_res = r
        return True, komo.getPath()[-1]
    self._last_res = r
    return False, None

  def _analytic_reject_ok(self, robot):
    # may we use analytic IK for an EXACT reachability REJECT on a FIXED-target reach? Only when
    # seeding is on, ssik has this arm calibrated, and the grasp is a full 6-DOF pose (two-finger):
    # then an empty solution set means KINEMATICALLY UNREACHABLE -> reject with no KOMO. Excluded:
    # the vacuum (5-DOF, free roll -> one roll's empty set is NOT "unreachable"), and uncalibrated
    # arms / absent ssik. (Only the fixed on-table pick uses this; a held add_grasp target can move.)
    if not self.ik_seeding or self.symmetric[robot]:
      return False
    import analytic_ik
    return analytic_ik.available(self.robot_arm[robot])

  def _analytic_seeds(self, C, robot, ee_target, target_frame):
    # analytic (ssik) joint solutions for the active arm to place its tool center at `ee_target`
    # (a 7-pose relative to `target_frame`; target_frame=None => ee_target is already a WORLD
    # pose, e.g. a place's computed gripper pose at the group's landing spot), in RAI joint
    # order, filtered to rai's arm limits. Empty unless ik_seeding is on AND this arm is
    # calibrated. Call AFTER _restrict_to_arm so C.getJointLimits() returns the 6 arm limits.
    import analytic_ik
    if not self.ik_seeding or not analytic_ik.available(self.robot_arm[robot]):
      return []
    if target_frame is None:
      ee_world = np.asarray(ee_target, dtype=float)
    else:
      tpose = np.asarray(C.getFrame(target_frame).getPose(), dtype=float)
      ee_world = _pose_compose(tpose, np.asarray(ee_target, dtype=float))  # tool target in world
    bl = np.asarray(C.getFrame(self.base_link[robot]).getPose(), dtype=float)
    rel = _pose_compose(_pose_invert(bl), ee_world)                       # tool target in base_link
    T = np.eye(4)
    T[:3, 3] = rel[:3]
    T[:3, :3] = _quat_to_R(rel[3:])
    try:
      lo, hi = C.getJointLimits()
      lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    except Exception:
      lo = hi = None
    out = []
    for q in analytic_ik.seed_configs(self.robot_arm[robot], T):
      if lo is not None and len(q) == len(lo) and (np.any(q < lo) or np.any(q > hi)):
        continue                                            # outside rai's joint limits
      out.append(q)
    return out

  def _reach_precheck_reject(self, action, C, komo_nc):
    # run the RELAXED (collision-free) solve; True => clear reach failure, reject the bind
    # without running the full solve. Residual-gated (see the reach_precheck comment in
    # __init__): a failed relaxed solve rejects only when it missed by a lot (eq >
    # precheck_eq_min); borderline failures fall through to the full solve, which decides.
    ok0, _ = self._solve_ik(komo_nc)
    if ok0:
      return False
    if (getattr(self, "_last_res", {}) or {}).get("eq", 0.0) <= self.precheck_eq_min:
      return False                                        # borderline: let the full solve decide
    self.precheck_rejects += 1
    # NOT logged to the failure recorder: a precheck reject is a CLEAR reach failure (arm can't
    # get near the pose) -- the failure log is for inspecting the IK-ITSELF failures (reachable
    # pose, but the full collision solve found no config), which is what is diagnostically useful.
    return True

  def _log_failure(self, action, C, komo):
    # IK-FAILURE INSTRUMENTATION (opt-in via self.fail_recorder = a viser_view.ViserRecorder;
    # None = zero cost): pose the skill's OWN scene at the LAST (failed) attempt and record
    # it as one browsable viser step, labelled with the action and the solver residuals --
    # so failed binds of the forward search can be inspected by eye. Bounded BOTH by entry
    # count (fail_log_cap) and a TIME WINDOW (fail_log_seconds, measured from the FIRST
    # failure so the heuristic build does not eat it) -- long runs would otherwise collect
    # useless thousands.
    import time as _time
    if self.fail_recorder is None:
      return
    done = len(self.fail_recorder.steps) >= self.fail_log_cap
    if not done and self._fail_log_t0 is not None:
      done = _time.perf_counter() - self._fail_log_t0 > self.fail_log_seconds
    if done:
      if self.fail_log_stop:
        raise IKFailureLogDone()
      return
    if self._fail_log_t0 is None:
      self._fail_log_t0 = _time.perf_counter()
    try:
      C.setJointState(komo.getPath()[-1])                  # best (infeasible) attempt
    except Exception:
      pass
    res = getattr(self, "_last_res", {}) or {}
    label = (f"FAIL {action}  \n"
             f"eq={res.get('eq', float('nan')):.3f} ineq={res.get('ineq', float('nan')):.3f} "
             f"sos={res.get('sos', float('nan')):.3f}")
    self.fail_recorder.record(C, label)

  def _ik_pick(self, action, s0, s1, assembly, seed):
    # simple IK: ONE robot (the picker) reaches a fixed grasp pose on an on-table part.
    # Every other robot -- free or holding something else -- stays at home (an obstacle).
    #
    # SINGLE ACTIVE ARM: a pick moves ONLY the picker, so we restrict the IK to the picker's
    # subtree (selectJointsBySubtree) -- the other arms become frozen obstacles instead of
    # co-optimized DOFs, cutting the problem from R*6 to 6 DOFs. (This also means brace grasp-
    # maintenance is skipped: a frozen brace arm can't satisfy it.)
    #
    # TODO(future): this per-node KOMO is overkill for a fixed grasp on a fixed on-table part
    # -- it is a plain 6-DOF reach with a known target, so an ANALYTICAL IK (e.g. ssik) would
    # give the joint solution directly (and far faster) instead of a nonlinear solve+restarts.
    _, robot, part, grasp_id = action
    C = self.scene_for(s0, assembly)                      # part sits on the table (fixed target)
    grasp = grasp_pose(assembly, part, grasp_id)          # gripper pose relative to `part`
    # align the grasp's +z approach with THIS EE's real approach axis (identity for the
    # two-finger, a rotation for the vacuum) -- so the tool-center is oriented correctly.
    ee_target = np.concatenate([grasp[:3], _quat_mul(grasp[3:], self.q_correction[robot])])

    robot_ee_name = self.gripper_frame[robot]
    part_name = part

    # the gripper is SUPPOSED to touch the part it grasps (two-finger palm/fingers close
    # around it), so exclude the target part from collision -- otherwise the collision
    # objective forbids the grasp itself.
    set_part_contact(C, part_name, 0)

    if self.strip_inactive_collision:
      self._disable_inactive_collision(C, robot)          # drop other arms' contact pairs
    full_names = self._restrict_to_arm(C, robot) if self.single_arm_ik else None

    def build(collisions):
      komo = ry.KOMO(
          C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=collisions
      )
      if collisions:
        # no scene-penetration: without this the arm reaches the grasp by driving its links
        # THROUGH the table and the other cubes -- the tool center looks right but the pose
        # is physically bogus, and grasps that are actually unreachable wrongly pass. (the
        # target part is excluded above, since the gripper is supposed to touch it.)
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
      if not self.single_arm_ik:
        self._add_grasp_maintenance(komo, s0, assembly)   # braces are DOFs only in full mode
      return komo

    # analytic (ssik) IK for the fixed grasp pose: EXACT and instant. When calibrated (two-finger),
    # an empty solution set means the pose is kinematically unreachable -> reject with no KOMO
    # (fail fast, replacing the slower + convergence-fallible relaxed reach precheck); a non-empty
    # set proves reachability -> skip the precheck and warm-start the collision solve below. Vacuum
    # / uncalibrated / no-ssik fall back to the relaxed precheck.
    seeds = self._analytic_seeds(C, robot, ee_target, part_name)
    if self._analytic_reject_ok(robot):
      if not seeds:
        return False, None                                # analytic: kinematically unreachable
    elif self.reach_precheck and self._reach_precheck_reject(action, C, build(False)):
      return False, None                                  # clear reach failure (relaxed solve)

    komo = build(True)
    feasible, witness = self._solve_ik(komo, seeds=seeds)
    if self.single_arm_ik:
      witness = self._expand_witness(C, witness, full_names)  # restricted -> full joint vector
    if not feasible:
      self._log_failure(action, C, komo)

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
    grasp = grasp_pose(assembly, gpart, grasp_id)
    ee_target = np.concatenate([grasp[:3], _quat_mul(grasp[3:], self.q_correction[robot])])
    set_part_contact(C, gpart, 0)                        # the new gripper is meant to touch gpart

    def build(collisions):
      komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=collisions)
      if collisions:
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1], [-0.0])
      komo.addControlObjective([], 0, 1e-1)
      komo.addControlObjective([], 1, 1e-1)
      komo.addObjective([1, 1], ry.FS.positionRel, [self.gripper_frame[robot], gpart],
                        ry.OT.eq, [1e1], ee_target[:3])
      komo.addObjective([1, 1], ry.FS.quaternionRel, [self.gripper_frame[robot], gpart],
                        ry.OT.eq, [1e1], ee_target[3:])
      self._add_grasp_maintenance(komo, s0, assembly)     # existing braces stay at their grasps
      return komo

    if self.reach_precheck and self._reach_precheck_reject(action, C, build(False)):
      return False, None                                  # clear reach failure (relaxed solve)

    komo = build(True)
    feasible, witness = self._solve_ik(komo)
    if not feasible:
      self._log_failure(action, C, komo)

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

    # SINGLE ACTIVE ARM: like pick, a place moves only the placer -- BUT only when the placer
    # is the group's SOLE holder. If another robot BRACES the group, that arm must co-move as
    # the group lands, so the optimisations don't apply and we fall back to the full KOMO.
    # (For a solo place _add_grasp_maintenance is already a no-op, so nothing is lost.)
    gparts = set(group)
    solo = not any(rr != robot and pt in gparts
                   for rr, (pt, _g) in self.holders(s0).items())
    if self.strip_inactive_collision and solo:
      self._disable_inactive_collision(C, robot)          # drop other arms' contact pairs
    single = self.single_arm_ik and solo
    full_names = self._restrict_to_arm(C, robot) if single else None

    def build(collisions):
      komo = ry.KOMO(
          C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=collisions
      )
      if collisions:
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
      if not single:
        self._add_grasp_maintenance(komo, s0, assembly)   # braces ride along / stay pinned
      return komo

    # analytic (ssik) IK: the place target is FULLY DETERMINED for the placer's tool too --
    # the group lands at `target` and is RIGIDLY attached to the gripper, so the tool's world
    # pose is target(ref) o (ref -> gripper), both known. Same contract as pick: an empty
    # analytic solution set on a calibrated two-finger arm proves the pose kinematically
    # unreachable -> reject with NO KOMO (replacing the relaxed reach precheck, the dominant
    # cost of place binds); non-empty proves reachability -> skip the precheck and warm-start
    # the collision solve. Braced (non-solo) places keep the old path: the KOMO then spans
    # several arms, so a single-arm seed doesn't fit its joint vector.
    seeds = []
    if single:
      gpose = np.asarray(C.getFrame(self.gripper_frame[robot]).getPose(), dtype=float)
      rpose = np.asarray(C.getFrame(ref).getPose(), dtype=float)
      ee_world = _pose_compose(target, _pose_compose(_pose_invert(rpose), gpose))
      seeds = self._analytic_seeds(C, robot, ee_world, None)
    if single and self._analytic_reject_ok(robot):
      if not seeds:
        return False, None                                # analytic: kinematically unreachable
    elif self.reach_precheck and self._reach_precheck_reject(action, C, build(False)):
      return False, None                                  # clear reach failure (relaxed solve)

    komo = build(True)
    feasible, witness = self._solve_ik(komo, seeds=seeds)
    if single:
      witness = self._expand_witness(C, witness, full_names)  # restricted -> full joint vector
    if not feasible:
      self._log_failure(action, C, komo)

    if self.debug_view:
        print("place feasible:", feasible)
        komo.view(True, f"place IK: {robot} sets down {root} (ref {ref})")

    return feasible, witness

  def _assemble_komo(self, C, part_name, onto, rel_pos, rel_quat, bias_axes,
                     collisions=True):
    # build the assemble KOMO for one orientation bias. Shared by every step of the
    # orientation sweep (only the soft bias objective changes). collisions=False builds the
    # RELAXED problem for the reach precheck.
    komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=collisions)
    if collisions:
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
    # solve for the added part's mate pose: the SEAM (`part`) must reach its mate pose
    # relative to the target subassembly. `part` may be a single cube OR the seam of a whole
    # held MODULE (a plate) -- in the latter case the module is rigid in the scene, so the
    # one seam->onto constraint positions the entire module (obj_5..7 ride along). Active =
    # the adder + whoever holds the target (if held). If the target is on the table it is a
    # fixed target and only the adder is active (others held near home by the control cost).
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

    # the mate seats the moving group flush against the target, so those contacts are intended
    # -- exclude the WHOLE moving group from collision (a single cube, or every cube of a
    # module: a stacked plate touches the target plate on all its cubes). Otherwise the
    # intended flush seating reads as penetration and IK fails. Arm-vs-rest collision stays on;
    # the cheap floating-EE check already covered the moved geometry vs the scene.
    moving = next((g for g in s0.subassemblies() if part in g), (part,))
    for mp in moving:
      set_part_contact(C, mp, 0)

    if self.reach_precheck:
      # relaxed (collision-free) reach: can the mate pose be reached AT ALL? A clear relaxed
      # failure rejects instantly -- skipping the whole bias sweep + restart fallback (the
      # most expensive failure path there is). Residual-gated like the other skills.
      komo_nc = self._assemble_komo(C, part_name, onto, rel_pos, rel_quat, None,
                                    collisions=False)
      self._add_grasp_maintenance(komo_nc, s0, assembly)
      if self._reach_precheck_reject(action, C, komo_nc):
        return False, None

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
    if not feasible:
      self._log_failure(action, C, komo)

    if self.debug_view:
        print("assemble feasible:", feasible)
        komo.view(True, f"assemble IK: {robot} mates {part} -> {onto}")

    return feasible, witness


# actions that move a part into a possibly-colliding pose (so they need the CHEAP geometry
# check on the RESULT). release does not move any part, so its geometry is unchanged.
_GEOMETRIC_ACTIONS = {"assemble", "place", "set_down", "stow"}

# geometric actions that OPEN the moving robot's grasp on arrival: the result state has no
# EE on the moved group anymore, so checking s1 alone misses the gripper at the moment of
# set-down. These get the opening grasp re-added as an extra EE (part at its PLACED pose --
# exactly the configuration the place IK solves). assemble is NOT here: it keeps the grasp
# (brace), so its EE is already in s1.
_OPENING_ACTIONS = {"place", "set_down", "stow"}


def _opening_grasp(action, s0):
  # the (gpart, grasp_id) that action's robot is about to open, read from s0.
  r, root = action[1], action[2]
  group = next((g for g in s0.subassemblies() if root in g), (root,))
  for p in group:
    for c in s0.supports.get(p, ()):
      if c[0] == "grasp" and c[1] == r:
        return (c[2], c[3])
  return None

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
  if action[0] in _GEOMETRIC_ACTIONS:
    extra = ()
    if action[0] in _OPENING_ACTIONS:
      g = _opening_grasp(action, s0)                       # EE at the about-to-open grasp,
      extra = (g,) if g is not None else ()                # part at its PLACED pose
    if not checker._check(s1, assembly, extra_grasps=extra, note=f"result of {action}"):
      return False
  if action[0] in _GRASP_ACTIONS and not checker.pick_collision_free(
      s0, action[2], action[3], assembly, note=f"grasp candidate of {action}"):
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


class Plan(list):
  """The action list PLUS `.states` = the exact state trajectory [s0, s1, ..., sN] the search
  visited (len == len(plan)+1). Behaves as a plain list everywhere (iteration, len, indexing,
  JSON via list()), but lets an in-session replay WALK the recorded states instead of RE-DERIVING
  each successor -- re-derivation must otherwise reproduce the search's exact successor flags
  (grasp table, bracing, ...) and has repeatedly drifted (the vacuum-table and add_grasp replay
  bugs). `.states` is None when a plan is built some other way (e.g. loaded from disk)."""
  states = None


def _reconstruct_plan(came_from, goal):
  # walk parent links back to the start; return the actions start -> goal, plus the state
  # trajectory (attached to the returned Plan as `.states`).
  actions, states = [], [goal]
  s = goal
  while came_from[s][0] is not None:
    prev, action = came_from[s]
    actions.append(action)
    states.append(prev)
    s = prev
  plan = Plan(reversed(actions))
  plan.states = list(reversed(states))
  return plan


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
                          for f in rest_faces(assembly, _mate_closure(assembly, p) or {base}))
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


class LabeledDisassemblyHeuristic:
  """GRASP-LABELED backward-disassembly abstraction (the tractability spike).

  Same idea as AbstractionHeuristic -- forget, solve the abstract problem exactly backward
  from the goal, look distances up at search time -- but the abstract state KEEPS the labels
  cheap geometry depends on:

      (ideal, G, face|H, L)
        ideal : frozenset of assembled parts
        G     : grasps ON the assembly, {(part, gid)}   (holders + braces; robots anonymous)
        face|H: the assembly's placement face, or 'H' = held in the air
        L     : held LOOSE parts with their grasps, {(part, gid)}

  Geometry enters through three fact families, all cheap-layer and cached:
    * grasp validity per ideal -- straight from the precomputed grasp table. MONOTONE:
      valid on a larger assembly => valid on any sub-ideal (removing parts removes
      obstacles), which keeps every inverse action closed under the invariants for free.
    * pick facts (LAZY): grasp (p,g) additionally collision-free vs the TABLE with the
      subassembly resting on face f (the finger-into-table cases the grasp table misses).
    * mate facts: the existing per-(part, face) _on_table_mateable table.

  The abstract actions mirror the real primitives one-for-one (pick/lift/assemble/release/
  place/fused-handover at the real costs; every gate is a RELAXATION of the real one), so
  distances are ADMISSIBLE. Solved by BACKWARD Dijkstra (handover costs 2) from all goal
  states via INVERSE actions -- only CO-REACHABLE states are enumerated, which is exactly
  the tractability question. A projection missing from the table falls back to the caller's
  counting heuristic (self.misses counts how often)."""

  H = "H"                                                  # face value for held-in-air

  def __init__(self, assembly, n_robots, grasp_table, gripper_path=None, verbose=False,
               max_states=2_000_000, clearance=0.02, gripper_kind=None, bracing=False):
    import heapq
    import time as _time

    t0 = _time.perf_counter()
    self.assembly = assembly
    self.R = n_robots
    self.misses = 0
    self._bracing = bracing                                # model the add_grasp (co-grasp) edge
    self._clearance = clearance                            # see _table_aware_pick_fact
    self._gk = gripper_kind                                # non-None => WRENCH facts active
    objs = list(assembly.objects)
    base = assembly.base
    # STATES ARE SORTED TUPLES (ideal, G, face, L) -- tuples hash ~2x faster than
    # frozensets, and the Dijkstra loop is hash-bound (py-spy: ~36% of the build).
    all_parts = _subasm_key(objs)
    self._closure = {p: frozenset(_mate_closure(assembly, p)) for p in objs}

    def tadd(t, x):
      return tuple(sorted(t + (x,)))

    def tsub(t, x):
      i = t.index(x)
      return t[:i] + t[i + 1:]

    # grasp validity per buildable ideal (+ singletons), from the precomputed table
    ideal_keys = [_subasm_key(s) for s in get_subassemblies(assembly) if base in s]
    self.valid = {S: tuple(sorted((p, g) for p, g, _ in grasp_table[S]))
                  for S in ideal_keys}
    self.single = {q: tuple(sorted(g for _p, g, _ in grasp_table[(q,)])) for q in objs}

    # lazy, cached table-aware pick fact -- keyed PER PART deliberately (no identical-parts
    # aliasing: per-part shapes/meshes and forces are coming; the template reuse makes the
    # extra per-part queries cheap anyway).
    self._ee_path = gripper_path
    self._pick_cache = {}

    def pick_ok(S_key, face, pg):
      key = (S_key, face, pg)
      hit = self._pick_cache.get(key)
      if hit is None:
        hit = self._pick_cache[key] = _table_aware_pick_fact(
            assembly, S_key, face, pg[0], pg[1], self._ee_path,
            clearance=self._clearance)
      return hit

    # placement faces are now looked up PER SUBASSEMBLY in the stable rest-pose table
    # (rest_faces) instead of the old global 0..5 -- so the abstraction offers exactly the
    # resting orientations the real places()/precompute_stable_faces do (e.g. a lone cylinder
    # only cap-down). Same faces on both sides => still a relaxation => still admissible.
    rf = lambda g: rest_faces(self.assembly, set(g))
    Hf = self.H

    # ---- WRENCH facts (gripper_kind set): gate the abstract mates/releases with the SAME
    # `_group_stable` LP the real gates run, on CRAFTED states -- since the labeled state
    # knows the grasps, these facts are EXACT for their (ideal, G, face) tuple, not a
    # relaxation. Without them the heuristic walks into wrench-deleted edges and the search
    # degenerates back to a plateau exactly on wrench-heavy configs (measured).
    flatposes = _flat_poses(assembly)
    self._wfacts = {}

    def crafted(ideal_key, G, face):
      # a minimal real State for the LP: ideal parts mate-parented under the base; base on
      # the table at `face` (or held when face==Hf); grasps G as support contacts. The
      # kinematic root choice is irrelevant to _group_stable (it reads supports + layout).
      parent, place, supports = {}, {}, {}
      ideal_set = set(ideal_key)
      for q in assembly.objects:
        if q in ideal_set:
          if q == assembly.base:
            parent[q] = ("gripper", "zz0") if face is Hf else ("table",)
            supports[q] = frozenset() if face is Hf else frozenset({("table",)})
            if face is not Hf:
              place[q] = ("slot", 0, face)
          else:
            onto = assembly.relative_pose(q)[0]
            parent[q] = ("part", onto)
            supports[q] = frozenset({("mate", onto)})
        else:
          parent[q] = ("table",)
          supports[q] = frozenset({("table",)})
          place[q] = ("slot", _part_index(q), _IDENTITY_FACE)
      for i, (p, g) in enumerate(G):
        supports[p] = supports[p] | {("grasp", f"zz{i}", p, g)}
      return State(parent=parent, place=place, supports=supports, poses=dict(flatposes))

    def held_single(q, gq):
      st = crafted((), (), _IDENTITY_FACE)                 # everything loose on the table
      st.parent[q] = ("gripper", "zzq")
      st.place.pop(q, None)
      st.supports[q] = frozenset({("grasp", "zzq", q, gq)})
      return st

    def w_hold_ok(ideal_key, G, face):
      # can supports G (+ table if resting) hold the assembly under gravity? (real release
      # gate / assemble's target-supported gate)
      if self._gk is None:
        return True
      key = ("H", ideal_key, G, face)
      hit = self._wfacts.get(key)
      if hit is None:
        st = crafted(ideal_key, G, face)
        hit = self._wfacts[key] = _group_stable(st, assembly, set(ideal_key), self._gk)
      return hit

    def w_assemble_ok(ideal_s_key, G2, face, q, gq):
      # the real assemble wrench gates: target (with braces G2, at `face` or held) must be
      # stable under gravity AND under the insertion reaction of q; the inserter's grasp gq
      # must transmit the press (mirrors PARAMS['check_inserter']).
      if self._gk is None:
        return True
      key = ("A", ideal_s_key, G2, face, q, gq)
      hit = self._wfacts.get(key)
      if hit is None:
        onto = assembly.relative_pose(q)[0]
        ok = w_hold_ok(ideal_s_key, G2, face)
        if ok and wrench.PARAMS.get("check_inserter", True):
          st_q = held_single(q, gq)
          ok = _group_stable_loaded(st_q, assembly, frozenset({q}), self._gk,
                                    _inserter_load_variants(st_q, assembly, q, onto))
        if ok:
          st_t = crafted(ideal_s_key, G2, face)
          ok = _group_stable_loaded(st_t, assembly, set(ideal_s_key), self._gk,
                                    _insertion_load_variants(st_t, assembly, q, onto))
        hit = self._wfacts[key] = ok
      return hit

    def preds(t):
      ideal, G, face, L = t
      cap_used = len(G) + len(L)
      # inverse ASSEMBLE: some (q,gq) in G was the inserter's brace; q maximal in ideal
      for q, gq in G:
        if q == base:
          continue
        if any(q in self._closure[r] for r in ideal if r != q):
          continue                                        # not maximal -> not last-added
        if face is not Hf and not _on_table_mateable_cached(self.assembly, q, face):
          continue
        G2 = tsub(G, (q, gq))
        if any(p == q for p, _ in G2):
          continue                                        # q was loose then: single-held only
        if face is Hf and len(G2) < 1:
          continue                                        # target had to be supported
        if not w_assemble_ok(tsub(ideal, q), G2, face, q, gq):
          continue                                        # wrench: press not transmittable
        yield (tsub(ideal, q), G2, face, tadd(L, (q, gq))), 1.0
      # inverse PICK-LOOSE: some (q,gq) in L was just picked off the table. The abstraction
      # does not track loose parts' resting faces (they can be reoriented via place+pick),
      # so the gate is EXISTS-a-face -- optimistic about the reorientation cost, admissible.
      for q, gq in L:
        if any(pick_ok((q,), f, (q, gq)) for f in rf((q,))):
          yield (ideal, G, face, tsub(L, (q, gq))), 1.0
      # inverse LIFT: held solo -> was resting on some face and got picked
      if face is Hf and len(G) == 1:
        pg = G[0]
        for f in rf(ideal):
          if pick_ok(ideal, f, pg):
            yield (ideal, (), f, L), 1.0
      # inverse PLACE-ASSEMBLY: on table now -> was held by one more grasp in the air
      if face is not Hf and cap_used < self.R:
        for pg in self.valid[ideal]:
          if pg not in G:
            yield (ideal, tadd(G, pg), Hf, L), 1.0
      # inverse RELEASE: some grasp was just opened (it existed in the pre-state). The real
      # release is wrench-gated on the POST-state (t) staying supported.
      if cap_used < self.R and w_hold_ok(ideal, G, face):
        for pg in self.valid[ideal]:
          if pg not in G:
            yield (ideal, tadd(G, pg), face, L), 1.0
      # inverse ADD_GRASP (bracing only): a brace was just added to the already-held assembly
      # (add_grasp KEEPS both holders, unlike handover's swap). Backward: the pre-state held it
      # with one FEWER grasp -- still held (>=1) and stable under gravity. This is the edge that
      # lets the abstraction FORM a >=2-grasp brace, required for offset horizontal mates (e.g.
      # car wheels: a single grasp cannot supply the yaw-cancelling couple). Without it the
      # abstraction stays admissible only when bracing is OFF; with bracing on it would wrongly
      # report inf (see [[cograsp-gap]]). Extra edges only relax the bound -> admissibility safe.
      if self._bracing:
        for pg in G:
          Gs = tsub(G, pg)
          if len(Gs) >= 1 and w_hold_ok(ideal, Gs, face):
            yield (ideal, Gs, face, L), 1.0
      # inverse fused HANDOVER (cost 2): solo-held; the grasp was swapped
      if face is Hf and len(G) == 1 and len(L) <= self.R - 2:
        pg_new = G[0]
        for pg_old in self.valid[ideal]:
          if pg_old != pg_new:
            yield (ideal, (pg_old,), Hf, L), 2.0
      # inverse PLACE-LOOSE: a loose part now on the table was held just before
      if cap_used < self.R:
        for q in objs:
          if q in ideal or any(q == p for p, _ in L):
            continue
          for qg in self.single[q]:
            yield (ideal, G, face, tadd(L, (q, qg))), 1.0

    # goal states: everything assembled, any supported holder configuration, nothing loose
    self.dist = {}
    pq = []
    from itertools import combinations
    vg_full = self.valid[all_parts]
    for k in range(self.R + 1):
      for combo in combinations(vg_full, k):
        for face in [Hf] + list(rf(all_parts)):
          if face is Hf and len(combo) < 1:
            continue
          st = (all_parts, combo, face, ())
          self.dist[st] = 0.0
          heapq.heappush(pq, (0.0, id(st), st))

    popped = 0
    while pq:
      d, _, t = heapq.heappop(pq)
      if d > self.dist.get(t, float("inf")):
        continue
      popped += 1
      if len(self.dist) > max_states:
        raise RuntimeError(f"labeled abstraction exceeded {max_states} states")
      for s, c in preds(t):
        nd = d + c
        if nd < self.dist.get(s, float("inf")):
          self.dist[s] = nd
          heapq.heappush(pq, (nd, id(s), s))

    self.n_states = len(self.dist)
    self.n_pick_facts = len(self._pick_cache)
    self.build_time = _time.perf_counter() - t0
    if verbose:
      print(f"[labeled-h] {self.n_states} coreachable states, {self.n_pick_facts} pick "
            f"facts, built in {self.build_time:.2f}s")

  def abstract(self, state):
    main = next(g for g in state.subassemblies() if self.assembly.base in g)
    ideal = _subasm_key(main)
    G = tuple(sorted((c[2], c[3]) for p in main
                     for c in state.supports.get(p, ()) if c[0] == "grasp"))
    on_table = any(("table",) in state.supports.get(p, ()) for p in main)
    face = _placement_face(state, main) if on_table else self.H
    L = tuple(sorted((c[2], c[3]) for p in state.parent if p not in main
                     for c in state.supports.get(p, ()) if c[0] == "grasp"))
    return (ideal, G, face, L)

  def value(self, state):
    # A projection with no backward distance PROVABLY cannot reach the goal: the abstract
    # actions cover every real action (relaxation), so no abstract path => no real path.
    # inf lets the search prune dead branches -- and detect infeasible INSTANCES at s0
    # (e.g. insert-force above every grasp cap: no assemble is ever wrench-feasible).
    v = self.dist.get(self.abstract(state))
    if v is None:
      self.misses += 1
      return float("inf")
    return v


class CollapsedLabeledHeuristic:
  """IDENTITY-COLLAPSED variant of LabeledDisassemblyHeuristic, for per-robot SCALING.

  Same backward-Dijkstra construction and the same fact machinery, but the abstract state
  drops the grasp identity of LOOSE held parts:

      (ideal, G, face, L)   with   L = sorted tuple of PART NAMES (no grasp ids).

  `G` (grasps ON the assembly) stays labeled, so every wrench fact (w_hold_ok /
  w_assemble_ok, incl. the inserter-press check that powers instance-infeasibility
  detection like isotropic@30N) is computed EXACTLY as in the fine class. What changes:

    * inverse PICK-LOOSE gates on EXISTS-a-grasp (and exists-a-face, as before): a loose
      part is backward-removable iff SOME grasp picks it off the table. Cached PER PART.
    * inverse PLACE-LOOSE emits ONE edge per part (not one per grasp).
    * inverse ASSEMBLE moves the inserted part to L as a bare name; its brace grasp gq
      (from the still-labeled G) is checked by w_assemble_ok exactly, then forgotten.

  This is a COARSENING of the fine abstraction: every gate is equal or more permissive, so
  the values are still ADMISSIBLE lower bounds (coarse_h <= fine_h pointwise) and an inf
  verdict is still a proof -- optimality/completeness guarantees are unchanged; only
  guidance tightness is traded for a state count that no longer multiplies grasp choices
  into L (C(parts,k) instead of C(all-grasps,k) per capacity slot).

  DELIBERATELY NOT DONE: gating inverse-assemble on "gq is pickable" (the pick/press
  linkage the fine class gets implicitly). With handovers enabled a press-capable but
  unpickable grasp is REAL-reachable (pick any grasp, hand over in the air), so that
  conjunction gate would be INADMISSIBLE. The collapse simply prices the swap optimistically.

  Kept fully separate from the fine class (own flag `collapsed_h`) so the well-tested one
  stays untouched; the duplication is the price of that isolation."""

  H = "H"                                                  # face value for held-in-air

  def __init__(self, assembly, n_robots, grasp_table, gripper_path=None, verbose=False,
               max_states=2_000_000, clearance=0.02, gripper_kind=None, bracing=False):
    import heapq
    import time as _time

    t0 = _time.perf_counter()
    self.assembly = assembly
    self.R = n_robots
    self.misses = 0
    self._bracing = bracing                                # model the add_grasp (co-grasp) edge
    self._clearance = clearance
    self._gk = gripper_kind
    objs = list(assembly.objects)
    base = assembly.base
    all_parts = _subasm_key(objs)
    self._closure = {p: frozenset(_mate_closure(assembly, p)) for p in objs}

    def tadd(t, x):
      return tuple(sorted(t + (x,)))

    def tsub(t, x):
      i = t.index(x)
      return t[:i] + t[i + 1:]

    ideal_keys = [_subasm_key(s) for s in get_subassemblies(assembly) if base in s]
    self.valid = {S: tuple(sorted((p, g) for p, g, _ in grasp_table[S]))
                  for S in ideal_keys}
    self.single = {q: tuple(sorted(g for _p, g, _ in grasp_table[(q,)])) for q in objs}

    self._ee_path = gripper_path
    self._pick_cache = {}

    def pick_ok(S_key, face, pg):
      key = (S_key, face, pg)
      hit = self._pick_cache.get(key)
      if hit is None:
        hit = self._pick_cache[key] = _table_aware_pick_fact(
            assembly, S_key, face, pg[0], pg[1], self._ee_path,
            clearance=self._clearance)
      return hit

    # placement faces are now looked up PER SUBASSEMBLY in the stable rest-pose table
    # (rest_faces) instead of the old global 0..5 -- so the abstraction offers exactly the
    # resting orientations the real places()/precompute_stable_faces do (e.g. a lone cylinder
    # only cap-down). Same faces on both sides => still a relaxation => still admissible.
    rf = lambda g: rest_faces(self.assembly, set(g))
    Hf = self.H

    # per-PART loose-pickability: exists-a-face AND exists-a-grasp (the collapse of the fine
    # class's per-(part,grasp) gate). Lazy: most parts are decided by their first grasp.
    self._loose_pickable = {}

    def loose_pickable(q):
      hit = self._loose_pickable.get(q)
      if hit is None:
        hit = self._loose_pickable[q] = any(
            pick_ok((q,), f, (q, g)) for g in self.single[q] for f in rf((q,)))
      return hit

    # ---- WRENCH facts: identical to the fine class (G stays labeled, so these are exact).
    flatposes = _flat_poses(assembly)
    self._wfacts = {}

    def crafted(ideal_key, G, face):
      parent, place, supports = {}, {}, {}
      ideal_set = set(ideal_key)
      for q in assembly.objects:
        if q in ideal_set:
          if q == assembly.base:
            parent[q] = ("gripper", "zz0") if face is Hf else ("table",)
            supports[q] = frozenset() if face is Hf else frozenset({("table",)})
            if face is not Hf:
              place[q] = ("slot", 0, face)
          else:
            onto = assembly.relative_pose(q)[0]
            parent[q] = ("part", onto)
            supports[q] = frozenset({("mate", onto)})
        else:
          parent[q] = ("table",)
          supports[q] = frozenset({("table",)})
          place[q] = ("slot", _part_index(q), _IDENTITY_FACE)
      for i, (p, g) in enumerate(G):
        supports[p] = supports[p] | {("grasp", f"zz{i}", p, g)}
      return State(parent=parent, place=place, supports=supports, poses=dict(flatposes))

    def held_single(q, gq):
      st = crafted((), (), _IDENTITY_FACE)
      st.parent[q] = ("gripper", "zzq")
      st.place.pop(q, None)
      st.supports[q] = frozenset({("grasp", "zzq", q, gq)})
      return st

    def w_hold_ok(ideal_key, G, face):
      if self._gk is None:
        return True
      key = ("H", ideal_key, G, face)
      hit = self._wfacts.get(key)
      if hit is None:
        st = crafted(ideal_key, G, face)
        hit = self._wfacts[key] = _group_stable(st, assembly, set(ideal_key), self._gk)
      return hit

    def w_assemble_ok(ideal_s_key, G2, face, q, gq):
      if self._gk is None:
        return True
      key = ("A", ideal_s_key, G2, face, q, gq)
      hit = self._wfacts.get(key)
      if hit is None:
        onto = assembly.relative_pose(q)[0]
        ok = w_hold_ok(ideal_s_key, G2, face)
        if ok and wrench.PARAMS.get("check_inserter", True):
          st_q = held_single(q, gq)
          ok = _group_stable_loaded(st_q, assembly, frozenset({q}), self._gk,
                                    _inserter_load_variants(st_q, assembly, q, onto))
        if ok:
          st_t = crafted(ideal_s_key, G2, face)
          ok = _group_stable_loaded(st_t, assembly, set(ideal_s_key), self._gk,
                                    _insertion_load_variants(st_t, assembly, q, onto))
        hit = self._wfacts[key] = ok
      return hit

    def preds(t):
      ideal, G, face, L = t                               # L: bare part names
      cap_used = len(G) + len(L)
      # inverse ASSEMBLE: some (q,gq) in G was the inserter's brace; q maximal in ideal.
      # The predecessor holds q LOOSE -- as a bare name (its grasp identity is forgotten).
      for q, gq in G:
        if q == base:
          continue
        if any(q in self._closure[r] for r in ideal if r != q):
          continue                                        # not maximal -> not last-added
        if face is not Hf and not _on_table_mateable_cached(self.assembly, q, face):
          continue
        G2 = tsub(G, (q, gq))
        if any(p == q for p, _ in G2):
          continue                                        # q was loose then: single-held only
        if face is Hf and len(G2) < 1:
          continue                                        # target had to be supported
        if not w_assemble_ok(tsub(ideal, q), G2, face, q, gq):
          continue                                        # wrench: press not transmittable
        yield (tsub(ideal, q), G2, face, tadd(L, q)), 1.0
      # inverse PICK-LOOSE: exists-a-face AND exists-a-grasp (the collapse)
      for q in L:
        if loose_pickable(q):
          yield (ideal, G, face, tsub(L, q)), 1.0
      # inverse LIFT: held solo -> was resting on some face and got picked (G-labeled, exact)
      if face is Hf and len(G) == 1:
        pg = G[0]
        for f in rf(ideal):
          if pick_ok(ideal, f, pg):
            yield (ideal, (), f, L), 1.0
      # inverse PLACE-ASSEMBLY: on table now -> was held by one more grasp in the air
      if face is not Hf and cap_used < self.R:
        for pg in self.valid[ideal]:
          if pg not in G:
            yield (ideal, tadd(G, pg), Hf, L), 1.0
      # inverse RELEASE: some grasp was just opened (wrench-gated on t staying supported)
      if cap_used < self.R and w_hold_ok(ideal, G, face):
        for pg in self.valid[ideal]:
          if pg not in G:
            yield (ideal, tadd(G, pg), face, L), 1.0
      # inverse ADD_GRASP (bracing only): pre-state held the assembly with one FEWER grasp --
      # forms the >=2-grasp brace offset horizontal mates need (see LabeledDisassemblyHeuristic
      # and [[cograsp-gap]]). Extra edges only relax the bound -> admissibility safe.
      if self._bracing:
        for pg in G:
          Gs = tsub(G, pg)
          if len(Gs) >= 1 and w_hold_ok(ideal, Gs, face):
            yield (ideal, Gs, face, L), 1.0
      # inverse fused HANDOVER (cost 2): solo-held; the grasp was swapped
      if face is Hf and len(G) == 1 and len(L) <= self.R - 2:
        pg_new = G[0]
        for pg_old in self.valid[ideal]:
          if pg_old != pg_new:
            yield (ideal, (pg_old,), Hf, L), 2.0
      # inverse PLACE-LOOSE: ONE edge per part (any valid grasp; existence suffices)
      if cap_used < self.R:
        for q in objs:
          if q in ideal or q in L:
            continue
          if self.single[q]:
            yield (ideal, G, face, tadd(L, q)), 1.0

    # goal states: identical seeding to the fine class (L=() either way)
    self.dist = {}
    pq = []
    from itertools import combinations
    vg_full = self.valid[all_parts]
    for k in range(self.R + 1):
      for combo in combinations(vg_full, k):
        for face in [Hf] + list(rf(all_parts)):
          if face is Hf and len(combo) < 1:
            continue
          st = (all_parts, combo, face, ())
          self.dist[st] = 0.0
          heapq.heappush(pq, (0.0, id(st), st))

    popped = 0
    while pq:
      d, _, t = heapq.heappop(pq)
      if d > self.dist.get(t, float("inf")):
        continue
      popped += 1
      if len(self.dist) > max_states:
        raise RuntimeError(f"collapsed labeled abstraction exceeded {max_states} states")
      for s, c in preds(t):
        nd = d + c
        if nd < self.dist.get(s, float("inf")):
          self.dist[s] = nd
          heapq.heappush(pq, (nd, id(s), s))

    self.n_states = len(self.dist)
    self.n_pick_facts = len(self._pick_cache)
    self.build_time = _time.perf_counter() - t0
    if verbose:
      print(f"[collapsed-h] {self.n_states} coreachable states, {self.n_pick_facts} pick "
            f"facts, built in {self.build_time:.2f}s")

  def abstract(self, state):
    main = next(g for g in state.subassemblies() if self.assembly.base in g)
    ideal = _subasm_key(main)
    G = tuple(sorted((c[2], c[3]) for p in main
                     for c in state.supports.get(p, ()) if c[0] == "grasp"))
    on_table = any(("table",) in state.supports.get(p, ()) for p in main)
    face = _placement_face(state, main) if on_table else self.H
    L = tuple(sorted({c[2] for p in state.parent if p not in main
                      for c in state.supports.get(p, ()) if c[0] == "grasp"}))
    return (ideal, G, face, L)

  def value(self, state):
    # inf is still a PROOF (the collapse only coarsens the relaxation) -- see class doc.
    v = self.dist.get(self.abstract(state))
    if v is None:
      self.misses += 1
      return float("inf")
    return v


_TABLE_PICK_FACT_CACHE = {}
_FACT_TEMPLATES = {}                                       # (ee_path, id(assembly)) -> template


def _fact_template(gripper_path, assembly):
  # ONE reusable config per (gripper, assembly): world + every part's shape parked far away
  # + the EE pre-parsed. Per fact we COPY it (rai ~0.4ms) and reposition -- profiling showed
  # place_floating_ee's addFile (re-parsing the gripper .g per fact) was a third of the
  # whole heuristic build. Frames are named by PART (shape-correct per part).
  key = (gripper_path, id(assembly))
  hit = _FACT_TEMPLATES.get(key)
  if hit is not None:
    return hit
  C = make_world()
  for i, p in enumerate(assembly.objects):
    add_part_frame(C, p, assembly.part_shape(p)) \
        .setPosition([0.0, -500.0 - 2.0 * i, 1.0])
  place_floating_ee(C, np.array([0.0, -900.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
                    gripper_path, prefix="eeFACT_")
  tool, base_rel_tool, q_corr = _gripper_rest_transforms(C, "eeFACT_")
  table = C.getFrame("table")
  top = table.getPosition()[2] + table.getSize()[2] / 2.0
  hit = _FACT_TEMPLATES[key] = (C, base_rel_tool, q_corr, top)
  return hit


def _table_aware_pick_fact(assembly, S_key, face, gpart, gid, gripper_path,
                           clearance=0.0):
  # can the floating EE sit at grasp (gpart,gid) while subassembly S_key rests on `face`,
  # without hitting the TABLE (or S itself)? The grasp table checks the grasp only against
  # S; the table is what kills e.g. two-finger side grasps whose lower finger dips under
  # the cube. One template-copied rai config per fact, cached globally (pure geometry).
  #
  # `clearance` raises the group above exact rest: the REAL scatter poses float ~15mm above
  # the table, so the real pick gate is more tolerant than an exactly-resting scene -- an
  # abstraction fact must be AT LEAST as permissive as any real context (over-permissive
  # only weakens the heuristic; over-strict breaks admissibility).
  key = (id(assembly), S_key, face, gpart, gid, gripper_path, round(clearance, 4))
  hit = _TABLE_PICK_FACT_CACHE.get(key)
  if hit is not None:
    return hit
  tmpl, base_rel_tool, q_corr, top = _fact_template(gripper_path, assembly)
  C = ry.Config()
  C.addConfigurationCopy(tmpl)
  q = _FACE_DOWN_QUATS[face]
  R = _quat_to_R(q)
  ref = S_key[0]
  rest = _group_rest_height(assembly, set(S_key), ref, face)
  ref_off = assembly.position_in_base(ref)
  base_pos = np.array([0.0, 0.0, top + rest + clearance])
  poses = {}
  for p in S_key:
    pos = base_pos + R @ (assembly.position_in_base(p) - ref_off)
    f = C.getFrame(p)
    f.setPosition(pos)
    f.setQuaternion(q)
    poses[p] = np.concatenate([pos, q])
  grasp_world = _pose_compose(poses[gpart], grasp_pose(assembly, gpart, gid))
  base_world = _ee_root_pose(grasp_world, base_rel_tool, q_corr)
  root = C.getFrame("eeFACT_ur_base")
  root.setPosition(base_world[:3])
  root.setQuaternion(base_world[3:])
  ok = all(d >= -_CHEAP_PENETRATION_TOL for _a, _b, d in C.getCollisions())
  _TABLE_PICK_FACT_CACHE[key] = ok
  return ok


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
      if hns == float("inf"):
        continue                                          # provable dead end (admissible h)
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
  names = C.getFrameNames()
  vacuum = any(n.endswith("_ur_vacuum") for n in names)
  # ...and mirror the ROBOT model: real arms get the wrist-extended cheap EE, a floating-EE
  # env gets the plain one (no wrist exists in that model). The wrist-3 capsule is arm-SPECIFIC,
  # so a real UR5 arm (detected via its ur_arm_ur5 marker) gets the ur5-calibrated cheap wrist,
  # not the ur10 one. See cheap_ee_path / UR5_CHEAP_EE_PATHS in env.py.
  floating = not any(n.endswith("_ur_wrist_3_joint") for n in names)
  is_ur5 = any(n.endswith("_ur_arm_ur5") for n in names)
  ee_path = cheap_ee_path(is_ur5, floating, vacuum)
  checker = CollisionChecker(assembly, robots, gripper_path=ee_path)
  # plain-EE variant for the labeled-h pick facts (clutter-free scenes; see the h_lab
  # construction in solve_assembly_problem for why the wrist is excluded there).
  checker.fact_gripper_path = os.path.join(os.path.dirname(__file__),
                                           UR10_PATHS[(True, vacuum)])
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
                           anonymous=False, labeled_h=False, collapsed_h=False,
                           bracing=False):
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

  checker, grasp_table = _search_context(C, assembly, robots)
  if gk is not None:
    reset_wrench_cache()

  # labeled_h=True swaps h_abs for the GRASP+FACE-LABELED disassembly abstraction (see
  # LabeledDisassemblyHeuristic) -- strictly tighter where labels matter, at a build cost
  # of seconds. collapsed_h=True uses the IDENTITY-COLLAPSED variant instead (same
  # guarantees, weaker guidance, per-robot-scalable build; see CollapsedLabeledHeuristic).
  # Same finite-weight gating as h_abs.
  h_lab = None
  if (labeled_h or collapsed_h) and use_heuristic:
    if math.isinf(weight):
      warnings.warn("labeled_h requested but weight=inf (greedy): abstraction heuristics "
                    "are gated to finite weights -- running WITHOUT it. Use e.g. weight=5.",
                    stacklevel=2)
    else:
      if gk is not None:
        reset_wrench_cache()                              # facts below use the wrench LP
      # wrench facts follow the search's wrench model (gk) by default; wrench_aware_h=False
      # builds the labeled abstraction wrench-BLIND for comparison (same toggle as h_abs).
      # pick facts use the PLAIN floating EE even when the checker runs the wrist variant:
      # fact scenes hold ONE part at a free table spot (no clutter), so the wrist capsule
      # adds nothing there except unsound wrist-vs-table rejections (see _wrist_table_pair).
      cls = CollapsedLabeledHeuristic if collapsed_h else LabeledDisassemblyHeuristic
      h_lab = cls(assembly, len(robots), grasp_table,
                  gripper_path=getattr(checker, "fact_gripper_path", checker.gripper_path),
                  verbose=verbose,
                  gripper_kind=gk if wrench_aware_h else None, bracing=bracing)
      h_abs = None

  def h_fn(state):
    if not use_heuristic:
      return 0.0
    hv = heuristic(state, assembly)
    if h_lab is not None:
      return max(hv, h_lab.value(state))
    return max(hv, h_abs.value(state)) if h_abs is not None else hv

  def succ_fn(s):
    return successors(s, assembly, robots, grasp_table=grasp_table, canonical=canonical,
                      gripper_kind=gk, handovers=handovers, allow_place=allow_place,
                      bracing=bracing)

  # FULLY LAZY feasibility: the cheap geometric check moved from PUSH to POP (edge check),
  # like the IK before it -- profiling showed ~20 pushed edges per expansion, each paying a
  # ~2ms rai collision query, while most pushed edges are never popped. Claim-at-first-
  # feasible-pop keeps this sound (an edge failing at pop is dropped; the state stays
  # reachable via its other queued edges).
  def edge_check_fn(prev, act, ns):
    if not cheap_feasible(act, prev, ns, assembly, checker):
      return False
    if robot_ik is not None and act[0] in _IK_ACTIONS:
      ok, _witness = robot_ik.bind(act, prev, ns, assembly)
      return ok
    return True

  # LATE BINDING (anonymous=True): search with a robot-anonymous state key -- states that
  # differ only in WHICH robot holds what merge, so robot-permutation redundancy vanishes and
  # extra idle robots add no states. The returned plan still names concrete robots (the merged
  # class's representative assignment). Sound for interchangeable (floating) grippers, like
  # `canonical`; with fixed-base arms + robot_ik a merged assignment may hide a reachability
  # difference -- use for the fast first answer, then verify/repair with concrete IK.
  if anonymous and robot_ik is not None and verbose:
    warnings.warn("anonymous=True with robot_ik: merged robot assignments may hide "
                  "per-robot reachability differences; treat the plan as a candidate.")
  start = get_initial_state(C, assembly)
  if h_lab is not None and h_lab.value(start) == float("inf"):
    if verbose:
      print("[labeled-h] the INITIAL state provably cannot reach the goal under the "
            "abstraction's cheap-geometry + wrench facts -> instance INFEASIBLE as "
            "configured (e.g. insert-force above every grasp capability).")
    return None

  tag = ("UCS" if not use_heuristic else "greedy" if math.isinf(weight)
         else "A*" if weight == 1.0 else f"wA*(w={weight:g})")
  set_anonymous_robots(anonymous)
  try:
    return _astar(start,
                  goal_fn=lambda s: s.is_goal(assembly),
                  succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn, cheap_fn=None,
                  edge_check_fn=edge_check_fn,
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

  def edge_check_fn(prev, act, ns):
    # fully lazy (cheap geometry + IK at POP; see solve_assembly_problem)
    if not cheap_feasible(act, prev, ns, assembly, checker):
      return False
    if robot_ik is not None and act[0] in _IK_ACTIONS:
      ok, _w = robot_ik.bind(act, prev, ns, assembly)
      return ok
    return True

  subplan = _astar(start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn,
                   cheap_fn=None, edge_check_fn=edge_check_fn,
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
                     handovers=True, allow_place=True, bracing=False):
  # the full edge set of the declutter sub-search: the ordinary successors plus the
  # stow-a-blocker moves. Shared by the sub-search and the plan-realisation walk so both
  # agree on what a "stow" edge produces. (Stowing a loose blocker aside is always allowed --
  # allow_place=False only forbids setting the ASSEMBLY down.) `bracing` must match the search
  # that PRODUCED the plan, else a plan using add_grasp can't be re-derived on replay.
  yield from successors(state, assembly, robots, grasp_table=grasp_table,
                        gripper_kind=gripper_kind, handovers=handovers,
                        allow_place=allow_place, bracing=bracing)
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

    def edge_check_fn(prev, act, ns):
      # fully lazy (cheap geometry + IK at POP; see solve_assembly_problem). Blocker
      # ATTRIBUTION stays here: a goal-relevant action failing the cheap check names the
      # loose parts in the way (same information as at push, just evaluated later).
      ok = cheap_feasible(act, prev, ns, assembly, checker)  # stow is a _GEOMETRIC_ACTION now
      if not ok:
        if act[2] == p or act[2] in ideal:                # a GOAL-RELEVANT action was blocked
          if act[0] == "pick":
            pairs = checker.penetrating_pairs(prev, assembly,
                                              extra_grasps=[(act[2], act[3])])
          else:
            pairs = checker.penetrating_pairs(ns, assembly)
          hit.update(_declutter_blockers(pairs, relevant, objects, checker.slots))
        return False
      if robot_ik is not None and (act[0] in _IK_ACTIONS or act[0] == "stow"):
        a = ("place",) + tuple(act[1:]) if act[0] == "stow" else act   # stow is a place aside
        ok, _w = robot_ik.bind(a, prev, ns, assembly)
        return ok
      return True

    subplan = _astar(start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost,
                     h_fn=make_h(movable), cheap_fn=None, edge_check_fn=edge_check_fn,
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
