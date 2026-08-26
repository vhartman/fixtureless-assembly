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

  def __init__(self, assembly, robots, gripper_path=None, robot_gripper_paths=None):
    self.gripper_path = gripper_path or os.path.join(os.path.dirname(__file__), DEFAULT_FLOATING_EE)
    # MIXED END EFFECTORS: robot_gripper_paths = {robot: cheap-EE path} loads each robot's
    # slot with ITS OWN gripper stand-in (vacuum pen vs two-finger palm differ a lot); None
    # (uniform scenes) keeps every slot on self.gripper_path -- byte-identical behaviour.
    self.robots_order = list(robots)
    paths = [robot_gripper_paths.get(r, self.gripper_path) for r in robots] \
        if robot_gripper_paths else [self.gripper_path] * len(robots)
    self.mixed = len(set(paths)) > 1
    # robot -> gripper KIND (the geometry class): same-kind arms are interchangeable in a
    # floating-EE check, so verdict signatures and slot assignment key on the KIND, not the
    # robot -- two vacuum arms share one verdict, exactly like the uniform identity-free key.
    self._robot_kind = {r: ("vacuum" if "vacuum" in os.path.basename(p) else "two_finger")
                        for r, p in zip(robots, paths)}
    self._slot_kind = [self._robot_kind[r] for r in robots]

    template = make_world()                                # floor + table, same as the real env
    for obj in assembly.objects:                           # the parts (posed per query)
      add_part_frame(template, obj, assembly.part_shape(obj))

    # one free-floating gripper slot per robot (at most one active grasp per robot),
    # pre-loaded and parked far away so unused slots touch nothing.
    self.slots = [f"eeSLOT{i}_" for i in range(len(robots))]
    from env import maybe_tighten_fingers        # cheap-EE proxy must match the real scene's
    for prefix, path in zip(self.slots, paths):
      template.addFile(path, namePrefix=prefix)
      maybe_tighten_fingers(template, prefix)
    # per-slot tool transforms (grippers may differ under mixed EEs; identical when uniform)
    self._slot_tool = [_gripper_rest_transforms(template, prefix)[1:] for prefix in self.slots]
    self._slot_of = {r: i for i, r in enumerate(self.robots_order)}
    self.base_rel_tool, self.q_correction = self._slot_tool[0] if self.slots else (None, None)
    for i, prefix in enumerate(self.slots):
      template.getFrame(prefix + "ur_base").setPosition(_park_pos(i))
    self.template = template
    self._verdicts = {}                                    # discrete scene signature -> bool
    # PERSISTENT scene reuse (2026-07-17): rai's stale-broadphase problem is solved by a
    # global C.setJointState(C.getJointState()) round-trip before computeCollisions -- it
    # refreshes ALL frame writes (even plain setPosition on jointless frames; measured
    # ~0.005ms/query vs ~1.25ms for copy+rebuild, and the query cost was ~50% of floating
    # searches). False falls back to the historical fresh-copy-per-query path.
    self.persistent = True

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
      # mixed EEs: the holder's gripper KIND decides which geometry stands there -- key on
      # the kind, NOT the robot (same-kind arms share verdicts, like the uniform identity-
      # free key). Uniform scenes keep the plain (part, gid) key.
      grasps = tuple(sorted(
          ((self._robot_kind.get(c[1]), c[2], c[3]) if self.mixed else (c[2], c[3]))
          for p in group
          for c in state.supports.get(p, ()) if c[0] == "grasp"))
      on_table = any(("table",) in state.supports.get(p, ()) for p in group)
      if on_table:
        ref = reference_part(group)
        anchor = tuple(np.round(np.asarray(state.poses[ref], dtype=float), 4))
        parts.append(("T", members, state.place.get(ref), anchor, grasps))
      else:
        parts.append(("H", members, grasps))
    # extras likewise: robot -> kind (3-tuples), so the pick gate's verdicts are shared
    # across same-kind arms. A ("pose", part, pose7) or ("pose", robot, part, pose7) extra
    # is an EXPLICIT grasp pose (the DB is still being built and has no ids yet).
    _g = []
    for g in extra_grasps:
      if len(g) == 4 and g[0] == "pose":
        _g.append(("pose", self._robot_kind.get(g[1], g[1]), g[2],
                   tuple(np.round(np.asarray(g[3], dtype=float), 4))))
      elif len(g) == 3 and g[0] == "pose":
        _g.append(("pose", g[1], tuple(np.round(np.asarray(g[2], dtype=float), 4))))
      else:
        _g.append(tuple(g))
    ex = tuple(sorted((self._robot_kind.get(g[0], g[0]),) + tuple(g[1:]) if len(g) == 3
                      else tuple(g) for g in _g)) if self.mixed \
        else tuple(sorted(tuple(g) for g in _g))
    return (tuple(sorted(parts)), ex)

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
    if self.persistent:
      C_col = self.template                                # in-place repose + refresh below
    else:
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

    # each grasp goes to a slot of ITS HOLDER'S GRIPPER KIND (same-kind arms are
    # interchangeable floating-EE geometry; with uniform grippers any assignment matches the
    # old sequential zip). extra_grasps entries may be (gpart, gid), (robot, gpart, gid) or
    # ("pose", gpart, pose7); robot-less extras take any free slot (callers pass the robot
    # when it matters), and a pose extra never calls grasp_pose (the DB may not exist yet).
    used = set()
    assigns = []                                           # (slot_idx, gpart, gid-or-pose)
    robot_grasps = [(c[1], c[2], c[3]) for p in state.bodies()
                    for c in state.supports.get(p, ()) if c[0] == "grasp"]
    extras = []
    for g in extra_grasps:
      if len(g) == 4 and g[0] == "pose":
        extras.append((g[1], g[2], np.asarray(g[3], dtype=float)))
      elif len(g) == 3 and g[0] == "pose":
        extras.append((None, g[1], np.asarray(g[2], dtype=float)))
      elif len(g) == 3:
        extras.append(g)
      else:
        extras.append((None, g[0], g[1]))

    def _slot_for(r):
      kind = self._robot_kind.get(r) if r is not None else None
      for j in range(len(self.slots)):
        if j not in used and (kind is None or self._slot_kind[j] == kind):
          return j
      return next(j for j in range(len(self.slots)) if j not in used)

    for r, gpart, gid in robot_grasps + extras:
      i = _slot_for(r)
      used.add(i)
      assigns.append((i, gpart, gid))
    for i, gpart, grasp_id in assigns:
      if isinstance(grasp_id, np.ndarray):                 # ("pose", part, pose7) extra
        gpose = np.asarray(grasp_id, dtype=float)
      else:
        gpose = grasp_pose(assembly, gpart, grasp_id)
      grasp_world = _pose_compose(np.asarray(part_poses[gpart], dtype=float), gpose)
      brt, qc = self._slot_tool[i]
      base_world = _ee_root_pose(grasp_world, brt, qc)
      root = C_col.getFrame(self.slots[i] + "ur_base")
      root.setPosition(base_world[:3])
      root.setQuaternion(base_world[3:])
    if self.persistent:
      # in-place mutation keeps STALE poses from the previous query: explicitly RE-PARK the
      # unused slots (parts are all reposed above every query), then force the global
      # kinematics/proxy REFRESH -- without it rai serves stale collision results.
      for i in range(len(self.slots)):
        if i in used:
          continue
        root = C_col.getFrame(self.slots[i] + "ur_base")
        root.setPosition(_park_pos(i))
        root.setQuaternion([1.0, 0.0, 0.0, 0.0])
      C_col.setJointState(C_col.getJointState())
      C_col.computeCollisions()
    # (fresh-copy path: remaining slots stay parked from the template.)
    return C_col

  def _check(self, state, assembly, extra_grasps=(), note=""):
    # tolerate grazing contacts (see _CHEAP_PENETRATION_TOL) instead of binary-exact
    # getCollisionFree -- the cheap layer must not be stricter than the (soft) IK.
    # VERDICT-MEMOIZED on the discrete scene signature (see _sig): building the rai scene
    # + collision query costs ~2ms and dominated whole-search profiles (~90%).
    key = self._sig(state, extra_grasps)                   # _sig normalizes every extra form
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

  def pick_collision_free(self, s0, gpart, grasp_id, assembly, note="", robot=None):
    # cheap gate for PICK edges: place a floating EE at the candidate grasp with the part
    # at its CURRENT (pre-pick, on-table) pose, in the actual environment. This is what
    # the precomputed grasp table does NOT see -- it checks a grasp only against its own
    # subassembly (no table, no neighbours), so e.g. a two-finger side grasp whose lower
    # finger dips below the cube (into the table) passes precompute yet can never be
    # picked. Rejecting it here costs ~2ms instead of a full (restarted) IK solve.
    # `robot`: the acting robot -- under mixed EEs its own gripper geometry stands at the
    # candidate (None = any slot, the uniform behaviour).
    extra = [(robot, gpart, grasp_id)] if robot is not None else [(gpart, grasp_id)]
    return self._check(s0, assembly, extra_grasps=extra, note=note)

  def pose_collision_free(self, state, gpart, grasp_pose_7, assembly, note="", robot=None):
    """The same floating-EE gate for an EXPLICIT grasp pose (part frame) instead of a grasp
    id. Used by the scene-aware grasp sampler: it filters candidates while the DB is still
    being built, so grasp ids do not exist yet and calling back through grasp_pose would
    recurse. The EE stands at compose(part's state pose, pose); everything else is the
    ordinary pick gate."""
    extra = (("pose", gpart, grasp_pose_7) if robot is None
             else ("pose", robot, gpart, grasp_pose_7))
    return self._check(state, assembly, extra_grasps=[extra], note=note)


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


# ---------------------------------------------------------------------------------------
# PRE-* funnel-mouth algebra. One definition per skill family so the exporter's backoff and
# the consistency pass's NOMINAL approach vector can never drift apart (they used to be four
# inline expressions, only reachable through a rai solve). Pure numpy: unit-testable.
# ---------------------------------------------------------------------------------------
def _backoff_ee(ee_target, delta):
  # pick / add_grasp: RETRACT the tool centre along -approach. The corrected EE +z points from
  # the standoff INTO the part and `ee_target` is expressed in the PART frame, so this is a
  # pure translation there.
  out = np.asarray(ee_target, dtype=float).copy()
  out[:3] = out[:3] - delta * (_quat_to_R(out[3:]) @ np.array([0.0, 0.0, 1.0]))
  return out


def _backoff_place(target, delta):
  # place: LIFT the group off its rest pose, +z in WORLD (the target is an absolute pose).
  out = np.asarray(target, dtype=float).copy()
  out[2] = out[2] + delta
  return out


def _backoff_mate(rel_pos, axis, delta):
  # assemble: back off along -mate_axis in `onto`'s frame (parts stay axis-aligned, so the
  # base-frame mate axis is also the offset direction there). `axis` None = the part has no
  # mate axis: the backoff is SKIPPED, i.e. the funnel has zero length. Historical behaviour,
  # kept bit-identical -- fuse_pre_skill_witnesses is the one that reports it.
  if axis is None:
    return rel_pos
  return rel_pos - delta * np.asarray(axis, dtype=float)


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
    self.tool_kind = {}                                # analytic calibration is EE-specific
    self.robot_base = {}                               # subtree root, for single-arm IK
    self.robot_arm = {}                                # 'ur5' | 'ur10' -- picks the analytic solver
    self.base_link = {}                                # URDF base frame (ssik's reference)
    self.ik_seeding = False                            # warm-start KOMO with analytic (ssik) IK
    names = set(self.C.getFrameNames())
    for r in self.robots:
      tool_center, _base_rel, q_corr = _gripper_rest_transforms(self.C, r + "_")
      self.gripper_frame[r] = tool_center
      self.q_correction[r] = q_corr
      # Vacuum is physically roll-symmetric.  Search IK currently chooses the grasp DB's
      # canonical ALIGNED roll and constrains that full pose; ``symmetric`` tells witness
      # ranking/deduplication that alternate vacuum wrist rolls are equivalent.
      self.symmetric[r] = "vacuum" in tool_center
      self.tool_kind[r] = "vacuum" if self.symmetric[r] else "two_finger"
      self.robot_base[r] = r + "_ur_base"              # UR base frame; parented to the table
      # a FLOATING EE has no arm at all: no base_link frame, no kinematic chain -> analytic
      # IK / reach bounds are meaningless for it (robot_arm=None disables every ssik path).
      if (r + "_openarm_right_base_link") in names or (r + "_openarm_left_base_link") in names:
        # OpenArm duo: per-SIDE calibrations (the two arms have different joint-sign
        # conventions vs the ssik artifact); ur_base_link alias exists here too, so this
        # branch MUST come before the generic UR check below.
        self.robot_arm[r] = ("openarm_right" if (r + "_openarm_right_base_link") in names
                             else "openarm_left")
        self.base_link[r] = r + ("_openarm_right_base_link" if
                                 (r + "_openarm_right_base_link") in names
                                 else "_openarm_left_base_link")
      elif (r + "_right_fr3v2_link0") in names or (r + "_left_fr3v2_link0") in names:
        # FR3 duo: the env's ur_base_link alias marks the arm, but there is NO ssik
        # calibration for franka yet -> robot_arm="fr3" keeps every analytic path disabled
        # while base_link still anchors the reach-sphere bound.
        self.robot_arm[r] = "fr3"
        self.base_link[r] = r + "_ur_base_link"
      elif (r + "_ur_base_link") in names:
        self.robot_arm[r] = "ur5" if (r + "_ur_arm_ur5") in names else "ur10"
        self.base_link[r] = r + "_ur_base_link"        # ssik solves in base_link's frame
      else:
        self.robot_arm[r] = None
        self.base_link[r] = None
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
    self._reach_bounds = {}                               # robot -> sound max-reach (lazy)
    self.reach_rejects = 0                                # rejects by the reach-sphere bound
    self.analytic_rejects = 0                             # rejects by empty ssik solution sets
    # A1 INSTRUMENTATION (pure counters/timers; ZERO behavior change): where bind/_solve_ik
    # time goes. Summarized by prof_report() at the end of main.run(). Stage buckets
    # (seeded/unseeded/restart) count EVERY _solve_ik call, including the reach precheck's
    # relaxed solves -- precheck_time is the wall-clock view, the buckets the NLP-CPU view.
    self.prof = {
        "bind_calls": 0, "bind_cache_hits": 0, "bind_solves": 0,
        "bind_success": 0, "bind_fail": 0, "bind_time": 0.0,
        "precheck_calls": 0, "precheck_time": 0.0,
        "seeded_solves": 0, "seeded_time": 0.0,
        "unseeded_solves": 0, "unseeded_time": 0.0,
        "restart_solves": 0, "restart_time": 0.0,
        "post_calls": 0, "post_time": 0.0,                # _pen_ok + _clearance_ok gates
        "getpath_calls": 0,                               # is getPath() worth hoisting?
        "by_action": {},                                  # (kind, part, gid) -> [n, fails, s]
        "calls_with_seeds": 0, "calls_without_seeds": 0,  # why so few analytic solves?
        "wins": {},                                       # winning attempt histogram
    }
    self.mate_eq_weight = None                            # None = 1e1 (rai default here)
    self.place_eq_weight = None                           # same, for _ik_place's object-pose eq
    self.mate_residual_reject = None                      # None = off; metres (--mate-accuracy)
      # THE acceptance criterion for an assemble. rai's `feasible` verdict accepts an eq
      # constraint that is still MILLIMETRES out, and tightening stopTolerance does NOT change
      # what it ACCEPTS -- measured: bind returns the same 4.5mm mate at stopTolerance 1e-2 and
      # 1e-3 alike. The only way to refuse an under-seated mate is to measure it and say no.
    self.nlp_stop_tolerance = None                        # None = rai's default 1e-2 (=10mm!)
      # rai's NLP_Solver stops at stopTolerance=0.01 by DEFAULT, so an eq-constrained mate is
      # declared solved while still MILLIMETRES out -- measured ~4.5mm on every cross-12 mate,
      # short along the part's own insertion axis. mr_bench then bakes that error in at attach
      # and the exported assembly disagrees with ours. Set to tighten.
    self.mate_pair_gate = False                           # --strict-mate-collisions (see _pen_ok)
    self.keep_contacts = False                            # --always-contact: never disable a
    # touched part's/group's contact in the skill IK -- the witness must clear the strict
    # gate with EVERY contact on (grasp/mate contact included); an overlapping plan then
    # fails its IK and the search re-runs. Default off (byte-identical).
    self._mate_exempt = None                              # (moving parts, target parts) while solving a mate
    self.harvest_branches = None                          # scratch list while harvesting
    self.harvest_all_branches = True                      # keep EVERY feasible analytic
      # branch as a ladder candidate, not just the first. `_solve_with_seeds` used to return on
      # the first feasible seed, so a solo pick offered ~3 candidates all in ONE IK branch and
      # the DP had nothing to choose between -- measured, the harvest took one pick from 1
      # candidate/1 branch to 6/2. Set False to restore the first-feasible-only behaviour.
    self.seed_restart_cap = 2                             # random restarts left after seeds (see _solve_ik)
    # RESTART BUDGETS (--ik-restarts / --precheck-restarts). MEASURED (peg-stool chamfer,
    # 2026-08-21): all 208 precheck successes came from the FIRST relaxed attempt, and full-
    # solve restarts win 25/202 with nothing at attempt >=7 -- see
    # proposals/search-improvements.md. Lowering trades stochastic completeness for time.
    self.restarts_full = 12
    self.restarts_precheck = 12
    # FAILURE-INFORMED ORDERING (--fail-order). Collects per-candidate outcome evidence and,
    # when ON, reorders candidate sweeps never-tried-first / ascending fail rate. Ordering
    # ONLY: every candidate stays reachable (the sweep is exhaustive either way), so
    # completeness is untouched; only which equal-f variant is tried first changes.
    # Evidence key -> [n, fails, eq_sum_on_fail] ("abias" = assemble bias sweep slot).
    self.fail_order = False
    # --analytic-seed-cap K: warm-start KOMO with only the K best-ranked exact branches
    # (None = legacy: every branch). Caps the seeded-solve cost when a bind's exact branch
    # set is large and mostly colliding.
    self.analytic_seed_cap = None
    self.action_stats = {}
    # --empty-seed-refutes: treat an EMPTY calibrated analytic seed set as a sound
    # unreachability proof (skip relaxed precheck + KOMO). Sound because q1..q5 are
    # roll-independent and q6 spans >= 2pi in the rai models; OFF preserves the legacy
    # tolerant-KOMO chance on exactly-unreachable poses.
    self.empty_seed_refutes = False
    # ANALYTIC-REFUTATION STUDY: classify each bind by exact-IK coverage vs outcome, so we can
    # decide whether ssik alone could replace KOMO acceptance/refutation. Coverage=True means
    # every attached arm had a nonempty exact seed set (canonical roll for vacuum -- sound by
    # q6-range >= 2pi in the rai models). Key stats: slack_rescued (coverage but KOMO only won
    # off-seed => tolerance-slack mattered) and exactless_success (won with EMPTY exact set =>
    # direct counterexample to analytic-only).
    self.ana_stats = {}
    self._ctx_seeds = None
    self._last_win = ""
    # OPT-IN open-witness probing.  The legacy bind/search path never reads these.  The
    # dedicated scheduler calls probe_bind_batch(), which spends one bounded skill-aware tier,
    # caches successes only, and revisits failures in later fairness epochs.
    self.open_witness = False
    self.open_witness_epochs = 2
    self.open_witness_batch = 7
    self._open_probe_attempt = None
    self._open_probe_budget = 1
    self._open_probe_random_only = False
    self.braced_analytic = True                           # per-arm analytic for BRACED place/assemble
    # PRE-INSERTION funnel (--pre-insertion), EXPORT/REPLAY ONLY: metres to back off along the
    # mate axis for the pre-assemble GOTO waypoint. 0.0 = off. The SEARCH is untouched -- the
    # mate is still verified/witnessed at the real seated pose (`_ik_assemble` backoff defaults
    # to 0). The exporter splits each assemble into (pre_assemble goto -> T_pre) + (assemble ->
    # mated witness), so MP plans the free approach to the funnel mouth and the short seating
    # mode (which the mr_bench skills branch replaces with the real insertion skill).
    self.pre_insertion_delta = 0.0
    # sibling pre-skill backoffs (same export-split mechanism, own flags): pick retracts the
    # gripper along -approach (its +z tool axis points INTO the object), place LIFTS the object
    # +z (world) off its rest pose. 0.0 = that skill is not split.
    self.pre_pick_delta = 0.0
    self.pre_place_delta = 0.0
    self.pre_add_grasp_delta = 0.0   # brace/handover reach: retract like pick (-z_EE)
    self.post_release_delta = 0.0    # depart from a released grasp along the same -z_EE axis
    self.post_release_used_delta = None  # actual adaptive departure used by the last solve
    self.pre_insertion_pull = 1.0   # qItself pull weight tying T_pre to the skill witness (<1e1)
    # A funnel is a short Cartesian approach, not permission to jump to another IK branch.
    # This absolute actor-joint corridor is shared with fuse_pre_skill_witnesses' gate.  It is
    # enforced in pre_skill_witness itself so it also covers single-arm pick/place scenes where
    # every other arm has already been frozen out and the pin pass has no joint left to pin.
    self.pre_skill_max_dq = 0.35
    # DEPTH-GATED collision reject on the exact analytic branches (None = off). The full
    # solve's tolerant pose eq can rescue a GRAZING branch (measured: rescues sat at <=10mm
    # penetration) but never a deep one (unrescued failures >=14mm, median 55mm) -- so if
    # EVERY branch penetrates deeper than this gate, reject before any KOMO. 15mm keeps every
    # measured rescue while killing the deep-collision failure path.
    self.collision_reject_depth = 0.015
    # --strict-collisions: POST-ACCEPT penetration gate on every KOMO witness. The collision
    # objective is a SOFT ineq -- the NLP accepts mm-scale violations ("rescued grazes" up to
    # ~10mm), which downstream consumers (MuJoCo execution, a 0-tolerance mr_bench export)
    # then pay for. With penetration_reject set (metres), a feasible KOMO solution is ALSO
    # required to have getCollisionsTotalPenetration() <= this on the skill scene; dirty
    # solutions are treated as infeasible and the restart loop keeps searching for a clean
    # basin. NOTE: the gate sees only what the skill scene contains -- combine with
    # --witness-robot-collisions, else stripped idle arms stay invisible. None = off
    # (byte-identical). col_weight: the accumulatedCollisions ineq weight (1e1 historical;
    # strict mode raises it so the solver steers to clean configs instead of leaning on the
    # slack and getting rejected).
    self.penetration_reject = None
    self.col_weight = 1e1
    self.strict_rejects = 0
    self.mate_rejects = 0                               # diagnostics: dirty witnesses vetoed
    self.grasp_pose_reject = None                       # None = off; metres (--grasp-accuracy)
    self.grasp_rejects = 0                              # diagnostics: off-pose grasps vetoed
    self.grasp_eq_weight = None                         # None = 1e1: NEW-grasp tool-pose pin
                                                        # weight (--grasp-eq-weight; the
                                                        # maintenance pins keep 1e1 -- raising
                                                        # a FROZEN pair's weight makes solves
                                                        # infeasible, see _add_pin's comment)
    # --witness-margin (metres, None/0 = off): SHARED SEARCH+WITNESS clearance gate. Every
    # accepted search IK and every later witness re-solve uses this SAME predicate; a post-pass
    # is never allowed to invent a stricter notion of feasibility than the discrete search.
    # The gate covers every non-intended arm/part, inter-arm and non-adjacent self-collision
    # pair reported by rai. Only pair-specific grasp contacts are exempt (a hand touching the
    # payload attached to THAT hand, or the acting hand touching THIS action's grasp target),
    # plus part/part and table pairs. This hard minimum complements witness_floor's summed
    # robot/table score: many far pairs can no longer hide one measure-near-zero pair.
    self.clearance_reject = None
    self.margin_rejects = 0
    self.margin_reject_pairs = {}
    self.last_margin_reject = None
    # Some rai configs with a released, formerly attached multi-part group expose six named
    # arm joints but retain a seventh internal qIndex slot. Constant-initializing such a KOMO
    # with the otherwise-correct six-value analytic seed raises instead of merely rejecting
    # the seed. Remember those action/scene shapes and fall back to KOMO's own initializer.
    self.seed_init_incompatible = set()
    self.seed_init_skips = 0
    # DEBUG: set env PEN_VIEW_ACTION='pick a1 obj_1 0' to open a BLOCKING C.view(True) on the
    # FIRST penetration check of that exact action (space-joined tuple) -- prints the colliding
    # frame pairs and shows the skill scene at the checked witness. For inspecting whether a
    # --strict-collisions witness rejection is real. Off unless the env var is set.
    self._pen_view_action = os.environ.get("PEN_VIEW_ACTION")
    self._pen_viewed = False
    self._funnel_solve = False                        # set while solving a --pre-*/departure
                                                      # funnel mouth: table pairs gate at ~0
    self._cur_action = None                               # set per _ik_* call (for the view)
    # HARD-reject depth (see _probe_branch_combos): only when EVERY exact branch penetrates
    # beyond this does the probe reject without a KOMO. Conservative on purpose (near the
    # measured unrescued MEDIAN 55mm): the old single 15mm gate falsely rejected feasible
    # thin-cylinder pinches near the table (chair posts/legs) that the soft solve rescues.
    self.collision_deep_reject = 0.05
    self.collision_rejects = 0                            # rejects by the depth gate
    self._ik_cache = {}                                   # (action, scene sig) -> (ok, witness)
    # WITNESS-OPTIMIZATION pull spec (post-pass only; None during search => every solve is
    # byte-identical to today). Set by optimize_witnesses: {'mode': 'joint'|'ee',
    # 'robot': r, 'qmap': {joint_name: prev_val}, 'ee_pos': xyz, 'ee_quat': wxyz,
    # 'weight': w, 'rot_weight': wr}. _add_pull injects a LOW-WEIGHT sos objective pulling
    # the ACTIVE arm toward its previous configuration (joint) or its previous EE pose
    # (position + weighted orientation) -- costs, never constraints, so feasibility never
    # yields to smoothness.
    self._opt_pull = None
    # PRE-SKILL PIN spec (post-pass only; None during search + export => byte-identical).
    # {joint_name: value}: _add_pin turns it into ONE hard qItself eq at the skill weight, so the
    # named joints are FROZEN for that solve. This is the term _opt_pull cannot be: a sos cost at
    # weight 1e0 against eq weights of 1e1 never stops a free body from translating, an eq does.
    # fuse_pre_skill_witnesses uses it to hold every arm that is NOT supposed to move during a
    # pre-* approach, so the funnel offset is paid for by the ACTOR instead of by the world.
    self._pin_joints = None
    self.fail_recorder = None                             # viser recorder for failed binds
    self.fail_log_cap = 300
    self.fail_log_seconds = 60.0                          # window from the FIRST failure
    self.fail_log_stop = False                            # abort the search when done?
    self._fail_log_t0 = None

  def holders(self, state):
    # {robot: (grasped_part, grasp_id)} -- who holds what, at which grasp, in this state.
    # Contacts are frozensets, whose iteration order changes with Python's hash seed.  Keep
    # this mapping in the scene's canonical robot order: several downstream task builders
    # preserve dict iteration order when concatenating multi-robot goals, so leaking the set
    # order here made identical plan rebuilds produce different task signatures (and could
    # swap goal-vector slices between robots).
    found = {c[1]: (c[2], c[3]) for p in state.bodies()
             for c in sorted(state.supports.get(p, ())) if c[0] == "grasp"}
    return {r: found[r] for r in self.robots if r in found}

  def ee_target(self, robot, part, grasp_id, assembly):
    # the gripper's target pose relative to its grasped part: the part's generated grasp,
    # approach-corrected for THIS robot's EE.
    grasp = grasp_pose(assembly, part, grasp_id)
    return np.concatenate([grasp[:3], _quat_mul(grasp[3:], self.q_correction[robot])])

  def _add_grasp_maintenance(self, komo, state, assembly, phase=1):
    # GRASP MAINTENANCE: every grasp that scene_for does NOT enforce kinematically (i.e.
    # every BRACE: a holder of a table-parented group, or a secondary holder of a held
    # group) becomes an explicit objective -- the gripper pinned at its grasp, the arm a
    # co-optimized DOF. Without this, brace arms sit at HOME in the IK scene (obstacles in
    # the wrong place) and a moved group slides out of its braces' grippers.
    # `phase` exists only so fuse_cograsp_witnesses can ground the SAME objectives on the
    # second keyframe of a 2-phase KOMO; every single-phase caller keeps the default.
    for r, pt, tgt in self.maintenance_grasps(state, assembly):
      if self._pin_joints and any(n.startswith(r + "_") for n in self._pin_joints):
        # this arm is FROZEN by the pre-skill pin, and so is what it holds (only arms that do
        # NOT move with the skill are ever pinned), so the grasp is already maintained. Keeping
        # the eq would demand that a frozen pair drive a residual the witness itself only met to
        # ~1e-4 to zero -- an unsatisfiable constraint that makes the WHOLE solve infeasible
        # (measured: 2 of 4 assembles infeasible, and the other 2 thrown 1.5-1.8 rad off branch).
        continue
      komo.addObjective([phase, phase], ry.FS.positionRel, [self.gripper_frame[r], pt],
                        ry.OT.eq, [1e1], tgt[:3])
      komo.addObjective([phase, phase], ry.FS.quaternionRel, [self.gripper_frame[r], pt],
                        ry.OT.eq, [1e1], tgt[3:])

  def _add_grasp_objectives(self, komo, robot, gpart, ee_target, phase=1):
    # the NEW gripper pinned at its grasp on `gpart` (the add_grasp/handover reach). Split
    # out of _ik_add_grasp so fuse_cograsp_witnesses can ground it on either keyframe of a
    # 2-phase KOMO; the single-phase caller keeps the default.
    _gw = [self.grasp_eq_weight if self.grasp_eq_weight is not None else 1e1]
    komo.addObjective([phase, phase], ry.FS.positionRel, [self.gripper_frame[robot], gpart],
                      ry.OT.eq, _gw, ee_target[:3])
    komo.addObjective([phase, phase], ry.FS.quaternionRel, [self.gripper_frame[robot], gpart],
                      ry.OT.eq, _gw, ee_target[3:])

  def maintenance_grasps(self, state, assembly):
    # the (robot, part, ee_target) triples _add_grasp_maintenance pins. Split out so the
    # fused-witness re-validation can score the SAME residuals the objectives constrain.
    grp_of = {p: g for g in state.subassemblies() for p in g}
    out = []
    for r, (pt, gid) in self.holders(state).items():
      g = grp_of[pt]
      root = _group_root(state, g)
      if pt == root and state.parent.get(root) == ("gripper", r):
        continue                            # kinematic holder: enforced by the attach
      out.append((r, pt, self.ee_target(r, pt, gid, assembly)))
    return out

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

  def _restrict_seed(self, w, C_restricted, full_names):
    # inverse of _expand_witness: a FULL config `w` (ordered like full_names) -> the restricted
    # arm's joint sub-vector (ordered like C_restricted's joints), so a full-config warm start
    # can seed a single-arm KOMO.
    idx = {nm: i for i, nm in enumerate(full_names)}
    return np.array([w[idx[nm]] for nm in C_restricted.getJointNames()], dtype=float)

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
    import time as _time
    kind = action[0]
    if kind == "release":
      return True, None                                   # no new reach
    if kind == "inhand_reorient":
      # in-place skill: the arm keeps its config; the part rotates IN the gripper (no IK
      # target of its own -- downstream actions solve IK against the new grasp).
      return True, None
    self.prof["bind_calls"] += 1                          # A1: real IK binds only
    self._ctx_seeds = None                                # set by the skill's seed computation
    self._last_win = ""
    _bt0 = _time.perf_counter()
    # VERDICT + WITNESS CACHE on a discrete scene signature: the search re-probes the SAME
    # (action, scene) from many parent states, and a FAILED solve costs ~135ms (initial +
    # 12 restarts) vs ~5ms for a success -- measured 81 re-probed pick failures per 40
    # expansions on cross7-R3. Unlike the cheap checker's signature this one KEEPS robot
    # identities (fixed bases: who reaches matters) and the grasp contacts' robots. Caching
    # also makes the stochastic restarts deterministic within a solve (same query -> same
    # verdict), which the search prefers anyway.
    key = (action, self._scene_sig(s0))
    if self._opt_pull is None:                            # pulled re-solves are step-specific:
      hit = self._ik_cache.get(key)                       # never read/write the cache for them
      if hit is not None:
        self.prof["bind_cache_hits"] += 1
        return hit
    if kind == "pick":
      res = self._ik_pick(action, s0, s1, assembly, seed)
    elif kind == "place":
      res = self._ik_place(action, s0, s1, assembly, seed)
    elif kind == "assemble":
      res = self._ik_assemble(action, s0, s1, assembly, seed)
    else:                                                 # add_grasp / handover: 2nd arm
      res = self._ik_add_grasp(action, s0, s1, assembly, seed)
    self.prof["bind_solves"] += 1
    self.prof["bind_success" if res[0] else "bind_fail"] += 1
    _dt = _time.perf_counter() - _bt0
    self.prof["bind_time"] += _dt
    # A1: per-(kind, part, grasp) attribution -- which actions burn the bind wall time.
    _ak = (kind, action[2] if len(action) > 2 else None,
           action[3] if len(action) > 3 else None)
    _e = self.prof["by_action"].setdefault(_ak, [0, 0, 0.0])
    _e[0] += 1
    if not res[0]:
      _e[1] += 1
    _e[2] += _dt
    if self._ctx_seeds is not None:                       # analytic-refutation study
      _ck, _cov, _ctool = self._ctx_seeds
      _k = f"{_ck}/{_ctool}"
      if res[0]:
        if self._last_win.startswith("seed"):
          _k += "/ok_seed"                                # exact branch certified the bind
        elif _cov:
          _k += "/slack_rescued"                          # exact branches existed but all failed
        else:
          _k += "/exactless_success"                      # won with NO exact solution at all
      else:
        _k += "/refutable_fail" if _cov else "/uncovered_fail"
      self.ana_stats[_k] = self.ana_stats.get(_k, 0) + 1
    if self._opt_pull is None:
      self._ik_cache[key] = res
    return res

  def probe_bind_once(self, action, s0, s1, assembly, attempt=0):
    """One bounded probe for the opt-in open-witness scheduler.

    This deliberately does not alter ``bind``'s legacy semantics.  A successful cached bind
    remains reusable, but a failed verdict is evicted before and after the probe so stochastic
    incompleteness cannot turn into a permanent edge blacklist.  Fixed-target analytic seeds
    are tried as one finite bundle on visit zero.  Later visits make new progress with fresh
    ``initRandom`` draws instead of replaying deterministic initializations.  Assemble probes
    spend the batch across structured biases first, then fresh unbiased initializations.
    """
    return self.probe_bind_batch(action, s0, s1, assembly, attempt=attempt, budget=1)

  def probe_bind_batch(self, action, s0, s1, assembly, attempt=0, budget=1):
    """One OPEN visit containing a bounded batch, with scene/precheck setup paid once."""
    budget = max(1, int(budget))
    key = (action, self._scene_sig(s0))
    cached = self._ik_cache.get(key)
    if cached is not None and cached[0]:
      return cached
    self._ik_cache.pop(key, None)                         # never inherit a legacy false verdict
    old_full, old_pre = self.restarts_full, self.restarts_precheck
    old_probe = self._open_probe_attempt
    old_budget = getattr(self, "_open_probe_budget", 1)
    old_random = getattr(self, "_open_probe_random_only", False)
    # Visit zero includes the ordinary initialization, hence B-1 random restarts.  A revisit
    # is random-only and _solve_ik below turns the whole B into new initRandom draws.
    self.restarts_full = max(0, budget - 1)
    self.restarts_precheck = 0
    self._open_probe_attempt = int(attempt)
    self._open_probe_budget = budget
    # Revisited ordinary skills must make new progress.  Assemble owns a structured bias cursor
    # and switches this on only for its unbiased fallback after the remaining biases are tried.
    self._open_probe_random_only = (int(attempt) > 0 and action[0] != "assemble")
    try:
      result = self.bind(action, s0, s1, assembly)
    finally:
      self.restarts_full, self.restarts_precheck = old_full, old_pre
      self._open_probe_attempt = old_probe
      self._open_probe_budget = old_budget
      self._open_probe_random_only = old_random
    if not result[0]:
      self._ik_cache.pop(key, None)                       # OPEN, not disproven
    return result

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

  def prof_report(self):
    # A1 INSTRUMENTATION summary: where real-IK time went. Pure print of self.prof (see
    # __init__); stage buckets include the reach precheck's relaxed solves, so
    # seeded+unseeded+restart time ~= the NLP-CPU total across precheck AND full solves.
    p = self.prof
    nlp_n = p["seeded_solves"] + p["unseeded_solves"] + p["restart_solves"]
    nlp_t = p["seeded_time"] + p["unseeded_time"] + p["restart_time"]
    print(f"[ik-prof] binds: {p['bind_calls']} called, {p['bind_cache_hits']} cache hits "
          f"({100 * p['bind_cache_hits'] / max(1, p['bind_calls']):.0f}%), "
          f"{p['bind_solves']} solved ({p['bind_success']} ok / {p['bind_fail']} failed), "
          f"bind wall {p['bind_time']:.1f}s")
    print(f"[ik-prof] NLP solves: {nlp_n} in {nlp_t:.1f}s "
          f"(seeded {p['seeded_solves']}/{p['seeded_time']:.1f}s, "
          f"unseeded {p['unseeded_solves']}/{p['unseeded_time']:.1f}s, "
          f"restarts {p['restart_solves']}/{p['restart_time']:.1f}s)")
    if nlp_n:
      print(f"[ik-prof] per-solve avg {1000 * nlp_t / nlp_n:.0f}ms; restart share of NLP "
            f"time {100 * p['restart_time'] / max(nlp_t, 1e-9):.0f}%")
    print(f"[ik-prof] reach precheck: {p['precheck_calls']} relaxed solves in "
          f"{p['precheck_time']:.1f}s ({self.precheck_rejects} rejects caught); "
          f"post-solve gates {p['post_calls']}x in {p['post_time']:.1f}s; "
          f"getPath calls {p['getpath_calls']}")
    print(f"[ik-prof] _solve_ik calls with seeds: {p['calls_with_seeds']}, without: "
          f"{p['calls_without_seeds']} "
          f"({100 * p['calls_without_seeds'] / max(1, p['calls_with_seeds'] + p['calls_without_seeds']):.0f}% unseeded)")
    if p["wins"]:
      wins = ", ".join(f"{k}={v}" for k, v in sorted(p["wins"].items()))
      print(f"[ik-prof] winning attempt distribution: {wins}")
    ba = p["by_action"]
    if ba:
      print("[ik-prof] top actions by bind time (kind/part/grasp: n (failed) time):")
      for (kind, part, gid), (n, nf, t) in sorted(ba.items(), key=lambda kv: -kv[1][2])[:12]:
        print(f"[ik-prof]   {kind:9s} {part} g{gid}: {n}x ({nf} failed) {t:.1f}s")
    if self.ana_stats:
      print("[ik-prof] analytic-coverage outcomes (decides ssik-refutation viability):")
      for k in sorted(self.ana_stats):
        print(f"[ik-prof]   {k}: {self.ana_stats[k]}")

  def _holders_of_group(self, state, group):
    # robots grasping any part of `group`
    return [r for r, (pt, _gid) in self.holders(state).items() if pt in group]

  def candidate_witnesses(self, action, s0, s1, assembly, resolves=4, limit=8):
    # CANDIDATE HARVEST for the ladder-DP witness optimization: multiple feasible
    # witnesses for one step. Sources: the cached original (the feasibility anchor);
    # for ASSEMBLE every orientation bias of the existing sweep (real diversity: each
    # bias settles the held assembly differently); for the fixed-target skills a few
    # fresh re-solves (analytic seeds first as always; random restarts explore distinct
    # basins run-to-run). Every candidate is feasible by construction (the solver's own
    # verdict). Deduped on rounded joints.
    kind = action[0]
    if kind not in _IK_ACTIONS:
      return []
    cands = []
    ok, w = self.bind(action, s0, s1, assembly)           # original (cached)
    if ok and w is not None:
      cands.append(w)
    if kind == "assemble":
      for bias in _ASSEMBLE_BIAS_ORIENTATIONS:
        ok, w = self._ik_assemble(action, s0, s1, assembly, None, bias_axes=bias)
        if ok and w is not None:
          cands.append(w)
    else:
      fn = {"pick": self._ik_pick, "place": self._ik_place}.get(kind, self._ik_add_grasp)
      # HARVEST EVERY ANALYTIC BRANCH ONCE, then top up with restarts. Without it the
      # fixed-target skills return the SAME first-feasible branch every call -- measured, a solo
      # pick gave ~3 candidates clustered within 14cm, so a ladder had no real choice.
      # OFF BY DEFAULT: this widens the pool the EXISTING dp-joint/dp-ee ladder consumes, and
      # that path's behaviour is measured. witness_floor turns it on for its own ladder.
      if self.harvest_all_branches:
        self.harvest_branches = []
        try:
          fn(action, s0, s1, assembly, None)
          cands.extend(self.harvest_branches)
        finally:
          self.harvest_branches = None
      for _ in range(resolves):
        ok, w = fn(action, s0, s1, assembly, None)
        if ok and w is not None:
          cands.append(w)
    uniq, seen = [], set()
    for w in cands:
      key = tuple(np.round(np.asarray(w, dtype=float), 3))
      if key not in seen:
        seen.add(key)
        uniq.append(w)
    return uniq[:limit]

  def branch_witnesses(self, action, s0, s1, assembly, base_witness, movers, *,
                       max_seeds=12, time_budget=6.0, min_dist=0.05):
    """ACTIVE IK-branch enumeration for --witness-sets: same-tool-pose alternatives for a
    step whose canonical witness is `base_witness` (a FULL joint vector over self.C's joints
    -- in practice bind's own witness, the base every emitted set member is built on).

    WHY THIS EXISTS. candidate_witnesses harvests whatever the skill solves happen to find,
    and that pool is STARVED of branch diversity: without ssik the analytic enumerator
    returns nothing, KOMO restarts land in the canonical basin, so picks offer 1-2 near-
    identical candidates (measured on cross: every step's pool within 0.05 rad of the
    canonical). Branches must be AIMED AT: for each acting arm, construct branch_seeds from
    its canonical config and re-solve THAT ARM ALONE for its canonical tool pose.

    CONDITIONED ENUMERATION -- what is frozen. The scene is scene_for(s0) posed at the FULL
    base witness, so every other arm and every part sits exactly where the canonical witness
    puts them; the varied arm is restricted to its own subtree (everything else is a frozen
    obstacle). Its tool pose is pinned to the canonical gripper WORLD pose, so:
      * a HOLDING arm's payload (attached to its gripper) lands at the canonical pose -- the
        object never moves, which is what makes sets safe for handovers;
      * a REACHING arm keeps the canonical grasp relation.
    The base MUST be the emitted canonical MEMBER (wit[i]), not bind's witness: at a handover
    the DP may legitimately choose a candidate that holds the object elsewhere than bind did
    (measured on cross steps 19/33: 717mm apart -- bind's own emitted config was penetrating
    and the least-bad full-scene candidate won), and members enumerated against bind's poses
    would then disagree with member 0 about where the "frozen" object is.
    The orientation is pinned via TWO WORLD AXES (vectorX + vectorZ), not a quaternion eq: a
    pi-flipped or 2*pi-wound branch reaches the same pose with the NEGATED quaternion, and a
    quaternion eq would score that exact solution as residual 2 and drive the solve back to
    the canonical basin.

    Acceptance mirrors _solve_ik: NLP feasible + _pen_ok + _clearance_ok, then a novelty
    check (>= min_dist from every kept config -- a seed that converged back to a known basin
    is not a find). branch_seeds' `exact` whole-turn shifts skip the solve: a full turn of
    one revolute joint moves no body, so they are pose- and collision-identical by identity.

    Returns (alts, funnel): alts = [{joint_name: value}] in the same per-candidate shape
    witness_floor's pool uses, so the existing gates + greedy selection apply unchanged.
    Each map covers EVERY mover arm's joints -- the varied arm at its new branch, the other
    movers at their base-witness values -- because the emission overwrites a candidate's
    joints onto BIND's witness: a map naming only the varied arm would let a multi-arm
    step's partner fall back to bind's (possibly different) config and break the very
    consistency the set promises. funnel = counters {seeds, solved, gated, exact, kept}.
    Called ONLY from the witness-sets block (witness_sets >= 2): flag off => never runs."""
    import time as _time
    t0 = _time.perf_counter()
    fun = {"seeds": 0, "solved": 0, "gated": 0, "exact": 0, "kept": 0}
    alts = []
    base_witness = np.asarray(base_witness, dtype=float).ravel()
    C0 = self.scene_for(s0, assembly)
    if base_witness.size != len(C0.getJointNames()):
      return alts, fun
    C0.setJointState(base_witness)
    # intended-contact exemptions, as the skill scenes set them: every gripper in the
    # canonical witness is AT its grasp (fingers around the part), so each grasped part's
    # contact must be off or the collision ineq carries a constant violation no candidate
    # can fix. Union over both states (a pick's part is grasped only in s1).
    for st in (s0, s1):
      if st is None:
        continue
      for _r, (pt, _g) in self.holders(st).items():
        if not self.keep_contacts:
          set_part_contact(C0, pt, 0)
    names0 = C0.getJointNames()
    # every mover arm's base-witness values ride along in each returned map (see docstring)
    mover_map = {n: float(base_witness[k]) for k, n in enumerate(names0)
                 if any(n.startswith(rr + "_") for rr in movers)}
    for r in sorted(movers):
      if self.robot_arm.get(r) is None:
        continue                                       # floating EE: no arm, no branches
      if _time.perf_counter() - t0 > time_budget or fun["seeds"] >= max_seeds:
        break
      Cr = ry.Config()
      Cr.addConfigurationCopy(C0)                      # contact flags + posed state ride along
      full_names = Cr.getJointNames()
      Cr.selectJointsBySubtree(Cr.getFrame(self.robot_base[r]))
      rnames = Cr.getJointNames()
      if len(rnames) != 6:
        continue                                       # not a plain 6R chain: out of scope
      q0 = self._restrict_seed(base_witness, Cr, full_names)
      lo, hi = (np.asarray(v, dtype=float) for v in Cr.getJointLimits())
      gf = Cr.getFrame(self.gripper_frame[r])
      p_t = np.asarray(gf.getPosition(), dtype=float)
      R_t = np.asarray(gf.getRotationMatrix(), dtype=float)
      exact, seeds = branch_seeds(q0, lo, hi, max_seeds=max_seeds)
      kept_here = [q0]
      for q in exact:                                  # pose/collision-identical by identity
        fun["exact"] += 1
        if min(float(np.linalg.norm(q - k)) for k in kept_here) < min_dist:
          continue
        m = dict(mover_map)
        m.update({n: float(v) for n, v in zip(rnames, q)})
        alts.append(m)
        kept_here.append(q)
        fun["kept"] += 1
      if not seeds:
        continue
      komo = ry.KOMO(Cr, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=True)
      komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq,
                        [self.col_weight], [-0.0])
      komo.addControlObjective([], 0, 1e-1)
      komo.addControlObjective([], 1, 1e-1)
      ee = self.gripper_frame[r]
      komo.addObjective([1, 1], ry.FS.position, [ee], ry.OT.eq, [1e1], p_t)
      komo.addObjective([1, 1], ry.FS.vectorX, [ee], ry.OT.eq, [1e1], R_t[:, 0])
      komo.addObjective([1, 1], ry.FS.vectorZ, [ee], ry.OT.eq, [1e1], R_t[:, 2])
      for q in seeds:
        if _time.perf_counter() - t0 > time_budget or fun["seeds"] >= max_seeds:
          break
        fun["seeds"] += 1
        komo.initWithConstant(q)
        res = self._nlp(komo).solve().dict()
        if not res["feasible"]:
          continue
        fun["solved"] += 1
        qs = np.asarray(komo.getPath()[-1], dtype=float).ravel()
        if not (self._pen_ok(Cr, qs) and self._clearance_ok(Cr, qs)):
          fun["gated"] += 1
          continue
        if min(float(np.linalg.norm(qs - k)) for k in kept_here) < min_dist:
          continue                                     # converged back to a known basin
        m = dict(mover_map)
        m.update({n: float(v) for n, v in zip(rnames, qs)})
        alts.append(m)
        kept_here.append(qs)
        fun["kept"] += 1
    return alts, fun

  def _add_pull(self, komo, C, robot):
    # inject the witness-optimization pull (no-op unless optimize_witnesses set _opt_pull
    # for THIS robot). Weights ~1e0 vs constraint weights 1e1: cost never beats feasibility.
    p = self._opt_pull
    if p is None or p.get("robot") != robot:
      return
    if p["mode"] == "joint":
      names = C.getJointNames()
      cur = C.getJointState()
      target = np.array([p["qmap"].get(n, cur[i]) for i, n in enumerate(names)])
      komo.addObjective([1, 1], ry.FS.qItself, [], ry.OT.sos, [p["weight"]], target)
    else:                                                 # 'ee': position + weighted quat
      ee = self.gripper_frame[robot]
      komo.addObjective([1, 1], ry.FS.position, [ee], ry.OT.sos, [p["weight"]],
                        np.asarray(p["ee_pos"], dtype=float))
      komo.addObjective([1, 1], ry.FS.quaternion, [ee], ry.OT.sos,
                        [p["weight"] * p["rot_weight"]],
                        np.asarray(p["ee_quat"], dtype=float))

  def _add_pin(self, komo, C):
    # inject the pre-skill joint PIN (no-op unless fuse_pre_skill_witnesses set _pin_joints).
    # Names absent from C are dropped, which is what makes this a no-op in a _restrict_to_arm
    # scene: there the other arms are already frozen, so there is nothing left to pin.
    # NOTE the four RELAXED reach prechecks are skipped outright while a pin is set (see their
    # call sites): that gate rejects on the LAST restart's residual against a threshold measured
    # on UN-pinned solves, and a random restart cannot satisfy a hard joint pin -- measured, it
    # threw away pinned solves that the full solve then found feasible.
    p = self._pin_joints
    if not p:
      return
    sel = [n for n in C.getJointNames() if n in p]
    # Record what the pin actually bit on. An EMPTY sel means the scene had already frozen
    # every joint we were asked to hold, so the solve that follows is the ordinary un-pinned
    # one -- the caller must not present its outcome as a repair (measured: every `pick` takes
    # this path, and 4 of 6 "accepted" splits were just solver noise on two draws of it).
    self.pin_applied = len(sel)
    if not sel:
      return
    komo.addObjective([1, 1], ry.FS.qItself, sel, ry.OT.eq, [1e1],
                      [p[n] for n in sel])

  def mate_penetration(self, C, moving, target):
    """Total penetration of `moving` against everything EXCEPT its mate `target`.

    The mate scene has the moving group's contact DISABLED -- required for the solve, since the
    seating is flush and the intended contact would otherwise read as penetration -- but that
    also hides part-vs-arm and part-vs-other-part, which the motion planner very much sees.
    `getCollisionsTotalPenetration()` cannot express a per-PAIR exclusion (set_part_contact is
    per-frame, all-or-nothing), so the group's contact is re-enabled just long enough to sum the
    pairs by hand, and restored before returning.
    """
    for mp in moving:
      set_part_contact(C, mp, 1)
    try:
      pen = 0.0
      for c in C.getCollisions():
        fa, fb, d = c[0], c[1], float(c[2])
        if d >= 0.0:
          continue
        a_in = any(fa.startswith(m) for m in moving)
        b_in = any(fb.startswith(m) for m in moving)
        if a_in and b_in:
          continue                                        # inside the moving group itself
        if a_in != b_in:
          other = fb if a_in else fa
          if any(other.startswith(t) for t in target):
            continue                                      # the INTENDED flush mate contact
        # EVERYTHING ELSE COUNTS -- including pairs the moving group is not part of at all.
        # An earlier version skipped those ("if a_in == b_in: continue"), which silently
        # dropped the ACTOR'S ARM against a part of the TARGET assembly. Measured: the exported
        # keyframe 18_assemble_a1_obj_4_obj_0 had obj_2 (already mated onto obj_0) 4.74mm
        # inside a1_ur_palm, and this gate scored it 0.00mm. The only contact a mate may make
        # is moving<->target; every other overlap is a real collision.
        pen += -d
      return pen
    finally:
      if not self.keep_contacts:
        for mp in moving:
          set_part_contact(C, mp, 0)

  def grasp_pose_residual(self, C, q, robot, part, ee_target):
    """|achieved positionRel(gripper, part) - the pinned grasp-DB target|, metres.

    mr_bench's attach and the sim's grasp weld FREEZE the ACHIEVED relation at the pick
    transition, while every LATER witness is solved with the part at the DB relation
    (scene_for reconstructs held parts from the grasp id). So any slack rai's feasible
    verdict tolerates here becomes a RIGID part-in-hand error downstream -- measured ~5mm
    along the approach on sampled grasps (the table's collision cost pushes the hand out),
    which surfaced as a 1mm seat penetration mr_bench cannot exempt (two held parts: the
    two picks' residuals DIFFER, and the difference lands in the mate). Face-center
    enumerator grasps land on the DB pose, which is why this stayed invisible. Same
    contract as mate_residual."""
    saved = C.getJointState()
    try:
      C.setJointState(np.asarray(q, dtype=float))
      v, _ = C.eval(ry.FS.positionRel, [self.gripper_frame[robot], part])
      return float(np.linalg.norm(np.asarray(v, dtype=float)
                                  - np.asarray(ee_target[:3], dtype=float)))
    finally:
      C.setJointState(saved)

  def mate_residual(self, C, q, part_name, onto, rel_pos):
    """|achieved positionRel(part, onto) - target|, in metres. This is what mr_bench bakes in at
    attach (C.attach keeps the WORLD pose at the mode switch), so an under-seated mate makes the
    exported assembly disagree with the scene the witness was solved in -- and the acting arm
    then collides with an already-mated part."""
    saved = C.getJointState()
    try:
      C.setJointState(np.asarray(q, dtype=float))
      v, _ = C.eval(ry.FS.positionRel, [part_name, onto])
      return float(np.linalg.norm(np.asarray(v, dtype=float) - np.asarray(rel_pos, dtype=float)))
    finally:
      C.setJointState(saved)

  def _nlp(self, komo):
    """The ONE place an NLP_Solver is built, so `nlp_stop_tolerance` cannot be applied to some
    solve paths and not others (there are three inside _solve_ik alone)."""
    s = ry.NLP_Solver(komo.nlp(), verbose=0)
    if self.nlp_stop_tolerance is not None:
      s.setOptions(stopTolerance=float(self.nlp_stop_tolerance))
    return s

  def _pen_ok(self, C, q):
    # --strict-collisions gate: total penetration of the skill scene at witness q (the same
    # contact flags the komo's collision objective saw -- target-part exclusions included).
    if self.penetration_reject is None or C is None or q is None:
      return True
    saved = C.getJointState()
    C.setJointState(np.asarray(q, dtype=float))
    if self._mate_exempt is None:
      pen = float(C.getCollisionsTotalPenetration())
    else:
      pen = self.mate_penetration(C, *self._mate_exempt)
    reject = pen > self.penetration_reject
    if not reject and getattr(self, "_funnel_solve", False):
      # FUNNEL-MOUTH solves (--pre-* backoffs): the payload/arm may not penetrate the TABLE.
      # The strict gate bounds only TOTAL penetration, and the margin gate exempts part/table
      # pairs (parts legitimately rest on the table at d == 0), so a backoff pointing into the
      # tabletop was accepted -- measured on the tight stool: a pre-insertion whose mate sits
      # close to the table sank the leg into the tabletop. Gate every table pair at a hard
      # near-zero tolerance; a rejected rung walks the delta ladder in pre_skill_witness.
      for _a, _b, d in C.getCollisions():
        if (str(_a) == "table" or str(_b) == "table") and d < -1e-4:
          reject = True
          break

    if (self._pen_view_action is not None and not self._pen_viewed
        and self._cur_action is not None
        and " ".join(str(x) for x in self._cur_action) == self._pen_view_action):
      # DEBUG (PEN_VIEW_ACTION): show the checked witness + its colliding pairs, BLOCKING.
      # self._pen_viewed = True
      pairs = sorted((c for c in C.getCollisions() if c[2] < 0), key=lambda c: c[2])
      print(f"[pen-view] {tuple(self._cur_action)}: total penetration {pen*1000:.2f}mm "
            f"({'REJECTED' if reject else 'accepted'}; gate {self.penetration_reject*1000:.2f}mm)")
      for a, b, d in pairs[:15]:
        print(f"   {a} ~ {b}: {d*1000:+.2f}mm")
      if not pairs:
        print("   (no penetrating pairs -- penetration is from another source / rounding)")
      C.view(True, f"pen check {tuple(self._cur_action)}: {pen*1000:.1f}mm "
                   f"{'REJECT' if reject else 'ok'} (close to continue)")

    C.setJointState(saved)
    if reject:
      self.strict_rejects += 1
      return False
    return True

  _HAND_TOKENS = ("gripper", "palm", "finger", "robotiq", "vacuum")

  def _arm_of_frame(self, frame):
    for r in self.robots:
      if str(frame).startswith(r + "_") or str(frame) == r:
        return r
    return None

  @staticmethod
  def _part_of_frame(frame):
    f = str(frame)
    if not f.startswith("obj_"):
      return None
    # Compound collision geometry is named obj_N__shape/...; the logical part is obj_N.
    return f.split("__", 1)[0]

  def _is_hand_frame(self, frame):
    return any(t in str(frame) for t in self._HAND_TOKENS)

  def _part_is_payload_of(self, C, part, robot):
    """Whether ``part`` sits below ``robot``'s gripper in this skill scene.

    scene_for attaches a held assembly root to its kinematic holder and the remaining members
    below that root. Walking parents therefore recognises the WHOLE payload, not only the exact
    frame named by the grasp. Failure is conservative: an uninspectable ancestry is not an
    intended contact.
    """
    if part is None or robot is None:
      return False
    try:
      f, seen = C.getFrame(part), set()
      while f is not None and f.name not in seen:
        if f.name == self.gripper_frame.get(robot):
          return True
        seen.add(f.name)
        f = f.getParent()
    except Exception:
      return False
    return False

  def _intended_hand_part(self, C, hand_frame, part_frame, robot):
    """Pair-specific grasp exemption; unrelated hand/part proximity remains gated."""
    part = self._part_of_frame(part_frame)
    if part is None or robot is None or not self._is_hand_frame(hand_frame):
      return False
    if self._part_is_payload_of(C, part, robot):
      return True
    action = tuple(self._cur_action or ())
    return (len(action) >= 3 and action[0] in ("pick", "add_grasp", "handover")
            and action[1] == robot and action[2] == part)

  def _clearance_pair_reason(self, C, fa, fb):
    """Return the hard-gate class for a pair, or None when it is out of scope/intended."""
    fa, fb = str(fa), str(fb)
    ra, rb = self._arm_of_frame(fa), self._arm_of_frame(fb)
    pa, pb = self._part_of_frame(fa), self._part_of_frame(fb)

    if ra is not None and rb is not None:
      if ra != rb:
        return "inter-robot"
      # rai's collidable-pair set already removes adjacent kinematic geometry. Keep the
      # gripper's own internal mechanics exempt, but gate every other reported same-arm pair
      # (including the measured ur_coll8~ur_coll5 Cube failure).
      if self._is_hand_frame(fa) and self._is_hand_frame(fb):
        return None
      return "self"

    if ra is not None and pb is not None:
      if self._intended_hand_part(C, fa, fb, ra):
        return None
      return "arm-part"
    if rb is not None and pa is not None:
      if self._intended_hand_part(C, fb, fa, rb):
        return None
      return "arm-part"
    return None

  def _clearance_ok(self, C, q):
    """Shared search/witness hard clearance gate (see ``clearance_reject`` above)."""
    M = self.clearance_reject
    if M is None or M <= 0.0 or C is None or q is None:
      return True
    saved = C.getJointState()
    C.setJointState(np.asarray(q, dtype=float))
    bad = None
    for c in C.getCollisions(M):
      fa, fb, d = c[0], c[1], float(c[2])
      if d >= M:
        continue
      reason = self._clearance_pair_reason(C, fa, fb)
      if reason is not None:
        bad = (str(fa), str(fb), d, reason)
        break
    C.setJointState(saved)
    if bad is not None:
      self.margin_rejects += 1
      self.last_margin_reject = bad
      key = (bad[3], bad[0], bad[1])
      self.margin_reject_pairs[key] = self.margin_reject_pairs.get(key, 0) + 1
      return False
    return True

  def _solve_ik(self, komo, restarts=None, seeds=None, C=None):
    # solve a skill KOMO, retrying from fresh inits until one converges. KOMO seeds its guess
    # with a little noise, so a borderline-reachable target flips feasible/infeasible run-to-
    # run; restarts make bind reliable -- a genuinely feasible action returns a witness (instead
    # of sometimes None, which would corrupt search + replay). Returns (feasible, witness joint
    # config or None). `seeds` (analytic ssik solutions, exact for the pose) are tried FIRST as
    # warm starts -- helps CONVERGENCE-limited reaches; note it does NOT help collision-limited
    # ones (an exact analytic pose may collide, whereas the KOMO's tolerant pose eq can settle on
    # a nearby collision-free pose the discrete IK solutions don't cover), hence the fallback.
    import time as _time
    if restarts is None:
      restarts = self.restarts_full                       # --ik-restarts (default 12)
    # An open-witness query is revisited by construction.  Visit zero tries the finite analytic
    # bundle + ordinary initialization.  Later ordinary-skill visits must make NEW progress:
    # replaying those deterministic initializations merely burns time, so use the tier's fresh
    # initRandom draws.  Assemble controls this switch around its unbiased fallback so a
    # structured bias is never accidentally expanded into a random batch.  Legacy calls have
    # _open_probe_attempt=None and are bit-identical.
    _open_random_only = (self._open_probe_attempt is not None
                         and getattr(self, "_open_probe_random_only", False)
                         and not getattr(self, "_pc", False))
    if _open_random_only:
      seeds = None
      restarts = max(1, int(self._open_probe_budget))
    if seeds:
      self.prof["calls_with_seeds"] += 1
    else:
      self.prof["calls_without_seeds"] += 1

    def _win(k):                                          # A1: winning-attempt histogram
      if getattr(self, "_pc", False):
        k = "pc_" + k                                     # precheck-sourced (relaxed solve)
      self.prof["wins"][k] = self.prof["wins"].get(k, 0) + 1
      self._last_win = k

    for _si, seed in enumerate(seeds or []):
      seed = np.asarray(seed, dtype=float)
      skey = (self._cur_action,
              tuple(C.getJointNames()) if C is not None else (), int(seed.size))
      if skey in self.seed_init_incompatible:
        break
      # KOMO's path exposes its INTERNAL q width.  It can be one larger than both
      # C.getJointState() and C.getJointNames() after the stable-joint reparenting described
      # above. Detect that before calling into rai: catching its setDofState exception is too
      # late because the C++ LogToken may subsequently abort while unwinding.
      path0 = np.asarray(komo.getPath())
      expected = int(path0.shape[-1]) if path0.ndim >= 2 and path0.size else seed.size
      if expected != seed.size:
        self.seed_init_incompatible.add(skey)
        self.seed_init_skips += 1
        break
      komo.initWithConstant(seed)
      import time as _time
      _t0 = _time.perf_counter()
      r = self._nlp(komo).solve().dict()
      self.prof["seeded_solves"] += 1
      self.prof["seeded_time"] += _time.perf_counter() - _t0
      if r["feasible"]:
        self.prof["getpath_calls"] += 1
        _t0 = _time.perf_counter()
        _okp = self._pen_ok(C, komo.getPath()[-1])
        _ok = _okp and self._clearance_ok(C, komo.getPath()[-1])
        self.prof["post_calls"] += 1
        self.prof["post_time"] += _time.perf_counter() - _t0
        if _ok:
          _win(f"seed{_si}")
          self._last_res = r
          if self.harvest_branches is not None:
            # CANDIDATE HARVEST: keep going. The analytic seed set ENUMERATES the arm's exact
            # pose branches (up to 8), but returning the FIRST feasible one is what makes the
            # witness-ladder's pool thin -- measured, solo picks yielded ~3 candidates all in one
            # basin, so the DP had nothing to choose between. Collect them all and let the ladder
            # pick; None (the default) restores the early return exactly.
            self.harvest_branches.append(komo.getPath()[-1])
            continue
          return True, komo.getPath()[-1]
    if self.harvest_branches:
      return True, self.harvest_branches[0]
    if seeds:
      # the analytic seed set ENUMERATES every exact pose branch (up to 8), so random restarts
      # can only re-derive branches already tried -- their residual value is exploring the eq
      # TOLERANCE slack (a nearby collision-free near-pose). A few tries cover that; the full
      # budget is for UNSEEDED solves where restarts must find the branches themselves.
      restarts = min(restarts, self.seed_restart_cap)
    import time as _time
    r = None
    if not _open_random_only:
      _t0 = _time.perf_counter()
      r = self._nlp(komo).solve().dict()
      self.prof["unseeded_solves"] += 1
      self.prof["unseeded_time"] += _time.perf_counter() - _t0
      if r["feasible"]:
        self.prof["getpath_calls"] += 1
        _t0 = _time.perf_counter()
        _okp = self._pen_ok(C, komo.getPath()[-1])
        _ok = _okp and self._clearance_ok(C, komo.getPath()[-1])
        self.prof["post_calls"] += 1
        self.prof["post_time"] += _time.perf_counter() - _t0
        if _ok:
          _win("unseeded")
          self._last_res = r
          return True, komo.getPath()[-1]
    for _ri in range(restarts):
      komo.initRandom()
      _t0 = _time.perf_counter()
      r = self._nlp(komo).solve().dict()
      self.prof["restart_solves"] += 1
      self.prof["restart_time"] += _time.perf_counter() - _t0
      if r["feasible"]:
        self.prof["getpath_calls"] += 1
        _t0 = _time.perf_counter()
        _okp = self._pen_ok(C, komo.getPath()[-1])
        _ok = _okp and self._clearance_ok(C, komo.getPath()[-1])
        self.prof["post_calls"] += 1
        self.prof["post_time"] += _time.perf_counter() - _t0
        if _ok:
          _win(f"restart{_ri}")
          self._last_res = r
          return True, komo.getPath()[-1]
    _win("none")
    if _os.environ.get("SOLVER_IK_DEBUG"):
      res_d = r or {}
      print(f"[ik-debug] FAIL {getattr(self, '_cur_action', '?')} "
            f"eq={res_d.get('eq', float('nan')):.3f} "
            f"ineq={res_d.get('ineq', float('nan')):.3f} "
            f"restarts={restarts} n_seeds={len(seeds or [])}")
    self._last_res = r or {}
    return False, None

  def _attached_arm_targets(self, C, s0, assembly, group, anchor_part, anchor_target, mover):
    # World tool poses of EVERY arm attached to `group` (the mover + any braces), at the group
    # pose that puts `anchor_part` at `anchor_target` (a 7-pose). The group is RIGID and the
    # target pose is FIXED, so each attached gripper's pose is fully determined:
    #   mover : rides the kinematic attach -> anchor_target o (anchor -> gripper, from scene)
    #   brace : pinned at its grasp by _add_grasp_maintenance -> anchor_target o
    #           (anchor -> grasped part) o ee_target(brace, part, grasp)
    # This is what makes a braced move DECOMPOSE per-arm for reachability: once the group pose
    # is pinned, the relaxed (collision-free) problem is exactly "each arm reaches its fixed
    # pose" -- there is no residual coupling between the arms.
    apose = np.asarray(C.getFrame(anchor_part).getPose(), dtype=float)
    ainv = _pose_invert(apose)
    gpose = np.asarray(C.getFrame(self.gripper_frame[mover]).getPose(), dtype=float)
    out = {mover: _pose_compose(anchor_target, _pose_compose(ainv, gpose))}
    gparts = set(group)
    for rr, (pt, gid) in self.holders(s0).items():
      if rr == mover or pt not in gparts:
        continue
      ptpose = np.asarray(C.getFrame(pt).getPose(), dtype=float)
      pt_land = _pose_compose(anchor_target, _pose_compose(ainv, ptpose))
      out[rr] = _pose_compose(pt_land, np.asarray(self.ee_target(rr, pt, gid, assembly),
                                                  dtype=float))
    return out

  def _compose_full_seeds(self, C, arm_seeds, cap=8):
    # Full joint-vector warm starts for a MULTI-ARM KOMO from per-arm analytic solutions:
    # each attached arm's joints overwritten with one of its solutions (cartesian product,
    # closest-to-current-config first, capped -- the product is the complete enumeration of
    # the exact pose branches when it fits under the cap).  Do not assume 6R here: OpenArm
    # and FR3 are 7R, and silently dropping their exact seeds is especially expensive for a
    # coupled place/assemble where the nonlinear solve otherwise has to discover every arm.
    import itertools
    q0 = np.asarray(C.getJointState(), dtype=float)
    names = C.getJointNames()
    idx = {r: [i for i, n in enumerate(names) if n.startswith(r + "_")] for r in arm_seeds}
    if any(not arm_seeds[r] or len(ix) != len(np.asarray(arm_seeds[r][0]).ravel())
           for r, ix in idx.items()):
      return []
    robots = sorted(arm_seeds)
    combos = []
    for combo in itertools.product(*(arm_seeds[r] for r in robots)):
      q = q0.copy()
      d = 0.0
      for r, qr in zip(robots, combo):
        q[idx[r]] = qr
        d += float(np.linalg.norm(q0[idx[r]] - qr))
      combos.append((d, q))
    combos.sort(key=lambda t: t[0])
    return [q for _, q in combos[:cap]]

  def _probe_branch_combos(self, C, arm_seeds, exclude=(), cap=8, probe_cap=256):
    # DEPTH-GATED collision probe over the FULL cartesian product of the involved arms'
    # exact analytic branches. Posing EVERY involved arm at one of its branches is a
    # CONSISTENT candidate solution of the whole skill (each tool at its true pose, the
    # moved group at its true target) -- so ALL collision pairs are evaluated at real
    # geometry, INCLUDING ARM-ARM between the placer and its brace (the dominant failure
    # mode seen in practice). Returns (reject, seeds):
    #   reject : every combo penetrates beyond its gate -> the bind has no near-pose rescue
    #            (only claimed when the FULL product was probed);
    #   seeds  : the `cap` best combos as full warm starts, shallowest penetration first
    #            (ties: closest to the current config) -- collision-INFORMED seeding.
    # Gates: `collision_reject_depth` vs the static scene (measured rescue ceiling ~10mm);
    # DOUBLED for involved-involved pairs (both arms can graze-shift under the tolerant eq,
    # so their relative clearance can change by the sum). `exclude` = robots whose pairs are
    # ignored: UNINVOLVED arms in a multi-arm KOMO are free DOFs that can dodge (in the
    # restricted single-arm case they are frozen AND contact-stripped, so exclude=()).
    # Self-collision pairs of an involved arm fall under the doubled involved-involved gate
    # (conservative -- rejecting them needs 2x the depth; a graze-shift reshapes the wrist).
    import itertools
    gate = self.collision_reject_depth
    q0 = np.asarray(C.getJointState(), dtype=float)
    names = C.getJointNames()
    idx = {r: [i for i, n in enumerate(names) if n.startswith(r + "_")] for r in arm_seeds}
    if (not arm_seeds or any(not s for s in arm_seeds.values())
        or any(len(ix) != len(np.asarray(arm_seeds[r][0]).ravel())
               for r, ix in idx.items())):
      return False, []
    robots = sorted(arm_seeds)
    n_combos = int(np.prod([len(arm_seeds[r]) for r in robots]))
    if gate is None or n_combos > probe_cap:
      return False, self._compose_full_seeds(C, arm_seeds, cap=cap)  # too many: no reject
    xpre = tuple(o + "_" for o in exclude)
    ipre = tuple(r + "_" for r in robots)
    scored = []
    try:
      for combo in itertools.product(*(arm_seeds[r] for r in robots)):
        qq = q0.copy()
        d0 = 0.0
        for r, qr in zip(robots, combo):
          qq[idx[r]] = qr
          d0 += float(np.linalg.norm(q0[idx[r]] - np.asarray(qr)))
        C.setJointState(qq)
        C.computeCollisions()
        excess = 0.0                                      # penetration beyond the pair's gate
        deep = 0.0                                        # ... beyond the HARD-reject gate
        for fa, fb, d in C.getCollisions(0.0):
          if d >= 0.0 or (xpre and (fa.startswith(xpre) or fb.startswith(xpre))):
            continue
          both = fa.startswith(ipre) and fb.startswith(ipre)
          g = 2.0 * gate if both else gate
          gd = 2.0 * self.collision_deep_reject if both else self.collision_deep_reject
          excess = max(excess, -d - g)
          deep = max(deep, -d - gd)
        scored.append((excess, d0, qq, deep))
    finally:
      C.setJointState(q0)
    # HARD reject only when every combo penetrates beyond the DEEP gate (~the measured
    # unrescued median). The shallow gate stays a RANKING signal: the chair's thin table-
    # adjacent cylinder pinches are rescued by the soft KOMO from well beyond 15mm, so the
    # 15mm band must fall through to the solve (with the ranked seeds), never hard-reject --
    # the cheap layers must not reject what the full solve can do.
    if all(s[3] > 0.0 for s in scored):
      self.collision_rejects += 1
      return True, []                                     # every exact combo VERY deeply colliding
    scored.sort(key=lambda t: (t[0], t[1]))
    top = scored[:cap]
    # COLLISION-PRECHECKED SEEDING: a combo already measured beyond the HARD gate
    # (collision_deep_reject, beyond the observed rescue ceiling) is not worth a KOMO
    # attempt -- drop it from the warm-start set instead of paying a doomed solve.
    kept = [q for _e, _d0, q, deep in top if deep <= 0.0]
    if not kept:                                          # cannot happen after the all-deep
      kept = [scored[0][2]]                               # ... reject above, but stay safe
    return False, kept

  def _reach_bound(self, robot):
    # SOUND kinematic max-reach for ANY arm (no analytic IK required): the tool centre can
    # never be farther from the arm base than the SUM of the fixed link translations along
    # the base_link -> gripper chain (revolute joints only reorient those rigid offsets, so
    # ||p_ee - p_base|| <= sum ||t_i||). An OVERESTIMATE of the true max reach (the offsets
    # are not colinear at full stretch), so a target beyond it is unreachable FOR SURE.
    # inf when the robot has no arm chain (floating EE) -> the filter never fires.
    b = self._reach_bounds.get(robot)
    if b is not None:
      return b
    b = np.inf
    try:
      f = self.C.getFrame(self.gripper_frame[robot])
      base = self.base_link[robot]
      total = 0.0
      while f is not None and f.name != base:
        total += float(np.linalg.norm(f.getRelativePosition()))
        f = f.getParent()
      if f is not None:                                   # reached base_link -> chain complete
        b = total
    except Exception:
      pass
    self._reach_bounds[robot] = b
    return b

  def _reach_reject(self, C, robot, p_world):
    # instant (~us) gross-unreachability reject for a FIXED tool target position. Sound for
    # every robot (pure kinematics, model-derived); catches only what NO configuration can
    # reach -- the analytic reject (when calibrated) subsumes it with exact geometry.
    b = self._reach_bound(robot)
    if not np.isfinite(b):
      return False
    base = np.asarray(C.getFrame(self.base_link[robot]).getPosition(), dtype=float)
    if np.linalg.norm(np.asarray(p_world, dtype=float) - base) <= b:
      return False
    self.reach_rejects += 1
    return True

  def _analytic_reject_ok(self, robot):
    # may we use analytic IK for an EXACT reachability REJECT on a FIXED-target reach? Only when
    # seeding is on and ssik has this exact ARM + EE calibrated. Vacuum uses the canonical
    # aligned roll already present in the target pose; an empty result remains UNKNOWN (the
    # relaxed precheck handles it), while a nonempty result proves that fixed pose reachable.
    if not self.ik_seeding:
      return False
    import analytic_ik
    return analytic_ik.available(self.robot_arm[robot], self.tool_kind[robot])

  def _analytic_seeds(self, C, robot, ee_target, target_frame):
    # analytic (ssik) joint solutions for the active arm to place its tool center at `ee_target`
    # (a 7-pose relative to `target_frame`; target_frame=None => ee_target is already a WORLD
    # pose, e.g. a place's computed gripper pose at the group's landing spot), in RAI joint
    # order, filtered to rai's arm limits. Empty unless ik_seeding is on AND this arm is
    # calibrated. Call AFTER _restrict_to_arm when possible, so C.getJointLimits() already
    # returns just this arm's limits (6R UR or 7R OpenArm/FR3).
    import analytic_ik
    tool_kind = self.tool_kind[robot]
    if not self.ik_seeding or not analytic_ik.available(self.robot_arm[robot], tool_kind):
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
      # selectJointsBySubtree updates RAI's active-joint bookkeeping lazily.  Asking for
      # limits first can therefore return shape (2, 0); names/state materializes the newly
      # selected set.  This order matters in every restricted single-arm OpenArm solve.
      joint_names = list(C.getJointNames())
      lo, hi = C.getJointLimits()
      # RAI exposes borrowed array views here; own them before crossing into the generated
      # ssik solver.  Together with the materialization above this prevents all 128
      # mathematical 7R branches from bypassing the physical-limit filter.
      lo, hi = np.array(lo, dtype=float, copy=True), np.array(hi, dtype=float, copy=True)
      robot_idx = [i for i, n in enumerate(joint_names) if n.startswith(robot + "_")]
      # UNRESTRICTED scene (e.g. an assemble reject probe): select this robot's arm limits
      # by joint-name prefix.  The old literal 6 made every OpenArm seed bypass its physical
      # limits, only to be rejected later by KOMO, and made multi-arm analytic seeding appear
      # unavailable for all 7R skills.
      if len(lo) != len(robot_idx):
        if robot_idx and max(robot_idx) < len(lo):
          lo, hi = lo[robot_idx], hi[robot_idx]
        else:
          lo = hi = None
    except Exception:
      lo = hi = None
    out = []
    for q in analytic_ik.seed_configs(self.robot_arm[robot], T, tool_kind=tool_kind):
      if lo is not None and len(q) == len(lo) and (np.any(q < lo) or np.any(q > hi)):
        continue                                            # outside rai's joint limits
      out.append(q)
    return out

  def _reach_precheck_reject(self, action, C, komo_nc):
    # run the RELAXED (collision-free) solve; True => clear reach failure, reject the bind
    # without running the full solve. Residual-gated (see the reach_precheck comment in
    # __init__): a failed relaxed solve rejects only when it missed by a lot (eq >
    # precheck_eq_min); borderline failures fall through to the full solve, which decides.
    import time as _time
    _t0 = _time.perf_counter()
    self._pc = True                                       # A1: tag wins as precheck-sourced
    ok0, _ = self._solve_ik(komo_nc, restarts=self.restarts_precheck)
    self._pc = False
    self.prof["precheck_calls"] += 1
    self.prof["precheck_time"] += _time.perf_counter() - _t0
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

  def _ik_pick(self, action, s0, s1, assembly, seed, backoff=0.0, warm=None):
    self._cur_action = action                             # for PEN_VIEW_ACTION debug
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
    # PRE-PICK backoff (--pre-pick, export/replay): RETRACT the tool-center along -approach.
    # The corrected EE +z points from the standoff INTO the part, so backing off is -z_EE, in
    # the PART frame (ee_target is EE-relative-to-part). Gives the pre-grasp funnel-mouth pose.
    if backoff > 0.0:
      ee_target = _backoff_ee(ee_target, backoff)

    robot_ee_name = self.gripper_frame[robot]
    part_name = part

    # the part sits on the table (FIXED target) -> the sound reach-sphere bound applies for
    # any arm, calibrated or not (the analytic reject below subsumes it when available).
    part_pose = np.asarray(C.getFrame(part_name).getPose(), dtype=float)
    if self._reach_reject(C, robot, _pose_compose(part_pose, ee_target)[:3]):
      return False, None

    # the gripper is SUPPOSED to touch the part it grasps (two-finger palm/fingers close
    # around it), so exclude the target part from collision -- otherwise the collision
    # objective forbids the grasp itself. --always-contact keeps it on: then the grasp must
    # clear the strict gate WITH the finger-part contact, and overlapping grasps are re-searched.
    if not self.keep_contacts:
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
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [self.col_weight], [-0.0]
        )
      komo.addControlObjective([], 0, 1e-1)
      komo.addControlObjective([], 1, 1e-1)
      self._add_pull(komo, C, robot)                      # witness-opt pull (no-op in search)
      self._add_pin(komo, C)                              # pre-skill pin (no-op in search)
      # 6-DOF: gripper tool-center must sit at the (approach-corrected) grasp on the part:
      #   pose(ee) relative to pose(part) == ee_target.
      _gw = [self.grasp_eq_weight if self.grasp_eq_weight is not None else 1e1]
      komo.addObjective(
          [1, 1], ry.FS.positionRel, [robot_ee_name, part_name], ry.OT.eq, _gw, ee_target[:3]
      )
      komo.addObjective(
          [1, 1], ry.FS.quaternionRel, [robot_ee_name, part_name], ry.OT.eq, _gw, ee_target[3:]
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
    self._ctx_seeds = ("pick", bool(seeds), self.tool_kind[robot])   # ana-refutation study
    if (self.empty_seed_refutes and self._pin_joints is None
        and self._analytic_reject_ok(robot) and not seeds):
      # --empty-seed-refutes: the calibrated closed-form solver enumerated EVERY joint branch
      # at the EXACT grasp pose and found none within limits. This is a sound unreachability
      # proof (q1..q5 are roll-independent; q6 spans >= 2pi, so no other vacuum roll helps
      # either) -- skip relaxed precheck and KOMO entirely.
      self.analytic_rejects += 1
      self.ana_stats[f"pick/{self.tool_kind[robot]}/empty_refuted"] = \
          self.ana_stats.get(f"pick/{self.tool_kind[robot]}/empty_refuted", 0) + 1
      return False, None
    if self._analytic_reject_ok(robot) and seeds:
      reject, ranked = self._probe_branch_combos(C, {robot: seeds})
      if reject:
        return False, None                                # every exact branch deeply colliding
      seeds = ranked or seeds                             # shallowest-collision branches first
      if self.analytic_seed_cap is not None:              # --analytic-seed-cap: warm-start
        seeds = seeds[:self.analytic_seed_cap]            # only the K best-ranked branches
    elif (self.reach_precheck and self._pin_joints is None
          and self._reach_precheck_reject(action, C, build(False))):
      # includes the EMPTY-analytic-set case: an empty EXACT set does NOT prove the SOFT
      # solve fails -- the tolerant pose eq settles within tolerance of exactly-unreachable
      # poses (measured: the chair posts' thin-cylinder side pinch). Only the residual-gated
      # relaxed precheck may reject there (clear misses only); borderline falls to the solve.
      if not seeds:
        self.analytic_rejects += 1
      return False, None                                  # clear reach failure (relaxed solve)

    komo = build(True)
    if warm is not None:                                   # couple pre-pick to the pick witness
      wseed = self._restrict_seed(warm, C, full_names) if full_names is not None \
          else np.asarray(warm, dtype=float)
      seeds = [wseed] + list(seeds or [])
    feasible, witness = self._solve_ik(komo, seeds=seeds, C=C)
    _w_raw = witness                                       # C-dimensioned (restricted under
    if self.single_arm_ik:                                 # single_arm_ik) -- for the gate
      witness = self._expand_witness(C, witness, full_names)  # restricted -> full joint vector
      # HARVESTED BRANCHES NEED THE SAME EXPANSION. _solve_with_seeds collects raw
      # komo.getPath() rows, which under single_arm_ik are the RESTRICTED 6-vector -- handing
      # those to a consumer that indexes by full joint name raised
      # KeyError: 'a2_ur_shoulder_pan_joint' in optimize_witnesses' clearance().
      if self.harvest_branches:
        self.harvest_branches = [self._expand_witness(C, h, full_names)
                                 for h in self.harvest_branches]
    if feasible and _w_raw is not None and self.grasp_pose_reject is not None \
        and backoff <= 0.0:                               # gate the REAL grasp, not the funnel
      if self.grasp_pose_residual(C, _w_raw, robot, part, ee_target) \
          > self.grasp_pose_reject:
        self.grasp_rejects += 1                           # off the DB pose: candidate dies,
        feasible, witness = False, None                   # bind tries the next grasp id
    if not feasible:
      self._log_failure(action, C, komo)

    if self.debug_view:
        print("pick feasible:", feasible)
        komo.view(True, f"pick IK: {robot} -> {part} (grasp {grasp_id})")

    return feasible, witness

  def _ik_add_grasp(self, action, s0, s1, assembly, seed, backoff=0.0, warm=None):
    self._cur_action = action                             # for PEN_VIEW_ACTION debug
    # a SECOND robot reaches a grasp on an ALREADY-HELD subassembly (the handover / brace
    # primitive). Like a pick, but the target is held (posed by its current holder) rather
    # than resting on the table: scene_for attaches the group to its kinematic holder, that
    # arm may co-optimise (control-cost-anchored) to present a face, and the new arm reaches
    # the grasp. Existing braces stay pinned via grasp maintenance.
    _, robot, gpart, grasp_id = action
    C = self.scene_for(s0, assembly)                      # group held at its current holder(s)
    grasp = grasp_pose(assembly, gpart, grasp_id)
    ee_target = np.concatenate([grasp[:3], _quat_mul(grasp[3:], self.q_correction[robot])])
    # PRE-ADD-GRASP backoff (--pre-add-grasp): RETRACT the tool-center along -approach (-z_EE),
    # exactly like pre-pick -- the pre-grasp funnel mouth for the brace/handover reach.
    if backoff > 0.0:
      ee_target = _backoff_ee(ee_target, backoff)
    if not self.keep_contacts:
      set_part_contact(C, gpart, 0)                      # the new gripper is meant to touch gpart

    def build(collisions):
      komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=collisions)
      if collisions:
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [self.col_weight], [-0.0])
      komo.addControlObjective([], 0, 1e-1)
      komo.addControlObjective([], 1, 1e-1)
      self._add_pull(komo, C, robot)                      # witness-opt pull (no-op in search)
      self._add_pin(komo, C)                              # pre-skill pin (no-op in search)
      self._add_grasp_objectives(komo, robot, gpart, ee_target)
      self._add_grasp_maintenance(komo, s0, assembly)     # existing braces stay at their grasps
      return komo

    if (self.reach_precheck and self._pin_joints is None
        and self._reach_precheck_reject(action, C, build(False))):
      return False, None                                  # clear reach failure (relaxed solve)

    komo = build(True)
    feasible, witness = self._solve_ik(          # `warm` (full config) couples pre-add-grasp to
        komo, seeds=([np.asarray(warm, dtype=float)] if warm is not None else None),  # the grasp
        C=C)
    if feasible and witness is not None and self.grasp_pose_reject is not None \
        and backoff <= 0.0:                               # same freeze as picks: the handover/
      if self.grasp_pose_residual(C, witness, robot, gpart, ee_target) \
          > self.grasp_pose_reject:                       # brace attach keeps the ACHIEVED pose
        self.grasp_rejects += 1
        feasible, witness = False, None
    if not feasible:
      self._log_failure(action, C, komo)

    if self.debug_view:
        print("add_grasp feasible:", feasible)
        komo.view(True, f"add_grasp IK: {robot} -> {gpart} (grasp {grasp_id})")

    return feasible, witness

  def _ik_place(self, action, s0, s1, assembly, seed, backoff=0.0, warm=None):
    self._cur_action = action                             # for PEN_VIEW_ACTION debug
    # ONE robot (the placer) sets its held subassembly onto the table, resting on the
    # placement FACE the action names. The target is the pose the DISCRETE model will
    # assume afterwards (scene_for/_pose_on_table's rest pose anchored at the lowest-index
    # part's initial table x/y) -- the IK must land the group exactly there.
    robot, root, face = action[1], action[2], action[3]   # 5th field (cell id) may be present
    C = self.scene_for(s0, assembly)                      # group attached to robot's gripper
    group = next(g for g in s0.subassemblies() if root in g)
    ref = min(group, key=_part_index)                     # scene_for's on-table reference
    # target the pose the RESULT state records for the group (its landing x/y): the initial
    # env pose for an ordinary place, but the stow cell when the decluttering search relocates
    # a blocker. Falls back to the env pose when the state carries no pose payload.
    anchor = s1.poses.get(ref)
    anchor = np.asarray(anchor) if anchor is not None else np.asarray(self.C.getFrame(ref).getPose())
    target = _group_rest_pose(assembly, group, ref, anchor, face)
    # PRE-PLACE backoff (--pre-place, export/replay): LIFT the group +z (world) off its rest
    # pose -> the object hovers above its landing spot (the pre-place funnel mouth), and the
    # place skill lowers the last DELTA.
    if backoff > 0.0:
      target = _backoff_place(target, backoff)

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
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [self.col_weight], [-0.0]
        )
      komo.addControlObjective([], 0, 1e-1)
      komo.addControlObjective([], 1, 1e-1)
      self._add_pull(komo, C, robot)                      # witness-opt pull (no-op in search)
      self._add_pin(komo, C)                              # pre-skill pin (no-op in search)
      # object-pose constraint (like assemble, not an EE pose): the group's reference part
      # must reach its canonical table pose, axis-aligned.
      # --place-eq-weight, the PLACE twin of --mate-eq-weight. A place lands the datum every
      # later mate is measured against, so its residual is not local: measured on car-mini
      # (seed 11019, --grasp-translations 2) the body placed 3.91mm off while every wheel
      # mated onto it at 0.01-0.23mm -- each assemble perfectly seated on a body in the wrong
      # place. --mate-eq-weight only reaches _ik_assemble, and tightening the global NLP
      # stopTolerance instead makes this place INFEASIBLE (measured: bind fails at 1e-4).
      _pw = self.place_eq_weight if self.place_eq_weight is not None else 1e1
      komo.addObjective([1, 1], ry.FS.position, [ref], ry.OT.eq, [_pw], target[:3])
      komo.addObjective([1, 1], ry.FS.quaternion, [ref], ry.OT.eq, [_pw], target[3:])
      if not single:
        self._add_grasp_maintenance(komo, s0, assembly)   # braces ride along / stay pinned
      return komo

    # analytic (ssik) IK: the landing pose is FIXED and the group RIGID, so EVERY attached
    # arm's tool pose is fully determined -- the placer's via the kinematic attach, each
    # BRACE's via its pinned grasp (see _attached_arm_targets). The braced place therefore
    # DECOMPOSES per-arm for reachability:
    #   reject   : any calibrated arm with an EMPTY analytic set cannot reach its pose ->
    #              the bind is infeasible, no KOMO at all (each arm must reach its pose);
    #   precheck : when EVERY attached arm is calibrated (and nonempty), the relaxed problem
    #              -- exactly "each arm reaches its fixed pose" -- is PROVEN feasible, so the
    #              precheck (up to 13 relaxed multi-arm solves) is skipped;
    #   seeds    : solo -> the placer's 6-dim solutions; braced -> full joint vectors composed
    #              from the per-arm solutions (closest-to-current first).
    # Any UNCALIBRATED attached arm (vacuum / no ssik) falls back to the relaxed precheck.
    arm_targets = self._attached_arm_targets(C, s0, assembly, group, ref, target, robot)
    for rr, ee in arm_targets.items():
      if self._reach_reject(C, rr, ee[:3]):                # sound for any arm, ~us
        return False, None
    if not self.braced_analytic and not single:
      arm_targets = {}                                     # ablation: braced -> old precheck path
    arm_seeds = {}
    all_calibrated = bool(arm_targets)
    _empty_exact = _uncalibrated = False                  # ana-refutation study trackers
    for rr, ee in arm_targets.items():
      if self._analytic_reject_ok(rr):
        sl = self._analytic_seeds(C, rr, ee, None)
        if not sl:
          all_calibrated = False                          # EMPTY exact set: borderline, not a
          _empty_exact = True                             # proof -- precheck decides (see pick)
          continue
        arm_seeds[rr] = sl
      else:
        all_calibrated = False
        _uncalibrated = True
    if (self.empty_seed_refutes and _empty_exact and not _uncalibrated
        and self._pin_joints is None):
      # every calibrated attached arm proved its fixed pose unreachable (sound, see pick);
      # no uncalibrated arm is present whose reachability would remain unknown.
      self.analytic_rejects += 1
      self.ana_stats[f"place/{self.tool_kind[robot]}/empty_refuted"] = \
          self.ana_stats.get(f"place/{self.tool_kind[robot]}/empty_refuted", 0) + 1
      return False, None
    self._ctx_seeds = ("place", all_calibrated and bool(arm_seeds),
                       self.tool_kind[robot])            # ana-refutation study
    if all_calibrated:
      # probe the FULL branch product: single -> the placer's branches (other arms frozen AND
      # contact-stripped in the restricted scene, so nothing to exclude); braced -> all combos
      # incl. ARM-ARM pairs between placer and brace (each combo is a consistent candidate:
      # every tool at its true pose, the group at its landing pose). Uninvolved arms are free
      # DOFs of the multi-arm KOMO (they can dodge) -> their pairs are excluded.
      exclude = [] if single else [rr for rr in self.robots if rr not in arm_seeds]
      reject, seeds = self._probe_branch_combos(C, arm_seeds, exclude=exclude)
      if reject:
        return False, None                                # every exact combo deeply colliding
    else:
      if arm_seeds.get(robot):                            # mover-only probe (others will move)
        reject, ranked = self._probe_branch_combos(
            C, {robot: arm_seeds[robot]},
            exclude=[] if single else [rr for rr in self.robots if rr != robot])
        if reject:
          return False, None
        arm_seeds[robot] = ranked or arm_seeds[robot]
      seeds = arm_seeds.get(robot, []) if single else []
      if (self.reach_precheck and self._pin_joints is None
          and self._reach_precheck_reject(action, C, build(False))):
        return False, None                                # clear reach failure (relaxed solve)

    komo = build(True)
    if warm is not None:                                   # couple pre-place to the place witness
      wseed = self._restrict_seed(warm, C, full_names) if full_names is not None \
          else np.asarray(warm, dtype=float)
      seeds = [wseed] + list(seeds or [])
    feasible, witness = self._solve_ik(komo, seeds=seeds, C=C)
    if single:
      witness = self._expand_witness(C, witness, full_names)  # restricted -> full joint vector
      # same expansion for the harvested branches (see the pick path) -- these are RESTRICTED
      # 6-vectors straight out of komo.getPath() and every consumer indexes by full joint name.
      if self.harvest_branches:
        self.harvest_branches = [self._expand_witness(C, h, full_names)
                                 for h in self.harvest_branches]
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
      komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [self.col_weight], [-0.0])
    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 1, 1e-1)
    self._assemble_objectives(komo, part_name, onto, rel_pos, rel_quat, bias_axes)
    return komo

  def _assemble_objectives(self, komo, part_name, onto, rel_pos, rel_quat, bias_axes,
                           phase=1):
    # the mate constraint + its soft pose biases. Split out of _assemble_komo so
    # fuse_cograsp_witnesses can ground it on the SECOND keyframe of a 2-phase KOMO; the
    # single-phase caller keeps the default.
    # cube-pose constraint (NOT an EE pose): pose(part) relative to pose(onto) == the mate.
    #   positionRel([A, B]) / quaternionRel([A, B]) = pose of A expressed in B's frame.
    # MATE EQ WEIGHT. rai's augmented-Lagrangian schedule trades the eq residual against the
    # other objectives; at the default 1e1 the mate settles MILLIMETRES out and rai still calls
    # it feasible. Asking for accuracy up front is cheaper than rejecting and re-solving blindly
    # (measured: --mate-accuracy 1e-3 vetoed 1315 mates and cost ~90x search time).
    _mw = self.mate_eq_weight if self.mate_eq_weight is not None else 1e1
    komo.addObjective([phase, phase], ry.FS.positionRel, [part_name, onto], ry.OT.eq, [_mw], rel_pos)
    komo.addObjective([phase, phase], ry.FS.quaternionRel, [part_name, onto], ry.OT.eq, [_mw], rel_quat)
    # slight position bias: pull the centre body toward the middle of the workspace. LOW
    # weight vs the mate eq -- only breaks the free-position DOF of a HELD assembly (inert on
    # the table). Keeps held assemblies centred instead of drifting to the reachable fringe.
    komo.addObjective(
        [phase, phase], ry.FS.position, [onto], ry.OT.sos, [_ASSEMBLE_POS_BIAS_W],
        list(_ASSEMBLE_POS_BIAS_TARGET),
    )
    # soft orientation bias on the centre body (`onto`): nudge its local +z / +x axes toward
    # the given world directions. LOW weight vs the mate eq (1e1), so it only breaks the
    # free-orientation DOF of a HELD assembly; inert when `onto` rests on the table (fixed).
    # Axis alignment is sign-free -- no quaternion double cover.
    if bias_axes is not None:
      z_dir, x_dir = bias_axes
      komo.addObjective([phase, phase], ry.FS.vectorZ, [onto], ry.OT.sos, [_ASSEMBLE_ORIENT_BIAS_W], list(z_dir))
      komo.addObjective([phase, phase], ry.FS.vectorX, [onto], ry.OT.sos, [_ASSEMBLE_ORIENT_BIAS_W], list(x_dir))

  def _ik_assemble(self, action, s0, s1, assembly, seed, bias_axes=None, backoff=0.0, warm=None):
    self._cur_action = action                             # for PEN_VIEW_ACTION debug
    arm_seeds = {}                                        # per-arm ssik insertion seeds
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
    # PRE-INSERTION backoff (export/replay only; `backoff`=0 in the search -> the SEARCH always
    # witnesses the REAL mated pose). When >0, solve the T_pre = T_mate shifted by -backoff*
    # mate_axis funnel-mouth keyframe (the seat direction is +mate_axis, so backing off is
    # -mate_axis). Parts stay axis-aligned (rel_quat=identity), so the base-frame mate_axis is
    # also the offset direction in `onto`'s frame -> a pure translation of rel_pos.
    if backoff > 0.0:
      rel_pos = _backoff_mate(rel_pos, assembly.mate_axis(part), backoff)

    # the mate seats the moving group flush against the target, so those contacts are intended
    # -- exclude the WHOLE moving group from collision (a single cube, or every cube of a
    # module: a stacked plate touches the target plate on all its cubes). Otherwise the
    # intended flush seating reads as penetration and IK fails. Arm-vs-rest collision stays on;
    # the cheap floating-EE check already covered the moved geometry vs the scene.
    moving = next((g for g in s0.subassemblies() if part in g), (part,))
    if not self.keep_contacts:
      for mp in moving:
        set_part_contact(C, mp, 0)
    # ...but tell the GATE what was excluded and why. Disabling the group's contact is needed
    # for the SOLVE (the mate is flush against the target, so the intended contact would read as
    # penetration and the IK would never converge) -- it is NOT a reason to stop checking the
    # part against the REST of the world. Without this the strict gate is blind to part-vs-arm
    # and part-vs-other-part at the mate, which is exactly where the motion planner then
    # disagrees with us at the same tolerance (measured on cross-12: the inserted obj_3 through
    # the bracing arm's palm by 5.2mm, obj_5 by 4.3mm, both silently accepted).
    if self.mate_pair_gate:
      target_grp = next((g for g in s0.subassemblies() if onto in g), (onto,))
      self._mate_exempt = (frozenset(moving), frozenset(target_grp))

    # FIXED-target fast rejects: when the TARGET group rests on the table (no robot holds any
    # of its parts) the mate pose is fixed in the world, and the moving group rides RIGIDLY on
    # its gripper(s) -> EVERY attached arm's tool pose (mover AND braces) is fully determined
    # (the same contract as place; a HELD target's pose is free -- the KOMO chooses it -- so
    # no fixed pose exists and these filters must not fire). Per arm: reach bound (any robot),
    # analytic reject (calibrated two-finger: empty set = unreachable = reject, no KOMO). When
    # ALL attached arms are calibrated+nonempty, the relaxed precheck -- exactly "each arm
    # reaches its fixed pose", no residual coupling -- is proven feasible and skipped.
    reachable_proven = False
    tgroup = next((g for g in s0.subassemblies() if onto in g), (onto,))
    if not any(pt in tgroup for pt, _g in self.holders(s0).values()):
      opose = np.asarray(C.getFrame(onto).getPose(), dtype=float)
      part_target = _pose_compose(opose, np.concatenate([rel_pos, rel_quat]))
      # like place: the mate pose is fixed and the moving group rigid, so EVERY attached
      # arm's tool pose (mover AND braces) is fully determined -> per-arm reject; and when
      # ALL attached arms are calibrated+nonempty the relaxed precheck is proven feasible.
      arm_targets = self._attached_arm_targets(C, s0, assembly, moving, part_name,
                                               part_target, robot)
      for rr, ee in arm_targets.items():
        if self._reach_reject(C, rr, ee[:3]):
          return False, None
      braced = len(arm_targets) > 1
      if not self.braced_analytic and braced:
        arm_targets = {robot: arm_targets[robot]}          # ablation: mover-only reject, no proof
      reachable_proven = self.braced_analytic or not braced
      _empty_exact = _uncalibrated = False                # ana-refutation study trackers
      for rr, ee in arm_targets.items():
        if not self._analytic_reject_ok(rr):
          reachable_proven = False                        # uncalibrated arm: precheck decides
          _uncalibrated = True
          continue
        sl = self._analytic_seeds(C, rr, ee, None)
        if not sl:
          reachable_proven = False                        # EMPTY exact set: borderline, not a
          _empty_exact = True                             # proof -- precheck decides (see pick)
          continue
        arm_seeds[rr] = sl
      if (self.empty_seed_refutes and _empty_exact and not _uncalibrated
          and self._pin_joints is None):
        # all calibrated attached arms proved their (rigid-group determined) pose
        # unreachable -- sound refutation of the whole multi-arm bind (see pick).
        self.analytic_rejects += 1
        self.ana_stats[f"assemble/{self.tool_kind[robot]}/empty_refuted"] = \
            self.ana_stats.get(f"assemble/{self.tool_kind[robot]}/empty_refuted", 0) + 1
        return False, None
      self._ctx_seeds = ("assemble", reachable_proven and bool(arm_seeds),
                         self.tool_kind[robot])           # ana-refutation study
      # collision probe on the exact branches (assemble KOMO is multi-arm: uninvolved arms can
      # dodge -> excluded). All attached arms calibrated -> full combo product incl. ARM-ARM;
      # otherwise mover-only (the uncalibrated arms will co-move, their pairs excluded).
      probe = arm_seeds if len(arm_seeds) == len(arm_targets) else (
          {robot: arm_seeds[robot]} if robot in arm_seeds else {})
      if probe:
        reject, _ = self._probe_branch_combos(
            C, probe, exclude=[rr for rr in self.robots if rr not in probe])
        if reject:
          return False, None                              # every exact combo deeply colliding

    if self.reach_precheck and not reachable_proven and self._pin_joints is None:
      # relaxed (collision-free) reach: can the mate pose be reached AT ALL? A clear relaxed
      # failure rejects instantly -- skipping the whole bias sweep + restart fallback (the
      # most expensive failure path there is). Residual-gated like the other skills.
      komo_nc = self._assemble_komo(C, part_name, onto, rel_pos, rel_quat, None,
                                    collisions=False)
      self._add_grasp_maintenance(komo_nc, s0, assembly)
      if self._reach_precheck_reject(action, C, komo_nc):
        return False, None

    _open_attempt = self._open_probe_attempt
    if bias_axes is not None:
      sweep = [(0, bias_axes)]                             # forced single orientation (export)
    elif _open_attempt is not None:
      # Fair batched probe: a bounded SLICE of structured basins per visit, all in this one
      # scene/precheck setup.  Once the six canonical basins have had a turn, later visits use
      # a bounded random-only fallback below.  This avoids the first prototype's worst cost:
      # rebuilding the scene and rerunning reach-precheck once PER bias.
      _oa = int(_open_attempt)
      _ob = max(1, int(self._open_probe_budget))
      sweep = list(enumerate(_ASSEMBLE_BIAS_ORIENTATIONS))[_oa:_oa + _ob]
    elif self.fail_order:
      # --fail-order: sweep the biases never-tried-first, then ascending fail rate, then
      # canonical index (stable + deterministic). Pure reordering -- the sweep stays exhaustive,
      # so completeness is untouched; only which equal-f variant is tried first changes.
      def _bias_rank(ib):
        i, _b = ib
        n, f, eq = self.action_stats.get(("abias", part, onto, i), (0, 0, 0.0))
        return (n == 0, f / n if n else 0.0, -eq, i)
      sweep = sorted(enumerate(_ASSEMBLE_BIAS_ORIENTATIONS), key=_bias_rank)
    else:
      sweep = list(enumerate(_ASSEMBLE_BIAS_ORIENTATIONS))  # canonical order (default)
    komo = None
    feasible, witness = False, None
    for bi, bias in sweep:                                # deterministic orientation sweep
      komo = self._assemble_komo(C, part_name, onto, rel_pos, rel_quat, bias)
      self._add_pull(komo, C, robot)                    # witness-opt pull (no-op in search)
      self._add_pin(komo, C)                            # pre-skill pin (no-op in search)
      self._add_grasp_maintenance(komo, s0, assembly)     # braces pinned at their grasps
      if __import__("os").environ.get("MILESTONE_DEBUG"):
        _sk = arm_seeds.get(robot)
        print(f"[assm-seed] {robot} analytic_ok={self._analytic_reject_ok(robot)} "
              f"arm_seeds[{robot}]={0 if _sk is None else len(_sk)}", flush=True)
      feasible, witness = self._solve_ik(              # `warm` (e.g. the mated witness for a
          komo, restarts=0,                            # pre-insertion goto) warm-starts the solve
          seeds=((([warm] if warm is not None else [])
                  + list(arm_seeds.get(robot) or [])) or None),  # warm basin first, then the
          C=C)                                       # analytic insertion seeds (calibrated arms)
      if feasible and witness is not None and self.mate_residual_reject is not None:
        _mr = self.mate_residual(C, witness, part_name, onto, rel_pos)
        if _mr > self.mate_residual_reject:
          self.mate_rejects += 1                       # under-seated: try the next bias/restart
          feasible = False
      if bias_axes is None:                            # evidence: canonical bias slot id `bi`
        _st = self.action_stats.setdefault(("abias", part, onto, bi), [0, 0, 0.0])
        _st[0] += 1
        if not feasible:
          _st[1] += 1
          _st[2] += float((getattr(self, "_last_res", {}) or {}).get("eq", 0.0) or 0.0)
      if feasible:
        break
    if (feasible and witness is not None and backoff <= 0.0
        and self.pre_insertion_delta > 0.0
        and not self._funnel_witness_ok(action, s0, s1, assembly, witness)):
      feasible, witness = False, None                     # -> the unbiased restart fallback
    # Legacy bind: exhaustive structured sweep + six-restart fallback, bit-for-bit as before.
    # Open batch: its first six effort slots are the structured biases; any slots LEFT AFTER
    # that fund the unbiased fallback in the SAME scene/precheck setup.  Thus batch=7 means
    # six canonical basins + one unbiased initialization, while a revisit starting at slot 7
    # spends its whole batch on fresh random draws.
    _open_fallback_budget = 0
    if _open_attempt is not None:
      _oa = int(_open_attempt)
      _ob = max(1, int(self._open_probe_budget))
      _nb = len(_ASSEMBLE_BIAS_ORIENTATIONS)
      _open_fallback_budget = (_ob if _oa >= _nb else max(0, _oa + _ob - _nb))
    _open_unbiased = _open_fallback_budget > 0
    if not feasible and (_open_attempt is None or _open_unbiased):
      komo = self._assemble_komo(C, part_name, onto, rel_pos, rel_quat, None)
      self._add_pull(komo, C, robot)                    # witness-opt pull (no-op in search)
      self._add_pin(komo, C)                            # pre-skill pin (no-op in search)
      self._add_grasp_maintenance(komo, s0, assembly)
      _fb_restarts = (6 if _open_attempt is None else
                      (max(0, _open_fallback_budget - 1)
                       if int(_open_attempt) < len(_ASSEMBLE_BIAS_ORIENTATIONS)
                       else _open_fallback_budget))
      _old_random = getattr(self, "_open_probe_random_only", False)
      _old_budget = self._open_probe_budget
      if _open_attempt is not None:
        self._open_probe_random_only = int(_open_attempt) > 0
        self._open_probe_budget = _open_fallback_budget
      try:
        feasible, witness = self._solve_ik(
            komo, restarts=_fb_restarts,
            seeds=([warm] if warm is not None else None), C=C)
      finally:
        self._open_probe_random_only = _old_random
        self._open_probe_budget = _old_budget
      if feasible and witness is not None and self.mate_residual_reject is not None:
        if self.mate_residual(C, witness, part_name, onto, rel_pos) > self.mate_residual_reject:
          self.mate_rejects += 1
          feasible, witness = False, None
    if (feasible and witness is not None and backoff <= 0.0
        and self.pre_insertion_delta > 0.0
        and not self._funnel_witness_ok(action, s0, s1, assembly, witness)):
      feasible, witness = False, None
    if not feasible:
      self._log_failure(action, C, komo)

    self._mate_exempt = None                              # scoped to this action's solves

    if self.debug_view:
        print("assemble feasible:", feasible)
        komo.view(True, f"assemble IK: {robot} mates {part} -> {onto}")

    return feasible, witness

  def _funnel_witness_ok(self, action, s0, s1, assembly, witness):
    """FUNNEL COUPLING (--pre-insertion): an ACCEPTED assemble witness must also stand a
    pre-insertion funnel mouth -- the exporter's own pre_skill_witness (its delta ladder +
    table gate), PINNED so only the mover and the moved group's co-holders may leave the
    witness config (the same movers rule as fuse_pre_skill_witnesses). This makes "the export
    finds a funnel keyframe" a property OF the witness, not of export-time luck: the witness
    solver now prefers target float heights that leave funnel room, and witness-optimization
    candidates (also produced by _ik_assemble) inherit the same guarantee. Delta remains a
    REQUESTED MAXIMUM (the ladder accepts the largest feasible rung), so the search never
    rejects a witness the exporter could have used. Only for the REAL mate solve (backoff==0);
    the funnel solves themselves must not recurse."""
    part = action[2]
    moving = next((g for g in s0.subassemblies() if part in g), (part,))
    movers = {action[1]} | {r for r, (pt, _g) in self.holders(s0).items() if pt in moving}
    names = self.C.getJointNames()
    pin = {n: float(witness[k]) for r in self.robots if r not in movers
           for k, n in enumerate(names) if n.startswith(r + "_")}
    return self.pre_skill_witness(action, s0, s1, assembly,
                                  skill_witness=witness, pin=pin) is not None

  def pre_skill_witness(self, action, s0, s1, assembly, skill_witness=None, pin=None):
    """EXPORT/REPLAY helper for --pre-pick/--pre-place/--pre-insertion: the PRE-skill GOTO
    keyframe -- the skill witness backed off along its approach axis (pick: -z_EE retract;
    place: +z lift; assemble: -mate_axis). Returns the full-config witness, or None (skill not
    split / IK failed -- caller falls back to the skill witness). Solved on demand; NOT used by
    the search (whose witness is always the real skill pose).

    COUPLED to the skill witness: the pre-skill solve is WARM-STARTED from `skill_witness` and
    given a full-config qItself pull toward it.  A hard post-solve actor-joint corridor then
    rejects a different IK branch (the pull alone is only a cost and cannot guarantee this).
    Without both, the two independent solves can land in different IK basins (a whole wrist
    revolution measured for assemble) -> a long, possibly-colliding 'skill' motion between the
    two keyframes. If no non-zero backoff rung stays in the corridor, return None and let the
    exporter use the skill witness itself: a zero-length funnel is safer than a branch jump.

    `pin` ({joint_name: value}, --fuse-pre-skill only) additionally FREEZES those joints by a
    hard eq. The pull alone cannot enforce "only the actor moves": the skill constraints are
    RELATIVE (positionRel(part, onto) for assemble, gripper-to-object for add_grasp), so a sos
    cost at weight 1e0 lets the optimiser pay for the backoff by moving the cheaper body -- the
    target assembly -- instead of the actor. None => byte-identical to the un-pinned solve."""
    kind = action[0]
    delta = {"pick": self.pre_pick_delta, "place": self.pre_place_delta,
             "add_grasp": self.pre_add_grasp_delta,
             "assemble": self.pre_insertion_delta}.get(kind, 0.0)
    if delta <= 0.0:
      return None
    warm = None
    if skill_witness is not None:
      warm = np.asarray(skill_witness, dtype=float)
      names = self.C.getJointNames()
      self._opt_pull = {"mode": "joint", "robot": action[1],
                        "qmap": {n: float(warm[i]) for i, n in enumerate(names)},
                        "weight": self.pre_insertion_pull, "rot_weight": 1.0}
    # A REQUESTED PIN THAT LANDS ON NOTHING IS NOT A REPAIR. _ik_pick/_ik_place restrict the
    # KOMO scene to the acting arm (6 joints), so a 12-joint pin drops to zero names and the
    # solve is the ordinary un-pinned one -- which the caller would then accept or reject on
    # solver noise between two draws of the SAME problem, and report as a repair. _add_pin
    # records whether it actually pinned anything so the caller can tell those apart.
    self.pin_applied = None
    self._pin_joints = pin
    self._funnel_solve = True
    try:
      # DELTA is a REQUESTED MAXIMUM: the full backoff can point into a nearby obstacle (the
      # classic case: an insertion whose mate sits just above the tabletop -- backing off
      # along -mate_axis sinks the part into the table). Walk a deterministic fraction
      # ladder (1.0/0.8/0.6/0.4/0.2/0.1); each rung is gated by the FULL skill acceptance
      # INCLUDING the funnel table-clearance in _pen_ok. A short non-zero funnel mouth is
      # strictly better than a colliding one, and a failed ladder falls back to the skill
      # witness (the caller keeps today's keyframe).
      for frac in (1.0, 0.8, 0.6, 0.4, 0.2, 0.1):
        d = delta * frac
        if kind == "pick":
          ok, w = self._ik_pick(action, s0, s1, assembly, None, backoff=d, warm=warm)
        elif kind == "place":
          ok, w = self._ik_place(action, s0, s1, assembly, None, backoff=d, warm=warm)
        elif kind == "add_grasp":
          ok, w = self._ik_add_grasp(action, s0, s1, assembly, None, backoff=d, warm=warm)
        else:
          ok, w = self._ik_assemble(action, s0, s1, assembly, None, backoff=d, warm=warm)
        if ok and w is not None:
          if warm is not None:
            names = self.C.getJointNames()
            actor_idx = [i for i, n in enumerate(names) if n.startswith(action[1] + "_")]
            dq = float(np.linalg.norm(np.asarray(w, dtype=float)[actor_idx] - warm[actor_idx]))
            if dq > self.pre_skill_max_dq:
              continue                                  # feasible pose, wrong IK branch
          return w
      return None
    finally:
      self._funnel_solve = False
      self._opt_pull = None
      self._pin_joints = None

  def post_release_witness(self, action, s0, s1, assembly, release_witness=None):
    """EXPORT helper for the missing second half of a contact funnel: RELEASE -> RETRACT.

    A release itself has no IK witness; it fires at the configuration inherited from the
    preceding grasp/assemble task. After the re-parent, solve a 5cm (configurable) departure
    along the released grasp's -approach axis while every other arm is hard-pinned at that
    release configuration. This makes the narrow contact exit explicit instead of asking a
    random free-space sampler to discover it from a measure-near-zero boundary state.

    The same full RobotIK clearance predicate used by search gates this solve. DELTA is a
    requested maximum: if the full departure is unreachable in the contact pocket, retry a
    deterministic 80/60/40/20/10% ladder.  A short non-zero escape is still a useful funnel
    mouth and is strictly better than silently collapsing the release back to a zero-length
    task. Failure of the whole ladder returns None; mp_export retains deterministic structure.
    """
    self.post_release_used_delta = None
    if self.post_release_delta <= 0.0 or release_witness is None or action[0] != "release":
      return None
    robot = action[1]
    held = self.holders(s0).get(robot)
    if held is None:
      return None
    gpart, gid = held
    fake = ("pick", robot, gpart, gid)
    warm = np.asarray(release_witness, dtype=float)
    names = list(self.C.getJointNames())
    pin = {n: float(warm[k]) for k, n in enumerate(names)
           if not n.startswith(robot + "_")}
    single0, strip0 = self.single_arm_ik, self.strip_inactive_collision
    # Full scene is deliberate: the release departure must clear the continuing holders at
    # their actual keyframe configs, not a restricted scene which silently homes them.
    self.single_arm_ik = False
    self.strip_inactive_collision = False
    # the departure may not sink into the TABLE either (same funnel gate as --pre-*)
    self._funnel_solve = True
    try:
      group = next((g for g in s1.subassemblies() if gpart in g), {gpart})
      other_holders = [r for r, (p, _g) in self.holders(s1).items() if p in group]
      if other_holders:
        fake = ("add_grasp", robot, gpart, gid)
      ok, w = False, None
      # Keep this ladder deterministic and bounded.  In car-mini the inserted hand could
      # leave by 5mm but not by 10-50mm; another release cleared at 20mm but self-collided at
      # 30mm.  Retrying the SAME gate at shorter distances completes both funnels without
      # weakening clearance or inventing a different feasibility rule after search.
      for frac in (1.0, 0.8, 0.6, 0.4, 0.2, 0.1):
        delta = self.post_release_delta * frac
        self._opt_pull = {"mode": "joint", "robot": robot,
                          "qmap": {n: float(warm[k]) for k, n in enumerate(names)},
                          "weight": self.pre_insertion_pull, "rot_weight": 1.0}
        self._pin_joints = pin
        if other_holders:
          ok, w = self._ik_add_grasp(
              fake, s1, s1, assembly, None, backoff=delta, warm=warm)
        else:
          # Do not add a constant warm start to this pick-style departure KOMO. After a
          # release from a now-mated multi-part group, rai can retain an internal stable-joint
          # qIndex hole while exposing the expected named arm joints. The joint pull above
          # still selects the release branch, and the ordinary initialization/restarts avoid
          # relying on that inconsistent internal indexing. _solve_ik also shape-checks any
          # analytic seeds before calling into rai.
          ok, w = self._ik_pick(
              fake, s1, s1, assembly, None, backoff=delta, warm=None)
        self._opt_pull = None
        self._pin_joints = None
        if ok and w is not None:
          self.post_release_used_delta = delta
          break
    finally:
      self.single_arm_ik, self.strip_inactive_collision = single0, strip0
      self._funnel_solve = False
      self._opt_pull = None
      self._pin_joints = None
    return w if ok else None


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
  # the (robot, gpart, grasp_id) that action's robot is about to open, read from s0 (the
  # robot rides along so mixed-EE scenes stand the RIGHT gripper at the opening grasp).
  r, root = action[1], action[2]
  group = next((g for g in s0.subassemblies() if root in g), (root,))
  for p in group:
    for c in s0.supports.get(p, ()):
      if c[0] == "grasp" and c[1] == r:
        return (c[1], c[2], c[3])
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
      s0, action[2], action[3], assembly, note=f"grasp candidate of {action}",
      robot=action[1]):
    import os as _os
    if _os.environ.get("MILESTONE_DEBUG"):
      try:
        pr = checker.penetrating_pairs(s0, assembly)
        print(f"[cheap-debug] {action}: {pr[:4]}", flush=True)
      except Exception as _e:
        print(f"[cheap-debug] {action}: report failed {_e}", flush=True)
    return False
  if action[0] == "inhand_reorient":
    # v1 gate: the RESULT state (part held at its NEW grasp, everything else unchanged) must
    # be collision-free. Deliberately NOT the _GRASP_ACTIONS candidate check -- that would
    # test the new EE against the holder's OWN old EE on the same part, which is meaningless
    # for an in-hand transition (the fingers move continuously, they never coexist).
    if not checker._check(s1, assembly, note=f"result of {action}"):
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
    "inhand_reorient": 1.0,      # one skill invocation (v1; refine when the skill has a model)
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
    if isinstance(gripper_kind, dict):
      gripper_kind = None                                  # mixed EEs v1: topological facts
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



# =============================================================================================
# WITNESS LADDER GEOMETRY -- what "the same pose" and "a different branch" actually mean.
# =============================================================================================
def gripper_roll_period(kind):
  """How much tool roll (wrist_3) maps a grasp onto ITSELF.

  A VACUUM cup is roll-symmetric: any wrist_3 gives the identical grasp, so roll is a FREE
  parameter the ladder may choose. A TWO-FINGER gripper maps onto itself every pi (the jaws
  swap). Treating roll as meaningful is why the exported goals showed a 4.712 rad jump on
  joint 5 -- 3*pi/2 of tool roll that changes nothing physically and costs a wrist flip."""
  return 0.0 if kind == "vacuum" else np.pi


def canonical_roll(q, roll_idx, prev_q, period, lo=None, hi=None):
  """Re-express `q` using the roll representative CLOSEST to `prev_q`, staying within limits.
  period 0.0 => roll is free (vacuum): snap it straight onto prev. Returns a NEW array."""
  out = np.array(q, dtype=float)
  if roll_idx is None or prev_q is None:
    return out
  cur, tgt = float(out[roll_idx]), float(prev_q[roll_idx])
  if period <= 0.0:                                        # fully free roll
    best = tgt
  else:
    k = round((tgt - cur) / period)
    best = cur + k * period
  if lo is not None and hi is not None and not (lo <= best <= hi):
    # walk back toward the valid range in whole periods (a free roll clamps instead)
    if period <= 0.0:
      best = min(max(best, lo), hi)
    else:
      while best > hi:
        best -= period
      while best < lo:
        best += period
      if not (lo <= best <= hi):
        return out
  out[roll_idx] = best
  return out


def ik_branch(q, elbow_idx, wrist_idx):
  """A coarse IK-branch label: (elbow up/down, wrist flipped/not).

  These are the two discrete choices a 6-DOF wrist-partitioned arm makes for the same tool
  pose, and they are what a "branch crossing" IS -- as opposed to a long but continuous reach.
  Tool ROLL is deliberately NOT part of the label (see gripper_roll_period): it is a symmetry,
  not a branch."""
  return (int(np.sign(q[elbow_idx]) or 1), int(np.sign(q[wrist_idx]) or 1))


def involved_robots(action, s_before, s_after, rik):
  """THE definition of "whose configuration does this action determine", shared by every
  consumer. Previously each side had its own copy and they were free to drift:
  `visualization._active_robots` (used by the replay AND by mp_export) computed it from the PRE
  state, while `witness_floor._involved` used the POST state -- and a handover's giver holds the
  group only before, the receiver only after, so the two could name different arms for the step
  whose IK solves them TOGETHER.

  Holders are unioned over BOTH states for that reason; pass `s_after=None` when only the pre
  state is available (the replay's case) and the pre state alone is used.

  A RELEASE returns [] -- `bind` produces no witness for one, so there are no DOFs to pose and
  nothing to choose. That was the only measured disagreement between the two old versions
  (28 releases across four plans); on all 91 witnessed actions they already agreed.

  For an ASSEMBLE the holders of the MOVED part's group count as well as the target's. An arm
  co-grasping the part being carried physically travels with it, so its configuration is
  determined by this action; the old pre-state version looked only at the target group and
  would have left such an arm behind. Measured byte-identical on every plan in the repo -- the
  scenes brace the TARGET, not the moved part -- so this is latent correctness, not a change."""
  kind = action[0]
  if kind == "release":
    return []
  r = action[1]
  out = [r]
  if kind in ("pick", "inhand_reorient"):
    return out                                            # these move no held group
  parts = [action[2], action[3]] if kind == "assemble" else \
          ([action[2]] if len(action) > 2 else [])
  for st in (s_before, s_after):
    if st is None:
      continue
    for p in parts:
      grp = next((g for g in st.subassemblies() if p in g), None)
      if grp is None:
        continue
      for h in rik._holders_of_group(st, grp):
        if h not in out:
          out.append(h)
  return out


def reconfig_crossings(qa, qb, roll_idx=None, thresh=np.pi / 2.0):
  """A crossing count that MEASURES the reconfiguration instead of labelling it.

  `ik_branch` partitions the solution set by (sign q_elbow, sign q_wrist_2), which is only 4 of
  the arm's 8 analytic branches -- the SHOULDER choice is missing, and it is real: measured over
  the harvested candidate pools on car-mini, two candidates the label calls IDENTICAL differ by
  up to 2.414 rad on the pan joint with no 2*pi wrap to explain it. Joint 3 shows the same thing
  (5.575 rad, 0.708 after removing whole turns) because wrist_1 is dragged along by a shoulder
  flip; it is not an independent branch of its own, so adding it to the label would chase the
  symptom.

  Deriving the shoulder branch from q alone needs the wrist centre, i.e. FK -- and the label is
  only ever COMPARED PAIRWISE, never stored, so the branch name is not actually needed. This
  counts one crossing per joint that must travel more than `thresh` along its shortest path,
  which catches shoulder, elbow and wrist flips alike. ROLL is excluded (a symmetry, already
  canonicalised by `canonical_roll`); whole turns are removed first, so a 2*pi wrap is free."""
  d = np.abs((np.asarray(qa, dtype=float) - np.asarray(qb, dtype=float) + np.pi)
             % (2.0 * np.pi) - np.pi)
  if roll_idx is not None and 0 <= roll_idx < d.size:
    d[roll_idx] = 0.0
  return int(np.count_nonzero(d > thresh))


def wrap_to_limits(x, lo, hi, ref):
  """The representative x + 2*pi*k inside [lo, hi] NEAREST to `ref`, or None when no whole-turn
  shift fits. This is what makes branch-seed construction limit-aware: a pi-offset always has an
  in-limits representative when the joint's range is >= 2*pi (the UR10 model's +-3.28 rad just
  qualifies), but the ur10 shoulder-lift range is only [-3.28, 0.28] and there a target either
  fits directly or the transform must be DROPPED, not clamped -- a clamped seed is not near any
  branch and only wastes a solve. Deterministic tie-break: the smaller value wins."""
  two_pi = 2.0 * np.pi
  k0 = round((float(ref) - float(x)) / two_pi)
  best = None
  for k in (k0 - 1, k0, k0 + 1):
    v = float(x) + two_pi * k
    if lo - 1e-9 <= v <= hi + 1e-9:
      if best is None or abs(v - ref) < abs(best - ref) - 1e-12:
        best = v
  return best


def branch_seeds(q, lo, hi, max_seeds=12):
  """Diverse 6-joint SEEDS aiming at the OTHER IK branches of a UR-style arm for the SAME tool
  pose, built from the one solved branch `q` (UR chain order: pan, lift, elbow, wrist_1,
  wrist_2, wrist_3). This is the ACTIVE generator behind --witness-sets: the candidate harvest
  is starved because KOMO restarts land in one basin and the analytic enumerator (ssik) may be
  unavailable, so the branches must be AIMED AT explicitly.

  Returns (exact, seeds):
    exact -- whole-turn (+-2*pi) shifts of a SINGLE joint, limit-checked. A full turn of one
             revolute joint moves NO downstream body, so these are bit-identical in pose and
             collision and need no solve; they are distinct C-space points the motion planner
             can exploit (a wound-up wrist goal avoids a full unwind). Empty on arms whose
             range is < 4*pi around q (the ur10's +-3.28 rad admits them only in corners).
    seeds -- approximate branch-flip WARM STARTS for a collision-aware IK re-solve: wrist flip
             (q3+pi, -q4, q5+pi), elbow flip (q1+q2, -q2, q3+q2 -- exact when upper arm ==
             forearm; the UR10's 0.612/0.572 split makes it a close guess), base flip (q0+pi
             with the lift/elbow/wrist chain mirrored), and their compositions -- the full
             8-branch family. Each is only a basin GUESS: the caller's solve + gates decide,
             and a seed that converges back to a known basin is discarded there.
  Every joint target is wrapped by wrap_to_limits (nearest in-limit representative);
  a transform with any unrepresentable joint is dropped. Deterministic."""
  q = np.asarray(q, dtype=float).ravel()
  lo = np.asarray(lo, dtype=float).ravel()
  hi = np.asarray(hi, dtype=float).ravel()
  if q.size != 6 or lo.size != 6 or hi.size != 6:
    return [], []
  pi = np.pi
  exact = []
  for j in (5, 3, 4):                                  # wrist joints: the spread metric's axes
    for s in (1.0, -1.0):
      v = q[j] + s * 2.0 * pi
      if lo[j] - 1e-9 <= v <= hi[j] + 1e-9:
        e = q.copy()
        e[j] = v
        exact.append(e)

  def t_wrist(v):
    o = v.copy()
    o[3] = v[3] + pi
    o[4] = -v[4]
    o[5] = v[5] + pi
    return o

  def t_elbow(v):
    o = v.copy()
    o[1] = v[1] + v[2]
    o[2] = -v[2]
    o[3] = v[3] + v[2]
    return o

  def t_base(v):
    o = v.copy()
    o[0] = v[0] + pi
    o[1] = -pi - v[1]                                  # mirror of the lift plane; -pi-x and
    o[2] = -v[2]                                       # pi-x are the same value mod 2*pi, so
    o[3] = -pi - v[3]                                  # the wrap picks whichever fits
    o[4] = -v[4]
    return o

  seeds = []
  for chain in ((t_wrist,), (t_elbow,), (t_elbow, t_wrist), (t_base,),
                (t_base, t_wrist), (t_base, t_elbow), (t_base, t_elbow, t_wrist)):
    v = q.copy()
    for t in chain:
      v = t(v)
    w = np.empty(6)
    ok = True
    for j in range(6):
      r = wrap_to_limits(v[j], lo[j], hi[j], q[j])
      if r is None:
        ok = False
        break
      w[j] = r
    if not ok:
      continue
    if np.linalg.norm(w - q) < 1e-6:
      continue                                         # degenerate (e.g. q2=0 elbow flip)
    if any(np.linalg.norm(w - s) < 1e-6 for s in seeds):
      continue                                         # two chains collapsed to one target
    seeds.append(w)
  return exact, seeds[:max(0, int(max_seeds))]


def optimize_witnesses(C, assembly, plan, robots, robot_ik, cost="joint",
                       pull_weight=1.0, ee_rot_weight=0.3, sweeps=2, verbose=True,
                       gripper_kind=None, handovers=True, allow_place=True, bracing=False,
                       grasp_table=None, clear_weight=0.0, clear_margin=0.4,
                       robot_collisions=False, baseline_witnesses=None):
  """POST-PASS witness optimization (v1 of the witness-optimization plan): the DISCRETE
  plan is FIXED; each IK action is re-solved warm-pulled toward the ACTIVE arm's previous
  configuration, minimizing per-step active-arm movement in 'joint' space or 'ee' space
  (position + ee_rot_weight * orientation angle). Idle arms stay home-pinned exactly as in
  the search (NO repinning -- it destroys feasibility in many-robot scenes; continuity
  enters ONLY as a low-weight cost on the active arm). FEASIBILITY-SAFE: the search's own
  witness is kept whenever the pulled re-solve fails or is costlier. Sequential sweeps
  (greedy in time -- v2 joint KOMO later). Returns (witnesses, report): witnesses[i] is
  the witness for plan[i] (None for non-IK actions), report = before/after totals."""
  import time
  t0 = time.perf_counter()
  # robot_collisions=True: keep the OTHER robots' collision geometry ON while re-solving each
  # witness (the search strips it on solo steps for speed), so the witnesses AVOID the idle
  # arms rather than passing through them. Restored before returning.
  _strip0 = robot_ik.strip_inactive_collision
  if robot_collisions:
    robot_ik.strip_inactive_collision = False
  if grasp_table is None:
    _checker, grasp_table = _search_context(C, assembly, robots)
  # PRIVATE FK config: robot_ik.C is the base every scene_for/bind copies ("arms at home")
  # and the replay's start view -- mutating it corrupts both (measured: weird start pose,
  # shifted collision context for pulled re-solves). All FK probes run on this copy.
  fkC = ry.Config()
  fkC.addConfigurationCopy(robot_ik.C)
  names = list(fkC.getJointNames())
  home_map = dict(zip(names, robot_ik.home))
  arm_joints = {r: [n for n in names if n.startswith(r + "_")] for r in robots}

  def fk_ee(qmap, r):
    q = np.array([qmap.get(n, home_map[n]) for n in names])
    fkC.setJointState(q)
    return np.asarray(fkC.getFrame(robot_ik.gripper_frame[r]).getPose(), dtype=float)

  robotset = set(robots)
  # collidable frames per robot (the ur_coll* capsules + gripper), from the flat pair list.
  _collset = set(fkC.getCollidablePairs())
  coll_frames = {r: [f for f in _collset if f.startswith(r + "_")] for r in robots}
  _table = [f for f in _collset if f in ("table",)]      # fixed obstacle; parts excluded
  other_frames = {r: [f for rr in robots if rr != r for f in coll_frames[rr]] + _table
                  for r in robots}

  def clearance(witness, r):
    # PER-STEP CLEARANCE of the ACTIVE arm `r` (posed at `witness`) from the OTHER robots
    # (at home) AND the table. C.eval(FS.distance) per frame pair = a TRUE signed gap valid
    # when separated. SUM over pairs (not min): the gripper is pinned near the grasped
    # object so it is almost always the closest thing -- min would fixate on that
    # un-improvable distance; summing optimizes the ARM-BODY separation, which has real DOF
    # freedom. PARTS are EXCLUDED (no optimization DOFs -- the arm touches them on purpose);
    # the TABLE is INCLUDED (a fixed obstacle the arm should not drag along).
    wmap = dict(zip(names, witness))
    q = np.array([wmap[n] if n in arm_joints[r] else home_map[n] for n in names])
    fkC.setJointState(q)
    total_gap = 0.0
    for fa in coll_frames[r]:
      for fb in other_frames[r]:
        v, _ = fkC.eval(ry.FS.distance, [fa, fb])
        # CAP each pair at clear_margin: a gap beyond the margin is "clear enough" and
        # contributes a constant, so far pairs (incl. links high above the table, and the
        # base mounted ON the table) can't dominate the sum -- it focuses on pairs still
        # inside the danger zone (< margin), which is where separation actually matters.
        total_gap += min(-float(v[0]), clear_margin)
    return total_gap

  dp = cost.startswith("dp-")
  metric = cost[3:] if dp else cost

  def step_cost(qmap_prev, witness, r):
    wmap = dict(zip(names, witness))
    if metric == "joint":
      return float(np.sqrt(sum((wmap[n] - qmap_prev.get(n, home_map[n])) ** 2
                               for n in arm_joints[r])))
    pe = fk_ee(qmap_prev, r)
    we = fk_ee(wmap, r)
    dq = min(1.0, abs(float(np.dot(pe[3:], we[3:]))))
    return (float(np.linalg.norm(we[:3] - pe[:3]))
            + ee_rot_weight * 2.0 * float(np.arccos(dq)))

  states = [get_initial_state(C, assembly)]
  for a in plan:
    ns = next((ns for act, ns in successors(
        states[-1], assembly, robots, grasp_table=grasp_table,
        gripper_kind=gripper_kind, handovers=handovers, allow_place=allow_place,
        bracing=bracing) if act == a), None)
    if ns is None:                                          # successor flags don't match the plan
      raise ValueError(
          f"optimize_witnesses: cannot reproduce action {tuple(a)} from the successor set -- "
          f"the successor flags likely differ from the search that made the plan "
          f"(bracing={bracing}, handovers={handovers}, allow_place={allow_place}). "
          f"An add_grasp/handover in the plan needs bracing=True.")
    states.append(ns)

  if baseline_witnesses is not None:
    from plan_io import require_complete_witnesses
    require_complete_witnesses(plan, baseline_witnesses,
                               who="optimize_witnesses baseline")
    wit = [None if w is None else np.asarray(w, dtype=float).copy()
           for w in baseline_witnesses]
  else:
    wit = [None] * len(plan)
    for i, a in enumerate(plan):                          # legacy: reconstruct the baseline
      if a[0] in _IK_ACTIONS:
        ok, w = robot_ik.bind(a, states[i], states[i + 1], assembly)
        wit[i] = w if ok else None

  def total(ws):
    qmap = dict(home_map)
    tot = 0.0
    for i, a in enumerate(plan):
      if ws[i] is None:
        continue
      r = a[1]
      tot += step_cost(qmap, ws[i], r)
      wmap = dict(zip(names, ws[i]))
      for n in arm_joints[r]:
        qmap[n] = wmap[n]                                 # only the ACTIVE arm advances
    return tot

  before = total(wit)

  def clearance_stats(ws):
    vals = [clearance(ws[i], plan[i][1]) for i, a in enumerate(plan)
            if ws[i] is not None]
    return (min(vals) if vals else float("nan"),
            float(np.mean(vals)) if vals else float("nan"))

  cl_before = clearance_stats(wit) if clear_weight > 0 else (None, None)

  if dp:
    # LADDER-GRAPH DP (Viterbi): per step harvest multiple feasible candidate witnesses,
    # then choose the globally optimal combination. The objective is a chain -- each
    # step's cost couples only to the ACTIVE arm's previous configuration -- and only
    # a[1]'s joints advance the reference (same semantics as the greedy pass), so each
    # robot's steps form an INDEPENDENT chain: exact per-robot Viterbi, no combinatorics.
    # The greedy pass runs first and its witnesses join the ladder alongside the search
    # originals, so DP >= greedy >= original by construction.
    g_wit, _g_rep = optimize_witnesses(
        C, assembly, plan, robots, robot_ik, cost=metric, pull_weight=pull_weight,
        ee_rot_weight=ee_rot_weight, sweeps=sweeps, verbose=False,
        gripper_kind=gripper_kind, handovers=handovers, allow_place=allow_place,
        bracing=bracing, grasp_table=grasp_table,
        baseline_witnesses=wit)  # warm start (no clearance filter)
    cands = {}
    for i, a in enumerate(plan):
      if wit[i] is None:
        continue
      # The persisted certificate is always a ladder member. Candidate harvesting can improve
      # this baseline, but a failed/stochastic redraw can never erase it.
      cs = [wit[i]] + list(
          robot_ik.candidate_witnesses(a, states[i], states[i + 1], assembly) or [])
      if g_wit[i] is not None:
        cs.append(g_wit[i])
      seen, uniq = set(), []
      for w in cs:
        key = tuple(np.round(np.asarray(w, dtype=float), 3))
        if key not in seen:
          seen.add(key)
          uniq.append(w)
      cands[i] = uniq

    def dist_r(qmap_prev, witness, r):
      wmap = dict(zip(names, witness))
      if metric == "joint":
        return float(np.sqrt(sum((wmap[n] - qmap_prev.get(n, home_map[n])) ** 2
                                 for n in arm_joints[r])))
      pe = fk_ee(qmap_prev, r)
      we = fk_ee(wmap, r)
      dq = min(1.0, abs(float(np.dot(pe[3:], we[3:]))))
      return (float(np.linalg.norm(we[:3] - pe[:3]))
              + ee_rot_weight * 2.0 * float(np.arccos(dq)))

    n_cand = 0
    for r in robots:                                      # independent chain per robot
      ks = [i for i, a in enumerate(plan) if wit[i] is not None and a[1] == r]
      if not ks:
        continue
      n_cand += sum(len(cands[i]) for i in ks)
      # Viterbi forward: best[j] = (cost-to-here, backpointer) per candidate of step ks[j]
      layers = []
      prev_maps = [dict(home_map)]
      best_prev = [0.0]
      bp_all = []
      for j, i in enumerate(ks):
        layer_cost, layer_bp, layer_maps = [], [], []
        for c_idx, w in enumerate(cands[i]):
          node = -clear_weight * clearance(w, r) if clear_weight > 0 else 0.0
          costs = [best_prev[p] + dist_r(prev_maps[p], w, r) + node
                   for p in range(len(prev_maps))]
          p_best = int(np.argmin(costs))
          layer_cost.append(costs[p_best])
          layer_bp.append(p_best)
          wmap = dict(zip(names, w))
          m2 = dict(prev_maps[p_best])
          for n in arm_joints[r]:
            m2[n] = wmap[n]
          layer_maps.append(m2)
        bp_all.append(layer_bp)
        best_prev, prev_maps = layer_cost, layer_maps
        layers.append(list(range(len(cands[i]))))
      # backtrack
      sel = int(np.argmin(best_prev))
      for j in range(len(ks) - 1, -1, -1):
        wit[ks[j]] = cands[ks[j]][sel]
        sel = bp_all[j][sel]
    after = total(wit)
    if verbose:
      tail = ""
      if clear_weight > 0:
        cmin, cmean = clearance_stats(wit)
        tail = (f"; clearance min {cl_before[0]:.3f}->{cmin:.3f} "
                f"mean {cl_before[1]:.3f}->{cmean:.3f}")
      print(f"[witopt] dp-{metric} cost: {before:.3f} -> {after:.3f}  "
            f"(ladder of {n_cand} candidates over "
            f"{sum(1 for w in wit if w is not None)} steps){tail} "
            f"in {time.perf_counter() - t0:.1f}s")
    robot_ik.strip_inactive_collision = _strip0
    return wit, {"before": before, "after": after, "improved": -1, "fallback": 0}

  improved = 0
  fallback = 0
  try:
    for _sweep in range(sweeps):
      changed = False
      qmap = dict(home_map)
      for i, a in enumerate(plan):
        if wit[i] is None:
          continue
        r = a[1]
        def obj(w):
          return step_cost(qmap, w, r) - clear_weight * clearance(w, r)
        old_c = obj(wit[i])
        pe = fk_ee(qmap, r)
        robot_ik._opt_pull = {
            "mode": cost, "robot": r,
            "qmap": {n: qmap.get(n, home_map[n]) for n in arm_joints[r]},
            "ee_pos": pe[:3], "ee_quat": pe[3:],
            "weight": pull_weight, "rot_weight": ee_rot_weight,
        }
        ok, w2 = robot_ik.bind(a, states[i], states[i + 1], assembly)
        robot_ik._opt_pull = None
        if ok and w2 is not None and obj(w2) < old_c - 1e-9:
          wit[i] = w2
          improved += 1
          changed = True
        elif not ok:
          fallback += 1                                   # keep the original witness
        wmap = dict(zip(names, wit[i]))
        for n in arm_joints[r]:
          qmap[n] = wmap[n]
      if not changed:
        break
  finally:
    robot_ik._opt_pull = None
  after = total(wit)
  if verbose:
    tail = ""
    if clear_weight > 0:
      cmin, cmean = clearance_stats(wit)
      tail = (f"; clearance min {cl_before[0]:.3f}->{cmin:.3f} "
              f"mean {cl_before[1]:.3f}->{cmean:.3f}")
    print(f"[witopt] {cost} cost: {before:.3f} -> {after:.3f}  "
          f"({improved} steps improved, {fallback} pulled re-solves failed -> originals "
          f"kept){tail} in {time.perf_counter() - t0:.1f}s")
  robot_ik.strip_inactive_collision = _strip0
  return wit, {"before": before, "after": after, "improved": improved,
               "fallback": fallback}


def _pose_deviation(pa, pb):
  # (mm, deg) between two rai 7-poses. |dot| on the quaternions absorbs the double cover.
  pa, pb = np.asarray(pa, dtype=float), np.asarray(pb, dtype=float)
  dot = min(1.0, abs(float(np.dot(pa[3:], pb[3:]))))
  return float(np.linalg.norm(pb[:3] - pa[:3])) * 1000.0, float(np.degrees(2.0 * np.arccos(dot)))


def _quat_err_deg(v, target):
  dot = min(1.0, abs(float(np.dot(np.asarray(v, dtype=float), np.asarray(target, dtype=float)))))
  return float(np.degrees(2.0 * np.arccos(dot)))


def _replay_states(C, assembly, plan, robots, gripper_kind, handovers, allow_place,
                   bracing, grasp_table, who):
  # states[i] = the discrete state plan[i] is applied IN (len = len(plan)+1), re-derived by
  # replaying the plan through the SAME successor generator the search used. Shared by the
  # witness post-passes, which all need the per-step state to rebuild IK scenes. `who` only
  # names the caller in the error, which fires when the successor flags do not match the
  # search that produced the plan.
  states = [get_initial_state(C, assembly)]
  for a in plan:
    ns = next((ns for act, ns in successors(
        states[-1], assembly, robots, grasp_table=grasp_table,
        gripper_kind=gripper_kind, handovers=handovers, allow_place=allow_place,
        bracing=bracing) if act == a), None)
    if ns is None:                                          # successor flags don't match the plan
      raise ValueError(
          f"{who}: cannot reproduce action {tuple(a)} from the successor set "
          f"-- the successor flags likely differ from the search that made the plan "
          f"(bracing={bracing}, handovers={handovers}, allow_place={allow_place}). "
          f"An add_grasp/handover in the plan needs bracing=True.")
    states.append(ns)
  return states


def collect_plan_witnesses(C, assembly, plan, robots, robot_ik, *, gripper_kind=None,
                           handovers=True, allow_place=True, bracing=False,
                           grasp_table=None, resolve_missing=False, verbose=True):
  """Collect exact IK certificates already found for ``plan``, aligned by plan index.

  Positive RobotIK cache entries are the primary source; the open-witness pool is an explicit
  second source so persistence does not depend on cache lifetime. ``resolve_missing`` is only
  for a fresh solve after a plan-rewriting pass such as release hoisting: if the rewrite changes
  an IK predecessor scene, that new edge must be certified before serialization. Loaded plans
  pass False and therefore cannot accidentally redo IK here.
  """
  if grasp_table is None:
    _checker, grasp_table = _search_context(C, assembly, robots)
  states = _replay_states(C, assembly, plan, robots, gripper_kind, handovers, allow_place,
                          bracing, grasp_table, "collect_plan_witnesses")
  out = [None] * len(plan)
  rebuilt = []
  pool = getattr(robot_ik, "_open_witness_pool", None)
  for i, a0 in enumerate(plan):
    a = tuple(a0)
    if a[0] not in _IK_ACTIONS:
      continue
    key = (a, robot_ik._scene_sig(states[i]))
    hit = robot_ik._ik_cache.get(key)
    w = hit[1] if hit is not None and hit[0] and hit[1] is not None else None
    if w is None and pool is not None:
      w = pool.witness_for(states[i], a)
    if w is None and resolve_missing:
      # A stale stochastic failure from a previously considered ordering is not a certificate.
      # Hoisting has selected this edge, so give it one normal full-budget certification.
      if hit is not None and not hit[0]:
        robot_ik._ik_cache.pop(key, None)
      ok, w2 = robot_ik.bind(a, states[i], states[i + 1], assembly)
      w = w2 if ok else None
      rebuilt.append((i, a, ok))
    if w is not None:
      out[i] = np.asarray(w, dtype=float).copy()
  if verbose and rebuilt:
    ok_n = sum(ok for _, _, ok in rebuilt)
    print(f"[witness-persist] release hoisting/state rewriting required {len(rebuilt)} "
          f"new IK certification(s): {ok_n} succeeded, {len(rebuilt) - ok_n} failed.")
  return out


def validate_export_witnesses(C, assembly, plan, witnesses, robots, robot_ik,
                              gripper_kind=None, handovers=True, allow_place=True,
                              bracing=False, grasp_table=None, tol=None, verbose=True,
                              baseline_witnesses=None):
  """LAST GATE before mp_export: is the witness we are about to SHIP collision-free?

  Every earlier gate checks a witness at the moment it is SOLVED. Nothing checked the one that
  actually leaves the pipeline -- and it is rarely the same object: `optimize_witnesses*` emits
  a SPLICE (bind's base with a candidate's joints written over it) and `fuse_cograsp_witnesses`
  re-solves the pair again afterwards. Measured on cross-12: mate keyframes reached mp_export
  with a part 4.7mm inside an arm while the search reported zero dirty witnesses.

  For each assemble, measures the mate scene pair-wise (only the intended moving<->target
  contact is exempt). If the shipped witness is dirty but the SEARCH's own witness is clean,
  swaps it back. Whatever is still dirty is returned as a violation list so the caller can fail
  loudly at the boundary instead of handing the motion planner a keyframe it will reject.
  Returns (witnesses, [(step, action, penetration_m, swapped_bool), ...])."""
  if tol is None:
    tol = robot_ik.penetration_reject
  if tol is None:
    return witnesses, []
  if grasp_table is None:
    _checker, grasp_table = _search_context(C, assembly, robots)
  states = _replay_states(C, assembly, plan, robots, gripper_kind, handovers, allow_place,
                          bracing, grasp_table, "validate_export_witnesses")
  out = list(witnesses)
  bad = []
  for i, a in enumerate(plan):
    act = tuple(a)
    if act[0] != "assemble" or out[i] is None:
      continue
    part, onto = act[2], act[3]
    moving = frozenset(next((g for g in states[i].subassemblies() if part in g), (part,)))
    target = frozenset(next((g for g in states[i].subassemblies() if onto in g), (onto,)))
    Ci = robot_ik.scene_for(states[i], assembly)
    if not robot_ik.keep_contacts:
      for mp in moving:
        set_part_contact(Ci, mp, 0)                       # same exclusion the solve used

    def pen_of(q):
      Ci.setJointState(np.asarray(q, dtype=float).ravel())
      return robot_ik.mate_penetration(Ci, moving, target)

    # MATE RESIDUAL: what the witness ACTUALLY achieves for positionRel(part, onto). This is
    # the number mr_bench bakes in -- C.attach preserves the current WORLD pose at the mode
    # switch, so the exported assembly inherits exactly this error and then disagrees with our
    # scene (which re-seats at the ideal). Measured cross-12: a constant 3.08mm.
    try:
      Ci.setJointState(np.asarray(out[i], dtype=float).ravel())
      _v, _ = Ci.eval(ry.FS.positionRel, [part, onto])
      _ideal = np.asarray(assembly.relative_pose(part)[1], dtype=float)
      _res = float(np.linalg.norm(np.asarray(_v, dtype=float) - _ideal)) * 1000.0
      if _res > 0.2 and verbose:
        # COMPONENTWISE in the target's frame: a pure +z offset is a spawn-height / shape-origin
        # convention error; mixed axes are a wrong mate target. A constant offset in THIS frame
        # shows up as mixed-axis WORLD deltas whenever the assembly is rotated -- which is what
        # made the world-space view so misleading.
        _dv = (np.asarray(_v, dtype=float) - _ideal) * 1000.0
        # ALSO measure BIND's own witness: the shipped one is the FLOOR splice re-solved by
        # fuse_cograsp, so if bind is exact and the shipped one is not, a LATER stage
        # reintroduced the error and tightening the IK cannot help.
        _bres = float("nan")
        try:
          if baseline_witnesses is not None:
            _w2 = baseline_witnesses[i]
            _ok2 = _w2 is not None
          else:
            _ok2, _w2 = robot_ik.bind(act, states[i], states[i + 1], assembly)
          if _ok2 and _w2 is not None:
            Ci.setJointState(np.asarray(_w2, dtype=float).ravel())
            _v2, _ = Ci.eval(ry.FS.positionRel, [part, onto])
            _bres = float(np.linalg.norm(np.asarray(_v2, dtype=float) - _ideal)) * 1000.0
          Ci.setJointState(np.asarray(out[i], dtype=float).ravel())
        except Exception:
          pass
        print(f"  [export-gate] step {i} {act[:4]}: MATE RESIDUAL shipped {_res:.2f}mm "
              f"(dx={_dv[0]:+.2f} dy={_dv[1]:+.2f} dz={_dv[2]:+.2f}) vs BIND {_bres:.2f}mm")
    except Exception:
      pass
    p = pen_of(out[i])
    if p <= tol:
      continue
    swapped = False
    if baseline_witnesses is not None:
      w0 = baseline_witnesses[i]
      ok = w0 is not None
    else:
      ok, w0 = robot_ik.bind(act, states[i], states[i + 1], assembly)
    if ok and w0 is not None and pen_of(w0) <= tol:
      out[i] = np.asarray(w0, dtype=float).ravel().copy()  # the search's own clean witness
      swapped, p = True, pen_of(out[i])
    else:
      bad.append((i, act, p, swapped))
      continue
    if verbose:
      print(f"  [export-gate] step {i} {act[:4]}: shipped witness was dirty -- "
            f"restored the search's own witness")
  return out, bad


def cograsp_assembly_pairs(plan, states, robot_ik):
  """Map each add_grasp to the first assemble that uses that brace without moving it.

  This is the structural definition shared by witness fusion and MP export. A handover uses
  the same reach IK but has different ownership semantics, so it is intentionally excluded.
  Returns ``(add_index, assemble_index, brace_robot, grasped_part, grasp_id)`` tuples.
  """
  out = []
  for i, a in enumerate(plan):
    if not a or a[0] != "add_grasp":
      continue
    rb, gpart, gid = a[1], a[2], a[3]
    for k in range(i + 1, len(plan)):
      b = plan[k]
      if b[0] != "assemble":
        if len(b) > 1 and b[1] == rb:
          break                                             # the brace arm moved first
        continue
      if robot_ik.holders(states[k]).get(rb) != (gpart, gid):
        break                                               # the brace is no longer in place
      target = next((g for g in states[k].subassemblies() if b[3] in g), ())
      if gpart in target:
        out.append((i, k, rb, gpart, gid))
      break
  return out


def fuse_cograsp_witnesses(C, assembly, plan, witnesses, robots, robot_ik,
                           gripper_kind=None, handovers=True, allow_place=True,
                           bracing=False, grasp_table=None, weight=1e0,
                           pos_tol=1e-3, ang_tol=1.0, verbose=True):
  """CO-GRASP WITNESS FUSION -- a post-pass over a FIXED discrete plan, run AFTER
  optimize_witnesses and BEFORE mp_export.

  THE DEFECT. mp_export emits add_grasp as a bare type="goto", so a brace is enforced only
  AT keyframes and is unconstrained between them. That would be harmless if the two
  keyframes agreed, but they are two INDEPENDENT single-keyframe IK solves: the
  object-relative gripper pose is identical at both by construction (same part, same grasp
  id), yet the arm lands in a DIFFERENT IK branch, several radians apart. The exported
  motion then swings the braced gripper across the workspace and back while it is nominally
  welded to the object.

  THE FIX. For each (add_grasp, assemble) pair the add_grasp enables, replace the two
  independent solves with ONE KOMO over TWO keyframes. With pre-insertion enabled, phase 1
  carries BOTH the add-grasp constraints and the insertion-funnel pose; phase 2 carries the
  seated assemble constraints. Every holder of the target group is hard-coupled across the
  phases. The inserter alone pays for the short insertion. The accepted phase-1 witness is
  also returned as the assemble pre-witness, so MP export gives add_grasp and pre-assemble
  the SAME all-arm goal. Without pre-insertion, the historical low-weight brace/object
  cross-phase costs are retained.

  NO MODE SWITCH is needed: the brace exists at both keyframes as an OBJECTIVE (scene_for
  attaches only the primary gripper-parented root), so both keyframes live in the SAME
  kinematic configuration. That is verified per pair at runtime and the pair is skipped if
  it ever stops being true.

  ALL-OR-NOTHING per pair: a pair is accepted only if the joint solve converged, every
  constraint residual and the penetration at BOTH keyframes are within tolerance (or no
  worse than the witnesses being replaced), AND the brace joint delta actually shrank.
  Otherwise BOTH original witnesses are kept -- a half-applied pair (the brace pinned at one
  keyframe only) is strictly worse than either extreme.

  Returns (witnesses, report). `witnesses` is a NEW list; the input is never mutated.
  ``report["pre_witnesses"]`` maps an accepted assemble index to the shared
  add-grasp/pre-insertion keyframe."""
  import time
  t0 = time.perf_counter()
  if grasp_table is None:
    _checker, grasp_table = _search_context(C, assembly, robots)
  states = _replay_states(C, assembly, plan, robots, gripper_kind, handovers, allow_place,
                          bracing, grasp_table, "fuse_cograsp_witnesses")

  names = list(robot_ik.C.getJointNames())
  # penetration budget: the --strict-collisions gate when one is configured, else a 0.1mm
  # floor. Either way a fused keyframe may never be dirtier than the witness it replaces.
  pen_gate = robot_ik.penetration_reject if robot_ik.penetration_reject is not None else 1e-4

  out = list(witnesses)
  rows = []
  chain_pre = {}
  claimed = {}                                              # assemble index -> the add_grasp that took it
  for i, j, rb, gpart, gid in cograsp_assembly_pairs(plan, states, robot_ik):
    tag = f"[fuse] {i}->{j} brace {rb} on {gpart}"
    if witnesses[i] is None or witnesses[j] is None:
      rows.append((i, j, rb, False, "a witness is missing"))
      if verbose:
        print(f"  {tag}: SKIPPED (no witness to fuse)")
      continue
    if j in claimed:
      # TWO braces before ONE assemble: pairs() maps each add_grasp to the first qualifying
      # assemble, so both would write out[j] and the SECOND write would silently replace the
      # assemble witness the first pair was fused against -- leaving that pair half-applied
      # (its add_grasp keyframe optimised against a config that no longer exists) while
      # printing ACCEPTED. Half-applied is strictly worse than untouched, so refuse. Fusing
      # both jointly needs a 3-keyframe solve and is deliberately out of scope.
      rows.append((i, j, rb, False, f"assemble {j} already fused with add_grasp {claimed[j]}"))
      if verbose:
        print(f"  {tag}: REJECTED (assemble {j} already fused with add_grasp "
              f"{claimed[j]}; a 3-keyframe solve would be needed) -- BOTH originals kept")
      continue
    Ca = robot_ik.scene_for(states[i], assembly)
    Cb = robot_ik.scene_for(states[j], assembly)
    # SAME-MODE GUARD. The whole design rests on both keyframes living in one configuration;
    # verify it rather than assume it, because a mismatch would silently solve phase 2 in the
    # wrong scene.
    parts = [p for p in states[j].bodies() if p in Ca.getFrameNames()]
    same = (Ca.getFrameNames() == Cb.getFrameNames() and Ca.getJointNames() == Cb.getJointNames()
            and all(_pose_deviation(Ca.getFrame(p).getPose(), Cb.getFrame(p).getPose())[0] < 1e-6
                    for p in parts))
    if not same:
      rows.append((i, j, rb, False, "keyframes are not in the same kinematic mode"))
      if verbose:
        print(f"  {tag}: SKIPPED (the two keyframes are NOT one kinematic mode)")
      continue

    part, onto = plan[j][2], plan[j][3]
    ee_t = robot_ik.ee_target(rb, gpart, gid, assembly)
    rel_pos = np.asarray(assembly.relative_pose(part)[1], dtype=float)
    rel_quat = np.array([1.0, 0.0, 0.0, 0.0])
    axis = assembly.mate_axis(part)
    pre_delta = float(getattr(robot_ik, "pre_insertion_delta", 0.0) or 0.0)
    chain_active = pre_delta > 0.0 and axis is not None
    pre_rel_pos = _backoff_mate(rel_pos, axis, pre_delta) if chain_active else rel_pos
    moving = sorted(next((g for g in states[j].subassemblies() if part in g), (part,)))
    grp = next((g for g in states[i + 1].subassemblies() if gpart in g), (gpart,))
    root = _group_root(states[i + 1], grp)
    held = states[i + 1].parent.get(root, (None,))[0] == "gripper"
    bidx = [k for k, n in enumerate(names) if n.startswith(rb + "_")]
    target_grp = next((g for g in states[j].subassemblies() if onto in g), (onto,))
    target_holders = robot_ik._holders_of_group(states[j], target_grp)
    locked_names = [n for n in names if any(n.startswith(r + "_") for r in target_holders)]
    locked_idx = [k for k, n in enumerate(names) if n in locked_names]

    def build(bias):
      komo = ry.KOMO(Ca, phases=2, slicesPerPhase=1, kOrder=1, enableCollisions=True)
      # PER-SLICE collision exclusions. The two skills exclude DIFFERENT parts (add_grasp
      # frees the part the new gripper is meant to touch, assemble frees the whole moving
      # group whose seating is intended) and one KOMO has ONE contact configuration, so the
      # union is unsound -- measured 42mm of penetration at a phase-1 keyframe in its own
      # add_grasp scene. Slice frames are independent objects; set them before the
      # collision objective is added.
      for _sf in part_shape_frames(Ca, gpart):             # every child of a compound part
        komo.getFrame(_sf, 1.0).setContact(0)
      for mp in moving:
        for _sf in part_shape_frames(Ca, mp):
          komo.getFrame(_sf, 2.0).setContact(0)
      komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq,
                        [robot_ik.col_weight], [-0.0])
      komo.addControlObjective([], 0, 1e-1)
      komo.addControlObjective([], 1, 1e-1)
      robot_ik._add_grasp_objectives(komo, rb, gpart, ee_t, phase=1)
      robot_ik._add_grasp_maintenance(komo, states[i], assembly, phase=1)
      if chain_active:
        # The insertion arm arrives at the funnel mouth WHILE the brace is added. Target
        # holders do not transport the assembly afterwards; only the inserter traverses the
        # final pre_delta. This is the atomic add_grasp -> assemble choreography.
        robot_ik._assemble_objectives(
            komo, part, onto, pre_rel_pos, rel_quat, bias, phase=1)
      robot_ik._assemble_objectives(komo, part, onto, rel_pos, rel_quat, bias, phase=2)
      robot_ik._add_grasp_maintenance(komo, states[j], assembly, phase=2)
      # A target held by two arms is a closed chain. With a funnel, freeze every target
      # holder across the two phases as a hard equality. The old soft brace-only pull was not
      # enough: rejected car-mini pairs later drifted 138-436mm inside the MP path.
      if chain_active and locked_names:
        komo.addObjective([2, 2], ry.FS.qItself, locked_names,
                          ry.OT.eq, [1e1], [], order=1)
      else:
        komo.addObjective([2, 2], ry.FS.qItself,
                          [n for n in names if n.startswith(rb + "_")],
                          ry.OT.sos, [weight], [], order=1)
      komo.addObjective([2, 2], ry.FS.position, [gpart], ry.OT.sos, [weight], [], order=1)
      komo.addObjective([2, 2], ry.FS.quaternion, [gpart], ry.OT.sos, [weight], [], order=1)
      return komo

    # STRUCTURED RESTARTS, as _ik_assemble: sweep the canonical orientation biases, each
    # seeded from the ORIGINAL witnesses (the assemble one first -- it is the harder
    # constraint and it won every measured pair), then unbiased random restarts. KOMO adds
    # init noise and rai's RNG is not seeded by PYTHONHASHSEED, so a single-shot solve flips
    # feasible/infeasible run to run; the sweep is what makes the pass reliable.
    candidates = []
    for bias in list(_ASSEMBLE_BIAS_ORIENTATIONS) + [None]:
      komo = build(bias)
      seeds = (("paired", np.vstack([witnesses[i], witnesses[j]])),
               ("assemble", np.vstack([witnesses[j], witnesses[j]])),
               ("add_grasp", np.vstack([witnesses[i], witnesses[i]])))
      for sname, seed_path in seeds:
        try:
          komo.initWithPath(np.asarray(seed_path, dtype=float))
        except Exception:
          komo.initWithConstant(np.asarray(seed_path[-1], dtype=float))
        if robot_ik._nlp(komo).solve().dict()["feasible"]:
          candidates.append((np.asarray(komo.getPath()).copy(),
                             f"bias={bias} seed={sname}"))
    komo = build(None)
    for restart in range(6):
      komo.initRandom()
      if robot_ik._nlp(komo).solve().dict()["feasible"]:
        candidates.append((np.asarray(komo.getPath()).copy(),
                           f"bias=None seed=random#{restart}"))

    # RE-VALIDATION in the two ORIGINAL single-phase scenes (contact flags included), which
    # is what catches an exclusion-set mismatch the fused KOMO could not see.
    if not robot_ik.keep_contacts:
      set_part_contact(Ca, gpart, 0)
      for mp in moving:
        set_part_contact(Cb, mp, 0)
    # ...but the assemble scene's penetration is then measured with the WHOLE moving group
    # switched off, so a fused witness that drives the part through an ARM reads as 0mm and
    # sails past the `n_p2 > max(pen_gate, o_p2)` gate below. That is how a mate collision
    # survives even with --strict-mate-collisions: the flag gates `_ik_assemble`, and THIS pass
    # produces the witnesses that actually reach mp_export. Under the flag, measure the mate
    # scene pair-wise instead, exempting only the intended moving<->target contact.
    target_grp = frozenset(next((g for g in states[j].subassemblies() if onto in g), (onto,)))
    _mv = frozenset(moving)

    def residuals(q1, q2):
      # (max position error m, max orientation error deg, pen1 m, pen2 m) over EVERY eq
      # constraint the two skill KOMOs impose, plus the total penetration of each scene.
      pos, ang = 0.0, 0.0
      Ca.setJointState(np.asarray(q1, dtype=float))
      Cb.setJointState(np.asarray(q2, dtype=float))
      checks = [(Ca, ry.FS.positionRel, [robot_ik.gripper_frame[rb], gpart], ee_t[:3], False),
                (Ca, ry.FS.quaternionRel, [robot_ik.gripper_frame[rb], gpart], ee_t[3:], True),
                (Cb, ry.FS.positionRel, [part, onto], rel_pos, False),
                (Cb, ry.FS.quaternionRel, [part, onto], rel_quat, True)]
      if chain_active:
        checks.extend([
            (Ca, ry.FS.positionRel, [part, onto], pre_rel_pos, False),
            (Ca, ry.FS.quaternionRel, [part, onto], rel_quat, True),
        ])
      for Cx, st in ((Ca, states[i]), (Cb, states[j])):
        for r, pt, tgt in robot_ik.maintenance_grasps(st, assembly):
          checks.append((Cx, ry.FS.positionRel, [robot_ik.gripper_frame[r], pt], tgt[:3], False))
          checks.append((Cx, ry.FS.quaternionRel, [robot_ik.gripper_frame[r], pt], tgt[3:], True))
      for Cx, feat, frames, tgt, is_quat in checks:
        v, _ = Cx.eval(feat, frames)
        if is_quat:
          ang = max(ang, _quat_err_deg(v, tgt))
        else:
          pos = max(pos, float(np.linalg.norm(np.asarray(v) - np.asarray(tgt))))
      p2 = (robot_ik.mate_penetration(Cb, _mv, target_grp) if robot_ik.mate_pair_gate
            else float(Cb.getCollisionsTotalPenetration()))
      return (pos, ang, float(Ca.getCollisionsTotalPenetration()), p2)

    o_pos, o_ang, o_p1, o_p2 = residuals(witnesses[i], witnesses[j])
    didx = locked_idx if chain_active and locked_idx else bidx
    o_dq = float(np.linalg.norm(np.asarray(witnesses[j])[didx] - np.asarray(witnesses[i])[didx]))
    Ca.setJointState(np.asarray(witnesses[i], dtype=float))
    Cb.setJointState(np.asarray(witnesses[j], dtype=float))
    o_mm, o_deg = _pose_deviation(Ca.getFrame(gpart).getPose(), Cb.getFrame(gpart).getPose())

    # "feasible" is only KOMO's internal verdict. The strict collision and residual gates
    # below are intentionally stronger. Historically the pass stopped at the FIRST feasible
    # basin, then rejected it here without trying the remaining analytic seeds. On car-mini
    # that discarded three otherwise promising pairs for micrometre-scale collision misses.
    # Evaluate every feasible basin until one passes the gates; retain the least-bad rejected
    # one only for an informative diagnostic.
    best = how = None
    best_metrics = None
    why = "KOMO infeasible (all biases + restarts)"
    best_score = np.inf
    for cand, cand_how in candidates:
      n_pos, n_ang, n_p1, n_p2 = residuals(cand[0], cand[1])
      n_dq = float(np.linalg.norm(cand[1][didx] - cand[0][didx]))
      Ca.setJointState(cand[0])
      Cb.setJointState(cand[1])
      n_mm, n_deg = _pose_deviation(Ca.getFrame(gpart).getPose(), Cb.getFrame(gpart).getPose())
      reason = None
      if n_pos > max(pos_tol, o_pos):
        reason = f"position residual {n_pos*1000:.2f}mm > gate"
      elif n_ang > max(ang_tol, o_ang):
        reason = f"orientation residual {n_ang:.2f}deg > gate"
      elif n_p1 > max(pen_gate, o_p1) or n_p2 > max(pen_gate, o_p2):
        reason = (f"penetration {n_p1*1e6:.2f}/{n_p2*1e6:.2f}um > gate "
                  f"{max(pen_gate, o_p1)*1e6:.2f}/{max(pen_gate, o_p2)*1e6:.2f}um")
      elif n_dq >= o_dq:
        reason = f"brace delta not improved ({o_dq:.3f} -> {n_dq:.3f} rad)"
      score = max(n_pos / max(pos_tol, o_pos), n_ang / max(ang_tol, o_ang),
                  n_p1 / max(pen_gate, o_p1), n_p2 / max(pen_gate, o_p2))
      if reason is None or score < best_score:
        best, how, why = cand, cand_how, reason
        best_metrics = (n_dq, n_mm, n_deg)
        best_score = score
      if reason is None:
        break
    if best_metrics is not None:
      n_dq, n_mm, n_deg = best_metrics

    case = "HELD" if held else "TABLE-RESTING"
    if verbose:
      print(f"  {tag} ({case} group): brace dq {o_dq:.3f}" +
            (f" -> {n_dq:.3f} rad" if best is not None else " rad") +
            f", object pose delta {o_mm:.1f}mm/{o_deg:.1f}deg" +
            (f" -> {n_mm:.1f}mm/{n_deg:.1f}deg" if best is not None else "") +
            (f"   REJECTED: {why} -- BOTH originals kept" if why else
             f"   ACCEPTED [{how}]"))
    if why is None:
      out[i], out[j] = best[0].copy(), best[1].copy()
      if chain_active:
        chain_pre[j] = best[0].copy()                       # add_grasp == pre-assemble goal
      claimed[j] = i                                        # only on ACCEPT: a rejected pair
                                                            # leaves out[j] untouched, so a
                                                            # later add_grasp may still use it
    rows.append((i, j, rb, why is None, why or how))

  acc = sum(1 for r in rows if r[3])
  if verbose:
    print(f"[fuse] {acc}/{len(rows)} (add_grasp, assemble) pair(s) fused in "
          f"{time.perf_counter() - t0:.1f}s; the rest keep BOTH original witnesses.")
  return out, {"pairs": len(rows), "fused": acc, "rows": rows,
               "pre_witnesses": chain_pre}


def _pre_gate(o, n, pen_gate, max_dq, hold_dq, hold_mm, pos_tol, ang_tol):
  """ALL-OR-NOTHING acceptance for ONE pre-skill keyframe. `n` = the pinned re-solve's metrics
  (None = it did not converge), `o` = today's (None = today's solve failed outright, so there
  is nothing to be worse than and only the absolute tolerances apply).

  Every hard gate is "within the absolute tolerance OR no worse than the keyframe we are
  replacing" -- the fuse convention, so the pass can never regress a metric. On top of that the
  HEADLINE metric (approach-axis deviation: how far the moved body's pre->skill travel departs
  from the nominal funnel vector) must STRICTLY improve, else there is nothing to buy and
  today's keyframe is kept. Returns None to accept, or the reason to reject. Pure arithmetic."""
  if n is None:
    return "pinned solve infeasible"
  lim = lambda k, t: t if o is None else max(t, o[k])
  if n["pos"] > lim("pos", pos_tol):
    return (f"funnel position residual {n['pos']*1000:.2f}mm > gate "
            f"{lim('pos', pos_tol)*1000:.2f}mm")
  if n["ang"] > lim("ang", ang_tol):
    return f"funnel orientation residual {n['ang']:.2f}deg > gate {lim('ang', ang_tol):.2f}deg"
  if n["pen"] > lim("pen", pen_gate):
    return f"penetration {n['pen']*1000:.2f}mm > gate {lim('pen', pen_gate)*1000:.2f}mm"
  if n["dq"] > lim("dq", max_dq):
    return f"actor joint delta {n['dq']:.3f} rad > gate {lim('dq', max_dq):.3f} rad"
  if n["hold"] > lim("hold", hold_dq):
    return f"a non-moving arm shifted {n['hold']:.3f} rad > gate {lim('hold', hold_dq):.3f} rad"
  if n["stat_mm"] > lim("stat_mm", hold_mm):
    return f"a stationary body moved {n['stat_mm']:.1f}mm > gate {lim('stat_mm', hold_mm):.1f}mm"
  if o is not None and n["dev"] >= o["dev"]:
    return f"approach deviation not improved ({o['dev']:.1f} -> {n['dev']:.1f} mm)"
  return None


def fuse_pre_skill_witnesses(C, assembly, plan, witnesses, robots, robot_ik,
                             gripper_kind=None, handovers=True, allow_place=True,
                             bracing=False, grasp_table=None, max_dq=0.35, hold_dq=0.02,
                             hold_mm=2.0, pos_tol=1e-3, ang_tol=1.0, verbose=True):
  """PRE-SKILL WITNESS CONSISTENCY -- a post-pass over a FIXED discrete plan, run AFTER
  optimize_witnesses / fuse_cograsp_witnesses and BEFORE mp_export.

  THE DEFECT. --pre-pick/--pre-place/--pre-add-grasp/--pre-insertion split a skill into a GOTO
  to a funnel mouth plus the skill itself, and mp_export gets that first keyframe from
  RobotIK.pre_skill_witness -- the skill's OWN single-keyframe KOMO with the target backed off,
  warm-started from the skill witness plus a weight-1.0 qItself pull. Nothing in that solve says
  "only the ACTOR may move". The skill constraints are RELATIVE (positionRel(part, onto) for
  assemble, gripper-to-object for add_grasp), so the optimiser satisfies the backoff by moving
  whichever body is cheaper -- and it moves the TARGET. Measured on car-mini-new: at every
  assemble the target assembly obj_0 slides 16.8-48.8mm between the funnel mouth and the seat
  while the inserted part covers only 6.1-35.8mm of the nominal 50mm. The exported funnel is
  then not an approach at all, and inside a --fuse-cograsp span this third keyframe is exactly
  the co-grasp drift fusion just removed (holder arms measured 0.085-0.200 rad off).

  THE FIX. Re-solve each pre keyframe with every arm that is NOT supposed to move PINNED by a
  hard qItself eq to the skill witness (RobotIK._add_pin). "Not supposed to move" = every robot
  except the actor and the holders of the group the skill actually moves -- the co-holders must
  ride along with it, so pinning them would be wrong (and infeasible). The pin is a CONSTRAINT,
  not another cost: an eq at 1e1 stops the world from absorbing the offset, a sos at 1e0 (the
  existing pull) never could.

  The SKILL keyframe is never re-solved. That is what makes this compose for free with
  --fuse-cograsp (a fused pair stays exactly as fused) and with --strict-collisions.

  ALL-OR-NOTHING per split (see _pre_gate): reject -> today's keyframe is kept, with the reason
  printed. The exported TASK STRUCTURE is identical either way -- the split always emits its two
  tasks -- because sim.py --v2 rebuilds the task list from the saved config alone and any
  per-action structural decision the exporter made would shift its task ids.

  Returns (pre_witnesses, report). pre_witnesses maps a plan index to the keyframe mp_export
  should use for that split (accepted or kept); mp_export.make_problem takes it as
  `pre_witnesses=` and solves nothing itself."""
  import time
  t0 = time.perf_counter()
  deltas = {"pick": robot_ik.pre_pick_delta, "place": robot_ik.pre_place_delta,
            "add_grasp": robot_ik.pre_add_grasp_delta,
            "assemble": robot_ik.pre_insertion_delta}
  split = [i for i, a in enumerate(plan) if deltas.get(a[0], 0.0) > 0.0]
  if not split:
    if verbose:
      print("[pre] no --pre-* split is active (--skill / --pre-pick / --pre-place / "
            "--pre-add-grasp / --pre-insertion): nothing to make consistent.")
    return {}, {"splits": 0, "consistent": 0, "rows": []}
  if grasp_table is None:
    _checker, grasp_table = _search_context(C, assembly, robots)
  states = _replay_states(C, assembly, plan, robots, gripper_kind, handovers, allow_place,
                          bracing, grasp_table, "fuse_pre_skill_witnesses")
  names = list(robot_ik.C.getJointNames())
  jidx = {r: [k for k, n in enumerate(names) if n.startswith(r + "_")] for r in robots}
  # penetration budget: the --strict-collisions gate when one is configured, else a 0.1mm
  # floor. Either way the repaired keyframe may never be dirtier than the one it replaces.
  pen_gate = robot_ik.penetration_reject if robot_ik.penetration_reject is not None else 1e-4

  pre, rows, pin_robots = {}, [], {}
  for i in split:
    a = plan[i]
    kind, actor, delta = a[0], a[1], deltas[a[0]]
    s0, s1 = states[i], states[i + 1]
    w = witnesses[i]
    tag = f"[pre] {i:02d} {kind} {' '.join(str(x) for x in a[1:])}"
    # BEFORE is today's export-time call, verbatim -- both the measurement baseline AND the
    # fallback, so a rejected split exports exactly what it exports today.
    before = robot_ik.pre_skill_witness(a, s0, s1, assembly, w)
    pre[i] = before
    if w is None:
      rows.append((i, kind, False, "no skill witness", None, None))
      if verbose:
        print(f"  {tag}: SKIPPED (no skill witness to be consistent with)")
      continue

    # WHO MAY MOVE, and WHAT the approach is supposed to displace.
    if kind == "assemble":
      part, onto = a[2], a[3]
      moving = set(next((g for g in s0.subassemblies() if part in g), (part,)))
      axis = assembly.mate_axis(part)
      body, ffr = part, [part, onto]                       # part travels +mate_axis in onto's frame
      fpos, fquat = ry.FS.positionRel, ry.FS.quaternionRel
    elif kind == "place":
      root = a[2]
      moving = set(next((g for g in s0.subassemblies() if root in g), (root,)))
      axis = None
      body = min(moving, key=_part_index)                  # group ref descends -delta in world z
      ffr = [body]
      fpos, fquat = ry.FS.position, ry.FS.quaternion
    else:                                                  # pick / add_grasp: the GRIPPER
      moving, axis = set(), None                           # approaches; NOTHING in the scene moves
      body, ffr = robot_ik.gripper_frame[actor], [robot_ik.gripper_frame[actor], a[2]]
      fpos, fquat = ry.FS.positionRel, ry.FS.quaternionRel
    movers = {actor} | {r for r, (pt, _g) in robot_ik.holders(s0).items() if pt in moving}
    pin = {names[k]: float(w[k]) for r in robots if r not in movers for k in jidx[r]}
    if not pin:
      rows.append((i, kind, False, "no arm to pin (every robot moves with the skill)", None, None))
      if verbose:
        print(f"  {tag}: SKIPPED (no arm to pin -- every robot co-moves)")
      continue

    # SCORING SCENE: the skill's own scene and the skill's own contact exclusions, so a
    # penetration here is comparable with the one the skill IK itself gated on.
    Cs = robot_ik.scene_for(s0, assembly)
    if not robot_ik.keep_contacts:
      if kind in ("pick", "add_grasp"):
        set_part_contact(Cs, a[2], 0)                      # the gripper is meant to touch it
      elif kind == "assemble":
        for mp in moving:
          set_part_contact(Cs, mp, 0)                      # the seating contact is intended
    parts = [p for p in s0.bodies() if p in Cs.getFrameNames()]
    stat = [p for p in parts if p not in moving]

    def read(q):
      Cs.setJointState(np.asarray(q, dtype=float))
      fp, _ = Cs.eval(fpos, ffr)
      fq, _ = Cs.eval(fquat, ffr)
      grasps = [(Cs.eval(ry.FS.positionRel, [robot_ik.gripper_frame[r], pt])[0],
                 Cs.eval(ry.FS.quaternionRel, [robot_ik.gripper_frame[r], pt])[0])
                for r, pt, _t in robot_ik.maintenance_grasps(s0, assembly)]
      return {"feat": (np.asarray(fp, dtype=float), np.asarray(fq, dtype=float)),
              "body": np.asarray(Cs.getFrame(body).getPose(), dtype=float),
              "parts": {p: np.asarray(Cs.getFrame(p).getPose(), dtype=float) for p in stat},
              "grasps": grasps, "pen": float(Cs.getCollisionsTotalPenetration()),
              "q": np.asarray(q, dtype=float)}

    ref = read(w)                                          # the skill keyframe: the reference
    # NOMINAL funnel vector: the world displacement the moved body SHOULD cover between the
    # pre keyframe and the skill keyframe. Derived from the same _backoff_* algebra the
    # exporter's solve used, so the two can never disagree.
    if kind == "assemble":
      onto_q = np.asarray(Cs.getFrame(onto).getPose(), dtype=float)[3:]   # Cs is still at `w`
      nom = delta * (_quat_to_R(onto_q) @ np.asarray(axis, dtype=float)) \
          if axis is not None else np.zeros(3)
      want = _backoff_mate(ref["feat"][0], axis, delta)
    elif kind == "place":
      nom = np.array([0.0, 0.0, -delta])
      want = _backoff_place(np.concatenate(ref["feat"]), delta)[:3]
    else:
      nom = delta * (_quat_to_R(ref["body"][3:]) @ np.array([0.0, 0.0, 1.0]))
      want = _backoff_ee(np.concatenate(ref["feat"]), delta)[:3]

    def metrics(q):
      if q is None:
        return None
      m = read(q)
      pos = float(np.linalg.norm(m["feat"][0] - want))
      ang = _quat_err_deg(m["feat"][1], ref["feat"][1])
      # every MAINTAINED grasp must sit at the SAME relative pose at both keyframes (scored
      # against the skill keyframe, not against the absolute target, so a single-arm-IK scene's
      # homed brace arms cancel instead of swamping the number).
      for (gp, gq), (rp, rq) in zip(m["grasps"], ref["grasps"]):
        pos = max(pos, float(np.linalg.norm(np.asarray(gp) - np.asarray(rp))))
        ang = max(ang, _quat_err_deg(gq, rq))
      smm, sdeg = 0.0, 0.0
      for p in stat:
        d_mm, d_deg = _pose_deviation(m["parts"][p], ref["parts"][p])
        smm, sdeg = max(smm, d_mm), max(sdeg, d_deg)
      return {"dq": float(np.linalg.norm(m["q"][jidx[actor]] - ref["q"][jidx[actor]])),
              "hold": max([float(np.linalg.norm(m["q"][jidx[r]] - ref["q"][jidx[r]]))
                           for r in robots if r not in movers] or [0.0]),
              "stat_mm": smm, "stat_deg": sdeg,
              "dev": float(np.linalg.norm((ref["body"][:3] - m["body"][:3]) - nom)) * 1000.0,
              "pos": pos, "ang": ang, "pen": m["pen"]}

    after = robot_ik.pre_skill_witness(a, s0, s1, assembly, w, pin=pin)
    o, n = metrics(before), metrics(after)
    why = _pre_gate(o, n, pen_gate, max_dq, hold_dq, hold_mm, pos_tol, ang_tol)
    if getattr(robot_ik, "pin_applied", None) == 0:
      # The pin bit on NOTHING: _ik_pick/_ik_place restrict the scene to the acting arm, so
      # the other arms were already frozen and this was just a second draw of the ordinary
      # un-pinned solve. Accepting it would be reporting solver noise as a repair (measured:
      # 4 of 6 "accepted" splits were exactly this). Keep today's keyframe.
      why = why or "pin bit on no joint: the scene had already frozen every arm to hold"
    if kind == "assemble" and axis is None:
      why = why or "no mate axis: the funnel has zero length"   # report it, don't hide it
    if why is None:
      pre[i] = after
      # The accepted solve is only consistent because these idle arms were fixed at their
      # skill-witness joints.  Preserve that fact for the pre-GOTO task exporter.  Otherwise
      # mr_bench treats them as free at the mode switch and tries to rediscover a safe sample
      # by chance (fixture-rack repeatedly exhausted 10k samples beside the held rail).
      pin_robots[i] = [r for r in robots if r not in movers]
    rows.append((i, kind, why is None, why, o, n))
    if verbose:
      pair = lambda k, f, u: ((f % o[k] if o else "n/a") + "->" +
                              (f % n[k] if n else "n/a") + u)
      print(f"  {tag}: actor dq {pair('dq', '%.3f', ' rad')}"
            f" | funnel dev {pair('dev', '%.1f', ' mm')}"
            f" | idle arms {pair('hold', '%.3f', ' rad')}"
            f" | static {pair('stat_mm', '%.1f', ' mm')}"
            + (f"   REJECTED: {why} -- today's keyframe kept" if why else "   ACCEPTED"))

  acc = sum(1 for r in rows if r[2])
  if verbose:
    print(f"[pre] {acc}/{len(rows)} pre-skill keyframe(s) made consistent in "
          f"{time.perf_counter() - t0:.1f}s; the rest keep today's solve. The exported task "
          f"structure is unchanged either way.")
  return pre, {"splits": len(rows), "consistent": acc, "rows": rows,
               "pin_robots": pin_robots}


def utilization_report(C, assembly, plan, robots, witnesses=None, verbose=True,
                       gripper_kind=None, handovers=True, allow_place=True, bracing=False):
  """SCHEDULED-MAKESPAN utilization metric (report only -- no optimization). List-schedules
  the plan on its RESOURCE partial order: each action starts when every action sharing a
  resource with it has finished; resources = the involved robots (actor + holders of the
  moved/target group) and every part of the touched group(s) in pre+post state (the same
  last-writer order mp_export._dependency_graph exports). Same-robot actions serialize;
  different-robot actions on independent parts overlap -- so makespan/busy%% quantify how
  parallelizable the plan is, which the action-count objective ignores.

  Durations: with `witnesses`, an action costs the max over its involved arms of the
  joint-space distance from that arm's previous commanded config (releases: 0 -- gripper
  open); units = radians. Without witnesses, 1.0 per action (structural parallelism only).
  Returns {makespan, serial, speedup, busy: {r: time}, durations}."""
  # re-derive with the SEARCH's grasp table: the on-the-fly fallback enumerates fewer grasps
  # (rest-face pruning), so a plan action (e.g. an inhand_reorient to a table-blocked grasp)
  # may not re-derive without it.
  _checker, grasp_table = _search_context(C, assembly, robots)
  states = [get_initial_state(C, assembly)]
  for a in plan:
    states.append(next(ns for act, ns in successors(
        states[-1], assembly, robots, grasp_table=grasp_table, gripper_kind=gripper_kind,
        handovers=handovers, allow_place=allow_place, bracing=bracing) if act == a))

  def holders_of(st, part):
    grp = next((g for g in st.subassemblies() if part in g), {part})
    hs = set()
    for p in grp:
      for c in st.supports.get(p, ()):
        if c[0] == "grasp":
          hs.add(c[1])
    return hs, grp

  names = list(C.getJointNames())
  idx = {r: [j for j, n in enumerate(names) if n.startswith(r + "_")] for r in robots}
  cur = {r: np.asarray(C.getJointState())[idx[r]].copy() for r in robots}

  finish, busy, durs, ends = {}, {r: 0.0 for r in robots}, [], []
  for i, a in enumerate(plan):
    kind, actor = a[0], a[1]
    inv = {actor}
    res = set()
    target = [a[2]] + ([a[3]] if kind == "assemble" else [])
    for st in (states[i], states[i + 1]):
      for p in target:
        hs, grp = holders_of(st, p)
        inv |= hs
        res |= set(grp)
    res |= inv
    if witnesses is not None:
      w = witnesses[i] if i < len(witnesses) else None
      if w is None:
        dur = 0.0                                          # release/no-IK: gripper open only
      else:
        w = np.asarray(w)
        dur = max(float(np.linalg.norm(w[idx[r]] - cur[r])) for r in inv)
        for r in inv:
          cur[r] = w[idx[r]].copy()
    else:
      dur = 1.0
    start = max((finish.get(x, 0.0) for x in res), default=0.0)
    end = start + dur
    for x in res:
      finish[x] = end
    for r in inv:
      busy[r] += dur
    durs.append(dur)
    ends.append(end)

  makespan = max(ends) if ends else 0.0
  serial = sum(durs)
  speedup = serial / makespan if makespan > 0 else 1.0
  if verbose:
    unit = "rad (witness joint-dist)" if witnesses is not None else "actions (unit durations)"
    frac = lambda r: f"{busy[r] / makespan * 100:.0f}%" if makespan > 0 else "-"
    print(f"[utilization] makespan {makespan:.2f} vs serial {serial:.2f} -> "
          f"speedup {speedup:.2f}x of {len(robots)} robots "
          f"(parallel efficiency {speedup / len(robots) * 100:.0f}%); units: {unit}")
    print("[utilization] busy: " + "  ".join(f"{r} {busy[r]:.2f} ({frac(r)})"
                                             for r in robots))
  return {"makespan": makespan, "serial": serial, "speedup": speedup,
          "busy": dict(busy), "durations": durs}


def hoist_releases(C, assembly, plan, robots, gripper_kind=None, handovers=True,
                   allow_place=True, bracing=False, verbose=True):
  """POST-PASS plan rewrite (search untouched): bubble every `release` as EARLY as legal.
  The search never prefers early releases -- release placement is invisible to both cost
  (#actions unchanged) and h (release-neutral), so greedy emits them just-in-time (arm
  loiters holding until next needed). But an early release frees the ROBOT resource sooner
  (shorter robot chains in the scheduled makespan / depgraph export) and retracts the arm
  out of the workspace (less idle-arm clutter for the motion planner).

  Sound-by-construction: a release swaps one position earlier only if (a) it is applicable
  in the earlier state per the REAL successor generator (which enforces wrench stability --
  a needed hold cannot release early), (b) the swapped-past action is still applicable
  after the release, (c) both re-ordered edges pass the search's own cheap geometric check,
  and (d) the state after the pair is IDENTICAL (key) to before -- so the plan suffix and
  final state are untouched. Returns a new Plan with a consistent .states trajectory."""
  checker, grasp_table = _search_context(C, assembly, robots)

  def apply(s, a):
    return next((ns for act, ns in successors(
        s, assembly, robots, grasp_table=grasp_table, gripper_kind=gripper_kind,
        handovers=handovers, allow_place=allow_place, bracing=bracing) if act == a), None)

  acts = list(plan)
  states = [get_initial_state(C, assembly)]
  for a in acts:                                           # re-simulate (don't trust payloads)
    ns = apply(states[-1], a)
    if ns is None:
      if verbose:
        print("[hoist] plan does not re-simulate under these successor flags; skipped.")
      return plan
    states.append(ns)

  moved, shifts = 0, 0
  for j in range(len(acts)):
    if acts[j][0] != "release":
      continue
    i = j
    while i > 0:
      rel, prev_a = acts[i], acts[i - 1]
      s0 = states[i - 1]
      s_mid = apply(s0, rel)
      if s_mid is None or not cheap_feasible(rel, s0, s_mid, assembly, checker):
        break
      s_after = apply(s_mid, prev_a)
      if s_after is None or s_after != states[i + 1] or \
         not cheap_feasible(prev_a, s_mid, s_after, assembly, checker):
        break
      acts[i - 1], acts[i] = rel, prev_a
      states[i] = s_mid                                    # states[i+1] proven identical
      shifts += 1
      i -= 1
    if i != j:
      moved += 1
  if verbose:
    print(f"[hoist] releases: {moved} moved earlier ({shifts} positions total) "
          f"of {sum(1 for a in acts if a[0] == 'release')}.")
  out = Plan(acts)
  out.states = states
  return out


def _labeled_geo_ok(checker, assembly, st, ideal_set, note=""):
  """GEO-fact core (--geo-facts): floating-gripper COEXISTENCE check for a crafted labeled
  state -- can all the state's grippers physically BE at their grasps on the assembly at
  its rest face (table included; a held/Hf assembly is parked by the checker, so only its
  pose-invariant internals count)? A floating gripper in collision is in collision under
  EVERY robot/arm realization, so False refutes every concrete state mapping to the label
  (sound relay gate, h stays admissible). Loose parts are RELOCATABLE, so they are parked
  far away first -- their incidental synthetic slots must never reject. The verdict is
  xy/yaw-invariant (gripper-vs-assembly is relative, the table is a z-plane), so the
  synthetic anchor stands in for any concrete one -- PROVIDED the anchor z matches the
  real scene's convention: rest heights are computed RELATIVE to the anchor pose, and the
  real anchors carry the part-float offset (parts hover float_height above the table).
  crafted()'s _flat_poses z is arbitrary, so the group anchor is replaced by the SAME
  _initial_anchor lookup the real places() uses (problem.py) -- without it every
  near-table gripper/part 'penetrates' by exactly the float height (measured on car-mini:
  8206/10230 facts falsely False, h inflated past admissibility near the goal)."""
  ref = reference_part(ideal_set)
  init = getattr(assembly, "_initial_anchor", {}).get(ref)
  if init is not None:
    st.poses[ref] = np.asarray(init, dtype=float).copy()
  i = 0
  for p in assembly.objects:
    if p not in ideal_set:
      pose = np.asarray(st.poses[p], dtype=float).copy()
      pose[1] += 1000.0 + 5.0 * i
      st.poses[p] = pose
      i += 1
  return checker._check(st, assembly, note=note)


def _probe_grasp_kinds(st, kinds_of, all_kinds):
  # for a crafted probe state, each grasp contact as (robot_id, allowed-kinds) -- the
  # kind-usability mask of the grasp (kinds_of[(part,), part, gid]) or all present kinds.
  out = []
  for p in st.bodies():
    for c in st.supports.get(p, ()):
      if c[0] == "grasp":
        mask = kinds_of.get(((c[2],), c[2], c[3])) if kinds_of is not None else None
        out.append((c[1], tuple(sorted(mask)) if mask else all_kinds))
  return out


def _stable_any_kind(st, assembly, group, grasp_kinds, load_variants=None):
  # MIXED-EE wrench fact: is the probe stable for SOME valid kind-assignment (each grasp a
  # kind from its mask)? Optimistic over assignments -> admissible (never over-estimates
  # cost); still False when NO kind can hold (e.g. a weak vacuum on a plate under a press),
  # which is exactly the pruning the topological fallback lost. Grasp counts are tiny
  # (hold+brace+inserter), so the product over masks (<=~16) is cheap; facts are cached.
  import itertools
  if any(len(ks) == 0 for _r, ks in grasp_kinds):
    return False                                           # a grasp NO present kind can make
  if not grasp_kinds:                                      # table-only: kind irrelevant
    return (_group_stable_loaded(st, assembly, group, "two_finger", load_variants)
            if load_variants else _group_stable(st, assembly, group, "two_finger"))
  for combo in itertools.product(*(ks for _r, ks in grasp_kinds)):
    gk = {r: combo[i] for i, (r, _ks) in enumerate(grasp_kinds)}
    ok = (_group_stable_loaded(st, assembly, group, gk, load_variants)
          if load_variants else _group_stable(st, assembly, group, gk))
    if ok:
      return True
  return False


def _kind_budget_ok(label_kinds, kind_counts, G, L):
  # MIXED-EE arm budget for the labeled abstractions: every held label needs an arm whose
  # kind is in the label's usability mask. With two kinds, Hall's condition reduces to
  # "single-kind labels do not exceed that kind's arm count" (flexible labels take either).
  # Every forward-reachable state satisfies this (each real grasp IS held by a usable arm),
  # so excluding violators keeps h ADMISSIBLE while pricing e.g. "two wide plates cannot
  # both be held when only one vacuum arm exists".
  need = {}
  for coll in (G, L):
    for e in coll:
      if not isinstance(e, tuple):
        continue                                           # pure-collapse L: bare part names
      m = label_kinds.get((e[0], e[1]))
      if m is not None and len(m) == 1:
        k = next(iter(m))
        need[k] = need.get(k, 0) + 1
  return all(n <= kind_counts.get(k, 0) for k, n in need.items())


def _staged_valid_labeled(assembly, ideal, G, face, held_face):
  # STAGED-MATE gate for the labeled abstractions (--staged-mates): the exact abstract mirror
  # of the forward successor invariant (_problem._staged_ok) -- an abstract state where a
  # staged part p is assembled (p in ideal), its fastener is NOT, no grasp in G is on p, and
  # the slot alone cannot hold p at `face` (held -> identity convention, same as forward) is
  # the image of NO forward-reachable state, so excluding it keeps h ADMISSIBLE while pricing
  # the hold-until-fastened arm commitment (states that drop p early become unreachable
  # backward => their h rises / goes inf). G keeps (part, ...) in both the fine and collapsed
  # classes, so the p-has-a-grasp test is faithful under label collapse.
  for p, fx in assembly.staged_mates.items():
    if p in ideal and fx not in ideal:
      if not any(g[0] == p for g in G):
        f = _problem._IDENTITY_FACE if face == held_face else face
        if not _problem._slot_holds_alone(assembly, p, f):
          return False
  return True


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
               max_states=2_000_000, clearance=0.02, gripper_kind=None, bracing=False,
               geo_checker=None, dist_cap=None):
    import heapq
    import time as _time

    t0 = _time.perf_counter()
    self.assembly = assembly
    self.R = n_robots
    self.misses = 0
    self._bracing = bracing                                # model the add_grasp (co-grasp) edge
    self._clearance = clearance                            # see _table_aware_pick_fact
    self._gk = gripper_kind                                # non-None => WRENCH facts active
    # MIXED EEs: gk is a {robot: kind} MAP. Wrench facts then evaluate over each grasp's
    # kind MASK (exists-a-feasible-assignment; see _stable_any_kind) -- admissible, and it
    # keeps the force/torque pruning the old topological fallback threw away.
    # One wrench contract: a supplied per-robot kind map means the backward model uses it.
    # Callers that deliberately want a topological ablation pass gripper_kind=None via
    # --no-wrench-aware-h. Mixed-tool physics must not silently disappear from the heuristic.
    self._mixed = isinstance(self._gk, dict)
    self._kinds_of = getattr(grasp_table, "kinds_of", None)
    self._all_kinds = tuple(sorted(set(getattr(grasp_table, "robot_kinds", {}).values())))
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
    # MIXED EEs: per-(part, raw grasp) kind mask + arm counts, for _kind_budget_ok.
    self._kind_counts = None
    self._label_kinds = {}
    _kmask = getattr(grasp_table, "kinds_of", None)
    if _kmask is not None:
      from collections import Counter
      self._kind_counts = dict(Counter(getattr(grasp_table, "robot_kinds", {}).values()))
      self._label_kinds = {(q, g): frozenset(_kmask.get(((q,), q, g), ()))
                           for q in objs for g in self.single[q]}

    # lazy, cached table-aware pick fact -- keyed PER PART deliberately (no identical-parts
    # aliasing: per-part shapes/meshes and forces are coming; the template reuse makes the
    # extra per-part queries cheap anyway).
    self._ee_path = gripper_path
    self._pick_cache = {}

    # MIXED EEs: _ee_path may be a {kind: fact-EE path} dict; a pick fact then ORs over the
    # kinds in the GRASP'S mask (usable-by-any-present-arm => optimistic => admissible).
    self._kinds_of = getattr(grasp_table, "kinds_of", None)

    def fact_paths(S_key, pg):
      if not isinstance(self._ee_path, dict):
        return (self._ee_path,)
      ks = (self._kinds_of or {}).get((S_key, pg[0], pg[1])) or self._ee_path.keys()
      return tuple(self._ee_path[k] for k in sorted(ks))

    def pick_ok(S_key, face, pg):
      key = (S_key, face, pg)
      hit = self._pick_cache.get(key)
      if hit is None:
        hit = self._pick_cache[key] = any(
            _table_aware_pick_fact(assembly, S_key, face, pg[0], pg[1], pth,
                                   clearance=self._clearance)
            for pth in fact_paths(S_key, pg))
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
    self._ins_lv = {}
    self._tgt_lv = {}

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
        if self._mixed:
          hit = self._wfacts[key] = _stable_any_kind(
              st, assembly, set(ideal_key),
              _probe_grasp_kinds(st, self._kinds_of, self._all_kinds))
        else:
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
        # ARM BUDGET over the assemble's SIMULTANEOUS grasps (braces G2 + the inserter):
        # single-kind labels must not exceed that kind's arm count -- e.g. two TF braces
        # plus a TF inserter with only two TF arms is unrealizable however the LP balances
        # (found via the chair stress config: the oracle kept proposing exactly these).
        ok = (self._kind_counts is None
              or _kind_budget_ok(self._label_kinds, self._kind_counts,
                                 tuple(G2) + ((q, gq),), ()))
        ok = ok and w_hold_ok(ideal_s_key, G2, face)
        if ok and wrench.PARAMS.get("check_inserter", True):
          st_q = held_single(q, gq)
          lv = self._ins_lv.get(q)                        # load independent of grasp/kind
          if lv is None:
            lv = self._ins_lv[q] = _inserter_load_variants(st_q, assembly, q, onto)
          if self._mixed:
            ok = _stable_any_kind(st_q, assembly, frozenset({q}),
                                  _probe_grasp_kinds(st_q, self._kinds_of, self._all_kinds),
                                  lv)
          else:
            ok = _group_stable_loaded(st_q, assembly, frozenset({q}), self._gk, lv)
        if ok:
          st_t = crafted(ideal_s_key, G2, face)
          lvt = self._tgt_lv.get((ideal_s_key, face, q))  # load independent of G2/gq
          if lvt is None:
            lvt = self._tgt_lv[(ideal_s_key, face, q)] = \
                _insertion_load_variants(st_t, assembly, q, onto)
          if self._mixed:
            ok = _stable_any_kind(st_t, assembly, set(ideal_s_key),
                                  _probe_grasp_kinds(st_t, self._kinds_of, self._all_kinds),
                                  lvt)
          else:
            ok = _group_stable_loaded(st_t, assembly, set(ideal_s_key), self._gk, lvt)
        hit = self._wfacts[key] = ok
      return hit

    # ---- GEO facts (--geo-facts / geo_checker set): gate the backward RELAY on each
    # labeled state's floating-gripper coexistence (see _labeled_geo_ok). Raw labels =>
    # a direct check; keyed alongside the wrench facts. G empty => nothing to check (and
    # a marginal rest-face graze must never kill a gripperless state).
    def geo_ok(ideal_key, G, face):
      if geo_checker is None or not G:
        return True
      key = ("G", ideal_key, G, face)
      hit = self._wfacts.get(key)
      if hit is None:
        hit = self._wfacts[key] = _labeled_geo_ok(
            geo_checker, assembly, crafted(ideal_key, G, face), set(ideal_key),
            note=f"geo fact {ideal_key} G={G} face={face}")
      return hit

    def preds(t):
      ideal, G, face, L = t
      if not geo_ok(ideal, G, face):
        return                                            # no concrete realization -> no relay
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
      # (rest_faces is shape-frame + neutral-rest aware via placement_quat, so scatter-style
      # side-lying poses are ordinary faces here -- no separate scatter fact needed.)
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
      # inverse LOOSE-HANDOVER (cost 2): a loose part's grasp was swapped in the air
      # (needs a transiently free arm, like the fused handover above). Without this edge a
      # press-capable-but-unpickable insertion grasp DEAD-ENDS the backward relay although
      # the real plan can pick any grasp and hand over -- i.e. the grasp-labeled class was
      # INADMISSIBLE with handovers on (see the collapsed class doc); this edge restores
      # the relaxation at the swap's real fused cost, keeping the pick/press linkage as a
      # +2 PRICE instead of a hard gate.
      if cap_used < self.R:
        for q, gq in L:
          for gq2 in self.single[q]:
            if gq2 != gq:
              yield (ideal, G, face, tadd(tsub(L, (q, gq)), (q, gq2))), 2.0
      # inverse IN-HAND REORIENT (cost 1, gated on --inhand-reorient): a loose part's grasp
      # was swapped IN HAND -- no place+pick, NO free arm (unlike the handover above, hence no
      # cap_used gate). v1 FULL connectivity => any grasp reachable from any other at +1; scope
      # = loose singleton parts (L), matching the real action's lone-part restriction. GATED:
      # without the flag the capability does not exist, so the edge must be ABSENT (else h is
      # optimistic about a disabled skill). Extra edges only relax => admissible; the +1 cost
      # dominates the +2 loose-handover for the same swap when on. Upgrades to a connectivity
      # GRAPH later by gating gq2 on the (q, gq, gq2) reachable set instead of "any other".
      if _problem._INHAND_REORIENT:
        for q, gq in L:
          for gq2 in self.single[q]:
            if gq2 != gq:
              yield (ideal, G, face, tadd(tsub(L, (q, gq)), (q, gq2))), 1.0
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
    # GOAL SPEC (set_goal): on_table restricts goal states to a resting FACE (not held-in-
    # air Hf); hands_off restricts to ZERO holders. Seeding exactly the image of the real
    # goals keeps h admissible AND tight (guides toward placing down). Default seeds all.
    _gm = _problem._GOAL_MODE
    _on_tab = _gm["on_table"] or _gm["pose"] is not None
    _faces = list(rf(all_parts)) if _on_tab else [Hf] + list(rf(all_parts))
    _kmax = 0 if _gm["hands_off"] else self.R
    for k in range(_kmax + 1):
      for combo in combinations(vg_full, k):
        for face in _faces:
          if face is Hf and len(combo) < 1:
            continue
          st = (all_parts, combo, face, ())
          self.dist[st] = 0.0
          heapq.heappush(pq, (0.0, id(st), st))

    _staged_gate = _problem._STAGED_MATES and bool(assembly.staged_mates)
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
        if dist_cap is not None and nd > dist_cap:
          continue                                        # CORRIDOR cap: miss => dist > cap,
        if _staged_gate and not _staged_valid_labeled(assembly, s[0], s[1], s[2], Hf):
          continue                                        # staged-mate invariant (admissible)
        if self._kind_counts is not None and not _kind_budget_ok(
            self._label_kinds, self._kind_counts, s[1], s[3]):
          continue                                        # mixed-EE arm budget (admissible)
        if nd < self.dist.get(s, float("inf")):           # h=inf sound under cost_bound<=cap
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


_SIG_COMPONENTS = ("press", "pickface", "contact", "solohold", "braceside")


_FILL_TOUCH_ALL = False                                    # --fill-touch-all diagnostic: let a
                                                          # fill manipulate ANY part, not just
                                                          # ideal+{p}. Unrelated ASSEMBLE stays
                                                          # forbidden: otherwise the concrete
                                                          # state outruns milestone bookkeeping.


_FILL_CAPTURE_DIR = None                                   # debug: dump failing fills' ENTRY
                                                          # states (pickle) for exact offline
                                                          # replay -- set via env var
                                                          # ASSEMBLY_FILL_CAPTURE=<dir>.
import os as _os
if _os.environ.get("ASSEMBLY_FILL_CAPTURE"):
  _FILL_CAPTURE_DIR = _os.environ["ASSEMBLY_FILL_CAPTURE"]


def _capture_fill_failure(s, q, ideal, tag):
  if _FILL_CAPTURE_DIR is None:
    return
  try:
    import pickle, os
    os.makedirs(_FILL_CAPTURE_DIR, exist_ok=True)
    n = len(os.listdir(_FILL_CAPTURE_DIR))
    with open(os.path.join(_FILL_CAPTURE_DIR, f"fillfail_{n:03d}_{q}.pkl"), "wb") as f:
      pickle.dump({"state": s, "q": q, "ideal": set(ideal), "tag": tag}, f)
  except Exception as e:
    print(f"[capture] failed: {e}")


def set_fill_touch_all(on=True):
  global _FILL_TOUCH_ALL
  _FILL_TOUCH_ALL = bool(on)


def _fill_action_in_scope(action, relevant, requested_parts, touch_all=None):
  """Whether a local fill may expose this ordinary successor.

  Release is always admitted so an inherited holder cannot strand an arm. The touch-all
  ablation additionally admits manipulation of future/blocking parts, but never their assembly:
  the outer driver grows its ideal only by ``requested_parts`` and would otherwise lose agreement
  with the concrete state. ``requested_parts`` is a singleton for the ordinary fill and a tuple
  for the adaptive width-k completeness net.
  """
  kind, part = action[0], action[2]
  if touch_all is None:
    touch_all = _FILL_TOUCH_ALL
  requested = ({requested_parts} if isinstance(requested_parts, str)
               else set(requested_parts))
  if kind == "assemble":
    return part in requested
  if kind == "release":
    return True
  return part in relevant or touch_all


def _fill_scope_passes(force_all=False, widen=False):
  """Narrow-to-broad action scopes for a fill.

  The broad pass is a completeness widening, not a simultaneous branch multiplier: it is reached
  only after every local-scope grasp/bracing phase fails. ``force_all`` retains the diagnostic
  that skips the cheap pass entirely.
  """
  if force_all:
    return (True,)
  return (False, True) if widen else (False,)


def set_mixed_wrench_h(on=True):
  """Deprecated compatibility hook; mixed wrench guidance follows gripper_kind directly."""
  return None


def set_sig_components(components):
  """Ablation hook for the signature collapse (--sig-components): which components
  participate in the grasp-equivalence classes. Fewer components => coarser classes =>
  smaller abstraction build, weaker discrimination."""
  global _SIG_COMPONENTS
  _SIG_COMPONENTS = tuple(components)


def _sig_held_single_state(assembly, q, g):
  # standalone copy of the heuristics' held_single crafted state: everything loose on the
  # table, q held solo by grasp g (kinematic root irrelevant to the LP -- see crafted()).
  parent, place, supports = {}, {}, {}
  for p in assembly.objects:
    parent[p] = ("table",)
    supports[p] = frozenset({("table",)})
    place[p] = ("slot", _part_index(p), _IDENTITY_FACE)
  parent[q] = ("gripper", "zzq")
  place.pop(q, None)
  supports[q] = frozenset({("grasp", "zzq", q, g)})
  return State(parent=parent, place=place, supports=supports,
               poses=dict(_flat_poses(assembly)))


def _sig_press_ok(assembly, q, g, gk, lv=None):
  # S1: can grasp g transmit the insertion press for q's OWN mate? (the per-state
  # inserter gate of w_assemble_ok, evaluated once per (part, grasp)). `lv` lets the
  # caller hoist the load variants (they depend only on q, not on g or the kind).
  onto = assembly.relative_pose(q)[0]
  st = _sig_held_single_state(assembly, q, g)
  if lv is None:
    lv = _inserter_load_variants(st, assembly, q, onto)
  return _group_stable_loaded(st, assembly, frozenset({q}), gk, lv)


def _sig_solo_hold_ok(assembly, q, g, gk):
  # S4: can grasp g on part q hold the FULL assembly alone in the air (endgame holder /
  # handover-recipient capability)?
  parent, supports = {}, {}
  for p in assembly.objects:
    if p == assembly.base:
      parent[p] = ("gripper", "zzq")
      supports[p] = frozenset()
    else:
      onto = assembly.relative_pose(p)[0]
      parent[p] = ("part", onto)
      supports[p] = frozenset({("mate", onto)})
  supports[q] = supports[q] | {("grasp", "zzq", q, g)}
  st = State(parent=parent, place={}, supports=supports,
             poses=dict(_flat_poses(assembly)))
  return _group_stable(st, assembly, set(assembly.objects), gk)


def compute_useful_grasps(assembly, grasp_table, gripper_kind=None):
  """SPEC 6.2 useful-grasp sets, reduced to what survives a free-handover domain: U[p] =
  the grasps of p that can serve p's MATE ITSELF -- collision-free at the op (present in
  the grasp table of p's mate-context subassembly) and press-transmitting for SOME kind in
  their usability mask (per-kind wrench facts; wrench off => no press constraint).

  The full regression closure (grasp x face reachability) COLLAPSES here: one handover
  reaches any grasp, so distance-to-U is 0 / 1 / (inf only when U[p] is empty, which the
  wrench facts already expose). What remains valuable -- and what the callers consume -- is
  (a) ORDERING every grasp choice U-first (the transit gradient the flat local h lacks) and
  (b) an ADMISSIBLE +1 in the fill h when the target part is held by a non-U grasp (the
  forced regrasp/handover before its mate). Returns {part: frozenset(useful grasp ids)}."""
  kinds_of = getattr(grasp_table, "kinds_of", None)
  robot_kinds = getattr(grasp_table, "robot_kinds", None)
  present = sorted(set(robot_kinds.values())) if robot_kinds else None
  out = {}
  for p in assembly.objects:
    if assembly.relative_pose(p) is None:
      continue                                            # base: no mate
    ids = sorted({g for _q, g, _ in grasp_table.get((p,), ())})
    if not ids:
      continue
    ctx = _subasm_key(_mate_closure(assembly, p) | {p})
    ctx_rows = grasp_table.get(ctx)
    at_op = {g for q, g, _ in ctx_rows if q == p} if ctx_rows is not None else set(ids)
    if gripper_kind is None:
      out[p] = frozenset(at_op)
      continue
    lv = None                                             # hoisted press load (per part)

    def pressable(g):
      nonlocal lv
      if lv is None:
        st0 = _sig_held_single_state(assembly, p, g)
        lv = _inserter_load_variants(st0, assembly, p, assembly.relative_pose(p)[0])
      ks = (kinds_of or {}).get(((p,), p, g)) if kinds_of else None
      kk = sorted(ks) if ks else (present or [gripper_kind] if isinstance(gripper_kind, str)
                                  else sorted(set(gripper_kind.values())))
      return any(_sig_press_ok(assembly, p, g, k, lv=lv) for k in kk)

    out[p] = frozenset(g for g in at_op if pressable(g))
  return out


def compute_grasp_signatures(assembly, grasp_table, gripper_kind=None, gripper_path=None,
                             clearance=0.02, components=None):
  """FACT-SIGNATURE grasp-equivalence classes, one map per part: {part: {raw_g: class}}.

  Two grasps share a class iff they agree on every enabled signature component -- i.e. on
  everything the abstraction's facts (are believed to) read from a grasp id:
    press     -- S1: transmits the part's own insertion press (wrench LP, needs gk)
    pickface  -- S2: per-rest-face table-aware pickability vector (splits wrist tilts
                 that dip toward the table; the component g%%nb wrongly merges)
    contact   -- S3: contact patch geometry (center, approach axis, closing axis up to
                 sign) -- the holder/brace lever-arm class; merges wrist tilts (rotation
                 ABOUT the closing axis), keeps faces/rolls apart
    solohold  -- S4: can hold the full assembly alone in the air (wrench LP, needs gk)
    braceside -- S5: alignment bits of the approach axis vs the assembly's mate axes
                 (the 'hold the opposite end and push' brace role)
  Classes are used as LABELS with the existing per-context OR over class members
  (_vgroups/_svgroups), so an incomplete signature only degrades to the usual
  OR-relaxation -- admissibility is never at stake. Returned class ids are per-part,
  deterministic (sorted signatures)."""
  comps = set(components if components is not None else _SIG_COMPONENTS)
  ee = gripper_path or os.path.join(os.path.dirname(__file__), DEFAULT_FLOATING_EE)
  # mixed EEs: a {kind: path} dict -> the pickface component records one vector PER KIND
  # (kind-tagged, so signatures split exactly where the kinds disagree).
  ee_paths = sorted(ee.items()) if isinstance(ee, dict) else [(None, ee)]
  axes = []
  for r in assembly.objects:
    rel = assembly.relative_pose(r)
    if rel is None:
      continue
    t = np.asarray(rel[1], dtype=float)
    n = np.linalg.norm(t)
    if n > 1e-9:
      u = tuple(np.round(t / n, 2))
      if u not in axes:
        axes.append(u)
  axes.sort()
  sigs = {}
  _press_lv = {}                                           # per-part hoisted press loads

  def press_lv(q):
    lv = _press_lv.get(q)
    if lv is None:
      st0 = _sig_held_single_state(assembly, q, 0)
      lv = _press_lv[q] = _inserter_load_variants(st0, assembly, q,
                                                  assembly.relative_pose(q)[0])
    return lv
  for q in assembly.objects:
    ids = sorted({g for _p, g, _ in grasp_table[(q,)]})
    faces = list(rest_faces(assembly, {q}))
    table = {}
    for g in ids:
      pose = grasp_pose(assembly, q, g)
      R = _quat_to_R(np.asarray(pose[3:], dtype=float))
      appr = R @ np.array([0.0, 0.0, 1.0])
      close = R @ np.array([0.0, 1.0, 0.0])
      for c in close:                                     # closing axis is sign-symmetric
        if abs(c) > 1e-6:
          close = close * np.sign(c)
          break
      sig = []
      # MIXED EEs: the grasp's KIND-USABILITY mask always joins the signature (independent
      # of --sig-components): grasps usable by different gripper sets are NOT interchangeable
      # as labels -- collapsing a vacuum-only cap grasp with a two-finger-only side pinch
      # would tell the abstraction any arm can serve either, and the fills then burn real-IK
      # binds discovering "wrong arm" one refutation at a time. Uniform scenes have no mask
      # (kinds_of is None) -> component absent -> byte-identical classes.
      kmask = getattr(grasp_table, "kinds_of", None)
      if kmask is not None:
        sig.append(("gripper", tuple(sorted(kmask.get(((q,), q, g), ())))))
      if "press" in comps and gripper_kind is not None and q != assembly.base:
        # MIXED EEs (gripper_kind is a {robot:kind} map): evaluate at the grasp's own
        # kind(s) from its mask, OR-ing -- the press-transmissible-by-SOME-usable-kind fact.
        if isinstance(gripper_kind, dict):
          ks = kmask.get(((q,), q, g), ()) if kmask else ()
          val = any(_sig_press_ok(assembly, q, g, k, lv=press_lv(q)) for k in ks)
        else:
          val = _sig_press_ok(assembly, q, g, gripper_kind, lv=press_lv(q))
        sig.append(("press", bool(val)))
      if "pickface" in comps:
        sig.append(("faces", tuple(
            (k, tuple(bool(_table_aware_pick_fact(assembly, (q,), f, q, g, pth,
                                                  clearance=clearance)) for f in faces))
            for k, pth in ee_paths)))
      if "contact" in comps:
        # STATIC grip capability only: the pads' patch = (center, closing axis). The
        # approach/wrist direction is deliberately EXCLUDED -- it does not change the
        # grip wrench (tilt variants rotate about the closing axis, pads unchanged), and
        # its clearance effects are what the pickface component measures with the true
        # EE pose. Including it makes S3 the identity map (measured: 12/12, 36/36).
        sig.append(("contact", tuple(np.round(np.asarray(pose[:3], dtype=float), 3)),
                    tuple(np.round(close, 2))))
      if "solohold" in comps and gripper_kind is not None and assembly.can_hold(q):
        if isinstance(gripper_kind, dict):                 # per-kind (see press, above)
          ks = kmask.get(((q,), q, g), ()) if kmask else ()
          val = any(_sig_solo_hold_ok(assembly, q, g, k) for k in ks)
        else:
          val = _sig_solo_hold_ok(assembly, q, g, gripper_kind)
        sig.append(("hold", bool(val)))
      if "braceside" in comps:
        # brace force is transmitted through the PADS (closing axis, sign-symmetric),
        # not along the wrist approach.
        sig.append(("brace", tuple(
            int(abs(float(np.dot(close, np.asarray(a, dtype=float)))) > 0.9)
            for a in axes)))
      table[g] = tuple(sig)
    classes = {s: i for i, s in enumerate(sorted(set(table.values()), key=repr))}
    sigs[q] = {g: classes[table[g]] for g in ids}
  return sigs


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
  (The fine class now models that swap EXPLICITLY -- inverse LOOSE-HANDOVER, cost 2 -- so
  its linkage is an admissible +2 price rather than a hard gate; 2026-07-17.)

  Kept fully separate from the fine class (own flag `collapsed_h`) so the well-tested one
  stays untouched; the duplication is the price of that isolation."""

  H = "H"                                                  # face value for held-in-air

  def __init__(self, assembly, n_robots, grasp_table, gripper_path=None, verbose=False,
               max_states=2_000_000, clearance=0.02, gripper_kind=None, bracing=False,
               collapse_grasps=False, geo_checker=None, grasp_classes=None,
               labeled_loose=False, dist_cap=None, ideal_chain=None,
               component_root=None, fill_goal=False, lazy_build=False):
    import heapq
    import time as _time

    t0 = _time.perf_counter()
    self.assembly = assembly
    self.R = n_robots
    self.misses = 0
    self._bracing = bracing                                # model the add_grasp (co-grasp) edge
    self._clearance = clearance
    self._gk = gripper_kind
    # Same contract as the fine labeled heuristic: a supplied kind map activates mixed-tool
    # wrench facts. The explicit wrench-aware-h ablation passes None at the caller.
    self._mixed = isinstance(self._gk, dict)
    self._kinds_of = getattr(grasp_table, "kinds_of", None)
    self._all_kinds = tuple(sorted(set(getattr(grasp_table, "robot_kinds", {}).values())))
    self._collapse = bool(collapse_grasps)                 # see GraspCollapsedHeuristic
    objs = list(assembly.objects)
    base = assembly.base if component_root is None else component_root
    if base not in objs:
      raise ValueError(f"component_root {base!r} is not an assembly object")
    self._component_root = base
    self._fill_goal = bool(fill_goal)
    self._lazy_build = bool(lazy_build)
    # A chain projection retains the FULL remaining-order signal but restricts the abstract
    # ideals to one caller-selected prefix chain and permits only that prefix's NEXT part to
    # be loose. This removes the unrelated loose-part/order-ideal cross product while keeping
    # exactly the future awareness a purely two-layer local heuristic lacked on car-mini.
    if ideal_chain is None:
      all_parts = _subasm_key(objs)
      chain = None
    else:
      chain = [_subasm_key(s) for s in ideal_chain]
      if not chain or base not in chain[0]:
        raise ValueError("ideal_chain must start with an ideal containing component_root")
      if len(set(chain)) != len(chain):
        raise ValueError("ideal_chain contains a repeated prefix")
      all_parts = chain[-1]
    self._tracked_loose = (None if chain is None else
                           frozenset(all_parts) - frozenset(chain[0]))
    self._closure = {p: frozenset(_mate_closure(assembly, p)) for p in objs}

    def tadd(t, x):
      return tuple(sorted(t + (x,)))

    def tsub(t, x):
      i = t.index(x)
      return t[:i] + t[i + 1:]

    ideal_keys = ([_subasm_key(s) for s in get_subassemblies(assembly) if base in s]
                  if chain is None else chain)
    ideal_key_set = set(ideal_keys)
    chain_prev, chain_next = {}, {}
    if chain is not None:
      for before, after in zip(chain, chain[1:]):
        added = set(after) - set(before)
        if not added or not set(before) < set(after):
          raise ValueError("each ideal_chain prefix must add a non-empty component")
        chain_prev[after] = (before, frozenset(added))
        chain_next[before] = tuple(sorted(added))
    # OPTIONAL DENSE-GRASP LABEL COLLAPSE (collapse_grasps=True, the GraspCollapsedHeuristic):
    # with the wrist-tilt variants on, grasp ids come in three blocks [base|+45|-45]; raw ids
    # as ROBOT LABELS explode the coreachable product space ~3^R (measured 24x on cross-4/3R).
    # The collapse labels by the BASE id and a fact is true if ANY variant assignment makes it
    # true (an OR = a further RELAXATION -> admissibility preserved); the real search keeps
    # the full variant set. Off (the default) => gcol is the identity and everything below
    # reduces exactly to the raw-label behaviour (singleton variant groups).
    nb = len(_problem._CUBE_FACE_NORMALS) * _problem._ROLLS_PER_FACE
    if grasp_classes is not None:                          # SIGNATURE collapse (per part):
      gcol = lambda p, g: grasp_classes.get(p, {}).get(g, 10000 + g)
      self._collapse = True                                # abstract() must map via gcol
    elif self._collapse and _problem._DENSE_GRASPS:
      gcol = lambda p, g: g % nb
    else:
      gcol = lambda p, g: g
    self._gcollapse = gcol
    self._labeled_loose = bool(labeled_loose)              # L keeps (part, class) labels
    self._vgroups = {}                                     # S_key -> {(p,gb): (raw g, ...)}
    self.valid = {}
    for S in ideal_keys:
      groups = {}
      for p, g, _ in grasp_table[S]:
        groups.setdefault((p, gcol(p, g)), []).append(g)
      self._vgroups[S] = {k: tuple(sorted(v)) for k, v in groups.items()}
      self.valid[S] = tuple(sorted(groups))
    self._svgroups = {}                                    # q -> {gb: (raw g, ...)}
    self.single = {}
    for q in objs:
      groups = {}
      for _p, g, _ in grasp_table[(q,)]:
        groups.setdefault(gcol(q, g), []).append(g)
      self._svgroups[q] = {k: tuple(sorted(v)) for k, v in groups.items()}
      self.single[q] = tuple(sorted(groups))
    # MIXED EEs: per-(part, label) kind-usability mask (union over the class's raw variants)
    # + arm counts per kind, for the _kind_budget_ok validity gate. None when uniform.
    self._kind_counts = None
    self._label_kinds = {}
    _kmask = getattr(grasp_table, "kinds_of", None)
    if _kmask is not None:
      from collections import Counter
      self._kind_counts = dict(Counter(getattr(grasp_table, "robot_kinds", {}).values()))
      for q in objs:
        for lbl, variants in self._svgroups[q].items():
          m = set()
          for g in variants:
            m |= set(_kmask.get(((q,), q, g), ()))
          self._label_kinds[(q, lbl)] = frozenset(m)

    def gvariants(S_key, pg):                              # raw variants of a collapsed label
      m = self._vgroups.get(S_key)
      if m is not None and pg in m:
        return m[pg]
      return self._svgroups.get(pg[0], {}).get(pg[1], (pg[1],))

    import itertools as _it

    def gcombos(S_key, G):                                 # every raw-variant assignment of G
      if not G:
        return [()]
      return _it.product(*[[(p, g) for g in gvariants(S_key, (p, gb))]
                           for p, gb in G])

    self._ee_path = gripper_path
    self._pick_cache = {}

    # MIXED EEs: same OR-over-mask-kinds as the fine class (see there), composed with the
    # collapsed label's raw variants.
    self._kinds_of = getattr(grasp_table, "kinds_of", None)

    def fact_paths(S_key, part, g):
      if not isinstance(self._ee_path, dict):
        return (self._ee_path,)
      ks = (self._kinds_of or {}).get((S_key, part, g)) or self._ee_path.keys()
      return tuple(self._ee_path[k] for k in sorted(ks))

    def pick_ok(S_key, face, pg):
      key = (S_key, face, pg)
      hit = self._pick_cache.get(key)
      if hit is None:
        hit = self._pick_cache[key] = any(
            _table_aware_pick_fact(assembly, S_key, face, pg[0], g, pth,
                                   clearance=self._clearance)
            for g in gvariants(S_key, pg) for pth in fact_paths(S_key, pg[0], g))
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
    self._ins_lv = {}
    self._tgt_lv = {}

    def crafted(ideal_key, G, face):
      parent, place, supports = {}, {}, {}
      ideal_set = set(ideal_key)
      for q in assembly.objects:
        if q in ideal_set:
          if q == base:
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
      # collapsed labels: stable if ANY raw-variant assignment of G is (OR = relaxation).
      if self._gk is None:
        return True
      key = ("H", ideal_key, G, face)
      hit = self._wfacts.get(key)
      if hit is None:
        def hold(st):
          return (_stable_any_kind(st, assembly, set(ideal_key),
                                   _probe_grasp_kinds(st, self._kinds_of, self._all_kinds))
                  if self._mixed
                  else _group_stable(st, assembly, set(ideal_key), self._gk))
        hit = self._wfacts[key] = any(
            hold(crafted(ideal_key, Gv, face)) for Gv in gcombos(ideal_key, G))
      return hit

    def w_assemble_ok(ideal_s_key, G2, face, q, gq):
      if self._gk is None:
        return True
      key = ("A", ideal_s_key, G2, face, q, gq)
      hit = self._wfacts.get(key)
      if hit is None:
        onto = assembly.relative_pose(q)[0]
        ok = (self._kind_counts is None                   # arm budget incl. the inserter --
              or _kind_budget_ok(self._label_kinds, self._kind_counts,
                                 tuple(G2) + ((q, gq),), ()))  # see fine class
        ok = ok and w_hold_ok(ideal_s_key, G2, face)
        if ok and wrench.PARAMS.get("check_inserter", True):
          lv = self._ins_lv.get(q)                        # load independent of grasp/kind:
          if lv is None:                                  # compute ONCE, reuse across every
            st0 = held_single(q, self._svgroups.get(q, {}).get(gq, (gq,))[0])
            lv = self._ins_lv[q] = _inserter_load_variants(st0, assembly, q, onto)

          def ins_ok(graw):
            st_q = held_single(q, graw)
            return (_stable_any_kind(st_q, assembly, frozenset({q}),
                                     _probe_grasp_kinds(st_q, self._kinds_of, self._all_kinds), lv)
                    if self._mixed
                    else _group_stable_loaded(st_q, assembly, frozenset({q}), self._gk, lv))
          ok = any(ins_ok(graw)
                   for graw in self._svgroups.get(q, {}).get(gq, (gq,)))
        if ok:
          lvt = self._tgt_lv.get((ideal_s_key, face, q))  # load independent of G2 combo
          if lvt is None:
            lvt = self._tgt_lv[(ideal_s_key, face, q)] = _insertion_load_variants(
                crafted(ideal_s_key, (), face), assembly, q, onto)

          def tgt_ok(G2v):
            st_t = crafted(ideal_s_key, G2v, face)
            return (_stable_any_kind(st_t, assembly, set(ideal_s_key),
                                     _probe_grasp_kinds(st_t, self._kinds_of, self._all_kinds), lvt)
                    if self._mixed
                    else _group_stable_loaded(st_t, assembly, set(ideal_s_key), self._gk, lvt))
          ok = any(tgt_ok(G2v) for G2v in gcombos(ideal_s_key, G2))
        hit = self._wfacts[key] = ok
      return hit

    # ---- GEO facts (--geo-facts / geo_checker set): gate the backward RELAY on each
    # labeled state's floating-gripper coexistence (see _labeled_geo_ok). Collapsed
    # labels: geometrically possible if ANY raw-variant assignment is (OR = a further
    # relaxation, admissibility preserved -- the same pattern as w_hold_ok).
    def geo_ok(ideal_key, G, face):
      if geo_checker is None or not G:
        return True
      key = ("G", ideal_key, G, face)
      hit = self._wfacts.get(key)
      if hit is None:
        hit = self._wfacts[key] = any(
            _labeled_geo_ok(geo_checker, assembly, crafted(ideal_key, Gv, face),
                            set(ideal_key),
                            note=f"geo fact {ideal_key} G={Gv} face={face}")
            for Gv in gcombos(ideal_key, G))
      return hit

    def preds(t):
      ideal, G, face, L = t                               # L: bare part names
      if not geo_ok(ideal, G, face):
        return                                            # no concrete realization -> no relay
      cap_used = len(G) + len(L)
      # inverse ASSEMBLE/JOIN: one held grasp belongs to the component added by this chain
      # edge. For the ordinary case that component is {q}; for a module join it may contain
      # several already-mated parts and the held grasp may be on any member. The predecessor
      # represents that entire mover optimistically by its one held grasp label in L.
      for held_part, gq in G:
        if held_part == base:
          continue
        if chain is not None:
          edge = chain_prev.get(ideal)
          if edge is None:
            continue
          prev_ideal, added = edge
          if held_part not in added:
            continue
          seams = [q for q in added
                   if assembly.relative_pose(q) is not None
                   and assembly.relative_pose(q)[0] in prev_ideal]
          if len(seams) != 1:
            continue
          q = seams[0]
          # All grasps on the moving component disappear from target-side G2. Ignoring an
          # additional mover brace is an admissible relaxation and frees capacity.
          G2 = tuple(pg for pg in G if pg[0] not in added)
        else:
          q = held_part
          if any(q in self._closure[r] for r in ideal if r != q):
            continue                                      # not maximal -> not last-added
          prev_ideal = tsub(ideal, q)
          added = frozenset({q})
          G2 = tsub(G, (q, gq))
        if face is not Hf and not _on_table_mateable_cached(self.assembly, q, face):
          continue
        if prev_ideal not in ideal_key_set:
          continue
        if any(p in added for p, _ in G2):
          continue                                        # mover was represented by one holder
        if face is Hf and len(G2) < 1:
          continue                                        # target had to be supported
        # The existing wrench fact is exact for a singleton mover grasped on its seam. A
        # module grasped elsewhere needs a component-load fact not yet represented here;
        # admitting it is the safe relaxed choice (concrete wrench gates remain authoritative).
        if (len(added) == 1 and held_part == q
            and not w_assemble_ok(prev_ideal, G2, face, q, gq)):
          continue                                        # wrench: press not transmittable
        yield (prev_ideal, G2, face, tadd(
            L, (held_part, gq) if self._labeled_loose else held_part)), 1.0
      if self._labeled_loose:
        # inverse PICK-LOOSE, labeled: the loose (part, class) must be pickable AS THAT
        # CLASS (exists-a-face, OR over class variants) -- the pick/press linkage at class
        # resolution; a swap is available via the LOOSE-HANDOVER edge below. (rest_faces is
        # shape-frame + neutral-rest aware via placement_quat -- scatter-style side-lying
        # poses are ordinary faces here.)
        for q, gq in L:
          if any(pick_ok((q,), f, (q, gq)) for f in rf((q,))):
            yield (ideal, G, face, tsub(L, (q, gq))), 1.0
      else:
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
        # SAME-LABEL variant: collapsed labels DEDUP -- two concrete holders can share one
        # label, so a second same-label grasp can lift the assembly with G unchanged.
        # Needs >=2 raw variants (two arms cannot share one raw grasp pose) + a free arm.
        if cap_used < self.R and len(gvariants(ideal, pg)) >= 2:
          for f in rf(ideal):
            if pick_ok(ideal, f, pg):
              yield (ideal, G, f, L), 1.0
      # inverse PLACE-ASSEMBLY: on table now -> was held by one more grasp in the air
      if face is not Hf and cap_used < self.R:
        for pg in self.valid[ideal]:
          if pg not in G:
            yield (ideal, tadd(G, pg), Hf, L), 1.0
        # SAME-LABEL variant: the placing holder shared its label with a remaining one, so
        # the forward place leaves G UNCHANGED (Hf -> face) -- the distinct-label edge
        # above cannot produce it. MEASURED: without this, coarse --sig-components classes
        # made car-mini's s0 backward-unreachable => false INFEASIBLE (the coarser the
        # collapse, the more same-label holder pairs; the fine class is immune).
        if any(len(gvariants(ideal, pg)) >= 2 for pg in G):
          yield (ideal, G, Hf, L), 1.0
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
      if self._labeled_loose:
        # inverse LOOSE-HANDOVER (cost 2): a loose part's grasp CLASS was swapped in the
        # air (transiently free arm required) -- keeps the pick/press linkage admissible
        # (+2 price, not a hard gate; mirrors the fine class edge).
        if cap_used < self.R:
          for q, gq in L:
            for gq2 in self.single[q]:
              if gq2 != gq:
                yield (ideal, G, face, tadd(tsub(L, (q, gq)), (q, gq2))), 2.0
        # inverse IN-HAND REORIENT (cost 1, gated): loose part's grasp CLASS swapped in hand,
        # no free arm needed (see the fine class for the full rationale). Pure-collapse mode
        # forgets loose grasps entirely -> reorient is a no-op there, so only labeled_loose.
        if _problem._INHAND_REORIENT:
          for q, gq in L:
            for gq2 in self.single[q]:
              if gq2 != gq:
                yield (ideal, G, face, tadd(tsub(L, (q, gq)), (q, gq2))), 1.0
        # inverse PLACE-LOOSE, labeled: one edge per (part, class)
        if cap_used < self.R:
          for q in (objs if chain is None else chain_next.get(ideal, ())):
            if q in ideal or any(q == p for p, _ in L):
              continue
            for qg in self.single[q]:
              yield (ideal, G, face, tadd(L, (q, qg))), 1.0
      else:
        # inverse PLACE-LOOSE: ONE edge per part (any valid grasp; existence suffices)
        if cap_used < self.R:
          for q in (objs if chain is None else chain_next.get(ideal, ())):
            if q in ideal or q in L:
              continue
            if self.single[q]:
              yield (ideal, G, face, tadd(L, q)), 1.0

    # goal states: identical seeding to the fine class (L=() either way)
    self.dist = {}
    pq = []
    from itertools import combinations
    vg_full = self.valid[all_parts]
    # A fill-local PDB has its OWN goal: the requested part is mated, no tracked loose part
    # remains, and the component is either on a stable table face or held by exactly one
    # carrier. This matches _add_one_part.goal_fn directly and removes the former shifted
    # full-assembly heuristic. The ordinary constructor path retains the global set_goal spec.
    if self._fill_goal:
      _faces = [Hf] + list(rf(all_parts))
      _kmax = 1
    else:
      _gm = _problem._GOAL_MODE
      _on_tab = _gm["on_table"] or _gm["pose"] is not None
      _faces = list(rf(all_parts)) if _on_tab else [Hf] + list(rf(all_parts))
      _kmax = 0 if _gm["hands_off"] else self.R
    for k in range(_kmax + 1):
      for combo in combinations(vg_full, k):
        for face in _faces:
          if face is Hf and len(combo) < 1:
            continue
          st = (all_parts, combo, face, ())
          self.dist[st] = 0.0
          heapq.heappush(pq, (0.0, id(st), st))

    _staged_gate = _problem._STAGED_MATES and bool(assembly.staged_mates)
    self._settled = set()
    self._reverse_pops = 0
    self._reverse_time = 0.0
    self._reverse_exhausted = False

    def advance(target=None):
      """Resume reverse Dijkstra until ``target`` is settled, or exhaust the graph.

      ``dist`` contains tentative labels as usual; callers may consume a value only after
      its state has been popped/settled. The closure owns the predecessor/fact caches built
      above, so a later query continues byte-for-byte from the retained frontier.
      """
      ta = _time.perf_counter()
      try:
        while pq:
          d, _, t = heapq.heappop(pq)
          if d > self.dist.get(t, float("inf")) or t in self._settled:
            continue
          self._settled.add(t)
          self._reverse_pops += 1
          if len(self.dist) > max_states:
            raise RuntimeError(f"collapsed labeled abstraction exceeded {max_states} states")
          for s, c in preds(t):
            nd = d + c
            if dist_cap is not None and nd > dist_cap:
              continue                                    # CORRIDOR cap: miss => dist > cap,
            if _staged_gate and not _staged_valid_labeled(assembly, s[0], s[1], s[2], Hf):
              continue                                    # staged-mate invariant (admissible)
            if self._kind_counts is not None and not _kind_budget_ok(
                self._label_kinds, self._kind_counts, s[1], s[3]):
              continue                                    # mixed-EE arm budget (admissible)
            if nd < self.dist.get(s, float("inf")):
              self.dist[s] = nd
              heapq.heappush(pq, (nd, id(s), s))
          # Expand the queried state before pausing.  A later query may reach one of its
          # predecessors only through this edge; returning immediately after POP would lose
          # that part of the retained frontier and make the oracle order-dependent.
          if target is not None and t == target:
            return True
        self._reverse_exhausted = True
        return target is None or target in self._settled
      finally:
        self._reverse_time += _time.perf_counter() - ta
        self.n_states = len(self.dist)

    self._advance_reverse = advance
    if not self._lazy_build:
      advance()

    self.n_states = len(self.dist)
    self.n_pick_facts = len(self._pick_cache)
    self.build_time = _time.perf_counter() - t0
    if verbose:
      tag = ("sig-h" if grasp_classes is not None else
             "raw-h" if self._labeled_loose else "collapsed-h")
      if self._fill_goal:
        tag = f"fill-{tag}"
      elif chain is not None:
        tag = f"chain-{tag}"
      mode = (f"lazy reverse oracle seeded with {self.n_states} goal states"
              if self._lazy_build else f"{self.n_states} coreachable states")
      print(f"[{tag}] {mode}, {self.n_pick_facts} pick facts, "
            f"built in {self.build_time:.2f}s")

  def abstract(self, state):
    main = next(g for g in state.subassemblies() if self._component_root in g)
    ideal = _subasm_key(main)
    if self._collapse:
      # grasp ids collapsed to their BASE variant; set-dedup so two variants of the same
      # base label map onto ONE abstract slot (fewer committed arms = more permissive).
      G = tuple(sorted({(c[2], self._gcollapse(c[2], c[3])) for p in main
                        for c in state.supports.get(p, ()) if c[0] == "grasp"}))
    else:
      G = tuple(sorted((c[2], c[3]) for p in main
                       for c in state.supports.get(p, ()) if c[0] == "grasp"))
    on_table = any(("table",) in state.supports.get(p, ()) for p in main)
    face = _placement_face(state, main) if on_table else self.H
    if self._labeled_loose:
      # loose parts keep their (part, grasp-class) label -- the fine class's pick/press
      # discrimination at class resolution (the 2026-07-17 plateau finding).
      L = tuple(sorted({(c[2], self._gcollapse(c[2], c[3])) for p in state.parent
                        if p not in main
                        and (self._tracked_loose is None or p in self._tracked_loose)
                        for c in state.supports.get(p, ()) if c[0] == "grasp"}))
    else:
      L = tuple(sorted({c[2] for p in state.parent
                        if p not in main
                        and (self._tracked_loose is None or p in self._tracked_loose)
                        for c in state.supports.get(p, ()) if c[0] == "grasp"}))
    return (ideal, G, face, L)

  def value(self, state):
    key = self.abstract(state)
    if self._lazy_build and key not in self._settled and not self._reverse_exhausted:
      self._advance_reverse(key)
      self.n_pick_facts = len(self._pick_cache)
      self.build_time = self._reverse_time
    # inf is still a PROOF after eager exhaustion. Fill-local callers deliberately treat a
    # miss as UNKNOWN, preserving their counting lower bound if a crafted projection is absent.
    v = self.dist.get(key) if key in self._settled else None
    if v is None:
      self.misses += 1
      return float("inf")
    return v

  def reverse_stats(self):
    """Small, stable instrumentation surface for eager and query-driven PDB builds."""
    return {
        "settled": len(self._settled),
        "discovered": len(self.dist),
        "pops": self._reverse_pops,
        "reverse_time": self._reverse_time,
        "pick_facts": len(self._pick_cache),
        "wrench_facts": len(self._wfacts),
        "exhausted": self._reverse_exhausted,
    }


class GraspCollapsedHeuristic(CollapsedLabeledHeuristic):
  """CollapsedLabeledHeuristic PLUS the dense-grasp LABEL collapse: robot labels use the BASE
  grasp id (wrist-tilt variants folded), facts are OR'd over the variant assignments (a
  further relaxation -> still admissible). The cure for --dense-grasps' ~3^R coreachable-
  state explosion (cross-4/3R floating: 408,771 states/52s raw -> 21,265/6s collapsed).
  A separate class so the well-tested CollapsedLabeledHeuristic stays byte-identical."""

  def __init__(self, *args, **kwargs):
    kwargs["collapse_grasps"] = True
    super().__init__(*args, **kwargs)


class SignatureCollapsedHeuristic(CollapsedLabeledHeuristic):
  """FACT-SIGNATURE grasp collapse + LABELED LOOSE parts (--sig-collapsed-h).

  The middle ground between the FINE labeled class (raw grasp alphabet -- kills the
  plateau, explodes at R=4/--dense-grasps) and the blind collapses (small, but the bare-L
  identity collapse discards exactly the pick/press discrimination; 2026-07-17 finding):
  grasp labels are EQUIVALENCE CLASSES under the fact signature (see
  compute_grasp_signatures / set_sig_components), and loose parts KEEP their class label,
  with the loose-handover swap edge pricing class changes at the real fused cost.
  Facts OR over class members via the inherited _vgroups/_svgroups machinery, so an
  incomplete signature only coarsens the relaxation -- admissibility is preserved."""

  def __init__(self, assembly, n_robots, grasp_table, **kwargs):
    # MIXED EEs: the signature's WRENCH components (press/solohold) evaluate at each grasp's
    # KIND(S) from its mask (compute_grasp_signatures handles the {robot:kind} map) -- so
    # classes split by per-kind press-transmissibility, and the collapsed abstraction's
    # per-kind wrench facts (see CollapsedLabeledHeuristic) hang off those classes.
    classes = compute_grasp_signatures(
        assembly, grasp_table,
        gripper_kind=kwargs.get("gripper_kind"),
        gripper_path=kwargs.get("gripper_path"),
        clearance=kwargs.get("clearance", 0.02))
    if kwargs.get("verbose"):
      summary = ", ".join(f"{q}:{len(set(m.values()))}/{len(m)}"
                          for q, m in sorted(classes.items()))
      print(f"[sig-h] components=({','.join(_SIG_COMPONENTS)}); "
            f"classes/raw grasps per part: {summary}")
    kwargs["grasp_classes"] = classes
    kwargs["labeled_loose"] = True
    try:
      super().__init__(assembly, n_robots, grasp_table, **kwargs)
    except RuntimeError as exc:
      # Retaining (part, grasp-class) labels for every LOOSE held part grows combinatorially
      # with robot capacity. Mixed-tool scenes split even conventional grasps by vacuum/finger
      # usability, so fixture-rack-10/4R crosses the guard before forward search starts. Do
      # not silently weaken the heuristic: explain the opt-in scalable relaxation instead.
      if not str(exc).startswith("collapsed labeled abstraction exceeded "):
        raise
      raise RuntimeError(
          f"{exc}. Failure is in the backward signature heuristic, before forward search: "
          "loose held parts retain their grasp class, and mixed vacuum/two-finger masks "
          f"multiply those labels across {n_robots} robots. Re-run with --optimistic "
          "--no-optimistic-fallback to use dependency-ordered local fills without "
          "constructing this global database. The local solver retains all concrete "
          "gripper-kind, wrench and IK gates."
      ) from exc


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
  ref = S_key[0]
  q = _problem.placement_quat(assembly, S_key, ref, face)   # shape-frame rest (change-2-lite)
  R = _quat_to_R(q)
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



# (the interim `_scatter_pick_fact` -- pick facts at the raw scatter orientation -- was
# removed: placement_quat + neutral rests made scatter-style poses ordinary rest-table
# faces, so the canonical exists-a-face pick gates cover them; see chair history.)


def _por_parts(a):
  # parts an action's manipulation touches (assemble touches the mate target too)
  return {a[2], a[3]} if a[0] == "assemble" else {a[2]}


def _por_keep(a_in, b):
  # PARTIAL-ORDER REDUCTION (sleep-set-lite, EXPERIMENTAL --por): among INDEPENDENT actions
  # -- different robots AND disjoint parts -- every interleaving reaches the same state, so
  # only the canonically SORTED one is expanded: prune successor `b` when it is independent
  # of the incoming action `a_in` and precedes it in the fixed total (tuple) order; the
  # b-before-a_in interleaving is generated elsewhere. KNOWN UNSOUND in general here:
  # feasibility (IK/cheap geometry) is order-COUPLED through arm/part positions, so the
  # pruned interleaving may be the only feasible one -- hence flag-gated, meant for spread
  # scenes (--min-sep) with floating EEs where the independence assumption actually holds.
  if a_in[1] == b[1]:
    return True                                           # same robot: dependent
  if _por_parts(a_in) & _por_parts(b):
    return True                                           # shared part: dependent
  return not (tuple(map(str, b)) < tuple(map(str, a_in)))


_LAST_SEARCH_OUTCOME = None    # "found" | "exhausted" | "budget" | "deadline" -- set by
                               # _astar/_pea_star at exit; read by the iter-bound driver
                               # (exhausted = proof, budget/deadline = no proof).


def _outcome(kind):
  global _LAST_SEARCH_OUTCOME
  _LAST_SEARCH_OUTCOME = kind


def _astar(start, *, goal_fn, succ_fn, cost_fn, h_fn, cheap_fn=None,
           edge_check_fn=None, weight=1.0, max_nodes=200000, verbose=True,
           tag="A*", t0=None, cost_bound=None, deadline=None, order_key_fn=None,
           stats_every=None, por=False, pathmax=False):
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
    _outcome("found")
    return []
  tie = itertools.count()                                 # stable tie-break (shallower g first)
  # optional ACTION-PREFERENCE tie-break, ranked between h and g: among equal-(f,h) nodes,
  # expand the one whose generating action has the smaller key first. Pure ORDERING of the
  # plateau -- claimed/pruning semantics untouched, so the w-bound and completeness hold.
  if order_key_fn is None:
    okey = lambda a: 0
  else:
    okey = lambda a: 0 if a is None else order_key_fn(a)
  claimed = {}
  came_from = {start: (None, None)}
  h0 = h_fn(start)
  # tie order within equal f: smaller h first (= effectively deeper, since g = f - w*h is
  # determined by (f,h) at finite weight -- a separate g tie-break was measured a NO-OP),
  # then the optional action preference, then insertion order.
  pq = [(f_value(0.0, h0), h0, 0, 0.0, next(tie), start, None, None)]
  optimal = weight == 1.0
  expanded = 0
  edge_checks = 0
  kind_stats = {}                                         # action kind -> [checked, failed]
  next_stats = (time.perf_counter() + stats_every) if stats_every else None
  while pq:
    if next_stats is not None and time.perf_counter() > next_stats:
      next_stats = time.perf_counter() + stats_every
      per_kind = "  ".join(f"{k}:{c}({x}f)" for k, (c, x) in sorted(kind_stats.items()))
      print(f"[{tag} stats {time.perf_counter() - t0:.0f}s] expanded={expanded} "
            f"claimed={len(claimed)} frontier={len(pq)} checks={edge_checks}  {per_kind}",
            flush=True)
    _f, _h, _ok, g, _, s, prev, action = heapq.heappop(pq)
    if s in claimed:
      continue                                            # already claimed via a cheaper/earlier edge
    if cost_bound is not None and g + _h >= cost_bound:
      continue                                            # cannot beat the incumbent (h admissible)
    if deadline is not None and time.perf_counter() > deadline:
      if verbose:
        print(f"[{tag}] deadline reached after {expanded} expansions")
      _outcome("deadline")
      return None
    if action is not None and edge_check_fn is not None:
      edge_checks += 1
      ks = kind_stats.setdefault(action[0], [0, 0])
      ks[0] += 1
      if not edge_check_fn(prev, action, s):
        ks[1] += 1
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
        if kind_stats:
          per_kind = "  ".join(f"{k}:{c}({x}f)" for k, (c, x) in sorted(kind_stats.items()))
          print(f"[{tag}] edge checks by kind (f=failed): {per_kind}")
      _outcome("found")
      return plan

    expanded += 1
    if expanded > max_nodes:
      if verbose:
        print(f"[{tag}] gave up after {max_nodes} expansions ({len(claimed)} states "
              f"claimed) in {time.perf_counter() - t0:.2f}s")
      _outcome("budget")
      return None

    for act, ns in succ_fn(s):
      if por and action is not None and not _por_keep(action, act):
        continue                                          # POR: sorted interleaving expanded elsewhere
      if ns in claimed:
        continue
      if cheap_fn is not None and not cheap_fn(s, act, ns):
        continue                                          # eager cheap filter (prunes heap growth)
      ng = g + cost_fn(act)
      hns = h_fn(ns)
      if hns == float("inf"):
        continue                                          # provable dead end (admissible h)
      if pathmax and _h - (ng - g) > hns:
        # PATHMAX consistency repair: h may not drop faster than the edge cost (triangle
        # inequality keeps this admissible). The labeled h is INCONSISTENT (measured:
        # 15/32 plan steps h-neutral, then 2-3-point plunges); smoothing the profile
        # turns each stall's f-bump -- and the frontier sweep it forces -- into a smooth
        # decline, leaving only ~slack-many bumps.
        hns = _h - (ng - g)
      if cost_bound is not None and ng + hns >= cost_bound:
        continue                                          # sound prune vs the incumbent
      heapq.heappush(pq, (f_value(ng, hns), hns, okey(act), ng, next(tie), ns, s, act))

  if verbose:
    print(f"[{tag}] exhausted {len(claimed)} states with no plan "
          f"in {time.perf_counter() - t0:.2f}s")
  _outcome("exhausted")
  return None


def _pea_star(start, *, goal_fn, succ_fn, cost_fn, h_fn, cheap_fn=None,
              edge_check_fn=None, weight=1.0, max_nodes=200000, verbose=True,
              tag="PEA*", t0=None, cost_bound=None, deadline=None, order_key_fn=None,
              stats_every=None, por=False, pathmax=False):
  """PARTIAL-EXPANSION A* (PEA*): identical semantics and guarantees to _astar
  (claim-at-pop, lazy edge checks, weight, cost_bound, POR), but the OPEN list holds only
  each expansion's current f-BAND: successors with f above the popped node's F are NOT
  pushed -- the node is re-queued ONCE at the smallest such f and regenerates its
  successors when that band comes up. Frontier memory drops by ~the branching factor
  (compute-for-memory: succ_fn/h_fn re-run per band; both are cached/cheap here).
  Motivation (measured, cube-plates-2x2x3): every finite-weight A* OOMs at 8GB because
  the open list holds the full interleaving-lattice frontier -- the states are cheap, the
  PUSHES are the memory. A successor with f=X is pushed no later than its parent's band
  reaches X <= the global minimum reaching X, so ordering/completeness/w-bound are all
  preserved."""
  import heapq
  import itertools
  import math
  import time

  if t0 is None:
    t0 = time.perf_counter()

  def f_value(g, hval):
    return hval if math.isinf(weight) else g + weight * hval

  if goal_fn(start):
    _outcome("found")
    return []
  EPS = 1e-9
  tie = itertools.count()
  if order_key_fn is None:
    okey = lambda a: 0
  else:
    okey = lambda a: 0 if a is None else order_key_fn(a)
  claimed = {}
  came_from = {start: (None, None)}
  h0 = h_fn(start)
  # entry: (F, h, okey, g, tie, state, prev, action, band_lo)
  #   band_lo None  => EDGE entry (validate + claim at pop, then expand band (-inf, F])
  #   band_lo set   => RE-EXPANSION of a claimed node over band (band_lo, F]
  pq = [(f_value(0.0, h0), h0, 0, 0.0, next(tie), start, None, None, None)]
  expanded = 0
  edge_checks = 0
  reexpansions = 0
  next_stats = (time.perf_counter() + stats_every) if stats_every else None
  while pq:
    if next_stats is not None and time.perf_counter() > next_stats:
      next_stats = time.perf_counter() + stats_every
      print(f"[{tag} stats {time.perf_counter() - t0:.0f}s] expanded={expanded} "
            f"(+{reexpansions} re) claimed={len(claimed)} frontier={len(pq)} "
            f"checks={edge_checks}", flush=True)
    F, _h, _ok, g, _, s, prev, action, band_lo = heapq.heappop(pq)
    if deadline is not None and time.perf_counter() > deadline:
      if verbose:
        print(f"[{tag}] deadline reached after {expanded} expansions")
      _outcome("deadline")
      return None
    if band_lo is None:
      if s in claimed:
        continue
      if cost_bound is not None and g + _h >= cost_bound:
        continue
      if action is not None and edge_check_fn is not None:
        edge_checks += 1
        if not edge_check_fn(prev, action, s):
          continue
      claimed[s] = g
      if action is not None:
        came_from[s] = (prev, action)
      if goal_fn(s):
        plan = _reconstruct_plan(came_from, s)
        if verbose:
          bound = ("optimal" if weight == 1.0 else
                   "greedy, no bound" if math.isinf(weight)
                   else f"cost <= {weight:g}x optimal")
          print(f"[{tag}] found a {len(plan)}-action plan (cost {g:g}, {bound}); "
                f"expanded {expanded} nodes (+{reexpansions} re-expansions), "
                f"{len(claimed)} states claimed, {edge_checks} edge checks, peak-ish "
                f"frontier {len(pq)} in {time.perf_counter() - t0:.2f}s")
        _outcome("found")
        return plan
      lo = None                                           # first expansion: no lower cut
      expanded += 1
      if expanded > max_nodes:
        if verbose:
          print(f"[{tag}] gave up after {max_nodes} expansions ({len(claimed)} states "
                f"claimed) in {time.perf_counter() - t0:.2f}s")
        _outcome("budget")
        return None
    else:
      if s not in claimed:
        continue                                          # superseded (should not happen)
      g = claimed[s]
      lo = band_lo
      reexpansions += 1
    gen_action = came_from.get(s, (None, None))[1]
    next_F = None
    for act, ns in succ_fn(s):
      if por and gen_action is not None and not _por_keep(gen_action, act):
        continue
      if ns in claimed:
        continue
      if cheap_fn is not None and not cheap_fn(s, act, ns):
        continue
      ng = g + cost_fn(act)
      hns = h_fn(ns)
      if hns == float("inf"):
        continue
      if pathmax and _h - (ng - g) > hns:
        hns = _h - (ng - g)                               # pathmax (see _astar)
      if cost_bound is not None and ng + hns >= cost_bound:
        continue
      fns = f_value(ng, hns)
      if fns <= F + EPS and (lo is None or fns > lo + EPS):
        heapq.heappush(pq, (fns, hns, okey(act), ng, next(tie), ns, s, act, None))
      elif fns > F + EPS and (next_F is None or fns < next_F):
        next_F = fns
    if next_F is not None:
      heapq.heappush(pq, (next_F, _h, 0, g, next(tie), s, None, None, F))
  if verbose:
    print(f"[{tag}] exhausted {len(claimed)} states with no plan "
          f"in {time.perf_counter() - t0:.2f}s")
  _outcome("exhausted")
  return None


def _beam(start, *, goal_fn, succ_fn, cost_fn, h_fn, cheap_fn=None, edge_check_fn=None,
          width=8, weight=float("inf"), max_nodes=200000, verbose=True, tag="beam",
          t0=None, deadline=None):
  """LEVEL-SYNCHRONOUS BEAM SEARCH over the same callbacks as _astar -- the memory-bounded
  generalization of greedy (width=1 ~ greedy, width=inf ~ wA*). Motivation (measured,
  cube-plates-2x2x3): on long horizons every finite-weight A* OOMs on the INTERLEAVING
  LATTICE frontier while the h gradient is near-perfect -- greedy works but is brittle
  (h-neutral wander on other instances). Beam keeps <= width states per level (memory
  width x branching), scores candidates by f = h (weight=inf) or g + weight*h, and
  VALIDATES edges in score order only until `width` survivors are found (lazy: losers
  never pay edge_check_fn). NO completeness/quality bound -- pair with the incumbent-
  bounded phase for certificates. Returns the action list or None (dead beam / budget)."""
  import math
  import time
  if t0 is None:
    t0 = time.perf_counter()
  if goal_fn(start):
    return []

  def f_value(g, hval):
    return hval if math.isinf(weight) else g + weight * hval

  came_from = {start: (None, None)}
  seen = {start}
  frontier = [(0.0, start)]
  expanded = 0
  edge_checks = 0
  depth = 0
  while frontier:
    depth += 1
    cands = []
    for g, s in frontier:
      expanded += 1
      if expanded > max_nodes:
        if verbose:
          print(f"[{tag}] gave up after {max_nodes} expansions (depth {depth}) "
                f"in {time.perf_counter() - t0:.2f}s")
        return None
      for act, ns in succ_fn(s):
        if ns in seen:
          continue
        if cheap_fn is not None and not cheap_fn(s, act, ns):
          continue
        ng = g + cost_fn(act)
        hns = h_fn(ns)
        if hns == float("inf"):
          continue
        cands.append((f_value(ng, hns), hns, ng, ns, s, act))
    cands.sort(key=lambda c: (c[0], c[1]))
    kept = []
    for fv, hns, ng, ns, s, act in cands:
      if len(kept) >= width:
        break                                             # losers never pay the edge check
      if ns in seen:
        continue                                          # duplicate candidate within level
      if deadline is not None and time.perf_counter() > deadline:
        if verbose:
          print(f"[{tag}] deadline reached at depth {depth}")
        return None
      if edge_check_fn is not None:
        edge_checks += 1
        if not edge_check_fn(s, act, ns):
          continue
      seen.add(ns)
      came_from[ns] = (s, act)
      if goal_fn(ns):
        plan = _reconstruct_plan(came_from, ns)
        if verbose:
          print(f"[{tag} w={width}] found a {len(plan)}-action plan (cost {ng:g}, no "
                f"bound); expanded {expanded} nodes, {edge_checks} edge checks, depth "
                f"{depth} in {time.perf_counter() - t0:.2f}s")
        return plan
      kept.append((ng, ns))
    frontier = kept
  if verbose:
    print(f"[{tag} w={width}] beam died at depth {depth} ({expanded} expanded) "
          f"in {time.perf_counter() - t0:.2f}s")
  return None


def _dfbnb(start, *, goal_fn, succ_fn, cost_fn, h_fn, cheap_fn=None, edge_check_fn=None,
           bound, verbose=True, tag="dfbnb", t0=None, max_nodes=200000, deadline=None,
           trans_cap=2_000_000):
  """DEPTH-FIRST BRANCH-AND-BOUND under an incumbent bound -- the LINEAR-MEMORY
  certificate engine for long horizons (the bounded A* fit under 8GB on cube-plates-2x2x3
  only barely: ~5GB open list). Children are ordered best-h-first; every edge is
  validated eagerly at descent (depth-first = it is on the current path); g + h >= bound
  prunes with the UNWEIGHTED admissible h, so exhaustion PROVES no plan cheaper than the
  final bound exists (claim-once-free: dominance pruning via a transposition table keyed
  state -> best g, LRU-capped -- eviction only costs re-expansion, never correctness).
  ANYTIME: each goal hit tightens the bound and search continues. Returns
  (best_plan_or_None, proven: bool)."""
  import time
  from collections import OrderedDict
  if t0 is None:
    t0 = time.perf_counter()
  if goal_fn(start):
    return [], True
  best_plan = None
  best_cost = bound
  trans = OrderedDict()
  trans[start] = 0.0
  expanded = 0
  edge_checks = 0
  # stack entries: (state, g, child iterator); path actions tracked alongside
  def children(s, g):
    out = []
    for act, ns in succ_fn(s):
      if cheap_fn is not None and not cheap_fn(s, act, ns):
        continue
      ng = g + cost_fn(act)
      hns = h_fn(ns)
      if hns == float("inf") or ng + hns >= best_cost:
        continue
      out.append((hns, ng, act, ns))
    out.sort(key=lambda c: (c[0], c[1]))
    return iter(out)

  stack = [(start, 0.0, children(start, 0.0))]
  path_actions = []
  on_path = {start}
  proven = True
  while stack:
    s, g, it = stack[-1]
    if deadline is not None and time.perf_counter() > deadline:
      proven = False
      break
    step = next(it, None)
    if step is None:
      stack.pop()
      on_path.discard(s)
      if path_actions:
        path_actions.pop()
      continue
    hns, ng, act, ns = step
    if ng + hns >= best_cost:                             # bound may have tightened since sort
      continue
    if ns in on_path:
      continue                                            # cycle on the current path
    tg = trans.get(ns)
    if tg is not None and tg <= ng:
      trans.move_to_end(ns)
      continue                                            # dominated: seen at <= g before
    if edge_check_fn is not None:
      edge_checks += 1
      if not edge_check_fn(s, act, ns):
        continue
    trans[ns] = ng
    trans.move_to_end(ns)
    while len(trans) > trans_cap:
      trans.popitem(last=False)                           # LRU eviction: pruning only
    if goal_fn(ns):
      best_cost = ng
      best_plan = path_actions + [act]
      if verbose:
        print(f"[{tag}] incumbent improved: cost {ng:g} ({len(best_plan)} actions) at "
              f"{time.perf_counter() - t0:.1f}s ({expanded} expanded)")
      continue                                            # keep searching under tighter bound
    expanded += 1
    if expanded > max_nodes:
      proven = False
      break
    stack.append((ns, ng, children(ns, ng)))
    path_actions.append(act)
    on_path.add(ns)
  if verbose:
    print(f"[{tag}] done: {'cost %g (%d actions)' % (best_cost, len(best_plan)) if best_plan else 'no plan under bound'}"
          f"{', PROVEN best under bound' if proven and stack == [] else ', budget/deadline hit (not proven)' if not proven else ', PROVEN'}"
          f"; {expanded} expanded, {edge_checks} edge checks, "
          f"{len(trans)} transpositions in {time.perf_counter() - t0:.2f}s")
  return best_plan, proven and not stack


def _search_context(C, assembly, robots, build_table=True):
  """The shared, built-once machinery a domain search needs: the cheap collision checker
  and the precomputed grasp table, both keyed to the real gripper (auto-detected)."""
  # the cheap checker's floating EE must MATCH the real gripper: a vacuum stand-in is much
  # slimmer than the two-finger, so it would let grasps through that the real palm/fingers
  # rule out (neighbouring cubes, finger-under-the-cube-into-the-table).
  names = C.getFrameNames()
  # ...and mirror the ROBOT model: real arms get the wrist-extended cheap EE, a floating-EE
  # env gets the plain one (no wrist exists in that model). The wrist-3 capsule is arm-SPECIFIC,
  # so a real UR5 arm (detected via its ur_arm_ur5 marker) gets the ur5-calibrated cheap wrist,
  # not the ur10 one. See cheap_ee_path / UR5_CHEAP_EE_PATHS in env.py.
  floating = not any(n.endswith("_ur_wrist_3_joint") for n in names)
  is_ur5 = any(n.endswith("_ur_arm_ur5") for n in names)
  is_openarm = any("_openarm_left_base_link" in n or "_openarm_right_base_link" in n
                   for n in names)
  kinds = robot_gripper_kinds(C, robots)
  distinct = sorted(set(kinds.values()))
  if is_openarm:
    if distinct != ["two_finger"]:
      raise ValueError("OpenArm currently exports its native pinch hand only")
    oa_ee = os.path.join(os.path.dirname(__file__), OPENARM_FLOATING_EE_PATH)
    ee_path = {"two_finger": oa_ee}
    fact_path = {"two_finger": oa_ee}
  else:
    ee_path = {k: cheap_ee_path(is_ur5, floating, k == "vacuum") for k in distinct}
    fact_path = {
        k: os.path.join(os.path.dirname(__file__), UR10_PATHS[(True, k == "vacuum")])
        for k in distinct}
  if len(distinct) == 1:                                   # UNIFORM: byte-identical to before
    k = distinct[0]
    checker = CollisionChecker(assembly, robots, gripper_path=ee_path[k])
    # plain-EE variant for the labeled-h pick facts (clutter-free scenes; see the h_lab
    # construction in solve_assembly_problem for why the wrist is excluded there).
    checker.fact_gripper_path = fact_path[k]
    return checker, (precompute_grasps(assembly, gripper_path=ee_path[k])
                     if build_table else None)
  # MIXED END EFFECTORS: per-robot cheap-EE slots; the grasp table is the UNION over kinds
  # with a per-grasp kind MASK (a grasp enters kind k's set iff k's EE stands there
  # collision-free vs the subassembly). Successor generators gate (robot, grasp) pairs on
  # the mask; heuristic facts OR over a grasp's mask kinds (admissible).
  checker = CollisionChecker(assembly, robots, gripper_path=ee_path[distinct[0]],
                             robot_gripper_paths={r: ee_path[kinds[r]] for r in robots})
  checker.fact_gripper_path = fact_path                    # dict per kind (mixed)
  if not build_table:
    return checker, None
  per_kind = {k: precompute_grasps(assembly, gripper_path=ee_path[k]) for k in distinct}
  table = GraspTable()
  table.kinds_of = {}
  table.robot_kinds = dict(kinds)
  rolls = _problem._ROLLS_PER_FACE
  dense = _problem._DENSE_GRASPS
  for key in set().union(*(t.keys() for t in per_kind.values())):
    merged = {}
    for k in distinct:
      for p, g, pose in per_kind[k].get(key, ()):
        if (k == "vacuum" and rolls > 1 and not dense and g % rolls != 0
            and _problem._SAMPLED_GRASPS is None):
          continue  # the pen is roll-symmetric: rolls of one face are IK-identical
                    # duplicates for the vacuum -- and vacuum binds are the expensive
                    # ones (no analytic reject) -> keep roll 0 only in ITS mask. The
                    # two-finger keeps every roll (its pads genuinely differ per roll).
                    # (--grasp-sampler ids are NOT (face, roll)-structured: the modulus
                    # would silently drop half the sampled vacuum candidates -- skip.)
        merged.setdefault((p, g), (p, g, pose))
        table.kinds_of.setdefault((key, p, g), set())
        table.kinds_of[(key, p, g)].add(k)
    table[key] = sorted(merged.values(), key=lambda t: (t[0], t[1]))
  table.kinds_of = {kk: frozenset(v) for kk, v in table.kinds_of.items()}
  return checker, table


def gripper_kind(C):
  # which gripper the scene uses, as the key into wrench.PARAMS['grasp']. MIXED scenes have
  # no single answer -- use robot_gripper_kinds and thread the map (accepted wherever a
  # gripper_kind flows; scalar stays the uniform fast path).
  return "vacuum" if any(n.endswith("_ur_vacuum") for n in C.getFrameNames()) \
      else "two_finger"


def robot_gripper_kinds(C, robots):
  # {robot: "vacuum" | "two_finger"} read off the scene (each robot's own gripper frames).
  names = set(C.getFrameNames())
  return {r: ("vacuum" if f"{r}_ur_vacuum" in names else "two_finger") for r in robots}


def wrench_kind_arg(C, robots):
  # the `gripper_kind` value to thread into the wrench gates: the plain scalar for uniform
  # scenes (byte-identical caches), the per-robot map only when genuinely mixed.
  kinds = robot_gripper_kinds(C, robots)
  distinct = set(kinds.values())
  return kinds if len(distinct) > 1 else next(iter(distinct))


def solve_assembly_problem(C, assembly, robots=None, max_nodes=200000, verbose=True,
                           use_heuristic=True, canonical=False, weight=1.0, robot_ik=None,
                           wrench=False, wrench_aware_h=True, handovers=True,
                           allow_place=True, cost_bound=None, deadline=None,
                           anonymous=False, labeled_h=False, collapsed_h=False,
                           grasp_collapsed_h=False, bracing=False, part_order_bias=False,
                           stats_every=None, por=False, geo_facts=False,
                           sig_collapsed_h=False, beam_width=None, dfbnb=False,
                           pea=False, iter_bound=False, pathmax=False,
                           h_dist_cap=None):
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
  gk = wrench_kind_arg(C, robots) if wrench else None      # scalar; {robot: kind} when MIXED
  if _problem._USEFUL_FLAG:
    _problem.set_useful_grasps(compute_useful_grasps(assembly, grasp_table, gk))
    if verbose:
      nU = {p: len(v) for p, v in (_problem._USEFUL_GRASPS or {}).items()}
      print(f"[useful-grasps] mate-viable per part: {nU}")

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
  # Same finite-weight gating as h_abs -- RE-MEASURED 2026-07-16 (don't retry): lifting the
  # gate and running greedy(weight=inf)+grasp_collapsed_h on dense cross-4/5 walked
  # h-non-increasing wander to 1098/2238-action "plans", slower than w=5. Without g in f,
  # greedy strolls the astronomical h-NEUTRAL state shell (robots x grasps); the g term is
  # what forbids that, not a tie-break.
  h_lab = None
  if (labeled_h or collapsed_h or grasp_collapsed_h or sig_collapsed_h) and use_heuristic:
    if math.isinf(weight):
      warnings.warn("labeled/collapsed heuristics are gated to finite weights (measured: "
                    "greedy+labeled wanders h-neutral shells into 1000+-action plans). "
                    "Running WITHOUT the abstraction. Use e.g. weight=5.", stacklevel=2)
    else:
      if gk is not None:
        reset_wrench_cache()                              # facts below use the wrench LP
      # wrench facts follow the search's wrench model (gk) by default; wrench_aware_h=False
      # builds the labeled abstraction wrench-BLIND for comparison (same toggle as h_abs).
      # pick facts use the PLAIN floating EE even when the checker runs the wrist variant:
      # fact scenes hold ONE part at a free table spot (no clutter), so the wrist capsule
      # adds nothing there except unsound wrist-vs-table rejections (see _wrist_table_pair).
      cls = (SignatureCollapsedHeuristic if sig_collapsed_h
             else GraspCollapsedHeuristic if grasp_collapsed_h
             else CollapsedLabeledHeuristic if collapsed_h
             else LabeledDisassemblyHeuristic)
      if _problem._DENSE_GRASPS and not (grasp_collapsed_h or sig_collapsed_h):
        warnings.warn("labeled_h/collapsed_h with dense grasps label RAW grasp ids, so the "
                      "tilt variants ~3x the label space per robot (build explodes ~3^R, "
                      "measured 24x). Use grasp_collapsed_h (--grasp-collapsed-h): it folds "
                      "variants onto their base grasp.")
      h_lab = cls(assembly, len(robots), grasp_table,
                  gripper_path=getattr(checker, "fact_gripper_path", checker.gripper_path),
                  verbose=verbose,
                  gripper_kind=gk if wrench_aware_h else None, bracing=bracing,
                  geo_checker=checker if geo_facts else None, dist_cap=h_dist_cap)
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
    # optional plateau tie-break: among equal-(f,h) frontier nodes, TOUCH EARLIER PARTS FIRST
    # (lower part index = earlier in the mate order). Pure ordering, so the w-bound holds;
    # the point is coherence on the huge equal-h plateaus (which loose part / grasp / robot
    # to touch next is invisible to the abstraction, so without a preference the search
    # floods the plateau breadth-first in insertion order).
    okey = (lambda a: _part_index(a[2])) if part_order_bias else None
    goal_fn = lambda s: s.is_goal(assembly)
    if beam_width is not None:
      return _beam(start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost,
                   h_fn=h_fn, cheap_fn=None, edge_check_fn=edge_check_fn,
                   width=beam_width, weight=weight, max_nodes=max_nodes,
                   verbose=verbose, t0=t0, deadline=deadline)
    if dfbnb:
      if cost_bound is None:
        warnings.warn("dfbnb requires an incumbent cost_bound (use the bounded driver); "
                      "falling back to the A* engine.")
      else:
        plan, _proven = _dfbnb(start, goal_fn=goal_fn, succ_fn=succ_fn,
                               cost_fn=action_cost, h_fn=h_fn, cheap_fn=None,
                               edge_check_fn=edge_check_fn, bound=cost_bound,
                               verbose=verbose, t0=t0, max_nodes=max_nodes,
                               deadline=deadline)
        return plan
    engine = _pea_star if pea else _astar
    etag = ("PEA*" if pea else tag)
    if iter_bound:
      # ITERATIVE COST-BOUND DEEPENING: ascending passes with cost_bound = h(s0)+1, +2, ...
      # Each pass's memory is exactly its bound's SUBLEVEL SET (the minimum any algorithm
      # needs to prove that bound); exhausting bound B proves no plan of cost <= B-1, so
      # the FIRST plan found is OPTIMAL -- regardless of the inner weight, which only
      # orders exploration inside the sublevel set (integer action costs; step 1).
      import math as _math
      B = h_fn(start) + 1.0
      if _math.isinf(B):
        return None
      while True:
        if verbose:
          print(f"[iter-bound] pass at cost_bound={B:g}", flush=True)
        plan = engine(start,
                      goal_fn=goal_fn,
                      succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn, cheap_fn=None,
                      edge_check_fn=edge_check_fn,
                      weight=weight, max_nodes=max_nodes, verbose=verbose, tag=etag,
                      t0=t0, cost_bound=B, deadline=deadline, order_key_fn=okey,
                      stats_every=stats_every, por=por, pathmax=pathmax)
        if plan is not None:
          if verbose:
            print(f"[iter-bound] cost {sum(action_cost(a) for a in plan):g} plan is "
                  f"PROVEN OPTIMAL (every lower bound exhausted).")
          return plan
        if _LAST_SEARCH_OUTCOME != "exhausted":
          if verbose:
            print(f"[iter-bound] pass at {B:g} ended on '{_LAST_SEARCH_OUTCOME}' -> "
                  f"stopping WITHOUT a proof.")
          return None
        B += 1.0
    return engine(start,
                  goal_fn=goal_fn,
                  succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn, cheap_fn=None,
                  edge_check_fn=edge_check_fn,
                  weight=weight, max_nodes=max_nodes, verbose=verbose, tag=etag, t0=t0,
                  cost_bound=cost_bound, deadline=deadline, order_key_fn=okey,
                  stats_every=stats_every, por=por, pathmax=pathmax)
  finally:
    set_anonymous_robots(False)


class _IncrementalLazySearch:
  """Persistent weighted-A* for the lazy-validation loop: LPA*-style INCREMENTAL repair
  specialised to our only edge change -- single DELETIONS (blacklisting) of edges that lie
  ON the current search tree. Instead of re-searching from scratch per blacklisted edge
  (the measured cost of naive LazySP: ~0.3s x rounds), the heap, claims and parent tree
  persist across rounds; deleting edge (s,a)->ns un-claims exactly the search-tree SUBTREE
  rooted at ns and re-injects the recorded ALTERNATE entries into those nodes, then the
  main loop simply continues from the retained frontier.

  Exactness bookkeeping:
    * claim-at-pop, one claim per state (same semantics/w-bound as _astar);
    * an entry popped for an already-claimed state is NOT discarded but stored in
      `alts[state]` -- the alternate routes a repair needs when that state is un-claimed;
    * a popped entry is STALE-CHECKED against its parent: parent must be claimed, the edge
      not blacklisted, and the entry's g must equal claimed[parent] + cost (a repair may
      have re-claimed the parent at a different g -> re-push with corrected priority);
    * successor lists are MEMOIZED per state (they depend only on the state), so
      re-expanding a repaired region skips the wrench/support gates of `successors`."""

  def __init__(self, start, goal_fn, succ_fn, cost_fn, h_fn, weight, blacklist,
               cheap_gate=None):
    import itertools
    self.goal_fn, self.cost_fn, self.h_fn = goal_fn, cost_fn, h_fn
    self.weight = weight
    self.blacklist = blacklist                            # shared, grows externally
    self.cheap_gate = cheap_gate                          # EAGER pop-time gate (cheap checks):
    # laziness follows the COST HIERARCHY -- ~ms cheap-collision checks prune the optimistic
    # tree exactly like the eager search (deferring them was measured to blow the search up
    # 200k+ expansions on dense floating scenes), while ~100ms IK binds stay plan-lazy.
    self._succ_raw = succ_fn                              # UNfiltered (blacklist at use time)
    self._succ_cache = {}
    # Successor generation allocates a fresh State for every edge, although many edges reach
    # value-equal states (different robot/grasp histories collapse to the same discrete state).
    # The cache otherwise retains all of those full pose/support dictionaries. Intern states
    # and actions by value so O(edges) references point at O(unique states/actions) payloads.
    self._state_intern = {start: start}
    self._action_intern = {}
    self.tie = itertools.count()
    self.pq = []
    self._pending = {}                                    # (state,parent,action) -> best entry
    self.claimed = {}
    self.came_from = {start: (None, None)}
    # A node can be unclaimed/reclaimed hundreds of times while lazy IK deletes candidate
    # edges.  Lists made both structures grow once PER REPAIR, even when the same tree child
    # or same incoming (parent, action) route was being re-recorded.  Cross-7/OpenArm reached
    # >1 GiB in a finite local graph this way.  Preserve every DISTINCT recovery route only.
    self.children = {}                                    # parent -> set(child state)
    self.alts = {}                                        # state -> {(parent, action): entry}
    self.expanded = 0
    self.budget_hit = False
    h0 = h_fn(start)
    self._queue((self._f(0.0, h0), h0, 0.0, next(self.tie), start, None, None))

  def _queue(self, entry):
    """Queue at most one best pending entry per distinct incoming graph edge.

    Repairs regenerate edges whose old copies are already waiting in the heap. The old code
    appended every copy, so the heap grew to 210k entries for a ~1k-state OpenArm fill. A
    changed parent cost may replace an entry; the superseded heap tuple is discarded lazily.
    """
    import heapq
    key = (entry[4], entry[5], entry[6])
    old = self._pending.get(key)
    if old is not None and old[:3] <= entry[:3]:
      return False                                        # identical or dominated regeneration
    self._pending[key] = entry
    heapq.heappush(self.pq, entry)
    return True

  def _remember_alt(self, state, entry):
    """Remember one best heap entry per distinct incoming edge to ``state``.

    Re-expanding a repaired parent regenerates the identical edge. Keeping all copies has no
    search meaning: if the parent cost changes, search() recomputes the entry when it is popped;
    if the edge is deleted, every copy is invalid. Distinct parent/action edges remain available
    for later repairs, preserving the incremental search's coverage.
    """
    edge = (entry[5], entry[6])
    bucket = self.alts.setdefault(state, {})
    old = bucket.get(edge)
    if old is None or entry[:4] < old[:4]:
      bucket[edge] = entry

  def footprint(self):
    """Compact search-storage counters for profiling lazy-fill scaling."""
    return {
        "frontier": len(self._pending),
        "heap_entries": len(self.pq),
        "claimed": len(self.claimed),
        "alternates": sum(len(v) for v in self.alts.values()),
        "unique_states": len(self._state_intern),
        "unique_actions": len(self._action_intern),
        "successor_states": len(self._succ_cache),
        "successor_edges": sum(len(v) for v in self._succ_cache.values()),
        "tree_edges": sum(len(v) for v in self.children.values()),
    }

  def _f(self, g, hval):
    import math
    return hval if math.isinf(self.weight) else g + self.weight * hval

  def _succ(self, s):
    hit = self._succ_cache.get(s)
    if hit is None:
      hit = []
      for action, state in self._succ_raw(s):
        action = self._action_intern.setdefault(action, action)
        state = self._state_intern.setdefault(state, state)
        hit.append((action, state))
      self._succ_cache[s] = hit
    return hit

  def search(self, max_nodes, deadline=None):
    # run until a goal state is claimed (returns the Plan) or the frontier empties (None).
    import heapq
    import time
    import os as _os
    _dbg = _os.environ.get("MILESTONE_DEBUG")
    _pops = 0
    while self.pq:
      entry = heapq.heappop(self.pq)
      _f, _h, g, _, s, prev, action = entry
      pkey = (s, prev, action)
      pending = self._pending.get(pkey)
      if pending is None or pending[3] != entry[3]:
        continue                                          # superseded pending entry
      del self._pending[pkey]
      if s in self.claimed:
        self._remember_alt(s, entry)                      # keep each incoming route once
        continue
      if _dbg and _pops < 12:
        try:
          gate = self.cheap_gate(prev, action, s) if self.cheap_gate is not None else None
        except Exception as _e:
          gate = f"ERR {_e}"
        print(f"[fill-debug] pop{_pops} act={action} gate={gate} h_env={_h:.2f}", flush=True)
      _pops += 1
      if prev is not None:
        if prev not in self.claimed or (prev, action) in self.blacklist:
          continue                                        # stale: parent gone / edge deleted
        g_true = self.claimed[prev] + self.cost_fn(action)
        if g_true != g:                                   # parent re-claimed at another g
          hs = self.h_fn(s)
          self._queue((self._f(g_true, hs), hs, g_true, next(self.tie),
                       s, prev, action))
          continue
        if self.cheap_gate is not None and not self.cheap_gate(prev, action, s):
          self.blacklist.add((prev, action))              # cheap-infeasible: dead forever
          continue
      self.claimed[s] = g
      if prev is not None:
        self.came_from[s] = (prev, action)
        self.children.setdefault(prev, set()).add(s)
      if self.goal_fn(s):
        self._goal = s
        return _reconstruct_plan(self.came_from, s)
      self.expanded += 1
      if self.expanded > max_nodes:
        self.budget_hit = True
        return None
      if deadline is not None and time.perf_counter() > deadline:
        self.budget_hit = True
        return None
      for a, ns in self._succ(s):
        if (s, a) in self.blacklist:
          continue
        ng = g + self.cost_fn(a)
        hns = self.h_fn(ns)
        if hns == float("inf"):
          continue
        entry = (self._f(ng, hns), hns, ng, next(self.tie), ns, s, a)
        if ns in self.claimed:
          self._remember_alt(ns, entry)                   # alternate for a future repair
        else:
          self._queue(entry)
    return None

  def delete_edge(self, s, a, ns):
    # (s, a) was just blacklisted (caller added it). If it is the tree edge of ns, un-claim
    # the SUBTREE rooted at ns (their g relied on the deleted edge) and re-inject each
    # un-claimed node's alternates; everything else is untouched. Queued entries THROUGH the
    # subtree are invalidated lazily by the stale checks at pop.
    import heapq
    if self.came_from.get(ns, (None, None)) != (s, a):
      return                                              # not a tree edge: nothing to repair
    self._pending.pop((ns, s, a), None)                   # queued copy is now invalid
    stack = [ns]
    while stack:
      x = stack.pop()
      if x not in self.claimed:
        continue
      del self.claimed[x]
      self.came_from.pop(x, None)
      stack.extend(self.children.pop(x, ()))
      for alt in self.alts.pop(x, {}).values():
        self._queue(alt)


def solve_assembly_problem_lazy(C, assembly, robots=None, max_nodes=200000, verbose=True,
                                use_heuristic=True, weight=1.0, robot_ik=None,
                                wrench=False, wrench_aware_h=True, handovers=True,
                                allow_place=True, cost_bound=None, deadline=None,
                                labeled_h=False, collapsed_h=False, grasp_collapsed_h=False,
                                bracing=False, max_rounds=2000, geo_facts=False,
                                sig_collapsed_h=False):
  """LAZY-VALIDATION exact search (LazySP-style; spec 6.5). Same discrete model, heuristics
  and guarantees as `solve_assembly_problem`, but the EXPENSIVE per-edge feasibility (cheap
  geometry + IK) moves from every popped edge to CANDIDATE PLANS ONLY:

    repeat:
      1. run the OPTIMISTIC search (wrench/support gates still live in `successors`; no
         cheap-collision, no IK) minus the blacklist -> a candidate plan;
      2. validate the plan's edges IN ORDER (cheap_feasible, then robot_ik.bind);
      3. all pass -> done (the plan is exactly as validated as the eager search's);
         first failure -> BLACKLIST that (state, action) edge and re-search.

  Each failed round blacklists one genuinely-infeasible edge (never a sound one), so
  completeness matches the eager search; termination: the edge set is finite. Re-searches
  are cheap: the heuristic is built ONCE, `validated` edges skip re-checking, and the IK /
  cheap / wrench caches stay warm. Pays off when the eager search burns most of its time
  validating popped edges that never make it into a plan (measured: ~50% of runtime, 64%
  of checked edges failing); loses when candidate plans are wildly infeasible (many rounds).
  Returns a Plan with .states (replay-ready), or None."""
  import math
  import time

  t0 = time.perf_counter()
  if robots is None:
    robots = get_robots(C)
  gk = wrench_kind_arg(C, robots) if wrench else None
  h_abs = AbstractionHeuristic(assembly, len(robots),
                               gripper_kind=gk if wrench_aware_h else None) \
      if use_heuristic and not math.isinf(weight) else None
  checker, grasp_table = _search_context(C, assembly, robots)
  if gk is not None:
    reset_wrench_cache()
  h_lab = None
  if (labeled_h or collapsed_h or grasp_collapsed_h or sig_collapsed_h) and use_heuristic \
      and not math.isinf(weight):
    cls = (SignatureCollapsedHeuristic if sig_collapsed_h
           else GraspCollapsedHeuristic if grasp_collapsed_h
           else CollapsedLabeledHeuristic if collapsed_h
           else LabeledDisassemblyHeuristic)
    h_lab = cls(assembly, len(robots), grasp_table,
                gripper_path=getattr(checker, "fact_gripper_path", checker.gripper_path),
                verbose=verbose,
                gripper_kind=gk if wrench_aware_h else None, bracing=bracing,
                geo_checker=checker if geo_facts else None)
    h_abs = None

  def h_fn(state):
    if not use_heuristic:
      return 0.0
    hv = heuristic(state, assembly)
    if h_lab is not None:
      return max(hv, h_lab.value(state))
    return max(hv, h_abs.value(state)) if h_abs is not None else hv

  blacklist = set()                                       # (state, action): validation-failed
  validated = set()                                       # (state, action): validation-passed

  def succ_raw(s):
    return successors(s, assembly, robots, grasp_table=grasp_table,
                      gripper_kind=gk, handovers=handovers,
                      allow_place=allow_place, bracing=bracing)

  start = get_initial_state(C, assembly)
  if h_lab is not None and h_lab.value(start) == float("inf"):
    if verbose or __import__("os").environ.get("MILESTONE_DEBUG"):
      import collections
      detail = getattr(h_lab, "last_infeasible_reason", None)
      print("[lazy] initial state provably cannot reach the goal (abstraction facts) -> "
            f"instance INFEASIBLE as configured. reason={detail}")
    return None

  engine = _IncrementalLazySearch(
      start, goal_fn=lambda s: s.is_goal(assembly),
      succ_fn=succ_raw, cost_fn=action_cost, h_fn=h_fn,
      weight=weight, blacklist=blacklist,
      cheap_gate=lambda prev, a, ns: cheap_feasible(a, prev, ns, assembly, checker))
  n_checks = 0
  for rounds in range(1, max_rounds + 1):
    plan = engine.search(max_nodes, deadline=deadline)    # INCREMENTAL: continues the frontier
    if plan is None:
      if verbose:
        why = "node/time budget hit" if engine.budget_hit else "optimistic graph exhausted"
        print(f"[lazy] {why} after {rounds} rounds, {engine.expanded} expansions, "
              f"{n_checks} IK validations, {time.perf_counter() - t0:.2f}s -- no plan")
      return None
    # only IK stays plan-lazy (cheap checks run eagerly inside the engine). FIRST-FAILURE
    # selector (measured better than validate-the-whole-candidate: suffix edges after a
    # failure mostly belong to futures that never recur -- 43.2s vs 28.6s on cross-7).
    n_bad = 0
    for s, a, ns in zip(plan.states, plan, plan.states[1:]):
      key = (s, a)
      if key in validated or robot_ik is None or a[0] not in _IK_ACTIONS:
        continue
      n_checks += 1
      ok, _witness = robot_ik.bind(a, s, ns, assembly)
      if not ok:
        blacklist.add(key)
        engine.delete_edge(s, a, ns)                      # LOCAL repair: un-claim ns's subtree
        n_bad += 1
        break
      validated.add(key)
    if n_bad == 0:
      if verbose:
        print(f"[lazy] found a {len(plan)}-action plan in {rounds} rounds, "
              f"{engine.expanded} expansions, {n_checks} edge validations "
              f"({len(blacklist)} blacklisted) in {time.perf_counter() - t0:.2f}s")
      return plan
    if verbose and rounds % 25 == 0:
      print(f"[lazy] round {rounds}: {len(blacklist)} edges blacklisted, "
            f"{n_checks} validations, {time.perf_counter() - t0:.1f}s")
    if deadline is not None and time.perf_counter() > deadline:
      if verbose:
        print(f"[lazy] deadline reached after {rounds} rounds")
      return None
  if verbose:
    print(f"[lazy] round budget ({max_rounds}) exhausted")
  return None


def solve_assembly_problem_bounded(C, assembly, robots=None, verbose=True, robot_ik=None,
                                   weight=1.0, max_nodes=200000, wrench=False,
                                   handovers=True, allow_place=True, bracing=False,
                                   labeled_h=False, collapsed_h=False,
                                   grasp_collapsed_h=True, stats_every=None,
                                   geo_facts=False, sig_collapsed_h=False,
                                   beam_width=None, dfbnb=False, pea=False,
                                   pathmax=False, corridor_h=False,
                                   wrench_aware_h=True):
  """INCUMBENT-BOUNDED exact search -- the principled wrapper around the fast decomposed
  solvers. Phase 1: the MILESTONE planner produces an incumbent plan fast (its greedy
  commitments carry no bound). Phase 2: the exact search runs with cost_bound = U (the
  incumbent's cost): `_astar` prunes g + h >= U with the UNWEIGHTED admissible h, which is
  sound -- every prefix of an optimal plan has g + h <= optimal < U and survives -- and it
  is precisely the high-g plateau shells that die. Returns the better plan.

  GUARANTEE: result cost <= min(U, weight x optimal). With weight=1 the phase-2 search is
  ordinary A*: finding nothing below U proves U cannot be beaten by more than the claim-once
  approximation, and any found plan is optimal. The incumbent is returned as-is if phase 2
  finds nothing better (bound-exhaustion / budget)."""
  import time
  t0 = time.perf_counter()
  inc = solve_assembly_problem_milestones(
      C, assembly, robots=robots, verbose=verbose, robot_ik=robot_ik,
      weight=max(weight, 5.0), wrench=wrench, wrench_aware_h=wrench_aware_h,
      handovers=handovers,
      allow_place=allow_place, bracing=bracing, geo_facts=geo_facts,
      sig_collapsed_h=sig_collapsed_h)
  if inc is None:
    # GREEDY fallback incumbent: milestones cannot handle every structure (e.g. module
    # joins, see cube-plates), but an unbounded phase-2 is a MEMORY BOMB on long horizons
    # (measured cube-plates-2x2x3: every finite weight OOMs at 8GB unbounded, while
    # greedy solves in seconds and its cost as bound lets w=5 COMPLETE in ~3min).
    if verbose:
      print("[bounded] no milestone incumbent -> greedy (weight=inf) fallback incumbent")
    inc = solve_assembly_problem(
        C, assembly, robots=robots, verbose=verbose, robot_ik=robot_ik,
        weight=float("inf"), max_nodes=max_nodes, wrench=wrench, handovers=handovers,
        allow_place=allow_place, bracing=bracing, wrench_aware_h=wrench_aware_h)
  if inc is None and beam_width is not None:
    # BEAM last-resort incumbent (measured 2x2x3: greedy beats beam -- its global
    # best-first frontier can back up where a fixed-width beam must commit; beam only
    # helps when greedy itself dies, so it runs LAST, at weight=inf ranking).
    if verbose:
      print(f"[bounded] no greedy incumbent -> beam (width {beam_width}) incumbent")
    inc = solve_assembly_problem(
        C, assembly, robots=robots, verbose=verbose, robot_ik=robot_ik,
        weight=float("inf"), max_nodes=max_nodes, wrench=wrench, handovers=handovers,
        allow_place=allow_place, bracing=bracing, labeled_h=labeled_h,
        collapsed_h=collapsed_h, grasp_collapsed_h=grasp_collapsed_h,
        geo_facts=geo_facts, sig_collapsed_h=sig_collapsed_h, beam_width=beam_width,
        wrench_aware_h=wrench_aware_h)
  U = sum(action_cost(a) for a in inc) if inc else None
  if verbose:
    print(f"[bounded] incumbent: {'cost %g (%d actions)' % (U, len(inc)) if inc else 'NONE'} "
          f"in {time.perf_counter() - t0:.1f}s -> exact search with cost_bound="
      f"{U if U is not None else 'inf'}")
  best = solve_assembly_problem(
      C, assembly, robots=robots, verbose=verbose, robot_ik=robot_ik, weight=weight,
      max_nodes=max_nodes, wrench=wrench, handovers=handovers, allow_place=allow_place,
      bracing=bracing, labeled_h=labeled_h, collapsed_h=collapsed_h,
      grasp_collapsed_h=grasp_collapsed_h, cost_bound=U, stats_every=stats_every,
      geo_facts=geo_facts, sig_collapsed_h=sig_collapsed_h, dfbnb=dfbnb, pea=pea,
      pathmax=pathmax, h_dist_cap=(U if corridor_h else None),
      wrench_aware_h=wrench_aware_h)
  if best is not None and (U is None or sum(action_cost(a) for a in best) < U):
    if verbose:
      print(f"[bounded] exact search IMPROVED the incumbent "
            f"({U} -> {sum(action_cost(a) for a in best):g}) "
            f"in {time.perf_counter() - t0:.1f}s total")
    return best
  if verbose and inc is not None:
    print(f"[bounded] incumbent stands (exact found nothing cheaper under the bound) "
          f"in {time.perf_counter() - t0:.1f}s total")
  return inc


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


def _make_fill_local_pdb(start, assembly, robots, checker, grasp_table, p,
                         grasp_classes, gripper_kind=None, bracing=False,
                         component_root=None, geo_facts=False, verbose=False,
                         lazy_build=True):
  """Build the backward pattern database for one settled fill goal.

  Every grasp label uses the same shared fact-signature projection. This uniform abstraction
  avoids a role-specific exception for the interface/end grasp and prevents raw holder ids on a
  detached moving module from creating a Cartesian product. It is an admissible relaxation: the
  concrete forward search still considers and validates every raw grasp on every part. The same
  projection applies to a singleton, detached module, chain, or dependency graph.
  """
  root = assembly.base if component_root is None else component_root
  comps = tuple(start.subassemblies())
  before = next((frozenset(g) for g in comps if root in g), None)
  mover = next((frozenset(g) for g in comps if p in g), None)
  if before is None or mover is None:
    raise ValueError(f"fill-local PDB cannot find target root {root!r} or mover {p!r}")
  if before == mover:
    raise ValueError(f"fill-local PDB requested for already-joined mover {p!r}")
  after = before | mover
  try:
    return CollapsedLabeledHeuristic(
        assembly, len(robots), grasp_table,
        gripper_path=getattr(checker, "fact_gripper_path", checker.gripper_path),
        verbose=verbose, max_states=200_000, gripper_kind=gripper_kind,
        bracing=bracing, geo_checker=checker if geo_facts else None,
        grasp_classes=grasp_classes, labeled_loose=True,
        ideal_chain=(before, after), component_root=root, fill_goal=True,
        lazy_build=lazy_build)
  except RuntimeError as exc:
    if "exceeded 200000 states" not in str(exc):
      raise
    raise RuntimeError(
        f"fill-local PDB for interface {p!r}, target {sorted(before)}, mover "
        f"{sorted(mover)} exceeded its 200000-state guard; rerun with "
        "--no-fill-local-h (or use --milestone-oracle global/chain). No heuristic "
        "fallback was applied.") from exc


def _add_one_part(start, p, ideal, assembly, robots, checker, grasp_table, robot_ik,
                  weight, max_nodes, verbose, gripper_kind=None, handovers=True,
                  allow_place=True, bracing=False, grasp_cap=None, h_extra=None,
                  part_grasp_allow=None, brace_allow=None, deadline=None,
                  lazy_fill=False, cost_bound=None, forbid_assemble=None, settle_ref=None,
                  h_extra_shift=None, fill_h=None, exact_fill=False,
                  skeleton_only=False, touch_all=None, excluded_end_states=None):
  import os as _os
  verbose = verbose or bool(_os.environ.get("MILESTONE_DEBUG"))
  # a small COMPLETE search that adds part `p` to the assembly `ideal` and leaves it SETTLED:
  # p mated, no loose part left in a gripper, and the assembly ending held by at most one
  # carrier OR resting on the table -- WHICHEVER the search finds. It forces neither a
  # set-down nor a hold; holding-through and setting-down are just options (allow_place=False
  # drops the set-down option, so the assembly naturally stays held). It may only touch `p`
  # and the assembly (parts in `ideal`). Returns (subplan, end_state), gripper_kind => wrench.
  #
  # `settle_ref`: which component "the assembly" above refers to -- default the BASE
  # component; a MODULE sub-fill passes its module root, so a module built IN THE AIR (held
  # by one carrier) counts as settled. Without this, the held module itself counted as
  # "loose held" and the goal FORCED a table set-down -- fine for the floating-EE checks,
  # IK-infeasible for real arms on the chair's back+posts module: every module fill died.
  relevant = set(ideal) | {p}
  excluded_end_states = frozenset(excluded_end_states or ())

  # A1 INSTRUMENTATION: per-fill bind deltas -- which fill burns the bind wall time, on what.
  _fp0 = None
  if robot_ik is not None:
    _fp0 = (dict(robot_ik.prof),
            {k: list(v) for k, v in robot_ik.prof["by_action"].items()})

  def _fill_prof(tag):
    if not verbose:
      return
    pdb = getattr(fill_h, "__self__", None)
    if pdb is not None and hasattr(pdb, "reverse_stats"):
      ps = pdb.reverse_stats()
      print(f"[fill-pdb {tag}] settled={ps['settled']} discovered={ps['discovered']} "
            f"reverse={ps['reverse_time']:.2f}s pick={ps['pick_facts']} "
            f"wrench={ps['wrench_facts']} exhausted={ps['exhausted']}")
    if robot_ik is None or _fp0 is None:
      return
    pr, (f0, fba) = robot_ik.prof, _fp0
    db = pr["bind_solves"] - f0["bind_solves"]
    if not db:
      return
    df = pr["bind_fail"] - f0["bind_fail"]
    dt = pr["bind_time"] - f0["bind_time"]
    dts = sorted(((k, v[0] - fba.get(k, [0, 0, 0.0])[0],
                   v[1] - fba.get(k, [0, 0, 0.0])[1],
                   v[2] - fba.get(k, [0, 0, 0.0])[2]) for k, v in pr["by_action"].items()),
                 key=lambda x: -x[3])
    tops = "; ".join(f"{k[0]}/{k[1]}/g{k[2]} x{n} ({nf} fail, {t:.1f}s)"
                     for k, n, nf, t in dts[:4] if t > 0.01)
    print(f"[fill-prof {tag}] binds={db} failed={df} bind_time={dt:.1f}s"
          + (f": {tops}" if tops else ""))
  settle = settle_ref if settle_ref is not None else assembly.base
  search_weight = 1.0 if exact_fill else weight

  def _comps(s):
    # (main, settle) components; identical when settle_ref is None/base (byte-identical
    # goal for all non-module fills).
    main = next((g for g in s.subassemblies() if assembly.base in g), frozenset())
    mod = main if settle in main else next(
        (g for g in s.subassemblies() if settle in g), frozenset())
    return main, mod

  def _loose_and_holders(s, exclude=None):
    main, mod = _comps(s)
    loose = sum(1 for q in s.bodies() if q not in main and q not in mod and q != exclude
                for c in s.supports.get(q, ()) if c[0] == "grasp")
    holders = sum(1 for q in mod for c in s.supports.get(q, ()) if c[0] == "grasp")
    return loose, holders

  def goal_fn(s):
    # p joined, nothing loose left in hand, and the settling component down to <=1 holder
    # (a clean rest point for the next step -- prevents inserter grasps / braces piling up).
    loose, holders = _loose_and_holders(s)
    return (_part_mated(s, p) and loose == 0 and holders <= 1
            and s not in excluded_end_states)

  def h_fn(s):
    # LOCAL lower bound with a MONOTONE gradient along the intended fill (found via the
    # chair's module fill): p's own pick is priced by `left`, NOT the loose count -- the old
    # `left=2 + loose(incl p)` made picking p RAISE h (2->3) while shuffling the target
    # between table faces kept it flat, so the mini search flooded the pick/place lattice
    # below the goal branch's f and exhausted max_nodes (mixed EEs' doubled grasp alphabet
    # tipped it; uniform squeaked under). left: mated 0 / held 1 (mate remains) / else 2.
    if _part_mated(s, p):
      left = 0
    elif any(c[0] == "grasp" and c[2] == p for c in s.supports.get(p, ())):
      left = 1
      # --useful-grasps: held by a grasp that CANNOT seat the mate -> one forced
      # handover/regrasp before it (admissible +1; see spec 6.2 pricing).
      U = _problem._USEFUL_GRASPS
      if U is not None and p in U:
        held_g = next(c[3] for c in s.supports.get(p, ())
                      if c[0] == "grasp" and c[2] == p)
        if held_g not in U[p]:
          left = 2
    else:
      left = 2
    loose, holders = _loose_and_holders(s, exclude=p)
    hv = left + loose + max(0, holders - 1)
    if fill_h is not None:
      # Goal-conditioned local PDB. A miss is UNKNOWN, never a pruning certificate: retain
      # the counting lower bound so an imperfect relaxation cannot remove a concrete route.
      fh = fill_h(s)
      if fh != float("inf") and fh > hv:
        hv = fh
    elif h_extra is not None:
      # The labeled abstraction (FULL-assembly cost-to-go) keeps exactly two roles here:
      #   * h=inf PRUNES states the full assembly cannot be finished from (sound);
      #   * its VARIATION prices resting-FACE and hold/brace choices the counting h is
      #     blind to -- e.g. under --insertion-noise the frictionless table-rest mate is
      #     infeasible and the abstraction is the ONLY component that knows the target
      #     must be robot-held (measured: without it the noise fill wandered the flat
      #     local-h shell for >180s on the 2-part stool while the plan sat 4 actions deep).
      # MODULE fills must NOT join via max(): for them he is a large CONSTANT (module-
      # internal progress is invisible to the base-ideal abstraction), and max(local, he)
      # flattened h -> the mini search degenerated to uniform-cost and exhausted max_nodes
      # on the chair's back+posts module (a 4-action solution in plain sight, "fill-in
      # failed" every time). Non-module fills (settle_ref is None) take max() -- there he
      # is a genuine cost-to-go and admissible-enough for a fill (suboptimal plans are
      # fine; the main search owns optimality).
      he = h_extra(s)
      if he == float("inf"):
        return float("inf")
      if settle_ref is None:
        # SHIFT to the FILL's goal: `he` is the FULL-assembly cost-to-go, which falls by
        # ~1 for every action that advances ANY part -- so along every plausible prefix
        # f = g + max(hv, he) is FLAT and the fill A* floods one giant f-shell before the
        # goal branch (measured: peg-stool-cuboid-4-tight x keep 20 x noise 0.001 +
        # --vacuum-no-assemble: the arm-held chain sits 5 actions deep at f = f_start and
        # the flat shell exhausts max_nodes first). Subtracting the settled {ideal+p}
        # value (the milestone oracle's dist) turns `he` into THIS fill's cost-to-go:
        # 4 -> 3 -> 3 -> 2 -> 1 -> 0 along the intended chain, so the shell collapses to
        # states that are actually close to finishing THIS part. The shift can overestimate
        # (the oracle is an underestimate at the settled states), which a fill accepts:
        # completeness needs a finite search, not optimality. Module fills keep the epsilon
        # join (their settle key hides in a big constant).
        sh = he - h_extra_shift if h_extra_shift is not None else he
        if sh > hv:
          hv = sh
      else:
        hv = hv + min(he, 1e3) * 1e-4          # epsilon tie-break below one local step
    return hv

  def _held_grasp(s, part):                              # raw grasp id `part` is currently held by
    for c in s.supports.get(part, ()):
      if c[0] == "grasp" and c[2] == part:
        return c[3]
    return None

  def succ_fn(s):
    for a, ns in successors(s, assembly, robots, grasp_table=grasp_table,
                            gripper_kind=gripper_kind, handovers=handovers,
                            allow_place=allow_place, bracing=bracing):
      # RELEASE of an inherited grasp is always legal. The broad scope pass additionally admits
      # ordinary manipulation of future/blocking parts, but _fill_action_in_scope prevents an
      # unrelated assembly join from making the concrete state disagree with outer `ideal`.
      if not _fill_action_in_scope(a, relevant, p, touch_all=touch_all):
        continue
      if (grasp_cap is not None and a[0] in ("pick", "add_grasp", "handover")
          and a[3] >= grasp_cap):
        continue                                          # grasp-WIDENING: only ids < cap this pass
      if part_grasp_allow is not None and a[0] == "assemble" and a[2] == p:
        # the abstraction's viable-inserter-grasp hint applies to the MATE -- the grasp `p` is
        # HELD with when it seats -- NOT the pick: `p` may need a reorient pick with a
        # non-viable grasp to reach a viable one (measured cross-7: obj_2 picks grasp 3 to
        # reorient, then re-picks grasp 11 to mate; 11 is in the hint, 3 is not). So free
        # pick/handover, gate only the mate's held grasp; the full unbraced phase (allow=None)
        # is the completeness net when the relaxed hint misses the concrete-viable grasp.
        g = _held_grasp(s, p)
        if g is not None and g not in part_grasp_allow:
          continue
      if (brace_allow is not None and a[0] == "add_grasp"
          and (a[2], a[3]) not in brace_allow):
        continue                                          # only the abstraction's known G2 braces
      if forbid_assemble is not None and a[0] == "assemble" and a[2] in forbid_assemble:
        continue                                          # e.g. the module JOIN belongs to the
      yield a, ns                                         # dedicated join fill (tiny space =>
                                                          # cheap lazy rounds; measured 96 join
                                                          # refusals x full re-search inside the
                                                          # obj_11 fill = the 118s bottleneck)

  def edge_check_fn(prev, act, ns):
    # fully lazy (cheap geometry + IK at POP; see solve_assembly_problem)
    if not cheap_feasible(act, prev, ns, assembly, checker):
      return False
    if robot_ik is not None and act[0] in _IK_ACTIONS:
      ok, _w = robot_ik.bind(act, prev, ns, assembly)
      return ok
    return True

  def _replay(sub):
    s = start
    for a in sub:
      s = next(ns for act, ns in successors(s, assembly, robots, grasp_table=grasp_table,
                                            gripper_kind=gripper_kind, handovers=handovers,
                                            allow_place=allow_place, bracing=bracing)
               if act == a)
    return s

  def _direct_skeleton_fill():
    """Probe the deliberately tiny reviewer-baseline manipulation family.

    This is a restriction of ``succ_fn``, not a new action model: every edge still comes from
    the ordinary successor generator and passes the ordinary wrench/cheap/IK gates.  The
    augmented counters prevent pick/place/regrasp cycles while representing the four intended
    schemas: direct insert, target hold, one fused handover, and one retained target brace.
    At most 3R candidate role assignments are validated (mover, target holder, target brace,
    each drawn from R robots). Failure is only failure of this probe; the caller explicitly
    runs the complete fill next.
    """
    import time as _time

    if robot_ik is not None and getattr(robot_ik, "open_witness", False):
      if verbose:
        print(f"[fill-skeleton {p}] skipped under --open-witness: a bounded stochastic "
              "failure must remain reopenable; running the ordinary open fill")
      return None, None
    if cost_bound is not None:
      if verbose:
        print(f"[fill-skeleton {p}] skipped under incumbent cost bound {cost_bound:g}: "
              "the complete bounded fill remains authoritative")
      return None, None

    # mover-picks, target-picks, handovers, retained braces, target placements, releases,
    # steps. The step limit is the sum of the structural action caps plus one assemble.
    zero = (0, 0, 0, 0, 0, 0, 0)
    limits = (1, 1, 1, 1, 1, 3, 9)
    sk_start = (start, zero)

    def step(node, a, ns):
      s, counts = node
      vals = list(counts)
      kind, part = a[0], a[2]
      group = next((g for g in s.subassemblies() if part in g), frozenset())
      if kind == "pick":
        if p in group and settle in group:                # do not lift the completed fill again
          return None
        if p in group:
          vals[0] += 1                                    # pick incoming component once
        elif settle in group:
          vals[1] += 1                                    # optionally lift target once
        else:
          return None
      elif kind == "handover":
        if p not in group and settle not in group:
          return None
        vals[2] += 1                                      # one fused regrasp total
      elif kind == "add_grasp":
        # A genuine brace belongs on the pre-assembly target. Regrasping the mover with an
        # add+release sequence duplicates the fused handover schema and restores its plateau.
        if settle not in group or p in group:
          return None
        vals[3] += 1
      elif kind == "place":
        if settle not in group:
          return None                                    # no mover set-down/re-pick reorientation
        vals[4] += 1
      elif kind == "release":
        if p not in group and settle not in group:
          return None
        vals[5] += 1
      elif kind == "assemble":
        if part != p or settle in group:
          return None
      else:
        return None                                      # e.g. in-hand reorientation is an extension
      vals[6] += 1
      if any(v > cap for v, cap in zip(vals, limits)):
        return None
      return (ns, tuple(vals))

    def sk_succ(node):
      s, _counts = node
      for a, ns in succ_fn(s):
        nxt = step(node, a, ns)
        if nxt is not None:
          yield a, nxt

    def transition(node, action):
      return next(nxt for a, nxt in sk_succ(node) if a == action)

    blacklist = set()
    engine = _IncrementalLazySearch(
        sk_start, goal_fn=lambda n: goal_fn(n[0]), succ_fn=sk_succ,
        # This is a feasibility probe, not the complete/optimal fill. Greedy ordering avoids
        # exhaustively enumerating the equal-f grasp-assignment shell before the first obvious
        # schema (measured Cross obj_2: 12.6 s at weight one despite zero failed IK edges).
        # Exhaustion and the ordinary fallback preserve coverage independently of this order.
        cost_fn=action_cost, h_fn=lambda n: h_fn(n[0]), weight=float("inf"),
        blacklist=blacklist,
        cheap_gate=lambda prev, act, nxt: cheap_feasible(
            act, prev[0], nxt[0], assembly, checker))
    shared_bl = shared_ok = None
    if robot_ik is not None:
      shared_bl = getattr(robot_ik, "_fill_bl", None)
      if shared_bl is None:
        shared_bl = robot_ik._fill_bl = set()
        shared_ok = robot_ik._fill_ok = set()
      else:
        shared_ok = robot_ik._fill_ok

    t_sk = _time.perf_counter()
    candidates = failed = 0
    candidate_cap = max(1, 3 * len(robots))
    while True:
      sub = engine.search(max_nodes, deadline=deadline)
      if sub is None:
        if verbose:
          why = "node/deadline budget" if engine.budget_hit else "family exhausted"
          print(f"[fill-skeleton {p}] {why} after {candidates} candidate path(s), "
                f"{failed} failed IK edge(s), {_time.perf_counter() - t_sk:.2f}s -> "
                "complete local fallback")
        return None, None
      candidates += 1
      node = sk_start
      valid = True
      for a in sub:
        nxt = transition(node, a)
        s0, s1 = node[0], nxt[0]
        if robot_ik is not None and a[0] in _IK_ACTIONS:
          if (s0, a) in shared_bl:
            okb = False
          elif (s0, a) in shared_ok:
            okb = True
          else:
            okb, _w = robot_ik.bind(a, s0, s1, assembly)
            (shared_ok if okb else shared_bl).add((s0, a))
          if not okb:
            blacklist.add((node, a))
            engine.delete_edge(node, a, nxt)
            failed += 1
            valid = False
        node = nxt
      if valid:
        if verbose:
          print(f"[fill-skeleton {p}] VALIDATED {len(sub)}-action direct schema after "
                f"{candidates} candidate path(s), {failed} failed IK edge(s), "
                f"{_time.perf_counter() - t_sk:.2f}s")
        return sub, node[0]
      if candidates >= candidate_cap:
        if verbose:
          print(f"[fill-skeleton {p}] assignment probe budget {candidate_cap} reached, "
                f"{failed} failed IK edge(s), {_time.perf_counter() - t_sk:.2f}s -> "
                "complete local fallback")
        return None, None

  if skeleton_only:
    return _direct_skeleton_fill()

  if lazy_fill and robot_ik is not None and getattr(robot_ik, "open_witness", False):
    # OPT-IN fair witness validation lives in its own module and rebuilds this SMALL engine
    # between effort epochs.  Nothing below (the measured permanent-blacklist path) changes.
    from open_witness import run_open_lazy_fill

    # Cheap-gate refutations are DEAD FOREVER by legacy semantics ("cheap-infeasible: dead
    # forever"), but fairness epochs rebuild the engine with a FRESH suspended set. Without a
    # persistent set, cheap-dead edges resurrect every epoch: they re-enter candidate plans and
    # force a delete_edge repair cycle before being re-refuted. Solve-scoped on the RobotIK,
    # mirroring _fill_bl (states are value-hashed, so pairs match across fills).
    cheap_bl = getattr(robot_ik, "_open_cheap_bl", None)
    if cheap_bl is None:
      cheap_bl = robot_ik._open_cheap_bl = set()

    def _open_cheap_gate(prev, act, ns):
      ok = cheap_feasible(act, prev, ns, assembly, checker)
      if not ok:
        cheap_bl.add((prev, act))                         # dead forever (cheap layer is sound)
      return ok

    def _make_open_engine(suspended):
      suspended.update(cheap_bl)                          # dead edges stay dead across epochs
      return _IncrementalLazySearch(
          start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn,
          weight=search_weight, blacklist=suspended,
          cheap_gate=_open_cheap_gate)

    def _walk_open_plan(sub):
      s = start
      for a in sub:
        ns = next(n2 for act2, n2 in successors(
            s, assembly, robots, grasp_table=grasp_table,
            gripper_kind=gripper_kind, handovers=handovers,
            allow_place=allow_place, bracing=bracing) if act2 == a)
        yield s, a, ns
        s = ns

    subplan, end, _orep = run_open_lazy_fill(
        robot_ik=robot_ik, assembly=assembly, make_engine=_make_open_engine,
        walk_plan=_walk_open_plan, is_ik=lambda a: a[0] in _IK_ACTIONS,
        max_nodes=max_nodes, deadline=deadline,
        max_epochs=robot_ik.open_witness_epochs, verbose=verbose, tag=p)
    if subplan is None:
      _fill_prof(f"{p}:open-effort")
      return None, None
    _fill_prof(f"{p}:ok-open")
    return subplan, end

  if lazy_fill and robot_ik is not None:
    # LAZY-VALIDATED fill (LazySP hybrid, mirroring the global lazy search): the fill's
    # inner search pays only the CHEAP gate at pop; real IK runs on CANDIDATE subplan
    # edges only, failures are blacklisted and the (tiny) search reruns. Imports the
    # measured lazy economics (61 validations vs 19,515 eager checks on the cross) into
    # the fills -- the 2x2x3 module-fill bottleneck (325-373s/fill at R=4-real was
    # dominated by failed IK binds at pop).
    # SOLVE-SCOPED blacklist/ok sets (stored on the RobotIK, which lives exactly one
    # scene): an IK verdict is a deterministic function of (action, scene signature) --
    # robot_ik.bind's own cache already guarantees the same query returns the same
    # verdict -- so a failure/acceptance learned in ONE fill/round stays valid in every
    # later one. Before this the sets were per-fill: each fill re-bound and re-searched
    # the same doomed edges (measured: the 2-part stool under --insertion-noise +
    # --vacuum-no-assemble spent >180s re-validating, and solved in seconds once the
    # sets were shared).
    bl = getattr(robot_ik, "_fill_bl", None)
    if bl is None:
      bl = robot_ik._fill_bl = set()
      ok_cache = robot_ik._fill_ok = set()               # (state, action) IK-validated
    else:
      ok_cache = robot_ik._fill_ok

    engine = _IncrementalLazySearch(
        start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn,
        weight=search_weight, blacklist=bl,
        cheap_gate=lambda prev, act, ns: cheap_feasible(act, prev, ns, assembly, checker))

    # The established fast path caps LazySP repairs at 200 candidate plans. Exact local
    # mode removes that arbitrary retry cutoff: every invalid candidate deletes at least
    # one new edge, so repair continues until a valid optimum is popped, the finite
    # frontier is exhausted, or the existing node/deadline budget is reached.
    _round = 0
    while exact_fill or _round < 200:
      # INCREMENTAL repair (the global lazy search's engine, LPA*-style): the heap and the
      # claimed tree PERSIST across rounds; a blacklisted edge only un-claims its own
      # search-tree subtree and re-injects the recorded alternate entries. The old
      # fresh-_astar loop restarted from scratch every round and, keyed on (state, action),
      # needed a round PER failing (grasp-combination state, action) pair -- measured on
      # the 5-part tight stool at keep 40: 16 fully-valid arm-held chains exist, yet 200
      # restarts never covered the leg-grasp x seat-grasp space (95 s x fill, then failure).
      subplan = engine.search(max_nodes, deadline=deadline)
      if subplan is None:
        _fill_prof(f"{p}:exhausted")
        return None, None
      s = start
      valid = True
      for a in subplan:
        ns = next(n2 for act2, n2 in successors(s, assembly, robots,
                                                grasp_table=grasp_table,
                                                gripper_kind=gripper_kind,
                                                handovers=handovers,
                                                allow_place=allow_place, bracing=bracing)
                  if act2 == a)
        if a[0] in _IK_ACTIONS and (s, a) not in ok_cache:
          okb, _w = robot_ik.bind(a, s, ns, assembly)
          if not okb:
            # VALIDATE-ALL per round: binds are checks, not state changes, so the suffix
            # states stay well-defined -- keep replaying and blacklist EVERY failing edge
            # this round, each with an incremental tree repair.
            bl.add((s, a))
            engine.delete_edge(s, a, ns)
            valid = False
          else:
            ok_cache.add((s, a))
        s = ns
      if valid:
        if verbose and _round >= 10:
          from collections import Counter
          kinds = Counter((a[0], a[1], a[3] if len(a) > 3 else None) for _s, a in bl)
          top = ", ".join(f"{k}:{c}" for k, c in kinds.most_common(6))
          print(f"[fill {p}] lazy converged after {_round + 1} rounds; refuted binds "
                f"by (kind, robot, arg): {top}")
        _fill_prof(f"{p}:ok")
        return subplan, s
      _round += 1
    _fill_prof(f"{p}:rounds-out")
    return None, None

  subplan = _astar(start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn,
                   cheap_fn=None, edge_check_fn=edge_check_fn,
                   weight=search_weight, max_nodes=max_nodes, verbose=False,
                   tag=f"add {p}", deadline=deadline, cost_bound=cost_bound)
  if subplan is None:
    _fill_prof(f"{p}:astar-fail")
    return None, None
  s = start
  for a in subplan:
    s = next(ns for act, ns in successors(s, assembly, robots, grasp_table=grasp_table,
                                          gripper_kind=gripper_kind, handovers=handovers,
                                          allow_place=allow_place, bracing=bracing)
             if act == a)
  _fill_prof(f"{p}:ok-astar")
  return subplan, s


def _settle_to_goal(start, assembly, robots, checker, grasp_table, robot_ik, weight,
                    max_nodes, gripper_kind=None, handovers=True, bracing=False,
                    deadline=None, lazy_fill=False):
  # FINAL SETTLE milestone: from a complete-but-not-yet-goal state (e.g. still held, or a
  # holder remaining), reach is_goal -- i.e. place the assembly on the table and release
  # (set_goal(on_table=True)). No new part is added; only place/release/handover/add_grasp
  # on the finished assembly. Small search (a place + release chain). Returns (subplan, end)
  # or (None, None) if the goal is unreachable (e.g. the assembly cannot stand alone).
  if start.is_goal(assembly):
    return [], start
  main = set(assembly.objects)

  def goal_fn(s):
    return s.is_goal(assembly)

  def h_fn(s):
    on_t = _group_is_on_table(s, main)
    return (0 if on_t else 1) + _assembly_holders(s, assembly)   # place + releases left

  def succ_fn(s):
    for a, ns in successors(s, assembly, robots, grasp_table=grasp_table,
                            gripper_kind=gripper_kind, handovers=handovers,
                            allow_place=True, bracing=bracing):
      yield a, ns                                          # all parts already mated

  def edge_check_fn(prev, act, ns):
    if not cheap_feasible(act, prev, ns, assembly, checker):
      return False
    if robot_ik is not None and act[0] in _IK_ACTIONS:
      ok, _w = robot_ik.bind(act, prev, ns, assembly)
      return ok
    return True

  subplan = _astar(start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn,
                   cheap_fn=None, edge_check_fn=edge_check_fn, weight=weight,
                   max_nodes=max_nodes, verbose=False, tag="settle", deadline=deadline)
  if subplan is None:
    return None, None
  s = start
  for a in subplan:
    s = next(ns for act, ns in successors(s, assembly, robots, grasp_table=grasp_table,
                                          gripper_kind=gripper_kind, handovers=handovers,
                                          allow_place=True, bracing=bracing)
             if act == a)
  return subplan, s


def _add_parts(start, ps, ideal, assembly, robots, checker, grasp_table, robot_ik,
               weight, max_nodes, gripper_kind=None, handovers=True,
               allow_place=True, bracing=False, deadline=None,
               lazy_fill=False, cost_bound=None, touch_all=None,
               excluded_end_states=None):
  # WIDTH-k milestone fill (adaptive-width completeness net): add ALL parts in `ps` in ONE
  # joint mini-search -- for plans that must INTERLEAVE several parts' glue, which the
  # width-1 fill cannot express (the milestone commitment's documented gap). Invoked with
  # growing k only after every smaller width failed at the node, BEFORE backjumping:
  # escalation on proof of local failure; k = n is the flat exact search, so completeness
  # is recovered in the limit. No hint phases (hints are per-part); one full phase.
  ps = tuple(ps)
  relevant = set(ideal) | set(ps)
  excluded_end_states = frozenset(excluded_end_states or ())

  def goal_fn(s):
    return (all(_part_mated(s, p) for p in ps)
            and _num_loose_held(s, assembly) == 0
            and _assembly_holders(s, assembly) <= 1
            and s not in excluded_end_states)

  def h_fn(s):
    left = sum(0 if _part_mated(s, p) else 2 for p in ps)
    return left + _num_loose_held(s, assembly) + max(0, _assembly_holders(s, assembly) - 1)

  def succ_fn(s):
    for a, ns in successors(s, assembly, robots, grasp_table=grasp_table,
                            gripper_kind=gripper_kind, handovers=handovers,
                            allow_place=allow_place, bracing=bracing):
      if _fill_action_in_scope(a, relevant, ps, touch_all=touch_all):
                                                           # inherited release always legal;
        yield a, ns                                       # touch-all may move future blockers

  def edge_check_fn(prev, act, ns):
    if not cheap_feasible(act, prev, ns, assembly, checker):
      return False
    if robot_ik is not None and act[0] in _IK_ACTIONS:
      ok, _w = robot_ik.bind(act, prev, ns, assembly)
      return ok
    return True

  def _replay_upto(sub):
    s = start
    for a in sub:
      s = next(ns for act, ns in successors(s, assembly, robots, grasp_table=grasp_table,
                                            gripper_kind=gripper_kind, handovers=handovers,
                                            allow_place=allow_place, bracing=bracing)
               if act == a)
    return s

  if lazy_fill and robot_ik is not None:
    bl = set()                                            # (state, action) IK-refuted edges
    ok_cache = set()

    engine = _IncrementalLazySearch(
        start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn,
        weight=weight, blacklist=bl,
        cheap_gate=lambda prev, act, ns: cheap_feasible(act, prev, ns, assembly, checker))

    for _round in range(200):
      subplan = engine.search(max_nodes, deadline=deadline)
      if subplan is None:
        return None, None
      s = start
      valid = True
      for a in subplan:
        ns = next(n2 for act2, n2 in successors(s, assembly, robots,
                                                grasp_table=grasp_table,
                                                gripper_kind=gripper_kind,
                                                handovers=handovers,
                                                allow_place=allow_place, bracing=bracing)
                  if act2 == a)
        if a[0] in _IK_ACTIONS and (s, a) not in ok_cache:
          okb, _w = robot_ik.bind(a, s, ns, assembly)
          if not okb:
            bl.add((s, a))                                # validate-all (see _add_one_part)
            engine.delete_edge(s, a, ns)
            valid = False
          else:
            ok_cache.add((s, a))
        s = ns
      if valid:
        return subplan, s
    return None, None

  subplan = _astar(start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost,
                   h_fn=h_fn, cheap_fn=None, edge_check_fn=edge_check_fn, weight=weight,
                   max_nodes=max_nodes, verbose=False, tag="add " + "+".join(ps),
                   deadline=deadline, cost_bound=cost_bound)
  if subplan is None:
    return None, None
  return subplan, _replay_upto(subplan)


def solve_assembly_problem_optimistic(C, assembly, robots=None, verbose=True,
                                      robot_ik=None, fallback=True, weight=float("inf"),
                                      max_nodes=20000, wrench=False, handovers=True,
                                      allow_place=True, bracing=False, lazy_fills=False,
                                      chain_h=False, fill_local_h=False, fill_exact=False,
                                      wrench_aware_h=True, fill_skeleton_first=False,
                                      **exact_kw):
  """OPTIMISTIC (Dogar-style) planner (see the section header): add the parts one at a time,
  each via its own small complete search that reorients/holds/picks/mates/sets-down, chained
  through the on-table assembly. This drops the exact search's cross-product blowup while
  staying complete. Falls back to the exact search if any single-part step is unsolvable
  within `max_nodes`. wrench=True turns on the stability gates. allow_place=False (--no-place)
  chains HELD states instead (hold-and-work with handovers). ``lazy_fills=True`` reuses the
  milestone solver's incremental LazySP local validation: search candidate subplans under the
  cheap gate, validate only their IK edges, then repair around refutations. ``chain_h=True``
  restricts the signature abstraction to this solver's fixed prefix order and one incoming
  loose part, retaining future wrench/support guidance without the global order-ideal product.
  Detached connect modules are deliberately rejected: their state needs multiple components,
  which the module-aware milestone solver represents. A final settle search makes the returned
  plan satisfy the configured global goal, including hands-off/on-table. Returns the action
  list, or None."""
  import time

  t0 = time.perf_counter()
  if robots is None:
    robots = get_robots(C)
  if chain_h and getattr(assembly, "connect_parts", set()):
    roots = ", ".join(sorted(assembly.connect_parts))
    raise RuntimeError(
        f"--optimistic-chain-h cannot represent detached connect module(s): {roots}. "
        "Its fixed prefix chain only grows the main component, whereas these roots must be "
        "built with their internal parts before joining it. Re-run with --milestones "
        "(the module-aware solver); no optimistic fallback was attempted."
    )
  checker, grasp_table = _search_context(C, assembly, robots)
  gk = wrench_kind_arg(C, robots) if wrench else None
  hgk = gk if wrench_aware_h else None
  if gk is not None:
    reset_wrench_cache()

  order = [p for p in assembly_order(assembly) if p != assembly.base]
  ch = None
  if chain_h:
    # One backward solve over the fixed optimistic prefix chain: future-aware like milestones,
    # but without alternative order ideals or combinations of unrelated loose parts.
    classes = compute_grasp_signatures(
        assembly, grasp_table, gripper_kind=hgk,
        gripper_path=getattr(checker, "fact_gripper_path", checker.gripper_path),
        clearance=0.02)
    if verbose:
      summary = ", ".join(f"{q}:{len(set(m.values()))}/{len(m)}"
                          for q, m in sorted(classes.items()))
      print(f"[chain-sig-h] components=({','.join(_SIG_COMPONENTS)}); "
            f"classes/raw grasps per part: {summary}")
    pref, cur = [{assembly.base}], {assembly.base}
    for p in order:
      cur = set(cur) | {p}
      pref.append(cur)
    ch = CollapsedLabeledHeuristic(
        assembly, len(robots), grasp_table,
        gripper_path=getattr(checker, "fact_gripper_path", checker.gripper_path),
        verbose=verbose, gripper_kind=hgk, bracing=bracing,
        grasp_classes=classes, labeled_loose=True, ideal_chain=pref)

  fill_classes, fill_pdb_cache = None, {}
  if fill_local_h:
    fill_classes = compute_grasp_signatures(
        assembly, grasp_table, gripper_kind=hgk,
        gripper_path=getattr(checker, "fact_gripper_path", checker.gripper_path),
        clearance=0.02)
    if verbose:
      print("[fill-local-h] ACTIVE: one goal-conditioned PDB per fill "
            "(all grasps fact-signature projected); "
            + f"concrete fill search weight={'1 (exact)' if fill_exact else weight}")

  s = get_initial_state(C, assembly)
  plan = []
  ideal = {assembly.base}
  for p in order:
    phases = []                                             # (braced, mate allow, brace allow)
    shift = None
    if ch is None:
      phases.append((False, None, None))
      if bracing:
        phases.append((True, None, None))
    else:
      allow, unbraced_ok, ballow = set(), False, set()
      ikey = _subasm_key(ideal)
      for k, ok in ch._wfacts.items():
        if ok and k[0] == "A" and k[1] == ikey and k[4] == p:
          allow.update(ch._svgroups.get(p, {}).get(k[5], (k[5],)))
          unbraced_ok |= len(k[2]) == 0
          for bp, bg in k[2]:
            for rg in ch._svgroups.get(bp, {}).get(bg, (bg,)):
              ballow.add((bp, rg))
      allow, ballow = (allow or None), (ballow or None)
      if verbose:
        print(f"[chain-sig-h {p}] hints: mate={0 if allow is None else len(allow)}, "
              f"unbraced={'yes' if unbraced_ok else 'no'}, "
              f"braces={0 if ballow is None else len(ballow)}")
      if not (bracing and allow is not None and not unbraced_ok):
        if allow is not None:
          phases.append((False, allow, None))
        phases.append((False, None, None))
      if bracing:
        if ballow is not None:
          if allow is not None:
            phases.append((True, allow, ballow))
          phases.append((True, None, ballow))
        phases.append((True, None, None))
      grown = _subasm_key(set(ideal) | {p})
      shift = min((ch.dist.get((grown, (), f, ()), float("inf"))
                   for f in rest_faces(assembly, set(grown))), default=float("inf"))
      if shift == float("inf"):
        shift = 0.0                                      # retain finite-search fallback phases

    lpdb = None
    if fill_local_h:
      comps = tuple(s.subassemblies())
      target = next(frozenset(g) for g in comps if assembly.base in g)
      mover = next(frozenset(g) for g in comps if p in g)
      key = (_subasm_key(target), _subasm_key(mover), p, assembly.base)
      lpdb = fill_pdb_cache.get(key)
      if lpdb is None:
        lpdb = _make_fill_local_pdb(
            s, assembly, robots, checker, grasp_table, p, fill_classes,
            gripper_kind=hgk, bracing=bracing, component_root=assembly.base,
            verbose=verbose)
        fill_pdb_cache[key] = lpdb

    sub, s2 = None, None
    if fill_skeleton_first:
      sub, s2 = _add_one_part(
          s, p, ideal, assembly, robots, checker, grasp_table, robot_ik,
          weight, max_nodes, verbose, gripper_kind=gk, handovers=handovers,
          allow_place=allow_place, bracing=bracing, lazy_fill=lazy_fills,
          h_extra=(None if lpdb is not None or ch is None else ch.value),
          h_extra_shift=shift,
          fill_h=(lpdb.value if lpdb is not None else None),
          exact_fill=fill_exact, skeleton_only=True)
    for br, ma, ba in phases:
      if sub is not None:
        break
      sub, s2 = _add_one_part(s, p, ideal, assembly, robots, checker, grasp_table, robot_ik,
                              weight, max_nodes, verbose, gripper_kind=gk, handovers=handovers,
                              allow_place=allow_place, bracing=br, lazy_fill=lazy_fills,
                              part_grasp_allow=ma, brace_allow=ba,
                              h_extra=(None if lpdb is not None or ch is None else ch.value),
                              h_extra_shift=shift,
                              fill_h=(lpdb.value if lpdb is not None else None),
                              exact_fill=fill_exact)
      if sub is not None:
        break
    if sub is None:
      if verbose:
        suffix = ("falling back to exact" if fallback else
                  "NO exact fallback requested (--no-optimistic-fallback)")
        print(f"[optimistic] could not add {p} in isolation -> {suffix}")
      if not fallback:
        return None
      return solve_assembly_problem(C, assembly, robots=robots, verbose=verbose,
                                    robot_ik=robot_ik, weight=weight, max_nodes=max_nodes,
                                    wrench=wrench, handovers=handovers,
                                    wrench_aware_h=wrench_aware_h,
                                    allow_place=allow_place, bracing=bracing, **exact_kw)
    plan += sub
    s = s2
    ideal.add(p)
    if verbose:
      print(f"[optimistic] added {p} in {len(sub)} actions "
            f"({time.perf_counter() - t0:.1f}s, {len(plan)} total)")

  if not s.is_goal(assembly):
    sub, s2 = _settle_to_goal(s, assembly, robots, checker, grasp_table, robot_ik,
                              max(weight, 5.0), max_nodes, gripper_kind=gk,
                              handovers=handovers, bracing=bracing,
                              lazy_fill=lazy_fills)
    if sub is None:
      if verbose:
        suffix = ("falling back to exact" if fallback else
                  "NO exact fallback requested (--no-optimistic-fallback)")
        print(f"[optimistic] final settle-to-goal failed -> {suffix}")
      if not fallback:
        return None
      return solve_assembly_problem(C, assembly, robots=robots, verbose=verbose,
                                    robot_ik=robot_ik, weight=weight, max_nodes=max_nodes,
                                    wrench=wrench, handovers=handovers,
                                    wrench_aware_h=wrench_aware_h,
                                    allow_place=allow_place, bracing=bracing, **exact_kw)
    plan += sub
    s = s2
    if verbose and sub:
      print(f"[optimistic] final settle in {len(sub)} actions "
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


def iter_assembly_problem_milestones(C, assembly, robots=None, verbose=True, robot_ik=None,
                                     weight=5.0, max_nodes=20000, wrench=False,
                                     wrench_aware_h=True,
                                     handovers=True, allow_place=True, bracing=False,
                                     max_repairs=100, geo_facts=False,
                                     sig_collapsed_h=False, deadline=None,
                                     fill_window=2, lazy_fills=True,
                                     cost_bound=None, milestone_oracle="global",
                                     fill_local_h=False, fill_exact=False,
                                     fill_skeleton_first=False,
                                     fill_scope_widening=False,
                                     milestone_boundaries=1):
  """MILESTONE (Dogar-style, abstraction-guided) planner. Two levels:

    LEVEL 1: the assembly SEQUENCE. The labeled abstraction's backward Dijkstra (the same
      dist table the exact search uses as its heuristic -- itself an assembly-by-disassembly
      solve of the relaxed model, wrench facts included) is used as an ORACLE: among the
      precedence-addable next parts, try them in order of the abstract cost-to-go AFTER
      adding them (min over stable rest faces). Grasps/reach are ASSUMED feasible here.
    LEVEL 2: the manipulation glue. Each chosen milestone is filled in by _add_one_part's
      small complete search (pick/handover/brace/mate/set-down), from the previous
      milestone's settled state.
    REPAIR: a fill exhaustion blacklists the exact (parent-state, assembled-set, part)
      transition. If its suffix fails, only that settled endpoint is rejected and the fill
      lazily generates another, up to ``milestone_boundaries``. Repair therefore happens at
      milestone granularity without the former claim-once boundary assumption.

  ``milestone_oracle='structural'`` is a controlled ablation: it skips the global backward
  signature database and orders dependency-valid milestones canonically. The fill heuristic
  remains independently selectable: counting by default, or one goal-conditioned local PDB
  with ``fill_local_h=True``. All other machinery is identical.

  Separate from the exact search by design (shares only _add_one_part + the heuristic
  class). Bounded settled-boundary commitment remains (an interleaving wider than
  ``fill_window`` is out of scope); vs the optimistic solver it SEARCHES the part order
  instead of committing to assembly_order, and supports bracing in the fill-in. Returns the
  action list, or None (no silent exact fallback)."""
  import time

  if milestone_oracle not in ("global", "chain", "structural"):
    raise ValueError("milestone_oracle must be 'global', 'chain', or 'structural'")
  if int(milestone_boundaries) < 1:
    raise ValueError("milestone_boundaries must be >= 1")
  milestone_boundaries = int(milestone_boundaries)

  t0 = time.perf_counter()
  if robots is None:
    robots = get_robots(C)
  checker, grasp_table = _search_context(C, assembly, robots)
  gk = wrench_kind_arg(C, robots) if wrench else None
  hgk = gk if wrench_aware_h else None
  if _problem._USEFUL_FLAG:
    _problem.set_useful_grasps(compute_useful_grasps(assembly, grasp_table, gk))
    if verbose:
      nU = {p: len(v) for p, v in (_problem._USEFUL_GRASPS or {}).items()}
      print(f"[useful-grasps] mate-viable per part: {nU}")
  if gk is not None:
    reset_wrench_cache()
  # GLOBAL mode uses the original backward abstraction both to order milestones and to seed
  # concrete fills. STRUCTURAL mode deliberately constructs no abstraction at all: this is
  # the clean ablation against the cost of the global DB, not another optimistic DB variant.
  h_lab = None
  if milestone_oracle == "global":
    h_cls = SignatureCollapsedHeuristic if sig_collapsed_h else GraspCollapsedHeuristic
    h_lab = h_cls(assembly, len(robots), grasp_table,
                  gripper_path=getattr(checker, "fact_gripper_path", checker.gripper_path),
                  verbose=verbose, gripper_kind=hgk, bracing=bracing,
                  geo_checker=checker if geo_facts else None)
  elif milestone_oracle == "chain":
    roots = getattr(assembly, "connect_parts", set())
    if roots:
      raise RuntimeError(
          "--milestone-oracle chain cannot represent detached connect module(s): "
          + ", ".join(sorted(roots)) + ". Use --milestone-oracle structural for the "
          "cheap module-aware ablation, or global for the full backward oracle."
      )
    # Same projection as --optimistic-chain-h, retained inside the milestone driver so
    # backtracking, width windows, lazy fills and final settling remain in the ablation.
    classes = compute_grasp_signatures(
        assembly, grasp_table,
        gripper_kind=hgk,
        gripper_path=getattr(checker, "fact_gripper_path", checker.gripper_path),
        clearance=0.02)
    if verbose:
      summary = ", ".join(f"{q}:{len(set(m.values()))}/{len(m)}"
                          for q, m in sorted(classes.items()))
      print(f"[chain-sig-h] components=({','.join(_SIG_COMPONENTS)}); "
            f"classes/raw grasps per part: {summary}")
    pref, cur = [frozenset({assembly.base})], {assembly.base}
    for p in assembly_order(assembly):
      if p == assembly.base:
        continue
      cur = set(cur) | {p}
      pref.append(frozenset(cur))
    h_lab = CollapsedLabeledHeuristic(
        assembly, len(robots), grasp_table,
        gripper_path=getattr(checker, "fact_gripper_path", checker.gripper_path),
        verbose=verbose, gripper_kind=hgk, bracing=bracing,
        geo_checker=checker if geo_facts else None,
        grasp_classes=classes, labeled_loose=True, ideal_chain=pref)
  elif verbose:
    print("[milestones] structural oracle: backward signature database SKIPPED; "
          "dependency/module eligibility + canonical order choose milestones; the local "
          "fill heuristic is selected independently")

  # ONE goal-conditioned pattern database per distinct fill. Every grasp uses the same
  # fact-signature projection; there is no role-specific exception for the interface grasp.
  # Each PDB contains
  # only ideal -> ideal+p, so it is independent of whether the surrounding assembly is a
  # chain, DAG, or detached module. The cache also prevents the hinted/full/braced widening
  # phases from rebuilding the same local backward solve.
  fill_classes, fill_pdb_cache = None, {}
  if fill_local_h:
    fill_classes = compute_grasp_signatures(
        assembly, grasp_table, gripper_kind=hgk,
        gripper_path=getattr(checker, "fact_gripper_path", checker.gripper_path),
        clearance=0.02)
    if verbose:
      print("[fill-local-h] ACTIVE: one goal-conditioned PDB per fill "
            "(all grasps fact-signature projected); "
            + f"concrete fill search weight={'1 (exact)' if fill_exact else weight}")

  def fill_pdb(s, q, settle):
    if not fill_local_h:
      return None
    root = assembly.base if settle is None else settle
    comps = tuple(s.subassemblies())
    target = next(frozenset(g) for g in comps if root in g)
    mover = next(frozenset(g) for g in comps if q in g)
    key = (_subasm_key(target), _subasm_key(mover), q, root)
    hit = fill_pdb_cache.get(key)
    if hit is None:
      hit = _make_fill_local_pdb(
          s, assembly, robots, checker, grasp_table, q, fill_classes,
          gripper_kind=hgk, bracing=bracing, component_root=root,
          geo_facts=geo_facts, verbose=verbose)
      fill_pdb_cache[key] = hit
    return hit

  def oracle(ideal_set, q):
    # abstract cost-to-go once `q` is mated and the grown assembly rests SETTLED on some
    # stable face, nothing held (the state _add_one_part chains through). inf => the
    # abstraction cannot finish from there (or never reached it backward) -> try last.
    if h_lab is None:
      return 0.0
    key = _subasm_key(ideal_set | {q})
    faces = rest_faces(assembly, ideal_set | {q})
    best = float("inf")
    for f in faces:
      d = h_lab.dist.get((key, (), f, ()))
      if d is not None and d < best:
        best = d
    return best

  # PHASED fill-in, cheapest/most-constrained action set FIRST but always widening to the full
  # set before escalating to braces. Two ideas kept strictly separate after a measured bug:
  #   * the abstraction's grasp hint (viable inserter grasps of q) is a RELAXED-MODEL guess --
  #     it does NOT transfer to the concrete state (real reachability/clutter/orientation), so
  #     it may EXCLUDE the true grasp (measured: cross-7 obj_2 hint=[10,11], concrete needs
  #     another -> hinted unbraced fails in 1.3s, then the braced phase FLOODS). So the hint
  #     only adds an EARLY narrow phase; a FULL unbraced phase always follows it, and always
  #     precedes any braced phase. Hint right -> fast; hint wrong -> the full phase catches it.
  #   * bracing is rarely needed and its add_grasp edges inflate the mini plateau, so braced
  #     phases run LAST. The `unbraced_ok=False` skip is sound (the relaxed model is optimistic:
  #     if it finds NO unbraced insertion, the more-constrained concrete cannot either).
  nb = len(_problem._CUBE_FACE_NORMALS) * _problem._ROLLS_PER_FACE

  def hints(ideal_key, q):
    # mine the abstraction's cached w_assemble facts ("A", target_key, G2, face, q, gq): the
    # TRUE entries for THIS milestone give (a) the wrench-viable inserter grasps gq, (b) whether
    # any works with an EMPTY target brace set (=> an unbraced insertion exists), and (c) the
    # G2 BRACE CONFIGS -- the specific (part, grasp) braces that make the insertion feasible.
    # (c) is the key to the braced fill: instead of regenerating EVERY add_grasp option (free
    # robot x part x grasp -- measured 90-170/fill), constrain the brace to the abstraction's
    # known set. Collapsed ids expand to raw variants. Facts cached lazily => absence = no info.
    if h_lab is None:
      # No relaxed wrench certificate exists in the structural ablation. Try the complete
      # unbraced phase, followed by the complete braced phase when enabled.
      return None, True, None
    gqs, unbraced_ok, braces = set(), False, set()
    for k, ok in h_lab._wfacts.items():
      if ok and k[0] == "A" and k[1] == ideal_key and k[4] == q:
        gqs.update(h_lab._svgroups.get(q, {}).get(k[5], (k[5],)))
        unbraced_ok |= len(k[2]) == 0
        for bp, bg in k[2]:                               # G2 = (part, collapsed grasp) braces
          for rg in h_lab._svgroups.get(bp, {}).get(bg, (bg,)):
            braces.add((bp, rg))
    return (gqs or None), unbraced_ok, (braces or None)

  def fill(s, q, ideal, fb=None, forbid=None, settle=None, excluded_end_states=None):
    allow, unbraced_ok, ballow = hints(_subasm_key(ideal), q)
    # h-shift for NON-module fills: the settled {ideal+q} abstraction value (see
    # _add_one_part.h_fn). Module fills (settle is not None) keep the epsilon join.
    shift = None
    if settle is None:
      shift = oracle(set(ideal), q)
    dense = _problem._DENSE_GRASPS
    lpdb = fill_pdb(s, q, settle)
    # phases, narrowest first: (grasp_cap, bracing, mate_allow, brace_allow). A FULL phase (all
    # grasps, no brace restriction) always precedes escalation -- the completeness net when a
    # relaxed hint misses the concrete-viable grasp/brace.
    phases = []
    if not (bracing and allow is not None and not unbraced_ok):
      if allow is not None:
        phases.append((nb if dense else None, False, allow, None))   # hinted mate
      if dense:
        phases.append((nb, False, None, None))
      phases.append((None, False, None, None))                       # full unbraced
    if bracing:
      if ballow is not None:                                         # abstraction knows the brace:
        if allow is not None:
          phases.append((nb if dense else None, True, allow, ballow))  # hinted mate + brace
        phases.append((None, True, None, ballow))                    # any mate + hinted brace
      phases.append((None, True, None, None))                        # full braced (net)
    scope_passes = _fill_scope_passes(_FILL_TOUCH_ALL, fill_scope_widening)
    if fill_skeleton_first:
      sub, s2 = _add_one_part(
          s, q, ideal, assembly, robots, checker, grasp_table, robot_ik,
          weight, max_nodes, verbose, gripper_kind=gk,
          handovers=handovers, allow_place=allow_place, bracing=bracing,
          deadline=deadline, lazy_fill=lazy_fills, cost_bound=fb,
          forbid_assemble=forbid, settle_ref=settle,
          h_extra=(None if lpdb is not None else
                   h_lab.value if h_lab is not None else None),
          h_extra_shift=shift,
          fill_h=(lpdb.value if lpdb is not None else None),
          exact_fill=fill_exact, skeleton_only=True,
          touch_all=scope_passes[0], excluded_end_states=excluded_end_states)
      if sub is not None:
        return sub, s2
    for touch_all in scope_passes:
      if touch_all and not _FILL_TOUCH_ALL and verbose:
        print(f"[fill-scope {q}] local grasp/bracing phases exhausted -> widening to "
              "all-part manipulation", flush=True)
      for cap, br, ma, ba in phases:
        sub, s2 = _add_one_part(
            s, q, ideal, assembly, robots, checker, grasp_table, robot_ik,
            weight, max_nodes, verbose, gripper_kind=gk,
            handovers=handovers, allow_place=allow_place, bracing=br,
            grasp_cap=cap,
            h_extra=(None if lpdb is not None else
                     h_lab.value if h_lab is not None else None),
            part_grasp_allow=ma,
            brace_allow=ba, deadline=deadline, lazy_fill=lazy_fills,
            cost_bound=fb, forbid_assemble=forbid, settle_ref=settle,
            h_extra_shift=shift,
            fill_h=(lpdb.value if lpdb is not None else None),
            exact_fill=fill_exact, touch_all=touch_all,
            excluded_end_states=excluded_end_states)
        if sub is not None:
          return sub, s2
    return None, None

  objs = frozenset(assembly.objects)
  plan = []
  # MODULE support (proper-DAG assemblies, e.g. cube-plates): a connect ROOT q stands for
  # its whole module -- the order ideal {p : q in closure(p)} that must be built DETACHED
  # and joined as a unit (assembles() already does the join; only this sequencer was
  # part-by-part-onto-main before, so it dead-ended on every connect root). A root becomes
  # a candidate once the join's OUT-OF-MODULE requirements (the plate below complete) are
  # in `ideal`; taking it runs a composite milestone: sub-fills over the module's parts on
  # the table (each settled -- the fill goal's loose-held=0 forces the set-down), then the
  # JOIN fill (pick up the finished module by any member, mate its seam, release).
  connect = getattr(assembly, "connect_parts", set()) & objs
  # NOTE _mate_closure(p) EXCLUDES p itself, so the root must be added explicitly (without
  # it the module misses its root, `grown` never includes it, and the sequencer loops on a
  # trivially-complete 0-action module forever -- measured).
  modules = {r: frozenset(p for p in objs if r in _mate_closure(assembly, p)) | {r}
             for r in connect}

  def fill_module(s, root, ideal, fb=None, excluded_end_states=None):
    seq = sorted(modules[root] - {root},
                 key=lambda m: (len(_mate_closure(assembly, m)), _part_index(m)))
    # A detached module used to commit to ONE endpoint after every internal part. If a later
    # module part or the final join failed, the whole outer transition was rejected even when
    # another holder/rest-face endpoint existed. Recurse over the fixed dependency-valid module
    # order and lazily ask each fill for another exact settled endpoint only after its suffix
    # fails. This is the same boundary policy as the outer checkpoint stack, not a Chair patch.
    def rec(i, s_cur, built, budget):
      if i == len(seq):
        return fill(s_cur, root, set(ideal) | set(modules[root]), fb=budget,
                    excluded_end_states=excluded_end_states)       # final module JOIN
      m = seq[i]
      rejected = set()
      for k in range(milestone_boundaries):
        # settle=root: the module may end each sub-fill HELD by one carrier (built in the
        # air); the base-centric default would count it as loose and force a table set-down.
        sub, s_next = fill(s_cur, m, built, fb=budget, forbid={root}, settle=root,
                           excluded_end_states=rejected)
        if sub is None:
          return None, None
        next_budget = (None if budget is None else
                       budget - sum(action_cost(a) for a in sub))
        tail, s_end = rec(i + 1, s_next, set(built) | {m}, next_budget)
        if tail is not None:
          return sub + tail, s_end
        rejected.add(s_next)
        if verbose:
          print(f"[boundaries module {root}/{m}] suffix rejected endpoint "
                f"{k + 1}/{milestone_boundaries}; requesting another", flush=True)
      return None, None

    return rec(0, s, {root}, fb)

  # Chronological BACKJUMPING over milestones. ``via`` is
  # ((exact parent State, parent_ideal_key, part-or-window), exact settled endpoint). A suffix
  # failure rejects only that concrete endpoint; the parent transition stays open until K
  # distinct endpoints fail. The parent State is essential: the same next part may fail from
  # boundary A and succeed from boundary B.
  # This is lazy K-boundary retention: ordinary successful runs still pay for one fill.
  stack = [(get_initial_state(C, assembly), frozenset({assembly.base}), 0, None)]
  blacklist = set()                                       # transitions exhausted in every endpoint
  boundary_rejects = {}                                   # transition -> {settled State endpoints}
  repairs = 0

  def rejected_for(decision):
    return boundary_rejects.setdefault(decision, set())

  def decision_for(state, ideal_key, choice):
    return (state, ideal_key, choice)

  def decision_label(decision):
    return decision[1:]

  def reject_via(via, reason):
    if via is None:
      return
    decision, endpoint = via
    rejected = rejected_for(decision)
    rejected.add(endpoint)
    exhausted = len(rejected) >= milestone_boundaries
    if exhausted:
      blacklist.add(decision)
    if verbose:
      print(f"[boundaries] {reason}: rejected endpoint {len(rejected)}/"
            f"{milestone_boundaries} for {decision_label(decision)}"
            + (" -> transition exhausted" if exhausted else " -> transition remains open"),
            flush=True)
  while stack:
    if deadline is not None and time.perf_counter() > deadline:
      if verbose:
        print(f"[milestones] deadline reached after {repairs} repairs "
              f"({time.perf_counter() - t0:.1f}s) -> no plan")
      return None
    s, ideal, plen, _via = stack[-1]
    if ideal == objs:
      final = plan[:plen]
      s_final = s
      if not s.is_goal(assembly):
        # FINAL SETTLE (set_goal on_table): all parts mated but the assembly is not yet at
        # the goal (still held / a holder remains) -> place it down and release. On failure
        # (cannot stand alone at this build), treat as a dead end and backjump.
        sub, s2 = _settle_to_goal(s, assembly, robots, checker, grasp_table, robot_ik,
                                  max(weight, 5.0), max_nodes, gripper_kind=gk,
                                  handovers=handovers, bracing=bracing,
                                  lazy_fill=lazy_fills, deadline=deadline)
        if sub is None:
          if verbose:
            print(f"[milestones] final settle-to-table FAILED -> backjump", flush=True)
          _s, _i, _p, via = stack.pop()
          if via is not None:
            reject_via(via, "final settle failure")
            repairs += 1
          continue
        final = final + sub
        s_final = s2
      if verbose:
        print(f"[milestones] complete: {len(final)} actions in "
              f"{time.perf_counter() - t0:.2f}s ({repairs} repairs)")
      # Stopping iteration here is the historical first-plan behavior. Resuming means an
      # EXTERNAL consumer suspended this complete realization (normally after finite MP
      # failure): keep all search/IK/PDB state, exclude this incumbent's final checkpoint for
      # the current enumeration, and continue. The outer loop retains `final` and may reopen
      # it later with greater MP effort; this rejection is not an infeasibility certificate.
      yield final
      _s, _i, _p, via = stack.pop()
      if via is not None:
        reject_via(via, "external complete-plan suspension")
        repairs += 1
      continue
    ideal_key = _subasm_key(ideal)
    cands = [q for q in objs - ideal
             if decision_for(s, ideal_key, q) not in blacklist
             and len(rejected_for(decision_for(s, ideal_key, q))) < milestone_boundaries
             and ((q in connect and (assembly.connect_requires.get(q, set())
                                     - modules[q]) <= ideal)
                  if q in connect
                  else (_mate_closure(assembly, q) - {q}) <= ideal)]
    cands.sort(key=lambda q: (oracle(set(ideal), q), _part_index(q)))
    progressed = False
    fb = None
    if cost_bound is not None:
      fb = cost_bound - sum(action_cost(a) for a in plan[:plen])
      if fb <= 0:
        _s, _i, _p, via = stack.pop()                    # prefix already >= bound
        if via is not None:
          reject_via(via, "incumbent cost bound")
          repairs += 1
        continue
    import os as _os
    if _os.environ.get("MILESTONE_DEBUG"):
      print(f"[mile-debug] ideal={sorted(ideal)} cands={cands} "
            f"connect={sorted(connect)} modules_keys={sorted(modules)}", flush=True)
    for q in cands:
      tq = time.perf_counter()
      decision = decision_for(s, ideal_key, q)
      excluded = rejected_for(decision)
      if q in connect:
        sub, s2 = fill_module(s, q, ideal, fb=fb, excluded_end_states=excluded)
      else:
        sub, s2 = fill(s, q, ideal, fb=fb, excluded_end_states=excluded)
      grown = (ideal | modules[q]) if q in connect else (ideal | {q})
      if sub is not None:
        del plan[plen:]
        plan.extend(sub)
        stack.append((s2, grown, len(plan), (decision, s2)))
        progressed = True
        if verbose:
          what = (f"module {q} ({len(modules[q])} parts, incl. join)"
                  if q in connect else q)
          print(f"[milestones] added {what} in {len(sub)} actions "
                f"({time.perf_counter() - tq:.1f}s, {len(plan)} total; boundary "
                f"{len(excluded) + 1}/{milestone_boundaries})")
        break
      blacklist.add(decision)
      repairs += 1
      _capture_fill_failure(s, q, ideal, f"repair{repairs}")
      if verbose:
        print(f"[milestones] fill-in for {q} failed ({time.perf_counter() - tq:.1f}s) "
              f"-> blacklisted ({repairs} repairs)")
      if repairs > max_repairs:
        if verbose:
          print("[milestones] repair budget exhausted")
        return None
    if fill_window >= 2 and not progressed:
      # WIDTH-k WINDOWS (adaptive-width escalation): every single-part candidate failed at
      # this node -- before backjumping, try adding k parts in ONE joint fill (see
      # _add_parts), k escalating 2..fill_window only as each smaller width exhausts.
      # Singles always get priority; windows fire only on proof of width-1 failure.
      for w in range(2, fill_window + 1):
        if progressed:
          break
        wins = [(p,) for p in sorted(objs - ideal)
                if p not in connect and (_mate_closure(assembly, p) - {p}) <= ideal]
        for _ in range(w - 1):
          wins = [t + (q,) for t in wins for q in sorted(objs - ideal - set(t))
                  if q not in connect
                  and (_mate_closure(assembly, q) - {q}) <= ideal | set(t)]
        uniq, seen_sets = [], set()
        for t in wins:                                    # one canonical ORDER per part-set
          fs = frozenset(t)
          decision = decision_for(s, ideal_key, tuple(sorted(t)))
          if (fs not in seen_sets and decision not in blacklist
              and len(rejected_for(decision)) < milestone_boundaries):
            seen_sets.add(fs)
            uniq.append(t)
        uniq.sort(key=lambda t: (oracle(set(ideal) | set(t[:-1]), t[-1]),
                                 tuple(_part_index(p) for p in t)))
        if len(uniq) > 100:
          if verbose:
            print(f"[milestones] width-{w} windows capped 100/{len(uniq)}")
          uniq = uniq[:100]
        for t in uniq:
          tq = time.perf_counter()
          decision = decision_for(s, ideal_key, tuple(sorted(t)))
          excluded = rejected_for(decision)
          sub, s2 = None, None
          for touch_all in _fill_scope_passes(_FILL_TOUCH_ALL, fill_scope_widening):
            if touch_all and not _FILL_TOUCH_ALL and verbose:
              print(f"[fill-scope window {t}] local scope exhausted -> widening to "
                    "all-part manipulation", flush=True)
            sub, s2 = _add_parts(
                s, t, ideal, assembly, robots, checker, grasp_table,
                robot_ik, weight, max_nodes, gripper_kind=gk,
                handovers=handovers, allow_place=allow_place,
                bracing=bracing, deadline=deadline,
                lazy_fill=lazy_fills, cost_bound=fb, touch_all=touch_all,
                excluded_end_states=excluded)
            if sub is not None:
              break
          if sub is not None:
            del plan[plen:]
            plan.extend(sub)
            stack.append((s2, ideal | set(t), len(plan), (decision, s2)))
            progressed = True
            if verbose:
              print(f"[milestones] added WINDOW {t} in {len(sub)} actions "
                    f"({time.perf_counter() - tq:.1f}s, {len(plan)} total; boundary "
                    f"{len(excluded) + 1}/{milestone_boundaries})")
            break
          blacklist.add(decision)
          repairs += 1
          if verbose:
            print(f"[milestones] window {t} failed ({time.perf_counter() - tq:.1f}s) "
                  f"-> blacklisted ({repairs} repairs)")
          if repairs > max_repairs:
            if verbose:
              print("[milestones] repair budget exhausted")
            return None
    if not progressed:
      _s, _i, _p, via = stack.pop()                       # dead end: backjump + blame the
      if via is not None:                                 # decision that created this node
        reject_via(via, f"dead end at {sorted(_i)}")
        repairs += 1
        if verbose:
          print(f"[milestones] dead end at {sorted(_i)} -> backjump "
                f"({repairs} repairs)")
        if repairs > max_repairs:
          if verbose:
            print("[milestones] repair budget exhausted")
          return None
  if verbose:
    print(f"[milestones] search space exhausted with no plan "
          f"in {time.perf_counter() - t0:.2f}s ({repairs} repairs)")
  return None


def solve_assembly_problem_milestones(C, assembly, robots=None, verbose=True, robot_ik=None,
                                      weight=5.0, max_nodes=20000, wrench=False,
                                      wrench_aware_h=True,
                                      handovers=True, allow_place=True, bracing=False,
                                      max_repairs=100, geo_facts=False,
                                      sig_collapsed_h=False, deadline=None,
                                      fill_window=2, lazy_fills=True,
                                      cost_bound=None, milestone_oracle="global",
                                      fill_local_h=False, fill_exact=False,
                                      fill_skeleton_first=False,
                                      fill_scope_widening=False,
                                      milestone_boundaries=1):
  """Compatibility wrapper returning only the resumable session's first candidate."""
  return next(iter_assembly_problem_milestones(
      C, assembly, robots=robots, verbose=verbose, robot_ik=robot_ik,
      weight=weight, max_nodes=max_nodes, wrench=wrench, wrench_aware_h=wrench_aware_h,
      handovers=handovers, allow_place=allow_place, bracing=bracing,
      max_repairs=max_repairs, geo_facts=geo_facts, sig_collapsed_h=sig_collapsed_h,
      deadline=deadline, fill_window=fill_window, lazy_fills=lazy_fills,
      cost_bound=cost_bound, milestone_oracle=milestone_oracle,
      fill_local_h=fill_local_h, fill_exact=fill_exact,
      fill_skeleton_first=fill_skeleton_first,
      fill_scope_widening=fill_scope_widening,
      milestone_boundaries=milestone_boundaries), None)


def solve_assembly_problem_milestones_anytime(C, assembly, robots=None, time_budget=300.0,
                                              verbose=True, robot_ik=None, weight=5.0,
                                              wrench=False, wrench_aware_h=True, handovers=True,
                                              allow_place=True, bracing=False,
                                              geo_facts=False, sig_collapsed_h=False,
                                              labeled_h=False, collapsed_h=False,
                                              grasp_collapsed_h=False, fill_window=2,
                                              milestone_oracle="global",
                                              fill_local_h=False, fill_exact=False,
                                              fill_skeleton_first=False,
                                              fill_scope_widening=False,
                                              milestone_boundaries=1):
  """THE SINGLE RECIPE (--milestones-anytime SEC): adaptive-width milestones under a hard
  deadline, then spend whatever budget remains on principled add-ons -- one algorithm plus
  its own refinements, not a solver lottery:

    1. MILESTONES (DAG modules + width-2 windows), deadline = the full budget. The only
       method with 100%% coverage on every measured instance/robot-count/embodiment.
    2. If it produced a plan and time remains: BOUNDED IMPROVEMENT -- the exact search
       under the incumbent's cost (PEA* engine: memory-safe), deadline = remaining budget.
       Returns the better plan; the incumbent survives a deadline hit by construction.
    3. If milestones itself was cut by the deadline: PEA-GREEDY net (weight=inf, banded
       frontier => no OOM) with the remaining time -- fast plan or honest None.

  Result <= min(milestone cost, weight x optimal) when phase 2 completes; always the best
  plan found; never OOM; never hangs past the budget."""
  import math
  import time
  t0 = time.perf_counter()
  t_end = t0 + time_budget
  # NET RESERVE: milestones must NOT get the whole budget -- if the deadline cuts it,
  # the greedy net needs time to run (measured: giving milestones deadline=t_end left
  # -0.3s for the net => 'no plan' despite greedy solving the instance in 23s).
  t_mile = t_end - min(60.0, 0.25 * time_budget)
  inc = solve_assembly_problem_milestones(
      C, assembly, robots=robots, verbose=verbose, robot_ik=robot_ik,
      weight=max(weight, 5.0), wrench=wrench, wrench_aware_h=wrench_aware_h,
      handovers=handovers,
      allow_place=allow_place, bracing=bracing, geo_facts=geo_facts,
      sig_collapsed_h=sig_collapsed_h, deadline=t_mile, fill_window=fill_window,
      milestone_oracle=milestone_oracle, fill_local_h=fill_local_h,
      fill_exact=fill_exact, fill_skeleton_first=fill_skeleton_first,
      fill_scope_widening=fill_scope_widening,
      milestone_boundaries=milestone_boundaries)
  remaining = t_end - time.perf_counter()
  if inc is None:
    if remaining > 5.0:
      if verbose:
        print(f"[mile-anytime] milestones cut/failed -> PEA-greedy net "
              f"({remaining:.0f}s left)")
      return solve_assembly_problem(
          C, assembly, robots=robots, verbose=verbose, robot_ik=robot_ik,
          weight=float("inf"), wrench=wrench, handovers=handovers,
          allow_place=allow_place, bracing=bracing, pea=True, deadline=t_end)
    if verbose:
      print("[mile-anytime] no plan within budget")
    return None
  U = sum(action_cost(a) for a in inc)
  if remaining > 10.0:
    if verbose:
      print(f"[mile-anytime] incumbent cost {U:g} ({len(inc)} actions) "
            f"in {time.perf_counter() - t0:.1f}s -> bounded improvement, "
            f"{remaining:.0f}s left")
    best = solve_assembly_problem(
        C, assembly, robots=robots, verbose=verbose, robot_ik=robot_ik, weight=weight,
        wrench=wrench, handovers=handovers, allow_place=allow_place, bracing=bracing,
        labeled_h=labeled_h, collapsed_h=collapsed_h,
        grasp_collapsed_h=grasp_collapsed_h, sig_collapsed_h=sig_collapsed_h,
        geo_facts=geo_facts, cost_bound=U, pea=True, deadline=t_end)
    if best is not None and sum(action_cost(a) for a in best) < U:
      if verbose:
        print(f"[mile-anytime] improved {U:g} -> {sum(action_cost(a) for a in best):g} "
              f"in {time.perf_counter() - t0:.1f}s total")
      return best
  if verbose:
    print(f"[mile-anytime] returning the milestone plan (cost {U:g}) "
          f"in {time.perf_counter() - t0:.1f}s total")
  return inc


def _refill_window(start, ps, ideal, target, assembly, robots, checker, grasp_table,
                   robot_ik, weight, max_nodes, h_lab, gripper_kind=None, handovers=True,
                   allow_place=True, bracing=False, deadline=None, cost_bound=None):
  """Re-fill the window's parts `ps` from state `start`. If `target` is a State, the search
  must arrive at THAT EXACT settled state (so the plan's suffix splices back unchanged); if
  target is None the goal is the real assembly goal (the k=m / whole-plan window). Uses the
  LABELED heuristic: to the real goal it is h_lab(s); to a fixed intermediate target it is
  max(0, h_lab(s) - h_lab(target)) -- admissible because the abstraction distance is
  consistent (h_lab(s) <= cost(s->target) + h_lab(target)). Lazy-validated like the fills."""
  ps = tuple(ps)
  relevant = set(ideal) | set(ps)
  # DIFFERENTIAL (landmark) heuristic, uniform across all k: h(s) = h_abs(s) - h_abs(target),
  # the same abstraction distance the whole system uses, with the goal as landmark --
  # admissible by consistency (h_abs(s) <= cost(s->target) + h_abs(target)). For the k=m
  # window target IS the goal, so h_abs(target)=0 and this is plain h_abs(s). No special
  # cases. `h_lab` is the CLI-selected abstraction (sig-press at R=4 -- no explosion).
  h_tgt = h_lab.value(target) if target is not None else 0.0
  if target is not None:
    tkey = target.key()
    def goal_fn(s):
      return s.key() == tkey
  else:
    def goal_fn(s):
      return s.is_goal(assembly)

  def h_fn(s):
    return max(0.0, h_lab.value(s) - h_tgt)

  def succ_fn(s):
    for a, ns in successors(s, assembly, robots, grasp_table=grasp_table,
                            gripper_kind=gripper_kind, handovers=handovers,
                            allow_place=allow_place, bracing=bracing):
      if a[2] in relevant or a[0] == "release":           # inherited-grasp release always
        yield a, ns                                       # legal (see _add_one_part)

  def cheap_only(prev, act, ns):
    return cheap_feasible(act, prev, ns, assembly, checker)

  if robot_ik is not None:                                # LAZY-validated (see _add_parts)
    bl, ok_cache = set(), set()

    def succ_lazy(s):
      for a, ns in succ_fn(s):
        if (s, a) not in bl:
          yield a, ns

    for _round in range(200):
      subplan = _pea_star(start, goal_fn=goal_fn, succ_fn=succ_lazy, cost_fn=action_cost,
                          h_fn=h_fn, cheap_fn=None, edge_check_fn=cheap_only, weight=weight,
                          max_nodes=max_nodes, verbose=False, tag="lns refill",
                          deadline=deadline, cost_bound=cost_bound)
      if subplan is None:
        return None
      s = start
      valid = True
      for a in subplan:
        ns = next(n2 for act2, n2 in successors(s, assembly, robots,
                                                grasp_table=grasp_table,
                                                gripper_kind=gripper_kind,
                                                handovers=handovers,
                                                allow_place=allow_place, bracing=bracing)
                  if act2 == a)
        if a[0] in _IK_ACTIONS and (s, a) not in ok_cache:
          okb, _w = robot_ik.bind(a, s, ns, assembly)
          if not okb:
            bl.add((s, a))
            valid = False
          else:
            ok_cache.add((s, a))
        s = ns
      if valid:
        return subplan
    return None

  def edge_check_fn(prev, act, ns):
    return cheap_feasible(act, prev, ns, assembly, checker)
  return _pea_star(start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn,
                   cheap_fn=None, edge_check_fn=edge_check_fn, weight=weight,
                   max_nodes=max_nodes, verbose=False, tag="lns refill", deadline=deadline,
                   cost_bound=cost_bound)


def _lns_improve(plan, C, assembly, robots, checker, grasp_table, robot_ik, weight,
                 max_nodes, verbose, h_lab, gripper_kind=None, handovers=True,
                 allow_place=True, bracing=False, deadline=None, t0=None, k_max=None,
                 object_unit=True):
  """LARGE-NEIGHBORHOOD improvement -- ONE axis: neighborhood size k.

  object_unit (default): k = number of consecutive MATED PARTS. Cut points are the states
  right after each assemble, so k=1 is one object, k=n is the whole plan. A window of k
  parts is re-filled from its start state to the EXACT state after the k-th part (k=n: to
  the goal), touching only the parts the window's actions touch, bounded by the window's
  current cost. Endpoints are pinned exactly (stricter -- may reproduce a held boundary
  state), which is relaxed automatically at higher k as boundaries reach the goal.

  object_unit=False (baseline): k = SETTLED segments (a part mated, nothing loose, <=1
  holder); coarser and uneven (segments merge under hold-through re-fills). Kept for
  ablation.

  Growing k covers reorderings a small window can't reach; k=n is the whole-plan bounded
  search. Standard LNS: RESTART the sweep on any accept (the plan, hence its partition,
  changes). No separate proof rung -- the k=n window is the last iteration."""
  import time
  if t0 is None:
    t0 = time.perf_counter()

  def cost(pl):
    return sum(action_cost(a) for a in pl)

  def _step(s, a):
    return next(ns for act, ns in successors(s, assembly, robots, grasp_table=grasp_table,
                                             gripper_kind=gripper_kind, handovers=handovers,
                                             allow_place=allow_place, bracing=bracing)
                if act == a)

  def checkpoints_object(pl):
    # cut after every assemble (mate): states[0..n], bnds[0..n], seg_parts[0..n-1] = the
    # part(s) mated in each cut. n = number of assembles. Last boundary extends to len(pl)
    # so trailing release-to-settle actions live in the final window (which targets goal).
    s = get_initial_state(C, assembly)
    states, bnds = [s], [0]
    mated_prev = {p for p in assembly.objects if _part_mated(s, p)}
    seg_parts = []
    for i, a in enumerate(pl):
      s = _step(s, a)
      mated = {p for p in assembly.objects if _part_mated(s, p)}
      if mated > mated_prev:
        states.append(s)
        bnds.append(i + 1)
        seg_parts.append(tuple(sorted(mated - mated_prev)))
        mated_prev = mated
    if bnds[-1] != len(pl):                               # trailing actions -> final window
      bnds[-1] = len(pl)
      states[-1] = s
    return states, bnds, seg_parts

  def checkpoints_segment(pl):
    # SETTLED-segment partition (baseline): cut only at settled checkpoints.
    s = get_initial_state(C, assembly)
    states, bnds, mated_prev = [s], [0], {p for p in assembly.objects if _part_mated(s, p)}
    seg_parts = []
    for i, a in enumerate(pl):
      s = _step(s, a)
      mated = {p for p in assembly.objects if _part_mated(s, p)}
      if (mated > mated_prev and _num_loose_held(s, assembly) == 0
          and _assembly_holders(s, assembly) <= 1):
        states.append(s)
        bnds.append(i + 1)
        seg_parts.append(tuple(sorted(mated - mated_prev)))
        mated_prev = mated
    if bnds[-1] != len(pl):
      bnds[-1] = len(pl)
      states[-1] = s
    return states, bnds, seg_parts

  checkpoints = checkpoints_object if object_unit else checkpoints_segment
  cur = list(plan)
  while True:
    if deadline is not None and time.perf_counter() > deadline:
      return cur
    states, bnds, seg_parts = checkpoints(cur)
    m = len(seg_parts)
    km = m if k_max is None else min(k_max, m)
    # LOWER-BOUND certificates (h admissible): if the plan cost has reached h(s0), it is
    # PROVABLY optimal -- stop; per window, if the span cost equals its differential lower
    # bound max(0, h(start)-h(target)), that window is provably unimprovable -- skip it.
    h0 = h_lab.value(states[0])
    if cost(cur) <= h0:
      if verbose:
        print(f"[lns] PROVEN OPTIMAL: cost {cost(cur):g} == heuristic lower bound "
              f"h(s0) at {time.perf_counter() - t0:.1f}s", flush=True)
      return cur
    accepted = False
    for k in range(1, km + 1):
      if accepted:
        break
      for i in range(m - k + 1):
        if deadline is not None and time.perf_counter() > deadline:
          return cur
        span = cur[bnds[i]:bnds[i + k]]
        old_cost = cost(span)
        is_last = (i + k == m)
        h_start = h_lab.value(states[i])
        h_tgt = 0.0 if is_last else h_lab.value(states[i + k])
        if old_cost <= max(0.0, h_start - h_tgt):
          continue                                        # window already at its lower bound
        ideal0 = {p for p in assembly.objects if _part_mated(states[i], p)}
        # RELEVANT parts the re-fill may touch = everything the window's actions touch
        # (mated-in-window + any staged/released part) beyond what is already mated. This
        # makes reaching the exact end state possible regardless of interleaving.
        ps = {a[2] for a in span} - ideal0
        parts_mated = sorted({p for sp in seg_parts[i:i + k] for p in sp})
        sub = _refill_window(states[i], ps, ideal0,
                             None if is_last else states[i + k], assembly, robots, checker,
                             grasp_table, robot_ik, weight, max_nodes, h_lab,
                             gripper_kind=gripper_kind, handovers=handovers,
                             allow_place=allow_place, bracing=bracing, deadline=deadline,
                             cost_bound=old_cost)
        if sub is not None and cost(sub) < old_cost:
          cur = cur[:bnds[i]] + list(sub) + cur[bnds[i + k]:]
          if verbose:
            print(f"[lns] k={k} parts[{i}:{i + k}] {tuple(parts_mated)}: "
                  f"-{old_cost - cost(sub):g} -> total {cost(cur):g} at "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
          accepted = True
          break
    if not accepted:
      return cur


def solve_assembly_problem_ladder(C, assembly, robots=None, time_budget=900.0,
                                  verbose=True, robot_ik=None, weight=5.0, wrench=False,
                                  handovers=True, allow_place=True, bracing=False,
                                  geo_facts=False, sig_collapsed_h=False, labeled_h=False,
                                  collapsed_h=False, grasp_collapsed_h=False,
                                  lns=False, rwa_proof=False, lns_object=True,
                                  wrench_aware_h=True):
  """THE LADDER (--ladder SEC): the unified anytime solver -- ONE commitment axis (fill
  window k), ONE monotone thread (the incumbent bound U). Rungs:

    k=1  milestones, lazy fills, bound U propagated INTO the fills
         (G0 + g_fill + max(mini-h, h_lab) >= U prunes -- any plan returned is strictly
         cheaper than U by construction)
    k=2  same with width-2 windows (less commitment)
    k=n  the limit: bounded exact search at WEIGHT 1 (PEA* engine). Under a bound the
         weight only orders exploration -- the sublevel set is fixed -- so w=1 costs no
         more to exhaust and its FIRST goal pop is optimal-under-U; exhaustion PROVES U
         optimal (claim-once caveat). No luck involved.

  Budget: rungs 1-2 share the first ~55%% (each cut by its slice), the proof rung gets
  the rest. Returns the best plan found; prints PROVEN OPTIMAL when the last rung
  exhausts."""
  import time
  t0 = time.perf_counter()
  t_end = t0 + time_budget
  U, best = None, None
  slices = {1: t0 + 0.35 * time_budget, 2: t0 + 0.55 * time_budget}
  rungs = (1,) if lns else (1, 2)
  for k in rungs:
    if time.perf_counter() > slices[k] - 5.0:
      continue
    if verbose:
      print(f"[ladder] rung k={k}"
            + (f" under bound {U:g}" if U is not None else "") , flush=True)
    plan = solve_assembly_problem_milestones(
        C, assembly, robots=robots, verbose=verbose, robot_ik=robot_ik,
        weight=max(weight, 5.0), wrench=wrench, wrench_aware_h=wrench_aware_h,
        handovers=handovers,
        allow_place=allow_place, bracing=bracing, geo_facts=geo_facts,
        sig_collapsed_h=sig_collapsed_h, deadline=slices[k], fill_window=k,
        lazy_fills=True, cost_bound=U)
    if plan is not None:
      U = sum(action_cost(a) for a in plan)
      best = plan
      if verbose:
        print(f"[ladder] rung k={k}: cost {U:g} ({len(plan)} actions) at "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
    elif verbose:
      print(f"[ladder] rung k={k}: nothing cheaper", flush=True)
  if lns and best is None and time.perf_counter() < slices[2] - 5.0:
    # FEASIBILITY fallback only (LNS subsumes k=2 for improvement): wider windows
    if verbose:
      print("[ladder] no incumbent -> k=2 feasibility rung", flush=True)
    plan = solve_assembly_problem_milestones(
        C, assembly, robots=robots, verbose=verbose, robot_ik=robot_ik,
        weight=max(weight, 5.0), wrench=wrench, wrench_aware_h=wrench_aware_h,
        handovers=handovers,
        allow_place=allow_place, bracing=bracing, geo_facts=geo_facts,
        sig_collapsed_h=sig_collapsed_h, deadline=slices[2], fill_window=2,
        lazy_fills=True)
    if plan is not None:
      U = sum(action_cost(a) for a in plan)
      best = plan
  if lns and best is not None and time.perf_counter() < t_end - 5.0:
    checker, grasp_table = _search_context(C, assembly, robots)
    gk = wrench_kind_arg(C, robots) if wrench else None
    if gk is not None:
      reset_wrench_cache()
    # CLI-SELECTED abstraction (respects --labeled-h/--collapsed-h/--sig-collapsed-h; safe
    # grasp-collapsed default) -> sig-press at R=4, NO explosion. Built once; used as the
    # differential landmark heuristic for every window (see _refill_window).
    h_cls = (SignatureCollapsedHeuristic if sig_collapsed_h
             else CollapsedLabeledHeuristic if collapsed_h
             else LabeledDisassemblyHeuristic if labeled_h
             else GraspCollapsedHeuristic)
    h_lab = h_cls(assembly, len(robots), grasp_table,
                  gripper_path=getattr(checker, "fact_gripper_path", checker.gripper_path),
                  verbose=False, gripper_kind=(gk if wrench_aware_h else None),
                  bracing=bracing)
    h0 = h_lab.value(get_initial_state(C, assembly))
    if U <= h0:
      if verbose:
        print(f"[ladder] PROVEN OPTIMAL: incumbent cost {U:g} == heuristic lower bound "
              f"h(s0) -- skipping LNS ({time.perf_counter() - t0:.1f}s)", flush=True)
      return best
    improved = _lns_improve(best, C, assembly, robots, checker, grasp_table, robot_ik,
                            max(weight, 5.0), 20000, verbose, h_lab, gripper_kind=gk,
                            handovers=handovers, allow_place=allow_place,
                            bracing=bracing, deadline=t_end, t0=t0,
                            object_unit=lns_object)
    ci = sum(action_cost(a) for a in improved)
    if ci < U:
      if verbose:
        print(f"[ladder] LNS: {U:g} -> {ci:g} at {time.perf_counter() - t0:.1f}s",
              flush=True)
      best, U = improved, ci
    elif verbose:
      print(f"[ladder] LNS: no improvement (fixpoint) at "
            f"{time.perf_counter() - t0:.1f}s", flush=True)
    if verbose:
      gap = U - h0
      print(f"[ladder] certificate: cost {U:g}, lower bound h(s0)={h0:g} -> "
            + ("PROVEN OPTIMAL" if gap <= 0 else f"within {gap:g} of optimal"),
            flush=True)
    return best                                           # k=n window IS the full search;
                                                          # no vestigial w=1 proof rung
  if rwa_proof and time.perf_counter() < t_end - 5.0:
    # RWA*-UNDER-BOUND proof rung: w=5 passes under the current bound -- each FIND
    # strictly tightens U (fast, measured ~30s where w=1 found nothing in 800s), and any
    # pass that EXHAUSTS is the optimality proof (exhaustion is weight-neutral: the bound
    # fixes the sublevel set, weight only orders it).
    while time.perf_counter() < t_end - 5.0:
      if verbose:
        print(f"[ladder] proof pass (w=5, PEA*"
              + (f", bound {U:g}" if U is not None else "")
              + f"), {t_end - time.perf_counter():.0f}s left", flush=True)
      res = solve_assembly_problem(
          C, assembly, robots=robots, verbose=verbose, robot_ik=robot_ik, weight=5.0,
          wrench=wrench, handovers=handovers, allow_place=allow_place, bracing=bracing,
          wrench_aware_h=wrench_aware_h,
          labeled_h=labeled_h, collapsed_h=collapsed_h,
          grasp_collapsed_h=grasp_collapsed_h, sig_collapsed_h=sig_collapsed_h,
          geo_facts=geo_facts, cost_bound=U, pea=True, deadline=t_end)
      if res is not None:
        best = res
        U = sum(action_cost(a) for a in res)
        if verbose:
          print(f"[ladder] proof pass improved to {U:g} at "
                f"{time.perf_counter() - t0:.1f}s", flush=True)
        continue
      if _LAST_SEARCH_OUTCOME == "exhausted" and U is not None:
        if verbose:
          print(f"[ladder] PROVEN OPTIMAL: cost {U:g} (bound exhausted, w=5) "
                f"in {time.perf_counter() - t0:.1f}s total")
      elif verbose:
        print(f"[ladder] proof pass ended on '{_LAST_SEARCH_OUTCOME}' -> best "
              f"{U if U is not None else float('nan'):g} in "
              f"{time.perf_counter() - t0:.1f}s total")
      break
    return best
  if time.perf_counter() < t_end - 5.0:
    if verbose:
      print(f"[ladder] proof rung (w=1, PEA*"
            + (f", bound {U:g}" if U is not None else "")
            + f"), {t_end - time.perf_counter():.0f}s left", flush=True)
    res = solve_assembly_problem(
        C, assembly, robots=robots, verbose=verbose, robot_ik=robot_ik, weight=1.0,
        wrench=wrench, handovers=handovers, allow_place=allow_place, bracing=bracing,
        wrench_aware_h=wrench_aware_h,
        labeled_h=labeled_h, collapsed_h=collapsed_h,
        grasp_collapsed_h=grasp_collapsed_h, sig_collapsed_h=sig_collapsed_h,
        geo_facts=geo_facts, cost_bound=U, pea=True, deadline=t_end)
    if res is not None:
      best = res
      U = sum(action_cost(a) for a in res)
      if verbose:
        print(f"[ladder] PROVEN OPTIMAL: cost {U:g} "
              f"in {time.perf_counter() - t0:.1f}s total (w=1 first pop under bound)")
    elif _LAST_SEARCH_OUTCOME == "exhausted" and U is not None:
      if verbose:
        print(f"[ladder] PROVEN OPTIMAL: cost {U:g} (bound exhausted) "
              f"in {time.perf_counter() - t0:.1f}s total")
    elif verbose:
      print(f"[ladder] proof rung ended on '{_LAST_SEARCH_OUTCOME}' -> "
            f"best cost {U if U is not None else float('nan'):g} WITHOUT proof "
            f"in {time.perf_counter() - t0:.1f}s total")
  return best


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
  gk = wrench_kind_arg(C, robots) if wrench else None
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
