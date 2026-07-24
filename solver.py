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
    for prefix, path in zip(self.slots, paths):
      template.addFile(path, namePrefix=prefix)
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
    # across same-kind arms.
    ex = tuple(sorted((self._robot_kind.get(g[0], g[0]),) + tuple(g[1:]) if len(g) == 3
                      else tuple(g) for g in extra_grasps)) if self.mixed \
        else tuple(sorted(tuple(g) for g in extra_grasps))
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
    # old sequential zip). extra_grasps entries may be (gpart, gid) or (robot, gpart, gid);
    # robot-less extras take any free slot (callers pass the robot when it matters).
    used = set()
    assigns = []                                           # (slot_idx, gpart, grasp_id)
    robot_grasps = [(c[1], c[2], c[3]) for p in state.bodies()
                    for c in state.supports.get(p, ()) if c[0] == "grasp"]
    extras = [g if len(g) == 3 else (None, g[0], g[1]) for g in extra_grasps]

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
      grasp_world = _pose_compose(np.asarray(part_poses[gpart], dtype=float),
                                  grasp_pose(assembly, gpart, grasp_id))
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
      # a FLOATING EE has no arm at all: no base_link frame, no kinematic chain -> analytic
      # IK / reach bounds are meaningless for it (robot_arm=None disables every ssik path).
      if (r + "_ur_base_link") in names:
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
    self.seed_restart_cap = 2                             # random restarts left after seeds (see _solve_ik)
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
    self.pre_insertion_pull = 1.0   # qItself pull weight tying T_pre to the skill witness (<1e1)
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
    self.strict_rejects = 0                               # diagnostics: dirty witnesses vetoed
    # DEBUG: set env PEN_VIEW_ACTION='pick a1 obj_1 0' to open a BLOCKING C.view(True) on the
    # FIRST penetration check of that exact action (space-joined tuple) -- prints the colliding
    # frame pairs and shows the skill scene at the checked witness. For inspecting whether a
    # --strict-collisions witness rejection is real. Off unless the env var is set.
    self._pen_view_action = os.environ.get("PEN_VIEW_ACTION")
    self._pen_viewed = False
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
    kind = action[0]
    if kind == "release":
      return True, None                                   # no new reach
    if kind == "inhand_reorient":
      # in-place skill: the arm keeps its config; the part rotates IN the gripper (no IK
      # target of its own -- downstream actions solve IK against the new grasp).
      return True, None
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
        return hit
    if kind == "pick":
      res = self._ik_pick(action, s0, s1, assembly, seed)
    elif kind == "place":
      res = self._ik_place(action, s0, s1, assembly, seed)
    elif kind == "assemble":
      res = self._ik_assemble(action, s0, s1, assembly, seed)
    else:                                                 # add_grasp / handover: 2nd arm
      res = self._ik_add_grasp(action, s0, s1, assembly, seed)
    if self._opt_pull is None:
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

  def _pen_ok(self, C, q):
    # --strict-collisions gate: total penetration of the skill scene at witness q (the same
    # contact flags the komo's collision objective saw -- target-part exclusions included).
    if self.penetration_reject is None or C is None or q is None:
      return True
    saved = C.getJointState()
    C.setJointState(np.asarray(q, dtype=float))
    pen = float(C.getCollisionsTotalPenetration())
    reject = pen > self.penetration_reject

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

  def _solve_ik(self, komo, restarts=12, seeds=None, C=None):
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
      if r["feasible"] and self._pen_ok(C, komo.getPath()[-1]):
        self._last_res = r
        return True, komo.getPath()[-1]
    if seeds:
      # the analytic seed set ENUMERATES every exact pose branch (up to 8), so random restarts
      # can only re-derive branches already tried -- their residual value is exploring the eq
      # TOLERANCE slack (a nearby collision-free near-pose). A few tries cover that; the full
      # budget is for UNSEEDED solves where restarts must find the branches themselves.
      restarts = min(restarts, self.seed_restart_cap)
    r = ry.NLP_Solver(komo.nlp(), verbose=0).solve().dict()
    if r["feasible"] and self._pen_ok(C, komo.getPath()[-1]):
      self._last_res = r
      return True, komo.getPath()[-1]
    for _ in range(restarts):
      komo.initRandom()
      r = ry.NLP_Solver(komo.nlp(), verbose=0).solve().dict()
      if r["feasible"] and self._pen_ok(C, komo.getPath()[-1]):
        self._last_res = r
        return True, komo.getPath()[-1]
    self._last_res = r
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
    # each attached arm's 6 joints overwritten with one of its solutions (cartesian product,
    # closest-to-current-config first, capped -- the product is the complete enumeration of
    # the exact pose branches when it fits under the cap).
    import itertools
    q0 = np.asarray(C.getJointState(), dtype=float)
    names = C.getJointNames()
    idx = {r: [i for i, n in enumerate(names) if n.startswith(r + "_")] for r in arm_seeds}
    if any(len(ix) != 6 for ix in idx.values()):
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
    if (not arm_seeds or any(len(ix) != 6 for ix in idx.values())
        or any(not s for s in arm_seeds.values())):
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
    return False, [q for _, _, q, _ in scored[:cap]]

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
      if len(lo) != 6:
        # UNRESTRICTED scene (e.g. an assemble reject probe): select this robot's 6 arm limits
        # by joint-name prefix -- without this the length mismatch silently skipped the filter.
        idx = [i for i, n in enumerate(C.getJointNames()) if n.startswith(robot + "_")]
        if len(idx) == 6:
          lo, hi = lo[idx], hi[idx]
        else:
          lo = hi = None
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
      ee_target = ee_target.copy()
      ee_target[:3] = ee_target[:3] - backoff * (_quat_to_R(ee_target[3:]) @ np.array([0.0, 0.0, 1.0]))

    robot_ee_name = self.gripper_frame[robot]
    part_name = part

    # the part sits on the table (FIXED target) -> the sound reach-sphere bound applies for
    # any arm, calibrated or not (the analytic reject below subsumes it when available).
    part_pose = np.asarray(C.getFrame(part_name).getPose(), dtype=float)
    if self._reach_reject(C, robot, _pose_compose(part_pose, ee_target)[:3]):
      return False, None

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
            [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [self.col_weight], [-0.0]
        )
      komo.addControlObjective([], 0, 1e-1)
      komo.addControlObjective([], 1, 1e-1)
      self._add_pull(komo, C, robot)                      # witness-opt pull (no-op in search)
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
    if self._analytic_reject_ok(robot) and seeds:
      reject, ranked = self._probe_branch_combos(C, {robot: seeds})
      if reject:
        return False, None                                # every exact branch deeply colliding
      seeds = ranked or seeds                             # shallowest-collision branches first
    elif self.reach_precheck and self._reach_precheck_reject(action, C, build(False)):
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
    if self.single_arm_ik:
      witness = self._expand_witness(C, witness, full_names)  # restricted -> full joint vector
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
      ee_target = ee_target.copy()
      ee_target[:3] = ee_target[:3] - backoff * (_quat_to_R(ee_target[3:]) @ np.array([0.0, 0.0, 1.0]))
    set_part_contact(C, gpart, 0)                        # the new gripper is meant to touch gpart

    def build(collisions):
      komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=collisions)
      if collisions:
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [self.col_weight], [-0.0])
      komo.addControlObjective([], 0, 1e-1)
      komo.addControlObjective([], 1, 1e-1)
      self._add_pull(komo, C, robot)                      # witness-opt pull (no-op in search)
      komo.addObjective([1, 1], ry.FS.positionRel, [self.gripper_frame[robot], gpart],
                        ry.OT.eq, [1e1], ee_target[:3])
      komo.addObjective([1, 1], ry.FS.quaternionRel, [self.gripper_frame[robot], gpart],
                        ry.OT.eq, [1e1], ee_target[3:])
      self._add_grasp_maintenance(komo, s0, assembly)     # existing braces stay at their grasps
      return komo

    if self.reach_precheck and self._reach_precheck_reject(action, C, build(False)):
      return False, None                                  # clear reach failure (relaxed solve)

    komo = build(True)
    feasible, witness = self._solve_ik(          # `warm` (full config) couples pre-add-grasp to
        komo, seeds=([np.asarray(warm, dtype=float)] if warm is not None else None),  # the grasp
        C=C)
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
      target = target.copy()
      target[2] = target[2] + backoff

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
      # object-pose constraint (like assemble, not an EE pose): the group's reference part
      # must reach its canonical table pose, axis-aligned.
      komo.addObjective([1, 1], ry.FS.position, [ref], ry.OT.eq, [1e1], target[:3])
      komo.addObjective([1, 1], ry.FS.quaternion, [ref], ry.OT.eq, [1e1], target[3:])
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
    for rr, ee in arm_targets.items():
      if self._analytic_reject_ok(rr):
        sl = self._analytic_seeds(C, rr, ee, None)
        if not sl:
          all_calibrated = False                          # EMPTY exact set: borderline, not a
          continue                                        # proof -- precheck decides (see pick)
        arm_seeds[rr] = sl
      else:
        all_calibrated = False
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
      if self.reach_precheck and self._reach_precheck_reject(action, C, build(False)):
        return False, None                                # clear reach failure (relaxed solve)

    komo = build(True)
    if warm is not None:                                   # couple pre-place to the place witness
      wseed = self._restrict_seed(warm, C, full_names) if full_names is not None \
          else np.asarray(warm, dtype=float)
      seeds = [wseed] + list(seeds or [])
    feasible, witness = self._solve_ik(komo, seeds=seeds, C=C)
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
      komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [self.col_weight], [-0.0])
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

  def _ik_assemble(self, action, s0, s1, assembly, seed, bias_axes=None, backoff=0.0, warm=None):
    self._cur_action = action                             # for PEN_VIEW_ACTION debug
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
      d = assembly.mate_axis(part)
      if d is not None:
        rel_pos = rel_pos - backoff * np.asarray(d, dtype=float)

    # the mate seats the moving group flush against the target, so those contacts are intended
    # -- exclude the WHOLE moving group from collision (a single cube, or every cube of a
    # module: a stacked plate touches the target plate on all its cubes). Otherwise the
    # intended flush seating reads as penetration and IK fails. Arm-vs-rest collision stays on;
    # the cheap floating-EE check already covered the moved geometry vs the scene.
    moving = next((g for g in s0.subassemblies() if part in g), (part,))
    for mp in moving:
      set_part_contact(C, mp, 0)

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
      arm_seeds = {}
      for rr, ee in arm_targets.items():
        if not self._analytic_reject_ok(rr):
          reachable_proven = False                        # uncalibrated arm: precheck decides
          continue
        sl = self._analytic_seeds(C, rr, ee, None)
        if not sl:
          reachable_proven = False                        # EMPTY exact set: borderline, not a
          continue                                        # proof -- precheck decides (see pick)
        arm_seeds[rr] = sl
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

    if self.reach_precheck and not reachable_proven:
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
      self._add_pull(komo, C, robot)                    # witness-opt pull (no-op in search)
      self._add_grasp_maintenance(komo, s0, assembly)     # braces pinned at their grasps
      feasible, witness = self._solve_ik(              # `warm` (e.g. the mated witness for a
          komo, restarts=0,                            # pre-insertion goto) warm-starts the solve
          seeds=([warm] if warm is not None else None),  # so it settles in the SAME IK basin
          C=C)
      if feasible:
        break
    if not feasible:                                      # fallback: unbiased random restarts
      komo = self._assemble_komo(C, part_name, onto, rel_pos, rel_quat, None)
      self._add_pull(komo, C, robot)                    # witness-opt pull (no-op in search)
      self._add_grasp_maintenance(komo, s0, assembly)
      feasible, witness = self._solve_ik(
          komo, restarts=6, seeds=([warm] if warm is not None else None), C=C)
    if not feasible:
      self._log_failure(action, C, komo)

    if self.debug_view:
        print("assemble feasible:", feasible)
        komo.view(True, f"assemble IK: {robot} mates {part} -> {onto}")

    return feasible, witness

  def pre_skill_witness(self, action, s0, s1, assembly, skill_witness=None):
    """EXPORT/REPLAY helper for --pre-pick/--pre-place/--pre-insertion: the PRE-skill GOTO
    keyframe -- the skill witness backed off along its approach axis (pick: -z_EE retract;
    place: +z lift; assemble: -mate_axis). Returns the full-config witness, or None (skill not
    split / IK failed -- caller falls back to the skill witness). Solved on demand; NOT used by
    the search (whose witness is always the real skill pose).

    COUPLED to the skill witness: the pre-skill solve is WARM-STARTED from `skill_witness` and
    given a full-config qItself pull toward it, so the goto config differs from the skill ONLY
    by the actor's delta backoff. Without this, the two independent solves can land in different
    IK basins (a whole wrist revolution measured for assemble) -> a long, possibly-colliding
    'skill' motion between the two keyframes."""
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
    try:
      if kind == "pick":
        ok, w = self._ik_pick(action, s0, s1, assembly, None, backoff=delta, warm=warm)
      elif kind == "place":
        ok, w = self._ik_place(action, s0, s1, assembly, None, backoff=delta, warm=warm)
      elif kind == "add_grasp":
        ok, w = self._ik_add_grasp(action, s0, s1, assembly, None, backoff=delta, warm=warm)
      else:
        ok, w = self._ik_assemble(action, s0, s1, assembly, None, backoff=delta, warm=warm)
    finally:
      self._opt_pull = None
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


def optimize_witnesses(C, assembly, plan, robots, robot_ik, cost="joint",
                       pull_weight=1.0, ee_rot_weight=0.3, sweeps=2, verbose=True,
                       gripper_kind=None, handovers=True, allow_place=True, bracing=False,
                       grasp_table=None, clear_weight=0.0, clear_margin=0.4,
                       robot_collisions=False):
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

  wit = [None] * len(plan)
  for i, a in enumerate(plan):                            # baseline = the search's witnesses
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
        bracing=bracing, grasp_table=grasp_table)  # warm start (no clearance filter)
    cands = {}
    for i, a in enumerate(plan):
      if wit[i] is None:
        continue
      cs = robot_ik.candidate_witnesses(a, states[i], states[i + 1], assembly)
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
    self._mixed = isinstance(self._gk, dict) and _MIXED_WRENCH_H
    if isinstance(self._gk, dict) and not _MIXED_WRENCH_H:
      self._gk = None                                     # fast fallback: topological facts
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


_MIXED_WRENCH_H = False                                    # --mixed-wrench-h: per-kind wrench
                                                          # facts under mixed EEs (precise but
                                                          # SLOW build); default False = the
                                                          # fast topological fallback.


_FILL_TOUCH_ALL = False                                    # --fill-touch-all (debug): let a
                                                          # fill act on ANY loose part, not
                                                          # just ideal+{p} -- so it can move a
                                                          # blocking scatter part aside. Big
                                                          # search blow-up; a diagnosis knob.


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


def set_mixed_wrench_h(on=True):
  global _MIXED_WRENCH_H
  _MIXED_WRENCH_H = bool(on)


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
               labeled_loose=False, dist_cap=None):
    import heapq
    import time as _time

    t0 = _time.perf_counter()
    self.assembly = assembly
    self.R = n_robots
    self.misses = 0
    self._bracing = bracing                                # model the add_grasp (co-grasp) edge
    self._clearance = clearance
    self._gk = gripper_kind
    self._mixed = isinstance(self._gk, dict) and _MIXED_WRENCH_H
    if isinstance(self._gk, dict) and not _MIXED_WRENCH_H:
      self._gk = None                                     # fast fallback: topological facts
    self._kinds_of = getattr(grasp_table, "kinds_of", None)
    self._all_kinds = tuple(sorted(set(getattr(grasp_table, "robot_kinds", {}).values())))
    self._collapse = bool(collapse_grasps)                 # see GraspCollapsedHeuristic
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
        yield (tsub(ideal, q), G2, face,
               tadd(L, (q, gq) if self._labeled_loose else q)), 1.0
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
          for q in objs:
            if q in ideal or any(q == p for p, _ in L):
              continue
            for qg in self.single[q]:
              yield (ideal, G, face, tadd(L, (q, qg))), 1.0
      else:
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
        raise RuntimeError(f"collapsed labeled abstraction exceeded {max_states} states")
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
      tag = "sig-h" if grasp_classes is not None else "collapsed-h"
      print(f"[{tag}] {self.n_states} coreachable states, {self.n_pick_facts} pick "
            f"facts, built in {self.build_time:.2f}s")

  def abstract(self, state):
    main = next(g for g in state.subassemblies() if self.assembly.base in g)
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
                        for c in state.supports.get(p, ()) if c[0] == "grasp"}))
    else:
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
        gripper_kind=(None if (isinstance(kwargs.get("gripper_kind"), dict)
                               and not _MIXED_WRENCH_H) else kwargs.get("gripper_kind")),
        gripper_path=kwargs.get("gripper_path"),
        clearance=kwargs.get("clearance", 0.02))
    if kwargs.get("verbose"):
      summary = ", ".join(f"{q}:{len(set(m.values()))}/{len(m)}"
                          for q, m in sorted(classes.items()))
      print(f"[sig-h] components=({','.join(_SIG_COMPONENTS)}); "
            f"classes/raw grasps per part: {summary}")
    kwargs["grasp_classes"] = classes
    kwargs["labeled_loose"] = True
    super().__init__(assembly, n_robots, grasp_table, **kwargs)


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


def _search_context(C, assembly, robots):
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
  kinds = robot_gripper_kinds(C, robots)
  distinct = sorted(set(kinds.values()))
  ee_path = {k: cheap_ee_path(is_ur5, floating, k == "vacuum") for k in distinct}
  fact_path = {k: os.path.join(os.path.dirname(__file__), UR10_PATHS[(True, k == "vacuum")])
               for k in distinct}
  if len(distinct) == 1:                                   # UNIFORM: byte-identical to before
    k = distinct[0]
    checker = CollisionChecker(assembly, robots, gripper_path=ee_path[k])
    # plain-EE variant for the labeled-h pick facts (clutter-free scenes; see the h_lab
    # construction in solve_assembly_problem for why the wrist is excluded there).
    checker.fact_gripper_path = fact_path[k]
    return checker, precompute_grasps(assembly, gripper_path=ee_path[k])
  # MIXED END EFFECTORS: per-robot cheap-EE slots; the grasp table is the UNION over kinds
  # with a per-grasp kind MASK (a grasp enters kind k's set iff k's EE stands there
  # collision-free vs the subassembly). Successor generators gate (robot, grasp) pairs on
  # the mask; heuristic facts OR over a grasp's mask kinds (admissible).
  checker = CollisionChecker(assembly, robots, gripper_path=ee_path[distinct[0]],
                             robot_gripper_paths={r: ee_path[kinds[r]] for r in robots})
  checker.fact_gripper_path = fact_path                    # dict per kind (mixed)
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
        if k == "vacuum" and rolls > 1 and not dense and g % rolls != 0:
          continue  # the pen is roll-symmetric: rolls of one face are IK-identical
                    # duplicates for the vacuum -- and vacuum binds are the expensive
                    # ones (no analytic reject) -> keep roll 0 only in ITS mask. The
                    # two-finger keeps every roll (its pads genuinely differ per roll).
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
    self.tie = itertools.count()
    self.pq = []
    self.claimed = {}
    self.came_from = {start: (None, None)}
    self.children = {}
    self.alts = {}
    self.expanded = 0
    self.budget_hit = False
    h0 = h_fn(start)
    self.pq.append((self._f(0.0, h0), h0, 0.0, next(self.tie), start, None, None))

  def _f(self, g, hval):
    import math
    return hval if math.isinf(self.weight) else g + self.weight * hval

  def _succ(self, s):
    hit = self._succ_cache.get(s)
    if hit is None:
      hit = self._succ_cache[s] = list(self._succ_raw(s))
    return hit

  def search(self, max_nodes, deadline=None):
    # run until a goal state is claimed (returns the Plan) or the frontier empties (None).
    import heapq
    import time
    while self.pq:
      entry = heapq.heappop(self.pq)
      _f, _h, g, _, s, prev, action = entry
      if s in self.claimed:
        self.alts.setdefault(s, []).append(entry)         # keep as an alternate route
        continue
      if prev is not None:
        if prev not in self.claimed or (prev, action) in self.blacklist:
          continue                                        # stale: parent gone / edge deleted
        g_true = self.claimed[prev] + self.cost_fn(action)
        if g_true != g:                                   # parent re-claimed at another g
          hs = self.h_fn(s)
          heapq.heappush(self.pq, (self._f(g_true, hs), hs, g_true, next(self.tie),
                                   s, prev, action))
          continue
        if self.cheap_gate is not None and not self.cheap_gate(prev, action, s):
          self.blacklist.add((prev, action))              # cheap-infeasible: dead forever
          continue
      self.claimed[s] = g
      if prev is not None:
        self.came_from[s] = (prev, action)
        self.children.setdefault(prev, []).append(s)
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
          self.alts.setdefault(ns, []).append(entry)      # alternate for a future repair
        else:
          heapq.heappush(self.pq, entry)
    return None

  def delete_edge(self, s, a, ns):
    # (s, a) was just blacklisted (caller added it). If it is the tree edge of ns, un-claim
    # the SUBTREE rooted at ns (their g relied on the deleted edge) and re-inject each
    # un-claimed node's alternates; everything else is untouched. Queued entries THROUGH the
    # subtree are invalidated lazily by the stale checks at pop.
    import heapq
    if self.came_from.get(ns, (None, None)) != (s, a):
      return                                              # not a tree edge: nothing to repair
    stack = [ns]
    while stack:
      x = stack.pop()
      if x not in self.claimed:
        continue
      del self.claimed[x]
      self.came_from.pop(x, None)
      stack.extend(self.children.pop(x, ()))
      for alt in self.alts.pop(x, ()):
        heapq.heappush(self.pq, alt)


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
    if verbose:
      print("[lazy] initial state provably cannot reach the goal (abstraction facts) -> "
            "instance INFEASIBLE as configured.")
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
                                   pathmax=False, corridor_h=False):
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
      weight=max(weight, 5.0), wrench=wrench, handovers=handovers,
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
        allow_place=allow_place, bracing=bracing)
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
        geo_facts=geo_facts, sig_collapsed_h=sig_collapsed_h, beam_width=beam_width)
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
      pathmax=pathmax, h_dist_cap=(U if corridor_h else None))
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


def _add_one_part(start, p, ideal, assembly, robots, checker, grasp_table, robot_ik,
                  weight, max_nodes, verbose, gripper_kind=None, handovers=True,
                  allow_place=True, bracing=False, grasp_cap=None, h_extra=None,
                  part_grasp_allow=None, brace_allow=None, deadline=None,
                  lazy_fill=False, cost_bound=None, forbid_assemble=None, settle_ref=None):
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
  settle = settle_ref if settle_ref is not None else assembly.base

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
    return _part_mated(s, p) and loose == 0 and holders <= 1

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
    if h_extra is not None:
      # The labeled abstraction (FULL-assembly cost-to-go) keeps exactly two roles here:
      #   * h=inf PRUNES states the full assembly cannot be finished from (sound);
      #   * its VARIATION tie-breaks locally-equal siblings -- the resting-FACE and
      #     hold/brace pricing the counting h is blind to.
      # It must NOT join via max(): for a MODULE fill it is a large CONSTANT (module-
      # internal progress is invisible to the base-ideal abstraction), and max(local, he)
      # flattened h -> the mini search degenerated to uniform-cost and exhausted max_nodes
      # on the chair's back+posts module under mixed EEs (a 4-action solution in plain
      # sight, "fill-in failed" every time). As an EPSILON term (<= 0.1, below one local
      # step) it orders siblings without ever drowning the local gradient.
      he = h_extra(s)
      if he == float("inf"):
        return float("inf")
      hv = hv + min(he, 1e3) * 1e-4
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
      if a[2] not in relevant and not _FILL_TOUCH_ALL:   # a[2] is the part each action acts on
        # RELEASE of an INHERITED grasp is always legal: the previous fill may end holding
        # the assembly by any part (<=1 holder persists), and the grasped part is then
        # outside this fill's `relevant` -- without this exemption the inherited holder
        # STRANDS an arm for the whole fill (module fills with 2 usable arms died on
        # exactly this; found via captured-state replay, 2026-07-24). Releasing frees a
        # resource and manipulates nothing.
        if a[0] != "release":
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

  if lazy_fill and robot_ik is not None:
    # LAZY-VALIDATED fill (LazySP hybrid, mirroring the global lazy search): the fill's
    # inner search pays only the CHEAP gate at pop; real IK runs on CANDIDATE subplan
    # edges only, failures are blacklisted and the (tiny) search reruns. Imports the
    # measured lazy economics (61 validations vs 19,515 eager checks on the cross) into
    # the fills -- the 2x2x3 module-fill bottleneck (325-373s/fill at R=4-real was
    # dominated by failed IK binds at pop).
    bl = set()
    ok_cache = set()                                      # (state, action) IK-validated

    def succ_lazy(s):
      for a, ns in succ_fn(s):
        if (s, a) not in bl:
          yield a, ns

    def cheap_only(prev, act, ns):
      return cheap_feasible(act, prev, ns, assembly, checker)

    for _round in range(200):
      subplan = _astar(start, goal_fn=goal_fn, succ_fn=succ_lazy, cost_fn=action_cost,
                       h_fn=h_fn, cheap_fn=None, edge_check_fn=cheap_only,
                       weight=weight, max_nodes=max_nodes, verbose=False,
                       tag=f"add {p} lazy", deadline=deadline, cost_bound=cost_bound)
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
            # VALIDATE-ALL per round: binds are checks, not state changes, so the suffix
            # states stay well-defined -- keep replaying and blacklist EVERY failing edge
            # this round. Collapses rounds from #failures to #failure-layers (measured:
            # first-failure needed 130 rounds/145s on the obj_11 fill). The global lazy
            # search measured validate-all HARMFUL -- there the doomed-suffix IK cost
            # dominates; in a tiny fill the re-search dominates, inverting the trade.
            bl.add((s, a))
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
        return subplan, s
    return None, None

  subplan = _astar(start, goal_fn=goal_fn, succ_fn=succ_fn, cost_fn=action_cost, h_fn=h_fn,
                   cheap_fn=None, edge_check_fn=edge_check_fn,
                   weight=weight, max_nodes=max_nodes, verbose=False,
                   tag=f"add {p}", deadline=deadline, cost_bound=cost_bound)
  if subplan is None:
    return None, None
  s = start
  for a in subplan:
    s = next(ns for act, ns in successors(s, assembly, robots, grasp_table=grasp_table,
                                          gripper_kind=gripper_kind, handovers=handovers,
                                          allow_place=allow_place, bracing=bracing)
             if act == a)
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
               lazy_fill=False, cost_bound=None):
  # WIDTH-k milestone fill (adaptive-width completeness net): add ALL parts in `ps` in ONE
  # joint mini-search -- for plans that must INTERLEAVE several parts' glue, which the
  # width-1 fill cannot express (the milestone commitment's documented gap). Invoked with
  # growing k only after every smaller width failed at the node, BEFORE backjumping:
  # escalation on proof of local failure; k = n is the flat exact search, so completeness
  # is recovered in the limit. No hint phases (hints are per-part); one full phase.
  ps = tuple(ps)
  relevant = set(ideal) | set(ps)

  def goal_fn(s):
    return (all(_part_mated(s, p) for p in ps)
            and _num_loose_held(s, assembly) == 0
            and _assembly_holders(s, assembly) <= 1)

  def h_fn(s):
    left = sum(0 if _part_mated(s, p) else 2 for p in ps)
    return left + _num_loose_held(s, assembly) + max(0, _assembly_holders(s, assembly) - 1)

  def succ_fn(s):
    for a, ns in successors(s, assembly, robots, grasp_table=grasp_table,
                            gripper_kind=gripper_kind, handovers=handovers,
                            allow_place=allow_place, bracing=bracing):
      if a[2] in relevant or a[0] == "release":           # inherited-grasp release always
        yield a, ns                                       # legal (see _add_one_part)

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

    def succ_lazy(s):
      for a, ns in succ_fn(s):
        if (s, a) not in bl:
          yield a, ns

    def cheap_only(prev, act, ns):
      return cheap_feasible(act, prev, ns, assembly, checker)

    for _round in range(200):
      subplan = _astar(start, goal_fn=goal_fn, succ_fn=succ_lazy, cost_fn=action_cost,
                       h_fn=h_fn, cheap_fn=None, edge_check_fn=cheap_only, weight=weight,
                       max_nodes=max_nodes, verbose=False,
                       tag="add " + "+".join(ps) + " lazy", deadline=deadline,
                       cost_bound=cost_bound)
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
                                      allow_place=True, bracing=False, **exact_kw):
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
  gk = wrench_kind_arg(C, robots) if wrench else None
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
    if sub is None and bracing:
      # braced retry (e.g. a car wheel under insertion noise: the mate needs a second grasp);
      # unbraced-first keeps the cheap path cheap (add_grasp edges inflate the mini plateau).
      sub, s2 = _add_one_part(s, p, ideal, assembly, robots, checker, grasp_table, robot_ik,
                              weight, max_nodes, verbose, gripper_kind=gk, handovers=handovers,
                              allow_place=allow_place, bracing=True)
    if sub is None:
      if verbose:
        print(f"[optimistic] could not add {p} in isolation -> falling back to exact")
      if not fallback:
        return None
      return solve_assembly_problem(C, assembly, robots=robots, verbose=verbose,
                                    robot_ik=robot_ik, weight=weight, max_nodes=max_nodes,
                                    wrench=wrench, handovers=handovers,
                                    allow_place=allow_place, bracing=bracing, **exact_kw)
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


def solve_assembly_problem_milestones(C, assembly, robots=None, verbose=True, robot_ik=None,
                                      weight=5.0, max_nodes=20000, wrench=False,
                                      handovers=True, allow_place=True, bracing=False,
                                      max_repairs=100, geo_facts=False,
                                      sig_collapsed_h=False, deadline=None,
                                      fill_window=2, lazy_fills=True,
                                      cost_bound=None):
  """MILESTONE (Dogar-style, abstraction-guided) planner. Two levels:

    LEVEL 1: the assembly SEQUENCE. The labeled abstraction's backward Dijkstra (the same
      dist table the exact search uses as its heuristic -- itself an assembly-by-disassembly
      solve of the relaxed model, wrench facts included) is used as an ORACLE: among the
      precedence-addable next parts, try them in order of the abstract cost-to-go AFTER
      adding them (min over stable rest faces). Grasps/reach are ASSUMED feasible here.
    LEVEL 2: the manipulation glue. Each chosen milestone is filled in by _add_one_part's
      small complete search (pick/handover/brace/mate/set-down), from the previous
      milestone's settled state.
    REPAIR: if a milestone's fill-in fails, that (assembled-set, part) transition is
      BLACKLISTED and the next-best candidate is tried -- repair happens at milestone
      granularity (dozens of nodes) instead of the exact search's per-edge flooding.

  Separate from the exact search by design (shares only _add_one_part + the heuristic
  class). Complete up to milestone commitment (like the optimistic solver, an interleaving
  that must overlap two parts' glue is out of scope); vs the optimistic solver it SEARCHES
  the part order (wrench-informed) instead of committing to assembly_order, and supports
  bracing in the fill-in. Returns the action list, or None (no silent exact fallback)."""
  import time

  t0 = time.perf_counter()
  if robots is None:
    robots = get_robots(C)
  checker, grasp_table = _search_context(C, assembly, robots)
  gk = wrench_kind_arg(C, robots) if wrench else None
  if _problem._USEFUL_FLAG:
    _problem.set_useful_grasps(compute_useful_grasps(assembly, grasp_table, gk))
    if verbose:
      nU = {p: len(v) for p, v in (_problem._USEFUL_GRASPS or {}).items()}
      print(f"[useful-grasps] mate-viable per part: {nU}")
  if gk is not None:
    reset_wrench_cache()
  # the milestone oracle: GraspCollapsedHeuristic == CollapsedLabeledHeuristic when dense
  # grasps are off, and the label-collapsed variant when they are on -- always the right one.
  h_cls = SignatureCollapsedHeuristic if sig_collapsed_h else GraspCollapsedHeuristic
  h_lab = h_cls(assembly, len(robots), grasp_table,
                                  gripper_path=getattr(checker, "fact_gripper_path",
                                                       checker.gripper_path),
                                  verbose=verbose, gripper_kind=gk, bracing=bracing,
                                  geo_checker=checker if geo_facts else None)
  Hf = h_lab.H

  def oracle(ideal_set, q):
    # abstract cost-to-go once `q` is mated and the grown assembly rests SETTLED on some
    # stable face, nothing held (the state _add_one_part chains through). inf => the
    # abstraction cannot finish from there (or never reached it backward) -> try last.
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
    gqs, unbraced_ok, braces = set(), False, set()
    for k, ok in h_lab._wfacts.items():
      if ok and k[0] == "A" and k[1] == ideal_key and k[4] == q:
        gqs.update(h_lab._svgroups.get(q, {}).get(k[5], (k[5],)))
        unbraced_ok |= len(k[2]) == 0
        for bp, bg in k[2]:                               # G2 = (part, collapsed grasp) braces
          for rg in h_lab._svgroups.get(bp, {}).get(bg, (bg,)):
            braces.add((bp, rg))
    return (gqs or None), unbraced_ok, (braces or None)

  def fill(s, q, ideal, fb=None, forbid=None, settle=None):
    allow, unbraced_ok, ballow = hints(_subasm_key(ideal), q)
    dense = _problem._DENSE_GRASPS
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
    for cap, br, ma, ba in phases:
      sub, s2 = _add_one_part(s, q, ideal, assembly, robots, checker, grasp_table, robot_ik,
                              weight, max_nodes, verbose, gripper_kind=gk,
                              handovers=handovers, allow_place=allow_place, bracing=br,
                              grasp_cap=cap, h_extra=h_lab.value, part_grasp_allow=ma,
                              brace_allow=ba, deadline=deadline, lazy_fill=lazy_fills,
                              cost_bound=fb, forbid_assemble=forbid, settle_ref=settle)
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

  def fill_module(s, root, ideal, fb=None):
    seq = sorted(modules[root] - {root},
                 key=lambda m: (len(_mate_closure(assembly, m)), _part_index(m)))
    parts_plan, s_cur, built = [], s, {root}
    for m in seq:
      # settle=root: the module may end each sub-fill HELD by one carrier (built in the
      # air); the base-centric default would count the held module as loose and force a
      # table set-down -- IK-infeasible for real arms on e.g. the chair's back+posts module.
      sub, s_cur = fill(s_cur, m, built, fb=fb, forbid={root}, settle=root)
      if sub is None:
        return None, None
      parts_plan.extend(sub)
      if fb is not None:
        fb -= sum(action_cost(a) for a in sub)            # global bound: spend as we go
      built.add(m)
    sub, s_end = fill(s_cur, root, set(ideal) | set(modules[root]), fb=fb)   # the JOIN
    if sub is None:
      return None, None
    return parts_plan + sub, s_end

  # chronological BACKJUMPING over milestones: each stack entry is a settled checkpoint
  # (state, assembled set, plan length so far, and the (parent_key, part) decision that led
  # here). A node with no workable candidate pops -- blacklisting the decision that led to it
  # -- and the search reroutes from the parent.
  stack = [(get_initial_state(C, assembly), frozenset({assembly.base}), 0, None)]
  blacklist = set()                                       # (ideal_key, part) fill-ins that failed
  repairs = 0
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
            blacklist.add(via)
            repairs += 1
          continue
        final = final + sub
        s_final = s2
      if verbose:
        print(f"[milestones] complete: {len(final)} actions in "
              f"{time.perf_counter() - t0:.2f}s ({repairs} repairs)")
      return final
    ideal_key = _subasm_key(ideal)
    cands = [q for q in objs - ideal
             if (ideal_key, q) not in blacklist
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
        stack.pop()                                       # prefix already >= bound
        continue
    for q in cands:
      tq = time.perf_counter()
      if q in connect:
        sub, s2 = fill_module(s, q, ideal, fb=fb)
      else:
        sub, s2 = fill(s, q, ideal, fb=fb)
      grown = (ideal | modules[q]) if q in connect else (ideal | {q})
      if sub is not None:
        del plan[plen:]
        plan.extend(sub)
        stack.append((s2, grown, len(plan), (ideal_key, q)))
        progressed = True
        if verbose:
          what = (f"module {q} ({len(modules[q])} parts, incl. join)"
                  if q in connect else q)
          print(f"[milestones] added {what} in {len(sub)} actions "
                f"({time.perf_counter() - tq:.1f}s, {len(plan)} total)")
        break
      blacklist.add((ideal_key, q))
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
          if fs not in seen_sets and (ideal_key, tuple(sorted(t))) not in blacklist:
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
          sub, s2 = _add_parts(s, t, ideal, assembly, robots, checker, grasp_table,
                               robot_ik, weight, max_nodes, gripper_kind=gk,
                               handovers=handovers, allow_place=allow_place,
                               bracing=bracing, deadline=deadline,
                               lazy_fill=lazy_fills, cost_bound=fb)
          if sub is not None:
            del plan[plen:]
            plan.extend(sub)
            stack.append((s2, ideal | set(t), len(plan), (ideal_key, tuple(sorted(t)))))
            progressed = True
            if verbose:
              print(f"[milestones] added WINDOW {t} in {len(sub)} actions "
                    f"({time.perf_counter() - tq:.1f}s, {len(plan)} total)")
            break
          blacklist.add((ideal_key, tuple(sorted(t))))
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
        blacklist.add(via)
        repairs += 1
        if verbose:
          print(f"[milestones] dead end at {sorted(_i)} -> backjump, blacklist {via} "
                f"({repairs} repairs)")
        if repairs > max_repairs:
          if verbose:
            print("[milestones] repair budget exhausted")
          return None
  if verbose:
    print(f"[milestones] search space exhausted with no plan "
          f"in {time.perf_counter() - t0:.2f}s ({repairs} repairs)")
  return None


def solve_assembly_problem_milestones_anytime(C, assembly, robots=None, time_budget=300.0,
                                              verbose=True, robot_ik=None, weight=5.0,
                                              wrench=False, handovers=True,
                                              allow_place=True, bracing=False,
                                              geo_facts=False, sig_collapsed_h=False,
                                              labeled_h=False, collapsed_h=False,
                                              grasp_collapsed_h=False, fill_window=2):
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
      weight=max(weight, 5.0), wrench=wrench, handovers=handovers,
      allow_place=allow_place, bracing=bracing, geo_facts=geo_facts,
      sig_collapsed_h=sig_collapsed_h, deadline=t_mile, fill_window=fill_window)
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
                                  lns=False, rwa_proof=False, lns_object=True):
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
        weight=max(weight, 5.0), wrench=wrench, handovers=handovers,
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
        weight=max(weight, 5.0), wrench=wrench, handovers=handovers,
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
                  verbose=False, gripper_kind=gk, bracing=bracing)
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
