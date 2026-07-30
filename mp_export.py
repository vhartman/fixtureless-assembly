"""Export a solved assembly plan (+ optimized witnesses) to the mr_bench multi-robot
motion planner as a SEQUENCE problem, and run / save / reload it.

The mr_bench package is used AS A MODULE and is never modified. Contract (reverse-engineered
from problems/rai/rai_envs.py, esp. rai_ur10_handover_env):
  * a problem is a `SequenceMixin, rai_env` subclass exposing self.C (rai scene),
    self.robots, self.tasks (list of Task), self.sequence (task-index order).
  * each Task(name, robots, SingleGoal(joint goal for those robots), type, frames);
    on task completion the planner does C.attach(frames[0], frames[1]) (child<-parent).
    type="goto" => no attach (pure motion goal).
  * our start scene (scattered parts + home arms) is directly compatible: the planner keys
    robots by joint-name PREFIX and ours are already "a1_ur_...".

ACTION -> TASK map (confirmed with the user 2026-07-20):
  pick(r,p,g)        -> pick,  frames=[gripper(r), p]
  place(r,p,.)       -> place, frames=[table, p]
  assemble(r,p,onto) -> place, frames=[onto, p]        (attach onto a PART, not only table)
  release(r,p)       -> goto   (retract; the re-parent already happened at assemble/place)
  add_grasp(r,p,g)   -> goto   (BIG TODO: should be a relative-pose gripper<->object
                                CONSTRAINT -- supported on another mr_bench branch; goto now)
  handover(rB,p,g)   -> handover, robots=[rA_giver, rB], frames=[gripper(rB), p]
"""
import sys
import json
import os as _os

import numpy as np

_MRBENCH_SRC = "/home/valentin/git/postdoc/robotic-venv/mr_bench/src"


def _ensure_mrbench():
  if _MRBENCH_SRC not in sys.path:
    sys.path.insert(0, _MRBENCH_SRC)
  # A3 FIXED mr_bench-side: problems/__init__.py now guards the optional mr_vamp_env /
  # pinocchio / mujoco imports with try/except ImportError (not find_spec), so the VAMP .so
  # failing to load in this venv no longer breaks the package import. The sys.modules stub we
  # used to pre-seed is gone.
  try:
    import multi_robot_multi_goal_planning  # noqa: F401
  except ImportError as e:
    raise ImportError(
        f"mr_bench not importable from {_MRBENCH_SRC}: {e}. It is used as a module; "
        "do not modify it -- check the path / that its deps are installed.") from e


import robotic as ry
import assembly as A
import solver as S
import problem as P
from env import make_four_robot_env


# ---------------------------------------------------------------------------
# action -> Task
# ---------------------------------------------------------------------------
def _states_along_plan(C, assembly, plan, robots, gk, bracing):
  # re-derive with the SEARCH's grasp table: the on-the-fly fallback enumerates fewer grasps
  # (rest-face pruning), so e.g. an inhand_reorient to a table-blocked grasp would not match.
  _checker, grasp_table = S._search_context(C, assembly, robots)
  s = A.get_initial_state(C, assembly)
  states = [s]
  for a in plan:
    s = next(ns for act, ns in A.successors(s, assembly, robots, grasp_table=grasp_table,
                                            gripper_kind=gk, handovers=True, allow_place=True,
                                            bracing=bracing) if act == a)
    states.append(s)
  return states


def _holder_of(state, part):
  for c in state.supports.get(part, ()):
    if c[0] == "grasp":
      return c[1]
  # part may be inside a held group -> find the group's gripper holder
  for group in state.subassemblies():
    if part in group:
      for q in group:
        for c in state.supports.get(q, ()):
          if c[0] == "grasp":
            return c[1]
  return None


def _root_of(state, part, mate_parent):
  # the FIXED mate-tree base of `part`'s subassembly: follow the static assemble 'onto'
  # edges (mate_parent, built from the plan's assemble actions) UP to the base, staying
  # inside the current group. This mirrors mr_bench's tree, which is built incrementally
  # by assemble => attach(onto, part) and whose root is PERMANENT (obj_0). Our State, by
  # contrast, kinematically RE-ROOTS the tree to whichever leaf is currently grasped
  # (P._group_root returns that leaf) -- attaching that leaf to a gripper would TEAR it off
  # the rest of the assembly in mr_bench's single-frame-attach model. A lone (not-yet-mated)
  # part is its own root: its mate_parent is not in its singleton group, so the loop stops.
  group = next((g for g in state.subassemblies() if part in g), {part})
  cur, seen = part, set()
  while mate_parent.get(cur) in group and cur not in seen:
    seen.add(cur)
    cur = mate_parent[cur]
  return cur


_SOLVE_IT = object()   # 'no keyframe supplied -- solve it here' (see build_tasks)


def build_tasks(assembly, plan, witnesses, states, robots, robot_idx, robot_ik,
                home_state, pre_witnesses=None):
  """Map the plan+witnesses to (tasks, sequence_names). robot_idx: {r: [joint indices]}
  in the problem config's joint order; witnesses[i] is the full-config keyframe for plan[i]
  (None for non-IK actions -> skipped, no motion goal needed).

  `pre_witnesses` (--fuse-pre-skill, {plan index: keyframe or None}) supplies the pre-* GOTO
  keyframes ready-made, from solver.fuse_pre_skill_witnesses. None (the default) keeps the
  historical behaviour: each split solves its own pre keyframe here, inside the export. A
  missing SKILL witness, by contrast, is a hard error -- it cannot be substituted."""
  from multi_robot_multi_goal_planning.problems.planning_env import Task
  from multi_robot_multi_goal_planning.problems.core.goals import SingleGoal

  # cur[r] = robot r's CURRENT joint sub-vector (start: home). A witness updates the
  # ACTIVE arm(s); a no-witness / release action leaves the arm where it was -- so its goal
  # keeps it AT ITS GRASP, which is what makes a release's re-parent capture the object at
  # its true (held) pose rather than at home.
  cur = {r: np.asarray(home_state)[robot_idx[r]].copy() for r in robots}

  # mate tree exactly as mr_bench builds it: each assemble(r, part, onto) -> attach(onto, part)
  mate_parent = {a[2]: a[3] for a in plan if a[0] == "assemble"}

  def goal_for(rs, wit):
    parts = []
    for r in rs:
      if wit is not None:
        cur[r] = np.asarray(wit)[robot_idx[r]]             # active arm -> witness config
      parts.append(cur[r])
    return SingleGoal(np.concatenate(parts))

  def _support_frame(state, root):
    # the frame the released/placed assembly root should attach to. Scan the root's WHOLE
    # group for a grounding grasp (some member still held by a gripper -> that gripper);
    # else it rests on the table. (root itself may be a re-rooting-agnostic base whose own
    # State carrier is a "part" edge, so we must look across the group, not just at root.)
    group = next((g for g in state.subassemblies() if root in g), {root})
    for p in group:
      car = state.parent.get(p)
      if car is not None and car[0] == "gripper":
        return robot_ik.gripper_frame[car[1]]
    return "table"

  from visualization import _active_robots

  pre_pick = getattr(robot_ik, "pre_pick_delta", 0.0) > 0.0       # --pre-pick:      split pick
  pre_place = getattr(robot_ik, "pre_place_delta", 0.0) > 0.0     # --pre-place:     split place
  pre_add_grasp = getattr(robot_ik, "pre_add_grasp_delta", 0.0) > 0.0  # --pre-add-grasp: split brace reach
  pre_ins = getattr(robot_ik, "pre_insertion_delta", 0.0) > 0.0   # --pre-insertion: split assemble

  def split_skill(a, i, wit, involved, name, kind, mk_skill):
    # --pre-*: prepend a GOTO to the pre-skill FUNNEL pose (skill witness backed off along its
    # approach axis) before the REAL skill task. Pure motion first (the collision-free approach
    # the free-space planner connects to), then the unchanged skill at the real witness -- so the
    # scene graph ends exactly like the un-split task; only the last DELTA is contact-rich (on the
    # mr_bench SKILLS branch the second task becomes the skill). goal_for mutates cur: T_pre then
    # the real witness, so downstream keyframes see the real skill config.
    # MISSING KEY vs EXPLICIT None. A consistency pass reports only the keyframes it
    # ACCEPTED; a rejected split must fall back to the exporter's OWN solve, not to a
    # zero-length funnel. Reading a missing key as None conflated the two, so turning
    # --fuse-pre-skill on silently degraded every split it rejected -- which is what made the
    # flag non-identity at zero accepts and blocked defaulting it. Only an EXPLICIT None now
    # means "this split has no funnel".
    # PRE keyframes are the one thing the exporter may solve itself (a rejected --fuse-pre-skill
    # split has to fall back to something, and a zero-length funnel is worse). The SKILL witness
    # is a different matter -- see the hard failure below.
    pre_wit = (pre_witnesses.get(i, _SOLVE_IT) if pre_witnesses is not None else _SOLVE_IT)
    if pre_wit is _SOLVE_IT:
      pre_wit = robot_ik.pre_skill_witness(a, states[i], states[i + 1], assembly, wit)
    pan = f"{i:02d}_pre_{kind}_{'_'.join(str(x) for x in a[1:])}"
    if pre_wit is None and pre_witnesses is not None and i in pre_witnesses:
      # the fallback is silent in the historical path; once the consistency pass owns these
      # keyframes, say it -- a None here exports a pre GOTO whose goal EQUALS the skill goal.
      print(f"  [pre] {i:02d} {kind}: NO pre keyframe -- the goto goal equals the skill goal "
            f"(zero-length funnel)")
    pre_goal = goal_for(involved, pre_wit if pre_wit is not None else wit)   # cur := T_pre
    return [(pan, Task(pan, involved, pre_goal, type="goto")),
            (name, mk_skill(goal_for(involved, wit)))]                       # cur := real witness

  tasks, seq, step_names = [], [], []                             # step_names[i] = names of plan step i
  _NEEDS_WITNESS = ("pick", "place", "stow", "assemble", "add_grasp", "handover")
  for i, a in enumerate(plan):
    kind, r = a[0], a[1]
    wit = witnesses[i]
    if wit is None and kind in _NEEDS_WITNESS:
      # `goal_for(rs, None)` deliberately leaves cur[] alone -- that is what lets a RELEASE
      # capture the object at its held pose. But for a skill that must REACH somewhere it means
      # the exported goal is the arm's PREVIOUS configuration, and mr_bench then attaches the
      # part there: measured 22_assemble_a2_obj_4_obj_0 at 4.54mm penetration with no witness
      # behind it at all. Loud by default (some scenes rely on the carried config for `place`),
      # FATAL under --strict-mate-collisions, which is the flag that asks for exactly this rigor.
      msg = (f"step {i} {tuple(a)[:4]} has NO witness -- the exported goal is the acting arm's "
             f"PREVIOUS configuration and the part gets attached there. A tight "
             f"--strict-collisions can veto every candidate; check the [strict-collisions] "
             f"veto count.")
      if getattr(robot_ik, "mate_pair_gate", False):
        raise ValueError(f"mp_export: {msg}")
      print(f"  [mp-export] WARNING: {msg}")
    emit = None                                                   # None => single task `t`; else [(name,Task)]
    name = f"{i:02d}_{kind}_{'_'.join(str(x) for x in a[1:])}"
    # ALL robots whose witness pose must be applied this keyframe: the actor PLUS the
    # holder(s) of the group the action moves (e.g. an assemble's `onto` group is held by
    # another arm). Declaring only the actor left the idle holder homed in mr_bench -> the
    # held part (obj_0) snapped to that arm's home pose and every downstream mate misaligned.
    # `goal_for` poses each in task-robot order at its witness config, pinning the holder.
    involved = _active_robots(a, states[i], robot_ik) or [r]
    if kind == "pick":
      # --pre-pick splits into a goto to the retracted pre-grasp + the real pick (grasp+attach).
      frames = [robot_ik.gripper_frame[r], _root_of(states[i], a[2], mate_parent)]
      mk = lambda g: Task(name, involved, g, type="pick", frames=frames)
      emit = split_skill(a, i, wit, involved, name, kind, mk) if pre_pick else None
      if emit is None:
        t = mk(goal_for(involved, wit))
    elif kind == "place":
      # --pre-place splits into a goto to the lifted pre-place + the real place (set-down+attach).
      frames = ["table", _root_of(states[i], a[2], mate_parent)]
      mk = lambda g: Task(name, involved, g, type="place", frames=frames)
      emit = split_skill(a, i, wit, involved, name, kind, mk) if pre_place else None
      if emit is None:
        t = mk(goal_for(involved, wit))
    elif kind == "assemble":
      # --pre-insertion splits into a goto to the funnel mouth + the real mate (seats at `wit`).
      frames = [a[3], a[2]]
      mk = lambda g: Task(name, involved, g, type="place", frames=frames)
      emit = split_skill(a, i, wit, involved, name, kind, mk) if pre_ins else None
      if emit is None:
        t = mk(goal_for(involved, wit))
    elif kind == "inhand_reorient":
      # IN-PLACE skill: the arm KEEPS its config (goal_for with wit=None keeps cur[r]); the
      # part's pose IN the gripper snaps to the NEW grasp transform. mr_bench mechanics: the
      # part is already gripper-parented (from its pick), so the make_appear side effect sets
      # the new RELATIVE pose (= inverse of the grasp EE transform), and the type="pick"
      # re-attach to the same gripper keeps that world pose (re-derives the same relative).
      # v1: lone parts only, so the attach child is the part itself.
      _g = A.grasp_pose(assembly, a[2], a[3])
      _ee = np.concatenate([_g[:3], P._quat_mul(_g[3:], robot_ik.q_correction[r])])
      t = Task(name, [r], goal_for([r], None), type="pick",
               frames=[robot_ik.gripper_frame[r], a[2]],
               side_effect="make_appear", side_effect_data=P._pose_invert(_ee))
    elif kind == "release":
      # DETACH: re-parent the released object to its post-release support (table or a
      # CONTINUING holder's gripper). We pin BOTH the releasing arm AND every remaining holder
      # at their HOLDING config (goal_for with wit=None keeps each cur[]): the releasing arm so
      # the re-parent captures the object at its true held pose, and the continuing holder so
      # the object is frozen relative to the RIGHT gripper at the RIGHT config -- otherwise that
      # holder is idle, drifts, and the object (now parented to its gripper) later JUMPS to
      # wherever that arm moved.
      #
      # BIG TODO(mp): a continuing holder should be a RELATIVE-POSE grasp CONSTRAINT
      # (gripper <-> object held rigid), NOT a joint goal. Pinning it to one specific config
      # OVER-CONSTRAINS the release mode -- the holder is forced to an exact pose instead of
      # merely "keep holding", which can make the release keyframe needlessly hard/infeasible
      # for the planner. The right fix is a per-mode held constraint (same family as the
      # add_grasp brace TODO / the other mr_bench constraint branch). Joint goal is the interim
      # that keeps the scene graph correct until that constraint exists.
      root = _root_of(states[i + 1], a[2], mate_parent)
      grp = next((g for g in states[i + 1].subassemblies() if root in g), {root})
      rel_robots = list(dict.fromkeys([r] + robot_ik._holders_of_group(states[i + 1], grp)))
      t = Task(name, rel_robots, goal_for(rel_robots, None), type="place",
               frames=[_support_frame(states[i + 1], root), root])
    elif kind == "add_grasp":
      # TODO(mp): a brace should be a RELATIVE-POSE CONSTRAINT gripper<->object, not a goto
      # keyframe (other mr_bench branch). goto = motion to the brace pose for now.
      # --pre-add-grasp splits into a goto to the retracted pre-grasp + the real brace reach.
      mk = lambda g: Task(name, involved, g, type="goto")
      emit = split_skill(a, i, wit, involved, name, kind, mk) if pre_add_grasp else None
      if emit is None:
        t = mk(goal_for(involved, wit))
    elif kind == "handover":
      rB = r
      rA = _holder_of(states[i], a[2])
      task_robots = [rA, rB] if rA is not None and rA != rB else [rB]
      t = Task(name, task_robots, goal_for(task_robots, wit), type="handover",
               frames=[robot_ik.gripper_frame[rB], _root_of(states[i], a[2], mate_parent)])
    else:
      step_names.append([])                                   # non-exported: keep index alignment
      continue
    if emit is None:
      emit = [(name, t)]
    for nm, tk in emit:
      tasks.append(tk)
      seq.append(nm)
    step_names.append([nm for nm, _ in emit])

  # terminal: all robots return home (a clean, always-reachable end mode)
  from multi_robot_multi_goal_planning.problems.planning_env import Task as _T
  from multi_robot_multi_goal_planning.problems.core.goals import SingleGoal as _SG
  term = _T("terminal", list(robots),
            _SG(np.concatenate([np.asarray(home_state)[robot_idx[r]] for r in robots])))
  tasks.append(term)
  seq.append("terminal")
  return tasks, seq, step_names


# ---------------------------------------------------------------------------
# problem instance
# ---------------------------------------------------------------------------
def _dependency_graph(plan, states, step_names, robots, robot_ik):
  """Build a mr_bench DependencyGraph = the PARTIAL order our linear plan is one linearization
  of. Sound-by-construction: for every action we chain it AFTER the last action that touched
  any RESOURCE it shares -- (a) each robot in the task (per-arm order), and (b) every part in
  the group(s) the action moves, in both the pre- and post-state (so all actions on the same
  growing assembly stay ordered; lone-part picks only chain on that part). Two actions that
  share NO robot and NO part are left unordered -> the motion planner may run them in either
  order / concurrently. All edges point to EARLIER tasks (indices only grow), so no cycles."""
  from multi_robot_multi_goal_planning.problems.core.dependency_graph import DependencyGraph
  from visualization import _active_robots
  g = DependencyGraph()
  for nms in step_names:                                     # step_names[i] = task names of plan[i]
    for nm in nms:
      g.add_node(nm)
  g.add_node("terminal")
  last = {}                                                  # resource -> most-recent task name
  for i, a in enumerate(plan):
    nms = step_names[i]
    if not nms:
      continue                                               # non-exported action (no task)
    first, leaf = nms[0], nms[-1]                            # a pre-insertion assemble => 2 tasks
    for prev, nm in zip(nms, nms[1:]):
      g.add_dependency(nm, prev)                             # chain within the step (pre-assemble -> assemble)
    res = set(_active_robots(a, states[i], robot_ik) or [a[1]])
    named = [a[2]] + ([a[3]] if a[0] == "assemble" else [])
    for st in (states[i], states[i + 1]):
      for p in named:
        res |= set(next((grp for grp in st.subassemblies() if p in grp), {p}))
    for r in res:
      if last.get(r) not in (None, first):
        g.add_dependency(first, last[r])                     # step's FIRST task waits on r's last writer
      last[r] = leaf                                         # step's LAST task becomes r's new writer
  for nm in set(last.values()):                              # terminal after every leaf
    g.add_dependency("terminal", nm)
  return g


def _install_robot_margin(env, margin, robots, gripper_frame, ramp=0.0, table_margin=0.0):
  """MONKEY-PATCH a robot-robot clearance margin onto the exported problem (mr_bench unmodified).

  `table_margin` adds the CARRIED-PART-vs-TABLE case (the transit graze: BiRRT+shortcut has no
  reason to stay high at a 0.1mm collision tolerance, so the executed path's mm-level tracking
  error drives the held part into the table and the contact force snaps the grasp -- measured
  obj_0~world at 40-120N against a 25N cap, the FIRST failure in every config tried). A part
  must touch the table at pick and set-down, so the margin is relaxed at BOTH ends of the
  segment: required = table_margin * min(1, d_to_goal/ramp, d_from_prev_goal/ramp), where
  prev-goal is the carrying arm's own previous task goal (its segment start). Both anchors are
  properties of the mode, so the predicate stays a pure function of (q, mode). Applies ONLY to
  bodies riding on an arm -- a RESTING part keeps zero table clearance (a place needs contact),
  and the arm itself may still work arbitrarily close to the table.

  TWO RELAXATIONS, so clearance is large where it can be and small only where it must be.

  1. BY PARTICIPATION. A handover needs its two arms close; it does not need either of them close
     to the third. The margin is therefore dropped only between robots that are BOTH in the
     current mode's task -- everyone else keeps the full margin for the whole segment. The mode
     already carries this (`task.robots`), so nothing has to be inferred.

  2. BY EE DISTANCE TO THE GOAL, for that involved pair: required = margin * min(1, d_ee/ramp),
     where d_ee is how far the arms' END EFFECTORS still are from their goal poses. Full margin
     in transit, relaxing to zero exactly at the keyframe where contact is intended.

     Keyed on EE distance rather than path position ON PURPOSE. "The last 10% of the path" is
     history-dependent and a sampling planner has no such notion -- its predicate must be a pure
     function of (q, mode) or edge checks and rewiring become inconsistent. Distance-to-goal is
     that pure function and expresses the same intent, so the ramp works INSIDE planning rather
     than as post-processing.

  Frames are classified by KINEMATIC ROOT, so a carried part counts as its holder; recomputed per
  call because attaches rewrite the parents. Goal EE poses are cached per mode (posing C to read
  them, restoring the caller's state) since they cost FK.
  """
  import numpy as _np

  def _owner(fname):
    try:
      f = env.C.getFrame(fname)
    except Exception:
      return None
    seen = set()
    while f is not None and f.name not in seen:
      seen.add(f.name)
      for r in robots:
        if f.name.startswith(r):
          return r
      f = f.getParent()
    return None

  _goal_ee = {}

  # install time is right after make_problem, so C is at HOME -- the fallback start anchor
  # for a robot whose mode has no earlier task of its own.
  _home_ee = {}
  try:
    for _r in robots:
      _home_ee[_r] = _np.asarray(env.C.getFrame(gripper_frame[_r]).getPosition(), dtype=float)
  except Exception:
    pass

  def _task_of(m):
    try:
      combos = env.get_valid_next_task_combinations(m)
      return env.get_active_task(m, combos[0] if combos else None)
    except Exception:
      return None

  def _goals(m):
    """{robot: goal EE position} for the mode's task; cached, C restored."""
    key = tuple(getattr(m, "task_ids", ()) or ()) if m is not None else ()
    hit = _goal_ee.get(key)
    if hit is not None:
      return hit
    out = {}
    task = _task_of(m)
    if task is not None:
      try:
        g = _np.asarray(task.goal.sample(m), dtype=float)
        saved = _np.asarray(env.C.getJointState(), dtype=float)
        qq, off = saved.copy(), 0
        for tr in task.robots:
          idx = _np.asarray(env.robot_idx[tr], dtype=int)
          qq[idx] = g[off:off + idx.size]
          off += idx.size
        env.C.setJointState(qq)
        for r in task.robots:
          out[r] = _np.asarray(env.C.getFrame(gripper_frame[r]).getPosition(), dtype=float)
        env.C.setJointState(saved)                        # never leave the caller's state moved
      except Exception:
        out = {}
    _goal_ee[key] = out
    return out

  _task_ee = {}                              # (robot, task_id) -> that robot's EE at the goal

  def _ee_at_task(r, tid, m):
    key = (r, tid)
    if key in _task_ee:
      return _task_ee[key]
    out = None
    try:
      task = env.tasks[tid]
      if r in (getattr(task, "robots", ()) or ()):
        g = _np.asarray(task.goal.sample(m), dtype=float)
        saved = _np.asarray(env.C.getJointState(), dtype=float)
        qq, off = saved.copy(), 0
        for tr in task.robots:
          idx = _np.asarray(env.robot_idx[tr], dtype=int)
          qq[idx] = g[off:off + idx.size]
          off += idx.size
        env.C.setJointState(qq)
        out = _np.asarray(env.C.getFrame(gripper_frame[r]).getPosition(), dtype=float)
        env.C.setJointState(saved)
    except Exception:
      out = None
    _task_ee[key] = out
    return out

  def _prev_ee(r, m):
    """r's EE at ITS OWN previous task goal -- where r's current segment started. env.tasks is
    in plan order for both mixins, so the latest earlier task involving r is the predecessor."""
    try:
      k = int((getattr(m, "task_ids", ()) or ())[robots.index(r)])
    except Exception:
      return _home_ee.get(r)
    for tid in range(k - 1, -1, -1):
      try:
        t = env.tasks[tid]
      except Exception:
        break
      if r in (getattr(t, "robots", ()) or ()):
        p = _ee_at_task(r, tid, m)
        if p is not None:
          return p
    return _home_ee.get(r)

  qmargin = max(margin, table_margin)

  def _too_close(m):
    task = _task_of(m)
    involved = set(getattr(task, "robots", ()) or ())
    goals = None
    for c in env.C.getCollisions(qmargin):
      fa, fb, d = c[0], c[1], float(c[2])
      if d >= qmargin:
        continue
      # CARRIED part vs the table: relaxed at both segment ends, full margin mid-transit.
      if table_margin > 0.0 and "table" in (fa, fb):
        other = fb if fa == "table" else fa
        ro = _owner(other)
        if ro is not None and not other.startswith(ro):  # a body RIDING on ro, not ro's arm
          f = 1.0
          if ramp > 0.0:
            # BOTH anchors are ro's OWN tasks (task_ids is per robot). Reading the goal off
            # _task_of(m) instead -- the next TRANSITION task, usually another robot's --
            # left d_to_goal at infinity for the carrier, so the margin never relaxed at its
            # set-down and every place-transition sample was invalid.
            try:
              ee = _np.asarray(env.C.getFrame(gripper_frame[ro]).getPosition(), dtype=float)
              tid = int((getattr(m, "task_ids", ()) or ())[robots.index(ro)])
              gp = _ee_at_task(ro, tid, m)
              d_go = float(_np.linalg.norm(ee - gp)) if gp is not None else _np.inf
              pv = _prev_ee(ro, m)
              d_pv = float(_np.linalg.norm(ee - pv)) if pv is not None else _np.inf
              f = min(1.0, min(d_go, d_pv) / ramp)
            except Exception:
              f = 0.0                          # fail OPEN -- the base predicate still applies
          # The requirement interpolates from CONTACT-ALLOWED (-margin/2) at the segment ends
          # to +margin mid-transit. A plain `d < margin*f` still vetoed at f == 0: a set-down
          # goal has the part IN CONTACT (d <= 0, measured obj_0~table -0.4mm), and any veto
          # of negative d there makes the place transition UNSATISFIABLE -- measured 0/4000
          # valid samples, which is exactly BiRRT's 10k-attempts wall. Penetration at the
          # ends stays the BASE predicate's job; the margin only shapes MID-TRANSIT clearance.
          if d < table_margin * (1.5 * f - 0.5):
            return True
        continue                             # table pairs never enter the robot-robot branch
      if margin <= 0.0 or d >= margin:
        continue
      ra, rb = _owner(fa), _owner(fb)
      if ra is None or rb is None or ra == rb:
        continue
      req = margin
      if ra in involved and rb in involved:
        req = 0.0
        if ramp > 0.0:
          if goals is None:
            goals = _goals(m)
          try:
            d_ee = max(float(_np.linalg.norm(
                _np.asarray(env.C.getFrame(gripper_frame[r]).getPosition(), dtype=float)
                - goals[r])) for r in (ra, rb) if r in goals)
            req = margin * min(1.0, d_ee / ramp)
          except Exception:
            req = 0.0
      elif ramp > 0.0 and (ra in involved) != (rb in involved):
        # ONE-SIDED relaxation (2026-07-30): an involved arm NEAR ITS GOAL is executing its
        # task posture, and that fixed posture can sit inside the margin of an UNINVOLVED
        # arm's BASE links -- which no sample of the free arm can change (measured: chair-8
        # task 28_assemble_a1_obj_7_obj_5, a1's pinned wrist 7.7mm from a2's base link at
        # HOME -> every transition sample invalid -> "Failed to sample after 10000
        # attempts"). Ramp the pair's requirement on the INVOLVED arm's own distance to its
        # goal: full margin while it transits, relaxed to the base predicate exactly where
        # the task pins it -- the same contact-allowed-at-anchors philosophy as the
        # involved-pair and table ramps. Fail OPEN like the branch above: a missing goal
        # must not veto a state the base predicate accepts.
        rinv = ra if ra in involved else rb
        if goals is None:
          goals = _goals(m)
        try:
          d_ee = float(_np.linalg.norm(
              _np.asarray(env.C.getFrame(gripper_frame[rinv]).getPosition(), dtype=float)
              - goals[rinv]))
          req = margin * min(1.0, d_ee / ramp)
        except Exception:
          req = 0.0
      # CLAMP AT CONTACT (2026-07-30): a fully-relaxed pair (req 0) must not be vetoed for a
      # sub-tolerance witness graze -- `d < req` with d = -0.1mm and req = 0 still fired,
      # so every transition whose goal formation TOUCHES (a seated mate, a co-grasp) was
      # unsatisfiable under the margin no matter the relaxation (chair-8's second wall).
      # Penetration is the BASE predicate's job (same rule the table branch documents);
      # clamping d at 0 keeps every positive requirement guarding real proximity while
      # handing at-or-below-contact states back to the tolerance.
      if max(d, 0.0) < req:
        if _os.environ.get("MP_MARGIN_DEBUG"):
          print(f"[margin-debug] VETO {fa} ~ {fb} d={d*1000:.1f}mm req={req*1000:.1f}mm "
                f"owners=({ra},{rb}) involved={sorted(involved)}")
        return True
    return False

  def _wrap(fn):
    def patched(*a, **kw):
      if not fn(*a, **kw):                               # poses C at (mode, q) as a side effect
        return False
      m = kw.get("m", a[1] if len(a) > 1 else None)
      return not _too_close(m)
    return patched

  # BOTH full-scene predicates: BiRRT leans on the _np variant, so wrapping only
  # is_collision_free would leave the margin unenforced on the hot path.
  # is_collision_free_for_robot is deliberately NOT wrapped -- it judges a SUBSET of robots and
  # explicitly discounts collisions with the others, so a cross-robot margin would invert it.
  env.is_collision_free = _wrap(env.is_collision_free)
  env.is_collision_free_np = _wrap(env.is_collision_free_np)
  return env


def make_problem(C_start, assembly, plan, witnesses, robots, robot_ik, gk, bracing,
                 depgraph=False, collision_tolerance=0.01, pre_witnesses=None,
                 robot_robot_margin=0.0, robot_margin_ramp=0.0, table_margin=0.0):
  """Build a live mr_bench problem from our solved plan. depgraph=False -> SequenceMixin (a
  fixed total order); depgraph=True -> DependencyGraphMixin (the partial order, so the planner
  may reorder independent actions). Same tasks/frames/witnesses either way."""
  _ensure_mrbench()
  from multi_robot_multi_goal_planning.problems.rai_base_env import rai_env
  from multi_robot_multi_goal_planning.problems.planning_env import (
      SequenceMixin, DependencyGraphMixin, BaseModeLogic)

  states = _states_along_plan(C_start, assembly, plan, robots, gk, bracing)
  home = np.asarray(C_start.getJointState())               # the initial home pose (== start)
  Mixin = DependencyGraphMixin if depgraph else SequenceMixin

  class AssemblyEnv(Mixin, rai_env):
    def __init__(self):
      self.C = ry.Config()
      self.C.addConfigurationCopy(C_start)
      self.robots = list(robots)
      # Default rai_env.__init__ (delete_visual_only=True): mr_bench's delete_visual_only_frames
      # is now COLLISION-AWARE (A2 fix) -- it keeps every non-visual frame AND its ancestors, so
      # the robotiq finger collision capsules (children of mesh frames) survive with their
      # parents instead of orphaning. So we no longer need to keep the full visual scene, and
      # genuinely-decorative meshes are pruned from the collision config. (A5 also fixed:
      # set_to_mode guards setContact against shapeless frames -> car-mini's obj_X stays
      # shapeless while its obj_X__shape cylinder keeps the wheel collision. All three of our
      # old workarounds -- finger re-parent, ssBox baking, sys.modules stub -- are removed.)
      rai_env.__init__(self)                               # -> robot_idx, start_pos, limits
      self.manipulating_env = True
      self.tasks, seq, step_names = build_tasks(
          assembly, plan, witnesses, states, self.robots, self.robot_idx, robot_ik, home,
          pre_witnesses=pre_witnesses)
      self.step_names = step_names                         # task names per plan action -- lets
      self.plan_actions = [tuple(a) for a in plan]         # sim.py map completed tasks back to
                                                           # plan actions (grasp/weld events)
      self.states = states                                 # discrete state trajectory -- the
                                                           # only place report_cograsp_drift can
                                                           # read WHO holds WHAT per mode
      if depgraph:
        self.graph = _dependency_graph(plan, states, step_names, self.robots, robot_ik)
      else:
        self.sequence = self._make_sequence_from_names(seq)   # flat order (incl. terminal)
      # total allowed PENETRATION at validated configs (mr_bench semantics). 0.01 is the
      # historical default; pass ~0 (with --strict-collisions witnesses) for a plan whose
      # validated configs contain no tolerated penetration at all. Must be set BEFORE
      # BaseModeLogic.__init__ (start-mode validation uses it).
      self.collision_tolerance = collision_tolerance
      BaseModeLogic.__init__(self)
      self.prev_mode = self.start_mode

  env = AssemblyEnv()
  if (robot_robot_margin and robot_robot_margin > 0.0) or (table_margin and table_margin > 0.0):
    if robot_robot_margin and robot_robot_margin > 0.0:
      print(f"[mp] --mp-robot-margin {robot_robot_margin:g}m: robot-robot clearance enforced by a "
            f"wrapper around mr_bench's predicate (that module stays unmodified). Planning will be "
            f"slower -- its early-out on 'nothing penetrates' no longer applies."
            + (f" Relaxed to 0 between the TASK'S OWN robots, ramped over the last "
               f"{robot_margin_ramp:g}m of EE travel to the goal." if robot_margin_ramp > 0
               else " Dropped entirely between the TASK'S OWN robots (no ramp)."))
    if table_margin and table_margin > 0.0:
      print(f"[mp] --mp-table-margin {table_margin:g}m: CARRIED parts keep this clearance to the "
            f"table mid-transit (the graze that snaps grasps), relaxed over {robot_margin_ramp:g}m "
            f"of EE travel at BOTH segment ends so pick and set-down still touch. Resting parts "
            f"and the arms themselves are unaffected.")
      if not robot_margin_ramp or robot_margin_ramp <= 0.0:
        print("[mp] WARNING: --mp-table-margin without --mp-margin-ramp demands the full margin "
              "AT the pick keyframe too, where the part is only part_float above the table -- "
              "the mode after every pick starts in violation and planning will wedge.")
    _install_robot_margin(env, float(robot_robot_margin or 0.0), list(robots),
                          robot_ik.gripper_frame, ramp=float(robot_margin_ramp),
                          table_margin=float(table_margin or 0.0))
  return env


# ---------------------------------------------------------------------------
# run / save / reload
# ---------------------------------------------------------------------------
def run_motion_planner(problem, runtime=30.0, verbose=True):
  """Run the composite (multi-mode) PRM on the sequence problem. Returns (path, info)."""
  _ensure_mrbench()
  from multi_robot_multi_goal_planning.planners.composite_prm_planner import (
      CompositePRM, CompositePRMConfig)
  from multi_robot_multi_goal_planning.planners.termination_conditions import (
      RuntimeTerminationCondition)
  cfg = CompositePRMConfig()
  path, info = CompositePRM(problem, config=cfg).plan(
      ptc=RuntimeTerminationCondition(runtime), optimize=False)
  if verbose:
    print(f"[mp] planner returned {'a path of %d states' % len(path) if path else 'NO path'}")
  return path, info


def run_birrt(problem, runtime=60.0, shortcut=True, shortcut_iters=1000,
              shortcut_resolution=None, optimize=False, verbose=True, seed=None,
              stall_timeout=None):
  """Run Bidirectional RRT* to the FIRST solution (optimize=False), then optionally apply
  robot/mode shortcutting as a post-process. Returns (path, info). No cost optimization --
  the tree stops as soon as a valid path is found; shortcutting only removes slack.

  NOTE: `runtime` bounds ONLY the BiRRT tree search. Shortcutting is a SEPARATE post-process
  with its OWN budget (`shortcut_iters` successful shortcuts, or ~2.5x that many attempts) and
  is NOT time-limited -- on a long path it can dominate wall-clock. Lower `shortcut_iters`
  (or pass shortcut=False) to cap/skip it.

  PERF: robot_mode_shortcut's OWN default collision resolution is 0.001 -- 10x FINER than the
  env's collision_resolution (0.01, what BiRRT itself used), for zero quality gain but ~10x
  the wall-clock (measured: 46s -> 4.6s, identical cost). We default `shortcut_resolution` to
  the env's collision_resolution so shortcutting checks no finer than the planner did."""
  _ensure_mrbench()
  if seed is not None:
    # mr_bench draws from the GLOBAL numpy RNG throughout (sampling, tree ops, shortcut index
    # picks) plus stdlib random.shuffle in shortcutting -- seeding both makes the whole
    # BiRRT+shortcut draw REPRODUCIBLE. Unseeded, the same command yields paths whose failure
    # CLASSES differ (one draw grazes the table at 57N, the next has no table contact at all),
    # so every A/B on top was noise. None (the default) keeps the historical unseeded draw.
    import random as _random
    np.random.seed(int(seed))
    _random.seed(int(seed))
    print(f"[mp/birrt] RNG seeded: {int(seed)} (np.random + random) -- reproducible draw; "
          f"runtime becomes a deterministic consultation budget, not wall seconds")
  from multi_robot_multi_goal_planning.planners.planner_birrtstar import BidirectionalRRTstar
  from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
  from multi_robot_multi_goal_planning.planners.shortcutting import robot_mode_shortcut
  from multi_robot_multi_goal_planning.planners.termination_conditions import (
      RuntimeTerminationCondition, PlannerTerminationCondition)
  cfg = BaseRRTConfig()
  # internal (tree_extension) shortcutting is only meaningful while optimizing; our EXPLICIT
  # robot_mode_shortcut below is separate. optimize=True keeps improving cost until `runtime`.
  cfg.shortcutting = optimize
  # DISABLE mr_bench's pre-planning mode validation. It pre-checks each mode for a collision-
  # free goal and blacklists infeasible ones, propagating back -- on a SEQUENCE (linear) problem
  # a single collision-y keyframe (e.g. car-mini's active arm vs an idle arm parked at its entry
  # config) blacklists the whole chain to the start -> assert "Tried to add initial mode to the
  # invalid modes". Off, BiRRT plans the modes directly and just routes/searches per mode.
  cfg.with_mode_validation = False
  # SEEDED runs also need a DETERMINISTIC clock: the tree search gates iterations on
  # time.time() elapsed (planner_birrtstar ptc.should_terminate), so machine jitter shifts a
  # phase boundary, the RNG stream offsets, and two identically-seeded draws diverge (measured:
  # identical to state 15, different after). With a seed we swap time.time for a fake clock
  # (+1ms per consultation) FOR THIS CALL ONLY: first-solution return still exits early, and
  # the runtime budget becomes a deterministic consultation budget instead of wall seconds.
  # perf_counter (pure accounting there) stays real. mr_bench itself is untouched.
  import time as _time_mod
  _real_time = _time_mod.time
  if seed is not None:
    _fake_t = [_real_time()]
    def _fake_time():
      _fake_t[0] += 1e-3
      return _fake_t[0]
    _time_mod.time = _fake_time
  planner = BidirectionalRRTstar(problem, config=cfg)
  # PER-MODE PROGRESS BUDGET (stall_timeout, seconds): a wall shows up as the MODE FRONTIER
  # not growing while the tree burns transition attempts -- the "Failed to sample" stdout is
  # a debug print, not an interface, so scripts must not parse it. This ptc closes over the
  # planner and stops the pull once the frontier has been flat for `stall_timeout` seconds;
  # the caller reads the STRUCTURED outcome from info["frontier_*"] / info["stalled"].
  class _FrontierStallPTC(PlannerTerminationCondition):
    def __init__(self, pl, max_s, stall_s):
      self.pl, self.max_s, self.stall_s = pl, float(max_s), float(stall_s)
      self.best, self.t_best, self.stalled = -1, 0.0, False
    def __repr__(self):
      return f"max runtime {self.max_s}s, frontier stall {self.stall_s}s"
    def should_terminate(self, current_iterations=None, current_time=None):
      if current_time is None:
        return False
      n = len(getattr(self.pl, "modes", ()) or ())
      if n > self.best:
        self.best, self.t_best = n, current_time
      elif current_time - self.t_best > self.stall_s:
        self.stalled = True
        return True
      return current_time > self.max_s
  ptc = (RuntimeTerminationCondition(runtime) if stall_timeout is None
         else _FrontierStallPTC(planner, runtime, stall_timeout))
  try:
    path, info = planner.plan(ptc=ptc, optimize=optimize)
  finally:
    _time_mod.time = _real_time
  info = dict(info or {})
  _modes = list(getattr(planner, "modes", ()) or ())
  info["frontier_modes"] = len(_modes)
  info["frontier_task_ids"] = (
      [int(t) for t in np.atleast_1d(_modes[-1].task_ids)] if _modes else None)
  info["stalled"] = bool(getattr(ptc, "stalled", False))
  if verbose or stall_timeout is not None:
    print(f"[mp/birrt] frontier: {info['frontier_modes']} modes, deepest task_ids "
          f"{info['frontier_task_ids']}"
          + (f" -- STALLED (no frontier growth for {stall_timeout:g}s)"
             if info["stalled"] else ""))
  if verbose:
    tag = "optimized" if optimize else "first solution"
    print(f"[mp/birrt] {tag}: {'%d states' % len(path) if path else 'NO path'}")
  if path and shortcut:
    from multi_robot_multi_goal_planning.problems.util import path_cost
    before = path_cost(path, problem.batch_config_cost)
    res = (shortcut_resolution if shortcut_resolution is not None
           else getattr(problem, "collision_resolution", 0.01))
    if verbose:
      print(f"[mp/birrt] shortcutting (up to {shortcut_iters} iters, resolution {res}, "
            "not time-bounded)...")
    path, sc_info = robot_mode_shortcut(problem, path, max_iter=shortcut_iters, resolution=res)
    info["shortcut"] = sc_info
    if verbose:
      after = path_cost(path, problem.batch_config_cost)
      print(f"[mp/birrt] shortcut: {len(path)} states, cost {before:.3f} -> {after:.3f}")
  return path, info


def save_problem(path_json, build_params, robots, tasks, sequence_names):
  """Persist the EXPORTED problem so the motion planner can be re-run later WITHOUT
  re-solving the assembly: the scene is rebuilt from build_params, the goals are plain
  joint vectors, types/frames are strings."""
  data = {
      "build_params": build_params,                        # to rebuild C_start
      "robots": list(robots),
      "sequence": list(sequence_names),
      "tasks": [{"name": t.name, "robots": list(t.robots), "type": t.type,
                 "frames": list(t.frames) if t.frames else None,
                 "goal": (t.goal.goal.tolist() if hasattr(t.goal, "goal") else None)}
                for t in tasks],
  }
  with open(path_json, "w") as f:
    json.dump(data, f, indent=2)


def task_signature(problem):
  """Fingerprint of the EXPORTED TASK STRUCTURE a path was planned against.

  A saved path indexes tasks by INTEGER id, so a plan/path pair whose task lists differ even
  slightly relinks at the wrong moments -- and the existing guard only compares task COUNTS, so
  a same-count-different-semantics pair passes silently (e.g. a --pre-* split that moved, or a
  different `bracing`/`handovers` flag changing which arms a task involves). Hash what the ids
  actually MEAN: name, type, involved robots, attach frames, in order."""
  import hashlib
  h = hashlib.sha256()
  for t in problem.tasks:
    h.update(("|".join([
        str(getattr(t, "name", "")),
        str(getattr(t, "type", "")),
        ",".join(str(r) for r in (getattr(t, "robots", None) or [])),
        ",".join(str(f) for f in (getattr(t, "frames", None) or [])),
    ]) + "\n").encode())
  return h.hexdigest()[:16]


def save_path(path_json, path, problem=None):
  """Persist a solved BiRRT PATH (main --motion-plan --mp-save). Each mr_bench State ->
  {"q": joint vector, "mode": task_ids}. The discrete PLAN (main --save) must be saved from
  the SAME run: load_path replays against a problem REBUILT from that plan (make_problem is
  deterministic, so task ids / modes align).

  FORMAT: a bare LIST historically; with `problem` it becomes
  {"states": [...], "task_signature": ...} so a mismatched plan/path pair is DETECTED rather
  than silently relinking at the wrong tasks. load_path reads both, so old files still work."""
  states = [s.to_dict() for s in path] if path else []
  blob = states if problem is None else {"states": states,
                                        "task_signature": task_signature(problem)}
  with open(path_json, "w") as f:
    json.dump(blob, f, indent=2)


class _Mode:
  """Lightweight stand-in for an mr_bench Mode: only .task_ids + equality are used downstream
  (execute_motion_plan compares modes and reads task_ids). Avoids reconstructing real Mode
  objects (which need the live problem's mode graph)."""
  __slots__ = ("task_ids",)

  def __init__(self, task_ids):
    self.task_ids = list(task_ids)

  def __eq__(self, o):
    return isinstance(o, _Mode) and self.task_ids == o.task_ids

  def __ne__(self, o):
    return not self.__eq__(o)

  def __hash__(self):
    return hash(tuple(self.task_ids))


class _PathState:
  """Stand-in for an mr_bench path State: .q (plain joint array; sim's full_q handles the
  no-.state() case) and .mode (a _Mode)."""
  __slots__ = ("q", "mode")

  def __init__(self, q, mode):
    self.q = np.asarray(q, dtype=float)
    self.mode = mode


def load_path(path_json):
  """Load a path saved by save_path into lightweight (_PathState) objects that sim's
  execute_motion_plan consumes directly -- no mr_bench State/Mode reconstruction needed."""
  with open(path_json) as f:
    data = json.load(f)
  if isinstance(data, dict):                                 # new format: states + signature
    states, sig = data.get("states", []), data.get("task_signature")
  else:                                                      # historical bare list
    states, sig = data, None
  out = [_PathState(d["q"], _Mode(d["mode"])) for d in states]
  load_path.last_signature = sig                             # checked by the caller, if present
  return out


def mode_sequence(problem):
  """Walk the SEQUENCE problem's mode chain (same walk as report_problem_clearance) and
  return [mode_0, mode_1, ...]: mode_k is the attachment state IN FORCE while per-mode
  stretch k of a saved path executes; the transition k -> k+1 fires stretch k's tasks.
  This is what a desynced executor needs to collision-check arms at DIFFERENT path
  parameters: attachments are a function of how many events have FIRED (time), not of
  where any arm is, so world state at `fired=f` == mode_sequence(problem)[f]."""
  env = problem
  q_home = env.get_start_pos()
  conf_type = type(q_home)

  def goal_config(m, task):                                  # as report_problem_clearance
    goal = task.goal.sample(m)
    q = []
    for j, r in enumerate(env.robots):
      if r in task.robots:
        off = 0
        for tr in task.robots:
          if tr == r:
            q.append(goal[off:off + env.robot_dims[tr]])
            break
          off += env.robot_dims[tr]
      else:
        q.append(q_home.robot_state(j))
    return conf_type.from_list(q)

  chain, m = [], env.start_mode
  while True:
    chain.append(m)
    if env.is_terminal_mode(m):
      return chain
    combos = env.get_valid_next_task_combinations(m)
    task = env.get_active_task(m, combos[0] if combos else None)
    m = env.get_next_modes(goal_config(m, task), m)[0]


def report_problem_clearance(problem, plan=None, margin=0.02, verbose=True,
                             ref_scene=None, problem_assembly=None):
  """Per-mode collision/clearance report at the exported TASK-GOAL keyframes -- the exact
  configs (our witnesses) and the exact collision check (mr_bench's is_collision_free ->
  getCollisionsTotalPenetration) that BiRRT validates. For each mode: which plan action it
  belongs to, total penetration, and the closest/penetrating FRAME PAIRS within `margin` m.
  This makes 'looks fine in viser but the planner rejects it' concrete: the offending pair
  (arm-vs-part, part-vs-part rounding, arm-vs-arm) is named with its signed distance. Idle
  (non-switching) robots sit at HOME here, exactly as mr_bench's mode validation poses them.
  Returns [(mode_index, task_name, action, penetration, [(f1,f2,dist), ...]), ...]."""
  env = problem
  q_home = env.get_start_pos()
  conf_type = type(q_home)
  name_to_act = {}
  step_names = getattr(env, "step_names", None)
  if step_names is not None and plan is not None:
    for i, nms in enumerate(step_names):
      for nm in nms:
        name_to_act[nm] = tuple(plan[i])

  def goal_config(m, task):
    goal = task.goal.sample(m)
    q = []
    for j, r in enumerate(env.robots):
      if r in task.robots:
        off = 0
        for tr in task.robots:
          if tr == r:
            q.append(goal[off:off + env.robot_dims[tr]])
            break
          off += env.robot_dims[tr]
      else:
        q.append(q_home.robot_state(j))
    return conf_type.from_list(q)

  name_to_idx_ref = {}
  if step_names is not None:
    for i, nms in enumerate(step_names):
      for nm in nms:
        name_to_idx_ref[nm] = i
  rows, m, k = [], env.start_mode, 0
  while True:
    combos = env.get_valid_next_task_combinations(m)
    task = env.get_active_task(m, combos[0] if combos else None)
    q = goal_config(m, task)
    env.is_collision_free(q, m)                            # poses (mode + joints) as BiRRT does

    def _held_by(fname):                                   # FULL attach chain up to the root
      # the immediate parent is not enough: 'obj_2 on obj_0' looks innocent, but if obj_0 is
      # itself parented to a GRIPPER of a robot this task does not move, the mode check poses
      # that arm at HOME and the whole assembly travels with it -- which reads as a mystery
      # collision against the acting arm. Walk to the root so the culprit is visible.
      if fname not in env.C.getFrameNames():
        return ""
      chain, f, seen = [], env.C.getFrame(fname), set()
      while True:
        p = f.getParent()
        if p is None or p.name in seen:
          break
        seen.add(p.name)
        chain.append(p.name)
        f = p
      return " <- ".join(chain)

    pen = float(env.C.getCollisionsTotalPenetration())
    pairs = sorted(([c[0], c[1], float(c[2])] for c in env.C.getCollisions(margin)),
                   key=lambda x: x[2])
    rows.append((k, task.name, name_to_act.get(task.name), pen, pairs))
    if ref_scene is not None and pen > 1e-6 and name_to_idx_ref.get(task.name) is not None:
      # SCENE DRIFT: the same joint configuration can be clean in OUR scene and penetrating
      # here, because the two scenes derive the parts' poses differently -- ours seats a
      # table-resting group at its canonical rest pose, this one ACCUMULATES the actual attach
      # /mate operations from the start config. Print the parts that actually sit elsewhere;
      # that difference is the whole discrepancy, and it is invisible from either side alone.
      Cref = ref_scene(name_to_idx_ref[task.name])
      if Cref is not None:
        drift = []
        for f1, f2, _d in pairs[:6]:
          for fn in (f1, f2):
            if not fn.startswith("obj") or fn in [x[0] for x in drift]:
              continue
            try:
              a = np.asarray(env.C.getFrame(fn).getPosition(), dtype=float)
              b = np.asarray(Cref.getFrame(fn).getPosition(), dtype=float)
            except Exception:
              continue
            # RELATIVE to the part's own parent as well: a world delta cannot distinguish
            # "this part is mated wrong" (relative pose differs -> baked-in mate residual) from
            # "the mate is fine but the whole assembly sits elsewhere" (relative matches, world
            # does not). Those have completely different fixes.
            rel = float("nan")
            try:
              pa_ = env.C.getFrame(fn).getParent()
              pb_ = Cref.getFrame(fn).getParent()
              if pa_ is not None and pb_ is not None:
                ra = np.asarray(env.C.getFrame(fn).getRelativePosition(), dtype=float)
                rb = np.asarray(Cref.getFrame(fn).getRelativePosition(), dtype=float)
                rel = float(np.linalg.norm(ra - rb)) * 1000.0
            except Exception:
              pass
            drift.append((fn, (a - b) * 1000.0, float(np.linalg.norm(a - b)) * 1000.0, rel))
        for fn, dv, dd, rel in drift:
          # COMPONENTWISE on purpose: a spawn-height offset (--part-float) is purely VERTICAL,
          # accumulated mate residual is mixed-axis. The magnitude alone cannot tell them apart.
          # the parent must be named: a REL number computed against DIFFERENT parents in the
          # two scenes compares nothing at all (measured 178mm of "drift" that way).
          def _pn(Cx, f):
            try:
              q = Cx.getFrame(f).getParent()
              return q.name if q is not None else "-"
            except Exception:
              return "?"
          _pe, _po = _pn(env.C, fn), _pn(Cref, fn)
          # WHICH SCENE IS WRONG? Compare each against the IDEAL mate geometry: the part's
          # position relative to the assembly ROOT should equal assembly.position_in_base()
          # (parts are axis-aligned, so orientation stays identity). Whichever scene deviates
          # is the one to fix -- guessing that from a world delta alone is how I got this
          # backwards twice.
          try:
            _root = None
            for _cand in ("obj_0",):
              if _cand in env.C.getFrameNames():
                _root = _cand
            if _root is not None and plan is not None:
              _ideal = np.asarray(problem_assembly.position_in_base(fn), dtype=float) - \
                       np.asarray(problem_assembly.position_in_base(_root), dtype=float)
              for _tag, _Cx in (("exported", env.C), ("ours", Cref)):
                # EXPRESS IN THE ROOT'S OWN FRAME. The ideal offset is a base-frame vector;
                # the assembly is arbitrarily oriented in the world (always so when held), so
                # comparing a world-frame delta to it is meaningless -- that mistake reported
                # ~76-103mm of bogus error on BOTH scenes at once.
                _R = np.asarray(_Cx.getFrame(_root).getRotationMatrix(), dtype=float)
                _rp = _R.T @ (np.asarray(_Cx.getFrame(fn).getPosition(), dtype=float)
                              - np.asarray(_Cx.getFrame(_root).getPosition(), dtype=float))
                print(f"          vs IDEAL mate geometry [{_tag:8s}]: "
                      f"{np.linalg.norm(_rp - _ideal)*1000:7.2f}mm off")
          except Exception as _ee:
            pass
          _same = " (same parent)" if _pe == _po else f"  PARENTS DIFFER: exported '{_pe}' vs ours '{_po}'"
          print(f"        [scene-drift] {fn}: world {dd:.2f}mm "
                f"(dx={dv[0]:+.2f} dy={dv[1]:+.2f} dz={dv[2]:+.2f})  "
                f"rel {rel:.2f}mm{_same}")
    if verbose and (pen > 1e-6 or (pairs and pairs[0][2] < margin * 0.5)):
      tag = f"  {name_to_act.get(task.name)}" if name_to_act.get(task.name) else ""
      flag = "   <<< COLLIDES" if pen > 1e-6 else "   (near)"
      print(f"  [{k:2d}] {task.name:34s}{tag}  pen={pen*1000:6.2f}mm{flag}")
      for f1, f2, d in pairs[:6]:
        h1 = f"  [{f1} on {_held_by(f1)}]" if f1.startswith("obj") else ""
        h2 = f"  [{f2} on {_held_by(f2)}]" if f2.startswith("obj") else ""
        print(f"        {f1:26s} ~ {f2:26s} {d*1000:+8.2f}mm{h1}{h2}")
    if env.is_terminal_mode(m):
      break
    m = env.get_next_modes(q, m)[0]
    k += 1

  if verbose:
    ncol = sum(1 for r in rows if r[3] > 1e-6)
    print(f"[clearance] {ncol}/{len(rows)} task-goals collide (margin {margin*1000:.0f}mm, "
          f"mr_bench check, idle arms at HOME). '[obj_X on a?_ur_vacuum]' = obj_X is ATTACHED "
          f"to that gripper at this keyframe -- a part that should REST but is welded to an arm "
          f"is a re-rooting bug, not a geometry clash.")
  return rows


def _pose_to_T(pose7):
  """rai pose [x y z qw qx qy qz] -> 4x4 homogeneous. Local because the drift metric needs
  matrix composition, and P._quat_mul/_pose_invert only speak the 7-vector convention."""
  w, x, y, z = (float(v) for v in pose7[3:7])
  M = np.eye(4)
  M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
               [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
               [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
  M[:3, 3] = np.asarray(pose7[:3], dtype=float)
  return M


def _rel_deviation(T_ref, T):
  # (mm, deg) of T relative to T_ref. The translation norm is read in T_ref's frame, so it is
  # NOT symmetric under swapping the pair -- see the ANCHOR note in report_cograsp_drift.
  d = np.linalg.inv(T_ref) @ T
  ang = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
  return float(np.linalg.norm(d[:3, 3])) * 1000.0, float(ang)


def report_cograsp_drift(C, problem, mp_path, robot_ik, tol_mm=1.0, tol_deg=1.0, verbose=True):
  """CO-GRASP RIGIDITY along a solved motion path -- the companion diagnostic to
  report_problem_clearance (that one checks the task-goal KEYFRAMES; this one checks what
  happens BETWEEN them).

  Two grippers on one group are a CLOSED kinematic loop, but the exporter never closes it:
  add_grasp leaves as a bare type="goto" (see the TODO at the add_grasp branch of build_tasks)
  and a continuing holder is pinned only by a joint goal, so between keyframes both tool
  centres are welded to the SAME object yet free to move independently. This replays the
  planned path through rai FK and measures, per co-grasp span and per gripper pair, how far
  the pair's RELATIVE transform drifts from its value at the span's first path state -- a
  perfect brace holds it at 0.

  FRAMES: robot_ik.gripper_frame[r], the tool centre the grasp IK constrains. Both frames are
  robot-mounted, so their relative pose is a pure function of q: the attach mode is irrelevant
  and no re-parenting has to be replayed (unlike sim.execute_motion_plan).

  SEQUENCE problems only -- the path-state -> discrete-state map walks mode_sequence's linear
  mode chain (a depgraph linearization reorders task firings; same limitation as mode_sequence).

  Returns [(i0, i1, group, holders, anchor, other, max_mm, k_mm, max_deg, k_deg), ...]."""
  states = getattr(problem, "states", None)
  if states is None or not hasattr(problem, "sequence"):
    if verbose:
      print("[cograsp] needs a SEQUENCE problem carrying .states (--mp-sequence); skipping.")
    return []
  if not mp_path:
    return []
  robots = list(problem.robots)

  # private FK copy: only the ARM joints are written, the caller's C is untouched.
  C_fk = ry.Config()
  C_fk.addConfigurationCopy(C)
  jnames = C_fk.getJointNames()
  ridx = {r: [i for i, jn in enumerate(jnames) if jn.startswith(r)] for r in robots}
  q_base = np.asarray(C_fk.getJointState()).copy()

  def full_q(st):
    flat = st.q.state() if hasattr(st.q, "state") else np.asarray(st.q)
    q, o = q_base.copy(), 0
    for r in robots:
      n = len(ridx[r])
      q[ridx[r]] = flat[o:o + n]
      o += n
    return q

  # path state -> index into `states`. On a SEQUENCE problem the modes advance one task at a
  # time, so a state carrying the k-th mode of mode_sequence() has k tasks FIRED; a plan
  # ACTION is done once its LAST task fires (--pre-* splits one action into two tasks). A mode
  # that no path state carries (two goals met by one config) is simply skipped -- it lasted
  # zero path length, so nothing can drift inside it.
  mkey = lambda m: tuple(np.atleast_1d(m.task_ids).tolist())
  order = {mkey(m): k for k, m in enumerate(mode_sequence(problem))}
  tid = {t.name: i for i, t in enumerate(problem.tasks)}
  last = sorted(max(tid[nm] for nm in nms) for nms in problem.step_names if nms)
  done = [int(np.searchsorted(last, order[mkey(st.mode)], side="left")) for st in mp_path]

  def cograsps(state):
    # holders -> group, for every subassembly held by 2+ grippers
    out = {}
    for g in state.subassemblies():
      hs = tuple(sorted({c[1] for p in g for c in state.supports.get(p, ()) if c[0] == "grasp"}))
      if len(hs) >= 2:
        out[hs] = frozenset(g)
    return out

  # MAXIMAL spans: consecutive path states in which the SAME holder set co-holds the SAME
  # group. The group may GROW (an assemble merges the mated part in) without ending the span --
  # the grippers do not move at a mate, so the reference stays valid across it.
  spans, open_ = [], {}                                    # holders -> [i0, group-so-far]
  for k in range(len(mp_path)):
    cg = cograsps(states[done[k]])
    for hs in list(open_):
      g = cg.get(hs)
      if g is None or not (g & open_[hs][1]):
        spans.append((open_[hs][0], k - 1, open_.pop(hs)[1], hs))
      else:
        open_[hs][1] = open_[hs][1] | g
    for hs, g in cg.items():
      open_.setdefault(hs, [k, g])
  for hs, (i0, g) in open_.items():
    spans.append((i0, len(mp_path) - 1, g, hs))
  spans.sort()

  # ANCHOR: the drift's translation norm is read in ONE gripper's frame, so it is not
  # symmetric in the pair (the rotation angle is). Anchor on the INCUMBENT -- the arm whose
  # UNINTERRUPTED grasp of this group started earliest -- so the number reads "how far the
  # newcomer (the brace) wanders in the frame of the arm that owns the object".
  def holds(state, r, grp):
    return any(c[0] == "grasp" and c[1] == r for p in grp for c in state.supports.get(p, ()))

  rows = []
  for i0, i1, grp, hs in spans:
    grp0 = cograsps(states[done[i0]])[hs]                  # the group as of the span START

    def since(r):
      j = done[i0]
      while j > 0 and holds(states[j - 1], r, grp0):
        j -= 1
      return j

    anchor = min(hs, key=lambda r: (since(r), r))
    Ts = []
    for k in range(i0, i1 + 1):
      C_fk.setJointState(full_q(mp_path[k]))
      Ts.append({r: _pose_to_T(C_fk.getFrame(robot_ik.gripper_frame[r]).getPose()) for r in hs})
    for other in [r for r in hs if r != anchor]:
      ref = np.linalg.inv(Ts[0][anchor]) @ Ts[0][other]
      dev = [_rel_deviation(ref, np.linalg.inv(T[anchor]) @ T[other]) for T in Ts]
      k_mm = int(np.argmax([v[0] for v in dev]))
      k_deg = int(np.argmax([v[1] for v in dev]))
      rows.append((i0, i1, grp, hs, anchor, other,
                   dev[k_mm][0], i0 + k_mm, dev[k_deg][1], i0 + k_deg))

  if verbose:
    bad = [r for r in rows if r[6] > tol_mm or r[8] > tol_deg]
    print(f"[cograsp] {len(spans)} co-grasp span(s) over {len(mp_path)} path states; "
          f"{len(bad)}/{len(rows)} gripper pair(s) drift past {tol_mm:g}mm / {tol_deg:g}deg")
    for i0, i1, grp, hs, anchor, other, mm, kmm, deg, kdeg in rows:
      acts = sorted({done[k] for k in range(i0, i1 + 1)})
      flag = "   <<< NOT RIGID" if (mm > tol_mm or deg > tol_deg) else ""
      print(f"  path[{i0:3d}..{i1:3d}]  after actions {acts[0]}..{acts[-1]}  "
            f"{{{','.join(sorted(grp))}}} held by {'+'.join(hs)}  "
            f"{other} rel {anchor}: max {mm:8.1f}mm @path[{kmm}], "
            f"{deg:6.1f}deg @path[{kdeg}]{flag}")
    if bad:
      print("[cograsp] a co-grasp is a CLOSED kinematic loop the export does not close "
            "(add_grasp -> type=\"goto\"): these are two grippers welded to ONE object moving "
            "independently. Deviation is measured from the span's FIRST path state and is 0 "
            "there by construction.")
  return rows
