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


def build_tasks(assembly, plan, witnesses, states, robots, robot_idx, robot_ik,
                home_state):
  """Map the plan+witnesses to (tasks, sequence_names). robot_idx: {r: [joint indices]}
  in the problem config's joint order; witnesses[i] is the full-config keyframe for plan[i]
  (None for non-IK actions -> skipped, no motion goal needed)."""
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
    pre_wit = robot_ik.pre_skill_witness(a, states[i], states[i + 1], assembly, wit)
    pan = f"{i:02d}_pre_{kind}_{'_'.join(str(x) for x in a[1:])}"
    pre_goal = goal_for(involved, pre_wit if pre_wit is not None else wit)   # cur := T_pre
    return [(pan, Task(pan, involved, pre_goal, type="goto")),
            (name, mk_skill(goal_for(involved, wit)))]                       # cur := real witness

  tasks, seq, step_names = [], [], []                             # step_names[i] = names of plan step i
  for i, a in enumerate(plan):
    kind, r = a[0], a[1]
    wit = witnesses[i]
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


def make_problem(C_start, assembly, plan, witnesses, robots, robot_ik, gk, bracing,
                 depgraph=False):
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
          assembly, plan, witnesses, states, self.robots, self.robot_idx, robot_ik, home)
      if depgraph:
        self.graph = _dependency_graph(plan, states, step_names, self.robots, robot_ik)
      else:
        self.sequence = self._make_sequence_from_names(seq)   # flat order (incl. terminal)
      self.collision_tolerance = 0.01
      BaseModeLogic.__init__(self)
      self.prev_mode = self.start_mode

  return AssemblyEnv()


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
              shortcut_resolution=None, optimize=False, verbose=True):
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
  from multi_robot_multi_goal_planning.planners.planner_birrtstar import BidirectionalRRTstar
  from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
  from multi_robot_multi_goal_planning.planners.shortcutting import robot_mode_shortcut
  from multi_robot_multi_goal_planning.planners.termination_conditions import (
      RuntimeTerminationCondition)
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
  path, info = BidirectionalRRTstar(problem, config=cfg).plan(
      ptc=RuntimeTerminationCondition(runtime), optimize=optimize)
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


def save_path(path_json, path):
  with open(path_json, "w") as f:
    json.dump([s.to_dict() for s in path] if path else [], f, indent=2)
