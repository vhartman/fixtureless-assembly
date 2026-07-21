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
import os
import sys
import json

import numpy as np

_MRBENCH_SRC = "/home/valentin/git/postdoc/robotic-venv/mr_bench/src"


def _ensure_mrbench():
  if _MRBENCH_SRC not in sys.path:
    sys.path.insert(0, _MRBENCH_SRC)
  # STUB the mr_planner_core compiled extension (a VAMP planner built for py3.10; our venv
  # is py3.12). problems/__init__.py eagerly imports mr_vamp_env behind find_spec(...) --
  # the 3.10 package IS found but its .so fails to load. We never use the VAMP env, so a
  # bare stub module (find_spec returns its __spec__, `import mr_planner_core` gets it) lets
  # the package import cleanly. No mr_bench change.
  import types
  import importlib.util as _ilu
  for _name in ("mr_planner_core",):                        # the compiled ext itself
    if _name not in sys.modules:
      _m = types.ModuleType(_name)
      _m.__spec__ = _ilu.spec_from_loader(_name, loader=None)
      sys.modules[_name] = _m
  # STUB the unused VAMP submodule wholesale: problems/__init__ does `from . import
  # mr_vamp_env` (its find_spec(mr_planner_core) guard passes since the 3.10 package is on
  # disk), and mr_vamp_env's body imports mr_planner_core's .so AND meshcat. We never use
  # it, so pre-seed sys.modules with an empty module under its full name -> the import binds
  # the stub and never runs the body. No mr_bench change.
  _vamp = "multi_robot_multi_goal_planning.problems.mr_vamp_env"
  if _vamp not in sys.modules:
    _mv = types.ModuleType(_vamp)
    _mv.__spec__ = _ilu.spec_from_loader(_vamp, loader=None)
    sys.modules[_vamp] = _mv
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
  s = A.get_initial_state(C, assembly)
  states = [s]
  for a in plan:
    s = next(ns for act, ns in A.successors(s, assembly, robots, gripper_kind=gk,
                                            handovers=True, allow_place=True,
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

  tasks, seq = [], []
  for i, a in enumerate(plan):
    kind, r = a[0], a[1]
    wit = witnesses[i]
    name = f"{i:02d}_{kind}_{'_'.join(str(x) for x in a[1:])}"
    # ALL robots whose witness pose must be applied this keyframe: the actor PLUS the
    # holder(s) of the group the action moves (e.g. an assemble's `onto` group is held by
    # another arm). Declaring only the actor left the idle holder homed in mr_bench -> the
    # held part (obj_0) snapped to that arm's home pose and every downstream mate misaligned.
    # `goal_for` poses each in task-robot order at its witness config, pinning the holder.
    involved = _active_robots(a, states[i], robot_ik) or [r]
    if kind == "pick":
      t = Task(name, involved, goal_for(involved, wit), type="pick",
               frames=[robot_ik.gripper_frame[r], _root_of(states[i], a[2], mate_parent)])
    elif kind == "place":
      t = Task(name, involved, goal_for(involved, wit), type="place",
               frames=["table", _root_of(states[i], a[2], mate_parent)])
    elif kind == "assemble":
      t = Task(name, involved, goal_for(involved, wit), type="place", frames=[a[3], a[2]])
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
      # TODO(mp): a brace should be a RELATIVE-POSE CONSTRAINT gripper<->object, not a
      # goto keyframe (other mr_bench branch). goto = motion to the brace pose for now.
      t = Task(name, involved, goal_for(involved, wit), type="goto")
    elif kind == "handover":
      rB = r
      rA = _holder_of(states[i], a[2])
      task_robots = [rA, rB] if rA is not None and rA != rB else [rB]
      t = Task(name, task_robots, goal_for(task_robots, wit), type="handover",
               frames=[robot_ik.gripper_frame[rB], _root_of(states[i], a[2], mate_parent)])
    else:
      continue
    tasks.append(t)
    seq.append(name)

  # terminal: all robots return home (a clean, always-reachable end mode)
  from multi_robot_multi_goal_planning.problems.planning_env import Task as _T
  from multi_robot_multi_goal_planning.problems.core.goals import SingleGoal as _SG
  term = _T("terminal", list(robots),
            _SG(np.concatenate([np.asarray(home_state)[robot_idx[r]] for r in robots])))
  tasks.append(term)
  seq.append("terminal")
  return tasks, seq


# ---------------------------------------------------------------------------
# problem instance
# ---------------------------------------------------------------------------
def _dependency_graph(plan, states, seq_names, robots, robot_ik):
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
  for nm in seq_names:
    g.add_node(nm)
  last = {}                                                  # resource -> most-recent task name
  for i, a in enumerate(plan):
    nm = seq_names[i]
    res = set(_active_robots(a, states[i], robot_ik) or [a[1]])
    named = [a[2]] + ([a[3]] if a[0] == "assemble" else [])
    for st in (states[i], states[i + 1]):
      for p in named:
        res |= set(next((grp for grp in st.subassemblies() if p in grp), {p}))
    for r in res:
      if last.get(r) not in (None, nm):
        g.add_dependency(nm, last[r])                        # nm depends on last writer of r
      last[r] = nm
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
      # RE-PARENT the robotiq's finger collision frames (finger1/2 capsules + inner-finger
      # PAD boxes) onto the surviving robotiq base link. Their kinematic PARENT is a MESH
      # frame, and rai_env.__init__'s delete_visual_only_frames deletes every mesh frame ->
      # the finger collision shapes ORPHAN (parent=NONE) and FLOAT in the air at the gripper's
      # start height. We keep the shapes (real finger collision for transit) but hang them off
      # a non-mesh ancestor (*_ur_robotiq_arg2f_base_link, ST.none, survives) so they follow
      # the gripper. unLink()+setParent(...,True) preserves each frame's WORLD pose exactly,
      # so the fingers stay put at their open-gripper location (fixed in transit). Done BEFORE
      # rai_env.__init__ so nothing orphans.
      for _f in list(self.C.getFrameNames()):
        if (_f.endswith("_ur_finger1") or _f.endswith("_ur_finger2")
            or "_inner_finger_pad" in _f):
          _pre = _f.split("_ur_")[0]
          _base = self.C.getFrame(_pre + "_ur_robotiq_arg2f_base_link")
          if _base is not None:
            self.C.getFrame(_f).unLink()
            self.C.getFrame(_f).setParent(_base, True)
      # Give SHAPELESS part frames a PRIMITIVE collision box (car-mini etc.). A part `obj_X`
      # may be ST.none with its collision geometry on a rigid `obj_X__shape` child (a rotated
      # cylinder wheel). mr_bench's set_to_mode does setContact(-1) on the attached PART frame;
      # on a shapeless frame rai fabricates a DEGENERATE ST.mesh (0 vertices) -> FCL
      # 'collision object with no vertices' crash in computeCollisions. We give obj_X an ssBox
      # = the AXIS-ALIGNED BOUNDING BOX of the child's mesh (transformed by its relative pose)
      # in the part frame: an ssBox is a PRIMITIVE (info shape != "mesh") so it survives
      # rai_env's delete_visual_only_frames (which deletes every "mesh" frame -- a plain
      # setMesh here would get obj_X itself DELETED). The child's own contact is dropped so the
      # part carries the single collision proxy. Cross parts (ST.ssBox, not shapeless) skip.
      def _quat_R(q):
        w, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                         [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
      for _sf in [f for f in self.C.getFrameNames() if f.endswith("__shape")]:
        _pf = self.C.getFrame(_sf[:-len("__shape")])
        _s = self.C.getFrame(_sf)
        if _pf is None or str(_pf.getShapeType()) != "ST.none":
          continue
        _V = np.asarray(_s.getMeshPoints())
        if _V.ndim < 2 or len(_V) == 0:
          continue
        _Vp = _V @ _quat_R(np.asarray(_s.getRelativeQuaternion())).T \
            + np.asarray(_s.getRelativePosition())
        _e = _Vp.max(0) - _Vp.min(0)                         # AABB full extents
        # ssBox sits at the part-frame origin, valid because the child shape is CENTRED there
        # (car wheels: AABB centre ~0). An off-centre shape would need a child frame instead.
        _pf.setShape(ry.ST.ssBox, [float(_e[0]), float(_e[1]), float(_e[2]), 0.002])
        _pf.setContact(1)
        _s.setContact(0)
      self.robots = list(robots)
      rai_env.__init__(self)                               # -> robot_idx, start_pos, limits
      self.manipulating_env = True
      self.tasks, seq_names = build_tasks(
          assembly, plan, witnesses, states, self.robots, self.robot_idx, robot_ik, home)
      if depgraph:
        self.graph = _dependency_graph(plan, states, seq_names, self.robots, robot_ik)
      else:
        self.sequence = self._make_sequence_from_names(seq_names)
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


def _patch_mrbench_numpy2():
  """Runtime compat shim (NOT a source edit to mr_bench). numpy>=2 -- our venv is 2.4, pinned
  by numba -- raises 'setting an array element with a sequence' when a size-1 array is assigned
  into a scalar array slot; numpy<2 (what mr_bench targets) broadcast it silently.
  env.batch_config_cost returns a size-1 array for single-pair queries, which BiRRT's
  extend/connect write straight into the scalar Node cost array. Every Node cost write funnels
  through Node.cost.setter -> operation.costs[id] = value, so coercing size-1 -> float there
  fixes all sites. Pure compatibility; no numeric change. Recommend an mr_bench-side fix later."""
  import numpy as _np
  from multi_robot_multi_goal_planning.planners import rrtstar_base as _rb
  if getattr(_rb.Node, "_np2_cost_patched", False):
    return
  _prop = _rb.Node.cost
  def _set(self, value):
    v = _np.asarray(value)
    self.operation.costs[self.id] = v.reshape(-1)[0] if v.size == 1 else value
  _rb.Node.cost = property(_prop.fget, _set)
  _rb.Node._np2_cost_patched = True


def _patch_mrbench_optimize():
  """Runtime shim (NOT a source edit to mr_bench) so optimize=True doesn't crash. In
  BaseRRTstar.tree_extension the path-improvement loop calls
  `near(mode, node, np.where(get_node_ids_subtree()==parent.id))` -- passing the np.where
  TUPLE as the scalar `n_nearest_idx`. When `parent` lives in the OTHER bidirectional subtree
  (A vs B, or a swapped tree) the lookup is EMPTY, and near()'s `n_nearest_idx not in indices`
  then broadcasts an empty (1,0) array against the near-set -> ValueError (numpy>=2). We patch
  BaseRRTstar.near to coerce n_nearest_idx to a scalar and, when the parent isn't in this
  subtree, simply not force-insert it (the radius near-set still drives find_parent/rewire, so
  optimization proceeds, just without pinning that one absent parent). Only reached with
  optimize=True. Proper mr_bench fix: search the subtree matching tree.order and handle the
  not-found case in tree_extension."""
  import numpy as _np
  from multi_robot_multi_goal_planning.planners import rrtstar_base as _rb
  if getattr(_rb.BaseRRTstar, "_near_idx_patched", False):
    return

  def _near(self, mode, n_new, n_nearest_idx, set_dists=None, tree=""):
    batch_subtree = self.trees[mode].get_batch_subtree(tree)
    if set_dists is None:
      set_dists = _rb.batch_config_dist(n_new.state.q, batch_subtree,
                                        self.config.distance_metric)
    vertices = self.trees[mode].get_number_of_nodes_in_tree()
    r = _np.minimum(self.gamma_rrtstar * (_np.log(vertices) / vertices) ** (1 / self.d),
                    self.eta)
    indices = _rb.find_nearest_indices(set_dists, r)
    # coerce np.where(...) tuple / array -> scalar; skip the force-insert if absent (empty)
    _ix = n_nearest_idx[0] if isinstance(n_nearest_idx, tuple) else n_nearest_idx
    _ix = _np.ravel(_np.asarray(_ix))
    if _ix.size and int(_ix[0]) not in indices:
      indices = _np.insert(indices, 0, int(_ix[0]))
    node_indices = self.trees[mode].get_node_ids_subtree(tree)[indices]
    n_near_costs = self.operation.costs[node_indices]
    N_near_batch = batch_subtree[indices]
    return N_near_batch, n_near_costs, node_indices

  _rb.BaseRRTstar.near = _near
  _rb.BaseRRTstar._near_idx_patched = True


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
  _patch_mrbench_numpy2()
  _patch_mrbench_optimize()                                # safe no-op for scalar near() calls
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
