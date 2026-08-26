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
  release(r,p)       -> goto   (zero-motion detach; the released object is re-parented)
  add_grasp(r,p,g)   -> goto   (standalone: ordinary goal; before assemble: shared
                                all-arm funnel rendezvous, coupled through its releases)
  handover(rB,p,g)   -> handover, robots=[rA_giver, rB], frames=[gripper(rB), p]
"""
import sys
import json
import os as _os

import numpy as np

_MRBENCH_SRC = _os.environ.get(
    "MRBENCH_SRC",
    _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "mr_bench", "src")),
)


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


class _PairedGoal:
  """A goal whose member is selected by the mode's entry configuration.

  Two independent GoalSets do NOT make a funnel pair: after reaching pre member A the planner
  may freely sample skill member B. This conditional goal binds each skill/post member to the
  corresponding pre/release member, using only the task robots' indices in the full entry
  configuration. It is deliberately local rather than an mr_bench patch and is fully persisted
  by the final-export task records below.
  """
  def __init__(self, conditions, goals, condition_indices):
    self.conditions = [np.asarray(q, dtype=float).ravel() for q in conditions]
    self.goals = [np.asarray(q, dtype=float).ravel() for q in goals]
    self.condition_indices = np.asarray(condition_indices, dtype=int).ravel()
    if len(self.conditions) != len(self.goals) or not self.goals:
      raise ValueError("paired goal needs the same non-zero number of conditions and goals")

  @staticmethod
  def _flat_entry(mode):
    q = mode.entry_configuration
    if hasattr(q, "state"):
      return np.asarray(q.state(), dtype=float).ravel()
    return np.asarray(q, dtype=float).ravel()

  def _member(self, mode, tolerance=1e-7):
    entry = self._flat_entry(mode)[self.condition_indices]
    dist = [float(np.linalg.norm(entry - c)) for c in self.conditions]
    k = int(np.argmin(dist))
    if dist[k] > tolerance:
      raise ValueError(
          f"No paired goal for mode entry (nearest condition is {dist[k]:.3g} away)")
    return k

  def satisfies_constraints(self, q, mode, tolerance):
    try:
      k = self._member(mode, max(float(tolerance), 1e-7))
    except ValueError:
      return False
    return float(np.linalg.norm(np.asarray(q, dtype=float).ravel() - self.goals[k])) < tolerance

  def sample(self, mode):
    return self.goals[self._member(mode)]

  def serialize(self):
    return {"conditions": [q.tolist() for q in self.conditions],
            "goals": [q.tolist() for q in self.goals],
            "condition_indices": self.condition_indices.tolist()}


def build_tasks(assembly, plan, witnesses, states, robots, robot_idx, robot_ik,
                home_state, pre_witnesses=None, witness_sets=None,
                resolved_pre_witnesses=None, resolve_missing_pre=True,
                pre_pin_robots=None):
  """Map the plan+witnesses to (tasks, sequence_names). robot_idx: {r: [joint indices]}
  in the problem config's joint order; witnesses[i] is the full-config keyframe for plan[i]
  (None for non-IK actions -> skipped, no motion goal needed).

  `pre_witnesses` (--fuse-pre-skill, {plan index: keyframe or None}) supplies the pre-* GOTO
  keyframes ready-made, from solver.fuse_pre_skill_witnesses. None (the default) keeps the
  historical behaviour: each split solves its own pre keyframe here, inside the export. A
  missing SKILL witness, by contrast, is a hard error -- it cannot be substituted.

  `pre_pin_robots` records the idle arms that an ACCEPTED fused pre-keyframe held fixed.  Those
  arms participate in that GOTO only, so the mode switch uses the known-safe full keyframe
  instead of randomizing joints whose safety was already established by the fused solve.

  `witness_sets` (--witness-sets, {plan index: [full vectors, member 0 == witnesses[i]]}):
  the covered task's goal becomes an mr_bench GoalSet over the members. Every member shares
  the grasp AND the grasped parts' world poses with the canonical witness, so every attach --
  which mr_bench fires at the ACHIEVED configuration -- is member-invariant, and all members
  hash to the SAME mode (no mode-graph growth). A later no-witness task (release,
  inhand_reorient) that pins arms last moved by a set-bearing task MIRRORS the set: member m
  pins each arm at ITS OWN member-m sub-vector (aligned whole-witness slices, never mixed),
  so the diversity survives past the release instead of collapsing back to member 0. A pre-*
  split preserves the diversity as PAIRED GoalSets: funnel member m leads to skill member m;
  failed alternative funnels drop that pair rather than mixing IK branches."""
  from multi_robot_multi_goal_planning.problems.planning_env import Task
  from multi_robot_multi_goal_planning.problems.core.goals import SingleGoal, GoalSet

  # cur[r] = robot r's CURRENT joint sub-vector (start: home). A witness updates the
  # ACTIVE arm(s); a no-witness / release action leaves the arm where it was -- so its goal
  # keeps it AT ITS GRASP, which is what makes a release's re-parent capture the object at
  # its true (held) pose rather than at home.
  cur = {r: np.asarray(home_state)[robot_idx[r]].copy() for r in robots}
  # cur_alt[r] = (source plan index, [per-member sub-vectors]) when r was last moved by a
  # set-bearing task; reset whenever a set-less task moves r. Consumed by the mirroring below.
  cur_alt = {r: None for r in robots}
  # Local cache because a cograsp rendezvous may need the following assemble's pre-witness
  # BEFORE the exporter reaches that assemble step. A supplied mapping remains caller-owned.
  pre_cache = dict(pre_witnesses or {})

  def require_or_solve_pre(i, action, witness):
    """Return the exact pre-skill keyframe, solving only during a FRESH export.

    A loaded final export sets ``resolve_missing_pre=False``.  In that mode a missing entry is
    corruption/incompatibility, never permission to call KOMO behind a viewer's back.
    """
    qpre = pre_cache.get(i, _SOLVE_IT)
    if qpre is _SOLVE_IT:
      if not resolve_missing_pre:
        raise ValueError(
            f"mp_export: final export is missing pre-skill keyframe for plan step {i} "
            f"{tuple(action)}; refusing to re-solve IK during replay")
      qpre = robot_ik.pre_skill_witness(
          action, states[i], states[i + 1], assembly, witness)
      pre_cache[i] = qpre
    return qpre

  # mate tree exactly as mr_bench builds it: each assemble(r, part, onto) -> attach(onto, part)
  mate_parent = {a[2]: a[3] for a in plan if a[0] == "assemble"}

  def goal_for(rs, wit, alts=None):
    parts = []
    for r in rs:
      if wit is not None:
        cur[r] = np.asarray(wit)[robot_idx[r]]             # active arm -> witness config
        cur_alt[r] = None
      parts.append(cur[r])
    return SingleGoal(np.concatenate(parts))

  def goal_for_set(rs, members, src_index):
    """GoalSet over aligned member slices; canonical member 0 feeds cur (so set-less
    downstream consumers see exactly the single-witness behaviour)."""
    goals = []
    for m, w in enumerate(members):
      goals.append(np.concatenate([np.asarray(w)[robot_idx[r]] for r in rs]))
    for r in rs:
      cur[r] = np.asarray(members[0])[robot_idx[r]]
      cur_alt[r] = (src_index, [np.asarray(w)[robot_idx[r]].copy() for w in members])
    return GoalSet(goals)

  def goal_for_paired(rs, conditions, members, src_index):
    """Conditional member-m goal after a member-m entry, and update canonical bookkeeping."""
    cond = [np.concatenate([np.asarray(w)[robot_idx[r]] for r in rs]) for w in conditions]
    goals = [np.concatenate([np.asarray(w)[robot_idx[r]] for r in rs]) for w in members]
    indices = np.concatenate([np.asarray(robot_idx[r], dtype=int) for r in rs])
    for r in rs:
      cur[r] = np.asarray(members[0])[robot_idx[r]]
      cur_alt[r] = (src_index, [np.asarray(w)[robot_idx[r]].copy() for w in members])
    return _PairedGoal(cond, goals, indices)

  def mirrored_pin(rs):
    """Goal for a no-witness task pinning `rs`: a GoalSet mirroring the members when every
    set-bearing arm shares ONE source task (aligned slices), else the canonical SingleGoal."""
    srcs = {cur_alt[r][0] for r in rs if cur_alt[r] is not None}
    if len(srcs) == 1:
      n = len(next(cur_alt[r] for r in rs if cur_alt[r] is not None)[1])
      goals = [np.concatenate([(cur_alt[r][1][m] if cur_alt[r] is not None else cur[r])
                               for r in rs]) for m in range(n)]
      return GoalSet(goals)
    if len(srcs) > 1:
      print("[witsets] pin over arms from DIFFERENT set sources "
            f"({sorted(srcs)}): falling back to the canonical pin.")
    return SingleGoal(np.concatenate([cur[r] for r in rs]))

  def current_full_members(rs):
    """Aligned full configurations represented by the current GoalSet source for ``rs``."""
    base = np.asarray(home_state, dtype=float).copy()
    for rr in robots:
      base[robot_idx[rr]] = cur[rr]
    srcs = {cur_alt[rr][0] for rr in rs if cur_alt[rr] is not None}
    if len(srcs) != 1:
      return [base]
    src = next(iter(srcs))
    n = len(next(cur_alt[rr] for rr in rs if cur_alt[rr] is not None)[1])
    out = []
    for m in range(n):
      q = base.copy()
      for rr in robots:
        alt = cur_alt[rr]
        if alt is not None and alt[0] == src:
          q[robot_idx[rr]] = alt[1][m]
      out.append(q)
    return out

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
  post_release = getattr(robot_ik, "post_release_delta", 0.0) > 0.0 # --post-release: split departure
  # An add_grasp that directly braces a later assemble is not an independent destination.
  # It is the start of one choreography: rendezvous/add brace -> short insertion -> release.
  # The structural relation is shared with solver.fuse_cograsp_witnesses so the witness pass
  # and task export cannot silently pair different actions.
  from solver import cograsp_assembly_pairs
  cograsp_pairs = cograsp_assembly_pairs(plan, states, robot_ik)
  cograsp_chain = {i: j for i, j, _r, _p, _g in cograsp_pairs}
  # Some valid plans switch the braced target's support before inserting (Cross does
  # add_grasp -> place -> assemble).  The robots holding the target before that switch meet at
  # one physically valid place keyframe; only holders that PERSIST across the switch remain
  # pinned for assembly.  The released holder must be allowed to retract so the inserter can
  # approach -- pinning the union of both sides made Cross's a1/a2 overlap by 106mm.
  cograsp_bridge = {}
  add_robots = {}
  assembly_robots = {}
  coupled_release_robots = {}
  for _i, j, _r, _p, _g in cograsp_pairs:
    ca = plan[j]
    _add_rs = _active_robots(plan[_i], states[_i], robot_ik) or [plan[_i][1]]
    _asm_rs = _active_robots(ca, states[j], robot_ik) or [ca[1]]
    add_robots[j] = list(dict.fromkeys(add_robots.get(j, []) + _add_rs))
    assembly_robots[j] = list(dict.fromkeys(assembly_robots.get(j, []) + _asm_rs))
    for k in range(_i + 1, j):
      old = cograsp_bridge.setdefault(k, j)
      if old != j:
        raise ValueError(
            f"overlapping add_grasp chains assign plan step {k} to assemblies {old} and {j}")
  bridged = set(cograsp_bridge.values())
  rendezvous_robots = {
      j: (list(add_robots[j]) if j in bridged else
          list(dict.fromkeys(add_robots[j] + assembly_robots[j])))
      for j in assembly_robots}
  persistent_robots = {
      j: [r for r in add_robots[j] if r in assembly_robots[j]] for j in assembly_robots}
  for j in assembly_robots:
    # Every immediately-following release is part of the atomic choreography. In particular,
    # the SECOND release must still pin the already-released inserter: otherwise mr_bench can
    # start that arm's next task early and detour a holder to avoid it while the co-grasp is
    # nominally closed (measured on car-mini as a 23.1mm loop between two equal endpoints).
    k = j + 1
    while k < len(plan) and plan[k][0] == "release":
      coupled_release_robots[k] = list(dict.fromkeys(
          coupled_release_robots.get(k, []) + list(assembly_robots[j])))
      k += 1
  if cograsp_chain:
    print(f"[cograsp-chain] coupled {len(cograsp_chain)} add_grasp task(s) to the following "
          "assembly rendezvous.")

  chain_goal_cache = {}

  def chain_goal(j):
    """Full-config rendezvous for one add_grasp -> assemble chain.

    For a direct chain, the assemble pre-witness is the all-arm rendezvous.  Across a support
    switch, persistent holders take that pre-witness while a holder released by the switch
    takes its place-witness slice.  The inserter is deliberately NOT pulled into the earlier
    add/place task: it approaches only after the former holder is free to retract.
    """
    if j in chain_goal_cache:
      return chain_goal_cache[j]
    ca = plan[j]
    qpre = require_or_solve_pre(j, ca, witnesses[j]) if pre_ins else witnesses[j]
    if qpre is None:
      chain_goal_cache[j] = None
      return None
    q = np.asarray(qpre, dtype=float).copy()
    former = set(rendezvous_robots[j]) - set(persistent_robots[j]) if j in bridged else set()
    if former:
      for k in sorted(k for k, jj in cograsp_bridge.items() if jj == j):
        wk = witnesses[k]
        if wk is None:
          continue
        step_robots = set(_active_robots(plan[k], states[k], robot_ik) or [plan[k][1]])
        for rr in former & step_robots:
          q[robot_idx[rr]] = np.asarray(wk, dtype=float)[robot_idx[rr]]
    chain_goal_cache[j] = q
    return q

  def split_skill(a, i, wit, involved, name, kind, mk_skill, members=None):
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
    pre_wit = require_or_solve_pre(i, a, wit)
    pan = f"{i:02d}_pre_{kind}_{'_'.join(str(x) for x in a[1:])}"
    if pre_wit is None and i in pre_cache:
      # the fallback is silent in the historical path; once the consistency pass owns these
      # keyframes, say it -- a None here exports a pre GOTO whose goal EQUALS the skill goal.
      print(f"  [pre] {i:02d} {kind}: NO pre keyframe -- the goto goal equals the skill goal "
            f"(zero-length funnel)")
    if not members:
      pin_rs = list((pre_pin_robots or {}).get(i, ()))
      pre_involved = list(dict.fromkeys(list(involved) + pin_rs))
      pre_goal = goal_for(pre_involved,
                          pre_wit if pre_wit is not None else wit)           # cur := T_pre
      return [(pan, Task(pan, pre_involved, pre_goal, type="goto")),
              (name, mk_skill(goal_for(involved, wit)))]                     # cur := real witness

    # GoalSets used to be silently discarded by every pre-skill split. Solve one funnel mouth
    # per skill member so the planner keeps both branch diversity and a short, branch-consistent
    # contact approach. Member 0 retains the canonical pre-witness (including any fused repair).
    # Alternatives pin arms which should stay still; add_grasp pins the existing holders too,
    # so only the arriving hand pays for the backoff.
    if not resolve_missing_pre:
      raise ValueError(
          f"mp_export: task records are required to replay paired pre-skill GoalSets at "
          f"plan step {i} {tuple(a)}; refusing to solve alternative funnels during replay")
    joint_names = list(robot_ik.C.getJointNames())
    moving = set(involved if kind == "place" else [a[1]])
    pre_members = [np.asarray(pre_wit if pre_wit is not None else members[0], dtype=float)]
    skill_members = [np.asarray(members[0], dtype=float)]
    for m, member in enumerate(members[1:], 1):
      member = np.asarray(member, dtype=float)
      pin = {n: float(member[k]) for rr in robots if rr not in moving
             for k, n in enumerate(joint_names) if n.startswith(rr + "_")}
      qpre = robot_ik.pre_skill_witness(
          a, states[i], states[i + 1], assembly, member, pin=pin or None)
      if qpre is None:
        print(f"[witsets] step {i} {kind}: alternative {m} has no feasible paired funnel -- "
              "dropped from BOTH GoalSets.")
        continue
      pre_members.append(np.asarray(qpre, dtype=float))
      skill_members.append(member)
    if len(skill_members) < len(members):
      print(f"[witsets] step {i} {kind}: {len(skill_members)}/{len(members)} paired "
            "pre-skill branch(es) retained.")
    if len(skill_members) >= 2:
      pre_goal = goal_for_set(involved, pre_members, i)                     # cur := pre set
      skill_goal = goal_for_paired(involved, pre_members, skill_members, i) # same member only
    else:
      pre_goal = goal_for(involved, pre_members[0])
      skill_goal = goal_for(involved, skill_members[0])
    return [(pan, Task(pan, involved, pre_goal, type="goto")),
            (name, mk_skill(skill_goal))]

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
    goal_wit = wit
    chain_j = cograsp_chain.get(i)
    bridge_j = cograsp_bridge.get(i)
    shared_j = chain_j if chain_j is not None else bridge_j
    shared_robots = []
    if shared_j is not None:
      # A direct chain brings every assembly participant to the funnel before add_grasp.  A
      # bridged chain instead pins only the pre-switch target holders through its place event;
      # the inserter approaches afterwards and the released holder may retract.
      shared_robots = rendezvous_robots[shared_j]
      involved = list(dict.fromkeys(list(involved) + list(shared_robots)))
      rendezvous = chain_goal(shared_j)
      if rendezvous is not None:
        if bridge_j is None:
          goal_wit = rendezvous
        else:
          # Hybrid full witness: chain participants take the shared rendezvous slices; any
          # unrelated actor keeps this bridge action's own witness (or its current config for
          # a no-witness side effect).  Assigning this to `wit` below makes every ordinary task
          # branch consume the pinned goal without duplicating the action-specific exporter.
          goal_wit = np.asarray(rendezvous, dtype=float).copy()
          source = None if wit is None else np.asarray(wit, dtype=float)
          for rr in robots:
            if rr not in shared_robots:
              goal_wit[robot_idx[rr]] = cur[rr] if source is None else source[robot_idx[rr]]
          wit = goal_wit
    # --witness-sets: members for THIS step, validated against the shipped canonical witness.
    smembers = None
    if witness_sets and i in witness_sets and wit is not None:
      _ms = witness_sets[i]
      if len(_ms) >= 2 and np.allclose(np.asarray(_ms[0]), np.asarray(wit), atol=1e-9):
        smembers = _ms
      elif len(_ms) >= 2:
        print(f"[witsets] step {i} {kind}: member 0 no longer matches the canonical witness "
              f"(rewritten by a later pass) -- set dropped.")
    if kind == "pick":
      # --pre-pick splits into a goto to the retracted pre-grasp + the real pick (grasp+attach).
      frames = [robot_ik.gripper_frame[r], _root_of(states[i], a[2], mate_parent)]
      mk = lambda g: Task(name, involved, g, type="pick", frames=frames)
      emit = (split_skill(a, i, wit, involved, name, kind, mk, smembers)
              if pre_pick and bridge_j is None else None)
      if emit is None:
        t = mk(goal_for_set(involved, smembers, i) if smembers
               else goal_for(involved, wit))
    elif kind == "place":
      # --pre-place splits into a goto to the lifted pre-place + the real place (set-down+attach).
      frames = ["table", _root_of(states[i], a[2], mate_parent)]
      mk = lambda g: Task(name, involved, g, type="place", frames=frames)
      emit = (split_skill(a, i, wit, involved, name, kind, mk, smembers)
              if pre_place and bridge_j is None else None)
      if emit is None:
        t = mk(goal_for_set(involved, smembers, i) if smembers
               else goal_for(involved, wit))
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
      _rs = involved if bridge_j is not None else [r]
      t = Task(name, _rs,
               (goal_for(_rs, wit) if bridge_j is not None else
                (mirrored_pin(_rs) if witness_sets else goal_for(_rs, None))), type="pick",
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
      if i in coupled_release_robots:
        rel_robots = list(dict.fromkeys(rel_robots + coupled_release_robots[i]))
      if bridge_j is not None:
        rel_robots = list(dict.fromkeys(rel_robots + shared_robots))
      release_members = current_full_members(rel_robots)
      release_goal = (mirrored_pin(rel_robots) if len(release_members) >= 2
                      else goal_for(rel_robots, None))
      t = Task(name, rel_robots, release_goal, type="place",
               frames=[_support_frame(states[i + 1], root), root])
      if post_release:
        # Detach first at the exact holding configuration, then add an explicit leave-contact
        # endpoint whose IK keeps continuing holders at that configuration. An incoming
        # GoalSet is retained member-for-member rather than allowing a branch switch. The
        # ordinary motion planner remains unconstrained between these endpoints.
        post_members = []
        for m, qrel in enumerate(release_members):
          qpost = robot_ik.post_release_witness(
              a, states[i], states[i + 1], assembly, qrel)
          if qpost is None:
            print(f"  [post-release] {i:02d} member {m}: no retract keyframe -- departure "
                  "goal equals release goal (zero-length funnel)")
            qpost = qrel
          elif getattr(robot_ik, "post_release_used_delta", None) is not None:
            used = float(robot_ik.post_release_used_delta)
            requested = float(robot_ik.post_release_delta)
            if used < requested - 1e-12:
              print(f"  [post-release] {i:02d} member {m}: full {requested*1000:.0f}mm "
                    f"departure infeasible; using the largest ladder escape found "
                    f"({used*1000:.0f}mm) under the same clearance gate.")
          post_members.append(np.asarray(qpost, dtype=float))
        pon = f"{i:02d}_post_release_{'_'.join(str(x) for x in a[1:])}"
        pgoal = (goal_for_paired(rel_robots, release_members, post_members, i)
                 if len(post_members) >= 2 else goal_for(rel_robots, post_members[0]))
        emit = [(name, t), (pon, Task(pon, rel_robots, pgoal, type="goto"))]
    elif kind == "add_grasp":
      # Direct coupled case: all participants rendezvous at the following assembly's funnel
      # mouth. With an intervening support switch, only its pre-switch holders rendezvous here;
      # the inserter approaches after the old holder releases. A standalone add_grasp keeps its
      # historical goto (or --pre-add-grasp split).
      mk = lambda g: Task(name, involved, g, type="goto")
      emit = (split_skill(a, i, goal_wit, involved, name, kind, mk, smembers)
              if pre_add_grasp and chain_j is None else None)
      if emit is None:
        t = mk(goal_for_set(involved, smembers, i) if smembers and chain_j is None
               else goal_for(involved, goal_wit))
    elif kind == "handover":
      rB = r
      rA = _holder_of(states[i], a[2])
      task_robots = [rA, rB] if rA is not None and rA != rB else [rB]
      if bridge_j is not None:
        task_robots = list(dict.fromkeys(task_robots + shared_robots))
      t = Task(name, task_robots,
               goal_for_set(task_robots, smembers, i) if smembers
               else goal_for(task_robots, wit), type="handover",
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
  if resolved_pre_witnesses is not None:
    resolved_pre_witnesses.clear()
    resolved_pre_witnesses.update(pre_cache)
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


def _install_robot_margin(env, margin, robots, gripper_frame, ramp=0.0, table_margin=0.0,
                          table_float=0.0):
  """MONKEY-PATCH our planning gates onto the exported problem (mr_bench unmodified).

  Joint limits are ALWAYS enforced here. mr_bench samples random configurations inside
  ``env.limits``, but its RAI collision predicates accept an arbitrary supplied/interpolated
  configuration without checking those limits. A BiRRT detour can therefore leave the legal
  range even when both endpoints are valid (measured on peg-stool: shoulder_lift -0.338 while
  its upper limit is -0.383 rad). Simulation correctly rejects such a path, much too late.
  Keeping the check in the same wrapper as our extra clearance predicates means planning,
  shortcutting, diagnostics, and simulation preflight all use the same hard boundary.

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

  2. BY EE DISTANCE TO BOTH SEGMENT ANCHORS, for that involved pair: required =
     margin * min(1, d_from_start/ramp, d_to_goal/ramp).  Full margin applies in transit and
     relaxes to zero at either keyframe where contact may be intended.  Relaxing only at the
     destination makes the predicate discontinuous at a mode switch: a pair allowed to touch
     at the preceding handover/release would suddenly owe the full margin before it had moved.

     Keyed on EE distance rather than path position ON PURPOSE. "The last 10% of the path" is
     history-dependent and a sampling planner has no such notion -- its predicate must be a pure
     function of (q, mode) or edge checks and rewiring become inconsistent. Distances to the
     mode's two anchors are that pure function, so the ramp works INSIDE planning rather than as
     post-processing.

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

  # `_too_close` runs once for every sampled configuration.  The current active task is
  # unchanged while a Mode object is reused along an interpolated segment, but
  # SequenceMixin finds it by repeatedly scanning `sequence`.  Cache per exact Mode object
  # (rather than merely task IDs, which could alias different scene graphs); this is local to
  # our margin wrapper and leaves mr_bench untouched.
  _task_by_mode = {}

  def _task_of(m):
    key = id(m)
    if key in _task_by_mode:
      return _task_by_mode[key]
    try:
      combos = env.get_valid_next_task_combinations(m)
      task = env.get_active_task(m, combos[0] if combos else None)
    except Exception:
      task = None
    _task_by_mode[key] = task
    return task

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
        g = _np.asarray(_goal_q(task.goal, m), dtype=float)
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
        g = _np.asarray(_goal_q(task.goal, m), dtype=float)
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
  # The base predicate cannot distinguish distances inside its accepted penetration budget.
  # Starting a positive clearance ramp inside that same band creates a fictitious, arbitrarily
  # thin violation next to an otherwise-valid keyframe (and makes feasibility depend on the
  # audit sampling rate).  Treat the base tolerance as the anchor's contact neighbourhood;
  # outside it, the requested margin still grows with exactly the configured slope.
  _anchor_deadband = max(0.0, float(getattr(env, "collision_tolerance", 0.0) or 0.0))

  # STRUCTURED REJECTION AUDIT.  MP_MARGIN_DEBUG used to print every veto, which turns a
  # moderately hard run into hundreds of thousands of indistinguishable lines and still does
  # not answer the useful question: "did OUR wrapper wedge this particular frontier mode?"
  # Keep counters on the environment instead.  run_birrt prints a compact per-frontier/top-pair
  # report at termination; callers/tests can inspect margin_rejection_stats directly.
  _stats = {}
  _last_rejection = [None]                    # final predicate call inside an edge check
  _limits = _np.asarray(getattr(env, "limits", _np.empty((0, 0))), dtype=float)
  try:
    _joint_names = list(env.C.getJointNames())
  except Exception:
    _joint_names = []

  def _mode_key(m):
    try:
      return tuple(int(x) for x in _np.atleast_1d(getattr(m, "task_ids", ()) or ()))
    except Exception:
      return ()

  def _reset_stats():
    _stats.clear()
    _stats.update({"calls": 0, "base_rejects": 0, "base_passes": 0,
                   "gate_rejects": 0, "base_by_mode": {}, "pass_by_mode": {},
                   "gate_by_mode": {}, "records": {}, "base_audited_by_mode": {},
                   "base_records": {}, "edge_rejects": 0, "edge_base_rejects": 0,
                   "edge_gate_rejects": 0, "edge_by_mode": {}, "edge_records": {}})
    _last_rejection[0] = None

  _reset_stats()

  def _bump(d, key):
    d[key] = d.get(key, 0) + 1

  def _veto(m, gate, fa, fb, observed, required, owners=(None, None), involved=(),
            factor=None):
    task = _task_of(m)
    return {"mode": _mode_key(m), "task": getattr(task, "name", None), "gate": gate,
            "frames": (str(fa), str(fb)), "owners": tuple(owners),
            "involved": tuple(sorted(involved)), "observed": float(observed),
            "required": float(required), "factor": None if factor is None else float(factor)}

  def _record(reason):
    mode = reason["mode"]
    _stats["gate_rejects"] += 1
    _bump(_stats["gate_by_mode"], mode)
    key = (mode, reason["task"], reason["gate"], reason["frames"], reason["owners"])
    rec = _stats["records"].get(key)
    slack = reason.get("slack", reason["observed"] - reason["required"])
    if rec is None:
      rec = dict(reason, count=0, worst_slack=slack)
      _stats["records"][key] = rec
    rec["count"] += 1
    # Retain the most strongly rejected concrete sample, so the printed d/requirement/factor
    # belong together instead of mixing independent extrema into an impossible example.
    if slack < rec["worst_slack"]:
      rec.update({k: reason[k] for k in
                  ("involved", "observed", "required", "factor")})
      rec["slack"] = slack
      rec["worst_slack"] = slack
    if _os.environ.get("MP_MARGIN_DEBUG"):
      print(f"[margin-debug] VETO gate={reason['gate']} mode={list(mode)} "
            f"task={reason['task']} {reason['frames'][0]} ~ {reason['frames'][1]} "
            f"d={reason['observed']*1000:.1f}mm req={reason['required']*1000:.1f}mm "
            f"owners={reason['owners']} involved={list(reason['involved'])}")

  def _record_base_pairs(m, cap=64):
    """Sample concrete base-predicate collision pairs without taxing every rejected check.

    The underlying predicate leaves C at the rejected (q, mode).  Auditing the first ``cap``
    failures per mode is enough to expose a systematic wall while bounding the extra RAI query
    cost on a run with hundreds of thousands of ordinary edge collisions.
    """
    mode = _mode_key(m)
    n = _stats["base_audited_by_mode"].get(mode, 0)
    if n >= cap:
      return
    _stats["base_audited_by_mode"][mode] = n + 1
    try:
      collisions = env.C.getCollisions(0.0)
    except Exception:
      return
    task = _task_of(m)
    for c in collisions:
      fa, fb, d = str(c[0]), str(c[1]), float(c[2])
      if d >= 0.0:
        continue
      key = (mode, getattr(task, "name", None), fa, fb, _owner(fa), _owner(fb))
      rec = _stats["base_records"].get(key)
      if rec is None:
        rec = {"mode": mode, "task": getattr(task, "name", None),
               "frames": (fa, fb), "owners": (_owner(fa), _owner(fb)),
               "count": 0, "min_distance": d}
        _stats["base_records"][key] = rec
      rec["count"] += 1
      rec["min_distance"] = min(rec["min_distance"], d)

  def _record_failed_edge(m):
    mode = _mode_key(m)
    _stats["edge_rejects"] += 1
    _bump(_stats["edge_by_mode"], mode)
    last = _last_rejection[0]
    layer = None if last is None else last.get("layer")
    if layer in ("margin", "gate"):
      _stats["edge_gate_rejects"] += 1
      return
    _stats["edge_base_rejects"] += 1
    try:
      collisions = env.C.getCollisions(0.0)
    except Exception:
      return
    task = _task_of(m)
    for c in collisions:
      fa, fb, d = str(c[0]), str(c[1]), float(c[2])
      if d >= 0.0:
        continue
      key = (mode, getattr(task, "name", None), fa, fb, _owner(fa), _owner(fb))
      rec = _stats["edge_records"].get(key)
      if rec is None:
        rec = {"mode": mode, "task": getattr(task, "name", None),
               "frames": (fa, fb), "owners": (_owner(fa), _owner(fb)),
               "count": 0, "min_distance": d}
        _stats["edge_records"][key] = rec
      rec["count"] += 1
      rec["min_distance"] = min(rec["min_distance"], d)

  def _report(frontier_task_ids=None, top=8):
    """Print why this wrapper rejected base-valid samples, with frontier attribution."""
    frontier = (() if frontier_task_ids is None else
                tuple(int(x) for x in _np.atleast_1d(frontier_task_ids)))
    calls, base, gate = _stats["calls"], _stats["base_rejects"], _stats["gate_rejects"]
    passed = _stats["base_passes"]
    print(f"[mp/margin] predicate audit: {calls} checks; base predicate rejected {base}; "
          f"our added margin gates rejected {gate}/{passed} base-valid checks.")
    if frontier:
      print(f"[mp/margin] frontier mode {list(frontier)}: "
            f"base rejected {_stats['base_by_mode'].get(frontier, 0)}, "
            f"base-valid {_stats['pass_by_mode'].get(frontier, 0)}, "
            f"our gates rejected {_stats['gate_by_mode'].get(frontier, 0)}.")
    if not gate:
      print("[mp/margin] our wrapper vetoed no base-valid sample; this run's blockage is not "
            "caused by --mp-robot-margin/--mp-table-margin.")
    else:
      rows = list(_stats["records"].values())
      # Put exact-frontier evidence first, then the most common concrete pair/reason overall.
      rows.sort(key=lambda r: (r["mode"] == frontier, r["count"]), reverse=True)
      for r in rows[:max(1, int(top))]:
        if r["gate"] == "joint-limit":
          side = r["frames"][1]
          print(f"[mp/gate] mode={list(r['mode'])} task={r['task']} gate=joint-limit "
                f"count={r['count']} joint={r['frames'][0]} value={r['observed']:.6f}rad "
                f"{side}={r['required']:.6f}rad "
                f"violation={-r['worst_slack']:.6g}rad")
          continue
        factor = "" if r["factor"] is None else f" ramp={r['factor']:.3f}"
        owners = "/".join("-" if x is None else str(x) for x in r["owners"])
        print(f"[mp/margin] mode={list(r['mode'])} task={r['task']} gate={r['gate']} "
              f"count={r['count']} pair={r['frames'][0]}~{r['frames'][1]} "
              f"owners={owners} d={r['observed']*1000:.2f}mm "
              f"required={r['required']*1000:.2f}mm{factor} "
              f"involved={list(r['involved'])}")
    # The base predicate is not our gate, but showing a bounded sample of its concrete pairs
    # makes a "not us" result actionable rather than merely exculpatory.
    brows = [r for r in _stats["base_records"].values()
             if not frontier or r["mode"] == frontier]
    brows.sort(key=lambda r: (r["count"], -r["min_distance"]), reverse=True)
    audited = _stats["base_audited_by_mode"].get(frontier, 0) if frontier else sum(
        _stats["base_audited_by_mode"].values())
    if audited:
      print(f"[mp/base] concrete collision pairs from {audited} sampled base rejection(s) "
            + (f"at frontier {list(frontier)}:" if frontier else ":"))
      for r in brows[:max(1, min(int(top), 5))]:
        owners = "/".join("-" if x is None else str(x) for x in r["owners"])
        print(f"[mp/base] task={r['task']} count={r['count']}/{audited} "
              f"pair={r['frames'][0]}~{r['frames'][1]} owners={owners} "
              f"min_penetration={r['min_distance']*1000:.2f}mm")
    edge_n = _stats["edge_by_mode"].get(frontier, 0) if frontier else _stats["edge_rejects"]
    if edge_n:
      print(f"[mp/edge] failed edges at frontier {list(frontier)}: {edge_n} "
            f"(base={_stats['edge_base_rejects']}, margin={_stats['edge_gate_rejects']} "
            "globally); concrete first-hit pairs:")
      erows = [r for r in _stats["edge_records"].values()
               if not frontier or r["mode"] == frontier]
      erows.sort(key=lambda r: (r["count"], -r["min_distance"]), reverse=True)
      for r in erows[:max(1, min(int(top), 5))]:
        owners = "/".join("-" if x is None else str(x) for x in r["owners"])
        print(f"[mp/edge] task={r['task']} count={r['count']}/{edge_n} "
              f"pair={r['frames'][0]}~{r['frames'][1]} owners={owners} "
              f"min_penetration={r['min_distance']*1000:.2f}mm")

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
          # DATUM FIX (2026-07-30, FRAGILITY 2.6 option A): the SIM raises its table by
          # part_float, so planner distance d overstates the REAL clearance by exactly that
          # much. Compare against the real table (d_eff), and enforce a hard never-below-
          # real-contact floor even inside the ramp: a carried part commanded below the
          # sim's surface is always a press the base predicate cannot see (its table is
          # lower). Ends stay contact-allowed at d_eff ~ 0 (pick/set-down rest ON the real
          # table); measured without this: 185N table press snapping the seat grasp.
          d_eff = d - table_float
          if d_eff < -0.001:
            return _veto(m, "table-contact-floor", fa, fb, d_eff, -0.001,
                         owners=(ro, None), involved=involved, factor=f)
          req = table_margin * (1.5 * f - 0.5)
          if max(d_eff, 0.0) < req:
            return _veto(m, "carried-part-table-margin", fa, fb, max(d_eff, 0.0), req,
                         owners=(ro, None), involved=involved, factor=f)
        continue                             # table pairs never enter the robot-robot branch
      if margin <= 0.0 or d >= margin:
        continue
      ra, rb = _owner(fa), _owner(fb)
      if ra is None or rb is None or ra == rb:
        continue
      req = margin
      gate = "robot-full-margin"
      factor = 1.0
      if ra in involved and rb in involved:
        req = 0.0
        gate = "robot-participant-ramp"
        factor = 0.0
        if ramp > 0.0:
          if goals is None:
            goals = _goals(m)
          try:
            pair = (ra, rb)
            d_goal = max(float(_np.linalg.norm(
                _np.asarray(env.C.getFrame(gripper_frame[r]).getPosition(), dtype=float)
                - goals[r])) for r in pair if r in goals)
            d_prev = max(float(_np.linalg.norm(
                _np.asarray(env.C.getFrame(gripper_frame[r]).getPosition(), dtype=float)
                - _prev_ee(r, m))) for r in pair if _prev_ee(r, m) is not None)
            # Exact FK re-evaluation of a persisted anchor can differ by roundoff.  Without
            # this clamp, observed==0 and required~=1e-15 spuriously rejects the anchor.
            d_anchor = max(0.0, min(d_goal, d_prev) - _anchor_deadband)
            factor = 0.0 if d_anchor <= 1e-9 else min(1.0, d_anchor / ramp)
            req = margin * factor
          except Exception:
            req = 0.0
            factor = 0.0
      elif ramp > 0.0 and (ra in involved) != (rb in involved):
        # ONE-SIDED relaxation (2026-07-30): an involved arm NEAR ITS GOAL is executing its
        # task posture, and that fixed posture can sit inside the margin of an UNINVOLVED
        # arm's BASE links -- which no sample of the free arm can change (measured: chair-8
        # task 28_assemble_a1_obj_7_obj_5, a1's pinned wrist 7.7mm from a2's base link at
        # HOME -> every transition sample invalid -> "Failed to sample after 10000
        # attempts"). Ramp the pair's requirement on the INVOLVED arm's own distance to its
        # two anchors: full margin while it transits, relaxed to the base predicate exactly
        # where either adjacent task pins it -- the same contact-allowed-at-anchors philosophy as the
        # involved-pair and table ramps. Fail OPEN like the branch above: a missing goal
        # must not veto a state the base predicate accepts.
        rinv = ra if ra in involved else rb
        gate = "robot-one-sided-ramp"
        if goals is None:
          goals = _goals(m)
        try:
          ee = _np.asarray(env.C.getFrame(gripper_frame[rinv]).getPosition(), dtype=float)
          d_goal = float(_np.linalg.norm(
              ee - goals[rinv]))
          prev = _prev_ee(rinv, m)
          d_prev = float(_np.linalg.norm(ee - prev)) if prev is not None else d_goal
          d_anchor = max(0.0, min(d_goal, d_prev) - _anchor_deadband)
          factor = 0.0 if d_anchor <= 1e-9 else min(1.0, d_anchor / ramp)
          req = margin * factor
        except Exception:
          req = 0.0
          factor = 0.0
      # CLAMP AT CONTACT (2026-07-30): a fully-relaxed pair (req 0) must not be vetoed for a
      # sub-tolerance witness graze -- `d < req` with d = -0.1mm and req = 0 still fired,
      # so every transition whose goal formation TOUCHES (a seated mate, a co-grasp) was
      # unsatisfiable under the margin no matter the relaxation (chair-8's second wall).
      # Penetration is the BASE predicate's job (same rule the table branch documents);
      # clamping d at 0 keeps every positive requirement guarding real proximity while
      # handing at-or-below-contact states back to the tolerance.
      if max(d, 0.0) < req:
        return _veto(m, gate, fa, fb, max(d, 0.0), req, owners=(ra, rb),
                     involved=involved, factor=factor)
    return None

  def _joint_limit_veto(q, m):
    """Return a structured veto for the first out-of-range joint, or None.

    ``is_collision_free`` receives a Configuration while ``is_collision_free_np`` receives a
    flat ndarray. q=None means "check the configuration already posed in C" in mr_bench.
    The 1e-9 allowance is numerical noise only, not a planning margin.
    """
    if _limits.ndim != 2 or _limits.shape[0] != 2 or _limits.shape[1] == 0:
      return None
    try:
      raw = env.C.getJointState() if q is None else (q.state() if hasattr(q, "state") else q)
      flat = _np.asarray(raw, dtype=float).ravel()
    except Exception:
      return None
    if flat.size != _limits.shape[1]:
      return None
    lo, hi = _limits[0], _limits[1]
    bad_lo = _np.flatnonzero(flat < lo - 1e-9)
    bad_hi = _np.flatnonzero(flat > hi + 1e-9)
    if not bad_lo.size and not bad_hi.size:
      return None
    candidates = []
    if bad_lo.size:
      j = int(bad_lo[_np.argmax(lo[bad_lo] - flat[bad_lo])])
      candidates.append((float(lo[j] - flat[j]), j, "lower", float(lo[j])))
    if bad_hi.size:
      j = int(bad_hi[_np.argmax(flat[bad_hi] - hi[bad_hi])])
      candidates.append((float(flat[j] - hi[j]), j, "upper", float(hi[j])))
    violation, j, side, bound = max(candidates)
    name = _joint_names[j] if j < len(_joint_names) else f"joint[{j}]"
    reason = _veto(m, "joint-limit", name, side, float(flat[j]), bound)
    reason["slack"] = -violation
    return reason

  def _wrap(fn):
    def patched(*a, **kw):
      m = kw.get("m", a[1] if len(a) > 1 else None)
      mode = _mode_key(m)
      _stats["calls"] += 1
      if not fn(*a, **kw):                               # poses C at (mode, q) as a side effect
        _stats["base_rejects"] += 1
        _bump(_stats["base_by_mode"], mode)
        _record_base_pairs(m)
        _last_rejection[0] = {"layer": "base", "mode": mode}
        return False
      _stats["base_passes"] += 1
      _bump(_stats["pass_by_mode"], mode)
      q = kw.get("q", a[0] if a else None)
      reason = _joint_limit_veto(q, m)
      if reason is None:
        reason = _too_close(m)
      if reason is not None:
        _record(reason)
        _last_rejection[0] = {"layer": "gate", "mode": mode, "reason": reason}
        return False
      _last_rejection[0] = None
      return True
    return patched

  # BOTH full-scene predicates: BiRRT leans on the _np variant, so wrapping only
  # is_collision_free would leave the margin unenforced on the hot path.
  # is_collision_free_for_robot is deliberately NOT wrapped -- it judges a SUBSET of robots and
  # explicitly discounts collisions with the others, so a cross-robot margin would invert it.
  env.is_collision_free = _wrap(env.is_collision_free)
  env.is_collision_free_np = _wrap(env.is_collision_free_np)
  _edge_fn = env.is_edge_collision_free

  def _edge_patched(*a, **kw):
    ok = _edge_fn(*a, **kw)
    if not ok:
      m = kw.get("m", a[2] if len(a) > 2 else None)
      _record_failed_edge(m)
    return ok

  env.is_edge_collision_free = _edge_patched
  env.margin_rejection_stats = _stats
  env.reset_margin_rejection_stats = _reset_stats
  env.report_margin_rejections = _report
  # Read-only structured reason for the most recent predicate call.  Post-processors that
  # deliberately reuse this exact predicate (phase scheduling, path audits) need to know
  # whether a rejection came from an arm pair, a carried-part/table margin, or a joint limit;
  # scraping debug text loses the owner information required for coordination-space repair.
  env.last_margin_rejection = lambda: (
      None if _last_rejection[0] is None else dict(_last_rejection[0]))
  return env


def make_problem(C_start, assembly, plan, witnesses, robots, robot_ik, gk, bracing,
                 depgraph=False, collision_tolerance=0.01, pre_witnesses=None,
                 robot_robot_margin=0.0, robot_margin_ramp=0.0, table_margin=0.0,
                 table_float=0.0, witness_sets=None, resolved_pre_witnesses=None,
                 resolve_missing_pre=True, task_records=None, saved_step_names=None,
                 saved_sequence_names=None, pre_pin_robots=None):
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
      if witness_sets:
        enable_goalset_cycling()
        _n = sum(1 for v in witness_sets.values() if len(v) >= 2)
        print(f"[witsets] make_problem: {_n} task(s) carry witness GoalSets "
              f"(K up to {max(len(v) for v in witness_sets.values())}).")
      if task_records is None:
        self.tasks, seq, step_names = build_tasks(
            assembly, plan, witnesses, states, self.robots, self.robot_idx, robot_ik, home,
            pre_witnesses=pre_witnesses, witness_sets=witness_sets,
            resolved_pre_witnesses=resolved_pre_witnesses,
            resolve_missing_pre=resolve_missing_pre,
            pre_pin_robots=pre_pin_robots)
      else:
        # FINAL-EXPORT replay: task goals/effects are data, not something inferred again from
        # today's exporter. This closes the second drift channel after IK: a later change to
        # involved-robot ordering or task splitting must not relink an old path creatively.
        self.tasks = [_task_from_record(rec) for rec in task_records]
        seq = list(saved_sequence_names or [t.name for t in self.tasks])
        step_names = [list(xs) for xs in (saved_step_names or [])]
        if len(step_names) != len(plan):
          raise ValueError(
              f"mp_export: persisted task map has {len(step_names)} plan steps, expected "
              f"{len(plan)}")
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
      self.sequence_names = list(seq)
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
            + (f" Relaxed to 0 between the TASK'S OWN robots, ramped over "
               f"{robot_margin_ramp:g}m of EE travel from/to each segment anchor." if robot_margin_ramp > 0
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
  # Always install: beyond the optional margin gates this supplies the joint-limit predicate
  # missing from mr_bench's RAI collision checker. With both margins at zero it is cheap
  # arithmetic and changes only configurations which simulation would reject anyway.
  _install_robot_margin(env, float(robot_robot_margin or 0.0), list(robots),
                        robot_ik.gripper_frame, ramp=float(robot_margin_ramp),
                        table_margin=float(table_margin or 0.0),
                        table_float=float(table_float or 0.0))
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


_BIRRT_INITIAL_TREE_CAPACITY = 1024


def _init_compact_bidir_tree(tree, env):
  """Initialize mr_bench's growable bidirectional arrays without its 100k-node reserve."""
  tree.order = 1
  robot_dims = sum(env.robot_dims.values())
  tree.subtree = {}
  tree.initial_capacity = _BIRRT_INITIAL_TREE_CAPACITY
  tree.batch_subtree = np.empty((tree.initial_capacity, robot_dims), dtype=np.float64)
  tree.node_ids_subtree = np.empty(tree.initial_capacity, dtype=np.int64)
  tree.subtree_b = {}
  tree.batch_subtree_b = np.empty((tree.initial_capacity, robot_dims), dtype=np.float64)
  tree.node_ids_subtree_b = np.empty(tree.initial_capacity, dtype=np.int64)
  tree.connected = False


def run_birrt(problem, runtime=60.0, shortcut=True, shortcut_iters=1000,
              shortcut_resolution=None, shortcut_strategy="native",
              shortcut_pair_checks=200, optimize=False, verbose=True, seed=None,
              stall_timeout=None, fake_clock=False):
  """Run Bidirectional RRT* to the FIRST solution (optimize=False), then optionally apply
  robot/mode shortcutting as a post-process. Returns (path, info). No cost optimization --
  the tree stops as soon as a valid path is found; shortcutting only removes slack.

  NOTE: `runtime` bounds ONLY the BiRRT tree search. Shortcutting is a SEPARATE post-process
  with its OWN budget (`shortcut_iters` successful shortcuts, or ~2.5x that many attempts) and
  is NOT time-limited -- on a long path it can dominate wall-clock. Lower `shortcut_iters`
  (or pass shortcut=False) to cap/skip it.

  `seed` seeds the GLOBAL numpy + stdlib RNGs mr_bench draws from (the whole BiRRT+shortcut
  draw). Termination still uses the REAL wall clock unless `fake_clock=True` swaps time.time
  for a deterministic fake clock (+1ms per consultation) FOR THIS CALL ONLY -- the runtime
  budget then becomes a CONSULTATION budget (NOT wall seconds), which is what makes a seeded
  run bit-exact under machine jitter; with the real clock a seeded run is only approximately
  reproducible. first-solution return still exits early; perf_counter stays real.

  PERF: robot_mode_shortcut's OWN default collision resolution is 0.001 -- 10x FINER than the
  env's collision_resolution (0.01, what BiRRT itself used), for zero quality gain but ~10x
  the wall-clock (measured: 46s -> 4.6s, identical cost). We default `shortcut_resolution` to
  the env's collision_resolution so shortcutting checks no finer than the planner did.

  `shortcut_strategy="pair_then_gain"` is an opt-in local experimental shortcutter.  It does
  not edit or monkey-patch mr_bench: an early, fixed number of compatible two-robot candidates
  is certified by the same native predicate, then the remaining budget uses gain-guided
  single-robot candidates.  "native" retains the historical mr_bench implementation."""
  _ensure_mrbench()
  if seed is not None:
    # mr_bench draws from the GLOBAL numpy RNG throughout (sampling, tree ops, shortcut index
    # picks) plus stdlib random.shuffle in shortcutting -- seeding both makes the whole
    # BiRRT+shortcut draw REPRODUCIBLE. Unseeded, the same command yields paths whose failure
    # CLASSES differ (one draw grazes the table at 57N, the next has no table contact at all),
    # so every A/B on top was noise. None (the default) keeps the historical unseeded draw.
    # The TERMINATION clock stays REAL (wall seconds) unless fake_clock=True below: a seeded
    # draw under the real clock is only approximately reproducible (jitter shifts phase
    # boundaries); bit-exact replay additionally needs fake_clock.
    import random as _random
    np.random.seed(int(seed))
    _random.seed(int(seed))
    print(f"[mp/birrt] RNG seeded: {int(seed)} (np.random + random) -- reproducible draw; "
          f"termination uses the "
          + ("fake deterministic clock (consultation budget)"
             if fake_clock else "REAL clock (wall seconds)"))
  _reset_margin_audit = getattr(problem, "reset_margin_rejection_stats", None)
  if callable(_reset_margin_audit):
    _reset_margin_audit()                                  # report THIS planner pull, not setup
  from multi_robot_multi_goal_planning.planners.planner_birrtstar import BidirectionalRRTstar
  from multi_robot_multi_goal_planning.planners.rrtstar_base import (
      BaseRRTConfig, BidirectionalTree)
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
  # DETERMINISTIC-CLOCK OPTION (fake_clock=True): the tree search gates iterations on
  # time.time() elapsed (planner_birrtstar ptc.should_terminate), so machine jitter shifts a
  # phase boundary, the RNG stream offsets, and two identically-seeded draws diverge (measured:
  # identical to state 15, different after). fake_clock swaps time.time for a fake clock
  # (+1ms per consultation) FOR THIS CALL ONLY: first-solution return still exits early, and
  # the runtime budget becomes a deterministic CONSULTATION budget instead of wall seconds --
  # the tree phase can then run far longer than `runtime` wall seconds. perf_counter (pure
  # accounting there) stays real. mr_bench itself is untouched.
  import time as _time_mod
  _real_time = _time_mod.time
  if fake_clock:
    _fake_t = [_real_time()]
    def _fake_time():
      _fake_t[0] += 1e-3
      return _fake_t[0]
    _time_mod.time = _fake_time
    print(f"[mp/birrt] fake clock ACTIVE: --mp-runtime {runtime:g} is a CONSULTATION budget "
          f"(not wall seconds)")
  # mr_bench gives EVERY mode tree two eagerly allocated 100000 x total-DOF arrays.  That is
  # ~43 MiB per mode for four 7-DOF arms, although the great majority of long-horizon mode
  # trees contain only a few nodes.  Cross-7 consequently used several GiB merely reaching
  # its late modes.  Its BaseTree.ensure_capacity already doubles arrays losslessly, so use a
  # compact initial allocation while this planner instance is being built.  This is strictly
  # a storage-policy shim: node order, samples, distances and collision predicates are
  # unchanged, and the upstream module remains unmodified.
  _tree_init = BidirectionalTree.__init__
  def _compact_tree_init(self, env):
    _init_compact_bidir_tree(self, env)
  BidirectionalTree.__init__ = _compact_tree_init
  try:
    planner = BidirectionalRRTstar(problem, config=cfg)
  except Exception:
    BidirectionalTree.__init__ = _tree_init
    raise
  # Constructing new trees happens throughout plan(), so restoring here would only compact
  # the initial tree. Keep the class patched until the planner pull has finished below.
  print(f"[mp/birrt] compact mode trees ACTIVE: initial capacity "
        f"{_BIRRT_INITIAL_TREE_CAPACITY} "
        "nodes per direction (grow-on-demand; planning semantics unchanged)")
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
    BidirectionalTree.__init__ = _tree_init
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
  _report_margin_audit = getattr(problem, "report_margin_rejections", None)
  if callable(_report_margin_audit):
    _report_margin_audit(info["frontier_task_ids"])
  if verbose:
    tag = "optimized" if optimize else "first solution"
    print(f"[mp/birrt] {tag}: {'%d states' % len(path) if path else 'NO path'}")
  if path and shortcut:
    from multi_robot_multi_goal_planning.problems.util import path_cost
    before = path_cost(path, problem.batch_config_cost)
    res = (shortcut_resolution if shortcut_resolution is not None
           else getattr(problem, "collision_resolution", 0.01))
    strategy = str(shortcut_strategy).strip().lower()
    if strategy == "native":
      if verbose:
        print(f"[mp/birrt] shortcutting (native; up to {shortcut_iters} iters, resolution {res}, "
              "not time-bounded)...")
      # PASS THE ENV'S OWN TOLERANCE. robot_mode_shortcut's default is tolerance=0.01 = 1cm of
      # ALLOWED PENETRATION, which silently overrode the strict 1e-4/1e-6 tolerance the path was
      # planned at -- the one loose stage in the zero-collision-tolerance chain. Cost impact is
      # nil (measured: finals within seed noise at 1e-4); acceptance is just honest now.
      path, sc_info = robot_mode_shortcut(problem, path, max_iter=shortcut_iters, resolution=res,
                                          tolerance=getattr(problem, "collision_tolerance", 0.01))
      info["shortcut"] = sc_info
    elif strategy == "pair_then_gain":
      if shortcut_pair_checks < 0:
        raise ValueError("shortcut_pair_checks must be non-negative")
      exp_dir = _os.path.join(_os.path.dirname(__file__), "experiments", "shortcut-sampling")
      if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)
      import ideation_samplers as _shortcut_exp
      # The pair sampler consumes NumPy's global stream like mr_bench.  Give gainprop a private
      # random stream as well; a supplied --mp-seed makes the whole shortcut pass repeatable.
      exp_seed = int(seed) if seed is not None else int(np.random.randint(0, 2**32 - 1))
      if verbose:
        print(f"[mp/birrt] shortcutting (pair_then_gain; {shortcut_pair_checks} pair checks, "
              f"up to {shortcut_iters} total, resolution {res}, seed {exp_seed})...")
      sc_stats = {}
      path, sc_info = _shortcut_exp.shortcut_run(
          problem, path, "pair_then_gain", max_iter=shortcut_iters,
          max_attempts=max(2500, 5 * shortcut_iters), resolution=res,
          tolerance=getattr(problem, "collision_tolerance", 0.01),
          interpolation_resolution=0.5, seed=exp_seed,
          pair_checks=int(shortcut_pair_checks), stats_out=sc_stats)
      info["shortcut"] = sc_info
      info["shortcut_stats"] = sc_stats
    else:
      raise ValueError(f"unknown shortcut_strategy {shortcut_strategy!r}")
    info["shortcut_strategy"] = strategy
    if verbose:
      after = path_cost(path, problem.batch_config_cost)
      print(f"[mp/birrt] shortcut: {len(path)} states, cost {before:.3f} -> {after:.3f}")
  if path:
    from multi_robot_multi_goal_planning.problems.planning_env import State as _State
    path, n_repaired = repair_mode_chain(path, mode_sequence(problem),
                                         lambda ref, m: _State(ref.q, m))
    if n_repaired:
      info["mode_repair"] = n_repaired
  return path, info


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
    fields = [
        str(getattr(t, "name", "")),
        str(getattr(t, "type", "")),
        ",".join(str(r) for r in (getattr(t, "robots", None) or [])),
        ",".join(str(f) for f in (getattr(t, "frames", None) or [])),
    ]
    h.update(("|".join(fields) + "\n").encode())
  return h.hexdigest()[:16]


_EXPORT_FORMAT = "assembly-mp-export-v1"


def _task_record(task):
  goal = task.goal
  members = getattr(goal, "goals", None)
  paired = isinstance(goal, _PairedGoal)
  return {"name": task.name, "robots": list(task.robots), "type": task.type,
          "frames": None if task.frames is None else list(task.frames),
          "goal_kind": "paired" if paired else ("set" if members is not None else "single"),
          "goals": ([np.asarray(g, dtype=float).ravel().tolist() for g in members]
                    if members is not None else
                    [np.asarray(goal.goal, dtype=float).ravel().tolist()]),
          "conditions": ([np.asarray(c, dtype=float).ravel().tolist()
                          for c in goal.conditions] if paired else None),
          "condition_indices": (goal.condition_indices.tolist() if paired else None),
          "side_effect": getattr(task, "side_effect", None),
          "side_effect_data": (None if getattr(task, "side_effect_data", None) is None else
                               np.asarray(task.side_effect_data, dtype=float).ravel().tolist())}


def _task_from_record(rec):
  from multi_robot_multi_goal_planning.problems.planning_env import Task
  from multi_robot_multi_goal_planning.problems.core.goals import SingleGoal, GoalSet
  members = [np.asarray(g, dtype=float) for g in rec.get("goals") or []]
  if not members:
    raise ValueError(f"mp_export: persisted task {rec.get('name')!r} has no goal")
  kind = rec.get("goal_kind")
  if kind == "paired":
    goal = _PairedGoal(rec.get("conditions") or [], members,
                       rec.get("condition_indices") or [])
  else:
    goal = GoalSet(members) if kind == "set" else SingleGoal(members[0])
  side_data = rec.get("side_effect_data")
  task = Task(str(rec["name"]), list(rec.get("robots") or []), goal,
              type=rec.get("type"), frames=rec.get("frames"),
              side_effect=rec.get("side_effect"),
              side_effect_data=(None if side_data is None else np.asarray(side_data, dtype=float)))
  return task


def make_export_bundle(problem, plan, joint_names, witnesses, pre_witnesses,
                       witness_sets=None, settings=None):
  """Serialize the FINAL, post-pass inputs that created an mr_bench problem.

  Unlike the plan JSON's ``witnesses`` block, these are the witnesses *after* cograsp fusion,
  plus every materialized pre-skill keyframe.  Supplying this bundle to ``make_problem`` with
  ``resolve_missing_pre=False`` is therefore a zero-IK rebuild: it may reconstruct symbolic
  attachments and collision geometry, but it can never call RobotIK.bind/KOMO.
  """
  vec = lambda w: None if w is None else [float(v) for v in np.asarray(w).ravel()]
  return {
      "format": _EXPORT_FORMAT,
      "plan": [list(a) for a in plan],
      "joints": list(joint_names),
      "witnesses": [vec(w) for w in witnesses],
      "pre_witnesses": {str(i): vec(w) for i, w in sorted((pre_witnesses or {}).items())},
      "witness_sets": {str(i): [vec(w) for w in ws]
                       for i, ws in sorted((witness_sets or {}).items())},
      "settings": dict(settings or {}),
      "tasks": [_task_record(t) for t in problem.tasks],
      "step_names": [list(xs) for xs in getattr(problem, "step_names", [])],
      "sequence_names": list(getattr(problem, "sequence_names",
                                     [t.name for t in problem.tasks])),
      "task_signature": task_signature(problem),
  }


def save_export_bundle(path_json, bundle):
  with open(path_json, "w") as f:
    json.dump(bundle, f, indent=2)


def _export_blob(data):
  if isinstance(data, dict) and data.get("format") == _EXPORT_FORMAT:
    return data
  if isinstance(data, dict):
    return data.get("mp_export")
  return None


def load_export_bundle(path_json, plan, joint_names, required=True):
  """Load and strictly validate a final export from a standalone file or saved MP path.

  Returns arrays ready for ``make_problem``.  A malformed/mismatched bundle is fatal: replay
  must never turn a persistence problem into an implicit IK re-solve.
  """
  with open(path_json) as f:
    raw = json.load(f)
  blob = _export_blob(raw)
  if blob is None:
    if required:
      raise ValueError(
          f"{path_json}: no {_EXPORT_FORMAT} block; refusing to rebuild witnesses during "
          "replay (make one once with --mp-export-save, or explicitly refresh/re-export)")
    return None
  if blob.get("format") != _EXPORT_FORMAT:
    raise ValueError(f"{path_json}: unsupported motion-export format {blob.get('format')!r}")
  want_plan = [list(a) for a in plan]
  if blob.get("plan") != want_plan:
    raise ValueError(f"{path_json}: final export belongs to a different discrete plan")
  if list(blob.get("joints") or []) != list(joint_names):
    raise ValueError(f"{path_json}: final export joint order differs from the current scene")
  n, nj = len(plan), len(joint_names)

  def vec(w, where):
    if w is None:
      return None
    q = np.asarray(w, dtype=float).ravel()
    if q.size != nj:
      raise ValueError(f"{path_json}: {where} has {q.size} joints, expected {nj}")
    return q

  ws = blob.get("witnesses") or []
  if len(ws) != n:
    raise ValueError(f"{path_json}: final export has {len(ws)} witnesses for {n} actions")
  pre = {}
  for k, w in (blob.get("pre_witnesses") or {}).items():
    i = int(k)
    if not 0 <= i < n:
      raise ValueError(f"{path_json}: pre-witness index {i} is outside the plan")
    pre[i] = vec(w, f"pre-witness {i}")
  sets = {}
  for k, members in (blob.get("witness_sets") or {}).items():
    i = int(k)
    if not 0 <= i < n:
      raise ValueError(f"{path_json}: witness-set index {i} is outside the plan")
    sets[i] = [vec(w, f"witness-set {i} member") for w in members]
  tasks = list(blob.get("tasks") or [])
  step_names = [list(xs) for xs in (blob.get("step_names") or [])]
  sequence_names = list(blob.get("sequence_names") or [])
  if not tasks or len(step_names) != n or len(sequence_names) != len(tasks):
    raise ValueError(
        f"{path_json}: final export lacks its exact task/step/sequence manifest; refusing "
        "to infer current task semantics during replay")
  return {"witnesses": [vec(w, f"witness {i}") for i, w in enumerate(ws)],
          "pre_witnesses": pre, "witness_sets": sets or None,
          "settings": dict(blob.get("settings") or {}),
          "tasks": tasks, "step_names": step_names, "sequence_names": sequence_names,
          "task_signature": blob.get("task_signature"), "raw": blob}


def save_path(path_json, path, problem=None, mp_export_bundle=None):
  """Persist a solved BiRRT PATH (main --motion-plan --mp-save). Each mr_bench State ->
  {"q": joint vector, "mode": task_ids}. The discrete PLAN (main --save) must be saved from
  the SAME run: load_path replays against a problem REBUILT from that plan (make_problem is
  deterministic, so task ids / modes align).

  FORMAT: a bare LIST historically; with `problem` it becomes a dict containing states and a
  task signature. Current main.py also supplies ``mp_export_bundle``: the exact post-fusion
  witnesses, pre-keyframes and task manifest are embedded beside the path, so viewers/sim need
  no IK or task inference. load_path still reads the historical forms, but execution requires
  an explicit legacy refresh when that final-export block is absent."""
  states = [s.to_dict() for s in path] if path else []
  blob = states if problem is None else {"states": states,
                                        "task_signature": task_signature(problem)}
  if mp_export_bundle is not None:
    if not isinstance(blob, dict):
      blob = {"states": states}
    blob["mp_export"] = mp_export_bundle
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
  load_path.last_export = _export_blob(data)
  return out



def enable_goalset_cycling():
  """MONKEYPATCH (mr_bench untouched on disk): GoalSet.sample CYCLES its members instead of
  drawing one uniformly at random. Reverse-tree transition seeding samples the goal a handful
  of times per task; uniform-with-replacement leaves a member with NO reverse root ~23% of the
  time at K=4, and an unseeded member is dead weight. Cycling covers every member first, then
  wraps -- and consumes no np.random draws, so --mp-seed streams are unchanged."""
  from multi_robot_multi_goal_planning.problems.core import goals as _goals
  if getattr(_goals.GoalSet, "_witsets_cycling", False):
    return
  def _sample(self, mode):
    k = getattr(self, "_witsets_next", 0)
    self._witsets_next = k + 1
    return self.goals[k % len(self.goals)]
  _goals.GoalSet.sample = _sample
  _goals.GoalSet._witsets_cycling = True
  print("[witsets] GoalSet.sample now CYCLES members (runtime monkeypatch, "
        "reverse-root coverage; mr_bench on disk untouched).")


def _goal_q(goal, mode):
  """Deterministic goal draw for chain/report walks: a GoalSet's CANONICAL member 0 (member 0
  is the shipped single witness by construction), so mode_sequence and the clearance report
  stay reproducible instead of consuming RNG on a random member."""
  gs = getattr(goal, "goals", None)
  if gs is not None and len(gs):
    return gs[0]
  return goal.sample(mode)

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
    goal = _goal_q(task.goal, m)
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


def repair_mode_chain(path, chain, make_state, verbose=True):
  """Re-insert task modes that shortcutting DROPPED from a path. mr_bench's
  remove_interpolated_nodes deletes the middle of three states whenever the outer two configs
  coincide WITHOUT checking the middle state's mode (the A.mode != C.mode guard sits on the
  collinear branch only), and a ZERO-MOTION task -- a release/goto whose goal pins every
  involved arm where it already is -- has its whole mode stretch at ONE config, so once
  shortcutting straightens the uninvolved arms through it the entire stretch is erased.
  The geometry stays valid; only the mode ANNOTATION loses an entry -- and every consumer
  that derives EVENTS from mode transitions (the sim's fire extraction) then silently skips
  that task's side effects (measured on chair-8: the holder-release never fired and the arm
  dragged the whole chair to its next pick; 10 of 15 saved paths on disk were missing ids).

  `chain` = mode_sequence(problem), the symbolically-expected mode order. Wherever adjacent
  path states jump PAST chain entries, the missing modes are re-inserted at the boundary
  config -- correct by construction: a mode can only collapse when it is zero-motion, so its
  one true config IS the boundary config. make_state(ref_state, chain_mode) builds an
  inserted state of the caller's path-state type. Clean paths come back unchanged.
  Returns (path, number of inserted states)."""
  ids = lambda m: frozenset(int(t) for t in np.atleast_1d(m.task_ids))
  where = {ids(m): k for k, m in enumerate(chain)}
  out, prev_k, inserted = [], None, []
  for st in path:
    k = where.get(ids(st.mode))
    if k is None:
      # a mode outside the problem's chain: this is not our path/problem pair -- leave it
      # alone and let the task-signature guards at the call sites do the complaining.
      if verbose:
        print(f"[mp/repair] path mode {sorted(ids(st.mode))} not in the problem's mode "
              f"chain; leaving the path unrepaired")
      return list(path), 0
    if prev_k is not None:
      for j in range(prev_k + 1, k):
        out.append(make_state(out[-1], chain[j]))
        inserted.append(sorted(ids(chain[j]) - ids(chain[j - 1])))
    out.append(st)
    prev_k = k
  if inserted and verbose:
    print(f"[mp/repair] re-inserted {len(inserted)} dropped zero-motion mode(s); newly "
          f"active task ids per inserted mode: {inserted} -- without this their attach/"
          f"release events would never fire")
  return out, len(inserted)


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
    goal = _goal_q(task.goal, m)
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

  Two grippers on one group are a CLOSED kinematic loop. mr_bench does not project sampled
  paths onto that relative-pose manifold. build_tasks couples the KEYFRAMES only: add_grasp
  and pre-assemble share one all-arm rendezvous, and adjacent releases reuse the assembly
  endpoint instead of adding a pointless transport goal. It deliberately does not add an
  interior controller/manifold constraint; ordinary BiRRT may detour. This diagnostic
  quantifies that distinction by replaying the path through rai FK and measuring how far
  each gripper pair's relative transform drifts from the span's first state.

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
  # Most split actions complete at their last task. A post-release departure is different:
  # the discrete release/attach side effect fires at the preceding framed task, and the arm
  # retracts in the already-released state. Prefer the last task carrying frames; fall back to
  # the syntactic last task for pure-goto actions such as add_grasp.
  by_name = {t.name: t for t in problem.tasks}
  def _completion_tid(nms):
    effect = [tid[nm] for nm in nms if getattr(by_name[nm], "frames", None)]
    return max(effect) if effect else max(tid[nm] for nm in nms)
  last = sorted(_completion_tid(nms) for nms in problem.step_names if nms)
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
      print("[cograsp] a co-grasp is a CLOSED kinematic loop. The planner does not project "
            "onto that manifold. Shared add/assemble/release KEYFRAMES prevent a separate "
            "joint-transport goal but do not constrain the path interior, so detour drift is "
            "expected here unless a controller-aware planner is used elsewhere. Deviation "
            "is measured from the span's FIRST path state and is 0 there by construction.")
  return rows
