"""CLEARANCE-FLOOR WITNESS LADDER -- pick witnesses with high clearance, then among those,
the ones that cross the fewest IK branches.

WHY NOT THE EXISTING WEIGHTED SUM. `optimize_witnesses(..., clear_weight=W)` scores a candidate
as `movement - W*clearance`, so the two objectives TRADE: a pose can buy its way past a tight
spot by being cheap to reach. Measured previously, that left 0.376-clearance candidates unused
and even regressed a step. What is actually wanted is a CONSTRAINT -- clearance is a floor, and
among poses that clear it we want the least branch-crossing chain. That is an epsilon-constraint,
and because feasibility is monotone in the floor it can be bisected.

THREE THINGS THIS FIXES BEYOND THE OBJECTIVE SHAPE.

1. WHERE THE OTHER ARMS SIT WHILE WE MEASURE. `optimize_witnesses`' clearance poses every
   non-acting joint at `home_map` (solver.py ~1972). An earlier draft of THIS module posed them
   at their CARRIED configs instead, reasoning that a handover's partner is emphatically not at
   home. That was WRONG and is reverted: every candidate is solved and verified feasible with
   the other arms AT EXIT, so scoring -- or worse, emitting -- arm A's candidate beside arm B's
   carried config is a combination nothing ever collision-checked, and it produced witnesses
   with arms visibly through each other. Home-pinning is also the convention the exported plan
   relies on (mp_export's cur[]/goal_for bookkeeping), so the emitted witness is home outside
   the arms its step actually moves. The interleaving case that motivated the change is still
   covered -- by (2), where the candidate itself holds both arms.

2. MULTI-ARM STEPS ARE JOINT NODES. A handover / add_grasp / braced assemble chooses BOTH arms'
   configs at once, so scoring them independently is meaningless -- the interleaving is the whole
   question. Those steps get one node over the involved arms, and the chain runs over all steps
   in plan order rather than per-robot.

3. TOOL ROLL IS A SYMMETRY, NOT A BRANCH. A vacuum cup is roll-symmetric (any wrist_3 is the
   same grasp) and a two-finger gripper repeats every pi. Measured on the exported goals, the
   single worst "branch crossing" was 4.712 rad on joint 5 -- 3*pi/2 of tool roll that changes
   nothing physically. Candidates are re-expressed on the roll representative nearest the
   previous config before anything is measured, so those vanish for free.

WHAT IS OPTIMAL AND WHAT IS NOT. Given the candidate sets, the chain is solved by Viterbi
carrying a full joint map per surviving candidate -- the same structure `optimize_witnesses`
already uses. That is exact for a first-order chain; where two different predecessors reach the
same candidate it keeps the cheaper, which is the standard Viterbi assumption and not exact in
general once arms interleave. The floor itself IS exact: a hard node filter cannot make a
surviving chain suboptimal.
"""
import time

import numpy as np
import robotic as ry

from solver import _replay_states, _search_context, ik_branch, reconfig_crossings, \
    involved_robots

ELBOW_IDX, WRIST_IDX, ROLL_IDX = 2, 4, 5      # UR joint order within an arm
_MULTI = ("handover", "add_grasp")            # steps that choose more than one arm's config


def _involved(action, state_before, state_after, robots, robot_ik):
  """Arms whose configuration this step CHOOSES -- THE shared definition (see
  solver.involved_robots), which unions holders across both states so a handover's giver and
  its receiver, solved together, are chosen together."""
  return involved_robots(action, state_before, state_after, robot_ik)


def optimize_witnesses_floor(C, assembly, plan, robots, robot_ik, *, gripper_kind=None,
                             handovers=True, allow_place=True, bracing=False,
                             grasp_table=None, clear_margin=0.4, alpha=None,
                             branch_weight=10.0, branch_budget=0, alpha_lo=0.0,
                             branch_metric="sign", reconfig_thresh=np.pi / 2.0, verbose=True,
                             draw_seed=None, witness_sets=0, set_spread=0.05,
                             baseline_witnesses=None):
  """Returns (witnesses, report). `alpha` fixes the floor; None => BISECT for the highest
  feasible one. The floor is per-step NORMALISED (`eps_i = alpha * best_clearance_at_step_i`)
  because a global floor would be dictated by the tightest step -- an insertion has the gripper
  inside the target by construction, and one such step must not collapse the floor everywhere."""
  t0 = time.perf_counter()
  _strip0 = robot_ik.strip_inactive_collision
  robot_ik.strip_inactive_collision = False
  _harvest0 = robot_ik.harvest_all_branches
  robot_ik.harvest_all_branches = True   # this ladder NEEDS the branch diversity
  if grasp_table is None:
    _checker, grasp_table = _search_context(C, assembly, robots)
  states = _replay_states(C, assembly, plan, robots, gripper_kind, handovers, allow_place,
                          bracing, grasp_table, "optimize_witnesses_floor")
  if baseline_witnesses is not None:
    from plan_io import require_complete_witnesses
    require_complete_witnesses(plan, baseline_witnesses,
                               who="optimize_witnesses_floor baseline")
    baseline_witnesses = [None if w is None else np.asarray(w, dtype=float).copy()
                          for w in baseline_witnesses]
  fkC = ry.Config()
  fkC.addConfigurationCopy(robot_ik.C)
  names = list(fkC.getJointNames())
  home_map = dict(zip(names, robot_ik.exit_configuration))
  name_idx = {n: k for k, n in enumerate(names)}
  arm_joints = {r: [n for n in names if n.startswith(r + "_")] for r in robots}
  arm_idx = {r: [k for k, n in enumerate(names) if n.startswith(r + "_")] for r in robots}
  _collset = set(fkC.getCollidablePairs())
  coll_frames = {r: [f for f in _collset if f.startswith(r + "_")] for r in robots}
  _table = [f for f in _collset if f == "table"]
  def clearance_at(qmap, movers, step=None):
    """(capped clearance SUM, min ROBOT-ROBOT gap) for the configuration this step would EMIT.

    SCORED ON THE SAME BASE THE EMISSION USES -- bind's own witness for the step, with the
    candidate's joints written over it. Two earlier bases were wrong. The DP's carried map
    paired arms from different steps, a combination nothing ever collision-checked. Posing the
    non-movers at EXIT is what `optimize_witnesses` does, but it disagrees with what this
    ladder emits: a candidate is solved with the others at exit, while the emitted vector puts
    them at their HOLDING configs, so a candidate could score clear and still overlap a holder
    (measured on car-mini: the selected chain penetrated by 5.02mm). Scoring the emitted
    configuration is the only way the floor means anything.

    Returns the SUM (capped per pair at clear_margin -- see solver.clearance for why sum and
    not min) and, separately, the true minimum ROBOT-ROBOT gap, which the caller uses as a hard
    non-penetration filter. The table is in the sum but NOT in the min: the base is mounted ON
    it, so a table-inclusive min is always negative and carries no information."""
    b = bind_wit.get(step) if step is not None else None
    q = (b.copy() if b is not None
         else np.array([home_map[n] for n in names], dtype=float))
    for n, v in qmap.items():
      q[name_idx[n]] = v
    fkC.setJointState(q)
    tot, mn = 0.0, float("inf")
    other_arms = [f for rr in robots if rr not in movers for f in coll_frames[rr]]
    for r in movers:
      for fa in coll_frames[r]:
        for fb in other_arms + _table:
          tot += min(-float(fkC.eval(ry.FS.distance, [fa, fb])[0][0]), clear_margin)
    # The MIN runs over every ROBOT-ROBOT pair with a mover on at least one side -- crucially
    # INCLUDING mover-vs-MOVER. Restricting it to mover-vs-other made the filter VACUOUS
    # exactly when it matters: `_involved` unions holders from both states, so an assemble
    # routinely makes EVERY arm a mover, `other_arms` is then empty and no pair is ever
    # measured (observed: a chosen a1_palm/a3_coll5 overlap waved through while bind's own
    # witness at that step had +4.89mm). Pairs with no mover are inherited from bind's witness
    # and no candidate can change them, so they are excluded rather than blamed on the choice.
    for ia, ra in enumerate(robots):
      for rb in robots[ia + 1:]:
        if ra not in movers and rb not in movers:
          continue
        for fa in coll_frames[ra]:
          for fb in coll_frames[rb]:
            gap = -float(fkC.eval(ry.FS.distance, [fa, fb])[0][0])
            if gap < mn:
              mn = gap
    return tot, mn

  def _grasped_parts(i):
    """{arm: part} it grips at step i -- union over both states, since a handover's giver holds
    only BEFORE and its receiver only AFTER."""
    out = {}
    for st in (states[i], states[i + 1]):
      for r, (pt, _gid) in robot_ik.holders(st).items():
        out.setdefault(r, pt)
    return out

  def _grasp_rel(q, r, part):
    """The gripper-to-object transform: object position AND orientation in the GRIPPER frame."""
    fkC.setJointState(q)
    g = fkC.getFrame(robot_ik.gripper_frame[r])
    o = fkC.getFrame(part)
    Rg = np.asarray(g.getRotationMatrix(), dtype=float)
    Ro = np.asarray(o.getRotationMatrix(), dtype=float)
    dp = Rg.T @ (np.asarray(o.getPosition(), dtype=float)
                 - np.asarray(g.getPosition(), dtype=float))
    return dp, Rg.T @ Ro

  def _rel_delta(qa, qb, r, part):
    """How far the GRIPPER-TO-OBJECT relation moves between two configurations: (mm, degrees).

    Must compare the RELATIVE ROTATION matrices, not scalar quaternion angles -- a pi tool roll
    leaves the quaternion-angle magnitude unchanged and puts the object origin on the tool axis,
    so a scalar comparison scores it as zero drift and waves through the exact case this
    filter exists to catch."""
    pa, Ra = _grasp_rel(qa, r, part)
    pb, Rb = _grasp_rel(qb, r, part)
    c = (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0
    return float(np.linalg.norm(pa - pb)), float(np.degrees(np.arccos(min(1.0, max(-1.0, c)))))

  # ---- candidates ---------------------------------------------------------------------
  cand, movers, bind_wit, grasped = {}, {}, {}, {}
  for i, a in enumerate(plan):
    act = tuple(a)
    if baseline_witnesses is not None:
      w0 = baseline_witnesses[i]
      if w0 is None:                                      # non-IK action
        continue
    else:
      ok, w0 = robot_ik.bind(act, states[i], states[i + 1], assembly)
      if not ok or w0 is None:
        continue
    _w0 = np.asarray(w0, dtype=float).ravel()
    if _w0.size == len(names):
      bind_wit[i] = _w0                                        # see the emission note below
    movers[i] = _involved(act, states[i], states[i + 1], robots, robot_ik)
    grasped[i] = _grasped_parts(i)
    # Candidate harvesting is stochastic and may fail. The exact persisted/search witness is
    # always present and is the immutable feasibility fallback for this layer.
    cs = [w0] + list(
        robot_ik.candidate_witnesses(act, states[i], states[i + 1], assembly) or [])
    norm = []
    for w in cs:
      w = np.asarray(w, dtype=float).ravel()
      if w.size == len(names):
        norm.append({n: float(w[k]) for k, n in enumerate(names)
                     if any(n.startswith(r + "_") for r in movers[i])})
      elif w.size == len(arm_joints[act[1]]):             # per-arm solve (restricted scene)
        norm.append(dict(zip(arm_joints[act[1]], [float(x) for x in w])))
    if norm:
      cand[i] = norm
  # --witness-draw (draw_seed != None): DIVERSITY over re-draws for the bandit outer loop.
  # The ladder is near-deterministic, so every re-run ships the same chain -- including a
  # chain whose hold homotopy is infeasible to CONNECT (chair-8: the grasp approaches the
  # legs from the wrong side; per-keyframe checks cannot see it). Each draw THINS the ladder
  # with a private seeded RNG (drop each non-first candidate with p=0.4, always keep >= 1),
  # and the DP then optimizes within what survives: locally-good but globally different
  # chains, with full coverage of the candidate alphabet in the limit of draws.
  if draw_seed is not None:
    import zlib
    _rng = np.random.default_rng(zlib.crc32(f"witdraw|{int(draw_seed)}".encode()))
    for i in list(cand):
      pool = list(cand[i])
      _rng.shuffle(pool)                                     # so the DP favourite can drop too
      cand[i] = [pool[0]] + [c for c in pool[1:] if _rng.random() > 0.4]
  steps = sorted(cand)
  if not steps:
    robot_ik.strip_inactive_collision = _strip0
    robot_ik.harvest_all_branches = _harvest0
    return (list(baseline_witnesses) if baseline_witnesses is not None
            else [None] * len(plan)), {"steps": 0}

  def solve(alpha_v):
    """Viterbi over the whole plan with a per-step clearance FLOOR. Returns (chain, cost) or
    (None, reason) if some step has no candidate clearing its floor."""
    layers, bp, prev = [], [], [(dict(home_map), 0.0, None, 0)]
    for i in steps:
      mv = movers[i]
      scored, gaps = [], []
      for c in cand[i]:
        t, mn = clearance_at(dict(c), mv, i)
        scored.append(t)
        gaps.append(mn)
      # HARD NON-PENETRATION FILTER, applied before the floor: a candidate whose emitted
      # configuration puts two arms THROUGH each other is not a trade-off, it is invalid. Kept
      # only if EVERY candidate penetrates (then take the least-bad and let the floor rank
      # them) -- never lose a step's feasibility over it.
      ok_idx = [k for k, g in enumerate(gaps) if g >= 0.0]
      if not ok_idx:
        ok_idx = [int(np.argmax(gaps))] if gaps else []
      # GRASP-RELATION FILTER. Downstream does NOT read the emitted gripper pose and infer the
      # object from it -- the replay's _align_group_to_holder and mp_export's attach frames
      # rebuild the object pose from the CANONICAL grasp transform for the action's grasp id.
      # So a candidate whose gripper-to-object relation differs from bind's is drawn and
      # attached WRONG even when it is a perfectly good grasp: a pi tool roll is a symmetry of
      # a two-finger gripper, and picking one flipped the picked subassembly 180 deg and lifted
      # it into the hand. Measured on cross-12. Judge the RELATION, not the raw roll -- for a
      # handover the object's world orientation is genuinely free, so the roll alone says
      # nothing. Least-bad kept if every candidate drifts, so no step loses feasibility.
      gp = grasped.get(i, {})
      base_q = bind_wit.get(i)
      if base_q is not None and gp:
        def _drift(k):
          q = base_q.copy()
          for n, v in cand[i][k].items():
            q[name_idx[n]] = v
          worst = 0.0
          for r, part in gp.items():
            if r not in mv:
              continue                            # not chosen here; inherited from bind
            dp, da = _rel_delta(q, base_q, r, part)
            worst = max(worst, dp * 1000.0, da)    # mm and degrees on one scale, both small
          return worst
        drifts = {k: _drift(k) for k in ok_idx}
        keep_rel = [k for k in ok_idx if drifts[k] <= 5.0]
        ok_idx = keep_rel or [min(drifts, key=drifts.get)]
      best_c = max((scored[k] for k in ok_idx), default=0.0)
      floor = alpha_v * best_c
      keep = [k for k in ok_idx if scored[k] >= floor - 1e-9]
      if not keep:
        return None, f"step {i}: no candidate clears {floor:.3f}"
      layer, layer_bp, layer_sel = [], [], []
      for k in keep:
        c = cand[i][k]
        best, bidx, bxb = None, 0, 0
        for p, (pmap, pcost, _pc, pxb) in enumerate(prev):
          cc = dict(c)
          e, xb = 0.0, 0
          for r in mv:
            jn = arm_joints[r]
            qprev = np.array([pmap.get(n, home_map[n]) for n in jn])
            qc = np.array([cc.get(n, pmap.get(n, home_map[n])) for n in jn])
            # NO ROLL REWRITE. This used to re-express qc on the roll representative nearest
            # qprev and WRITE IT BACK into the emitted joints, on the theory that tool roll is a
            # symmetry. It is a symmetry of the two-finger GRASP -- fingers swap, same faces --
            # but NOT of anything downstream:
            #   * an arm that is HOLDING the object has its payload rotated by the roll, so a
            #     pi turn mates an assemble 180 deg out and sets a place down flipped (measured
            #     on cross-12: exactly +1.00*pi written onto the holding arm at steps 6, 8, 27,
            #     29, matching 180 deg object rotations at those very steps);
            #   * even on a FREE arm at a pick, the replay's _align_group_to_holder and
            #     mp_export reconstruct the object pose from the CANONICAL grasp transform, so a
            #     rolled gripper makes them draw/attach the group 180 deg over and lifted into
            #     the hand -- the reported "picked subassembly is rotated and in the air".
            # Nothing is lost by dropping it: the crossing metrics already ignore roll
            # (`ik_branch` labels joints 2 and 4; `reconfig_crossings` zeroes the roll index),
            # so its only remaining effect was to shrink the continuity norm below the arm's
            # true travel.
            # "sign" = the (elbow, wrist_2) sign label; "motion" = measured reconfiguration,
            # which also catches the shoulder flip the label is blind to (see reconfig_crossings).
            if branch_metric == "motion":
              nb = reconfig_crossings(qc, qprev, ROLL_IDX, reconfig_thresh)
            else:
              nb = sum(1 for x, y in zip(ik_branch(qc, ELBOW_IDX, WRIST_IDX),
                                         ik_branch(qprev, ELBOW_IDX, WRIST_IDX)) if x != y)
            e += branch_weight * nb + float(np.linalg.norm(qc - qprev))
            xb += nb
          tot = pcost + e
          if best is None or tot < best[1]:
            best, bidx, bxb = (cc, tot), p, pxb + xb
        m2 = dict(prev[bidx][0]); m2.update(best[0])
        layer.append((m2, best[1], best[0], bxb))
        layer_bp.append(bidx)
        layer_sel.append(best[0])
      layers.append(layer_sel)
      bp.append(layer_bp)
      prev = layer
    sel = int(np.argmin([c for _, c, _, _ in prev]))
    cost, crossings = prev[sel][1], prev[sel][3]
    # BACKTRACK PROPERLY. `chosen[j]` is the candidate dict the winning chain used at steps[j];
    # `layers[j][k]` is the k-th SURVIVING candidate of that layer (post-floor), so the layer
    # lists and the backpointers must be indexed together. Emitting from a separate greedy pass
    # -- as an earlier draft did -- silently threw the whole DP away.
    chosen = [None] * len(steps)
    for j in range(len(steps) - 1, -1, -1):
      chosen[j] = layers[j][sel]
      sel = bp[j][sel]
    return (cost, chosen, crossings), None

  # ---- choose the floor -----------------------------------------------------------------
  # NOT "maximise alpha". Bisecting for the highest feasible floor drives it to ~0.98, where a
  # single candidate survives per step and the branch objective has nothing left to choose --
  # measured, that produced MORE crossings than no floor at all. The question is "how much
  # clearance is free?", so: take the crossings achievable with NO floor as the reference, then
  # take the HIGHEST floor that still achieves it (plus an optional budget). That is literally
  # the least-branch-crossing chain among those satisfying as high a clearance as is free.
  front = []
  base, _ = solve(0.0)
  ref = base[2] if base is not None else 0
  grid = [0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.96, 0.99] if alpha is None else [alpha]
  for av in grid:
    got, why = solve(av)
    if got is None:
      front.append((av, None, None))
      continue
    front.append((av, got[2], got[0]))
  ok = [(av, xb, c) for av, xb, c in front if xb is not None and xb <= ref + branch_budget]
  best_a = max((av for av, _, _ in ok), default=0.0)
  # ---- emit the DP's own chain ---------------------------------------------------------
  # Start from the complete feasibility certificate and overwrite only layers for which the
  # DP selected an improvement. Thus no optimizer outcome can turn a required keyframe null.
  wit = ([None] * len(plan) if baseline_witnesses is None
         else [None if w is None else w.copy() for w in baseline_witnesses])
  qm = dict(home_map)
  got, _why = solve(best_a)
  cost, chosen, crossings = got if got is not None else (None, [dict(cand[i][0]) for i in steps], -1)
  for j, i in enumerate(steps):
    qm.update(chosen[j])                                   # carried map: EDGE costs only
    mv = movers[i]
    # EMIT on top of BIND'S OWN witness for this step, overwriting only the arms this ladder
    # actually chose. Neither of the two obvious alternatives is right. The carried map `qm` is
    # wrong: consumers apply witness DOFs for a set of arms (visualization._active_robots) that
    # does NOT match `movers` -- it is computed on the PRE state, takes the group containing
    # `onto` rather than `part` for an assemble, and for a BRACED PLACE includes holders that
    # `_involved` omits entirely -- so those arms would be posed from a stale config chosen at a
    # different step and never validated alongside this one. That is what put robots visibly
    # through each other. Home is wrong for the same set: homing a brace holder tears its grasp.
    # `bind`'s witness already holds a feasible, mutually-consistent value for EVERY arm, so it
    # is the only safe base; `qm` exists solely to measure branch crossings against the arm's
    # own previous witness.
    b = bind_wit.get(i)
    w = np.array([home_map[n] for n in names], dtype=float) if b is None else b.copy()
    for n, v in chosen[j].items():          # EXACTLY the joints this candidate defines
      w[name_idx[n]] = v                    # (not "every mover joint, read from qm" -- qm is
    wit[i] = w                              # seeded from home and would carry a stale value
                                            # for any mover arm the candidate leaves unset)

  # ---- witness SETS (--witness-sets K): diverse same-grasp alternatives per step ---------
  # Selected from the SAME candidate pool the DP scored, gated exactly like the canonical
  # witness (hard non-penetration, the DP's own clearance floor, the grasp-relation filter)
  # PLUS a frozen-object gate: every grasped part must sit at the canonical member's world
  # pose (<=2mm/1deg). That last gate is what makes sets safe for handovers/add_grasps where
  # the object pose is genuinely free -- diversity then lives ONLY in the arms' IK branches
  # for the same EE poses (the wrist-basin diversity the planner needs), never in the object.
  # Member 0 is the DP's canonical witness itself, byte-equal to wit[i].
  wit_alts = {}
  if witness_sets and int(witness_sets) >= 2:
    _K = int(witness_sets)
    _SET_KINDS = ("pick", "place", "add_grasp", "handover")
    def _part_world(q, part):
      fkC.setJointState(q)
      f = fkC.getFrame(part)
      return (np.asarray(f.getPosition(), dtype=float),
              np.asarray(f.getRotationMatrix(), dtype=float))
    # ALL eligible plan steps, NOT just the ladder's -- the r9 endgame proved the difference:
    # its wall release pins arms whose last moves (handover 33, add_grasps 36/37) are outside
    # `steps` (the ladder harvested no candidates there), so the harvest skipped exactly the
    # tasks whose diversity the terminal reverse-seeding needs. The generator needs no ladder
    # candidates; for non-ladder steps the emitted witness itself is the base.
    _eligible = [i for i, a in enumerate(plan)
                 if a[0] in _SET_KINDS and wit[i] is not None]
    for i in _eligible:
      b = bind_wit.get(i)
      if b is None:
        b = wit[i]                                         # non-ladder step: emitted == base
      mv = movers.get(i) or _involved(plan[i], states[i], states[i + 1], robots, robot_ik)
      gp = grasped.get(i) if grasped.get(i) is not None else _grasped_parts(i)
      base_w = wit[i]
      ref_pw = {p: _part_world(base_w, p) for p in set(gp.values())}
      # ACTIVE branch enumeration (solver.branch_witnesses). The harvested pool alone is
      # STARVED: restarts land in the canonical basin and without ssik there is no analytic
      # enumerator, so picks offer 1-2 candidates within 0.05 rad of the canonical and
      # handovers' diversity lives in the object-moving dimension the gates must freeze
      # (measured on cross: ZERO sets from 110 pool candidates). The generator re-solves each
      # acting arm alone for its canonical tool pose from branch-flip seeds -- everything else
      # frozen at bind's witness -- and its finds join the SAME pool, gates and greedy
      # selection as the harvested candidates.
      gen_maps, gfun = robot_ik.branch_witnesses(
          tuple(plan[i]), states[i], states[i + 1], assembly, base_w, mv,
          max_seeds=12, time_budget=6.0, min_dist=float(set_spread))
      if verbose:
        print(f"[witsets-gen] step {i} {plan[i][0]}: {gfun['seeds']} seeds tried, "
              f"{gfun['solved']} converged, {gfun['gated']} gated, "
              f"{gfun['exact']} exact turns, {gfun['kept']} kept")
      scored2, gaps2, emitted = [], [], []
      for c in [dict(x) for x in cand.get(i, [])] + gen_maps:
        t, mn = clearance_at(dict(c), mv, i)
        scored2.append(t)
        gaps2.append(mn)
        w = b.copy()
        for n, v in c.items():
          w[name_idx[n]] = v
        emitted.append(w)
      best_c2 = max((scored2[k] for k in range(len(emitted)) if gaps2[k] >= 0.0),
                    default=0.0)
      # ALTERNATIVES ARE GATED LOOSER THAN THE CHAIN. The DP floor (best_a, often 0.99 when
      # clearance is free) exists to CHOOSE one chain; holding alternatives to it leaves only
      # the canonical candidate and no set at all (measured on cross: 110 candidates, 0 sets).
      # A member must be SAFE -- non-penetrating and at least half the step's best clearance --
      # not clearance-optimal; the planner picks among members by connectability, not comfort.
      alt_floor = min(best_a, 0.5) * best_c2
      pool = []
      n_gap = n_floor = n_grasp = n_pose = 0
      for k, w in enumerate(emitted):
        if gaps2[k] < 0.0:
          n_gap += 1
          continue
        if scored2[k] < alt_floor - 1e-9:
          n_floor += 1
          continue
        ok = True
        for r, part in gp.items():
          if r not in mv:
            continue                                       # inherited from bind: identical
          # judged against the canonical MEMBER (base_w), NOT bind's witness: at a handover
          # the DP may hold the object elsewhere than bind did (its grasp gate has a
          # least-bad fallback; measured 717mm on cross steps 19/33), and a member gated
          # against bind would then disagree with member 0 about the frozen object's pose.
          dp, da = _rel_delta(w, base_w, r, part)
          if dp * 1000.0 > 5.0 or da > 5.0:                # same grasp as the canonical member
            ok = False
            n_grasp += 1
            break
        if ok:
          for part, (p0, R0) in ref_pw.items():
            p1, R1 = _part_world(w, part)
            ca = (float(np.trace(R0.T @ R1)) - 1.0) / 2.0
            if (float(np.linalg.norm(p1 - p0)) > 2e-3
                or float(np.degrees(np.arccos(min(1.0, max(-1.0, ca))))) > 1.0):
              ok = False
              n_pose += 1
              break
        if ok:
          pool.append(w)
      # greedy farthest-point on the movers' WRIST joints (the IK-branch coordinates)
      last3 = [name_idx[n] for r in mv for n in arm_joints[r][-3:]]
      def _wd(a, bq):
        return float(np.linalg.norm(a[last3] - bq[last3]))
      members = [base_w.copy()]
      while len(members) < _K and pool:
        dmin = [min(_wd(w, m) for m in members) for w in pool]
        kb = int(np.argmax(dmin))
        if dmin[kb] < float(set_spread):
          break
        members.append(pool.pop(kb))
      if len(members) >= 2:
        wit_alts[i] = members
        if verbose:
          ds = [_wd(members[0], m) for m in members[1:]]
          print(f"[witsets] step {i} {plan[i][0]}: {len(members)} members, wrist spread "
                f"{min(ds):.2f}-{max(ds):.2f} rad")
      elif verbose:
        print(f"[witsets] step {i} {plan[i][0]}: NO set -- {len(emitted)} candidates, "
              f"dropped {n_gap} penetration / {n_floor} floor / {n_grasp} grasp / "
              f"{n_pose} object-pose; pool {len(pool)}"
              + (" (all within spread of the canonical)" if pool else ""))
    if verbose and wit_alts:
      print(f"[witsets] --witness-sets {_K}: {len(wit_alts)} step(s) carry witness sets")

  robot_ik.strip_inactive_collision = _strip0
  robot_ik.harvest_all_branches = _harvest0
  rep = {"alpha": best_a, "cost": cost, "crossings": crossings,
         "crossings_no_floor": ref, "front": front, "steps": len(steps),
         "candidates": sum(len(cand[i]) for i in steps),
         "seconds": time.perf_counter() - t0}
  if witness_sets and int(witness_sets) >= 2:
    rep["wit_alts"] = wit_alts
  if verbose:
    print(f"[witfloor] alpha={best_a:.2f}: {crossings} branch crossings "
          f"(={ref} with NO floor, so the clearance is free), {rep['candidates']} candidates "
          f"over {len(steps)} steps in {rep['seconds']:.1f}s")
    print("[witfloor] front  " + "  ".join(
        f"a={a:.2f}:" + ("infeasible" if x is None else f"{x}x") for a, x, _ in front))
  return wit, rep
