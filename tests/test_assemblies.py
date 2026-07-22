"""Regression net for the assembly planner: run a spread of assemblies through the
GEOMETRIC solver (no real-arm IK -- the cheap floating-EE feasibility layer only) and check
each returned plan is a genuine, feasible, goal-reaching sequence.

Kept deliberately fast (~20s total): geometry only for the bulk, a few SMALL wrench cases
(the wrench search expands far more, so large wrench instances are excluded on purpose).
Difficulty rises within each family (parts, robots).

The load-bearing assertion is `valid` (replay the plan through successors + cheap_feasible
and reach the goal) -- that invariant is hash-seed independent. Plan LENGTH is only bounded
loosely (`<= baseline + SLACK`): wA* is suboptimal and successor order is hash-ordered, so
the exact length jitters by ~±1-2 between runs. `LEN` below is the seed-0 baseline, kept as
documentation and the anchor for the upper bound.

Run:  python3 -m pytest tests/ -q      (from the repo root, or anywhere -- conftest fixes the path)
"""
import time

import pytest

import assembly as A
import wrench as W
from env import make_four_robot_env


def _solve(kind, n, robots, weight, wrench, model, insert):
  """Deterministic geometric solve; returns (plan, assembly, C, robots_list, gripper_kind)."""
  A.set_seed(0)
  A.set_rolls_per_face(1)                                   # vacuum floating EE
  if wrench:
    W.PARAMS["grasp_model"] = model
    W.PARAMS["insert_force"]["_default"] = insert
  asm = A.make_assembly(kind, n)
  C = make_four_robot_env(num_robots=robots, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots_list = A.get_robots(C)
  gk = A.gripper_kind(C) if wrench else None
  plan = A.solve_assembly_problem(C, asm, robots=robots_list, weight=weight,
                                  verbose=False, wrench=wrench, robot_ik=None)
  return plan, asm, C, robots_list, gk


def _replay_ok(plan, asm, C, robots_list, gk, bracing=False):
  """Independently re-derive the plan: every action must be an enumerated successor that
  passes the cheap feasibility layer, and the final state must be the goal."""
  checker, grasp_table = A._search_context(C, asm, robots_list)
  s = A.get_initial_state(C, asm)
  for act in plan:
    succ = [ns for a, ns in A.successors(s, asm, robots_list, grasp_table, gripper_kind=gk,
                                         bracing=bracing)
            if a == act]
    assert succ, f"action is not an enumerated successor: {act}"
    ns = succ[0]
    assert A.cheap_feasible(act, s, ns, asm, checker), f"action failed cheap layer: {act}"
    s = ns
  return s.is_goal(asm)


SLACK = 3                                                  # tolerated length jitter over baseline

# (kind, n_parts, robots, weight, baseline_len). weight inf == greedy (keeps the hardest
# geometric case fast). baseline_len is the seed-0 length; the test bounds len <= it + SLACK.
GEOMETRIC = [
    ("cross", 3, 2, 3.0, 6),
    ("cross", 5, 2, 3.0, 12),
    ("cross", 5, 3, 3.0, 11),
    ("cross", 7, 3, float("inf"), 21),                     # greedy: exact wA* is ~70s
    ("cube", 8, 3, 3.0, 18),
    ("arch-seq", 7, 2, 3.0, 16),
    ("arch-seq", 7, 3, 3.0, 16),
    ("cuboid-tower", 3, 2, 3.0, 4),                        # NON-CUBE: jenga tower (2:1:1 boxes,
    ("cuboid-tower", 4, 2, 3.0, 8),                        # 2/level, levels rotated 90 deg)
    ("cuboid-tower", 6, 2, 3.0, 13),
    ("car", 3, 2, 3.0, 8),                                 # ROTATED shapes: sideways wheels
    ("car", 5, 2, float("inf"), 15),                       # full car (wA* too slow for the suite)
    ("cube-plates", 4, 2, 3.0, 7),
    ("cube-plates", 6, 2, 3.0, 13),
    # 8-part: the two-plate build is FORCED (top plate assembled as a unit, then joined), which
    # makes wA* slow -- use greedy, which still produces the forced module join, fast.
    ("cube-plates", 8, 2, float("inf"), 20),
    ("cube-plates", 8, 3, float("inf"), 20),
]

# small WRENCH cases (asymmetric grasp model, 10 N insert). The wrench search is much
# heavier, so these stay tiny to keep the suite quick.
WRENCH = [
    ("cross", 3, 2, 3.0, "asymmetric", 10.0, 7),
    ("cross", 4, 2, 3.0, "asymmetric", 10.0, 12),
    ("arch-seq", 5, 2, 3.0, "asymmetric", 10.0, 14),
    ("cube-plates", 4, 2, 3.0, "asymmetric", 10.0, 16),   # +place/+release: on-table goal
    ("cuboid-tower", 4, 2, 3.0, "asymmetric", 10.0, 10),   # shaped support polygon + levers
    ("car", 3, 2, 3.0, "asymmetric", 10.0, 9),             # axial wheel press, rim contacts
]


@pytest.mark.parametrize("kind,n,robots,weight,exp_len", GEOMETRIC,
                         ids=[f"{k}-{n}p-{r}R" for k, n, r, _w, _l in GEOMETRIC])
def test_geometric(kind, n, robots, weight, exp_len):
  t0 = time.time()
  plan, asm, C, robots_list, gk = _solve(kind, n, robots, weight, False, None, None)
  assert plan is not None, f"{kind} n={n} R={robots}: no plan found"
  assert _replay_ok(plan, asm, C, robots_list, gk), "plan does not reach the goal on replay"
  assert len(plan) <= exp_len + SLACK, \
      f"plan length {len(plan)} >> baseline {exp_len} (regression?)"
  assert time.time() - t0 < 30.0, "case got unexpectedly slow"


@pytest.mark.parametrize("kind,n,robots,weight,model,insert,exp_len", WRENCH,
                         ids=[f"{k}-{n}p-{r}R-wr" for k, n, r, _w, _m, _i, _l in WRENCH])
def test_wrench(kind, n, robots, weight, model, insert, exp_len):
  plan, asm, C, robots_list, gk = _solve(kind, n, robots, weight, True, model, insert)
  assert plan is not None, f"{kind} n={n} R={robots} wrench: no plan found"
  assert _replay_ok(plan, asm, C, robots_list, gk), "wrench plan does not reach the goal"
  assert len(plan) <= exp_len + SLACK, \
      f"plan length {len(plan)} >> baseline {exp_len} (regression?)"


def _two_module_state(asm, C, robots, held_top):
  """Craft a state: bottom plate complete on the table, and the given top-plate cubes held
  as a module in the air (rooted at obj_4). Uses a brace grasp on obj_4 that is free in the
  assembled cube, and real scatter/table heights."""
  full = A._search_context(C, asm, robots)[1][A._subasm_key(list(asm.objects))]
  gid = next(g for p, g, _pose in full if p == "obj_4")
  # serpentine chains in chain-order naming: bottom 0->1->2->3, top 4->5->6->7 (held subset).
  top_parent = {"obj_5": ("part", "obj_4"), "obj_6": ("part", "obj_5"), "obj_7": ("part", "obj_6")}
  parent = {"obj_0": ("table",), "obj_1": ("part", "obj_0"), "obj_2": ("part", "obj_1"),
            "obj_3": ("part", "obj_2"), "obj_4": ("gripper", "a1")}
  supports = {"obj_0": frozenset({("table",)}), "obj_1": frozenset({("mate", "obj_0")}),
              "obj_2": frozenset({("mate", "obj_1")}), "obj_3": frozenset({("mate", "obj_2")}),
              "obj_4": frozenset({("grasp", "a1", "obj_4", gid)})}
  for x in ("obj_5", "obj_6", "obj_7"):
    if x in held_top:
      parent[x] = top_parent[x]
      supports[x] = frozenset({("mate", top_parent[x][1])})
  poses = A.get_initial_state(C, asm).poses
  return A.State(parent=parent, place={"obj_0": ("slot", 0, 0)}, supports=supports,
                 poses=dict(poses))


def test_wrench_uncertainty_flags():
  """The two toggleable insertion-robustness knobs. OFF => byte-identical (a singleton load
  variant); the CURVED-torque derate makes holding the car by a WHEEL (a curved grip) unable
  to resist an off-centre insertion torque that a flat BODY grip still takes -- the emergent
  'don't hold by the wheel' without forbidding any action by hand."""
  A.set_seed(0); A.set_rolls_per_face(2)
  car = A.make_assembly("car", 5)
  car.no_hold = set()      # this test isolates the curved/noise WRENCH mechanism -- it holds the
                           # car by a wheel on purpose, which the no_hold POLICY (tested separately
                           # in test_no_hold_parts) would otherwise forbid.
  sw = A.part_grasp_surfaces(car.part_shape("obj_1"))
  sb = A.part_grasp_surfaces(car.part_shape("obj_0"))
  g_wheel = next(i for i, s in enumerate(sw) if s == "curved")
  g_body = next(i for i, s in enumerate(sb) if s == "flat")
  poses = A._flat_poses(car)

  def held(holder, gid):                                    # {obj_0,obj_1,obj_2} held in air
    parent, supports, place = {}, {}, {}
    for q in car.objects:
      if q in ("obj_0", "obj_1", "obj_2"):
        parent[q] = ("gripper", "zz") if q == "obj_0" else ("part", "obj_0")
        supports[q] = frozenset() if q == "obj_0" else frozenset({("mate", "obj_0")})
      else:
        parent[q] = ("table",); supports[q] = frozenset({("table",)})
        place[q] = ("slot", A._part_index(q), A._IDENTITY_FACE)
    supports[holder] = supports[holder] | {("grasp", "zz", holder, gid)}
    return A.State(parent=parent, place=place, supports=supports, poses=dict(poses))

  def gate(holder, gid):
    st = held(holder, gid)
    return A._group_stable_loaded(st, car, {"obj_0", "obj_1", "obj_2"}, "two_finger",
                                  A._insertion_load_variants(st, car, "obj_3", "obj_0"))

  saved = (W.PARAMS.get("grasp_model"), W.PARAMS.get("curved_grasp_torque_scale"),
           W.PARAMS.get("insertion_uncertainty"))
  try:
    W.PARAMS["grasp_model"] = "isotropic"
    W.PARAMS["curved_grasp_torque_scale"] = None
    W.PARAMS["insertion_uncertainty"] = None
    st = held("obj_0", g_body)
    # off: one load variant, and both holds feasible (the pre-change behaviour)
    assert len(A._insertion_load_variants(st, car, "obj_3", "obj_0")) == 1
    assert gate("obj_1", g_wheel) and gate("obj_0", g_body)
    # curved derate: the WHEEL grip can no longer resist the insertion torque; body still can
    W.PARAMS["curved_grasp_torque_scale"] = 0.2
    assert not gate("obj_1", g_wheel)
    assert gate("obj_0", g_body)
    # insertion noise = an uncertainty BOX, returned as its CORNERS (exact, no sampling):
    # lateral only => a 2-D box => 4 corners; lateral+offset => a 4-D box => 16 corners.
    W.PARAMS["insertion_uncertainty"] = {"lateral": 0.3, "offset": 0.0}
    assert len(A._insertion_load_variants(st, car, "obj_3", "obj_0")) == 4
    W.PARAMS["insertion_uncertainty"] = {"lateral": 0.3, "offset": 0.004}
    assert len(A._insertion_load_variants(st, car, "obj_3", "obj_0")) == 16
  finally:
    W.PARAMS["grasp_model"], W.PARAMS["curved_grasp_torque_scale"], \
        W.PARAMS["insertion_uncertainty"] = saved


def test_equilibrium_backend_agrees():
  """The warm-startable LP backend (highspy persistent model / numpy warm-basis pool) must give
  the SAME feasibility verdict as a cold scipy solve on the same (G, ub, wrench) -- across many
  loads sharing a geometry, which is exactly when the warm path is exercised."""
  import numpy as np
  from scipy.optimize import linprog
  rng = np.random.default_rng(7)
  for _ in range(300):
    K = int(rng.integers(6, 30))
    G = rng.standard_normal((6, K)); ub = np.abs(rng.standard_normal(K)) * 3 + 0.5
    gk = (G.tobytes(), ub.tobytes())
    for _ in range(5):                                       # several loads on the SAME geometry
      w = rng.standard_normal(6) * rng.uniform(0.1, 5)
      got = W.equilibrium_feasible_matrices(G, ub, w, geo_key=gk)
      truth = bool(linprog(np.zeros(K), A_eq=G, b_eq=-w,
                           bounds=[(0.0, float(u)) for u in ub], method="highs").success)
      assert got == truth


def test_robust_affine_sound():
  """equilibrium_robust_affine (the CONSERVATIVE robust mode) must be SOUND: whenever it reports
  feasible, every corner of the uncertainty box is genuinely balanceable. It may reject some
  corner-feasible loads (that is the conservatism), but it must never accept an infeasible one."""
  import numpy as np
  rng = np.random.default_rng(8)
  unsound = 0
  for _ in range(800):
    K = int(rng.integers(6, 24)); G = rng.standard_normal((6, K)); ub = np.abs(rng.standard_normal(K)) * 3 + 0.5
    w0 = rng.standard_normal(6) * rng.uniform(0.1, 4)
    D = [rng.standard_normal(6) * rng.uniform(0.05, 1.0) for _ in range(2)]
    if W.equilibrium_robust_affine(G, ub, w0, D):
      corners = [w0 + s1 * D[0] + s2 * D[1] for s1 in (-1, 1) for s2 in (-1, 1)]
      if not all(W.equilibrium_feasible_matrices(G, ub, w) for w in corners):
        unsound += 1
  assert unsound == 0


def test_no_hold_parts():
  """assembly.no_hold: a grasp on a no_hold part (a car WHEEL) may hold only the LONE part (the
  inserter) -- never a multi-part subassembly. So you can put a wheel on the car, but can't then
  hold the car BY it. (Fast: crafted states, no search.)"""
  A.set_seed(0); A.set_rolls_per_face(2)
  saved = W.PARAMS.get("grasp_model")
  try:
    W.PARAMS["grasp_model"] = "asymmetric"; W.PARAMS["insertion_uncertainty"] = None
    asm = A.make_assembly("car-mini", 3)
    assert asm.no_hold == {"obj_1", "obj_2"} and not asm.can_hold("obj_1") and asm.can_hold("obj_0")
    C = make_four_robot_env(num_robots=3, floating_ee=True, vacuum=False)
    A.add_disassembled_obj(C, asm, spacing=0.2)
    gk = A.gripper_kind(C); poses = A._flat_poses(asm)

    def held(holder):                                       # {obj_0,obj_1} welded, held by holder
      parent = {"obj_0": ("gripper", "a1"), "obj_1": ("part", "obj_0"), "obj_2": ("table",)}
      supports = {"obj_0": frozenset(), "obj_1": frozenset({("mate", "obj_0")}),
                  "obj_2": frozenset({("table",)})}
      supports[holder] = supports[holder] | {("grasp", "a1", holder, 0)}
      return A.State(parent=parent, place={"obj_2": ("slot", 2, 5)}, supports=supports,
                     poses=dict(poses))

    assert not held("obj_1").is_supported({"obj_0", "obj_1"}, asm, gk), "wheel must not hold the car"
    assert held("obj_0").is_supported({"obj_0", "obj_1"}, asm, gk), "body may hold the car"
    # the lone wheel (inserter) is still holdable
    lone = A.State(parent={"obj_1": ("gripper", "a1"), "obj_0": ("table",), "obj_2": ("table",)},
                   place={}, poses=dict(poses),
                   supports={"obj_1": frozenset({("grasp", "a1", "obj_1", 0)}),
                             "obj_0": frozenset({("table",)}), "obj_2": frozenset({("table",)})})
    assert lone.is_supported({"obj_1"}, asm, gk), "a lone wheel (the inserter) stays holdable"
    # all-cube assemblies are unaffected (empty no_hold)
    assert A.make_assembly("cube", 4).no_hold == set()
  finally:
    W.PARAMS["grasp_model"] = saved


def test_cograsp_bracing():
  """The add_grasp (co-grasp) primitive + its labeled-h edge. An OFFSET HORIZONTAL mate (a car
  wheel: ±y press at an x-offset -> a yaw moment) needs TWO braces -- a single grasp can't supply
  the yaw couple and the frictionless table gives no lateral help. So WITHOUT bracing the instance
  is infeasible (labeled-h h(s0)=inf); WITH bracing (+ the abstraction's inverse-add_grasp edge)
  it's feasible and a real plan uses add_grasp. `bracing` must also toggle the primitive."""
  import solver as S
  A.set_seed(0); A.set_rolls_per_face(2)                    # two-finger
  saved = (W.PARAMS.get("grasp_model"), W.PARAMS.get("insertion_uncertainty"),
           W.PARAMS["insert_force"].get("_default"))
  try:
    W.PARAMS["grasp_model"] = "asymmetric"; W.PARAMS["insert_force"]["_default"] = 20.0
    W.PARAMS["insertion_uncertainty"] = {"lateral": 0.05, "offset": 0.0}
    asm = A.make_assembly("car-mini", 3)
    C = make_four_robot_env(num_robots=3, floating_ee=True, vacuum=False)
    A.add_disassembled_obj(C, asm, spacing=0.2)
    robots = A.get_robots(C); gk = A.gripper_kind(C)
    checker, gt = S._search_context(C, asm, robots)
    s0 = A.get_initial_state(C, asm)
    # the primitive is gated by `bracing`: add_grasp on a held body appears only when it's on
    held = next(ns for a, ns in A.picks(s0, asm, robots) if a[2] == "obj_0")
    def has_addgrasp(bracing):
      return any(a[0] == "add_grasp" and a[2] == "obj_0" for a, _ in
                 A.successors(held, asm, robots, gt, gripper_kind=gk, bracing=bracing))
    assert not has_addgrasp(False) and has_addgrasp(True), "bracing must toggle add_grasp"
    # WITHOUT bracing: soundly infeasible (labeled-h abstraction has no co-grasp edge)
    gp = getattr(checker, "fact_gripper_path", checker.gripper_path)
    h_off = S.LabeledDisassemblyHeuristic(asm, len(robots), gt, gripper_path=gp,
                                          gripper_kind=gk, bracing=False)
    assert h_off.value(s0) == float("inf"), "offset horizontal mate needs a co-grasp brace"
    # WITH bracing: a real plan exists, uses add_grasp, and replays to the goal
    plan = A.solve_assembly_problem(C, asm, robots=robots, weight=10.0, verbose=False,
                                    wrench=True, robot_ik=None, labeled_h=True, bracing=True)
    assert plan is not None and any(a[0] == "add_grasp" for a in plan)
    assert _replay_ok(plan, asm, C, robots_list=robots, gk=gk, bracing=True)
  finally:
    (W.PARAMS["grasp_model"], W.PARAMS["insertion_uncertainty"],
     W.PARAMS["insert_force"]["_default"]) = saved


def test_mini_car_two_finger():
  """car-mini + the TWO-FINGER gripper: the scaled wheels (diameter 0.048 < the ~0.078
  finger gap) are graspable ACROSS THE DIAMETER -- the curved-surface radial grasps the
  vacuum filters out. Asserts the radial grasps exist in the table and a plan solves."""
  A.set_seed(0)
  A.set_rolls_per_face(2)                                  # two-finger
  asm = A.make_assembly("car-mini", 3)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=False)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  checker, gt = A._search_context(C, asm, robots)
  wheel_ids = sorted(g for _p, g, _pose in gt[("obj_1",)])
  radial = {0, 1, 2, 3, 6, 7, 8, 9}                        # non-cap directions x rolls
  assert radial <= set(wheel_ids), f"radial wheel grasps missing: {wheel_ids}"
  plan = A.solve_assembly_problem(C, asm, robots=robots, weight=3.0, verbose=False,
                                  robot_ik=None)
  assert plan is not None and len(plan) <= 8 + SLACK
  assert _replay_ok(plan, asm, C, robots_list=robots, gk=None)


def test_part_shapes():
  """Shape plumbing: per-part specs flow into every consumer, cubes stay byte-identical.
  (a) the cube grasp DB reproduces cube_grasp_pose exactly for both roll settings;
  (b) cuboid grasps sit at per-face standoffs (end caps at w + offset, side faces w/2 + o);
  (c) rest heights are orientation-aware (laid-down w/2, tipped upright w)."""
  import numpy as np
  for rolls in (1, 2):
    A.set_rolls_per_face(rolls)
    db = A.part_grasp_poses(("box", [0.05, 0.05, 0.05, 0.005]), rolls=rolls)
    assert len(db) == 6 * rolls
    for gid in range(6 * rolls):
      assert np.allclose(db[gid], A.cube_grasp_pose(gid, 0.05), atol=1e-12)
  A.set_rolls_per_face(1)
  asm = A.make_assembly("cuboid-tower", 4)
  assert asm.part_shape("obj_0") == ("box", [0.1, 0.05, 0.05, 0.005])
  standoffs = [round(float(np.linalg.norm(g[:3])), 3)
               for g in A.part_grasp_poses(asm.part_shape("obj_0"))]
  assert standoffs == [0.06, 0.06, 0.035, 0.035, 0.035, 0.035], standoffs
  flat = A._group_rest_height(asm, {"obj_0"}, "obj_0", A._IDENTITY_FACE)
  upright = A._group_rest_height(asm, {"obj_0"}, "obj_0", 0)
  assert abs(flat - 0.025) < 1e-9 and abs(upright - 0.05) < 1e-9, (flat, upright)


def test_module_connect():
  """The two-plate build folded into `assemble`: the top plate {obj_4..7} is an independent
  module, and mating it onto the built bottom plate is a single `assemble` on the module
  (seam obj_4). Asserts (a) the COMPLETE module joins to reach the goal, and (b) a PARTIAL
  module cannot -- the join is FORCED to wait for the whole plate (connect_requires = the
  last chain cube)."""
  A.set_seed(0)
  A.set_rolls_per_face(1)
  asm = A.make_assembly("cube-plates", 8)
  C = make_four_robot_env(num_robots=3, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  checker, _gt = A._search_context(C, asm, robots)

  # the top plate is an independently buildable subassembly.
  subs = [set(s) for s in A.get_subassemblies(asm)]
  assert {"obj_4", "obj_5", "obj_6", "obj_7"} in subs, "top plate is not a module ideal"

  # (a) COMPLETE top module -> a single `assemble` joins it and reaches the goal.
  s_full = _two_module_state(asm, C, robots, {"obj_5", "obj_6", "obj_7"})
  joins = [(a, ns) for a, ns in A.assembles(s_full, asm, robots)
           if a[2] == "obj_4" and a[3] == "obj_0" and A.cheap_feasible(a, s_full, ns, asm, checker)]
  assert joins, "no cheap-feasible module join for the complete top plate"
  act, ns = joins[0]
  assert act[0] == "assemble", "the join must be a plain `assemble`, not a separate action"
  assert len(next(iter(ns.subassemblies()))) == 8, "join did not fuse into one 8-part body"
  assert ns.is_complete(asm), "join did not complete the assembly"   # still held ->
  # not is_goal under the on-table default; this test checks the JOIN, not the goal

  # (b) PARTIAL top module (missing the last chain cube obj_7) -> the join is FORBIDDEN.
  s_part = _two_module_state(asm, C, robots, {"obj_5", "obj_6"})
  partial = [a for a, _ns in A.assembles(s_part, asm, robots)
             if a[2] == "obj_4" and a[3] == "obj_0"]
  assert not partial, "a partial top plate must not be joinable (forcing broken)"


def test_inhand_reorient():
  """v1 IN-HAND REORIENT macro (flag-gated): a held lone part offers a reorient edge to
  every other grasp (full-connectivity assumption); the edge is well-formed and passes the
  cheap layer; a solve with the flag on still reaches the goal and re-derives; and with the
  flag OFF no reorient edges are emitted (default path untouched)."""
  A.set_seed(0)
  A.set_rolls_per_face(1)
  asm = A.make_assembly("cross", 4)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  checker, gt = A._search_context(C, asm, robots)
  s0 = A.get_initial_state(C, asm)
  act, held = next((a, ns) for a, ns in A.picks(s0, asm, robots, grasp_table=gt)
                   if A.cheap_feasible(a, s0, ns, asm, checker))
  p, old_gid = act[2], act[3]

  # flag OFF (default): no reorient successors anywhere
  assert not [a for a, _ in A.successors(held, asm, robots, gt)
              if a[0] == "inhand_reorient"], "reorient emitted with flag off"

  A.set_inhand_reorient(True)
  try:
    re = [(a, ns) for a, ns in A.successors(held, asm, robots, gt)
          if a[0] == "inhand_reorient"]
    assert re, "no reorient edges from a held lone part"
    assert all(a[1] == act[1] and a[2] == p and a[3] != old_gid for a, _ in re), \
        "reorient must keep the holder + part and change the grasp"
    a2, ns2 = re[0]
    assert ns2.grasp[p] == a2[3], "new grasp not recorded"
    assert ns2.parent[p] == ("gripper", act[1]), "part must stay gripper-parented"
    grasps = [c for c in ns2.supports[p] if c[0] == "grasp"]
    assert grasps == [("grasp", act[1], p, a2[3])], "supports must carry exactly the new grasp"
    assert A.cheap_feasible(a2, held, ns2, asm, checker)

    plan = A.solve_assembly_problem(C, asm, robots=robots, weight=3.0, verbose=False,
                                    wrench=False, robot_ik=None)
    assert plan is not None
    assert _replay_ok(plan, asm, C, robots, None)
  finally:
    A.set_inhand_reorient(False)


def test_staged_mates():
  """STAGED/PARTIAL mates (A11, flag-gated): slot-pin's bar mate is a SLOT until the pin
  lands. Flag ON: (a) the orientation fact -- upright the slot cannot hold the bar alone,
  slot-opening-up it can; (b) releasing the bar's grasp before the pin is NOT a successor
  (hold-until-fastened emerges), and is again once the pin is mated; (c) the exact solve
  reaches the goal, re-derives, and orders the bar's release AFTER the pin; (d) the gated
  labeled h finds the same optimal cost (admissibility spot check). Flag OFF: the early
  release IS offered (baseline untouched)."""
  import numpy as np
  A.set_seed(0)
  A.set_rolls_per_face(1)
  asm = A.make_assembly("slot-pin", 3)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  s0 = A.get_initial_state(C, asm)
  _, gt = A._search_context(C, asm, robots)

  # (a) the per-(part, face) slot fact
  assert not A._slot_holds_alone(asm, "obj_1", A._IDENTITY_FACE), "upright cantilever held?!"
  up = next(f for f, n in enumerate(A._CUBE_FACE_NORMALS) if np.allclose(n, [-1.0, 0, 0]))
  assert A._slot_holds_alone(asm, "obj_1", up), "slot-opening-up face must hold the bar"

  def apply(s, match):
    return next(ns for a, ns in A.successors(s, asm, robots, gt) if match(a))

  def bar_releases(s):
    return [a for a, _ in A.successors(s, asm, robots, gt)
            if a[0] == "release" and a[2] == "obj_1"]

  s1 = apply(s0, lambda a: a[0] == "pick" and a[2] == "obj_1")
  s2 = apply(s1, lambda a: a[0] == "assemble" and a[2] == "obj_1")   # mated, unfastened
  assert bar_releases(s2), "flag off: the early release must be offered (baseline)"

  A.set_staged_mates(True)
  try:
    # (b) hold-until-fastened: no release while unfastened; legal again after the pin
    assert not bar_releases(s2), "unfastened bar released -- staged gate failed"
    s3 = apply(s2, lambda a: a[0] == "pick" and a[2] == "obj_2")
    s4 = apply(s3, lambda a: a[0] == "assemble" and a[2] == "obj_2")
    assert bar_releases(s4), "release must be legal once fastened"

    # (c) end-to-end: solves, re-derives, and the ordering is forced in the plan
    plan = A.solve_assembly_problem(C, asm, robots=robots, weight=1.0, verbose=False,
                                    wrench=False, robot_ik=None)
    assert plan is not None
    assert _replay_ok(plan, asm, C, robots, None)
    i_pin = next(i for i, a in enumerate(plan) if a[0] == "assemble" and a[2] == "obj_2")
    i_rel = next(i for i, a in enumerate(plan) if a[0] == "release" and a[2] == "obj_1")
    assert i_pin < i_rel, "bar released before the pin landed"

    # (d) gated labeled h: same optimal cost as the plain-h exact solve
    plan_lab = A.solve_assembly_problem(C, asm, robots=robots, weight=1.0, verbose=False,
                                        wrench=False, robot_ik=None, labeled_h=True)
    assert plan_lab is not None and len(plan_lab) == len(plan)
  finally:
    A.set_staged_mates(False)


def test_staged_press_noise():
  """Per-part PRESS check on staged parts (+--insertion-noise corners): pressing the pin
  onto the mated-but-unfastened bar. UPRIGHT geometry: the nominal press is vertical at
  zero lever -> the slot walls take it even UNHELD; a noisy press adds shear -> a couple at
  the interface lever -> the point slot fails -> the bar must be HELD while its fastener is
  pressed, even though it rests stably. A grasped bar passes both. (The 1-robot slot-up
  route dies under wrench for the same reason: there the press is lateral, so even the
  NOMINAL press torques the unheld bar -- checked via the end-to-end matrix, not here.)"""
  A.set_seed(0)
  A.set_rolls_per_face(1)
  W.PARAMS["grasp_model"] = "isotropic"
  W.PARAMS["insert_force"]["_default"] = 10.0
  asm = A.make_assembly("slot-pin", 3)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  gk = A.gripper_kind(C)
  _, gt = A._search_context(C, asm, robots)
  s0 = A.get_initial_state(C, asm)

  def apply(s, match):
    return next(ns for a, ns in A.successors(s, asm, robots, gt) if match(a))

  held = apply(apply(s0, lambda a: a[0] == "pick" and a[2] == "obj_1"),
               lambda a: a[0] == "assemble" and a[2] == "obj_1")   # bar mated, grasp kept
  unheld = apply(held, lambda a: a[0] == "release" and a[2] == "obj_1")  # flag off: legal

  A.set_staged_mates(True)
  try:
    nom = A._insertion_load_variants(held, asm, "obj_2", "obj_1")
    assert len(nom) == 1, "noise off must yield the single nominal load"
    assert A._staged_press_stable(held, asm, "obj_1", gk, nom), "grasped bar must take the press"
    # UNHELD upright fails even nominally: the point slot's wall reaction acts at the SEAT
    # (offset from the centroid), so a single contact only carries forces along the
    # centroid<->seat line -- a vertical press through a SIDE slot is unbalanceable. (This is
    # the same fact as _slot_holds_alone(upright)=False; the release gate already forbids
    # reaching this state, the press gate is consistent with it.)
    assert not A._staged_press_stable(unheld, asm, "obj_1", gk, nom)
    W.PARAMS["insertion_uncertainty"] = {"lateral": 0.3}
    noisy = A._insertion_load_variants(held, asm, "obj_2", "obj_1")
    assert len(noisy) > 1, "noise on must yield box corners"
    assert A._staged_press_stable(held, asm, "obj_1", gk, noisy), \
        "a grasp on the bar must absorb the noisy press"
  finally:
    W.PARAMS["insertion_uncertainty"] = None
    A.set_staged_mates(False)

  # the NOISE-FLIP itself, isolated at the wrench level (the slot-pin geometry has no state
  # where nominal passes unheld -- the pin presses PERPENDICULAR to the bar's slot): a press
  # ALONG the slot axis of a slot-opening-UP part. Nominal = pure compression into the seat,
  # zero torque -> feasible with NO grasp; the noise corners add lateral shear at the
  # centroid<->seat lever -> a couple the point slot cannot resist -> infeasible.
  import numpy as np
  h, m, F = 0.025, 0.1, 10.0
  seat, axis = np.array([0.0, 0.0, -h]), np.array([0.0, 0.0, -1.0])   # slot opening UP
  G, ub = W.cap_slot(seat, axis)
  bounds = [(0.0, None) for _ in ub]
  grav = W.gravity_wrench([(m, np.zeros(3))])

  def press_ok(variants):
    return all(W.equilibrium_feasible_matrices(
        G, ub, grav + W.point_force_wrench(f, p), bounds=bounds) for f, p in variants)

  f_nom, pt = np.array([0.0, 0.0, -F]), np.array([0.0, 0.0, h])       # pressed down, from above
  assert press_ok(A._insertion_uncertainty_variants(f_nom, pt, F)), \
      "axis-aligned nominal press must be pure compression (noise off)"
  W.PARAMS["insertion_uncertainty"] = {"lateral": 0.3}
  try:
    corners = A._insertion_uncertainty_variants(f_nom, pt, F)
    assert len(corners) > 1
    assert not press_ok(corners), "noisy shear couple must defeat the unheld point slot"
  finally:
    W.PARAMS["insertion_uncertainty"] = None
