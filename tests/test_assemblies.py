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


def test_vacuum_engagement_force_is_a_pre_contact_support_load():
  """The cup's sealing push is checked before the new vacuum contact can help balance it.

  A cube on the frictionless table accepts a top-down push, but cannot balance a horizontal
  side push. A two-finger acquisition is unaffected by this vacuum-specific transient.
  """
  A.set_rolls_per_face(1)
  asm = A.make_assembly("cross", 1)
  C = make_four_robot_env(num_robots=1, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  state = A.get_initial_state(C, asm)
  group = state.subassemblies()[0]
  saved = W.PARAMS.get("vacuum_engage_force", 0.0)
  W.PARAMS["vacuum_engage_force"] = 5.0
  try:
    kinds = {"a1": "vacuum"}
    # With one roll per face, ids 4 and 0 are the +z (top) and +x (side) surface grasps.
    assert A._vacuum_engagement_stable(state, asm, group, "a1", "obj_0", 4, kinds)
    assert not A._vacuum_engagement_stable(state, asm, group, "a1", "obj_0", 0, kinds)
    assert A._vacuum_engagement_stable(
        state, asm, group, "a1", "obj_0", 0, {"a1": "two_finger"})
  finally:
    W.PARAMS["vacuum_engage_force"] = saved


def test_release_reroots_same_part_cograsps_by_robot_roster():
  """Releasing the primary holder must choose a deterministic remaining root.

  Chair ends with a1 and a3 co-grasping obj_0 after a2 releases it, followed by
  ``place a1``.  Both remaining contacts are on the same part, so a part-only tie-break
  leaked frozenset/PYTHONHASHSEED order into the symbolic state and intermittently made
  that saved place action impossible to replay.
  """
  contacts = frozenset({("grasp", "a1", "obj_0", 8),
                        ("grasp", "a2", "obj_0", 6),
                        ("grasp", "a3", "obj_0", 1)})
  state = A.State(parent={"obj_0": ("gripper", "a2")},
                  grasp={"obj_0": 6}, supports={"obj_0": contacts})

  def after_a2_release(roster):
    return next(ns for action, ns in A.releases(state, None, roster)
                if action == ("release", "a2", "obj_0"))

  assert after_a2_release(["a1", "a2", "a3"]).parent["obj_0"] == ("gripper", "a1")
  # The selection follows the explicit canonical roster, not lexical or set order.
  assert after_a2_release(["a3", "a2", "a1"]).parent["obj_0"] == ("gripper", "a3")


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
    ("fixture-rack", 3, 2, 3.0, 6),                       # rail + top-down post + lateral arm
    ("fixture-rack", 5, 3, float("inf"), 13),             # two independent station branches
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


def test_fixture_rack_lateral_press_is_solved_with_support_actions():
  """Unlike the small unbraced WRENCH table above, the rack intentionally needs the target
  actively supported during its horizontal bracket press. With bracing/support successors
  enabled, the geometric+wrench solver must find and replay that manipulation sequence."""
  A.set_seed(0)
  A.set_rolls_per_face(1)
  W.PARAMS["grasp_model"] = "asymmetric"
  W.PARAMS["insert_force"]["_default"] = 10.0
  asm = A.make_assembly("fixture-rack", 3)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  gk = A.gripper_kind(C)
  plan = A.solve_assembly_problem(C, asm, robots=robots, weight=3.0, verbose=False,
                                  wrench=True, robot_ik=None, bracing=True)
  assert plan is not None
  assert _replay_ok(plan, asm, C, robots, gk, bracing=True)
  assert len(plan) <= 8 + SLACK


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


def test_milestone_fill_releases_inherited_grasp():
  """Regression for the STRANDED-ARM milestone-fill bug (root-caused 2026-07-24 via captured-
  state replay). A milestone fill may ENTER holding a part it does not own: the previous fill
  can end held-through (<=1 holder persists), so the next fill inherits a grasp on a part
  OUTSIDE its own `relevant = ideal u {p}` (e.g. an arm still bracing the settled stool while
  the backrest MODULE is filled). The fill's successor filter drops actions on non-relevant
  parts -- but `release` MUST stay exempt, else that arm is STRANDED for the whole fill (a
  module needing >1 arm then goes infeasible -> the 60-repair backjump cascade, 380s).

  Crafted so the exemption is FORCED: a1 braces obj_2 (a resting, NON-relevant part), so the
  fill goal (nothing loose) is unreachable UNLESS obj_2's grasp is released. WITH the exemption
  the fill succeeds and its subplan releases obj_2; WITHOUT it the fill returns None (loose
  never clears) and this test fails -- the direct guard for the one-line fix. (Fast: one
  crafted state, one mini-fill.)"""
  import solver as S
  A.set_seed(0); A.set_rolls_per_face(1)
  asm = A.make_assembly("cross", 3)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  checker, gt = S._search_context(C, asm, robots)
  s0 = A.get_initial_state(C, asm)
  gid = next(g for _p, g, _pose in gt[("obj_2",)])
  # a1 braces obj_2 while it rests: an INHERITED grasp on a part outside the fill's relevant set
  sup = dict(s0.supports)
  sup["obj_2"] = frozenset(sup["obj_2"]) | {("grasp", "a1", "obj_2", gid)}
  start = A.State(parent=dict(s0.parent), place=dict(s0.place), supports=sup, poses=dict(s0.poses))
  assert any(a[2] == "obj_2" for a, _ in A.releases(start, asm, robots)), \
      "precondition: bracing a resting part must itself be releasable"

  # fill obj_1 into ideal={obj_0}; obj_2 (braced by a1) is NON-relevant to this fill
  sub, end = S._add_one_part(start, "obj_1", {"obj_0"}, asm, robots, checker, gt, None,
                             weight=5.0, max_nodes=20000, verbose=False)
  assert sub is not None, "fill STRANDED: inherited non-relevant grasp could not be released"
  assert any(a[0] == "release" and a[2] == "obj_2" for a in sub), \
      "the fill must clear the inherited grasp via the release exemption"
  assert S._part_mated(end, "obj_1"), "fill did not mate its part"
  assert not any(c[0] == "grasp" for c in end.supports.get("obj_2", ())), "obj_2 left stranded"

  # control: WITHOUT the inherited brace the same fill needs NO release of obj_2 -- proving the
  # release above is caused by the inherited grasp, not incidental to building obj_1.
  sub0, _ = S._add_one_part(s0, "obj_1", {"obj_0"}, asm, robots, checker, gt, None,
                            weight=5.0, max_nodes=20000, verbose=False)
  assert sub0 is not None and not any(a[0] == "release" and a[2] == "obj_2" for a in sub0)


def test_check_initial_scene():
  """The initial-scene validator (collision-free + every part graspable by >=1 robot). The
  default scatter line is fully usable; a tight --grid-init block is collision-free but buries
  the interior cubes so a SIDE-grasp two-finger reaches only the protruding one (NOT usable,
  caught up front); the VACUUM top-pick reaches every cube in the same grid (usable). Purely an
  INITIAL-STATE check -- the assembly/topology is never exercised."""
  # default scatter line, two-finger: collision-free, all 7 graspable
  A.set_seed(0); A.set_rolls_per_face(2)
  asm = A.make_assembly("cross", 7)
  C = make_four_robot_env(num_robots=4, floating_ee=True, vacuum=False)
  A.add_disassembled_obj(C, asm)
  ok, rep = A.check_initial_scene(C, asm, verbose=False)
  assert ok and rep["collision_free"] and not rep["ungraspable"], rep

  # tight 2-wide grid, two-finger: collision-free but the block is frozen (only the +1 cube)
  asm2 = A.make_assembly("cross", 7)
  C2 = make_four_robot_env(num_robots=4, floating_ee=True, vacuum=False)
  A.add_disassembled_obj(C2, asm2, grid=2)
  ok2, rep2 = A.check_initial_scene(C2, asm2, verbose=False)
  assert rep2["collision_free"] and not ok2 and len(rep2["ungraspable"]) >= 5, rep2

  # same tight grid, VACUUM: top-pick reaches every cube -> usable
  A.set_rolls_per_face(1)
  asm3 = A.make_assembly("cross", 7)
  C3 = make_four_robot_env(num_robots=4, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C3, asm3, grid=2)
  ok3, rep3 = A.check_initial_scene(C3, asm3, verbose=False)
  assert ok3 and not rep3["ungraspable"], rep3


def test_milestone_module_fill():
  """The MILESTONE planner on a MODULE assembly -- the solver family the stranded-arm fix lives
  in (the net had no milestone case before). cube-plates-8 builds the top plate {obj_4..7} as a
  DETACHED module, then JOINS it; module fills inherit grasps across milestones exactly as the
  fixed path requires. Asserts a valid, goal-reaching plan that actually performs the module
  join (so the module path is genuinely exercised), within a loose time bound -- the bug
  manifested as a ~20x slowdown, so the bound also guards the repair-cascade regression."""
  A.set_seed(0); A.set_rolls_per_face(1)
  asm = A.make_assembly("cube-plates", 8)
  C = make_four_robot_env(num_robots=3, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  t0 = time.time()
  plan = A.solve_assembly_problem_milestones(C, asm, robots=robots, weight=5.0, verbose=False)
  assert plan is not None, "milestones found no plan for the module assembly"
  assert _replay_ok(plan, asm, C, robots_list=robots, gk=None), "milestone plan misses the goal"
  assert any(a[0] == "assemble" and a[2] == "obj_4" and a[3] == "obj_0" for a in plan), \
      "plan never performs the module join -- the module path was not exercised"
  assert len(plan) <= 21 + SLACK, f"milestone plan length {len(plan)} >> baseline 21 (regression?)"
  assert time.time() - t0 < 40.0, "milestone module fill got unexpectedly slow (repair cascade?)"


def test_structural_milestones_skip_the_global_database(monkeypatch):
  """The ablation must not quietly build a differently-shaped global abstraction."""
  A.set_seed(0); A.set_rolls_per_face(1)
  asm = A.make_assembly("cross", 3)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)

  def forbidden(*_args, **_kwargs):
    raise AssertionError("structural milestone mode constructed a global heuristic")

  monkeypatch.setattr(A, "SignatureCollapsedHeuristic", forbidden)
  monkeypatch.setattr(A, "GraspCollapsedHeuristic", forbidden)
  plan = A.solve_assembly_problem_milestones(
      C, asm, robots=robots, weight=5.0, verbose=False,
      milestone_oracle="structural", milestone_boundaries=4)
  assert plan is not None
  assert _replay_ok(plan, asm, C, robots_list=robots, gk=None)


def test_milestones_lazily_retry_a_distinct_boundary_after_suffix_failure(monkeypatch):
  """A suffix failure rejects an exact checkpoint, not the whole part-order decision.

  This deliberately tiny symbolic harness isolates the outer milestone control policy.  The
  first settled endpoint for obj_1 makes obj_2 impossible, while the next endpoint admits it.
  K=1 therefore reproduces the historical claim-once failure and K=2 must recover without
  changing the dependency-valid part order.
  """
  import solver as S

  class DummyState:
    def __init__(self, label, goal=False):
      self.label = label
      self.goal = goal

    def __hash__(self):
      return hash((self.label, self.goal))

    def __eq__(self, other):
      return (isinstance(other, DummyState)
              and (self.label, self.goal) == (other.label, other.goal))

    def is_goal(self, _assembly):
      return self.goal

  initial = DummyState("initial")
  bad = DummyState("obj1-bad")
  good = DummyState("obj1-good")
  final = DummyState("final", goal=True)
  asm = A.make_assembly("cross", 3)
  calls = []

  monkeypatch.setattr(S, "get_initial_state", lambda *_args: initial)
  monkeypatch.setattr(S, "_search_context", lambda *_args, **_kwargs: (object(), {}))
  monkeypatch.setattr(S, "_capture_fill_failure", lambda *_args, **_kwargs: None)

  def fake_fill(start, part, *_args, excluded_end_states=None, **_kwargs):
    excluded = set(excluded_end_states or ())
    calls.append((start.label, part, frozenset(s.label for s in excluded)))
    if part == "obj_1":
      endpoint = good if bad in excluded else bad
      return [("pick", "a1", part, endpoint.label)], endpoint
    if part == "obj_2" and start == bad:
      return None, None
    if part == "obj_2" and start == good:
      return [("pick", "a1", part, "final")], final
    raise AssertionError((start.label, part))

  monkeypatch.setattr(S, "_add_one_part", fake_fill)
  common = dict(C=None, assembly=asm, robots=["a1"], verbose=False, robot_ik=None,
                milestone_oracle="structural", fill_window=1, max_repairs=10)
  assert S.solve_assembly_problem_milestones(**common, milestone_boundaries=1) is None

  calls.clear()
  plan = S.solve_assembly_problem_milestones(**common, milestone_boundaries=2)
  assert plan is not None
  assert ("initial", "obj_1", frozenset({"obj1-bad"})) in calls


def test_milestone_session_resumes_after_external_complete_plan_suspension(monkeypatch):
  """A consumer can request the next exact boundary without restarting symbolic/IK search."""
  import solver as S

  class DummyState:
    def __init__(self, label, goal=False):
      self.label = label
      self.goal = goal

    def __hash__(self):
      return hash((self.label, self.goal))

    def __eq__(self, other):
      return (isinstance(other, DummyState)
              and (self.label, self.goal) == (other.label, other.goal))

    def is_goal(self, _assembly):
      return self.goal

  initial = DummyState("initial")
  middle = DummyState("middle")
  final_a = DummyState("final-a", goal=True)
  final_b = DummyState("final-b", goal=True)
  asm = A.make_assembly("cross", 3)
  calls = []

  monkeypatch.setattr(S, "get_initial_state", lambda *_args: initial)
  monkeypatch.setattr(S, "_search_context", lambda *_args, **_kwargs: (object(), {}))
  monkeypatch.setattr(S, "_capture_fill_failure", lambda *_args, **_kwargs: None)

  def fake_fill(start, part, *_args, excluded_end_states=None, **_kwargs):
    excluded = set(excluded_end_states or ())
    calls.append((start.label, part, frozenset(s.label for s in excluded)))
    if part == "obj_1":
      return [("pick", "a1", part, "middle")], middle
    if part == "obj_2" and start == middle:
      endpoint = final_b if final_a in excluded else final_a
      return [("pick", "a1", part, endpoint.label)], endpoint
    raise AssertionError((start.label, part))

  monkeypatch.setattr(S, "_add_one_part", fake_fill)
  session = S.iter_assembly_problem_milestones(
      None, asm, robots=["a1"], verbose=False, robot_ik=None,
      milestone_oracle="structural", fill_window=1, max_repairs=10,
      milestone_boundaries=2)
  first = next(session)
  second = next(session)
  assert first != second
  assert first[-1][-1] == "final-a" and second[-1][-1] == "final-b"
  assert ("middle", "obj_2", frozenset({"final-a"})) in calls


def test_chain_milestone_oracle_plans_a_sequence():
  A.set_seed(0); A.set_rolls_per_face(1)
  asm = A.make_assembly("cross", 3)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  plan = A.solve_assembly_problem_milestones(
      C, asm, robots=robots, weight=5.0, verbose=False,
      milestone_oracle="chain")
  assert plan is not None
  assert _replay_ok(plan, asm, C, robots_list=robots, gk=None)


def test_fill_local_exact_plans_a_sequence_without_global_oracle():
  A.set_seed(0); A.set_rolls_per_face(1)
  asm = A.make_assembly("cross", 3)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  plan = A.solve_assembly_problem_milestones(
      C, asm, robots=robots, weight=100.0, verbose=False,
      milestone_oracle="structural", fill_local_h=True, fill_exact=True)
  assert plan is not None
  assert _replay_ok(plan, asm, C, robots_list=robots, gk=None)


def test_fill_touch_all_expands_manipulation_scope_but_not_milestone_assembly_scope():
  """The locality ablation may move blockers without silently growing the outer ideal."""
  import solver as S

  relevant = {"obj_0", "obj_1"}
  S.set_fill_touch_all(False)
  try:
    assert S._fill_action_in_scope(("pick", "a1", "obj_1", 0), relevant, "obj_1")
    assert not S._fill_action_in_scope(("pick", "a1", "obj_2", 0), relevant, "obj_1")
    assert S._fill_action_in_scope(("release", "a1", "obj_2", 0), relevant, "obj_1")

    S.set_fill_touch_all(True)
    assert S._fill_action_in_scope(("pick", "a1", "obj_2", 0), relevant, "obj_1")
    assert S._fill_action_in_scope(("place", "a1", "obj_2", 0), relevant, "obj_1")
    assert S._fill_action_in_scope(("handover", "a1", "obj_2", 0), relevant, "obj_1")
    assert not S._fill_action_in_scope(
        ("assemble", "a1", "obj_2", "obj_0"), relevant, "obj_1")
    assert S._fill_action_in_scope(
        ("assemble", "a1", "obj_1", "obj_0"), relevant, "obj_1")
  finally:
    S.set_fill_touch_all(False)


def test_fill_scope_widening_is_local_first_and_force_all_skips_the_fast_pass():
  import solver as S

  assert S._fill_scope_passes(False, False) == (False,)
  assert S._fill_scope_passes(False, True) == (False, True)
  assert S._fill_scope_passes(True, False) == (True,)
  assert S._fill_scope_passes(True, True) == (True,)


def test_direct_fill_skeleton_plans_a_simple_join():
  """The reviewer baseline uses ordinary actions and returns a replayable direct schema."""
  A.set_seed(0); A.set_rolls_per_face(1)
  asm = A.make_assembly("cross", 2)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  plan = A.solve_assembly_problem_milestones(
      C, asm, robots=robots, weight=1.0, verbose=False,
      milestone_oracle="structural", fill_skeleton_first=True)
  assert plan is not None
  assert _replay_ok(plan, asm, C, robots_list=robots, gk=None)
  assert sum(a[0] == "handover" for a in plan) <= 1
  assert sum(a[0] == "add_grasp" for a in plan) <= 1


def test_direct_fill_skeleton_failure_explicitly_uses_complete_fallback(monkeypatch, capsys):
  """Exhausting the restricted family must not become a new infeasibility declaration."""
  A.set_seed(0); A.set_rolls_per_face(1)
  asm = A.make_assembly("cross", 2)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  original = A._IncrementalLazySearch.search
  calls = {"n": 0}

  def exhaust_first(self, *args, **kwargs):
    calls["n"] += 1
    if calls["n"] == 1:
      return None
    return original(self, *args, **kwargs)

  monkeypatch.setattr(A._IncrementalLazySearch, "search", exhaust_first)
  plan = A.solve_assembly_problem_milestones(
      C, asm, robots=robots, weight=1.0, verbose=True,
      milestone_oracle="structural", fill_skeleton_first=True)
  assert plan is not None
  assert _replay_ok(plan, asm, C, robots_list=robots, gk=None)
  assert "complete local fallback" in capsys.readouterr().out


def test_fill_local_reverse_oracle_is_lazy_and_mixed_wrench_aware():
  """A local PDB starts from goal seeds and retains mixed-tool physics without an opt-in."""
  A.set_seed(0); A.set_rolls_per_face(1)
  asm = A.make_assembly("cross", 3)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=[True, False])
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  checker, grasp_table = A._search_context(C, asm, robots)
  kinds = {"a1": "vacuum", "a2": "two_finger"}
  classes = A.compute_grasp_signatures(
      asm, grasp_table, gripper_kind=kinds,
      gripper_path=checker.fact_gripper_path, clearance=0.02)
  start = A.get_initial_state(C, asm)
  pdb = A._make_fill_local_pdb(
      start, asm, robots, checker, grasp_table, "obj_1", classes,
      gripper_kind=kinds, component_root=asm.base, verbose=False)
  gids = sorted(classes["obj_1"])
  assert gids
  assert all(pdb._gcollapse("obj_1", g) == classes["obj_1"][g] for g in gids)
  assert set(pdb._svgroups["obj_1"]) == set(classes["obj_1"].values())
  assert pdb._lazy_build and pdb._mixed and pdb._gk == kinds
  assert not pdb._settled and not pdb._reverse_exhausted
  seeded = len(pdb.dist)
  queried = pdb.value(start)
  assert queried < float("inf")
  assert pdb._settled and pdb._reverse_pops > 0
  assert len(pdb.dist) >= seeded
  assert pdb._wfacts, "mixed-tool --wrench oracle never evaluated a per-kind wrench fact"
  blind = A._make_fill_local_pdb(
      start, asm, robots, checker, grasp_table, "obj_1", classes,
      gripper_kind=None, component_root=asm.base, verbose=False)
  assert blind.value(start) < float("inf")
  assert not blind._mixed and not blind._wfacts
  # Exhausting a paused oracle must reproduce eager Dijkstra exactly. This specifically
  # catches pausing after settling a query without first expanding its predecessors.
  eager = A._make_fill_local_pdb(
      start, asm, robots, checker, grasp_table, "obj_1", classes,
      gripper_kind=kinds, component_root=asm.base, verbose=False, lazy_build=False)
  pdb._advance_reverse()
  assert pdb._reverse_exhausted
  assert pdb.dist == eager.dist
  assert queried == eager.value(start)


def test_fill_local_exact_handles_a_detached_module():
  """The local projection follows the requested component root; it is not a chain hotfix."""
  A.set_seed(0); A.set_rolls_per_face(1)
  asm = A.make_assembly("cube-plates", 8)
  C = make_four_robot_env(num_robots=3, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  plan = A.solve_assembly_problem_milestones(
      C, asm, robots=robots, weight=100.0, verbose=False,
      milestone_oracle="structural", fill_local_h=True, fill_exact=True)
  assert plan is not None
  assert _replay_ok(plan, asm, C, robots_list=robots, gk=None)
  assert any(a[0] == "assemble" and a[2] == "obj_4" for a in plan)


def test_optimistic_chain_heuristic_plans_a_non_module_assembly():
  """The fixed-order chain projection must remain a usable heuristic, not only a small DB."""
  A.set_seed(0); A.set_rolls_per_face(1)
  asm = A.make_assembly("cross", 3)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  plan = A.solve_assembly_problem_optimistic(
      C, asm, robots=robots, weight=5.0, verbose=False, fallback=False,
      wrench=False, bracing=False, chain_h=True)
  assert plan is not None
  assert _replay_ok(plan, asm, C, robots_list=robots, gk=None)
