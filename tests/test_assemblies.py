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


def _replay_ok(plan, asm, C, robots_list, gk):
  """Independently re-derive the plan: every action must be an enumerated successor that
  passes the cheap feasibility layer, and the final state must be the goal."""
  checker, grasp_table = A._search_context(C, asm, robots_list)
  s = A.get_initial_state(C, asm)
  for act in plan:
    succ = [ns for a, ns in A.successors(s, asm, robots_list, grasp_table, gripper_kind=gk)
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
    ("cube-plates", 4, 2, 3.0, "asymmetric", 10.0, 12),
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
  # bottom chain 0->1->3->2 (complete, on table); top chain 4->5->7->6 (held subset).
  top_parent = {"obj_5": ("part", "obj_4"), "obj_7": ("part", "obj_5"), "obj_6": ("part", "obj_7")}
  parent = {"obj_0": ("table",), "obj_1": ("part", "obj_0"), "obj_3": ("part", "obj_1"),
            "obj_2": ("part", "obj_3"), "obj_4": ("gripper", "a1")}
  supports = {"obj_0": frozenset({("table",)}), "obj_1": frozenset({("mate", "obj_0")}),
              "obj_3": frozenset({("mate", "obj_1")}), "obj_2": frozenset({("mate", "obj_3")}),
              "obj_4": frozenset({("grasp", "a1", "obj_4", gid)})}
  for x in ("obj_5", "obj_7", "obj_6"):
    if x in held_top:
      parent[x] = top_parent[x]
      supports[x] = frozenset({("mate", top_parent[x][1])})
  poses = A.get_initial_state(C, asm).poses
  return A.State(parent=parent, place={"obj_0": ("slot", 0, 0)}, supports=supports,
                 poses=dict(poses))


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
  s_full = _two_module_state(asm, C, robots, {"obj_5", "obj_7", "obj_6"})
  joins = [(a, ns) for a, ns in A.assembles(s_full, asm, robots)
           if a[2] == "obj_4" and a[3] == "obj_0" and A.cheap_feasible(a, s_full, ns, asm, checker)]
  assert joins, "no cheap-feasible module join for the complete top plate"
  act, ns = joins[0]
  assert act[0] == "assemble", "the join must be a plain `assemble`, not a separate action"
  assert len(next(iter(ns.subassemblies()))) == 8, "join did not fuse into one 8-part body"
  assert ns.is_goal(asm), "joined state is not the goal"

  # (b) PARTIAL top module (missing the last chain cube obj_6) -> the join is FORBIDDEN.
  s_part = _two_module_state(asm, C, robots, {"obj_5", "obj_7"})
  partial = [a for a, _ns in A.assembles(s_part, asm, robots)
             if a[2] == "obj_4" and a[3] == "obj_0"]
  assert not partial, "a partial top plate must not be joinable (forcing broken)"
