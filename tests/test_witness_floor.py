"""Regression tests for the clearance-floor witness ladder.

Both bugs these guard were shipped and only caught by hand: the emitted witnesses were
RESTRICTED 6-vectors (every consumer indexes by full joint name -> KeyError), and they filled
non-chosen arms from the DP's carried joint map, so a consumer that poses those arms
(visualization._active_robots covers a wider set than witness_floor._involved) put robots
visibly through each other. The suite ran green through both.
"""
import numpy as np
import pytest
import robotic as ry

import assembly as A
import replay as R
from plan_io import read_plan
from solver import RobotIK, _replay_states, _search_context, involved_robots
from witness_floor import optimize_witnesses_floor

SCENE = "car-mini-new.json"


def _setup():
  cfg, plan = read_plan(SCENE)
  C, asm = R.build_scene(cfg)
  robots = A.get_robots(C)
  gk = A.gripper_kind(C) if cfg.get("wrench") else None
  rik = RobotIK(C, robots)
  rik.ik_seeding = True
  rik.strip_inactive_collision = False
  return cfg, plan, C, asm, robots, gk, rik


def _cross_arm_pairs(C, robots):
  cs = set(C.getCollidablePairs())
  cf = {r: [f for f in cs if f.startswith(r + "_")] for r in robots}
  return [(a, b) for i, r in enumerate(robots) for rr in robots[i + 1:]
          for a in cf[r] for b in cf[rr]]


@pytest.fixture(scope="module")
def floor_result():
  cfg, plan, C, asm, robots, gk, rik = _setup()
  _ck, gt = _search_context(C, asm, robots)
  wit, rep = optimize_witnesses_floor(C, asm, plan, robots, rik, gripper_kind=gk,
                                      bracing=bool(cfg.get("bracing")), grasp_table=gt,
                                      verbose=False)
  return wit, rep, C, robots, rik


def test_emitted_witnesses_are_full_joint_vectors(floor_result):
  # the harvested branches are restricted 6-vectors straight out of komo.getPath(); every
  # consumer indexes them by FULL joint name, so an un-expanded one is a KeyError at use time.
  wit, _rep, C, _robots, rik = floor_result
  n = len(rik.C.getJointNames())
  sizes = {np.asarray(w, dtype=float).ravel().size for w in wit if w is not None}
  assert sizes, "the ladder emitted no witnesses at all"
  assert sizes == {n}, f"expected only full {n}-vectors, got sizes {sorted(sizes)}"


def test_no_gross_arm_vs_arm_penetration(floor_result):
  # LOOSE bound on purpose: sub-millimetre overlap is the IK constraint tolerance and is
  # present in the search's own witnesses too. The bug this guards produced arms visibly
  # through each other, which is orders of magnitude past this threshold.
  wit, _rep, C, robots, rik = floor_result
  fkC = ry.Config()
  fkC.addConfigurationCopy(rik.C)
  pairs = _cross_arm_pairs(fkC, robots)
  worst, where = 1e9, None
  for i, w in enumerate(wit):
    if w is None:
      continue
    fkC.setJointState(np.asarray(w, dtype=float).ravel())
    g = min((-float(fkC.eval(ry.FS.distance, [a, b])[0][0]), a, b) for a, b in pairs)
    if g[0] < worst:
      worst, where = g[0], (i, g[1], g[2])
  assert worst > -0.02, f"gross arm-vs-arm penetration {worst * 1000:.1f}mm at {where}"


def test_report_is_populated(floor_result):
  _wit, rep, _C, _robots, _rik = floor_result
  assert rep.get("alpha") is not None
  assert rep.get("crossings", -1) >= 0


# ---------------------------------------------------------------------------------------
# THE shared "involved robots" definition. These two used to be separate copies keyed on
# DIFFERENT states -- the replay/exporter on the pre state, the witness ladder on the post one
# -- so for a handover (giver holds only before, receiver only after) they could name different
# arms for the very step whose IK solves both together. They now both delegate to
# solver.involved_robots; these tests are what stops them drifting apart again.

def test_active_robots_and_involved_agree_on_every_witnessed_action():
  from visualization import _active_robots
  from witness_floor import _involved
  cfg, plan, C, asm, robots, gk, rik = _setup()
  _ck, gt = _search_context(C, asm, robots)
  states = _replay_states(C, asm, plan, robots, gk, True, True,
                          bool(cfg.get("bracing")), gt, "test")
  seen = 0
  for i, a in enumerate(plan):
    act = tuple(a)
    ok, w = rik.bind(act, states[i], states[i + 1], asm)
    if not ok or w is None:
      continue                                   # no witness => no DOFs to pose or choose
    seen += 1
    assert set(_active_robots(act, states[i], rik)) == \
        set(_involved(act, states[i], states[i + 1], robots, rik)), \
        f"step {i} {act}: the poser and the chooser disagree"
  assert seen > 0, "the plan produced no witnessed actions to check"


def test_release_involves_nobody():
  """A release has no witness (bind returns None), so there is nothing to pose or choose. This
  was the ONLY disagreement between the two old copies -- 28 releases across four plans."""
  from witness_floor import _involved
  from visualization import _active_robots
  cfg, plan, C, asm, robots, gk, rik = _setup()
  _ck, gt = _search_context(C, asm, robots)
  states = _replay_states(C, asm, plan, robots, gk, True, True,
                          bool(cfg.get("bracing")), gt, "test")
  rel = [i for i, a in enumerate(plan) if a[0] == "release"]
  assert rel, "car-mini has releases; the fixture scene changed"
  for i in rel:
    act = tuple(plan[i])
    assert _involved(act, states[i], states[i + 1], robots, rik) == []
    assert _active_robots(act, states[i], rik) == []


def test_emitted_witness_preserves_the_grasp_relation():
  """The emitted witness must hold each object the SAME WAY bind does.

  This is the invariant behind "the picked subassembly is rotated 180deg and in the air".
  Downstream never infers the object from the emitted gripper pose -- the replay's
  _align_group_to_holder and mp_export's attach frames rebuild it from the CANONICAL grasp
  transform for the action's grasp id. So a witness whose gripper-to-object relation differs
  from bind's is drawn and attached wrong even when it is a perfectly good grasp.

  NOT a test on the raw tool roll: a pi change in wrist_3 is legitimate when it comes with the
  compensating wrist flip (q3+pi, -q4, q5+pi), which leaves the tool pose identical -- measured,
  such a candidate moves the grasp relation by only 0.4mm/deg. What was NOT legitimate was
  rewriting wrist_3 ALONE (the old canonical_roll write-back), which moved the tool and turned
  any held payload with it: exactly +1.00*pi at four steps on cross-12, each matching a 180deg
  object rotation, plus a 110mm shift of the gripper on the part.
  """
  cfg, plan, C, asm, robots, gk, rik = _setup()
  _ck, gt = _search_context(C, asm, robots)
  states = _replay_states(C, asm, plan, robots, gk, True, True,
                          bool(cfg.get("bracing")), gt, "test")
  # SAME instance for the ladder and the reference binds: bind() is stochastic and caches per
  # instance, so a fresh one would measure solver noise instead of the ladder's own base.
  wit, _rep = optimize_witnesses_floor(C, asm, plan, robots, rik, gripper_kind=gk,
                                       bracing=bool(cfg.get("bracing")), grasp_table=gt,
                                       verbose=False)

  def grasp_rel(Ci, q, r, part):
    Ci.setJointState(np.asarray(q, dtype=float).ravel())
    g, o = Ci.getFrame(rik.gripper_frame[r]), Ci.getFrame(part)
    Rg = np.asarray(g.getRotationMatrix(), dtype=float)
    dp = Rg.T @ (np.asarray(o.getPosition(), dtype=float)
                 - np.asarray(g.getPosition(), dtype=float))
    return dp, Rg.T @ np.asarray(o.getRotationMatrix(), dtype=float)

  worst, where = 0.0, None
  for i, a in enumerate(plan):
    act = tuple(a)
    ok, w0 = rik.bind(act, states[i], states[i + 1], asm)
    if not ok or w0 is None or wit[i] is None:
      continue
    held = {}
    for st in (states[i], states[i + 1]):
      for r, (pt, _gid) in rik.holders(st).items():
        held.setdefault(r, pt)
    Ci = rik.scene_for(states[i], asm)
    for r in involved_robots(act, states[i], states[i + 1], rik):
      part = held.get(r)
      if part is None:
        continue
      pa, Ra = grasp_rel(Ci, w0, r, part)
      pb, Rb = grasp_rel(Ci, wit[i], r, part)
      c = (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0
      d = max(float(np.linalg.norm(pa - pb)) * 1000.0,
              float(np.degrees(np.arccos(min(1.0, max(-1.0, c))))))
      if d > worst:
        worst, where = d, (i, act[:3], r, part)
  assert worst < 10.0, (
      f"the emitted witness holds the object differently from bind's by {worst:.1f} (mm or deg) "
      f"at {where} -- downstream rebuilds the object from the canonical grasp transform, so it "
      f"will be drawn and attached wrong")
