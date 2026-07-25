"""Pre-skill funnel consistency: the backoff algebra, the all-or-nothing gate, the hard joint
pin, and the exporter's consumption of ready-made pre keyframes.

WHAT IS BEING PROTECTED. --pre-* splits a skill into a GOTO to a funnel mouth plus the skill.
The funnel keyframe used to be solved inside mp_export by the skill's own KOMO with the target
backed off and only a weight-1.0 pull toward the skill witness. Because the skill constraints
are RELATIVE, that solve is free to satisfy the backoff by moving the TARGET instead of the
actor -- measured on car-mini-new, the target assembly slid 16.8-48.8mm at every assemble while
the inserted part covered 6.1-35.8mm of the nominal 50mm. fuse_pre_skill_witnesses re-solves
those keyframes with the non-moving arms PINNED, gated all-or-nothing.

No MuJoCo, no BiRRT, and no search: the algebra and the gate are pure, the pin is checked on a
real (tiny) rai KOMO, and the exporter is driven with a stub IK that refuses to solve.
"""
import numpy as np
import pytest
import robotic as ry

import main
import mp_export as MP
import solver as S
from config import RunConfig
from env import make_four_robot_env
from problem import _quat_to_R


# --------------------------------------------------------------------------- flag gate
def test_fuse_pre_skill_defaults_off_and_the_flag_turns_it_on():
  assert RunConfig().fuse_pre_skill is False
  assert main.parse_args([]).fuse_pre_skill is False
  assert main.parse_args(["--fuse-pre-skill"]).fuse_pre_skill is True


# --------------------------------------------------------------------------- backoff algebra
def test_backoff_ee_retracts_along_the_tools_own_minus_z():
  """pick / add_grasp. The corrected EE +z points INTO the part and the target is expressed in
  the PART frame, so the funnel mouth is a pure translation there -- and it must follow the
  tool's own axis, not the part's. Checked against a 90deg-rotated tool, where the two differ."""
  s = np.sqrt(0.5)
  flat = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])            # tool +z == part +z
  assert S._backoff_ee(flat, 0.05) == pytest.approx([0.1, 0.2, 0.25, 1.0, 0.0, 0.0, 0.0])
  tipped = np.array([0.1, 0.2, 0.3, s, s, 0.0, 0.0])              # +90deg about x: tool z -> part y
  out = S._backoff_ee(tipped, 0.05)
  assert out[:3] == pytest.approx([0.1, 0.25, 0.3], abs=1e-12)
  assert out[3:] == pytest.approx(tipped[3:])                     # orientation untouched
  assert flat[2] == 0.3, "the caller's array must not be mutated"


def test_backoff_ee_matches_the_expression_it_replaced_bit_for_bit():
  """The four inline sites were folded into these helpers; --pre-* with the flag OFF must stay
  byte-identical, so reproduce the ORIGINAL expression and demand exact equality."""
  rng = np.random.default_rng(0)
  for _ in range(20):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    t = np.concatenate([rng.normal(size=3), q])
    for delta in (0.01, 0.05, 0.2):
      want = t.copy()
      want[:3] = want[:3] - delta * (_quat_to_R(want[3:]) @ np.array([0.0, 0.0, 1.0]))
      assert np.array_equal(S._backoff_ee(t, delta), want)


def test_backoff_place_lifts_in_world_z_only():
  t = np.array([0.4, -0.2, 0.03, 0.0, 1.0, 0.0, 0.0])
  out = S._backoff_place(t, 0.05)
  assert out[2] == pytest.approx(0.08)
  assert np.array_equal(out[[0, 1, 3, 4, 5, 6]], t[[0, 1, 3, 4, 5, 6]])
  assert t[2] == 0.03


def test_backoff_mate_moves_against_the_mate_axis_and_a_missing_axis_is_a_zero_length_funnel():
  """The seat direction is +mate_axis, so the funnel mouth is -mate_axis. A part with NO mate
  axis has always silently kept the mated pose (a zero-length funnel); that behaviour is
  preserved here on purpose -- fuse_pre_skill_witnesses is what reports it."""
  rel = np.array([0.1, -0.2, 0.3])
  assert S._backoff_mate(rel, [0.0, 1.0, 0.0], 0.05) == pytest.approx([0.1, -0.25, 0.3])
  assert S._backoff_mate(rel, None, 0.05) is rel


# --------------------------------------------------------------------------- the gate
_GOOD = {"dq": 0.05, "hold": 0.0, "stat_mm": 0.01, "stat_deg": 0.0,
         "dev": 0.03, "pos": 1e-5, "ang": 0.01, "pen": 0.0}
_TODAY = {"dq": 0.034, "hold": 0.104, "stat_mm": 43.2, "stat_deg": 1.7,
          "dev": 43.6, "pos": 7.5e-4, "ang": 0.02, "pen": 0.0}
_GATE = dict(pen_gate=1e-4, max_dq=0.35, hold_dq=0.02, hold_mm=2.0, pos_tol=1e-3, ang_tol=1.0)


def _gate(o, n):
  return S._pre_gate(o, n, **_GATE)


def test_gate_accepts_the_measured_repair():
  """The car-mini-new assemble signature: today's keyframe pays for the backoff by sliding the
  target 43.2mm (funnel deviation 43.6mm); the pinned re-solve does not."""
  assert _gate(_TODAY, _GOOD) is None


def test_gate_rejects_when_the_approach_deviation_did_not_improve():
  """Nothing to buy -> keep today's keyframe. This is what makes the benign splits (pick: the
  target is table-fixed, deviation already ~0.1mm) cost nothing."""
  n = dict(_GOOD, dev=_TODAY["dev"])
  assert "not improved" in _gate(_TODAY, n)
  assert "not improved" in _gate(_TODAY, dict(_GOOD, dev=_TODAY["dev"] + 1.0))


def test_gate_rejects_a_branch_change_however_good_the_funnel_is():
  """MEASURED on car-mini-new step 05: pinning makes the funnel perfect (43.6 -> 0.7mm) but
  costs a 1.484 rad actor reconfiguration. A 50mm approach that reconfigures the arm is not an
  approach, so the gate refuses it and today's keyframe is kept."""
  why = _gate(_TODAY, dict(_GOOD, dev=0.7, dq=1.484))
  assert why is not None and "actor joint delta" in why


def test_gate_rejects_an_infeasible_resolve():
  assert _gate(_TODAY, None) == "pinned solve infeasible"


@pytest.mark.parametrize("key,bad,word", [
    ("pos", 2e-3, "position residual"),
    ("ang", 2.0, "orientation residual"),
    ("pen", 1e-3, "penetration"),
    ("hold", 0.5, "non-moving arm"),
    ("stat_mm", 9.0, "stationary body"),
])
def test_every_hard_metric_has_an_absolute_gate(key, bad, word):
  """`o` is already clean here, so each limit reduces to the absolute tolerance."""
  clean = dict(_GOOD)
  why = _gate(clean, dict(_GOOD, dev=0.001, **{key: bad}))
  assert why is not None and word in why


@pytest.mark.parametrize("key,bad", [("pos", 2e-3), ("ang", 2.0), ("pen", 1e-3),
                                     ("hold", 0.5), ("stat_mm", 9.0), ("dq", 1.0)])
def test_a_metric_may_exceed_its_tolerance_if_todays_keyframe_was_worse(key, bad):
  """The fuse convention: "within tolerance OR no worse than what we replace". Without it the
  pass could never touch a split whose skill witness itself sits at the tolerance."""
  o = dict(_TODAY, **{key: bad})
  assert _gate(o, dict(_GOOD, **{key: bad})) is None
  worse = bad * 1.5 if key != "pen" else bad + 1e-4
  assert _gate(o, dict(_GOOD, **{key: worse})) is not None


def test_gate_falls_back_to_absolute_tolerances_when_todays_solve_failed():
  """pre_skill_witness returning None is a LIVE path (measured: a place that would not bind).
  There is nothing to be worse than, so only the absolute gates apply -- and the
  strictly-improved rule cannot be evaluated, so it must not veto."""
  assert _gate(None, _GOOD) is None
  assert _gate(None, dict(_GOOD, stat_mm=9.0)) is not None
  assert _gate(None, None) == "pinned solve infeasible"


# --------------------------------------------------------------------------- the pin
@pytest.fixture(scope="module")
def arms():
  C = make_four_robot_env(num_robots=3, floating_ee=False)
  import assembly as A
  return C, A.get_robots(C)


def test_pin_is_a_hard_eq_that_actually_freezes_the_named_joints(arms):
  """The load-bearing claim: an eq at the skill weight holds a joint where a weight-1.0 sos
  pull (the pre-insertion pull, which is what shipped) cannot. Solved for real."""
  C, robots = arms
  names = C.getJointNames()
  target = {n: v for n, v in zip([n for n in names if n.startswith(robots[1] + "_")],
                                 [0.11, -0.22, 0.33, -0.44, 0.55, -0.66])}
  rik = S.RobotIK(C, robots)
  rik._pin_joints = target
  komo = ry.KOMO(C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=False)
  komo.addControlObjective([], 0, 1e-1)
  komo.addControlObjective([], 1, 1e-1)
  rik._add_pin(komo, C)
  komo.initRandom()
  assert ry.NLP_Solver(komo.nlp(), verbose=0).solve().dict()["feasible"]
  q = komo.getPath()[-1]
  got = np.array([q[names.index(n)] for n in target])
  assert got == pytest.approx(list(target.values()), abs=1e-3)


def test_pin_is_a_no_op_when_unset_and_when_no_named_joint_is_in_the_config(arms):
  """None => the search and a flag-off export are byte-identical. The name filter is what makes
  the pin vanish in a _restrict_to_arm scene, where the other arms are frozen already."""
  C, robots = arms
  rik = S.RobotIK(C, robots)
  calls = []

  class _Komo:
    def addObjective(self, *a, **k):
      calls.append((a, k))

  rik._pin_joints = None
  rik._add_pin(_Komo(), C)
  rik._pin_joints = {}
  rik._add_pin(_Komo(), C)
  rik._pin_joints = {"not_a_joint_in_this_config": 0.0}
  rik._add_pin(_Komo(), C)
  assert calls == []


def test_pre_skill_witness_always_clears_the_pin(arms):
  """A leaked _pin_joints would silently freeze arms in the NEXT solve -- including the search's
  if the object were reused. Cleared in a finally, even when the IK raises."""
  C, robots = arms
  rik = S.RobotIK(C, robots)
  rik.pre_pick_delta = 0.05

  def boom(*a, **k):
    raise RuntimeError("ik exploded")

  rik._ik_pick = boom
  with pytest.raises(RuntimeError):
    rik.pre_skill_witness(("pick", robots[0], "obj_0", 0), None, None, None,
                          skill_witness=np.zeros(len(C.getJointNames())), pin={"a": 0.0})
  assert rik._pin_joints is None and rik._opt_pull is None


def test_a_pinned_arms_grasp_maintenance_is_dropped(arms):
  """A frozen brace holding a frozen object cannot drive its own eq residual to zero, so keeping
  the objective makes the WHOLE solve infeasible -- measured, 2 of 4 assembles died that way and
  the other 2 were thrown 1.5-1.8 rad off branch. Arms that co-move with the skill are never
  pinned, so dropping this is safe by construction."""
  C, robots = arms
  rik = S.RobotIK(C, robots)
  grasps = [(robots[0], "obj_0", np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])),
            (robots[2], "obj_1", np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]))]
  rik.maintenance_grasps = lambda state, assembly: grasps
  seen = []

  class _Komo:
    def addObjective(self, slc, feat, frames, ot, w, tgt):
      seen.append(frames[0])

  rik._pin_joints = None
  rik._add_grasp_maintenance(_Komo(), None, None)
  assert seen == [rik.gripper_frame[robots[0]]] * 2 + [rik.gripper_frame[robots[2]]] * 2

  seen.clear()
  rik._pin_joints = {robots[0] + "_ur_elbow_joint": 0.0}
  rik._add_grasp_maintenance(_Komo(), None, None)
  assert seen == [rik.gripper_frame[robots[2]]] * 2       # the pinned arm's pair is gone


def test_the_relaxed_reach_precheck_is_skipped_while_a_pin_is_set():
  """Its reject threshold was measured on UN-pinned residuals and it decides on the LAST random
  restart -- which cannot satisfy a hard joint pin. Measured: it threw away pinned solves the
  full solve then found feasible (car-mini-new add_grasp 04 and 20)."""
  import inspect
  for fn in (S.RobotIK._ik_pick, S.RobotIK._ik_add_grasp, S.RobotIK._ik_place,
             S.RobotIK._ik_assemble):
    src = inspect.getsource(fn)
    assert src.count("self._pin_joints is None") >= src.count("_reach_precheck_reject"), \
        f"{fn.__name__}: a reach precheck is left un-guarded against the pin"
    assert src.index("self._pin_joints is None") < src.index("_reach_precheck_reject"), \
        f"{fn.__name__}: the pin guard must precede the precheck call"


# --------------------------------------------------------------------------- the exporter
class _StubIK:
  """The surface build_tasks touches for a plan of picks only."""

  def __init__(self, robots, delta):
    self.robots = list(robots)
    self.pre_pick_delta = delta
    self.pre_place_delta = self.pre_add_grasp_delta = self.pre_insertion_delta = 0.0
    self.gripper_frame = {r: r + "_tool" for r in self.robots}
    self.solved = 0

  def pre_skill_witness(self, *a, **k):
    self.solved += 1
    return np.full(6 * len(self.robots), 0.5)


class _State:
  def subassemblies(self):
    return [{"obj_0"}, {"obj_1"}]


def _tasks(pre_witnesses, delta=0.05, ik=None):
  MP._ensure_mrbench()
  robots = ["a1", "a2"]
  plan = [("pick", "a1", "obj_0", 0), ("pick", "a2", "obj_1", 0)]
  wit = [np.full(12, 0.1), np.full(12, 0.2)]
  states = [_State(), _State(), _State()]
  idx = {"a1": list(range(6)), "a2": list(range(6, 12))}
  rik = ik if ik is not None else _StubIK(robots, delta)
  return MP.build_tasks(None, plan, wit, states, robots, idx, rik, np.zeros(12),
                        pre_witnesses=pre_witnesses), rik


def test_build_tasks_consumes_supplied_keyframes_and_solves_nothing():
  pre = {0: np.full(12, 0.7), 1: np.full(12, 0.8)}
  (tasks, seq, step_names), rik = _tasks(pre)
  assert rik.solved == 0, "the exporter must not re-solve a supplied pre keyframe"
  assert seq == ["00_pre_pick_a1_obj_0_0", "00_pick_a1_obj_0_0",
                 "01_pre_pick_a2_obj_1_0", "01_pick_a2_obj_1_0", "terminal"]
  assert tasks[0].goal.goal[:6] == pytest.approx(np.full(6, 0.7))   # T_pre, from `pre`
  assert tasks[1].goal.goal[:6] == pytest.approx(np.full(6, 0.1))   # the skill witness


def test_build_tasks_without_the_argument_keeps_solving_them_itself():
  """Flag OFF is the historical path, untouched."""
  (_t, seq, _s), rik = _tasks(None)
  assert rik.solved == 2 and len(seq) == 5


def test_a_rejected_or_missing_keyframe_never_changes_the_task_structure():
  """THE fail-safe. sim.py --v2 rebuilds the task list from the saved config alone and executes
  a SAVED path against it, so a per-action structural decision the exporter made would shift
  every downstream task id (the documented pre_insertion bug). A None keyframe therefore still
  emits BOTH tasks -- it only makes the goto goal equal the skill goal."""
  (full, seq_full, names_full), _ = _tasks({0: np.full(12, 0.7), 1: np.full(12, 0.8)})
  (none, seq_none, names_none), _ = _tasks({0: None, 1: None})
  assert len(full) == len(none) and seq_full == seq_none and names_full == names_none
  assert none[0].goal.goal[:6] == pytest.approx(none[1].goal.goal[:6])   # zero-length funnel


def test_no_split_means_no_pre_tasks_at_all():
  (tasks, seq, _s), _ = _tasks({}, delta=0.0)
  assert seq == ["00_pick_a1_obj_0_0", "01_pick_a2_obj_1_0", "terminal"]


def test_the_pass_declines_cleanly_when_no_pre_split_is_active(arms):
  """--fuse-pre-skill without any --pre-* is a user error, not a crash: it must say so and
  return an empty mapping (which build_tasks then treats as "solve them yourself")."""
  C, robots = arms
  rik = S.RobotIK(C, robots)
  pre, rep = S.fuse_pre_skill_witnesses(C, None, [("pick", robots[0], "obj_0", 0)], [None],
                                        robots, rik, verbose=False)
  assert pre == {} and rep == {"splits": 0, "consistent": 0, "rows": []}


def test_empty_pin_is_not_reported_as_a_repair():
  """A pin that lands on NO joint means the scene had already frozen those arms, so the solve
  is just a second draw of the ordinary un-pinned one. Accepting it reports solver noise as a
  repair -- measured: every `pick` takes this path (scene restricted to 6 joints, pin asks for
  12), and 4 of 6 'accepted' splits were exactly that."""
  import inspect
  import solver as S
  pin_src = inspect.getsource(S.RobotIK._add_pin)
  assert "pin_applied" in pin_src, "_add_pin no longer records whether it pinned anything"
  pass_src = inspect.getsource(S.fuse_pre_skill_witnesses)
  assert "pin_applied" in pass_src, "the pass no longer checks that the pin bit on something"
  # the check must be able to VETO, i.e. it feeds `why` (the rejection reason)
  i = pass_src.index("pin_applied")
  assert "why" in pass_src[i:i + 400], "an empty pin must produce a rejection reason"
