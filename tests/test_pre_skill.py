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
import assembly_planner.mp_export as MP
import assembly_planner.solver as S
from assembly_planner.config import RunConfig
from assembly_planner.env import make_four_robot_env


# --------------------------------------------------------------------------- flag gate
def test_fuse_pre_skill_is_tri_state_and_auto_enables_for_insertion_splits():
  # None = AUTO. run() resolves it to ON when a --pre-insertion/--pre-add-grasp split exists,
  # because that is the only place the pin can bite (pick/solo place restrict the scene to the
  # acting arm already, so pin_applied == 0 and the pass rejects). Explicit flags still win.
  assert RunConfig().fuse_pre_skill is None
  assert main.parse_args([]).fuse_pre_skill is None
  assert main.parse_args(["--fuse-pre-skill"]).fuse_pre_skill is True
  assert main.parse_args(["--no-fuse-pre-skill"]).fuse_pre_skill is False


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


def test_assemble_funnel_gate_checks_the_edge_not_only_its_endpoints():
  """Regression: a clear pre-insertion pose and clear seat do not imply a clear skill."""
  class Config:
    def getJointNames(self):
      return ["a1_joint"]

  class State:
    def subassemblies(self):
      return [{"obj_0"}, {"obj_1"}]

  rik = object.__new__(S.RobotIK)
  rik.skill_funnel_collision_res = 0.1
  rik.funnel_edge_rejects = 0
  rik.last_funnel_edge_reject = None
  rik._cur_action = None
  rik._mate_exempt = None
  rik._funnel_solve = False
  rik.scene_for = lambda _state, _assembly: Config()
  samples = []

  def pen_ok(_C, q):
    samples.append(float(q[0]))
    return not 0.45 <= q[0] <= 0.55

  rik._pen_ok = pen_ok
  action = ("assemble", "a1", "obj_1", "obj_0")
  assert not rik._funnel_edge_ok(action, State(), object(), np.array([0.0]), np.array([1.0]))
  assert samples[0] == 0.0 and samples[-1] == pytest.approx(0.5)
  assert rik.funnel_edge_rejects == 1
  assert rik.last_funnel_edge_reject[:2] == (action, "collision")


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
  import assembly_planner.assembly as A
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


def test_pre_pick_never_returns_a_different_actor_ik_branch():
  """Single-arm pick scenes have no idle joint left for fuse-pre-skill to pin.  Branch
  continuity therefore belongs in the common pre-skill generator, before export fallback."""
  rik = object.__new__(S.RobotIK)
  rik.pre_pick_delta = 0.05
  rik.pre_place_delta = rik.pre_add_grasp_delta = rik.pre_insertion_delta = 0.0
  rik.pre_insertion_pull = 1.0
  rik.pre_skill_max_dq = 0.35
  rik.C = type("C", (), {"getJointNames": lambda _self: ["a1_j0", "a2_j0"]})()
  far = np.array([1.0, 0.0])
  calls = []

  def solve(*_args, **kwargs):
    calls.append(kwargs["backoff"])
    return True, far.copy()

  rik._ik_pick = solve
  warm = np.zeros(2)
  got = rik.pre_skill_witness(("pick", "a1", "obj_0", 0), None, None, None,
                              skill_witness=warm)
  assert got is None                                    # exporter will use the skill endpoint
  assert len(calls) == 6                                # every shorter rung was considered
  assert rik._pin_joints is None and rik._opt_pull is None

  rik._ik_pick = lambda *_a, **_k: (True, np.array([0.1, 0.0]))
  got = rik.pre_skill_witness(("pick", "a1", "obj_0", 0), None, None, None,
                              skill_witness=warm)
  np.testing.assert_allclose(got, [0.1, 0.0])


def test_pre_assemble_requires_the_full_configured_backoff():
  """An insertion mouth is a hard controller boundary, not a best-effort approach hint.

  Regression from Real Bench: the 50mm solve failed, the shared fallback ladder accepted its
  5mm rung, and BiRRT could reach that almost-seated keyframe only with centimetres of allowed
  penetration. A failed full mouth must reject this witness so search can try another one.
  """
  rik = object.__new__(S.RobotIK)
  rik.pre_pick_delta = rik.pre_place_delta = rik.pre_add_grasp_delta = 0.0
  rik.pre_insertion_delta = 0.05
  rik.pre_insertion_pull = 1.0
  rik.pre_skill_max_dq = 0.35
  rik.C = type("C", (), {"getJointNames": lambda _self: ["a1_j0", "a2_j0"]})()
  assembly = type("Assembly", (), {
      "screw_skill": lambda _self, _part: None,
      "mate_axis": lambda _self, _part: np.array([1.0, 0.0, 0.0]),
  })()
  calls = []

  def fail_full(*_args, **kwargs):
    calls.append(kwargs["backoff"])
    # This would have succeeded at the next, shorter ladder rung before the fix.
    return (kwargs["backoff"] < 0.05, np.array([0.1, 0.0]))

  rik._ik_assemble = fail_full
  rik._funnel_edge_ok = lambda *_a, **_k: True
  got = rik.pre_skill_witness(
      ("assemble", "a1", "obj_1", "obj_0"), None, None, assembly,
      skill_witness=np.zeros(2))
  assert got is None
  assert calls == [0.05]
  assert rik._pin_joints is None and rik._opt_pull is None

  calls.clear()
  rik._ik_assemble = lambda *_a, **k: (
      calls.append(k["backoff"]) or True, np.array([0.1, 0.0]))
  got = rik.pre_skill_witness(
      ("assemble", "a1", "obj_1", "obj_0"), None, None, assembly,
      skill_witness=np.zeros(2))
  np.testing.assert_allclose(got, [0.1, 0.0])
  assert calls == [0.05]


def test_pre_add_grasp_funnel_dispatches_handovers_to_the_same_geometry_solver():
  """A handover differs from add_grasp only after acquisition, so it must not bypass the
  incoming-hand standoff controlled by --pre-add-grasp."""
  rik = object.__new__(S.RobotIK)
  rik.pre_pick_delta = rik.pre_place_delta = rik.pre_insertion_delta = 0.0
  rik.pre_add_grasp_delta = 0.05
  rik.pre_insertion_pull = 1.0
  rik.pre_skill_max_dq = 0.35
  rik.C = type("C", (), {"getJointNames": lambda _self: ["a1_j0", "a2_j0"]})()
  calls = []

  def solve(action, *_args, **kwargs):
    calls.append((tuple(action), kwargs["backoff"]))
    return True, np.array([0.0, 0.1])

  rik._ik_add_grasp = solve
  warm = np.zeros(2)
  got = rik.pre_skill_witness(
      ("handover", "a2", "obj_0", 3), None, None, None, skill_witness=warm)
  np.testing.assert_allclose(got, [0.0, 0.1])
  assert calls == [(('handover', 'a2', 'obj_0', 3), 0.05)]
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


# --------------------------------------------------------------------------- the exporter
class _StubIK:
  """The surface build_tasks touches for a plan of picks only."""

  def __init__(self, robots, delta):
    self.robots = list(robots)
    self.pre_pick_delta = delta
    self.pre_place_delta = self.pre_add_grasp_delta = self.pre_insertion_delta = 0.0
    self.gripper_frame = {r: r + "_tool" for r in self.robots}
    self.solved = 0
    self.post_solved = 0

  def pre_skill_witness(self, *a, **k):
    self.solved += 1
    return np.full(6 * len(self.robots), 0.5)

  def post_pick_witness(self, action, before, after, assembly, q):
    self.post_solved += 1
    out = np.asarray(q, dtype=float).copy()
    robot_index = self.robots.index(action[1])
    out[6 * robot_index:6 * (robot_index + 1)] += 0.3
    return out


class _State:
  def subassemblies(self):
    return [{"obj_0"}, {"obj_1"}]


def _tasks(pre_witnesses, delta=0.05, ik=None, resolve_missing_pre=True,
           witness_sets=None, pre_pin_robots=None):
  MP._ensure_mrbench()
  robots = ["a1", "a2"]
  plan = [("pick", "a1", "obj_0", 0), ("pick", "a2", "obj_1", 0)]
  wit = [np.full(12, 0.1), np.full(12, 0.2)]
  states = [_State(), _State(), _State()]
  idx = {"a1": list(range(6)), "a2": list(range(6, 12))}
  rik = ik if ik is not None else _StubIK(robots, delta)
  return MP.build_tasks(None, plan, wit, states, robots, idx, rik, np.zeros(12),
                        pre_witnesses=pre_witnesses,
                        witness_sets=witness_sets,
                        resolve_missing_pre=resolve_missing_pre,
                        pre_pin_robots=pre_pin_robots), rik


def test_build_tasks_consumes_supplied_keyframes_and_solves_nothing():
  pre = {0: np.full(12, 0.7), 1: np.full(12, 0.8)}
  (tasks, seq, step_names), rik = _tasks(pre)
  assert rik.solved == 0, "the exporter must not re-solve a supplied pre keyframe"
  assert seq == ["00_pre_pick_a1_obj_0_0", "00_pick_a1_obj_0_0",
                 "01_pre_pick_a2_obj_1_0", "01_pick_a2_obj_1_0", "terminal"]
  assert tasks[0].goal.goal[:6] == pytest.approx(np.full(6, 0.7))   # T_pre, from `pre`
  assert tasks[1].goal.goal[:6] == pytest.approx(np.full(6, 0.1))   # the skill witness


def test_pick_gets_a_solved_held_object_departure_after_attachment():
  """The pick funnel has two geometric scenes around one attachment event.

  PRE uses the empty-hand standoff while the object rests on the table. POST is solved in the
  attached scene and need not return to that pre-grasp configuration.
  """
  pre = {0: np.full(12, 0.7), 1: np.full(12, 0.8)}
  rik = _StubIK(["a1", "a2"], 0.05)
  rik.post_pick_delta = 0.05
  (tasks, seq, step_names), rik = _tasks(pre, ik=rik)
  assert rik.solved == 0
  assert seq[:3] == ["00_pre_pick_a1_obj_0_0", "00_pick_a1_obj_0_0",
                     "00_post_pick_a1_obj_0_0"]
  assert step_names[0] == seq[:3]
  assert tasks[0].goal.goal == pytest.approx(np.full(6, 0.7))
  assert tasks[1].goal.goal == pytest.approx(np.full(6, 0.1))
  assert tasks[2].goal.goal == pytest.approx(np.full(6, 0.4))
  assert tasks[2].robots == ["a1"]
  assert rik.post_solved == 2       # one independently solved departure per pick action


def test_build_tasks_keeps_search_exit_out_of_the_physical_task_sequence():
  MP._ensure_mrbench()
  robots = ["a1", "a2"]
  idx = {"a1": list(range(6)), "a2": list(range(6, 12))}
  home = np.zeros(12)
  exit_q = np.concatenate([np.full(6, -0.3), np.full(6, 0.3)])
  rik = _StubIK(robots, 0.0)
  tasks, seq, names = MP.build_tasks(
      None, [], [], [_State()], robots, idx, rik, home, exit_state=exit_q)
  assert names == []
  assert seq == ["terminal"]
  assert len(tasks) == 1
  assert tasks[0].robots == robots
  assert tasks[0].goal.goal == pytest.approx(home)


def test_accepted_pre_pin_does_not_turn_an_idle_arm_into_a_task_participant():
  qpre = np.concatenate([np.full(6, 0.7), np.full(6, 0.9)])
  (tasks, _seq, _names), rik = _tasks(
      {0: qpre, 1: np.full(12, 0.8)}, pre_pin_robots={0: ["a2"]})
  assert rik.solved == 0
  # The pin proved one safe full IK configuration, but a2 is not physically part of a1's pick.
  # make_problem supplies qpre separately as a transition hint; build_tasks must not turn an
  # internal solver pin into a multi-robot rendezvous constraint.
  assert tasks[0].robots == ["a1"]
  assert tasks[0].goal.goal == pytest.approx(qpre[:6])
  assert tasks[1].robots == ["a1"]
  assert tasks[1].goal.goal == pytest.approx(np.full(6, 0.1))
  assert tasks[2].robots == ["a2"]                         # unrelated split unchanged


def test_transition_hint_never_turns_idle_exit_pins_into_path_coordinates():
  """The full IK certificate uses EXIT for idle arms, but the physical hint must not."""
  class _Problem:
    robot_idx = {"a1": list(range(6)), "a2": list(range(6, 12))}

  class _Mode:
    entry_configuration = np.concatenate([np.full(6, 1.0), np.full(6, 2.0)])

  class _Task:
    robots = ["a1"]

  # a2=-9 represents the collision-safe EXIT background in the IK certificate.
  certificate = np.concatenate([np.full(6, 3.0), np.full(6, -9.0)])
  merged = MP._active_slice_transition_hint(_Problem(), _Mode(), _Task(), certificate)
  assert merged[:6] == pytest.approx(np.full(6, 3.0))
  assert merged[6:] == pytest.approx(np.full(6, 2.0))



def test_loaded_final_export_refuses_a_missing_pre_keyframe_instead_of_solving():
  ik = _StubIK(["a1", "a2"], 0.05)
  with pytest.raises(ValueError, match="refusing to re-solve IK during replay"):
    _tasks({0: np.full(12, 0.7)}, ik=ik, resolve_missing_pre=False)
  assert ik.solved == 0


def test_a_rejected_or_missing_keyframe_never_changes_the_task_structure():
  """THE fail-safe. sim.py rebuilds the task list from the saved config alone and executes
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


def test_pre_skill_preserves_witness_set_and_conditions_skill_on_entry_member():
  """A pair of plain GoalSets can cross A->B. The skill goal must read which pre member
  actually entered the mode and return only the corresponding skill member."""
  class _PairIK(_StubIK):
    class _C:
      @staticmethod
      def getJointNames():
        return [f"a{r}_j{j}" for r in (1, 2) for j in range(6)]

    C = _C()

    def pre_skill_witness(self, *args, **kwargs):
      self.solved += 1
      return np.asarray(args[4], dtype=float) + 0.4

  ik = _PairIK(["a1", "a2"], 0.05)
  canonical = np.full(12, 0.1)
  alternative = canonical.copy()
  alternative[:6] = 1.1
  # Canonical pre is supplied by the consistency pass; only the alternative needs solving.
  pre0 = canonical + 0.3
  (tasks, _seq, _names), _ = _tasks(
      {0: pre0, 1: np.full(12, 0.6)}, ik=ik,
      witness_sets={0: [canonical, alternative]})
  pre, skill = tasks[:2]
  assert len(pre.goal.goals) == 2
  assert isinstance(skill.goal, MP._PairedGoal)
  assert ik.solved == 1

  class _Mode:
    pass
  m = _Mode()
  qentry = np.zeros(12)
  qentry[:6] = pre.goal.goals[1]
  m.entry_configuration = qentry
  assert skill.goal.sample(m) == pytest.approx(alternative[:6])
  # Persisted task records must retain the conditional mapping for zero-IK replay.
  rebuilt = MP._task_from_record(MP._task_record(skill))
  assert isinstance(rebuilt.goal, MP._PairedGoal)
  assert rebuilt.goal.sample(m) == pytest.approx(alternative[:6])


class _ChainState:
  def __init__(self, groups, grasps, parent):
    self._groups = [set(g) for g in groups]
    self.parent = dict(parent)
    self.supports = {p: [] for g in self._groups for p in g}
    for r, part, gid in grasps:
      self.supports[part].append(("grasp", r, part, gid))

  def subassemblies(self):
    return [set(g) for g in self._groups]

  def bodies(self):
    return set().union(*self._groups)


class _ChainIK:
  def __init__(self, robots):
    self.robots = list(robots)
    self.pre_pick_delta = self.pre_place_delta = self.pre_add_grasp_delta = 0.0
    self.pre_insertion_delta = 0.05
    self.gripper_frame = {r: r + "_tool" for r in robots}
    self.solved = 0

  def holders(self, state):
    return {c[1]: (c[2], c[3]) for p in state.bodies()
            for c in state.supports.get(p, ()) if c[0] == "grasp"}

  def _holders_of_group(self, state, group):
    return [r for r, (part, _gid) in self.holders(state).items() if part in group]

  def pre_skill_witness(self, *args, **kwargs):
    self.solved += 1
    raise AssertionError("the supplied fused rendezvous must prevent an exporter re-solve")


def _task_goal_by_robot(task, width=6):
  q = np.asarray(task.goal.goal)
  return {r: q[k * width:(k + 1) * width] for k, r in enumerate(task.robots)}


def test_handover_uses_the_pre_add_grasp_funnel():
  MP._ensure_mrbench()
  robots = ["a1", "a2"]
  idx = {"a1": list(range(6)), "a2": list(range(6, 12))}
  plan = [("handover", "a2", "obj_0", 4)]
  s0 = _ChainState([{"obj_0"}], [("a1", "obj_0", 3)],
                   {"obj_0": ("gripper", "a1")})
  s1 = _ChainState([{"obj_0"}], [("a2", "obj_0", 4)],
                   {"obj_0": ("gripper", "a2")})
  q_skill = np.concatenate([np.full(6, 1.0), np.full(6, 2.0)])
  q_pre = np.concatenate([np.full(6, 1.0), np.full(6, 1.8)])
  rik = _ChainIK(robots)
  rik.pre_add_grasp_delta = 0.05

  tasks, seq, names = MP.build_tasks(
      None, plan, [q_skill], [s0, s1], robots, idx, rik, np.zeros(12),
      pre_witnesses={0: q_pre})

  assert rik.solved == 0
  assert seq == ["00_pre_handover_a2_obj_0_4", "00_handover_a2_obj_0_4", "terminal"]
  assert names[0] == seq[:2]
  assert tasks[0].robots == ["a1", "a2"]
  assert tasks[1].robots == ["a1", "a2"]
  np.testing.assert_allclose(tasks[0].goal.goal, q_pre)
  np.testing.assert_allclose(tasks[1].goal.goal, q_skill)


def test_place_gets_a_certified_post_detach_departure():
  """Set-down is a complete contact funnel, not a one-sided approach.

  Once the place task detaches the payload, the free-space planner must not be asked to
  discover an escape from the contact boundary.  The departure is certified in the new scene,
  with the payload left on its support. Only the placing arm retracts; an unrelated robot
  stays out of the departure task.
  """
  MP._ensure_mrbench()
  robots = ["a1", "a2"]
  idx = {"a1": list(range(6)), "a2": list(range(6, 12))}
  plan = [("place", "a1", "obj_0", 0, 0)]
  s0 = _ChainState([{"obj_0"}], [("a1", "obj_0", 3)],
                   {"obj_0": ("gripper", "a1")})
  s1 = _ChainState([{"obj_0"}], [], {"obj_0": ("table",)})
  q_place = np.concatenate([np.full(6, 1.0), np.full(6, 2.0)])
  q_pre = np.concatenate([np.full(6, 1.25), np.full(6, 2.0)])
  class _PostPlaceIK(_ChainIK):
    def post_release_witness(self, action, before, after, assembly, q):
      out = np.asarray(q, dtype=float).copy()
      out[idx[action[1]]] += 0.4
      return out

  rik = _PostPlaceIK(robots)
  rik.pre_place_delta = 0.05
  rik.pre_insertion_delta = 0.0

  tasks, seq, names = MP.build_tasks(
      None, plan, [q_place], [s0, s1], robots, idx, rik, np.zeros(12),
      pre_witnesses={0: q_pre})

  assert rik.solved == 0
  assert seq == ["00_pre_place_a1_obj_0_0_0", "00_place_a1_obj_0_0_0",
                 "00_post_place_a1_obj_0_0_0", "terminal"]
  assert names[0] == seq[:3]
  pre, place, post = tasks[:3]
  assert pre.goal.goal == pytest.approx(q_pre[idx["a1"]])
  assert place.goal.goal == pytest.approx(q_place[idx["a1"]])
  assert post.robots == ["a1"]
  assert post.goal.goal == pytest.approx(q_place[idx["a1"]] + 0.4)


def test_add_grasp_assembly_release_is_one_coupled_keyframe_chain():
  """The brace closes only after every assembly participant reaches the funnel mouth. The
  assembly and releases reuse adjacent endpoints, so no extra joint-transport keyframe is
  introduced. The unconstrained motion planner may still detour between those endpoints."""
  MP._ensure_mrbench()
  robots = ["a1", "a2", "a3"]
  idx = {r: list(range(6 * k, 6 * (k + 1))) for k, r in enumerate(robots)}
  plan = [("add_grasp", "a1", "obj_0", 0),
          ("assemble", "a3", "obj_1", "obj_0"),
          ("release", "a3", "obj_1"),
          ("release", "a1", "obj_0")]
  # a2 initially holds the target and a3 the loose part; a1 joins at add_grasp.
  s0 = _ChainState([{"obj_0"}, {"obj_1"}],
                   [("a2", "obj_0", 1), ("a3", "obj_1", 2)],
                   {"obj_0": ("gripper", "a2"), "obj_1": ("gripper", "a3")})
  s1 = _ChainState([{"obj_0"}, {"obj_1"}],
                   [("a1", "obj_0", 0), ("a2", "obj_0", 1),
                    ("a3", "obj_1", 2)],
                   {"obj_0": ("gripper", "a2"), "obj_1": ("gripper", "a3")})
  s2 = _ChainState([{"obj_0", "obj_1"}],
                   [("a1", "obj_0", 0), ("a2", "obj_0", 1),
                    ("a3", "obj_1", 2)],
                   {"obj_0": ("gripper", "a2"), "obj_1": ("part", "obj_0")})
  s3 = _ChainState([{"obj_0", "obj_1"}],
                   [("a1", "obj_0", 0), ("a2", "obj_0", 1)],
                   {"obj_0": ("gripper", "a2"), "obj_1": ("part", "obj_0")})
  s4 = _ChainState([{"obj_0", "obj_1"}],
                   [("a2", "obj_0", 1)],
                   {"obj_0": ("gripper", "a2"), "obj_1": ("part", "obj_0")})
  q_add = np.concatenate([np.full(6, 1.0), np.full(6, 2.0), np.full(6, 3.0)])
  q_pre = np.concatenate([np.full(6, 10.0), np.full(6, 20.0), np.full(6, 30.0)])
  q_seat = np.concatenate([np.full(6, 11.0), np.full(6, 21.0), np.full(6, 31.0)])
  rik = _ChainIK(robots)
  tasks, seq, _names = MP.build_tasks(
      None, plan, [q_add, q_seat, None, None], [s0, s1, s2, s3, s4], robots, idx, rik,
      np.zeros(18), pre_witnesses={1: q_pre})

  assert rik.solved == 0
  assert seq == ["00_add_grasp_a1_obj_0_0",
                 "01_pre_assemble_a3_obj_1_obj_0",
                 "01_assemble_a3_obj_1_obj_0",
                 "02_release_a3_obj_1", "03_release_a1_obj_0", "terminal"]
  add, pre, seat, release, final_release = tasks[:5]
  assert set(add.robots) == set(robots)                    # inserter joins the rendezvous task
  for r in robots:
    assert _task_goal_by_robot(add)[r] == pytest.approx(q_pre[idx[r]])
    assert _task_goal_by_robot(pre)[r] == pytest.approx(q_pre[idx[r]])
    assert _task_goal_by_robot(seat)[r] == pytest.approx(q_seat[idx[r]])
    assert _task_goal_by_robot(release)[r] == pytest.approx(q_seat[idx[r]])
    assert _task_goal_by_robot(final_release)[r] == pytest.approx(q_seat[idx[r]])
  assert set(final_release.robots) == set(robots)            # same coupled release endpoint


def test_post_release_retract_is_conditional_on_the_incoming_witness_member():
  MP._ensure_mrbench()
  robots = ["a1", "a2"]
  idx = {"a1": list(range(6)), "a2": list(range(6, 12))}
  plan = [("pick", "a1", "obj_0", 0), ("release", "a1", "obj_0")]
  s0 = _ChainState([{"obj_0"}], [], {"obj_0": ("table",)})
  s1 = _ChainState([{"obj_0"}], [("a1", "obj_0", 0)],
                   {"obj_0": ("gripper", "a1")})
  s2 = _ChainState([{"obj_0"}], [], {"obj_0": ("table",)})
  q0 = np.concatenate([np.full(6, 1.0), np.full(6, 2.0)])
  q1 = np.concatenate([np.full(6, 3.0), np.full(6, 2.0)])

  class _PostIK(_ChainIK):
    def __init__(self, rs):
      super().__init__(rs)
      self.post_release_delta = 0.05
      self.pre_insertion_delta = 0.0

    def post_release_witness(self, action, before, after, assembly, q):
      out = np.asarray(q, dtype=float).copy()
      out[idx[action[1]]] += 0.25
      return out

  rik = _PostIK(robots)
  tasks, seq, names = MP.build_tasks(
      None, plan, [q0, None], [s0, s1, s2], robots, idx, rik, np.zeros(12),
      witness_sets={0: [q0, q1]})
  assert seq == ["00_pick_a1_obj_0_0", "01_release_a1_obj_0",
                 "01_post_release_a1_obj_0", "terminal"]
  assert names[1] == ["01_release_a1_obj_0", "01_post_release_a1_obj_0"]
  release, post = tasks[1:3]
  assert len(release.goal.goals) == 2
  assert isinstance(post.goal, MP._PairedGoal)

  class _Mode:
    pass
  m = _Mode()
  m.entry_configuration = q1.copy()
  assert post.goal.sample(m) == pytest.approx(q1[idx["a1"]] + 0.25)


def test_cograsp_rendezvous_shares_an_intervening_place_mode_switch():
  """Cross has add_grasp -> place(target) -> assemble.  The place only switches the target
  from a gripper parent to the table; it must happen at the shared rendezvous instead of
  sending every holder to the old independently-solved place witness and back again."""
  MP._ensure_mrbench()
  robots = ["a1", "a2", "a3"]
  idx = {r: list(range(6 * k, 6 * (k + 1))) for k, r in enumerate(robots)}
  plan = [("add_grasp", "a1", "obj_0", 0),
          ("place", "a2", "obj_0", 7),
          ("assemble", "a3", "obj_1", "obj_0"),
          ("release", "a3", "obj_1"),
          ("release", "a1", "obj_0")]
  s0 = _ChainState([{"obj_0"}, {"obj_1"}],
                   [("a2", "obj_0", 1), ("a3", "obj_1", 2)],
                   {"obj_0": ("gripper", "a2"), "obj_1": ("gripper", "a3")})
  s1 = _ChainState([{"obj_0"}, {"obj_1"}],
                   [("a1", "obj_0", 0), ("a2", "obj_0", 1),
                    ("a3", "obj_1", 2)],
                   {"obj_0": ("gripper", "a2"), "obj_1": ("gripper", "a3")})
  s2 = _ChainState([{"obj_0"}, {"obj_1"}],
                   [("a1", "obj_0", 0), ("a3", "obj_1", 2)],
                   {"obj_0": ("table",), "obj_1": ("gripper", "a3")})
  s3 = _ChainState([{"obj_0", "obj_1"}],
                   [("a1", "obj_0", 0), ("a3", "obj_1", 2)],
                   {"obj_0": ("table",), "obj_1": ("part", "obj_0")})
  s4 = _ChainState([{"obj_0", "obj_1"}], [("a1", "obj_0", 0)],
                   {"obj_0": ("table",), "obj_1": ("part", "obj_0")})
  s5 = _ChainState([{"obj_0", "obj_1"}], [],
                   {"obj_0": ("table",), "obj_1": ("part", "obj_0")})
  q_add = np.concatenate([np.full(6, 1.0), np.full(6, 2.0), np.full(6, 3.0)])
  q_old_place = np.concatenate([np.full(6, 4.0), np.full(6, 5.0), np.full(6, 6.0)])
  q_pre = np.concatenate([np.full(6, 10.0), np.full(6, 20.0), np.full(6, 30.0)])
  q_seat = np.concatenate([np.full(6, 11.0), np.full(6, 21.0), np.full(6, 31.0)])
  rik = _ChainIK(robots)
  tasks, seq, _names = MP.build_tasks(
      None, plan, [q_add, q_old_place, q_seat, None, None],
      [s0, s1, s2, s3, s4, s5], robots, idx, rik, np.zeros(18),
      pre_witnesses={2: q_pre})

  assert rik.solved == 0
  assert seq[:4] == ["00_add_grasp_a1_obj_0_0", "01_place_a2_obj_0_7",
                     "02_pre_assemble_a3_obj_1_obj_0",
                     "02_assemble_a3_obj_1_obj_0"]
  add, support_switch, pre, seat = tasks[:4]
  assert set(add.robots) == {"a1", "a2"}
  assert set(support_switch.robots) == {"a1", "a2"}
  for r in add.robots:
    # Persistent brace a1 comes from the assemble funnel; former holder a2 is absent from that
    # post-place IK scene and must instead keep its physically meaningful place-witness slice.
    expected_pre = q_old_place if r == "a2" else q_pre
    assert _task_goal_by_robot(add)[r] == pytest.approx(expected_pre[idx[r]])
    assert _task_goal_by_robot(support_switch)[r] == pytest.approx(expected_pre[idx[r]])
  assert set(pre.robots) == {"a1", "a3"}
  assert set(seat.robots) == {"a1", "a3"}
  for r in pre.robots:
    assert _task_goal_by_robot(pre)[r] == pytest.approx(q_pre[idx[r]])
    assert _task_goal_by_robot(seat)[r] == pytest.approx(q_seat[idx[r]])
  assert "a2" not in pre.robots                         # released holder may retract


def test_cograsp_bridge_does_not_replace_an_intervening_pick_witness():
  """The former target holder may place the target and then pick the insertion part.

  Only the persistent brace is stationary across that bridge.  Treating the former holder as
  persistent because it later reappears as the *inserter* replaces the pick goal by the mate
  rendezvous: mr_bench then freezes a completely unrelated gripper/object transform at the
  pick side effect (the OpenArm car-mini wheel-through-table regression).
  """
  MP._ensure_mrbench()
  robots = ["a1", "a2"]
  idx = {r: list(range(6 * k, 6 * (k + 1))) for k, r in enumerate(robots)}
  plan = [("add_grasp", "a1", "obj_0", 0),
          ("place", "a2", "obj_0", 7),
          ("pick", "a2", "obj_1", 2),
          ("assemble", "a2", "obj_1", "obj_0")]
  s0 = _ChainState([{"obj_0"}, {"obj_1"}], [("a2", "obj_0", 1)],
                   {"obj_0": ("gripper", "a2"), "obj_1": ("table",)})
  s1 = _ChainState([{"obj_0"}, {"obj_1"}],
                   [("a1", "obj_0", 0), ("a2", "obj_0", 1)],
                   {"obj_0": ("gripper", "a2"), "obj_1": ("table",)})
  s2 = _ChainState([{"obj_0"}, {"obj_1"}], [("a1", "obj_0", 0)],
                   {"obj_0": ("table",), "obj_1": ("table",)})
  s3 = _ChainState([{"obj_0"}, {"obj_1"}],
                   [("a1", "obj_0", 0), ("a2", "obj_1", 2)],
                   {"obj_0": ("table",), "obj_1": ("gripper", "a2")})
  s4 = _ChainState([{"obj_0", "obj_1"}],
                   [("a1", "obj_0", 0), ("a2", "obj_1", 2)],
                   {"obj_0": ("table",), "obj_1": ("part", "obj_0")})
  q_add = np.concatenate([np.full(6, 1.0), np.full(6, 2.0)])
  q_place = np.concatenate([np.full(6, 4.0), np.full(6, 5.0)])
  q_pick = np.concatenate([np.full(6, 7.0), np.full(6, 8.0)])
  q_pre = np.concatenate([np.full(6, 10.0), np.full(6, 20.0)])
  q_seat = np.concatenate([np.full(6, 11.0), np.full(6, 21.0)])
  rik = _ChainIK(robots)
  tasks, seq, _names = MP.build_tasks(
      None, plan, [q_add, q_place, q_pick, q_seat], [s0, s1, s2, s3, s4],
      robots, idx, rik, np.zeros(12), pre_witnesses={3: q_pre})

  assert seq[:5] == ["00_add_grasp_a1_obj_0_0", "01_place_a2_obj_0_7",
                     "02_pick_a2_obj_1_2", "03_pre_assemble_a2_obj_1_obj_0",
                     "03_assemble_a2_obj_1_obj_0"]
  add, place, pick, pre, seat = tasks[:5]
  assert _task_goal_by_robot(add)["a1"] == pytest.approx(q_pre[idx["a1"]])
  assert _task_goal_by_robot(add)["a2"] == pytest.approx(q_place[idx["a2"]])
  assert _task_goal_by_robot(place)["a1"] == pytest.approx(q_pre[idx["a1"]])
  assert _task_goal_by_robot(place)["a2"] == pytest.approx(q_place[idx["a2"]])
  # This is the safety-critical assertion: the pick actor keeps the actual pick witness.
  assert _task_goal_by_robot(pick)["a1"] == pytest.approx(q_pre[idx["a1"]])
  assert _task_goal_by_robot(pick)["a2"] == pytest.approx(q_pick[idx["a2"]])
  assert _task_goal_by_robot(pre)["a2"] == pytest.approx(q_pre[idx["a2"]])
  assert _task_goal_by_robot(seat)["a2"] == pytest.approx(q_seat[idx["a2"]])


def test_the_pass_declines_cleanly_when_no_pre_split_is_active(arms):
  """--fuse-pre-skill without any --pre-* is a user error, not a crash: it must say so and
  return an empty mapping (which build_tasks then treats as "solve them yourself")."""
  C, robots = arms
  rik = S.RobotIK(C, robots)
  pre, rep = S.fuse_pre_skill_witnesses(C, None, [("pick", robots[0], "obj_0", 0)], [None],
                                        robots, rik, verbose=False)
  assert pre == {} and rep == {"splits": 0, "consistent": 0, "rows": []}
