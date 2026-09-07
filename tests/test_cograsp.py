"""Co-grasp rigidity: the mp_export drift validator and the 2-keyframe witness-fusion pass.

The validator is exercised on a SYNTHETIC 2-arm co-grasp whose drift is known ANALYTICALLY:
rotating one revolute joint upstream of a tool centre by theta changes that gripper's pose by
A -> A o M, where M is a conjugation of a rotation by theta. The pair deviation
inv(ref) o cur is then a conjugation of inv(M), and conjugation preserves the rotation ANGLE
-- so the reported degrees must equal theta EXACTLY, whichever arm of the pair moved. The
millimetres are not frame-independent (they are read in the anchor's frame), which is why
they are only bounded here and pinned separately on hand-built matrices.

No MuJoCo and no mr_bench: the mode chain / task list the reporter walks are stubbed.
"""
import numpy as np
import pytest

import mp_export as MP
from env import make_four_robot_env


# --------------------------------------------------------------------------- stubs
class _Goal:
  def __init__(self, q):
    self.q = q

  def sample(self, mode):
    return self.q


class _Task:
  def __init__(self, name, robots, q):
    self.name, self.robots, self.goal = name, robots, _Goal(q)


class _Conf(list):
  # mr_bench's configuration type, reduced to what mode_sequence's goal_config touches.
  def robot_state(self, j):
    return self[j]

  @classmethod
  def from_list(cls, parts):
    return cls(parts)


class _State:
  def __init__(self, groups, supports):
    self._groups, self.supports = groups, supports

  def subassemblies(self):
    return [set(g) for g in self._groups]


class _Env:
  """The SequenceMixin surface report_cograsp_drift + mode_sequence actually use."""

  def __init__(self, robots, dims, tasks, chain, states, step_names):
    self.robots, self.robot_dims, self.tasks = robots, dims, tasks
    self._chain, self.states, self.step_names = chain, states, step_names
    self.sequence = list(range(len(tasks)))
    self.start_mode = chain[0]

  def get_start_pos(self):
    return _Conf([np.zeros(self.robot_dims[r]) for r in self.robots])

  def is_terminal_mode(self, m):
    return m == self._chain[-1]

  def get_valid_next_task_combinations(self, m):
    return []

  def get_active_task(self, m, combo):
    return self.tasks[self._chain.index(m)]

  def get_next_modes(self, q, m):
    return [self._chain[self._chain.index(m) + 1]]


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def scene():
  """A real 2-arm rai scene: the validator's only geometric dependency is FK on the two
  tool-centre frames, so the arms must be real (a floating-EE scene has no joint chain).
  The tool-centre names come from a real RobotIK -- its __init__ is a config copy plus frame
  lookups, and hardcoding 'a1_ur_gripper_center' would silently rot when the EE changes."""
  import assembly as A
  import solver as S
  C = make_four_robot_env(num_robots=2, floating_ee=False)
  robots = A.get_robots(C)
  rik = S.RobotIK(C, robots)
  assert len(robots) == 2
  return C, robots, rik


def _problem(robots, holders1, holders2):
  """3 discrete states / 2 plan actions / 3 modes. holders1 and holders2 are the grasp
  contacts on part A (and the merged group) at states 1 and 2."""
  dims = {r: 6 for r in robots}
  tasks = [_Task("t0", [robots[0]], np.zeros(6)), _Task("t1", [robots[0]], np.zeros(6))]
  chain = [MP._Mode([0]), MP._Mode([1]), MP._Mode([2])]
  states = [
      _State([{"A"}, {"B"}], {"A": (("grasp", robots[1], "A", 0),)}),
      _State([{"A"}, {"B"}], {"A": holders1}),
      _State([{"A", "B"}], {"A": holders2[0], "B": holders2[1]}),
  ]
  return _Env(list(robots), dims, tasks, chain, states, [["t0"], ["t1"]])


def _path(C, robots, chain_counts, perturb=None):
  """chain_counts = how many path states carry mode 0 / 1 / 2. `perturb` = (path index,
  robot, joint-name substring, radians)."""
  chain = [MP._Mode([0]), MP._Mode([1]), MP._Mode([2])]
  jn = C.getJointNames()
  blocks = {r: [i for i, n in enumerate(jn) if n.startswith(r)] for r in robots}
  flat0 = np.zeros(sum(len(blocks[r]) for r in robots))
  states, k = [], 0
  for m, n in zip(chain, chain_counts):
    for _ in range(n):
      q = flat0.copy()
      if perturb is not None and perturb[0] == k:
        _, r, sub, ang = perturb
        off = sum(len(blocks[rr]) for rr in robots[:robots.index(r)])
        loc = next(i for i, gi in enumerate(blocks[r]) if sub in jn[gi])
        q[off + loc] += ang
      states.append(MP._PathState(q, m))
      k += 1
  return states


# --------------------------------------------------------------------------- validator
def test_identical_configs_have_exactly_zero_drift(scene):
  C, robots, rik = scene
  hold = (("grasp", robots[1], "A", 0), ("grasp", robots[0], "A", 1))
  prob = _problem(robots, hold, (hold, (("grasp", robots[0], "B", 1),)))
  rows = MP.report_cograsp_drift(C, prob, _path(C, robots, (2, 4, 3)), rik, verbose=False)
  assert len(rows) == 1
  assert rows[0][6] < 1e-9 and rows[0][8] < 1e-9


@pytest.mark.parametrize("mover", [0, 1])
def test_one_joint_of_theta_reads_back_as_exactly_theta(scene, mover):
  """The load-bearing metric check. Conjugation preserves the rotation angle, so it does not
  matter WHICH of the two co-grasping arms moved -- both must report theta."""
  C, robots, rik = scene
  theta = 0.7
  hold = (("grasp", robots[1], "A", 0), ("grasp", robots[0], "A", 1))
  prob = _problem(robots, hold, (hold, (("grasp", robots[0], "B", 1),)))
  path = _path(C, robots, (2, 4, 3), perturb=(4, robots[mover], "wrist_1", theta))
  rows = MP.report_cograsp_drift(C, prob, path, rik, verbose=False)
  assert len(rows) == 1
  assert rows[0][8] == pytest.approx(np.degrees(theta), abs=1e-6)
  assert rows[0][9] == 4                                   # peak at the perturbed path state
  assert rows[0][6] > 1.0                                  # a wrist swing also moves the TCP


def test_span_merges_across_group_growth_and_anchors_on_the_incumbent(scene):
  """An assemble merges the mated part in without moving the grippers, so the span must GROW
  rather than restart. The anchor is the arm whose UNINTERRUPTED grasp started earliest --
  here robots[1], which is NOT the alphabetically first, so the rule is discriminated."""
  C, robots, rik = scene
  hold = (("grasp", robots[1], "A", 0), ("grasp", robots[0], "A", 1))
  prob = _problem(robots, hold, (hold, (("grasp", robots[0], "B", 1),)))
  rows = MP.report_cograsp_drift(C, prob, _path(C, robots, (2, 4, 3)), rik, verbose=False)
  i0, i1, grp, hs, anchor, other = rows[0][:6]
  assert (i0, i1) == (2, 8)                                # one span, not two
  assert grp == frozenset({"A", "B"}) and set(hs) == set(robots)
  assert anchor == robots[1] and other == robots[0]


def test_a_single_holder_is_not_a_cograsp(scene):
  C, robots, rik = scene
  solo = (("grasp", robots[1], "A", 0),)
  prob = _problem(robots, solo, (solo, (("grasp", robots[1], "B", 0),)))
  assert MP.report_cograsp_drift(C, prob, _path(C, robots, (2, 4, 3)), rik, verbose=False) == []


def test_declines_without_the_state_stash(scene):
  """make_problem stashes .states only for this reporter; a DEPGRAPH problem has no
  .sequence. Either way the reporter must decline, not raise -- main.py defaults to
  DEPGRAPH, so this is the common path."""
  C, robots, rik = scene
  hold = (("grasp", robots[1], "A", 0), ("grasp", robots[0], "A", 1))
  path = _path(C, robots, (2, 4, 3))
  prob = _problem(robots, hold, (hold, (("grasp", robots[0], "B", 1),)))
  del prob.states
  assert MP.report_cograsp_drift(C, prob, path, rik, verbose=False) == []
  prob = _problem(robots, hold, (hold, (("grasp", robots[0], "B", 1),)))
  del prob.sequence
  assert MP.report_cograsp_drift(C, prob, path, rik, verbose=False) == []
  prob = _problem(robots, hold, (hold, (("grasp", robots[0], "B", 1),)))
  assert MP.report_cograsp_drift(C, prob, [], rik, verbose=False) == []


def test_translation_is_read_in_the_anchor_frame_but_the_angle_is_not():
  """Why the anchor rule must be deterministic and documented: swapping which gripper of the
  pair anchors changes the MILLIMETRES (measured 1106.4 vs 1190.3 on the shipped car-mini
  path) but never the degrees. Hand-computed here: anchor spins 90 deg in place while the
  other sits 1 m away -- anchored on the spinner the other swings sqrt(2) m, anchored on the
  stationary one it does not move at all."""
  s = np.sqrt(0.5)
  A0 = MP._pose_to_T([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  B0 = MP._pose_to_T([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  A = MP._pose_to_T([0.0, 0.0, 0.0, s, 0.0, 0.0, s])       # A spun 90 deg about z
  B = B0                                                   # B unmoved
  inv = np.linalg.inv
  mm_a, deg_a = MP._rel_deviation(inv(A0) @ B0, inv(A) @ B)      # anchored on A
  mm_b, deg_b = MP._rel_deviation(inv(B0) @ A0, inv(B) @ A)      # anchored on B
  assert deg_a == pytest.approx(90.0) and deg_b == pytest.approx(90.0)
  assert mm_a == pytest.approx(1000.0 * np.sqrt(2.0), abs=1e-6)
  assert mm_b == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- fusion gate
def test_holder_mapping_uses_canonical_robot_order():
  """A state's grasp contacts are frozensets; their hash order must not leak into MP goals."""
  import solver as S

  class _HolderState:
    supports = {"obj_0": frozenset({
        ("grasp", "a2", "obj_0", 2),
        ("grasp", "a0", "obj_0", 0),
        ("grasp", "a1", "obj_0", 1),
    })}

    def bodies(self):
      return ["obj_0"]

  rik = object.__new__(S.RobotIK)
  rik.robots = ["a0", "a1", "a2"]
  holders = rik.holders(_HolderState())
  assert list(holders) == rik.robots
  assert rik._holders_of_group(_HolderState(), {"obj_0"}) == rik.robots


def test_fuse_is_an_exact_noop_when_there_is_no_cograsp(monkeypatch):
  """A plan without add_grasp has no (add_grasp, assemble) pair, so the pass must hand back
  the SAME witnesses in a NEW list -- no re-solve, no mutation of the caller's list."""
  import assembly as A
  import solver as S

  A.set_seed(0)
  A.set_rolls_per_face(1)
  asm = A.make_assembly("cross", 3)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  plan = A.solve_assembly_problem(C, asm, robots=robots, verbose=False, robot_ik=None)
  assert plan and not any(a[0] == "add_grasp" for a in plan)

  class _StubIK:                                           # the no-pair path reads only these
    C = None
    penetration_reject = None

  stub = _StubIK()
  stub.C = C
  monkeypatch.setattr(S, "_search_context",
                      lambda *_a, **_k: pytest.fail("empty fusion must not build context"))
  monkeypatch.setattr(S, "_replay_states",
                      lambda *_a, **_k: pytest.fail("empty fusion must not replay plan"))
  wit = [np.arange(3.0) if i % 2 else None for i in range(len(plan))]
  out, rep = S.fuse_cograsp_witnesses(C, asm, plan, wit, robots, stub, verbose=False)
  assert rep == {"pairs": 0, "fused": 0, "rows": [], "pre_witnesses": {}}
  assert out is not wit and len(out) == len(wit)
  assert all(a is b for a, b in zip(out, wit))             # untouched, not merely equal


def test_known_action_scoped_lookup_matches_global_successor():
  """Fixed-plan replay may skip unrelated loops, but must return the same states."""
  import assembly as A
  import solver as S

  A.set_seed(0)
  A.set_rolls_per_face(1)
  asm = A.make_assembly("cross", 3)
  C = make_four_robot_env(num_robots=2, floating_ee=True, vacuum=True)
  A.add_disassembled_obj(C, asm, spacing=0.2)
  robots = A.get_robots(C)
  _checker, table = S._search_context(C, asm, robots)
  plan = A.solve_assembly_problem(C, asm, robots=robots, verbose=False, robot_ik=None)
  state = A.get_initial_state(C, asm)
  for action in plan:
    expected = next(ns for act, ns in S.successors(
        state, asm, robots, grasp_table=table) if act == action)
    actual = S._apply_known_action(state, action, asm, robots, table)
    assert actual == expected
    state = actual
