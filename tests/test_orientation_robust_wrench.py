"""Exact continuous all-orientation gravity robustness."""

import copy

import numpy as np

import assembly_planner.problem as P
import assembly_planner.wrench as W


def test_gravity_wrench_map_matches_direct_rigid_body_wrench():
  masses_points = [(0.3, np.array([0.1, -0.2, 0.05])),
                   (0.7, np.array([-0.04, 0.03, 0.2]))]
  A = W.gravity_wrench_map(masses_points)
  d = np.array([0.2, -0.7, 0.4])
  d /= np.linalg.norm(d)
  expected = np.zeros(6)
  for mass, point in masses_points:
    force = mass * W.GRAVITY * d
    expected += np.r_[force, np.cross(point, force)]
  np.testing.assert_allclose(A @ d, expected, atol=1e-12)


def test_all_orientation_containment_is_continuous_and_handles_asymmetry():
  # G[0,ub] = [-1, 2]. The complete 1-D gravity ball [-a,a] fits through a=1 exactly;
  # checking only the nominal centre would miss the a=1.1 violation on the weak side.
  G = np.array([[1.0, -1.0]])
  ub = np.array([2.0, 1.0])
  assert W.equilibrium_all_orientations(
      G, ub, np.zeros(1), np.array([[1.0, 0.0, 0.0]]))
  assert not W.equilibrium_all_orientations(
      G, ub, np.zeros(1), np.array([[1.001, 0.0, 0.0]]))


def test_all_orientation_zonotope_check_implies_every_sampled_lp():
  # A full-dimensional 6-D zonotope with redundant, non-axis-aligned generators exercises
  # the general facet path (not the one-grasp/box intuition). If continuous containment says
  # yes, every sampled sphere direction must pass the ordinary LP membership test.
  rng = np.random.default_rng(17)
  G = np.c_[np.eye(6), -np.eye(6), rng.normal(size=(6, 5))]
  ub = np.r_[np.full(12, 2.0), np.full(5, 0.3)]
  w0 = rng.normal(size=6) * 0.1
  A = rng.normal(size=(6, 3)) * 0.15
  assert W.equilibrium_all_orientations(G, ub, w0, A)
  for _ in range(300):
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    assert W.equilibrium_feasible_matrices(G, ub, w0 + A @ d)


class _OnePart:
  objects = ["obj_0"]
  no_hold = set()
  DEFAULT_MASS = 0.1

  def mass(self, _part):
    return 0.1

  def part_shape(self, _part):
    return ("box", 0.1, 0.1, 0.1)

  def position_in_base(self, _part):
    return np.zeros(3)


def test_planner_applies_orientation_robustness_only_to_freely_held_groups(monkeypatch):
  """The flag strengthens airborne holds but must not rotate a table-rest placement."""
  asm = _OnePart()
  poses = {"obj_0": np.array([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])}
  held = P.State(parent={"obj_0": ("gripper", "a1")}, grasp={"obj_0": 0},
                 supports={"obj_0": frozenset({("grasp", "a1", "obj_0", 0)})},
                 poses=poses)
  resting = P.State(parent={"obj_0": ("table",)}, place={"obj_0": ("slot", 0, 5)},
                    supports={"obj_0": frozenset({("table",)})}, poses=poses)

  # A deliberately weak off-centre grasp: it carries the nominal top-down orientation, but
  # cannot carry the cube when gravity is transverse. Avoid depending on the grasp sampler;
  # this test is about _group_stable's dispatch, not grasp construction.
  G = np.c_[np.eye(6), -np.eye(6)]
  one_side = np.array([2.0, 2.0, 2.0, 0.01, 0.01, 0.01])
  ub = np.r_[one_side, one_side]

  def caps(state, assembly, group, gripper_kind, table_override=None):
    if ("table",) in state.supports["obj_0"]:
      # Fixed table pose remains feasible and never enters the orientation-robust branch.
      return np.zeros(3), np.zeros(6), G, np.full(12, 100.0), \
          [(0.0, 100.0)] * 12, (G.tobytes(), b"table")
    # Nominal gravity is exactly balanceable, while the robust gravity map below is not.
    return np.zeros(3), np.zeros(6), G, ub, [(0.0, float(x)) for x in ub], \
        (G.tobytes(), ub.tobytes())

  monkeypatch.setattr(P, "_grounding_caps", caps)
  monkeypatch.setattr(W, "gravity_wrench_map",
                      lambda _mp: np.array([[0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0],
                                            [0.02, 0.0, 0.0],
                                            [0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0]]))
  saved = copy.deepcopy(W.PARAMS)
  try:
    W.PARAMS["orientation_robust_gravity"] = False
    P.reset_wrench_cache()
    assert P._group_stable(held, asm, {"obj_0"}, {"a1": "two_finger"})
    W.PARAMS["orientation_robust_gravity"] = True
    P.reset_wrench_cache()
    assert not P._group_stable(held, asm, {"obj_0"}, {"a1": "two_finger"})
    assert P._group_stable(resting, asm, {"obj_0"}, {"a1": "two_finger"})
  finally:
    W.PARAMS.clear()
    W.PARAMS.update(saved)
    P.reset_wrench_cache()
