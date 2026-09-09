import numpy as np

import problem as P
import wrench as W


class _VerticalChain(P.Assembly):
  def define(self):
    self.objects = ["obj_0", "obj_1"]
    self.base = "obj_0"
    self.cube_size = 0.05
    self.cube_radius = 0.0
    self.assembly_dependencies = {"obj_1": ["obj_0"]}
    self.assembly_relative_poses = {
        "obj_1": ["obj_0", np.array([0.0, 0.0, 0.05])],
    }
    self.masses = {"obj_0": 0.1, "obj_1": 0.1}


def _assembled_on_table(assembly):
  return P.State(
      parent={"obj_0": ("table",), "obj_1": ("part", "obj_0")},
      place={"obj_0": ("slot", 0, P._IDENTITY_FACE)},
      supports={"obj_0": frozenset({("table",)}),
                "obj_1": frozenset({("mate", "obj_0")})},
      poses=P._flat_poses(assembly),
  )


def test_finite_mate_must_carry_child_weight_and_ablation_restores_ideal_weld():
  saved_ignore = W.PARAMS.get("ignore_gravity")
  try:
    W.PARAMS["ignore_gravity"] = False
    assembly = _VerticalChain()
    state = _assembled_on_table(assembly)
    group = {"obj_0", "obj_1"}

    # obj_1 weighs 0.981N and only reaches the table through this interface.
    assembly.mate_wrench_limits["obj_1"] = {
        "force": [0.5, 0.5, 0.5], "torque": [0.1, 0.1, 0.1]}
    P.set_mate_wrench_limits(True)
    P.reset_wrench_cache()
    assert not P._group_stable(state, assembly, group, "two_finger")

    assembly.mate_wrench_limits["obj_1"] = {
        "force": [2.0, 2.0, 2.0], "torque": [0.1, 0.1, 0.1]}
    P.reset_wrench_cache()
    assert P._group_stable(state, assembly, group, "two_finger")

    # The explicit baseline contracts the same mate into one rigid body and is therefore
    # feasible even under a deliberately tiny authored interface cap.
    assembly.mate_wrench_limits["obj_1"] = {
        "force": [0.01, 0.01, 0.01], "torque": [0.0, 0.0, 0.0]}
    P.set_mate_wrench_limits(False)
    P.reset_wrench_cache()
    assert P._group_stable(state, assembly, group, "two_finger")
  finally:
    W.PARAMS["ignore_gravity"] = saved_ignore
    P.set_mate_wrench_limits(True)
    P.reset_wrench_cache()


def test_uncapped_group_never_builds_multibody_matrix(monkeypatch):
  assembly = _VerticalChain()
  state = _assembled_on_table(assembly)
  P.set_mate_wrench_limits(True)
  monkeypatch.setattr(P, "_multibody_grounding_caps",
                      lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("new path")))
  assert P._group_stable(state, assembly, {"obj_0", "obj_1"}, "two_finger")
