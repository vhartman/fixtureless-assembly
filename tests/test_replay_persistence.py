"""Replay must be a reader: opening a saved plan may not resolve IK implicitly."""
import numpy as np
import pytest

import replay as R
from config import RunConfig


class _C:
  def getJointNames(self):
    return ["a1_j0", "a1_j1"]


class _IK:
  def __init__(self, C, robots):
    self.C = C
    self.pre_insertion_delta = self.pre_pick_delta = 0.0
    self.pre_place_delta = self.pre_add_grasp_delta = 0.0
    self.strip_inactive_collision = True
    self.penetration_reject = None
    self.col_weight = 0.0


def _patch_scene(monkeypatch, witnesses):
  cfg = {"assembly": "cross", "parts": 1, "num_robots": 1, "real_ik": True,
         "bracing": False, "pre_insertion": 0.0}
  plan = [("pick", "a1", "obj_0", 0)]
  monkeypatch.setattr(R, "read_plan", lambda _p: (cfg, plan))
  monkeypatch.setattr(R, "build_scene", lambda _cfg, **_kw: (_C(), object()))
  monkeypatch.setattr(R, "get_robots", lambda _C: ["a1"])
  monkeypatch.setattr(R, "RobotIK", _IK)
  monkeypatch.setattr(R, "read_witnesses", lambda *_a: witnesses)
  return plan


def test_replay_passes_persisted_witnesses_to_visualizer(monkeypatch):
  witnesses = [np.array([0.1, 0.2])]
  _patch_scene(monkeypatch, witnesses)
  seen = {}
  monkeypatch.setattr(
      R, "visualize_plan_ik",
      lambda *_a, **kw: seen.setdefault("witnesses", kw.get("witnesses")))
  R.replay("plan.json", view=True)
  assert seen["witnesses"] is witnesses


def test_replay_refuses_missing_witness_without_explicit_resolve(monkeypatch):
  _patch_scene(monkeypatch, None)
  monkeypatch.setattr(R, "visualize_plan_ik",
                      lambda *_a, **_kw: pytest.fail("visualizer must not be reached"))
  with pytest.raises(SystemExit, match="refusing to resolve IK implicitly"):
    R.replay("plan.json", view=True)


def test_replay_restores_vacuum_engagement_successor_gate():
  import wrench

  R.apply_wrench_params({"vacuum_engage_force": 7.5})
  assert wrench.PARAMS["vacuum_engage_force"] == 7.5
  R.apply_wrench_params({})
  assert wrench.PARAMS["vacuum_engage_force"] == 0.0


def test_replay_restores_orientation_robust_wrench_gate():
  import wrench

  R.apply_wrench_params({"orientation_robust_wrench": True})
  assert wrench.PARAMS["orientation_robust_gravity"] is True
  R.apply_wrench_params({})
  assert wrench.PARAMS["orientation_robust_gravity"] is False


def test_replay_restores_max_robotiq_profile_and_then_resets_it():
  import wrench

  R.apply_wrench_params({"wrench_profile": "max_robotiq", "grasp_model": "asymmetric"})
  assert wrench.PARAMS["grasp"]["asymmetric"]["two_finger"] == {
      "F_push": 50.0, "F_pull": 50.0, "mu": 1.0,
      "F_t1": 50.0, "F_t2": 50.0,
      "T_axis": 3.0, "T_lat": 5.0, "T_close": 5.0,
      "T_t1": 5.0, "T_t2": 5.0,
  }
  assert wrench.PARAMS["grasp"]["directional"]["two_finger"] == {
      "F_axial": 50.0, "mu": 1.0, "F_t1": 50.0, "F_t2": 50.0,
      "T_axis": 3.0, "T_lat": 5.0, "T_t1": 5.0, "T_t2": 5.0,
  }
  R.apply_wrench_params({})
  assert wrench.PARAMS["grasp"]["asymmetric"]["two_finger"]["T_close"] == 0.4
  assert wrench.PARAMS["grasp"]["asymmetric"]["two_finger"]["F_pull"] == 25.0


def test_replay_restores_measured_axis_resolved_profile_and_then_resets_it():
  import wrench

  R.apply_wrench_params({"wrench_profile": "measured", "grasp_model": "asymmetric"})
  tf = wrench.PARAMS["grasp"]["asymmetric"]["two_finger"]
  vac = wrench.PARAMS["grasp"]["asymmetric"]["vacuum"]
  assert (tf["F_t1"], tf["F_t2"], tf["F_push"], tf["F_pull"]) == \
      (50.0, 39.7, 50.0, 50.0)
  assert (tf["T_t1"], tf["T_close"], tf["T_axis"]) == (4.7, 4.5, 3.0)
  assert (vac["F_t1"], vac["F_t2"], vac["F_push"], vac["F_pull"]) == \
      (5.0, 5.0, 110.0, 110.0)
  assert (vac["T_t1"], vac["T_t2"], vac["T_axis"]) == (0.1, 0.1, 0.1)

  R.apply_wrench_params({})
  assert "F_t1" not in wrench.PARAMS["grasp"]["asymmetric"]["two_finger"]
  assert wrench.PARAMS["grasp"]["asymmetric"]["vacuum"]["T_axis"] == 0.8


def test_build_scene_roundtrips_centered_scatter_and_legacy_layout():
  """Replay must rebuild the saved world poses, not today's profile-owned layout.

  In particular, the compact centred lattice used by the standard profile is materially
  different from the historical left-anchored line.  A missing field is an old plan and must
  retain that old layout.
  """
  cfg = RunConfig(assembly="cross", parts=3, num_robots=1, floating_ee=True,
                  vacuum=True, scatter_spacing=0.2, center_scatter=True).to_dict()
  C, assembly = R.build_scene(cfg)
  centered = [C.getFrame(p).getRelativePosition()[:2] for p in assembly.objects]
  np.testing.assert_allclose(centered, [[-0.1, -0.1], [0.1, -0.1], [0.0, 0.1]])

  legacy = dict(cfg)
  legacy.pop("center_scatter")
  C_old, assembly_old = R.build_scene(legacy)
  old_xy = [C_old.getFrame(p).getRelativePosition()[:2] for p in assembly_old.objects]
  np.testing.assert_allclose(old_xy, [[-0.5, 0.0], [-0.3, 0.0], [-0.1, 0.0]])


def test_build_scene_restores_discrete_mate_symmetry_and_legacy_off():
  cfg = RunConfig(assembly="real-stool", parts=2, num_robots=1, floating_ee=True,
                  mate_symmetry=True).to_dict()
  _C, assembly = R.build_scene(cfg)
  assert len(assembly.mate_quaternions("obj_1")) == 2

  legacy = dict(cfg)
  legacy.pop("mate_symmetry")
  _C_old, assembly_old = R.build_scene(legacy)
  assert len(assembly_old.mate_quaternions("obj_1")) == 1


def test_build_scene_restores_sampling_expansion_and_resets_legacy(monkeypatch):
  import problem as P
  seen = []
  original = P.set_sampled_grasps
  def capture(mode):
    seen.append(mode)
    original(mode)
  monkeypatch.setattr(P, "set_sampled_grasps", capture)
  cfg = RunConfig(assembly="cross", parts=1, num_robots=1, floating_ee=True,
                  grasp_sampler=True, grasp_sampler_full_probe=True,
                  grasp_sampler_roll_refine=True, grasp_sampler_keep=8,
                  grasp_widen=True, grasp_widen_max=24).to_dict()
  R.build_scene(cfg)
  assert seen[-1]["full_probe"] and seen[-1]["roll_refine"]
  assert seen[-1]["master_keep"] == 24 and seen[-1]["fixed_master"]
  cfg.pop("grasp_sampler_full_probe")
  cfg.pop("grasp_sampler_roll_refine")
  cfg.pop("grasp_widen")
  R.build_scene(cfg)
  assert not seen[-1]["full_probe"] and not seen[-1]["roll_refine"]
  assert seen[-1]["master_keep"] == 8 and not seen[-1]["fixed_master"]
