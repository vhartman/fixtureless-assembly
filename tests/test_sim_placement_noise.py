import json
from types import SimpleNamespace

import numpy as np


def test_sim_report_has_structured_first_failure():
  import sim as S
  r = S.SimReport()
  r.results.append(S.ActionResult(("pick", "a0", "obj_0", 0), True))
  r.results.append(S.ActionResult(("assemble", "a0", "obj_1", "obj_0"),
                                  False, "not_seated", "12mm"))
  d = r.to_dict({"seed": 7})
  assert d["success"] is False
  assert d["first_failure_index"] == 1
  assert d["first_failure"] == "not_seated"
  assert d["results"][0]["action"][0] == "pick"
  assert d["metadata"] == {"seed": 7}
  json.dumps(d)


def test_zero_placement_perturbation_is_exact_copy():
  import sim as S
  from problem import State
  state = State({"obj_0": ("table",)}, poses={
      "obj_0": np.array([1., 2., 3., 1., 0., 0., 0.])})
  got, meta = S._sample_initial_placement_perturbation(
      None, None, state, xy=0.0, yaw_deg=0.0, seed=19)
  assert got is not state
  assert np.array_equal(got.poses["obj_0"], state.poses["obj_0"])
  assert got.poses["obj_0"] is not state.poses["obj_0"]
  assert meta["attempts"] == 0


def test_planar_placement_draw_is_collision_free_and_reproducible():
  import assembly as A
  from env import make_four_robot_env
  import sim as S
  asm = A.make_assembly("car-mini", 5)
  C = make_four_robot_env(num_robots=1, floating_ee=True, vacuum=False)
  A.add_disassembled_obj(C, asm, spacing=0.2, float_height=0.002)
  state = A.get_initial_state(C, asm)
  a, ma = S._sample_initial_placement_perturbation(
      C, asm, state, xy=0.005, yaw_deg=3.0, seed=11)
  b, mb = S._sample_initial_placement_perturbation(
      C, asm, state, xy=0.005, yaw_deg=3.0, seed=11)
  assert ma == mb and ma["attempts"] >= 1
  assert all(np.array_equal(a.poses[p], b.poses[p]) for p in asm.objects)
  assert all(a.poses[p][2] == state.poses[p][2] for p in asm.objects)


def test_noisy_runner_builds_fail_closed_reproduction_command(tmp_path):
  from benchmarks.run_noisy_sim import build_command
  plan, mp = tmp_path / "p.json", tmp_path / "mp.json"
  plan.write_text("{}")
  mp.write_text("{}")
  args = SimpleNamespace(
      plan=plan, mp_path=mp, sim_profile="standard", max_placement_tries=50,
      grasp_capture=0.008, grasp_capture_angle=7.0, apply_process_wrench=True,
      process_noise="sample", process_noise_corner=0, extra=[], memory_gb=0.0)
  trial = {"xy_m": 0.005, "yaw_deg": 3.0, "seed": 11,
           "observation": "observed"}
  cmd = build_command(args, trial, tmp_path / "r.json", tmp_path / "x.npz")
  assert cmd[:3] == [__import__("sys").executable,
                     str(__import__("benchmarks.run_noisy_sim",
                                    fromlist=["ROOT"]).ROOT / "sim.py"), "run"]
  assert "--sim-profile" in cmd and "standard" in cmd
  assert cmd[cmd.index("--object-perturb-seed") + 1] == "11"
  assert cmd[cmd.index("--process-noise") + 1] == "sample"
  assert "--report-json" in cmd and "--record" in cmd
