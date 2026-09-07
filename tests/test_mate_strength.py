"""Finite mate strength (--mate-force-cap/--mate-torque-cap): mate welds break emergently.

The motivating case (car-mini-3): the planner may hold the whole car by ONE WHEEL. With
infinite welds that is legal; with finite mate strength the wheel-body weld must carry the
whole rest of the group and FAILS -- while holding by the BODY loads each wheel mate with
only its own part and stays fine. The masses are scaled x20 (2 kg parts) because the repo's
0.1 kg parts load a mate with ~3 N statically, below even a normal insertion press (15 N).

The lift uses a cosine ramp: a constant-velocity mocap jump excites the weld chain with a
~1.6x dynamic overshoot and wheel-swing torque spikes, which is a property of the abusive
TEST INPUT, not of the model under test.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sim as S


def _have(*files):
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  return all(os.path.exists(os.path.join(root, f)) for f in files)


def _assembled_scene(mate_caps):
  """The assembled car resting on the table, every mate welded at its ACHIEVED pose.

  Welding at the exact datum instead SNAPS whatever residual the settled poses carry into
  the constraint (measured 3.6 kN transients) -- the pipeline never does that either: it
  datum-welds only within the seat tolerance."""
  import mujoco
  import replay as R
  from plan_io import read_plan
  import assembly as A
  cfg, plan = read_plan("car-mini-new.json")
  C, asm = R.build_scene(cfg)
  robots = A.get_robots(C)
  states = S._states_from_plan(cfg, C, asm, robots, plan)
  for p in list(asm.masses):
    asm.masses[p] = asm.masses[p] * 20.0
  scene = S.SimScene(asm, states[-1], robots)
  scene.mate_caps = mate_caps
  mujoco.mj_forward(scene.model, scene.data)
  scene.step(int(0.3 / S.DT))
  for p, (eq, par) in scene._mate_eqs.items():
    scene._activate_weld(eq, scene._body_id(par), scene._body_id(p))
  scene.step(int(0.2 / S.DT))
  scene._mate_max.clear()
  scene._mate_over.clear()
  return scene, robots


def _hang(scene, robots, grab):
  """Grasp `grab` with an UNBREAKABLE grip (isolate the mate) and lift 0.3 m."""
  import mujoco
  report = S.SimReport()
  gp = scene.part_pose(grab)
  scene.drive_mocap(robots[0], np.r_[gp[:3] + [0.0, 0.0, 0.10], 1.0, 0.0, 0.0, 0.0])
  mujoco.mj_forward(scene.model, scene.data)
  scene.grasp(robots[0], grab)
  g = scene.active_grasps[robots[0]]
  scene.active_grasps[robots[0]] = (g[0], g[1], 1e9, 1e9)
  mid = scene.model.body_mocapid[scene._body_id(f"ee_{robots[0]}")]
  z0 = float(scene.data.mocap_pos[mid][2])
  n = int(4.0 / S.DT)
  for k in range(n):
    s = 0.5 * (1.0 - np.cos(np.pi * (k + 1) / n))
    scene.data.mocap_pos[mid][2] = z0 + 0.3 * s
    scene.step(1, report, ("hang", robots[0], grab))
  scene.step(int(0.5 / S.DT), report, ("hold",))
  broke = [r for r in report.results if r.failure == "mate_broke"]
  return broke, float(scene.part_pose("obj_0")[2])


CAPS = (40.0, float("inf"))                                # between 20 N (own part) and 78 N


@pytest.mark.skipif(not _have("car-mini-new.json"), reason="car-mini plan not present")
def test_active_mate_weld_suppresses_only_its_internal_contact_pair():
  """A weld and a contact constraint between the same bodies are contradictory.

  Collision stays live before insertion and returns if a breakable mate deactivates; only an
  active direct mate pair is filtered.
  """
  scene, _robots = _assembled_scene(None)
  eq, _parent = scene._mate_eqs["obj_1"]
  geom = {}
  for gid in range(scene.model.ngeom):
    owner = scene._part_owner_by_body[int(scene.model.geom_bodyid[gid])]
    if owner in ("obj_0", "obj_1") and owner not in geom:
      geom[owner] = gid
  assert set(geom) == {"obj_0", "obj_1"}
  scene.data.eq_active[eq] = 0
  assert scene._filter_active_mate_contact(
      scene.data, geom["obj_0"], geom["obj_1"]) == 0
  scene.data.eq_active[eq] = 1
  assert scene._filter_active_mate_contact(
      scene.data, geom["obj_0"], geom["obj_1"]) == 1
  # Collision against another, independently welded child is not hidden by obj_1's weld.
  assert scene._filter_active_mate_contact(
      scene.data, geom["obj_1"], next(g for g in range(scene.model.ngeom)
                                     if scene._part_owner_by_body[
                                         int(scene.model.geom_bodyid[g])] == "obj_2")) == 0


@pytest.mark.skipif(not _have("car-mini-new.json"), reason="car-mini plan not present")
def test_hang_by_wheel_breaks_the_wheel_mate():
  scene, robots = _assembled_scene(CAPS)
  broke, body_z = _hang(scene, robots, "obj_1")
  assert broke, "expected the overloaded wheel-body mate to break"
  assert any("obj_1~obj_0" in r.detail for r in broke), broke
  assert body_z < 0.35, f"the car should have fallen off the wheel, body z={body_z:.3f}"


@pytest.mark.skipif(not _have("car-mini-new.json"), reason="car-mini plan not present")
def test_hang_by_body_holds_same_caps():
  scene, robots = _assembled_scene(CAPS)
  broke, body_z = _hang(scene, robots, "obj_0")
  assert not broke, f"holding by the body must spread the load: {broke}"
  assert body_z > 0.4, f"the whole car should have lifted, body z={body_z:.3f}"


@pytest.mark.skipif(not _have("car-mini-new.json"), reason="car-mini plan not present")
def test_caps_none_is_the_historical_infinite_weld():
  scene, robots = _assembled_scene(None)
  broke, body_z = _hang(scene, robots, "obj_1")
  assert not broke and body_z > 0.4


def test_mate_caps_cli_mapping():
  class _A:
    mate_force_cap = None
    mate_torque_cap = None
  assert S._mate_caps_from_args(_A) is None
  _A.mate_force_cap = 40.0
  assert S._mate_caps_from_args(_A) == (40.0, float("inf"))
  _A.mate_force_cap, _A.mate_torque_cap = None, 0.4
  assert S._mate_caps_from_args(_A) == (float("inf"), 0.4)
