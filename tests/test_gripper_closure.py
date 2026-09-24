"""Regression tests for the disabled-joint virtual gripper-close certificate."""

import numpy as np
import robotic as ry

from assembly import get_robots
from env import make_openarm_duo_env
from gripper_closure import VERSION, certify, closure_sweep_configs, finger_sweep_radius_bound
from problem import add_disassembled_obj, get_initial_state, grasp_pose, make_assembly
from solver import _search_context, isolated_pick_state


def _robotiq_scene(with_blocker=False):
  C = ry.Config()
  C.addFile("src/rai/ur10/ur10_two_finger_floating.g", namePrefix="x_")
  centre = np.asarray(C.getFrame("x_ur_gripper_center").getPosition(), dtype=float)
  # A 20mm-wide object is clear of the fixed-open fingers but demands a nonzero closure.
  C.addFrame("obj_0").setShape(ry.ST.box, [.02, .02, .02, .0]).setPosition(centre).setContact(1)
  if with_blocker:
    # Clear at open jaw (inner finger surface y~39mm), but in the right finger's inward
    # sweep.  It models exactly the handover defect: both endpoint proxies can look fine
    # while a receiving finger has to pass through a neighbouring gripper.
    C.addFrame("other_gripper").setShape(ry.ST.sphere, [.005]).setPosition(
        centre + np.array([0.0, .03, 0.0])).setContact(1)
  C.setJointState(C.getJointState())
  return C


def test_robotiq_virtual_close_finds_object_dependent_aperture():
  out = certify(_robotiq_scene(), "x_", "obj_0")
  assert out.feasible and out.model == "robotiq"
  assert 0.0 < out.aperture < 0.042
  assert VERSION >= 2


def test_finger_sweep_bound_contains_both_gripper_models():
  """A table broad-phase reject must cover every finger pose in the exact sweep."""
  import gripper_closure as G
  scenes = [(_robotiq_scene(), "x_")]
  openarm = ry.Config()
  openarm.addFile("src/rai/openarm/openarm_pinch_floating.g", namePrefix="x_")
  centre = np.asarray(openarm.getFrame("x_gripper_center").getPosition(), dtype=float)
  openarm.addFrame("obj_0").setShape(ry.ST.box, [.03, .04, .03, .0]).setPosition(
      centre).setContact(1)
  scenes.append((openarm, "x_"))
  for C, prefix in scenes:
    radius = finger_sweep_radius_bound(C, prefix)
    assert radius is not None and radius > 0.0
    _result, frames = closure_sweep_configs(C, prefix, "obj_0", steps=5)
    for sweep, _label in frames:
      _model, groups, center_name = G._finger_groups(sweep, prefix)
      center = np.asarray(sweep.getFrame(center_name).getPosition(), dtype=float)
      for group in groups:
        for name in group:
          frame = sweep.getFrame(name)
          distance = np.linalg.norm(np.asarray(frame.getPosition(), dtype=float) - center)
          assert distance + np.sum(np.asarray(frame.getSize(), dtype=float)) <= radius


def test_aperture_cache_distinguishes_compound_offsets_and_finger_sizes():
  import gripper_closure as G
  C = _robotiq_scene()
  C.addFrame("obj_0__shape0", "obj_0").setShape(ry.ST.box, [.01, .01, .01])
  model, groups, _center = G._finger_groups(C, "x_")
  before = G._aperture_key(C, model, groups, "obj_0")
  C.getFrame("obj_0__shape0").setRelativePosition([0, .01, 0])
  shifted = G._aperture_key(C, model, groups, "obj_0")
  assert shifted != before
  finger = C.getFrame(groups[0][0])
  finger.setShape(ry.ST.box, [.01, .02, .03])
  assert G._aperture_key(C, model, groups, "obj_0") != shifted


def test_aperture_cache_distinguishes_meshes_with_same_extents():
  import gripper_closure as G
  C = _robotiq_scene()
  points = np.array([[0, 0, 0], [.02, 0, 0], [0, .02, 0], [0, 0, .02]])
  child = C.addFrame("obj_0__shape0", "obj_0").setConvexMesh(points)
  model, groups, _center = G._finger_groups(C, "x_")
  before = G._aperture_key(C, model, groups, "obj_0")
  child.setConvexMesh(np.vstack([points, [.02, .02, .02]]))
  assert G._aperture_key(C, model, groups, "obj_0") != before


def test_held_endpoint_does_not_repeat_acquisition_sweep():
  C = _robotiq_scene(with_blocker=True)
  assert not certify(C, "x_", "obj_0").feasible
  assert certify(C, "x_", "obj_0", closing=False).feasible
  center = np.asarray(C.getFrame("x_ur_gripper_center").getPosition())
  C.getFrame("other_gripper").setPosition(center + [0, .013, 0])
  assert not certify(C, "x_", "obj_0", closing=False).feasible


def test_sampler_pair_cache_shares_identical_tools_but_retains_direction():
  from solver import CollisionChecker
  from problem import State
  checker = object.__new__(CollisionChecker)
  checker._sampled_pair_verdicts = {}
  checker.closure_check = False
  checker.robot_gripper_paths = dict(a1='tool.g', a2='tool.g', a3='tool.g')
  checker._sig = lambda *args: ('scene',)
  calls = []
  checker._check = lambda *args, **kw: calls.append(1) or False
  state = State({}, {}, {}, {}, {})
  a = np.array([0, 0, .01, 1, 0, 0, 0.])
  b = np.array([0, 0, .02, 1, 0, 0, 0.])
  assert not checker.pose_pair_grasp_free(state, 'obj_0', a, b, None, 'a1', 'a2')
  assert not checker.pose_pair_grasp_free(state, 'obj_0', a, b, None, 'a2', 'a3')
  assert len(calls) == 1
  assert not checker.pose_pair_grasp_free(state, 'obj_0', b, a, None, 'a2', 'a1')
  assert len(calls) == 2


def test_aperture_cache_eviction_preserves_the_certificate(monkeypatch):
  import gripper_closure as G
  monkeypatch.setattr(G, '_APERTURE_CACHE', {})
  monkeypatch.setattr(G, '_APERTURE_CACHE_LIMIT', 1)
  C = _robotiq_scene()
  first = G.certify(C, 'x_', 'obj_0')
  C.getFrame('obj_0').setShape(ry.ST.box, [.02, .03, .02, .0])
  wider = G.certify(C, 'x_', 'obj_0')
  assert first.feasible and wider.feasible
  assert wider.aperture < first.aperture
  assert len(G._APERTURE_CACHE) == 1
  C.getFrame('obj_0').setShape(ry.ST.box, [.02, .02, .02, .0])
  assert G.certify(C, 'x_', 'obj_0') == first


def test_virtual_close_rejects_an_off_centre_bilateral_grasp():
  """A common aperture must not reach one jaw by driving the other through the part."""
  C = _robotiq_scene()
  centre = np.asarray(C.getFrame("x_ur_gripper_center").getPosition(), dtype=float)
  # A 5mm shift along the pinch axis is comfortably inside the total jaw stroke, so both
  # fingers can individually reach.  It is nevertheless not a valid rigid, symmetric grasp:
  # closing to the later contact penetrates the early side by about 10mm.
  C.getFrame("obj_0").setPosition(centre + np.array([0.0, .005, 0.0]))
  C.setJointState(C.getJointState())

  out = certify(C, "x_", "obj_0")
  assert not out.feasible
  assert out.reason == "miscentred-contact"
  assert out.detail is not None and out.detail[2] < -.005
  # The certificate exposes the kinematic projection used by the grasp sampler: moving the
  # tool centre by the measured half-difference of first-contact travel recentres the jaws.
  assert out.centering_offset is not None
  assert abs(out.centering_offset - .005) < 2e-4


def test_virtual_close_keeps_submillimetre_centring_noise_feasible():
  C = _robotiq_scene()
  centre = np.asarray(C.getFrame("x_ur_gripper_center").getPosition(), dtype=float)
  C.getFrame("obj_0").setPosition(centre + np.array([0.0, .0005, 0.0]))
  C.setJointState(C.getJointState())

  out = certify(C, "x_", "obj_0")
  assert out.feasible


def test_virtual_close_rejects_collision_only_in_sweep():
  out = certify(_robotiq_scene(with_blocker=True), "x_", "obj_0")
  assert not out.feasible
  assert out.reason == "sweep-collision"
  assert out.detail is not None and "other_gripper" in out.detail[:2]


def test_closure_visualization_samples_the_same_certificate_without_mutation():
  C = _robotiq_scene(with_blocker=True)
  before = np.asarray(C.getFrameState(), dtype=float).copy()
  result, frames = closure_sweep_configs(C, "x_", "obj_0", steps=9)
  assert not result.feasible and result.reason == "sweep-collision"
  assert len(frames) == 9
  assert any("BLOCKED" in label for _Ci, label in frames)
  assert "inward=0.00mm" in frames[0][1]
  np.testing.assert_array_equal(np.asarray(C.getFrameState(), dtype=float), before)


def test_openarm_pinch_uses_its_own_collision_proxy_calibration():
  C = ry.Config()
  C.addFile("src/rai/openarm/openarm_pinch_floating.g", namePrefix="x_")
  centre = np.asarray(C.getFrame("x_gripper_center").getPosition(), dtype=float)
  C.addFrame("obj_0").setShape(ry.ST.box, [.03, .04, .03, .0]).setPosition(centre).setContact(1)
  C.setJointState(C.getJointState())
  out = certify(C, "x_", "obj_0")
  assert out.feasible and out.model == "openarm"
  assert out.aperture is not None and out.aperture > 0.0


def test_compound_target_never_enables_shapeless_logical_parent():
  C = ry.Config()
  C.addFile("src/rai/openarm/openarm_pinch_floating.g", namePrefix="x_")
  centre = np.asarray(C.getFrame("x_gripper_center").getPosition(), dtype=float)
  C.addFrame("obj_0")                                      # logical compound parent: no shape
  C.addFrame("obj_0__shape0", "obj_0").setShape(
      ry.ST.box, [.03, .04, .03, .0]).setPosition(centre).setContact(1)
  C.setJointState(C.getJointState())
  out = certify(C, "x_", "obj_0")
  assert out.feasible and out.model == "openarm"


def test_openarm_cross_initial_pick_gate_uses_native_pinch_scene():
  """Exercise the gate on the four-OpenArm Cross layout, not only a synthetic proxy."""
  C = make_openarm_duo_env(2)
  assembly = make_assembly("cross", 7)
  C = add_disassembled_obj(C, assembly, grid=4, grid_gap=2.0)
  robots = get_robots(C)
  checker, table = _search_context(C, assembly, robots)
  checker.closure_check = True
  s0 = get_initial_state(C, assembly)

  open_clear = close_clear = 0
  for part in assembly.objects:
    for _part, gid, _pose in table[(part,)]:
      action = ("pick", robots[0], part, gid)
      if checker.pick_collision_free(s0, part, gid, assembly, robot=robots[0]):
        open_clear += 1
        close_id = checker.grasp_closure_free(s0, action, assembly)
        close_pose = checker.pose_grasp_closure_free(
            s0, part, grasp_pose(assembly, part, gid), assembly, robots[0])
        assert close_pose == close_id
        close_clear += close_id
  assert open_clear > 0
  # This specific initial layout is a non-regression smoke: no genuine initial pick is lost.
  assert close_clear == open_clear


def test_isolated_pick_state_preserves_target_and_parks_all_clutter():
  C = make_openarm_duo_env(2)
  assembly = make_assembly("cross", 3)
  C = add_disassembled_obj(C, assembly, grid=3, grid_gap=0.1)
  s0 = get_initial_state(C, assembly)
  iso = isolated_pick_state(s0, assembly, "obj_1")
  checker, _table = _search_context(C, assembly, get_robots(C))
  D = checker._scene_for(iso, assembly)

  np.testing.assert_allclose(D.getFrame("obj_1").getPose(), C.getFrame("obj_1").getPose())
  for part in ("obj_0", "obj_2"):
    assert np.linalg.norm(np.asarray(D.getFrame(part).getPosition())[:2]) > 900.0
  np.testing.assert_allclose(D.getFrame("table").getPose(), C.getFrame("table").getPose())
