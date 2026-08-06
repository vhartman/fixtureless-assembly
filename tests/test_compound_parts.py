"""Compound part shapes: ("compound", [{"shape": spec, "pos": [x,y,z]}, ...]).

ONE rigid part built from several primitives/meshes, realized as __shape{i} children.
Validated end to end on compound-demo (cube base + box-with-boss + cube cap; mixed
grippers, --grasp-sampler, --grasp-eq-weight 100 --grasp-accuracy 0.001
--mate-accuracy 0.0005): search -> witnesses -> BiRRT -> MuJoCo, 6/6. These tests pin
the fast pieces.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import problem as P

SPEC = ("compound", [
    {"shape": ("box", [0.05, 0.05, 0.03, 0.005]), "pos": [0.0, 0.0, -0.015]},
    {"shape": ("cylinder", [0.03, 0.018]), "pos": [0.0, 0.0, 0.015]},
])
ASYM = ("compound", [  # deliberately NOT symmetric about the origin
    {"shape": ("box", [0.04, 0.04, 0.02, 0.005]), "pos": [0.0, 0.0, 0.03]},
])


@pytest.fixture(autouse=True)
def _no_sampler_leak():
  yield
  P.set_sampled_grasps(None)


def test_half_extents_are_the_symmetric_cover():
  assert np.allclose(P._shape_half_extents(SPEC), [0.025, 0.025, 0.03], atol=1e-9)
  assert P._shape_half_extent_along(SPEC, np.array([0.0, 0.0, 1.0])) == pytest.approx(0.03)


def test_half_z_is_exact_per_child_even_asymmetric():
  # child centre at +0.03 with half 0.01 -> lowest point at +0.02 -> DOWNWARD extent -0.02
  assert P._shape_half_z(ASYM, np.eye(3)) == pytest.approx(-0.02)
  R_flip = P._quat_to_R(np.array([0.0, 1.0, 0.0, 0.0]))    # upside down: extent +0.04
  assert P._shape_half_z(ASYM, R_flip) == pytest.approx(0.04)


def test_support_points_union():
  pose = np.array([0.0, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0])
  pts = P._part_support_points_world(pose, SPEC)
  assert len(pts) == 8 + 16                                # box corners + cylinder rims
  assert min(x[2] for x in pts) == pytest.approx(0.0, abs=1e-9)
  assert max(x[2] for x in pts) == pytest.approx(0.06, abs=1e-9)


def test_rai_children_and_contact_toggle():
  import robotic as ry
  C = ry.Config()
  P.add_part_frame(C, "cp", SPEC)
  assert P.part_shape_frames(C, "cp") == ["cp__shape0", "cp__shape1"]
  P.set_part_contact(C, "cp", 0)                           # must not raise; toggles BOTH
  P.set_part_contact(C, "cp", 1)


def test_enumerator_refuses_compound_without_sampler():
  P.set_sampled_grasps(None)
  with pytest.raises(ValueError, match="grasp-sampler"):
    P.part_grasp_poses(SPEC)


def test_sampled_views_over_the_union():
  P.set_sampled_grasps({"seed": 0, "candidates": 200, "keep": 12})
  poses, surfs, mins = P._sampled_grasp_views(SPEC, 0.01)
  assert len(poses) == 12
  assert "flat" in surfs                                   # box faces / boss cap


def test_spec_key_distinguishes_compounds():
  other = ("compound", [
      {"shape": ("box", [0.05, 0.05, 0.03, 0.005]), "pos": [0.0, 0.0, -0.015]},
      {"shape": ("cylinder", [0.03, 0.012]), "pos": [0.0, 0.0, 0.015]},
  ])
  assert P._spec_key(SPEC) != P._spec_key(other)
  assert P._spec_key(SPEC) == P._spec_key(("compound", list(SPEC[1])))


def test_no_nested_compounds():
  bad = ("compound", [{"shape": SPEC, "pos": [0, 0, 0]}])
  with pytest.raises(ValueError, match="nested"):
    P._compound_children(bad)


def test_sim_geoms_split_mass_by_volume():
  import sim as S
  xml = S._compound_geoms_xml(SPEC, 0.8, 1.0, 3, "1 0 0 1")
  assert xml.count("<geom") == 2
  import re
  masses = [float(m) for m in re.findall(r'mass="([0-9.e-]+)"', xml)]
  assert sum(masses) == pytest.approx(1.0)
  v_box = 0.05 * 0.05 * 0.03
  v_cyl = np.pi * 0.018 ** 2 * 0.03
  assert masses[0] == pytest.approx(v_box / (v_box + v_cyl))
