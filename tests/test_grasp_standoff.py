"""--grasp-standoff: how far the tool centre stands off the face it grasps.

The knob exists because the Robotiq 2F-85 is a four-bar linkage: closing the jaws swings the
inner finger ~13mm further along the approach axis, so a fingertip that clears the table with
the jaws OPEN can still scrape it once closed. Raising the standoff lifts the whole grasp.

What is pinned here: the DEFAULT is exactly today's behaviour (0.01, so old plans are
untouched); a raised standoff moves every pose outward along its own face normal by exactly
the difference; the grasp-DB content hash follows the standoff (grasp ids index that DB, so a
silent change would reinterpret every saved id); and a replay installs the standoff the plan
was solved at.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import problem as P
from config import RunConfig, saved_cfg

CUBE = ("box", (0.05, 0.05, 0.05, 0.005))                  # 50mm cube -> 25mm half extent
MODE = {"seed": 3, "candidates": 64, "keep": 8}            # small sampler, as in test_grasp_sampler
DEFAULT = 0.01
RAISED = 0.018                                             # the bench-leg value (see the flag help)

PLAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "experiments", "real-stool-husky-current", "plan.json")


@pytest.fixture(autouse=True)
def _restore_standoff():
  """! The standoff is a module GLOBAL. Left raised it would silently reinterpret every grasp
  id in every test module that runs after this one (the FRAGILITY 1.10 stale-global hazard --
  same reason tests/test_layout.py snapshots its scene globals)."""
  before = P._GRASP_STANDOFF
  yield
  P.set_grasp_standoff(before)
  P.set_sampled_grasps(None)
  P.reset_grasp_pose_cache()


def test_default_is_todays_behaviour():
  """The shipped default puts the tool centre 10mm off the face, exactly as before the flag."""
  assert P._GRASP_STANDOFF == DEFAULT
  assert RunConfig.default_for("grasp_standoff") == DEFAULT
  P.set_sampled_grasps(None)
  P.reset_grasp_pose_cache()
  poses = P.part_grasp_poses(CUBE)
  rolls = len(poses) // len(P._CUBE_FACE_NORMALS)
  for i, pose in enumerate(poses):
    normal = P._CUBE_FACE_NORMALS[i // rolls]
    np.testing.assert_allclose(pose[:3], (0.025 + DEFAULT) * normal, atol=1e-12)


def test_raising_it_moves_every_pose_out_along_its_own_face_normal():
  """+8mm of standoff = +8mm of position along the approach direction, and nothing else."""
  P.set_sampled_grasps(None)
  P.reset_grasp_pose_cache()
  base = [np.array(p) for p in P.part_grasp_poses(CUBE)]
  P.set_grasp_standoff(RAISED)
  raised = P.part_grasp_poses(CUBE)

  assert len(raised) == len(base)
  rolls = len(base) // len(P._CUBE_FACE_NORMALS)
  for i, (a, b) in enumerate(zip(base, raised)):
    normal = P._CUBE_FACE_NORMALS[i // rolls]
    np.testing.assert_allclose(b[:3] - a[:3], (RAISED - DEFAULT) * normal, atol=1e-12)
    np.testing.assert_array_equal(b[3:], a[3:])            # orientation is untouched


def test_sampled_path_is_rebound_not_stale_cached():
  """The mesh-sampled DB is the second grasp path; it must not keep serving the old 10mm rows.

  ! Unlike the enumerator, a sampled DB does not merely translate: the standoff is part of the
  ! key that seeds the sampler's private RNG, so a different standoff draws a different
  ! candidate cloud and REBINDS every grasp id. That is exactly why the value is saved with the
  ! plan. What is checked here is that the rows really are recomputed, plus that the
  ! conventional prefix the sampler shares with the enumerator honours the new standoff.
  """
  P.set_sampled_grasps(MODE)
  P.reset_grasp_pose_cache()
  base = [np.array(p) for p in P.part_grasp_poses(CUBE)]
  P.set_grasp_standoff(RAISED)
  raised = [np.array(p) for p in P.part_grasp_poses(CUBE)]

  assert len(raised) == len(base) == MODE["keep"]
  assert any(not np.allclose(a, b) for a, b in zip(base, raised)), \
      "the sampled DB was served from the cache built at the old standoff"

  # The conventional prefix rows are the enumerator's, and those DO simply move outward.
  rows_lo = P._canonical_sample_rows(CUBE, DEFAULT)
  rows_hi = P._canonical_sample_rows(CUBE, RAISED)
  for (pose_lo, *_), (pose_hi, *_) in zip(rows_lo, rows_hi):
    normal = pose_lo[:3] / np.linalg.norm(pose_lo[:3])
    np.testing.assert_allclose(pose_hi[:3] - pose_lo[:3],
                               (RAISED - DEFAULT) * normal, atol=1e-12)


def test_db_signature_follows_the_standoff():
  """Grasp ids index this DB, so its content hash must change when the poses move."""
  from problem import make_assembly

  asm = make_assembly("cross", 3)
  P.set_sampled_grasps(MODE)
  P.reset_grasp_pose_cache()
  at_default = P.sampled_grasp_db_signature(asm)
  P.set_grasp_standoff(RAISED)
  at_raised = P.sampled_grasp_db_signature(asm)
  assert at_default and at_raised and at_default != at_raised


def test_a_non_positive_standoff_is_refused():
  """0 would put the tool centre ON the face and a negative one inside the part."""
  for bad in (0.0, -0.005):
    with pytest.raises(ValueError):
      P.set_grasp_standoff(bad)


def test_a_plan_saved_before_the_flag_replays_at_the_old_default():
  assert saved_cfg({}, "grasp_standoff") == DEFAULT
  assert saved_cfg({"grasp_standoff": RAISED}, "grasp_standoff") == RAISED


@pytest.mark.skipif(not os.path.exists(PLAN),
                    reason="the real-stool husky plan is not in this checkout")
def test_replay_installs_the_standoff_the_plan_was_solved_at():
  """--load-plan, sim.py and export_open_loop.py all rebuild the DB through build_scene."""
  import importlib

  import replay as R
  from plan_io import read_plan

  # ! build_scene installs the whole scene configuration as module globals; snapshot and put
  # ! them back, exactly as tests/test_layout.py does.
  modules = ("problem", "solver", "env", "wrench")
  before = {n: dict(vars(importlib.import_module(n))) for n in modules}
  try:
    cfg, _ = read_plan(PLAN)
    R.build_scene(cfg)
    assert P._GRASP_STANDOFF == saved_cfg(cfg, "grasp_standoff")
  except ValueError as exc:                                # a fixture older than today's DB
    pytest.skip(f"the plan fixture predates the current scene/grasp code: {exc}")
  finally:
    for name, values in before.items():
      mod = importlib.import_module(name)
      for key, value in values.items():
        setattr(mod, key, value)
