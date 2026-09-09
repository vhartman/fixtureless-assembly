"""A measured layout must reproduce a scene exactly, and move it when it says so."""
import copy
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import replay as R
from config import RunConfig
from export_layout import export_layout
from plan_io import LAYOUT_SCHEMA, load_layout_file, read_plan
from problem import add_disassembled_obj, make_assembly

PLAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "experiments", "real-stool-husky-current", "plan.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(PLAN), reason="the real-stool husky plan is not in this checkout")

# ! build_scene installs the whole scene configuration as module GLOBALS (the grasp sampler
# ! settings, roll count, finger proxy, ...). They stay installed for the rest of the pytest
# ! process, and a later test module that builds its own scene then reads THIS scene's
# ! settings -- the stale-global hazard of FRAGILITY 1.10. Left unhandled, this file made
# ! tests/test_mesh_parts.py fail purely because it runs after it. So snapshot every private
# ! module global these scenes touch and put them back afterwards.
_STATEFUL_MODULES = ("problem", "solver", "env")


def _snapshot():
  import importlib
  snap = {}
  for name in _STATEFUL_MODULES:
    mod = importlib.import_module(name)
    snap[name] = {k: (v.copy() if isinstance(v, dict) else v)
                  for k, v in vars(mod).items()
                  if k.startswith("_")
                  and isinstance(v, (bool, int, float, str, tuple, frozenset, dict, type(None)))}
  return snap


@pytest.fixture(autouse=True, scope="module")
def _restore_scene_globals():
  before = _snapshot()
  yield
  import importlib
  for name, values in before.items():
    mod = importlib.import_module(name)
    for key, value in values.items():
      setattr(mod, key, value)


def _scene_from_layout(layout):
  """Rebuild the saved plan's scene with the parts placed by `layout`."""
  cfg, _ = read_plan(PLAN)
  C, assembly = R.build_scene(cfg)
  # build_scene already scattered the parts; a second pass would re-add the frames, so
  # rebuild from a bare env instead.
  from env import make_husky_duo_env
  C = make_husky_duo_env()
  assembly = make_assembly(cfg["assembly"], cfg["parts"])
  add_disassembled_obj(C, assembly, float_height=cfg["part_float"], layout=layout)
  return C, assembly


def _poses(C, assembly):
  return {n: (np.asarray(C.getFrame(n).getPosition(), dtype=float),
              np.asarray(C.getFrame(n).getQuaternion(), dtype=float))
          for n in assembly.objects}


def _table(C):
  f = C.getFrame("table")
  return (np.asarray(f.getPosition(), dtype=float),
          np.asarray(f.getSize(), dtype=float))


def test_exported_layout_rebuilds_the_identical_scene():
  """Export -> rebuild is the identity: no part and not the table may move by a micron."""
  cfg, _ = read_plan(PLAN)
  C_ref, assembly = R.build_scene(cfg)
  ref_poses, ref_table = _poses(C_ref, assembly), _table(C_ref)

  layout = export_layout(PLAN)
  assert layout["schema"] == LAYOUT_SCHEMA
  assert set(layout["parts"]) == set(assembly.objects)

  C, assembly2 = _scene_from_layout(layout)
  got_poses, got_table = _poses(C, assembly2), _table(C)

  np.testing.assert_allclose(got_table[0], ref_table[0], atol=1e-9)
  np.testing.assert_allclose(got_table[1], ref_table[1], atol=1e-9)
  for name in assembly.objects:
    np.testing.assert_allclose(got_poses[name][0], ref_poses[name][0], atol=1e-9,
                               err_msg=f"{name} position moved")
    # A quaternion and its negation are the same rotation, so compare up to sign.
    q_ref, q_got = ref_poses[name][1], got_poses[name][1]
    if np.dot(q_ref, q_got) < 0:
      q_got = -q_got
    np.testing.assert_allclose(q_got, q_ref, atol=1e-9, err_msg=f"{name} orientation moved")


def test_layout_edits_move_the_parts_and_the_table():
  """A shifted/turned/raised layout puts the scene exactly where it asks."""
  layout = export_layout(PLAN)
  moved = copy.deepcopy(layout)
  shift, dyaw, dz = np.array([0.05, -0.02]), 0.3, 0.01
  for entry in moved["parts"].values():
    entry["xy"] = [entry["xy"][0] + shift[0], entry["xy"][1] + shift[1]]
    entry["yaw"] = entry["yaw"] + dyaw
  moved["table"]["top_z"] += dz

  C_ref, assembly = _scene_from_layout(layout)
  C, _ = _scene_from_layout(moved)
  ref, got = _poses(C_ref, assembly), _poses(C, assembly)

  # The table top rose, so every part rose with it: it rests on the surface, it does not
  # keep an absolute height.
  np.testing.assert_allclose(_table(C)[0][2], _table(C_ref)[0][2] + dz, atol=1e-9)
  for name in assembly.objects:
    np.testing.assert_allclose(got[name][0][:2], ref[name][0][:2] + shift, atol=1e-9)
    np.testing.assert_allclose(got[name][0][2], ref[name][0][2] + dz, atol=1e-9)

  # The measured yaw composes onto the scatter orientation, so the turn shows up as exactly
  # dyaw about world z relative to the reference pose.
  from problem import _quat_to_R
  for name in assembly.objects:
    R_rel = _quat_to_R(got[name][1]) @ _quat_to_R(ref[name][1]).T
    assert np.isclose(np.arctan2(R_rel[1, 0], R_rel[0, 0]), dyaw, atol=1e-9), name


def test_missing_part_is_refused_rather_than_silently_scattered():
  layout = export_layout(PLAN)
  layout["parts"].pop("obj_2")
  with pytest.raises(KeyError, match="obj_2"):
    _scene_from_layout(layout)


def test_a_plan_carrying_a_layout_is_still_a_valid_config_file(tmp_path):
  """save_plan embeds the layout; RunConfig must ignore that key, not reject the file."""
  cfg, plan = read_plan(PLAN)
  cfg = dict(cfg)
  cfg["layout"] = export_layout(PLAN)
  path = tmp_path / "plan_with_layout.json"
  path.write_text(json.dumps({"config": cfg, "plan": plan}))
  loaded = RunConfig.from_file(str(path))
  assert loaded.assembly == "real-stool"


def test_load_layout_file_refuses_a_foreign_file(tmp_path):
  path = tmp_path / "not_a_layout.json"
  path.write_text(json.dumps({"schema": "something-else", "table": {}, "parts": {}}))
  with pytest.raises(ValueError, match="schema"):
    load_layout_file(str(path))
