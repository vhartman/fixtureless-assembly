"""A fixed motion export consumes selected grasps, not the search's candidate database."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import plan_io
import problem


def _certificate(plan):
  pose = [0., 0., 0., 1., 0., 0., 0.]
  before = {
      "parent": {"obj_0": ["table"]}, "grasp": {},
      "place": {"obj_0": ["slot", 0, 0]},
      "supports": {"obj_0": [["table"]]}, "poses": {"obj_0": pose}}
  after = {
      "parent": {"obj_0": ["gripper", "a1"]}, "grasp": {"obj_0": 7},
      "place": {}, "supports": {"obj_0": [["grasp", "a1", "obj_0", 7]]},
      "poses": {"obj_0": pose}}
  body = {
      "version": 1, "plan_sha256": plan_io._motion_plan_digest(plan),
      "grasp_db_sha256": "db", "states": [before, after],
      "selected_grasps": [["obj_0", 7, pose]]}
  raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
  body["sha256"] = hashlib.sha256(raw).hexdigest()
  return body


def test_motion_data_roundtrip_and_selected_lookup(tmp_path):
  plan = [("pick", "a1", "obj_0", 7)]
  path = tmp_path / "plan.json"
  plan_io.write_plan(path, {"grasp_sampler_db_hash": "db"}, plan,
                     motion_data=_certificate(plan))
  body = plan_io.read_motion_data(path, plan, grasp_db_hash="db")
  states = plan_io.motion_data_states(body, SimpleNamespace(objects=("obj_0",)))
  assert len(states) == 2
  assert states[1].grasp["obj_0"] == 7
  assert ("grasp", "a1", "obj_0", 7) in states[1].supports["obj_0"]
  problem.set_frozen_plan_grasps(plan_io.motion_data_grasps(body))
  try:
    np.testing.assert_array_equal(problem.grasp_pose(None, "obj_0", 7),
                                  [0, 0, 0, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="lacks selected grasp"):
      problem.grasp_pose(None, "obj_0", 8)
  finally:
    problem.set_frozen_plan_grasps(None)


def test_motion_data_rejects_plan_or_payload_drift(tmp_path):
  plan = [("pick", "a1", "obj_0", 7)]
  path = tmp_path / "plan.json"
  plan_io.write_plan(path, {}, plan, motion_data=_certificate(plan))
  with pytest.raises(ValueError, match="action list"):
    plan_io.read_motion_data(path, [("pick", "a1", "obj_0", 8)],
                             grasp_db_hash="db")
  with pytest.raises(ValueError, match="provenance"):
    plan_io.read_motion_data(path, plan, grasp_db_hash="different")
  payload = json.loads(path.read_text())
  payload["motion_data"]["selected_grasps"][0][2][0] = 0.01
  path.write_text(json.dumps(payload))
  with pytest.raises(ValueError, match="checksum"):
    plan_io.read_motion_data(path, plan, grasp_db_hash="db")


def test_old_plan_without_motion_data_uses_legacy_path(tmp_path):
  plan = [("pick", "a1", "obj_0", 7)]
  path = tmp_path / "plan.json"
  plan_io.write_plan(path, {}, plan)
  assert plan_io.read_motion_data(path, plan, grasp_db_hash=None) is None
