import copy

import numpy as np
import pytest

from assembly_planner.artifact_contract import build_artifact_contract, validate_artifact_contract


class _Frame:
  def __init__(self, *, contact, pose, size=(1.0, 1.0, 1.0)):
    self._contact = contact
    self._pose = np.asarray(pose, dtype=float)
    self._size = np.asarray(size, dtype=float)

  def info(self):
    return {"contact": self._contact}

  def getShapeType(self):
    return "box"

  def getSize(self):
    return self._size

  def getPose(self):
    return self._pose

  def getMeshPoints(self):
    return np.empty((0, 3))

  def getMeshTriangles(self):
    return np.empty((0, 3), dtype=int)


class _Config:
  def __init__(self):
    self.frames = {
        "collision": _Frame(contact=1, pose=[0, 0, 0, 1, 0, 0, 0]),
        "visual": _Frame(contact=0, pose=[0, 0, 0, 1, 0, 0, 0]),
    }

  def getFrameNames(self):
    return list(self.frames)

  def getFrame(self, name):
    return self.frames[name]

  def getJointNames(self):
    return ["j0"]

  def getJointLimits(self):
    return np.array([-1.0]), np.array([1.0])


def test_contract_ignores_visual_edits_but_rejects_collision_edits():
  C = _Config()
  saved = build_artifact_contract(C, grasp_db_hash="db", closure_enabled=True)
  C.frames["visual"]._pose[0] = 2.0
  assert build_artifact_contract(C, grasp_db_hash="db", closure_enabled=True) == saved
  C.frames["collision"]._pose[0] = 0.01
  actual = build_artifact_contract(C, grasp_db_hash="db", closure_enabled=True)
  with pytest.raises(ValueError, match="collision_model_sha256"):
    validate_artifact_contract(saved, actual)


def test_contract_is_fail_closed_for_missing_or_changed_closure_semantics():
  C = _Config()
  actual = build_artifact_contract(C, grasp_db_hash="db", closure_enabled=True)
  with pytest.raises(ValueError, match="predates"):
    validate_artifact_contract(None, actual)
  saved = copy.deepcopy(actual)
  saved["closure_enabled"] = False
  with pytest.raises(ValueError, match="closure_enabled"):
    validate_artifact_contract(saved, actual)
