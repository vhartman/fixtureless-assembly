"""Versioned compatibility contract for saved task-planning artifacts.

A saved grasp id is meaningful only under the collision geometry and closure semantics that
created it.  The sampled-grasp database hash catches changes which alter the retained rows, but
it cannot catch a model edit whose current finite prefix happens to stay identical.  This module
therefore fingerprints the complete contact-bearing RAI scene (including joint order/limits)
and binds that fingerprint to the virtual-close implementation and grasp database.

The contract deliberately excludes visual-only meshes and colors.  Those may be improved
without invalidating a task or motion certificate.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


VERSION = 1


def _array_bytes(value):
  a = np.asarray(value)
  if a.dtype.kind in "iu":
    return np.asarray(a, dtype="<i8").tobytes()
  return np.asarray(a, dtype="<f8").tobytes()


def collision_model_signature(C):
  """Hash contact geometry and robot coordinates, excluding visual-only shapes."""
  h = hashlib.sha256()
  h.update(b"assembly-contact-scene-v1\0")
  for name in sorted(map(str, C.getFrameNames())):
    frame = C.getFrame(name)
    info = frame.info() or {}
    if not bool(info.get("contact", 0)):
      continue
    h.update(name.encode() + b"\0")
    try:
      h.update(str(frame.getShapeType()).encode() + b"\0")
      h.update(_array_bytes(frame.getSize()))
    except Exception:
      h.update(b"shape-unavailable\0")
    # World poses intentionally join the signature: robot bases, table datum and initial
    # object placement are all collision semantics, not presentation details.
    h.update(_array_bytes(np.round(np.asarray(frame.getPose(), dtype=float), 12)))
    try:
      vertices = np.asarray(frame.getMeshPoints())
      triangles = np.asarray(frame.getMeshTriangles())
      if vertices.size:
        h.update(_array_bytes(np.round(vertices, 12)))
      if triangles.size:
        h.update(_array_bytes(triangles))
    except Exception:
      pass
  h.update(repr(tuple(map(str, C.getJointNames()))).encode())
  try:
    lo, hi = C.getJointLimits()
    h.update(_array_bytes(np.round(np.asarray(lo, dtype=float), 12)))
    h.update(_array_bytes(np.round(np.asarray(hi, dtype=float), 12)))
  except Exception:
    pass
  return h.hexdigest()


def build_artifact_contract(C, *, grasp_db_hash=None, closure_enabled=False):
  from gripper_closure import VERSION as closure_version
  body = {
      "version": VERSION,
      "collision_model_sha256": collision_model_signature(C),
      "grasp_db_sha256": grasp_db_hash,
      "closure_enabled": bool(closure_enabled),
      "closure_model_version": int(closure_version),
  }
  payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
  body["sha256"] = hashlib.sha256(payload).hexdigest()
  return body


def validate_artifact_contract(saved, actual, *, who="artifact consumer"):
  """Fail closed on a missing or mismatched modern planning contract."""
  if not saved:
    raise ValueError(
        f"{who}: saved plan predates the versioned grasp/model contract. Regenerate its "
        "task plan with the current standard profile, or use --refresh-witnesses from the "
        "original plan as an explicit one-time recertification; refusing silent legacy "
        "grasp reinterpretation.")
  if int(saved.get("version", -1)) != VERSION:
    raise ValueError(
        f"{who}: unsupported artifact-contract version {saved.get('version')!r}; "
        "regenerate the task plan with this version.")
  fields = ("collision_model_sha256", "grasp_db_sha256", "closure_enabled",
            "closure_model_version", "sha256")
  bad = [k for k in fields if saved.get(k) != actual.get(k)]
  if bad:
    detail = ", ".join(f"{k}: {saved.get(k)!r} != {actual.get(k)!r}" for k in bad)
    raise ValueError(
        f"{who}: saved task plan is incompatible with the current grasp/model contract "
        f"({detail}). Regenerate the task plan; refusing to reuse its grasp ids or witnesses.")
  return True
