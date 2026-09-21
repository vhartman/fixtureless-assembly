#!/usr/bin/env python3
"""Export a recorded MuJoCo NPZ for the static browser trajectory viewer.

The output matches the lightweight bundle used by the MRMG web viewer:

    <out>/data.json   scene, timing, action, and playback metadata
    <out>/scene.bin   local mesh vertices, triangles, and colors
    <out>/poses.bin   sampled world poses of moving geometry instances

No physics or planning runs in the browser. This script replays the recorded qpos values through
MuJoCo offline, exports only visible geometry groups, and downsamples poses to a web-sized rate.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import json
from pathlib import Path

import mujoco
import numpy as np

from sim import (_recording_frame, _relocate_recording_mesh_assets,
                 _scene_export_geometry, GROUP_ENV, GROUP_PART, GROUP_VISUAL)


VISIBLE_GROUPS = {int(GROUP_PART), int(GROUP_ENV), int(GROUP_VISUAL)}


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("recording", type=Path, help="recording produced by sim.py --record")
  parser.add_argument("--out", type=Path, required=True, help="output bundle directory")
  parser.add_argument("--fps", type=float, default=20.0,
                      help="target browser playback rate (default: 20)")
  parser.add_argument("--label", default=None, help="display label (default: NPZ stem)")
  return parser


def _load_json_scalar(z, key, default):
  if key not in z.files:
    return default
  try:
    return json.loads(str(z[key]))
  except (TypeError, ValueError, json.JSONDecodeError):
    return default


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
  quat = np.empty(4, dtype=np.float64)
  mujoco.mju_mat2Quat(quat, np.ascontiguousarray(matrix, dtype=np.float64).reshape(9))
  # q and -q encode the same rotation. A stable hemisphere prevents false dynamic-frame tests.
  if quat[0] < 0:
    quat *= -1
  return quat


def _action_record(raw, index: int) -> dict:
  try:
    action = ast.literal_eval(raw)
  except (ValueError, SyntaxError):
    action = ("other", str(raw))
  if not isinstance(action, tuple):
    action = tuple(action) if isinstance(action, list) else ("other", str(action))
  kind = str(action[0]) if action else "other"
  args = [str(x) for x in action[1:]]
  if kind == "pick" and len(args) >= 2:
    name = f"{args[0]} picks {args[1]}"
  elif kind == "assemble" and len(args) >= 3:
    name = f"{args[0]} assembles {args[1]} to {args[2]}"
  elif kind == "handover" and len(args) >= 2:
    name = f"handover {args[1]} to {args[0]}"
  elif kind == "release" and len(args) >= 2:
    name = f"{args[0]} releases {args[1]}"
  elif kind == "place" and len(args) >= 2:
    name = f"{args[0]} places {args[1]}"
  else:
    name = " ".join([kind, *args])
  return {"id": index, "name": name, "type": kind, "raw": repr(action)}


def _sample_indices(nframes: int, sample_period: float, fps: float,
                    marks: list, gripper_events: list) -> list[int]:
  if fps <= 0:
    raise ValueError("--fps must be positive")
  stride = max(1, int(round(1.0 / max(sample_period * fps, 1e-9))))
  indices = set(range(0, nframes, stride))
  indices.update(int(frame) for frame, _ in marks if 0 <= int(frame) < nframes)
  indices.update(int(event.get("frame", -1)) for event in gripper_events
                 if 0 <= int(event.get("frame", -1)) < nframes)
  indices.add(nframes - 1)
  return sorted(indices)


def _geometry_bundle(model, data):
  try:
    import trimesh
  except ImportError as exc:
    raise RuntimeError("web replay export requires trimesh") from exc

  frames = []
  unique = []
  unique_by_key = {}

  for geom_id in range(model.ngeom):
    group = int(model.geom_group[geom_id])
    rgba = np.asarray(model.geom_rgba[geom_id], dtype=float)
    if group not in VISIBLE_GROUPS or rgba[3] <= 1e-4:
      continue
    built = _scene_export_geometry(model, geom_id, trimesh)
    if built is None:
      continue
    mesh, local, shape_key = built
    if not np.allclose(local, np.eye(4)):
      mesh.apply_transform(local)
    # Imported CAD is often STL-style (three private vertices per triangle). Welding exact
    # local-space duplicates cuts browser payloads by several times without changing geometry.
    mesh.merge_vertices(digits_vertex=8, merge_tex=True, merge_norm=True)
    mesh.remove_unreferenced_vertices()
    color = tuple(int(x) for x in np.clip(np.round(rgba * 255.0), 0, 255))
    key = (shape_key, color)
    geometry_index = unique_by_key.get(key)
    if geometry_index is None:
      vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
      triangles = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
      colors = np.empty((len(vertices), 4), dtype=np.uint8)
      colors[:] = color
      geometry_index = len(unique)
      unique_by_key[key] = geometry_index
      unique.append({"vertices": vertices, "triangles": triangles, "colors": colors})

    body_id = int(model.geom_bodyid[geom_id])
    frames.append({
        "name": (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                 or f"geom_{geom_id}"),
        "body": (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                 or f"body_{body_id}"),
        "geomId": geom_id,
        "group": group,
        "geometryIndex": geometry_index,
        "vCount": 0,
        "tCount": 0,
        "vOffset": 0,
        "tOffset": 0,
        "cOffset": 0,
        "nOffset": -1,
        "flat": False,
        "color": list(color),
    })

  if not frames:
    raise ValueError("recording contains no visible geometry")

  v_base = 0
  t_base = sum(rec["vertices"].nbytes for rec in unique)
  c_base = t_base + sum(rec["triangles"].nbytes for rec in unique)
  v_off = t_off = c_off = 0
  for rec in unique:
    rec["vOffset"] = v_base + v_off
    rec["tOffset"] = t_base + t_off
    rec["cOffset"] = c_base + c_off
    v_off += rec["vertices"].nbytes
    t_off += rec["triangles"].nbytes
    c_off += rec["colors"].nbytes

  for frame in frames:
    rec = unique[frame.pop("geometryIndex")]
    frame.update({
        "vCount": int(len(rec["vertices"])),
        "tCount": int(len(rec["triangles"])),
        "vOffset": int(rec["vOffset"]),
        "tOffset": int(rec["tOffset"]),
        "cOffset": int(rec["cOffset"]),
    })

  scene_bin = b"".join(rec["vertices"].tobytes() for rec in unique)
  scene_bin += b"".join(rec["triangles"].tobytes() for rec in unique)
  scene_bin += b"".join(rec["colors"].tobytes() for rec in unique)
  return frames, scene_bin


def _pose_bundle(model, data, frames, qpos, mpos, mquat, indices):
  rows = np.empty((len(indices), len(frames), 7), dtype=np.float32)
  for step, source_frame in enumerate(indices):
    _recording_frame(model, data, qpos, mpos, mquat, source_frame)
    for frame_index, frame in enumerate(frames):
      geom_id = int(frame["geomId"])
      rows[step, frame_index, :3] = data.geom_xpos[geom_id]
      rows[step, frame_index, 3:] = _quat_from_matrix(data.geom_xmat[geom_id])

  pos_dev = np.max(np.abs(rows[:, :, :3] - rows[0:1, :, :3]), axis=(0, 2))
  # Quaternion signs are stabilized above; abs() also makes the test robust near pi rotations.
  quat_dot = np.abs(np.sum(rows[:, :, 3:] * rows[0:1, :, 3:], axis=2))
  rot_dev = np.max(1.0 - np.clip(quat_dot, 0.0, 1.0), axis=0)
  dynamic = (pos_dev > 1e-6) | (rot_dev > 1e-8)
  dyn_frames = np.flatnonzero(dynamic).tolist()
  static = {str(i): rows[0, i].tolist() for i in np.flatnonzero(~dynamic)}
  packed = np.ascontiguousarray(rows[:, dyn_frames, :].reshape(-1), dtype=np.float32)
  return rows, dyn_frames, static, packed


def _camera_from_scene(model, data, frames, geometry_frames):
  points = []
  for frame, pose in zip(frames, geometry_frames):
    # Ignore the large floor plane when fitting the camera; robots and parts determine framing.
    if int(frame["group"]) == int(GROUP_ENV) and int(model.geom_type[frame["geomId"]]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
      continue
    geom_id = int(frame["geomId"])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    radius = max(float(np.linalg.norm(size)), 0.03)
    center = np.asarray(pose[:3], dtype=float)
    points.extend([center - radius, center + radius])
  pts = np.asarray(points, dtype=float)
  lo, hi = pts.min(axis=0), pts.max(axis=0)
  center = 0.5 * (lo + hi)
  radius = max(float(np.linalg.norm(hi - lo) / 2.0), 0.8)
  direction = np.array([1.15, -1.7, 1.1], dtype=float)
  direction /= np.linalg.norm(direction)
  target = center.copy()
  target[2] = max(0.35, target[2] * 0.8)
  return {
      "position": (target + direction * radius * 2.4).tolist(),
      "target": target.tolist(),
      "up": [0.0, 0.0, 1.0],
  }


def export(args) -> None:
  z = np.load(args.recording, allow_pickle=False)
  xml = _relocate_recording_mesh_assets(str(z["xml"]))
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  qpos = np.asarray(z["qpos"])
  if len(qpos) == 0:
    raise ValueError(f"recording has no frames: {args.recording}")
  mpos = np.asarray(z["mpos"]) if "mpos" in z.files else None
  mquat = np.asarray(z["mquat"]) if "mquat" in z.files else None
  metadata = _load_json_scalar(z, "metadata", {})
  marks = _load_json_scalar(z, "marks", [])
  gripper_events = _load_json_scalar(z, "gripper_events", [])
  results = _load_json_scalar(z, "results", [])
  robots = [str(x) for x in z["robots"].tolist()] if "robots" in z.files else []
  parts = [str(x) for x in z["parts"].tolist()] if "parts" in z.files else []
  controller_modes = ([str(x) for x in z["controller_mode"].tolist()]
                      if "controller_mode" in z.files else ["recorded"] * len(qpos))

  dt = float(metadata.get("dt", model.opt.timestep))
  record_every = int(metadata.get("record_every", 1))
  sample_period = dt * record_every
  indices = _sample_indices(len(qpos), sample_period, args.fps, marks, gripper_events)

  _recording_frame(model, data, qpos, mpos, mquat, indices[0])
  frames, scene_bin = _geometry_bundle(model, data)
  rows, dyn_frames, static, poses = _pose_bundle(
      model, data, frames, qpos, mpos, mquat, indices)

  actions = [_action_record(raw, i) for i, (_, raw) in enumerate(marks)]
  complete_id = len(actions)
  task_info = actions + [{"id": complete_id, "name": "execution complete", "type": "terminal"}]
  mark_frames = [int(frame) for frame, _ in marks]
  action_ids = []
  transition = []
  for source_frame in indices:
    action_id = bisect.bisect_right(mark_frames, source_frame)
    action_ids.append(min(action_id, complete_id))
    transition.append(source_frame in mark_frames)

  selected_modes = [controller_modes[i] for i in indices]
  selected_times = [float(i * sample_period) for i in indices]
  frame_to_step = {source: step for step, source in enumerate(indices)}
  mapped_gripper_events = []
  for event in gripper_events:
    source = int(event.get("frame", 0))
    nearest = min(indices, key=lambda value: abs(value - source))
    mapped = dict(event)
    mapped["sourceFrame"] = source
    mapped["step"] = frame_to_step[nearest]
    mapped_gripper_events.append(mapped)

  label = args.label or args.recording.stem.replace("-", " ").title()
  path_label = "recorded execution"
  path_record = {
      "label": path_label,
      "steps": len(indices),
      "stride": None,
      "dynFrames": dyn_frames,
      "static": static,
      "poseOffset": 0,
      "poseCount": int(poses.size),
      "actionIds": action_ids,
      "transition": transition,
      "controllerModes": selected_modes,
      "times": selected_times,
      "sourceFrames": indices,
  }
  manifest = {
      "format": "assembly-web-replay-v1",
      "env": label,
      "source": str(args.recording),
      "meta": {
          "num_robots": len(robots),
          "source_frames": len(qpos),
          "exported_frames": len(indices),
          "sample_period_s": sample_period,
          "duration_s": float((len(qpos) - 1) * sample_period),
          "target_fps": float(args.fps),
          "success": bool(z["success"]) if "success" in z.files else None,
      },
      "robots": robots,
      "parts": parts,
      "frames": frames,
      "taskInfo": task_info,
      "gripperEvents": mapped_gripper_events,
      "results": results,
      "paths": [path_record],
      "pathCosts": {path_label: float((len(qpos) - 1) * sample_period)},
      "camera": _camera_from_scene(model, data, frames, rows[0]),
  }

  args.out.mkdir(parents=True, exist_ok=True)
  (args.out / "data.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
  (args.out / "scene.bin").write_bytes(scene_bin)
  (args.out / "poses.bin").write_bytes(poses.tobytes())
  print(f"[web-export] {len(qpos)} source frames -> {len(indices)} browser frames "
        f"({sample_period:.4f}s source period, target {args.fps:g}fps)")
  print(f"[web-export] {len(frames)} visible geometry instances, "
        f"{len(dyn_frames)} dynamic")
  print(f"[web-export] scene.bin {len(scene_bin) / 1e6:.1f}MB, "
        f"poses.bin {poses.nbytes / 1e6:.1f}MB")
  print(f"[web-export] wrote {args.out}")


if __name__ == "__main__":
  export(build_parser().parse_args())
