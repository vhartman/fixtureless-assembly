#!/usr/bin/env python3
"""Interactive MuJoCo audit viewer for imported Fabrica meshes.

This deliberately does no planning and loads no saved plan.  It puts the exact renderer
meshes and the collision decomposition in the same bodies so their alignment and authored
assembly offsets can be inspected directly.

Keys:
  I  current pipeline scatter pose
  S  independently stable loose-part poses (computed from each exact source mesh)
  A  authored assembled pose
  P  pause/resume physics (paused initially)
  R  reset the current pose
  2  toggle collision geometry
  3  toggle exact renderer meshes
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import mujoco
import numpy as np

import problem as P
import sim


MODELS = {
    "car": 5,
    "beam": 5,
    "cooling-manifold": 7,
    "duct": 8,
    "gamepad": 6,
    "plumbers-block": 5,
    "circular-stool": 9,
}


def _pipeline_scatter(assembly, part_float):
  C = P.make_world()
  P.add_disassembled_obj(C, assembly, spacing=1.0 / 7.0, center_scatter=True,
                         float_height=part_float)
  return C, {p: np.asarray(C.getFrame(p).getPose(), dtype=float)
             for p in assembly.objects}


def _stable_scatter(assembly, poses, table_top, part_float):
  out = {}
  for p, pose in poses.items():
    rests = P.stable_rest_poses(assembly.grasp_shape(p), min_prob=0.0)
    q = np.asarray(rests[0][0] if rests else pose[3:], dtype=float)
    R = P._quat_to_R(q)
    xyz = np.asarray(pose[:3], dtype=float).copy()
    xyz[2] = table_top + P._shape_half_z(assembly.part_shape(p), R) + part_float
    out[p] = np.r_[xyz, q]
  return out


def _assembled(assembly, table_top, part_float):
  group = set(assembly.objects)
  q = np.array([1.0, 0.0, 0.0, 0.0])
  z = table_top + P._group_rest_height(
      assembly, group, assembly.base, P._IDENTITY_FACE) + part_float
  root = np.array([0.0, 0.0, z])
  return {p: np.r_[root + assembly.position_in_base(p), q]
          for p in assembly.objects}


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("model", choices=sorted(MODELS))
  ap.add_argument("--parts", type=int, default=None,
                  help="prefix part count (default: complete imported assembly)")
  ap.add_argument("--part-float", type=float, default=0.002)
  ap.add_argument("--physics", action="store_true", help="start with physics running")
  ap.add_argument("--pose", choices=["initial", "stable", "assembled"], default="initial")
  args = ap.parse_args()

  n = MODELS[args.model] if args.parts is None else args.parts
  assembly = P.make_assembly("fabrica-" + args.model, n)
  C, initial = _pipeline_scatter(assembly, args.part_float)
  table = sim.planner_table_geometry(C)
  stable = _stable_scatter(assembly, initial, table[5], args.part_float)
  assembled = _assembled(assembly, table[5], args.part_float)
  poses = {"initial": initial, "stable": stable, "assembled": assembled}

  xml = sim.build_mjcf(
      assembly, SimpleNamespace(poses=initial), [], planner_table_top=table[5],
      table_geometry=table)
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  manifest_path = Path(P.HERE) / "part_meshes" / "fabrica" / assembly.model / "manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest_id = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
  print(f"[fabrica-view] manifest={manifest_id} path={manifest_path}")
  expected = {}
  for p in assembly.objects:
    source = Path(assembly.visual_meshes[p]).stem
    collision = manifest["parts"][source]["collision"]
    expected[p] = int(collision["hull_count"])
    print(f"[fabrica-view] {p} <- {source}.obj: {collision['method']}, "
          f"{collision['hull_count']} hulls, asset={collision['asset_sha256'][:12]}")
  actual = {}
  for geom in range(model.ngeom):
    if int(model.geom_group[geom]) != sim.GROUP_COLL:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
    part = name.split("__", 1)[0]
    if part in expected:
      actual[part] = actual.get(part, 0) + 1
  print(f"[fabrica-view] compiled group-2 collision geoms={sum(actual.values())}: {actual}")
  if actual != expected:
    raise RuntimeError(f"compiled collision counts do not match manifest: {actual} != {expected}")
  current = ["initial"]
  playing = [bool(args.physics)]

  def apply(name):
    current[0] = name
    for p, pose in poses[name].items():
      body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p)
      joint = int(model.body_jntadr[body])
      adr = int(model.jnt_qposadr[joint])
      data.qpos[adr:adr + 7] = pose
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"[fabrica-view] pose={name}; physics={'running' if playing[0] else 'paused'}")

  def key_cb(key):
    if key in (ord("I"), ord("i")):
      apply("initial")
    elif key in (ord("S"), ord("s")):
      apply("stable")
    elif key in (ord("A"), ord("a")):
      apply("assembled")
    elif key in (ord("P"), ord("p")):
      playing[0] = not playing[0]
      print(f"[fabrica-view] physics={'running' if playing[0] else 'paused'}")
    elif key in (ord("R"), ord("r")):
      apply(current[0])

  from mujoco import viewer
  print(__doc__)
  apply(args.pose)
  with viewer.launch_passive(model, data, key_callback=key_cb) as handle:
    sim._default_geomgroups(handle, True)
    while handle.is_running():
      if playing[0]:
        mujoco.mj_step(model, data)
      handle.sync()
      time.sleep(max(0.0, float(model.opt.timestep)))


if __name__ == "__main__":
  main()
