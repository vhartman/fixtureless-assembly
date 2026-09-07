#!/usr/bin/env python3
"""Check Fabrica insertion sweeps using the actual RAI and MuJoCo colliders.

This is a sampled geometry audit (default 0.5 mm), not a continuous collision
certificate. It checks the mover against every already assembled part, including
the intended mate. Negative distances below --tolerance fail the command.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

import problem as P
import sim
from view_fabrica import MODELS


def check_containment(asm):
  import trimesh
  sys.path.insert(0, str(Path(__file__).parent / "part_meshes" / "fabrica"))
  from tetra_collision import BoundaryCheck, solid_contains
  for part in asm.objects:
    mesh = trimesh.load(Path(P.HERE) / asm.visual_meshes[part], force="mesh", process=True)
    if not mesh.is_watertight:
      mesh.merge_vertices(digits_vertex=7, merge_norm=True, merge_tex=True)
      mesh.update_faces(mesh.nondegenerate_faces())
      mesh.update_faces(mesh.unique_faces())
      mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
      raise ValueError(f"{part}: exact CAD is not watertight")
    check = BoundaryCheck(mesh)
    centers = []
    crossings = 0
    source = Path(asm.visual_meshes[part]).stem
    collision = asm._manifest["parts"][source]["collision"]
    approved = collision.get("approximation_boxes", [])
    approved_primitive = collision.get("settings", {}).get("explicit_approximation", False)
    approximated = 0
    for shape, offset in P._compound_children(asm.part_shape(part)):
      hull = P._load_mesh(shape[1])["hull_mesh"].copy()
      hull.apply_translation(offset)
      if approved_primitive:
        approximated += 1
        continue
      if any(np.all(hull.bounds[0] >= np.array(box["lower"]) - 1e-8) and
             np.all(hull.bounds[1] <= np.array(box["upper"]) + 1e-8) for box in approved):
        approximated += 1
        continue
      centers.append(hull.vertices.mean(axis=0))
      crossings += int(check.intersects(hull))
    outside = int((~solid_contains(mesh, centers)).sum()) if centers else 0
    print(f"[containment] {part}: {len(centers)} cells, "
          f"{crossings} boundary intersections, {outside} outside centres, "
          f"{approximated} cells in approved explicit approximations", flush=True)
    if crossings or outside:
      raise ValueError(f"{part}: exported collision cells are not contained in exact CAD")


def audit(name, distance=.05, step=.0005, tolerance=1e-6, asset_root=None,
          containment=False, robustness=0.0):
  asm = P.make_assembly("fabrica-" + name, MODELS[name])
  if asset_root:
    root = Path(asset_root).resolve() / asm.model
    manifest = json.loads((root / "manifest.json").read_text())
    asm._manifest = manifest
    for p in asm.objects:
      source = Path(asm.visual_meshes[p]).stem
      asm.part_shapes[p] = ("compound", [
          {"shape": ("mesh", {"file": str(root / h["file"]), "geometry": h["geometry"]}),
           "pos": h["pos"]} for h in manifest["parts"][source]["collision"]["hulls"]])
      asm.visual_meshes[p] = str(root / (source + ".obj"))
  if containment:
    check_containment(asm)
  C = P.add_disassembled_obj(P.make_world(), asm, center_scatter=True, float_height=.002)
  poses = {p: np.r_[asm.position_in_base(p) + [0, 0, 2], [1, 0, 0, 0]]
           for p in asm.objects}
  for p, pose in poses.items():
    C.getFrame(p).setPosition(pose[:3])
    C.getFrame(p).setQuaternion(pose[3:])
  model = mujoco.MjModel.from_xml_string(sim.build_mjcf(asm, SimpleNamespace(poses=poses), []))
  data = mujoco.MjData(model)
  body_parts = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p): p
                for p in asm.objects}
  addresses = {p: model.jnt_qposadr[model.body_jntadr[b]] for b, p in body_parts.items()}
  for p, pose in poses.items():
    data.qpos[addresses[p]:addresses[p] + 7] = pose
  rows = []
  for index, part in enumerate(asm.objects[1:], 1):
    fixed = set(asm.objects[:index])
    row = {"part": part, "rai_max_penetration_mm": 0., "mujoco_max_penetration_mm": 0.,
           "rai_bad_samples": 0, "mujoco_bad_samples": 0}
    offsets = np.linspace(distance, 0, int(np.ceil(distance / step)) + 1)
    for offset in offsets:
      xyz = poses[part][:3] - asm.mate_axis(part) * offset
      C.getFrame(part).setPosition(xyz)
      # Rigid frame changes must refresh FCL's transforms before reading proxies.
      C.setJointState(C.getJointState())
      C.computeCollisions()
      ds = [d for a, b, d in C.getCollisions()
            if ((a.split("__")[0] == part and b.split("__")[0] in fixed)
                or (b.split("__")[0] == part and a.split("__")[0] in fixed))]
      depth = -min(ds, default=0.)
      row["rai_max_penetration_mm"] = max(row["rai_max_penetration_mm"], depth * 1000)
      row["rai_bad_samples"] += int(depth > tolerance)
      data.qpos[addresses[part]:addresses[part] + 3] = xyz
      mujoco.mj_forward(model, data)
      ds = []
      for contact in data.contact:
        a, b = (body_parts.get(int(model.geom_bodyid[g])) for g in contact.geom)
        if (a == part and b in fixed) or (b == part and a in fixed):
          ds.append(float(contact.dist))
      depth = -min(ds, default=0.)
      row["mujoco_max_penetration_mm"] = max(row["mujoco_max_penetration_mm"], depth * 1000)
      row["mujoco_bad_samples"] += int(depth > tolerance)
    C.getFrame(part).setPosition(poses[part][:3])
    C.getFrame(part).setQuaternion(poses[part][3:])
    data.qpos[addresses[part]:addresses[part] + 7] = poses[part]
    if robustness > 0:
      robust = {"rai_aggregate_penetration_mm": 0.0,
                "mujoco_aggregate_penetration_mm": 0.0,
                "worst_offset": None}
      for axis in range(3):
        for direction in (-1.0, 1.0):
          delta = np.zeros(3)
          delta[axis] = direction * robustness
          xyz = poses[part][:3] + delta
          C.getFrame(part).setPosition(xyz)
          C.setJointState(C.getJointState())
          C.computeCollisions()
          ds = [float(d) for a, b, d in C.getCollisions()
                if ((a.split("__")[0] == part and b.split("__")[0] in fixed)
                    or (b.split("__")[0] == part and a.split("__")[0] in fixed))]
          rai_sum = sum(max(0.0, -d) for d in ds)
          data.qpos[addresses[part]:addresses[part] + 3] = xyz
          mujoco.mj_forward(model, data)
          ds = []
          for contact in data.contact:
            a, b = (body_parts.get(int(model.geom_bodyid[g])) for g in contact.geom)
            if (a == part and b in fixed) or (b == part and a in fixed):
              ds.append(float(contact.dist))
          mj_sum = sum(max(0.0, -d) for d in ds)
          if max(rai_sum, mj_sum) > max(robust["rai_aggregate_penetration_mm"],
                                       robust["mujoco_aggregate_penetration_mm"]) / 1000:
            robust["worst_offset"] = delta.tolist()
          robust["rai_aggregate_penetration_mm"] = max(
              robust["rai_aggregate_penetration_mm"], rai_sum * 1000)
          robust["mujoco_aggregate_penetration_mm"] = max(
              robust["mujoco_aggregate_penetration_mm"], mj_sum * 1000)
      C.getFrame(part).setPosition(poses[part][:3])
      C.setJointState(C.getJointState())
      data.qpos[addresses[part]:addresses[part] + 7] = poses[part]
      row["robustness"] = robust
      row["robust_bad"] = int(max(robust["rai_aggregate_penetration_mm"],
                                  robust["mujoco_aggregate_penetration_mm"])
                              > tolerance * 1000)
    row["samples"] = len(offsets)
    rows.append(row)
    print(json.dumps(row), flush=True)
  return rows


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("model", choices=sorted(MODELS))
  ap.add_argument("--step", type=float, default=.0005)
  ap.add_argument("--distance", type=float, default=.05)
  ap.add_argument("--tolerance", type=float, default=1e-6)
  ap.add_argument("--asset-root", type=Path, help="audit a staged importer output")
  ap.add_argument("--containment", action="store_true", help="also check every cell against exact CAD")
  ap.add_argument("--robustness", type=float, default=0.0,
                  help="also perturb each seated part by +/- this many metres on x/y/z and "
                       "check the planner's aggregate penetration metric")
  args = ap.parse_args()
  if args.step <= 0 or args.distance < 0 or args.tolerance < 0 or args.robustness < 0:
    ap.error("step must be positive; distance, tolerance and robustness must be nonnegative")
  rows = audit(args.model, args.distance, args.step, args.tolerance, args.asset_root,
               args.containment, args.robustness)
  failed = any(r["rai_bad_samples"] or r["mujoco_bad_samples"] or r.get("robust_bad")
               for r in rows)
  print("FAIL: collision on insertion sweep" if failed else "PASS: sampled insertion sweeps clear")
  raise SystemExit(int(failed))


if __name__ == "__main__":
  main()
