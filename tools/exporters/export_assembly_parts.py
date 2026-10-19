#!/usr/bin/env python3
"""Export an authored assembly's individual parts as STL (mm) and GLB (m).

This generalises the one-off export previously made for
``peg-stool-cuboid-4-tight-chamfer``.  Geometry comes directly from
``Assembly.part_shape``: the files therefore describe the same authored primitive, mesh,
rotated-shape, or compound solids, in each part's own local frame.  It does not export a second
hand-maintained CAD interpretation. As in the earlier stool export and the grasp-sampling mesh,
an ``ssBox`` is emitted as its nominal sharp box; its collision-only corner sweep is not CAD.

Examples::

    python3 tools/exporters/export_assembly_parts.py --assembly cross
    python3 tools/exporters/export_assembly_parts.py --assembly cube --parts 6
    python3 tools/exporters/export_assembly_parts.py --assembly chair --output exports/chair

STL has no standard unit metadata, so its vertex coordinates are deliberately millimetres.
GLB vertices remain in repository-native metres.  For a compound part, GLB retains one named
mesh per child solid; STL concatenates those solids into the one physical part file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import zipfile

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import trimesh

from assembly_planner.problem import (assembly_max_parts, make_assembly, _compound_children, _quat_to_R,
                     _shape_quat, _spec_mesh)


_PART_LABELS = {
    "cross": ("center", "top", "bottom", "positive_y", "negative_y",
              "positive_x", "negative_x"),
    "chair": ("seat", "leg_1", "leg_2", "leg_3", "leg_4", "backrest",
              "back_post_1", "back_post_2"),
}


def _safe_name(value: str) -> str:
  value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
  if not value:
    raise ValueError("part filename became empty after sanitising")
  return value


def _part_stem(assembly_kind: str, part: str) -> str:
  index = int(part.split("_")[-1])
  labels = _PART_LABELS.get(assembly_kind, ())
  label = labels[index] if index < len(labels) else ("cube_%02d" % index
                                                     if assembly_kind == "cube" else part)
  return _safe_name(f"{part}_{label}" if label != part else part)


def _solid_meshes(spec) -> list[trimesh.Trimesh]:
  """Part-local child solids with every shape quaternion and offset baked in."""
  if spec[0] == "compound":
    solids = []
    for child, offset in _compound_children(spec):
      # Nested compounds are rejected by the assembly representation itself. Keeping this
      # recursive makes the exporter robust if that representation is relaxed later.
      for mesh in _solid_meshes(child):
        mesh = mesh.copy()
        mesh.apply_translation(np.asarray(offset, dtype=float))
        solids.append(mesh)
    return solids

  mesh = _spec_mesh(spec).copy()
  quat = _shape_quat(spec)
  if quat is not None:
    transform = np.eye(4)
    transform[:3, :3] = _quat_to_R(np.asarray(quat, dtype=float))
    mesh.apply_transform(transform)
  return [mesh]


def _bounds_mm(solids: list[trimesh.Trimesh]) -> list[list[float]]:
  mesh = trimesh.util.concatenate([m.copy() for m in solids])
  return np.round(np.asarray(mesh.bounds, dtype=float) * 1000.0, 6).tolist()


def export_assembly(assembly_kind: str, parts: int | None = None,
                    output: str | Path | None = None, make_zip: bool = True) -> dict:
  """Export all requested parts and return the manifest dictionary."""
  max_parts = assembly_max_parts(assembly_kind)
  part_count = max_parts if parts is None else int(parts)
  if not 1 <= part_count <= max_parts:
    raise ValueError(f"--parts must be in [1, {max_parts}] for {assembly_kind!r}")
  assembly = make_assembly(assembly_kind, part_count)
  out = Path(output) if output is not None else Path("exports") / assembly_kind
  out.mkdir(parents=True, exist_ok=True)

  records = []
  generated = []
  for part in assembly.objects:
    stem = _part_stem(assembly_kind, part)
    solids = _solid_meshes(assembly.part_shape(part))
    if not solids or any(len(m.vertices) == 0 or len(m.faces) == 0 for m in solids):
      raise ValueError(f"{assembly_kind}:{part} produced empty export geometry")

    glb_path = out / f"{stem}.glb"
    scene = trimesh.Scene()
    for i, mesh in enumerate(solids):
      name = f"{stem}_solid_{i:02d}"
      scene.add_geometry(mesh.copy(), node_name=name, geom_name=name)
    scene.export(glb_path)

    stl_path = out / f"{stem}.stl"
    stl = trimesh.util.concatenate([m.copy() for m in solids])
    stl.apply_scale(1000.0)
    stl.export(stl_path)
    generated.extend((glb_path, stl_path))

    rel = assembly.relative_pose(part)
    records.append({
        "part": part,
        "stem": stem,
        "glb": glb_path.name,
        "stl": stl_path.name,
        "solid_count": len(solids),
        "bounds_mm": _bounds_mm(solids),
        "assembled_position_mm": np.round(
            np.asarray(assembly.position_in_base(part), dtype=float) * 1000.0, 6).tolist(),
        "mate_parent": None if rel is None else rel[0],
    })

  manifest = {
      "assembly": assembly_kind,
      "parts": len(assembly.objects),
      "base": assembly.base,
      "coordinate_frame": "part-local; authored shape rotations are baked into vertices",
      "units": {"glb": "metres", "stl": "millimetres"},
      "part_files": records,
  }
  manifest_path = out / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  generated.append(manifest_path)

  lines = [
      f"# {assembly_kind} part meshes",
      "",
      f"Generated from the repository assembly model with `{len(assembly.objects)}` parts:",
      "",
      f"    python3 tools/exporters/export_assembly_parts.py --assembly {assembly_kind} "
      f"--parts {len(assembly.objects)}",
      "",
      "Each file is in the corresponding part's local coordinate frame. STL coordinates are",
      "in **millimetres**; GLB coordinates are in **metres**. Compound GLBs preserve their",
      "child solids as separately named meshes, while each STL contains one concatenated part.",
      "Collision-only ssBox corner sweeps are exported as their nominal sharp boxes.",
      "",
      "| Part | STL / GLB stem | Solids | Size (mm) | Assembled position (mm) |",
      "|---|---|---:|---:|---:|",
  ]
  for row in records:
    bounds = np.asarray(row["bounds_mm"], dtype=float)
    size = bounds[1] - bounds[0]
    pos = row["assembled_position_mm"]
    lines.append(
        f"| `{row['part']}` | `{row['stem']}` | {row['solid_count']} | "
        f"{' x '.join(f'{x:g}' for x in size)} | "
        f"({', '.join(f'{x:g}' for x in pos)}) |")
  readme_path = out / "README.md"
  readme_path.write_text("\n".join(lines) + "\n")
  generated.append(readme_path)

  if make_zip:
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
      for path in sorted(generated, key=lambda p: p.name):
        archive.write(path, arcname=path.name)
    manifest["zip"] = str(zip_path)
  return manifest


def _parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--assembly", required=True,
                      help="assembly name accepted by main.py, e.g. cross, cube, chair")
  parser.add_argument("--parts", type=int,
                      help="prefix part count (default: the assembly's maximum)")
  parser.add_argument("--output", type=Path,
                      help="output directory (default: exports/<assembly>)")
  parser.add_argument("--no-zip", action="store_true", help="do not create the sibling zip")
  return parser.parse_args()


def main():
  args = _parse_args()
  result = export_assembly(args.assembly, args.parts, args.output, not args.no_zip)
  root = args.output or Path("exports") / args.assembly
  print(f"[export] {result['assembly']}: {result['parts']} parts -> {root}")
  if "zip" in result:
    print(f"[export] archive -> {result['zip']}")


if __name__ == "__main__":
  main()
