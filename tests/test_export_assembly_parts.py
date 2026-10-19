"""The generic part exporter must preserve planner geometry and the established units."""

import json

import numpy as np
import pytest
import trimesh

from tools.exporters.export_assembly_parts import export_assembly


@pytest.mark.parametrize("kind,count", [("cross", 7), ("cube", 8), ("chair", 8)])
def test_export_cross_cube_and_chair(tmp_path, kind, count):
  out = tmp_path / kind
  manifest = export_assembly(kind, output=out)
  saved = json.loads((out / "manifest.json").read_text())
  assert manifest["parts"] == saved["parts"] == count
  assert len(saved["part_files"]) == count
  assert (tmp_path / f"{kind}.zip").is_file()

  for row in saved["part_files"]:
    glb = trimesh.load(out / row["glb"], force="scene")
    stl = trimesh.load(out / row["stl"], force="scene")
    assert len(glb.geometry) == row["solid_count"]
    # STL is exactly the same local geometry with metre -> millimetre conversion.
    np.testing.assert_allclose(stl.extents, glb.extents * 1000.0, rtol=2e-5, atol=2e-4)
    bounds = np.asarray(row["bounds_mm"], dtype=float)
    np.testing.assert_allclose(glb.bounds * 1000.0, bounds, rtol=2e-5, atol=2e-4)

  # The chair seat is authored as a rotated primitive. Its thin source-y axis must be baked
  # into exported z, matching the physical chair rather than the unrotated parameter array.
  if kind == "chair":
    seat = trimesh.load(out / saved["part_files"][0]["glb"], force="scene")
    np.testing.assert_allclose(seat.extents, [.19, .175, .025], atol=2e-6)


def test_export_rejects_invalid_part_count(tmp_path):
  with pytest.raises(ValueError, match="--parts"):
    export_assembly("cross", parts=8, output=tmp_path / "bad")
