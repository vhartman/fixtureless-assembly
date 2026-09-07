#!/usr/bin/env python3
"""Convert the Y&V Rhino OBJ exports to centred, metre-scale, rai Z-up visuals.

The output is visualization only. The planner and MuJoCo collision model use the cuboids in
YandVStoolAssembly. Run from anywhere; output is deterministic and written beside this script.
"""
from pathlib import Path

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "260904_YandV_obj_v2"


def convert(source_name, target_name, cad_origin):
  mesh = trimesh.load(SOURCE / source_name, force="mesh", process=False)
  V = np.asarray(mesh.vertices, dtype=float) - np.asarray(cad_origin, dtype=float)
  # Rhino CAD: (x, y-up, z). Planner: (x, y=-cad_z, z=cad_y), in metres.
  V = np.column_stack((V[:, 0], -V[:, 2], V[:, 1])) * 1e-3
  out = trimesh.Trimesh(V, np.asarray(mesh.faces, dtype=int), process=False)
  out.export(HERE / target_name)
  print(f"{target_name}: {len(out.vertices)} vertices, {len(out.faces)} faces, "
        f"bounds {np.round(out.bounds, 6).tolist()}")


def main():
  HERE.mkdir(parents=True, exist_ok=True)
  # Part frames are inferred from the authored assembly, rather than from a visual bounding
  # box.  The seat frame is at the centre of the 35mm main slab (CAD Y 80.25..115.25);
  # the rear upstand consequently occupies local z=17.5..57.5mm.  Every leg keeps its own
  # handed CAD surface and is centred on its 105mm long axis.  The four files are close, but
  # not identical: reusing LF for all four loses the handed details in the assembled model.
  convert("260904_YandV_table_4holes_v2.obj", "seat_4hole_visual.obj",
          [-30.0, 97.75, 30.0])
  convert("260904_YandV_table_2holes_v2.obj", "seat_2hole_visual.obj",
          [12.5, 97.75, 30.0])
  for corner, origin in {
      "LF": [24.0, 52.5, -24.0],
      "RF": [-84.0, 52.5, -24.0],
      "LB": [24.0, 52.5, 84.0],
      "RB": [-84.0, 52.5, 84.0],
  }.items():
    convert(f"260904_YandV_stool_{corner}_v2.obj",
            f"leg_{corner.lower()}_visual.obj", origin)


if __name__ == "__main__":
  main()
