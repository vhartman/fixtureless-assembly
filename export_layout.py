"""Export a saved plan's nominal part layout, and a true-scale outline sheet to print.

* This is the FIRST half of the pickup-pose workflow. It answers "where does the planner
* think the parts are?" in a form the operator can carry to the robot:
*
*   1. a layout JSON (schema assembly-pickup-layout-v1) with each part's x/y, yaw, footprint
*      and height, plus the table, all in the robot base frame;
*   2. an optional outline sheet -- the parts drawn at 1:1, so the sheet can be printed, laid
*      on the table, and the real parts placed on their outlines.
*
* The relative layout is what matters: the sheet fixes every part's pose RELATIVE to the
* others, so the operator only has to measure ONE anchor part on the robot. The teleop
* package's pickup_calib tool reads this file, records punch-tip touches, and writes the
* measured layout back out for `main.py --layout-json`.

Usage:
  python3 export_layout.py experiments/real-stool-husky-current/plan.json \
      --out experiments/real-stool-husky-current/layout_nominal.json \
      --svg experiments/real-stool-husky-current/layout_outline.svg
"""

import argparse
import json
import os

import numpy as np

from config import saved_cfg
from plan_io import LAYOUT_SCHEMA, read_plan
from problem import (HERE, _quat_to_R, _shape_half_extents, _shape_half_z)
from replay import build_scene

# The frame every coordinate in a layout file lives in. The calibrated husky rai world and
# the teleop package's PyBullet URDF share it exactly (z=0 floor, same arm base links), so a
# point measured on the robot is directly a coordinate here -- no extra transform.
LAYOUT_FRAME = "husky base_footprint == fixtureless rai world (z=0 floor)"

# Metres -> inches, so a matplotlib figure sized in inches prints at true scale.
INCHES_PER_M = 39.3701


def part_yaw(part_quat, scatter_quat) -> float:
  """Turn about world z that takes the part from how it LIES to where it points.

  A part's orientation is its scatter orientation (how it lies on the table, e.g. a leg on
  its side) followed by a turn about the world z axis. Only that turn is a layout degree of
  freedom, and only it can be measured by touching corners, so it is what the layout stores.

  Args:
    part_quat: The part's world quaternion (wxyz).
    scatter_quat: How the assembly says the part lies (wxyz), or None for upright.

  Returns:
    float: The yaw in radians.
  """
  R_part = _quat_to_R(np.asarray(part_quat, dtype=float))
  if scatter_quat is None:
    R_yaw = R_part
  else:
    R_yaw = R_part @ _quat_to_R(np.asarray(scatter_quat, dtype=float)).T
  return float(np.arctan2(R_yaw[1, 0], R_yaw[0, 0]))


def part_layout_entry(assembly, C, name) -> dict:
  """One part's layout record, read from the rebuilt scene.

  Args:
    assembly: The Assembly, for the part's shape and scatter orientation.
    C: The rebuilt rai configuration.
    name (str): Part name.

  Returns:
    dict: The layout entry (see the module docstring for the schema).
  """
  frame = C.getFrame(name)
  pos = np.asarray(frame.getPosition(), dtype=float)
  scatter_q = assembly.scatter_orientations.get(name)
  spec = assembly.part_shape(name)
  R_scatter = (_quat_to_R(np.asarray(scatter_q, dtype=float))
               if scatter_q is not None else np.eye(3))
  # Footprint of the part AS IT LIES, before the yaw: rotate the shape's half extents into
  # the lying orientation and take the absolute cover, so a lying leg reads 105 x 22 mm.
  half_xy = (np.abs(R_scatter) @ _shape_half_extents(spec))[:2]
  mesh = assembly.visual_meshes.get(name)
  return {
      "xy": [float(pos[0]), float(pos[1])],
      "yaw": part_yaw(frame.getQuaternion(), scatter_q),
      "footprint_half_xy": [float(half_xy[0]), float(half_xy[1])],
      "height": float(2.0 * _shape_half_z(spec, R_scatter)),
      "scatter_quat_wxyz": ([float(v) for v in scatter_q]
                            if scatter_q is not None else [1.0, 0.0, 0.0, 0.0]),
      "visual_mesh": (os.path.join(HERE, mesh) if mesh else None),
      # The exporter only ever writes NOMINAL poses; pickup_calib flips this as it measures.
      "measured": False,
  }


def table_layout_entry(C) -> dict:
  """The work surface, read from the rai table frame.

  Args:
    C: The rebuilt rai configuration.

  Returns:
    dict: ``center_xy``, ``size_xy``, ``thickness`` and ``top_z`` in metres.
  """
  f = C.getFrame("table")
  pos = np.asarray(f.getPosition(), dtype=float)
  size = np.asarray(f.getSize(), dtype=float)
  return {
      "center_xy": [float(pos[0]), float(pos[1])],
      "size_xy": [float(size[0]), float(size[1])],
      "thickness": float(size[2]),
      "top_z": float(pos[2] + size[2] / 2.0),
  }


def export_layout(plan_path) -> dict:
  """Rebuild a saved plan's scene and describe its nominal layout.

  Args:
    plan_path (str): Path to a plan JSON saved by ``main.py --save``.

  Returns:
    dict: The layout, ready to write as JSON.
  """
  cfg, _plan = read_plan(plan_path)
  C, assembly = build_scene(cfg)
  return {
      "schema": LAYOUT_SCHEMA,
      "frame": LAYOUT_FRAME,
      "source_plan": os.path.abspath(plan_path),
      # Parts hover this far above the table top; pickup_calib needs it to place the meshes
      # at the same height the planner does.
      "part_float": float(saved_cfg(cfg, "part_float", legacy=0.015)),
      "table": table_layout_entry(C),
      "parts": {name: part_layout_entry(assembly, C, name) for name in assembly.objects},
  }


def corner_world_xy(entry) -> np.ndarray:
  """The part outline's four corners in world x/y, in +x+y, -x+y, -x-y, +x-y order.

  Args:
    entry (dict): One layout part entry.

  Returns:
    np.ndarray: A (4, 2) array of corner positions [m].
  """
  hx, hy = entry["footprint_half_xy"]
  yaw = float(entry["yaw"])
  c, s = np.cos(yaw), np.sin(yaw)
  R = np.array([[c, -s], [s, c]])
  local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
  return np.asarray(entry["xy"], dtype=float) + local @ R.T


def write_outline_sheet(layout, path) -> None:
  """Draw the layout at 1:1 so it can be printed and laid on the table.

  ! Print this at 100% / "actual size". Any "fit to page" scaling silently shrinks the
  ! sheet and every part then sits in the wrong place. The 100 mm scale bar is there to be
  ! measured with a ruler after printing -- check it before trusting the sheet.

  Args:
    layout (dict): The layout to draw.
    path (str): Output path; a PDF is written next to it as well.
  """
  # Agg-only figure (no pyplot): this may run headless.
  from matplotlib.figure import Figure
  from matplotlib.patches import Rectangle

  table = layout["table"]
  (cx, cy), (sx, sy) = table["center_xy"], table["size_xy"]
  margin = 0.05
  width, height = sx + 2 * margin, sy + 2 * margin
  fig = Figure(figsize=(width * INCHES_PER_M, height * INCHES_PER_M))
  ax = fig.add_subplot(111)
  ax.set_xlim(cx - sx / 2 - margin, cx + sx / 2 + margin)
  ax.set_ylim(cy - sy / 2 - margin, cy + sy / 2 + margin)
  ax.set_aspect("equal")
  ax.set_axis_off()

  # Table outline, and the robot-facing edge marked so the sheet cannot be laid on backwards.
  ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fill=False,
                         edgecolor="0.6", linewidth=1.0, linestyle="--"))
  ax.plot([cx - sx / 2, cx + sx / 2], [cy - sy / 2, cy - sy / 2], color="0.3", linewidth=2.0)
  ax.text(cx, cy - sy / 2 - 0.015, "<< THIS EDGE TOWARDS THE ROBOT >>",
          ha="center", va="top", fontsize=7, color="0.3")

  for name, entry in sorted(layout["parts"].items()):
    corners = corner_world_xy(entry)
    ax.add_patch(Rectangle((0, 0), 0, 0))          # keeps the patch list non-empty for older mpl
    ax.plot(np.append(corners[:, 0], corners[0, 0]),
            np.append(corners[:, 1], corners[0, 1]),
            color="black", linewidth=1.2)
    ax.text(entry["xy"][0], entry["xy"][1], name, ha="center", va="center", fontsize=8)
    # Label the corners with the very names pickup_calib's dropdown offers, so the operator
    # touches the corner the tool is asking for.
    for (px, py), label in zip(corners, ("+x+y", "-x+y", "-x-y", "+x-y")):
      ax.text(px, py, label, ha="center", va="center", fontsize=5, color="0.45")

  # Scale bar: 100 mm, to verify the print came out at true size.
  bar_y = cy + sy / 2 + margin / 2
  bar_x = cx - sx / 2
  ax.plot([bar_x, bar_x + 0.1], [bar_y, bar_y], color="black", linewidth=1.5)
  ax.text(bar_x + 0.05, bar_y + 0.006, "100 mm -- measure me after printing",
          ha="center", va="bottom", fontsize=6)

  fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
  fig.savefig(path)
  fig.savefig(os.path.splitext(path)[0] + ".pdf")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("plan", help="plan JSON saved by main.py --save")
  ap.add_argument("--out", default=None,
                  help="layout JSON to write (default: <plan dir>/layout_nominal.json)")
  ap.add_argument("--svg", default=None,
                  help="also draw a true-scale outline sheet here (a PDF is written too)")
  args = ap.parse_args()

  layout = export_layout(args.plan)
  out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.plan)),
                                 "layout_nominal.json")
  with open(out, "w") as f:
    json.dump(layout, f, indent=2)

  table = layout["table"]
  print(f"[layout] table {table['size_xy'][0]:.3f} x {table['size_xy'][1]:.3f} m centred at "
        f"({table['center_xy'][0]:.4f}, {table['center_xy'][1]:.4f}), "
        f"top z={table['top_z']:.4f} m")
  for name, entry in sorted(layout["parts"].items()):
    print(f"[layout] {name}: xy ({entry['xy'][0]:.4f}, {entry['xy'][1]:.4f}) "
          f"yaw {np.rad2deg(entry['yaw']):7.2f} deg  footprint "
          f"{2 * entry['footprint_half_xy'][0] * 1000:6.1f} x "
          f"{2 * entry['footprint_half_xy'][1] * 1000:6.1f} mm  "
          f"height {entry['height'] * 1000:5.1f} mm")
  print(f"[layout] wrote {len(layout['parts'])} part(s) -> {out}")

  if args.svg:
    write_outline_sheet(layout, args.svg)
    print(f"[layout] outline sheet (print at 100%) -> {args.svg} and "
          f"{os.path.splitext(args.svg)[0] + '.pdf'}")


if __name__ == "__main__":
  main()
