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

# A4 in metres, and the printable area left after a margin every printer can manage. The
# sheet is tiled across as many A4 pages as the layout needs, always at 1:1.
A4_W, A4_H = 0.210, 0.297
PAGE_MARGIN = 0.008
# ! Parts are never split across a page. Each sheet is INDEPENDENT: it holds whole parts and
# ! its own four crosses, and the robot measures each sheet on its own. So the sheets never
# ! have to be aligned or taped to each other -- a seam through a part outline would turn
# ! taping error straight into part-placement error.
# Smallest gap left between a part and a cross, and how close a cross may come to the page
# edge. Crosses are pushed as far out as those two allow, because a longer baseline between
# them makes the fitted turn more accurate.
MIN_MARK_MARGIN = 0.006
PAGE_EDGE_INSET = 0.006


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


def export_layout(plan_path, pack=True) -> dict:
  """Rebuild a saved plan's scene and describe its nominal layout.

  Args:
    plan_path (str): Path to a plan JSON saved by ``main.py --save``.
    pack (bool): Repack the parts compactly so they print on one sheet (see
      `pack_parts_onto_one_sheet`). False keeps the plan's own scatter.

  Returns:
    dict: The layout, ready to write as JSON.
  """
  cfg, _plan = read_plan(plan_path)
  C, assembly = build_scene(cfg)
  layout = {
      "schema": LAYOUT_SCHEMA,
      "frame": LAYOUT_FRAME,
      "source_plan": os.path.abspath(plan_path),
      # Parts hover this far above the table top; pickup_calib needs it to place the meshes
      # at the same height the planner does.
      "part_float": float(saved_cfg(cfg, "part_float", legacy=0.015)),
      "table": table_layout_entry(C),
      "parts": {name: part_layout_entry(assembly, C, name) for name in assembly.objects},
  }
  if pack:
    if pack_parts_onto_one_sheet(layout):
      layout["packed"] = True
      print("[layout] PACKED onto one sheet: the parts have been MOVED from the plan's "
            "scatter into a compact block. Solve again with --layout-json <this file>; the "
            "plan it came from no longer describes these positions.")
    else:
      print("[layout] could not pack every part onto one A4 sheet; keeping the plan's "
            "layout and splitting across sheets instead.")
  # The crosses the operator touches. They depend on where the parts ended up, so they are
  # added once the parts are in.
  layout["calibration_marks"] = calibration_marks(layout)
  return layout


# Clear space left between packed parts. Measured with problem.check_initial_scene on the
# packed bench (real-bench, husky, rolled legs): at 10 and 20 mm only the seat is graspable;
# at 40 mm all three parts are. Note the bounding-box gap on its own is NOT the criterion --
# a 20 mm pack that the spread step later opened to 43-115 mm was still ungraspable, because
# what matters is where each part ends up relative to the arms, not merely how far apart the
# parts are. So this value is an empirical floor for THIS cell; re-check it with
# `main.py --check-init` whenever the parts, gripper or table move.
PACK_GAP = 0.040


def pack_parts_onto_one_sheet(layout, gap=PACK_GAP) -> bool:
  """Re-place every part compactly, so the whole layout prints on ONE A4 sheet.

  The planner's scatter spreads the parts across the table for its own reasons (reach,
  clearance), which is far wider than A4 even though the parts themselves are small. Since
  the operator lays the sheet down and the planner re-solves for wherever the parts end up,
  the sheet may just as well define a COMPACT arrangement: one sheet, one set of crosses,
  one measurement, nothing to tape or align.

  ! This CHANGES where the parts are. The plan the layout was exported from no longer
  ! matches it, so the plan must be solved again with `--layout-json <this file>`.

  Parts keep the way they LIE; a part may be turned a quarter turn about world z, because
  that is an ordinary layout degree of freedom and it is often what makes the set fit. The
  packed block is centred on the original layout's centroid, so it stays in the same
  reachable patch of table.

  Args:
    layout (dict): The layout, modified in place.
    gap (float): Clear space left between parts [m]. Below PACK_GAP the scene is not
      graspable; see that constant.

  Returns:
    bool: True if everything fitted and the layout was repacked.
  """
  import itertools
  usable = np.array([A4_W - 2 * PAGE_MARGIN, A4_H - 2 * PAGE_MARGIN])
  avail = usable - 2 * (MIN_MARK_MARGIN + 0.006)
  names = sorted(layout["parts"])
  base = {}
  for name in names:
    lo, hi = _part_bounds(layout["parts"][name])
    base[name] = (hi - lo, (lo + hi) / 2.0)

  # Only a handful of parts, so try every combination of quarter turns against BOTH a
  # row-major and a column-major arrangement, and keep the most compact result. Row-major
  # alone misses the obvious answer for this set -- two slim legs side by side in a column
  # with the seat next to them -- because it fills a row before starting the next.
  best = None
  for turns in itertools.product((0, 1), repeat=len(names)):
    sizes = {n: (base[n][0][::-1].copy() if turn else base[n][0].copy())
             for n, turn in zip(names, turns)}
    for by_column in (False, True):
      # Along the packing direction, tallest/widest first keeps the shelves even.
      order = sorted(names, key=lambda n: -sizes[n][0 if by_column else 1])
      a, b = (1, 0) if by_column else (0, 1)     # a = along the shelf, b = across it
      pos_a = pos_b = shelf = 0.0
      placed, failed = {}, False
      for name in order:
        size = sizes[name]
        if np.any(size > avail):
          failed = True
          break
        if pos_a + size[a] > avail[a]:
          pos_a, pos_b, shelf = 0.0, pos_b + shelf + gap, 0.0
        if pos_b + size[b] > avail[b]:
          failed = True
          break
        centre = np.empty(2)
        centre[a] = pos_a + size[a] / 2.0
        centre[b] = pos_b + size[b] / 2.0
        placed[name] = centre
        pos_a += size[a] + gap
        shelf = max(shelf, size[b])
      if failed:
        continue
      extent = np.array([max(placed[n][0] + sizes[n][0] / 2.0 for n in placed),
                         max(placed[n][1] + sizes[n][1] / 2.0 for n in placed)])
      score = float(extent[0] * extent[1])
      if best is None or score < best[0]:
        best = (score, dict(placed), dict(zip(names, turns)), extent, dict(sizes))
  if best is None:
    return False

  _score, placed, turns, extent, sizes = best
  # Spend whatever the sheet has left on WIDER GAPS. Clearance is what decides whether the
  # gripper can reach a part at all (see PACK_GAP), so a packing that merely fits is worse
  # than the same packing relaxed to fill the page.
  centre_block = extent / 2.0
  for axis in (0, 1):
    ratios = [(avail[axis] / 2.0 - sizes[n][axis] / 2.0) / abs(placed[n][axis] - centre_block[axis])
              for n in placed if abs(placed[n][axis] - centre_block[axis]) > 1e-9]
    scale = float(np.clip(min(ratios), 1.0, 4.0)) if ratios else 1.0
    for name in placed:
      placed[name][axis] = centre_block[axis] + scale * (placed[name][axis] - centre_block[axis])
  extent = np.array([max(placed[n][axis] + sizes[n][axis] / 2.0 for n in placed)
                     for axis in (0, 1)])
  old_centre = np.mean([base[n][1] for n in names], axis=0)
  offset = old_centre - extent / 2.0
  for name in names:
    entry = layout["parts"][name]
    if turns[name]:
      entry["yaw"] = float(entry["yaw"] + np.pi / 2.0)
    # A part's stored xy is its FRAME, which need not be the centre of its outline (the
    # seat's upstand makes it off-centre), so recompute the offset AFTER any turn.
    lo, hi = _part_bounds(entry)
    frame_to_outline = (lo + hi) / 2.0 - np.asarray(entry["xy"], dtype=float)
    new_xy = placed[name] + offset - frame_to_outline
    entry["xy"] = [float(new_xy[0]), float(new_xy[1])]
  return True


def _part_bounds(entry):
  """One part's axis-aligned outline bounds: ``(lo_xy, hi_xy)`` [m]."""
  corners = corner_world_xy(entry)
  return corners.min(axis=0), corners.max(axis=0)


def assign_parts_to_pages(layout) -> list:
  """Group whole parts onto A4 sheets, largest first.

  A part is never split. Parts are added to a sheet while the group still leaves room for
  the crosses; otherwise a new sheet is started. Biggest-first keeps the awkward part (the
  seat) on a sheet of its own instead of straddling two.

  Args:
    layout (dict): The layout.

  Returns:
    list: One list of part names per page.
  """
  usable = np.array([A4_W - 2 * PAGE_MARGIN, A4_H - 2 * PAGE_MARGIN])
  room = usable - 2 * MIN_MARK_MARGIN
  order = sorted(layout["parts"],
                 key=lambda n: -np.prod(np.diff(np.array(_part_bounds(layout["parts"][n])),
                                                axis=0)))
  pages, bounds = [], []
  for name in order:
    lo, hi = _part_bounds(layout["parts"][name])
    if np.any(hi - lo > room):
      print(f"[layout] WARNING: {name} is {1000 * (hi - lo)[0]:.0f} x "
            f"{1000 * (hi - lo)[1]:.0f} mm and does not fit an A4 sheet; it gets its own "
            f"page and will be clipped.")
      pages.append([name])
      bounds.append((lo, hi))
      continue
    for i, (plo, phi) in enumerate(bounds):
      grown_lo, grown_hi = np.minimum(plo, lo), np.maximum(phi, hi)
      if np.all(grown_hi - grown_lo <= room):
        pages[i].append(name)
        bounds[i] = (grown_lo, grown_hi)
        break
    else:
      pages.append([name])
      bounds.append((lo, hi))
  # Present the sheets in a stable, human order: nearest the robot first.
  order_idx = sorted(range(len(pages)), key=lambda i: (bounds[i][0][0], bounds[i][0][1]))
  return [sorted(pages[i]) for i in order_idx]


def page_marks(layout, names, page_index) -> list:
  """The four crosses for one sheet, pushed as far apart as the page allows.

  Args:
    layout (dict): The layout.
    names (list): Part names on this sheet.
    page_index (int): 1-based page number.

  Returns:
    list: ``[{"name", "xy", "page"}, ...]`` in the robot base frame.
  """
  usable = np.array([A4_W - 2 * PAGE_MARGIN, A4_H - 2 * PAGE_MARGIN])
  lo = np.min([_part_bounds(layout["parts"][n])[0] for n in names], axis=0)
  hi = np.max([_part_bounds(layout["parts"][n])[1] for n in names], axis=0)
  # The sheet is centred on its parts, so this much page is free on each side.
  gap = (usable - (hi - lo)) / 2.0
  margin = np.maximum(MIN_MARK_MARGIN, gap - PAGE_EDGE_INSET)
  mlo, mhi = lo - margin, hi + margin
  # Named the way the operator sees them: +x points away from the robot, +y to its left.
  return [{"name": f"p{page_index}-near-right", "xy": [float(mlo[0]), float(mlo[1])],
           "page": page_index},
          {"name": f"p{page_index}-near-left", "xy": [float(mlo[0]), float(mhi[1])],
           "page": page_index},
          {"name": f"p{page_index}-far-right", "xy": [float(mhi[0]), float(mlo[1])],
           "page": page_index},
          {"name": f"p{page_index}-far-left", "xy": [float(mhi[0]), float(mhi[1])],
           "page": page_index}]


def calibration_marks(layout) -> list:
  """Every sheet's crosses, and stamp each part with the page it belongs to.

  The crosses are the calibration reference, NOT the parts: they are printed, so their
  positions are exact by construction, they are crisp marks rather than a moulded edge, and
  they span the whole sheet. Each sheet is measured on its own, so the sheets never need to
  be aligned to one another.

  Args:
    layout (dict): The layout, with its parts already filled in.

  Returns:
    list: All marks, each carrying its 1-based ``page``.
  """
  marks = []
  for i, names in enumerate(assign_parts_to_pages(layout), start=1):
    for name in names:
      layout["parts"][name]["page"] = i
    marks += page_marks(layout, names, i)
  return marks


def _parts_bounds(layout):
  """Axis-aligned bounds of every part outline: ``(lo_xy, hi_xy)`` [m]."""
  lo = np.array([np.inf, np.inf])
  hi = np.array([-np.inf, -np.inf])
  for entry in layout["parts"].values():
    corners = corner_world_xy(entry)
    lo = np.minimum(lo, corners.min(axis=0))
    hi = np.maximum(hi, corners.max(axis=0))
  return lo, hi


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


def part_profile_xy(entry):
  """The part's true top-down silhouette in world x/y, or None.

  Traces the actual moulded profile -- the snap-fit joint included -- instead of a bounding
  rectangle, so a part can be placed on its outline by eye with real accuracy. The visual
  mesh is rotated the way the part LIES (its scatter orientation, then the layout yaw) and
  projected straight down, which is exactly the view from above the table.

  ! Needs shapely (through trimesh). It is not in requirements.txt, so fall back to the
  ! rectangle rather than failing: a bounding box is a worse tracing guide, not a broken one.

  Args:
    entry (dict): One layout part entry.

  Returns:
    list or None: ``[exterior, hole, ...]``, each an (n, 2) array of world x/y [m].
  """
  mesh_path = entry.get("visual_mesh")
  if not mesh_path or not os.path.exists(mesh_path):
    return None
  try:
    import trimesh
    from trimesh.path import polygons
  except ImportError:
    return None
  try:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    # Lay the part down the way it rests, then turn it by the measured yaw.
    T = np.eye(4)
    T[:3, :3] = (_yaw_R(float(entry["yaw"]))
                 @ _quat_to_R(np.asarray(entry["scatter_quat_wxyz"], dtype=float)))
    mesh = mesh.copy()
    mesh.apply_transform(T)
    poly = polygons.projected(mesh, normal=[0.0, 0.0, 1.0])
    if poly is None:
      return None
    # The mesh frame is the part frame, so the projection is centred on the part's origin.
    offset = np.asarray(entry["xy"], dtype=float)
    rings = [np.asarray(poly.exterior.coords, dtype=float) + offset]
    rings += [np.asarray(r.coords, dtype=float) + offset for r in poly.interiors]
    return rings
  except Exception:
    return None


def _yaw_R(yaw):
  """3x3 rotation about world z."""
  c, s = np.cos(yaw), np.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def write_outline_sheet(layout, path) -> None:
  """Draw one A4 sheet per group of parts, at 1:1, to print and lay on the table.

  ! Print at 100% / "actual size". Any "fit to page" scaling shrinks the sheet and every
  ! part then sits in the wrong place. Each page carries a 100 mm scale bar -- measure one
  ! with a ruler before trusting the print.

  Each sheet is self-contained: whole parts (never split across a seam) and its own four
  crosses. The sheets are measured independently on the robot, so they are laid down
  separately and never have to be aligned or taped to each other.

  Args:
    layout (dict): The layout to draw.
    path (str): Output path. A multi-page PDF is written next to it, and one SVG per page.

  Returns:
    int: The number of pages written.
  """
  # Agg-only figures (no pyplot): this may run headless.
  from matplotlib.backends.backend_pdf import PdfPages
  from matplotlib.figure import Figure
  from matplotlib.patches import Polygon as MplPolygon

  marks = layout.get("calibration_marks") or []
  by_page = {}
  for mark in marks:
    by_page.setdefault(int(mark.get("page", 1)), []).append(mark)
  parts_by_page = {}
  for name, entry in layout["parts"].items():
    parts_by_page.setdefault(int(entry.get("page", 1)), []).append(name)
  pages = sorted(set(by_page) | set(parts_by_page))

  pdf_path = os.path.splitext(path)[0] + ".pdf"
  # ! Delete any sheet left by a previous export first. A layout that used to need four
  # ! pages and now needs two would otherwise leave _p3/_p4 on disk, and printing those
  # ! stale pages would place parts by a paging that no longer exists.
  import glob as _glob
  stem = os.path.splitext(path)[0]
  for old_page in _glob.glob(stem + "_p*.svg") + [stem + ".svg"]:
    if os.path.exists(old_page):
      os.remove(old_page)

  usable_w, usable_h = A4_W - 2 * PAGE_MARGIN, A4_H - 2 * PAGE_MARGIN
  profiles = {n: part_profile_xy(e) for n, e in layout["parts"].items()}
  exact = sum(1 for v in profiles.values() if v)

  with PdfPages(pdf_path) as pdf:
    for page in pages:
      names = sorted(parts_by_page.get(page, []))
      page_marks_ = by_page.get(page, [])
      # Centre the sheet on everything it carries.
      pts = [corner_world_xy(layout["parts"][n]) for n in names]
      pts += [np.asarray([m["xy"]], dtype=float) for m in page_marks_]
      allpts = np.vstack(pts)
      centre = (allpts.min(axis=0) + allpts.max(axis=0)) / 2.0
      x0, y0 = centre[0] - usable_w / 2.0, centre[1] - usable_h / 2.0

      fig = Figure(figsize=(A4_W * INCHES_PER_M, A4_H * INCHES_PER_M))
      ax = fig.add_axes([PAGE_MARGIN / A4_W, PAGE_MARGIN / A4_H,
                         usable_w / A4_W, usable_h / A4_H])
      # The axes box is exactly the usable area and the limits are exactly that area in
      # metres, so one metre of layout is one metre of paper: the print is true scale.
      ax.set_xlim(x0, x0 + usable_w)
      ax.set_ylim(y0, y0 + usable_h)
      ax.set_aspect("equal", adjustable="box")
      ax.set_axis_off()

      for name in names:
        entry = layout["parts"][name]
        rings = profiles.get(name)
        if rings:
          for i, ring in enumerate(rings):
            ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="black",
                                    linewidth=1.0 if i == 0 else 0.6,
                                    linestyle="-" if i == 0 else (0, (3, 2))))
        else:
          c = corner_world_xy(entry)
          ax.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]),
                  color="black", linewidth=1.0)
        ax.text(entry["xy"][0], entry["xy"][1], name, ha="center", va="center",
                fontsize=7, color="0.35")

      # Calibration crosses: what the punch tip actually touches.
      # ! The arms are shortened near the page edge. A cross is pushed as far out as the
      # ! sheet allows (a long baseline makes the fitted turn accurate), which can leave it
      # ! ~14 mm from the paper edge -- and a fixed 12 mm arm would then run into the strip
      # ! most printers cannot print. A clipped arm would move the cross the operator aims
      # ! at, so the arm shrinks instead. The centre is what matters, not the arm length.
      for mark in page_marks_:
        mx, my = mark["xy"]
        edge = min(mx - x0, x0 + usable_w - mx, my - y0, y0 + usable_h - my)
        arm = float(np.clip(edge - 0.004, 0.003, 0.012))
        ax.plot([mx - arm, mx + arm], [my, my], color="red", linewidth=0.8)
        ax.plot([mx, mx], [my - arm, my + arm], color="red", linewidth=0.8)
        ax.add_patch(MplPolygon(_circle_xy(mx, my, 0.0035), closed=True, fill=False,
                                edgecolor="red", linewidth=0.6))
        short = mark["name"].split("-", 1)[-1]
        # Put the label on whichever side of the cross has room.
        below = my - y0 > 0.020
        ax.text(mx, my + (-arm - 0.003 if below else arm + 0.003), short, ha="center",
                va="top" if below else "bottom", fontsize=6, color="red")

      # 100 mm scale bar, to check the print came out at true size. Butt caps, or the
      # default projecting cap would add half a line width at each end.
      bar_x, bar_y = x0 + 0.012, y0 + 0.010
      ax.plot([bar_x, bar_x + 0.1], [bar_y, bar_y], color="black", linewidth=1.2,
              solid_capstyle="butt")
      for end in (bar_x, bar_x + 0.1):
        ax.plot([end, end], [bar_y - 0.003, bar_y + 0.003], color="black", linewidth=1.2)
      ax.text(bar_x + 0.05, bar_y + 0.004, "100 mm -- measure me after printing",
              ha="center", va="bottom", fontsize=6)

      ax.text(x0 + usable_w / 2, y0 + usable_h - 0.004,
              f"sheet {page} of {len(pages)}: {', '.join(names)}"
              f"   |   PRINT AT 100%, NOT 'FIT TO PAGE'   |   lay this sheet down on its own",
              ha="center", va="top", fontsize=6, color="0.35")
      # The robot-facing direction, so a sheet cannot be laid down backwards.
      ax.annotate("towards the robot", xy=(x0 + usable_w / 2, y0 + 0.026),
                  xytext=(x0 + usable_w / 2, y0 + 0.050),
                  ha="center", fontsize=6, color="0.35",
                  arrowprops=dict(arrowstyle="->", color="0.35", linewidth=0.8))

      pdf.savefig(fig)
      fig.savefig(f"{stem}_p{page}.svg")

  print(f"[layout] outline: {exact}/{len(layout['parts'])} part(s) traced from their exact "
        f"mesh profile" + ("" if exact == len(layout["parts"])
                           else " (the rest fall back to a bounding rectangle -- "
                                "`pip install shapely` for the exact profile)"))
  return len(pages)


def _circle_xy(cx, cy, r, n=24):
  """Points of a small circle, for drawing a ring round a calibration cross."""
  a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
  return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("plan", help="plan JSON saved by main.py --save")
  ap.add_argument("--out", default=None,
                  help="layout JSON to write (default: <plan dir>/layout_nominal.json)")
  ap.add_argument("--no-pack", action="store_true",
                  help="keep the plan's own part positions instead of packing them compactly "
                       "onto one sheet (the packed layout needs a fresh solve)")
  ap.add_argument("--svg", default=None,
                  help="also draw a true-scale outline sheet here (a PDF is written too)")
  args = ap.parse_args()

  layout = export_layout(args.plan, pack=not args.no_pack)
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

  for mark in layout.get("calibration_marks") or []:
    print(f"[layout] calibration mark {mark['name']:>10s}: "
          f"({mark['xy'][0]:.4f}, {mark['xy'][1]:.4f})")

  if args.svg:
    pages = write_outline_sheet(layout, args.svg)
    print(f"[layout] outline sheet: {pages} A4 page(s) at 1:1 -> "
          f"{os.path.splitext(args.svg)[0] + '.pdf'}  (PRINT AT 100%, NOT 'fit to page')")


if __name__ == "__main__":
  main()
