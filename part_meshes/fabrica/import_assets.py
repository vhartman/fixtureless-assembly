#!/usr/bin/env python3
"""Import Fabrica's common-frame OBJ assets as local visual and collision meshes.

Fabrica stores every part in the final assembly frame and uses centimetres. Assembly's
parts instead have a frame at their own AABB centre and use metres. This script performs
exactly that mechanical conversion and records enough metadata to reproduce/audit it.

Collision geometry is a contained convex decomposition of the same authored surface.
A part is still one rigid body: the generated hulls are merely several
collision geoms belonging to it.  This is important for these assets because one whole-part
convex hull fills wheel/peg holes, mortises and ducts. Both RAI and MuJoCo consume the same
generated hull list; the exact source mesh remains the renderer geometry.

Mate parents and insertion axes are intentionally *not* inferred here.  Those are task
semantics, not file-format facts, and remain reviewed in ``FabricaAssembly`` in
``assembly_planner/problem.py``.

Usage (from the assembly repository):
  python3 part_meshes/fabrica/import_assets.py \
      --source ../Fabrica/assets/fabrica
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import trimesh

# Leave numerical headroom above MuJoCo's minimum mesh volume (SI units).
_MIN_COLLISION_VOLUME = 1e-14

# The source CAD is nominal geometry, not a tolerance model.  In the car the inner wheel
# faces have only about 0.05 mm clearance from the body.  That is smaller than the accepted
# mate-pose residual and turns a visually valid seated wheel into hundreds of tiny convex
# contacts.  Preserve the visible outer face and retreat only the collision proxy's inner
# face by a modest manufacturing clearance.  The standard witness contract accepts as much
# as 1 mm of mate translation residual, so the proxy clearance must cover that *whole*
# envelope, with some numerical headroom; otherwise an IK witness can pass and the exact same
# assembled state can fail as soon as the mate contact exemption is removed.  The exact render
# mesh and mate transform stay untouched.  Each transformed cell is still certified inside the
# authored CAD below.
_AXIAL_FIT_TRIMS = {
    "car": {
        # source: (axis, inner face, authored interface coordinate,
        #          axial retreat, inserted-feature radial retreat), all in metres
        # The wheel hub crosses the interface into the body's bore and must remain collision.
        # Only cells belonging to the tread outside that plane are clipped.
        "0": (1, "min", +.0025, .0015, .0015),
        "3": (1, "max", -.0025, .0015, .0015),
        "4": (1, "max", -.0025, .0015, .0015),
        "5": (1, "min", +.0025, .0015, .0015),
    },
    "stool_circular": {
        # Four long legs are authored line-to-line with both the base sockets and their
        # end caps.  A 2.5mm local manufacturing clearance covers the certified mate-pose
        # residual, including its angular component.  The 0.5mm, 1mm, 1.5mm and 2mm
        # candidates still left accumulated attachment-chain contacts; at 2mm the two
        # limiting hull pairs overlap by 0.25--0.28mm even with a 40um translation
        # certificate.  The exact render mesh, exposed leg body and mate transform remain
        # untouched.
        "1": (2, "min", +.0475, .0025, .0025),
        "3": (2, "min", +.0475, .0025, .0025),
        "5": (2, "min", +.0475, .0025, .0025),
        "7": (2, "min", +.0475, .0025, .0025),
    },
}

# These four CAD parts are surfaces of revolution: a rounded wheel tread and one coaxial
# locating peg.  Tetrahedralizing each one into ~106 convex cells is both redundant and very
# expensive for sampling-based planning (four wheels turn one predicate call into hundreds of
# FCL pairs).  Two contained cylinders preserve the collision-relevant outer tread and the
# collidable peg.  The peg radius carries the same mating clearance as the general fit trim.
_CAR_WHEEL_PRIMITIVES = {
    "0": ("min", +.0025),
    "3": ("max", -.0025),
    "4": ("max", -.0025),
    "5": ("min", +.0025),
}

# The four manifold fasteners are threaded CAD with thousands of tiny facets.  Their
# collision-relevant insertion feature is the central peg; modelling every thread crest
# makes a nominal mate line-to-line and multiplies collision pairs. Keep the exact CAD for
# rendering and grasp sampling, but use one deliberately filled cylinder with manufacturing
# clearance. The CAD thread has helical voids, so this is an explicit user-approved proxy,
# not an inner approximation; the independent insertion audit remains mandatory.
_COOLING_PEG_PRIMITIVES = {"2", "3", "5", "6"}

# Collision proxies for authored male parts whose mating geometry is a cylinder or a
# rectangular tenon.  Keeping their exact threaded/sculpted CAD as 10--40 convex cells is
# expensive but adds no useful collision information to an insertion query.  The visual and
# grasp meshes remain the exact OBJ; only the rigid collision proxy is simplified.  Dimensions
# below describe the nominal CAD envelope.  Manufacturing clearance is applied in a separate,
# feature-local pass below rather than by scaling the complete part.
_SEMANTIC_INSERT_PRIMITIVES = {
    "duct": {
        # Six identical tilted screws: hex head followed by the male cylindrical shaft.
        source: {"kind": "headed_pin", "axis": [-.5, 0., -np.sqrt(.75)],
                 "head": [-.0174, -.0005, .0110, 6],
                 "stem": [-.0010, .0192, .0060, 32]}
        for source in ("2", "3", "4", "5", "6", "7")
    },
    "beam": {
        # Rectangular uprights with a long visible body and a tenon at either end.
        source: {"kind": "double_tenon", "axis": 2,
                 "body": [-.0245, .0245, [.0113, .0118]],
                 "lower": [-.0380, -.0240, [.0090, .0090]],
                 "upper": [.0240, .0380, [.0090, .0090]]}
        for source in ("2", "3")
    },
    "gamepad": {
        # The two small axisymmetric pins are unambiguous. The larger sticks and curved side
        # connector retain their contained decompositions: replacing those non-prismatic
        # profiles with bounding primitives would fill their actual mating clearance.
        "0": {"kind": "stepped_pin", "axis": 2,
              "segments": [[-.0140, .0005, .0048, 32],
                           [.0000, .0140, .0058, 32]]},
        "5": {"kind": "stepped_pin", "axis": 2,
              "segments": [[-.0140, .0005, .0048, 32],
                           [.0000, .0140, .0058, 32]]},
    },
    "plumbers_block": {
        # Source 0 is a pipe-like horizontal insert; its decorative axial bore is irrelevant
        # to assembly collision. Sources 1/4 are identical headed vertical pins.
        "0": {"kind": "stepped_pin", "axis": 1,
              "segments": [[-.0395, .0395, .0205, 32]]},
        "1": {"kind": "headed_pin", "axis": 2,
              "stem": [-.0345, .0110, .0045, 32],
              "head": [.0105, .0345, .0120, 32]},
        "4": {"kind": "headed_pin", "axis": 2,
              "stem": [-.0345, .0110, .0045, 32],
              "head": [.0105, .0345, .0120, 32]},
    },
}

# Clearance is deliberately expressed per *male feature*, never as a whole-part scale.
# ``radial`` retreats a cylinder radius (or each side of a tenon's transverse box), while
# ``axial`` retreats only the leading end of an inserted shaft.  The larger beam value is
# required by its authored tapered/dovetail mortise: the nominal rectangular tenon proxy
# otherwise overlaps the exact female surface by roughly 2 mm.  All values are recorded in
# the generated manifest and verified by tools/diagnostics/audit_fabrica.py against both
# collision backends.
_SEMANTIC_INSERT_CLEARANCE = {
    "duct": {source: {"radial": .0010, "axial": .0005, "leading": "upper"}
             for source in ("2", "3", "4", "5", "6", "7")},
    "beam": {source: {"radial": .0020, "axial": .0010}
             for source in ("2", "3")},
    "gamepad": {
        "0": {"radial": .0010, "axial": .0005, "leading": "lower"},
        "5": {"radial": .0010, "axial": .0005, "leading": "lower"},
    },
    "plumbers_block": {
        "0": {"radial": .0010, "axial": .0005, "leading": "upper"},
        "1": {"radial": .0010, "axial": .0005, "leading": "lower"},
        "4": {"radial": .0010, "axial": .0005, "leading": "lower"},
    },
}

# The manifold body's four corner sockets are exact annuli centred at +/-50mm.  A sparse
# contained decomposition can retain enough *volume* while leaving those thin rings visually
# and collision-wise discontinuous.  Eight contained convex wedges per socket preserve the
# complete insertion feature without filling its bore.
_COOLING_SOCKET_CENTRES = tuple(
    (x, y) for x in (-.05, .05) for y in (-.05, .05))

# Local manufacturing clearance for authored male insertion features.  These are deliberately
# feature intervals, not whole-part scales: exposed collision geometry keeps its authored size.
# Beam sources 2/3 are symmetric posts whose narrow tenons begin outside +/-24.5 mm.  The
# nominal Fabrica CAD is a detailed dovetail-like, effectively line-to-line joint. The
# semantic shoulder is clipped and the male tenon is retreated by 1.5 mm; the exposed central
# post body remains exact and the independent insertion audit is authoritative.
_AXIAL_END_RETREATS = {
    "beam": {
        "2": (2, -.0245, .0245, .0015),
        "3": (2, -.0245, .0245, .0015),
    },
}


def _cylinder_along_axis(radius, lower, upper, axis=1, sections=32):
  cylinder = trimesh.creation.cylinder(radius=radius, height=upper - lower,
                                       sections=sections)
  if axis != 2:
    rotation_axis = [0., 1., 0.] if axis == 0 else [1., 0., 0.]
    cylinder.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2,
                                                                      rotation_axis))
  centre = np.zeros(3)
  centre[axis] = (lower + upper) / 2
  cylinder.apply_translation(centre)
  return cylinder.convex_hull


def _cylinder_along_vector(radius, lower, upper, axis, sections=32):
  """Create a cylinder on an arbitrary part-frame axis through the origin."""
  if isinstance(axis, (int, np.integer)):
    return _cylinder_along_axis(radius, lower, upper, int(axis), sections)
  direction = np.asarray(axis, dtype=float)
  direction /= np.linalg.norm(direction)
  cylinder = trimesh.creation.cylinder(radius=radius, height=upper - lower,
                                       sections=sections)
  transform = trimesh.geometry.align_vectors([0., 0., 1.], direction)
  cylinder.apply_transform(transform)
  cylinder.apply_translation(direction * ((lower + upper) / 2.))
  return cylinder.convex_hull


def _box_along_axis(lower, upper, transverse_extents, axis):
  """Create an axis-aligned box described by one axial and two transverse extents."""
  extents = np.empty(3)
  extents[axis] = upper - lower
  transverse = [i for i in range(3) if i != axis]
  extents[transverse] = transverse_extents
  box = trimesh.creation.box(extents)
  centre = np.zeros(3)
  centre[axis] = (lower + upper) / 2.
  box.apply_translation(centre)
  return box.convex_hull


def _semantic_insert_collision(spec, clearance=None):
  """Build a reviewed proxy, retreating only its explicitly identified male feature."""
  kind, axis = spec["kind"], spec["axis"]
  clearance = clearance or {"radial": 0., "axial": 0.}
  radial = float(clearance.get("radial", 0.))
  axial = float(clearance.get("axial", 0.))
  leading = clearance.get("leading")

  def male_interval(lower, upper):
    if leading == "lower":
      lower += axial
    elif leading == "upper":
      upper -= axial
    elif leading is not None:
      raise ValueError(f"invalid insertion leading end {leading!r}")
    return lower, upper

  hulls = []
  if kind in ("stepped_pin", "headed_pin"):
    segments = spec.get("segments")
    if segments is None:
      segments = [spec["stem"], spec["head"]]
    # Segment zero is the male shaft; following segments are exposed heads/bodies.
    for index, (lower, upper, radius, sections) in enumerate(segments):
      if index == 0:
        lower, upper = male_interval(lower, upper)
        radius -= radial
      if radius <= 0 or lower >= upper:
        raise ValueError("semantic insert clearance removes a cylindrical feature")
      hulls.append(_cylinder_along_vector(radius, lower, upper, axis, sections))
  elif kind == "stepped_box":
    if not isinstance(axis, (int, np.integer)):
      raise ValueError("stepped boxes currently require a coordinate axis")
    # The final, narrow segment is the male connector tongue.
    for index, (lower, upper, extents) in enumerate(spec["segments"]):
      extents = np.asarray(extents, dtype=float)
      if index == len(spec["segments"]) - 1:
        lower, upper = male_interval(lower, upper)
        extents -= 2. * radial
      if np.any(extents <= 0) or lower >= upper:
        raise ValueError("semantic insert clearance removes a rectangular feature")
      hulls.append(_box_along_axis(lower, upper, extents, int(axis)))
  elif kind == "double_tenon":
    if not isinstance(axis, (int, np.integer)):
      raise ValueError("double tenons currently require a coordinate axis")
    for name in ("body", "lower", "upper"):
      lower, upper, extents = spec[name]
      extents = np.asarray(extents, dtype=float)
      if name != "body":
        extents -= 2. * radial
        if name == "lower":
          lower += axial
        else:
          upper -= axial
      if np.any(extents <= 0) or lower >= upper:
        raise ValueError("semantic insert clearance removes a tenon")
      hulls.append(_box_along_axis(lower, upper, extents, int(axis)))
  else:
    raise ValueError(f"unknown semantic insertion primitive {kind!r}")
  settings = {
      "primitive": kind,
      "explicit_approximation": True,
      "approximation_reason": "reviewed semantic male insertion feature",
      "axis": axis,
      "nominal_spec": spec,
      "manufacturing_clearance_m": {"radial": radial, "axial": axial},
      "clearance_scope": "male insertion feature only; visual/body/head unchanged",
      "minimum_volume_fraction": .05,
  }
  return hulls, settings


def _partitioned_cylinder(radius, lower, upper, axis=1, sections=32, axial_bands=1):
  """Partition a polygonal cylinder into convex triangular prisms.

  The cells tile exactly the same occupied polygonal volume as
  :func:`_cylinder_along_axis`.  This is intended only for controlled collision-cost
  ablations: it changes component count without changing the proxy envelope.
  """
  transverse = [i for i in range(3) if i != axis]
  levels = np.linspace(lower, upper, axial_bands + 1)
  angles = np.linspace(0., 2. * np.pi, sections + 1)
  cells = []
  for band_lo, band_hi in zip(levels[:-1], levels[1:]):
    for angle_lo, angle_hi in zip(angles[:-1], angles[1:]):
      points = []
      for along in (band_lo, band_hi):
        centre = np.zeros(3)
        centre[axis] = along
        points.append(centre)
        for angle in (angle_lo, angle_hi):
          point = centre.copy()
          point[transverse[0]] = radius * np.cos(angle)
          point[transverse[1]] = radius * np.sin(angle)
          points.append(point)
      cells.append(trimesh.convex.convex_hull(np.asarray(points)))
  return cells


def _annular_prisms(centres, inner_radius, outer_radius, lower, upper, sections=8):
  """Return convex wedge prisms tiling a conservative polygonal annulus."""
  cells = []
  for cx, cy in centres:
    for i in range(sections):
      angles = (2. * np.pi * i / sections, 2. * np.pi * (i + 1) / sections)
      xy = []
      for angle in angles:
        xy.append((cx + inner_radius * np.cos(angle),
                   cy + inner_radius * np.sin(angle)))
        xy.append((cx + outer_radius * np.cos(angle),
                   cy + outer_radius * np.sin(angle)))
      # Order does not matter to the 3-D convex hull. With inner_radius=7.6mm and eight
      # sectors, the inner chord remains outside 7.0mm; the authored bore is never filled.
      points = np.asarray([[x, y, z] for z in (lower, upper) for x, y in xy])
      cells.append(trimesh.convex.convex_hull(points))
  return cells


def _car_wheel_collision(mesh, inner_face, interface, clearance=.0015,
                         components=2):
  """Return a feature-aware two-cylinder proxy, optionally partitioned for ablation."""
  axis = 1
  lo, hi = map(float, mesh.bounds[:, axis])
  transverse = [i for i in range(3) if i != axis]
  outer_radius = float(np.max(np.linalg.norm(mesh.vertices[:, transverse], axis=1)))
  # The CAD's constant-radius barrel occupies the middle 10 mm.  Keeping it inset by 0.5 mm
  # radially avoids representing the rounded shoulder as an over-approximation.
  tread_radius = outer_radius - .0005
  shoulder = .0025
  if inner_face == "min":
    tread_lo, tread_hi = interface + shoulder, hi - shoulder
    peg_lo, peg_hi = lo + 5e-5, interface - 5e-5
  else:
    tread_lo, tread_hi = lo + shoulder, interface - shoulder
    peg_lo, peg_hi = interface + 5e-5, hi - 5e-5
  peg_radius = .005 - clearance
  if not (tread_lo < tread_hi and peg_lo < peg_hi and peg_radius > 0):
    raise ValueError("invalid analytic car-wheel dimensions")
  if components == 2:
    hulls = [
        _cylinder_along_axis(tread_radius, tread_lo, tread_hi, axis),
        _cylinder_along_axis(peg_radius, peg_lo, peg_hi, axis),
    ]
    partition = "none"
  elif components == 96:
    # 2 axial bands * 32 angular cells for the tread, plus 32 angular cells for the peg.
    # This is close to the old decomposition's ~106 cells per wheel while occupying the
    # exact same polygonal cylinders as the production two-component proxy.
    hulls = (_partitioned_cylinder(tread_radius, tread_lo, tread_hi, axis, 32, 2)
             + _partitioned_cylinder(peg_radius, peg_lo, peg_hi, axis, 32, 1))
    partition = "triangular-prisms"
  else:
    raise ValueError("car wheel proxy supports exactly 2 or 96 components")
  settings = {
      "primitive": "two_coaxial_cylinders",
      "axis": axis,
      "tread_radius_m": tread_radius,
      "tread_interval_m": [tread_lo, tread_hi],
      "peg_radius_m": peg_radius,
      "peg_interval_m": [peg_lo, peg_hi],
      "mate_translation_envelope_m": clearance,
      "sections": 32,
      "component_partition": partition,
      "component_count": components,
  }
  ratio = abs(float(mesh.convex_hull.volume)) / max(abs(float(mesh.volume)), 1e-15)
  return hulls, settings, ratio


def _retreat_axial_ends(hulls, axis, central_lower, central_upper, amount):
  """Retreat two male end features and separate cells which bridge their shoulders."""
  transverse = [i for i in range(3) if i != axis]
  changed = 0
  shoulder_clips = 0

  def clipped_halfspace(hull, threshold, keep_above):
    vertices = np.asarray(hull.vertices)
    sign = 1.0 if keep_above else -1.0
    signed = sign * (vertices[:, axis] - threshold)
    points = list(vertices[signed >= -1e-12])
    for edge in hull.edges_unique:
      a, b = map(int, edge)
      if (signed[a] < 0) == (signed[b] < 0):
        continue
      t = signed[a] / (signed[a] - signed[b])
      points.append(vertices[a] + t * (vertices[b] - vertices[a]))
    points = np.unique(np.round(np.asarray(points), 14), axis=0)
    if len(points) < 4 or np.linalg.matrix_rank(points - points.mean(0), tol=1e-12) < 3:
      return None
    return trimesh.convex.convex_hull(points)

  result = []
  for original in hulls:
    hull = original
    lo, hi = map(float, hull.bounds[:, axis])
    if lo < central_lower < hi:
      hull = clipped_halfspace(hull, central_lower + amount, True)
      shoulder_clips += 1
    elif lo < central_upper < hi:
      hull = clipped_halfspace(hull, central_upper - amount, False)
      shoulder_clips += 1
    elif hi <= central_lower or lo >= central_upper:
      centre = np.asarray(hull.vertices).mean(axis=0)
      local = hull.vertices[:, transverse] - centre[transverse]
      radius = float(np.linalg.norm(local, axis=1).max())
      if radius > amount:
        # Contract inside the already-certified convex cell. Scaling about its own centre is
        # guaranteed to remain a subset of that cell even for asymmetric dovetail pieces.
        hull.vertices[:, transverse] = (centre[transverse]
                                        + (radius - amount) / radius * local)
        changed += 1
    if hull is not None:
      result.append(hull)
  hulls[:] = result
  if not changed:
    raise ValueError("axial-end retreat did not identify any feature cells")
  return {
      "axis": int(axis),
      "central_interval_m": [float(central_lower), float(central_upper)],
      "transverse_retreat_m": float(amount),
      "retreated_hulls": changed,
      "shoulder_clips": shoulder_clips,
  }

_ACD_COMMON = {
    "minimumVolumePercentErrorAllowed": 0.5,
    "maxRecursionDepth": 12,
    "shrinkWrap": True,
    "fillMode": "flood",
    "maxNumVerticesPerCH": 64,
    # Parallel ACD can return its hulls in scheduler-dependent order.  Stable files are part
    # of the saved scene contract, so keep the importer deterministic.
    "asyncACD": False,
}


def _vhacd_parts(mesh):
  """Return (hulls, settings, hull/mesh volume ratio).

  Detail is selected from topology/concavity rather than benchmark-part names. Genus-bearing
  parts and strongly concave shells need enough components to preserve bores and openings;
  lower caps visibly bridge those voids even though their outer AABB looks accurate. The
  resulting cost is intentional and measured later--collision fidelity must not be silently
  traded away to make planning faster.
  """
  volume = max(abs(float(mesh.volume)), 1e-15)
  ratio = abs(float(mesh.convex_hull.volume)) / volume
  strongly_concave = ratio > 3.0
  genus_bearing = bool(mesh.is_watertight and mesh.euler_number < 2)
  if strongly_concave:
    max_hulls, resolution = 128, 1_000_000
  elif genus_bearing:
    max_hulls, resolution = 96, 750_000
  elif ratio > 1.4:
    # Moderately concave bored/hubbed parts need enough proposals that the subsequent
    # contained contraction does not discard a large fraction of their physical boundary.
    max_hulls, resolution = 96, 750_000
  else:
    max_hulls, resolution = 32, 250_000
  settings = dict(_ACD_COMMON)
  settings.update({
      "maxConvexHulls": max_hulls,
      "resolution": resolution,
  })
  pieces = mesh.convex_decomposition(**settings)
  if not isinstance(pieces, list):
    pieces = [pieces]
  # Do not depend on VHACD's output order.  Sort spatially, then by size, and explicitly
  # hull every result so RAI and MuJoCo see exactly the same convex geometry.
  hulls = []
  for piece in pieces:
    if len(piece.vertices) < 4 or len(piece.faces) < 4:
      continue
    hull = piece.convex_hull
    # VHACD can emit a zero-thickness sliver for duplicated/coplanar CAD triangles.  RAI's
    # FCL bridge quite correctly refuses such a collision object, so omit it at import time.
    span = max(float(np.prod(np.maximum(hull.extents, 1e-15))), 1e-15)
    if np.linalg.matrix_rank(hull.vertices - hull.vertices.mean(axis=0), tol=1e-10) < 3:
      continue
    if abs(float(hull.volume)) <= span * 1e-8:
      continue
    hulls.append(hull)
  hulls.sort(key=lambda h: tuple(np.round(np.r_[h.centroid, h.volume], 12)))
  if not hulls:
    raise RuntimeError("convex decomposition produced no non-degenerate hulls")
  return hulls, settings, ratio


def _part_key(path: Path):
  try:
    return (0, int(path.stem))
  except ValueError:
    return (1, path.stem)


def _trim_inner_face(hulls, axis, inner_face, interface, amount, radial_retreat=0.0,
                     radial_center=None):
  """Clip one axial face while preserving the opposite, externally visible face.

  Clipping, rather than anisotropic scaling, matters for a bored/tapered part: moving an
  existing surface can cross a concavity even though both endpoints lie inside the CAD.
  Intersecting each convex cell with one half-space can only remove material.
  """
  import trimesh
  lo = min(float(h.bounds[0, axis]) for h in hulls)
  hi = max(float(h.bounds[1, axis]) for h in hulls)
  if inner_face not in ("min", "max") or not 0 < amount < hi - lo:
    raise ValueError(f"invalid axial fit trim face={inner_face!r}, amount={amount:g}")
  threshold = interface + amount if inner_face == "min" else interface - amount
  sign = 1.0 if inner_face == "min" else -1.0
  inserted = [h for h in hulls
              if ((inner_face == "min" and h.bounds[0, axis] < interface - 1e-5) or
                  (inner_face == "max" and h.bounds[1, axis] > interface + 1e-5))]
  transverse = [i for i in range(3) if i != axis]
  radial_center = np.zeros(2) if radial_center is None else np.asarray(radial_center, dtype=float)
  if radial_center.shape != (2,):
    raise ValueError("radial fit-trim centre must have two transverse coordinates")
  radial_extent = max((float(np.linalg.norm(
      h.vertices[:, transverse] - radial_center, axis=1).max())
                       for h in inserted), default=0.0)
  if radial_retreat and not 0 < radial_retreat < radial_extent:
    raise ValueError(f"radial fit trim {radial_retreat:g} must lie in (0, {radial_extent:g})")
  radial_scale = ((radial_extent - radial_retreat) / radial_extent
                  if radial_retreat else 1.0)
  clipped = []
  for hull in hulls:
    # A pin/hub may cross the seating plane. It is a distinct physical feature and remains
    # collidable; only components wholly on the tread/flange side receive the axial gap.
    if ((inner_face == "min" and hull.bounds[0, axis] < interface - 1e-5) or
        (inner_face == "max" and hull.bounds[1, axis] > interface + 1e-5)):
      hull.vertices[:, transverse] = (radial_center + radial_scale *
                                       (hull.vertices[:, transverse] - radial_center))
      clipped.append(hull)
      continue
    vertices = np.asarray(hull.vertices)
    signed = sign * (vertices[:, axis] - threshold)
    points = list(vertices[signed >= -1e-12])
    for edge in hull.edges_unique:
      a, b = int(edge[0]), int(edge[1])
      if (signed[a] < 0) == (signed[b] < 0):
        continue
      t = signed[a] / (signed[a] - signed[b])
      points.append(vertices[a] + t * (vertices[b] - vertices[a]))
    if len(points) < 4:
      continue
    points = np.unique(np.round(np.asarray(points), 14), axis=0)
    if len(points) < 4 or np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-12) < 3:
      continue
    clipped.append(trimesh.convex.convex_hull(points))
  if not clipped:
    raise ValueError("axial fit trim removed every collision cell")
  hulls[:] = clipped
  return {"axis": int(axis), "inner_face": inner_face, "retreat_m": float(amount),
          "interface_coordinate_m": float(interface), "clip_coordinate_m": float(threshold),
          "inserted_feature_radial_retreat_m": float(radial_retreat),
          "inserted_feature_original_radius_m": float(radial_extent)}


def compact_existing(root, model, *, merge=True):
  """Compact an already contained import without recomputing its tetrahedra."""
  from tetra_collision import BoundaryCheck, merge_contained, solid_contains
  directory = root / model
  manifest_path = directory / "manifest.json"
  metadata = json.loads(manifest_path.read_text())
  for part, record in metadata["parts"].items():
    collision = record["collision"]
    if not collision["method"].endswith("-inner"):
      raise ValueError("--compact-existing requires contained (-inner) assets")
    mesh = trimesh.load(directory / record["file"], force="mesh", process=True)
    if not mesh.is_watertight:
      mesh.merge_vertices(digits_vertex=7, merge_norm=True, merge_tex=True)
      mesh.update_faces(mesh.nondegenerate_faces())
      mesh.update_faces(mesh.unique_faces())
      mesh.remove_unreferenced_vertices()
    check = BoundaryCheck(mesh)
    def approved(hull):
      return any(np.all(hull.bounds[0] >= np.array(box["lower"]) - 1e-8) and
                 np.all(hull.bounds[1] <= np.array(box["upper"]) + 1e-8)
                 for box in collision.get("approximation_boxes", []))
    scene = trimesh.load(directory / collision["hulls"][0]["file"], force="scene")
    hulls = []
    repaired = 0
    outside_omitted = 0
    for rec in collision["hulls"]:
      # Downstream `_load_mesh` explicitly convex-hulls every GLB geometry. Reproduce that
      # exact object here; checking the pre-hull serialization can disagree for duplicated
      # or nearly coplanar GLB triangles and make refit require repeated passes.
      hull = scene.geometry[rec["geometry"]].copy().convex_hull
      hull.apply_translation(np.array(rec["pos"]) - hull.bounds.mean(axis=0))
      if not approved(hull) and not bool(solid_contains(mesh, [hull.vertices.mean(axis=0)])[0]):
        # Old importers only checked boundary crossings. A convex cell wholly inside a
        # cavity has no crossing but is still not part of the solid. It cannot be repaired by
        # contraction, so omit it and let the retained-volume gate below decide whether the
        # cached decomposition remains usable.
        outside_omitted += 1
        continue
      if not approved(hull) and check.intersects(hull):
        center, vertices = hull.vertices.mean(axis=0), hull.vertices.copy()
        lo, hi = 0., 1.
        for _ in range(16):
          mid = (lo + hi) / 2
          hull.vertices = center + mid * (vertices - center)
          if check.intersects(hull):
            hi = mid
          else:
            lo = mid
        hull.vertices = center + lo * .999 * (vertices - center)
        if lo < .001 or check.intersects(hull):
          raise ValueError(f"{model}/{part}: cannot fit staged cell inside exact CAD")
        repaired += 1
      hulls.append(hull)
    cell_volume_sum_ratio = sum(abs(h.volume) for h in hulls) / abs(mesh.volume)
    if cell_volume_sum_ratio < .8:
      raise ValueError(f"{model}/{part}: refitting retains only {cell_volume_sum_ratio:.1%}")
    print(f"[compact] {model}/{part}: refitted {repaired}, omitted {outside_omitted} "
          "cavity/outside cells against solid-membership + convex/triangle gate", flush=True)
    if merge:
      hulls = merge_contained(hulls, mesh)
    geometries, records = {}, []
    omitted_volume = 0.
    for i, hull in enumerate(hulls):
      center = hull.bounds.mean(axis=0)
      hull.apply_translation(-center)
      vertices, pivot = hull.vertices.copy(), hull.vertices.mean(axis=0)
      for attempt in range(21):
        hull.vertices = np.asarray(pivot + (.999 ** (attempt + 1)) * (vertices - pivot),
                                   dtype=np.float32)
        check_hull = hull.copy()
        check_hull.apply_translation(center)
        if approved(check_hull) or not check.intersects(check_hull):
          break
      else:
        omitted_volume += abs(hull.volume)
        continue
      # _load_mesh recentres the stored float32 AABB. Compensate for its tiny
      # rounding offset so the consumer sees exactly the geometry checked above.
      if abs(hull.volume) < _MIN_COLLISION_VOLUME:
        omitted_volume += abs(hull.volume)
        continue
      center = center + hull.bounds.mean(axis=0)
      name = f"hull_{i:03d}"
      geometries[name] = hull
      records.append({"file": f"collision/{part}.glb", "geometry": name,
                      "pos": center.tolist()})
    if omitted_volume:
      print(f"[compact] {model}/{part}: omitted degenerate/unrepresentable cells "
            f"({omitted_volume / abs(mesh.volume):.8%} of CAD volume)", flush=True)
    if cell_volume_sum_ratio - omitted_volume / abs(mesh.volume) < .8:
      raise ValueError(f"{model}/{part}: float32 representation loses too much CAD volume")
    path = directory / "collision" / (part + ".glb")
    trimesh.Scene(geometries).export(path)
    collision.update(hulls=records, hull_count=len(records),
                     asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if merge:
      collision["settings"]["contained_merge_passes"] = 12
      collision["settings"]["contained_merge_neighbours"] = 32
    collision["settings"]["premerge_cell_volume_sum_fraction"] = cell_volume_sum_ratio
    collision["settings"]["boundary_method"] = "triangle-convex-halfspace-clipping"
    collision["settings"]["outside_cells_omitted"] = outside_omitted
    collision["settings"]["boundary_tolerance_m"] = 1e-12
    collision["settings"]["minimum_cell_volume_m3"] = _MIN_COLLISION_VOLUME
    collision["decomposed_volume_ratio"] = sum(abs(h.volume) for h in hulls) / abs(mesh.volume)
    # Keep the manifest consistent after each complete part.
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _import_model(source_root: Path, output_root: Path, model: str,
                 method="tetra-inner", tetra_faces=6000, part_methods=None,
                 wheel_proxy_components=2, only_parts=None,
                 min_retained_volume=.8):
  source_dir = source_root / model
  if not source_dir.is_dir():
    raise FileNotFoundError(f"Fabrica model directory does not exist: {source_dir}")
  files = sorted(source_dir.glob("*.obj"), key=_part_key)
  if not files:
    raise ValueError(f"no OBJ parts in {source_dir}")
  if only_parts is not None:
    requested = set(only_parts)
    available = {path.stem for path in files}
    missing = requested - available
    if missing:
      raise ValueError(f"unknown {model} source part(s): {sorted(missing)}")
    files = [path for path in files if path.stem in requested]
  out_dir = output_root / model
  out_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = out_dir / "manifest.json"
  if manifest_path.exists():
    metadata = json.loads(manifest_path.read_text())
  else:
    metadata = {
        "source": "https://github.com/yunshengtian/Fabrica",
        "source_model": f"assets/fabrica/{model}",
        "source_units": "centimetres",
        "output_units": "metres",
        "scale": 0.01,
        "centering": "axis-aligned bounding-box centre",
        "parts": {},
    }
  default_method = method
  for path in files:
    method = (part_methods or {}).get(path.stem, default_method)
    print(f"[{method}] {model}/{path.name}", flush=True)
    raw = path.read_bytes()
    mesh = trimesh.load(path, force="mesh", process=False, maintain_order=True)
    bounds_cm = np.asarray(mesh.bounds, dtype=float)
    center_cm = bounds_cm.mean(axis=0)
    mesh.apply_translation(-center_cm)
    mesh.apply_scale(0.01)
    target = out_dir / path.name
    mesh.export(target, file_type="obj", include_normals=True, include_color=False)
    collision_root = out_dir / "collision"
    collision_dir = collision_root / path.stem
    if collision_dir.exists():                            # old one-file-per-hull layout
      shutil.rmtree(collision_dir)
    collision_root.mkdir(parents=True, exist_ok=True)
    collision_path = collision_root / f"{path.stem}.glb"
    # Process a copy for ACD (weld duplicate vertices / orient faces) without changing the
    # renderer mesh above. Each hull's part-frame offset is recorded explicitly below.
    # Validate against the exported OBJ that downstream tools actually read, including
    # its decimal quantization. Checking only the higher-precision source can miss tiny
    # boundary crossings after serialization.
    collision_source = trimesh.load(target, force="mesh", process=False)
    collision_source.process(validate=True)
    # Some CAD exports duplicate seam vertices with sub-micrometre roundoff.
    # Welding at 0.1 micrometre closes those seams without filling actual holes.
    if not collision_source.is_watertight:
      collision_source.merge_vertices(digits_vertex=7, merge_norm=True, merge_tex=True)
      collision_source.update_faces(collision_source.nondegenerate_faces())
      collision_source.update_faces(collision_source.unique_faces())
      collision_source.remove_unreferenced_vertices()
    wheel_primitive = (_CAR_WHEEL_PRIMITIVES.get(path.stem) if model == "car" else None)
    cooling_peg = model == "cooling_manifold" and path.stem in _COOLING_PEG_PRIMITIVES
    semantic_insert = _SEMANTIC_INSERT_PRIMITIVES.get(model, {}).get(path.stem)
    semantic_clearance = _SEMANTIC_INSERT_CLEARANCE.get(model, {}).get(path.stem)
    if wheel_primitive is not None:
      hulls, acd, convexity_ratio = _car_wheel_collision(
          collision_source, *wheel_primitive, components=wheel_proxy_components)
      method = "primitive-inner"
    elif cooling_peg:
      # Preserve both semantically distinct features.  The narrow stem is the insertion
      # geometry; the wider hexagonal head remains collidable after seating.
      hulls = [
          _cylinder_along_axis(.0035, -.0174, -.0024, axis=2, sections=32),
          _cylinder_along_axis(.0090, -.0026, .0174, axis=2, sections=6),
      ]
      acd = {
          "primitive": "central_peg_cylinder_plus_hex_head",
          "explicit_approximation": True,
          "approximation_reason": "user-approved filled threaded peg",
          "axis": 2,
          "stem_radius_m": .0035,
          "stem_interval_m": [-.0174, -.0024],
          "head_radius_m": .0090,
          "head_interval_m": [-.0026, .0174],
          "head_sections": 6,
          "mate_translation_envelope_m": .0015,
          "sections": 32,
          "minimum_volume_fraction": .3,
      }
      convexity_ratio = abs(float(collision_source.convex_hull.volume)) / max(
          abs(float(collision_source.volume)), 1e-15)
      method = "primitive-approved"
    elif semantic_insert is not None:
      hulls, acd = _semantic_insert_collision(semantic_insert, semantic_clearance)
      convexity_ratio = abs(float(collision_source.convex_hull.volume)) / max(
          abs(float(collision_source.volume)), 1e-15)
      method = "primitive-approved"
    elif method.endswith("-inner"):
      from tetra_collision import inner_decompose
      if method == "vhacd-inner":
        proposals, acd, _ = _vhacd_parts(collision_source)
        hulls, acd, convexity_ratio = inner_decompose(
            collision_source, tetra_faces, min_volume_fraction=min_retained_volume,
            hulls=proposals, settings=acd)
      else:
        hulls, acd, convexity_ratio = inner_decompose(
            collision_source, tetra_faces, min_volume_fraction=min_retained_volume)
    else:
      hulls, acd, convexity_ratio = _vhacd_parts(collision_source)
    if model == "cooling_manifold" and path.stem == "1":
      socket_hulls = _annular_prisms(
          _COOLING_SOCKET_CENTRES, .0076, .0093, -.0170, -.0030, sections=8)
      hulls.extend(socket_hulls)
      acd["semantic_socket_rings"] = {
          "count": 4,
          "sections_each": 8,
          "centres_m": [list(x) for x in _COOLING_SOCKET_CENTRES],
          "inner_radius_m": .0076,
          "outer_radius_m": .0093,
          "interval_m": [-.0170, -.0030],
          "containment": "every serialized wedge is checked against exact CAD",
      }
    fit_trim = (_AXIAL_FIT_TRIMS.get(model, {}).get(path.stem)
                if wheel_primitive is None else None)
    if fit_trim:
      acd["axial_fit_trim"] = _trim_inner_face(hulls, *fit_trim)
    end_retreat = (_AXIAL_END_RETREATS.get(model, {}).get(path.stem)
                   if semantic_insert is None else None)
    if end_retreat:
      acd["axial_end_retreat"] = _retreat_axial_ends(hulls, *end_retreat)
    print(f"[{method}] {model}/{path.name}: {len(hulls)} hulls", flush=True)
    collision_hulls = []
    collision_geometry = {}
    omitted_volume = 0.0
    if method.endswith("-inner"):
      from tetra_collision import BoundaryCheck
      boundary_check = BoundaryCheck(collision_source)
    for i, hull in enumerate(hulls):
      # Assembly's generic mesh shape contract recentres each file on its AABB.  Preserve
      # the decomposition component's position in the *part* frame explicitly; otherwise
      # every hull gets piled at the part origin in both RAI and MuJoCo.
      hull_center = np.asarray(hull.bounds, dtype=float).mean(axis=0)
      exported = hull.copy()
      exported.apply_translation(-hull_center)
      if method.endswith("-inner"):
        # GLB stores local float32 vertices. Validate that exact representation,
        # including its part-frame offset; repeat contraction if rounding touches CAD.
        local = exported.vertices.copy()
        local_center = exported.vertices.mean(axis=0)
        for attempt in range(21):
          exported.vertices = np.asarray(local_center + (.999 ** attempt) *
                                         (local - local_center), dtype=np.float32)
          candidate = exported.copy()
          candidate.apply_translation(hull_center)
          if not boundary_check.intersects(candidate):
            break
        else:
          raise ValueError(f"{model}/{path.name}: float32 collision cell cannot fit inside CAD")
      if abs(exported.volume) < _MIN_COLLISION_VOLUME:
        omitted_volume += abs(exported.volume)
        continue
      geometry = f"hull_{i:03d}"
      hull_center = hull_center + exported.bounds.mean(axis=0)
      collision_geometry[geometry] = exported
      collision_hulls.append({
          "file": collision_path.relative_to(out_dir).as_posix(),
          "geometry": geometry,
          "pos": hull_center.tolist(),
      })
    # One binary scene per part avoids thousands of tiny files. Components remain named and
    # are instantiated as individual RAI frames/MuJoCo geoms under one rigid part body.
    if omitted_volume:
      print(f"[{method}] {model}/{path.name}: omitted degenerate cells "
            f"({omitted_volume / abs(collision_source.volume):.8%} of CAD volume)", flush=True)
    retained_floor = (.6 if (method == "primitive-inner" or end_retreat)
                      else min_retained_volume)
    if not collision_geometry or (method.endswith("-inner") and
        (sum(abs(h.volume) for h in hulls) - omitted_volume)
        / abs(collision_source.volume) < retained_floor):
      raise ValueError(f"{model}/{path.name}: usable collision cells retain too little CAD volume")
    acd["minimum_cell_volume_m3"] = _MIN_COLLISION_VOLUME
    trimesh.Scene(collision_geometry).export(collision_path)
    metadata["parts"][path.stem] = {
        "file": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "center_cm": center_cm.tolist(),
        "extents_cm": (bounds_cm[1] - bounds_cm[0]).tolist(),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "collision": {
            "method": method,
            "settings": acd,
            "convex_hull_volume_ratio": convexity_ratio,
            "hulls": collision_hulls,
            "asset_sha256": hashlib.sha256(collision_path.read_bytes()).hexdigest(),
            "hull_count": len(collision_hulls),
            "decomposed_volume_ratio": (
                sum(abs(float(h.volume)) for h in hulls)
                / max(abs(float(collision_source.volume)), 1e-15)),
        },
    }
  (out_dir / "manifest.json").write_text(
      json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(f"imported {len(files)} parts: {source_dir} -> {out_dir}")


def import_model(source_root: Path, output_root: Path, model: str,
                 method="tetra-inner", tetra_faces=6000, part_methods=None,
                 wheel_proxy_components=2, only_parts=None,
                 min_retained_volume=.8):
  """Finish generation in a staging directory before replacing any existing asset."""
  output_root.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix=f".{model}-", dir=output_root) as temporary:
    staged_root = Path(temporary)
    target = output_root / model
    if only_parts is not None:
      if not (target / "manifest.json").exists():
        raise FileNotFoundError(
            f"--only-parts requires an existing imported model at {target}")
      shutil.copytree(target, staged_root / model)
    _import_model(source_root, staged_root, model, method, tetra_faces, part_methods,
                  wheel_proxy_components, only_parts, min_retained_volume)
    if model == "cooling_manifold" and method.endswith("-inner"):
      approximate_cooling_fins(staged_root / model)
    source = staged_root / model
    target = output_root / model
    # Commit the manifest last. A failed decomposition above never alters a live model.
    paths = sorted(source.rglob("*"), key=lambda p: (p.name == "manifest.json", str(p)))
    for path in paths:
      if path.is_file():
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)


def approximate_cooling_fins(directory):
  """User-approved rectangular fin envelopes, with both connector bores kept open."""
  manifest_path = directory / "manifest.json"
  metadata = json.loads(manifest_path.read_text())
  collision = metadata["parts"]["1"]["collision"]
  if collision.get("approximation_boxes"):
    return
  # CAD fin stack: 80 x 80 x 10 mm, on top of the solid manifold core.
  # Reserve rectangular columns around both 17.65 x 17.5 mm connectors, including
  # 0.5 mm clearance. Keep the original bore/rim cells in those columns.
  base_center = np.array(metadata["parts"]["1"]["center_cm"]) * .01
  exclusions = []
  for source in ("0", "4"):
    item = metadata["parts"][source]
    center = np.array(item["center_cm"]) * .01 - base_center
    half = np.array(item["extents_cm"]) * .005 + .0005
    exclusions.append((center - half, center + half))
  cuts = sorted(set([-.04, .04] + [float(np.clip(b[0], -.04, .04))
                                 for pair in exclusions for b in pair]))
  boxes = []
  for left, right in zip(cuts[:-1], cuts[1:]):
    mid = (left + right) / 2
    excluded = [(lo[1], hi[1]) for lo, hi in exclusions if lo[0] <= mid <= hi[0]]
    intervals = [(-.04, .04)] if not excluded else [(-.04, min(x[0] for x in excluded)),
                                                   (max(x[1] for x in excluded), .04)]
    for low, high in intervals:
      boxes.append((np.array([left, low, .0075]), np.array([right, high, .0175])))
  path = directory / "collision" / "1.glb"
  scene = trimesh.load(path, force="scene")
  geometries, records = {}, []
  for rec in collision["hulls"]:
    hull = scene.geometry[rec["geometry"]].copy()
    hull.apply_translation(np.array(rec["pos"]) - hull.bounds.mean(axis=0))
    if any(np.all(hull.bounds[0] >= lo - 1e-8) and np.all(hull.bounds[1] <= hi + 1e-8)
           for lo, hi in boxes):
      continue
    # Preserve the already-validated float32 geometry and its exact offset. Recentring
    # it again can move thin boundary cells outside the CAD through rounding alone.
    geometries[rec["geometry"]] = scene.geometry[rec["geometry"]].copy()
    records.append(dict(rec))
  removed = len(collision["hulls"]) - len(records)
  for i, (lo, hi) in enumerate(boxes):
    hull = trimesh.creation.box(hi - lo)
    hull.apply_translation((lo + hi) / 2)
    center = hull.bounds.mean(axis=0)
    hull.apply_translation(-center)
    hull.vertices = np.asarray(hull.vertices, dtype=np.float32)
    name = f"fin_envelope_{i:03d}"
    geometries[name] = hull
    records.append({"file": "collision/1.glb", "geometry": name,
                    "pos": (center + hull.bounds.mean(axis=0)).tolist()})
  trimesh.Scene(geometries).export(path)
  collision.update(hulls=records, hull_count=len(records),
                   asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   approximation_boxes=[{"lower": lo.tolist(), "upper": hi.tolist()} for lo, hi in boxes],
                   approximation_reason="user-approved solid fin envelopes; connector bores excluded")
  manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
  print(f"[fins] replaced {removed} cells with {len(boxes)} boxes; "
        f"{len(records)} manifold cells remain", flush=True)


def main():
  here = Path(__file__).resolve().parent
  default_source = here.parents[3] / "Fabrica" / "assets" / "fabrica"
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--source", type=Path, default=default_source)
  ap.add_argument("--output", type=Path, default=here)
  ap.add_argument("--method", choices=["tetra-inner", "vhacd-inner", "vhacd"], default="tetra-inner")
  ap.add_argument("--tetra-faces", type=int, default=6000)
  ap.add_argument("--min-retained-volume", type=float, default=.8,
                  help="minimum summed inner-cell/CAD volume ratio (default 0.8); "
                       "lower values are explicit diagnostic approximations and are "
                       "recorded in the manifest")
  ap.add_argument("--wheel-proxy-components", type=int, choices=[2, 96], default=2,
                  help="Car-only controlled ablation: tile the same two cylinders into "
                       "2 or 96 convex components")
  ap.add_argument("--part-method", action="append", default=[], metavar="SOURCE_ID=METHOD",
                  help="explicit per-source-part decomposition method; recorded in manifest")
  ap.add_argument("--only-parts", nargs="+", metavar="SOURCE_ID",
                  help="atomically regenerate only these source IDs, preserving every other "
                       "approved part and its decomposition byte-for-byte")
  ap.add_argument("--compact-existing", action="store_true",
                  help="compact contained cells already present under --output")
  ap.add_argument("--refit-existing", action="store_true",
                  help="revalidate/refit contained cells against exported CAD without merging")
  ap.add_argument("--models", nargs="+", default=[
      "car", "beam", "cooling_manifold", "duct", "gamepad", "plumbers_block",
      "stool_circular"])
  args = ap.parse_args()
  if not 0 < args.min_retained_volume <= 1:
    ap.error("--min-retained-volume must be in (0, 1]")
  part_methods = {}
  for override in args.part_method:
    source, separator, method = override.partition("=")
    if not separator or not source or method not in ("tetra-inner", "vhacd-inner", "vhacd"):
      ap.error("--part-method must be SOURCE_ID=tetra-inner|vhacd-inner|vhacd")
    part_methods[source] = method
  for model in args.models:
    if args.compact_existing or args.refit_existing:
      compact_existing(args.output.resolve(), model, merge=not args.refit_existing)
    else:
      import_model(args.source.resolve(), args.output.resolve(), model,
                   args.method, args.tetra_faces, part_methods,
                   args.wheel_proxy_components, args.only_parts,
                   args.min_retained_volume)


if __name__ == "__main__":
  main()
