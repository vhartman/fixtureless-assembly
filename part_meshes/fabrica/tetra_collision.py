"""Constrained tetrahedralization followed by convex, face-adjacent merging.

TetGen preserves the input boundary. Merging is accepted only when the convex hull
volume equals the sum of the cells' volumes (up to numerical roundoff). Unlike
approximate convex decomposition, this does not deliberately bridge concavities.
"""

import heapq
import time

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def solid_contains(mesh, points, chunk_size=16):
  """Deterministic generalized-winding-number solid membership.

  ``trimesh.contains`` uses ray parity.  On detailed, nearly coplanar CAD such as the
  Fabrica duct, identical points can receive different answers when a ray grazes an edge.
  The signed solid angle is deterministic and classifies a consistently oriented closed
  surface by a winding magnitude of 4*pi inside and zero outside.  Multiple overlapping
  authored shells may yield larger integral windings, so only the magnitude threshold
  matters here.
  """
  points = np.atleast_2d(np.asarray(points, dtype=float))
  triangles = np.asarray(mesh.triangles, dtype=float)
  result = []
  for start in range(0, len(points), max(1, int(chunk_size))):
    p = points[start:start + chunk_size, None, :]
    a = triangles[None, :, 0, :] - p
    b = triangles[None, :, 1, :] - p
    c = triangles[None, :, 2, :] - p
    la = np.linalg.norm(a, axis=2)
    lb = np.linalg.norm(b, axis=2)
    lc = np.linalg.norm(c, axis=2)
    numerator = np.einsum("pti,pti->pt", a, np.cross(b, c))
    denominator = (la * lb * lc
                   + np.einsum("pti,pti->pt", a, b) * lc
                   + np.einsum("pti,pti->pt", b, c) * la
                   + np.einsum("pti,pti->pt", c, a) * lb)
    winding = 2.0 * np.arctan2(numerator, denominator).sum(axis=1)
    result.extend(np.abs(winding) > 2.0 * np.pi)
  return np.asarray(result, dtype=bool)


class BoundaryCheck:
  def __init__(self, mesh):
    self.triangles = np.asarray(mesh.triangles)
    self.tree = mesh.triangles_tree

  def intersects(self, candidate):
    """Triangle/convex intersection by half-space clipping, including enclosed faces.

    libccd's MPR portal discovery can loop on near-degenerate triangle/convex pairs.
    Clipping has bounded work and does not need a penetration-depth optimizer. Touching
    counts as intersection; the tiny tolerance is conservative, not a clearance waiver.
    """
    tol = 1e-12
    bounds = candidate.bounds + np.array([[-tol] * 3, [tol] * 3])
    indices = list(self.tree.intersection(bounds.ravel()))
    if not indices:
      return False
    origin = candidate.vertices.mean(axis=0)
    planes = np.unique(ConvexHull(candidate.vertices - origin).equations, axis=0)
    normals, offsets = planes[:, :3], planes[:, 3]
    triangles = self.triangles[indices] - origin
    distances = triangles @ normals.T + offsets - tol
    # A single separating facet rejects most triangles without polygon construction.
    possible = ~(distances.min(axis=1) > 0).any(axis=1)
    for polygon, signed in zip(triangles[possible], distances[possible]):
      for index in np.argsort(-signed.max(axis=0)):
        n, b = normals[index], offsets[index] - tol
        d = polygon @ n + b
        inside = d <= 0
        if inside.all():
          continue
        if not inside.any():
          break
        clipped = []
        for i in range(len(polygon)):
          j = (i + 1) % len(polygon)
          if inside[i]:
            clipped.append(polygon[i])
          if inside[i] != inside[j]:
            clipped.append(polygon[i] + d[i] / (d[i] - d[j]) * (polygon[j] - polygon[i]))
        polygon = np.asarray(clipped)
      else:
        return True
    return False


def merge_contained(hulls, mesh, max_sweeps=12, neighbour_count=32):
  """Greedily merge neighboring cells only when their hull stays within the CAD.

  Nearest neighbors limit offline work; this affects compactness, not validity.
  Triangle clipping also catches CAD surfaces enclosed by a candidate.
  """
  import trimesh
  from scipy.spatial import cKDTree

  check = BoundaryCheck(mesh)
  groups = dict(enumerate(hulls))
  # Rebuilding the neighbourhood is what exposes larger merge opportunities after small
  # cells have coalesced. Four 16-neighbour passes left 600--1900 geoms on several Fabrica
  # shells, making a single KOMO collision evaluation take seconds. This remains an offline
  # representation compaction: every accepted hull passes the exact same boundary test.
  for sweep in range(max(1, int(max_sweeps))):
    keys = list(groups)
    centers = np.array([groups[k].vertices.mean(axis=0) for k in keys])
    tree = cKDTree(centers)
    pairs = set()
    for i, near in enumerate(tree.query(
        centers, k=min(max(2, int(neighbour_count)) + 1, len(keys)))[1]):
      for j in np.atleast_1d(near):
        if i != j:
          pairs.add(tuple(sorted((keys[i], keys[int(j)]))))
    merged = 0
    for a, b in sorted(pairs, key=lambda pair: (
        -min(groups[pair[0]].volume, groups[pair[1]].volume), *pair)):
      if a not in groups or b not in groups:
        continue
      candidate = trimesh.convex.convex_hull(np.vstack([groups[a].vertices, groups[b].vertices]))
      if len(candidate.vertices) > 128:
        continue
      if check.intersects(candidate):
        continue
      groups[a] = candidate
      del groups[b]
      merged += 1
    print(f"[inner-merge] pass {sweep + 1}: {len(groups)} cells", flush=True)
    if not merged:
      break
  return list(groups.values())


def decompose(mesh, max_vertices=128):
  import meshpy.tet as tet

  if not mesh.is_watertight or not mesh.is_winding_consistent:
    raise ValueError("TetGen collision import requires a watertight, consistently oriented mesh")
  start = time.monotonic()
  # Work in a unit-size domain to give the hull tests a stable numerical scale.
  origin = mesh.bounds.mean(axis=0)
  scale = float(mesh.extents.max())
  info = tet.MeshInfo()
  info.set_points((mesh.vertices - origin) / scale)
  info.set_facets(mesh.faces)
  out = tet.build(info, options=tet.Options("pQT1e-10"))
  vertices = np.asarray(out.points)
  cells = np.asarray(out.elements, dtype=int)
  if len(cells) > 50000:
    raise ValueError(f"TetGen produced {len(cells)} cells, exceeding the offline work limit "
                     "of 50000; try --tetra-faces 2000 to reduce redundant surface detail")
  corners = vertices[cells]
  volumes = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6
  expected = abs(float(mesh.volume)) / scale**3
  if abs(volumes.sum() - expected) > expected * 1e-3:
    raise ValueError("TetGen volume differs from CAD; check enclosed cavities or invalid surfaces")
  groups = {i: (np.unique(c), float(volumes[i])) for i, c in enumerate(cells)}
  neighbors = {i: set() for i in groups}
  faces = {}
  queue = []
  versions = {i: 0 for i in groups}

  def enqueue(a, b):
    if a > b:
      a, b = b, a
    heapq.heappush(queue, (-(groups[a][1] + groups[b][1]),
                           a, b, versions[a], versions[b]))

  for i, cell in enumerate(cells):
    for omit in range(4):
      face = tuple(sorted(np.delete(cell, omit)))
      j = faces.pop(face, None)
      if j is None:
        faces[face] = i
      else:
        neighbors[i].add(j)
        neighbors[j].add(i)
        enqueue(i, j)
  attempts = 0
  while queue:
    _, a, b, va, vb = heapq.heappop(queue)
    if a not in groups or b not in groups or versions[a] != va or versions[b] != vb:
      continue
    indices = np.union1d(groups[a][0], groups[b][0])
    attempts += 1
    try:
      hull = ConvexHull(vertices[indices])
    except QhullError:
      continue
    volume = groups[a][1] + groups[b][1]
    if len(hull.vertices) > max_vertices or hull.volume > volume * (1 + 1e-8) + 1e-14:
      continue
    groups[a] = (indices[hull.vertices], volume)
    del groups[b]
    versions[a] += 1
    adjacent = (neighbors[a] | neighbors[b]) - {a, b}
    for n in adjacent:
      neighbors[n].discard(b)
      neighbors[n].add(a)
    neighbors[a] = adjacent
    del neighbors[b]
    for n in adjacent:
      enqueue(a, n)
  import trimesh
  hulls = [trimesh.convex.convex_hull(vertices[idx] * scale + origin)
           for idx, _ in groups.values()]
  hulls.sort(key=lambda h: tuple(np.round(np.r_[h.centroid, h.volume], 12)))
  print(f"[tetra] {len(cells)} tetrahedra -> {len(hulls)} convex cells; "
        f"{attempts} merge tests, {time.monotonic() - start:.1f}s", flush=True)
  return hulls, {"tetgen_options": "pQT1e-10", "merge_relative_volume_tolerance": 1e-8,
                 "max_vertices": max_vertices, "tetrahedra": len(cells)}


def inner_decompose(mesh, face_count=6000, min_volume_fraction=0.8, hulls=None, settings=None):
  """Build cells checked inside the original surface with triangle clipping queries.

  Simplification is only a proposal. A cell is retained only if its centre is
  inside the original CAD and its boundary does not intersect the CAD boundary.
  Convex contraction around that centre is monotone, so bisection finds a fit.
  These are floating-point geometric checks, not an exact-arithmetic proof.
  """
  import trimesh

  proposed = mesh
  if hulls is None:
    proposed = (mesh.simplify_quadric_decimation(face_count=face_count, aggression=3)
                if len(mesh.faces) > face_count else mesh.copy())
    hulls, settings = decompose(proposed)
  settings = dict(settings or {})
  check = BoundaryCheck(mesh)
  inside = solid_contains(mesh, np.array([h.vertices.mean(axis=0) for h in hulls]))
  kept = []
  for hull, valid in zip(hulls, inside):
    if not valid:
      continue
    center = hull.vertices.mean(axis=0)
    vertices = hull.vertices.copy()
    lo, hi = 0.0, 1.0
    hull.vertices = center + .999 * (vertices - center)
    if not check.intersects(hull):
      lo = .999
    else:
      for _ in range(14):
        mid = (lo + hi) / 2
        hull.vertices = center + mid * (vertices - center)
        if check.intersects(hull):
          hi = mid
        else:
          lo = mid
    # Include GLB float32 quantization in the final boundary check.
    hull.vertices = (center + lo * .999 * (vertices - center)).astype(np.float32)
    if lo < 1e-3 or hull.volume <= 1e-16:
      continue
    for _ in range(20):
      if not check.intersects(hull):
        break
      lo *= .99
      hull.vertices = (center + lo * .999 * (vertices - center)).astype(np.float32)
    else:
      continue
    kept.append(hull)
  fraction = sum(abs(float(h.volume)) for h in kept) / abs(float(mesh.volume))
  print(f"[inner] {len(kept)} cells, cell-volume sum/CAD {fraction:.2%}", flush=True)
  if fraction < min_volume_fraction:
    raise ValueError(f"inner decomposition retains only {fraction:.1%} of CAD volume; "
                     "increase --tetra-faces and inspect the source geometry")
  kept = merge_contained(kept, mesh)
  settings.update({"proposal_faces": len(proposed.faces), "face_budget": face_count,
                   "boundary_method": "triangle-convex-halfspace-clipping",
                   "boundary_tolerance_m": 1e-12,
                   "containment": "inside centre plus nonintersecting triangle boundaries",
                   "premerge_cell_volume_sum_fraction": fraction,
                   "minimum_volume_fraction": min_volume_fraction})
  return kept, settings, abs(float(mesh.convex_hull.volume) / float(mesh.volume))
