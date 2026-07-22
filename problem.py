"""Problem definition: assemblies, part geometry, grasps, and the discrete scene-graph
search model (State + successors). No feasibility/solver or visualization code here."""

import os

import robotic as ry
import numpy as np

from env import *
import wrench


def set_seed(seed=0):
  """Make runs as reproducible as possible: seed numpy (part colours / start-pose
  perturbation) AND rai's RNG (KOMO seeds its IK guess with noise, which is the main source
  of run-to-run variation). Call once, before building configs / solving."""
  ry.params_add({"seed": int(seed)})
  np.random.seed(int(seed))


class Assembly:
  """Backend-independent description of an assembly.

  An assembly is defined by four things, which subclasses fill in via `define`:
    objects:                 list of part names.
    base:                    the part the assembly is built on / posed relative to.
    assembly_dependencies:   {part: [parts that must be assembled before it]}. Encodes
                             the order(s) in which parts may be added.
    assembly_relative_poses: {part: [parent, relative_position]} — where each part sits
                             once assembled. Parts are axis-aligned, so translation only.

  The base class only stores this structure and derives things from it (e.g.
  `position_in_base`); it has no geometry/planner assumptions.

  Physics (for the wrench/stability model, see wrench.py):
    masses:      {part: kg}          per-part mass (NOT rai's frame mass), default
                                     DEFAULT_MASS. Used by the equilibrium check.
    mate_types:  {part: type_str}    the contact TYPE of the mate that joins `part` to its
                                     parent (a key into wrench.py's capability builders).
                                     Default "rigid_weld" for every mate.
    mate_axes:   {part: unit 3-vec}  the ASSEMBLY (press/seating) DIRECTION of the mate, in
                                     the base frame: the direction the part is pressed along
                                     while seating. Default: -offset for axis-aligned mates
                                     (press onto the parent face); for DIAGONAL offsets an
                                     axis-aligned direction with a FREE approach cell,
                                     preferring top-down -- a diagonal offset is a
                                     bookkeeping artifact, real insertions are not diagonal.
  Subclasses may override any of these in define(); anything left unset gets the default.
  """

  DEFAULT_MASS = 0.1                                        # kg, a single cube
  DEFAULT_MATE_TYPE = "rigid_weld"

  def __init__(self):
    self.objects = []
    self.base = None
    self.assembly_dependencies = {}
    self.assembly_relative_poses = {}
    self.masses = {}
    self.mate_types = {}
    self.mate_axes = {}
    # PER-PART SHAPES: {part: (kind, params)} -- ("box", [sx, sy, sz, corner_r]),
    # ("cylinder", [length, radius]) (axis = local z), ("sphere", [radius]). Anything unset
    # defaults to the assembly's cube (cube_size/cube_radius) after define(), so classic
    # all-cube assemblies are untouched. Every scene builder, grasp generator, rest-pose and
    # wrench-contact computation reads shapes through part_shape() -- ONE source of truth,
    # so the cheap scenes, IK scenes and real env always agree.
    self.part_shapes = {}
    # SCATTER ORIENTATIONS: {part: quat} -- how a part initially LIES on the table (default
    # identity). E.g. the car's wheels rest FLAT (cap down) in the scatter although their
    # assembled orientation is axle-sideways; the anchor pose carries the rotation, and the
    # whole placement-face system composes on the anchor, so nothing else changes.
    self.scatter_orientations = {}
    # CONNECT (module-join) mates: parts whose mate onto their parent joins two independently
    # built MODULES rather than adding a part within one. Their mate edge is NOT traversed by
    # `_mate_closure` (so each module's internal build precedence excludes the join), which is
    # what lets a module (e.g. a plate) be an order ideal on its own -- buildable detached and
    # connected as a UNIT. `connect_requires[p]` = the parts of the OTHER module that must be
    # complete before p may connect (the join precedence, enforced by the connect action, not
    # by module enumeration). Default empty -> the classic single-tree assemblies are
    # unchanged. See CubeTwoPlateAssembly.
    self.connect_parts = set()
    self.connect_requires = {}
    # NO-HOLD parts: a grasp on one of these may only manipulate the LONE part (pick it, press
    # it in as the inserter) -- it can never HOLD/BRACE a multi-part subassembly. Models parts
    # that are not a stable grip for the rest (e.g. a wheel that would rotate): you can put it
    # on the car, but you can't then hang the car off it for later steps. NOT a physics change
    # (mates stay rigid welds); just a discrete restriction on which grasps count as support.
    self.no_hold = set()
    # STAGED MATES (A11, --staged-mates): {part: fastener_part}. `part`'s mate is only a
    # PARTIAL contact -- a slot (wrench.cap_slot: lateral walls + seat, NO couples, no
    # extraction resistance) -- until `fastener_part` is mated into the same subassembly,
    # which upgrades it to the usual rigid weld. Until then the part may fall out of its own
    # group (gravity moment / extraction component), so the search must keep a grasp on it or
    # rest the assembly on a face where the slot alone balances gravity -- discovered, not
    # scripted (see _staged_ok). Empty (default) + flag off => byte-identical behavior.
    self.staged_mates = {}
    self.define()
    for p in self.objects:                                 # fill physics defaults
      self.masses.setdefault(p, self.DEFAULT_MASS)
      cs = getattr(self, "cube_size", 0.05)
      cr = getattr(self, "cube_radius", 0.005)
      self.part_shapes.setdefault(p, ("box", [cs, cs, cs, cr]))
    for p in self.assembly_relative_poses:                 # every mate has a type + an axis
      self.mate_types.setdefault(p, self.DEFAULT_MATE_TYPE)
      self.mate_axes.setdefault(p, self._default_mate_axis(p))
    self.validate()

  def part_shape(self, part):
    # (kind, params) collision/appearance shape of `part` -- see part_shapes in __init__.
    return self.part_shapes[part]

  def mass(self, part):
    return self.masses.get(part, self.DEFAULT_MASS)

  def can_hold(self, part):
    # may a grasp on `part` HOLD/BRACE a subassembly? False for no_hold parts (see __init__).
    return part not in self.no_hold

  def mate_type(self, part):
    # contact type of the mate joining `part` to its parent (None for the base / free part).
    return self.mate_types.get(part)

  def mate_axis(self, part):
    # THE press/seating direction of `part`'s mate, base frame (None for the base / a free
    # part). A SINGLE direction -- like a real assembly manual, the mate specifies how the
    # part goes in; the planner does not get to choose. Subclasses may override in define().
    a = self.mate_axes.get(part)
    return None if a is None else np.asarray(a, dtype=float)

  def _default_mate_axis(self, part):
    # sensible default assembly direction: PRESS AGAINST AN ALREADY-BUILT FLANKING NEIGHBOR
    # (a mate must seat against a face). Axis-aligned offset: the only such face is the
    # parent's -> -offset (unchanged face-glue semantics). Diagonal offset (a grid pocket;
    # single-parent bookkeeping makes it look diagonal): press along an offset component
    # whose neighbor cell is in the MATE CLOSURE (already built when `part` is inserted),
    # preferring in-plane x then y then z -- the row press ('hold the row, push from the
    # open end'), which a gripper can transmit as compression with the fingers clear;
    # a top-down press into a two-sided pocket traps the fingers against the flanks.
    off = np.asarray(self.assembly_relative_poses[part][1], dtype=float)
    nz = np.abs(off) > 1e-9
    if nz.sum() <= 1:
      return -off / (np.linalg.norm(off) + 1e-12)          # axis-aligned: unchanged semantics
    step = min(abs(c) for c in off[nz])                    # lattice pitch
    seat = self.position_in_base(part)
    built = [self.position_in_base(q) for q in _mate_closure(self, part)]
    axes = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    for i in (0, 1, 2):                                    # prefer the in-plane row press
      if not nz[i]:
        continue
      d = -np.sign(off[i]) * axes[i]
      neighbor = seat + d * step                           # the face the press seats against
      if any(np.linalg.norm(b - neighbor) < 0.5 * step for b in built):
        return d
    return -off / (np.linalg.norm(off) + 1e-12)            # nothing flanking: old behavior

  def define(self):
    # subclasses populate objects / base / dependencies / relative poses here
    raise NotImplementedError("Assembly subclasses must implement define()")

  def validate(self):
    assert self.base in self.objects, "base must be one of the objects"
    for part, deps in self.assembly_dependencies.items():
      assert part in self.objects, f"unknown part in dependencies: {part}"
      for dep in deps:
        assert dep in self.objects, f"unknown dependency {dep} for {part}"
    for part, (parent, _pos) in self.assembly_relative_poses.items():
      assert part in self.objects, f"unknown part in relative poses: {part}"
      assert parent in self.objects, f"unknown parent {parent} for {part}"

  def dependencies_of(self, part):
    return self.assembly_dependencies.get(part, [])

  def relative_pose(self, part):
    # [parent, relative_position], or None for the base / a free part
    return self.assembly_relative_poses.get(part)

  def position_in_base(self, part):
    # position of a part's origin in the base frame, following the dependency chain.
    # parts are axis-aligned, so orientation stays identity throughout.
    rel = self.assembly_relative_poses.get(part)
    if rel is None:
      return np.zeros(3)
    parent, rel_pos = rel
    return self.position_in_base(parent) + np.asarray(rel_pos, dtype=float)

def add_disassembled_obj(C, assembly, perturb_start_pose=False, pos_noise=0.03, max_tries=200,
                         spacing=1.0 / 7.0, per_row=8, random_init=None, float_height=0.015,
                         min_sep=None):
  # scatter the assembly's parts on the table as loose cubes (test scene). Geometry
  # (cube_size / cube_radius) comes from the assembly, not a global. `spacing` = the x-pitch
  # of the scatter line (default = historical 1/7 m ~ 0.143); a knob for analysing how much
  # search/IK pain comes from the cramped test scene (arms diving between close neighbours)
  # rather than the planner. Parts beyond `per_row` WRAP into further rows (y += spacing per
  # row) -- a single line of >8 parts walks off the reachable table. Row 0 is byte-identical
  # to the historical line, so existing <=8-part scenes are unchanged.
  cs = assembly.cube_size

  table_half = C.getFrame("table").getSize()[2] / 2.0

  for i, name in enumerate(assembly.objects):
      row, col = divmod(i, per_row)
      # scatter orientation (how the part initially LIES, e.g. wheels flat) and a
      # SHAPE-AWARE height: table top + half extent at that orientation + `float_height`
      # (historical default 15mm; for the classic cube that is exactly the old fixed 0.07).
      # The float propagates through the placement ANCHORS, so every later set-down keeps it.
      scatter_q = assembly.scatter_orientations.get(name)
      spec = assembly.part_shape(name)
      Rs = _quat_to_R(scatter_q) if scatter_q is not None else np.eye(3)
      z = table_half + _shape_half_z(spec, Rs) + float_height
      pos = np.array([-0.5 + col * spacing, row * spacing, z])

      sq = _shape_quat(spec)
      obj = C.addFrame(name).setParent(C.getFrame("table"))
      if sq is None:                                       # shape on the part frame itself
        _set_raw_shape(obj, spec).setContact(1)
        shape_f = obj
      else:                                                # rotated shape -> rigid child
        shape_f = _set_raw_shape(C.addFrame(f"{name}__shape", name), spec)
        shape_f.setRelativeQuaternion(list(sq))
        shape_f.setContact(1)
      obj.setRelativePosition([pos[0], pos[1], pos[2]]).setJoint(ry.JT.rigid)
      if scatter_q is not None:
        obj.setRelativeQuaternion(list(scatter_q))
      shape_f.setMass(0.1)                                 # rai wants the mass on the shape
      shape_f.setColor(np.random.rand(3))

      # ORIENTATION MARKER: a rai axis-triad marker per cube, so rotations of the
      # uniformly-colored cubes are visible. A child frame -> follows the part wherever
      # the state graph moves it; no contact shape, so the planner never sees it. The
      # ViserRecorder renders marker frames as viser axis frames (they have no mesh).
      C.addFrame(f"{name}_marker", name).setShape(ry.ST.marker, [cs * 1.5])

      base_q = scatter_q if scatter_q is not None else np.array([1.0, 0.0, 0.0, 0.0])
      if random_init:
          # FULL randomization (seeded, hence reproducible): uniform x/y in the central
          # square [-random_init, +random_init]^2 between the robots + a random yaw,
          # rejection-sampled collision-free against everything already placed (parts,
          # table, robot bases/arms). `min_sep` additionally enforces a pairwise x/y
          # SEPARATION from every earlier part (e.g. 0.2 so graspability/geometry of
          # different parts provably cannot interact -- the POR independence assumption).
          # Falls back to the nominal line spot if exhausted.
          placed = [np.asarray(C.getFrame(q).getPosition()[:2])
                    for q in assembly.objects[:i]]
          for _ in range(max_tries):
              xy = np.random.uniform(-float(random_init), float(random_init), size=2)
              if min_sep is not None and any(
                  np.linalg.norm(xy - p) < float(min_sep) for p in placed):
                  continue
              yaw = np.random.uniform(-np.pi, np.pi)
              obj.setRelativePosition([xy[0], xy[1], pos[2]])
              obj.setRelativeQuaternion(list(_quat_mul(
                  np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), base_q)))
              if C.getCollisionFree():
                  break
          else:
              obj.setRelativePosition([pos[0], pos[1], pos[2]])
              obj.setRelativeQuaternion(list(base_q))
      elif perturb_start_pose:
          # rejection-sample a perturbation that keeps the config collision free (the yaw
          # composes ONTO the scatter orientation, so e.g. flat wheels stay flat)
          for _ in range(max_tries):
              dxy = np.random.uniform(-pos_noise, pos_noise, size=2)
              yaw = np.random.uniform(-np.pi, np.pi)
              obj.setRelativePosition([pos[0] + dxy[0], pos[1] + dxy[1], pos[2]])
              obj.setRelativeQuaternion(list(_quat_mul(
                  np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), base_q)))
              if C.getCollisionFree():
                  break
          else:
              # could not place without collision: fall back to the nominal pose
              obj.setRelativePosition([pos[0], pos[1], pos[2]])
              obj.setRelativeQuaternion(list(base_q))

  return C

class CubeCrossAssembly(Assembly):
  """A central cube with one cube glued to each of its six faces (7 parts total).

  Parts are assembled one at a time in index order (obj_1 needs obj_0, obj_2 needs
  obj_1, ...), and every leaf cube sits one `spacing` away from the base along a
  distinct axis.
  """

  def __init__(self, n_parts=7, spacing=0.055, cube_size=0.05, cube_radius=0.005):
    assert 1 <= n_parts <= 7, "a cube has 6 faces, so at most 7 parts"
    self.n_parts = n_parts
    self.spacing = spacing
    # part geometry lives on the assembly (this is a cube test-assembly; a real one would
    # carry per-part meshes). The scene builder, grasp planner and collision checker all
    # read cube_size / cube_radius from here rather than a global.
    self.cube_size = cube_size
    self.cube_radius = cube_radius
    super().__init__()

  def define(self):
    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"

    self.assembly_dependencies = {
        f"obj_{i}": [f"obj_{i - 1}"] for i in range(1, self.n_parts)
    }

    o = self.spacing
    face_offsets = [
        [0, 0, o], [0, 0, -o],
        [0, o, 0], [0, -o, 0],
        [o, 0, 0], [-o, 0, 0],
    ]
    self.assembly_relative_poses = {
        f"obj_{i}": ["obj_0", np.array(face_offsets[i - 1])]
        for i in range(1, self.n_parts)
    }

class CubeGridAssembly(Assembly):
  """A 2x2x2 cube built from 8 smaller cubes (fewer with n_parts < 8).

  Parts are added in index order (chain dependencies): first the bottom layer
  (obj_0..obj_3), then the top layer (obj_4..obj_7); every part's mate pose is
  expressed relative to the corner cube obj_0."""

  def __init__(self, n_parts=8, spacing=0.055, cube_size=0.05, cube_radius=0.005):
    assert 1 <= n_parts <= 8, "a 2x2x2 cube has 8 cells"
    self.n_parts = n_parts
    self.spacing = spacing
    self.cube_size = cube_size
    self.cube_radius = cube_radius
    super().__init__()

  def define(self):
    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"
    self.assembly_dependencies = {
        f"obj_{i}": [f"obj_{i - 1}"] for i in range(1, self.n_parts)
    }
    o = self.spacing
    cells = [
        [0, 0, 0], [o, 0, 0], [0, o, 0], [o, o, 0],      # bottom layer
        [0, 0, o], [o, 0, o], [0, o, o], [o, o, o],      # top layer
    ]
    self.assembly_relative_poses = {
        f"obj_{i}": ["obj_0", np.array(cells[i], dtype=float)]
        for i in range(1, self.n_parts)
    }


class ArchAssembly(Assembly):
  """An arch / bridge: two 2-cube pillars and a 3-cube lintel spanning them (7 parts).

  KNOWN-FLAWED PLACEHOLDER (kept as the motivating case for the deferred non-tree /
  connect-two-subassemblies redesign): every part mates onto obj_0, so it is a STAR, not
  the physical structure -- e.g. the right pillar is "supported through the air" via obj_0.
  A faithful arch needs TWO ground bases + a loop-closing connect step (see the memory
  note). For a sequentially buildable arch shape, use `ArchSequentialAssembly`."""

  def __init__(self, n_parts=7, spacing=0.055, cube_size=0.05, cube_radius=0.005):
    assert 1 <= n_parts <= 7, "the arch has 7 parts"
    self.n_parts = n_parts
    self.spacing = spacing
    self.cube_size = cube_size
    self.cube_radius = cube_radius
    super().__init__()

  def define(self):
    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"
    self.assembly_dependencies = {
        f"obj_{i}": [f"obj_{i - 1}"] for i in range(1, self.n_parts)
    }
    o = self.spacing
    cells = [
        [0, 0, 0],                                        # left pillar, bottom
        [0, 0, o],                                        # left pillar, top
        [2 * o, 0, 0],                                    # right pillar, bottom
        [2 * o, 0, o],                                    # right pillar, top
        [0, 0, 2 * o],                                    # lintel, left
        [o, 0, 2 * o],                                    # lintel, middle (over the gap!)
        [2 * o, 0, 2 * o],                                # lintel, right
    ]
    self.assembly_relative_poses = {
        f"obj_{i}": ["obj_0", np.array(cells[i], dtype=float)]
        for i in range(1, self.n_parts)
    }


class ArchSequentialAssembly(Assembly):
  """The same 7-cube arch shape, but built as a single sequential CHAIN traced along the
  structure: UP the left pillar, ACROSS the lintel, DOWN the right pillar. Each part mates
  onto its physically ADJACENT predecessor obj_{i-1} (not a star onto obj_0), so it
  assembles in strict index / left-to-right order with no through-air mates and stays one
  connected component throughout.

  (Gravity stability of the cantilevered span is still ignored -- that is the deferred
  wrench layer; the faithful two-bases-then-connect model is the TODO on `ArchAssembly`.)"""

  def __init__(self, n_parts=7, spacing=0.055, cube_size=0.05, cube_radius=0.005):
    assert 1 <= n_parts <= 7, "the arch has 7 parts"
    self.n_parts = n_parts
    self.spacing = spacing
    self.cube_size = cube_size
    self.cube_radius = cube_radius
    super().__init__()

  def define(self):
    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"
    self.assembly_dependencies = {
        f"obj_{i}": [f"obj_{i - 1}"] for i in range(1, self.n_parts)
    }
    o = self.spacing
    # the arch cells ORDERED as an adjacency path (consecutive cells touch): up the left
    # pillar, across the lintel, down the right pillar.
    cells = [np.array(c, dtype=float) for c in (
        [0, 0, 0],          # obj_0 left pillar, bottom
        [0, 0, o],          # obj_1 left pillar, top
        [0, 0, 2 * o],      # obj_2 lintel, left
        [o, 0, 2 * o],      # obj_3 lintel, middle (over the gap)
        [2 * o, 0, 2 * o],  # obj_4 lintel, right
        [2 * o, 0, o],      # obj_5 right pillar, top
        [2 * o, 0, 0],      # obj_6 right pillar, bottom
    )]
    # each part mates onto its immediate predecessor; offset = the (unit) step between them.
    self.assembly_relative_poses = {
        f"obj_{i}": [f"obj_{i - 1}", cells[i] - cells[i - 1]]
        for i in range(1, self.n_parts)
    }


class CubePlatesAssembly(Assembly):
  """A width x depth x height cube LATTICE built as `height` stacked width-x-depth PLATES.

  Each plate is a linear CHAIN of face-adjacent cubes (a boustrophedon/serpentine walk over
  the w x d grid), so the intra-plate precedence is a simple sequence -- no diamonds. Cubes
  are NAMED IN CHAIN ORDER: plate k occupies obj_{k*w*d} .. obj_{(k+1)*w*d - 1}, rooted at
  obj_{k*w*d}. The bottom root is the global base; every higher plate's root mates onto the
  root directly BENEATH it, and that mate is the module JOIN (a `connect` mate: not
  traversed by `_mate_closure`, so each plate is a self-contained order ideal, buildable
  DETACHED).

  The join order is FORCED by structure, not by a special rule: `connect_requires[root_k]`
  is the LAST cube of plate k's chain (own plate complete, in the held module) and the LAST
  cube of plate k-1's chain (the plate below complete, in the target -- lower plates are
  then complete transitively, since plate k-1 could itself only have joined complete).
  Because each chain is linear, needing its last cube needs the whole plate. `assemble`
  handles the join (a module is just a subassembly-valued insert); no separate action.

  Truncation (n_parts < w*d*h) keeps the first n_parts cubes: a PARTIAL top plate keeps its
  root as a REGULAR mate (buildable cube-by-cube on the stack); the root becomes a connect
  only when its whole plate exists. This preserves the old 2x2x2 partial-instance behavior
  (used by the test suite at 4/6 parts)."""

  def __init__(self, width=2, depth=2, height=2, n_parts=None,
               spacing=0.055, cube_size=0.05, cube_radius=0.005):
    total = width * depth * height
    self.width, self.depth, self.height = width, depth, height
    self.n_parts = total if n_parts is None else min(n_parts, total)
    assert self.n_parts >= 1
    self.spacing = spacing
    self.cube_size = cube_size
    self.cube_radius = cube_radius
    super().__init__()

  def define(self):
    w, per = self.width, self.width * self.depth
    o = self.spacing
    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"

    def cell(j):
      # grid cell of chain index j within a plate: serpentine rows (even row left->right,
      # odd row right->left), so consecutive chain indices are always face-adjacent.
      row, col = divmod(j, w)
      return (col if row % 2 == 0 else w - 1 - col, row)

    mates, deps = {}, {}
    for i in range(1, self.n_parts):
      k, j = divmod(i, per)
      if j == 0:                                # plate ROOT: stacks onto the root below (JOIN)
        mates[f"obj_{i}"] = [f"obj_{(k - 1) * per}", [0.0, 0.0, o]]
        # no dependency entry: the plate stays a self-contained ideal (buildable detached)
      else:                                     # serpentine step within the plate
        (x0, y0), (x1, y1) = cell(j - 1), cell(j)
        mates[f"obj_{i}"] = [f"obj_{i - 1}", [(x1 - x0) * o, (y1 - y0) * o, 0.0]]
        deps[f"obj_{i}"] = [f"obj_{i - 1}"]
    self.assembly_relative_poses = {
        p: [par, np.array(off, dtype=float)] for p, (par, off) in mates.items()
    }
    self.assembly_dependencies = deps
    # a plate root is a CONNECT iff its plate is FULLY present; the join requires the last
    # chain cube of its own plate and of the plate below (see class doc).
    for k in range(1, self.height):
      root, last_own, last_below = k * per, (k + 1) * per - 1, k * per - 1
      if last_own < self.n_parts:
        self.connect_parts.add(f"obj_{root}")
        self.connect_requires[f"obj_{root}"] = {f"obj_{last_own}", f"obj_{last_below}"}


class CuboidTowerAssembly(Assembly):
  """A JENGA-STYLE tower of laid-down 2:1:1 cuboids: TWO per level, side by side, each
  level rotated 90 degrees against the one below.

  Every level is a 2w x 2w slab. Even levels use x-long parts ("box",[2w,w,w]); odd levels
  use the ROTATED variant expressed as its own shape ("box",[w,2w,w]) -- parts stay
  axis-aligned, the 90-degree turn lives in the per-part shape. Parts form a plain mate
  chain (each onto its predecessor; level roots sit diagonally on the slab below), and
  every mate seats TOP-DOWN (mate_axes = -z: blocks are lowered into place, like real
  block stacking)."""

  def __init__(self, n_parts=4, width=0.05, cube_radius=0.005):
    self.n_parts = n_parts
    self.width = width
    self.cube_size = width                # scalar fallback (marker sizing, defaults)
    self.cube_radius = cube_radius
    super().__init__()

  def define(self):
    w, r = self.width, self.cube_radius
    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"

    def cell(i):
      # center of part i relative to obj_0, and its shape (levels alternate orientation;
      # both cover the same 2w x 2w slab footprint, centered above each other).
      k, a = divmod(i, 2)                                  # level, position-in-level
      if k % 2 == 0:                                       # x-long pair, side by side in y
        return np.array([0.0, a * w, k * w]), ("box", [2.0 * w, w, w, r])
      return np.array([-w / 2.0 + a * w, w / 2.0, k * w]), ("box", [w, 2.0 * w, w, r])

    self.assembly_relative_poses = {}
    for i in range(self.n_parts):
      pos, spec = cell(i)
      self.part_shapes[f"obj_{i}"] = spec
      if i > 0:
        prev, _ = cell(i - 1)
        self.assembly_relative_poses[f"obj_{i}"] = [f"obj_{i - 1}", pos - prev]
        self.mate_axes[f"obj_{i}"] = np.array([0.0, 0.0, -1.0])   # lowered into place
    self.assembly_dependencies = {}       # the mate chain IS the precedence


class CarAssembly(Assembly):
  """A toy car: a cuboid BODY and four CYLINDER WHEELS, two per side, axes sideways.

  The wheels are the first ROTATED shapes: ("cylinder", [l, rad], q) with q turning the
  cylinder's local z onto the part's y axis -- the part frame stays axis-aligned (mates are
  translation-only as everywhere), the rotation lives inside the shape spec (realized as a
  `__shape` child frame in every scene). Each wheel mates AXIALLY onto the body's side
  (mate_axes = -+y: pressed onto its axle); the assembled car rests on its wheels (the
  rest-height machinery reads the wheel rims as the lowest contacts)."""

  def __init__(self, n_parts=5, width=0.05, cube_radius=0.005):
    self.n_parts = min(n_parts, 5)
    self.width = width
    self.cube_size = width
    self.cube_radius = cube_radius
    super().__init__()

  def define(self):
    w, r = self.width, self.cube_radius
    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"                                    # the body
    body = ("box", [4.0 * w, 2.0 * w, w, r])
    q_axis_y = np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0, 0.0])  # z -> y
    wheel = ("cylinder", [0.4 * w, 0.8 * w], q_axis_y)     # thin disc, radius 0.8w
    self.part_shapes["obj_0"] = body
    # wheels at the four side positions: x = +-1.2w (front/back), y = +-1.2w so the inner
    # cap just touches the body side (body half-depth w + half wheel length 0.2w), the
    # axle slightly below the body's midline.
    slots = [(1.4 * w, 1.2 * w), (1.4 * w, -1.2 * w),
             (-1.4 * w, 1.2 * w), (-1.4 * w, -1.2 * w)]
    # wheels start FLAT on the table (cap down): rotate the part -90 deg about x so the
    # axle (part y) points up. The assembled orientation stays identity (axle sideways).
    q_flat = np.array([np.cos(np.pi / 4.0), -np.sin(np.pi / 4.0), 0.0, 0.0])
    for i in range(1, self.n_parts):
      x, y = slots[i - 1]
      self.part_shapes[f"obj_{i}"] = wheel
      self.scatter_orientations[f"obj_{i}"] = q_flat
      self.assembly_relative_poses[f"obj_{i}"] = ["obj_0", np.array([x, y, -0.3 * w])]
      self.mate_axes[f"obj_{i}"] = np.array([0.0, -np.sign(y), 0.0])  # axial, onto the axle
      self.no_hold.add(f"obj_{i}")                          # a wheel would rotate -> can't brace
    self.assembly_dependencies = {}


class SlotPinAssembly(Assembly):
  """STAGED-MATE test assembly (A11): base + a CANTILEVER bar slotted into its side + a PIN
  that fastens the bar. The bar's mate is staged (a slot until the pin lands): unheld and
  unpinned, its own gravity moment tips it out of the slot -- so with --staged-mates the
  search must keep a hand on the bar until the pin is assembled (or rest the assembly slot-
  opening-up). Without the flag it behaves like any rigid-weld 3-chain.
  """

  def __init__(self, n_parts=3, spacing=0.055, cube_size=0.05, cube_radius=0.005):
    assert 2 <= n_parts <= 3, "slot-pin is base + bar + pin"
    self.n_parts = n_parts
    self.spacing = spacing
    self.cube_size = cube_size
    self.cube_radius = cube_radius
    super().__init__()

  def define(self):
    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"
    o = self.spacing
    self.assembly_dependencies = {f"obj_{i}": [f"obj_{i - 1}"] for i in range(1, self.n_parts)}
    self.assembly_relative_poses = {"obj_1": ["obj_0", np.array([o, 0.0, 0.0])]}
    if self.n_parts >= 3:
      self.assembly_relative_poses["obj_2"] = ["obj_1", np.array([0.0, 0.0, o])]
      self.staged_mates = {"obj_1": "obj_2"}               # bar is a slot until the pin lands
    self.masses = {"obj_0": 0.3}                           # heavy base: group stays on its feet


class ChairAssembly(Assembly):
  """A CHAIR with sensible non-cube shapes, GEOMETRY IMPORTED FROM chair.g (a rai scene the
  user modelled): a flat SEAT plate, four CYLINDER LEGS below its corners, a reclined BACKREST
  plate, and two reclined POSTS joining the backrest to the seat. Each part's world ORIENTATION
  from chair.g rides in its shape quaternion (the part frame stays axis-aligned, mates stay
  translation-only -- the same `__shape`-child trick the car's wheels use); positions are the
  chair.g world poses re-anchored to the seat at the origin. Verified collision-free assembled.

  DAG (user-specified): the 4 legs mate onto the seat; the 2 posts mate onto the backrest; and
  the backrest sub-assembly (back + 2 posts) then joins the seat. Part counts: 5 = a STOOL
  (seat + 4 legs), 8 = the full chair (+ backrest + 2 posts)."""

  def __init__(self, n_parts=8, cube_radius=0.005, scale=0.25):
    self.n_parts = 5 if n_parts < 8 else 8
    # chair.g is modelled at a realistic ~0.4 m; the planner's table/reach/scatter are built
    # for ~0.05 m parts, so scale the whole chair down (shape preserved) to fit -- else parts
    # overlap in the scatter line and are out of reach. Orientations are scale-invariant.
    self.scale = scale
    self.cube_size = 0.38 * scale                            # ~ the seat width, for scatter pitch
    self.cube_radius = cube_radius
    super().__init__()

  def define(self):
    s = self.scale
    r = self.cube_radius * s
    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"                                     # the seat
    # shape orientations = chair.g world quats (wxyz); part frames stay axis-aligned. Cylinders
    # get a lie-flat SCATTER orientation (axis -> world x) so BOTH end grasps are reachable on
    # the table -- without it a leg stands upright, only its top cap is graspable, and that grasp
    # jams the gripper against the seat at the mate (see the leg-mate diagnosis).
    q_seat = [0.707, 0.707, 0.008, 0.0]
    q_leg  = [0.0, 1.0, 0.006, -0.006]
    q_back = [0.703, 0.073, 0.075, 0.703]
    q_post = [0.011, 0.99, -0.093, -0.103]
    leg_scatter  = np.array([0.7029, 0.0, -0.7113, 0.0])
    post_scatter = np.array([0.63, 0.0, -0.7766, 0.0])
    self.part_shapes["obj_0"] = ("box", [0.38 * s, 0.05 * s, 0.35 * s, r], q_seat)
    self.masses["obj_0"] = 0.4
    # 4 legs, seat-relative offsets (chair.g world pos - seat world pos [0,0,1]); pressed UP.
    leg_off = {"obj_1": [0.150, -0.148, -0.182], "obj_2": [-0.150, -0.152, -0.178],
               "obj_3": [-0.154, 0.148, -0.178], "obj_4": [0.146, 0.152, -0.182]}
    for p, off in leg_off.items():
      self.part_shapes[p] = ("cylinder", [0.30 * s, 0.025 * s], q_leg)
      self.scatter_orientations[p] = leg_scatter
      self.assembly_relative_poses[p] = ["obj_0", np.array(off) * s]
      self.mate_axes[p] = np.array([0.0, 0.0, 1.0])         # leg pressed up onto the seat bottom
    self.assembly_dependencies = {p: ["obj_0"] for p in leg_off}
    if self.n_parts >= 8:
      # backrest joins the seat (the sub-assembly join); posts mate onto the backrest.
      self.part_shapes["obj_5"] = ("box", [0.23 * s, 0.05 * s, 0.20 * s, r], q_back)
      self.scatter_orientations["obj_5"] = np.array([0.7772, 0.0, 0.6292, 0.0])  # backrest lies flat
      self.assembly_relative_poses["obj_5"] = ["obj_0", np.array([0.19, 0.0, 0.25]) * s]
      self.mate_axes["obj_5"] = np.array([0.0, 0.0, -1.0])  # sub-assembly lowered onto the seat
      self.assembly_dependencies["obj_5"] = ["obj_0"]
      post_off = {"obj_6": [-0.009, 0.152, -0.052], "obj_7": [-0.006, -0.148, -0.052]}
      for p, off in post_off.items():
        self.part_shapes[p] = ("cylinder", [0.33 * s, 0.025 * s], q_post)
        self.scatter_orientations[p] = post_scatter
        self.assembly_relative_poses[p] = ["obj_5", np.array(off) * s]   # post -> backrest
        self.mate_axes[p] = np.array([0.0, 0.0, -1.0])
        self.assembly_dependencies[p] = ["obj_5"]


def make_assembly(kind, n_parts):
  """Assembly factory for the CLI: cross (cube cross), cube (2x2x2 grid), arch (bridge --
  known-flawed star, see class doc), arch-seq (arch shape built as a sequential chain),
  cube-plates (2x2x2 as two stacked 2x2 plates), cube-plates-2x2x3 (three stacked 2x2
  plates, 12 cubes), cuboid-tower (laid-down 2:1:1 cuboids stacked along z -- the first
  non-cube assembly). The plates family generalizes to any WxDxH via CubePlatesAssembly."""
  kinds = {"cross": (CubeCrossAssembly, 7), "cube": (CubeGridAssembly, 8),
           "arch": (ArchAssembly, 7), "arch-seq": (ArchSequentialAssembly, 7),
           "cube-plates": (CubePlatesAssembly, 8),
           "cube-plates-2x2x3": (lambda n_parts: CubePlatesAssembly(2, 2, 3, n_parts=n_parts), 12),
           "cuboid-tower": (CuboidTowerAssembly, 8),
           "car": (CarAssembly, 5),
           # mini car: whole car scaled to width=0.03 -> wheels are discs of diameter
           # 0.8w*2 = 0.048, comfortably inside the two-finger's ~0.078 inner gap, so the
           # fingers can take a wheel ACROSS THE DIAMETER (radial side grasp) -- the
           # grasp-DB's curved-surface entries, which the two-finger keeps (vacuum drops).
           "car-mini": (lambda n_parts: CarAssembly(n_parts=n_parts, width=0.03), 5),
           "slot-pin": (SlotPinAssembly, 3),
           "chair": (ChairAssembly, 8)}
  cls, max_n = kinds[kind]
  return cls(n_parts=min(n_parts, max_n))

def _part_index(part):
  # "obj_3" -> 3, used only to give subassemblies a stable, readable ordering
  return int(part.split("_")[1])


def get_subassemblies(assembly):
  # traverse the dependency graph to collect the parts that each of the subassemblies consist of.
  #
  # a (multi-part) subassembly is any set of parts that is closed under the
  # assembly dependencies: if a part is in the set, everything it depends on
  # must be too. These are exactly the states the assembly can pass through
  # while it is being built up. In poset terms, the order ideals of the
  # dependency DAG.
  deps = {p: set(assembly.assembly_dependencies.get(p, [])) for p in assembly.objects}

  # enumerate all dependency-closed subsets by growing from the empty set,
  # only ever adding a part once all of its dependencies are already present.
  ideals = {frozenset()}
  frontier = [frozenset()]
  while frontier:
    current = frontier.pop()
    for p in assembly.objects:
      if p in current or not deps[p] <= current:
        continue
      grown = current | {p}
      if grown not in ideals:
        ideals.add(grown)
        frontier.append(grown)

  subassemblies = set(ideals)
  subassemblies.discard(frozenset())  # the empty assembly is not a subassembly

  # every individual part is a trivial subassembly as well, even if its
  # dependencies are not satisfied yet (it exists on its own before assembly).
  for p in assembly.objects:
    subassemblies.add(frozenset([p]))

  # deterministic output: parts within a subassembly sorted by index, and
  # subassemblies ordered by size then by their part indices.
  ordered = [tuple(sorted(s, key=_part_index)) for s in subassemblies]
  ordered.sort(key=lambda t: (len(t), [_part_index(p) for p in t]))
  return ordered

DEFAULT_FLOATING_EE = "src/rai/ur10/ur10_vacuum_floating.g"

# --- pose helpers, pose = [x, y, z, qw, qx, qy, qz] (rai convention) ---
_QUAT_R_CACHE = {}                                          # quat bytes -> R (read-only shared)


def _quat_to_R(q):
  # MEMOIZED: profiles show millions of calls over a HANDFUL of distinct quats (identity, the
  # 6 face-down rotations, a few anchors). Returned arrays are shared -- callers only read /
  # rebind (audited); never mutate them in place. Bounded: cleared if it ever grows large.
  qa = np.asarray(q, dtype=float)
  key = qa.tobytes()
  R = _QUAT_R_CACHE.get(key)
  if R is None:
    if len(_QUAT_R_CACHE) > 4096:
      _QUAT_R_CACHE.clear()
    w, x, y, z = qa
    R = _QUAT_R_CACHE[key] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
  return R


def _quat_mul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ])


def _pose_compose(a, b):
  # pose of b expressed in a's parent frame (apply a, then b)
  return np.concatenate([a[:3] + _quat_to_R(a[3:]) @ b[:3], _quat_mul(a[3:], b[3:])])


def _pose_invert(a):
  qi = np.array([a[3], -a[4], -a[5], -a[6]])  # conjugate (unit quaternion)
  return np.concatenate([-_quat_to_R(qi) @ a[:3], qi])


def _align_quat(u, v):
  # shortest-arc quaternion rotating unit vector u onto unit vector v
  u = u / np.linalg.norm(u)
  v = v / np.linalg.norm(v)
  d = float(np.dot(u, v))
  if d > 1 - 1e-8:
    return np.array([1.0, 0.0, 0.0, 0.0])
  if d < -1 + 1e-8:
    # antiparallel: rotate 180 deg about any axis perpendicular to u
    axis = np.cross(u, [1.0, 0.0, 0.0])
    if np.linalg.norm(axis) < 1e-6:
      axis = np.cross(u, [0.0, 1.0, 0.0])
    return np.concatenate([[0.0], axis / np.linalg.norm(axis)])
  axis = np.cross(u, v)
  q = np.array([1.0 + d, axis[0], axis[1], axis[2]])
  return q / np.linalg.norm(q)


def make_subassembly_scene(assembly, subassembly, contact=1):
  # fresh Config holding the subassembly's parts at their assembled relative poses,
  # expressed in the first obj's frame (ref part at the origin). Frames are named by
  # their obj id; geometry comes from each part's shape spec.
  C = ry.Config()
  ref_pos = assembly.position_in_base(subassembly[0])
  for obj in subassembly:
    obj_in_ref = assembly.position_in_base(obj) - ref_pos
    (add_part_frame(C, obj, assembly.part_shape(obj), contact=contact)
        .setPosition(obj_in_ref))
  return C


def _gripper_rest_transforms(C, prefix):
  # from a freshly-added EE (root at the origin, so a frame's world pose == its pose
  # relative to the root): the tool-center frame name, the root->... transform needed to
  # place the tool center on a target, and the approach-axis correction.
  names = set(C.getFrameNames())
  tool_center = next(
      (n for n in (prefix + "ur_gripper_center", prefix + "ur_vacuum",
                   prefix + "ur_ee_marker", prefix + "ur_ee_link")
       if n in names),
      None,
  )
  assert tool_center is not None, "no tool-center frame found in gripper"
  gc_pose = np.asarray(C.getFrame(tool_center).getPose())
  base_rel_tool = _pose_invert(gc_pose)
  ee_pose = np.asarray(C.getFrame(prefix + "ur_ee_link").getPose())
  approach_axis = _quat_to_R(gc_pose[3:]).T @ (gc_pose[:3] - ee_pose[:3])
  q_correction = _align_quat(approach_axis, np.array([0.0, 0.0, 1.0]))
  return tool_center, base_rel_tool, q_correction


def _ee_root_pose(target_pose, base_rel_tool, q_correction):
  # world pose for the EE root so its tool center lands on target_pose (aligned to +z).
  target_pose = np.asarray(target_pose, dtype=float)
  tool_world = np.concatenate([target_pose[:3], _quat_mul(target_pose[3:], q_correction)])
  return _pose_compose(tool_world, base_rel_tool)


def place_floating_ee(C, target_pose, gripper_path=None, prefix="ee_"):
  # add a free-floating end effector to C and place it so its tool center (the grasp
  # point) sits at target_pose, with the EE's approach axis aligned to target's +z.
  # different EEs name their tool center differently and extend along different axes
  # (two-finger: +z, vacuum: +x); both are auto-detected. target_pose is a 7-vector
  # [x,y,z,qw,qx,qy,qz] in C's world frame. returns the tool-center frame name.
  if gripper_path is None:
    gripper_path = os.path.join(os.path.dirname(__file__), DEFAULT_FLOATING_EE)

  C.addFile(gripper_path, namePrefix=prefix)
  tool_center, base_rel_tool, q_correction = _gripper_rest_transforms(C, prefix)
  base_world = _ee_root_pose(target_pose, base_rel_tool, q_correction)
  root = C.getFrame(prefix + "ur_base")
  root.setPosition(base_world[:3])
  root.setQuaternion(base_world[3:])
  return tool_center


# outward unit normal of each cube face, in the object frame. The grasp_id (0..5)
# indexes this list (for a cube it is a face; for a mesh it would index a grasp DB).
_CUBE_FACE_NORMALS = [
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
]


def _mat_to_quat(R):
  # rotation matrix (columns = frame axes) -> [w, x, y, z] quaternion (rai convention)
  m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
  tr = m00 + m11 + m22
  if tr > 0.0:
    s = np.sqrt(tr + 1.0) * 2.0
    w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
  elif m00 > m11 and m00 > m22:
    s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
    w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
  elif m11 > m22:
    s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
    w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
  else:
    s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
    w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
  q = np.array([w, x, y, z])
  return q / np.linalg.norm(q)


def _grasp_quat_for_normal(normal):
  # a grasp frame whose z-axis (approach) points from the standoff back into the cube,
  # i.e. along -normal; x/y are chosen orthonormal, deterministically.
  z = -normal / np.linalg.norm(normal)
  ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
  x = np.cross(ref, z)
  x /= np.linalg.norm(x)
  y = np.cross(z, x)
  return _mat_to_quat(np.column_stack([x, y, z]))


# the 6 canonical PLACEMENT orientations (design doc: PlacementId = (stable_face, slot)):
# rotate the group so cube face f's outward normal points DOWN (rests on that face).
# Face 5 (normal [0,0,-1]) is the identity -- the default upright placement.
_FACE_DOWN_QUATS = [_align_quat(n, np.array([0.0, 0.0, -1.0])) for n in _CUBE_FACE_NORMALS]
_IDENTITY_FACE = 5


# grasps per cube face. 1 = a single (vacuum-like) grasp; 2 = the two in-plane
# orientations a two-finger gripper has (90 deg apart, since it is 180 deg-symmetric).
# A property of the GRIPPER, not the cube: set it via set_rolls_per_face BEFORE any
# grasp precomputation (grasp_ids encode (face, roll), so it must not change mid-run).
# The two-finger NEEDS both rolls: with a single roll, side-face grasps put one finger
# under the cube (into the table) and are all infeasible.
_ROLLS_PER_FACE = 1
_N_CUBE_GRASPS = len(_CUBE_FACE_NORMALS) * _ROLLS_PER_FACE


def set_rolls_per_face(n):
  """Configure the grasps-per-face for the gripper in use (vacuum: 1, two-finger: 2).
  Call once at startup, before precompute_grasps / solve_assembly_problem."""
  global _ROLLS_PER_FACE, _N_CUBE_GRASPS
  _ROLLS_PER_FACE = int(n)
  _N_CUBE_GRASPS = len(_CUBE_FACE_NORMALS) * _ROLLS_PER_FACE


# DENSE grasps (two-finger): add, per base face grasp, the two WRIST-TILTED variants -- the
# pose rotated +-45 deg about the finger-CLOSING axis (grasp-local y, measured on the robotiq:
# the pads separate along gripper y) THROUGH THE TOOL CENTER. The pads' normal is invariant
# under a rotation about y, so the pinch stays a flat FACE grasp at the same contact -- only
# the wrist/approach swings into the diagonal "between the faces" (reaches past obstacles a
# face-aligned approach hits). NOT a roll about the approach axis (that would put the fingers
# across the part's edges).
_DENSE_GRASPS = False


def set_dense_grasps(flag):
  """Enable the +-45deg wrist-tilted grasp variants (two-finger). Call once at startup,
  before any grasp precomputation (grasp ids index the per-part DB)."""
  global _DENSE_GRASPS
  _DENSE_GRASPS = bool(flag)


# EXTRA PLACE CELLS (experimental, OFF by default): classic `place` always lands a group at
# its reference part's ORIGINAL scatter anchor -- parts at the scatter extremes then get set
# down at the table edge where half the arms cannot reach them. With N > 0, places() ALSO
# offers up to N TRANSLATED anchors from the fixed cell list below (near the table centre,
# the most-reachable region). Cell placements are distinguished in the STATE via the place
# slot tag ("slot", ("cell", k), face) -- poses are NOT part of the state key -- and carry
# the landing anchor in the state's pose payload (the same mechanism declutter stowing uses).
# Infeasible cells (occupied / colliding) are rejected by the cheap layer like any placement.
_EXTRA_PLACE_CELLS = 0
_PLACE_CELL_XY = ((0.0, 0.0), (0.0, 0.35), (0.0, -0.35), (0.3, 0.0), (-0.3, 0.0),
                  (0.3, 0.35), (-0.3, -0.35), (-0.3, 0.35), (0.3, -0.35))
_PREFER_CENTER = False


def set_extra_place_cells(n):
  """Offer up to n extra TRANSLATED placement anchors per (group, face) in places().
  Cell 0 is the TABLE CENTRE (0, 0) -- the most-reachable spot for every arm."""
  global _EXTRA_PLACE_CELLS
  _EXTRA_PLACE_CELLS = int(n)


def set_prefer_center(flag):
  """With extra place cells active: yield the CELL placements (centre first) BEFORE the
  classic scatter-anchor placement, so at equal f the search tries central placements
  first (successor order is the FIFO tie-break). Pure ordering -- no pruning."""
  global _PREFER_CENTER
  _PREFER_CENTER = bool(flag)


# GOAL SPEC (configurable; extensible to a specific target pose). DEFAULT (module and
# CLI alike): the finished assembly must REST ON THE TABLE, hands-off (placed AND
# released). set_goal(on_table=False) restores the historical complete-in-air goal
# (--no-goal-on-table on the CLI). `pose` is the hook for a future specific target
# placement (face / cell / anchor) -- see _goal_pose_matches.
_GOAL_MODE = {"on_table": True, "hands_off": True, "pose": None}


def set_goal(on_table=True, hands_off=None, pose=None):
  """Configure the search goal. on_table: the completed assembly must rest on the table.
  hands_off: no gripper may hold it (defaults to on_table -- 'placed' means set down AND
  released). pose: (future) a specific target placement on the table; None = any placement."""
  _GOAL_MODE["on_table"] = bool(on_table) or pose is not None
  _GOAL_MODE["hands_off"] = bool(on_table if hands_off is None else hands_off)
  _GOAL_MODE["pose"] = pose


def _goal_pose_matches(state, group, pose, assembly):
  """Whether the on-table `group` sits at the target `pose`. STUB for the future specific-
  pose goal -- when `pose` describes a placement (face / cell / anchor xy+yaw), compare it
  to the group's actual placement here. `pose is None` never reaches this."""
  raise NotImplementedError("specific goal pose not implemented yet")


# BRACE-FOR-ASSEMBLY-ONLY (experimental, OFF by default). Domain assumption: a brace (a 2nd
# grasp via add_grasp) is EVER needed only to stabilise a target during an ASSEMBLE (regrasp is
# the separate `handover`). So a brace should be added right before the assemble it enables and
# dropped right after -- never carried around. The literal "must assemble right after add_grasp"
# rule is PATH-DEPENDENT (non-Markovian). The equivalent STATE-BASED rule is Markovian and is
# what we enforce: a state in which some group is held by >=2 grippers (a brace is present) may
# only ASSEMBLE (consume the brace) or RELEASE (drop it) -- pick/place/handover/further-brace are
# forbidden while braced. The brace is fully in the state, so identical states get identical
# successors (no history). Effect: the search must pick/reorient the inserted part BEFORE
# bracing, cannot interleave unrelated work while braced, and one brace may still serve several
# consecutive assembles. COMPLETE under the assumption (any plan has a normal form with braces
# added-used-dropped contiguously); if the assumption is ever false it can miss plans -> flagged.
_BRACE_FOR_ASSEMBLY_ONLY = False


def set_brace_for_assembly_only(flag):
  global _BRACE_FOR_ASSEMBLY_ONLY
  _BRACE_FOR_ASSEMBLY_ONLY = bool(flag)


def _has_brace(state):
  # True iff some subassembly is held by >=2 distinct grippers (a brace, per add_grasp).
  for g in state.subassemblies():
    holders = set()
    for p in g:
      for c in state.supports.get(p, ()):
        if c[0] == "grasp":
          holders.add(c[1])
          if len(holders) >= 2:
            return True
  return False


_GRASP_POSE_CACHE = {}


# ---------------------------------------------------------------------------
# Per-part shape helpers: ONE place that knows how to realize a shape spec as a rai frame,
# and how to measure its extents. Shape spec: ("box",[sx,sy,sz,r]) | ("cylinder",[l,rad])
# (axis = local z) | ("sphere",[rad]) -- optionally with a THIRD element, a quaternion
# [w,x,y,z] posing the shape INSIDE the part frame (e.g. a wheel: a cylinder whose axis
# lies along the part's y). Rotated shapes are realized the rai way: the part frame stays
# shapeless and a rigid CHILD frame `<part>__shape` carries the rotated geometry (contact
# lives on the child; use set_part_contact / part_shape_frame to address it).
# ---------------------------------------------------------------------------

def _shape_quat(spec):
  return np.asarray(spec[2], dtype=float) if len(spec) > 2 else None


def _set_raw_shape(f, spec):
  kind, p = spec[0], spec[1]
  if kind == "box":
    f.setShape(ry.ST.ssBox, list(p))                       # [sx, sy, sz, corner_r]
  elif kind == "cylinder":
    f.setShape(ry.ST.cylinder, list(p))                    # [length, radius]
  elif kind == "sphere":
    f.setShape(ry.ST.sphere, list(p))                      # [radius]
  else:
    raise ValueError(f"unknown shape kind {kind!r}")
  return f


def _set_frame_shape(f, spec):
  # only valid for unrotated specs (shape aligned with the part frame).
  assert _shape_quat(spec) is None, "rotated shape needs add_part_frame (child frame)"
  return _set_raw_shape(f, spec)


def add_part_frame(C, name, spec, contact=1):
  q = _shape_quat(spec)
  f = C.addFrame(name)
  if q is None:
    return _set_raw_shape(f, spec).setContact(contact)
  shp = C.addFrame(f"{name}__shape", name)                 # rigid child carries the shape
  _set_raw_shape(shp, spec).setContact(contact)
  shp.setRelativeQuaternion(list(q))
  return f


def part_shape_frame(C, name):
  # the frame that CARRIES a part's collision shape: the `__shape` child for rotated
  # specs, the part frame itself otherwise.
  child = f"{name}__shape"
  return child if child in C.getFrameNames() else name


def set_part_contact(C, name, v):
  # contact toggle that works for both plain parts and rotated-shape parts.
  C.getFrame(part_shape_frame(C, name)).setContact(v)


def _shape_half_extents(spec):
  # SHAPE-frame half extents [hx, hy, hz] of the shape's bounding box (before any shape
  # rotation inside the part frame).
  kind, p = spec[0], spec[1]
  if kind == "box":
    return np.array([p[0], p[1], p[2]], dtype=float) / 2.0
  if kind == "cylinder":
    return np.array([p[1], p[1], p[0] / 2.0], dtype=float)
  if kind == "sphere":
    return np.array([p[0], p[0], p[0]], dtype=float)
  raise ValueError(f"unknown shape kind {kind!r}")


def _shape_half_extent_along(spec, d_part):
  # half extent of the shape along direction `d_part` given in the PART frame (accounts
  # for the spec's shape rotation). Exact for axis-aligned directions.
  q = _shape_quat(spec)
  d = d_part if q is None else _quat_to_R(q).T @ np.asarray(d_part, dtype=float)
  return float(np.abs(d) @ _shape_half_extents(spec))


def _shape_half_z(spec, R):
  # half extent along WORLD z when the PART frame is rotated by R (the shape's own
  # rotation inside the part frame is composed in). Exact for our axis-aligned placement
  # rotations; a safe over-approximation for general R.
  if spec[0] == "sphere":
    return _shape_half_extents(spec)[2]
  q = _shape_quat(spec)
  R_total = R if q is None else R @ _quat_to_R(q)
  return float(np.abs(R_total[2, :]) @ _shape_half_extents(spec))


def part_grasp_poses(spec, rolls=None, offset=0.01):
  """All candidate grasp poses of a part shape, in the part's OWN frame -- the per-part
  grasp DB the labeled tables index into (grasp_id = position in this list). V1 generator:
  the six axis-direction grasps at the FACE CENTER (approach along -direction, standoff =
  the shape's half-extent along that direction + offset) x `rolls` spins about the approach
  axis. For a cube this reproduces cube_grasp_pose exactly (same ids, same poses); for a
  cuboid the standoff per face follows its half-extents; for a cylinder the +-z grasps sit
  on the caps and the side grasps on the curved surface (vacuum allowed there, same force
  -- per current model decision); spheres get the same six directions at radius standoff.
  Rotated/off-center grasp variants are a later, purely additive extension."""
  if rolls is None:
    rolls = _ROLLS_PER_FACE
  sq = _shape_quat(spec)
  key = ("DB", spec[0], tuple(spec[1]),
         tuple(sq) if sq is not None else None, rolls, offset, _DENSE_GRASPS)
  hit = _GRASP_POSE_CACHE.get(key)
  if hit is not None:
    return hit
  h = _shape_half_extents(spec)
  poses = []
  for normal in _CUBE_FACE_NORMALS:                        # six axis directions (SHAPE frame)
    standoff = float(np.abs(normal) @ h) + offset
    for roll in range(rolls):
      angle = roll * (np.pi / rolls)
      roll_q = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
      q = _quat_mul(_grasp_quat_for_normal(normal), roll_q)
      pose = np.concatenate([standoff * normal, q])
      if sq is not None:                                   # rotate the grasp into the PART frame
        pose = np.concatenate([_quat_to_R(sq) @ pose[:3], _quat_mul(sq, pose[3:])])
      pose.setflags(write=False)
      poses.append(pose)
  if _DENSE_GRASPS:
    # wrist-tilted variants (see set_dense_grasps): SAME tool position + pinch, pose rotated
    # +-45deg about the grasp's CLOSING axis (local y, in the part frame) through the tool
    # center. Appended base-order per sign so grasp ids stay stable: [base | +45 | -45].
    for sgn in (1.0, -1.0):
      for base in list(poses[:len(_CUBE_FACE_NORMALS) * rolls]):
        a = _quat_to_R(np.asarray(base[3:])) @ np.array([0.0, 1.0, 0.0])
        half = sgn * (np.pi / 4.0) / 2.0
        q_t = np.concatenate([[np.cos(half)], np.sin(half) * a])
        tilted = np.concatenate([base[:3], _quat_mul(q_t, base[3:])])
        tilted.setflags(write=False)
        poses.append(tilted)
  hit = _GRASP_POSE_CACHE[key] = tuple(poses)
  return hit


def grasp_pose(assembly, part, grasp_id, offset=0.01):
  # grasp `grasp_id` of `part`, in the part's own frame (the shape-aware replacement for
  # raw cube_grasp_pose(gid, cube_size) -- identical values for cube parts).
  return part_grasp_poses(assembly.part_shape(part), offset=offset)[grasp_id]


def part_grasp_surfaces(spec, rolls=None):
  # surface TYPE contacted by each grasp in part_grasp_poses (parallel list): "flat" or
  # "curved". Box faces are flat; a cylinder's +-z caps are flat, its side is curved; a
  # sphere is curved everywhere. Used to restrict the VACUUM to flat surfaces (a suction
  # cup cannot seal on strong curvature) -- the two-finger keeps curved grasps.
  if rolls is None:
    rolls = _ROLLS_PER_FACE
  kind = spec[0]
  out = []
  for normal in _CUBE_FACE_NORMALS:
    if kind == "box":
      surf = "flat"
    elif kind == "cylinder":
      surf = "flat" if abs(normal[2]) > 0.5 else "curved"  # caps flat, side curved
    else:                                                  # sphere
      surf = "curved"
    out.extend([surf] * rolls)
  if _DENSE_GRASPS:
    out = out * 3                                          # tilted variants pinch the SAME faces
  return out


def cube_grasp_pose(grasp_id, cube_size, offset=0.01):
  # grasp pose [x,y,z, qw,qx,qy,qz] in the grasped cube's OWN frame. grasp_id encodes
  # (face, roll): the face sets the approach (-normal); the roll spins the gripper about
  # that approach axis (in-plane finger orientation). cube_size comes from the assembly.
  # MEMOIZED (pure function of a handful of inputs; the cheap collision checker calls it
  # hundreds of thousands of times -- 13% of a whole search before caching). _ROLLS_PER_FACE
  # is part of the key: it changes what a grasp_id means.
  key = (grasp_id, cube_size, offset, _ROLLS_PER_FACE)
  hit = _GRASP_POSE_CACHE.get(key)
  if hit is not None:
    return hit
  face, roll = divmod(grasp_id, _ROLLS_PER_FACE)
  normal = _CUBE_FACE_NORMALS[face]
  standoff = cube_size / 2.0 + offset
  angle = roll * (np.pi / _ROLLS_PER_FACE)                 # 0, 90 deg for two-finger
  roll_q = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])  # about approach (local z)
  q = _quat_mul(_grasp_quat_for_normal(normal), roll_q)
  pose = np.concatenate([standoff * normal, q])
  pose.setflags(write=False)                               # shared across callers: freeze
  _GRASP_POSE_CACHE[key] = pose
  return pose


def _subasm_key(group):
  # canonical key for a subassembly: its parts in index order (so subassembly[0] is the
  # reference part). Used both to precompute grasps and to look them up.
  return tuple(sorted(group, key=_part_index))


def precompute_grasps(assembly, offset=0.01, gripper_path=None):
  """Pregenerate the reachable grasps for EVERY subassembly that can arise, once, up
  front (design doc: G = pregenerate_grasps). Returns {subassembly_key: labeled grasps}.
  The result is independent of the search state and of which robot grasps (all EEs are
  identical), so the search just looks it up -- no grasp computation in the hot loop."""
  table = {}
  for sub in get_subassemblies(assembly):        # all reachable subassemblies + singletons
    key = _subasm_key(sub)
    table[key] = compute_cube_grasps(assembly, key, offset, gripper_path,
                                     return_labeled=True)
  return table


def compute_cube_grasps(assembly, subassembly, offset=0.01,
                        gripper_path=None, return_labeled=False):
  # placeholder grasp planner for ONE subassembly: assume every part is a cube, generate
  # one grasp per face (6 total), keep the collision-free ones. To be replaced by a real
  # grasp planner. return_labeled=True -> [(part, grasp_id, pose)]; else -> [pose].
  # This is preprocessing -- call it via precompute_grasps, not inside the search loop.
  if gripper_path is None:
    gripper_path = os.path.join(
        os.path.dirname(__file__), "src/rai/ur10/ur10_vacuum_floating.g"
    )

  cube_size = assembly.cube_size
  ref_pos = assembly.position_in_base(subassembly[0])

  # collision-check scene: the subassembly parts, in the ref frame, PLUS one pre-loaded
  # gripper (added once, then just repositioned per candidate -- avoids re-parsing the
  # gripper .g for every face). A fresh copy per query, because rai caches the broadphase.
  base_scene = make_subassembly_scene(assembly, subassembly)
  base_scene.addFile(gripper_path, namePrefix="ee_")
  _, base_rel_tool, q_correction = _gripper_rest_transforms(base_scene, "ee_")

  def grasp_is_collision_free(grasp_pose, grasped_obj):
    # move the pre-loaded EE so its tool center hits the grasp, then check the EE against
    # every part except the one being grasped (touching it is allowed). ONLY EE-involving
    # pairs may reject: the subassembly's INTERNAL part-part contacts are the mates
    # themselves (flush-seated parts touch at d=-0.0) and are irrelevant to whether the
    # gripper fits -- a binary getCollisionFree() here wrongly killed EVERY grasp on any
    # flush-mated subassembly (found via the car: wheel caps seat flush on the body).
    C_col = ry.Config()
    C_col.addConfigurationCopy(base_scene)
    base_world = _ee_root_pose(grasp_pose, base_rel_tool, q_correction)
    root = C_col.getFrame("ee_ur_base")
    root.setPosition(base_world[:3])
    root.setQuaternion(base_world[3:])
    set_part_contact(C_col, grasped_obj, 0)
    return not any((a.startswith("ee_") or b.startswith("ee_")) and d < -1e-9
                   for a, b, d in C_col.getCollisions())

  # a VACUUM cannot seal on curved surfaces -> restrict it to flat-surface grasps (grasp
  # ids stay positions in the FULL per-shape DB, so id-based lookups are unaffected -- the
  # table simply never contains curved candidates for the vacuum). The two-finger grasps
  # curved surfaces fine and keeps everything. --allow-curved-grasps overrides the drop (an
  # idealised suction on cylinders/spheres, e.g. to test whether curved-side grasps unblock
  # thin cylinder legs a real cup could not seal on).
  vacuum = "vacuum" in os.path.basename(gripper_path) and not _ALLOW_CURVED_GRASPS

  labeled = []
  for obj in subassembly:
    # obj origin expressed in the reference (first obj) frame
    obj_in_ref = assembly.position_in_base(obj) - ref_pos
    # candidate grasps come from the part's own grasp DB (shape-aware; grasp_id indexes
    # the part's set). The DB pose is in the part frame; shift it into the subassembly
    # ref frame (parts are axis-aligned, so translation only).
    surfaces = part_grasp_surfaces(assembly.part_shape(obj))
    for grasp_id, local in enumerate(part_grasp_poses(assembly.part_shape(obj),
                                                      offset=offset)):
      if vacuum and surfaces[grasp_id] == "curved":
        continue
      grasp = np.concatenate([obj_in_ref + local[:3], local[3:]])

      if grasp_is_collision_free(grasp, obj):
        labeled.append((obj, grasp_id, grasp))

  return labeled if return_labeled else [g[2] for g in labeled]

def compute_assembly_grasps_and_wrenches(C, assembly):
  # make all intermediate assemblies
  subassemblies = get_subassemblies(assembly)
  # show_subassemblies(C, assembly, subassemblies, [0, 0, 0.5])

  # compute grasps: for now, look at special cases of cubes
  grasps = {}

  for subassembly in subassemblies:
    grasps[subassembly] = compute_cube_grasps(assembly, subassembly)

  return grasps

# ===========================================================================
# Search state
# ===========================================================================
#
# The world is a scene graph. Each part (body) has exactly one KINEMATIC parent
# (table / a robot gripper / another part) -- this is the tree that determines its
# pose. Separately, each body has a SET of supports -- the contacts currently holding
# it up. Parent and supports are kept apart on purpose: a part can be parented to the
# table yet also be braced by a gripper, or mated to a neighbour yet also held. The
# question "may this contact be removed?" is answered from `supports`, never `parent`.
#
# A ContactId records one supporter of a body:
#   ("grasp", robot, part, grasp_id) -- held by a robot with grasp `grasp_id` (an index
#                                  into that part's pregenerated grasp set; not a face) (grounding)
#   ("table",)                  -- resting on the table                   (grounding)
#   ("mate", parent_part)       -- placed/mated onto another part   (internal, relayed)
# Grounding contacts tie a body to the world; a mate only relays support to its parent,
# so a subassembly is supported iff *some* member has a grounding contact.

GROUNDING_CONTACTS = ("grasp", "table")


def reference_part(group):
  # canonical "first" part of a subassembly (lowest index). Poses and grasps of a
  # subassembly are expressed relative to this part (matches compute_cube_grasps,
  # which uses subassembly[0]); it is also the subassembly's kinematic base here.
  return min(group, key=_part_index)


def is_grounding(contact):
  return contact[0] in GROUNDING_CONTACTS


def _grasp_holds(assembly, group, gpart):
  # does a grasp on `gpart` count as HOLDING `group`? A no_hold part (e.g. a wheel) may hold
  # only a LONE part -- itself, being picked / pressed in as the inserter -- never a multi-part
  # subassembly. So you can put a wheel on the car, but you can't then hold the car by it.
  return assembly.can_hold(gpart) or len(group) == 1


# LATE BINDING (robot-anonymous search). When on, State.key() relabels robots to canonical
# ids so states that differ ONLY in which robot holds what collapse to one -- killing the
# robot-assignment redundancy that makes the search blow up as robots are added, and making
# it GRACEFUL (extra idle robots add no states). Sound for interchangeable (floating) grippers;
# with fixed-base arms it plans robot-free and the concrete robots are bound/verified afterward.
_ANON_ROBOTS = False


def set_anonymous_robots(on=True):
  global _ANON_ROBOTS
  _ANON_ROBOTS = bool(on)


def _relabel_robots(parent, supports):
  # canonical robot ids: order the holders by the (grasped_part, grasp_id) they hold (unique
  # per robot), then relabel every ("gripper", r) / ("grasp", r, ...) to that rank. Robot
  # permutations map to the same relabelled parent/supports.
  held = {}
  for p, cs in supports.items():
    for c in cs:
      if c[0] == "grasp":
        held[c[1]] = (c[2], c[3])
  canon = {r: i for i, r in enumerate(sorted(held, key=held.get))}
  new_parent = {p: (("gripper", canon[c[1]]) if c[0] == "gripper" else c)
                for p, c in parent.items()}
  new_supports = {p: frozenset(((c[0], canon[c[1]]) + tuple(c[2:])) if c[0] == "grasp" else c
                               for c in cs)
                  for p, cs in supports.items()}
  return new_parent, new_supports


def wrench_sufficient(contacts, external=None):
  # STUB (design doc §5.2): stability / wrench balance is ignored for now. Any
  # non-empty contact set is treated as able to resist the load. Replace this body
  # with a real force-balance check later; the call sites already pass the contact
  # set, the situation's external wrench, and (later) the per-contact capabilities.
  return True


class State:
  """Discrete scene-graph state of the assembly search (the hashable identity).

  Corresponds to the design doc's `NodeId`. The continuous witness (robot configs)
  is deferred; `poses` here is a lightweight payload for reference/vis, NOT part of
  the identity.

  Fields
  ------
  parent   : {part: carrier}     carrier is ("table",), ("gripper", robot), or
                                 ("part", parent_part). The single kinematic parent.
  grasp    : {part: grasp_id}    grasp of the *kinematic* gripper parent, if held.
                                 grasp_id indexes the part's pregenerated grasp set
                                 (shape-agnostic: a face for a cube, a DB index for a mesh).
  place    : {part: slot_id}     the table slot, if parented to the table.
  supports : {part: frozenset[ContactId]}  everything holding the part up (>= the
                                 parent's own contact; plus braces / mates).
  poses    : {part: pose7}       world pose payload (continuous; not hashed).
  """

  def __init__(self, parent, grasp=None, place=None, supports=None, poses=None):
    self.parent = dict(parent)
    self.grasp = dict(grasp or {})
    self.place = dict(place or {})
    self.supports = {p: frozenset(cs) for p, cs in (supports or {}).items()}
    self.poses = dict(poses or {})

  # --- structure ---
  def bodies(self):
    return list(self.parent.keys())

  def subassemblies(self):
    # connected components of parts joined by mate edges (union-find over "part" parents)
    root = {p: p for p in self.parent}

    def find(x):
      while root[x] != x:
        root[x] = root[root[x]]
        x = root[x]
      return x

    for p, carrier in self.parent.items():
      if carrier[0] == "part":
        root[find(p)] = find(carrier[1])

    comps = {}
    for p in self.parent:
      comps.setdefault(find(p), set()).add(p)
    return [frozenset(s) for s in comps.values()]

  # --- supports / statics ---
  def grounding_contacts(self, group):
    # grounding contacts (grasp / table) over all members of a subassembly
    return frozenset(c for p in group for c in self.supports.get(p, ()) if is_grounding(c))

  def is_supported(self, group, assembly=None, gripper_kind=None):
    # a group is supported iff it has grounding contacts AND they can hold it. With
    # gripper_kind None this is the TOPOLOGICAL baseline (grounded + wrench stub); with a
    # gripper kind it is the real static-equilibrium check (gravity only -- callers add
    # insertion loads separately). NO-HOLD parts (assembly.no_hold) don't count as holding a
    # multi-part group, so a car braced only by a wheel is UNSUPPORTED (see _grasp_holds).
    grounding = self.grounding_contacts(group)
    if assembly is not None and assembly.no_hold:
      grounding = frozenset(c for c in grounding
                            if c[0] != "grasp" or _grasp_holds(assembly, group, c[2]))
    if len(grounding) == 0:
      return False
    if gripper_kind is None:
      return wrench_sufficient(grounding)
    return _group_stable(self, assembly, group, gripper_kind)

  def robots_holding(self):
    return {c[1] for cs in self.supports.values() for c in cs if c[0] == "grasp"}

  def free_robots(self, roster):
    return [r for r in roster if r not in self.robots_holding()]

  # --- goal ---
  def is_complete(self, assembly):
    # every part joined into a single subassembly
    comps = self.subassemblies()
    return len(comps) == 1 and set(comps[0]) == set(assembly.objects)

  def is_goal(self, assembly):
    # the assembly is complete: every part joined into a single subassembly. By default we
    # do NOT require it to rest on the table -- held-in-the-air counts (every reachable
    # state is supported by construction anyway). set_goal(on_table=True) additionally
    # demands it rest ON THE TABLE, hands-off by default, and (future) at a specific pose.
    if not self.is_complete(assembly):
      return False
    g = _GOAL_MODE
    if not (g["on_table"] or g["pose"] is not None):
      return True
    main = set(assembly.objects)                          # the single group (complete)
    if not _group_is_on_table(self, main):
      return False
    if g["hands_off"] and _group_holders(self, main):
      return False
    if g["pose"] is not None and not _goal_pose_matches(self, main, g["pose"], assembly):
      return False
    return True

  # --- identity ---
  def key(self):
    # MEMOIZED: a State is never structurally edited after it is first queried (successors
    # finish their edits before yielding, same invariant _state_part_poses relies on), so the
    # identity key is stable once computed. It was being rebuilt on every __hash__/__eq__ --
    # millions of times in a search -- and the sorts dominated the profile. Cached lazily so
    # the (rare) construction paths that never hash a state pay nothing. _ANON_ROBOTS is fixed
    # for the duration of a solve, so it cannot change under a cached key.
    k = getattr(self, "_key", None)
    if k is not None:
      return k
    parent, supports = self.parent, self.supports
    if _ANON_ROBOTS:                                       # LATE-BINDING: robot-anonymous key
      parent, supports = _relabel_robots(parent, supports)  # merge robot permutations
    k = self._key = (
        tuple(sorted(parent.items())),
        tuple(sorted(self.grasp.items())),
        tuple(sorted(self.place.items())),
        tuple(sorted((p, tuple(sorted(cs))) for p, cs in supports.items())),
    )
    return k

  def __eq__(self, other):
    return isinstance(other, State) and self.key() == other.key()

  def __hash__(self):
    h = getattr(self, "_hash", None)
    if h is None:
      h = self._hash = hash(self.key())
    return h

  def __repr__(self):
    def where(group):
      holders = sorted({c[1] for p in group for c in self.supports.get(p, ()) if c[0] == "grasp"})
      on_table = any(("table",) in self.supports.get(p, ()) for p in group)
      tags = (["table"] if on_table else []) + [f"@{r}" for r in holders]
      return "+".join(tags) if tags else "FLOATING"

    parts = []
    for group in sorted(self.subassemblies(), key=lambda g: sorted(g)):
      parts.append("{" + ",".join(sorted(group)) + "}:" + where(group))
    return "State(" + ", ".join(parts) + ")"


def get_robots(C):
  # robot ids present in the scene, by frame-naming convention "a<idx>_..."
  robots = set()
  for name in C.getFrameNames():
    head = name.split("_", 1)[0]
    if len(head) >= 2 and head[0] == "a" and head[1:].isdigit():
      robots.add(head)
  return sorted(robots)


def get_initial_state(C, assembly):
  """Initial search state read from the initial scene: every part is its own singleton
  subassembly, resting on the table at its scanned pose, and nothing is held."""
  parent, place, supports, poses = {}, {}, {}, {}
  for i, p in enumerate(assembly.objects):
    parent[p] = ("table",)
    place[p] = ("slot", i, _IDENTITY_FACE)
    supports[p] = frozenset({("table",)})
    poses[p] = np.asarray(C.getFrame(p).getPose())
  # ORIGINAL anchors, recorded on the assembly: the CLASSIC place restores its ref part's
  # pose to this anchor, so the state key <-> world-pose mapping stays a BIJECTION even when
  # cell/stow placements have moved the pose payload around (poses are NOT part of the key;
  # without the restore, a classic re-place after a cell placement would alias two different
  # worlds under one key).
  assembly._initial_anchor = {p: np.asarray(v, dtype=float).copy() for p, v in poses.items()}
  return State(parent=parent, place=place, supports=supports, poses=poses)


class Node:
  # a search node: a State plus search bookkeeping (the doc's Node = id + witness + g)
  def __init__(self, state, cost=0.0, parent=None, action=None):
    self.state = state
    self.cost = cost
    self.parent = parent
    self.action = action

# ===========================================================================
# Actions (search edges)
# ===========================================================================
#
# For now, the 3 FUSED primitives, wrench ignored, assembly == placement:
#   pick(r, base, face)     : a free robot grabs an on-table subassembly and lifts it.
#   place(r, base)          : a robot sets its held subassembly back on the table (opens).
#   assemble(r, part, onto) : a robot mates its held (single) part onto the on-table
#                             assembly per the manual, then lets go (mate now supports it).
# The unfused primitives (add_grasp / release / set_down) are deferred; they slot in
# right here once reachability/wrench give handovers/bracing something to be forced by.


def _copy_edit_support(state):
  # helper: a fresh State copy whose dicts can be edited in place
  return State(state.parent, state.grasp, state.place, state.supports, state.poses)


def _group_is_on_table(state, group):
  return any(("table",) in state.supports[p] for p in group)


def _group_holders(state, group):
  return {c[1] for p in group for c in state.supports[p] if c[0] == "grasp"}


def _group_root(state, group):
  # the subassembly's kinematic root: the one part whose parent is not another part
  # (it is the part attached to the table or a gripper). Unique for a valid tree.
  for p in group:
    if state.parent.get(p, (None,))[0] != "part":
      return p
  return reference_part(group)


def _rerooted(parent, new_root):
  # re-root the kinematic tree at new_root by reversing the chain of ("part", q) edges
  # from new_root up to its current root. Returns a new parent dict; the caller sets
  # new_root's own carrier (gripper/table). Lets us grab a subassembly by ANY part, not
  # just its base -- e.g. hold it by a leaf when the base's faces are all buried.
  newp = dict(parent)
  path = [new_root]
  cur = new_root
  while newp[cur][0] == "part":
    cur = newp[cur][1]
    path.append(cur)
  for child, par in zip(path, path[1:]):                  # reverse child->par to par->child
    newp[par] = ("part", child)
  return newp


def picks(state, assembly, robots, grasp_table=None, canonical=False):
  # a free robot grabs an on-table, not-yet-held subassembly on a collision-free face of
  # ANY of its parts, and lifts it (that part -> gripper; the tree is re-rooted there).
  #
  # grasp_table: {subassembly_key: labeled grasps} from precompute_grasps -- a lookup, no
  # grasp computation here. (If None, grasps are computed on the fly; used only by the
  # viz helpers, never the search.)
  #
  # canonical=True breaks ROBOT symmetry only: with interchangeable (floating) grippers
  # the gripper pose depends on the grasp, not on which robot holds it, so we assign only
  # the lowest-index free robot. UNSOUND once robots have fixed bases / different
  # reachability. Grasp choice is NOT canonicalized: now that state_collision_free sees
  # the gripper, different grasp faces give different gripper collisions.
  free = state.free_robots(robots)
  if not free:
    return
  if canonical:
    free = free[:1]
  for group in state.subassemblies():
    if not _group_is_on_table(state, group) or _group_holders(state, group):
      continue
    key = _subasm_key(group)
    labeled = grasp_table[key] if grasp_table is not None else \
        compute_cube_grasps(assembly, key, return_labeled=True)
    for r in free:
      for gpart, grasp_id, _pose in labeled:              # grasp ANY part of the group
        ns = _copy_edit_support(state)
        ns.parent = _rerooted(ns.parent, gpart)
        ns.parent[gpart] = ("gripper", r)
        for p in group:                                   # lift: drop table supports/slots
          ns.supports[p] = frozenset(c for c in ns.supports[p] if c != ("table",))
          ns.place.pop(p, None)
        ns.grasp[gpart] = grasp_id
        ns.supports[gpart] = ns.supports[gpart] | {("grasp", r, gpart, grasp_id)}
        yield ("pick", r, gpart, grasp_id), ns


def places(state, assembly, robots):
  # a robot sets a held subassembly back on the table and opens (its root -> table).
  #
  # CANONICALIZATION: the resulting on-table group is re-rooted at its reference part
  # (lowest index) and the table contact/slot tagged there -- NOT at whichever part
  # happened to be grasped. Root orientation and the table-tag carrier are bookkeeping,
  # not physics; if they leaked into the state identity, a pick+place cycle would return
  # to a "different" state and the closed list could never prune the (physically
  # identical) revisit -- the search then walks orbits of representations.
  for group in state.subassemblies():
    root = _group_root(state, group)
    if state.parent.get(root, (None,))[0] != "gripper":
      continue
    r = state.parent[root][1]
    ref = reference_part(group)                           # canonical on-table root
    for face in rest_faces(assembly, group):              # one placement per STABLE resting face
      def landed(slot_tag):
        ns = _copy_edit_support(state)
        for p in group:                                   # open r's grasp on this group
          ns.supports[p] = frozenset(
              c for c in ns.supports[p] if not (c[0] == "grasp" and c[1] == r))
        ns.grasp.pop(root, None)
        ns.parent = _rerooted(ns.parent, ref)             # canonical tree orientation
        ns.parent[ref] = ("table",)
        ns.place[ref] = ("slot", slot_tag, face)
        ns.supports[ref] = ns.supports[ref] | {("table",)}
        return ns
      def classic():
        ns = landed(_part_index(ref))
        # RESTORE ref's ORIGINAL anchor: poses are not part of the state key, so the classic
        # slot tag must always mean the original anchor pose -- otherwise a classic re-place
        # after a cell/stow placement would ALIAS two different worlds under one key.
        orig = getattr(assembly, "_initial_anchor", {}).get(ref)
        if orig is not None:
          ns.poses = dict(ns.poses)
          ns.poses[ref] = orig.copy()
        return ("place", r, root, face), ns

      def cell(k):
        # TRANSLATED anchor (see _EXTRA_PLACE_CELLS): same placement, landing at table cell
        # k -- the anchor override rides the pose payload, the cell id the slot tag.
        ns = landed(("cell", k))
        anchor = np.asarray(state.poses[ref], dtype=float).copy()
        anchor[0], anchor[1] = _PLACE_CELL_XY[k]
        ns.poses = dict(ns.poses)
        ns.poses[ref] = anchor
        return ("place", r, root, face, k), ns

      n_cells = min(_EXTRA_PLACE_CELLS, len(_PLACE_CELL_XY))
      if _PREFER_CENTER and n_cells:
        for k in range(n_cells):
          yield cell(k)
        yield classic()
      else:
        yield classic()
        for k in range(n_cells):
          yield cell(k)


def assembles(state, assembly, robots, gripper_kind=None):
  # Mate a HELD subassembly onto a supported target per the manual: the group's SEAM part (the
  # one whose mate parent lies in the target) welds on. A single held part is the size-1 case;
  # a whole MODULE (a finished plate) is the general case -- ONE action either way (`assemble`),
  # so real-IK covers both. The holder KEEPS gripping (its grasp becomes a brace); a separate
  # support-gated `release` frees it. So assemble only adds the mate -- it does not let go.
  connect_parts = getattr(assembly, "connect_parts", set())
  for group in state.subassemblies():
    gset = set(group)
    roots = [q for q in group if state.parent.get(q, (None,))[0] == "gripper"]
    if len(roots) != 1:
      continue                                            # the group must be HELD (one gripper)
    if any(("table",) in state.supports.get(q, ()) for q in group):
      continue                                            # ... in the air (not resting)
    holder = state.parent[roots[0]][1]
    # the SEAM: the single part of the held group whose mate parent lies OUTSIDE the group.
    # (0 -> a held base, nothing to mate; >1 -> ambiguous, not a clean insert.)
    external = [q for q in group if assembly.relative_pose(q) is not None
                and assembly.relative_pose(q)[0] not in gset]
    if len(external) != 1:
      continue
    p = external[0]
    onto = assembly.relative_pose(p)[0]
    target = next((g for g in state.subassemblies() if onto in g), None)
    if target is None:
      continue
    if not _mate_closure(assembly, p).issubset(gset | set(target)):
      # precedence: p's transitive predecessors must be built -- inside the held group (the
      # module's own chain) or in the target. Keeps the target an order ideal and forbids
      # mating a chain onto a floating mid-chain piece.
      continue
    if p in connect_parts and not assembly.connect_requires.get(p, set()).issubset(
        gset | set(target)):
      # a module JOIN: the seam's required last-cubes (the end of each plate's chain) must be
      # present -- in the held module (=> this plate complete) and the target (=> that plate
      # complete). Being a chain, needing the last cube needs the whole plate, so a partial
      # module has no complete seam to join with: the two-plates-then-join order is FORCED.
      continue
    if not state.is_supported(target, assembly, gripper_kind):
      continue                                            # target must be supported
    if gripper_kind is not None:
      # WRENCH (single part or whole module alike): the mate's press direction (mate_axis, a
      # single per-mate direction like a real manual) must satisfy BOTH
      #   (a) the target withstands the insertion reaction while the seam seats, and
      #   (b) [check_inserter] the HELD group -- its own weight/lever plus the seating reaction
      #       -- is transmittable through its grasp(s). Passing `group` to _group_stable prices
      #       in a heavy module's gravity automatically; for a single part it is the old check.
      d = assembly.mate_axis(p)
      if not _insertion_stable(state, assembly, target, gripper_kind, p, onto, d):
        continue
      if wrench.PARAMS.get("check_inserter", True) and not _inserter_stable(
          state, assembly, group, gripper_kind, p, onto, d):
        continue
      if (_STAGED_MATES and onto in assembly.staged_mates
          and assembly.staged_mates[onto] not in target):
        # (c) [--staged-mates] the press lands ON a staged, still-unfastened part: the GROUP
        # check above treats it as welded, but its OWN contacts (slot + grasps on it) must
        # take the press -- at every insertion-noise corner. Nominal press along the slot
        # axis is pure compression (passes unheld); a NOISY press adds shear + couple the
        # point slot cannot resist -> the part must be HELD while its fastener is hammered
        # in, even if it rests stably. (p == the fastener still counts: it fastens only
        # AFTER seating; the press happens before the weld.)
        if not _staged_press_stable(state, assembly, onto, gripper_kind,
                                    _insertion_load_variants(state, assembly, p, onto, d)):
          continue
    if _group_is_on_table(state, target) and not _on_table_mateable_cached(
        assembly, p, _placement_face(state, target)):
      # the mate would jam the seam into the table at the target's CURRENT rest pose/face
      # (a different face may admit it -- that is what reorientation placements are for).
      continue
    ns = _copy_edit_support(state)
    ns.parent = _rerooted(ns.parent, p)          # re-root the held group at its seam (no-op for
    ns.parent[p] = ("part", onto)                # a single part)... and weld the seam onto target
    for x in group:                              # no held part is a gripper child now; the grasp
      ns.grasp.pop(x, None)                      # CONTACTS stay in supports as braces until a
      ns.place.pop(x, None)                      # later `release`
    ns.supports[p] = ns.supports[p] | {("mate", onto)}
    yield ("assemble", holder, p, onto), ns


def _on_table_mateable_cached(assembly, part, face=_IDENTITY_FACE):
  # memoized _on_table_mateable (a pure geometric fact of the assembly, per placement
  # face); computed lazily so viz helpers and the search share one evaluation each.
  cache = getattr(assembly, "_otm_cache", None)
  if cache is None:
    cache = assembly._otm_cache = {}
  key = (part, face)
  if key not in cache:
    cache[key] = _on_table_mateable(assembly, part, face=face)
  return cache[key]


def add_grasps(state, assembly, robots, grasp_table=None):
  # a free robot adds a SECOND grasp (a brace) to an already-HELD subassembly -- the unfused
  # primitive behind HANDOVERS (add_grasp; release) and bracing. The new grasp is a SUPPORT
  # contact only; the kinematic `parent` is unchanged (the group stays rooted at its current
  # holder -- this is the whole point of parent(single)/supports(set)). A later release of the
  # original holder then re-roots the tree onto this grasp (see `releases`).
  free = state.free_robots(robots)
  if not free:
    return
  for group in state.subassemblies():
    if not _group_holders(state, group):
      continue                                            # only brace an already-held group
    key = _subasm_key(group)
    labeled = grasp_table[key] if grasp_table is not None else \
        compute_cube_grasps(assembly, key, return_labeled=True)
    taken = {(c[2], c[3]) for p in group for c in state.supports[p] if c[0] == "grasp"}
    for r in free:
      for gpart, grasp_id, _pose in labeled:              # grasp ANY free face of the group
        if (gpart, grasp_id) in taken:
          continue                                        # that grasp is already occupied
        ns = _copy_edit_support(state)
        ns.supports[gpart] = ns.supports[gpart] | {("grasp", r, gpart, grasp_id)}
        yield ("add_grasp", r, gpart, grasp_id), ns


def fused_handovers(state, assembly, robots, grasp_table=None):
  # FUSED handover: a free robot B takes a singly-held group from its holder A in ONE step
  # (B grasps a free face, A releases, the tree re-roots to B). Net holder count is UNCHANGED,
  # so unlike the unfused add_grasp;release it introduces no higher-h two-holder intermediate
  # -- it is the heuristically neutral, air analogue of the place;pick regrasp. (Unfused
  # add_grasp stays for genuine bracing, where you keep both holders.)
  free = state.free_robots(robots)
  if not free:
    return
  for group in state.subassemblies():
    if len(_group_holders(state, group)) != 1:
      continue                                              # clean 1->1 take-over only
    root = _group_root(state, group)
    if state.parent.get(root, (None,))[0] != "gripper":
      continue                                              # must be gripper-held (in the air)
    rA = state.parent[root][1]
    key = _subasm_key(group)
    labeled = grasp_table[key] if grasp_table is not None else \
        compute_cube_grasps(assembly, key, return_labeled=True)
    taken = {(c[2], c[3]) for p in group for c in state.supports[p] if c[0] == "grasp"}
    for rB in free:
      for gpart, grasp_id, _pose in labeled:                # B grabs any free face...
        if (gpart, grasp_id) in taken:
          continue
        ns = _copy_edit_support(state)
        ns.supports[root] = frozenset(                      # ...A lets go...
            c for c in ns.supports[root] if not (c[0] == "grasp" and c[1] == rA))
        ns.grasp.pop(root, None)
        ns.supports[gpart] = ns.supports[gpart] | {("grasp", rB, gpart, grasp_id)}
        ns.parent = _rerooted(ns.parent, gpart)             # ...tree re-roots to B
        ns.parent[gpart] = ("gripper", rB)
        ns.grasp[gpart] = grasp_id
        yield ("handover", rB, gpart, grasp_id), ns


def releases(state, assembly, robots, gripper_kind=None):
  # a robot opens its gripper without setting the part down. Support-gated: legal only if
  # the subassembly is still supported once this grasp is gone. This is the whole "you
  # can't always let go" rule -- topologically it just means "something else still grounds
  # it"; with wrench it means the REMAINING contacts can still balance gravity (so a
  # cantilever cannot be let go until the loop/ground carries it).
  for group in state.subassemblies():
    for p in group:
      for c in [c for c in state.supports[p] if c[0] == "grasp"]:
        r = c[1]
        ns = _copy_edit_support(state)
        ns.supports[p] = frozenset(x for x in ns.supports[p] if x != c)
        if not ns.is_supported(group, assembly, gripper_kind):
          continue                    # letting go would drop it -> not allowed
        if ns.parent.get(p) == ("gripper", r):
          # HANDOVER completion: the released grasp was the kinematic root, but another grasp
          # still holds the group -> re-root the tree onto that remaining holder (canonical
          # choice: the lowest-index still-grasped part). This is the case add_grasp enables.
          remaining = [(cc[2], cc[1], cc[3]) for pp in group
                       for cc in ns.supports[pp] if cc[0] == "grasp"]
          if not remaining:
            continue                  # no other holder to take it (should not happen if supported)
          q, r2, gid2 = min(remaining, key=lambda t: _part_index(t[0]))
          ns.parent = _rerooted(ns.parent, q)
          ns.parent[q] = ("gripper", r2)
          ns.grasp.pop(p, None)
          ns.grasp[q] = gid2
        yield ("release", r, p), ns


# LRU ROBOT ORDERING (utilization debugging knob). Successor generators iterate robots in
# roster order, and the greedy/milestone searches take the FIRST feasible successor -- so the
# first free robot wins every tie and plans over-use a1 while the rest idle. With this on,
# `successors` orders robots LEAST-RECENTLY-USED first (recency tracked as a non-identity
# payload on yielded states: _lru_last/_lru_step; State.key() ignores it). Deterministic (no
# RNG; stable sort keeps roster order for ties), pure successor ORDERING -- the reachable set,
# heuristics, and w-bounds are untouched. Expect: better robot spread (-> depgraph-export
# parallelism), possibly slightly worse action counts (forced alternation can add reaches).
_LRU_ROBOTS = False


def set_lru_robots(on=True):
  global _LRU_ROBOTS
  _LRU_ROBOTS = bool(on)


# IN-HAND REORIENTATION (Tier-2 skills backlog, v1). New macro action
# ("inhand_reorient", r, p, new_gid): a robot rotates its held part to a DIFFERENT grasp
# without place+pick -- kills table-regrasp cycles (pick wrong face -> place -> re-pick).
# V1 SIMPLIFYING ASSUMPTIONS (each an explicit extension point):
#   (A) FULL in-hand connectivity: any grasp -> any other grasp of the part. Later: a
#       per-part in-hand connectivity graph gates which (old_gid, new_gid) pairs the skill
#       can actually reach (finger gaiting / pivoting limits).
#   (B) NO transition capability change: grip support is assumed unchanged mid-reorient.
#       Later: a reduced TRANSITION wrench capability (grip momentarily weakened) and
#       optional EXTERNAL support (extrinsic dexterity) via the swappable-cap machinery.
#   (C) LONE parts only (single-part group, single grasp): one gripper cannot in-hand
#       re-grasp a multi-part assembly, and a braced part's other grasps would tear.
# NOTE the labeled/collapsed heuristics do NOT model this edge, so with the flag on they
# may OVERESTIMATE (inadmissible): greedy/milestones/LNS guidance is fine; do not trust
# bounded-search optimality proofs.
_INHAND_REORIENT = False


def set_inhand_reorient(on=True):
  global _INHAND_REORIENT
  _INHAND_REORIENT = bool(on)


def inhand_reorients(state, assembly, robots, grasp_table=None):
  # a robot rotates its held LONE part onto another grasp of the same part (v1 assumptions
  # in the flag doc above). Only when the part is SOLO-held (its single grasp contact is the
  # kinematic holder's) -- a second grasp (brace) would be torn by the rotation.
  for group in state.subassemblies():
    if len(group) != 1:
      continue                                             # (C) lone parts only
    p = next(iter(group))
    carrier = state.parent.get(p)
    if carrier is None or carrier[0] != "gripper":
      continue
    r = carrier[1]
    if len([c for c in state.supports.get(p, ()) if c[0] == "grasp"]) != 1:
      continue                                             # braced: reorient would tear it
    old_gid = state.grasp.get(p)
    key = _subasm_key(group)
    labeled = grasp_table[key] if grasp_table is not None else \
        compute_cube_grasps(assembly, key, return_labeled=True)
    for _gpart, gid, _pose in labeled:                     # _gpart == p (singleton group)
      if gid == old_gid:
        continue                                           # (A) any other grasp reachable
      ns = _copy_edit_support(state)
      ns.grasp[p] = gid
      ns.supports[p] = frozenset(
          c for c in ns.supports[p] if c[0] != "grasp") | {("grasp", r, p, gid)}
      yield ("inhand_reorient", r, p, gid), ns


# STAGED / PARTIAL MATES (A11, --staged-mates). A staged mate (assembly.staged_mates:
# {part: fastener}) is only a SLOT until the fastener is mated into the same group: the part
# is geometrically in place (kinematics/scene graph rigid, unchanged) but the FORCE model
# treats it as separable -- its own supports must balance its own gravity. V1 model, applied
# uniformly forward and in the labeled heuristics (so h mirrors the exact reachable set):
#   * a grasp ON the part counts as holding it (topological, like the no-wrench baseline);
#   * otherwise the SLOT alone must balance the part's gravity at the group's orientation --
#     a pure per-(part, face) LP fact (wrench.cap_slot vs gravity; gripper-independent), so
#     resting the assembly slot-opening-UP is discovered as an alternative to holding.
# Enforced as a SUCCESSOR filter (one point, every action gated) => the invariant "no
# reachable state has an unheld, unfastened, slot-insufficient part" holds globally; the
# fastener's own assemble upgrades the mate implicitly (fastener in group => check skipped).
_STAGED_MATES = False


def set_staged_mates(on=True):
  global _STAGED_MATES
  _STAGED_MATES = bool(on)


# ALLOW CURVED GRASPS (--allow-curved-grasps): let a VACUUM keep curved-surface grasps
# (normally dropped, since a suction cup cannot seal on a cylinder/sphere side). An idealised
# test knob -- e.g. to check whether curved-side grasps make thin cylinder legs graspable.
# Read in compute_cube_grasps; must be set BEFORE the grasp table is built.
_ALLOW_CURVED_GRASPS = False


def set_allow_curved_grasps(on=True):
  global _ALLOW_CURVED_GRASPS
  _ALLOW_CURVED_GRASPS = bool(on)


def _slot_holds_alone(assembly, part, face):
  # LP fact, memoized per (part, face): can the SLOT capability alone balance `part`'s gravity
  # when the assembly rests on placement face `face` (identity face = held/upright)? Gravity
  # acts at the part's centroid (zero torque about it), so the slot's point forces at the seat
  # must supply (m*g*z, 0) -- orientation is everything, the masses/lever lengths cancel.
  cache = getattr(assembly, "_slot_fact", None)
  if cache is None:
    cache = assembly._slot_fact = {}
  key = (part, face)
  if key not in cache:
    d = assembly.mate_axis(part)
    if d is None:
      cache[key] = True                                    # no mate axis: nothing to check
    else:
      R = _quat_to_R(np.asarray(_FACE_DOWN_QUATS[face], dtype=float))
      kind, params = assembly.part_shape(part)
      if kind == "box":
        half = float(np.dot(np.abs(d), np.asarray(params[:3], dtype=float))) / 2.0
      elif kind == "cylinder":
        half = abs(d[2]) * params[0] / 2.0 + (1 - abs(d[2])) * params[1]
      else:                                                # sphere
        half = params[0]
      seat = R @ (half * np.asarray(d, dtype=float))       # seat face center, centroid-relative
      G, ub = wrench.cap_slot(seat, R @ np.asarray(d, dtype=float))
      w = wrench.gravity_wrench([(assembly.mass(part), np.zeros(3))])
      bounds = [(0.0, None) for _ in ub]
      cache[key] = wrench.equilibrium_feasible_matrices(G, ub, w, bounds=bounds)
  return cache[key]


def _staged_press_stable(state, assembly, q, gripper_kind, load_variants):
  # per-part ROBUST press check for a staged, unfastened part q that a mate is being pressed
  # onto (v1 scope: q == the mate target `onto`). The group-level _insertion_stable treats q
  # as welded into the target; here q's OWN contacts -- its slot, any grasps on q, the table
  # if q touches it -- must balance q's gravity plus EVERY realization of the press reaction
  # (the insertion-noise corners; a singleton nominal load when the flag is off). All wrenches
  # about q's centroid (gravity torque-free there), verdicts memoized in _WRENCH_CACHE.
  d = assembly.mate_axis(q)
  if d is None:
    return True                                            # base/free part: no slot to check
  poses = _state_part_poses(state, assembly, lift=0.0)
  pos = np.asarray(poses[q][:3], dtype=float)
  Rq = _quat_to_R(np.asarray(poses[q][3:], dtype=float))
  d = np.asarray(d, dtype=float)
  seat = Rq @ (d * _shape_half_extent_along(assembly.part_shape(q), d))
  caps = [wrench.cap_slot(seat, Rq @ d)]
  derate = wrench.PARAMS.get("curved_grasp_torque_scale") is not None
  for c in sorted(state.supports.get(q, ())):
    if c[0] == "grasp":
      gp, n, t1, t2 = _grasp_frame(assembly, c[2], poses[c[2]], c[3])
      surf = _grasp_surface(assembly.part_shape(c[2]), c[3]) if derate else "flat"
      caps.append(wrench.grasp_capability(gp - pos, n, t1, t2, gripper_kind, surface=surf))
  if ("table",) in state.supports.get(q, ()):
    fp = _table_footprint(assembly, poses, [q])
    caps.append(wrench.table_capability([np.asarray(x, dtype=float) - pos for x in fp]))
  G = np.hstack([c[0] for c in caps])
  ub = np.concatenate([c[1] for c in caps])
  bounds = [(0.0, None if not np.isfinite(u) else float(u)) for u in ub]
  gbytes = (np.round(G, 4).tobytes(), np.round(ub, 4).tobytes())
  w_grav = wrench.gravity_wrench([(assembly.mass(q), np.zeros(3))])
  for f, pt in load_variants:
    w = w_grav + wrench.point_force_wrench(f, np.asarray(pt, dtype=float) - pos)
    key = (gbytes[0], gbytes[1], np.round(w, 4).tobytes())
    hit = _WRENCH_CACHE.get(key)
    if hit is None:
      hit = _WRENCH_CACHE[key] = wrench.equilibrium_feasible_matrices(
          G, ub, w, bounds=bounds, geo_key=gbytes)
    if not hit:
      return False
  return True


def _staged_ok(state, assembly):
  # the staged-mate support invariant (see block comment above). True when no group contains
  # a mated-but-unfastened staged part that is neither grasped nor slot-supported at the
  # group's orientation (held groups use the identity/upright convention, like the rest of
  # the discrete model).
  for group in state.subassemblies():
    for p, fastener in assembly.staged_mates.items():
      if p not in group or fastener in group:
        continue                                           # absent, or fastened -> weld
      sup = state.supports.get(p, ())
      if not any(c[0] == "mate" for c in sup):
        continue                                           # p not mated yet (loose/held alone)
      if any(c[0] == "grasp" and c[2] == p for c in sup):
        continue                                           # a hand on p holds it (v1)
      face = _placement_face(state, group) if _group_is_on_table(state, group) \
          else _IDENTITY_FACE
      if not _slot_holds_alone(assembly, p, face):
        return False
  return True


def successors(state, assembly, robots, grasp_table=None, canonical=False,
               gripper_kind=None, handovers=True, allow_place=True, bracing=False):
  # handovers=True (default) enables regrasp-in-air via the FUSED `handover`: a free robot
  # takes a held group from its holder in ONE heuristically-neutral edge (no higher-h
  # two-holder intermediate). That covers regrasp; the unfused `add_grasp` (a 2nd grasp you
  # KEEP -- genuine bracing/multi-hold) is a separate, rarely-needed capability behind
  # `bracing=` (off by default: it just adds branching everywhere). `release` reparents onto a
  # remaining holder either way. allow_place=False FORBIDS setting a held group back on the
  # table -> the search holds + regrasps in the air (no forced regime; just fewer options).
  if _LRU_ROBOTS:
    _last = getattr(state, "_lru_last", {})
    robots = sorted(robots, key=lambda r: _last.get(r, -1))  # stable: ties keep roster order

  def _gen():
    # ORDER MEASURED 2026-07-17 (assembles-first vs this picks-first): bit-identical searches
    # on cross-6/cross-7; on car-mini 2.1x fewer expansions but a 33% WORSE plan (cost 20->28
    # -- diving into the first workable mate commits before better variants are seen). Keep
    # picks-first; emission order is a tie-break lottery, don't retune it.
    yield from picks(state, assembly, robots, grasp_table=grasp_table, canonical=canonical)
    yield from assembles(state, assembly, robots, gripper_kind=gripper_kind)  # incl. module joins
    if handovers:
      yield from fused_handovers(state, assembly, robots, grasp_table=grasp_table)
    if bracing:
      yield from add_grasps(state, assembly, robots, grasp_table=grasp_table)
    if _INHAND_REORIENT:
      yield from inhand_reorients(state, assembly, robots, grasp_table=grasp_table)
    yield from releases(state, assembly, robots, gripper_kind=gripper_kind)
    if allow_place:
      yield from places(state, assembly, robots)
    # deferred (unfused): set_downs

  def _filtered():
    it = _gen()
    if _BRACE_FOR_ASSEMBLY_ONLY and _has_brace(state):
      # a brace is present -> only consume it (assemble) or drop it (release); see the flag doc.
      it = (x for x in it if x[0][0] in ("assemble", "release"))
    if _STAGED_MATES and assembly.staged_mates:
      # staged-mate invariant: gate EVERY successor here (one point, all actions) -- see
      # _staged_ok. Off / no staged mates => generator passes through untouched.
      it = (x for x in it if _staged_ok(x[1], assembly))
    yield from it

  if not _LRU_ROBOTS:
    yield from _filtered()
    return
  # stamp the LRU payload on every successor: the acting robot (and a handover's GIVER, read
  # off the pre-state's grasp contact) used at step k+1. Non-identity payload -- key() ignores.
  _step = getattr(state, "_lru_step", 0) + 1
  for a, ns in _filtered():
    nl = dict(_last)
    nl[a[1]] = _step
    if a[0] == "handover":
      for c in state.supports.get(a[2], ()):
        if c[0] == "grasp":
          nl[c[1]] = _step
    ns._lru_last, ns._lru_step = nl, _step
    yield a, ns


def assembly_order(assembly):
  # parts in a dependency-respecting order, base first: a part comes after everything in its
  # transitive closure. (For the chain DAGs here this is index order; the key keeps it
  # correct for any DAG.) The order the parts get mated in.
  return sorted(assembly.objects, key=lambda p: (len(_mate_closure(assembly, p)), _part_index(p)))


def context_scatter_state(assembly, context, face, poses):
  """The isolated scene for one optimistic assemble step: the parts in `context` assembled
  onto the base and resting on the table on `face`, and EVERY other part still a loose
  singleton on the table at its scanned pose. No braces, no held part yet -- phase 1 then
  picks the target part off this scene and mates it, so the per-part check exercises exactly
  the pick->mate progression the realizer will run, in isolation (which is what makes the
  per-part (grasp, face) choices decouple). `poses` is the world-pose payload."""
  parent, place, supports = {}, {}, {}
  base = assembly.base
  ctx = set(context)
  for q in assembly.objects:
    if q in ctx:
      if q == base:
        parent[q] = ("table",)
        supports[q] = frozenset({("table",)})
        place[q] = ("slot", _part_index(q), face)
      else:
        onto = assembly.relative_pose(q)[0]
        parent[q] = ("part", onto)
        supports[q] = frozenset({("mate", onto)})
    else:                                                 # still a loose part on the table
      parent[q] = ("table",)
      supports[q] = frozenset({("table",)})
      place[q] = ("slot", _part_index(q), _IDENTITY_FACE)
  return State(parent=parent, place=place, supports=supports, poses=dict(poses))



# ===========================================================================
# State visualization (debugging)
# ===========================================================================

def _group_rest_height(assembly, group, ref, face=_IDENTITY_FACE, anchor_q=None):
  # height of the reference part's CENTER above the table when the group rests on it,
  # rotated so `face` points down COMPOSED ONTO the anchor's own orientation (the group's
  # world rotation is face_quat o anchor_quat, see _group_rest_pose): the max over parts of
  # (part-bottom-to-ref-center), shape-aware via each part's half extent along world z at
  # that total orientation. For an all-cube identity-anchor group this is cs/2 - min_rel
  # (the classic rule). anchor_q=None means identity (all classic callers).
  R = _quat_to_R(_FACE_DOWN_QUATS[face])
  if anchor_q is not None:
    R = R @ _quat_to_R(np.asarray(anchor_q, dtype=float))
  ref_off = assembly.position_in_base(ref)
  return max(-(R @ (assembly.position_in_base(p) - ref_off))[2]
             + _shape_half_z(assembly.part_shape(p), R) for p in group)


def _group_rest_lift(assembly, group, ref, face=_IDENTITY_FACE, anchor_q=None):
  # how far the group's reference part must sit ABOVE its anchor height so the group's
  # LOWEST part rests on the table, for the given PLACEMENT face (group rotated so `face`
  # points down, composed onto the anchor's orientation). The anchor height carries ref's
  # half extent at the ANCHOR's orientation (plus any float), so the lift is rest_height
  # (total orientation) minus that half extent -- for uniform cubes the halves cancel and
  # this is the classic -min_rel; for shaped parts a reorientation changes rest height too.
  #
  # MEMOIZED ON THE ASSEMBLY: a pure function of (group, ref, face, anchor_q) given fixed
  # part shapes/offsets -- the heuristic build crafts a fresh context state per fact and
  # re-derived the same lifts hundreds of thousands of times (the dominant build cost).
  cache = getattr(assembly, "_rest_lift_cache", None)
  if cache is None:
    cache = assembly._rest_lift_cache = {}
  key = (_subasm_key(group), ref, face,
         None if anchor_q is None else np.round(np.asarray(anchor_q, dtype=float), 9).tobytes())
  hit = cache.get(key)
  if hit is not None:
    return hit
  # half_anchor cancels the half-extent encoded in the anchor's z -> always at the anchor's
  # TRUE orientation. The rest HEIGHT uses the actual landing orientation: for a REAL face
  # that is yawZ(anchor) o face (see _group_rest_pose), whose world-z extents equal the
  # face-only ones (z is yaw-invariant) -> anchor_q=None there; the identity face keeps the
  # anchor orientation (initial scatter poses).
  half_anchor = _shape_half_z(assembly.part_shape(ref),
                              _quat_to_R(anchor_q) if anchor_q is not None else np.eye(3))
  hq = anchor_q if face == _IDENTITY_FACE else None
  lift = _group_rest_height(assembly, group, ref, face, anchor_q=hq) - half_anchor
  cache[key] = lift
  return lift


def _placement_face(state, group):
  # the placement face of an on-table group, read from its reference part's slot entry
  # (("slot", idx, face)); identity for legacy/unknown entries.
  pl = state.place.get(reference_part(group))
  return pl[2] if pl is not None and len(pl) > 2 else _IDENTITY_FACE


def _anchor_yaw_quat(q):
  # the WORLD-Z yaw component of an anchor orientation (the only part of a placement anchor
  # that is free on the table; everything else is decided by the resting face).
  R = _quat_to_R(np.asarray(q, dtype=float))
  yaw = np.arctan2(R[1, 0], R[0, 0])
  return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _group_rest_pose(assembly, group, ref, anchor_pose, face):
  # the reference part's full rest pose for a placement: anchor x/y, rotated so `face`
  # points down, lifted so the group's lowest part touches the table.
  #
  # ORIENTATION composition (fixed 2026-07-17): for the IDENTITY face the anchor orientation
  # is preserved as-is (initial states: scatter orientations like flat wheels + any random
  # yaw). For a REAL face, only the anchor's WORLD-Z YAW survives, composed AFTER the face
  # rotation: q = yawZ(anchor) o face_quat. The old `face o anchor` composite rotated a yawed
  # anchor's z-axis OUT of vertical (a pre-yaw about part-z is a TILT once the face quat maps
  # that axis horizontal) -- --random-init's yaws produced tilted 'rest' poses that the
  # wrench gates then rightly rejected as unstable.
  anchor_pose = np.asarray(anchor_pose, dtype=float)
  if face == _IDENTITY_FACE:
    q = anchor_pose[3:]
  else:
    q = _quat_mul(_anchor_yaw_quat(anchor_pose[3:]), _FACE_DOWN_QUATS[face])
  pos = anchor_pose[:3].copy()
  pos[2] += _group_rest_lift(assembly, group, ref, face, anchor_q=anchor_pose[3:])
  return np.concatenate([pos, q])


def _state_part_poses(state, assembly, lift=0.35):
  # synthesize a world pose per part from the scene graph: each subassembly is laid out
  # around its base part's anchor pose; held subassemblies are raised by `lift` so it is
  # visually obvious they are off the table.
  #
  # MEMOIZED PER STATE (keyed by lift): the cheap checker and the wrench LP call this for
  # every feasibility query -- recomputing the layout dominated profiles. Callers may
  # REPLACE entries of the returned dict (the checker parks held groups) but must not
  # mutate the pose arrays in place, so a shallow copy per call is enough. Safe because a
  # State is never structurally edited after it is first queried (successors finish their
  # edits before yielding).
  cache = getattr(state, "_ppc", None)
  if cache is None:
    cache = state._ppc = {}
  hit = cache.get(lift)
  if hit is not None:
    return dict(hit)
  poses = {}
  for group in state.subassemblies():
    base = reference_part(group)                            # layout reference (obj_0)
    base_pose = np.asarray(state.poses[base], dtype=float).copy()
    # lifted iff the group is held and not resting on the table (independent of which
    # part is grasped, so it works after re-rooting onto a leaf).
    held = any(c[0] == "grasp" for p in group for c in state.supports.get(p, ()))
    on_table = any(("table",) in state.supports.get(p, ()) for p in group)
    if held and not on_table:
      base_pose[2] += lift
    elif on_table:
      # table rest rule: rotate per the PLACEMENT face and lift the reference so the
      # group's LOWEST part rests on the table.
      base_pose = _group_rest_pose(assembly, group, base, base_pose,
                                   _placement_face(state, group))
    base_in_asm = assembly.position_in_base(base)
    for p in group:
      rel = assembly.position_in_base(p) - base_in_asm     # translation, identity rot
      poses[p] = _pose_compose(base_pose, np.concatenate([rel, [1.0, 0.0, 0.0, 0.0]]))
  cache[lift] = poses
  return dict(poses)


# ===========================================================================
# Wrench / stability evaluation (bridges the discrete state to wrench.py's LP)
# ===========================================================================
#
# Mates are RIGID WELDS, so a subassembly is one rigid body: its stability depends only on
# the GROUNDING contacts (table / grasp) versus the external load (gravity + any insertion
# reaction). We read the contact GEOMETRY from the state's world poses and hand capability
# sets to wrench.equilibrium_feasible. Enabled per call via `gripper_kind` (None => the
# topological baseline is used instead; nothing here runs).

def _part_support_points_world(pose, spec):
  # candidate contact points of a part for the table support polygon, world frame. Box:
  # its 8 corners (per-axis half extents). Cylinder: 8 points on each cap rim. Sphere: the
  # bottom/top poles (a resting sphere touches at a point). The spec's shape rotation
  # (inside the part frame) is composed into R.
  c = np.asarray(pose[:3], dtype=float)
  R = _quat_to_R(np.asarray(pose[3:], dtype=float))
  sq = _shape_quat(spec)
  if sq is not None:
    R = R @ _quat_to_R(sq)
  kind, p = spec[0], spec[1]
  if kind == "box":
    h = _shape_half_extents(spec)
    return [c + R @ (np.array([sx, sy, sz]) * h)
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
  if kind == "cylinder":
    l, r = p[0], p[1]
    return [c + R @ np.array([r * np.cos(a), r * np.sin(a), sz * l / 2.0])
            for sz in (-1.0, 1.0) for a in np.arange(8) * (np.pi / 4.0)]
  if kind == "sphere":
    return [c + np.array([0.0, 0.0, sz * p[0]]) for sz in (-1.0, 1.0)]
  raise ValueError(f"unknown shape kind {kind!r}")


def _table_footprint(assembly, poses, group, tol=1e-3):
  # the support polygon of a table-resting group: the world contact points of whichever
  # parts sit at the group's lowest z (the ones actually touching the table).
  by_part = {p: _part_support_points_world(poses[p], assembly.part_shape(p)) for p in group}
  zmin = min(x[2] for cs in by_part.values() for x in cs)
  return [x for cs in by_part.values() for x in cs if x[2] <= zmin + tol]


def _rest_layout_poses(assembly, group, ref, face, anchor_q):
  # world poses of every part when `group` rests on the table at placement `face` composed
  # onto anchor_q, with the anchor at the origin. Mirrors the on-table branch of
  # _state_part_poses (so rest_faces judges the SAME geometry the search/checker will build).
  anchor_pose = np.concatenate([np.zeros(3), np.asarray(anchor_q, dtype=float)])
  base_pose = _group_rest_pose(assembly, group, ref, anchor_pose, face)
  base_in_asm = assembly.position_in_base(ref)
  return {p: _pose_compose(base_pose,
                           np.concatenate([assembly.position_in_base(p) - base_in_asm,
                                           [1.0, 0.0, 0.0, 0.0]]))
          for p in group}


def _rest_stable(assembly, group, ref, face, anchor_q, tol=1e-4, span_tol=1e-3):
  # Does `group` rest STABLY on the table at placement `face` (composed onto anchor_q)? The
  # parts touching the table (lowest support points) must span a 2-D polygon, not degenerate
  # to a line or a point -- only then can the contact resist tipping about a horizontal axis.
  # This is exactly what excludes a lone CYLINDER resting on its curved side (a line contact)
  # or a sphere (a point), while a box on any face (>=3 non-collinear corners) and a multi-
  # wheel group (>=3 contact points, e.g. the car on its wheels) stay stable. Same geometric
  # criterion trimesh.compute_stable_poses uses for meshes -- the off-axis generalisation is
  # the deferred change 2 [[rest-pose-offaxis-change2]].
  poses = _rest_layout_poses(assembly, group, ref, face, anchor_q)
  low = np.asarray(_table_footprint(assembly, poses, group, tol=tol))
  if len(low) < 3:
    return False                                             # point / line contact
  xy = low[:, :2] - low[:, :2].mean(axis=0)
  return float(np.linalg.svd(xy, compute_uv=False)[1]) > span_tol


def rest_faces(assembly, group):
  # PRECOMPUTED per-subassembly rest-pose TABLE: the canonical placement faces (a subset of
  # range(len(_FACE_DOWN_QUATS))) on which `group` rests STABLY, accounting for the reference
  # part's scatter anchor (so a wheel scattered cap-down is judged in its cap-down frame).
  # Replaces the old implicit "every one of the 6 faces is placeable" enumeration -- places(),
  # precompute_stable_faces and both labeled heuristics look placements up here, so the search
  # and the abstraction stay consistent (same faces => admissibility preserved). Cached on the
  # assembly, keyed by subassembly, like the grasp table.
  #   change-1 era: entries are still the 6 canonical face rotations, only stability-filtered;
  #   turning them into arbitrary per-group quats is change 2 [[rest-pose-offaxis-change2]].
  cache = getattr(assembly, "_rest_faces_cache", None)
  if cache is None:
    cache = assembly._rest_faces_cache = {}
  key = _subasm_key(group)
  hit = cache.get(key)
  if hit is None:
    grp = set(group)
    ref = reference_part(grp)
    anchor_q = assembly.scatter_orientations.get(ref, np.array([1.0, 0.0, 0.0, 0.0]))
    hit = cache[key] = tuple(f for f in range(len(_FACE_DOWN_QUATS))
                             if _rest_stable(assembly, grp, ref, f, anchor_q))
  return hit


def _grasp_frame(assembly, part, part_pose, grasp_id):
  # world frame of a grasp on a part: contact point (the grasped part's centre, matching the
  # original isotropic model) plus the APPROACH axis (grasp local z) and the two LATERAL axes
  # (local x/y) -- the axes feed the DIRECTIONAL grasp model (axial vs shear vs torque); the
  # isotropic model ignores them.
  gpose = _pose_compose(np.asarray(part_pose, dtype=float),
                        grasp_pose(assembly, part, grasp_id))
  R = _quat_to_R(np.asarray(gpose[3:], dtype=float))
  return np.asarray(part_pose[:3], dtype=float), R[:, 2], R[:, 0], R[:, 1]


# memoized equilibrium verdicts. The LP dominates the wrench-on search cost, but the result
# is invariant to a global translation of all contacts+loads and to absolute x/y position,
# so we canonicalise every query to the group centroid and round -- states that share the
# same discrete structure (parts, placement face, grounding contacts, applied load) then hit
# the same entry. Cleared per solve via reset_wrench_cache().
_WRENCH_CACHE = {}


def reset_wrench_cache():
  _WRENCH_CACHE.clear()


def _grounding_caps(state, assembly, group, gripper_kind):
  # Everything in the stability LP that does NOT depend on the applied load: the grounding
  # generator matrix (G [6xK], ub [K]) + its precomputed linprog `bounds` and rounded cache-
  # key BYTES, the group centroid, and the gravity wrench. TWO cache levels:
  #   L1 per state (state._gcaps)      -- repeated queries on one state (16 noise corners).
  #   L2 CONTENT-ADDRESSED on the ASSEMBLY -- everything except the centroid is TRANSLATION
  #      INVARIANT (G/w_grav are centroid-relative), so the caps depend only on the group,
  #      its ORIENTATION class (placement face + the reference part's anchor quat), which
  #      grasps hold it, whether it touches the table, the gripper kind, and the grasp-model
  #      params. The heuristic build crafts a FRESH context state per fact, so the L1 cache
  #      alone recomputed identical geometry thousands of times ("should be lookups" -- now
  #      it is; an L2 hit skips pose layout, footprint, and cap construction entirely).
  cache = getattr(state, "_gcaps", None)
  if cache is None:
    cache = state._gcaps = {}
  ckey = (_subasm_key(group), gripper_kind)
  hit = cache.get(ckey)
  if hit is not None:
    return hit
  members_sorted = _subasm_key(group)
  base = reference_part(group)
  on_table = any(("table",) in state.supports.get(p, ()) for p in group)
  grasps = tuple(sorted((c[2], c[3]) for p in group
                        for c in state.supports.get(p, ()) if c[0] == "grasp"))
  anchor_q = np.round(np.asarray(state.poses[base][3:], dtype=float), 9).tobytes()
  l2 = getattr(assembly, "_caps_cache", None)
  if l2 is None:
    l2 = assembly._caps_cache = {}
  l2key = (members_sorted, on_table, _placement_face(state, group) if on_table else None,
           anchor_q, grasps, gripper_kind, wrench.PARAMS.get("grasp_model"),
           wrench.PARAMS.get("curved_grasp_torque_scale"), _ROLLS_PER_FACE)
  anchor_pos = np.asarray(state.poses[base][:3], dtype=float)
  l2hit = l2.get(l2key)
  if l2hit is not None:
    # the centroid is the ONLY translation-dependent output; its offset from the group's raw
    # anchor position is content-invariant (rest lift + mean of rotated part offsets), so an
    # L2 hit needs NO pose layout at all -- just re-anchor.
    coff, w_grav, G, ub, bounds, gbytes = l2hit
    hit = cache[ckey] = (anchor_pos + coff, w_grav, G, ub, bounds, gbytes)
    return hit
  poses = _state_part_poses(state, assembly, lift=0.0)
  members = sorted(group, key=_part_index)                 # stable order -> stable cache key
  pos = {p: np.asarray(poses[p][:3], dtype=float) for p in members}
  centroid = np.mean([pos[p] for p in members], axis=0)    # canonicalise position away
  w_grav = wrench.gravity_wrench([(assembly.mass(p), pos[p] - centroid) for p in members])
  caps = []
  if any(("table",) in state.supports.get(p, ()) for p in members):
    fp = _table_footprint(assembly, poses, members)
    caps.append(wrench.table_capability([np.asarray(x, dtype=float) - centroid for x in fp]))
  derate = wrench.PARAMS.get("curved_grasp_torque_scale") is not None
  for p in members:
    for c in sorted(state.supports.get(p, ())):
      if c[0] == "grasp":
        if not _grasp_holds(assembly, members, c[2]):
          continue                                          # no_hold part can't brace a group
        gp, n, t1, t2 = _grasp_frame(assembly, c[2], poses[c[2]], c[3])  # c=(grasp,r,gpart,gid)
        surf = _grasp_surface(assembly.part_shape(c[2]), c[3]) if derate else "flat"
        caps.append(wrench.grasp_capability(gp - centroid, n, t1, t2, gripper_kind, surface=surf))
  G = np.hstack([c[0] for c in caps]) if caps else np.zeros((6, 0))
  ub = np.concatenate([c[1] for c in caps]) if caps else np.zeros(0)
  bounds = [(0.0, None if not np.isfinite(u) else float(u)) for u in ub]
  gbytes = (np.round(G, 4).tobytes(), np.round(ub, 4).tobytes())
  hit = cache[ckey] = (centroid, w_grav, G, ub, bounds, gbytes)
  l2[l2key] = (centroid - anchor_pos,) + hit[1:]           # store the ANCHOR-RELATIVE centroid
  return hit


def _group_stable(state, assembly, group, gripper_kind, applied=()):
  # static-equilibrium feasibility for a subassembly: can its grounding contacts balance
  # gravity plus any `applied` [(force_vec_world, point_world), ...] (e.g. insertion)? The
  # load-independent contacts come from _grounding_caps (cached); here we only add the applied
  # load to the gravity wrench and hit the LP cache / solve.
  centroid, w_grav, G, ub, bounds, gbytes = _grounding_caps(state, assembly, group, gripper_kind)
  w_ext = w_grav
  for f, pt in applied:
    w_ext = w_ext + wrench.point_force_wrench(f, np.asarray(pt, dtype=float) - centroid)
  geo_key = (gbytes[0], gbytes[1])
  key = (geo_key[0], geo_key[1], np.round(w_ext, 4).tobytes())
  hit = _WRENCH_CACHE.get(key)
  if hit is not None:
    return hit
  res = wrench.equilibrium_feasible_matrices(G, ub, w_ext, bounds=bounds, geo_key=geo_key)
  _WRENCH_CACHE[key] = res
  return res


_SURFACE_CACHE = {}


def _grasp_surface(spec, grasp_id):
  # "flat" | "curved" surface a grasp contacts, memoized (part_grasp_surfaces depends only on
  # the shape KIND, the roll count, and dense mode). Only consulted when
  # curved_grasp_torque_scale is set.
  key = (spec[0], _ROLLS_PER_FACE, _DENSE_GRASPS)
  surfaces = _SURFACE_CACHE.get(key)
  if surfaces is None:
    surfaces = _SURFACE_CACHE[key] = part_grasp_surfaces(spec)
  return surfaces[grasp_id]


def _group_stable_loaded(state, assembly, group, gripper_kind, load_variants):
  # ROBUST stability under an UNCERTAIN applied load: the group must balance EVERY worst-case
  # realization in `load_variants` (a list of (force_world, point) corners from
  # _insertion_uncertainty_variants). With insertion_uncertainty off that list is the single
  # nominal load, so this is exactly the old one-shot _group_stable check.
  return all(_group_stable(state, assembly, group, gripper_kind, applied=[lv])
             for lv in load_variants)


def _group_stable_affine(state, assembly, group, gripper_kind, f0, point, gens):
  # CONSERVATIVE robust check: one AFFINE contact-force policy must hold across the whole
  # uncertainty box (wrench.equilibrium_robust_affine), vs _group_stable_loaded's EXACT per-corner
  # test. Builds the nominal wrench w0 and the box's generator wrenches delta_j about the group
  # centroid, then a single LP. Feasibility here implies every corner is stable (sound); it is
  # stricter (rejects loads that would need the contacts to switch strategy per realization).
  centroid, w_grav, G, ub, bounds, _ = _grounding_caps(state, assembly, group, gripper_kind)
  f0 = np.asarray(f0, dtype=float); point = np.asarray(point, dtype=float)
  w0 = w_grav + wrench.point_force_wrench(f0, point - centroid)
  deltas = []
  for kind, axis, amt in gens:
    axis = np.asarray(axis, dtype=float)
    if kind == "f":                                         # transverse force amt*axis at `point`
      deltas.append(wrench.point_force_wrench(amt * axis, point - centroid))
    else:                                                   # point shift amt*axis -> pure couple
      deltas.append(np.concatenate([np.zeros(3), _cross3(amt * axis, f0)]))
  return wrench.equilibrium_robust_affine(G, ub, w0, deltas, bounds=bounds)


def _robust_stable(state, assembly, group, gripper_kind, f_world, point, mag):
  # dispatch the robust insertion-stability check by wrench.PARAMS['robust_mode']:
  #   'corners' (default, EXACT): every uncertainty-box corner must be balanceable, the contacts
  #             free to rechoose their force split per realization (fully-adjustable robustness).
  #   'affine'  (CONSERVATIVE): one fixed affine force-response policy must cover the whole box
  #             (an open-loop / fixed-compliance grasp) -- a single LP, stricter, still sound.
  # Identical to the plain nominal check when insertion_uncertainty is off (no generators).
  if wrench.PARAMS.get("robust_mode", "corners") == "affine" and wrench.PARAMS.get("insertion_uncertainty"):
    f0, pt, gens = _insertion_uncertainty_gens(f_world, point, mag)
    return _group_stable_affine(state, assembly, group, gripper_kind, f0, pt, gens)
  return _group_stable_loaded(state, assembly, group, gripper_kind,
                              _insertion_uncertainty_variants(f_world, point, mag))


def _cross3(a, b):
  # explicit 3-vector cross (np.cross carries huge per-call dispatch overhead; see wrench._cross3)
  return np.array([a[1] * b[2] - a[2] * b[1],
                   a[2] * b[0] - a[0] * b[2],
                   a[0] * b[1] - a[1] * b[0]])


def _transverse_basis(dvec):
  # two orthonormal vectors spanning the plane perpendicular to (unit) `dvec`.
  d = np.asarray(dvec, dtype=float)
  d = d / (np.linalg.norm(d) + 1e-12)
  ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
  t1 = _cross3(ref, d)
  t1 /= (np.linalg.norm(t1) + 1e-12)
  return t1, _cross3(d, t1)


def _insertion_uncertainty_gens(f_world, point, mag):
  # the seating force as an UNCERTAINTY BOX, returned as (nominal_force, nominal_point, gens)
  # where each gen is (apply-to, axis, half-width). Real seating is misaligned in two independent
  # ways: the press direction TILTS (angular) -> a transverse shear up to lateral*mag ("f"), and
  # the contact point SHIFTS (positional) -> the axial force lands up to offset m off-centre, a
  # couple ("p"). Each is one box axis in the plane transverse to the mate axis (2 or 4 axes).
  unc = wrench.PARAMS.get("insertion_uncertainty")
  f_world = np.asarray(f_world, dtype=float)
  point = np.asarray(point, dtype=float)
  if not unc:
    return f_world, point, []
  t1, t2 = _transverse_basis(f_world)
  lat = float(unc.get("lateral", 0.0)) * mag              # transverse-force half-widths (N)
  off = float(unc.get("offset", 0.0))                     # application-offset half-widths (m)
  gens = []                                               # (apply-to, axis, half-width)
  if lat > 0.0:
    gens += [("f", t1, lat), ("f", t2, lat)]              # angular tilt -> transverse force
  if off > 0.0:
    gens += [("p", t1, off), ("p", t2, off)]              # positional shift -> off-centre couple
  return f_world, point, gens


def _insertion_uncertainty_variants(f_world, point, mag):
  # the uncertainty box as its CORNERS (list of (force, point)). Because the set of balanceable
  # external wrenches is CONVEX, requiring every CORNER stable (_group_stable_loaded) is EXACT for
  # the box -- no sampling, no tunable count. (The box is a conservative OUTER polytope of the true
  # circular cone/disk, so it rejects marginally more, in the safe direction.) Off => [(nominal)].
  f_world, point, gens = _insertion_uncertainty_gens(f_world, point, mag)
  if not gens:
    return [(f_world, point)]
  out = []
  for bits in range(1 << len(gens)):                      # every sign combination = a box corner
    fv, pv = f_world.copy(), point.copy()
    for i, (kind, axis, amt) in enumerate(gens):
      s = 1.0 if (bits >> i) & 1 else -1.0
      if kind == "f":
        fv = fv + s * amt * axis
      else:
        pv = pv + s * amt * axis
    out.append((fv, pv))
  return out


def _insertion_load(state, assembly, part, onto, d=None):
  # the insertion reaction the mate of `part` exerts on the target during seating: a force
  # of magnitude wrench.insert_force(part) along THE mate's press direction (d, defaulting
  # to assembly.mate_axis(part) -- a single per-mate direction, like a real manual),
  # applied at the mate interface. On a table-supported target a
  # top-down seating pushes into the table (fine); a sideways one has no frictional
  # resistance and topples/slides it -- so it needs a hold. Returns (force_world, point_world).
  poses = _state_part_poses(state, assembly, lift=0.0)
  R = _quat_to_R(np.asarray(poses[onto][3:], dtype=float))  # target's world orientation
  rel = np.asarray(assembly.relative_pose(part)[1], dtype=float)
  if d is None:
    d = assembly.mate_axis(part)
  f_world = wrench.insert_force(part) * (R @ d)             # force ON target, seating dir
  point = np.asarray(poses[onto][:3], dtype=float) + R @ rel  # ~ part's seated position
  return f_world, point


def _inserter_load(state, assembly, part, onto, d=None):
  # the seating reaction ON THE INSERTED PART itself: the actor presses `part` in along the
  # mate's ASSEMBLY DIRECTION (assembly.mate_axis), so the mate pushes back with
  # insert_force opposite that press, applied at `part`'s pressing face. Expressed through
  # PART's current pose in the check state -- the press direction is fixed in the part's
  # own frame (parts mate axis-aligned), so the load-vs-GRASP-axis relation (compression vs
  # pull-off vs shear: the thing the grasp capability distinguishes) is exact wherever the
  # held part happens to be posed. (Gravity-vs-load angle is approximate, same class of
  # approximation as _insertion_load.)
  poses = _state_part_poses(state, assembly, lift=0.0)
  R = _quat_to_R(np.asarray(poses[part][3:], dtype=float))  # part's world orientation
  if d is None:
    d = assembly.mate_axis(part)
  f_world = wrench.insert_force(part) * (R @ (-d))          # reaction ON part: opposite press
  half_d = _shape_half_extent_along(assembly.part_shape(part), d)
  point = np.asarray(poses[part][:3], dtype=float) \
      + R @ (np.asarray(d, dtype=float) * half_d)           # at the pressing face (real lever)
  return f_world, point


def _insertion_load_variants(state, assembly, part, onto, d=None):
  # the target-side insertion reaction, as a list of worst-case realizations under the
  # insertion-uncertainty model (a singleton = the nominal load when the flag is off).
  f_world, point = _insertion_load(state, assembly, part, onto, d)
  return _insertion_uncertainty_variants(f_world, point, wrench.insert_force(part))


def _inserter_load_variants(state, assembly, part, onto, d=None):
  # the inserter-side seating reaction (on the held part), likewise as worst-case realizations.
  f_world, point = _inserter_load(state, assembly, part, onto, d)
  return _insertion_uncertainty_variants(f_world, point, wrench.insert_force(part))


def _insertion_stable(state, assembly, group, gripper_kind, part, onto, d=None):
  # `group` (the TARGET) withstands the robust insertion reaction of `part` seating onto it.
  f_world, point = _insertion_load(state, assembly, part, onto, d)
  return _robust_stable(state, assembly, group, gripper_kind, f_world, point, wrench.insert_force(part))


def _inserter_stable(state, assembly, group, gripper_kind, part, onto, d=None):
  # the HELD `group` transmits its own weight plus the robust seating reaction through its grasp(s).
  f_world, point = _inserter_load(state, assembly, part, onto, d)
  return _robust_stable(state, assembly, group, gripper_kind, f_world, point, wrench.insert_force(part))


# reference anchor poses for the wrench precompute: absolute position is irrelevant (the
# stability LP is canonicalised to the group centroid), so a single fixed table-height
# anchor per part suffices -- the layout / face rotation / rest lift are recomputed from it.
def _flat_poses(assembly):
  return {p: np.array([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]) for p in assembly.objects}


def precompute_stable_faces(assembly, gripper_kind):
  """For every assembly STEP (mate part p onto subassembly S), the set of table PLACEMENT
  FACES on which the step is both geometrically reachable AND wrench-stable to seat top-down
  (S itself stable resting on the table on that face, and stable under the insertion
  reaction). A bounded table keyed by (subassembly_key, part), computed once like the grasp
  table. Its uses:
    - an EMPTY set => the step cannot be done on the table at all -> it REQUIRES holding the
      assembly in the air (a cantilever); a NON-empty set => it needs the assembly reoriented
      onto one of those faces. This is exactly the wrench-forced structure a stability-aware
      heuristic needs, and it prunes on-table mate generation to the stable faces.
  Returns {} when gripper_kind is None (wrench disabled)."""
  if gripper_kind is None:
    return {}
  poses = _flat_poses(assembly)
  table = {}
  for group in get_subassemblies(assembly):
    S = frozenset(group)
    for p in assembly.objects:
      rel = assembly.relative_pose(p)
      if p in S or rel is None or rel[0] not in S:
        continue                                            # p must mate onto a part of S
      if not _mate_closure(assembly, p).issubset(S):
        continue                                            # S must be an order ideal for p
      onto = rel[0]
      faces = set()
      for f in rest_faces(assembly, S):                     # only STABLE rest faces of S
        if not _on_table_mateable_cached(assembly, p, f):
          continue                                          # mate site not reachable at f
        st = context_scatter_state(assembly, S, f, poses)
        if not _group_stable(st, assembly, S, gripper_kind):
          continue                                          # S topples resting on face f
        if not _insertion_stable(st, assembly, S, gripper_kind, p, onto):
          continue                                          # topples under the insertion
        faces.add(f)
      table[(_subasm_key(S), p)] = frozenset(faces)
  return table


# penetration the CHEAP checker tolerates before rejecting. The real IK's collision
# objective is a SOFT inequality (weight 1e1 + the NLP feasibility threshold), so it
# accepts grazing contacts of a few mm; a binary-exact cheap check would then be STRICTER
# than L3 on borderline contacts -- breaking its lower-bound property (cheap-reject must
# imply IK-reject). Real rejections (finger-into-table, neighbouring cubes) penetrate
# 18-57mm, grazes are <5mm: cleanly separated by this margin.
_CHEAP_PENETRATION_TOL = 0.005


def _mate_closure(assembly, part):
  # every part that must already be assembled before `part` can be mated WITHIN ITS MODULE:
  # transitive dependencies plus the chain of mate parents (both required by `assembles`).
  # A CONNECT part's mate edge is a module JOIN, not an intra-module build step, so it is NOT
  # traversed here -- that is exactly what keeps a module (e.g. a plate) a self-contained
  # order ideal, buildable before it is joined. The join precedence lives in connect_requires
  # (enforced by the connect action), not in this closure.
  connect = getattr(assembly, "connect_parts", set())
  need = set()
  frontier = [part]
  while frontier:
    p = frontier.pop()
    deps = list(assembly.dependencies_of(p))
    rel = assembly.relative_pose(p)
    if rel is not None and p not in connect:
      deps.append(rel[0])                                  # intra-module mate parent
    for q in deps:
      if q not in need:
        need.add(q)
        frontier.append(q)
  need.discard(part)
  return need


def _on_table_mateable(assembly, part, tol=None, face=_IDENTITY_FACE):
  # geometric FACT from the cheap layer, computed once per (part, placement face): can
  # `part` reach its mate pose while the assembly rests on the table on `face`? Scene =
  # world + the parts that must already be assembled (at their assembled offsets, rotated
  # per the placement, resting per the lift rule) + `part` at its (rotated) mate offset.
  # A collision (e.g. the mate slot pointing into the table) is a SOUND rejection: the
  # real scene only has MORE obstacles. This is what forces base-lifts / reorientations.
  if tol is None:
    tol = _CHEAP_PENETRATION_TOL
  C = make_world()
  table = C.getFrame("table")
  top = table.getPosition()[2] + table.getSize()[2] / 2.0
  # the CONTEXT group (everything that must already be assembled) rests on the table per
  # the placement rule: rotated so `face` points down, lifted so ITS lowest part touches
  # the table. `part` is then posed at its rotated mate offset -- does the mate fit?
  context = _mate_closure(assembly, part)
  q = _FACE_DOWN_QUATS[face]
  R = _quat_to_R(q)
  rest = _group_rest_height(assembly, context or {assembly.base}, assembly.base, face)
  base_pos = np.array([0.0, 0.0, top + rest])
  for p in sorted(context | {part}, key=_part_index):
    add_part_frame(C, p, assembly.part_shape(p)).setPosition(
        base_pos + R @ assembly.position_in_base(p)).setQuaternion(q)
  return all(d >= -tol for _a, _b, d in C.getCollisions())
