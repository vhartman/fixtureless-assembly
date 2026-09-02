"""Problem definition: assemblies, part geometry, grasps, and the discrete scene-graph
search model (State + successors). No feasibility/solver or visualization code here."""

import os
from dataclasses import dataclass

import robotic as ry
import numpy as np

from env import *
import wrench


@dataclass(frozen=True)
class ScrewSkill:
  """Execution contract for one helical mate.

  The symbolic planner still sees one atomic ``assemble`` action.  This record only refines
  how that action is executed: the already-grasped part moves from a backed-off,
  counter-rotated entry pose to the ordinary authored mate pose without changing grasp.
  ``angle`` is signed (radians), ``depth`` is axial travel (metres), and ``torque`` is the
  nominal pure couple (Nm) that both the target and inserter wrench checks must sustain.

  ``samples`` is a certification resolution, not a task-level discretisation: intermediate
  poses never become symbolic actions or planner states.  A future controller may implement
  internal release/rewind/re-engage while preserving this exact external contract; v1 sets
  ``single_grasp`` and rejects any certificate that cannot maintain it continuously.
  """
  angle: float = np.pi
  depth: float = 0.02
  torque: float = 0.25
  angular_speed: float = 0.75
  samples: int = 16
  single_grasp: bool = True

  def __post_init__(self):
    if not np.isfinite(self.angle) or abs(float(self.angle)) < 1e-9:
      raise ValueError("screw angle must be finite and non-zero")
    if not np.isfinite(self.depth) or float(self.depth) <= 0.0:
      raise ValueError("screw depth must be finite and positive")
    if not np.isfinite(self.torque) or float(self.torque) < 0.0:
      raise ValueError("screw torque must be finite and non-negative")
    if not np.isfinite(self.angular_speed) or float(self.angular_speed) <= 0.0:
      raise ValueError("screw angular speed must be finite and positive")
    if int(self.samples) < 2:
      raise ValueError("screw certification needs at least two intervals")
    if not self.single_grasp:
      raise ValueError("only single-grasp screw skills are implemented")

  def to_dict(self):
    return {"kind": "screw", "angle": float(self.angle), "depth": float(self.depth),
            "torque": float(self.torque), "angular_speed": float(self.angular_speed),
            "samples": int(self.samples), "single_grasp": bool(self.single_grasp)}


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
    self.mate_interfaces = {}                # part -> PART-FRAME contact-patch offset (see
                                             # mate_interface); {} = geometric default
    # EXECUTION REFINEMENTS: {mated_part: ScrewSkill|...}.  These do not add symbolic
    # actions.  They refine one existing assemble edge into a certified controller-owned
    # motion with the same pre/post symbolic state.  Keeping a typed per-mate map (rather
    # than a global ``--screw`` boolean) lets ordinary insertion and screw mates coexist.
    self.mate_skills = {}
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
    # Optional offsets for the nominal line layout. Long heterogeneous parts can overlap a
    # neighbour even when their frame centres obey the generic pitch. Random/grid layouts
    # deliberately ignore these offsets because they choose complete poses themselves.
    self.scatter_offsets = {}
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

  def grasp_lever(self, part):
    """Moment arm (m) of a grasp on `part`, in the WRENCH MODEL'S OWN convention.

    The LP builds gravity as `gravity_wrench([(mass, zeros(3))])` and reduces moments ABOUT
    THE PART'S CENTROID, so gravity contributes no torque there and the only lever in the
    model is the centroid-to-CONTACT offset -- a surface half-extent. It is therefore always
    SHORT. Centralised here (rather than re-derived per caller) because it is a property of
    the assembly's geometry: sim.py's execution-side wrench limits and any future
    inertial/LP term must agree on it. NB a gripper-frame-to-COM distance is NOT this
    quantity -- using that gave 326mm levers whose weight moment alone blew the torque cap."""
    spec = self.part_shapes.get(part)
    if not spec:
      return 0.05
    if spec[0] in ("mesh", "compound"):
      return float(np.max(_shape_half_extents(spec)))     # same convention: max half-extent
    kind, p = spec[0], np.asarray(spec[1], dtype=float)
    if kind in ("box", "ssBox"):
      return 0.5 * float(np.max(p[:3]))
    if kind == "cylinder":
      return 0.5 * float(max(p[0], 2.0 * p[1]))
    if kind == "sphere":
      return float(p[0])
    return 0.05

  def group_lever(self, parts, root, position_of):
    """Moment arm for a whole held GROUP welded to the gripper through `root`:
    how far the group's centroid sits off `root`, plus `root`'s own half-extent.
    `position_of(part) -> xyz` supplies world positions (rai frame, sim body, ...) so this
    stays free of any particular backend."""
    parts = list(parts) or [root]
    m = np.array([self.mass(q) for q in parts], dtype=float)
    pos = np.array([np.asarray(position_of(q), dtype=float) for q in parts])
    com = (m[:, None] * pos).sum(0) / max(float(m.sum()), 1e-9)
    off = float(np.linalg.norm(com - np.asarray(position_of(root), dtype=float)))
    return off + self.grasp_lever(root), float(m.sum()), com

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

  def mate_skill(self, part):
    return self.mate_skills.get(part)

  def screw_skill(self, part):
    skill = self.mate_skill(part)
    return skill if isinstance(skill, ScrewSkill) else None

  def configure_screw_skills(self, *, angle=None, depth=None, torque=None,
                             angular_speed=None, samples=None):
    """Override fields of every authored screw mate without changing its symbolic schema."""
    for part, old in list(self.mate_skills.items()):
      if not isinstance(old, ScrewSkill):
        continue
      self.mate_skills[part] = ScrewSkill(
          angle=old.angle if angle is None else float(angle),
          depth=old.depth if depth is None else float(depth),
          torque=old.torque if torque is None else float(torque),
          angular_speed=(old.angular_speed if angular_speed is None
                         else float(angular_speed)),
          samples=old.samples if samples is None else int(samples),
          single_grasp=True)

  def mate_contact_interface(self, part, n_axial=48, n_ang=32):
    """COMPUTED contact patch: the PART-FRAME point of `part`'s shape surface CLOSEST to its
    mate parent's shape at the assembled relative pose ('touches == is close to'). The
    physical interface for non-face mates -- e.g. a pin lying alongside a plate contacts on
    its SIDE at the overlap height, off the central axis, not at its end cap. Sampled
    surface (analytic point-to-shape distances), one-time per part; assign the result to
    mate_interfaces[part] in define(). Face-glued cubes keep the half-extent default (their
    contact is a full face; its centre equals the default)."""
    parent, rel = self.relative_pose(part)
    rel = np.asarray(rel, dtype=float)
    spec_p, spec_o = self.part_shape(part), self.part_shape(parent)

    def surface_points(spec):
      kind, params = spec[0], spec[1]
      sq = _shape_quat(spec)
      Rs = np.eye(3) if sq is None else _quat_to_R(np.asarray(sq, dtype=float))
      pts = []
      if kind == "compound":                               # union of the children's samples
        return np.vstack([surface_points(sub) + off
                          for sub, off in _compound_children(spec)])
      if kind == "mesh":
        m = _load_mesh(params)["mesh"]                     # deterministic samples: vertices
        vs = np.vstack([np.asarray(m.vertices, dtype=float),   # + face centroids
                        np.asarray(m.triangles_center, dtype=float)])
        return vs @ Rs.T
      if kind == "cylinder":
        L, r = float(params[0]), float(params[1])
        for z in np.linspace(-L / 2.0, L / 2.0, n_axial):
          for a in np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False):
            pts.append(Rs @ np.array([r * np.cos(a), r * np.sin(a), z]))
      else:                                                # box (sphere: centre suffices)
        hx, hy, hz = [float(v) / 2.0 for v in params[:3]]
        for axis, h in ((0, hx), (1, hy), (2, hz)):
          for s in (-h, h):
            u = [i for i in range(3) if i != axis]
            hu, hv = (hx, hy, hz)[u[0]], (hx, hy, hz)[u[1]]
            for a in np.linspace(-hu, hu, 12):
              for b in np.linspace(-hv, hv, 12):
                q = np.zeros(3); q[axis] = s; q[u[0]] = a; q[u[1]] = b
                pts.append(Rs @ q)
      return np.asarray(pts)

    def dist_to_shape(pts_local, spec):
      # distance of points (in the shape OWNER's part frame) to the shape's surface-or-interior
      kind, params = spec[0], spec[1]
      sq = _shape_quat(spec)
      Rs = np.eye(3) if sq is None else _quat_to_R(np.asarray(sq, dtype=float))
      if kind == "compound":                               # nearest of the children
        return np.min([dist_to_shape(pts_local - off, sub)
                       for sub, off in _compound_children(spec)], axis=0)
      q = pts_local @ Rs                                   # into the shape's own frame
      if kind == "mesh":
        import trimesh
        sd = trimesh.proximity.signed_distance(_load_mesh(params)["mesh"], q)
        return np.maximum(0.0, -np.asarray(sd, dtype=float))   # interior counts as touching
      if kind == "cylinder":
        L, r = float(params[0]), float(params[1])
        dr = np.maximum(np.linalg.norm(q[:, :2], axis=1) - r, 0.0)
        dz = np.maximum(np.abs(q[:, 2]) - L / 2.0, 0.0)
        return np.sqrt(dr * dr + dz * dz)
      half = np.array([float(v) / 2.0 for v in params[:3]])
      dd = np.maximum(np.abs(q) - half, 0.0)
      return np.linalg.norm(dd, axis=1)

    pts = surface_points(spec_p)                           # part-frame samples of part surface
    d = dist_to_shape(pts + rel, spec_o)                   # parent centre at -rel in part frame
    return pts[int(np.argmin(d))]

  def mate_interface_points(self, part, d=None):
    # LIST of part-frame contact-patch offsets (from the part's centre) for `part`'s mate --
    # where the seam transmits the seating reaction. mate_interfaces[part] may be a single
    # offset (a 3-vector) OR a list of them (a mate that touches at several places, e.g. a
    # bracket pinned at two points). Default: one point on the part's face along the press
    # axis (exact for face-glued cubes). Set mate_interfaces[part] explicitly for off-axis
    # or multi-point contacts (e.g. the chair posts join the backrest at their TOP END).
    mi = self.mate_interfaces.get(part)
    if mi is not None:
      arr = np.asarray(mi, dtype=float)
      return [arr] if arr.ndim == 1 else [np.asarray(p, dtype=float) for p in arr]
    if d is None:
      d = self.mate_axis(part)
    d = np.asarray(d, dtype=float)
    return [d * _shape_half_extent_along(self.part_shape(part), d)]

  def mate_interface(self, part, d=None):
    # the SINGLE point at which the net insertion press acts, for the wrench load: the
    # CENTROID of the mate's contact points. For one point this is that point (unchanged);
    # for several, the centroid carries the same NET force+moment as an equal split of a
    # parallel press across them (exact for the seating reaction; finite-mate-strength would
    # instead treat each point as its own reaction -- reserved). See mate_interface_points.
    pts = self.mate_interface_points(part, d)
    return pts[0] if len(pts) == 1 else np.mean(pts, axis=0)

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
    for part, skill in self.mate_skills.items():
      assert part in self.assembly_relative_poses, \
          f"mate skill assigned to non-mated part {part}"
      assert isinstance(skill, ScrewSkill), f"unsupported mate skill for {part}: {skill!r}"

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
                         min_sep=None, grid=None, grid_gap=0.1, center_scatter=False):
  # scatter the assembly's parts on the table as loose cubes (test scene). Geometry
  # (cube_size / cube_radius) comes from the assembly, not a global. `spacing` = the x-pitch
  # of the scatter line (default = historical 1/7 m ~ 0.143); a knob for analysing how much
  # search/IK pain comes from the cramped test scene (arms diving between close neighbours)
  # rather than the planner. Parts beyond `per_row` WRAP into further rows (y += spacing per
  # row) -- a single line of >8 parts walks off the reachable table. Row 0 is byte-identical
  # to the historical line, so existing <=8-part scenes are unchanged.
  #
  # INITIAL-STATE LAYOUT MODES (mutually exclusive; the ASSEMBLY class is untouched -- this is
  # purely where the parts START):
  #   default          -- the scatter line above.
  #   random_init      -- randomised square between the robots (seeded).
  #   perturb_start_pose -- the line + a small collision-free jitter.
  #   grid=<cols>      -- a TIGHT rectangular grid, `cols` wide, packed row-major and centred
  #                       on the table, pitch = cube_size*(1+grid_gap). With a small gap the
  #                       cubes sit shoulder-to-shoulder, so interior cubes are not directly
  #                       graspable -- the planner must RELOCATE perimeter cubes to reach the
  #                       ones the assembly needs. (cross-7, cols=2 -> a 2x3 block + 1.)
  cs = assembly.cube_size

  table_half = C.getFrame("table").getSize()[2] / 2.0

  for i, name in enumerate(assembly.objects):
      # scatter orientation (how the part initially LIES, e.g. wheels flat) and a
      # SHAPE-AWARE height: table top + half extent at that orientation + `float_height`
      # (historical default 15mm; for the classic cube that is exactly the old fixed 0.07).
      # The float propagates through the placement ANCHORS, so every later set-down keeps it.
      scatter_q = assembly.scatter_orientations.get(name)
      spec = assembly.part_shape(name)
      Rs = _quat_to_R(scatter_q) if scatter_q is not None else np.eye(3)
      z = table_half + _shape_half_z(spec, Rs) + float_height
      if grid is not None:                                 # tight grid, centred on the table
          cols = max(1, int(grid))
          rows = -(-len(assembly.objects) // cols)         # ceil
          r, c = divmod(i, cols)
          pitch = cs * (1.0 + grid_gap)
          pos = np.array([(c - (cols - 1) / 2.0) * pitch,
                          (r - (rows - 1) / 2.0) * pitch, z])
      else:
          # The historic line spans [-0.5,+0.5]. With a larger explicit pitch, reduce the
          # number of columns instead of placing late parts at x=0.7/0.9 beyond real-arm
          # reach. Default 1/7m spacing still yields eight columns and stays byte-identical.
          pitch = max(float(spacing), 1e-9)
          reachable_cols = int(np.floor(1.0 / pitch + 1e-9)) + 1
          max_cols = max(1, min(int(per_row), reachable_cols))
          # A centred standard scatter is a compact deterministic lattice, not a 1xN line:
          # with fixed pitch the latter's workspace span grows linearly in part count and put
          # opposite car-mini wheels outside a short arm's reach. Roughly square rows preserve
          # the declared density while making spatial extent grow as sqrt(N).
          cols = (min(max_cols, max(1, int(np.ceil(np.sqrt(len(assembly.objects))))))
                  if center_scatter else max_cols)
          row, col = divmod(i, cols)
          if center_scatter:
              rows = -(-len(assembly.objects) // cols)
              n_this_row = min(cols, len(assembly.objects) - row * cols)
              pos = np.array([(col - (n_this_row - 1) / 2.0) * spacing,
                              (row - (rows - 1) / 2.0) * spacing, z])
          else:
              pos = np.array([-0.5 + col * spacing, row * spacing, z])
          if random_init is None:
              pos = pos + np.asarray(assembly.scatter_offsets.get(name, np.zeros(3)),
                                     dtype=float)

      sq = _shape_quat(spec) if spec[0] != "compound" else None
      obj = C.addFrame(name).setParent(C.getFrame("table"))
      _extra_shape_fs = []
      if spec[0] == "compound":                            # one rigid child per sub-shape
        for k, (sub, off) in enumerate(_compound_children(spec)):
          shp = _set_raw_shape(C.addFrame(f"{name}__shape{k}", name), sub)
          ssq = _shape_quat(sub)
          if ssq is not None:
            shp.setRelativeQuaternion(list(ssq))
          shp.setRelativePosition([float(off[0]), float(off[1]), float(off[2])])
          shp.setContact(1)
          _extra_shape_fs.append(shp)
        shape_f = _extra_shape_fs.pop(0)
      elif sq is None:                                     # shape on the part frame itself
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
      _col = np.random.rand(3)                             # ONE draw, as before: the global
      shape_f.setColor(_col)                               # np.random stream is id-bearing
      for _s in _extra_shape_fs:                           # downstream (scatter layout)
        _s.setColor(_col)

      # ORIENTATION MARKER: a rai axis-triad marker per cube, so rotations of the
      # uniformly-colored cubes are visible. A child frame -> follows the part wherever
      # the state graph moves it; no contact shape, so the planner never sees it. The
      # ViserRecorder renders marker frames as viser axis frames (they have no mesh).
      C.addFrame(f"{name}_marker", name).setShape(ry.ST.marker, [cs * 1.0])

      base_q = scatter_q if scatter_q is not None else np.array([1.0, 0.0, 0.0, 0.0])
      if grid is not None:
          pass                                             # tight grid: keep the nominal grid pose
      elif random_init:
          # FULL randomization (seeded, hence reproducible): uniform x/y in the central
          # square [-random_init, +random_init]^2 between the robots + a random yaw,
          # rejection-sampled collision-free against everything already placed (parts,
          # table, robot bases/arms). `min_sep` additionally enforces a pairwise x/y
          # SEPARATION from every earlier part (e.g. 0.2 so graspability/geometry of
          # different parts provably cannot interact -- the POR independence assumption).
          # Falls back to the nominal line spot if exhausted.
          placed = [np.asarray(C.getFrame(q).getPosition()[:2])
                    for q in assembly.objects[:i]]
          # Two-stage sampling so a crowded square never DETERMINISTICALLY parks the last
          # parts on the nominal line (min_sep 0.4 fits only ~5 parts in a 1x1 square, so the
          # module -- always placed last -- used to land on the fixed line every seed). Stage
          # 1: honour min_sep. Stage 2 (only if stage 1 exhausts): drop min_sep, keep
          # collision-free + random. The fixed line is the last resort, now rarely reached.
          def _try(enforce_sep):
              for _ in range(max_tries):
                  xy = np.random.uniform(-float(random_init), float(random_init), size=2)
                  if enforce_sep and min_sep is not None and any(
                      np.linalg.norm(xy - p) < float(min_sep) for p in placed):
                      continue
                  yaw = np.random.uniform(-np.pi, np.pi)
                  obj.setRelativePosition([xy[0], xy[1], pos[2]])
                  obj.setRelativeQuaternion(list(_quat_mul(
                      np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), base_q)))
                  if C.getCollisionFree():
                      return True
              return False
          if not (_try(True) or _try(False)):
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


def check_initial_scene(C, assembly, robots=None, robot_ik=None, verbose=True):
  """Validate the INITIAL scene produced by add_disassembled_obj -- BEFORE committing a solve.
  Two checks, both about the START state only (the assembly/topology is not exercised):

    (1) COLLISION-FREE: no part penetrates another part, the table, or a robot.
    (2) GRASPABLE: every part admits at least one cheap-feasible PICK by at least one robot
        (i.e. some (robot, grasp) whose gripper reaches the part without collision). With
        `robot_ik` given, the pick is additionally required to solve real-arm IK -- true
        reachability, not just the floating-gripper proxy.

  A part that no robot can pick (e.g. a cube buried in a tight --grid-init block) makes the
  instance unsolvable no matter the planner, so this catches a broken layout up front. Returns
  (ok, report) with report = {collision_free, graspable: {part: [robots]}, ungraspable: [...]}.
  """
  from solver import _search_context, feasible                     # local: solver imports us
  if robots is None:
    robots = get_robots(C)
  checker, grasp_table = _search_context(C, assembly, robots)
  s0 = get_initial_state(C, assembly)

  collision_free = bool(C.getCollisionFree())
  graspable = {p: [] for p in assembly.objects}
  winners = {p: [] for p in assembly.objects}
  for a, ns in picks(s0, assembly, robots, grasp_table=grasp_table):
    part, robot = a[2], a[1]
    if robot in graspable[part]:                                    # this robot already qualifies
      continue
    # feasible = cheap floating-EE collision, plus real-arm IK reachability when robot_ik given.
    if feasible(a, s0, ns, assembly, checker, robot_ik=robot_ik):
      graspable[part].append(robot)
      winners[part].append((robot, a[3]))

  ungraspable = [p for p in assembly.objects if not graspable[p]]
  ok = collision_free and not ungraspable
  if verbose:
    tag = "real-IK" if robot_ik is not None else "cheap"
    print(f"[scene-check] collision-free: {collision_free} | graspable ({tag}) by >=1 robot: "
          f"{len(assembly.objects) - len(ungraspable)}/{len(assembly.objects)} parts")
    print("[scene-check] first reachable grasp per robot: "
          + ", ".join(f"{p}={winners[p]}" for p in assembly.objects))
    if ungraspable:
      print(f"[scene-check] UNGRASPABLE parts (no robot can pick them here): {ungraspable}")
    print(f"[scene-check] => scene {'OK' if ok else 'NOT USABLE'}")
  return ok, {"collision_free": collision_free, "graspable": graspable,
              "grasp_winners": winners, "ungraspable": ungraspable}


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


class FixtureRackAssembly(Assembly):
  """Scalable modular fixture rack: a long BASE RAIL carries repeated UPRIGHT POSTS, and
  every post carries one laterally inserted CANTILEVER BRACKET.

  This is a synthetic benchmark rather than a replica of one product, but its structure is
  deliberately closer to an assembly fixture, tool rack, or small shelving rail than the
  cube families:

  * parts are heterogeneous cuboids with realistic aspect ratios;
  * posts are lowered top-down into the base;
  * brackets are pressed horizontally onto the upper side of their own post;
  * stations are independent branches, while each bracket depends on its post; and
  * bracket sides alternate, so larger instances do not grow one unbalanced comb.

  The horizontal bracket mate is the important wrench discriminator. With the default
  frictionless table model, a rail merely resting on the table cannot react that lateral
  insertion force; the forward planner must arrange a target holder or brace. The family
  therefore scales both the precedence choices and the repeated support decisions without
  introducing artificial cube-face symmetries.

  Naming is station-major: obj_0 is the rail, then (obj_1 post, obj_2 bracket),
  (obj_3 post, obj_4 bracket), ... . Truncation at any --parts value is valid; an odd final
  part is simply a station whose bracket is not present. Up to six stations / 13 parts are
  exposed through the CLI.
  """

  def __init__(self, n_parts=7, max_stations=6, station_pitch=0.075,
               rail_depth=0.050, rail_height=0.025, post_width=0.025,
               post_height=0.100, bracket_length=0.070, bracket_height=0.025,
               corner_radius=0.002, mate_clearance=0.002):
    self.max_stations = int(max_stations)
    self.n_parts = min(max(1, int(n_parts)), 1 + 2 * self.max_stations)
    self.station_pitch = float(station_pitch)
    self.rail_depth = float(rail_depth)
    self.rail_height = float(rail_height)
    self.post_width = float(post_width)
    self.post_height = float(post_height)
    self.bracket_length = float(bracket_length)
    self.bracket_height = float(bracket_height)
    # Keep nominally assembled parts outside one another's collision skin.  The fixture was
    # originally authored with exactly coincident faces; unlike a socket, there is no free
    # space around those faces, so tiny IK/interpolation residuals made an otherwise valid
    # assembled group fail the motion planner's 0.1 mm penetration contract.  The symbolic
    # mate still makes the parts one rigid group after the switch; this is only a small,
    # visible seam in the benchmark geometry (the same role as fit clearance in peg scenes).
    self.mate_clearance = float(mate_clearance)
    assert self.mate_clearance >= 0.0
    self.cube_size = max(self.rail_depth, self.post_height)  # scatter/marker scale only
    self.cube_radius = float(corner_radius)
    super().__init__()

  def define(self):
    r = self.cube_radius
    n_stations = (self.n_parts - 1 + 1) // 2
    rail_length = max(0.12, self.station_pitch * max(0, n_stations - 1) + 0.08)
    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"
    # The rail grows to 0.38m while ordinary parts are only 25--70mm wide. Give it a separate
    # initial lane so fixture-rack-10 at 0.2m pitch does not intersect obj_1 before search.
    self.scatter_offsets["obj_0"] = np.array([0.0, -0.15, 0.0])
    self.part_shapes["obj_0"] = (
        "box", [rail_length, self.rail_depth, self.rail_height, r])
    # The fixture is a small wood/polymer rack, not a solid steel tooling block.  The old
    # values implied roughly 3.2 t/m^3 for the rail and made a small five-part assembly weigh
    # 1.29 kg.  That can pass the quasi-static LP only by sharing load perfectly across
    # co-grasps, while MuJoCo's individual 0.4 Nm breakaway welds necessarily see transiently
    # unequal load and fail.  The calibrated values correspond to a consistent ~500--700
    # kg/m^3 wood/polymer material density.
    self.masses["obj_0"] = 0.084375 + 0.0225 * n_stations

    x0 = -0.5 * self.station_pitch * max(0, n_stations - 1)
    for station in range(n_stations):
      post_i = 1 + 2 * station
      bracket_i = post_i + 1
      post = f"obj_{post_i}"
      x = x0 + station * self.station_pitch
      if post_i < self.n_parts:
        self.part_shapes[post] = (
            "box", [self.post_width, self.post_width, self.post_height, r])
        self.masses[post] = 0.03375
        self.assembly_relative_poses[post] = [
            "obj_0", np.array([x, 0.0,
                               0.5 * (self.rail_height + self.post_height)
                               + self.mate_clearance])]
        self.mate_axes[post] = np.array([0.0, 0.0, -1.0])

      if bracket_i < self.n_parts:
        bracket = f"obj_{bracket_i}"
        side = 1.0 if station % 2 == 0 else -1.0
        self.part_shapes[bracket] = (
            "box", [self.post_width, self.bracket_length, self.bracket_height, r])
        self.masses[bracket] = 0.0225
        # Approach the post side and sit in its upper quarter.  The small final seam is the
        # collision clearance above; the mate axis still points back toward the post, so this
        # remains a true lateral press rather than a diagonal bookkeeping mate.
        self.assembly_relative_poses[bracket] = [
            post,
            np.array([0.0, side * (0.5 * (self.post_width + self.bracket_length)
                                  + self.mate_clearance),
                      0.25 * self.post_height])]
        self.mate_axes[bracket] = np.array([0.0, -side, 0.0])
        self.assembly_dependencies[bracket] = [post]


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
      self.assembly_relative_poses[f"obj_{i}"] = ["obj_0", np.array([x * 1.1, y * 1.1, -0.3 * w])]
      self.mate_axes[f"obj_{i}"] = np.array([0.0, -np.sign(y), 0.0])  # axial, onto the axle
      self.no_hold.add(f"obj_{i}")                          # a wheel would rotate -> can't brace
    self.assembly_dependencies = {}


class MeshDemoAssembly(Assembly):
  """The MESH smoke assembly: cube base + a real triangle-mesh part (a chamfered cube,
  part_meshes/chamfered_box.stl) lowered onto it + a cube cap on top. Exists to exercise
  the ("mesh", {...}) shape kind end to end: rai convex-hull collision, sampled grasps
  (--grasp-sampler REQUIRED for mesh parts), hull support points in the wrench LP, and the
  MuJoCo mesh-asset geom. All mates are top-down stacks (default axes from the offsets)."""

  def __init__(self, n_parts=3, width=0.05, cube_radius=0.005):
    self.n_parts = min(int(n_parts), 3)
    self.width = width
    self.cube_size = width
    self.cube_radius = cube_radius
    super().__init__()

  def define(self):
    w, r = self.width, self.cube_radius
    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"
    self.part_shapes["obj_0"] = ("box", [w, w, w, r])
    if self.n_parts > 1:
      self.part_shapes["obj_1"] = ("mesh", {"file": "part_meshes/chamfered_box.stl"})
    if self.n_parts > 2:
      self.part_shapes["obj_2"] = ("box", [w, w, w, r])
    self.assembly_relative_poses = {
        f"obj_{i}": [f"obj_{i - 1}", np.array([0.0, 0.0, w])]
        for i in range(1, self.n_parts)}
    self.assembly_dependencies = {}


class CompoundDemoAssembly(Assembly):
  """The COMPOUND smoke assembly: cube base + a compound part (a plinth box carrying a
  cylinder boss -- ONE rigid part made of TWO primitives) + a cube cap resting on the boss.
  Exists to exercise the ("compound", [...]) shape kind end to end: multiple __shape{i}
  children in every rai scene, sampled grasps over the union surface (--grasp-sampler
  REQUIRED), per-child support points and rest height, and the multi-geom MuJoCo body.
  The compound is authored symmetric about the part origin (box spans z in [-0.03, 0],
  boss in [0, 0.03]), so the symmetric half-extent cover is exact."""

  def __init__(self, n_parts=3, width=0.05, cube_radius=0.005):
    self.n_parts = min(int(n_parts), 3)
    self.width = width
    self.cube_size = width
    self.cube_radius = cube_radius
    super().__init__()

  def define(self):
    w, r = self.width, self.cube_radius
    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"
    self.part_shapes["obj_0"] = ("box", [w, w, w, r])
    if self.n_parts > 1:
      self.part_shapes["obj_1"] = ("compound", [
          {"shape": ("box", [w, w, 0.03, r]), "pos": [0.0, 0.0, -0.015]},
          # boss cap 3.6cm: a centered vacuum cup (~2cm) seals with margin, and the box-top
          # annulus is too narrow for the sampler to park a cup on (it would overhang the
          # boss wall, which the tight vacuum band in the DB filter rejects).
          {"shape": ("cylinder", [0.03, 0.018]), "pos": [0.0, 0.0, 0.015]},
      ])
    if self.n_parts > 2:
      self.part_shapes["obj_2"] = ("box", [w, w, w, r])
    self.assembly_relative_poses = {
        f"obj_{i}": [f"obj_{i - 1}", np.array([0.0, 0.0, 0.055])]
        for i in range(1, self.n_parts)}
    self.assembly_dependencies = {}


class PegHoleAssembly(Assembly):
  """A two-part concavity smoke test: a cylindrical peg inserted into a blind socket.

  The socket is ONE rigid part represented as a ``compound`` of five convex boxes: a floor
  and four non-overlapping walls around a square pocket.  Keeping those convex children
  separate is essential -- their union has a real cavity which the peg can enter, while the
  rest of the pipeline can continue using convex collision geometry per child.  The final
  peg pose leaves 0.5mm axial seating clearance above the pocket floor and 4mm radial
  clearance to each wall, so strict collision checks test the cavity rather than numerical
  contact at its boundary.

  This intentionally stays tiny: the socket begins upright on the table, the peg begins
  upright beside it, and the only mate is a top-down insertion along -z.  Use
  ``--grasp-sampler`` because the socket is compound; ``--stable-rest-poses`` exercises the
  geometry-derived placement table as well.
  """

  def __init__(self, n_parts=2, outer=0.08, hole=0.032, depth=0.04,
               floor=0.01, peg_radius=0.012, peg_length=0.055, chamfer=0.0,
               hole_chamfer=0.0):
    assert 1 <= int(n_parts) <= 2, "peg-hole is a socket and one peg"
    self.n_parts = int(n_parts)
    self.outer = float(outer)
    self.hole = float(hole)
    self.depth = float(depth)
    self.floor = float(floor)
    self.peg_radius = float(peg_radius)
    self.peg_length = float(peg_length)
    self.chamfer = float(chamfer)
    assert self.chamfer >= 0.0
    self.hole_chamfer = float(hole_chamfer)
    assert self.hole_chamfer >= 0.0
    self.cube_size = self.outer                         # scatter/marker sizing only
    self.cube_radius = 0.0
    super().__init__()

  def define(self):
    outer, hole = self.outer, self.hole
    depth, floor = self.depth, self.floor
    wall = (outer - hole) / 2.0
    assert wall > 0.0 and self.peg_radius < hole / 2.0
    wall_at = hole / 2.0 + wall / 2.0
    floor_z = -(depth + floor) / 2.0                    # total bounds: [-.025, +.025]
    wall_z = (floor - depth) / 2.0                      # walls start on the floor's top
    sharp = 0.0                                         # preserve the authored hole clearance

    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"
    walls = [
        {"shape": ("box", [wall, outer, depth, sharp]), "pos": [wall_at, 0.0, wall_z]},
        {"shape": ("box", [wall, outer, depth, sharp]), "pos": [-wall_at, 0.0, wall_z]},
        {"shape": ("box", [hole, wall, depth, sharp]), "pos": [0.0, wall_at, wall_z]},
        {"shape": ("box", [hole, wall, depth, sharp]), "pos": [0.0, -wall_at, wall_z]},
    ]
    if self.hole_chamfer > 0.0:
      # SOCKET-ENTRY CHAMFER: shorten every wall by c on the rim side and add a 45-degree
      # trapezoid funnel prism at each rim edge -- the mouth widens from 2h to 2(h+c) and
      # converges to the original pocket at depth c (the socket opens UP here).
      c = self.hole_chamfer
      rim_z = wall_z + depth / 2.0
      for w in walls:
        sx, sy, _, r = w["shape"][1]
        w["shape"] = ("box", [sx, sy, depth - c, r])
        w["pos"][2] -= c / 2.0
      walls += [{"shape": sub, "pos": list(pos)}
                for sub, pos in _hole_chamfer_children(hole / 2.0, wall, depth, rim_z, c,
                                                       +1.0, outer, hole)]
    self.part_shapes["obj_0"] = ("compound",
                                 [{"shape": ("box", [outer, outer, floor, sharp]),
                                   "pos": [0.0, 0.0, floor_z]}] + walls)
    if self.n_parts > 1:
      if self.chamfer > 0.0:
        # CHAMFERED peg (--assembly peg-hole-chamfer): the -z end enters the socket and
        # carries a 45-degree frustum lead-in, so a slightly misaligned descent slides
        # instead of catching the rim. The peg becomes a 2-child compound (shortened
        # cylinder body + inline convex-mesh tip); the total length is unchanged.
        body, tip = _chamfer_tip(("cylinder", [self.peg_length, self.peg_radius]),
                                 self.chamfer, -1.0)
        self.part_shapes["obj_1"] = ("compound", [
            {"shape": body[0], "pos": list(body[1])},
            {"shape": tip[0], "pos": list(tip[1])}])
      else:
        self.part_shapes["obj_1"] = ("cylinder", [self.peg_length, self.peg_radius])
      # Pocket floor top = -depth/2.  Put the peg bottom 0.5mm above it: enough to avoid a
      # numerical contact being counted as penetration while still representing a seated peg.
      seated_z = -depth / 2.0 + self.peg_length / 2.0 + 0.0005
      self.assembly_relative_poses["obj_1"] = ["obj_0", np.array([0.0, 0.0, seated_z])]
      self.mate_axes["obj_1"] = np.array([0.0, 0.0, -1.0])
    self.assembly_dependencies = {}
    self.masses = {"obj_0": 0.35, "obj_1": 0.10}


class PegHoleScrewAssembly(PegHoleAssembly):
  """Two-part screw-skill experiment derived from :class:`PegHoleAssembly`.

  Geometry deliberately remains a generous real concavity so the first controller test is
  about the helical motion rather than thread-mesh contact.  The peg carries a small
  asymmetric head above the socket; its rotation is therefore visible, while its cylindrical
  shaft can turn freely in the square pocket.  This is a *virtual-thread* experiment: the
  authored :class:`ScrewSkill` imposes pitch/rotation and the wrench model prices axial force
  plus torque.  It does not claim that smooth MuJoCo contact generates a physical thread lead.
  """

  def __init__(self, n_parts=2, screw_angle=np.pi, screw_depth=0.020,
               screw_torque=0.25, screw_speed=0.75, screw_samples=16, **kwargs):
    self._screw_spec = ScrewSkill(angle=screw_angle, depth=screw_depth,
                                  torque=screw_torque, angular_speed=screw_speed,
                                  samples=screw_samples)
    super().__init__(n_parts=n_parts, chamfer=0.004, hole_chamfer=0.006, **kwargs)

  def define(self):
    super().define()
    if self.n_parts < 2:
      return
    # A fixture-style side handle gives a second arm a realistic, REMOTE place to oppose
    # screw torque.  Keeping the clamp hand away from the screw axis is important: a small
    # lug directly on the socket makes the perfectly valid force closure unusable because
    # the two robot wrists necessarily occupy the same space at the assembled keyframe.
    # The thin rail is part of the same rigid target and ends in a vertical grip block.
    socket_children = [
        {"shape": sub, "pos": np.asarray(pos, dtype=float).tolist()}
        for sub, pos in _compound_children(self.part_shapes["obj_0"])]
    socket_children.extend([
        {"shape": ("box", [0.120, 0.014, 0.014, 0.001]),
         "pos": [self.outer / 2.0 + 0.060, 0.0, 0.012]},
        {"shape": ("box", [0.025, 0.035, 0.050, 0.001]),
         "pos": [self.outer / 2.0 + 0.120, 0.0, 0.030]},
    ])
    self.part_shapes["obj_0"] = ("compound", socket_children)
    # A raised offset head makes a 180-degree turn observable and gives the fingers enough
    # axial room to remain clear of the socket rim for the WHOLE helix.  The narrow neck is
    # deliberately part of the screw, not a kinematic offset hidden in gripper calibration.
    shaft = self.part_shapes["obj_1"]
    head_z = self.peg_length / 2.0 + 0.030
    # Compound children are deliberately flat throughout the geometry backend.  The
    # chamfered peg is already a body+tip compound, so splice those children rather than
    # nesting it under the head compound.
    children = ([{"shape": sub, "pos": np.asarray(pos, dtype=float).tolist()}
                 for sub, pos in _compound_children(shaft)]
                if shaft[0] == "compound" else
                [{"shape": shaft, "pos": [0.0, 0.0, 0.0]}])
    children.extend([
        {"shape": ("cylinder", [0.024, 0.008]),
         "pos": [0.0, 0.0, self.peg_length / 2.0 + 0.012]},
        {"shape": ("box", [0.050, 0.025, 0.012, 0.001]),
         "pos": [0.010, 0.0, head_z]},
    ])
    self.part_shapes["obj_1"] = ("compound", children)
    # The fixture handle enlarges obj_0 beyond the nominal 8cm scatter cell.  Move the loose
    # screw one extra cell outward so the INITIAL pick scene is not artificially occluded by
    # that handle; this is scene geometry, not a search-order or reachability heuristic.
    self.scatter_offsets["obj_1"] = np.array([0.10, 0.0, 0.0])
    self.mate_skills["obj_1"] = self._screw_spec


class PegStoolAssembly(Assembly):
  """A three-legged stool whose legs really insert into blind sockets in the seat.

  ``ChairAssembly(..., parts=5)`` is intentionally faithful to the original ``chair.g``
  geometry, where the legs merely meet the underside of the seat.  This variant is the
  slightly richer follow-up to :class:`PegHoleAssembly`: the seat is ONE rigid compound
  part consisting of a plate and three downward-facing square socket collars.  Each round
  leg enters a collar by 24.5mm and stops 0.5mm short of the plate.  Thus every leg mate
  exercises a genuine concavity while the completed object has the familiar stool support
  geometry needed by the wrench checks.

  The compound has 13 non-overlapping convex children (the seat plate plus four walls per
  socket).  Its top is deliberately one plain broad surface: no artificial vacuum pedestal.
  Use ``--grasp-sampler`` for the compound seat.  A fleet with at least one vacuum gripper
  is recommended: the vacuum holds the broad seat while two-finger grippers insert legs.

  ``leg_kind="box"`` (--assembly peg-stool-cuboid) keeps the SAME seat and sockets but gives
  every leg a SQUARE cross-section.  A square peg fits a square hole when side < hole (the
  peg's DIAGONAL clears the hole's diagonal); the stock side = 0.020 in the 0.030 hole leaves
  a generous ~7mm corner gap.

  ``legs=4`` (--assembly peg-stool-4 / peg-stool-cuboid-4) puts FOUR legs under the seat in
  a symmetric rectangle.  The stock stool is deliberately three-legged: a triangle of
  contacts is always coplanar (no wobble) and strictly contains the seat COM, whereas any
  three corners of the four-leg rectangle put the COM on a support DIAGONAL.  All four legs
  restore a strict interior COM with the classic stool look.

  ``peg_clearance`` (the -tight kinds) derives the leg size from the socket hole so the
  NOMINAL clearance is exactly the given value instead of the stock ~5mm: a round peg gets
  radius = hole/2 - c, a square peg gets side = hole - c*sqrt(2) (its CORNER is the binding
  constraint, so this yields corner gap == c). Set it in code to try any clearance; the
  constructor asserts the value is geometrically possible.
  """

  def __init__(self, n_parts=4, seat_x=0.18, seat_y=0.14, seat_thickness=0.018,
               socket_outer=0.04, socket_hole=0.030, socket_depth=0.025,
               leg_radius=0.0095, leg_length=0.11, leg_kind="cylinder",
               leg_side=0.020, legs=3, peg_clearance=None, chamfer=0.0,
               hole_chamfer=0.0):
    assert legs in (3, 4), f"peg-stool has three or four legs, not {legs}"
    assert leg_kind in ("cylinder", "box"), f"peg-stool leg kind {leg_kind!r}"
    self.legs = int(legs)
    assert 1 <= int(n_parts) <= self.legs + 1, \
        f"peg-stool is one seat and up to {self.legs} legs"
    self.n_parts = int(n_parts)
    self.seat_x = float(seat_x)
    self.seat_y = float(seat_y)
    self.seat_thickness = float(seat_thickness)
    self.socket_outer = float(socket_outer)
    self.socket_hole = float(socket_hole)
    self.socket_depth = float(socket_depth)
    self.leg_radius = float(leg_radius)
    self.leg_length = float(leg_length)
    self.leg_kind = leg_kind
    self.leg_side = float(leg_side)
    self.peg_clearance = None if peg_clearance is None else float(peg_clearance)
    self.chamfer = float(chamfer)
    assert self.chamfer >= 0.0
    self.hole_chamfer = float(hole_chamfer)
    assert self.hole_chamfer >= 0.0
    self.cube_size = self.seat_x                       # scatter/marker sizing only
    self.cube_radius = 0.0
    super().__init__()

  def define(self):
    sx, sy, st = self.seat_x, self.seat_y, self.seat_thickness
    outer, hole, depth = self.socket_outer, self.socket_hole, self.socket_depth
    wall = (outer - hole) / 2.0
    assert wall > 0.0
    # Keep the sockets well inside the plate and well separated from one another.
    # THREE legs sit in a triangle that strictly contains the seat COM (unlike any three
    # corners of a symmetric four-leg rectangle, where the COM lies exactly on a support
    # diagonal). FOUR legs sit in the full symmetric rectangle -- the COM is strictly
    # interior again, and four contacts give the classic non-wobbling stool.
    if self.legs == 4:
      socket_centres = [(0.055, 0.035), (-0.055, 0.035), (0.055, -0.035), (-0.055, -0.035)]
    else:
      socket_centres = [(0.055, 0.035), (-0.055, 0.035), (0.0, -0.045)]
    assert all(abs(x) + outer / 2.0 <= sx / 2.0 and
               abs(y) + outer / 2.0 <= sy / 2.0 for x, y in socket_centres)
    wall_at = hole / 2.0 + wall / 2.0
    wall_z = -st / 2.0 - depth / 2.0                # collar touches the plate underside

    children = [
        {"shape": ("box", [sx, sy, st, 0.0]), "pos": [0.0, 0.0, 0.0]},
    ]
    for cx, cy in socket_centres:
      if self.hole_chamfer > 0.0:
        # SOCKET-ENTRY CHAMFER: the collars open DOWN; every wall is shortened by c at the
        # mouth and a 45-degree trapezoid funnel prism widens the mouth from 2h to 2(h+c).
        c = self.hole_chamfer
        rim_z = wall_z - depth / 2.0
        children.extend([
            {"shape": ("box", [wall, outer, depth - c, 0.0]),
             "pos": [cx + wall_at, cy, wall_z + c / 2.0]},
            {"shape": ("box", [wall, outer, depth - c, 0.0]),
             "pos": [cx - wall_at, cy, wall_z + c / 2.0]},
            {"shape": ("box", [hole, wall, depth - c, 0.0]),
             "pos": [cx, cy + wall_at, wall_z + c / 2.0]},
            {"shape": ("box", [hole, wall, depth - c, 0.0]),
             "pos": [cx, cy - wall_at, wall_z + c / 2.0]},
        ])
        for sub, pos in _hole_chamfer_children(hole / 2.0, wall, depth, rim_z, c,
                                               -1.0, outer, hole):
          children.append({"shape": sub, "pos": [cx + pos[0], cy + pos[1], pos[2]]})
      else:
        children.extend([
            {"shape": ("box", [wall, outer, depth, 0.0]),
             "pos": [cx + wall_at, cy, wall_z]},
            {"shape": ("box", [wall, outer, depth, 0.0]),
             "pos": [cx - wall_at, cy, wall_z]},
            {"shape": ("box", [hole, wall, depth, 0.0]),
             "pos": [cx, cy + wall_at, wall_z]},
            {"shape": ("box", [hole, wall, depth, 0.0]),
             "pos": [cx, cy - wall_at, wall_z]},
        ])

    self.objects = [f"obj_{i}" for i in range(self.n_parts)]
    self.base = "obj_0"
    self.part_shapes["obj_0"] = ("compound", children)
    # This is a small wooden seat/support fixture, not another 50mm benchmark cube.  Giving
    # it a realistic base mass also makes the wrench model's table-friction prediction match
    # execution: at 0.45kg a nominally axial 20N insertion could move the whole fixture after
    # a tiny transient side contact, although the quasi-static wrench check correctly saw no
    # lateral insertion load.  Keep the slender legs light; scaling the whole assembly instead
    # overloads their two-finger grasps without addressing the support mismatch.
    self.masses["obj_0"] = 0.75
    # Legs: round pegs (the stock stool) or SQUARE pegs (peg-stool-cuboid).  A square peg
    # fits when its DIAGONAL clears the hole's diagonal: side < hole, and the corner gap is
    # (hole - side)/sqrt(2). A peg_clearance derives the size from the hole for the tight
    # kinds (corner gap == c for a square peg).
    if self.peg_clearance is not None:
      c = self.peg_clearance
      if self.leg_kind == "box":
        assert c * np.sqrt(2.0) < hole, \
            f"square-peg clearance {c} is impossible in socket hole {hole}"
        self.leg_side = hole - c * np.sqrt(2.0)
      else:
        assert c < hole / 2.0, f"round-peg clearance {c} is impossible in hole {hole}"
        self.leg_radius = hole / 2.0 - c
    if self.leg_kind == "box":
      assert self.leg_side < hole, \
          f"square leg side {self.leg_side} does not fit socket hole {hole}"
      leg_spec = ("box", [self.leg_side, self.leg_side, self.leg_length, 0.0])
      # LIE FLAT on the table (long axis -> world x), like the round legs: both end faces
      # stay reachable and the grasp on the end that must enter the socket stays clear.
      # Standing upright would leave only the top 20mm-square cap graspable.
      leg_scatter = np.array([np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0])
      leg_mass = 0.11                    # volume-proportional to the round 19mm peg
    else:
      assert self.leg_radius < hole / 2.0
      leg_spec = ("cylinder", [self.leg_length, self.leg_radius])
      # Round legs scatter horizontally: both end faces stay reachable and a two-finger grasp
      # does not occupy the end that enters the socket. The cylinder's axial roll is invisible,
      # so with sampled grasps the free roll is chosen so one retained suction contact sits on
      # the upper crown (a random barrel sample would let a vacuum push roll the loose leg).
      leg_scatter = _horizontal_cylinder_scatter_with_top_suction(leg_spec)
      leg_mass = 0.08
    if self.chamfer > 0.0:
      # CHAMFERED insertion tip (the -chamfer stool kinds): the leg keeps its length, but
      # its +z end -- the one that enters the downward-facing socket -- tapers by
      # `chamfer` over the same distance (45-degree lead-in). The leg becomes a 2-child
      # compound (full body + inline convex-mesh frustum tip). The socket is unchanged.
      body, tip = _chamfer_tip(leg_spec, self.chamfer, 1.0)
      leg_spec = ("compound", [
          {"shape": body[0], "pos": list(body[1])},
          {"shape": tip[0], "pos": list(tip[1])}])
    leg_top = -st / 2.0 - 0.0005                    # 0.5mm short of the blind end
    leg_z = leg_top - self.leg_length / 2.0
    for i, (cx, cy) in enumerate(socket_centres[:max(0, self.n_parts - 1)], start=1):
      part = f"obj_{i}"
      self.part_shapes[part] = leg_spec
      if leg_scatter is not None:
        self.scatter_orientations[part] = leg_scatter
      self.assembly_relative_poses[part] = ["obj_0", np.array([cx, cy, leg_z])]
      self.assembly_dependencies[part] = ["obj_0"]
      self.mate_axes[part] = np.array([0.0, 0.0, 1.0])
      self.masses[part] = leg_mass


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

  def __init__(self, n_parts=8, cube_radius=0.005, scale=0.5):
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
    # seat/legs SNAPPED to exact axis rotations: chair.g's d(90 90 1 0) axis is (90,1,0) --
    # x with accidental authoring noise (~0.5deg) that breaks flush table contact (rest-pose
    # tolerance is 1e-4). The posts' d(90 1 -0.2 0) recline and the backrest's tilt are
    # DESIGN, kept verbatim (placement_quat rests their shapes flush regardless).
    q_seat = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
    q_leg  = [0.0, 1.0, 0.0, 0.0]
    q_back = [0.703, 0.073, 0.075, 0.703]
    q_post = [0.011, 0.99, -0.093, -0.103]
    leg_scatter  = np.array([np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0])   # exact: axis -> world x
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
      # backrest joins the seat as a MODULE (user-specified DAG: posts mate onto the BACK,
      # then the back+posts subassembly joins the seat as a unit) -- the connect mechanism,
      # like cube-plates: obj_5's mate onto the seat is the JOIN, and it requires BOTH posts
      # present (in the held module), so the module is built detached first.
      self.part_shapes["obj_5"] = ("box", [0.23 * s, 0.05 * s, 0.20 * s, r], q_back)
      self.connect_parts.add("obj_5")
      self.connect_requires["obj_5"] = {"obj_6", "obj_7"}
      self.scatter_orientations["obj_5"] = np.array([0.7772, 0.0, 0.6292, 0.0])  # backrest lies flat
      self.assembly_relative_poses["obj_5"] = ["obj_0", np.array([0.19, 0.0, 0.25]) * s]
      self.mate_axes["obj_5"] = np.array([0.0, 0.0, -1.0])  # sub-assembly lowered onto the seat
      # NO assembly_dependencies for obj_5: a dependency would drag obj_0 into the POSTS'
      # mate closure (dependencies_of feeds _mate_closure) and forbid building the module
      # detached. The join precedence lives ENTIRELY in connect_requires (cube-plates rule).
      post_off = {"obj_6": [-0.009, 0.152, -0.052], "obj_7": [-0.006, -0.148, -0.052]}
      # posts join the backrest SIDEWAYS (the post-to-back offset is y-dominant: the posts
      # flank the back). Their contact patch is COMPUTED: the closest point of the post's
      # SIDE surface to the back plate at the assembled pose (off the central axis, at the
      # overlap height -- 'touches == is close to'), via mate_contact_interface below.
      for p, off in post_off.items():
        self.part_shapes[p] = ("cylinder", [0.33 * s, 0.025 * s], q_post)
        self.scatter_orientations[p] = post_scatter
        self.assembly_relative_poses[p] = ["obj_5", np.array(off) * s]   # post -> backrest
        self.mate_axes[p] = np.array([0.0, -1.0, 0.0]) if p == "obj_6" \
            else np.array([0.0, 1.0, 0.0])               # press toward the back (sideways)
        self.assembly_dependencies[p] = ["obj_5"]
      # for p in post_off:
        # self.mate_interfaces[p] = self.mate_contact_interface(p)
      self.mate_interfaces["obj_6"] = [0, -0.025*s, 0.33*s / 8]
      self.mate_interfaces["obj_7"] = [0, 0.025*s, 0.33*s / 8]

      # the MODULE JOIN (back -> seat) transmits its press through the POSTS' BOTTOM ends
      # standing on the seat -- not through the backrest's own bottom (the back never
      # touches the seat). Interface (in the BACK's frame, parts axis-aligned): midpoint of
      # the two post bottoms = post offset + half_len along the post's downward long axis.
      post_down = _quat_to_R(np.asarray(q_post, dtype=float))[:, 2]      # long axis
      if post_down[2] > 0:                                               # orient downward
        post_down = -post_down
      bottoms = [np.array(off) * s + post_down * (0.33 * s / 2.0)
                 for off in post_off.values()]
      self.mate_interfaces["obj_5"] = np.mean(bottoms, axis=0)


# ============================================================================================
# TODO -- NON-TOP-DOWN ASSEMBLY FAMILY (shared-shaft / coaxial capture).
# Motivation: assemblies that CANNOT be built as a top-down table stack -- at some step the
# mate axis is not vertical-down onto a table-resting piece, so the growing subassembly must
# be HELD in the air (and often braced by a second arm). The car already proves horizontal
# (+-y) mates work [[CarAssembly]]; these push the same idea in non-car directions for
# experiment diversity. All three want the co-grasp / two-arm brace path (see cograsp-gap).
#
# TODO(1) CarThruAxleAssembly -- the car, but the two wheels of a side share ONE continuous
#   through-axle instead of each pressing onto its own stub. Minimal delta from CarAssembly:
#   add a rod part on the shared axis; both wheels mate onto the SAME rod (same +-y axis), so
#   the wheel/rod stack is captured coaxially and never rests to receive a part from above.
#   Tests: coaxial capture + forced inner->outer ordering on one axis. Closest to today's car.
#
# TODO(2) BarbellAssembly -- a horizontal BAR; weight plates (thin cylinders) slide on from
#   BOTH ends, a COLLAR caps each side to trap them. Symmetric two-ended loading; plates are
#   heavy. This is the natural home for the OVERLOAD / wrench-magnitude story (stack enough
#   plates and the holding grasp or a joint is overloaded -- ties to finite-mate-strength).
#   Parts: bar (long cylinder, base) + N plates (mate +-x onto the bar) + 2 collars (cap).
#   Tests: symmetric horizontal skewer + load stacking. Simplest to author, good control.
#
# TODO(3) SpoolShaftAssembly -- a SHAFT carrying a spool (+ spacers) mounted BETWEEN two
#   bearing BLOCKS. The spool is a MOVABLE part (rides/rotates on the shaft) -- what the user
#   liked: it is not fixed like a wheld weld. The shaft cannot be added last from the top --
#   the second block TRAPS it, so the shaft goes in axially and the blocks close around it.
#   Parts: 2 blocks (boxes, table-resting supports) + shaft (cylinder) threading the spool
#   (cylinder, no_hold: it rotates) -> shaft mates axially into block bores. Tests: shaft
#   captured BETWEEN supports (capture by the supports, not by end collars) + a free-riding
#   part. May want staged-mates once the second block closes (see skills-and-staged-mates).
# ============================================================================================


_ASSEMBLY_KINDS = {"cross": (CubeCrossAssembly, 7), "cube": (CubeGridAssembly, 8),
                   "arch": (ArchAssembly, 7), "arch-seq": (ArchSequentialAssembly, 7),
                   "cube-plates": (CubePlatesAssembly, 8),
                   "cube-plates-2x2x3": (lambda n_parts: CubePlatesAssembly(2, 2, 3, n_parts=n_parts), 12),
                   "cuboid-tower": (CuboidTowerAssembly, 8),
                   "fixture-rack": (FixtureRackAssembly, 13),
                   "car": (CarAssembly, 5),
                   # mini car: whole car scaled to width=0.03 -> wheels are discs of diameter
                   # 0.8w*2 = 0.048, comfortably inside the two-finger's ~0.078 inner gap, so the
                   # fingers can take a wheel ACROSS THE DIAMETER (radial side grasp) -- the
                   # grasp-DB's curved-surface entries, which the two-finger keeps (vacuum drops).
                   "car-mini": (lambda n_parts: CarAssembly(n_parts=n_parts, width=0.03), 5),
                   "slot-pin": (SlotPinAssembly, 3),
                   "peg-hole": (PegHoleAssembly, 2),
                   "peg-hole-screw": (PegHoleScrewAssembly, 2),
                   "peg-stool": (PegStoolAssembly, 4),
                   "peg-stool-4": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, legs=4), 5),
                   "peg-stool-cuboid": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, leg_kind="box"), 4),
                   "peg-stool-cuboid-4": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, leg_kind="box", legs=4), 5),
                   # TIGHT-PEG variants: the leg size derives from the hole so the nominal
                   # clearance is 1mm (stock: ~5mm). Square pegs clear by corner gap == 1mm.
                   "peg-stool-tight": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, peg_clearance=0.001), 4),
                   "peg-stool-4-tight": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, legs=4, peg_clearance=0.001), 5),
                   "peg-stool-cuboid-tight": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, leg_kind="box", peg_clearance=0.004), 4),
                   "peg-stool-cuboid-4-tight": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, leg_kind="box", legs=4, peg_clearance=0.004), 5),
                   # CHAMFERED kinds: every peg/leg's INSERTION end carries a 4mm 45-degree
                   # frustum lead-in (an inline convex mesh) AND every socket mouth carries a
                   # 45-degree ENTRY chamfer of its own (the rim widens into a funnel): peg
                   # 6mm (wall 24mm), stool 4mm (wall 5mm). A slightly misaligned insertion
                   # slides down both facets instead of catching the rim -- the execution-
                   # friendly variants of the same assemblies (same clearances, mate axes).
                   "peg-hole-chamfer": (lambda n_parts: PegHoleAssembly(
                       n_parts=n_parts, chamfer=0.004, hole_chamfer=0.006), 2),
                   "peg-stool-chamfer": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, chamfer=0.004, hole_chamfer=0.004), 4),
                   "peg-stool-4-chamfer": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, legs=4, chamfer=0.004, hole_chamfer=0.004), 5),
                   "peg-stool-cuboid-chamfer": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, leg_kind="box", chamfer=0.004,
                       hole_chamfer=0.004), 4),
                   "peg-stool-cuboid-4-chamfer": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, leg_kind="box", legs=4, chamfer=0.004,
                       hole_chamfer=0.004), 5),
                   "peg-stool-tight-chamfer": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, peg_clearance=0.001, chamfer=0.004,
                       hole_chamfer=0.004), 4),
                   "peg-stool-4-tight-chamfer": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, legs=4, peg_clearance=0.001, chamfer=0.004,
                       hole_chamfer=0.004), 5),
                   "peg-stool-cuboid-tight-chamfer": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, leg_kind="box", peg_clearance=0.004,
                       chamfer=0.004, hole_chamfer=0.004), 4),
                   "peg-stool-cuboid-4-tight-chamfer": (lambda n_parts: PegStoolAssembly(
                       n_parts=n_parts, leg_kind="box", legs=4, peg_clearance=0.004,
                       chamfer=0.004, hole_chamfer=0.004), 5),
                   "mesh-demo": (MeshDemoAssembly, 3),
                   "compound-demo": (CompoundDemoAssembly, 3),
                   "chair": (ChairAssembly, 8)}


def assembly_max_parts(kind):
  """The largest --parts value the assembly builds; make_assembly clamps to it silently."""
  return _ASSEMBLY_KINDS[kind][1]


def make_assembly(kind, n_parts):
  """Assembly factory for the CLI: cross (cube cross), cube (2x2x2 grid), arch (bridge --
  known-flawed star, see class doc), arch-seq (arch shape built as a sequential chain),
  cube-plates (2x2x2 as two stacked 2x2 plates), cube-plates-2x2x3 (three stacked 2x2
  plates, 12 cubes), cuboid-tower (laid-down 2:1:1 cuboids stacked along z -- the first
  non-cube assembly), fixture-rack (a scalable rail with post/bracket station branches),
  peg-stool / peg-stool-cuboid (three-legged stool with round or square legs into real blind
  sockets). The plates family generalizes to any WxDxH via CubePlatesAssembly."""
  cls, max_n = _ASSEMBLY_KINDS[kind]
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
  from env import maybe_tighten_fingers          # same proxy as the real scene, or they disagree
  maybe_tighten_fingers(C, prefix)
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


def _quat_conj(q):
  q = np.asarray(q, dtype=float)
  return np.array([q[0], -q[1], -q[2], -q[3]])


_STABLE_REST_POSES = False   # --stable-rest-poses: see placement_table
_PLACEMENT_YAWS = 1          # --placement-yaws N: uniform world-z fibre per stable rest
_PLACEMENT_YAW_PREFIX = None # None=all configured yaw bands; integer=progressive prefix


def set_stable_rest_poses(on=True):
  """Fill the placement table from the shape's TRUE stable poses instead of the six
  axis-aligned face-down rotations. Call BEFORE any scene or table is built (like
  set_plain_box / set_sampled_grasps): placement ids index the table, so flipping this
  mid-run rebinds every saved placement."""
  global _STABLE_REST_POSES
  _STABLE_REST_POSES = bool(on)


def set_placement_yaws(n=1):
  """Discretise the physically free world-z yaw of every stable table placement.

  ``n=1`` is the historical placement table, byte-for-byte.  For ``n>1`` the table keeps
  every old placement id and appends the rotations ``Rz(2*pi*k/n) R_rest``.  The bound is a
  guard against accidentally multiplying every place edge by an unbounded CLI value; it is
  not a geometric assumption.
  """
  n = int(n)
  if not 1 <= n <= 24:
    raise ValueError("placement yaw count must be in [1, 24]")
  global _PLACEMENT_YAWS
  _PLACEMENT_YAWS = n


def set_placement_yaw_prefix(n=None):
  """Expose only the first ``n`` coarse-to-fine yaw bands in concrete place actions.

  This is search scheduling, not a different placement table: every quaternion/id is built
  once from ``_PLACEMENT_YAWS`` and remains resolvable in states and replay. ``None`` exposes
  the complete configured lattice. The milestone driver uses finite monotone prefixes, then
  restores ``None`` before witness/replay passes.
  """
  if n is not None:
    n = int(n)
    if not 1 <= n <= _PLACEMENT_YAWS:
      raise ValueError(f"active placement-yaw prefix must be in [1, {_PLACEMENT_YAWS}]")
  global _PLACEMENT_YAW_PREFIX
  _PLACEMENT_YAW_PREFIX = n


def placement_yaw_levels(n=None):
  """Exponential finite widening levels ending in the complete yaw lattice."""
  n = _PLACEMENT_YAWS if n is None else int(n)
  levels, k = [1], 2
  while k < n:
    levels.append(k)
    k *= 2
  if n > 1:
    levels.append(n)
  return tuple(dict.fromkeys(levels))


def placement_yaw_order(n=None):
  """Indices of a uniform yaw lattice, in deterministic coarse-to-fine order.

  Farthest-point insertion gives useful prefixes (0, pi, then quarter turns, then their
  gaps) while the complete set remains exactly the ordinary uniform ``n``-bin lattice.
  """
  n = _PLACEMENT_YAWS if n is None else int(n)
  if n <= 1:
    return (0,)
  selected = [0]
  left = set(range(1, n))
  while left:
    def separation(k):
      return min(min((k - j) % n, (j - k) % n) for j in selected)
    k = max(left, key=lambda x: (separation(x), -x))
    selected.append(k)
    left.remove(k)
  return tuple(selected)


def _placement_id(quat):
  """The GLOBAL placement id of a rotation: the same rotation always gets the same id, in
  every group and in every run.

  This is what makes a placement id survive its group CHANGING, and the reason is physical
  rather than bookkeeping: when a part is mated onto a resting group, the group's ORIENTATION
  does not change -- only the set of orientations it could rest in does. An id that indexed a
  per-group list would silently mean a different rotation after the group grew (measured on
  car-mini: id 8, assigned when the group was one shape, resolved against a two-part group
  that had eight ids). Deriving the id FROM the rotation removes the question.

  Ids 0..15 stay free for the legacy face table and for the reserved 'as anchored' id, so
  the two modes never collide.
  """
  import zlib
  q = np.asarray(quat, dtype=float)
  if q[0] < 0.0:
    q = -q                                                 # q and -q are the same rotation
  q = np.round(q, 6) + 0.0                                 # kill -0.0
  if abs(q[0] - 1.0) < 1e-9 and float(np.abs(q[1:]).max()) < 1e-9:
    return _IDENTITY_FACE                                  # the reserved 'as anchored' id
  return int(zlib.crc32(q.tobytes()) & 0x00FFFFFF) + 16


def placement_table(assembly, group, ref):
  """{placement id -> rotation of the reference PART FRAME}, for one subassembly.

  THE ONE PLACE that decides what a placement id means. Two ways to fill it, and the point
  of the partial migration is that they are the same kind of object -- a face direction IS
  just a placement pose:

    legacy (default)         the six axis-aligned face-down rotations, with the reference
                             shape's own rotation composed out. Ids 0..5 exactly as before,
                             so saved plans and witnesses keep their meaning.
    --stable-rest-poses      the TRUE stable rest orientations of the shape (one part) or of
                             the ASSEMBLED GROUP (several parts), computed from the convex
                             hull and the mass-weighted centre of mass. Ids come from
                             `_placement_id`, so they are global and survive group growth.

  ID 5 IS RESERVED IN BOTH MODES for the "as anchored / as scattered" placement, which keeps
  the part's own anchor orientation instead of imposing a rotation (`_group_rest_pose` and
  `_group_rest_lift` special-case it, and it is the default of several helpers).

  Every table is also recorded in a per-assembly REGISTRY, so a rotation stays resolvable
  from any group -- see placement_quat.
  """
  cache = getattr(assembly, "_placement_table_cache", None)
  if cache is None:
    cache = assembly._placement_table_cache = {}
  reg = getattr(assembly, "_placement_registry", None)
  if reg is None:
    reg = assembly._placement_registry = {}
  key = (_subasm_key(group), ref, _STABLE_REST_POSES, _PLACEMENT_YAWS)
  hit = cache.get(key)
  if hit is not None:
    return hit
  ident = np.array([1.0, 0.0, 0.0, 0.0])
  if not _STABLE_REST_POSES:
    table = {f: _legacy_placement_quat(assembly, ref, f)
             for f in range(len(_FACE_DOWN_QUATS))}
  else:
    grp = tuple(group)
    poses = (group_stable_rest_poses(assembly, grp, ref) if len(grp) > 1
             else stable_rest_poses(assembly.part_shape(ref)))
    table = {}
    for q, _prob in _thin_rest_poses(poses, grp):
      table[_placement_id(q)] = np.asarray(q, dtype=float)
    table[_IDENTITY_FACE] = ident
  # Every stable rest has an S1 fibre: rotating the whole resting group about world gravity
  # preserves its support polygon and gravity/wrench geometry, but changes fixed-arm reach.
  # Keep the historical table as a stable prefix.  Variant ids are derived from the complete
  # quaternion, so the state identity and saved plans cannot alias two different yaws.
  yaw_base = getattr(assembly, "_placement_yaw_base", None)
  if yaw_base is None:
    yaw_base = assembly._placement_yaw_base = {}
  yaw_ranks = getattr(assembly, "_placement_yaw_ranks", None)
  if yaw_ranks is None:
    yaw_ranks = assembly._placement_yaw_ranks = {}
  # Rank is TABLE-LOCAL. Quaternion ids are intentionally global, and the same rotation can
  # be a historical base of one subassembly but a yaw copy of another. A global pid->rank map
  # would then expose that second table's copy at prefix 1 and violate the N=1 compatibility
  # contract. The yaw-orbit class below may remain global because equivalent quaternions really
  # are the same abstract rest class.
  yaw_rank = {face: 0 for face in table}
  for face in table:
    yaw_base.setdefault(face, face)
  if _PLACEMENT_YAWS > 1:
    base_items = list(table.items())
    anchor = np.asarray(assembly.scatter_orientations.get(ref, ident), dtype=float)
    anchor_tilt = _quat_mul(_quat_conj(_anchor_yaw_quat(anchor)), anchor)
    for rank, k in enumerate(placement_yaw_order(_PLACEMENT_YAWS)[1:], start=1):
      theta = 2.0 * np.pi * float(k) / float(_PLACEMENT_YAWS)
      qyaw = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])
      for face, qrest in base_items:
        # Id 5 means "as anchored", including a wheel's scatter tilt.  Factor out only its
        # existing world yaw; _group_rest_pose composes the anchor yaw back in later.
        qbase = anchor_tilt if face == _IDENTITY_FACE else np.asarray(qrest, dtype=float)
        q = _quat_mul(qyaw, qbase)
        pid = _placement_id(q)
        table.setdefault(pid, q)
        yaw_base.setdefault(pid, face)
        yaw_rank[pid] = min(yaw_rank.get(pid, rank), rank)
  yaw_ranks[key] = yaw_rank
  cache[key] = table
  reg.update(table)                                        # ids are global; keep them all
  return table


_STABLE_REST_MAX = 12        # placements kept per group under --stable-rest-poses
_STABLE_REST_PER_TIER = 6    # ... and per equal-probability tier (see _thin_rest_poses).
                             # 6 so a CUBE keeps all six of its faces: its rests are one
                             # equal-probability tier, and dropping any of them would make
                             # the flag lose placements the legacy table had.


def _thin_rest_poses(poses, group=()):
  """Keep the placement table small WITHOUT losing a qualitatively different rest.

  A rotationally symmetric shape has a CONTINUUM of rests, and the enumerator returns it as
  many poses of equal probability: a 30cm leg comes back with 24 barrel rolls, which would
  become 24 placement ids and multiply the search's branching factor for no new behaviour.
  Measured before this: chair legs 25 ids each, the assembled cross 25.

  Thinning by probability ALONE would be wrong -- it drops exactly the rare-but-essential
  rests (a leg standing on its cap is an unlikely drop and a necessary placement). So the
  poses are grouped into equal-probability TIERS, which is what a symmetry looks like from
  here, and each tier keeps at most `_STABLE_REST_PER_TIER` members. A leg then keeps a few
  barrel rolls AND both caps; a cube (6 equal, one tier) keeps all six.
  """
  tiers = {}
  for q, prob in poses:                                   # already sorted, high prob first
    tiers.setdefault(round(float(prob), 4), []).append((q, prob))
  out = []
  for _k, members in tiers.items():
    out.extend(members[:_STABLE_REST_PER_TIER])
  out = out[:_STABLE_REST_MAX]
  if len(out) < len(poses):
    print(f"[rest-poses] {tuple(group)!r}: {len(poses)} stable rests thinned to {len(out)} "
          f"({len(tiers)} probability tier(s)) -- a rotationally symmetric shape returns a "
          f"CONTINUUM of rests, and each tier keeps at most {_STABLE_REST_PER_TIER}.")
  return out


def _legacy_placement_quat(assembly, ref, face):
  # the pre-change-2 rule, kept verbatim as one of the two ways to fill the table.
  q = _FACE_DOWN_QUATS[face]
  if face == _IDENTITY_FACE:
    return q
  sq = _shape_quat(assembly.part_shape(ref))
  return q if sq is None else _quat_mul(q, _quat_conj(sq))


def placement_quat(assembly, group, ref, face):
  # REST-POSE ANCHOR (change-2-lite, [[rest-pose-offaxis-change2]]): the world orientation of
  # the reference PART FRAME for placement id `face` -- rest poses are computed FROM THE
  # SHAPE, not from the bookkeeping part frame.
  #   * Compose the INVERSE of the REFERENCE part's own shape rotation, so `face` puts that
  #     SHAPE's face flush on the table (a reclined post rests on its cap; the 90deg-rotated
  #     seat plate rests flat). Unrotated reference shapes: byte-identical.
  #   * GROUPS use the same rule via their reference part (change-2 GROUP extension,
  #     2026-07-24): e.g. the chair's backrest module (ref = the reclined back plate) can
  #     lie FLAT on the plate's face with the posts up; under the old canonical-only group
  #     anchors the module had NO rest pose at all -- an unparkable hot potato that forced
  #     hold-through-join plans and mass milestone backtracking. Stability is still judged
  #     per pose by _rest_stable, so a bad reference choice only yields unstable (dropped)
  #     candidates, never a wrong "stable" verdict.
  #   * face 5 (_IDENTITY_FACE) keeps its established semantics "as anchored/scattered"
  #     (_group_rest_pose preserves the anchor orientation there), so it is NOT composed --
  #     placement ids 0..4 are shape faces, 5 is the anchor pose.
  # The enumerator now lives in placement_table, which fills this from either the six faces
  # (default) or the shape's true stable poses (--stable-rest-poses). This function stays
  # the single lookup point, so every consumer works in both modes unchanged.
  table = placement_table(assembly, group, ref)
  q = table.get(face)
  if q is None:
    # NOT in THIS group's table. That is normal and it is not an error: a placement id names
    # a ROTATION, and a group keeps the rotation it was placed at when parts are mated onto
    # it, even though the set of rotations it could newly rest in has changed. So fall back
    # to the per-assembly registry, which holds every rotation any table has ever named.
    q = getattr(assembly, "_placement_registry", {}).get(face)
  if q is None:
    # genuinely unknown -- e.g. the labeled heuristic asking a geometric fact about a group
    # it has not built a table for. Callers treat this as "no information"; see
    # _on_table_mateable_cached, which must then NOT prune.
    raise KeyError(f"placement id {face} is unknown for {tuple(group)!r} "
                   f"(ref {ref!r}, ids {sorted(table)})")
  return q


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


_GRASP_TRANSLATIONS = 0        # extra TRANSLATED variants per base grasp (0 = off)
_GRASP_PAD_HALF = 0.011        # finger-pad HALF-WIDTH. robotiq_tight.g's pad is a capsule
  # `size: [.04, .01]` = 40mm long, 10mm RADIUS -> ~22mm wide. The width is what limits sliding
  # across the face; using the 40mm half-LENGTH instead (as this first did) halved the travel and
  # made the variants nearly invisible on a 60mm cube.


def set_grasp_translations(n, pad_half=None):
  """Enable N extra grasps per base pose: the same pinch, SLID along the directions the gripper
  is open in. Call once at startup, before any grasp precomputation (grasp ids index the DB).

  The pads close along the tool's local y and contact the two faces perpendicular to it, so the
  contact patch lives in the tool's x-z plane: sliding along local X (across the face) or local
  Z (deeper/shallower) keeps the pads on the object, while sliding along Y would open the pinch.
  The travel is bounded by the object's half-extent along that axis MINUS the pad half-length,
  so the pad still overlaps the part; a bound of <=0 means the part is too small to slide on and
  the variant collapses onto the centred grasp (harmless -- the grasp table filters duplicates).
  """
  global _GRASP_TRANSLATIONS, _GRASP_PAD_HALF
  _GRASP_TRANSLATIONS = max(0, int(n))
  if pad_half is not None:
    _GRASP_PAD_HALF = float(pad_half)


def _translation_pattern(n):
  """(local axis, fraction of the safe travel) for each extra grasp, in a stable order:
  +-half travel along X first (sliding across the face is the most useful for re-orienting),
  then +-half along Z, then the full-travel pair of each. Deterministic so grasp ids are
  reproducible run to run."""
  base = [((1.0, 0.0, 0.0), +0.5), ((1.0, 0.0, 0.0), -0.5),
          ((0.0, 0.0, 1.0), +0.5), ((0.0, 0.0, 1.0), -0.5),
          ((1.0, 0.0, 0.0), +1.0), ((1.0, 0.0, 0.0), -1.0),
          ((0.0, 0.0, 1.0), +1.0), ((0.0, 0.0, 1.0), -1.0)]
  out = []
  while len(out) < n:
    out.extend(base)
  return out[:n]


def set_dense_grasps(flag):
  """Enable the +-45deg wrist-tilted grasp variants (two-finger). Call once at startup,
  before any grasp precomputation (grasp ids index the per-part DB)."""
  global _DENSE_GRASPS
  _DENSE_GRASPS = bool(flag)


# ---------------------------------------------------------------------------------------
# SAMPLED grasps (--grasp-sampler): the mesh-based second code path. Instead of the fixed
# face-centred enumeration, tessellate the part shape into an actual MESH (trimesh) and
# SAMPLE candidate grasps off its surface -- a suction-style block (surface point + inward
# normal, canonical roll) plus an antipodal pinch block (opposing surface-point pairs within
# the finger aperture) -- then build a bounded, scene-reserved, diversity-ordered database.
# The precursor to a real learned/mesh grasp generator: swap _sample_candidates for the
# generator, keep everything else. grasp_id stays a positional index into the per-shape
# list, and part_grasp_poses / part_grasp_surfaces / part_grasp_face_min_dims all delegate
# to ONE jointly-computed, memoized (poses, surfaces, min_dims) triple so the three
# parallel views can never shear. OFF (default) = byte-identical to the enumerator.
_SAMPLED_GRASPS = None         # None = off; else current sampler configuration
_SAMPLED_VIEWS_CACHE = {}
_SAMPLED_META_CACHE = {}          # same key: per-candidate gate diagnostics (not the DB)
_SAMPLED_SCENE_CTX = None         # --grasp-sampler-scenes: pick/assembly filter context
_SAMPLED_CANONICAL_ONLY = False   # staged search refinement; IDs/DB itself never changes
_MAX_GRASP_WIDTH = 0.078       # two-finger max pinch width. The fixed-open robotiq pads
  # measure ~82mm inner gap (robotiq_tight.g capsules); 78mm keeps a seating margin. The EE
  # collision filter stays the final arbiter -- this only prunes hopeless candidates early.


def set_sampled_grasps(mode):
  """Enable/disable the sampled-grasp path. `mode` = None (off) or a dict with keys
  seed / candidates / keep. Call once at startup, BEFORE any grasp precomputation, like
  set_rolls_per_face -- grasp ids index the per-shape candidate list, so flipping this
  mid-run rebinds every saved grasp id."""
  global _SAMPLED_GRASPS, _SAMPLED_CANONICAL_ONLY
  if mode is not None and mode.get("version") not in (None, 4):
    raise ValueError("grasp sampler versions 1-3 are no longer supported")
  _SAMPLED_GRASPS = None if mode is None else {
      "seed": int(mode.get("seed", 0)),
      "candidates": max(8, int(mode.get("candidates", 200))),
      "keep": max(2, int(mode.get("keep", 12))),
      # A search may construct a larger immutable master while exposing only a prefix.
      "master_keep": max(2, int(mode.get("keep", 12)),
                         int(mode.get("master_keep", mode.get("keep", 12)))),
      "margin": float(mode.get("margin", 0.008)),
      "pad_support": float(mode.get("pad_support", 0.75)),
      "wrap": (None if mode.get("wrap") is None else float(mode.get("wrap")))}
  _SAMPLED_VIEWS_CACHE.clear()
  _SAMPLED_META_CACHE.clear()
  _SAMPLED_CANONICAL_ONLY = False


def set_sampled_canonical_only(on=False):
  """Restrict primitive grasp tables to their conventional prefix for a search epoch.

  Mesh/compound parts have no conventional database and always retain their samples. This
  changes only which already-stable ids enter precomputed grasp tables; the DB, hashes, saved
  actions and replay meanings never change.
  """
  global _SAMPLED_CANONICAL_ONLY
  _SAMPLED_CANONICAL_ONLY = bool(on)


def install_sampled_scene_ctx(C, assembly, robots=None):
  """--grasp-sampler-scenes: give the sampled DB the two SCENE views it filters candidates
  against -- the INITIAL scene (can this grasp pick the part where it lies?) and the FULLY
  ASSEMBLED scene (can this grasp stand at the part's seated pose?) -- using the search's
  OWN cheap floating-EE gate, so the filter and the search agree exactly. Rebuilds the
  sampled caches; call BEFORE any grasp-table consumer. Every grasp-table consumer must then
  thread the PART name: scene feasibility is per part INSTANCE (two identical legs sit in
  different sockets), and a part-less query raises instead of silently serving unfiltered ids."""
  global _SAMPLED_SCENE_CTX
  from solver import _search_context            # deferred: solver imports us at module top
  if robots is None:
    robots = get_robots(C)
  checker, _table = _search_context(C, assembly, robots, build_table=False)
  # One representative robot per gripper KIND. Explicit sampler poses have no action/robot
  # yet, so test each geometry separately and retain a per-kind verdict mask. The historical
  # path passed a robot-less pose to a mixed checker, which silently used slot zero.
  kind_robot = {}
  for r in robots:
    kind_robot.setdefault(checker._robot_kind[r], r)
  s0 = get_initial_state(C, assembly)
  s_full = _full_assembly_state(assembly, s0)
  poses_pick = tuple(sorted((p, tuple(np.round(np.asarray(s0.poses[p], dtype=float), 4)))
                            for p in assembly.objects))
  poses_asm = tuple(sorted((p, tuple(np.round(np.asarray(
      _state_part_poses(s_full, assembly, lift=0.0)[p], dtype=float), 4)))
                            for p in assembly.objects))
  sig = ("scenes", poses_pick, poses_asm)
  _SAMPLED_SCENE_CTX = {"checker": checker, "kind_robot": kind_robot,
                        "s0": s0, "s_full": s_full,
                        "assembly": assembly, "sig": sig,
                        "robots": tuple(robots)}
  _SAMPLED_VIEWS_CACHE.clear()
  _SAMPLED_META_CACHE.clear()


def clear_sampled_scene_ctx():
  """Drop the scene-filter context (a new scene is about to be built in this process)."""
  global _SAMPLED_SCENE_CTX
  _SAMPLED_SCENE_CTX = None
  _SAMPLED_VIEWS_CACHE.clear()
  _SAMPLED_META_CACHE.clear()


def set_sampled_grasp_keep(keep):
  """Monotonically resize the active sampled database without changing its source.

  Raw clouds and master rankings are independent of ``keep``. Increasing it therefore
  appends grasp ids; decreasing it restores an earlier prefix.  This is the only mutation the
  failure-driven search widening path needs, and deliberately retains the installed scene/IK
  context so the new prefix is ranked under exactly the same physical problem.
  """
  if _SAMPLED_GRASPS is None:
    raise ValueError("sampled grasp database is disabled")
  _SAMPLED_GRASPS["keep"] = max(2, int(keep))
  # This API is the legacy/global widening operation: it physically grows the evaluated
  # DB.  Fill-local widening never calls it; that path constructs master_keep once and
  # gates stable ids inside the concrete fill.
  _SAMPLED_GRASPS["master_keep"] = _SAMPLED_GRASPS["keep"]
  _SAMPLED_VIEWS_CACHE.clear()
  _SAMPLED_META_CACHE.clear()


def _full_assembly_state(assembly, s0):
  # the fully-assembled on-table State: every part mated to the base, anchored at the base's
  # initial pose. _state_part_poses' rest rule then lifts the group so its lowest part (the
  # legs) rests on the table -- the exact seated geometry the assemble filter must see.
  base = assembly.base
  return State(parent={base: ("table",),
                       **{p: ("part", base) for p in assembly.objects if p != base}},
               supports={base: frozenset({("table",)}),
                         **{p: frozenset({("mate", base)})
                            for p in assembly.objects if p != base}},
               poses={base: np.asarray(s0.poses[base], dtype=float).copy()})


_FEATURE_EDGE_DEG = 30.0     # a CREASE, not a tessellation step -- see _mesh_feature_edges


def _mesh_feature_edges(mesh, tol_deg=_FEATURE_EDGE_DEG):
  """(N,2,3) segment endpoints of the mesh's FEATURE edges -- the real geometric edges a
  contact must stay clear of: every edge whose dihedral angle exceeds `tol_deg`, plus the
  border edges of an open mesh. Cached on the mesh.

  THE THRESHOLD IS NOT THE FLATNESS ONE. _mesh_patch_class asks "is this region planar?"
  and 5 deg is right for that. Here the question is "is this a CREASE, or just how finely
  a smooth surface was tessellated?" -- and a tessellation step can be far larger than 5
  deg: our 48-section cylinder steps 7.5 deg, so a 5 deg rule marked every barrel edge a
  feature and read 0.97mm (the chord half-width) of clearance ANYWHERE on the barrel,
  which silently made the margin gate a no-op on every curved surface. 30 deg separates
  the two populations by a wide band: real creases are 90 deg (cube edges, cylinder cap
  rims) or ~55 deg (a chamfer), tessellation steps are single digits."""
  seg = mesh.metadata.get("_feature_edges")
  if seg is None:
    E = np.asarray(mesh.face_adjacency_edges, dtype=int)
    ang = np.asarray(mesh.face_adjacency_angles, dtype=float)
    keep = E[ang > np.radians(tol_deg)] if len(E) else np.zeros((0, 2), dtype=int)
    border = np.asarray(getattr(mesh, "edges_boundary", np.zeros((0, 2))), dtype=int)
    idx = np.vstack([keep, border]) if len(border) else keep
    V = np.asarray(mesh.vertices, dtype=float)
    seg = (V[idx[:, 0]], V[idx[:, 1]]) if len(idx) else (np.zeros((0, 3)), np.zeros((0, 3)))
    mesh.metadata["_feature_edges"] = seg
  return seg


# Contact PATCH of each end effector, as half extents in the contact plane. These mirror the
# collision proxies the rest of the pipeline uses, so "the pad is supported" here means the
# same surface the sim and the DB filter collide with.
#   two-finger: robotiq_tight.g's finger capsule is 0.04 long with a 0.01 radius, so the pad
#               is ~20mm across the closing direction and ~40mm along the approach.
#   vacuum:     a disc of wrench.PARAMS['vacuum_seal_diameter'] (0.025 by default).
_PAD_HALF_ACROSS = 0.010
_PAD_HALF_ALONG = 0.020


def _pad_samples_rect(half_u, half_v, nu=3, nv=5):
  # deterministic grid over the pad rectangle (grasp ids must be reproducible)
  us = np.linspace(-half_u, half_u, nu)
  vs = np.linspace(-half_v, half_v, nv)
  return np.array([[u, v] for u in us for v in vs], dtype=float)


def _pad_samples_disc(radius, rings=2, per_ring=8):
  pts = [[0.0, 0.0]]
  for i in range(1, rings + 1):
    r = radius * i / rings
    for k in range(per_ring):
      a = 2.0 * np.pi * k / per_ring
      pts.append([r * np.cos(a), r * np.sin(a)])
  return np.array(pts, dtype=float)


def _pad_support(mesh, p, n, u, v, offsets, lift=0.002, depth=0.003):
  """Fraction of the CONTACT PATCH that is backed by material.

  Edge clearance keeps the contact POINT away from a crease. This answers the other half of
  the question -- how much of the PAD actually lies on the part -- which is what decides
  whether a pinch holds or pivots and whether a cup can seal. A pad whose centre clears every
  edge can still hang half over a step or off the end of a short face.

  Each patch sample is lifted `lift` off the surface and cast back along -n; it counts as
  supported when it meets material within `lift + depth`. The depth bound is what makes the
  measure local: without it, a sample hanging over an edge still hits the far side of the
  part far below and would count as supported.
  """
  O = (np.asarray(p, dtype=float)[None, :]
       + offsets[:, 0:1] * np.asarray(u, dtype=float)[None, :]
       + offsets[:, 1:2] * np.asarray(v, dtype=float)[None, :]
       + float(lift) * np.asarray(n, dtype=float)[None, :])
  D = np.repeat(-np.asarray(n, dtype=float)[None, :], len(O), axis=0)
  loc, idx_ray, _tri = mesh.ray.intersects_location(O, D, multiple_hits=False)
  if len(loc) == 0:
    return 0.0
  d = np.linalg.norm(np.asarray(loc, dtype=float) - O[np.asarray(idx_ray, dtype=int)], axis=1)
  ok = np.zeros(len(O), dtype=bool)
  ok[np.asarray(idx_ray, dtype=int)[d <= lift + depth]] = True
  return float(ok.mean())


def _pad_support_wrap(mesh, p, n, u, v, offsets, lift=0.002, wrap=0.006):
  """CURVATURE-FAIR support fraction (--grasp-sampler-wrap): how much of the pad the surface
  follows within the pad's WRAP (deformation) budget.

  The legacy _pad_support lifts the whole pad along the CENTRE contact normal and counts a
  sample only when a ray along that normal meets material within lift+depth. That model is
  rigid: on a curved barrel the pad's outer columns sit the surface SAGITTA below the centre
  normal, so a physically sound side pinch of a 12.5mm-radius peg (a compliant pad wrapping
  the barrel) scored ~0.3 and every side grasp of the two-finger was dropped -- and the wrap
  roll (pads across the circumference, radial approach) scored 0.6 no matter how generous
  the budget, because a rigid ray along the centre normal MISSES the barrel once the pad
  bends away from the tangent plane. A pad does not need the surface to be coplanar with the
  centre contact -- it needs each sample to REACH the surface within what the pad can
  deform.

  Deformation distance: the signed distance from each lifted sample to the surface (positive
  outside the material, pressed-in samples count as fully supported). The reference is the
  NEAREST contact (the sample that deforms least) and a sample counts as supported when
  d_j - d_min <= wrap. A far-side hit fails through the wrap budget (its deformation is
  large). On a flat face every deformation is 0, so flat-face gating is unchanged. The peg
  barrel: axis-roll pads deform ~3mm (sagitta over the 20mm half-length) and pass a small
  wrap; the wrap roll (approach from the side) deforms ~10mm and needs ~0.012. Off
  (wrap None) = the legacy metric, byte-identical. Non-watertight meshes have no signed
  distance -- fall back to the legacy ray model there."""
  O = (np.asarray(p, dtype=float)[None, :]
       + offsets[:, 0:1] * np.asarray(u, dtype=float)[None, :]
       + offsets[:, 1:2] * np.asarray(v, dtype=float)[None, :]
       + float(lift) * np.asarray(n, dtype=float)[None, :])
  try:
    d = np.asarray(mesh.nearest.signed_distance(O), dtype=float)
  except Exception:                                        # non-watertight mesh
    D = np.repeat(-np.asarray(n, dtype=float)[None, :], len(O), axis=0)
    loc, idx_ray, _tri = mesh.ray.intersects_location(O, D, multiple_hits=False)
    d = np.full(len(O), np.inf)
    if len(loc):
      ii = np.asarray(idx_ray, dtype=int)
      d[ii] = np.linalg.norm(np.asarray(loc, dtype=float) - O[ii], axis=1)
  d[~np.isfinite(d)] = np.inf
  d[d < 0.0] = 0.0                                        # pressed into the material: contact
  dmin = float(d.min())
  if not np.isfinite(dmin):
    return 0.0
  return float((np.isfinite(d) & (d - dmin <= float(wrap))).mean())


def _pad_end_drop(mesh, p, n, along, half_len, miss=1.0):
  """How far the two ENDS of the pad fall away from the contact plane, in metres.

  A CONTINUOUS measure, unlike counting supported samples: near the best roll the hit count
  saturates (every probe point still hits), so it cannot tell 22.5 degrees off-axis from
  on-axis -- measured. The drop keeps falling all the way to the optimum, so it can be
  refined. An end that misses the part entirely scores `miss`, which dominates any real drop.
  """
  ends = np.array([p + half_len * along, p - half_len * along], dtype=float)
  O = ends + 0.02 * np.asarray(n, dtype=float)[None, :]     # start well clear of the surface
  D = np.repeat(-np.asarray(n, dtype=float)[None, :], 2, axis=0)
  loc, idx, _t = mesh.ray.intersects_location(O, D, multiple_hits=False)
  drop = np.full(2, float(miss))
  if len(loc):
    idx = np.asarray(idx, dtype=int)
    d = np.linalg.norm(np.asarray(loc, dtype=float) - O[idx], axis=1) - 0.02
    drop[idx] = np.abs(d)
  return float(drop.max())


def _best_pad_axes(mesh, p, p2, y, u0, v0, n_roll=8):
  """Choose the pinch's ROLL about the closing axis so the pads lie ALONG the part.

  The closing axis of an antipodal grasp is fixed by its two contact points, but the roll
  about it is free -- and the sampler used to fix it with an arbitrary reference vector.
  That is why a pinch on a 30mm barrel scored 0.20 pad support: the pad is 40mm long, and an
  arbitrary roll wraps that length AROUND the cylinder instead of laying it along the axis.
  The roll is the difference between a pad touching on a line and a pad lying flat.

  The objective is the worst END DROP over the two pads (see _pad_end_drop) -- minimised by
  a coarse sweep plus one refinement pass. Returns (across, along) unit axes in the plane the
  closing axis is normal to; `along` is the pad's long axis and the grasp's approach.
  Deterministic: a fixed angle grid, ties broken by the smaller angle.
  """
  half = _PAD_HALF_ALONG

  def _cost(a):
    across = np.cos(a) * u0 + np.sin(a) * v0
    along = -np.sin(a) * u0 + np.cos(a) * v0
    return (max(_pad_end_drop(mesh, p, -y, along, half),
                _pad_end_drop(mesh, p2, y, along, half)), across, along)

  step = np.pi / float(n_roll)                             # a pad is 180deg symmetric
  best_a, best = 0.0, float("inf")
  for i in range(int(n_roll)):
    c, _x, _z = _cost(i * step)
    if c < best - 1e-12:
      best, best_a = c, i * step
  for k in (-0.5, -0.25, 0.25, 0.5):                       # refine around the winner
    a = best_a + k * step
    c, _x, _z = _cost(a)
    if c < best - 1e-12:
      best, best_a = c, a
  _c, across, along = _cost(best_a)
  return across, along


def _plane_axes(n):
  """Two unit axes spanning the plane with normal `n` (deterministic choice)."""
  n = np.asarray(n, dtype=float)
  ref = np.array([0.0, 0.0, 1.0]) if abs(float(n[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
  u = np.cross(n, ref)
  u /= max(np.linalg.norm(u), 1e-12)
  return u, np.cross(n, u)


def _edge_clearance(mesh, pts):
  """Distance from each point to the nearest FEATURE edge (see _mesh_feature_edges), in
  metres. This is what keeps a sampled contact off the corners: a pad or a suction cup
  parked on an edge has half its area over nothing, and the analytic patch reasoning
  (surface class, seal diameter) silently assumes the contact sits INSIDE its face.
  inf when the shape has no feature edges at all (e.g. a sphere)."""
  A, B = _mesh_feature_edges(mesh)
  P = np.atleast_2d(np.asarray(pts, dtype=float))
  if len(A) == 0:
    return np.full(len(P), np.inf)
  AB = B - A                                               # (E,3)
  L2 = np.einsum("ij,ij->i", AB, AB)
  L2 = np.where(L2 > 1e-18, L2, 1.0)
  t = np.einsum("pej,ej->pe", P[:, None, :] - A[None, :, :], AB) / L2   # (P,E) foot param
  t = np.clip(t, 0.0, 1.0)
  closest = A[None, :, :] + t[:, :, None] * AB[None, :, :]
  return np.linalg.norm(P[:, None, :] - closest, axis=2).min(axis=1)


def _spec_mesh(spec):
  # tessellate a shape spec into a trimesh (SHAPE frame; the caller composes the spec's
  # own rotation). ssBox corner rounding is ignored: the flat faces are where grasps live,
  # and the EE collision filter runs against the true rai geometry anyway.
  import trimesh
  kind, p = spec[0], spec[1]
  if kind == "box":
    return trimesh.creation.box(extents=[float(p[0]), float(p[1]), float(p[2])])
  if kind == "cylinder":
    return trimesh.creation.cylinder(radius=float(p[1]), height=float(p[0]), sections=48)
  if kind == "sphere":
    return trimesh.creation.icosphere(subdivisions=3, radius=float(p[0]))
  if kind == "mesh":
    return _load_mesh(p)["mesh"]                           # the REAL surface, not the hull
  if kind == "compound":
    return _compound_mesh(spec)                            # children transformed + concatenated
  raise ValueError(f"--grasp-sampler: unknown shape kind {kind!r}")


def _mesh_patch_class(mesh, f0, tol_deg=5.0):
  """(surface label, flat-patch min extent) at face `f0` of a real mesh -- the mesh
  counterpart of _surface_class_and_min_dim. Flood-fills the coplanar patch around f0
  (adjacent faces within `tol_deg` dihedral), then measures the patch's smallest in-plane
  extent; that is the number the vacuum seal gate compares against its cup diameter. A
  patch thinner than 1 mm counts as curved (no seal, torque de-rated like any curved
  grasp). The adjacency map is cached on the mesh's metadata (one mesh, many samples)."""
  nbr = mesh.metadata.get("_patch_nbr")
  if nbr is None:
    nbr = {}
    for (a, b), t in zip(np.asarray(mesh.face_adjacency, dtype=int),
                         np.asarray(mesh.face_adjacency_angles, dtype=float)):
      if t <= np.radians(tol_deg):
        nbr.setdefault(int(a), []).append(int(b))
        nbr.setdefault(int(b), []).append(int(a))
    mesh.metadata["_patch_nbr"] = nbr
  seen, stack = {int(f0)}, [int(f0)]
  while stack:
    x = stack.pop()
    for y in nbr.get(x, ()):
      if y not in seen:
        seen.add(y)
        stack.append(y)
  faces = np.fromiter(seen, dtype=int)
  vs = mesh.vertices[np.unique(np.asarray(mesh.faces, dtype=int)[faces].ravel())]
  n = np.asarray(mesh.face_normals[int(f0)], dtype=float)
  ref = np.array([0.0, 0.0, 1.0]) if abs(float(n[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
  u = np.cross(n, ref); u /= max(np.linalg.norm(u), 1e-12)
  v = np.cross(n, u)
  proj = np.column_stack([vs @ u, vs @ v])
  ext = proj.max(0) - proj.min(0)
  mind = float(min(ext))
  return ("flat", mind) if mind > 1e-3 else ("curved", 0.0)


def _surface_class_and_min_dim(spec, n):
  # (surface label, flat-patch min extent) at a surface point with outward normal `n`
  # (SHAPE frame). Analytic for the primitives; a mesh version estimates the patch instead.
  kind, p = spec[0], spec[1]
  if kind == "box":
    i = int(np.argmax(np.abs(n)))
    ext = [float(p[0]), float(p[1]), float(p[2])]
    return "flat", min(e for j, e in enumerate(ext) if j != i)
  if kind == "cylinder":
    if abs(float(n[2])) > 0.99:                            # cap disc
      return "flat", 2.0 * float(p[1])
    return "curved", 0.0
  if kind in ("mesh", "compound"):
    raise ValueError("mesh/compound surfaces are classified per-face via _mesh_patch_class")
  return "curved", 0.0                                     # sphere


def _ray_exit(spec, p, d):
  """(exit point, outward normal there) of the ray p + t*d, t > 0, leaving the shape
  (SHAPE frame; p inside or on the surface). Analytic for the primitives -- the mesh is
  only the SAMPLING surface, so the opposite-point query needs no ray-tracing backend;
  a real mesh swaps this helper for a mesh ray cast."""
  kind, prm = spec[0], spec[1]
  d = np.asarray(d, dtype=float); p = np.asarray(p, dtype=float)
  if kind == "box":
    h = _shape_half_extents(spec)
    ts = [(((h[i] if d[i] > 0 else -h[i]) - p[i]) / d[i], i) for i in range(3)
          if abs(d[i]) > 1e-12]
    t, i = min((t, i) for t, i in ts if t > -1e-12) if ts else (0.0, 2)
    n = np.zeros(3); n[i] = 1.0 if d[i] > 0 else -1.0
    return p + t * d, n
  if kind == "cylinder":
    hz, r = float(prm[0]) / 2.0, float(prm[1])
    cands = []
    if abs(d[2]) > 1e-12:                                  # caps
      t = ((hz if d[2] > 0 else -hz) - p[2]) / d[2]
      if t > -1e-12:
        cands.append((t, np.array([0.0, 0.0, 1.0 if d[2] > 0 else -1.0]), True))
    a = d[0] * d[0] + d[1] * d[1]
    if a > 1e-12:                                          # barrel
      b = p[0] * d[0] + p[1] * d[1]
      c = p[0] * p[0] + p[1] * p[1] - r * r
      disc = b * b - a * c
      if disc >= 0.0:
        t = (-b + np.sqrt(disc)) / a
        if t > -1e-12:
          e = p + t * d
          nr = np.array([e[0], e[1], 0.0])
          cands.append((t, nr / max(np.linalg.norm(nr), 1e-12), False))
    valid = [(t, n) for t, n, inside in cands
             if (inside and np.hypot(*(p + t * d)[:2]) <= r + 1e-9)
             or (not inside and abs((p + t * d)[2]) <= hz + 1e-9)]
    if not valid:
      return p, np.array([0.0, 0.0, 1.0])
    t, n = min(valid, key=lambda tn: tn[0])
    return p + t * d, n
  if kind in ("mesh", "compound"):
    m = _load_mesh(prm)["mesh"] if kind == "mesh" else _compound_mesh(spec)
    loc, _ri, tri = m.ray.intersects_location([p], [d], multiple_hits=True)   # the mesh ray
                                                           # cast the docstring promised
    if len(loc):
      ts = (np.asarray(loc, dtype=float) - p) @ d / float(d @ d)
      ok = np.flatnonzero(ts > 1e-12)
      if len(ok):
        i = int(ok[np.argmin(ts[ok])])                     # first wall along d, like the
        return (np.asarray(loc[i], dtype=float),           # primitives' smallest positive t
                np.asarray(m.face_normals[int(tri[i])], dtype=float))
    return p, -d / max(np.linalg.norm(d), 1e-12)           # no exit: degenerate, caller gates
  # sphere
  r = float(prm[0])
  b = float(p @ d); c = float(p @ p) - r * r
  disc = b * b - float(d @ d) * c
  t = (-b + np.sqrt(max(disc, 0.0))) / float(d @ d)
  e = p + t * d
  return e, e / max(np.linalg.norm(e), 1e-12)


def _fps_row_order(rows):
  """Deterministic farthest-first ORDER, independent of the final retained count.

  Taking a longer prefix appends grasp ids instead of rebinding old ones, which makes
  search-side grasp widening safe.
  """
  if not rows:
    return []
  def feat(row):
    pose = np.asarray(row[0], dtype=float)
    R = _quat_to_R(pose[3:])
    return np.concatenate([pose[:3], 0.02 * R[:, 0], 0.02 * R[:, 1], 0.02 * R[:, 2]])
  F = np.asarray([feat(r) for r in rows], dtype=float)
  chosen, left = [0], np.ones(len(rows), dtype=bool)
  left[0] = False
  d = np.linalg.norm(F - F[0], axis=1)
  while np.any(left):
    i = int(np.argmax(np.where(left, np.round(d, 12), -np.inf)))
    chosen.append(i); left[i] = False
    d = np.minimum(d, np.linalg.norm(F - F[i], axis=1))
  return [rows[i] for i in chosen]


def _canonical_sample_rows(spec, offset):
  """Conventional primitive grasps in the sampled row representation.

  The order is deliberately byte-for-byte the legacy enumerator order (face-major, then roll):
  existing search tie-breaks and action ids remain unchanged. Mesh and compound shapes have no
  honest six-face semantics and contribute no conventional prefix.
  """
  if spec[0] in ("mesh", "compound"):
    return []
  h = _shape_half_extents(spec)
  rows = []
  for normal in _CUBE_FACE_NORMALS:
    for roll in range(_ROLLS_PER_FACE):
      standoff = float(np.abs(normal) @ h) + offset
      angle = roll * (np.pi / _ROLLS_PER_FACE)
      roll_q = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
      q = _quat_mul(_grasp_quat_for_normal(normal), roll_q)
      pose = np.concatenate([standoff * normal, q])
      surf, mind = _surface_class_and_min_dim(spec, normal)
      contact = (standoff - offset) * normal
      rows.append((pose, surf, float(mind), np.concatenate([contact, 0.02 * normal]),
                   float("inf"), 1.0, "canonical", False))
  return rows


def _poses_near(a, b, pos_tol=1e-3, angle_tol=np.deg2rad(5.0)):
  a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
  if float(np.linalg.norm(a[:3] - b[:3])) > pos_tol:
    return False
  dot = min(1.0, abs(float(np.dot(a[3:], b[3:]))))
  return 2.0 * np.arccos(dot) <= angle_tol


def _closure_ranked_canonical(entries):
  """Deterministic, K-independent ordering of conventional grasps by scene utility.

  Primitive parts contribute 24 conventional face/roll rows, so a practical ``keep`` of
  16 or 20 otherwise truncates the list *before* the sampled-row scene ranking below is ever
  reached.  Under the explicit virtual-close flag, rank those conventional rows by the number
  of (gripper kind, pick/assemble role) certificates they satisfy.  Farthest-first ordering
  within each equal-coverage tier preserves geometric diversity.  Zero-certificate rows stay
  at the end as a fallback: this is prioritisation, not a new hard sampler rejection.

  The complete order is independent of ``keep``, so increasing K still only appends ids.
  Without the closure flag the caller retains the byte-for-byte legacy canonical order.
  """
  tiers = {}
  for entry in entries:
    meta = entry[1]
    coverage = sum(len(roles) for roles in
                   (meta.get("closure_by_kind") or {}).values())
    tiers.setdefault(coverage, []).append(entry)
  out = []
  for coverage in sorted(tiers, reverse=True):
    block = tiers[coverage]
    by_id = {id(e[0]): e for e in block}
    out.extend(by_id[id(row)] for row in _fps_row_order([e[0] for e in block]))
  return out


def _sampled_master_rows(spec, offset, part, suction, antip, keep, ctx, shape_quat):
  """Select the bounded scene-reserved DB and return ``[(row, metadata), ...]``.

  Conventional primitive grasps are the stable source prefix. Under the explicit closure
  model, a source whose fingers graze the table may additionally emit a deterministic,
  contact-preserving approach-retracted pose; the colliding source remains eligible as a
  free-space fallback if the bounded set has room. Sampled rows are ordered by a round-robin over fleet/scene/family strata,
  with farthest-first order inside every stratum. All rankings are independent of ``keep``;
  a larger budget therefore preserves every old id for a fixed scene/configuration.
  """
  def part_pose(row):
    pose = np.asarray(row[0], dtype=float)
    if shape_quat is None:
      return pose
    return np.concatenate([_quat_to_R(shape_quat) @ pose[:3],
                           _quat_mul(shape_quat, pose[3:])])

  checker = None if ctx is None else ctx["checker"]
  kind_robot = {} if ctx is None else ctx["kind_robot"]
  asm = None if ctx is None else ctx["assembly"]
  seal = float(wrench.PARAMS.get("vacuum_seal_diameter", 0.0))

  # Physical generation policy, not a collision relaxation. A candidate already places the
  # pads at a valid antipodal contact; translating the hand OUTWARD along its approach slides
  # those long pads over the same target. At most half a pad length is available for this
  # adjustment, and the strict bilateral-contact closure certificate remains authoritative.
  # The 0.2mm positive clearance avoids handing a numerical tangent to strict real-IK.
  retract_max = 0.5 * _PAD_HALF_ALONG
  retract_clearance = 2.0e-4

  def row_at_part_pose(row, pose):
    """Replace a row pose supplied in PART coordinates (rows precede shape rotation)."""
    pose = np.asarray(pose, dtype=float)
    if shape_quat is None:
      raw = pose
    else:
      qi = _quat_conj(shape_quat)
      raw = np.concatenate([_quat_to_R(shape_quat).T @ pose[:3],
                            _quat_mul(qi, pose[3:])])
    return (raw,) + tuple(row[1:])

  def repaired_table_pose(row, kind, robot, role, state):
    """Smallest analytic approach retraction that clears a finger/table overlap.

    Only the table case is projected: clutter and another hand are discrete obstacles, not
    permission to silently change the grasp. The returned pose is re-certified against the
    complete open hand and closure sweep, so this never converts collision into tolerance.
    """
    if checker is None or not checker.closure_check or kind != "two_finger":
      return None
    pose = part_pose(row)
    res = checker.pose_grasp_closure_result(state, part, pose, asm, robot)
    if res is None or res.feasible or res.reason != "sweep-collision" or res.detail is None:
      return None
    a, b, depth = res.detail
    if "table" not in (str(a), str(b)) or float(depth) >= 0.0:
      return None

    # Retraction direction in the part frame is -tool-z (the same convention as _backoff_ee).
    # Project it into world z to turn the measured table penetration into the minimum travel.
    retract_part = -_quat_to_R(pose[3:]) @ np.array([0.0, 0.0, 1.0])
    part_world = _state_part_poses(state, asm, lift=0.0)[part]
    lift_per_m = float((_quat_to_R(part_world[3:]) @ retract_part)[2])
    if lift_per_m <= 1e-6:
      return None                                          # this approach cannot leave the table
    delta = (-float(depth) + retract_clearance) / lift_per_m
    if delta <= 0.0 or delta > retract_max:
      return None

    # One analytic step normally suffices. The two bounded cushions handle rounded-box/FCL
    # distance noise without introducing an assembly-specific search grid.
    for cushion in (0.0, 2.5e-4, 5.0e-4):
      d = delta + cushion
      if d > retract_max:
        continue
      candidate = pose.copy()
      candidate[:3] += d * retract_part
      if not checker.pose_collision_free(state, part, candidate, asm, robot=robot):
        continue
      cert = checker.pose_grasp_closure_result(state, part, candidate, asm, robot)
      if cert is not None and cert.feasible:
        return candidate, float(d), ("approach-retract", kind, role)
    return None

  def enrich(row, origin, refinement=None):
    roles = {}
    closure_roles = {}
    for kind, robot in sorted(kind_robot.items()):
      # Pinch rows deliberately carry min_dim=0 and are not vacuum candidates. Canonical and
      # suction rows still need a flat patch large enough to seal.
      kind_ok = not (kind == "vacuum" and
                     (row[6] == "pinch" or row[1] == "curved" or row[2] < seal))
      hits = []
      close_hits = []
      if kind_ok:
        pose = part_pose(row)
        for role, state in (("pick", ctx["s0"]), ("assemble", ctx["s_full"])):
          if not checker.pose_collision_free(state, part, pose, asm, robot=robot):
            continue
          # With --grasp-closure-check, a scene label means OPEN geometry clears and the
          # receiver can actually close in that scene.  Failed closes are not thrown away:
          # they remain in the family/FPS fallback pools below, but no longer consume the
          # scarce pick/assemble-reserved slots ahead of closure-capable candidates.
          close_ok = checker.pose_grasp_closure_free(
              state, part, pose, asm, robot) if checker.closure_check else True
          if close_ok:
            hits.append(role)
            close_hits.append(role)
      roles[kind] = tuple(hits)
      closure_roles[kind] = tuple(close_hits)
    meta = {"origin": origin, "scene_by_kind": roles,
            "closure_by_kind": closure_roles,
            "sets": tuple(sorted({x for xs in roles.values() for x in xs}))}
    if refinement is not None:
      tag, delta, source = refinement
      meta["refinement"] = (tag[0], tag[1], tag[2], round(float(delta), 6), source)
    return (row, meta)

  def enrich_variants(row, origin):
    """Original free-space pose plus unique strict scene-repaired poses."""
    out = [enrich(row, origin)]
    if checker is None or not checker.closure_check:
      return out
    poses = [part_pose(row)]
    for kind, robot in sorted(kind_robot.items()):
      kind_ok = not (kind == "vacuum" and
                     (row[6] == "pinch" or row[1] == "curved" or row[2] < seal))
      if not kind_ok:
        continue
      for role, state in (("pick", ctx["s0"]), ("assemble", ctx["s_full"])):
        repaired = repaired_table_pose(row, kind, robot, role, state)
        if repaired is None:
          continue
        pose, delta, tag = repaired
        if any(_poses_near(pose, old, pos_tol=1e-4, angle_tol=1e-6) for old in poses):
          continue
        poses.append(pose)
        refined_row = row_at_part_pose(row, pose)
        out.append(enrich(refined_row, origin + "-refined",
                          refinement=(tag, delta, origin)))
    return out

  canonical = [entry for r in _canonical_sample_rows(spec, offset)
               for entry in enrich_variants(r, "canonical")]
  # Conventional primitive grasps usually fill the entire K budget.  Let the explicit closure
  # model influence that prefix too; otherwise closure-aware ranking would only affect meshes
  # or unusually large K and be a no-op for the cuboid assemblies that motivated the check.
  if checker is not None and checker.closure_check:
    canonical = _closure_ranked_canonical(canonical)
  closure_select = bool(checker is not None and checker.closure_check)
  # A strict repaired pose is useless if the immutable source prefix still consumes the whole
  # bounded K before sampled scene-valid rows are considered. Under closure semantics reserve
  # the prefix only for candidates that actually serve PICK/ASSEMBLE; zero-role originals stay
  # available as the LAST fallback below. Off = the historical immutable-prefix path exactly.
  if closure_select:
    canonical_seed = [e for e in canonical if any(e[1]["scene_by_kind"].values())][:keep]
    canonical_fallback = [e for e in canonical if not any(e[1]["scene_by_kind"].values())]
  else:
    canonical_seed, canonical_fallback = list(canonical), []

  margin = float(_SAMPLED_GRASPS.get("margin", 0.0))
  support = float(_SAMPLED_GRASPS.get("pad_support", 0.0))
  raw = [r for r in list(suction) + list(antip)
         if not any(_poses_near(part_pose(r), part_pose(c[0])) for c in canonical)]
  raw_good = [r for r in raw if r[4] >= margin and r[5] >= support]
  raw_good_ids = {id(r) for r in raw_good}
  raw_bad = sorted((r for r in raw if id(r) not in raw_good_ids),
                   key=lambda r: (-r[5], -r[4], tuple(np.round(r[0], 9))))

  # Scene collision checks are much dearer than mesh sampling. Classifying all ~600 raw
  # pinch-roll rows cost 60-80 seconds on car-mini before search even began. Probe a fixed,
  # K-INDEPENDENT farthest-first master pool instead: enough diversity for any practical K,
  # while preserving prefix identity when only `keep` grows during refinement.
  probe_categories = []
  for pred in (lambda r: r[6] == "suction",
               lambda r: r[6] == "pinch" and not r[7],
               lambda r: r[6] == "pinch" and r[7]):
    block = [r for r in raw_good if pred(r)]
    if block:
      probe_categories.append(_fps_row_order(block))
  if raw_bad:
    probe_categories.append(_fps_row_order(raw_bad))
  probe, probe_seen, cursors = [], set(), [0] * len(probe_categories)
  probe_cap = min(len(raw), 64 if spec[0] in ("mesh", "compound") else 32)
  while len(probe) < probe_cap:
    progressed = False
    for j, block in enumerate(probe_categories):
      while cursors[j] < len(block) and id(block[cursors[j]]) in probe_seen:
        cursors[j] += 1
      if cursors[j] >= len(block):
        continue
      row = block[cursors[j]]; cursors[j] += 1
      probe.append(row); probe_seen.add(id(row)); progressed = True
      if len(probe) >= probe_cap:
        break
    if not progressed:
      break
  sampled = [entry for r in probe for entry in enrich_variants(r, "sampled")]

  good = [e for e in sampled if e[0][4] >= margin and e[0][5] >= support]
  good_ids = {id(e[0]) for e in good}
  bad = sorted((e for e in sampled if id(e[0]) not in good_ids),
               key=lambda e: (-e[0][5], -e[0][4], tuple(np.round(e[0][0], 9))))

  def ordered(entries):
    by_id = {id(e[0]): e for e in entries}
    return [by_id[id(r)] for r in _fps_row_order([e[0] for e in entries])]

  categories = []
  # Scene/fleet needs first. A row may appear in several categories; round-robin deduplication
  # turns that into useful multi-purpose priority rather than duplicate ids.
  for kind in sorted(kind_robot):
    usable = [e for e in good if (kind != "vacuum" or
              (e[0][6] != "pinch" and e[0][1] != "curved" and e[0][2] >= seal))]
    for role in ("pick", "assemble"):
      block = [e for e in usable if role in e[1]["scene_by_kind"].get(kind, ())]
      if block:
        categories.append(ordered(block))
  for pred in (lambda e: e[0][6] == "suction",
               lambda e: e[0][6] == "pinch" and not e[0][7],
               lambda e: e[0][6] == "pinch" and e[0][7]):
    block = [e for e in good if pred(e)]
    if block:
      categories.append(ordered(block))
  if good:
    categories.append(ordered(good))
  # Materialise the complete stable tail before slicing. Under the closure model zero-role
  # conventional fallbacks remain after the sampled scene-positive pools instead of being
  # promoted merely because they are conventional. Likewise, below-gate sampled rows are
  # appended only after every gate-passing row; they fill an otherwise undersized database but
  # must never be round-robin interleaved into a prefix that already has enough valid rows.
  tail, tail_seen = list(canonical_seed), {id(e[0]) for e in canonical_seed}
  cursors = [0] * len(categories)
  while True:
    progressed = False
    for j, block in enumerate(categories):
      while cursors[j] < len(block) and id(block[cursors[j]][0]) in tail_seen:
        cursors[j] += 1
      if cursors[j] >= len(block):
        continue
      e = block[cursors[j]]; cursors[j] += 1
      tail.append(e); tail_seen.add(id(e[0])); progressed = True
    if not progressed:
      break
  for e in ordered(bad):                                  # finite fallback for tiny/curved parts
    if id(e[0]) not in tail_seen:
      tail.append(e); tail_seen.add(id(e[0]))
  for e in canonical_fallback:                           # free-space source poses: last resort
    if id(e[0]) not in tail_seen:
      tail.append(e); tail_seen.add(id(e[0]))

  # Reserve one deterministic row for every available (gripper-kind, PICK/ASSEMBLE) fact,
  # then append the stable tail. A bridge row can satisfy several facts and is emitted once.
  # The complete order is built before slicing, so widening never rebinds an old grasp id.
  def has_scene_fact(entry, fact):
    kind, role = fact
    return role in entry[1]["scene_by_kind"].get(kind, ())

  facts = [(kind, role) for kind in sorted(kind_robot)
           for role in ("pick", "assemble")]
  primary = list(canonical) + list(good)
  reserve_blocks = []
  for fact in facts:
    block = [e for e in primary if has_scene_fact(e, fact)]
    if not block:
      block = [e for e in bad if has_scene_fact(e, fact)]
    if block:
      reserve_blocks.append((fact, ordered(block)))

  out, seen, covered = [], set(), set()
  for fact, block in reserve_blocks:
    if fact in covered:
      continue
    entry = next((e for e in block if id(e[0]) not in seen), None)
    if entry is None:
      continue
    out.append(entry)
    seen.add(id(entry[0]))
    for kind, roles in entry[1]["scene_by_kind"].items():
      for role in roles:
        if role in ("pick", "assemble"):
          covered.add((kind, role))

  for entry in tail:
    if id(entry[0]) not in seen:
      out.append(entry); seen.add(id(entry[0]))
  return out[:keep]


def _sampled_keys(spec, offset, part=None):
  """(base_key, cache_key) for the sampled DB. base_key seeds the RAW candidate cloud --
  identical parts share it, so their clouds match. The cache key adds the PART INSTANCE and
  the scene-filter signature (--grasp-sampler-scenes), so each part filters its own cloud."""
  cfgv = _SAMPLED_GRASPS
  # The constant 2 is the established raw-cloud RNG salt. It is not a selectable algorithm
  # version; retaining it keeps current sampler outputs stable while the public version switch
  # and all obsolete generation branches are gone.
  base = (_spec_key(spec), float(offset), 2, cfgv["seed"], cfgv["candidates"],
          float(cfgv.get("margin", 0.0)), float(cfgv.get("pad_support", 0.0)),
          cfgv.get("wrap"))
  ctx = _SAMPLED_SCENE_CTX
  cache = base + (cfgv.get("master_keep", cfgv["keep"]),)
  return base, cache + (part, (ctx["sig"] if ctx is not None else None))


def _sampled_grasp_views(spec, offset, part=None):
  """The joint (poses, surfaces, min_dims) triple for the sampled path -- ONE memoized
  computation so the three parallel per-gid views cannot drift apart. Poses follow the
  enumerator's frame conventions exactly: local z = approach INTO the part (the asymmetric
  wrench model's press/pull sign hangs on it), local y = finger closing axis, position =
  tool centre standing `offset` off the approach-side surface (the wrench moment arm).

  With the scene context installed (--grasp-sampler-scenes), `part` names the PART INSTANCE
  and every candidate is additionally tagged pick-feasible (floating EE clear at the part's
  INITIAL pose) and assemble-feasible (floating EE clear at the part's ASSEMBLED pose); the
  keep budget then splits into per-set quotas so each purpose keeps a guaranteed share."""
  import zlib
  cfgv = _SAMPLED_GRASPS
  ctx = _SAMPLED_SCENE_CTX
  if ctx is not None and part is None:
    raise ValueError("--grasp-sampler-scenes: the grasp DB is per PART INSTANCE; this spec "
                     f"({spec!r}) was queried without a part name. Every grasp-table "
                     "consumer must thread the part.")
  sq = _shape_quat(spec) if spec[0] != "compound" else None
  wrap = cfgv.get("wrap")
  base_key, key = _sampled_keys(spec, offset, part)
  hit = _SAMPLED_VIEWS_CACHE.get(key)
  if hit is not None:
    return hit
  import trimesh
  mesh = _spec_mesh(spec)
  # PRIVATE RNG, seeded from (sampler seed, canonical spec key): byte-identical across
  # processes and entry points -- the global np.random stream is drained path-dependently
  # by scatter layout and must never feed the id-bearing pipeline. Identical parts share the
  # same RAW cloud (the base_key has no part); only the scene filter below differs per part.
  rng_seed = zlib.crc32(repr(base_key).encode())
  pts, fidx = trimesh.sample.sample_surface(mesh, cfgv["candidates"], seed=int(rng_seed))
  normals = mesh.face_normals[np.asarray(fidx, int)]

  # EDGE CLEARANCE of every sampled point, up front (vectorised): the margin gate below
  # keeps contacts INSIDE their face instead of hanging over a corner.
  clr = _edge_clearance(mesh, np.asarray(pts, dtype=float))
  margin = float(cfgv.get("margin", 0.0))
  min_support = float(cfgv.get("pad_support", 0.0))
  _cup = _pad_samples_disc(0.5 * float(wrench.PARAMS.get("vacuum_seal_diameter", 0.025)))
  _rect = _pad_samples_rect(_PAD_HALF_ACROSS, _PAD_HALF_ALONG)

  suction, antip = [], []              # (pose, surf, min_dim, feature, clr, support, family)
  for k in range(len(pts)):
    p, n = np.asarray(pts[k], float), np.asarray(normals[k], float)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
      continue
    n = n / nn
    if spec[0] in ("mesh", "compound"):
      surf, mind = _mesh_patch_class(mesh, int(fidx[k]))
    else:
      surf, mind = _surface_class_and_min_dim(spec, n)
    # ---- suction-style candidate: approach along -n, canonical roll ----
    q = _grasp_quat_for_normal(n)
    pose = np.concatenate([p + offset * n, q])
    _u, _v = _plane_axes(n)                                # the cup sits in the contact plane
    if min_support > 0.0:
      _sup = (_pad_support_wrap(mesh, p, n, _u, _v, _cup, wrap=wrap) if wrap is not None
              else _pad_support(mesh, p, n, _u, _v, _cup))
    else:
      _sup = 1.0
    suction.append((pose, surf, mind, np.concatenate([p, 0.02 * n]), float(clr[k]), _sup,
                    "suction", False))
    # ---- antipodal candidate: the opposing surface point within the aperture ----
    p2, n2 = _ray_exit(spec, p - 1e-9 * n, -n)
    width = float(np.linalg.norm(p2 - p))
    if width > 1e-4 and width <= _MAX_GRASP_WIDTH and float(n2 @ (-n)) > 0.95:
      y = (p2 - p) / width                                 # closing axis: pad-to-pad
      _u0, _v0 = _plane_axes(y)                            # the free roll lives in this plane
      if min_support > 0.0:
        _, z = _best_pad_axes(mesh, p, p2, y, _u0, _v0)    # lay the pads ALONG the part
      else:
        ref = (np.array([0.0, 0.0, 1.0]) if abs(float(y[2])) < 0.9
               else np.array([1.0, 0.0, 0.0]))
        z = np.cross(y, ref); z /= np.linalg.norm(z)       # legacy: an arbitrary roll
      # x FROM y AND z, always: the frame must stay RIGHT-handed. Taking the roll search's
      # own two axes as (x, z) makes the triple LEFT-handed -- _mat_to_quat then returns a
      # reflected frame, so the recovered approach and closing axes are not the ones that
      # were chosen. The existing convention tests caught exactly that.
      x = np.cross(y, z)
      mid = 0.5 * (p + p2)
      # label = the contacted surface class; min_dim 0.0 keeps the vacuum seal gate shut
      # for pinch candidates (the suction block is the vacuum's own candidate family).
      # BOTH pads must clear the edges -- a pinch is only as good as its worse contact.
      # each PAD lies in the plane the closing axis y is normal to, and it is long along
      # the approach z -- so the rectangle's axes are (x across, z along).
      # A pinch has TWO physically different approach directions for the same contact pair.
      # Keeping only ``z`` is not merely a lost wrist-roll: for a peg, +axis puts the palm
      # above the peg (through the socket plate at the seated pose), while -axis keeps the
      # palm below the socket.  Emit both and let FPS retain directional diversity.  This is
      # the sampled/mesh counterpart of the face enumerator's two in-plane rolls.
      # --grasp-sampler-wrap adds the COMPLEMENTARY ROLL: the roll search above lays the pad
      # LONG axis along the part (a barrel contact approaches along the peg axis; a long
      # BOX's pads run along its long axis). A peg lying flat on a table is only pickable
      # the other way -- pads across the short dimension, fingers across the thickness,
      # approach from the side -- for a curved barrel that roll bends the pad to the
      # surface (its support needs a wrap budget: sagitta over the pad length, ~12 mm on
      # the 25 mm peg); for a flat box the same roll just crosses the narrow faces, but it
      # is the ONLY grasp whose palm does not sit at the peg END (which collides with the
      # seat during insertion). The wrap metric scores both exactly the same way.  Emit
      # the complementary roll as an extra +- pair whenever the flag is set -- keeping the
      # default DB byte-identical requires the default path to emit nothing extra.
      z_rolls = [z, -z]
      if wrap is not None:
        z_rolls += [x, -x]
      for zi, z_dir in enumerate(z_rolls):
        x_dir = np.cross(y, z_dir)                         # right-handed (x, y, z)
        if min_support > 0.0:                              # support per roll: the wrap roll
          if wrap is not None:                             # bends differently
            _sup = min(_pad_support_wrap(mesh, p, -y, x_dir, z_dir, _rect, wrap=wrap),
                       _pad_support_wrap(mesh, p2, y, x_dir, z_dir, _rect, wrap=wrap))
          else:
            _sup = min(_pad_support(mesh, p, -y, x_dir, z_dir, _rect),
                       _pad_support(mesh, p2, y, x_dir, z_dir, _rect))
        else:
          _sup = 1.0
        p_exit, _n_exit = _ray_exit(spec, mid, -z_dir)     # approach-side surface
        t = float(np.linalg.norm(p_exit - mid))
        pose = np.concatenate([
            mid - z_dir * (t + offset),
            _mat_to_quat(np.column_stack([x_dir, y, z_dir])),
        ])
        antip.append((pose, surf, 0.0,
                      np.concatenate([mid, 0.02 * y, 0.02 * z_dir]),
                      min(float(clr[k]), float(_edge_clearance(mesh, p2)[0])),
                      _sup, "pinch", zi >= 2))             # zi >= 2: the wrap-roll pair

  # The positional list is the finite MASTER. Concrete local fills may expose a smaller
  # per-part prefix without regenerating this list, so ids and all cached IK witnesses stay
  # valid across widening epochs.
  keep = cfgv.get("master_keep", cfgv["keep"])
  selected = _sampled_master_rows(spec, offset, part, suction, antip, keep, ctx, sq)
  if not selected:
    raise ValueError(f"--grasp-sampler: no feasible candidates for shape {spec!r}")
  poses, surfs, mins, meta = [], [], [], []
  for row, extra in selected:
    pose, surf, mind, _f, _clr, _sup, _fam = row[:7]
    pose = np.asarray(pose, dtype=float)
    if sq is not None:
      pose = np.concatenate([_quat_to_R(sq) @ pose[:3], _quat_mul(sq, pose[3:])])
    pose.setflags(write=False)
    poses.append(pose); surfs.append(surf); mins.append(float(mind))
    meta.append({"edge_clearance": float(_clr), "pad_support": float(_sup),
                 "family": _fam, **extra})
  _SAMPLED_META_CACHE[key] = tuple(meta)
  hit = _SAMPLED_VIEWS_CACHE[key] = (tuple(poses), tuple(surfs), tuple(mins))
  return hit


def sampled_grasp_meta(spec, offset=0.01, part=None):
  """Per-candidate diagnostics of the sampled DB, parallel to part_grasp_poses: edge
  clearance, pad support, and the family a candidate came from. Empty when the sampler is
  off. It exists because a pinch's PAD contact points cannot be recovered from the grasp
  pose alone, so neither a caller nor a test can re-derive what the gates measured. With
  --grasp-sampler-scenes each entry also carries "set": free/pick/asm."""
  if _SAMPLED_GRASPS is None:
    return ()
  _sampled_grasp_views(spec, offset, part)                  # fills both caches
  return _SAMPLED_META_CACHE.get(_sampled_keys(spec, offset, part)[1], ())


def sampled_grasp_db_signature(assembly):
  """Content hash of the actual positional grasp DB used by a plan.

  Sampler flags and seeds are necessary but not sufficient provenance: changing generator code
  under the same flags used to silently reinterpret every saved grasp id. New plans persist this
  hash and replay refuses a mismatch. The per-part name is included because scene-aware DBs
  intentionally differ across geometrically identical instances.
  """
  import hashlib
  if _SAMPLED_GRASPS is None:
    return None
  h = hashlib.sha256()
  h.update(b"assembly-grasp-db\0")
  for part in assembly.objects:
    spec = assembly.part_shape(part)
    poses, surfaces, mins = _sampled_grasp_views(spec, 0.01, part)
    meta = sampled_grasp_meta(spec, 0.01, part)
    h.update(part.encode() + b"\0")
    h.update(np.asarray(poses, dtype="<f8").tobytes())
    h.update(repr(tuple(surfaces)).encode())
    h.update(np.asarray(mins, dtype="<f8").tobytes())
    # Only identity-shaping metadata: diagnostic floats are already represented by poses and
    # selection; roles/origin document why each positional member was retained.
    identity = tuple((m.get("origin"), m.get("family"), m.get("sets"),
                      tuple(sorted((k, tuple(v)) for k, v in
                                   (m.get("scene_by_kind") or {}).items()))) for m in meta)
    h.update(repr(identity).encode())
  return h.hexdigest()


def _horizontal_cylinder_scatter_with_top_suction(spec):
  """Horizontal cylinder pose whose free axial roll puts one sampled suction push through COM.

  The base rotation maps cylinder-local +z (its axis) to world -x and local +x to world +z.
  Rotating about local z changes no cylinder geometry, but lets a retained radial suction
  candidate's approach axis map to world -z. The table can then balance its sealing push
  without fictitious friction. With the face enumerator/no sampler, preserve the historical
  scatter quaternion exactly.
  """
  q_flat = np.array([np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0])
  if _SAMPLED_GRASPS is None or spec[0] != "cylinder":
    return q_flat
  poses = part_grasp_poses(spec)
  meta = sampled_grasp_meta(spec)
  candidates = []
  for i, (pose, md) in enumerate(zip(poses, meta)):
    approach = _quat_to_R(np.asarray(pose[3:], dtype=float))[:, 2]
    if md.get("family") != "suction" or abs(float(approach[2])) > 0.1:
      continue                                             # cap or tilted candidate, not barrel
    candidates.append((float(md.get("pad_support", 0.0)),
                       float(md.get("edge_clearance", 0.0)), -i, approach))
  if not candidates:
    return q_flat
  approach = max(candidates, key=lambda x: x[:3])[3]
  # q_flat maps local -x to world -z. Pre-roll the chosen radial approach onto local -x.
  theta = np.pi - np.arctan2(float(approach[1]), float(approach[0]))
  q_roll = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])
  return _quat_mul(q_flat, q_roll)


# EXTRA PLACE CELLS (experimental, OFF by default): classic `place` always lands a group at
# its reference part's ORIGINAL scatter anchor -- parts at the scatter extremes then get set
# down at the table edge where half the arms cannot reach them. With N > 0, places() ALSO
# offers up to N TRANSLATED anchors from the fixed cell list below (near the table centre,
# the most-reachable region). Cell placements are distinguished in the STATE via the place
# slot tag ("slot", ("cell", k), face) -- poses are NOT part of the state key -- and carry
# the landing anchor in the state's pose payload (the same mechanism declutter stowing uses).
# Infeasible cells (occupied / colliding) are rejected by the cheap layer like any placement.
# --ban-actions (SEARCH-scoped): action patterns the successor generator must not yield.
# The bandit outer loop's DISCRETE escalation: when motion planning stalls on a hold whose
# acquisition locked an unconnectable homotopy (chair-8: the handover receiving grasp), ban
# that acquisition and re-solve -- the search then picks a different grasp/pairing. Patterns
# are (kind, robot, part, arg) with None as wildcard per field. Reset per scene by
# replay.build_scene; NEVER applied to replay walks (a saved plan cannot contain a banned
# action anyway -- the search that produced it avoided them).
_BANNED_ACTIONS = ()


def set_banned_actions(patterns):
  """patterns: 'kind:robot:part:arg' strings ('*' = wildcard per field), optionally with a
  STATE CONTEXT suffix '@partA+partB+...': the ban then applies only when a subassembly
  involving one of the action's parts CONTAINS all the listed parts. This makes a ban
  exactly as strong as the measured counterexample -- the same action on a SMALLER
  subassembly (e.g. the seat before its legs are attached) stays legal, so the search can
  restructure the ORDER instead of losing the action outright."""
  global _BANNED_ACTIONS
  out = []
  for pat in patterns or ():
    pat = str(pat)
    ctx = None
    if "@" in pat:
      pat, ctx_s = pat.split("@", 1)
      ctx = frozenset(x for x in ctx_s.split("+") if x)
    f = [None if x in ("*", "") else x for x in pat.split(":")]
    out.append((tuple(f + [None] * (4 - len(f))), ctx))
  _BANNED_ACTIONS = tuple(out)


def _action_banned(a, state=None):
  if not _BANNED_ACTIONS:
    return False
  for pat, ctx in _BANNED_ACTIONS:
    if not all(p is None or str(a[k]) == p for k, p in enumerate(pat) if k < len(a)):
      continue
    if ctx is None:
      return True
    if state is None:
      continue                                             # contextual ban needs the state
    parts = [x for x in a[2:] if isinstance(x, str)]
    for g in state.subassemblies():
      if any(p in g for p in parts) and ctx <= set(g):
        return True
  return False


# --kind-role-forbid (SEARCH-scoped, like --ban-actions): a gripper KIND banned from an action
# KIND. Unlike a ban pattern (which names a robot), this names a CAPABILITY: 'vacuum:assemble'
# drops every assemble whose acting arm, inserted-part holder, target holder or brace contact
# is a vacuum -- the vacuum may pick and hand over but never touch a mate. Reset per scene by
# replay.build_scene; NEVER applied to replay walks (a plan saved under the forbid contains no
# forbidden action anyway). Needs the gripper_kind map, which the search threads only under
# --wrench (mixed EEs) -- without it the filter has no kind information and passes everything.
_KIND_FORBID = {}


def set_kind_forbid(patterns):
  """patterns: 'kind:action' strings ('*' wildcard per field)."""
  global _KIND_FORBID
  out = {}
  for pat in patterns or ():
    kind, act = str(pat).split(":", 1)
    if act == "*":
      continue                                          # no per-kind blanket yet
    out.setdefault(act, set()).add(kind)
  _KIND_FORBID = {k: frozenset(v) for k, v in out.items()}


def _kind_forbidden(a, state, gripper_kind):
  if not _KIND_FORBID or a[0] not in _KIND_FORBID or gripper_kind is None:
    return False
  kinds = _KIND_FORBID[a[0]]
  kind_of = gripper_kind.get if isinstance(gripper_kind, dict) else (lambda _r: gripper_kind)
  if kind_of(a[1]) in kinds:
    return True                                          # the acting arm itself
  touched = {p for p in a[2:] if isinstance(p, str)}
  for g in state.subassemblies():
    if not touched & set(g):
      continue
    for p in g:
      for c in state.supports.get(p, ()):
        if c[0] == "grasp" and kind_of(c[1]) in kinds:
          return True                                    # banned-arm hold or brace in the path
  return False


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


# --- mesh shape kind: ("mesh", {"file": path, "scale": s}[, quat]) --------------------------
# The file is any trimesh-loadable mesh (.stl/.obj/.ply), path relative to this repo (or
# absolute). v1 CONTRACT: the part COLLIDES AS ITS CONVEX HULL (rai setConvexMesh and the
# MuJoCo geom both get the hull) -- a non-convex part needs a decomposition into several
# parts/shape children first. The mesh is re-CENTERED on its AABB centre at load so the
# symmetric half-extent helpers (rest heights, standoffs, levers) stay exact; author mate
# poses against that centre. Grasps for mesh parts REQUIRE --grasp-sampler: the 6-face
# enumerator's ids are cube-face semantics and would silently mis-model a free-form part.
_MESH_CACHE = {}


def _mesh_key(prm):
  if "verts" in prm:
    # INLINE convex mesh (generated shapes, e.g. chamfer tips): key on the rounded
    # vertices so identical meshes share one asset. No file path involved.
    v = np.asarray(prm["verts"], dtype=float)
    return ("verts", tuple(np.round(v.ravel(), 12).tolist()))
  path = prm["file"]
  if not os.path.isabs(path):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
  return (os.path.normpath(path), float(prm.get("scale", 1.0)))


# --- compound shape kind: ("compound", [{"shape": spec, "pos": [x,y,z]}, ...]) -------------
# ONE rigid part built from SEVERAL sub-shapes (primitives or meshes; no nesting). Each child
# is an ordinary (kind, params[, quat]) spec plus an offset in the part frame. Realized the
# rai way: the part frame stays shapeless and rigid children `<part>__shape0..N-1` carry the
# shapes (the same pattern rotated single shapes already use -- address them via
# part_shape_frames / set_part_contact). Author the compound roughly CENTERED on the part
# origin: the half-extent helpers return a symmetric cover (exact per-child for rest
# heights). Grasps REQUIRE --grasp-sampler (same reason as mesh: the 6-face enumerator's ids
# are cube-face semantics); the sampler tessellates the union, and candidates buried between
# children are dropped by the DB filter's self-penetration gate.


def _compound_children(spec):
  """[(child_spec, offset)] of a compound spec. Children are (kind, params[, quat]) tuples
  -- any non-compound kind; nesting is refused (flatten by hand, it is the same shape)."""
  out = []
  for ch in spec[1]:
    sub = tuple(ch["shape"])
    if sub[0] == "compound":
      raise ValueError("no nested compounds -- flatten the children into one list")
    out.append((sub, np.asarray(ch.get("pos", (0.0, 0.0, 0.0)), dtype=float)))
  return out


def _spec_key(spec):
  """Canonical hashable identity of a shape spec, for memo keys. Meshes key on their
  (path, scale) -- NOT the kind string, two meshes are two surfaces; compounds key on the
  child keys plus offsets."""
  kind = spec[0]
  if kind == "compound":
    return (kind, tuple((_spec_key(sub), tuple(map(float, pos)))
                        for sub, pos in _compound_children(spec)))
  sq = _shape_quat(spec)
  qk = tuple(np.asarray(sq, dtype=float)) if sq is not None else None
  if kind == "mesh":
    return (kind, _mesh_key(spec[1]), qk)
  return (kind, tuple(spec[1]), qk)


_COMPOUND_MESH_CACHE = {}


def _compound_mesh(spec):
  """Tessellation of a compound part (child meshes transformed + concatenated), memoized.
  The sampler samples and raycasts THIS union. Vertices are not merged across children, so
  coplanar-patch flood fills stay within one child's surface -- conservative, never wrong."""
  key = _spec_key(spec)
  hit = _COMPOUND_MESH_CACHE.get(key)
  if hit is None:
    import trimesh
    parts = []
    for sub, pos in _compound_children(spec):
      m = _spec_mesh(sub).copy()
      sq = _shape_quat(sub)
      T = np.eye(4)
      if sq is not None:
        T[:3, :3] = _quat_to_R(sq)
      T[:3, 3] = pos
      m.apply_transform(T)
      parts.append(m)
    hit = _COMPOUND_MESH_CACHE[key] = trimesh.util.concatenate(parts)
  return hit


def _load_mesh(prm):
  """Memoized mesh load for a ("mesh", {...}) spec's params. Returns
  {"mesh": centered trimesh, "hull": convex-hull vertices (N,3), "half": AABB half
  extents} -- one object per mesh, shared by every consumer. Two sources: a FILE
  ({"file", ["scale"]}) or INLINE VERTICES ({"verts": (N,3)}) -- the latter for generated
  convex shapes (chamfer tips), authored anywhere in shape space and centred here like a
  file mesh so every downstream contract (half extents, ray exit, mass) is identical."""
  key = _mesh_key(prm)
  hit = _MESH_CACHE.get(key)
  if hit is None:
    import trimesh
    if key[0] == "verts":
      v = np.asarray(prm["verts"], dtype=float)
      lo, hi = v.min(axis=0), v.max(axis=0)
      m = trimesh.PointCloud(v - (lo + hi) / 2.0).convex_hull
      hull = m
    else:
      m = trimesh.load(key[0], force="mesh")
      if key[1] != 1.0:
        m.apply_scale(key[1])
      lo, hi = np.asarray(m.bounds, dtype=float)
      m.apply_translation(-(lo + hi) / 2.0)
      hull = m.convex_hull
    hit = _MESH_CACHE[key] = {
        "mesh": m, "hull": np.asarray(hull.vertices, dtype=float), "hull_mesh": hull,
        "half": (np.asarray(m.bounds[1], dtype=float)
                 - np.asarray(m.bounds[0], dtype=float)) / 2.0}
  return hit


def _chamfer_tip(spec, chamfer, end):
  """((body_spec, body_pos), (tip_spec, tip_pos)) -- split a cylinder/box peg at its
  INSERTION end (`end`: +1.0 = +z end, -1.0 = -z end) into a shortened full-section body
  and a convex-mesh FRUSTUM tip of height `chamfer`: the full cross-section at the inner
  end tapers linearly down to (full - chamfer) at the outer end -- a 45-degree chamfered
  lead-in. The total part length is preserved (body - chamfer, tip + chamfer), so
  part-frame extents, grasp standoffs and rest heights are unchanged. The tip is an
  inline-vertices mesh, realized as a convex hull in rai and a mesh geom in MuJoCo."""
  kind, p = spec[0], spec[1]
  assert kind in ("box", "cylinder"), f"chamfer tips need a box/cylinder peg, not {kind!r}"
  L = float(p[2]) if kind == "box" else float(p[0])
  c = float(chamfer)
  e = float(end)
  assert 0.0 < c < L / 2.0, f"chamfer {c:g} is impossible on a peg of length {L:g}"
  if kind == "box":
    s = float(p[0])
    assert abs(float(p[1]) - s) < 1e-12, "chamfered box pegs need a square cross-section"
    assert 2.0 * c < s, f"chamfer {c:g} consumes a square peg of side {s:g}"
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    base = np.column_stack([corners * (s / 2.0), np.full(4, -e * c / 2.0)])
    endc = np.column_stack([corners * ((s - 2.0 * c) / 2.0), np.full(4, e * c / 2.0)])
    body = (("box", [s, s, L - c, float(p[3])]), np.array([0.0, 0.0, -e * c / 2.0]))
  else:
    r = float(p[1])
    assert c < r, f"chamfer {c:g} consumes a peg of radius {r:g}"
    n = 12
    a = np.arange(n) * (2.0 * np.pi / n)
    base = np.column_stack([r * np.cos(a), r * np.sin(a), np.full(n, -e * c / 2.0)])
    endc = np.column_stack([(r - c) * np.cos(a), (r - c) * np.sin(a),
                            np.full(n, e * c / 2.0)])
    body = (("cylinder", [L - c, r]), np.array([0.0, 0.0, -e * c / 2.0]))
  tip = (("mesh", {"verts": np.vstack([base, endc])}),
         np.array([0.0, 0.0, e * (L / 2.0 - c / 2.0)]))
  return body, tip


def _hole_chamfer_children(h, wall, depth, z_rim, c, open_dir, len_outer, len_hole):
  """[(child_spec, pos)] -- the four 45-degree SOCKET-ENTRY chamfer funnels of one square
  pocket. Each wall's inner rim corner is rebuilt as a widening trapezoid prism (an inline
  convex mesh): at the rim plane the mouth half-width is h+c, converging to the original h
  at depth c -- a funnel that guides a slightly misaligned insertion down the facet instead
  of catching the rim. open_dir: +1.0 = the socket opens UP (rim at the top), -1.0 = opens
  DOWN. The CALLER shortens each wall box by c on the rim side; the prisms fill exactly that
  band and stay inside the wall footprint (c < wall and c < depth asserted). The +-x prisms
  run along y over `len_outer`, so the four funnels meet at the pocket corners."""
  assert 0.0 < c < wall and 0.0 < c < depth, \
      f"hole chamfer {c:g} exceeds wall {wall:g} or depth {depth:g}"
  s = float(open_dir)
  assert abs(s) == 1.0
  # cross-section in the (across, z) plane, LOCAL to the prism centre:
  #   A=(h, -s*c)  B=(h+wall, -s*c)  C=(h+wall, 0)  D=(h+c, 0)
  # A-D is the 45-degree guide facet (slope 1); B-C is the wall's outer face.
  a_c, z_c = h + wall / 2.0, -s * c / 2.0
  ac = np.array([h, h + wall, h + wall, h + c], dtype=float) - a_c
  zl = np.array([-s * c, -s * c, 0.0, 0.0], dtype=float) - z_c
  out = []
  for axis, along in (("x", len_outer), ("y", len_hole)):
    ends = np.repeat([-along / 2.0, along / 2.0], 4)
    if axis == "x":
      base = np.column_stack([np.tile(ac, 2), ends, np.tile(zl, 2)])
      across_col = 0
    else:
      base = np.column_stack([ends, np.tile(ac, 2), np.tile(zl, 2)])
      across_col = 1
    for sign in (+1.0, -1.0):
      # MIRROR the across axis for the -sign wall so the guide facet ALWAYS faces the
      # pocket (the -sign wall's inner face is at -h, not +h): without the mirror the
      # facet lands on the wall's OUTER face and the mouth chamfer faces outside.
      v = base.copy()
      v[:, across_col] *= sign
      pos = (np.array([sign * a_c, 0.0, z_rim + z_c]) if axis == "x"
             else np.array([0.0, sign * a_c, z_rim + z_c]))
      out.append((("mesh", {"verts": v}), pos))
  return out


def _rest_axis_spec(spec):
  """A compound whose children are ONE cylinder plus mesh tip(s) -- a chamfered round peg
  -- rests like the bare cylinder (neutral on its curved side). Returns the cylinder child,
  or None when the spec is not cylinder-like."""
  if spec[0] == "cylinder":
    return spec
  if spec[0] == "compound":
    cyl = [s for s, _ in _compound_children(spec) if s[0] == "cylinder"]
    other = [s for s, _ in _compound_children(spec) if s[0] != "cylinder"]
    if len(cyl) == 1 and other and all(s[0] == "mesh" for s in other):
      return cyl[0]
  return None


_PLAIN_BOX = False                                         # --plain-box: sharp ST.box instead of
                                                          # the sphere-swept ssBox (drops the
                                                          # corner_r inflation). A geometry
                                                          # A/B test knob -- see set_plain_box.


def set_plain_box(on=True):
  """Use rai's sharp ST.box for cube parts instead of ssBox (which sphere-SWEEPS the box by
  corner_r, inflating every face by r -> two 'flush' mated cubes overlap by 2r, and a gripper
  grazing a face sees the inflated surface). Plain box = the nominal [sx,sy,sz] with no
  inflation. Must be set BEFORE the scene/grasp-table is built (both solve and replay/export),
  so it is saved in the plan config and reapplied on reconstruction."""
  global _PLAIN_BOX
  _PLAIN_BOX = bool(on)


def _set_raw_shape(f, spec):
  kind, p = spec[0], spec[1]
  if kind == "box":
    if _PLAIN_BOX:
      f.setShape(ry.ST.box, list(p[:3]))                   # sharp box, no corner rounding
    else:
      f.setShape(ry.ST.ssBox, list(p))                     # [sx, sy, sz, corner_r]
  elif kind == "cylinder":
    f.setShape(ry.ST.cylinder, list(p))                    # [length, radius]
  elif kind == "sphere":
    f.setShape(ry.ST.sphere, list(p))                      # [radius]
  elif kind == "mesh":
    f.setConvexMesh(_load_mesh(p)["hull"])                 # v1 contract: collide as the hull
  elif kind == "compound":
    raise ValueError("compound is several shapes -- realize via add_part_frame "
                     "(one __shape{i} child per sub-shape), never on a single frame")
  else:
    raise ValueError(f"unknown shape kind {kind!r}")
  return f


def _set_frame_shape(f, spec):
  # only valid for unrotated specs (shape aligned with the part frame).
  assert _shape_quat(spec) is None, "rotated shape needs add_part_frame (child frame)"
  return _set_raw_shape(f, spec)


def add_part_frame(C, name, spec, contact=1):
  f = C.addFrame(name)
  if spec[0] == "compound":                                # one rigid child per sub-shape
    for i, (sub, pos) in enumerate(_compound_children(spec)):
      shp = C.addFrame(f"{name}__shape{i}", name)
      _set_raw_shape(shp, sub).setContact(contact)
      sq = _shape_quat(sub)
      if sq is not None:
        shp.setRelativeQuaternion(list(sq))
      shp.setRelativePosition([float(pos[0]), float(pos[1]), float(pos[2])])
    return f
  q = _shape_quat(spec)
  if q is None:
    return _set_raw_shape(f, spec).setContact(contact)
  shp = C.addFrame(f"{name}__shape", name)                 # rigid child carries the shape
  _set_raw_shape(shp, spec).setContact(contact)
  shp.setRelativeQuaternion(list(q))
  return f


def part_shape_frames(C, name):
  # ALL frames carrying `name`'s collision shape: the `__shape` child (rotated single
  # shape), the `__shape{i}` children (compound), or the part frame itself.
  names = C.getFrameNames()
  child = f"{name}__shape"
  if child in names:
    return [child]
  out, i = [], 0
  while f"{name}__shape{i}" in names:
    out.append(f"{name}__shape{i}")
    i += 1
  return out or [name]


def part_shape_frame(C, name):
  # the FIRST frame that carries a part's collision shape (complete for single-shape
  # parts). Anything that must cover a compound's every sub-shape uses part_shape_frames.
  return part_shape_frames(C, name)[0]


def set_part_contact(C, name, v):
  # contact toggle that works for plain, rotated-shape, and compound parts (all children).
  for f in part_shape_frames(C, name):
    C.getFrame(f).setContact(v)


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
  if kind == "mesh":
    return np.asarray(_load_mesh(p)["half"], dtype=float).copy()
  if kind == "compound":
    # symmetric COVER of the children (|offset| + rotated child half, per axis). Exact for
    # compounds authored symmetric about the part origin; an over-approximation otherwise.
    h = np.zeros(3)
    for sub, pos in _compound_children(spec):
      hs = _shape_half_extents(sub)
      sq = _shape_quat(sub)
      if sq is not None:
        hs = np.abs(_quat_to_R(sq)) @ hs
      h = np.maximum(h, np.abs(pos) + hs)
    return h
  raise ValueError(f"unknown shape kind {kind!r}")


def _shape_half_extent_along(spec, d_part):
  # half extent of the shape along direction `d_part` given in the PART frame (accounts
  # for the spec's shape rotation). Exact for axis-aligned directions and for cylinders
  # (support function), box-cover otherwise.
  if spec[0] == "compound":
    d = np.asarray(d_part, dtype=float)
    return max(abs(float(d @ pos)) + _shape_half_extent_along(sub, d)
               for sub, pos in _compound_children(spec))
  q = _shape_quat(spec)
  d = d_part if q is None else _quat_to_R(q).T @ np.asarray(d_part, dtype=float)
  if spec[0] == "cylinder":
    # EXACT support along d in the SHAPE frame: |d_z|*l/2 + sqrt(dx^2+dy^2)*r. The
    # bounding-box form |d|@[r,r,l/2] is right only at cardinal axes and WRONG in between
    # (a flat wheel got l/2 = 6mm instead of r = 24mm and started inside the sim table).
    l, rad = spec[1][0], spec[1][1]
    return float(np.abs(d[2]) * l / 2.0 + np.hypot(d[0], d[1]) * rad)
  return float(np.abs(d) @ _shape_half_extents(spec))


def _shape_half_z(spec, R):
  # half extent along WORLD z when the PART frame is rotated by R (the shape's own
  # rotation inside the part frame is composed in). Exact for our axis-aligned placement
  # rotations, for spheres, and for cylinders (support function); a safe over-approximation
  # for general rotations of meshes.
  if spec[0] == "compound":
    # per child: how far its centre drops below the part origin, plus its own half-z.
    # EXACT per-child bound, so an asymmetric compound still rests at the right height.
    return max(float(-(R @ pos)[2]) + _shape_half_z(sub, R)
               for sub, pos in _compound_children(spec))
  if spec[0] == "sphere":
    return _shape_half_extents(spec)[2]
  q = _shape_quat(spec)
  R_total = R if q is None else R @ _quat_to_R(q)
  if spec[0] == "cylinder":
    # EXACT support along world z: |u_z|*l/2 + sqrt(1-u_z^2)*r with u = the world-z
    # direction in the shape frame. The old bounding-box form |u|@[r,r,l/2] undercounts
    # whenever the axis is off the world axes (measured: a scattered flat car wheel got
    # l/2 = 6mm instead of r = 24mm, and the sim started it 18mm inside the table).
    l, rad = spec[1][0], spec[1][1]
    u = R_total[2, :]
    return float(np.abs(u[2]) * l / 2.0 + np.hypot(u[0], u[1]) * rad)
  return float(np.abs(R_total[2, :]) @ _shape_half_extents(spec))


def part_grasp_poses(spec, rolls=None, offset=0.01, part=None):
  """All candidate grasp poses of a part shape, in the part's OWN frame -- the per-part
  grasp DB the labeled tables index into (grasp_id = position in this list). Enumerator path:
  the six axis-direction grasps at the FACE CENTER (approach along -direction, standoff =
  the shape's half-extent along that direction + offset) x `rolls` spins about the approach
  axis. For a cube this reproduces cube_grasp_pose exactly (same ids, same poses); for a
  cuboid the standoff per face follows its half-extents; for a cylinder the +-z grasps sit
  on the caps and the side grasps on the curved surface (vacuum allowed there, same force
  -- per current model decision); spheres get the same six directions at radius standoff.
  Rotated/off-center grasp variants are a later, purely additive extension."""
  if _SAMPLED_GRASPS is not None:                          # --grasp-sampler second path
    return _sampled_grasp_views(spec, offset, part)[0]
  if spec[0] in ("mesh", "compound"):
    raise ValueError("mesh/compound parts need --grasp-sampler: the 6-face enumerator's "
                     "grasp ids are cube-face semantics and would silently mis-model them")
  if rolls is None:
    rolls = _ROLLS_PER_FACE
  sq = _shape_quat(spec)
  key = ("DB", spec[0], tuple(spec[1]),
         tuple(sq) if sq is not None else None, rolls, offset, _DENSE_GRASPS,
         _GRASP_TRANSLATIONS, _GRASP_PAD_HALF)
  hit = _GRASP_POSE_CACHE.get(key)
  if hit is not None:
    return hit
  h = _shape_half_extents(spec)
  poses = []
  pose_n = []                                              # face normal behind each pose
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
      pose_n.append(np.asarray(normal, dtype=float))
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
        pose_n.append(pose_n[len(poses) - 1 - len(_CUBE_FACE_NORMALS) * rolls]
                      if len(pose_n) >= len(_CUBE_FACE_NORMALS) * rolls else np.zeros(3))
  if _GRASP_TRANSLATIONS:
    # TRANSLATED variants of every pose so far, appended base-order so existing grasp ids keep
    # their meaning. Same orientation and same pinch; only the tool SLIDES in its open plane.
    pat = _translation_pattern(_GRASP_TRANSLATIONS)
    _src = list(poses)          # SNAPSHOT: translating the growing list compounds the slides
    _srcn = list(pose_n)        # (12 -> 24 -> 48 poses, and diagonal double-translations)
    for axis, frac in pat:
      for base, nrm in zip(_src, _srcn):
        R = _quat_to_R(np.asarray(base[3:]))
        d = R @ np.asarray(axis, dtype=float)              # the slide direction, in PART frame
        # HOW FAR THE PAD MAY SLIDE. The pad must stay inside the contacted face, so the bound
        # is the distance from the face centre to the boundary of the face INSET by the pad
        # half-size, measured along d: min_i (h_i - pad) / |d_i|.
        # NOT the box's support radius |d|.h -- that is the distance to the CORNER, which for a
        # 45deg tilted grasp is 1.41x the axis-aligned value, so the diagonal variants slid 22mm
        # while the perpendicular ones moved 10mm and looked untranslated next to them.
        ad = np.abs(d)
        lim = [(h[k] - _GRASP_PAD_HALF) / ad[k] for k in range(3) if ad[k] > 1e-9]
        travel = max(0.0, min(lim)) if lim else 0.0
        # A SLIDE ONLY STAYS ON THE SURFACE IF THE SURFACE IS FLAT ALONG IT.
        #   box      -- every face is flat: both in-plane directions are fine.
        #   cylinder -- a SIDE grasp sits on the barrel, which is flat only ALONG THE AXIS, so
        #               only the axis-parallel slide is admissible (a tangential one would walk
        #               the pads off the curve -- that wants a ROTATION about the axis, which is
        #               a different variant). This axis slide is also the useful one: it takes
        #               the cylinder higher up so the fingers clear the table. A CAP grasp is a
        #               flat disc, so any in-plane direction is fine, bounded by the radius.
        #   sphere   -- curved everywhere: no translation is admissible.
        # An inadmissible direction yields travel 0, i.e. the variant collapses onto the centred
        # grasp. Deliberate: it keeps the count uniform so grasp ids stay aligned with
        # part_grasp_surfaces / part_grasp_face_min_dims.
        kind = spec[0]
        if kind == "sphere":
          travel = 0.0
        elif kind == "cylinder":
          on_cap = abs(float(nrm[2])) > 0.5
          if on_cap:
            travel = max(0.0, min(float(h[0]), float(h[1])) - _GRASP_PAD_HALF)
          else:
            axis_par = abs(float(d[2]))                    # part-frame z IS the cylinder axis
            travel = (max(0.0, float(h[2]) - _GRASP_PAD_HALF) if axis_par > 0.9 else 0.0)
        moved = np.concatenate([base[:3] + frac * travel * d, base[3:]])
        moved.setflags(write=False)
        poses.append(moved)
  hit = _GRASP_POSE_CACHE[key] = tuple(poses)
  return hit


def grasp_pose(assembly, part, grasp_id, offset=0.01):
  # grasp `grasp_id` of `part`, in the part's own frame (the shape-aware replacement for
  # raw cube_grasp_pose(gid, cube_size) -- identical values for cube parts).
  return part_grasp_poses(assembly.part_shape(part), offset=offset, part=part)[grasp_id]


def part_grasp_surfaces(spec, rolls=None, part=None):
  # surface TYPE contacted by each grasp in part_grasp_poses (parallel list): "flat" or
  # "curved". Box faces are flat; a cylinder's +-z caps are flat, its side is curved; a
  # sphere is curved everywhere. Used to restrict the VACUUM to flat surfaces (a suction
  # cup cannot seal on strong curvature) -- the two-finger keeps curved grasps.
  if _SAMPLED_GRASPS is not None:                          # --grasp-sampler second path
    return list(_sampled_grasp_views(spec, 0.01, part)[1])
  if spec[0] in ("mesh", "compound"):
    raise ValueError("mesh/compound parts need --grasp-sampler (see part_grasp_poses)")
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
  if _GRASP_TRANSLATIONS:
    out = out * (1 + _GRASP_TRANSLATIONS)                  # a slid grasp contacts the SAME face
  return out


def part_grasp_face_min_dims(spec, rolls=None, part=None):
  # smallest extent of the FLAT face each grasp contacts (parallel to part_grasp_poses /
  # part_grasp_surfaces; inf for curved surfaces -- the flat gate never fires there, the
  # curved gate already handles them). Used for the VACUUM SEAL gate: a suction cup cannot
  # seal on a face smaller than its own diameter (wrench.PARAMS['vacuum_seal_diameter']).
  if _SAMPLED_GRASPS is not None:                          # --grasp-sampler second path
    return list(_sampled_grasp_views(spec, 0.01, part)[2])
  if spec[0] in ("mesh", "compound"):
    raise ValueError("mesh/compound parts need --grasp-sampler (see part_grasp_poses)")
  if rolls is None:
    rolls = _ROLLS_PER_FACE
  kind, params = spec[0], spec[1]
  out = []
  for normal in _CUBE_FACE_NORMALS:
    if kind == "box":
      ext = [params[0], params[1], params[2]]
      i = int(np.argmax(np.abs(normal)))
      dim = min(e for j, e in enumerate(ext) if j != i)    # the face's two in-plane extents
    elif kind == "cylinder":
      dim = 2.0 * params[1] if abs(normal[2]) > 0.5 else float("inf")  # cap disc diameter
    else:                                                  # sphere: curved everywhere
      dim = float("inf")
    out.extend([dim] * rolls)
  if _DENSE_GRASPS:
    out = out * 3
  if _GRASP_TRANSLATIONS:
    out = out * (1 + _GRASP_TRANSLATIONS)                  # a slid grasp contacts the SAME face
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


class GraspTable(dict):
  """{subassembly_key: [(part, grasp_id, pose), ...]} plus, for MIXED-end-effector scenes,
  per-grasp KIND masks: kinds_of[(S_key, part, gid)] = frozenset of gripper kinds whose EE
  can stand at that grasp (collision-checked per kind), and robot_kinds = {robot: kind}.
  Uniform scenes use a plain dict (masks None) -- zero behavioural change. `usable` is the
  one gate the successor generators call."""

  kinds_of = None                                          # {(S_key, part, gid): frozenset}
  robot_kinds = None                                       # {robot: "vacuum"|"two_finger"}

  def usable(self, S_key, part, gid, robot):
    if self.kinds_of is None or self.robot_kinds is None:
      return True                                          # uniform: everything for everyone
    kind = self.robot_kinds.get(robot)
    if kind is None:
      return True
    return kind in self.kinds_of.get((S_key, part, gid), ())


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
  from env import maybe_tighten_fingers          # the grasp DB must be filtered against the
  maybe_tighten_fingers(base_scene, "ee_")       # SAME proxy the search and the sim collide with
  _, base_rel_tool, q_correction = _gripper_rest_transforms(base_scene, "ee_")

  def grasp_is_collision_free(grasp_pose, grasped_obj, conventional=False):
    # move the pre-loaded EE so its tool center hits the grasp, then check the EE against
    # every part except the one being grasped (touching it is allowed). ONLY EE-involving
    # pairs may reject: the subassembly's INTERNAL part-part contacts are the mates
    # themselves (flush-seated parts touch at d=-0.0) and are irrelevant to whether the
    # gripper fits -- a binary getCollisionFree() here wrongly killed EVERY grasp on any
    # flush-mated subassembly (found via the car: wheel caps seat flush on the body).
    #
    # SAMPLED DBs additionally gate the GRASPED part itself: an off-center sampled grasp on
    # a FLAT face routinely parks an OPEN finger capsule ON the surface (measured 5-10mm on
    # cubes and the chamfered mesh; face-center enumerator grasps are clean by construction,
    # <=0.5mm capsule graze). Such a candidate survives every assembly-side gate (the
    # witness gate exempts the acting grasp pair) and then hard-kills motion planning:
    # mr_bench exempts nothing, so the pick mode's transition never samples. 1mm separates
    # the fat-capsule grazing band from a finger through the surface. Enumerator path
    # unchanged (byte-identical DBs for every existing scene).
    C_col = ry.Config()
    C_col.addConfigurationCopy(base_scene)
    base_world = _ee_root_pose(grasp_pose, base_rel_tool, q_correction)
    root = C_col.getFrame("ee_ur_base")
    root.setPosition(base_world[:3])
    root.setQuaternion(base_world[3:])
    # The sampler's conventional rows pass through the exact enumerator filter too. Applying
    # the sampled own-part capsule tolerance to those same poses changed which canonical ids
    # survived, perturbing otherwise-unrelated searches before an added sample was even used.
    if _SAMPLED_GRASPS is None or conventional:
      set_part_contact(C_col, grasped_obj, 0)
      return not any((a.startswith("ee_") or b.startswith("ee_")) and d < -1e-9
                     for a, b, d in C_col.getCollisions())
    _own = lambda n: n == grasped_obj or n.startswith(grasped_obj + "__shape")
    # the OWN-part band absorbs the two-finger's fat capsule proxy (clean face-center
    # grasps graze <=0.5mm). A VACUUM has no proxy -- its cup either stands clean at the
    # 1cm offset (~0) or overhangs an adjacent feature (measured 0.26mm: the cup lip on a
    # compound's boss wall, which motion planning then rejects at 1e-4) -- tight band.
    _own_tol = 1e-4 if "vacuum" in os.path.basename(gripper_path) else 1e-3
    for a, b, d in C_col.getCollisions():
      if not (a.startswith("ee_") or b.startswith("ee_")):
        continue
      own = _own(a) or _own(b)
      if d < (-_own_tol if own else -1e-9):
        return False
    return True

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
    surfaces = part_grasp_surfaces(assembly.part_shape(obj), part=obj)
    face_dims = part_grasp_face_min_dims(assembly.part_shape(obj), part=obj)
    sample_meta = sampled_grasp_meta(assembly.part_shape(obj), part=obj)
    seal = wrench.PARAMS.get("vacuum_seal_diameter", 0.0)
    for grasp_id, local in enumerate(part_grasp_poses(assembly.part_shape(obj),
                                                      offset=offset, part=obj)):
      if (_SAMPLED_CANONICAL_ONLY and sample_meta
          and assembly.part_shape(obj)[0] not in ("mesh", "compound")
          and sample_meta[grasp_id].get("origin") != "canonical"):
        continue                                          # staged search: samples reopen later
      if vacuum and (surfaces[grasp_id] == "curved" or face_dims[grasp_id] < seal):
        continue                                          # no seal on curvature or on a face
                                                          # smaller than the cup (chair caps)
      grasp = np.concatenate([obj_in_ref + local[:3], local[3:]])

      conventional = bool(sample_meta and sample_meta[grasp_id].get("origin") == "canonical")
      if grasp_is_collision_free(grasp, obj, conventional=conventional):
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
    # connected components of parts joined by mate edges (union-find over "part" parents).
    # MEMOIZED with the same lazy-cache invariant as key(): a State is never structurally
    # edited after it is first queried (successors finish their edits before yielding). The
    # fill's successor sweep calls this per assemble candidate and per h evaluation --
    # measured 437k calls / 1.0s on the 2-part stool -- so the memo is a real constant win.
    cached = getattr(self, "_subs", None)
    if cached is not None:
      return cached
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
    self._subs = [frozenset(s) for s in comps.values()]
    return self._subs

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


def _copy_edit_support(state, *, copy_poses=False):
  # helper: a fresh State copy whose dicts can be edited in place.
  # `poses` is a non-identity payload (State.key() ignores it) and is
  # read-only in the standard milestone search (place_cells=0,
  # declutter/stow off). Copying it per successor is pure overhead: the
  # scatter anchors are identical for every WIP state, so the dict and
  # its ndarray values can be shared copy-on-write. Only the narrow
  # stow/declutter edge mutates a single entry and must request a
  # private copy (`copy_poses=True`); the two `places` helpers already
  # do `ns.poses = dict(ns.poses)` before writing, so sharing there is
  # safe. This cuts ~25% of the successor-state allocation and avoids
  # a dict copy per edge (20k states * 7 poses ~= 20 MB per fill).
  if copy_poses:
    return State(state.parent, state.grasp, state.place, state.supports, state.poses)
  # fast path: bypass State.__init__'s `dict(poses)` for the common case
  ns = State.__new__(State)
  ns.parent = dict(state.parent)
  ns.grasp = dict(state.grasp)
  ns.place = dict(state.place)
  # `supports` values are already frozensets
  ns.supports = dict(state.supports)
  ns.poses = state.poses  # share reference — CoW; caller must copy before mutating
  return ns


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


def _grasp_usable(grasp_table, key, gpart, gid, robot):
  # MIXED-EE gate: may `robot`'s gripper stand at this table grasp? Plain-dict tables /
  # on-the-fly grasps (uniform scenes) always pass.
  return not isinstance(grasp_table, GraspTable) or grasp_table.usable(key, gpart, gid, robot)


def _vacuum_engagement_stable(state, assembly, group, robot, gpart, grasp_id,
                              gripper_kind):
  """Can the OLD supports take a vacuum cup's transient sealing push?

  The approach axis is grasp-local +z and points from the gripper into the part.  Crucially,
  this checks ``state`` before the new grasp contact exists: letting the new vacuum contact
  balance its own application force would make the gate vacuous.  Once attached, the cup's
  pressure/contact preload is internal; ordinary grasp_capability prices only its net wrench.
  """
  global _VEG_CACHE
  if gripper_kind is None:
    return True                                             # --no-wrench baseline
  kind_of = gripper_kind.get if isinstance(gripper_kind, dict) \
      else (lambda _r: gripper_kind)
  if kind_of(robot) != "vacuum":
    return True
  force = float(wrench.PARAMS.get("vacuum_engage_force", 0.0))
  if force <= 0.0:
    return True
  # SCOPED STATIC FRICTION (--vacuum-engage-friction): the transient sealing push happens
  # while the part is only supported by the table / old holders. A real table resists the
  # small lateral component of a near-vertical approach through STATIC friction, which the
  # global frictionless unilateral model cannot (it rejects ANY lateral load). This gives the
  # TABLE a friction cone for THIS check only -- every other gate keeps the global table
  # model, and a held part's gripper contacts are untouched.
  mu_engage = float(wrench.PARAMS.get("vacuum_engage_friction", 0.0) or 0.0)
  # SOLVE-SCOPED VERDICT CACHE (same pattern/validity as the insertion gates, keyed on the
  # STATE IDENTITY so a state re-generated in another fill round hits too -- object-level
  # caches miss those). The verdict is a deterministic function of the state under fixed
  # PARAMS (the key <-> pose bijection covers the pose payload). Cleared per solve/scene via
  # reset_wrench_cache, so no cross-scene leakage.
  key = (state.key(), _subasm_key(group), robot, gpart, grasp_id, _wrench_params_sig())
  hit = _VEG_CACHE.get(key)
  if hit is not None:
    return hit
  poses = _state_part_poses(state, assembly, lift=0.0)
  point, approach, _t1, _t2 = _grasp_frame(
      assembly, gpart, poses[gpart], grasp_id)
  res = _group_stable(
      state, assembly, group, gripper_kind,
      applied=[(force * approach, point)],
      table_override=("friction", mu_engage) if mu_engage > 0.0 else None)
  _VEG_CACHE[key] = res
  return res


def picks(state, assembly, robots, grasp_table=None, canonical=False, gripper_kind=None):
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
  free = _canonical_free(free, grasp_table)               # --dominance: one twin per kind
  for group in state.subassemblies():
    if not _group_is_on_table(state, group) or _group_holders(state, group):
      continue
    key = _subasm_key(group)
    labeled = grasp_table[key] if grasp_table is not None else \
        compute_cube_grasps(assembly, key, return_labeled=True)
    labeled = _useful_first(labeled)                      # --useful-grasps: mate-viable first
    for r in free:
      for gpart, grasp_id, _pose in labeled:              # grasp ANY part of the group
        if not _grasp_usable(grasp_table, key, gpart, grasp_id, r):
          continue                                        # mixed EEs: not this robot's gripper
        if not _vacuum_engagement_stable(
            state, assembly, group, r, gpart, grasp_id, gripper_kind):
          continue                                        # existing table/holders cannot take seal push
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
        if not _staged_press_stable(
            state, assembly, onto, gripper_kind,
            _insertion_load_variants(state, assembly, p, onto, d,
                                     scale=_compliance_factor(state, assembly, target,
                                                              gripper_kind))):
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
    try:
      cache[key] = _on_table_mateable(assembly, part, face=face)
    except KeyError:
      # --stable-rest-poses: placement ids are PER GROUP, and this fact is asked with an id
      # taken from a DIFFERENT group's table (the heuristic reasons about an abstract group
      # and then queries the mate closure). Under the legacy table every group shared the
      # same six ids, so the question always made sense; now it sometimes does not.
      # "Unknown" must resolve to True: this fact only ever PRUNES candidates, so answering
      # True keeps the search complete, while answering False could remove a real plan.
      cache[key] = True
  return cache[key]


def add_grasps(state, assembly, robots, grasp_table=None, gripper_kind=None):
  # a free robot adds a SECOND grasp (a brace) to an already-HELD subassembly -- the unfused
  # primitive behind HANDOVERS (add_grasp; release) and bracing. The new grasp is a SUPPORT
  # contact only; the kinematic `parent` is unchanged (the group stays rooted at its current
  # holder -- this is the whole point of parent(single)/supports(set)). A later release of the
  # original holder then re-roots the tree onto this grasp (see `releases`).
  free = _canonical_free(state.free_robots(robots), grasp_table)  # --dominance
  if not free:
    return
  for group in state.subassemblies():
    if not _group_holders(state, group):
      # --brace-resting: a TABLE-RESTING group may also be braced (steadied); without the
      # flag, only an already-held group (the historical model).
      if not (_BRACE_RESTING and _group_is_on_table(state, group)):
        continue
    key = _subasm_key(group)
    labeled = grasp_table[key] if grasp_table is not None else \
        compute_cube_grasps(assembly, key, return_labeled=True)
    taken = {(c[2], c[3]) for p in group for c in state.supports[p] if c[0] == "grasp"}
    for r in free:
      for gpart, grasp_id, _pose in labeled:              # grasp ANY free face of the group
        if (gpart, grasp_id) in taken:
          continue                                        # that grasp is already occupied
        if not _grasp_usable(grasp_table, key, gpart, grasp_id, r):
          continue                                        # mixed EEs: not this robot's gripper
        if not _vacuum_engagement_stable(
            state, assembly, group, r, gpart, grasp_id, gripper_kind):
          continue                                        # check before adding the new support
        ns = _copy_edit_support(state)
        ns.supports[gpart] = ns.supports[gpart] | {("grasp", r, gpart, grasp_id)}
        yield ("add_grasp", r, gpart, grasp_id), ns


def fused_handovers(state, assembly, robots, grasp_table=None, gripper_kind=None):
  # FUSED handover: a free robot B takes a singly-held group from its holder A in ONE step
  # (B grasps a free face, A releases, the tree re-roots to B). Net holder count is UNCHANGED,
  # so unlike the unfused add_grasp;release it introduces no higher-h two-holder intermediate
  # -- it is the heuristically neutral, air analogue of the place;pick regrasp. (Unfused
  # add_grasp stays for genuine bracing, where you keep both holders.)
  free = _canonical_free(state.free_robots(robots), grasp_table)  # --dominance
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
    labeled = _useful_first(labeled)                      # --useful-grasps: mate-viable first
    taken = {(c[2], c[3]) for p in group for c in state.supports[p] if c[0] == "grasp"}
    for rB in free:
      for gpart, grasp_id, _pose in labeled:                # B grabs any free face...
        if (gpart, grasp_id) in taken:
          continue
        if not _grasp_usable(grasp_table, key, gpart, grasp_id, rB):
          continue                                          # mixed EEs: not B's gripper
        if not _vacuum_engagement_stable(
            state, assembly, group, rB, gpart, grasp_id, gripper_kind):
          continue                                          # giver must withstand B's seal push
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
          # still holds the group -> re-root the tree onto that remaining holder.  Contacts are
          # frozensets, so the tie-break must cover BOTH the grasped part and the robot.  Chair's
          # final co-grasp has a1 and a3 on the same obj_0; breaking the tie only on obj_0 made
          # the new root depend on PYTHONHASHSEED, and a saved ``place a1`` then disappeared
          # from the successor set whenever a3 happened to win the set iteration order.
          remaining = [(cc[2], cc[1], cc[3]) for pp in group
                       for cc in ns.supports[pp] if cc[0] == "grasp"]
          if not remaining:
            continue                  # no other holder to take it (should not happen if supported)
          robot_rank = {rr: i for i, rr in enumerate(robots)}
          q, r2, gid2 = min(
              remaining,
              key=lambda t: (_part_index(t[0]), robot_rank.get(t[1], len(robot_rank)), t[1]))
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
      if not _grasp_usable(grasp_table, key, p, gid, r):
        continue                                           # mixed EEs: not r's gripper
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


# BRACE RESTING (--brace-resting): allow add_grasp on a TABLE-RESTING group, not only an
# already-held one. Physically routine (a hand steadying a resting assembly against a
# press); without it, "resting + braced" exists ONLY as the leftover of a mate (the kept
# inserter grasp) and is IRREVERSIBLE once released -- the artificial asymmetry that made
# single milestone fills wrench-infeasible where windows (which never release) succeed,
# and the abstract facts (which happily craft resting+braced states) systematically
# optimistic. With the flag, the concrete model matches both physics and the abstraction.
_BRACE_RESTING = False


def set_brace_resting(on=True):
  global _BRACE_RESTING
  _BRACE_RESTING = bool(on)


# USEFUL GRASPS (--useful-grasps, spec 6.2 reduced): _USEFUL_GRASPS = {part: frozenset of
# mate-viable grasp ids} (collision-at-op x per-kind press; computed by
# solver.compute_useful_grasps after the table build when the flag is on). Consumers:
#   * picks()/fused_handovers() ORDER candidates U-first -- the transit gradient (grasps
#     that can seat the mate are tried before ones that force a regrasp);
#   * the milestone fill h prices a held non-U grasp at +1 (the forced handover/regrasp
#     before the mate -- admissible).
# ORDERING only (plus the admissible h term): nothing is removed, completeness untouched.
_USEFUL_GRASPS = None
_USEFUL_FLAG = False                                       # --useful-grasps requested; the
                                                           # solvers populate after table build


def set_useful_grasps_enabled(on=True):
  global _USEFUL_FLAG
  _USEFUL_FLAG = bool(on)


def set_useful_grasps(mapping):
  global _USEFUL_GRASPS
  _USEFUL_GRASPS = mapping


def _useful_first(labeled, part_of=None):
  # stable U-first ordering of [(part, gid, pose), ...] rows; identity when the flag is off.
  if _USEFUL_GRASPS is None:
    return labeled
  U = _USEFUL_GRASPS
  return sorted(labeled, key=lambda t: 0 if t[1] in U.get(t[0], ()) else 1)


# DOMINANCE (--dominance, spec 8.3 "same node modulo free robots"): SAME-KIND FREE-ARM
# symmetry breaking -- when several currently-FREE arms of the same gripper kind could act,
# only the LOWEST-INDEX one does. Any plan using a higher-index twin permutes to the
# canonical one, so completeness holds EXACTLY when same-kind arms are interchangeable:
# true for FLOATING EEs (identical geometry, no base); a REAL fixed-base arm at a different
# table side is NOT a twin -- there the flag over-prunes (documented; the flag prints a
# warning on real-arm scenes). Kills the x2-x3 idle-twin branching of the transit lattice.
_DOMINANCE = False


def set_dominance(on=True):
  global _DOMINANCE
  _DOMINANCE = bool(on)


def _canonical_free(free, grasp_table):
  # reduce the free-arm choice set to one representative per gripper KIND (lowest index).
  # Uniform scenes (no robot_kinds): all arms are one kind -> single representative.
  if not _DOMINANCE or len(free) <= 1:
    return free
  kinds = getattr(grasp_table, "robot_kinds", None) or {}
  seen, out = set(), []
  for r in free:                                          # free_robots is index-sorted
    k = kinds.get(r, "_uniform")
    if k not in seen:
      seen.add(k)
      out.append(r)
  return out


# ALLOW CURVED GRASPS (--allow-curved-grasps): let a VACUUM keep curved-surface grasps
# (normally dropped, since a suction cup cannot seal on a cylinder/sphere side). An idealised
# test knob -- e.g. to check whether curved-side grasps make thin cylinder legs graspable.
# Read in compute_cube_grasps; must be set BEFORE the grasp table is built.
_ALLOW_CURVED_GRASPS = False


def set_allow_curved_grasps(on=True):
  global _ALLOW_CURVED_GRASPS
  _ALLOW_CURVED_GRASPS = bool(on)


def _slot_holds_alone(assembly, part, face):
  # NOTE (--stable-rest-poses): `face` must index THIS part's own placement table. A foreign
  # id raises from placement_quat rather than silently scoring a wrong rotation; the caller
  # decides what to do, and both callers treat it as "no information".
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
      R = _quat_to_R(np.asarray(placement_quat(assembly, (part,), part, face), dtype=float))
      shp = assembly.part_shape(part)
      kind, params = shp[0], shp[1]
      if kind == "box":
        half = float(np.dot(np.abs(d), np.asarray(params[:3], dtype=float))) / 2.0
      elif kind == "cylinder":
        half = abs(d[2]) * params[0] / 2.0 + (1 - abs(d[2])) * params[1]
      elif kind in ("mesh", "compound"):
        half = _shape_half_extent_along(shp, d)            # AABB extent along the press
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
  kind_of = gripper_kind.get if isinstance(gripper_kind, dict) else (lambda _r: gripper_kind)
  for c in sorted(state.supports.get(q, ())):
    if c[0] == "grasp":
      gp, n, t1, t2 = _grasp_frame(assembly, c[2], poses[c[2]], c[3])
      surf = _grasp_surface(assembly.part_shape(c[2]), c[3]) if derate else "flat"
      caps.append(wrench.grasp_capability(gp - pos, n, t1, t2, kind_of(c[1]), surface=surf))
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
    yield from picks(state, assembly, robots, grasp_table=grasp_table, canonical=canonical,
                     gripper_kind=gripper_kind)
    yield from assembles(state, assembly, robots, gripper_kind=gripper_kind)  # incl. module joins
    if handovers:
      yield from fused_handovers(state, assembly, robots, grasp_table=grasp_table,
                                 gripper_kind=gripper_kind)
    if bracing:
      yield from add_grasps(state, assembly, robots, grasp_table=grasp_table,
                            gripper_kind=gripper_kind)
    if _INHAND_REORIENT:
      yield from inhand_reorients(state, assembly, robots, grasp_table=grasp_table)
    yield from releases(state, assembly, robots, gripper_kind=gripper_kind)
    if allow_place:
      yield from places(state, assembly, robots)
    # deferred (unfused): set_downs

  def _filtered():
    it = _gen()
    if _BANNED_ACTIONS:
      it = (x for x in it if not _action_banned(x[0], state))
    if _KIND_FORBID:
      it = (x for x in it if not _kind_forbidden(x[0], state, gripper_kind))
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
  R = _quat_to_R(placement_quat(assembly, group, ref, face))
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
    # placement_quat: rest poses come from the SHAPE (a rotated singleton's face rests
    # flush), not the bookkeeping part frame -- see its doc.
    q = _quat_mul(_anchor_yaw_quat(anchor_pose[3:]),
                  placement_quat(assembly, group, ref, face))
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
  if kind == "mesh":
    return [c + R @ v for v in _load_mesh(p)["hull"]]     # hull vertices = candidate contacts
  if kind == "compound":                                   # union over the children, each at
    pts = []                                               # its own offset (child quat handled
    for sub, off in _compound_children(spec):              # by the recursive call)
      pts += _part_support_points_world(np.concatenate([c + R @ off, pose[3:]]), sub)
    return pts
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
  if len(low) >= 3:
    xy = low[:, :2] - low[:, :2].mean(axis=0)
    if float(np.linalg.svd(xy, compute_uv=False)[1]) > span_tol:
      return True                                            # 2-D support polygon: stable
  # NEUTRAL (rolling) rest -- change-2-lite: a lone CYLINDER on its curved side (line
  # contact) or a SPHERE (point) does not TIP: as it rolls, the contact follows the curved
  # surface and the COM stays above it (neutral equilibrium) -- it stays put when placed,
  # which is what a rest pose means here. Real scatters rest cylinders exactly this way
  # (the chair's legs/posts lie flat), so the table must offer it. A BOX on an edge/corner
  # genuinely tips and is still rejected (this branch is curved-singletons only). The
  # cylinder must be lying on the SIDE (axis horizontal) with its COM over the contact
  # line's span; a tilted cylinder rests on a rim point and tips -> rejected.
  if len(group) == 1 and len(low) >= 1:
    spec = assembly.part_shape(ref)
    if spec[0] == "sphere":
      return True
    # A cylinder-LIKE spec also covers a chamfered round peg (compound of one cylinder
    # body + mesh tip): it still rolls neutrally on its curved side, so the lying-side
    # placement stays available to the chamfered kinds.
    axis_spec = _rest_axis_spec(spec)
    if axis_spec is not None:
      pose = np.asarray(poses[ref], dtype=float)
      R = _quat_to_R(pose[3:])
      sq = _shape_quat(axis_spec)
      if sq is not None:
        R = R @ _quat_to_R(sq)
      axis = R[:, 2]                                         # cylinder axis, world frame
      if abs(axis[2]) < 0.05:                                # lying on its side
        d2 = axis[:2] / (np.linalg.norm(axis[:2]) + 1e-12)
        t = low[:, :2] @ d2
        c = float(pose[:2] @ d2)
        return float(t.min()) - span_tol <= c <= float(t.max()) + span_tol
  return False


def rest_faces(assembly, group):
  # PRECOMPUTED per-subassembly rest-pose TABLE: the placement ids on which `group` rests
  # STABLY, accounting for the reference
  # part's scatter anchor (so a wheel scattered cap-down is judged in its cap-down frame).
  # Replaces the old implicit "every one of the 6 faces is placeable" enumeration -- places(),
  # precompute_stable_faces and concrete search look placements up here. The backward symbolic
  # models use rest_face_classes(), the exact quotient by free world-z table yaw. Cached on the
  # assembly, keyed by subassembly and placement settings, like the grasp table.
  #   The candidate ids come from placement_table, so this is the same filter in both modes:
  #   six canonical faces by default, arbitrary per-group stable quats under
  #   --stable-rest-poses. The stability test is kept in BOTH modes -- computed poses are
  #   stable for the shape by construction, but this re-checks the ASSEMBLED GROUP at its
  #   scatter anchor, which is a different question.
  cache = getattr(assembly, "_rest_faces_cache", None)
  if cache is None:
    cache = assembly._rest_faces_cache = {}
  key = (_subasm_key(group), _STABLE_REST_POSES, _PLACEMENT_YAWS,
         _PLACEMENT_YAW_PREFIX)
  hit = cache.get(key)
  if hit is None:
    grp = set(group)
    ref = reference_part(grp)
    anchor_q = assembly.scatter_orientations.get(ref, np.array([1.0, 0.0, 0.0, 0.0]))
    table = placement_table(assembly, grp, ref)
    # A world-z yaw is an exact symmetry of table stability: it rotates the support polygon
    # and COM projection together and leaves all heights unchanged.  Certify each BASE rest
    # once, then inherit that verdict for its yaw copies.  Apart from avoiding repeated
    # geometry work, this prevents tiny convex-hull tolerances from accepting one yaw and
    # rejecting an equivalent one.
    yaw_base = getattr(assembly, "_placement_yaw_base", {})
    stable = {}
    # Preserve the historical sorted-id order when N=1.  With yaw augmentation, table
    # insertion order is intentional: all historical rests first, then coarse-to-fine yaw
    # bands, giving the search a useful deterministic prefix without changing its costs.
    candidates = sorted(table) if _PLACEMENT_YAWS == 1 else tuple(table)
    if _PLACEMENT_YAW_PREFIX is not None:
      table_key = (_subasm_key(grp), ref, _STABLE_REST_POSES, _PLACEMENT_YAWS)
      rank = getattr(assembly, "_placement_yaw_ranks", {}).get(table_key, {})
      candidates = tuple(face for face in candidates
                         if rank.get(face, 0) < _PLACEMENT_YAW_PREFIX)
    out = []
    for face in candidates:
      base = yaw_base.get(face, face)
      if base not in stable:
        stable[base] = _rest_stable(assembly, grp, ref, base, anchor_q)
      if stable[base]:
        out.append(face)
    hit = cache[key] = tuple(out)
  return hit


def rest_face_classes(assembly, group):
  """Stable-rest ids modulo the physically free table-yaw symmetry.

  Concrete search must retain every yaw because fixed-base arm IK is not yaw-invariant.
  The symbolic backward models deliberately omit arm reachability, however, and all facts
  they do retain (support topology, planar-table collision and wrench feasibility) rotate
  equivariantly about gravity.  Returning one representative per base rest is therefore an
  exact quotient of that abstraction, not an extra heuristic or a pruning rule.
  """
  faces = rest_faces(assembly, group)                    # also populates the base registry
  base_of = getattr(assembly, "_placement_yaw_base", {})
  return tuple(dict.fromkeys(base_of.get(face, face) for face in faces))


def placement_rest_class(assembly, face):
  """Abstract stable-rest id for a concrete placement id (quotienting table yaw only)."""
  return getattr(assembly, "_placement_yaw_base", {}).get(face, face)


# ---------------------------------------------------------------------------------------------
# TRUE stable rest orientations (the enumerator half of change 2)
#
# What the search uses TODAY is `rest_faces`: the six axis-aligned face-down rotations of
# _FACE_DOWN_QUATS, filtered by _rest_stable's support-polygon test. That is exact for a box
# and a reasonable approximation for a cylinder, and it is WRONG for anything else -- a
# chamfered part can rest on a chamfer, an L-bracket rests on two faces at once, and a mesh
# part usually has no axis-aligned resting face at all.
#
# The functions below compute the real thing from the geometry. They are DELIBERATELY not
# wired into the search yet: a placement id is today an index into a 6-element table and it
# lives inside State.place, so replacing the enumerator REBINDS EVERY SAVED PLAN AND WITNESS
# (the same compatibility class as rebinding grasp ids). See the module note in
# `stable_rest_poses` for the full list of what integration still needs.
# ---------------------------------------------------------------------------------------------
def _part_frame_mesh(spec):
  """The shape's mesh expressed in the PART frame (the spec's own shape rotation applied)."""
  m = (_compound_mesh(spec) if spec[0] == "compound" else _spec_mesh(spec)).copy()
  q = _shape_quat(spec)
  if q is not None and spec[0] != "compound":              # compounds bake theirs per child
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(q)
    m.apply_transform(T)
  return m


def _mat_to_quat_wxyz(R):
  q = np.zeros(4)
  M = np.eye(4)
  M[:3, :3] = np.asarray(R, dtype=float)
  import trimesh
  return np.asarray(trimesh.transformations.quaternion_from_matrix(M), dtype=float)


def stable_rest_poses(spec, mass_center=None, *, min_prob=0.02, max_poses=24, sigma=0.0):
  """TRUE stable resting orientations of one part shape, as [(quat, probability), ...].

  `quat` is [w,x,y,z] and rotates the PART FRAME into a pose in which the part rests on a
  horizontal table. The probability is trimesh's estimate of how likely that rest is when the
  part is dropped -- the solid angle over which the centre of mass falls inside the
  supporting facet -- so it also ranks the poses, which the current 6-face table cannot do.

  The criterion is the same one `_rest_stable` applies today (the projected centre of mass
  must lie inside the support polygon), but over the shape's OWN convex hull facets instead
  of over six assumed axis-aligned faces.

  DEGENERATE SHAPES. A rotationally symmetric part has a CONTINUUM of rests, not a set: a
  sphere rests in every orientation, a cylinder rolls on its barrel. trimesh answers with
  many near-equal-probability poses; `max_poses` caps that, and `min_prob` drops the noise.
  A caller that must distinguish "a set of rests" from "a continuum" should look at whether
  the returned probabilities are all equal and numerous -- see `rest_pose_report`.

  WHAT INTEGRATION STILL NEEDS (this function is only the enumerator):
    1. placement ids become opaque per-shape ids, not indices into _FACE_DOWN_QUATS. They sit
       in State.place, so every saved plan and witness rebinds -- plan a migration or a
       version tag.
    2. `placement_quat` composes the reference part's shape rotation; with real rest poses the
       pose IS the answer and that composition disappears.
    3. GROUPS need their own poses, not their reference part's -- use `group_stable_rest_poses`.
    4. rest heights (`_group_rest_height`, `_shape_half_z`) come free from the pose: the
       transform puts the supporting facet on z=0.
    5. the labeled heuristics assume a canonical rest face exists for every part; a part with
       NO stable pose (a sphere) must keep an explicit answer, or h goes to infinity and the
       milestone fills die (that failure is on record).
    6. scatter orientations should be drawn from these poses too, so a part starts in a rest
       the planner knows about.
  """
  import trimesh
  mesh = _part_frame_mesh(spec)
  com = None if mass_center is None else np.asarray(mass_center, dtype=float)
  T, P = trimesh.poses.compute_stable_poses(mesh, center_mass=com, sigma=float(sigma),
                                            n_samples=1, threshold=float(min_prob))
  out = []
  for i in range(len(T)):
    out.append((_mat_to_quat_wxyz(np.asarray(T[i])[:3, :3]), float(P[i])))
    if len(out) >= int(max_poses):
      break
  return out


def group_stable_rest_poses(assembly, group, ref=None, **kw):
  """Stable rest orientations of a whole SUBASSEMBLY, in its reference part's frame.

  A group is not its root part: a car body with wheels on rests on the WHEELS, and the chair
  backrest module rests flat on its plate. This builds the union mesh of the group at its
  assembled relative poses and uses the MASS-WEIGHTED centre of mass (parts carry different
  masses, so the volume centroid trimesh would assume is the wrong pivot), then defers to
  `stable_rest_poses`.
  """
  import trimesh
  parts = list(group)
  ref = ref or reference_part(set(parts))
  meshes, coms, masses = [], [], []
  for p in parts:
    m = _part_frame_mesh(assembly.part_shape(p))
    off = np.asarray(assembly.position_in_base(p), dtype=float) \
        - np.asarray(assembly.position_in_base(ref), dtype=float)
    T = np.eye(4)
    T[:3, 3] = off
    m.apply_transform(T)
    meshes.append(m)
    coms.append(np.asarray(m.center_mass, dtype=float))
    masses.append(float(assembly.mass(p)))
  union = trimesh.util.concatenate(meshes)
  w = np.asarray(masses, dtype=float)
  com = (np.asarray(coms) * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-12)
  return _stable_poses_of_mesh(union, com, **kw)


def _stable_poses_of_mesh(mesh, com, *, min_prob=0.02, max_poses=24, sigma=0.0):
  import trimesh
  T, P = trimesh.poses.compute_stable_poses(mesh, center_mass=np.asarray(com, dtype=float),
                                            sigma=float(sigma), n_samples=1,
                                            threshold=float(min_prob))
  return [(_mat_to_quat_wxyz(np.asarray(T[i])[:3, :3]), float(P[i]))
          for i in range(min(len(T), int(max_poses)))]


def rest_pose_report(spec, mass_center=None, **kw):
  """Summary of `stable_rest_poses`, with the two cases the discrete placement model cannot
  express called out:

    `continuum`  MANY rests of equal probability. That is the signature of a tessellated
                 CURVED support -- a 30cm leg returns a dozen equal rests, which are one
                 physical rest (lying on its barrel) sampled at a dozen roll angles. Six
                 equal rests are NOT a continuum: that is a cube, and they are genuinely
                 distinct placements.
    `neutral`    NO discrete rest at all (a sphere). Physically it rests in EVERY
                 orientation; the current heuristics assume a canonical rest face exists and
                 send h to infinity when it does not, so this case needs an explicit answer.

  NOTE ON PROBABILITY. trimesh ranks rests by how likely they are when the part is DROPPED.
  That is not the same question as which rests are USEFUL: a long leg standing on its cap is
  a rare drop outcome but a perfectly good (and often necessary) placement. So a planner must
  keep every rest above a stability threshold and must not take the top-k by probability.
  """
  poses = stable_rest_poses(spec, mass_center, **kw)
  probs = np.array([p for _q, p in poses], dtype=float)
  equal = bool(len(probs) >= 2 and float(probs.std()) < 1e-3)
  return {"n": len(poses), "probs": probs.round(4).tolist(),
          "equal_probs": equal,
          "continuum": bool(equal and len(probs) >= 8),
          "neutral": len(poses) == 0,
          "top_quat": (poses[0][0].round(6).tolist() if poses else None)}


def _grasp_frame(assembly, part, part_pose, grasp_id):
  # world frame of a grasp on a part: the CONTACT POINT (the grasp's actual location on the
  # part -- the DB grasp pose position, NOT the part centre) plus the APPROACH axis (grasp
  # local z) and the two LATERAL axes (local x/y). The axes feed the directional grasp model
  # (axial vs shear vs torque); the point is the MOMENT ARM of the grasp's forces. Using the
  # true location (not the centre) is what lets a grip NEAR the mate interface earn a short
  # lever -- a mid-part pinch and an end grip are no longer collapsed to the same point (the
  # chair post: a top-cap grip resists the sideways seating torque a mid-pin pinch cannot).
  # Symmetric grippers keep symmetric CAPABILITY (the axes are unchanged); only the arm moves.
  gpose = _pose_compose(np.asarray(part_pose, dtype=float),
                        grasp_pose(assembly, part, grasp_id))
  R = _quat_to_R(np.asarray(gpose[3:], dtype=float))
  return np.asarray(gpose[:3], dtype=float), R[:, 2], R[:, 0], R[:, 1]


# memoized equilibrium verdicts. The LP dominates the wrench-on search cost, but the result
# is invariant to a global translation of all contacts+loads and to absolute x/y position,
# so we canonicalise every query to the group centroid and round -- states that share the
# same discrete structure (parts, placement face, grounding contacts, applied load) then hit
# the same entry. Cleared per solve via reset_wrench_cache().
_WRENCH_CACHE = {}
# solve-scoped vacuum-engagement verdict cache (cleared with _WRENCH_CACHE).
_VEG_CACHE = {}


def reset_wrench_cache():
  _WRENCH_CACHE.clear()
  _VEG_CACHE.clear()


def reset_grasp_pose_cache():
  """The grasp-pose cache is keyed on the SHAPE plus rolls/offset/dense -- NOT on every global
  that shapes grasp generation (_ALLOW_CURVED_GRASPS and _PLAIN_BOX are absent). Two scenes in
  one process therefore share entries computed under different settings, and the grasp table
  silently stops matching the search that produced the plan."""
  _GRASP_POSE_CACHE.clear()
  _SAMPLED_VIEWS_CACHE.clear()                             # --grasp-sampler: same hazard class
  _SAMPLED_META_CACHE.clear()


def _grounding_caps(state, assembly, group, gripper_kind, table_override=None):
  # table_override: (model, mu) for the vacuum ENGAGEMENT gate's scoped static friction --
  # changes ONLY the table columns of THIS query, never the global PARAMS['table_model']
  # every other gate reads. Everything in the stability LP that does NOT depend on the
  # applied load: the grounding
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
  # MIXED EEs: gripper_kind may be a {robot: kind} MAP -- each grasp contact is priced with
  # its holder's kind. kind_of normalizes; cache keys replace the robot by its KIND (two
  # same-kind arms share caps, exactly like the old identity-free key). Scalars unchanged.
  if isinstance(gripper_kind, dict):
    kind_of = gripper_kind.get
    gk_key = tuple(sorted(gripper_kind.items()))
  else:
    kind_of = lambda _r: gripper_kind
    gk_key = gripper_kind
  cache = getattr(state, "_gcaps", None)
  if cache is None:
    cache = state._gcaps = {}
  ckey = (_subasm_key(group), gk_key, wrench.PARAMS.get("table_model"),
          wrench.PARAMS.get("table_mu"), table_override)
  hit = cache.get(ckey)
  if hit is not None:
    return hit
  members_sorted = _subasm_key(group)
  base = reference_part(group)
  on_table = any(("table",) in state.supports.get(p, ()) for p in group)
  grasps = tuple(sorted((kind_of(c[1]), c[2], c[3]) for p in group
                        for c in state.supports.get(p, ()) if c[0] == "grasp"))
  anchor_q = np.round(np.asarray(state.poses[base][3:], dtype=float), 9).tobytes()
  l2 = getattr(assembly, "_caps_cache", None)
  if l2 is None:
    l2 = assembly._caps_cache = {}
  l2key = (members_sorted, on_table, _placement_face(state, group) if on_table else None,
           anchor_q, grasps, wrench.PARAMS.get("grasp_model"),
           wrench.PARAMS.get("curved_grasp_torque_scale"),
           wrench.PARAMS.get("table_model"), wrench.PARAMS.get("table_mu"), _ROLLS_PER_FACE,
           table_override)
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
    fpts = [np.asarray(x, dtype=float) - centroid for x in fp]
    caps.append(wrench.table_capability_override(fpts, *table_override)
                if table_override else wrench.table_capability(fpts))
  derate = wrench.PARAMS.get("curved_grasp_torque_scale") is not None
  for p in members:
    for c in sorted(state.supports.get(p, ())):
      if c[0] == "grasp":
        if not _grasp_holds(assembly, members, c[2]):
          continue                                          # no_hold part can't brace a group
        gp, n, t1, t2 = _grasp_frame(assembly, c[2], poses[c[2]], c[3])  # c=(grasp,r,gpart,gid)
        surf = _grasp_surface(assembly.part_shape(c[2]), c[3]) if derate else "flat"
        caps.append(wrench.grasp_capability(gp - centroid, n, t1, t2, kind_of(c[1]),
                                            surface=surf))
  G = np.hstack([c[0] for c in caps]) if caps else np.zeros((6, 0))
  ub = np.concatenate([c[1] for c in caps]) if caps else np.zeros(0)
  bounds = [(0.0, None if not np.isfinite(u) else float(u)) for u in ub]
  gbytes = (np.round(G, 4).tobytes(), np.round(ub, 4).tobytes())
  hit = cache[ckey] = (centroid, w_grav, G, ub, bounds, gbytes)
  l2[l2key] = (centroid - anchor_pos,) + hit[1:]           # store the ANCHOR-RELATIVE centroid
  return hit


def _group_stable(state, assembly, group, gripper_kind, applied=(), table_override=None):
  # static-equilibrium feasibility for a subassembly: can its grounding contacts balance
  # gravity plus any `applied` [(force_vec_world, point_world[, pure_torque_world]), ...]
  # (e.g. insertion or screwing)? The optional third vector is a FREE COUPLE, not a force at
  # another invented lever arm. Keeping it explicit is essential for screw feasibility.
  # load-independent contacts come from _grounding_caps (cached); here we only add the applied
  # load to the gravity wrench and hit the LP cache / solve. table_override: scoped table
  # model for the engagement gate, see _grounding_caps.
  centroid, w_grav, G, ub, bounds, gbytes = _grounding_caps(
      state, assembly, group, gripper_kind, table_override=table_override)
  w_ext = w_grav
  for load in applied:
    f, pt = load[:2]
    w_ext = w_ext + wrench.point_force_wrench(f, np.asarray(pt, dtype=float) - centroid)
    if len(load) >= 3:
      w_ext = w_ext + np.concatenate([np.zeros(3), np.asarray(load[2], dtype=float)])
  geo_key = (gbytes[0], gbytes[1])
  key = (geo_key[0], geo_key[1], np.round(w_ext, 4).tobytes())
  hit = _WRENCH_CACHE.get(key)
  if hit is not None:
    return hit
  res = wrench.equilibrium_feasible_matrices(G, ub, w_ext, bounds=bounds, geo_key=geo_key)
  _WRENCH_CACHE[key] = res
  return res


def state_contact_wrenches(state, assembly, gripper_kind, applied=(), poses=None):
  """Per-contact equilibrium wrenches of a state, for VISUALIZATION (--show-wrenches).

  For every subassembly with grounding contacts, rebuild the capability matrix like
  _grounding_caps (uncached; per-contact column SPANS kept) and solve one balancing LP for
  gravity plus any `applied` [(force_world, point_world, group_hint_part), ...] loads.
  Returns [(point_world, force_3vec, torque_3vec, label), ...]: one entry per grasp contact,
  one per table support point carrying force, plus each applied load itself (label 'press').
  Infeasible groups contribute a single ('!' label) marker at the group centroid instead."""
  out = []
  kind_of = gripper_kind.get if isinstance(gripper_kind, dict) else (lambda _r: gripper_kind)
  if poses is None:                                        # callers rendering REAL poses (the
    poses = _state_part_poses(state, assembly, lift=0.35)  # IK replay) pass their own dict
  else:
    poses = dict(poses)
  for group in state.subassemblies():
    contacts = [(p, c) for p in sorted(group, key=_part_index)
                for c in sorted(state.supports.get(p, ())) if c[0] == "grasp"]
    on_table = any(("table",) in state.supports.get(p, ()) for p in group)
    if not contacts and not on_table:
      continue
    members = sorted(group, key=_part_index)
    pos = {p: np.asarray(poses[p][:3], dtype=float) for p in members}
    centroid = np.mean([pos[p] for p in members], axis=0)
    w_ext = wrench.gravity_wrench([(assembly.mass(p), pos[p] - centroid) for p in members])
    g_applied = [(f, pt) for f, pt, hint in applied if hint in group]
    for f, pt in g_applied:
      w_ext = w_ext + wrench.point_force_wrench(f, np.asarray(pt, dtype=float) - centroid)
    caps, spans = [], []                                   # spans: (kind, anchor_pt, label)
    if on_table:
      fp = _table_footprint(assembly, poses, members)
      for x in fp:                                        # one unilateral column per point
        caps.append(wrench.table_capability([np.asarray(x, dtype=float) - centroid]))
        spans.append(("table", np.asarray(x, dtype=float), "table"))
    for p, c in contacts:
      gp, n, t1, t2 = _grasp_frame(assembly, c[2], poses[c[2]], c[3])
      kk = kind_of(c[1]) if gripper_kind is not None else "two_finger"
      caps.append(wrench.grasp_capability(gp - centroid, n, t1, t2, kk))
      spans.append(("grasp", gp, f"{c[1]}:{c[2]}"))
    G = np.hstack([cp[0] for cp in caps]) if caps else np.zeros((6, 0))
    ub = np.concatenate([cp[1] for cp in caps]) if caps else np.zeros(0)
    ok, x = wrench.equilibrium_solution(G, ub, w_ext)
    if not ok:
      out.append((centroid, np.zeros(3), np.zeros(3), "!infeasible"))
      continue
    col = 0
    for (kind, pt, label), (Gi, ubi) in zip(spans, caps):
      k = Gi.shape[1]
      xi = x[col:col + k]
      col += k
      lin = Gi[:3, :]
      is_couple = np.all(np.abs(lin) < 1e-12, axis=0)
      force = lin @ xi                                     # net force at the contact point
      torque = (Gi[3:, :][:, is_couple] @ xi[is_couple]) if is_couple.any() else np.zeros(3)
      if np.linalg.norm(force) > 1e-6 or np.linalg.norm(torque) > 1e-6:
        out.append((pt, force, torque, label))
    for f, pt in g_applied:
      out.append((np.asarray(pt, dtype=float), np.asarray(f, dtype=float),
                  np.zeros(3), "press"))
  return out


_SURFACE_CACHE = {}


def _grasp_surface(spec, grasp_id):
  # "flat" | "curved" surface a grasp contacts, memoized (part_grasp_surfaces depends only on
  # the shape KIND, the roll count, and dense mode). Only consulted when
  # curved_grasp_torque_scale is set.
  # keyed on the canonical spec identity: mesh/compound parts must not alias on the kind
  # string -- two different meshes/compounds are two different surfaces.
  key = (_spec_key(spec), _ROLLS_PER_FACE, _DENSE_GRASPS, _GRASP_TRANSLATIONS)
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


def _group_stable_affine(state, assembly, group, gripper_kind, f0, point, gens,
                         pure_torque=None):
  # CONSERVATIVE robust check: one AFFINE contact-force policy must hold across the whole
  # uncertainty box (wrench.equilibrium_robust_affine), vs _group_stable_loaded's EXACT per-corner
  # test. Builds the nominal wrench w0 and the box's generator wrenches delta_j about the group
  # centroid, then a single LP. SOUNDNESS LIMIT: the deltas linearize the box, so feasibility
  # implies every TRUE corner is stable only when the box has no point-shift generators
  # (offset == 0) -- with shift and tilt together the corners carry a bilinear cross couple
  # that no linear policy can express. _robust_stable guards this (falls back to corners).
  centroid, w_grav, G, ub, bounds, _ = _grounding_caps(state, assembly, group, gripper_kind)
  f0 = np.asarray(f0, dtype=float); point = np.asarray(point, dtype=float)
  w0 = w_grav + wrench.point_force_wrench(f0, point - centroid)
  if pure_torque is not None:
    w0 = w0 + np.concatenate([np.zeros(3), np.asarray(pure_torque, dtype=float)])
  deltas = []
  for kind, axis, amt in gens:
    axis = np.asarray(axis, dtype=float)
    if kind == "f":                                         # transverse force amt*axis at `point`
      deltas.append(wrench.point_force_wrench(amt * axis, point - centroid))
    else:                                                   # point shift amt*axis -> pure couple
      deltas.append(np.concatenate([np.zeros(3), _cross3(amt * axis, f0)]))
  return wrench.equilibrium_robust_affine(G, ub, w0, deltas, bounds=bounds)


# one-shot process warning: affine robust mode cannot express the tilt-x-shift cross couple
_AFFINE_FALLBACK_WARNED = [False]


def _robust_stable(state, assembly, group, gripper_kind, f_world, point, mag, scale=1.0,
                   pure_torque=None):
  # dispatch the robust insertion-stability check by wrench.PARAMS['robust_mode']:
  #   'corners' (default, EXACT): every uncertainty-box corner must be balanceable, the contacts
  #             free to rechoose their force split per realization (fully-adjustable robustness).
  #   'affine': one fixed affine force-response policy (open-loop grasp) covers the LINEARIZED
  #             box -- a single LP, stricter per axis. It is sound ONLY when the box has one
  #             axis KIND (force-only or shift-only): with both --insertion-noise (tilt shear)
  #             and --insertion-offset (point shift) the true corners carry a bilinear cross
  #             couple that no linear policy can express, so the dispatch FALLS BACK to the
  #             exact corners test and warns once per process.
  # Identical to the plain nominal check when insertion_uncertainty is off (no generators).
  unc = wrench.PARAMS.get("insertion_uncertainty")
  if wrench.PARAMS.get("robust_mode", "corners") == "affine" and unc:
    if float(unc.get("lateral", 0.0)) > 0.0 and float(unc.get("offset", 0.0)) > 0.0:
      if not _AFFINE_FALLBACK_WARNED[0]:
        _AFFINE_FALLBACK_WARNED[0] = True
        print("[wrench] --robust-mode affine with BOTH --insertion-noise and --insertion-offset "
              "cannot be sound (an affine policy cannot express the tilt-x-shift cross "
              "couple): using the exact 'corners' test instead.")
    else:
      f0, pt, gens = _insertion_uncertainty_gens(f_world, point, mag, scale=scale)
      return _group_stable_affine(state, assembly, group, gripper_kind, f0, pt, gens,
                                  pure_torque=pure_torque)
  variants = _insertion_uncertainty_variants(f_world, point, mag, scale=scale)
  if pure_torque is not None:
    variants = [(f, pt, np.asarray(pure_torque, dtype=float)) for f, pt in variants]
  return _group_stable_loaded(state, assembly, group, gripper_kind, variants)


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


def _insertion_uncertainty_gens(f_world, point, mag, scale=1.0):
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
  # `scale` inflates the box (GRASP COMPLIANCE, --grasp-compliance): a compliant holder --
  # the vacuum cup -- DEFLECTS under the press, so the same commanded seating wanders more:
  # effectively a larger misalignment box for the same noise level. Stiff holds scale 1.0.
  lat = float(unc.get("lateral", 0.0)) * mag * scale      # transverse-force half-widths (N)
  off = float(unc.get("offset", 0.0)) * scale             # application-offset half-widths (m)
  gens = []                                               # (apply-to, axis, half-width)
  if lat > 0.0:
    gens += [("f", t1, lat), ("f", t2, lat)]              # angular tilt -> transverse force
  if off > 0.0:
    gens += [("p", t1, off), ("p", t2, off)]              # positional shift -> off-centre couple
  return f_world, point, gens


def _insertion_uncertainty_variants(f_world, point, mag, scale=1.0):
  # the uncertainty box as its CORNERS (list of (force, point)). Because the set of balanceable
  # external wrenches is CONVEX, requiring every CORNER stable (_group_stable_loaded) is EXACT for
  # the box -- no sampling, no tunable count. (The box is a conservative OUTER polytope of the true
  # circular cone/disk, so it rejects marginally more, in the safe direction.) Off => [(nominal)].
  f_world, point, gens = _insertion_uncertainty_gens(f_world, point, mag, scale=scale)
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
  # applied at the MATE INTERFACE (assembly.mate_interface: the explicit contact patch, or
  # the part's face along the press axis by default) -- where the seam transmits the load.
  # Along-axis default: nominal press is centre-equivalent (parallel offset, no moment) and
  # only the noise's lateral components gain the lever. OFF-AXIS interface (chair post:
  # contact at the TOP, press SIDEWAYS): the NOMINAL press itself becomes a real torque.
  point = np.asarray(poses[onto][:3], dtype=float) + R @ (
      rel + assembly.mate_interface(part, d))
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
  point = np.asarray(poses[part][:3], dtype=float) \
      + R @ assembly.mate_interface(part, d)                # at the mate interface (real lever)
  return f_world, point


def _screw_torque_load(state, assembly, part, onto, *, target_side):
  """Pure world-frame couple for a screw mate, or ``None`` for ordinary insertion.

  The signed rotation fixes the torque direction.  Target and inserted groups receive equal
  and opposite couples.  The torque is independent of the contact point by definition; it
  must therefore enter the 6-D equilibrium load explicitly rather than through a fake force
  lever arm.
  """
  skill = assembly.screw_skill(part)
  if skill is None or skill.torque <= 0.0:
    return None
  poses = _state_part_poses(state, assembly, lift=0.0)
  frame = onto if target_side else part
  R = _quat_to_R(np.asarray(poses[frame][3:], dtype=float))
  axis = R @ np.asarray(assembly.mate_axis(part), dtype=float)
  axis = axis / (np.linalg.norm(axis) + 1e-12)
  direction = np.sign(skill.angle) * (-1.0 if target_side else 1.0)
  return direction * float(skill.torque) * axis


def _insertion_load_variants(state, assembly, part, onto, d=None, scale=1.0):
  # the target-side insertion reaction, as a list of worst-case realizations under the
  # insertion-uncertainty model (a singleton = the nominal load when the flag is off).
  f_world, point = _insertion_load(state, assembly, part, onto, d)
  return _insertion_uncertainty_variants(f_world, point, wrench.insert_force(part), scale=scale)


def _inserter_load_variants(state, assembly, part, onto, d=None):
  # the inserter-side seating reaction (on the held part), likewise as worst-case realizations.
  f_world, point = _inserter_load(state, assembly, part, onto, d)
  return _insertion_uncertainty_variants(f_world, point, wrench.insert_force(part))


def _compliance_factor(state, assembly, group, gripper_kind):
  # GRASP COMPLIANCE (--grasp-compliance): the multiplier a group's holding grasps put on
  # the insertion-uncertainty box -- the WORST (most compliant) grasp in the load path
  # decides (a chain deflects at its softest link). Table contact is stiff (1.0). Off
  # (PARAMS None) or no gripper model => 1.0, byte-identical.
  comp = wrench.PARAMS.get("grasp_compliance")
  if not comp or gripper_kind is None:
    return 1.0
  kind_of = gripper_kind.get if isinstance(gripper_kind, dict) else (lambda _r: gripper_kind)
  f = 1.0
  for p in group:
    for c in state.supports.get(p, ()):
      if c[0] == "grasp":
        f = max(f, float(comp.get(kind_of(c[1]), 1.0)))
  return f


def _wrench_params_sig():
  # a compact signature of every wrench.PARAMS knob these verdicts read, for cache keys.
  # PARAMS may be re-tuned between scenes in one process (replay, tests), so a per-state
  # verdict cache without this would serve stale answers across a re-tune.
  def _sigv(x):
    if isinstance(x, dict):
      return tuple(sorted((k, _sigv(v)) for k, v in x.items()))
    return x
  unc = wrench.PARAMS.get("insertion_uncertainty")
  return (wrench.PARAMS["grasp_model"],
          _sigv(wrench.PARAMS.get("grasp", {})),      # the CAP VALUES themselves: --vacuum/
                                                      # --two-finger-scale retunes them in-process,
                                                      # and a sig without them serves stale LP
                                                      # verdicts across a re-tune (measured:
                                                      # --two-finger-scale 5 read the x1 verdict)
          wrench.PARAMS.get("curved_grasp_torque_scale"),
          wrench.PARAMS.get("table_model"), wrench.PARAMS.get("table_mu"),
          _sigv(wrench.PARAMS.get("grasp_compliance")),
          wrench.PARAMS.get("robust_mode"),
          wrench.PARAMS.get("ignore_gravity"),
          _sigv(wrench.PARAMS.get("insert_force", {})),
          wrench.PARAMS.get("vacuum_engage_force"),
          wrench.PARAMS.get("vacuum_engage_friction"),
          None if not unc else _sigv(unc))


def _insertion_stable(state, assembly, group, gripper_kind, part, onto, d=None):
  # `group` (the TARGET) withstands the robust insertion reaction of `part` seating onto it.
  # A compliantly-held target wanders under the press -> its uncertainty box inflates.
  # PER-STATE VERDICT CACHE: the gate is deterministic in the state (poses/supports are part
  # of the state identity -- the search keeps the key <-> world-pose mapping a BIJECTION), and
  # under --insertion-noise it costs 4 LP corners per call -- the fill asks it once per (state,
  # assemble candidate), which measured thousands of redundant re-asks in a wandering fill.
  # The PARAMS signature keeps the cache sound across in-process re-tunes (replay, tests).
  key = (state.key(), part, onto,
         _compliance_factor(state, assembly, group, gripper_kind), _wrench_params_sig())
  cache = getattr(state, "_ins_cache", None)
  if cache is None:
    cache = state._ins_cache = {}
  hit = cache.get(key)
  if hit is not None:
    return hit
  f_world, point = _insertion_load(state, assembly, part, onto, d)
  res = _robust_stable(state, assembly, group, gripper_kind, f_world, point,
                       wrench.insert_force(part), scale=key[3],
                       pure_torque=_screw_torque_load(
                           state, assembly, part, onto, target_side=True))
  cache[key] = res
  return res


def _inserter_stable(state, assembly, group, gripper_kind, part, onto, d=None):
  # the HELD `group` transmits its own weight plus the robust seating reaction through its
  # grasp(s). A compliant inserter grasp cannot place the press accurately -> box inflates.
  # Same per-state verdict cache as _insertion_stable (the inserter side is asked just as
  # often, with the same 4-corner cost under noise).
  key = (state.key(), part, onto,
         _compliance_factor(state, assembly, group, gripper_kind), _wrench_params_sig())
  cache = getattr(state, "_iser_cache", None)
  if cache is None:
    cache = state._iser_cache = {}
  hit = cache.get(key)
  if hit is not None:
    return hit
  f_world, point = _inserter_load(state, assembly, part, onto, d)
  res = _robust_stable(state, assembly, group, gripper_kind, f_world, point,
                       wrench.insert_force(part), scale=key[3],
                       pure_torque=_screw_torque_load(
                           state, assembly, part, onto, target_side=False))
  cache[key] = res
  return res


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
  q = placement_quat(assembly, context or {assembly.base}, assembly.base, face)
  R = _quat_to_R(q)
  rest = _group_rest_height(assembly, context or {assembly.base}, assembly.base, face)
  base_pos = np.array([0.0, 0.0, top + rest])
  for p in sorted(context | {part}, key=_part_index):
    add_part_frame(C, p, assembly.part_shape(p)).setPosition(
        base_pos + R @ assembly.position_in_base(p)).setQuaternion(q)
  return all(d >= -tol for _a, _b, d in C.getCollisions())
