"""Problem definition: assemblies, part geometry, grasps, and the discrete scene-graph
search model (State + successors). No feasibility/solver or visualization code here."""

import os

import robotic as ry
import numpy as np

from env import *


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
  """

  def __init__(self):
    self.objects = []
    self.base = None
    self.assembly_dependencies = {}
    self.assembly_relative_poses = {}
    self.define()
    self.validate()

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

def add_disassembled_obj(C, assembly, perturb_start_pose=False, pos_noise=0.03, max_tries=200):
  # scatter the assembly's parts on the table as loose cubes (test scene). Geometry
  # (cube_size / cube_radius) comes from the assembly, not a global.
  cs, cr = assembly.cube_size, assembly.cube_radius

  for i, name in enumerate(assembly.objects):
      pos = np.array([-0.5 + i * 1 / 7, 0, 0.07])

      obj = (
          C.addFrame(name).setParent(C.getFrame("table")).setShape(
              ry.ST.ssBox, [cs, cs, cs, cr]
          ).setRelativePosition([pos[0], pos[1], pos[2]]).setMass(0.1).setColor(
              np.random.rand(3)
          ).setContact(1).setJoint(ry.JT.rigid)
      )

      if perturb_start_pose:
          # rejection-sample a perturbation that keeps the config collision free
          for _ in range(max_tries):
              dxy = np.random.uniform(-pos_noise, pos_noise, size=2)
              yaw = np.random.uniform(-np.pi, np.pi)
              obj.setRelativePosition([pos[0] + dxy[0], pos[1] + dxy[1], pos[2]])
              obj.setRelativeQuaternion(
                  [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
              )
              if C.getCollisionFree():
                  break
          else:
              # could not place without collision: fall back to the nominal pose
              obj.setRelativePosition([pos[0], pos[1], pos[2]])
              obj.setRelativeQuaternion([1, 0, 0, 0])

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


def make_assembly(kind, n_parts):
  """Assembly factory for the CLI: cross (cube cross), cube (2x2x2 grid), arch (bridge --
  known-flawed star, see class doc), arch-seq (arch shape built as a sequential chain)."""
  kinds = {"cross": (CubeCrossAssembly, 7), "cube": (CubeGridAssembly, 8),
           "arch": (ArchAssembly, 7), "arch-seq": (ArchSequentialAssembly, 7)}
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
def _quat_to_R(q):
  w, x, y, z = q
  return np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ])


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
  # their obj id. Placeholder geometry: every part is a cube (size from the assembly).
  cs, cr = assembly.cube_size, assembly.cube_radius
  C = ry.Config()
  ref_pos = assembly.position_in_base(subassembly[0])
  for obj in subassembly:
    obj_in_ref = assembly.position_in_base(obj) - ref_pos
    (C.addFrame(obj)
        .setShape(ry.ST.ssBox, [cs, cs, cs, cr])
        .setPosition(obj_in_ref)
        .setContact(contact))
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


def cube_grasp_pose(grasp_id, cube_size, offset=0.01):
  # grasp pose [x,y,z, qw,qx,qy,qz] in the grasped cube's OWN frame. grasp_id encodes
  # (face, roll): the face sets the approach (-normal); the roll spins the gripper about
  # that approach axis (in-plane finger orientation). cube_size comes from the assembly.
  face, roll = divmod(grasp_id, _ROLLS_PER_FACE)
  normal = _CUBE_FACE_NORMALS[face]
  standoff = cube_size / 2.0 + offset
  angle = roll * (np.pi / _ROLLS_PER_FACE)                 # 0, 90 deg for two-finger
  roll_q = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])  # about approach (local z)
  q = _quat_mul(_grasp_quat_for_normal(normal), roll_q)
  return np.concatenate([standoff * normal, q])


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
    # every part except the one being grasped (touching it is allowed).
    C_col = ry.Config()
    C_col.addConfigurationCopy(base_scene)
    base_world = _ee_root_pose(grasp_pose, base_rel_tool, q_correction)
    root = C_col.getFrame("ee_ur_base")
    root.setPosition(base_world[:3])
    root.setQuaternion(base_world[3:])
    C_col.getFrame(grasped_obj).setContact(0)
    return C_col.getCollisionFree()

  labeled = []
  for obj in subassembly:
    # obj origin expressed in the reference (first obj) frame
    obj_in_ref = assembly.position_in_base(obj) - ref_pos
    # _N_CUBE_GRASPS candidate grasps per part (faces x rolls); grasp_id indexes the
    # part's grasp set. cube_grasp_pose gives the grasp in the part frame; shift it into
    # the subassembly ref frame (parts are axis-aligned, so translation only).
    for grasp_id in range(_N_CUBE_GRASPS):
      local = cube_grasp_pose(grasp_id, cube_size, offset)
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

  def is_supported(self, group):
    # topological support (some member is grounded) gated by the wrench stub
    grounding = self.grounding_contacts(group)
    return len(grounding) > 0 and wrench_sufficient(grounding)

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
    # the assembly is complete: every part joined into a single subassembly. We do NOT
    # require it to rest on the table or be hands-off -- held-in-the-air counts as done.
    # (Every reachable state is supported by construction anyway.)
    return self.is_complete(assembly)

  # --- identity ---
  def key(self):
    return (
        tuple(sorted(self.parent.items())),
        tuple(sorted(self.grasp.items())),
        tuple(sorted(self.place.items())),
        tuple(sorted((p, tuple(sorted(cs))) for p, cs in self.supports.items())),
    )

  def __eq__(self, other):
    return isinstance(other, State) and self.key() == other.key()

  def __hash__(self):
    return hash(self.key())

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
    for face in range(len(_FACE_DOWN_QUATS)):             # one placement per resting face
      ns = _copy_edit_support(state)
      for p in group:                                     # open r's grasp on this group
        ns.supports[p] = frozenset(
            c for c in ns.supports[p] if not (c[0] == "grasp" and c[1] == r))
      ns.grasp.pop(root, None)
      ns.parent = _rerooted(ns.parent, ref)               # canonical tree orientation
      ns.parent[ref] = ("table",)
      ns.place[ref] = ("slot", _part_index(ref), face)
      ns.supports[ref] = ns.supports[ref] | {("table",)}
      yield ("place", r, root, face), ns


def assembles(state, assembly, robots):
  # a robot mates its held single part onto a supported target assembly per the manual.
  # The robot KEEPS holding (its grasp becomes a brace); freeing it is a separate,
  # support-gated `release`. So assemble only adds the mate -- it does not let go.
  for group in state.subassemblies():
    if len(group) != 1:
      continue                                            # simplest: add single parts only
    (p,) = tuple(group)
    if state.parent.get(p, (None,))[0] != "gripper":
      continue                                            # the added part must be held
    holder = state.parent[p][1]
    rel = assembly.assembly_relative_poses.get(p)
    if rel is None:
      continue                                            # base part has no mate parent
    onto = rel[0]
    target = next((g for g in state.subassemblies() if onto in g), None)
    if target is None:
      continue
    if not _mate_closure(assembly, p).issubset(target):
      # the target must contain ALL of p's transitive dependencies (deps + mate parents),
      # not just the direct one -- otherwise a chain that mates onto its PREDECESSOR (not
      # the base, e.g. arch-seq) could build a FLOATING mid-chain group that is not a
      # dependency-closed order ideal, so it is never enumerated in get_subassemblies /
      # the grasp table (KeyError) and is not a sensible sequential build anyway.
      continue                                            # precedence: target stays an ideal
    if not state.is_supported(target):
      continue                                            # target must be supported
    if _group_is_on_table(state, target) and not _on_table_mateable_cached(
        assembly, p, _placement_face(state, target)):
      # the mate would jam p into the table at the group's CURRENT rest pose -- for its
      # CURRENT placement face (e.g. the -z cube under an upright base; a different face
      # may well admit it: that is what reorientation placements are for). The table
      # contact must genuinely be removed first -- pick, mate in the air, place back (or
      # re-place on a suitable face). Without this gate the state would carry a FALSE
      # persistent table support through a braced lift, wrongly legalizing releases.
      continue
    ns = _copy_edit_support(state)
    ns.parent[p] = ("part", onto)     # pose now defined by the mate, not the gripper
    ns.grasp.pop(p, None)             # ... so it is no longer the kinematic gripper child
    ns.place.pop(p, None)
    ns.supports[p] = ns.supports[p] | {("mate", onto)}    # grasp contact stays as a brace
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


def releases(state, assembly, robots):
  # a robot opens its gripper without setting the part down. Support-gated: legal only if
  # the subassembly is still supported once this grasp is gone. This is the whole "you
  # can't always let go" rule -- with wrench ignored it just means "something else still
  # grounds it" (a mate to a grounded neighbour, the table, or another gripper).
  for group in state.subassemblies():
    for p in group:
      for c in [c for c in state.supports[p] if c[0] == "grasp"]:
        r = c[1]
        ns = _copy_edit_support(state)
        ns.supports[p] = frozenset(x for x in ns.supports[p] if x != c)
        if not ns.is_supported(group):
          continue                    # letting go would drop it -> not allowed
        if ns.parent.get(p) == ("gripper", r):
          # the released grasp was the kinematic parent yet the part stays supported by
          # another contact -> needs reparenting to that supporter; only arises with
          # add_grasp (handover/brace), which is deferred. Skip for now.
          continue
        yield ("release", r, p), ns


def successors(state, assembly, robots, grasp_table=None, canonical=False):
  yield from picks(state, assembly, robots, grasp_table=grasp_table, canonical=canonical)
  yield from assembles(state, assembly, robots)
  yield from releases(state, assembly, robots)
  yield from places(state, assembly, robots)
  # deferred (unfused): add_grasps, set_downs



# ===========================================================================
# State visualization (debugging)
# ===========================================================================

def _group_rest_lift(assembly, group, ref, face=_IDENTITY_FACE):
  # how far the group's reference part must sit ABOVE the single-cube rest height so the
  # group's LOWEST part rests on the table (0 for a bare cube or top-heavy groups), for
  # the given PLACEMENT face (the group is rotated so cube face `face` points down).
  # This is the table-placement rule: keep ref's x/y, rotate per the placement, lift to
  # clear parts hanging below (e.g. the -z cube of the cross when placed upright).
  R = _quat_to_R(_FACE_DOWN_QUATS[face])
  ref_off = assembly.position_in_base(ref)
  min_rel = min((R @ (assembly.position_in_base(p) - ref_off))[2] for p in group)
  return -min_rel                                          # >= 0 (ref itself gives 0)


def _placement_face(state, group):
  # the placement face of an on-table group, read from its reference part's slot entry
  # (("slot", idx, face)); identity for legacy/unknown entries.
  pl = state.place.get(reference_part(group))
  return pl[2] if pl is not None and len(pl) > 2 else _IDENTITY_FACE


def _group_rest_pose(assembly, group, ref, anchor_pose, face):
  # the reference part's full rest pose for a placement: anchor x/y, rotated so `face`
  # points down (composed onto the anchor's own orientation, so a perturbed initial yaw
  # is preserved), lifted so the group's lowest part touches the table.
  anchor_pose = np.asarray(anchor_pose, dtype=float)
  q = _quat_mul(_FACE_DOWN_QUATS[face], anchor_pose[3:])
  pos = anchor_pose[:3].copy()
  pos[2] += _group_rest_lift(assembly, group, ref, face)
  return np.concatenate([pos, q])


def _state_part_poses(state, assembly, lift=0.35):
  # synthesize a world pose per part from the scene graph: each subassembly is laid out
  # around its base part's anchor pose; held subassemblies are raised by `lift` so it is
  # visually obvious they are off the table.
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
  return poses


# penetration the CHEAP checker tolerates before rejecting. The real IK's collision
# objective is a SOFT inequality (weight 1e1 + the NLP feasibility threshold), so it
# accepts grazing contacts of a few mm; a binary-exact cheap check would then be STRICTER
# than L3 on borderline contacts -- breaking its lower-bound property (cheap-reject must
# imply IK-reject). Real rejections (finger-into-table, neighbouring cubes) penetrate
# 18-57mm, grazes are <5mm: cleanly separated by this margin.
_CHEAP_PENETRATION_TOL = 0.005


def _mate_closure(assembly, part):
  # every part that must already be assembled before `part` can be mated: transitive
  # dependencies plus the chain of mate parents (both are required by `assembles`).
  need = set()
  frontier = [part]
  while frontier:
    p = frontier.pop()
    for q in list(assembly.dependencies_of(p)) + (
        [assembly.relative_pose(p)[0]] if assembly.relative_pose(p) else []):
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
  cs, cr = assembly.cube_size, assembly.cube_radius
  C = make_world()
  table = C.getFrame("table")
  top = table.getPosition()[2] + table.getSize()[2] / 2.0
  # the CONTEXT group (everything that must already be assembled) rests on the table per
  # the placement rule: rotated so `face` points down, lifted so ITS lowest part touches
  # the table. `part` is then posed at its rotated mate offset -- does the mate fit?
  context = _mate_closure(assembly, part)
  q = _FACE_DOWN_QUATS[face]
  R = _quat_to_R(q)
  lift = _group_rest_lift(assembly, context, assembly.base, face) if context else 0.0
  base_pos = np.array([0.0, 0.0, top + cs / 2.0 + lift])
  for p in sorted(context | {part}, key=_part_index):
    C.addFrame(p).setShape(ry.ST.ssBox, [cs, cs, cs, cr]).setPosition(
        base_pos + R @ assembly.position_in_base(p)).setQuaternion(q).setContact(1)
  return all(d >= -tol for _a, _b, d in C.getCollisions())
