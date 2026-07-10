import os

import robotic as ry
import numpy as np

from env import *

def add_disassembled_obj(C, perturb_start_pose=False, pos_noise=0.03, max_tries=200):
  # just add a bunch of cubes. the first thing we want to do is a cube with another cube on each of its faces, assembled through magic
  # 7 cubes

  size = [0.05, 0.05, 0.05]

  for i in range(7):
      pos = np.array([-0.5 + i * 1 / 7, 0, 0.07])

      obj = (
          C.addFrame("obj_" + str(i)).setParent(C.getFrame("table")).setShape(
              # ry.ST.sphere, [0.05, 0.025]
              ry.ST.ssBox, [size[0], size[1], size[2], 0.005]
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


class CubeCrossAssembly(Assembly):
  """A central cube with one cube glued to each of its six faces (7 parts total).

  Parts are assembled one at a time in index order (obj_1 needs obj_0, obj_2 needs
  obj_1, ...), and every leaf cube sits one `spacing` away from the base along a
  distinct axis.
  """

  def __init__(self, spacing=0.055):
    self.spacing = spacing
    super().__init__()

  def define(self):
    self.objects = [f"obj_{i}" for i in range(7)]
    self.base = "obj_0"

    self.assembly_dependencies = {
        f"obj_{i}": [f"obj_{i - 1}"] for i in range(1, 7)
    }

    o = self.spacing
    face_offsets = [
        [0, 0, o], [0, 0, -o],
        [0, o, 0], [0, -o, 0],
        [o, 0, 0], [-o, 0, 0],
    ]
    self.assembly_relative_poses = {
        f"obj_{i}": ["obj_0", np.array(face_offsets[i - 1])] for i in range(1, 7)
    }

def display_final_assembly(C, assembly, base_pose):
  C_final = ry.Config()
  C_final.addConfigurationCopy(C)

  for obj, parent in assembly.assembly_relative_poses.items():
    C_final.attach(
        parent[0],
        obj,
    )

    C_final.getFrame(obj).setRelativePosition(parent[1])

    # keep the object's color, just lower its opacity
    mesh_colors = np.asarray(C_final.getFrame(obj).getMeshColors(), dtype=float)
    rgb = (mesh_colors.reshape(-1, mesh_colors.shape[-1])[0][:3]) / 255.0
    C_final.getFrame(obj).setColor([rgb[0], rgb[1], rgb[2], 0.3])

  C_final.getFrame(assembly.base).setPosition(base_pose)

  C_final.view(True)

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

def show_subassemblies(C, assembly, subassemblies, base_pose):
  for subassembly in subassemblies:
    if len(subassembly) == 1:
      continue 

    C_final = ry.Config()
    C_final.addConfigurationCopy(C)

    for obj, parent in assembly.assembly_relative_poses.items():
      if obj in subassembly:
        C_final.attach(
            parent[0],
            obj,
        )

        C_final.getFrame(obj).setRelativePosition(parent[1])

      # keep the object's color, just lower its opacity
      mesh_colors = np.asarray(C_final.getFrame(obj).getMeshColors(), dtype=float)
      rgb = (mesh_colors.reshape(-1, mesh_colors.shape[-1])[0][:3]) / 255.0
      C_final.getFrame(obj).setColor([rgb[0], rgb[1], rgb[2], 0.3])

    C_final.getFrame(assembly.base).setPosition(base_pose)

    C_final.view(True)

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


def make_subassembly_scene(assembly, subassembly, cube_size=0.05, contact=1):
  # fresh Config holding the subassembly's parts at their assembled relative poses,
  # expressed in the first obj's frame (ref part at the origin). Frames are named by
  # their obj id. Placeholder geometry: every part is assumed to be a cube.
  C = ry.Config()
  ref_pos = assembly.position_in_base(subassembly[0])
  for obj in subassembly:
    obj_in_ref = assembly.position_in_base(obj) - ref_pos
    (C.addFrame(obj)
        .setShape(ry.ST.ssBox, [cube_size, cube_size, cube_size, 0.005])
        .setPosition(obj_in_ref)
        .setContact(contact))
  return C


def place_floating_ee(C, target_pose, gripper_path=None, prefix="ee_"):
  # add a free-floating end effector to C and place it so its tool center (the grasp
  # point) sits at target_pose, with the EE's approach axis aligned to target's +z.
  # different EEs name their tool center differently and extend along different axes
  # (two-finger: +z, vacuum: +x); both are auto-detected. target_pose is a 7-vector
  # [x,y,z,qw,qx,qy,qz] in C's world frame. returns the tool-center frame name.
  if gripper_path is None:
    gripper_path = os.path.join(os.path.dirname(__file__), DEFAULT_FLOATING_EE)
  target_pose = np.asarray(target_pose, dtype=float)

  C.addFile(gripper_path, namePrefix=prefix)
  names = set(C.getFrameNames())
  tool_center = next(
      (n for n in (prefix + "ur_gripper_center", prefix + "ur_vacuum",
                   prefix + "ur_ee_marker", prefix + "ur_ee_link")
       if n in names),
      None,
  )
  assert tool_center is not None, f"no tool-center frame found in {gripper_path}"

  # rest transforms: the EE root is at the origin right after addFile, so a frame's
  # world pose equals its pose relative to the root.
  gc_pose = np.asarray(C.getFrame(tool_center).getPose())
  base_rel_tool = _pose_invert(gc_pose)
  ee_pose = np.asarray(C.getFrame(prefix + "ur_ee_link").getPose())
  approach_axis = _quat_to_R(gc_pose[3:]).T @ (gc_pose[:3] - ee_pose[:3])
  q_correction = _align_quat(approach_axis, np.array([0.0, 0.0, 1.0]))

  tool_world = np.concatenate([target_pose[:3], _quat_mul(target_pose[3:], q_correction)])
  base_world = _pose_compose(tool_world, base_rel_tool)
  root = C.getFrame(prefix + "ur_base")
  root.setPosition(base_world[:3])
  root.setQuaternion(base_world[3:])
  return tool_center


def compute_cube_grasps(assembly, subassembly, offset=0.01, cube_size=0.05, gripper_path=None):
  # placeholder grasp planner: assume every part in the subassembly is a cube and
  # generate one grasp per face (6 total), then keep only the collision-free ones.
  # To be replaced by a real grasp planner.
  #
  # offset: standoff distance from the cube face to the grasp origin, along the face
  #         normal. `offset` is the tunable variable requested.
  # gripper_path: floating EE used for the collision filter (default: vacuum).
  #
  # a grasp is just a relative pose (gripper w.r.t. the object), [x,y,z, qw,qx,qy,qz].
  # every grasp is expressed in the frame of the first obj in the subassembly, so the
  # whole subassembly shares one reference frame (parts are axis-aligned, so only the
  # translation between parts matters).

  if gripper_path is None:
    gripper_path = os.path.join(
        os.path.dirname(__file__), "src/rai/ur10/ur10_vacuum_floating.g"
    )

  # outward unit normal of each cube face, in the object frame
  face_normals = [
      np.array([1.0, 0.0, 0.0]),
      np.array([-1.0, 0.0, 0.0]),
      np.array([0.0, 1.0, 0.0]),
      np.array([0.0, -1.0, 0.0]),
      np.array([0.0, 0.0, 1.0]),
      np.array([0.0, 0.0, -1.0]),
  ]

  def quat_from_rotation(R):
    # rotation matrix (columns = grasp-frame axes in object frame) -> [w, x, y, z] quaternion (rai convention)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
      s = np.sqrt(tr + 1.0) * 2.0
      w = 0.25 * s
      x = (R[2, 1] - R[1, 2]) / s
      y = (R[0, 2] - R[2, 0]) / s
      z = (R[1, 0] - R[0, 1]) / s
    elif m00 > m11 and m00 > m22:
      s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
      w = (R[2, 1] - R[1, 2]) / s
      x = 0.25 * s
      y = (R[0, 1] + R[1, 0]) / s
      z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
      s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
      w = (R[0, 2] - R[2, 0]) / s
      x = (R[0, 1] + R[1, 0]) / s
      y = 0.25 * s
      z = (R[1, 2] + R[2, 1]) / s
    else:
      s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
      w = (R[1, 0] - R[0, 1]) / s
      x = (R[0, 2] + R[2, 0]) / s
      y = (R[1, 2] + R[2, 1]) / s
      z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

  def grasp_quat_for_normal(normal):
    # build a grasp frame whose z-axis (approach direction) points from the standoff
    # back into the cube, i.e. along -normal. x/y are chosen orthonormal, deterministically.
    z = -normal / np.linalg.norm(normal)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return quat_from_rotation(R)

  standoff = cube_size / 2.0 + offset
  ref_pos = assembly.position_in_base(subassembly[0])

  # collision-check scene: the subassembly parts, in the ref frame (the frame the
  # grasps live in). We copy a fresh config per query, because rai's collision result
  # is cached after the first query on a config and does not refresh on a frame move.
  base_scene = make_subassembly_scene(assembly, subassembly, cube_size)

  def grasp_is_collision_free(grasp_pose, grasped_obj):
    # place the floating EE so its tool center hits the grasp, then check the EE
    # against every part except the one being grasped (touching it is allowed).
    C_col = ry.Config()
    C_col.addConfigurationCopy(base_scene)
    place_floating_ee(C_col, grasp_pose, gripper_path)
    C_col.getFrame(grasped_obj).setContact(0)
    return C_col.getCollisionFree()

  grasps = []
  for obj in subassembly:
    # obj origin expressed in the reference (first obj) frame
    obj_in_ref = assembly.position_in_base(obj) - ref_pos
    # we assume that all objs that are part of the subassembly are cubes -> compute the 6 grasps, and filter after
    for normal in face_normals:
      pos = obj_in_ref + standoff * normal
      grasp = np.concatenate([pos, grasp_quat_for_normal(normal)])

      if grasp_is_collision_free(grasp, obj):
        grasps.append(grasp)

  return grasps

def compute_assembly_grasps_and_wrenches(C, assembly):
  # make all intermediate assemblies
  subassemblies = get_subassemblies(assembly)
  # show_subassemblies(C, assembly, subassemblies, [0, 0, 0.5])

  # compute grasps: for now, look at special cases of cubes
  grasps = {}

  for subassembly in subassemblies:
    grasps[subassembly] = compute_cube_grasps(assembly, subassembly)

  return grasps

def visualize_grasps(C, grasps, assembly, subassembly_base_pose, gripper_path=None):
  # for each subassembly, show its candidate grasps by placing a free-floating
  # gripper (the first robot's end effector) at every grasp pose.
  # grasps: {subassembly (tuple): [pose, ...]}, each pose in the first obj's frame.

  for subassembly, subassembly_grasps in grasps.items():
    C_viz = ry.Config()
    C_viz.addConfigurationCopy(C)

    # build the subassembly at base_pose (inspired by show_subassemblies)
    for obj, parent in assembly.assembly_relative_poses.items():
      if obj in subassembly and parent[0] in subassembly:
        C_viz.attach(parent[0], obj)
        C_viz.getFrame(obj).setRelativePosition(parent[1])

    # grasps are expressed in the first obj's frame; move it to base_pose
    ref = subassembly[0]
    C_viz.getFrame(ref).setPosition(subassembly_base_pose)
    ref_pose = np.asarray(C_viz.getFrame(ref).getPose())

    # drop a free-floating gripper at each grasp pose (grasps are in the ref frame)
    for i, grasp in enumerate(subassembly_grasps):
      grasp_world = _pose_compose(ref_pose, np.asarray(grasp))
      place_floating_ee(C_viz, grasp_world, gripper_path, prefix=f"grasp{i}_")

    C_viz.view(True)

class NodeId:
    def __init__(self):
      self.kinematic_tree = {}
      self.grasps = []
      self.supports = []
      self.finished_assembly_steps = []

class Node:
  id: NodeId
  cost: float

def picks(s):
  pass

def places(s):
  pass

def add_grasps(s):
  pass

def releases(s):
  pass

def set_down(s):
  pass

def assembles(s):
  pass

def successors(s: Node):
    yield from picks(s)        
    yield from places(s)       
    yield from add_grasps(s)   
    yield from set_downs(s)   
    yield from releases(s)     
                               
    yield from assembles(s)

def feasible(action, s0, s1):
  # collision free floating ees
  # reachable -> IK
  # stable
  pass

def solve_assembly_problem(C, assembly):
  grasps = compute_assembly_grasps_and_wrenches(C, assembly)

  visualize_grasps(C, grasps, assembly, [0, 0, 0.5])

  queue = []
  parents = {}

  # search
  while queue:
    state = queue.pop()

    if done(state):
      plan = reconstruct_plan()

      feasible = validate(plan)

      if feasible:
        return plan

      blacklist(plan)
    
    for a, s1 in successors(state):
      if not feasible(a, state, s1):
        continue

      new_cost = old_cost + cost(a)
      
      # TODO: add to queue

  pass
