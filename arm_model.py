"""Generate an ARTICULATED MuJoCo arm from rai's own kinematics.

WHY GENERATE RATHER THAN VENDOR A URDF. The planner's every guarantee -- witnesses, strict
collision gates, seat tolerances -- is expressed in rai's joint space. If MuJoCo's kinematics
disagree even slightly, the same q means a different end-effector pose in the two and every
witness silently lands somewhere else. Hunting for a URDF that happens to match rai's `ur10.g`
(its own offsets, its own axis conventions) makes that agreement an empirical question you can
only ever spot-check. Reading rai's tree instead makes it true BY CONSTRUCTION -- and it is
robot-agnostic for free: point it at a ur5e config and you get a ur5e model, no porting.

WHAT IT REPLACES. sim.py currently mirrors each arm as N DISCONNECTED `mocap="true"` bodies,
every one teleported to its rai-FK world pose each step -- there are no joints between links at
all. This emits the real chain: one body per rai frame, a `<joint>` wherever rai has one, so
the model can be driven by SETTING qpos (kinematic, still perfectly stiff -- the intermediate
step) and later by actuators without changing the model.

rai -> MJCF conventions relied on here, all verified in fk_compare.py rather than assumed:
  * both use quaternions as [w, x, y, z], so relative quats copy across verbatim;
  * a rai `hingeX/Y/Z` frame rotates about that local axis, which is exactly a MuJoCo
    `<joint type="hinge" axis="...">` placed at the body origin;
  * a rai joint frame's relative transform CONTAINS the current joint value, so the tree must
    be read at q = 0 to recover the fixed mounting offset (see `_zeroed`).
"""
import os
import numpy as np
import robotic as ry

# rai joint type -> the local axis MuJoCo should hinge about.
HINGE_AXES = {"hingeX": (1, 0, 0), "hingeY": (0, 1, 0), "hingeZ": (0, 0, 1)}
# Frames rai marks as these carry no geometry we can mirror; they still become bodies so the
# FK comparison can check them, they simply get no <geom>.
_NO_GEOM = (None, "marker", "mesh")


def _zeroed(C):
  """A copy of `C` with every joint at 0, because a rai joint frame's relative transform has
  the CURRENT joint value baked into it -- read the tree at home and the mounting offsets come
  out rotated by whatever pose the arm happened to be in."""
  Z = ry.Config()
  Z.addConfigurationCopy(C)
  Z.setJointState(np.zeros(len(Z.getJointState())))
  return Z


def arm_frames(C, prefix):
  """(root_frame, {parent: [children]}) for the sub-tree belonging to robot `prefix`.
  The root is the arm frame whose parent lies OUTSIDE the arm (rai mounts it on the table)."""
  names = [n for n in C.getFrameNames() if n.startswith(prefix)]
  owned = set(names)
  kids, root = {}, None
  for n in names:
    p = C.getFrame(n).getParent()
    pname = p.name if p is not None else None
    if pname in owned:
      kids.setdefault(pname, []).append(n)
    else:
      if root is not None:
        raise ValueError(f"{prefix}: two roots ({root}, {n}); expected one arm base")
      root = n
  return root, kids


def joint_of(C, name):
  """(rai joint type, axis, (lo, hi)) if `name` is an actuated hinge, else None. rai's
  `limits` is a longer vector (limits + ctrl limits); only the first two are the range."""
  info = C.getFrame(name).info()
  jt = info.get("joint")
  if jt not in HINGE_AXES:
    return None                                              # 'rigid' / None / unsupported
  lim = np.asarray(info.get("limits", []), dtype=float).ravel()
  rng = (float(lim[0]), float(lim[1])) if lim.size >= 2 and lim[0] < lim[1] else None
  return jt, HINGE_AXES[jt], rng


_MESH_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_meshes")


_ARM_COLORS = ["0.2 0.2 0.9 1", "0.9 0.2 0.2 1", "0.2 0.8 0.8 1", "0.9 0.7 0.1 1",
               "0.6 0.2 0.8 1", "0.4 0.9 0.3 1"]           # per-robot tint for the capsules


def _write_obj(name, V, T):
  """Cache a mesh as an .obj (MuJoCo has no .ply decoder), return its abs path. Written once."""
  os.makedirs(_MESH_CACHE_DIR, exist_ok=True)
  path = os.path.join(_MESH_CACHE_DIR, name + ".obj")
  if not os.path.exists(path):
    with open(path, "w") as f:
      f.write("".join(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n" for v in V))
      f.write("".join(f"f {int(t[0])+1} {int(t[1])+1} {int(t[2])+1}\n" for t in T))
  return path


def link_meshes(C, frame):
  """Split a link's visual mesh by VERTEX COLOR into single-color sub-meshes -- MuJoCo geoms
  take one rgba and its .obj loader ignores per-vertex colors, so this is how the real UR look
  (blue caps / grey body / dark joints) is preserved. Returns [(obj_path, rgba_str), ...], one
  per distinct region color; falls back to a single light-grey mesh if colors are missing."""
  try:
    V = np.asarray(C.getFrame(frame).getMeshPoints(), dtype=float)
    T = np.asarray(C.getFrame(frame).getMeshTriangles()).astype(int)
  except Exception:
    return []
  # SHAPE-CHECK, not len(): a frame rai marks as `mesh` may carry no mesh at all (joint-origin
  # helpers), and np.asarray of that is 0-d, where len() raises TypeError rather than returning
  # 0. The articulated path walks EVERY mesh frame, so it meets those; sim's mocap path never
  # did and the old len() guard looked sufficient for years.
  if V.ndim != 2 or T.ndim != 2 or V.shape[0] < 4 or T.shape[0] == 0:
    return []                                              # joint-origin helper meshes: skip
  try:
    cols = np.asarray(C.getFrame(frame).getMeshColors())[:, :3].astype(int)
  except Exception:
    cols = None
  if cols is None or len(cols) != len(V):
    return [(_write_obj(frame, V, T), "0.72 0.72 0.75 1")]
  tri_col = cols[T[:, 0]]                                  # triangle color = its 1st vertex
  out = []
  for k, c in enumerate(np.unique(tri_col, axis=0)):
    faces = T[np.all(tri_col == c, axis=1)]
    used = np.unique(faces)
    if len(used) < 4:
      continue
    remap = -np.ones(int(used.max()) + 1, dtype=int)
    remap[used] = np.arange(len(used))
    obj = _write_obj(f"{frame}__c{k}", V[used], remap[faces])
    out.append((obj, f"{c[0]/255:.3f} {c[1]/255:.3f} {c[2]/255:.3f} 1"))
  return out


def _geom_xml(kind, size, rgba, group, contype, conaffinity="1"):
  """MJCF for one rai shape. rai capsule/cylinder size = [length, radius]; MJCF wants
  'radius half-length'. Mirrors sim._arm_geom_xml so the two models collide identically."""
  s = np.asarray(size, dtype=float).ravel()
  common = (f'rgba="{rgba}" group="{group}" contype="{contype}" '
            f'conaffinity="{conaffinity}"')
  if kind in ("capsule", "cylinder"):
    return f'<geom type="{kind}" size="{s[1]} {s[0] / 2}" {common}/>'
  if kind == "sphere":
    return f'<geom type="sphere" size="{s[0]}" {common}/>'
  if kind in ("box", "ssBox"):
    return f'<geom type="box" size="{s[0] / 2} {s[1] / 2} {s[2] / 2}" {common}/>'
  return None


def collides_like_sim(name):
  """sim.py's rule: rai's `_coll*` capsules AND the gripper primitives (palm, fingers,
  gripper_fill, vacuum tip) are the collision model -- those DO collide in rai and show up
  as colliding pairs in the oracle. Emitting the gripper primitives visual-only let the
  executed gripper pass through parts with no physics response (user-caught: palm 28mm into
  a leg, finger2 25mm into another palm)."""
  if "_coll" in name:
    return True
  return any(k in name for k in ("palm", "finger", "gripper_fill", "vacuum"))


def build_arm_mjcf(C, prefix, collide=True, rgba="0.6 0.6 0.65 1", group=2, indent=4,
                   collide_fn=None, contype="2", conaffinity="1", link_mass=1e-3,
                   link_inertia=1e-6, armature=0.0, limits=True, static_mass=None,
                   meshes=False, assets=None, mesh_group=1):
  """MJCF <body> tree for one articulated arm, read out of rai at q=0.

  Returns (xml, joint_names) with joint_names in MuJoCo's own qpos order for this arm -- the
  caller needs that to map a rai joint vector onto qpos (rai's global joint order is not
  necessarily the per-arm tree order).

  Every rai frame becomes a body, including shapeless ones, so fk_compare can check the whole
  tree rather than just the links that happen to carry geometry. Bodies get a token inertial:
  MuJoCo rejects a dof-carrying body with zero mass. For a PURE FK check the value is
  irrelevant, but a kinematically driven arm is still integrated by mj_step, so a link must
  DOMINATE anything welded to it -- with the 1e-3 default a 0.5 kg part welded to a 1 g
  gripper made the constraint solver explode (measured: parts thrown 258 m). Pass a link_mass
  that swamps the payload; the value stops being a free choice only when actuators arrive."""
  Z = _zeroed(C)
  root, kids = arm_frames(Z, prefix)
  jnames, out = [], []

  def emit(name, depth):
    f = Z.getFrame(name)
    pad = " " * (indent + 2 * depth)
    if depth == 0:                                           # root: place it in world coords
      pos, quat = np.asarray(f.getPosition(), float), np.asarray(f.getQuaternion(), float)
    else:
      pos, quat = (np.asarray(f.getRelativePosition(), float),
                   np.asarray(f.getRelativeQuaternion(), float))
    out.append(f'{pad}<body name="{name}" pos="{pos[0]:.9g} {pos[1]:.9g} {pos[2]:.9g}" '
               f'quat="{quat[0]:.9g} {quat[1]:.9g} {quat[2]:.9g} {quat[3]:.9g}">')
    j = joint_of(Z, name)
    if j is not None:
      _jt, axis, rng = j
      # ENFORCING LIMITS ON A KINEMATICALLY DRIVEN ARM IS ACTIVELY HARMFUL. qpos comes from the
      # planner, which already respects rai's limits, but MuJoCo still turns each range into a
      # LIMIT_JOINT constraint -- and at the home pose FOUR of them sit ACTIVE. Those compete
      # with the grasp weld in the solver's LCP and the weld's force is zeroed: MEASURED, MuJoCo
      # reported a 24mm weld residual while applying 0N to close it, and no solref / link mass /
      # armature / iteration-count setting could move that. limited="false" -> 0.16mm.
      # Ranges are still written out (harmless, and actuators will want them) unless limits=False.
      on = "true" if limits else "false"
      lim = f' range="{rng[0]:.9g} {rng[1]:.9g}" limited="{on}"' if rng else ' limited="false"'
      # `armature` is added inertia ON THE DOF, not a controller. A huge value is what makes a
      # KINEMATICALLY driven chain behave: mj_step must still integrate the arm (the parts
      # need it), so without this the weld accelerates the chain, the next qpos write undoes
      # that, the weld sees a jump, and it diverges. Unaccelerable ~= mocap's free lunch.
      arm = f' armature="{armature:g}"' if armature else ""
      out.append(f'{pad}  <joint name="{name}" type="hinge" '
                 f'axis="{axis[0]} {axis[1]} {axis[2]}"{lim}{arm}/>')
      jnames.append(name)
    # MASS BELONGS ON THE MOVING LINKS, NOT ON EVERY FRAME. rai has ~69 frames per arm
    # (joint origins, collision capsules, finger meshes); giving each one `link_mass` builds a
    # ~276 kg arm with fictitious inertia. That is invisible while the arm is driven
    # kinematically, and fatal once actuators have to move it -- measured 3.6-6.5 rad of
    # tracking error that got WORSE with higher gain, i.e. divergence. Only DOF-carrying bodies
    # get real mass; the rigidly-attached rest get a token one.
    _m = link_mass if (j is not None or static_mass is None) else static_mass
    _i = link_inertia if (j is not None or static_mass is None) else static_mass
    out.append(f'{pad}  <inertial pos="0 0 0" mass="{_m:g}" '
               f'diaginertia="{_i:g} {_i:g} {_i:g}"/>')
    info = f.info()
    kind = info.get("shape")
    if kind not in _NO_GEOM:
      on = collide and (collide_fn(name) if collide_fn else True)
      g = _geom_xml(kind, info.get("size", []), rgba, group,
                    contype if on else "0", conaffinity if on else "0")
      if g:
        out.append(f'{pad}  {g}')
    elif kind == "mesh" and meshes:
      # THE REAL LINK MESHES, read out of rai exactly like the tree itself -- so a ur5e config
      # yields ur5e meshes with no porting, same guarantee as the FK. VISUAL ONLY
      # (contype/conaffinity 0, own group): the certified collision model stays rai's _coll*
      # capsules, so the physics is unchanged whether meshes are emitted or not. Split by VERTEX
      # COLOR because a MuJoCo geom carries a single rgba and its .obj loader drops per-vertex
      # colors -- that split is what keeps the real UR look (blue caps / grey body / dark
      # joints) instead of one flat grey arm.
      for _k, (_obj, _mrgba) in enumerate(link_meshes(Z, name)):
        _mn = f"vm_{name}__{_k}"
        if assets is not None:
          assets.append(f'<mesh name="{_mn}" file="{_obj}"/>')
        out.append(f'{pad}  <geom type="mesh" mesh="{_mn}" rgba="{_mrgba}" '
                   f'group="{mesh_group}" contype="0" conaffinity="0"/>')
    for c in sorted(kids.get(name, [])):
      emit(c, depth + 1)
    out.append(f'{pad}</body>')

  emit(root, 0)
  return "\n".join(out), jnames


def build_arms_mjcf(C, prefixes, **kw):
  """MJCF for several arms plus the combined joint-name order."""
  xml, names = [], []
  for p in prefixes:
    x, jn = build_arm_mjcf(C, p, **kw)
    xml.append(x)
    names.extend(jn)
  return "\n".join(xml), names


def standalone_model(C, prefixes, timestep=0.002, **kw):
  """A complete, self-contained MJCF holding only the arms -- what fk_compare loads. Kept
  separate from sim.py's full world so the kinematics can be checked without parts, table,
  welds or contacts confusing the picture."""
  _assets = []
  kw.setdefault("assets", _assets)
  body, jnames = build_arms_mjcf(C, prefixes, **kw)
  asset = ("" if not kw["assets"] else
           "\n  <asset>" + "".join(kw["assets"]) + "</asset>")
  xml = f"""<mujoco model="rai_arms">
  <compiler angle="radian"/>
  <option timestep="{timestep}" gravity="0 0 0"/>{asset}
  <worldbody>
{body}
  </worldbody>
</mujoco>"""
  return xml, jnames


# =============================================================================================
# ACTUATION -- the step after kinematic set-pose.
#
# A MuJoCo `<position>` actuator IS a PD servo (kp on position error, kv on velocity), computed
# inside the solver every step. That is simpler than a hand-written PID in Python AND better
# behaved: no integral term to wind up over a long move, and the control law participates in
# the same constraint solve as the welds instead of fighting it from outside.
#
# Gravity droop -- the usual reason to reach for an I term -- is handled properly by a
# feedforward instead (see sim's gravity compensation): cancel qfrc_bias and the PD only has to
# reject the payload and tracking error, which makes the gains far less critical.
#
# FORCE RANGE COMES FROM RAI, NOT FROM A GUESS. rai's `ctrl_limits` on each joint carries the
# real UR10 numbers -- [vel_limit, ?, effort_limit] = 330/330/150/54/54/54 Nm and 2.16..3.2
# rad/s. Emitting those as `forcerange` is the first time the sim models what the arm can
# actually PUSH, which is the placeholder (`F_push`=100N) the wrench model has carried all
# along.
# =============================================================================================
def joint_ctrl_limits(C, name):
  """(velocity limit, effort limit) from rai's ctrl_limits, or (None, None)."""
  cl = C.getFrame(name).info().get("ctrl_limits")
  if cl is None:
    return None, None
  cl = np.asarray(cl, dtype=float).ravel()
  vel = float(cl[0]) if cl.size >= 1 and cl[0] > 0 else None
  eff = float(cl[2]) if cl.size >= 3 and cl[2] > 0 else None
  return vel, eff


def actuator_mjcf(C, joint_names, kp=3000.0, kv=None, indent=4, forcerange=True,
                  dampratio=1.0):
  """`<actuator>` block: one position servo per joint, with rai's own effort limits.

  DAMPING IS PER-JOINT, VIA `dampratio`, NOT A SHARED kv. A single kv is critical damping for
  ONE inertia only, and a UR10's joint inertia spans 42x (measured from MuJoCo's mass matrix:
  wrist_3 0.40 -> shoulder_lift 16.97). With kv = 2*sqrt(kp) that put zeta at 0.24 on
  shoulder_lift and 1.58 on wrist_3 -- the big joints rang after every stop (error still GREW
  after the reference stopped, 8.7 -> 11.0 mrad, 3 sign flips, 0.79s to settle against a 0.15s
  DWELL_T), so attaches fired mid-oscillation and the grasp weld froze the swing into the hand.
  MuJoCo's `dampratio` sets kv = 2*zeta*sqrt(kp*M) from the actuator's OWN effective inertia, so
  every joint lands on the ratio asked for and nothing needs retuning for a different arm.
  Pass `kv` to force an explicit shared value instead (the old behaviour).
  Returns (xml, actuator_names) with actuator names == joint names."""
  pad = " " * indent
  out, names = [], []
  for jn in joint_names:
    _vel, eff = joint_ctrl_limits(C, jn)
    fr = f' forcerange="{-eff:.9g} {eff:.9g}"' if (forcerange and eff) else ""
    damp = (f'dampratio="{dampratio:g}"' if kv is None else f'kv="{kv:g}"')
    out.append(f'{pad}<position name="{jn}" joint="{jn}" kp="{kp:g}" {damp}{fr}/>')
    names.append(jn)
  return "\n".join(out), names
