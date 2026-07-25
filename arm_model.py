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
  """sim.py's rule: only rai's `_coll*` capsules are the collision model. The other arm
  primitives (gripper_fill, vacuum tip) are decoration there, so an articulated arm must make
  the same choice or its contacts would not be comparable with every result measured so far."""
  return "_coll" in name


def build_arm_mjcf(C, prefix, collide=True, rgba="0.6 0.6 0.65 1", group=2, indent=4,
                   collide_fn=None, contype="2", conaffinity="1", link_mass=1e-3,
                   link_inertia=1e-6, armature=0.0, limits=True):
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
    out.append(f'{pad}  <inertial pos="0 0 0" mass="{link_mass:g}" '
               f'diaginertia="{link_inertia:g} {link_inertia:g} {link_inertia:g}"/>')
    info = f.info()
    kind = info.get("shape")
    if kind not in _NO_GEOM:
      on = collide and (collide_fn(name) if collide_fn else True)
      g = _geom_xml(kind, info.get("size", []), rgba, group,
                    contype if on else "0", conaffinity if on else "0")
      if g:
        out.append(f'{pad}  {g}')
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
  body, jnames = build_arms_mjcf(C, prefixes, **kw)
  xml = f"""<mujoco model="rai_arms">
  <compiler angle="radian"/>
  <option timestep="{timestep}" gravity="0 0 0"/>
  <worldbody>
{body}
  </worldbody>
</mujoco>"""
  return xml, jnames
