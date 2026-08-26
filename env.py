import robotic as ry
import os

HERE = os.path.dirname(__file__)

# UR10 end-effector variants, keyed by (floating_ee, vacuum).
UR10_PATHS = {
    (False, False): "./src/rai/ur10/ur10_two_finger.g",
    (False, True): "./src/rai/ur10/ur10_vacuum.g",
    (True, False): "./src/rai/ur10/ur10_two_finger_floating.g",
    (True, True): "./src/rai/ur10/ur10_vacuum_floating.g",
}

# UR10 with a RESTRICTED workspace (tighter shoulder-lift/elbow limits). Only the REAL-arm
# variants differ (the two *_limited .g files override two joint limits); the FLOATING
# variants have no arm joints so the limits are moot -> reuse the standard ur10 floating.
UR10_LIMITED_PATHS = {
    (False, False): "./src/rai/ur10/ur10_two_finger_limited.g",
    (False, True): "./src/rai/ur10/ur10_vacuum_limited.g",
    (True, False): "./src/rai/ur10/ur10_two_finger_floating.g",
    (True, True): "./src/rai/ur10/ur10_vacuum_floating.g",
}

# Cheap-collision EE stand-in PAIRED with the actual robot model, keyed like UR10_PATHS
# by (floating_ee, vacuum) of the env. Real arms -> the *_wrist variant (the wrist-3 link
# exists on the real robot, so the cheap model carries its calibrated capsule, radius .035
# = an under-approximation that never rejects a true IK success); floating-EE envs -> the
# plain floating EE (no wrist exists in that model, so the cheap check must not invent one).
CHEAP_EE_PATHS = {
    (False, False): "./src/rai/ur10/ur10_two_finger_floating_wrist.g",
    (False, True): "./src/rai/ur10/ur10_vacuum_floating_wrist.g",
    (True, False): "./src/rai/ur10/ur10_two_finger_floating.g",
    (True, True): "./src/rai/ur10/ur10_vacuum_floating.g",
}

# UR5 cheap EE: the REAL-arm variants carry the ur5-CALIBRATED wrist-3 capsule (the ur10 wrist
# is placed for the ur10 arm and is wrong here). The FLOATING variants have no wrist to model,
# so they reuse the ur10 plain floating gripper (arm-independent). Only the wrist differs by arm.
UR5_CHEAP_EE_PATHS = {
    (False, False): "./src/rai/ur5/ur5_two_finger_floating_wrist.g",
    (False, True): "./src/rai/ur5/ur5_vacuum_floating_wrist.g",
    (True, False): "./src/rai/ur10/ur10_two_finger_floating.g",
    (True, True): "./src/rai/ur10/ur10_vacuum_floating.g",
}


def cheap_ee_path(is_ur5, floating, vacuum):
    """Cheap-collision EE model matching the arm: ur5 gets the ur5-calibrated wrist for the
    REAL arm, everything else the ur10 table (the floating cases coincide -- no arm/wrist)."""
    table = UR5_CHEAP_EE_PATHS if is_ur5 else CHEAP_EE_PATHS
    return os.path.join(HERE, table[(floating, vacuum)])

# UR5 EE variants, keyed by (floating_ee, vacuum). The REAL-ARM variants are the normal
# src/rai/ur5 model (ur5.g = arm + robotiq two-finger; ur5_vacuum.g = arm + vacuum pen). The
# FLOATING variants are arm-INDEPENDENT (a 6-DOF floating gripper, no arm), so they reuse the
# UR10 floating files -- a floating EE is the same whichever arm it stands in for.
UR5_PATHS = {
    (False, False): "./src/rai/ur5/ur5.g",
    (False, True): "./src/rai/ur5/ur5_vacuum.g",
    (True, False): "./src/rai/ur10/ur10_two_finger_floating.g",
    (True, True): "./src/rai/ur10/ur10_vacuum_floating.g",
}

# UR5 with a RESTRICTED workspace -- same convention as UR10_LIMITED_PATHS (the *_limited
# .g files override joint limits; the FLOATING variants have no arm joints -> reuse ur10).
UR5_LIMITED_PATHS = {
    (False, False): "./src/rai/ur5/ur5_two_finger_limited.g",
    (False, True): "./src/rai/ur5/ur5_vacuum_limited.g",
    (True, False): "./src/rai/ur10/ur10_two_finger_floating.g",
    (True, True): "./src/rai/ur10/ur10_vacuum_floating.g",
}

# Prefix used by each robot type's .g file.
ROBOT_PREFIXES = {"ur5": "ur_", "ur5_limited": "ur_", "ur10": "ur_",
                  "ur10_limited": "ur_", "kuka": "kuka_"}

# Placement on the table: {robot_type: (rotation, position)} per robot slot.
# Two layouts, one for <=4 robots and a tighter one for 5-6 robots.
_LEFT = [0.7071, 0, 0, -0.7071]
_RIGHT = [0.7071, 0, 0, 0.7071]

LAYOUT_4 = {
    "ur10": {
        "rot": [_LEFT, _LEFT, _RIGHT, _RIGHT],
        "pos": [[-0.5, 0.5, 0.03], [0.5, 0.5, 0.03],
                [0.5, -0.6, 0.03], [-0.5, -0.6, 0.03]],
    },
    "ur5": {
        "rot": [_LEFT, _LEFT, _RIGHT, _RIGHT],
        "pos": [[-0.3, 0.4, 0.03], [0.5, 0.4, 0.03],
                [0.5, -0.5, 0.03], [-0.3, -0.5, 0.03]],
    },
}

LAYOUT_6 = {
    "ur10": {
        "rot": [_LEFT, _LEFT, _RIGHT, _RIGHT, [1, 0, 0, 0], [0, 0, 0, 1]],
        "pos": [[-0.4, 0.5, 0.03], [0.4, 0.5, 0.03],
                [0.4, -0.6, 0.03], [-0.4, -0.6, 0.03],
                [-0.7, 0.0, 0.03], [0.7, 0.0, 0.03]],
    },
    "ur5": {
        "rot": [_LEFT, _LEFT, _RIGHT, _RIGHT, [0, 0, 0, 1], [1, 0, 0, 0]],
        "pos": [[-0.1, 0.3, 0.03], [0.3, 0.3, 0.03],
                [0.3, -0.4, 0.03], [-0.1, -0.4, 0.03],
                [0.6, -0.1, 0.03], [-0.4, -0.1, 0.03]],
    },
}


# distinctive per-robot gripper colors (a1, a2, ...) so robots are easy to follow in the
# viewer: red, green, blue, yellow, magenta, cyan.
GRIPPER_COLORS = [
    [0.90, 0.15, 0.15, 0.3],
    [0.15, 0.70, 0.20, 0.3],
    [0.20, 0.40, 0.90, 0.3],
    [0.95, 0.80, 0.10, 0.3],
    [0.80, 0.20, 0.80, 0.3],
    [0.10, 0.80, 0.80, 0.3],
]

# frame-name fragments that identify gripper bodies (two-finger palm/fingers, vacuum pen).
_GRIPPER_BODY_KEYS = ("vacuum", "finger", "palm", "gripper_fill", "gripper_center")


_TIGHT_FINGERS = False         # shrink the finger COLLISION proxy (see tighten_fingers)

# robotiq.g's finger proxy vs robotiq_tight.g's: the tight one halves the capsule RADIUS and
# shifts the capsule laterally by the difference, so the gripping SURFACE stays put (measured:
# inner surface 39.2mm -> 41.2mm from the tool axis) while the swept volume drops 4x.
_FAT_FINGER_RADIUS = 0.02
_TIGHT_FINGER_RADIUS = 0.01
_TIGHT_FINGER_SHIFT = 0.008


def set_tight_fingers(on):
    """Select robotiq_tight.g's slim finger proxy over robotiq.g's fat one. Call BEFORE any
    scene is built -- the geometry is baked into every C the run creates."""
    global _TIGHT_FINGERS
    _TIGHT_FINGERS = bool(on)


def tighten_fingers(C, prefix=""):
    """Shrink the two-finger COLLISION capsules in place to robotiq_tight.g's dimensions.

    The visible robotiq pads carry contact:0 -- they are decoration. All grasping and all
    collision run on finger1/finger2, capsules that robotiq.g sizes at [.04, .02]: 40mm long
    with a 20mm RADIUS, so 80mm end to end against a real pad of 37.5mm. That proxy reaches
    32.9mm past the tool centre, which is why the fingers poke out the far side of a small
    part and why a part must float off the table to be graspable from the side. ur5_vamp
    already includes the tight file; the ur10 line has it commented out.
    """
    n = 0
    for name in C.getFrameNames():
        if prefix and not name.startswith(prefix):
            continue
        if not (name.endswith("finger1") or name.endswith("finger2")):
            continue
        f = C.getFrame(name)
        try:
            size = list(f.getSize())
        except Exception:
            continue
        if len(size) != 2 or abs(size[1] - _FAT_FINGER_RADIUS) > 1e-9:
            continue                  # already slim, or not the robotiq.g capsule proxy
        rel = list(f.getRelativePosition())
        f.setShape(ry.ST.capsule, [float(size[0]), _TIGHT_FINGER_RADIUS])
        f.setRelativePosition([rel[0], rel[1] - _TIGHT_FINGER_SHIFT, rel[2]])
        f.setContact(-2)              # setShape drops the contact flag; restore the .g's value
        n += 1
    return n


def color_gripper(C, prefix, color):
    """Tint the gripper bodies of the robot with frame prefix `prefix` (e.g. 'a1_')."""
    for name in C.getFrameNames():
        if name.startswith(prefix) and any(k in name for k in _GRIPPER_BODY_KEYS):
            try:
                C.getFrame(name).setColor(color)
            except Exception:
                pass                      # shapeless frames (markers) may not take a color


def maybe_tighten_fingers(C, prefix=""):
    """tighten_fingers, gated on the flag. EVERY site that loads a gripper .g must call this:
    the cheap-EE collision scenes and the grasp precompute build their own Configs, and a
    finger proxy that differs between those and the real scene is exactly the model mismatch
    that makes a witness read feasible in the search and collide in motion planning."""
    if _TIGHT_FINGERS:
        return tighten_fingers(C, prefix)
    return 0


def _robot_path(robot_type, floating_ee, vacuum):
    if robot_type == "ur10":
        return os.path.join(HERE, UR10_PATHS[(floating_ee, vacuum)])
    if robot_type == "ur10_limited":
        return os.path.join(HERE, UR10_LIMITED_PATHS[(floating_ee, vacuum)])
    if robot_type == "ur5":
        return os.path.join(HERE, UR5_PATHS[(floating_ee, vacuum)])
    if robot_type == "ur5_limited":
        return os.path.join(HERE, UR5_LIMITED_PATHS[(floating_ee, vacuum)])
    return os.path.join(HERE, "./src/rai/kuka_drake/kuka_two_finger.g")


# --- FR3 duo (static humanoid-style pair) -----------------------------------------------
# Two Franka FR3 arms mounted on the franka_spine tower of the static TMR base
# (src/rai/fr3/static_mobile_fr3_base.g), exactly like mr_bench's make_static_fr3_duo_config:
# base on the floor at the -y edge facing +y into the build area, shoulders at z~0.50,
# arms tilted outward. Each arm carries the shared robotiq two-finger gripper at its
# flange (link8), and ur_* ALIAS frames are added so every pipeline convention (RobotIK
# tool-center/base discovery, get_robots a<i>_ prefixes, subtree joint restriction) works
# unmodified. No analytic IK for fr3 yet (ssik has a franka prebuilt -- calibration is
# future work); validation runs the relaxed-precheck + KOMO path. Pair with a CENTRAL
# initial layout (--random-init R / --grid-init N): the standard scatter line at y=0 spans
# beyond the FR3's ~0.85m reach from the -y edge.
FR3_DUO_BASE_PATH = "./src/rai/fr3/static_mobile_fr3_base.g"
FR3_ARM_PATHS = {"right": "./src/rai/fr3/right_fr3.g", "left": "./src/rai/fr3/left_fr3.g"}
ROBOTIQ_TIGHT_PATH = "./src/rai/robotiq/robotiq_tight.g"
OPENARM_TORSO_PATH = "./src/rai/openarm/openarm_torso.g"
OPENARM_ARM_PATHS = {"right": "./src/rai/openarm/right_openarm.g",
                     "left": "./src/rai/openarm/left_openarm.g"}


def make_fr3_duo_env(num_duos=1):
    """Humanoid-style static FR3 duos (robotiq two-finger) around a raised workbench.

    Each duo = TMR torso on the floor + two FR3s on its spine tower (right then left
    shoulder), exactly like mr_bench's make_static_fr3_duo_config. num_duos=2 places the
    second duo rotated 180deg on the opposite table edge (robots a3/a4). The table is
    raised to near-shoulder height ('human-like' bench) and sized so the torso hulls
    (contact-disabled decoration) clear it.
    """
    C = make_world()
    # OpenArm reach is only ~0.64m (vs FR3 0.85): the workbench sits IN the reach
    # envelope -- narrow (0.5m deep) slab close to the torso, top at 0.52 so top-down
    # grasps keep wrist headroom under the 0.72m shoulders.
    table_h = 0.06                                       # slim slab, top at 0.52
    C.getFrame("table").setShape(
        ry.ST.box, size=[1.6, 1.2, table_h, 0.005]).setPosition(
        [0.0, -0.10, 0.52 - table_h / 2.0]).setContact(1)
    duos = []
    for k in range(num_duos):
        y = -0.66 if k % 2 == 0 else 0.66
        yaw = 0.70710678 if k % 2 == 0 else -0.70710678  # rotZ +-90: face the table
        duos.append((k, y, [0.70710678, 0.0, 0.0, yaw]))
    for k, by, quat in duos:
        base = C.addFile(os.path.join(HERE, FR3_DUO_BASE_PATH),
                         namePrefix=f"fr3{k}_").setParent(C.getFrame("floor"))
        base.setPosition([0.0, by, 0.0]).setQuaternion(quat).setJoint(ry.JT.rigid)
        for j, side in enumerate(("right", "left")):
            r = f"a{2 * k + j + 1}"
            prefix = f"{r}_"
            sh_m = C.addFrame(f"{prefix}ur_base").setParent(
                C.getFrame(f"fr3{k}_{side}_base_joint_origin"))
            root = C.addFile(os.path.join(HERE, FR3_ARM_PATHS[side]), namePrefix=prefix)
            root.setParent(sh_m)
            root.setJoint(ry.JT.rigid)
            # explicit link colors: the fr3v2 DAEs carry no rai-side color (color=None),
            # so viser exports fall back to per-mesh defaults -> random-looking odd links.
            # Uniform franka white keeps every view (rai, viser, sim meshes) consistent.
            for li in range(8):
                ln = C.getFrame(f"{prefix}{side}_fr3v2_link{li}_0")
                if ln is not None:
                    ln.setColor([0.93, 0.93, 0.95, 1.0])
            link0 = C.getFrame(f"{prefix}{side}_fr3v2_link0")
            C.addFrame(f"{prefix}ur_base_link").setParent(link0)
            link8 = C.getFrame(f"{prefix}{side}_fr3v2_link8")
            C.addFrame(f"{prefix}ur_ee_link").setParent(link8)
            # gripper: robotiq_tight at the flange, rotated 90deg about the flange axis
            # (FR3 flange convention differs from the UR one -- the straight mount points
            # the fingers sideways)
            rg = C.addFile(os.path.join(HERE, ROBOTIQ_TIGHT_PATH), namePrefix=prefix)
            rg.setParent(link8).setRelativePosition([0.0, 0.0, 0.036]).setRelativeQuaternion(
                [0, 0, 0, 1]).setJoint(ry.JT.rigid)
            C.addFrame(f"{prefix}ur_gripper_center").setParent(
                C.getFrame(f"{prefix}gripper_center")).setShape(
                ry.ST.sphere, [0.004]).setContact(0)
            maybe_tighten_fingers(C, prefix)
            color_gripper(C, prefix, GRIPPER_COLORS[(k * 2 + j) % len(GRIPPER_COLORS)])
        # home pose: FR3's all-zero joints fold the wrist into the forearm once a gripper
        # is mounted -- pick the first candidate home config that is collision-free
        # import re as _re
        # homes = ([0.0, 0.55, 0.0, -1.15, 0.0, 0.45, 0.10],
        #          [0.0, -0.25, 0.0, -2.10, 0.0, 1.90, -0.80],
        #          [0.0, 0.30, 0.0, -2.20, 0.0, 1.60, -0.75],
        #          [0.0, 0.70, 0.0, -1.00, 0.0, 0.20, 0.15],
        #          [0.0, -0.10, 0.0, -1.80, 0.0, 1.50, -0.60])
        # jn = C.getJointNames()
        # pat = _re.compile(_re.escape(prefix) + side + r"_fr3v2_joint(\d)$")
        # idx = {int(pat.match(n).group(1)) - 1: i
        #        for i, n in enumerate(jn) if pat.match(n)}
        # for hq in homes:
        #     hq = list(hq)
        #     if prefix == "a1_":
        #         hq = [hq[0], -hq[1], hq[2], -hq[3], hq[4], -hq[5], -hq[6]]  # mirror
        #     full = list(C.getJointState())
        #     for bi, i in idx.items():
        #         full[i] = hq[bi]
        #     C.setJointState(full)
        #     if C.getCollisionsTotalPenetration() < 1e-6:
        #         break
        full = C.getJointState()
        full[5] = 1
        full[5 + 1 * 7] = 1
        if len(full) > 2 * 7:
            full[5 + 2 * 7] = 1
        if len(full) > 3 * 7:
            full[5 + 3 * 7] = 1
        C.setJointState(full)

    # torso hulls are decoration: raised table clears them anyway, but keep them out of
    # every collision query regardless (they overhang the table edge by design)
    for n in list(C.getFrameNames()):
        if n.endswith("base_coll") or n.endswith("tower_coll"):
            C.getFrame(n).setContact(0)
    return C


def make_openarm_duo_env(num_duos=1):
    """Humanoid-style static OpenArm v2 duos (pinch gripper, fingers deactivated open).

    Mirrors make_fr3_duo_env: OpenArm's own torso (body_link0) on the floor at the
    table edge facing the build area, both 7-DOF arms bolted at its shoulder mounts
    (z=0.698). Fingers are rigid at the open pose -- closing proxies come later via
    the same mechanism as tighten_fingers. num_duos=2 places the second duo rotated
    180deg on the opposite table edge (robots a3/a4).
    """
    C = make_world()
    # OpenArm reach is only ~0.64m (vs FR3 0.85): the workbench sits IN the reach
    # envelope -- narrow (0.5m deep) slab close to the torso, top at 0.52 so top-down
    # grasps keep wrist headroom under the 0.72m shoulders.
    table_h = 0.06                                       # slim slab, top at 0.52
    C.getFrame("table").setShape(
        ry.ST.box, size=[1.2, 0.8, table_h, 0.005]).setPosition(
        [0.0, -0.2, 0.37 - table_h / 2.0]).setContact(1)
    duos = []
    for k in range(num_duos):
        y = -0.66 if k % 2 == 0 else 0.26
        yaw = 0.70710678 if k % 2 == 0 else -0.70710678  # rotZ +-90: face the table
        duos.append((k, y, [0.70710678, 0.0, 0.0, yaw]))
    for k, by, quat in duos:
        torso = C.addFile(os.path.join(HERE, OPENARM_TORSO_PATH),
                          namePrefix=f"oa{k}_").setParent(C.getFrame("floor"))
        torso.setPosition([0.0, by, 0.02]).setQuaternion(quat).setJoint(ry.JT.rigid)
        # torso boxes are decoration (the arms bolt on right next to them):
        # exclude from every collision query, like the fr3 torso hulls
        for tn in list(C.getFrameNames()):
            if tn.startswith(f"oa{k}_torso_"):
                C.getFrame(tn).setContact(0)
        for j, side in enumerate(("right", "left")):
            r = f"a{2 * k + j + 1}"
            prefix = f"{r}_"
            mount = C.getFrame(f"oa{k}_{side}_mount_origin")
            # {r}_ur_base marks the single-arm IK subtree root (solver.py robot_base)
            sh_m = C.addFrame(f"{prefix}ur_base").setParent(mount)
            root = C.addFile(os.path.join(HERE, OPENARM_ARM_PATHS[side]),
                             namePrefix=prefix)
            root.setParent(sh_m)
            root.setJoint(ry.JT.rigid)
            base_body = C.getFrame(f"{prefix}openarm_{side}_base_link")
            C.addFrame(f"{prefix}ur_base_link").setParent(base_body)
            ee_flange = C.getFrame(f"{prefix}openarm_{side}_ee_base_link")
            C.addFrame(f"{prefix}ur_ee_link").setParent(ee_flange)
            # tool-center between the fingertips (approach = ee_base -z); tiny sphere
            # so mr_bench's delete_visual_only pass cannot prune it (shapeless frames
            # get deleted there and set_t then dereferences NULL -- FR3 lesson).
            gc = C.addFrame(f"{prefix}ur_gripper_center").setParent(
                C.getFrame(f"{prefix}gripper_center"))
            gc.setShape(ry.ST.sphere, [0.004]).setContact(0)
            color_gripper(C, prefix, GRIPPER_COLORS[(k * 2 + j) % len(GRIPPER_COLORS)])
            # home pose: sampled collision-free ready config (hands above table height);
            # applied per side HERE so every duo's arms map correctly (a hardcoded a1/a2
            # broke num_duos=2 -- a3/a4 silently kept file-default q).
            HOMES = {"right": [0, -0.2, 0, -1.2, 0., -0.5, 0.],
                     "left": [0, 0.2, 0, -1.2, 0., -0.5, 0.]}
            import re as _re
            jn = C.getJointNames()
            pat = _re.compile(_re.escape(prefix) + r"openarm_" + side + r"_joint(\d)$")
            idx = {int(pat.match(n).group(1)) - 1: i
                   for i, n in enumerate(jn) if pat.match(n)}
            full = list(C.getJointState())
            for bi, i in idx.items():
                full[i] = HOMES[side][bi]
            C.setJointState(full)

    return C


def make_world():
    # truth, so the real-robot env and the floating-EE collision scene agree exactly.
    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    C.addFrame("table").setPosition([0, 0, 0.2 - 0.06 / 2]).setShape(
        ry.ST.box, size=[3, 3, 0.06, 0.005]
    ).setColor([0.6, 0.6, 0.6]).setContact(1)

    return C


def make_four_robot_env(
    num_robots=4, floating_ee=False, vacuum=False, robot_types="ur10"
):
    assert num_robots <= 6, "A maximum of 6 robots are supported"

    if isinstance(robot_types, str):
        robot_types = [robot_types] * num_robots
    assert num_robots <= len(robot_types)

    # MIXED END EFFECTORS: `vacuum` may be a per-robot list (like robot_types), e.g.
    # [True, False, False] = a1 vacuum + a2/a3 two-finger. A scalar keeps the uniform
    # behaviour byte-identical.
    if isinstance(vacuum, (bool, int)):
        vacuum = [bool(vacuum)] * num_robots
    assert num_robots <= len(vacuum)

    C = make_world()

    layout = LAYOUT_6 if num_robots > 4 else LAYOUT_4

    for i in range(num_robots):
        robot_type = robot_types[i]
        layout_key = ("ur10" if robot_type == "ur10_limited"
                      else "ur5" if robot_type == "ur5_limited"
                      else robot_type)
        place = layout[layout_key]
        prefix = f"a{i + 1}_"

        C.addFile(
            _robot_path(robot_type, floating_ee, vacuum[i]), namePrefix=prefix
        ).setParent(C.getFrame("table")).setRelativePosition(
            place["pos"][i]
        ).setRelativeQuaternion(place["rot"][i]).setJoint(ry.JT.rigid)

        maybe_tighten_fingers(C, prefix)

        # unique gripper colour per robot, so they are easy to tell apart in the viewer
        color_gripper(C, prefix, GRIPPER_COLORS[i % len(GRIPPER_COLORS)])

    return C

def make_free_floating_env():
  return make_four_robot_env(floating_ee=True)