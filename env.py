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

# Prefix used by each robot type's .g file.
ROBOT_PREFIXES = {"ur5": "ur_", "ur10": "ur_", "ur10_limited": "ur_", "kuka": "kuka_"}

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
        "pos": [[-0.3, 0.3, 0.03], [0.5, 0.3, 0.03],
                [0.5, -0.4, 0.03], [-0.3, -0.4, 0.03]],
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
    return os.path.join(HERE, "./src/rai/kuka_drake/kuka_two_finger.g")


def make_world():
    # the static environment shared by every scene: floor + table. Single source of
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
        place = layout["ur10" if robot_type == "ur10_limited" else robot_type]
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