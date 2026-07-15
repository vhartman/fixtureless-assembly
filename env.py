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
ROBOT_PREFIXES = {"ur5": "ur_", "ur10": "ur_", "kuka": "kuka_"}

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


def color_gripper(C, prefix, color):
    """Tint the gripper bodies of the robot with frame prefix `prefix` (e.g. 'a1_')."""
    for name in C.getFrameNames():
        if name.startswith(prefix) and any(k in name for k in _GRIPPER_BODY_KEYS):
            try:
                C.getFrame(name).setColor(color)
            except Exception:
                pass                      # shapeless frames (markers) may not take a color


def _robot_path(robot_type, floating_ee, vacuum):
    if robot_type == "ur10":
        return os.path.join(HERE, UR10_PATHS[(floating_ee, vacuum)])
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

    C = make_world()

    layout = LAYOUT_6 if num_robots > 4 else LAYOUT_4

    for i in range(num_robots):
        robot_type = robot_types[i]
        place = layout[robot_type]
        prefix = f"a{i + 1}_"

        C.addFile(
            _robot_path(robot_type, floating_ee, vacuum), namePrefix=prefix
        ).setParent(C.getFrame("table")).setRelativePosition(
            place["pos"][i]
        ).setRelativeQuaternion(place["rot"][i]).setJoint(ry.JT.rigid)

        # unique gripper colour per robot, so they are easy to tell apart in the viewer
        color_gripper(C, prefix, GRIPPER_COLORS[i % len(GRIPPER_COLORS)])

    return C

def make_free_floating_env():
  return make_four_robot_env(floating_ee=True)