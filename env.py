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


def _robot_path(robot_type, floating_ee, vacuum):
    if robot_type == "ur10":
        return os.path.join(HERE, UR10_PATHS[(floating_ee, vacuum)])
    if robot_type == "ur5":
        return os.path.join(HERE, "./src/rai/ur5_vamp/ur5_vamp.g")
    return os.path.join(HERE, "./src/rai/kuka_drake/kuka_two_finger.g")


def make_four_robot_env(
    num_robots=4, floating_ee=False, vacuum=False, robot_types="ur10"
):
    assert num_robots <= 6, "A maximum of 6 robots are supported"

    if isinstance(robot_types, str):
        robot_types = [robot_types] * num_robots
    assert num_robots <= len(robot_types)

    C = ry.Config()

    C.addFrame("floor").setPosition([0, 0, 0.0]).setShape(
        ry.ST.box, size=[20, 20, 0.02, 0.005]
    ).setColor([0.9, 0.9, 0.9]).setContact(0)

    C.addFrame("table").setPosition([0, 0, 0.2 - 0.06 / 2]).setShape(
        ry.ST.box, size=[3, 3, 0.06, 0.005]
    ).setColor([0.6, 0.6, 0.6]).setContact(1)

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

    return C

def make_free_floating_env():
  return make_four_robot_env(floating_ee=True)