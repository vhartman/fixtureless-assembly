import robotic as ry
import os
import argparse

import numpy as np

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

def parse_args():
  parser = argparse.ArgumentParser(
      description="Spawn a multi-robot UR10 environment."
  )
  parser.add_argument(
      "--num-robots", type=int, default=4,
      help="Number of robots to spawn (max 6). Default: 4",
  )
  parser.add_argument(
      "--floating-ee", action="store_true",
      help="Use free-floating end-effectors instead of the full arm.",
  )
  parser.add_argument(
      "--vacuum", action="store_true",
      help="Use the vacuum end-effector instead of the two-finger gripper.",
  )
  parser.add_argument(
      "--robot-types", default="ur10",
      help="Robot type for all robots (e.g. ur10, ur5). Default: ur10",
  )
  parser.add_argument(
      "--perturb-start-pose", action="store_true",
      help="Randomly perturb the object start poses (kept collision free).",
  )
  parser.add_argument(
      "--no-view", dest="view", action="store_false",
      help="Do not open the interactive viewer.",
  )
  parser.add_argument(
      "--no-disp-final", dest="disp_final", action="store_false",
      help="Do not show the final desired result.",
  )
  return parser.parse_args()


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
  def __init__(self):
    self.objects = [f"obj_{i}" for i in range(7)]

    self.base = "obj_0"

    self.assembly_dependencies = {
      "obj_1": ["obj_0"],
      "obj_2": ["obj_1"],
      "obj_3": ["obj_2"],
      "obj_4": ["obj_3"],
      "obj_5": ["obj_4"],
      "obj_6": ["obj_5"]
    }

    o = 0.055
    self.assembly_relative_poses = {
      "obj_1": ["obj_0", np.array([0, 0, o])],
      "obj_2": ["obj_0", np.array([0, 0, -o])],
      "obj_3": ["obj_0", np.array([0, o, 0])],
      "obj_4": ["obj_0", np.array([0, -o, 0])],
      "obj_5": ["obj_0", np.array([o, 0, 0])],
      "obj_6": ["obj_0", np.array([-o, 0, 0])]
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

def compute_grasps_and_wrenches(C, assembly):
  # make all intermediate assemblies
  pass

def solve_assembly_problem(C, assembly):
  compute_grasps_and_wrenches(C, assembly)

  # search
  pass

def main():
  args = parse_args()

  C = make_four_robot_env(
      num_robots=args.num_robots,
      floating_ee=args.floating_ee,
      vacuum=args.vacuum,
      robot_types=args.robot_types,
  )

  C = add_disassembled_obj(C, perturb_start_pose=args.perturb_start_pose)

  if args.view:
    C.view(True)

  assembly = Assembly()

  if args.disp_final:
    display_final_assembly(C, assembly, [0, 0, 0.5])

  solve_assembly_problem(C, assembly)

if __name__ == "__main__":
  main()