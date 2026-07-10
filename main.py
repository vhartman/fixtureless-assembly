import argparse

import numpy as np

from env import *
from assembly import *

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

  assembly = CubeCrossAssembly()

  if args.disp_final:
    display_final_assembly(C, assembly, [0, 0, 0.5])

  solve_assembly_problem(C, assembly)

if __name__ == "__main__":
  main()