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
  parser.add_argument(
      "--show-successors", action="store_true",
      help="Visualize a few successor states of the initial search state, then exit.",
  )
  parser.add_argument(
      "--parts", type=int, default=7,
      help="Number of parts in the assembly (clamped per type). Default: 7.",
  )
  parser.add_argument(
      "--assembly", choices=["cross", "cube", "arch", "arch-seq"], default="cross",
      help="Assembly to build: cross (cube cross, 7), cube (2x2x2 grid, 8), "
           "arch (two pillars + lintel, 7 -- known-flawed star model), "
           "arch-seq (arch shape built as a sequential chain, 7). Default: cross.",
  )
  parser.add_argument(
      "--solve", action="store_true",
      help="Run the A* assembly search and print the plan.",
  )
  parser.add_argument(
      "--canonical", action="store_true",
      help="Break robot symmetry in the search (lowest-index free robot). TEST-ONLY: "
           "unsound once robots have fixed bases -- emits a warning.",
  )
  parser.add_argument(
      "--weight", type=float, default=1.0,
      help="Weighted-A* factor f=g+w*h: 1=optimal, >1 faster & bounded-suboptimal, "
           "inf=greedy best-first. Use a large value (e.g. 5 or inf) for fast solutions.",
  )
  parser.add_argument(
      "--test-pick-ik", nargs="?", const="a1:obj_1:0", default=None,
      metavar="ROBOT:PART:GRASP",
      help="Pop up the real-robot pick IK for a (robot, part, grasp) and exit, e.g. "
           "--test-pick-ik a1:obj_1:0 (default a1:obj_1:0).",
  )
  parser.add_argument(
      "--test-assemble-ik", nargs="?", const="a1:obj_1:0", default=None,
      metavar="ROBOT:PART:GRASP",
      help="Pop up the real-robot assemble IK: robot holds PART at GRASP and mates it "
           "onto its manual parent, then exit, e.g. --test-assemble-ik a1:obj_1:0.",
  )
  parser.add_argument(
      "--test-assemble-held-ik", nargs="?", const="a1:obj_1:0:a2:1", default=None,
      metavar="ROBOT:PART:GRASP:TARGET_ROBOT:ONTO_GRASP",
      help="Like --test-assemble-ik but the target (manual parent) is HELD by TARGET_ROBOT "
           "at ONTO_GRASP, so both cubes are in the air and both arms co-optimize. "
           "E.g. --test-assemble-held-ik a1:obj_1:0:a2:1.",
  )
  parser.add_argument(
      "--test-place-ik", nargs="?", const="a1:obj_1:0:1:5", default=None,
      metavar="ROBOT:PART:GRASP:N_ASSEMBLED[:FACE]",
      help="Pop up the real-robot place IK: with N_ASSEMBLED parts already mated, ROBOT "
           "picks the subassembly by PART at GRASP and sets it on the table resting on "
           "placement FACE (0-5, 5=upright default), then exit. "
           "E.g. --test-place-ik a1:obj_1:0:1:0.",
  )
  parser.add_argument(
      "--real-ik", action="store_true",
      help="Run the search with real-robot IK in the feasibility loop (RobotIK), and "
           "replay the plan with the actual arm poses instead of floating-EE stand-ins.",
  )
  parser.add_argument(
      "--optimistic", action="store_true",
      help="Use the optimistic (Dogar-style) per-part search -- add one part at a time via "
           "its own small complete sub-search (reorient/hold/pick/mate/set-down), with exact "
           "fallback -- instead of the exact search. Much faster on the cases that blow up.",
  )
  parser.add_argument(
      "--declutter", action="store_true",
      help="With --optimistic: allow each per-part step to RELOCATE blocking loose parts "
           "(stow them aside on demand) instead of only touching that part and the assembly. "
           "Removes the 'nothing else is in the way' assumption.",
  )
  parser.add_argument(
      "--wrench", action="store_true",
      help="Turn on the static-equilibrium (stability) model: support-gate assemble/release "
           "with the insertion reaction, so sideways mates on a resting assembly need a hold "
           "(mating becomes perpendicular unless braced). Default off (topological baseline).",
  )
  parser.add_argument(
      "--no-handovers", dest="handovers", action="store_false",
      help="Disable the add_grasp/handover primitives (regrasp-in-air). On by default; "
           "turning it off gives a leaner reference search (no second-grasp branching).",
  )
  parser.add_argument(
      "--no-place", dest="allow_place", action="store_false",
      help="Forbid setting a held assembly back on the table: the search must hold it and "
           "regrasp in the air (handovers become mandatory -- the Dogar hold-and-work "
           "regime). Meaningful for the exact search; the optimistic search needs to set down.",
  )
  parser.add_argument(
      "--seed", type=int, default=0,
      help="RNG seed for numpy + rai (KOMO IK), for reproducible runs. Default: 0.",
  )
  return parser.parse_args()

def main():
  args = parse_args()
  set_seed(args.seed)                           # reproducible numpy + rai (KOMO) RNG
  # grasp rolls are a property of the gripper: the vacuum is symmetric about its suction
  # axis (1 grasp per face); the two-finger has two in-plane orientations (90 deg apart).
  set_rolls_per_face(1 if args.vacuum else 2)

  if args.test_pick_ik is not None:
    robot, part, grasp = args.test_pick_ik.split(":")
    visualize_pick_ik(part=part, grasp_id=int(grasp), robot=robot, n_parts=args.parts,
                      num_robots=args.num_robots, vacuum=args.vacuum)
    return

  if args.test_assemble_ik is not None:
    robot, part, grasp = args.test_assemble_ik.split(":")
    visualize_assemble_ik(part=part, grasp_id=int(grasp), robot=robot, n_parts=args.parts,
                          num_robots=args.num_robots, vacuum=args.vacuum)
    return

  if args.test_place_ik is not None:
    fields = args.test_place_ik.split(":")
    robot, part, grasp, n_asm = fields[:4]
    face = int(fields[4]) if len(fields) > 4 else 5
    visualize_place_ik(part=part, grasp_id=int(grasp), robot=robot,
                       n_assembled=int(n_asm), n_parts=args.parts,
                       num_robots=args.num_robots, vacuum=args.vacuum, face=face)
    return

  if args.test_assemble_held_ik is not None:
    robot, part, grasp, target_robot, onto_grasp = args.test_assemble_held_ik.split(":")
    visualize_assemble_ik(part=part, grasp_id=int(grasp), robot=robot,
                          target_robot=target_robot, onto_grasp=int(onto_grasp),
                          n_parts=args.parts, num_robots=args.num_robots, vacuum=args.vacuum)
    return

  C = make_four_robot_env(
      num_robots=args.num_robots,
      floating_ee=args.floating_ee,
      vacuum=args.vacuum,
      robot_types=args.robot_types,
  )

  assembly = make_assembly(args.assembly, args.parts)

  C = add_disassembled_obj(C, assembly, perturb_start_pose=args.perturb_start_pose)


  if args.disp_final:
    display_final_assembly(C, assembly, [0, 0, 0.5])
    
  if args.show_successors:
    visualize_successor_states(C, assembly)
    return

  if args.solve:
    robots = get_robots(C)
    robot_ik = RobotIK(C, robots) if args.real_ik else None
    if args.optimistic and args.declutter:
      plan = solve_assembly_problem_optimistic_declutter(
          C, assembly, robots=robots, robot_ik=robot_ik, weight=args.weight,
          wrench=args.wrench, handovers=args.handovers, allow_place=args.allow_place)
    elif args.optimistic:
      plan = solve_assembly_problem_optimistic(C, assembly, robots=robots,
                                               robot_ik=robot_ik, weight=args.weight,
                                               wrench=args.wrench, handovers=args.handovers,
                                               allow_place=args.allow_place)
    else:
      plan = solve_assembly_problem(C, assembly, robots=robots, canonical=args.canonical,
                                    weight=args.weight, robot_ik=robot_ik, wrench=args.wrench,
                                    handovers=args.handovers, allow_place=args.allow_place)
    if plan is not None:
      print(f"plan ({len(plan)} actions):")
      for step in plan:
        print("  ", step)
      if args.view:
        if args.real_ik:
          visualize_plan_ik(C, assembly, plan, robots=robots, robot_ik=robot_ik)
        else:
          visualize_plan(C, assembly, plan)
    return

  if args.view:
    C.view(True)

if __name__ == "__main__":
  main()