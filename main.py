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
      "--grasp-model", choices=["isotropic", "directional", "asymmetric"],
      default="isotropic",
      help="Grasp wrench-capability model (with --wrench): isotropic = the original "
           "direction-agnostic box (default); directional = weak-shear vacuum / weak-torque "
           "two-finger in the grasp frame; asymmetric = directional + push/pull asymmetry "
           "along the approach axis (pressing into the palm/cup is structural, pull-off is "
           "the weak suction/friction path).",
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
      "--anytime", type=float, nargs="?", const=60.0, default=None, metavar="SECONDS",
      help="Anytime mode (exact search): restarting weighted A* -- a fast suboptimal plan "
           "first, then keep improving it (and possibly prove optimality) until the time "
           "budget runs out. Optional value = budget in seconds (default 60).",
  )
  parser.add_argument(
      "--anon-robots", action="store_true",
      help="LATE BINDING: search with a robot-anonymous state key -- robot-permutation "
           "states merge, so adding (idle) robots does not blow up the search. Sound for "
           "interchangeable floating grippers; with --real-ik treat the plan as a candidate.",
  )
  parser.add_argument(
      "--insert-force", type=float, default=10.0, metavar="N",
      help="Insertion (mating) reaction magnitude in newtons (with --wrench). Default 10. "
           "Sets what binds: e.g. two-finger caps are shear 15 / pull 25 / push 100, so at "
           "10 N held side-mates are free, at 30 N only press-aligned grasps survive.",
  )
  parser.add_argument(
      "--no-inserter-check", dest="inserter_check", action="store_false",
      help="With --wrench: skip the inserter-side check (whether the pressing arm's own "
           "grasp can transmit the seating reaction) -- equivalent to giving that grasp "
           "unbounded capability. The target-side stability check stays on.",
  )
  parser.add_argument(
      "--viser", action="store_true",
      help="Display via a viser web viewer (http://localhost:<port>) instead of the rai "
           "window: the whole replay is recorded and served with a step slider / playback "
           "controls (no per-step blocking windows).",
  )
  parser.add_argument(
      "--viser-port", type=int, default=8080,
      help="Port for the viser server (with --viser). Default: 8080.",
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
  import wrench
  wrench.PARAMS["grasp_model"] = args.grasp_model  # capability model (used when --wrench)
  wrench.PARAMS["check_inserter"] = args.inserter_check
  wrench.PARAMS["insert_force"]["_default"] = args.insert_force

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
    elif args.anytime is not None:
      plan, _cost, _opt = solve_assembly_problem_anytime(
          C, assembly, robots=robots, time_budget=args.anytime, robot_ik=robot_ik,
          wrench=args.wrench, handovers=args.handovers, allow_place=args.allow_place,
          anonymous=args.anon_robots)
    else:
      plan = solve_assembly_problem(C, assembly, robots=robots, canonical=args.canonical,
                                    weight=args.weight, robot_ik=robot_ik, wrench=args.wrench,
                                    handovers=args.handovers, allow_place=args.allow_place,
                                    anonymous=args.anon_robots)
    if plan is not None:
      print(f"plan ({len(plan)} actions):")
      for step in plan:
        print("  ", step)
      if args.view or args.viser:
        recorder = None
        if args.viser:
          from viser_view import ViserRecorder
          recorder = ViserRecorder()
        if args.real_ik:
          visualize_plan_ik(C, assembly, plan, robots=robots, robot_ik=robot_ik,
                            recorder=recorder)
        else:
          visualize_plan(C, assembly, plan, recorder=recorder)
        if recorder is not None:
          recorder.serve(port=args.viser_port)
    return

  if args.viser:
    from viser_view import ViserRecorder
    recorder = ViserRecorder()
    recorder.record(C, "scene")
    recorder.serve(port=args.viser_port)
  elif args.view:
    C.view(True)

if __name__ == "__main__":
  main()