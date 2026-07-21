"""Show the EXPORTED mr_bench problem's modes using mr_bench's OWN viewer
(scripts/show_problems.py :: visualize_modes). This is the authentic export-side view --
compare it against mp_viser_witness.py (the witness ground truth) to see the relinking gap.

    python mp_show_modes.py                 # viser (scrubbable), default cross 5 / 3 robots
    python mp_show_modes.py --rai           # mr_bench's blocking rai viewer instead
    python mp_show_modes.py cross 5 3

NOTE: visualize_modes poses SWITCHING robots at their task goal and every IDLE robot at HOME
(show_problems.py L92-106) -- so a part held by an idle arm is dragged to that arm's home pose.
That is mr_bench's native assumption and is exactly where our held-part poses diverge.
Blocks while serving. Not for background runs.
"""
import sys, warnings
warnings.simplefilter("ignore")
import assembly as A, solver as S, problem as P, wrench, mp_export as MP
from env import make_four_robot_env

kind = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "cross"
parts = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 5
R = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 3
use_viser = "--rai" not in sys.argv

A.set_seed(0); A.set_rolls_per_face(2)
wrench.PARAMS["grasp_model"] = "asymmetric"
wrench.PARAMS["insert_force"]["_default"] = 20.0
asm = A.make_assembly(kind, parts)
C = make_four_robot_env(num_robots=R, floating_ee=False, vacuum=False, robot_types="ur10")
A.add_disassembled_obj(C, asm, spacing=0.2)
robots = A.get_robots(C)
rik = A.RobotIK(C, robots); rik.ik_seeding = True
P.set_goal(on_table=False)
plan = A.solve_assembly_problem_milestones(C, asm, robots=robots, weight=5.0, verbose=False,
                                           wrench=True, robot_ik=rik, bracing=True)
gk = A.gripper_kind(C)
wit, _ = S.optimize_witnesses(C, asm, plan, robots, rik, cost="ee", gripper_kind=gk,
                              bracing=True, verbose=False)

MP._ensure_mrbench()                                          # path + stubs, then import the tool
import os
sys.path.insert(0, os.path.join(os.path.dirname(MP._MRBENCH_SRC), "scripts"))
from show_problems import visualize_modes

prob = MP.make_problem(C, asm, plan, wit, robots, rik, gk, bracing=True)
print(f"[modes] {len(prob.sequence)} tasks; {'viser' if use_viser else 'rai viewer'}")
visualize_modes(prob, use_viser=use_viser)
