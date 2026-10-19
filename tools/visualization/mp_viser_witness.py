"""Ground-truth viser replay of a solved plan, driven DIRECTLY by the optimized witnesses
and OUR State.parent re-rooting (via visualize_plan_ik, witnesses=...). This is exactly the
scene the mr_bench export (mp_export.py) tries to mirror -- open both side by side to compare
how the relinking SHOULD look vs. what the export produces.

    python3 tools/visualization/mp_viser_witness.py
    python3 tools/visualization/mp_viser_witness.py cross 5 3 --port 8081
    python3 tools/visualization/mp_viser_witness.py --no-serve

Blocks while serving (viser). Not for background/benchmark runs.
"""
from pathlib import Path
import sys
import warnings

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

warnings.simplefilter("ignore")
import assembly_planner.assembly as A, assembly_planner.solver as S, assembly_planner.problem as P, assembly_planner.wrench as wrench
from assembly_planner.env import make_four_robot_env
from tools.visualization.viser_view import ViserRecorder

kind = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "cross"
parts = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 5
R = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 3
port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8080
serve = "--no-serve" not in sys.argv
cost = sys.argv[sys.argv.index("--cost") + 1] if "--cost" in sys.argv else "ee"

A.set_seed(0); A.set_rolls_per_face(2)
wrench.PARAMS["grasp_model"] = "asymmetric"
wrench.PARAMS["insert_force"]["_default"] = 20.0
asm = A.make_assembly(kind, parts)
C = make_four_robot_env(num_robots=R, floating_ee=False, vacuum=False, robot_types="ur10")
A.add_disassembled_obj(C, asm, spacing=0.2)
robots = A.get_robots(C)
rik = A.RobotIK(C, robots); rik.ik_seeding = True
P.set_goal(on_table=False)                                    # match linkdump / export scene

plan = A.solve_assembly_problem_milestones(C, asm, robots=robots, weight=5.0, verbose=False,
                                           wrench=True, robot_ik=rik, bracing=True)
print(f"[plan] {len(plan)} actions")
gk = A.gripper_kind(C)
wit, rep = S.optimize_witnesses(C, asm, plan, robots, rik, cost=cost, gripper_kind=gk,
                                bracing=True, verbose=False)
print(f"[witnesses] cost={cost} before {rep['before']:.3f} after {rep['after']:.3f}")

rec = ViserRecorder()
A.visualize_plan_ik(C, asm, plan, robots=robots, robot_ik=rik, recorder=rec,
                    bracing=True, witnesses=wit)
print(f"[recorded] {len(rec.steps)} steps (expected {len(plan) + 1})")
if serve:
    print(f"[viser] serving on http://localhost:{port}  (Ctrl-C to stop)")
    rec.serve(port=port)
