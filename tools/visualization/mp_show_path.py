"""Plan a motion for the EXPORTED problem and show the resulting PATH in mr_bench's own viser
viewer (rai_env.display_path_viser). Defaults to the DEPENDENCY-GRAPH problem so the planner is
free to reorder independent actions.

    python3 tools/visualization/mp_show_path.py
    python3 tools/visualization/mp_show_path.py car-mini --robots 4
    python3 tools/visualization/mp_show_path.py cross --parts 7 --witness dp-ee
    python3 tools/visualization/mp_show_path.py --sequence --optimize --runtime 20
    python3 tools/visualization/mp_show_path.py --witness none
    python3 tools/visualization/mp_show_path.py --no-shortcut --no-serve

`--runtime` bounds ONLY the BiRRT search; shortcutting is a separate, un-timed post-process
(cap with --shortcut-iters / --shortcut-res, or skip with --no-shortcut). Blocks while serving.
Not for background runs.
"""
import argparse
from pathlib import Path
import sys
import warnings

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

warnings.simplefilter("ignore")
import numpy as np, assembly_planner.assembly as A, assembly_planner.solver as S, assembly_planner.problem as P, assembly_planner.wrench as wrench, assembly_planner.mp_export as MP
from assembly_planner.env import make_four_robot_env

pa = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
pa.add_argument("object", nargs="?", default="cross",
                help="assembly kind: cross, car-mini, cube-plates, ... (default: cross)")
pa.add_argument("--parts", type=int, default=5, help="number of parts (default: 5)")
pa.add_argument("--robots", type=int, default=3, help="number of robots (default: 3)")
pa.add_argument("--robot-type", default="ur10_limited",
                help="ur10 | ur10_limited | ur10_vacuum ... (default: ur10_limited)")
pa.add_argument("--witness", default="ee",
                choices=["ee", "joint", "dp-joint", "dp-ee", "none"],
                help="witness POST-optimization mode; 'none' = raw search witnesses (default: ee)")
pa.add_argument("--witness-clearance", type=float, default=0.0,
                help="robot-robot clearance weight in witness optimization (default: 0.0)")
pa.add_argument("--sequence", action="store_true",
                help="fixed-sequence problem instead of the dependency graph")
pa.add_argument("--optimize", action="store_true",
                help="RRT* cost optimization until --runtime (default: stop at first solution)")
pa.add_argument("--runtime", type=float, default=180.0, help="BiRRT search budget, s (default: 180)")
pa.add_argument("--no-shortcut", dest="shortcut", action="store_false", help="skip shortcutting")
pa.add_argument("--shortcut-iters", type=int, default=1000, help="shortcut iteration cap (default: 1000)")
pa.add_argument("--shortcut-res", type=float, default=None,
                help="shortcut collision resolution (default: env collision_resolution)")
pa.add_argument("--no-serve", dest="serve", action="store_false", help="plan only, no viewer")
pa.add_argument("--collision-view", action="store_true",
                help="show only PRIMITIVE (collision) shapes, hide file-loaded visual meshes")
pa.add_argument("--port", type=int, default=8080, help="viser port (default: 8080)")
args = pa.parse_args()

vacuum = "vacuum" in args.robot_type
A.set_seed(0); A.set_rolls_per_face(1 if vacuum else 2)
np.random.seed(0)
wrench.PARAMS["grasp_model"] = "asymmetric"
wrench.PARAMS["insert_force"]["_default"] = 20.0
asm = A.make_assembly(args.object, args.parts)
C = make_four_robot_env(num_robots=args.robots, floating_ee=False, vacuum=vacuum,
                        robot_types=args.robot_type)
A.add_disassembled_obj(C, asm, spacing=0.2)
robots = A.get_robots(C)
rik = A.RobotIK(C, robots); rik.ik_seeding = True
P.set_goal(on_table=False)
plan = A.solve_assembly_problem_milestones(C, asm, robots=robots, weight=5.0, verbose=False,
                                           wrench=True, robot_ik=rik, bracing=True)
gk = A.gripper_kind(C)
# witness post-optimization: 'none' -> sweeps=0 returns the raw (search) witnesses unchanged.
_wcost = "ee" if args.witness == "none" else args.witness
wit, wrep = S.optimize_witnesses(C, asm, plan, robots, rik, cost=_wcost, gripper_kind=gk,
                                 bracing=True, verbose=False,
                                 clear_weight=args.witness_clearance,
                                 sweeps=0 if args.witness == "none" else 2)
print(f"[witness] {args.witness} (clearance {args.witness_clearance}): "
      f"before {wrep['before']:.3f} after {wrep['after']:.3f}")
prob = MP.make_problem(C, asm, plan, wit, robots, rik, gk, bracing=True,
                       depgraph=not args.sequence)
print(f"[built] {args.object} x{args.parts}, {args.robots} robots ({args.robot_type}), "
      f"{'SEQUENCE' if args.sequence else 'DEPGRAPH'}, {len(prob.tasks)} tasks")

path, info = MP.run_birrt(prob, runtime=args.runtime, shortcut=args.shortcut,
                          shortcut_iters=args.shortcut_iters, shortcut_resolution=args.shortcut_res,
                          optimize=args.optimize, verbose=True)
if not path:
    print("[no path] nothing to show"); raise SystemExit(1)
print(f"[path] {len(path)} states, reaches terminal={prob.is_terminal_mode(path[-1].mode)}")
if args.serve:
    from multi_robot_multi_goal_planning.problems.util import interpolate_path
    disp = interpolate_path(path, 0.05, kind="euclidean")     # dense path for smooth playback
    print(f"[viser] serving interpolated path ({len(disp)} states) on "
          f"http://localhost:{args.port}  (Ctrl-C to stop)")
    prob.display_path_viser(disp, port=args.port, primitives_only=args.collision_view)
