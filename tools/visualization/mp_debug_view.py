"""Visual debug for the motion-planning export: step through the exported SEQUENCE modes in
the rai viewer, each posed at its task's goal keyframe, so scene-graph / collision issues
are visible. Run:  ./.venv/bin/python tools/visualization/mp_debug_view.py
(close each window to advance)."""
from pathlib import Path
import sys
import warnings

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.simplefilter('ignore')
import numpy as np, assembly_planner.assembly as A, assembly_planner.solver as S, assembly_planner.problem as P, assembly_planner.wrench as wrench, assembly_planner.mp_export as MP
from assembly_planner.env import make_four_robot_env

A.set_seed(0); A.set_rolls_per_face(2)
wrench.PARAMS['grasp_model'] = 'asymmetric'; wrench.PARAMS['insert_force']['_default'] = 20.0
asm = A.make_assembly('cross', 5)
C = make_four_robot_env(num_robots=3, floating_ee=False, vacuum=False, robot_types='ur10')
A.add_disassembled_obj(C, asm, spacing=0.2); robots = A.get_robots(C)
rik = A.RobotIK(C, robots); rik.ik_seeding = True
P.set_goal(on_table=False)
plan = A.solve_assembly_problem_milestones(C, asm, robots=robots, weight=5.0, verbose=False,
                                           wrench=True, robot_ik=rik, bracing=True)
gk = A.gripper_kind(C)
wit, _ = S.optimize_witnesses(C, asm, plan, robots, rik, cost='dp-ee', clear_weight=1.0,
                              gripper_kind=gk, bracing=True, verbose=False)
prob = MP.make_problem(C, asm, plan, wit, robots, rik, gk, bracing=True)

m = prob.start_mode
for step in range(len(prob.sequence)):
    ti = prob.get_current_seq_index(m); task = prob.tasks[prob.sequence[ti]]
    prob.set_to_mode(m)
    q = prob.C.getJointState()
    if getattr(task.goal, 'goal', None) is not None:
        g = np.asarray(task.goal.goal); off = 0
        for r in task.robots:
            idx = prob.robot_idx[r]; q[idx] = g[off:off + len(idx)]; off += len(idx)
    prob.C.setJointState(q)
    pen = prob.C.getCollisionsTotalPenetration()
    cols = [f"{a}x{b}:{d:.3f}" for a, b, d in prob.C.getCollisions() if d < -0.02][:4]
    prob.C.view(True, f"mode {step}: {task.name}  pen={pen:.3f}\n" + "\n".join(cols))
    try:
        nxt = prob.get_next_modes(prob.start_pos.from_flat(q), m)
    except Exception:
        break
    if not nxt:
        break
    m = nxt[0]
