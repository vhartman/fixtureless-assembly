"""Event-driven execution of a persisted assembly task/motion plan.

This is independent of sim.execute_motion_plan. `compile` needs no simulator and
never resolves witnesses or IK. `run` consumes the compiled event sequence with a
simple synchronized path follower and measured skill completion. The calibrated
MuJoCo model/plant is reused through simulation.mujoco_backend.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from execution.program import (FreeMotion, Skill, compile_program,
                               load_program, save_program)


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('compile', 'run', 'run-program'):
        q = sub.add_parser(command)
        if command == 'run-program':
            q.add_argument('program', help='checksummed program from simv2 compile')
        else:
            q.add_argument('plan', help='saved task-plan JSON with motion_data')
            q.add_argument('--mp-path', required=True,
                           help='saved motion-path JSON with final mp_export')
            q.add_argument('--program-json', help='write the explicit execution program')
        if command != 'compile':
            q.add_argument('--record', help='MuJoCo replay NPZ, also saved on failure')
            q.add_argument('--report-json')
            q.add_argument('--physics-step', type=float, default=0.002)
            q.add_argument('--joint-rate', type=float, default=0.5)
            q.add_argument('--joint-accel', type=float, default=2.0,
                           help='maximum scalar progress acceleration along each saved path '
                                '(rad/s^2 equivalent); sharp geometric corners can still '
                                'change joint velocity abruptly')
            q.add_argument('--actuated', action='store_true',
                           help='bounded PD arm drive (reference baseline is kinematic arms)')
            q.add_argument('--contact-grasps', action='store_true',
                           help='physical two-finger contact, experimental')
            q.add_argument('--no-process-wrench', action='store_true')
            q.add_argument('--gui', action='store_true')
            q.add_argument('--no-visual', action='store_true')
            q.add_argument('--settle-tolerance', type=float, default=0.01)
            q.add_argument('--settle-timeout', type=float, default=1.5)
            q.add_argument('--mate-grasp-clearance', type=float, default=None,
                           help='minimum measured-grasp gripper/receiver gap at the seated '
                                'mate target (m); physical-contact mode defaults to +1 mm '
                                'to reject a hand blocked at the exact seated pose')
            q.add_argument('--skill-timeout', type=float, default=2.0)
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == 'run-program':
        program = load_program(args.program)
        print(f'[simv2] loaded checksummed execution program from {args.program}')
    else:
        program = compile_program(args.plan, args.mp_path)
        if args.program_json:
            save_program(args.program_json, program)
            print(f'[simv2] execution program -> {args.program_json}')
    motions = sum(isinstance(s, FreeMotion) for s in program.segments)
    skills = sum(isinstance(s, Skill) for s in program.segments)
    print(f'[simv2] program: {motions} path segments and {skills} explicit events '
          f'from {len(program.plan)} task actions')
    if args.command == 'compile':
        for s in program.segments:
            if isinstance(s, Skill):
                print(f'  task {s.task_id:3d} {s.kind:8s} action={s.action_index} '
                      f'robots={s.robots} frames={s.frames}')
        return 0
    if args.contact_grasps and not args.actuated:
        raise SystemExit('[simv2] physical contacts require --actuated')
    if args.contact_grasps and abs(args.physics_step - 0.002) > 1e-12:
        raise SystemExit('[simv2] physical finger contacts require a 2 ms physics step')
    from execution.runtime import ReferenceExecutor
    mate_grasp_clearance = args.mate_grasp_clearance
    if args.contact_grasps and mate_grasp_clearance is None:
        mate_grasp_clearance = 0.001
    executor = ReferenceExecutor(
        program, dt=args.physics_step, joint_rate=args.joint_rate,
        joint_accel=args.joint_accel,
        actuated=args.actuated, contact_grasps=args.contact_grasps,
        process_wrench=not args.no_process_wrench,
        record=bool(args.record), gui=args.gui, visual=not args.no_visual,
        settle_tolerance=args.settle_tolerance,
        settle_timeout=args.settle_timeout, skill_timeout=args.skill_timeout,
        mate_grasp_clearance=mate_grasp_clearance)
    report = executor.run()
    metadata = {'executor': 'simv2-event-lockstep',
                'plan_path': program.plan_path,
                'motion_path': program.motion_path, 'actuated': args.actuated,
                'contact_grasps': args.contact_grasps,
                'mate_grasp_clearance': mate_grasp_clearance,
                'process_wrench': not args.no_process_wrench,
                'physics_step': args.physics_step, 'joint_rate': args.joint_rate,
                'seat_tolerance': executor.seat_tolerance,
                'joint_accel': args.joint_accel}
    if args.record:
        executor.backend.save(args.record, report, metadata)
    if args.report_json:
        Path(args.report_json).write_text(json.dumps({
            'success': bool(report.success),
            'results': [repr(r) for r in report.results],
            'final_pose_err': report.final_pose_err,
            'grasp_wrench_metrics': report.grasp_wrench_metrics,
            'mate_preflight': executor.mate_preflight,
            'seat_clearance': executor.seat_clearance,
            'table_wrench_metrics': report.table_wrench_metrics,
            'metadata': metadata}, indent=2) + '\n')
    print(report.summary())
    return 0 if report.success else 1


if __name__ == '__main__':
    raise SystemExit(main())
