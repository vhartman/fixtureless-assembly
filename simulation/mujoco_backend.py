"""Thin MuJoCo plant adapter. It does not infer tasks or schedule events.

The legacy sim module still owns the calibrated model export and low-level plant.
Only that stable boundary is reused here; execute_motion_plan and its schedulers
are never called.
"""
from __future__ import annotations

import numpy as np

import assembly_planner.assembly as A
import replay as R
import sim
from assembly_planner.plan_io import read_motion_data, read_plan, motion_data_states, read_grasp_closures
from execution.program import FreeMotion, Skill


class MujocoBackend:
    def __init__(self, program, *, dt=0.002, actuated=False, animate_grippers=True,
                 contact_grasps=False, record=False, gui=False, visual=True):
        import robotic as ry
        cfg, actions = read_plan(program.plan_path)
        if tuple(tuple(a) for a in actions) != program.plan:
            raise ValueError('task plan changed since execution-program compilation')
        md = read_motion_data(program.plan_path, actions,
                              grasp_db_hash=cfg.get('grasp_sampler_db_hash'))
        if md is None or md['sha256'] != program.motion_data_sha256:
            raise ValueError('saved symbolic state certificate changed')
        for segment in program.segments:
            if isinstance(segment, FreeMotion):
                if segment.supports != md['states'][segment.support_state]['supports']:
                    raise ValueError('motion segment support mode differs from saved plan')
            elif isinstance(segment, Skill):
                if (segment.supports_before != md['states'][segment.state_before]['supports']
                        or segment.supports_after != md['states'][segment.state_after]['supports']):
                    raise ValueError('skill support transition differs from saved plan')
        C, assembly = R.build_scene(cfg, require_artifact_contract=True, motion_data=md)
        sim._restore_saved_wrench_config(cfg)
        A.get_initial_state(C, assembly)
        states = motion_data_states(md, assembly)
        robots = A.get_robots(C)
        if tuple(robots) != program.robots:
            raise ValueError('scene robot roster differs from saved motion path')
        fk = ry.Config()
        fk.addConfigurationCopy(C)
        joint_names = tuple(fk.getJointNames())
        if joint_names != program.joints:
            raise ValueError('current scene joint order differs from saved motion path')
        self.ridx = {r: tuple(i for i, name in enumerate(joint_names)
                              if name.startswith(r)) for r in robots}
        self.flat_slices = {}
        cursor = 0
        for robot in robots:
            n = len(self.ridx[robot])
            self.flat_slices[robot] = slice(cursor, cursor + n)
            cursor += n
        if cursor != len(program.joints):
            raise ValueError('saved joint vector does not match robot joint partition')
        arms = sim._arm_shape_frames(fk, robots, visual=visual)
        frame_names = set(fk.getFrameNames())
        ee_anchor = {}
        for r in robots:
            semantic = next((name for name in (f'{r}_ur_gripper_center',
                                               f'{r}_ur_vacuum')
                             if name in frame_names), None)
            if semantic is None:
                raise ValueError(f'{r}: no semantic end-effector frame')
            ee_anchor[r] = semantic
        shape_names = {entry[1] for entry in arms}
        shape_names.update(ee_anchor.values())
        shape_frames = [(name, fk.getFrame(name)) for name in sorted(shape_names)]
        def fk_poses(q):
            fk.setJointState(q)
            return {name: np.asarray(frame.getPose(), dtype=float)
                    for name, frame in shape_frames}
        self.fk_poses = fk_poses
        self.fk = fk
        self.C = C
        self.assembly = assembly
        self.pre_insertion = float(cfg.get('pre_insertion') or 0.05)
        self.post_pick = float(cfg.get('post_pick') or 0.0)
        self.states = states
        self.robots = tuple(robots)
        self.gripper_kinds = sim._gripper_kinds_from_cfg(cfg, robots)
        if contact_grasps:
            if set(self.gripper_kinds.values()) != {'two_finger'}:
                raise ValueError('physical-contact baseline currently needs all two-finger hands')
            if not cfg.get('grasp_physics_check'):
                raise ValueError('physical contacts need a saved grasp-physics certificate')
        self.grasp_closures = read_grasp_closures(
            program.plan_path, actions,
            required=bool(sim._saved(cfg, 'grasp_closure_check', legacy=False)))
        self.dt = float(dt)
        self.actuated = bool(actuated)
        self._qprev = None
        self._armed = False
        sim.DT = self.dt
        sim.RECORD_EVERY = max(1, round(0.016 / self.dt))
        sim.ARM_ARM_CONTACTS = True
        q0 = self.full_q(next(seg for seg in program.segments
                              if hasattr(seg, 'waypoints')).waypoints[0])
        if contact_grasps:
            from sim_contact import ContactScene
            scene_type = ContactScene
        else:
            scene_type = sim.SimScene
        self.scene = scene_type(
            assembly, states[0], robots, gripper_kinds=self.gripper_kinds,
            arms=arms, arm_poses=fk_poses(q0), ee_anchor=ee_anchor, gui=gui,
            articulated_C=fk, actuated=actuated,
            table_geometry=sim.planner_table_geometry(C),
            planner_table_top=sim.planner_table_top(C),
            table_mu=cfg.get('table_mu'), animate_grippers=animate_grippers,
            contact_grasps=contact_grasps)
        self.scene.directional_caps = sim._resolve_directional_caps(cfg, None)
        self.scene.mate_caps = None
        self.scene.set_arm_q(q0)
        if self.actuated:
            self.scene._cmd_q = None
        self.record = bool(record)
        if record:
            self.scene.recorder = {
                'qpos': [], 'mpos': [], 'mquat': [], 'forces': [], 'marks': [],
                'controller_mode': [], 'physics_tick': [], 'gripper_events': []}

    def full_q(self, flat):
        flat = np.asarray(flat, dtype=float)
        q = np.asarray(self.fk.getJointState(), dtype=float).copy()
        for robot in self.robots:
            q[list(self.ridx[robot])] = flat[self.flat_slices[robot]]
        return q

    def drive(self, q):
        q = np.asarray(q, dtype=float)
        qd = np.zeros_like(q) if self._qprev is None else (q - self._qprev) / self.dt
        if self.actuated:
            self.scene.command_arm_q(q, qd)
            self.scene.gravity_compensate()
        else:
            self.scene.set_arm_q(q, qd)
        self._qprev = q.copy()

    def step(self, q, report, action):
        self.drive(q)
        self.scene.step(1, report, action)

    def mark(self, label):
        rec = self.scene.recorder
        if rec is not None:
            rec['marks'].append((len(rec['qpos']), label))

    def save(self, path, report, metadata):
        rec = self.scene.recorder
        if rec is None:
            raise ValueError('recording was not enabled')
        sim._save_recording(path, self.scene, rec, report, metadata)
