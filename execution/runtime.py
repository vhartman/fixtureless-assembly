"""Independent, event-driven reference executor for a saved execution program.

The first coordinator is intentionally lockstep. Skill completion depends on
measured state; no precomputed skill duration or offline phase trace is used.
"""
from __future__ import annotations

import math
import os
import numpy as np

import sim
from execution.program import FreeMotion, Skill
from execution.coordinator import LockstepCoordinator
from execution.grasp_preflight import mate_gripper_clearance
from execution.seat_clearance import SeatClearanceFilter
from execution.seat_command import SeatCommandGovernor
from simulation.mujoco_backend import MujocoBackend


class ExecutionFailure(RuntimeError):
    pass




class ReferenceExecutor:
    def __init__(self, program, *, dt=0.002, joint_rate=0.5, joint_accel=2.0, actuated=False,
                 contact_grasps=False, process_wrench=True, record=False,
                 gui=False, visual=True, settle_tolerance=0.01,
                 settle_timeout=1.5, skill_timeout=2.0, mate_grasp_clearance=None):
        if dt <= 0 or joint_rate <= 0:
            raise ValueError('dt and joint_rate must be positive')
        if mate_grasp_clearance is not None and not math.isfinite(mate_grasp_clearance):
            raise ValueError('mate_grasp_clearance must be finite')
        self.program = program
        self.backend = MujocoBackend(program, dt=dt, actuated=actuated,
                                     contact_grasps=contact_grasps,
                                     record=record, gui=gui, visual=visual)
        self.scene = self.backend.scene
        self.report = sim.SimReport()
        self.dt = dt
        self.coordinator = LockstepCoordinator(dt, joint_rate, joint_accel)
        self.settle_tolerance = settle_tolerance
        self.settle_timeout = settle_timeout
        self.skill_timeout = skill_timeout
        self.process_wrench = process_wrench
        self.qref = self.backend.full_q(next(s for s in program.segments
                                             if isinstance(s, FreeMotion)).waypoints[0])
        self.mate_grasp_clearance = (None if mate_grasp_clearance is None
                                     else float(mate_grasp_clearance))
        self.mate_preflight = []
        self.seat_clearance = []
        self.realizations = sim._process_realizations(program.plan)
        self.seat_tolerance = 0.002
        self.servo = (sim._SkillServo(self.scene, dt,
                                    tol=self.seat_tolerance)
                      if actuated else None)
        self.completed_actions = set()
        self.contact_locks = {}
        self._failure = None

    def _apply_contact_locks(self, command):
        """Track a continuing hand relative to its already-mated part, not an old q."""
        if not getattr(self, 'contact_locks', None):
            return command
        command = np.asarray(command, dtype=float).copy()
        for robot, (part, part_T_anchor, q_hold, anchor_at_mate,
                    table_rest, leader, leader_q) in list(self.contact_locks.items()):
            grasp = self.scene.active_grasps.get(robot)
            if grasp is None or grasp[1] != part:
                del self.contact_locks[robot]
                continue
            if leader is not None and not table_rest and leader not in self.scene.active_grasps:
                del self.contact_locks[robot]
                continue
            lead_moving = (leader is not None and not table_rest and
                           np.max(np.abs(self.qref[self.servo.qidx[leader]] - leader_q)) >
                           2.0 * self.settle_tolerance)
            if lead_moving and os.environ.get('SIMV2_TRACE') and not getattr(self, '_leader_trace', False):
                print(f'[simv2/trace] {robot} following {leader}: nominal delta '
                      f'{np.max(np.abs(self.qref[self.servo.qidx[leader]] - leader_q)):.4f}rad', flush=True)
                self._leader_trace = True
            goal = (sim._pose_compose(self.scene.part_pose(part), part_T_anchor)
                    if lead_moving else anchor_at_mate)
            ep = goal[:3] - anchor_at_mate[:3]
            qe = sim._quat_mul(goal[3:], sim._quat_conj(anchor_at_mate[3:]))
            if qe[0] < 0:
                qe = -qe
            nv = float(np.linalg.norm(qe[1:]))
            er = (np.zeros(3) if nv < 1e-12 else
                  qe[1:] * (2.0 * math.atan2(nv, max(float(qe[0]), 0.0)) / nv))
            # Hold the achieved joint posture when the assembly is stationary.
            # If another support moves it, use its measured pose to shift that
            # posture. This fixed reference damps momentum after seating.
            J = self.servo._anchor_jac(robot)
            dq = np.linalg.lstsq(J, np.r_[ep, er], rcond=1e-5)[0]
            idx = self.servo.qidx[robot]
            measured = np.asarray([self.scene.data.qpos[
                self.scene._qadr[self.scene._rai_joints[j]]] for j in idx])
            goal_q = q_hold + dq
            command[idx] = measured + np.clip(goal_q - measured, -0.03, 0.03)
        return command

    def _start_contact_lock(self, event, child, holder, *, path_exit_q=None):
        if not self.scene.contact_grasps or holder is None:
            return
        component = self.scene._active_mate_component(child)
        supports = [c for part in component
                    for c in event.supports_after.get(part, ())]
        table_rest = any(c[0] == 'table' for c in supports)
        leaders = sorted({c[1] for c in supports
                          if c[0] == 'grasp' and c[1] != holder})
        independent = table_rest or bool(leaders)
        if not independent:
            return  # A lone carrier must remain free to transport the assembly.
        anchor = self.scene.grasp_anchor[holder]
        anchor_pose = self.scene.part_pose(anchor)
        idx = self.servo.qidx[holder]
        q_hold = np.asarray([self.scene.data.qpos[
            self.scene._qadr[self.scene._rai_joints[j]]] for j in idx])
        self.contact_locks[holder] = (
            child, sim._pose_compose(sim._pose_invert(self.scene.part_pose(child)),
                                     anchor_pose), q_hold, anchor_pose, table_rest,
            leaders[0] if leaders else None,
            (np.asarray(path_exit_q if path_exit_q is not None else self.qref,
                        dtype=float)[self.servo.qidx[leaders[0]]].copy()
             if leaders else None))
        self.servo.off.pop(holder, None)
        self.servo.capture_qoff.pop(holder, None)


        if os.environ.get('SIMV2_TRACE'):
            print(f'[simv2/trace] contact lock {holder}/{child}: '
                  f'component={sorted(component)}', flush=True)
    def _tick(self, q, label='path'):
        self.qref = np.asarray(q, dtype=float).copy()
        # The saved path is a geometric reference, not an exact physical command.
        # Apply force-responsive reference correction at the plant boundary;
        # the event program and its geometric path remain unchanged.
        if self.servo is not None:
            self.servo.update()
            command = self.servo.adjust(self.qref)
        else:
            command = self.qref
        command = self._apply_contact_locks(command)
        for robot in self.contact_locks:
            idx = self.servo.qidx[robot]
            self.qref[idx] = command[idx]
        self.backend.step(command, self.report, (label,))
        bad = next((r for r in self.report.results if not r.ok), None)
        if bad is not None:
            raise ExecutionFailure(repr(bad))

    def _hold(self, seconds, label='settle'):
        for _ in range(max(0, math.ceil(seconds / self.dt))):
            self._tick(self.qref, label)

    def _settle(self, participants):
        if not self.backend.actuated or not participants:
            return
        dofs = [d for robot in participants for d in self.servo.dofs[robot]]
        for _ in range(math.ceil(self.settle_timeout / self.dt)):
            target = self.servo.adjust(self.qref)
            speed = float(np.max(np.abs(self.scene.data.qvel[dofs]))) if dofs else 0.0
            if (self.scene.tracking_error(target, participants) <= self.settle_tolerance
                    and speed <= 0.02):
                return
            self._tick(self.qref, 'settle')
        target = self.servo.adjust(self.qref)
        speed = float(np.max(np.abs(self.scene.data.qvel[dofs]))) if dofs else 0.0
        raise ExecutionFailure(
            f'{participants}: tracking error {self.scene.tracking_error(target, participants):.4f}rad '
            f'and speed {speed:.4f}rad/s after {self.settle_timeout:g}s')

    def _rejoin_path(self, target, participants):
        """Move from a measured skill endpoint back to the saved path boundary.

        This short joint-space handoff is executed under the same plant contact
        and breakaway monitors as free motion. Larger deviations fail closed.
        """
        target = self._apply_contact_locks(np.asarray(target, dtype=float))
        self.qref = target.copy()
        # A continuing contact hand has a measured part-relative target, not the
        # nominal post-skill joint waypoint from a different physical realization.
        actual = target.copy()
        for j, name in enumerate(self.scene._rai_joints):
            actual[j] = self.scene.data.qpos[self.scene._qadr[name]]
        gap = float(np.max(np.abs(target - actual)))
        if gap <= 0.03:
            return
        if gap > 0.15:
            # Joint distance alone is a poor localness metric near a redundant arm:
            # a sizeable wrist change can leave the grasp tool almost unchanged.
            # Retain a hard joint cap, and additionally require a short tool-space
            # connector for every event participant. The ensuing motion remains
            # physically monitored by the same contact and breakaway checks.
            fk_actual = self.backend.fk_poses(actual)
            fk_target = self.backend.fk_poses(target)
            tool_error = {}
            for robot in participants:
                anchor = self.scene.grasp_anchor[robot]
                here, goal = fk_actual[anchor], fk_target[anchor]
                shift = float(np.linalg.norm(here[:3] - goal[:3]))
                dot = min(1.0, abs(float(np.dot(here[3:], goal[3:]))))
                angle = 2.0 * math.acos(dot)
                tool_error[robot] = (shift, angle)
            if (gap > 0.30 or any(shift > 0.020 or angle > math.radians(5.0)
                                  for shift, angle in tool_error.values())):
                by_robot = {robot: float(np.max(np.abs(
                    target[list(self.backend.ridx[robot])] -
                    actual[list(self.backend.ridx[robot])])))
                    for robot in self.backend.robots}
                raise ExecutionFailure(
                    f'post-skill path mismatch {gap:.4f}rad exceeds local rejoin range; '
                    f'per robot={by_robot}, tool error={tool_error}')
        connector = LockstepCoordinator(self.dt, joint_rate=0.1, joint_accel=0.5)
        for q in connector.samples([actual, target]):
            self._tick(q, 'rejoin')
        self._hold(0.1, 'rejoin')
        self._settle(participants)
    def _drive_motion(self, segment, index):
        points = [self.backend.full_q(q) for q in segment.waypoints]
        approach = {}
        if self.backend.actuated and index + 1 < len(self.program.segments):
            event = self.program.segments[index + 1]
            if (isinstance(event, Skill) and event.frames is not None and
                    event.frames[0] in self.backend.assembly.objects):
                child = event.frames[1]
                holder = next((robot for robot, grasp in self.scene.active_grasps.items()
                               if grasp[1] == child), None)
                if holder is not None:
                    anchor = self.scene.grasp_anchor[holder]
                    approach = {
                        'cartesian_positions': [self.backend.fk_poses(q)[anchor][:3]
                                                for q in points],
                        'cartesian_rate': 0.01,
                    }
        event = (self.program.segments[index + 1]
                 if index + 1 < len(self.program.segments) else None)
        mate = (event.frames if isinstance(event, Skill) and event.frames is not None
                and event.frames[0] in self.backend.assembly.objects else None)
        for q in self.coordinator.samples(points, **approach):
            self._tick(q, 'path')
            if (self.backend.actuated and self.scene.contact_grasps and mate is not None and
                    self.scene.part_pair_contact_force(*mate) >= 0.5 and
                    float(np.max(np.abs(q - points[-1]))) <= 0.05):
                # Intended contact near the endpoint belongs to the skill.
                return points[-1]
        return None

    def _gripper(self, robot, part, close, action_index):
        if self.backend.gripper_kinds[robot] != 'two_finger':
            return
        if close:
            if self.scene.contact_grasps:
                target = 0.045 if self.scene._gripper_model.get(robot) == 'openarm' else 0.042
            else:
                cert = (self.backend.grasp_closures or {}).get(action_index)
                if cert is not None:
                    if cert.get('robot') != robot:
                        raise ExecutionFailure('saved closure belongs to another robot')
                    target = float(cert['aperture'])
                else:
                    target = self.scene.visual_contact_aperture(robot, part, 0.0)
        else:
            target = 0.0
        self.scene.command_gripper_aperture(robot, target, 0.25)
        if self.scene.recorder is not None:
            self.scene.recorder['gripper_events'].append({
                'event': 'close_start' if close else 'open_start',
                'robot': robot, 'part': part,
                'physics_tick': int(self.scene._phys),
                'time_s': float(self.scene._phys * self.dt),
                'frame': len(self.scene.recorder['qpos']), 'animated': True})
        self._hold(0.25 + (0.1 if close and self.scene.contact_grasps else 0.0),
                   'gripper_close' if close else 'gripper_open')

    def _servo_pick(self, event, robot, child):
        if self.servo is None or event.target_relpose is None:
            return
        desired = sim._pose_compose(
            self.scene.part_pose(child),
            sim._pose_invert(np.asarray(event.target_relpose, dtype=float)))
        metrics = self.servo.align_pick(
            self.qref, robot, child, desired,
            self.backend.drive,
            lambda: self.scene.step(1, self.report, ('servo',)),
            self.skill_timeout)
        bad = next((r for r in self.report.results if not r.ok), None)
        if bad is not None:
            raise ExecutionFailure(repr(bad))
        if (metrics['position_m'] > 0.01 or
                metrics['angle_deg'] > 10.0):
            raise ExecutionFailure(
                f'{event.name}: grasp residual {metrics["position_m"]*1000:.1f}mm/'
                f'{metrics["angle_deg"]:.1f}deg')
        if event.action is not None and event.action[0] == 'pick':
            self.servo.inherit_pick_configuration(
                self.qref, robot, position_correction=metrics['position_initial_m'])

    def _servo_seat(self, event, parent, child, *, posture_target=None,
                    budget=None, flow_segment=None):
        if self.backend.assembly.screw_skill(child) is not None:
            raise ExecutionFailure(
                f'{event.name}: screw skill is not implemented in simv2 yet')
        rel = self.backend.assembly.relative_pose(child)
        if rel is None or rel[0] != parent:
            raise ExecutionFailure(f'{event.name}: part does not mate with {parent}')
        cur = sim._pose_compose(sim._pose_invert(self.scene.part_pose(parent)),
                                self.scene.part_pose(child))
        quat = self.backend.assembly.closest_mate_quaternion(child, cur[3:])
        target = np.r_[np.asarray(rel[1], dtype=float), np.asarray(quat, dtype=float)]
        holder = next((robot for robot, grasp in self.scene.active_grasps.items()
                       if grasp[1] == child), None)
        if holder is None:
            raise ExecutionFailure(f'{event.name}: inserted part has no active gripper')
        child_goal = sim._pose_compose(self.scene.part_pose(parent), target)
        try:
            check = mate_gripper_clearance(
                self.scene, holder, parent, child, child_goal,
                max_distance=max(0.1, (self.mate_grasp_clearance or 0.0) + 0.01))
        except ValueError as exc:
            raise ExecutionFailure(f'{event.name}: mate preflight unavailable: {exc}') from exc
        check = {'event': event.name, 'robot': holder, 'parent': parent,
                 'child': child, **check}
        self.mate_preflight.append(check)
        print(f'[simv2/preflight] {event.name}: achieved-grasp tool/receiver gap '
              f'{check["clearance_m"]*1000:.2f}mm {check["pair"]}; '
              f'moving-pad gap {check["pad_clearance_m"]*1000:.2f}mm', flush=True)
        if (self.mate_grasp_clearance is not None and
                check['clearance_m'] < self.mate_grasp_clearance - 1e-4):
            raise ExecutionFailure(
                f'{event.name}: achieved-grasp tool/receiver clearance below required reserve '
                f'({check["clearance_m"]*1000:.2f}mm < '
                f'{self.mate_grasp_clearance*1000:.2f}mm; {check["pair"]})')
        if self.servo is not None:
            clearance_filter = SeatClearanceFilter(
                self.scene, self.servo, holder, parent)
            governor = SeatCommandGovernor(self.dt)
            # The skill may advance, retreat once, and re-advance across the
            # authored funnel. A fixed two-second timeout is shorter than that
            # motion at the controller's guarded contact speed.
            rate = 0.01 if self.scene.contact_grasps else sim.INSERT_CONTACT_SPEED
            seat_budget = max(self.skill_timeout, 12.0, 2.0 * self.backend.pre_insertion / rate + 1.0)
            flow = (tuple(self.coordinator.samples(
                [self.backend.full_q(q) for q in flow_segment.waypoints]))
                if flow_segment is not None else ())
            other = [j for j, name in enumerate(self.program.joints)
                     if not any(name.startswith(robot + '_') for robot in event.robots)]
            moving_other = bool(flow and other and
                                np.max(np.abs(flow[-1][other] - flow[0][other])) > 1e-3)
            flow_i = 0
            def skill_drive(q):
                mixed = np.asarray(q, dtype=float).copy()
                if flow and other:
                    mixed[other] = flow[min(flow_i, len(flow) - 1)][other]
                self.backend.drive(mixed)
            def skill_tick():
                nonlocal flow_i
                self.scene.step(1, self.report, ('servo',))
                flow_i += 1
            self.servo.seat(self.qref, parent, child, event.name,
                            skill_drive, skill_tick,
                            max(seat_budget, 0.0 if budget is None else budget,
                                (len(flow) - 1) * self.dt + 1.0), target,
                            self.backend.assembly.mate_axis(child),
                            sim._mate_intrinsic_axis(self.backend.assembly, child),
                            min_steps=len(flow) - 1 if moving_other else 0,
                            posture_target=posture_target,
                            progress_backoff=False,
                            clearance_filter=clearance_filter,
                            command_governor=governor,
                            angular_rate_limit=0.15,
                            tool_pose_target=True,
                            contact_tolerance=self.seat_tolerance)
            self.seat_clearance.append({
                'event': event.name,
                'outcome': self.servo.seat_outcome.get(event.name),
                'minimum_m': (clearance_filter.minimum if np.isfinite(
                    clearance_filter.minimum) else None),
                'constrained_ticks': governor.clearance_ticks,
                'command_infeasible_ticks': governor.infeasible_ticks})
            outcome = self.servo.seat_outcome.get(event.name, {})
            if outcome.get('status') not in ('converged', 'seated_contact'):
                raise ExecutionFailure(
                    f'{event.name}: mating controller did not converge: {outcome}')
            # A measured skill may finish away from its entry witness. Hold
            # that achieved posture across the support switch; otherwise the
            # next dwell commands the pre-insertion pose back through the mate.
            self.qref = np.asarray([
                self.scene.data.qpos[self.scene._qadr[name]]
                for name in self.scene._rai_joints], dtype=float)
        bad = next((r for r in self.report.results if not r.ok), None)
        if bad is not None:
            raise ExecutionFailure(repr(bad))
        metrics = sim._mate_pose_error(
            self.scene.part_pose(parent), self.scene.part_pose(child), target,
            self.backend.assembly.mate_axis(child),
            sim._mate_intrinsic_axis(self.backend.assembly, child))
        if metrics['position'] > self.seat_tolerance or metrics['angle'] > sim.SEAT_ANG_TOL:
            raise ExecutionFailure(
                f'{event.name}: relative seat error {metrics["position"]*1000:.1f}mm/'
                f'{np.rad2deg(metrics["angle"]):.1f}deg; '
                f'servo={self.servo.seat_outcome.get(event.name) if self.servo else None}')

    def _skill(self, event, *, path_exit_q=None, flow_segment=None):
        if (event.frames is None and self.servo is not None and
                '_post_pick_' in event.name and event.action is not None and
                event.action[0] == 'pick'):
            self.servo.finish_pick_retreat(event.action[1], q_ref=self.qref)
            self._settle(event.robots)
            return
        self._settle(event.robots)
        if event.frames is None:
            return
        if event.action_index is None:
            raise ExecutionFailure(f'{event.name}: attachment has no task-plan action')
        aid = event.action_index
        if os.environ.get('SIMV2_TRACE'):
            target = sim._state_part_poses(self.backend.states[aid + 1],
                                           self.backend.assembly, lift=sim.CARRY_LIFT)
            locations = {part: (self.scene.part_pose(part)[:3].round(4).tolist(),
                                target[part][:3].round(4).tolist())
                         for part in self.backend.assembly.objects}
            print(f'[simv2/trace] before {event.name}: holders={self.scene.active_grasps} '
                  f'actual/target={locations}', flush=True)
        if aid in self.completed_actions:
            raise ExecutionFailure(f'action {aid} has multiple attachment events')
        scene = self.scene
        parent, child = event.frames
        before_grasps = sim._symbolic_grasps(self.backend.states[aid])
        after_grasps = sim._symbolic_grasps(self.backend.states[aid + 1])
        added_grasps = after_grasps - before_grasps
        removed_grasps = before_grasps - after_grasps
        scene.expect_state = self.backend.states[aid + 1]
        incoming = parent.split('_', 1)[0] if parent not in self.backend.assembly.objects and parent != 'table' else None
        old_holders = {r for r, g in scene.active_grasps.items() if g[1] == child}
        needs_new_grasp = incoming is not None and any(r == incoming for r, _ in added_grasps)
        if needs_new_grasp:
            self._servo_pick(event, incoming, child)
            self._gripper(incoming, child, True, aid)
        if parent in self.backend.assembly.objects:
            self._servo_seat(event, parent, child,
                             posture_target=self.qref if path_exit_q is None else path_exit_q,
                             budget=max(self.skill_timeout, 8.0) if path_exit_q is not None
                             else None, flow_segment=flow_segment)
        if incoming is not None and not needs_new_grasp:
            # MP may re-root the kinematic tree onto a *continuing* holder.
            # That is not a new grasp: preserve its existing weld/contact and
            # release only the hands absent from the saved successor state.
            for robot, _part in sorted(removed_grasps):
                scene.release(robot)
            ok = True
        elif event.action and event.action[0] == 'handover' and scene.contact_grasps:
            ok = scene.contact_handover(parent, child, old_holders,
                                        grasp_relpose=event.target_relpose)
        else:
            ok = scene.attach(parent, child, grasp_relpose=event.target_relpose,
                              preserve_existing_grasps=(event.action is not None and
                                                        event.action[0] == 'add_grasp'))
        if not ok:
            raise ExecutionFailure(f'{event.name}: attachment/contact acquisition failed')
        # The exported frame is a kinematic re-root, not a complete list of
        # physical hand releases. Reconcile the actual holders with the saved
        # successor's support change before any post-event physics step.
        for robot, _part in sorted(removed_grasps):
            scene.release(robot)
        # An ideal mate can replace the inserter's weld inside the low-level
        # plant, while the symbolic successor still says that hand supports the
        # newly assembled component. Restore exactly those persisted supports;
        # never invent a grasp from a task name or re-root frame alone.
        for robot, part in sorted(after_grasps):
            if robot not in scene.active_grasps:
                grasp_part, gid = sim._held_grasp_id(scene.expect_state, part,
                                                     robot=robot)
                if not scene.grasp(robot, part, gid=gid, grasp_part=grasp_part):
                    raise ExecutionFailure(
                        f'{event.name}: continuing {robot}/{part} support was lost')
        if parent in self.backend.assembly.objects and scene.contact_grasps:
            holder = next((robot for robot, grasp in scene.active_grasps.items()
                           if grasp[1] == child), None)
            self._start_contact_lock(event, child, holder,
                                     path_exit_q=path_exit_q)
        if (self.servo is not None and not scene.contact_grasps and
                event.action is not None and event.action[0] == 'pick' and
                any(grasp[1] == child for grasp in scene.active_grasps.values())):
            robot = event.action[1]
            retreat = self.backend.post_pick
            budget = max(self.settle_timeout,
                         retreat / max(0.8 * self.servo.vmax, 1e-9) + 0.25)
            self.servo.pick_lift(
                self.qref, robot, child, event.name, self.backend.drive,
                lambda: scene.step(1, self.report, ('servo',)), budget,
                retreat_clearance=retreat)
            bad = next((r for r in self.report.results if not r.ok), None)
            if bad is not None:
                raise ExecutionFailure(repr(bad))
        if self.process_wrench:
            scene.set_process_wrenches([])
        self.backend.mark(str(event.action))
        for robot, part in sorted(sim._symbolic_grasp_opens(self.backend.states, aid)):
            self._gripper(robot, part, False, aid)
        self._hold(0.05, 'skill')
        if (self.backend.actuated and not
                (self.servo is not None and event.action is not None and
                 event.action[0] == 'pick' and event.action[1] in self.servo.capture_hold)):
            if path_exit_q is not None:
                self.qref = np.asarray(path_exit_q, dtype=float).copy()
            self._rejoin_path(self.qref, event.robots)
            handoff_error = scene.tracking_error(self.qref, event.robots)
            if os.environ.get('SIMV2_TRACE'):
                print(f'[simv2/trace] {event.name} path-entry error '
                      f'{handoff_error:.4f}rad', flush=True)
            if handoff_error > 0.03:
                raise ExecutionFailure(
                    f'{event.name}: {handoff_error:.4f}rad from saved path entry; '
                    'no certified local connector exists')
        result = sim._judge_resting(scene, self.backend.assembly, event.action,
                                    self.backend.states[aid + 1])
        self.report.results.append(result)
        self.completed_actions.add(aid)
        if not result.ok:
            raise ExecutionFailure(repr(result))
        print(f'[simv2] action {aid + 1}/{len(self.program.plan)} {event.action}: ok', flush=True)

    def _set_next_process_load(self, index):
        if not self.process_wrench:
            return
        next_event = next((s for s in self.program.segments[index + 1:]
                           if isinstance(s, Skill)), None)
        if (next_event is not None and next_event.frames is not None and
                next_event.action and next_event.action[0] == 'assemble' and
                next_event.frames[0] in self.backend.assembly.objects):
            parent, child = next_event.frames
            self.scene.set_process_wrenches([
                (child, parent, self.realizations.get(next_event.action_index, {}))])
        else:
            self.scene.set_process_wrenches([])

    def run(self):
        try:
            i = 0
            while i < len(self.program.segments):
                segment = self.program.segments[i]
                if isinstance(segment, FreeMotion):
                    previous = self.program.segments[i - 1] if i else None
                    following = (self.program.segments[i + 1]
                                 if i + 1 < len(self.program.segments) else None)
                    if (self.servo is not None and isinstance(previous, Skill) and
                            isinstance(following, Skill) and
                            '_pre_assemble_' in previous.name and
                            following.action is not None and
                            following.action[0] == 'assemble' and
                            previous.action_index == following.action_index):
                        # The exported final funnel is a skill domain, not free
                        # joint motion. Other arms still follow their saved path.
                        self._set_next_process_load(i)
                        self._skill(following,
                                    path_exit_q=self.backend.full_q(segment.waypoints[-1]),
                                    flow_segment=segment)
                        i += 2
                        continue
                    self._set_next_process_load(i)
                    path_exit_q = self._drive_motion(segment, i)
                    if path_exit_q is not None:
                        self._skill(self.program.segments[i + 1],
                                    path_exit_q=path_exit_q)
                        i += 2
                        continue
                else:
                    self._skill(segment)
                i += 1
            self.scene.set_process_wrenches([])
            self._hold(0.3, 'settle')
            missing = set(range(len(self.program.plan))) - self.completed_actions
            if missing:
                raise ExecutionFailure(f'plan actions with no executed skill: {sorted(missing)}')
        except ExecutionFailure as exc:
            self._failure = str(exc)
            if not any(not result.ok for result in self.report.results):
                self.report.results.append(sim.ActionResult(
                    ('execution',), False, 'execution_contract', str(exc)))
            print(f'[simv2] FAILED: {exc}', flush=True)
        self.report.final_pose_err = {}
        expected = sim._state_part_poses(self.backend.states[-1], self.backend.assembly,
                                         lift=sim.CARRY_LIFT)
        for part in self.backend.assembly.objects:
            exp, got = np.asarray(expected[part]), self.scene.part_pose(part)
            self.report.final_pose_err[part] = (
                float(np.linalg.norm(got[:3] - exp[:3])),
                sim._part_orientation_error(self.backend.assembly, part, got[3:], exp[3:]))
        self.report.grasp_wrench_metrics = self.scene.grasp_wrench_metrics()
        self.report.table_wrench_metrics = self.scene.table_wrench_metrics()
        return self.report
