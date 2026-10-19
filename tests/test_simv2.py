"""Artifact and event-contract tests for the new independent executor."""
import json

import pytest

from execution import program as ep


def _artifact(tmp_path, *, tasks=None, states=None, plan=None, steps=None):
    plan = plan or [['pick', 'a1', 'obj_1', 0]]
    tasks = tasks or [
        {'name': 'pick', 'type': 'pick', 'robots': ['a1'],
         'frames': ['a1_ur_gripper_center', 'obj_1'],
         'canonical_relpose': [0, 0, 0, 1, 0, 0, 0]},
        {'name': 'terminal', 'type': None, 'robots': ['a1', 'a2'],
         'frames': None}]
    states = states or [
        {'q': [0.0, 0.0], 'mode': [0, 1]},
        {'q': [0.0, 0.0], 'mode': [1, 1]}]
    steps = steps or [['pick']]
    signature = ep._task_signature(tasks)
    mp = {'task_signature': signature, 'optimized': True, 'states': states,
          'mp_export': {'plan': plan, 'task_signature': signature,
                        'joints': ['a1_joint1', 'a2_joint1'],
                        'tasks': tasks, 'step_names': steps,
                        'sequence_names': [t['name'] for t in tasks]}}
    path = tmp_path / 'mp.json'
    path.write_text(json.dumps(mp))
    return path


def _patch_plan(monkeypatch):
    monkeypatch.setattr(ep, 'read_plan', lambda _p: (
        {'num_robots': 2}, [('pick', 'a1', 'obj_1', 0)]))
    monkeypatch.setattr(ep, 'read_motion_data', lambda *_a, **_kw: {'sha256': 'md', 'states': [
        {'supports': {'obj_1': [['table']]}},
        {'supports': {'obj_1': [['grasp', 'a1', 'obj_1', 0]]}}]})


def test_compiler_emits_explicit_motion_and_pick(tmp_path, monkeypatch):
    _patch_plan(monkeypatch)
    result = ep.compile_program('plan.json', _artifact(tmp_path))
    assert [type(s).__name__ for s in result.segments] == [
        'FreeMotion', 'Skill', 'FreeMotion']
    event = result.segments[1]
    assert event.frames == ('a1_ur_gripper_center', 'obj_1')
    assert event.action_index == 0
    assert event.state_after == 1
    assert result.to_dict()['segments'][1]['type'] == 'skill'


def test_compiler_rejects_different_plan_and_missing_event(tmp_path, monkeypatch):
    _patch_plan(monkeypatch)
    path = _artifact(tmp_path)
    data = json.loads(path.read_text())
    data['mp_export']['plan'] = [['release', 'a1', 'obj_1']]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='different task plan'):
        ep.compile_program('plan.json', path)
    path = _artifact(tmp_path, states=[
        {'q': [0.0, 0.0], 'mode': [0, 1]},
        {'q': [0.1, 0.0], 'mode': [0, 1]}])
    with pytest.raises(ValueError, match='omits attachment tasks'):
        ep.compile_program('plan.json', path)


def test_compiler_fires_only_participants_on_skipped_task_ids(tmp_path, monkeypatch):
    _patch_plan(monkeypatch)
    path = _artifact(tmp_path, states=[
        {'q': [0.0, 0.0], 'mode': [0, 0]},
        {'q': [0.0, 0.0], 'mode': [0, 1]},
        {'q': [0.0, 0.0], 'mode': [1, 1]}])
    result = ep.compile_program('plan.json', path)
    events = [s for s in result.segments if isinstance(s, ep.Skill)]
    assert len(events) == 1 and events[0].path_index == 2


def test_compiler_rejects_participant_moving_across_event(tmp_path, monkeypatch):
    _patch_plan(monkeypatch)
    path = _artifact(tmp_path, states=[
        {'q': [0.0, 0.0], 'mode': [0, 1]},
        {'q': [0.1, 0.0], 'mode': [1, 1]}])
    with pytest.raises(ValueError, match='participant moves'):
        ep.compile_program('plan.json', path)


def test_reference_coordinator_keeps_path_and_rate():
    import numpy as np
    from execution.coordinator import LockstepCoordinator
    coordinator = LockstepCoordinator(0.1, 0.5)
    samples = list(coordinator.samples([[0.0, 0.0], [0.2, -0.1], [0.2, 0.1]]))
    assert np.allclose(samples[-1], [0.2, 0.1])
    assert max(float(np.max(np.abs(b - a))) for a, b in zip(
        [np.array([0.0, 0.0])] + samples[:-1], samples)) <= 0.05 + 1e-12
    assert all(abs(q[1] + 0.5*q[0]) < 1e-12 for q in samples[:4])


def test_reference_coordinator_does_not_stop_at_internal_waypoint():
    import numpy as np
    from execution.coordinator import LockstepCoordinator

    dt = 0.01
    samples = np.asarray(list(LockstepCoordinator(dt, 0.5, 2.0).samples(
        [[0.0, 0.0], [0.2, -0.1], [0.4, -0.2]])))
    crossing = int(np.argmin(np.abs(samples[:, 0] - 0.2)))
    speed = np.diff(samples[:, 0]) / dt
    assert min(speed[crossing - 2:crossing + 2]) > 0.4
    assert np.max(np.abs(np.diff(samples, axis=0))) <= 0.5 * dt + 1e-12
    assert np.allclose(samples[-1], [0.4, -0.2])


def test_post_skill_rejoin_uses_tool_space_bound():
    import numpy as np
    from types import SimpleNamespace
    from execution.runtime import ExecutionFailure, ReferenceExecutor

    executor = ReferenceExecutor.__new__(ReferenceExecutor)
    executor.dt = 0.01
    executor.scene = SimpleNamespace(
        _rai_joints=('joint',), _qadr={'joint': 0},
        data=SimpleNamespace(qpos=np.array([0.2])),
        grasp_anchor={'a1': 'tool'})
    executor.backend = SimpleNamespace(
        robots=('a1',), ridx={'a1': (0,)},
        fk_poses=lambda q: {'tool': np.array([0.05 * q[0], 0, 0, 1, 0, 0, 0])})
    samples = []
    executor._tick = lambda q, label: samples.append((np.asarray(q), label))
    executor._hold = lambda *_args: None
    executor._settle = lambda *_args: None

    executor._rejoin_path(np.array([0.0]), ('a1',))
    assert samples and np.allclose(samples[-1][0], [0.0])
    assert all(label == 'rejoin' for _, label in samples)

    executor.backend.fk_poses = lambda q: {
        'tool': np.array([0.2 * q[0], 0, 0, 1, 0, 0, 0])}
    with pytest.raises(ExecutionFailure, match='tool error'):
        executor._rejoin_path(np.array([0.0]), ('a1',))


def test_program_artifact_roundtrip_and_checksum(tmp_path, monkeypatch):
    _patch_plan(monkeypatch)
    program = ep.compile_program('plan.json', _artifact(tmp_path))
    path = tmp_path / 'program.json'
    ep.save_program(path, program)
    assert ep.load_program(path) == program
    raw = json.loads(path.read_text())
    raw['segments'][1]['name'] = 'tampered'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='checksum mismatch'):
        ep.load_program(path)

def test_contact_lock_holds_stationary_support_and_follows_commanded_leader():
    from types import SimpleNamespace
    import numpy as np
    from execution.runtime import ReferenceExecutor

    executor = ReferenceExecutor.__new__(ReferenceExecutor)
    ident = np.array([0., 0., 0., 1., 0., 0., 0.])
    executor.settle_tolerance = 0.01
    executor.scene = SimpleNamespace(
        active_grasps={'a1': (-1, 'base'), 'a2': (-1, 'child')},
        grasp_anchor={'a2': 'tool'},
        part_pose=lambda part: (np.array([0.1, 0., 0., 1., 0., 0., 0.])
                                if part == 'child' else ident),
        _rai_joints=('a2_joint',), _qadr={'a2_joint': 0},
        data=SimpleNamespace(qpos=np.array([0.4])))
    executor.servo = SimpleNamespace(
        qidx={'a2': [0], 'a1': [1]},
        _anchor_jac=lambda _robot: np.array([[1.], [0.], [0.], [0.], [0.], [0.]]))
    executor.contact_locks = {
        'a2': ('child', ident, np.array([0.4]), ident, False, 'a1', np.array([0.]))}
    executor.qref = np.array([0.4, 0.])
    assert executor._apply_contact_locks(np.array([0.7, 0.]))[0] == pytest.approx(0.4)
    executor.qref[1] = 0.2
    assert executor._apply_contact_locks(np.array([0.7, 0.]))[0] > 0.4
    executor.contact_locks['a2'] = (
        'child', ident, np.array([0.4]), ident, True, 'a1', np.array([0.]))
    assert executor._apply_contact_locks(np.array([0.7, 0.]))[0] == pytest.approx(0.4)
    del executor.scene.active_grasps['a2']
    assert executor._apply_contact_locks(np.array([0.7, 0.]))[0] == pytest.approx(0.7)
