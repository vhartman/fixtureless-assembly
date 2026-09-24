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
