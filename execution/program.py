"""Compile a saved task plan and motion path into explicit execution events.

This module deliberately has no RAI, MuJoCo, IK, or planner dependency. The saved
post-fusion motion export is the authority for task identities and attachments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from assembly_planner.plan_io import read_motion_data, read_plan


@dataclass(frozen=True)
class FreeMotion:
    mode: tuple[int, ...]
    waypoints: tuple[tuple[float, ...], ...]
    start_index: int
    end_index: int
    support_state: int
    supports: dict[str, list[list[Any]]]


@dataclass(frozen=True)
class Skill:
    task_id: int
    name: str
    kind: str
    action_index: int | None
    action: tuple[Any, ...] | None
    robots: tuple[str, ...]
    frames: tuple[str, str] | None
    target_relpose: tuple[float, ...] | None
    state_before: int
    state_after: int
    supports_before: dict[str, list[list[Any]]]
    supports_after: dict[str, list[list[Any]]]
    path_index: int


@dataclass(frozen=True)
class ExecutionProgram:
    version: int
    plan_path: str
    motion_path: str
    plan: tuple[tuple[Any, ...], ...]
    robots: tuple[str, ...]
    joints: tuple[str, ...]
    segments: tuple[FreeMotion | Skill, ...]
    task_signature: str
    motion_data_sha256: str

    def to_dict(self):
        out = asdict(self)
        out['segments'] = [dict(type=('motion' if isinstance(s, FreeMotion) else 'skill'),
                                **asdict(s)) for s in self.segments]
        return out


def _task_signature(tasks):
    digest = hashlib.sha256()
    for task in tasks:
        fields = [
            str(task.get('name', '')),
            str(task.get('type', '')),
            ','.join(str(r) for r in (task.get('robots') or [])),
            ','.join(str(f) for f in (task.get('frames') or [])),
            json.dumps(task.get('skill_data'), sort_keys=True, separators=(',', ':')),
        ]
        digest.update(('|'.join(fields) + '\n').encode())
    return digest.hexdigest()[:16]


def _mode(raw: dict, robots: tuple[str, ...], n_tasks: int, n_q: int, index: int):
    q, mode = raw.get('q'), raw.get('mode')
    if not isinstance(q, list) or len(q) != n_q:
        raise ValueError(f'motion state {index}: expected {n_q} joint values')
    if not isinstance(mode, list) or len(mode) != len(robots):
        raise ValueError(f'motion state {index}: expected {len(robots)} mode ids')
    if any(not isinstance(x, int) or not 0 <= x < n_tasks for x in mode):
        raise ValueError(f'motion state {index}: invalid task id')
    return tuple(float(x) for x in q), tuple(mode)


def compile_program(plan_path: str | Path, motion_path: str | Path) -> ExecutionProgram:
    plan_path, motion_path = Path(plan_path), Path(motion_path)
    cfg, actions = read_plan(plan_path)
    supported = {'pick', 'assemble', 'release', 'place', 'add_grasp', 'handover'}
    unknown = sorted({a[0] if a else "<empty>" for a in actions if not a or a[0] not in supported})
    if unknown:
        raise ValueError(f'unsupported execution action(s): {unknown}')
    motion_data = read_motion_data(plan_path, actions,
                                   grasp_db_hash=cfg.get('grasp_sampler_db_hash'))
    if motion_data is None:
        raise ValueError('task plan lacks persisted motion states; refusing successor replay')
    with motion_path.open() as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict) or not isinstance(raw.get('mp_export'), dict):
        raise ValueError('motion path lacks the final post-fusion MP export')
    if raw.get('optimized') is not True:
        raise ValueError('motion path is not marked optimized')
    export = raw['mp_export']
    if export.get('plan') != [list(a) for a in actions]:
        raise ValueError('motion export belongs to a different task plan')
    signature = raw.get('task_signature')
    if not signature or signature != export.get('task_signature'):
        raise ValueError('motion-path and export task signatures differ')
    tasks = export.get('tasks')
    step_names = export.get('step_names')
    joints = export.get('joints')
    if not isinstance(tasks, list) or not tasks or not isinstance(joints, list):
        raise ValueError('motion export has no task records or joint order')
    if not isinstance(step_names, list) or len(step_names) != len(actions):
        raise ValueError('motion export action-to-task map has the wrong length')
    names = [task.get('name') for task in tasks]
    if any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError('motion export has missing or duplicate task names')
    if export.get('sequence_names') != names:
        raise ValueError('motion export sequence manifest differs from task records')
    if _task_signature(tasks) != signature:
        raise ValueError('saved task records do not match their signature')
    action_of = {}
    for i, group in enumerate(step_names):
        if not isinstance(group, list) or not group:
            raise ValueError(f'action {i} has no exported task sequence')
        for name in group:
            if name not in names or name in action_of:
                raise ValueError(f'unknown or multiply owned exported task {name!r}')
            action_of[name] = i
    robots = tuple(f'a{i + 1}' for i in range(int(cfg['num_robots'])))
    paths = raw.get('states')
    if not isinstance(paths, list) or len(paths) < 2:
        raise ValueError('motion path must have at least two states')
    states = [_mode(st, robots, len(tasks), len(joints), i)
              for i, st in enumerate(paths)]
    if any(not any(j.startswith(r + '_') for r in robots) for j in joints):
        raise ValueError('motion export includes joints outside the robot roster')
    segments: list[FreeMotion | Skill] = []
    fired: set[int] = set()
    applied = 0
    saved_states = motion_data['states']
    start = 0
    for end in range(1, len(states)):
        old, new = states[end - 1][1], states[end][1]
        if any(a > b for a, b in zip(old, new)):
            raise ValueError(f'motion state {end}: task mode moves backwards')
        if old == new:
            continue
        points = tuple(states[k][0] for k in range(start, end))
        # The first state of the next mode is the previous mode's endpoint.
        points += (states[end][0],)
        segments.append(FreeMotion(old, points, start, end, applied,
                                   saved_states[applied]['supports']))
        crossed = sorted({tid for a, b in zip(old, new) for tid in range(a, b)})
        for tid in crossed:
            if tid in fired:
                continue
            task = tasks[tid]
            participants = tuple(task.get('robots') or ())
            if not participants or any(r not in robots for r in participants):
                raise ValueError(f'task {tid}: invalid participant roster')
            # Nonparticipants may skip past another robot's task id. Fire only
            # after *its own* participants have all left the task.
            if any(new[robots.index(r)] <= tid for r in participants):
                continue
            if any(old[robots.index(r)] != tid for r in participants):
                raise ValueError(f'task {tid}: participant was not in its task mode')
            changed = [k for r in participants for k, name in enumerate(joints)
                       if name.startswith(r + '_') and
                       abs(states[end - 1][0][k] - states[end][0][k]) > 1e-6]
            if changed:
                raise ValueError(f'task {tid}: participant moves across its event boundary')
            frames = task.get('frames')
            if frames is not None and (not isinstance(frames, list) or len(frames) != 2):
                raise ValueError(f'task {tid}: malformed attachment frames')
            aid = action_of.get(task['name'])
            before = applied
            if frames is not None:
                if aid != applied:
                    raise ValueError(f'task {tid}: attachment is out of action order')
                applied += 1
            segments.append(Skill(tid, task['name'], str(task.get('type') or 'terminal'),
                                  aid, tuple(actions[aid]) if aid is not None else None,
                                  participants, tuple(frames) if frames else None,
                                  tuple(task['canonical_relpose']) if task.get('canonical_relpose')
                                  else None,
                                  before, applied,
                                  saved_states[before]['supports'],
                                  saved_states[applied]['supports'], end))
            fired.add(tid)
        start = end
    segments.append(FreeMotion(states[start][1],
                               tuple(s[0] for s in states[start:]),
                               start, len(states) - 1, applied,
                               saved_states[applied]['supports']))
    required = {i for i, task in enumerate(tasks) if task.get('frames') is not None}
    if not required <= fired:
        raise ValueError(f'motion path omits attachment tasks {sorted(required - fired)}')
    physical_order = [s.action_index for s in segments
                      if isinstance(s, Skill) and s.frames is not None]
    if physical_order != list(range(len(actions))):
        raise ValueError('executed attachment events do not match task-plan action order')
    return ExecutionProgram(2, str(plan_path), str(motion_path),
                            tuple(tuple(a) for a in actions), robots, tuple(joints),
                            tuple(segments), signature, motion_data['sha256'])


def save_program(path: str | Path, program: ExecutionProgram):
    """Persist an execution program with a checksum over its complete payload."""
    payload = program.to_dict()
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    payload['sha256'] = hashlib.sha256(raw).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


def load_program(path: str | Path) -> ExecutionProgram:
    """Load an already compiled program; no motion-planner or IK call is made."""
    with Path(path).open() as stream:
        payload = json.load(stream)
    expected = payload.pop('sha256', None)
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    if expected != hashlib.sha256(raw).hexdigest():
        raise ValueError('execution-program checksum mismatch')
    if payload.get('version') != 2:
        raise ValueError('unsupported execution-program version')
    segments = []
    for item in payload['segments']:
        item = dict(item)
        kind = item.pop('type')
        if kind == 'motion':
            segments.append(FreeMotion(tuple(item['mode']),
                                       tuple(tuple(q) for q in item['waypoints']),
                                       int(item['start_index']), int(item['end_index']),
                                       int(item['support_state']), item['supports']))
        elif kind == 'skill':
            segments.append(Skill(
                int(item['task_id']), str(item['name']), str(item['kind']),
                item['action_index'],
                None if item['action'] is None else tuple(item['action']),
                tuple(item['robots']),
                None if item['frames'] is None else tuple(item['frames']),
                None if item['target_relpose'] is None else tuple(item['target_relpose']),
                int(item['state_before']), int(item['state_after']),
                item['supports_before'], item['supports_after'],
                int(item['path_index'])))
        else:
            raise ValueError(f'unknown execution-program segment type {kind!r}')
    return ExecutionProgram(
        2, str(payload['plan_path']), str(payload['motion_path']),
        tuple(tuple(a) for a in payload['plan']), tuple(payload['robots']),
        tuple(payload['joints']), tuple(segments),
        str(payload['task_signature']), str(payload['motion_data_sha256']))
