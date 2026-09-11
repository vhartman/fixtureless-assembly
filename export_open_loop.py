#!/usr/bin/env python3
"""Export a hardware-oriented open-loop joint trajectory from a sim recording.

The input is the replay NPZ written by ``sim.py run --record``.  The output is another,
smaller NPZ containing only robot joint samples and execution annotations; it deliberately
does not contain MuJoCo bodies, object poses, or contact forces.

Examples:

    python3 export_open_loop.py replay.npz open-loop-plan.npz
    python3 export_open_loop.py replay.npz open-loop-plan.json

The exported ``q`` is the joint motion that actually occurred in simulation, not the nominal
motion-planner command.  ``qd`` and ``qdd`` are deterministic finite differences at the saved
recording rate.  Every sample also carries ``attached_objects`` (one object name or ``None`` per
robot), reconstructed from the recorded close/open transitions.  Samples annotated with
``controller_active`` depended on an online execution
hook in simulation and therefore require an equivalent hardware controller rather than blind
position playback.
"""

import argparse
import ast
import json
from pathlib import Path

import mujoco
import numpy as np


SCHEMA = "assembly-open-loop-v2"
JSON_SCHEMA = "assembly-open-loop-json-v3"
_OPEN_LOOP_MODE = "trajectory"


def _json_scalar(z, name, default):
  if name not in z.files:
    return default
  return json.loads(str(z[name]))


def _robot_joints(model, robots):
  """Return model-ordered scalar arm joints grouped in the recording's robot order."""
  rows = []
  slices = {}
  for robot in robots:
    start = len(rows)
    prefix = robot + "_"
    for jid in range(int(model.njnt)):
      name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
      if not name or not name.startswith(prefix) or name.startswith("sim_grip__"):
        continue
      kind = int(model.jnt_type[jid])
      if kind not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                      int(mujoco.mjtJoint.mjJNT_SLIDE)):
        continue
      rows.append((robot, name, int(model.jnt_qposadr[jid]),
                   kind == int(mujoco.mjtJoint.mjJNT_HINGE)))
    slices[robot] = [start, len(rows)]
  if not rows:
    raise ValueError("recording XML contains no named scalar robot joints")
  empty = [r for r, (a, b) in slices.items() if a == b]
  if empty:
    raise ValueError(f"recording XML contains no arm joints for robot(s) {empty}")
  return rows, slices


def _differentiate(q, t, hinge):
  """Finite-difference a sampled configuration without inventing a smoothing policy."""
  if len(q) < 2:
    return np.zeros_like(q), np.zeros_like(q)
  qc = np.asarray(q, dtype=float).copy()
  if np.any(hinge):
    qc[:, hinge] = np.unwrap(qc[:, hinge], axis=0)
  edge = 2 if len(q) >= 3 else 1
  qd = np.gradient(qc, t, axis=0, edge_order=edge)
  qdd = np.gradient(qd, t, axis=0, edge_order=edge)
  return qd, qdd


def _event_time(event, *, physics_dt, time_origin, source_t):
  """Return one event in the recording-relative time coordinate."""
  if "physics_tick" in event:
    value = (int(event["physics_tick"]) - time_origin) * physics_dt
  elif event.get("inferred") and "frame" in event:
    frame = min(max(int(event["frame"]), 0), len(source_t) - 1)
    value = float(source_t[frame])
  elif "time_s" in event:
    # New recordings store absolute simulation time. Recordings normally start on the first
    # physics tick, so remove that origin just as for physics_tick.
    value = float(event["time_s"]) - time_origin * physics_dt
  else:
    frame = min(max(int(event.get("frame", 0)), 0), len(source_t) - 1)
    value = float(source_t[frame])
  return min(max(float(value), 0.0), float(source_t[-1]))


def _attachment_trace(t, events, robots, *, physics_dt, time_origin):
  """Return the object held by each robot at every source-recording sample.

  Events are applied in their recorded order.  This matters for a handover: the receiver's
  close is emitted before the giver's open at the same physics tick.  Empty strings are used in
  the numeric NPZ channel because fixed-width arrays cannot represent JSON ``null``; JSON rows
  convert them to ``None``.
  """
  n = len(t)
  state = {str(robot): "" for robot in robots}
  by_index = [[] for _ in range(n)]
  for order, event in enumerate(events):
    kind = event.get("event")
    robot = str(event.get("robot", ""))
    if kind not in ("close_start", "open_start") or robot not in state:
      continue
    # A phase-scheduled close begins before the attachment mode switch. New recordings retain
    # that distinction in the close event; legacy events still attach at close_start.
    timing_event = event
    if kind == "close_start" and "attachment_physics_tick" in event:
      timing_event = dict(event)
      timing_event["physics_tick"] = event["attachment_physics_tick"]
      if "attachment_time_s" in event:
        timing_event["time_s"] = event["attachment_time_s"]
      if "attachment_frame" in event:
        timing_event["frame"] = event["attachment_frame"]
    et = _event_time(timing_event, physics_dt=physics_dt, time_origin=time_origin, source_t=t)
    index = int(np.searchsorted(t, et, side="right") - 1)
    index = min(max(index, 0), n - 1)
    by_index[index].append((order, kind, robot, str(event.get("part", ""))))

  width = max([len(str(x)) for e in events for x in [e.get("part", "")] if x] + [1])
  out = np.full((n, len(robots)), "", dtype=f"U{width}")
  robot_index = {str(robot): i for i, robot in enumerate(robots)}
  for i in range(n):
    for _order, kind, robot, part in sorted(by_index[i], key=lambda row: row[0]):
      state[robot] = part if kind == "close_start" else ""
    for robot, j in robot_index.items():
      out[i, j] = state[robot]
  return out


def _json_samples(q, t, hinge, controller_active, events, robots, attached_objects, *, dt,
                  grasp_wait, physics_dt, time_origin):
  """Uniform row-oriented JSON samples with explicit post-close dwell intervals.

  Close commands at the same source time are concurrent and therefore insert one global dwell,
  not one dwell per robot. Opening is a one-sample command; closing remains asserted throughout
  its dwell so a simple player can issue the command and wait before advancing the arm path. For
  a handover, the giver's atomic source open is delayed to the end of the receiver's dwell.
  """
  dt = float(dt)
  grasp_wait = float(grasp_wait)
  if not np.isfinite(dt) or dt <= 0.0:
    raise ValueError("JSON sample dt must be finite and positive")
  if not np.isfinite(grasp_wait) or grasp_wait < 0.0:
    raise ValueError("grasp wait must be finite and nonnegative")

  tagged = []
  for order, event in enumerate(events):
    kind, robot = event.get("event"), event.get("robot")
    if kind not in ("close_start", "open_start") or robot not in robots:
      continue
    tagged.append((_event_time(event, physics_dt=physics_dt, time_origin=time_origin,
                               source_t=t), order, str(kind), str(robot),
                   str(event.get("part", ""))))

  # Simulation performs a handover's two logical weld changes at one physics tick. Hardware
  # cannot: the receiver must finish closing before the giver starts opening. Identify only
  # same-part close/open pairs here; unrelated parallel pick/release events need no ordering.
  handover_opens = {}
  for et, order, kind, robot, part in tagged:
    if kind != "open_start" or not part:
      continue
    receivers = {
        cr for ct, _co, ck, cr, cp in tagged
        if ck == "close_start" and cr != robot and cp == part
        and abs(ct - et) <= 0.5 * physics_dt
    }
    if receivers:
      handover_opens[order] = (et, robot, part, receivers)
  if handover_opens and grasp_wait <= 0.0:
    raise ValueError("a handover requires positive --grasp-wait so the receiver can close "
                     "before the giver opens")

  # Group simultaneous close commands before constructing the time warp.
  close_groups = []
  for et, _order, kind, robot, _part in sorted(tagged):
    if kind != "close_start":
      continue
    if close_groups and abs(et - close_groups[-1][0]) <= 0.5 * physics_dt:
      close_groups[-1][1].add(robot)
    else:
      close_groups.append([et, {robot}])

  def shifted_event_time(source_time):
    return source_time + grasp_wait * sum(c < source_time - 0.5 * physics_dt
                                          for c, _rs in close_groups)

  duration = float(t[-1]) + grasp_wait * len(close_groups)
  # Ceil and clamp the source time at the end: every timestamp remains exactly dt-spaced while
  # the final configuration is never lost merely because the source duration is off-grid.
  count = int(np.ceil(duration / dt - 1e-12)) + 1
  tout = np.arange(count, dtype=float) * dt
  source_at = np.empty(count, dtype=float)
  closing = [set() for _ in range(count)]
  for i, ot in enumerate(tout):
    shift = 0.0
    mapped = None
    for close_t, close_robots in close_groups:
      close_out = close_t + shift
      if ot < close_out - 1e-12:
        mapped = ot - shift
        break
      if grasp_wait > 0.0 and ot < close_out + grasp_wait - 1e-12:
        mapped = close_t
        closing[i].update(close_robots)
        break
      if grasp_wait == 0.0 and abs(ot - close_out) <= 0.5 * dt:
        closing[i].update(close_robots)
      shift += grasp_wait
    if mapped is None:
      mapped = ot - shift
    source_at[i] = min(max(mapped, 0.0), float(t[-1]))

  # Guarantee a command sample even when an off-grid close has a zero dwell.
  if grasp_wait == 0.0:
    for close_t, close_robots in close_groups:
      idx = min(int(np.floor(shifted_event_time(close_t) / dt + 0.5)), count - 1)
      closing[idx].update(close_robots)

  qc = np.asarray(q, dtype=float).copy()
  if np.any(hinge):
    qc[:, hinge] = np.unwrap(qc[:, hinge], axis=0)
  qout = np.column_stack([np.interp(source_at, t, qc[:, j])
                          for j in range(qc.shape[1])])
  qd, qdd = _differentiate(qout, tout, hinge)

  cin = np.asarray(controller_active, dtype=bool)
  source_indices = np.searchsorted(t, source_at, side="right") - 1
  source_indices = np.clip(source_indices, 0, len(t) - 1)
  servo = cin[source_indices]
  attached = np.asarray(attached_objects)[source_indices].copy()

  opening = [set() for _ in range(count)]
  for et, order, kind, robot, _part in tagged:
    if kind != "open_start":
      continue
    # The source handover is atomic, but its hardware realization is not. The close dwell is
    # inserted immediately after this source instant; put the giver's open command at its end.
    out_time = shifted_event_time(et) + (grasp_wait if order in handover_opens else 0.0)
    # Never round an opening command *backwards* across its event time. In particular, an
    # off-grid handover dwell endpoint rounded to the nearest sample can still lie inside the
    # receiver's close interval, recreating the unsafe overlap this expansion is meant to fix.
    idx = min(int(np.ceil(out_time / dt - 1e-12)), count - 1)
    opening[idx].add(robot)

  # At an atomic source handover the source trace already says "receiver only" at the event.
  # During the expanded hardware dwell, the giver in fact remains attached while the receiver
  # closes. Represent that intentional co-grasp interval, and let the source trace take over at
  # the end when the delayed open command fires.
  robot_index = {str(robot): i for i, robot in enumerate(robots)}
  for _order, (et, giver, part, receivers) in handover_opens.items():
    begin, end = shifted_event_time(et), shifted_event_time(et) + grasp_wait
    overlap = np.flatnonzero((tout >= begin - 1e-12) & (tout < end - 1e-12))
    attached[overlap, robot_index[giver]] = part
    for receiver in receivers:
      attached[overlap, robot_index[receiver]] = part

  samples = []
  for i in range(count):
    samples.append({
        "q": qout[i].tolist(),
        "qd": qd[i].tolist(),
        "qdd": qdd[i].tolist(),
        "time": float(tout[i]),
        "opening_gripper": sorted(opening[i]),
        "closing_gripper": sorted(closing[i]),
        "servo_controller": bool(servo[i]),
        "attached_objects": {
            robot: (str(attached[i, j]) if str(attached[i, j]) else None)
            for j, robot in enumerate(robots)
        },
    })
  return samples


def _json_samples_preserved(q, t, hinge, controller_active, events, robots,
                            attached_objects, *, dt, grasp_wait, physics_dt, time_origin):
  """Downsample a source trajectory whose robot-local gripper waits are already scheduled.

  No time warp is applied: doing so would alter relative robot timing after the phase
  scheduler's collision certificate. Close annotations cover the recorded command-to-attach
  interval; opening remains an edge-triggered command at its recorded source time.
  """
  dt, grasp_wait = float(dt), float(grasp_wait)
  if not np.isfinite(dt) or dt <= 0.0:
    raise ValueError("JSON sample dt must be finite and positive")
  if not np.isfinite(grasp_wait) or grasp_wait < 0.0:
    raise ValueError("grasp wait must be finite and nonnegative")
  count = int(np.ceil(float(t[-1]) / dt - 1e-12)) + 1
  tout = np.arange(count, dtype=float) * dt
  source_at = np.minimum(tout, float(t[-1]))
  qc = np.asarray(q, dtype=float).copy()
  if np.any(hinge):
    qc[:, hinge] = np.unwrap(qc[:, hinge], axis=0)
  qout = np.column_stack([np.interp(source_at, t, qc[:, j])
                          for j in range(qc.shape[1])])
  qd, qdd = _differentiate(qout, tout, hinge)
  source_indices = np.clip(
      np.searchsorted(t, source_at, side="right") - 1, 0, len(t) - 1)
  servo = np.asarray(controller_active, dtype=bool)[source_indices]
  attached = np.asarray(attached_objects)[source_indices].copy()
  closing = [set() for _ in range(count)]
  opening = [set() for _ in range(count)]
  for event in events:
    kind, robot = event.get("event"), str(event.get("robot", ""))
    if kind not in ("close_start", "open_start") or robot not in robots:
      continue
    begin = _event_time(event, physics_dt=physics_dt, time_origin=time_origin, source_t=t)
    if kind == "open_start":
      opening[min(int(np.ceil(begin / dt - 1e-12)), count - 1)].add(robot)
      continue
    if "attachment_physics_tick" in event:
      end_event = dict(event)
      end_event["physics_tick"] = event["attachment_physics_tick"]
      end = _event_time(
          end_event, physics_dt=physics_dt, time_origin=time_origin, source_t=t)
    else:
      end = min(begin + grasp_wait, float(t[-1]))
    hits = np.flatnonzero((tout >= begin - 0.5 * dt) & (tout < end - 1e-12))
    if not len(hits):
      hits = [min(int(np.ceil(begin / dt - 1e-12)), count - 1)]
    for i in hits:
      closing[int(i)].add(robot)

  return [{
      "q": qout[i].tolist(), "qd": qd[i].tolist(), "qdd": qdd[i].tolist(),
      "time": float(tout[i]), "opening_gripper": sorted(opening[i]),
      "closing_gripper": sorted(closing[i]), "servo_controller": bool(servo[i]),
      "attached_objects": {
          robot: (str(attached[i, j]) if str(attached[i, j]) else None)
          for j, robot in enumerate(robots)},
  } for i in range(count)]


def _legacy_gripper_events(z, robots, t, known_events=()):
  """Recover missing logical gripper commands from legacy action marks.

  Old recordings marked every completed symbolic action but, depending on their age, either
  logged no gripper events or logged closes only.  A pick closes its actor; assemble/place/
  release open their actor.  A handover closes its receiver and opens the previous holder,
  but the latter is emitted only when the accumulated command state makes that giver unique.
  Exact recorded events always win; inferred events merely fill an entirely missing event
  kind and retain the old recording-frame resolution via ``inferred=True``.
  """
  known_events = [dict(e) for e in known_events]
  need_close = not any(e.get("event") == "close_start" for e in known_events)
  need_open = not any(e.get("event") == "open_start" for e in known_events)
  if not (need_close or need_open):
    return []

  parsed = []
  for frame, label in _json_scalar(z, "marks", []):
    if str(label).endswith(" MISS"):
      continue                         # no weld fired, hence no hardware grip transition
    try:
      action = ast.literal_eval(str(label))
    except (SyntaxError, ValueError):
      continue
    if not isinstance(action, tuple) or len(action) < 3:
      continue
    kind, robot, part = str(action[0]), str(action[1]), str(action[2])
    if robot not in robots or kind not in ("pick", "handover", "assemble", "place", "release"):
      continue
    i = min(max(int(frame), 0), len(t) - 1)
    parsed.append((i, kind, robot, part))

  # Reconstruct the logical closed/open state. Exact close records are needed here even when
  # only opens are missing (the common 2026 legacy format).
  closes_at = {}
  opens_at = {}
  for e in known_events:
    kind, robot = e.get("event"), str(e.get("robot", ""))
    if robot not in robots or kind not in ("close_start", "open_start"):
      continue
    i = min(max(int(e.get("frame", 0)), 0), len(t) - 1)
    (closes_at if kind == "close_start" else opens_at).setdefault(i, []).append(e)

  def inferred(kind, robot, part, frame):
    return {"event": kind, "robot": robot, "part": part, "frame": int(frame),
            "time_s": float(t[frame]), "inferred": True}

  out = []
  active = {}                         # robot -> last explicitly/logically grasped part
  event_frames = sorted(set(closes_at) | set(opens_at) | {x[0] for x in parsed})
  actions_at = {}
  for row in parsed:
    actions_at.setdefault(row[0], []).append(row[1:])
  for i in event_frames:
    for e in closes_at.get(i, ()):
      active[str(e["robot"])] = str(e.get("part", ""))
    for e in opens_at.get(i, ()):
      active.pop(str(e["robot"]), None)
    for kind, robot, part in actions_at.get(i, ()):
      if kind in ("pick", "handover") and need_close:
        out.append(inferred("close_start", robot, part, i))
        active[robot] = part
      if kind == "handover" and need_open:
        # Prefer an exact same-part holder. If group re-rooting changed the named part, a
        # single other active robot is still unambiguous. Never guess among multiple holders.
        candidates = [r for r, p in active.items() if r != robot and p == part]
        if len(candidates) != 1:
          candidates = [r for r in active if r != robot]
        if len(candidates) == 1:
          giver = candidates[0]
          out.append(inferred("open_start", giver, active[giver], i))
          active.pop(giver, None)
      elif kind in ("assemble", "place", "release") and need_open:
        out.append(inferred("open_start", robot, active.get(robot, part), i))
        active.pop(robot, None)
  return out


def export_recording(source, destination, *, allow_failed=False, sample_dt=0.05,
                     grasp_wait=0.5, grasp_wait_mode="auto"):
  source, destination = Path(source), Path(destination)
  with np.load(source, allow_pickle=False) as z:
    required = {"xml", "qpos", "robots"}
    missing = sorted(required - set(z.files))
    if missing:
      raise ValueError(f"{source} is not a sim recording; missing {missing}")
    metadata = _json_scalar(z, "metadata", {})
    source_local_wait = bool(
        metadata.get("wait_for_gripper")
        and (metadata.get("phase_schedule")
             or (metadata.get("resolved_profile") or {}).get("phase_schedule")))
    if grasp_wait_mode not in ("auto", "insert", "preserve"):
      raise ValueError("grasp_wait_mode must be auto, insert, or preserve")
    resolved_wait_mode = ("preserve" if source_local_wait else "insert") \
        if grasp_wait_mode == "auto" else grasp_wait_mode
    if resolved_wait_mode == "preserve" and not source_local_wait:
      raise ValueError(
          "--grasp-wait-mode preserve requires a recording made with "
          "--phase-schedule --wait-for-gripper")
    source_grasp_wait = (float(metadata.get("gripper_motion_time", grasp_wait))
                         if resolved_wait_mode == "preserve" else float(grasp_wait))
    source_success = (bool(np.asarray(z["success"]).item())
                      if "success" in z.files else None)
    if source_success is False and not allow_failed:
      raise ValueError(
          "source simulation reports failure; refusing an open-loop export. "
          "Use --allow-failed only to create a diagnostic trajectory.")
    robots = [str(x) for x in np.asarray(z["robots"]).tolist()]
    model = mujoco.MjModel.from_xml_string(str(z["xml"]))
    rows, slices = _robot_joints(model, robots)
    qpos = np.asarray(z["qpos"], dtype=float)
    q = qpos[:, [r[2] for r in rows]]

    physics_dt = float(metadata.get("dt", 0.002))
    record_every = int(metadata.get("record_every", 8))
    if "physics_tick" in z.files and len(z["physics_tick"]) == len(q):
      ticks = np.asarray(z["physics_tick"], dtype=np.int64)
      t = (ticks - ticks[0]).astype(float) * physics_dt
      time_origin = int(ticks[0])
    else:
      t = np.arange(len(q), dtype=float) * physics_dt * record_every
      time_origin = 0

    qd, qdd = _differentiate(q, t, np.asarray([r[3] for r in rows], dtype=bool))

    if "controller_mode" in z.files and len(z["controller_mode"]) == len(q):
      controller = np.asarray(z["controller_mode"], dtype="U24")
      annotation_source = "recorded"
    else:
      # Do not silently call a legacy trajectory open-loop-safe: the missing interval trace
      # means we cannot know where a feedback hook changed the reference.
      controller = np.full(len(q), "unknown", dtype="U24")
      annotation_source = "missing"
    controller_active = controller != _OPEN_LOOP_MODE

    events = _json_scalar(z, "gripper_events", [])
    events = list(events) + _legacy_gripper_events(z, robots, t, events)
    attached_objects = _attachment_trace(
        t, events, robots, physics_dt=physics_dt, time_origin=time_origin)
    close = np.zeros((len(q), len(robots)), dtype=bool)
    for event in events:
      if event.get("event") != "close_start" or event.get("robot") not in robots:
        continue
      if "physics_tick" in event:
        et = (int(event["physics_tick"]) - time_origin) * physics_dt
        i = int(np.searchsorted(t, et, side="left"))
      else:
        i = int(event.get("frame", 0))
      i = min(max(i, 0), len(q) - 1)
      close[i, robots.index(event["robot"])] = True

    src_meta = dict(metadata)
    out_meta = {
        "schema": SCHEMA,
        "source_recording": str(source),
        "source_signal": "simulated_joint_qpos",
        "sample_period_s": (float(np.median(np.diff(t))) if len(t) > 1 else None),
        "derivatives": "numpy.gradient; hinge columns unwrapped; no smoothing",
        "controller_annotation": annotation_source,
        "controller_active_semantics": "controller_mode != 'trajectory'",
        "source_success": source_success,
        "source_metadata": src_meta,
    }

  destination.parent.mkdir(parents=True, exist_ok=True)
  if destination.suffix.lower() == ".json":
    # JSON is the compact hardware handoff: one self-contained record per 20Hz command sample.
    # Keep the model-order metadata outside the list so each row only carries execution data.
    _sample_fn = (_json_samples_preserved
                  if resolved_wait_mode == "preserve" else _json_samples)
    samples = _sample_fn(
        q, t, np.asarray([r[3] for r in rows], dtype=bool), controller_active, events,
        robots, attached_objects, dt=sample_dt, grasp_wait=source_grasp_wait,
        physics_dt=physics_dt, time_origin=time_origin)
    payload = {
        "schema": JSON_SCHEMA,
        "dt": float(sample_dt),
        "grasp_wait": source_grasp_wait,
        "grasp_wait_mode": resolved_wait_mode,
        "robots": robots,
        "joint_names": [r[1] for r in rows],
        "robot_slices": slices,
        "samples": samples,
    }
    with destination.open("w", encoding="utf-8") as stream:
      json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
      stream.write("\n")
  elif destination.suffix.lower() == ".npz":
    np.savez_compressed(
        destination,
        schema=np.asarray(SCHEMA),
        time=t,
        robots=np.asarray(robots),
        joint_names=np.asarray([r[1] for r in rows]),
        robot_slices=np.asarray(json.dumps(slices, sort_keys=True)),
        q=q, qd=qd, qdd=qdd,
        attached_objects=attached_objects,
        controller_mode=controller,
        controller_active=controller_active,
        gripper_close_start=close,
        gripper_events=np.asarray(json.dumps(events, sort_keys=True)),
        metadata=np.asarray(json.dumps(out_meta, sort_keys=True)))
  else:
    raise ValueError("destination must end in .npz or .json")
  _count = len(samples) if destination.suffix.lower() == ".json" else len(q)
  _dt = float(sample_dt) if destination.suffix.lower() == ".json" else out_meta["sample_period_s"]
  print(f"[open-loop] {_count} samples, {q.shape[1]} arm joints, "
        f"dt={_dt:.6g}s -> {destination}")
  modes, counts = np.unique(controller, return_counts=True)
  print("[open-loop] controller modes: " +
        ", ".join(f"{m}={int(n)}" for m, n in zip(modes, counts)))
  _nopen = sum(e.get("event") == "open_start" for e in events)
  print(f"[open-loop] gripper close/open-start events: {int(close.sum())}/{_nopen}; "
        f"source success={out_meta['source_success']}")
  if annotation_source != "recorded":
    print("[open-loop] WARNING: source predates controller tracing; controller intervals are "
          "UNKNOWN and must not be treated as open-loop-safe")
  return out_meta


def parse_args(argv=None):
  p = argparse.ArgumentParser(
      description="Export robot q/qd/qdd and controller/gripper annotations from a sim NPZ")
  p.add_argument("source", help="sim.py recording (.npz)")
  p.add_argument("destination", help="output open-loop trajectory (.npz or .json)")
  p.add_argument(
      "--allow-failed", action="store_true",
      help="export a failed simulation for diagnostics (never silently hardware-ready)")
  p.add_argument("--dt", type=float, default=0.05, metavar="SEC",
                 help="uniform JSON sample period (default 0.05s = 20Hz; NPZ is unchanged)")
  p.add_argument("--grasp-wait", type=float, default=0.5, metavar="SEC",
                 help="JSON-only wait used in insert mode (default 0.5s); preserve mode reads "
                      "the source recording's gripper motion time")
  p.add_argument("--grasp-wait-mode", choices=("auto", "insert", "preserve"), default="auto",
                 help="auto preserves collision-certified robot-local waits when present, "
                      "otherwise inserts the legacy global dwell; preserve refuses an "
                      "unscheduled source (default: auto)")
  return p.parse_args(argv)


if __name__ == "__main__":
  args = parse_args()
  export_recording(args.source, args.destination, allow_failed=args.allow_failed,
                   sample_dt=args.dt, grasp_wait=args.grasp_wait,
                   grasp_wait_mode=args.grasp_wait_mode)
