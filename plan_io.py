"""Serialize / deserialize an assembly plan together with the scene CONFIG needed to
reconstruct the exact environment for a later replay.

The persisted JSON is `{"config": {...}, "plan": [[action...], ...]}` plus an optional
`"witnesses"` block. `config` holds the scene-determining knobs (assembly, parts, robots,
seed, scatter, wrench settings, ...) so replay.py can rebuild the identical scene; `plan` is
the discrete action list produced by solve_assembly_problem (tuples <-> lists across JSON);
`witnesses` (when present) persists the OPTIMIZED per-action IK keyframes, so the motion-plan
and sim stages are pure functions of this file instead of re-solving the witness pass.
Written by main.py --save, read by replay.py / main.py --load-plan / sim.py."""
import json


def witness_block(joint_names, witnesses):
  """Pack per-action full-config witnesses for persistence. Entry i belongs to plan[i];
  None entries (actions without an IK keyframe) stay null. `joint_names` records the joint
  ORDER the vectors are expressed in -- readers must refuse the block if their scene's joint
  order differs (a silent reorder would pose every arm wrong)."""
  return {"joints": list(joint_names),
          "wit": [None if w is None else [float(v) for v in w] for w in witnesses]}


def write_plan(path, config, plan, witnesses=None):
  data = {"config": config, "plan": [list(step) for step in plan]}
  if witnesses is not None:
    data["witnesses"] = witnesses
  with open(path, "w") as f:
    json.dump(data, f, indent=2)


def read_plan(path):
  with open(path) as f:
    data = json.load(f)
  return data["config"], [tuple(step) for step in data["plan"]]


def read_witnesses(path, joint_names, n_actions):
  """The persisted witness list (np arrays / None per action), or None -- with the reason
  printed -- when the file has no block or the block does not fit the current scene/plan."""
  import numpy as np
  with open(path) as f:
    data = json.load(f)
  blk = data.get("witnesses")
  if blk is None:
    return None
  if list(blk.get("joints") or []) != list(joint_names):
    print(f"[witness-reuse] {path}: saved joint order differs from the current scene -- "
          "ignoring the persisted witnesses.")
    return None
  wit = blk.get("wit") or []
  if len(wit) != n_actions:
    print(f"[witness-reuse] {path}: {len(wit)} witness entries vs {n_actions} plan actions "
          "-- ignoring the persisted witnesses.")
    return None
  return [None if w is None else np.asarray(w, dtype=float) for w in wit]
