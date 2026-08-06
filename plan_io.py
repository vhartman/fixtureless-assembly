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


def witness_block(joint_names, witnesses, alts=None):
  """Pack per-action full-config witnesses for persistence. Entry i belongs to plan[i];
  None entries (actions without an IK keyframe) stay null. `joint_names` records the joint
  ORDER the vectors are expressed in -- readers must refuse the block if their scene's joint
  order differs (a silent reorder would pose every arm wrong).

  `alts` (--witness-sets): {plan_index: [full vectors, member 0 == the canonical witness]}
  adds an OPTIONAL sibling key; alts=None leaves the block byte-identical to before, and old
  files without the key load unchanged."""
  out = {"joints": list(joint_names),
         "wit": [None if w is None else [float(v) for v in w] for w in witnesses]}
  if alts:
    out["wit_alts"] = {str(i): [[float(v) for v in m] for m in ms]
                       for i, ms in sorted(alts.items())}
  return out


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


def read_witness_sets(path, joint_names, n_actions):
  """The persisted witness SETS ({plan_index: [np member vectors]}), or None. Validated like
  read_witnesses (joint order, index range, member length); a step whose member 0 disagrees
  with the canonical witness is dropped with a print -- some later pass rewrote the canonical
  and the set no longer describes it."""
  import numpy as np
  with open(path) as f:
    data = json.load(f)
  blk = data.get("witnesses") or {}
  alts = blk.get("wit_alts")
  if not alts:
    return None
  if list(blk.get("joints") or []) != list(joint_names):
    print(f"[witsets-reuse] {path}: saved joint order differs -- ignoring witness sets.")
    return None
  wit = blk.get("wit") or []
  out = {}
  for k, ms in alts.items():
    i = int(k)
    if not (0 <= i < n_actions) or i >= len(wit) or wit[i] is None:
      print(f"[witsets-reuse] {path}: set at index {k} has no matching witness -- dropped.")
      continue
    members = [np.asarray(m, dtype=float) for m in ms]
    if any(len(m) != len(joint_names) for m in members):
      print(f"[witsets-reuse] {path}: set at index {k} has a wrong-length member -- dropped.")
      continue
    if not np.allclose(members[0], np.asarray(wit[i], dtype=float), atol=1e-9):
      print(f"[witsets-reuse] {path}: set at index {k}: member 0 no longer matches the "
            f"canonical witness -- dropped.")
      continue
    out[i] = members
  return out or None
