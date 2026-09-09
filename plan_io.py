"""Serialize / deserialize an assembly plan together with the scene CONFIG needed to
reconstruct the exact environment for a later replay.

The persisted JSON is `{"config": {...}, "plan": [[action...], ...]}` plus an optional
`"witnesses"` block. `config` holds the scene-determining knobs (assembly, parts, robots,
seed, scatter, wrench settings, ...) so replay.py can rebuild the identical scene; `plan` is
the discrete action list produced by solve_assembly_problem (tuples <-> lists across JSON);
`witnesses` (when present) persists the exact SEARCH per-action IK certificates, optionally
replaced by feasibility-safe optimized keyframes, so motion-plan and sim stages are pure
functions of this file instead of re-solving the witness pass.
Written by main.py --save, read by replay.py / main.py --load-plan / sim.py."""
import json


# Keep this deliberately local instead of importing solver._IK_ACTIONS: plan persistence is a
# low-level module and must remain usable without importing robotic/RAI.  These are precisely
# the actions for which RobotIK.bind produces a keyframe.
IK_WITNESS_ACTIONS = frozenset(
    {"pick", "assemble", "place", "set_down", "add_grasp", "handover"})


def missing_required_witnesses(plan, witnesses):
  """Indices of IK actions whose persisted keyframe is absent.

  A null is correct for release and other purely discrete actions.  It is never a valid result
  for an IK action: downstream code otherwise silently drops the corresponding task and only
  much later reports a mode/path mismatch.
  """
  if witnesses is None or len(witnesses) != len(plan):
    return [i for i, a in enumerate(plan) if a and a[0] in IK_WITNESS_ACTIONS]
  return [i for i, a in enumerate(plan)
          if a and a[0] in IK_WITNESS_ACTIONS and witnesses[i] is None]


def require_complete_witnesses(plan, witnesses, *, who="witness export"):
  """Fail closed unless every IK action has an aligned witness; return ``witnesses``."""
  if witnesses is None:
    raise ValueError(f"{who}: no witness list was produced")
  if len(witnesses) != len(plan):
    raise ValueError(
        f"{who}: {len(witnesses)} witness entries for {len(plan)} plan actions")
  missing = missing_required_witnesses(plan, witnesses)
  if missing:
    detail = ", ".join(f"{i}:{tuple(plan[i])}" for i in missing[:8])
    if len(missing) > 8:
      detail += f", ... ({len(missing)} total)"
    raise ValueError(f"{who}: required IK witnesses are missing at {detail}")
  return witnesses


def witness_block(joint_names, witnesses, alts=None, stage=None, skill_paths=None,
                  pre_witnesses=None):
  """Pack per-action full-config witnesses for persistence. Entry i belongs to plan[i];
  None entries (actions without an IK keyframe) stay null. `joint_names` records the joint
  ORDER the vectors are expressed in -- readers must refuse the block if their scene's joint
  order differs (a silent reorder would pose every arm wrong).

  `alts` (--witness-sets): {plan_index: [full vectors, member 0 == the canonical witness]}
  adds an OPTIONAL sibling key; alts=None leaves the block byte-identical to before, and old
  files without the key load unchanged."""
  out = {"joints": list(joint_names),
         "wit": [None if w is None else [float(v) for v in w] for w in witnesses]}
  if stage is not None:
    if stage not in ("search", "optimized"):
      raise ValueError(f"unknown witness stage {stage!r}")
    out["stage"] = stage
  if alts:
    out["wit_alts"] = {str(i): [[float(v) for v in m] for m in ms]
                       for i, ms in sorted(alts.items())}
  if skill_paths:
    # Path-valued skill witnesses (currently screw) are part of the certificate, not an
    # export-time IK request.  Entry i is ordered entry->terminal and its last member must
    # equal wit[i].  Keeping it in the same joint-order block gives readers one provenance
    # check for endpoints and continuations alike.
    out["skill_paths"] = {
        str(i): [[float(v) for v in q] for q in path]
        for i, path in sorted(skill_paths.items())}
  if pre_witnesses:
    # Short execution funnels are certificates too. In particular, an assemble witness is
    # only useful when the exact pre-insertion mouth which search collision-checked against it
    # survives serialization. Re-solving this endpoint in an MP/replay process changes the
    # certified edge and reintroduces stochastic IK into artifact consumption.
    out["pre_wit"] = {
        str(i): [float(v) for v in q]
        for i, q in sorted(pre_witnesses.items()) if q is not None}
  return out


def read_witness_stage(path):
  """`search` or `optimized`; old pre-marker witness blocks were optimized exports."""
  with open(path) as f:
    blk = json.load(f).get("witnesses") or {}
  return blk.get("stage", "optimized") if blk else None


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


def read_skill_paths(path, joint_names, n_actions, witnesses=None):
  """Read aligned path-valued skill certificates, or ``None``.

  A malformed path is a hard artifact error: unlike an optional witness set, silently
  dropping it would authorize downstream IK re-solving and defeat replay reproducibility.
  """
  import numpy as np
  with open(path) as f:
    data = json.load(f)
  blk = data.get("witnesses") or {}
  raw = blk.get("skill_paths")
  if not raw:
    return None
  if list(blk.get("joints") or []) != list(joint_names):
    raise ValueError(f"{path}: skill-path joint order differs from the current scene")
  out = {}
  for key, rows in raw.items():
    i = int(key)
    if not (0 <= i < n_actions) or len(rows) < 2:
      raise ValueError(f"{path}: invalid skill path index/length at {key}")
    qs = [np.asarray(q, dtype=float) for q in rows]
    if any(q.shape != (len(joint_names),) for q in qs):
      raise ValueError(f"{path}: wrong-length skill-path member at step {i}")
    if witnesses is not None:
      if i >= len(witnesses) or witnesses[i] is None:
        raise ValueError(f"{path}: skill path at step {i} has no canonical witness")
      if not np.allclose(qs[-1], np.asarray(witnesses[i]), atol=1e-9, rtol=0.0):
        raise ValueError(f"{path}: skill path at step {i} does not end at its witness")
    out[i] = qs
  return out or None


def read_pre_witnesses(path, joint_names, n_actions, witnesses=None):
  """Read exact pre-skill endpoint certificates aligned to persisted witnesses.

  Unlike optional witness-set diversity, malformed pre-witness data is a hard artifact error:
  silently dropping it would cause export to redo IK and certify a different local edge.
  """
  import numpy as np
  with open(path) as f:
    data = json.load(f)
  blk = data.get("witnesses") or {}
  raw = blk.get("pre_wit")
  if not raw:
    return None
  if list(blk.get("joints") or []) != list(joint_names):
    raise ValueError(f"{path}: pre-witness joint order differs from the current scene")
  out = {}
  for key, row in raw.items():
    i = int(key)
    if not (0 <= i < n_actions):
      raise ValueError(f"{path}: invalid pre-witness index {key}")
    q = np.asarray(row, dtype=float)
    if q.shape != (len(joint_names),):
      raise ValueError(f"{path}: wrong-length pre-witness at step {i}")
    if witnesses is not None and (i >= len(witnesses) or witnesses[i] is None):
      raise ValueError(f"{path}: pre-witness at step {i} has no canonical skill witness")
    out[i] = q
  return out or None


# --- Measured pickup layout ------------------------------------------------
# A layout file records where the parts and the table REALLY are, as measured on the robot
# by the teleop package's pickup_calib tool. It replaces the synthetic scatter so the plan is
# solved for the physical cell. Coordinates are metres/radians in the robot base frame, which
# is the same frame as the rai world of env.make_husky_duo_env (z=0 floor).
LAYOUT_SCHEMA = "assembly-pickup-layout-v1"


def load_layout_file(path):
  """Read a measured pickup layout and check that it is one.

  Args:
    path (str): Path to the layout JSON.

  Returns:
    dict: The layout, with "table" and "parts" guaranteed present.

  Raises:
    ValueError: If the schema marker or a required block is missing.
  """
  with open(path) as f:
    layout = json.load(f)
  schema = layout.get("schema")
  if schema != LAYOUT_SCHEMA:
    raise ValueError(f"{path}: schema {schema!r} is not {LAYOUT_SCHEMA!r}")
  for key in ("table", "parts"):
    if key not in layout:
      raise ValueError(f"{path}: layout has no {key!r} block")
  return layout


def read_plan_layout(path):
  """The layout block a saved plan carries, or None if it was solved without one.

  The block travels inside the plan's config so that --load-plan reproduces the measured
  scene without the operator having to pass the layout file again.

  Args:
    path (str): Path to a saved plan JSON.

  Returns:
    dict or None: The layout block.
  """
  with open(path) as f:
    data = json.load(f)
  return (data.get("config") or {}).get("layout")
