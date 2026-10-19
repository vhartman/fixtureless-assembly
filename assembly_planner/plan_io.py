"""Serialize / deserialize an assembly plan together with the scene CONFIG needed to
reconstruct the exact environment for a later replay.

The persisted JSON is `{"config": {...}, "plan": [[action...], ...]}` plus an optional
`"witnesses"` block. `config` holds the scene-determining knobs (assembly, parts, robots,
seed, scatter, wrench settings, ...) so replay.py can rebuild the identical scene; `plan` is
the discrete action list produced by solve_assembly_problem (tuples <-> lists across JSON);
`witnesses` (when present) persists the exact SEARCH per-action IK certificates, optionally
replaced by feasibility-safe optimized keyframes; `grasp_physics` carries the complete eager
handover table certificate; `motion_data` freezes the selected grasps and symbolic state
chain so downstream motion/sim stages do not regenerate the candidate grasp database.
Motion-plan and sim stages are therefore pure functions of this file and the scene models.
Written by main.py --save, read by replay.py / main.py --load-plan / sim.py."""
import json
import math
from pathlib import Path


# Keep this deliberately local instead of importing solver._IK_ACTIONS: plan persistence is a
# low-level module and must remain usable without importing robotic/RAI.  These are precisely
# the actions for which RobotIK.bind produces a keyframe.
IK_WITNESS_ACTIONS = frozenset(
    {"pick", "assemble", "place", "set_down", "add_grasp", "handover"})

# Semantic version of persisted pre-insertion endpoints. Older files may contain a mouth
# selected from the historical 100/80/60/40/20/10% fallback ladder. Its vector is structurally
# valid, but it does not certify that the moving part reached the configured skill boundary.
PRE_WITNESS_CONTRACT = "full-assemble-v1"


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
                  pre_witnesses=None, optimization_candidates=None,
                  diagnostic_candidates=None, grasp_closures=None):
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
  if optimization_candidates:
    # The witness optimiser's actual discrete ladder, kept separate from ``wit_alts``.
    # ``wit_alts`` is a post-pass GoalSet restricted to the same object pose/grasp for
    # motion planning; these entries are the broader keyframes the optimiser compared.
    # Member zero is normalized to the selected/canonical witness for straightforward
    # inspection and validation on replay.
    out["optimization_candidates"] = {
        str(i): [[float(v) for v in q] for q in qs]
        for i, qs in sorted(optimization_candidates.items())}
  if diagnostic_candidates:
    # Broad visualization-only pools. These include individually solved IK branches that a
    # later clearance/continuity/interchangeability gate rejected. Keep them distinct from
    # the optimizer ladder so an illustration cannot misrepresent what was optimized over.
    out["diagnostic_candidates"] = {
        str(i): [[float(v) for v in q] for q in qs]
        for i, qs in sorted(diagnostic_candidates.items())}
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
    out["pre_wit_contract"] = PRE_WITNESS_CONTRACT
  if grasp_closures:
    out["grasp_closures"] = {
        str(i): dict(cert) for i, cert in sorted(grasp_closures.items())}
  return out


def read_grasp_closures(path, plan, *, required=False):
  """Load and structurally validate saved two-finger closure/aperture certificates."""
  with open(path) as f:
    block = (json.load(f).get("witnesses") or {}).get("grasp_closures")
  if not block:
    if required:
      raise ValueError(
          "closure-certified plan has no persisted grasp apertures; regenerate the task "
          "plan or explicitly refresh its witnesses")
    return {}
  out = {}
  for raw_i, raw in block.items():
    try:
      i = int(raw_i)
      action = tuple(raw["action"])
      aperture = float(raw["aperture"])
    except (KeyError, TypeError, ValueError) as exc:
      raise ValueError(f"malformed grasp-closure certificate at {raw_i!r}") from exc
    if not 0 <= i < len(plan) or action != tuple(plan[i]):
      raise ValueError(
          f"grasp-closure certificate {i} is not aligned to its saved plan action")
    if action[0] not in {"pick", "add_grasp", "handover"}:
      raise ValueError(f"grasp-closure certificate {i} belongs to non-grasp action {action}")
    if not math.isfinite(aperture) or aperture < 0.0:
      raise ValueError(f"invalid saved aperture at plan step {i}: {aperture!r}")
    if raw.get("aperture_semantics") != "per-finger-inward-travel-m":
      raise ValueError(
          f"unsupported aperture semantics at step {i}: "
          f"{raw.get('aperture_semantics')!r}")
    from assembly_planner.gripper_closure import VERSION as closure_version
    if raw.get("contract") != f"virtual-close-v{closure_version}":
      raise ValueError(
          f"unsupported grasp-closure contract at step {i}: {raw.get('contract')!r}")
    if (raw.get("robot") != action[1] or raw.get("part") != action[2]
        or int(raw.get("grasp_id", -1)) != int(action[3])):
      raise ValueError(f"grasp-closure fields disagree with action at plan step {i}")
    out[i] = dict(raw, aperture=aperture)
  return out


def read_witness_stage(path):
  """`search` or `optimized`; old pre-marker witness blocks were optimized exports."""
  with open(path) as f:
    blk = json.load(f).get("witnesses") or {}
  return blk.get("stage", "optimized") if blk else None


def write_plan(path, config, plan, witnesses=None, grasp_physics=None, motion_data=None):
  data = {"config": config, "plan": [list(step) for step in plan]}
  if motion_data is not None:
    data["motion_data"] = motion_data
  if witnesses is not None:
    data["witnesses"] = witnesses
  if grasp_physics is not None:
    # Complete a-priori handover certificate. Consumers restore this before constructing any
    # search/replay context; physics is not an implicit side effect of loading a task plan.
    data["grasp_physics"] = grasp_physics
  # `--save experiments/new-run/plan.json` is a normal CLI workflow.  Persisting a
  # successfully solved certificate must not fail merely because the run directory was
  # not pre-created by the caller.
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  with target.open("w") as f:
    json.dump(data, f, indent=2)


def read_plan(path):
  with open(path) as f:
    data = json.load(f)
  return data["config"], [tuple(step) for step in data["plan"]]


def _motion_plan_digest(plan):
  import hashlib
  raw = json.dumps([list(a) for a in plan], separators=(",", ":")).encode()
  return hashlib.sha256(raw).hexdigest()


def build_motion_data(C, assembly, plan, robots, gripper_kind, *,
                      handovers=True, allow_place=True, bracing=False,
                      grasp_db_hash=None):
  """Freeze the symbolic state chain and ONLY the grasps used by a fixed plan.

  This is computed once while the search grasp table is available. Motion planning can
  subsequently build attachment modes without enumerating candidate grasps again.
  """
  import hashlib
  import numpy as np
  from assembly_planner.problem import grasp_pose
  from assembly_planner.solver import _replay_states, _search_context

  _checker, table = _search_context(C, assembly, robots)
  states = _replay_states(C, assembly, plan, robots, gripper_kind,
                          handovers, allow_place, bracing, table,
                          "saved motion-data certificate")
  selected = set()
  for state in states:
    selected.update((str(part), int(gid)) for part, gid in state.grasp.items())
    for contacts in state.supports.values():
      selected.update((str(c[2]), int(c[3])) for c in contacts
                      if c[0] == "grasp")
  for action in plan:
    if action and action[0] in {"pick", "add_grasp", "handover",
                                "inhand_reorient"}:
      selected.add((str(action[2]), int(action[3])))

  snapshots = []
  for state in states:
    snapshots.append({
        "parent": {p: list(v) for p, v in state.parent.items()},
        "grasp": {p: int(v) for p, v in state.grasp.items()},
        "place": {p: list(v) for p, v in state.place.items()},
        "supports": {p: [list(c) for c in sorted(cs)]
                     for p, cs in state.supports.items()},
        "poses": {p: np.asarray(v, dtype=float).tolist()
                  for p, v in state.poses.items()},
    })
  body = {
      "version": 1,
      "plan_sha256": _motion_plan_digest(plan),
      "grasp_db_sha256": grasp_db_hash,
      "states": snapshots,
      "selected_grasps": [
          [part, gid, np.asarray(grasp_pose(assembly, part, gid), dtype=float).tolist()]
          for part, gid in sorted(selected)],
  }
  raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
  body["sha256"] = hashlib.sha256(raw).hexdigest()
  return body


def read_motion_data(path, plan, *, grasp_db_hash=None):
  """Load a fixed-plan certificate; None means an old plan requiring legacy reconstruction."""
  import hashlib
  import numpy as np
  with open(path) as f:
    body = json.load(f).get("motion_data")
  if body is None:
    return None
  if body.get("version") != 1 or body.get("plan_sha256") != _motion_plan_digest(plan):
    raise ValueError(f"{path}: motion-data certificate does not match the saved action list")
  if body.get("grasp_db_sha256") != grasp_db_hash:
    raise ValueError(f"{path}: motion-data grasp provenance differs from plan config")
  payload = {k: v for k, v in body.items() if k != "sha256"}
  raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
  if hashlib.sha256(raw).hexdigest() != body.get("sha256"):
    raise ValueError(f"{path}: motion-data checksum mismatch")
  if len(body.get("states", ())) != len(plan) + 1:
    raise ValueError(f"{path}: motion-data state count does not match the plan")
  seen = set()
  for item in body.get("selected_grasps", ()):
    if not isinstance(item, list) or len(item) != 3:
      raise ValueError(f"{path}: malformed selected-grasp record")
    part, gid, pose = item
    key = (str(part), int(gid))
    if key in seen or np.asarray(pose).shape != (7,) or not np.all(np.isfinite(pose)):
      raise ValueError(f"{path}: invalid selected grasp {key}")
    seen.add(key)
  for state in body["states"]:
    for part, gid in state.get("grasp", {}).items():
      if (str(part), int(gid)) not in seen:
        raise ValueError(f"{path}: state references an unpersisted grasp")
    for contacts in state.get("supports", {}).values():
      for c in contacts:
        if c[0] == "grasp" and (str(c[2]), int(c[3])) not in seen:
          raise ValueError(f"{path}: support references an unpersisted grasp")
  return body


def motion_data_states(body, assembly):
  """Decode the exact state payload used to export attachment modes."""
  import numpy as np
  from assembly_planner.problem import State
  parts = set(assembly.objects)
  out = []
  for raw in body["states"]:
    if set(raw["parent"]) != parts or set(raw["poses"]) != parts:
      raise ValueError("motion-data state has a different part roster")
    poses = {p: np.asarray(v, dtype=float) for p, v in raw["poses"].items()}
    if any(v.shape != (7,) or not np.all(np.isfinite(v)) for v in poses.values()):
      raise ValueError("motion-data state has an invalid part pose")
    out.append(State(
        parent={p: tuple(v) for p, v in raw["parent"].items()},
        grasp={p: int(v) for p, v in raw["grasp"].items()},
        place={p: _motion_tuple(v) for p, v in raw["place"].items()},
        supports={p: frozenset(_motion_tuple(c) for c in cs)
                  for p, cs in raw["supports"].items()},
        poses=poses))
  return out


def _motion_tuple(value):
  return tuple(_motion_tuple(v) if isinstance(v, list) else v for v in value)


def motion_data_grasps(body):
  return {(part, int(gid)): pose for part, gid, pose in body["selected_grasps"]}


def read_grasp_physics(path):
  """Return the optional a-priori handover certificate carried by a task plan."""
  with open(path) as f:
    payload = json.load(f).get("grasp_physics")
  return payload


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


def _read_candidate_pools(path, joint_names, n_actions, field, tag):
  """Read a selected-witness-aligned diagnostic candidate-pool field."""
  import numpy as np
  with open(path) as f:
    data = json.load(f)
  blk = data.get("witnesses") or {}
  pools = blk.get(field)
  if not pools:
    return None
  if list(blk.get("joints") or []) != list(joint_names):
    print(f"[{tag}] {path}: saved joint order differs -- ignoring candidates.")
    return None
  wit = blk.get("wit") or []
  out = {}
  for k, raw in pools.items():
    i = int(k)
    if not (0 <= i < n_actions) or i >= len(wit) or wit[i] is None:
      print(f"[{tag}] {path}: pool at index {k} has no witness -- dropped.")
      continue
    members = [np.asarray(q, dtype=float) for q in raw]
    if (not members or
        any(q.ndim != 1 or len(q) != len(joint_names) for q in members)):
      print(f"[{tag}] {path}: pool at index {k} is malformed -- dropped.")
      continue
    if not np.allclose(members[0], np.asarray(wit[i], dtype=float), atol=1e-9, rtol=0.0):
      print(f"[{tag}] {path}: pool at index {k} no longer matches the selected "
            "witness -- dropped.")
      continue
    out[i] = members
  return out or None


def read_optimization_candidates(path, joint_names, n_actions):
  """Read the exact per-step candidate ladders considered by witness optimization."""
  return _read_candidate_pools(path, joint_names, n_actions,
                               "optimization_candidates", "witopt-candidates")


def read_diagnostic_candidates(path, joint_names, n_actions):
  """Read broad visualization-only IK branches rejected by later selection gates."""
  return _read_candidate_pools(path, joint_names, n_actions,
                               "diagnostic_candidates", "witdiag-candidates")


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


def read_pre_witnesses(path, joint_names, n_actions, witnesses=None, *,
                       require_full_assemble=False):
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
  if (require_full_assemble
      and blk.get("pre_wit_contract") != PRE_WITNESS_CONTRACT):
    raise ValueError(
        f"{path}: persisted pre-insertion endpoints predate the hard full-separation "
        "contract and may contain a collapsed funnel. Refusing to replay them; rerun the "
        "search, or pass --refresh-witnesses to recertify this fixed plan.")
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
