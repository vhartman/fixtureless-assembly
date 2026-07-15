"""Serialize / deserialize an assembly plan together with the scene CONFIG needed to
reconstruct the exact environment for a later replay.

The persisted JSON is `{"config": {...}, "plan": [[action...], ...]}`. `config` holds the
scene-determining knobs (assembly, parts, robots, seed, scatter, wrench settings, ...) so
replay.py can rebuild the identical scene; `plan` is the discrete action list produced by
solve_assembly_problem (tuples <-> lists across JSON). Written by main.py --save, read by
replay.py."""
import json


def write_plan(path, config, plan):
  with open(path, "w") as f:
    json.dump({"config": config, "plan": [list(step) for step in plan]}, f, indent=2)


def read_plan(path):
  with open(path) as f:
    data = json.load(f)
  return data["config"], [tuple(step) for step in data["plan"]]
