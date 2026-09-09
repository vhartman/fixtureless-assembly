#!/usr/bin/env python3
"""Report conservative joint-continuity metrics for persisted witness chains.

This is deliberately an artifact diagnostic, not a collision or motion-planning certificate.
It reads only the joint vectors saved in a plan and makes no attempt to interpolate them. Full
witness vectors are collision-scene certificates: inactive arms sit at the internal IK EXIT
pose, which is not a physical goal. Therefore this tool compares only consecutive witnessed
appearances of each action's named actor. This is a conservative lower-bound diagnostic; a
final MP export is required to include coupled holders exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(path: Path, threshold: float = np.pi / 2.0):
  with path.open(encoding="utf-8") as f:
    doc = json.load(f)
  block = doc.get("witnesses") or {}
  names = list(block.get("joints") or ())
  raw = list(block.get("wit") or ())
  if not names or not raw:
    raise ValueError(f"{path}: no persisted witness chain")
  plan = list(doc.get("plan") or ())
  indexed = [(i, np.asarray(q, dtype=float)) for i, q in enumerate(raw) if q is not None]
  if not indexed:
    raise ValueError(f"{path}: witness chain contains no joint vectors")
  bad = [(i, q.size) for i, q in indexed if q.size != len(names)]
  if bad:
    raise ValueError(f"{path}: witness dimensions do not match {len(names)} joints: {bad[:5]}")

  # NEVER compare adjacent full certificates: an idle arm's EXIT padding would be counted as
  # required motion. Group the guaranteed-active named actor's slices across its consecutive
  # witnessed appearances instead. Coupled non-actor holders are intentionally omitted rather
  # than guessed from an action tuple that does not encode the current support state.
  robot_idx = {}
  for k, name in enumerate(names):
    robot = name.split("_", 1)[0]
    robot_idx.setdefault(robot, []).append(k)
  by_actor = {}
  for i, q in indexed:
    if i >= len(plan) or len(plan[i]) < 2:
      continue
    actor = str(plan[i][1])
    if actor in robot_idx:
      by_actor.setdefault(actor, []).append((i, q))
  deltas, edges = [], []
  for actor, rows in by_actor.items():
    idx = np.asarray(robot_idx[actor], dtype=int)
    for (ia, qa), (ib, qb) in zip(rows, rows[1:]):
      d = np.zeros(len(names), dtype=float)
      d[idx] = np.abs(qb[idx] - qa[idx])
      deltas.append(d)
      edges.append((ia, ib))
  D = np.vstack(deltas) if deltas else np.zeros((0, len(names)))
  # Spans over all full certificates would have the same EXIT-padding problem. Compute spans
  # only from each actor's own appearances, leaving unrelated coordinates at zero.
  spans = np.zeros(len(names), dtype=float)
  for actor, rows in by_actor.items():
    idx = np.asarray(robot_idx[actor], dtype=int)
    spans[idx] = np.ptp(np.vstack([q[idx] for _i, q in rows]), axis=0)
  wrapped = np.minimum(D, np.abs(2.0 * np.pi - D)) if D.size else D
  crossing_mask = D > float(threshold)
  hidden_mask = crossing_mask & (wrapped <= float(threshold))
  l2 = np.linalg.norm(D, axis=1) if D.size else np.zeros(0)
  total_variation = np.sum(D, axis=0) if D.size else np.zeros(len(names))
  max_delta = np.max(D, axis=0) if D.size else np.zeros(len(names))

  top = np.argsort(-total_variation)[:min(12, len(names))]
  largest = []
  for k in np.argsort(-l2)[:min(12, len(l2))]:
    j = int(np.argmax(D[k]))
    largest.append({
        "from_action": edges[k][0], "to_action": edges[k][1],
        "l2_rad": float(l2[k]), "largest_joint": names[j],
        "largest_joint_delta_rad": float(D[k, j]),
    })

  cfg = doc.get("config") or {}
  return {
      "path": str(path), "stage": block.get("stage"),
      "metric_scope": "consecutive named-actor appearances; coupled-holder lower bound",
      "actions": len(doc.get("plan") or ()), "witness_states": len(indexed),
      "witness_edges": len(edges), "joints": len(names),
      "branch_metric": cfg.get("witness_branch_metric"),
      "branch_budget": cfg.get("witness_branch_budget"),
      "threshold_rad": float(threshold),
      "chain_l2_sum_rad": float(np.sum(l2)),
      "chain_l2_max_rad": float(np.max(l2)) if l2.size else 0.0,
      "large_coordinate_transitions": int(np.count_nonzero(crossing_mask)),
      "edges_with_large_transition": int(np.count_nonzero(np.any(crossing_mask, axis=1)))
          if D.size else 0,
      "modulo_hidden_transitions": int(np.count_nonzero(hidden_mask)),
      "joint_total_variation_sum_rad": float(np.sum(total_variation)),
      "top_joints": [{
          "joint": names[i], "total_variation_rad": float(total_variation[i]),
          "max_step_rad": float(max_delta[i]), "span_rad": float(spans[i]),
      } for i in top],
      "largest_edges": largest,
  }


def markdown(rows):
  lines = [
      "| artifact | stage | metric / budget | witness edges | chain L2 sum | max edge | "
      "large coordinates | modulo-hidden |",
      "|---|---|---:|---:|---:|---:|---:|---:|",
  ]
  for r in rows:
    lines.append(
        f"| `{Path(r['path']).name}` | {r['stage']} | "
        f"{r['branch_metric']} / {r['branch_budget']} | {r['witness_edges']} | "
        f"{r['chain_l2_sum_rad']:.3f} | {r['chain_l2_max_rad']:.3f} | "
        f"{r['large_coordinate_transitions']} | {r['modulo_hidden_transitions']} |")
  return "\n".join(lines) + "\n"


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("plans", type=Path, nargs="+")
  p.add_argument("--threshold", type=float, default=float(np.pi / 2.0), metavar="RAD")
  p.add_argument("--json", type=Path, default=None, dest="json_path")
  args = p.parse_args()
  rows = [analyze(path, args.threshold) for path in args.plans]
  print(markdown(rows), end="")
  if args.json_path is not None:
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.json_path.with_suffix(args.json_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
      json.dump({"schema_version": 1, "rows": rows}, f, indent=2, sort_keys=True)
      f.write("\n")
    tmp.replace(args.json_path)


if __name__ == "__main__":
  main()
