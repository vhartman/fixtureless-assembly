#!/usr/bin/env python3
"""Diagnose Fabrica grasp interfaces without running task or motion planning.

The report deliberately separates three questions which are easy to conflate in a failed
search:

* does the bounded grasp database contain a collision-clear floating-hand grasp?
* does a fixed-base arm have an exact within-limit IK branch for that grasp?
* which collision class removes those exact branches (self, table, assembly, other arm)?

It also measures the sampled lateral tolerance of an authored insertion against the exact
RAI collision decomposition.  The latter is a geometry audit, not a continuous certificate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

import assembly_planner.problem as P
from assembly_planner.plan_io import read_plan
from replay import build_scene
from assembly_planner.solver import (RobotIK, _search_context, isolated_pick_state)


def _root(name: str) -> str:
  return str(name).split("__", 1)[0]


def _scene(config_path: str, keep: int, candidates: int | None = None):
  cfg, _plan = read_plan(config_path)
  cfg = dict(cfg)
  cfg["floating_ee"] = False
  cfg["grasp_sampler_keep"] = int(keep)
  cfg["grasp_widen_max"] = int(keep)
  if candidates is not None:
    cfg["grasp_sampler_candidates"] = int(candidates)
  cfg["grasp_local_widen"] = False
  # The diagnostic intentionally changes floating-vs-arm geometry and possibly K.  It must
  # rebuild rather than pretend the old positional database identity still applies.
  cfg["grasp_sampler_db_hash"] = None
  C, assembly = build_scene(cfg)
  return cfg, C, assembly


def _corner_report(C, assembly, part: str):
  robots = P.get_robots(C)
  checker, table = _search_context(C, assembly, robots)
  s0 = P.get_initial_state(C, assembly)
  iso = isolated_pick_state(s0, assembly, part)
  rows = [(gid, pose) for gp, gid, pose in table.get((part,), ()) if gp == part]
  meta = P.sampled_grasp_meta(assembly.grasp_shape(part), part=part)
  half = P._shape_half_extents(assembly.grasp_shape(part))
  representative = robots[0]
  ranked = []
  for gid, _pose in rows:
    pose = np.asarray(P.grasp_pose(assembly, part, gid), dtype=float)
    roles = tuple((meta[gid] or {}).get("scene_by_kind", {}).get("two_finger", ()))
    open_clear = checker.pick_collision_free(
        iso, part, gid, assembly, robot=representative)
    # A plate-corner grasp has a tool/contact centre displaced in both horizontal axes.
    corner = min(abs(float(pose[0])) / max(float(half[0]), 1e-9),
                 abs(float(pose[1])) / max(float(half[1]), 1e-9))
    ranked.append((corner, gid, open_clear, roles, pose[:3],
                   (meta[gid] or {}).get("family", "unknown")))
  ranked.sort(reverse=True)
  print(f"[corner-grasps] {part}: {len(rows)} singleton rows; "
        f"{sum(r[2] for r in ranked)} isolated-pick clear")
  shown = ranked[:12]
  shown_ids = {r[1] for r in shown}
  shown += [r for r in ranked if (r[2] or r[3]) and r[1] not in shown_ids]
  for corner, gid, clear, roles, xyz, family in shown:
    print(f"[corner-grasps]   g{gid:02d} score={corner:.3f} "
          f"tool_xyz_mm={np.round(1000.0 * xyz, 2).tolist()} "
          f"family={family} roles={roles} pick_clear={clear}")
  return checker, table


def _bridge_report(assembly, parts):
  """Report grasp IDs that clear both authored endpoint scenes for each tool kind."""
  for part in parts:
    meta = P.sampled_grasp_meta(assembly.grasp_shape(part), part=part)
    kinds = sorted({kind for row in meta if row
                    for kind in row.get("scene_by_kind", {})})
    for kind in kinds:
      pick = {i for i, row in enumerate(meta) if row and
              "pick" in row.get("scene_by_kind", {}).get(kind, ())}
      assemble = {i for i, row in enumerate(meta) if row and
                  "assemble" in row.get("scene_by_kind", {}).get(kind, ())}
      both = sorted(pick & assemble)
      print(f"[grasp-bridge] {part}/{kind}: pick={len(pick)} "
            f"assemble={len(assemble)} both={len(both)} ids={both}")


def _active_collision_classes(C, robot: str, target: str, hand_frames: set[str], tol: float):
  pre = robot + "_"
  bad = {"self": [], "table": [], "assembly": [], "own_grasp": [], "other_robot": []}
  for a, b, distance in C.getCollisions():
    d = float(distance)
    if d >= -tol:
      continue
    a, b = str(a), str(b)
    ar, br = a.startswith(pre), b.startswith(pre)
    if not (ar or br):
      continue
    other = b if ar else a
    active = a if ar else b
    if ar and br:
      kind = "self"
    elif other in ("table", "floor"):
      kind = "table"
    elif _root(other).startswith("obj_"):
      if _root(other) == target and active in hand_frames:
        kind = "own_grasp"
      else:
        kind = "assembly"
    elif other.startswith("a") and "_" in other:
      kind = "other_robot"
    else:
      continue
    bad[kind].append((a, b, d))
  return bad


def _arm_stage_report(C, assembly, table, parts, tol=1e-6):
  robots = P.get_robots(C)
  rik = RobotIK(C, robots)
  rik.ik_seeding = True
  s0 = P.get_initial_state(C, assembly)
  full = P._full_assembly_state(assembly, s0)
  for part in parts:
    meta = P.sampled_grasp_meta(assembly.grasp_shape(part), part=part)
    singleton = {gid for gp, gid, _pose in table.get((part,), ()) if gp == part}
    gids = [gid for gid in sorted(singleton)
            if "assemble" in ((meta[gid] or {}).get("scene_by_kind", {})
                              .get("two_finger", ()))]
    totals = {"queries": 0, "reachable": 0, "self": 0, "table": 0,
              "assembly": 0, "other_robot": 0, "clean": 0}
    clean_rows = []
    blockers = {k: {} for k in ("self", "table", "assembly", "other_robot")}
    for gid in gids:
      for robot in robots:
        totals["queries"] += 1
        Ci = rik.scene_for(full, assembly)
        target = rik.ee_target(robot, part, gid, assembly)
        seeds = rik._analytic_seeds(Ci, robot, target, part)
        if not seeds:
          continue
        totals["reachable"] += 1
        rik._restrict_to_arm(Ci, robot)
        stage_seen = {k: False for k in ("self", "table", "assembly", "other_robot", "clean")}
        for seed in seeds:
          Ci.setJointState(np.asarray(seed, dtype=float))
          Ci.computeCollisions()
          classes = _active_collision_classes(Ci, robot, part, rik.hand_frames, tol)
          cumulative = True
          for kind in ("self", "table", "assembly", "other_robot"):
            cumulative = cumulative and not classes[kind]
            if cumulative:
              stage_seen[kind] = True
            for a, b, _d in classes[kind]:
              key = f"{a}~{b}"
              blockers[kind][key] = blockers[kind].get(key, 0) + 1
          if cumulative:
            stage_seen["clean"] = True
        for kind, passed in stage_seen.items():
          totals[kind] += int(passed)
        if stage_seen["clean"]:
          clean_rows.append((robot, gid, len(seeds)))
    print(f"[arm-stages] {part}: assembly-clear grasp ids={gids}")
    print("[arm-stages]   candidate-arm queries={queries}; exact-reachable={reachable}; "
          "then cumulative self-clear={self}, +table={table}, +assembly={assembly}, "
          "+other-robot/clean={clean}".format(**totals))
    print(f"[arm-stages]   clean rows (robot,gid,exact branches)={clean_rows[:20]}")
    for kind in ("self", "table", "assembly", "other_robot"):
      top = sorted(blockers[kind].items(), key=lambda x: (-x[1], x[0]))[:4]
      if top:
        print(f"[arm-stages]   top {kind} blockers={top}")


def _repose(C, part, xyz):
  C.getFrame(part).setPosition(np.asarray(xyz, dtype=float))
  C.setJointState(C.getJointState())
  C.computeCollisions()


def _insertion_penetration(C, part: str, fixed: set[str], nominal: np.ndarray,
                           axis: np.ndarray, lateral: np.ndarray, distance: float,
                           step: float) -> tuple[float, float]:
  worst = 0.0
  min_distance = float("inf")
  for offset in np.linspace(distance, 0.0, int(np.ceil(distance / step)) + 1):
    xyz = nominal - axis * offset + lateral
    _repose(C, part, xyz)
    ds = [float(d) for a, b, d in C.getCollisions()
          if ((_root(a) == part and _root(b) in fixed)
              or (_root(b) == part and _root(a) in fixed))]
    if ds:
      min_distance = min(min_distance, min(ds))
      worst = max(worst, max(0.0, -min(ds)))
  return worst, min_distance


def _lateral_clearance_report(assembly, parts, distance=.05, step=.001, tol=1e-6,
                              search_max=.005):
  C = P.add_disassembled_obj(P.make_world(), assembly, center_scatter=True, float_height=.002)
  positions = {p: assembly.position_in_base(p) + np.array([0.0, 0.0, 2.0])
               for p in assembly.objects}
  for p, xyz in positions.items():
    C.getFrame(p).setPosition(xyz)
    C.getFrame(p).setQuaternion([1.0, 0.0, 0.0, 0.0])
  C.setJointState(C.getJointState())
  for part in parts:
    idx = assembly.objects.index(part)
    fixed = set(assembly.objects[:idx])
    axis = np.asarray(assembly.mate_axis(part), dtype=float)
    axis /= np.linalg.norm(axis)
    basis = [v for v in np.eye(3) if abs(float(v @ axis)) < 1e-8]
    nominal_pen, nominal_gap = _insertion_penetration(
        C, part, fixed, positions[part], axis, np.zeros(3), distance, step)
    print(f"[insertion-clearance] {part}: mate_axis={axis.tolist()} "
          f"nominal_worst_penetration={1000*nominal_pen:.4f}mm "
          f"minimum_reported_pair_distance={1000*nominal_gap:.4f}mm")
    for direction in [sign * v for v in basis for sign in (-1.0, 1.0)]:
      lo, hi = 0.0, search_max
      hi_pen, _ = _insertion_penetration(
          C, part, fixed, positions[part], axis, hi * direction, distance, step)
      if hi_pen <= tol:
        result = f">={1000*search_max:.2f}mm"
      else:
        for _ in range(18):
          mid = (lo + hi) / 2.0
          pen, _ = _insertion_penetration(
              C, part, fixed, positions[part], axis, mid * direction, distance, step)
          if pen <= tol:
            lo = mid
          else:
            hi = mid
        result = f"{1000*lo:.3f}mm"
      print(f"[insertion-clearance]   lateral {direction.tolist()}: {result} "
            f"before >{1000*tol:.3f}mm penetration")
    _repose(C, part, positions[part])


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("config", help="saved plan/config JSON defining the Fabrica scene")
  ap.add_argument("--keep", type=int, default=64)
  ap.add_argument("--candidates", type=int, default=None,
                  help="override the raw mesh-surface sample count")
  ap.add_argument("--corner-part", action="append", default=[])
  ap.add_argument("--bridge-part", action="append", default=[])
  ap.add_argument("--arm-part", action="append", default=[])
  ap.add_argument("--insertion-part", action="append", default=[])
  ap.add_argument("--insertion-distance", type=float, default=.05)
  ap.add_argument("--insertion-step", type=float, default=.001)
  args = ap.parse_args()

  _cfg, C, assembly = _scene(args.config, args.keep, args.candidates)
  for part in (args.corner_part + args.bridge_part + args.arm_part
               + args.insertion_part):
    if part not in assembly.objects:
      ap.error(f"unknown part {part!r}; choose from {assembly.objects}")
  checker = table = None
  _bridge_report(assembly, args.bridge_part)
  for part in args.corner_part:
    checker, table = _corner_report(C, assembly, part)
  if args.arm_part:
    if table is None:
      checker, table = _search_context(C, assembly, P.get_robots(C))
    _arm_stage_report(C, assembly, table, args.arm_part)
  if args.insertion_part:
    _lateral_clearance_report(
        assembly, args.insertion_part, distance=args.insertion_distance,
        step=args.insertion_step)


if __name__ == "__main__":
  main()
