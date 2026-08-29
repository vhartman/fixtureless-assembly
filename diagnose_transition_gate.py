"""Report the geometric snap hidden by each symbolic transition in a saved plan.

No IK is resolved.  The predecessor is posed at the persisted witness, then compared with
the exact successor scene at the same robot configuration.  A non-zero part-pose delta is
the error that the discrete rebuild hides but an attachment-preserving exporter accumulates.
"""
import argparse

import numpy as np

from plan_io import read_plan, read_witnesses
from problem import get_robots
from replay import build_scene
from solver import (RobotIK, _quat_mul, _replay_states, _search_context, grasp_pose,
                    wrench_kind_arg)


def _quat_angle(a, b):
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  a /= max(np.linalg.norm(a), 1e-12)
  b /= max(np.linalg.norm(b), 1e-12)
  return 2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("plan")
  parser.add_argument("step", type=int, nargs="?", help="only report this action index")
  args = parser.parse_args()

  cfg, plan = read_plan(args.plan)
  C, assembly = build_scene(cfg)
  robots = get_robots(C)
  rik = RobotIK(C, robots)
  rik.penetration_reject = float(cfg.get("strict_collisions") or 1e-6)
  rik.keep_contacts = bool(cfg.get("always_contact", False))
  rik.strip_inactive_collision = not bool(cfg.get("witness_robot_collisions", False))
  _checker, grasp_table = _search_context(C, assembly, robots)
  gk = wrench_kind_arg(C, robots) if cfg.get("wrench") else None
  states = _replay_states(
      C, assembly, plan, robots, gk,
      bool(cfg.get("handovers", True)), bool(cfg.get("allow_place", True)),
      bool(cfg.get("bracing", False)), grasp_table, "transition-gate diagnostic")
  witnesses = read_witnesses(args.plan, list(rik.C.getJointNames()), len(plan))
  if witnesses is None:
    raise SystemExit("plan has no persisted witnesses")
  indices = [args.step] if args.step is not None else range(len(plan))
  for i in indices:
    action = tuple(plan[i])
    q = witnesses[i]
    if q is None:
      print(f"{i:02d} {action}: no witness")
      continue
    before = rik.scene_for(states[i], assembly)
    after = rik.scene_for(states[i + 1], assembly)
    q = np.asarray(q, dtype=float).ravel()
    before.setJointState(q)
    after.setJointState(q)
    deltas = []
    for part in assembly.objects:
      p0 = np.asarray(before.getFrame(part).getPose(), dtype=float)
      p1 = np.asarray(after.getFrame(part).getPose(), dtype=float)
      deltas.append((float(np.linalg.norm(p1[:3] - p0[:3])),
                     float(_quat_angle(p0[3:], p1[3:])), part))
    dp, da, who = max(deltas)
    detail = ""
    if action[0] in ("pick", "add_grasp", "handover"):
      robot, part, gid = action[1], action[2], int(action[3])
      grasp = grasp_pose(assembly, part, gid)
      ee_target = np.concatenate([grasp[:3],
                                  _quat_mul(grasp[3:], rik.q_correction[robot])])
      # RobotIK exposes the authoritative translation residual helper; use the same target
      # construction as its bind path without duplicating quaternion details here.
      detail = f" grasp={1000 * rik.grasp_pose_residual(before, q, robot, part, ee_target):.3f}mm"
    elif action[0] == "assemble":
      part, onto = action[2], action[3]
      rel = np.asarray(assembly.relative_pose(part)[1], dtype=float)
      detail = f" mate={1000 * rik.mate_residual(before, q, part, onto, rel):.3f}mm"
    print(f"{i:02d} {action}: symbolic snap max={1000 * dp:.3f}mm/{da:.4f}rad "
          f"({who}){detail}")


if __name__ == "__main__":
  main()
