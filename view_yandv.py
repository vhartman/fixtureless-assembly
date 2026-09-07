#!/usr/bin/env python3
"""View the Y&V stool/bench initial or assembled scene in rai or MuJoCo.

Examples:
  python view_yandv.py --backend rai --assembled
  python view_yandv.py --backend mujoco

MuJoCo starts with exact CAD visuals (group 3) and environment (group 1) visible. Press ``0``
in the viewer to toggle the simplified collision boxes. Press ``3`` to toggle CAD visuals.
"""
import argparse
import time

import numpy as np

import problem as P
from env import make_world


def build(kind, assembled):
  n = 5 if kind in ("real-stool", "yandv-stool") else 3
  assembly = P.make_assembly(kind, n)
  C = P.add_disassembled_obj(make_world(), assembly, spacing=0.3, float_height=0.002,
                             center_scatter=True)
  if not assembled:
    return assembly, C, P.get_initial_state(C, assembly)

  # Put the complete object's feet 2mm above the planner table. This is only an inspection
  # scene; no planning or mate dynamics occurs here.
  table = C.getFrame("table")
  table_top = float(table.getPosition()[2] + table.getSize()[2] / 2.0)
  # The leg stops 0.5mm below the blind-hole floor. Put its bottom 2mm above the table.
  leg_bottom = (-assembly.seat_thickness / 2.0 + assembly.socket_depth - 0.0005
                - assembly.leg_length)
  base_z = table_top - leg_bottom + 0.002
  poses = {}
  parent = {assembly.base: ("table",)}
  supports = {assembly.base: frozenset({("table",)})}
  for p in assembly.objects:
    pose = np.r_[assembly.position_in_base(p) + [0.0, 0.0, base_z],
                 [1.0, 0.0, 0.0, 0.0]]
    poses[p] = pose
    C.getFrame(p).setPosition(pose[:3]).setQuaternion(pose[3:])
    if p != assembly.base:
      parent[p] = ("part", assembly.base)
      supports[p] = frozenset({("mate", assembly.base)})
  return assembly, C, P.State(parent=parent, supports=supports, poses=poses)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--backend", choices=("rai", "mujoco"), default="mujoco")
  ap.add_argument("--assembly", choices=("real-stool", "real-bench",
                                          "yandv-stool", "yandv-bench"),
                  default="real-stool")
  ap.add_argument("--assembled", action="store_true",
                  help="Show assembled offsets instead of the lying-leg initial layout.")
  args = ap.parse_args()
  assembly, C, state = build(args.assembly, args.assembled)
  title = f"{args.assembly}: {'assembled' if args.assembled else 'initial layout'}"

  if args.backend == "rai":
    C.view(True, title)
    return

  import mujoco
  from mujoco import viewer
  import sim

  model = mujoco.MjModel.from_xml_string(sim.build_mjcf(assembly, state, []))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  with viewer.launch_passive(model, data) as v:
    v.opt.geomgroup[sim.GROUP_PART] = 0                    # exact visuals only initially
    v.opt.geomgroup[sim.GROUP_ENV] = 1
    v.opt.geomgroup[sim.GROUP_VISUAL] = 1
    print(f"[view-yandv] {title}; key 0=collision cuboids, key 3=exact CAD visuals")
    while v.is_running():
      v.sync()
      time.sleep(0.02)


if __name__ == "__main__":
  main()
