"""One-off audit helper: rebuild the planner scene of a saved plan and print every part's
vertical extent vs the planner table top, to confirm nothing starts inside the table (the
sim raises ITS table to the lowest part bottom, so a part whose half-z estimate is too small
starts embedded). Exercises the cylinder support-function fix in problem._shape_half_z."""
import json

import numpy as np

import replay
from problem import _quat_to_R, _shape_half_z

with open("car-mini-3r-plan.json") as f:
    d = json.load(f)
cfg = d["config"]

C, asm = replay.build_scene(cfg)
tf = C.getFrame("table")
table_top = float(np.asarray(tf.getPosition(), dtype=float)[2]) + float(tf.getSize()[2]) / 2.0
print(f"planner table top z = {table_top:.4f}")

gaps = []
for p in asm.objects:
    fr = C.getFrame(p)
    pos = np.asarray(fr.getPosition(), dtype=float)
    quat = np.asarray(fr.getQuaternion(), dtype=float)
    R = _quat_to_R(quat)
    hz = _shape_half_z(asm.part_shape(p), R)
    gap = pos[2] - hz - table_top
    gaps.append(gap)
    print(f"{p}: half_z={hz * 1000:6.1f}mm  z={pos[2]:.4f}  gap_to_table={gap * 1000:+.2f}mm")

want = float(cfg.get("part_float", 0.002))
print(f"min gap {min(gaps) * 1000:+.2f}mm, max gap {max(gaps) * 1000:+.2f}mm "
      f"(expect all ~= part_float {want * 1000:.1f}mm, none negative)")
