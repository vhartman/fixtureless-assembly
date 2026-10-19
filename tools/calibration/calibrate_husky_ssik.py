"""Emit exact per-arm MJCF inputs for the calibrated Husky SSIK artifacts.

The two calibrated UR5e chains differ slightly, so they intentionally produce separate
artifacts instead of aliasing or modifying the repository's generic ``ur5`` calibration.

Usage:
  python3 tools/calibration/calibrate_husky_ssik.py /tmp/husky-ssik
  ssik build /tmp/husky-ssik/husky_right.mjcf \
      --base a1_ur_base_link --ee a1_ur_tool0 --out ik_artifacts/husky_right_ik.py
  ssik build /tmp/husky-ssik/husky_left.mjcf \
      --base a2_ur_base_link --ee a2_ur_tool0 --out ik_artifacts/husky_left_ik.py
"""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import assembly_planner.arm_model as arm_model
from assembly_planner.env import make_husky_duo_env


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("out_dir", type=Path)
  args = parser.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)

  C = make_husky_duo_env()
  for robot, side in (("a1", "right"), ("a2", "left")):
    xml, joints = arm_model.standalone_model(C, [robot], collide=False)
    expected = [n for n in C.getJointNames() if n.startswith(robot + "_")]
    if joints != expected or len(joints) != 6:
      raise RuntimeError(f"{side}: unexpected exported chain {joints}; expected {expected}")
    path = args.out_dir / f"husky_{side}.mjcf"
    path.write_text(xml, encoding="utf-8")
    print(path)


if __name__ == "__main__":
  main()
