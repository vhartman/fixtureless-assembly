"""Compare forward kinematics between the rai scene and the generated MuJoCo model.

This is the acceptance test for `arm_model.build_arm_mjcf`, and the thing that makes swapping
in a new robot repeatable: point it at a different robot type and it tells you, in metres and
degrees, whether MuJoCo agrees with rai over the whole joint range -- rather than leaving that
as a spot-check you find out about later via a mysteriously missed seat.

    python3 tools/validation/fk_compare.py
    python3 tools/validation/fk_compare.py --robot ur5e
    python3 tools/validation/fk_compare.py --num-robots 3 -n 500 --verbose

Every arm frame is compared, not just the tool, because a mid-chain offset error can cancel at
one pose and reappear at another. Configs are drawn inside rai's own joint limits, plus the
zero config and the home config as fixed cases.
"""
import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


def quat_angle(qa, qb):
  """Angle (rad) between two [w,x,y,z] quaternions, sign-insensitive (q and -q are one
  rotation, and rai/MuJoCo need not agree on which representative they store)."""
  d = abs(float(np.dot(np.asarray(qa, float), np.asarray(qb, float))))
  return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def compare(C, prefixes, n=200, seed=0, verbose=False):
  """Sample joint configs; return (rows, worst) where rows is per-frame max error.
  rows: {frame: (max_pos_err_m, max_ang_err_rad)}."""
  import mujoco
  import assembly_planner.arm_model as arm_model

  xml, jnames = arm_model.standalone_model(C, prefixes)
  m = mujoco.MjModel.from_xml_string(xml)
  d = mujoco.MjData(m)

  rai_names = list(C.getJointNames())
  # MuJoCo qpos order is the model's own; map each MuJoCo joint to its index in rai's vector.
  # They coincide today, but relying on that silently would be exactly the class of bug this
  # script exists to catch.
  mj_to_rai = []
  for jn in jnames:
    if jn not in rai_names:
      raise KeyError(f"generated joint {jn!r} is not a rai joint; tree read is inconsistent")
    mj_to_rai.append(rai_names.index(jn))

  frames = [n_ for p in prefixes for n_ in C.getFrameNames() if n_.startswith(p)]
  lo, hi = C.getJointLimits()[:2] if hasattr(C, "getJointLimits") else (None, None)
  rng = np.random.default_rng(seed)
  q_home = np.asarray(C.getJointState(), float)
  nq = len(q_home)

  def sample(i):
    if i == 0:
      return np.zeros(nq)                                    # the config the model was read at
    if i == 1:
      return q_home.copy()                                   # the pose everything starts from
    if lo is not None and hi is not None and len(lo) == nq:
      return rng.uniform(np.asarray(lo, float), np.asarray(hi, float))
    return rng.uniform(-np.pi, np.pi, nq)

  rows = {f: [0.0, 0.0, None] for f in frames}                # frame -> [pos, ang, worst q]
  for i in range(max(n, 2)):
    q = sample(i)
    C.setJointState(q)
    d.qpos[:] = q[mj_to_rai]
    mujoco.mj_kinematics(m, d)
    for f in frames:
      rp = np.asarray(C.getFrame(f).getPosition(), float)
      rq = np.asarray(C.getFrame(f).getQuaternion(), float)
      bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f)
      mp, mq = d.xpos[bid], d.xquat[bid]
      dp = float(np.linalg.norm(rp - mp))
      da = quat_angle(rq, mq)
      if dp > rows[f][0] or da > rows[f][1]:
        rows[f][2] = q.copy()
      rows[f][0] = max(rows[f][0], dp)
      rows[f][1] = max(rows[f][1], da)
  C.setJointState(q_home)                                     # leave the caller's scene as found
  worst = max(((v[0], v[1], k) for k, v in rows.items()), default=(0.0, 0.0, None))
  return rows, worst


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--robot", default="ur10", help="rai robot type (ur10, ur5e, ...)")
  ap.add_argument("--num-robots", type=int, default=1)
  ap.add_argument("-n", type=int, default=200, help="random configs (default 200)")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--tol-pos", type=float, default=1e-5, help="pass threshold, metres")
  ap.add_argument("--tol-ang", type=float, default=1e-5, help="pass threshold, radians")
  ap.add_argument("--verbose", action="store_true", help="per-frame table")
  a = ap.parse_args()

  from assembly_planner.env import make_four_robot_env
  C = make_four_robot_env(num_robots=a.num_robots, floating_ee=False, vacuum=False,
                          robot_types=a.robot)
  import assembly_planner.assembly as A
  prefixes = A.get_robots(C)
  rows, (wp, wa, wf) = compare(C, prefixes, n=a.n, seed=a.seed)

  if a.verbose:
    for f, (dp, da, _) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
      print(f"  {f:34s} {dp * 1000:9.5f} mm   {np.degrees(da):9.5f} deg")
  print(f"[fk] {a.robot}, {len(prefixes)} arm(s), {len(rows)} frames, {max(a.n, 2)} configs")
  print(f"[fk] worst frame: {wf}  pos {wp * 1000:.6f} mm   ang {np.degrees(wa):.6f} deg")
  ok = wp <= a.tol_pos and wa <= a.tol_ang
  print(f"[fk] {'PASS' if ok else 'FAIL'} (tolerance {a.tol_pos * 1000:g} mm / "
        f"{np.degrees(a.tol_ang):g} deg)")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
