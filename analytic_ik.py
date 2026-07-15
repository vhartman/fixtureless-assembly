"""Analytic IK seeding via the `ssik` closed-form solver, calibrated to our rai UR models.

The search's single-arm skills (pick / place / add_grasp) are plain 6-DOF reaches to a KNOWN EE
pose -- closed-form UR IK returns every joint solution directly, which we use to WARM-START the
per-skill KOMO. It helps CONVERGENCE-limited reaches; it does not help collision-limited ones (an
exact analytic pose may collide, whereas the KOMO's tolerant pose eq can settle on a nearby
collision-free pose the discrete IK solutions don't cover), so the KOMO fallback stays.

Per-arm calibration (rai model <-> ssik solver), each fit to RMS ~1e-8 and validated to 0.0
rai-FK error over random targets:

    q_ssik = SIGN * q_rai + OFFSET
    T_gripper_in_baselink = ssik_fk(q_ssik) @ B_TOOL          (B_TOOL = tool0 -> gripper_center)

so IK is:  q_rai = SIGN * (ssik.solve(T_target @ B_TOOL^-1).q - OFFSET).

  ur5  -> ssik.prebuilt.ur5e_ik   (rai ur5.g is a standard UR5e; exact)
  ur10 -> ik_artifacts/ur10_ik.py (built with `ssik build src/rai/ur10/ur10.urdf --base base_link
          --ee tool0` -- ssik ships only UR10e, and our arm is a plain UR10, so we bake an exact
          artifact from the URDF instead of an approximate ur10e fit)

`ssik` is an optional dependency -- import failures degrade gracefully to "no seeds".
"""
import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(__file__)


def _prebuilt(name):
  import importlib
  return importlib.import_module("ssik.prebuilt." + name)


def _artifact(name):
  # load a self-contained `ssik build` artifact from ik_artifacts/ by file path (no sys.path change)
  path = os.path.join(_HERE, "ik_artifacts", name + ".py")
  spec = importlib.util.spec_from_file_location("_ssik_art_" + name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


# --- per-arm calibration: SIGN, OFFSET, B_TOOL (tool0 -> gripper_center), and a solver loader ---
_CAL = {
    "ur5": {
        "sign": np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0]),
        "offset": np.array([-np.pi, -np.pi / 2.0, 0.0, -np.pi / 2.0, -np.pi, 0.0]),
        "B": np.array([[0.0, -1.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0, -0.3659],
                       [0.0, 0.0, 0.0, 1.0]]),
        "load": lambda: _prebuilt("ur5e_ik"),
    },
    "ur10": {
        "sign": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "offset": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.195343]),
        "B": np.array([[0.930342, -0.366694, 0.0, 0.0],
                       [0.366694, 0.930342, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.166],
                       [0.0, 0.0, 0.0, 1.0]]),
        "load": lambda: _artifact("ur10_ik"),
    },
}
for _c in _CAL.values():
  _c["Binv"] = np.linalg.inv(_c["B"])
  _c["solver"] = None                                        # lazily loaded ssik module


def _solver(robot_type):
  c = _CAL[robot_type]
  if c["solver"] is None:
    c["solver"] = c["load"]()                                # may raise if ssik unavailable
  return c["solver"]


def available(robot_type):
  """Is there a calibrated analytic solver for this arm (and is ssik importable)?"""
  if robot_type not in _CAL:
    return False
  try:
    _solver(robot_type)
    return True
  except Exception:
    return False


def seed_configs(robot_type, T_gripper_in_baselink):
  """Analytic joint solutions (in RAI convention) placing the tool center at the target, a 4x4
  SE3 pose expressed in the robot's `base_link` frame. Returns [] if the arm is uncalibrated,
  ssik is unavailable, or the pose is unreachable."""
  if robot_type not in _CAL:
    return []
  try:
    U = _solver(robot_type)
  except Exception:
    return []
  c = _CAL[robot_type]
  T_tool0 = np.asarray(T_gripper_in_baselink, dtype=float) @ c["Binv"]
  try:
    sols = U.solve(T_tool0, respect_limits=False)
  except Exception:
    return []
  return [c["sign"] * (np.asarray(s.q, dtype=float) - c["offset"]) for s in sols]
