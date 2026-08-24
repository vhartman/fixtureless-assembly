"""Analytic IK seeding via the `ssik` closed-form solver, calibrated to our rai UR models.

The search's single-arm skills (pick / place / add_grasp) are plain 6-DOF reaches to a KNOWN EE
pose -- closed-form UR IK returns every joint solution directly, which we use to WARM-START the
per-skill KOMO. It helps CONVERGENCE-limited reaches; it does not help collision-limited ones (an
exact analytic pose may collide, whereas the KOMO's tolerant pose eq can settle on a nearby
collision-free pose the discrete IK solutions don't cover), so the KOMO fallback stays.

Per-arm AND per-tool calibration (rai model <-> ssik solver), each validated against rai FK
over random targets:

    q_ssik = SIGN * q_rai + OFFSET
    T_ee_in_baselink = ssik_fk(q_ssik) @ B_TOOL               (B_TOOL = tool0 -> EE centre)

so IK is:  q_rai = SIGN * (ssik.solve(T_target @ B_TOOL^-1).q - OFFSET).

  ur5  -> ssik.prebuilt.ur5e_ik   (rai ur5.g is a standard UR5e; exact)
  ur10 -> ik_artifacts/ur10_ik.py (built with `ssik build src/rai/ur10/ur10.urdf --base base_link
          --ee tool0` -- ssik ships only UR10e, and our arm is a plain UR10, so we bake an exact
          artifact from the URDF instead of an approximate ur10e fit)

`ssik` is an optional dependency -- import failures degrade gracefully to "no seeds".
"""
import importlib
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


# --- per-arm calibration: SIGN/OFFSET, tool0 -> EE-centre transforms, and solver loader ---
#
# The vacuum matrices are deliberately NOT inferred from the two-finger frame at call time.
# They are the fixed transforms measured from the actual RAI models and FK-validated over
# random configurations.  In particular the UR10 vacuum approaches along +x whereas its
# Robotiq centre has a 90-degree tool rotation; reusing the latter produced seeds 106mm and
# 90deg away from the requested vacuum pose, then incorrectly activated the seeded restart cap.
_CAL = {
    "ur5": {
        "sign": np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0]),
        "offset": np.array([-np.pi, -np.pi / 2.0, 0.0, -np.pi / 2.0, -np.pi, 0.0]),
        "tools": {
            "two_finger": np.array([[0.0, -1.0, 0.0, 0.0],
                                     [-1.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, -1.0, -0.3659],
                                     [0.0, 0.0, 0.0, 1.0]]),
            "vacuum": np.array([[0.0, -1.0, 0.0, 0.0],
                                 [-1.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, -1.0, -0.2959],
                                 [0.0, 0.0, 0.0, 1.0]]),
        },
        "load": lambda: _prebuilt("ur5e_ik"),
    },
    "ur10": {
        "sign": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "offset": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.195343]),
        "tools": {
            "two_finger": np.array([[0.930342, -0.366694, 0.0, 0.0],
                                     [0.366694, 0.930342, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.166],
                                     [0.0, 0.0, 0.0, 1.0]]),
            "vacuum": np.array([[0.0, -0.366694, -0.930342, 0.0],
                                 [0.0, 0.930342, -0.366694, 0.0],
                                 [1.0, 0.0, 0.0, 0.060],
                                 [0.0, 0.0, 0.0, 1.0]]),
        },
        "load": lambda: _artifact("ur10_ik"),
    },
    "fr3": {
        # built via: ssik build franka_description/robots/fr3/fr3.urdf.xacro
        #   (--xacro-arg hand:=false --base fr3_link0 --ee fr3_link8)
        # Validated: ssik fk == rai fr3v2 FK to 0.0 over random configs (identity joint
        # mapping). NOTE: FR3 is 7-DOF/redundant -- solve(T) returns discrete samples of
        # the solution MANIFOLD (nearest ~mrad from any true config), ideal as KOMO
        # seeds; empty set is strong unreachability evidence but see
        # --empty-seed-refutes caveats.
        "sign": np.ones(7),
        "offset": np.zeros(7),
        "tools": {
            # ssik EE frame == flange; tool0 -> EE centre == flange -> gripper centre of
            # robotiq_tight.g mounted straight at the flange (measured)
            "two_finger": np.array([[0.0, 0.0, 1.0, 0.13],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [-1.0, 0.0, 0.0, 0.036],
                                    [0.0, 0.0, 0.0, 1.0]]),
        },
        "load": lambda: _artifact("fr3_ik"),
    },
}
# NOTE on the ssik franka prebuilt (ssik.prebuilt.franka.fr3_ik): it models DIFFERENT
# geometry than our rai fr3v2 asset -- offset/sign/permutation voting over 15k solution
# pairs found no mapping (calibrate_fr3.py). Use the locally built artifact above instead.
# ur10_limited is a ur10 with a tighter shoulder-lift/elbow workspace. The KINEMATICS are
# identical, so the analytic solver is the same; the caller applies the joint limits to the
# returned branches (seed_configs returns all branches, limit filtering happens downstream).
_CAL["ur10_limited"] = _CAL["ur10"]
# ur5_limited likewise: same ur5e kinematics, tighter limits applied downstream.
_CAL["ur5_limited"] = _CAL["ur5"]
for _c in _CAL.values():
  _c["tool_inv"] = {k: np.linalg.inv(B) for k, B in _c["tools"].items()}
  _c["solver"] = None                                        # lazily loaded ssik module


def _solver(robot_type):
  c = _CAL[robot_type]
  if c["solver"] is None:
    c["solver"] = c["load"]()                                # may raise if ssik unavailable
  return c["solver"]


def available(robot_type, tool_kind="two_finger"):
  """Is this arm/tool pair calibrated (and is its optional ssik solver importable)?"""
  if robot_type not in _CAL or tool_kind not in _CAL[robot_type]["tools"]:
    return False
  try:
    _solver(robot_type)
    return True
  except Exception:
    return False


def seed_configs(robot_type, T_ee_in_baselink, tool_kind="two_finger"):
  """Analytic RAI joint solutions placing the selected EE centre at a target 4x4 SE(3)
  pose in ``base_link``.  Vacuum uses the canonical aligned EE roll supplied by the grasp
  target; its physical roll symmetry does not require a second nonlinear variable here.
  Returns [] if the pair is uncalibrated, ssik is unavailable, or the pose is unreachable."""
  if robot_type not in _CAL or tool_kind not in _CAL[robot_type]["tools"]:
    return []
  try:
    U = _solver(robot_type)
  except Exception:
    return []
  c = _CAL[robot_type]
  T_tool0 = np.asarray(T_ee_in_baselink, dtype=float) @ c["tool_inv"][tool_kind]
  try:
    sols = U.solve(T_tool0, respect_limits=False)
  except Exception:
    return []
  return [c["sign"] * (np.asarray(s.q, dtype=float) - c["offset"]) for s in sols]
