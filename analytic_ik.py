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
                                 [0.0, 0.0, -1.0, -0.3109],
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
                                 [1.0, 0.0, 0.0, 0.115],
                                 [0.0, 0.0, 0.0, 1.0]]),
        },
        "load": lambda: _artifact("ur10_ik"),
    },
    "husky_right": {
        # Generated directly from the live calibrated RAI subtree by
        # calibrate_husky_ssik.py.  The calibrated right and left UR5e chains differ by
        # millimetres/milliradians and therefore intentionally use separate artifacts.
        "sign": np.ones(6),
        "offset": np.zeros(6),
        "tools": {
            # calibrated tool0 is the Robotiq mounting flange; the physical two-finger
            # tool is rolled +90deg about that flange axis and gripper_center is 130mm
            # along the resulting +z axis.  This must match husky_calibrated.g exactly.
            "two_finger": np.array([[0.0, -1.0, 0.0, 0.0],
                                     [1.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.13],
                                     [0.0, 0.0, 0.0, 1.0]]),
        },
        "load": lambda: _artifact("husky_right_ik"),
    },
    "husky_left": {
        "sign": np.ones(6),
        "offset": np.zeros(6),
        "tools": {
            # Same physical +90deg tool roll as the right arm; the arm itself is
            # calibrated independently.
            "two_finger": np.array([[0.0, -1.0, 0.0, 0.0],
                                     [1.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.13],
                                     [0.0, 0.0, 0.0, 1.0]]),
        },
        "load": lambda: _artifact("husky_left_ik"),
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
    "openarm_left": {
        # built via: ssik build openarm_v2.0/urdf/example/v2.urdf
        #   (--base openarm_left_base_link --ee openarm_left_ee_base_link)
        # Validated: ssik fk == rai left_openarm.g FK to 0.0 over random configs with the
        # per-joint SIGN map below (rai joint_scale/-axis conventions flip joints 2,3,4;
        # j5/j6 happen to re-flip). Tool: ee_base_link -> gripper_center measured constant.

        "sign": np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0]),   # rescaled .g (no joint_scale) -- re-derived empirically
        "offset": np.zeros(7),
        "tools": {
            # tool0 -> EE centre == pose of gripper_center in the flange frame.
            # left_openarm.g: Q "t(-.0014 0 -.14) d(180 1 0 0)" -- translation then
            # 180deg about x, so the tool centre's +z is the approach axis (like ur5).
            "two_finger": np.array([[1.0, 0.0, 0.0, -0.0014],
                                     [0.0, -1.0, 0.0, 0.0],
                                     [0.0, 0.0, -1.0, -0.14],
                                     [0.0, 0.0, 0.0, 1.0]]),
        },
        "load": lambda: _artifact("openarm_left_ik"),
    },
    "openarm_right": {
        # same source URDF, right chain; sign map differs only in j1 (mirrored shoulder)
        "sign": np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]),   # rescaled .g -- re-derived empirically
        "offset": np.zeros(7),
        "tools": {
            # right_openarm.g: same Q "t(-.0014 0 -.14) d(180 1 0 0)" as the left arm.
            "two_finger": np.array([[1.0, 0.0, 0.0, -0.0014],
                                     [0.0, -1.0, 0.0, 0.0],
                                     [0.0, 0.0, -1.0, -0.14],
                                     [0.0, 0.0, 0.0, 1.0]]),
        },
        "load": lambda: _artifact("openarm_right_ik"),
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


def _rai_limits_in_solver_coordinates(calibration, joint_limits):
  """Map a ``(lo, hi)`` RAI hardware box through the calibrated joint signs.

  The generated artifacts bake the limits from their source URDF.  Our runtime models may
  deliberately use a different calibrated box (OpenArm's two wrist joints do), so asking the
  artifact to respect its baked limits is not the same physical query as RobotIK's.  Keep the
  geometry baked, but replace only the limit intervals used by the 7R swivel resolver.
  """
  limits = np.asarray(joint_limits, dtype=float)
  if limits.shape != (2, len(calibration["sign"])):
    return None
  lo, hi = limits
  a = calibration["sign"] * lo + calibration["offset"]
  b = calibration["sign"] * hi + calibration["offset"]
  return np.minimum(a, b), np.maximum(a, b)


def _limit_aware_redundant_solutions(solver, T_target, solver_limits):
  """Representatives of every feasible 7R swivel interval under runtime limits.

  ``ssik`` intentionally exposes this resolver as the generated SRS solver's in-limit
  fallback.  Calling it with a fresh KinBody avoids mutating the module-global baked model
  and therefore remains safe if independent searches ever share one Python process.
  """
  if solver_limits is None or getattr(solver, "DOF", None) != 7:
    return []
  try:
    kb = solver._build_kb()
    for joint, lo, hi in zip(kb.joints, solver_limits[0], solver_limits[1]):
      joint.limits = (float(lo), float(hi))
    return solver._resolve_in_limits(
        kb, np.asarray(T_target, dtype=float), policy=solver.DEFAULT_TOLERANCE_POLICY)
  except Exception:
    # Non-SRS/custom artifacts need not provide the generated resolver internals.  Their
    # ordinary finite solution set is still useful as warm-start guidance below.
    return []


def seed_configs(robot_type, T_ee_in_baselink, tool_kind="two_finger", joint_limits=None):
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
    raw = U.solve(T_tool0, respect_limits=False)
  except Exception:
    return []
  solver_limits = _rai_limits_in_solver_coordinates(c, joint_limits) \
      if joint_limits is not None else None
  sols = list(raw)
  # A 7R target has a one-dimensional self-motion manifold.  The artifact's ordinary solve
  # samples that manifold at 16 swivel values; post-filtering those samples is not a complete
  # physical-limit query.  Add representatives found by its interval-based resolver instead.
  sols.extend(_limit_aware_redundant_solutions(U, T_tool0, solver_limits))

  out = []
  rai_limits = np.asarray(joint_limits, dtype=float) if joint_limits is not None else None
  for s in sols:
    q = c["sign"] * (np.asarray(s.q, dtype=float) - c["offset"])
    if rai_limits is not None and rai_limits.shape == (2, len(q)):
      # Analytic solvers may return a revolute angle in an equivalent 2pi winding outside
      # RAI's interval (the generated Husky solvers commonly return q-2pi).  That is not a
      # physical-limit rejection. Canonicalize each revolute solution into the runtime box;
      # reject only when no equivalent winding lies inside. All currently calibrated SSIK
      # arms are revolute-only, so this does not reinterpret any prismatic coordinate.
      q_fit = np.asarray(q, dtype=float).copy()
      valid = True
      for i, (qi, qlo, qhi) in enumerate(zip(q_fit, rai_limits[0], rai_limits[1])):
        if qi < qlo - 1e-9 or qi > qhi + 1e-9:
          k0 = int(np.ceil((qlo - qi - 1e-9) / (2.0 * np.pi)))
          k1 = int(np.floor((qhi - qi + 1e-9) / (2.0 * np.pi)))
          if k0 > k1:
            valid = False
            break
          centre = 0.5 * (qlo + qhi)
          k = min(range(k0, k1 + 1), key=lambda kk: abs(qi + 2.0 * np.pi * kk - centre))
          q_fit[i] = qi + 2.0 * np.pi * k
      if not valid:
        continue
      q = q_fit
    if not any(np.linalg.norm(q - old) < 1e-5 for old in out):
      out.append(q)
  return out
