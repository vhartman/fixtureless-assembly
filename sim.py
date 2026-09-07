"""sim.py -- MuJoCo EXECUTION of solved plans (validation harness; rai stays the planner).

PURPOSE (paper: the wrench-LP validation leg). We do NOT plan here. A plan solved on the rai
backend is EXECUTED open-loop in MuJoCo, and we measure whether it physically succeeds. The
claim under test: plans passing the static-equilibrium LP succeed; plans planned WITHOUT it
(wrench off) fail in exactly the ways the LP predicts (target slides during a side press,
suction breaks under load, part rotates out of the pinch, unbraced assembly topples). Run the
benchmark grid wrench-on vs wrench-off and report success rates + a failure taxonomy.

BACKEND: MuJoCo (user's call, 2026-07-24). Wins for this study: wxyz quaternions (matches
problem.py), first-class WELD equality constraints toggleable at runtime (grasp/mate managers),
and constraint-force readout (data.efc_*) which lets grasp welds BREAK on overload -- both
FORCE and TORQUE caps, so vacuum peeling and the two-finger T_close (rotate-out-of-jaws) story
are representable (pybullet's maxForce was isotropic-force-only). The model is built as an
MJCF XML STRING (schema-stable across MuJoCo versions; mjSpec would also do).

DESIGN DECISIONS (modeling side):
  * ACTUATED REAL ARMS / DYNAMIC PARTS. The RAI arm model is exported to MuJoCo, joint position
    actuators track the persisted mr_bench trajectory, cross-arm contacts are physical, and
    parts are free bodies.
  * MATE == WELD-ON-SEAT (assumption, validated in REAL). Proper pin/hole contact at real
    clearances is hostile to any rigid-body engine (thin features, solver softness, timestep
    blow-up exactly in the phase of interest) -- in BOTH pybullet and mujoco. So: the seating
    approach/press is REAL contact dynamics (that is where the LP's claims live: does the
    receiving structure hold still, does the grasp carry the press reaction), and a part
    seated within tolerance activates a pre-declared weld equality to its manual parent. The
    hardware demo carries the "funnel-guided insertion actually seats" claim.
  * GRASPS = BREAKAWAY WELDS with the PLANNER'S OWN caps (wrench.PARAMS): each step we read
    the weld's constraint wrench and DEACTIVATE the grasp when it exceeds the cap (force or
    torque, per gripper kind). Honest scope note: this validates that the LP's STATIC load
    predictions match the sim's DYNAMIC loads (the paper claim), not the truth of the caps
    themselves -- cap truth is a hardware question. Real frictional finger plates remain the
    future upgrade if reviewers want contact-level grasp failure.
  * FRICTION from the planner's mu; ssBox rounding dropped (plain box; contact patch grows by
    ~r -- conservative for stability, note in the paper). Fixed timestep, default implicit
    solver, seeded => deterministic (the PYTHONHASHSEED discipline, for physics).

TRAJECTORY SOURCE: persisted mr_bench BiRRT joint trajectories. Mode-transition task frames
drive the grasp/release/mate managers; strict preflight and phase scheduling use the motion
planner's collision predicate before actuated execution.

Recordings: sim_recordings/*.npz -- replay with `python3 sim.py replay <file> [speed]`.
"""
import os
import time as _time
import numpy as np
from math import sqrt as _sqrt
import arm_model

from config import saved_cfg as _saved

import retiming                                              # portable path->trajectory retiming

try:
  import mujoco
  HAVE_MUJOCO = True
except ImportError:                                        # module must import without the sim
  mujoco = None
  HAVE_MUJOCO = False


# MuJoCo's contact filter is process-global. Dispatch by model so tests/sequential scenes do
# not leave a callback bound to the wrong SimScene. Values are weak: compiling a temporary
# scene must not keep its (potentially large) model alive.
_CONTACT_FILTER_SCENES = {}


def _assembly_contact_filter(model, data, geom1, geom2):
  """Exclude a direct mate pair only while its rigid mate weld is active."""
  ref = _CONTACT_FILTER_SCENES.get(id(model))
  scene = ref() if ref is not None else None
  if scene is None:
    _CONTACT_FILTER_SCENES.pop(id(model), None)
    return 0
  return scene._filter_active_mate_contact(data, int(geom1), int(geom2))

from problem import (grasp_pose, _pose_compose, _pose_invert, _quat_to_R, _quat_mul,
                     _shape_half_z, _state_part_poses)


# ---------------------------------------------------------------------------------------------
# tunables (every one is a MODELING decision -> report in the paper)
# ---------------------------------------------------------------------------------------------
DT = 0.002                      # s; MuJoCo default; parts are >=3cm primitives, stable
SEAT_POS_TOL = 0.010            # m   : seated when relative pose within these of the manual
                                # (= the pre-insertion funnel's CAPTURE tolerance; the datum
                                # weld then enforces the exact manual pose. Residual carry
                                # sag of the grasp weld is ~2-3mm after stiffening.)
DROP_Z = 0.05                   # m   : below expected by this much => "dropped"
SLIDE_TOL = 0.015               # m   : xy drift of a NON-actor part during an action => "slid"
TOPPLE_ANG = np.deg2rad(20.0)   # rad : orientation change of a resting group => "toppled"
CARRY_LIFT = 0.12               # m   : held parts are carried this high above their pick spot.
                                # Planner State.poses do NOT model the lift ("held" is
                                # symbolic); without it a mate onto a HELD target jams into
                                # the table (measured: 21.8N > 15N cap on cross-3 rollout 2).
RECORD_EVERY = 8                # frames: sample qpos+grasp forces every N sim steps (~62fps)
PATH_PREFLIGHT_TOL = 1e-4       # m: strict MP contract; above RAI's ~micrometre collision noise
BREAK_SUSTAIN = 0.02            # s : a grasp breaks only if overloaded CONTINUOUSLY this long.
                                # Instantaneous 2ms contact-impulse spikes (measured 15.6N on
                                # a carried group, invisible at recording resolution) are
                                # survivable by a real seal/pinch; loss of grip is a leak/slip
                                # process, not a single-sample event. Static overloads (what
                                # the LP prices) persist and still break.
WELD_TIMECONST = 0.004          # s   : solref timeconst of the grasp/mate welds (stiff: with
                                # MuJoCo's soft default the mocap-anchored weld SAGS mm-scale
                                # under carry loads -- measured 7-8mm, enough to miss a seat).


class PathPreflightError(RuntimeError):
  """A saved motion path violates an invariant retiming cannot repair."""


KINK_TAU = 2.0 * WELD_TIMECONST # s   : time the grasp interface takes to absorb a VELOCITY
                                # STEP (a C0 path corner), i.e. peak |F| = m|dv|/KINK_TAU.
                                # MEASURED on car-mini: a 1.24 m/s step on a 0.30kg group
                                # peaked at 39.9N => 9.3ms, ~2 solref timeconsts. On hardware
                                # the corner-rounding time of the tracking controller plays
                                # this role; here it is the weld, which IS the interface.
CAPTURE_RANGE = 0.05            # m   : a part-part mate datum-welds only within this of the
                                # seat (the funnel mouth); a larger miss = the part never
                                # arrived -> fail cleanly instead of snapping it in (explodes).
RAMP_DIST = 0.2                 # rad : trapezoid -- ramp to/from ~rest over this much joint
                                # arc at each per-mode stretch boundary (task goals), so every
                                # attach event happens quasi-statically. Interim stand-in for
                                # kinodynamic MP: execution-side timing only, geometry untouched.
RAMP_MIN = 0.04                 # frac: floor of the ramp (never exactly 0 -> guaranteed arrival)
DWELL_T = 0.15                  # s   : hold at each transition before/after firing attaches
SETTLE_T = 0.5                  # s   : settle after each action before judging
TABLE_HALF = (1.2, 1.2, 0.05)   # matches the rai table footprint; z-offset checked on 1st run


def _grasp_caps(kind):
  """SCALAR fallback: the tightest force/torque the planner's model allows in ANY direction.

  Only used when the grasp id is unknown, so the grasp frame cannot be reconstructed. It is
  CONSERVATIVE by construction (min over directions) and therefore safe, but it is also the
  reason the sim used to reject loads the planner considers nearly free -- with the asymmetric
  model this collapses F_push=100 and F_pull=25 to 25N in EVERY direction, so a compressive
  insertion reaction reads as a 3x overload. See _grasp_polytope for the real test."""
  import wrench as W
  model = W.PARAMS["grasp"].get(W.PARAMS.get("grasp_model", "isotropic"),
                                W.PARAMS["grasp"]["isotropic"])
  g = model.get(kind, {})
  fcap = min([v for k, v in g.items() if k.startswith("F_")] or [40.0])
  tcap = min([v for k, v in g.items() if k.startswith("T_")] or [1.0])
  return float(fcap), float(tcap)


def _grasp_polytope(kind):
  """(model_name, caps dict) -- the PLANNER's grasp capability, unreduced.

  wrench.py builds a BOX in the grasp's own frame: an axial bound along the approach axis n
  (asymmetric splits it into F_push when the part is pressed INTO the palm/cup and the much
  weaker F_pull when it is pulled off), a per-axis lateral bound mu*F_axial from friction at the
  preload, and separate couple bounds about n / the two lateral axes. Reproducing that box here
  is what makes the sim's verdict the same statement as the planner's."""
  import wrench as W
  name = W.PARAMS.get("grasp_model", "isotropic")
  fam = W.PARAMS["grasp"].get(name)
  if fam is None or kind not in fam:
    return None, None                                      # unknown -> caller uses the scalar
  return name, dict(fam[kind])


def _mate_caps_from_args(a):
  """CLI -> SimScene.mate_caps. None unless at least one cap is given (feature OFF = the
  historical infinite-strength rigid weld, byte-identical); a missing partner cap is inf."""
  if a.mate_force_cap is None and a.mate_torque_cap is None:
    return None
  return (float(a.mate_force_cap) if a.mate_force_cap is not None else float("inf"),
          float(a.mate_torque_cap) if a.mate_torque_cap is not None else float("inf"))


def _transverse_basis(direction):
  """Deterministic orthonormal basis of the plane normal to ``direction``."""
  d = np.asarray(direction, dtype=float)
  d /= np.linalg.norm(d) + 1e-12
  ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
  t1 = np.cross(ref, d)
  t1 /= np.linalg.norm(t1) + 1e-12
  return t1, np.cross(d, t1)


def _process_wrench_pair(assembly, part, parent, part_pose, parent_pose, realization=None):
  """One realized process load on both sides of a mate, in world coordinates.

  The search checks these as two separate external loads: the receiving component sees the
  insertion force along the mate axis, while the inserted component sees the opposite reaction.
  A screw additionally contributes an equal-and-opposite pure couple.  Return the same physical
  pair as ``{body: (force, pure_torque, application_point)}``. ``realization`` contains fixed
  coefficients in the mate's transverse frame.  Keeping those coefficients constant for the
  whole skill models a quasi-static alignment error; resampling every MuJoCo tick would instead
  create timestep-dependent vibration that the planner never claimed to cover.

  Both forces act at ONE world-space interface point and are exactly opposite.  The target's
  current pose defines the mate frame; this preserves Newton's third law even after a failed
  execution lets the two bodies drift apart.
  """
  import wrench as W
  pp = np.asarray(part_pose, dtype=float)
  pt = np.asarray(parent_pose, dtype=float)
  Rp, Rt = _quat_to_R(pp[3:]), _quat_to_R(pt[3:])
  axis = np.asarray(assembly.mate_axis(part), dtype=float)
  axis = axis / (np.linalg.norm(axis) + 1e-12)
  mag = float(W.insert_force(part))
  d = Rt @ axis
  t1, t2 = _transverse_basis(d)
  realization = realization or {}
  lat = float(realization.get("lateral", 0.0))
  off = float(realization.get("offset", 0.0))
  coeff = np.asarray(realization.get("coefficients", (0.0, 0.0, 0.0, 0.0)), dtype=float)
  if coeff.shape != (4,):
    raise ValueError("process-wrench realization needs four coefficients")
  ft = mag * (d + lat * (coeff[0] * t1 + coeff[1] * t2))
  fp = -ft
  interface = np.asarray(assembly.mate_interface(part, axis), dtype=float)
  rel = np.asarray(assembly.relative_pose(part)[1], dtype=float)
  # One physical contact point. The optional point error is transverse to the mate axis.
  xt = pt[:3] + Rt @ (rel + interface) + off * (coeff[2] * t1 + coeff[3] * t2)
  xp = xt.copy()
  tp, tt = np.zeros(3), np.zeros(3)
  skill = assembly.screw_skill(part)
  if skill is not None and float(skill.torque) > 0.0:
    s = float(np.sign(skill.angle))
    tp = s * float(skill.torque) * d
    tt = -tp
  return {part: (fp, tp, xp), parent: (ft, tt, xt)}


def _process_interface_gap(assembly, part, part_pose, parent_pose):
  """Distance between the two nominal mate-interface points at the current poses."""
  pp = np.asarray(part_pose, dtype=float)
  pt = np.asarray(parent_pose, dtype=float)
  Rp, Rt = _quat_to_R(pp[3:]), _quat_to_R(pt[3:])
  axis = np.asarray(assembly.mate_axis(part), dtype=float)
  axis /= np.linalg.norm(axis) + 1e-12
  interface = np.asarray(assembly.mate_interface(part, axis), dtype=float)
  rel = np.asarray(assembly.relative_pose(part)[1], dtype=float)
  xp = pp[:3] + Rp @ interface
  xt = pt[:3] + Rt @ (rel + interface)
  return float(np.linalg.norm(xp - xt))


def _process_realizations(plan, mode="nominal", corner=0, seed=0):
  """Plan-indexed, replayable insertion-uncertainty realizations.

  The planner's set has two lateral-force axes and, when ``--insertion-offset`` is nonzero,
  two contact-point axes. ``corner`` chooses one of its 4/16 vertices. ``sample`` draws once
  per assemble action from the box and never redraws during execution.
  """
  import wrench as W
  unc = W.PARAMS.get("insertion_uncertainty") or {}
  lat = float(unc.get("lateral", 0.0) or 0.0)
  off = float(unc.get("offset", 0.0) or 0.0)
  if mode not in ("nominal", "corner", "sample"):
    raise ValueError(f"unknown process-noise mode {mode!r}")
  if mode != "nominal" and lat <= 0.0 and off <= 0.0:
    raise ValueError(
        f"--process-noise {mode} needs a plan saved with --insertion-noise and/or "
        "--insertion-offset")
  dims = (2 if lat > 0.0 else 0) + (2 if off > 0.0 else 0)
  if mode == "corner" and not 0 <= int(corner) < (1 << dims):
    raise ValueError(f"process-noise corner {corner} outside [0, {(1 << dims) - 1}]")
  rng = np.random.default_rng(int(seed))
  out = {}
  for i, action in enumerate(plan):
    if action[0] != "assemble":
      continue
    vals = np.zeros(4)
    active = ([0, 1] if lat > 0.0 else []) + ([2, 3] if off > 0.0 else [])
    if mode == "corner":
      for bit, j in enumerate(active):
        vals[j] = 1.0 if (int(corner) >> bit) & 1 else -1.0
    elif mode == "sample":
      vals[active] = rng.uniform(-1.0, 1.0, size=len(active))
    out[i] = {"lateral": lat, "offset": off, "coefficients": vals.tolist()}
  return out


# ---------------------------------------------------------------------------------------------
# report / failure taxonomy
# ---------------------------------------------------------------------------------------------
class ActionResult:
  def __init__(self, action, ok, failure=None, detail=""):
    self.action, self.ok, self.failure, self.detail = tuple(action), ok, failure, detail

  def __repr__(self):
    tag = "ok" if self.ok else f"FAIL[{self.failure}]"
    return f"{self.action} -> {tag}{(' ' + self.detail) if self.detail else ''}"


class SimReport:
  """Per-action results + final assembly error. failure in
  {dropped, slid, toppled, grasp_broke, mate_broke, not_seated}."""
  def __init__(self):
    self.results = []
    self.final_pose_err = None                             # {part: (pos_err_m, ang_err_rad)}

  @property
  def success(self):
    return all(r.ok for r in self.results)

  def summary(self):
    """LEAD WITH THE FIRST FAILURE, not the final pose error.

    A final part error is almost always a CASCADE: one mate welds to a wrong datum, or one grasp
    breaks, and everything after it inherits the displacement -- so the headline mm number
    describes the wreckage, not the cause (measured: a 33.7mm first miss became a 1894mm final
    error). What is actionable is WHICH step failed first and WHY, with the wrench that broke it
    against the cap it was judged by. The final error is kept, demoted, as a rough severity hint."""
    n_ok = sum(r.ok for r in self.results)
    bad = [r for r in self.results if not r.ok]
    lines = [f"[sim] {n_ok}/{len(self.results)} actions ok; plan "
             f"{'SUCCEEDED' if self.success else 'FAILED'}"]
    if bad:
      first = bad[0]
      idx = self.results.index(first)
      lines.append(f"  FIRST FAILURE at result #{idx} -- everything after this may be a "
                   f"consequence, not an independent fault:")
      lines.append(f"      {first}")
      if len(bad) > 1:
        lines.append(f"  ... then {len(bad) - 1} more (likely downstream):")
        lines += [f"      {r}" for r in bad[1:]]
    if self.final_pose_err:
      worst = max(self.final_pose_err.items(), key=lambda kv: kv[1][0])
      lines.append(f"  (final-state severity hint, NOT the cause: worst part {worst[0]} "
                   f"pos {worst[1][0]*1000:.1f}mm ang {np.rad2deg(worst[1][1]):.1f}deg)")
    return "\n".join(lines)

  def to_dict(self, metadata=None):
    """Machine-readable counterpart of :meth:`summary` for benchmark runners."""
    first = next((i for i, r in enumerate(self.results) if not r.ok), None)
    return {
        "success": bool(self.success),
        "num_actions": len(self.results),
        "num_actions_ok": sum(bool(r.ok) for r in self.results),
        "first_failure_index": first,
        "first_failure": (None if first is None else self.results[first].failure),
        "results": [{"action": list(r.action), "ok": bool(r.ok),
                     "failure": r.failure, "detail": r.detail}
                    for r in self.results],
        "final_pose_err": {p: [float(v[0]), float(v[1])]
                           for p, v in (self.final_pose_err or {}).items()},
        "metadata": dict(metadata or {}),
    }


def _clone_state_with_poses(state, poses):
  """Copy a symbolic state while replacing only its non-identity pose payload."""
  from problem import State
  return State(state.parent, state.grasp, state.place, state.supports,
               {p: np.asarray(x, dtype=float).copy() for p, x in poses.items()})


def _sample_initial_placement_perturbation(C, assembly, state, xy=0.0, yaw_deg=0.0,
                                           seed=0, max_tries=200):
  """Sample a collision-free planar perturbation of the initial table scene.

  Each initially disconnected subassembly receives one rigid SE(2) displacement, sampled
  uniformly from ``[-xy, xy]^2 x [-yaw, yaw]``.  Z and the authored resting roll/pitch stay
  unchanged, so this models table-placement error rather than objects floating or changing
  support face.  The *whole* draw is rejected on collision; we never silently fall back to the
  nominal scene because that would bias a robustness benchmark toward success.
  """
  xy, yaw_deg = float(xy), float(yaw_deg)
  if xy < 0.0 or yaw_deg < 0.0:
    raise ValueError("object perturbation bounds must be non-negative")
  if int(max_tries) < 1:
    raise ValueError("object perturbation max tries must be at least one")
  nominal = {p: np.asarray(x, dtype=float).copy() for p, x in state.poses.items()}
  if xy == 0.0 and yaw_deg == 0.0:
    return _clone_state_with_poses(state, nominal), {
        "xy_bound_m": xy, "yaw_bound_deg": yaw_deg, "seed": int(seed),
        "attempts": 0, "groups": []}

  import robotic as ry
  Ct = ry.Config()
  Ct.addConfigurationCopy(C)
  groups = sorted((sorted(g) for g in state.subassemblies()), key=lambda g: tuple(g))
  # Only table-supported groups are placement observations. A preassembled group is moved
  # rigidly; a held group in an unusual initial state is deliberately left untouched.
  groups = [g for g in groups if any(state.parent.get(p) == ("table",) for p in g)]
  rng = np.random.default_rng(int(seed))
  last_collisions = None
  for attempt in range(1, int(max_tries) + 1):
    poses = {p: x.copy() for p, x in nominal.items()}
    records = []
    for group in groups:
      dx, dy = rng.uniform(-xy, xy, size=2) if xy else (0.0, 0.0)
      yaw = float(rng.uniform(-np.deg2rad(yaw_deg), np.deg2rad(yaw_deg))) \
          if yaw_deg else 0.0
      qyaw = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
      # Rotate a multi-part group about its first part, not the world origin.
      pivot = nominal[group[0]][:3]
      Rz = _quat_to_R(qyaw)
      for p in group:
        poses[p][:3] = pivot + np.array([dx, dy, 0.0]) + Rz @ (nominal[p][:3] - pivot)
        poses[p][3:] = _quat_mul(qyaw, nominal[p][3:])
      records.append({"parts": list(group), "dx_m": float(dx), "dy_m": float(dy),
                      "yaw_deg": float(np.rad2deg(yaw))})
    for p, pose in poses.items():
      f = Ct.getFrame(p)
      if f is not None:
        # Python RAI's Frame.setPose accepts its compact string syntax only; keep the numeric
        # representation explicit so no precision is lost in a benchmark draw.
        f.setPosition(pose[:3])
        f.setQuaternion(pose[3:])
    if Ct.getCollisionFree():
      return _clone_state_with_poses(state, poses), {
          "xy_bound_m": xy, "yaw_bound_deg": yaw_deg, "seed": int(seed),
          "attempts": attempt, "groups": records}
    last_collisions = Ct.getCollisions()
  raise ValueError(
      f"no collision-free initial placement perturbation in {max_tries} draws "
      f"(xy={xy:g}m, yaw={yaw_deg:g}deg, seed={seed}); last collisions: "
      f"{list(last_collisions or [])[:4]}")


# ---------------------------------------------------------------------------------------------
# MJCF construction: assembly spec + a State's poses -> model XML
# ---------------------------------------------------------------------------------------------
_PART_MESH_ASSETS = {}                 # (source identity, geometry digest) -> (name, obj path)
_PART_VISUAL_MESH_ASSETS = {}                              # exact display mesh path -> asset


def _part_mesh_asset(prm):
  """MuJoCo <mesh> asset for a ("mesh", {...}) part spec: the part's CONVEX HULL written to
  the sim_meshes/ cache as .obj (MuJoCo has no .stl/.ply loader we rely on, and it needs a
  convex geom anyway -- same v1 contract as the rai side). Deterministic name per
  (path, scale, geometry, *content*), memoized; the file is written once.

  The content term is essential for generated multi-geometry assets. Re-running VHACD keeps
  names such as ``hull_017`` but changes their vertices and often their ordering. The old
  path-only cache silently paired a new manifest offset with an old cached hull, visibly
  scattering Car and Duct collision pieces centimetres away from their renderer meshes.
  """
  import problem as P
  import hashlib
  hm = P._load_mesh(prm)["hull_mesh"]
  digest = hashlib.sha256(
      np.ascontiguousarray(hm.vertices, dtype=np.float64).tobytes()
      + np.ascontiguousarray(hm.faces, dtype=np.int64).tobytes()).hexdigest()[:16]
  key = (P._mesh_key(prm), digest)
  hit = _PART_MESH_ASSETS.get(key)
  if hit is None:
    import zlib
    name = f"part_{zlib.crc32(repr(key).encode()) & 0xffffffff:08x}"
    os.makedirs(_MESH_CACHE_DIR, exist_ok=True)            # defined below (arm_model's cache)
    path = os.path.join(_MESH_CACHE_DIR, name + ".obj")
    if not os.path.exists(path):
      with open(path, "w") as f:
        for v in hm.vertices:
          f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for a, b, c in hm.faces:
          f.write(f"f {a + 1} {b + 1} {c + 1}\n")
    hit = _PART_MESH_ASSETS[key] = (name, path)
  return hit


def _part_visual_mesh_asset(spec):
  """MuJoCo asset for an exact renderer-only part mesh (no hull or decomposition)."""
  import problem as P
  prm = {"file": spec} if isinstance(spec, str) else dict(spec)
  source = prm["file"]
  source = source if os.path.isabs(source) else os.path.join(P.HERE, source)
  source = os.path.realpath(source)
  hit = _PART_VISUAL_MESH_ASSETS.get(source)
  if hit is None:
    if not os.path.exists(source):
      raise FileNotFoundError(f"visual part mesh does not exist: {source}")
    import hashlib
    digest = hashlib.sha1(source.encode()).hexdigest()[:10]
    hit = _PART_VISUAL_MESH_ASSETS[source] = (f"part_visual_{digest}", source)
  return hit


def _spec_volume(spec):
  # volume of ONE sub-shape (ssBox corner rounding ignored; mesh = its convex hull) --
  # used to split a compound part's mass across its geoms so the inertia is roughly right.
  kind, p = spec[0], spec[1]
  if kind == "box":
    return float(p[0] * p[1] * p[2])
  if kind == "cylinder":
    return float(np.pi * p[1] * p[1] * p[0])
  if kind == "sphere":
    return float(4.0 / 3.0 * np.pi * p[0] ** 3)
  if kind == "mesh":
    import problem as P
    return float(P._load_mesh(p)["hull_mesh"].volume)
  raise ValueError(f"unknown shape kind {kind!r}")


def _compound_geoms_xml(spec, mu, mass, group, rgba, name_prefix=None):
  """FINISHED <geom> strings for a compound part: one geom per sub-shape, each at its own
  pos/quat inside the body, mass split by sub-shape volume (no mass_placeholder -- the
  caller's single-geom replace would stamp the FULL mass onto every child)."""
  import problem as P
  chs = P._compound_children(spec)
  vols = np.array([_spec_volume(s) for s, _ in chs], dtype=float)
  vols = vols / max(float(vols.sum()), 1e-12)
  out = []
  for i, ((sub, pos), w) in enumerate(zip(chs, vols)):
    sq = sub[2] if len(sub) > 2 else None
    gq = ' quat="1 0 0 0"' if sq is None else f' quat="{sq[0]} {sq[1]} {sq[2]} {sq[3]}"'
    gp = f' pos="{pos[0]} {pos[1]} {pos[2]}"'
    xml = _geom_xml(sub, mu)
    if name_prefix is not None:
      xml = xml.replace("<geom ", f'<geom name="{name_prefix}__collision{i}" ', 1)
    out.append(xml
               .replace("mass_placeholder",
                        f'mass="{mass * w}" group="{group}" rgba="{rgba}"{gq}{gp}')
               .replace("vis_placeholder",
                        f'group="{group}" rgba="{rgba}"{gq}{gp}'))
  return "".join(out)


def _contact_friction(mu, *, table=False):
  """MuJoCo sliding/torsional/rolling coefficients with the historical mu=0.5 values.

  The old generator changed only sliding friction: asking for ``mu=0`` silently retained
  0.005 torsional and 0.002 rolling friction. Scale all three together so low-friction and
  frictionless experiments mean what their saved configuration says, while mu=0.5 remains
  byte-for-byte the established model.
  """
  mu = max(0.0, float(mu))
  torsion = 0.01 * mu
  rolling = (0.004 if table else 0.0002) * mu
  return f"{mu:g} {torsion:g} {rolling:g}"


def _geom_xml(spec, mu):
  kind, p = spec[0], list(spec[1]) if not isinstance(spec[1], dict) else spec[1]
  fr = f'friction="{_contact_friction(mu)}"'
  if kind == "mesh":
    return f'<geom type="mesh" mesh="{_part_mesh_asset(p)[0]}" {fr} mass_placeholder/>'
  if kind == "compound":
    raise ValueError("compound parts are emitted via _compound_geoms_xml (several geoms)")
  if kind == "box":
    return f'<geom type="box" size="{p[0]/2} {p[1]/2} {p[2]/2}" {fr} mass_placeholder/>'
  if kind == "cylinder":
    # MuJoCo cylinder size is [radius, half-length]. The geom quaternion added by
    # build_mjcf carries the shape-local cylinder axis into the part frame.
    return f'<geom type="cylinder" size="{p[1]} {p[0]/2}" {fr} mass_placeholder/>'
  if kind == "sphere":
    return f'<geom type="sphere" size="{p[0]}" {fr} mass_placeholder/>'
  raise ValueError(f"unknown shape kind {kind}")


def _save_recording(path, scene, rec, report, metadata=None):
  import json
  np.savez_compressed(
      path, xml=scene.xml, qpos=np.asarray(rec["qpos"]),
      mpos=np.asarray(rec["mpos"]), mquat=np.asarray(rec["mquat"]),
      forces=np.asarray(rec["forces"]), robots=np.array(scene.robots),
      parts=np.array(scene.assembly.objects), marks=json.dumps(rec["marks"]),
      controller_mode=np.asarray(rec.get("controller_mode", []), dtype="U24"),
      physics_tick=np.asarray(rec.get("physics_tick", []), dtype=np.int64),
      gripper_events=json.dumps(rec.get("gripper_events", []), sort_keys=True),
      results=json.dumps([repr(r) for r in report.results]),
      success=np.asarray(bool(report.success)),
      final_pose_err=json.dumps(report.final_pose_err or {}),
      metadata=json.dumps(metadata or {}, sort_keys=True))
  print(f"[sim] recording ({len(rec['qpos'])} frames) -> {path}")


def _record_controller_mode(action):
  """Stable annotation for the control law owning one recorded physics sample.

  Ordinary retimed-path tracking is the open-loop reference.  The named modes below are
  feedback/event hooks that a downstream executor must either implement or deliberately
  replace.  Phase vectors and other non-string trace payloads are ordinary trajectory samples.
  """
  head = action[0] if isinstance(action, (tuple, list)) and action else action
  return {
      "servo": "skill_servo",
      "settle": "settle",
      "approach": "approach",
      "guard": "wrench_guard",
      "path": "trajectory",
  }.get(head, "trajectory") if isinstance(head, str) else "trajectory"


# viewer geom GROUPS (toggled by number keys 0-5 in the MuJoCo viewer / replay window):
GROUP_PART = 0                  # the assembly parts (always on)
GROUP_ENV = 1                   # floor + table
GROUP_COLL = 2                  # arm COLLISION capsules (the planner's model)
GROUP_VISUAL = 3                # arm VISUAL link meshes + gripper primitives (nice look)

# MESH EXTRACTION LIVES IN arm_model: it is "read geometry out of rai", the same job as the
# frame tree, and the articulated model needs it too. Re-exported here under the old private
# names so the mocap path reads unchanged.
_MESH_CACHE_DIR = arm_model._MESH_CACHE_DIR
_write_obj = arm_model._write_obj
_link_meshes = arm_model.link_meshes
_ARM_COLORS = arm_model._ARM_COLORS


def _obj_usable(obj):
    """MuJoCo rejects meshes whose computed inertia is degenerate ('volume too small'):
    filter converted display meshes that are flat/tiny before they reach build_mjcf."""
    try:
        vs = []
        with open(obj) as fh:
            for ln in fh:
                if ln.startswith("v "):
                    p = ln.split()
                    vs.append((float(p[1]), float(p[2]), float(p[3])))
        if len(vs) < 4:
            return False
        v = np.asarray(vs)
        return bool(np.all(np.isfinite(v)) and float((v.max(0) - v.min(0)).max()) > 0.03)
    except Exception:
        return False


def _arm_shape_frames(C, robots, visual=True):
  """Kinematic-body specs for the robots, each mirrored as an FK-synced mocap body. Returns
  [(body, fk_frame, geom, collide, exempt, group, rgba), ...] where
      geom = ("capsule"|"cylinder"|"sphere"|"box", size_list) | ("mesh", obj_path).
  `body` is the (unique) MuJoCo body name; `fk_frame` is the rai frame whose FK pose drives it
  (a color-split link mesh yields several bodies sharing one fk_frame). COLLISION model =
  rai's _coll* CAPSULES (group COLL) -- the exact geometry the planner certified; VISUAL link
  meshes (group VISUAL, contype 0, real UR colors) are added when `visual`."""
  out = []
  for j, r in enumerate(robots):
    tint = _ARM_COLORS[j % len(_ARM_COLORS)]
    for n in C.getFrameNames():
      if not n.startswith(r):
        continue
      info = C.getFrame(n).info()
      k = info.get("shape")
      if k == "mesh":
        if visual:
          try:
            mi = 0
            for obj, rgba in _link_meshes(C, n):
              if not _obj_usable(obj):
                continue                       # degenerate mesh: MuJoCo would reject it
              out.append((f"{n}__m{mi}", n, ("mesh", obj), False, False,
                          GROUP_VISUAL, rgba))
              mi += 1
          except Exception:
            pass                                           # degenerate / no mesh: skip
      elif k in ("capsule", "cylinder", "sphere", "box", "ssBox"):
        sz = list(np.asarray(info.get("size"), dtype=float))
        if "_coll" in n:
          # rai contact=-2 (WRIST cluster nearest the tool) is parent-chain exempt vs the
          # ATTACHED part (a held cube hangs 35mm from the tip, inside the wrist capsule).
          out.append((n, n, (k, sz), True, info.get("contact") == -2, GROUP_COLL, tint))
        else:
          # gripper primitives (palm, fingers, vacuum tip, gripper_fill): these DO collide in
          # rai (the oracle sees them as colliding pairs), so the executed model must collide
          # them too -- visual-only made the rendered gripper pass through parts with no
          # physics response (user-caught: palm 28mm into a leg). NOT exempt: the palm keeps
          # >5mm clearance from its own held part (measured), and net_contact_force already
          # excludes weld partners, so hold contacts cannot pollute the guard.
          out.append((n, n, (k, sz), True, False, GROUP_VISUAL, tint))
  return out


def scale_arm_collision_shapes(C, robots, factor, verbose=True):
  """DEBUG AID: shrink every contact-enabled ARM collision primitive by `factor`
  (1.0 = untouched, byte-identical). Covers not just the `_coll*` capsules but the gripper
  primitives too (palm / fingers / gripper_fill DO collide in rai -- they show up as
  colliding pairs in the oracle). Radius-only for capsule/cylinder/sphere (thinner arm,
  unchanged reach/kinematics); all extents for boxes.

  PARTS ARE NEVER SCALED -- mate seating and the assembly's own geometry depend on their
  true size. Scaling changes what the planner/oracle considers free, so clearances no
  longer match the certified path: for debugging desync/wedges, not for validation runs.
  Applied to the rai config BEFORE witnesses/problem/sim, so IK, the collision oracle and
  the MuJoCo capsules all see the same shrunken geometry. Returns #frames scaled."""
  if factor is None or abs(float(factor) - 1.0) < 1e-12:
    return 0
  import robotic as ry
  ST = {"capsule": ry.ST.capsule, "cylinder": ry.ST.cylinder, "sphere": ry.ST.sphere,
        "box": ry.ST.box, "ssBox": ry.ST.ssBox}
  factor = float(factor)
  n = 0
  for nm in C.getFrameNames():
    if not any(nm.startswith(r) for r in robots):
      continue
    f = C.getFrame(nm)
    info = f.info()
    kind = info.get("shape")
    if kind not in ST or not info.get("contact"):            # shapeless / visual-only: skip
      continue
    sz = list(np.asarray(info.get("size"), dtype=float))
    if kind in ("capsule", "cylinder"):
      sz[1] *= factor                                        # radius: thinner, same length
    elif kind == "sphere":
      sz[0] *= factor
    elif kind == "ssBox":
      sz = [v * factor for v in sz[:3]] + list(sz[3:])       # extents; keep corner radius
    else:
      sz = [v * factor for v in sz]
    contact = info.get("contact")
    f.setShape(ST[kind], sz)
    f.setContact(int(contact))                               # setShape may clear the flag
    n += 1
  if verbose and n:
    print(f"[sim] --coll-scale {factor:g}: shrank {n} arm collision primitives (parts "
          f"untouched). DEBUG ONLY -- clearances no longer match the certified geometry.")
  return n


GRAVITY = 9.81
ARM_ARM_CONTACTS = False   # --arm-arm-contacts: per-robot bits enable cross-arm physics while
                           # excluding same-chain self contacts. Legacy mode masks all arm-arm.
ARTICULATED_LINK_MASS = 50.0     # kg: --articulated links are driven kinematically, so this is
ARTICULATED_LINK_INERTIA = 5.0   # not a real inertia -- it only has to dominate a held part
                                 # (~0.5kg) so the weld reaction cannot move the arm.
ACTUATED_LINK_MASS = 4.0         # kg per link for --actuated. A first-order stand-in for real
                                 # UR10 link masses (4.0/7.8/12.9/3.9/2.0/0.2): gravity torque
                                 # must land inside rai's real effort limits (330/150/54 Nm),
                                 # which the 50kg kinematic links would blow past. Per-link
                                 # inertias are the obvious next refinement.
ACTUATED_KP = 3000.0             # position-servo gain; kv defaults to 2*sqrt(kp) in arm_model
ARTICULATED_ARMATURE = 0.0       # MEASURED DEAD END: 1e6 was meant to make the chain
                                 # unaccelerable, but it ill-conditions the mass matrix and made
                                 # the first missed seat 66mm -> 20001mm. The arm is pinned by
                                 # re-asserting qpos after each step instead (SimScene.step).
LEVER_SANITY = 0.35   # m: a grasp lever beyond this means the support info is stale, not a long arm


def _static_load(mass, lever):
  """Static wrench a held group already imposes on its grasp: (|F|, |T|).

  SWAPPABLE: this is the cheap weight-based estimate (m*g, and its torque about the grasp
  through the lever arm). The principled replacement is the wrench LP's own per-action grasp
  load (equilibrium_feasible), which also accounts for bracing and insertion reactions --
  drop it in here and nothing else changes."""
  return float(mass) * GRAVITY, float(mass) * GRAVITY * float(lever)


def _make_wrench_tracer(scene, predict=None, out=print):
  """SIM_WRENCH_TRACE=1 -- why a grasp weld breaks (same style as SIM_MP_TRACE/RETIMING_TRACE).

  For every ACTIVE grasp it lines up three wrenches at the SAME instant:
    meas  what MuJoCo's weld constraint actually carries -- the value step() has already
          read for its breakaway check, so tracing adds no efc work;
    model what --force-limits predicts at this arc position (`predict(robot) -> dict`),
          including whether the model believes the arm is holding anything at all there
          (m == 0 means the limit is INACTIVE, not that the load is zero);
    rb    the exact rigid-body wrench the held group's MEASURED motion requires, split into
          the point-mass part |sum r_i x m_i(a_i-g)| the model keeps and the rotational part
          |sum I_i alpha_i + w_i x I_i w_i| it omits.
  plus `cmd`, the COMMANDED motion of the grasp anchor itself (finite-differenced from the
  mocap poses sync_arms writes -- mocap bodies carry no MuJoCo velocity, so this is the only
  way to see it). cmd is the ground truth the model is trying to predict: cmd >> model means
  the executed path is not the path the limits were computed on (e.g. a C0 waypoint kink,
  which is a VELOCITY step and therefore an unbounded acceleration); cmd small while rb is
  large means the part is ringing on the weld rather than being driven.

  Printed every SIM_WRENCH_EVERY steps (default 200 = 0.4s) AND every step once the measured
  wrench passes SIM_WRENCH_FRAC of either cap (default 0.7), so a break is captured at full
  500Hz without flooding the log. SIM_WRENCH_ARM=a3 restricts it to one robot."""
  every = max(1, int(os.environ.get("SIM_WRENCH_EVERY", "200")))
  frac = float(os.environ.get("SIM_WRENCH_FRAC", "0.7"))
  only = os.environ.get("SIM_WRENCH_ARM")
  g_vec = np.asarray(scene.model.opt.gravity, dtype=float)
  bodies = {p: scene._body_id(p) for p in scene.assembly.objects}
  vel = {p: (np.zeros(3), np.zeros(3)) for p in bodies}      # part -> (w, v), previous step
  acc = {p: (np.zeros(3), np.zeros(3)) for p in bodies}      # part -> (alpha, a), this step
  anchors = {r: scene._body_id(b) for r, b in scene.grasp_anchor.items()}
  cmd_p = {r: None for r in anchors}                         # robot -> (pos, quat) prev step
  cmd_v = {r: (np.zeros(3), np.zeros(3)) for r in anchors}   # robot -> (w, v) commanded
  cmd_a = {r: (np.zeros(3), np.zeros(3)) for r in anchors}   # robot -> (alpha, a) commanded
  cmd_turn = {r: 0.0 for r in anchors}                       # robot -> deg turned this step
  last = [-1]

  def sample():
    """Cache every part's world velocity + finite-differenced acceleration, and the same for
    the COMMANDED grasp-anchor poses. Per step, but ~one mj_objectVelocity call per part --
    negligible next to mj_step."""
    v6 = np.zeros(6)
    for p, b in bodies.items():
      mujoco.mj_objectVelocity(scene.model, scene.data, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
      w, v = v6[:3].copy(), v6[3:].copy()
      acc[p] = ((w - vel[p][0]) / DT, (v - vel[p][1]) / DT)
      vel[p] = (w, v)
    q, dq, om = np.zeros(4), np.zeros(4), np.zeros(3)
    for r, b in anchors.items():
      pos = np.asarray(scene.data.xpos[b], dtype=float).copy()
      mujoco.mju_mat2Quat(q, np.asarray(scene.data.xmat[b], dtype=float).ravel())
      if cmd_p[r] is not None:
        p0, q0 = cmd_p[r]
        mujoco.mju_negQuat(dq, q0)
        mujoco.mju_mulQuat(dq, q, dq)                        # q * conj(q0) = world-frame delta
        mujoco.mju_quat2Vel(om, dq, DT)
        v = (pos - p0) / DT
        v0 = cmd_v[r][1]
        cmd_a[r] = ((om - cmd_v[r][0]) / DT, (v - v0) / DT)
        # TURN vs SPEED CHANGE: a C0 path corner turns the velocity without changing its
        # magnitude; a scheduler velocity step (corridor/event clamp) does the opposite.
        nn = float(np.linalg.norm(v) * np.linalg.norm(v0))
        cmd_turn[r] = (np.degrees(np.arccos(np.clip(float(v @ v0) / nn, -1.0, 1.0)))
                       if nn > 1e-12 else 0.0)
        cmd_v[r] = (om.copy(), v)
      cmd_p[r] = (pos, q.copy())

  def group_of(part):
    """The parts rigidly welded to `part` RIGHT NOW (active mate welds) -- what the grasp
    actually carries, which is not the planner's symbolic group after a missed seat."""
    adj = {}
    for eq in range(scene.model.neq):
      if not scene.data.eq_active[eq]:
        continue
      nm = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_EQUALITY, eq)
      if nm and nm.startswith("mate_"):
        c = nm[len("mate_"):]
        p = scene.assembly.relative_pose(c)[0]
        adj.setdefault(c, set()).add(p)
        adj.setdefault(p, set()).add(c)
    seen, stack = {part}, [part]
    while stack:
      x = stack.pop()
      for y in adj.get(x, ()):
        if y in bodies and y not in seen:
          seen.add(y)
          stack.append(y)
    return sorted(seen)

  def rigid_body(parts, anchor):
    """Newton-Euler wrench these bodies impose about `anchor` at their measured motion:
    (mass, |F|, |T_total|, |T_pointmass|, |T_rot|, |a|, |w|, |alpha|)."""
    F = np.zeros(3)
    Tpm = np.zeros(3)
    Trot = np.zeros(3)
    m_tot = a_max = w_max = al_max = 0.0
    for p in parts:
      b = bodies[p]
      m = float(scene.model.body_mass[b])
      R = np.asarray(scene.data.ximat[b], dtype=float).reshape(3, 3)
      I = R @ np.diag(np.asarray(scene.model.body_inertia[b], dtype=float)) @ R.T
      w, _ = vel[p]
      alpha, a = acc[p]
      f = m * (a - g_vec)
      r = np.asarray(scene.data.xipos[b], dtype=float) - anchor
      F += f
      Tpm += np.cross(r, f)
      Trot += I @ alpha + np.cross(w, I @ w)
      m_tot += m
      a_max, w_max = max(a_max, float(np.linalg.norm(a))), max(w_max, float(np.linalg.norm(w)))
      al_max = max(al_max, float(np.linalg.norm(alpha)))
    return (m_tot, float(np.linalg.norm(F)), float(np.linalg.norm(Tpm + Trot)),
            float(np.linalg.norm(Tpm)), float(np.linalg.norm(Trot)), a_max, w_max, al_max)

  def trace(robot, part, fmag, tmag, fcap, tcap):
    if scene.step_i != last[0]:                              # once per sim step, not per grasp
      sample()
      last[0] = scene.step_i
    if only and robot != only:
      return
    hot = fmag > frac * fcap or tmag > frac * tcap
    if not hot and scene.step_i % every:
      return
    grp = group_of(part)
    # CO-HOLDERS: two grippers on one rigid group are kinematically locked -- desyncing them
    # does not "collide", it TEARS the part out of a gripper (pure force, no torque). The
    # corridor forces those pairs together only where the plan's supports show them, so a
    # co-hold the trace sees but the plan does not is the difference that matters.
    held = sorted(r for r, g in scene.active_grasps.items() if g[1] in grp)
    anchor = np.asarray(scene.data.xpos[scene._body_id(scene.grasp_anchor[robot])], dtype=float)
    m_rb, F_rb, T_rb, T_pm, T_rot, a_rb, w_rb, al_rb = rigid_body(grp, anchor)
    pm = predict(robot) if predict is not None else None
    ms = ("model none" if pm is None else
          f"model m={pm['m']:.2f}kg L={pm['L']*1000:.0f}mm s={pm['s']:.2f} sdot={pm['v']:.3f} "
          f"sddot={pm['acc']:+.2f} |a|={pm['a']:.2f} F={pm['F']:.1f} T={pm['T']:.3f} "
          f"vmax={pm['vmax']:.2f} kink={pm['kink']:.2f} dbar={pm['dbar']:.2f}")
    cw, cv = cmd_v[robot]
    ca, cl = cmd_a[robot][0], cmd_a[robot][1]
    out(f"[wrench] k={scene.step_i} t={scene.step_i * DT:.3f} {robot}/{part} "
        f"meas F={fmag:.1f}/{fcap:g} T={tmag:.3f}/{tcap:g} | {ms} | "
        f"rb m={m_rb:.2f}kg({len(grp)}p, held by {'+'.join(held)}) "
        f"F={F_rb:.1f} T={T_rb:.3f} (pm {T_pm:.3f} + rot "
        f"{T_rot:.3f}) |a|={a_rb:.2f} |w|={w_rb:.2f} |alpha|={al_rb:.1f} | "
        f"cmd |v|={np.linalg.norm(cv):.3f} |a|={np.linalg.norm(cl):.2f} "
        f"|w|={np.linalg.norm(cw):.3f} |alpha|={np.linalg.norm(ca):.1f} "
        f"turn={cmd_turn[robot]:.1f}deg")

  return trace


def _default_geomgroups(viewer, has_visual):
  """Initial viewer group visibility. MuJoCo HIDES groups >=3 by default, so we set them
  explicitly: when exact part or real link meshes exist, SHOW them (group VISUAL) and HIDE the
  arm/part collision geometry (group COLL) so the scene reads cleanly. Imported part CAD
  collision decompositions use group 2; their exact renderer meshes use group 3. Procedural
  parts without a visual counterpart remain in group 0. Toggle with number keys 2 and 3."""
  if viewer is None:
    return
  if has_visual:
    viewer.opt.geomgroup[GROUP_VISUAL] = 1                 # show the real link meshes
    viewer.opt.geomgroup[GROUP_COLL] = 0                   # hide the collision capsules


def _arm_geom_xml(geom):
  """MJCF geom shape+size for a non-mesh arm geom. rai capsule/cylinder size = [length,
  radius]; MJCF wants 'radius half-length'. (mesh geoms are emitted via an asset ref.)"""
  kind, size = geom[0], geom[1]
  if kind in ("capsule", "cylinder"):
    return f'type="{kind}" size="{size[1]} {size[0]/2}"'
  if kind == "sphere":
    return f'type="sphere" size="{size[0]}"'
  return f'type="box" size="{size[0]/2} {size[1]/2} {size[2]/2}"'   # box / ssBox


def joint_vmax(C):
  """Per-joint velocity limits (rad/s) in the config's joint order, from rai. `info()["limits"]`
  is [lo, hi, vel, ?, effort, ...]; a missing/zero entry becomes inf so that joint never binds.
  UR10: 2.16 pan/lift, 3.15 elbow, 3.20 wrists -- against the nominal JOINT_RATE of 1.0."""
  out = []
  for n in C.getJointNames():
    try:
      lim = np.asarray(C.getFrame(n).info().get("limits", []), dtype=float).ravel()
      out.append(float(lim[2]) if lim.size > 2 and lim[2] > 0 else np.inf)
    except Exception:
      out.append(np.inf)
  return np.asarray(out, dtype=float)


def planner_table_geometry(C):
  """Table geometry the search used: ``(x, y, half_x, half_y, half_z, top_z)``.

  The old simulation only recovered the optional top height and otherwise used the global
  ``TABLE_HALF`` footprint at the world origin. That happens to be unobtrusive for the old
  UR scenes, but turns the FR3's 1.2 x 0.9 m bench into a 2.4 x 2.4 m platform. Read the
  footprint and xy placement from the same rebuilt rai scene as the motion-planning oracle.
  """
  try:
    f = C.getFrame("table")
    pos = np.asarray(f.getPosition(), dtype=float)
    size = np.asarray(f.getSize(), dtype=float)
    return (float(pos[0]), float(pos[1]), float(size[0]) / 2.0,
            float(size[1]) / 2.0, float(size[2]) / 2.0,
            float(pos[2]) + float(size[2]) / 2.0)
  except Exception:
    return None


def planner_table_top(C):
  """Backward-compatible scalar accessor used by older callers/tests."""
  geometry = planner_table_geometry(C)
  return None if geometry is None else geometry[5]


def build_mjcf(assembly, state, robots, mu_table=0.5, arms=None, arm_poses=None,
               ee_anchor=None, articulated_C=None, actuated=False, planner_table_top=None,
               weld_timeconst=None, table_geometry=None, animate_grippers=False,
               object_damping=0.0):
  """The whole validation world as one MJCF string:
  - table (static), parts (free bodies at state.poses, shape quat baked into the geom),
  - robots: EITHER one mocap-driven gripper puck per robot (lightweight physics tests) OR,
    when `arms` is given, one mocap body per ARM SHAPE FRAME
    (_arm_shape_frames), positioned by rai FK each waypoint -- the planner's own kinematics
    by construction. arm_poses = {frame: initial 7-pose}; ee_anchor = {robot: frame} names
    the body grasp welds attach to (the tool frame).
  - PRE-DECLARED, INACTIVE weld equalities for (i) every (robot, part) grasp pair and
    (ii) every mate edge of the manual -- MuJoCo equalities live in the model, so all
    potentially-needed welds are declared up front and toggled at runtime."""
  # table TOP from the parts themselves: State.poses are rai WORLD poses where parts hover
  # `part_float` (~15mm) above the top; placing the top at min(part bottom) lets every part
  # START RESTING (no settle drop) while keeping x/y exactly as planned. The planner's
  # expected z stays within DROP_Z of the sim's, so the judge is unaffected.
  tops = []
  for p in assembly.objects:
    pose = np.asarray(state.poses[p], dtype=float)
    spec = assembly.part_shape(p)
    tops.append(pose[2] - _shape_half_z(spec, _quat_to_R(pose[3:])))
  table_top = float(min(tops))
  if planner_table_top is not None:
    # --planner-table: use the table the SEARCH reasoned about, not one raised to meet the
    # floated parts. The derivation above puts the top at the lowest part's BOTTOM, i.e.
    # planner_top + part_float (15mm by default) -- so the arm meets the table 15mm early and
    # presses against geometry the plan never saw (the enabling condition for the 81N break).
    # Cost: the parts now START part_float above the table and settle onto it.
    if abs(float(planner_table_top) - table_top) > 1e-9:
      print(f"[flag] --planner-table: table top {table_top:.4f} -> {float(planner_table_top):.4f} "
            f"({(table_top - float(planner_table_top))*1000:+.1f}mm); parts start that far above "
            f"it and settle instead of resting.")
    table_top = float(planner_table_top)
  table_x, table_y = 0.0, 0.0
  table_half_x, table_half_y = TABLE_HALF[:2]
  table_half_z = table_top / 2.0
  if table_geometry is not None:
    table_x, table_y, table_half_x, table_half_y, table_half_z, _ = table_geometry
  parts_xml, eq_xml, contact_xml = [], [], []
  # <mesh> assets for MESH-shaped parts and mesh CHILDREN of compound parts (deduped by
  # asset name; _geom_xml references them)
  def _mesh_specs():
    import problem as _P
    for p in assembly.objects:
      spec = assembly.part_shape(p)
      if spec[0] == "mesh":
        yield spec[1]
      elif spec[0] == "compound":
        for sub, _off in _P._compound_children(spec):
          if sub[0] == "mesh":
            yield sub[1]
  part_mesh_assets = "".join(dict.fromkeys(
      f'<mesh name="{_part_mesh_asset(prm)[0]}" file="{_part_mesh_asset(prm)[1]}"/>'
      for prm in _mesh_specs()))
  visual_mesh_assets = "".join(dict.fromkeys(
      f'<mesh name="{_part_visual_mesh_asset(v)[0]}" '
      f'file="{_part_visual_mesh_asset(v)[1]}" inertia="shell"/>'
      for v in getattr(assembly, "visual_meshes", {}).values()))
  # deterministic per-part colors (tab10-ish) -- rai's colors are np.random.rand, not
  # reproducible here; a fixed palette keeps replays comparable across runs.
  palette = ["0.84 0.15 0.16 1", "0.12 0.47 0.71 1", "0.17 0.63 0.17 1", "1.0 0.5 0.05 1",
             "0.58 0.4 0.74 1", "0.55 0.34 0.29 1", "0.89 0.47 0.76 1", "0.74 0.74 0.13 1"]
  for i, p in enumerate(assembly.objects):
    spec = assembly.part_shape(p)
    visual = getattr(assembly, "visual_meshes", {}).get(p)
    # Keep the viewer contract explicit: key 2 is collision geometry and key 3 is renderer
    # geometry. Imported CAD has both, so putting its convex decomposition in group 0 made
    # the two inseparable and made visual inspection misleading.
    collision_group = GROUP_COLL if visual else GROUP_PART
    collision_rgba = palette[i % len(palette)]
    if visual:
      collision_rgba = " ".join(collision_rgba.split()[:3] + ["0.32"])
    pose = np.asarray(state.poses[p], dtype=float)
    q = pose[3:]
    sq = spec[2] if len(spec) > 2 else None
    gq = ' quat="1 0 0 0"' if sq is None else f' quat="{sq[0]} {sq[1]} {sq[2]} {sq[3]}"'
    mass = assembly.masses.get(p, 0.1)
    if spec[0] == "compound":                              # several geoms, mass by volume
      g = _compound_geoms_xml(spec, mu_table, mass, collision_group,
                              collision_rgba, name_prefix=p)
    else:
      g = _geom_xml(spec, mu_table).replace("<geom ",
          f'<geom name="{p}__collision" ', 1).replace(
          "mass_placeholder",
          f'mass="{mass}" group="{collision_group}" rgba="{collision_rgba}"{gq}').replace(
          "vis_placeholder",
          f'group="{collision_group}" rgba="{collision_rgba}"{gq}')
    if visual:
      vprm = {"file": visual} if isinstance(visual, str) else dict(visual)
      vname, _vpath = _part_visual_mesh_asset(vprm)
      vpos = vprm.get("pos", (0.0, 0.0, 0.0))
      vquat = vprm.get("quat", (1.0, 0.0, 0.0, 0.0))
      vrgba = vprm.get("rgba", "0.82 0.66 0.42 1")
      # Exact CAD surface is display-only. The simplified cuboids above remain the one
      # collision/inertia model shared with rai and the planner.
      g += (f'<geom name="{p}__visual" type="mesh" mesh="{vname}" '
            f'pos="{vpos[0]} {vpos[1]} {vpos[2]}" '
            f'quat="{vquat[0]} {vquat[1]} {vquat[2]} {vquat[3]}" '
            # ``contype=conaffinity=0`` removes contacts, but it does NOT remove a geom's
            # inertia contribution: without an explicit zero mass MuJoCo applies its default
            # density to the CAD shell.  The real-stool seat was consequently compiled as
            # 108.94 kg instead of its authored 0.75 kg (and each 80 g leg as 12.05 kg), so
            # perfectly ordinary table support appeared as 1069 N / 118 N and every grasp
            # broke.  Display meshes must be inert both geometrically AND dynamically; the
            # simplified collision geoms above are the sole mass/inertia model.
            f'rgba="{vrgba}" group="{GROUP_VISUAL}" mass="0" '
            f'contype="0" conaffinity="0"/>')
    # Renderer-only motion indicators are not part of the collision/grasp/wrench model.
    # Give them negligible mass and disable both contact masks; they remain rigid children
    # of the part body and therefore expose rotation in recordings.
    for marker in getattr(assembly, "visual_markers", {}).get(p, []):
      mspec = marker["shape"]
      mpos = marker.get("pos", (0.0, 0.0, 0.0))
      mrgba = marker.get("rgba", "0.95 0.95 0.1 1")
      mq = mspec[2] if len(mspec) > 2 else None
      mquat = (' quat="1 0 0 0"' if mq is None else
               f' quat="{mq[0]} {mq[1]} {mq[2]} {mq[3]}"')
      mpose = f' pos="{mpos[0]} {mpos[1]} {mpos[2]}"{mquat}'
      g += _geom_xml(mspec, mu_table).replace(
          "mass_placeholder",
          f'mass="1e-8" group="{GROUP_PART}" rgba="{mrgba}" '
          f'contype="0" conaffinity="0"{mpose}').replace(
          "vis_placeholder",
          f'group="{GROUP_PART}" rgba="{mrgba}" contype="0" conaffinity="0"{mpose}')
    free_joint = ("<freejoint/>" if float(object_damping) == 0.0 else
                  f'<joint type="free" damping="{float(object_damping):g}"/>')
    parts_xml.append(
        f'<body name="{p}" pos="{pose[0]} {pose[1]} {pose[2]}" '
        f'quat="{q[0]} {q[1]} {q[2]} {q[3]}">{free_joint}{g}</body>')
  ee_colors = ["0.2 0.2 0.9 1", "0.9 0.2 0.2 1", "0.2 0.8 0.8 1", "0.9 0.7 0.1 1",
               "0.6 0.2 0.8 1", "0.4 0.9 0.3 1"]
  arm_mesh_assets = ""
  # MuJoCo has no semantic notion of "same robot". One shared arm contact bit therefore
  # enables both cross-robot contacts and catastrophic self-contact between adjacent links.
  # Allocate one bit per robot; each arm accepts the part bit (1) and every OTHER arm bit.
  _arm_bits = {r: 1 << (j + 1) for j, r in enumerate(robots)}
  _all_arm_bits = sum(_arm_bits.values())
  def _arm_contact_mask(robot):
    bit = _arm_bits[robot]
    return str(bit), str(1 | (_all_arm_bits ^ bit))
  if arms is None:                                         # lightweight test fixture
    for j, r in enumerate(robots):
      parts_xml.append(
          f'<body name="ee_{r}" mocap="true" pos="0 0 1"><geom type="cylinder" '
          f'size="0.02 0.02" mass="0.2" contype="0" conaffinity="0" group="{GROUP_VISUAL}" '
          f'rgba="{ee_colors[j % len(ee_colors)]}"/></body>')
    anchor = {r: f"ee_{r}" for r in robots}
  else:                                                    # FK-synced arm link bodies
    mesh_assets = []
    # ensure every robot's ee-anchor exists as a drivable mocap body: shapeless alias
    # frames (e.g. the fr3 duo's ur_gripper_center) never reach _arm_shape_frames, but
    # grasp welds anchor to exactly this body name.
    have = {a[0] for a in arms}
    for r in robots:
      n = ee_anchor[r]
      if n not in have:
        arms = list(arms) + [(n, n, ("sphere", [0.004]), False, False,
                              GROUP_VISUAL, [0.9, 0.9, 0.9, 0.4])]
    for body, fkf, geom_spec, collide, _ex, group, rgba in arms:
      pose = arm_poses[fkf]
      q = pose[3:]
      # contype 2 / conaffinity 1: arm links collide with PARTS and the table, never with
      # each other (self/arm-arm clearance was the MOTION PLANNER's job; re-checking it
      # here on mocap bodies would only produce huge phantom forces on interpenetration).
      _owner = next((r for r in robots if body.startswith(r)), None)
      if collide and ARM_ARM_CONTACTS and _owner is not None:
        _ct, _ca = _arm_contact_mask(_owner)
        con = f'contype="{_ct}" conaffinity="{_ca}"'
      else:
        con = ('contype="2" conaffinity="1"' if collide
               else 'contype="0" conaffinity="0"')
      if geom_spec[0] == "mesh":                           # real link mesh (visual only)
        mesh_assets.append(f'<mesh name="m_{body}" file="{geom_spec[1]}" inertia="shell"/>')
        geom = f'type="mesh" mesh="m_{body}"'
      else:
        geom = _arm_geom_xml(geom_spec)
      # some rai colors arrive as 3-vectors, strings ('0.2 0.2 0.9 1'), or odd dtypes
      if isinstance(rgba, str):
        rgba = [float(v) for v in rgba.replace(",", " ").split()]
      _rgba = [float(v) for v in np.asarray(rgba, dtype=float).ravel()] + [1.0] * 4
      _rgba = ' '.join(f'{v:g}' for v in _rgba[:4])
      parts_xml.append(
          f'<body name="arm_{body}" mocap="true" pos="{pose[0]} {pose[1]} {pose[2]}" '
          f'quat="{q[0]} {q[1]} {q[2]} {q[3]}">'
          f'<inertial pos="0 0 0" mass="1" diaginertia="1e-4 1e-4 1e-4"/>'
          f'<geom {geom} mass="1.0" {con} group="{group}" rgba="{_rgba}"/></body>')
    anchor = {r: f"arm_{ee_anchor[r]}" for r in robots}
    arm_mesh_assets = "".join(mesh_assets)
  if articulated_C is not None:
    # ARTICULATED (--articulated): one real kinematic chain per arm, generated from rai's own
    # tree (arm_model), replacing the N disconnected mocap bodies above. Bodies keep the rai
    # FRAME NAMES, so the grasp weld anchors to ee_anchor[r] directly with no prefix. Driven
    # by SETTING qpos (SimScene.set_arm_q) -- still perfectly stiff, so the grasp weld stays
    # the clean measuring device every wrench result so far was calibrated on.
    # VISUAL LINK MESHES follow whatever the mocap mirrors were built with (`arms` is already
    # filtered by the headless rule), so --articulated shows the real UR10/UR5 exactly when the
    # mocap path would. They are visual-only, so the physics is identical either way.
    _arm_assets = []
    _arm_meshes = any(e[2][0] == "mesh" for e in arms)
    _contact_masks = ([_arm_contact_mask(r) for r in robots]
                      if ARM_ARM_CONTACTS else None)
    body_xml, articulated_joints = arm_model.build_arms_mjcf(
        articulated_C, robots, collide_fn=arm_model.collides_like_sim,
        group=GROUP_COLL, contype=("1" if ARM_ARM_CONTACTS else "2"), conaffinity="1",
        contact_masks=_contact_masks,
        meshes=_arm_meshes, assets=_arm_assets, mesh_group=GROUP_VISUAL,
        # links must SWAMP the payload: the chain is integrated by mj_step even though we
        # drive it kinematically, so a light link welded to a part lets the solver throw it.
        armature=ARTICULATED_ARMATURE, limits=bool(actuated),
        link_mass=(ACTUATED_LINK_MASS if actuated else ARTICULATED_LINK_MASS),
        link_inertia=((ACTUATED_LINK_MASS if actuated else ARTICULATED_LINK_MASS) / 10.0),
        # only DOF-carrying bodies get real mass when actuated: rai has ~69 frames per arm, so
        # massing all of them built a 276kg arm and the servos diverged (3.65 -> 0.020 rad).
        static_mass=(1e-4 if actuated else None),
        animate_grippers=bool(animate_grippers))
    # FR3, OpenArm and Husky mount ordinary aN_ arm trees on separately-prefixed fixed carriers.
    # They are decoration (no contacts or actuators), but belong in the visual model and in
    # recorded NPZ XML. Export every discovered carrier from the same rai tree as the arms.
    _base_xml = []
    for _prefix in arm_model.carrier_base_prefixes(articulated_C):
      _xml, _ = arm_model.build_arm_mjcf(
          articulated_C, _prefix, collide=False, group=GROUP_VISUAL,
          meshes=_arm_meshes, assets=_arm_assets, mesh_group=GROUP_VISUAL,
          link_mass=1e-4, link_inertia=1e-5, static_mass=1e-4,
          limits=False, frame_colors=True)
      _base_xml.append(_xml)
    if _base_xml:
      body_xml = "\n".join(_base_xml + [body_xml])
    parts_xml = [x for x in parts_xml if 'name="arm_' not in x]      # drop the mocap mirrors
    parts_xml.append(body_xml)
    anchor = {r: ee_anchor[r] for r in robots}
    arm_mesh_assets = "".join(_arm_assets)
  _actuators = []
  if articulated_C is not None and actuated:
    a, _ = arm_model.actuator_mjcf(articulated_C, articulated_joints, kp=ACTUATED_KP)
    _actuators.append(a)
  if articulated_C is not None and animate_grippers:
    a, _ = arm_model.gripper_actuator_mjcf(articulated_C, robots)
    if a:
      _actuators.append(a)
  _actuator_body = "\n".join(_actuators)
  act_xml = ("" if not _actuators else
             f"\n  <actuator>\n{_actuator_body}\n  </actuator>")
  # GRASP INTERFACE COMPLIANCE. The 4ms default is near-rigid, chosen because a MOCAP anchor
  # sagged 7-8mm under carry loads. Under --actuated the anchor is the real link, and a rigid
  # weld converts the tracking DIFFERENCE between two arms co-grasping one part into force
  # through the part: measured 119.7N against a 25N cap on a3+a1 both holding obj_6, where the
  # kinematic (--articulated) run places both arms exactly and sees none of it. A real pinch has
  # finite stiffness; peak |F| = m|dv| / (2*timeconst), so softening trades sag for force.
  _wtc = float(weld_timeconst) if weld_timeconst else WELD_TIMECONST
  for r in robots:
    # grasp welds: anchor body <-> each part (inactive; relpose set at grasp time).
    for p in assembly.objects:
      # stiff solref: the anchor is a mocap body, so a soft (default) weld SAGS mm-scale
      # under carry loads -- measured 7-8mm drift, enough to miss the seat tolerance.
      eq_xml.append(f'<weld name="grasp_{r}_{p}" body1="{anchor[r]}" body2="{p}" '
                    f'active="false" solref="{_wtc} 1"/>')
  for p in assembly.objects:                               # mate welds along the manual's DAG
    rel = assembly.relative_pose(p)
    if rel is not None:
      eq_xml.append(f'<weld name="mate_{p}" body1="{rel[0]}" body2="{p}" active="false" '
                    f'solref="{_wtc} 1"/>')
  # Preserve the planning scene's real slab thickness. Only its z centre follows the selected
  # top height (normally the lowest floated part bottom; exactly the planner top under
  # --planner-table). Stretching the box down to the floor made raised FR3 workbenches look
  # like giant solid pedestals and introduced side/underside geometry absent from planning.
  table_z = table_top - table_half_z
  return f"""
<mujoco model="assembly_validation">
  <!-- MuJoCo's compiler defaults to angle="degree", so rai's joint ranges (radians) were being
       read as degrees and shrunk 57x: shoulder_lift [-3.28, 0.283] became [-0.057, 0.005].
       Every joint then sat ~2 rad outside its "limit" and MuJoCo answered with 23550 Nm, which
       is what threw the actuated arm (32 rad/s) and what zeroed the grasp weld's force in the
       kinematic mode. Body poses are quaternions, so only ranges were affected. -->
  <compiler angle="radian"/>
  <!-- A flat native cylinder against a box otherwise gets only one contact in MuJoCo and
       slowly sinks through the table (the car-mini wheel rim ended up 13mm below the top).
       multiccd asks the convex collider for the full contact patch: the wheel stays a single
       cylinder for both collision and rendering, with no hidden box proxy. -->
  <option timestep="{DT}" integrator="implicitfast">
    <flag multiccd="enable"/>
  </option>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.25 0.3"
             rgb2="0.25 0.3 0.35" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.7 0.9" rgb2="0.9 0.9 0.95"
             width="64" height="64"/>
    {part_mesh_assets}{visual_mesh_assets}{arm_mesh_assets}
  </asset>
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/></visual>
  <worldbody>
    <light pos="0.5 -0.5 2.5" dir="-0.2 0.2 -1" directional="true" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid" pos="0 0 0"
          group="{GROUP_ENV}"/>
    <!-- condim=6 is required for MuJoCo to use the third (rolling) friction coefficient.
         Without it, horizontally scattered chair legs/posts are ideal cylinders with no
         rolling resistance: micron-scale contact noise makes them roll centimetres before
         their pick. 2mm is modest relative to their 12.5mm radius and models a real bench's
         finite rolling resistance without pinning or otherwise special-casing loose parts. -->
    <geom name="table" type="box" size="{table_half_x} {table_half_y} {table_half_z}"
          pos="{table_x} {table_y} {table_z}" rgba="0.55 0.42 0.28 1"
          friction="{_contact_friction(mu_table, table=True)}" condim="6"
          group="{GROUP_ENV}"/>
    {''.join(parts_xml)}
  </worldbody>
  <contact>{''.join(contact_xml)}</contact>
  <equality>{''.join(eq_xml)}</equality>{act_xml}
</mujoco>"""


# ---------------------------------------------------------------------------------------------
# the sim world: weld managers + stepping + breakaway
# ---------------------------------------------------------------------------------------------
class SimScene:
  def __init__(self, assembly, state, robots, gripper_kinds=None, gui=False,
               arms=None, arm_poses=None, ee_anchor=None, articulated_C=None,
               actuated=False, planner_table_top=None, weld_timeconst=None,
               table_geometry=None, table_mu=None, process_engagement_distance=0.01,
               process_wrench_ramp=0.05, animate_grippers=False, object_damping=0.0):
    if not HAVE_MUJOCO:
      raise RuntimeError("mujoco not installed (pip install mujoco)")
    self.assembly = assembly
    self.table_mu = 0.5 if table_mu is None else float(table_mu)
    self.xml = build_mjcf(assembly, state, robots, mu_table=self.table_mu,
                          arms=arms, arm_poses=arm_poses,
                          planner_table_top=planner_table_top, weld_timeconst=weld_timeconst,
                          table_geometry=table_geometry,
                          ee_anchor=ee_anchor, articulated_C=articulated_C,
                          actuated=actuated, animate_grippers=animate_grippers,
                          object_damping=object_damping)
    self.model = mujoco.MjModel.from_xml_string(self.xml)
    self.data = mujoco.MjData(self.model)
    self.robots = list(robots)
    self.arms = arms                                       # [(frame, kind, size)]
    self.articulated = articulated_C is not None
    self.actuated = bool(actuated) and self.articulated
    self._cmd_ref = None                                   # last commanded q (tracking metric)
    self._cmd_qdot = None                                  # ... and its velocity feedforward
    self._phys = 0                                         # bumped by mj_step / mj_forward
    self._eq_cache = (None, None, None)                    # (eq_active key, rows, ids)
    self.verify_efc_cache = False                          # --verify-efc-cache
    self._mocap_ids = self._mocap_fkf = None               # resolved on first sync_arms
    self.grasp_anchor = ({r: f"ee_{r}" for r in robots} if arms is None
                         else {r: (ee_anchor[r] if self.articulated else f"arm_{ee_anchor[r]}")
                               for r in robots})
    self.recorder = None                                   # set by the executor (see record=)
    self._rec_count = 0
    self._over = {}                                        # robot -> consecutive overload steps
    self._grasp_trace = int(os.environ.get("SIM_GRASP_TRACE", "0"))   # weld-slip diagnostic
    self.wrench_trace = None                               # SIM_WRENCH_TRACE hook, see
    self.trace_ctx = None                                  # _make_wrench_tracer (driver's
    self.step_i = 0                                        # sample index + the sim clock)
    # Optional declared process loads. Each entry is (inserted part, receiving part). The
    # executor activates them only over the final entry->seat stretch of an assemble action;
    # ``step`` recomputes their world-frame forces from the current body poses every tick.
    self.process_wrenches = []
    self._process_wrench_bodies = set()
    self._process_wrench_ticks = 0
    self.process_engagement_distance = float(process_engagement_distance)
    self.process_wrench_ramp = float(process_wrench_ramp)
    # per-robot geom ids of the rai contact=-2 wrist capsules (un-collided while holding)
    self._holdex = {r: [] for r in self.robots}
    if arms is not None:
      for body, fkf, _geom, collide, exempt, _grp, _rgba in arms:
        if collide and exempt:
          r = next(rb for rb in self.robots if body.startswith(rb))
          nm = body if self.articulated else f"arm_{body}"
          try:
            bid = self._body_id(nm)
          except Exception:
            continue                                       # visual-only frame: no body here
          self._holdex[r].append(int(self.model.body_geomadr[bid]))
    # ARTICULATED: qpos index per rai joint name, so a rai joint vector maps onto MuJoCo's
    # own ordering rather than by position (they coincide today; relying on that silently is
    # exactly the class of bug fk_compare exists to catch).
    # NB qpos and qvel are DIFFERENT address spaces -- a free joint occupies 7 qpos but only
    # 6 qvel -- so a qpos address indexes qvel out of bounds once free-floating parts precede
    # the arms (measured: IndexError at 48 with qvel size 48). Keep both.
    self._qadr, self._vadr, self._ctrl_adr = {}, {}, {}
    self._rai_joints = []
    self._ctrl_i = self._ctrl_j = self._vadr_i = self._qadr_i = self._qadr_j = \
        np.zeros(0, dtype=int)
    self._rai_hinge = np.zeros(0, dtype=bool)
    self._ctrl_ff = np.zeros(0, dtype=float)
    if self.articulated:
      for jn in articulated_C.getJointNames():
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
          self._qadr[jn] = int(self.model.jnt_qposadr[jid])
          self._vadr[jn] = int(self.model.jnt_dofadr[jid])
      self._rai_joints = list(articulated_C.getJointNames())
      self._rai_hinge = np.zeros(len(self._rai_joints), dtype=bool)
      for _i, _jn in enumerate(self._rai_joints):
        _jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, _jn)
        if _jid >= 0 and self.model.jnt_type[_jid] == mujoco.mjtJoint.mjJNT_HINGE:
          self._rai_hinge[_i] = True
      self._ctrl_adr = {}
      for jn in self._rai_joints:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
        if aid >= 0:
          self._ctrl_adr[jn] = int(aid)
    # PRECOMPUTED INDEX ARRAYS for the per-step hot paths (command_arm_q / gravity_compensate /
    # tracking_error). The model is immutable after compile, so these never change; building
    # them once turns three per-joint Python loops per step into single vectorised assignments.
    # `_ctrl_j` maps into the COMMANDED q vector (rai joint order), `_ctrl_i` into data.ctrl.
    _ci, _cj, _cff = [], [], []
    for _i, _jn in enumerate(self._rai_joints):
      _adr = self._ctrl_adr.get(_jn) if self.articulated else None
      if _adr is None:
        continue
      _kp = float(self.model.actuator_gainprm[_adr, 0])
      _kv = -float(self.model.actuator_biasprm[_adr, 2])
      _ci.append(_adr)
      _cj.append(_i)
      _cff.append((_kv / _kp) if _kp > 0 else 0.0)
    self._ctrl_i = np.asarray(_ci, dtype=int)
    self._ctrl_j = np.asarray(_cj, dtype=int)
    self._ctrl_ff = np.asarray(_cff, dtype=float)
    self._vadr_i = np.asarray(sorted(self._vadr.values()), dtype=int)
    _qi = [(self._qadr[_jn], _i) for _i, _jn in enumerate(self._rai_joints)
           if _jn in self._qadr]
    self._qadr_i = np.asarray([a for a, _ in _qi], dtype=int)
    self._qadr_j = np.asarray([j for _, j in _qi], dtype=int)
    self._hinge_i = self._rai_hinge[[_j for _, _j in _qi]]
    self.gripper_kinds = gripper_kinds or {}
    self.animate_grippers = bool(animate_grippers)
    self._gripper_qadr = {r: [] for r in self.robots}
    self._gripper_vadr = {r: [] for r in self.robots}
    self._gripper_ctrl = {r: [] for r in self.robots}
    self._gripper_gain = {r: [] for r in self.robots}
    self._gripper_angle = {r: [] for r in self.robots}
    self._gripper_visual_geoms = {r: [[], []] for r in self.robots}
    self._gripper_aperture = {r: 0.0 for r in self.robots}
    self._gripper_motion = {}
    self._gripper_model = {r: "robotiq" for r in self.robots}
    if self.animate_grippers:
      for _jid in range(self.model.njnt):
        _nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, _jid)
        if not _nm or not _nm.startswith("sim_grip__"):
          continue
        _frame = _nm[len("sim_grip__"):]
        _owner = next((r for r in self.robots if _frame.startswith(r + "_")), None)
        if _owner is None:
          continue
        self._gripper_qadr[_owner].append(int(self.model.jnt_qposadr[_jid]))
        self._gripper_vadr[_owner].append(int(self.model.jnt_dofadr[_jid]))
        _aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, _nm)
        if _aid < 0:
          raise RuntimeError(f"animated gripper joint {_nm} has no position actuator")
        self._gripper_ctrl[_owner].append(int(_aid))
        _user = np.asarray(self.model.jnt_user[_jid], dtype=float)
        self._gripper_gain[_owner].append(float(_user[0]))
        self._gripper_angle[_owner].append(float(_user[1]))
        _side = int(round(float(_user[2])))
        _joint_body = int(self.model.jnt_bodyid[_jid])
        # The real finger mesh may be several rigid helper frames below its hinge (especially
        # Robotiq's four-bar linkage). Collect visual descendant geoms, not just direct ones.
        for _body in range(1, self.model.nbody):
          _p = _body
          while _p > 0 and _p != _joint_body:
            _p = int(self.model.body_parentid[_p])
          if _p != _joint_body:
            continue
          _ga = int(self.model.body_geomadr[_body])
          _gn = int(self.model.body_geomnum[_body])
          self._gripper_visual_geoms[_owner][_side].extend(
              g for g in range(_ga, _ga + _gn)
              if int(self.model.geom_group[g]) == GROUP_VISUAL)
        if "openarm" in _frame:
          self._gripper_model[_owner] = "openarm"
      for _r, _sides in self._gripper_visual_geoms.items():
        self._gripper_visual_geoms[_r] = [sorted(set(x)) for x in _sides]
      missing = [r for r in self.robots
                 if self.gripper_kinds.get(r, "vacuum") != "vacuum"
                 and not self._gripper_qadr[r]]
      if missing:
        raise RuntimeError("gripper animation requested, but no visual finger meshes were "
                           f"exported for {missing}; keep arm meshes enabled")
    self.active_grasps = {}                                # robot -> (eq_id, part, fcap, tcap)
    self._hold_info = {}                                   # robot -> grasp-frame info (directional caps)
    # FINITE MATE STRENGTH (--mate-force-cap/--mate-torque-cap): mate welds get the same
    # breakaway treatment as grasp welds. None = infinite strength = the historical model
    # (monitoring fully off, byte-identical). The map is built once from the pre-declared
    # mate_<part> equalities; parent comes from the weld's own body1, not the assembly DAG.
    self.mate_caps = None                                  # (F N, T Nm) scalar caps, or None
    self._mate_over = {}                                   # part -> consecutive overload steps
    self._mate_max = {}                                    # part -> (max |F|, max |T|) observed
    self._mate_eqs = {}                                    # part -> (eq id, parent body name)
    for _eq in range(self.model.neq):
      _nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, _eq)
      if _nm and _nm.startswith("mate_"):
        _par = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(self.model.eq_obj1id[_eq]))
        self._mate_eqs[_nm[len("mate_"):]] = (_eq, _par)
    # CONTACT/WELD CONSISTENCY. A fitted CAD mate can have sub-millimetre numerical overlap.
    # Once the pair is rigidly welded, retaining its contact constraint creates an internally
    # contradictory closed constraint (duct: 0.25mm produced a 33N impulse and slid the whole
    # assembly). Keep collision active throughout approach/insertion, suppress only this direct
    # pair while its mate equality is active, and automatically restore it if the weld breaks.
    part_root = {}
    for _part in assembly.objects:
      _bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, _part)
      if _bid >= 0:
        part_root[_part] = int(_bid)
    root_part = {root: name for name, root in part_root.items()}
    self._part_owner_by_body = [None] * self.model.nbody
    for _bid in range(self.model.nbody):
      _p = _bid
      while _p > 0:
        _owner = root_part.get(_p)
        if _owner is not None:
          self._part_owner_by_body[_bid] = _owner
          break
        _p = int(self.model.body_parentid[_p])
    self._mate_contact_eq = {}
    for _child, (_eq, _parent) in self._mate_eqs.items():
      if _child in part_root and _parent in part_root:
        self._mate_contact_eq[frozenset((_child, _parent))] = int(_eq)
    # An ideal grasp is represented by a weld, so retaining the gripper's authored contact
    # against that same payload duplicates a constraint just like mate contact does.  Keep
    # the arm and every unrelated object collidable; identify only terminal-tool bodies.
    # Name tokens cover the calibrated UR/Robotiq, OpenArm, vacuum and Husky exports while
    # deliberately excluding wrist/link collision capsules.
    _tool_tokens = ("gripper", "robotiq", "finger", "knuckle", "palm", "ftsensor",
                    "vacuum", "suction", "cup")
    self._gripper_robot_by_body = [None] * self.model.nbody
    for _bid in range(self.model.nbody):
      _nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, _bid) or ""
      for _robot in self.grasp_anchor:
        if _nm.startswith(f"{_robot}_") and any(tok in _nm.lower() for tok in _tool_tokens):
          self._gripper_robot_by_body[_bid] = _robot
          break
    import weakref
    _CONTACT_FILTER_SCENES[id(self.model)] = weakref.ref(self)
    mujoco.set_mjcb_contactfilter(_assembly_contact_filter)
    self.expect_state = None                               # set by the executor: lets attach() recover a grasp id
    # DIRECTIONAL CAPS default OFF. The test is faithful to the LP (force at the grasp point +
    # residual pure couple, matching wrench._force_gen/_couple_gen), and it is MORE PERMISSIVE
    # in force -- a compressive load gets F_push=100 instead of a blanket 25N. But it is
    # STRICTER in torque, because the scalar path compared the couple about the PART ORIGIN
    # while the LP's bound is on the residual couple at the GRASP: for a 34mm grasp offset at
    # ~6N that is ~0.2Nm, the same order as T_close=0.4 itself. Enabling it regressed two
    # measured-passing configs (car-mini actuated 17/17 -> T_close +0.4/0.4; cross-new
    # articulated 36/36 -> 29/40). That is evidence the COUPLE CAPS are wrong (they are
    # hand-set placeholders), not a reason to ship the regression. Turn on with
    # --directional-caps to work on them.
    self.directional_caps = False
    self.viewer = None
    if gui:
      from mujoco import viewer as _viewer                 # NOT `import mujoco.viewer`: that
      self.viewer = _viewer.launch_passive(self.model, self.data)  # would shadow the module-
      _default_geomgroups(self.viewer,
                          bool(getattr(assembly, "visual_meshes", {})) or
                          (arms is not None and any(e[2][0] == "mesh" for e in arms)))
    mujoco.mj_forward(self.model, self.data)
    self._phys += 1

  def _filter_active_mate_contact(self, data, geom1, geom2):
    """MuJoCo callback: one means discard this candidate contact pair."""
    body1 = int(self.model.geom_bodyid[geom1])
    body2 = int(self.model.geom_bodyid[geom2])
    part1 = self._part_owner_by_body[body1]
    part2 = self._part_owner_by_body[body2]
    tool1 = self._gripper_robot_by_body[body1]
    tool2 = self._gripper_robot_by_body[body2]
    for robot, part in ((tool1, part2), (tool2, part1)):
      if robot is None or part is None or robot not in self.active_grasps:
        continue
      held_root = self.active_grasps[robot][1]
      info = self._hold_info.get(robot) or {}
      contacted = info.get("grasp_part", held_root)
      if part == contacted:
        return 1
    if part1 is None or part2 is None or part1 == part2:
      return 0
    eq = self._mate_contact_eq.get(frozenset((part1, part2)))
    return int(eq is not None and bool(data.eq_active[eq]))

  # -- naming helpers --------------------------------------------------------------------
  def _eq_id(self, name):
    return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

  def _body_id(self, name):
    return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

  def set_process_wrenches(self, pairs):
    """Activate mate loads; entries are ``(inserted, parent[, realization])``."""
    for bid in self._process_wrench_bodies:
      self.data.xfrc_applied[bid] = 0.0
    self.process_wrenches = [tuple(p) for p in pairs]
    self._process_wrench_bodies = {
        self._body_id(name) for pair in self.process_wrenches for name in pair[:2]}
    self._process_wrench_ticks = 0

  def _apply_process_wrenches(self):
    """Write current-pose process forces into MuJoCo's body-COM wrench input."""
    if not self._process_wrench_bodies:
      return
    for bid in self._process_wrench_bodies:
      self.data.xfrc_applied[bid] = 0.0
    ramp_u = 1.0 if self.process_wrench_ramp <= 0.0 else min(
        1.0, self._process_wrench_ticks * DT / self.process_wrench_ramp)
    ramp = ramp_u * ramp_u * (3.0 - 2.0 * ramp_u)          # smoothstep: no force impulse
    for entry in self.process_wrenches:
      part, parent = entry[:2]
      realization = entry[2] if len(entry) > 2 else None
      bp, bt = self._body_id(part), self._body_id(parent)
      part_pose = np.concatenate([self.data.xpos[bp], self.data.xquat[bp]])
      parent_pose = np.concatenate([self.data.xpos[bt], self.data.xquat[bt]])
      loads = _process_wrench_pair(
          self.assembly, part, parent, part_pose, parent_pose, realization=realization)
      skill = self.assembly.screw_skill(part)
      if skill is None:
        # A press reaction exists only near contact, not throughout a potentially 50mm
        # collision-free pre-insertion approach. Smoothly reach full load at the datum.
        gap = _process_interface_gap(self.assembly, part, part_pose, parent_pose)
        if self.process_engagement_distance <= 0.0:
          engage = 1.0
        else:
          u = np.clip(1.0 - gap / self.process_engagement_distance, 0.0, 1.0)
          engage = float(u * u * (3.0 - 2.0 * u))
      else:
        # A thread is engaged throughout its certified helix; only suppress the turn-on impulse.
        engage = 1.0
      scale = ramp * engage
      for body, (force, torque, point) in loads.items():
        bid = self._body_id(body)
        # xfrc_applied is a world-frame force/couple at the body's inertial COM. Translate
        # the interface force to that point; the pure screw couple is origin-independent.
        couple = np.asarray(torque) + np.cross(
            np.asarray(point) - np.asarray(self.data.xipos[bid]), np.asarray(force))
        self.data.xfrc_applied[bid, :3] += scale * force
        self.data.xfrc_applied[bid, 3:] += scale * couple
    self._process_wrench_ticks += 1

  def part_pose(self, p):
    b = self._body_id(p)
    return np.r_[self.data.xpos[b], self.data.xquat[b]]    # wxyz -- matches problem.py

  # -- weld managers ---------------------------------------------------------------------
  def weld_violation(self, eq_id, body1, body2):
    """How far body2 actually sits from where this weld says it should, in (mm, deg).

    The direct measure of GRASP SLIP: `_activate_weld` freezes the relative pose into eq_data,
    so any later disagreement is the constraint failing to hold the part. Zero for a mocap
    anchor (it sits outside the constrained dynamics); non-zero for a jointed chain is exactly
    the hypothesis under test."""
    p1, q1 = self.data.xpos[body1], self.data.xquat[body1]
    p2, q2 = self.data.xpos[body2], self.data.xquat[body2]
    R1 = _quat_to_R(q1)
    rel_p = R1.T @ (p2 - p1)
    rel_q = _pose_compose(np.r_[0, 0, 0, _quat_conj(q1)], np.r_[0, 0, 0, q2])[3:]
    tgt_p = np.asarray(self.model.eq_data[eq_id][3:6], dtype=float)
    tgt_q = np.asarray(self.model.eq_data[eq_id][6:10], dtype=float)
    return (float(np.linalg.norm(rel_p - tgt_p)) * 1000.0,
            float(np.degrees(_quat_angle(rel_q, tgt_q))))

  def _activate_weld(self, eq_id, body1, body2):
    """Freeze the CURRENT relative pose of body2 in body1 into the weld's eq_data, then
    activate. TODO(verify): eq_data layout for mjEQ_WELD is [anchor(3), relpose(7), torquescale]
    in mujoco 3.x -- check indices on first run."""
    p1, q1 = self.data.xpos[body1], self.data.xquat[body1]
    p2, q2 = self.data.xpos[body2], self.data.xquat[body2]
    R1 = _quat_to_R(q1)
    rel_p = R1.T @ (p2 - p1)
    rel_q = _pose_compose(np.r_[0, 0, 0, _quat_conj(q1)], np.r_[0, 0, 0, q2])[3:]
    self.model.eq_data[eq_id][:3] = 0.0                    # anchor
    self.model.eq_data[eq_id][3:6] = rel_p
    self.model.eq_data[eq_id][6:10] = rel_q
    self.model.eq_data[eq_id][10] = 1.0                    # torquescale
    self.data.eq_active[eq_id] = 1

  def grasp(self, robot, part, gid=None, grasp_part=None):
    eq = self._eq_id(f"grasp_{robot}_{part}")
    self._activate_weld(eq, self._body_id(self.grasp_anchor[robot]), self._body_id(part))
    kind = self.gripper_kinds.get(robot, "vacuum")
    self.active_grasps[robot] = (eq, part) + _grasp_caps(kind)
    # DIRECTIONAL CAPS need the grasp FRAME, which is a property of the grasp id. Stored in a
    # parallel dict so active_grasps keeps its 4-tuple shape for every existing consumer. With
    # no gid we cannot build the frame and fall back to the conservative scalar.
    self._hold_info.pop(robot, None)
    if gid is None:
      return
    _model, _caps = _grasp_polytope(kind)
    if _model is None:
      return
    import problem as _P
    # mr_bench welds a gripper to the fixed ROOT of an already-mated group, while the actual
    # contact may be on a descendant.  A handover of a stool held by one leg is exactly this:
    # frames=[gripper, obj_0] but State records grasp(robot, obj_3, gid).  Reusing ``gid`` as
    # though it indexed obj_0 loses the grasp frame and makes the breakaway fall back to a
    # scalar torque cap. Express the contacted part's grasp pose in the welded root's frame.
    grasp_part = part if grasp_part is None else grasp_part
    _g7 = np.asarray(_P.grasp_pose(self.assembly, grasp_part, gid), dtype=float)
    if grasp_part != part:
      root_T_part = _pose_compose(_pose_invert(self.part_pose(part)),
                                  self.part_pose(grasp_part))
      _g7 = _pose_compose(root_T_part, _g7)
    _gp, _gq = _g7[:3], _g7[3:]
    _R = np.zeros(9)
    mujoco.mju_quat2Mat(_R, _gq)
    self._hold_info[robot] = {"eq": eq, "part": part, "grasp_part": grasp_part,
                              "gid": gid, "model": _model,
                              "caps": _caps, "gp": _gp,
                              "Rg": _R.reshape(3, 3)}
    for gid in self._holdex.get(robot, ()):                # wrist cluster exempt while holding
      self.model.geom_contype[gid] = 0
      self.model.geom_conaffinity[gid] = 0

  def tracking_error(self, q, robots=None):
    """max |q_actual - q_commanded| over the arm dofs, rad. With actuators the trajectory is a
    REFERENCE, so this is the number that decides whether a mate still lands on its witness --
    SEAT_POS_TOL is 10mm, so even ~0.01 rad at a 1m lever is already marginal. Hinge joints
    are wrapped into (-pi, pi]: q and q+2pi are the SAME pose, so a plant resting at the +
    side of a +-pi joint while the command says - is CONVERGED, not 6.28 rad off (measured:
    every settle ran its full budget because exactly this wrap defeated the test).

    `robots`: optional iterable of robot names -- measure ONLY their dofs. Fire-flow settle
    needs this: arms that are deliberately still MOVING through the corridor carry their own
    tracking residual, which must not count against the acting arms' arrival test."""
    q = np.asarray(q, dtype=float)
    if self._qadr_i.size == 0:
      return 0.0
    if robots is None:
      d = self.data.qpos[self._qadr_i] - q[self._qadr_j]
      d = np.where(self._hinge_i, (d + np.pi) % (2.0 * np.pi) - np.pi, d)
      return float(np.abs(d).max())
    mask_i, mask_j, mask_h = [], [], []
    for i, jn in enumerate(self._rai_joints):
      if jn.split("_", 1)[0] in robots:
        mask_i.append(self._qadr[jn])                      # QPOS address -- _vadr is the DOF
        mask_j.append(i)                                   # space (qvel) and indexes qpos
        mask_h.append(bool(self._rai_hinge[i]))            # wrongly once free joints precede
    if not mask_i:
      return 0.0
    d = self.data.qpos[mask_i] - q[mask_j]
    d = np.where(np.asarray(mask_h, dtype=bool),
                 (d + np.pi) % (2.0 * np.pi) - np.pi, d)
    return float(np.abs(d).max())

  def command_arm_q(self, q, qdot=None):
    """ACTUATED: hand the joint targets to the position servos and let MuJoCo track them.

    The trajectory becomes a REFERENCE rather than a command -- the arm now has tracking
    error, reacts to the payload, and is limited by rai's real effort limits. Gravity (and
    Coriolis) are cancelled by feedforward so the PD only has to reject the payload and the
    tracking error; that is what removes the need for an integral term, whose windup over a
    long move is worse than the droop it would fix.

    VELOCITY FEEDFORWARD via the SETPOINT. A `<position kp kv>` servo applies
    kp*(ctrl - q) - kv*qdot, i.e. it damps ABSOLUTE velocity and so fights the commanded
    motion, leaving a lag of kv/kp*qdot_ref (~0.03 rad at cruise -- 30mm at a 1m lever,
    against a 10mm seat tolerance). Commanding ctrl = q_ref + kv/kp*qdot_ref turns that into
    kp*(q_ref - q) + kv*(qdot_ref - qdot), the PD tracker we actually want. Doing it through
    the SETPOINT rather than qfrc_applied keeps the torque subject to rai's `forcerange`.
    kv/kp is read back from the compiled model so it cannot drift from the emitted XML."""
    q = np.asarray(q, dtype=float)
    self._cmd_ref = q
    self._cmd_qdot = None if qdot is None else np.asarray(qdot, dtype=float)
    # VECTORISED over the arm dofs. The per-joint Python loop (with two model reads each) was
    # 1.67s of a 13.7s car-mini run -- second only to mj_step. Same arithmetic, same order,
    # so the setpoints are bit-identical; only the dispatch is gone. Index arrays and kv/kp
    # are precomputed once in __init__ (the model is immutable after compile).
    if self._cmd_qdot is None:
      self.data.ctrl[self._ctrl_i] = q[self._ctrl_j]
    else:
      self.data.ctrl[self._ctrl_i] = q[self._ctrl_j] + self._ctrl_ff * self._cmd_qdot[self._ctrl_j]

  def set_gripper_aperture(self, robot, inward):
    """Instantly set the articulated hand (initialisation/contact-query/replay helper)."""
    if not self.animate_grippers or not self._gripper_qadr.get(robot):
      return
    model = self._gripper_model.get(robot, "robotiq")
    physical_max = 0.045 if model == "openarm" else 0.042
    value = float(np.clip(float(inward), 0.0, physical_max))
    fraction = value / physical_max
    targets = (np.asarray(self._gripper_gain[robot], dtype=float)
               * np.asarray(self._gripper_angle[robot], dtype=float) * fraction)
    self._gripper_aperture[robot] = value
    self.data.qpos[np.asarray(self._gripper_qadr[robot], dtype=int)] = targets
    self.data.qvel[np.asarray(self._gripper_vadr[robot], dtype=int)] = 0.0
    self.data.ctrl[np.asarray(self._gripper_ctrl[robot], dtype=int)] = targets

  def _set_gripper_target(self, robot, inward):
    """Command the real hinge actuators without teleporting their joint state."""
    model = self._gripper_model.get(robot, "robotiq")
    physical_max = 0.045 if model == "openarm" else 0.042
    value = float(np.clip(float(inward), 0.0, physical_max))
    fraction = value / physical_max
    targets = (np.asarray(self._gripper_gain[robot], dtype=float)
               * np.asarray(self._gripper_angle[robot], dtype=float) * fraction)
    self._gripper_aperture[robot] = value
    self.data.ctrl[np.asarray(self._gripper_ctrl[robot], dtype=int)] = targets

  def gripper_aperture(self, robot):
    return float(self._gripper_aperture.get(robot, 0.0))

  def command_gripper_aperture(self, robot, inward, duration=0.25):
    """Start a smooth visual transition that advances alongside the arm trajectory."""
    if not self.animate_grippers or not self._gripper_qadr.get(robot):
      return
    target = float(max(0.0, inward))
    start = self.gripper_aperture(robot)
    if abs(target - start) <= 1e-6:
      self._gripper_motion.pop(robot, None)
      self.set_gripper_aperture(robot, target)
      return
    self._gripper_motion[robot] = [start, target, 0.0, max(float(duration), DT)]

  def visual_contact_aperture(self, robot, part, lower=0.0, contact_eps=5e-4):
    """Smallest rendered aperture at which both visual fingers touch ``part``.

    The search closure certificate operates on simple finger capsules, whereas the viewer
    renders the manufacturer's meshes. Their surfaces are offset by several millimetres on
    Robotiq, so blindly copying the capsule aperture visibly stops short. This query uses
    MuJoCo's geometry-distance routine on the visual meshes only, without enabling contact or
    stepping physics, and therefore calibrates presentation without changing feasibility.
    """
    sides = self._gripper_visual_geoms.get(robot, ((), ()))
    if not all(sides):
      return float(lower)
    part_body = self._body_id(part)

    def belongs_to_part(body):
      while body > 0:
        if body == part_body:
          return True
        body = int(self.model.body_parentid[body])
      return False

    part_geoms = [g for g in range(self.model.ngeom)
                  if belongs_to_part(int(self.model.geom_bodyid[g]))]
    if not part_geoms:
      return float(lower)
    model = self._gripper_model.get(robot, "robotiq")
    upper = 0.045 if model == "openarm" else 0.042
    old = self.gripper_aperture(robot)
    qidx = np.asarray(self._gripper_qadr[robot], dtype=int)
    vidx = np.asarray(self._gripper_vadr[robot], dtype=int)
    cidx = np.asarray(self._gripper_ctrl[robot], dtype=int)
    old_q, old_v, old_ctrl = (self.data.qpos[qidx].copy(), self.data.qvel[vidx].copy(),
                              self.data.ctrl[cidx].copy())

    def gaps(aperture):
      self.set_gripper_aperture(robot, aperture)
      mujoco.mj_kinematics(self.model, self.data)
      out = []
      for side in sides:
        best = float("inf")
        for g in side:
          for h in part_geoms:
            best = min(best, float(mujoco.mj_geomDistance(
                self.model, self.data, int(g), int(h), 0.2, np.empty(6))))
        out.append(best)
      return out

    lo = float(np.clip(lower, 0.0, upper))
    found = None
    # The query occurs once per grasp event, not per tick. A 1mm bracket plus bisection is
    # cheap and avoids a robot/model-specific cosmetic offset table.
    for aperture in np.arange(lo, upper + 5e-4, 1e-3):
      aperture = min(float(aperture), upper)
      if max(gaps(aperture)) <= float(contact_eps):
        found = aperture
        break
    if found is not None:
      a, b = max(lo, found - 1e-3), found
      for _ in range(6):
        mid = 0.5 * (a + b)
        if max(gaps(mid)) <= float(contact_eps):
          b = mid
        else:
          a = mid
      found = b
    else:
      found = upper
      final_gaps = gaps(found)
      print(f"[sim/gripper] WARNING: {robot}->{part} visual meshes do not reach bilateral "
            f"contact within the {upper*1000:.0f}mm stroke (gaps "
            f"{[round(1000*x, 2) for x in final_gaps]}mm); using the mechanical limit.")
    self.data.qpos[qidx] = old_q
    self.data.qvel[vidx] = old_v
    self.data.ctrl[cidx] = old_ctrl
    self._gripper_aperture[robot] = old
    mujoco.mj_kinematics(self.model, self.data)
    return float(max(lo, found))

  def _advance_grippers(self):
    for robot, motion in list(self._gripper_motion.items()):
      start, target, elapsed, duration = motion
      elapsed = min(duration, elapsed + DT)
      u = elapsed / duration
      s = u * u * (3.0 - 2.0 * u)
      self._set_gripper_target(robot, start + s * (target - start))
      motion[2] = elapsed
      if elapsed >= duration:
        self._gripper_motion.pop(robot, None)

  def gravity_compensate(self):
    """Cancel qfrc_bias (gravity + Coriolis) on the ARM dofs only -- parts keep real gravity,
    Gravity/Coriolis is the only term applied OUTSIDE the actuator; note that qfrc_applied
    bypasses `forcerange`, so nothing speed-dependent belongs here (measured: feeding the
    velocity term in this way injected up to 93 Nm into a 54 Nm wrist and cost 3 seats).
    The velocity feedforward goes through the setpoint instead -- see command_arm_q."""
    self.data.qfrc_applied[self._vadr_i] = self.data.qfrc_bias[self._vadr_i]

  def set_arm_q(self, q, qdot=None):
    """ARTICULATED set-pose: drive the chain by writing qpos directly (and zeroing qvel), the
    kinematic stand-in for a controller. The arm is therefore perfectly stiff -- parts still
    receive contact and weld forces, the arm ignores the reaction -- which is deliberate for
    this stage: it keeps the grasp weld an exact measure of the payload's own m(a-g), the
    property the corner/wrench calibrations depend on. Swapping in actuators later changes
    only this method, not the model."""
    q = np.asarray(q, dtype=float)
    # qvel must be the COMMANDED joint velocity, not zero. A held part genuinely moves, so a
    # gripper that claims zero velocity forces the weld to reconcile the two every step --
    # which injects energy and diverges (measured: grasp force ramping 28 -> 311 -> 982 ->
    # 15989 N). mocap bodies get away with it only because they sit outside the constrained
    # dynamics entirely; a real chain does not.
    self._cmd_q = (np.asarray(q, dtype=float),
                   None if qdot is None else np.asarray(qdot, dtype=float))
    v = None if qdot is None else np.asarray(qdot, dtype=float)
    for i, jn in enumerate(self._rai_joints):
      adr = self._qadr.get(jn)
      if adr is not None:
        self.data.qpos[adr] = q[i]
        self.data.qvel[self._vadr[jn]] = 0.0 if v is None else v[i]
    mujoco.mj_forward(self.model, self.data)
    self._phys += 1

  def sync_arms(self, poses):
    """Set every arm link mocap body to its rai-FK world pose. poses = {frame: 7-pose}.

    The mocap ids are resolved ONCE and cached. Doing the f-string + mj_name2id per body per
    step was the single largest cost in the DEFAULT path: 129 name lookups x 79k steps =
    10.2M mj_name2id calls, and sync_arms alone was 63% of a 36.6s car-mini run. Nothing about
    the mapping can change after the model is compiled."""
    if self._mocap_ids is None:
      self._mocap_ids = np.asarray(
          [self.model.body_mocapid[self._body_id(f"arm_{b}")] for b, *_ in self.arms], dtype=int)
      self._mocap_fkf = [e[1] for e in self.arms]
    P = np.asarray([poses[f] for f in self._mocap_fkf], dtype=float)
    self.data.mocap_pos[self._mocap_ids] = P[:, :3]
    self.data.mocap_quat[self._mocap_ids] = P[:, 3:]

  def release(self, robot):
    g = self.active_grasps.pop(robot, None)
    if g is not None:
      self.data.eq_active[g[0]] = 0
    for gid in self._holdex.get(robot, ()):                # restore wrist collision
      self.model.geom_contype[gid] = 2
      self.model.geom_conaffinity[gid] = 1

  def weld_mate(self, part):
    """Activate the mate weld at the MANUAL's exact relative pose (not the achieved one):
    the seat CHECK verified we are inside the funnel tolerance; the weld then models the
    mate as a self-aligning datum (snap-fit semantics -- matches the planner's exact rigid
    welds and stops per-mate seat error from ACCUMULATING; measured 9.7deg tilt after 3
    welded-as-achieved mates on cross-5).

    THE MATE WELD REPLACES THE INSERTER'S GRASP WELD (mr_bench attach semantics: at the
    assemble transition the part re-parents onto the assembly and the gripper is
    kinematically free -- the post-assemble motion is PLANNED with the part bound to the
    structure, not the arm, so keeping the grasp weld makes the wandering goto segment a
    tug-of-war: 191N stiff / 72N with a compliance window, both measured. Any KEEP-HOLDING
    the plan needs is expressed by the path's pinned joint goals, exactly as in the export
    -- the known mr_bench B2c gap, same modeling on both sides)."""
    for r, g in list(self.active_grasps.items()):
      if g[1] == part:
        self.data.eq_active[g[0]] = 0                      # gripper lets the datum carry it
        del self.active_grasps[r]
    rel = self.assembly.relative_pose(part)
    eq = self._eq_id(f"mate_{part}")
    self.model.eq_data[eq][:3] = 0.0                       # anchor
    self.model.eq_data[eq][3:6] = np.asarray(rel[1], dtype=float)
    parent = rel[0]
    # The search keeps symmetry quotienting out of the symbolic state.  Recover the concrete
    # finite member from the bodies at the certified seat and snap to that declared datum,
    # rather than erasing it with the historical identity weld.
    qp = np.asarray(self.data.xquat[self._body_id(parent)], dtype=float)
    qc = np.asarray(self.data.xquat[self._body_id(part)], dtype=float)
    achieved = _quat_mul(_quat_conj(qp), qc)
    chosen = self.assembly.closest_mate_quaternion(part, achieved)
    self.model.eq_data[eq][6:10] = np.asarray(chosen, dtype=float)
    self.model.eq_data[eq][10] = 1.0                       # torquescale
    self.data.eq_active[eq] = 1

  # -- frame-based attach (mirrors mr_bench set_to_mode) ---------------------------------
  def _detach(self, child):
    """Remove whatever weld currently parents `child` -- its gripper grasp OR its mate."""
    for r, g in list(self.active_grasps.items()):
      if g[1] == child:
        self.release(r)                                    # deactivates + restores wrist coll
    meq = self._eq_id(f"mate_{child}")
    if meq >= 0 and self.data.eq_active[meq]:
      self.data.eq_active[meq] = 0

  def attach(self, parent_frame, child):
    """Replay ONE mr_bench task attach (frames = [parent, child]) as a weld, EXACTLY as
    set_to_mode does. This is the correct event model for a saved motion PATH: the path was
    planned under these attaches, so the sim must mirror them or the arm drags the part.
      parent == 'table'         -> child (its group) rests FREE (all its grippers let go).
      parent == a gripper frame -> weld child to that arm (breakaway-capped grasp).
      parent == a part          -> weld child to that part at the MANUAL datum (mate/snap-fit).
    Re-rooting (release -> [table|other-gripper, obj]) and hold-through transfers fall out of
    the frames automatically -- no action-kind inference, no continuing-holder patch."""
    self._detach(child)
    if parent_frame == "table":
      return                                               # free body -> gravity + contact
    if parent_frame in self.assembly.objects:              # part-part MATE at the datum
      self.weld_mate(child)
    else:                                                  # gripper GRASP (frozen relpose)
      # recover the GRASP ID so the breakaway can use the planner's directional caps: the
      # frame-based replay only carries [parent, child], but the discrete state knows which
      # grasp of `child` is held. Falls back to the conservative scalar if unavailable.
      _gid = None
      _grasp_part = None
      if self.expect_state is not None:
        try:
          _grasp_part, _gid = _held_grasp_id(
              self.expect_state, child, robot=parent_frame.split("_", 1)[0])
        except ValueError:
          _gid = None
      self.grasp(parent_frame.split("_", 1)[0], child, gid=_gid,
                 grasp_part=_grasp_part)

  def net_contact_force(self, part):
    """Net world-frame CONTACT force ON `part` (N). This is the admittance trigger for
    --skill-servo: it points AWAY from whatever is pressing (the direction to yield), unlike
    the grasp-weld wrench, whose efc rows live in the weld frame. mj_contactForce returns the
    wrench in the CONTACT frame (contact.frame rows are the axes, normal first); MuJoCo's
    convention has the positive normal force acting on geom2, so the sign flips when `part`
    is geom1. NOTE the sign convention was VALIDATED EMPIRICALLY (a flip makes the admittance
    press harder instead of yielding, which the seeded A/B shows immediately as rising force)."""
    bid = self._body_id(part)
    # EXCLUDE partners welded to `part` (active mate/grasp eqs): intra-assembly contact is not
    # "the environment pressing". Counting it made the admittance yield a holder away from its
    # OWN subassembly during a two-arm hold -- the controller itself built the 1.28Nm that
    # broke the grasp (measured, seed 2 online: obj_0~obj_3, mated pair, both holders lost).
    welded = set()
    for e in range(int(self.model.neq)):
      if not self.data.eq_active[e] or self.model.eq_type[e] != mujoco.mjtEq.mjEQ_WELD:
        continue
      b1, b2 = int(self.model.eq_obj1id[e]), int(self.model.eq_obj2id[e])
      if b1 == bid:
        welded.add(b2)
      elif b2 == bid:
        welded.add(b1)
    F = np.zeros(3)
    for i in range(int(self.data.ncon)):
      c = self.data.contact[i]
      b1 = int(self.model.geom_bodyid[c.geom1])
      b2 = int(self.model.geom_bodyid[c.geom2])
      if bid != b1 and bid != b2:
        continue
      other = b2 if b1 == bid else b1
      if other in welded:
        continue
      f6 = np.zeros(6)
      mujoco.mj_contactForce(self.model, self.data, i, f6)
      fw = np.asarray(c.frame, dtype=float).reshape(3, 3).T @ f6[:3]
      F += fw if b2 == bid else -fw
    return F

  def part_pair_contact_force(self, part_a, part_b):
    """Total normal force across the physical interface between two rigid parts.

    Imported CAD parts are compound bodies, so testing only their root body ids silently
    misses contacts on child collision components.  Resolve every contact through the same
    part-owner table as the mate contact filter.  This is used only as a terminal skill guard:
    it does not change collision masks or make a distant pose eligible for mating.
    """
    wanted = frozenset((part_a, part_b))
    force = 0.0
    for i in range(int(self.data.ncon)):
      c = self.data.contact[i]
      b1 = int(self.model.geom_bodyid[c.geom1])
      b2 = int(self.model.geom_bodyid[c.geom2])
      p1 = self._part_owner_by_body[b1]
      p2 = self._part_owner_by_body[b2]
      if p1 is None or p2 is None or frozenset((p1, p2)) != wanted:
        continue
      f6 = np.zeros(6)
      mujoco.mj_contactForce(self.model, self.data, i, f6)
      force += abs(float(f6[0]))
    return force

  def branch_contact_force(self, robot):
    """Net world-frame contact force ON `robot`'s WHOLE branch -- arm links + gripper +
    held cargo -- from the environment. The --compliant-follow admittance trigger:
    net_contact_force(part) only sees a HELD part, so a press on the bare gripper (a pick
    approach before the weld exists) or on an arm LINK never yields. Same sign convention
    and the same welded-partner exclusion as net_contact_force (intra-assembly contact is
    not the environment pressing); intra-branch contacts cancel and are skipped."""
    if not hasattr(self, "_branch_bodies"):
      self._branch_bodies = {}
      for r in self.grasp_anchor:
        pre = f"{r}_"
        self._branch_bodies[r] = frozenset(
            b for b in range(int(self.model.nbody))
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                ).startswith(pre))
    branch = set(self._branch_bodies.get(robot, ()))
    g = self.active_grasps.get(robot)
    welded = set()
    if g is not None:
      cid = self._body_id(g[1])
      branch.add(cid)
      for e in range(int(self.model.neq)):
        if not self.data.eq_active[e] or self.model.eq_type[e] != mujoco.mjtEq.mjEQ_WELD:
          continue
        b1, b2 = int(self.model.eq_obj1id[e]), int(self.model.eq_obj2id[e])
        if b1 == cid:
          welded.add(b2)
        elif b2 == cid:
          welded.add(b1)
    F = np.zeros(3)
    for i in range(int(self.data.ncon)):
      c = self.data.contact[i]
      b1 = int(self.model.geom_bodyid[c.geom1])
      b2 = int(self.model.geom_bodyid[c.geom2])
      in1, in2 = b1 in branch, b2 in branch
      if in1 == in2:                                       # intra-branch or foreign: skip
        continue
      if (b1 if in2 else b2) in welded:
        continue
      f6 = np.zeros(6)
      mujoco.mj_contactForce(self.model, self.data, i, f6)
      fw = np.asarray(c.frame, dtype=float).reshape(3, 3).T @ f6[:3]
      F += fw if in2 else -fw
    return F

  # -- stepping with grasp breakaway -----------------------------------------------------
  def _eq_index(self):
    """Row indices of the EQUALITY constraints, and their eq ids, for the CURRENT state.

    Hoisted out of _weld_wrench, which used to scan the two full efc arrays per ACTIVE GRASP per
    step (125k calls / 2.5s of a 13.7s car-mini run, the largest cost after mj_step itself).

    CACHED ON eq_active, NOT on the step. MuJoCo lays the EQUALITY block at the head of efc and
    gives every active weld exactly 6 rows in eq order, so which rows belong to which eq id is a
    function of the ACTIVE SET ALONE -- contacts and limits are appended after it and cannot
    shift it. So this only needs rebuilding when a grasp activates or releases (a handful of
    times per run) rather than once per step. `--verify-efc-cache` re-derives it every step and
    asserts equality, which is how the assumption above is checked rather than trusted."""
    # KEY ON (eq_active, ne), NOT eq_active ALONE. step() breaks a grasp by writing
    # eq_active[eq]=0 and then reads the NEXT grasp's wrench in the SAME loop iteration --
    # before any mj_step/mj_forward has rebuilt efc. Keyed on the bytes alone that rebuilt the
    # cache from STALE efc rows and stored it under the new key, so it was never corrected:
    # measured IndexError (index 6 into a size-6 efc_force) on the next step with no contacts,
    # and -- worse -- with contacts padding efc, no crash at all, just a permanently wrong
    # reading (2.3x understated), which silently defeats every breakaway and force check.
    # d.ne still holds the PRE-write row count at that moment and drops after the step, so it
    # distinguishes the stale read; the bytes still catch an activate+release that keeps ne.
    key = (self.data.eq_active.tobytes(), int(self.data.ne))
    if self._eq_cache[0] != key:
      idx = np.flatnonzero(self.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
      self._eq_cache = (key, idx, self.data.efc_id[idx])
    elif self.verify_efc_cache:
      _i = np.flatnonzero(self.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
      if not (np.array_equal(_i, self._eq_cache[1])
              and np.array_equal(self.data.efc_id[_i], self._eq_cache[2])):
        raise AssertionError(
            f"[efc-cache] equality-row layout changed with the cache key unchanged at step "
            f"{self.step_i}: "
            f"cached rows {self._eq_cache[1]} vs live {_i}")
    return self._eq_cache[1], self._eq_cache[2]

  def _grasp_violation(self, h):
    """Directional breakaway test in the GRASP's own frame -- the planner's box, not a scalar.

    FRAMES (measured, not assumed -- see the ground-truth probes):
      * efc_force[0:3] is a force in the WORLD frame; the wrench ON THE PART is its negative.
      * eq_data[:3] is 0 here, so it acts at the PART BODY ORIGIN.
      * efc_force[3:6] is a PURE COUPLE in the PART's frame, scaled by 2/torquescale.
    Doing the algebra in the PART frame means no gripper-anchor offset table is needed and it is
    valid in every drive mode (mocap / articulated / actuated), since R^T(Ra x Rb) = a x b.
    The grasp frame's columns are [t1, t2, n] with n pointing FROM the gripper INTO the part, so
    a POSITIVE axial component is the part pressed into the palm/cup (F_push) and a negative one
    is pull-off (F_pull). Returns (worst_ratio, label, |F|, |T|); ratio > 1 means broken."""
    rows = self._eq_index()[0][self._eq_index()[1] == h["eq"]]
    if len(rows) < 6:
      return 0.0, "", 0.0, 0.0
    f = self.data.efc_force[rows]
    ts = float(self.model.eq_data[h["eq"]][10])
    bid = self._body_id(h["part"])
    Rp = self.data.xmat[bid].reshape(3, 3)
    F_p = Rp.T @ (-f[:3])                                  # world force -> part frame
    T_p = (-0.5 * ts) * f[3:6]                             # couple about the part ORIGIN
    T_gp = T_p - np.cross(h["gp"], F_p)                    # ... moved to the GRASP POINT
    Fg, Tg = h["Rg"].T @ F_p, h["Rg"].T @ T_gp             # -> [t1, t2, n]
    c, name = h["caps"], h["model"]
    worst, lbl = 0.0, ""

    def _hit(val, cap, tag):
      nonlocal worst, lbl
      if cap and cap > 0 and abs(val) / cap > worst:
        worst, lbl = abs(val) / cap, f"{tag} {val:+.1f}/{cap:g}"

    if name == "asymmetric":
      _hit(Fg[2], c["F_push"] if Fg[2] > 0 else c["F_pull"],
           "axial-push" if Fg[2] > 0 else "axial-PULL")
      lat = c["mu"] * c["F_pull"]
      _hit(Fg[0], lat, "lat1"); _hit(Fg[1], lat, "lat2")
      _hit(Tg[2], c.get("T_axis"), "T_axis")
      _hit(Tg[0], c.get("T_lat"), "T_lat")
      _hit(Tg[1], c.get("T_close", c.get("T_lat")), "T_close")
    elif name == "directional":
      _hit(Fg[2], c["F_axial"], "axial")
      lat = c["mu"] * c["F_axial"]
      _hit(Fg[0], lat, "lat1"); _hit(Fg[1], lat, "lat2")
      _hit(Tg[2], c.get("T_axis"), "T_axis")
      _hit(Tg[0], c.get("T_lat"), "T_lat"); _hit(Tg[1], c.get("T_lat"), "T_lat")
    else:                                                  # isotropic: magnitude bounds
      _hit(float(np.linalg.norm(F_p)), c.get("F_max"), "F")
      _hit(float(np.linalg.norm(T_gp)), c.get("T_max"), "T")
    return (worst, lbl, float(np.linalg.norm(F_p)),
            float(np.linalg.norm(T_gp)))

  def _contact_context(self, r, part):
    """WHAT ELSE IS TOUCHING at the moment a grasp breaks.

    A grasp load is a REACTION: a large force with no torque is something pushing back on the
    gripper, not the part being flung. Naming the contacts distinguishes "the arm pressed the
    held part into the table" from "the part is restrained by another body" -- which the
    magnitude alone cannot, and which sent this diagnosis down two wrong paths.
    """
    try:
      import mujoco
      names, tot = {}, {}
      for i in range(int(self.data.ncon)):
        c = self.data.contact[i]
        b1 = self.model.body(self.model.geom_bodyid[c.geom1]).name
        b2 = self.model.body(self.model.geom_bodyid[c.geom2]).name
        # ``table`` and ``floor`` are worldbody geoms in the generated MJCF, so reporting
        # only their body name used to collapse both to the unhelpful label ``world``.  Keep
        # the logical body names for aggregation, but retain the concrete geom pair in the
        # diagnostic.  This is especially important for imported CAD: it distinguishes an
        # actually low insertion from a collision with some unrelated workcell fixture.
        g1 = self.model.geom(int(c.geom1)).name or f"geom{int(c.geom1)}"
        g2 = self.model.geom(int(c.geom2)).name or f"geom{int(c.geom2)}"
        if not (b1.startswith(r) or b2.startswith(r) or part in (b1, b2)):
          continue
        f = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, i, f)
        k = tuple(sorted((b1, b2)))
        tot[k] = tot.get(k, 0.0) + float(np.linalg.norm(f[:3]))
        names[k] = f"{g1}~{g2} {c.dist * 1000:+.2f}mm"
      hot = sorted(tot.items(), key=lambda kv: -kv[1])[:3]
      if not hot:
        return "  [no contacts on the gripper or the part -- the load is NOT a reaction]"
      return "  contacts: " + ", ".join(
          f"{a}~{b} {v:.1f}N @[{names[(a, b)]}]" for (a, b), v in hot)
    except Exception as e:
      return f"  [contact probe failed: {type(e).__name__}]"

  def _weld_wrench(self, eq_id):
    """Constraint wrench currently exerted by an active weld: sum its efc rows.
    TODO(verify): efc bookkeeping -- rows with data.efc_type == mjCNSTR_EQUALITY and
    data.efc_id == eq_id; a weld contributes 6 rows (3 force + 3 torque)."""
    idx, ids = self._eq_index()
    rows = idx[ids == eq_id]
    if len(rows) < 6:
      return 0.0, 0.0
    f = self.data.efc_force[rows]
    # THE TORQUE ROWS ARE NOT N.m. For mjEQ_WELD, efc_force[3:6] is a couple scaled by
    # 2/torquescale, so comparing the raw rows to the planner's caps DOUBLED every torque --
    # measured against ground truth (2kg at a 0.25m COM offset => true couple 4.9050 N.m,
    # raw rows 9.8094, ratio 1.9999; 0.5*ts*raw = 4.9047). That made the sim reject grasps the
    # planner considers fine: car-mini's only actuated-corridor failure was |T|=0.56 against a
    # 0.4 cap, i.e. 0.28 N.m in reality. torquescale is read per-equality rather than assumed.
    # explicit sqrt of the dot: np.linalg.norm on a 3-vector is dominated by its own dispatch
    # (251k calls, 0.66s) and this is the innermost per-step measurement in the executor.
    ts = 0.5 * float(self.model.eq_data[eq_id][10])
    return (_sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]),
            ts * _sqrt(f[3] * f[3] + f[4] * f[4] + f[5] * f[5]))

  def step(self, n=1, report=None, action=None):
    """Advance; after each step check every active grasp weld against its breakaway caps --
    an overloaded grasp DROPS (deactivates) and is recorded as the LP-predicted failure."""
    sustain = max(1, int(BREAK_SUSTAIN / DT))
    for _ in range(n):
      self._advance_grippers()
      self._apply_process_wrenches()
      mujoco.mj_step(self.model, self.data)
      self._phys += 1
      if self._grasp_trace and self.step_i % self._grasp_trace == 0:
        for r, g in sorted(self.active_grasps.items()):
          eq, part = g[0], g[1]
          mm, deg = self.weld_violation(eq, self._body_id(self.grasp_anchor[r]),
                                        self._body_id(part))
          f, t = self._weld_wrench(eq)
          if self.actuated and self._cmd_ref is not None:
            # ALSO the commanded speed: lag is VELOCITY-dependent and systematic (cancellable by
            # feedforward), whereas a gravity/inertia error is not. Logging both makes the two
            # separable instead of guessing from the deviation profile alone.
            _qd = (0.0 if self._cmd_qdot is None
                   else float(np.abs(np.asarray(self._cmd_qdot, dtype=float)).max()))
            trk = (f" trk={self.tracking_error(self._cmd_ref):7.4f}rad"
                   f" qdot={_qd:7.4f}rps")
          elif self._mocap_ids is not None:
            # ARTICULATED/MOCAP: the arm bodies are PLACED, so any gap between the commanded
            # mocap pose and where the body actually is would be a solver artifact, not control
            # lag. Reported in the same trace so the two modes are comparable -- "articulated is
            # exact" should be measured, not assumed.
            _bid = [self._body_id(f"arm_{b}") for b, *_ in self.arms]
            _d = np.abs(np.asarray(self.data.xpos[_bid], dtype=float)
                        - np.asarray(self.data.mocap_pos[self._mocap_ids], dtype=float))
            trk = f" mocapdev={float(_d.max()) * 1000.0:7.4f}mm"
          else:
            trk = ""
          print(f"[grasp] k={self.step_i:7d} {r}/{part:6s} slip {mm:8.3f}mm {deg:7.3f}deg "
                f"|F|={f:8.2f} |T|={t:6.3f}  mode="
                f"{'act' if self.actuated else ('artic' if self.articulated else 'mocap')}{trk}")
      if self.articulated and getattr(self, "_cmd_q", None) is not None:
        # RE-ASSERT the commanded arm pose. mj_step has to integrate the chain (the PARTS
        # need stepping and MuJoCo cannot step them selectively), so the weld accelerates the
        # arm a little every step; left alone that error compounds and the grasp force runs
        # away (measured 28 -> 351 -> 921 -> 15989 N). Rewriting qpos here bounds it to a
        # single step, which is exactly the free lunch a mocap body gets by sitting outside
        # the constrained dynamics. This is what makes the KINEMATIC arm behave; it goes away
        # when actuators make the chain track for real.
        self.set_arm_q(*self._cmd_q)
      self.step_i += 1
      for r, (eq, part, fcap, tcap) in list(self.active_grasps.items()):
        fmag, tmag = self._weld_wrench(eq)
        if self.wrench_trace is not None:                   # diagnostic: reuses THIS wrench
          self.wrench_trace(r, part, fmag, tmag, fcap, tcap)
        _h = self._hold_info.get(r) if self.directional_caps else None
        if _h is not None:                                  # the planner's BOX, in the grasp frame
          _ratio, _lbl, fmag, tmag = self._grasp_violation(_h)
          _over = _ratio > 1.0
        else:                                               # no grasp id -> conservative scalar
          _lbl, _over = "", (fmag > fcap or tmag > tcap)
        if _over:
          self._over[r] = self._over.get(r, 0) + 1
          if self._over[r] >= sustain:                     # sustained overload -> seal fails
            self.data.eq_active[eq] = 0
            del self.active_grasps[r]
            self._hold_info.pop(r, None)
            if report is not None:
              report.results.append(ActionResult(
                  action or ("step",), False, "grasp_broke",
                  f"{r}/{part} |F|={fmag:.1f}N |T|={tmag:.2f}Nm"
                  + (f" -- worst bound {_lbl}" if _lbl else
                     f" (scalar caps {fcap:g}N/{tcap:g}Nm)")
                  + f" sustained {BREAK_SUSTAIN*1000:.0f}ms"
                  + self._contact_context(r, part)))
        else:
          self._over[r] = 0
      if self.mate_caps is not None:
        # FINITE MATE STRENGTH: the same sustained-overload breakaway as grasps, applied to
        # every ACTIVE mate weld. A broken mate deactivates -- the child separates and the
        # judge sees the cascade (toppled/not_seated) with mate_broke as the first failure.
        # This is what makes "hold the car by one wheel" fail EMERGENTLY: the wheel-body weld
        # carries the whole group's weight, exceeds its cap, and lets go.
        _mf, _mt = self.mate_caps
        for _p, (_eq, _par) in self._mate_eqs.items():
          if not self.data.eq_active[_eq]:
            self._mate_over.pop(_p, None)
            continue
          _fm, _tm = self._weld_wrench(_eq)
          _mx = self._mate_max.get(_p, (0.0, 0.0))
          self._mate_max[_p] = (max(_mx[0], _fm), max(_mx[1], _tm))
          if _fm > _mf or _tm > _mt:
            self._mate_over[_p] = self._mate_over.get(_p, 0) + 1
            if self._mate_over[_p] >= sustain:              # sustained overload -> joint fails
              self.data.eq_active[_eq] = 0
              self._mate_over.pop(_p, None)
              if report is not None:
                report.results.append(ActionResult(
                    action or ("step",), False, "mate_broke",
                    f"{_p}~{_par} |F|={_fm:.1f}N |T|={_tm:.2f}Nm "
                    f"(mate caps {_mf:g}N/{_mt:g}Nm) "
                    f"sustained {BREAK_SUSTAIN*1000:.0f}ms"))
          else:
            self._mate_over[_p] = 0
      if self.recorder is not None:
        self._rec_count += 1
        if self._rec_count % RECORD_EVERY == 0:
          # per-robot grasp wrench sample aligned with the frame (nan = no active grasp)
          fr = np.full((len(self.robots), 2), np.nan)
          for j, r in enumerate(self.robots):
            g = self.active_grasps.get(r)
            if g is not None:
              fr[j] = self._weld_wrench(g[0])
          self.recorder["qpos"].append(self.data.qpos.copy())
          # mocap bodies (EE pucks / arm links) live OUTSIDE qpos -- record them too, else
          # the replay shows frozen robots (old recordings had exactly that bug).
          self.recorder["mpos"].append(self.data.mocap_pos.copy())
          self.recorder["mquat"].append(self.data.mocap_quat.copy())
          self.recorder["forces"].append(fr)
          self.recorder["controller_mode"].append(_record_controller_mode(action))
          self.recorder["physics_tick"].append(int(self._phys))
      if self.viewer is not None:
        self.viewer.sync()

  def drive_mocap(self, robot, pose_wxyz):
    """Set the mocap target (position-level kinematic driving; MuJoCo pulls the welded
    gripper body along through the constraint solver)."""
    mid = self.model.body_mocapid[self._body_id(f"ee_{robot}")]
    self.data.mocap_pos[mid] = pose_wxyz[:3]
    self.data.mocap_quat[mid] = pose_wxyz[3:]


# ---------------------------------------------------------------------------------------------
# pose utilities (wxyz, matching problem.py) + state readers
# ---------------------------------------------------------------------------------------------
def _quat_conj(q):
  return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_angle(q1, q2):
  d = abs(float(np.dot(np.asarray(q1) / np.linalg.norm(q1),
                       np.asarray(q2) / np.linalg.norm(q2))))
  return 2.0 * np.arccos(min(1.0, d))


def _part_orientation_error(assembly, part, q1, q2):
  """Physical orientation error modulo a primitive's continuous symmetries.

  A round loose peg can roll on the table without toppling: rotation about a cylinder's
  local z axis changes its quaternion but not its geometry.  Comparing raw quaternions made
  the old pose judge stop a peg-stool replay after an innocuous 4.9mm roll (the cylinder axis had
  moved only 0.00019deg).  Spheres have no observable orientation; boxes, compounds and
  meshes retain the ordinary full-quaternion comparison.  The absolute cylinder-axis dot
  also accounts for its identical end caps.
  """
  kind = assembly.part_shape(part)[0]
  if kind == "sphere":
    return 0.0
  if kind == "cylinder":
    a1 = _quat_to_R(np.asarray(q1, dtype=float))[:, 2]
    a2 = _quat_to_R(np.asarray(q2, dtype=float))[:, 2]
    d = abs(float(np.dot(a1, a2)))
    return float(np.arccos(np.clip(d, -1.0, 1.0)))
  return _quat_angle(q1, q2)


def _rel_pos(pose_a, pose_b):
  return _quat_to_R(np.asarray(pose_a[3:])).T @ (np.asarray(pose_b[:3])
                                                 - np.asarray(pose_a[:3]))


def _held_grasp_id(state, part, robot=None):
  """Return ``(contacted_part, gid)`` for a grasp holding ``part``'s rigid group.

  The motion-planning attach child is the group's fixed mate-tree root, not necessarily the
  part touched by the gripper.  Looking only in ``state.supports[part]`` therefore discarded
  the grasp id on every post-assembly handover and silently selected scalar sim caps.
  """
  group = next((g for g in state.subassemblies() if part in g), (part,))
  for p in sorted(group):
    for c in state.supports.get(p, ()):
      if c[0] == "grasp" and c[2] == p and (robot is None or c[1] == robot):
        return p, c[3]
  who = "" if robot is None else f" by {robot}"
  raise ValueError(f"{part}'s group not held{who} in this state")


# ---------------------------------------------------------------------------------------------
# execute an mr_bench MOTION PLAN (BiRRT path) with FK-synced arm links
# ---------------------------------------------------------------------------------------------
JOINT_RATE = 1.0                # rad/s joint-space tracking speed along RAW path segments.
                                # Arms are joint-lerped + FK-synced at EVERY sim substep:
                                # coarser stepping teleports the links and the grasp weld
                                # YANKS the held part (measured 24.8N > 15N cap -- a pure
                                # tracking artifact). The raw path is driven directly; we do
                                # NOT use mr_bench interpolate_path -- its mode tagging put
                                # the first "new-mode" state up to a fifth of a (shortcut-
                                # long) segment past the goal config, so events fired ~50mm
                                # off the witness (measured; raw transition states are exact
                                # duplicated goal configs, verified by FK tip distance).


class _SkillServo:
  """--skill-servo: the EXECUTION half of the assemble/place funnel -- guarded, compliant
  skills. Actuated-only. Two pieces:

  (a) ADMITTANCE IN MOTION. The measured touchdown/graze failures are 27-82N presses of a
      HELD part against the world for ~20ms while tracking a reference that legally skims the
      table (only penetration is illegal to the planner, and executed = commanded +- ~1-3mm
      near contact). Per control tick, each held part's net contact force is read; above
      `force` N the arm's REFERENCE yields along the push direction (bounded by `max_dev`,
      decaying back when free). The trigger is UNINTENDED FORCE, not proximity -- so intended
      contacts (set-down, seating) still happen; the arm just stops pressing through them.

  (b) TERMINAL SEAT SERVO. Before a part-part mate weld fires, the mate's residual is whatever
      seat error tracking left, and attach() then snaps to the ideal datum -- a constraint
      impulse proportional to the miss. This servos the CHILD's position to the datum via the
      carrying arm's Jacobian first (rate-limited, force-guarded), so the weld starts from
      millimetres, not centimetres. It replaces error ACCUMULATION with per-skill contraction:
      each mate re-zeroes the error instead of inheriting it.
  """

  def __init__(self, scene, dt, force=6.0, yield_gain=0.004, max_dev=0.010,
               decay_tau=0.3, kp=4.0, vmax=0.05, tol=0.002, torque_guard=None,
               compliant=False, compliant_force=3.0):
    self.scene = scene
    self.dt = float(dt)
    self.f0 = float(force)
    # --servo-torque-guard FRAC (historic flag name): engage guarded hold + yield when EITHER
    # component of the measured grasp wrench exceeds FRAC of its own scalar cap. The former
    # torque-only check still let a 25N grasp reach its force cap before the generic 24N
    # bulldoze freeze could react; a normalized wrench guard uses the same force/torque and
    # caps as the breakage monitor and therefore has an actual safety margin.
    self.torque_guard = None if torque_guard is None else float(torque_guard)
    self.wrench_holds = 0                                  # normalized-wrench-triggered ticks
    # --compliant-follow: WHOLE-BRANCH admittance (arm links + gripper + cargo) with its own
    # LOWER deadband -- compliant following as the default tracking law, not a held-cargo-only
    # feature. The guard/funnel triggers (f0) keep their meaning: the reference starts
    # deviating at c_f0 before the clock even pauses at f0.
    self.compliant = bool(compliant)
    self.c_f0 = float(compliant_force)
    self.place_ok = False                                  # --servo-place-seat (set by caller)
    self.place_log = []                                    # (task, xy_err0, F_end, touched)
    self.pick_ok = False                                   # --servo-pick (set by caller)
    self.pick_log = []                                     # (task, F_at_fire, F_end, lift s)
    self.kf = float(yield_gain)                            # m/(N*s) of yield per N of excess
    self.max_dev = float(max_dev)
    self.decay = self.dt / float(decay_tau)
    self.kp = float(kp)                                    # 1/s task-space P gain (seat servo)
    self.vmax = float(vmax)                                # m/s task-space rate cap
    self.tol = float(tol)                                  # seat servo convergence (m)
    self.off = {}                                          # robot -> world task-space offset
    self.peak = {}                                         # robot -> peak |offset| (diagnostic)
    self.seat_log = []                                     # (task name, before m, after m)
    self.seat_touch = set()                                # contact-terminated mate skills
    # per-robot (MuJoCo dof addresses, commanded-q indices), from the scene's own joint table
    # so the mapping cannot drift from what drive_q commands.
    self.dofs, self.qidx = {}, {}
    for i, jn in enumerate(scene._rai_joints):
      r = jn.split("_", 1)[0]
      self.dofs.setdefault(r, []).append(scene._vadr[jn])
      self.qidx.setdefault(r, []).append(i)

  def _jac(self, robot, part):
    """3 x n positional Jacobian of `part`'s position w.r.t. the carrying arm's joints.

    NOT mj_jacBody(part): a part is a FREEJOINT body attached to the gripper only by an
    equality weld, so its body Jacobian has ALL-ZERO columns at the arm dofs -- lstsq then
    returns dq=0 and every task-space servo built on it silently commands nothing (verified
    with a minimal MuJoCo model; explains why yields never moved an actuated arm and pacing
    was the load-bearing layer). The cargo rides rigidly on the ANCHOR body, so the correct
    map is the point-Jacobian of the part's current position treated as attached to the
    anchor -- same lever-arm folding, real arm columns. For part == the anchor itself (bare
    arm) this degenerates to the anchor's own point Jacobian."""
    jacp = np.zeros((3, self.scene.model.nv))
    jacr = np.zeros((3, self.scene.model.nv))
    point = np.asarray(self.scene.data.xpos[self.scene._body_id(part)], dtype=float)
    mujoco.mj_jac(self.scene.model, self.scene.data, jacp, jacr, point,
                  self.scene._body_id(self.scene.grasp_anchor[robot]))
    return jacp[:, self.dofs[robot]]

  def _part_pose_jac(self, robot, part):
    """6-D Jacobian of a welded cargo pose through its grasp-anchor body."""
    jacp = np.zeros((3, self.scene.model.nv))
    jacr = np.zeros((3, self.scene.model.nv))
    point = np.asarray(self.scene.data.xpos[self.scene._body_id(part)], dtype=float)
    mujoco.mj_jac(self.scene.model, self.scene.data, jacp, jacr, point,
                  self.scene._body_id(self.scene.grasp_anchor[robot]))
    return np.vstack((jacp[:, self.dofs[robot]], jacr[:, self.dofs[robot]]))

  def _anchor_jac(self, robot):
    """World-frame 6-D Jacobian of the physical grasp anchor."""
    jacp = np.zeros((3, self.scene.model.nv))
    jacr = np.zeros((3, self.scene.model.nv))
    body = self.scene._body_id(self.scene.grasp_anchor[robot])
    point = np.asarray(self.scene.data.xpos[body], dtype=float)
    mujoco.mj_jac(self.scene.model, self.scene.data, jacp, jacr, point, body)
    return np.vstack((jacp[:, self.dofs[robot]], jacr[:, self.dofs[robot]]))

  def align_pick(self, q_ref, robot, part, nominal_part_pose, nominal_anchor_pose,
                 drive, tick, budget, observed=True, angular_tol=np.deg2rad(2.0)):
    """Measure, optionally correct, and return the terminal grasp-pose residual.

    The desired anchor is obtained by left-multiplying the nominal anchor with the measured
    world displacement of the part. This preserves the exact saved grasp transform and all
    robot-specific EE calibration; no grasp geometry is reconstructed in the simulator.
    """
    actual_part = np.asarray(self.scene.part_pose(part), dtype=float)
    delta = _pose_compose(actual_part, _pose_invert(np.asarray(nominal_part_pose, dtype=float)))
    desired = _pose_compose(delta, np.asarray(nominal_anchor_pose, dtype=float))
    q = np.asarray(q_ref, dtype=float).copy()
    # Continue from the plant rather than jumping back to the nominal reference after settle.
    for j in self.qidx[robot]:
      q[j] = float(self.scene.data.qpos[self.scene._qadr[self.scene._rai_joints[j]]])

    def residual():
      body = self.scene._body_id(self.scene.grasp_anchor[robot])
      cur = np.r_[self.scene.data.xpos[body], self.scene.data.xquat[body]]
      ep = desired[:3] - cur[:3]
      qe = _quat_mul(desired[3:], _quat_conj(cur[3:]))
      if qe[0] < 0.0:                                    # shortest representative
        qe = -qe
      nv = float(np.linalg.norm(qe[1:]))
      if nv < 1e-12:
        er = np.zeros(3)
      else:
        er = qe[1:] * (2.0 * np.arctan2(nv, max(float(qe[0]), 0.0)) / nv)
      return ep, er

    ep0, er0 = residual()
    ticks = guarded = 0
    if observed:
      for _ in range(int(float(budget) / self.dt)):
        ep, er = residual()
        if np.linalg.norm(ep) <= self.tol and np.linalg.norm(er) <= angular_tol:
          break
        if np.linalg.norm(self.scene.branch_contact_force(robot)) > 4.0 * self.f0:
          guarded = 1
          break
        vp = self.kp * ep
        wp = self.kp * er
        npv, nw = float(np.linalg.norm(vp)), float(np.linalg.norm(wp))
        if npv > self.vmax:
          vp *= self.vmax / npv
        angular_vmax = 0.75
        if nw > angular_vmax:
          wp *= angular_vmax / nw
        dq = np.linalg.lstsq(self._anchor_jac(robot), np.r_[vp, wp] * self.dt,
                             rcond=None)[0]
        q[self.qidx[robot]] += np.clip(dq, -0.02, 0.02)
        drive(q)
        tick()
        ticks += 1
    ep, er = residual()
    return {"observed": bool(observed), "position_initial_m": float(np.linalg.norm(ep0)),
            "angle_initial_deg": float(np.rad2deg(np.linalg.norm(er0))),
            "position_m": float(np.linalg.norm(ep)),
            "angle_deg": float(np.rad2deg(np.linalg.norm(er))),
            "duration_s": ticks * self.dt, "guarded": bool(guarded)}

  def update(self, only=None):
    """Once per control tick, BEFORE building the adjusted reference: read each held part's
    net contact force (last step's state -- one tick late is immaterial at 5ms) and integrate
    the per-arm yield offsets. `only`: optional iterable of robot names to update -- the
    fire-flow window uses it to keep integrating ONLY the flowing arms, so a window spent at
    a contact barrier cannot silently integrate yield offsets on the arms pinned there."""
    self._hot = 0.0
    if only is None:
      self._hot_by = {}
    if self.compliant:
      self._update_compliant(only)
      return
    for r, g in self.scene.active_grasps.items():
      if only is not None and r not in only:
        continue
      part = g[1]
      F = self.scene.net_contact_force(part)
      mag = float(np.sqrt(F @ F))
      hot = mag
      excess = mag - self.f0
      if self.torque_guard is not None:
        # Normalized force+torque trigger: same weld wrench/caps the break monitor judges.
        fmag, tmag = self.scene._weld_wrench(g[0])
        util = max(fmag / max(self.torque_guard * float(g[2]), 1e-9),
                   tmag / max(self.torque_guard * float(g[3]), 1e-9))
        # The weld wrench includes the payload's STATIC gravity reaction. It is safe (and
        # necessary) to keep transporting a grasp whose nominal gravity utilization exceeds
        # FRAC; freezing that case deadlocks forever because stopping cannot remove gravity.
        # Turn utilization into a hard guard only when an EXTERNAL contact is simultaneously
        # loading the branch. This is the failure we need to catch (car: 13.5N part contact
        # drove a 0.67Nm grasp), and the contact direction gives admittance a way to recover.
        if util > 1.0 and mag > self.f0:
          hot = max(hot, 4.0 * self.f0 + 1e-3)          # engages guard + funnel like a press
          # Map normalized excess through f0 so the admittance gain keeps its m/(N*s) units.
          # Direction remains the contact push; when the wrench is purely inertial the hard
          # hold still removes its cause even though there is no contact direction to yield.
          excess = max(excess, (util - 1.0) * self.f0)
          self.wrench_holds += 1
      self._hot_by[r] = hot
      if hot > self._hot:
        self._hot = hot
      o = self.off.get(r)
      if o is None:
        o = np.zeros(3)
      if excess > 0.0 and mag > 1e-9:
        o = o + (self.kf * excess * self.dt) * (F / mag)
        n = float(np.sqrt(o @ o))
        if n > self.max_dev:
          o = o * (self.max_dev / n)
        if n > self.peak.get(r, 0.0):
          self.peak[r] = n
      else:
        o = o * (1.0 - self.decay)
      self.off[r] = o

  def _update_compliant(self, only=None):
    """--compliant-follow: ONE admittance loop over EVERY arm, driven by the whole-branch
    contact force -- a press on a bare gripper (pick approach, pre-weld) or an arm link
    yields exactly like a cargo press. Yield deadband c_f0 < f0, so the reference deviates
    before the guard/funnel (which keep reading the same magnitudes via _hot) engage."""
    for r in self.dofs:
      if only is not None and r not in only:
        continue
      g = self.scene.active_grasps.get(r)
      F = self.scene.branch_contact_force(r)
      mag = float(np.sqrt(F @ F))
      hot = mag
      excess = mag - self.c_f0
      if self.torque_guard is not None and g is not None:
        # Identical normalized wrench trigger to the held-cargo path above.
        fmag, tmag = self.scene._weld_wrench(g[0])
        util = max(fmag / max(self.torque_guard * float(g[2]), 1e-9),
                   tmag / max(self.torque_guard * float(g[3]), 1e-9))
        if util > 1.0 and mag > self.f0:
          hot = max(hot, 4.0 * self.f0 + 1e-3)
          excess = max(excess, (util - 1.0) * self.f0)
          self.wrench_holds += 1
      self._hot_by[r] = hot
      if hot > self._hot:
        self._hot = hot
      o = self.off.get(r)
      if o is None:
        o = np.zeros(3)
      if excess > 0.0 and mag > 1e-9:
        o = o + (self.kf * excess * self.dt) * (F / mag)
        n = float(np.sqrt(o @ o))
        if n > self.max_dev:
          o = o * (self.max_dev / n)
        if n > self.peak.get(r, 0.0):
          self.peak[r] = n
      else:
        o = o * (1.0 - self.decay)
      self.off[r] = o

  def pressing(self):
    """True while any held part carries contact force above the threshold -- the GUARDED-HOLD
    trigger: the schedule stops advancing so the (slow, plant-bandwidth) yield can win against
    the (fast, 20ms) breakaway window."""
    return getattr(self, "_hot", 0.0) > self.f0

  def _cargo_clearance(self, robot):
    """Min table clearance of `robot`'s held cargo (m), or None when not carrying/no table.
    Exact per-geom bottom projection from the EXECUTED MuJoCo state, no collision queries."""
    m, d = self.scene.model, self.scene.data
    if not hasattr(self, "_ttop"):
      self._ttop = None
      for gi in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if "table" in nm:
          self._ttop = float(m.geom_pos[gi][2] + m.geom_size[gi][2])
          break
      self._part_geoms = {}
    if self._ttop is None:
      return None
    g = self.scene.active_grasps.get(robot)
    if g is None:
      # BARE-GRIPPER clearance (--compliant-follow / --servo-pick): the funnel was
      # cargo-gated, so the one descent that needed it most -- an empty gripper's pick
      # approach -- ran unpaced (measured: 13mm in 96ms into the grasp keyframe, 48N press
      # sustained on the fresh weld). Same bottom-projection, over the EE subtree's geoms.
      if not (self.compliant or self.pick_ok):
        return None
      return self._bare_ee_clearance(robot)
    part = g[1]
    gids = self._part_geoms.get(part)
    if gids is None:
      bid = self.scene._body_id(part)
      gids = [gi for gi in range(m.ngeom) if int(m.geom_bodyid[gi]) == bid]
      self._part_geoms[part] = gids
    lo = None
    for gi in gids:
      R = d.geom_xmat[gi].reshape(3, 3)
      c = float(d.geom_xpos[gi][2] - np.abs(R[2]) @ m.geom_size[gi]) - self._ttop
      if lo is None or c < lo:
        lo = c
    return lo

  def _held_bodies(self):
    """Body ids that currently ride an arm: every grasped cargo plus everything transitively
    welded to it. A MATE weld alone does NOT mean held -- a mated subassembly resting on the
    table is exactly what a re-pick descends onto, and treating it as held made the approach
    funnel blind to the tallest obstacle in the scene."""
    m, d = self.scene.model, self.scene.data
    frontier = [self.scene._body_id(g[1]) for g in self.scene.active_grasps.values()]
    held = set(frontier)
    while frontier:
      b = frontier.pop()
      for e in range(int(m.neq)):
        if not d.eq_active[e] or m.eq_type[e] != mujoco.mjtEq.mjEQ_WELD:
          continue
        b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        other = b2 if b1 == b else (b1 if b2 == b else None)
        if other is not None and other not in held:
          held.add(other)
          frontier.append(other)
    return held

  def _bare_ee_clearance(self, robot):
    """Min clearance (m) of an EMPTY gripper above the table or any LOOSE part beneath it,
    so the funnel paces a pick approach onto what it is actually descending toward -- a part
    top sits ~12mm above the table, and table-only clearance would let the EE arrive at
    part height still doing 20mm/s."""
    m, d = self.scene.model, self.scene.data
    if not hasattr(self, "_ee_geoms"):
      self._ee_geoms = {}
    gids = self._ee_geoms.get(robot)
    if gids is None:
      sub = {self.scene._body_id(self.scene.grasp_anchor[robot])}
      grew = True                                          # anchor body + its descendants
      while grew:
        grew = False
        for b in range(int(m.nbody)):
          if b not in sub and int(m.body_parentid[b]) in sub:
            sub.add(b)
            grew = True
      gids = [gi for gi in range(m.ngeom) if int(m.geom_bodyid[gi]) in sub]
      self._ee_geoms[robot] = gids
    if not gids:
      return None
    bot, xy = None, None
    for gi in gids:
      R = d.geom_xmat[gi].reshape(3, 3)
      zb = float(d.geom_xpos[gi][2] - np.abs(R[2]) @ m.geom_size[gi])
      if bot is None or zb < bot:
        bot, xy = zb, np.asarray(d.geom_xpos[gi][:2])
    c = bot - self._ttop
    held = self._held_bodies()
    for p in self.scene.assembly.objects:
      if self.scene._body_id(p) in held:
        continue                                           # rides an arm: not an obstacle
      pg = self._part_geoms.get(p)
      if pg is None:
        bid = self.scene._body_id(p)
        pg = [gi for gi in range(m.ngeom) if int(m.geom_bodyid[gi]) == bid]
        self._part_geoms[p] = pg
      for gi in pg:
        dx = d.geom_xpos[gi][:2] - xy
        if float(dx @ dx) > (float(np.max(m.geom_size[gi])) + 0.03) ** 2:
          continue
        R = d.geom_xmat[gi].reshape(3, 3)
        cp = bot - float(d.geom_xpos[gi][2] + np.abs(R[2]) @ m.geom_size[gi])
        if -0.005 < cp < c:                                # beside/below a part != above it
          c = cp
    return c

  def rate_factors(self, robot_names, near=0.012, pace=6):
    # NB: unlike the retired static schedule caps, a factor here lasts exactly as long as its
    # trigger -- a horizontal low carry only pays while it is actually low, and nobody else
    # pays at all (the stepper lets the other arms flow within the certified corridor).
    """PER-ARM runtime rate multipliers for the ONLINE corridor stepper: 1.0 free, 1/pace
    while THAT arm's cargo is near the table, 0.0 while THAT arm's cargo is being pressed.
    This is the whole point of executing online: a slowdown exists only while its trigger
    holds, and only on the arm that owns the trigger -- the corridor lets everyone else flow."""
    # VELOCITY FUNNEL on the cargo's MEASURED descent, not a blanket arc-rate scale. The
    # depth-scaled arc factor still grazed at -2.05mm/64N while the offline executor's spread
    # pass -- which happens to glide into barriers -- touched at -0.24mm on the same draw:
    # arc rate maps to vertical speed too loosely. Bound the physical quantity instead:
    #     |vz_down| <= k * (clearance - c0)
    # (k=2/s, c0=2mm) -- the classic guarded-approach funnel: 46mm/s allowed at 25mm out,
    # 6mm/s at 5mm, asymptotically soft at contact, and only while actually DESCENDING; a
    # horizontal low carry pays nothing. Per-arm, so nobody else pays either.
    f = np.ones(len(robot_names))
    near, c0, k = 0.025, 0.002, 2.0
    if not hasattr(self, "_c_prev"):
      self._c_prev = {}
    for i, rn in enumerate(robot_names):
      if getattr(self, "_hot_by", {}).get(rn, 0.0) > 4.0 * self.f0:
        # HARD FREEZE ONLY AT THE BULLDOZE SCALE (same 4xf0 the seat/place ceremonies
        # use). A planned contact (the co-hold carry, the pre-weld touch) reads a few
        # newtons and must NOT stop the stepper -- freezing at 1xf0 deadlocked an
        # 0.5 rad carry for 38 s when the carried leg touched the co-held seat
        # (measured, stretch k=20). The yield offsets still engage at f0, so the
        # reference yields softly; only a breakage-scale shove freezes the arm.
        f[i] = 0.0
        continue
      c = self._cargo_clearance(rn)
      if c is None:
        self._c_prev.pop(rn, None)
        continue
      vz = (c - self._c_prev.get(rn, c)) / self.dt           # + rising, - descending
      self._c_prev[rn] = c
      if c < near and vz < -1e-6:
        allowed = k * max(c - c0, 0.0005)
        f[i] = min(1.0, allowed / (-vz))
    return f

  # (the binary approach_pace is GONE: the velocity funnel in rate_factors supersedes it in
  # BOTH executors -- offline as a min-over-arms hold on the global clock, online per-arm.)

  def adjust(self, q):
    """Fold the task-space offsets into the joint reference (minimum-norm dq per arm)."""
    out = None
    for r, o in self.off.items():
      if float(o @ o) < 1e-10:
        continue
      g = self.scene.active_grasps.get(r)
      if g is None:
        if not self.compliant:
          continue
        body = self.scene.grasp_anchor[r]                  # bare arm: yield at the EE
      else:
        body = g[1]
      dq = np.linalg.lstsq(self._jac(r, body), o, rcond=None)[0]
      if out is None:
        out = np.asarray(q, dtype=float).copy()
      out[self.qidx[r]] += np.clip(dq, -0.05, 0.05)
    return q if out is None else out

  def place_seat(self, q_ref, part, target_pose, name, drive, tick, budget):
    """--servo-place-seat: force-terminated PLACE. A place's terminal objective is a CONTACT
    STATE -- the part resting on the real support -- not a coordinate: the joint reference's
    z was authored against the PLANNER's table, which sits part_float BELOW the executed one,
    so tracking it grinds the part into the surface (measured on the chair seat: 0.22mm/s
    descent ratcheting the grasp torque to its cap). Here the part is servoed in TASK SPACE:
    XY toward the rest anchor, z descending at a bounded rate, TERMINATED by sustained
    contact -- the executed world, not the model, decides where the set-down ends."""
    holder = next((r for r, g in self.scene.active_grasps.items() if g[1] == part), None)
    if holder is None:
      return
    tgt = np.asarray(target_pose[:3], dtype=float)
    q_ref = np.asarray(q_ref, dtype=float).copy()
    e0 = touched = 0
    fmag = 0.0
    for k in range(int(budget / self.dt)):
      cur = np.asarray(self.scene.part_pose(part)[:3], dtype=float)
      if k == 0:
        e0 = float(np.hypot(*(tgt[:2] - cur[:2])))
      F = self.scene.net_contact_force(part)
      fmag = float(np.sqrt(F @ F))
      if fmag > 4.0 * self.f0:
        break                                              # bulldoze guard, as in seat()
      touched = touched + 1 if fmag > 0.5 * self.f0 else 0
      if touched * self.dt >= 0.03:
        break                                              # 30ms of sustained support: seated
      e = np.zeros(3)
      e[:2] = np.clip(self.kp * (tgt[:2] - cur[:2]), -self.vmax, self.vmax)
      e[2] = -0.015                                        # bounded descent toward touch (m/s)
      dq = np.linalg.lstsq(self._jac(holder, part), e * self.dt, rcond=None)[0]
      q_ref[self.qidx[holder]] += np.clip(dq, -0.02, 0.02)
      drive(q_ref)
      tick()
    self.place_log.append((name, e0, fmag, touched * self.dt >= 0.03))

  def pick_lift(self, q_ref, robot, part, name, drive, tick, budget):
    """--servo-pick: force-terminated LIFT-OFF owning the fresh-weld window. A pick's grasp
    keyframe is a CONTACT terminal: it is authored ~1mm into the part, plant lag bottoms out
    PAST it right after the fire, and the press lands exactly inside the 20ms breakage
    sustain that judges the brand-new weld (measured on chair-8 obj_3: 48N, identical with
    reference-side compliance active -- the onset outruns the plant). So do not track
    through it: rise in task space at a bounded rate until the cargo is contact-free
    (sustained), then hand back to the schedule."""
    q = np.asarray(q_ref, dtype=float).copy()
    # START FROM THE PLANT, NOT THE SCHEDULE. The fire-sample reference sits at the authored
    # grasp depth (~1mm into the part), and the funnel-paced plant can still be several mm
    # ABOVE it at fire time -- re-commanding that baseline plus a slow rise still NET-DESCENDS
    # the arm into the press (measured: 8.4mm plunge, 48N, weld broke inside the lift's first
    # ticks while its own force check read "free").
    jn = self.scene._rai_joints
    for j in self.qidx[robot]:
      q[j] = float(self.scene.data.qpos[self.scene._qadr[jn[j]]])
    F0 = float(np.linalg.norm(self.scene.net_contact_force(part)))
    free = ticks = 0
    for _ in range(int(budget / self.dt)):
      F = self.scene.net_contact_force(part)
      if float(F @ F) < 1.0:                               # < 1N: cargo is free...
        free += 1
        c = self._cargo_clearance(robot)
        # ...but free NOW is not enough: the schedule resumes at the pressing keyframe, so
        # terminate only with real CLEARANCE in hand (or when the scene has no table datum).
        if free * self.dt >= 0.03 and (c is None or c >= 0.004):
          break
      else:
        free = 0
      dq = np.linalg.lstsq(self._jac(robot, part),
                           np.array([0.0, 0.0, 0.8 * self.vmax]) * self.dt, rcond=None)[0]
      q[self.qidx[robot]] += np.clip(dq, -0.01, 0.01)
      drive(q)
      tick()
      ticks += 1
    # PARK the achieved lift in the admittance offset: adjust() keeps applying it to the
    # resumed schedule (which dwells AT the pressing keyframe post-fire) and it decays away
    # over decay_tau as the retreat takes over -- otherwise tracking pulls straight back
    # into the press the lift just relieved.
    o = np.array([0.0, 0.0, min(ticks * 0.8 * self.vmax * self.dt, self.max_dev)])
    if o[2] > self.off.get(robot, np.zeros(3))[2]:
      self.off[robot] = o
    self.pick_log.append((name, F0, float(np.linalg.norm(
        self.scene.net_contact_force(part))), ticks * self.dt))

  def seat(self, q_ref, parent, child, name, drive, tick, budget,
           relative_target=None):
    """Servo `child` to its ideal datum on `parent` before the mate weld fires.

    Nominal execution retains the historical position-only contraction. An observed placement
    perturbation supplies the saved full relative target and contracts orientation as well, so
    an initially yawed parent is not snapped back by the weld. Force-guarded: pressing past 4x
    the admittance threshold stops the approach.
    """
    holder = next((r for r, g in self.scene.active_grasps.items() if g[1] == child), None)
    if holder is None:
      return
    target_off = np.asarray(self.scene.assembly.relative_pose(child)[1], dtype=float)
    relative_target = (None if relative_target is None else
                       np.asarray(relative_target, dtype=float))
    q_ref = np.asarray(q_ref, dtype=float).copy()
    err0 = err = None
    for _ in range(int(budget / self.dt)):
      pp = self.scene.part_pose(parent)
      cp = self.scene.part_pose(child)
      desired = (_pose_compose(pp, relative_target) if relative_target is not None else
                 np.r_[pp[:3] + _quat_to_R(pp[3:]) @ target_off, cp[3:]])
      e = desired[:3] - cp[:3]
      err = float(np.sqrt(e @ e))
      if err0 is None:
        err0 = err
      ang = _quat_angle(desired[3:], cp[3:]) if relative_target is not None else 0.0
      if err < self.tol and ang < np.deg2rad(2.0):
        break
      # A mate is a CONTACT skill, not a Cartesian position regulator.  Tight CAD often
      # touches a fraction of a millimetre before its nominal datum because the planning and
      # MuJoCo collision backends have finite tolerances.  Continuing to integrate toward the
      # datum then bulldozes the table-supported parent (duct: 33N and 16mm of base slide).
      # Once the *intended pair* is physically loaded inside the already-existing capture
      # neighbourhood, stop the servo and let the datum weld complete the constrained mate.
      # Unrelated collisions do not trigger this, and a distant accidental touch cannot turn
      # into an attach.
      if (err <= CAPTURE_RANGE and
          self.scene.part_pair_contact_force(parent, child) >= 0.5):
        self.seat_touch.add(name)
        break
      F = self.scene.net_contact_force(child)
      if float(np.sqrt(F @ F)) > 4.0 * self.f0:
        break                                              # guarded: do not bulldoze the seat
      vp = self.kp * e
      npv = float(np.linalg.norm(vp))
      if npv > self.vmax:
        vp *= self.vmax / npv
      if relative_target is None:
        J, step = self._jac(holder, child), vp * self.dt
      else:
        qe = _quat_mul(desired[3:], _quat_conj(cp[3:]))
        if qe[0] < 0.0:
          qe = -qe
        nv = float(np.linalg.norm(qe[1:]))
        er = (np.zeros(3) if nv < 1e-12 else
              qe[1:] * (2.0 * np.arctan2(nv, max(float(qe[0]), 0.0)) / nv))
        wp = self.kp * er
        nw = float(np.linalg.norm(wp))
        if nw > 0.75:
          wp *= 0.75 / nw
        J, step = self._part_pose_jac(holder, child), np.r_[vp, wp] * self.dt
      dq = np.linalg.lstsq(J, step, rcond=None)[0]
      q_ref[self.qidx[holder]] += np.clip(dq, -0.02, 0.02)
      drive(q_ref)
      tick()
    if err0 is not None:
      self.seat_log.append((name, err0, err))


def execute_motion_plan(C, assembly, plan, states, problem, mp_path, gripper_kinds=None,
                        record=None, verbose=True, gui=False, visual=True,
                        record_metadata=None,
                        initial_state=None, object_perturbation=None,
                        object_perturb_observation="observed",
                        object_grasp_capture=0.010,
                        object_grasp_capture_angle=np.deg2rad(10.0),
                        rate_boost=False,
                        corridor=False, corridor_ds=1.0, corridor_ds_lead=0.25,
                        corridor_lead_max=4.0, corridor_verify=0, corridor_refine=0,
                        corridor_cross_only=False, corridor_online=False,
                        rate_clamp=False, settle_tol=0.0, settle_max=1.0, couple_handovers=False,
                        pace=False, dwell=DWELL_T, pace_trace=None, phase_schedule=False,
                        phase_servo_time=0.15, phase_smooth=False, phase_smooth_time=0.12,
                        phase_batch=False, phase_batch_min_motion=0.08,
                        phase_batch_max_gap=0.35, phase_intent=False,
                        phase_intent_ramp=0.15, phase_intent_rate=1.5,
                        phase_audit_resolution=0.01,
                        directional_caps=False, mate_caps=None,
                        verify_efc_cache=False, force_limits=False, wrench_margin=1.0, mass_scale=1.0, inertia_factor=1.0,
                        corner_limits=True,
                        spline=False, spline_per_seg=8, spline_tension=0.5,
                        c2_spline=False, c2_samples_per_seg=8, c2_control_ratio=0.55,
                        c2_smooth_weight=0.03, c2_max_deviation=0.20,
                        c2_resolution=0.01,
                        articulated=False, actuated=False, use_planner_table=False,
                        corridor_margin=0.0, weld_timeconst=None, limit_rate=False,
                        object_damping=0.0,
                        path_preflight=True, path_preflight_tolerance=PATH_PREFLIGHT_TOL,
                        path_preflight_resolution=None,
                        apply_process_wrench=False, table_mu=None,
                        process_noise_mode="nominal", process_noise_corner=0,
                        process_noise_seed=0, process_engagement_distance=0.01,
                        process_wrench_ramp=0.05,
                        skill_servo=False, servo_force=6.0, servo_max_dev=0.010,
                        servo_tol=0.002, servo_torque_guard=None,
                        servo_place_seat=False, compliant_follow=False,
                        compliant_force=3.0, servo_pick=False,
                        animate_grippers=False, gripper_motion_time=0.25):
  """Execute an mr_bench PATH (from mp_export.run_birrt) in MuJoCo with the REAL ARMS:
  every rai robot shape frame (the capsule collision model + gripper primitives) is mirrored
  as a kinematic body positioned by rai FK along the path -- the planner's own kinematics by
  construction, so witness configs land exactly. Parts stay fully dynamic.

  RELINKING = FRAME-BASED ATTACH REPLAY (fixed 2026-07-25). At each MODE TRANSITION the
  completed task's mr_bench frames [parent, child] are replayed as a weld -- EXACTLY what
  set_to_mode does -- via scene.attach: parent 'table' => the part rests free; a gripper frame
  => breakaway grasp; a part => datum mate (within CAPTURE_RANGE). This is required because the
  path was PLANNED under those attaches: a `release` is a re-root to `table` or to the
  CONTINUING HOLDER's gripper (not a detach at the releasing arm), so inferring welds from the
  action KIND made the arm drag a part it had "released" (measured: release fired with the arm
  1.3m away). Hold-through transfers and handovers fall out of the frames automatically.
  SEQUENCE problems only for now; depgraph linearizations reorder events -- TODO.

  C: the rai config the problem was built from (used read-only; FK runs on a copy).
  plan/states: the discrete plan + its state trajectory (event semantics + judging).
  problem: mp_export.make_problem(...) result. mp_path: run_birrt(...)[0]."""
  import mujoco  # noqa: F401  (fail early with the guarded-import message otherwise)
  import robotic as ry

  robots = list(problem.robots)
  import os as _osd
  if _osd.environ.get("SIM_DEBUG_CONTACTS"):
    for _n in ("a2_ur_palm", "a3_ur_finger2", "a2_ur_vacuum", "a2_gripper_fill",
               "a2_coll1", "a2_coll7"):
      for _tag, _cfg in (("C", C), ("problem.C", getattr(problem, "C", None))):
        if _cfg is None:
          continue
        _i = _cfg.getFrame(_n).info()
        print(f"[sim/debug] {_tag} {_n}: shape={_i.get('shape')} "
              f"contact={_i.get('contact')} size={_i.get('size')}")
    raise SystemExit(0)
  # FK config: a private copy so the caller's C is untouched.
  C_fk = ry.Config()
  C_fk.addConfigurationCopy(C)
  jnames = C_fk.getJointNames()
  ridx = {r: [i for i, jn in enumerate(jnames) if jn.startswith(r)] for r in robots}

  arms = _arm_shape_frames(C_fk, robots, visual=visual)
  # Frame SET and frame OBJECTS hoisted out of fk_poses: it ran per step, rebuilding the set
  # and re-resolving ~90 frames BY NAME every time. rai Frame handles stay valid for the life
  # of the config, so the lookups are pure waste.
  _fk_frame_names = {e[1] for e in arms}
  # A semantic EE alias can deliberately be shapeless/contact-free and therefore absent
  # from _arm_shape_frames.  build_mjcf still creates a tiny anchor body for it; cache its
  # FK pose here as well so that body is initialized and subsequently driven from the same
  # frame as the planner's attach task.
  for r in robots:
    semantic = f"{r}_ur_gripper_center"
    if C_fk.getFrame(semantic) is not None:
      _fk_frame_names.add(semantic)
  _fk_frames = sorted(_fk_frame_names)
  _fk_objs = [C_fk.getFrame(f) for f in _fk_frames]

  def fk_poses(qfull):
    C_fk.setJointState(qfull)
    return {f: np.asarray(o.getPose(), dtype=float) for f, o in zip(_fk_frames, _fk_objs)}

  def full_q(state_q):
    flat = state_q.state() if hasattr(state_q, "state") else np.asarray(state_q)
    qfull = np.asarray(C_fk.getJointState()).copy()
    o = 0
    for r in robots:
      n = len(ridx[r])
      qfull[ridx[r]] = flat[o:o + n]
      o += n
    return qfull

  # EE anchor frame for grasp welds.  Every supported real robot exposes the SAME semantic
  # frame used by RobotIK/mp_export's pick tasks: ``<robot>_ur_gripper_center``.  Do not infer
  # this from shape ordering.  In particular, Husky's last non-mesh shape is ``finger2``;
  # welding cargo there made the execution wrench and Cartesian pick servo act about one
  # fingertip while the saved keyframe was defined at the tool centre.  Keep the old rule
  # only for genuinely legacy models which predate the common alias.
  ee_anchor = {}
  for r in robots:
    semantic = f"{r}_ur_gripper_center"
    if C_fk.getFrame(semantic) is not None:
      ee_anchor[r] = semantic
      continue
    cands = [e[1] for e in arms if e[1].startswith(r) and e[2][0] != "mesh"]
    if not cands:
      raise ValueError(f"simulation export found no end-effector frame for {r}")
    ee_anchor[r] = next((f for f in cands if "gripper_fill" in f), cands[-1])
    print(f"[sim/mp] WARNING: {r} has no {semantic}; using legacy grasp anchor "
          f"{ee_anchor[r]}. Add the standard ur_gripper_center alias to the robot model.")

  q0 = full_q(mp_path[0].q)
  scene = SimScene(assembly, initial_state if initial_state is not None else states[0], robots,
                   gripper_kinds=gripper_kinds,
                   arms=arms, arm_poses=fk_poses(q0), ee_anchor=ee_anchor, gui=gui,
                   articulated_C=(C_fk if (articulated or actuated) else None),
                   actuated=actuated,
                   planner_table_top=(planner_table_top(C) if use_planner_table else None),
                   table_geometry=planner_table_geometry(C),
                   weld_timeconst=weld_timeconst, table_mu=table_mu,
                   process_engagement_distance=process_engagement_distance,
                   process_wrench_ramp=process_wrench_ramp,
                   animate_grippers=animate_grippers, object_damping=object_damping)
  if object_damping:
    print(f"[sim/mp] --object-damping {object_damping:g}: viscous damping applied to free "
          "part joints (simulation only).")
  print(f"[sim/mp] table contact friction from saved plan: mu={scene.table_mu:g} "
        f"(MuJoCo sliding/torsional/rolling={_contact_friction(scene.table_mu, table=True)}).")
  if animate_grippers:
    _n = sum(len(v) for v in scene._gripper_qadr.values())
    print(f"[sim/gripper] --animate-grippers ACTIVE: {_n} simulation-only articulated finger "
          f"hinge(s) with position actuators, object-dependent terminal apertures, "
          f"{gripper_motion_time:g}s smooth commands concurrent with arm motion. Planner/MP "
          "joint vectors are unchanged; NPZ qpos includes the finger joints.")
  if record_metadata is not None:
    # Persist the RESOLVED value, not the nullable plan field.  Legacy plans omit table_mu
    # and therefore intentionally use 0.5; recording null here made that physical choice
    # ambiguous even though the compiled XML happened to contain the right coefficient.
    record_metadata["table_mu"] = float(scene.table_mu)
    record_metadata["directional_caps"] = bool(directional_caps)
    if object_perturbation is not None:
      record_metadata["object_perturbation"] = dict(object_perturbation)
      record_metadata["object_perturbation"]["observation"] = object_perturb_observation
      record_metadata["object_perturbation"]["grasp_capture_m"] = float(
          object_grasp_capture)
      record_metadata["object_perturbation"]["grasp_capture_angle_deg"] = float(
          np.rad2deg(object_grasp_capture_angle))
  if scene.table_mu == 0.0:
    print("[sim/mp] WARNING: exactly zero table friction is an idealized stress test; loose "
          "round parts can drift from micron-scale contact noise before they are picked. "
          "Prefer a small measured value such as --table-friction 0.05 for a reproducible "
          "low-friction counterfactual.")
  scene.verify_efc_cache = bool(verify_efc_cache)
  scene.directional_caps = bool(directional_caps)
  scene.mate_caps = mate_caps
  if mate_caps is not None:
    print(f"[sim/mp] --mate-*-cap ACTIVE: finite mate strength {mate_caps[0]:g}N/"
          f"{mate_caps[1]:g}Nm -- an overloaded mate weld BREAKS (mate_broke) after "
          f"{BREAK_SUSTAIN*1000:.0f}ms, same breakaway rule as grasps. Per-mate load "
          f"maxima print at the end of the run.")
  articulated = articulated or actuated
  if articulated:
    print(f"[sim/mp] --articulated ACTIVE: real kinematic chains generated from rai "
          f"(arm_model), driven by SETTING qpos -- not {len(arms)} teleported mocap bodies. "
          f"Still perfectly stiff; run fk_compare.py to re-check the model.")

  _hook_log = {"settle": [], "seat": []}                    # fire-window durations (sim s)

  def settle_before_fire(q, touch_robots=(), touch_mates=(), flow=None, owned=None):
    """ACTUATED: hold the reference until the arm has actually ARRIVED, then fire.

    The schedule's dwell is a fixed DWELL_T, but a servo needs as long as it needs: measured on
    cross-new, a stop settles in ~0.54s against DWELL_T=0.15s, so every attach fired while the
    arm was still converging. _activate_weld anchors the grasp at the CURRENT gripper-to-part
    pose, so that residual is frozen into how the part sits in the hand and every downstream
    mate inherits it (measured pick-to-pick drift 0.05 -> 5 -> 40 -> 56mm before the damping
    fix). Waiting on the MEASURED error costs only the time actually needed, unlike raising
    DWELL_T for every transition whether it needs it or not. The stiff modes never need this
    (their error is 0 by construction), so it is actuated-only.

    `flow`: FIRE-FLOW (--corridor-online) callback. When given, each iteration calls it with
    the held barrier config; it advances every NON-acting arm through its certified corridor,
    drives the mixed config, steps the scene and returns the mixed q -- the tracking test
    below then judges exactly what was commanded. `owned`: robot names the arrival test is
    limited to under fire-flow (the acting arms); None = every arm (offline semantics)."""
    if not (actuated and settle_tol > 0.0):
      return
    its = 0
    _err0 = scene.tracking_error(q, owned)
    _touch = False
    _mate_touch = False
    _plateau = False
    _plateau_ticks = max(1, int(round(0.1 / DT)))
    _plateau_err = _err0
    for _k in range(int(settle_max / DT)):
      _err = scene.tracking_error(q, owned)
      if _err <= settle_tol:
        break
      # A mate is a contact goal.  Once the intended pair is loaded inside the existing
      # capture neighbourhood, continuing to erase actuator lag commands penetration and
      # can push a light table-supported base away.  Unrelated collisions and distant
      # accidental contacts cannot take this exit.
      for _parent, _child in touch_mates:
        _got = _rel_pos(scene.part_pose(_parent), scene.part_pose(_child))
        _miss = float(np.linalg.norm(
            _got - np.asarray(assembly.relative_pose(_child)[1], dtype=float)))
        if (_miss <= CAPTURE_RANGE and
            scene.part_pair_contact_force(_parent, _child) >= 0.5):
          _mate_touch = True
          break
      if _mate_touch:
        break
      # Finite-gain position servos have a small payload-biased steady-state residual. Waiting
      # the full timeout at that floor does nothing (car: 1.9mrad; chair: 3.3mrad, unchanged
      # for 1s). Accept only a SMALL, demonstrably flat residual: below 2x the requested gate
      # and improving by <25% of it over 100ms. A large error or one still converging cannot
      # take this exit and retains the hard settle_max bound.
      if _k and _k % _plateau_ticks == 0:
        if _err <= 2.0 * settle_tol and _plateau_err - _err < 0.25 * settle_tol:
          _plateau = True
          break
        _plateau_err = _err
      if touch_robots and any(
          float(np.linalg.norm(scene.branch_contact_force(r))) > 1.0 for r in touch_robots):
        # --servo-pick: the picking arm has TOUCHED its part. The remaining tracking error
        # IS the authored grasp interference (~1mm into the part) -- settling it closed is
        # what drove the 48-52N press the fresh weld then inherited and broke under. Fire
        # at touch instead; pick_lift owns the rest of the window.
        _touch = True
        break
      if flow is not None:
        q = flow(q)                                            # mixed: acting hold, others flow
      else:
        drive_q(q)                                             # qdot=0: hold, do not lead
        scene.step(1, report, ("settle",))
      its += 1
    _errs = {}
    for _r in (owned if owned is not None else robots):
      _errs[_r] = scene.tracking_error(q, [_r])
    _jerrs = []
    for _i, _jn in enumerate(scene._rai_joints):
      if _jn not in scene._qadr:
        continue
      _r = _jn.split("_", 1)[0]
      if (owned is not None and _r not in owned) or (owned is None and _r not in robots):
        continue
      _d = float(scene.data.qpos[scene._qadr[_jn]] - q[_i])
      if scene._rai_hinge[_i]:
        _d = (_d + np.pi) % (2.0 * np.pi) - np.pi
      _jerrs.append((_jn, _d))
    _hook_log["settle"].append(
        (its * DT, _err0, scene.tracking_error(q, owned), _touch, _errs,
         list(owned) if owned is not None else None, _jerrs, _plateau, _mate_touch))

  def seat_before_fire(tid, q_ref, flow_drive=None, flow_tick=None):
    """--skill-servo: contract the seat error toward the mate datum BEFORE the weld fires,
    so attach()'s ideal-datum snap starts from millimetres instead of the tracking residual.

    `flow_drive`/`flow_tick`: FIRE-FLOW (--corridor-online) drive/tick pair -- `flow_drive`
    mixes the servoed q_ref with the advancing non-acting arms' corridor configs before
    driving; `flow_tick` steps the scene. Non-acting arms keep flowing while the seat
    converges."""
    if servo is None:
      return
    t = problem.tasks[tid]
    fr = getattr(t, "frames", None)
    if not fr:
      return
    _cnt = [0]
    _base_tick = flow_tick if flow_tick is not None else \
        (lambda: scene.step(1, report, ("servo",)))
    def _tck():
      _cnt[0] += 1
      _base_tick()
    _drv = flow_drive if flow_drive is not None else drive_q
    if fr[0] == "table" and len(fr) > 1 and fr[1] in assembly.objects and servo.place_ok:
      # --servo-place-seat: a PLACE fires by CONTACT, not by coordinate (see place_seat).
      _l = name_of.get(t.name)
      if _l is not None:
        exp = _state_part_poses(states[min(_l[0] + 1, len(states) - 1)], assembly,
                                lift=CARRY_LIFT)[fr[1]]
        servo.place_seat(q_ref, fr[1], np.asarray(exp, dtype=float), t.name, _drv, _tck,
                         settle_max)
        _hook_log["seat"].append(_cnt[0] * DT)
      return
    if fr[0] not in assembly.objects:
      return                                               # grasps/releases: no datum to seat
    _relative_target = None
    if (_object_perturb_active and object_perturb_observation == "observed"):
      _l = name_of.get(t.name)
      if _l is not None:
        _nom = _state_part_poses(states[min(_l[0] + 1, len(states) - 1)], assembly,
                                 lift=CARRY_LIFT)
        _relative_target = _pose_compose(
            _pose_invert(np.asarray(_nom[fr[0]], dtype=float)),
            np.asarray(_nom[fr[1]], dtype=float))
    servo.seat(q_ref, fr[0], fr[1], t.name, _drv, _tck, settle_max,
               relative_target=_relative_target)
    _hook_log["seat"].append(_cnt[0] * DT)

  def _touch_robots(tids):
    """Acquiring arms whose settle terminates on physical touch.

    Pick and handover both drive a previously bare receiver to an authored grasp pose whose
    fixed-finger proxy may overlap the payload slightly.  The giver in a handover is excluded:
    it is already attached and its ordinary support load is not evidence that the receiver
    has arrived.
    """
    if servo is None or not servo.pick_ok:
      return ()
    out = []
    for tid in tids:
      t = problem.tasks[tid]
      lbl = name_of.get(t.name)
      if lbl is None:
        continue
      action = tuple(plan[lbl[0]])
      if action and action[0] in ("pick", "handover") and action[1] in robots:
        out.append(action[1])
    return tuple(out)

  def _touch_mates(tids):
    """Intended part interfaces allowed to terminate terminal settling on contact."""
    out = []
    for tid in tids:
      fr = getattr(problem.tasks[int(tid)], "frames", None)
      if fr and len(fr) > 1 and fr[0] in assembly.objects and fr[1] in assembly.objects:
        out.append((fr[0], fr[1]))
    return tuple(out)

  def _pick_task(tid):
    """The part a PICK task acquires, or None. Keys on the PLAN ACTION kind, not on
    Task.type: mp_export gives inhand_reorient `type="pick"` too (it re-attaches to the same
    gripper), and running a lift-off on an already-held, in-air part is a spurious 1s stall."""
    t = problem.tasks[tid]
    lbl = name_of.get(t.name)
    if lbl is None or plan[lbl[0]][0] != "pick":
      return None
    fr = getattr(t, "frames", None)
    if not fr or len(fr) < 2 or fr[1] not in assembly.objects:
      return None
    return fr[1]

  _pick_capture = {}
  _object_perturb_active = bool(object_perturbation and
      (object_perturbation.get("xy_bound_m", 0.0) > 0.0 or
       object_perturbation.get("yaw_bound_deg", 0.0) > 0.0))

  def align_pick_before_fire(tid, q_ref, flow_drive=None, flow_tick=None):
    """Retarget a pick to an observed displaced part, then enforce physical capture.

    In an unobserved trial the controller follows the nominal saved path unchanged. We still
    measure the residual here as an outcome gate: MuJoCo weld activation freezes the current
    relative transform and would otherwise attach an arbitrarily distant object.
    """
    if not _object_perturb_active:
      return
    part = _pick_task(tid)
    if part is None:
      return
    t = problem.tasks[tid]
    lbl = name_of.get(t.name)
    if lbl is None:
      return
    act = tuple(plan[lbl[0]])
    robot = act[1]
    nominal_parts = _state_part_poses(states[lbl[0]], assembly, lift=CARRY_LIFT)
    nominal_anchor = fk_poses(np.asarray(q_ref, dtype=float))[ee_anchor[robot]]
    _drv = flow_drive if flow_drive is not None else drive_q
    _tck = flow_tick if flow_tick is not None else (lambda: scene.step(1, report, ("servo",)))
    if servo is not None:
      metrics = servo.align_pick(
          q_ref, robot, part, nominal_parts[part], nominal_anchor, _drv, _tck, settle_max,
          observed=(object_perturb_observation == "observed"))
    else:
      actual_part = np.asarray(scene.part_pose(part), dtype=float)
      delta = _pose_compose(actual_part, _pose_invert(np.asarray(nominal_parts[part], dtype=float)))
      desired = _pose_compose(delta, nominal_anchor)
      body = scene._body_id(scene.grasp_anchor[robot])
      current = np.r_[scene.data.xpos[body], scene.data.xquat[body]]
      metrics = {"observed": False,
                 "position_initial_m": float(np.linalg.norm(desired[:3] - current[:3])),
                 "angle_initial_deg": float(np.rad2deg(_quat_angle(desired[3:], current[3:]))),
                 "position_m": float(np.linalg.norm(desired[:3] - current[:3])),
                 "angle_deg": float(np.rad2deg(_quat_angle(desired[3:], current[3:]))),
                 "duration_s": 0.0, "guarded": False}
    ok = (metrics["position_m"] <= float(object_grasp_capture) and
          metrics["angle_deg"] <= float(np.rad2deg(object_grasp_capture_angle)))
    _pick_capture[tid] = (ok, metrics)
    print(f"[sim/placement] {t.name}: {'observed correction' if metrics['observed'] else 'nominal'} "
          f"pick residual {metrics['position_initial_m']*1000:.1f} -> "
          f"{metrics['position_m']*1000:.1f}mm, {metrics['angle_initial_deg']:.1f} -> "
          f"{metrics['angle_deg']:.1f}deg ({'CAPTURE' if ok else 'MISS'})")

  def lift_after_fire(tid, q_ref, flow_drive=None, flow_tick=None):
    """--servo-pick: own the fresh-weld window AFTER a pick fires -- rise until the new
    cargo is contact-free instead of tracking through the pressing grasp keyframe (see
    _SkillServo.pick_lift). Handovers/places keep their own terminals. `flow_drive`/
    `flow_tick`: FIRE-FLOW pair, as in seat_before_fire."""
    if servo is None or not servo.pick_ok:
      return
    part = _pick_task(tid)
    if part is None:
      return
    holder = next((r for r, g in scene.active_grasps.items() if g[1] == part), None)
    if holder is None:
      return
    _drv = flow_drive if flow_drive is not None else drive_q
    _tck = flow_tick if flow_tick is not None else (lambda: scene.step(1, report, ("servo",)))
    servo.pick_lift(q_ref, holder, part, problem.tasks[tid].name, _drv, _tck, settle_max)

  def servo_summary():
    # called from EVERY executor tail -- the corridor branch returns early, and the first
    # servo build silently missed it: banner printed, loop never consulted the servo, and the
    # recording came out BIT-IDENTICAL to baseline. One summary helper, called from each tail,
    # is the guard against the next execution branch making the same mistake quietly.
    if servo is None:
      return
    if servo.peak:
      print("[sim/mp] --skill-servo admittance peaks: "
            + ", ".join(f"{r} {v*1000:.1f}mm" for r, v in sorted(servo.peak.items())))
    else:
      print("[sim/mp] --skill-servo: admittance never triggered "
            f"(no held-part contact above {servo.f0:g}N).")
    for nm, e0, e1 in servo.seat_log:
      why = " (contact-terminated)" if nm in servo.seat_touch else ""
      print(f"[sim/mp] --skill-servo seat {nm}: {e0*1000:.1f}mm -> {e1*1000:.1f}mm{why}")
    if _hook_log.get("settle"):
      _n = len(_hook_log["settle"])
      _ms = [s[0] for s in _hook_log["settle"]]
      _e0 = [s[1] for s in _hook_log["settle"]]
      _e1 = [s[2] for s in _hook_log["settle"]]
      _nt = sum(1 for s in _hook_log["settle"] if s[3])
      _np = sum(1 for s in _hook_log["settle"] if len(s) > 7 and s[7])
      _nm = sum(1 for s in _hook_log["settle"] if len(s) > 8 and s[8])
      print(f"[sim/mp] settle windows: {_n} fired ({_nt} touch-broken), total "
            f"{sum(_ms)*1000:.0f}ms sim, max {max(_ms)*1000:.0f}ms; "
            f"err {max(_e0)*1000:.1f} -> {max(_e1)*1000:.1f} mrad; "
            f"{_np} small-residual plateau(s) accepted, "
            f"{_nm} mate-contact termination(s)")
      _rmax = {}
      for s in _hook_log["settle"]:
        for r, v in s[4].items():
          _rmax[r] = max(_rmax.get(r, 0.0), v)
      print("[sim/mp] settle err by robot (max over windows, mrad): "
            + ", ".join(f"{r} {v*1000:.1f}" for r, v in sorted(_rmax.items())))
      print(f"[sim/mp] last settle owned: {_hook_log['settle'][-1][5]}")
      _worst = max(_hook_log["settle"], key=lambda s: s[2])
      _jtop = sorted(_worst[6], key=lambda jv: abs(jv[1]), reverse=True)[:10]
      print("[sim/mp] worst-window joints (mrad): "
            + ", ".join(f"{jn} {d*1000:+.1f}" for jn, d in _jtop))
    if _hook_log.get("seat"):
      print(f"[sim/mp] seat-servo windows: {len(_hook_log['seat'])} fired, total "
            f"{sum(_hook_log['seat'])*1000:.0f}ms sim, "
            f"max {max(_hook_log['seat'])*1000:.0f}ms")
    for nm, e0, fN, ok in servo.place_log:
      print(f"[sim/mp] --servo-place-seat {nm}: xy_err {e0*1000:.1f}mm, "
            f"end |F| {fN:.1f}N, {'TOUCHED' if ok else 'no touch (guard/budget)'}")
    for nm, f0_, fN, secs in servo.pick_log:
      print(f"[sim/mp] --servo-pick {nm}: fire |F| {f0_:.1f}N -> {fN:.1f}N "
            f"after {secs*1000:.0f}ms lift")
    if servo.wrench_holds:
      print(f"[sim/mp] --servo-torque-guard: {servo.wrench_holds} tick(s) engaged by the "
            "normalized FORCE/TORQUE grasp-wrench trigger")

  # NO-FUNNEL WARNING. An assemble whose export is a single task drives the held part STRAIGHT to
  # a pose that is already in contact, so it arrives LATERALLY and sweeps the target: measured on
  # cross8, 0.213mm of penetration while the target was still at rest, then ~800 steps of shoving
  # and a weld fired at 32mm lateral error (Z near-perfect) -> every downstream mate missed by
  # 130-1900mm. --pre-insertion splits it into a goto to a funnel mouth along the mate axis plus
  # the real mate, which is the difference between "approaches down the axis" and "swipes at it".
  # Checked on the EXPORTED TASKS, not on cfg, so it also catches a saved path whose funnel was
  # dropped or whose pre-witness solve failed.
  _asm_idx = [i for i, a in enumerate(plan) if tuple(a)[0] == "assemble"]
  _funnelled = set()
  for _t in problem.tasks:
    _nm = getattr(_t, "name", "") or ""
    if "_pre_assemble" in _nm:
      try:
        _funnelled.add(int(_nm.split("_", 1)[0]))
      except ValueError:
        pass
  _bare = [i for i in _asm_idx if i not in _funnelled]
  if _bare:
    print(f"[sim] !!! NO INSERTION FUNNEL on {len(_bare)}/{len(_asm_idx)} assemble(s) "
          f"(plan steps {_bare[:8]}{'...' if len(_bare) > 8 else ''}): the part is driven "
          f"straight to a seated pose that is already in contact, so it arrives laterally and "
          f"can shove the target. Re-export with --pre-insertion (and --fuse-pre-skill) if a "
          f"mate misses or the target slides.")

  _last_q = {}

  _armed = {}

  def drive_q(q):
    """One driver for every arm model, so the executor does not care which is in use."""
    if actuated:
      if not _armed:
        # SEED THE STATE. The model is generated at q=0 and an actuated arm is only ever given
        # a ctrl TARGET, so without this it starts ~2 rad from the trajectory and spends the
        # first actions swinging into place (measured: 1.7-2.9 rad tracking error, seats missed
        # by up to 882mm). The kinematic modes never showed this because they write qpos.
        scene.set_arm_q(q)
        scene._cmd_q = None                                # servos own the arm from here on
        _armed["yes"] = True
      q = np.asarray(q, dtype=float)
      prev = _last_q.get("q")
      scene.command_arm_q(q, qdot=None if prev is None else (q - prev) / DT)
      scene.gravity_compensate()
      _last_q["q"] = q.copy()
    elif articulated:
      q = np.asarray(q, dtype=float)
      prev = _last_q.get("q")
      scene.set_arm_q(q, qdot=None if prev is None else (q - prev) / DT)
      _last_q["q"] = q.copy()
    else:
      scene.sync_arms(fk_poses(q))

  def _command_gripper_targets(targets):
    """Schedule articulated finger motion alongside the existing arm trajectory."""
    targets = {r: float(v) for r, v in targets.items()
               if scene._gripper_qadr.get(r)
               and abs(scene.gripper_aperture(r) - float(v)) > 1e-6}
    for r, target in targets.items():
      scene.command_gripper_aperture(r, target, gripper_motion_time)

  def _closing_aperture(robot, child):
    """Object-dependent aperture from the same geometry routine as the closure gate."""
    if (gripper_kinds or {}).get(robot, "vacuum") == "vacuum":
      return 0.0
    target = child
    try:
      target, _gid = _held_grasp_id(scene.expect_state, child, robot=robot)
    except (TypeError, ValueError):
      pass
    try:
      Cc = ry.Config()
      Cc.addConfigurationCopy(C_fk)
      Cc.setJointState(np.asarray(_last_q.get("q", q0), dtype=float))
      # Use the actual target pose at capture time. The contact-aperture solve ignores every
      # non-target object, so this does not make gripper animation depend on scene clutter.
      _pose = scene.part_pose(target)
      Cc.getFrame(target).setPosition(_pose[:3]).setQuaternion(_pose[3:])
      from gripper_closure import contact_aperture
      result = contact_aperture(Cc, robot + "_", target)
      if result.feasible and result.aperture is not None:
        return scene.visual_contact_aperture(robot, target, float(result.aperture))
      reason = result.reason or "no aperture"
    except Exception as exc:
      reason = f"{type(exc).__name__}: {exc}"
    # This flag is a renderer feature, not a new feasibility gate. Keep old plans viewable,
    # but make approximation unmistakable rather than silently claiming exact contact.
    fallback = 0.5 * (0.045 if scene._gripper_model.get(robot) == "openarm" else 0.042)
    print(f"[sim/gripper] WARNING: could not infer {robot}->{target} aperture ({reason}); "
          f"rendering a {fallback*1000:.1f}mm half-stroke close.")
    return scene.visual_contact_aperture(robot, target, fallback)
  rec = None
  if record is not None:
    rec = scene.recorder = {
        "qpos": [], "mpos": [], "mquat": [], "forces": [], "marks": [],
        "controller_mode": [], "physics_tick": [], "gripper_events": []}
  report = SimReport()

  servo = None
  if skill_servo:
    if not actuated:
      print("[sim/mp] --skill-servo needs --actuated (stiff modes have no tracking error to "
            "yield with); IGNORED.")
    else:
      servo = _SkillServo(scene, DT, force=servo_force, max_dev=servo_max_dev, tol=servo_tol,
                          torque_guard=servo_torque_guard, compliant=compliant_follow,
                          compliant_force=compliant_force)
      servo.place_ok = bool(servo_place_seat)
      servo.pick_ok = bool(servo_pick)
      if servo_pick:
        print("[sim/mp] --servo-pick ACTIVE: pick/handover acquisition settles terminate on "
              "touch; after each pick fire the arm RISES until the fresh cargo is contact-"
              "free instead of tracking through the pressing grasp keyframe.")
      if compliant_follow:
        print(f"[sim/mp] --compliant-follow ACTIVE: whole-branch admittance -- arm links, "
              f"bare gripper AND cargo forces all yield the reference above "
              f"{compliant_force:g}N, everywhere along the path (guard/funnel keep engaging "
              f"at {servo_force:g}N as before).")
      if servo_place_seat:
        print("[sim/mp] --servo-place-seat ACTIVE: places terminate by CONTACT (task-space "
              "XY-to-anchor + bounded descent until sustained touch) instead of tracking the "
              "joint reference's z, which is authored against the planner's lower table.")
      if servo_torque_guard is not None:
        _guard_action = ("local compliant yield engages; the independently scheduled arms "
                         "keep flowing" if phase_schedule else "hold + yield engage")
        print(f"[sim/mp] --servo-torque-guard {servo_torque_guard:g}: contact-conditioned "
              f"grasp-wrench guard -- during external contact, {_guard_action} when FORCE "
              f"or TORQUE reaches {servo_torque_guard:.0%} of its scalar breakaway cap "
              "(static gravity alone never freezes transport).")
      print(f"[sim/mp] --skill-servo ACTIVE: held-part admittance (yield above "
            f"{servo_force:g}N, bounded {servo_max_dev*1000:g}mm, force-triggered so intended "
            f"contacts still happen) + terminal seat servo to the mate datum before each "
            f"part-part weld (tol {servo_tol*1000:g}mm). The reference DEVIATES from the "
            f"certified path by up to the bound -- that is the point.")

  # task name -> (plan index, is-final-task-of-that-action)
  name_of = {}
  for i, nms in enumerate(problem.step_names):
    for k, nm in enumerate(nms):
      name_of[nm] = (i, k == len(nms) - 1)

  try:
    process_realizations = _process_realizations(
        plan, mode=process_noise_mode, corner=process_noise_corner,
        seed=process_noise_seed)
  except ValueError as e:
    raise SystemExit(f"[sim/process-wrench] {e}")
  if record_metadata is not None and apply_process_wrench:
    import wrench as _Wproc
    _unc = _Wproc.PARAMS.get("insertion_uncertainty") or {}
    record_metadata["process_wrench"] = {
        "noise_mode": process_noise_mode,
        "noise_corner": int(process_noise_corner),
        "noise_seed": int(process_noise_seed),
        "lateral": float(_unc.get("lateral", 0.0) or 0.0),
        "offset": float(_unc.get("offset", 0.0) or 0.0),
        "engagement_distance": float(process_engagement_distance),
        "ramp_time": float(process_wrench_ramp),
        "realizations": {str(i): v for i, v in process_realizations.items()},
    }

  import os as _os
  _trace = bool(_os.environ.get("SIM_MP_TRACE"))

  def fire(task_id):
    # REPLAY mr_bench's exact attach for this task (frames = [parent, child]); goto = no attach.
    # This -- not action-kind inference -- is what a saved PATH requires: the path was planned
    # under these attaches (release re-roots to table or to the continuing holder; hold-through
    # transfers are just an attach to the other gripper), so mirroring the frames keeps the
    # welds consistent with the motion and the arm never drags a part it "released".
    t = problem.tasks[task_id]
    # The discrete state this task lands in -- attach() reads it to recover a grasp id so the
    # breakaway can use the planner's DIRECTIONAL caps instead of the conservative scalar.
    _l = name_of.get(t.name)
    scene.expect_state = states[min(_l[0] + 1, len(states) - 1)] if _l else None
    _capture = _pick_capture.pop(task_id, None)
    if _capture is not None and not _capture[0]:
      act = tuple(plan[_l[0]]) if _l else ("pick",)
      m = _capture[1]
      detail = (f"terminal grasp residual {m['position_m']*1000:.1f}mm/"
                f"{m['angle_deg']:.1f}deg exceeds capture "
                f"{object_grasp_capture*1000:.1f}mm/"
                f"{np.rad2deg(object_grasp_capture_angle):.1f}deg")
      print(f"[sim] !!! {t.name}: {detail}; grasp weld NOT activated")
      report.results.append(ActionResult(act, False, "not_grasped", detail))
      if rec is not None:
        rec["marks"].append((len(rec["qpos"]), repr(act) + " MISS"))
      return
    if t.name == "terminal" or not getattr(t, "frames", None):
      return                                               # goto / terminal: nothing to relink
    parent, child = t.frames[0], t.frames[1]
    _before_grasps = {r: g[1] for r, g in scene.active_grasps.items()}
    _incoming = (None if parent == "table" or parent in assembly.objects
                 else parent.split("_", 1)[0])
    if rec is not None and _incoming is not None:
      # Logical hardware event, independent of whether --animate-grippers adds simulation-only
      # finger joints.  It occurs at the same instant the grasp/transfer task fires.  Preserve
      # the exact physics time as well as the next recording-frame index: RECORD_EVERY may be
      # greater than one, so a frame index alone loses sub-frame event timing.
      rec["gripper_events"].append({
          "event": "close_start", "robot": _incoming, "part": child,
          "physics_tick": int(scene._phys), "time_s": float(scene._phys * DT),
          "frame": int(len(rec["qpos"])), "animated": bool(animate_grippers)})
    if animate_grippers and _incoming is not None:
      _command_gripper_targets({_incoming: _closing_aperture(_incoming, child)})
    # seat check BEFORE the datum weld (part-part attach only) -- report a genuine miss.
    seat_miss = None
    if parent in assembly.objects:
      got = _rel_pos(scene.part_pose(parent), scene.part_pose(child))
      err = float(np.linalg.norm(got - np.asarray(assembly.relative_pose(child)[1])))
      if err > SEAT_POS_TOL:
        seat_miss = err
    # only DATUM-WELD when the part is within CAPTURE range -- snapping a far part in would
    # fire a huge constraint impulse (measured 4000N) and explode the scene. But the KINEMATIC
    # side of the attach must ALWAYS be replayed: after this transition the path assumes the
    # gripper is free (mr_bench re-parents the part), so the grasp weld must go even on a
    # missed seat -- keeping it made the arm keep holding through the "assemble" and shove the
    # target around (user-reported: 'not releasing when the assembly should happen').
    if seat_miss is not None and seat_miss > CAPTURE_RANGE:
      scene._detach(child)                                 # gripper lets go; no datum snap
      # ALWAYS loud (user ask): a skipped datum weld must be unmistakable, not a trace line.
      print(f"[sim] !!! {t.name}: {child} MISSED its seat on {parent} by "
            f"{seat_miss*1000:.0f}mm (> {CAPTURE_RANGE*1000:.0f}mm capture range) -- mate NOT "
            f"welded, gripper released; expect downstream failures.")
    else:
      scene.attach(parent, child)
    # Transfers close the receiver before the weld changes hands, then open every giver.
    # Releases and mates simply open the former holder(s). This is a LOGICAL hardware event
    # even when finger animation is disabled; older code accidentally omitted every open event
    # from ordinary standard-profile recordings because it logged them inside the visual flag.
    _open = {r: 0.0 for r, p in _before_grasps.items()
             if p == child and r not in scene.active_grasps}
    if rec is not None:
      for _r in sorted(_open):
        rec["gripper_events"].append({
            "event": "open_start", "robot": _r, "part": child,
            "physics_tick": int(scene._phys), "time_s": float(scene._phys * DT),
            "frame": int(len(rec["qpos"])), "animated": bool(animate_grippers)})
    if animate_grippers:
      _command_gripper_targets(_open)
    # Every framed task is the physical side effect of its plan action. Usually that is the
    # final split task; --post-release deliberately appends a frame-less departure AFTER the
    # detach, so requiring `is-final` here would lose the release result entirely.
    lbl = name_of.get(t.name)
    act = tuple(plan[lbl[0]]) if lbl else ("attach", parent, child)
    if _trace:
      pp = scene.part_pose(child)[:3]
      pj = scene.part_pose(parent)[:3] if parent in assembly.objects else (
          np.asarray(C_fk.getFrame(ee_anchor.get(parent.split("_", 1)[0], "")).getPose()[:3])
          if parent != "table" else None)
      d = float(np.linalg.norm(pp - pj)) * 1000 if pj is not None else -1
      print(f"  [fire] task{task_id} {t.name} attach[{parent}<-{child}] "
            f"child-to-parent={d:.0f}mm seat_miss={seat_miss}")
    if rec is not None:
      rec["marks"].append((len(rec["qpos"]), repr(act)))
    if seat_miss is not None:
      report.results.append(ActionResult(act, False, "not_seated", f"{seat_miss*1000:.1f}mm"))
    elif lbl:
      report.results.append(_judge_resting(scene, assembly, act, states[lbl[0] + 1]))

  if verbose:
    print(f"[sim/mp] driving RAW path: {len(mp_path)} states, {len(problem.tasks)} tasks")
  q_prev = full_q(mp_path[0].q)
  drive_q(q_prev)

  # --- split the raw path into PER-MODE STRETCHES (transition = duplicated goal config) ---
  # Decomposition + the trapezoidal profile now live in the portable `retiming` module
  # (input = sequence/DAG + collision-free path, output = trajectory with v=0 at goals),
  # so the same contract can move to the mr_bench side. `st.mode` is passed as the mode
  # handle (compared with !=); full_q maps a path state's q to the full joint config.
  stretches = retiming.split_stretches(                     # [(full-q waypoints, done tids)]
      mp_path, q_of=lambda st: full_q(st.q),
      mode_of=lambda st: st.mode,
      taskids_of=lambda m: np.atleast_1d(m.task_ids))

  def _process_pairs(done_ids):
    """Mate loads owned by the stretch ending in ``done_ids`` (pre-skill GOTOs own none)."""
    if not apply_process_wrench:
      return []
    out = []
    for tid in done_ids:
      task = problem.tasks[int(tid)]
      lbl = name_of.get(task.name)
      if lbl is None or not lbl[1]:                         # pre-assemble / unnamed task
        continue
      action = tuple(plan[lbl[0]])
      if action[0] == "assemble":
        out.append((action[2], action[3], process_realizations.get(lbl[0], {})))
    # A barrier can list the same completed task for several robots. Deduplicate by mate,
    # retaining its plan-indexed realization (dicts themselves are deliberately unhashable).
    unique = {}
    for entry in out:
      unique[(entry[0], entry[1])] = entry
    return list(unique.values())

  _active_process_pairs = [None]
  def _set_process_for(done_ids):
    pairs = _process_pairs(done_ids)
    if pairs == _active_process_pairs[0]:
      return
    scene.set_process_wrenches(pairs)
    _active_process_pairs[0] = pairs
    if pairs:
      import wrench as W
      desc = []
      for part, parent, realization in pairs:
        skill = assembly.screw_skill(part)
        tau = 0.0 if skill is None else float(skill.torque)
        c = np.asarray(realization.get("coefficients", np.zeros(4)), dtype=float)
        noise = ("" if not np.any(c) else
                 f" noise=[{','.join(f'{x:+.2f}' for x in c)}]")
        desc.append(f"{part}->{parent} {W.insert_force(part):g}N/{tau:g}Nm{noise}")
      print("[sim/process-wrench] ACTIVE over entry-to-seat stretch: " + ", ".join(desc))

  if apply_process_wrench:
    print("[sim/mp] --apply-process-wrench ACTIVE: nominal equal-and-opposite mate loads "
          "are applied directly at both interface bodies; ordinary insertions engage over "
          f"the final {process_engagement_distance*1000:g}mm, screws remain loaded over the "
          f"helix, and turn-on is ramped over {process_wrench_ramp*1000:g}ms. Holders, grasp "
          "welds, and arm servos must physically react them. "
          f"Insertion realization={process_noise_mode}. Relative noise coefficients are "
          "fixed per action and persisted in the NPZ.")
  # FIRE-COVERAGE GUARD: every task's completion must land in some stretch's done set -- a
  # task in none of them simply never fires, and its attach/release side effects are skipped
  # SILENTLY (the dropped-holder-release failure class: the arm keeps its weld and drags the
  # assembly). repair_mode_chain heals the known cause upstream at the load/plan sites, so
  # anything still missing here is real path corruption: stop, do not execute.
  _fired_ids = set()
  for _w, _tids in stretches:
    _fired_ids.update(int(t) for t in _tids)
  _unfired = sorted(set(range(len(problem.tasks))) - _fired_ids)
  if _unfired:
    raise RuntimeError(
        f"[sim/mp] fire-coverage: task id(s) {_unfired} "
        f"({', '.join(problem.tasks[t].name for t in _unfired)}) complete at NO stretch "
        f"boundary -- their attach events would never fire. The path is missing their modes "
        f"and repair could not restore them; re-export the path for this plan.")

  # A shortcut path stores every collision-check interpolation sample. They certify the edge,
  # but they are not geometric knots: feeding all of them to Catmull-Rom multiplies spline
  # certification work and makes the simulator depend on serialization density. Compact each
  # MODE independently, so every task boundary and its exact witness remain pinned.
  stretches, _compact = retiming.compact_stretches(stretches)
  if _compact["removed"]:
    _drift = abs(_compact["arc_after"] - _compact["arc_before"])
    print(f"[sim/mp] path compaction: {_compact['before']} stretch waypoint(s) -> "
          f"{_compact['after']} ({_compact['removed']} redundant interpolation samples "
          f"removed); arc {_compact['arc_after']:.2f} rad, change {_drift:.2e}. "
          f"Mode barriers and witness configurations preserved exactly.")

  # CONTROLLER-OWNED SCREW STRETCHES.  The global planner only has to reach the persisted
  # helical entry keyframe.  For the following assemble stretch, replace whatever free-space
  # detour BiRRT happened to return with the warm-started, single-grasp IK continuation that
  # was certified and embedded in the final export.  This does NOT add task actions or mode
  # switches: entry and final datum remain the two ordinary pre-assemble/assemble barriers.
  # The strict preflight immediately below collision-checks every interpolated edge of this
  # replacement against the exact pre-attach mode, so a bad skill certificate fails before
  # physics rather than silently falling back to the arbitrary planner segment.
  _screw_stretches = []
  for _k, (_wp, _done) in enumerate(stretches):
    _skills = [(int(tid), getattr(problem.tasks[int(tid)], "skill_data", None))
               for tid in _done
               if getattr(problem.tasks[int(tid)], "skill_data", None)]
    _skills = [(tid, sd) for tid, sd in _skills if sd.get("kind") == "screw"]
    if not _skills:
      continue
    if len(_skills) != 1:
      raise RuntimeError(f"[sim/mp] stretch {_k} completes multiple screw skills; v1 requires "
                         "one atomic single-grasp skill per stretch")
    _tid, _sd = _skills[0]
    _task = problem.tasks[_tid]
    _jp = [np.asarray(q, dtype=float).ravel() for q in (_sd.get("joint_path") or [])]
    if len(_jp) < 3 or any(q.shape != np.asarray(_wp[0]).shape for q in _jp):
      raise RuntimeError(f"[sim/mp] {getattr(_task, 'name', _tid)} has a malformed screw "
                         "joint-path certificate")
    _active = sorted({j for r in _task.robots for j in ridx[r]})
    _inactive = [j for j in range(len(_jp[0])) if j not in set(_active)]
    _w0, _w1 = np.asarray(_wp[0], dtype=float), np.asarray(_wp[-1], dtype=float)
    if (np.max(np.abs(_jp[0][_active] - _w0[_active])) > 2e-3 or
        np.max(np.abs(_jp[-1][_active] - _w1[_active])) > 2e-3):
      raise RuntimeError(
          f"[sim/mp] {getattr(_task, 'name', _tid)} screw certificate endpoints do not "
          "match the saved motion path's entry/final task goals; regenerate the MP export")
    # Idle full-q slices in an IK certificate are implementation detail (typically EXIT).
    # Preserve unrelated robots' ORIGINAL planner geometry while replacing only task-owned
    # joints.  A long multi-robot plan may advance another robot in the same mode stretch;
    # requiring it to be stationary made valid atomic skills spuriously unexecutable.  The
    # merged knots preserve both projected polylines exactly.  Their synchronization is new,
    # so the strict interpolated-edge preflight below is the actual safety certificate (and
    # fails rather than falling back if the composition is unsafe).
    _owned = retiming.compose_owned_path(_wp, _jp, _active)
    _inactive_motion = (np.max(np.abs(_w1[_inactive] - _w0[_inactive]))
                        if _inactive else 0.0)
    stretches[_k] = (_owned, _done)
    _screw_stretches.append((_k, _tid, _sd, _inactive_motion))
  for _k, _tid, _sd, _inactive_motion in _screw_stretches:
    _arc = retiming.arc_length(stretches[_k][0])[1]
    _angle = abs(float(_sd["angle"]))
    _omega = float(_sd["angular_speed"])
    # The IK certificate is parameterized linearly in screw angle. Convert requested object
    # angular cruise into the equivalent composite joint-arc cruise, capped by the ordinary
    # scene cruise. Acceleration ramps remain the standard smooth retimer's responsibility.
    _rate = min(JOINT_RATE, _arc * _omega / max(_angle, 1e-12))
    _sd["retime_rate"] = max(_rate, 1e-6)
    print(f"[sim/mp] screw controller owns stretch {_k} ({problem.tasks[_tid].name}): "
          f"{len(_sd['joint_path'])} certified single-grasp helix knots, "
          f"angle {np.degrees(float(_sd['angle'])):.1f}deg, depth "
          f"{float(_sd['depth'])*1000:.1f}mm, torque {float(_sd['torque']):.3f}Nm. "
          f"Angular cruise {float(_sd['angular_speed']):.3f}rad/s -> joint-arc cruise "
          f"{_sd['retime_rate']:.3f}rad/s. The free-space planner segment is ignored; "
          f"unrelated-arm endpoint motion {_inactive_motion:.3g}rad follows its saved "
          "planner polyline; strict composed-path preflight follows.")
  _stretch_rates = [float(JOINT_RATE)] * len(stretches)
  for _k, _tid, _sd, _inactive_motion in _screw_stretches:
    _stretch_rates[_k] = float(_sd["retime_rate"])

  # ALIGNMENT BY MODE IDENTITY: the saved path's lightweight modes carry task_ids only, so
  # match each per-stretch path mode to the rebuilt chain mode with the SAME active-task set
  # (unique in a sequence problem -- progression is monotone). This is needed by both the
  # strict base-path preflight and corridor construction.
  stretch_mode = None
  _conf = None
  if corridor:
    import mp_export as _MP
    chain = _MP.mode_sequence(problem)
    _ids = lambda m: frozenset(int(t) for t in np.atleast_1d(m.task_ids))
    chain_by_ids = {_ids(m): m for m in chain}
    path_mode_ids, pm = [], mp_path[0].mode
    for st in mp_path:
      if st.mode != pm or not path_mode_ids:
        pm = st.mode
        path_mode_ids.append(_ids(pm))
    missing = [i for i in path_mode_ids if i not in chain_by_ids]
    if len(path_mode_ids) != len(stretches) or missing:
      raise RuntimeError(f"[sim/mp] mode alignment failed: {len(path_mode_ids)} path modes vs "
                         f"{len(stretches)} stretches, {len(missing)} unmatched in the rebuilt "
                         f"{len(chain)}-mode chain -- saved path and rebuilt problem disagree; "
                         f"re-export with matching flags.")
    stretch_mode = [chain_by_ids[i] for i in path_mode_ids]
    conf_type = type(problem.get_start_pos())
    def _conf(fq):
      return conf_type.from_list([np.asarray(fq)[ridx[r]] for r in robots])

    # STRICT BASE-PATH PREFLIGHT. Retiming can only change WHEN a configuration is visited; it
    # cannot repair a collision or an out-of-range joint already present on the diagonal. Do
    # this before spline/corridor work so a relaxed planner path fails in seconds with the
    # concrete mode/edge/pair instead of spending minutes in futile corridor refinement.
    _ptol = float(path_preflight_tolerance)
    if path_preflight:
      _pres = (float(path_preflight_resolution) if path_preflight_resolution is not None else
               float(getattr(problem, "collision_resolution", 0.01)))

      def _strict_free(fq, mode):
        ok = bool(problem.is_collision_free(
            _conf(fq), mode, collision_tolerance=_ptol))
        if ok:
          return True, None
        rejection = (problem.last_margin_rejection()
                     if callable(getattr(problem, "last_margin_rejection", None)) else None)
        try:
          pen = float(problem.C.getCollisionsTotalPenetration())
          pairs = sorted(
              [(str(c[0]), str(c[1]), float(c[2]))
               for c in problem.C.getCollisions(0.0) if float(c[2]) < 0.0],
              key=lambda x: x[2])[:5]
        except Exception:
          pen, pairs = float("nan"), []
        return False, {"penetration": pen, "pairs": pairs, "rejection": rejection}

      _limits = getattr(problem, "limits", None)
      if _limits is None:
        _limits = C_fk.getJointLimits()
      _audit = retiming.audit_path(
          stretches, stretch_mode, _strict_free, limits=_limits, resolution=_pres)
      if not _audit["ok"]:
        _mids = sorted(_ids(stretch_mode[_audit["stretch"]]))
        _where = (f"stretch {_audit['stretch']} mode task_ids={_mids}, "
                  f"edge {_audit['edge']} at {_audit['alpha']:.3f}")
        if _audit["kind"] == "joint_limits":
          rows = []
          for j, lo, hi in zip(_audit["joint_indices"], _audit["lower"], _audit["upper"]):
            nm = jnames[j] if j < len(jnames) else f"joint[{j}]"
            rows.append(f"{nm}={_audit['q'][j]:.6g} outside [{lo:.6g},{hi:.6g}]")
          raise PathPreflightError(
              f"[sim/mp] strict base-path preflight FAILED at {_where}: joint limit "
              f"violation(s): {'; '.join(rows)}. Spline/corridor construction was not "
              f"started; regenerate the motion path.")
        detail = _audit.get("detail") or {}
        rejection = detail.get("rejection") or {}
        pairs = detail.get("pairs") or []
        pair_msg = (", ".join(f"{a}~{b} {-d*1000:.2f}mm" for a, b, d in pairs)
                    if pairs else "no concrete penetrating pair reported")
        if rejection.get("layer") == "gate":
          reason = rejection.get("reason") or {}
          why = (f"planning gate {reason.get('gate', 'unknown')} rejected "
                 f"{reason.get('frames', ('?', '?'))[0]}~"
                 f"{reason.get('frames', ('?', '?'))[1]}: observed "
                 f"{float(reason.get('observed', float('nan')))*1000:.3f}mm, required "
                 f"{float(reason.get('required', float('nan')))*1000:.3f}mm")
        else:
          why = (f"total penetration {detail.get('penetration', float('nan'))*1000:.3f}mm "
                 f"exceeds {_ptol*1000:.3f}mm; {pair_msg}")
        raise PathPreflightError(
            f"[sim/mp] strict base-path preflight FAILED at {_where}: {why}. Checked "
            f"{_audit['samples']} sample(s) at {_pres:g}rad resolution. Spline/corridor "
            f"construction was not started; re-plan this edge instead of retiming or "
            f"relaxing it.")
      print(f"[sim/mp] strict base-path preflight: {_audit['samples']} sample(s) at "
            f"{_pres:g}rad resolution, collision tolerance {_ptol:g}, joint limits OK.")

  arm_dims = [np.asarray(ridx[r], int) for r in robots]
  # --arm-rate: JOINT_RATE is nominally a PER-ARM joint speed, but it is applied to the
  # COMPOSITE arc, so with k arms moving together each one only advances at |dq_arm|/|dq| of
  # it (measured on car-mini: 0.61/0.51/0.41 -- arms at half their intended speed for a purely
  # parametric reason). The boost restores the intended semantics. On the COMMON clock this
  # is a time-only rescaling: identical configurations, so the planner's collision
  # certificate carries over untouched and no extra checking is needed.
  boost_dims = arm_dims if rate_boost else None
  boost_vmax = None
  if limit_rate:
    # --limit-rate: saturate the BINDING JOINT against rai's real per-joint limit instead of the
    # nominal 1.0 rad/s. Time-only rescaling on the common clock, so the certificate carries
    # over; it does NOT bound grasp wrenches or tracking error, which is why --force-limits
    # still applies its own per-segment vmax on top.
    _vm = joint_vmax(C)
    _lb = retiming.joint_limit_boost(list(retiming._composite(stretches)[0]), _vm,
                                     joint_rate=JOINT_RATE)
    boost_vmax = _vm                                       # <- APPLIED below, not just printed
    _fin = _vm[np.isfinite(_vm)]
    print(f"[sim/mp] --limit-rate ACTIVE: binding-joint boost x{_lb.mean():.2f} mean "
          f"(min x{_lb.min():.2f}, max x{_lb.max():.2f}) against per-joint limits "
          f"{_fin.min():.2f}-{_fin.max():.2f} rad/s vs JOINT_RATE {JOINT_RATE:g}. "
          f"Same path, time-rescaled only -- expect a LARGER tracking envelope and watch for "
          f"grasp_broke; --force-limits still caps each segment.")
  if rate_boost:
    _b = retiming.arm_rate_boost(list(retiming._composite(stretches)[0]), arm_dims)
    print(f"[sim/mp] --arm-rate ACTIVE: per-arm rate boost x{_b.mean():.2f} mean "
          f"(max x{_b.max():.2f}, cap x{np.sqrt(len(robots)):.2f}) -- same path, "
          f"time-rescaled only; watch for grasp_broke at the higher speeds.")

  # (drive_stretch is GONE: the lockstep path is now built as data by retiming.build_trajectory
  # -- same retime_stretch, same dwell placement -- and executed by THE ONE loop below, so the
  # servo/settle/seat/trace hooks exist in exactly one place.)

  if corridor:
    # DESYNCED, EVENT-DRIVEN drive: only the arm(s) ACTING at a barrier ramp to rest there;
    # uninvolved arms cross without stopping. (The uncertified `--prioritized` variant of this
    # is GONE -- the corridor is its certified successor and measured the same physics, so one
    # preprocessing mode -> one trajectory; everything downstream is shared.)
    robot_index = {r: i for i, r in enumerate(robots)}
    def holders_of(state, part):                             # arms grasping part's group
      grp = next((g for g in state.subassemblies() if part in g), {part})
      return {c[1] for p in grp for c in state.supports.get(p, ()) if c[0] == "grasp"}
    def acting_for(done_ids):                                # arms that must be at rest here
      # An event needs at rest not only the ACTING arm but every arm HOLDING a part it
      # touches: a mate onto obj_0 needs obj_0's holder parked at its witness too, else the
      # target drifts off and the seat is missed (car body held while wheels go on). The
      # composite path has every arm at its witness config at each barrier, so pinning the
      # holder here places it exactly on the mate pose.
      arms_set = set()
      for tid in done_ids:
        t = problem.tasks[int(tid)]
        lbl = name_of.get(t.name)
        # A pre-skill GOTO is a geometric funnel-mouth waypoint, not an execution event: it
        # changes no attachment and has no servo/fire hook. Treating every named subtask as
        # owned made the actor brake to rest there, then perform the tiny final approach as a
        # second motion (most visibly a1's first peg-stool pick). Only an attaching task or the
        # action's final task is a physical rest rendezvous. Unowned GOTO barriers remain exact
        # path/event anchors but crossing robots may retain nonzero velocity through them.
        # A pre-HANDOVER is the exception: it is a two-arm rendezvous at which the giver is
        # in position and the receiver is still clear of the payload. If it is crossed like
        # an ordinary waypoint, the faster receiver can sit inside the payload while waiting
        # for the giver (measured Husky Cross: 4.18mm / 76N before the ownership switch).
        _pre_handover = bool(
            lbl and not lbl[1] and tuple(plan[lbl[0]])[0] == "handover"
            and "_pre_handover_" in (getattr(t, "name", "") or ""))
        _physical_event = (bool(getattr(t, "frames", None)) or bool(lbl and lbl[1])
                           or _pre_handover)
        if lbl and _physical_event:
          pre = states[lbl[0]]                               # state entering the action
          for tok in plan[lbl[0]]:
            if tok in robot_index:                           # the acting robot(s)
              arms_set.add(robot_index[tok])
            elif tok in assembly.objects:                    # holders of every touched part
              for r in holders_of(pre, tok):
                if r in robot_index:
                  arms_set.add(robot_index[r])
        for fr in (getattr(t, "frames", None) or []):        # fallback: gripper-frame owner
          if isinstance(fr, str):
            for r in robots:
              if fr.startswith(r):
                arms_set.add(robot_index[r])
      return arms_set
    acting = [acting_for(done) for _, done in stretches]
    # COLLISION ORACLE: mr_bench's own is_collision_free (the check BiRRT certified the path
    # with), queried at DESYNCED configs against the mode in force after `fired` events --
    # attachments are a function of events fired, not arm positions, so mode_sequence[fired]
    # is the exact world state. A crossing arm whose next step would collide (e.g. sweep a
    # region where a part still sits, or another arm's lane) HOLDS until the world clears.
    def _pair_pens(qc, m):                                   # (planner ok?, {pair: depth>0})
      # `depth` is measured against the MARGIN, not against contact: depth = margin - gap, so a
      # pair merely CLOSER than the margin already scores > 0 and a penetrating one scores
      # margin + |penetration|. At margin=0 this is exactly the old "penetration only" measure
      # (byte-identical), and because the per-pair allowances are CALIBRATED with the same
      # function over the certified diagonal, the margin shifts both sides together: the rule
      # stays "never come closer than the certified path itself did", now with `margin` of
      # extra room wherever that path had room to give.
      ok = bool(problem.is_collision_free(qc, m))            # side effect: C <- (mode, q)
      out = {}
      for c in problem.C.getCollisions(corridor_margin):
        d = float(c[2])
        dep = corridor_margin - d
        if dep > 0.0:
          k = (c[0], c[1]) if c[0] < c[1] else (c[1], c[0])
          out[k] = max(out.get(k, 0.0), dep)
      return ok, out
    # CALIBRATE per-pair thresholds from the CERTIFIED diagonal: the planner's 0.01 total-
    # penetration tolerance is a witness allowance, not a license -- desynced arms may grind
    # ~10mm into each other while is_collision_free still says yes (measured: exactly the
    # earlier false deadlock). Block instead any candidate whose per-pair penetration exceeds
    # what the saved path itself ever exhibits for that pair (+1mm sampling margin): arm-arm
    # pairs never touch on the diagonal, so desync stops at FIRST contact, and the laggard's
    # catch-up step (reducing overlap) stays legal -> tight formations reconverge on their own.
    # DENSELY SAMPLED: BiRRT certifies the INTERPOLATED path, not just the saved waypoints --
    # waypoint-only calibration blocked the fused diagonal itself mid-segment (measured: a
    # fully fused 3-arm group froze at s=31.1 with no outside blocker).
    # MODE-LOCAL allowances (key = (stretch, pair)): a pair's certified proximity in one
    # mode neither licenses nor is judged by another -- global maxima let a blunt fallback
    # to the planner's TOTAL-penetration bool block desynced configs whose per-pair pens
    # were each certified somewhere (measured: cross8 froze at fired=32 on exactly that).
    _CAL_DS = 0.02                                           # rad between calibration samples
    allowed = {}
    for k_st, (wps, _) in enumerate(stretches):
      m_st = stretch_mode[k_st]
      for wa, wb in zip(wps, wps[1:]):
        wa, wb = np.asarray(wa, float), np.asarray(wb, float)
        nseg = max(1, int(np.ceil(float(np.linalg.norm(wb - wa)) / _CAL_DS)))
        for ti in range(nseg + 1):
          fq = wa + (wb - wa) * (ti / nseg)
          for pr, dep in _pair_pens(_conf(fq), m_st)[1].items():
            key = (k_st, pr)
            allowed[key] = max(allowed.get(key, 0.0), dep)
    if allowed:
      wp = max(allowed.items(), key=lambda kv: kv[1])
      print(f"[sim/mp] diagonal calibration: {len(allowed)} (stretch, pair) allowance(s), "
            f"worst {wp[0][1][0]}~{wp[0][1][1]} {wp[1]*1000:.1f}mm (stretch {wp[0][0]}) -- "
            f"desync blocked beyond the certified per-mode penetration")
    # ZERO. A desynced configuration may not be ANY deeper than the nominal path already is.
    # This was 1mm of "waypoint-sampling slack" on the diagonal calibration, and slack here is
    # not free: the certificate's currency is millimetres but the consequence is newtons. A part
    # held by a vacuum gripper has a lateral bound of mu*F_pull = 0.15*25 = 3.75N, and ~1mm of
    # penetration into it is a ~14N shove -- measured on cross-new, a3_ur_coll5 into obj_6 at
    # -1.29mm, breaking a2's and a1's grasps on lat2 +14.0/3.75 and lat1 -14.2/3.75. Tracking
    # (0.052 deg) and grasp slip (0.31mm) were negligible; the COMMANDED config already
    # overlapped by 1.317mm, i.e. the corridor certified it.
    # If this starts producing false violations, the fix is a FINER diagonal calibration
    # (_CAL_DS) or a finer corridor grid -- not a fudge factor that buys makespan with a
    # millimetre the grasp model cannot pay for.
    PEN_MARGIN = 0.0
    _frame_arm = {}                                          # frame -> arm index (prefix match)
    def _arm_of(fname):
      if fname in _frame_arm:
        return _frame_arm[fname]
      a = next((robot_index[r] for r in robots if fname.startswith(r)), None)
      if a is None:
        base = fname.split("__", 1)[0]                       # obj_X__shape -> obj_X
        if base in assembly.objects:
          f = problem.C.getFrame(base)                       # blame the HOLDER (attach parent
          while f is not None:                               # chain in the posed config)
            a = next((robot_index[r] for r in robots if f.name.startswith(r)), None)
            if a is not None:
              break
            f = f.getParent()
          return a                                           # holder changes with mode: no cache
      _frame_arm[fname] = a
      return a
    # pairs the certified path itself drives beyond ~2x the planner's tolerance are rai-
    # DEACTIVATED links (obj rooted to the table at 975mm; a gripper-vs-held-target at 47mm
    # during an insertion) -- an ACTIVE pair that deep would have failed is_collision_free,
    # so the path could not have validated. mr_bench ignores them; policing them against
    # steep-gradient mode-local allowances falsely froze fused groups (both measured). Only
    # plausibly-ACTIVE pairs are policed, against their MODE-LOCAL certified allowance.
    # (The planner's total-penetration bool is NOT used at desynced configs: certified
    # per-pair pens from different modes can sum past its budget -- measured at fired=32.)
    _PAIR_ACTIVE_MAX = max(0.02, 2.0 * float(getattr(problem, "collision_tolerance", 0.01)))
    def collide(fq, fired):
      k_idx = min(fired, len(stretch_mode) - 1)
      _, pens = _pair_pens(_conf(fq), stretch_mode[k_idx])
      who, viol = set(), []
      for pr, dep in pens.items():
        base = allowed.get((k_idx, pr), 0.0)
        if base < _PAIR_ACTIVE_MAX and dep > base + PEN_MARGIN:
          viol.append((pr, dep))
          for f in pr:
            a = _arm_of(f)
            if a is not None:
              who.add(a)
      collide.last_pairs = viol                              # diagnostics for retiming traces
      return who if viol else False                          # blocker arms (empty set = static)

    if corridor:
      # ============ CORRIDOR RETIMING (stages A/B/C -- see retiming.py) ============
      # A: measure ONCE how far arms may drift apart before anything touches, as a function
      # of arc position. B/C then plan inside that certified envelope with NO per-step
      # collision checks at all: arms rest only at their OWN events, and each runs at its own
      # joint limit. Probes pose arm i at s+d with everyone else on the certified diagonal at
      # s (world state from s), and read the violation's ARMS -- pairwise vs another arm, or
      # "world" vs the table/resting parts. Collisions being pairwise is what makes this a
      # handful of 2-D scans instead of a search in n dimensions.
      # Map mode stretches back to symbolic actions once, before either geometry
      # postprocessor.  The experimental C2 fitter uses only this small protection mask; the
      # established rigid-coupling/corridor logic below remains untouched.
      _stretch_plan = [next((name_of[problem.tasks[int(t)].name][0] for t in stretches[k][1]
                             if problem.tasks[int(t)].name in name_of), None)
                       for k in range(len(stretches))]
      _stretch_direct = [st is not None for st in _stretch_plan]
      _last = 0
      for k in range(len(_stretch_plan)):
        if _stretch_plan[k] is None:
          _stretch_plan[k] = _last
        else:
          _last = _stretch_plan[k]

      _c2_protected = set()
      if c2_spline:
        for k, st_idx in enumerate(_stretch_plan):
          if st_idx is None:
            continue
          scan_states = ([states[st_idx]] if _stretch_direct[k]
                         else [states[min(st_idx + 1, len(states) - 1)]])
          for state in scan_states:
            for grp in state.subassemblies():
              holders = {robot_index[c[1]] for p in grp
                         for c in state.supports.get(p, ()) if c[0] == "grasp"
                         and c[1] in robot_index}
              if len(holders) > 1:
                _c2_protected.add(k)
          # Geometry must respect the execution overlap even when the legacy corridor option
          # that couples handover phases was not requested.  Collision checking cannot see a
          # closed-chain relative-pose error, so always preserve these stretches.
          if tuple(plan[st_idx])[0] == "handover":
            _c2_protected.add(k)

      _P, _bi = retiming._composite(stretches)
      if c2_spline:
        # EXPERIMENTAL GEOMETRY STAGE.  This is intentionally a sibling of --spline, not a
        # modification of the now-stable phase scheduler: it consumes mode-local polylines,
        # emits a certified replacement, and the existing pipeline sees ordinary waypoints.
        import c2_smoothing as _c2
        _limits = getattr(problem, "limits", None)
        if _limits is None:
          _limits = C_fk.getJointLimits()

        def _c2_free(q, stretch_index):
          return bool(problem.is_collision_free(
              _conf(q), stretch_mode[min(stretch_index, len(stretch_mode) - 1)]))

        _c2_stretches, _c2_rows, _c2_rep = _c2.smooth_stretches(
            stretches, _c2_free, limits=_limits, protected=_c2_protected,
            per_segment=c2_samples_per_seg, control_ratio=c2_control_ratio,
            smooth_weight=c2_smooth_weight, max_deviation=c2_max_deviation,
            resolution=c2_resolution)
        _P, _bi = retiming._composite(_c2_stretches)
        print(f"[sim/mp] --c2-spline ACTIVE: mode-local cubic B-splines, "
              f"{_c2_rep['fitted']} stretch(es) fitted, "
              f"{_c2_rep['protected']} rigid co-hold/handover stretch(es) protected, "
              f"{_c2_rep['fallback']} unsafe fit(s) kept on the original path; "
              f"{_c2_rep['input_points']} -> {_c2_rep['output_points']} waypoints, "
              f"{_c2_rep['checked_samples']} dense certificate sample(s), max deviation "
              f"{_c2_rep['max_deviation']:.3f} rad. Task witnesses exact.")
        for _row in _c2_rows:
          if _row.status == "fallback":
            print(f"[sim/mp] --c2-spline stretch {_row.index} FALLBACK after "
                  f"{_row.attempts} fit(s): {_row.reason}")
      elif spline:
        # C1 SMOOTHING (--spline). The saved path is C0: every waypoint is a velocity
        # discontinuity, which is what forces the crawl-through-corners behaviour and what
        # makes a curvature-based wrench limit blow up. Catmull-Rom is C1 AND interpolates
        # every waypoint, so the task WITNESSES stay exact -- only the geometry BETWEEN
        # waypoints moves, and each smoothed span is certified with the planner's own check.
        # Barrier-adjacent spans stay straight so approach/retreat directions are preserved.
        _wp_stretch, _kk = [], 0
        for _i in range(len(_P)):
          while _kk + 1 < len(_bi) and _i > _bi[_kk]:
            _kk += 1
          _wp_stretch.append(min(_kk, len(stretch_mode) - 1))

        def _sp_free(Q, j0, j1, i):
          m = stretch_mode[_wp_stretch[min(i, len(_wp_stretch) - 1)]]
          return all(bool(problem.is_collision_free(_conf(Q[j]), m))
                     for j in range(j0, j1 + 1))
        _Q, _imap, _rep = retiming.spline_path(_P, bar_idx=_bi, per_seg=spline_per_seg,
                                               tension=spline_tension, free_of=_sp_free)
        _P, _bi = _Q, [_imap[b] for b in _bi]
        print(f"[sim/mp] --spline ACTIVE: C1 Catmull-Rom, {len(_P)} waypoints "
              f"(was {len(_imap)}); {_rep['checked']} spans certified, {_rep['shrunk']} "
              f"pulled back ({_rep['straightened']} fully straightened), max deviation "
              f"{_rep['deviation']:.3f} rad. Witnesses exact; barrier approaches straight.")
      _sg, _D = retiming.arc_length(list(_P))
      _cs = np.concatenate([[0.0], np.cumsum(_sg)])
      _bar = [_cs[b] for b in _bi]

      def _pose(sv):                                         # composite config at per-arm arcs
        fq = np.empty(_P.shape[1])
        for r in range(len(robots)):
          fq[arm_dims[r]] = retiming.config_at(list(_P), _cs, _sg,
                                               min(max(sv[r], 0.0), _D))[arm_dims[r]]
        return fq

      def _fired_at(s):                                      # world state = barriers passed
        return int(np.searchsorted(_bar, s + 1e-9, side="right"))

      # POSITION-LOCAL baseline instead of the blanket "skip deep pairs" rule. A pair may
      # legitimately be deep at ONE place on the path (a gripper inside the target it is
      # inserting) -- skipping it everywhere made Stage A blind exactly where desync hurts
      # most, and it certified a wide-open corridor on both scenes. Here the allowance is
      # the penetration the CERTIFIED DIAGONAL shows for that pair AT THESE ARC POSITIONS.
      _CG = 0.5                                              # rad per baseline bucket
      _nb = int(_D / _CG) + 2
      diag_pen = {}
      for _i in range(len(_P) - 1):
        _a, _b, _L = _P[_i], _P[_i + 1], _sg[_i]
        _ns = max(1, int(np.ceil(_L / _CAL_DS)))
        for _t in range(_ns + 1):
          _s = _cs[_i] + _L * _t / _ns
          _k = min(_fired_at(_s), len(stretch_mode) - 1)
          for pr, dep in _pair_pens(_conf(_a + (_b - _a) * (_t / _ns)), stretch_mode[_k])[1].items():
            arr = diag_pen.setdefault(pr, np.zeros(_nb))
            _g = min(int(_s / _CG), _nb - 1)
            arr[_g] = max(arr[_g], dep)

      def _allow(pr, sa, sb):
        arr = diag_pen.get(pr)
        if arr is None:
          return 0.0
        ga = min(int(min(max(sa, 0.0), _D) / _CG), _nb - 1)
        gb = min(int(min(max(sb, 0.0), _D) / _CG), _nb - 1)
        return max(arr[ga], arr[gb])                         # deep here on the diagonal => ok

      def _viol_arms(sv, s_ref):
        """Violating (arm, arm) pairs and (arm, world) hits at this desynced combination."""
        k = min(_fired_at(s_ref), len(stretch_mode) - 1)
        _, pens = _pair_pens(_conf(_pose(sv)), stretch_mode[k])
        pairs, world = set(), set()
        for pr, dep in pens.items():
          a, b = _arm_of(pr[0]), _arm_of(pr[1])
          sa = sv[a] if a is not None else s_ref
          sb = sv[b] if b is not None else s_ref
          if dep <= _allow(pr, sa, sb) + PEN_MARGIN:
            continue
          if a is not None and b is not None and a != b:
            pairs.add((min(a, b), max(a, b)))
          elif a is not None or b is not None:
            world.add(a if a is not None else b)
        return pairs, world

      def world_free(i, s, d):
        sv = [s] * len(robots)
        sv[i] = s + d
        return i not in _viol_arms(sv, s)[1]

      def pair_free(i, j, s, d):
        sv = [s] * len(robots)
        sv[i] = s + d
        return (min(i, j), max(i, j)) not in _viol_arms(sv, s)[0]

      # RIGID COUPLINGS the oracle can NEVER see: while two arms hold the SAME group (a
      # handover overlap, a co-grasp/brace) they are kinematically locked -- desyncing them
      # does not "collide", it tears the part out of a gripper (measured: 175.9N grasp break
      # with a wide-open geometric corridor). Derive those spans from the plan's own support
      # structure and force the corridor to 0 there.
      # stretch index -> PLAN ACTION index. These are NOT the same: car-mini has 26 barriers
      # for 17 actions (pre-insertion splits), so indexing `states` by barriers reads the
      # wrong support structure -- it claimed arms held parts they do not, which produced
      # 2-metre "lever arms" and absurd wrench limits (measured).
      _locked = []
      for k in range(len(stretches)):
        st_idx = _stretch_plan[k]
        if st_idx is None:
          continue
        a0 = _bar[k - 1] if k else 0.0
        # The WORLD STATE during a stretch is the state ENTERING its own action for a DIRECT
        # stretch (the action's events fire at its END), and the previous action's POST state
        # for an INHERITED goto stretch. Scanning BOTH states for every stretch locked the
        # pair over the pre-assemble goto too, where the arms still hold SEPARATE parts --
        # measured: a2 (leg in hand) forced to move with a3 (seat in hand) through the whole
        # approach. A merge only exists in the post state, so it may only lock the stretches
        # AFTER the action, never the approach before it.
        _states = ([states[st_idx]] if _stretch_direct[k]
                   else [states[min(st_idx + 1, len(states) - 1)]])
        for state in _states:
          for grp in state.subassemblies():
            hold = sorted({robot_index[c[1]] for p in grp
                           for c in state.supports.get(p, ()) if c[0] == "grasp"
                           and c[1] in robot_index})
            for x in range(len(hold)):
              for y in range(x + 1, len(hold)):
                _locked.append((hold[x], hold[y], a0, _bar[k]))
        if couple_handovers:
          # A HANDOVER IS ATOMIC IN THE DISCRETE MODEL, so the state scan above cannot see it:
          # the giver holds in states[i] and the receiver in states[i+1], never both. In
          # EXECUTION both grippers are on the part across the transition, so the two arms are
          # rigidly coupled through it exactly like a co-grasp, and without this the corridor may
          # desync them by the full width while they co-hold. cross-new has 10 handovers and 0
          # add_grasp, so the state scan finds ONE co-grasp in 36 actions (car-mini, with 4
          # add_grasp, finds 12) and the corridor locks 0-5% of arc instead of 29-42%.
          # Stiff arms hide this: they satisfy both welds by moving the PART and the qpos
          # re-assert discards the reaction. It only bites once the arms are compliant.
          _hact = tuple(plan[st_idx])
          if _hact[0] == "handover":
            _grp = next((g for g in states[st_idx].subassemblies() if _hact[2] in g),
                        {_hact[2]})
            _recv = robot_index.get(_hact[1])
            _give = next((robot_index[c[1]] for p in _grp
                          for c in states[st_idx].supports.get(p, ())
                          if c[0] == "grasp" and c[1] in robot_index
                          and robot_index[c[1]] != _recv), None)
            if _recv is not None and _give is not None:
              _locked.append((min(_recv, _give), max(_recv, _give), a0, _bar[k]))
      if _locked:
        spans = {}
        for i, j, a, b in _locked:
          spans[(i, j)] = spans.get((i, j), 0.0) + (b - a)
        print(f"[sim/mp] rigid couplings from the plan (same group held by 2 arms): " +
              ", ".join(f"arm{i}-arm{j} over {v:.0f} rad" for (i, j), v in sorted(spans.items())))
        _ls = sorted({(i, j, round(a, 2), round(b, 2)) for i, j, a, b in _locked})
        print(f"[sim/mp] rigid-coupling lock spans ({len(_ls)}): " +
              ", ".join(f"arm{i}-arm{j}[{a},{b}]" for i, j, a, b in _ls))
      _vmax = _amax = None
      if force_limits:
        # ---- WRENCH-LIMITED MOTION -------------------------------------------------------
        # The wrench LP is QUASI-STATIC: it sizes a grasp for gravity, not for the inertial
        # load of being swung around. So faster execution can overload a grasp the planner
        # certified (measured: a3's grasp on the car body at 0.43Nm vs a 0.40 cap, which
        # `--joint-rate 0.6` papered over by slowing EVERY arm). Here only the LOADED arm
        # slows, and only where it is loaded.
        #
        # Along a fixed path the held group's Cartesian acceleration is
        #     a = p'(s) * s_ddot  +  p''(s) * s_dot^2,
        # so one wrench budget yields BOTH a speed limit (the curvature term) and an
        # acceleration limit (the tangential term) at each arc position -- a mini-TOPP.
        # Point-mass model: |F| = m|a|, |T| = m|a|*lever (no I*alpha term).
        # SAMPLE PER STRETCH, NOT PER ARC GRID POINT. The held group is a property of the
        # symbolic state, which is constant within a stretch and changes only at barriers --
        # so an arc grid samples the wrong thing at the wrong resolution. Measured on
        # car-mini: the states where an arm holds the WHOLE subassembly are the release /
        # co-grasp stretches, and four of them have exactly ZERO arc after mode repair, so a
        # 1-rad grid could never see them -- the un-optimized path reported "0.20kg max" and
        # ran its force limits off a payload 2.5x too light, while the optimized path (same
        # plan!) happened to put arc into those stretches, saw the true 0.50kg, and deadlocked.
        # One sample per stretch is also CHEAPER (one oracle call per stretch, not per radian).
        _K = len(stretches)
        _p = np.zeros((len(robots), _K, 3))                  # held-group COM, per stretch
        _lev = np.zeros((len(robots), _K))                    # |COM - gripper| lever arm
        _m = np.zeros((len(robots), _K))                      # held mass (SHARE of it, below)
        _bad_lever = 0
        for _k in range(_K):
          _a0 = _bar[_k - 1] if _k else 0.0
          _s = 0.5 * (_a0 + _bar[_k])                        # mid-stretch config
          problem.is_collision_free(_conf(_pose([_s] * len(robots))), stretch_mode[_k])
          _st = states[min(_stretch_plan[_k], len(states) - 1)]   # by ACTION, not by barrier
          for _r, _rn in enumerate(robots):
            grasped = [q for q in assembly.objects
                       if any(c[0] == "grasp" and c[1] == _rn
                              for c in _st.supports.get(q, ()))]
            if not grasped:
              continue                                      # empty gripper: unconstrained
            root = grasped[0]                                # the part the weld actually holds
            grp = next((g for g in _st.subassemblies() if root in g), {root})
            # one definition of the lever, owned by the assembly (Assembly.group_lever)
            lev, gm, com = assembly.group_lever(
                grp, root, lambda q: problem.C.getFrame(q).getPosition())
            if lev > LEVER_SANITY:
              _bad_lever += 1
              continue
            # CO-HOLDERS SHARE THE LOAD. group_lever returns the WHOLE group's mass, and
            # charging that in full to every arm holding it contradicts the planner's own
            # certificate: _grounding_caps feeds ALL of a group's grasps (plus the table)
            # into ONE equilibrium LP, which shares the load by construction. Charging each
            # wrist the full mass made the executor refuse a hold the search proved stable
            # (car-mini: 0.5kg x 91mm = 0.446Nm against a 0.40Nm cap -> zero speed -> the
            # schedule deadlocked). Even split is the INTERIM; the principled successor is
            # wrench.equilibrium_solution's per-contact multipliers, exactly as _static_load's
            # docstring specifies -- geometry, not arithmetic, decides the true split.
            _nhold = sum(1 for _o in robots
                         if any(c[0] == "grasp" and c[1] == _o
                                for q in grp for c in _st.supports.get(q, ())))
            _share = gm / max(_nhold, 1)
            _p[_r, _k] = com
            _m[_r, _k] = _share
            _lev[_r, _k] = max(lev, 5e-3)
            if _os.environ.get("SIM_FORCE_TRACE"):
              print(f"[force-trace] stretch={_k:3d} arc=[{_a0:.2f},{_bar[_k]:.2f}] "
                    f"plan={_stretch_plan[_k]} "
                    f"({tuple(plan[_stretch_plan[_k]]) if _stretch_plan[_k] is not None else '-'}) "
                    f"arm{_r} root={root} grp={sorted(grp)} m={gm:.3f}/{_nhold}={_share:.3f} "
                    f"lev={lev*1000:.0f}mm")
        # FINE geometric sampling: p'' on the 1-rad corridor grid smooths away all real
        # curvature and yields limits of 3-15 rad/s that never bind (measured). Curvature
        # needs a fine grid; this is pure FK (no oracle), so it is cheap. The LIMITS live on
        # this fine grid too (corridor_schedule reads them with limit_ds): reducing them onto
        # the 1-rad corridor grid by MIN let one sharp sample dictate a whole radian of arc,
        # and it is far too coarse for the corner bound below, which is a per-waypoint event.
        _dsf = min(0.05, corridor_ds)
        _sf = np.arange(0.0, _D + _dsf, _dsf)
        _GF = len(_sf)
        _pf = np.zeros((len(robots), _GF, 3))
        for _gi, _s in enumerate(_sf):
          C_fk.setJointState(retiming.config_at(list(_P), _cs, _sg, min(_s, _D)))
          for _r, _rn in enumerate(robots):
            _pf[_r, _gi] = C_fk.getFrame(ee_anchor[_rn]).getPosition()
        # CORNERS (on by default; --no-corner-limits to ablate). What is EXECUTED is the
        # piecewise-linear path through _P's waypoints, so at every waypoint the Cartesian
        # tangent JUMPS -- and every mode
        # barrier is such a waypoint (each stretch begins and ends at a task witness). Lockstep
        # hides this by ramping EVERY arm to rest at every barrier; the corridor deliberately
        # does not, so an arm that merely CROSSES a barrier turns its grasp point by up to
        # 122 deg in ONE 2ms step (measured on car-mini at s=79.9, dbar=0.00: |dv|=1.24m/s at
        # 0.81m/s cruise -> 621 m/s^2 for one step, 39.9N / 2.0Nm through the weld, and the
        # 0.4Nm grasp broke). That is a VELOCITY STEP, not curvature: no p'' bound can see it
        # (the smoothing above removes it by construction) and no constant safety factor covers
        # it, since the implied acceleration grows as 1/dt. What IS bounded is the MOMENTUM
        # step -- the interface must absorb m*|dv| within its response time -- so
        # |dv| <= a_allow*KINK_TAU caps the arc speed AT the corner. On a C1 path the jump is
        # ~0 and this never binds.
        #
        # This is the term the "conservatism" was standing in for, and it is the ONLY one that
        # had to be added: with it, both scenes pass at inertia factor 1 -- no fudge at all
        # (measured, --corridor --force-limits, factor 1; --no-corner-limits reproduces the
        # left column):
        #   car-mini-new  OFF 14/20 FAILED (a3 breaks on the s=79.9 corner)   ON 17/17 ok
        #   cross8        OFF 39/40 (one 25.7N break, |T|=0.06Nm)             ON 39/39 ok
        # and the corridor keeps its advantage over lockstep (+4.5% / +16.1% makespan), since
        # only loaded arms slow and only within ~0.1 rad of a corner.
        _jump = np.zeros((len(robots), _GF))                  # tangent jump [m/rad] at corners
        if corner_limits:
          _pw = np.zeros((len(robots), len(_P), 3))
          for _i in range(len(_P)):
            C_fk.setJointState(_P[_i])
            for _r, _rn in enumerate(robots):
              _pw[_r, _i] = C_fk.getFrame(ee_anchor[_rn]).getPosition()
          _uniq = [0]                                         # waypoints at DISTINCT arc: a
          for _i in range(1, len(_P)):                        # repeated config (mode markers)
            if _cs[_i] - _cs[_uniq[-1]] > 1e-9:               # has no tangent of its own, and
              _uniq.append(_i)                                # skipping it hid barrier corners
          for _a, _b, _c in zip(_uniq, _uniq[1:], _uniq[2:]):
            _gi = int(min(max(round(_cs[_b] / _dsf), 0), _GF - 1))
            for _r in range(len(robots)):
              _t0 = (_pw[_r, _b] - _pw[_r, _a]) / (_cs[_b] - _cs[_a])
              _t1 = (_pw[_r, _c] - _pw[_r, _b]) / (_cs[_c] - _cs[_b])
              _j = float(np.linalg.norm(_t1 - _t0))
              for _gj in (_gi - 1, _gi):                      # the arm must ARRIVE slow: the
                if _gj >= 0:                                  # corner sits mid-cell, half a
                  _jump[_r, _gj] = max(_jump[_r, _gj], _j)    # cell < the braking distance
        # The saved path is only C0: at a waypoint kink p' JUMPS, so a finite-differenced p''
        # spikes to near-infinity and the curvature speed limit collapses to ~0.01 rad/s at
        # isolated points (measured: worse motion than a blunt global slowdown). A kink is a
        # velocity discontinuity, NOT curvature. Smoothing the geometry over the ramp length
        # before differentiating models what a jerk-limited controller actually tracks -- and
        # the --spline path removes the kinks outright.
        _w = max(3, int(round(2.0 * RAMP_DIST / _dsf)))
        _ker = np.ones(_w) / _w
        for _r in range(len(robots)):
          for _c in range(3):
            _pf[_r, :, _c] = np.convolve(np.pad(_pf[_r, :, _c], (_w // 2, _w // 2),
                                                mode="edge"), _ker, mode="same")[
                _w // 2: _w // 2 + _pf.shape[1]]
        _vmax = np.full((len(robots), _GF), 1e6)
        _amax = np.full((len(robots), _GF), 1e6)
        _bind, _cbind, _infeasible = {}, {}, {}
        _d1 = {}                                              # kept for SIM_WRENCH_TRACE: the
        _d2 = {}                                              # trace must use THESE derivatives
        for _r, _rn in enumerate(robots):
          fcap, tcap = _grasp_caps((gripper_kinds or {}).get(_rn, "two_finger"))
          d1 = np.gradient(_pf[_r], _dsf, axis=0)             # d(grasp point)/ds [m/rad]
          d2 = np.gradient(d1, _dsf, axis=0)                  # d2/ds2            [m/rad^2]
          _d1[_r], _d2[_r] = d1, d2
          n1, n2 = np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1)
          for _gi, _s in enumerate(_sf):
            # ROUND, not floor: corridor_schedule reads the coarse arrays with _cor_at, which
            # snaps to the NEAREST grid point -- flooring here shifted every limit by up to
            # half a cell, so the schedule applied the neighbouring radian's budget (measured
            # on car-mini: executed 1.04 rad/s where the limit said 0.73).
            # BY STRETCH: the held group is constant within a stretch and changes only at
            # barriers, so resolving the fine sample to its stretch is exact -- the old
            # round(_s / corridor_ds) also bled a payload up to half a corridor cell PAST the
            # barrier where the arm had already let go.
            _kk = min(_fired_at(_s), _K - 1)
            m, L = _m[_r, _kk], _lev[_r, _kk]
            if m <= 0:
              continue                                       # empty gripper: unconstrained
            # remaining margin after the STATIC load (swap _static_load for the exact LP)
            fs, ts = _static_load(m, L)
            # Conservatism belongs on the INERTIAL prediction, not on the mass: scaling mass
            # also inflates the STATIC term, which eats the whole budget and collapses the
            # limit to ~0.01 rad/s (measured: passes, but 8-18x slower). Here the static load
            # stays true and only the dynamic estimate is derated. Both scenes now pass at
            # k=1: the residual model error (no I*alpha -- measured at 2.8% of the wrench at
            # the break -- and the grasp-anchor path standing in for the COM path) is small
            # next to the corner term above, which is what k>1 was really compensating for.
            k = max(inertia_factor, 1e-9)
            # STATIC LOAD ALONE OVER THE CAP = an INFEASIBLE HOLD, not a slow one. Without
            # this report the only symptom is retiming's "exceeded max_steps" much later: the
            # numerator clamps to 0, a_allow floors at 1e-4, and the arm is told it may
            # accelerate its payload at 0.1mm/s^2 -- a speed limit of ~0 that deadlocks the
            # schedule with no hint of why. Name the arm, the load and the cap instead.
            if (wrench_margin * tcap - ts <= 0.0 or wrench_margin * fcap - fs <= 0.0):
              _key = (_r, round(m, 3), round(L, 3))
              if _key not in _infeasible:
                _infeasible[_key] = (_s, ts, tcap, fs, fcap)
            aF = max(wrench_margin * fcap - fs, 0.0) / (m * k)      # m/s^2 from the force cap
            aT = max(wrench_margin * tcap - ts, 0.0) / (m * L * k)  # ... and the torque cap
            a_allow = max(min(aF, aT), 1e-4)
            if n2[_gi] > 1e-9:
              _vmax[_r, _gi] = min(_vmax[_r, _gi], float(np.sqrt(a_allow / n2[_gi])))
            if n1[_gi] > 1e-9:
              _amax[_r, _gi] = min(_amax[_r, _gi], a_allow / n1[_gi])
            if _jump[_r, _gi] > 1e-9:                          # corner: bound the MOMENTUM step
              _vmax[_r, _gi] = min(_vmax[_r, _gi],
                                   a_allow * KINK_TAU / _jump[_r, _gi])
              _cbind[_r] = _cbind.get(_r, 0) + int(_vmax[_r, _gi] < JOINT_RATE)
          _bind[_r] = int(np.sum(_vmax[_r] < JOINT_RATE))
        _bind = {r: c for r, c in _bind.items() if c}
        # (the STATIC clearance-derived caps that briefly lived here are GONE: baked-in
        # near-table caps slowed whole low carries and barrier coupling still stalled the
        # other arms -- measured +7% makespan over the global pacing they meant to replace.
        # The slowdown is a RUNTIME phenomenon and now lives in the ONLINE stepper's per-arm
        # rate factors, applied only while their trigger actually holds.)
        _vmax = retiming.brakeable(_vmax, _amax, _dsf)
        for (_r, _mm, _LL), (_s, _ts, _tc, _fs, _fc) in sorted(_infeasible.items()):
          print(f"[sim/mp] !!! --force-limits: arm{_r}'s STATIC hold already exceeds its "
                f"grasp cap at s={_s:.1f} rad ({_mm:.2f}kg at {_LL*1000:.0f}mm lever: "
                f"{_ts:.2f}Nm vs {wrench_margin * _tc:.2f}Nm cap"
                + (f", {_fs:.1f}N vs {wrench_margin * _fc:.1f}N" if _fs > wrench_margin * _fc
                   else "") + "). NO speed makes this hold feasible -- the schedule will "
                f"crawl or stall there. The planner's wrench LP shares a group's load across "
                f"ALL its grasps and the table; this check sizes ONE grasp, so they can "
                f"disagree. Check the plan's grasp choice, or --wrench-margin.")
        if _bad_lever:
          print(f"[sim/mp] --force-limits: ignored {_bad_lever} grid/arm samples whose lever "
                f"exceeded {LEVER_SANITY*1000:.0f}mm (stale support info, not a real grasp)")
        print(f"[sim/mp] --force-limits ACTIVE (margin {wrench_margin:g} of the grasp caps, "
              f"point-mass{' + corner momentum steps' if corner_limits else ''}, inertia "
              f"factor {inertia_factor:g}): speed limit binds below cruise on " +
              ", ".join(f"arm{r} {100.0 * c / _GF:.0f}% of arc" for r, c in sorted(_bind.items()))
              + (" -- nowhere" if not _bind else "")
              + (f"; corners bind at {sum(_cbind.values())} waypoint(s)" if _cbind else ""))
        for _r in range(len(robots)):
          _f = _vmax[_r][_vmax[_r] < 1e5]
          if len(_f):
            # the WORST-LOADED stretch reported as a pair, not two independent maxima (which
            # advertised a mass and a lever that need never co-occur)
            _w = int(np.argmax(_m[_r] * _lev[_r]))
            print(f"    arm{_r}: worst load {_m[_r][_w]:.2f}kg at {_lev[_r][_w] * 1000:.0f}mm "
                  f"lever (share of the group; stretch {_w}), "
                  + (f"worst corner {_jump[_r].max():.2f} m/rad, " if corner_limits else "")
                  + f"speed limit min {_f.min():.2f} rad/s (cruise {JOINT_RATE:g})")
      # ---- STAGE A + VERIFY + REFINE ---------------------------------------------------
      # STAGE A IS A SAMPLED CERTIFICATE, AND ITS SAMPLING IS COARSE NEXT TO WHAT IT MUST
      # CATCH. At the 1 rad default, consecutive grid cells on cross-new are ~641mm of summed
      # EE travel apart, while the failure mode is a ~1.3mm overlap lasting 10ms -- ~500x
      # finer than the grid. MEASURED: the corridor certified a configuration in which
      # a3_ur_coll5 penetrated obj_6 (held by a2) by 1.29mm, breaking two grasps on their
      # LATERAL bound; tracking (0.052deg) and grasp slip (0.31mm) were negligible and the
      # COMMANDED config already overlapped by 1.317mm. Zeroing PEN_MARGIN changed nothing --
      # the certificate had simply never evaluated that instant.
      # So: certify coarse, then VERIFY THE SCHEDULE THAT CAME OUT at fine resolution against
      # the exact oracle, and REFINE where it fails. Verification is cheap next to
      # certification -- it walks ONE 1-D trajectory instead of a 2-D envelope over all leads.
      # MEASURED cost of refining cross-new: stage A 21s -> 36s -> 56s at ds 1.0/0.5/0.25,
      # corridor width essentially unchanged (3.64 -> 3.66) and makespan unchanged (+21.4% ->
      # +21.3%), and at 0.25 the collision-driven breaks disappear. Refinement is nearly free.
      # WHO LEGITIMATELY TOUCHES WHOM. A gripper is IN CONTACT with the part it holds, and a
      # mated part is IN CONTACT with its group -- reporting those as "collisions" makes the
      # verifier cry wolf (measured: a1_gripper_fill~obj_0 at 7.21mm is a1 holding obj_0, and
      # obj_0~obj_3__shape at 0.11mm on car-mini is a seated mate). Skip them by asking the
      # DISCRETE STATE, which is exactly where "who holds what" is authoritative.
      def _legit_pair(pr, fired):
        k = min(fired, len(_stretch_plan) - 1)
        st = states[min(_stretch_plan[k] + 1, len(states) - 1)] if _stretch_plan else None
        if st is None:
          return False
        f0, f1 = pr[0].split("__", 1)[0], pr[1].split("__", 1)[0]
        p0, p1 = f0 in assembly.objects, f1 in assembly.objects
        if p0 and p1:                                        # two parts: mated => touching
          return any(f0 in g and f1 in g for g in st.subassemblies())
        if p0 or p1:                                         # arm vs part: is it ITS OWN part?
          part, arm_f = (f0, pr[1]) if p0 else (f1, pr[0])
          # ...and is it the GRIPPER? Only the hand legitimately touches its payload. Exempting
          # the WHOLE arm let a holder's own BASE link bury itself in the part it carries and
          # still count as certified -- measured a1_ur_coll0 ~ obj_6 at 1.65mm / 39.7N, which
          # broke a1's grasp while the corridor reported no violation (and which no
          # --corridor-margin could catch, since the pair was filtered out before the margin
          # was applied). A link that is not the hand has no business touching the payload.
          if not any(t in arm_f for t in ("gripper", "palm", "finger", "robotiq", "vacuum")):
            return False
          grp = next((g for g in st.subassemblies() if part in g), {part})
          for q in grp:
            for c in st.supports.get(q, ()):
              if c[0] == "grasp" and arm_f.startswith(c[1]):
                return True
        return False

      def _verify_schedule(_t, step):
        """Exact per-sample check of the SCHEDULE. Returns [(sample, arcs, pairs, arms), ...]."""
        if _t is None or not len(_t.q):
          return []
        # traj.fires is [(sample, [task_ids])]; all we need is HOW MANY events have fired by
        # sample i, because that selects the attachment mode the oracle checks against.
        fire_samples = sorted(int(_si) for _si, _ in (getattr(_t, "fires", []) or []))
        prog = getattr(_t, "progress", None)
        out = []
        self_worst = {}                                      # same-arm overlaps seen (cross-only)
        for i in range(0, len(_t.q), max(1, step)):
          fired = int(np.searchsorted(fire_samples, i, side="right"))
          if not collide(_t.q[i], fired):
            continue
          real = [(pr, dp) for pr, dp in getattr(collide, "last_pairs", [])
                  if not _legit_pair(pr, fired)]
          if corridor_cross_only and real:
            # A SAME-ARM pair is INVARIANT under everything the scheduler controls: desync,
            # locks and ds only change the RELATIVE timing between arms, while a same-arm
            # overlap depends solely on that arm's own arc position -- every schedule
            # (lockstep included) passes through it identically. Treating one as a scheduling
            # counterexample sent CEGAR through 4 futile refinement passes (~3min each, 95% of
            # a 14.6min sim run) on a1_ur_coll5~a1_ur_coll8, a base-path overlap the PLANNER
            # legally accepted at tolerance 0.01. Report it once as a PATH property instead.
            keep = []
            for pr, dp in real:
              a0, a1 = _arm_of(pr[0]), _arm_of(pr[1])
              if a0 is not None and a0 == a1:
                k = (pr[0], pr[1])
                if dp > self_worst.get(k, -1.0): self_worst[k] = dp
              else:
                keep.append((pr, dp))
            real = keep
          if not real:
            continue                                         # only grasps/mates: not a violation
          arms = {a for pr, _ in real for a in (_arm_of(pr[0]), _arm_of(pr[1]))
                  if a is not None}
          out.append((i, (list(prog[i]) if prog is not None else None), real, sorted(arms)))
        if self_worst:
          worst = sorted(self_worst.items(), key=lambda kv: -kv[1])[:3]
          print("[sim/mp] --corridor-cross-only: ignored SAME-ARM overlap(s) for scheduling "
                "(schedule-invariant, a BASE-PATH property -- tighten --mp-tolerance to remove "
                "them at the source): "
                + ", ".join(f"{a}~{b} {d*1000:.2f}mm" for (a, b), d in worst))
        return out

      _ds = corridor_ds
      _extra_locks, cor, traj = [], None, None

      # OFFLINE PHASE SCHEDULING.  This is deliberately separate from the sampled corridor:
      # it plans a monotone coordination-space curve (one phase per arm), checks every
      # proposed configuration with the planner's exact predicate, and then audits the whole
      # result again.  The objective is a calm, repeatable reference -- preferred cruise,
      # few slow/stop episodes, and nominal ceremony reservations.  Makespan is only a
      # consequence.  Runtime servos may hold on an exceptional overrun, but never re-time
      # the nominal profile tick by tick.
      if phase_schedule:
        if corridor_online:
          raise ValueError("--phase-schedule and --corridor-online are mutually exclusive")

        _phase_stretches = []
        for _k, _b in enumerate(_bi):
          _a = 0 if _k == 0 else _bi[_k - 1]
          _phase_stretches.append(([np.asarray(x, float) for x in _P[_a:_b + 1]],
                                   list(stretches[_k][1])))

        def _seat_has_budget(_tid):
          if servo is None:
            return False
          _t = problem.tasks[_tid]
          _fr = getattr(_t, "frames", None)
          if not _fr:
            return False
          if _fr[0] in assembly.objects:
            return True
          return (_fr[0] == "table" and len(_fr) > 1
                  and _fr[1] in assembly.objects and servo.place_ok)

        _event_budget = []
        for _tids in (list(st[1]) for st in stretches):
          _b = phase_servo_time if (actuated and settle_tol > 0.0) else 0.0
          _b += phase_servo_time * sum(_seat_has_budget(_tid) for _tid in _tids)
          _event_budget.append(_b)

        def _strict_collide(fq, fired):
          """Exact planner contract used both while scheduling and for final acceptance."""
          _ki = min(int(fired), len(stretch_mode) - 1)
          _collide_t0 = _time.perf_counter()
          ok = bool(problem.is_collision_free(
              _conf(fq), stretch_mode[_ki], collision_tolerance=_ptol))
          _strict_collide.calls += 1
          _strict_collide.seconds += _time.perf_counter() - _collide_t0
          if ok:
            _strict_collide.last_pairs = []
            return False
          pairs, who = [], set()
          _raw_penetrating = _legit_penetrating = 0
          for _c in problem.C.getCollisions(0.0):
            _d = float(_c[2])
            if _d >= 0.0:
              continue
            _pr = (str(_c[0]), str(_c[1]))
            _raw_penetrating += 1
            # The planner predicate normally removes current grasp/mate contacts through its
            # mode. Some exported zero-actor transitions briefly select an adjacent task-mode
            # encoding, however, and the raw accumulated depth then rejects the holder's own
            # finger/payload contact (measured OpenArm car-mini: obj_1~a2 finger, 1.43mm).
            # The discrete symbolic state is authoritative here, exactly as in the established
            # corridor verifier below: exempt only a gripper touching the component that arm
            # actually holds, or two already-mated parts. Arm links, other arms and unrelated
            # parts remain ordinary blockers.
            if _legit_pair(_pr, fired):
              _legit_penetrating += 1
              continue
            pairs.append((_pr, -_d))
            for _f in _pr:
              _ar = _arm_of(_f)
              if _ar is not None:
                who.add(_ar)
          # A strict rejection can come from our planning wrapper while every raw distance
          # is still positive (notably --mp-robot-margin).  Preserve that wrapper's owners:
          # otherwise the scheduler mislabels an arm-arm clearance veto as a static wall and
          # cannot infer the local coupling that recovers the certified diagonal.
          _last = (problem.last_margin_rejection()
                   if hasattr(problem, "last_margin_rejection") else None)
          _reason = (_last or {}).get("reason") if (_last or {}).get("layer") == "gate" else None
          if _reason is not None:
            _fr = tuple(str(x) for x in _reason.get("frames", ("?", "?")))
            _gap = max(float(_reason.get("required", 0.0))
                       - float(_reason.get("observed", 0.0)), 0.0)
            pairs.append((_fr, _gap))
            for _owner in _reason.get("owners", ()):
              if _owner in robot_index:
                who.add(robot_index[_owner])
          if (not pairs and _reason is None and _raw_penetrating > 0
              and _legit_penetrating == _raw_penetrating):
            _strict_collide.last_pairs = []
            return False                                  # only intended grasp/mate contacts
          _strict_collide.last_pairs = sorted(pairs, key=lambda x: -x[1])
          return who                                      # empty set means static-world block

        _strict_collide.calls = 0
        _strict_collide.seconds = 0.0

        print("[sim/mp] --phase-schedule ACTIVE: visual-quality coordination-space plan; "
              f"preferred cruise {JOINT_RATE:g}rad/s, nominal servo reservations "
              f"{min(_event_budget or [0]):g}-{max(_event_budget or [0]):g}s, exact "
              f"{_ptol:g}m collision predicate on every accepted step. Makespan is not "
              "the objective; the synchronized reference is the certified fallback.")
        if limit_rate:
          print("[sim/mp] --phase-schedule: --limit-rate is intentionally not used to chase "
                "joint limits; the preferred visual cruise remains --joint-rate. Joint "
                "limits themselves remain hard constraints of the audited path.")
        if force_limits:
          print("[sim/mp] --phase-schedule: geometric scheduling uses the conservative "
                "preferred cruise; the existing wrench speed/acceleration tables are not "
                "yet part of this scheduler. Runtime wrench guards remain authoritative.")

        def _phase_audit(_traj, collect=False):
          _audit_t0 = _time.perf_counter()
          _audit_checks = 0
          _audit_indices = retiming.motion_audit_indices(
              _traj.q, _traj.fires or [], resolution=phase_audit_resolution)
          def _audit_done(_result):
            _phase_audit.last = {
                "checks": _audit_checks,
                "seconds": _time.perf_counter() - _audit_t0,
                "samples": len(_audit_indices),
                "trajectory_samples": len(_traj.q)}
            return _result
          _at = {}
          _bad = []
          for _ii, _ev in (_traj.fires or []):
            _at.setdefault(int(_ii), 0)
            _at[int(_ii)] += 1                              # one mode transition per fire point
          _fired = 0
          for _ii in _audit_indices:
            _ii = int(_ii)
            _q = _traj.q[_ii]
            _audit_checks += 1
            if _strict_collide(_q, _fired) is not False:
              _hit = (_ii, _fired, list(_strict_collide.last_pairs[:5]))
              if not collect:
                return _audit_done(_hit)
              _bad.append(_hit)
            if _ii in _at:
              for _ in range(_at[_ii]):
                _post = min(_fired + 1, len(stretch_mode) - 1)
                _audit_checks += 1
                if _strict_collide(_q, _post) is not False:
                  _hit = (_ii, _post, list(_strict_collide.last_pairs[:5]))
                  if not collect:
                    return _audit_done(_hit)
                  _bad.append(_hit)
                _fired += 1
          return _audit_done(_bad if collect else None)

        _fallback_reason, traj = None, None
        # Scheduling at the 2ms physics rate repeats the expensive full-scene predicate five
        # times more often than the motion planner's own 0.01rad edge scale (chair took >6min
        # before reaching simulation). Plan at 10ms, interpolate back to 2ms, then keep the
        # SAME path-resolution acceptance audit below. A counterexample triggers 4ms and finally
        # 2ms retries. Every planning grid is an integer multiple of the 2ms controller tick,
        # so fire points survive expansion exactly (5ms cannot be represented on that grid).
        _plan_dts = list(dict.fromkeys([max(DT, 0.01), max(DT, 0.004), DT]))
        _q_start = np.asarray(_phase_stretches[0][0][0], dtype=float)
        for _pdt in _plan_dts:
          _search_t0 = _time.perf_counter()
          _search_calls0 = _strict_collide.calls
          _search_collision_s0 = _strict_collide.seconds
          try:
            _coarse = retiming.plan_schedule(
                _phase_stretches, arm_dims, acting, _pdt, joint_rate=JOINT_RATE,
                ramp_dist=RAMP_DIST, ramp_min=RAMP_MIN, collide=_strict_collide,
                # Effectively zero cells: candidates are exact; cache only identical states
                # revisited by a CEGAR restart.
                collide_ds=1e-9, dwell=dwell, event_durations=_event_budget,
                per_arm_rate=True, stretch_rates=_stretch_rates)
          except RuntimeError as _e:
            _fallback_reason = f"{_pdt:g}s planning grid: {_e}"
            print(f"[sim/mp] --phase-schedule {_pdt:g}s search failed after "
                  f"{_time.perf_counter() - _search_t0:.2f}s; collision predicate "
                  f"{_strict_collide.calls - _search_calls0} calls/"
                  f"{_strict_collide.seconds - _search_collision_s0:.2f}s: {_e}")
            continue
          print(f"[sim/mp] --phase-schedule {_pdt:g}s search: "
                f"{_time.perf_counter() - _search_t0:.2f}s; collision predicate "
                f"{_strict_collide.calls - _search_calls0} calls/"
                f"{_strict_collide.seconds - _search_collision_s0:.2f}s.")
          traj = retiming.upsample_trajectory(_coarse, DT, q0=_q_start)
          _bad_phase = _phase_audit(traj)
          if _bad_phase is None:
            print(f"[sim/mp] --phase-schedule planning grid {_pdt:g}s -> {DT:g}s control "
                  f"samples; path-resolution audit accepted on attempt "
                  f"{_plan_dts.index(_pdt) + 1}/{len(_plan_dts)}.")
            break
          _fallback_reason = (f"{_pdt:g}s schedule's path-resolution audit failed at sample "
                              f"{_bad_phase[0]}, mode {_bad_phase[1]}: {_bad_phase[2]}")
          print(f"[sim/mp] --phase-schedule RETRY FINER: {_fallback_reason}")
          traj = None
        if traj is None:
          print(f"[sim/mp] --phase-schedule FALLBACK TO SYNCHRONIZED REFERENCE: "
                f"{_fallback_reason}")
          traj = retiming.build_trajectory(
              _phase_stretches, DT, joint_rate=JOINT_RATE, ramp_dist=RAMP_DIST,
              ramp_min=RAMP_MIN, dwell=dwell, event_durations=_event_budget,
              stretch_rates=_stretch_rates)
          _bad_phase = _phase_audit(traj)
          if _bad_phase is not None:
            # Distinguish a genuinely invalid synchronized sample from mutable collision-
            # scene/cache contamination after the coordination search has queried modes in
            # a non-monotone order.  The saved-path preflight above walks the same geometry
            # monotonically, so without this retry both failure classes looked identical.
            _bi_bad, _mode_bad, _pairs_bad = _bad_phase
            _retry_bad = None
            if hasattr(problem, "prev_mode"):
              problem.prev_mode = None
              _retry_verdict = _strict_collide(traj.q[_bi_bad], _mode_bad)
              _retry_bad = (_retry_verdict is not False,
                            list(getattr(_strict_collide, "last_pairs", [])[:5]))
            raise RuntimeError(
                f"[sim/mp] synchronized fallback failed its exact audit at sample "
                f"{_bad_phase[0]}, mode {_bad_phase[1]}: {_bad_phase[2]}; "
                f"fresh-mode retry={_retry_bad}")
        print(f"[sim/mp] --phase-schedule exact final audit: "
              f"{_phase_audit.last['samples']}/{len(traj.q)} trajectory samples at "
              f"<={phase_audit_resolution:g}rad joint-path variation + "
              f"{len(traj.fires)} post-switch poses collision-free; "
              f"{_phase_audit.last['checks']} predicate calls in "
              f"{_phase_audit.last['seconds']:.2f}s.")
        if phase_smooth:
          if traj.progress is None:
            raise RuntimeError(
                "[sim/mp] --phase-smooth REQUIRED but the phase scheduler fell back to a "
                "synchronized trajectory with no independent phase trace. Aborting before "
                "physics; rerun without --phase-smooth only if raw/synchronized execution "
                "is explicitly desired.")
          else:
            _raw_traj = traj
            _restore = set()
            _smooth_rep = None
            # The plan-derived locks protect rigid co-holds; the phase search's CEGAR locks
            # protect locally fused collision passages discovered during this particular run.
            _smooth_locks = list(_locked) + list(getattr(traj, "couplings", ()))
            for _smooth_pass in range(3):
              _cand, _smooth_rep = retiming.smooth_phase_trajectory(
                  _raw_traj, _P, arm_dims, acting, window=phase_smooth_time,
                  locked=_smooth_locks, restore_intervals=_restore,
                  barrier_indices=_bi, ramp_dist=RAMP_DIST,
                  corner_dv=retiming.DEFAULT_CORNER_DV)
              _bad_smooth = _phase_audit(_cand, collect=True)
              if not _bad_smooth:
                _br, _ar = _smooth_rep["before"], _smooth_rep["after"]
                _jr = 100.0 * (1.0 - _ar["jerk_rms"] / max(_br["jerk_rms"], 1e-12))
                _acr = 100.0 * (1.0 - _ar["accel_rms"] / max(_br["accel_rms"], 1e-12))
                if not retiming.smoothness_not_worse(_br, _ar, tolerance=0.02):
                  _pa0, _pa1 = _smooth_rep["phase_before"], _smooth_rep["phase_after"]
                  _arms = ", ".join(
                      f"a{_r} accel {100 * (_a['accel_rms'] / max(_b['accel_rms'], 1e-12) - 1):+.1f}%/"
                      f"jerk {100 * (_a['jerk_rms'] / max(_b['jerk_rms'], 1e-12) - 1):+.1f}%"
                      for _r, (_b, _a) in enumerate(zip(
                          _smooth_rep["by_arm_before"], _smooth_rep["by_arm_after"])))
                  raise RuntimeError(
                      f"[sim/mp] --phase-smooth REQUIRED but its collision-free proposal "
                      f"would change joint accel RMS by {-_acr:+.1f}% and jerk RMS by "
                      f"{-_jr:+.1f}% (more than 2% worse is rejected). Phase-space accel "
                      f"changes by {100 * (_pa1['accel_rms'] / max(_pa0['accel_rms'], 1e-12) - 1):+.1f}% "
                      f"and jerk by {100 * (_pa1['jerk_rms'] / max(_pa0['jerk_rms'], 1e-12) - 1):+.1f}%; "
                      f"max phase displacement {_smooth_rep['progress_max_delta']:.4g}rad; "
                      f"corner-fixed increments {_smooth_rep['corner_fixed_increments']}; "
                      f"per arm: {_arms}. Aborting before "
                      "physics instead of executing the unsmoothed phase schedule.")
                traj = _cand
                print(f"[sim/mp] --phase-smooth ACTIVE: Gaussian phase-rate smoothing over "
                      f"{phase_smooth_time:g}s; every fire vector pinned, monotonicity and "
                      f"actor stops preserved. Exact {phase_audit_resolution:g}rad "
                      f"path-resolution audit passed; restored "
                      f"{len(_restore)}/{_smooth_rep['intervals']} collision-sensitive "
                      f"interval(s) and {_smooth_rep['locked_arm_intervals']} rigid arm-"
                      f"interval(s); {_smooth_rep['rate_limited_arm_intervals']} arm-"
                      f"interval(s) kept raw to prevent peak-rate growth; "
                      f"{_smooth_rep['corner_fixed_increments']} intentional corner-rate "
                      f"increment(s) preserved. Joint accel RMS "
                      f"reduction {_acr:+.1f}%, jerk RMS reduction {_jr:+.1f}%.")
                break
              _new = {retiming.phase_interval_at(_raw_traj, _i)
                      for _i, _m, _p in _bad_smooth} - _restore
              if not _new:
                raise RuntimeError(
                    "[sim/mp] --phase-smooth REQUIRED but a collision remained on already "
                    "restored source intervals. Aborting before physics instead of executing "
                    "the unsmoothed phase schedule.")
              _restore |= _new
              print(f"[sim/mp] --phase-smooth local fallback pass {_smooth_pass + 1}: "
                    f"restoring fire interval(s) {sorted(_new)} after "
                    f"{len(_bad_smooth)} path-resolution collision rejection(s).")
            else:
              raise RuntimeError(
                  "[sim/mp] --phase-smooth REQUIRED but its collision repair did not "
                  "converge after 3 passes. Aborting before physics instead of executing "
                  "the unsmoothed phase schedule.")
        if (phase_batch or phase_intent) and traj.progress is not None:
          # Batch short move-stop-move islands only after the ordinary scheduler/smoother has
          # finished. This sibling pass does not care whether geometry came from Catmull-Rom
          # or C2. It may flow through another arm's event, but checks both worlds there.
          _batch_source = traj
          _fire_samples = [int(i) for i, _ in traj.fires]
          # Scheduler-created collision couplings certify its particular trace but are not a
          # physical equality. The intent pass deliberately searches a new trace and certifies
          # it again, so only true co-hold/handover locks remain immutable there. Conservative
          # local batching retains the scheduler couplings as before.
          _batch_locks = (list(_locked) if phase_intent else
                          list(_locked) + list(getattr(traj, "couplings", ())))

          def _batch_free(_q, _sample):
            _bt0 = _time.perf_counter()
            def _bdone(_ok):
              _batch_free.calls += 1
              _batch_free.seconds += _time.perf_counter() - _bt0
              return _ok
            _batch_free.last_detail = None
            _fired = int(np.searchsorted(_fire_samples, int(_sample), side="left"))
            if _strict_collide(_q, _fired) is not False:
              _batch_free.last_detail = (
                  int(_sample), _fired, list(_strict_collide.last_pairs[:5]))
              return _bdone(False)
            # The event changes attachments at this exact configuration. Check every
            # successive post-switch mode if multiple zero-duration events share a sample.
            for _j, _fi in enumerate(_fire_samples):
              if _fi != int(_sample):
                continue
              _post = min(_fired + 1, len(stretch_mode) - 1)
              if _strict_collide(_q, _post) is not False:
                _batch_free.last_detail = (
                    int(_sample), _post, list(_strict_collide.last_pairs[:5]))
                return _bdone(False)
              _fired += 1
            return _bdone(True)
          _batch_free.calls = 0
          _batch_free.seconds = 0.0

          if phase_intent:
            _batch, _batch_rep = retiming.intentional_phase_motions(
                traj, _P, arm_dims, acting, _bar, locked=_batch_locks,
                stopped_rate=0.05 * JOINT_RATE, collision_free=_batch_free,
                intent_rate=phase_intent_rate, ramp_time=phase_intent_ramp)
            # One Gauss-Seidel refinement removes robot-order bias. On the first pass a1 is
            # placed against a2's still-spread source trace; after a2 is compacted, a1 may be
            # able to shift later without reopening an all-robot gap. Equal-support edits are
            # allowed only when their arrival moves later and both global-idle guards hold.
            _refined, _ref_rep = retiming.intentional_phase_motions(
                _batch, _P, arm_dims, acting, _bar, locked=_batch_locks,
                stopped_rate=0.05 * JOINT_RATE, collision_free=_batch_free,
                intent_rate=phase_intent_rate, ramp_time=phase_intent_ramp)
            if _ref_rep["accepted"]:
              _batch = _refined
              for _key in ("collision_rejected", "event_rejected", "rate_rejected",
                           "noeffect_rejected"):
                _batch_rep[_key] += _ref_rep[_key]
              for _row in _ref_rep["collision_examples"]:
                if (len(_batch_rep["collision_examples"]) < 64
                    and not any(tuple(_old[:3]) == tuple(_row[:3])
                                for _old in _batch_rep["collision_examples"])):
                  _batch_rep["collision_examples"].append(_row)
              _wins = list(dict.fromkeys(
                  list(_batch_rep["accepted_windows"])
                  + list(_ref_rep["accepted_windows"])))
              _batch_rep["accepted_windows"] = _wins
              _batch_rep["accepted"] = len(_wins)
              _resolved = set(_ref_rep["accepted_windows"])
              _batch_rep["unresolved"] = [
                  row for row in _batch_rep["unresolved"] if tuple(row[:3]) not in _resolved]
              for _key in ("support_after", "global_idle_after", "smoothness_after",
                           "islands_after", "slow_after", "foreign_stops_after"):
                _batch_rep[_key] = _ref_rep[_key]
              _batch_rep["refined"] = _ref_rep["accepted"]
            else:
              _batch_rep["refined"] = 0
          else:
            _batch, _batch_rep = retiming.batch_phase_motions(
                traj, _P, arm_dims, acting, _bar, locked=_batch_locks,
                min_motion=phase_batch_min_motion, max_gap=phase_batch_max_gap,
                stopped_rate=0.05 * JOINT_RATE, collision_free=_batch_free)
          _bad_batch = _phase_audit(_batch)
          _restored_batch_windows = set()
          # The planner predicate has internal state/caches, so retain the same final
          # audit used everywhere else. If it finds a counterexample missed during a local
          # proposal check, restore the connected accepted window(s), not the whole batch.
          while _bad_batch is not None:
            _sample = int(_bad_batch[0])
            _wins = list(enumerate(_batch_rep.get("accepted_windows", ())))
            _seed = {i for i, (_r, _lo, _hi) in _wins
                     if i not in _restored_batch_windows and _lo <= _sample <= _hi}
            if not _seed:
              break
            _group = set(_seed)
            _changed = True
            while _changed:                                # overlapping windows form one edit
              _changed = False
              for i, (r, lo, hi) in _wins:
                if i in _group or i in _restored_batch_windows:
                  continue
                if any(r == _wins[j][1][0]
                       and lo <= _wins[j][1][2] and _wins[j][1][1] <= hi for j in _group):
                  _group.add(i)
                  _changed = True
            for i in _group:
              r, lo, hi = _wins[i][1]
              _batch.q[lo:hi + 1, arm_dims[r]] = _batch_source.q[lo:hi + 1, arm_dims[r]]
              _batch.progress[lo:hi + 1, r] = _batch_source.progress[lo:hi + 1, r]
            _restored_batch_windows |= _group
            _bad_batch = _phase_audit(_batch)
          if _bad_batch is None:
            traj = _batch
            _batch_rows = retiming.motion_islands(
                traj.q, traj.dt, arm_dims, stopped_rate=0.05 * JOINT_RATE)
            _batch_after = len(_batch_rows)
            _batch_small = sum(x["arc"] < phase_batch_min_motion for x in _batch_rows)
            _batch_internal_slow = retiming.internal_slow_episodes(
                traj.q, traj.dt, arm_dims, stopped_rate=0.05 * JOINT_RATE)
            _batch_support = retiming.motion_support(
                traj.q, traj.dt, arm_dims, stopped_rate=0.05 * JOINT_RATE)
            _batch_global_idle = retiming.global_idle_stats(
                traj.q, traj.dt, arm_dims, stopped_rate=0.05 * JOINT_RATE)
            _batch_smoothness = retiming.trajectory_smoothness(
                traj.q, traj.dt, arm_dims)
            _batch_foreign_stops = retiming.foreign_switch_stops(
                traj.q, traj.dt, arm_dims, traj.fires, acting,
                stopped_rate=0.05 * JOINT_RATE)
            _proposed = (_batch_rep['accepted'] if phase_intent else _batch_rep['merged'])
            _kept = _proposed - len(_restored_batch_windows)
            if phase_intent:
              _sb = _batch_rep["support_before"]
              _gi0 = _batch_rep["global_idle_before"]
              _sm0, _sm1 = (_batch_rep["smoothness_before"],
                            _batch_smoothness)
              if (_kept <= 0
                  or _batch_support["moving_samples"] >= _sb["moving_samples"]):
                raise RuntimeError(
                    "[sim/mp] --phase-intent REQUIRED but no collision-certified reduction "
                    "in motion support survived the final audit. Aborting before physics.")
              print(f"[sim/mp] --phase-intent ACTIVE: kept {_kept}/"
                    f"{_proposed} certified robot-owned interval(s); moving support "
                    f"{_sb['moving_samples']} -> {_batch_support['moving_samples']} samples "
                    f"(idle {100 * _sb['idle_fraction']:.1f}% -> "
                    f"{100 * _batch_support['idle_fraction']:.1f}%), conditional moving rate "
                    f"{_sb['mean_moving_rate']:.2f} -> "
                    f"{_batch_support['mean_moving_rate']:.2f}rad/s; islands "
                    f"{_batch_rep['islands_before']} -> {_batch_after}, internal slow gaps "
                    f"{_batch_rep['slow_before']} -> {_batch_internal_slow}, foreign-switch "
                    f"stops {_batch_rep['foreign_stops_before']} -> "
                    f"{_batch_foreign_stops}; all-robot idle "
                    f"{_gi0['duration']:.1f}s -> {_batch_global_idle['duration']:.1f}s "
                    f"(longest {_gi0['longest']:.1f}s -> "
                    f"{_batch_global_idle['longest']:.1f}s); "
                    f"accel RMS {100 * (_sm1['accel_rms'] / max(_sm0['accel_rms'], 1e-12) - 1):+.1f}%, "
                    f"jerk RMS {100 * (_sm1['jerk_rms'] / max(_sm0['jerk_rms'], 1e-12) - 1):+.1f}%. "
                    f"Rejected {_batch_rep['collision_rejected']} collision-sensitive, "
                    f"{_batch_rep['event_rejected']} event-passage, "
                    f"{_batch_rep['rate_rejected']} peak-rate, "
                    f"{len(_batch_rep['unresolved'])} unresolved interval(s), "
                    f"{_batch_rep['protected']} rigidly protected and "
                    f"{_batch_rep.get('funnel_protected', 0)} skill-funnel interval(s); final "
                    f"audit locally restored {len(_restored_batch_windows)} interval(s). Exact "
                    f"{phase_audit_resolution:g}rad path-resolution audit passed, including "
                    f"both modes at switches. Coordinate "
                    f"refinement shifted {_batch_rep.get('refined', 0)} interval(s) later. "
                    f"Collision profile: candidate screen {_batch_free.calls} calls/"
                    f"{_batch_free.seconds:.2f}s; final audit "
                    f"{_phase_audit.last['checks']} calls/"
                    f"{_phase_audit.last['seconds']:.2f}s.")
              if _batch_rep["unresolved"]:
                print(f"[sim/mp] --phase-intent unresolved examples: "
                      f"{_batch_rep['unresolved'][:6]}")
              if _batch_rep["collision_examples"]:
                print(f"[sim/mp] --phase-intent first collision by interval "
                      f"(arm,a,b,steps,offset,sample,detail): "
                      f"{_batch_rep['collision_examples'][:12]}")
            else:
              print(f"[sim/mp] --phase-batch ACTIVE: kept {_kept}/{_batch_rep['merged']} "
                    f"certified short-motion merge(s); motion islands "
                    f"{_batch_rep['before']} -> {_batch_after}, small islands "
                    f"(<{phase_batch_min_motion:g}rad) {_batch_rep['small_before']} -> "
                    f"{_batch_small}, internal slow gaps "
                    f"{_batch_rep['internal_slow_before']} -> "
                    f"{_batch_internal_slow}. Rejected "
                    f"{_batch_rep['collision_rejected']} collision-sensitive, "
                    f"{_batch_rep['event_rejected']} event/coupling, "
                    f"{_batch_rep['rate_rejected']} peak-rate, "
                    f"{_batch_rep['quality_rejected']} visual-quality proposal(s); final audit locally "
                    f"restored {len(_restored_batch_windows)} merge(s). Exact "
                    f"{phase_audit_resolution:g}rad path-resolution audit passed, "
                    "including both modes at switches.")
          else:
            traj = _batch_source
            _touch = [w for w in _batch_rep.get("accepted_windows", ())
                      if w[1] <= int(_bad_batch[0]) <= w[2]]
            if phase_intent:
              raise RuntimeError(
                  f"[sim/mp] --phase-intent REQUIRED but final audit rejected sample "
                  f"{_bad_batch[0]}, mode {_bad_batch[1]}: {_bad_batch[2]}; accepted windows "
                  f"touching it: {_touch}. Aborting before physics.")
            print(f"[sim/mp] --phase-batch FALLBACK: final exact audit rejected sample "
                  f"{_bad_batch[0]}, mode {_bad_batch[1]}: {_bad_batch[2]}; accepted windows "
                  f"touching it: {_touch}. Executing the preceding certified trajectory.")
        elif phase_batch or phase_intent:
          raise RuntimeError(
              f"[sim/mp] {'--phase-intent' if phase_intent else '--phase-batch'} REQUIRED "
              "but no independent phase trace exists. "
              "Aborting before physics instead of executing a synchronized/raw trajectory.")
        # Localize visually objectionable pauses in the planned reference.  The aggregate
        # motion-quality report below says how much stopping occurred, but not whether a long
        # all-arm plateau sits immediately before a mode switch (usually phase bookkeeping,
        # not useful settling).  Keep this compact and only report plateaus over 0.5s.
        _tq = np.asarray(traj.q, dtype=float)
        if len(_tq) > 1:
          _moving = np.zeros(len(_tq) - 1, dtype=bool)
          for _ad in arm_dims:
            _moving |= np.linalg.norm(np.diff(_tq[:, _ad], axis=0), axis=1) >= 0.05 * DT
          _pauses, _start = [], None
          for _ii, _mv in enumerate(_moving):
            if not _mv and _start is None:
              _start = _ii
            if (_mv or _ii == len(_moving) - 1) and _start is not None:
              _end = _ii if _mv else _ii + 1
              if (_end - _start) * DT >= 0.5:
                _next = next(((_fi, _ev) for _fi, _ev in traj.fires if _fi >= _end), None)
                _pauses.append((_start, _end, _next))
              _start = None
          if _pauses:
            _worst = max(_pauses, key=lambda _x: _x[1] - _x[0])
            print(f"[sim/mp] --phase-schedule longest all-arm reference pause: "
                  f"{(_worst[1] - _worst[0]) * DT:.2f}s at samples "
                  f"{_worst[0]}:{_worst[1]}, next fire {_worst[2]}.")

      elif corridor_online:
        # ONLINE MODE: certify ONCE (Stage A, with the plan's rigid-coupling locks), then hand
        # the certificate to the executor -- Stage B/C offline scheduling (previously ~75% of
        # a run's wall time), the spread pass (and its known vmax hole) and schedule-VERIFY
        # all disappear: there is no precomputed schedule to verify, every state the stepper
        # visits is inside the certified set by construction (same conservative lookups).
        print(f"[sim/mp] --corridor-online: STAGE A certificate only (D={_D:.0f} rad, "
              f"ds={_ds:g}); B/C scheduling + VERIFY replaced by the ONLINE stepper -- "
              f"slowdowns are per-arm and exist only while their runtime trigger holds.")
        t0 = _time.time()
        cor = retiming.certify_corridor(len(robots), _D, world_free, pair_free,
                                        ds_grid=_ds, ds_lead=corridor_ds_lead,
                                        lead_max=corridor_lead_max, locked=_locked,
                                        progress=(print if verbose else None))
        print(f"[sim/mp] STAGE A done in {_time.time() - t0:.0f}s")
        print(retiming.corridor_width_report(cor))
        def _seat_has_budget(_tid):
          if servo is None:
            return False
          _t = problem.tasks[_tid]
          _fr = getattr(_t, "frames", None)
          if not _fr:
            return False
          if _fr[0] in assembly.objects:
            return True
          return (_fr[0] == "table" and len(_fr) > 1
                  and _fr[1] in assembly.objects and servo.place_ok)

        # Actual convergence time is unknown, but it is not unbounded: each servo hook has
        # the explicit `settle_max` budget. Schedule crossing arms against that upper bound.
        # If a hook finishes early the mode opens early and they simply continue; if it uses
        # its full budget they reach (rather than wait at) the no-pass barrier. This avoids
        # pretending that an EMA of past, unrelated contacts predicts the next insertion.
        _pre_fire_budget = []
        for _tids in (list(st[1]) for st in stretches):
          _b = settle_max if (actuated and settle_tol > 0.0) else 0.0
          _b += settle_max * sum(_seat_has_budget(_tid) for _tid in _tids)
          _pre_fire_budget.append(_b)
        online_stepper = retiming.CorridorStepper(
            _P, arm_dims, _bar, acting, cor, DT, joint_rate=JOINT_RATE,
            ramp_dist=RAMP_DIST, ramp_min=RAMP_MIN, dwell=dwell,
            vmax=_vmax, amax=_amax, limit_ds=(_dsf if force_limits else None),
            rate_clamp=rate_clamp, fire_all=False,
            pace=pace, settle_nom=_pre_fire_budget, trace=(pace_trace is not None))
        online_fire_tids = [list(stretches[k][1]) for k in range(len(stretches))]
        if pace:
          print(f"[sim/mp] --pace ACTIVE: feed-forward phase scheduling -- one fixed rate "
                f"factor per corridor-coupled arm group and mode stretch. Acting groups meet "
                f"at fire; crossing groups reserve the bounded pre-fire servo budget "
                f"({min(_pre_fire_budget):g}-{max(_pre_fire_budget):g}s here). Runtime guards "
                f"stay hard safety overrides.")
      for _pass in range(0 if (corridor_online or phase_schedule)
                         else max(0, corridor_refine) + 1):
        print(f"[sim/mp] --corridor STAGE A: certifying the desync envelope "
              f"(D={_D:.0f} rad, {len(robots)} arms, ds={_ds:g}"
              + (f", +{len(_extra_locks)} counterexample lock(s)" if _extra_locks else "") + ") ...")
        t0 = _time.time()
        cor = retiming.certify_corridor(len(robots), _D, world_free, pair_free,
                                        ds_grid=_ds, ds_lead=corridor_ds_lead,
                                        lead_max=corridor_lead_max,
                                        locked=_locked + _extra_locks,
                                        progress=(print if verbose else None))
        print(f"[sim/mp] STAGE A done in {_time.time() - t0:.0f}s")
        print(retiming.corridor_width_report(cor))
        if rate_clamp:
          print("[flag] --corridor-rate-clamp ACTIVE: each corridor step is bounded by the "
                "joint motion it actually produces (rate_of's share is taken from the segment "
                "the arm STARTS in, so a step landing in a busier segment overshoots joint_rate).")
        traj = retiming.corridor_schedule(_P, arm_dims, _bar, acting, cor, DT,
                                          joint_rate=JOINT_RATE, ramp_dist=RAMP_DIST,
                                          ramp_min=RAMP_MIN, dwell=DWELL_T, spread=True,
                                          vmax=_vmax, amax=_amax,
                                          limit_ds=(_dsf if force_limits else None),
                                          rate_clamp=rate_clamp)
        if corridor_verify <= 0:
          break
        t0 = _time.time()
        _bad = _verify_schedule(traj, corridor_verify)
        print(f"[sim/mp] VERIFY: {len(traj.q) // max(1, corridor_verify)} schedule samples "
              f"checked exactly in {_time.time() - t0:.0f}s -> "
              f"{len(_bad) if _bad else 'NO'} real violation(s)")
        if not _bad:
          print("[sim/mp] the executed schedule is certified collision-free at this resolution.")
          break
        for _i, _arcs, _prs, _arms in _bad[:4]:
          print(f"[sim/mp] !!! schedule COLLIDES at sample {_i} (t={_i*DT:.2f}s), arms {_arms}: "
                + ", ".join(f"{p[0]}~{p[1]} {d*1000:.2f}mm" for p, d in _prs[:3]))
        if _pass >= corridor_refine:
          print(f"[sim/mp] !!! refinement budget exhausted; EXECUTING ANYWAY -- expect the sim "
                f"to report the consequence. Raise --corridor-refine.")
          break
        # LOCAL REFINEMENT: turn each counterexample into a LOCK on the offending arm pair over
        # a window around where it happened, instead of re-gridding the whole path. Global
        # halving does not converge -- it reschedules everything, so each pass yields a
        # DIFFERENT trajectory with different violations (measured on cross-new: 3 -> 2 -> 1 ->
        # 2 with a different pair each time). A lock removes exactly the desync freedom the
        # counterexample used, at the SAME certification cost.
        _win = max(2.0 * _ds, 1.0)
        _added = 0
        for _i, _arcs, _prs, _arms in _bad:
          if _arcs is None or len(_arms) < 2:
            continue                                         # arm-vs-world: locking cannot help
          for _x in range(len(_arms)):
            for _y in range(_x + 1, len(_arms)):
              _a, _b = _arms[_x], _arms[_y]
              _s0 = max(0.0, min(_arcs[_a], _arcs[_b]) - _win)
              _s1 = min(_D, max(_arcs[_a], _arcs[_b]) + _win)
              _extra_locks.append((min(_a, _b), max(_a, _b), _s0, _s1))
              _added += 1
        if _added:
          print(f"[sim/mp] REFINING LOCALLY: locking {_added} arm pair(s) over +/-{_win:g} rad "
                f"around the counterexample(s) (pass {_pass + 2}/{corridor_refine + 1})")
        else:
          _ds *= 0.5                                         # world violations: finer grid
          print(f"[sim/mp] REFINING: no lockable arm pair; ds -> {_ds:g} "
                f"(pass {_pass + 2}/{corridor_refine + 1})")
      # (Continuous-motion PACING lived here and is REMOVED, measured: to pass the dense
      # gate the paced phasing needed 2381 local pair locks -- i.e. the star-shaped
      # certificate forced it back into near-lockstep, with stops WORSE than the greedy
      # schedule (32-37% vs 11-13%), a 543s execution and a 607mm seat miss (19/22).
      # Continuous cruise with desync needs a certificate over the paced desync set, not
      # star probes -- a separate piece of work. The greedy schedule below is dense-clean
      # (0/43548) and validated 21/21; its remaining stops are the ceremonies and the
      # genuine syncs, i.e. task physics, not scheduling waste.)
      if traj is None and not corridor_online:
        raise RuntimeError("[sim/mp] corridor scheduling produced no trajectory")
      if traj is not None:
        base = retiming.build_trajectory(stretches, DT, joint_rate=JOINT_RATE,
                                         ramp_dist=RAMP_DIST, ramp_min=RAMP_MIN, dwell=DWELL_T,
                                         stretch_rates=_stretch_rates)
        _ff = getattr(traj, "force_floor", None) or {}
        if _ff:
          print("[sim/mp] !!! --force-limits: the wrench cap was BELOW the never-stall floor "
                f"({RAMP_MIN * JOINT_RATE:g} rad/s) at " +
                ", ".join(f"arm{r} {n} sample(s)" for r, n in sorted(_ff.items())) +
                " -- those waypoints are traversed at the CAP now (previously the floor "
                "silently won). Expect a longer schedule; a near-zero cap means the LP "
                "considers that waypoint untraversable.")
        if phase_schedule:
          print(f"[sim/mp] PHASE SCHEDULE: {len(traj.q)} samples vs {len(base.q)} "
                f"synchronized ({100 * (1 - len(traj.q) / max(len(base.q), 1)):+.1f}% "
                f"duration, informational only), {len(traj.fires)} fire points; every "
                "scheduler candidate and final path-resolution sample checked exactly")
        else:
          print(f"[sim/mp] STAGES B+C: {len(traj.q)} samples vs {len(base.q)} lockstep "
                f"({100 * (1 - len(traj.q) / max(len(base.q), 1)):+.1f}% makespan), "
                f"{len(traj.fires)} fire points, ZERO collision checks during scheduling")
        print(retiming.format_motion_report(
            retiming.motion_report(base.q, DT, arm_dims, JOINT_RATE), "LOCKSTEP baseline"))
        print(retiming.format_motion_report(
            retiming.motion_report(traj.q, DT, arm_dims, JOINT_RATE),
            "PHASE SCHEDULE" if phase_schedule else "CORRIDOR"))
        fire_at = {}
        for idx, ev in traj.fires:
          if phase_schedule:                                 # actual task ids, like lockstep
            fire_at.setdefault(int(idx), []).extend(int(t) for t in ev)
          else:
            for k in ev:                                     # corridor carries stretch indices
              fire_at.setdefault(int(idx), []).extend(int(t) for t in stretches[int(k)][1])
      if _os.environ.get("SIM_WRENCH_TRACE"):
        # The trace must read the model AS EXECUTED: same arc grid, same smoothed derivatives,
        # same static term -- so it can only be built here, where those arrays exist.
        _prog = getattr(traj, "progress", None)
        _rix = {r: i for i, r in enumerate(robots)}

        def _wpred(rn):
          if _prog is None or not force_limits or scene.trace_ctx is None:
            return None
          i, k = int(scene.trace_ctx), _rix[rn]
          if i >= len(_prog):
            return None
          s = float(_prog[i][k])
          sm = float(_prog[i - 1][k]) if i > 0 else s
          sp = float(_prog[i + 1][k]) if i + 1 < len(_prog) else s
          v = (sp - sm) / (2.0 * DT)
          sdd = (sp - 2.0 * s + sm) / (DT * DT)
          gi = int(min(max(round(s / _dsf), 0), len(_sf) - 1))
          _kk = min(_fired_at(s), _m.shape[1] - 1)           # per-STRETCH, as the limits are
          m, L = float(_m[k, _kk]), float(_lev[k, _kk])
          av = _d1[k][gi] * sdd + _d2[k][gi] * v * v          # a = p'*s_ddot + p''*s_dot^2
          a = float(np.linalg.norm(av))
          fs, ts = _static_load(m, L)
          return {"s": s, "v": v, "acc": sdd, "m": m, "L": L, "a": a,
                  "F": fs + m * a, "T": ts + m * a * L, "vmax": float(_vmax[k][gi]),
                  "kink": float(_jump[k][gi]),
                  "dbar": min([abs(s - b) for b in _bar] or [-1.0])}

        scene.wrench_trace = _make_wrench_tracer(scene, predict=_wpred)
  else:
    # LOCKSTEP: the same drive -> dwell -> fire -> dwell sequence the old per-stretch loop
    # produced imperatively, built as DATA by build_trajectory (designed to mirror it; command
    # stream probe-verified identical) so every preprocessing mode funnels into ONE executor.
    traj = retiming.build_trajectory(stretches, DT, joint_rate=JOINT_RATE,
                                     ramp_dist=RAMP_DIST, ramp_min=RAMP_MIN, dwell=DWELL_T,
                                     arm_dims=boost_dims, vmax=boost_vmax,
                                     stretch_rates=_stretch_rates)
    fire_at = {}
    for idx, ev in traj.fires:
      fire_at.setdefault(int(idx), []).extend(int(t) for t in ev)

  if traj is None:
    # ---------------- ONLINE EXECUTION (--corridor-online) ----------------
    # The stepper advances per-arm arc clocks with the exact offline rules (conservative
    # certificate lookups, no-pass-the-pending-barrier, laggard never capped) -- but rate
    # factors come from RUNTIME triggers, so a slowdown exists only while its trigger holds
    # and only on the arm that owns it. Fires reuse the same settle/seat/fire hooks.
    _cap_steps = int(40 * len(robots) * online_stepper.D
                     / max(JOINT_RATE * DT, 1e-9)) + 10000
    _n_tick = 0
    _paced = np.zeros(len(robots))                         # diagnostic: ticks with factor < 1
    _held = np.zeros(len(robots))                          # ticks with factor == 0 (guard)
    # EXACT COMMAND GATE + DIAGONAL COLLAPSE. The Stage A certificate is STAR-SHAPED (arm i at s+d,
    # everyone else AT s): two arms simultaneously off-diagonal is not certified, and offline
    # that unsoundness was caught by schedule-VERIFY + CEGAR -- which the online mode dropped.
    # Measured cost of dropping it: a3's link 5.2mm inside a1's cargo at 87N (seed 1), cargo
    # vs cargo at 3.1mm/47N (seed 2). Replacement: every VCHK ticks, one EXACT collision check
    # of the current commanded config (the same oracle Stage A probed with, ~0.5ms); on
    # violation, LEADERS freeze and the laggard advances -- collapsing toward the diagonal,
    # which is the planner's own certified path, until the check clears. Judge the ACTUAL
    # servo.adjust(fq), not the raw phase reference: the admittance may move a command by
    # 10mm after the old checker accepted it, which invalidated its verdict. Keep the exact
    # geometric oracle at its 3-tick / 6ms cadence: running it at 500Hz triples wall time,
    # while the normalized physical wrench guard below still runs every 2ms and the break
    # monitor itself requires 20ms sustained overload.
    _VCHK = 3
    _n_checks = 0
    _n_gate = 0                                            # ticks the gate held an arm
    _n_servo_gate = 0                                      # violations introduced by adjustment
    _stall_fire, _stall_tick = online_stepper.fired, 0
    if online_fire_tids:
      _set_process_for(online_fire_tids[0])
    def _safe(fq):
      nonlocal _n_checks, _n_gate, _n_servo_gate
      if _n_tick % _VCHK:
        return False
      _n_checks += 1
      _qcmd = servo.adjust(fq) if servo is not None else fq
      _who = collide(_qcmd, online_stepper.fired)
      if _who:
        _n_gate += 1
        if servo is not None and not collide(fq, online_stepper.fired):
          _n_servo_gate += 1
      return _who or False
    while not online_stepper.done():
      _n_tick += 1
      if _n_tick > _cap_steps:
        raise RuntimeError(f"[sim/mp] online stepper exceeded {_cap_steps} ticks at "
                           f"fired {online_stepper.fired}/{online_stepper.K}")
      rf = None
      if servo is not None:
        servo.update()
        rf = servo.rate_factors(robots)
      if rf is not None:
        _paced += (np.asarray(rf) < 1.0)
        _held += (np.asarray(rf) <= 0.0)
      qc0, fire_k = online_stepper.tick(rf, free_of=_safe)
      if online_stepper.fired != _stall_fire:
        _stall_fire, _stall_tick = online_stepper.fired, _n_tick
      elif _n_tick - _stall_tick >= 5000 and (_n_tick - _stall_tick) % 5000 == 0:
        # A normal car-mini stretch is <3500 ticks. If a mode stays unchanged longer, print
        # enough live state to distinguish pacing from a force/torque or geometry deadlock.
        _wr = []
        if servo is not None:
          for _rn, _g in sorted(scene.active_grasps.items()):
            _wf, _wt = scene._weld_wrench(_g[0])
            _cf = float(np.linalg.norm(scene.branch_contact_force(_rn)))
            _off = float(np.linalg.norm(servo.off.get(_rn, np.zeros(3))))
            _wr.append(f"{_rn}:contact={_cf:.1f}N,wrench="
                       f"{_wf/max(float(_g[2]),1e-9):.2f}F/"
                       f"{_wt/max(float(_g[3]),1e-9):.2f}T,off={_off*1000:.1f}mm")
        print(f"[sim/mp] STALL DIAGNOSTIC: event {online_stepper.fired}, "
              f"{_n_tick - _stall_tick} ticks; s={np.round(online_stepper.s, 3).tolist()}, "
              f"pace={np.round(online_stepper._pace_factor, 3).tolist()}, "
              f"runtime={None if rf is None else np.round(rf, 3).tolist()}, "
              f"gate-holds={_n_gate}; " + ("; ".join(_wr) if _wr else "no active grasps"))
      qc = servo.adjust(qc0) if servo is not None else qc0
      drive_q(qc)
      scene.step(1, report, ("path",))
      if fire_k is not None:
        # FIRE-FLOW: the online gate (fire_all=False) fires as soon as the ACTING arms rest
        # at the barrier; laggard arms use the settle/servo window to keep advancing through
        # their certified corridor (they were capped at the pending barrier pre-fire, so they
        # ARRIVE -- never cross -- during the window). `acting_for` pins the acting arms plus
        # every holder of a touched part; everyone else flows, with the micro-verify still
        # policing the desync every _VCHK ticks and freezing any violating arm for the rest
        # of this window.
        _act = acting_for(online_fire_tids[fire_k])
        print(f"[sim/mp] fire {fire_k}: tids {online_fire_tids[fire_k]} -> act {sorted(_act)}")
        _freeze = set()
        def _fire_factors():
          if servo is None:
            rf_f = np.ones(len(robots))
            for _r in _act:
              rf_f[_r] = 0.0
            for _r in _freeze:
              rf_f[_r] = 0.0
            return rf_f
          _fl = [r for r in range(len(robots)) if r not in _act]
          servo.update(only=[robots[r] for r in _fl])
          rf_f = servo.rate_factors(robots)
          for _r in _act:
            rf_f[_r] = 0.0
          for _r in _freeze:
            rf_f[_r] = 0.0
          return rf_f
        _fw_ticks = [0]
        def _fire_mix(q_ref):
          nonlocal _n_tick
          _n_tick += 1
          _qflow = online_stepper.tick(_fire_factors())[0]
          _qm = np.asarray(q_ref, dtype=float).copy()
          for _r in range(len(robots)):
            if _r not in _act:
              _qm[arm_dims[_r]] = _qflow[arm_dims[_r]]
          return _qm
        def _fire_verify(q_m):
          _fw_ticks[0] += 1
          _who = collide(q_m, online_stepper.fired)
          _bad = (set(_who) if _who else set()) - _act
          if _bad:
            _new = _bad - _freeze
            _freeze.update(_bad)
            if _new:
              print(f"[sim/mp] fire-flow: froze arm(s) {sorted(_new)} for this fire "
                    f"after a desync violation")
        def _flow_settle(q_ref):
          _qm = _fire_mix(q_ref)
          _fire_verify(_qm)
          drive_q(_qm)
          scene.step(1, report, ("settle",))
          return _qm
        def _flow_drive(q_ref):
          _qm = _fire_mix(q_ref)
          _fire_verify(_qm)
          drive_q(_qm)
          return _qm
        _flow_tick = lambda: scene.step(1, report, ("servo",))
        _fw0 = online_stepper.steps
        settle_before_fire(qc0, _touch_robots(online_fire_tids[fire_k]),
                           _touch_mates(online_fire_tids[fire_k]), flow=_flow_settle,
                           owned=[robots[r] for r in _act])
        # THE MODE SWITCH IS THE ATTACHMENT CHANGE, not the post-fire cleanup. Keep the old
        # no-pass barrier only through the genuinely uncertain settle/seat phase. Once all
        # relinks have fired, advance the stepper's topology immediately; non-acting arms may
        # then cross the barrier while pick-lift/other cleanup continues. Keeping confirm()
        # after lift made its variable duration part of global synchronization for no semantic
        # reason, which no phase retimer can hide without eventually parking at the barrier.
        for tid in online_fire_tids[fire_k]:
          seat_before_fire(tid, qc0, flow_drive=_flow_drive, flow_tick=_flow_tick)
          align_pick_before_fire(tid, qc0, flow_drive=_flow_drive, flow_tick=_flow_tick)
          fire(tid)
        _pre_fire = (online_stepper.steps - _fw0) * DT
        online_stepper.confirm_fire(defer_pace_replan=True)
        _set_process_for(online_fire_tids[online_stepper.fired]
                         if online_stepper.fired < len(online_fire_tids) else [])
        for tid in online_fire_tids[fire_k]:
          lift_after_fire(tid, qc0, flow_drive=_flow_drive, flow_tick=_flow_tick)
        # Hold the old constant rates through cleanup, then plan the next stretch ONCE from
        # the state actually reached. This is an event-triggered replan, not a tick feedback
        # loop, and means the unknown lift duration cannot corrupt the next arrival forecast.
        if pace:
          online_stepper.replan_pace()
        _budget = _pre_fire_budget[fire_k]
        print(f"[sim/mp] fire {online_stepper.fired - 1}: pre-fire servo "
              f"{_pre_fire*1000:.0f}ms / {_budget*1000:.0f}ms reserved; post-fire cleanup "
              "overlapped with the next mode"
              + (", then phase profile replanned once" if pace else ""))
    print(f"[sim/mp] --corridor-online: {_n_tick} ticks executed "
          f"({online_stepper.fired}/{online_stepper.K} events), no precomputed schedule; "
          f"paced ticks per arm {[int(x) for x in _paced]}, held {[int(x) for x in _held]}; "
          f"preventive gate: {_n_checks} adjusted-command checks (every {_VCHK} ticks), "
          f"{_n_gate} hold(s), "
          f"{_n_servo_gate} adjustment-only")
    if pace:
      print(f"[sim/mp] feed-forward profiles planned: {online_stepper._pace_replans}; "
            "pace factors mean per arm: "
            + ", ".join(f"{r} {v:.3f}" for r, v in zip(
                robots, online_stepper._pf_sum / max(online_stepper._pf_n, 1))))
      _lbl = ["at-cap", "hold", "pace<.05", "rf=0", "at-end", "moving", "corr-cap", "crawl"]
      print("[sim/mp] stop causes per arm (ticks):")
      for _i, _r in enumerate(robots):
        print(f"  {_r}: " + ", ".join(f"{_lbl[_j]} {int(online_stepper._why[_i, _j])}"
                                      for _j in range(8)))
      _wk = online_stepper._why_k.sum(axis=1)
      _nonact = np.zeros(len(robots), dtype=np.int64)
      for _k in range(online_stepper.K):
        for _r in range(len(robots)):
          if _r not in online_stepper.act[_k]:
            _nonact[_r] += online_stepper._why_k[_k, _r]
      print("[sim/mp] avoidable no-pass waits (non-acting arms only, ticks): "
            + ", ".join(f"{_r} {int(_nonact[_i])}" for _i, _r in enumerate(robots)))
      _hot = [k for k in range(online_stepper.K) if _wk[k] > 500]
      if _hot:
        print("[sim/mp] at-cap waits by barrier (ticks): "
              + ", ".join(f"k{k} {int(_wk[k])}" for k in _hot))
      if online_stepper._trace is not None and pace_trace:
        _tr = online_stepper._trace
        np.savez(pace_trace,
                 s=np.asarray([t[0] for t in _tr]),
                 pf=np.asarray([t[1] for t in _tr]),
                 rf=np.asarray([t[2] for t in _tr]),
                 fired=np.asarray([t[3] for t in _tr]),
                 bar=np.asarray(online_stepper.bar_s),
                 acting=np.asarray([[r in a for r in range(len(robots))]
                                    for a in online_stepper.act], dtype=bool),
                 stop_causes=online_stepper._why.copy(),
                 cap_waits=online_stepper._why_k.copy())
        print(f"[sim/mp] pace trace saved: {pace_trace} ({len(_tr)} ticks)")
  else:
    # ------------- THE OFFLINE EXECUTION LOOP (precomputed trajectory modes) -------------
    # Preprocessing (corridor schedule | lockstep trapezoid) yields `traj` + `fire_at`;
    # execution is a single loop so a hook (servo, settle, seat, trace) cannot silently miss
    # a branch -- the first --skill-servo build did exactly that (banner on, corridor loop
    # untouched, recording bit-identical to baseline). The uncertified --prioritized branch
    # is REMOVED. The GLOBAL pacing/guard below is the offline modes' blunt instrument; the
    # per-arm version lives in the --corridor-online stepper above.
    _fire_samples = sorted(fire_at)

    def _process_after(sample):
      nxt = next((j for j in _fire_samples if j > sample), None)
      _set_process_for(fire_at[nxt] if nxt is not None else [])

    if _fire_samples:
      _set_process_for(fire_at[_fire_samples[0]])

    def _pending_place(i):
      """If the NEXT fire is a lone PLACE of a currently-PRESSED cargo, return
      (fire sample, tid, part, expected rest pose); else None."""
      if servo is None or not servo.place_ok:
        return None
      j = next((sm for sm in _fire_samples if sm >= i), None)
      if j is None or len(fire_at[j]) != 1:                 # bundled foreign fires: no jump --
        return None                                        # other arms are mid-path at i
      tid = fire_at[j][0]
      t = problem.tasks[tid]
      fr = getattr(t, "frames", None)
      if not fr or fr[0] != "table" or len(fr) < 2 or fr[1] not in assembly.objects:
        return None
      holder = next((r for r, g in scene.active_grasps.items() if g[1] == fr[1]), None)
      if holder is None or getattr(servo, "_hot_by", {}).get(holder, 0.0) <= servo.f0:
        return None
      _l = name_of.get(t.name)
      if _l is None:
        return None
      # NO JUMP OVER OTHER ARMS' MOTION. Skipping to the fire sample teleports the reference
      # of every arm that was mid-carry across [i, j]: drive_q derives qdot from the
      # reference delta, so a one-tick velocity spike hits arms that have nothing to do with
      # this place. Only jump when the span is the holder's alone.
      _hold_dims = set(int(x) for x in arm_dims[robots.index(holder)])
      _others = [k for k in range(traj.q[i].shape[0]) if k not in _hold_dims]
      if _others and float(np.max(np.abs(traj.q[j][_others] - traj.q[i][_others]))) > 1e-3:
        return None
      exp = _state_part_poses(states[min(_l[0] + 1, len(states) - 1)], assembly,
                              lift=CARRY_LIFT)[fr[1]]
      return j, tid, fr[1], np.asarray(exp, dtype=float)

    def _pending_mate(i):
      """Return a nearby lone mate fire when its intended interface has seated by contact.

      The joint-space endpoint is an ideal datum, but the physical skill terminates at
      contact.  Consuming the remaining samples after a fitted CAD interface has touched can
      only command penetration.  As for the place shortcut, never jump over another arm's
      motion; that keeps this a local execution skill rather than an implicit rescheduler.
      """
      if servo is None:
        return None
      j = next((sm for sm in _fire_samples if sm >= i), None)
      if j is None or j == i or len(fire_at[j]) != 1:
        return None
      tid = fire_at[j][0]
      fr = getattr(problem.tasks[tid], "frames", None)
      if not fr or len(fr) < 2 or fr[0] not in assembly.objects or fr[1] not in assembly.objects:
        return None
      parent, child = fr[0], fr[1]
      holder = next((r for r, g in scene.active_grasps.items() if g[1] == child), None)
      if holder is None:
        return None
      got = _rel_pos(scene.part_pose(parent), scene.part_pose(child))
      miss = float(np.linalg.norm(
          got - np.asarray(assembly.relative_pose(child)[1], dtype=float)))
      if miss > SEAT_POS_TOL or scene.part_pair_contact_force(parent, child) < 0.5:
        return None
      hold_dims = set(int(x) for x in arm_dims[robots.index(holder)])
      others = [k for k in range(traj.q[i].shape[0]) if k not in hold_dims]
      if others and float(np.max(np.abs(traj.q[j][others] - traj.q[i][others]))) > 1e-3:
        return None
      return j, tid, parent, child, miss

    i, _NQ = 0, len(traj.q)
    while i < _NQ:
      qc = traj.q[i]
      if servo is not None:
        servo.update()
        qc = servo.adjust(qc)
      if servo is not None and not phase_schedule:
        # VELOCITY-FUNNEL PACING (GLOBAL CLOCK, funnel-scoped): hold the reference WHILE the
        # funnel demands slowness, re-evaluating every tick -- the executed descent
        # decelerates, measured vz drops, the factor recovers and the hold releases on its
        # own. Unlike the old binary near-zone (which crawled through every horizontal low
        # carry -- the user-reported "annoying global slowdown"), the funnel triggers only
        # while a cargo is actually DESCENDING near the table: ~hundreds of ticks per run.
        # The clock is still global in the legacy lockstep/corridor trajectories.  A phase
        # schedule already has independent arm clocks and feed-forward servo reservations;
        # applying this old min-over-arms loop underneath it freezes EVERY robot whenever
        # one placed part touches the table (measured: 14s all-arm pause on chair).  Phase
        # execution therefore keeps the compliant offset above but does not introduce a
        # second, global reactive clock here.
        _fmin = float(np.min(servo.rate_factors(robots)))
        _reps = 0
        while _fmin < 0.999 and _reps < 11:
          drive_q(qc)
          scene.step(1, report, ("approach",))
          servo.update()
          qc = servo.adjust(traj.q[i])
          _fmin = float(np.min(servo.rate_factors(robots)))
          _reps += 1
        # GUARDED HOLD: while a held part is PRESSED the schedule stops advancing and the
        # admittance yields in place. Budget-capped so a hard jam cannot hang the run.
        held = 0
        while servo.pressing() and held < int(1.0 / DT):
          drive_q(qc)
          scene.step(1, report, ("guard",))
          servo.update()
          qc = servo.adjust(traj.q[i])
          held += 1
        if held >= int(1.0 / DT) and servo.pressing():
          # GUARD EXHAUSTED, STILL PRESSED. If what the schedule is grinding toward is a
          # PLACE of the pressed cargo, stop consuming the joint reference (its z is
          # authored against the planner's lower table) and let the skill finish by
          # CONTACT: task-space servo to the rest anchor, touch-terminate, fire the
          # place, resume after its sample. --servo-place-seat only; else old behaviour.
          _pp = _pending_place(i)
          if _pp is not None:
            _j, _tid, _part, _exp = _pp
            servo.place_seat(qc, _part, _exp, problem.tasks[_tid].name, drive_q,
                             lambda: scene.step(1, report, ("servo",)), settle_max)
            fire(_tid)                                     # capture the TOUCHED pose
            _process_after(_j)
            i = _j + 1
            continue
      drive_q(qc)
      scene.trace_ctx = i
      scene.step(1, report, (traj.phase[i],))
      _pm = _pending_mate(i)
      if _pm is not None:
        _j, _tid, _parent, _child, _miss = _pm
        print(f"[sim/mp] contact-complete mate {_child}->{_parent} "
              f"{_miss*1000:.1f}mm from datum; skipping {_j-i} penetration-command sample(s)")
        fire(_tid)
        _process_after(_j)
        i = _j + 1
        continue
      if i in fire_at:
        # A phase-scheduled fire deliberately allows unrelated arms to keep moving through
        # the rendezvous. Requiring THEIR instantaneous tracking residual to converge here
        # turns that valid crossing motion into a global stop and was the remaining long
        # pause in car-mini. Only the event's actors and holders have to be at the witness.
        # Lockstep/corridor retain their historical all-arm check.
        _owned = ([robots[r] for r in sorted(acting_for(fire_at[i]))]
                  if phase_schedule else None)
        settle_before_fire(traj.q[i], _touch_robots(fire_at[i]),
                           _touch_mates(fire_at[i]), owned=_owned)
        for tid in fire_at[i]:
          seat_before_fire(tid, traj.q[i])
          align_pick_before_fire(tid, traj.q[i])
          fire(tid)
        _process_after(i)                                   # the completed mate no longer loads
        for tid in fire_at[i]:
          lift_after_fire(tid, traj.q[i])                    # schedule's own post-fire dwell
      i += 1
  scene.set_process_wrenches([])
  scene.step(int(SETTLE_T / DT), report, ("settle",))
  servo_summary()

  report.final_pose_err = {}
  for p in assembly.objects:
    exp = np.asarray(_state_part_poses(states[-1], assembly, lift=CARRY_LIFT)[p], dtype=float)
    report.final_pose_err[p] = (
        float(np.linalg.norm(scene.part_pose(p)[:3] - exp[:3])),
        _part_orientation_error(assembly, p, scene.part_pose(p)[3:], exp[3:]))
  if scene.mate_caps is not None and scene._mate_max:
    # per-mate load maxima -- this is the calibration readout: caps must sit ABOVE every
    # value seen in a normal run and BELOW the load of an abusive hold (car by one wheel).
    _mm = ", ".join(f"{p}~{scene._mate_eqs[p][1]} {f:.1f}N/{t:.2f}Nm"
                    for p, (f, t) in sorted(scene._mate_max.items()))
    print(f"[sim/mate] per-mate max weld load over the run: {_mm}")
  if rec is not None:
    _save_recording(record, scene, rec, report, metadata=record_metadata)
  return report


def _judge_resting(scene, assembly, action, expect_state):
  """Taxonomy judge for the motion-plan executor: only parts whose group RESTS in the
  expected state are compared (held groups ride the arm at witness poses, which the symbolic
  layout does not model)."""
  exp_poses = _state_part_poses(expect_state, assembly, lift=CARRY_LIFT)
  for p in assembly.objects:
    grp = next(g for g in expect_state.subassemblies() if p in g)
    held = any(c[0] == "grasp" for q in grp for c in expect_state.supports.get(q, ()))
    if held:
      continue
    exp = np.asarray(exp_poses[p], dtype=float)
    got = scene.part_pose(p)
    if got[2] < exp[2] - DROP_Z:
      return ActionResult(action, False, "dropped", p)
    if p != action[2]:
      if float(np.linalg.norm(got[:2] - exp[:2])) > SLIDE_TOL:
        return ActionResult(action, False, "slid",
                            f"{p} {float(np.linalg.norm(got[:2]-exp[:2]))*1000:.0f}mm")
      if _part_orientation_error(assembly, p, got[3:], exp[3:]) > TOPPLE_ANG:
        return ActionResult(action, False, "toppled", p)
  return ActionResult(action, True)


# ---------------------------------------------------------------------------------------------
# replay: look at a recorded rollout, any time, no re-simulation
# ---------------------------------------------------------------------------------------------
def replay_recording(path, speed=1.0, loop=True):
  """Open the MuJoCo viewer on a recording made with execute(record=...): kinematic playback
  of the stored qpos frames (no physics re-run -- exactly what happened, bit for bit). BLOCKS
  while the viewer is open (close it or ctrl-C to stop).

  SCRUBBING (viser-style): a Tk window with a draggable FRAME SLIDER + play/pause opens next
  to the viewer (stdlib tkinter; skipped gracefully if unavailable). The slider label shows
  time + the most recent plan action. Keyboard in the MuJoCo window works too:
      SPACE play/pause | LEFT/RIGHT step 1 | PgUp/PgDn jump 5% | HOME/END | UP/DOWN speed x2
  """
  import json, time
  z = np.load(path, allow_pickle=False)
  model = mujoco.MjModel.from_xml_string(str(z["xml"]))
  data = mujoco.MjData(model)
  marks = dict((int(f), a) for f, a in json.loads(str(z["marks"])))
  qpos = z["qpos"]
  T = len(qpos)
  print(f"[replay] {path}: {T} frames, {len(marks)} actions; results:")
  for r in json.loads(str(z["results"])):
    print("   ", r)
  print("[replay] controls: SPACE pause | arrows step | PgUp/PgDn jump | HOME/END | "
        "UP/DOWN speed | slider window (if Tk)")
  print("[replay] geom groups (number keys): 0=parts 1=floor/table 2=arm-collision-capsules "
        "3=arm-visual-meshes  (capsules start HIDDEN when meshes exist)")
  from mujoco import viewer as _viewer
  mpos = z["mpos"] if "mpos" in z.files else None          # mocap tracks (arms / EE pucks)
  mquat = z["mquat"] if "mquat" in z.files else None
  mark_keys = sorted(marks)

  st = {"i": 0.0, "playing": True, "speed": float(speed), "scrub": False}

  def key_cb(key):                                         # GLFW keycodes
    if key == 32:                                          # SPACE
      st["playing"] = not st["playing"]
    elif key == 263:                                       # LEFT
      st["playing"] = False; st["i"] = max(0.0, st["i"] - 1)
    elif key == 262:                                       # RIGHT
      st["playing"] = False; st["i"] = min(T - 1.0, st["i"] + 1)
    elif key == 267:                                       # PGDN: back 5%
      st["i"] = max(0.0, st["i"] - T * 0.05)
    elif key == 266:                                       # PGUP: fwd 5%
      st["i"] = min(T - 1.0, st["i"] + T * 0.05)
    elif key == 268:                                       # HOME
      st["i"] = 0.0
    elif key == 269:                                       # END
      st["playing"] = False; st["i"] = T - 1.0
    elif key == 265:                                       # UP
      st["speed"] = min(16.0, st["speed"] * 2)
    elif key == 264:                                       # DOWN
      st["speed"] = max(0.0625, st["speed"] / 2)

  # optional Tk slider window (viser-style scrub bar)
  root = scale = lbl = None
  try:
    import tkinter as tk
    root = tk.Tk()
    root.title(f"replay: {path.split('/')[-1]}")
    root.geometry("640x110")
    lbl = tk.Label(root, text="", anchor="w", font=("monospace", 9))
    lbl.pack(fill="x", padx=8, pady=(6, 0))

    def on_scale(val):
      if st["scrub"]:                                      # user drag (not our .set feedback)
        st["i"] = float(val)
    scale = tk.Scale(root, from_=0, to=T - 1, orient="horizontal", showvalue=True,
                     length=620, command=on_scale)
    scale.pack(fill="x", padx=8)
    scale.bind("<ButtonPress-1>", lambda e: st.update(scrub=True, playing=False))
    scale.bind("<ButtonRelease-1>", lambda e: st.update(scrub=False))
    row = tk.Frame(root); row.pack(pady=2)
    tk.Button(row, text="play/pause", command=lambda: st.update(playing=not st["playing"])
              ).pack(side="left", padx=4)
    tk.Button(row, text="-", command=lambda: st.update(speed=max(0.0625, st["speed"] / 2))
              ).pack(side="left")
    tk.Button(row, text="+", command=lambda: st.update(speed=min(16.0, st["speed"] * 2))
              ).pack(side="left")
    for frac, txt in ((0.0, "|<"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%")):
      tk.Button(row, text=txt, command=lambda f=frac: st.update(i=f * (T - 1))
                ).pack(side="left", padx=2)
  except Exception as e:                                   # headless Tk / no display: keys only
    print(f"[replay] no Tk slider ({e}); keyboard controls still active")
    root = None

  def last_mark(i):
    import bisect
    k = bisect.bisect_right(mark_keys, i) - 1
    return marks[mark_keys[k]] if k >= 0 else "(start)"

  prev_shown = -1
  with _viewer.launch_passive(model, data, key_callback=key_cb) as v:
    _default_geomgroups(v, model.nmesh > 0)               # meshes on, collision capsules off
    while v.is_running():
      i = int(st["i"])
      if i != prev_shown:
        data.qpos[:] = qpos[i]
        if mpos is not None and len(mpos):
          data.mocap_pos[:] = mpos[i]
          data.mocap_quat[:] = mquat[i]
        mujoco.mj_forward(model, data)
        prev_shown = i
      v.sync()
      if root is not None:
        try:
          if not st["scrub"]:
            scale.set(i)
          lbl.config(text=f"t={i*DT*RECORD_EVERY:7.2f}s  frame {i}/{T-1}  "
                          f"speed x{st['speed']:g}  {'PLAYING' if st['playing'] else 'PAUSED'}"
                          f"  |  {last_mark(i)}")
          root.update()
        except Exception:                                  # window closed -> keys only
          root = None
      if st["playing"] and not st["scrub"]:
        st["i"] += st["speed"]                             # frames advance at speed x realtime
        if st["i"] >= T:
          st["i"] = 0.0 if loop else T - 1.0
          if not loop:
            st["playing"] = False
      time.sleep(DT * RECORD_EVERY)
  if root is not None:
    try:
      root.destroy()
    except Exception:
      pass


# ---------------------------------------------------------------------------------------------
# CLI: EXECUTE a saved plan in physics (`run`), or view a recording (`replay`).
# We do NOT solve here -- run the SOLVER with main.py (`--save plan.json`); sim just consumes
# that. The scene is rebuilt with replay.build_scene (the same reconstruction the rai replay
# uses), so there is a single source of truth for the scene and zero solver duplication.
# ---------------------------------------------------------------------------------------------
_STANDARD_SIM_PROFILE = {
    # Persisted real-arm path and the established collision-audited timing pipeline.
    "actuated": True,
    "skill_servo": True,
    "arm_arm_contacts": True,
    "spline": True,
    "phase_schedule": True,
    "phase_servo_time": 0.15,
    "phase_smooth": True,
    "phase_smooth_time": 0.12,
    "phase_batch": True,
    "phase_batch_min_motion": 0.08,
    "phase_batch_max_gap": 0.35,
    "phase_audit_resolution": 0.002,
    # Runtime convergence/contact guards. The normalized wrench guard is intentionally
    # profile-owned: its historical global default is off for old artifact reproduction.
    "settle_tol": 0.002,
    "settle_max": 1.0,
    "servo_pick": True,
    "servo_force": 6.0,
    "servo_max_dev": 0.010,
    "servo_torque_guard": 0.6,
    "servo_tol": 0.002,
    # Wrench-aware rate tables remain a physical backstop even though phase scheduling uses
    # the visual cruise directly. Corner impulses are part of that contract.
    "force_limits": True,
    "corner_limits": True,
    "inertia_factor": 1.0,
    "wrench_margin": 1.0,
    # Fail closed on a saved diagonal that violates the MP geometry or joint limits.
    "path_preflight": True,
    "path_preflight_tolerance": PATH_PREFLIGHT_TOL,
    "path_preflight_resolution": 0.002,
    # Robot-independent visual/physics baseline. FR3 currently overrides these explicitly.
    "joint_rate": 1.0,
    "mass_scale": 1.0,
    "coll_scale": 1.0,
    # Make the selected algorithm unambiguous; these remain available as explicit ablations.
    "corridor": False,
    "corridor_online": False,
    "pace": False,
    "phase_intent": False,
    "c2_spline": False,
    "limit_rate": False,
    # A paper reproduction consumes a persisted path and returns a failing process status for
    # any physical failure. Historical ad-hoc runs retain their permissive defaults.
    "require_mp_path": True,
    "require_success": True,
}

_SIM_PROFILES = {"standard": dict(_STANDARD_SIM_PROFILE)}


def _gripper_kinds_from_cfg(cfg, robots):
  ks = [k.strip() for k in (cfg.get("grippers") or "").split(",") if k.strip()]
  if ks:                                                    # mixed EEs (--grippers CSV)
    ks = (ks + [ks[-1]] * len(robots))[:len(robots)]
    return {r: ks[i] for i, r in enumerate(robots)}
  return {r: ("vacuum" if _saved(cfg, "vacuum") else "two_finger") for r in robots}


def _states_from_plan(cfg, C, asm, robots, plan):
  """Re-derive the discrete state trajectory the same way tests/_replay_ok does -- the saved
  JSON carries only actions, and the successor flags (gripper kind, bracing) must match the
  search that produced it (both are in the saved config)."""
  import assembly as A
  # MIXED-EE aware: gripper_kind(C) says "vacuum" whenever ANY vacuum frame exists, which
  # prices every arm's grasps with vacuum caps on a mixed scene -- the successor set then
  # cannot reproduce two-finger actions (user-hit on the chair: "cannot reproduce action 2").
  gk = A.wrench_kind_arg(C, robots) if _saved(cfg, "wrench") else None
  brace = bool(_saved(cfg, "bracing", legacy=False))
  _, gt = A._search_context(C, asm, robots)
  states = [A.get_initial_state(C, asm)]
  for i, act in enumerate(plan):
    ns = next((ns for a, ns in A.successors(states[-1], asm, robots, gt,
                                            gripper_kind=gk, bracing=brace)
               if a == act), None)
    if ns is None:
      # a bare StopIteration here cost a debugging session -- name the action and the cause
      # class (FRAGILITY: successor-affecting setting differing from the search, usually an
      # unsaved flag; ignore_gravity was exactly this).
      raise SystemExit(
          f"[sim] cannot reproduce action {i} {act!r} from the successor set -- a successor-"
          f"affecting setting differs from the search that produced this plan (wrench caps, "
          f"bracing, ignore_gravity, min_sep/place_cells, ...). If the plan predates a flag "
          f"being SAVED, re-solve it; the saved config block is what the sim replays.")
    states.append(ns)
  return states


def _restore_saved_wrench_config(cfg):
  """Restore the successor/physics wrench model before replaying a saved plan.

  ``sim.py`` historically imported ``wrench.PARAMS`` at module defaults.  A plan carrying a
  non-default grasp model, insertion load, gravity policy, or calibrated cap was therefore
  re-derived and executed under different mechanics (car-mini persisted T_close=0.5 Nm while
  MuJoCo still broke it at the 0.4 Nm default).  Scene construction and simulation now share
  replay's one idempotent reset-and-apply path; search-only printing remains in main.py.
  """
  import wrench as W
  # replay.build_scene has already applied this configuration once. Reuse its idempotent
  # reset-and-apply implementation rather than multiplying saved cap scales a second time
  # (e.g. --two-finger-scale 4 used to become x16 in simulation).
  from replay import apply_wrench_params
  apply_wrench_params(cfg)

  _tf = W.PARAMS["grasp"][_saved(cfg, "grasp_model")]["two_finger"]
  print(f"[sim] restored saved wrench model: {_saved(cfg, 'grasp_model')}, "
        f"profile={_saved(cfg, 'wrench_profile') or 'provisional-default'}, "
        f"two-finger caps={_tf}, "
        f"insert-force={W.PARAMS['insert_force']['_default']:g}N, "
        f"ignore-gravity={bool(W.PARAMS['ignore_gravity'])}.")


def _resolve_directional_caps(cfg, cli_value):
  """Explicit sim flag wins; otherwise an anisotropic named cap profile keeps its axes."""
  if cli_value is not None:
    return bool(cli_value)
  return _saved(cfg, "wrench_profile") == "max_robotiq"


def _cli_run(a):
  global JOINT_RATE, ARM_ARM_CONTACTS
  if getattr(a, "sim_profile", None):
    print("[profile] sim 'standard' resolved: saved real-arm path, actuated skill servo, "
          "collision-audited phase schedule + smoothing/batching, strict preflight, and "
          "0.6 normalized wrench guard. Explicit individual flags remain authoritative.")
  if a.require_mp_path and not a.mp_path:
    raise SystemExit(
        "[sim] --require-mp-path: this reproduction profile will not run a fresh stochastic "
        "motion planner. Supply --mp-path MP.json, or explicitly pass --no-require-mp-path "
        "for an exploratory run.")
  if a.phase_schedule and a.corridor_online:
    raise SystemExit("[sim] --phase-schedule and --corridor-online are mutually exclusive")
  if a.phase_servo_time < 0.0:
    raise SystemExit("[sim] --phase-servo-time must be non-negative")
  if a.phase_smooth and not a.phase_schedule:
    raise SystemExit("[sim] --phase-smooth requires --phase-schedule")
  if a.phase_batch and not a.phase_schedule:
    raise SystemExit("[sim] --phase-batch requires --phase-schedule")
  if a.phase_intent and not a.phase_schedule:
    raise SystemExit("[sim] --phase-intent requires --phase-schedule")
  if a.phase_intent and a.phase_batch:
    raise SystemExit("[sim] --phase-intent and --phase-batch are alternative final stages")
  if a.c2_spline and not (a.corridor or a.phase_schedule):
    raise SystemExit("[sim] --c2-spline requires --corridor or --phase-schedule")
  if a.phase_smooth_time <= 0.0:
    raise SystemExit("[sim] --phase-smooth-time must be positive")
  if a.phase_batch_min_motion <= 0.0 or a.phase_batch_max_gap <= 0.0:
    raise SystemExit("[sim] --phase-batch thresholds must be positive")
  if a.phase_intent_ramp <= 0.0:
    raise SystemExit("[sim] --phase-intent-ramp must be positive")
  if a.phase_intent_rate <= 0.0:
    raise SystemExit("[sim] --phase-intent-rate must be positive")
  if a.phase_audit_resolution < 0.0:
    raise SystemExit("[sim] --phase-audit-resolution must be non-negative")
  if a.process_engagement_distance < 0.0:
    raise SystemExit("[sim] --process-engagement-distance must be non-negative")
  if a.process_wrench_ramp < 0.0:
    raise SystemExit("[sim] --process-wrench-ramp must be non-negative")
  if a.process_noise != "nominal" and not a.apply_process_wrench:
    raise SystemExit("[sim] --process-noise requires --apply-process-wrench")
  if a.object_perturb_xy < 0.0 or a.object_perturb_yaw < 0.0:
    raise SystemExit("[sim] --object-perturb-xy/--object-perturb-yaw must be non-negative")
  if a.object_perturb_max_tries < 1:
    raise SystemExit("[sim] --object-perturb-max-tries must be at least one")
  if a.object_grasp_capture <= 0.0 or a.object_grasp_capture_angle <= 0.0:
    raise SystemExit("[sim] object grasp capture tolerances must be positive")
  if a.animate_grippers and not (a.actuated or a.articulated):
    raise SystemExit("[sim] --animate-grippers requires --actuated or --articulated")
  if a.gripper_motion_time <= 0.0:
    raise SystemExit("[sim] --gripper-motion-time must be positive")
  _has_object_perturb = bool(a.object_perturb_xy > 0.0 or a.object_perturb_yaw > 0.0)
  if (_has_object_perturb and a.object_perturb_observation == "observed"
      and not (a.actuated and a.skill_servo)):
    raise SystemExit(
        "[sim] observed object perturbations need --actuated --skill-servo for terminal "
        "pick retargeting (the standard sim profile enables both). Use "
        "--object-perturb-observation unobserved for an open-loop counterfactual.")
  if a.c2_samples_per_seg < 1:
    raise SystemExit("[sim] --c2-samples-per-seg must be at least 1")
  if not 0.0 < a.c2_control_ratio <= 1.0:
    raise SystemExit("[sim] --c2-control-ratio must be in (0, 1]")
  if a.c2_smooth_weight < 0.0:
    raise SystemExit("[sim] --c2-smooth-weight must be non-negative")
  if a.c2_max_deviation <= 0.0:
    raise SystemExit("[sim] --c2-max-deviation must be positive")
  if a.c2_resolution <= 0.0:
    raise SystemExit("[sim] --c2-resolution must be positive")
  _aac = getattr(a, "arm_arm_contacts", None)
  ARM_ARM_CONTACTS = bool(a.phase_schedule) if _aac is None else bool(_aac)
  if ARM_ARM_CONTACTS:
    print("[sim] arm-arm contacts ACTIVE: per-robot contact masks enable cross-arm physics "
          "but exclude same-chain self contacts -- physical backstop for the phase "
          "schedule's exact geometric collision certificate.")
  if getattr(a, "joint_rate", None):
    JOINT_RATE = float(a.joint_rate)                          # per-arm cruise speed override
    print(f"[sim] --joint-rate {JOINT_RATE:g} rad/s (default 1.0)")
  import assembly as A
  import replay as R
  from plan_io import read_plan
  cfg, plan = read_plan(a.path)
  C, asm = R.build_scene(cfg)                               # exact reconstruction (shared)
  robots = A.get_robots(C)
  _restore_saved_wrench_config(cfg)                         # before successor replay + physics
  # A named anisotropic hardware envelope would lose its Mx/My versus Mz distinction under
  # the legacy scalar min-over-directions break monitor.  Select the component-wise monitor
  # by default for max_robotiq, while retaining an explicit --no-directional-caps ablation and
  # leaving every historical/provisional plan byte-compatible.
  _directional_was_unspecified = a.directional_caps is None
  a.directional_caps = _resolve_directional_caps(cfg, a.directional_caps)
  if _directional_was_unspecified:
    if a.directional_caps:
      print("[sim] max_robotiq selects component-wise breakaway caps by default; "
            "--no-directional-caps restores the conservative scalar ablation.")
  # debug knob: shrink arm collision geometry BEFORE witnesses/problem/sim so IK, the
  # collision oracle and the MuJoCo capsules stay mutually consistent. 1.0 = no-op.
  scale_arm_collision_shapes(C, robots, getattr(a, "coll_scale", 1.0))
  gripper_kinds = _gripper_kinds_from_cfg(cfg, robots)
  # ARM VISUAL MESHES ARE FOR EYES ONLY: the collision model is the capsules either way, so on a
  # headless run every mesh body is pure cost -- a rai getPose, a mocap write and a
  # mj_kinematics slot per step. cross-new carries 60 mesh bodies against 30 real ones, and
  # dropping them took a 2m15s run to 1m57s with a BIT-IDENTICAL result. Kept whenever anything
  # actually looks at them (--gui or --record), and --no-arm-meshes still forces them off.
  _want_meshes = (not a.no_arm_meshes) and bool(getattr(a, "gui", False) or a.record)
  if not _want_meshes and not a.no_arm_meshes:
    print("[sim] headless: skipping arm VISUAL meshes (capsules-only collision is unchanged). "
          "Pass --gui/--record to keep them.")
  # MASSES MUST STILL BE THE PLANNING MASSES HERE. _states_from_plan re-simulates the saved
  # actions through the wrench-gated successor generator, so scaling first makes it judge a
  # DIFFERENT action set than the search did and the saved action stops being reachable
  # (measured on cross-new at --mass-scale 0.25: step 6 assemble a3 obj_2 obj_0 had ZERO
  # assemble successors -> StopIteration). Lighter parts are not automatically safer: a fixed
  # --insert-force presses against less self-weight, so the LP calls the mate unstable.
  states = _states_from_plan(cfg, C, asm, robots, plan)
  try:
    initial_state, object_perturbation = _sample_initial_placement_perturbation(
        C, asm, states[0], xy=a.object_perturb_xy, yaw_deg=a.object_perturb_yaw,
        seed=a.object_perturb_seed, max_tries=a.object_perturb_max_tries)
  except ValueError as e:
    raise SystemExit(f"[sim/placement] {e}")
  if _has_object_perturb:
    print(f"[sim/placement] collision-free table-pose draw seed={a.object_perturb_seed}: "
          f"xy +/-{a.object_perturb_xy*1000:g}mm, yaw +/-{a.object_perturb_yaw:g}deg "
          f"({object_perturbation['attempts']} draw(s)); controller "
          f"{a.object_perturb_observation}.")
  if getattr(a, "mass_scale", 1.0) != 1.0:
    # scale the ASSEMBLY masses, so MuJoCo's bodies and the wrench estimate stay consistent
    # (build_mjcf reads assembly.masses too). A one-sided estimate-only scale would be a lie.
    # AFTER the replay: the plan is fixed, and the point of the knob is to ask whether THAT
    # plan survives different masses -- not to re-derive which plan the search would pick.
    for _p in list(asm.masses):
      asm.masses[_p] = asm.masses[_p] * a.mass_scale
    asm.DEFAULT_MASS = getattr(asm, "DEFAULT_MASS", 0.1) * a.mass_scale
    print(f"[sim] --mass-scale {a.mass_scale:g}: part masses scaled (sim bodies AND the "
          f"wrench estimate)")
  print(f"[sim] {a.path}: {cfg['assembly']} ({cfg['parts']}p, {cfg['num_robots']}R), "
        f"{len(plan)} actions; executing real-arm motion plan ...")

  if cfg.get("floating_ee"):
    raise SystemExit(
      "[sim] execution needs a REAL-ARM plan: save from a run WITHOUT --floating-ee (and "
        "with --real-ik). This plan is floating-EE.")
  from solver import RobotIK, optimize_witnesses
  import mp_export as MP
  gk = A.wrench_kind_arg(C, robots) if cfg.get("wrench") else None   # mixed-EE aware
  rik = RobotIK(C, robots)                                # geometric export only -- the
  # RobotIK's own default is ik_seeding=False, but the CONFIG default is True and main.py
  # turns it on before it exports; every other export path (mp_show_path, mp_debug_view,
  # mp_show_modes, mp_viser_witness) sets it explicitly too. Leaving it off HERE made the
  # witness IK fail for re-picks and reorienting places -- optimize_witnesses returned None
  # and mp_export silently substituted the arm's PREVIOUS config, so the part was never
  # carried anywhere (measured on car-mini: obj_0's two reorienting places became 0.07deg
  # no-ops, the base stayed flat, and every wheel then missed its mate by ~70mm).
  rik.ik_seeding = bool(_saved(cfg, "ik_seeding"))
  # VERBATIM from the saved config -- 0.0 means NO pre-insertion split. (A silent `or 0.05`
  # default here rebuilt the problem WITH split tasks while the saved path used unsplit
  # task ids -> every attach after the first assemble fired at the wrong task; user-hit.)
  rik.pre_insertion_delta = float(_saved(cfg, "pre_insertion", legacy=0.0))
  # ...and the OTHER four splits, for the same reason: each one ADDS a task, so a rebuild
  # that misses them produces a shorter task list than the saved --mp-path was planned
  # against, and the path indexes tasks by integer id (the signature guard below catches it,
  # but only if we even try). pre_insertion alone was restored here.
  rik.pre_pick_delta = float(_saved(cfg, "pre_pick", legacy=0.0))
  rik.pre_place_delta = float(_saved(cfg, "pre_place", legacy=0.0))
  rik.pre_add_grasp_delta = float(_saved(cfg, "pre_add_grasp", legacy=0.0))
  rik.post_release_delta = float(_saved(cfg, "post_release", 0.0))
  # ...and the eq WEIGHTS, which decide how exactly a mate/place lands. Both were saved but
  # never restored here, so the sim re-solved every witness at the default 1e1 and got
  # different keyframes than the export it is supposed to be replaying. (LEGACY None: the
  # 1e2 defaults came later; pre-flag plans were solved at rai's 1e1 = flag off.)
  _mew = _saved(cfg, "mate_eq_weight", legacy=None)
  if _mew is not None:
    rik.mate_eq_weight = float(_mew)
  _pew = _saved(cfg, "place_eq_weight", legacy=None)
  if _pew is not None:
    rik.place_eq_weight = float(_pew)
  _ma = _saved(cfg, "mate_accuracy")
  if _ma is not None:
    rik.mate_residual_reject = float(_ma)
  _ga = _saved(cfg, "grasp_accuracy")
  if _ga is not None:
    rik.grasp_pose_reject = float(_ga)
  _gew = _saved(cfg, "grasp_eq_weight")
  if _gew is not None:
    rik.grasp_eq_weight = float(_gew)
  _ika = _saved(cfg, "ik_accuracy")
  if _ika is not None:
    rik.nlp_stop_tolerance = float(_ika)
  _wm = _saved(cfg, "witness_margin")
  if _wm is not None:
    rik.clearance_reject = float(_wm)   # robust-keyframe gate, as exported
  _wjm = _saved(cfg, "witness_joint_margin")
  if _wjm is not None:
    rik.joint_margin_reject = float(_wjm)
  if _saved(cfg, "witness_robot_collisions", legacy=False):
    rik.strip_inactive_collision = False
  _sc = _saved(cfg, "strict_collisions", legacy=None)
  if _sc is not None:
    rik.strip_inactive_collision = False
    rik.penetration_reject = float(_sc)
    rik.col_weight = 1e2
  # successor flags MUST match the search that produced the plan, or the state re-derivation
  # inside optimize_witnesses can't reproduce an action (e.g. car-mini's add_grasp needs
  # bracing=True). bracing is saved; handovers/allow_place aren't (they default True).
  brace = bool(_saved(cfg, "bracing", legacy=False))
  _hand = bool(_saved(cfg, "handovers"))
  _aplace = bool(_saved(cfg, "allow_place"))
  _saved_mp_path = None
  _final_export = None
  if a.mp_path:
    _saved_mp_path = MP.load_path(a.mp_path)
    try:
      _final_export = MP.load_export_bundle(
          a.mp_path, plan, list(rik.C.getJointNames()), required=False)
    except ValueError as e:
      raise SystemExit(f"[sim/mp] invalid final export: {e}")
    if _final_export is None and not getattr(a, "refresh_witnesses", False):
      raise SystemExit(
          "[sim/mp] saved motion path has no final post-fusion export block. Refusing to "
          "re-solve IK during replay. Re-export the path with current --mp-save, or pass "
          "--refresh-witnesses to explicitly authorize the legacy rebuild.")
    if _final_export is not None and getattr(a, "refresh_witnesses", False):
      print("[sim/mp] --refresh-witnesses explicitly requested: ignoring the embedded final "
            "export and running the legacy witness/pre-skill rebuild.")
      _final_export = None
    if _final_export is not None:
      _es = _final_export["settings"]
      rik.pre_insertion_delta = float(_es.get("pre_insertion", rik.pre_insertion_delta))
      rik.pre_pick_delta = float(_es.get("pre_pick", rik.pre_pick_delta))
      rik.pre_place_delta = float(_es.get("pre_place", rik.pre_place_delta))
      rik.pre_add_grasp_delta = float(
          _es.get("pre_add_grasp", rik.pre_add_grasp_delta))
      rik.post_release_delta = float(
          _es.get("post_release", rik.post_release_delta))
      brace = bool(_es.get("bracing", brace))
      print(f"[sim/mp] final post-fusion export embedded in {a.mp_path}: "
            f"{sum(w is not None for w in _final_export['witnesses'])} skill and "
            f"{len(_final_export['pre_witnesses'])} pre-skill keyframes; IK solves: 0.")
  # WITNESS PERSISTENCE: the plan file may carry the export-time optimized keyframes
  # (main.py --save writes them since 2026-07-29). The sim restores every witness-shaping
  # setting from the SAME file's config block, so a present block is by construction the
  # set this rebuild would try to reproduce -- reusing it removes both the re-solve time
  # and the last source of witness drift between the export and the sim.
  wit = _final_export["witnesses"] if _final_export is not None else None
  if wit is None and not getattr(a, "refresh_witnesses", False):
    from plan_io import read_witnesses
    wit = read_witnesses(a.path, list(rik.C.getJointNames()), len(plan))
    if wit is not None:
      print(f"[sim/mp] {sum(w is not None for w in wit)} persisted witnesses loaded from "
            f"{a.path} -- witness re-solve SKIPPED (--refresh-witnesses to re-solve).")
  if not getattr(a, "refresh_witnesses", False):
    if wit is None:
      raise SystemExit(
          "[sim/mp] plan has no compatible persisted search-witness certificate. "
          "Refusing to resolve IK implicitly; regenerate the plan with current main.py, "
          "or pass --refresh-witnesses for the explicit legacy repair path.")
    from plan_io import require_complete_witnesses
    try:
      require_complete_witnesses(plan, wit, who="sim replay")
    except ValueError as e:
      raise SystemExit(
          f"[sim/mp] {e}. Refusing an incomplete artifact; regenerate it or pass "
          "--refresh-witnesses explicitly.")
  # WHICH witness optimizer ran at export time, reproduced here. build_tasks SKIPS a task
  # whose witness is None, so a different optimizer -> a different None-pattern -> a task
  # chain that no longer matches the saved --mp-path ("mode alignment failed: N unmatched").
  _witopt = cfg.get("optimize_witnesses")
  if wit is not None:
    pass
  elif _witopt == "floor":
    from witness_floor import optimize_witnesses_floor
    wit, _ = optimize_witnesses_floor(
        C, asm, plan, robots, rik, gripper_kind=gk, handovers=_hand, allow_place=_aplace,
        bracing=brace, branch_budget=_saved(cfg, "witness_branch_budget"),
        branch_metric=_saved(cfg, "witness_branch_metric"),
        reconfig_thresh=_saved(cfg, "witness_reconfig_thresh"),
        draw_seed=_saved(cfg, "witness_draw"))
  elif _witopt:
    wit, _ = optimize_witnesses(C, asm, plan, robots, rik, cost=_witopt,
                                gripper_kind=gk, handovers=_hand, allow_place=_aplace,
                                bracing=brace, clear_weight=float(_saved(cfg, "witness_clearance")),
                                robot_collisions=bool(_saved(cfg, "witness_robot_collisions", legacy=False)),
                                verbose=False)
  else:
    wit, _ = optimize_witnesses(C, asm, plan, robots, rik, cost="ee", sweeps=0,
                                gripper_kind=gk, bracing=brace, handovers=_hand,
                                allow_place=_aplace,
                                robot_collisions=bool(_saved(cfg, "witness_robot_collisions", legacy=False)),
                                verbose=False)
  # depgraph=False is DELIBERATE and must stay: execute_motion_plan consumes a SEQUENCE-style
  # mode chain (one stretch per mode, in order). Rebuilding main's depgraph problem here to
  # "match the export" instead broke alignment on the sim's OWN planned path (measured: 6/11/
  # 15/14 unmatched modes across the pre-split configs, where sequence ran clean). Matching a
  # saved --mp-path needs the chain PERSISTED with the path, not re-derived under a graph the
  # executor cannot walk.
  # ROBOT-ROBOT MARGIN, as the export applies it. A CARRIED part inherits its holder (frames
  # are owned by kinematic root), so the margin covers part-vs-OTHER-ARM -- while a RESTING
  # part is owned by nobody and keeps zero clearance to the table, which a place needs.
  # sim's internal planning never passed this, so --mp-robot-margin was inert on this route.
  _es = _final_export["settings"] if _final_export is not None else {}
  prob = MP.make_problem(
      C, asm, plan, wit, robots, rik, gk, bracing=brace,
      depgraph=bool(_es.get("depgraph", False)),
      collision_tolerance=float(_es.get("collision_tolerance", a.mp_tolerance)),
      pre_witnesses=(_final_export["pre_witnesses"] if _final_export is not None else None),
      robot_robot_margin=float(_es.get(
          "robot_robot_margin", _saved(cfg, "mp_robot_margin", legacy=0.0))),
      robot_margin_ramp=float(_es.get(
          "robot_margin_ramp", _saved(cfg, "mp_margin_ramp", legacy=0.0))),
      table_margin=float(_es.get("table_margin", _saved(cfg, "mp_table_margin"))),
      table_float=float(_es.get(
          "table_float", (float(_saved(cfg, "part_float", legacy=0.015))
                          if _saved(cfg, "mp_table_margin") else 0.0))),
      witness_sets=(_final_export["witness_sets"] if _final_export is not None else None),
      resolve_missing_pre=(_final_export is None),
      task_records=(None if _final_export is None else _final_export["tasks"]),
      saved_step_names=(None if _final_export is None else _final_export["step_names"]),
      saved_sequence_names=(None if _final_export is None else
                            _final_export["sequence_names"]))
  if a.mp_path:                                          # load a saved BiRRT path -> no re-plan
    path = _saved_mp_path
    print(f"[sim] loaded motion path ({len(path)} states) from {a.mp_path} -- skipping BiRRT")
    # ALIGNMENT GUARD: the path's mode task-ids index MAIN's task list; the rebuilt problem
    # must have the identical task structure or every attach fires at the wrong time.
    _want = getattr(MP.load_path, "last_signature", None)
    if _want is not None:
      _have = MP.task_signature(prob)
      if _have != _want:
        raise SystemExit(
            f"[sim] !!! TASK-SIGNATURE MISMATCH: the path was planned against task structure "
            f"{_want}, the rebuild produced {_have}. Same task COUNT can still mean different "
            f"tasks (a moved --pre-* split, a different bracing/handovers flag), and the path "
            f"indexes tasks by integer id -- so every attach would fire at the wrong moment. "
            f"Aborting. Re-export the path from this plan, or use the plan the path came from.")
    max_tid = max(int(t) for st in path for t in np.atleast_1d(st.mode.task_ids))
    if max_tid + 1 != len(prob.tasks):
      raise SystemExit(
          f"[sim] !!! TASK-STRUCTURE MISMATCH: saved path references {max_tid + 1} tasks, "
          f"rebuilt problem has {len(prob.tasks)}. The plan.json/mp.json pair does not "
          f"match this rebuild (usually a pre-insertion/--skill difference: saved config "
          f"has pre_insertion={cfg.get('pre_insertion')}). Aborting -- executing would "
          f"relink at the wrong times.")
    # HEAL DROPPED ZERO-MOTION MODES: shortcutting can erase a zero-motion task's whole
    # mode stretch from a saved path (see MP.repair_mode_chain), and the fire extraction
    # derives events from mode transitions -- on chair-8 the holder-release never fired and
    # the arm dragged the chair to its next pick. Dirty files on disk heal here on load.
    path, _ = MP.repair_mode_chain(
        path, MP.mode_sequence(prob),
        lambda ref, m: MP._PathState(ref.q,
                                     MP._Mode([int(t) for t in np.atleast_1d(m.task_ids)])))
  else:
    # seed precedence: sim CLI > the seed the plan was SOLVED/exported with > unseeded.
    _mseed = a.mp_seed if a.mp_seed is not None else cfg.get("mp_seed")
    path, _ = MP.run_birrt(prob, runtime=a.mp_runtime, verbose=a.verbose, seed=_mseed,
                           optimize=a.mp_optimize,
                           fake_clock=bool(_saved(cfg, "mp_fake_clock")))
  if not path:
    raise SystemExit("[sim] motion planner found no path.")
  if a.mp_cograsp:
    print("[flag] --mp-cograsp ACTIVE: checking co-grasp rigidity along the loaded path.")
    MP.report_cograsp_drift(C, prob, path, rik)
  _run_metadata = {
      "profile": a.sim_profile,
      "plan_path": a.path,
      "motion_path": a.mp_path,
      "dt": DT,
      "record_every": RECORD_EVERY,
      "apply_process_wrench": bool(a.apply_process_wrench),
      "animate_grippers": bool(a.animate_grippers),
      "gripper_motion_time": float(a.gripper_motion_time),
      "table_mu": _saved(cfg, "table_mu", legacy=None),
      "resolved_profile": {k: getattr(a, k) for k in _STANDARD_SIM_PROFILE},
  }
  rep = execute_motion_plan(C, asm, plan, states, prob, path, gripper_kinds=gripper_kinds,
                            record=a.record, gui=a.gui, visual=_want_meshes,
                            record_metadata=_run_metadata,
                            initial_state=initial_state,
                            object_perturbation=object_perturbation,
                            object_perturb_observation=a.object_perturb_observation,
                            object_grasp_capture=a.object_grasp_capture,
                            object_grasp_capture_angle=np.deg2rad(a.object_grasp_capture_angle),
                            verbose=a.verbose,
                            rate_boost=a.arm_rate,
                            corridor=(a.corridor or a.phase_schedule),
                            phase_schedule=a.phase_schedule,
                            phase_servo_time=a.phase_servo_time,
                            phase_smooth=a.phase_smooth,
                            phase_smooth_time=a.phase_smooth_time,
                            phase_batch=a.phase_batch,
                            phase_batch_min_motion=a.phase_batch_min_motion,
                            phase_batch_max_gap=a.phase_batch_max_gap,
                            phase_intent=a.phase_intent,
                            phase_intent_ramp=a.phase_intent_ramp,
                            phase_intent_rate=a.phase_intent_rate,
                            phase_audit_resolution=a.phase_audit_resolution,
                            corridor_ds=a.corridor_ds,
                            corridor_ds_lead=a.corridor_ds_lead,
                            corridor_lead_max=a.corridor_lead_max,
                            corridor_verify=a.corridor_verify,
                            corridor_refine=a.corridor_refine,
                            corridor_cross_only=a.corridor_cross_only,
                            corridor_online=a.corridor_online,
                            skill_servo=a.skill_servo, servo_force=a.servo_force,
                            compliant_follow=a.compliant_follow,
                            compliant_force=a.compliant_force,
                            servo_pick=a.servo_pick,
                            servo_max_dev=a.servo_max_dev, servo_tol=a.servo_tol,
                            servo_torque_guard=a.servo_torque_guard,
                            servo_place_seat=a.servo_place_seat,
                            rate_clamp=a.corridor_rate_clamp,
                            settle_tol=a.settle_tol, settle_max=a.settle_max,
                            pace=a.pace, dwell=a.dwell, pace_trace=a.pace_trace,
                            directional_caps=a.directional_caps,
                            mate_caps=_mate_caps_from_args(a),
                            couple_handovers=a.couple_handovers,
                            verify_efc_cache=a.verify_efc_cache,
                            force_limits=a.force_limits, wrench_margin=a.wrench_margin,
                            mass_scale=a.mass_scale, inertia_factor=a.inertia_factor,
                            corner_limits=a.corner_limits,
                            spline=a.spline, spline_per_seg=a.spline_per_seg,
                            spline_tension=a.spline_tension,
                            c2_spline=a.c2_spline,
                            c2_samples_per_seg=a.c2_samples_per_seg,
                            c2_control_ratio=a.c2_control_ratio,
                            c2_smooth_weight=a.c2_smooth_weight,
                            c2_max_deviation=a.c2_max_deviation,
                            c2_resolution=a.c2_resolution,
                            articulated=a.articulated, actuated=a.actuated,
                            use_planner_table=getattr(a, 'planner_table', False),
                            corridor_margin=getattr(a, 'corridor_margin', 0.0),
                            weld_timeconst=getattr(a, 'weld_timeconst', 0.0),
                            object_damping=a.object_damping,
                            limit_rate=getattr(a, 'limit_rate', False),
                            path_preflight=a.path_preflight,
                            path_preflight_tolerance=(
                                a.path_preflight_tolerance
                                if a.path_preflight_tolerance is not None else
                                PATH_PREFLIGHT_TOL),
                            path_preflight_resolution=a.path_preflight_resolution,
                            apply_process_wrench=a.apply_process_wrench,
                            table_mu=_saved(cfg, "table_mu", legacy=None),
                            process_noise_mode=a.process_noise,
                            process_noise_corner=a.process_noise_corner,
                            process_noise_seed=a.process_noise_seed,
                            process_engagement_distance=a.process_engagement_distance,
                            process_wrench_ramp=a.process_wrench_ramp,
                            animate_grippers=a.animate_grippers,
                            gripper_motion_time=a.gripper_motion_time)
  print(rep.summary())
  if a.report_json:
    import json
    from pathlib import Path
    _rp = Path(a.report_json)
    _rp.parent.mkdir(parents=True, exist_ok=True)
    _tmp = _rp.with_name(_rp.name + f".tmp.{os.getpid()}")
    with _tmp.open("w", encoding="utf-8") as _f:
      json.dump(rep.to_dict(_run_metadata), _f, indent=2, sort_keys=True)
      _f.write("\n")
    os.replace(_tmp, _rp)
    print(f"[sim] structured report -> {_rp}")
  if a.require_success and not rep.success:
    raise SystemExit(3)


def _build_cli():
  import argparse
  p = argparse.ArgumentParser(
      prog="sim.py",
      description="MuJoCo validation harness. EXECUTE a saved plan in physics (`run`) or view "
                  "a recording (`replay`). Solve with main.py --save; sim does not re-solve. "
                  "Set PYTHONHASHSEED=0 for reproducible runs.")
  sub = p.add_subparsers(dest="cmd", required=True)

  r = sub.add_parser("run", help="execute a saved plan (main.py --save) in physics")
  r.add_argument("path", help="plan JSON written by `main.py ... --save plan.json`")
  r.add_argument("--sim-profile", choices=tuple(_SIM_PROFILES), default=None, metavar="NAME",
                 help="Named execution defaults. 'standard' selects the fail-closed saved-"
                      "path reproduction pipeline: actuated skill servo, physical arm-arm "
                      "contacts, spline + phase schedule/smoothing/batching, normalized wrench "
                      "guard, and strict path preflight. Explicit individual flags win. "
                      "--mp-path/--record and robot-specific calibrations remain per run.")
  r.add_argument("--require-mp-path", action=argparse.BooleanOptionalAction, default=False,
                 help="Fail unless --mp-path supplies a persisted path; prevents a reproduction "
                      "command from silently running a fresh stochastic BiRRT. Enabled by the "
                      "standard profile.")
  r.add_argument("--require-success", action=argparse.BooleanOptionalAction, default=False,
                 help="Return process status 3 when physics reports any failed action. The NPZ "
                      "and summary are still written first. Enabled by the standard profile.")
  r.add_argument("--mp-path", metavar="FILE", default=None,
                 help="Load a saved BiRRT path (main --motion-plan --mp-save FILE) instead "
                      "of re-planning. Must be from the SAME run as the plan JSON.")
  r.add_argument("--mp-tolerance", type=float, default=1e-4,
                 help="Exported collision tolerance (1e-6 = zero-tol; the plan's witnesses "
                      "must be clean -- save it from a --strict-collisions run). Ignored with "
                      "--mp-path (the saved path already encodes it).")
  r.add_argument("--mp-seed", type=int, default=None, metavar="N",
                 help="Seed the internal BiRRT+shortcut draw. Default: reuse the mp_seed "
                      "saved with the plan (if any); unseeded only when neither is given. "
                      "Without a seed the same command yields DIFFERENT failure classes run "
                      "to run, so A/B comparisons are noise.")
  r.add_argument("--mp-optimize", action=argparse.BooleanOptionalAction, default=True,
                 help="BiRRT* keeps IMPROVING the path until --mp-runtime instead of "
                      "returning at the first solution -- the actual path-quality knob "
                      "(--mp-runtime alone is only a ceiling). With --mp-seed the budget is a "
                      "deterministic consultation budget, so optimized runs are reproducible "
                      "too. Ignored with --mp-path.")
  r.add_argument("--mp-runtime", type=float, default=30.0,
                 help="BiRRT budget (s); ignored with --mp-path")
  r.add_argument("--verify-efc-cache", action="store_true",
                 help="debug: re-derive the equality-row index EVERY step and assert it matches "
                      "the eq_active-keyed cache. Checks the MuJoCo layout assumption that the "
                      "equality block is a function of the active set alone. Slow; off by default.")
  r.add_argument("--couple-handovers", action="store_true",
                 help="corridor: also lock giver+receiver over a handover's stretch. A handover "
                      "is ATOMIC discretely (giver holds before, receiver after, never both), so "
                      "the state-scan coupling misses it entirely -- yet both grippers are on "
                      "the part in execution. OFF = byte-identical.")
  r.add_argument("--apply-process-wrench", action=argparse.BooleanOptionalAction, default=False,
                 help="Apply the planner's nominal mate process load during every final "
                      "entry-to-seat assemble stretch, even when collision contact does not "
                      "generate it. Equal-and-opposite forces act at the inserted and receiving "
                      "interfaces; screw skills also receive their declared pure torque. This "
                      "physically tests holders, grasp breakaway, and arm tracking. The robust "
                      "--insertion-noise set stays nominal unless --process-noise explicitly "
                      "selects a corner or seeded sample. Opt-in; use "
                      "--no-apply-process-wrench to reproduce the historical contact-only sim.")
  r.add_argument("--process-noise", choices=("nominal", "corner", "sample"),
                 default="nominal", metavar="MODE",
                 help="Realization of the saved --insertion-noise uncertainty set when "
                      "--apply-process-wrench is active. nominal uses its centre; corner uses "
                      "--process-noise-corner; sample draws once per assemble action from the "
                      "box using --process-noise-seed. A realization is fixed throughout its "
                      "skill and stored in the NPZ; it is never white noise per physics tick.")
  r.add_argument("--process-noise-corner", type=int, default=0, metavar="N",
                 help="Zero-based robust-box corner: 0..3 for lateral-only uncertainty, 0..15 "
                      "when a saved --insertion-offset is also active (default 0).")
  r.add_argument("--process-noise-seed", type=int, default=0, metavar="S",
                 help="Seed for --process-noise sample; one independent, persisted draw per "
                      "assemble action (default 0).")
  r.add_argument("--object-perturb-xy", type=float, default=0.0, metavar="M",
                 help="Bounded initial table-placement error: independently sample x/y for "
                      "each disconnected part group from [-M,M] metres (default 0 = off).")
  r.add_argument("--object-perturb-yaw", type=float, default=0.0, metavar="DEG",
                 help="Bounded initial yaw error in degrees. The authored support face, z, "
                      "roll and pitch are retained (default 0 = off).")
  r.add_argument("--object-perturb-seed", type=int, default=0, metavar="S",
                 help="Seed for the initial placement draw (default 0); the accepted draw is "
                      "persisted in the NPZ/report.")
  r.add_argument("--object-perturb-max-tries", type=int, default=200, metavar="N",
                 help="Maximum whole-scene rejection draws used to obtain a collision-free "
                      "perturbed initial placement. Exhaustion is a loud failure; the nominal "
                      "scene is never substituted (default 200).")
  r.add_argument("--object-perturb-observation", choices=("observed", "unobserved"),
                 default="observed", metavar="MODE",
                 help="observed retargets each terminal pick to the measured displaced part "
                      "with the skill servo; unobserved executes the nominal reference. Both "
                      "apply a physical capture gate before activating a grasp weld.")
  r.add_argument("--object-grasp-capture", type=float, default=0.010, metavar="M",
                 help="Maximum terminal translational residual for a perturbed-scene grasp "
                      "weld (default 0.010m). A larger miss is reported as not_grasped.")
  r.add_argument("--object-grasp-capture-angle", type=float, default=10.0, metavar="DEG",
                 help="Maximum terminal angular residual for a perturbed-scene grasp weld "
                      "(default 10deg).")
  r.add_argument("--process-engagement-distance", type=float, default=0.01, metavar="M",
                 help="Ordinary insertion force ramps from zero to full as the mate-interface "
                      "gap closes through the final M metres (default 0.01). Screw torque is "
                      "active over the whole certified helix.")
  r.add_argument("--process-wrench-ramp", type=float, default=0.05, metavar="S",
                 help="Smooth process-load turn-on time in seconds (default 0.05); avoids an "
                      "unphysical one-tick force impulse.")
  r.add_argument("--mate-force-cap", type=float, default=None, metavar="N",
                 help="FINITE MATE STRENGTH: max force a mate weld transmits before it breaks "
                      "(mate_broke, same sustained-overload rule as grasps). Default None = "
                      "infinite = the historical rigid-weld model, byte-identical. Giving "
                      "either mate cap enables monitoring; the other defaults to infinity. "
                      "This is what makes 'hold the car by one wheel' fail emergently: the "
                      "wheel-body weld carries the whole group and lets go.")
  r.add_argument("--mate-torque-cap", type=float, default=None, metavar="NM",
                 help="FINITE MATE STRENGTH: max torque a mate weld transmits before it "
                      "breaks. See --mate-force-cap.")
  r.add_argument("--directional-caps", action=argparse.BooleanOptionalAction, default=None,
                 help="breakaway uses the planner's DIRECTIONAL grasp polytope (force at the "
                      "grasp point + residual pure couple) instead of min-over-directions. More "
                      "permissive in force (F_push vs a blanket F_pull), STRICTER in torque. "
                      "OFF by default for provisional/legacy caps; max_robotiq enables it so "
                      "Mx/My=5Nm and Mz=3Nm remain distinct. --no-directional-caps explicitly "
                      "selects the conservative scalar ablation.")
  r.add_argument("--dwell", type=float, default=0.15, metavar="SEC",
                 help="--corridor-online: rest time of the acting arms before/after each fire "
                      "(default 0.15s). The settle hook already holds the arm until it is "
                      "converged, so a shorter dwell is safe and removes most of the standing "
                      "time at every barrier.")
  r.add_argument("--pace-trace", metavar="FILE", default=None,
                 help="--pace: save the per-tick stepper state (s, pace factors, rate "
                      "factors, fired) to FILE for offline analysis.")
  r.add_argument("--pace", action="store_true",
                 help="--corridor-online: FEED-FORWARD PHASE SCHEDULING. At each mode switch, "
                      "predict the complete next stretch once and choose one fixed rate "
                      "factor per corridor-coupled arm group. Acting groups meet at fire; "
                      "crossing groups reserve the hard --settle-max budgets of applicable "
                      "pre-fire hooks. No per-tick remaining-distance feedback; post-fire "
                      "cleanup overlaps the next mode, and runtime guards remain hard safety "
                      "overrides.")
  r.add_argument("--phase-schedule", action=argparse.BooleanOptionalAction, default=False,
                 help="Precompute a monotone coordination-space trajectory with one phase "
                      "per robot. Visual quality is primary: preferred cruise motion, few "
                      "unnecessary stops, and fixed nominal servo reservations; makespan is "
                      "only a weak consequence. Every accepted phase step and the complete "
                      "result are checked with the strict planner collision predicate. On a "
                      "coordination failure it falls back to the synchronized reference. "
                      "Incompatible with --corridor-online; replaces corridor Stage A/B/C.")
  r.add_argument("--phase-servo-time", type=float, default=0.15, metavar="SEC",
                 help="--phase-schedule: nominal feed-forward duration reserved for each "
                      "joint-settle or seat-servo ceremony (default 0.15s each, matching the "
                      "measured car-mini mean rather than the 1s worst-case cap). This is a "
                      "planning guess, not runtime feedback; execution still checks actual "
                      "convergence before the event and holds on an overrun.")
  r.add_argument("--phase-smooth", action=argparse.BooleanOptionalAction, default=False,
                 help="--phase-schedule: map each robot's monotone shared phase to its own "
                      "joint-space path arc and Gaussian-filter that physical RATE while "
                      "pinning every complete mode-switch phase vector, actor stops, sharp "
                      "corner slowdowns, and rigid coupling intervals. The joint paths are never "
                      "splined again. The result receives the same exact joint-path-resolution "
                      "pre+post-switch "
                      "audit; collision-sensitive fire intervals revert locally to the raw "
                      "certified schedule. Strict: if no independent phase trace exists or a "
                      "useful collision-free smooth result cannot be certified, abort before "
                      "physics instead of executing the unsmoothed schedule.")
  r.add_argument("--phase-smooth-time", type=float, default=0.12, metavar="SEC",
                 help="--phase-smooth kernel support in seconds (default 0.12). Larger values "
                      "remove longer rate stutters but are more likely to require a local "
                      "collision-certified fallback.")
  r.add_argument("--phase-batch", action=argparse.BooleanOptionalAction, default=False,
                 help="--phase-schedule: final common postprocessor for both the default and "
                      "C2 geometries. Merge a small motion island and its neighbouring motion "
                      "across a short stop into one smooth burst. Own task stops and rigid "
                      "couplings remain exact; crossing another robot's event is allowed only "
                      "after checking the global passage constraint and both event modes. "
                      "Every merge is peak-rate guarded and the result gets the exact "
                      "joint-path-resolution "
                      "collision audit. Opt-in; the established pipeline is unchanged.")
  r.add_argument("--phase-batch-min-motion", type=float, default=0.08, metavar="RAD",
                 help="--phase-batch: an adjacent motion island must be smaller than this "
                      "joint-space arc to be batched (default 0.08 rad).")
  r.add_argument("--phase-batch-max-gap", type=float, default=0.35, metavar="SEC",
                 help="--phase-batch: maximum stopped gap bridged into one motion "
                      "(default 0.35s).")
  r.add_argument("--phase-intent", action=argparse.BooleanOptionalAction, default=False,
                 help="--phase-schedule: experimental alternative to --phase-batch. Minimize "
                      "the temporal support of motion: between each robot's own events, pack "
                      "its intrinsic joint-space path arc into one delayed, quintic-ramped, "
                      "intent-rate-bounded motion. Foreign switches are not stop anchors; "
                      "own events and true rigid couplings remain exact. Multiple placements "
                      "are collision-screened and receive the final joint-path-resolution "
                      "audit. Strict: abort "
                      "if no support reduction survives.")
  r.add_argument("--phase-intent-ramp", type=float, default=0.15, metavar="SEC",
                 help="--phase-intent quintic velocity-ramp duration at each end of a packed "
                      "motion (default 0.15s).")
  r.add_argument("--phase-intent-rate", type=float, default=1.5, metavar="RAD_S",
                 help="--phase-intent target peak joint-space path rate (default 1.5rad/s). "
                      "This is deliberately above the ordinary 1rad/s visual cruise so motion "
                      "support can shrink, while remaining below the limited UR arm rates.")
  r.add_argument("--phase-audit-resolution", type=float, default=0.01, metavar="RAD",
                 help="--phase-schedule: maximum accumulated L-infinity joint-path motion "
                      "between final collision-audit samples (default 0.01rad, matching the "
                      "motion planner edge scale). Every event pose is still checked in both "
                      "modes. 0 restores dense checking at every 2ms controller tick.")
  r.add_argument("--settle-tol", type=float, default=2e-3, metavar="RAD",
                 help="--actuated: before each attach fires, hold the reference until the max "
                      "joint tracking error is under RAD. Default 2e-3: the actuated car has "
                      "a repeatable ~1.9mrad payload-biased floor, so the former 1e-3 default "
                      "waited the full 1s timeout without improving. Two mrad is roughly 2mm "
                      "at a 1m reach and matches the seat-servo scale; 0 disables the gate. "
                      "The gate still prevents firing during the much larger transient that "
                      "previously froze tracking residual into a new weld. A residual below "
                      "2x RAD that improves by less than 0.25x RAD over 0.1s is accepted as "
                      "the finite-gain servo's steady-state plateau instead of waiting the "
                      "full --settle-max.")
  r.add_argument("--settle-max", type=float, default=1.0, metavar="SEC",
                 help="--settle-tol: give up waiting after this long (default 1.0s).")
  r.add_argument("--servo-pick", action=argparse.BooleanOptionalAction, default=False,
                 help="with --skill-servo: grasp acquisitions (pick and handover) OWN their "
                      "terminal -- the empty gripper's "
                      "approach is funnel-paced by EE clearance (to the table and to loose "
                      "parts), and after the fire the arm rises until the fresh cargo is "
                      "contact-free instead of tracking through the pressing grasp keyframe "
                      "(the measured 48N/20ms fresh-weld break class).")
  r.add_argument("--compliant-follow", action=argparse.BooleanOptionalAction, default=False,
                 help="with --skill-servo: WHOLE-BRANCH admittance during all path following "
                      "-- contact force on arm links and the bare gripper (a pick approach "
                      "before any weld exists) yields the reference exactly like a cargo "
                      "press, with its own lower deadband --compliant-force. Compliant "
                      "following as the tracking law instead of held-cargo-only. EXPERIMENTAL: "
                      "an unresolved branch collision can saturate the 10mm yield and hold "
                      "the phase indefinitely; the stall diagnostic identifies this case.")
  r.add_argument("--compliant-force", type=float, default=3.0, metavar="N",
                 help="--compliant-follow yield deadband (N); the guard/funnel keep engaging "
                      "at --servo-force")
  r.add_argument("--skill-servo", action=argparse.BooleanOptionalAction, default=True,
                 help="--actuated: compliant skills -- the execution half of the funnel. "
                      "(a) held-part ADMITTANCE: above --servo-force the arm's reference "
                      "yields along the measured contact push (bounded --servo-max-dev, "
                      "decays when free), so the 27-82N table/graze presses cannot build -- "
                      "the trigger is unintended FORCE, so intended contacts still happen; "
                      "(b) TERMINAL SEAT SERVO: before each part-part weld, the child is "
                      "servoed to the mate datum via the carrying arm's Jacobian, so the "
                      "ideal-datum snap starts from mm instead of the tracking residual -- "
                      "each mate CONTRACTS error instead of inheriting it.")
  r.add_argument("--servo-force", type=float, default=6.0, metavar="N",
                 help="--skill-servo: admittance threshold (default 6N -- above any carry "
                      "load at --mass-scale 0.25, well below the 25N grasp cap).")
  r.add_argument("--servo-max-dev", type=float, default=0.010, metavar="M",
                 help="--skill-servo: task-space yield bound (default 10mm -- the tube the "
                      "reference may leave the certified path by).")
  r.add_argument("--servo-place-seat", action=argparse.BooleanOptionalAction, default=False,
                 help="--skill-servo: PLACES terminate by CONTACT -- task-space servo of the "
                      "held part (XY to the rest anchor, bounded descent) until sustained "
                      "touch, instead of tracking a joint reference whose z was authored "
                      "against the PLANNER's table (part_float below the real one; the "
                      "datum-band grind that ratchets long-lever grasps to their torque "
                      "cap). Fires the place at the touched pose. Default off.")
  r.add_argument("--servo-torque-guard", type=float, default=None, metavar="FRAC",
                 help="--skill-servo: normalized grasp-wrench guard (historic option name): "
                      "during external contact above --servo-force, engage guarded hold + "
                      "yield when either "
                      "measured FORCE or TORQUE exceeds FRAC of its corresponding scalar "
                      "breakaway cap (e.g. 0.6). Uses the same weld wrench and caps as the "
                      "breakage monitor, but ignores gravity-only utilization so transport "
                      "cannot deadlock. Default off (byte-identical).")
  r.add_argument("--no-servo-torque-guard", action="store_const", const=None,
                 dest="servo_torque_guard", default=argparse.SUPPRESS,
                 help="Disable the standard profile's normalized grasp-wrench guard.")
  r.add_argument("--servo-tol", type=float, default=0.002, metavar="M",
                 help="--skill-servo: seat-servo convergence tolerance (default 2mm).")
  r.add_argument("--corridor-online", action="store_true",
                 help="--corridor: execute ONLINE inside the Stage A certificate instead of "
                      "precomputing a Stage B/C schedule. Per-arm arc clocks advance greedily "
                      "each control tick under the exact offline rules (conservative envelope "
                      "lookups, no arm passes the pending barrier, laggard never capped), and "
                      "--skill-servo's slowdowns become PER-ARM RUNTIME rate factors: an arm "
                      "slows only while ITS cargo is low or pressed; everyone else flows. "
                      "Also removes the B/C scheduling phase (~75%% of a run's wall time) and "
                      "the spread pass (whose known hole exceeded wrench vmax 1.9x at "
                      "corners). No schedule-VERIFY: every visited state is inside the "
                      "certified set by construction.")
  r.add_argument("--corridor-cross-only", action=argparse.BooleanOptionalAction, default=True,
                 help="--corridor: VERIFY counts only CROSS-ARM pairs as scheduling "
                      "counterexamples. A same-arm overlap is invariant under desync/locks/ds "
                      "(it depends only on that arm's own arc, which every schedule traverses "
                      "identically), so refining against one is provably futile -- measured 4 "
                      "futile CEGAR passes = 11.5min of a 14.6min run on a 9.38mm self-overlap "
                      "the planner had legally accepted at --mp-tolerance 0.01. Same-arm "
                      "overlaps are still REPORTED once, as a base-path property.")
  r.add_argument("--corridor-verify", type=int, default=50, metavar="EVERY",
                 help="--corridor: after scheduling, re-check the RESULTING trajectory exactly "
                      "every EVERY samples. DEFAULT 50 (0 = off) -- verification costs ~0s "
                      "against 11-97s of certification, because it walks ONE 1-D trajectory "
                      "instead of a 2-D envelope. "
                      "every EVERY samples (0 = off). Stage A is a SAMPLED certificate -- at the "
                      "1.0 default its grid cells are ~641mm of EE travel apart on cross-new "
                      "while the failure mode is a 1.3mm overlap, so it can certify a "
                      "configuration it never evaluated. Verification walks ONE 1-D trajectory, "
                      "so it is cheap next to certification.")
  r.add_argument("--corridor-refine", type=int, default=4, metavar="N",
                 help="--corridor-verify: on a violation, halve --corridor-ds and re-certify, "
                      "up to N times (0 = report only). MEASURED on cross-new: ds 1.0 -> 0.25 "
                      "took stage A 21s -> 56s, left the corridor width and makespan unchanged "
                      "(+21.4%% -> +21.3%%), and removed the collision-driven grasp breaks.")
  r.add_argument("--corridor-rate-clamp", action=argparse.BooleanOptionalAction, default=True,
                 help="corridor: bound each step by the joint motion it ACTUALLY produces. "
                      "rate_of() budgets joint_rate using the arm's arc share in the segment it "
                      "is IN, but a step can land in a segment where that arm carries more of "
                      "the arc, so the executed joint speed overshoots (measured 1.98 rad/s at "
                      "joint_rate 1.0). Stiff arms absorb it; --actuated turns it into grasp "
                      "load. OFF = byte-identical.")
  r.add_argument("--mp-cograsp", action="store_true",
                 help="Before executing, report how far each pair of co-grasping tool "
                      "centres drifts from its keyframe relative pose along the path. Lets a "
                      "SAVED plan+path pair be audited without re-planning.")
  r.add_argument("--no-arm-meshes", action="store_true",
                 help="Skip the real link VISUAL meshes (default on) -- capsules only. "
                      "Faster to build; the collision model is the capsules either way.")
  r.add_argument("--joint-rate", type=float, default=None, metavar="RAD_S",
                 help="Override the per-arm joint-space cruise speed (default JOINT_RATE=%g). "
                      "Lower it when faster motion overloads a grasp (a heavy held part's "
                      "inertial wrench is NOT in the quasi-static LP)." % JOINT_RATE)
  r.add_argument("--corridor", action=argparse.BooleanOptionalAction, default=True,
                 help="CORRIDOR RETIMING. STAGE A first "
                      "MEASURES how far arms may desync before anything touches -- a certified "
                      "envelope, computed once with the same oracle -- then STAGES B+C plan "
                      "inside it with ZERO collision checks: an arm rests only at ITS OWN "
                      "events (never at a barrier it merely crosses) and runs at its own joint "
                      "limit. A pair whose corridor is 0 must move together, which is how "
                      "handovers/co-grasps express themselves. Prints the corridor widths.")
  r.add_argument("--actuated", action=argparse.BooleanOptionalAction, default=True,
                 help="DYNAMICS: drive the articulated arm with per-joint POSITION SERVOS "
                      "(MuJoCo <position> = a PD, computed in-solver -- no Python control loop "
                      "and no integral term to wind up) instead of writing qpos. forcerange "
                      "comes from rai's OWN ctrl_limits, i.e. the real UR10 effort limits "
                      "(330/150/54 Nm), so this is the first mode in which what the arm can "
                      "PUSH is modelled rather than assumed. Gravity+Coriolis are cancelled by "
                      "feedforward so the gains stay uncritical. Implies --articulated. The "
                      "trajectory becomes a REFERENCE: expect tracking error, and re-derive "
                      "the wrench/corner calibrations against it.")
  r.add_argument("--articulated", action="store_true",
                 help="NOTE: --actuated defaults ON (2026-07-29) and wins over this flag -- "
                      "pass --no-actuated --articulated for the qpos-driven stiff chains. "
                      "Drive REAL kinematic chains instead of teleported mocap bodies. Today "
                      "each arm is N DISCONNECTED mocap bodies, every one snapped to its rai-FK "
                      "world pose each step -- there are no joints between links at all. This "
                      "generates the actual chain from rai's own tree (arm_model) and drives it "
                      "by SETTING qpos: still perfectly stiff, so the grasp weld remains an "
                      "exact measure of the payload's m(a-g) and every wrench/corner calibration "
                      "stays interpretable. The step before actuators. Verify the model with "
                      "`python fk_compare.py --robot <type>` (agreement is machine-precision).")
  r.add_argument("--spline", action=argparse.BooleanOptionalAction, default=True,
                 help="With --corridor: replace the C0 path by a C1 Catmull-Rom curve through "
                      "the SAME waypoints, so there are no velocity discontinuities to crawl "
                      "through and curvature is finite everywhere (which is what --force-limits "
                      "needs). Task witnesses stay exact because Catmull-Rom interpolates every "
                      "waypoint; only the geometry BETWEEN waypoints moves, and each smoothed "
                      "span is certified with mr_bench's own check, pulled back toward the "
                      "straight segment wherever it would collide (so never worse than input).")
  r.add_argument("--spline-per-seg", type=int, default=8, metavar="N",
                 help="--spline: samples generated per original segment (default 8).")
  r.add_argument("--spline-tension", type=float, default=0.5, metavar="T",
                 help="--spline: Catmull-Rom tension; 0.5 = uniform, 0 = the original "
                      "polyline (default 0.5). Lower it for less deviation.")
  r.add_argument("--c2-spline", action=argparse.BooleanOptionalAction, default=False,
                 help="EXPERIMENTAL opt-in geometry postprocessor. Replace each free mode "
                      "stretch with a cubic C2 configuration-space B-spline whose task "
                      "witness endpoints remain exact. Ordinary MP waypoints are soft fit "
                      "targets, so the curve may shorten/cut corners. Every emitted edge is "
                      "densely checked for joint limits and collisions; an unsafe stretch "
                      "falls back whole to the certified input. Rigid co-hold and handover "
                      "stretches are left unchanged. When active this replaces --spline; "
                      "phase scheduling/smoothing are otherwise unchanged.")
  r.add_argument("--c2-samples-per-seg", type=int, default=8, metavar="N",
                 help="--c2-spline: emitted curve samples per input waypoint interval "
                      "(default 8). Certification also subdivides these at --c2-resolution.")
  r.add_argument("--c2-control-ratio", type=float, default=0.55, metavar="R",
                 help="--c2-spline: initial B-spline control-point count relative to input "
                      "waypoints (default 0.55, range (0,1]). Fewer controls smooth more.")
  r.add_argument("--c2-smooth-weight", type=float, default=0.03, metavar="W",
                 help="--c2-spline: squared control-polygon curvature weight (default 0.03).")
  r.add_argument("--c2-max-deviation", type=float, default=0.20, metavar="RAD",
                 help="--c2-spline: hard composite-joint deviation bound from the source "
                      "polyline at matched chord phase (default 0.20 rad).")
  r.add_argument("--c2-resolution", type=float, default=0.01, metavar="RAD",
                 help="--c2-spline: maximum per-joint step in the dense collision/joint-limit "
                      "certificate (default 0.01 rad).")
  r.add_argument("--weld-timeconst", type=float, default=0.0, metavar="S",
                 help="solref timeconst of the grasp/mate welds (default 0.004 = near-rigid). "
                      "Two arms co-grasping one part form a closed chain, so under --actuated "
                      "their tracking DIFFERENCE becomes force through the part (measured "
                      "119.7N vs a 25N cap). A real pinch is compliant: peak |F| = m|dv|/"
                      "(2*timeconst), so a larger value absorbs the mismatch -- at the cost of "
                      "the weld sagging under carry loads. 0 = keep the default.")
  r.add_argument("--limit-rate", action="store_true",
                 help="Time-rescale so the BINDING JOINT sits on its real velocity limit (rai "
                      "reports them: UR10 2.16 rad/s pan/lift, 3.15 elbow, 3.20 wrists) instead "
                      "of the nominal JOINT_RATE=1.0. Sibling of --arm-rate, which only "
                      "normalises the busiest ARM to that nominal. Time-only rescaling: the "
                      "configurations, and so the planner's certificate, are unchanged. Speed "
                      "raises tracking error and grasp wrenches -- --force-limits still caps "
                      "each segment.")
  r.add_argument("--corridor-margin", type=float, default=0.0, metavar="M",
                 help="Require M metres of CLEARANCE in the corridor, not just non-contact. The "
                      "desync envelope is certified on penetration alone, so an object carried "
                      "by a lagging arm may pass millimetres from another robot and still be "
                      "legal. Measured against the certified diagonal, so it only asks for room "
                      "the saved path itself had. 0 = today's behaviour.")
  r.add_argument("--planner-table", action="store_true",
                 help="Put the sim table where the SEARCH has it. build_mjcf otherwise sets the "
                      "top to the LOWEST PART'S BOTTOM = planner_top + --part-float (15mm by "
                      "default), so the arm meets the table 15mm early and presses on geometry "
                      "the plan never saw. Parts then start that high and settle onto it.")
  r.add_argument("--force-limits", action=argparse.BooleanOptionalAction, default=True,
                 help="With --corridor: bound each arm's speed AND acceleration so the "
                      "INERTIAL wrench on its grasp stays inside the wrench.PARAMS caps. The "
                      "planner's LP is quasi-static (gravity only), so fast execution can "
                      "overload a grasp it certified; this slows ONLY the loaded arm, only "
                      "where it is loaded, instead of the blunt global --joint-rate. "
                      "Point-mass model (m*a, m*a*lever) along the path PLUS the momentum step "
                      "at each C0 corner (see --no-corner-limits); arrivals shift later as "
                      "needed. Both benchmark scenes pass with no safety factor.")
  r.add_argument("--no-corner-limits", action="store_false", dest="corner_limits",
                 help="ABLATION: drop the corner term from --force-limits, leaving the pure "
                      "point-mass curvature/tangential bound. The corner term limits the arc "
                      "speed at the C0 corners of the EXECUTED path (every mode barrier is "
                      "one) so the velocity STEP a crossing arm takes there fits what its "
                      "grasp can absorb in KINK_TAU -- and that step, not any missing "
                      "conservatism, is what the curvature model cannot see: measured on "
                      "car-mini the model predicted 2.93 m/s^2 at the breaking corner while "
                      "the commanded motion did 621 m/s^2 for one 2ms step (a 122 deg turn at "
                      "0.81 m/s), 39.9N/2.0Nm through a 25N/0.4Nm grasp. With the term, both "
                      "scenes pass at --inertia-factor 1 (car-mini 17/17, cross8 39/39); "
                      "without it they fail (14/20 and 39/40).")
  r.add_argument("--inertia-factor", type=float, default=1.0, metavar="K",
                 help="Safety factor on the PREDICTED INERTIAL wrench for --force-limits "
                      "(default 1.0 = none needed). The point-mass estimate still omits the "
                      "angular term and follows the grasp anchor rather than the true COM "
                      "path, but those are small: the gap that once looked like '4x is not "
                      "enough' was the CORNER momentum step, now modelled (see "
                      "--no-corner-limits), and at factor 1 both scenes pass. Prefer this over "
                      "--mass-scale if you do want margin: it leaves the STATIC load true, so "
                      "the budget is not eaten before any motion (scaling mass cost 8-18x "
                      "makespan). NB factor 4 with the corner term is very slow (cross8 went "
                      "to -1.2%% makespan vs lockstep and still failed) -- the two compound.")
  r.add_argument("--mass-scale", type=float, default=1.0, metavar="F",
                 help="Scale every part mass (default 1.0), in BOTH the MuJoCo bodies and the "
                      "wrench estimate, so they stay consistent. A test knob: the grasp caps in "
                      "wrench.PARAMS are hand-set placeholders and the planner only checks a "
                      "held group UPRIGHT, so a rotated carry can exceed the torque cap on "
                      "gravity alone -- scale down to take that out of the picture while "
                      "validating everything else.")
  r.add_argument("--object-damping", type=float, default=0.0, metavar="D",
                 help="Simulation-only viscous damping on each free part joint (default 0). "
                      "Useful for long lockstep replays of very light loose parts, where "
                      "finite-precision table contacts can otherwise accumulate drift. "
                      "This does not alter the plan, motion path, collision geometry, or "
                      "grasp wrench limits.")
  r.add_argument("--wrench-margin", type=float, default=1.0, metavar="F",
                 help="Fraction of each grasp cap --force-limits may use (default 1.0 = the "
                      "full cap). Lower it (e.g. 0.7) for headroom against the unmodelled "
                      "angular term and tracking error.")
  r.add_argument("--corridor-ds", type=float, default=1.0, metavar="RAD",
                 help="--corridor stage A: arc spacing of the certification grid (default 1.0).")
  r.add_argument("--corridor-ds-lead", type=float, default=0.25, metavar="RAD",
                 help="--corridor stage A: lead step probed outward from the diagonal (0.25).")
  r.add_argument("--corridor-lead-max", type=float, default=4.0, metavar="RAD",
                 help="--corridor stage A: largest lead probed (default 4.0 rad).")
  r.add_argument("--arm-rate", action="store_true",
                 help="Apply JOINT_RATE per ARM instead of to the composite arc. Today an arm "
                      "moving alongside others advances at only |dq_arm|/|dq_composite| of the "
                      "nominal speed (measured 0.41-0.61 on car-mini), so arms crawl for a "
                      "purely parametric reason; this restores the intended semantics and never "
                      "exceeds it (bounded by sqrt(#arms)). On the shared clock it rescales TIME "
                      "only -- same configurations, so the planner's collision certificate still "
                      "holds with no extra checking.")
  r.add_argument("--coll-scale", type=float, default=1.0, metavar="F",
                 help="DEBUG: shrink every contact-enabled ARM collision primitive by F "
                      "(capsule/cylinder radius, sphere radius, box extents; parts NEVER "
                      "scaled). Try 0.95/0.9 to give desync room when the oracle wedges. "
                      "Applied before witnesses/problem/sim so IK, the oracle and the MuJoCo "
                      "capsules agree. Default 1.0 = untouched. Not for validation runs: "
                      "clearances no longer match the planner's certified geometry.")
  r.add_argument("--arm-arm-contacts", action=argparse.BooleanOptionalAction, default=None,
                 help="Use per-robot MuJoCo contact bits for real cross-arm collision physics "
                      "without enabling same-chain self contacts. Defaults ON with "
                      "--phase-schedule (physical "
                      "backstop for its exact audit) and OFF for legacy execution. Use "
                      "--no-arm-arm-contacts only as a diagnostic override.")
  r.add_argument("--path-preflight", action=argparse.BooleanOptionalAction, default=True,
                 help="Before spline/corridor construction, densely check the SAVED diagonal "
                      "path against strict collision and joint limits. Default ON: retiming "
                      "cannot repair a bad base path, so fail immediately with the exact "
                      "mode/edge/pair instead of spending minutes refining its corridor.")
  r.add_argument("--path-preflight-tolerance", type=float, default=None, metavar="M",
                 help=f"Strict base-path penetration tolerance (default {PATH_PREFLIGHT_TOL:g}m, "
                      "the production MP contract and above RAI's micrometre numerical noise; "
                      "independent of any relaxed --mp-collision-tolerance used to produce "
                      "the path).")
  r.add_argument("--path-preflight-resolution", type=float, default=None, metavar="RAD",
                 help="Maximum per-joint interpolation step in the base-path audit (default: "
                      "the planner collision resolution, normally 0.01rad).")
  r.add_argument("--refresh-witnesses", action="store_true",
                 help="Explicitly IGNORE an embedded final MP export and run the legacy "
                      "witness/pre-keyframe rebuild. Required to open an old --mp-path that "
                      "predates final-export persistence; normal replay never resolves IK.")
  r.add_argument("--record", metavar="FILE", default=None, help="save a replayable .npz")
  r.add_argument("--animate-grippers", action=argparse.BooleanOptionalAction, default=False,
                 help="Animate two-finger hands around pick/handover/release events using "
                      "simulation-only articulated hinge joints and position actuators. "
                      "Planning joint vectors remain unchanged; vacuum hands are a no-op. "
                      "The joints live in qpos, so recordings replay without a side channel.")
  r.add_argument("--gripper-motion-time", type=float, default=0.25, metavar="SEC",
                 help="Smooth close/open duration for --animate-grippers (default 0.25s).")
  r.add_argument("--report-json", metavar="FILE", default=None,
                 help="write the structured per-action SimReport and resolved perturbation "
                      "metadata as JSON (also written before --require-success exits 3).")
  r.add_argument("--gui", action="store_true", help="live MuJoCo viewer during execution")
  # (--prioritized and --smooth are REMOVED: the corridor is the certified successor of the
  # async retiming experiment -- same measured physics, none of the oracle gaps -- and the
  # executor now has exactly one drive loop. Preprocessing picks the trajectory (--corridor or
  # lockstep); the drive mode (--articulated/--actuated/--skill-servo) is an independent axis.)
  r.add_argument("--verbose", action="store_true")

  v = sub.add_parser("replay", help="view a recording (kinematic playback)")
  v.add_argument("path")
  v.add_argument("speed", nargs="?", type=float, default=1.0)
  v.add_argument("--no-loop", action="store_true")
  return p


def parse_args(argv=None):
  """Parse sim CLI with explicit flag > named profile > built-in default precedence."""
  parser = _build_cli()
  args = parser.parse_args(argv)
  if getattr(args, "cmd", None) == "run" and getattr(args, "sim_profile", None):
    # Profile fields belong to the run subparser; root defaults would be overwritten by that
    # subparser's action defaults. Layer them there, then let the raw CLI parse last.
    sub = next(a for a in parser._actions if getattr(a, "choices", None)
               and "run" in a.choices)
    sub.choices["run"].set_defaults(**_SIM_PROFILES[args.sim_profile])
    args = parser.parse_args(argv)
  return args


if __name__ == "__main__":
  args = parse_args()
  if args.cmd == "run":
    try:
      _cli_run(args)
    except PathPreflightError as e:
      import sys
      print(str(e), file=sys.stderr)
      raise SystemExit(2)
  else:
    replay_recording(args.path, speed=args.speed, loop=not args.no_loop)
