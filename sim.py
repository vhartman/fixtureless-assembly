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
  * KINEMATIC ROBOTS / DYNAMIC PARTS. Each gripper is a MOCAP body dragged along the planned
    EE trajectory (weld from mocap to a small gripper body); parts are free bodies. The LP's
    claims are about PART statics under imposed contacts/loads -- the robot is a boundary
    condition. v2 (real arms): UR10 MJCF + joint position actuators along mr_bench
    trajectories (TODO).
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
    v2 upgrade if reviewers want contact-level grasp failure.
  * FRICTION from the planner's mu; ssBox rounding dropped (plain box; contact patch grows by
    ~r -- conservative for stability, note in the paper). Fixed timestep, default implicit
    solver, seeded => deterministic (the PYTHONHASHSEED discipline, for physics).

TRAJECTORY SOURCES (transfer side):
  v1 (implemented): STATE KEYFRAMES. Plans carry `.states` (per-step State.poses); the EE pose
     for a held part is part_pose (+) grasp_pose(assembly, part, gid). Quasi-static mocap
     interpolation + a GUARDED INSERT for assemble (funnel-mouth approach along the mate axis,
     slow contact segment; the press force emerges from dragging the mocap into contact).
  v2 (TODO): mr_bench BiRRT joint trajectories (real arms, depgraph order) -- full-pipeline
     validation; map mode-transition attach events onto the same weld managers.

USAGE (intended):
    from sim import SimExecutor
    ex = SimExecutor(assembly, gripper_kinds={"a1": "vacuum"})
    report = ex.execute(plan, states=plan.states)
    print(report.summary())

STATUS (2026-07-24, mujoco 3.10.0): v1 VERIFIED end-to-end. cross-3/2R vacuum: 8/8, 5.8mm.
DISCRIMINATOR cross-5 (side mates, 2R vacuum): wrench-ON plan 20/20 SUCCEEDED (4.0mm/0.3deg);
wrench-OFF plan FAILED physically (release -> assembly topples, 800mm/90deg) -- the paper's
core validation result in miniature. All executor bugs en route were found by trace, never
by guess: symbolic State.poses (-> _state_part_poses everywhere), carry lift, WORLD-frame
mate axis after rotated places, slerp rotation-rate cap, vertical DEPART before reorienting,
and datum welds (weld at the manual's exact relpose after the seat check -- funnel
semantics; welding at the ACHIEVED pose accumulated 9.7deg over 3 mates). Remaining stubs:
real finger-contact two-finger (breakaway-weld stand-in for now); v2 mr_bench joint
trajectories; start-perturbation trials; --sim-validate flag in main.py.
Recordings: sim_recordings/*.npz -- replay with `python3 sim.py replay <file> [speed]`.

V2 STATUS (2026-07-24, later same day): execute_motion_plan VERIFIED end-to-end WITH the
zero-tolerance chain -- --strict-collisions witnesses (0 penetration) -> make_problem
collision_tolerance=1e-6 -> BiRRT -> arms-in-sim execution: cross-3/2R ur10 vacuum 9/9
SUCCEEDED, 2.0mm/1.2deg final error (recording: sim_recordings/mp_cross3_strict.npz). The
earlier 27.8N carry graze was the planner's tolerated 10mm penetration -- gone at 1e-6.
fire() release events re-attach CONTINUING HOLDERS (hold-through groups; mirrors mp_export's
release fix). v1 discriminator re-verified after the weld-semantics change (ON 20/20;
OFF fails physically, mode topple -> dropped-at-release).
"""
import os
import time as _time
import numpy as np
from math import sqrt as _sqrt
import arm_model

import retiming                                              # portable path->trajectory retiming

try:
  import mujoco
  HAVE_MUJOCO = True
except ImportError:                                        # module must import without the sim
  mujoco = None
  HAVE_MUJOCO = False

from problem import (grasp_pose, _pose_compose, _quat_to_R, _shape_half_z,
                     _state_part_poses)


# ---------------------------------------------------------------------------------------------
# tunables (every one is a MODELING decision -> report in the paper)
# ---------------------------------------------------------------------------------------------
DT = 0.002                      # s; MuJoCo default; parts are >=3cm primitives, stable
SEAT_POS_TOL = 0.010            # m   : seated when relative pose within these of the manual
                                # (= the pre-insertion funnel's CAPTURE tolerance; the datum
                                # weld then enforces the exact manual pose. Residual carry
                                # sag of the grasp weld is ~2-3mm after stiffening.)
SEAT_ANG_TOL = np.deg2rad(7.0)  # rad
DROP_Z = 0.05                   # m   : below expected by this much => "dropped"
SLIDE_TOL = 0.015               # m   : xy drift of a NON-actor part during an action => "slid"
TOPPLE_ANG = np.deg2rad(20.0)   # rad : orientation change of a resting group => "toppled"
EE_SPEED = 0.25                 # m/s : keyframe interpolation speed (quasi-static regime)
INSERT_SPEED = 0.05             # m/s : contact-rich seating segment
CARRY_LIFT = 0.12               # m   : held parts are carried this high above their pick spot.
                                # Planner State.poses do NOT model the lift ("held" is
                                # symbolic); without it a mate onto a HELD target jams into
                                # the table (measured: 21.8N > 15N cap on cross-3 rollout 2).
RECORD_EVERY = 8                # frames: sample qpos+grasp forces every N sim steps (~62fps)
BREAK_SUSTAIN = 0.02            # s : a grasp breaks only if overloaded CONTINUOUSLY this long.
                                # Instantaneous 2ms contact-impulse spikes (measured 15.6N on
                                # a carried group, invisible at recording resolution) are
                                # survivable by a real seal/pinch; loss of grip is a leak/slip
                                # process, not a single-sample event. Static overloads (what
                                # the LP prices) persist and still break.
WELD_TIMECONST = 0.004          # s   : solref timeconst of the grasp/mate welds (stiff: with
                                # MuJoCo's soft default the mocap-anchored weld SAGS mm-scale
                                # under carry loads -- measured 7-8mm, enough to miss a seat).
KINK_TAU = 2.0 * WELD_TIMECONST # s   : time the grasp interface takes to absorb a VELOCITY
                                # STEP (a C0 path corner), i.e. peak |F| = m|dv|/KINK_TAU.
                                # MEASURED on car-mini: a 1.24 m/s step on a 0.30kg group
                                # peaked at 39.9N => 9.3ms, ~2 solref timeconsts. On hardware
                                # the corner-rounding time of the tracking controller plays
                                # this role; here it is the weld, which IS the interface.
CAPTURE_RANGE = 0.05            # m   : a part-part mate datum-welds only within this of the
                                # seat (the funnel mouth); a larger miss = the part never
                                # arrived -> fail cleanly instead of snapping it in (explodes).
RAMP_DIST = 0.2                 # rad : v2 trapezoid -- ramp to/from ~rest over this much joint
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
  {dropped, slid, toppled, grasp_broke, not_seated}."""
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


# ---------------------------------------------------------------------------------------------
# MJCF construction: assembly spec + a State's poses -> model XML
# ---------------------------------------------------------------------------------------------
def _geom_xml(spec, mu):
  kind, p = spec[0], list(spec[1])
  fr = f'friction="{mu} 0.005 0.0001"'
  if kind == "box":
    return f'<geom type="box" size="{p[0]/2} {p[1]/2} {p[2]/2}" {fr} mass_placeholder/>'
  if kind == "cylinder":
    return f'<geom type="cylinder" size="{p[1]} {p[0]/2}" {fr} mass_placeholder/>'
  if kind == "sphere":
    return f'<geom type="sphere" size="{p[0]}" {fr} mass_placeholder/>'
  raise ValueError(f"unknown shape kind {kind}")


def _save_recording(path, scene, rec, report):
  import json
  np.savez_compressed(
      path, xml=scene.xml, qpos=np.asarray(rec["qpos"]),
      mpos=np.asarray(rec["mpos"]), mquat=np.asarray(rec["mquat"]),
      forces=np.asarray(rec["forces"]), robots=np.array(scene.robots),
      parts=np.array(scene.assembly.objects), marks=json.dumps(rec["marks"]),
      results=json.dumps([repr(r) for r in report.results]))
  print(f"[sim] recording ({len(rec['qpos'])} frames) -> {path}")


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
            for si, (obj, rgba) in enumerate(_link_meshes(C, n)):
              out.append((f"{n}__m{si}", n, ("mesh", obj), False, False, GROUP_VISUAL, rgba))
          except Exception:
            pass                                           # degenerate / no mesh: skip
      elif k in ("capsule", "cylinder", "sphere", "box", "ssBox"):
        sz = list(np.asarray(info.get("size"), dtype=float))
        if "_coll" in n:
          # rai contact=-2 (WRIST cluster nearest the tool) is parent-chain exempt vs the
          # ATTACHED part (a held cube hangs 35mm from the tip, inside the wrist capsule).
          out.append((n, n, (k, sz), True, info.get("contact") == -2, GROUP_COLL, tint))
        else:
          # gripper primitives (vacuum tip, gripper_fill): visual-only, never collide.
          out.append((n, n, (k, sz), False, False, GROUP_VISUAL, tint))
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


def _default_geomgroups(viewer, has_mesh):
  """Initial viewer group visibility. MuJoCo HIDES groups >=3 by default, so we set them
  explicitly: when real link meshes exist, SHOW them (group VISUAL) and HIDE the collision
  capsules (group COLL) so the robot reads cleanly. Toggle with number keys (COLL / VISUAL)."""
  if viewer is None:
    return
  if has_mesh:
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


def planner_table_top(C):
  """Top z of the table the SEARCH reasoned about, read from the rai scene rather than
  hard-coded: frame z + half thickness. build_mjcf otherwise derives the top from the lowest
  PART bottom, which sits part_float above this."""
  try:
    f = C.getFrame("table")
    return float(np.asarray(f.getPosition(), dtype=float)[2]) + float(f.getSize()[2]) / 2.0
  except Exception:
    return None


def build_mjcf(assembly, state, robots, mu_table=0.5, arms=None, arm_poses=None,
               ee_anchor=None, articulated_C=None, actuated=False, planner_table_top=None,
               weld_timeconst=None):
  """The whole validation world as one MJCF string:
  - table (static), parts (free bodies at state.poses, shape quat baked into the geom),
  - robots: EITHER one mocap-driven gripper puck per robot (v1 keyframe mode) OR, when
    `arms` is given (v2 motion-plan mode), one mocap body per ARM SHAPE FRAME
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
  parts_xml, eq_xml = [], []
  # deterministic per-part colors (tab10-ish) -- rai's colors are np.random.rand, not
  # reproducible here; a fixed palette keeps replays comparable across runs.
  palette = ["0.84 0.15 0.16 1", "0.12 0.47 0.71 1", "0.17 0.63 0.17 1", "1.0 0.5 0.05 1",
             "0.58 0.4 0.74 1", "0.55 0.34 0.29 1", "0.89 0.47 0.76 1", "0.74 0.74 0.13 1"]
  for i, p in enumerate(assembly.objects):
    spec = assembly.part_shape(p)
    pose = np.asarray(state.poses[p], dtype=float)
    q = pose[3:]
    sq = spec[2] if len(spec) > 2 else None
    gq = ' quat="1 0 0 0"' if sq is None else f' quat="{sq[0]} {sq[1]} {sq[2]} {sq[3]}"'
    mass = assembly.masses.get(p, 0.1)
    g = _geom_xml(spec, mu_table).replace(
        "mass_placeholder",
        f'mass="{mass}" group="{GROUP_PART}" rgba="{palette[i % len(palette)]}"{gq}')
    parts_xml.append(
        f'<body name="{p}" pos="{pose[0]} {pose[1]} {pose[2]}" '
        f'quat="{q[0]} {q[1]} {q[2]} {q[3]}"><freejoint/>{g}</body>')
  ee_colors = ["0.2 0.2 0.9 1", "0.9 0.2 0.2 1", "0.2 0.8 0.8 1", "0.9 0.7 0.1 1",
               "0.6 0.2 0.8 1", "0.4 0.9 0.3 1"]
  arm_mesh_assets = ""
  if arms is None:                                         # v1: one gripper puck per robot
    for j, r in enumerate(robots):
      parts_xml.append(
          f'<body name="ee_{r}" mocap="true" pos="0 0 1"><geom type="cylinder" '
          f'size="0.02 0.02" mass="0.2" contype="0" conaffinity="0" group="{GROUP_VISUAL}" '
          f'rgba="{ee_colors[j % len(ee_colors)]}"/></body>')
    anchor = {r: f"ee_{r}" for r in robots}
  else:                                                    # v2: FK-synced arm link bodies
    mesh_assets = []
    for body, fkf, geom_spec, collide, _ex, group, rgba in arms:
      pose = arm_poses[fkf]
      q = pose[3:]
      # contype 2 / conaffinity 1: arm links collide with PARTS and the table, never with
      # each other (self/arm-arm clearance was the MOTION PLANNER's job; re-checking it
      # here on mocap bodies would only produce huge phantom forces on interpenetration).
      con = 'contype="2" conaffinity="1"' if collide else 'contype="0" conaffinity="0"'
      if geom_spec[0] == "mesh":                           # real link mesh (visual only)
        mesh_assets.append(f'<mesh name="m_{body}" file="{geom_spec[1]}"/>')
        geom = f'type="mesh" mesh="m_{body}"'
      else:
        geom = _arm_geom_xml(geom_spec)
      parts_xml.append(
          f'<body name="arm_{body}" mocap="true" pos="{pose[0]} {pose[1]} {pose[2]}" '
          f'quat="{q[0]} {q[1]} {q[2]} {q[3]}">'
          f'<geom {geom} mass="1.0" {con} group="{group}" rgba="{rgba}"/></body>')
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
    body_xml, articulated_joints = arm_model.build_arms_mjcf(
        articulated_C, robots, collide_fn=arm_model.collides_like_sim,
        group=GROUP_COLL, contype="2", conaffinity="1",
        meshes=_arm_meshes, assets=_arm_assets, mesh_group=GROUP_VISUAL,
        # links must SWAMP the payload: the chain is integrated by mj_step even though we
        # drive it kinematically, so a light link welded to a part lets the solver throw it.
        armature=ARTICULATED_ARMATURE, limits=bool(actuated),
        link_mass=(ACTUATED_LINK_MASS if actuated else ARTICULATED_LINK_MASS),
        link_inertia=((ACTUATED_LINK_MASS if actuated else ARTICULATED_LINK_MASS) / 10.0),
        # only DOF-carrying bodies get real mass when actuated: rai has ~69 frames per arm, so
        # massing all of them built a 276kg arm and the servos diverged (3.65 -> 0.020 rad).
        static_mass=(1e-4 if actuated else None))
    parts_xml = [x for x in parts_xml if 'name="arm_' not in x]      # drop the mocap mirrors
    parts_xml.append(body_xml)
    anchor = {r: ee_anchor[r] for r in robots}
    arm_mesh_assets = "".join(_arm_assets)
  act_xml = ""
  if articulated_C is not None and actuated:
    a, _ = arm_model.actuator_mjcf(articulated_C, articulated_joints, kp=ACTUATED_KP)
    act_xml = f"\n  <actuator>\n{a}\n  </actuator>"
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
  # the table box reaches from the FLOOR (z=0) to table_top: collision-identical for the
  # parts (only the top face matters) but it looks like a table instead of a floating slab.
  th = table_top / 2.0
  return f"""
<mujoco model="assembly_validation">
  <!-- MuJoCo's compiler defaults to angle="degree", so rai's joint ranges (radians) were being
       read as degrees and shrunk 57x: shoulder_lift [-3.28, 0.283] became [-0.057, 0.005].
       Every joint then sat ~2 rad outside its "limit" and MuJoCo answered with 23550 Nm, which
       is what threw the actuated arm (32 rad/s) and what zeroed the grasp weld's force in the
       kinematic mode. Body poses are quaternions, so only ranges were affected. -->
  <compiler angle="radian"/>
  <option timestep="{DT}" integrator="implicitfast"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.25 0.3"
             rgb2="0.25 0.3 0.35" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.7 0.9" rgb2="0.9 0.9 0.95"
             width="64" height="64"/>
    {arm_mesh_assets}
  </asset>
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/></visual>
  <worldbody>
    <light pos="0.5 -0.5 2.5" dir="-0.2 0.2 -1" directional="true" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid" pos="0 0 0"
          group="{GROUP_ENV}"/>
    <geom name="table" type="box" size="{TABLE_HALF[0]} {TABLE_HALF[1]} {th}"
          pos="0 0 {th}" rgba="0.55 0.42 0.28 1" friction="{mu_table} 0.005 0.0001"
          group="{GROUP_ENV}"/>
    {''.join(parts_xml)}
  </worldbody>
  <equality>{''.join(eq_xml)}</equality>{act_xml}
</mujoco>"""


# ---------------------------------------------------------------------------------------------
# the sim world: weld managers + stepping + breakaway
# ---------------------------------------------------------------------------------------------
class SimScene:
  def __init__(self, assembly, state, robots, gripper_kinds=None, gui=False,
               arms=None, arm_poses=None, ee_anchor=None, articulated_C=None,
               actuated=False, planner_table_top=None, weld_timeconst=None):
    if not HAVE_MUJOCO:
      raise RuntimeError("mujoco not installed (pip install mujoco)")
    self.assembly = assembly
    self.xml = build_mjcf(assembly, state, robots, arms=arms, arm_poses=arm_poses,
                          planner_table_top=planner_table_top, weld_timeconst=weld_timeconst,
                          ee_anchor=ee_anchor, articulated_C=articulated_C,
                          actuated=actuated)
    self.model = mujoco.MjModel.from_xml_string(self.xml)
    self.data = mujoco.MjData(self.model)
    self.robots = list(robots)
    self.arms = arms                                       # v2: [(frame, kind, size)]
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
    self._ctrl_ff = np.zeros(0, dtype=float)
    if self.articulated:
      for jn in articulated_C.getJointNames():
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
          self._qadr[jn] = int(self.model.jnt_qposadr[jid])
          self._vadr[jn] = int(self.model.jnt_dofadr[jid])
      self._rai_joints = list(articulated_C.getJointNames())
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
    self.gripper_kinds = gripper_kinds or {}
    self.active_grasps = {}                                # robot -> (eq_id, part, fcap, tcap)
    self._hold_info = {}                                   # robot -> grasp-frame info (directional caps)
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
                          arms is not None and any(e[2][0] == "mesh" for e in arms))               # global `mujoco` in this scope
    mujoco.mj_forward(self.model, self.data)
    self._phys += 1

  # -- naming helpers --------------------------------------------------------------------
  def _eq_id(self, name):
    return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

  def _body_id(self, name):
    return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

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

  def grasp(self, robot, part, gid=None):
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
    _g7 = np.asarray(_P.grasp_pose(self.assembly, part, gid), dtype=float)   # flat [pos, quat]
    _gp, _gq = _g7[:3], _g7[3:]
    _R = np.zeros(9)
    mujoco.mju_quat2Mat(_R, _gq)
    self._hold_info[robot] = {"eq": eq, "part": part, "gid": gid, "model": _model,
                              "caps": _caps, "gp": _gp,
                              "Rg": _R.reshape(3, 3)}
    for gid in self._holdex.get(robot, ()):                # wrist cluster exempt while holding
      self.model.geom_contype[gid] = 0
      self.model.geom_conaffinity[gid] = 0

  def tracking_error(self, q):
    """max |q_actual - q_commanded| over the arm dofs, rad. With actuators the trajectory is a
    REFERENCE, so this is the number that decides whether a mate still lands on its witness --
    SEAT_POS_TOL is 10mm, so even ~0.01 rad at a 1m lever is already marginal."""
    q = np.asarray(q, dtype=float)
    if self._qadr_i.size == 0:
      return 0.0
    return float(np.abs(self.data.qpos[self._qadr_i] - q[self._qadr_j]).max())

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
    """v2: set every arm link mocap body to its rai-FK world pose. poses = {frame: 7-pose}.

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
    self.model.eq_data[eq][6:10] = [1.0, 0.0, 0.0, 0.0]    # identity relative rotation
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
      rel = self.assembly.relative_pose(child)
      eq = self._eq_id(f"mate_{child}")
      self.model.eq_data[eq][:3] = 0.0
      self.model.eq_data[eq][3:6] = np.asarray(rel[1], dtype=float)
      self.model.eq_data[eq][6:10] = [1.0, 0.0, 0.0, 0.0]
      self.model.eq_data[eq][10] = 1.0
      self.data.eq_active[eq] = 1
    else:                                                  # gripper GRASP (frozen relpose)
      # recover the GRASP ID so the breakaway can use the planner's directional caps: the
      # frame-based replay only carries [parent, child], but the discrete state knows which
      # grasp of `child` is held. Falls back to the conservative scalar if unavailable.
      _gid = None
      if self.expect_state is not None:
        try:
          _gid = _held_grasp_id(self.expect_state, child)
        except ValueError:
          _gid = None
      self.grasp(parent_frame.split("_", 1)[0], child, gid=_gid)

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
        if not (b1.startswith(r) or b2.startswith(r) or part in (b1, b2)):
          continue
        f = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, i, f)
        k = tuple(sorted((b1, b2)))
        tot[k] = tot.get(k, 0.0) + float(np.linalg.norm(f[:3]))
        names[k] = f"{c.dist * 1000:+.2f}mm"
      hot = sorted(tot.items(), key=lambda kv: -kv[1])[:3]
      if not hot:
        return "  [no contacts on the gripper or the part -- the load is NOT a reaction]"
      return "  contacts: " + ", ".join(
          f"{a}~{b} {v:.1f}N @{names[(a, b)]}" for (a, b), v in hot)
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
          # the replay shows frozen robots (v1 recordings had exactly that bug).
          self.recorder["mpos"].append(self.data.mocap_pos.copy())
          self.recorder["mquat"].append(self.data.mocap_quat.copy())
          self.recorder["forces"].append(fr)
      if self.viewer is not None:
        self.viewer.sync()

  def drive_mocap(self, robot, pose_wxyz):
    """Set the mocap target (position-level kinematic driving; MuJoCo pulls the welded
    gripper body along through the constraint solver)."""
    mid = self.model.body_mocapid[self._body_id(f"ee_{robot}")]
    self.data.mocap_pos[mid] = pose_wxyz[:3]
    self.data.mocap_quat[mid] = pose_wxyz[3:]


# ---------------------------------------------------------------------------------------------
# the executor: plan + per-step states -> rollout -> SimReport
# ---------------------------------------------------------------------------------------------
class SimExecutor:
  """v1 keyframe executor. EE targets from the plan's state poses (held part pose (+)
  grasp_pose()); quasi-static mocap interpolation; guarded insert for assemble; settle and
  judge after every action (failure taxonomy vs the planner's expected next state)."""

  def __init__(self, assembly, gripper_kinds=None, gui=False):
    self.assembly = assembly
    self.gripper_kinds = gripper_kinds or {}
    self.gui = gui
    self.scene = None

  # DISCOVERED ON FIRST ROLLOUTS: planner State.poses are SYMBOLIC payloads -- they update on
  # neither assemble nor place (mated geometry lives in the mate tree, placement faces in
  # state.place). The canonical state -> world-pose synthesis is problem._state_part_poses
  # (the same function the cheap checker and the wrench LP judge geometry with): mates
  # chained, placement faces + rest heights applied, held groups lifted. The executor uses IT
  # for every EE target and every judgement, with lift = CARRY_LIFT so expected and simulated
  # carry heights agree.
  def _expected_pose(self, state, part):
    return np.asarray(_state_part_poses(state, self.assembly,
                                        lift=CARRY_LIFT)[part], dtype=float)

  def _seated_pose_sim(self, part):
    """Seat target from the mate PARENT's actual sim pose (mates are translation-only in
    the part frame; orientation = parent's) -- tracks any slide of the target group."""
    rel = self.assembly.relative_pose(part)
    return _pose_compose(self.scene.part_pose(rel[0]),
                         np.r_[np.asarray(rel[1], dtype=float), 1.0, 0.0, 0.0, 0.0])

  def _ee_pose_for(self, state, part, gid):
    return _pose_compose(self._expected_pose(state, part),
                         grasp_pose(self.assembly, part, gid))

  def _reattach_holders(self, s_post, part):
    """CONTINUING HOLDERS after a release: weld_mate removed the inserter's grasp weld, but if
    the post-state still has another arm HOLDING this group (hold-through -- the milestone/
    wrench plans lean on it heavily), that arm's grasp weld was also removed at its own
    assemble and must be re-activated, else the group is orphaned and drops. Same fix as the
    v2 executor's fire(); the arm's puck is parked at its grasp, so the frozen relpose is
    correct. (Found by trace on a main-solved cross-5 plan: release a1/obj_0 with the airborne
    {obj_0,obj_1,obj_2} held-through by a2/obj_2 -> obj_0 dropped.)"""
    group = next((g for g in s_post.subassemblies() if part in g), frozenset())
    for p in group:
      for c in s_post.supports.get(p, ()):
        if c[0] == "grasp" and c[1] not in self.scene.active_grasps:
          self.scene.grasp(c[1], c[2], gid=c[3])

  def execute(self, plan, states, record=None):
    """record=<path.npz>: dump a REPLAYABLE recording (model XML + qpos frames + per-robot
    grasp wrenches + action marks). View later with sim.replay_recording(path) or
    `python3 sim.py replay path.npz`."""
    assert states is not None and len(states) == len(plan) + 1, "need per-step states"
    robots = sorted({a[1] for a in plan})
    self.scene = SimScene(self.assembly, states[0], robots,
                          gripper_kinds=self.gripper_kinds, gui=self.gui)
    rec = None
    if record is not None:
      rec = self.scene.recorder = {"qpos": [], "mpos": [], "mquat": [], "forces": [],
                                   "marks": []}
    report = SimReport()
    for i, act in enumerate(plan):
      s_pre, s_post = states[i], states[i + 1]
      kind, robot = act[0], act[1]
      if rec is not None:
        rec["marks"].append((len(rec["qpos"]), repr(tuple(act))))
      pre_n = len(report.results)                          # breakaways appended during steps
      if kind == "pick":
        self._move(robot, self._ee_pose_for(s_pre, act[2], act[3]), report, act)
        self.scene.grasp(robot, act[2], gid=act[3])
        # DEPART: straight up at the CURRENT orientation first, THEN move/reorient to the
        # carry pose. Slerping while still at table level dragged a rotated resting group's
        # corners through the table (measured 25.1N > 25N cap re-picking the face-3 stack).
        mid = self.scene.model.body_mocapid[self.scene._body_id(f"ee_{robot}")]
        up = np.r_[np.asarray(self.scene.data.mocap_pos[mid]) + [0, 0, CARRY_LIFT],
                   np.asarray(self.scene.data.mocap_quat[mid])]
        self._move(robot, up, report, act)
        # s_post has the group held -> _state_part_poses lifts it by CARRY_LIFT
        self._move(robot, self._ee_pose_for(s_post, act[2], act[3]), report, act)
      elif kind == "assemble":
        gid = _held_grasp_id(s_pre, act[2])
        target = _pose_compose(self._seated_pose_sim(act[2]),      # seat = parent's SIM pose
                               grasp_pose(self.assembly, act[2], gid))
        # mate axis is in the PART/ASSEMBLY frame -> rotate into WORLD by the parent's actual
        # sim orientation (bug found by trace: after a face-rotated place the world approach
        # direction changes; pressing along the unrotated axis knocked the group over).
        rel = self.assembly.relative_pose(act[2])
        Rp = _quat_to_R(self.scene.part_pose(rel[0])[3:])
        d = Rp @ np.asarray(self.assembly.mate_axis(act[2]), dtype=float)
        mouth = np.r_[target[:3] - 0.03 * d, target[3:]]
        self._move(robot, mouth, report, act)
        self._move(robot, target, report, act, speed=INSERT_SPEED)   # contact-rich press
        if self._seated(act[2]):
          self.scene.weld_mate(act[2])
      elif kind == "place":
        gid = _held_grasp_id(s_pre, act[2])
        self._move(robot, self._ee_pose_for(s_post, act[2], gid), report, act)
        self.scene.release(robot)
        self._reattach_holders(s_post, act[2])
      elif kind == "release":
        self.scene.release(robot)
        self._reattach_holders(s_post, act[2])
      elif kind in ("handover", "add_grasp"):
        gid = act[3]
        self._move(robot, self._ee_pose_for(s_pre, act[2], gid), report, act)
        self.scene.grasp(robot, act[2], gid=act[3])
        if kind == "handover":                             # giver = whoever held it before
          giver = _holder_of(s_pre, act[2], exclude=robot)
          if giver:
            self.scene.release(giver)
      self.scene.step(int(SETTLE_T / DT), report, act)
      report.results.append(self._judge(act, s_post))
      if any(not r.ok for r in report.results[pre_n:]):
        break                                              # open-loop: first failure ends it
    report.final_pose_err = {}
    for p in self.assembly.objects:
      exp = self._expected_pose(states[-1], p)
      report.final_pose_err[p] = (
          float(np.linalg.norm(self.scene.part_pose(p)[:3] - exp[:3])),
          _quat_angle(self.scene.part_pose(p)[3:], exp[3:]))
    if rec is not None:
      _save_recording(record, self.scene, rec, report)
    return report

  # -- primitives ------------------------------------------------------------------------
  def _move(self, robot, target, report, act, speed=EE_SPEED):
    mid = self.scene.model.body_mocapid[self.scene._body_id(f"ee_{robot}")]
    cur = np.asarray(self.scene.data.mocap_pos[mid]).copy()
    cur_q = np.asarray(self.scene.data.mocap_quat[mid]).copy()
    dist = float(np.linalg.norm(np.asarray(target[:3]) - cur))
    ang = _quat_angle(cur_q, np.asarray(target[3:]))
    # step count covers BOTH translation and rotation (a face-rotated place turns the held
    # group 90deg -- snapping the quat rips the weld; slerp keeps it quasi-static).
    steps = max(1, int(dist / (speed * DT)), int(ang / (1.0 * DT)))   # <=1 rad/s rotation
    for t in range(steps):
      a = (t + 1) / steps
      pose = np.r_[(1 - a) * cur + a * np.asarray(target[:3]),
                   _slerp(cur_q, np.asarray(target[3:]), a)]
      self.scene.drive_mocap(robot, pose)
      self.scene.step(1, report, act)

  def _seated(self, part):
    rel = self.assembly.relative_pose(part)
    got = _rel_pos(self.scene.part_pose(rel[0]), self.scene.part_pose(part))
    return float(np.linalg.norm(got - np.asarray(rel[1]))) <= SEAT_POS_TOL

  def _judge(self, action, expect_state):
    a = self.assembly
    for p in a.objects:
      exp = self._expected_pose(expect_state, p)
      got = self.scene.part_pose(p)
      if got[2] < exp[2] - DROP_Z:
        return ActionResult(action, False, "dropped", p)
      if p != action[2]:
        if float(np.linalg.norm(got[:2] - exp[:2])) > SLIDE_TOL:
          return ActionResult(action, False, "slid",
                              f"{p} {float(np.linalg.norm(got[:2]-exp[:2]))*1000:.0f}mm")
        if _quat_angle(got[3:], exp[3:]) > TOPPLE_ANG:
          return ActionResult(action, False, "toppled", p)
    if action[0] == "assemble" and not self._seated(action[2]):
      return ActionResult(action, False, "not_seated", action[2])
    return ActionResult(action, True)


# ---------------------------------------------------------------------------------------------
# pose utilities (wxyz, matching problem.py) + state readers
# ---------------------------------------------------------------------------------------------
def _quat_conj(q):
  return np.array([q[0], -q[1], -q[2], -q[3]])


def _slerp(q0, q1, a):
  q0 = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
  q1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
  d = float(np.dot(q0, q1))
  if d < 0.0:
    q1, d = -q1, -d
  if d > 0.9995:                                           # nearly parallel: lerp
    q = (1 - a) * q0 + a * q1
    return q / np.linalg.norm(q)
  th = np.arccos(min(1.0, d))
  return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def _quat_angle(q1, q2):
  d = abs(float(np.dot(np.asarray(q1) / np.linalg.norm(q1),
                       np.asarray(q2) / np.linalg.norm(q2))))
  return 2.0 * np.arccos(min(1.0, d))


def _rel_pos(pose_a, pose_b):
  return _quat_to_R(np.asarray(pose_a[3:])).T @ (np.asarray(pose_b[:3])
                                                 - np.asarray(pose_a[:3]))


def _held_grasp_id(state, part):
  for c in state.supports.get(part, ()):
    if c[0] == "grasp" and c[2] == part:
      return c[3]
  raise ValueError(f"{part} not held in this state")


def _holder_of(state, part, exclude=None):
  for c in state.supports.get(part, ()):
    if c[0] == "grasp" and c[1] != exclude:
      return c[1]
  return None


# ---------------------------------------------------------------------------------------------
# batch validation entry (the paper table)
# ---------------------------------------------------------------------------------------------
def run_validation(assembly, plan, states, gripper_kinds=None, n_trials=1, gui=False):
  """Execute a plan n_trials times (deterministic -> 1 unless start perturbations are added;
  the perturbed study reuses --perturb-start-pose semantics on states[0] -- TODO) and return
  the SimReports. Intended caller: a --sim-validate branch in main.py after --solve."""
  return [SimExecutor(assembly, gripper_kinds=gripper_kinds, gui=gui).execute(plan, states)
          for _ in range(n_trials)]


# ---------------------------------------------------------------------------------------------
# v2: execute an mr_bench MOTION PLAN (BiRRT path) with FK-synced arm links
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


def execute_motion_plan(C, assembly, plan, states, problem, mp_path, gripper_kinds=None,
                        record=None, verbose=True, gui=False, visual=True, prioritized=False,
                        rate_boost=False, smooth=False,
                        corridor=False, corridor_ds=1.0, corridor_ds_lead=0.25,
                        corridor_lead_max=4.0, corridor_verify=0, corridor_refine=0,
                        rate_clamp=False, settle_tol=0.0, settle_max=1.0, couple_handovers=False,
                        directional_caps=False,
                        verify_efc_cache=False, force_limits=False, wrench_margin=1.0, mass_scale=1.0, inertia_factor=1.0,
                        corner_limits=True,
                        spline=False, spline_per_seg=8, spline_tension=0.5,
                        articulated=False, actuated=False, use_planner_table=False,
                        corridor_margin=0.0, weld_timeconst=None, limit_rate=False):
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
  # FK config: a private copy so the caller's C is untouched.
  C_fk = ry.Config()
  C_fk.addConfigurationCopy(C)
  jnames = C_fk.getJointNames()
  ridx = {r: [i for i, jn in enumerate(jnames) if jn.startswith(r)] for r in robots}

  arms = _arm_shape_frames(C_fk, robots, visual=visual)
  # Frame SET and frame OBJECTS hoisted out of fk_poses: it ran per step, rebuilding the set
  # and re-resolving ~90 frames BY NAME every time. rai Frame handles stay valid for the life
  # of the config, so the lookups are pure waste.
  _fk_frames = sorted({e[1] for e in arms})
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

  # ee anchor frame for grasp welds: the tool cylinder if present, else the last shape frame
  ee_anchor = {}
  for r in robots:
    cands = [e[1] for e in arms if e[1].startswith(r) and e[2][0] != "mesh"]
    ee_anchor[r] = next((f for f in cands if "gripper_fill" in f), cands[-1])

  q0 = full_q(mp_path[0].q)
  scene = SimScene(assembly, states[0], robots, gripper_kinds=gripper_kinds,
                   arms=arms, arm_poses=fk_poses(q0), ee_anchor=ee_anchor, gui=gui,
                   articulated_C=(C_fk if (articulated or actuated) else None),
                   actuated=actuated,
                   planner_table_top=(planner_table_top(C) if use_planner_table else None),
                   weld_timeconst=weld_timeconst)
  scene.verify_efc_cache = bool(verify_efc_cache)
  scene.directional_caps = bool(directional_caps)
  articulated = articulated or actuated
  if articulated:
    print(f"[sim/mp] --articulated ACTIVE: real kinematic chains generated from rai "
          f"(arm_model), driven by SETTING qpos -- not {len(arms)} teleported mocap bodies. "
          f"Still perfectly stiff; run fk_compare.py to re-check the model.")

  def settle_before_fire(q):
    """ACTUATED: hold the reference until the arm has actually ARRIVED, then fire.

    The schedule's dwell is a fixed DWELL_T, but a servo needs as long as it needs: measured on
    cross-new, a stop settles in ~0.54s against DWELL_T=0.15s, so every attach fired while the
    arm was still converging. _activate_weld anchors the grasp at the CURRENT gripper-to-part
    pose, so that residual is frozen into how the part sits in the hand and every downstream
    mate inherits it (measured pick-to-pick drift 0.05 -> 5 -> 40 -> 56mm before the damping
    fix). Waiting on the MEASURED error costs only the time actually needed, unlike raising
    DWELL_T for every transition whether it needs it or not. The stiff modes never need this
    (their error is 0 by construction), so it is actuated-only."""
    if not (actuated and settle_tol > 0.0):
      return
    for _ in range(int(settle_max / DT)):
      if scene.tracking_error(q) <= settle_tol:
        break
      drive_q(q)                                             # qdot=0: hold, do not lead
      scene.step(1, report, ("settle",))

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
  rec = None
  if record is not None:
    rec = scene.recorder = {"qpos": [], "mpos": [], "mquat": [], "forces": [], "marks": []}
  report = SimReport()

  # task name -> (plan index, is-final-task-of-that-action)
  name_of = {}
  for i, nms in enumerate(problem.step_names):
    for k, nm in enumerate(nms):
      name_of[nm] = (i, k == len(nms) - 1)

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
    if t.name == "terminal" or not getattr(t, "frames", None):
      return                                               # goto / terminal: nothing to relink
    parent, child = t.frames[0], t.frames[1]
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
    # map to the plan action (for the taxonomy label + expected-state judge), if this task is
    # the FINAL task of an action (pre-insertion gotos have no frames and never reach here).
    lbl = name_of.get(t.name)
    act = tuple(plan[lbl[0]]) if (lbl and lbl[1]) else ("attach", parent, child)
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
    elif lbl and lbl[1]:
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

  def drive_stretch(qs):
    # INTERIM ZERO-VELOCITY-AT-GOALS (kinodynamic MP is the real fix, out of scope for now):
    # the raw path has no dynamics, so constant-rate tracking blew THROUGH the task goals and
    # the held part carried momentum into every grasp/mate/release event. Each per-mode stretch
    # gets an execution-side TRAPEZOIDAL profile -- ramp from ~0, cruise at JOINT_RATE, ramp
    # back to ~0 over the last RAMP_DIST rad (retiming.retime_stretch) -- so arms arrive at
    # every transition (where attaches fire) essentially at rest. Path GEOMETRY is untouched;
    # this is exactly what a jerk-limited tracking controller would do on hardware.
    for q in retiming.retime_stretch(qs, DT, joint_rate=JOINT_RATE, ramp_dist=RAMP_DIST,
                                     ramp_min=RAMP_MIN, arm_dims=boost_dims,
                                     vmax=boost_vmax):
      drive_q(q)
      scene.step(1, report, ("path",))

  if prioritized or corridor:
    # DESYNCED, EVENT-DRIVEN drive (retiming.plan_schedule): only the arm(s) ACTING at a
    # barrier ramp to rest there; uninvolved arms cross without stopping. The acting arm is
    # already at rest at each fire sample, so no pre-fire dwell is needed; we keep only the
    # post-fire dwell (freezing briefly) to let the new weld take hold -- exactly the composite
    # path's second dwell. Everything else (fire semantics, judging, settle, tail) is shared.
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
        if lbl:
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
    import mp_export as _MP
    chain = _MP.mode_sequence(problem)
    # ALIGNMENT BY MODE IDENTITY: the saved path's lightweight modes carry task_ids only, so
    # match each per-stretch path mode to the rebuilt chain mode with the SAME active-task set
    # (unique in a sequence problem -- progression is monotone). The path may collapse chain
    # transitions that share a config (invisible goto completions), so counting fired tasks
    # under-counts; the task_ids set is the unambiguous key. World state after `fired`
    # barriers = the mode stretch `fired` executes under.
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
    stretch_mode = [chain_by_ids[i] for i in path_mode_ids]  # mode in force during stretch k
    conf_type = type(problem.get_start_pos())
    def _conf(fq):
      return conf_type.from_list([np.asarray(fq)[ridx[r]] for r in robots])
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
      _P, _bi = retiming._composite(stretches)
      if spline:
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
      _stretch_plan = [next((name_of[problem.tasks[int(t)].name][0] for t in stretches[k][1]
                             if problem.tasks[int(t)].name in name_of), None)
                       for k in range(len(stretches))]
      _last = 0
      for k in range(len(_stretch_plan)):                     # a goto-only stretch inherits
        if _stretch_plan[k] is None:
          _stretch_plan[k] = _last
        else:
          _last = _stretch_plan[k]
      _locked = []
      for k in range(len(stretches)):
        st_idx = _stretch_plan[k]
        if st_idx is None:
          continue
        a0 = _bar[k - 1] if k else 0.0
        for state in (states[st_idx], states[min(st_idx + 1, len(states) - 1)]):
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
        _G = len(cor_grid := np.arange(0.0, _D + corridor_ds, corridor_ds))
        _p = np.zeros((len(robots), _G, 3))                  # held-group COM path
        _lev = np.zeros((len(robots), _G))                    # |COM - gripper| lever arm
        _m = np.zeros((len(robots), _G))                      # held mass
        _bad_lever = 0
        for _g, _s in enumerate(cor_grid):
          _k = min(_fired_at(_s), len(stretch_mode) - 1)
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
            _p[_r, _g] = com
            _m[_r, _g] = gm
            _lev[_r, _g] = max(lev, 5e-3)
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
        _bind, _cbind = {}, {}
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
            _g = int(min(max(round(_s / corridor_ds), 0), _G - 1))
            m, L = _m[_r, _g], _lev[_r, _g]
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
        _vmax = retiming.brakeable(_vmax, _amax, _dsf)
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
            print(f"    arm{_r}: held mass up to {_m[_r].max():.2f}kg, lever up to "
                  f"{_lev[_r].max() * 1000:.0f}mm, "
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
        for i in range(0, len(_t.q), max(1, step)):
          fired = int(np.searchsorted(fire_samples, i, side="right"))
          if not collide(_t.q[i], fired):
            continue
          real = [(pr, dp) for pr, dp in getattr(collide, "last_pairs", [])
                  if not _legit_pair(pr, fired)]
          if not real:
            continue                                         # only grasps/mates: not a violation
          arms = {a for pr, _ in real for a in (_arm_of(pr[0]), _arm_of(pr[1]))
                  if a is not None}
          out.append((i, (list(prog[i]) if prog is not None else None), real, sorted(arms)))
        return out

      _ds = corridor_ds
      _extra_locks, cor, traj = [], None, None
      for _pass in range(max(0, corridor_refine) + 1):
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
      base = retiming.build_trajectory(stretches, DT, joint_rate=JOINT_RATE,
                                       ramp_dist=RAMP_DIST, ramp_min=RAMP_MIN, dwell=DWELL_T)
      _ff = getattr(traj, "force_floor", None) or {}
      if _ff:
        print("[sim/mp] !!! --force-limits: the wrench cap was BELOW the never-stall floor "
              f"({RAMP_MIN * JOINT_RATE:g} rad/s) at " +
              ", ".join(f"arm{r} {n} sample(s)" for r, n in sorted(_ff.items())) +
              " -- those waypoints are traversed at the CAP now (previously the floor silently "
              "won). Expect a longer schedule; a near-zero cap means the LP considers that "
              "waypoint untraversable.")
      print(f"[sim/mp] STAGES B+C: {len(traj.q)} samples vs {len(base.q)} lockstep "
            f"({100 * (1 - len(traj.q) / max(len(base.q), 1)):+.1f}% makespan), "
            f"{len(traj.fires)} fire points, ZERO collision checks during scheduling")
      print(retiming.format_motion_report(
          retiming.motion_report(base.q, DT, arm_dims, JOINT_RATE), "LOCKSTEP baseline"))
      print(retiming.format_motion_report(
          retiming.motion_report(traj.q, DT, arm_dims, JOINT_RATE), "CORRIDOR"))
      fire_at = {}
      for idx, ev in traj.fires:
        for k in ev:                                         # fires carry stretch indices here
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
          g = int(min(max(round(s / corridor_ds), 0), _G - 1))
          gi = int(min(max(round(s / _dsf), 0), len(_sf) - 1))
          m, L = float(_m[k, g]), float(_lev[k, g])
          av = _d1[k][gi] * sdd + _d2[k][gi] * v * v          # a = p'*s_ddot + p''*s_dot^2
          a = float(np.linalg.norm(av))
          fs, ts = _static_load(m, L)
          return {"s": s, "v": v, "acc": sdd, "m": m, "L": L, "a": a,
                  "F": fs + m * a, "T": ts + m * a * L, "vmax": float(_vmax[k][gi]),
                  "kink": float(_jump[k][gi]),
                  "dbar": min([abs(s - b) for b in _bar] or [-1.0])}

        scene.wrench_trace = _make_wrench_tracer(scene, predict=_wpred)
      for i in range(len(traj.q)):
        drive_q(traj.q[i])
        scene.trace_ctx = i
        scene.step(1, report, (traj.phase[i],))
        if i in fire_at:
          settle_before_fire(traj.q[i])
          for tid in fire_at[i]:
            fire(tid)
      scene.step(int(SETTLE_T / DT), report, ("settle",))
      report.final_pose_err = {}
      for p in assembly.objects:
        exp = np.asarray(_state_part_poses(states[-1], assembly, lift=CARRY_LIFT)[p], dtype=float)
        report.final_pose_err[p] = (
            float(np.linalg.norm(scene.part_pose(p)[:3] - exp[:3])),
            _quat_angle(scene.part_pose(p)[3:], exp[3:]))
      if rec is not None:
        _save_recording(record, scene, rec, report)
      return report

    traj = retiming.plan_schedule(stretches, arm_dims, acting, DT, joint_rate=JOINT_RATE,
                                  ramp_dist=RAMP_DIST, ramp_min=RAMP_MIN,
                                  collide=collide, dwell=DWELL_T, rate_boost=rate_boost)
    base = retiming.build_trajectory(stretches, DT, joint_rate=JOINT_RATE,   # lockstep, as data
                                     ramp_dist=RAMP_DIST, ramp_min=RAMP_MIN, dwell=DWELL_T,
                                     arm_dims=boost_dims)
    print(f"[sim/mp] prioritized retiming ACTIVE: {len(traj.q)} samples vs {len(base.q)} "
          f"lockstep ({100 * (1 - len(traj.q) / max(len(base.q), 1)):+.1f}% makespan), "
          f"{len(traj.fires)} fire points, collision oracle: mr_bench is_collision_free "
          f"(tol {getattr(problem, 'collision_tolerance', '?')})")
    if smooth:
      # SPREAD each arm's motion continuously between ITS OWN events instead of sprinting
      # to the next barrier and parking there. Event indices (and therefore the whole firing
      # schedule) are unchanged -- only the pacing between arrivals. Oracle-checked per
      # interval, so it can never introduce a conflict the schedule did not already accept.
      _P, _bi = retiming._composite(stretches)
      _sg, _D = retiming.arc_length(list(_P))
      _cs = np.concatenate([[0.0], np.cumsum(_sg)])
      _bar = [_cs[b] for b in _bi]
      _fire_at = sorted(int(i) for i, _ in traj.fires)
      def _fired_of(i):
        return int(np.searchsorted(_fire_at, i, side="right"))
      before = retiming.motion_report(traj.q, DT, arm_dims, JOINT_RATE)
      traj.q = retiming.spread_between_events(
          traj, arm_dims, _bar, acting, DT, JOINT_RATE, ramp_dist=RAMP_DIST,
          P=_P, csum=_cs, segs=_sg, collide=collide, fired_of=_fired_of)
      print(f"[sim/mp] --smooth ACTIVE: rush-then-wait removed between each arm's own "
            f"events (schedule/fire times unchanged, oracle-checked per interval)")
      print(retiming.format_motion_report(before, "PRIORITIZED before --smooth"))
    # MOTION QUALITY is the goal metric (smooth arms), not makespan -- print both profiles.
    print(retiming.format_motion_report(
        retiming.motion_report(base.q, DT, arm_dims, JOINT_RATE), "LOCKSTEP baseline"))
    print(retiming.format_motion_report(
        retiming.motion_report(traj.q, DT, arm_dims, JOINT_RATE), "PRIORITIZED"))
    if verbose:
      print(f"[sim/mp] acting sets {[sorted(a) for a in acting]}")
    fire_at = {}
    for idx, ev in traj.fires:
      fire_at.setdefault(int(idx), []).extend(int(t) for t in ev)
    for i in range(len(traj.q)):
      drive_q(traj.q[i])
      scene.step(1, report, (traj.phase[i],))
      if i in fire_at:
        settle_before_fire(traj.q[i])
        for tid in fire_at[i]:
          fire(tid)                                          # acting arms rest through the
                                                             # schedule's own post-fire dwell
  else:
    for qs, done in stretches:
      drive_stretch(qs)
      scene.step(int(DWELL_T / DT), report, ("dwell",))      # settle: v ~= 0 before relinking
      for tid in done:
        fire(tid)
      scene.step(int(DWELL_T / DT), report, ("dwell",))      # let the new attach take hold
  scene.step(int(SETTLE_T / DT), report, ("settle",))

  report.final_pose_err = {}
  for p in assembly.objects:
    exp = np.asarray(_state_part_poses(states[-1], assembly, lift=CARRY_LIFT)[p], dtype=float)
    report.final_pose_err[p] = (
        float(np.linalg.norm(scene.part_pose(p)[:3] - exp[:3])),
        _quat_angle(scene.part_pose(p)[3:], exp[3:]))
  if rec is not None:
    _save_recording(record, scene, rec, report)
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
      if _quat_angle(got[3:], exp[3:]) > TOPPLE_ANG:
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
def _gripper_kinds_from_cfg(cfg, robots):
  ks = [k.strip() for k in (cfg.get("grippers") or "").split(",") if k.strip()]
  if ks:                                                    # mixed EEs (--grippers CSV)
    ks = (ks + [ks[-1]] * len(robots))[:len(robots)]
    return {r: ks[i] for i, r in enumerate(robots)}
  return {r: ("vacuum" if cfg.get("vacuum") else "two_finger") for r in robots}


def _states_from_plan(cfg, C, asm, robots, plan):
  """Re-derive the discrete state trajectory the same way tests/_replay_ok does -- the saved
  JSON carries only actions, and the successor flags (gripper kind, bracing) must match the
  search that produced it (both are in the saved config)."""
  import assembly as A
  gk = A.gripper_kind(C) if cfg.get("wrench") else None
  brace = bool(cfg.get("bracing"))
  _, gt = A._search_context(C, asm, robots)
  states = [A.get_initial_state(C, asm)]
  for act in plan:
    states.append(next(ns for a, ns in A.successors(states[-1], asm, robots, gt,
                                                     gripper_kind=gk, bracing=brace)
                       if a == act))
  return states


def _cli_run(a):
  global JOINT_RATE
  if getattr(a, "joint_rate", None):
    JOINT_RATE = float(a.joint_rate)                          # per-arm cruise speed override
    print(f"[sim] --joint-rate {JOINT_RATE:g} rad/s (default 1.0)")
  import assembly as A
  import replay as R
  from plan_io import read_plan
  cfg, plan = read_plan(a.path)
  C, asm = R.build_scene(cfg)                               # exact reconstruction (shared)
  robots = A.get_robots(C)
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
        f"{len(plan)} actions; executing "
        f"({'v2 real-arm motion plan' if a.v2 else 'v1 keyframe'}) ...")

  if a.v2:
    if cfg.get("floating_ee"):
      print("[sim] --v2 needs a REAL-ARM plan: save from a run WITHOUT --floating-ee (and "
            "with --real-ik). This plan is floating-EE.")
      return
    from solver import RobotIK, optimize_witnesses
    import mp_export as MP
    gk = A.gripper_kind(C) if cfg.get("wrench") else None
    rik = RobotIK(C, robots)                                # geometric export only -- the
    # RobotIK's own default is ik_seeding=False, but the CONFIG default is True and main.py
    # turns it on before it exports; every other export path (mp_show_path, mp_debug_view,
    # mp_show_modes, mp_viser_witness) sets it explicitly too. Leaving it off HERE made the
    # witness IK fail for re-picks and reorienting places -- optimize_witnesses returned None
    # and mp_export silently substituted the arm's PREVIOUS config, so the part was never
    # carried anywhere (measured on car-mini: obj_0's two reorienting places became 0.07deg
    # no-ops, the base stayed flat, and every wheel then missed its mate by ~70mm).
    _seed = cfg.get("ik_seeding")
    rik.ik_seeding = True if _seed is None else bool(_seed)
    # VERBATIM from the saved config -- 0.0 means NO pre-insertion split. (A silent `or 0.05`
    # default here rebuilt the problem WITH split tasks while the saved path used unsplit
    # task ids -> every attach after the first assemble fired at the wrong task; user-hit.)
    rik.pre_insertion_delta = float(cfg.get("pre_insertion") or 0.0)
    # ...and the OTHER three splits, for the same reason: each one ADDS a task, so a rebuild
    # that misses them produces a shorter task list than the saved --mp-path was planned
    # against, and the path indexes tasks by integer id (the signature guard below catches it,
    # but only if we even try). pre_insertion alone was restored here.
    rik.pre_pick_delta = float(cfg.get("pre_pick") or 0.0)
    rik.pre_place_delta = float(cfg.get("pre_place") or 0.0)
    rik.pre_add_grasp_delta = float(cfg.get("pre_add_grasp") or 0.0)
    # ...and the eq WEIGHTS, which decide how exactly a mate/place lands. Both were saved but
    # never restored here, so the sim re-solved every witness at the default 1e1 and got
    # different keyframes than the export it is supposed to be replaying.
    if cfg.get("mate_eq_weight") is not None:
      rik.mate_eq_weight = float(cfg["mate_eq_weight"])
    if cfg.get("place_eq_weight") is not None:
      rik.place_eq_weight = float(cfg["place_eq_weight"])
    if cfg.get("mate_accuracy") is not None:
      rik.mate_residual_reject = float(cfg["mate_accuracy"])
    if cfg.get("ik_accuracy") is not None:
      rik.nlp_stop_tolerance = float(cfg["ik_accuracy"])
    if cfg.get("witness_robot_collisions"):
      rik.strip_inactive_collision = False
    if cfg.get("strict_collisions") is not None:
      rik.strip_inactive_collision = False
      rik.penetration_reject = float(cfg["strict_collisions"])
      rik.col_weight = 1e2
    # successor flags MUST match the search that produced the plan, or the state re-derivation
    # inside optimize_witnesses can't reproduce an action (e.g. car-mini's add_grasp needs
    # bracing=True). bracing is saved; handovers/allow_place aren't (they default True).
    brace = bool(cfg.get("bracing"))
    _hand = cfg.get("handovers"); _hand = True if _hand is None else bool(_hand)
    _aplace = cfg.get("allow_place"); _aplace = True if _aplace is None else bool(_aplace)
    # WHICH witness optimizer ran at export time, reproduced here. build_tasks SKIPS a task
    # whose witness is None, so a different optimizer -> a different None-pattern -> a task
    # chain that no longer matches the saved --mp-path ("mode alignment failed: N unmatched").
    _witopt = cfg.get("optimize_witnesses")
    if _witopt == "floor":
      from witness_floor import optimize_witnesses_floor
      wit, _ = optimize_witnesses_floor(
          C, asm, plan, robots, rik, gripper_kind=gk, handovers=_hand, allow_place=_aplace,
          bracing=brace, branch_budget=cfg.get("witness_branch_budget"),
          branch_metric=cfg.get("witness_branch_metric"),
          reconfig_thresh=cfg.get("witness_reconfig_thresh"))
    elif _witopt:
      wit, _ = optimize_witnesses(C, asm, plan, robots, rik, cost=_witopt,
                                  gripper_kind=gk, handovers=_hand, allow_place=_aplace,
                                  bracing=brace, clear_weight=cfg.get("witness_clearance") or 0.0,
                                  robot_collisions=bool(cfg.get("witness_robot_collisions")),
                                  verbose=False)
    else:
      wit, _ = optimize_witnesses(C, asm, plan, robots, rik, cost="ee", sweeps=0,
                                  gripper_kind=gk, bracing=brace, handovers=_hand,
                                  allow_place=_aplace,
                                  robot_collisions=bool(cfg.get("witness_robot_collisions")),
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
    prob = MP.make_problem(C, asm, plan, wit, robots, rik, gk, bracing=brace,
                           depgraph=False, collision_tolerance=a.mp_tolerance,
                           robot_robot_margin=float(cfg.get("mp_robot_margin") or 0.0),
                           robot_margin_ramp=float(cfg.get("mp_margin_ramp") or 0.0))
    if a.mp_path:                                          # load a saved BiRRT path -> no re-plan
      path = MP.load_path(a.mp_path)
      print(f"[sim] loaded motion path ({len(path)} states) from {a.mp_path} -- skipping BiRRT")
      # ALIGNMENT GUARD: the path's mode task-ids index MAIN's task list; the rebuilt problem
      # must have the identical task structure or every attach fires at the wrong time.
      _want = getattr(MP.load_path, "last_signature", None)
      if _want is not None:
        _have = MP.task_signature(prob)
        if _have != _want:
          print(f"[sim] !!! TASK-SIGNATURE MISMATCH: the path was planned against task structure "
                f"{_want}, the rebuild produced {_have}. Same task COUNT can still mean different "
                f"tasks (a moved --pre-* split, a different bracing/handovers flag), and the path "
                f"indexes tasks by integer id -- so every attach would fire at the wrong moment. "
                f"Aborting. Re-export the path from this plan, or use the plan the path came from.")
          return
      max_tid = max(int(t) for st in path for t in np.atleast_1d(st.mode.task_ids))
      if max_tid + 1 != len(prob.tasks):
        print(f"[sim] !!! TASK-STRUCTURE MISMATCH: saved path references {max_tid + 1} tasks, "
              f"rebuilt problem has {len(prob.tasks)}. The plan.json/mp.json pair does not "
              f"match this rebuild (usually a pre-insertion/--skill difference: saved config "
              f"has pre_insertion={cfg.get('pre_insertion')}). Aborting -- executing would "
              f"relink at the wrong times.")
        return
    else:
      path, _ = MP.run_birrt(prob, runtime=a.mp_runtime, verbose=a.verbose)
    if not path:
      print("[sim] motion planner found no path.")
      return
    if a.mp_cograsp:
      print("[flag] --mp-cograsp ACTIVE: checking co-grasp rigidity along the loaded path.")
      MP.report_cograsp_drift(C, prob, path, rik)
    rep = execute_motion_plan(C, asm, plan, states, prob, path, gripper_kinds=gripper_kinds,
                              record=a.record, gui=a.gui, visual=_want_meshes,
                              verbose=a.verbose, prioritized=a.prioritized,
                              rate_boost=a.arm_rate, smooth=a.smooth,
                              corridor=a.corridor, corridor_ds=a.corridor_ds,
                              corridor_ds_lead=a.corridor_ds_lead,
                              corridor_lead_max=a.corridor_lead_max,
                              corridor_verify=a.corridor_verify,
                              corridor_refine=a.corridor_refine,
                              rate_clamp=a.corridor_rate_clamp,
                              settle_tol=a.settle_tol, settle_max=a.settle_max,
                              directional_caps=a.directional_caps,
                              couple_handovers=a.couple_handovers,
                              verify_efc_cache=a.verify_efc_cache,
                              force_limits=a.force_limits, wrench_margin=a.wrench_margin,
                              mass_scale=a.mass_scale, inertia_factor=a.inertia_factor,
                              corner_limits=a.corner_limits,
                              spline=a.spline, spline_per_seg=a.spline_per_seg,
                              spline_tension=a.spline_tension,
                              articulated=a.articulated, actuated=a.actuated,
                              use_planner_table=getattr(a, 'planner_table', False),
                              corridor_margin=getattr(a, 'corridor_margin', 0.0),
                              weld_timeconst=getattr(a, 'weld_timeconst', 0.0),
                              limit_rate=getattr(a, 'limit_rate', False))
  else:
    rep = SimExecutor(asm, gripper_kinds=gripper_kinds, gui=a.gui).execute(
        plan, states, record=a.record)
  print(rep.summary())


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
  r.add_argument("--v2", action="store_true",
                 help="REAL-ARM motion-plan execution: rebuild witnesses + run mr_bench BiRRT "
                      "for the SAVED discrete plan, then FK-synced arms follow the path in "
                      "sim. Needs a plan saved from a --real-ik (non-floating) run. Default "
                      "(v1) executes the plan with floating-gripper keyframes.")
  r.add_argument("--mp-path", metavar="FILE", default=None,
                 help="v2: load a saved BiRRT path (main --motion-plan --mp-save FILE) instead "
                      "of re-planning. Must be from the SAME run as the plan JSON.")
  r.add_argument("--mp-tolerance", type=float, default=0.01,
                 help="v2: exported collision tolerance (1e-6 = zero-tol; the plan's witnesses "
                      "must be clean -- save it from a --strict-collisions run). Ignored with "
                      "--mp-path (the saved path already encodes it).")
  r.add_argument("--mp-runtime", type=float, default=60.0,
                 help="v2: BiRRT budget (s); ignored with --mp-path")
  r.add_argument("--verify-efc-cache", action="store_true",
                 help="debug: re-derive the equality-row index EVERY step and assert it matches "
                      "the eq_active-keyed cache. Checks the MuJoCo layout assumption that the "
                      "equality block is a function of the active set alone. Slow; off by default.")
  r.add_argument("--couple-handovers", action="store_true",
                 help="corridor: also lock giver+receiver over a handover's stretch. A handover "
                      "is ATOMIC discretely (giver holds before, receiver after, never both), so "
                      "the state-scan coupling misses it entirely -- yet both grippers are on "
                      "the part in execution. OFF = byte-identical.")
  r.add_argument("--directional-caps", action="store_true",
                 help="breakaway uses the planner's DIRECTIONAL grasp polytope (force at the "
                      "grasp point + residual pure couple) instead of min-over-directions. More "
                      "permissive in force (F_push vs a blanket F_pull), STRICTER in torque. "
                      "OFF by default: it regresses configs that pass today, which indicts the "
                      "hand-set couple caps rather than the test.")
  r.add_argument("--settle-tol", type=float, default=1e-3, metavar="RAD",
                 help="--actuated: before each attach fires, hold the reference until the max "
                      "joint tracking error is under RAD. DEFAULT 1e-3 (0 disables): settling "
                      "takes ~0.54s against a 0.15s DWELL_T, so attaches used to fire mid-"
                      "convergence and _activate_weld froze the residual into the hand -- "
                      "measured pick-to-pick drift 0.05 -> 56mm, and enabling this removed EVERY "
                      "not_seated failure on cross-new. No effect unless --actuated. "
                      "joint tracking error is under RAD (0 = off, byte-identical). The "
                      "schedule dwells a fixed DWELL_T, but a servo settles when it settles "
                      "(measured 0.54s vs a 0.15s dwell), and the grasp weld freezes whatever "
                      "residual exists at fire time into the hand.")
  r.add_argument("--settle-max", type=float, default=1.0, metavar="SEC",
                 help="--settle-tol: give up waiting after this long (default 1.0s).")
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
  r.add_argument("--corridor-rate-clamp", action="store_true",
                 help="corridor: bound each step by the joint motion it ACTUALLY produces. "
                      "rate_of() budgets joint_rate using the arm's arc share in the segment it "
                      "is IN, but a step can land in a segment where that arm carries more of "
                      "the arc, so the executed joint speed overshoots (measured 1.98 rad/s at "
                      "joint_rate 1.0). Stiff arms absorb it; --actuated turns it into grasp "
                      "load. OFF = byte-identical.")
  r.add_argument("--mp-cograsp", action="store_true",
                 help="v2: before executing, report how far each pair of co-grasping tool "
                      "centres drifts from its keyframe relative pose along the path. Lets a "
                      "SAVED plan+path pair be audited without re-planning.")
  r.add_argument("--no-arm-meshes", action="store_true",
                 help="v2: skip the real link VISUAL meshes (default on) -- capsules only. "
                      "Faster to build; the collision model is the capsules either way.")
  r.add_argument("--joint-rate", type=float, default=None, metavar="RAD_S",
                 help="Override the per-arm joint-space cruise speed (default JOINT_RATE=%g). "
                      "Lower it when faster motion overloads a grasp (a heavy held part's "
                      "inertial wrench is NOT in the quasi-static LP)." % JOINT_RATE)
  r.add_argument("--corridor", action="store_true",
                 help="CORRIDOR RETIMING (separate from --smooth/--prioritized). STAGE A first "
                      "MEASURES how far arms may desync before anything touches -- a certified "
                      "envelope, computed once with the same oracle -- then STAGES B+C plan "
                      "inside it with ZERO collision checks: an arm rests only at ITS OWN "
                      "events (never at a barrier it merely crosses) and runs at its own joint "
                      "limit. A pair whose corridor is 0 must move together, which is how "
                      "handovers/co-grasps express themselves. Prints the corridor widths.")
  r.add_argument("--actuated", action="store_true",
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
                 help="Drive REAL kinematic chains instead of teleported mocap bodies. Today "
                      "each arm is N DISCONNECTED mocap bodies, every one snapped to its rai-FK "
                      "world pose each step -- there are no joints between links at all. This "
                      "generates the actual chain from rai's own tree (arm_model) and drives it "
                      "by SETTING qpos: still perfectly stiff, so the grasp weld remains an "
                      "exact measure of the payload's m(a-g) and every wrench/corner calibration "
                      "stays interpretable. The step before actuators. Verify the model with "
                      "`python fk_compare.py --robot <type>` (agreement is machine-precision).")
  r.add_argument("--spline", action="store_true",
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
  r.add_argument("--force-limits", action="store_true",
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
                      "to -1.2% makespan vs lockstep and still failed) -- the two compound.")
  r.add_argument("--mass-scale", type=float, default=1.0, metavar="F",
                 help="Scale every part mass (default 1.0), in BOTH the MuJoCo bodies and the "
                      "wrench estimate, so they stay consistent. A test knob: the grasp caps in "
                      "wrench.PARAMS are hand-set placeholders and the planner only checks a "
                      "held group UPRIGHT, so a rotated carry can exceed the torque cap on "
                      "gravity alone -- scale down to take that out of the picture while "
                      "validating everything else.")
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
  r.add_argument("--smooth", action="store_true",
                 help="With --prioritized: spread each arm's motion CONTINUOUSLY between its "
                      "own events instead of sprinting to the next barrier and parking there "
                      "until the event fires. Fire times and the whole schedule are unchanged "
                      "-- only the pacing between arrivals, so an arm decelerates just at its "
                      "OWN events. Oracle-checked per interval; an interval that would conflict "
                      "keeps its scheduled timing.")
  r.add_argument("--arm-rate", action="store_true",
                 help="Apply JOINT_RATE per ARM instead of to the composite arc. Today an arm "
                      "moving alongside others advances at only |dq_arm|/|dq_composite| of the "
                      "nominal speed (measured 0.41-0.61 on car-mini), so arms crawl for a "
                      "purely parametric reason; this restores the intended semantics and never "
                      "exceeds it (bounded by sqrt(#arms)). On the shared clock it rescales TIME "
                      "only -- same configurations, so the planner's collision certificate still "
                      "holds with no extra checking. Works with and without --prioritized.")
  r.add_argument("--coll-scale", type=float, default=1.0, metavar="F",
                 help="DEBUG: shrink every contact-enabled ARM collision primitive by F "
                      "(capsule/cylinder radius, sphere radius, box extents; parts NEVER "
                      "scaled). Try 0.95/0.9 to give desync room when the oracle wedges. "
                      "Applied before witnesses/problem/sim so IK, the oracle and the MuJoCo "
                      "capsules agree. Default 1.0 = untouched. Not for validation runs: "
                      "clearances no longer match the planner's certified geometry.")
  r.add_argument("--record", metavar="FILE", default=None, help="save a replayable .npz")
  r.add_argument("--gui", action="store_true", help="live MuJoCo viewer during execution")
  r.add_argument("--prioritized", action="store_true",
                 help="EXPERIMENTAL v2 ASYNC retiming: arms desync so only the arm(s) acting at "
                      "each event (incl. holders) ramp to rest; uninvolved arms cross mode "
                      "barriers without stopping. Conflicts are found with mr_bench's own check "
                      "at desynced configs and resolved by FUSING the pair over disjoint arc "
                      "windows (CEGAR restarts). KNOWN GAP: the oracle SKIPS pairs the certified "
                      "path itself drives deep (gripper-vs-held-target during insertion), so "
                      "desync can drive those together unchecked -- measured on car-mini as a "
                      "grasp break at 81.8N and two missed seats. cross8 passes 39/39. Gains "
                      "little here anyway: ~2 of 3 arms must rest at each event, so most stops "
                      "are physically required. RETIMING_TRACE=1 traces couplings. Default = "
                      "lockstep. For smoother motion prefer --arm-rate, which is safe.")
  r.add_argument("--verbose", action="store_true")

  v = sub.add_parser("replay", help="view a recording (kinematic playback)")
  v.add_argument("path")
  v.add_argument("speed", nargs="?", type=float, default=1.0)
  v.add_argument("--no-loop", action="store_true")
  return p


if __name__ == "__main__":
  args = _build_cli().parse_args()
  if args.cmd == "run":
    _cli_run(args)
  else:
    replay_recording(args.path, speed=args.speed, loop=not args.no_loop)
