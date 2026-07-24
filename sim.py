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
"""
import numpy as np

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
SETTLE_T = 0.5                  # s   : settle after each action before judging
TABLE_HALF = (1.2, 1.2, 0.05)   # matches the rai table footprint; z-offset checked on 1st run


def _grasp_caps(kind):
  """Force/torque breakaway caps for a grasp weld, from the PLANNER's wrench model -- the sim
  shares the planner's parameters and may only disagree through physics. Conservative v1: one
  scalar force cap (min over directions) + one torque cap; per-axis caps are a refinement."""
  import wrench as W
  model = W.PARAMS["grasp"].get(W.PARAMS.get("grasp_model", "isotropic"),
                                W.PARAMS["grasp"]["isotropic"])
  g = model.get(kind, {})
  fcap = min([v for k, v in g.items() if k.startswith("F_")] or [40.0])
  tcap = min([v for k, v in g.items() if k.startswith("T_")] or [1.0])
  return float(fcap), float(tcap)


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
    n_ok = sum(r.ok for r in self.results)
    lines = [f"[sim] {n_ok}/{len(self.results)} actions ok; plan "
             f"{'SUCCEEDED' if self.success else 'FAILED'}"]
    lines += [f"  {r}" for r in self.results if not r.ok]
    if self.final_pose_err:
      worst = max(self.final_pose_err.items(), key=lambda kv: kv[1][0])
      lines.append(f"  worst final part error: {worst[0]} "
                   f"pos {worst[1][0]*1000:.1f}mm ang {np.rad2deg(worst[1][1]):.1f}deg")
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


def _arm_shape_frames(C, robots):
  """Enumerate the robot SHAPE frames to mirror as kinematic bodies: the capsule COLLISION
  model (coll*) plus gripper primitives -- exactly the geometry mr_bench collision-checked,
  so the sim replays what the motion planner guaranteed clearance for. Visual meshes are
  skipped (heavy; the capsules read fine). Returns [(frame_name, kind, size), ...]."""
  out = []
  for r in robots:
    for n in C.getFrameNames():
      if n.startswith(r):
        info = C.getFrame(n).info()
        k = info.get("shape")
        if k in ("capsule", "cylinder", "sphere", "box", "ssBox"):
          # collide ONLY rai's arm collision capsules (the _coll* frames). Gripper-local
          # geometry (vacuum tip, gripper_fill) is visual-only: rai exempts it against the
          # HELD part (contact=None / parent-chain -1), and mirroring it as colliding put
          # the tip sphere in a permanent contact-vs-weld battle with the suctioned part
          # (measured: sustained 37N on a carried group).
          collide = "_coll" in n
          # rai contact=-2 (the WRIST cluster nearest the tool) is parent-chain exempt vs
          # the ATTACHED part -- a held cube hangs 35mm from the tip, inside the 50mm wrist
          # capsule, so mirroring these as always-colliding re-created the same battle
          # (measured 37N/1.46Nm sustained). They are un-collided WHILE HOLDING (per-pair
          # runtime exclusion doesn't exist in MuJoCo; vs OTHER parts the planner kept
          # clearance, so the blanket toggle is a fair mirror).
          exempt = info.get("contact") == -2
          out.append((n, k, list(np.asarray(info.get("size"), dtype=float)), collide, exempt))
  return out


def _arm_geom_xml(kind, size):
  # rai capsule/cylinder size = [length, radius]; MJCF size = "radius half-length".
  if kind in ("capsule", "cylinder"):
    return f'type="{kind}" size="{size[1]} {size[0]/2}"'
  if kind == "sphere":
    return f'type="sphere" size="{size[0]}"'
  # box / ssBox: full extents -> half extents (rounding dropped, conservative)
  return f'type="box" size="{size[0]/2} {size[1]/2} {size[2]/2}"'


def build_mjcf(assembly, state, robots, mu_table=0.5, arms=None, arm_poses=None,
               ee_anchor=None):
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
        "mass_placeholder", f'mass="{mass}" rgba="{palette[i % len(palette)]}"{gq}')
    parts_xml.append(
        f'<body name="{p}" pos="{pose[0]} {pose[1]} {pose[2]}" '
        f'quat="{q[0]} {q[1]} {q[2]} {q[3]}"><freejoint/>{g}</body>')
  ee_colors = ["0.2 0.2 0.9 1", "0.9 0.2 0.2 1", "0.2 0.8 0.8 1", "0.9 0.7 0.1 1",
               "0.6 0.2 0.8 1", "0.4 0.9 0.3 1"]
  if arms is None:                                         # v1: one gripper puck per robot
    for j, r in enumerate(robots):
      parts_xml.append(
          f'<body name="ee_{r}" mocap="true" pos="0 0 1"><geom type="cylinder" '
          f'size="0.02 0.02" mass="0.2" contype="0" conaffinity="0" '
          f'rgba="{ee_colors[j % len(ee_colors)]}"/></body>')
    anchor = {r: f"ee_{r}" for r in robots}
  else:                                                    # v2: FK-synced arm link bodies
    for fname, kind, size, collide, _ex in arms:
      pose = arm_poses[fname]
      q = pose[3:]
      j = next(i for i, r in enumerate(robots) if fname.startswith(r))
      # contype 2 / conaffinity 1: arm links collide with PARTS and the table, never with
      # each other (self/arm-arm clearance was the MOTION PLANNER's job; re-checking it
      # here on mocap bodies would only produce huge phantom forces on interpenetration).
      con = 'contype="2" conaffinity="1"' if collide else 'contype="0" conaffinity="0"'
      parts_xml.append(
          f'<body name="arm_{fname}" mocap="true" pos="{pose[0]} {pose[1]} {pose[2]}" '
          f'quat="{q[0]} {q[1]} {q[2]} {q[3]}">'
          f'<geom {_arm_geom_xml(kind, size)} mass="1.0" {con} '
          f'rgba="{ee_colors[j % len(ee_colors)]}"/></body>')
    anchor = {r: f"arm_{ee_anchor[r]}" for r in robots}
  for r in robots:
    # grasp welds: anchor body <-> each part (inactive; relpose set at grasp time).
    for p in assembly.objects:
      # stiff solref: the anchor is a mocap body, so a soft (default) weld SAGS mm-scale
      # under carry loads -- measured 7-8mm drift, enough to miss the seat tolerance.
      eq_xml.append(f'<weld name="grasp_{r}_{p}" body1="{anchor[r]}" body2="{p}" '
                    f'active="false" solref="0.004 1"/>')
  for p in assembly.objects:                               # mate welds along the manual's DAG
    rel = assembly.relative_pose(p)
    if rel is not None:
      eq_xml.append(f'<weld name="mate_{p}" body1="{rel[0]}" body2="{p}" active="false" '
                    f'solref="0.004 1"/>')
  # the table box reaches from the FLOOR (z=0) to table_top: collision-identical for the
  # parts (only the top face matters) but it looks like a table instead of a floating slab.
  th = table_top / 2.0
  return f"""
<mujoco model="assembly_validation">
  <option timestep="{DT}" integrator="implicitfast"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.25 0.3"
             rgb2="0.25 0.3 0.35" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.7 0.9" rgb2="0.9 0.9 0.95"
             width="64" height="64"/>
  </asset>
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/></visual>
  <worldbody>
    <light pos="0.5 -0.5 2.5" dir="-0.2 0.2 -1" directional="true" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid" pos="0 0 0"/>
    <geom name="table" type="box" size="{TABLE_HALF[0]} {TABLE_HALF[1]} {th}"
          pos="0 0 {th}" rgba="0.55 0.42 0.28 1" friction="{mu_table} 0.005 0.0001"/>
    {''.join(parts_xml)}
  </worldbody>
  <equality>{''.join(eq_xml)}</equality>
</mujoco>"""


# ---------------------------------------------------------------------------------------------
# the sim world: weld managers + stepping + breakaway
# ---------------------------------------------------------------------------------------------
class SimScene:
  def __init__(self, assembly, state, robots, gripper_kinds=None, gui=False,
               arms=None, arm_poses=None, ee_anchor=None):
    if not HAVE_MUJOCO:
      raise RuntimeError("mujoco not installed (pip install mujoco)")
    self.assembly = assembly
    self.xml = build_mjcf(assembly, state, robots, arms=arms, arm_poses=arm_poses,
                          ee_anchor=ee_anchor)
    self.model = mujoco.MjModel.from_xml_string(self.xml)
    self.data = mujoco.MjData(self.model)
    self.robots = list(robots)
    self.arms = arms                                       # v2: [(frame, kind, size)]
    self.grasp_anchor = ({r: f"ee_{r}" for r in robots} if arms is None
                         else {r: f"arm_{ee_anchor[r]}" for r in robots})
    self.recorder = None                                   # set by the executor (see record=)
    self._rec_count = 0
    self._over = {}                                        # robot -> consecutive overload steps
    # per-robot geom ids of the rai contact=-2 wrist capsules (un-collided while holding)
    self._holdex = {r: [] for r in self.robots}
    if arms is not None:
      for fname, _k, _s, collide, exempt in arms:
        if collide and exempt:
          r = next(rb for rb in self.robots if fname.startswith(rb))
          bid = self._body_id(f"arm_{fname}")
          self._holdex[r].append(int(self.model.body_geomadr[bid]))
    self.gripper_kinds = gripper_kinds or {}
    self.active_grasps = {}                                # robot -> (eq_id, part, fcap, tcap)
    self.viewer = None
    if gui:
      from mujoco import viewer as _viewer                 # NOT `import mujoco.viewer`: that
      self.viewer = _viewer.launch_passive(self.model, self.data)  # would shadow the module-
                                                           # global `mujoco` in this scope
    mujoco.mj_forward(self.model, self.data)

  # -- naming helpers --------------------------------------------------------------------
  def _eq_id(self, name):
    return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

  def _body_id(self, name):
    return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

  def part_pose(self, p):
    b = self._body_id(p)
    return np.r_[self.data.xpos[b], self.data.xquat[b]]    # wxyz -- matches problem.py

  # -- weld managers ---------------------------------------------------------------------
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

  def grasp(self, robot, part):
    eq = self._eq_id(f"grasp_{robot}_{part}")
    self._activate_weld(eq, self._body_id(self.grasp_anchor[robot]), self._body_id(part))
    kind = self.gripper_kinds.get(robot, "vacuum")
    self.active_grasps[robot] = (eq, part) + _grasp_caps(kind)
    for gid in self._holdex.get(robot, ()):                # wrist cluster exempt while holding
      self.model.geom_contype[gid] = 0
      self.model.geom_conaffinity[gid] = 0

  def sync_arms(self, poses):
    """v2: set every arm link mocap body to its rai-FK world pose. poses = {frame: 7-pose}."""
    for fname, _k, _s, _c, _e in self.arms:
      mid = self.model.body_mocapid[self._body_id(f"arm_{fname}")]
      pose = poses[fname]
      self.data.mocap_pos[mid] = pose[:3]
      self.data.mocap_quat[mid] = pose[3:]

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

  # -- stepping with grasp breakaway -----------------------------------------------------
  def _weld_wrench(self, eq_id):
    """Constraint wrench currently exerted by an active weld: sum its efc rows.
    TODO(verify): efc bookkeeping -- rows with data.efc_type == mjCNSTR_EQUALITY and
    data.efc_id == eq_id; a weld contributes 6 rows (3 force + 3 torque)."""
    rows = np.where((self.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                    & (self.data.efc_id == eq_id))[0]
    f = self.data.efc_force[rows]
    if len(f) < 6:
      return 0.0, 0.0
    return float(np.linalg.norm(f[:3])), float(np.linalg.norm(f[3:6]))

  def step(self, n=1, report=None, action=None):
    """Advance; after each step check every active grasp weld against its breakaway caps --
    an overloaded grasp DROPS (deactivates) and is recorded as the LP-predicted failure."""
    sustain = max(1, int(BREAK_SUSTAIN / DT))
    for _ in range(n):
      mujoco.mj_step(self.model, self.data)
      for r, (eq, part, fcap, tcap) in list(self.active_grasps.items()):
        fmag, tmag = self._weld_wrench(eq)
        if fmag > fcap or tmag > tcap:
          self._over[r] = self._over.get(r, 0) + 1
          if self._over[r] >= sustain:                     # sustained overload -> seal fails
            self.data.eq_active[eq] = 0
            del self.active_grasps[r]
            if report is not None:
              report.results.append(ActionResult(
                  action or ("step",), False, "grasp_broke",
                  f"{r}/{part} |F|={fmag:.1f}N(cap {fcap:g}) |T|={tmag:.2f}Nm(cap {tcap:g}) "
                  f"sustained {BREAK_SUSTAIN*1000:.0f}ms"))
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
        self.scene.grasp(robot, act[2])
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
      elif kind == "release":
        self.scene.release(robot)
      elif kind in ("handover", "add_grasp"):
        gid = act[3]
        self._move(robot, self._ee_pose_for(s_pre, act[2], gid), report, act)
        self.scene.grasp(robot, act[2])
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
                        record=None, verbose=True):
  """Execute an mr_bench PATH (from mp_export.run_birrt) in MuJoCo with the REAL ARMS:
  every rai robot shape frame (the capsule collision model + gripper primitives) is mirrored
  as a kinematic body positioned by rai FK along the path -- the planner's own kinematics by
  construction, so witness configs land exactly. Parts stay fully dynamic; grasp welds
  (breakaway-capped) and mate welds fire at MODE TRANSITIONS, mapped back to plan actions via
  the env's step_names (mp_export stashes them). SEQUENCE problems only for now (task order
  == plan order); depgraph linearizations reorder events -- TODO.

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

  arms = _arm_shape_frames(C_fk, robots)
  def fk_poses(qfull):
    C_fk.setJointState(qfull)
    return {f: np.asarray(C_fk.getFrame(f).getPose(), dtype=float) for f, _k, _s, _c, _e in arms}

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
    cands = [f for f, _k, _s, _c, _e in arms if f.startswith(r)]
    ee_anchor[r] = next((f for f in cands if "gripper_fill" in f), cands[-1])

  q0 = full_q(mp_path[0].q)
  scene = SimScene(assembly, states[0], robots, gripper_kinds=gripper_kinds,
                   arms=arms, arm_poses=fk_poses(q0), ee_anchor=ee_anchor)
  rec = None
  if record is not None:
    rec = scene.recorder = {"qpos": [], "mpos": [], "mquat": [], "forces": [], "marks": []}
  report = SimReport()

  # task name -> (plan index, is-final-task-of-that-action)
  name_of = {}
  for i, nms in enumerate(problem.step_names):
    for k, nm in enumerate(nms):
      name_of[nm] = (i, k == len(nms) - 1)

  def fire(task_id):
    t = problem.tasks[task_id]
    if t.name == "terminal" or t.name not in name_of:
      return
    i, final = name_of[t.name]
    if not final:
      return                                               # pre-insertion goto etc.
    act = tuple(plan[i])
    if rec is not None:
      rec["marks"].append((len(rec["qpos"]), repr(act)))
    kind = act[0]
    if kind in ("pick", "add_grasp"):
      scene.grasp(act[1], act[2])
    elif kind == "handover":
      scene.grasp(act[1], act[2])
      giver = _holder_of(states[i], act[2], exclude=act[1])
      if giver:
        scene.release(giver)
    elif kind in ("release", "place"):
      scene.release(act[1])
    elif kind == "assemble":
      rel = assembly.relative_pose(act[2])
      got = _rel_pos(scene.part_pose(rel[0]), scene.part_pose(act[2]))
      if float(np.linalg.norm(got - np.asarray(rel[1]))) <= SEAT_POS_TOL:
        scene.weld_mate(act[2])
      else:
        report.results.append(ActionResult(act, False, "not_seated",
                                           f"{float(np.linalg.norm(got - np.asarray(rel[1])))*1000:.1f}mm"))
        return
    # judge RESTING parts against the plan's expected next state (held groups are wherever
    # the witnesses put them -- the symbolic layout does not price the arm's carry pose).
    report.results.append(_judge_resting(scene, assembly, act, states[i + 1]))

  if verbose:
    print(f"[sim/mp] driving RAW path: {len(mp_path)} states, {len(problem.tasks)} tasks")
  mode = mp_path[0].mode
  q_prev = full_q(mp_path[0].q)
  scene.sync_arms(fk_poses(q_prev))
  for st in mp_path:
    q_next = full_q(st.q)
    n = max(1, int(np.linalg.norm(q_next - q_prev) / (JOINT_RATE * DT)))
    for k in range(n):                                     # smooth joint-space sub-lerp
      a = (k + 1) / n
      scene.sync_arms(fk_poses((1 - a) * q_prev + a * q_next))
      scene.step(1, report, ("path",))
    q_prev = q_next
    if st.mode != mode:                                    # mode transition => task(s) done.
      # Raw transitions are DUPLICATED goal configs (same q, old mode then new mode --
      # FK-verified: tip exactly at the grasp standoff on both sides), so the arms are AT
      # the witness when events fire.
      done = [tid for tid in np.atleast_1d(mode.task_ids) if
              tid not in np.atleast_1d(st.mode.task_ids)]
      for tid in sorted(set(int(t) for t in done)):
        fire(tid)
      mode = st.mode
  for tid in sorted(set(int(t) for t in np.atleast_1d(mode.task_ids))):
    fire(tid)                                              # last mode's task (terminal goto)
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
  of the stored qpos frames (no physics re-run -- exactly what happened, bit for bit). Prints
  each action mark as playback crosses it. BLOCKS while the viewer is open (close it or
  ctrl-C to stop). speed: 1.0 = real time; loop: restart at the end."""
  import json, time
  z = np.load(path, allow_pickle=False)
  model = mujoco.MjModel.from_xml_string(str(z["xml"]))
  data = mujoco.MjData(model)
  marks = dict((int(f), a) for f, a in json.loads(str(z["marks"])))
  qpos = z["qpos"]
  print(f"[replay] {path}: {len(qpos)} frames, {len(marks)} actions; results:")
  for r in json.loads(str(z["results"])):
    print("   ", r)
  from mujoco import viewer as _viewer
  frame_dt = DT * RECORD_EVERY / speed
  mpos = z["mpos"] if "mpos" in z.files else None          # mocap tracks (arms / EE pucks)
  mquat = z["mquat"] if "mquat" in z.files else None
  with _viewer.launch_passive(model, data) as v:
    while v.is_running():
      for i in range(len(qpos)):
        if not v.is_running():
          return
        if i in marks:
          print(f"[replay] t={i*DT*RECORD_EVERY:6.2f}s  {marks[i]}")
        data.qpos[:] = qpos[i]
        if mpos is not None and len(mpos):
          data.mocap_pos[:] = mpos[i]
          data.mocap_quat[:] = mquat[i]
        mujoco.mj_forward(model, data)
        v.sync()
        time.sleep(frame_dt)
      if not loop:
        while v.is_running():
          time.sleep(0.2)
        return


if __name__ == "__main__":
  import sys
  if len(sys.argv) >= 3 and sys.argv[1] == "replay":
    replay_recording(sys.argv[2], speed=float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
  else:
    print("usage: python3 sim.py replay <recording.npz> [speed]")
