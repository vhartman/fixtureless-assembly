"""Static-equilibrium (wrench) feasibility -- the swappable stability model.

A subassembly is STABLE iff its contacts can supply a set of wrenches, each within its
own capability, that balances the external load (gravity + any insertion reaction). That
is a linear-program feasibility test:

    find  lambda >= 0  (<= per-generator bounds)
    s.t.  sum_i  G_i @ lambda_i  =  -w_external          (6 equations: force + torque)

where each contact contributes generator wrenches G_i (columns of a 6xk matrix, each a
unit force at the contact point or a unit couple, expressed about a common reference).
Feasible => the contacts can hold the load => supported.

THE ABSTRACTION (swappable): every contact TYPE is just a different capability set, built
by `capability(kind, ...)`. Rigid weld is the special case whose set is the whole 6-D
wrench space; unilateral table support, bounded friction/suction grasps, etc. are proper
subsets. Swap a type's builder (or the numbers in `PARAMS`) to change the physics; the LP
and every call site stay the same.

Conventions: wrenches are 6-vectors [fx,fy,fz, tx,ty,tz] taken about the WORLD ORIGIN; a
force f applied at world point p is the wrench [f, p x f]. Quasi-static, gravity is -z.
"""

import os
import time

import numpy as np
from collections import OrderedDict
from itertools import combinations
from scipy.optimize import linprog

# Lightweight benchmark instrumentation.  The planner normally calls HiGHS through either
# scipy.optimize.linprog or the persistent highspy model below; timing only linprog therefore
# under-reports the force-feasibility work on machines with highspy installed.  Keep the
# accounting here, immediately around the actual optimizer call, and let callers mark reverse
# heuristic construction with ``wrench_timing_phase("backward")``.  It is deliberately dormant
# unless ASSEMBLY_BENCH_PROFILE is set, so ordinary planning does not pay for perf_counter calls.
_WRENCH_TIMING_ENABLED = os.environ.get("ASSEMBLY_BENCH_PROFILE", "").strip().lower() not in (
    "", "0", "false", "no")
_WRENCH_TIMING_PHASE = "forward"
_WRENCH_TIMING = {
    "backward_calls": 0, "backward_s": 0.0,
    "forward_calls": 0, "forward_s": 0.0,
}


class wrench_timing_phase:
  """Temporarily attribute equilibrium optimizer calls to a planner phase."""

  def __init__(self, phase):
    self.phase = phase
    self.previous = None

  def __enter__(self):
    global _WRENCH_TIMING_PHASE
    self.previous = _WRENCH_TIMING_PHASE
    _WRENCH_TIMING_PHASE = self.phase
    return self

  def __exit__(self, _typ, _value, _traceback):
    global _WRENCH_TIMING_PHASE
    _WRENCH_TIMING_PHASE = self.previous


def _record_wrench_optimizer(dt):
  if not _WRENCH_TIMING_ENABLED:
    return
  phase = "backward" if _WRENCH_TIMING_PHASE == "backward" else "forward"
  _WRENCH_TIMING[f"{phase}_calls"] += 1
  _WRENCH_TIMING[f"{phase}_s"] += float(dt)


def reset_wrench_timing():
  for key in tuple(_WRENCH_TIMING):
    _WRENCH_TIMING[key] = 0.0 if key.endswith("_s") else 0


def wrench_timing_snapshot():
  return dict(_WRENCH_TIMING)


def current_wrench_timing_phase():
  """Expose the current attribution label to other benchmarked preprocessing stages."""
  return _WRENCH_TIMING_PHASE


def profile_wrench_phase(phase):
  """Decorator used by backward-search constructors and lazy reverse expansions."""
  def decorate(fn):
    def wrapped(*args, **kwargs):
      with wrench_timing_phase(phase):
        return fn(*args, **kwargs)
    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    wrapped.__doc__ = getattr(fn, "__doc__")
    return wrapped
  return decorate

try:
  import highspy                                            # optional: warm-startable LP backend
  _HS_OK = True
  _HS_INF = highspy.kHighsInf
  _HS_OPTIMAL = highspy.HighsModelStatus.kOptimal
except Exception:
  _HS_OK = False

GRAVITY = 9.81                      # m/s^2, along -z


# ---------------------------------------------------------------------------------------
# Tunable physics -- a plain registry so every number is swappable in one place. Contact
# capabilities and the insertion force are read from here; nothing hard-codes them.
# ---------------------------------------------------------------------------------------
PARAMS = {
    # which capability MODEL each contact uses -- swap here (the LP is model-agnostic, so a
    # new model is just a new cap_* builder + an entry below). The no-wrench path is
    # unaffected either way (topological support; these are only read when wrench is on).
    "grasp_model": "isotropic",     # "isotropic" (original box) | "directional" | "asymmetric"
    "table_model": "unilateral",    # wrench table model (no-wrench table stays topological)
    # FRICTION table model (--table-friction MU sets table_model to "friction"): the friction
    # cone coefficient. The unilateral default is frictionless, so a table-rest target cannot
    # resist ANY lateral load (every --insertion-noise corner adds one).
    "table_mu": 0.3,
    # ENGAGEMENT-scoped static friction (--vacuum-engage-friction MU): the frictionless
    # unilateral table rejects ANY lateral component of the cup's transient sealing push --
    # even a ~2deg-tilted top-down approach. This gives the TABLE a static friction cone ONLY
    # inside that transient check; every other gate keeps table_model/table_mu.
    "vacuum_engage_friction": 0.0,

    "grasp": {
        # ISOTROPIC (original): a bounded axis-aligned wrench box -- push/pull up to F_max on
        # any (world) axis, couple up to T_max. Direction-agnostic.
        "isotropic": {"two_finger": {"F_max": 40.0, "T_max": 4.0},
                      "vacuum":     {"F_max": 15.0, "T_max": 1.0}},
        # DIRECTIONAL: a box in the GRASP's own frame -- |axial|<=F_axial, |lateral|<=mu*F_axial,
        # couple <=T_axis about the approach axis / <=T_lat about the lateral axes. two_finger =
        # strong grip but WEAK TORQUE; vacuum = decent suction but WEAK LATERAL/SHEAR (small mu).
        "directional": {"two_finger": {"F_axial": 40.0, "mu": 0.6, "T_axis": 0.4, "T_lat": 0.4},
                        "vacuum":     {"F_axial": 25.0, "mu": 0.15, "T_axis": 0.8, "T_lat": 0.8}},
        # ASYMMETRIC: directional + PUSH/PULL asymmetry along the approach axis. Pressing the
        # part INTO the gripper is a STRUCTURAL path (palm stop / cup housing) -- F_push is
        # large (really arm-limited); pulling it OFF is the suction / finger-friction path --
        # F_pull is the weak number. Lateral <= mu * F_pull (friction at the grip preload).
        # => an insertion reaction is "free" when it COMPRESSES the grasp: emergent grasp-side
        # selection (e.g. brace the assembly from BELOW when a part is pressed on from above).
        "asymmetric": {"two_finger": {"F_push": 100.0, "F_pull": 25.0, "mu": 0.6,
                                      "T_axis": 0.4, "T_lat": 0.4, "T_close": 0.4},
                       "vacuum":     {"F_push": 100.0, "F_pull": 25.0, "mu": 0.15,
                                      "T_axis": 0.8, "T_lat": 0.8}},
    },
    # CURVED-surface torque derate: a grip on a cylinder side / sphere (a "curved" grasp,
    # per part_grasp_surfaces) resists far less TORQUE than a flat hold -- the part rolls/
    # pivots in the jaws or cup. None/1.0 => off (byte-identical); e.g. 0.25 => a curved
    # grasp's COUPLE capability is quartered (force caps unchanged). This is what makes
    # holding a car by a WHEEL (a curved grip) fail under an off-centre insertion torque
    # while a flat body-grasp survives.
    "curved_grasp_torque_scale": None,
    # INSERTION UNCERTAINTY (robust seating): None => the mate reaction is the clean nominal
    # force along the mate axis (byte-identical). Else a dict {"lateral":a, "offset":r}: real
    # seating is misaligned, modelled as an uncertainty BOX transverse to the mate axis -- an
    # angular tilt (shear up to a*insert_force) and a positional shift (axial force off-centre
    # up to r metres, a couple). The state must be stable at EVERY CORNER of that box; since
    # the balanceable-wrench set is convex, checking corners is EXACT (no sampling). a>0 alone
    # => a 4-corner box; a,r>0 => 16 corners. Adds an unavoidable shear+torque no favourable
    # lever can cancel, so a weak grasp holding far from the mate fails -- emergently.
    "insertion_uncertainty": None,
    # how the insertion-uncertainty box is discharged into a stability check:
    #   "corners" (default, EXACT): every box corner must be balanceable, contacts free to
    #             rechoose their force split per realization (fully-adjustable robustness);
    #   "affine": one fixed AFFINE force-response policy must cover the LINEARIZED box
    #             -- an open-loop / fixed-compliance grasp that cannot sense the seating error and
    #             re-optimize. A single LP (vs 2^k corner solves); stricter per axis, but sound
    #             only without a positional offset (see equilibrium_robust_affine; the planner
    #             falls back to 'corners' when both box axes are nonzero). Ignored when
    #             insertion_uncertainty is None.
    "robust_mode": "corners",
    # ORIENTATION-ROBUST GRAVITY: for every freely held group, require the same contacts to
    # balance gravity for every possible rigid-body orientation. This is a continuous robust
    # zonotope-containment test, not an orientation sample grid. Table-resting groups retain
    # their authored discrete rest orientation. Opt-in while its deliberate conservatism is
    # evaluated on the benchmark suite.
    "orientation_robust_gravity": False,
    # GRASP COMPLIANCE (--grasp-compliance): per-kind multiplier on the INSERTION-UNCERTAINTY
    # box when that kind holds a body in the press's load path. A suction cup is a COMPLIANT
    # contact -- it deflects under contact loads, so the same commanded seating wanders more:
    # picking/carrying (gravity only, no accuracy demand) is unaffected, but assembling with
    # a vacuum-held target/inserter must survive a LARGER misalignment box. None = off
    # (byte-identical); needs insertion_uncertainty to have any effect (it scales the box).
    # Avoidance of imprecise-holder assembly is then EMERGENT, not a hardcoded prohibition.
    "grasp_compliance": None,                           # e.g. {"two_finger": 1.0, "vacuum": 3.5}
    # VACUUM ENGAGEMENT: a real cup is pushed onto the surface before the seal/contact exists.
    # This is a TRANSIENT point load along the grasp approach axis, balanced by the table or
    # the group's PRE-EXISTING holders. problem.py gates pick/add_grasp/handover with it before
    # adding the new contact. It is intentionally not part of grasp_capability: after sealing,
    # suction and cup compression are an internal preload and only their NET holding envelope
    # belongs in the equilibrium LP. Zero preserves old saved-plan successor sets.
    "vacuum_engage_force": 0.0,                         # N; --vacuum-engage-force
    # VACUUM SEAL: the cup can only grip a FLAT face at least as large as its seal -- a face
    # smaller than the cup leaves the seal partially open (no vacuum, no grip). The grasp
    # generator drops vacuum candidates whose contacted face's smallest extent is below this
    # diameter (metres). Cube faces (50mm) pass; the chair's 20mm cylinder caps do not.
    # The force/torque numbers above assume a FULL seal, so this gate is what keeps them
    # honest on small parts.
    "vacuum_seal_diameter": 0.025,
    # insertion (mating) reaction: ONE magnitude for now, but read per-mate from a dict so
    # individual mates can override it later without touching call sites. `_default` applies
    # to any mate not listed.
    "insert_force": {"_default": 10.0},                 # newtons, along the mate axis
    # check that the INSERTING arm's own grasp can transmit the press (the seating reaction
    # also acts on the held part). False = skip the check, equivalent to giving the inserter
    # grasp unbounded capability ("a very high value") -- the target-side check stays on.
    "check_inserter": True,
}


# Named hardware envelopes are deliberately orthogonal to the grasp-model family.  They are
# not empirical slip calibrations: these numbers are the gripper manufacturer's absolute
# structural limits, before the integrator's safety factor and before object/pad friction.
# Keeping this separate from main.py's search profiles lets one run the same search algorithm
# with either the historical provisional caps or a declared hardware ceiling.
WRENCH_PROFILES = {
    "max_robotiq": {
        # Robotiq 2F-85 manual section 6.2.4.  The model's grasp-frame columns are
        # [t1, t2, n] == gripper-local [x, y, z]; z is the approach axis.  Mx/My are specified
        # at the fingertip base.  Our current box represents the corresponding free-couple
        # components at the grasp datum; coupling the base moment to contact-force lever arms
        # is intentionally left to the later measured calibration.
        "isotropic": {"F_max": 50.0, "T_max": 3.0},  # conservative: one scalar torque cap
        "directional": {"F_axial": 50.0, "mu": 1.0,
                        "T_axis": 3.0, "T_lat": 5.0},
        "asymmetric": {"F_push": 50.0, "F_pull": 50.0, "mu": 1.0,
                       "T_axis": 3.0, "T_lat": 5.0, "T_close": 5.0},
    },
}


def apply_wrench_profile(name):
  """Install a named two-finger hardware envelope into every grasp-model family.

  Scaling/axis-specific CLI overrides are intentionally applied by callers *after* this
  function, giving the usual explicit-flag > profile precedence.  ``None`` is a no-op so old
  plans retain the historical provisional caps.
  """
  if name is None:
    return
  if name not in WRENCH_PROFILES:
    raise ValueError(f"unknown wrench profile {name!r}; choose from {sorted(WRENCH_PROFILES)}")
  for model, caps in WRENCH_PROFILES[name].items():
    PARAMS["grasp"][model]["two_finger"].update(caps)


def insert_force(part):
  """Insertion-force magnitude for mating `part` (per-mate dict, falling back to default)."""
  d = PARAMS["insert_force"]
  return d.get(part, d["_default"])


# ---------------------------------------------------------------------------------------
# Contact capabilities: each returns (G [6xk] generators, ub [k] per-generator upper bounds).
# A suppliable wrench is { G @ lambda : 0 <= lambda <= ub }. ub = inf means unbounded. Every
# model here is a bounded polytope of this form; models are selected via PARAMS + the
# grasp_capability / table_capability dispatchers (the LP itself is model-agnostic).
# ---------------------------------------------------------------------------------------
def _cross3(a, b):
  # explicit 3-vector cross product. np.cross is astonishingly slow for single 3-vectors
  # (axis-normalization / moveaxis overhead dwarfs the arithmetic) and this is the hottest
  # line in a wrench search -- _force_gen alone calls it hundreds of thousands of times.
  return np.array([a[1] * b[2] - a[2] * b[1],
                   a[2] * b[0] - a[0] * b[2],
                   a[0] * b[1] - a[1] * b[0]])


def _force_gen(direction, point):
  # generator wrench of a unit force along `direction` applied at world `point`.
  d = np.asarray(direction, float)
  p = np.asarray(point, float)
  return np.concatenate([d, _cross3(p, d)])


def _couple_gen(axis):
  # generator wrench of a unit pure couple about `axis`.
  return np.concatenate([np.zeros(3), np.asarray(axis, float)])


_AXES = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]


def cap_rigid_weld(point):
  # RIGID WELD -- the whole 6-D wrench space: any force at the interface + any couple. The
  # special case the others specialise. Point is the mate interface centre.
  G = [_force_gen(s * a, point) for a in _AXES for s in (+1.0, -1.0)]
  G += [_couple_gen(s * a) for a in _AXES for s in (+1.0, -1.0)]
  return np.array(G).T, np.full(len(G), np.inf)


def cap_mate_box(point, rotation, force, torque):
  """Finite bilateral mate wrench box expressed in the mate parent's local frame.

  ``force=[Fx,Fy,Fz]`` and ``torque=[Mx,My,Mz]`` are independent component bounds.  The
  returned columns are the wrench *on the child* about the caller's common reference point;
  a multi-body equilibrium matrix installs their negatives on the parent.  Component limits
  keep the model polyhedral and therefore preserve the planner's LP formulation exactly.
  """
  R = np.asarray(rotation, dtype=float)
  fcap = np.asarray(force, dtype=float).ravel()
  tcap = np.asarray(torque, dtype=float).ravel()
  if R.shape != (3, 3) or fcap.shape != (3,) or tcap.shape != (3,):
    raise ValueError("mate box needs rotation [3,3] and force/torque [3]")
  if (not np.all(np.isfinite(R)) or not np.all(np.isfinite(fcap))
      or not np.all(np.isfinite(tcap)) or np.any(fcap < 0.0) or np.any(tcap < 0.0)):
    raise ValueError("mate box values must be finite and non-negative")
  G, ub = [], []
  for i in range(3):
    for s in (+1.0, -1.0):
      G.append(_force_gen(s * R[:, i], point)); ub.append(float(fcap[i]))
  for i in range(3):
    for s in (+1.0, -1.0):
      G.append(_couple_gen(s * R[:, i])); ub.append(float(tcap[i]))
  return np.asarray(G, dtype=float).T, np.asarray(ub, dtype=float)


def cap_slot(point, press_axis, depth=0.0, F_wall=np.inf):
  # PARTIAL (staged) mate -- a SLOT/pocket. Default (depth=0) is the v1 POINT-CONTACT model:
  # the walls constrain the part LATERALLY (bilateral, both walls), the seat bottom pushes
  # BACK against the press (unilateral along -press_axis: it resists seating deeper, never
  # holds against extraction), and NO couples -- a cantilevered part tips out under its own
  # gravity moment. This is what makes "pushed-through-but-unpinned" (A11) expressible; the
  # fastener upgrades the mate to cap_rigid_weld (the group becomes one rigid body again).
  # DEEP-SLOT extension (dormant, not wired to any flag yet): depth>0 duplicates the wall
  # forces at the slot MOUTH (point - depth*axis), giving couple capability ~ depth*F_wall
  # about the lateral axes -- pass a FINITE F_wall with it, else any depth>0 resists every
  # moment (rigid unbounded walls) and the staging phenomenon disappears.
  d = np.asarray(press_axis, float)
  d = d / (np.linalg.norm(d) + 1e-12)
  ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
  t1 = _cross3(ref, d); t1 /= np.linalg.norm(t1)
  t2 = _cross3(d, t1)
  pts = [np.asarray(point, float)]
  if depth > 0.0:
    pts.append(np.asarray(point, float) - depth * d)       # the mouth: second wall contact
  G = [_force_gen(s * t, p) for p in pts for t in (t1, t2) for s in (+1.0, -1.0)]
  ub = [F_wall] * len(G)
  G.append(_force_gen(-d, pts[0]))                         # seat pushes back, only
  ub.append(np.inf)
  return np.array(G).T, np.array(ub, float)


def cap_table_unilateral(points):
  # UNILATERAL table support: at each support point, a normal force that can only PUSH UP
  # (+z), never pull, and no tangential (frictionless) component. The set of such points is
  # the support polygon. Unbounded magnitude (the table takes any downward load).
  # NOTE the frictionless choice means a table-rest target rejects ANY lateral load --
  # including the 1e-3 N shear a --insertion-noise box adds. The friction model exists
  # for exactly that.
  n = np.array([0.0, 0.0, 1.0])
  G = [_force_gen(n, p) for p in points]
  return np.array(G).T, np.full(len(G), np.inf)


def cap_table_friction(points, mu):
  # FRICTION table support: per point a LINEARIZED FRICTION PYRAMID (the standard cone
  # polyhedron): generators n + mu*e for e in +-x, +-y. A nonneg combination of the four
  # edges yields f_normal = sum(lambda) and |f_tangential| <= mu * f_normal -- a real table
  # takes lateral loads through friction (mu ~ 0.3), so a table-rest target CAN resist the
  # shear an --insertion-noise box adds. Unbounded magnitude; the unilateral model is the
  # mu -> 0 limit.
  n = np.array([0.0, 0.0, 1.0])
  G, ub = [], []
  for p in points:
    for e in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
      G.append(_force_gen(n + mu * np.array([e[0], e[1], 0.0]), p))
      ub.append(np.inf)
  return np.array(G).T, np.array(ub, float)


def cap_grasp_box(point, F_max, T_max):
  # ISOTROPIC grasp (the ORIGINAL model): push/pull up to F_max along any (world) axis and a
  # couple up to T_max, at the grasp point -- an axis-aligned wrench box. Direction-agnostic.
  G, ub = [], []
  for a in _AXES:
    for s in (+1.0, -1.0):
      G.append(_force_gen(s * a, point)); ub.append(F_max)
  for a in _AXES:
    for s in (+1.0, -1.0):
      G.append(_couple_gen(s * a)); ub.append(T_max)
  return np.array(G).T, np.array(ub, float)


def cap_friction_grasp(point, n, t1, t2, F_axial, mu, T_axis, T_lat):
  # DIRECTIONAL grasp: a wrench box in the GRASP's own frame (approach axis n, laterals t1,t2).
  #   FORCE : |axial| <= F_axial ; |lateral| <= mu*F_axial. The lateral bound is the friction
  #           at the grip/suction PRELOAD (holds even at zero external axial, unlike the
  #           table's unilateral cone). Small mu => weak shear (suction: holds from the top,
  #           slips from the side).
  #   MOMENT: |couple about n| <= T_axis ; |couple about t1,t2| <= T_lat. Small T_lat => weak
  #           torque (a two-finger grip lets the held part pivot in the jaws).
  n = np.asarray(n, float); t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)
  G, ub = [], []
  for vec, bound in ((n, F_axial), (t1, mu * F_axial), (t2, mu * F_axial)):
    for s in (+1.0, -1.0):
      G.append(_force_gen(s * vec, point)); ub.append(bound)
  for axis, T in ((n, T_axis), (t1, T_lat), (t2, T_lat)):
    for s in (+1.0, -1.0):
      G.append(_couple_gen(s * axis)); ub.append(T)
  return np.array(G).T, np.array(ub, float)


def cap_asymmetric_grasp(point, n, t1, t2, F_push, F_pull, mu, T_axis, T_lat, T_close=None):
  # DIRECTIONAL + PUSH/PULL asymmetry. Grasp local z (`n`) points from the gripper INTO the
  # part, so a contact force ON the part
  #   along +n  = the palm / cup housing PUSHING the part -- a structural stop, not a
  #               friction path -> F_push (large; really arm-limited),
  #   along -n  = holding the part against pull-OFF -- the suction / finger-friction path
  #               -> F_pull (the weak number).
  # Lateral <= mu * F_pull (friction at the grip/suction preload); couples as directional.
  # Consequence: a load is cheap when it COMPRESSES the grasp -- e.g. a carrier holding the
  # assembly from BELOW takes an insertion press from above through the strong +n path,
  # while a top grasp must carry it on suction/friction.
  n = np.asarray(n, float); t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)
  G, ub = [], []
  G.append(_force_gen(n, point)); ub.append(F_push)      # push (compression) -- strong
  G.append(_force_gen(-n, point)); ub.append(F_pull)     # pull-off -- weak
  for vec in (t1, t2):
    for s in (+1.0, -1.0):
      G.append(_force_gen(s * vec, point)); ub.append(mu * F_pull)
  # torque caps per axis. t2 is the CLOSING axis (the line joining the two finger contacts):
  # the contacts are collinear on it, so they form NO couple to resist rotation ABOUT it --
  # only pad spin-friction does, so a two-finger's out-of-hand torque there is genuinely
  # weak. T_close (default = T_lat) makes that separately tunable; the vacuum is roll-
  # symmetric so it leaves T_close == T_lat.
  T_close = T_lat if T_close is None else T_close
  for axis, T in ((n, T_axis), (t1, T_lat), (t2, T_close)):
    for s in (+1.0, -1.0):
      G.append(_couple_gen(s * axis)); ub.append(T)
  return np.array(G).T, np.array(ub, float)


# --- model selection: dispatch a contact to its configured capability builder --------------
def grasp_capability(point, n, t1, t2, gripper_kind, surface="flat"):
  # build the grasp cap per PARAMS['grasp_model']. `point` is the grasp/contact point; n,t1,t2
  # the grasp frame (used only by the directional/asymmetric models; isotropic ignores them).
  # `surface` ("flat"|"curved") lets a curved contact be derated -- see below.
  model = PARAMS["grasp_model"]
  p = PARAMS["grasp"][model][gripper_kind]
  if model == "isotropic":
    G, ub = cap_grasp_box(point, **p)
  elif model == "asymmetric":
    G, ub = cap_asymmetric_grasp(point, n, t1, t2, **p)
  else:
    G, ub = cap_friction_grasp(point, n, t1, t2, **p)
  scale = PARAMS.get("curved_grasp_torque_scale")
  if scale is not None and surface == "curved":
    # a curved grip resists less TORQUE: derate the COUPLE generators only -- the columns with
    # no linear-force part (built by _couple_gen). Model-agnostic: works for every builder
    # above. Force capabilities are left untouched. scale in [0,1].
    is_couple = np.all(np.abs(G[:3, :]) < 1e-12, axis=0)
    ub = ub.copy()
    ub[is_couple] = ub[is_couple] * float(scale)
  return G, ub


_TABLE_BUILDERS = {"unilateral": cap_table_unilateral, "friction": cap_table_friction}


def table_capability(points):
  model = PARAMS["table_model"]
  if model == "friction":
    return _TABLE_BUILDERS["friction"](points, float(PARAMS.get("table_mu", 0.3)))
  return _TABLE_BUILDERS[model](points)


def table_capability_override(points, model, mu):
  # SCOPED table model for ONE transient check (the vacuum engagement gate's static friction)
  # that must NOT touch the global PARAMS['table_model'] -- every other stability gate reads
  # that. model in {"unilateral", "friction"}; mu is the friction coefficient.
  if model == "friction":
    return cap_table_friction(points, float(mu))
  return cap_table_unilateral(points)


# ---------------------------------------------------------------------------------------
# External load and the equilibrium LP.
# ---------------------------------------------------------------------------------------
def gravity_wrench(masses_points):
  # total gravity wrench about the origin for [(mass, com_world), ...].
  # PARAMS["ignore_gravity"]: WEIGHTLESS mode -- drop the part-weight load entirely, so the
  # only external wrench a hold must resist is the INSERTION reaction (and other applied
  # loads). Isolates insertion/bracing feasibility from gravity (e.g. probe whether a grasp
  # fails on weight vs. on the press force). Fixed for a solve, so the LP cache stays valid.
  if PARAMS.get("ignore_gravity"):
    return np.zeros(6)
  w = np.zeros(6)
  for m, c in masses_points:
    f = np.array([0.0, 0.0, -m * GRAVITY])
    w += np.concatenate([f, _cross3(np.asarray(c, float), f)])
  return w


def point_force_wrench(force_vec, point):
  # wrench about the origin of a force applied at a world point (e.g. insertion reaction).
  f = np.asarray(force_vec, float)
  return np.concatenate([f, _cross3(np.asarray(point, float), f)])


def gravity_wrench_map(masses_points):
  """Return ``A`` such that ``A @ d`` is gravity for body-frame direction ``d``.

  ``masses_points`` uses the same reference point convention as :func:`gravity_wrench`.
  The direction is dimensionless and has ``||d|| <= 1``; ``GRAVITY`` is included in ``A``.
  Letting ``d`` range over the unit ball is exactly the convex hull of every possible gravity
  direction under a rigid-body orientation.  Since the contact wrench set is convex, containing
  this ball image is equivalent to supporting the group at every orientation.
  """
  A = np.zeros((6, 3), dtype=float)
  for mass, point in masses_points:
    m = float(mass)
    p = np.asarray(point, dtype=float)
    # cross(p, f) = [p]_x f
    px = np.array([[0.0, -p[2], p[1]],
                   [p[2], 0.0, -p[0]],
                   [-p[1], p[0], 0.0]])
    A[:3] += m * GRAVITY * np.eye(3)
    A[3:] += m * GRAVITY * px
  return A


_ZONOTOPE_NORMALS = OrderedDict()                         # rounded G -> (Q, facet normals, bytes)
_ZONOTOPE_NORMALS_BYTES = 0
# A facet set grows combinatorially with the number of distinct contact-generator directions.
# An entry-count LRU is therefore not a memory bound: 256 four-grasp entries can occupy many
# hundreds of MiB.  Bound the actual ndarray payload instead.  Oversize one-off entries remain
# usable for their caller but are deliberately not retained.
_ZONOTOPE_NORMALS_BYTE_CAP = 64 * 1024 * 1024


def _cache_zonotope_normals(key, Q, H):
  global _ZONOTOPE_NORMALS_BYTES
  nbytes = int(Q.nbytes + H.nbytes)
  if nbytes > _ZONOTOPE_NORMALS_BYTE_CAP:
    return Q, H
  while (_ZONOTOPE_NORMALS
         and _ZONOTOPE_NORMALS_BYTES + nbytes > _ZONOTOPE_NORMALS_BYTE_CAP):
    _, (_, _, old_bytes) = _ZONOTOPE_NORMALS.popitem(last=False)
    _ZONOTOPE_NORMALS_BYTES -= old_bytes
  _ZONOTOPE_NORMALS[key] = (Q, H, nbytes)
  _ZONOTOPE_NORMALS_BYTES += nbytes
  return Q, H


def _zonotope_facet_normals(G, ub, tol=1e-10):
  """Affine-hull basis and all facet-normal directions of ``G [0, ub]``.

  A bounded contact generator is a line segment, hence their Minkowski sum is a zonotope.
  In an ``r``-dimensional zonotope every facet normal is orthogonal to at least ``r-1``
  independent generator directions.  Enumerating those direction sets gives an exact finite
  halfspace certificate.  Collinear +/- wrench generators are collapsed for enumeration only;
  their distinct bounds remain in the support-function calculation.
  """
  G = np.asarray(G, dtype=float)
  ub = np.asarray(ub, dtype=float)
  if G.ndim != 2 or ub.shape != (G.shape[1],):
    raise ValueError("zonotope needs G[d,k] and ub[k]")
  if np.any(ub < 0.0) or not np.all(np.isfinite(ub)):
    raise ValueError("all-orientation gravity requires finite non-negative contact bounds")
  active = (ub > tol) & (np.linalg.norm(G, axis=0) > tol)
  Ga = G[:, active]
  key = (G.shape[0], Ga.shape[1], np.round(Ga, 10).tobytes())
  hit = _ZONOTOPE_NORMALS.get(key)
  if hit is not None:
    _ZONOTOPE_NORMALS.move_to_end(key)
    return hit[:2]
  if Ga.shape[1] == 0:
    ans = (np.zeros((G.shape[0], 0)), np.zeros((0, 0)))
    return _cache_zonotope_normals(key, *ans)

  U, s, _ = np.linalg.svd(Ga, full_matrices=True)
  rank_tol = tol * max(Ga.shape) * max(float(s[0]), 1.0)
  r = int(np.sum(s > rank_tol))
  Q = U[:, :r]                                           # orthonormal affine-hull basis
  Gr = Q.T @ Ga

  # One representative of every unoriented generator line. Facets depend on directions,
  # not segment lengths or on whether a paired column spells +v or -v.
  directions = []
  seen = set()
  for col in Gr.T:
    d = col / np.linalg.norm(col)
    nz = np.flatnonzero(np.abs(d) > tol)
    if nz.size and d[nz[0]] < 0.0:
      d = -d
    k = np.round(d, 10).tobytes()
    if k not in seen:
      seen.add(k)
      directions.append(d)

  normals = []
  nseen = set()

  def add_normal(h):
    h = np.asarray(h, dtype=float)
    h /= np.linalg.norm(h)
    nz = np.flatnonzero(np.abs(h) > tol)
    if nz.size and h[nz[0]] < 0.0:
      h = -h
    k = np.round(h, 9).tobytes()
    if k not in nseen:
      nseen.add(k)
      normals.append(h)

  if r == 1:
    add_normal(np.ones(1))
  else:
    for ids in combinations(range(len(directions)), r - 1):
      M = np.asarray([directions[i] for i in ids], dtype=float)
      _, ss, vh = np.linalg.svd(M, full_matrices=True)
      if int(np.sum(ss > tol * max(M.shape))) != r - 1:
        continue
      add_normal(vh[-1])

  H = np.asarray(normals, dtype=float)
  return _cache_zonotope_normals(key, Q, H)


def equilibrium_all_orientations(G, ub, w_non_gravity, gravity_map, tol=1e-7):
  """Can bounded contacts balance ``w_non_gravity + A d`` for every ``||d|| <= 1``?

  This is an exact continuous robust test, not an orientation grid.  For each facet normal
  ``h`` of the contact zonotope, the worst gravity direction is analytic:

      max_d h'(-w0 - A d - centre) = h'(-w0-centre) + ||A' h||_2.

  Checking that value against the zonotope support function at every facet is necessary and
  sufficient for containment. Contacts may redistribute load for each orientation, matching
  the fully-adjustable static-equilibrium semantics of the existing corner robustness test.
  """
  G = np.asarray(G, dtype=float)
  ub = np.asarray(ub, dtype=float)
  w0 = np.asarray(w_non_gravity, dtype=float)
  A = np.asarray(gravity_map, dtype=float)
  if w0.shape != (G.shape[0],) or A.shape != (G.shape[0], 3):
    raise ValueError("wrench centre must be [d] and gravity map [d,3]")
  Q, H = _zonotope_facet_normals(G, ub)
  if Q.shape[1] == 0:
    return bool(np.linalg.norm(w0) <= tol and np.linalg.norm(A) <= tol)

  # The contact set has no extent normal to span(G). Every point of the gravity ellipsoid
  # must therefore have zero component there, not merely its nominal centre.
  w_perp = w0 - Q @ (Q.T @ w0)
  A_perp = A - Q @ (Q.T @ A)
  if np.linalg.norm(w_perp) > tol or np.linalg.norm(A_perp, ord=2) > tol:
    return False

  Gr = Q.T @ G
  wr = Q.T @ w0
  Ar = Q.T @ A
  Z = 0.5 * Gr * ub[None, :]
  centre = np.sum(Z, axis=1)
  for h0 in H:
    for h in (h0, -h0):
      required = float(h @ (-wr - centre) + np.linalg.norm(Ar.T @ h))
      available = float(np.sum(np.abs(h @ Z)))
      if required > available + tol:
        return False
  return True


def equilibrium_feasible(caps, w_external, tol=1e-6):
  """Can the contacts (list of (G, ub)) supply wrenches summing to -w_external?

  Solves the LP feasibility  { G_all @ lambda = -w_external, 0 <= lambda <= ub }.
  Returns True iff a feasible multiplier vector exists. An empty contact set is stable
  only under (near-)zero external load."""
  if not caps:
    return bool(np.linalg.norm(w_external) <= tol)
  G = np.hstack([c[0] for c in caps])                   # 6 x K
  ub = np.concatenate([c[1] for c in caps])             # K
  return equilibrium_feasible_matrices(G, ub, w_external, tol)


_BASIS_CACHE = {}                                           # geo_key -> pool of warm bases (see below)
_BASIS_POOL = 8                                             # max distinct bases kept per geometry
_HS_MODELS = OrderedDict()                                  # geo_key -> [Highs, K]  (LRU warm models)
_HS_CAP = 1024                                              # max persistent models kept alive
_WARM_ON = True                                             # master switch (diagnostics/ablation)
_WARM = [0, 0, 0]                                           # [warm/hs solves, scipy lp, extractions]


def _hs_model(geo_key, G, ub):
  # a persistent HiGHS model per geometry: 0 objective, K vars in [0,ub], D equality rows Glam=b
  # whose RHS we overwrite per load. Keeping the model alive lets HiGHS DUAL-SIMPLEX WARM-START
  # from the previous load's basis -- an RHS-only change re-solves in ~0.04ms vs ~0.85ms cold.
  m = _HS_MODELS.get(geo_key)
  if m is not None:
    _HS_MODELS.move_to_end(geo_key)
    return m
  h = highspy.Highs()
  h.setOptionValue("output_flag", False)
  K = G.shape[1]
  upper = np.where(np.isfinite(ub), ub, _HS_INF).astype(float)
  empty_i = np.array([], dtype=np.int32)
  h.addCols(K, np.zeros(K), np.zeros(K), upper, 0, empty_i, empty_i, np.array([]))
  idx = np.arange(K, dtype=np.int32)
  for i in range(G.shape[0]):
    h.addRow(0.0, 0.0, K, idx, np.ascontiguousarray(G[i, :], dtype=float))
  _HS_MODELS[geo_key] = m = [h, K]
  if len(_HS_MODELS) > _HS_CAP:
    _HS_MODELS.popitem(last=False)                          # evict least-recently-used geometry
  return m


def _hs_feasible(geo_key, G, ub, beq):
  h, _ = _hs_model(geo_key, G, ub)
  for i in range(G.shape[0]):
    h.changeRowBounds(i, float(beq[i]), float(beq[i]))
  t0 = time.perf_counter() if _WRENCH_TIMING_ENABLED else None
  h.run()
  if t0 is not None:
    _record_wrench_optimizer(time.perf_counter() - t0)
  return h.getModelStatus() == _HS_OPTIMAL                  # 0-objective LP: optimal <=> feasible


def _extract_basis(G, ub, x, tol=1e-7):
  # Turn a HiGHS basic-feasible vertex `x` into a reusable warm basis. A feasibility BFS of a
  # D-row system has (generically) exactly D variables strictly between their bounds -- the
  # BASIC columns, an invertible DxD -- with the rest pinned at 0 or ub (NONBASIC). We store the
  # basic columns' inverse and the constant wrench the pinned nonbasics contribute, so a later
  # load w' on the SAME geometry is tested by one back-substitution instead of a fresh LP.
  ubv = np.where(np.isfinite(ub), ub, np.inf)
  basic = np.where((x > tol) & (x < ubv - tol))[0]
  if basic.size != G.shape[0]:
    return None                                             # degenerate/non-vertex -> don't warm
  B = G[:, basic]
  try:
    Binv = np.linalg.inv(B)
  except np.linalg.LinAlgError:
    return None
  if not np.all(np.isfinite(Binv)) or np.linalg.cond(B) > 1e10:
    return None
  mask = np.ones(G.shape[1], dtype=bool); mask[basic] = False
  nb = np.where(mask)[0]
  nb_vals = np.where(x[nb] > tol, ub[nb], 0.0)              # snap each nonbasic to its bound
  const = G[:, nb] @ nb_vals                                # constant wrench from pinned nonbasics
  return (basic, Binv, ub[basic], const)


def equilibrium_solution(G, ub, w_external, bounds=None):
  """Like equilibrium_feasible_matrices but returns (feasible, x) -- the actual generator
  multipliers of one balancing solution. VISUALIZATION-ONLY cold path (no caches, no
  warm starts): the per-contact forces drawn by --show-wrenches come from x."""
  if G.shape[1] == 0:
    return bool(np.linalg.norm(w_external) < 1e-9), np.zeros(0)
  if bounds is None:
    bounds = [(0.0, None if not np.isfinite(u) else float(u)) for u in ub]
  t0 = time.perf_counter() if _WRENCH_TIMING_ENABLED else None
  res = linprog(c=np.zeros(G.shape[1]), A_eq=G, b_eq=-np.asarray(w_external, dtype=float),
                bounds=bounds, method="highs")
  if t0 is not None:
    _record_wrench_optimizer(time.perf_counter() - t0)
  return bool(res.status == 0), (res.x if res.status == 0 else np.zeros(G.shape[1]))


def equilibrium_feasible_matrices(G, ub, w_external, tol=1e-6, bounds=None, geo_key=None):
  """The LP feasibility test on the already-assembled generator matrix (G [DxK], ub [K]).
  The contact-INDEPENDENT parts (G, ub, and the derived `bounds` list) can be built ONCE and
  reused across many external loads -- e.g. the corners of an insertion-uncertainty box --
  so the caller (`_group_stable`/`_grounding_caps`) hoists them out of the per-load loop.

  `geo_key` (a hashable geometry signature) turns on same-geometry WARM-START, since ~6 distinct
  loads share each geometry and only the RHS changes between them:
    * with `highspy` installed we keep a persistent HiGHS model per geometry and only overwrite
      its RHS, so HiGHS dual-simplex warm-starts from the previous basis (~0.04ms vs ~0.85ms);
    * without it we fall back to scipy + a per-geometry numpy warm-basis POOL -- cache each
      feasible vertex's basis and first try it as a cheap ACCEPT test (a 6x6 back-substitution
      yielding an explicit in-box multiplier -> a SOUND feasibility witness, no LP); the scipy LP
      runs only on a miss or a genuinely infeasible load.
  Both stay exact (the LP is always the authority on a warm miss)."""
  if G.shape[1] == 0:
    return bool(np.linalg.norm(w_external) <= tol)
  beq = -np.asarray(w_external, float)
  if geo_key is not None and _WARM_ON and _HS_OK:
    _WARM[0] += 1
    return _hs_feasible(geo_key, G, ub, beq)
  if geo_key is not None and _WARM_ON:
    pool = _BASIS_CACHE.get(geo_key)
    if pool:
      # a geometry accumulates loads from several mates, each with its own optimal basis, so we
      # keep a small POOL and try each (still microseconds vs a 1.1ms LP). Move a hit to the
      # front so the working set stays hot.
      for i, (basic, Binv, ubB, const) in enumerate(pool):
        lamB = Binv @ (beq - const)                         # basics that make Glam = beq exactly
        if np.all(lamB >= -tol) and np.all(lamB <= ubB + tol):
          if i:
            pool.insert(0, pool.pop(i))
          _WARM[0] += 1
          return True                                       # explicit feasible point -> accept
  if bounds is None:
    bounds = [(0.0, None if not np.isfinite(u) else float(u)) for u in ub]
  t0 = time.perf_counter() if _WRENCH_TIMING_ENABLED else None
  res = linprog(c=np.zeros(G.shape[1]), A_eq=G, b_eq=beq, bounds=bounds, method="highs")
  if t0 is not None:
    _record_wrench_optimizer(time.perf_counter() - t0)
  _WARM[1] += 1
  if res.success and geo_key is not None and _WARM_ON:
    nb = _extract_basis(G, np.asarray(ub, float), np.asarray(res.x, float))
    if nb is not None:
      pool = _BASIS_CACHE.setdefault(geo_key, [])
      bset = frozenset(nb[0].tolist())
      if all(bset != frozenset(b[0].tolist()) for b in pool):  # only keep DISTINCT bases
        pool.insert(0, nb)
        del pool[_BASIS_POOL:]                              # cap (LRU)
        _WARM[2] += 1
  return bool(res.success)


def equilibrium_robust_affine(G, ub, w0, deltas, bounds=None, tol=1e-6):
  """CONSERVATIVE robust equilibrium via an AFFINE decision rule (adjustable-robust counterpart).

  The exact corner test asks 'for every disturbance realization, does SOME in-box contact-force
  vector balance it?' -- the contacts may pick a different force distribution per realization.
  This asks the stricter question: is there ONE affine force-response policy

      lam(s) = lam0 + sum_j s_j * lam_j ,   s in [-1,1]^m

  that stays in [0,ub] AND balances w0 + sum_j s_j*delta_j for EVERY s in the box? Because an
  affine policy is one admissible way to respond, its feasibility implies the corners OF THE
  LINEARIZED BOX (w0 +- delta_j) are stable. SOUNDNESS LIMIT: when the box mixes force and
  point-shift generators, the TRUE corners additionally carry a bilinear cross couple
  (shift x tilt) that no linear policy can express -- this certificate is then not valid for
  the true box (problem._robust_stable falls back to the exact corner test in that case).
  One LP replaces the 2^m corner solves.

  Physical reading: lam0 is the nominal grasp/support load; lam_j is a fixed linear reaction gain
  to disturbance axis j -- an open-loop / fixed-compliance grasp that cannot sense the exact
  seating error and re-optimize, only respond linearly. delta_j are the generator wrenches
  (transverse force / off-centre couple) of the uncertainty box, about the same origin as w0."""
  K = G.shape[1]
  ub = np.asarray(ub, float)
  m = len(deltas)
  if K == 0:                                                # no contacts: only zero wrench holds
    return np.linalg.norm(w0) <= tol and all(np.linalg.norm(d) <= tol for d in deltas)
  n = K + 2 * K * m                                         # [lam0 | lam_1..lam_m | t_1..t_m]
  I = np.eye(K)

  def col(block):                                           # column slice of a K-block
    return slice(block * K, block * K + K)

  A_eq = np.zeros((6 * (1 + m), n)); b_eq = np.zeros(6 * (1 + m))
  A_eq[0:6, col(0)] = G; b_eq[0:6] = -np.asarray(w0, float)          # G lam0 = -w0
  for j in range(m):
    A_eq[6 * (j + 1):6 * (j + 2), col(1 + j)] = G                    # G lam_j = -delta_j
    b_eq[6 * (j + 1):6 * (j + 2)] = -np.asarray(deltas[j], float)

  rows = []; bub = []
  for j in range(m):                                        # t_j >= |lam_j|
    r = np.zeros((K, n)); r[:, col(1 + j)] = I;  r[:, col(1 + m + j)] = -I; rows.append(r); bub.append(np.zeros(K))
    r = np.zeros((K, n)); r[:, col(1 + j)] = -I; r[:, col(1 + m + j)] = -I; rows.append(r); bub.append(np.zeros(K))
  r = np.zeros((K, n)); r[:, col(0)] = -I                   # lam0 - sum_j t_j >= 0  (worst-case >= 0)
  for j in range(m):
    r[:, col(1 + m + j)] += I
  rows.append(r); bub.append(np.zeros(K))
  fin = np.isfinite(ub)                                     # lam0 + sum_j t_j <= ub  (worst-case <= ub)
  if fin.any():
    idx = np.where(fin)[0]; r = np.zeros((idx.size, n))
    for rr, k in enumerate(idx):
      r[rr, col(0).start + k] = 1.0
      for j in range(m):
        r[rr, col(1 + m + j).start + k] = 1.0
    rows.append(r); bub.append(ub[fin])

  vb = [(0.0, None if not fin[k] else float(ub[k])) for k in range(K)]   # lam0 in [0,ub]
  vb += [(None, None)] * (K * m)                                         # lam_j free
  vb += [(0.0, None)] * (K * m)                                          # t_j >= 0
  t0 = time.perf_counter() if _WRENCH_TIMING_ENABLED else None
  res = linprog(np.zeros(n), A_ub=np.vstack(rows), b_ub=np.concatenate(bub),
                A_eq=A_eq, b_eq=b_eq, bounds=vb, method="highs")
  if t0 is not None:
    _record_wrench_optimizer(time.perf_counter() - t0)
  return bool(res.success)
