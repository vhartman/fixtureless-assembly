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

import numpy as np
from scipy.optimize import linprog

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
                                      "T_axis": 0.4, "T_lat": 0.4},
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
    # insertion (mating) reaction: ONE magnitude for now, but read per-mate from a dict so
    # individual mates can override it later without touching call sites. `_default` applies
    # to any mate not listed.
    "insert_force": {"_default": 10.0},                 # newtons, along the mate axis
    # check that the INSERTING arm's own grasp can transmit the press (the seating reaction
    # also acts on the held part). False = skip the check, equivalent to giving the inserter
    # grasp unbounded capability ("a very high value") -- the target-side check stays on.
    "check_inserter": True,
}


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


def cap_table_unilateral(points):
  # UNILATERAL table support: at each support point, a normal force that can only PUSH UP
  # (+z), never pull, and no tangential (frictionless) component. The set of such points is
  # the support polygon. Unbounded magnitude (the table takes any downward load).
  n = np.array([0.0, 0.0, 1.0])
  G = [_force_gen(n, p) for p in points]
  return np.array(G).T, np.full(len(G), np.inf)


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


def cap_asymmetric_grasp(point, n, t1, t2, F_push, F_pull, mu, T_axis, T_lat):
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
  for axis, T in ((n, T_axis), (t1, T_lat), (t2, T_lat)):
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


_TABLE_BUILDERS = {"unilateral": cap_table_unilateral}


def table_capability(points):
  return _TABLE_BUILDERS[PARAMS["table_model"]](points)


# ---------------------------------------------------------------------------------------
# External load and the equilibrium LP.
# ---------------------------------------------------------------------------------------
def gravity_wrench(masses_points):
  # total gravity wrench about the origin for [(mass, com_world), ...].
  w = np.zeros(6)
  for m, c in masses_points:
    f = np.array([0.0, 0.0, -m * GRAVITY])
    w += np.concatenate([f, _cross3(np.asarray(c, float), f)])
  return w


def point_force_wrench(force_vec, point):
  # wrench about the origin of a force applied at a world point (e.g. insertion reaction).
  f = np.asarray(force_vec, float)
  return np.concatenate([f, _cross3(np.asarray(point, float), f)])


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


def equilibrium_feasible_matrices(G, ub, w_external, tol=1e-6, bounds=None):
  """The LP feasibility test on the already-assembled generator matrix (G [6xK], ub [K]).
  The contact-INDEPENDENT parts (G, ub, and the derived `bounds` list) can be built ONCE and
  reused across many external loads -- e.g. the corners of an insertion-uncertainty box --
  so the caller (`_group_stable`/`_grounding_caps`) hoists them out of the per-load loop."""
  if G.shape[1] == 0:
    return bool(np.linalg.norm(w_external) <= tol)
  if bounds is None:
    bounds = [(0.0, None if not np.isfinite(u) else float(u)) for u in ub]
  res = linprog(c=np.zeros(G.shape[1]), A_eq=G, b_eq=-np.asarray(w_external, float),
                bounds=bounds, method="highs")
  return bool(res.success)
