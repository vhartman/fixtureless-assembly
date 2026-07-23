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
from collections import OrderedDict
from scipy.optimize import linprog

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
    #   "affine"  (CONSERVATIVE): one fixed AFFINE force-response policy must cover the whole box
    #             -- an open-loop / fixed-compliance grasp that cannot sense the seating error and
    #             re-optimize. A single LP (vs 2^k corner solves); sound but stricter. See
    #             equilibrium_robust_affine. Ignored when insertion_uncertainty is None.
    "robust_mode": "corners",
    # GRASP COMPLIANCE (--grasp-compliance): per-kind multiplier on the INSERTION-UNCERTAINTY
    # box when that kind holds a body in the press's load path. A suction cup is a COMPLIANT
    # contact -- it deflects under contact loads, so the same commanded seating wanders more:
    # picking/carrying (gravity only, no accuracy demand) is unaffected, but assembling with
    # a vacuum-held target/inserter must survive a LARGER misalignment box. None = off
    # (byte-identical); needs insertion_uncertainty to have any effect (it scales the box).
    # Avoidance of imprecise-holder assembly is then EMERGENT, not a hardcoded prohibition.
    "grasp_compliance": None,                           # e.g. {"two_finger": 1.0, "vacuum": 3.5}
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


_TABLE_BUILDERS = {"unilateral": cap_table_unilateral}


def table_capability(points):
  return _TABLE_BUILDERS[PARAMS["table_model"]](points)


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
  # a persistent HiGHS model per geometry: 0 objective, K vars in [0,ub], 6 equality rows Glam=b
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
  for i in range(6):
    h.addRow(0.0, 0.0, K, idx, np.ascontiguousarray(G[i, :], dtype=float))
  _HS_MODELS[geo_key] = m = [h, K]
  if len(_HS_MODELS) > _HS_CAP:
    _HS_MODELS.popitem(last=False)                          # evict least-recently-used geometry
  return m


def _hs_feasible(geo_key, G, ub, beq):
  h, _ = _hs_model(geo_key, G, ub)
  for i in range(6):
    h.changeRowBounds(i, float(beq[i]), float(beq[i]))
  h.run()
  return h.getModelStatus() == _HS_OPTIMAL                  # 0-objective LP: optimal <=> feasible


def _extract_basis(G, ub, x, tol=1e-7):
  # Turn a HiGHS basic-feasible vertex `x` into a reusable warm basis. A feasibility BFS of a
  # 6-row system has (generically) exactly 6 variables strictly between their bounds -- the
  # BASIC columns, an invertible 6x6 -- with the rest pinned at 0 or ub (NONBASIC). We store the
  # basic columns' inverse and the constant wrench the pinned nonbasics contribute, so a later
  # load w' on the SAME geometry is tested by one 6x6 back-substitution instead of a fresh LP.
  ubv = np.where(np.isfinite(ub), ub, np.inf)
  basic = np.where((x > tol) & (x < ubv - tol))[0]
  if basic.size != 6:
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
  res = linprog(c=np.zeros(G.shape[1]), A_eq=G, b_eq=-np.asarray(w_external, dtype=float),
                bounds=bounds, method="highs")
  return bool(res.status == 0), (res.x if res.status == 0 else np.zeros(G.shape[1]))


def equilibrium_feasible_matrices(G, ub, w_external, tol=1e-6, bounds=None, geo_key=None):
  """The LP feasibility test on the already-assembled generator matrix (G [6xK], ub [K]).
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
  res = linprog(c=np.zeros(G.shape[1]), A_eq=G, b_eq=beq, bounds=bounds, method="highs")
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
  affine policy is one admissible way to respond, its feasibility IMPLIES every corner is stable
  (sound, conservative). It rejects cases that would require the contacts to switch strategy
  between realizations. One LP replaces the 2^m corner solves.

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
  res = linprog(np.zeros(n), A_ub=np.vstack(rows), b_ub=np.concatenate(bub),
                A_eq=A_eq, b_eq=b_eq, bounds=vb, method="highs")
  return bool(res.success)
