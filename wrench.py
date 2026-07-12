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
    # friction-pyramid grasp (cap_friction_grasp) per gripper type, in the grasp's own frame
    # (approach axis = local z). Fields: F_axial [N] max push/pull along the axis; mu the
    # friction coefficient -> lateral (shear) force <= mu*axial; T_axis / T_lat [Nm] the
    # couple it can resist about the approach axis / the two lateral axes.
    "grasp": {
        # two-finger: strong grip, decent shear, but WEAK TORQUE (the part pivots in the jaws)
        "two_finger": {"F_axial": 40.0, "mu": 0.6, "T_axis": 0.4, "T_lat": 0.4},
        # vacuum: decent axial suction, but WEAK LATERAL/SHEAR (small mu -> holds from the top,
        # slips from the side); the cup's area gives it a bit more couple than a bare point
        "vacuum":     {"F_axial": 25.0, "mu": 0.15, "T_axis": 0.8, "T_lat": 0.8},
    },
    # insertion (mating) reaction: ONE magnitude for now, but read per-mate from a dict so
    # individual mates can override it later without touching call sites. `_default` applies
    # to any mate not listed.
    "insert_force": {"_default": 10.0},                 # newtons, along the mate axis
}


def insert_force(part):
  """Insertion-force magnitude for mating `part` (per-mate dict, falling back to default)."""
  d = PARAMS["insert_force"]
  return d.get(part, d["_default"])


# ---------------------------------------------------------------------------------------
# Contact capabilities. Each returns (G, A_ub, b_ub): G is 6xk generator wrenches, and the
# suppliable set is { G @ lambda : lambda >= 0, A_ub @ lambda <= b_ub }. The extra linear
# inequalities let a FRICTION PYRAMID bound its normal force (a plain per-generator cap
# cannot -- the normal is a SUM of the pyramid rays). Unbounded contacts (table, rigid weld)
# have no inequalities (empty A_ub) -> their rays are limited only by lambda >= 0.
# ---------------------------------------------------------------------------------------
def _force_gen(direction, point):
  # generator wrench of a unit force along `direction` applied at world `point`.
  d = np.asarray(direction, float)
  p = np.asarray(point, float)
  return np.concatenate([d, np.cross(p, d)])


def _couple_gen(axis):
  # generator wrench of a unit pure couple about `axis`.
  return np.concatenate([np.zeros(3), np.asarray(axis, float)])


_AXES = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]


def _no_bounds(k):
  return np.zeros((0, k)), np.zeros(0)                  # unbounded: no inequality rows


def cap_rigid_weld(point):
  # RIGID WELD -- the whole 6-D wrench space: any force at the interface + any couple. The
  # special case the others specialise. Point is the mate interface centre.
  G = [_force_gen(s * a, point) for a in _AXES for s in (+1.0, -1.0)]
  G += [_couple_gen(s * a) for a in _AXES for s in (+1.0, -1.0)]
  G = np.array(G).T
  return (G,) + _no_bounds(G.shape[1])


def cap_table_unilateral(points):
  # UNILATERAL table support: at each support point, a normal force that can only PUSH UP
  # (+z), never pull, and no tangential (frictionless) component. The set of such points is
  # the support polygon. Unbounded magnitude (the table takes any downward load).
  n = np.array([0.0, 0.0, 1.0])
  G = np.array([_force_gen(n, p) for p in points]).T
  return (G,) + _no_bounds(G.shape[1])


def cap_friction_grasp(point, n, t1, t2, F_axial, mu, T_axis, T_lat):
  # DIRECTIONAL grasp capability at `point`, approach axis `n` (unit), lateral axes `t1`,`t2`
  # -- a wrench BOX in the grasp frame:
  #   FORCE : |axial| <= F_axial ; |lateral| <= mu * F_axial. The lateral bound is the
  #           friction at the grip/suction PRELOAD, so it holds even at zero external axial
  #           (a preloaded contact, unlike the table's unilateral cone whose friction would
  #           scale with the instantaneous normal). A SMALL mu => weak shear (suction cup:
  #           holds from the top, slips from the side).
  #   MOMENT: |couple about n| <= T_axis ; |couple about t1,t2| <= T_lat. A SMALL T_lat =>
  #           weak torque (a two-finger grip lets the held part pivot in the jaws).
  # Per-generator upper bounds (A_ub = I) -- no coupling sum needed for a box.
  n = np.asarray(n, float); t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)
  G, ub = [], []
  for vec, bound in ((n, F_axial), (t1, mu * F_axial), (t2, mu * F_axial)):
    for s in (+1.0, -1.0):
      G.append(_force_gen(s * vec, point)); ub.append(bound)
  for axis, T in ((n, T_axis), (t1, T_lat), (t2, T_lat)):
    for s in (+1.0, -1.0):
      G.append(_couple_gen(s * axis)); ub.append(T)
  G = np.array(G).T
  return G, np.eye(G.shape[1]), np.array(ub, float)


# ---------------------------------------------------------------------------------------
# External load and the equilibrium LP.
# ---------------------------------------------------------------------------------------
def gravity_wrench(masses_points):
  # total gravity wrench about the origin for [(mass, com_world), ...].
  w = np.zeros(6)
  for m, c in masses_points:
    f = np.array([0.0, 0.0, -m * GRAVITY])
    w += np.concatenate([f, np.cross(np.asarray(c, float), f)])
  return w


def point_force_wrench(force_vec, point):
  # wrench about the origin of a force applied at a world point (e.g. insertion reaction).
  f = np.asarray(force_vec, float)
  return np.concatenate([f, np.cross(np.asarray(point, float), f)])


def equilibrium_feasible(caps, w_external, tol=1e-6):
  """Can the contacts (list of (G, A_ub, b_ub)) supply wrenches summing to -w_external?

  Solves the LP feasibility  { G_all @ lambda = -w_external, lambda >= 0,
  A_ub @ lambda <= b_ub } where each contact's inequalities act on ITS OWN lambda block.
  Returns True iff a feasible multiplier vector exists. An empty contact set is stable only
  under (near-)zero external load."""
  if not caps:
    return bool(np.linalg.norm(w_external) <= tol)
  G = np.hstack([c[0] for c in caps])                   # 6 x K
  K = G.shape[1]
  A_blocks, b_parts, col = [], [], 0                    # place each cap's A_ub at its columns
  for Gi, Ai, bi in caps:
    ki = Gi.shape[1]
    if Ai.shape[0] > 0:
      full = np.zeros((Ai.shape[0], K)); full[:, col:col + ki] = Ai
      A_blocks.append(full); b_parts.append(bi)
    col += ki
  A_ub = np.vstack(A_blocks) if A_blocks else None
  b_ub = np.concatenate(b_parts) if b_parts else None
  res = linprog(c=np.zeros(K), A_eq=G, b_eq=-np.asarray(w_external, float),
                A_ub=A_ub, b_ub=b_ub, bounds=(0.0, None), method="highs")
  return bool(res.success)
