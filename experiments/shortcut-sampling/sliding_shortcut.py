"""Drop-in replacement for mr_bench's ``robot_mode_shortcut`` with pluggable interval-space
samplers (random / Halton / sliding+Halton) and an optional candidate pre-filter.

Everything here lives OUTSIDE mr_bench: mr_bench is read-only and is only imported and
monkeypatched.  ``install()`` rebinds the name everywhere it has already been imported.

Background
----------
Shortcutting is sampling in the interval space  I = {(s1,s2) in [0,1]^2 : s1 < s2}
("Shortcutting Revisited: Improved Sampling and Irreplaceable Optimality", WAFR 2026,
Awad/Erickson/LaValle/Prencipe/Sakcak/Timperi/LaValle).  The paper contributes

  * Halton shortcutting     -- scrambled 2D Halton (bases 2,3) over I instead of uniform draws;
  * Sliding + Halton        -- Algorithm 2: fix an interval LENGTH lambda_i, slide the pair
                               across [0,1] with overlap omega_i, then shrink lambda and repeat;
                               longest intervals are tried FIRST.  Falls back to a randomized
                               Halton sequence on exhaustion or on no-improvement-over-last-15;
  * candidate pre-filter    -- Definition 2(1): a point of the sequence is a *candidate
                               shortcut* only if the sub-path it spans is not ALREADY a geodesic.
                               Non-candidates are discarded before any collision check.

Multi-robot extension (OURS, not the paper's -- the paper is single-robot)
-------------------------------------------------------------------------
``robot_mode_shortcut`` shortcuts ONE robot at a time and rejects the proposal outright when
that robot's ``mode.task_ids[r]`` differs at i and at j (the "task-id gate", lines 149-156 of
the original).  So the eligible interval space for robot r is NOT the whole triangle: it is the
UNION of triangles over r's maximal runs of constant ``task_ids[r]``.  On our plans a run is
6-21 indices out of a 127-236 state path, so uniform draws over the whole path waste almost
every attempt.  The Halton and sliding samplers therefore sample PER ROBOT, PER RUN: they pick a
robot (round-robin, as the original does), pick one of that robot's eligible runs, and map
(s1,s2) onto that run's index range.  The task-id gate is kept verbatim but never fires for
them.

Everything else -- the candidate construction, the ``k-1`` correction at doubled mode-switch
states, the cost test, the asserts, the collision acceptance, the ``costs``/``times``
bookkeeping and the return value -- is copied verbatim from mr_bench, so any measured
difference between samplers is attributable to the sampling alone.

Usage
-----
    import sliding_shortcut as ss
    ss.install(sampler="sliding", prefilter=True, seed=0)      # patches every binding
    ...  run mp_export.run_birrt(...) ...
    print(ss.LAST_STATS)                                       # side channel
    ss.uninstall()

or call it directly:

    new_path, (costs, times) = ss.robot_mode_shortcut(
        env, path, max_iter=1000, resolution=0.01, sampler="sliding", prefilter=True)
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "robot_mode_shortcut",
    "install",
    "uninstall",
    "sliding_numbers",
    "constant_task_runs",
    "LAST_STATS",
    "STATS_HISTORY",
]

# --------------------------------------------------------------------------------------
# side channel for the benchmark
# --------------------------------------------------------------------------------------

LAST_STATS: Dict = {}
STATS_HISTORY: List[Dict] = []


# --------------------------------------------------------------------------------------
# lazy mr_bench imports (keeps `import sliding_shortcut` cheap and import-order agnostic)
# --------------------------------------------------------------------------------------

_MRB = {}


def _mrb():
    if not _MRB:
        from multi_robot_multi_goal_planning.problems.planning_env import State
        from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost
        from multi_robot_multi_goal_planning.planners.shortcutting import (
            remove_interpolated_nodes,
        )

        _MRB.update(
            State=State,
            interpolate_path=interpolate_path,
            path_cost=path_cost,
            remove_interpolated_nodes=remove_interpolated_nodes,
        )
    return _MRB


# --------------------------------------------------------------------------------------
# Algorithm 2 of the paper -- sliding numbers
# --------------------------------------------------------------------------------------


def sliding_numbers(
    lambdas: Sequence[float], omegas: Sequence[float]
) -> List[Tuple[float, float]]:
    """Paper Algorithm 2, verbatim.

    ``lambdas`` are interval *lengths*, ``omegas`` the *overlaps* (0 <= omega < 1).
    With lambda_i = 2**-i and omega_i = 1/2 for i = 0..6 this returns the paper's 247 points
    (1 + 3 + 7 + 15 + 31 + 63 + 127).  The longest intervals come first.
    """
    L: List[Tuple[float, float]] = []
    for lam, om in zip(lambdas, omegas):
        alpha = lam * (1.0 - om)
        s1 = 0.0
        s2 = lam
        # guard against a degenerate schedule (alpha == 0 would loop forever)
        if alpha <= 0.0:
            L.append((0.0, 1.0))
            continue
        while s2 < 1.0:
            L.append((s1, s2))
            s1 += alpha
            s2 += alpha
        L.append((s1, 1.0))
    return L


def _default_schedule(n_levels: int = 7) -> Tuple[List[float], List[float]]:
    """lambda_i = 2**-i, omega_i = 1/2, i = 0 .. n_levels-1 (the paper's choice, n_levels=7)."""
    return [2.0**-i for i in range(n_levels)], [0.5] * n_levels


# --------------------------------------------------------------------------------------
# eligible interval space:  maximal runs of constant task_ids[r]
# --------------------------------------------------------------------------------------


def constant_task_runs(path, r: int, min_span: int = 2, mode: str = "runs") -> List[np.ndarray]:
    """Index sets on which robot ``r`` passes the task-id gate.

    ``mode="runs"``  (default): maximal runs of consecutive indices with constant
        ``task_ids[r]``.  This is a *subset* of what the original gate accepts -- the gate only
        compares the two endpoints, so it also admits an i and a j that carry the same task id
        but sit in two non-adjacent blocks with a different task in between.  On assembly plans
        a robot's task index is monotone, so the two coincide; ``mode="value"`` covers the
        general case.
    ``mode="value"``: all indices sharing a task-id value, contiguous or not.  A shortcut then
        interpolates robot r straight through a stretch where it was doing something else --
        legal for the gate, but it erases intermediate task modes.

    Returns a list of sorted index arrays, each with at least ``min_span + 1`` entries and an
    index span of at least ``min_span``.
    """
    ids = [s.mode.task_ids[r] for s in path]
    n = len(ids)
    out: List[np.ndarray] = []

    if mode == "value":
        buckets: Dict[int, List[int]] = {}
        for k, v in enumerate(ids):
            buckets.setdefault(v, []).append(k)
        for v in sorted(buckets):
            idxs = np.asarray(buckets[v], dtype=int)
            if len(idxs) >= 2 and idxs[-1] - idxs[0] >= min_span:
                out.append(idxs)
        return out

    a = 0
    for k in range(1, n + 1):
        if k == n or ids[k] != ids[k - 1]:
            b = k - 1
            if b - a >= min_span:
                out.append(np.arange(a, b + 1, dtype=int))
            a = k
    return out


# --------------------------------------------------------------------------------------
# proposal samplers
# --------------------------------------------------------------------------------------


class _Stop(Exception):
    """Raised by a sampler when it can never produce another proposal."""


class _RandomSampler:
    """The ORIGINAL uniform draws, verbatim (lines 127-147 of mr_bench's shortcutting.py).

    Reproduces the original bit-for-bit: same two ``np.random.randint`` calls per iteration,
    same swap, same ``abs(j-i) < 2`` rejection, same placement of the ``rr_robot`` increment
    (after the short-interval rejection, before the task-id gate).
    """

    name = "random"

    def __init__(self, env, new_path, robot_choice, **_):
        self.n = len(new_path)
        self.n_robots = len(env.robots)
        self.robot_choice = robot_choice
        self.rr_robot = 0

    def propose(self):
        i = np.random.randint(0, self.n)
        j = np.random.randint(0, self.n)

        if i > j:
            q = i
            i = j
            j = q

        if abs(j - i) < 2:
            return None

        if self.robot_choice == "round_robin":
            robots_to_shortcut = [self.rr_robot % self.n_robots]
            self.rr_robot += 1
        else:
            robots_to_shortcut = [np.random.randint(0, self.n_robots)]

        return i, j, robots_to_shortcut

    def feedback(self, improved: bool) -> None:
        pass

    def report(self) -> Dict:
        return {}


class _Pair:
    """Per-(robot, run) sampling state: its own Halton stream and its own sliding schedule."""

    __slots__ = (
        "robot",
        "run_id",
        "idxs",
        "span",
        "used",
        "halton",
        "_buf",
        "_buf_i",
        "sched",
        "pos",
        "passes",
        "stall",
        "improved_this_pass",
        "fallback",
        "sliding_points",
        "seed",
    )

    def __init__(self, robot: int, run_id: int, idxs: np.ndarray, seed: int):
        self.robot = robot
        self.run_id = run_id
        self.idxs = idxs
        self.span = len(idxs) - 1
        self.used = 0
        self.seed = seed
        self.halton = None
        self._buf = None
        self._buf_i = 0
        self.sched: List[Tuple[int, int]] = []
        self.pos = 0
        self.passes = 0
        self.stall = 0
        self.improved_this_pass = 0
        self.fallback: Optional[str] = None
        self.sliding_points = 0

    # -- Halton -------------------------------------------------------------------------
    def _halton_pair(self) -> Optional[Tuple[float, float]]:
        from scipy.stats import qmc

        if self.halton is None:
            self.halton = qmc.Halton(d=2, scramble=True, seed=self.seed)
        if self._buf is None or self._buf_i >= len(self._buf):
            self._buf = self.halton.random(64)
            self._buf_i = 0
        u = self._buf[self._buf_i]
        self._buf_i += 1
        s1, s2 = float(u[0]), float(u[1])
        if s1 > s2:
            s1, s2 = s2, s1
        if s1 == s2:
            return None
        return s1, s2

    def map_pair(self, s1: float, s2: float) -> Optional[Tuple[int, int]]:
        """(s1,s2) in I  ->  (i,j) indices of this run, or None if degenerate (j-i < 2)."""
        p1 = int(np.floor(s1 * self.span + 0.5))
        p2 = int(np.floor(s2 * self.span + 0.5))
        if p2 <= p1:
            return None
        i = int(self.idxs[p1])
        j = int(self.idxs[p2])
        if j - i < 2:
            return None
        return i, j

    def next_halton(self, retries: int = 8) -> Optional[Tuple[int, int]]:
        for _ in range(retries):
            u = self._halton_pair()
            if u is None:
                continue
            ij = self.map_pair(*u)
            if ij is not None:
                return ij
        return None

    # -- sliding ------------------------------------------------------------------------
    def build_schedule(self, lambdas, omegas, min_span: int = 2) -> None:
        """Precompute this run's sliding schedule as INDEX pairs.

        Two discretisation adaptations of Algorithm 2, both of which only remove guaranteed-
        useless points (documented in the module docstring / design notes):
          * levels with ``lambda_i * span < min_span`` are dropped -- on a discrete path they
            can only produce (i,j) with j-i < 2, which the original rejects outright;
          * points that map onto an (i,j) already produced earlier in the schedule are dropped.
        """
        seen = set()
        sched: List[Tuple[int, int]] = []
        for lam, om in zip(lambdas, omegas):
            if lam * self.span < min_span:
                break
            for s1, s2 in sliding_numbers([lam], [om]):
                ij = self.map_pair(s1, s2)
                if ij is None or ij in seen:
                    continue
                seen.add(ij)
                sched.append(ij)
        self.sched = sched
        if not sched:
            self.fallback = "no_sliding_levels"

    def next_sliding(self, max_passes: Optional[int]) -> Optional[Tuple[int, int]]:
        if self.fallback is not None:
            return None
        while self.pos >= len(self.sched):
            # a full pass over the sliding schedule has been consumed
            self.passes += 1
            if self.improved_this_pass == 0 or (
                max_passes is not None and self.passes >= max_passes
            ):
                self.fallback = "exhausted"
                return None
            self.pos = 0
            self.stall = 0
            self.improved_this_pass = 0
        ij = self.sched[self.pos]
        self.pos += 1
        self.sliding_points += 1
        return ij


class _QmcSampler:
    """Per-robot, per-run Halton ('halton') or sliding+Halton ('sliding') sampling.

    Robot choice: round-robin over the robots that have at least one eligible run -- the same
    policy the original uses, so the robots get the same share of the budget.

    Run choice: a deterministic *length-proportional* divisor (D'Hondt) rule -- always take the
    run maximising ``len(run) / (used + 1)``.  Justification: (a) it is deterministic, so the
    sliding schedule stays reproducible and each run's "longest interval first" ordering is
    preserved (a random or plain round-robin choice would interleave levels across runs and
    destroy that ordering); (b) attempts land in proportion to run length, which is where the
    slack -- and the achievable cost reduction -- actually is.  ``run_choice="round_robin"``
    switches to a plain cycle.
    """

    def __init__(
        self,
        env,
        new_path,
        robot_choice,
        kind: str = "sliding",
        seed: int = 0,
        eligible: str = "runs",
        run_choice: str = "proportional",
        sliding_stall: int = 15,
        sliding_max_passes: Optional[int] = None,
        sliding_levels: int = 7,
        **_,
    ):
        self.name = kind
        self.kind = kind
        self.n_robots = len(env.robots)
        self.run_choice = run_choice
        self.sliding_stall = sliding_stall
        self.sliding_max_passes = sliding_max_passes
        self.lambdas, self.omegas = _default_schedule(sliding_levels)

        self.pairs: Dict[int, List[_Pair]] = {}
        for r in range(self.n_robots):
            runs = constant_task_runs(new_path, r, min_span=2, mode=eligible)
            plist = []
            for k, idxs in enumerate(runs):
                p = _Pair(r, k, idxs, seed=(int(seed) + 1000003 * r + 7919 * k) % (2**31 - 1))
                if kind == "sliding":
                    p.build_schedule(self.lambdas, self.omegas)
                plist.append(p)
            self.pairs[r] = plist

        self.robots_with_runs = [r for r in range(self.n_robots) if self.pairs[r]]
        self.rr_robot = 0
        self._last: Optional[_Pair] = None
        self.degenerate = 0

    def _pick_pair(self, r: int) -> _Pair:
        plist = self.pairs[r]
        if len(plist) == 1:
            return plist[0]
        if self.run_choice == "round_robin":
            tot = sum(p.used for p in plist)
            return plist[tot % len(plist)]
        # D'Hondt / Jefferson divisor rule -> deterministic length-proportional allocation
        best = None
        best_score = -1.0
        for p in plist:
            score = len(p.idxs) / (p.used + 1.0)
            if score > best_score:
                best_score = score
                best = p
        return best

    def propose(self):
        if not self.robots_with_runs:
            raise _Stop("no robot has an eligible constant-task run of span >= 2")

        r = self.robots_with_runs[self.rr_robot % len(self.robots_with_runs)]
        self.rr_robot += 1

        pair = self._pick_pair(r)
        pair.used += 1
        self._last = pair

        ij = None
        if self.kind == "sliding":
            ij = pair.next_sliding(self.sliding_max_passes)
        if ij is None:
            ij = pair.next_halton()
        if ij is None:
            self.degenerate += 1
            return None

        i, j = ij
        return i, j, [r]

    def feedback(self, improved: bool) -> None:
        p = self._last
        if p is None or self.kind != "sliding" or p.fallback is not None:
            return
        if improved:
            p.stall = 0
            p.improved_this_pass += 1
            return
        p.stall += 1
        if self.sliding_stall and p.stall >= self.sliding_stall:
            # end of pass: retry the schedule only if this pass actually bought something
            p.passes += 1
            if p.improved_this_pass == 0 or (
                self.sliding_max_passes is not None and p.passes >= self.sliding_max_passes
            ):
                p.fallback = "stall"
            else:
                p.pos = 0
                p.stall = 0
                p.improved_this_pass = 0

    def report(self) -> Dict:
        allp = [p for r in self.pairs for p in self.pairs[r]]
        passes = [p.passes for p in allp]
        fb: Dict[str, int] = {}
        for p in allp:
            fb[p.fallback or "still_sliding"] = fb.get(p.fallback or "still_sliding", 0) + 1
        return {
            "eligible_pairs": len(allp),
            "eligible_runs_per_robot": {r: len(self.pairs[r]) for r in self.pairs},
            "run_spans": sorted(len(p.idxs) for p in allp),
            "sliding_schedule_sizes": sorted(len(p.sched) for p in allp),
            "sliding_points_consumed": sum(p.sliding_points for p in allp),
            "sliding_passes_total": sum(passes),
            "sliding_passes_min": min(passes) if passes else 0,
            "sliding_passes_max": max(passes) if passes else 0,
            "sliding_passes_mean": (sum(passes) / len(passes)) if passes else 0.0,
            "sliding_fallbacks": fb,
            "degenerate_draws": self.degenerate,
        }


def _make_sampler(kind: str, env, new_path, robot_choice, **kw):
    if kind == "random":
        return _RandomSampler(env, new_path, robot_choice, **kw)
    if kind in ("halton", "sliding"):
        return _QmcSampler(env, new_path, robot_choice, kind=kind, **kw)
    raise ValueError(f"unknown sampler {kind!r} (expected 'random', 'halton' or 'sliding')")


# --------------------------------------------------------------------------------------
# candidate pre-filter (paper Definition 2 (1))
# --------------------------------------------------------------------------------------


class _Geodesics:
    """Cached per-robot sub-path arc lengths, so the pre-filter is O(1) per query."""

    def __init__(self, env, new_path):
        n_robots = len(env.robots)
        self.Q = [
            np.asarray([s.q[r] for s in new_path], dtype=np.float64) for r in range(n_robots)
        ]
        self.cum = [self._cum(q) for q in self.Q]

    @staticmethod
    def _cum(q: np.ndarray) -> np.ndarray:
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
        out = np.empty(len(q), dtype=np.float64)
        out[0] = 0.0
        np.cumsum(seg, out=out[1:])
        return out

    def is_geodesic(self, r: int, i: int, j: int, eps: float) -> bool:
        arc = self.cum[r][j] - self.cum[r][i]
        direct = float(np.linalg.norm(self.Q[r][j] - self.Q[r][i]))
        return (arc - direct) <= eps * max(1.0, direct)

    def update(self, r: int, i: int, j: int, new_path) -> None:
        for k in range(i, j + 1):
            self.Q[r][k] = new_path[k].q[r]
        self.cum[r] = self._cum(self.Q[r])


# --------------------------------------------------------------------------------------
# the drop-in replacement
# --------------------------------------------------------------------------------------


def robot_mode_shortcut(
    env,
    path: List,
    max_iter: int = 1000,
    resolution=0.001,
    tolerance=0.01,
    robot_choice="round_robin",
    interpolation_resolution: float = 0.5,
    *,
    sampler: str = "random",
    prefilter: bool = False,
    geodesic_eps: float = 1e-9,
    max_attempts: Optional[int] = None,
    max_checks: Optional[int] = None,
    seed: int = 0,
    eligible: str = "runs",
    run_choice: str = "proportional",
    sliding_stall: int = 15,
    sliding_max_passes: Optional[int] = None,
    sliding_levels: int = 7,
    deep_counters: bool = True,
    stats_out: Optional[Dict] = None,
    verbose: bool = True,
):
    """Same signature / same return contract as mr_bench's ``robot_mode_shortcut``.

    Extra keyword-only knobs:
      sampler         'random' (bit-identical to the original) | 'halton' | 'sliding'
      prefilter       discard non-candidate shortcuts (already-geodesic sub-paths) before the
                      candidate is even built
      geodesic_eps    relative tolerance of that test
      max_attempts    overrides the hard-coded ``250 * 10``
      max_checks      extra stop condition: stop after this many collision checks (for
                      cost-vs-#checks benchmarking).  None = off, i.e. original behaviour
      seed            seed of the (scrambled) Halton streams; does NOT touch np.random
      eligible        'runs' (maximal constant-task runs) | 'value' (all indices per task id)
      run_choice      'proportional' (D'Hondt) | 'round_robin'
      sliding_stall   switch a run off sliding after this many non-improving sliding points
      sliding_max_passes  cap on repeats of the sliding schedule per run (None = unbounded;
                      a pass with zero improvements always ends sliding for that run)
      sliding_levels  number of lambda_i = 2**-i levels (7 = the paper's 247 numbers)

    Returns ``(new_path, [costs, times])`` exactly like the original.  Instrumentation goes to
    ``stats_out`` (if given), to ``sliding_shortcut.LAST_STATS`` and to ``STATS_HISTORY``.
    """
    M = _mrb()
    State = M["State"]
    interpolate_path = M["interpolate_path"]
    path_cost = M["path_cost"]
    remove_interpolated_nodes = M["remove_interpolated_nodes"]

    non_redundant_path = remove_interpolated_nodes(path)
    new_path = interpolate_path(non_redundant_path, interpolation_resolution)

    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]

    st: Dict = {
        "sampler": sampler,
        "prefilter": bool(prefilter),
        "n_states": len(new_path),
        "n_robots": len(env.robots),
        "iters": 0,
        "proposals": 0,
        "skipped_short": 0,
        "skipped_gate": 0,
        "skipped_prefilter": 0,
        "cost_rejects": 0,
        "cost_passes": 0,
        "collision_checks": 0,
        "accepted": 0,
        "rejected_collision": 0,
        "accepted_per_robot": {},
        "checks_at_cost_sample": [0],
        "cost_initial": float(costs[0]),
        "stop_reason": None,
    }
    deep = {"is_collision_free": 0, "is_edge_collision_free": 0}

    sampler_obj = _make_sampler(
        sampler,
        env,
        new_path,
        robot_choice,
        seed=seed,
        eligible=eligible,
        run_choice=run_choice,
        sliding_stall=sliding_stall,
        sliding_max_passes=sliding_max_passes,
        sliding_levels=sliding_levels,
    )
    geo = _Geodesics(env, new_path) if prefilter else None

    # ---- optional deep counters: wrap the low-level checkers on the INSTANCE ------------
    # (the primary counter -- the paper's axis -- is st["collision_checks"], incremented at our
    #  single env.is_path_collision_free call site; these two are only a finer breakdown)
    _restore = []  # (method name, previous instance attribute or _MISSING)
    _MISSING = object()
    if deep_counters:
        for meth in ("is_collision_free", "is_edge_collision_free"):
            try:
                orig = getattr(env, meth)
            except Exception:
                continue
            prev = getattr(env, "__dict__", {}).get(meth, _MISSING)

            def _wrap(orig=orig, meth=meth):
                def f(*a, **k):
                    deep[meth] += 1
                    return orig(*a, **k)

                return f

            try:
                setattr(env, meth, _wrap())
                _restore.append((meth, prev))
            except Exception:
                pass

    start_time = time.time()

    cnt = 0
    if max_attempts is None:
        max_attempts = 250 * 10
    iter = 0

    try:
        while True:
            iter += 1
            if cnt >= max_iter or iter >= max_attempts:
                st["stop_reason"] = "max_iter" if cnt >= max_iter else "max_attempts"
                break
            if max_checks is not None and st["collision_checks"] >= max_checks:
                st["stop_reason"] = "max_checks"
                break

            # ---------------- the ONLY behavioural change: where (i, j, robot) come from --
            try:
                proposal = sampler_obj.propose()
            except _Stop as e:
                st["stop_reason"] = f"sampler exhausted: {e}"
                break
            if proposal is None:
                st["skipped_short"] += 1
                continue
            i, j, robots_to_shortcut = proposal
            cost_before = costs[-1]

            # -------- verbatim from mr_bench (task-id gate) -------------------------------
            can_shortcut_this = True
            for r in robots_to_shortcut:
                if new_path[i].mode.task_ids[r] != new_path[j].mode.task_ids[r]:
                    can_shortcut_this = False
                    break

            if not can_shortcut_this:
                st["skipped_gate"] += 1
                sampler_obj.feedback(False)
                continue
            # -----------------------------------------------------------------------------

            # -------- candidate pre-filter (paper Def. 2 (1)) -- OPTIONAL, ours ----------
            if geo is not None and all(
                geo.is_geodesic(r, i, j, geodesic_eps) for r in robots_to_shortcut
            ):
                st["skipped_prefilter"] += 1
                sampler_obj.feedback(False)
                continue
            # -----------------------------------------------------------------------------

            st["proposals"] += 1

            # -------- verbatim from mr_bench (candidate construction) --------------------
            q0 = new_path[i].q
            q1 = new_path[j].q

            # precopmute all the differences
            q0_tmp = {}
            q1_tmp = {}
            diff_tmp = {}
            for r in robots_to_shortcut:
                q0_tmp[r] = q0[r] * 1
                q1_tmp[r] = q1[r] * 1
                diff_tmp[r] = (q1_tmp[r] - q0_tmp[r]) / (j - i)

            # constuct pth element for the shortcut
            path_element = []
            for k in range(j - i + 1):
                q = new_path[i + k].q.state() * 1.0

                r_cnt = 0
                for r in range(len(env.robots)):
                    dim = env.robot_dims[env.robots[r]]
                    if r in robots_to_shortcut:
                        # we assume that we double the mode switch configurations
                        if (
                            k != 0
                            and i + k != j
                            and new_path[i + k].mode != new_path[i + k - 1].mode
                        ):
                            q_interp = q0_tmp[r] + diff_tmp[r] * (k - 1)
                        else:
                            q_interp = q0_tmp[r] + diff_tmp[r] * k
                        q[r_cnt : r_cnt + dim] = q_interp

                    r_cnt += dim

                path_element.append(State(q0.from_flat(q), new_path[i + k].mode))

            # check if the shortcut improves cost
            if path_cost(path_element, env.batch_config_cost) >= path_cost(
                new_path[i : j + 1], env.batch_config_cost
            ):
                st["cost_rejects"] += 1
                sampler_obj.feedback(False)
                continue

            assert np.linalg.norm(path_element[0].q.state() - q0.state()) < 1e-6
            assert np.linalg.norm(path_element[-1].q.state() - q1.state()) < 1e-6

            cnt += 1
            st["cost_passes"] = cnt

            st["collision_checks"] += 1
            if env.is_path_collision_free(
                path_element,
                resolution=resolution,
                tolerance=tolerance,
                check_start_and_end=False,
            ):
                for k in range(j - i + 1):
                    new_path[i + k].q = path_element[k].q
                st["accepted"] += 1
                for r in robots_to_shortcut:
                    st["accepted_per_robot"][r] = st["accepted_per_robot"].get(r, 0) + 1
                    if geo is not None:
                        geo.update(r, i, j, new_path)
            else:
                st["rejected_collision"] += 1

            current_time = time.time()
            times.append(current_time - start_time)
            costs.append(path_cost(new_path, env.batch_config_cost))
            # -----------------------------------------------------------------------------

            st["checks_at_cost_sample"].append(st["collision_checks"])
            sampler_obj.feedback(costs[-1] < cost_before - 1e-12)
    finally:
        for meth, prev in _restore:
            try:
                if prev is _MISSING:
                    delattr(env, meth)
                else:
                    setattr(env, meth, prev)
            except Exception:
                pass

    st["iters"] = iter
    st["wall_time"] = time.time() - start_time
    st["cost_final"] = float(costs[-1])
    st["cost_reduction"] = st["cost_initial"] - st["cost_final"]
    st["deep_counters"] = dict(deep)
    st.update(sampler_obj.report())

    LAST_STATS.clear()
    LAST_STATS.update(st)
    STATS_HISTORY.append(st)
    if stats_out is not None:
        stats_out.clear()
        stats_out.update(st)

    # -------- verbatim from mr_bench --------------------------------------------------
    assert new_path[-1].mode == path[-1].mode
    assert np.linalg.norm(new_path[-1].q.state() - path[-1].q.state()) < 1e-6
    assert np.linalg.norm(new_path[0].q.state() - path[0].q.state()) < 1e-6

    print("original cost:", path_cost(path, env.batch_config_cost))
    print("Attempted shortcuts", cnt)
    print("new cost:", path_cost(new_path, env.batch_config_cost))
    # -----------------------------------------------------------------------------------

    if verbose:
        print(
            f"[sliding_shortcut] sampler={sampler} prefilter={bool(prefilter)} "
            f"iters={st['iters']} proposals={st['proposals']} "
            f"short={st['skipped_short']} gate={st['skipped_gate']} "
            f"prefilter_skips={st['skipped_prefilter']} cost_rejects={st['cost_rejects']} "
            f"checks={st['collision_checks']} accepted={st['accepted']} "
            f"cost {st['cost_initial']:.4f} -> {st['cost_final']:.4f} "
            f"in {st['wall_time']:.2f}s ({st['stop_reason']})"
        )
        if sampler == "sliding":
            print(
                f"[sliding_shortcut] runs={st.get('eligible_pairs')} "
                f"spans={st.get('run_spans')} sched={st.get('sliding_schedule_sizes')} "
                f"sliding_points={st.get('sliding_points_consumed')} "
                f"passes(total/min/max/mean)={st.get('sliding_passes_total')}/"
                f"{st.get('sliding_passes_min')}/{st.get('sliding_passes_max')}/"
                f"{st.get('sliding_passes_mean'):.2f} "
                f"fallbacks={st.get('sliding_fallbacks')}"
            )

    return new_path, [costs, times]


# --------------------------------------------------------------------------------------
# monkeypatching
# --------------------------------------------------------------------------------------

_ORIGINAL = None
_PATCHED = None
_PATCHED_MODULES: List[str] = []


def _load_original():
    global _ORIGINAL
    if _ORIGINAL is None:
        from multi_robot_multi_goal_planning.planners import shortcutting as _sc

        cur = _sc.robot_mode_shortcut
        _ORIGINAL = getattr(cur, "_sliding_shortcut_original", cur)
    return _ORIGINAL


def _make_patched(defaults: Dict):
    def _patched(
        env,
        path,
        max_iter: int = 1000,
        resolution=0.001,
        tolerance=0.01,
        robot_choice="round_robin",
        interpolation_resolution: float = 0.5,
        **user_kw,
    ):
        merged = dict(defaults)
        merged.update(user_kw)
        return robot_mode_shortcut(
            env,
            path,
            max_iter,
            resolution,
            tolerance,
            robot_choice,
            interpolation_resolution,
            **merged,
        )

    _patched.__name__ = "robot_mode_shortcut"
    _patched.__qualname__ = "sliding_shortcut.robot_mode_shortcut(patched)"
    _patched._sliding_shortcut_patch = True
    _patched._sliding_shortcut_original = _load_original()
    _patched._sliding_shortcut_defaults = defaults
    return _patched


def install(**defaults) -> Dict:
    """Rebind ``robot_mode_shortcut`` to this implementation EVERYWHERE it is already bound.

    ``defaults`` are the keyword-only knobs of :func:`robot_mode_shortcut`
    (``sampler=``, ``prefilter=``, ``seed=`` ...) applied to every patched call.

    Why a scan and not just the module attribute: several mr_bench modules do
    ``from ...shortcutting import robot_mode_shortcut`` at MODULE level (``rrtstar_base``,
    ``prioritized_planner``, ``solve``, ``planners/__init__``), which binds the NAME -- patching
    only ``shortcutting.robot_mode_shortcut`` would miss them.  ``assembly/mp_export.py`` happens
    to import it INSIDE ``run_birrt`` (line 722), so for that caller the module attribute alone
    is enough; the scan covers it anyway if it is ever hoisted.

    Returns a report dict listing every module that was patched.
    """
    global _PATCHED, _PATCHED_MODULES
    orig = _load_original()
    _PATCHED = _make_patched(dict(defaults))

    patched_in: List[str] = []
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            val = getattr(mod, "robot_mode_shortcut", None)
        except Exception:
            continue
        if val is orig or getattr(val, "_sliding_shortcut_patch", False):
            try:
                setattr(mod, "robot_mode_shortcut", _PATCHED)
                patched_in.append(name)
            except Exception:
                pass

    # the defining module, even if the scan somehow missed it
    from multi_robot_multi_goal_planning.planners import shortcutting as _sc

    if _sc.robot_mode_shortcut is not _PATCHED:
        _sc.robot_mode_shortcut = _PATCHED
        patched_in.append(_sc.__name__)

    _PATCHED_MODULES = sorted(set(patched_in))
    return {"patched_modules": _PATCHED_MODULES, "defaults": dict(defaults)}


def uninstall() -> Dict:
    """Undo :func:`install`."""
    global _PATCHED
    orig = _load_original()
    restored = []
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            val = getattr(mod, "robot_mode_shortcut", None)
        except Exception:
            continue
        if getattr(val, "_sliding_shortcut_patch", False):
            try:
                setattr(mod, "robot_mode_shortcut", orig)
                restored.append(name)
            except Exception:
                pass
    _PATCHED = None
    return {"restored_modules": sorted(set(restored))}
