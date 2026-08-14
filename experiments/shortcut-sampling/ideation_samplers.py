"""Ideation-round shortcut samplers, benchmarked against the LANDED in-repo sampler.

Arms (all plug into the same verbatim mr_bench shortcut loop below; mr_bench untouched):

  baseline        the landed in-repo robot_mode_shortcut sampler reproduced exactly:
                  round-robin over ALL robots (runless robots burn the attempt), D'Hondt
                  divisor over the robot's constant-task runs with score span/(used+1)
                  (ties -> first), two np.random.randint integer draws inside the run,
                  |j-i| < 2 rejected.  Uses GLOBAL np.random exactly like the landed code
                  (the runner seeds it per run).
  r2              RANK 1: identical robot/run selection, but the in-run pair comes from a
                  Kronecker/R2 additive-recurrence stream per (robot, run) with a
                  Cranley-Patterson rotation from its own default_rng.  Sort -> floor(s*span
                  + 0.5) mapping, p2<=p1 / j-i<2 rejected, 8-retry loop (matches
                  sliding_shortcut._Pair.map_pair / next_halton).
  sweep_bisect    RANK 2: deterministic potential-gain sweep prefix (full-run straight-line
                  candidates in decreasing initial-G order, G>0 only), then the baseline
                  allocator; every collision-rejected candidate with span >= 4 pushes its two
                  midpoint halves onto that run's LIFO stack (cap 64, children below span 4
                  dropped), and a selected run with a nonempty stack pops instead of drawing.
  sweep_bisect_loc  same + split at the first colliding edge (captured by the runner's env
                  wrapper) instead of the midpoint; midpoint fallback.
  gainprop        RANK 3: run selection by (max(G,0) + g0)/(used+1) GLOBALLY over all
                  (robot, run) arms (robot round-robin dropped by design); G = full-run
                  straight-line composite gain, recomputed lazily for every run overlapping an
                  accepted shortcut; g0 = 0.05 * mean(initial clamped G) (1.0 if all zero) so
                  zero-G runs keep a trickle (the cost gate rewards retiming too - Def. 2 is
                  unsound here).  In-run draws uniform from the sampler's own rng.
  pairs           compatible two-robot shortcuts over intersections of their constant-task
                  runs.  This changes candidate type, not the collision certificate.
  pair_then_gain  a fixed early budget of pair candidates, followed by gainprop singles whose
                  gain estimates are refreshed after every accepted pair.
  current_gain_beta  the current landed ``run_choice='gain'`` policy, including its global
                  NumPy RNG stream and run-level Beta acceptance factor.  This arm exists so
                  observational audits can target the current code path exactly.
  persistent_tree  replacement-aware, largest-interval-first binary frontiers inside every
                  constant-task run. A collision splits at the observed failing edge; other
                  outcomes split at the midpoint. Each tree interval is proposed at most once,
                  then the exact current gain/Beta sampler takes over.
  lazy_portfolio  keeps one pending persistent-tree proposal and one pending current-gain/Beta
                  proposal.  The ordinary exact cost gate is evaluated for both, then a
                  two-arm Thompson race chooses one collision query using exact global gain,
                  posterior feasibility, and observed verification work.  The losing proposal
                  remains pending and is re-evaluated on the current path.  There is no fixed
                  source allocation or planner/final-pass special case.
  containment_portfolio  the same lazy race, with a container-before-descendant partial order
                  whenever both pending proposals act on the same robot and are nested.

Extended feedback contract (ideation "shared harness contract"):
    sampler.feedback(outcome, dcost=0.0, ij=None, robot=None, fail_k=None)
      outcome in {"gate", "cost", "collision", "accepted"}; dcost = composite cost decrease
      (only for "accepted"); fail_k = absolute index of the first colliding edge if the
      runner captured one.  Proposal-side rejections (short draws) never reach feedback.

The loop is copied verbatim from mr_bench's landed robot_mode_shortcut (candidate
construction, k-1 doubled-switch correction, cost gate, asserts, bookkeeping, stop rule),
so any measured difference is attributable to sampling alone.  Sampler wall time is
accounted separately (propose + feedback + init) for the per-sample-overhead axis.
"""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# the plastic-constant R2 increment (Roberts 2018): g = (1/rho, 1/rho^2)
_R2_G1 = 0.7548776662466927
_R2_G2 = 0.5698402909980532


def _mrb():
    from multi_robot_multi_goal_planning.problems.planning_env import State
    from multi_robot_multi_goal_planning.problems.util import interpolate_path, path_cost
    from multi_robot_multi_goal_planning.planners.shortcutting import (
        remove_interpolated_nodes,
        constant_task_runs,  # the LANDED in-repo run decomposition -- imported, not copied
    )

    return State, interpolate_path, path_cost, remove_interpolated_nodes, constant_task_runs


def preserve_input_nodes(path, tolerance=1e-15):
    """Identity replacement for the unsafe, uncertified cleanup pass.

    Even merging geometrically collinear same-mode edges changes the finite collision-check
    sample grid.  Preserving every certified input edge is therefore the only preprocessing
    behavior that cannot silently alter the path under the checker actually used here.
    ``tolerance`` remains in the signature solely for drop-in monkeypatch compatibility.
    """
    return path


# ---------------------------------------------------------------------------------------
# full-run straight-line gain (the G of RANK 2/3) -- verbatim candidate construction
# ---------------------------------------------------------------------------------------


def _full_run_candidate(env, new_path, r, lo, hi, State):
    q0 = new_path[lo].q
    q1 = new_path[hi].q
    q0r = q0[r] * 1
    diff = (q1[r] * 1 - q0r) / (hi - lo)
    elem = []
    for k in range(hi - lo + 1):
        q = new_path[lo + k].q.state() * 1.0
        r_cnt = 0
        for rr in range(len(env.robots)):
            dim = env.robot_dims[env.robots[rr]]
            if rr == r:
                # we assume that we double the mode switch configurations (verbatim)
                if (
                    k != 0
                    and lo + k != hi
                    and new_path[lo + k].mode != new_path[lo + k - 1].mode
                ):
                    qi = q0r + diff * (k - 1)
                else:
                    qi = q0r + diff * k
                q[r_cnt : r_cnt + dim] = qi
            r_cnt += dim
        elem.append(State(q0.from_flat(q), new_path[lo + k].mode))
    return elem


def _proposal_candidate(env, new_path, i, j, robots_to_shortcut, State):
    """Materialize the landed shortcut construction for an arbitrary proposal.

    This helper is shared by the ordinary loop and the lazy portfolio's cheap pre-collision
    race.  Keeping one implementation is important: the portfolio must rank exactly the
    candidate that the unchanged native cost gate will subsequently consider.
    """
    q0 = new_path[i].q
    q1 = new_path[j].q
    q0_tmp = {}
    q1_tmp = {}
    diff_tmp = {}
    for r in robots_to_shortcut:
        q0_tmp[r] = q0[r] * 1
        q1_tmp[r] = q1[r] * 1
        diff_tmp[r] = (q1_tmp[r] - q0_tmp[r]) / (j - i)

    path_element = []
    for k in range(j - i + 1):
        q = new_path[i + k].q.state() * 1.0
        r_cnt = 0
        for r in range(len(env.robots)):
            dim = env.robot_dims[env.robots[r]]
            if r in robots_to_shortcut:
                # Verbatim doubled mode-switch configuration handling.
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
    return path_element


def _collision_work_bound(path_element, resolution):
    """Cheap candidate-specific proxy for a fully certified native path check.

    The native checker sets this same per-edge interpolation count before checking edges in
    midpoint/binary order.  Infeasible checks usually consume only a fraction of this bound;
    the portfolio learns that fraction separately for each proposal source.
    """
    from multi_robot_multi_goal_planning.problems.util import config_dist

    work = 0
    for k in range(len(path_element) - 1):
        n = int(config_dist(path_element[k].q, path_element[k + 1].q, "max") / resolution) + 1
        work += max(2, n)
    return max(1, int(work))


def full_run_gain(env, new_path, r, lo, hi, State, path_cost) -> float:
    """path_cost(segment) - path_cost(full-run straightened candidate).  Can be negative
    (max-composite cost); callers clamp where needed."""
    elem = _full_run_candidate(env, new_path, r, lo, hi, State)
    return float(
        path_cost(new_path[lo : hi + 1], env.batch_config_cost)
        - path_cost(elem, env.batch_config_cost)
    )


# ---------------------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------------------


class _Stop(Exception):
    pass


class WholePathRandomSampler:
    """The pre-run-decomposition shortcut proposal stream, reproduced verbatim.

    Two indices are uniform over the entire interpolated path.  Only after a nontrivial pair is
    drawn does the old routine choose a robot round-robin; the ordinary task-id gate therefore
    discards most proposals on multi-mode paths.  This is intentionally inefficient and exists
    solely as the historical completely-random control.
    """

    name = "whole_path_random"

    def __init__(self, env, new_path, seed=0, **_):
        self.n = len(new_path)
        self.n_robots = len(env.robots)
        self.rr = 0

    def propose(self):
        i = int(np.random.randint(0, self.n))
        j = int(np.random.randint(0, self.n))
        if i > j:
            i, j = j, i
        if j - i < 2:
            return None
        robot = self.rr % self.n_robots
        self.rr += 1
        return i, j, [robot]

    def feedback(self, outcome, dcost=0.0, ij=None, robot=None, fail_k=None):
        pass

    def report(self):
        return {"whole_path_states": self.n}


class BaselineSampler:
    """The landed in-repo sampler, reproduced exactly (global np.random, same call order)."""

    name = "baseline"

    def __init__(self, env, new_path, seed=0, **_):
        _, _, _, _, constant_task_runs = _mrb()
        self.n_robots = len(env.robots)
        self.runs = [constant_task_runs(new_path, r) for r in range(self.n_robots)]
        self.used = [[0] * len(rs) for rs in self.runs]
        self.rr = 0
        if not any(self.runs):
            self.dead = True
        else:
            self.dead = False

    def _pick_run(self, r):
        rs = self.runs[r]
        us = self.used[r]
        return max(range(len(rs)), key=lambda k: (rs[k][1] - rs[k][0]) / (us[k] + 1))

    def propose(self):
        if self.dead:
            raise _Stop("no eligible runs")
        r = self.rr % self.n_robots
        self.rr += 1
        if not self.runs[r]:
            return None
        k = self._pick_run(r)
        self.used[r][k] += 1
        lo, hi = self.runs[r][k]
        i = np.random.randint(lo, hi + 1)
        j = np.random.randint(lo, hi + 1)
        if i > j:
            i, j = j, i
        if j - i < 2:
            return None
        return i, j, [r]

    def feedback(self, outcome, dcost=0.0, ij=None, robot=None, fail_k=None):
        pass

    def report(self):
        return {"n_runs": [len(rs) for rs in self.runs]}


class R2Sampler(BaselineSampler):
    """Baseline robot/run selection; the in-run pair comes from an R2 lattice per (robot,run)."""

    name = "r2"

    def __init__(self, env, new_path, seed=0, **_):
        super().__init__(env, new_path, seed=seed)
        # per-(robot,run) 2-float state, Cranley-Patterson rotated by its own rng
        self.u = []
        for r in range(self.n_robots):
            row = []
            for k in range(len(self.runs[r])):
                rng = np.random.default_rng(int(seed) + 1000003 * r + 7919 * k)
                row.append(rng.random(2))
            self.u.append(row)
        self.exhausted_retries = 0

    def propose(self):
        if self.dead:
            raise _Stop("no eligible runs")
        r = self.rr % self.n_robots
        self.rr += 1
        if not self.runs[r]:
            return None
        k = self._pick_run(r)
        self.used[r][k] += 1
        lo, hi = self.runs[r][k]
        span = hi - lo
        u = self.u[r][k]
        for _ in range(8):
            u[0] = (u[0] + _R2_G1) % 1.0
            u[1] = (u[1] + _R2_G2) % 1.0
            s1, s2 = (u[0], u[1]) if u[0] <= u[1] else (u[1], u[0])
            p1 = int(np.floor(s1 * span + 0.5))
            p2 = int(np.floor(s2 * span + 0.5))
            if p2 <= p1:
                continue
            i, j = lo + p1, lo + p2
            if j - i < 2:
                continue
            return i, j, [r]
        self.exhausted_retries += 1
        return None

    def report(self):
        d = super().report()
        d["r2_retry_exhausted"] = self.exhausted_retries
        return d


class SweepBisectSampler(BaselineSampler):
    """RANK 2: deterministic G-sweep prefix, then baseline allocator + bisection stacks."""

    name = "sweep_bisect"

    def __init__(self, env, new_path, seed=0, locate_split=False,
                 stack_cap=64, child_min_span=4, split_min_span=4, **_):
        super().__init__(env, new_path, seed=seed)
        State, _, path_cost, _, _ = _mrb()
        self._State, self._path_cost = State, path_cost
        self.env, self.new_path = env, new_path
        self.locate_split = bool(locate_split)
        self.stack_cap = stack_cap
        self.child_min_span = child_min_span
        self.split_min_span = split_min_span
        self.rng = np.random.default_rng(int(seed) + 777)

        # phase-1 queue: (G, r, k, lo, hi) decreasing G, G > 0 only
        sweep = []
        for r in range(self.n_robots):
            for k, (lo, hi) in enumerate(self.runs[r]):
                g = full_run_gain(env, new_path, r, lo, hi, State, path_cost)
                if g > 1e-12:
                    sweep.append((g, r, k, lo, hi))
        sweep.sort(key=lambda t: -t[0])
        self.sweep_queue = sweep
        self.sweep_total = len(sweep)
        # bisection stacks per (robot, run)
        self.stack = [[[] for _ in rs] for rs in self.runs]
        self._last = None  # (r, k) of the last proposal, for feedback routing
        self._last_was_sweep = False
        self.stat = dict(sweep_proposed=0, sweep_accepted=0, pops=0, splits=0,
                         split_located=0, stack_dropped=0)

    def _run_of(self, r, i, j):
        for k, (lo, hi) in enumerate(self.runs[r]):
            if lo <= i and j <= hi:
                return k
        return None

    def propose(self):
        if self.dead:
            raise _Stop("no eligible runs")
        if self.sweep_queue:
            g, r, k, lo, hi = self.sweep_queue.pop(0)
            self.used[r][k] += 1
            self._last = (r, k)
            self._last_was_sweep = True
            self.stat["sweep_proposed"] += 1
            return lo, hi, [r]
        # phase 2: baseline allocator, stacks first
        r = self.rr % self.n_robots
        self.rr += 1
        if not self.runs[r]:
            return None
        k = self._pick_run(r)
        self.used[r][k] += 1
        self._last = (r, k)
        self._last_was_sweep = False
        if self.stack[r][k]:
            i, j = self.stack[r][k].pop()
            self.stat["pops"] += 1
            return i, j, [r]
        lo, hi = self.runs[r][k]
        i = int(self.rng.integers(lo, hi + 1))
        j = int(self.rng.integers(lo, hi + 1))
        if i > j:
            i, j = j, i
        if j - i < 2:
            return None
        return i, j, [r]

    def _push(self, r, k, i, j):
        if j - i >= self.child_min_span:
            if len(self.stack[r][k]) < self.stack_cap:
                self.stack[r][k].append((i, j))
            else:
                self.stat["stack_dropped"] += 1

    def feedback(self, outcome, dcost=0.0, ij=None, robot=None, fail_k=None):
        if outcome == "accepted" and self._last_was_sweep:
            self.stat["sweep_accepted"] += 1
        if outcome == "collision" and ij is not None and robot is not None:
            i, j = ij
            if j - i < self.split_min_span:
                return
            k = self._run_of(robot, i, j)
            if k is None:
                return
            m = i + (j - i) // 2
            if self.locate_split and fail_k is not None and i < fail_k < j:
                m = fail_k
                self.stat["split_located"] += 1
            if m <= i or m >= j:
                m = i + (j - i) // 2
            self.stat["splits"] += 1
            self._push(robot, k, i, m)
            self._push(robot, k, m, j)

    def report(self):
        d = super().report()
        d.update(self.stat)
        d["sweep_total"] = self.sweep_total
        d["stack_left"] = sum(len(s) for rs in self.stack for s in rs)
        return d


class GainPropSampler:
    """RANK 3: global divisor selection by remaining full-run straight-line gain."""

    name = "gainprop"

    def __init__(self, env, new_path, seed=0, **_):
        State, _, path_cost, _, constant_task_runs = _mrb()
        self._State, self._path_cost = State, path_cost
        self.env, self.new_path = env, new_path
        self.n_robots = len(env.robots)
        self.runs = [constant_task_runs(new_path, r) for r in range(self.n_robots)]
        self.used = [[0] * len(rs) for rs in self.runs]
        self.dead = not any(self.runs)
        self.rng = np.random.default_rng(int(seed) + 424243)
        self.G = [[0.0] * len(rs) for rs in self.runs]
        self.arms = [(r, k) for r in range(self.n_robots) for k in range(len(self.runs[r]))]
        allg = []
        for r, k in self.arms:
            lo, hi = self.runs[r][k]
            g = max(0.0, full_run_gain(env, new_path, r, lo, hi, State, path_cost))
            self.G[r][k] = g
            allg.append(g)
        mean_g = float(np.mean(allg)) if allg else 0.0
        self.g0 = 0.05 * mean_g if mean_g > 0 else 1.0
        self.recomputes = 0

    def propose(self):
        if self.dead:
            raise _Stop("no eligible runs")
        best, best_score = None, -np.inf
        for r, k in self.arms:
            score = (self.G[r][k] + self.g0) / (self.used[r][k] + 1)
            if score > best_score:
                best_score, best = score, (r, k)
        r, k = best
        self.used[r][k] += 1
        lo, hi = self.runs[r][k]
        i = int(self.rng.integers(lo, hi + 1))
        j = int(self.rng.integers(lo, hi + 1))
        if i > j:
            i, j = j, i
        if j - i < 2:
            return None
        return i, j, [r]

    def observe_path_change(self, ij):
        """Refresh gains after an accepted edit, including one made by another source."""
        if ij is None:
            return
        i, j = ij
        # composite cost is max-based: an accepted shortcut on one robot changes EVERY
        # overlapping run's remaining gain, so refresh all of them (still O(span) each)
        for r, k in self.arms:
            lo, hi = self.runs[r][k]
            if lo < j and i < hi:
                self.G[r][k] = max(
                    0.0,
                    full_run_gain(self.env, self.new_path, r, lo, hi,
                                  self._State, self._path_cost),
                )
                self.recomputes += 1

    def feedback(self, outcome, dcost=0.0, ij=None, robot=None, fail_k=None):
        if outcome == "accepted":
            self.observe_path_change(ij)

    def report(self):
        return {
            "n_runs": [len(rs) for rs in self.runs],
            "g0": self.g0,
            "recomputes": self.recomputes,
            "G_left": float(sum(sum(row) for row in self.G)),
        }


class GainBetaSampler(GainPropSampler):
    """gainprop with a Beta-posterior acceptance estimate: score =
    G * (1 + accepts) / (2 + proposals) / (used + 1)  + g0 / (used + 1).
    With no observations the estimate is 0.5 for every run, so the initial ranking equals
    gainprop's. Evidence then moves the score: runs whose proposals fail sink, runs whose
    proposals get accepted rise. No tuned constant."""

    name = "gainbeta"

    def __init__(self, env, new_path, seed=0, **kw):
        super().__init__(env, new_path, seed=seed, **kw)
        self.acc = [[0] * len(rs) for rs in self.runs]
        self.prop = [[0] * len(rs) for rs in self.runs]
        self._last = None

    def propose(self):
        if self.dead:
            raise _Stop("no eligible runs")
        best, best_score = None, -np.inf
        for r, k in self.arms:
            p_hat = (1.0 + self.acc[r][k]) / (2.0 + self.prop[r][k])
            score = (self.G[r][k] * p_hat + self.g0) / (self.used[r][k] + 1)
            if score > best_score:
                best_score, best = score, (r, k)
        r, k = best
        self.used[r][k] += 1
        self._last = (r, k)
        lo, hi = self.runs[r][k]
        i = int(self.rng.integers(lo, hi + 1))
        j = int(self.rng.integers(lo, hi + 1))
        if i > j:
            i, j = j, i
        if j - i < 2:
            return None
        self.prop[r][k] += 1
        return i, j, [r]

    def feedback(self, outcome, dcost=0.0, ij=None, robot=None, fail_k=None):
        if outcome == "accepted" and self._last is not None:
            r, k = self._last
            self.acc[r][k] += 1
        super().feedback(outcome, dcost=dcost, ij=ij, robot=robot, fail_k=fail_k)


class CurrentGainBetaSampler(GainBetaSampler):
    """Bit-for-bit proposal stream of the current landed gain/Beta selector.

    ``GainBetaSampler`` predates the landed version and intentionally owns a ``default_rng``.
    The production routine uses global ``np.random.randint``; retaining that detail is required
    for an observational audit to compare its output against the native implementation at a
    fixed seed.
    """

    name = "current_gain_beta"

    def propose(self):
        if self.dead:
            raise _Stop("no eligible runs")
        best, best_score = None, -np.inf
        for r, k in self.arms:
            p_hat = (1.0 + self.acc[r][k]) / (2.0 + self.prop[r][k])
            score = (self.G[r][k] * p_hat + self.g0) / (self.used[r][k] + 1)
            if score > best_score:
                best_score, best = score, (r, k)
        r, k = best
        self.used[r][k] += 1
        self._last = (r, k)
        lo, hi = self.runs[r][k]
        i = int(np.random.randint(lo, hi + 1))
        j = int(np.random.randint(lo, hi + 1))
        if i > j:
            i, j = j, i
        if j - i < 2:
            return None
        self.prop[r][k] += 1
        return i, j, [r]


class PersistentTreeSampler:
    """Cost-agnostic structural prefix that tests containers before descendants.

    The frontier contains intervals, not learned configuration-space regions. Its priority is
    only interval containment/span; the ordinary exact cost gate remains authoritative. Every
    outcome subdivides the interval, so arbitrary scalar costs are supported rather than assuming
    that a cost-rejected parent makes its children useless. Once this finite structural prefix is
    exhausted, the sampler instantiates the current landed gain/Beta selector on the *current*
    path and continues normally.
    """

    name = "persistent_tree"

    def __init__(self, env, new_path, seed=0, tree_only=False, **_):
        _, _, _, _, constant_task_runs = _mrb()
        self.env = env
        self.new_path = new_path
        self.seed = int(seed)
        self.runs = [constant_task_runs(new_path, r) for r in range(len(env.robots))]
        self._rng = np.random.default_rng(self.seed + 1900813)
        self._heap = []
        self._serial = 0
        for r, runs in enumerate(self.runs):
            for lo, hi in runs:
                self._push(r, lo, hi)
        self._last = None
        self._fallback = None
        self._tree_only = bool(tree_only)
        self.stat = {
            "tree_proposals": 0,
            "tree_accepted": 0,
            "tree_collision": 0,
            "tree_cost_reject": 0,
            "tree_splits": 0,
            "tree_failure_splits": 0,
            "fallback_proposals": 0,
        }

    def _push(self, r, i, j):
        if j - i < 2:
            return
        # Random tie-breaking avoids a fixed robot/run bias while preserving the structural
        # rule: a broader interval is considered before every interval it contains.
        tie = float(self._rng.random())
        heapq.heappush(self._heap, (-(j - i), tie, self._serial, int(r), int(i), int(j)))
        self._serial += 1

    def propose(self):
        if self._heap:
            _, _, _, r, i, j = heapq.heappop(self._heap)
            self._last = ("tree", r, i, j)
            self.stat["tree_proposals"] += 1
            return i, j, [r]

        if self._tree_only:
            raise _Stop("tree frontier exhausted")
        if self._fallback is None:
            self._fallback = CurrentGainBetaSampler(
                self.env, self.new_path, seed=self.seed
            )
        self._last = ("fallback",)
        self.stat["fallback_proposals"] += 1
        return self._fallback.propose()

    def feedback(self, outcome, dcost=0.0, ij=None, robot=None, fail_k=None):
        if self._last is None:
            return
        if self._last[0] == "fallback":
            self._fallback.feedback(
                outcome, dcost=dcost, ij=ij, robot=robot, fail_k=fail_k
            )
            return

        _, r, i, j = self._last
        if outcome == "accepted":
            self.stat["tree_accepted"] += 1
        elif outcome == "collision":
            self.stat["tree_collision"] += 1
        elif outcome == "cost":
            self.stat["tree_cost_reject"] += 1

        m = i + (j - i) // 2
        if outcome == "collision" and fail_k is not None and i < fail_k < j:
            m = int(fail_k)
            self.stat["tree_failure_splits"] += 1
        if not i < m < j:
            m = i + (j - i) // 2
        if i < m < j:
            self._push(r, i, m)
            self._push(r, m, j)
            self.stat["tree_splits"] += 1

    def report(self):
        out = dict(self.stat)
        out["tree_frontier_left"] = len(self._heap)
        if self._fallback is not None:
            out.update({"fallback_" + k: v for k, v in self._fallback.report().items()})
        return out


class LazyPortfolioSampler:
    """Lazy, cost-gated race between structural and current proposal sources.

    One proposal from each live source remains pending until it is either rejected by the exact
    native cost gate or receives a collision certificate.  Consequently the unchosen proposal
    is not lost, and an accepted edit merely causes it to be reconstructed and re-scored on the
    updated path.

    For each cost-improving candidate ``c`` from source ``a`` the race samples feasibility
    ``p_a`` from a Jeffreys Beta posterior and ranks

        p_a * exact_global_gain(c) / E[verification_work(c) | p_a].

    Verification work is normalized by the native per-edge interpolation bound.  Source- and
    outcome-specific observed fractions are used when available, then pooled observations, then
    the common bound.  This gives both sources identical, scale-free initial information and
    learns that collision failures are cheap without a fixed mixing probability or time window.
    """

    name = "lazy_portfolio"
    source_order = ("tree", "current")

    def __init__(self, env, new_path, seed=0, resolution=0.01, shared_state=None, **_):
        State, _, path_cost, _, _ = _mrb()
        self._State = State
        self._path_cost = path_cost
        self.env = env
        self.new_path = new_path
        self.resolution = float(resolution)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed + 8675309)
        self.sources = {
            "tree": PersistentTreeSampler(
                env, new_path, seed=self.seed, tree_only=True
            ),
            "current": CurrentGainBetaSampler(env, new_path, seed=self.seed),
        }
        self.live = set(self.source_order)
        self.pending = {}
        self.pending_prepared = {}
        self.path_version = 0
        self._active = None
        self._prepared = None
        self.stat = {
            "portfolio_calls": 0,
            "portfolio_empty_calls": 0,
            "portfolio_generated": {a: 0 for a in self.source_order},
            "portfolio_short_draws": {a: 0 for a in self.source_order},
            "portfolio_cost_rejects": {a: 0 for a in self.source_order},
            "portfolio_gate_rejects": {a: 0 for a in self.source_order},
            "portfolio_chosen": {a: 0 for a in self.source_order},
            "portfolio_checked": {a: 0 for a in self.source_order},
            "portfolio_accepted": {a: 0 for a in self.source_order},
            "portfolio_collisions": {a: 0 for a in self.source_order},
            "portfolio_selection_time": 0.0,
        }
        if shared_state is None:
            shared_state = {}
        self.shared = shared_state
        self.shared.setdefault("accepted", {a: 0 for a in self.source_order})
        self.shared.setdefault("collisions", {a: 0 for a in self.source_order})
        self.shared.setdefault("work", {
            a: {
                "accepted_sum": 0.0, "accepted_n": 0,
                "collision_sum": 0.0, "collision_n": 0,
            }
            for a in self.source_order
        })
        self.work = self.shared["work"]

    def _source_feedback(self, source, outcome, dcost=0.0, ij=None, robot=None,
                         fail_k=None):
        self.sources[source].feedback(
            outcome, dcost=dcost, ij=ij, robot=robot, fail_k=fail_k
        )
        self.pending.pop(source, None)
        self.pending_prepared.pop(source, None)

    def _fill_pending(self):
        for source in self.source_order:
            if source not in self.live or source in self.pending:
                continue
            try:
                proposal = self.sources[source].propose()
            except _Stop:
                self.live.discard(source)
                continue
            if proposal is None:
                self.stat["portfolio_short_draws"][source] += 1
                continue
            self.pending[source] = proposal
            self.stat["portfolio_generated"][source] += 1

    def _prepare(self, source, proposal):
        i, j, robots = proposal
        for r in robots:
            if self.new_path[i].mode.task_ids[r] != self.new_path[j].mode.task_ids[r]:
                self.stat["portfolio_gate_rejects"][source] += 1
                self._source_feedback(source, "gate", ij=(i, j), robot=robots[0])
                return None

        element = _proposal_candidate(
            self.env, self.new_path, i, j, robots, self._State
        )
        candidate_segment_cost = self._path_cost(element, self.env.batch_config_cost)
        old_segment_cost = self._path_cost(
            self.new_path[i : j + 1], self.env.batch_config_cost
        )
        if candidate_segment_cost >= old_segment_cost:
            self.stat["portfolio_cost_rejects"][source] += 1
            self._source_feedback(source, "cost", ij=(i, j), robot=robots[0])
            return None

        segment_gain = float(old_segment_cost - candidate_segment_cost)
        return {
            "source": source,
            "proposal": proposal,
            "path_element": element,
            "candidate_segment_cost": float(candidate_segment_cost),
            "old_segment_cost": float(old_segment_cost),
            # path_cost is the sum of the supplied batch edge costs. Because candidate and old
            # segments have identical endpoints, their exact segment difference is also the
            # exact whole-path marginal gain; no O(path length) second evaluation is needed.
            "segment_gain": segment_gain,
            "global_gain": max(0.0, segment_gain),
            "work_bound": _collision_work_bound(element, self.resolution),
        }

    def _pooled_rate(self, outcome):
        key_sum, key_n = outcome + "_sum", outcome + "_n"
        total = sum(self.work[a][key_sum] for a in self.source_order)
        count = sum(self.work[a][key_n] for a in self.source_order)
        if count:
            return total / count
        # In particular, never infer the cost of a successful full certificate from a cheap
        # early collision. Until that outcome has been observed, use the known full-work bound.
        return 1.0

    def _rate(self, source, outcome):
        n = self.work[source][outcome + "_n"]
        if n:
            return self.work[source][outcome + "_sum"] / n
        return self._pooled_rate(outcome)

    def _priority_choice(self, prepared):
        """Optional proposal partial order; the base portfolio has none."""
        return None

    def propose(self):
        if self._active is not None:
            raise RuntimeError("portfolio requested a new proposal before feedback")
        self.stat["portfolio_calls"] += 1
        self._fill_pending()
        if not self.pending and not self.live:
            raise _Stop("both portfolio sources exhausted")
        if not self.pending:
            self.stat["portfolio_empty_calls"] += 1
            return None

        t0 = time.perf_counter()
        prepared = []
        # _prepare may remove cost-rejected pending entries, hence iterate over a snapshot.
        for source, proposal in list(self.pending.items()):
            cached = self.pending_prepared.get(source)
            if cached is not None and cached[0] == self.path_version:
                item = cached[1]
            else:
                item = self._prepare(source, proposal)
                if item is not None:
                    self.pending_prepared[source] = (self.path_version, item)
            if item is not None:
                prepared.append(item)
        if not prepared:
            self.stat["portfolio_empty_calls"] += 1
            self.stat["portfolio_selection_time"] += time.perf_counter() - t0
            return None

        item = self._priority_choice(prepared)
        if item is None:
            best = None
            for candidate in prepared:
                source = candidate["source"]
                accepted = self.shared["accepted"][source]
                collisions = self.shared["collisions"][source]
                # Jeffreys' prior is invariant under Bernoulli reparameterization and
                # contributes no policy-specific tuning constant.
                p_free = float(self.rng.beta(accepted + 0.5, collisions + 0.5))
                expected_fraction = (
                    p_free * self._rate(source, "accepted")
                    + (1.0 - p_free) * self._rate(source, "collision")
                )
                expected_work = max(
                    1e-12, candidate["work_bound"] * expected_fraction
                )
                candidate["p_free_draw"] = p_free
                candidate["expected_work"] = expected_work
                candidate["score"] = (
                    p_free * candidate["global_gain"] / expected_work
                )
                # If both candidates have zero immediate composite gain, the native segment
                # gate still says they are useful retimings. Segment gain is a scale-free tie.
                candidate["tie_score"] = candidate["segment_gain"] / expected_work
                rank = (
                    candidate["score"], candidate["tie_score"],
                    float(self.rng.random()),
                )
                if best is None or rank > best[0]:
                    best = (rank, candidate)
            item = best[1]
        source = item["source"]
        self._active = source
        self._prepared = item
        self.stat["portfolio_chosen"][source] += 1
        self.stat["portfolio_selection_time"] += time.perf_counter() - t0
        return item["proposal"]

    def prepared_candidate(self):
        """Return the already materialized winning candidate for the ordinary loop."""
        return self._prepared

    def observe_check(self, free, config_checks=None, collision_wall_s=None):
        if self._active is None or self._prepared is None:
            return
        source = self._active
        bound = float(self._prepared["work_bound"])
        if config_checks is not None and config_checks > 0:
            fraction = max(0.0, float(config_checks)) / bound
        else:
            # Do not mix seconds and configuration counts. Runners that want learned checking
            # work pass the observational counter; without it the conservative full bound is
            # still a valid, policy-neutral selector.
            fraction = 1.0
        outcome = "accepted" if free else "collision"
        self.work[source][outcome + "_sum"] += fraction
        self.work[source][outcome + "_n"] += 1
        self.stat["portfolio_checked"][source] += 1

    def feedback(self, outcome, dcost=0.0, ij=None, robot=None, fail_k=None):
        if self._active is None:
            return
        source = self._active
        if outcome == "accepted":
            self.stat["portfolio_accepted"][source] += 1
            self.shared["accepted"][source] += 1
        elif outcome == "collision":
            self.stat["portfolio_collisions"][source] += 1
            self.shared["collisions"][source] += 1
        self._source_feedback(
            source, outcome, dcost=dcost, ij=ij, robot=robot, fail_k=fail_k
        )
        if outcome == "accepted" and source != "current":
            # The current selector shares the mutable path but otherwise would not learn that
            # the tree changed gains beneath its pending proposal.
            self.sources["current"].observe_path_change(ij)
        if outcome == "accepted":
            self.path_version += 1
            self.pending_prepared.clear()
        self._active = None
        self._prepared = None

    def report(self):
        out = dict(self.stat)
        out["portfolio_live"] = sorted(self.live)
        out["portfolio_pending"] = sorted(self.pending)
        out["portfolio_work"] = {
            source: dict(values) for source, values in self.work.items()
        }
        out["portfolio_shared_accepted"] = dict(self.shared["accepted"])
        out["portfolio_shared_collisions"] = dict(self.shared["collisions"])
        for source in self.source_order:
            child = self.sources[source].report()
            out.update({source + "_" + key: value for key, value in child.items()})
        return out


class ContainmentPortfolioSampler(LazyPortfolioSampler):
    """Lazy portfolio plus container-before-descendant precedence.

    If its two cost-improving pending candidates act on the same robot and one interval strictly
    contains the other, certifying the descendant first risks doing work that a later accepted
    container entirely replaces.  The containing candidate therefore wins this local partial
    order.  Disjoint, crossing, and different-robot proposals retain the ordinary Thompson
    expected-gain-rate decision.  No source quota, interval scale, or timing threshold is added.
    """

    name = "containment_portfolio"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat["portfolio_containment_choices"] = 0
        self.stat["portfolio_containment_source"] = {
            source: 0 for source in self.source_order
        }

    def _priority_choice(self, prepared):
        if len(prepared) != 2:
            return None
        a, b = prepared
        ai, aj, ar = a["proposal"]
        bi, bj, br = b["proposal"]
        if len(ar) != 1 or ar != br:
            return None
        chosen = None
        if ai <= bi and bj <= aj and (ai < bi or bj < aj):
            chosen = a
        elif bi <= ai and aj <= bj and (bi < ai or aj < bj):
            chosen = b
        if chosen is not None:
            self.stat["portfolio_containment_choices"] += 1
            self.stat["portfolio_containment_source"][chosen["source"]] += 1
        return chosen


class PairSampler:
    """D'Hondt allocation over intersections of two robots' constant-task runs.

    A multi-robot candidate is valid only when *both* task IDs agree at its endpoints.
    Intersecting the existing per-robot runs enforces that structural gate before the expensive
    candidate construction.  The ordinary shortcut loop still applies its task-id guard and
    the unchanged native full-path collision certificate.
    """

    name = "pairs"

    def __init__(self, env, new_path, seed=0, **_):
        _, _, _, _, constant_task_runs = _mrb()
        runs = [constant_task_runs(new_path, r) for r in range(len(env.robots))]
        self.arms = []  # (robot_a, robot_b, lo, hi), inclusive indices
        for a in range(len(env.robots)):
            for b in range(a + 1, len(env.robots)):
                for lo_a, hi_a in runs[a]:
                    for lo_b, hi_b in runs[b]:
                        lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
                        if hi - lo >= 2:
                            self.arms.append((a, b, lo, hi))
        self.used = [0] * len(self.arms)
        self.dead = not self.arms

    def propose(self):
        if self.dead:
            raise _Stop("no pairwise eligible runs")
        k = max(
            range(len(self.arms)),
            key=lambda ix: (self.arms[ix][3] - self.arms[ix][2]) / (self.used[ix] + 1),
        )
        self.used[k] += 1
        a, b, lo, hi = self.arms[k]
        i = np.random.randint(lo, hi + 1)
        j = np.random.randint(lo, hi + 1)
        if i > j:
            i, j = j, i
        if j - i < 2:
            return None
        return i, j, [a, b]

    def feedback(self, outcome, dcost=0.0, ij=None, robot=None, fail_k=None):
        pass

    def report(self):
        return {"pair_arms": len(self.arms)}


class PairThenGainSampler:
    """Use an early pair phase, then gainprop single-arm cleanup.

    Pair candidates cheaply remove coupled slack while the path is rough.  After
    ``pair_checks`` native collision validations, gainprop focuses the remaining budget on the
    individual runs with the most residual composite gain.  Both samplers share ``new_path``;
    gainprop is explicitly refreshed after accepted pairs so its scores cannot go stale.
    """

    name = "pair_then_gain"

    def __init__(self, env, new_path, seed=0, pair_checks=200, **_):
        self.pairs = PairSampler(env, new_path, seed=seed)
        self.gain = GainPropSampler(env, new_path, seed=seed)
        self.pair_limit = int(pair_checks)
        self.pair_checks = 0
        self.last = None

    def propose(self):
        if self.pair_checks < self.pair_limit and not self.pairs.dead:
            self.last = "pair"
            return self.pairs.propose()
        self.last = "gain"
        return self.gain.propose()

    def feedback(self, outcome, dcost=0.0, ij=None, robot=None, fail_k=None):
        if self.last == "pair":
            self.pairs.feedback(outcome, dcost=dcost, ij=ij, robot=robot, fail_k=fail_k)
            # `new_path` has already been mutated by the loop when this is called.  An accepted
            # pair changes every overlapping single-run gain under the max-composite objective.
            self.gain.feedback(outcome, dcost=dcost, ij=ij, robot=robot, fail_k=fail_k)
            if outcome in ("accepted", "collision"):
                self.pair_checks += 1
            return
        self.gain.feedback(outcome, dcost=dcost, ij=ij, robot=robot, fail_k=fail_k)

    def report(self):
        d = {"pair_checks": self.pair_checks, "pair_limit": self.pair_limit}
        d.update({"pair_" + k: v for k, v in self.pairs.report().items()})
        d.update({"gain_" + k: v for k, v in self.gain.report().items()})
        return d


SAMPLERS = {
    "whole_path_random": WholePathRandomSampler,
    "baseline": BaselineSampler,
    "r2": R2Sampler,
    "gainbeta": GainBetaSampler,
    "current_gain_beta": CurrentGainBetaSampler,
    "persistent_tree": PersistentTreeSampler,
    "lazy_portfolio": LazyPortfolioSampler,
    "containment_portfolio": ContainmentPortfolioSampler,
    "sweep_bisect": lambda env, p, **kw: SweepBisectSampler(env, p, locate_split=False, **kw),
    "sweep_bisect_loc": lambda env, p, **kw: SweepBisectSampler(env, p, locate_split=True, **kw),
    "gainprop": GainPropSampler,
    "pairs": PairSampler,
    "pair_then_gain": PairThenGainSampler,
}


# ---------------------------------------------------------------------------------------
# the loop -- verbatim landed mr_bench semantics, extended feedback, sampler timing
# ---------------------------------------------------------------------------------------


def shortcut_run(
    env,
    path,
    arm: str,
    max_iter: int = 1000,
    max_attempts: Optional[int] = None,
    resolution=0.01,
    tolerance=0.01,
    interpolation_resolution: float = 0.5,
    seed: int = 0,
    pair_checks: int = 200,
    fail_cell: Optional[dict] = None,
    stats_out: Optional[Dict] = None,
    audit_out: Optional[Dict] = None,
    check_counter=None,
    shared_state=None,
):
    """Same contract as robot_mode_shortcut: returns (new_path, [costs, times]).

    ``audit_out`` enables observational replacement instrumentation.  It does not alter
    proposal, cost-gate, collision, or acceptance decisions.  ``check_counter`` may be the
    instance wrapper from :mod:`benchenv`; when supplied, per-candidate configuration/edge
    counts are recorded without issuing any additional collision queries.
    """
    State, interpolate_path, path_cost, remove_interpolated_nodes, _ = _mrb()

    non_redundant_path = remove_interpolated_nodes(path)
    new_path = interpolate_path(non_redundant_path, interpolation_resolution)

    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]

    st: Dict = dict(
        arm=arm, n_states=len(new_path), iters=0, skipped_short=0, skipped_gate=0,
        cost_rejects=0, collision_checks=0, accepted=0, rejected_collision=0,
        sampler_time=0.0, sampler_calls=0, cost_initial=float(costs[0]), stop_reason=None,
        accepted_per_robot={}, fail_located=0,
    )
    if audit_out is not None:
        # Lazy import keeps the ordinary monkeypatch module standalone; several existing tests
        # load this file directly through importlib without placing its directory on sys.path.
        from replacement_audit import ReplacementAudit
        audit = ReplacementAudit(check_counter)
    else:
        audit = None

    t_s = time.perf_counter()
    sampler = SAMPLERS[arm](
        env, new_path, seed=seed, pair_checks=pair_checks, resolution=resolution,
        shared_state=shared_state,
    )
    st["sampler_time"] += time.perf_counter() - t_s

    start_time = time.time()
    if max_attempts is None:
        max_attempts = 5 * max_iter
    cnt = 0
    iter = 0

    while True:
        iter += 1
        if cnt >= max_iter or iter >= max_attempts:
            st["stop_reason"] = "max_iter" if cnt >= max_iter else "max_attempts"
            break

        t_s = time.perf_counter()
        try:
            proposal = sampler.propose()
        except _Stop as e:
            st["sampler_time"] += time.perf_counter() - t_s
            st["stop_reason"] = f"sampler exhausted: {e}"
            break
        st["sampler_time"] += time.perf_counter() - t_s
        st["sampler_calls"] += 1
        if proposal is None:
            st["skipped_short"] += 1
            continue
        i, j, robots_to_shortcut = proposal
        cost_before = costs[-1]

        # ---- verbatim: task-id gate ----------------------------------------------------
        can_shortcut_this = True
        for r in robots_to_shortcut:
            if new_path[i].mode.task_ids[r] != new_path[j].mode.task_ids[r]:
                can_shortcut_this = False
                break
        if not can_shortcut_this:
            st["skipped_gate"] += 1
            t_s = time.perf_counter()
            sampler.feedback("gate", ij=(i, j), robot=robots_to_shortcut[0])
            st["sampler_time"] += time.perf_counter() - t_s
            continue

        # ---- verbatim: candidate construction ------------------------------------------
        q0 = new_path[i].q
        q1 = new_path[j].q
        prepared = (
            sampler.prepared_candidate()
            if hasattr(sampler, "prepared_candidate") else None
        )
        if (
            prepared is not None
            and prepared["proposal"] == proposal
        ):
            path_element = prepared["path_element"]
            candidate_segment_cost = prepared["candidate_segment_cost"]
            old_segment_cost = prepared["old_segment_cost"]
        else:
            path_element = _proposal_candidate(
                env, new_path, i, j, robots_to_shortcut, State
            )
            candidate_segment_cost = path_cost(path_element, env.batch_config_cost)
            old_segment_cost = path_cost(new_path[i : j + 1], env.batch_config_cost)

        # ---- verbatim: cost gate -------------------------------------------------------
        if candidate_segment_cost >= old_segment_cost:
            st["cost_rejects"] += 1
            t_s = time.perf_counter()
            sampler.feedback("cost", ij=(i, j), robot=robots_to_shortcut[0])
            st["sampler_time"] += time.perf_counter() - t_s
            continue

        assert np.linalg.norm(path_element[0].q.state() - q0.state()) < 1e-6
        assert np.linalg.norm(path_element[-1].q.state() - q1.state()) < 1e-6

        cnt += 1
        st["collision_checks"] += 1

        if fail_cell is not None:
            fail_cell["q"] = None
        before_counts = audit.snapshot() if audit is not None else None
        portfolio_configs_before = (
            int(check_counter.configs)
            if check_counter is not None and hasattr(check_counter, "configs") else None
        )
        collision_t0 = time.perf_counter()
        free = env.is_path_collision_free(
            path_element, resolution=resolution, tolerance=tolerance,
            check_start_and_end=False,
        )
        collision_wall_s = time.perf_counter() - collision_t0
        after_counts = audit.snapshot() if audit is not None else None
        portfolio_config_checks = None
        if portfolio_configs_before is not None:
            portfolio_config_checks = int(check_counter.configs) - portfolio_configs_before

        fail_k = None
        if not free and fail_cell is not None and fail_cell.get("q") is not None:
            fq = fail_cell["q"]
            for k in range(j - i + 1):
                if path_element[k].q is fq:
                    fail_k = i + k
                    st["fail_located"] += 1
                    break

        descriptor = None
        if audit is not None:
            descriptor = audit.describe_candidate(
                new_path, path_element, i, j, robots_to_shortcut
            )

        if free:
            for k in range(j - i + 1):
                new_path[i + k].q = path_element[k].q
            st["accepted"] += 1
            for r in robots_to_shortcut:
                st["accepted_per_robot"][r] = st["accepted_per_robot"].get(r, 0) + 1
        else:
            st["rejected_collision"] += 1

        current_time = time.time()
        times.append(current_time - start_time)
        costs.append(path_cost(new_path, env.batch_config_cost))

        # ---- extended feedback ---------------------------------------------------------
        t_s = time.perf_counter()
        if hasattr(sampler, "observe_check"):
            sampler.observe_check(
                free,
                config_checks=portfolio_config_checks,
                collision_wall_s=collision_wall_s,
            )
        if free:
            sampler.feedback("accepted", dcost=float(cost_before - costs[-1]),
                             ij=(i, j), robot=robots_to_shortcut[0])
        else:
            sampler.feedback("collision", ij=(i, j), robot=robots_to_shortcut[0],
                             fail_k=fail_k)
        st["sampler_time"] += time.perf_counter() - t_s

        if audit is not None:
            # Remove instrumentation time already incurred from the reported algorithmic
            # elapsed time.  Collision time itself is measured directly around the native call.
            audit.record_check(
                descriptor,
                free=free,
                proposed_gain=float(old_segment_cost - candidate_segment_cost),
                accepted_gain=float(cost_before - costs[-1]) if free else 0.0,
                collision_wall_s=collision_wall_s,
                before_counts=before_counts,
                after_counts=after_counts,
                elapsed_s=float(current_time - start_time - audit.overhead_s),
                check_index=st["collision_checks"],
                fail_k=fail_k,
            )

    st["iters"] = iter
    st["wall_time"] = time.time() - start_time
    st["cost_final"] = float(costs[-1])
    st.update(sampler.report())

    if audit is not None:
        audit_out.clear()
        audit_out.update(audit.finalize(
            elapsed_s=float(time.time() - start_time - audit.overhead_s),
            check_index=st["collision_checks"],
        ))

    if stats_out is not None:
        stats_out.clear()
        stats_out.update(st)

    # ---- verbatim: final asserts -------------------------------------------------------
    assert new_path[-1].mode == path[-1].mode
    assert np.linalg.norm(new_path[-1].q.state() - path[-1].q.state()) < 1e-6
    assert np.linalg.norm(new_path[0].q.state() - path[0].q.state()) < 1e-6

    return new_path, [costs, times]
