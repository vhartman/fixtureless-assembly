"""Round-3 shortcut arms on top of the ideation-round samplers.  New file only.

Nothing in ``ideation_samplers.py`` or ``sliding_shortcut.py`` is modified, and mr_bench is
still only imported and monkeypatched.

Arms (all share the same verbatim candidate construction, task-id gate, cost semantics,
collision certificate, and stop rule as the landed loop; the v2 loop only replaces WHERE the
cost gate is computed and adds optional, pre-registered stopping rules):

  tree_v2          ``persistent_tree`` proposal stream + the exact vectorized gate.  Gate
                   decisions are identical to the native construction gate (verified with the
                   eps fallback), so this arm must produce the same accepted-edit set as
                   ``persistent_tree`` at a fixed seed; it just builds fewer candidates.
  tree_adaptive    ``tree_v2`` + adaptive stop of the fallback: stop when 40 validations in
                   fallback mode produce no acceptance.  The finite tree prefix always
                   completes first, which is the coverage guarantee.
  tree_partial     bounded structural prefix: at most 25 tree proposals per shortcut call,
                   then the gain/Beta fallback.  Targets the full-loop startup lag.
  tree_warm        ``tree_partial`` plus a once-per-planner-run warm flag: after the first
                   call has spent its prefix budget, later calls run the fallback only.
  tree_credit      cadence arm: calls with ``max_iter <= 100`` (the planner's internal
                   shortcut-and-reinject calls) run at most 25 validations; the final pass
                   keeps its full budget.  Applied by the driver, not inside the loop.
  length_uniform   gain/Beta run selection; the in-run pair is drawn with a uniformly random
                   span instead of two uniform indices (which bias toward short spans).
  length_tilted    same, with span density proportional to span**tilt (tilt = 2).
  max_compatible   composite candidate type: at a drawn pair, shortcut every robot whose task
                   stays constant across the interval (the IROS composite idea adapted to the
                   constant-task gate).  The exact gate handles the robot set.

Every v2 arm also records an irreplaceability diagnostic from the WAFR paper: the remaining
exact full-run straight-line gain summed over all (robot, run) arms, relative to the final
cost.  It is observational; no decision uses it.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import ideation_samplers as IS
from exact_gate import SegmentCostTable

try:
    from ideation_samplers import _Stop, _proposal_candidate, _mrb
except ImportError:  # pragma: no cover
    from .ideation_samplers import _Stop, _proposal_candidate, _mrb  # type: ignore


ADAPTIVE_STOP_K = 40
PREFIX_CAP = 25
CREDIT_INTERNAL_CHECKS = 25


# ---------------------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------------------


class CappedTreeSampler(IS.PersistentTreeSampler):
    """``persistent_tree`` with a per-call prefix cap and an optional warm flag.

    The warm flag lives in the driver's per-run shared state, so it persists across the
    planner's internal shortcut calls within one process.
    """

    name = "capped_tree"

    def __init__(self, env, new_path, seed=0, tree_proposal_cap=None, warm=False,
                 shared_state=None, warm_key="v5_prefix_done", tree_only=False, **_):
        super().__init__(env, new_path, seed=seed, tree_only=tree_only)
        self.cap = tree_proposal_cap
        self.warm = bool(warm)
        self.shared = shared_state if shared_state is not None else {}
        self.warm_key = warm_key
        self.tree_count = 0
        self.stat["tree_cap_hit"] = False
        self.stat["tree_warm_skip"] = False

    def _fallback_propose(self):
        if self._fallback is None:
            self._fallback = IS.CurrentGainBetaSampler(
                self.env, self.new_path, seed=self.seed
            )
        self._last = ("fallback",)
        self.stat["fallback_proposals"] += 1
        return self._fallback.propose()

    def propose(self):
        if self.warm and self.shared.get(self.warm_key):
            self.stat["tree_warm_skip"] = True
            return self._fallback_propose()
        if self.cap is not None and self.tree_count >= self.cap:
            self.stat["tree_cap_hit"] = True
            if self.warm:
                self.shared[self.warm_key] = True
            return self._fallback_propose()
        proposal = super().propose()
        if proposal is not None and self._last[0] == "tree":
            self.tree_count += 1
        elif self._last[0] == "fallback" and self.warm:
            self.shared[self.warm_key] = True
        return proposal


class SimpleTreeSampler(CappedTreeSampler):
    """``tree_simple``: the structural prefix with ONE splitting rule.

    On a collision with a located failing edge the interval splits there; the children keep
    the exact cost gate.  Every other outcome DROPS the interval: no midpoint splits on cost
    rejects or unlocated collisions.  Minimal-diff variant to quantify how much of the full
    tree's final-cost edge comes from the midpoint-splitting refinement.
    """

    name = "tree_simple"

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
        if outcome == "collision" and fail_k is not None and i < fail_k < j:
            self.stat["tree_failure_splits"] += 1
            self._push(r, i, int(fail_k))
            self._push(r, int(fail_k), j)
            self.stat["tree_splits"] += 1
        # every other outcome drops the interval; no midpoint refinement


class RandomFallbackTreeSampler(SimpleTreeSampler):
    """``tree_rand``: the tree_simple prefix with a PLAIN random fallback.

    After the prefix, pairs come from the landed span selector (round-robin robot, longest-run
    divisor, uniform pair inside the run): no remaining-gain scoring, no Beta posterior.
    Quantifies whether the gain/Beta machinery is still required once the prefix has run.
    """

    name = "tree_rand"

    def propose(self):
        # NOTE: PersistentTreeSampler.propose hardcodes CurrentGainBetaSampler when the
        # frontier empties, so the plain prefix path never reaches _fallback_propose.
        # This override routes the exhausted frontier to the plain span fallback instead.
        if self._heap:
            return super().propose()
        return self._fallback_propose()

    def _fallback_propose(self):
        if self._fallback is None:
            self._fallback = IS.BaselineSampler(
                self.env, self.new_path, seed=self.seed
            )
        self._last = ("fallback",)
        self.stat["fallback_proposals"] += 1
        return self._fallback.propose()


class Split2TreeSampler(SimpleTreeSampler):
    """``tree_split2``: the tree_simple prefix with each initial run pre-split in half.

    Every maximal constant-task run enters the frontier as its two halves instead of the
    whole run; everything else is tree_simple (split at the located failing edge only, no
    midpoint splits, native fallback).  Tests whether skipping the whole-run proposal
    changes the final cost or only saves its (cheap) collision check.
    """

    name = "tree_split2"

    def __init__(self, env, new_path, seed=0, shared_state=None, **kwargs):
        super().__init__(env, new_path, seed=seed, shared_state=shared_state, **kwargs)
        # re-seed the frontier with halves instead of whole runs
        self._heap = []
        for r, runs in enumerate(self.runs):
            for lo, hi in runs:
                m = lo + (hi - lo) // 2
                self._push(r, lo, m)
                self._push(r, m, hi)


class CappedSeedTreeSampler(RandomFallbackTreeSampler):
    """``tree_cap``: tree_simple + random fallback, but the initial runs enter the
    frontier as cap-sized chunks instead of one whole-run proposal.

    The whole-run proposal on very long runs almost surely collides, which is why the tree
    curve starts flat: the first validations buy no acceptance.  Seeding cap-sized chunks
    keeps the broad-first structural order while the first proposals are of a size that can
    actually accept.
    """

    name = "tree_cap"

    def __init__(self, env, new_path, seed=0, shared_state=None, seed_cap=40, **kwargs):
        super().__init__(env, new_path, seed=seed, shared_state=shared_state, **kwargs)
        cap = int(seed_cap)
        self._heap = []
        for r, runs in enumerate(self.runs):
            for lo, hi in runs:
                k = lo
                while k < hi:
                    nxt = min(hi, k + cap)
                    self._push(r, k, nxt)
                    k = nxt


class MixTreeSampler(SimpleTreeSampler):
    """``tree_mix``: two levers — the tree frontier and the native gain/Beta selector —
    and a Bernoulli draw per proposal picks the source.

    No phase boundary: the native lever contributes its feasible acceptances from the very
    first validation (the early drop), while the tree drains its coarse structural items in
    parallel.  After the frontier is empty only the native lever remains.
    """

    name = "tree_mix"

    def __init__(self, env, new_path, seed=0, shared_state=None, mix_p=0.5, **kwargs):
        super().__init__(env, new_path, seed=seed, shared_state=shared_state, **kwargs)
        self.mix_p = float(mix_p)
        self._mix_rng = np.random.default_rng(int(seed) + 770011)

    def propose(self):
        if self._heap and self._mix_rng.random() < self.mix_p:
            return super().propose()
        return self._fallback_propose()

    def _fallback_propose(self):
        if self._fallback is None:
            self._fallback = IS.CurrentGainBetaSampler(
                self.env, self.new_path, seed=self.seed
            )
        self._last = ("fallback",)
        self.stat["fallback_proposals"] += 1
        return self._fallback.propose()


class WarmBetaTreeSampler(RandomFallbackTreeSampler):
    """``tree_warmbeta``: a Beta warmup of W validations, then the tree prefix on the
    current path, then the plain random fallback.

    Directly combines the native early drop (small feasible acceptances) with the tree's
    structural coarse-to-fine prefix and the cheap random fallback.
    """

    name = "tree_warmbeta"

    def __init__(self, env, new_path, seed=0, shared_state=None,
                 warm_validations=50, **kwargs):
        super().__init__(env, new_path, seed=seed, shared_state=shared_state, **kwargs)
        self.phase = "beta"
        self.warm_validations = int(warm_validations)
        self.beta = IS.CurrentGainBetaSampler(env, new_path, seed=self.seed)
        self.beta_count = 0

    def _enter_tree(self):
        _, _, _, _, constant_task_runs = _mrb()
        self.runs = [constant_task_runs(self.new_path, r)
                     for r in range(len(self.env.robots))]
        self._heap = []
        for r, runs in enumerate(self.runs):
            for lo, hi in runs:
                self._push(r, lo, hi)
        self.phase = "tree"

    def propose(self):
        if self.phase == "beta":
            if self.beta_count < self.warm_validations:
                self._last = ("beta",)
                return self.beta.propose()
            self._enter_tree()
        return super().propose()

    def feedback(self, outcome, dcost=0.0, ij=None, robot=None, fail_k=None):
        if self.phase == "beta" and self._last is not None and self._last[0] == "beta":
            if outcome in ("accepted", "collision"):
                self.beta_count += 1
            self.beta.feedback(
                outcome, dcost=dcost, ij=ij, robot=robot, fail_k=fail_k
            )
            return
        super().feedback(outcome, dcost=dcost, ij=ij, robot=robot, fail_k=fail_k)


class LengthGainSampler(IS.GainBetaSampler):
    """gain/Beta run selection with a pre-registered in-run span distribution.

    ``tilt=None`` draws the span uniformly over [2, span]; otherwise the span density is
    proportional to ``span**tilt`` (continuous inverse-transform approximation, pre-registered).
    The start is uniform over the remaining positions.
    """

    name = "length_uniform"

    def __init__(self, env, new_path, seed=0, tilt=None, **_):
        super().__init__(env, new_path, seed=seed)
        self.tilt = float(tilt) if tilt is not None else None

    def _draw_pair(self, lo, hi):
        span = hi - lo
        if span < 2:
            return None
        if self.tilt is None:
            length = int(self.rng.integers(2, span + 1))
        else:
            u = float(self.rng.random())
            length = 2 + int(round((span - 2) * (u ** (1.0 / (self.tilt + 1.0)))))
            length = min(max(length, 2), span)
        start = int(self.rng.integers(0, span - length + 1))
        return lo + start, lo + start + length

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
        pair = self._draw_pair(*self.runs[r][k])
        if pair is None:
            return None
        self.prop[r][k] += 1
        i, j = pair
        return i, j, [r]


class LengthTiltedSampler(LengthGainSampler):
    name = "length_tilted"


class MaxCompatibleSampler(IS.GainBetaSampler):
    """gain/Beta pair proposal, but the candidate shortcuts every robot whose task is
    constant across the drawn interval.  The unchanged task-id gate accepts the whole set,
    and the exact vectorized gate handles multi-robot candidates."""

    name = "max_compatible"

    def __init__(self, env, new_path, seed=0, **_):
        super().__init__(env, new_path, seed=seed)
        self.modes = [
            tuple(int(x) for x in np.atleast_1d(s.mode.task_ids)) for s in new_path
        ]

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
        robots = [rr for rr in range(self.n_robots) if self.modes[i][rr] == self.modes[j][rr]]
        return i, j, robots


class MaxBanditSampler(MaxCompatibleSampler):
    """``max_compatible`` with a two-arm Thompson choice of the candidate TYPE.

    Per proposal, both candidate types share the drawn pair (i, j): the full compatible set
    and the drawn-robot singleton.  The exact gate gives the gain of each type.  The type is
    chosen by a Jeffreys Thompson draw over pooled acceptance/collision evidence per type
    (shared across the planner run), ranking ``theta_type * exact_gain``.  The full set has
    the highest gain; the singleton has the higher acceptance probability; the draw learns
    which trade wins in each environment.  No window, percentage, or threshold is used, and
    a type with a non-positive exact gain is never chosen over one with a positive gain."""

    name = "max_bandit"

    def __init__(self, env, new_path, seed=0, shared_state=None, **_):
        super().__init__(env, new_path, seed=seed)
        from exact_gate import SegmentCostTable

        self.table = SegmentCostTable(env, lambda: self.new_path)
        self.shared = shared_state if shared_state is not None else {}
        self.shared.setdefault("max_bandit_evidence", {
            "full": [0.5, 0.5], "single": [0.5, 0.5],
        })
        self.evidence = self.shared["max_bandit_evidence"]
        self._last_type = None
        self.stat = {}
        self.stat["bandit_table_ok"] = bool(self.table.ok)
        self.stat["bandit_full"] = 0
        self.stat["bandit_single"] = 0
        self.stat["bandit_type_fallback"] = 0

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
        full = [rr for rr in range(self.n_robots) if self.modes[i][rr] == self.modes[j][rr]]
        if len(full) <= 1:
            self._last_type = "full" if full else "single"
            return i, j, full
        chosen = self._choose_type(i, j, full, r)
        if chosen == "single":
            self.stat["bandit_single"] += 1
            self._last_type = "single"
            return i, j, [r]
        self.stat["bandit_full"] += 1
        self._last_type = "full"
        return i, j, full

    def _choose_type(self, i, j, full, r):
        """Pick the candidate type with the higher Thompson score theta * exact gain."""
        if not self.table.ensure():
            self.stat["bandit_type_fallback"] += 1
            return "full"
        g_full = self.table.delta(full, i, j)
        g_single = self.table.delta([r], i, j)
        # A non-positive-gain type is dominated: the gate would reject it.
        if g_single <= 0.0 and g_full > 0.0:
            return "full"
        if g_full <= 0.0 and g_single > 0.0:
            return "single"
        if g_full <= 0.0 and g_single <= 0.0:
            return "full"  # the loop's gate rejects this proposal either way
        a_full, b_full = self.evidence["full"]
        a_single, b_single = self.evidence["single"]
        theta_full = float(self.rng.beta(a_full, b_full))
        theta_single = float(self.rng.beta(a_single, b_single))
        if theta_single * g_single > theta_full * g_full:
            return "single"
        return "full"

    def feedback(self, outcome, dcost=0.0, ij=None, robot=None, fail_k=None):
        if outcome in ("accepted", "collision") and self._last_type is not None:
            entry = self.evidence[self._last_type]
            if outcome == "accepted":
                entry[0] += 1.0
            else:
                entry[1] += 1.0
        if outcome == "accepted":
            self.table.mark_dirty()
        super().feedback(outcome, dcost=dcost, ij=ij, robot=robot, fail_k=fail_k)

    def report(self):
        out = super().report()
        out.update(self.stat)
        out["bandit_evidence"] = {k: list(v) for k, v in self.evidence.items()}
        return out


_SAMPLER_KWARGS = {
    "tree_v2": {},
    "tree_adaptive": {},
    "tree_credit": {},
    "tree_simple": {},
    "tree_split2": {},
    "tree_rand": {},
    "tree_cap": {"seed_cap": 40},
    "tree_warmbeta": {"warm_validations": 50},
    "tree_mix": {"mix_p": 0.5},
    "tree_only": {"tree_only": True},
    "tree_partial": {"tree_proposal_cap": PREFIX_CAP},
    "tree_warm": {"tree_proposal_cap": PREFIX_CAP, "warm": True},
    "length_uniform": {},
    "length_tilted": {"tilt": 2.0},
    "max_compatible": {},
    "max_bandit": {},
}


def _make_sampler(arm, env, new_path, seed, shared_state):
    if arm in ("tree_v2", "tree_adaptive", "tree_partial", "tree_warm", "tree_credit",
               "tree_simple", "tree_split2", "tree_rand", "tree_cap",
               "tree_warmbeta", "tree_mix", "tree_only"):
        kwargs = dict(_SAMPLER_KWARGS[arm])
        kwargs["shared_state"] = shared_state
        if arm == "tree_simple":
            return SimpleTreeSampler(env, new_path, seed=seed, **kwargs)
        if arm == "tree_split2":
            return Split2TreeSampler(env, new_path, seed=seed, **kwargs)
        if arm == "tree_rand":
            return RandomFallbackTreeSampler(env, new_path, seed=seed, **kwargs)
        if arm == "tree_cap":
            return CappedSeedTreeSampler(env, new_path, seed=seed, **kwargs)
        if arm == "tree_warmbeta":
            return WarmBetaTreeSampler(env, new_path, seed=seed, **kwargs)
        if arm == "tree_mix":
            return MixTreeSampler(env, new_path, seed=seed, **kwargs)
        return CappedTreeSampler(env, new_path, seed=seed, **kwargs)
    if arm == "length_tilted":
        return LengthTiltedSampler(env, new_path, seed=seed, tilt=2.0)
    if arm == "length_uniform":
        return LengthGainSampler(env, new_path, seed=seed, tilt=None)
    if arm == "max_compatible":
        return MaxCompatibleSampler(env, new_path, seed=seed)
    if arm == "max_bandit":
        return MaxBanditSampler(env, new_path, seed=seed, shared_state=shared_state)
    raise ValueError(f"unknown v2 arm: {arm}")


# ---------------------------------------------------------------------------------------
# the v2 loop -- verbatim landed semantics, exact gate, optional pre-registered stops
# ---------------------------------------------------------------------------------------


def shortcut_run_v2(
    env,
    path,
    arm: str,
    max_iter: int = 1000,
    max_attempts: Optional[int] = None,
    resolution=0.01,
    tolerance=0.01,
    interpolation_resolution: float = 0.5,
    seed: int = 0,
    fail_cell: Optional[dict] = None,
    stats_out: Optional[Dict] = None,
    check_counter=None,
    shared_state=None,
    adaptive_stop: Optional[int] = None,
    checks_cap: Optional[int] = None,
    tilt: Optional[float] = None,
):
    """Same contract as ``robot_mode_shortcut``: returns ``(new_path, [costs, times])``.

    Differences from the landed loop, all pre-registered:

    * The cost gate uses the exact vectorized table when available.  An indeterminate table
      verdict falls back to the unchanged native construction gate.
    * ``adaptive_stop``: after the tree prefix, stop the fallback once this many consecutive
      fallback validations produce no acceptance.
    * ``checks_cap``: stop the whole call after this many validations (used by tree_credit).
    """
    State, interpolate_path, path_cost, remove_interpolated_nodes, constant_task_runs = _mrb()

    non_redundant_path = remove_interpolated_nodes(path)
    new_path = interpolate_path(non_redundant_path, interpolation_resolution)

    table = SegmentCostTable(env, lambda: new_path)

    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]

    st: Dict = dict(
        arm=arm, n_states=len(new_path), iters=0, skipped_short=0, skipped_gate=0,
        cost_rejects=0, collision_checks=0, accepted=0, rejected_collision=0,
        sampler_time=0.0, sampler_calls=0, cost_initial=float(costs[0]), stop_reason=None,
        accepted_per_robot={}, fail_located=0,
        gate_table_ok=bool(table.ok), gate_uses=0, gate_positive=0, gate_negative=0,
        gate_eps_fallback=0, adaptive_stop_hit=False, checks_cap_hit=False,
        irreplaceability_gap=None, irreplaceability_gap_per_robot=None,
    )

    t_s = time.perf_counter()
    sampler = _make_sampler(arm, env, new_path, seed, shared_state)
    st["sampler_time"] += time.perf_counter() - t_s

    start_time = time.time()
    if max_attempts is None:
        max_attempts = 5 * max_iter
    cnt = 0
    iter = 0
    since_accept = 0

    while True:
        iter += 1
        if cnt >= max_iter or iter >= max_attempts:
            st["stop_reason"] = "max_iter" if cnt >= max_iter else "max_attempts"
            break
        if checks_cap is not None and cnt >= checks_cap:
            st["checks_cap_hit"] = True
            st["stop_reason"] = "checks_cap"
            break

        t_s = time.perf_counter()
        try:
            proposal = sampler.propose()
        except _Stop as exc:
            st["sampler_time"] += time.perf_counter() - t_s
            st["stop_reason"] = f"sampler exhausted: {exc}"
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

        # ---- exact vectorized cost gate with native fallback ---------------------------
        improves = None
        path_element = None
        if table.ensure():
            verdict = table.decide(robots_to_shortcut, i, j)
            st["gate_uses"] += 1
            if verdict is not None:
                improves = verdict
                if verdict:
                    st["gate_positive"] += 1
                else:
                    st["gate_negative"] += 1
            else:
                st["gate_eps_fallback"] += 1

        if improves is None:
            path_element = _proposal_candidate(
                env, new_path, i, j, robots_to_shortcut, State
            )
            candidate_segment_cost = path_cost(path_element, env.batch_config_cost)
            old_segment_cost = path_cost(new_path[i : j + 1], env.batch_config_cost)
            improves = candidate_segment_cost < old_segment_cost

        if not improves:
            st["cost_rejects"] += 1
            t_s = time.perf_counter()
            sampler.feedback("cost", ij=(i, j), robot=robots_to_shortcut[0])
            st["sampler_time"] += time.perf_counter() - t_s
            continue

        # ---- verbatim: construction for the accepted path ------------------------------
        if path_element is None:
            path_element = _proposal_candidate(
                env, new_path, i, j, robots_to_shortcut, State
            )
        q0 = new_path[i].q
        q1 = new_path[j].q
        assert np.linalg.norm(path_element[0].q.state() - q0.state()) < 1e-6
        assert np.linalg.norm(path_element[-1].q.state() - q1.state()) < 1e-6

        cnt += 1
        st["collision_checks"] += 1

        if fail_cell is not None:
            fail_cell["q"] = None
        free = env.is_path_collision_free(
            path_element, resolution=resolution, tolerance=tolerance,
            check_start_and_end=False,
        )

        fail_k = None
        if not free and fail_cell is not None and fail_cell.get("q") is not None:
            fq = fail_cell["q"]
            for k in range(j - i + 1):
                if path_element[k].q is fq:
                    fail_k = i + k
                    st["fail_located"] += 1
                    break

        if free:
            for k in range(j - i + 1):
                new_path[i + k].q = path_element[k].q
            st["accepted"] += 1
            for r in robots_to_shortcut:
                st["accepted_per_robot"][r] = st["accepted_per_robot"].get(r, 0) + 1
            table.mark_dirty()
        else:
            st["rejected_collision"] += 1

        current_time = time.time()
        times.append(current_time - start_time)
        costs.append(path_cost(new_path, env.batch_config_cost))

        # ---- extended feedback ----------------------------------------------------------
        t_s = time.perf_counter()
        if free:
            sampler.feedback("accepted", dcost=float(cost_before - costs[-1]),
                             ij=(i, j), robot=robots_to_shortcut[0])
        else:
            sampler.feedback("collision", ij=(i, j), robot=robots_to_shortcut[0],
                             fail_k=fail_k)
        st["sampler_time"] += time.perf_counter() - t_s

        # ---- adaptive stop of the fallback (tree arms only) ------------------------------
        fallback_active = (
            adaptive_stop is not None
            and getattr(sampler, "_last", None) is not None
            and sampler._last[0] == "fallback"
        )
        if fallback_active:
            if free:
                since_accept = 0
            else:
                since_accept += 1
                if since_accept >= adaptive_stop:
                    st["adaptive_stop_hit"] = True
                    st["stop_reason"] = "adaptive_stop"
                    break

    st["iters"] = iter
    st["wall_time"] = time.time() - start_time
    st["cost_final"] = float(costs[-1])

    # ---- irreplaceability diagnostic (observational, WAFR concept) -----------------------
    if table.ensure():
        runs = [constant_task_runs(new_path, r) for r in range(len(env.robots))]
        gap = 0.0
        per_robot = {}
        for r, rs in enumerate(runs):
            robot_gap = 0.0
            for lo, hi in rs:
                robot_gap += max(0.0, table.delta([r], lo, hi))
            per_robot[str(r)] = float(robot_gap)
            gap += robot_gap
        st["irreplaceability_gap"] = float(gap) / max(float(costs[-1]), 1e-12)
        st["irreplaceability_gap_per_robot"] = per_robot

    st.update(sampler.report())

    if stats_out is not None:
        stats_out.clear()
        stats_out.update(st)

    # ---- verbatim: final asserts ---------------------------------------------------------
    assert new_path[-1].mode == path[-1].mode
    assert np.linalg.norm(new_path[-1].q.state() - path[-1].q.state()) < 1e-6
    assert np.linalg.norm(new_path[0].q.state() - path[0].q.state()) < 1e-6

    return new_path, [costs, times]
