"""Opt-in fair, non-terminal IK witness validation for lazy milestone fills.

The legacy LazySP path asks one edge for its full IK budget and permanently blacklists it on
failure.  Here a failure means only "not on this visit": each fairness epoch gives an edge one
bounded BATCH, suspends it for the rest of that epoch, and lets alternate plans run.  When the
cheap search exhausts those alternatives, a fresh epoch re-opens every failed edge with its
next batch.  Successes persist; failures never enter a permanent cache.  All attempts in a
batch share scene construction and reach-precheck setup.

This module is intentionally search-engine agnostic.  ``solver._add_one_part`` supplies three
small adapters (construct engine, walk a plan, identify IK edges), keeping the existing engine
and default code path unchanged.
"""
from collections import defaultdict
from dataclasses import dataclass
import time


@dataclass
class OpenWitnessReport:
  epochs: int = 0
  probes: int = 0
  successes: int = 0
  open_failures: int = 0
  exhausted: bool = False
  deadline: bool = False
  budget_hits: int = 0
  max_frontier: int = 0
  max_heap_entries: int = 0
  max_alternates: int = 0
  max_unique_states: int = 0
  max_successor_states: int = 0
  max_successor_edges: int = 0
  max_tree_edges: int = 0
  max_node_budget: int = 0
  reordered_paths: int = 0
  prior_failure_first: int = 0


class OpenWitnessPool:
  """Solve-scoped compact effort state. Stores counters and exact successful vectors."""

  def __init__(self, robot_ik):
    self.robot_ik = robot_ik
    self.attempts = defaultdict(int)                     # scene-query -> next attempt index
    self.visits = defaultdict(int)                       # scene-query -> completed effort tiers
    # query -> the exact vector that certified this edge.  Keeping this (not a KOMO/scene)
    # makes search witness persistence independent of RobotIK's incidental memoization cache.
    self.success = {}

  def key(self, state, action):
    return action, self.robot_ik._scene_sig(state)

  def known(self, state, action):
    return self.key(state, action) in self.success

  def witness_for(self, state, action):
    return self.success.get(self.key(state, action))

  def failed_effort(self, state, action):
    """Exact-query evidence available before another stochastic probe.

    A key absent from ``success`` has failed every completed visit recorded here.  Keep this
    deliberately narrower than RobotIK's action-level statistics: fixed-base reachability and
    collision depend on the complete support scene, so evidence from another scene must not
    reorder this query.
    """
    key = self.key(state, action)
    return self.visits.get(key, 0), self.attempts.get(key, 0)

  def _batch_for(self, action, key):
    """Skill-aware geometric effort ladder, capped by ``--open-witness-batch``.

    Ordinary skills receive 2, 4, then B fresh-attempt slots.  Assemble keeps its structural
    bundle atomic: up to B slots cover the six canonical orientation basins and one unbiased
    solve in one shared scene.  Splitting that bundle across epochs was measured to redirect the
    greedy search into many alternatives before the promising edge could reopen.  Small explicit
    B values remain respected throughout.
    """
    cap = max(1, int(getattr(self.robot_ik, "open_witness_batch", 1)))
    visit = self.visits[key]
    if action[0] == "assemble":
      return cap                                         # six biases + random is one cheap bundle
    return min(cap, 2 << min(visit, 2))                  # 2, 4, then capped full batches

  def probe(self, action, state, successor, assembly):
    key = self.key(state, action)
    if key in self.success:
      return True, self.success[key], False
    batch = self._batch_for(action, key)
    attempt = self.attempts[key]
    self.attempts[key] += batch
    self.visits[key] += 1
    if hasattr(self.robot_ik, "probe_bind_batch"):
      ok, witness = self.robot_ik.probe_bind_batch(
          action, state, successor, assembly, attempt=attempt, budget=batch)
      if ok:
        self.success[key] = witness
      return ok, witness, batch
    # Tiny fake/backward-compatible adapter used by unit tests and external RobotIK-like
    # implementations: emulate the batch with their one-at-a-time interface.
    witness = None
    for spent in range(1, batch + 1):
      ok, witness = self.robot_ik.probe_bind_once(
          action, state, successor, assembly, attempt=attempt + spent - 1)
      if ok:
        self.success[key] = witness
        return True, witness, spent
    return False, witness, batch


def run_open_lazy_fill(*, robot_ik, assembly, make_engine, walk_plan, is_ik,
                       max_nodes, deadline=None, max_epochs=2, verbose=False, tag="fill"):
  """Fairly validate candidate plans; return ``(plan, end_state, report)``.

  ``max_epochs`` is an effort limit, not an infeasibility certificate.  Zero means unlimited
  and therefore requires a finite ``deadline``.  Each epoch rebuilds the small lazy engine;
  this isolation is deliberate and avoids invasive suspend/resume changes to the measured
  legacy incremental search.
  """
  if max_epochs == 0 and deadline is None:
    raise ValueError("unlimited open-witness epochs require a finite deadline")
  pool = getattr(robot_ik, "_open_witness_pool", None)
  if pool is None:
    pool = robot_ik._open_witness_pool = OpenWitnessPool(robot_ik)
  report = OpenWitnessReport()
  selector = getattr(robot_ik, "lazy_edge_selector", "forward")
  if selector not in ("forward", "terminal-first", "reverse", "fail-first"):
    raise ValueError("lazy edge selector must be 'forward', 'terminal-first', 'reverse', "
                     "or 'fail-first'")

  def validation_order(walked):
    """Return unresolved IK edges in the requested LazySP validation order.

    Witnesses certify individual endpoints in a fully known symbolic scene, so their solve
    order need not follow execution order.  ``fail-first`` uses only failures of the EXACT
    ``(action, scene signature)`` query.  An unseen tie falls back to reverse path order,
    putting the contact-critical suffix in front without inventing cross-scene evidence.
    """
    unresolved = [(i, edge) for i, edge in enumerate(walked)
                  if is_ik(edge[1]) and not pool.known(edge[0], edge[1])]
    if selector == "forward":
      return [edge for _i, edge in unresolved]
    if selector == "terminal-first":
      # Concrete disassembly proposal without a duplicate inverse action model: a complete
      # symbolic candidate already exposes its exact pre-assembly scene. Certify the LAST mate
      # first, then validate its causal prefix in execution order. A failed terminal mode is a
      # direct candidate refutation; a successful witness persists in the ordinary exact-scene
      # pool. No edge is pruned merely because another terminal mode failed.
      terminal = next((item for item in reversed(unresolved)
                       if item[1][1][0] == "assemble"), None)
      if terminal is None:
        return [edge for _i, edge in unresolved]
      terminal_i, terminal_edge = terminal
      return [terminal_edge] + [edge for i, edge in unresolved if i != terminal_i]
    if selector == "reverse":
      return [edge for _i, edge in reversed(unresolved)]

    def priority(item):
      i, (state, action, _successor) = item
      visits, slots = pool.failed_effort(state, action)
      # A previously failed exact query is the strongest candidate-path refutation.  Among
      # such queries, prefer more accumulated evidence/effort; otherwise use the deterministic
      # reverse-path fallback.  ``max`` therefore selects the desired edge first.
      return (visits > 0, visits, slots, i)

    ranked = sorted(unresolved, key=priority, reverse=True)
    if ranked and pool.failed_effort(ranked[0][1][0], ranked[0][1][1])[0] > 0:
      report.prior_failure_first += 1
    return [edge for _i, edge in ranked]

  def sample(engine):
    if not hasattr(engine, "footprint"):
      return
    f = engine.footprint()
    report.max_frontier = max(report.max_frontier, f["frontier"])
    report.max_heap_entries = max(report.max_heap_entries, f["heap_entries"])
    report.max_alternates = max(report.max_alternates, f["alternates"])
    report.max_unique_states = max(report.max_unique_states, f["unique_states"])
    report.max_successor_states = max(report.max_successor_states, f["successor_states"])
    report.max_successor_edges = max(report.max_successor_edges, f["successor_edges"])
    report.max_tree_edges = max(report.max_tree_edges, f["tree_edges"])

  def storage_tail():
    select = (f"; selector={selector}, reordered={report.reordered_paths}, "
              f"prior-failure-first={report.prior_failure_first}")
    succ = (f"{report.max_successor_states}/{report.max_successor_edges}"
            if report.max_successor_states else "regenerated")
    return (f"; search peak pending/heap={report.max_frontier}/{report.max_heap_entries}, "
            f"alts={report.max_alternates}, unique={report.max_unique_states}, "
            f"succ={succ}, "
            f"tree={report.max_tree_edges}, node-cap={report.max_node_budget}" + select)

  epoch = 0
  while max_epochs == 0 or epoch < max_epochs:
    if deadline is not None and time.perf_counter() >= deadline:
      report.deadline = True
      return None, None, report
    epoch += 1
    report.epochs = epoch
    # A node-budget stop is search incompleteness, not new IK evidence.  The old scheduler
    # restarted every epoch with the identical cap, so a large equal-f shell could be replayed
    # forever while only the per-edge IK batch grew.  Geometric widening is the standard
    # anytime remedy: finite profiles get a predictable 1x,2x,... ladder; unlimited epochs
    # remain complete in the limit under their required wall-clock deadline.
    epoch_nodes = int(max_nodes) * (1 << (epoch - 1))
    report.max_node_budget = max(report.max_node_budget, epoch_nodes)
    suspended = set()                                    # cleared at the next fairness epoch
    engine = make_engine(suspended)
    while True:
      if deadline is not None and time.perf_counter() >= deadline:
        report.deadline = True
        return None, None, report
      subplan = engine.search(epoch_nodes, deadline=deadline)
      sample(engine)
      if subplan is None:
        if getattr(engine, "budget_hit", False):
          if deadline is not None and time.perf_counter() >= deadline:
            report.deadline = True
            return None, None, report
          # The NODE budget (not the fair-effort limit) ended this epoch. Reopening stays
          # intentional -- a fresh epoch restarts the discrete expansion allowance while the
          # effort batches keep escalating -- but the outcome must not be reported as a plain
          # "effort limit" (that used to conflate the two).
          report.budget_hits += 1
          if verbose:
            print(f"[open-witness {tag}] epoch {epoch} ended on the node budget "
                  f"{epoch_nodes} (not exhaustion); reopening with IK + node effort"
                  + (f" (next cap {epoch_nodes * 2})"
                     if max_epochs == 0 or epoch < max_epochs else ""))
        break                                             # reopen all and increase effort
      valid = True
      walked = list(walk_plan(subplan))
      end_state = walked[-1][2] if walked else None
      ordered = validation_order(walked)
      forward_unresolved = [edge for edge in walked
                            if is_ik(edge[1]) and not pool.known(edge[0], edge[1])]
      if ordered != forward_unresolved:
        report.reordered_paths += 1
      for state, action, successor in ordered:
        ok, _witness, spent = pool.probe(action, state, successor, assembly)
        report.probes += spent
        if ok:
          report.successes += 1
          continue
        report.open_failures += 1
        suspended.add((state, action))
        engine.delete_edge(state, action, successor)
        valid = False
        break                                            # incumbent is invalid: replan NOW
      if valid:
        if verbose:
          print(f"[open-witness {tag}] solved in {epoch} epoch(s), {report.probes} effort slots "
                f"({report.open_failures} open failures, {report.successes} successes)"
                f"{storage_tail()}")
        return subplan, end_state, report
  report.exhausted = True
  if verbose:
    tail = (f", {report.budget_hits} node-budget epoch(s)" if report.budget_hits else "")
    print(f"[open-witness {tag}] effort limit after {report.epochs} epoch(s), "
          f"{report.probes} effort slots{tail}; failures remain OPEN (not blacklisted)"
          f"{storage_tail()}")
  return None, None, report
