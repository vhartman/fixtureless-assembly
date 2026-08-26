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
    if not report.max_successor_states:
      return ""
    return (f"; search peak pending/heap={report.max_frontier}/{report.max_heap_entries}, "
            f"alts={report.max_alternates}, unique={report.max_unique_states}, "
            f"succ={report.max_successor_states}/{report.max_successor_edges}, "
            f"tree={report.max_tree_edges}")

  epoch = 0
  while max_epochs == 0 or epoch < max_epochs:
    if deadline is not None and time.perf_counter() >= deadline:
      report.deadline = True
      return None, None, report
    epoch += 1
    report.epochs = epoch
    suspended = set()                                    # cleared at the next fairness epoch
    engine = make_engine(suspended)
    while True:
      if deadline is not None and time.perf_counter() >= deadline:
        report.deadline = True
        return None, None, report
      subplan = engine.search(max_nodes, deadline=deadline)
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
                  f"(not exhaustion); reopening with escalated effort")
        break                                             # reopen all and increase effort
      valid = True
      end_state = None
      for state, action, successor in walk_plan(subplan):
        end_state = successor
        if not is_ik(action):
          continue
        if pool.known(state, action):
          continue
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
