"""Memory/correctness regressions for incremental lazy-search repair bookkeeping."""

from solver import _IncrementalLazySearch


def test_duplicate_repair_routes_are_stored_once():
  e = _IncrementalLazySearch(
      "S", goal_fn=lambda _s: False, succ_fn=lambda _s: (), cost_fn=lambda _a: 1,
      h_fn=lambda _s: 0, weight=1, blacklist=set())
  entry = (2, 0, 2, 10, "X", "P", "p-x")
  e._remember_alt("X", entry)
  e._remember_alt("X", entry)
  # A regenerated copy differs only in its heap tie id; it is the same incoming graph edge.
  e._remember_alt("X", (2, 0, 2, 99, "X", "P", "p-x"))
  assert e.footprint()["alternates"] == 1


def test_distinct_alternates_survive_successive_edge_deletions():
  # Make all three parents expand before X, then make X's other two incoming entries pop and
  # become alternates before the expensive X->G edge reaches the front of the heap.
  graph = {
      "S": [(('s-a', 1), "A"), (('s-b', 1), "B"), (('s-c', 1), "C")],
      "A": [(('a-x', 1), "X")],
      "B": [(('b-x', 1), "X")],
      "C": [(('c-x', 1), "X")],
      "X": [(('x-g', 10), "G")],
      "G": [],
  }
  blacklist = set()
  e = _IncrementalLazySearch(
      "S", goal_fn=lambda s: s == "G", succ_fn=lambda s: graph[s],
      cost_fn=lambda a: a[1], h_fn=lambda _s: 0, weight=1, blacklist=blacklist)
  assert e.search(100) == [('s-a', 1), ('a-x', 1), ('x-g', 10)]
  assert e.footprint()["alternates"] == 2

  blacklist.add(("A", ('a-x', 1)))
  e.delete_edge("A", ('a-x', 1), "X")
  assert e.search(100) == [('s-b', 1), ('b-x', 1), ('x-g', 10)]

  blacklist.add(("B", ('b-x', 1)))
  e.delete_edge("B", ('b-x', 1), "X")
  assert e.search(100) == [('s-c', 1), ('c-x', 1), ('x-g', 10)]
  # Reclaims do not append the same child to its parent's tree bookkeeping indefinitely.
  assert all(isinstance(v, set) for v in e.children.values())


def test_regenerated_successors_preserve_incremental_repair_result():
  graph = {
      "S": [(('s-a', 1), "A"), (('s-b', 1), "B")],
      "A": [(('a-x', 1), "X")],
      "B": [(('b-x', 1), "X")],
      "X": [(('x-g', 5), "G")],
      "G": [],
  }

  def run(memoize):
    blacklist = set()
    e = _IncrementalLazySearch(
        "S", goal_fn=lambda s: s == "G", succ_fn=lambda s: graph[s],
        cost_fn=lambda a: a[1], h_fn=lambda _s: 0, weight=1, blacklist=blacklist,
        memoize_successors=memoize)
    first = e.search(100)
    blacklist.add(("A", ('a-x', 1)))
    e.delete_edge("A", ('a-x', 1), "X")
    return first, e.search(100), e.footprint()

  cached = run(True)
  regenerated = run(False)
  assert cached[:2] == regenerated[:2]
  assert cached[2]["successor_edges"] > 0
  assert regenerated[2]["successor_edges"] == 0


def test_compact_frontier_regenerates_and_recovers_an_alternate_route():
  graph = {
      "S": [(('s-a', 1), "A"), (('s-b', 2), "B")],
      "A": [(('a-x', 1), "X")],
      "B": [(('b-x', 1), "X")],
      "X": [(('x-g', 1), "G")],
      "G": [],
  }
  blacklist = set()
  e = _IncrementalLazySearch(
      "S", goal_fn=lambda s: s == "G", succ_fn=lambda s: graph[s],
      cost_fn=lambda a: a[1], h_fn=lambda _s: 0, weight=1, blacklist=blacklist,
      memoize_successors=False, compact_frontier=True)
  assert e.search(100) == [('s-a', 1), ('a-x', 1), ('x-g', 1)]

  blacklist.add(("A", ('a-x', 1)))
  e.delete_edge("A", ('a-x', 1), "X")
  assert e.search(100) == [('s-b', 2), ('b-x', 1), ('x-g', 1)]
  footprint = e.footprint()
  assert footprint["alternates"] <= footprint["unique_states"]
  assert footprint["successor_edges"] == 0
  assert footprint["compact_frontier"] is True


def test_compact_frontier_regenerates_after_a_cheap_edge_refutation():
  graph = {
      "S": [(('bad', 1), "X"), (('good', 2), "X")],
      "X": [(('finish', 1), "G")],
      "G": [],
  }
  blacklist = set()
  e = _IncrementalLazySearch(
      "S", goal_fn=lambda s: s == "G", succ_fn=lambda s: graph[s],
      cost_fn=lambda a: a[1], h_fn=lambda _s: 0, weight=1, blacklist=blacklist,
      cheap_gate=lambda _s, a, _ns: a[0] != "bad",
      memoize_successors=False, compact_frontier=True)
  assert e.search(100) == [('good', 2), ('finish', 1)]
  assert ("S", ('bad', 1)) in blacklist


def test_opt_in_forward_search_profile_counts_work(monkeypatch):
  monkeypatch.setenv("FILL_SEARCH_PROFILE", "1")
  graph = {
      "S": [(('go', 1), "A")],
      "A": [(('finish', 1), "G")],
      "G": [],
  }
  e = _IncrementalLazySearch(
      "S", goal_fn=lambda s: s == "G", succ_fn=lambda s: graph[s],
      cost_fn=lambda a: a[1], h_fn=lambda _s: 0, weight=1, blacklist=set(),
      cheap_gate=lambda _s, _a, _ns: True)
  assert e.search(10) == [('go', 1), ('finish', 1)]
  p = e.profile_snapshot()
  assert p["search_calls"] == 1
  assert p["search_wall"] >= 0
  assert p["succ_calls"] == 2
  assert p["succ_edges"] == 2
  assert p["cheap_calls"] == 2
  assert p["heap_pops"] == 3


def test_heuristic_is_evaluated_once_per_value_state_across_repairs():
  graph = {
      "S": [(('left', 1), "A"), (('right', 1), "B")],
      "A": [(('bad', 1), "X")],
      "B": [(('good', 1), "X")],
      "X": [(('finish', 1), "G")],
      "G": [],
  }
  calls = {}

  def h(state):
    calls[state] = calls.get(state, 0) + 1
    return 0

  blacklist = set()
  e = _IncrementalLazySearch(
      "S", goal_fn=lambda s: s == "G", succ_fn=lambda s: graph[s],
      cost_fn=lambda a: a[1], h_fn=h, weight=1, blacklist=blacklist)
  assert e.search(20) == [('left', 1), ('bad', 1), ('finish', 1)]
  blacklist.add(("A", ('bad', 1)))
  e.delete_edge("A", ('bad', 1), "X")
  assert e.search(20) == [('right', 1), ('good', 1), ('finish', 1)]
  assert calls == {state: 1 for state in graph}


def test_strict_cost_bound_is_independent_of_greedy_queue_order():
  graph = {
      "S": [(('cheap-start', 1), "A"), (('greedy-start', 3), "B")],
      "A": [(('cheap-finish', 1), "G")],
      "B": [(('greedy-finish', 1), "G")],
      "G": [],
  }
  h = {"S": 1, "A": 1, "B": 0, "G": 0}
  greedy = _IncrementalLazySearch(
      "S", goal_fn=lambda s: s == "G", succ_fn=lambda s: graph[s],
      cost_fn=lambda a: a[1], h_fn=h.__getitem__, weight=float("inf"), blacklist=set())
  assert greedy.search(20) == [('greedy-start', 3), ('greedy-finish', 1)]

  improve = _IncrementalLazySearch(
      "S", goal_fn=lambda s: s == "G", succ_fn=lambda s: graph[s],
      cost_fn=lambda a: a[1], h_fn=h.__getitem__, weight=1, blacklist=set(), cost_bound=4)
  assert improve.search(20) == [('cheap-start', 1), ('cheap-finish', 1)]
  # The bound is strict: asking for a plan cheaper than the optimum exhausts cleanly.
  prove = _IncrementalLazySearch(
      "S", goal_fn=lambda s: s == "G", succ_fn=lambda s: graph[s],
      cost_fn=lambda a: a[1], h_fn=h.__getitem__, weight=1, blacklist=set(), cost_bound=2)
  assert prove.search(20) is None
