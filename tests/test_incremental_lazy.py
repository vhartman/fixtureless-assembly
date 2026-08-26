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
