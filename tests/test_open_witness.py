import pytest

from open_witness import OpenWitnessPool, run_open_lazy_fill
import solver as solver_mod
from solver import RobotIK, _IncrementalLazySearch, _deterministic_ik_seed


class _State:
  def __init__(self, name):
    self.name = name

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other):
    return isinstance(other, _State) and self.name == other.name


def test_physical_query_seed_is_container_order_independent():
  key_a = ("assemble", {"parts": frozenset(("obj_2", "obj_1")),
                         "poses": [0.0, -0.0, 0.25]})
  key_b = ("assemble", {"poses": [0.0, -0.0, 0.25],
                         "parts": frozenset(("obj_1", "obj_2"))})
  assert _deterministic_ik_seed(11023, key_a, 0) \
      == _deterministic_ik_seed(11023, key_b, 0)


def test_physical_query_seed_partitions_root_query_and_effort():
  key = ("pick", "a1", "obj_6", 20, ("table-scene",))
  seeds = {
      _deterministic_ik_seed(11023, key, 0),
      _deterministic_ik_seed(11024, key, 0),
      _deterministic_ik_seed(11023, key + ("other",), 0),
      _deterministic_ik_seed(11023, key, 1),
  }
  assert len(seeds) == 4
  assert all(1 <= seed <= 2_147_483_646 for seed in seeds)


def test_incremental_search_pop_observer_sees_the_real_claim_order():
  s, a, b = _State("s"), _State("a"), _State("b")
  go_a = ("pick", "a1", "a", 0)
  go_b = ("pick", "a1", "b", 0)
  graph = {s: [(go_a, a), (go_b, b)], a: [], b: []}
  seen = []
  engine = _IncrementalLazySearch(
      s, goal_fn=lambda x: x == b, succ_fn=lambda x: graph[x],
      cost_fn=lambda _a: 1.0, h_fn=lambda x: {s: 1.0, a: 2.0, b: 0.0}[x],
      weight=1.0, blacklist=set(),
      pop_observer=lambda _engine, state, f, h, g: seen.append((state.name, f, h, g)))
  assert engine.search(10) == [go_b]
  assert seen == [("s", 1.0, 1.0, 0.0), ("b", 1.0, 0.0, 1.0)]


def test_probe_bind_once_never_caches_a_stochastic_failure():
  rik = RobotIK.__new__(RobotIK)
  action, state = ("pick", "a1", "obj", 0), _State("s")
  key = (action, ("scene",))
  rik._scene_sig = lambda _s: ("scene",)
  rik._ik_cache = {key: (False, None)}                   # legacy false must not poison probe
  rik.restarts_full, rik.restarts_precheck = 12, 7
  rik._open_probe_attempt = None
  seen = []

  def bind(_action, _s0, _s1, _assembly):
    seen.append((rik.restarts_full, rik.restarts_precheck, rik._open_probe_attempt))
    rik._ik_cache[key] = (False, None)
    return False, None

  rik.bind = bind
  assert rik.probe_bind_once(action, state, _State("n"), object(), attempt=3) == (False, None)
  assert seen == [(0, 0, 3)]
  assert key not in rik._ik_cache
  assert (rik.restarts_full, rik.restarts_precheck, rik._open_probe_attempt) == (12, 7, None)


def test_probe_bind_batch_pays_bind_setup_once_and_restores_probe_state():
  rik = RobotIK.__new__(RobotIK)
  action, state = ("pick", "a1", "obj", 0), _State("s")
  key = (action, ("scene",))
  rik._scene_sig = lambda _s: ("scene",)
  rik._ik_cache = {}
  rik.restarts_full, rik.restarts_precheck = 12, 7
  rik._open_probe_attempt, rik._open_probe_budget = None, 1
  rik._open_probe_random_only = False
  seen = []

  def bind(_action, _s0, _s1, _assembly):
    seen.append((rik.restarts_full, rik.restarts_precheck,
                 rik._open_probe_attempt, rik._open_probe_budget,
                 rik._open_probe_random_only))
    return True, "witness"

  rik.bind = bind
  assert rik.probe_bind_batch(
      action, state, _State("n"), object(), attempt=14, budget=7) == (True, "witness")
  assert seen == [(6, 0, 14, 7, True)]                   # one scene/bind, fresh random draws
  assert (rik.restarts_full, rik.restarts_precheck,
          rik._open_probe_attempt, rik._open_probe_budget) == (12, 7, None, 1)
  assert not rik._open_probe_random_only
  assert key not in rik._ik_cache                         # the fake bind did not cache anything


def test_open_scheduler_reopens_failures_only_after_alternatives_get_a_turn():
  s, na, nb = _State("s"), _State("na"), _State("nb")
  a, b = ("pick", "a1", "a", 0), ("pick", "a1", "b", 0)
  edges = {a: (s, a, na), b: (s, b, nb)}
  calls = []

  class RIK:
    def _scene_sig(self, state):
      return state.name

    def probe_bind_once(self, action, _state, _successor, _assembly, attempt):
      calls.append((action, attempt))
      return (action == a and attempt == 1), None

  class Engine:
    def __init__(self, suspended):
      self.suspended = suspended

    def search(self, _max_nodes, deadline=None):
      for action in (a, b):
        if (s, action) not in self.suspended:
          return [action]
      return None

    def delete_edge(self, state, action, successor):
      assert (state, action, successor) == edges[action]

  def walk(plan):
    for action in plan:
      yield edges[action]

  plan, end, report = run_open_lazy_fill(
      robot_ik=RIK(), assembly=object(), make_engine=Engine, walk_plan=walk,
      is_ik=lambda _a: True, max_nodes=10, max_epochs=3)
  assert plan == [a]
  assert end == na
  assert calls == [(a, 0), (b, 0), (a, 1)]
  assert report.epochs == 2
  assert report.open_failures == 2
  assert report.successes == 1


def test_node_budget_epochs_geometrically_widen_discrete_search():
  s, n = _State("s"), _State("n")
  action = ("release", "a1", "obj")
  budgets = []
  made = [0]

  class RIK:
    lazy_edge_selector = "forward"

    def _scene_sig(self, state):
      return state.name

  class Engine:
    def __init__(self, epoch):
      self.epoch = epoch
      self.budget_hit = False

    def search(self, max_nodes, deadline=None):
      budgets.append(max_nodes)
      if self.epoch == 0:
        self.budget_hit = True
        return None
      return [action]

  def make_engine(_suspended):
    epoch = made[0]
    made[0] += 1
    return Engine(epoch)

  def walk(_plan):
    yield s, action, n

  plan, end, report = run_open_lazy_fill(
      robot_ik=RIK(), assembly=object(), make_engine=make_engine, walk_plan=walk,
      is_ik=lambda _a: False, max_nodes=10, max_epochs=2)
  assert (plan, end) == ([action], n)
  assert budgets == [10, 20]
  assert report.budget_hits == 1
  assert report.max_node_budget == 20


def test_milestone_anytime_preserves_standard_incumbent_search_policy(monkeypatch):
  class RIK:
    open_witness = True
    open_witness_epochs = 2

  rik = RIK()
  seen = {}

  def fake_milestones(*_args, **kwargs):
    seen["epochs"] = rik.open_witness_epochs
    seen["weight"] = kwargs.get("weight")
    seen["deadline"] = kwargs.get("deadline")
    return [("release", "a1", "obj")]

  monkeypatch.setattr(solver_mod, "solve_assembly_problem_milestones", fake_milestones)
  out = solver_mod.solve_assembly_problem_milestones_anytime(
      object(), object(), robots=("a1",), time_budget=1.0, verbose=False, robot_ik=rik,
      weight=1.0)
  assert out == [("release", "a1", "obj")]
  assert seen["epochs"] == 2 and seen["weight"] == 1.0 and seen["deadline"] is not None
  assert rik.open_witness_epochs == 2


def test_milestone_anytime_defaults_to_full_horizon_lns_not_separate_full_astar(monkeypatch):
  inc = [("release", "a1", "obj")]
  seen = {}

  class Checker:
    gripper_path = None

  class H:
    def __init__(self, *_args, **_kwargs):
      pass

    def value(self, _state):
      return 0.0

  monkeypatch.setattr(solver_mod, "solve_assembly_problem_milestones",
                      lambda *_args, **_kwargs: inc)
  monkeypatch.setattr(solver_mod, "_search_context",
                      lambda *_args, **_kwargs: (Checker(), object()))
  monkeypatch.setattr(solver_mod, "GraspCollapsedHeuristic", H)

  def fake_lns(plan, *_args, **kwargs):
    seen["plan"] = plan
    seen["k_max"] = kwargs["k_max"]
    return plan

  monkeypatch.setattr(solver_mod, "_lns_improve", fake_lns)
  monkeypatch.setattr(
      solver_mod, "solve_assembly_problem",
      lambda *_args, **_kwargs: pytest.fail("full-state A* must not be the improvement layer"))
  out = solver_mod.solve_assembly_problem_milestones_anytime(
      object(), object(), robots=("a1",), time_budget=20.0, verbose=False,
      fill_window=2)
  assert out == inc
  assert seen == {"plan": inc, "k_max": None}


def test_lns_refill_uses_astar_and_freezes_parts_outside_destroyed_span(monkeypatch):
  class State(_State):
    def key(self):
      return self.name

    def is_goal(self, _assembly):
      return False

  class Heuristic:
    def value(self, _state):
      return 0.0

  class Assembly:
    objects = ("moving", "target", "outside")

  start, target = State("start"), State("target-state")
  actions = [
      (("handover", "a1", "moving", 0), State("inside")),
      (("release", "a2", "outside"), State("outside-release")),
      (("assemble", "a1", "moving", "target"), State("inside-assemble")),
      (("assemble", "a1", "moving", "outside"), State("outside-assemble")),
  ]
  monkeypatch.setattr(solver_mod, "successors", lambda *_args, **_kwargs: actions)
  monkeypatch.setattr(
      solver_mod, "_pea_star",
      lambda *_args, **_kwargs: pytest.fail("local LNS must not use PEA*"))
  calls = []

  def fake_astar(_start, **kwargs):
    calls.append(kwargs)
    assert [a for a, _ns in kwargs["succ_fn"](start)] == [actions[0][0], actions[2][0]]
    assert kwargs["max_nodes"] == 20000
    assert kwargs["cost_bound"] == 7
    assert kwargs["tag"] == "lns refill A*"
    kwargs["stats_out"].update(outcome="found", expanded=3, edge_checks=2,
                                peak_frontier=4)
    plan = solver_mod.Plan()
    plan.states = [start]
    return plan

  monkeypatch.setattr(solver_mod, "_astar", fake_astar)
  common = dict(
      start=start, touched_parts={"moving", "target"}, target=target,
      assembly=Assembly(), robots=("a1", "a2"), checker=object(), grasp_table=object(),
      weight=5.0, max_nodes=20000, h_lab=Heuristic(), cost_bound=7)
  assert solver_mod._refill_window(robot_ik=None, **common) == []

  engines = []
  class FakeIncremental:
    def __init__(self, _start, **kwargs):
      engines.append(kwargs)
      assert [a for a, _ns in kwargs["succ_fn"](start)] == [actions[0][0], actions[2][0]]
      assert kwargs["cost_bound"] == 7
      self.expanded = 3
      self.budget_hit = False
      self.repairable = kwargs.get("repairable", True)

    def search(self, max_nodes, deadline=None):
      assert max_nodes == 20000
      plan = solver_mod.Plan()
      plan.states = [start]
      return plan

    def footprint(self):
      return {"frontier": 2, "heap_entries": 4,
              "peak_frontier": 3, "peak_heap_entries": 4}

  monkeypatch.setattr(solver_mod, "_IncrementalLazySearch", FakeIncremental)
  assert solver_mod._refill_window(robot_ik=object(), **common) == []
  assert len(calls) == 2
  assert len(engines) == 0                 # no IK failure -> no repair graph allocated


def test_lns_refill_memoizes_identical_heuristic_and_cheap_queries(monkeypatch):
  class State(_State):
    def key(self):
      return self.name

    def is_goal(self, _assembly):
      return False

  class Heuristic:
    def __init__(self):
      self.calls = []

    def value(self, state):
      self.calls.append(state.name)
      return 0.0

  class Assembly:
    objects = ("moving",)

  start, target, nxt = State("start"), State("target"), State("next")
  action = ("pick", "a1", "moving", 0)
  heuristic = Heuristic()
  cheap_calls = []
  successor_calls = []
  def successors(*_args, **_kwargs):
    successor_calls.append(1)
    return [(action, nxt)]
  monkeypatch.setattr(solver_mod, "successors", successors)
  monkeypatch.setattr(
      solver_mod, "cheap_feasible",
      lambda act, prev, ns, *_args: cheap_calls.append((prev, act, ns)) or True)

  def fake_astar(_start, **kwargs):
    assert kwargs["h_fn"](start) == kwargs["h_fn"](start) == 0.0
    edge = next(iter(kwargs["succ_fn"](start)))
    assert next(iter(kwargs["succ_fn"](start))) == edge
    assert next(iter(kwargs["succ_fn"](start))) == edge
    assert kwargs["edge_check_fn"](start, *edge) is True
    assert kwargs["edge_check_fn"](start, *edge) is True
    kwargs["stats_out"].update(outcome="found", expanded=1, edge_checks=2,
                                peak_frontier=1)
    return []

  monkeypatch.setattr(solver_mod, "_astar", fake_astar)
  stats = {}
  assert solver_mod._refill_window(
      start, {"moving"}, target, Assembly(), ("a1",), object(), object(), None,
      1.0, 10, heuristic, stats=stats) == []
  # target is evaluated once for the differential offset; the duplicate start query and
  # duplicate exact edge gate are each computed only once.
  assert heuristic.calls == ["target", "start"]
  assert len(cheap_calls) == 1
  assert stats["heuristic_calls"] == 2
  assert stats["heuristic_misses"] == 1
  assert stats["cheap_checks"] == 2
  assert stats["cheap_cache_hits"] == 1
  # Cache-on-second-access: the second request admits, and the third hits.
  assert len(successor_calls) == 2
  assert stats["successor_calls"] == 3
  assert stats["successor_cache_hits"] == 1


def test_lns_uses_weight_one_cleanup_then_weighted_wider_windows(monkeypatch, capsys):
  class State(_State):
    def key(self):
      return self.name

    def subassemblies(self):
      if self.name == "s0":
        return (frozenset({"obj_0"}), frozenset({"obj_1"}), frozenset({"obj_2"}))
      if self.name == "s1":
        return (frozenset({"obj_0", "obj_1"}), frozenset({"obj_2"}))
      return (frozenset({"obj_0", "obj_1", "obj_2"}),)

  s0, s1, s2 = State("s0"), State("s1"), State("s2")
  a1 = ("assemble", "a1", "obj_1", "obj_0")
  a2 = ("assemble", "a1", "obj_2", "obj_0")
  graph = {s0: [(a1, s1)], s1: [(a2, s2)], s2: []}

  class Assembly:
    base = "obj_0"
    objects = ("obj_0", "obj_1", "obj_2")

  class H:
    def value(self, _state):
      return 0.0

  monkeypatch.setattr(solver_mod, "get_initial_state", lambda *_args: s0)
  monkeypatch.setattr(
      solver_mod, "successors", lambda state, *_args, **_kwargs: graph[state])
  monkeypatch.setattr(
      solver_mod, "_part_mated",
      lambda state, part: part == "obj_0" or part == "obj_1" and state in (s1, s2)
      or part == "obj_2" and state == s2)
  calls = []

  def fake_refill(*_args, **kwargs):
    report = kwargs["stats"]
    report.update(outcome="exhausted", wall=0.01, search_calls=1, expanded=7,
                  peak_frontier=11, successors_scoped=13, successors_seen=17,
                  cheap_checks=5, ik_binds=2, ik_failures=1)
    calls.append((kwargs.get("deadline"), _args[8]))       # positional weight
    return None

  monkeypatch.setattr(solver_mod, "_refill_window", fake_refill)
  import time
  end = time.perf_counter() + 30.0
  plan = solver_mod._lns_improve(
      [a1, a2], object(), Assembly(), ("a1",), object(), object(), None,
      weight=5.0, max_nodes=20000, verbose=True, h_lab=H(), deadline=end,
      k_max=2)
  assert plan == [a1, a2]
  assert [weight for _deadline, weight in calls] == [
      float("inf"), 1.0, float("inf"), 1.0, float("inf"), 1.0]
  # Both passes share the run's global wall deadline; reproducible node budgets, rather
  # than six-second per-window slices, bound each consultation.
  assert all(calls[j][0] == calls[j + 1][0] == end for j in (0, 2, 4))
  out = capsys.readouterr().out
  assert out.count("[lns-window]") == 6
  assert "pass=greedy w=inf" in out and "pass=exact w=1" in out
  assert "expanded=7" in out and "peak-open=11" in out and "IK=2(1f)" in out


def test_lns_visits_largest_admissible_gap_first_at_each_window_size(monkeypatch, capsys):
  class State(_State):
    def __init__(self, name, mated):
      super().__init__(name)
      self.mated = frozenset(mated)

    def subassemblies(self):
      main = frozenset({"obj_0"}) | self.mated
      return ((main,) if "obj_2" in self.mated else
              (main, frozenset({"obj_2"})))

  s0 = State("s0", ())
  s1 = State("s1", ("obj_1",))
  s1a = State("s1a", ("obj_1",))
  s1b = State("s1b", ("obj_1",))
  s2 = State("s2", ("obj_1", "obj_2"))
  a1 = ("assemble", "a1", "obj_1", "obj_0")
  h1 = ("handover", "a2", "obj_2", 1)
  h2 = ("handover", "a1", "obj_2", 2)
  a2 = ("assemble", "a1", "obj_2", "obj_0")
  graph = {s0: [(a1, s1)], s1: [(h1, s1a)], s1a: [(h2, s1b)],
           s1b: [(a2, s2)], s2: []}

  class Assembly:
    base = "obj_0"
    objects = ("obj_0", "obj_1", "obj_2")

  class H:
    def value(self, _state):
      return 0.0

  monkeypatch.setattr(solver_mod, "get_initial_state", lambda *_args: s0)
  monkeypatch.setattr(
      solver_mod, "successors", lambda state, *_args, **_kwargs: graph[state])
  monkeypatch.setattr(
      solver_mod, "_part_mated",
      lambda state, part: part == "obj_0" or part in state.mated)
  monkeypatch.setattr(solver_mod, "_make_lns_endpoint_pdb", lambda *_args: None)
  monkeypatch.setattr(solver_mod, "_release_fill_scratch", lambda *_args, **_kwargs: None)
  starts = []

  def fake_refill(*args, **kwargs):
    starts.append(args[0].name)
    kwargs["stats"].update(outcome="exhausted", wall=0.0)
    return None

  monkeypatch.setattr(solver_mod, "_refill_window", fake_refill)
  plan = solver_mod._lns_improve(
      [a1, h1, h2, a2], object(), Assembly(), ("a1", "a2"), object(), object(),
      None, weight=5.0, max_nodes=20000, verbose=True, h_lab=H(), k_max=1)

  assert plan == [a1, h1, h2, a2]
  # The second segment costs 5 versus the first segment's 1. Both have lower bound zero,
  # hence both greedy/exact consultations for the second segment precede the first.
  assert starts == ["s1", "s1", "s0", "s0"]
  out = capsys.readouterr().out
  assert ("consultation=1 pending=1 k=1 parts[1:2] ('obj_2',) "
          "bound=5 landmark-lb=0 gap=5 nodes=512") in out


def test_lns_reopens_a_node_capped_window_with_geometric_effort(monkeypatch, capsys):
  class State(_State):
    def __init__(self, name, done=False):
      super().__init__(name)
      self.done = done

    def subassemblies(self):
      return ((frozenset({"obj_0", "obj_1"}),) if self.done else
              (frozenset({"obj_0"}), frozenset({"obj_1"})))

  s0, s1 = State("s0"), State("s1", True)
  action = ("assemble", "a1", "obj_1", "obj_0")

  class Assembly:
    base = "obj_0"
    objects = ("obj_0", "obj_1")

  class H:
    def value(self, _state):
      return 0.0

  monkeypatch.setattr(solver_mod, "get_initial_state", lambda *_args: s0)
  monkeypatch.setattr(
      solver_mod, "successors",
      lambda state, *_args, **_kwargs: [(action, s1)] if state is s0 else [])
  monkeypatch.setattr(
      solver_mod, "_part_mated",
      lambda state, part: part == "obj_0" or state.done)
  monkeypatch.setattr(solver_mod, "_release_fill_scratch", lambda *_args, **_kwargs: None)
  calls = []

  def fake_refill(*args, **kwargs):
    calls.append((args[8], args[9]))
    # First greedy/exact consultation reaches its cap. The reopened consultation then
    # exhausts, which closes the window without another retry.
    kwargs["stats"].update(outcome="budget" if len(calls) <= 2 else "exhausted", wall=0.0)
    return None

  monkeypatch.setattr(solver_mod, "_refill_window", fake_refill)
  out = solver_mod._lns_improve(
      [action], object(), Assembly(), ("a1",), object(), object(), None,
      weight=5.0, max_nodes=2048, verbose=True, h_lab=H(), k_max=1)

  assert out == [action]
  assert calls == [(float("inf"), 512), (1.0, 512),
                   (float("inf"), 1024), (1.0, 1024)]
  assert "reopen k=1 parts[0:1] ('obj_1',) nodes=512->1024" in capsys.readouterr().out


def test_lns_endpoint_pdb_bound_skips_only_exact_splice_window(monkeypatch, capsys):
  class State(_State):
    def key(self):
      return self.name

    def subassemblies(self):
      if self.name == "s0":
        return (frozenset({"obj_0"}), frozenset({"obj_1"}), frozenset({"obj_2"}))
      if self.name == "s1":
        return (frozenset({"obj_0", "obj_1"}), frozenset({"obj_2"}))
      return (frozenset({"obj_0", "obj_1", "obj_2"}),)

  s0, s1, s2 = State("s0"), State("s1"), State("s2")
  a1 = ("assemble", "a1", "obj_1", "obj_0")
  a2 = ("assemble", "a1", "obj_2", "obj_0")
  graph = {s0: [(a1, s1)], s1: [(a2, s2)], s2: []}

  class Assembly:
    base = "obj_0"
    objects = ("obj_0", "obj_1", "obj_2")

  class Landmark:
    def value(self, _state):
      return 0.0

  class Endpoint:
    def value(self, _state):
      return float("inf")

    def reverse_stats(self):
      return {"settled": 4, "discovered": 4, "reverse_time": 0.001}

  monkeypatch.setattr(solver_mod, "get_initial_state", lambda *_args: s0)
  monkeypatch.setattr(
      solver_mod, "successors", lambda state, *_args, **_kwargs: graph[state])
  monkeypatch.setattr(
      solver_mod, "_part_mated",
      lambda state, part: part == "obj_0" or part == "obj_1" and state in (s1, s2)
      or part == "obj_2" and state == s2)
  endpoint_calls = []

  def fake_endpoint(start, target, chain, *_args):
    endpoint_calls.append((start, target, chain))
    return Endpoint()

  refills = []

  def fake_refill(*args, **kwargs):
    refills.append((args, kwargs))
    kwargs["stats"].update(outcome="exhausted", wall=0.0)
    return None

  monkeypatch.setattr(solver_mod, "_make_lns_endpoint_pdb", fake_endpoint)
  monkeypatch.setattr(solver_mod, "_refill_window", fake_refill)
  plan = solver_mod._lns_improve(
      [a1, a2], object(), Assembly(), ("a1",), object(), object(), None,
      weight=5.0, max_nodes=20000, verbose=True, h_lab=Landmark(), k_max=1)

  assert plan == [a1, a2]
  assert endpoint_calls == [(s0, s1,
                             [("obj_0",), ("obj_0", "obj_1")])]
  # The final window has a set-valued assembly goal, not the exact incumbent endpoint.
  assert len(refills) == 2
  assert all(call[0][0] is s1 and call[0][2] is None for call in refills)
  assert [call[0][8] for call in refills] == [float("inf"), 1.0]
  out = capsys.readouterr().out
  assert "outcome=endpoint-bound" in out
  assert "status=bounded-out" in out


def test_lns_greedy_incumbent_tightens_the_weight_one_bound(monkeypatch):
  class State(_State):
    def key(self):
      return self.name

    def subassemblies(self):
      return (("obj_0",), ("obj_1",)) if self.name == "s0" else (("obj_0", "obj_1"),)

  s0, s1 = State("s0"), State("s1")
  assemble = ("assemble", "a1", "obj_1", "obj_0")

  class Assembly:
    base = "obj_0"
    objects = ("obj_0", "obj_1")

  class H:
    def value(self, _state):
      return 0.0

  monkeypatch.setattr(solver_mod, "get_initial_state", lambda *_args: s0)
  monkeypatch.setattr(
      solver_mod, "successors",
      lambda state, *_args, **_kwargs: [(assemble, s1)] if state is s0 else [])
  monkeypatch.setattr(
      solver_mod, "_part_mated",
      lambda state, part: part == "obj_0" or state is s1)
  calls = []

  def fake_refill(*args, **kwargs):
    calls.append((args[8], kwargs["cost_bound"]))
    kwargs["stats"].update(outcome="found", wall=0.0)
    return [] if args[8] == float("inf") else None

  monkeypatch.setattr(solver_mod, "_refill_window", fake_refill)
  out = solver_mod._lns_improve(
      [assemble], object(), Assembly(), ("a1",), object(), object(), None,
      weight=5.0, max_nodes=20000, verbose=False, h_lab=H())

  assert out == []
  assert calls == [(float("inf"), solver_mod.action_cost(assemble)), (1.0, 0)]


def test_lns_memory_exhaustion_preserves_the_validated_incumbent(monkeypatch):
  class State(_State):
    def key(self):
      return self.name

    def subassemblies(self):
      return (("obj_0",), ("obj_1",)) if self.name == "s0" else (("obj_0", "obj_1"),)

  s0, s1 = State("s0"), State("s1")
  assemble = ("assemble", "a1", "obj_1", "obj_0")

  class Assembly:
    base = "obj_0"
    objects = ("obj_0", "obj_1")

  class H:
    def value(self, _state):
      return 0.0

  monkeypatch.setattr(solver_mod, "get_initial_state", lambda *_args: s0)
  monkeypatch.setattr(
      solver_mod, "successors",
      lambda state, *_args, **_kwargs: [(assemble, s1)] if state is s0 else [])
  monkeypatch.setattr(
      solver_mod, "_part_mated",
      lambda state, part: part == "obj_0" or state is s1)
  monkeypatch.setattr(
      solver_mod, "_refill_window",
      lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError()))
  # In particular, the emergency path must not call the allocating RSS/scratch reporter.
  monkeypatch.setattr(
      solver_mod, "_release_fill_scratch",
      lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unsafe cleanup")))

  out = solver_mod._lns_improve(
      [assemble], object(), Assembly(), ("a1",), object(), object(), None,
      weight=5.0, max_nodes=20000, verbose=False, h_lab=H())

  assert out == [assemble]


def test_open_scheduler_replans_immediately_after_first_failed_plan_edge():
  s0, s1, s2 = _State("s0"), _State("s1"), _State("s2")
  a = ("pick", "a1", "a", 0)
  b = ("pick", "a1", "b", 0)
  calls = []

  class RIK:
    def _scene_sig(self, state):
      return state.name

    def probe_bind_once(self, action, _state, _successor, _assembly, attempt):
      calls.append((action, attempt))
      return False, None

  class Engine:
    def __init__(self, suspended):
      self.suspended = suspended

    def search(self, _max_nodes, deadline=None):
      return None if (s0, a) in self.suspended else [a, b]

    def delete_edge(self, state, action, successor):
      assert (state, action, successor) == (s0, a, s1)

  def walk(_plan):
    yield s0, a, s1
    yield s1, b, s2

  plan, end, report = run_open_lazy_fill(
      robot_ik=RIK(), assembly=object(), make_engine=Engine, walk_plan=walk,
      is_ik=lambda _a: True, max_nodes=10, max_epochs=1)
  assert (plan, end) == (None, None)
  assert calls == [(a, 0)]                               # b belongs to an invalid incumbent
  assert report.open_failures == 1


def test_reverse_selector_checks_the_candidate_suffix_first():
  s0, s1, s2 = _State("s0"), _State("s1"), _State("s2")
  a = ("pick", "a1", "obj", 0)
  b = ("assemble", "a1", "obj", "base")
  calls = []

  class RIK:
    lazy_edge_selector = "reverse"

    def _scene_sig(self, state):
      return state.name

    def probe_bind_once(self, action, _state, _successor, _assembly, attempt):
      calls.append((action, attempt))
      return False, None

  class Engine:
    def __init__(self, suspended):
      self.suspended = suspended

    def search(self, _max_nodes, deadline=None):
      return None if (s1, b) in self.suspended else [a, b]

    def delete_edge(self, state, action, successor):
      assert (state, action, successor) == (s1, b, s2)

  def walk(_plan):
    yield s0, a, s1
    yield s1, b, s2

  plan, end, report = run_open_lazy_fill(
      robot_ik=RIK(), assembly=object(), make_engine=Engine, walk_plan=walk,
      is_ik=lambda _a: True, max_nodes=10, max_epochs=1)
  assert (plan, end) == (None, None)
  assert calls == [(b, 0)]
  assert report.reordered_paths == 1


def test_terminal_first_checks_last_assemble_then_forward_prefix():
  s0, s1, s2, s3 = [_State(f"s{i}") for i in range(4)]
  a = ("pick", "a1", "obj", 0)
  b = ("handover", "a2", "obj", 1)
  c = ("assemble", "a2", "obj", "base")
  calls = []

  class RIK:
    lazy_edge_selector = "terminal-first"

    def _scene_sig(self, state):
      return state.name

    def probe_bind_once(self, action, _state, _successor, _assembly, attempt):
      calls.append((action, attempt))
      return action == c, object() if action == c else None

  class Engine:
    def __init__(self, suspended):
      self.suspended = suspended

    def search(self, _max_nodes, deadline=None):
      return None if (s0, a) in self.suspended else [a, b, c]

    def delete_edge(self, state, action, successor):
      assert (state, action, successor) == (s0, a, s1)

  def walk(_plan):
    yield s0, a, s1
    yield s1, b, s2
    yield s2, c, s3

  plan, end, report = run_open_lazy_fill(
      robot_ik=RIK(), assembly=object(), make_engine=Engine, walk_plan=walk,
      is_ik=lambda _a: True, max_nodes=10, max_epochs=1)
  assert (plan, end) == (None, None)
  assert calls == [(c, 0), (a, 0)]
  assert report.successes == 1 and report.open_failures == 1
  assert report.reordered_paths == 1


def test_terminal_first_is_forward_when_candidate_has_no_assemble():
  s0, s1, s2 = _State("s0"), _State("s1"), _State("s2")
  a = ("pick", "a1", "obj", 0)
  b = ("place", "a1", "obj", 1)
  calls = []

  class RIK:
    lazy_edge_selector = "terminal-first"

    def _scene_sig(self, state):
      return state.name

    def probe_bind_once(self, action, _state, _successor, _assembly, attempt):
      calls.append(action)
      return False, None

  class Engine:
    def __init__(self, suspended):
      self.suspended = suspended

    def search(self, _max_nodes, deadline=None):
      return None if (s0, a) in self.suspended else [a, b]

    def delete_edge(self, *_args):
      pass

  def walk(_plan):
    yield s0, a, s1
    yield s1, b, s2

  run_open_lazy_fill(robot_ik=RIK(), assembly=object(), make_engine=Engine,
                     walk_plan=walk, is_ik=lambda _a: True, max_nodes=10, max_epochs=1)
  assert calls == [a]


def test_fail_first_uses_exact_scene_failure_before_reverse_fallback():
  s0, s1, s2, s3 = [_State(f"s{i}") for i in range(4)]
  a = ("pick", "a1", "obj", 0)
  b = ("handover", "a2", "obj", 1)
  c = ("assemble", "a2", "obj", "base")
  calls = []

  class RIK:
    lazy_edge_selector = "fail-first"
    open_witness_batch = 2

    def _scene_sig(self, state):
      return state.name

    def probe_bind_batch(self, action, _state, _successor, _assembly, attempt, budget):
      calls.append((action, attempt, budget))
      return False, None

  rik = RIK()
  pool = rik._open_witness_pool = OpenWitnessPool(rik)
  # Only this exact pick-in-s0 query has prior failed effort. An identical action in another
  # support scene would have a different key and must not inherit the priority.
  ka = pool.key(s0, a)
  pool.visits[ka], pool.attempts[ka] = 2, 6

  class Engine:
    def __init__(self, suspended):
      self.suspended = suspended

    def search(self, _max_nodes, deadline=None):
      return None if (s0, a) in self.suspended else [a, b, c]

    def delete_edge(self, state, action, successor):
      assert (state, action, successor) == (s0, a, s1)

  def walk(_plan):
    yield s0, a, s1
    yield s1, b, s2
    yield s2, c, s3

  plan, end, report = run_open_lazy_fill(
      robot_ik=rik, assembly=object(), make_engine=Engine, walk_plan=walk,
      is_ik=lambda _a: True, max_nodes=10, max_epochs=1)
  assert (plan, end) == (None, None)
  assert calls == [(a, 6, 2)]
  assert report.prior_failure_first == 1


def test_fail_first_unseen_tie_is_reverse_and_cached_success_is_skipped():
  s0, s1, s2 = _State("s0"), _State("s1"), _State("s2")
  a = ("pick", "a1", "obj", 0)
  b = ("assemble", "a1", "obj", "base")
  calls = []

  class RIK:
    lazy_edge_selector = "fail-first"
    open_witness_batch = 2

    def _scene_sig(self, state):
      return state.name

    def probe_bind_batch(self, action, _state, _successor, _assembly, attempt, budget):
      calls.append(action)
      return True, object()

  rik = RIK()
  pool = rik._open_witness_pool = OpenWitnessPool(rik)
  pool.success[pool.key(s1, b)] = object()

  class Engine:
    def __init__(self, _suspended):
      pass

    def search(self, _max_nodes, deadline=None):
      return [a, b]

    def delete_edge(self, *_args):
      raise AssertionError("successful candidate must not delete an edge")

  def walk(_plan):
    yield s0, a, s1
    yield s1, b, s2

  plan, end, report = run_open_lazy_fill(
      robot_ik=rik, assembly=object(), make_engine=Engine, walk_plan=walk,
      is_ik=lambda _a: True, max_nodes=10, max_epochs=1)
  assert plan == [a, b] and end == s2
  assert calls == [a]                                   # cached terminal witness was skipped
  assert report.successes == 1


def test_defer_failed_probes_one_physical_alias_per_epoch_then_moves_on():
  s = _State("s")
  n1, n2, ng = _State("n1"), _State("n2"), _State("ng")
  a1 = ("pick", "a1", "obj", 0)
  a2 = ("pick", "a2", "obj", 0)
  good = ("pick", "a3", "other", 0)
  edges = {a1: (s, a1, n1), a2: (s, a2, n2), good: (s, good, ng)}
  calls = []

  class RIK:
    lazy_edge_selector = "defer-failed"

    def _query_key(self, action, _state, _successor, _assembly):
      return "same-failed-pick" if action in (a1, a2) else "good-pick"

    def probe_bind_once(self, action, _state, _successor, _assembly, attempt):
      calls.append((action, attempt))
      return (action == good), ("witness" if action == good else None)

  class Engine:
    def __init__(self, suspended):
      self.suspended = suspended

    def search(self, _max_nodes, deadline=None):
      for action in (a1, a2, good):
        if (s, action) not in self.suspended:
          return [action]
      return None

    def delete_edge(self, state, action, successor):
      assert (state, action, successor) == edges[action]

  def walk(plan):
    yield edges[plan[0]]

  plan, end, report = run_open_lazy_fill(
      robot_ik=RIK(), assembly=object(), make_engine=Engine, walk_plan=walk,
      is_ik=lambda _a: True, max_nodes=10, max_epochs=1)
  assert (plan, end) == ([good], ng)
  assert calls == [(a1, 0), (good, 0)]       # a2 aliases a1 and receives no second IK call
  assert report.open_failures == 1 and report.successes == 1
  assert report.physical_deferrals == 1 and report.deferred_physical_queries == 1


def test_defer_failed_reopens_the_physical_query_in_the_next_epoch():
  s, n = _State("s"), _State("n")
  action = ("pick", "a1", "obj", 0)
  calls = []

  class RIK:
    lazy_edge_selector = "defer-failed"

    def _query_key(self, _action, _state, _successor, _assembly):
      return "physical-pick"

    def probe_bind_once(self, _action, _state, _successor, _assembly, attempt):
      calls.append(attempt)
      return attempt == 1, ("witness" if attempt == 1 else None)

  class Engine:
    def __init__(self, suspended):
      self.suspended = suspended

    def search(self, _max_nodes, deadline=None):
      return None if (s, action) in self.suspended else [action]

    def delete_edge(self, state, act, successor):
      assert (state, act, successor) == (s, action, n)

  def walk(_plan):
    yield s, action, n

  plan, end, report = run_open_lazy_fill(
      robot_ik=RIK(), assembly=object(), make_engine=Engine, walk_plan=walk,
      is_ik=lambda _a: True, max_nodes=10, max_epochs=2)
  assert (plan, end) == ([action], n)
  assert calls == [0, 1]
  assert report.epochs == 2 and report.open_failures == 1


def test_skill_specific_effort_ladders_keep_assemble_bundle_atomic():
  calls = []

  class RIK:
    open_witness_batch = 7

    def _scene_sig(self, state):
      return state.name

    def probe_bind_batch(self, action, _state, _successor, _assembly, attempt, budget):
      calls.append((action[0], attempt, budget))
      return False, None

  pool = OpenWitnessPool(RIK())
  s, n = _State("s"), _State("n")
  pick = ("pick", "a1", "obj", 0)
  assemble = ("assemble", "a1", "obj", "onto")
  for _ in range(3):
    pool.probe(pick, s, n, object())
  for _ in range(3):
    pool.probe(assemble, s, n, object())
  assert calls == [
      ("pick", 0, 2), ("pick", 2, 4), ("pick", 6, 7),
      ("assemble", 0, 7), ("assemble", 7, 7), ("assemble", 14, 7),
  ]


def test_open_pool_retains_the_exact_successful_witness():
  witness = object()

  class RIK:
    open_witness_batch = 2

    def _scene_sig(self, state):
      return state.name

    def probe_bind_batch(self, _action, _state, _successor, _assembly, attempt, budget):
      assert (attempt, budget) == (0, 2)
      return True, witness

  pool = OpenWitnessPool(RIK())
  s, n = _State("s"), _State("n")
  action = ("pick", "a1", "obj", 0)
  assert pool.probe(action, s, n, object()) == (True, witness, 2)
  assert pool.witness_for(s, action) is witness
  # A later validation of the same edge returns the certificate, not merely a bool.
  assert pool.probe(action, s, n, object()) == (True, witness, False)


def test_open_pool_shares_attempt_cursor_for_identical_physical_query():
  calls = []

  class RIK:
    open_witness_batch = 2

    def _query_key(self, _action, _state, _successor, _assembly):
      return ("same-physical-query",)

    def probe_bind_batch(self, _action, _state, _successor, _assembly, attempt, budget):
      calls.append((attempt, budget))
      return False, None

  assembly = object()
  pool = OpenWitnessPool(RIK(), assembly)
  action = ("pick", "a1", "obj", 0)
  pool.probe(action, _State("symbolic-a"), _State("next-a"), assembly)
  pool.probe(action, _State("symbolic-b"), _State("next-b"), assembly)

  # The second symbolic alias continues at fresh effort. It does not replay attempt zero or
  # the analytic initialization bundle.
  assert calls == [(0, 2), (2, 2)]


def test_open_pool_retains_failure_effort_across_fill_sessions():
  calls = []

  class RIK:
    lazy_edge_selector = "forward"
    open_witness_batch = 2

    def _query_key(self, _action, _state, _successor, _assembly):
      return ("same-physical-query",)

    def probe_bind_batch(self, _action, _state, _successor, _assembly, attempt, budget):
      calls.append((attempt, budget))
      return False, None

  rik = RIK()
  assembly = object()
  s0, s1 = _State("symbolic-a"), _State("next-a")
  action = ("pick", "a1", "obj", 0)

  class Engine:
    budget_hit = False

    def __init__(self, suspended):
      self.suspended = suspended

    def search(self, _max_nodes, deadline=None):
      return None if (s0, action) in self.suspended else [action]

    def delete_edge(self, *_args):
      pass

  def walk(_plan):
    yield s0, action, s1

  for tag in ("obj_1", "obj_2"):
    run_open_lazy_fill(
        robot_ik=rik, assembly=assembly, make_engine=Engine, walk_plan=walk,
        is_ik=lambda _a: True, max_nodes=10, max_epochs=1, tag=tag)

  pool = rik._open_witness_pool
  assert calls == [(0, 2), (2, 2)]
  assert pool.sessions == 2
  assert pool.revisit_probes == 1
  assert pool.cross_session_revisit_probes == 1
  assert pool.cross_tag_revisit_probes == 1
  assert pool.prior_slots_not_replayed == 2


def test_physical_alias_revisit_uses_fresh_effort_in_the_same_epoch():
  calls = []
  a = ("pick", "a1", "obj", 0)
  b = ("pick", "a1", "obj", 1)
  s0, s1, s2 = _State("s0"), _State("s1"), _State("s2")

  class RIK:
    lazy_edge_selector = "forward"
    open_witness_batch = 2

    def _query_key(self, _action, _state, _successor, _assembly):
      return ("same-physical-query",)

    def probe_bind_batch(self, action, _state, _successor, _assembly, attempt, budget):
      calls.append((action, attempt, budget))
      return False, None

  class Engine:
    budget_hit = False

    def __init__(self, suspended):
      self.suspended = suspended

    def search(self, _max_nodes, deadline=None):
      if (s0, a) not in self.suspended:
        return [a]
      if (s0, b) not in self.suspended:
        return [b]
      return None

    def delete_edge(self, *_args):
      pass

  def walk(plan):
    action = plan[0]
    yield s0, action, s1 if action == a else s2

  rik = RIK()
  _plan, _end, report = run_open_lazy_fill(
      robot_ik=rik, assembly=object(), make_engine=Engine, walk_plan=walk,
      is_ik=lambda _a: True, max_nodes=10, max_epochs=1, tag="obj")

  # A finite stochastic failure is not a reusable negative verdict. The alias shares the
  # cursor, so it skips attempt zero but receives genuinely fresh effort immediately.
  assert calls == [(a, 0, 2), (b, 2, 2)]
  assert report.open_failures == 2
  assert rik._open_witness_pool.revisit_probes == 1


def test_zero_epochs_requires_a_deadline():
  class RIK:
    pass
  try:
    run_open_lazy_fill(robot_ik=RIK(), assembly=None, make_engine=None,
                       walk_plan=None, is_ik=None, max_nodes=1, max_epochs=0)
  except ValueError as exc:
    assert "deadline" in str(exc)
  else:
    raise AssertionError("unbounded open-witness loop accepted without a deadline")
