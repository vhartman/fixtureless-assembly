from open_witness import OpenWitnessPool, run_open_lazy_fill
from solver import RobotIK


class _State:
  def __init__(self, name):
    self.name = name

  def __hash__(self):
    return hash(self.name)

  def __eq__(self, other):
    return isinstance(other, _State) and self.name == other.name


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
