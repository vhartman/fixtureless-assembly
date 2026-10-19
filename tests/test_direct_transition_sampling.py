"""Focused tests for the direct informed-transition sampling experiment."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _module():
  path = (Path(__file__).parents[1] / "experiments" /
          "direct-informed-transition-sampling" / "direct_transition_sampling.py")
  spec = importlib.util.spec_from_file_location("direct_transition_sampling", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _factored_module():
  path = (Path(__file__).parents[1] / "experiments" /
          "direct-informed-transition-sampling" / "persistent_factored_sampler.py")
  spec = importlib.util.spec_from_file_location("persistent_factored_sampler", path)
  module = importlib.util.module_from_spec(spec)
  # Dataclasses resolve their defining module through sys.modules during decoration.
  import sys
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def test_phs_samples_obey_two_focus_bound_and_are_repeatable():
  M = _module()
  a = np.array([-1.2, 0.3, 0.5])
  b = np.array([0.8, -0.4, 1.1])
  c = np.linalg.norm(b - a) + 1.7
  np.random.seed(17)
  x1 = M.sample_robot_phs(a, b, c, n=1000)
  np.random.seed(17)
  x2 = M.sample_robot_phs(a, b, c, n=1000)
  assert np.array_equal(x1, x2)
  lengths = np.linalg.norm(x1 - a[:, None], axis=0) + np.linalg.norm(
      x1 - b[:, None], axis=0)
  assert np.max(lengths) <= c + 1e-10


def test_phs_handles_coincident_empty_and_degenerate_cases():
  M = _module()
  a = np.array([0.2, -0.7])
  np.random.seed(2)
  ball = M.sample_robot_phs(a, a, 1.4, n=200)
  assert np.all(np.isfinite(ball))
  assert np.max(np.linalg.norm(ball - a[:, None], axis=0)) <= 0.7 + 1e-10

  b = np.array([1.2, -0.7])
  assert M.sample_robot_phs(a, b, 0.9, n=2) is None
  line = M.sample_robot_phs(a, b, 1.0, n=200)
  assert np.all(np.isfinite(line))
  assert np.max(np.abs(line[1] - a[1])) < 1e-12
  lengths = np.linalg.norm(line - a[:, None], axis=0) + np.linalg.norm(
      line - b[:, None], axis=0)
  assert np.max(np.abs(lengths - 1.0)) < 1e-10


def test_sum_completion_pins_task_robot_and_allocates_one_joint_budget():
  M = _module()
  from assembly_planner.mp_export import _ensure_mrbench
  _ensure_mrbench()
  from multi_robot_multi_goal_planning.problems.core.configuration import NpConfiguration

  robots = ["a", "b", "c"]
  env = SimpleNamespace(robots=robots, cost_metric="euclidean", cost_reduction="sum")
  sampler = SimpleNamespace(env=env, conf_type=NpConfiguration)
  q0 = NpConfiguration.from_list([np.array([0.0, 0.0])] * 3)
  q1 = NpConfiguration.from_list([np.array([1.0, 0.0])] * 3)
  path = [SimpleNamespace(q=q0), SimpleNamespace(q=q1)]
  pinned = {0: np.array([0.25, 0.0])}
  np.random.seed(5)
  q, reason, bounds = M.direct_completion(
      sampler, path, 0, 1, current_cost=7.0, pinned=pinned)
  assert reason is None
  assert np.array_equal(q[0], pinned[0])
  assert set(bounds) == {1, 2}
  assert all(v >= 1.0 for v in bounds.values())
  assert abs(sum(bounds.values()) + 1.0 - 7.0) < 1e-10
  total = sum(np.linalg.norm(q0[i] - q[i]) + np.linalg.norm(q[i] - q1[i])
              for i in range(3))
  assert total <= 7.0 + 1e-10

  impossible, reason, _ = M.direct_completion(
      sampler, path, 0, 1, current_cost=2.9, pinned=pinned)
  assert impossible is None and reason == "infeasible_budget"


class _Goal:
  def sample(self, _mode):
    return np.array([0.5])


class _FixedGoal:
  def __init__(self, *values):
    self.values = np.asarray(values, dtype=float)

  def sample(self, _mode):
    return self.values.copy()


class _TinyEnv:
  def __init__(self, metric="euclidean"):
    from multi_robot_multi_goal_planning.problems.core.configuration import NpConfiguration
    self.robots = ["active", "free1", "free2"]
    self.robot_dims = {r: 1 for r in self.robots}
    self.robot_idx = {r: np.array([i]) for i, r in enumerate(self.robots)}
    self.start_pos = NpConfiguration.from_list([np.zeros(1) for _ in self.robots])
    self.limits = np.array([[-20.0] * 3, [20.0] * 3])
    self.cost_metric = metric
    self.cost_reduction = "max"
    self.tasks = [object(), object()]
    self.uniform_calls = 0
    self.collision_calls = 0
    self.task = SimpleNamespace(robots=["active"], goal=_Goal())

  def sample_config_uniform_in_limits(self):
    self.uniform_calls += 1
    return self.start_pos.from_list([np.zeros(1), np.zeros(1), np.zeros(1)])

  def batch_config_cost(self, starts, ends):
    def flat(x):
      if hasattr(x, "q") and not hasattr(x, "state"):
        x = x.q
      return x.state() if hasattr(x, "state") else np.asarray(x)
    if isinstance(starts, (list, tuple)):
      return np.array([np.max(np.abs(flat(a) - flat(b))) + 0.01 *
                       np.sum(np.abs(flat(a) - flat(b)))
                       for a, b in zip(starts, ends)])
    s = flat(starts)
    return np.array([np.max(np.abs(s - flat(e))) + 0.01 * np.sum(np.abs(s - flat(e)))
                     for e in ends])

  def config_cost(self, a, b):
    return float(self.batch_config_cost(a, [b])[0])

  def get_valid_next_task_combinations(self, _mode):
    return [[0]]

  def get_active_task(self, _mode, _ids):
    return self.task

  def is_terminal_mode(self, _mode):
    return False

  def get_next_modes(self, _q, _mode):
    return ["next"]

  def is_collision_free(self, _q, _mode):
    self.collision_calls += 1
    return True


def _tiny_sampler(metric="euclidean"):
  from multi_robot_multi_goal_planning.problems.core.configuration import NpConfiguration
  env = _TinyEnv(metric)
  sampler = SimpleNamespace(
      env=env, conf_type=NpConfiguration, locally_informed_sampling=False,
      planning_approach="sampling_based",
      sample_mode=lambda reached_modes: reached_modes[0],
      can_transition_improve=lambda *args, **kwargs: True)
  # Deliberately non-straight path so an improving local set exists.
  vals = ([0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [-3.0, -3.0, -3.0], [1.0, 1.0, 1.0])
  path = [SimpleNamespace(q=NpConfiguration.from_list([np.array([v]) for v in row]),
                          mode="m") for row in vals]
  return sampler, path


def test_transition_control_uses_uniform_and_direct_avoids_it():
  from assembly_planner.mp_export import _ensure_mrbench
  _ensure_mrbench()
  M = _module()

  control, path = _tiny_sampler()
  q0 = M._generate_transitions(
      control, ["m"], 1, path, active_mode="m", try_direct_sampling=False, seed=3)
  assert q0 is not None and control.env.uniform_calls == 1
  assert control.last_transition_sampling_stats["accepted"] == 1

  direct, path = _tiny_sampler()
  q1 = M._generate_transitions(
      direct, ["m"], 1, path, active_mode="m", try_direct_sampling=True, seed=3)
  assert q1 is not None and direct.env.uniform_calls == 0
  assert np.array_equal(q1[0], np.array([0.5]))
  assert direct.last_transition_sampling_stats["accepted"] == 1
  assert direct.last_transition_sampling_stats["collision_checks"] == 1


def test_non_euclidean_direct_request_falls_back_to_uniform():
  from assembly_planner.mp_export import _ensure_mrbench
  _ensure_mrbench()
  M = _module()
  sampler, path = _tiny_sampler(metric="manhattan")
  q = M._generate_transitions(
      sampler, ["m"], 1, path, active_mode="m", try_direct_sampling=True, seed=3)
  assert q is not None
  assert sampler.env.uniform_calls == 1


def test_preincumbent_hold_and_linear_use_mode_entry_and_each_robots_goal():
  from assembly_planner.mp_export import _ensure_mrbench
  _ensure_mrbench()
  from multi_robot_multi_goal_planning.problems.core.configuration import NpConfiguration
  M = _module()

  env = _TinyEnv()
  env.tasks = [
      SimpleNamespace(robots=["active"], goal=_FixedGoal(0.5)),
      SimpleNamespace(robots=["free1"], goal=_FixedGoal(2.0)),
      SimpleNamespace(robots=["free2"], goal=_FixedGoal(-4.0)),
  ]
  entry = NpConfiguration.from_list(
      [np.array([0.1]), np.array([1.0]), np.array([-1.0])])
  mode = SimpleNamespace(entry_configuration=entry, task_ids=[0, 1, 2])
  planner = SimpleNamespace(env=env)

  held, pinned, inactive, progress = M.initial_completion(
      planner, mode, env.tasks[0], np.array([0.5]), "hold")
  assert inactive == [1, 2] and progress == {}
  assert np.array_equal(held[0], np.array([0.5]))
  assert np.array_equal(held[1], entry[1])
  assert np.array_equal(held[2], entry[2])

  np.random.seed(19)
  linear, _, _, progress = M.initial_completion(
      planner, mode, env.tasks[0], np.array([0.5]), "linear")
  assert set(progress) == {1, 2}
  assert 0.0 <= progress[1] <= 1.0 and 0.0 <= progress[2] <= 1.0
  assert np.allclose(linear[1], (1.0 - progress[1]) * entry[1]
                     + progress[1] * np.array([2.0]))
  assert np.allclose(linear[2], (1.0 - progress[2]) * entry[2]
                     + progress[2] * np.array([-4.0]))
  assert pinned == {0: np.array([0.5])}

  # Without an incumbent, the PHS arm intentionally uses the same parameter-free centre-line
  # proposal.  The post-incumbent sampler is what expands this line into a PHS.
  np.random.seed(19)
  phs_initial, _, _, phs_progress = M.initial_completion(
      planner, mode, env.tasks[0], np.array([0.5]), "phs")
  assert np.array_equal(phs_initial.state(), linear.state())
  assert phs_progress == progress


def test_lazy_source_uses_logical_forward_subtree_across_birrt_swaps():
  M = _module()
  class HashableMode:
    entry_configuration = "entry"

  forward_a = SimpleNamespace(state=SimpleNamespace(q="forward-a"))
  reverse_b = SimpleNamespace(state=SimpleNamespace(q="reverse-b"))
  tree = SimpleNamespace(order=1, subtree={1: forward_a}, subtree_b={2: reverse_b})
  mode = HashableMode()
  planner = SimpleNamespace(trees={mode: tree})
  assert M._logical_forward_reference(planner, mode) == "forward-a"

  # Physical dictionaries swap, while order records which one is logically forward.
  tree.order = -1
  tree.subtree = {2: reverse_b}
  tree.subtree_b = {1: forward_a}
  assert M._logical_forward_reference(planner, mode) == "forward-a"


def test_persistent_factored_sampler_keeps_untouched_baseline_branch():
  from assembly_planner.mp_export import _ensure_mrbench
  _ensure_mrbench()
  from multi_robot_multi_goal_planning.problems.core.configuration import NpConfiguration
  M = _factored_module()

  class Baseline:
    def __init__(self, q):
      self.q = q
      self.calls = 0
    def sample(self, _mode, _pinned):
      self.calls += 1
      return self.q.from_flat(self.q.state().copy())

  q = NpConfiguration.from_list([np.array([0.2]), np.array([-0.4])])
  env = SimpleNamespace(robots=["a", "b"])
  baseline = Baseline(q)
  sampler = M.PersistentFactoredSampler(
      env, baseline, lambda _mode: q, full_support_probability=1.0)
  result = sampler.sample("mode")
  assert baseline.calls == 1
  assert np.array_equal(result.state(), q.state())
  assert sampler.diagnostics()["pool_sizes"] == {"a": 1, "b": 1}


def test_persistent_factored_sampler_repairs_only_a_colliding_robot():
  from assembly_planner.mp_export import _ensure_mrbench
  _ensure_mrbench()
  from multi_robot_multi_goal_planning.problems.core.configuration import NpConfiguration
  from unittest.mock import patch
  M = _factored_module()

  class Env:
    robots = ["a", "b"]
    robot_idx = {"a": np.array([0]), "b": np.array([1])}
    limits = np.array([[-2.0, -2.0], [2.0, 2.0]])
    def is_collision_free(self, q, _mode):
      return not np.isclose(q[0][0], q[1][0])
    def is_collision_free_for_robot(self, robots, q, _mode):
      if len(robots) == 1:
        return True
      return not np.isclose(q[0], q[1])

  q0 = NpConfiguration.from_list([np.array([0.0]), np.array([0.0])])
  baseline = SimpleNamespace(sample=lambda *_args: None)
  sampler = M.PersistentFactoredSampler(
      Env(), baseline, lambda _mode: q0,
      full_support_probability=0.1, max_repairs=2)
  sampler._fresh_local = lambda robot, _q, _mode: np.array([0.0])
  proposed_b = iter((np.array([0.0]), np.array([1.0])))
  sampler._pooled_local = lambda robot, _q, _mode: next(proposed_b)
  with patch.object(M.random, "choice", return_value="b"):
    result = sampler._factored("mode", {})
  assert result is not None
  assert np.array_equal(result[0], np.array([0.0]))
  assert np.array_equal(result[1], np.array([1.0]))
  assert sampler.stats.repairs == 1
  assert sampler.stats.pair_checks == 1
