"""Focused scheduling invariants for the assembly-side lazy shortcut portfolio."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).parents[1] / "experiments" / "shortcut-sampling" / "ideation_samplers.py"
    spec = importlib.util.spec_from_file_location("_portfolio_shortcut_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Mode:
    def __init__(self):
        self.task_ids = [0, 0]


class _Point:
    def __init__(self, value):
        self.value = float(value)
        self.mode = _Mode()


class _Env:
    robots = ["a", "b"]
    batch_config_cost = object()


class _FixedRng:
    def beta(self, *_args):
        return 0.5

    def random(self):
        return 0.0


class _Q:
    def __init__(self, value):
        self._value = np.atleast_1d(float(value))

    def state(self):
        return self._value


class _Waypoint:
    def __init__(self, value, mode=0):
        self.q = _Q(value)
        self.mode = mode


def test_experiment_preprocessor_preserves_every_certified_input_edge():
    mod = _load_module()

    backtrack = [_Waypoint(0), _Waypoint(2), _Waypoint(1)]
    assert mod.preserve_input_nodes(backtrack) is backtrack

    mode_switch = [_Waypoint(0, 0), _Waypoint(0.5, 1), _Waypoint(1, 0)]
    assert mod.preserve_input_nodes(mode_switch) is mode_switch

    redundant = [_Waypoint(0), _Waypoint(0.5), _Waypoint(1)]
    assert mod.preserve_input_nodes(redundant) is redundant


def test_old_random_draw_stream_is_whole_path_round_robin():
    mod = _load_module()
    env = _Env()
    path = [_Point(0) for _ in range(20)]
    np.random.seed(31)
    sampler = mod.WholePathRandomSampler(env, path)
    proposals = [sampler.propose() for _ in range(8)]

    np.random.seed(31)
    expected = []
    rr = 0
    for _ in range(8):
        i, j = int(np.random.randint(0, 20)), int(np.random.randint(0, 20))
        if i > j:
            i, j = j, i
        if j - i < 2:
            expected.append(None)
        else:
            expected.append((i, j, [rr % 2]))
            rr += 1
    assert proposals == expected


def test_portfolio_keeps_and_reuses_unchosen_pending_candidate(monkeypatch):
    mod = _load_module()

    class FakeSource:
        def __init__(self, _env, _path, *, robot, **_kwargs):
            self.robot = robot
            self.feedbacks = []
            self.refreshes = []

        def propose(self):
            return 0, 2, [self.robot]

        def feedback(self, outcome, **_kwargs):
            self.feedbacks.append(outcome)

        def observe_path_change(self, ij):
            self.refreshes.append(ij)

        def report(self):
            return {}

    class FakeTree(FakeSource):
        def __init__(self, env, path, **kwargs):
            super().__init__(env, path, robot=0, **kwargs)

    class FakeCurrent(FakeSource):
        def __init__(self, env, path, **kwargs):
            super().__init__(env, path, robot=1, **kwargs)

    def path_cost(points, _batch):
        return sum(point.value for point in points)

    constructions = {0: 0, 1: 0}

    def candidate(_env, _path, _i, _j, robots, _State):
        robot = robots[0]
        constructions[robot] += 1
        # Old segment costs 10. Tree gain is 4 with work 10; current gain is 3 with work 5,
        # so the fixed equal posterior selects current.
        middle = 6 if robot == 0 else 7
        return [_Point(0), _Point(middle), _Point(0)]

    monkeypatch.setattr(mod, "_mrb", lambda: (object, None, path_cost, None, None))
    monkeypatch.setattr(mod, "PersistentTreeSampler", FakeTree)
    monkeypatch.setattr(mod, "CurrentGainBetaSampler", FakeCurrent)
    monkeypatch.setattr(mod, "_proposal_candidate", candidate)
    monkeypatch.setattr(
        mod, "_collision_work_bound", lambda element, _resolution: 10 if element[1].value == 6 else 5
    )

    path = [_Point(0), _Point(10), _Point(0)]
    sampler = mod.LazyPortfolioSampler(_Env(), path, seed=3)
    sampler.rng = _FixedRng()

    assert sampler.propose() == (0, 2, [1])
    assert constructions == {0: 1, 1: 1}
    assert "tree" in sampler.pending
    sampler.observe_check(False, config_checks=2)
    sampler.feedback("collision", ij=(0, 2), robot=1)

    # Current is replaced. The still-pending tree candidate is re-used without reconstructing
    # it because a failed collision check did not mutate the path.
    sampler.propose()
    assert constructions[0] == 1


def test_external_tree_accept_refreshes_current_and_invalidates_cache(monkeypatch):
    mod = _load_module()

    class FakeSource:
        def __init__(self, _env, _path, *, robot, **_kwargs):
            self.robot = robot
            self.refreshes = []

        def propose(self):
            return 0, 2, [self.robot]

        def feedback(self, *_args, **_kwargs):
            pass

        def observe_path_change(self, ij):
            self.refreshes.append(ij)

        def report(self):
            return {}

    class FakeTree(FakeSource):
        def __init__(self, env, path, **kwargs):
            super().__init__(env, path, robot=0, **kwargs)

    class FakeCurrent(FakeSource):
        def __init__(self, env, path, **kwargs):
            super().__init__(env, path, robot=1, **kwargs)

    monkeypatch.setattr(mod, "_mrb", lambda: (object, None, lambda p, _b: sum(x.value for x in p), None, None))
    monkeypatch.setattr(mod, "PersistentTreeSampler", FakeTree)
    monkeypatch.setattr(mod, "CurrentGainBetaSampler", FakeCurrent)
    monkeypatch.setattr(
        mod, "_proposal_candidate",
        lambda _e, _p, _i, _j, robots, _s: [_Point(0), _Point(1 if robots[0] == 0 else 9), _Point(0)],
    )
    monkeypatch.setattr(mod, "_collision_work_bound", lambda *_args: 1)

    sampler = mod.LazyPortfolioSampler(_Env(), [_Point(0), _Point(10), _Point(0)])
    sampler.rng = _FixedRng()
    assert sampler.propose() == (0, 2, [0])
    sampler.observe_check(True, config_checks=1)
    sampler.feedback("accepted", ij=(0, 2), robot=0)

    assert sampler.sources["current"].refreshes == [(0, 2)]
    assert sampler.path_version == 1
    assert sampler.pending_prepared == {}


def test_containment_policy_orders_container_before_descendant():
    mod = _load_module()
    sampler = object.__new__(mod.ContainmentPortfolioSampler)
    sampler.stat = {
        "portfolio_containment_choices": 0,
        "portfolio_containment_source": {"tree": 0, "current": 0},
    }
    tree = {"source": "tree", "proposal": (2, 12, [0])}
    current = {"source": "current", "proposal": (4, 8, [0])}

    assert sampler._priority_choice([tree, current]) is tree
    assert sampler.stat["portfolio_containment_choices"] == 1
    assert sampler.stat["portfolio_containment_source"] == {"tree": 1, "current": 0}

    crossing = {"source": "current", "proposal": (8, 15, [0])}
    assert sampler._priority_choice([tree, crossing]) is None
    other_robot = {"source": "current", "proposal": (4, 8, [1])}
    assert sampler._priority_choice([tree, other_robot]) is None
