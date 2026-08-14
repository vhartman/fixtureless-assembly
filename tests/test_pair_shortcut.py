"""Unit invariants for the local pair-then-gain shortcut sampler.

The implementation lives in the experiment directory (which is intentionally not a Python
package); load it directly so this test does not import or modify mr_bench.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).parents[1] / "experiments" / "shortcut-sampling" / "ideation_samplers.py"
    spec = importlib.util.spec_from_file_location("_pair_shortcut_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Mode:
    def __init__(self, task_ids):
        self.task_ids = list(task_ids)


class _State:
    def __init__(self, task_ids):
        self.mode = _Mode(task_ids)


class _Env:
    robots = ["a", "b", "c"]


def _constant_runs(path, robot, min_span=2):
    out, start = [], 0
    for k in range(1, len(path) + 1):
        if k == len(path) or path[k].mode.task_ids[robot] != path[start].mode.task_ids[robot]:
            if k - 1 - start >= min_span:
                out.append((start, k - 1))
            start = k
    return out


def test_pair_sampler_only_proposes_jointly_task_constant_intervals(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "_mrb", lambda: (None, None, None, None, _constant_runs))
    path = [_State(ids) for ids in (
        (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 1, 0), (0, 1, 0),
        (1, 1, 0), (1, 1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 1),
        (1, 1, 1),
    )]
    np.random.seed(7)
    sampler = mod.PairSampler(_Env(), path)

    assert sampler.arms
    for _ in range(100):
        proposal = sampler.propose()
        if proposal is None:
            continue
        i, j, robots = proposal
        assert j - i >= 2
        assert len(robots) == 2
        for r in robots:
            assert path[i].mode.task_ids[r] == path[j].mode.task_ids[r]


def test_pair_then_gain_waits_for_validated_pair_budget(monkeypatch):
    mod = _load_module()

    class FakePair:
        dead = False
        def __init__(self, *_args, **_kwargs):
            self.feedbacks = []
        def propose(self):
            return 0, 2, [0, 1]
        def feedback(self, outcome, **_kwargs):
            self.feedbacks.append(outcome)
        def report(self):
            return {"pair_arms": 1}

    class FakeGain:
        def __init__(self, *_args, **_kwargs):
            self.feedbacks = []
        def propose(self):
            return 1, 3, [2]
        def feedback(self, outcome, **_kwargs):
            self.feedbacks.append(outcome)
        def report(self):
            return {"recomputes": 0}

    monkeypatch.setattr(mod, "PairSampler", FakePair)
    monkeypatch.setattr(mod, "GainPropSampler", FakeGain)
    sampler = mod.PairThenGainSampler(_Env(), [], pair_checks=1)

    assert sampler.propose() == (0, 2, [0, 1])
    sampler.feedback("cost")
    assert sampler.pair_checks == 0
    sampler.feedback("accepted", ij=(0, 2))
    assert sampler.pair_checks == 1
    assert sampler.propose() == (1, 3, [2])
    assert sampler.gain.feedbacks == ["cost", "accepted"]
