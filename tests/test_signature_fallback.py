"""Regression tests for actionable signature-heuristic size failures."""

import pytest

import assembly_planner.solver as S


class _ModuleAssembly:
  connect_parts = {"obj_5"}


def test_chain_heuristic_rejects_detached_modules_before_building_search_context(monkeypatch):
  monkeypatch.setattr(S, "get_robots", lambda _C: ["a1"])
  monkeypatch.setattr(
      S, "_search_context",
      lambda *_args, **_kwargs: pytest.fail("module rejection must happen before context build"),
  )
  with pytest.raises(RuntimeError, match=r"Re-run with --milestones"):
    S.solve_assembly_problem_optimistic(None, _ModuleAssembly(), chain_h=True)


def test_signature_state_guard_is_actionable_and_does_not_silently_fallback(monkeypatch):
  calls = []

  monkeypatch.setattr(
      S, "compute_grasp_signatures",
      lambda *_args, **_kwargs: {"obj_0": {0: 0}, "obj_1": {0: 0}},
  )

  def fake_parent_init(self, _assembly, _n_robots, _grasp_table, **kwargs):
    calls.append(kwargs["labeled_loose"])
    raise RuntimeError("collapsed labeled abstraction exceeded 2000000 states")

  monkeypatch.setattr(S.CollapsedLabeledHeuristic, "__init__", fake_parent_init)

  with pytest.raises(RuntimeError, match="Re-run with --optimistic"):
    S.SignatureCollapsedHeuristic(None, 4, None, verbose=True)
  assert calls == [True]


def test_signature_heuristic_does_not_hide_unrelated_runtime_errors(monkeypatch):
  monkeypatch.setattr(
      S, "compute_grasp_signatures", lambda *_args, **_kwargs: {"obj_0": {0: 0}},
  )

  def fake_parent_init(self, _assembly, _n_robots, _grasp_table, **_kwargs):
    raise RuntimeError("bad wrench fact")

  monkeypatch.setattr(S.CollapsedLabeledHeuristic, "__init__", fake_parent_init)

  with pytest.raises(RuntimeError, match="bad wrench fact"):
    S.SignatureCollapsedHeuristic(None, 4, None)
