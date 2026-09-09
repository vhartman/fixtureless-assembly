"""Regression tests for live rigid-component contact bookkeeping."""

from types import SimpleNamespace

import numpy as np

import sim as S


def _scene(active):
  scene = S.SimScene.__new__(S.SimScene)
  # obj_0--obj_1--obj_2 is a potential chain; obj_3 is a separate branch.
  scene._mate_eqs = {
      "obj_1": (0, "obj_0"),
      "obj_2": (1, "obj_1"),
      "obj_3": (2, "obj_0"),
  }
  scene.data = SimpleNamespace(eq_active=np.asarray(active, dtype=np.int32))
  return scene


def test_active_mate_component_is_transitive_and_live():
  scene = _scene([1, 1, 0])

  assert scene._active_mate_component("obj_0") == {"obj_0", "obj_1", "obj_2"}
  assert scene._active_mate_component("obj_2") == {"obj_0", "obj_1", "obj_2"}
  assert scene._active_mate_component("obj_3") == {"obj_3"}

  # A broken/deactivated intermediate mate immediately splits the live component.
  scene.data.eq_active[0] = 0
  assert scene._active_mate_component("obj_0") == {"obj_0"}
  assert scene._active_mate_component("obj_2") == {"obj_1", "obj_2"}
