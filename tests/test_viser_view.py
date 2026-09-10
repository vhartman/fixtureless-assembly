import numpy as np
import pytest

from viser_view import ViserRecorder


def _pose(x):
  return (np.array([x, 0.0, 0.0], dtype=np.float32),
          np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_alternative_overlay_keeps_selected_opaque_and_adds_same_step_ghosts():
  rec = ViserRecorder()
  tri = np.array([[0, 1, 2]], dtype=np.uint32)
  verts = np.zeros((3, 3), dtype=np.float32)
  rec.geometry = {
      "a1_link": (verts, tri, (10, 20, 30), None),
      "table": (verts, tri, (40, 50, 60), None),
  }
  rec.steps = [
      {"a1_link": _pose(0.0), "table": _pose(10.0)},
      {"a1_link": _pose(1.0), "table": _pose(10.0)},
  ]
  rec.labels = ["initial", "pick"]
  rec.step_vectors = [[], []]
  rec.step_interfaces = [[], []]

  rec.overlay_alternatives(
      {1: [{"a1_link": _pose(2.0), "table": _pose(10.0)},
           {"a1_link": _pose(3.0)}]}, count=3, opacity=0.2)

  assert rec.geometry["a1_link"][3] is None
  assert rec.geometry["keyframe_alt_1/a1_link"][3] == pytest.approx(0.2)
  assert "keyframe_alt_1/table" not in rec.geometry
  assert rec.steps[1]["keyframe_alt_1/a1_link"][0][0] == pytest.approx(2.0)
  assert rec.steps[1]["keyframe_alt_2/a1_link"][0][0] == pytest.approx(3.0)
  assert "selected candidate=0" in rec.labels[1]


def test_alternative_overlay_validates_arguments_and_cannot_be_applied_twice():
  rec = ViserRecorder()
  rec.steps = [{}, {}]
  rec.labels = ["a", "b"]
  with pytest.raises(ValueError, match=">= 2"):
    rec.overlay_alternatives({}, count=1)
  with pytest.raises(ValueError, match="strictly between"):
    rec.overlay_alternatives({}, count=2, opacity=1.0)
  rec.overlay_alternatives({}, count=2)
  with pytest.raises(ValueError, match="already"):
    rec.overlay_alternatives({}, count=2)


class _Frame:
  def __init__(self, name, info):
    self.name = name
    self._info = info

  def info(self):
    return self._info


def test_collision_display_filter_keeps_visuals_and_opaque_scene_primitives():
  rec = ViserRecorder(hide_collision_primitives=True)
  assert rec._is_display_collision_primitive(
      _Frame("a1_ur_coll4", {"contact": -2, "color": [1, 1, 1, 0.5]}))
  assert rec._is_display_collision_primitive(
      _Frame("a1_finger1", {"contact": -2, "color": [1, 1, 1, 0.2]}))
  assert not rec._is_display_collision_primitive(
      _Frame("a1_upper_arm_visual", {"visual": True, "contact": 0,
                                      "color": [0.7, 0.7, 0.7, 1]}))
  # Primitive-only assembly geometry remains visible even though it is collidable.
  assert not rec._is_display_collision_primitive(
      _Frame("obj_2", {"contact": 1, "color": [0.8, 0.2, 0.2, 1]}))
