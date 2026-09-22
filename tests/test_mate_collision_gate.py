import pytest

import solver as S


class _CollisionScene:
  def __init__(self, rows):
    self.rows = list(rows)

  def getCollisions(self):
    return list(self.rows)


def test_strict_mate_gate_exempts_only_the_authored_interface(monkeypatch):
  """A named mate must not hide contact with neighbours in its target component."""
  contact = []
  monkeypatch.setattr(S, "set_part_contact",
                      lambda _C, part, enabled: contact.append((part, enabled)))
  rik = object.__new__(S.RobotIK)
  rik.keep_contacts = False
  C = _CollisionScene([
      ("obj_new__collision", "obj_root__collision", -0.001),  # intended interface
      ("obj_new__collision", "obj_neighbour__collision", -0.002),
      ("module_aux__collision", "obj_root__collision", -0.003),
      ("arm_collision", "obj_neighbour__collision", -0.004),
      ("obj_new__collision", "module_aux__collision", -0.005),  # internal mover
  ])

  pen = rik.mate_penetration(
      C, frozenset(("obj_new", "module_aux")), frozenset(("obj_root",)),
      frozenset(("obj_new",)))

  assert pen == pytest.approx(0.009)
  assert sorted(contact[:2]) == [("module_aux", 1), ("obj_new", 1)]
  assert sorted(contact[2:]) == [("module_aux", 0), ("obj_new", 0)]
