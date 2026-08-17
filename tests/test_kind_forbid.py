"""--vacuum-no-assemble: the successor-level kind-role forbid."""
import problem as P


class _FakeState:
    def __init__(self, supports):
        self.supports = supports

    def subassemblies(self):
        return [frozenset({"obj_0"}), frozenset({"obj_1"})]


KINDS = {"a1": "vacuum", "a2": "two_finger", "a3": "two_finger"}


def test_kind_forbid_actor_holder_brace():
    P.set_kind_forbid(("vacuum:assemble",))
    s = _FakeState({"obj_0": {("grasp", "a1", 3)}})
    # acting arm is a vacuum
    assert P._kind_forbidden(("assemble", "a1", "obj_1", "obj_0"), s, KINDS)
    # two-finger actor, but the TARGET group is held by a vacuum
    assert P._kind_forbidden(("assemble", "a2", "obj_1", "obj_0"), s, KINDS)
    # the same forbid sees a vacuum BRACE on the inserted part too
    s2 = _FakeState({"obj_1": {("grasp", "a1", 5)}})
    assert P._kind_forbidden(("assemble", "a2", "obj_1", "obj_0"), s2, KINDS)
    # no vacuum anywhere -> legal
    s3 = _FakeState({"obj_0": {("grasp", "a2", 0)}, "obj_1": {("grasp", "a2", 1)}})
    assert not P._kind_forbidden(("assemble", "a2", "obj_1", "obj_0"), s3, KINDS)


def test_kind_forbid_needs_kinds_and_scope():
    P.set_kind_forbid(("vacuum:assemble",))
    s = _FakeState({})
    # no kind information (wrench off) -> passes everything (the flag is a no-op there)
    assert not P._kind_forbidden(("assemble", "a1", "obj_1", "obj_0"), s, None)
    # uniform scalar kind: every arm is the forbidden kind -> the acting arm triggers
    assert P._kind_forbidden(("assemble", "a2", "obj_1", "obj_0"), s, "vacuum")
    # a pick is never filtered by an assemble-scoped forbid
    assert not P._kind_forbidden(("pick", "a1", "obj_0", 1), s, KINDS)
    # cleared -> everything passes
    P.set_kind_forbid(())
    assert not P._kind_forbidden(("assemble", "a1", "obj_1", "obj_0"), s, KINDS)
