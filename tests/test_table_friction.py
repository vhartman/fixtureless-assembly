"""--table-friction: the friction-cone table model vs the frictionless default."""
import numpy as np

import problem as P
import wrench


def _table_rest_state():
    asm = P.make_assembly("peg-stool-cuboid", 2)
    seat_z = 0.20 + 0.015 + 0.018 / 2.0
    leg_z = seat_z - 0.018 / 2.0 - 0.0005 - 0.11 / 2.0
    poses = {
        "obj_0": np.array([0.0, 0.0, seat_z, 1.0, 0.0, 0.0, 0.0]),
        "obj_1": np.array([0.055, 0.035, leg_z, 1.0, 0.0, 0.0, 0.0]),
    }
    s = P.State(parent={"obj_0": ("table",), "obj_1": ("gripper", "a2")},
                grasp={"obj_1": 0}, place={"obj_0": ("slot", 0, P._IDENTITY_FACE)},
                supports={"obj_0": frozenset({("table",)}),
                          "obj_1": frozenset({("grasp", "a2", 0)})},
                poses=poses)
    return asm, s


KINDS = {"a1": "vacuum", "a2": "two_finger", "a3": "two_finger"}


def test_table_rest_survives_noise_only_with_friction():
    asm, s = _table_rest_state()
    wrench.PARAMS["insert_force"]["_default"] = 0.1
    target = frozenset({"obj_0"})
    d = asm.mate_axis("obj_1")

    wrench.PARAMS["table_model"] = "unilateral"
    wrench.PARAMS["insertion_uncertainty"] = {"lateral": 0.01, "offset": 0.0}
    assert not P._insertion_stable(s, asm, target, KINDS, "obj_1", "obj_0", d), \
        "the frictionless table must reject any lateral load"

    wrench.PARAMS["table_model"] = "friction"
    wrench.PARAMS["table_mu"] = 0.3
    assert P._insertion_stable(s, asm, target, KINDS, "obj_1", "obj_0", d), \
        "a friction table takes the tiny noise shear"

    # beyond the cone: lateral > mu x the available normal (~0.3 x 7.3 N = 2.2 N)
    wrench.PARAMS["insertion_uncertainty"] = {"lateral": 30.0, "offset": 0.0}
    assert not P._insertion_stable(s, asm, target, KINDS, "obj_1", "obj_0", d)

    # restore the defaults
    wrench.PARAMS["table_model"] = "unilateral"
    wrench.PARAMS["insertion_uncertainty"] = None


def test_friction_model_is_the_unilateral_limit():
    # with mu -> 0 the pyramid degenerates to 4 copies of the pure normal generator per point
    pts = [np.array([0.1, 0.0, 0.0]), np.array([-0.1, 0.0, 0.0])]
    G0, _ = wrench.cap_table_unilateral(pts)
    Gf, _ = wrench.cap_table_friction(pts, 0.0)
    assert Gf.shape[1] == 4 * G0.shape[1]
    for j in range(G0.shape[1]):
        dup = np.isclose(Gf, G0[:, [j]]).all(axis=0).sum()
        assert dup == 4, f"normal generator column {j} must appear 4x in the mu=0 pyramid"
