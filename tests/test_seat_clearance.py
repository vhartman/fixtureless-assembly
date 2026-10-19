import numpy as np

from execution.seat_command import SeatCommandGovernor
from execution.seat_clearance import project_box_halfspace


def test_clearance_projection_respects_joint_box():
    dq, active, infeasible = project_box_halfspace(
        np.array([-0.02, 0.0]), np.array([1.0, 1.0]), 0.005, 0.02)
    assert active and not infeasible
    assert np.all(np.abs(dq) <= 0.02 + 1e-10)
    assert float(sum(dq)) >= 0.005 - 1e-7
    assert np.allclose(dq, [-0.0075, 0.0125], atol=1e-6)


def test_infeasible_clearance_chooses_maximal_gap_motion():
    dq, active, infeasible = project_box_halfspace(
        np.array([-0.02]), np.array([1.0]), 0.03, 0.02)
    assert active and infeasible
    assert np.allclose(dq, [0.02])


def test_seat_command_governor_bounds_acceleration_and_tracking():
    governor = SeatCommandGovernor(0.01, rate=2.0, acceleration=1.0,
                                   tracking=0.03)
    first = governor.step(np.array([0.1]), np.array([0.0]), np.array([0.0]))
    assert np.allclose(first, [0.0001])
    second = governor.step(np.array([0.1]), first, np.array([0.0]))
    assert np.allclose(second, [0.0002])
    # A lagging physical joint cannot induce an unbounded command setpoint.
    reference = first + second
    for _ in range(100):
        reference += governor.step(np.array([0.1]), reference,
                                   np.array([0.0]))
    assert float(governor.previous_velocity[0]) <= 2.0 + 1e-8
    assert abs(reference[0]) <= 0.031
