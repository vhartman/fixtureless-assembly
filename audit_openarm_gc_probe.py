"""Probe: gripper_center orientation vs the analytic ssik calibration (openarm).

Measures the REAL pose of gripper_center relative to the flange in each rai model,
compares it with analytic_ik._CAL B_TOOL, and checks whether ssik seeds still land
the gripper_center frame on target (position AND orientation).
"""
import numpy as np
import robotic as ry

import analytic_ik


def _quat_to_R(q):
    w, x, y, z = q[3:] if len(q) == 7 else q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _pose(pose):
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = _quat_to_R(pose)
    return T


def _rel(ee_pose, gc_pose):
    return np.linalg.inv(_pose(ee_pose)) @ _pose(gc_pose)


def measure_gc(file, side):
    """Measure gripper_center pose relative to the flange, as env.py builds the frames."""
    C = ry.Config()
    C.addFile(file, namePrefix="a1_")
    flange = C.getFrame(f"a1_openarm_{side}_ee_base_link")
    C.addFrame("a1_ur_ee_link").setParent(flange)
    C.addFrame("a1_ur_gripper_center").setParent(C.getFrame("a1_gripper_center"))
    return _rel(np.asarray(C.getFrame("a1_ur_ee_link").getPose(), dtype=float),
                np.asarray(C.getFrame("a1_ur_gripper_center").getPose(), dtype=float))


def measure_pinch(file):
    C = ry.Config()
    C.addFile(file, namePrefix="ee_")
    return _rel(np.asarray(C.getFrame("ee_ur_ee_link").getPose(), dtype=float),
                np.asarray(C.getFrame("ee_ur_gripper_center").getPose(), dtype=float))


def seed_check(arm, B, n=8):
    """rai FK of the real arm's gripper_center vs ssik FK composed with B."""
    cal = analytic_ik._CAL[arm]
    U = analytic_ik._solver(arm)
    side = "right" if arm.endswith("right") else "left"
    C = ry.Config()
    C.addFile(f"src/rai/openarm/{side}_openarm.g", namePrefix="a1_")
    flange = C.getFrame(f"a1_openarm_{side}_ee_base_link")
    C.addFrame("a1_ur_ee_link").setParent(flange)
    C.addFrame("a1_ur_gripper_center").setParent(C.getFrame("a1_gripper_center"))
    names = [n for n in C.getJointNames()
             if n.startswith(f"a1_openarm_{side}_joint")]
    rng = np.random.default_rng(7)
    worst = (0.0, 0.0)
    for _ in range(n):
        q = -1.2 + 2.4 * rng.random(7)
        C.setJointState(q, names)
        T_rai = np.asarray(C.getFrame("a1_ur_gripper_center").getPose(), dtype=float)
        T_rai = _pose(T_rai)
        q_ssik = cal["sign"] * q + cal["offset"]
        T_ssik = np.asarray(U.fk(q_ssik), dtype=float) @ B
        dp = np.linalg.norm(T_ssik[:3, 3] - T_rai[:3, 3])
        da = np.linalg.norm(T_ssik[:3, :3] - T_rai[:3, :3])
        worst = (max(worst[0], dp), max(worst[1], da))
    return worst


def main():
    for arm, side in (("openarm_right", "right"), ("openarm_left", "left")):
        B_true = measure_gc(f"src/rai/openarm/{side}_openarm.g", side)
        B_cal = analytic_ik._CAL[arm]["tools"]["two_finger"]
        print(f"\n== {arm} ==")
        print("true flange->gripper_center:")
        print(np.round(B_true, 6))
        print("analytic_ik B_TOOL:")
        print(np.round(B_cal, 6))
        print("max |B_true - B_cal|:", float(np.abs(B_true - B_cal).max()))
        if analytic_ik.available(arm, "two_finger"):
            dp, da = seed_check(arm, B_true)
            print(f"seed FK error with TRUE B:   dp={dp:.2e} da={da:.2e}")
            dp, da = seed_check(arm, B_cal)
            print(f"seed FK error with CAL B:    dp={dp:.2e} da={da:.2e}")
        else:
            print("ssik unavailable -> skipped FK check")
    print("\npinch proxy flange->gripper_center:")
    print(np.round(measure_pinch("src/rai/openarm/openarm_pinch_floating.g"), 6))


if __name__ == "__main__":
    main()
