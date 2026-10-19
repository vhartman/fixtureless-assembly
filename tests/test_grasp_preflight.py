"""Measured-grasp mate preflight checks actual tool collision geometry."""
import mujoco
import numpy as np

from execution.grasp_preflight import mate_gripper_clearance


class _Scene:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <body name="a1_ur_robotiq_base" pos="0.5 0 0">
              <body name="a1_ur_gripper_center"/>
              <body name="a1sim_grip_fixed___finger" pos="0.05 0 0">
                <geom type="box" size="0.02 0.02 0.02" group="3"/>
              </body>
            </body>
            <body name="obj_0">
              <geom type="box" size="0.05 0.05 0.05"/>
            </body>
            <body name="obj_1" pos="0.5 0 0">
              <geom type="box" size="0.02 0.02 0.02"/>
            </body>
          </worldbody>
        </mujoco>
        """)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.grasp_anchor = {"a1": "a1_ur_gripper_center"}
        self._part_owner_by_body = [None] * self.model.nbody
        for part in ("obj_0", "obj_1"):
            self._part_owner_by_body[self._body_id(part)] = part

    def _body_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _active_mate_component(self, parent):
        return [parent]

    def part_pose(self, name):
        body = self._body_id(name)
        return np.r_[self.data.xpos[body], self.data.xquat[body]]


def test_projection_includes_fixed_finger_sibling_and_does_not_move_plant():
    scene = _Scene()
    before_pos = scene.data.geom_xpos.copy()
    before_mat = scene.data.geom_xmat.copy()

    # Current achieved grasp has anchor and child both at x=0.5. Seating the
    # child at x=0 therefore carries the fixed finger proxy into the receiver.
    blocked = mate_gripper_clearance(
        scene, "a1", "obj_0", "obj_1", [0, 0, 0, 1, 0, 0, 0])
    assert blocked["clearance_m"] < -0.01
    assert "sim_grip_fixed" in blocked["pair"][0]
    assert blocked["pair"][1] == "obj_0"

    clear = mate_gripper_clearance(
        scene, "a1", "obj_0", "obj_1", [0.2, 0, 0, 1, 0, 0, 0])
    assert clear["clearance_m"] > 0.05
    np.testing.assert_array_equal(scene.data.geom_xpos, before_pos)
    np.testing.assert_array_equal(scene.data.geom_xmat, before_mat)


def test_projection_ignores_inconsistent_zero_mesh_distance(monkeypatch):
    scene = _Scene()

    def invalid_distance(model, data, hand, receiver, limit, witness):
        witness[:] = [0, 0, 0, 0.02, 0, 0]
        return 0.0

    monkeypatch.setattr(mujoco, "mj_geomDistance", invalid_distance)
    result = mate_gripper_clearance(
        scene, "a1", "obj_0", "obj_1", [0, 0, 0, 1, 0, 0, 0])
    assert result["clearance_m"] == 0.1
    assert result["pair"] is None
