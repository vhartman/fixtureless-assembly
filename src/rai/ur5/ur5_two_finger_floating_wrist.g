## Cheap-collision EE for the REAL UR5 arm (two-finger): the generic floating robotiq gripper
## (shared with the ur10 -- the gripper hardware is identical) PLUS the UR5's wrist-3 collision
## volume, rigidly attached behind the tool centre. The ur10 wrist file's capsule is calibrated
## to the ur10 arm and is wrong here; measured on src/rai/ur5/ur5.g, the ur5 wrist-3 capsule sits
## 0.24 m behind ur_gripper_center along the tool axis. Anchored to gripper_center (not ur_ee_link)
## so it is independent of the gripper-mount convention. Sound lower bound: this volume is a subset
## of the real arm, so a collision here implies a real collision.
Include: <../ur10/ur10_two_finger_floating.g>

ur_wrist_coll (ur_gripper_center) { shape:capsule, color:[1., 1., 1., .5], size:[.07 .045],
    contact:-1, Q:"t(0 0 -.26)" }


ur_ee_marker_back (ur_gripper_center){
    shape: marker,
    size:[0.05]
    Q:"t(0 0 -.34)",
}
