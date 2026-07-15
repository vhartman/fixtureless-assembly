## Cheap-collision EE for the REAL UR5 arm (vacuum): the ur10 floating vacuum gripper + the UR5
## wrist-3 volume. The ur10 vacuum's tool axis is +x, so "behind the tip" is -x; the ur5 wrist-3
## measures 0.17 m behind ur_vacuum. Capsule axis rotated to lie along the tool axis.
Include: <../ur10/ur10_vacuum_floating.g>

ur_wrist_coll (ur_vacuum) { shape:capsule, color:[1., 1., 1., .5], size:[.1 .05],
    contact:-1, Q:"t(-.17 0 0) d(90 0 1 0)" }
