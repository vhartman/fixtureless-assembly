# UR5 vacuum with a RESTRICTED workspace -- see ur5_two_finger_limited.g. Only the joint
# limits below are overridden on top of ur5_vacuum.g; no file duplication.
Include: <ur5_vacuum.g>

Edit ur_shoulder_lift_joint: {limits: [-1.58319, 1.78319, 2.16, -1, 330]}
Edit ur_elbow_joint: {limits: [-0.68319, 2.68319, 3.15, -1, 150]}
Edit ur_wrist_1_joint: {limits: [-2.28319, 1.28319, 3.2, -1, 54]}
Edit ur_wrist_2_joint: {limits: [-3.28319, 0.28319, 3.2, -1, 54]}
