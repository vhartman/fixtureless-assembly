# UR10 vacuum with a RESTRICTED workspace -- see ur10_two_finger_limited.g. Only the two
# joint limits are overridden on top of the standard ur10_vacuum; no file duplication.
Include: <ur10_vacuum.g>

Edit ur_shoulder_lift_joint: {limits: [-3.28319, -0.38319, 2.16, -1, 330]}
Edit ur_elbow_joint: {limits: [-0.68319, 2.68319, 3.15, -1, 150]}
