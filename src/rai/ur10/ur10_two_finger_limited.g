# UR10 two-finger with a RESTRICTED workspace (tighter shoulder-lift upper + elbow lower
# bound). Everything is the standard ur10_two_finger; only the two joint limits below are
# overridden (a later Edit wins, same pattern ur10.g uses over ur10_clean.g), so there is
# NO file duplication -- change the base and this follows.
Include: <ur10_two_finger.g>

Edit ur_shoulder_lift_joint: {limits: [-3.28319, -0.38319, 2.16, -1, 330]}
Edit ur_elbow_joint: {limits: [-0.68319, 2.68319, 3.15, -1, 150]}
Edit ur_wrist_1_joint: {limits: [-4.28319, 1.28319, 3.2, -1, 54]}
Edit ur_wrist_2_joint: {limits: [-3.28319, 0.28319, 3.2, -1, 54]}