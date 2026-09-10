## UR5 real arm with a VACUUM end effector (no robotiq). Mirrors ur5.g's arm + collisions and
## ur10_vacuum.g's vacuum pen. The tool centre is `ur_vacuum`; the assembly code keys the tool
## root on `ur_ee_link` and derives the approach axis from (ur_vacuum - ur_ee_link).

ur_base: {multibody: true}
ur_arm_ur5 (ur_base): {}          # marker so the search picks the ur5-calibrated cheap wrist model

Prefix: "ur_"
Include: <ur5e_clean.g>
Prefix: false

Edit ur_base_link(ur_base): { Q:"d(180 0 0 1)" }   # face the workspace (ur5 base convention is 180deg off ur10)

coll1(ur_shoulder_pan_joint_origin)  { shape:capsule color:[1.,1.,1.,.5] size:[.18 .08] Q:"t(0 0 0) d(90 90 1 0)", contact:-2 }
coll1_1(ur_shoulder_lift_joint)      { shape:capsule color:[1.,1.,1.,.5] size:[.16 .065] Q:"t(0 .0 0.01) d(0 0 1 0)", contact:-2 }
coll2(ur_shoulder_lift_joint)        { shape:capsule color:[1.,1.,1.,.5] size:[.35 .065] Q:"t(0 .18 .064) d(90 90 1 0)", contact:-2 }
coll3(ur_elbow_joint_origin)         { shape:capsule color:[1.,1.,1.,.5] size:[.16 .065] Q:"t(0 .0 0.01) d(0 0 1 0)", contact:-2 }
coll4(ur_elbow_joint)                { shape:capsule color:[1.,1.,1.,.5] size:[.37 .06] Q:"t(0 .2 -.07) d(90 90 1 0)", contact:-2 }
coll5(ur_wrist_1_joint)              { shape:capsule color:[1.,1.,1.,.5] size:[.1 .047] Q:"t(0 0 -.02) d(0 90 1 0)", contact:-2 }
coll6(ur_wrist_2_joint)              { shape:capsule color:[1.,1.,1.,.5] size:[.12 .05] Q:"t(0 -.01 0) d(90 90 1 0)", contact:-2 }
coll7(ur_wrist_3_joint)              { shape:capsule color:[1.,1.,1.,.5] size:[.1 .05] Q:"t(0 0 -.02) d(0 0 1 0)", contact:-2 }

# gripper-mount flange (tool root), like ur5.g; its +z is the tool/approach axis (wrist-3 +z).
ur_ee_link (ur_wrist_3_joint) { Q:"d(90 0 0 1) t(-.0 -.0 .09)" }

# Robotiq ePick CAD, display only.  Keep its pose separate from the collision/tool frames so
# calibration only requires editing this Q; the cylinder below remains the planning proxy.
ur_epick_visual (ur_ee_link) { shape:mesh, mesh:<../robotiq/meshes/visual/robotiq_epick.dae>, meshscale:0.001, Q:"t(0 0 0.055)", visual:True, contact:0 }

# vacuum pen: a fill cylinder along the tool axis (+z) and the suction tip `ur_vacuum` at the
# grasp standoff. Approach axis = (ur_vacuum - ur_ee_link) resolves to +z.
gripper_fill (ur_ee_link){ shape:cylinder, color:[1, 1, 1, .5], Q:"t(0 0 .025)", size:[.07 .04], contact:-1 }
ur_vacuum (ur_ee_link){ shape:sphere, color:[.9, 0, 0, 1], Q:"t(0 0 .075)", size:[0.005], contact:0 }
ur_ee_marker (ur_ee_link){ shape:marker, size:[0.05] }

Edit ur_shoulder_pan_joint: { q: 0.0 }
Edit ur_shoulder_lift_joint: { q: -0.5 }
Edit ur_elbow_joint: { q: 1.0 }
Edit ur_wrist_1_joint: { q: 0.5 }
Edit ur_wrist_2_joint: { q: -1.571 }
Edit ur_wrist_3_joint: { q: 0.0 }
