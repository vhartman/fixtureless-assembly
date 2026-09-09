# Floating vacuum EE + the WRIST-3 collision volume, rigidly attached (joint removed).
# See ur10_two_finger_floating_wrist.g for the derivation: the wrist_3 capsule lies ALONG
# the (removed) rotation axis, centered on it, so the rigid attachment is exact. The vacuum
# is additionally roll-symmetric, so this stays exact for the 5-DOF vacuum IK as well.
# NOTE (future generalization): a reduced IK with floating base + the last couple of joints.

ur_base { multibody }

ur_floatX (ur_base){ joint:transX, limits:[-2 2], mass:.01 }
ur_floatY (ur_floatX){ joint:transY, limits:[-2 2], mass:.01 }
ur_floatZ (ur_floatY){ joint:transZ, limits:[0 3], mass:.01, q: 1 }
ur_floatRX (ur_floatZ){ joint:hingeX, limits:[-3.14159 3.14159], mass:.01 }
ur_floatRY (ur_floatRX){ joint:hingeY, limits:[-3.14159 3.14159], mass:.01 }
ur_ee_link (ur_floatRY){ joint:hingeZ, limits:[-3.14159 3.14159], mass:.01 }

# Robotiq ePick CAD, display only; calibrated to the same tool-root convention as
# ur10_vacuum.g.  The cylinder remains the collision proxy used by planning.
ur_epick_visual (ur_ee_link) { shape:mesh, mesh:<../robotiq/meshes/visual/robotiq_epick.dae>, meshscale:0.001, Q:"t(0.095 0.0 0.) d(90 0 1 0)", visual:True, contact:0 }

gripper_fill (ur_ee_link){ shape:cylinder, color:[1, 1, 1, 0.5], Q:"d(90 0 1 0) t(-.0 -.0 .06)",
	size:[.07 .04],
	contact:-1
}

# wrist_3 link collision, rigid (the removed joint's rotation axis == the capsule axis).
# radius UNDER-approximates the real Wrist3 mesh (.035 vs the arm's .05 capsule): a cheap
# REJECT layer must never be stricter than the soft real-arm IK (measured: the .05 capsule
# wrongly rejected 10mm table grazes the real arm accepts).
ur_wrist_coll (ur_ee_link){ shape:capsule color:[1.,1.,1.,.5] size:[.09 .05] Q:"t(-.0722 -.0 .0) d(90 0 1 0)", contact:-1 }

# pen
ur_vacuum (ur_ee_link){
    shape:sphere,
    color:[.9, 0, 0 ,1],
    Q:"t(.115 0.0 0.)",
    size:[0.005],
    contact:0
}

ur_ee_marker (ur_ee_link){
    shape: marker,
    size:[0.05]
}

#ur_ee_marker_back (ur_ee_link){
#    shape: marker,
#    size:[0.05]
#    Q:"t(-.165 0.0 0.)",
#}
#
#ur_ee_marker_back (ur_ee_link){
#    shape: marker,
#    size:[0.05]
#    Q:"t(0.02 0.0 0.)",
#}
