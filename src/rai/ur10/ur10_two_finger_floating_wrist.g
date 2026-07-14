# Floating two-finger EE + the WRIST-3 collision volume, rigidly attached (joint removed).
#
# The real arm's last joint (wrist_3) rotates ABOUT the gripper axis, and its collision
# capsule (ur_coll8 in ur10.g) lies ALONG that axis, centered on it -- so under the removed
# rotation the capsule is invariant and attaching it rigidly is EXACT, not conservative.
# Placement derived from ur10_clean.g: ee_link = wrist_3_joint @ t(0,.0922,0) Rz(90deg),
# coll8 = capsule[.09 .05] at t(0,.02,0) along wrist_3 y  =>  in the EE frame: center
# (-0.0722, 0, 0), axis = EE x (the tool axis), i.e. the EE is ~9cm "longer" backwards.
#
# Use as a MORE ACCURATE CHEAP MODEL (still a sound lower bound of the full arm: this
# volume is a subset of the real arm, so a collision here implies a real collision).
# NOTE (future generalization): instead of freezing links, solve a reduced IK with a
# floating base + the last couple of JOINTS kept -- a middle layer between the floating
# EE and the full arm.

ur_base { multibody }

ur_floatX (ur_base){ joint:transX, limits:[-2 2], mass:.01 }
ur_floatY (ur_floatX){ joint:transY, limits:[-2 2], mass:.01 }
ur_floatZ (ur_floatY){ joint:transZ, limits:[0 3], mass:.01, q: 1 }
ur_floatRX (ur_floatZ){ joint:hingeX, limits:[-3.14159 3.14159], mass:.01 }
ur_floatRY (ur_floatRX){ joint:hingeY, limits:[-3.14159 3.14159], mass:.01 }
ur_ee_link (ur_floatRY){ joint:hingeZ, limits:[-3.14159 3.14159], mass:.01 }

ur_base_coll(ur_ee_link)   { shape:cylinder color:[1.,1.,1.,.5] size:[.04 .039] Q:"t(.04 -.0 .0) d(90 0 1 0)", contact:-1  }

# wrist_3 link collision, rigid (the removed joint's rotation axis == the capsule axis).
# radius UNDER-approximates the real Wrist3 mesh (.035 vs the arm's .05 capsule): a cheap
# REJECT layer must never be stricter than the soft real-arm IK (measured: the .05 capsule
# wrongly rejected 10mm table grazes the real arm accepts).
ur_wrist_coll(ur_ee_link)  { shape:capsule color:[1.,1.,1.,.5] size:[.09 .05] Q:"t(-.0722 -.0 .0) d(90 0 1 0)", contact:-1 }

Prefix: "ur_"
Include: <../robotiq/robotiq.g>
Edit ur_robotiq_base (ur_ee_link) { Q:"d(90 0 1 0) t(-.0 -.0 .036)" }

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
