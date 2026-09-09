Include: <ur10.g>

#Edit ur_world_joint_origin: { rel: [0, 0, 0] }

Edit ur_shoulder_pan_joint: { q: 0.0 }
Edit ur_shoulder_lift_joint: { q: -2.0 }
Edit ur_elbow_joint: { q: 1.0 }
Edit ur_wrist_1_joint: { q: -1.0 }
Edit ur_wrist_2_joint: { q: -1.571 }
Edit ur_wrist_3_joint: { q: 1.0 }

# Robotiq ePick CAD, display only.  Keep its pose separate from the collision/tool frames so
# calibration only requires editing this Q; the cylinder below remains the planning proxy.
ur_epick_visual (ur_ee_link) { shape:mesh, mesh:<../robotiq/meshes/visual/robotiq_epick.dae>, meshscale:0.001, Q:"t(0.095 0.0 0.) d(90 0 1 0)", visual:True, contact:0 }

gripper_fill (ur_ee_link){ shape:cylinder, color:[1, 1, 1 , 0.5], Q:"d(90 0 1 0) t(-.0 -.0 .06)",
	size:[.07 .04], 
	contact:-1
}

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
