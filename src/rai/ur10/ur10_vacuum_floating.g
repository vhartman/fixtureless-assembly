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
