# Floating collision proxy for the OpenArm v2 pinch hand.
#
# This has the same six free coordinates and frame API as the UR floating EEs, but its
# gripper_center follows the repository-wide convention (its +z is the grasp approach); the
# 180deg tool rotation maps that convention onto OpenArm's physical flange -z.  Its collision
# bodies mirror the real OpenArm .g files.
# Grasp precomputation and the cheap scene gate must use this model: a Robotiq proxy has a
# different palm length/finger sweep and admitted side grasps whose real OpenArm fingers pass
# through the table.  The single-# collision candidates in left/right_openarm.g correspond to
# these bodies.  The double-## arm capsules are deliberately not represented.

ur_base { multibody }

ur_floatX (ur_base)  { joint: transX, limits: [-2 2], mass: .01 }
ur_floatY (ur_floatX) { joint: transY, limits: [-2 2], mass: .01 }
ur_floatZ (ur_floatY) { joint: transZ, limits: [0 3], mass: .01, q: 1 }
ur_floatRX (ur_floatZ) { joint: hingeX, limits: [-3.14159 3.14159], mass: .01 }
ur_floatRY (ur_floatRX) { joint: hingeY, limits: [-3.14159 3.14159], mass: .01 }
ur_ee_link (ur_floatRY) { joint: hingeZ, limits: [-3.14159 3.14159], mass: .01 }

# Palm/EE housing.  Keep these three capsules byte-for-byte geometrically aligned with the
# active ee_coll/ee_coll1/ee_coll2 candidates in left_openarm.g and right_openarm.g.  The
# floating proxy is the grasp sampler's collision model; stale geometry here creates a large
# set of candidates that only fail later, in the expensive real-arm bind.
openarm_ee_coll (ur_ee_link): {
  shape: capsule, size: [.02 .04], Q: "t(.005 0 -.03) d(0 1 0 0)",
  color: [1 1 1 .5], contact: -1
}
openarm_ee_coll1 (ur_ee_link): {
  shape: capsule, size: [.06 .03], Q: "t(-.005 0 -.075) d(90 1 0 0)",
  color: [1 1 1 .5], contact: -1
}
openarm_ee_coll2 (ur_ee_link): {
  shape: capsule, size: [.06 .03], Q: "t(.02 0 -.075) d(90 1 0 0)",
  color: [1 1 1 .5], contact: -1
}

# Open-pose finger sweep used by the real model.  The two parallel capsules approximate
# each finger's 55mm width; their parent offsets are folded into these flange-relative poses.
openarm_finger1_coll (ur_ee_link): {
  shape: capsule, size: [.06 .015], Q: "t(-.01043 .058 -.1205)",
  color: [1 1 1 .5], contact: -1
}
openarm_finger1_coll2 (ur_ee_link): {
  shape: capsule, size: [.06 .015], Q: "t(.00857 .058 -.1205)",
  color: [1 1 1 .5], contact: -1
}
openarm_finger2_coll (ur_ee_link): {
  shape: capsule, size: [.06 .015], Q: "t(-.01043 -.058 -.1205)",
  color: [1 1 1 .5], contact: -1
}
openarm_finger2_coll2 (ur_ee_link): {
  shape: capsule, size: [.06 .015], Q: "t(.00857 -.058 -.1205)",
  color: [1 1 1 .5], contact: -1
}

gripper_center (ur_ee_link): {
  shape: marker, size: [.03], Q: "t(-.0014 0 -.14) d(180 1 0 0)", color: [.9 .9 .9]
}
ur_gripper_center (gripper_center): {
  shape: sphere, size: [.004], contact: 0
}

openarm_right_ee_base_link_0 (ur_ee_link): { rel: "d(-90 1 0 0) t(0.0205 0.0 0.0) E(0.0 -0.0 0.0)", mesh: <meshes/ee/visual/ee_base_link.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }
openarm_right_ee_link1: { mass: 0.125979047366671, inertia: [3.61105363396623e-05 1.35880208293387e-06 7.13916341634141e-06 3.62137424319396e-05 -3.60343788693423e-06 1.74958961632907e-05],}
openarm_right_ee_link1_0 (openarm_right_ee_link1): { rel: "d(-90 1 0 0)", mesh: <meshes/ee/visual/finger_inner.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }
openarm_right_ee_link2: { mass: 0.11372, inertia: [3.4565e-05 -1.3392e-06 7.1576e-06 3.5306e-05 3.7172e-06 1.6571e-05],}
openarm_right_ee_link2_0 (openarm_right_ee_link2): { rel: "d(-90 1 0 0)", mesh: <meshes/ee/visual/finger_outer.dae>, color: [0.752941176470588 0.752941176470588 0.752941176470588 1], visual: true }

openarm_right_finger_joint1_origin (ur_ee_link): { Q: "t(-0.0014300000000005003 0.0180000000000009 -0.0680000000000174) E(0.0 -0.0 0.0)" }
openarm_right_finger_joint1 (openarm_right_finger_joint1_origin openarm_right_ee_link1): { joint: rigid, q: 0.5 }
openarm_right_finger_joint2_origin (ur_ee_link): { Q: "t(-0.0014300000000005003 -0.0180000000000012 -0.0680000000000215) E(0.0 -0.0 0.0)" }
openarm_right_finger_joint2 (openarm_right_finger_joint2_origin openarm_right_ee_link2): { joint: rigid, q: -0.5 }