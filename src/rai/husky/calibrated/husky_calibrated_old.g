world_link: {}
base_link: { mass: 0.1, inertia: [0.03 0.0 0.0 0.03 0.0 0.03],}
base_link_0 (base_link): { mesh: <husky_description/meshes/base_link.stl>, visual: true }
base_footprint: {}
inertial_link: { mass: 46.034, inertia: [0.6022 -0.02364 -0.1197 1.7386 -0.001544 2.0296],}
imu_link: {}
front_left_wheel_link: { mass: 2.637, inertia: [0.02467 0 0 0.04411 0 0.02467],}
front_left_wheel_link_0 (front_left_wheel_link): { mesh: <husky_description/meshes/wheel.obj>, visual: true }
front_right_wheel_link: { mass: 2.637, inertia: [0.02467 0 0 0.04411 0 0.02467],}
front_right_wheel_link_0 (front_right_wheel_link): { mesh: <husky_description/meshes/wheel.obj>, visual: true }
rear_left_wheel_link: { mass: 2.637, inertia: [0.02467 0 0 0.04411 0 0.02467],}
rear_left_wheel_link_0 (rear_left_wheel_link): { mesh: <husky_description/meshes/wheel.obj>, visual: true }
rear_right_wheel_link: { mass: 2.637, inertia: [0.02467 0 0 0.04411 0 0.02467],}
rear_right_wheel_link_0 (rear_right_wheel_link): { mesh: <husky_description/meshes/wheel.obj>, visual: true }
top_chassis_link: {}
top_chassis_link_0 (top_chassis_link): { mesh: <husky_description/meshes/top_chassis.obj>, visual: true }
front_bumper_link: {}
front_bumper_link_0 (front_bumper_link): { mesh: <husky_description/meshes/bumper.obj>, visual: true }
rear_bumper_link: {}
rear_bumper_link_0 (rear_bumper_link): { mesh: <husky_description/meshes/bumper.obj>, visual: true }
dual_arm_bulkhead_link: { mass: 0.1, inertia: [0.03 0.0 0.0 0.03 0.0 0.03],}
dual_arm_bulkhead_link_0 (dual_arm_bulkhead_link): { mesh: <husky_ur_description/meshes/dual_arm_bulkhead.obj>, visual: true }
left_arm_bulkhead_link: {}
right_arm_bulkhead_link: {}
right_ur_arm_base_link: {}
right_ur_arm_base_link_inertia: { mass: 4.0, inertia: [0.00443333156 0.0 0.0 0.00443333156 0.0 0.0072],}
right_ur_arm_base_link_inertia_0 (right_ur_arm_base_link_inertia): { rel: "E(0 0 3.14159265359)", mesh: <ur_description/meshes/ur5e/visual/base.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
right_ur_arm_shoulder_link: { mass: 3.7, inertia: [0.010267495893 0.0 0.0 0.010267495893 0.0 0.00666],}
right_ur_arm_shoulder_link_0 (right_ur_arm_shoulder_link): { rel: "E(0 0 3.14159265359)", mesh: <ur_description/meshes/ur5e/visual/shoulder.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
right_ur_arm_upper_arm_link: { mass: 8.393, inertia: [0.133885781862 0.0 0.0 0.133885781862 0.0 0.0151074],}
right_ur_arm_upper_arm_link_0 (right_ur_arm_upper_arm_link): { rel: "t(0 0 0.138) E(1.57079632679 0 -1.57079632679)", mesh: <ur_description/meshes/ur5e/visual/upperarm.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
right_ur_arm_forearm_link: { mass: 2.275, inertia: [0.0312093550996 0.0 0.0 0.0312093550996 0.0 0.004095],}
right_ur_arm_forearm_link_0 (right_ur_arm_forearm_link): { rel: "t(0 0 0.007) E(1.57079632679 0 -1.57079632679)", mesh: <ur_description/meshes/ur5e/visual/forearm.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
right_ur_arm_wrist_1_link: { mass: 1.219, inertia: [0.00255989897604 0.0 0.0 0.00255989897604 0.0 0.0021942],}
right_ur_arm_wrist_1_link_0 (right_ur_arm_wrist_1_link): { rel: "t(0 0 -0.127) E(1.57079632679 0 0)", mesh: <ur_description/meshes/ur5e/visual/wrist1.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
right_ur_arm_wrist_2_link: { mass: 1.219, inertia: [0.00255989897604 0.0 0.0 0.00255989897604 0.0 0.0021942],}
right_ur_arm_wrist_2_link_0 (right_ur_arm_wrist_2_link): { rel: [0 0 -0.0997], mesh: <ur_description/meshes/ur5e/visual/wrist2.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
right_ur_arm_wrist_3_link: { mass: 0.1879, inertia: [9.89041005217e-05 0.0 0.0 9.89041005217e-05 0.0 0.0001321171875],}
right_ur_arm_wrist_3_link_0 (right_ur_arm_wrist_3_link): { rel: "t(0 0 -0.0989) E(1.57079632679 0 0)", mesh: <ur_description/meshes/ur5e/visual/wrist3.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
right_ur_arm_base: {}
right_ur_arm_flange: {}
right_ur_arm_tool0: {}
left_ur_arm_base_link: {}
left_ur_arm_base_link_inertia: { mass: 4.0, inertia: [0.00443333156 0.0 0.0 0.00443333156 0.0 0.0072],}
left_ur_arm_base_link_inertia_0 (left_ur_arm_base_link_inertia): { rel: "E(0 0 3.14159265359)", mesh: <ur_description/meshes/ur5e/visual/base.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
left_ur_arm_shoulder_link: { mass: 3.7, inertia: [0.010267495893 0.0 0.0 0.010267495893 0.0 0.00666],}
left_ur_arm_shoulder_link_0 (left_ur_arm_shoulder_link): { rel: "E(0 0 3.14159265359)", mesh: <ur_description/meshes/ur5e/visual/shoulder.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
left_ur_arm_upper_arm_link: { mass: 8.393, inertia: [0.133885781862 0.0 0.0 0.133885781862 0.0 0.0151074],}
left_ur_arm_upper_arm_link_0 (left_ur_arm_upper_arm_link): { rel: "t(0 0 0.138) E(1.57079632679 0 -1.57079632679)", mesh: <ur_description/meshes/ur5e/visual/upperarm.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
left_ur_arm_forearm_link: { mass: 2.275, inertia: [0.0312093550996 0.0 0.0 0.0312093550996 0.0 0.004095],}
left_ur_arm_forearm_link_0 (left_ur_arm_forearm_link): { rel: "t(0 0 0.007) E(1.57079632679 0 -1.57079632679)", mesh: <ur_description/meshes/ur5e/visual/forearm.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
left_ur_arm_wrist_1_link: { mass: 1.219, inertia: [0.00255989897604 0.0 0.0 0.00255989897604 0.0 0.0021942],}
left_ur_arm_wrist_1_link_0 (left_ur_arm_wrist_1_link): { rel: "t(0 0 -0.127) E(1.57079632679 0 0)", mesh: <ur_description/meshes/ur5e/visual/wrist1.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
left_ur_arm_wrist_2_link: { mass: 1.219, inertia: [0.00255989897604 0.0 0.0 0.00255989897604 0.0 0.0021942],}
left_ur_arm_wrist_2_link_0 (left_ur_arm_wrist_2_link): { rel: [0 0 -0.0997], mesh: <ur_description/meshes/ur5e/visual/wrist2.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
left_ur_arm_wrist_3_link: { mass: 0.1879, inertia: [9.89041005217e-05 0.0 0.0 9.89041005217e-05 0.0 0.0001321171875],}
left_ur_arm_wrist_3_link_0 (left_ur_arm_wrist_3_link): { rel: "t(0 0 -0.0989) E(1.57079632679 0 0)", mesh: <ur_description/meshes/ur5e/visual/wrist3.obj>, color: [0.7 0.7 0.7 1.0], visual: true }
left_ur_arm_base: {}
left_ur_arm_flange: {}
left_ur_arm_tool0: {}
world_footprint (world_link base_footprint): { joint: rigid,}
base_footprint_joint_origin (base_footprint): { Q: [0 0 0.13228] }
base_footprint_joint (base_footprint_joint_origin base_link): { joint: rigid,}
inertial_joint (base_link inertial_link): { joint: rigid,}
imu_joint_origin (base_link): { Q: "t(0.19 0 0.149) E(0 -1.5708 3.1416)" }
imu_joint (imu_joint_origin imu_link): { joint: rigid,}
front_left_wheel_origin (base_link): { Q: [0.256 0.2854 0.03282] }
front_left_wheel (front_left_wheel_origin front_left_wheel_link): { joint: rigid,}
front_right_wheel_origin (base_link): { Q: [0.256 -0.2854 0.03282] }
front_right_wheel (front_right_wheel_origin front_right_wheel_link): { joint: rigid,}
rear_left_wheel_origin (base_link): { Q: [-0.256 0.2854 0.03282] }
rear_left_wheel (rear_left_wheel_origin rear_left_wheel_link): { joint: rigid,}
rear_right_wheel_origin (base_link): { Q: [-0.256 -0.2854 0.03282] }
rear_right_wheel (rear_right_wheel_origin rear_right_wheel_link): { joint: rigid,}
top_chassis_joint (base_link top_chassis_link): { joint: rigid,}
front_bumper_origin (base_link): { Q: [0.48 0 0.091] }
front_bumper (front_bumper_origin front_bumper_link): { joint: rigid,}
rear_bumper_origin (base_link): { Q: "t(-0.48 0 0.091) E(0 0 3.14159)" }
rear_bumper (rear_bumper_origin rear_bumper_link): { joint: rigid,}
dual_arm_bulkhead_joint_origin (base_link): { Q: [0 0 0.224] }
dual_arm_bulkhead_joint (dual_arm_bulkhead_joint_origin dual_arm_bulkhead_link): { joint: rigid,}
left_arm_bulkhead_joint_origin (dual_arm_bulkhead_link): { Q: "t(0.1225 0.14891 0.13371) E(0 -0.785398163397 -1.57079632679)" }
left_arm_bulkhead_joint (left_arm_bulkhead_joint_origin left_arm_bulkhead_link): { joint: rigid,}
right_arm_bulkhead_joint_origin (dual_arm_bulkhead_link): { Q: "t(0.1225 -0.14891 0.13371) E(0 0.785398163397 -1.57079632679)" }
right_arm_bulkhead_joint (right_arm_bulkhead_joint_origin right_arm_bulkhead_link): { joint: rigid,}
left_arm_mount_joint (left_arm_bulkhead_link left_ur_arm_base_link): { joint: rigid,}
right_arm_mount_joint_origin (right_arm_bulkhead_link): { Q: "t(0.0011497288942337036 -0.0006287879659794271 0.002082839608192444) E(0.005849556200320057 -0.004160215708736291 -0.0035193052610289406)" }
right_arm_mount_joint (right_arm_mount_joint_origin right_ur_arm_base_link): { joint: rigid,}
right_ur_arm_base_link-base_link_inertia_origin (right_ur_arm_base_link): { Q: "E(0 0 3.14159265359)" }
right_ur_arm_base_link-base_link_inertia (right_ur_arm_base_link-base_link_inertia_origin right_ur_arm_base_link_inertia): { joint: rigid,}
right_ur_arm_shoulder_pan_joint_origin (right_ur_arm_base_link_inertia): { Q: [0 0 0.1627782525221701] }
right_ur_arm_shoulder_pan_joint (right_ur_arm_shoulder_pan_joint_origin right_ur_arm_shoulder_link): { joint: hingeZ, limits: [-6.28318530718 6.28318530718], ctrl_limits: [3.14159265359 -1 150.0],}
right_ur_arm_shoulder_lift_joint_origin (right_ur_arm_shoulder_link): { Q: "t(0.00017951387097150235 0 0) E(1.5703872163784984 0 -0.0000044300287598275637)" }
right_ur_arm_shoulder_lift_joint (right_ur_arm_shoulder_lift_joint_origin right_ur_arm_upper_arm_link): { joint: hingeZ, limits: [-4.71238898038 4.71238898038], ctrl_limits: [3.14159265359 -1 150.0],}
right_ur_arm_elbow_joint_origin (right_ur_arm_upper_arm_link): { Q: "t(-0.42484922240722683 0 0) E(0.000017598778356909152 -0.0017646390504777454 0.000022896679262627857)" }
right_ur_arm_elbow_joint (right_ur_arm_elbow_joint_origin right_ur_arm_forearm_link): { joint: hingeZ, limits: [-3.14159265359 3.14159265359], ctrl_limits: [3.14159265359 -1 150.0],}
right_ur_arm_wrist_1_joint_origin (right_ur_arm_forearm_link): { Q: "t(-0.39267466285441494 -0.0007932258609161224 0.13276574606781064) E(0.0059745562860579046 -0.0017403184893875676 -0.000012182431672809177)" }
right_ur_arm_wrist_1_joint (right_ur_arm_wrist_1_joint_origin right_ur_arm_wrist_1_link): { joint: hingeZ, limits: [-4.18879020479 4.18879020479], ctrl_limits: [3.14159265359 -1 28.0],}
right_ur_arm_wrist_2_joint_origin (right_ur_arm_wrist_1_link): { Q: "t(-0.000057633117397977084 -0.099557455213209314 0.00015287024635024139) E(1.569260830273018 0 0.00000037474165375667162)" }
right_ur_arm_wrist_2_joint (right_ur_arm_wrist_2_joint_origin right_ur_arm_wrist_2_link): { joint: hingeZ, limits: [-4.18879020479 4.18879020479], ctrl_limits: [3.14159265359 -1 28.0],}
right_ur_arm_wrist_3_joint_origin (right_ur_arm_wrist_2_link): { Q: "t(0.000058942281949617339 0.099514048224567206 0.00020241494029680382) E(1.5728303578162324 3.1415926535897931 3.1415917754729397)" }
right_ur_arm_wrist_3_joint (right_ur_arm_wrist_3_joint_origin right_ur_arm_wrist_3_link): { joint: hingeZ, limits: [-4.18879020479 4.18879020479], ctrl_limits: [3.14159265359 -1 28.0],}
right_ur_arm_base_link-base_fixed_joint_origin (right_ur_arm_base_link): { Q: "E(0 0 -1.570796326795)" }
right_ur_arm_base_link-base_fixed_joint (right_ur_arm_base_link-base_fixed_joint_origin right_ur_arm_base): { joint: rigid,}
right_ur_arm_wrist_3-flange_origin (right_ur_arm_wrist_3_link): { Q: "E(0 -1.57079632679 -1.57079632679)" }
right_ur_arm_wrist_3-flange (right_ur_arm_wrist_3-flange_origin right_ur_arm_flange): { joint: rigid,}
right_ur_arm_flange-tool0_origin (right_ur_arm_flange): { Q: "E(1.57079632679 0 1.57079632679)" }
right_ur_arm_flange-tool0 (right_ur_arm_flange-tool0_origin right_ur_arm_tool0): { joint: rigid,}
left_ur_arm_base_link-base_link_inertia_origin (left_ur_arm_base_link): { Q: "E(0 0 3.14159265359)" }
left_ur_arm_base_link-base_link_inertia (left_ur_arm_base_link-base_link_inertia_origin left_ur_arm_base_link_inertia): { joint: rigid,}
left_ur_arm_shoulder_pan_joint_origin (left_ur_arm_base_link_inertia): { Q: [0 0 0.16257807277263761] }
left_ur_arm_shoulder_pan_joint (left_ur_arm_shoulder_pan_joint_origin left_ur_arm_shoulder_link): { joint: hingeZ, limits: [-6.28318530718 6.28318530718], ctrl_limits: [3.14159265359 -1 150.0],}
left_ur_arm_shoulder_lift_joint_origin (left_ur_arm_shoulder_link): { Q: "t(0.00019007197441139241 0 0) E(1.5713901981958054 0 -0.0000026681008682416518)" }
left_ur_arm_shoulder_lift_joint (left_ur_arm_shoulder_lift_joint_origin left_ur_arm_upper_arm_link): { joint: hingeZ, limits: [-4.71238898038 4.71238898038], ctrl_limits: [3.14159265359 -1 150.0],}
left_ur_arm_elbow_joint_origin (left_ur_arm_upper_arm_link): { Q: "t(-0.42492076346025309 0 0) E(0.00053885028029254363 -0.00088899660595502864 -0.0000090893692096143134)" }
left_ur_arm_elbow_joint (left_ur_arm_elbow_joint_origin left_ur_arm_forearm_link): { joint: hingeZ, limits: [-3.14159265359 3.14159265359], ctrl_limits: [3.14159265359 -1 150.0],}
left_ur_arm_wrist_1_joint_origin (left_ur_arm_forearm_link): { Q: "t(-0.39208100648997279 -0.00090760192761138176 0.13352893003432456) E(0.0067969386915389153 0.0021092387583197063 -0.0000069072557868343565)" }
left_ur_arm_wrist_1_joint (left_ur_arm_wrist_1_joint_origin left_ur_arm_wrist_1_link): { joint: hingeZ, limits: [-4.18879020479 4.18879020479], ctrl_limits: [3.14159265359 -1 28.0],}
left_ur_arm_wrist_2_joint_origin (left_ur_arm_wrist_1_link): { Q: "t(0.00013627350480989177 -0.099746834502379381 0.0000073743752427287777) E(1.5707223958750225 0 -0.00000021941801741677536)" }
left_ur_arm_wrist_2_joint (left_ur_arm_wrist_2_joint_origin left_ur_arm_wrist_2_link): { joint: hingeZ, limits: [-4.18879020479 4.18879020479], ctrl_limits: [3.14159265359 -1 28.0],}
left_ur_arm_wrist_3_joint_origin (left_ur_arm_wrist_2_link): { Q: "t(0.000025819376543512202 0.099466410942852038 0.000018368979052949448) E(1.5709810019899699 3.1415926535897931 -3.1415925563992939)" }
left_ur_arm_wrist_3_joint (left_ur_arm_wrist_3_joint_origin left_ur_arm_wrist_3_link): { joint: hingeZ, limits: [-4.18879020479 4.18879020479], ctrl_limits: [3.14159265359 -1 28.0],}
left_ur_arm_base_link-base_fixed_joint_origin (left_ur_arm_base_link): { Q: "E(0 0 3.14159265359)" }
left_ur_arm_base_link-base_fixed_joint (left_ur_arm_base_link-base_fixed_joint_origin left_ur_arm_base): { joint: rigid,}
left_ur_arm_wrist_3-flange_origin (left_ur_arm_wrist_3_link): { Q: "E(0 -1.57079632679 -1.57079632679)" }
left_ur_arm_wrist_3-flange (left_ur_arm_wrist_3-flange_origin left_ur_arm_flange): { joint: rigid,}
left_ur_arm_flange-tool0_origin (left_ur_arm_flange): { Q: "E(1.57079632679 0 1.57079632679)" }
left_ur_arm_flange-tool0 (left_ur_arm_flange-tool0_origin left_ur_arm_tool0): { joint: rigid,}
