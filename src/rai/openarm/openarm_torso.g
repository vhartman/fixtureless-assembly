# OpenArm v2 torso -- synced with openarm_duo.g (user-tuned sizes/orientations).
openarm_torso: {}

openarm_body_link0 (openarm_torso): { joint: rigid, }

# --- box frame (collision decoration); shoulders mount at z=0.698 ---
torso_base (openarm_torso): { shape: box, size: [.26, .22, .04, .02], color: [0.35 0.35 0.38 0.5], Q: "d(0 1 0 0) t(0 0 .02)", visual: true, contact: 1 }
torso_column (openarm_torso): { shape: box, size: [.07, .07, .56, .02], color: [0.45 0.45 0.48 0.5], Q: "d(0 1 0 0) t(0 0 .36)", visual: true, contact: 1 }
#torso_shoulder_plate (openarm_torso): { shape: box, size: [.15, .12, .15, .02], color: [0.55 0.55 0.58 0.5], Q: "d(0 1 0 0) t(0 0 .673)", visual: true, contact: -1 }

# --- original body mesh (mm units, downscaled) ---
openarm_body_link0_0 (openarm_torso): { mesh: <meshes/arm/visual/body_link0.dae>, meshscale: [0.001 0.001 0.001], color: [0.75 0.75 0.75 1], visual: true }

left_mount_origin (openarm_torso): { Q: [0, 0.031, 0.698] }
right_mount_origin (openarm_torso): { Q: [0, -0.031, 0.698] }
