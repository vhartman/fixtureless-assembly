# OpenArm v2 humanoid duo -- driver file.
# Arms live in left_openarm.g / right_openarm.g (edit those for link/capsule
# changes); this file sets the torso and where the arms bolt on.

# --- torso (box frame) ---
openarm_torso: {}
torso_base (openarm_torso): { shape: box, size: [.26, .22, .04, .02], color: [0.35 0.35 0.38 0.5], Q: "d(0 1 0 0) t(0 0 .02)", visual: true, contact: -1 }
torso_column (openarm_torso): { shape: box, size: [.07, .07, .56, .02], color: [0.45 0.45 0.48 0.5], Q: "d(0 1 0 0) t(0 0 .36)", visual: true, contact: -1 }
#torso_shoulder_plate (openarm_torso): { shape: box, size: [.15, .12, .15, .02], color: [0.55 0.55 0.58 0.5], Q: "d(0 1 0 0) t(0 0 .673)", visual: true, contact: -1 }

# --- original body mesh (mm units, downscaled) -- toggle back if preferred ---
openarm_body_link0_0 (openarm_torso): { mesh: <meshes/arm/visual/body_link0.dae>, meshscale: [0.001 0.001 0.001], color: [0.75 0.75 0.75 1], visual: true }

# --- arms ---
Include: <right_openarm.g>
Include: <left_openarm.g>

Edit right_base (openarm_torso): { Q: [0, -0.031, 0.698], joint: rigid }
Edit left_base (openarm_torso): { Q: [0, 0.031, 0.698], joint: rigid }
