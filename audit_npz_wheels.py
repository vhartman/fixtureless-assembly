"""One-off audit helper: inspect the recorded npz at frame 0 and print the wheel geoms'
vertical extent vs the MuJoCo table top (diagnoses 'wheels start inside the table')."""
import numpy as np
import mujoco

z = np.load("car-mini-3r.npz", allow_pickle=False)
model = mujoco.MjModel.from_xml_string(str(z["xml"]))
data = mujoco.MjData(model)
data.qpos[:] = z["qpos"][0]
mujoco.mj_forward(model, data)

tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
if tid >= 0:
    gpos = data.geom_xpos[tid]
    size = model.geom_size[tid]
    print(f"table geom: z={gpos[2]:.4f} size_z={size[2]:.4f} -> top={gpos[2] + size[2]:.4f}")

print("frame 0 part geoms:")
for p in ["obj_0", "obj_1", "obj_2", "obj_3", "obj_4"]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p)
    if bid < 0:
        print(f"{p}: body missing")
        continue
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid:
            continue
        gxp = data.geom_xpos[gid]
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            size = model.geom_size[gid]
            axis_w = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)[:, 2]
            bottom = gxp[2] - (abs(axis_w[2]) * size[1]
                               + np.hypot(axis_w[0], axis_w[1]) * size[0])
            print(f"{p}: cyl z={gxp[2]:.4f} axis_w={np.round(axis_w, 3)} "
                  f"half_len={size[1]:.4f} r={size[0]:.4f} bottom={bottom:.4f}")
        else:
            print(f"{p}: geom type {model.geom_type[gid]}, z={gxp[2]:.4f}")
