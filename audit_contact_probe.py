"""One-off audit probe: load the recorded MJCF model and step it in isolation to check whether
the wheel geoms generate contact with the table. Prints wheel z, table top, and contact count."""
import numpy as np

import mujoco

print("mujoco version:", mujoco.__version__)
import importlib.metadata as _md
try:
    print("mjVersion:", mujoco.mj_version())
except Exception as e:
    print("mj_version:", e)
print("installed dist:", _md.version("mujoco"))

d = np.load("car-mini-3r.npz", allow_pickle=True)
m = mujoco.MjModel.from_xml_string(str(d["xml"]))
data = mujoco.MjData(m)
print("ngeom:", m.ngeom, "ncon max:", m.nconmax, "table top z ~0.202")
q0 = d["qpos"][0].copy()
data.qpos[:] = q0
names = {m.body(i).name: i for i in range(m.nbody)}
wbodies = ["obj_1", "obj_2", "obj_3", "obj_4", "obj_0"]


def qpos_of(bname):
    a = m.jnt_qposadr[m.body(bname).jntadr[0]]
    return data.qpos[a: a + 7]


for step in range(60):
    mujoco.mj_step(m, data)
    if step in (0, 1, 2, 5, 10, 20, 59):
        zs = [round(float(qpos_of(b)[2]), 4) for b in wbodies]
        print(f"step {step}: obj_1..4,obj_0 z {zs} ncon={data.ncon}")
for i in range(data.ncon):
    c = data.contact[i]
    print("contact", i, repr(m.geom(c.geom1).name), "<->", repr(m.geom(c.geom2).name),
          "dist", round(c.dist, 5), "dim", c.dim)
print("cfrc_ext obj_1:", np.round(data.cfrc_ext[m.body("obj_1").id], 3))
print("cfrc_ext obj_0:", np.round(data.cfrc_ext[m.body("obj_0").id], 3))
print("wheel vel:", np.round(data.qvel[m.jnt_dofadr[m.body("obj_1").jntadr[0]]:
                                     m.jnt_dofadr[m.body("obj_1").jntadr[0]] + 3], 4))
print("nconmax:", m.nconmax)
a = m.jnt_dofadr[m.body("obj_1").jntadr[0]]
print("qfrc_constraint obj_1 (first 6):", np.round(data.qfrc_constraint[a:a + 6], 4))
a0 = m.jnt_dofadr[m.body("obj_0").jntadr[0]]
print("qfrc_constraint obj_0 (first 6):", np.round(data.qfrc_constraint[a0:a0 + 6], 4))

# wheel geom world orientation and contact frame in the FULL model
gid = m.body("obj_1").geomadr[0]
xmat = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
print("wheel geom xmat:\n", np.round(xmat, 4), "axis(z-col):", np.round(xmat[:, 2], 4))
print("wheel geom xpos:", np.round(data.geom_xpos[gid], 4))
c = data.contact[4]
print("contact4 frame (pos+normal):", np.round(c.frame[:7], 4))
print("contact4 friction:", c.friction, "solref:", c.solref, "solimp:", c.solimp)

# minimal model: one flat cylinder on a static box, same option line
def mini(extra_opt="", geom="<geom type=\"cylinder\" size=\"0.024 0.006\" quat=\"0.70710678 0.70710678 0 0\" mass=\"0.1\"/>", z=0.208, face_down=False):
    body_q = ' quat="0.70710678 -0.70710678 0 0"' if face_down else ""
    xml = f"""
<mujoco>
  <option timestep="0.002" {extra_opt}/>
  <worldbody>
    <geom type="box" size="1 1 0.101" pos="0 0 0.101"/>
    <body name="disc" pos="0 0 {z}"{body_q}>
      <freejoint/>
      {geom}
    </body>
  </worldbody>
</mujoco>
"""
    mm = mujoco.MjModel.from_xml_string(xml)
    md = mujoco.MjData(mm)
    for s in range(600):
        mujoco.mj_step(mm, md)
    # Both body orientations make the cylinder axis vertical after composition with the
    # geom quaternion. Its vertical half-extent is therefore half-length (6mm), not the
    # diameter or radius.
    bottom = md.qpos[2] - 0.006
    print(f"{('fd ' if face_down else 'st ') + (extra_opt or 'impfast'):30s} "
          f"{geom[:44]:44s} z={md.qpos[2]:.4f} bottom={bottom:.4f} (table 0.202) ncon={md.ncon}")


mini()
mini(face_down=True)
mini(face_down=True, geom="<geom type=\"cylinder\" size=\"0.024 0.006\" quat=\"0.70710678 0.70710678 0 0\" mass=\"0.1\" friction=\"1 1 1\"/>")
mini(face_down=True, geom="<geom type=\"cylinder\" size=\"0.024 0.006\" quat=\"0.70710678 0.70710678 0 0\" mass=\"0.1\" condim=\"1\"/>")
mini(face_down=True, geom="<geom type=\"cylinder\" size=\"0.024 0.006\" quat=\"0.70710678 0.70710678 0 0\" mass=\"0.1\" solref=\"0.005 1\" solimp=\"0.9 0.95 0.0001 0.5 2\"/>")
mini(face_down=True, geom="<geom type=\"cylinder\" size=\"0.024 0.006\" quat=\"0.70710678 0.70710678 0 0\" mass=\"0.1\" solimp=\"0.99 0.999 0.0001 0.5 2\"/>")
# upright peg (normal cylinder proportions, face down on the table):
mini(face_down=True, geom="<geom type=\"cylinder\" size=\"0.006 0.024\" quat=\"0.70710678 0.70710678 0 0\" mass=\"0.1\"/>", z=0.226)
# flat pancake with less extreme ratio h=12mm r=24mm:
mini(face_down=True, geom="<geom type=\"cylinder\" size=\"0.024 0.012\" quat=\"0.70710678 0.70710678 0 0\" mass=\"0.1\"/>", z=0.214)
# flat pancake with geom margin:
mini(face_down=True, geom="<geom type=\"cylinder\" size=\"0.024 0.006\" quat=\"0.70710678 0.70710678 0 0\" mass=\"0.1\" margin=\"0.001\"/>", z=0.214)
# collision box + visual cylinder (the proposed fix):
mini(face_down=True, z=0.214,
     geom="<geom type=\"box\" size=\"0.024 0.024 0.006\" mass=\"0.1\"/>"
          "<geom type=\"cylinder\" size=\"0.024 0.006\" quat=\"0.70710678 0.70710678 0 0\" contype=\"0\" conaffinity=\"0\"/>")
# sphere instead of capsule (sphere r=0.024, half-flattened not possible) -- control:
mini(face_down=True, geom="<geom type=\"sphere\" size=\"0.024\" mass=\"0.1\"/>", z=0.226)
