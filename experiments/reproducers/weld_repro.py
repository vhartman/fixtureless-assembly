"""MINIMAL REPRODUCER for the --articulated grasp-slip defect (sim.py).

One arm + one welded box, nothing else: no parts, table, contacts, other arms or executor. A
weld to a MOCAP body holds the box to ~2.7mm; the SAME weld to a JOINTED body lets it slip
~30mm. That is the entire difference between sim.py's mocap arms (which work) and
--articulated (which misses seats by 66mm and then cascades).

    python3 experiments/reproducers/weld_repro.py

THE SHARPEST FACT, and the one to start from: with the arm held COMPLETELY STATIONARY the box
still slips 24.18mm (mocap: 0.02mm). So this is not a tracking lag and has nothing to do with
motion -- a weld to a jointed body simply does not hold a body still, and the slip is
INVARIANT to every parameter that should matter. Slip is 0.0000mm at the instant of
activation, then grows to a bounded ~24mm.

SIX HYPOTHESES, ALL REFUTED HERE BY MEASUREMENT -- do not re-try them:
  * "the weld accelerates the arm"      -> the arm moves 0.0000mm during the step, both modes
  * "the weld is too soft"              -> solref 0.004 / 0.002 / 0.001 / 0.0002 all give
                                            BYTE-IDENTICAL 24.1771mm (tested here, in the XML
                                            directly, not via a monkeypatch)
  * "the arm takes a share of the
     correction, so give it more inertia"-> armature 0/1/1e2/1e4/1e6 makes it monotonically
                                            WORSE (30 -> 31 -> 40 -> 50 -> 50mm)
  * "the mass ratio is bad"             -> link_mass 0.1/1/5/50 kg (500x) -> 30.49mm, identical
  * "the solver has not converged"      -> iterations 200 / 1000 -> 30.4941mm, identical
  * "the velocity constraint holds the
     box to a stationary anchor"        -> commanded qvel changes nothing (30.46 vs 30.49mm)
  * "the eq_data relpose convention is
     backwards"                         -> body2-in-body1 (what sim does) is the BETTER one in
                                            both modes; the inverse is worse (mocap 2.7 -> 23.5)

Still open. A constraint that converges to a fixed 24mm offset, unaffected by stiffness, mass,
inertia or iteration count, looks like it is being satisfied against a target that is not what
we think -- but the target is provably correct at activation (slip 0.0000mm), and the arm is
provably where rai says (fk_compare: machine precision; and 0.0000mm inside the full sim
scene). ALSO REFUTED: MuJoCo evaluating the weld against INERTIAL rather than body frames -- measured
xpos vs xipos offset is 0.0000mm for BOTH the anchor link and the box, so the two frames
coincide and cannot explain it.

Nothing about the dynamics explains a fixed, parameter-invariant 24mm. The next step should be
to ask MuJoCo directly instead of inferring: dump the weld's own rows of d.efc_pos / efc_force
and mj_constraintUpdate for both models, i.e. find out whether MuJoCo thinks the constraint is
violated at all. If efc says residual ~0 while our metric says 24mm, the disagreement is in
what the constraint is anchored to; if efc says 24mm, the solver is choosing not to close
it."""
import os
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)
import numpy as np, mujoco, robotic as ry
import assembly_planner.arm_model as arm_model
from assembly_planner.env import make_four_robot_env
import assembly_planner.assembly as A

C = make_four_robot_env(num_robots=1, floating_ee=False, vacuum=False, robot_types="ur10")
prefix = A.get_robots(C)[0]
# mirror sim._arm_shape_frames/ee_anchor: last non-mesh shape frame, or gripper_fill if present
cands = [n for n in C.getFrameNames() if n.startswith(prefix)
         and C.getFrame(n).info().get("shape") not in (None, "mesh", "marker")]
anchor = next((f for f in cands if "gripper_fill" in f), cands[-1])
print("anchor frame:", anchor)
q0 = np.asarray(C.getJointState(), float)
C.setJointState(q0)
gp = np.asarray(C.getFrame(anchor).getPosition(), float)

def model_xml(articulated, armature=0.0, mass=50.0, iters=None, solref="0.004 1"):
  body, jn = arm_model.build_arms_mjcf(C, [prefix], collide_fn=lambda n: False,
                                       link_mass=mass, link_inertia=mass / 10.0, armature=armature)
  if articulated:
    arm = body
    a_body = anchor
  else:                                        # a single mocap puck standing in for the arm
    arm = (f'<body name="mock" mocap="true" pos="{gp[0]} {gp[1]} {gp[2]}">'
           f'<geom type="sphere" size="0.02" contype="0" conaffinity="0" mass="1"/></body>')
    a_body = "mock"
  it = "" if iters is None else f' iterations="{iters}" ls_iterations="{max(iters // 2, 20)}"'
  return f"""<mujoco>
  <option timestep="0.002"{it}/>
  <worldbody>
{arm}
    <body name="box" pos="{gp[0]} {gp[1]} {gp[2] - 0.05}">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <equality><weld name="g" body1="{a_body}" body2="box" active="false" solref="{solref}"/></equality>
</mujoco>""", jn, a_body

def run(articulated, reassert=False, armature=0.0, use_qvel=False, conv='b2_in_b1',
        mass=50.0, iters=None, speed=1.0, solref="0.004 1"):
  xml, jn, a_body = model_xml(articulated, armature, mass, iters, solref)
  m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
  bid_a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, a_body)
  bid_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
  eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "g")
  rai_j = list(C.getJointNames())
  qadr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
          for n in rai_j if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0}
  vadr = {n: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
          for n in qadr}

  def place(q, qd=None):
    if articulated:
      for i, n in enumerate(rai_j):
        if n in qadr:
          d.qpos[qadr[n]] = q[i]
          d.qvel[vadr[n]] = 0.0 if (qd is None or not use_qvel) else qd[i]
    else:
      C.setJointState(q)
      d.mocap_pos[0] = np.asarray(C.getFrame(anchor).getPosition(), float)
      d.mocap_quat[0] = np.asarray(C.getFrame(anchor).getQuaternion(), float)
    mujoco.mj_forward(m, d)

  place(q0)
  # activate the weld exactly the way sim._activate_weld does
  p1, q1 = d.xpos[bid_a].copy(), d.xquat[bid_a].copy()
  p2, q2 = d.xpos[bid_b].copy(), d.xquat[bid_b].copy()
  R1 = np.zeros(9); mujoco.mju_quat2Mat(R1, q1); R1 = R1.reshape(3, 3)
  R2 = np.zeros(9); mujoco.mju_quat2Mat(R2, q2); R2 = R2.reshape(3, 3)
  if conv == "b2_in_b1":                       # what sim._activate_weld does today
    rel_p = R1.T @ (p2 - p1)
    qc = np.zeros(4); mujoco.mju_negQuat(qc, q1)
    rel_q = np.zeros(4); mujoco.mju_mulQuat(rel_q, qc, q2)
  else:                                        # the inverse convention: body1 expressed in body2
    rel_p = R2.T @ (p1 - p2)
    qc = np.zeros(4); mujoco.mju_negQuat(qc, q2)
    rel_q = np.zeros(4); mujoco.mju_mulQuat(rel_q, qc, q1)
  m.eq_data[eq][:3] = 0.0; m.eq_data[eq][3:6] = rel_p
  m.eq_data[eq][6:10] = rel_q; m.eq_data[eq][10] = 1.0
  d.eq_active[eq] = 1
  mujoco.mj_forward(m, d)

  out = []
  for k in range(1500):
    amp = np.array([0.4, 0.25, -0.3, 0.2, 0.15, 0.1])
    q = q0 + speed * amp * np.sin(2 * np.pi * k / 1500.0)
    qd = speed * amp * (2 * np.pi / 1500.0) * np.cos(2 * np.pi * k / 1500.0) / 0.002
    place(q, qd)
    pre = d.xpos[bid_a].copy()                 # anchor pose the weld was told about
    mujoco.mj_step(m, d)
    moved = np.linalg.norm(d.xpos[bid_a] - pre) * 1000   # did the ARM move during the step?
    if reassert:
      place(q, qd)
    if k % 300 == 0 or k == 1499:
      p1, q1 = d.xpos[bid_a], d.xquat[bid_a]
      p2 = d.xpos[bid_b]
      R1 = np.zeros(9); mujoco.mju_quat2Mat(R1, q1); R1 = R1.reshape(3, 3)
      if conv == "b2_in_b1":
        slip = np.linalg.norm(R1.T @ (p2 - p1) - m.eq_data[eq][3:6]) * 1000
      else:
        R2m = np.zeros(9); mujoco.mju_quat2Mat(R2m, d.xquat[bid_b]); R2m = R2m.reshape(3, 3)
        slip = np.linalg.norm(R2m.T @ (p1 - p2) - m.eq_data[eq][3:6]) * 1000
      out.append((k, slip, moved))
  return out

print(f"{'config':28s} {'max slip (mm)':>14s}")
rows = [("MOCAP  solref 0.004 1 (reference)", dict(articulated=False, speed=0.0))]
for sr in ("0.004 1", "0.002 1", "0.001 1", "0.0002 1"):
  rows.append((f"ARTIC  stationary, solref {sr}",
               dict(articulated=True, speed=0.0, solref=sr)))

for label, kw in rows:
  res = run(**kw)
  print(f"{label:28s} {max(s for _, s, _ in res):14.4f}")
