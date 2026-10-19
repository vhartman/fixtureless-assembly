"""Generate MuJoCo MJCF models of the rai FR3 arms (right/left) for `ssik build`.

Extracts the 7-joint chain straight from the live rai scene (relative body poses,
hinge axes, limits), so the artifact matches OUR fr3v2 asset exactly -- unlike the
ssik franka prebuilt, which fits no affine map to this model.

Run:  .venv/bin/python tools/calibration/calibrate_fr3.py
      -> writes ik_artifacts/fr3{r,l}.mjcf
Then: .venv/bin/ssik build ik_artifacts/fr3r.mjcf --base fr3r_base --ee fr3r_tool0 \
        --out ik_artifacts/fr3r_ik.py   (and fr3l likewise)
"""
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import robotic as ry


def pose(C, n):
    f = C.getFrame(n)
    p = f.getPosition()
    q = f.getQuaternion()
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def rel(C, child, parent):
    return np.linalg.inv(pose(C, parent)) @ pose(C, child)


def fmt(T):
    q = np.zeros(4)
    tr = T[0, 0] + T[1, 1] + T[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        q = [0.25 * s, (T[2, 0] - T[0, 2]) / s, (T[1, 0] - T[2, 0]) / s, (T[0, 1] - T[1, 0]) / s]
    elif T[0, 0] > T[1, 1] and T[0, 0] > T[2, 2]:
        s = np.sqrt(1 + T[0, 0] - T[1, 1] - T[2, 2]) * 2
        q = [(T[2, 1] - T[1, 2]) / s, 0.25 * s, (T[0, 1] + T[1, 0]) / s, (T[1, 2] + T[2, 0]) / s]
    elif T[1, 1] > T[2, 2]:
        s = np.sqrt(1 + T[1, 1] - T[0, 0] - T[2, 2]) * 2
        q = [(T[0, 2] - T[2, 0]) / s, (T[0, 1] + T[1, 0]) / s, 0.25 * s, (T[1, 2] + T[2, 0]) / s]
    else:
        s = np.sqrt(1 + T[2, 2] - T[0, 0] - T[1, 1]) * 2
        q = [(T[1, 0] - T[0, 1]) / s, (T[1, 2] + T[2, 1]) / s, (T[0, 2] + T[2, 0]) / s, 0.25 * s]
    p = T[:3, 3]
    return (f'pos="{p[0]:.9f} {p[1]:.9f} {p[2]:.9f}" '
            f'quat="{q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {q[0]:.9f}"')


def gen(side):
    C = ry.Config()
    w = C.addFrame("world")
    root = C.addFile(f"./src/rai/fr3/{side}_fr3.g")
    root.setParent(w)
    root.setJoint(ry.JT.rigid)
    _ = C.getJointNames()
    lo, hi = C.getJointLimits()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)

    L = [f"{side}_fr3v2_link{i}" for i in range(9)]      # link0 .. link8
    out = ['<mujoco model="fr3_' + side + '">', "  <worldbody>"]
    # base anchor body (rigid mount plane == the .g file root)
    out.append(f'    <body name="{side}_base">')
    # link0 hangs rigidly off the base
    t = rel(C, L[0], f"{side}_base")
    out.append(f'      <body name="{L[0]}" {fmt(t)}>')
    out.append('        <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>')
    out.append("      </body>")
    prev = L[0]
    for i in range(1, 8):
        jf = f"{side}_fr3v2_joint{i}_origin"
        t = rel(C, jf, prev)
        out.append(f'      <body name="{L[i]}" {fmt(t)}>')
        out.append(f'        <joint name="{side}_joint{i}" type="hinge" '
                   f'axis="0 0 1" range="{lo[i - 1]:.9f} {hi[i - 1]:.9f}"/>')
        out.append('        <inertial pos="0 0 0" mass="1e-6" '
                   'diaginertia="1e-9 1e-9 1e-9"/>')
        prev = L[i]
    out.append(f'      <body name="{side}_tool0"/>')       # EE = flange
    out.append("      </body>" * 9)
    out.append("  </worldbody>")
    out.append("</mujoco>")
    path = f"ik_artifacts/fr3{'r' if side == 'right' else 'l'}.mjcf"
    open(path, "w").write("\n".join(out) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    gen("right")
    gen("left")



def build_scene():
    C = ry.Config()
    world = C.addFrame("world")
    root = C.addFile("./src/rai/fr3/right_fr3.g")
    root.setParent(world)
    root.setJoint(ry.JT.rigid)
    return C


def rai_flange_pose(C):
    """base(root)->flange(link8) 4x4 from the current joint state."""
    import robotic as _ry
    T_root = np.eye(4)
    p = C.getFrame("right_fr3v2_link8").getPosition()
    q = C.getFrame("right_fr3v2_link8").getQuaternion()
    T_root[:3, 3] = p
    w, x, y, z = q
    T_root[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return T_root


def main():
    C = build_scene()
    _ = C.getJointNames()          # rai quirk: limits are empty until names were queried
    lo, hi = C.getJointLimits()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    from ssik.prebuilt.franka import fr3_ik as U
    print("dof:", len(lo), "| ssik DOF:", U.DOF, "| EE:", U.EE_LINK, "| base:", U.BASE_LINK)
    rng = np.random.default_rng(0)

    # ---- stage 1: does the IDENTITY mapping hold? --------------------------------------
    ident_hits = 0
    sol_counts = []
    match_res = []
    for _ in range(N_SAMPLES):
        q = rng.uniform(lo, hi)
        C.setJointState(q)
        T = rai_flange_pose(C)
        sols = U.solve(T, respect_limits=False)
        sol_counts.append(len(sols))
        best = None
        for s in sols:
            qs = np.asarray(s.q, float)[:7]
            d = np.abs(q - qs)
            if best is None or d.max() < best[0]:
                best = (d.max(), d)
        if best is not None:
            match_res.append(best[1])
            if best[0] < 1e-6:
                ident_hits += 1
    print(f"samples={N_SAMPLES} avg sols={np.mean(sol_counts):.1f} "
          f"identity-exact hits={ident_hits}")
    if match_res:
        mr = np.array(match_res)
        print("per-joint |q_rai - q_ssik| MEDIAN (identity, any branch):")
        print(" ", np.round(np.median(mr, axis=0), 3))

    # ---- stage 2: OFFSET/SIGN VOTING ----------------------------------------------------
    # For the true mapping q_rai = s*q_sol + o (mod 2pi), every (sample, solution) pair
    # contributes one CORRECT vote o = wrap(q_rai_i - s*q_sol_i) per joint; wrong pairings
    # spread uniformly. A sharp histogram peak reveals the true offset per joint and sign.
    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    pairs = []                                           # (q_rai, [sol arrays])
    for _ in range(N_SAMPLES):
        q = rng.uniform(lo, hi)
        C.setJointState(q)
        C.setJointState(C.getJointState())               # refresh (stale-pose quirk)
        T = rai_flange_pose(C)
        sols = [np.asarray(s.q, float)[:7] for s in U.solve(T, respect_limits=False)]
        if sols:
            pairs.append((q, sols))
    print(f"voting pool: {len(pairs)} samples x {sum(len(s) for _, s in pairs)} sols")

    results = {}
    for si in (1.0, -1.0):
        bins = np.linspace(-np.pi, np.pi, 721)
        for i in range(7):
            votes = []
            for q, sols in pairs:
                for sol in sols:
                    votes.append(wrap(q[i] - si * sol[i]))
            hist, edges = np.histogram(votes, bins=bins)
            k = int(hist.argmax())
            frac = hist[k] / max(1, len(votes))
            results[(si, i)] = (frac, (edges[k] + edges[k + 1]) / 2)
    print("per-joint peak sharpness (+1 vs -1 sign) and peak offset:")
    for i in range(7):
        fp, op = results[(1.0, i)]
        fm, om = results[(-1.0, i)]
        star = "+" if fp >= fm else "-"
        off = op if fp >= fm else -om
        print(f"  joint{i+1}: sign{star} peak={max(fp, fm):.2f} offset={off:+.4f}")
    return C


if __name__ == "__main__":
    main()
