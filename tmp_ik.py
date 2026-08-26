from env import make_openarm_duo_env

C = make_openarm_duo_env(1)
jn = C.getJointNames()
idx = [i for i,n in enumerate(jn) if n.startswith('a2_openarm')]
q = [-0.60, -0.24, -0.34, -0.19, 1.28, -0.30, 0.78]
st = list(C.getJointState())
for k,i in enumerate(idx): st[i] = q[k]
C.setJointState(st)
C.view(True)
