import sys, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, ".")
import cases, matrices as m
from iso import T
for lab in ["K_6, random","T_6","BN_6^q, q random on arc","P_7^q, q random"]:
    H=cases.C[lab](); print(lab, "||H-H^t|| =", round(np.abs(H-H.T).max(),3))
rng=np.random.default_rng(9)
H=cases.C["P_7^q, q random"]()
P1=np.eye(7)[rng.permutation(7)]; P2=np.eye(7)[rng.permutation(7)]
D1=np.diag(np.exp(2j*np.pi*rng.random(7))); D2=np.diag(np.exp(2j*np.pi*rng.random(7)))
H=D1@P1@H@P2@D2
print("scrambled P_7: ||H-H^t|| =",round(np.abs(H-H.T).max(),3))
for k in [2,3,4]:
    a=np.linalg.eigvalsh(T(H,k)[0]); b=np.linalg.eigvalsh(T(H.T,k)[0])
    print("  k=%d diff %.1e, c_k = %d, %d"%(k,np.abs(np.sort(a)-np.sort(b)).max(),(a>1-1e-9).sum(),(b>1-1e-9).sum()))
