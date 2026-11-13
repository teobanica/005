import sys, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import cases
from phis import Tk
for lab in ["K_6, random","T_6"]:
    H=cases.C[lab](); N=6; e,V=np.linalg.eigh(Tk(H,2))
    simple=[j for j in range(N*N) if (np.abs(e-e[j])<1e-8).sum()==1]
    ranks=[np.linalg.svd(V[:,j].reshape(N,N),compute_uv=False) for j in simple]
    print(lab, "simple eigvecs:",len(simple)," schmidt rank (sv>1e-8):",[int((s>1e-8).sum()) for s in ranks][:12])
    # commutation test: do T_2 and T_1 (x) T_1-type or rotations commute?
    T2=Tk(H,2); R=np.eye(N*N).reshape(N,N,N,N).transpose(0,1,3,2).reshape(N*N,N*N)  # flip legs
    print("   ||[T_2, flip]|| =",round(np.linalg.norm(T2@R-R@T2),6))
