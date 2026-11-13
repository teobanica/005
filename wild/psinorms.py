import sys, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import cases, ck
from annular import lam_basis
N=6; Qs={t:lam_basis(N,t).astype(complex) for t in [2,3,4,5]}
for lab in ["T_6","H_6^q, q random","BN_6^q, q random on arc","K_6, random","BF_6"]:
    H=cases.C[lab](); out=[]
    for t in [2,3,4,5]:
        Q=Qs[t]; M=Q.conj().T@ck.T_block_blas(H,t)(np.ascontiguousarray(Q)); out.append(np.linalg.eigvalsh((M+M.conj().T)/2).max())
    print("%-26s ||Psi_t||, t=2..5: %s"%(lab," ".join("%.4f"%x for x in out)))
