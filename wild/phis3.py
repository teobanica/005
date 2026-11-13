import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import cases, ck, matrices as m
from phis import jw_basis
from phis2 import contain
def phi_spec_blk(H,s,B):
    TB=ck.T_block_blas(H,s)(np.ascontiguousarray(B.astype(complex)))
    return np.sort(np.linalg.eigvalsh(B.T@TB))[::-1]
t=time.time()
B={s:jw_basis(6,s) for s in [4,5]}; print("dims",[B[s].shape[1] for s in [4,5]], round(time.time()-t))
for lab in ["K_6, random","T_6","H_6^q, q random"]:
    H=cases.C[lab](); a=phi_spec_blk(H,4,B[4]); b=phi_spec_blk(H,5,B[5])
    mi=contain(a,b); print(lab,": Phi_4 in Phi_5:",len(mi),"unmatched of",len(a),"; traces %.4f %.4f"%(a.sum(),b.sum()), "; top Phi_5", np.round(b[:4],4))
for lab in ["P_7^q, q random","R_7^q, q random","F_{8,2}, random","M_{8,1}, random"]:
    H=cases.C[lab](); N=H.shape[0]
    a=phi_spec_blk(H,2,jw_basis(N,2)); b=phi_spec_blk(H,3,jw_basis(N,3))
    mi=contain(a,b); print(lab,": Phi_2 in Phi_3:",len(mi),"unmatched of",len(a))
