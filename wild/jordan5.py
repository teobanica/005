import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m, ck
from annular import lam_basis
from jordan import ops, apply2, TLbasis
from jordan2 import min_defect
H=m.F6rs(np.exp(0.4j),np.exp(1.9j)); N=6; k=5
t=time.time(); Q=lam_basis(N,k).astype(complex); print("dim Lambda_5 =",Q.shape[1], round(time.time()-t))
TQ=ck.T_block_blas(H,k)(np.ascontiguousarray(Q)); M=Q.conj().T@TQ; M=(M+M.conj().T)/2
e,V=np.linalg.eigh(M); L=Q@V[:,e>1-1e-9]; f=L.shape[1]
print("f_5 (Dita 2x3) =",f,"; next eigenvalue of Psi_5:",round(e[e<1-1e-9].max(),4), "; residual", np.abs(ck.T_block_blas(H,k)(np.ascontiguousarray(L))-L).max())
P=np.eye(N*N)-TLbasis(N)@TLbasis(N).conj().T; O=ops(N,k)
B=np.zeros((f,f,len(O)*N*N),complex)
for a in range(f):
    for b in range(f): B[a,b]=np.concatenate([P@apply2(op,L[:,a],L[:,b],N,k).reshape(-1) for op in O])
print("2x3 Dita generic, level 5: min level-2 defect = %.3e  (%.0f s)"%(min_defect(B,f),time.time()-t))
