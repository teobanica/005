import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m, ck, cases
from annular import lam_basis
from jordan import ops, apply2, TLbasis
from jordan2 import min_defect
import inspect
print(inspect.getsource(m.F82).splitlines()[0:6])
rng=np.random.default_rng(7); b=m.E(rng.random())
H=m.F82(1,b); N=8
print("Hadamard check:", np.abs(H@H.conj().T-N*np.eye(N)).max())
for k in [2,3,4]:
    t=time.time(); Q=lam_basis(N,k).astype(complex)
    TQ=ck.T_block_blas(H,k)(np.ascontiguousarray(Q)); M=Q.conj().T@TQ; M=(M+M.conj().T)/2
    e,V=np.linalg.eigh(M); L=Q@V[:,e>1-1e-9]; f=L.shape[1]
    if k==2: print("f_2 =",f,"(c_2 = %d)"%(2+f)); continue
    P=np.eye(N*N)-TLbasis(N)@TLbasis(N).conj().T; O=ops(N,k)
    B=np.zeros((f,f,len(O)*N*N),complex)
    for a in range(f):
        for bb in range(f): B[a,bb]=np.concatenate([P@apply2(op,L[:,a],L[:,bb],N,k).reshape(-1) for op in O])
    print("F_{8,2}(1,b), 2x4 Dita point, level %d: f_k = %d, min level-2 defect = %.3e (%.0f s)"%(k,f,min_defect(B,f,starts=20),time.time()-t),flush=True)
