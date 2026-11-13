import sys, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import cases, matrices as m
from annular import lam_basis, sectors
from phis import jw_basis, Tk
def dvals(N,S):
    d=[1,N-1]
    while len(d)<=S: d.append((N-2)*d[-1]-d[-2])
    return d
for N,S in [(5,5),(6,5),(7,4),(8,4)]:
    lam={t:lam_basis(N,t).shape[1] for t in range(1,S+1)}; d=dvals(N,S)
    pred=[sum(lam[t] for t in range(2,s+1))+(N-1)*(s%2) for s in range(2,S+1)]
    print("N=%d: lambda_t = %s ; d_s = %s ; sum_{t=2}^s lambda_t + [s odd](N-1) = %s"%(N,[lam[t] for t in range(1,S+1)],d[2:S+1],pred))
# f_t for a Dita matrix: eigenvalue 1 in Psi_t, vs e_s
rng=np.random.default_rng(31); q=[m.E(x) for x in rng.random(5)]; H=m.F85b(*q); N=8
for t in [2,3,4]:
    T=Tk(H,t) if t<4 else None
    if t==4:
        import ck
        Q=lam_basis(N,4).astype(complex); TQ=ck.T_block_blas(H,4)(np.ascontiguousarray(Q)); M=Q.conj().T@TQ
    else:
        Q=lam_basis(N,t); M=Q.T@T@Q
    ev=np.linalg.eigvalsh((M+M.conj().T)/2); print("F85b: f_%d = %d   (top below 1: %.4f)"%(t,(ev>1-1e-9).sum(), ev[ev<1-1e-9].max()))
