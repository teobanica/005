import sys, numpy as np, scipy.linalg as sl
sys.path.insert(0, "/home/claude/day")
from annular import lam_basis
from phis import jw_basis
def set_partitions(s):
    if s==0: yield []; return
    for p in set_partitions(s-1):
        for i in range(len(p)): yield p[:i]+[p[i]+[s-1]]+p[i+1:]
        yield p+[[s-1]]
def fix_SN(N,s):
    n=N**s; idx=np.array(np.unravel_index(np.arange(n),(N,)*s)).T; vecs=[]
    for p in set_partitions(s):
        lab=np.zeros(s,int)
        for b,bl in enumerate(p): lab[bl]=b
        # xi_pi: indicator of words constant on blocks (ker >= pi)
        ok=np.ones(n,bool)
        for bl in p:
            for a in bl[1:]: ok&=idx[:,a]==idx[:,bl[0]]
        vecs.append(ok.astype(float))
    return sl.orth(np.array(vecs).T)
def inter_dim(A,B):
    return A.shape[1]+B.shape[1]-np.linalg.matrix_rank(np.hstack([A,B]),tol=1e-8)
N=6
for s in [2,3,4]:
    F=fix_SN(N,s); e=inter_dim(F,jw_basis(N,s)); f=inter_dim(F,lam_basis(N,s))
    print("S_6, level %d: c_s = %d, e_s = %d, f_s = %d"%(s,F.shape[1],e,f))
