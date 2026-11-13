import sys, numpy as np
sys.path.insert(0, "/home/claude/day")
from phis import jw_basis
def perm_map(N,s,f):
    """e_{i} -> e_{f(i)}, f maps tuple of length s to tuple of length s+1"""
    n=N**s; idx=np.array(np.unravel_index(np.arange(n),(N,)*s)).T
    new=np.array([f(t) for t in idx]); rows=np.ravel_multi_index(new.T,(N,)*(s+1))
    M=np.zeros((N**(s+1),n)); M[rows,np.arange(n)]=1; return M
def maps(N,s):
    out={}
    for i in range(s):                     # double leg i, then rotate by j
        for j in range(s+1):
            def f(t,i=i,j=j):
                d=list(t[:i+1])+[t[i]]+list(t[i+1:]); return tuple(d[j:]+d[:j])
            out[(i,j)]=perm_map(N,s,f)
    return out
rng=np.random.default_rng(0)
for N in [5,6,7]:
    for s in ([2,3,4] if N<=6 else [2,3]):
        Bs,Bt=jw_basis(N,s),jw_basis(N,s+1); Ms=maps(N,s)
        Gs={k:Bt.T@M@Bs for k,M in Ms.items()}
        nz={k:G for k,G in Gs.items() if np.linalg.norm(G)>1e-9}
        ranks={k:int((np.linalg.svd(G,compute_uv=False)>1e-9).sum()) for k,G in nz.items()}
        comb=sum(rng.standard_normal()*G for G in nz.values()); rc=int((np.linalg.svd(comb,compute_uv=False)>1e-9).sum())
        joint=np.vstack(list(nz.values())); rj=int((np.linalg.svd(joint,compute_uv=False)>1e-9).sum())
        print("N=%d s=%d d_s=%d: single ranks %s ; generic combination rank %d ; joint rank %d"%(N,s,Bs.shape[1],sorted(set(ranks.values())),rc,rj))
