import sys, numpy as np, scipy.linalg as sl
sys.path.insert(0, "/home/claude/day")
from annular import sectors
from math import comb
N=6; S=5
def rot(x,k): return np.moveaxis(x.reshape((N,)*k),list(range(k)),[(a+1)%k for a in range(k)]).reshape(-1)
def gens_up(x,k):
    X=x.reshape((N,)*k)
    yield np.multiply.outer(X,np.ones(N)).reshape(-1)                      # append 1
    D=np.zeros((N,)*(k+1),complex); 
    idx=np.indices((N,)*k).reshape(k,-1); D[tuple(idx)+(idx[-1],)]=X.reshape(-1); yield D.reshape(-1)   # double last leg
def gens_down(x,k):
    X=x.reshape((N,)*k)
    if k>=2: yield (k-1, X.sum(axis=-1).reshape(-1))                        # leg sum
    if k>=2: yield (k-1, np.einsum('...aa->...a',X).reshape(-1))            # merge last 2
    if k>=3: yield (k-2, np.einsum('...aa->...',X).reshape(-1))             # cap last 2
def module_dims(v,t):
    spans={t:sl.orth(v[:,None])}
    changed=True
    while changed:
        changed=False
        for k in sorted(list(spans)):
            B=spans[k]; new={}
            for j in range(B.shape[1]):
                x=B[:,j]
                cands=[(k,rot(x,k))]+[(k+1,y) for y in gens_up(x,k) if k<S+1]+list(gens_down(x,k))
                for kk,y in cands:
                    if np.linalg.norm(y)<1e-12: continue
                    new.setdefault(kk,[]).append(y)
            for kk,ys in new.items():
                A=np.array(ys).T if kk not in spans else np.hstack([spans[kk],np.array(ys).T])
                Q=sl.orth(A,rcond=1e-10)
                if kk not in spans or Q.shape[1]>spans[kk].shape[1]:
                    spans[kk]=Q; changed=True
    return {k:spans[k].shape[1] for k in sorted(spans) if k<=S}
v1=np.zeros(N,complex); v1[0]=1; v1[1]=-1
print("t=1:", module_dims(v1,1), " predicted binom(2s-1,s-1):", {s:comb(2*s-1,s-1) for s in range(1,S+1)})
for t in [2,3]:
    for w,Q in sectors(N,t).items():
        v=Q[:,0]; print("t=%d, sector %d:"%(t,w), module_dims(v,t), " predicted binom(2s,s-t):", {s:comb(2*s,s-t) for s in range(t,S+1)}); break
