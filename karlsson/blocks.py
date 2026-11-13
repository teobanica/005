import numpy as np, sys
from h2red import *
def to_blocks(H, rm, cm):
    q=len(rm); B=[[H[np.ix_(rm[i],cm[j])] for j in range(q)] for i in range(q)]
    return B
def normalize(B):
    q=len(B); Y=[B[0][j] for j in range(q)]; X=[B[i][0]@np.linalg.inv(B[0][0]) for i in range(q)]
    W=[[np.linalg.inv(X[i])@B[i][j]@np.linalg.inv(Y[j]) for j in range(q)] for i in range(q)]
    return X,Y,W
def analyze(H):
    out=[]
    for rm,cm in h2_structures(H):
        X,Y,W=normalize(to_blocks(H,rm,cm)); q=len(W)
        flat=[W[i][j] for i in range(1,q) for j in range(1,q)]
        comm=max(np.abs(a@b-b@a).max() for a in flat for b in flat)
        unit=max(np.abs(w@w.conj().T-np.eye(2)).max() for w in flat)
        Xu=max(np.abs(x@x.conj().T-np.eye(2)).max() for x in X)
        out.append((comm,unit,Xu))
    return out
if __name__=="__main__":
    rng=np.random.default_rng(0); r=lambda: np.exp(2j*np.pi*rng.random())
    tests={'H6':M.H6(r()),'BF6':M.BF6(0),'BN6':M.BN6(np.exp(2j*np.pi*0.37)),'K6':M.K6(0.7,1.1,0.4),
           'F84':M.F84(r(),r(),r(),r()),'F82':M.F82(r(),r()),'K82':M.K82(r(),r()),'M81':M.M81(r())}
    for k,H in tests.items():
        for HH,t in ((H,''),(H.T,'t')):
            a=analyze(HH); print(k+t, len(a), 'commuting structures:', sum(c<1e-9 for c,_,_ in a), 'max unit err', max(u for _,u,_ in a), max(x for *_,x in a))
