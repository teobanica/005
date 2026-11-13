import numpy as np, sys
sys.path.insert(0,'/home/claude/kar/day6/bobcat')
from pmatrices import paley2
from sectors import Aop
for args in [(5,),(3,2),(13,)]:
    H=np.array(paley2(*args)).real.astype(int); N=H.shape[0]; m=N//2; q=m-1
    A=Aop(H)
    a=np.arange(N)%2
    par=((a[:,None]+a[None,:])%2).ravel()
    for p in (0,1):
        idx=np.where(par==p)[0]; B=A[np.ix_(idx,idx)]
        w=np.linalg.eigvalsh(B)
        print("q=%d parity %d block: dim %d, mult of 64q: %d"%(q,p,len(idx),np.sum(np.abs(w-64*q)<1e-6)))
    C=A[np.ix_(np.where(par==0)[0],np.where(par==1)[0])]
    print("   cross block max entry", C.max(), " distinct entries", np.unique(C)[:10])
    o=np.where(par==1)[0]; B=A[np.ix_(o,o)]; print("   odd block distinct entries", np.unique(B)[:12], len(np.unique(B)))
