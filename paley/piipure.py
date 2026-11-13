import numpy as np, sys
sys.path.insert(0,'/home/claude/kar/day6/bobcat')
from pmatrices import paley2
from sectors import Aop
def blocks(H):
    N=H.shape[0]; A=Aop(H); a=np.arange(N)%2; par=((a[:,None]+a[None,:])%2).ravel()
    e=np.where(par==0)[0]; o=np.where(par==1)[0]
    return A,A[np.ix_(e,e)],A[np.ix_(e,o)],A[np.ix_(o,o)],e,o
if __name__=="__main__":
 for args in [(5,),(3,2),(13,),(17,)]:
     H=np.array(paley2(*args)).real.astype(int); N=H.shape[0]; q=N//2-1
     A,Aee,Aeo,Aoo,e,o=blocks(H)
     for name,M,X in (("odd",Aoo,Aeo),("even",Aee,Aeo.T)):
         K=np.vstack([M-64*q*np.eye(len(M)),X])
         s=np.linalg.svd(K,compute_uv=False)
         print("q=%d pure-%s eigenvectors for 64q: %d"%(q,name,np.sum(s<1e-6*s[0])), flush=True)
     w=np.linalg.eigvalsh(Aoo); vals=sorted(set(np.round(w,4)))
     print("   spectrum of odd block:", vals[:12])
