import numpy as np, sys
sys.path.insert(0,'/home/claude/kar/day6/bobcat')
from pmatrices import paley2
from sectors import Aop
for args in [(3,2),(17,)]:
    H=np.array(paley2(*args)).real.astype(int); N=H.shape[0]; m=N//2; q=m-1
    A=Aop(H); w,V=np.linalg.eigh(A)
    E=V[:,np.abs(w-64*q)<1e-6]
    d=E.shape[1]
    T=E.reshape(m,2,m,2,d).transpose(1,3,0,2,4).reshape(4,-1)   # (a,b) x rest
    s=np.linalg.svd(T,compute_uv=False)
    U,_,_=np.linalg.svd(T); 
    print("q=%d dim=%d (q+1)^2=%d ; (a,b)-pattern singular values:"%(q,d,m*m), np.round(s,3))
    print(" dominant (a,b) patterns:\n", np.round(U[:,:np.sum(s>1e-6)].T.reshape(-1,2,2),3))
    # x,y part: for pattern space, rank in (x,y)
    Txy=E.reshape(m,2,m,2,d).transpose(0,2,1,3,4).reshape(m*m,-1)
    print(" (x,y)-rank:", np.linalg.matrix_rank(Txy,1e-8))
