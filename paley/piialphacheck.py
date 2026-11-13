# alpha = 8T^2+32 on S(x)I and S(x)sigma_x for Paley II, T = sum chi(s(s-1)t(t-1)(t-s)) over F_q
import numpy as np, sys
sys.path.insert(0,'/home/claude/kar/day6/bobcat')
from pmatrices import paley2
from sectors import Aop
for args in [(5,),(3,2),(13,),(5,2),(29,)]:
    H=np.array(paley2(*args)).real.astype(int); N=H.shape[0]; n=N//2; q=n-1
    A=Aop(H)
    C=H.reshape(n,2,n,2)[:,0,:,0]-np.eye(n); S=C.copy(); S[0,:]=0; S[:,0]=0
    K=S[1:,1:]   # Jacobsthal chi(y-x) in the field labeling used by paley2
    # T via the matrix identity T*chi(z'-z) = g^T S g, with z,z' = field elements 0,1 -> indices 1,2 of C
    g=C[1:,1]*C[1:,2]          # chi(y-z)chi(y-z') over finite y (C_{y z})
    T=int(g@K@g)*int(C[1,2])
    alpha=8*T*T+32
    res=[]
    for E in (np.eye(2),np.array([[0,1],[1,0]])):
        X=np.kron(S,E).ravel(); res.append(np.linalg.norm(A@X-alpha*X)/np.linalg.norm(X))
    print("q=%d T=%d alpha=8T^2+32=%d residuals"%(q,T,alpha),["%.1e"%r for r in res])
