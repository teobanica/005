# Paley II (any symmetric conference matrix C): X((x,a),(y,b)) = (1-delta_xy)(-1)^{a+b} has N^3 T_2 X = beta X,
# beta = N^3 - 32(q-1)(2q+1), N=2(q+1). Also Montgomery's vector S (x) D with eigenvalue 64q.
import numpy as np, sys
sys.path.insert(0,'/home/claude/kar/day6/bobcat')
from pmatrices import paley2
from sectors import Aop
from fractions import Fraction as Fr
for args in [(5,),(3,2),(13,),(17,),(5,2),(29,)]:
    H=np.array(paley2(*args)).real.astype(int); N=H.shape[0]; n=N//2; q=n-1
    A=Aop(H)
    x=np.arange(N)//2; a=np.arange(N)%2
    X=((x[:,None]!=x[None,:])*(-1.0)**(a[:,None]+a[None,:])).ravel()
    beta=N**3-32*(q-1)*(2*q+1)
    r1=np.linalg.norm(A@X-beta*X)/np.linalg.norm(X)
    C=H.reshape(n,2,n,2)[:,0,:,0]-np.eye(n)
    Sm=C.copy(); Sm[0,:]=0; Sm[:,0]=0
    D=np.diag([1,-1])
    M=np.kron(Sm,D).ravel()
    r2=np.linalg.norm(A@M-64*q*M)/np.linalg.norm(M)
    print("q=%d N=%d: beta=%d residual %.1e ; N mu = beta/N^2 = %s ; Montgomery 64q residual %.1e"%(q,N,beta,r1,Fr(beta,N*N),r2))
