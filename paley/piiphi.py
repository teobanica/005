# 4x4 matrix of Phi on the patterns P1=[x!=y,a=b], P2=[x!=y,a!=b], P3=[x=y,a!=b], P4=[x=y,a=b]; check against numerics
import numpy as np, sys
sys.path.insert(0,'/home/claude/kar/day6/bobcat')
from pmatrices import paley2
from fractions import Fraction as Fr
def M(q):
    N=2*(q+1)
    P4=[N,N,N,N]; P3=[2*(q-3),-2*(q-3),-2*(q+1),2*(q+1)]
    D=[8,-4*(q-1),4*(q-1)**2-4*(q+1),0]
    J=[0,0,0,N*N]
    S=[J[k]-P3[k]-P4[k] for k in range(4)]
    P1=[(S[k]+D[k])/2 for k in range(4)]; P2=[(S[k]-D[k])/2 for k in range(4)]
    return np.array([P1,P2,P3,P4],dtype=float).T
for args in [(5,),(3,2),(13,),(17,)]:
    H=np.array(paley2(*args)).real; N=H.shape[0]; n=N//2; q=n-1
    x=np.arange(N)//2; a=np.arange(N)%2
    pats=[(x[:,None]!=x[None,:])&(a[:,None]==a[None,:]),(x[:,None]!=x[None,:])&(a[:,None]!=a[None,:]),
          (x[:,None]==x[None,:])&(a[:,None]!=a[None,:]),(x[:,None]==x[None,:])&(a[:,None]==a[None,:])]
    def Phi(Y):
        # Phi(Y)_{uv} = sum_ij H_iu H_iv H_ju H_jv Y_ij
        R=(H[:,:,None]*H[:,None,:])           # i,u,v
        return np.einsum('iuv,ij,juv->uv',R,Y,R)
    ok=True
    Mq=M(q)
    for k,P in enumerate(pats):
        Z=Phi(P.astype(float)); pred=sum(Mq[l,k]*pats[l] for l in range(4))
        ok&=np.allclose(Z,pred)
    M2=Mq@Mq; X=np.array([1,-1,0,0.])
    beta=N**3-32*(q-1)*(2*q+1)
    print("q=%d Phi-matrix verified: %s ; M^2 X = %s, beta X = %s ; eig(M)="%(q,ok,M2@X,beta*X), np.round(np.sort(np.linalg.eigvals(Mq).real),3))
