# Borel sectors of A = N^3 T_2 for Paley I, q prime = 3 mod 4
import numpy as np, sys
from eigcheck import paley1
from trace_fast import isprime
def Aop(H):
    N=H.shape[0]; R=(H[:,None,:]*H[None,:,:]).reshape(N*N,N).astype(float); S=R@R.T; return S*S
if __name__=="__main__":
    for q in map(int,sys.argv[1:]):
        H=paley1(q); N=q+1; A=Aop(H)
        sq=set((x*x)%q for x in range(1,q))
        # orbit indicators of B on V x V, V = {inf}=0, 1+x
        lab=np.zeros((N,N),int)
        for i in range(N):
            for j in range(N):
                if i==0 and j==0: l=0
                elif i==0: l=1
                elif j==0: l=2
                elif i==j: l=3
                else: l=4 if ((j-i)%q) in sq else 5
                lab[i,j]=l
        Bm=np.array([(lab.ravel()==l).astype(float) for l in range(6)]).T
        Bm/=np.linalg.norm(Bm,axis=0)
        M=Bm.T@A@Bm
        ev=np.sort(np.linalg.eigvalsh(M))
        print("q=%d N=%d trivial sector eigenvalues:"%(q,N), np.round(ev,4), " /N^2:", np.round(ev/N**2,4))
    