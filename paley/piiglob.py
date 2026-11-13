# PSL(2,q)-invariant sector of A for Paley II: 8 orbits (x=y or x!=y) x (a,b)
import numpy as np, sys
from piiinv import paley2p
from trace_fast import isprime
def R8(q):
    H,sq=paley2p(q); N=H.shape[0]
    idx=np.arange(N); x=idx//2; a=idx%2
    lab=((x[:,None]!=x[None,:]).astype(int)*4 + a[:,None]*2 + a[None,:]).ravel()
    Rw=(H[:,None,:]*H[None,:,:]).reshape(N*N,N).astype(np.int64)
    R=np.zeros((8,8),np.int64)
    for l in range(8):
        k=int(np.where(lab==l)[0][0]) if l<4 else int(np.where((lab==l))[0][0])
        # choose representative with x1=0 (infinity) side? check invariance by using 2 reps
        Arow=(Rw@Rw[k])**2
        R[l]=np.bincount(lab,weights=Arow,minlength=8).astype(np.int64)
        k2=int(np.where(lab==l)[0][-1]); Arow2=(Rw@Rw[k2])**2
        assert (np.bincount(lab,weights=Arow2,minlength=8)==R[l]).all(), "not invariant"
    return R
if __name__=="__main__":
    for q in [q for q in range(5,int(sys.argv[1])) if q%4==1 and isprime(q)]:
        R=R8(q); N=2*(q+1)
        print("q=%d"%q, R.tolist(), np.round(np.sort(np.linalg.eigvals(R).real),2), flush=True)
