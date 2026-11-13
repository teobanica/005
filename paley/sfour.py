# exact sum of S^4 over all 4-tuples of rows, for Paley I, q prime = 3 mod 4, using translation invariance
import numpy as np, sys
from eigcheck import paley1
from trace_fast import isprime
def sumS4(q):
    H=paley1(q).astype(np.float64); N=q+1
    R=(H[:,None,:]*H[None,:,:]).reshape(N*N,N)
    tot=0
    for i1,w in ((0,1),(1,q)):
        U=H[i1][None,:]*H          # N x N, rows i2
        S=U@R.T                    # N x N^2 : S(i1,i2,(j1,j2))
        Si=np.rint(S).astype(np.int64)
        tot+=w*int(((Si**2)**2).sum(dtype=np.int64)) if N<300 else None
    return tot
if __name__=="__main__":
    lo,hi=int(sys.argv[1]),int(sys.argv[2])
    for q in range(lo,hi):
        if q%4==3 and isprime(q):
            N=q+1; s=sumS4(q)
            from fractions import Fraction as F
            print(q, s, F(s,N**4), flush=True)
