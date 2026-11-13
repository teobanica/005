# Tr((N T_2)^r) = N^{-2r} sum over spectrum of A=S∘S ... exact: Tr((N^3T_2)^r) integer; Tr((NT_2)^r)=Tr(A^r)/N^{2r}
import numpy as np, sys
from eigcheck import paley1
from fractions import Fraction as F
for q in map(int,sys.argv[1:]):
    H=paley1(q); N=q+1
    R=(H[:,None,:]*H[None,:,:]).reshape(N*N,N).astype(np.int64)
    S=R@R.T; A=S*S
    out=[]
    Ar=np.eye(N*N,dtype=object) if False else None
    t2=int((A.astype(object)**2).sum()) if N<=24 else int(sum(int(x) for x in (A.astype(np.float64)**2).sum(axis=1)))
    tr2=F(int((A*A).sum()) ,N**4)   # Tr(A^2)=sum A_ij^2 (A symmetric)
    print(q,N,"Tr((NT2)^2) =",tr2, "integer" if tr2.denominator==1 else "NOT integer", flush=True)
