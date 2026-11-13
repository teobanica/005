from lattice import *
import sys
for (M,N) in [(2,4),(4,2),(2,3),(3,2)]:
    var,gens,blocks,A,Dfull=setup(M,N); n=len(gens); low={}
    for R in itertools.product(range(-3,4),repeat=n):
        R=np.array(R)
        if not R.any() or np.gcd.reduce(np.abs(R))!=1: continue
        S=np.array([Dfull[(k,e)]@R-Dfull[(k,0)]@R for k in range(M) for e in range(N)])
        r=np.linalg.matrix_rank(S)
        if r<len(var): low.setdefault(r,[]).append(tuple(R))
    print((M,N),{r:v[:6] for r,v in low.items()})
