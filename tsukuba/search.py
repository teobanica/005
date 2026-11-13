from lattice import *
import itertools, sys
for M,N in [(2,2),(2,3),(3,2),(2,4),(4,2),(2,5),(5,2),(3,3)]:
    var,gens,blocks,A,_=setup(M,N); n=len(var); hits=[]
    for v in itertools.product(range(-3,4),repeat=n):
        if not any(v): continue
        if np.gcd.reduce(np.abs(v))!=1: continue
        d,_,_=nq_dim(M,N,[list(v)])
        if d>0: hits.append(v)
    print((M,N),"single relations Q^v=1 giving nonzero N_Q:",len(hits),hits[:12],flush=True)
