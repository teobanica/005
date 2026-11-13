from lattice import *
from fractions import Fraction
M,N=2,4; v=np.array([1,-1,1])
var,gens,blocks,A,Dfull=setup(M,N)
print("vars",var,"gens",gens)
for R in itertools.product(range(-4,5),repeat=3):
    R=np.array(R)
    if not R.any(): continue
    ok=True; vals={}
    for k in range(M):
        base=Dfull[(k,0)]@R
        for e in range(N):
            diff=Dfull[(k,e)]@R-base
            # diff must be multiple of v
            t=diff@v/(v@v)
            if not np.allclose(diff,t*v) or abs(t-round(t))>1e-9: ok=False
        vals[k]=base
    if ok: print("R",R,"block exponents (q1,q2,q3):",{k:list(vals[k]) for k in vals}); 
