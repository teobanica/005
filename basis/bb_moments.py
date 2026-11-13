# Moments of F_X (x)_Q F_Y, Q generic, via the multiset formula of Banica-Bichon (Prop 5.1),
# and check of c2 = M+N-1, c3 = M^2+N^2+3MN-3M-3N+2.  Output also written to bb_moments.out
import itertools
from collections import Counter
def c(M,N,p):
    cnt=0
    for i in itertools.product(range(M),repeat=p):
        for d in itertools.product(range(N),repeat=p):
            cnt+= Counter(zip(i,d))==Counter(zip(i,(d[-1],)+d[:-1]))
    assert cnt%(M*N)==0
    return cnt//(M*N)
lines=[]
for M,N,P in [(2,2,5),(2,3,5),(3,2,5),(2,4,4),(3,3,4),(4,3,3),(4,4,3),(5,2,3)]:
    cs=[c(M,N,p) for p in range(1,P+1)]
    ok=cs[1]==M+N-1 and cs[2]==M*M+N*N+3*M*N-3*M-3*N+2
    lines.append(f"{M}x{N}: {cs}  formulas c2,c3 ok: {ok}")
open('bb_moments.out','w').write("\n".join(lines)+"\n"); print("\n".join(lines))
