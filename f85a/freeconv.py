from fractions import Fraction as Fr
from math import comb
import itertools
def nc_partitions(n):
    # generate noncrossing partitions of range(n) as lists of blocks
    def rec(elems):
        if not elems: yield []; return
        first=elems[0]; rest=elems[1:]
        for k in range(len(rest)+1):
            for sub in itertools.combinations(rest,k):
                block=[first]+list(sub)
                # noncrossing: elements between consecutive block elems form separate groups
                gaps=[]; prev=first; ok=True
                pts=[first]+list(sub)
                segs=[]
                for a,b in zip(pts,pts[1:]):
                    segs.append([x for x in rest if a<x<b and x not in sub])
                segs.append([x for x in rest if x>pts[-1] and x not in sub])
                def prod(i):
                    if i==len(segs): yield []; return
                    for p in rec(segs[i]):
                        for q in prod(i+1): yield p+q
                for q in prod(0): yield [block]+q
    yield from rec(list(range(n)))
def moments_from_cum(kap,n):
    return [sum( eval('*'.join(str(kap[len(b)]) for b in P) or '1') for P in nc_partitions(k)) for k in range(1,n+1)]
def cum_from_mom(m,n):
    kap={}
    for k in range(1,n+1):
        s=sum(eval('*'.join(str(kap[len(b)]) for b in P)) for P in nc_partitions(k) if len(P)>1)
        kap[k]=Fr(m[k-1])-s
    return kap
n=6
m=[Fr(comb(2*k,k),2) for k in range(1,n+1)]
kap=cum_from_mom(m,n); kap2={k:2*v for k,v in kap.items()}
mm=moments_from_cum({k:Fr(v) for k,v in kap2.items()},n)
print("H2+ wr Z2 :", [x/2 for x in mm])
m4=[Fr(comb(2*k,k)//(k+1)) for k in range(1,n+1)]  # S4+ Catalan check
k4=cum_from_mom(m4,n); print("S4+ wr Z2:",[x/2 for x in moments_from_cum({k:2*v for k,v in k4.items()},n)])
