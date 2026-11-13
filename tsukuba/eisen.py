from table6 import index
from math import gcd
from itertools import combinations
def mul(z,w):  # (x+yω)(u+vω), ω^2=-1-ω
    x,y=z; u,v=w; return (x*u-y*v, x*v+y*u-y*v)
def lattice_index(vecs):
    g=0
    for v,w in combinations(vecs,2): g=gcd(g,abs(v[0]*w[1]-v[1]*w[0]))
    return g
def norm_gcd(L,w):
    vecs=[(L,0),(0,L),w,mul(w,(0,1))]
    return lattice_index(vecs)
bad=0; n=0
for L in range(1,25):
    for a in range(L):
        for b in range(L):
            w=mul((2,0),mul((1,-1),(a,b)))   # 2(1-ω)(a+bω)
            pred=L*L//norm_gcd(L,w)
            for MN in [(2,3),(3,2)]:
                n+=1
                if index(*MN,a,b,L)!=pred: bad+=1
print("checked",n,"mismatches",bad)
