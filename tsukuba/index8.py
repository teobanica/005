import numpy as np, itertools
from unified import *
from nongen import counts
def ring_mul(u,v,n):
    w=[0]*n
    for i in range(n):
        for j in range(n): w[(i+j)%n]+=u[i]*v[j]
    return w
def subgroup_size(gens,L,n):
    # brute force span in (Z/L)^n
    S={tuple([0]*n)}; frontier=list(S)
    while frontier:
        new=[]
        for s in frontier:
            for g in gens:
                t=tuple((s[i]+g[i])%L for i in range(n))
                if t not in S: S.add(t); new.append(t)
        frontier=new
    return len(S)
def index_formula(M,N,ints,L):     # Q phases = ints/L
    n=max(M,N); Th=[0]*n
    for d in range(1,n):
        if M==2: Th[d]=2*ints[d-1]
        else: Th[(-d)%n]=2*ints[d-1]
    x1=[-1,1]+[0]*(n-2)                 # x-1
    psi=ring_mul(x1,Th,n)
    gens=[ring_mul([(-1 if i==0 else 0)+(1 if i==c else 0) for i in range(n)],psi,n) for c in range(1,n)]
    gens=[[g%L for g in v] for v in gens]
    return subgroup_size(gens,L,n)
def index_brute(M,N,ints,L):
    n=max(M,N); args=[x/L for x in ints]
    # N_Q contains L*Z^{n-1}; count cosets of Z^{n-1}/L in N_Q
    cnt=sum(1 for R in itertools.product(range(L),repeat=n-1) if in_NQ_formula(M,N,args,R))
    return L**(n-1)//cnt
if __name__=="__main__":
    bad=0
    for (M,N) in [(2,4),(4,2),(2,3),(3,2)]:
        n=max(M,N)
        for L in (2,4,8):
            for ints in itertools.product(range(L),repeat=n-1):
                if index_formula(M,N,ints,L)!=index_brute(M,N,ints,L): bad+=1
    print("index formula mismatches:",bad)
