"""M=2 or N=2, cyclic n = the other size. N_Q via group ring Z[Z_n]."""
import numpy as np, itertools
from nongen import shifts, scalar_blocks
E=lambda t: np.exp(2j*np.pi*t)
def theta(M,N,args):          # args: phases (in turns) of Q_{1d} (M=2) or Q_{i1} (N=2), index 1..n-1
    n=max(M,N); th=np.zeros(n)
    for d in range(1,n):
        if M==2: th[d]=2*args[d-1]
        else:    th[(-d)%n]=2*args[d-1]
    return th
def conv(R,th):
    n=len(th); return np.array([sum(R[c]*th[(e-c)%n] for c in range(n)) for e in range(n)])
def in_NQ_formula(M,N,args,R):   # R: exponents of t_1..t_{n-1}
    n=max(M,N); Rt=np.zeros(n); Rt[1:]=R; Rt[0]=-sum(R)
    v=conv(Rt,theta(M,N,args)); d=(v-v[0]+0.5)%1-0.5
    return np.allclose(d,0,atol=1e-9)
def in_NQ_brute(M,N,args,R):
    n=max(M,N)
    Q=[[1]+[E(x) for x in args]]
    Q=[[1]*N]+Q if M==2 else [[1,1]]+[[1,E(x)] for x in args]
    if M==2: Q=[[1]*N,[1]+[E(x) for x in args]]
    P=shifts(M,N,Q); A=np.eye(M*N,dtype=complex)
    for c,r in enumerate(R,start=1):
        word=[(1,c),(0,(-c)%N)] if M==2 else [(c,1),(0,1)]
        B=np.eye(M*N,dtype=complex)
        for l in word: B=P[l]@B
        A=np.linalg.matrix_power(B if r>=0 else np.linalg.inv(B),abs(r))@A
    return scalar_blocks(A,M,N) is not None
if __name__=="__main__":
    rng=np.random.default_rng(0); bad=0; tot=0
    for (M,N) in [(2,3),(3,2),(2,4),(4,2),(2,5),(5,2),(2,6),(6,2)]:
        n=max(M,N)
        for trial in range(6):
            L=int(rng.choice([2,3,4,6,8,12]))
            args=[int(rng.integers(L))/L for _ in range(n-1)]
            for R in itertools.product(range(-2,3),repeat=n-1):
                tot+=1
                if in_NQ_formula(M,N,args,R)!=in_NQ_brute(M,N,args,R): bad+=1
    print("tested",tot,"mismatches",bad)
