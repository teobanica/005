"""General cyclic X=Z_M, Y=Z_N: T = I_X (x) I_Y via t_{ic} -> (x^i-1)(y^c-1);
conjecture: N_Q = { r : r*theta is of the form a(j)+b(e) mod 1 }, theta_{jd} = arg Q_{jd}/2pi."""
import numpy as np, itertools
from nongen import shifts, scalar_blocks
E=lambda t: np.exp(2j*np.pi*t)
def rvec(M,N,R):   # R dict (i,c)->int
    r=np.zeros((M,N))
    for (i,c),v in R.items():
        r[i,c]+=v; r[i,0]-=v; r[0,c]-=v; r[0,0]+=v
    return r
def conv2(r,th):
    M,N=th.shape; F=np.zeros((M,N))
    for i in range(M):
        for c in range(N):
            if r[i,c]: F+=r[i,c]*np.roll(np.roll(th,i,axis=0),c,axis=1)
    return F
def formula(M,N,th,R):
    F=conv2(rvec(M,N,R),th)
    G=F-F[:,[0]]-F[[0],:]+F[0,0]
    return np.allclose((G+0.5)%1-0.5,0,atol=1e-9)
def brute(M,N,th,R):
    Q=[[E(th[j,d]) for d in range(N)] for j in range(M)]
    P=shifts(M,N,Q)
    for x in range(M):
        A=np.eye(M*N,dtype=complex)
        for (i,c),v in R.items():
            B=np.eye(M*N,dtype=complex)
            for (ii,cc) in [(i,c),(0,(-c)%N)]: B=P[((ii+x)%M,cc)]@B
            A=np.linalg.matrix_power(B if v>=0 else np.linalg.inv(B),abs(v))@A
        if scalar_blocks(A,M,N) is None: return False
    return True
if __name__=="__main__":
    rng=np.random.default_rng(1); bad=tot=0; pos=0
    for (M,N) in [(3,3),(2,4),(4,2),(3,4),(4,3),(2,3),(3,2)]:
        gens=[(i,c) for i in range(1,M) for c in range(1,N)]
        for trial in range(5):
            L=int(rng.choice([2,3,4,6,9,12]))
            th=np.zeros((M,N)); th[1:,1:]=rng.integers(L,size=(M-1,N-1))/L
            for _ in range(250):
                R={g:int(rng.integers(-2,3)) for g in gens}
                if not any(R.values()): continue
                tot+=1; f=formula(M,N,th,R); b=brute(M,N,th,R); pos+=b
                if f!=b: bad+=1
    print("tested",tot,"in N_Q:",pos,"mismatches",bad)
