"""Non-generic Dita deformations F_X (x)_Q F_Y, X=Z_M, Y=Z_N cyclic.
Model of C*(Gamma)xC(X): pi(c^(i)) eps_{ke} = theta^{ke}_{ic} eps_{k,e-c}.
N = elements acting by scalars on all blocks, together with all X-translates.
c_k = (1/M) #{words of length k in N}."""
import numpy as np, itertools, sys
sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw')
def shifts(M,N,Q):
    P={}
    for i in range(M):
        for c in range(N):
            A=np.zeros((M*N,M*N),complex)
            for k in range(M):
                for e in range(N):
                    th=Q[i%M][(e-c)%N]*Q[(i-k)%M][e%N]/(Q[i%M][e%N]*Q[(i-k)%M][(e-c)%N])
                    A[k*N+(e-c)%N, k*N+e]=th
            P[(i,c)]=A
    return P
def scalar_blocks(A,M,N,tol=1e-9):
    vals=[]
    for k in range(M):
        B=A[k*N:(k+1)*N,k*N:(k+1)*N]
        if not np.allclose(B,B[0,0]*np.eye(N),atol=tol): return None
        vals.append(B[0,0])
    return vals
def counts(M,N,Q,kmax):
    P=shifts(M,N,Q); letters=list(P.keys()); out=[]
    for k in range(1,kmax+1):
        cnt=0
        for w in itertools.product(letters,repeat=k):
            if sum(c for _,c in w)%N: continue
            ok=True
            for x in range(M):
                A=np.eye(M*N,dtype=complex)
                for (i,c) in w: A=P[((i+x)%M,c)]@A
                if scalar_blocks(A,M,N) is None: ok=False; break
            cnt+=ok
        out.append(cnt//M if cnt%M==0 else cnt/M)
    return out
def dita_Q(M,N,Q):
    import matrices as Mx
    return Mx.dita(Mx.fourier(M),Mx.fourier(N),Q)
if __name__=="__main__":
    from magic import magic, ck_fast
    E=lambda t: np.exp(2j*np.pi*t)
    tests=[("2x3 generic",2,3,[[1,1,1],[1,E(.1234),E(.3711)]]),
           ("2x3 p=w,q=w (w=e(1/3))",2,3,[[1,1,1],[1,E(1/3),E(1/3)]]),
           ("2x3 p=i,q=1",2,3,[[1,1,1],[1,1j,1]]),
           ("2x3 p=e(1/5),q=e(2/5)",2,3,[[1,1,1],[1,E(1/5),E(2/5)]]),
           ("3x2 generic",3,2,[[1,1],[1,E(.1234)],[1,E(.3711)]]),
           ("3x2 p=i,q=1",3,2,[[1,1],[1,1j],[1,1]]),
           ("3x2 p=e(1/6),q=e(1/4)",3,2,[[1,1],[1,E(1/6)],[1,E(1/4)]]),
           ("3x2 p=e(1/8),q=e(3/8)",3,2,[[1,1],[1,E(1/8)],[1,E(3/8)]])]
    for nm,M,N,Q in tests:
        H=dita_Q(M,N,Q)
        print(nm,"count:",counts(M,N,Q,4),"model:",[ck_fast(magic(H),k) for k in range(1,5)],flush=True)
