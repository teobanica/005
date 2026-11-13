"""General finite abelian X, Y (products of cyclic groups). Brute-force scalar test vs group-ring formula."""
import numpy as np, itertools
E=lambda t: np.exp(2j*np.pi*t)
class Ab:
    def __init__(s,ords):
        s.ords=tuple(ords); s.els=list(itertools.product(*[range(o) for o in ords])); s.n=len(s.els)
        s.idx={g:i for i,g in enumerate(s.els)}
    def add(s,a,b): return tuple((x+y)%o for x,y,o in zip(a,b,s.ords))
    def neg(s,a): return tuple((-x)%o for x,o in zip(a,s.ords))
    def sub(s,a,b): return s.add(a,s.neg(b))
def shifts(X,Y,Q):   # Q dict (j,d)->complex, keys group elements
    M,N=X.n,Y.n; P={}
    for i in X.els:
        for c in Y.els:
            A=np.zeros((M*N,M*N),complex)
            for k in X.els:
                for e in Y.els:
                    ec=Y.sub(e,c); ik=X.sub(i,k)
                    th=Q[(i,ec)]*Q[(ik,e)]/(Q[(i,e)]*Q[(ik,ec)])
                    A[X.idx[k]*N+Y.idx[ec], X.idx[k]*N+Y.idx[e]]=th
            P[(i,c)]=A
    return P
def scalar_blocks(A,M,N,tol=1e-9):
    for k in range(M):
        B=A[k*N:(k+1)*N,k*N:(k+1)*N]
        if not np.allclose(B,B[0,0]*np.eye(N),atol=tol): return False
    return True
def brute(X,Y,Q,R):
    M,N=X.n,Y.n; P=shifts(X,Y,Q); z=X.els[0]
    for x in X.els:
        A=np.eye(M*N,dtype=complex)
        for (i,c),v in R.items():
            B=np.eye(M*N,dtype=complex)
            for (ii,cc) in [(i,c),(z,Y.neg(c))]: B=P[(X.add(ii,x),cc)]@B
            A=np.linalg.matrix_power(B if v>=0 else np.linalg.inv(B),abs(v))@A
        if not scalar_blocks(A,M,N): return False
    return True
def formula(X,Y,th,R):   # th dict (j,d)->turns, theta(j,d)=arg Q_{-j,d}
    # r = sum R_{ic} (x^i-1)(y^c-1) as dict over X x Y
    r={}
    z0,w0=X.els[0],Y.els[0]
    for (i,c),v in R.items():
        for (a,b,s) in [(i,c,1),(i,w0,-1),(z0,c,-1),(z0,w0,1)]:
            r[(a,b)]=r.get((a,b),0)+s*v
    F={}
    for j in X.els:
        for d in Y.els:
            F[(j,d)]=sum(v*th[(X.sub(j,a),Y.sub(d,b))] for (a,b),v in r.items())
    for j in X.els:
        for d in Y.els:
            g=F[(j,d)]-F[(j,w0)]-F[(z0,d)]+F[(z0,w0)]
            if abs(((g+0.5)%1)-0.5)>1e-9: return False
    return True
if __name__=="__main__":
    rng=np.random.default_rng(2); tot=bad=pos=0
    for (xo,yo) in [((2,),(2,2)),((2,2),(2,)),((2,2),(2,2)),((2,),(2,4)),((3,),(2,2)),((2,2),(3,)),((2,),(2,2,2))]:
        X,Y=Ab(xo),Ab(yo)
        gens=[(i,c) for i in X.els[1:] for c in Y.els[1:]]
        for L in (2,3,4,8):
            for trial in range(2):
                phase={(j,d):(0 if (j==X.els[0] or d==Y.els[0]) else int(rng.integers(L))/L) for j in X.els for d in Y.els}
                Q={k:E(v) for k,v in phase.items()}
                th={(j,d):phase[(X.neg(j),d)] for j in X.els for d in Y.els}
                for _ in range(20):
                    R={g:int(rng.integers(-2,3)) for g in gens}
                    if not any(R.values()): continue
                    b=brute(X,Y,Q,R); f=formula(X,Y,th,R); tot+=1; pos+=b; bad+=(b!=f)
        print(xo,yo,"so far tested",tot,"positives",pos,"mismatches",bad,flush=True)
