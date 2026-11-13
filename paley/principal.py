# test the translation-invariant vectors X_lambda(x,y) = lambda(sq(y-x)) on F_q (zero on the infinity row/column)
import sys, numpy as np
sys.path.insert(0,'.'); from certify import A_matrix
sys.path.insert(0,'/home/claude/v21/code-29/paper'); import matrices as m
for q in [int(a) for a in sys.argv[1:]]:
    H=np.real(m.paley1(q)).astype(int); N=q+1; A=A_matrix(H).astype(float)
    sq=[x*x%q for x in range(1,q)]; S=sorted(set(sq))
    # generator of squares: g^2 with g primitive root
    g=next(a for a in range(2,q) if len({pow(a,k,q) for k in range(1,q)})==q-1); h=g*g%q; n=(q-1)//2
    log={pow(h,k,q):k for k in range(n)}
    isq=lambda d: d in log
    out=[]
    for t in range(n):
        lam=lambda s: np.exp(2j*np.pi*t*log[s]/n)
        X=np.zeros((N,N),complex)
        for x in range(q):
            for y in range(q):
                if x!=y:
                    d=(y-x)%q; s=d if isq(d) else (-d)%q
                    X[1+x,1+y]=lam(s)
        v=X.ravel(); Av=A@v; mu=(v.conj()@Av)/(v.conj()@v); res=np.linalg.norm(Av-mu*v)/np.linalg.norm(v)
        out.append((t,round(mu.real,4),round(res,6)))
    print("q",q,"target 32(q-1) =",32*(q-1),":",out,flush=True)
