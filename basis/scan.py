import sys, json, time, itertools, numpy as np, matrices as m, ck
from fractions import Fraction
E=m.E; rng=np.random.default_rng(2026)
def cdense(H,k):
    T=ck.T_matrix(H,k); w=np.linalg.eigvalsh((T+T.conj().T)/2); return int((w>1-1e-7).sum())
def roots(nmax):
    return sorted({Fraction(j,n) for n in range(1,nmax+1) for j in range(n)})
R24=[float(f) for f in roots(24)]; R12=[float(f) for f in roots(12)]; R8=[float(f) for f in roots(8)]
fams={
 'H6':  (lambda t: m.H6(E(t[0])), [ (x,) for x in list(np.linspace(0,1,721)[:-1])+R24 ]),
 'BN6': (lambda t: m.BN6(E(t[0])), [ (x,) for x in list(np.linspace(0,1,721)[:-1])+R24 ]),
 'P7':  (lambda t: m.P7(E(t[0])), [ (x,) for x in list(np.linspace(0,1,721)[:-1])+R24 ]),
 'R7':  (lambda t: m.R7(E(t[0])), [ (x,) for x in list(np.linspace(0,1,721)[:-1])+R24 ]),
 'M81': (lambda t: m.M81(E(t[0])), [ (x,) for x in list(np.linspace(0,1,721)[:-1])+R24 ]),
 'F82': (lambda t: m.F82(E(t[0]),E(t[1])), list(itertools.product(R12,R12))+[tuple(rng.random(2)) for _ in range(800)]),
 'K82': (lambda t: m.K82(E(t[0]),E(t[1])), list(itertools.product(R12,R12))+[tuple(rng.random(2)) for _ in range(800)]),
 'F84': (lambda t: m.F84(*[E(x) for x in t]), list(itertools.product(R8[:8],repeat=4))[::3]+[tuple(rng.random(4)) for _ in range(600)]),
 'K6':  (lambda t: m.K6(*[np.pi*x for x in t]), list(itertools.product([j/12 for j in range(12)],repeat=3))+[tuple(rng.random(3)) for _ in range(500)]),
}
which=sys.argv[1:] or list(fams)
for name in which:
    f,pts=fams[name]; out=open('results/scan_%s.jsonl'%name,'w'); t0=time.time(); cnt={}
    for t in pts:
        try:
            H=f(t); N=H.shape[0]
        except Exception: continue
        if not np.all(np.isfinite(H)) or not np.allclose(H@H.conj().T, N*np.eye(N), atol=1e-8) or not np.allclose(abs(H),1,atol=1e-8): continue
        c2=cdense(H,2); c3=cdense(H,3)
        key=(c2,c3); cnt[key]=cnt.get(key,0)+1
        out.write(json.dumps({'t':[float(x) for x in t],'c2':c2,'c3':c3})+"\n")
    out.close()
    print(name, len(pts),"pts", {str(k):v for k,v in sorted(cnt.items())}, "%.0fs"%(time.time()-t0), flush=True)
