import numpy as np, itertools, sys
sys.path.insert(0,'/home/claude/jw/jw')
import matrices as M

def col_pairings(rho, tol=1e-8):
    """all perfect matchings of columns with rho[c'] = -rho[c]"""
    n=len(rho); out=[]
    def rec(rem, acc):
        if not rem: out.append(frozenset(acc)); return
        c=rem[0]
        for d in rem[1:]:
            if abs(rho[c]+rho[d])<tol:
                rec([x for x in rem if x not in (c,d)], acc+[(c,d)])
    rec(list(range(n)),[])
    return out

def row_matchings(n):
    def rec(rem):
        if not rem: yield []; return
        a=rem[0]
        for b in rem[1:]:
            for m in rec([x for x in rem if x not in (a,b)]): yield [(a,b)]+m
    yield from rec(list(range(n)))

def h2_structures(H):
    n=H.shape[0]; P={}
    for a,b in itertools.combinations(range(n),2):
        P[(a,b)]=set(col_pairings(H[a]/H[b]))
    res=[]
    for rm in row_matchings(n):
        s=None
        for pr in rm:
            s=P[pr] if s is None else s & P[pr]
            if not s: break
        if s:
            for cp in s: res.append((rm, sorted(cp)))
    return res

if __name__=="__main__":
    rng=np.random.default_rng(0); r=lambda: np.exp(2j*np.pi*rng.random())
    tests={'T6':M.T6(),'H6':M.H6(r()),'BF6':M.BF6(0),'BN6':M.BN6(np.exp(2j*np.pi*0.37)),
           'K6':M.K6(0.7,1.1,0.4),'P7':M.P7(r()),
           'F84':M.F84(r(),r(),r(),r()),'F84t':M.F84t(r(),r(),r(),r()),'F82':M.F82(r(),r()),'F82t':M.F82t(r(),r()),
           'K82':M.K82(r(),r()),'M81':M.M81(r()),'F85a':M.F85a(r(),r(),r(),r(),r())}
    for k,H in tests.items():
        if H.shape[0]%2: print(k,'odd'); continue
        s=h2_structures(H); s2=h2_structures(H.T)
        print(k, len(s), len(s2), s[0] if s else '')
