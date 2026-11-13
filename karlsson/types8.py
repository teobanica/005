from blocks import *
def types(H):
    out=[]
    for rm,cm in h2_structures(H):
        X,Y,W=normalize(to_blocks(H,rm,cm)); q=len(W)
        flat=[W[i][j] for i in range(1,q) for j in range(1,q)]
        comm=max(np.abs(a@b-b@a).max() for a in flat for b in flat)<1e-9
        su2=max(abs(np.linalg.det(w)-1) for w in flat)<1e-9
        scal=max(np.abs(w-w[0,0]*np.eye(2)).max() for w in flat)<1e-9
        out.append(('scalar' if scal else 'comm' if comm else 'noncomm')+('/SU2' if su2 else ''))
    from collections import Counter
    return dict(Counter(out))
if __name__=="__main__":
    rng=np.random.default_rng(11); r=lambda: np.exp(2j*np.pi*rng.random())
    L=[('F84',M.F84(r(),r(),r(),r())),('F84t',M.F84t(r(),r(),r(),r())),('F82',M.F82(r(),r())),('F82t',M.F82t(r(),r())),('K82',M.K82(r(),r())),('M81',M.M81(r())),
       ('I812',M.from_log(M.I812_L,12)),('K830',M.from_log(M.K830_L,30)),('L830',M.from_log(M.L830_L,30)),('L830t',M.from_log(M.L830t_L,30)),
       ('F85a',M.F85a(r(),r(),r(),r(),r())),('F85b',M.F85b(r(),r(),r(),r(),r())),('F85c',M.F85c(r(),r(),r(),r(),r())),('K6',M.K6(.7,1.1,.4)),('H6',M.H6(r())),('BN6',M.BN6(np.exp(2j*np.pi*.37))),('BF6',M.BF6(0))]
    for nm,H in L:
        print(nm, types(H), ' | t:', types(H.T))
