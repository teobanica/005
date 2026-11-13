import numpy as np, sys
sys.path.insert(0,'/home/claude/kar/day6/bobcat')
from pmatrices import paley2
from sectors import Aop
rng=np.random.default_rng(1)
for args in [(3,2),(13,),(17,)]:
    H=np.array(paley2(*args)).real.astype(int); N=H.shape[0]; m=N//2; q=m-1
    A=Aop(H)
    for kind in ("sym0","sym","anti","any"):
        M=rng.standard_normal((m,m))
        if kind.startswith("sym"): M=M+M.T
        if kind=="sym0": np.fill_diagonal(M,0)
        if kind=="anti": M=M-M.T
        for eps in ((1,1),(1,-1)):
            X=np.zeros((m,2,m,2))
            X[:,0,:,1]=eps[0]*M; X[:,1,:,0]=eps[1]*M
            v=X.reshape(-1); Av=A@v
            lam=v@Av/(v@v)
            print("q=%d %s eps=%s: Rayleigh %.4f (64q=%d), residual %.2e"%(q,kind,eps,lam,64*q,np.linalg.norm(Av-lam*v)/np.linalg.norm(v)))
