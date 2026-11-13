import sys, time, numpy as np, scipy.optimize as so
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
from jordan import fix_space, inter, ops, apply2, TLbasis
from annular import lam_basis
def forms_tensor(H,k):
    N=H.shape[0]; F=fix_space(H,k); L=inter(F,lam_basis(N,k).astype(complex)); f=L.shape[1]
    Q=TLbasis(N); P=np.eye(N*N)-Q@Q.conj().T; O=ops(N,k)
    B=np.zeros((f,f,len(O)*N*N),complex)
    for a in range(f):
        for b in range(f):
            B[a,b]=np.concatenate([P@apply2(op,L[:,a],L[:,b],N,k).reshape(-1) for op in O])
    return B,f
def min_defect(B,f,starts=30,seed=0):
    def fg(x):
        c=x[:f]+1j*x[f:]; nc=np.vdot(c,c).real
        z=np.einsum('a,b,abn->n',c,c.conj(),B); F=np.vdot(z,z).real
        g=np.einsum('n,a,abn->b',z.conj(),c,B)+np.einsum('n,b,abn->a',z,c,B.conj())   # dF/d conj(c)
        val=F/nc**2; gc=g/nc**2-2*F/nc**3*c
        return val, np.concatenate([2*gc.real,2*gc.imag])
    rng=np.random.default_rng(seed); best=np.inf
    for _ in range(starts):
        r=so.minimize(fg,rng.standard_normal(2*f),jac=True,method="L-BFGS-B",options={"maxiter":5000,"gtol":1e-12})
        best=min(best,r.fun)
    return best
if __name__=="__main__":
    for lab,H in [("Z_6 (Fourier)",m.fourier(6)),("2x3 Dita generic",m.F6rs(np.exp(0.4j),np.exp(1.9j)))]:
        for k in [3,4]:
            t=time.time(); B,f=forms_tensor(H,k); d=min_defect(B,f)
            print("%s, level %d: f_k = %d, min level-2 defect = %.3e  (%.0f s)"%(lab,k,f,d,time.time()-t),flush=True)
