import sys, numpy as np, scipy.linalg as sl, scipy.optimize as so
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m, ck
from annular import lam_basis
def fix_space(H,k):
    N=H.shape[0]; n=N**k
    if n<=1296:
        from phis import Tk
        e,V=np.linalg.eigh(Tk(H,k)); return V[:,e>1-1e-9]
    raise ValueError
def inter(A,B):
    # orthonormal basis of span(A) ∩ span(B)
    QA=sl.orth(A); QB=sl.orth(B); M=QA.conj().T@QB; U,s,Vh=np.linalg.svd(M,full_matrices=False)
    return QA@U[:,s>1-1e-9]
def rot(x,N,k,i): return np.roll(x.reshape((N,)*k),i,axis=tuple(range(k)) if False else None) if False else np.moveaxis(x.reshape((N,)*k),list(range(k)),[(a+i)%k for a in range(k)]).reshape(-1)
def ops(N,k):
    """bilinear maps (x,y) -> N x N matrix, all planar: partial inner products of rotations, pairing k-1 legs,
    and 'merge one leg, pair k-2 legs' variants"""
    out=[]
    for i in range(k):
        for j in range(k):
            out.append(("pair",i,j))
            if k>=3: out.append(("mergepair",i,j))
    return out
def apply(op,x,y,N,k):
    kind,i,j=op; X=rot(x,N,k,i).reshape(N,-1); Y=rot(y,N,k,j).reshape(N,-1)
    if kind=="pair": return X@Y.conj().T                       # legs 2..k paired
    X3=rot(x,N,k,i).reshape(N,N**(k-2),N); Y3=rot(y,N,k,j).reshape(N,N**(k-2),N)
    return np.einsum('arc,brc->ab',X3,Y3.conj())              # last leg merged-then-summed = paired too; use merge into first leg:
def apply2(op,x,y,N,k):
    kind,i,j=op
    if kind=="pair":
        X=rot(x,N,k,i).reshape(N,-1); Y=rot(y,N,k,j).reshape(N,-1); return X@Y.conj().T
    # merge leg 1 of x with leg 1 of y (same letter), keep it, pair legs 2..k-1, and keep the last legs: gives level 3 -> sum last leg of y to land at level 2
    X3=rot(x,N,k,i).reshape(N,N**(k-2),N); Y3=rot(y,N,k,j).reshape(N,N**(k-2),N)
    return np.einsum('arc,ard->cd',X3,Y3.conj())*0+np.einsum('arc,ard->ac',X3,Y3.conj().sum(axis=2)[...,None].repeat(N,axis=2)) if False else np.einsum('arc,arc->ac',X3,Y3.conj())
TLbasis=lambda N: sl.orth(np.array([np.ones(N*N),np.eye(N).reshape(-1)]).T)
def jordan_test(H,k,label):
    N=H.shape[0]; F=fix_space(H,k); L=inter(F,lam_basis(N,k).astype(complex)); f=L.shape[1]
    Q=TLbasis(N); P=np.eye(N*N)-Q@Q.conj().T
    O=ops(N,k); forms=[]
    for op in O:
        Bm=np.zeros((f,f,N*N),complex)
        for a in range(f):
            for b in range(f): Bm[a,b]=P@apply2(op,L[:,a],L[:,b],N,k).reshape(-1)
        forms.append(Bm)
    def obj(c):
        c=c[:f]+1j*c[f:]; c=c/np.linalg.norm(c)
        return sum(np.linalg.norm(np.einsum('a,b,abn->n',c,c.conj(),Bm))**2 for Bm in forms)
    rng=np.random.default_rng(0); best=1e9
    for _ in range(20 if f>1 else 1):
        r=so.minimize(obj,rng.standard_normal(2*f),method="BFGS"); best=min(best,r.fun)
    print("%s, level %d: f_k(G_0) = %d, min over unit lowest-weight vectors of the level-2 defect = %.3e"%(label,k,f,best))
if __name__=="__main__":
    Hd=m.F6rs(np.exp(0.4j),np.exp(1.9j))
    for k in [4]:
        jordan_test(m.fourier(6),k,"Z_6 (Fourier)")
        jordan_test(Hd,k,"2x3 Dita generic")
