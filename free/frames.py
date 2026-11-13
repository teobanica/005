import numpy as np, sys, itertools
sys.path.insert(0,'/home/claude/dita')
from magic import M
from scipy.optimize import least_squares
I2=np.eye(2); sx=np.array([[0,1],[1,0]]); sy=np.array([[0,-1j],[1j,0]]); sz=np.diag([1.,-1])
C=[I2,1j*sx,1j*sy,1j*sz]
def su2(p):
    n=np.linalg.norm(p)+1e-300
    return np.cos(n)*I2+1j*np.sin(n)*(p[0]*sx+p[1]*sy+p[2]*sz)/n
def pauli(x):
    return np.array([[ (lambda v: np.outer(v,v.conj()))((C[a]@x@C[b]).ravel()/np.sqrt(2)) for b in range(4)] for a in range(4)])
def magicpair(A,B):
    out=np.zeros((4,4,4,4),complex)
    for a in range(4):
        for b in range(4):
            v=A[a]*B[b].conj(); v=v/np.linalg.norm(v); out[a,b]=np.outer(v,v.conj())
    return out
def inv(Mg):
    P=[Mg[a,b] for a in range(4) for b in range(4)]
    return np.array([np.trace(P[0]@P[i]@P[j]) for i in range(16) for j in range(16)])
def commdim(Mg):
    rows=[np.kron(Mg[a,b],np.eye(4))-np.kron(np.eye(4),Mg[a,b].T) for a in range(4) for b in range(4)]
    s=np.linalg.svd(np.vstack(rows),compute_uv=False); return int(np.sum(s<1e-8))
def intertw(Mg,Pg):
    # F with F Pg = Mg F
    rows=[np.kron(Mg[a,b],np.eye(4))-np.kron(np.eye(4),Pg[a,b].T) for a in range(4) for b in range(4)]
    u,s,vh=np.linalg.svd(np.vstack(rows)); F=vh[-1].conj().reshape(4,4)
    return s[-1], F/np.sqrt(abs(np.trace(F@F.conj().T))/4)
def fitx(Mg,tries=40):
    t=inv(Mg); best=(1e9,None)
    for sd in range(tries):
        r=least_squares(lambda p: np.concatenate([(inv(pauli(su2(p)))-t).real,(inv(pauli(su2(p)))-t).imag]), np.random.default_rng(sd).normal(size=3)*2)
        if r.cost<best[0]: best=(r.cost,r.x)
        if best[0]<1e-24: break
    return best
def frame(Mg):
    c,p=fitx(Mg); x=su2(p); s,F=intertw(Mg,pauli(x)); return c,s,x,F
if __name__=="__main__":
    rng=np.random.default_rng(11); r=lambda: np.exp(2j*np.pi*rng.random())
    H=M.F85b(*[r() for _ in range(5)])
    K=H[np.ix_([0,1,4,5],[0,1,2,3])]; L=H[np.ix_([2,3,6,7],[0,1,2,3])]
    mods={'UK':magicpair(K,K),'UL':magicpair(L,L),'RKL':magicpair(K,L),'RLK':magicpair(L,K)}
    fr={}
    for nm,Mg in mods.items():
        cd=commdim(Mg); print(nm,'commutant dim',cd)
        if cd==1:
            c,s,x,F=frame(Mg); print('   fit cost',c,'intertwiner residual',s,'unitary?',np.allclose(F@F.conj().T,np.eye(4)))
            fr[nm]=(x,F)
    np.save('/tmp/frames.npy',np.array([0]))
