import numpy as np, sys, itertools
from scipy.optimize import least_squares
sys.path.insert(0,'/home/claude/jw/jw')
from ck import ck
from fam import rot, nvec
F2=np.array([[1,1],[1,-1]],dtype=complex); D=lambda z: np.diag([1,z])
sx=np.array([[0,1],[1,0]]); sy=np.array([[0,-1j],[1j,0]]); sz=np.diag([1.,-1])
def resid(v,A,th):
    q=A.shape[0]; ph=v[0]; al=v[1:q]; be=v[q:]
    n=nvec(th,ph)
    a=[np.array([0,0,1.])]+[np.array([0,np.sin(t),np.cos(t)]) for t in al]
    b=[np.array([1.,0,0])]+[np.array([np.cos(t),np.sin(t),0]) for t in be]
    return np.array([a[i]@rot(n,A[i,j])@b[j] for i in range(1,q) for j in range(1,q)])
def build(K,L,th,ph,al,be,sgn):
    q=K.shape[0]; n=nvec(th,ph); Lam=n[0]*sx+n[1]*sy+n[2]*sz
    w,U=np.linalg.eigh(Lam); U=U[:,::-1]  # +1 first
    if sgn<0: U=U[:,::-1]
    W=lambda i,j: U@np.diag([K[i,j],L[i,j]])@U.conj().T
    X=[np.eye(2)]+[0.5*F2@D(np.exp(1j*t))@F2 for t in al]
    Y=[F2]+[D(np.exp(1j*t))@F2 for t in be]
    return np.block([[X[i]@W(i,j)@Y[j] for j in range(q)] for i in range(q)])
def solve(K,L,th,trials=20,seed=0):
    q=K.shape[0]; A=np.angle(K*L.conj()); rng=np.random.default_rng(seed); out=[]
    for t in range(trials):
        r=least_squares(lambda v: resid(v,A,th), rng.uniform(0,2*np.pi,2*q-1), xtol=1e-15,ftol=1e-15,gtol=1e-15)
        if np.abs(r.fun).max()<1e-11:
            v=r.x; q_=q
            for sgn in (1,-1):
                for ca in (1,-1):
                    H=build(K,L,th,v[0],ca*v[1:q],v[q:],sgn)
                    if np.abs(np.abs(H)-1).max()<1e-9: out.append(H); break
                else: continue
                break
    return out
if __name__=="__main__":
    q=5; z=np.exp(2j*np.pi/5); F=np.array([[z**(i*j) for j in range(q)] for i in range(q)])
    perms=[p for p in itertools.permutations(range(1,q))]
    from collections import Counter
    res={}
    for s in perms:
        for t in perms:
            Ls=F[np.ix_((0,)+s,(0,)+t)]
            Hs=solve(F,Ls,0.8,8)
            res[(s,t)]=Counter(ck(H,2)[0] for H in Hs)
    allc=Counter()
    for k,v in res.items(): allc.update(v)
    print('q=5 all pairings: c2 distribution', dict(allc), 'pairings with solutions', sum(1 for v in res.values() if v))
    print('c2=2 pairings:', [k for k,v in res.items() if 2 in v][:5])
