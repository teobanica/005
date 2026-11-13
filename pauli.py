import numpy as np, sys, itertools
from scipy.optimize import least_squares
sys.path.insert(0,'/home/claude/dita')
from magic import M
I2=np.eye(2); sx=np.array([[0,1],[1,0]]); sy=np.array([[0,-1j],[1j,0]]); sz=np.diag([1.,-1])
C=[I2,1j*sx,1j*sy,1j*sz]
def su2(p):
    a,b,c=p; n=np.sqrt(a*a+b*b+c*c)+1e-300
    return np.cos(n)*I2+1j*np.sin(n)*(a*sx+b*sy+c*sz)/n
def pauli_gram(x):
    V=np.array([[(C[a]@x@C[b]).ravel()/np.sqrt(2) for b in range(4)] for a in range(4)])
    V=V.reshape(16,4); return np.abs(V.conj()@V.T)**2
def model_gram(vecs):
    V=vecs.reshape(16,4); V=V/np.linalg.norm(V,axis=1,keepdims=True); return np.abs(V.conj()@V.T)**2
def fit(vecs):
    G=model_gram(vecs); best=(1e9,None)
    for s in itertools.permutations(range(4)):
        for t in itertools.permutations(range(4)):
            Gp=model_gram(vecs[np.ix_(s,t)])
            for seed in range(3):
                r=least_squares(lambda p: (pauli_gram(su2(p))-Gp).ravel(), np.random.default_rng(seed).normal(size=3))
                if r.cost<best[0]: best=(r.cost,(s,t,r.x))
            if best[0]<1e-20: return best
    return best
if __name__=="__main__":
    rng=np.random.default_rng(7); r=lambda: np.exp(2j*np.pi*rng.random())
    H=M.F85b(r(),r(),r(),r(),r())
    K=H[np.ix_([0,1,4,5],[0,1,2,3])]; L=H[np.ix_([2,3,6,7],[0,1,2,3])]
    for nm,(A,B) in [('U^K',(K,K)),('U^L',(L,L)),('R',(K,L))]:
        vecs=np.array([[A[a]*B[b].conj() for b in range(4)] for a in range(4)])
        cost,(s,t,p)=fit(vecs)
        print(nm,'best fit cost',cost,'perms',s,t,'x params',np.round(p,4))
