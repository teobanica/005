import numpy as np, sys
sys.path.insert(0,'/home/claude/dita')
from magic import M
rng=np.random.default_rng(3); r=lambda: np.exp(2j*np.pi*rng.random())
a,b,c,d,e=[r() for _ in range(5)]
H=M.F85b(a,b,c,d,e)
K=H[np.ix_([0,1,4,5],[0,1,2,3])]; L=H[np.ix_([2,3,6,7],[0,1,2,3])]
s=b*d; w=np.conj(c*e)
V3=np.array([[2*(1-s),(1+s)*(1+w),(1+s)*(1-w)],[0,2*(1-w),2*(1+w)],[2*(1+s),(1-s)*(1+w),(1-s)*(1-w)]])/4
V=np.zeros((4,4),complex); V[0,0]=1; V[1:,1:]=V3
xi=lambda i,j: K[i]*L[j].conj()
ip=lambda u,v: np.vdot(v,u)  # linear at left
A=np.array([[ip(xi(i,j),xi(0,0))/4 for j in range(4)] for i in range(4)])
B=np.array([[ip(xi(i,0),xi(0,j))/4 for j in range(4)] for i in range(4)])
print("K,L ok?", np.allclose(K, [[1,1,1,1],[1,-1,-s,s],[1,1,-1,-1],[1,-1,s,-s]]))
print("|<xi_ab,xi_00>|/4 vs |V|:", np.abs(np.abs(A)-np.abs(V)).max())
print("|<xi_a0,xi_0b>|/4:\n",np.round(np.abs(B),4),"\n|V|:\n",np.round(np.abs(V),4))
from pauli import C, su2, pauli_gram, model_gram
from scipy.optimize import least_squares
x=su2(rng.normal(size=3))
tr=lambda m: np.trace(m)/2
rho=np.array([[ -tr(C[l]@x@C[k]@x.conj().T) for k in range(1,4)] for l in range(1,4)]).real
print("rho orthogonal:",np.allclose(rho@rho.T,np.eye(3)), np.linalg.det(rho))
P=lambda a,b: (C[a]@x@C[b]).ravel()/np.sqrt(2)
O=np.array([[abs(np.vdot(P(0,0),P(a,b))) for b in range(1,4)] for a in range(1,4)])
print("overlap = |rho| or |rho^t|:", np.allclose(O,abs(rho)), np.allclose(O,abs(rho.T)))
vecs=np.array([[K[i]*L[j].conj() for j in range(4)] for i in range(4)])
Gp=model_gram(vecs); best=1e9
for seed in range(20):
    rr=least_squares(lambda p:(pauli_gram(su2(p))-Gp).ravel(), np.random.default_rng(seed).normal(size=3))
    best=min(best,rr.cost)
print("identity-label Gram fit cost:",best)
