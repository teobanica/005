from frames import *
from magic import ck_fast
from scipy.stats import unitary_group
rng=np.random.default_rng(5)
x=su2(rng.normal(size=3)); y=su2(rng.normal(size=3))
Px,Py=pauli(x),pauli(y)
def blk(A,B):
    W=np.zeros((8,8,4,4),complex); W[:4,:4]=A; W[4:,4:]=B; return W
def adj(U,P): return np.einsum('ij,abjk,lk->abil',U,P,U.conj())
op=lambda k,l: np.kron(C[k],C[l].T)
pb=np.array([C[m].ravel()/np.sqrt(2) for m in range(4)]).T   # Pauli basis columns
tests={"W=1":np.eye(4),"W=Pauli op(1,2)":op(1,2),
 "W diagonal in Pauli basis":pb@np.diag(np.exp(2j*np.pi*rng.random(4)))@pb.conj().T,
 "W random":unitary_group.rvs(4,random_state=3),
 "W = exp(i t X), t=0.01, X random herm":None}
X=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); X=X+X.conj().T
from scipy.linalg import expm
tests["W = exp(i t X), t=0.01, X random herm"]=expm(0.01j*X)
for nm,W in tests.items():
    print(nm,[ck_fast(blk(Px,adj(W,Py)),k) for k in (1,2,3)],flush=True)
