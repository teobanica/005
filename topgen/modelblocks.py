import numpy as np, sys
sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw')
import matrices as Mx
from magic import magic
rng=np.random.default_rng(7)
a,b,c,d,e=[np.exp(2j*np.pi*rng.random()) for _ in range(5)]
H=Mx.F85a(a,b,c,d,e); W=magic(H); N=8
# commutant of A_1: find via random element of commutant -> use simultaneous block decomposition
gens=[W[i,j] for i in range(N) for j in range(N)]
# center element: build commutant basis
B=np.eye(64,dtype=complex)
for Q in gens:
    M=np.array([(Q@B[:,t].reshape(8,8)-B[:,t].reshape(8,8)@Q).ravel() for t in range(B.shape[1])]).T
    u,s,vh=np.linalg.svd(M,full_matrices=False); r=(s>1e-8).sum(); B=B@vh[r:].conj().T
X=sum(np.random.randn()*B[:,t].reshape(8,8) for t in range(B.shape[1])); X=X+X.conj().T
w,V=np.linalg.eigh(X)
groups=[]
for i,x in enumerate(w):
    if groups and abs(x-w[groups[-1][0]])<1e-6: groups[-1].append(i)
    else: groups.append([i])
for g in groups:
    Vb=V[:,g]; r=len(g)
    R=np.array([[np.trace(Vb.conj().T@W[i,j]@Vb).real for j in range(N)] for i in range(N)])
    print("block dim",r,"\nrank matrix:\n",np.round(R).astype(int))
    if r==1: print("vector:",np.round(Vb[:,0]/Vb[0,0],4))
