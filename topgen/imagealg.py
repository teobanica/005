# image algebras A_k = pi^{*k}(C(G)) for the F85a model: commutant, center, block sizes
import numpy as np, sys
sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw')
import matrices as Mx
from magic import magic
rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 7)
a,b,c,d,e=[np.exp(2j*np.pi*rng.random()) for _ in range(5)]
H=Mx.F85a(a,b,c,d,e); W=magic(H); N=8
k=int(sys.argv[1]) if len(sys.argv)>1 else 1
def conv(U,V):
    n=U.shape[0]
    return np.array([[sum(np.kron(U[i,l],V[l,j]) for l in range(n)) for j in range(n)] for i in range(n)])
U=W
for _ in range(k-1): U=conv(U,W)
D=U.shape[2]; gens=[U[i,j] for i in range(N) for j in range(N)]
def commutant(gens,D):
    B=np.eye(D*D,dtype=complex)  # columns = vec of basis matrices
    for Q in gens:
        r=B.shape[1]
        M=np.zeros((D*D,r),complex)
        for t in range(r):
            X=B[:,t].reshape(D,D); M[:,t]=(Q@X-X@Q).ravel()
        u,s,vh=np.linalg.svd(M,full_matrices=False) if r<=D*D else None
        null=vh[s.size-(s<1e-8).sum():].conj().T if (s<1e-8).any() else np.zeros((r,0))
        # include zero singular dims beyond rank
        rank=(s>1e-8).sum(); null=vh[rank:].conj().T
        B=B@null
        if B.shape[1]==0: break
    return B
C=commutant(gens,D)
print("k=",k,"D=",D,"dim commutant:",C.shape[1])
# center = commutant elements commuting with commutant? center of A = A ∩ A'. Since A''=A, center = A' ∩ A'' = center of A'.
mats=[C[:,t].reshape(D,D) for t in range(C.shape[1])]
Z=commutant(mats,D) if mats else None
# restrict to Z within A': center of A' = A' ∩ (A')'
# intersect span(C) with span(Z)
P=np.hstack([C,-Z]); u,s,vh=np.linalg.svd(P,full_matrices=False); rank=(s>1e-8).sum()
ns=vh[rank:].conj().T; cen=C@ns[:C.shape[1]]
print("dim center:",cen.shape[1])
# minimal central projections: diagonalize a random central element
X=sum(np.random.randn()*cen[:,t].reshape(D,D) for t in range(cen.shape[1])); X=X+X.conj().T
w,V=np.linalg.eigh(X)
vals=[]; 
for x in w:
    if not vals or abs(x-vals[-1][0])>1e-6: vals.append([x,1])
    else: vals[-1][1]+=1
# for each block: rank of projection, dim of A' restricted -> multiplicity m, irrep dim = rank/m
out=[]
for x,r in vals:
    Vb=V[:,np.abs(w-x)<1e-6]
    mdim=np.linalg.matrix_rank(np.array([ (Vb.conj().T@M@Vb).ravel() for M in mats]),tol=1e-7)
    m=int(round(np.sqrt(mdim))); out.append((r, m, r//max(m,1)))
print("blocks (rank, multiplicity, irrep dim):",sorted(out))
