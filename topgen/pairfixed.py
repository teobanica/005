# pair-fixed sectors of pi^{*k}: maximal invariant subspace on which the pair-level magic Y is the identity.
# there the flips generate a group dual, which is a quantum subgroup of G.
import numpy as np, sys, itertools
sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw')
import matrices as Mx
from magic import magic
k=int(sys.argv[1]) if len(sys.argv)>1 else 2
rng=np.random.default_rng(7); a,b,c,d,e=[np.exp(2j*np.pi*rng.random()) for _ in range(5)]
W=magic(Mx.F85a(a,b,c,d,e)); N=8
def conv(U,V):
    return np.array([[sum(np.kron(U[i,l],V[l,j]) for l in range(N)) for j in range(N)] for i in range(N)])
U=W
for _ in range(k-1): U=conv(U,W)
D=U.shape[2]; I=np.eye(D)
pairs=[[0,3],[1,5],[2,4],[6,7]]
Y=[[sum(U[i,j] for j in pairs[y]) for y in range(4)] for i in [p[0] for p in pairs]]
# subspace where Y_xx = 1 for all pairs x
M=np.vstack([I-Y[x][x] for x in range(4)])
u_,s,vh=np.linalg.svd(M); r=(s>1e-8).sum(); V=vh[r:].conj().T
print("k=",k,"D=",D,"dim{Y=id}:",V.shape[1])
# maximal invariant subspace inside
while True:
    Pp=I-V@V.conj().T
    M=np.vstack([Pp@U[i,j]@V for i in range(N) for j in range(N)])
    u_,s,vh=np.linalg.svd(M,full_matrices=False); r=(s>1e-8).sum()
    if r==0: break
    V=V@vh[r:].conj().T
    if V.shape[1]==0: break
print("maximal invariant pair-fixed subspace dim:",V.shape[1])
if V.shape[1]:
    f=[sum(V.conj().T@U[p[0],j]@V*(1 if j==q[0] else -1) for q in pairs[x:x+1] for j in q) for x,p in enumerate(pairs)]
    # flip of pair x = u_{p0,q0} - u_{p0,q1} with q = same pair (pairs fixed)
    f=[V.conj().T@(U[pairs[x][0],pairs[x][0]]-U[pairs[x][0],pairs[x][1]])@V for x in range(4)]
    n=V.shape[1]
    print("flips involutions:",[np.allclose(x@x,np.eye(n)) for x in f])
    print("traces of flips:",[round(np.trace(x).real,4) for x in f])
    for i,j in itertools.combinations(range(4),2):
        ev=np.linalg.eigvals(f[i]@f[j]); print(" f%d f%d args/2pi:"%(i,j),np.round(np.sort(np.angle(ev)/(2*np.pi)),5))
    found=[];frontier=[((),np.eye(n))]
    for L in range(1,11):
        new=[]
        for word,Mw in frontier:
            for s_ in range(4):
                if word and word[-1]==s_: continue
                M2=Mw@f[s_]; w2=word+(s_,)
                if np.allclose(M2,np.eye(n),atol=1e-8): found.append(w2)
                new.append((w2,M2))
        frontier=new
    print("reduced words of length<=10 equal to 1:",len(found),found[:5])
