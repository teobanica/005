# sectors of pi^{*k} for F85a: which generators (t, m1, m2, flips) act trivially -> quantum subgroups of G
import numpy as np, sys
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
D=U.shape[2]
X=[[0,3,1,5],[2,4,6,7]]
t=2*sum(U[0,bb] for bb in X[0])-np.eye(D)
def nu(i,aa,bb): return sum(U[X[i][aa],X[j][bb]] for j in range(2))
m=[2*(nu(i,0,0)+nu(i,0,1))-np.eye(D) for i in range(2)]
fl=[nu(i,2*p,2*q)-nu(i,2*p,2*q+1) for i in range(2) for p in range(2) for q in range(1)]
fl=[sum(nu(i,2*p,cc)*(1 if cc%2==0 else -1) for cc in range(4)) for i in range(2) for p in range(2)]
gens={'t':t,'m1':m[0],'m2':m[1],'fA':fl[0],'fB':fl[1],'fC':fl[2],'fD':fl[3]}
print("involutions:",{k_:np.allclose(g@g,np.eye(D)) for k_,g in gens.items()})
# central decomposition of the image algebra via random hermitian element of the commutant (iterative nullspace)
G=[U[i,j] for i in range(N) for j in range(N)]
Bm=np.eye(D*D,dtype=complex)
for Q in G:
    r=Bm.shape[1]
    M=np.empty((D*D,r),complex)
    for s in range(r):
        Xm=Bm[:,s].reshape(D,D); M[:,s]=(Q@Xm-Xm@Q).ravel()
    u_,sv,vh=np.linalg.svd(M,full_matrices=False); rk=(sv>1e-8).sum(); Bm=Bm@vh[rk:].conj().T
mats=[Bm[:,s].reshape(D,D) for s in range(Bm.shape[1])]
print("dim commutant",len(mats))
# center of commutant: elements of commutant commuting with all commutant
Cm=np.zeros((len(mats)*D*D,len(mats)),complex)
Z=None
Y=sum(rng.standard_normal()*Mm for Mm in mats); Y=Y+Y.conj().T
# eigenspaces of a generic hermitian commutant element refine to isotypic+multiplicity; group by joint action instead:
w,V=np.linalg.eigh(Y)
grp=[]
for i,x in enumerate(w):
    if grp and abs(x-w[grp[-1][0]])<1e-6: grp[-1].append(i)
    else: grp.append([i])
seen={}
for g in grp:
    Vb=V[:,g]
    # check invariance under algebra
    inv=all(np.allclose(Vb@(Vb.conj().T@Q@Vb),Q@Vb,atol=1e-7) for Q in G)
    sig=[]
    for nm,gm in gens.items():
        R=Vb.conj().T@gm@Vb; n=len(g)
        if np.allclose(R,np.eye(n),atol=1e-7): sig.append(nm+'=+1')
        elif np.allclose(R,-np.eye(n),atol=1e-7): sig.append(nm+'=-1')
    key=(len(g),inv,tuple(sig)); seen[key]=seen.get(key,0)+1
for kk,vv in sorted(seen.items()): print("dim",kk[0],"invariant",kk[1],"x",vv,":",kk[2])
