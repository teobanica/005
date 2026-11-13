import sys, numpy as np, scipy.linalg as sl
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import cases
from phis import jw_basis, Tk
H=cases.C[sys.argv[1] if len(sys.argv)>1 else "K_6, random"](); N=H.shape[0]
B2,B3=jw_basis(N,2),jw_basis(N,3)
e2,W2=np.linalg.eigh(B2.T@Tk(H,2)@B2); e3,W3=np.linalg.eigh(B3.T@Tk(H,3)@B3); W2=B2@W2; W3=B3@W3
pairs=[]
for j in range(len(e2)):
    i3=np.where(np.abs(e3-e2[j])<1e-8)[0]
    if len(i3)==1 and (np.abs(e2-e2[j])<1e-8).sum()==1: pairs.append((W2[:,j],W3[:,i3[0]]))
print(len(pairs),"simple matched pairs")
def solve(name, build):   # build(y) -> L as (N^3 x N^2) linear in y ; we pass basis maps
    n=N**3; cols=[]
    for idx in range(n):
        y=np.zeros(n); y[idx]=1; cols.append(build(y))
    rows=[]
    for v2,v3 in pairs:
        Pp=np.eye(N**3)-np.outer(v3,v3.conj())
        rows.append(np.array([Pp@(L@v2) for L in cols]).T)
    M=np.vstack(rows); s=np.linalg.svd(M,compute_uv=False)
    print("  %-28s smallest singular values: %s"%(name,np.round(s[-4:],6)))
    return M
def ins_end(y):
    Y=y.reshape(N,N,N); L=np.zeros((N**3,N**2),complex)
    for a in range(N):
        for b in range(N):
            for c in range(N): L[(a*N+b)*N+c,a*N+b]=Y[a,b,c]
    return L
def ins_mid(y):
    Y=y.reshape(N,N,N); L=np.zeros((N**3,N**2),complex)
    for a in range(N):
        for b in range(N):
            for c in range(N): L[(a*N+c)*N+b,a*N+b]=Y[a,c,b]
    return L
def ins_front(y):
    Y=y.reshape(N,N,N); L=np.zeros((N**3,N**2),complex)
    for a in range(N):
        for b in range(N):
            for c in range(N): L[(c*N+a)*N+b,a*N+b]=Y[c,a,b]
    return L
for nm,f in [("insert at end",ins_end),("insert in middle",ins_mid),("insert at front",ins_front)]: solve(nm,f)
print("--- nontrivial solutions (L v2 not all zero) ---")
def nontriv(name, build):
    n=N**3; Ls=[build(np.eye(n)[i]) for i in range(n)]
    Mfull=np.vstack([np.array([(np.eye(N**3)-np.outer(v3,v3.conj()))@(L@v2) for L in Ls]).T for v2,v3 in pairs])
    Mzero=np.vstack([np.array([L@v2 for L in Ls]).T for v2,v3 in pairs])
    Z=sl.null_space(Mzero); NS=sl.null_space(Mfull)
    # dimension of NS modulo Z
    r=np.linalg.matrix_rank(np.hstack([NS,Z]),tol=1e-8)-np.linalg.matrix_rank(Z,tol=1e-8)
    print("  %-28s dim null = %d, trivial = %d, nontrivial = %d"%(name,NS.shape[1],Z.shape[1],r))
    return NS,Z,Ls
for nm,f in [("insert at end",ins_end),("insert in middle",ins_mid),("insert at front",ins_front)]: nontriv(nm,f)
