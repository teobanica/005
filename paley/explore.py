import sys, numpy as np, collections
sys.path.insert(0,'.'); from certify import A_matrix
sys.path.insert(0,'/home/claude/v21/code-29/paper'); import matrices as m
q=int(sys.argv[1]) if len(sys.argv)>1 else 19
H=np.real(m.paley1(q)).astype(int); N=q+1
A=A_matrix(H).astype(float); w,V=np.linalg.eigh(A)
mval=32*(q-1)
E=V[:,np.abs(w-mval)<1e-6]; print("q",q,"eigenvalue",mval,"mult",E.shape[1])
P=E@E.T
# row 0 = infinity, rows 1..q = F_q elements 0..q-1
for (a,b) in [(0,0),(1,1),(0,1),(1,2)]:
    v=P[:,a*N+b].reshape(N,N)
    vals=collections.Counter(np.round(v*1e4).astype(int).ravel().tolist())
    print("P_E e_(%d,%d): norm %.4f, distinct values:"%(a,b,np.linalg.norm(v)), sorted(vals.items())[:12])
# symmetry: is E inside symmetric or antisymmetric matrices?
S=np.array([E[:,t].reshape(N,N) for t in range(E.shape[1])])
print("sym part", np.linalg.norm(S-S.transpose(0,2,1)), "antisym part", np.linalg.norm(S+S.transpose(0,2,1)))
print("diag norm", np.linalg.norm(np.array([np.diag(x) for x in S])), "row sums", np.abs(S.sum(axis=2)).max(), "col sums", np.abs(S.sum(axis=1)).max())
