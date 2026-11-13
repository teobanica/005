import sys, numpy as np, itertools
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import cases
from phis import jw_basis, Tk
H=cases.C["K_6, random"](); N=6
B2,B3=jw_basis(N,2),jw_basis(N,3)
P2,P3=B2@B2.T,B3@B3.T
w2v,W2=np.linalg.eigh(B2.T@Tk(H,2)@B2); w3v,W3=np.linalg.eigh(B3.T@Tk(H,3)@B3)
o=np.argsort(w2v)[::-1]; w2v,W2=w2v[o],B2@W2[:,o]; o=np.argsort(w3v)[::-1]; w3v,W3=w3v[o],B3@W3[:,o]
lam=w2v[0]; print("lambda",lam)
v2=W2[:,0]; j=np.argmin(np.abs(w3v-lam)); v3=W3[:,j]; print("matched Phi_3 eigenvalue",w3v[j], "multiplicity in Phi_3:",(np.abs(w3v-lam)<1e-8).sum())
I=np.eye(N)
def span_test(name, vecs):
    M=np.array(vecs).T; Q,_=np.linalg.qr(M); r=np.linalg.norm(v3-Q@(Q.conj().T@v3)); print("  %-40s residual %.2e"%(name,r))
V2=v2.reshape(N,N)
# candidates: p_3(v2 (x) e_c), p_3(e_c (x) v2), insertions in the middle, and H-dependent ones
span_test("p3(v2 (x) e_c)", [P3@np.kron(v2,I[c]) for c in range(N)])
span_test("p3(e_c (x) v2)", [P3@np.kron(I[c],v2) for c in range(N)])
span_test("p3(middle insertion)", [P3@np.einsum('ab,c->acb',V2,I[c]).reshape(-1) for c in range(N)])
span_test("all three", [P3@np.kron(v2,I[c]) for c in range(N)]+[P3@np.kron(I[c],v2) for c in range(N)]+[P3@np.einsum('ab,c->acb',V2,I[c]).reshape(-1) for c in range(N)])
