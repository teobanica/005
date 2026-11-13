import sys, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import cases
from phis import jw_basis, Tk
H=cases.C["K_6, random"](); N=6
B2,B3=jw_basis(N,2),jw_basis(N,3)
e2,W2=np.linalg.eigh(B2.T@Tk(H,2)@B2); e3,W3=np.linalg.eigh(B3.T@Tk(H,3)@B3)
W2=B2@W2; W3=B3@W3
for j in [-1,-2,-3]:
    lam=e2[j]; v2=W2[:,j]; i3=np.argmin(np.abs(e3-lam)); v3=W3[:,i3].reshape(N,N,N)
    print("lambda %.4f"%lam)
    V2=v2.reshape(N,N)
    for name,M in [("legs(1,2)",np.einsum('abc,dec->abde',v3,v3.conj())),("legs(2,3)",np.einsum('cab,cde->abde',v3,v3.conj())),("legs(1,3)",np.einsum('acb,dce->abde',v3,v3.conj())),
                   ("legs(2,1)",np.einsum('bac,edc->abde',v3,v3.conj())),("legs(3,2)",np.einsum('cba,ced->abde',v3,v3.conj())),("legs(3,1)",np.einsum('bca,ecd->abde',v3,v3.conj()))]:
        M=M.reshape(N*N,N*N); print("   %s: <v2|M|v2>/TrM = %.3f, top eigvalue share %.3f"%(name,np.real(v2.conj()@M@v2)/np.trace(M).real, np.linalg.eigvalsh(M)[-1]/np.trace(M).real))
