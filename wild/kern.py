import sys, numpy as np, scipy.linalg as sl
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import cases
from phis import jw_basis, Tk
from gamma2 import maps
N=6
B2,B3,B4=jw_basis(N,2),jw_basis(N,3),jw_basis(N,4)
joint=np.vstack([B4.T@M@B3 for M in maps(N,3).values()])
Kc=sl.null_space(joint); print("dim K =",Kc.shape[1])
K=B3@Kc    # in C^{N^3}
# structure of K: is it spanned by rotation-invariant / antisymmetric vectors? check rotation eigenvalues
n=N**3; idx=np.array(np.unravel_index(np.arange(n),(N,)*3)).T
R=np.zeros((n,n)); R[np.ravel_multi_index(np.roll(idx,1,axis=1).T,(N,)*3),np.arange(n)]=1
print("K rotation-invariant subspace? ||R K - K (K^* R K)|| =", np.linalg.norm(R@K-K@(K.T@R@K)))
print("rotation eigenvalues on K:", np.round(np.linalg.eigvals(K.T@R@K),4))
for lab in ["K_6, random","BN_6^q, q random on arc","T_6"]:
    H=cases.C[lab]()
    e2=np.sort(np.linalg.eigvalsh(B2.T@Tk(H,2)@B2))[::-1]
    eK=np.sort(np.linalg.eigvalsh(K.T@Tk(H,3)@K))[::-1]
    e4=np.sort(np.linalg.eigvalsh(B4.T@Tk(H,4)@B4))[::-1]
    print(lab,"\n  Phi_3 on K:",np.round(eK,4),"\n  Phi_2 top :",np.round(e2[:8],4))
    print("  K-eigenvalues found in Phi_2:",[bool(np.min(np.abs(e2-x))<1e-8) for x in eK]," in Phi_4:",[bool(np.min(np.abs(e4-x))<1e-8) for x in eK])
