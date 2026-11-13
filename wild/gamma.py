import sys, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import cases
from phis import jw_basis, Tk
def rotD(N,s):
    """matrix of v -> rho(D v): e_{i1..is} -> e_{is i1 ... is}, as N^{s+1} x N^s"""
    n=N**s; idx=np.array(np.unravel_index(np.arange(n),(N,)*s)).T
    new=np.concatenate([idx[:,-1:],idx],axis=1)
    rows=np.ravel_multi_index(new.T,(N,)*(s+1)); M=np.zeros((N**(s+1),n)); M[rows,np.arange(n)]=1; return M
# 1) intertwining check on K_6
H=cases.C["K_6, random"](); N=6; s=2
Bs,Bt=jw_basis(N,s),jw_basis(N,s+1)
G=Bt.T@rotD(N,s)@Bs          # Gamma_s in JW coordinates, d_{s+1} x d_s
Ps=Bs.T@Tk(H,s)@Bs; Pt=Bt.T@Tk(H,s+1)@Bt
print("||Phi_3 Gamma - Gamma Phi_2|| =", np.linalg.norm(Pt@G-G@Ps))
# 2) injectivity of Gamma_s, universal
for N in [4,5,6,7]:
    out=[]
    for s in ([2,3,4] if N<=6 else [2,3]):
        Bs,Bt=jw_basis(N,s),jw_basis(N,s+1); G=Bt.T@rotD(N,s)@Bs
        sv=np.linalg.svd(G,compute_uv=False); out.append("s=%d: rank %d/%d, min sv %.3f"%(s,(sv>1e-9).sum(),Bs.shape[1],sv.min()))
    print("N=%d  "%N+" | ".join(out))
