import sys, numpy as np, scipy.linalg as sl
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import cases, matrices as m
def Tk(H,k):
    N=H.shape[0]; n=N**k
    idx=np.array(np.unravel_index(np.arange(n),(N,)*k)).T
    A=np.ones((n,n),complex)
    for s in range(k):
        A*=np.conj(H[idx[:,s]][:,idx[:,s]])* H[idx[:,(s+1)%k]][:,idx[:,s]]
    return A@A.conj().T/N**(k+1)
def jw_basis(N,k):
    n=N**k; idx=np.array(np.unravel_index(np.arange(n),(N,)*k)).T
    W=np.where(np.all(idx[:,1:]!=idx[:,:-1],axis=1))[0] if k>1 else np.arange(n)
    rows=[]
    for a in range(k):
        others=[t for t in range(k) if t!=a]
        key=np.ravel_multi_index(idx[W][:,others].T,(N,)*(k-1)) if k>1 else np.zeros(len(W),int)
        C=np.zeros((N**(k-1),len(W))); C[key,np.arange(len(W))]=1; rows.append(C)
    C=np.vstack(rows); B=sl.null_space(C)
    full=np.zeros((n,B.shape[1])); full[W]=B; return full
def phi_spec(H,s,B=None):
    N=H.shape[0]; B=jw_basis(N,s) if B is None else B
    return np.sort(np.linalg.eigvalsh(B.T@Tk(H,s)@B))[::-1]
if __name__=="__main__":
    N=6; Bs={s:jw_basis(N,s) for s in [2,3,4]}; print("dims",[Bs[s].shape[1] for s in [2,3,4]])
    for lab in ["T_6","K_6, random","BN_6^q, q random on arc","H_6^q, q random"]:
        H=cases.C[lab](); sp={s:phi_spec(H,s,Bs[s]) for s in [2,3,4]}
        print(lab)
        for s in [2,3,4]: print("  Phi_%d top:"%s, np.round(sp[s][:8],4), " trace %.4f"%sp[s].sum())
        for s in [2,3]:
            # containment test: each eigenvalue of Phi_s (with multiplicity) matched in Phi_{s+1}?
            a=sp[s]; b=list(sp[s+1]); miss=0
            for x in a:
                j=int(np.argmin(np.abs(np.array(b)-x)))
                if abs(b[j]-x)<1e-8: b.pop(j)
                else: miss+=1
            print("  spec Phi_%d in spec Phi_%d: %d of %d eigenvalues unmatched"%(s,s+1,miss,len(a)))
