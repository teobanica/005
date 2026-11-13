import sys, numpy as np
sys.path.insert(0, "/home/claude/jw/jw")
import cases, matrices as m
def T(H,k):
    N=H.shape[0]; n=N**k
    idx=np.array(np.unravel_index(np.arange(n),(N,)*k)).T     # n x k
    # A_{i,x} = prod_s conj(H[i_s,x_s]) H[i_{s+1},x_s]
    A=np.ones((n,n),complex)
    for s in range(k):
        A*=np.conj(H[idx[:,s]][:,idx[:,s]])* H[idx[:,(s+1)%k]][:,idx[:,s]]
    return A@A.conj().T/N**(k+1), A
def Tdirect(H,k):
    N=H.shape[0]; n=N**k
    idx=np.array(np.unravel_index(np.arange(n),(N,)*k)).T
    xi=lambda a,b: H[a]*np.conj(H[b])/np.sqrt(N)
    out=np.zeros((n,n),complex)
    for I in range(n):
        for J in range(n):
            M=np.eye(N,dtype=complex)
            for s in range(k):
                v=xi(idx[I,s],idx[J,s]); M=M@np.outer(v,v.conj())
            out[I,J]=np.trace(M)/N
    return out
rng=np.random.default_rng(5)
for lab in ["K_6, random","T_6","BN_6^q, q random on arc"]:
    H=cases.C[lab]()
    t,_=T(H,2); print(lab,"check vs definition k=2:",np.abs(t-Tdirect(H,2)).max())
    for k in [2,3]:
        a=np.linalg.eigvalsh(T(H,k)[0]); b=np.linalg.eigvalsh(T(H.T,k)[0])
        print("  k=%d  spectra H vs H^t: max diff %.1e   top: %s"%(k,np.abs(np.sort(a)-np.sort(b)).max(),np.round(np.sort(a)[::-1][:7],4)))
# a Dita one with H != H^t type behaviour
q=[m.E(x) for x in np.random.default_rng(31).random(5)]
H=m.F85a(*q)
for k in [2,3]:
    a=np.linalg.eigvalsh(T(H,k)[0]); b=np.linalg.eigvalsh(T(H.T,k)[0])
    print("F85a k=%d  diff %.1e  mult of 1: %d %d"%(k,np.abs(np.sort(a)-np.sort(b)).max(),(a>1-1e-9).sum(),(b>1-1e-9).sum()))
