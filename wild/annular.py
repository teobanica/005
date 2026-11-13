import sys, numpy as np, scipy.linalg as sl
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import cases
from phis import jw_basis, Tk
def lam_basis(N,t):
    """annular lowest weight space: leg sums zero, zero on cyclically adjacent equal letters"""
    n=N**t; idx=np.array(np.unravel_index(np.arange(n),(N,)*t)).T
    ok=np.all(idx!=np.roll(idx,-1,axis=1),axis=1) if t>1 else np.ones(n,bool)
    W=np.where(ok)[0]; rows=[]
    for a in range(t):
        others=[x for x in range(t) if x!=a]
        key=np.ravel_multi_index(idx[W][:,others].T,(N,)*(t-1)) if t>1 else np.zeros(len(W),int)
        C=np.zeros((N**(t-1),len(W))); C[key,np.arange(len(W))]=1; rows.append(C)
    B=sl.null_space(np.vstack(rows)); full=np.zeros((n,B.shape[1])); full[W]=B; return full
def rot(N,t):
    n=N**t; idx=np.array(np.unravel_index(np.arange(n),(N,)*t)).T
    R=np.zeros((n,n)); R[np.ravel_multi_index(np.roll(idx,1,axis=1).T,(N,)*t),np.arange(n)]=1; return R
def sectors(N,t):
    L=lam_basis(N,t).astype(complex); R=rot(N,t); M=L.conj().T@R@L
    w,V=np.linalg.eig(M); out={}
    for k in range(t):
        om=np.exp(2j*np.pi*k/t); sel=np.abs(w-om)<1e-6
        if sel.any():
            Q,_=np.linalg.qr(L@V[:,sel]); out[k]=Q
    return out
if __name__=="__main__":
    N=6; H=cases.C["K_6, random"]()
    S={t:sectors(N,t) for t in [1,2,3,4]}
    for t in S: print("t=%d: dim Lambda_t = %d, by rotation sector k (omega=e^{2pi i k/t}): %s"%(t,sum(Q.shape[1] for Q in S[t].values()),{k:Q.shape[1] for k,Q in S[t].items()}))
    psi={}
    for t in S:
        T=Tk(H,t)
        for k,Q in S[t].items(): psi[(t,k)]=np.sort(np.linalg.eigvalsh(Q.conj().T@T@Q))[::-1]
    phi={s:np.sort(np.linalg.eigvalsh(jw_basis(N,s).T@Tk(H,s)@jw_basis(N,s)))[::-1] for s in [2,3,4]}
    # greedy decomposition of Phi_s into Psi's: solve for multiplicities by matching
    import itertools
    for s in [2,3,4]:
        rest=list(phi[s]); mult={}
        for key in sorted(psi, key=lambda x:(-x[0],x[1])):
            if key[0]>s: continue
            c=0
            while True:
                tmp=list(rest); okk=True
                for x in psi[key]:
                    j=int(np.argmin(np.abs(np.array(tmp)-x))) if tmp else -1
                    if j>=0 and abs(tmp[j]-x)<1e-8: tmp.pop(j)
                    else: okk=False; break
                if not okk: break
                rest=tmp; c+=1
            mult[key]=c
        print("Phi_%d (dim %d) = "%(s,len(phi[s]))+" + ".join("%d x Psi_%d,%d"%(c,k[0],k[1]) for k,c in mult.items() if c)+"   leftover: %d"%len(rest))
