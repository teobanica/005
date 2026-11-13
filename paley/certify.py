# Infinite depth certificates (Montgomery's criterion): a rational eigenvalue mu = m/N^3 of T_2 (A = N^3 T_2 is an
# integer matrix for real H) with N*mu = m/N^2 not an integer. Eigenvectors are found numerically, then made exact.
import sys, numpy as np
from fractions import Fraction
from math import lcm
sys.path.insert(0,'/home/claude/v21/code-29/paper')
import matrices as m
def A_matrix(H):
    N=H.shape[0]; H=np.real(H).astype(np.int64)
    # s[(i1,j1),(i2,j2)] = sum_x H_i1x H_j1x H_i2x H_j2x ; T_2 = s^2/N^3
    R=np.einsum('ix,jx->ijx',H,H).reshape(N*N,N)
    S=R@R.T
    return (S*S)                            # = N^3 T_2, integer
def certify(H, maxden=10**6):
    N=H.shape[0]; A=A_matrix(H); n=A.shape[0]
    w,V=np.linalg.eigh(A.astype(float))
    out=[]
    for mval in sorted(set(np.round(w).astype(np.int64).tolist()),reverse=True):
        idx=np.abs(w-mval)<1e-6*max(1,abs(mval))
        if not idx.any() or mval==0: continue
        Vm=V[:,idx]; d=Vm.shape[1]
        # pick d well-conditioned coordinates
        Q,Rr,piv=__import__('scipy.linalg',fromlist=['qr']).qr(Vm.T,pivoting=True)
        J=piv[:d]; v=Vm@np.linalg.solve(Vm[J,:],np.eye(d)[:,0])
        fr=[Fraction(float(x)).limit_denominator(maxden) for x in v]
        L=1
        for f in fr: L=lcm(L,f.denominator)
        iv=[int(f*L) for f in fr]
        Av=[sum(int(A[i,j])*iv[j] for j in range(n) if iv[j]) for i in range(n)]
        ok=all(Av[i]==mval*iv[i] for i in range(n)) and any(iv)
        nmu=Fraction(mval,N*N)
        out.append((mval,d,ok,nmu))
    return out
if __name__=="__main__":
    cases=[("Paley I",3,1),("Paley I",7,1),("Paley I",11,1),("Paley I",19,1),("Paley I",23,1),("Paley I",3,3),("Paley I",31,1),
           ("Paley II",5,1),("Paley II",3,2),("Paley II",13,1),("Paley II",17,1)]
    for kind,p,r in cases:
        q=p**r; H=m.paley1(p,r) if kind=="Paley I" else m.paley2(p,r); N=H.shape[0]
        res=certify(H)
        good=[(str(nmu),d) for (mv,d,ok,nmu) in res if ok and nmu.denominator!=1]
        print("%s q=%d N=%d: exact rational eigenvalues N*mu (mult, verified): %s"%(kind,q,N,[(str(nmu),d,ok) for (mv,d,ok,nmu) in res]),flush=True)
        print("   ==> %s"%("INFINITE DEPTH, certificate N*mu = %s"%good[0][0] if good else "no rational certificate"),flush=True)
