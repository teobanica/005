# irrational eigenvalues: find conjugate pairs (theta1, theta2) of A = N^3 T_2 with x^2 - s x + p integer, certify the
# kernel of A^2 - s A + p exactly, and test whether theta/N^2 is an algebraic integer (N^2 | s and N^4 | p).
import sys, numpy as np, itertools
from fractions import Fraction
from math import lcm
sys.path.insert(0,'.'); from certify import A_matrix
sys.path.insert(0,'/home/claude/v21/code-29/paper'); import matrices as m
import scipy.linalg as sl
def pairs(H):
    N=H.shape[0]; A=A_matrix(H); w,V=np.linalg.eigh(A.astype(float))
    vals=[]
    for x in sorted(w,reverse=True):
        if not vals or abs(x-vals[-1][0])>1e-6: vals.append([x,1])
        else: vals[-1][1]+=1
    irr=[(x,k) for x,k in vals if abs(x-round(x))>1e-6]
    out=[]
    for (a,ka),(b,kb) in itertools.combinations(irr,2):
        s,p=a+b,a*b
        if ka==kb and abs(s-round(s))<1e-6 and abs(p-round(p))<1e-4:
            s,p=int(round(s)),int(round(p))
            # exact check: kernel of B = A^2 - sA + pI contains a nonzero rational vector
            n=A.shape[0]; Ao=A.astype(object)
            idx=(np.abs(w-a)<1e-6)|(np.abs(w-b)<1e-6); Vm=V[:,idx]; d=Vm.shape[1]
            Q,R,piv=sl.qr(Vm.T,pivoting=True); J=piv[:d]; v=Vm@np.linalg.solve(Vm[J,:],np.eye(d)[:,0])
            fr=[Fraction(float(x)).limit_denominator(10**7) for x in v]; L=1
            for f in fr: L=lcm(L,f.denominator)
            iv=np.array([int(f*L) for f in fr],dtype=object)
            Av=Ao.dot(iv); AAv=Ao.dot(Av); ok=all(AAv-s*Av+p*iv==0) and any(iv!=0)
            out.append((s,p,ka,ok,(s%(N*N)==0) and (p%(N**4)==0)))
    return out
if __name__=="__main__":
    for kind,pp,r in [("Paley I",23,1),("Paley I",3,3),("Paley I",31,1)]:
        H=m.paley1(pp,r); N=H.shape[0]
        for s,p,k,ok,integral in pairs(H):
            print("%s q=%d: x^2 - %d x + %d, mult %d, exact %s, theta/N^2 algebraic integer: %s"%(kind,pp**r,s,p,k,ok,integral),flush=True)
