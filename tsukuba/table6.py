from formula6 import *
from magic import magic, ck_fast
import itertools
def index(M,N,a,b,L):
    pts=set()
    for m in range(L):
        for n in range(L):
            if (M,N)==(2,3):
                al,be=2*a,2*b; v=((al*(m+2*n)+be*(m-n))%L,(al*(n-m)+be*(2*m+n))%L)
            else:
                B=2*a-b; C=2*b-a; v=((B*2*m+C*(2*m+2*n))%L,(B*(2*m+2*n)+C*2*n)%L)
            pts.add(v)
    return len(pts)
def phi_nontriv(M,N,a,b,L):
    for m in range(L):
        for n in range(L):
            if inN_formula(M,N,a,b,L,m,n):
                if (M,N)==(2,3):
                    if (2*b*m+2*a*n)%L: return True
                else:
                    A=a+b; B=2*a-b; C=2*b-a
                    if (A*m+C*n)%L or (B*m+A*n)%L: return True
    return False
if __name__=="__main__":
    import sys
    for (M,N) in [(2,3),(3,2)]:
        print("=== %dx%d"%(M,N))
        seen={}
        for L in (1,2,3,4,6,8,12):
            for a in range(L):
                for b in range(L):
                    if gcd(gcd(a,b),L)!=1 and L>1: continue
                    idx=index(M,N,a,b,L); key=(idx,phi_nontriv(M,N,a,b,L))
                    if key not in seen: seen[key]=(a,b,L)
        for (idx,ph),(a,b,L) in sorted(seen.items()):
            Q=[[1,1,1],[1,E(a/L),E(b/L)]] if (M,N)==(2,3) else [[1,1],[1,E(a/L)],[1,E(b/L)]]
            print(" p=e(%d/%d), q=e(%d/%d): [Z^2:N]=%d, dim=%d, Phi nontrivial=%s, c_k="%(a,L,b,L,idx,6*idx,ph),counts(M,N,Q,5),flush=True)
