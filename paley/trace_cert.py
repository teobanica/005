# Exact infinite-depth certificate for Paley I, q prime = 3 mod 4.
# X_psi(x,y)=psi(y-x) (psi even char of F_q^*, zero on infinity) is an eigenvector of A=N^3 T_2,
# eigenvalue m_psi = mu(psi) mu'(psi), mu = sum_t psi(t) C(t), mu' = sum_t psi(t) C'(t),
# C, C' autocorrelations of g(a)=h(-a)h(1-a), g'(a)=h(a)h(a-1), h(0)=1, h(u)=chi(u).
# Galois orbit traces P_d = sum_{ord psi = d} m_psi are integers; if N^2 does not divide P_d, then
# m_psi/N^2 (= N mu) is not an algebraic integer, so G_H is infinite.
import sys, numpy as np
from math import gcd
def mob(n):
    r=1;p=2
    while p*p<=n:
        if n%p==0:
            n//=p
            if n%p==0: return 0
            r=-r
        p+=1
    return -r if n>1 else r
def ramanujan(d,k): 
    g=gcd(d,k) if k else d
    return sum(mob(d//e)*e for e in range(1,g+1) if g%e==0 and d%e==0)
def data(q):
    sq=np.zeros(q,bool); sq[[x*x%q for x in range(1,q)]]=True
    chi=np.where(sq,1,-1); chi[0]=0
    h=chi.copy(); h[0]=1
    a=np.arange(q)
    g=h[(-a)%q]*h[(1-a)%q]; gp=h[a]*h[(a-1)%q]
    C=np.array([int(np.dot(g,np.roll(g,-t))) for t in range(q)])
    Cp=np.array([int(np.dot(gp,np.roll(gp,-t))) for t in range(q)])
    pr=next(x for x in range(2,q) if all(pow(x,(q-1)//p,q)!=1 for p in range(2,q) if (q-1)%p==0 and all(p%r for r in range(2,p))))
    hh=pr*pr%q; n=(q-1)//2
    sqs=[pow(hh,k,q) for k in range(n)]
    c=[int(C[s]) for s in sqs]; cp=[int(Cp[s]) for s in sqs]
    # m_psi for psi=k-th char: 4 sum_{i,j} w^{k(i+j)} c_i cp_j ; orbit trace by Ramanujan sums on index r=i+j mod n
    conv=[0]*n
    for i in range(n):
        ci=c[i]
        for j in range(n): conv[(i+j)%n]+=ci*cp[j]
    return n,conv,c,cp
def cert(q):
    N=q+1; n,conv,c,cp=data(q); out={}
    for d in [d for d in range(2,n+1) if n%d==0]:
        P=4*sum(ramanujan(d,r)*conv[r] for r in range(n))
        out[d]=P
    return N,n,out
if __name__=="__main__":
    lo,hi=int(sys.argv[1]),int(sys.argv[2])
    def isprime(p): return p>1 and all(p%r for r in range(2,int(p**.5)+1))
    bad=[]
    for q in range(lo,hi):
        if q%4!=3 or not isprime(q): continue
        N,n,out=cert(q)
        good=[d for d,P in out.items() if P%(N*N)]
        if not good: bad.append(q)
        print(q, "N=%d"%N, {d:(P, P%(N*N)!=0) for d,P in out.items()} if q<100 else ("certified by orbits of order",good), flush=True)
    print("uncertified:",bad)
