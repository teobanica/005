# Paley I, q prime = 3 mod 8: eigenvalue m_h of N^3 T_2 on the 2 half representations of PSL_2(F_q),
# via the trace formula  m_h = (1/(2|G|)) [ (q-1) tau(1) - sum_{u unipotent != 1} tau(u) - 2 sum_{t elliptic} theta0(t) tau(t) ],
# tau(g) = Tr(A g) = || U^T U ||_F^2, with U_{i x} = H_{i x} H_{g i, x}  (signs of the monomial action cancel).
import numpy as np, sys
from eigcheck import paley1
def mobius(q):
    INF=q
    def act(M,x):
        a,b,c,d=M
        if x==INF: return INF if c%q==0 else a*pow(c,q-2,q)%q
        den=(c*x+d)%q
        if den==0: return INF
        return (a*x+b)*pow(den,q-2,q)%q
    return act
def perm_rows(q,M):
    act=mobius(q); INF=q
    # row index 0 = infinity, 1+x = x
    lab=lambda r: INF if r==0 else r-1
    unl=lambda x: 0 if x==INF else x+1
    return np.array([unl(act(M,lab(r))) for r in range(q+1)])
def tau(H,p):
    U=(H*H[p]).astype(np.float64)   # U_{i x}=H_{ix}H_{p(i),x}
    G=U.T@U
    return float((G*G).sum())
def mh(q):
    H=paley1(q); N=q+1
    assert (H@H.T==N*np.eye(N)).all()
    # check: A commutes with the permutation action -> verify S-invariance on a random sample
    order=q*(q*q-1)//2
    tau1=float(N**4)
    unip=sum(tau(H,perm_rows(q,(1,c,0,1))) for c in range(1,q))*(q+1)
    # nonsplit torus generator of order (q+1)/2 in PSL
    act=mobius(q)
    def ord_(M):
        x=0; 
        P=list(range(q+1)); 
        # compose by iterating on the matrix
        a,b,c,d=M; A=(1,0,0,1); k=0
        while True:
            A=((A[0]*a+A[1]*c)%q,(A[0]*b+A[1]*d)%q,(A[2]*a+A[3]*c)%q,(A[2]*b+A[3]*d)%q); k+=1
            if (A[1]==0 and A[2]==0 and A[0]==A[3]): return k,A
    gen=None
    for t in range(q):
        if all((x*x-t*x+1)%q for x in range(q)):
            M=(0,q-1,1,t); k,_=ord_(M)
            if k in (q+1, (q+1)//2) :
                gen=M; break
    k,_=ord_(gen); ordT=(q+1)//2
    # powers of gen as Moebius maps; image order in PSL is (q+1)/2
    pw=[(1,0,0,1)]; A=(1,0,0,1)
    a,b,c,d=gen
    for j in range(1,ordT):
        A=((A[0]*a+A[1]*c)%q,(A[0]*b+A[1]*d)%q,(A[2]*a+A[3]*c)%q,(A[2]*b+A[3]*d)%q); pw.append(A)
    ell=sum(((-1)**j)*tau(H,perm_rows(q,pw[j])) for j in range(1,ordT))*(q*(q-1)//2)
    m=((q-1)*tau1-unip-2*ell)/(2*order)
    return m
if __name__=="__main__":
    for q in map(int,sys.argv[1:]):
        m=mh(q); print(q, q%8, round(m,4), m/64, flush=True)
