# Test of the smash-coproduct criterion for F85a:
# C(W) = (C(H2+)*C(H2+)) x| C(Z2); theta_1 = model restricted to the block-preserving sector.
# Compare c_k of the Hopf image of theta_1 (as a rep of the dual free product) with m_k(mu [+] mu).
import numpy as np, sys
from fractions import Fraction as Fr
from math import comb
sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw')
import matrices as Mx
from magic import magic, ck_fast
K=int(sys.argv[1]) if len(sys.argv)>1 else 4
# moments of mu = 1/2 delta_0 + 1/2 arcsine[0,4], then of mu boxplus mu
m=[Fr(1)]+[Fr(comb(2*k,k),2) for k in range(1,K+1)]
def mom_to_cum(m,K):
    # free cumulants via m_n = sum_{s=1}^n kappa_s * [z^{n-s}] M(z)^s  (standard recursion)
    kap=[Fr(0)]*(K+1)
    def coeff_pow(s,n):  # [z^n] (sum_j m_j z^j)^s
        poly=[Fr(1)]+[Fr(0)]*n
        for _ in range(s):
            new=[Fr(0)]*(n+1)
            for i in range(n+1):
                if poly[i]==0: continue
                for j in range(n+1-i): new[i+j]+=poly[i]*m[j]
            poly=new
        return poly[n]
    for n in range(1,K+1):
        kap[n]=m[n]-sum(kap[s]*coeff_pow(s,n-s) for s in range(1,n))
    return kap,coeff_pow
kap,_=mom_to_cum(m,K)
kap2=[2*x for x in kap]
# moments from cumulants
def cum_to_mom(kap,K):
    mm=[Fr(1)]+[Fr(0)]*K
    for n in range(1,K+1):
        # m_n = sum_s kap_s [z^{n-s}] M(z)^s with M built from mm so far
        tot=Fr(0)
        for s in range(1,n+1):
            poly=[Fr(1)]+[Fr(0)]*(n-s)
            for _ in range(s):
                new=[Fr(0)]*(n-s+1)
                for i in range(n-s+1):
                    if poly[i]==0: continue
                    for j in range(n-s+1-i): new[i+j]+=poly[i]*mm[j]
                poly=new
            tot+=kap[s]*poly[n-s]
        mm[n]=tot
    return mm
mpp=cum_to_mom(kap2,K)
print("m_k(mu+mu) =",[int(x) if x.denominator==1 else x for x in mpp[1:]])
print("c_k(W)     =",[x/2 for x in mpp[1:]])
rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 7)
a,b,c,d,e=[np.exp(2j*np.pi*rng.random()) for _ in range(5)]
P=magic(Mx.F85a(a,b,c,d,e))
blocks=[[0,3,1,5],[2,4,6,7]]
# block-level magic V and the stay projection
V=np.array([[sum(P[i,blocks[y][0]] for i in blocks[x]) for y in range(2)] for x in range(2)])
E0=V[0,0]  # should be central, = projection onto stay sector
print("E0 rank",round(np.trace(E0).real),"central:",all(np.allclose(E0@P[i,j],P[i,j]@E0) for i in range(8) for j in range(8)))
w,Vec=np.linalg.eigh(E0); S=Vec[:,w>0.5]
# u^{(x)}_{ab} = sum_j W_{(x,a),(j,b)} with a,b indexing position inside block (same labeling in both blocks)
U=np.zeros((8,8,4,4),complex)
for x in range(2):
    for ai in range(4):
        for bi in range(4):
            Ux=sum(P[blocks[x][ai],blocks[y][bi]] for y in range(2))
            U[4*x+ai,4*x+bi]=S.conj().T@Ux@S
print("theta_1 magic:",np.allclose(U.sum(axis=1),np.eye(4)), all(np.allclose(U[i,j]@U[i,j],U[i,j]) for i in range(8) for j in range(8)))
print("c_k(theta_1) =",[ck_fast(U,k) for k in range(1,K+1)])
# also full model
print("c_k(full)    =",[ck_fast(P,k) for k in range(1,K+1)])
# refined criterion: rho = theta_1 (+) theta_s * (theta_s o swap), on V_1 (+) V_s (x) V_s
w2,Vec2=np.linalg.eigh(np.eye(8)-E0); T=Vec2[:,w2>0.5]
Th=np.zeros((2,4,4,4,4),complex)   # Th[x,a,c] = theta_s(u^{(x)}_{ac})
for x in range(2):
    for ai in range(4):
        for ci in range(4):
            Ux=sum(P[blocks[x][ai],blocks[y][ci]] for y in range(2))
            Th[x,ai,ci]=T.conj().T@Ux@T
R=np.zeros((8,8,20,20),complex)
for x in range(2):
    for ai in range(4):
        for bi in range(4):
            R[4*x+ai,4*x+bi,:4,:4]=U[4*x+ai,4*x+bi]
            R[4*x+ai,4*x+bi,4:,4:]=sum(np.kron(Th[x,ai,ci],Th[1-x,ci,bi]) for ci in range(4))
print("rho magic:",np.allclose(R.sum(axis=1),np.eye(20)), all(np.allclose(R[i,j]@R[i,j],R[i,j]) for i in range(8) for j in range(8)))
print("c_k(rho)     =",[ck_fast(R,k) for k in range(1,K+1)])
