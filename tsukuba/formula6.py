from nongen import *
from fractions import Fraction
from math import gcd
E=lambda t: np.exp(2j*np.pi*t)
def inN_brute(M,N,Q,m,n):
    P=shifts(M,N,Q)
    if (M,N)==(2,3): w=[(1,1),(0,2)]*1; t1=[(1,1),(0,2)]; t2=[(1,2),(0,1)]
    else: t1=[(1,1),(0,1)]; t2=[(2,1),(0,1)]
    A=np.eye(M*N,dtype=complex)
    def pw(word,e):
        B=np.eye(M*N,dtype=complex)
        for (i,c) in word: B=P[(i,c)]@B
        return np.linalg.matrix_power(B,e) if e>=0 else np.linalg.matrix_power(np.linalg.inv(B),-e)
    A=pw(t1,m)@pw(t2,n)
    return scalar_blocks(A,M,N) is not None
def inN_formula(M,N,a,b,L,m,n):
    # p=e(a/L), q=e(b/L)
    if (M,N)==(2,3):
        al,be=2*a,2*b   # alpha=p^2, beta=q^2 exponents
        c1=al*(m+2*n)+be*(m-n); c2=al*(n-m)+be*(2*m+n)
    else:
        B=2*a-b; C=2*b-a
        c1=B*2*m+C*(2*m+2*n); c2=B*(2*m+2*n)+C*2*n
    return c1%L==0 and c2%L==0
bad=0
for (M,N) in [(2,3),(3,2)]:
    for (a,b,L) in [(1,1,3),(1,0,4),(1,2,5),(1,3,8),(2,1,6),(1,1,12),(5,7,12),(0,1,2)]:
        Q=[[1,1,1],[1,E(a/L),E(b/L)]] if (M,N)==(2,3) else [[1,1],[1,E(a/L)],[1,E(b/L)]]
        for m in range(-4,5):
            for n in range(-4,5):
                if inN_brute(M,N,Q,m,n)!=inN_formula(M,N,a,b,L,m,n): bad+=1
print("mismatches:",bad)
