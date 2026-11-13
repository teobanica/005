from nongen import *
from lattice import setup
E=lambda t: np.exp(2j*np.pi*t)
M,N=3,2
p,q=E(0.1234),E(0.3711)
Q=[[1,1],[1,p],[1,q]]
P=shifts(M,N,Q)
def op(word,x=0):
    A=np.eye(M*N,dtype=complex)
    for (i,c) in word: A=P[((i+x)%M,c)]@A
    return A
t={1:[(1,1),(0,1)],2:[(2,1),(0,1)]}   # t_i = (-1)^{(0)} 1^{(i)}, -1=1 in Z_2
def diag(A): return np.round(np.angle(np.diag(A))/(2*np.pi),6)
# exponents of p,q: solve diag phase = a*arg p + b*arg q
def expo(A):
    ph=np.angle(np.diag(A))/(2*np.pi)
    out=[]
    for v in ph:
        best=None
        for a in range(-4,5):
            for b in range(-4,5):
                if abs(((v-a*0.1234-b*0.3711+0.5)%1)-0.5)<1e-7: best=(a,b)
        out.append(best)
    return out
for i in (1,2):
    for x in range(3):
        print("x=",x,"· t_%d"%i," block-diag exponents (p,q):",expo(op(t[i],x)))
