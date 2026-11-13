import numpy as np, sys
sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw')
import matrices as Mx
from magic import magic, find_outer
rng=np.random.default_rng(2); r=lambda: np.exp(2j*np.pi*rng.random())
H=Mx.F85a(*[r() for _ in range(5)]); W=magic(H)
P2=find_outer(W,2); P4=find_outer(W,4)
print("2 blocks of 4:",P2); print("4 blocks of 2:",P4)
from magic import ck_fast
import itertools
B0=[0,3,1,5]; B1=[2,4,6,7]
def inner(order0,order1,i):
    blocks=[order0,order1]; Bi=blocks[i]
    U=np.zeros((4,4,8,8),complex)
    for a in range(4):
        for b in range(4):
            U[a,b]=sum(W[Bi[a],blocks[j][b]] for j in range(2))
    return U
for nm,o1 in [("aligned",B1),("misaligned",[2,6,4,7])]:
    for i in (0,1):
        U=inner(B0,o1,i)
        ok=all(np.allclose(U[a,b]@U[a,b],U[a,b]) for a in range(4) for b in range(4))
        print(nm,"copy",i,"magic:",ok,"c_k:",[ck_fast(U,k) for k in (1,2,3,4)])
# middle-level projections of each copy (aligned labels: pairs {0,1},{2,3})
U0=inner(B0,B1,0); U1=inner(B0,B1,1)
def mid(U):  # y_{00} = sum over beta,beta' of u_{(0beta),(0beta')}
    return sum(U[a,b] for a in (0,1) for b in (0,1))
P0,P1=mid(U0),mid(U1)
print("P0 proj:",np.allclose(P0@P0,P0)," P1 proj:",np.allclose(P1@P1,P1)," [P0,P1]=",np.abs(P0@P1-P1@P0).max())
# inner involutions of copy 0: x^{(alpha)} entries u_{(alpha 0),(alpha' 0)}... check commutation of all copy-0 entries with all copy-1 entries
M=max(np.abs(U0[a,b]@U1[c,d]-U1[c,d]@U0[a,b]).max() for a in range(4) for b in range(4) for c in range(4) for d in range(4))
print("max commutator between copies:",M)
P0=U0[0,0]+U0[0,1]; P1=U1[0,0]+U1[0,1]
print("P0 proj:",np.allclose(P0@P0,P0),np.allclose(P0,U0[1,0]+U0[1,1])," P1 proj:",np.allclose(P1@P1,P1)," [P0,P1]=",np.abs(P0@P1-P1@P0).max())
Wd=np.zeros((8,8,8,8),complex); Wd[:4,:4]=U0; Wd[4:,4:]=U1
print("pair of copies c_k:",[ck_fast(Wd,k) for k in (1,2,3,4)])
