from frames import *
from magic import ck_fast
from wstruct import getW, schmidt
pars,x1,x2,F1,F2,W=getW(11)
Px=pauli(x1)
PS=np.transpose(pauli(x1.conj()),(1,0,2,3))     # pi_{xbar} o S
RKL=np.einsum('ij,abjk,lk->abil',F1,Px,F1.conj())
RLK_pred=np.transpose(RKL,(1,0,3,2))            # (R_ba)^T
K_=None
Mm=F1.conj().T@F1.conj()
print("M=F1^* F1bar symmetric:",np.allclose(Mm,Mm.T)," Schmidt:",np.round(schmidt(Mm),4))
pb=np.array([C[m].ravel()/np.sqrt(2) for m in range(4)]).T
print("|M| in Bell basis\n",np.round(np.abs(pb.conj().T@Mm@pb),4))
print("check pi^T o S = Ad(F1bar) pi_xbar o S :",np.allclose(RLK_pred,np.einsum('ij,abjk,lk->abil',F1.conj(),PS,F1.T)))
def blk(A,B):
    Wm=np.zeros((8,8,4,4),complex); Wm[:4,:4]=A; Wm[4:,4:]=B; return Wm
print("standard frame pair (pi_x, pi_xbar o S):",[ck_fast(blk(Px,PS),k) for k in (1,2,3,4)],flush=True)
