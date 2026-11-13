from frames import *
from magic import ck_fast
rng=np.random.default_rng(21); x=su2(rng.normal(size=3)); y=su2(rng.normal(size=3))
def blk(A,B):
    Wm=np.zeros((8,8,4,4),complex); Wm[:4,:4]=A; Wm[4:,4:]=B; return Wm
Px=pauli(x); PyS=np.transpose(pauli(y),(1,0,2,3))
print("anti pair (pi_x, pi_y o S), generic x,y:",[ck_fast(blk(Px,PyS),k) for k in (1,2,3,4)],flush=True)
