from frames import *
from magic import ck_fast
rng=np.random.default_rng(11); r=lambda: np.exp(2j*np.pi*rng.random())
H=M.F85b(*[r() for _ in range(5)])
K=H[np.ix_([0,1,4,5],[0,1,2,3])]; L=H[np.ix_([2,3,6,7],[0,1,2,3])]
UK,UL,RKL,RLK=magicpair(K,K),magicpair(L,L),magicpair(K,L),magicpair(L,K)
def box(A,B):  # (A⊠B)_ab = sum_c A_ac ⊗ B_cb
    return np.array([[sum(np.kron(A[a,c],B[c,b]) for c in range(4)) for b in range(4)] for a in range(4)])
def dsum(*Ms):
    d=sum(m.shape[2] for m in Ms); out=np.zeros((4,4,d,d),complex); o=0
    for m in Ms:
        n=m.shape[2]; out[:,:,o:o+n,o:o+n]=m; o+=n
    return out
def blockdiag(A,B):  # 8x8 magic from two 4x4 copies
    d=A.shape[2]; W=np.zeros((8,8,d,d),complex); W[:4,:4]=A; W[4:,4:]=B; return W
if __name__=="__main__":
  for nm,(c0,c1) in {"sigma0":(UK,UL),"tau":(box(RKL,RLK),box(RLK,RKL)),
                   "sigma0+tau":(dsum(UK,box(RKL,RLK)),dsum(UL,box(RLK,RKL))),
                   "sigma1 (not even)":(RKL,RLK)}.items():
    W=blockdiag(c0,c1); print(nm,[ck_fast(W,k) for k in (1,2,3)],flush=True)
