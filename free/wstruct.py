from frames import *
def getW(seed):
    rng=np.random.default_rng(seed); r=lambda: np.exp(2j*np.pi*rng.random())
    pars=[r() for _ in range(5)]
    H=M.F85b(*pars); K=H[np.ix_([0,1,4,5],[0,1,2,3])]; L=H[np.ix_([2,3,6,7],[0,1,2,3])]
    x1,F1=frame(magicpair(K,L))[2:]; x2,F2=frame(magicpair(L,K))[2:]
    return pars,x1,x2,F1,F2,F1.conj().T@F2
def schmidt(W):  # operator Schmidt coefficients w.r.t. C^2 ⊗ C^2 (row-major vec of M2)
    R=W.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
    return np.linalg.svd(R,compute_uv=False)
if __name__=="__main__":
 for sd in (11,12):
     pars,x1,x2,F1,F2,W=getW(sd)
     print("seed",sd,"Schmidt coeffs of W:",np.round(schmidt(W),4))
     print("   of F1:",np.round(schmidt(F1),4),"  of F2:",np.round(schmidt(F2),4))
     pb=np.array([C[m].ravel()/np.sqrt(2) for m in range(4)]).T
     print("   |W| in Bell basis:\n",np.round(np.abs(pb.conj().T@W@pb),4))
