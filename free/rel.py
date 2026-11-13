from frames import *
rng=np.random.default_rng(11); r=lambda: np.exp(2j*np.pi*rng.random())
H=M.F85b(*[r() for _ in range(5)])
K=H[np.ix_([0,1,4,5],[0,1,2,3])]; L=H[np.ix_([2,3,6,7],[0,1,2,3])]
x1,F1=frame(magicpair(K,L))[2:]; x2,F2=frame(magicpair(L,K))[2:]
W=F1.conj().T@F2
# Pauli normalizer elements: m -> c_k m c_l on vec(M2) (row-major ravel)
def op(k,l): return np.kron(C[k],C[l].T)
best=max(abs(np.trace(op(k,l).conj().T@W))/4 for k in range(4) for l in range(4))
print("max overlap of W with Pauli normalizer:",best)
# decompose W in basis op(k,l)
coef=np.array([[np.trace(op(k,l).conj().T@W)/4 for l in range(4)] for k in range(4)])
print(np.round(np.abs(coef),4))
rho=lambda x: np.array([[ -np.trace(C[l]@x@C[k]@x.conj().T).real/2 for k in range(1,4)] for l in range(1,4)])
print("rho(x1)\n",np.round(rho(x1),4),"\nrho(x2)\n",np.round(rho(x2),4))
