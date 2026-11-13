import numpy as np, sys
sys.path.insert(0,'/home/claude/kar/day6/bobcat')
from pmatrices import paley2
from piipure import blocks
for args in [(3,2),(17,)]:
    H=np.array(paley2(*args)).real.astype(int); N=H.shape[0]; m=N//2; q=m-1
    A,Aee,Aeo,Aoo,e,o=blocks(H)
    K=np.vstack([Aoo-64*q*np.eye(len(o)),Aeo])
    _,s,Vt=np.linalg.svd(K); Z=Vt[s.size-np.sum(s<1e-6*s[0]):] if False else Vt[np.sum(s>1e-6*s[0]):]
    print("q=%d pure odd dim %d"%(q,Z.shape[0]))
    full=np.zeros((Z.shape[0],N*N)); full[:,o]=Z
    X=full.reshape(-1,m,2,m,2)
    P=X[:,:,0,:,1]; Q=X[:,:,1,:,0]
    print("  ||Q+P^T||/||P||:", np.linalg.norm(Q+P.transpose(0,2,1))/np.linalg.norm(P), " ||Q-P^T||:", np.linalg.norm(Q-P.transpose(0,2,1))/np.linalg.norm(P))
    # symmetric/antisymmetric content of P
    print("  P sym part frac:", np.linalg.norm(P+P.transpose(0,2,1))/2/np.linalg.norm(P), " diag frac:", np.linalg.norm(np.einsum('kii->ki',P))/np.linalg.norm(P))
    # is P in span of something like M with C M + M C structure? test rowsums
    print("  row sums frac:", np.linalg.norm(P.sum(2))/np.linalg.norm(P), " col sums:", np.linalg.norm(P.sum(1))/np.linalg.norm(P))
