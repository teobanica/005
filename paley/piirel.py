import numpy as np, sys
sys.path.insert(0,'/home/claude/kar/day6/bobcat')
from pmatrices import paley2
from piipure import blocks
for args in [(3,2),(17,)]:
    H=np.array(paley2(*args)).real.astype(int); N=H.shape[0]; m=N//2; q=m-1
    C=H.reshape(m,2,m,2)[:,0,:,0]-np.eye(m)   # recover C: H_(x0,y0)=C_xy+delta
    A,Aee,Aeo,Aoo,e,o=blocks(H)
    K=np.vstack([Aoo-64*q*np.eye(len(o)),Aeo]); _,s,Vt=np.linalg.svd(K); Z=Vt[np.sum(s>1e-6*s[0]):]
    full=np.zeros((Z.shape[0],N*N)); full[:,o]=Z; P=full.reshape(-1,m,2,m,2)[:,:,0,:,1]
    nrm=np.linalg.norm(P)
    tests={"CP-P^TC":lambda P:C@P-P.T@C,"CP+P^TC":lambda P:C@P+P.T@C,"CP-PC":lambda P:C@P-P@C,"CP+PC":lambda P:C@P+P@C,
           "CPC-qP^T":lambda P:C@P@C-q*P.T,"CPC+qP^T":lambda P:C@P@C+q*P.T,"CPC-qP":lambda P:C@P@C-q*P,"CPC+qP":lambda P:C@P@C+q*P,
           "P∘C":lambda P:P*C, "P∘(1-C^2..)":lambda P:P*(np.abs(C)==0)}
    print("q=%d"%q)
    for k,f in tests.items():
        r=np.sqrt(sum(np.linalg.norm(f(p))**2 for p in P))/nrm
        print("   %-12s %.3e"%(k,r))
