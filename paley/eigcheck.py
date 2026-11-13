import numpy as np, sys
from trace_fast import isprime
def paley1(q):
    sq=np.zeros(q,bool); sq[(np.arange(1,q)**2)%q]=True
    h=np.where(sq,1,-1); h[0]=1; N=q+1
    H=np.zeros((N,N),int); H[0,:]=1; H[1:,0]=-1
    a=np.arange(q); H[1:,1:]=h[(a[None,:]-a[:,None])%q]
    return H
for q in map(int,sys.argv[1:]):
    H=paley1(q); N=q+1; assert (H@H.T==N*np.eye(N)).all()
    R=(H[:,None,:]*H[None,:,:]).reshape(N*N,N).astype(float); S=R@R.T; A=S*S
    ev=np.linalg.eigvalsh(A)
    target=32*(q-1)
    mult=int(np.sum(np.abs(ev-target)<1e-6*target))
    rat=sorted(set(int(round(e)) for e in ev if abs(e-round(e))<1e-6))
    print("q=%d q mod 8=%d  mult of 32(q-1)=%d : %d ; integer eigenvalues:"%(q,q%8,target,mult), [(m, int(np.sum(np.abs(ev-m)<1e-6*max(1,m)))) for m in rat], flush=True)
