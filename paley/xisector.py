import numpy as np, sys
from eigcheck import paley1
from sectors import Aop
from eigq import eig as princ
def xisec(q,xi=1):
    H=paley1(q); N=q+1; A=Aop(H)
    w=np.exp(2j*np.pi*xi*np.arange(q)/q)
    B=[]
    for d in range(q):
        X=np.zeros((N,N),complex)
        for x in range(q): X[1+x,1+(x+d)%q]=w[x]
        B.append(X.ravel())
    X=np.zeros((N,N),complex); X[0,1:]=w; B.append(X.ravel())
    X=np.zeros((N,N),complex); X[1:,0]=w; B.append(X.ravel())
    B=np.array(B).T; B/=np.linalg.norm(B,axis=0)
    M=B.conj().T@A@B
    return np.sort(np.linalg.eigvalsh((M+M.conj().T)/2))
for q in map(int,sys.argv[1:]):
    N=q+1
    ev=xisec(q)
    n,res=princ(q); pv=sorted(set(np.round([v.real for v in res.values()],4)))
    nz=[round(e,4) for e in ev if abs(e)>1e-6]
    rest=[e for e in nz if min(abs(e-p) for p in pv)>1e-3]
    print("q=%d: nonzero xi-sector eigenvalues %d, principal %d distinct, others:"%(q,len(nz),len(pv)), rest, flush=True)
