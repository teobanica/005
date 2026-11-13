import numpy as np, sys
from piiinv import R_matrix
from piicheck import two_squares
np.set_printoptions(linewidth=200,suppress=True)
for q in (17,29,37):
    R=R_matrix(q).astype(float); a,b=two_squares(q); N=2*(q+1)
    alpha=32*((a*a-b*b)**2+1); beta=8*(q**3-5*q**2+7*q+5)
    w,V=np.linalg.eig(R)
    for name,lam in (("alpha",alpha),("64q",64*q)):
        E=V[:,np.abs(w-lam)<1e-6*N**3].real
        # rref-ish: orthonormal basis then show
        Q_,_=np.linalg.qr(E)
        print("q=%d %s=%d eigenvectors (rows t=0..5, cols ab=00,01,10,11):"%(q,name,lam))
        for k in range(E.shape[1]):
            v=Q_[:,k]; v=v/np.max(np.abs(v)); print(np.round(v.reshape(6,4),4))
