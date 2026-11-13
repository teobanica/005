import numpy as np, itertools, sys
from pairs import *
from scipy.optimize import least_squares
def defect(H):
    N=H.shape[0]; rows=[]
    for i in range(N):
        for j in range(i+1,N):
            # sum_k H_ik conj(H_jk) (R_ik - R_jk) = 0 : complex eq -> 2 real
            c=np.zeros((N,N),dtype=complex)
            for k in range(N):
                c[i,k]+=H[i,k]*np.conj(H[j,k]); c[j,k]-=H[i,k]*np.conj(H[j,k])
            rows.append(c.real.ravel()); rows.append(c.imag.ravel())
    A=np.array(rows); return N*N-np.linalg.matrix_rank(A,1e-9)-(2*N-1)
q=5; z=np.exp(2j*np.pi/5); F=np.array([[z**(i*j) for j in range(q)] for i in range(q)])
if __name__=="__main__":
    s=(1,2,3,4); t=(1,2,4,3); L=F[np.ix_((0,)+s,(0,)+t)]
    A=np.angle(F*L.conj())
    found=[]
    for th in [0.5,0.8,1.2]:
        rng=np.random.default_rng(1)
        for tr in range(20):
            r=least_squares(lambda v: resid(v,A,th), rng.uniform(0,2*np.pi,2*q-1), xtol=1e-15,ftol=1e-15,gtol=1e-15)
            if np.abs(r.fun).max()<1e-11:
                # full jacobian including theta
                f=lambda w: resid(w[1:],A,w[0]); w0=np.r_[th,r.x]; h=1e-6
                J=np.array([(f(w0+h*e)-f(w0-h*e))/(2*h) for e in np.eye(len(w0))]).T
                nul=len(w0)-np.linalg.matrix_rank(J,1e-6)
                for sgn in (1,-1):
                    for ca in (1,-1):
                        H=build(F,L,th,r.x[0],ca*r.x[1:q],r.x[q:],sgn)
                        if np.abs(np.abs(H)-1).max()<1e-9:
                            c2=ck(H,2)[0]
                            if c2==2: found.append((th,r.x,H,nul))
                            break
                    else: continue
                    break
    print(len(found),'c2=2 solutions; local dims (incl theta):', sorted(set(f[3] for f in found)))
    th,x,H,nul=found[0]
    print('unitarity',np.abs(H@H.conj().T-10*np.eye(10)).max(),'unimod',np.abs(np.abs(H)-1).max())
    print('defect',defect(H),' c3',ck(H,3)[0])
    np.save('H10.npy',H); np.save('H10x.npy',np.r_[th,x])
