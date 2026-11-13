import numpy as np, sys
from blocks import *
sx=np.array([[0,1],[1,0]]); sy=np.array([[0,-1j],[1j,0]]); sz=np.diag([1.,-1])
def blochvec(psi):
    psi=psi/np.linalg.norm(psi); r=np.outer(psi,psi.conj())
    return np.real([np.trace(r@s) for s in (sx,sy,sz)])
def rotvec(U):  # SO(3) of U: R e = bloch(U |.>)...
    U=U/np.sqrt(np.linalg.det(U)+0j)
    R=np.zeros((3,3))
    for k,s in enumerate((sx,sy,sz)):
        M=U@s@U.conj().T
        R[:,k]=np.real([np.trace(M@t)/2 for t in (sx,sy,sz)])
    return R
def config(H, rm, cm):
    X,Y,W=normalize(to_blocks(H,rm,cm)); q=len(W)
    a=[rotvec(X[i]).T@np.array([0,0,1.]) for i in range(q)]
    b=[rotvec(Y[j]/np.sqrt(2))@np.array([0,0,1.]) for j in range(q)]
    R=[[rotvec(W[i][j]) for j in range(q)] for i in range(q)]
    return a,b,R,W
def k6_axis(H, rm, cm):
    a,b,R,W=config(H,rm,cm)
    Om=W[1][1]; w,V=np.linalg.eig(Om)
    # axis n of Om: Om = -1/2 + i sqrt3/2 Lam  -> Lam = (Om+1/2)/(i sqrt3/2)
    Lam=(Om+0.5*np.eye(2))/(1j*np.sqrt(3)/2)
    n=np.real([np.trace(Lam@s)/2 for s in (sx,sy,sz)])
    chk=max(abs(np.dot(a[i],R[i][j]@b[j])) for i in range(3) for j in range(3))
    return n, np.array([n@x for x in a]), np.array([n@x for x in b]), chk, np.linalg.norm(Lam@Lam-np.eye(2))
if __name__=="__main__":
    np.set_printoptions(precision=4,suppress=True)
    rng=np.random.default_rng(5); r=lambda: np.exp(2j*np.pi*rng.random())
    for nm,H in [('K6',M.K6(0.7,1.1,0.4)),('K6b',M.K6(1.3,0.2,2.0)),('H6',M.H6(r())),('H6b',M.H6(r())),('BF6',M.BF6(0)),('BN6',M.BN6(np.exp(2j*np.pi*0.37))),('BN6b',M.BN6(np.exp(2j*np.pi*0.3)))]:
        for rm,cm in h2_structures(H)[:2]:
            n,ca,cb,chk,lam=k6_axis(H,rm,cm)
            print(nm,'cos(n,a)',ca,'cos(n,b)',cb,' sum sq',round(ca@ca,4),round(cb@cb,4),' orth chk',f'{chk:.1e}', f'{lam:.1e}')
