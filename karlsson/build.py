import numpy as np, sys
from scipy.linalg import expm
from fam import *
sys.path.insert(0,'/home/claude/jw/jw')
F2=np.array([[1,1],[1,-1]],dtype=complex); D=lambda z: np.diag([1,z])
sx=np.array([[0,1],[1,0]]); sy=np.array([[0,-1j],[1j,0]]); sz=np.diag([1.,-1])
def build(q,m,th,ph,al,be,sgn):
    n=nvec(th,ph); L=n[0]*sx+n[1]*sy+n[2]*sz
    Om=expm(sgn*2j*np.pi*m*L/q)
    X=[np.eye(2)]+[0.5*F2@D(np.exp(1j*t))@F2 for t in al]
    Y=[F2]+[D(np.exp(1j*t))@F2 for t in be]
    return np.block([[X[i]@np.linalg.matrix_power(Om,(i*j)%q)@Y[j] for j in range(q)] for i in range(q)])
def find(q,m,th,seed=0):
    for s in explore(q,m,th,30,seed):
        v=s[0]; ph=v[0]; al=v[1:q]; be=v[q:]
        for sgn in (1,-1):
            for conjal in (1,-1):
                H=build(q,m,th,ph,conjal*al,be,sgn)
                if np.abs(np.abs(H)-1).max()<1e-9: return H,(ph,conjal*al,be,sgn)
    return None,None
if __name__=="__main__":
    for q,m,th in [(3,1,0.7),(4,1,0.7),(5,1,0.7),(5,2,0.7)]:
        H,p=find(q,m,th)
        if H is None: print(q,m,'no build'); continue
        N=2*q; print(q,m,'unimod',np.abs(np.abs(H)-1).max(),'unitary',np.abs(H@H.conj().T-N*np.eye(N)).max())
        np.save(f'H_{q}_{m}.npy',H)
