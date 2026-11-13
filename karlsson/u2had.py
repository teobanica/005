import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import expm
sx=np.array([[0,1],[1,0]]); sy=np.array([[0,-1j],[1j,0]]); sz=np.diag([1.,-1]); I2=np.eye(2)
def U(p): return expm(1j*(p[0]*I2+p[1]*sx+p[2]*sy+p[3]*sz))
def W_of(v,q):
    W=[[I2]*q for _ in range(q)]; W=[list(r) for r in W]; k=0
    for i in range(1,q):
        for j in range(1,q):
            W[i][j]=U(v[4*k:4*k+4]); k+=1
    return W
def res(v,q):
    W=W_of(v,q); out=[]
    for i in range(q):
        for k in range(i+1,q):
            S=sum(W[i][j]@W[k][j].conj().T for j in range(q)); out+= list(S.real.ravel())+list(S.imag.ravel())
    return np.array(out)
def classify(W,q):
    flat=[W[i][j] for i in range(1,q) for j in range(1,q)]
    comm=max(np.abs(a@b-b@a).max() for a in flat for b in flat)
    d=np.array([[np.linalg.det(W[i][j]) for j in range(q)] for i in range(q)])
    # rank-one test for dets: d_ij d_00 = d_i0 d_0j (all ones on border) -> d_ij == 1 up to sqrt? allow phases lambda_i^2 mu_j^2: border is 1 so need d_ij==1
    return comm, np.abs(d-1).max()
if __name__=="__main__":
    q=4; rng=np.random.default_rng(0); stats=[]
    for t in range(300):
        r=least_squares(lambda v: res(v,q), rng.normal(0,2,4*(q-1)**2), xtol=1e-15,ftol=1e-15,gtol=1e-15)
        if np.abs(r.fun).max()<1e-10:
            c,d=classify(W_of(r.x,q),q); nul=r.jac.shape[1]-np.linalg.matrix_rank(r.jac,1e-8)
            stats.append((c<1e-8, d<1e-8, nul))
    from collections import Counter
    print(Counter(stats))
