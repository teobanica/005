import numpy as np
from scipy.optimize import least_squares
def rot(n, ang):
    n=n/np.linalg.norm(n); K=np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
    return np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*K@K
def nvec(th,ph): return np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
def residual(v, q, m, th=None):
    # v = [th?, ph, alpha_1..q-1, beta_1..q-1]
    if th is None: th, v = v[0], v[1:]
    ph=v[0]; al=v[1:q]; be=v[q:2*q-1]
    n=nvec(th,ph); R=rot(n, 4*np.pi*m/q)
    a=[np.array([0,0,1.])]+[np.array([0,np.sin(t),np.cos(t)]) for t in al]
    b=[np.array([1.,0,0])]+[np.array([np.cos(t),np.sin(t),0]) for t in be]
    out=[]
    for i in range(1,q):
        for j in range(1,q):
            out.append(a[i]@np.linalg.matrix_power(R,(i*j)%q)@b[j])
    return np.array(out)
def explore(q,m,th,trials=200,seed=0):
    rng=np.random.default_rng(seed); sols=[]
    for t in range(trials):
        v0=rng.uniform(0,2*np.pi,2*q-1)
        r=least_squares(lambda v: residual(v,q,m,th), v0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if np.abs(r.fun).max()<1e-11:
            J=r.jac; rk=np.linalg.matrix_rank(J,1e-8)
            sols.append((r.x, J.shape[1]-rk))
    return sols
if __name__=="__main__":
    for q,m in [(3,1),(4,1),(5,1),(5,2)]:
        for th in [0.0, 0.3, 1.0]:
            s=explore(q,m,th,100)
            dims=sorted(set(d for _,d in s))
            print(f"q={q} m={m} theta={th}: {len(s)}/100 solved, local dims (fixed n) {dims}")
