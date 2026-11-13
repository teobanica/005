# eigenvalue m_lambda of A = N^3 T_2 on X_lambda, Paley I, q prime = 3 mod 4, computed from one entry (no big matrices)
import sys, numpy as np
def chi_table(q):
    sq={x*x%q for x in range(1,q)}
    return np.array([0]+[1 if d in sq else -1 for d in range(1,q)])
def eig(q):
    chi=chi_table(q); N=q+1
    # Paley I rows: index 0 = infinity, 1+x for x in F_q. H = I + S
    H=np.zeros((N,N),int); H[0,:]=1; H[1:,0]=-1
    for x in range(q):
        for z in range(q): H[1+x,1+z]=chi[(z-x)%q]+(1 if z==x else 0)
    H[0,0]=1
    assert (H@H.T==N*np.eye(N)).all()
    g=next(a for a in range(2,q) if len({pow(a,k,q) for k in range(1,q)})==q-1); h=g*g%q; n=(q-1)//2
    log={pow(h,k,q):k for k in range(n)}
    sqd=np.array([0]+[d if d in log else (-d)%q for d in range(1,q)])
    # S row at (x0,y0)=(0,1) i.e. indices 1,2
    R=(H[:,None,:]*H[None,:,:]).reshape(N*N,N)
    srow=(R@(H[1]*H[2]))**2      # A[(1,2),(k,l)]
    res={}
    for t in range(1,n):
        lam=np.zeros(q,complex)
        for d in range(1,q): lam[d]=np.exp(2j*np.pi*t*log[sqd[d]]/n)
        X=np.zeros((N,N),complex)
        for x in range(q):
            X[1+x,1:]=[lam[(y-x)%q] if y!=x else 0 for y in range(q)]
        val=(srow@X.ravel())/X[1,2]
        res[t]=val
    return n,res
if __name__=="__main__":
 for q in [int(a) for a in sys.argv[1:]]:
     n,res=eig(q); N=q+1
     from math import gcd
     print("q=%d N=%d (q-1)/2=%d:"%(q,N,n), ["order %d: %.4f"%(n//gcd(t,n), res[t].real) for t in range(1,n) if t<=n//2+1], flush=True)
