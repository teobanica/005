# brute-force check over GF(27): build Paley I, full A=N^3 T_2, check X_psi eigenvectors and orbit trace
import numpy as np, sys
from trace_gf import field
p,r=3,3
q,exp,log,D,w=field(p,r); N=q+1; n=(q-1)//2
add=lambda a,b: ((D[a]+D[b])%p)@w; sub=lambda a,b: ((D[a]-D[b])%p)@w
chi=lambda u: 0 if u==0 else (1 if log[u]%2==0 else -1)
H=np.zeros((N,N),int); H[0,:]=1; H[1:,0]=-1
for x in range(q):
    for z in range(q): H[1+x,1+z]=chi(sub(z,x))+(1 if z==x else 0)
assert (H@H.T==N*np.eye(N)).all(); print("Paley I over GF(27) Hadamard: ok")
R=(H[:,None,:]*H[None,:,:]).reshape(N*N,N); S=R@R.T; A=S*S
tot=0
for t in range(1,n):
    X=np.zeros((N,N),complex)
    for x in range(q):
        for y in range(q):
            if x!=y:
                d=sub(y,x); k=log[d]//2 if log[d]%2==0 else log[sub(0,d)]//2
                X[1+x,1+y]=np.exp(2j*np.pi*t*k/n)
    v=X.ravel(); Av=A@v; m=(v.conj()@Av)/(v.conj()@v)
    print(t, "residual %.1e"%(np.linalg.norm(Av-m*v)/np.linalg.norm(v)), "m=%.4f"%m.real); tot+=m.real
print("orbit trace (order 13) =",round(tot))
