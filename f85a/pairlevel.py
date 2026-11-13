import numpy as np, sys
sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw')
import matrices as Mx
from magic import magic, ck_fast
rng=np.random.default_rng(7); a,b,c,d,e=[np.exp(2j*np.pi*rng.random()) for _ in range(5)]
H=Mx.F85a(a,b,c,d,e); W=magic(H)
pairs=[[0,3],[1,5],[2,4],[6,7]]
Y=np.zeros((4,4,8,8),complex)
for p,I in enumerate(pairs):
    for q,J in enumerate(pairs):
        vals=[sum(W[i,j] for i in I) for j in J]
        assert np.allclose(vals[0],vals[1]); Y[p,q]=vals[0]
print("magic:",all(np.allclose(Y[p,q]@Y[p,q],Y[p,q]) for p in range(4) for q in range(4)), np.allclose(Y.sum(axis=1),np.eye(8)))
print("pair-level c_k:",[ck_fast(Y,k) for k in (1,2,3,4,5)])
E=Y[0,0]+Y[0,1]
x0=Y[0,0]+Y[0,2]; x1=Y[2,0]+Y[2,2]
g0,g1=2*x0-np.eye(8),2*x1-np.eye(8)
print("[E,x0],[E,x1]:",np.abs(E@x0-x0@E).max(),np.abs(E@x1-x1@E).max())
def blockeig(X,P):
    w,V=np.linalg.eigh(P); B=V[:,w>0.5]; return np.linalg.eigvals(B.conj().T@X@B)
T=g0@g1
for nm,P in [("E",E),("1-E",np.eye(8)-E)]:
    ev=blockeig(T,P); print(nm,"eig g0g1:",np.round(ev,6),"args/2pi:",np.round(np.angle(ev)/(2*np.pi),6))
print("params args/2pi:",{k:round(float(np.angle(v)/(2*np.pi)),6) for k,v in dict(a=a,b=b,c=c,d=d,e=e).items()})
