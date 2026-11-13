# F85a model in free-wreath coordinates: W = Z2 wr_* F, F = H2+ acting on the 4 pairs
import numpy as np, sys, itertools
sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw')
import matrices as Mx
from magic import magic
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 7)
a,b,c,d,e=[np.exp(2j*np.pi*rng.random()) for _ in range(5)]
H=Mx.F85a(a,b,c,d,e); P=magic(H)
pairs=[[0,3],[1,5],[2,4],[6,7]]
# find labelings (orientation of each pair) with u_{(i,1),(j,1)} = u_{(i,2),(j,2)}
best=None
for ori in itertools.product([0,1],repeat=4):
    L=[p if o==0 else p[::-1] for p,o in zip(pairs,ori)]
    err=max(np.abs(P[L[i][0],L[j][0]]-P[L[i][1],L[j][1]]).max() for i in range(4) for j in range(4))
    print(ori,round(err,8))
    if err<1e-9 and best is None: best=L
L=best; print("labeling:",L)
I8=np.eye(8)
f=[sum(P[L[i][0],L[j][0]]-P[L[i][0],L[j][1]] for j in range(4)) for i in range(4)]
print("f_i involutions:",[np.allclose(x@x,I8) for x in f])
v=np.array([[P[L[i][0],L[j][0]]+P[L[i][0],L[j][1]] for j in range(4)] for i in range(4)])
print("f_i commutes with row i of v:",[all(np.allclose(f[i]@v[i,j],v[i,j]@f[i]) for j in range(4)) for i in range(4)])
print("f_i f_j commute?",[[np.allclose(f[i]@f[j],f[j]@f[i]) for j in range(4)] for i in range(4)])
np.save('wreathcoords_f.npy',np.array(f)); np.save('wreathcoords_v.npy',v)
# eigenvalues of products f_i f_j
for i,j in itertools.combinations(range(4),2):
    ev=np.linalg.eigvals(f[i]@f[j]); print(i,j,"args/2pi of f_i f_j:",np.round(np.sort(np.angle(ev)/(2*np.pi)),5))
print("params args/2pi:",[round(float(np.angle(x)/(2*np.pi)),5) for x in (a,b,c,d,e)])
