import sys,time; sys.path.insert(0,'/home/claude/work/night'); sys.path.insert(0,'/home/claude/work/spec')
import numpy as np, matrices as m, hw, spectro
# dimension check of X_k
for N,k in [(5,2),(5,3),(6,3),(4,4)]:
    L,Lt,M=hw.lowering(N,k); n=N**k
    A=np.stack([L(np.eye(n)[:,j]) for j in range(n)],axis=1)
    print(N,k,"dim X_k",n-np.linalg.matrix_rank(A),"d_k",spectro.dims(N,k)[k])
H=m.T6()
for k in (3,4,5):
    t=time.time(); w,res,nmv,leak=hw.top_phi(H,k)
    print("T6 k=%d"%k, np.round(w,8), res.max(), nmv, "leak %.1e"%leak, round(time.time()-t,1),flush=True)
