import numpy as np, sys
sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw')
import matrices as Mx
from magic import magic, ck_fast
rng=np.random.default_rng(1); E=lambda t: np.exp(2j*np.pi*t)
a,b,c=E(rng.random()),E(rng.random()),E(rng.random())
F2,F4=Mx.fourier(2),Mx.fourier(4)
for nm,(q1,q2,q3) in {"q2=q1*q3":(a,a*c,c),"generic":(a,b,c)}.items():
    H=Mx.dita(F2,F4,[[1,1,1,1],[1,q1,q2,q3]])
    print(nm,"F2xF4 c4",ck_fast(magic(H),4),flush=True)
