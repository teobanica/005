import numpy as np, sys
sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw')
import matrices as Mx
from magic import magic, ck_fast
rng=np.random.default_rng(1)
E=lambda t: np.exp(2j*np.pi*t)
a,b,c=E(rng.random()),E(rng.random()),E(rng.random())   # irrational-ish angles
F2,F4=Mx.fourier(2),Mx.fourier(4)
cases={"generic":(a,b,c),"q2=q1*q3":(a,a*c,c),"q2=q1^2":(a,a*a,c),"q3=q1":(a,b,a),
       "q3=q1^-1":(a,b,1/a),"q3=q1^3":(a,b,a**3),"q2=q1^2,q3=q1^3":(a,a**2,a**3),
       "q1q3=q2^2":(a,b,b*b/a)}
K=int(sys.argv[1]) if len(sys.argv)>1 else 3
for nm,(q1,q2,q3) in cases.items():
    for lab,H in [("F2xF4",Mx.dita(F2,F4,[[1,1,1,1],[1,q1,q2,q3]])),("F4xF2",Mx.dita(F4,F2,[[1,1],[1,q1],[1,q2],[1,q3]]))]:
        assert np.allclose(H@H.conj().T,8*np.eye(8))
        print(nm,lab,[ck_fast(magic(H),k) for k in range(1,K+1)],flush=True)
