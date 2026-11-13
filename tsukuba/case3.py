import numpy as np, sys
sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw')
import matrices as Mx
from magic import magic, ck_fast
E=lambda t: np.exp(2j*np.pi*t); a=E(0.1234567)
F2,F4=Mx.fourier(2),Mx.fourier(4)
cases={"q2=i,q3=q1":(a,1j,a),"q2=-1,q3=-q1":(a,-1,-a),"q2=q1q3*i":(a,a*E(0.3721)*1j,E(0.3721))}
for nm,(q1,q2,q3) in cases.items():
    H=Mx.dita(F2,F4,[[1,1,1,1],[1,q1,q2,q3]])
    print(nm,[ck_fast(magic(H),k) for k in range(1,5)],flush=True)
