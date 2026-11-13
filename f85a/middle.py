import numpy as np, sys
sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw')
import matrices as Mx
from magic import magic
rng=np.random.default_rng(7); a,b,c,d,e=[np.exp(2j*np.pi*rng.random()) for _ in range(5)]
H=Mx.F85a(a,b,c,d,e)
K=H[[0,1,3,5]][:,[0,1,2,3]]; L=H[[2,4,6,7]][:,[0,1,2,3]]
pr=lambda v: np.outer(v,v.conj())/np.vdot(v,v).real
P0=pr(L[0].conj())+pr(L[1].conj()); P1=pr(L[0])+pr(L[0]*K[2].conj())
S0,S1=2*P0-np.eye(4),2*P1-np.eye(4)
ev=np.linalg.eigvals(S0@S1)
pred=[c**2,c**-2,(b/a)**2,(b/a)**-2]
print("P0,P1 projections:",np.allclose(P0@P0,P0),np.allclose(P1@P1,P1))
print("eig S0S1:",np.sort_complex(np.round(ev,8))); print("pred    :",np.sort_complex(np.round(pred,8)))
# check full model middle projections on P- block via W
W=magic(H)
