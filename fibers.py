import numpy as np, sys
sys.path.insert(0,'/home/claude/dita')
from magic import ck_model
from pauli import C
I2=np.eye(2); sx=np.array([[0,1],[1,0]]); sy=np.array([[0,-1j],[1j,0]]); sz=np.diag([1.,-1])
def model(x):
    U=np.zeros((4,4,4,4),complex)
    for a in range(4):
        for b in range(4):
            v=(C[a]@x@C[b]).ravel()/np.sqrt(2); U[a,b]=np.outer(v,v.conj())
    return U
t=0.7317
for nm,n in [("axis x",sx),("axis z",sz),("axis (1,1,0)",(sx+sy)/np.sqrt(2))]:
    x=np.cos(t)*I2+1j*np.sin(t)*n
    print(nm,[ck_model(model(x),k) for k in (1,2,3)])
