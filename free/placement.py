import numpy as np
from scipy.stats import special_ortho_group
I2=np.eye(2); s=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1.,-1])]
t=[1j*m for m in s]
def kron(*a):
    out=np.eye(1)
    for m in a: out=np.kron(out,m)
    return out
y=special_ortho_group.rvs(3,random_state=1); z=special_ortho_group.rvs(3,random_state=2)
# (theta_y ⊗ theta_z)Delta(a_ij) on qubits 1,2,3,4
T=lambda i,j: sum(y[i,k]*z[k,j]*kron(t[i],t[k],t[k],t[j]) for k in range(3))
# Bell basis on middle qubits (2,3): eigenvectors of tau_k⊗tau_k
B=np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,1,-1,0]]).T/np.sqrt(2)
U=kron(I2,B,I2)   # columns: q1 ⊗ bell(23) ⊗ q4
P=np.zeros((16,16)); 
for a in range(2):
  for m in range(4):
    for b in range(2): P[(a*4+m)*2+b, m*4+a*2+b]=1   # reorder to bell ⊗ (q1,q4)
V=[np.diag([np.real(B[:,m].conj()@kron(t[k],t[k])@B[:,m]) for k in range(3)]) for m in range(4)]
ok=True
for i in range(3):
  for j in range(3):
    Mt=P.T@U.conj().T@T(i,j)@U@P
    target=np.zeros((16,16),complex)
    for m in range(4): target[4*m:4*m+4,4*m:4*m+4]=(y@V[m]@z)[i,j]*kron(t[i],t[j])
    ok&=np.allclose(Mt,target)
print("(theta_y⊗theta_z)Δ = ⊕_v theta_{yvz}:",ok,"  V elements det:",[round(np.linalg.det(v)) for v in V])
