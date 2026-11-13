import sys, numpy as np, scipy.linalg as sl
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
from phis import Tk
from lwsector import lw_fixed
H=m.F6rs(np.exp(0.4j),np.exp(1.9j)); N=6
e,V=np.linalg.eigh(Tk(H,2)); F2=V[:,e>1-1e-9]                       # Fix_2, 4-dim
IJ=sl.orth(np.array([np.eye(N).reshape(-1),np.ones(N*N)]).T)
R=F2-IJ@(IJ.conj().T@F2); R=sl.orth(R)                               # Fix_2 minus span(I,J), 2-dim
Ds=[]
for j in range(R.shape[1]):
    D=R[:,j].reshape(N,N); Ds.append(D)
# hermitian basis of span(Ds) (closed under adjoint?)
herm=[(D+D.conj().T)/2 for D in Ds]+[(D-D.conj().T)/2j for D in Ds]
Hb=sl.orth(np.array([h.reshape(-1) for h in herm]).T); print("hermitian dim:",Hb.shape[1])
D1,D2=[Hb[:,j].reshape(N,N) for j in range(2)]
print("D1 real?",np.abs(D1.imag).max()<1e-12, " D1 pattern:\n",np.round((D1/np.abs(D1).max()).real,2))
print("D2 pattern:\n",np.round((D2/np.abs(D2).max()).real,2))
def comp(Vk,D,k):
    X=Vk.T.reshape(Vk.shape[1],N,-1)                                   # leg 1 vs rest
    return np.einsum('iac,ab,jbc->ij',X.conj(),D,X)
def best_angle(K1,K2):
    th=np.linspace(0,2*np.pi,721); vals=[np.linalg.eigvalsh(np.cos(t)*K1+np.sin(t)*K2).min() for t in th]
    i=int(np.argmax(vals)); return vals[i], th[i]
store={}
for k in [3,4,5]:
    for w in range(k):
        Vk,_,_=lw_fixed(H,k,w,b=16)
        if Vk.shape[1]==0: continue
        K1,K2=comp(Vk,D1,k),comp(Vk,D2,k); v,t=best_angle(K1,K2)
        print("level %d sector %d (dim %d): best min eigenvalue %.4f at angle %.3f"%(k,w,Vk.shape[1],v,t),flush=True)
D6=np.load("lw_dita23_k6.npz")
for w in range(6):
    Vk=D6["w%d"%w]; K1,K2=comp(Vk,D1,6),comp(Vk,D2,6); v,t=best_angle(K1,K2)
    print("level 6 sector %d (dim %d): best min eigenvalue %.4f at angle %.3f"%(w,Vk.shape[1],v,t),flush=True)
