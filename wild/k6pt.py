import sys, numpy as np, itertools
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
from phis import Tk
def ck(H,k):
    e=np.linalg.eigvalsh(Tk(H,k)); return int((e>1-1e-8).sum())
found=[]
for a,b,c in itertools.product(range(6),repeat=3):
    q=np.pi/6*np.array([a,b,c])
    try: H=m.K6(*q)
    except Exception: continue
    if not np.all(np.isfinite(H)): continue
    if np.abs(np.abs(H)-1).max()>1e-8 or np.abs(H@H.conj().T-6*np.eye(6)).max()>1e-8: continue
    if ck(H,2)==4 and ck(H,3)==18: found.append((a,b,c))
print(len(found),"points with c_2,c_3 = 4,18:",found)
for p in found[:4]:
    H=m.K6(*(np.pi/6*np.array(p))); print(p,"c_4 =",ck(H,4))
