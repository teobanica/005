import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
from pert import top_ritz
q0=np.pi/6*np.array([0,2,1]); k=6; tot=0
for w in range(k):
    e=top_ritz(m.K6(*q0),k,w,b=48,thr=0.99)
    one=(e>1-1e-8).sum(); tot+=one
    print("sector %d: eigenvalue 1 x %d ; others above 0.99: %s"%(w,one,np.round(e[e<=1-1e-8],6)),flush=True)
print("f_6 at q0 =",tot)
