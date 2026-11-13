import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
from pert import top_ritz
q0=np.pi/6*np.array([0,2,1]); d=np.array([1.0,0,0]); k=6; eps=0.01; tot=0
for w in range(k):
    t=time.time(); e=top_ritz(m.K6(*(q0+eps*d)),k,w,b=48,thr=0.995); a=np.sort((1-e)/eps**2); tot+=len(a)
    print("level 6 sector %d: %d eigenvalues near 1, a_i = %s  [%.0f s]"%(w,len(a),np.round(a,3),time.time()-t),flush=True)
print("total",tot,"(f_6 = 66 expected)")
