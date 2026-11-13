import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
from pert import top_ritz
q0=np.pi/6*np.array([0,2,1]); d=np.array([1.0,0,0]); k=int(sys.argv[1]); allA=[]
for w in range(k):
    res=[]
    for eps in [1e-3,2e-3]:
        e=top_ritz(m.K6(*(q0+eps*d)),k,w,b=48,thr=1-5e-4); res.append(np.sort((1-e)/eps**2))
    allA+=list(res[0])
    print("level %d sector %d: %d branches; a_i = %s ; max change eps 1e-3 -> 2e-3: %.1e"%(k,w,len(res[0]),np.round(res[0],3),np.abs(res[0]-res[1]).max() if len(res[0])==len(res[1]) else -1),flush=True)
print("level %d: %d branches, min a = %.4f"%(k,len(allA),min(allA)))
