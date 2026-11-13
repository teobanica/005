import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
from lwsector import lw_fixed
H=m.F6rs(np.exp(0.4j),np.exp(1.9j))
for k in [3,4,5]:
    t=time.time(); dims=[]
    for w in range(k):
        V,res,nxt=lw_fixed(H,k,w,b=12); dims.append(V.shape[1])
    print("k=%d: f_k by sector %s, total %d (%.0f s), last residual %.1e"%(k,dims,sum(dims),time.time()-t,res),flush=True)
