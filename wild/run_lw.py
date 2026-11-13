import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
from lwsector import lw_fixed
H=m.F6rs(np.exp(0.4j),np.exp(1.9j)); k=int(sys.argv[1]); b=int(sys.argv[2])
out={}
for w in range(k):
    t=time.time(); bb=b
    while True:
        V,res,nxt=lw_fixed(H,k,w,b=bb)
        if V.shape[1]<=bb-8: break
        bb*=2
    out["w%d"%w]=V
    print("k=%d sector %d: dim %d (block %d), residual %.1e, next Ritz %s, %.0f s"%(k,w,V.shape[1],bb,res,None if nxt is None else round(nxt,4),time.time()-t),flush=True)
np.savez_compressed("lw_dita23_k%d.npz"%k,**out)
print("total f_%d ="%k, sum(v.shape[1] for v in out.values()))
