import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
from pert import top_ritz
q0=np.pi/6*np.array([0,2,1]); d=np.array([1.0,0,0])
Hl=m.K6(*(q0+1e-12*d))
for k in [int(x) for x in sys.argv[1:]]:
    allA=[]; ftot=0
    for w in range(k):
        b=16 if k==3 else (32 if k==4 else 48)
        e0=top_ritz(Hl,k,w,b=b,thr=0.99); f=int((e0>1-1e-8).sum()); ftot+=f
        other=e0[e0<=1-1e-8]
        if f==0: continue
        res=[]
        for eps in [1e-3,2e-3]:
            e=top_ritz(m.K6(*(q0+eps*d)),k,w,b=b,thr=1-5e-4); res.append(np.sort((1-e)/eps**2))
        ok=len(res[0])==f and len(res[1])==f
        allA+=list(res[0])
        print("level %d sector %d: f=%d (others >0.99 at H_lim: %s), branches %d/%d, a_i = %s, stable to %.0e"%(k,w,f,np.round(other,4),len(res[0]),len(res[1]),np.round(res[0],3),np.abs(res[0]-res[1]).max() if ok else -1),flush=True)
    print("LEVEL %d: f(H_lim) = %d, branches %d, min a = %.4f"%(k,ftot,len(allA),min(allA) if allA else float('nan')),flush=True)
