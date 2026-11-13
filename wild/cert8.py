import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
from lwsector import lw_fixed
from jordan_full import forms
from sdpcert import real_forms_span, certify
rng=np.random.default_rng(7); b=m.E(rng.random()); H=m.F82(1,b); N=8
for k in [3,4,5]:
    for w in range(k):
        t=time.time(); bb=16
        while True:
            V,res,_=lw_fixed(H,k,w,b=bb)
            if V.shape[1]<=bb-6: break
            bb*=2
        f=V.shape[1]
        if f==0: print("F_{8,2}(1,b) level %d sector %d: f=0"%(k,w),flush=True); continue
        F=forms(V,N,k,max_level=min(k-1,4)); lam,_=certify(real_forms_span(F,f),iters=3)
        print("F_{8,2}(1,b) level %d sector %d: f=%d, best lambda_min %.3e  [%.0f s]"%(k,w,f,lam,time.time()-t),flush=True)
