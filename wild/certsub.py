import sys, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
import jordan_full as JF
from lwsector import lw_fixed
from sdpcert import real_forms_span, certify
H=m.F6rs(np.exp(0.4j),np.exp(1.9j)); N=6
def forms_tagged(V,k):
    # recompute with tags: monkeypatch by calling forms for each (kind, r, merge) through max_level filter trick
    out=[]
    f=V.shape[1]
    full=JF.forms(V,N,k)
    return full
def subset_forms(V,k,kinds,maxlev):
    F=JF.forms(V,N,k,max_level=maxlev)
    return [(kd,B) for kd,B in F if kd in kinds]
D6=np.load("lw_dita23_k6.npz")
for k in [3,4,5,6]:
    for w in range(k):
        V=D6["w%d"%w] if k==6 else lw_fixed(H,k,w,b=16)[0]
        if V.shape[1]==0: continue
        res=[]
        for kinds,ml,name in [(("sesq",),2,"sesq,lev<=2"),(("sesq","bil"),2,"all,lev<=2"),(("sesq",),k-1,"sesq,all lev"),(("bil",),k-1,"bil,all lev")]:
            F=subset_forms(V,k,kinds,ml)
            if not F: res.append("%s: -"%name); continue
            lam,_=certify(real_forms_span(F,V.shape[1]),iters=2); res.append("%s: %+.3f"%(name,lam))
        print("level %d sector %d (f=%d): "%(k,w,V.shape[1])+" | ".join(res),flush=True)
