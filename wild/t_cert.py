import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
from lwsector import lw_fixed
from jordan_full import forms
from sdpcert import real_forms_span, certify
H=m.F6rs(np.exp(0.4j),np.exp(1.9j))
for lab,HH,ks in [("Z_6",m.fourier(6),[4]),("2x3 Dita",H,[3,4,5])]:
    for k in ks:
        for w in range(k):
            V,_,_=lw_fixed(HH,k,w,b=40 if lab=="Z_6" else 16); f=V.shape[1]
            if f==0: continue
            t=time.time(); F=forms(V,6,k); basis=real_forms_span(F,f)
            lam,y=certify(basis)
            print("%s level %d sector %d: f=%d, span of forms %d / %d, best lambda_min %.3e  [%.0f s]"%(lab,k,w,f,len(basis),f*(2*f+1),lam,time.time()-t),flush=True)
