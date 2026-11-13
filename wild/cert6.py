import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
from jordan_full import forms
from sdpcert import real_forms_span, certify
k=int(sys.argv[1]); D=np.load("lw_dita23_k%d.npz"%k); ml=int(sys.argv[2]) if len(sys.argv)>2 else None
for w in range(k):
    V=D["w%d"%w]; f=V.shape[1]; t=time.time()
    F=forms(V,6,k,max_level=ml); basis=real_forms_span(F,f); lam,y=certify(basis,iters=3)
    print("2x3 Dita level %d sector %d: f=%d, span of forms %d / %d, best lambda_min %.3e  [%.0f s]"%(k,w,f,len(basis),f*(2*f+1),lam,time.time()-t),flush=True)
