import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
from jordan_full import forms, min_defect
k=int(sys.argv[1]); D=np.load("lw_dita23_k%d.npz"%k)
for w in range(k):
    V=D["w%d"%w]; t=time.time()
    F=forms(V,6,k,max_level=int(sys.argv[2]) if len(sys.argv)>2 else None)
    b,_=min_defect(F,V.shape[1],starts=20)
    print("2x3 Dita level %d, sector %d: dim %d, %d forms, min defect %.3e  [%.0f s]"%(k,w,V.shape[1],len(F),b,time.time()-t),flush=True)
