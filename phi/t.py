import time, numpy as np, phiprof, cases
for lab,k in [("T_6",2),("T_6",3),("T_6",4),("T_6",5),("H_6^q, q random",4),("BN_6^q, q random on arc",5),("P_12",3),("P_7^q, q random",4)]:
    t=time.time(); w,r,l=phiprof.top_phi(cases.C[lab](),k); print(lab,k,np.round(w,8),np.max(r) if len(r) else 0,l,round(time.time()-t,1),flush=True)
