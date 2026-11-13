from even import *
W=blockdiag(box(RKL,RLK),box(RLK,RKL)); print("tau c4",ck_fast(W,4),flush=True)
W=blockdiag(RKL,RLK); print("sigma1 c4",ck_fast(W,4),flush=True)
