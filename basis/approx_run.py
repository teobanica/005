import sys, time, json, numpy as np, matrices as m, ck
name=sys.argv[1]; K=int(sys.argv[2])
H=m.from_hex(m.HAD16[int(name[-1])])
t0=time.time(); L=open("results/approx_%s.log"%name,"a")
def say(s):
    s="%6.0fs %s"%(time.time()-t0,s); print(s,flush=True); L.write(s+"\n"); L.flush()
ck.ck_approx(H,K,log=lambda k,r: say("%s c_%d = %d   (next %.4f, seeds %d)"%(name,k,r[0],r[1],r[2])),progress=say,b=40)
say("done")
