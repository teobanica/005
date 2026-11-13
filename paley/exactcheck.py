# re-derive C, C', conv with pure integer arithmetic for given q and compare with FFT version
import sys, numpy as np

qs=[int(x) for x in sys.argv[1:]]
import trace_fast as tf
for q in qs:
    n=(q-1)//2
    sq=np.zeros(q,bool); sq[(np.arange(1,q)**2)%q]=True
    h=np.where(sq,1,-1).astype(np.int64); h[0]=1; a=np.arange(q)
    g=h[(-a)%q]*h[(1-a)%q]; gp=h[a]*h[(a-1)%q]
    C=np.array([int(g@np.roll(g,-t)) for t in range(q)]); Cp=np.array([int(gp@np.roll(gp,-t)) for t in range(q)])
    hh=tf.primroot(q)**2%q; sqs=np.array([pow(hh,k,q) for k in range(n)])
    c=C[sqs]; cp=Cp[sqs]
    conv=np.zeros(n,np.int64)
    for i in range(n): conv+=c[i]*np.roll(cp,i)
    N,_,out,T=tf.run(q)
    from trace_cert import ramanujan
    ok=True
    for d,P in out.items():
        r=np.arange(n); 
        Pex=4*sum(int(tf.mob(d//e))*e*int(conv[(np.gcd(r,d)%e)==0].sum()) for e in range(1,d+1) if d%e==0)
        ok&=(Pex==P)
    print(q,"exact integer recomputation agrees:",ok,flush=True)
