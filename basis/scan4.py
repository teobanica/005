import sys, json, time, numpy as np, matrices as m, ck
E=m.E
fams={'H6':lambda t: m.H6(E(t[0])),'BN6':lambda t: m.BN6(E(t[0])),'P7':lambda t: m.P7(E(t[0])),'R7':lambda t: m.R7(E(t[0])),
 'M81':lambda t: m.M81(E(t[0])),'F82':lambda t: m.F82(E(t[0]),E(t[1])),'K82':lambda t: m.K82(E(t[0]),E(t[1])),
 'F84':lambda t: m.F84(*[E(x) for x in t]),'K6':lambda t: m.K6(*[np.pi*x for x in t])}
for name in sys.argv[1:]:
    f=fams[name]; out=open('results/scan4_%s.jsonl'%name,'w'); t0=time.time(); cnt={}; n=0
    for l in open('results/scan_%s.jsonl'%name):
        r=json.loads(l)
        if r['c2']!=2: continue
        H=f(r['t']); b=ck.ck_approx(H,4,exact_max=600,b=20,deg=40); c4=b[4].shape[1]
        cnt[c4]=cnt.get(c4,0)+1; n+=1
        out.write(json.dumps({'t':r['t'],'c4':c4})+"\n"); out.flush()
    print(name, n, "wild pts, c4 counts:", cnt, "%.0fs"%(time.time()-t0), flush=True)
