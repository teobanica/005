import sys, json, time, numpy as np, matrices as m, ck, dita_test as dt, io, contextlib
order=int(sys.argv[1]); K=int(sys.argv[2]) if len(sys.argv)>2 else 3
lines=[l for l in open('data/had%d.txt'%order) if l.strip()]
out=open('results/sweep%d.log'%order,'w'); res=[]
t0=time.time()
for i,l in enumerate(lines):
    H=m.from_hex(l).real
    assert np.allclose(H@H.T, order*np.eye(order))
    row={'i':i}
    for side,M in [('H',H),('Ht',H.T.copy())]:
        cs=[]
        ck.ck_approx(M.astype(complex),K,log=lambda k,r: cs.append(int(r[0])),exact_max=1000,b=30)
        row[side]=cs
    f=io.StringIO()
    with contextlib.redirect_stdout(f):
        rep=dt.dita_report(H.astype(complex),"")
    row['dita']=str(rep) if rep is not None else f.getvalue().strip()
    res.append(row)
    s="%3d %s %s | %s  (%.0fs)"%(i,row['H'],row['Ht'],row['dita'][:80],time.time()-t0)
    print(s,flush=True); out.write(s+"\n"); out.flush()
    json.dump(res,open('results/sweep%d.json'%order,'w'))
