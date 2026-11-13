import sys, json, time, numpy as np, matrices as m, ck, dita_test as dt, io, contextlib
order=int(sys.argv[1])
lines=[l for l in open('data/had%d.txt'%order) if l.strip()]
res=[]; t0=time.time()
for i,l in enumerate(lines):
    H=m.from_hex(l)
    c2=[]
    for M in (H, H.T.copy()):
        T=ck.T_matrix(M,2); w=np.linalg.eigvalsh((T+T.conj().T)/2); c2.append(int((w>1-1e-8).sum()))
    with contextlib.redirect_stdout(io.StringIO()): rep=dt.dita_report(H,"")
    dita = not (len(rep)==1 and 'no Dita' in rep[0])
    res.append({'i':i,'c2':c2,'dita':dita,'rep':[r[:60] for r in rep]})
json.dump(res,open('results/c2_%d.json'%order,'w'))
from collections import Counter
print(order, len(res), "time %.0fs"%(time.time()-t0))
print(" (c2(H),c2(Ht),Dita) counts:", Counter((r['c2'][0],r['c2'][1],r['dita']) for r in res))
