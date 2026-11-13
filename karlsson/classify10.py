import numpy as np, itertools
from pairs import *
q=5; z=np.exp(2j*np.pi/5); F=np.array([[z**(i*j) for j in range(q)] for i in range(q)])
units={(1,2,3,4),(2,4,1,3),(3,1,4,2),(4,3,2,1)}  # x->ux
out={}
for s in itertools.permutations(range(1,q)):
    for t in itertools.permutations(range(1,q)):
        L=F[np.ix_((0,)+s,(0,)+t)]
        c=set(ck(H,2)[0] for H in solve(F,L,0.8,4))
        out[(s,t)]=c
import pickle; pickle.dump(out,open('classify10.pkl','wb'))
wild=[k for k,v in out.items() if 2 in v]
print('pairings with a c2=2 solution:',len(wild),'of',len(out))
print('both units:', [ (k,out[k]) for k in out if k[0] in units and k[1] in units][:8])
print('s unit, t not, sample:', [(k,out[k]) for k in out if k[0] in units and k[1] not in units][:6])
