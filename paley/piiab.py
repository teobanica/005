import numpy as np, sys
from piiinv import R_matrix
for q in map(int,sys.argv[1:]):
    R=R_matrix(q); N=2*(q+1)
    ev=np.sort(np.linalg.eigvals(R).real); ev=[e for e in ev if abs(e)>1e-6]
    vals=sorted(set(int(round(e)) for e in ev))
    rest=[v for v in vals if v not in (64*q,N**3)]
    print(q, q%8, "N^3=%d 64q=%d others:"%(N**3,64*q), rest, " /N^2:", [v/N**2 for v in rest], " /64:", [v/64 for v in rest], flush=True)
