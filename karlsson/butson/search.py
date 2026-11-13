import numpy as np, itertools, sys
p=int(sys.argv[1]); z=np.exp(2j*np.pi/p); I=np.arange(p)
def blk(m,a,b): return z**((m*np.outer(I,I)+a*I[:,None]**2+b*I[None,:]**2)%p)
params=[(m,a,b) for m in range(1,p) for a in range(p) for b in range(p)]
B={q:blk(*q) for q in params}
F=blk(1,0,0); sols=[]
# row 0: [F, X]; row orthogonality within block-row 0 needs X X* = p: any blk works
for q01 in params:
    X=B[q01]
    for q10 in params:
        Y=B[q10]
        # need F Y^* + X Z^* = 0 => Z^* = -X^{-1} F Y^* => Z = -(F Y^*)^* (X^*)^{-1} ... solve directly and test entries
        Z=-(np.linalg.solve(X, F@Y.conj().T)).conj().T
        if np.abs(np.abs(Z)-1).max()<1e-9:
            H=np.block([[F,X],[Y,Z]])
            if np.abs(H@H.conj().T-2*p*np.eye(2*p)).max()<1e-8: sols.append((q01,q10,H))
print(p,'solutions',len(sols))
np.save('sols%d.npy'%p, np.array([s[2] for s in sols]) if sols else np.zeros(0))
for s in sols[:5]: print(s[0],s[1])
