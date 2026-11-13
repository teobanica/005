import numpy as np, itertools
def close(gens):
    G=[np.eye(3)]; 
    while True:
        new=[g@h for g in G for h in gens]; add=[m for m in new if not any(np.allclose(m,g) for g in G)]
        if not add: return G
        for m in add:
            if not any(np.allclose(m,g) for g in G): G.append(m)
phi=(1+5**.5)/2
def rot(n,t):
    n=np.array(n,float); n/=np.linalg.norm(n); K=np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*K@K
P=np.array([[0,0,1],[1,0,0],[0,1,0.]])
Ostd=close([rot([0,0,1],np.pi/2),P]); I=close([P,rot([0,1,phi],2*np.pi/5)])
R=rot([0,0,1],np.pi/4); Onn=[R@g@R.T for g in Ostd]
V=[np.diag(d) for d in ([1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1])]
for nm,G in [("O",Ostd),("O nonnormal",Onn),("I",I)]:
    print(nm,len(G),"contains V:",all(any(np.allclose(v,g) for g in G) for v in V),
          sorted(set(np.round(np.abs(np.array(G)).ravel(),6))))
F2=sorted(set(np.round(np.array([0,.5,1/2**.5,phi/2,1/(2*phi),1])**2,9)))
print("pairs summing to 1:",[(a,b) for a in F2 for b in F2 if a>0 and b>0 and a<=b and abs(a+b-1)<1e-9])
