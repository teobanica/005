# PSL(2,q) on P^1(F_q), q prime = 3 mod 4: count orbits on U x U, U = unordered pairs {i,j} (i=j allowed)
import sys, itertools
def run(q):
    INF=q
    def inv(a): return pow(a,q-2,q)
    def mob(M,x):
        a,b,c,d=M
        if x==INF: return INF if c==0 else a*inv(c)%q
        den=(c*x+d)%q
        if den==0: return INF
        return (a*x+b)*inv(den)%q
    gens=[(1,1,0,1),(4%q,0,0,1),(0,q-1,1,0)]   # translation, scaling by 4=2^2 (square; generates squares if 2 primitive-ish) 
    # use a generator of squares
    g=next(a for a in range(2,q) if all(pow(a,(q-1)//p,q)!=1 for p in range(2,q) if (q-1)%p==0 and all(p%r for r in range(2,p))))
    gens=[(1,1,0,1),(g*g%q,0,0,1),(0,q-1,1,0)]
    P=list(range(q+1))
    U=sorted({tuple(sorted(p)) for p in itertools.product(P,P)})
    idx={u:k for k,u in enumerate(U)}
    n=len(U); parent=list(range(n*n))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for M in gens:
        img=[idx[tuple(sorted((mob(M,u[0]),mob(M,u[1]))))] for u in U]
        for a in range(n):
            ia=img[a]
            for b in range(n):
                x,y=find(a*n+b),find(ia*n+img[b])
                if x!=y: parent[x]=y
    return len({find(k) for k in range(n*n)}), n
for q in map(int,sys.argv[1:]):
    o,n=run(q); print("q=%d |U|=%d  orbits on UxU (= sum of mult^2) = %d ; (q-3)/4=%d (q+1)/4=%d"%(q,n,o,(q-3)//4,(q+1)//4),flush=True)
