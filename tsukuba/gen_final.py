from gen_fit import *
rng=np.random.default_rng(11); tot=bad=pos=0
for (M,N) in [(2,2),(2,3),(3,2),(2,4),(4,2),(3,3),(3,4),(4,3),(4,4),(2,5),(5,2),(3,5)]:
    gens=[(i,c) for i in range(1,M) for c in range(1,N)]
    for L in (2,3,4,5,6,8,9,12):
        for trial in range(2):
            th=np.zeros((M,N)); th[1:,1:]=rng.integers(L,size=(M-1,N-1))/L
            for _ in range(25):
                R={g:int(rng.integers(-2,3)) for g in gens}
                if not any(R.values()): continue
                b=brute(M,N,th,R); f=formula_t(M,N,th,R,-1,1,1,1)
                tot+=1; pos+=b; bad+=(b!=f)
print("tested",tot,"positives",pos,"mismatches",bad)
