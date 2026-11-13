from general import *
import itertools
def formula_t(M,N,th,R,sj,sd,ri,rc):
    th2=np.zeros_like(th)
    for j in range(M):
        for d in range(N): th2[(sj*j)%M,(sd*d)%N]=th[j,d]
    R2={((ri*i)%M,(rc*c)%N):v for (i,c),v in R.items()}
    return formula(M,N,th2,R2)
if __name__=="__main__":
  rng=np.random.default_rng(3)
  for (M,N) in [(3,3),(3,4),(4,3),(3,5)]:
      gens=[(i,c) for i in range(1,M) for c in range(1,N)]; tests=[]
      for L in (2,4,5,7,8):
          for trial in range(3):
              th=np.zeros((M,N)); th[1:,1:]=rng.integers(L,size=(M-1,N-1))/L
              for _ in range(60):
                  R={g:int(rng.integers(-2,3)) for g in gens}
                  if any(R.values()): tests.append((th,R,brute(M,N,th,R)))
      res=[]
      for sj,sd,ri,rc in itertools.product((1,-1),repeat=4):
          bad=sum(formula_t(M,N,th,R,sj,sd,ri,rc)!=b for th,R,b in tests)
          res.append((bad,(sj,sd,ri,rc)))
      res.sort(); print((M,N),"positives",sum(b for *_,b in tests),"/",len(tests),"best:",res[:4])
