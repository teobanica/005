from index8 import *
import sys
sys.path.insert(0,'/home/claude/dita'); from magic import magic, ck_fast
from nongen import dita_Q
def Qmat(M,N,ints,L):
    if M==2: return [[1]*N,[1]+[E(x/L) for x in ints]]
    return [[1,1]]+[[1,E(x/L)] for x in ints]
if __name__=="__main__":
  seen={}
  for (M,N) in [(2,4)]:
      for L in (1,2,4,8):
          for ints in itertools.product(range(L),repeat=3):
              if L>1 and np.gcd.reduce(list(ints)+[L])!=1: continue
              idx=index_formula(M,N,ints,L)
              if idx not in seen: seen[idx]=(ints,L)
  for idx,(ints,L) in sorted(seen.items()):
      out=[]
      for (M,N) in [(2,4),(4,2)]:
          I=ints if M==2 else tuple(reversed(ints))
          Q=Qmat(M,N,I,L)
          out.append(counts(M,N,Q,4))
      print("q=e(%s/%d): index %d, dim %d, c_k 2x4 %s, 4x2 %s"%(",".join(map(str,ints)),L,idx,8*idx,out[0],out[1]),flush=True)
