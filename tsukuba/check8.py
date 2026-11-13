from table8 import Qmat
from nongen import dita_Q
import sys; sys.path.insert(0,'/home/claude/dita')
from magic import magic, ck_fast
for (M,N,ints,L) in [(4,2,(1,1,0),4),(2,4,(0,1,1),8),(4,2,(1,0,0),4)]:
    Q=Qmat(M,N,ints,L)
    print(M,N,ints,L,"model:",[ck_fast(magic(dita_Q(M,N,Q)),k) for k in (1,2,3,4)],flush=True)
