import numpy as np, sys, time
sys.path.insert(0,'/home/claude/jw/jw')
from ck import ck_deflated
H=np.load('H10.npy'); t0=time.time()
print('c5', ck_deflated(H,5)[:2], f'{time.time()-t0:.0f}s', flush=True)
