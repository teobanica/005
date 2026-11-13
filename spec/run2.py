import sys; sys.path.insert(0,'/home/claude/work/night'); sys.path.insert(0,'/home/claude/work/spec')
import numpy as np, matrices as m, spectro
rng=np.random.default_rng(5)
def phis(H,K):
    N=H.shape[0]; mt=spectro.mult_table(K); Phi={0:np.array([1.0]),1:np.zeros(N-1)}
    for k in range(2,K+1):
        S=spectro.spectrum(H,k); rem,_=spectro.subtract(S,[(Phi[s],mt[k][s]) for s in range(k)],1e-8); Phi[k]=rem
    return Phi
def incl(A,B,tol=1e-8):  # fraction of distinct values of A found in B
    ua=np.unique(np.round(A,9)); return np.mean([np.min(np.abs(B-x))<tol for x in ua]), len(ua)
if __name__=="__main__":
  for lab,H in [("T6",m.T6()),("H6q",m.H6(np.exp(2j*np.pi*rng.random()))),("BN6",m.BN6(np.exp(2j*np.pi*rng.random())))]:
      Phi=phis(H,5)
      print(lab, [ (s, incl(Phi[s],Phi[s+1])) for s in range(2,5)], "tops",[round(float(Phi[s].max()),6) for s in range(2,6)], "zeros",[int(np.sum(np.abs(Phi[s])<1e-9)) for s in range(2,6)],flush=True)
