import numpy as np, csv
from common import *
cells=list(csv.DictReader(open(O+'gate/cells.tsv'),delimiter='\t'))
z=np.array([float(c['z']) for c in cells]); wt=np.array([float(c['w_test']) for c in cells])
cr=np.array([int(c['crossed']) for c in cells]); ev=np.array([int(c['evaluated']) for c in cells])
Ds=(cr==1)&(ev==1)
V={}
for l in open(WORK+'dump.tsv'):
    k,v=l.rstrip('\n').split('\t'); V[k]=np.array(v.split(','),dtype=float)
def rmse(w,mask=Ds): return float(np.sqrt(np.mean((w[mask]-wt[mask])**2)))
