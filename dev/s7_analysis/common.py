import csv,glob,collections,tomllib,math,statistics as st
from scipy import stats
import os
# Run output directory (read-only) and work directory for intermediate files and outputs.
O=os.environ.get('S7_OUT','/mnt/big_data/agent-builds/diffpumas-s7-out').rstrip('/')+'/'
WORK=os.environ.get('S7_WORK','.').rstrip('/')+'/'
def load(pat):
    rows=[]
    for f in sorted(glob.glob(O+pat)):
        for r in csv.DictReader(open(f),delimiter='\t'):
            if r['stage']!='stage': rows.append(r)
    return rows
sel=tomllib.load(open(O+'selection.toml','rb'))
TAU=sel['meta']['exposures']
def tl(i):
    return {1000.0:'1e3',3160.0:'3.16e3',10000.0:'1e4',31600.0:'3.16e4',100000.0:'1e5',1e6:'1e6',1e8:'1e8'}[TAU[i-1]]
def ci(d):
    n=len(d); m=sum(d)/n
    if n<2: return n,m,float('nan'),float('nan'),float('nan')
    sd=st.stdev(d); h=stats.t.ppf(0.975,n-1)*sd/math.sqrt(n)
    return n,m,sd,m-h,m+h
