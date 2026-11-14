from common import *
T=load('test/rows_shard*.tsv')
tab=collections.defaultdict(dict)
for r in T:
    if r['selected']=='1' and r['weights']=='obs':
        tab[(r['arm'],r['solver'],int(r['exp_index']))][int(r['rep'])]=float(r['rmse_s'])
def contrast(A,B):
    out=[]
    for arm in ['X','Mc','M']:
        for e in range(1,8):
            a=tab.get((arm,A,e)); b=tab.get((arm,B,e))
            if not a or not b: continue
            reps=sorted(set(a)&set(b)); d=[a[r]-b[r] for r in reps]
            n,m,sd,lo,hi=ci(d)
            lab='A lower (CI<0)' if hi<0 else ('B lower (CI>0)' if lo>0 else 'CI contains 0')
            out.append((arm,tl(e),n,m,lo,hi,sum(1 for x in d if x<0),lab))
    return out
print('## Item 1: paired contrasts per realisation (test, selected rows, obs weights; rep index matched; 95 % t-interval, df = n-1)\n')
print('Delta = RMSE_s(A) - RMSE_s(B); negative = A lower. n_A<B = number of realisations with Delta < 0. Pairing: same rule as `paired_contrast` in Tomography_protocol.jl:1241-1250 (intersection of rep keys; same observation vector). t critical value scipy.stats.t.ppf(0.975, n-1) (code uses a bisection on the same CDF; check: C1 rows below reproduce test_summary.tsv).\n')
res={}
for name,A,B in [('GN-0 minus GN-lin','GN-0','GN-lin'),('MLEM->GN minus GN-0','MLEM->GN','GN-0'),('GN-lin minus MLEM','GN-lin','MLEM'),('MLEM->GN minus MLEM (C1 check)','MLEM->GN','MLEM')]:
    print(f'### {name}\n'); print('| arm | tau | n | mean Delta | lo | hi | n(A<B) | reading |'); print('|---|---|---|---|---|---|---|---|')
    for arm,t,n,m,lo,hi,nn,lab in contrast(A,B):
        print(f'| {arm} | {t} | {n} | {m:+.4f} | {lo:+.4f} | {hi:+.4f} | {nn} | {lab} |')
    print()
# check against summary
summ=list(csv.DictReader(open(O+'test_summary.tsv'),delimiter='\t'))
mx=0
for r in summ:
    if r['kind']=='C1':
        e=int(r['exp_index']); c=[x for x in contrast('MLEM->GN','MLEM') if x[0]==r['arm'] and x[1]==tl(e)][0]
        mx=max(mx,abs(c[3]-float(r['mean'])),abs(c[4]-float(r['lo'])),abs(c[5]-float(r['hi'])))
print(f'Check: max abs difference of mean/lo/hi against the 14 C1 rows of test_summary.tsv = {mx:.2e}.\n')
