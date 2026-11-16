from load_vec import *
T=load('test/rows_shard*.tsv')
mx=0;n=0
for r in T:
    if r['rec'] in V and r['selected']=='1':
        d=abs(rmse(V[r['rec']])-float(r['rmse_s'])); mx=max(mx,d); n+=1
print(Ds.sum(), n, mx)
