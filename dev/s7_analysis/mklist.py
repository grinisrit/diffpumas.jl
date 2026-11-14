from common import *
T=load('test/rows_shard*.tsv')
need=set()
for r in T:
    if r['rec'] and (r['selected']=='1' or (r['solver'] in('GN-0','MLEM->GN','GN-lin') and int(r['rep'])<=2)):
        need.add(r['rec'])
open(WORK+'need.txt','w').write('\n'.join(sorted(need))+'\n'); print(len(need))
