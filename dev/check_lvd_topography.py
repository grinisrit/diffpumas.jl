#!/usr/bin/env python3
"""Compare traced rock distances with the supplied legacy column under explicit unit hypotheses.
This is an input-consistency check, not an independent topographic survey.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mask', required=True)
p.add_argument('--output', required=True)
p.add_argument('--density', type=float, default=2710.)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
legacy=pd.read_csv(root/'examples/data/lvd_conf/lvd_single_muon_flux_2d.csv')
m=pd.read_csv(a.mask).merge(legacy[['bin_index','slant_rock_m']], left_on='bin_idx',right_on='bin_index',validate='one_to_one')
m=m.rename(columns={'slant_rock_m':'legacy_depth_raw'})
m['legacy_geometric_m_if_mwe']=m.legacy_depth_raw/(a.density/1000.)
m['relative_difference_if_mwe']=m.slant_m/m.legacy_geometric_m_if_mwe-1
m['relative_difference_if_m']=m.slant_m/m.legacy_depth_raw-1
m['flagged_sector']=(m.azimuth_deg.between(135,180)&m.zenith_deg.between(40,55))
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
m.to_csv(out/'topography_consistency.csv',index=False)
rows=[]
for name,sel in [('all_covered',m.covered),('retained',m.covered&(m.fit_mask>0)),('excluded',m.covered&(m.fit_mask==0)),('flagged_sector',m.covered&m.flagged_sector),('outside_sector',m.covered&~m.flagged_sector)]:
 d=m[sel & np.isfinite(m.relative_difference_if_mwe)]
 rows.append(dict(group=name,n=len(d),median_mwe_assumption=d.relative_difference_if_mwe.median(),p95_abs_mwe_assumption=d.relative_difference_if_mwe.abs().quantile(.95),median_m_assumption=d.relative_difference_if_m.median()))
pd.DataFrame(rows).to_csv(out/'topography_consistency_summary.csv',index=False)
(out/'topography_consistency_scope.txt').write_text('The supplied metadata labels legacy depths as metres. Values near 3640 in the vertical bin are also consistent with the historical convention of metres water equivalent. Both interpretations are preserved; neither is verified by this script. Conversion of m.w.e. to geometric metres uses density '+str(a.density)+' kg/m^3. The traced geometry and legacy table may share their source, so agreement is not independent terrain validation. Input mask: '+str(Path(a.mask).resolve())+'\n')
print(pd.DataFrame(rows).to_string(index=False))
