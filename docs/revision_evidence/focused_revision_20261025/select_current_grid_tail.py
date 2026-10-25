#!/usr/bin/env python3
"""Freeze deterministic geometry-spread selections using saved Jacobian, before FD evaluation."""
from pathlib import Path
import csv,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[3];EV=ROOT/'docs/revision_evidence';OUT=Path(__file__).resolve().parent
read=lambda p:list(csv.DictReader(p.open()))
jfile=EV/'final_field_mc_coverage_20261024/final_jacobian_sparse.csv';cfile=EV/'final_field_mc_coverage_20261024/final_field_coverage.csv';ffile=EV/'material_main_recovery_20261024/completed/main/data/lvd_fit_field.csv'
J=read(jfile);coverage={int(r['bin_idx']):r for r in read(cfile)};fields={int(r['cell_idx']):r for r in read(ffile)};support=[c for c,r in fields.items() if r['aquifer_mask']=='true'];entries=[];byray={b:set() for b in coverage}
for r in J:
 if r['in_support']!='true' or float(r['derivative'])==0:continue
 b,c=int(r['bin_idx']),int(r['cell_idx']);byray[b].add(c);entries.append(dict(bin=b,cell=c,stored_ad=float(r['derivative']),magnitude=abs(float(r['derivative']))))
def decorate(r,band):
 b,c=r['bin'],r['cell'];g=coverage[b];f=fields[c]
 return dict(band=band,bin=b,cell=c,stored_ad=r['stored_ad'],zenith_deg=float(g['zenith_deg']),slant_m=float(g['slant_m']),cell_z_m=float(f['z_m']),local_fraction=float(f['w']),fit_mask=g['fit_mask']=='true')
def spread(rows,n):
 rows=sorted(rows,key=lambda r:(r['bin'],r['cell']));features=np.array([[r['zenith_deg'],r['slant_m'],r['cell_z_m']] for r in rows]);span=np.ptp(features,axis=0);x=(features-features.min(axis=0))/np.maximum(span,1e-30);chosen=[0]
 while len(chosen)<n:
  d=np.min(np.sum((x[:,None,:]-x[chosen][None,:,:])**2,axis=2),axis=1);d[chosen]=-1;chosen.append(int(np.argmax(d)))
 return [rows[i] for i in chosen]
m=np.array([r['magnitude'] for r in entries]);a,b,c,d=np.quantile(m,[.1,.4,.6,.9]);bands={'weak':[r for r in entries if r['magnitude']<=a],'middle':[r for r in entries if b<=r['magnitude']<=c],'high':[r for r in entries if r['magnitude']>=d]};selection=[]
for label,pool in bands.items():selection+=spread([decorate(r,label) for r in pool],12)
coords={c:np.array([float(fields[c][k]) for k in ['x_m','y_m','z_m']]) for c in support};zeros=[]
for ray,crossed in sorted(byray.items()):
 candidates=[c for c in support if c not in crossed]
 def distance(c):return min((float(np.sum((coords[c]-coords[k])**2)) for k in crossed),default=float(np.sum(coords[c]**2)))
 cell=max(candidates,key=lambda c:(distance(c),-c));zeros.append(decorate(dict(bin=ray,cell=cell,stored_ad=0.),'structural_zero'))
selection+=spread(zeros,12)
assert len(selection)==48 and len({(r['bin'],r['cell']) for r in selection})==48
for i,r in enumerate(selection,1):r['case']=i
plan=dict(selection_policy='12 each from bottom10%,middle40-60%,top10% nonzero magnitude bands of saved covered supported J; deterministic maximin geometry-spread in zenith/slant/cell-height, lexicographic tie.12 zero candidates chosen spatially farthest from ray-supported nonzeros then same geometry-spread; must assert actual ray-segment disjointness before any model evaluation. Not probability sampling or domain-wide certification.',cases=selection,magnitude_band_edges=dict(p10=float(a),p40=float(b),p60=float(c),p90=float(d)),step_ladder=[1e-2,1e-3,1e-4,1e-5,1e-6,1e-7],bounds=[0,.9],stencils='central if feasible else second-order one-sided; no invalid negative fractions',resolution='per-ray max(10*abs(value-only repeat -assembled value),100eps(assembled value)); report absolute errors always; relative errors unresolved if all finite perturbations below floor.',no_mc=True,input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [jfile,cfile,ffile]},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'current_grid_tail_selection.json').write_text(json.dumps(plan,indent=2)+'\n')
with (OUT/'current_grid_tail_selection.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(selection[0]));w.writeheader();w.writerows(selection)
print('Frozen48 current-grid cases; FD outcomes not evaluated.')
