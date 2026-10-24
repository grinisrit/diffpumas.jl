#!/usr/bin/env python3
import csv,hashlib,json,math,sys,tomllib
from pathlib import Path
main=Path(sys.argv[1]).resolve();column=Path(sys.argv[2]).resolve();output=Path(sys.argv[3])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
def audit(root):
 end=json.loads((root/'end_status.json').read_text());assert end['returncode']==0
 assert all(end[k] is True for k in ('execution_completed','source_unchanged','runtime_unchanged'))
 for name,digest in end['output_sha256'].items():assert sha(root/name)==digest,name
 return end
m=audit(main);q=audit(column)
record=json.loads((main/'recovery_provenance.json').read_text());original=Path(record['original_failed_stage'])
o=json.loads((original/'end_status.json').read_text());assert o['returncode']==1 and o['execution_completed'] is False
for name,digest in o['output_sha256'].items():
 assert sha(original/name)==digest
 if name.startswith('data/'):assert m['output_sha256'][name]==digest,name
c=rows(main/'data/lvd_mc_common_subset.csv');coverage=rows(main/'data/lvd_mc_coverage.csv');selected={int(r['bin_idx']) for r in coverage if r['selected_mc']=='true'}
assert len(c)==630 and len({int(r['bin_idx']) for r in c})==630 and {int(r['bin_idx']) for r in c}==selected
assert sum(float(r['zenith_deg'])<60 for r in c)==550
for r in c:
 assert all(math.isfinite(float(v)) for v in r.values())
 assert float(r['wet_mc_sigma_unscaled'])>=0 and float(r['wet_mc_unscaled'])>0
summary=rows(main/'data/lvd_mc_common_subset_summary.csv');computed=[]
for s,key in zip(summary,['dry_csda_unscaled','wet_csda_unscaled','wet_mc_unscaled']):
 obs=[float(r['measured_raw']) for r in c];pred=[float(r[key]) for r in c]
 def residual(scale):return math.sqrt(sum((scale*p-o)**2 for p,o in zip(pred,obs))/sum(o*o for o in obs))
 common=residual(float(s['common_scale']));profiled=residual(float(s['profiled_subset_scale']))
 assert math.isclose(common,float(s['relative_residual_common_scale']),rel_tol=1e-14)
 assert math.isclose(profiled,float(s['relative_residual_profiled_scale']),rel_tol=1e-14)
 computed.append(dict(model=s['model'],common_scale_relative_residual=common,profiled_subset_relative_residual=profiled,scale=float(s['profiled_subset_scale'])))
curv=rows(main/'data/lvd_stable_curvature.csv');v=[float(r['residual_gauss_newton_curvature_diagonal']) for r in curv]
assert len(v)==600 and [int(r['cell_idx']) for r in curv]==list(range(1,601)) and all(math.isfinite(x) and x>=0 for x in v)
col=tomllib.loads((column/'column_domain_summary.toml').read_text());assert not col['changed_inputs']
assert col['source_inputs_sha256'][str(main/'data/lvd_fit_field.csv')]==m['output_sha256']['data/lvd_fit_field.csv']
assert col['covered']['count']==1003 and col['fitted']['count']==690
assert col['covered']['below_training_count']==1003 and col['fitted']['below_training_count']==690
result=dict(schema_version=1,passed=True,scope='Independent raw digest and arithmetic audit; not water-recovery accuracy or statistical coverage',review_required=True,revision_complete=False,
 helper_sha256=sha(Path(__file__)),main_end_sha256=sha(main/'end_status.json'),column_end_sha256=sha(column/'end_status.json'),
 original_failed_end_sha256=sha(original/'end_status.json'),all_original_data_unchanged=True,
 actual_mc_bins=630,mc_below60=550,mc_atleast60=80,mc_histories_per_bin=8192,seed=42,energy_seed=819,transport_seed='900042+k, covered-position k starts1',
 mc_standard_error_unscaled_min=min(float(r['wet_mc_sigma_unscaled']) for r in c),mc_standard_error_unscaled_max=max(float(r['wet_mc_sigma_unscaled']) for r in c),
 curvature_rows=600,curvature_min=min(v),curvature_max=max(v),curvature_scope='Local residual-GN diagonal, not calibrated posterior covariance',
 residuals=computed,column_domain={k:col[k] for k in ('training_q_min','training_q_max','covered','fitted')},
 input_sha256={str(p):sha(p) for p in (main/'data/lvd_mc_common_subset.csv',main/'data/lvd_mc_common_subset_summary.csv',main/'data/lvd_stable_curvature.csv',column/'column_domain.csv',column/'column_domain_summary.toml')},
 permanent_source_scope_fix_pending=True)
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
