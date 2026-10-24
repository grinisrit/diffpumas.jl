#!/usr/bin/env python3
import csv,hashlib,json,math,statistics,sys,tomllib
from pathlib import Path
folder=Path(sys.argv[1]).resolve();main=Path(sys.argv[2]).resolve();output=Path(sys.argv[3])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
end=json.loads((folder/'end_status.json').read_text());assert end['returncode']==0
assert all(end[k] is True for k in ('execution_completed','source_unchanged','runtime_unchanged'))
for name,digest in end['output_sha256'].items():assert sha(folder/name)==digest,name
provenance=json.loads((folder/'derived_provenance.json').read_text());summary=json.loads((folder/'derived_summary.json').read_text())
for key,path in [('quadrature_end_status_sha256',folder/'end_status.json'),('comparison_sha256',folder/'comparison.csv'),('summary_sha256',folder/'derived_summary.json'),('main_end_status_sha256',main/'end_status.json'),('main_field_sha256',main/'data/lvd_fit_field.csv')]:assert provenance[key]==sha(path),key
execution=json.loads((folder/'execution.json').read_text());sources=execution['source_sha256']
watchers=[digest for path,digest in sources.items() if Path(path).name=='watch_final_material_quadrature.py']
assert watchers==[provenance['watcher_sha256']]
config=tomllib.loads((folder/'config.toml').read_text());status=tomllib.loads((folder/'status.toml').read_text())
assert status['source_unchanged'] and not status['changed_inputs'] and status['rows']==140
assert config['reference_nodes']==512 and config['nodes']==[32,64,128,256,512] and config['random_sampling'] is False
coverage=rows(folder/'path_coverage.csv');assert len(coverage)==7
allowed=[float(r['zenith']) for r in coverage if r['derivative_status']=='allowed_support_cell']
flux_only=[float(r['zenith']) for r in coverage if r['derivative_status']=='no_free_parameter']
assert allowed==[0,30,45,60,75] and flux_only==[85,86]
raw=rows(folder/'measurements.csv');comparison=rows(folder/'comparison.csv');assert len(raw)==len(comparison)==140
key=lambda r:(r['corrected'],r['field'],r['zenith'],r['azimuth'])
ref={key(r):r for r in raw if r['nodes']=='512'}
for row,comp in zip(raw,comparison):
 for name in row:assert row[name]==comp[name]
 reference=ref[key(row)]
 for name in ('flux','gradient'):
  value=float(row[name]);baseline=float(reference[name]);defined=math.isfinite(baseline) and baseline!=0
  assert comp['relative_'+name+'_defined']==str(defined).lower()
  if defined:assert math.isclose(float(comp['relative_'+name+'_error']),value/baseline-1,rel_tol=1e-14,abs_tol=1e-15)
 if row['derivative_status']=='allowed_support_cell':assert 0<=float(row['fd_lower'])<=float(row['fd_upper'])<=.9
 else:assert int(row['cell'])==0 and math.isnan(float(row['gradient']))
for group in summary['groups']:
 selected=[r for r in comparison if r['corrected']==str(group['corrected']).lower() and r['field']==group['field'] and int(r['nodes'])==group['nodes']]
 for name in ('flux','gradient'):
  values=[abs(float(r['relative_'+name+'_error'])) for r in selected if r['relative_'+name+'_defined']=='true']
  assert group['valid_'+name+'_comparisons']==len(values)
  for stat,fn in [('maximum',max),('median',statistics.median)]:assert group[stat+'_absolute_relative_'+name+'_error']==fn(values)
half=max(abs(float(r['reference_gradient_half_step'])/float(r['gradient'])-1) for r in raw if r['nodes']=='512' and r['derivative_status']=='allowed_support_cell')
result=dict(schema_version=1,passed=True,scope='Deterministic sampled convergence against512nodes, not physical truth or domain-wide error bound',review_required=True,revision_complete=False,
 helper_sha256=sha(Path(__file__)),derived_provenance_sha256=sha(folder/'derived_provenance.json'),quadrature_end_sha256=sha(folder/'end_status.json'),
 raw_rows=140,valid_paths=7,selected_derivative_zeniths=allowed,flux_only_zeniths=flux_only,
 derivative_selection='Longest crossed cell within recorded allowed support; central or second-order one-sided finite differences, not AD validation',
 maximum_512_half_step_relative_derivative_change=half,groups=summary['groups'],conditional_primary_field=summary['selected_primary_field'])
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2))
