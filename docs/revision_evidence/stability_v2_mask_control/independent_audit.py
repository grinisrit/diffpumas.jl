#!/usr/bin/env python3
import csv,json,math,hashlib,sys,tomllib
from pathlib import Path
collector=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]);report=json.loads(collector.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
def audit(path):
 e=json.loads((path/'end_status.json').read_text());assert e['returncode']==0
 assert all(e[k] is True for k in ('execution_completed','source_unchanged','runtime_unchanged'))
 for k,v in e['output_sha256'].items():assert sha(path/k)==v,k
 return e
runs={k:Path(report['runs'][k]['path']) for k in ('dry/start0','dry/maskInf')};ends={k:audit(p) for k,p in runs.items()}
fields={k:rows(p/'results/lvd_fit_field.csv') for k,p in runs.items()};predictions={k:{int(r['bin_idx']):r for r in rows(p/'results/lvd_fit_predictions.csv')} for k,p in runs.items()}
base,mask=fields.values();active=[i for i,r in enumerate(base) if r['aquifer_mask']=='true']
identity=lambda r:[(x['cell_idx'],x['x_m'],x['y_m'],x['z_m'],x['aquifer_mask']) for x in r]
assert identity(base)==identity(mask) and len(active)==147 and len(base)==600
b=[float(base[i]['w']) for i in active];m=[float(mask[i]['w']) for i in active];d=[y-x for x,y in zip(b,m)]
norm=lambda x:math.sqrt(sum(t*t for t in x))
means={k:sum(float(r[i]['w']) for i in active)/len(active) for k,r in fields.items()}
computed=dict(field_relative_l2=norm(d)/norm(b),field_rmse=norm(d)/math.sqrt(len(d)),field_max_abs=max(map(abs,d)))
comparison=report['comparisons']['dry/maskInf']
for k,v in computed.items():assert math.isclose(v,comparison[k],rel_tol=1e-14)
bp,mp=predictions.values();common=sorted(set(bp)&set(mp));fitted=[k for k in common if float(bp[k]['fit_mask'])>0 and float(mp[k]['fit_mask'])>0]
assert len(common)==1003 and len(fitted)==690 and sum(float(r['fit_mask'])>0 for r in mp.values())==1003
for label,keys in [('common_covered',common),('common_fit',fitted)]:
 for name,pred in [('baseline',bp),('candidate',mp)]:
  obs=[float(pred[k]['measured_flux']) for k in keys];fit=[float(pred[k]['matched_csda_flux'])*float(pred[k]['normalization']) for k in keys]
  relative=norm([x-y for x,y in zip(fit,obs)])/norm(obs);log=math.sqrt(sum(math.log(x/y)**2 for x,y in zip(fit,obs))/len(keys))
  expected=comparison[label][name]
  assert math.isclose(relative,expected['relative_l2'],rel_tol=1e-14) and math.isclose(log,expected['log_rmse'],rel_tol=1e-14)
settings={k:tomllib.loads((p/'results/lvd_solver_configuration.toml').read_text()) for k,p in runs.items()}
assert settings['dry/start0']['support_sha256']==settings['dry/maskInf']['support_sha256']
for key in ('absolute_prior_weight','adjoint_backend','objective_model','robust_log_scale','normalization_mode','initial_field_water','correction_artifact_sha256'):
 assert settings['dry/start0'][key]==settings['dry/maskInf'][key],key
assert settings['dry/start0']['baseline_mask_ratio']==15 and math.isinf(settings['dry/maskInf']['baseline_mask_ratio'])
assert comparison['same_support'] is True and comparison['same_screen'] is False and comparison['objective_difference_comparable'] is False
for k,path in runs.items():
 execution=json.loads((path/'execution.json').read_text());other=json.loads((runs['dry/start0']/'execution.json').read_text())
 assert execution['julia_runtime']==other['julia_runtime']
 for key,digest in other['source_sha256'].items():
  if key.startswith('src/') or key in ('Project.toml','Manifest.toml','examples/lvd_tomography.jl'):assert execution['source_sha256'][key]==digest
status=dict(x.split('=',1) for x in (runs['dry/maskInf']/'results/lvd_solver_status.txt').read_text().splitlines() if '=' in x)
assert status['kkt_converged']=='false' and float(status['projected_gradient'])>float(status['projected_gradient_tolerance'])
result=dict(schema_version=1,passed=True,scope='Current fixed pseudo-Huber dry-correction screen control; not legacy IRLS and not physical water accuracy',review_required=True,
 collector_sha256=sha(collector),helper_sha256=sha(Path(__file__)),case_end_sha256={k:sha(p/'end_status.json') for k,p in runs.items()},
 support_means=means,mask_fit_bins=1003,reference_fit_bins=690,support_cells=147,comparison=comparison,
 projected_gradient=float(status['projected_gradient']),tolerance=float(status['projected_gradient_tolerance']),kkt_converged=False,
 nonzero_screen_disabled_field=True,conditional_normalization='Each fit retains its own profiled scale; common-bin residual comparisons do not hold normalization fixed',
 objective_comparison='Screen changes the configured objective; total objective values cannot be ranked as same-objective minima',revision_complete=False)
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='comparison'},indent=2))
