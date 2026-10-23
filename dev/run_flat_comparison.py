#!/usr/bin/env python3
"""Fresh nominal C/Julia reference; run inside run_bounded.py, preserving archives."""
import argparse,csv,hashlib,json,os,pathlib,re,subprocess,time,resource
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-energy-max',type=float,default=1e9);p.add_argument('--reference',type=pathlib.Path,required=True);p.add_argument('--julia-dump',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--samples',type=int,default=10000);p.add_argument('--seed',type=int,default=42);p.add_argument('--scope',choices=['smoke','full','control','deep'],default='full');p.add_argument('--mode',choices=['nominal','csda'],default='nominal');a=p.parse_args()
if a.samples<=0:p.error('samples must be positive')
if not 0<a.source_energy_max<=1e9:p.error('source-energy-max must be positive and no larger than1e9GeV')
if not 0<=a.seed<2**32:p.error('seed must be an unsigned32bit integer')
root=pathlib.Path(__file__).resolve().parents[1];ref=a.reference.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'c.csv').exists() or (out/'julia.csv').exists():raise SystemExit('Refusing to replace existing comparison CSVs')
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
archive=root/'examples/data/flux_comparison.xml';archivehash=sha(archive)
reference_meta=json.loads((ref/'c-reference-provenance.json').read_text())
adapted=ref/'geometry_configurable.c'
if sha(adapted)!=reference_meta['adapted_geometry_sha256'] or 'PUMAS_REFERENCE_SOURCE_MAX' not in adapted.read_text():
 raise SystemExit('Reference does not implement verified source-energy cutoff; prepare a NEW reference directory')
for filename,key in [('pumas-flat-reference','binary_sha256'),('libpumas.so.1','c_library_sha256'),('materials-c.pumas','c_dump_sha256')]:
 if sha(ref/filename)!=reference_meta[key]:raise SystemExit('Reference hash mismatch: '+filename)
metadata=dict(cpu_affinity=sorted(os.sched_getaffinity(0)),address_space_limit=resource.getrlimit(resource.RLIMIT_AS)[0],julia_version=subprocess.check_output(['julia','--version'],text=True).strip(),julia_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),julia_source_sha256={str(f.relative_to(root)):sha(f) for f in sorted((root/'src').glob('*.jl'))},manifest_sha256=sha(root/'Manifest.toml') if (root/'Manifest.toml').exists() else None,driver_sha256=sha(pathlib.Path(__file__).resolve()),samples=a.samples,seed=a.seed,scope=a.scope,mode=a.mode,reference_provenance=reference_meta,julia_dump_sha256=sha(a.julia_dump.resolve()),julia_driver_sha256=sha(root/'dev/flat_comparison_julia.jl'),archive_xml_sha256=archivehash,primary_altitude_m=1000,rock_density_kg_m3=2650,energy_min_GeV=1e-3,energy_max_GeV=1e9,threshold_GeV=100,source_energy_max_GeV=a.source_energy_max,nominal_only=True,seed_policy='Same integer seeds, independent C and Julia RNG implementations within each point; seed resets across grid points induce grid correlations; z-scores are not102independent tests',sigma_policy='Population second moment divided by sample count, including open sky; no hardcoded zero-error exception')
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
points=[(0.,0.),(100.,45.),(800.,75.)] if a.scope=='smoke' else [(100.,45.),(400.,60.),(800.,75.)] if a.scope=='control' else [(800.,75.)] if a.scope=='deep' else [(float(t),float(z)) for t in range(0,801,50) for z in [0,15,30,45,60,75]]
with (out/'c.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['thickness_m','zenith_deg','flux','sigma_mc','samples','seed','seconds']);w.writeheader()
 for t,z in points:
  env={**os.environ,'PUMAS_REFERENCE_DUMP':str(ref/'materials-c.pumas'),'PUMAS_REFERENCE_SAMPLES':str(a.samples),'PUMAS_REFERENCE_SEED':str(a.seed),'PUMAS_REFERENCE_CSDA':'1' if a.mode=='csda' else '0','PUMAS_REFERENCE_SOURCE_MAX':str(a.source_energy_max)}
  command=[str(ref/'pumas-flat-reference'),str(t),str(90-z),'0.001','1000000000'];start=time.monotonic();r=subprocess.run(command,env=env,capture_output=True,text=True,check=True)
  (out/f'c_{int(t)}_{int(z)}.log').write_text(r.stdout+r.stderr)
  m=re.search(r'Flux\s*:\s*([\d.Ee+\-]+)\s*\\pm\s*([\d.Ee+\-]+)',r.stdout)
  if not m:raise RuntimeError('Cannot parse C output '+r.stdout)
  w.writerow(dict(thickness_m=t,zenith_deg=z,flux=float(m[1]),sigma_mc=float(m[2]),samples=a.samples,seed=a.seed,seconds=time.monotonic()-start));f.flush()
with (out/'julia.log').open('w') as log:subprocess.run(['julia','--project=.',str(root/'dev/flat_comparison_julia.jl'),str(a.julia_dump.resolve()),str(out/'julia.csv'),str(a.samples),str(a.seed),a.scope,a.mode],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,"PUMAS_REFERENCE_SOURCE_MAX":str(a.source_energy_max)})
with (out/'c.csv').open() as f:cs=list(csv.DictReader(f))
with (out/'julia.csv').open() as f:js=list(csv.DictReader(f))
with (out/'comparison.csv').open('w') as f:
 fields=['thickness_m','zenith_deg','c_flux','c_sigma','julia_flux','julia_sigma','difference_percent','z_independent_mc'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for c,j in zip(cs,js,strict=True):
  assert (c['thickness_m'],c['zenith_deg'])==(j['thickness_m'],j['zenith_deg'])
  fc,fj,sc,sj=map(float,[c['flux'],j['flux'],c['sigma_mc'],j['sigma_mc']]);err=(sc*sc+sj*sj)**.5
  w.writerow(dict(thickness_m=c['thickness_m'],zenith_deg=c['zenith_deg'],c_flux=fc,c_sigma=sc,julia_flux=fj,julia_sigma=sj,difference_percent=100*(fj-fc)/fc if fc else float('nan'),z_independent_mc=(fj-fc)/err if err else float('nan')))
assert sha(archive)==archivehash,'Archived XML changed'
(out/'completed.json').write_text(json.dumps(dict(completed_utc=time.time(),points=len(cs)))+'\n')
