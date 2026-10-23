#!/usr/bin/env python3
"""Summarize a completed fresh nominal comparison without independent-grid claims."""
import argparse,csv,hashlib,json,math,pathlib,statistics
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=pathlib.Path);a=p.parse_args();root=a.directory
if not (root/'completed.json').exists():raise SystemExit('Comparison not complete')
metadata=json.loads((root/'metadata.json').read_text())
with (root/'comparison.csv').open() as f:rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
expected=102 if metadata['scope']=='full' else 3
assert len(rows)==expected
assert all(math.isfinite(v) for r in rows for v in r.values())
def stats(rs):
 differences=[r['difference_percent'] for r in rs];zs=[abs(r['z_independent_mc']) for r in rs]
 return dict(points=len(rs),median_signed_difference_percent=statistics.median(differences),median_abs_difference_percent=statistics.median(map(abs,differences)),max_abs_difference_percent=max(map(abs,differences)),median_abs_z=statistics.median(zs),max_abs_z=max(zs),points_abs_z_gt2=sum(z>2 for z in zs),points_abs_z_gt3=sum(z>3 for z in zs),positive_differences=sum(x>0 for x in differences),negative_differences=sum(x<0 for x in differences))
mc={}
for language in ['c','julia']:
 errors=[100*r[language+'_sigma']/r[language+'_flux'] for r in rows]
 mc[language]=dict(median_relative_mc_sigma_percent=statistics.median(errors),minimum_relative_mc_sigma_percent=min(errors),maximum_relative_mc_sigma_percent=max(errors))
summary=dict(schema_version=1,scope=metadata['scope'],samples_per_language_per_point=metadata['samples'],seed=metadata['seed'],c_version=metadata['reference_provenance']['version'],c_commit=metadata['reference_provenance']['commit'],julia_version=metadata['julia_version'],comparison_csv_sha256=hashlib.sha256((root/'comparison.csv').read_bytes()).hexdigest(),overall=stats(rows),mc_precision=mc,by_zenith={str(z):stats([r for r in rows if r['zenith_deg']==z]) for z in sorted({r['zenith_deg'] for r in rows})},by_thickness={str(t):stats([r for r in rows if r['thickness_m']==t]) for t in sorted({r['thickness_m'] for r in rows})},largest_abs_z=sorted(rows,key=lambda r:abs(r['z_independent_mc']),reverse=True)[:10],largest_abs_relative_difference=sorted(rows,key=lambda r:abs(r['difference_percent']),reverse=True)[:10],limitations=['Nominal transport comparison only; no systematic/model uncertainty added.','Same seed resets across grid points induce correlations; counts of z outliers are descriptive, not102independent hypothesis tests.','Per-point z combines independent-language MC errors in quadrature; it does not establish equivalence or a global goodness-of-fit.','10k sampled histories do not imply high precision; inspect the recorded per-bin Monte Carlo standard errors.','C geometry uses zero geomagnetic field and aligned surface-flux mass constant for common input; upstream physics library is unchanged.','Archived C version/seed/sample/input provenance remains unknown; this fresh reference is not a reproduction of the archived binary.'])
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(overall=summary['overall'],mc_precision=mc),indent=2))
