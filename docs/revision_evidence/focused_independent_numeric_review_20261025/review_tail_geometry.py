#!/usr/bin/env python3
"""Independent source/provenance/grouping review of geometry-only tail metadata."""
import csv,hashlib,json,math,statistics,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];E=ROOT/'docs/revision_evidence';A=E/'focused_revision_20261025';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
run=A/'tail_geometry_run';end=json.loads((run/'end_status.json').read_text());exe=json.loads((run/'execution.json').read_text())
assert all(end[k] for k in ('execution_completed','source_unchanged','runtime_unchanged'))
for name,digest in end['output_sha256'].items():assert sha(run/name)==digest
for name,digest in exe['source_sha256'].items():assert sha(Path(name) if Path(name).is_absolute() else ROOT/name)==digest
source=A/'tail_geometry_depth.jl';code=source.read_text();production=(ROOT/'examples/lvd_tomography.jl').read_text()
assert 'midpoint_depth=surf-mz' in code and 'centroiddepth=local_surface_height(emap,x,y)-z' in code
assert '@assert [(r.cell,r.distance) for r in rows]==[(s.cell_idx,s.distance) for s in original.segments]' in code
assert 'x+=dir[1]*(step+POINT_EPS)' in code and 'x += dir[1] * (step + POINT_EPS)' in production
assert not any(x in code for x in ['compute_directional_flux','assemble_forward_and_jacobian','run_backward_mc'])
status=tomllib.load((run/'results/status.toml').open('rb'));assert status['source_tracer_segment_parity'] and status['cases']==48
rows=list(csv.DictReader((run/'results/selected_geometry_depth.csv').open()));selection=list(csv.DictReader((A/'current_grid_tail_selection.csv').open()));fieldfile=E/'material_main_recovery_20261024/completed/main/data/lvd_fit_field.csv';fields={int(r['cell_idx']):r for r in csv.DictReader(fieldfile.open())}
keys={int(r['case']):r for r in selection};assert len(rows)==48 and len({int(r['case']) for r in rows})==48
for r in rows:
    selected=keys[int(r['case'])];assert r['band']==selected['band'] and int(r['bin'])==int(selected['bin']) and int(r['cell'])==int(selected['cell'])
    assert float(r['cell_height_above_detector_m'])==float(fields[int(r['cell'])]['z_m'])
    intersects=r['selected_ray_has_segment']=='true';assert intersects==(r['band']!='structural_zero')
    if intersects:assert float(r['ray_segment_distance_m'])>0 and all(math.isfinite(float(r[k])) for k in ['ray_midpoint_height_above_detector_m','ray_midpoint_depth_below_local_surface_m'])
    else:assert all(r[k]=='' for k in ['ray_segment_distance_m','ray_midpoint_height_above_detector_m','ray_midpoint_depth_below_local_surface_m'])
claimed=json.loads((A/'tail_geometry_summary.json').read_text());checked={}
for band in ['weak','middle','high','structural_zero']:
    group=[r for r in rows if r['band']==band];assert len(group)==12
    depth=[float(r['ray_midpoint_depth_below_local_surface_m']) for r in group if r['selected_ray_has_segment']=='true'];height=[float(r['cell_height_above_detector_m']) for r in group]
    values=dict(cases=len(group),segment_midpoint_depth_range_m=[min(depth),max(depth)] if depth else None,segment_midpoint_depth_median_m=statistics.median(depth) if depth else None,centroid_height_range_m=[min(height),max(height)],centroids_above_surface=sum(float(r['cell_centroid_depth_below_local_surface_m'])<0 for r in group))
    assert values==claimed['groups'][band],band;checked[band]=values
nonzero=[r for r in rows if r['selected_ray_has_segment']=='true'];depth=[float(r['ray_midpoint_depth_below_local_surface_m']) for r in nonzero]
inputs=[source,ROOT/'examples/lvd_tomography.jl',run/'execution.json',run/'end_status.json',run/'results/status.toml',run/'results/selected_geometry_depth.csv',A/'tail_geometry_summary.json',A/'current_grid_tail_selection.csv',fieldfile]
report=dict(passed=True,scope='Independent source/formula/tracer-parity gate and raw grouping/provenance review; no geometry/transport rerun and no physical DEM validation.',groups=checked,nonzero_cases=len(nonzero),nonzero_centroids_above_surface=sum(float(r['cell_centroid_depth_below_local_surface_m'])<0 for r in nonzero),selected_segment_midpoint_depth_range_m=[min(depth),max(depth)],definition='Provided local vertical surface height minus traced segment midpoint height at midpoint horizontal coordinates; not slant depth, centroid height, or an independently measured terrain depth.',input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},review_source_sha256=sha(Path(__file__)))
(OUT/'tail_geometry_independent_review.json').write_text(json.dumps(report,indent=2)+'\n');print('Independent selected-tail geometry metadata: PASS')
