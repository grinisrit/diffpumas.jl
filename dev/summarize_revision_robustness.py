#!/usr/bin/env python3
"""Summarize completed nonlinear LVD refits; preserve incomplete runs explicitly."""
import csv
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
scenarios = ['baseline', 'cutoff350', 'cutoff650', 'maskInf', 'rho2660', 'rho2760',
             'initialwater03', 'scale_minus5pct', 'scale_plus5pct']

def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def norm(x):
    return math.sqrt(sum(v*v for v in x))

baseline_path = root / 'baseline' / 'lvd_fit_field.csv'
baseline = read_csv(baseline_path) if baseline_path.exists() else None
base_w = [float(r['w']) for r in baseline] if baseline else None
rows = []
for scenario in scenarios:
    d = root / scenario
    if not (d / 'completed.json').exists():
        rows.append(dict(scenario=scenario, status='failed' if (d/'failed.json').exists() else 'pending'))
        continue
    field = read_csv(d/'lvd_fit_field.csv')
    pred = read_csv(d/'lvd_fit_predictions.csv')
    status = dict(line.strip().split('=', 1) for line in (d/'lvd_recursion_status.txt').read_text().splitlines() if '=' in line)
    water = [float(r['w']) for r in field]
    aq = [r['aquifer_mask'].lower() == 'true' for r in field]
    active = [r for r in pred if float(r['fit_mask']) > 0]
    residual = [float(r['matched_csda_flux'])-float(r['measured_flux'])/float(r['normalization']) for r in active]
    obs = [float(r['measured_flux'])/float(r['normalization']) for r in active]
    logs = [math.log(max(float(r['matched_csda_flux']), 1e-300))-math.log(max(o, 1e-300)) for r,o in zip(active,obs)]
    logmean = sum(logs)/len(logs)
    if base_w is not None:
        if [r['cell_idx'] for r in field] != [r['cell_idx'] for r in baseline]:
            raise ValueError('Cannot compare unequal cell order')
        relchange = norm([v-b for v,b in zip(water,base_w)])/max(norm(base_w),1e-30)
    else:
        relchange = math.nan
    mass = sum(water)
    row = dict(scenario=scenario,status='complete',mean_band_water=sum(v for v,m in zip(water,aq) if m)/max(sum(aq),1),
        mean_all_water=sum(water)/len(water),aquifer_cells=sum(aq),positive_cells_gt_1e3=sum(v>1e-3 for v in water),
        positive_cells_gt_002=sum(v>0.02 for v in water),relative_residual=norm(residual)/max(norm(obs),1e-30),
        log_residual_std=math.sqrt(sum((v-logmean)**2 for v in logs)/len(logs)),
        field_relative_change=relchange,fit_bins=len(active),covered_bins=len(pred),
        converged=status.get('converged'),passes=status.get('passes'),termination=status.get('termination'),
        normalization=float(pred[0]['normalization']),rock_density_kg_m3=status.get('rock_density_kg_m3'))
    for coordinate in ('x_m','y_m','z_m'):
        row['water_center_'+coordinate] = sum(float(r[coordinate])*v for r,v in zip(field,water))/mass if mass>0 else math.nan
    history = read_csv(d/'lvd_recursion_history.csv')
    row.update(final_dfield=float(history[-1]['dfield']),final_dcoef=float(history[-1]['dcoef']))
    rows.append(row)
keys = list(dict.fromkeys(k for row in rows for k in row))
with (root/'comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
