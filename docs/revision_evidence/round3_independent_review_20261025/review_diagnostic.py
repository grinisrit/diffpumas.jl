#!/usr/bin/env python3
"""Independent structural/numerical consistency audit; no stationarity certificate."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib

p = Path(sys.argv[1]).resolve()
r = tomllib.loads((p / 'diagnostic.toml').read_text())
directions = tomllib.loads((p / 'directions.toml').read_text())['directions']
rows = list(csv.DictReader((p / 'directional_checks.csv').open()))
assert len(rows) == len(directions) * 6 <= 258
assert r['direction_count'] == len(directions)
assert r['planned_max_evaluations'] == 258
assert math.isclose(r['objective'], 100.59269073659428, rel_tol=1e-12, abs_tol=1e-10)
assert math.isclose(r['projected_gradient_inf'], .003311897033113187, rel_tol=1e-7, abs_tol=1e-10)
assert r['absolute_prior_weight'] == 3.954930197858131
assert 'no generalized stationarity' in r['certificate']
for name, d in directions.items():
    assert len(d) == 600 and all(math.isfinite(x) for x in d)
    assert math.isclose(max(map(abs, d)), 1., abs_tol=1e-15)
    actual = [x for x in rows if x['direction'] == name]
    assert [float(x['step']) for x in actual] == [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
for x in rows:
    vals = {k: float(x[k]) for k in ('step', 'actual_step_inf', 'objective', 'objective_change', 'slope', 'data_objective', 'prior_objective', 'normalization')}
    assert all(map(math.isfinite, vals.values()))
    assert 0 < vals['actual_step_inf'] <= vals['step'] * (1 + 1e-8)
    assert vals['normalization'] > 0
    assert math.isclose(vals['objective'], vals['data_objective'] + vals['prior_objective'], abs_tol=1e-12)
    assert vals['objective_change'] == vals['objective'] - r['objective']
    assert vals['slope'] == vals['objective_change'] / vals['step']
    assert (x['resolved'] == 'true') == (abs(vals['objective_change']) > r['noise_floor'])
resolved_descents = [x for x in rows if x['resolved'] == 'true' and float(x['objective_change']) < 0]
report = dict(passed=True, rows=len(rows), directions=len(directions), resolved_descents=len(resolved_descents),
              best_trial=min(rows, key=lambda x: float(x['objective'])),
              scope='Independent consistency checks; no objective recomputation or stationarity certification.',
              sha256={name: hashlib.sha256((p / name).read_bytes()).hexdigest() for name in ('diagnostic.toml','directions.toml','directional_checks.csv')})
print(json.dumps(report, indent=2))
