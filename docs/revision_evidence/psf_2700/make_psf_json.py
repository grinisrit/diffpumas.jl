#!/usr/bin/env python3
"""Sidecar JSON and summary statistics of the resolution map, same fields as the
fig_resolution_map.json of the paper repo. Usage:
  make_psf_json.py PAPER_DIR TXT OUT_JSON [SUMMARY_JSON]
PAPER_DIR holds make_figures.py (its parse_resolution_map is reused); TXT is the
lvd_tomography_inverse.txt written by examples/lvd_tomography.jl --resolution-only."""
import sys, os, json, hashlib
import numpy as np
paper, txt, out = sys.argv[1:4]
summary_out = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, paper)
import make_figures as mf
mf.DATADIR = os.path.dirname(os.path.abspath(txt))
az, depths, mat, fmask, floors, counts = mf.parse_resolution_map(os.path.basename(txt))
finite = np.isfinite(mat)
data = {"azimuth_sectors": az, "depths_m": depths, "fwhm_m": mat.tolist(),
        "floor_limited_mask": fmask.tolist(), "geometric_floor_m": floors.tolist(),
        "probed_cell_counts": counts.tolist(),
        "floor_scope": "Vertical-grid diagnostic floor convention; anisotropic cells; not a certified isotropic resolution upper bound or subvoxel resolution",
        "count_scope": "Finite recovered-PSF estimates contributing to each median; hatch means no finite estimate, not no candidate cells"}
data["fwhm_m"] = [[None if not np.isfinite(v) else v for v in row] for row in data["fwhm_m"]]
sha = hashlib.sha256(open(txt, "rb").read()).hexdigest()
obj = {"figure": "fig_resolution_map.png", "sources": ["lvd_tomography_inverse.txt"], "data": data,
       "source_sha256": {"lvd_tomography_inverse.txt": sha},
       "adoption_scope": "Resolution map regenerated on the 2700-direction grid of the S7 protocol (zenith 0..58 step 2, azimuth 0..356 step 4); only the ray set differs from the earlier map"}
json.dump(obj, open(out, "w"), separators=(",", ":"))
s = {"sectors_total": int(mat.size), "sectors_finite": int(finite.sum()),
     "sectors_at_floor": int((fmask & finite).sum()), "sectors_no_psf": int((~finite).sum()),
     "finite_cell_estimates": int(counts.sum()),
     "fwhm_min": float(np.nanmin(mat)), "fwhm_max": float(np.nanmax(mat)),
     "fwhm_median_over_finite_sectors": float(np.nanmedian(mat)),
     "floor_m": sorted(set(floors.tolist())), "sectors_with_fwhm_above_floor": int(((mat > floors[:, None] + 0.5) & finite).sum())}
print(json.dumps(s, indent=1))
if summary_out:
    json.dump(s, open(summary_out, "w"), indent=1)
