#!/usr/bin/env python3
"""Build LVD tower-1 counter layout from Agafonova thesis Fig. 1.6 (dis.pdf p.16).

Axis labels on the figure are raster; values below were read from the PDF at 3x zoom
and checked for internal consistency (column width 1.05 m, y/z pitch 1.548 / 1.504 m).

Run: python digitize_lvd_fig16_layout.py
Writes: lvd_tower1_counters.csv, updates lvd_detector_geometry.json layout section.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Fig. 1.6a — column x bounds (m), tower-local frame
COLUMNS = [
    ("C1", 0.0, 1.05),
    ("C2", 2.747, 3.797),
    ("C3", 5.494, 6.544),
    ("C4", 8.241, 9.291),
    ("C5", 10.988, 12.038),
]

# Lower y edge of each counter row (m); row 1 is top (counters 1,2)
# Fig. 1.6a tick marks: 0, 1.548, 3.096, 4.644 = lower edges of rows 4→1
Y_ROW_LO = {4: 0.0, 3: 1.548, 2: 3.096, 1: 4.644}
Y_PITCH_M = 1.548

# Fig. 1.6b — level z lower bounds (m)
Z_LEVEL_LO = {i: (i - 1) * 1.504 for i in range(1, 8)}
Z_PITCH_M = 1.504

# Thesis counter outer size (m): x × y × z in tower-local frame
COUNTER_SIZE_M = (1.0, 1.5, 1.0)

# Counter index within module (Fig. 1.6a): (row from top, x side)
# row 1 y_lo=4.644: 2 left, 1 right; row 2: 4,3; row 3: 6,5; row 4: 8,7
COUNTER_GRID = {
    1: (1, "right"),
    2: (1, "left"),
    3: (2, "right"),
    4: (2, "left"),
    5: (3, "right"),
    6: (3, "left"),
    7: (4, "left"),
    8: (4, "right"),
}


def counter_center_x(x_lo: float, x_hi: float, side: str) -> float:
    mid = 0.5 * (x_lo + x_hi)
    half = 0.5 * (x_hi - x_lo)
    return mid + 0.25 * half if side == "right" else mid - 0.25 * half


def counter_center_y(row: int) -> float:
    y_lo = Y_ROW_LO[row]
    return y_lo + 0.5 * Y_PITCH_M


def counter_center_z(level: int) -> float:
    return Z_LEVEL_LO[level] + 0.5 * Z_PITCH_M


def half_extent_along_row() -> float:
    # Fig. 1.6a row pitch; counter nominal 1.5 m along y fits inside 1.548 m cell
    return 0.5 * Y_PITCH_M


def half_extent_along_col(x_lo: float, x_hi: float) -> float:
    # Two counters share the 1.05 m column footprint in top view
    return 0.25 * (x_hi - x_lo)


def half_extent_along_z() -> float:
    return 0.5 * Z_PITCH_M


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for col_id, (col_name, x_lo, x_hi) in enumerate(COLUMNS, start=1):
        for level in range(1, 8):
            module_id = (col_id - 1) * 7 + level
            for counter_in_module in range(1, 9):
                row_idx, side = COUNTER_GRID[counter_in_module]
                cx = counter_center_x(x_lo, x_hi, side)
                cy = counter_center_y(row_idx)
                cz = counter_center_z(level)
                hx = half_extent_along_col(x_lo, x_hi)
                hy = half_extent_along_row()
                hz = half_extent_along_z()
                rows.append(
                    {
                        "tower_id": 1,
                        "column_id": col_id,
                        "column_name": col_name,
                        "level_id": level,
                        "module_id": module_id,
                        "counter_in_module": counter_in_module,
                        "counter_id": (module_id - 1) * 8 + counter_in_module,
                        "x_center_m": round(cx, 4),
                        "y_center_m": round(cy, 4),
                        "z_center_m": round(cz, 4),
                        "x_lo_m": round(cx - hx, 4),
                        "x_hi_m": round(cx + hx, 4),
                        "y_lo_m": round(cy - hy, 4),
                        "y_hi_m": round(cy + hy, 4),
                        "z_lo_m": round(cz - hz, 4),
                        "z_hi_m": round(cz + hz, 4),
                    }
                )
    return rows


def apply_tower_offset(rows: list[dict], tower_id: int, y_offset: float) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        rr = dict(r)
        rr["tower_id"] = tower_id
        rr["y_center_m"] = round(r["y_center_m"] + y_offset, 4)
        rr["y_lo_m"] = round(r["y_lo_m"] + y_offset, 4)
        rr["y_hi_m"] = round(r["y_hi_m"] + y_offset, 4)
        rr["counter_id"] = (tower_id - 1) * 280 + r["counter_id"]
        out.append(rr)
    return out


def main() -> None:
    tower1 = build_rows()
    assert len(tower1) == 280

    csv_path = ROOT / "lvd_tower1_counters.csv"
    fields = list(tower1[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(tower1)

    offsets = {"T1": 0.0, "T2": 8.6, "T3": 17.2}
    all_rows: list[dict] = []
    for tid, yoff in enumerate(offsets.values(), start=1):
        all_rows.extend(apply_tower_offset(tower1, tid, yoff))
    all_path = ROOT / "lvd_all_towers_counters.csv"
    with all_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    geom_path = ROOT / "lvd_detector_geometry.json"
    geom = json.loads(geom_path.read_text())
    geom["layout_coordinates"] = {
        "status": "digitized",
        "source_figure": "Agafonova 2014 thesis Fig. 1.6 (PDF p. 16)",
        "source_pdf_url": geom["reference"]["pdf_url"],
        "coordinate_frame": {
            "name": "tower1_local",
            "origin": "south-west bottom of tower 1 active volume (Fig. 1.6a origin)",
            "x_m": "along columns C1→C5 (long tower horizontal axis in top view)",
            "y_m": "along counter rows; y=0 is bottom row (counters 7,8)",
            "z_m": "vertical upward from floor; L=1 base z=0",
        },
        "fig_1_6a_column_x_bounds_m": {n: [a, b] for n, a, b in COLUMNS},
        "fig_1_6a_row_y_lower_edge_m": {f"row_{k}": v for k, v in Y_ROW_LO.items()},
        "fig_1_6b_level_z_lower_edge_m": {f"L{k}": v for k, v in Z_LEVEL_LO.items()},
        "pitch_m": {"column_width": 1.05, "inter_column_gap": 1.697, "row": Y_PITCH_M, "level": Z_PITCH_M},
        "counter_outer_m": {"x": 1.0, "y": 1.5, "z": 1.0},
        "counter_numbering_in_module": {
            "1": "row1 right (high x)",
            "2": "row1 left",
            "3": "row2 right",
            "4": "row2 left",
            "5": "row3 right",
            "6": "row3 left",
            "7": "row4 left",
            "8": "row4 right",
        },
        "modules_per_tower": 35,
        "counters_per_tower": 280,
        "counters_csv_tower1": "examples/data/lvd_conf/lvd_tower1_counters.csv",
        "counters_csv_all_towers": "examples/data/lvd_conf/lvd_all_towers_counters.csv",
        "notes": [
            "Row-1 lower edge 4.644 m; top of tower y ≈ 6.192 m (4.644 + 1.548).",
            "L=7 base 9.024 m; top z ≈ 10.528 m (9.024 + 1.504).",
            "Towers T2/T3: offset along hall y by (tower_width + 2 m gap); see three_tower_offsets_m.",
        ],
    }
    geom["three_tower_offsets_m"] = {
        "axis": "hall_y_perpendicular_to_tower_column_x",
        "tower_width_m": 6.6,
        "inter_tower_gap_m": 2.0,
        "tower_pitch_m": 8.6,
        "offsets": {"T1": 0.0, "T2": 8.6, "T3": 17.2},
    }
    geom_path.write_text(json.dumps(geom, indent=2) + "\n")
    print(f"Wrote {csv_path} ({len(tower1)} counters)")
    print(f"Wrote {all_path} ({len(all_rows)} counters)")
    print(f"Updated {geom_path}")


if __name__ == "__main__":
    main()
