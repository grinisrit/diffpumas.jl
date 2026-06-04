# LVD muon event reconstruction (implementation spec)

English specification for reimplementing the Agafonova (2014) LVD muon-direction and
multiplicity reconstruction on **future raw event data**. Source: thesis PDF only
(not stored in-repo): <https://www.inr.ru/rus/referat/agafonova/dis.pdf>.

Machine-readable geometry: `lvd_detector_geometry.json`, digitized Fig. 1.6 layout in
`lvd_tower1_counters.csv` (280 counters, tower 1). Regenerate with
`digitize_lvd_fig16_layout.py` after re-reading the thesis PDF.

Tomography **validation** outputs (2D single-muon flux map after reconstruction) live in
`lvd_single_muon_flux_2d.csv` and are not required inputs to the event algorithm.

---

## What you must supply per event

Implementations should accept, at minimum:

| Field | Description |
|--------|-------------|
| Scintillator hits | Counter ID, measured energy `E_exp` (MeV), optional pulse time relative to trigger |
| Tracking hits | Module/plane ID, fired-strip clusters (or cluster centres), strip coordinates |
| Trigger context | First HET pulse (≥ 5 MeV) defining event time; optional tower/quarter ID |

Optional but useful: calibrated counter positions `(x, y, z)` in a fixed detector frame
(see **Layout coordinates** below).

**Event output** (per accepted muon group):

- Arrival direction `(theta_deg, phi_deg)` in the **native** convention (see geometry JSON).
- Multiplicity `n_mu`.
- Event class: single / wide pair / narrow pair / bundle (`n_mu >= 3`).
- Quality flags: `zeta`, and for bundles `Delta_delta` per removed fake track.

---

## Detector geometry (summary)

Full numeric constants: `lvd_detector_geometry.json`.

| Item | Value |
|------|--------|
| Scintillator counter outer size | 100 × 150 × 100 cm³ |
| Scintillator density ρ | 0.778 g/cm³ |
| Tower outer size | 13 × 6.6 × 12 m |
| Modules / counters per tower | 35 / 280 (8 counters per module) |
| Columns per tower | 5, separated by 70 cm corridors |
| Gap between towers | 2 m (no tracking in gaps) |
| Long axis vs south–north | 38.4° toward CERN |
| Module layout | L-tracking on bottom + one longitudinal side of each portatank |

**Digitized (Fig. 1.6):**

| File | Content |
|------|---------|
| `lvd_tower1_counters.csv` | 280 counters, tower-1 local frame |
| `lvd_all_towers_counters.csv` | 840 counters, T2/T3 shifted by 8.6 m along hall `y` |
| `lvd_detector_geometry.json` | Column/level grid, pitches, tower offsets |
| `digitize_lvd_fig16_layout.py` | Regenerate CSVs from Fig. 1.6 constants |

Frame: `x` along C1→C5, `y` along rows (0 at bottom row 7–8), `z` up from L=1.
Strip-level tracking planes inside modules are **not** digitized.

**Tomography-only coarse box** (DiffPumas `lvd_muography.jl`, not per-counter):
22.7 × 13.2 × 10.0 m, rotation 90° in the LVD plotting frame — used for acceptance
weighting, not for track fitting.

---

## Coordinate systems

**Native reconstruction frame** (use for algorithm I/O):

- `theta`: zenith from vertical, degrees.
- `phi`: from the front of the first tower toward the side face, clockwise (view from above).

**Paper / Fig. 7 frame** (tomography diagnostics only):

```text
phi_fig7 = 43 deg - phi_native  (mod 360)
```

Do not mix frames inside track reconstruction unless simulated rays use the same convention.

---

## Stage 0 — LVD event envelope (DAQ)

- An LVD event is scintillator + tracking data within **1 ms** after the trigger.
- Trigger: first pulse with energy **> 5 MeV** (HET); cluster of HET pulses can last ~300 ns.
- After trigger, LET threshold **0.5 MeV** in counters of the triggered tower quarter (or whole tower if all four quarters fire).
- Tracking data attached by program **BILDER**.

---

## Stage 1 — Muon event selection

**Initial muon file:**

- ≥ 2 triggered scintillator counters;
- each selected counter: energy **> ~50 MeV** (ADC channel 90);
- include full tracking data and trigger pulses **≥ 5 MeV**.

**Muon event:**

- ≥ 2 triggered counters;
- total deposited energy **≥ 55 MeV**;
- if multiple trigger pulses on one counter, keep the **maximum** amplitude only.

**Tracking pre-filter:**

- ≥ 3 triggered tracking-plane modules;
- fired strips per plane: **2 ≤ n_strips ≤ 30** (exclude plane if n_strips > 30);
- replace each strip cluster by its **centre point**.

---

## Stage 2 — Tracking-only reconstruction

1. Cluster centres → candidate straight tracks (strip-plane intersections).
2. Tracks in one bundle are **parallel**.
3. Minimum separation between distinct tracks in a bundle: **~40 cm** (accompanying particle cluster scale).
4. Fit group arrival direction `(theta, phi)`; quoted precision **~0.5°**.
5. Multiplicity `n_mu`.

**Classification by track separation `r`:**

| Class | Criterion |
|--------|-----------|
| Single | `n_mu = 1` |
| Wide pair | `n_mu = 2`, `r > 1.5 m` |
| Narrow pair | `n_mu = 2`, `r < 1.5 m` |
| Bundle | `n_mu ≥ 3` |

---

## Stage 3 — Scintillator consistency

Expected energy in counter `i` for a candidate track:

```text
E_cal_i = (dE/dx) * x_i * rho
```

- `dE/dx = 2.4 MeV/(g cm²)`
- `x_i` = path length through counter `i` (cm) — needs Fig. 1.6 geometry
- `rho = 0.778 g/cm³`

**Single muons and wide pairs:** verify tracking with **ζ** statistic:

```text
k    = sum_i E_exp_i / sum_i E_cal_i   (with E_exp_i = 0 where E_cal_i = 0)
zeta = sum_i |E_exp_i - k * E_cal_i| / sum_i E_exp_i
```

- Cap `E_exp_i` at **200 MeV** when minimising ζ (Landau tails / saturation).
- Reject / reprocess if **ζ > 70** (~1% mis-reconstruction below cut; threshold from visual scan of three one-month samples).

**Bundles (`n_mu ≥ 3`):**

- Each track: ≥ 2 tracking intersection points and ≥ 2 triggered scintillator counters.
- If any counter sees overlapping tracks → narrow-subgroup logic with **Δδ**:

```text
Delta_delta = delta_0 - delta_1
```

- `delta_0`: relative mismatch of total calculated vs measured energy (all triggered counters).
- `delta_1`: same after removing one candidate track.
- Real tracks: **Δδ ≥ 0.045**; fake tracks: **Δδ < 0.045** (remove iteratively).
- Accept bundle when **ζ < 70** and fake-track removal criteria pass; else retry with alternate criteria.

**Narrow pairs and high multiplicity:** rely more on scintillator pattern (L-shaped tracking can fake multiplicity).

---

## Angular flux map (downstream, not input)

After many reconstructed single muons, the thesis builds `I(θ, φ)`. For DiffPumas
tomography comparison use:

- `lvd_single_muon_flux_2d.csv` — documented 2D map (1359 bins)
- `lvd_single_muon_flux_2d_metadata.json` — column definitions
- Legacy: `examples/data/lvd_conf/rock_int.txt`

---

## Reference thresholds (quick lookup)

| Parameter | Value |
|-----------|--------|
| HET trigger | 5 MeV |
| Selection ADC (~) | 50 MeV (ch. 90) |
| Min total E (muon event) | 55 MeV |
| Min track separation | 40 cm |
| Wide / narrow pair split | 1.5 m |
| ζ cut | 70 |
| Δδ fake-track cut | 0.045 |
| E_exp cap in ζ | 200 MeV |
| Tracking strips per plane | 2–30 |
