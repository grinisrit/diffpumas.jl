#!/usr/bin/env julia
"""
lvd_muography.jl — Gran Sasso LVD directional muon flux, topography, and paper checks

Part 1 — Directional muon flux on the paper-matching nmap angular grid
  Backward-MC flux computed for each non-underground (θ, φ) bin using the
  slant rock distance R(θ,φ) from nm_c.inc, converted to vertical depth
  D = R·cos θ.  Rock density 2.71 g/cm³ (matching rr.for line 13).
  Gaisser primary flux sampled 10 000 m above the detector.
  Result saved as an interactive 3D surface of log₁₀(flux) for zenith angles
  up to 60° over the full 0°–360° azimuth range.

Part 2 — 3D topography & trajectory visualisation
  Elevation map derived from nm_c.inc measured rock thicknesses (48×91, 2°×4°),
  inverted via ray–surface bisection.  Six diverse trajectory directions are
  selected from the Part 1 flux grid (one per 60° azimuth sector, picking
  high-flux bins at moderate zenith).  Geometric rays and backward-MC muon
  trajectories are plotted over an interactive 3D terrain surface.

Part 3 — Reproduction checks against arXiv:0810.4635v1
  Rebuild the Gran Sasso MUSUN observables reported in the paper: the LVD
  acceptance-weighted azimuthal intensity for θ ≤ 60°, the underground muon
  energy spectrum and mean energy, and the angular occupancy map over the same
  θ ≤ 60° range in the LVD reference frame.  The paper curves are loaded from
  digitized benchmark assets extracted from the original EPS source files.

The dgsm table in nm_c.inc stores slant rock thickness in metres for each
(zenith, azimuth) direction from the LVD detector.  The `nmap` lookup adds
180° to the physical azimuth before indexing (rr.for line 19).  Values above
100 000 flag underground directions.

Materials: Standard Rock only for all three parts.
           Slant-depth conversion uses 2710 kg/m³ (matching rr.for line 13);
           transport uses the StandardRock tables.

For flat-geometry baseline flux studies and aquifer-detection scenarios, see
flat_muography.jl.

Usage:
    julia --project=. examples/lvd_muography.jl --output-dir PATH [OPTIONS]

Options:
    --output-dir, -o PATH     Output directory for plots (REQUIRED)
    --dump, -d PATH           Path to physics binary dump file
    --samples, -n INT         MC samples per flux bin (default: 1000)
    --paper-samples INT       MC samples for Part 3 energy spectrum (default: 20000)
    --threshold FLOAT         Energy threshold for mode switching (default: 100.0)
    --threshold-scan-low FLOAT
                              Lower multiplicative factor for transport systematic
                              threshold scan (default: 0.5)
    --threshold-scan-high FLOAT
                              Upper multiplicative factor for transport systematic
                              threshold scan (default: 2.0)
    --energy-min FLOAT        Minimum energy in GeV (default: 1e-3)
    --energy-max FLOAT        Maximum energy in GeV (default: 1e9)
    --no-straggling           Disable straggling
    --no-scattering           Disable scattering
    --part INT                Run only part 1, 2, or 3 (default: all)

Examples:
    julia --project=. examples/lvd_muography.jl -o out --part 1 --samples 500
    julia --project=. examples/lvd_muography.jl -o out --part 2
    julia --project=. examples/lvd_muography.jl -o out --part 3 --paper-samples 10000
"""

using DiffPumas
using DiffPumas.Physics: get_material_index, property_range, ENERGY_LOSS_CSDA
using DiffPumas.Loader: print_physics_summary
using DiffPumas.Geometry: PRIMARY_ALTITUDE, compute_flux, compute_flux_single_with_state,
                          locals_air_at_altitude
using DiffPumas.Geometry: transport_backward_step, transport_backward_step_full, transport_backward_step_mixed
using DiffPumas.Geometry: TwoLayerGeometry
using DiffPumas.GaisserFlux: flux_gccly
using DiffPumas.Types: State, Vec3
using DiffPumas.Pumas: load_or_create_physics
using DiffPumas.Turtle: ElevationMap, MapInfo, Stepper,
                        map_create, map_fill, map_elevation, map_meta, map_node,
                        stepper_create, stepper_add_flat, stepper_add_layer, stepper_add_map
using PlotlyJS
using Printf
using Random
using LinearAlgebra
using Statistics

const DEFAULT_DUMP = joinpath(@__DIR__, "data", "lvd_standardrock.pumas")
const DEFAULT_MDF  = joinpath(@__DIR__, "data", "materials.xml")

# ═══════════════════════════════════════════════════════════════════════════════
# Topography model from measured rock thickness (nm_c.inc)
# ═══════════════════════════════════════════════════════════════════════════════

const DETECTOR_ELEVATION = 963.0
const GRID_HALF_KM       = 8.0
const GRID_STEP_KM       = 0.1
const VALLEY_FLOOR       = 600.0
const ROCK_DENSITY_REF   = 2.71   # g/cm³ used in rr.for (line 13)

const NMC_PATH = joinpath(@__DIR__, "data", "nm_c.inc")
const PART1_ZENITH_MAX_DEG = 60.0
const PAPER_FULL_ZENITH_MAX_DEG = PART1_ZENITH_MAX_DEG
const NMAP_ZENITH_STEP_DEG = 2.0
const NMAP_AZIMUTH_STEP_DEG = 4.0

const PAPER_FIG7_CURVE_PATH  = joinpath(@__DIR__, "data", "lvd_paper_fig7_curve.csv")
const PAPER_FIG7_POINTS_PATH = joinpath(@__DIR__, "data", "lvd_paper_fig7_points.csv")
const PAPER_FIG8_CURVE_PATH  = joinpath(@__DIR__, "data", "lvd_paper_fig8_curve.csv")
const PAPER_FIG9_PEAKS_PATH  = joinpath(@__DIR__, "data", "lvd_paper_fig9_peaks.csv")
const PAPER_MEAN_ENERGY_GEV  = 273.0
const PAPER_MEAN_ENERGY_MEAS_GEV = 270.0
const PAPER_MEAN_ENERGY_MEAS_STAT_GEV = 3.0
const PAPER_MEAN_ENERGY_MEAS_SYST_GEV = 18.0

const LVD_BOX_LENGTH_M = 22.7
const LVD_BOX_WIDTH_M  = 13.2
const LVD_BOX_HEIGHT_M = 10.0
const LVD_MEAN_ACCEPTANCE_M2 = 298.0
const LVD_BOX_MEAN_PROJECTION_M2 = (
    2.0 * (LVD_BOX_LENGTH_M * LVD_BOX_WIDTH_M +
           LVD_BOX_LENGTH_M * LVD_BOX_HEIGHT_M +
           LVD_BOX_WIDTH_M  * LVD_BOX_HEIGHT_M)
) / 4.0
const LVD_ACCEPTANCE_SCALE = LVD_MEAN_ACCEPTANCE_M2 / LVD_BOX_MEAN_PROJECTION_M2

const PAPER_SPECTRUM_LOG10_EDGES = collect(range(0.0, stop=6.0, length=61))

struct NMapFluxResult
    zeniths::Vector{Float64}
    azimuths::Vector{Float64}
    flux::Matrix{Float64}
    sigma::Matrix{Float64}
    sigma_syst::Matrix{Float64}
    sigma_total::Matrix{Float64}
    rock::Matrix{Float64}
end

"""
    load_dgsm(filepath) -> Matrix{Float64}

Parse the Fortran DATA statements in `nm_c.inc` to extract the `dgsm(48,91)`
slant rock-thickness table (metres).

Grid layout (matching rr.for / nmap conventions):
  - Rows  1–48: zenith  θ = 0°, 2°, 4°, …, 94°
  - Cols  1–91: lookup azimuth φ′ = 0°, 4°, 8°, …, 360°
                (physical azimuth = φ′ − 180°)
  - Values > 100 000 flag underground lines of sight.
"""
function load_dgsm(filepath::String)
    dgsm = zeros(Float64, 48, 91)
    lines = readlines(filepath)

    current_row = 0
    values      = Float64[]

    for line in lines
        m = match(r"DATA\s*\(Dgsm\(\s*(\d+)", line)
        if m !== nothing
            if current_row in 1:48 && length(values) == 91
                dgsm[current_row, :] .= values
            end
            current_row = parse(Int, m.captures[1])
            values = Float64[]
            continue
        end

        if current_row > 0 && occursin(r"^\s+\*", line)
            data_part = replace(line, r"^\s+\*\s*" => "")
            data_part = replace(data_part, r"/\s*$" => "")
            for nm in eachmatch(r"-?\d+", data_part)
                push!(values, parse(Float64, nm.match))
            end
        end
    end

    if current_row in 1:48 && length(values) == 91
        dgsm[current_row, :] .= values
    end

    return dgsm
end

"""
    nmap_lookup(dgsm, theta_deg, phi_deg) -> (rock_m, is_underground)

Bilinear interpolation on the `dgsm` rock-thickness table, faithfully
replicating the Fortran `nmap` subroutine in `rr.for`.

Inputs use geographic azimuth (0° = North, 90° = East).  The +180° internal
shift and wrap match the Fortran storage layout.
"""
function nmap_lookup(dgsm::Matrix{Float64}, theta_deg::Float64, phi_deg::Float64)
    fi = phi_deg + 180.0
    fi >= 360.0 && (fi -= 360.0)
    tet = theta_deg

    jt = floor(Int, tet / 2.0) + 1
    jf = floor(Int, fi / 4.0) + 1

    (jt >= 48 || jf >= 91) && return (-999999.0, true)

    d1 = dgsm[jt,     jf]
    d2 = dgsm[jt + 1, jf]
    d3 = dgsm[jt,     jf + 1]
    d4 = dgsm[jt + 1, jf + 1]

    underground = d1 > 100_000 || d2 > 100_000 || d3 > 100_000 || d4 > 100_000

    d1 > 100_000 && (d1 -= 100_000)
    d2 > 100_000 && (d2 -= 100_000)
    d3 > 100_000 && (d3 -= 100_000)
    d4 > 100_000 && (d4 -= 100_000)

    t1 = 2.0 * (jt - 1)
    f1 = 4.0 * (jf - 1)
    r  = (tet - t1) / 2.0
    u  = (fi  - f1) / 4.0

    gsroc = (1 - r) * (1 - u) * d1 + r * (1 - u) * d2 +
            (1 - r) *      u  * d3 + r *      u  * d4

    underground && (gsroc += 100_000)

    return (gsroc, underground)
end

"""
    terrain_elevation(x_km, y_km, dgsm) -> Float64

Compute surface elevation (m ASL) at map coordinate `(x_km, y_km)` by inverting
the measured rock-thickness data.

For the direction (θ, φ) from the detector toward `(x, y)`, the horizontal
reach of a ray with slant thickness `R(θ,φ)` is `R·sin θ`.  We bisect on θ to
match the target horizontal distance `d_h`, then recover the elevation from
`z = DETECTOR_ELEVATION + R·cos θ`.
"""
function terrain_elevation(x_km::Float64, y_km::Float64, dgsm::Matrix{Float64})
    d_h_m = sqrt(x_km^2 + y_km^2) * 1000.0

    if d_h_m < 1.0
        gsroc, _ = nmap_lookup(dgsm, 0.0, 0.0)
        return DETECTOR_ELEVATION + gsroc
    end

    phi_deg = mod(atand(x_km, y_km), 360.0)

    θ_lo     = 0.0
    last_elev = DETECTOR_ELEVATION + 1377.0
    max_reach = 0.0

    dθ = 0.5
    for θ in dθ:dθ:88.0
        gsroc, ug = nmap_lookup(dgsm, θ, phi_deg)
        R = ug ? gsroc - 100_000.0 : gsroc
        R = max(R, 0.0)
        reach = R * sind(θ)

        if !ug
            last_elev = DETECTOR_ELEVATION + R * cosd(θ)
            max_reach = max(max_reach, reach)
        end

        if reach >= d_h_m
            θ_hi = θ
            for _ in 1:40
                θ_mid = 0.5 * (θ_lo + θ_hi)
                g, ug2 = nmap_lookup(dgsm, θ_mid, phi_deg)
                Rm = ug2 ? g - 100_000.0 : g
                Rm = max(Rm, 0.0)
                if Rm * sind(θ_mid) < d_h_m
                    θ_lo = θ_mid
                else
                    θ_hi = θ_mid
                end
            end
            θ_sol   = 0.5 * (θ_lo + θ_hi)
            g, _    = nmap_lookup(dgsm, θ_sol, phi_deg)
            R_sol   = g > 100_000 ? g - 100_000.0 : g
            return max(DETECTOR_ELEVATION + R_sol * cosd(θ_sol), VALLEY_FLOOR)
        end

        θ_lo = θ
    end

    if max_reach > 0.0
        overshoot = d_h_m / max_reach - 1.0
        fade = clamp(overshoot * 2.0, 0.0, 1.0)
        return last_elev * (1.0 - fade) + VALLEY_FLOOR * fade
    end
    return VALLEY_FLOOR
end

function build_elevation_map()
    dgsm = load_dgsm(NMC_PATH)
    println("  Loaded nm_c.inc rock-thickness map ($(size(dgsm,1))×$(size(dgsm,2)))")

    h  = GRID_HALF_KM
    s  = GRID_STEP_KM
    nx = round(Int, 2h / s) + 1
    ny = nx

    z_peak = DETECTOR_ELEVATION
    for row in 1:min(30, size(dgsm, 1))
        θ = 2.0 * (row - 1)
        for col in 1:size(dgsm, 2)
            v = dgsm[row, col]
            v > 100_000 && continue
            z_peak = max(z_peak, DETECTOR_ELEVATION + v * cosd(θ))
        end
    end
    z_max = z_peak + 200.0
    info  = MapInfo(nx, ny, (-h, h), (-h, h), (VALLEY_FLOOR, z_max))
    emap  = map_create(info)

    for iy in 0:(ny - 1)
        y_km = -h + iy * s
        for ix in 0:(nx - 1)
            x_km = -h + ix * s
            map_fill(emap, ix, iy, terrain_elevation(x_km, y_km, dgsm))
        end
    end

    return emap
end

wrap_azimuth_deg(phi_deg::Float64) = mod(phi_deg, 360.0)

function circular_distance_deg(a_deg::Float64, b_deg::Float64)
    return abs(mod(a_deg - b_deg + 180.0, 360.0) - 180.0)
end

function nmap_zenith_grid(zenith_min_deg::Float64, zenith_max_deg::Float64)
    min_deg = clamp(zenith_min_deg, 0.0, 94.0)
    max_deg = clamp(zenith_max_deg, min_deg, 94.0)
    start_deg = NMAP_ZENITH_STEP_DEG *
                ceil(Int, min_deg / NMAP_ZENITH_STEP_DEG - 1e-9)
    zeniths = start_deg > max_deg + 1e-9 ?
        Float64[] : collect(start_deg:NMAP_ZENITH_STEP_DEG:max_deg)
    if isempty(zeniths) || !isapprox(zeniths[end], max_deg; atol=1e-9)
        push!(zeniths, max_deg)
    end
    return zeniths
end

nmap_zenith_grid(zenith_max_deg::Float64) = nmap_zenith_grid(0.0, zenith_max_deg)

function centered_bin_edges(centers::Vector{Float64};
                            lower::Float64,
                            upper::Float64)
    isempty(centers) && return Float64[lower, upper]
    edges = zeros(Float64, length(centers) + 1)
    edges[1] = lower
    for i in 1:(length(centers) - 1)
        edges[i + 1] = 0.5 * (centers[i] + centers[i + 1])
    end
    edges[end] = upper
    return edges
end

function zenith_bin_solid_angles(zeniths::Vector{Float64};
                                 upper_deg::Float64)
    edges = centered_bin_edges(zeniths; lower=0.0, upper=upper_deg)
    dphi = deg2rad(NMAP_AZIMUTH_STEP_DEG)
    return [dphi * max(cosd(edges[i]) - cosd(edges[i + 1]), 0.0)
            for i in 1:length(zeniths)]
end

function drop_duplicate_azimuth_column(azimuths::Vector{Float64},
                                       grid::Matrix{Float64})
    if length(azimuths) >= 2 &&
       isapprox(wrap_azimuth_deg(azimuths[end]), wrap_azimuth_deg(azimuths[1]); atol=1e-9)
        return azimuths[1:(end - 1)], grid[:, 1:(end - 1)]
    end
    return azimuths, grid
end

function periodic_interp(x::Vector{Float64},
                         y::Vector{Float64},
                         xq::Vector{Float64};
                         period::Float64 = 360.0)
    length(x) == length(y) || error("periodic_interp: x/y length mismatch")
    isempty(x) && return Float64[]

    order = sortperm(x)
    xs = x[order]
    ys = y[order]

    xs_ext = vcat(xs, xs[1] + period)
    ys_ext = vcat(ys, ys[1])

    result = similar(xq)
    for (i, q_raw) in enumerate(xq)
        q = mod(q_raw - xs[1], period) + xs[1]
        idx = searchsortedlast(xs_ext, q)
        idx = clamp(idx, 1, length(xs))
        x1 = xs_ext[idx]
        x2 = xs_ext[idx + 1]
        y1 = ys_ext[idx]
        y2 = ys_ext[idx + 1]
        t = x2 == x1 ? 0.0 : (q - x1) / (x2 - x1)
        result[i] = y1 + t * (y2 - y1)
    end

    return result
end

function fit_positive_scale(reference::Vector{Float64},
                            model::Vector{Float64})
    denom = dot(model, model)
    denom <= 0.0 && return 1.0
    return max(dot(reference, model) / denom, 0.0)
end

function load_csv_columns(path::String, ncols::Int)
    cols = [Float64[] for _ in 1:ncols]
    open(path, "r") do io
        header_seen = false
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            startswith(line, "#") && continue
            if !header_seen
                header_seen = true
                continue
            end
            parts = split(line, ',')
            length(parts) == ncols || error("Expected $ncols columns in $path")
            for i in 1:ncols
                push!(cols[i], parse(Float64, strip(parts[i])))
            end
        end
    end
    return cols
end

function lvd_box_projected_area(zenith_deg::Float64,
                                azimuth_lvd_deg::Float64)
    ux = sind(zenith_deg) * cosd(azimuth_lvd_deg)
    uy = sind(zenith_deg) * sind(azimuth_lvd_deg)
    uz = cosd(zenith_deg)

    projected =
        LVD_BOX_WIDTH_M  * LVD_BOX_HEIGHT_M * abs(ux) +
        LVD_BOX_LENGTH_M * LVD_BOX_HEIGHT_M * abs(uy) +
        LVD_BOX_LENGTH_M * LVD_BOX_WIDTH_M  * abs(uz)

    return LVD_ACCEPTANCE_SCALE * projected
end

"""
    trace_ray(emap, azimuth_deg, zenith_deg; step_m) -> Vector{NTuple{3,Float64}}

Trace a straight ray from the underground detector outward until it exits
the terrain.  Returns the path as `(east_km, north_km, elevation_m)` tuples.
"""
function trace_ray(emap::ElevationMap, azimuth_deg::Float64, zenith_deg::Float64;
                   step_m::Float64 = 15.0)
    az  = deg2rad(azimuth_deg)
    zen = deg2rad(zenith_deg)

    dz = cos(zen)
    dh = sin(zen)
    dx = dh * sin(az)
    dy = dh * cos(az)

    path = NTuple{3,Float64}[(0.0, 0.0, DETECTOR_ELEVATION)]

    for i in 1:100_000
        s   = step_m * i
        x_m = dx * s
        y_m = dy * s
        z_m = DETECTOR_ELEVATION + dz * s

        x_km = x_m / 1000.0
        y_km = y_m / 1000.0

        elev, inside = map_elevation(emap, x_km, y_km)

        push!(path, (x_km, y_km, z_m))

        if !inside || z_m > elev
            break
        end
    end

    return path
end

"""
    create_topo_plot(emap, ray_paths; muon_paths, output_path)

Create interactive 3D PlotlyJS plot showing terrain, detector, geometric
rays, and optionally a few representative muon trajectories.
"""
function create_topo_plot(emap::ElevationMap,
                          ray_paths::Vector{Tuple{Float64,Float64,Vector{NTuple{3,Float64}}}};
                          muon_paths::Vector{Vector{NTuple{3,Float64}}} = Vector{NTuple{3,Float64}}[],
                          output_path::String)
    traces = GenericTrace[]

    # ── Terrain surface ─────────────────────────────────────────────────────
    x_arr = Float64[emap.x0 + i * emap.dx for i in 0:(emap.nx - 1)]
    y_arr = Float64[emap.y0 + j * emap.dy for j in 0:(emap.ny - 1)]

    stride = max(1, emap.nx ÷ 120)
    x_sub = x_arr[1:stride:end]
    y_sub = y_arr[1:stride:end]
    z_sub = emap.data[1:stride:end, 1:stride:end]

    push!(traces, surface(
        x = x_sub, y = y_sub, z = z_sub,
        colorscale = [
            [0.0,  "rgb(60,120,60)"],
            [0.25, "rgb(140,160,80)"],
            [0.50, "rgb(180,160,120)"],
            [0.70, "rgb(160,160,170)"],
            [0.85, "rgb(200,200,210)"],
            [1.0,  "rgb(240,245,255)"]
        ],
        opacity    = 0.80,
        showscale  = true,
        colorbar   = attr(title = "Elevation (m)", x = 1.05, len = 0.55),
        name       = "Terrain",
        hovertemplate = "E: %{x:.1f} km<br>N: %{y:.1f} km<br>" *
                        "Elev: %{z:.0f} m<extra></extra>"
    ))

    # ── Detector marker ─────────────────────────────────────────────────────
    push!(traces, scatter3d(
        x = [0.0], y = [0.0], z = [DETECTOR_ELEVATION],
        mode   = "markers+text",
        marker = attr(size = 9, color = "red", symbol = "diamond",
                      line = attr(width = 1, color = "darkred")),
        text         = ["LVD"],
        textposition = "top center",
        textfont     = attr(size = 13, color = "darkred", family = "Arial Black"),
        name         = "LVD Detector ($(Int(DETECTOR_ELEVATION)) m ASL)",
        showlegend   = true
    ))

    # ── Laboratory level plane ──────────────────────────────────────────────
    s = 1.5
    z0 = DETECTOR_ELEVATION
    push!(traces, mesh3d(
        x = [-s, s, s, -s],
        y = [-s, -s, s, s],
        z = [z0, z0, z0, z0],
        i = Int32[0, 0], j = Int32[1, 2], k = Int32[2, 3],
        color      = "rgba(255,60,60,0.12)",
        flatshading = true,
        name       = "Lab level",
        showlegend = true,
        hoverinfo  = "skip"
    ))

    # ── Geometric rays (coloured) ───────────────────────────────────────────
    palette = [
        "rgb(20,120,230)", "rgb(20,180,80)",  "rgb(130,50,200)",
        "rgb(220,170,20)", "rgb(20,200,180)", "rgb(180,130,30)",
        "rgb(30,130,130)", "rgb(150,90,180)", "rgb(200,100,50)",
        "rgb(50,180,120)", "rgb(230,80,20)",  "rgb(200,30,130)",
    ]

    for (idx, (az, zen, path)) in enumerate(ray_paths)
        xs = [p[1] for p in path]
        ys = [p[2] for p in path]
        zs = [p[3] for p in path]
        c  = palette[mod1(idx, length(palette))]

        rock_len = norm([xs[end] - xs[1], ys[end] - ys[1],
                         (zs[end] - zs[1]) / 1000]) * 1000
        az_s  = isinteger(az)  ? string(Int(az))  : string(az)
        zen_s = isinteger(zen) ? string(Int(zen)) : string(zen)
        label = "az=$(az_s)° θ=$(zen_s)° ($(round(Int, rock_len)) m)"

        push!(traces, scatter3d(
            x = xs, y = ys, z = zs,
            mode = "lines",
            line = attr(width = 4, color = c),
            name = label,
            showlegend = true
        ))

        if length(path) > 1
            push!(traces, scatter3d(
                x = [xs[end]], y = [ys[end]], z = [zs[end]],
                mode = "markers",
                marker = attr(size = 3, color = c),
                showlegend = false, hoverinfo = "skip"
            ))
        end
    end

    # ── Muon trajectories (thin red lines) ──────────────────────────────────
    for (idx, mpath) in enumerate(muon_paths)
        length(mpath) < 2 && continue
        xs = [p[1] for p in mpath]
        ys = [p[2] for p in mpath]
        zs = [p[3] for p in mpath]

        push!(traces, scatter3d(
            x = xs, y = ys, z = zs,
            mode = "lines",
            line = attr(width = 2, color = "rgba(210,30,30,0.7)"),
            name = idx == 1 ? "Muon trajectories" : "",
            showlegend = idx == 1,
            legendgroup = "muons",
            hovertemplate = "E: %{x:.2f} km<br>N: %{y:.2f} km<br>" *
                            "Z: %{z:.0f} m<extra>muon $idx</extra>"
        ))
    end

    # ── Layout ──────────────────────────────────────────────────────────────
    title_sub = isempty(muon_paths) ?
        "Geometric rays from underground detector" :
        "Geometric rays + $(length(muon_paths)) backward-MC muon trajectories (red)"

    layout = Layout(
        title = attr(
            text = "Gran Sasso LNGS — LVD Detector & Mountain Topography<br>" *
                   "<sub>Measured rock thickness (nm_c.inc / nmap) · $title_sub</sub>",
            font = attr(size = 16)
        ),
        scene = attr(
            xaxis = attr(title = "East (km)"),
            yaxis = attr(title = "North (km)"),
            zaxis = attr(title = "Elevation (m)", range = [400, 3200]),
            aspectmode  = "manual",
            aspectratio = attr(x = 1.0, y = 1.0, z = 0.25),
            camera      = attr(eye = attr(x = 1.5, y = -1.8, z = 0.7))
        ),
        width  = 1200,
        height = 900,
        legend = attr(x = 0.01, y = 0.99,
                      bgcolor = "rgba(255,255,255,0.85)",
                      font    = attr(size = 10))
    )

    fig = Plot(traces, layout)
    mkpath(dirname(output_path))
    savefig(fig, output_path)
    println("Saved plot: $output_path")
    return fig
end

"""
    simulate_muon_trajectories(physics, rock_idx, air_idx, emap, directions; ...)

Run backward-MC muon trajectories through a flat TwoLayerGeometry whose
thickness matches the local rock overburden, recording the 3D path in the
map coordinate system (east_km, north_km, elevation_m) for plotting.

`directions` is a vector of `(azimuth_deg, zenith_deg)` pairs.
Returns a vector of paths, each a `Vector{NTuple{3,Float64}}`.
"""
function simulate_muon_trajectories(physics, rock_idx::Int, air_idx::Int,
                                     emap::ElevationMap,
                                     directions::Vector{Tuple{Float64,Float64}};
                                     straggling::Bool = true,
                                     scattering::Bool = true,
                                     energy_threshold_low::Float64 = 100.0,
                                     porous_material::Int = -1,
                                     porous_density::Float64 = 0.0,
                                     porous_thickness::Float64 = 0.0,
                                     seed::Int = 99)

    rock_density = Float64(physics.tables[rock_idx].density)
    elev_above, _ = map_elevation(emap, 0.0, 0.0)
    rock_thickness = elev_above - DETECTOR_ELEVATION

    geometry = TwoLayerGeometry{Float64}(
        rock_thickness, rock_density, rock_idx, air_idx,
        porous_material, porous_density, porous_thickness)

    rng = MersenneTwister(seed)
    paths = Vector{NTuple{3,Float64}}[]

    n_use = length(directions)

    for mi in 1:n_use
        az_deg, zen_deg = directions[mi]
        az  = deg2rad(az_deg)
        zen = deg2rad(zen_deg)

        elevation = 90.0 - zen_deg
        theta = deg2rad(zen_deg)
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        dir_x = sin_theta * sin(az)
        dir_y = sin_theta * cos(az)
        dir_z = cos_theta

        kf = 1.0 + 99.0 * rand(rng)
        charge = rand(rng) > 0.5 ? 1.0 : -1.0

        state = State{Float64}(
            charge    = charge,
            energy    = kf,
            distance  = 0.0,
            grammage  = 0.0,
            time      = 0.0,
            weight    = 1.0,
            position  = Vec3{Float64}(0.0, 0.0, 0.0),
            direction = Vec3{Float64}(-dir_x, -dir_y, -dir_z),
            decayed   = false
        )

        energy_threshold = 1e12
        STEP_EPSILON = 1e-7
        STEP_MIN = 1e-6

        mpath = NTuple{3,Float64}[(0.0, 0.0, DETECTOR_ELEVATION)]
        record_every = 50
        step_count   = 0
        max_outer    = 500

        outer = 0
        while outer < max_outer && state.energy < energy_threshold - eps()
            outer += 1
            z = state.position[3]

            (z < 0.0 || z >= PRIMARY_ALTITUDE - eps()) && break
            (state.weight <= 0.0 || !isfinite(state.weight)) && break

            if state.energy < energy_threshold_low - eps()
                use_straggled = straggling; use_scattering = scattering
                current_limit = energy_threshold_low
            else
                use_straggled = false; use_scattering = false
                current_limit = energy_threshold
            end

            inner = 0
            while inner < 20000
                inner += 1
                z = state.position[3]
                (z < 0.0 || z >= PRIMARY_ALTITUDE - eps()) && break
                (state.energy >= current_limit - eps()) && break
                (state.weight <= 0.0 || !isfinite(state.weight)) && break

                uz_motion = -state.direction[3]

                has_porous = porous_material > 0 && porous_thickness > 0.0
                porous_z = has_porous ? rock_thickness - porous_thickness : rock_thickness

                if z < rock_thickness
                    if has_porous && z >= porous_z
                        mat = porous_material; dens = porous_density
                    else
                        mat = rock_idx; dens = rock_density
                    end
                    if uz_motion > eps()
                        nb = (has_porous && z < porous_z) ? porous_z : rock_thickness
                        stb = (nb - z) / uz_motion
                    elseif uz_motion < -eps()
                        pb = (has_porous && z >= porous_z) ? porous_z : 0.0
                        stb = (pb - z) / uz_motion
                    else
                        stb = 1000.0
                    end
                else
                    mat = air_idx
                    dens = locals_air_at_altitude(z).density
                    if uz_motion > eps()
                        stb = (PRIMARY_ALTITUDE - z) / uz_motion
                    elseif uz_motion < -eps()
                        stb = (rock_thickness - z) / uz_motion
                    else
                        stb = 1000.0
                    end
                end

                Xi = property_range(physics, ENERGY_LOSS_CSDA, mat, state.energy)
                max_gs = 0.01 * Xi / max(dens, 1e-6)

                if z >= rock_thickness
                    h_atm = 12e3
                    uz_abs = max(abs(uz_motion), 0.05)
                    max_gs = min(max_gs, 0.05 * h_atm / uz_abs)
                end

                step_size = clamp(min(stb + STEP_EPSILON, max_gs), STEP_MIN, 1000.0)

                if state.energy < energy_threshold_low - eps()
                    if use_straggled && use_scattering
                        state, _ = transport_backward_step_full(physics, state, mat, dens,
                                        step_size, rng; mode=:straggled, scattering=true,
                                        energy_limit=current_limit)
                    elseif use_straggled
                        state = transport_backward_step(physics, state, mat, dens,
                                        step_size; rng=rng, straggling=true)
                    else
                        state = transport_backward_step(physics, state, mat, dens,
                                        step_size; rng=nothing, straggling=false)
                    end
                else
                    if straggling
                        state = transport_backward_step_mixed(physics, state, mat, dens,
                                        step_size, rng; energy_limit=current_limit)
                    else
                        state = transport_backward_step(physics, state, mat, dens,
                                        step_size; rng=nothing, straggling=false)
                    end
                end

                step_count += 1
                if step_count % record_every == 0 && state.position[3] <= rock_thickness + 200.0
                    pos = state.position
                    push!(mpath, (pos[1] / 1000.0, pos[2] / 1000.0,
                                  DETECTOR_ELEVATION + pos[3]))
                end
            end
        end

        final_pos = state.position
        push!(mpath, (final_pos[1] / 1000.0, final_pos[2] / 1000.0,
                      DETECTOR_ELEVATION + final_pos[3]))

        if length(mpath) >= 2
            push!(paths, mpath)
        end
    end

    return paths
end

"""
    select_diverse_directions(flux_grid, zeniths, azimuths; n, seed) -> Vector{Tuple{Float64,Float64}}

Pick `n` well-spread (azimuth, zenith) directions from the flux grid.
Divides the azimuth range into `n` equal sectors and, within each sector,
selects a random valid (non-NaN, θ ∈ 10°–60°) bin.
"""
function select_diverse_directions(flux_grid::Matrix{Float64},
                                   zeniths::Vector{Float64},
                                   azimuths::Vector{Float64};
                                   n::Int = 6, seed::Int = 77)
    rng = MersenneTwister(seed)
    sector_width = 360.0 / n
    dirs = Tuple{Float64,Float64}[]

    for s in 0:(n - 1)
        az_lo = s * sector_width
        az_hi = az_lo + sector_width

        candidates = Tuple{Float64,Float64,Float64}[]
        for (a_idx, φ) in enumerate(azimuths)
            (φ < az_lo || φ >= az_hi) && continue
            for (z_idx, θ) in enumerate(zeniths)
                (θ < 10.0 || θ > PART1_ZENITH_MAX_DEG) && continue
                f = flux_grid[z_idx, a_idx]
                (!isnan(f) && f > 0.0) && push!(candidates, (φ, θ, f))
            end
        end

        if !isempty(candidates)
            sort!(candidates; by = c -> -c[3])
            top = candidates[1:min(5, length(candidates))]
            pick = top[rand(rng, 1:length(top))]
            push!(dirs, (pick[1], pick[2]))
        end
    end

    return dirs
end

function run_part2(physics, rock_idx::Int, air_idx::Int, emap::ElevationMap,
                   directions::Vector{Tuple{Float64,Float64}};
                   straggling::Bool, scattering::Bool,
                   energy_threshold_low::Float64,
                   porous_material::Int, porous_density::Float64,
                   porous_thickness::Float64,
                   output_dir::String)

    println("=" ^ 60)
    println(" Part 2: 3D Topography & Muon Trajectories")
    println("=" ^ 60)
    println()
    println("  Trajectory directions from Part 1 flux grid:")
    for (i, (az, zen)) in enumerate(directions)
        @printf("    %d. φ = %5.0f°  θ = %4.0f°\n", i, az, zen)
    end
    println()

    ray_paths = Tuple{Float64,Float64,Vector{NTuple{3,Float64}}}[]
    for (az, zen) in directions
        path = trace_ray(emap, az, zen)
        length(path) > 2 && push!(ray_paths, (az, zen, path))
    end

    println("  Traced $(length(ray_paths)) geometric rays")

    println("  Simulating backward-MC muon trajectories...")
    muon_paths = simulate_muon_trajectories(physics, rock_idx, air_idx, emap,
        directions;
        straggling=straggling, scattering=scattering,
        energy_threshold_low=energy_threshold_low,
        porous_material=porous_material,
        porous_density=porous_density,
        porous_thickness=porous_thickness)
    println("  Got $(length(muon_paths)) muon trajectories")
    println()

    create_topo_plot(emap, ray_paths;
        muon_paths=muon_paths,
        output_path=joinpath(output_dir, "part2_topography.html"))
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: Muon flux on the 48×91 measured rock-thickness grid
# ═══════════════════════════════════════════════════════════════════════════════

const NMAP_ROCK_DENSITY = 2710.0  # kg/m³ = 2.71 g/cm³ per rr.for line 13
const GAISSER_HEIGHT    = 10_000.0  # m above detector for primary flux sampling

"""
    compute_nmap_flux_grid(physics; kwargs...) -> NMapFluxResult

Compute backward-MC muon flux on the 48×91 angular grid matching the dgsm
rock-thickness table.  For each non-underground (θ, φ) bin the slant rock
distance R is converted to a vertical depth `D = R·cos θ` and fed to
`compute_flux` with ρ = 2710 kg/m³ (rr.for convention).

`zenith_max_deg` selects how much of the original 2° grid is used.  In this
example all three parts intentionally share the same θ ≤ 60° angular range.
"""
function compute_nmap_flux_grid(physics;
                                n_samples::Int = 1000,
                                straggling::Bool = true,
                                scattering::Bool = true,
                                energy_threshold_low::Float64 = 100.0,
                                threshold_factors::Tuple{Float64,Float64} = (0.5, 2.0),
                                energy_min::Float64 = 1e-3,
                                energy_max::Float64 = 1e9,
                                zenith_min_deg::Float64 = 0.0,
                                zenith_max_deg::Float64 = PART1_ZENITH_MAX_DEG,
                                porous_material::Int = -1,
                                porous_density::Float64 = 0.0,
                                porous_thickness::Float64 = 0.0,
                                progress_note::String = "")

    dgsm = load_dgsm(NMC_PATH)

    zeniths  = nmap_zenith_grid(zenith_min_deg, zenith_max_deg)
    n_azi = 91
    azimuths = [4.0 * (j - 1) for j in 1:n_azi]
    n_zen = length(zeniths)
    progress_prefix = isempty(progress_note) ? "" : "$(progress_note) "

    flux_grid  = fill(NaN, n_zen, n_azi)
    sigma_grid = fill(NaN, n_zen, n_azi)
    sigma_syst_grid = fill(NaN, n_zen, n_azi)
    sigma_total_grid = fill(NaN, n_zen, n_azi)
    rock_grid  = fill(NaN, n_zen, n_azi)

    n_total   = n_zen * n_azi
    n_done    = 0
    n_skipped = 0

    for (z_idx, θ) in enumerate(zeniths)
        if z_idx == 1 || z_idx == n_zen || θ >= 86.0
            @info @sprintf("%sStarting zenith row %d/%d (θ=%.0f°)",
                           progress_prefix, z_idx, n_zen, θ)
        end
        elevation = 90.0 - θ
        cos_θ = cosd(θ)
        for (a_idx, φ) in enumerate(azimuths)
            n_done += 1

            gsroc, underground = nmap_lookup(dgsm, θ, φ)
            if underground || gsroc <= 0.0
                n_skipped += 1
                continue
            end

            rock_grid[z_idx, a_idx] = gsroc
            depth = gsroc * cos_θ
            p_thick = min(porous_thickness, depth)

            if n_done % 200 == 0 || n_done == 1 || n_done == n_total
                @info @sprintf("%s[%d/%d] θ=%.0f° φ=%.0f° R=%.0fm D=%.0fm",
                               progress_prefix, n_done, n_total, θ, φ, gsroc, depth)
            end

            budget = compute_flux_uncertainty(physics,
                NMAP_ROCK_DENSITY, depth, elevation, energy_min, energy_max;
                n_samples    = n_samples,
                straggling   = straggling,
                scattering   = scattering,
                energy_threshold_low = energy_threshold_low,
                threshold_factors = threshold_factors,
                porous_material  = porous_material,
                porous_density   = porous_density,
                porous_thickness = p_thick,
                primary_altitude = GAISSER_HEIGHT)

            flux_grid[z_idx, a_idx] = budget.value
            sigma_grid[z_idx, a_idx] = budget.sigma_mc
            sigma_syst_grid[z_idx, a_idx] = budget.sigma_syst
            sigma_total_grid[z_idx, a_idx] = budget.sigma_total
        end
    end

    @info @sprintf("%sComputed %d bins (%d underground/skipped)",
                   progress_prefix, n_done, n_skipped)
    return NMapFluxResult(zeniths, azimuths, flux_grid, sigma_grid, sigma_syst_grid, sigma_total_grid, rock_grid)
end

function merge_nmap_flux_results(base::NMapFluxResult,
                                 extension::NMapFluxResult)
    isempty(extension.zeniths) && return base
    size(base.flux, 2) == size(extension.flux, 2) ||
        error("Cannot merge nmap flux grids with different azimuth dimensions")
    all(isapprox.(base.azimuths, extension.azimuths; atol=1e-9)) ||
        error("Cannot merge nmap flux grids with different azimuth axes")
    maximum(base.zeniths) < minimum(extension.zeniths) - 1e-9 ||
        error("Cannot merge overlapping zenith ranges")

    return NMapFluxResult(
        vcat(base.zeniths, extension.zeniths),
        copy(base.azimuths),
        vcat(base.flux, extension.flux),
        vcat(base.sigma, extension.sigma),
        vcat(base.sigma_syst, extension.sigma_syst),
        vcat(base.sigma_total, extension.sigma_total),
        vcat(base.rock, extension.rock),
    )
end

"""
    create_nmap_flux_plot(zeniths, azimuths, flux_grid, rock_grid; output_path)

3D PlotlyJS surface of log₁₀(flux) over the (azimuth, zenith) angular grid,
with a companion surface showing the raw slant rock thickness.
"""
function create_nmap_flux_plot(zeniths::Vector{Float64},
                               azimuths::Vector{Float64},
                               flux_grid::Matrix{Float64},
                               sigma_grid::Matrix{Float64},
                               sigma_syst_grid::Matrix{Float64},
                               sigma_total_grid::Matrix{Float64},
                               rock_grid::Matrix{Float64};
                               output_path::String)

    log_flux = map(f -> (isnan(f) || f <= 0.0) ? NaN : log10(f), flux_grid)
    log_flux_upper = map((f, σ) -> (isnan(f) || f <= 0.0 || !isfinite(σ)) ? NaN : log10(f + σ),
                         flux_grid, sigma_total_grid)
    log_flux_lower = map((f, σ) -> (isnan(f) || f <= 0.0 || !isfinite(σ) || f <= σ) ? NaN : log10(f - σ),
                         flux_grid, sigma_total_grid)
    hover_text = [@sprintf(
        "φ=%.0f°<br>θ=%.0f°<br>Flux=%.5e<br>MC=%.2f%%<br>Transport syst=%.2f%%<br>Total=%.2f%%",
        azimuths[a_idx],
        zeniths[z_idx],
        flux_grid[z_idx, a_idx],
        100 * sigma_grid[z_idx, a_idx] / max(abs(flux_grid[z_idx, a_idx]), 1e-30),
        100 * sigma_syst_grid[z_idx, a_idx] / max(abs(flux_grid[z_idx, a_idx]), 1e-30),
        100 * sigma_total_grid[z_idx, a_idx] / max(abs(flux_grid[z_idx, a_idx]), 1e-30),
    ) for z_idx in 1:length(zeniths), a_idx in 1:length(azimuths)]

    valid = filter(!isnan, log_flux)
    zmin = isempty(valid) ? -12.0 : floor(minimum(valid))
    zmax = isempty(valid) ?   0.0 : ceil(maximum(valid))

    traces = GenericTrace[]

    push!(traces, surface(
        x = azimuths, y = zeniths, z = log_flux,
        colorscale = [
            [0.0,  "rgb(4,14,90)"],
            [0.15, "rgb(20,55,160)"],
            [0.30, "rgb(30,120,180)"],
            [0.45, "rgb(50,180,130)"],
            [0.60, "rgb(120,210,60)"],
            [0.75, "rgb(220,220,30)"],
            [0.90, "rgb(240,130,20)"],
            [1.0,  "rgb(180,10,10)"]
        ],
        cmin = zmin, cmax = zmax,
        showscale  = true,
        colorbar   = attr(title = "log₁₀ Φ<br>(m⁻²s⁻¹sr⁻¹)",
                          x = 1.08, len = 0.6),
        name       = "Muon flux",
        text = hover_text,
        hovertemplate = "%{text}<extra></extra>"
    ))

    push!(traces, surface(
        x = azimuths, y = zeniths, z = log_flux_upper,
        colorscale = "Oranges",
        opacity = 0.18,
        showscale = false,
        name = "Flux total +1σ"
    ))
    push!(traces, surface(
        x = azimuths, y = zeniths, z = log_flux_lower,
        colorscale = "Oranges",
        opacity = 0.18,
        showscale = false,
        name = "Flux total -1σ"
    ))

    rock_log = map(r -> (isnan(r) || r <= 0.0) ? NaN : log10(r), rock_grid)
    push!(traces, surface(
        x = azimuths, y = zeniths, z = rock_log,
        colorscale = "Greys",
        opacity    = 0.35,
        showscale  = false,
        name       = "Rock thickness (log₁₀ m)",
        visible    = "legendonly",
        hovertemplate = "φ=%{x:.0f}°<br>θ=%{y:.0f}°<br>" *
                        "log₁₀R=%{z:.2f} m<extra>rock</extra>"
    ))

    layout = Layout(
        title = attr(
            text = "Gran Sasso LVD — Muon Flux on $(length(zeniths))×$(length(azimuths)) nmap Grid<br>" *
                   "<sub>ρ=$(NMAP_ROCK_DENSITY) kg/m³ · Gaisser at $(Int(GAISSER_HEIGHT))m · " *
                   "measured rock thickness (nm_c.inc) · θ ≤ $(round(Int, maximum(zeniths)))°</sub>",
            font = attr(size = 15)
        ),
        scene = attr(
            xaxis = attr(title = "Azimuth φ (°)", dtick = 30),
            yaxis = attr(title = "Zenith θ (°)",  dtick = 10),
            zaxis = attr(title = "log₁₀ Flux (m⁻²s⁻¹sr⁻¹)",
                         range = [zmin, zmax]),
            aspectmode  = "manual",
            aspectratio = attr(x = 1.5, y = 1.0, z = 0.6),
            camera      = attr(eye = attr(x = -1.6, y = -1.6, z = 0.8))
        ),
        width  = 1200,
        height = 900,
        legend = attr(x = 0.01, y = 0.99,
                      bgcolor = "rgba(255,255,255,0.85)",
                      font    = attr(size = 11))
    )

    fig = Plot(traces, layout)
    mkpath(dirname(output_path))
    savefig(fig, output_path)
    println("Saved plot: $output_path")
    return fig
end

function run_part1(physics;
                   n_samples::Int,
                   straggling::Bool,
                   scattering::Bool,
                   energy_threshold_low::Float64,
                   threshold_factors::Tuple{Float64,Float64},
                   energy_min::Float64,
                   energy_max::Float64,
                   porous_material::Int,
                   porous_density::Float64,
                   porous_thickness::Float64,
                   output_dir::String)

    println("=" ^ 60)
    println(" Part 1: Muon Flux on the θ ≤ 60° nmap Slice")
    println("=" ^ 60)
    println()
    println("  Rock density:     $(NMAP_ROCK_DENSITY) kg/m³ (rr.for)")
    println("  Gaisser altitude: $(Int(GAISSER_HEIGHT)) m above detector")
    println("  MC samples/bin:   $n_samples")
    println("  Zenith coverage:  0 – $(Int(PART1_ZENITH_MAX_DEG))°")
    println("  Energy range:     $energy_min – $energy_max GeV")
    println()

    flux_result = compute_nmap_flux_grid(physics;
            n_samples = n_samples,
            straggling = straggling, scattering = scattering,
            energy_threshold_low = energy_threshold_low,
            threshold_factors = threshold_factors,
            energy_min = energy_min, energy_max = energy_max,
            zenith_max_deg = PART1_ZENITH_MAX_DEG,
            porous_material = porous_material,
            porous_density  = porous_density,
            porous_thickness = porous_thickness)

    n_valid = count(!isnan, flux_result.flux)
    flux_valid = filter(f -> !isnan(f) && f > 0, flux_result.flux)
    if !isempty(flux_valid)
        println()
        @info @sprintf("Flux range: %.3e – %.3e m⁻²s⁻¹sr⁻¹ (%d valid bins)",
                       minimum(flux_valid), maximum(flux_valid), n_valid)
    end
    println()

    create_nmap_flux_plot(flux_result.zeniths, flux_result.azimuths,
                          flux_result.flux, flux_result.sigma,
                          flux_result.sigma_syst, flux_result.sigma_total,
                          flux_result.rock;
        output_path = joinpath(output_dir, "part1_nmap_flux.html"))

    println()
    return flux_result
end

function compute_direction_weight_grid(zeniths::Vector{Float64},
                                       azimuths::Vector{Float64},
                                       flux_grid::Matrix{Float64};
                                       zenith_upper_deg::Float64,
                                       azimuth_offset_deg::Float64 = 0.0,
                                       weight_mode::Symbol = :none)
    azimuths_u, flux_u = drop_duplicate_azimuth_column(azimuths, flux_grid)
    solid_angles = zenith_bin_solid_angles(zeniths; upper_deg=zenith_upper_deg)

    weights = zeros(Float64, size(flux_u))
    for (z_idx, θ) in enumerate(zeniths)
        for (a_idx, φ_geo) in enumerate(azimuths_u)
            flux = flux_u[z_idx, a_idx]
            (!isfinite(flux) || flux <= 0.0) && continue

            geom = if weight_mode == :none
                1.0
            elseif weight_mode == :lvd_box
                lvd_box_projected_area(θ, wrap_azimuth_deg(φ_geo + azimuth_offset_deg))
            else
                error("Unknown angular weight mode: $weight_mode")
            end

            weights[z_idx, a_idx] = flux * solid_angles[z_idx] * geom
        end
    end

    azimuths_shifted = wrap_azimuth_deg.(azimuths_u .+ azimuth_offset_deg)
    order = sortperm(azimuths_shifted)
    return azimuths_shifted[order], weights[:, order]
end

function compute_direction_uncertainty_terms(zeniths::Vector{Float64},
                                             azimuths::Vector{Float64},
                                             sigma_grid::Matrix{Float64};
                                             zenith_upper_deg::Float64,
                                             azimuth_offset_deg::Float64 = 0.0,
                                             weight_mode::Symbol = :none)
    azimuths_u, sigma_u = drop_duplicate_azimuth_column(azimuths, sigma_grid)
    solid_angles = zenith_bin_solid_angles(zeniths; upper_deg=zenith_upper_deg)

    weighted_sigma_sq = zeros(Float64, size(sigma_u))
    for (z_idx, θ) in enumerate(zeniths)
        for (a_idx, φ_geo) in enumerate(azimuths_u)
            sigma = sigma_u[z_idx, a_idx]
            (!isfinite(sigma) || sigma < 0.0) && continue

            geom = if weight_mode == :none
                1.0
            elseif weight_mode == :lvd_box
                lvd_box_projected_area(θ, wrap_azimuth_deg(φ_geo + azimuth_offset_deg))
            else
                error("Unknown angular weight mode: $weight_mode")
            end

            weighted_sigma_sq[z_idx, a_idx] = (sigma * solid_angles[z_idx] * geom)^2
        end
    end

    azimuths_shifted = wrap_azimuth_deg.(azimuths_u .+ azimuth_offset_deg)
    order = sortperm(azimuths_shifted)
    return azimuths_shifted[order], weighted_sigma_sq[:, order]
end

function compute_figure7_profile_with_uncertainty(flux_result::NMapFluxResult;
                                                  azimuth_offset_deg::Float64)
    z_end = findlast(θ -> θ <= PART1_ZENITH_MAX_DEG + 1e-9, flux_result.zeniths)
    z_end === nothing && error("No θ ≤ $(PART1_ZENITH_MAX_DEG)° rows available for Figure 7")

    zeniths = flux_result.zeniths[1:z_end]
    flux = flux_result.flux[1:z_end, :]
    sigma_mc = flux_result.sigma[1:z_end, :]
    sigma_syst = flux_result.sigma_syst[1:z_end, :]

    azimuths_lvd, weights = compute_direction_weight_grid(
        zeniths, flux_result.azimuths, flux;
        zenith_upper_deg=PART1_ZENITH_MAX_DEG,
        azimuth_offset_deg=azimuth_offset_deg,
        weight_mode=:lvd_box)
    _, mc_terms = compute_direction_uncertainty_terms(
        zeniths, flux_result.azimuths, sigma_mc;
        zenith_upper_deg=PART1_ZENITH_MAX_DEG,
        azimuth_offset_deg=azimuth_offset_deg,
        weight_mode=:lvd_box)
    _, syst_terms = compute_direction_uncertainty_terms(
        zeniths, flux_result.azimuths, sigma_syst;
        zenith_upper_deg=PART1_ZENITH_MAX_DEG,
        azimuth_offset_deg=azimuth_offset_deg,
        weight_mode=:lvd_box)

    profile = vec(sum(weights, dims=1)) / NMAP_AZIMUTH_STEP_DEG
    profile_sigma_mc = vec(sqrt.(sum(mc_terms, dims=1))) / NMAP_AZIMUTH_STEP_DEG
    profile_sigma_syst = vec(sqrt.(sum(syst_terms, dims=1))) / NMAP_AZIMUTH_STEP_DEG
    profile_sigma_total = sqrt.(profile_sigma_mc.^2 .+ profile_sigma_syst.^2)

    return azimuths_lvd, profile, profile_sigma_mc, profile_sigma_syst, profile_sigma_total
end

function compute_figure7_profile(flux_result::NMapFluxResult;
                                 azimuth_offset_deg::Float64)
    azimuths_lvd, profile, _, _, _ = compute_figure7_profile_with_uncertainty(
        flux_result; azimuth_offset_deg=azimuth_offset_deg)
    return azimuths_lvd, profile
end

function compute_figure9_occupancy(flux_result::NMapFluxResult;
                                   azimuth_offset_deg::Float64)
    azimuths_lvd, weights = compute_direction_weight_grid(
        flux_result.zeniths, flux_result.azimuths, flux_result.flux;
        zenith_upper_deg=maximum(flux_result.zeniths),
        azimuth_offset_deg=azimuth_offset_deg,
        weight_mode=:none)

    total = sum(weights)
    occupancy = total > 0.0 ? weights / total : weights
    return azimuths_lvd, occupancy
end

function load_paper_figure7_benchmark()
    curve_cols = load_csv_columns(PAPER_FIG7_CURVE_PATH, 2)
    point_cols = load_csv_columns(PAPER_FIG7_POINTS_PATH, 2)
    return (
        curve_az = curve_cols[1],
        curve_intensity = curve_cols[2],
        point_az = point_cols[1],
        point_intensity = point_cols[2],
    )
end

function load_paper_figure8_benchmark()
    cols = load_csv_columns(PAPER_FIG8_CURVE_PATH, 2)
    return (energy = cols[1], relative_count = cols[2])
end

function load_paper_figure9_peaks()
    cols = load_csv_columns(PAPER_FIG9_PEAKS_PATH, 3)
    peaks = NamedTuple{(:cos_zenith, :azimuth_deg, :relative_height),
                       Tuple{Float64, Float64, Float64}}[]
    for i in eachindex(cols[1])
        push!(peaks, (
            cos_zenith = cols[1][i],
            azimuth_deg = cols[2][i],
            relative_height = cols[3][i],
        ))
    end
    return peaks
end

function infer_lvd_reference_offset(flux_result::NMapFluxResult,
                                    paper_az::Vector{Float64},
                                    paper_intensity::Vector{Float64})
    function score_offset(offset_deg::Float64)
        model_az, model_profile = compute_figure7_profile(
            flux_result; azimuth_offset_deg=offset_deg)
        model_interp = periodic_interp(model_az, model_profile, paper_az)
        scale = fit_positive_scale(paper_intensity, model_interp)
        residual = scale .* model_interp .- paper_intensity
        rmse = sqrt(mean(residual .^ 2)) /
               max(maximum(paper_intensity), eps(Float64))
        return rmse, scale
    end

    best_offset = 0.0
    best_scale = 1.0
    best_rmse = Inf

    for offset_deg in 0.0:1.0:359.0
        rmse, scale = score_offset(offset_deg)
        if rmse < best_rmse
            best_offset = offset_deg
            best_scale = scale
            best_rmse = rmse
        end
    end

    for offset_deg in (best_offset - 2.0):0.25:(best_offset + 2.0)
        wrapped = wrap_azimuth_deg(offset_deg)
        rmse, scale = score_offset(wrapped)
        if rmse < best_rmse
            best_offset = wrapped
            best_scale = scale
            best_rmse = rmse
        end
    end

    return (offset_deg=best_offset, scale=best_scale, rmse=best_rmse)
end

function sample_paper_energy_spectrum(physics,
                                      flux_result::NMapFluxResult,
                                      rock_idx::Int,
                                      air_idx::Int;
                                      n_samples::Int,
                                      straggling::Bool,
                                      scattering::Bool,
                                      energy_threshold_low::Float64,
                                      energy_min::Float64,
                                      energy_max::Float64,
                                      porous_material::Int,
                                      porous_density::Float64,
                                      porous_thickness::Float64,
                                      seed::Int = 2026,
                                      verbose::Bool = true)
    energy_max > energy_min || error("Part 3 spectrum range must have energy_max > energy_min")

    azimuths_u, flux_u = drop_duplicate_azimuth_column(flux_result.azimuths, flux_result.flux)
    _, rock_u = drop_duplicate_azimuth_column(flux_result.azimuths, flux_result.rock)
    solid_angles = zenith_bin_solid_angles(
        flux_result.zeniths; upper_deg=maximum(flux_result.zeniths))

    angle_indices = Tuple{Int, Int}[]
    angle_weights = Float64[]
    for (z_idx, θ) in enumerate(flux_result.zeniths)
        for a_idx in eachindex(azimuths_u)
            flux = flux_u[z_idx, a_idx]
            rock = rock_u[z_idx, a_idx]
            (!isfinite(flux) || flux <= 0.0 || !isfinite(rock) || rock <= 0.0) && continue
            push!(angle_indices, (z_idx, a_idx))
            push!(angle_weights, flux * solid_angles[z_idx])
        end
    end

    isempty(angle_indices) && error("No valid angular bins available for Part 3 spectrum sampling")

    total_angle_weight = sum(angle_weights)
    cdf = cumsum(angle_weights) ./ total_angle_weight
    log_span = log(energy_max / energy_min)

    hist = zeros(Float64, length(PAPER_SPECTRUM_LOG10_EDGES) - 1)
    hist_w2 = zeros(Float64, length(PAPER_SPECTRUM_LOG10_EDGES) - 1)
    total_weight = 0.0
    weighted_energy = 0.0

    rng = MersenneTwister(seed)
    report_every = max(1, n_samples ÷ 10)

    for sample_idx in 1:n_samples
        if verbose && (sample_idx == 1 || sample_idx % report_every == 0 || sample_idx == n_samples)
            @info @sprintf("Part 3 spectrum sample %d / %d", sample_idx, n_samples)
        end

        draw = rand(rng)
        draw_idx = searchsortedfirst(cdf, draw)
        z_idx, a_idx = angle_indices[draw_idx]
        θ = flux_result.zeniths[z_idx]

        sampled_energy = energy_min * exp(log_span * rand(rng))
        energy_weight = sampled_energy * log_span
        charge = rand(rng) > 0.5 ? 1.0 : -1.0

        rock_slant = rock_u[z_idx, a_idx]
        depth = rock_slant * cosd(θ)
        p_thick = min(porous_thickness, depth)
        geometry = TwoLayerGeometry{Float64}(
            depth, NMAP_ROCK_DENSITY, rock_idx, air_idx,
            porous_material, porous_density, p_thick, GAISSER_HEIGHT)

        flux_differential, _ = compute_flux_single_with_state(
            physics, geometry, sampled_energy, 90.0 - θ, charge;
            rng=rng, straggling=straggling, scattering=scattering,
            energy_threshold_low=energy_threshold_low)

        flux_differential <= 0.0 && continue

        angle_probability = angle_weights[draw_idx] / total_angle_weight
        weight = flux_differential * solid_angles[z_idx] *
                 energy_weight * 2.0 / angle_probability

        loge = log10(sampled_energy)
        bin_idx = searchsortedlast(PAPER_SPECTRUM_LOG10_EDGES, loge)
        if bin_idx == length(PAPER_SPECTRUM_LOG10_EDGES)
            bin_idx -= 1
        end

        if 1 <= bin_idx <= length(hist)
            hist[bin_idx] += weight
            hist_w2[bin_idx] += weight^2
            total_weight += weight
            weighted_energy += weight * sampled_energy
        end
    end

    centers = [10.0^((PAPER_SPECTRUM_LOG10_EDGES[i] + PAPER_SPECTRUM_LOG10_EDGES[i + 1]) / 2)
               for i in 1:length(hist)]
    hist_sum = sum(hist)
    hist_norm = hist_sum > 0.0 ? hist / hist_sum : hist
    sigma_relative_count = hist_sum > 0.0 ? sqrt.(hist_w2) / hist_sum : zeros(Float64, length(hist))
    mean_energy = total_weight > 0.0 ? weighted_energy / total_weight : NaN

    return (
        energy = centers,
        relative_count = hist_norm,
        sigma_relative_count = sigma_relative_count,
        mean_energy = mean_energy,
    )
end

function identify_top_occupancy_peaks(occupancy::Matrix{Float64},
                                      zeniths::Vector{Float64},
                                      azimuths::Vector{Float64};
                                      n::Int = 12)
    candidates = Tuple{Float64, Float64, Float64}[]
    for (a_idx, azimuth_deg) in enumerate(azimuths)
        for (z_idx, θ) in enumerate(zeniths)
            value = occupancy[z_idx, a_idx]
            value <= 0.0 && continue
            push!(candidates, (value, cosd(θ), azimuth_deg))
        end
    end

    isempty(candidates) && return NamedTuple[]
    sort!(candidates; by = c -> -c[1])
    max_value = candidates[1][1]

    peaks = NamedTuple{(:cos_zenith, :azimuth_deg, :relative_height),
                       Tuple{Float64, Float64, Float64}}[]
    for (value, cos_zenith, azimuth_deg) in candidates
        separated = all(
            abs(cos_zenith - peak.cos_zenith) > 0.08 ||
            circular_distance_deg(azimuth_deg, peak.azimuth_deg) > 25.0
            for peak in peaks)
        separated || continue

        push!(peaks, (
            cos_zenith = cos_zenith,
            azimuth_deg = azimuth_deg,
            relative_height = value / max_value,
        ))
        length(peaks) >= n && break
    end

    return peaks
end

function compare_peak_sets(model_peaks, paper_peaks)
    (isempty(model_peaks) || isempty(paper_peaks)) &&
        return (matched_fraction=0.0, mean_distance=Inf)

    distances = Float64[]
    matched = 0
    for paper_peak in paper_peaks
        best = Inf
        for model_peak in model_peaks
            d_az = circular_distance_deg(model_peak.azimuth_deg, paper_peak.azimuth_deg) / 25.0
            d_cos = abs(model_peak.cos_zenith - paper_peak.cos_zenith) / 0.08
            best = min(best, sqrt(d_az^2 + d_cos^2))
        end
        push!(distances, best)
        best <= 1.0 && (matched += 1)
    end

    return (
        matched_fraction = matched / length(paper_peaks),
        mean_distance = mean(distances),
    )
end

function add_curve_bands!(traces::Vector{GenericTrace},
                          x_values,
                          y_values,
                          sigma_syst,
                          sigma_total;
                          xaxis::String,
                          yaxis::String,
                          total_fill::String,
                          syst_fill::String)
    total_upper = y_values .+ sigma_total
    total_lower = max.(y_values .- sigma_total, 1e-30)
    syst_upper = y_values .+ sigma_syst
    syst_lower = max.(y_values .- sigma_syst, 1e-30)
    transparent = "rgba(0,0,0,0)"

    push!(traces, scatter(
        x = x_values, y = total_upper,
        mode = "lines",
        line = attr(color = transparent, width = 0),
        hoverinfo = "skip",
        showlegend = false,
        xaxis = xaxis, yaxis = yaxis
    ))
    push!(traces, scatter(
        x = x_values, y = total_lower,
        mode = "lines",
        line = attr(color = transparent, width = 0),
        fill = "tonexty",
        fillcolor = total_fill,
        hoverinfo = "skip",
        showlegend = false,
        xaxis = xaxis, yaxis = yaxis
    ))
    push!(traces, scatter(
        x = x_values, y = syst_upper,
        mode = "lines",
        line = attr(color = transparent, width = 0),
        hoverinfo = "skip",
        showlegend = false,
        xaxis = xaxis, yaxis = yaxis
    ))
    push!(traces, scatter(
        x = x_values, y = syst_lower,
        mode = "lines",
        line = attr(color = transparent, width = 0),
        fill = "tonexty",
        fillcolor = syst_fill,
        hoverinfo = "skip",
        showlegend = false,
        xaxis = xaxis, yaxis = yaxis
    ))
end

function create_paper_reproduction_plot(fig7, fig8, fig9, metrics;
                                        output_path::String)
    fig7_model_scaled = metrics.fig7_scale .* fig7.model_profile
    fig7_sigma_mc = metrics.fig7_scale .* fig7.sigma_mc
    fig7_sigma_syst = metrics.fig7_scale .* fig7.sigma_syst
    fig7_sigma_total = metrics.fig7_scale .* fig7.sigma_total

    fig8_paper = [v > 0.0 ? v : NaN for v in fig8.paper_relative_count]
    fig8_model = [v > 0.0 ? v : NaN for v in fig8.model_relative_count]

    cos_zeniths = reverse(cosd.(fig9.zeniths))
    occupancy_heatmap = permutedims(reverse(fig9.occupancy, dims=1), (2, 1))

    traces = GenericTrace[]

    add_curve_bands!(traces, fig7.model_azimuths, fig7_model_scaled .* 1e9,
                     fig7_sigma_syst .* 1e9, fig7_sigma_total .* 1e9;
                     xaxis="x", yaxis="y",
                     total_fill="rgba(40,110,220,0.10)",
                     syst_fill="rgba(40,110,220,0.18)")
    add_curve_bands!(traces, fig8.model_energy, fig8.model_relative_count,
                     fig8.sigma_syst, fig8.sigma_total;
                     xaxis="x2", yaxis="y2",
                     total_fill="rgba(40,110,220,0.10)",
                     syst_fill="rgba(40,110,220,0.18)")

    push!(traces, scatter(
        x = fig7.paper_curve_azimuths,
        y = fig7.paper_curve_intensity .* 1e9,
        mode = "lines",
        line = attr(color = "rgb(190,40,40)", dash = "dash", width = 2),
        name = "Figure 7 paper curve",
        xaxis = "x",
        yaxis = "y"
    ))
    push!(traces, scatter(
        x = fig7.paper_point_azimuths,
        y = fig7.paper_point_intensity .* 1e9,
        mode = "markers",
        marker = attr(color = "black", size = 5),
        name = "Figure 7 LVD points",
        xaxis = "x",
        yaxis = "y"
    ))
    push!(traces, scatter(
        x = fig7.model_azimuths,
        y = fig7_model_scaled .* 1e9,
        mode = "lines",
        line = attr(color = "rgb(40,110,220)", width = 3),
        name = "DiffPumas model",
        text = [@sprintf("φ=%.1f°<br>Intensity=%.4e<br>MC=%.2f%%<br>Transport syst=%.2f%%<br>Total=%.2f%%",
                         x, y,
                         100 * mc / max(abs(y), 1e-30),
                         100 * syst / max(abs(y), 1e-30),
                         100 * total / max(abs(y), 1e-30))
                for (x, y, mc, syst, total) in zip(fig7.model_azimuths, fig7_model_scaled,
                                                   fig7_sigma_mc, fig7_sigma_syst, fig7_sigma_total)],
        hovertemplate = "%{text}<extra></extra>",
        xaxis = "x",
        yaxis = "y"
    ))

    push!(traces, scatter(
        x = fig8.paper_energy,
        y = fig8_paper,
        mode = "lines",
        line = attr(color = "rgb(190,40,40)", dash = "dash", width = 2),
        name = "Figure 8 paper curve",
        xaxis = "x2",
        yaxis = "y2",
        showlegend = false
    ))
    push!(traces, scatter(
        x = fig8.model_energy,
        y = fig8_model,
        mode = "lines",
        line = attr(color = "rgb(40,110,220)", width = 3),
        name = "DiffPumas spectrum",
        text = [@sprintf("E=%.3e GeV<br>Relative count=%.4e<br>MC=%.2f%%<br>Transport syst=%.2f%%<br>Total=%.2f%%",
                         x, y,
                         100 * mc / max(abs(y), 1e-30),
                         100 * syst / max(abs(y), 1e-30),
                         100 * total / max(abs(y), 1e-30))
                for (x, y, mc, syst, total) in zip(fig8.model_energy, fig8.model_relative_count,
                                                   fig8.sigma_mc, fig8.sigma_syst, fig8.sigma_total)],
        hovertemplate = "%{text}<extra></extra>",
        xaxis = "x2",
        yaxis = "y2",
        showlegend = false
    ))

    push!(traces, heatmap(
        x = cos_zeniths,
        y = fig9.azimuths,
        z = occupancy_heatmap,
        colorscale = "Viridis",
        colorbar = attr(title = "Relative occupancy", x = 1.01, len = 0.58),
        xaxis = "x3",
        yaxis = "y3",
        name = "Figure 9 occupancy",
        showscale = true
    ))
    push!(traces, scatter(
        x = [peak.cos_zenith for peak in fig9.paper_peaks],
        y = [peak.azimuth_deg for peak in fig9.paper_peaks],
        mode = "markers",
        marker = attr(symbol = "x", size = 10, color = "rgb(210,50,50)",
                      line = attr(width = 2, color = "rgb(210,50,50)")),
        name = "Figure 9 paper peaks",
        xaxis = "x3",
        yaxis = "y3"
    ))
    push!(traces, scatter(
        x = [peak.cos_zenith for peak in fig9.model_peaks],
        y = [peak.azimuth_deg for peak in fig9.model_peaks],
        mode = "markers",
        marker = attr(symbol = "circle-open", size = 10,
                      color = "rgba(255,255,255,0.2)",
                      line = attr(width = 2, color = "black")),
        name = "DiffPumas peaks",
        xaxis = "x3",
        yaxis = "y3"
    ))

    annotations = [
        attr(x = 0.22, y = 1.03, xref = "paper", yref = "paper",
             text = "Figure 7: Azimuthal intensity", showarrow = false,
             font = attr(size = 15)),
        attr(x = 0.22, y = 0.45, xref = "paper", yref = "paper",
             text = "Figure 8: Underground energy spectrum", showarrow = false,
             font = attr(size = 15)),
        attr(x = 0.79, y = 1.03, xref = "paper", yref = "paper",
             text = "Figure 9: Angular occupancy", showarrow = false,
             font = attr(size = 15)),
        attr(x = 0.22, y = 0.50, xref = "paper", yref = "paper",
             text = @sprintf("Offset %.2f° · Fig. 7 NRMSE %.3f", metrics.azimuth_offset_deg, metrics.fig7_nrmse),
             showarrow = false, font = attr(size = 12)),
        attr(x = 0.22, y = -0.04, xref = "paper", yref = "paper",
             text = @sprintf("Mean E = %.1f GeV (paper %.0f GeV) · Fig. 8 corr %.3f",
                             metrics.mean_energy_gev, PAPER_MEAN_ENERGY_GEV, metrics.fig8_corr),
             showarrow = false, font = attr(size = 12)),
        attr(x = 0.79, y = -0.04, xref = "paper", yref = "paper",
             text = @sprintf("Peak match %.0f%% · mean scaled distance %.2f",
                             100 * metrics.fig9_match_fraction, metrics.fig9_mean_distance),
             showarrow = false, font = attr(size = 12)),
    ]

    layout = Layout(
        title = attr(
            text = "Gran Sasso LVD — Paper Reproduction Check<br>" *
                   "<sub>Figure 7 uses an inferred LVD-frame offset and a simple box acceptance model; " *
                   "Figures 8–9 use the raw underground angular flux</sub>",
            font = attr(size = 16)
        ),
        xaxis = attr(domain = [0.00, 0.46], anchor = "y",
                     title = "Azimuth φ_LVD (°)", range = [0, 360], dtick = 60),
        yaxis = attr(domain = [0.58, 1.00], anchor = "x",
                     title = "Intensity (10⁻⁹ cm⁻² s⁻¹ deg⁻¹)"),
        xaxis2 = attr(domain = [0.00, 0.46], anchor = "y2",
                      title = "Muon energy (GeV)", type = "log"),
        yaxis2 = attr(domain = [0.00, 0.42], anchor = "x2",
                      title = "Normalized counts", type = "log"),
        xaxis3 = attr(domain = [0.58, 1.00], anchor = "y3",
                      title = "cos(θ)", range = [0, 1]),
        yaxis3 = attr(domain = [0.00, 1.00], anchor = "x3",
                      title = "Azimuth φ_LVD (°)", range = [0, 360], dtick = 60),
        annotations = annotations,
        width = 1450,
        height = 920,
        legend = attr(x = 0.60, y = 0.98,
                      bgcolor = "rgba(255,255,255,0.85)",
                      font = attr(size = 11))
    )

    fig = Plot(traces, layout)
    mkpath(dirname(output_path))
    savefig(fig, output_path)
    println("Saved plot: $output_path")
    return fig
end

function write_paper_metrics(metrics; output_path::String)
    lines = [
        "Gran Sasso LVD paper reproduction summary",
        "",
        @sprintf("Inferred LVD azimuth offset: %.2f deg", metrics.azimuth_offset_deg),
        @sprintf("Figure 7 normalized RMSE: %.4f", metrics.fig7_nrmse),
        @sprintf("Figure 8 mean energy: %.2f GeV", metrics.mean_energy_gev),
        @sprintf("Figure 8 delta vs paper mean (%.0f GeV): %.2f GeV",
                 PAPER_MEAN_ENERGY_GEV, metrics.mean_energy_gev - PAPER_MEAN_ENERGY_GEV),
        @sprintf("Figure 8 delta vs measurement (%.0f ± %.0f stat ± %.0f syst GeV): %.2f GeV",
                 PAPER_MEAN_ENERGY_MEAS_GEV,
                 PAPER_MEAN_ENERGY_MEAS_STAT_GEV,
                 PAPER_MEAN_ENERGY_MEAS_SYST_GEV,
                 metrics.mean_energy_gev - PAPER_MEAN_ENERGY_MEAS_GEV),
        @sprintf("Figure 8 shape correlation: %.4f", metrics.fig8_corr),
        @sprintf("Figure 9 matched paper peaks: %.1f%%",
                 100 * metrics.fig9_match_fraction),
        @sprintf("Figure 9 mean scaled peak distance: %.4f",
                 metrics.fig9_mean_distance),
        "",
        "Assumptions:",
        "- All three parts use StandardRock only; no shallow wet-rock or porous composite layer is applied.",
        "- Part 3 uses the same θ ≤ 60° zenith range as Parts 1 and 2.",
        "- Figure 7 uses a simple box projected-area model for the LVD acceptance.",
        "- Figures 8 and 9 use the raw underground angular flux without detector acceptance weighting.",
        "- The LVD reference-system offset is inferred by matching the paper's Figure 7 azimuthal curve.",
    ]

    open(output_path, "w") do io
        println(io, join(lines, '\n'))
    end
    println("Saved summary: $output_path")
end

function write_part3_status(output_dir::String, stage::String;
                            details::Vector{String} = String[])
    mkpath(output_dir)
    status_path = joinpath(output_dir, "part3_status.txt")
    open(status_path, "w") do io
        println(io, "Gran Sasso LVD Part 3 status")
        println(io)
        println(io, "Stage: $stage")
        for line in details
            println(io, line)
        end
    end
    println("Saved status: $status_path")
end

function run_part3(physics, rock_idx::Int, air_idx::Int;
                   flux_result::Union{Nothing, NMapFluxResult},
                   n_samples::Int,
                   paper_samples::Int,
                   straggling::Bool,
                   scattering::Bool,
                   energy_threshold_low::Float64,
                   threshold_factors::Tuple{Float64,Float64},
                   energy_min::Float64,
                   energy_max::Float64,
                   porous_material::Int,
                   porous_density::Float64,
                   porous_thickness::Float64,
                   output_dir::String)
    println("=" ^ 60)
    println(" Part 3: Reproduce the LVD Paper Observables")
    println("=" ^ 60)
    println()
    println("  Paper MC samples: $paper_samples")
    println("  Full zenith range: 0 – $(Int(PAPER_FULL_ZENITH_MAX_DEG))°")
    println()

    target_zenith_max_deg = PAPER_FULL_ZENITH_MAX_DEG
    write_part3_status(output_dir, "starting";
        details = [
            "paper_samples = $paper_samples",
            @sprintf("full_zenith_range = 0 – %d deg", Int(PAPER_FULL_ZENITH_MAX_DEG)),
            flux_result === nothing ?
                "input_flux_grid = none" :
                @sprintf("input_flux_grid_max_zenith = %.0f deg", maximum(flux_result.zeniths)),
        ])

    full_flux_result = if flux_result === nothing
        println("  Building full 0°–$(Int(target_zenith_max_deg))° transport grid for paper checks...")
        write_part3_status(output_dir, "building full flux grid";
            details = [
                "mode = fresh build",
                @sprintf("computing rows from 0 to %.0f deg", target_zenith_max_deg),
            ])
        compute_nmap_flux_grid(physics;
            n_samples = n_samples,
            straggling = straggling,
            scattering = scattering,
            energy_threshold_low = energy_threshold_low,
            threshold_factors = threshold_factors,
            energy_min = energy_min,
            energy_max = energy_max,
            zenith_max_deg = target_zenith_max_deg,
            porous_material = porous_material,
            porous_density = porous_density,
            porous_thickness = porous_thickness,
            progress_note = "Part 3")
    elseif maximum(flux_result.zeniths) < target_zenith_max_deg - 1e-9
        zenith_resume = maximum(flux_result.zeniths) + NMAP_ZENITH_STEP_DEG
        println(@sprintf("  Extending existing flux grid from %.0f° to %.0f° for paper checks...",
                         zenith_resume, target_zenith_max_deg))
        write_part3_status(output_dir, "extending flux grid";
            details = [
                @sprintf("reusing rows through %.0f deg", maximum(flux_result.zeniths)),
                @sprintf("computing rows from %.0f to %.0f deg", zenith_resume, target_zenith_max_deg),
            ])
        extension = compute_nmap_flux_grid(physics;
            n_samples = n_samples,
            straggling = straggling,
            scattering = scattering,
            energy_threshold_low = energy_threshold_low,
            threshold_factors = threshold_factors,
            energy_min = energy_min,
            energy_max = energy_max,
            zenith_min_deg = zenith_resume,
            zenith_max_deg = target_zenith_max_deg,
            porous_material = porous_material,
            porous_density = porous_density,
            porous_thickness = porous_thickness,
            progress_note = "Part 3 extension")
        merge_nmap_flux_results(flux_result, extension)
    else
        println("  Reusing an existing flux grid for paper checks.")
        write_part3_status(output_dir, "reusing full flux grid";
            details = [
                @sprintf("rows already cover %.0f deg", maximum(flux_result.zeniths)),
            ])
        flux_result
    end

    write_part3_status(output_dir, "flux grid ready";
        details = [
            @sprintf("zenith_rows = %d", length(full_flux_result.zeniths)),
            @sprintf("valid_flux_bins = %d",
                     count(x -> !isnan(x) && x > 0.0, full_flux_result.flux)),
        ])

    fig7_benchmark = load_paper_figure7_benchmark()
    fig8_benchmark = load_paper_figure8_benchmark()
    fig9_paper_peaks = load_paper_figure9_peaks()

    offset_fit = infer_lvd_reference_offset(
        full_flux_result,
        fig7_benchmark.curve_az,
        fig7_benchmark.curve_intensity)

    println(@sprintf("  Inferred LVD azimuth offset: %.2f°", offset_fit.offset_deg))

    fig7_model_azimuths, fig7_model_profile, fig7_sigma_mc, fig7_sigma_syst, fig7_sigma_total =
        compute_figure7_profile_with_uncertainty(
            full_flux_result; azimuth_offset_deg=offset_fit.offset_deg)
    fig7_interp = periodic_interp(
        fig7_model_azimuths, fig7_model_profile, fig7_benchmark.curve_az)
    fig7_scale = fit_positive_scale(fig7_benchmark.curve_intensity, fig7_interp)
    fig7_nrmse = sqrt(mean((fig7_scale .* fig7_interp .-
                            fig7_benchmark.curve_intensity) .^ 2)) /
                 max(maximum(fig7_benchmark.curve_intensity), eps(Float64))

    write_part3_status(output_dir, "figure 7 fit ready";
        details = [
            @sprintf("azimuth_offset_deg = %.2f", offset_fit.offset_deg),
            @sprintf("fig7_nrmse = %.4f", fig7_nrmse),
        ])

    spectrum_energy_min = max(energy_min, 10.0^first(PAPER_SPECTRUM_LOG10_EDGES))
    spectrum_energy_max = min(energy_max, 10.0^last(PAPER_SPECTRUM_LOG10_EDGES))
    write_part3_status(output_dir, "sampling energy spectrum";
        details = [
            @sprintf("paper_samples = %d", paper_samples),
            @sprintf("energy_range = %.3e – %.3e GeV", spectrum_energy_min, spectrum_energy_max),
        ])
    spectrum_nominal = sample_paper_energy_spectrum(
        physics, full_flux_result, rock_idx, air_idx;
        n_samples = paper_samples,
        straggling = straggling,
        scattering = scattering,
        energy_threshold_low = energy_threshold_low,
        energy_min = spectrum_energy_min,
        energy_max = spectrum_energy_max,
        porous_material = porous_material,
        porous_density = porous_density,
        porous_thickness = porous_thickness,
        seed = 2026,
        verbose = true)

    function spectrum_evaluate(variation)
        result = if variation.straggling == straggling &&
                    variation.scattering == scattering &&
                    isapprox(variation.energy_threshold_low, energy_threshold_low; atol=1e-12, rtol=1e-12) &&
                    variation.seed == 2026
            spectrum_nominal
        else
            sample_paper_energy_spectrum(
                physics, full_flux_result, rock_idx, air_idx;
                n_samples = paper_samples,
                straggling = variation.straggling,
                scattering = variation.scattering,
                energy_threshold_low = variation.energy_threshold_low,
                energy_min = spectrum_energy_min,
                energy_max = spectrum_energy_max,
                porous_material = porous_material,
                porous_density = porous_density,
                porous_thickness = porous_thickness,
                seed = variation.seed,
                verbose = false,
            )
        end
        return result.relative_count, result.sigma_relative_count
    end

    spectrum_budget = estimate_transport_uncertainty(
        spectrum_evaluate;
        straggling = straggling,
        scattering = scattering,
        energy_threshold_low = energy_threshold_low,
        seed = 2026,
        threshold_factors = threshold_factors,
    )

    fig8_mask = (fig8_benchmark.relative_count .> 0.0) .& (spectrum_budget.value .> 0.0)
    fig8_corr = count(fig8_mask) >= 2 ?
        cor(log10.(fig8_benchmark.relative_count[fig8_mask]),
            log10.(spectrum_budget.value[fig8_mask])) : NaN

    write_part3_status(output_dir, "spectrum ready";
        details = [
            @sprintf("mean_energy_gev = %.2f", spectrum_nominal.mean_energy),
            @sprintf("fig8_corr = %.4f", fig8_corr),
        ])

    fig9_azimuths, fig9_occupancy = compute_figure9_occupancy(
        full_flux_result; azimuth_offset_deg=offset_fit.offset_deg)
    fig9_model_peaks = identify_top_occupancy_peaks(
        fig9_occupancy, full_flux_result.zeniths, fig9_azimuths;
        n = length(fig9_paper_peaks))
    fig9_metrics = compare_peak_sets(fig9_model_peaks, fig9_paper_peaks)

    metrics = (
        azimuth_offset_deg = offset_fit.offset_deg,
        fig7_scale = fig7_scale,
        fig7_nrmse = fig7_nrmse,
        mean_energy_gev = spectrum_nominal.mean_energy,
        fig8_corr = fig8_corr,
        fig9_match_fraction = fig9_metrics.matched_fraction,
        fig9_mean_distance = fig9_metrics.mean_distance,
    )

    write_part3_status(output_dir, "finalizing outputs";
        details = [
            @sprintf("fig7_nrmse = %.4f", metrics.fig7_nrmse),
            @sprintf("mean_energy_gev = %.2f", metrics.mean_energy_gev),
            @sprintf("fig8_corr = %.4f", metrics.fig8_corr),
            @sprintf("fig9_match_fraction = %.4f", metrics.fig9_match_fraction),
            @sprintf("fig9_mean_distance = %.4f", metrics.fig9_mean_distance),
        ])

    create_paper_reproduction_plot(
        (
            paper_curve_azimuths = fig7_benchmark.curve_az,
            paper_curve_intensity = fig7_benchmark.curve_intensity,
            paper_point_azimuths = fig7_benchmark.point_az,
            paper_point_intensity = fig7_benchmark.point_intensity,
            model_azimuths = fig7_model_azimuths,
            model_profile = fig7_model_profile,
            sigma_mc = fig7_sigma_mc,
            sigma_syst = fig7_sigma_syst,
            sigma_total = fig7_sigma_total,
        ),
        (
            paper_energy = fig8_benchmark.energy,
            paper_relative_count = fig8_benchmark.relative_count,
            model_energy = spectrum_nominal.energy,
            model_relative_count = spectrum_budget.value,
            sigma_mc = spectrum_budget.sigma_mc,
            sigma_syst = spectrum_budget.sigma_syst,
            sigma_total = spectrum_budget.sigma_total,
        ),
        (
            zeniths = full_flux_result.zeniths,
            azimuths = fig9_azimuths,
            occupancy = fig9_occupancy,
            paper_peaks = fig9_paper_peaks,
            model_peaks = fig9_model_peaks,
        ),
        metrics;
        output_path = joinpath(output_dir, "part3_paper_reproduction.html"))

    write_paper_metrics(metrics;
        output_path = joinpath(output_dir, "part3_paper_metrics.txt"))
    write_part3_status(output_dir, "complete";
        details = [
            @sprintf("fig7_nrmse = %.4f", metrics.fig7_nrmse),
            @sprintf("mean_energy_gev = %.2f", metrics.mean_energy_gev),
            @sprintf("fig8_corr = %.4f", metrics.fig8_corr),
            @sprintf("fig9_match_fraction = %.4f", metrics.fig9_match_fraction),
            @sprintf("fig9_mean_distance = %.4f", metrics.fig9_mean_distance),
        ])
    println()
    return full_flux_result
end

# =============================================================================
# Main
# =============================================================================

function parse_commandline()
    dump_path = DEFAULT_DUMP
    n_samples = 1000
    paper_samples = 20_000
    energy_threshold_low = 100.0
    threshold_scan_low = 0.5
    threshold_scan_high = 2.0
    energy_min = 1e-3
    energy_max = 1e9
    straggling = true
    scattering = true
    output_dir = nothing
    part = -1

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg in ("--dump", "-d")
            dump_path = ARGS[i + 1]; i += 2
        elseif arg in ("--samples", "-n")
            n_samples = parse(Int, ARGS[i + 1]); i += 2
        elseif arg == "--paper-samples"
            paper_samples = parse(Int, ARGS[i + 1]); i += 2
        elseif arg == "--threshold"
            energy_threshold_low = parse(Float64, ARGS[i + 1]); i += 2
        elseif arg == "--threshold-scan-low"
            threshold_scan_low = parse(Float64, ARGS[i + 1]); i += 2
        elseif arg == "--threshold-scan-high"
            threshold_scan_high = parse(Float64, ARGS[i + 1]); i += 2
        elseif arg == "--energy-min"
            energy_min = parse(Float64, ARGS[i + 1]); i += 2
        elseif arg == "--energy-max"
            energy_max = parse(Float64, ARGS[i + 1]); i += 2
        elseif arg == "--no-straggling"
            straggling = false; i += 1
        elseif arg == "--no-scattering"
            scattering = false; i += 1
        elseif arg in ("--output-dir", "-o")
            output_dir = ARGS[i + 1]; i += 2
        elseif arg == "--part"
            part = parse(Int, ARGS[i + 1]); i += 2
        elseif arg in ("--help", "-h")
            println(@doc lvd_muography)
            exit(0)
        else
            error("Unknown argument: $arg")
        end
    end

    output_dir === nothing && error("--output-dir is required.")

    return (dump_path=dump_path, n_samples=n_samples, paper_samples=paper_samples,
            energy_threshold_low=energy_threshold_low,
            threshold_factors=(threshold_scan_low, threshold_scan_high),
            energy_min=energy_min, energy_max=energy_max,
            straggling=straggling, scattering=scattering,
            output_dir=output_dir, part=part)
end

function main()
    args = parse_commandline()

    println("=" ^ 60)
    println(" DiffPumas — Gran Sasso LVD Topography & Muon Flux")
    println("=" ^ 60)
    println()
    println("Configuration:")
    println("  Dump file:       $(args.dump_path)")
    println("  MC samples:      $(args.n_samples)")
    println("  Paper samples:   $(args.paper_samples)")
    println("  Threshold:       $(args.energy_threshold_low) GeV")
    println("  Threshold scan:  ×$(args.threshold_factors[1]) / ×$(args.threshold_factors[2])")
    println("  Energy range:    $(args.energy_min) – $(args.energy_max) GeV")
    println("  Straggling:      $(args.straggling ? "enabled" : "disabled")")
    println("  Scattering:      $(args.scattering ? "enabled" : "disabled")")
    println("  Output dir:      $(args.output_dir)")
    part_str = args.part < 0 ? "all" : string(args.part)
    println("  Running part:    $part_str")
    println()

    mkpath(args.output_dir)

    physics = load_or_create_physics(args.dump_path)
    if physics === nothing
        println("ERROR: Failed to load physics"); return 1
    end
    print_physics_summary(physics)
    println()

    rock_idx   = get_material_index(physics, "StandardRock")
    air_idx    = get_material_index(physics, "Air")

    if rock_idx == -1 || air_idx == -1
        println("ERROR: Required materials (StandardRock, Air) not found"); return 1
    end

    println("Material indices: rock=$rock_idx, air=$air_idx")
    println("  Material model: StandardRock only (no porous/wet-rock layer)")
    println()

    porous_material = -1
    porous_density = 0.0
    porous_thickness = 0.0

    # ── Part 1: Muon flux on nmap angular grid (runs first) ──────────────
    part1_flux_result = nothing

    if args.part < 0 || args.part == 1
        part1_flux_result = run_part1(physics;
            n_samples = args.n_samples,
            straggling = args.straggling,
            scattering = args.scattering,
            energy_threshold_low = args.energy_threshold_low,
            threshold_factors = args.threshold_factors,
            energy_min = args.energy_min, energy_max = args.energy_max,
            porous_material = porous_material, porous_density = porous_density,
            porous_thickness = porous_thickness,
            output_dir = args.output_dir)
    end

    # ── Part 2: 3D topography & trajectories (uses Part 1 directions) ───
    if args.part < 0 || args.part == 2
        if part1_flux_result === nothing
            println("Computing flux grid for trajectory direction selection...")
            part1_flux_result = compute_nmap_flux_grid(physics;
                    n_samples = args.n_samples,
                    straggling = args.straggling, scattering = args.scattering,
                    energy_threshold_low = args.energy_threshold_low,
                    threshold_factors = args.threshold_factors,
                    energy_min = args.energy_min, energy_max = args.energy_max,
                    zenith_max_deg = PART1_ZENITH_MAX_DEG,
                    porous_material = porous_material, porous_density = porous_density,
                    porous_thickness = porous_thickness)
        end

        directions = select_diverse_directions(
            part1_flux_result.flux, part1_flux_result.zeniths, part1_flux_result.azimuths)
        if isempty(directions)
            println("WARNING: no valid directions found, skipping Part 2"); return 1
        end

        println("Building elevation map from measured rock thickness (nm_c.inc)...")
        emap = build_elevation_map()
        info, _ = map_meta(emap)
        println("  Grid: $(info.nx) × $(info.ny)")
        elev_above, _ = map_elevation(emap, 0.0, 0.0)
        println("  Surface above detector: $(round(elev_above; digits=0)) m ASL")
        println("  Rock overburden: $(round(elev_above - DETECTOR_ELEVATION; digits=0)) m")
        println()

        run_part2(physics, rock_idx, air_idx, emap, directions;
            straggling=args.straggling, scattering=args.scattering,
            energy_threshold_low=args.energy_threshold_low,
            porous_material=porous_material, porous_density=porous_density,
            porous_thickness=porous_thickness, output_dir=args.output_dir)
    end

    # ── Part 3: LVD paper reproduction checks ────────────────────────────
    if args.part < 0 || args.part == 3
        run_part3(physics, rock_idx, air_idx;
            flux_result=part1_flux_result,
            n_samples=args.n_samples,
            paper_samples=args.paper_samples,
            straggling=args.straggling,
            scattering=args.scattering,
            energy_threshold_low=args.energy_threshold_low,
            threshold_factors=args.threshold_factors,
            energy_min=args.energy_min,
            energy_max=args.energy_max,
            porous_material=porous_material,
            porous_density=porous_density,
            porous_thickness=porous_thickness,
            output_dir=args.output_dir)
    end

    println()
    println("=" ^ 60)
    println(" All parts complete!")
    println("=" ^ 60)
    println()
    println("Output files saved to: $(args.output_dir)")
    println()
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
