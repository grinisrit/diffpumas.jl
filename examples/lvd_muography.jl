#!/usr/bin/env julia
"""
lvd_muography.jl — Gran Sasso LVD directional muon flux & 3D topography

Part 1 — Directional muon flux on the 48×91 nmap angular grid
  Backward-MC flux computed for each non-underground (θ, φ) bin using the
  slant rock distance R(θ,φ) from nm_c.inc, converted to vertical depth
  D = R·cos θ.  Rock density 2.71 g/cm³ (matching rr.for line 13).
  Gaisser primary flux sampled 10 000 m above the detector.
  Result saved as an interactive 3D surface of log₁₀(flux).

Part 2 — 3D topography & trajectory visualisation
  Elevation map derived from nm_c.inc measured rock thicknesses (48×91, 2°×4°),
  inverted via ray–surface bisection.  Six diverse trajectory directions are
  selected from the Part 1 flux grid (one per 60° azimuth sector, picking
  high-flux bins at moderate zenith).  Geometric rays and backward-MC muon
  trajectories are plotted over an interactive 3D terrain surface.

The dgsm table in nm_c.inc stores slant rock thickness in metres for each
(zenith, azimuth) direction from the LVD detector.  The `nmap` lookup adds
180° to the physical azimuth before indexing (rr.for line 19).  Values above
100 000 flag underground directions.

Materials: Standard Rock (2710 kg/m³ for flux, 2650 for transport tables),
           PorousWetRock (composite, top 100 m).

For flat-geometry baseline flux studies and aquifer-detection scenarios, see
flat_muography.jl.

Usage:
    julia --project=. examples/lvd_muography.jl --output-dir PATH [OPTIONS]

Options:
    --output-dir, -o PATH     Output directory for plots (REQUIRED)
    --dump, -d PATH           Path to physics binary dump file
    --samples, -n INT         MC samples per flux bin (default: 1000)
    --threshold FLOAT         Energy threshold for mode switching (default: 100.0)
    --energy-min FLOAT        Minimum energy in GeV (default: 1e-3)
    --energy-max FLOAT        Maximum energy in GeV (default: 1e9)
    --no-straggling           Disable straggling
    --no-scattering           Disable scattering
    --part INT                Run only part 1 or 2 (default: all)

Examples:
    julia --project=. examples/lvd_muography.jl -o out --part 1 --samples 500
    julia --project=. examples/lvd_muography.jl -o out --part 2
"""

using DiffPumas
using DiffPumas.Physics: get_material_index, property_range, ENERGY_LOSS_CSDA
using DiffPumas.Loader: print_physics_summary
using DiffPumas.Geometry: PRIMARY_ALTITUDE, compute_flux, locals_air_at_altitude
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

const DEFAULT_DUMP = joinpath(@__DIR__, "data", "materials.pumas")
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
selects a random valid (non-NaN, θ ∈ 10°–65°) bin.
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
                (θ < 10.0 || θ > 65.0) && continue
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
    compute_nmap_flux_grid(physics; kwargs...) -> (zeniths, azimuths, flux, sigma, rock_m)

Compute backward-MC muon flux on the 48×91 angular grid matching the dgsm
rock-thickness table.  For each non-underground (θ, φ) bin the slant rock
distance R is converted to a vertical depth `D = R·cos θ` and fed to
`compute_flux` with ρ = 2710 kg/m³ (rr.for convention).

Returns zenith (48-vector), azimuth (91-vector), flux (48×91), sigma (48×91),
and the raw slant rock thickness (48×91, NaN for underground).
"""
function compute_nmap_flux_grid(physics;
                                n_samples::Int = 1000,
                                straggling::Bool = true,
                                scattering::Bool = true,
                                energy_threshold_low::Float64 = 100.0,
                                energy_min::Float64 = 1e-3,
                                energy_max::Float64 = 1e9,
                                porous_material::Int = -1,
                                porous_density::Float64 = 0.0,
                                porous_thickness::Float64 = 0.0)

    dgsm = load_dgsm(NMC_PATH)

    n_zen = 48
    n_azi = 91
    zeniths  = [2.0 * (i - 1) for i in 1:n_zen]
    azimuths = [4.0 * (j - 1) for j in 1:n_azi]

    flux_grid  = fill(NaN, n_zen, n_azi)
    sigma_grid = fill(NaN, n_zen, n_azi)
    rock_grid  = fill(NaN, n_zen, n_azi)

    n_total   = n_zen * n_azi
    n_done    = 0
    n_skipped = 0

    for (z_idx, θ) in enumerate(zeniths)
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

            if n_done % 200 == 0 || n_done == 1
                @info @sprintf("[%d/%d] θ=%.0f° φ=%.0f° R=%.0fm D=%.0fm",
                               n_done, n_total, θ, φ, gsroc, depth)
            end

            flux, sigma = compute_flux(physics,
                NMAP_ROCK_DENSITY, depth, elevation, energy_min, energy_max;
                n_samples    = n_samples,
                straggling   = straggling,
                scattering   = scattering,
                energy_threshold_low = energy_threshold_low,
                porous_material  = porous_material,
                porous_density   = porous_density,
                porous_thickness = p_thick,
                primary_altitude = GAISSER_HEIGHT)

            flux_grid[z_idx, a_idx]  = flux
            sigma_grid[z_idx, a_idx] = sigma
        end
    end

    @info @sprintf("Computed %d bins (%d underground/skipped)", n_done, n_skipped)
    return zeniths, azimuths, flux_grid, sigma_grid, rock_grid
end

"""
    create_nmap_flux_plot(zeniths, azimuths, flux_grid, rock_grid; output_path)

3D PlotlyJS surface of log₁₀(flux) over the (azimuth, zenith) angular grid,
with a companion surface showing the raw slant rock thickness.
"""
function create_nmap_flux_plot(zeniths::Vector{Float64},
                               azimuths::Vector{Float64},
                               flux_grid::Matrix{Float64},
                               rock_grid::Matrix{Float64};
                               output_path::String)

    log_flux = map(f -> (isnan(f) || f <= 0.0) ? NaN : log10(f), flux_grid)

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
        hovertemplate = "φ=%{x:.0f}°<br>θ=%{y:.0f}°<br>" *
                        "log₁₀Φ=%{z:.2f}<extra></extra>"
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
            text = "Gran Sasso LVD — Muon Flux on 48×91 nmap Grid<br>" *
                   "<sub>ρ=$(NMAP_ROCK_DENSITY) kg/m³ · Gaisser at $(Int(GAISSER_HEIGHT))m · " *
                   "measured rock thickness (nm_c.inc)</sub>",
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
                   energy_min::Float64,
                   energy_max::Float64,
                   porous_material::Int,
                   porous_density::Float64,
                   porous_thickness::Float64,
                   output_dir::String)

    println("=" ^ 60)
    println(" Part 1: Muon Flux on 48×91 nmap Angular Grid")
    println("=" ^ 60)
    println()
    println("  Rock density:     $(NMAP_ROCK_DENSITY) kg/m³ (rr.for)")
    println("  Gaisser altitude: $(Int(GAISSER_HEIGHT)) m above detector")
    println("  MC samples/bin:   $n_samples")
    println("  Energy range:     $energy_min – $energy_max GeV")
    println()

    zeniths, azimuths, flux_grid, sigma_grid, rock_grid =
        compute_nmap_flux_grid(physics;
            n_samples = n_samples,
            straggling = straggling, scattering = scattering,
            energy_threshold_low = energy_threshold_low,
            energy_min = energy_min, energy_max = energy_max,
            porous_material = porous_material,
            porous_density  = porous_density,
            porous_thickness = porous_thickness)

    n_valid = count(!isnan, flux_grid)
    flux_valid = filter(f -> !isnan(f) && f > 0, flux_grid)
    if !isempty(flux_valid)
        println()
        @info @sprintf("Flux range: %.3e – %.3e m⁻²s⁻¹sr⁻¹ (%d valid bins)",
                       minimum(flux_valid), maximum(flux_valid), n_valid)
    end
    println()

    create_nmap_flux_plot(zeniths, azimuths, flux_grid, rock_grid;
        output_path = joinpath(output_dir, "part1_nmap_flux.html"))

    println()
    return zeniths, azimuths, flux_grid
end

# =============================================================================
# Main
# =============================================================================

function parse_commandline()
    dump_path = DEFAULT_DUMP
    n_samples = 1000
    energy_threshold_low = 100.0
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
        elseif arg == "--threshold"
            energy_threshold_low = parse(Float64, ARGS[i + 1]); i += 2
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

    return (dump_path=dump_path, n_samples=n_samples,
            energy_threshold_low=energy_threshold_low,
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
    println("  Threshold:       $(args.energy_threshold_low) GeV")
    println("  Energy range:    $(args.energy_min) – $(args.energy_max) GeV")
    println("  Straggling:      $(args.straggling ? "enabled" : "disabled")")
    println("  Scattering:      $(args.scattering ? "enabled" : "disabled")")
    println("  Output dir:      $(args.output_dir)")
    part_str = args.part < 0 ? "all" : string(args.part)
    println("  Running part:    $part_str")
    println()

    mkpath(args.output_dir)

    physics = load_or_create_physics(args.dump_path; mdf_path=DEFAULT_MDF)
    if physics === nothing
        println("ERROR: Failed to load physics"); return 1
    end
    print_physics_summary(physics)
    println()

    rock_idx   = get_material_index(physics, "StandardRock")
    air_idx    = get_material_index(physics, "Air")
    porous_idx = get_material_index(physics, "PorousWetRock")

    if rock_idx == -1 || air_idx == -1
        println("ERROR: Required materials (StandardRock, Air) not found"); return 1
    end

    println("Material indices: rock=$rock_idx, air=$air_idx")
    if porous_idx != -1
        println("  PorousWetRock (composite): $porous_idx " *
                "(density=$(round(physics.tables[porous_idx].density; digits=1)) kg/m³)")
    end
    println()

    p_density = porous_idx != -1 ? Float64(physics.tables[porous_idx].density) : 0.0
    shallow_depth = 100.0

    # ── Part 1: Muon flux on nmap angular grid (runs first) ──────────────
    zeniths_grid  = Float64[]
    azimuths_grid = Float64[]
    flux_grid     = Matrix{Float64}(undef, 0, 0)

    if args.part < 0 || args.part == 1
        zeniths_grid, azimuths_grid, flux_grid = run_part1(physics;
            n_samples = args.n_samples,
            straggling = args.straggling,
            scattering = args.scattering,
            energy_threshold_low = args.energy_threshold_low,
            energy_min = args.energy_min, energy_max = args.energy_max,
            porous_material = porous_idx, porous_density = p_density,
            porous_thickness = shallow_depth,
            output_dir = args.output_dir)
    end

    # ── Part 2: 3D topography & trajectories (uses Part 1 directions) ───
    if args.part < 0 || args.part == 2
        if isempty(flux_grid)
            println("Computing flux grid for trajectory direction selection...")
            zeniths_grid, azimuths_grid, flux_grid, _, _ =
                compute_nmap_flux_grid(physics;
                    n_samples = args.n_samples,
                    straggling = args.straggling, scattering = args.scattering,
                    energy_threshold_low = args.energy_threshold_low,
                    energy_min = args.energy_min, energy_max = args.energy_max,
                    porous_material = porous_idx, porous_density = p_density,
                    porous_thickness = shallow_depth)
        end

        directions = select_diverse_directions(flux_grid, zeniths_grid, azimuths_grid)
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
            porous_material=porous_idx, porous_density=p_density,
            porous_thickness=shallow_depth, output_dir=args.output_dir)
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
