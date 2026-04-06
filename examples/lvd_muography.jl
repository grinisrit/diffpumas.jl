#!/usr/bin/env julia
"""
lvd_muography.jl — Gran Sasso LVD muon flux muography with real topography

Combines:
  - Synthetic elevation model from geological section data
    (gran-sasso-lvd-raytracing-geometry.yaml) via Turtle.ElevationMap
  - Full muon transport physics (backward MC, straggling, scattering)
  - 3D topography visualisation with representative muon trajectories

Materials: Standard Rock (2650 kg/m³), Water (1000 kg/m³), PorousWetRock (composite).
Top 100m: porous wet rock (precomputed PorousWetRock composite from materials.xml).

This script demonstrates muography techniques:
0. 3D topography plot with Turtle elevation map and representative muon trajectories
1. Baseline studies (0–1000m depth in 100m steps, density from physics table)
   Top 100m of rock column uses PorousWetRock composite.
   1.1 Flux vs zenith angle for various rock depths (one line per depth)
   1.2 Zenith angle scattering std. deviation vs zenith for each depth
   1.3 Flux vs zenith at 1000m depth for 100 fixed energies (one curve per energy)
2. Aquifer detection using water material in a cubic rock volume (detector at 1000m)
   - 2.1a: Water fraction sweep from 0% (pure rock) to 90% (always ≥10% rock)
            at 500m depth, 100m × 100m × 100m aquifer
   - 2.1b: 90% water aquifer, enlarge from 100m to 500m at 500m depth
   - 2.2a: Moving 90% water aquifer (100m cube) vertically from detector to surface
   - 2.2b: Moving 90% water aquifer (100m cube) along y (zenith direction) at 100m depth

Usage:
    julia --project=. examples/lvd_muography.jl --output-dir PATH [OPTIONS]

Options:
    --output-dir, -o PATH     Output directory for plots (REQUIRED)
    --dump, -d PATH           Path to physics binary dump file
    --samples, -n INT         Number of MC samples per point (default: 1000)
    --threshold FLOAT         Energy threshold for mode switching (default: 100.0)
    --energy-min FLOAT        Minimum energy in GeV (default: 1e-3)
    --energy-max FLOAT        Maximum energy in GeV (default: 1e9)
    --no-straggling           Disable straggling
    --no-scattering           Disable scattering
    --part INT                Run only part 0, 1, 2 or all (default: all)

Example:
    julia --project=. examples/lvd_muography.jl --output-dir examples/lvd_muography --samples 2000
"""

using DiffPumas
using DiffPumas.Physics: get_material_index, property_range, property_kinetic_energy
using DiffPumas.Physics: property_stopping_power, ENERGY_LOSS_CSDA
using DiffPumas.Loader: print_physics_summary
using DiffPumas.Geometry: PRIMARY_ALTITUDE, compute_air_grammage, compute_flux
using DiffPumas.Geometry: compute_decay_weight_from_path, locals_air_at_altitude
using DiffPumas.Geometry: transport_backward_step, transport_backward_step_full, transport_backward_step_mixed
using DiffPumas.Geometry: compute_flux_single_with_state, TwoLayerGeometry
using DiffPumas.GaisserFlux: flux_gccly
using DiffPumas.Types: State, Vec3, MaterialMixture, is_single_material, single_material
using DiffPumas.Materials: STANDARD_ROCK, AIR, WATER, CompositeMaterial, composite_density
using DiffPumas: zenith_to_elevation, sample_energy_loguniform
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
# Part 0: Topography model & 3D visualisation
# ═══════════════════════════════════════════════════════════════════════════════

const DETECTOR_ELEVATION = 963.0
const GRID_HALF_KM       = 8.0
const GRID_STEP_KM       = 0.1
const VALLEY_FLOOR       = 600.0

function terrain_elevation(x_km::Float64, y_km::Float64)
    r = sqrt(x_km^2 + y_km^2)

    θr = deg2rad(15.0)
    xr = x_km * cos(θr) + y_km * sin(θr)
    yr = -x_km * sin(θr) + y_km * cos(θr) - 1.0
    ridge = (2100.0 - VALLEY_FLOOR) * exp(-yr^2 / 5.0) * exp(-xr^2 / 80.0)

    peaks = (
        ( 0.0,   0.5,  2450.0, 1.2),
        (-0.5,   2.8,  2912.0, 0.9),
        (-0.2,   3.5,  2655.0, 0.7),
        (-2.0,   0.5,  2400.0, 1.2),
        ( 0.3,  -0.8,  2300.0, 1.0),
        (-2.5,  -1.5,  2494.0, 1.3),
        ( 4.5,  -0.5,  2385.0, 1.4),
        ( 6.5,  -1.0,  2561.0, 1.0),
        ( 9.0,  -2.5,  2564.0, 1.4),
    )

    peak_max = 0.0
    for (px, py, pz, ps) in peaks
        d2 = (x_km - px)^2 + (y_km - py)^2
        peak_max = max(peak_max, (pz - VALLEY_FLOOR) * exp(-d2 / (2.0 * ps^2)))
    end

    ci_d2 = (x_km - 3.0)^2 + (y_km + 3.0)^2
    plateau = (1800.0 - VALLEY_FLOOR) * exp(-ci_d2 / 18.0)

    elevation = VALLEY_FLOOR + max(ridge, max(peak_max, plateau))

    if r > 5.0
        fade = clamp((9.0 - r) / 4.0, 0.0, 1.0)
        elevation = elevation * fade + VALLEY_FLOOR * (1.0 - fade)
    end

    return max(elevation, VALLEY_FLOOR)
end

function build_elevation_map()
    h = GRID_HALF_KM
    s = GRID_STEP_KM
    nx = round(Int, 2h / s) + 1
    ny = nx

    info = MapInfo(nx, ny, (-h, h), (-h, h), (VALLEY_FLOOR, 3000.0))
    emap = map_create(info)

    for iy in 0:(ny - 1)
        y_km = -h + iy * s
        for ix in 0:(nx - 1)
            x_km = -h + ix * s
            map_fill(emap, ix, iy, terrain_elevation(x_km, y_km))
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
                   "<sub>Synthetic elevation from geological sections · $title_sub</sub>",
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
    simulate_muon_trajectories(physics, rock_idx, air_idx, emap;
                               n_muons, rock_thickness, ...)

Run a handful of backward-MC muon trajectories through a flat TwoLayerGeometry
whose thickness matches the local rock overburden, recording the 3D path in
the map coordinate system (east_km, north_km, elevation_m) for plotting.

Returns a vector of paths, each a `Vector{NTuple{3,Float64}}`.
"""
function simulate_muon_trajectories(physics, rock_idx::Int, air_idx::Int,
                                     emap::ElevationMap;
                                     n_muons::Int = 6,
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

    ray_specs = [
        (  0.0, 10.0),
        ( 45.0, 40.0),
        (135.0, 45.0),
        (225.0, 35.0),
        (315.0, 50.0),
        ( 90.0, 55.0),
        (180.0, 25.0),
        (337.5, 30.0),
    ]

    n_use = min(n_muons, length(ray_specs))

    for mi in 1:n_use
        az_deg, zen_deg = ray_specs[mi]
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

function run_part0(physics, rock_idx::Int, air_idx::Int, emap::ElevationMap;
                   straggling::Bool, scattering::Bool,
                   energy_threshold_low::Float64,
                   porous_material::Int, porous_density::Float64,
                   porous_thickness::Float64,
                   output_dir::String)

    println("=" ^ 60)
    println(" Part 0: 3D Topography & Muon Trajectories")
    println("=" ^ 60)
    println()

    stepper = stepper_create()
    stepper_add_flat(stepper, 0.0)
    stepper_add_layer(stepper)
    stepper_add_map(stepper, emap)

    ray_configs = [
        (  0.0,  5.0), (  0.0, 35.0), ( 90.0, 35.0),
        (180.0, 35.0), (270.0, 35.0), ( 45.0, 50.0),
        (135.0, 50.0), (225.0, 50.0), (315.0, 50.0),
        (337.5, 25.0), (157.5, 60.0), ( 90.0, 65.0),
    ]

    ray_paths = Tuple{Float64,Float64,Vector{NTuple{3,Float64}}}[]
    for (az, zen) in ray_configs
        path = trace_ray(emap, az, zen)
        length(path) > 2 && push!(ray_paths, (az, zen, path))
    end

    println("  Traced $(length(ray_paths)) geometric rays")

    println("  Simulating backward-MC muon trajectories...")
    muon_paths = simulate_muon_trajectories(physics, rock_idx, air_idx, emap;
        n_muons=6, straggling=straggling, scattering=scattering,
        energy_threshold_low=energy_threshold_low,
        porous_material=porous_material,
        porous_density=porous_density,
        porous_thickness=porous_thickness)
    println("  Got $(length(muon_paths)) muon trajectories")
    println()

    create_topo_plot(emap, ray_paths;
        muon_paths=muon_paths,
        output_path=joinpath(output_dir, "part0_topography.html"))
    println()
end

# =============================================================================
# Part 1: Baseline flux vs angle for various depths
# =============================================================================

function compute_flux_vs_angle(physics,
                                rock_idx::Int,
                                depths::Vector{Float64},
                                zeniths::Vector{Float64};
                                n_samples::Int = 1000,
                                straggling::Bool = true,
                                scattering::Bool = true,
                                energy_threshold_low::Float64 = 100.0,
                                energy_min::Float64 = 1e-3,
                                energy_max::Float64 = 1e9,
                                porous_material::Int = -1,
                                porous_density::Float64 = 0.0,
                                porous_thickness::Float64 = 0.0,
                                verbose::Bool = true)

    rock_density = Float64(physics.tables[rock_idx].density)
    n_depths = length(depths)
    n_zeniths = length(zeniths)

    flux_grid = zeros(n_depths, n_zeniths)
    sigma_grid = zeros(n_depths, n_zeniths)

    n_total = n_depths * n_zeniths
    n_done = 0

    for (d_idx, depth) in enumerate(depths)
        for (z_idx, zenith) in enumerate(zeniths)
            n_done += 1
            elevation = zenith_to_elevation(zenith)

            if verbose && (n_done % 10 == 0 || n_done == 1)
                @info "[$n_done/$n_total] depth=$(depth)m, θ=$(zenith)°"
            end

            flux, sigma = compute_flux(physics, rock_density, depth, elevation,
                                       energy_min, energy_max;
                                       n_samples=n_samples,
                                       straggling=straggling,
                                       scattering=scattering,
                                       energy_threshold_low=energy_threshold_low,
                                       porous_material=porous_material,
                                       porous_density=porous_density,
                                       porous_thickness=porous_thickness)

            flux_grid[d_idx, z_idx] = flux
            sigma_grid[d_idx, z_idx] = sigma
        end
    end

    return flux_grid, sigma_grid
end

function create_flux_vs_angle_plot(depths::Vector{Float64},
                                    zeniths::Vector{Float64},
                                    flux_grid::Matrix{Float64},
                                    sigma_grid::Matrix{Float64};
                                    output_path::String,
                                    title::String = "Muon Flux vs Zenith Angle")

    traces = GenericTrace[]

    n_depths = length(depths)
    colors = ["hsl($(round(Int, 240 - 240 * i / n_depths)), 70%, 50%)" for i in 1:n_depths]

    for (d_idx, depth) in enumerate(depths)
        flux_values = flux_grid[d_idx, :]
        sigma_values = sigma_grid[d_idx, :]

        trace = scatter(
            x = zeniths, y = flux_values,
            mode = "lines+markers",
            name = "$(round(depth; digits=0))m",
            line = attr(color = colors[d_idx], width = 2),
            marker = attr(size = 6),
            error_y = attr(type = "data", array = sigma_values,
                           visible = true, color = colors[d_idx])
        )
        push!(traces, trace)
    end

    layout = Layout(
        title = title,
        xaxis = attr(title = "Zenith Angle θ (°)",
                     range = [minimum(zeniths) - 1, maximum(zeniths) + 1]),
        yaxis = attr(title = "Flux (m⁻² s⁻¹ sr⁻¹)", type = "log"),
        legend = attr(title = attr(text = "Rock Depth"), x = 1.02, y = 1.0),
        width = 1000, height = 700, hovermode = "closest"
    )

    fig = Plot(traces, layout)
    mkpath(dirname(output_path))
    savefig(fig, output_path)
    println("Saved plot: $output_path")
    return fig
end

# --- Part 1.2: Zenith angle scattering standard deviation ---

function compute_zenith_std(physics,
                                  rock_idx::Int,
                                  air_idx::Int,
                                  depths::Vector{Float64},
                                  zeniths::Vector{Float64};
                                  n_samples::Int = 1000,
                                  straggling::Bool = true,
                                  scattering::Bool = true,
                                  energy_threshold_low::Float64 = 100.0,
                                  energy_min::Float64 = 1e-3,
                                  energy_max::Float64 = 1e9,
                                  porous_material::Int = -1,
                                  porous_density::Float64 = 0.0,
                                  porous_thickness::Float64 = 0.0,
                                  seed::Int = 42,
                                  verbose::Bool = true)

    rock_density = Float64(physics.tables[rock_idx].density)
    n_depths  = length(depths)
    n_zeniths = length(zeniths)

    std_grid = zeros(n_depths, n_zeniths)
    rk = log(energy_max / energy_min)

    n_total = n_depths * n_zeniths
    n_done  = 0

    for (d_idx, depth) in enumerate(depths)
        for (z_idx, zenith) in enumerate(zeniths)
            n_done += 1
            if verbose && (n_done % 10 == 0 || n_done == 1)
                @info "[zenith-var $n_done/$n_total] depth=$(round(depth;digits=0))m, θ=$(zenith)°"
            end

            elevation = zenith_to_elevation(zenith)
            geometry  = TwoLayerGeometry{Float64}(depth, rock_density, rock_idx, air_idx,
                                                  porous_material, porous_density, porous_thickness)
            rng = Random.MersenneTwister(seed)

            delta_sum  = 0.0
            delta2_sum = 0.0
            n_valid    = 0

            for _ in 1:n_samples
                kf = rk > 0.0 ? energy_min * exp(rk * rand(rng)) : energy_min
                charge = rand(rng) > 0.5 ? 1.0 : -1.0

                _, zenith_final = compute_flux_single_with_state(
                    physics, geometry, kf, elevation, charge;
                    rng=rng, straggling=straggling, scattering=scattering,
                    energy_threshold_low=energy_threshold_low)

                if !isnan(zenith_final)
                    δ = zenith_final - zenith
                    delta_sum  += δ
                    delta2_sum += δ * δ
                    n_valid    += 1
                end
            end

            if n_valid > 1
                mean_δ = delta_sum / n_valid
                std_grid[d_idx, z_idx] = sqrt(max(delta2_sum / n_valid - mean_δ^2, 0.0))
            end
        end
    end

    return std_grid
end

function create_zenith_std_plot(depths::Vector{Float64},
                                 zeniths::Vector{Float64},
                                 std_grid::Matrix{Float64};
                                 output_path::String,
                                 title::String = "Part 1.2: Zenith Angle Scattering Std. Dev.")

    traces = GenericTrace[]
    n_depths = length(depths)
    colors = ["hsl($(round(Int, 240 - 240 * i / n_depths)), 70%, 50%)" for i in 1:n_depths]

    for (d_idx, depth) in enumerate(depths)
        trace = scatter(
            x = zeniths, y = std_grid[d_idx, :],
            mode = "lines+markers",
            name = "$(round(depth; digits=0))m",
            line = attr(color = colors[d_idx], width = 2),
            marker = attr(size = 5)
        )
        push!(traces, trace)
    end

    layout = Layout(
        title = title,
        xaxis = attr(title = "Zenith Angle θ (°)",
                     range = [minimum(zeniths) - 1, maximum(zeniths) + 1]),
        yaxis = attr(title = "σ(θ_final − θ_detector)  (deg)", type = "log"),
        legend = attr(title = attr(text = "Rock Depth"), x = 1.02, y = 1.0),
        width = 1000, height = 700, hovermode = "closest"
    )

    fig = Plot(traces, layout)
    mkpath(dirname(output_path))
    savefig(fig, output_path)
    println("Saved plot: $output_path")
    return fig
end

# --- Part 1.3: Flux vs zenith for fixed energies ---

function compute_flux_fixed_energy(physics,
                                    rock_idx::Int,
                                    air_idx::Int,
                                    depth::Float64,
                                    zeniths::Vector{Float64};
                                    n_samples::Int = 100,
                                    n_energies::Int = 100,
                                    straggling::Bool = true,
                                    scattering::Bool = true,
                                    energy_threshold_low::Float64 = 100.0,
                                    energy_min::Float64 = 1e-3,
                                    energy_max::Float64 = 1e9,
                                    porous_material::Int = -1,
                                    porous_density::Float64 = 0.0,
                                    porous_thickness::Float64 = 0.0,
                                    seed::Int = 42,
                                    verbose::Bool = true)

    rock_density = Float64(physics.tables[rock_idx].density)
    energies = exp.(range(log(energy_min), log(energy_max); length=n_energies))
    geometry = TwoLayerGeometry{Float64}(depth, rock_density, rock_idx, air_idx,
                                         porous_material, porous_density, porous_thickness)

    n_z = length(zeniths)
    n_e = length(energies)

    flux_grid  = zeros(n_z, n_e)
    sigma_grid = zeros(n_z, n_e)

    total = n_z * n_e
    done  = 0

    for (z_idx, zenith) in enumerate(zeniths)
        elevation = zenith_to_elevation(zenith)
        for (e_idx, kf) in enumerate(energies)
            done += 1
            if verbose && (done % 500 == 0 || done == 1)
                @info "[fixed-E $done/$total] θ=$(zenith)°, E=$(round(kf;sigdigits=3)) GeV"
            end

            rng = Random.MersenneTwister(seed + e_idx)
            w_sum  = 0.0
            w2_sum = 0.0

            for _ in 1:n_samples
                charge = rand(rng) > 0.5 ? 1.0 : -1.0
                wf = 2.0

                fw, _ = compute_flux_single_with_state(
                    physics, geometry, kf, elevation, charge;
                    rng=rng, straggling=straggling, scattering=scattering,
                    energy_threshold_low=energy_threshold_low)
                wi = wf * fw
                w_sum  += wi
                w2_sum += wi * wi
            end

            flux_avg = w_sum / n_samples
            variance = (w2_sum / n_samples) - flux_avg^2
            sigma = sqrt(max(variance, 0.0) / n_samples)

            flux_grid[z_idx, e_idx]  = flux_avg
            sigma_grid[z_idx, e_idx] = sigma
        end
    end

    return energies, flux_grid, sigma_grid
end

function create_flux_fixed_energy_plot(energies::Vector{Float64},
                                        zeniths::Vector{Float64},
                                        flux_grid::Matrix{Float64};
                                        depth::Float64,
                                        output_path::String,
                                        n_curves::Int = 12)

    traces = GenericTrace[]
    n_e = length(energies)
    step = max(1, n_e ÷ n_curves)
    sel = 1:step:n_e
    n_sel = length(sel)
    colors = ["hsl($(round(Int, 0 + 270 * i / n_sel)), 75%, 45%)" for i in 1:n_sel]

    for (ci, e_idx) in enumerate(sel)
        e_val = energies[e_idx]
        label = e_val >= 1.0 ? @sprintf("%.1f GeV", e_val) : @sprintf("%.1e GeV", e_val)
        trace = scatter(
            x = zeniths, y = flux_grid[:, e_idx],
            mode = "lines+markers",
            name = label,
            line = attr(color = colors[ci], width = 2),
            marker = attr(size = 4)
        )
        push!(traces, trace)
    end

    layout = Layout(
        title = "Part 1.3: Flux vs Zenith at $(round(Int, depth))m depth (fixed energies)",
        xaxis = attr(title = "Zenith Angle θ (°)",
                     range = [minimum(zeniths) - 1, maximum(zeniths) + 1]),
        yaxis = attr(title = "Flux (m⁻² s⁻¹ sr⁻¹)", type = "log"),
        legend = attr(title = attr(text = "Energy"), x = 1.02, y = 1.0),
        width = 1000, height = 700, hovermode = "closest"
    )

    fig = Plot(traces, layout)
    mkpath(dirname(output_path))
    savefig(fig, output_path)
    println("Saved plot: $output_path")
    return fig
end

# =============================================================================
# Part 2: Cube geometry with density anomalies
# =============================================================================

struct MuographyCube{T<:Real}
    x_min::T; x_max::T
    y_min::T; y_max::T
    z_min::T; z_max::T
    nx::Int; ny::Int; nz::Int
    cell_size::T
    rock_material::Int
    air_material::Int
end

function MuographyCube(rock_thickness::T, width::T, cell_size::T,
                        rock_material::Int, air_material::Int) where T<:Real
    return MuographyCube(rock_thickness, width, width, cell_size, rock_material, air_material)
end

function MuographyCube(rock_thickness::T, width_x::T, width_y::T, cell_size::T,
                        rock_material::Int, air_material::Int) where T<:Real
    x_min = -width_x / 2; x_max = width_x / 2
    y_min = -width_y / 2; y_max = width_y / 2
    z_min = T(0);          z_max = rock_thickness

    nx = Int(ceil(width_x / cell_size))
    ny = Int(ceil(width_y / cell_size))
    nz = Int(ceil(rock_thickness / cell_size))

    return MuographyCube{T}(x_min, x_max, y_min, y_max, z_min, z_max,
                            nx, ny, nz, cell_size, rock_material, air_material)
end

function get_cell_indices(cube::MuographyCube{T}, x::T, y::T, z::T) where T<:Real
    (x < cube.x_min || x > cube.x_max ||
     y < cube.y_min || y > cube.y_max ||
     z < cube.z_min || z > cube.z_max) && return nothing
    i = clamp(floor(Int, (x - cube.x_min) / cube.cell_size) + 1, 1, cube.nx)
    j = clamp(floor(Int, (y - cube.y_min) / cube.cell_size) + 1, 1, cube.ny)
    k = clamp(floor(Int, (z - cube.z_min) / cube.cell_size) + 1, 1, cube.nz)
    return (i, j, k)
end

@inline cell_linear_index(cube::MuographyCube, i::Int, j::Int, k::Int) =
    i + (j - 1) * cube.nx + (k - 1) * cube.nx * cube.ny

@inline num_cells(cube::MuographyCube) = cube.nx * cube.ny * cube.nz

function get_cell_center(cube::MuographyCube{T}, i::Int, j::Int, k::Int) where T<:Real
    x = cube.x_min + (i - 0.5) * cube.cell_size
    y = cube.y_min + (j - 0.5) * cube.cell_size
    z = cube.z_min + (k - 0.5) * cube.cell_size
    return (x, y, z)
end

function compute_step_to_cell_boundary(cube::MuographyCube{T}, x::T, y::T, z::T,
                                        ux::T, uy::T, uz::T,
                                        i::Int, j::Int, k::Int) where T<:Real
    eps_val = T(1e-10)
    step_min = T(Inf)
    x_lo = cube.x_min + (i - 1) * cube.cell_size; x_hi = cube.x_min + i * cube.cell_size
    y_lo = cube.y_min + (j - 1) * cube.cell_size; y_hi = cube.y_min + j * cube.cell_size
    z_lo = cube.z_min + (k - 1) * cube.cell_size; z_hi = cube.z_min + k * cube.cell_size

    if abs(ux) > eps_val
        t = ux > 0 ? (x_hi - x) / ux : (x_lo - x) / ux
        t > eps_val && (step_min = min(step_min, t))
    end
    if abs(uy) > eps_val
        t = uy > 0 ? (y_hi - y) / uy : (y_lo - y) / uy
        t > eps_val && (step_min = min(step_min, t))
    end
    if abs(uz) > eps_val
        t = uz > 0 ? (z_hi - z) / uz : (z_lo - z) / uz
        t > eps_val && (step_min = min(step_min, t))
    end
    return step_min == T(Inf) ? T(1000.0) : step_min
end

function compute_flux_cube_single(physics, cube::MuographyCube{T},
                                   cell_densities::Vector{T},
                                   elevation::T, energy_final::T, charge::T;
                                   rng::AbstractRNG,
                                   straggling::Bool=true,
                                   scattering::Bool=true,
                                   energy_threshold_low::T=T(100.0),
                                   cell_materials::Union{Vector{MaterialMixture}, Nothing}=nothing) where T<:Real

    theta = (T(90) - elevation) * T(π) / T(180)
    cos_theta = cos(theta); sin_theta = sin(theta)
    azimuth = T(2π) * rand(rng)
    dir_x = sin_theta * sin(azimuth)
    dir_y = sin_theta * cos(azimuth)
    dir_z = cos_theta

    state = State{T}(charge=charge, energy=energy_final, distance=zero(T),
        grammage=zero(T), time=zero(T), weight=one(T),
        position=Vec3{T}(zero(T), zero(T), zero(T)),
        direction=Vec3{T}(-dir_x, -dir_y, -dir_z), decayed=false)

    energy_threshold = T(1e12); STEP_EPSILON = T(1e-7); STEP_MIN = T(1e-6)
    max_outer_steps = 1000; outer_step = 0

    while outer_step < max_outer_steps && state.energy < energy_threshold - eps(T)
        outer_step += 1
        x, y, z = state.position[1], state.position[2], state.position[3]
        (z < zero(T) || z >= PRIMARY_ALTITUDE - eps(T)) && break
        (state.weight <= zero(T) || !isfinite(state.weight)) && break

        if state.energy < energy_threshold_low - eps(T)
            use_straggled = straggling; use_scattering = scattering
            current_energy_limit = energy_threshold_low
        else
            use_straggled = false; use_scattering = false
            current_energy_limit = energy_threshold
        end

        inner_steps = 0
        while inner_steps < 50000
            inner_steps += 1
            x, y, z = state.position[1], state.position[2], state.position[3]
            (z < zero(T) || z >= PRIMARY_ALTITUDE - eps(T)) && break
            (state.energy >= current_energy_limit - eps(T)) && break
            (state.weight <= zero(T) || !isfinite(state.weight)) && break

            ux_motion = -state.direction[1]
            uy_motion = -state.direction[2]
            uz_motion = -state.direction[3]

            mix::MaterialMixture = MaterialMixture(cube.rock_material)
            density = cell_densities[1]
            step_to_boundary = T(1000.0)

            if z < cube.z_max
                cell_idx = get_cell_indices(cube, x, y, z)
                if cell_idx === nothing
                    if uz_motion > eps(T)
                        step_to_boundary = min(step_to_boundary, (cube.z_max - z) / uz_motion)
                    elseif uz_motion < -eps(T)
                        step_to_boundary = min(step_to_boundary, -z / uz_motion)
                    end
                else
                    i, j, k = cell_idx
                    linear_idx = cell_linear_index(cube, i, j, k)
                    mix = cell_materials !== nothing ? cell_materials[linear_idx] : MaterialMixture(cube.rock_material)
                    density = cell_densities[linear_idx]
                    step_to_boundary = compute_step_to_cell_boundary(cube, x, y, z,
                        ux_motion, uy_motion, uz_motion, i, j, k)
                end
            else
                mix = MaterialMixture(cube.air_material)
                density = locals_air_at_altitude(z).density
                if uz_motion > eps(T)
                    step_to_boundary = (PRIMARY_ALTITUDE - z) / uz_motion
                elseif uz_motion < -eps(T)
                    step_to_boundary = (cube.z_max - z) / uz_motion
                end
            end

            Xi = property_range(physics, ENERGY_LOSS_CSDA, mix, state.energy)
            max_geometric_step = T(0.01) * Xi / max(density, T(1e-6))
            if z >= cube.z_max
                uz_abs = max(abs(uz_motion), T(0.05))
                max_geometric_step = min(max_geometric_step, T(0.05) * T(12e3) / uz_abs)
            end
            step_size = clamp(min(step_to_boundary + STEP_EPSILON, max_geometric_step), STEP_MIN, T(1000.0))

            if state.energy < energy_threshold_low - eps(T)
                if use_straggled && use_scattering
                    state, _ = transport_backward_step_full(physics, state, mix, density,
                        step_size, rng; mode=:straggled, scattering=true,
                        energy_limit=current_energy_limit)
                elseif use_straggled
                    state = transport_backward_step(physics, state, mix, density,
                        step_size; rng=rng, straggling=true)
                else
                    state = transport_backward_step(physics, state, mix, density,
                        step_size; rng=nothing, straggling=false)
                end
            else
                if straggling
                    state = transport_backward_step_mixed(physics, state, mix, density,
                        step_size, rng; energy_limit=current_energy_limit)
                else
                    state = transport_backward_step(physics, state, mix, density,
                        step_size; rng=nothing, straggling=false)
                end
            end
        end
    end

    if state.position[3] >= PRIMARY_ALTITUDE - T(1.0)
        cos_theta_final = clamp(-state.direction[3], zero(T), one(T))
        cos_theta_final <= zero(T) && return zero(T)
        flux = flux_gccly(cos_theta_final, state.energy, charge)
        return state.weight * flux
    else
        return zero(T)
    end
end

function compute_cube_flux_vs_angle(physics, cube::MuographyCube{T},
                                     cell_densities::Vector{T},
                                     zeniths::Vector{Float64};
                                     n_samples::Int = 1000,
                                     straggling::Bool = true,
                                     scattering::Bool = true,
                                     energy_threshold_low::Float64 = 100.0,
                                     energy_min::Float64 = 1e-3,
                                     energy_max::Float64 = 1e9,
                                     verbose::Bool = true,
                                     cell_materials::Union{Vector{MaterialMixture}, Nothing} = nothing) where T<:Real

    rng = MersenneTwister(42)
    n_zeniths = length(zeniths)
    flux_values = zeros(n_zeniths)
    sigma_values = zeros(n_zeniths)

    for (z_idx, zenith) in enumerate(zeniths)
        elevation = T(90) - zenith
        verbose && @info "Computing zenith=$(zenith)°..."

        w_sum = zero(T); w2_sum = zero(T)
        for _ in 1:n_samples
            kf, w_energy = sample_energy_loguniform(Float64(energy_min), Float64(energy_max), rng)
            charge = rand(rng) > 0.5 ? T(1) : T(-1)
            flux_single = compute_flux_cube_single(physics, cube, cell_densities,
                elevation, T(kf), charge; rng=rng, straggling=straggling, scattering=scattering,
                energy_threshold_low=T(energy_threshold_low), cell_materials=cell_materials)
            wi = T(2) * T(w_energy) * flux_single
            w_sum += wi; w2_sum += wi * wi
        end

        n = T(n_samples)
        flux_values[z_idx] = w_sum / n
        variance = (w2_sum / n - (w_sum / n)^2) / max(one(T), n - one(T))
        sigma_values[z_idx] = sqrt(max(zero(T), variance))
    end

    return flux_values, sigma_values
end

function create_cell_config(cube::MuographyCube{T},
                            rock_idx::Int, water_idx::Int,
                            porous_wet_rock_idx::Int,
                            porous_density::T,
                            aquifer_center::Tuple{T,T,T},
                            aquifer_half_size::Tuple{T,T,T},
                            water_fraction::T;
                            shallow_depth::T = T(100),
                            rock_density::T = T(2650),
                            water_density::T = T(1000)) where T<:Real

    n = num_cells(cube)
    densities = fill(rock_density, n)
    materials = fill(MaterialMixture(rock_idx), n)

    ax, ay, az = aquifer_center
    hx, hy, hz = aquifer_half_size
    shallow_z_threshold = cube.z_max - shallow_depth

    for k in 1:cube.nz, j in 1:cube.ny, i in 1:cube.nx
        cx, cy, cz = get_cell_center(cube, i, j, k)
        idx = cell_linear_index(cube, i, j, k)

        in_aquifer = abs(cx - ax) <= hx && abs(cy - ay) <= hy && abs(cz - az) <= hz
        in_shallow = shallow_depth > zero(T) && cz >= shallow_z_threshold

        if in_shallow
            materials[idx] = MaterialMixture(porous_wet_rock_idx)
            densities[idx] = porous_density

            if in_aquifer && water_fraction > zero(T)
                wf_clamped = min(water_fraction, T(0.9))
                porous_frac = (one(T) - wf_clamped) * T(0.5)
                rock_frac   = (one(T) - wf_clamped) * T(0.5)
                materials[idx] = MaterialMixture(
                    [porous_wet_rock_idx, rock_idx, water_idx],
                    [porous_frac, rock_frac, wf_clamped])
                densities[idx] = porous_frac * porous_density +
                                 rock_frac * rock_density +
                                 wf_clamped * water_density
            end
        elseif in_aquifer && water_fraction > zero(T)
            wf_clamped = min(water_fraction, T(0.9))
            rock_frac = one(T) - wf_clamped
            materials[idx] = MaterialMixture(
                [rock_idx, water_idx], [rock_frac, wf_clamped])
            densities[idx] = rock_frac * rock_density + wf_clamped * water_density
        end
    end

    return densities, materials
end

function create_aquifer_plot(zeniths::Vector{Float64},
                              scenarios::Vector{Tuple{String, Vector{Float64}, Vector{Float64}}};
                              output_path::String, title::String)
    traces = GenericTrace[]
    n_scenarios = length(scenarios)
    colors = ["hsl($(round(Int, 360 * i / n_scenarios)), 70%, 50%)" for i in 0:n_scenarios-1]

    for (idx, (name, flux, sigma)) in enumerate(scenarios)
        trace = scatter(x = zeniths, y = flux, mode = "lines+markers",
            name = name, line = attr(color = colors[idx], width = 2),
            marker = attr(size = 6),
            error_y = attr(type = "data", array = sigma, visible = true, color = colors[idx]))
        push!(traces, trace)
    end

    layout = Layout(title = title,
        xaxis = attr(title = "Zenith Angle θ (°)",
                     range = [minimum(zeniths) - 1, maximum(zeniths) + 1]),
        yaxis = attr(title = "Flux (m⁻² s⁻¹ sr⁻¹)", type = "log"),
        legend = attr(x = 1.02, y = 1.0),
        width = 1100, height = 700, hovermode = "closest")

    fig = Plot(traces, layout)
    mkpath(dirname(output_path))
    savefig(fig, output_path)
    println("Saved plot: $output_path")
    return fig
end

# =============================================================================
# Part 2.1 & 2.2 runners (unchanged logic from muography.jl)
# =============================================================================

function run_part2_1(physics, rock_idx, air_idx, water_idx, porous_idx, porous_density;
                      n_samples, straggling, scattering, energy_threshold_low,
                      energy_min, energy_max, output_dir)
    println()
    println("=" ^ 60)
    println(" Part 2.1: Water-Fraction Sweep and Hole Enlargement")
    println("=" ^ 60)
    println()

    rock_thickness = 1000.0; cube_width_x = 100.0; cube_width_y = 500.0; cell_size = 10.0
    cube = MuographyCube(rock_thickness, cube_width_x, cube_width_y, cell_size, rock_idx, air_idx)
    println("  Grid: $(cube.nx) × $(cube.ny) × $(cube.nz)")

    zeniths = collect(0.0:2.0:60.0)
    aquifer_depth = 500.0
    aquifer_hs = (50.0, 50.0, 50.0)
    aquifer_center = (0.0, 0.0, aquifer_depth - aquifer_hs[3])

    println("Part 2.1a: Water-fraction sweep at 500m depth")
    println("-" ^ 50)
    scenarios_water = Tuple{String, Vector{Float64}, Vector{Float64}}[]
    for water_pct in 0:10:90
        wf = Float64(water_pct) / 100.0
        densities, cell_mats = create_cell_config(cube, rock_idx, water_idx,
            porous_idx, porous_density, aquifer_center, aquifer_hs, wf)
        label = water_pct == 0 ? "Baseline (100% rock)" : "$(water_pct)% water"
        @info "Computing $label..."
        flux, sigma = compute_cube_flux_vs_angle(physics, cube, densities, zeniths;
            n_samples=n_samples, straggling=straggling, scattering=scattering,
            energy_threshold_low=energy_threshold_low, energy_min=energy_min,
            energy_max=energy_max, cell_materials=cell_mats)
        push!(scenarios_water, (label, flux, sigma))
    end
    create_aquifer_plot(zeniths, scenarios_water;
        output_path=joinpath(output_dir, "part2_1a_water_fraction.html"),
        title="Part 2.1a: Flux vs Angle - Water Fraction at 500m<br>" *
              "<sub>100m cube, 0-90% water, porous wet rock above 900m</sub>")

    println("\nPart 2.1b: Hole enlargement with 90% water aquifer")
    println("-" ^ 50)
    scenarios_hole = Tuple{String, Vector{Float64}, Vector{Float64}}[]
    densities_bl, mats_bl = create_cell_config(cube, rock_idx, water_idx,
        porous_idx, porous_density, aquifer_center, aquifer_hs, 0.0)
    flux_bl, sigma_bl = compute_cube_flux_vs_angle(physics, cube, densities_bl, zeniths;
        n_samples=n_samples, straggling=straggling, scattering=scattering,
        energy_threshold_low=energy_threshold_low, energy_min=energy_min,
        energy_max=energy_max, cell_materials=mats_bl)
    push!(scenarios_hole, ("Baseline", flux_bl, sigma_bl))

    for half_size in 50.0:50.0:250.0
        sz = 2 * half_size
        center_z = aquifer_depth + half_size
        hs = (50.0, half_size, half_size)
        center = (0.0, 0.0, center_z)
        densities, cell_mats = create_cell_config(cube, rock_idx, water_idx,
            porous_idx, porous_density, center, hs, 0.9)
        @info "Computing hole=$(Int(sz))m (90% water)..."
        flux, sigma = compute_cube_flux_vs_angle(physics, cube, densities, zeniths;
            n_samples=n_samples, straggling=straggling, scattering=scattering,
            energy_threshold_low=energy_threshold_low, energy_min=energy_min,
            energy_max=energy_max, cell_materials=cell_mats)
        push!(scenarios_hole, ("$(Int(sz))m aquifer", flux, sigma))
    end
    create_aquifer_plot(zeniths, scenarios_hole;
        output_path=joinpath(output_dir, "part2_1b_hole_enlargement.html"),
        title="Part 2.1b: Flux vs Angle - 90% Water Aquifer Enlargement<br>" *
              "<sub>Bottom at 500m, enlarging upward; porous wet rock above 900m</sub>")
    println("\nPart 2.1 complete!")
end

function run_part2_2(physics, rock_idx, air_idx, water_idx, porous_idx, porous_density;
                      n_samples, straggling, scattering, energy_threshold_low,
                      energy_min, energy_max, output_dir)
    println()
    println("=" ^ 60)
    println(" Part 2.2: Moving 90% Water Aquifer")
    println("=" ^ 60)
    println()

    rock_thickness = 1000.0; cube_width_x = 100.0; cube_width_y = 1100.0; cell_size = 10.0
    cube = MuographyCube(rock_thickness, cube_width_x, cube_width_y, cell_size, rock_idx, air_idx)
    println("  Grid: $(cube.nx) × $(cube.ny) × $(cube.nz)")

    zeniths = collect(0.0:2.0:60.0)
    aquifer_hs = (50.0, 50.0, 50.0)

    densities_bl, mats_bl = create_cell_config(cube, rock_idx, water_idx,
        porous_idx, porous_density, (0.0, 0.0, -1000.0), aquifer_hs, 0.0)
    flux_baseline, sigma_baseline = compute_cube_flux_vs_angle(
        physics, cube, densities_bl, zeniths;
        n_samples=n_samples, straggling=straggling, scattering=scattering,
        energy_threshold_low=energy_threshold_low, energy_min=energy_min,
        energy_max=energy_max, cell_materials=mats_bl)

    println("Part 2.2a: Moving 90% water aquifer vertically (on-axis)")
    println("-" ^ 50)
    scenarios_vert = Tuple{String, Vector{Float64}, Vector{Float64}}[]
    push!(scenarios_vert, ("Baseline", flux_baseline, sigma_baseline))

    for depth in 100.0:100.0:1000.0
        center_z = depth - aquifer_hs[3]
        center = (0.0, 0.0, center_z)
        densities, cell_mats = create_cell_config(cube, rock_idx, water_idx,
            porous_idx, porous_density, center, aquifer_hs, 0.9)
        @info "Computing depth=$(Int(depth))m ..."
        flux, sigma = compute_cube_flux_vs_angle(physics, cube, densities, zeniths;
            n_samples=n_samples, straggling=straggling, scattering=scattering,
            energy_threshold_low=energy_threshold_low, energy_min=energy_min,
            energy_max=energy_max, cell_materials=cell_mats)
        push!(scenarios_vert, ("depth=$(Int(depth))m", flux, sigma))
    end
    create_aquifer_plot(zeniths, scenarios_vert;
        output_path=joinpath(output_dir, "part2_2a_vertical_movement.html"),
        title="Part 2.2a: Flux vs Angle - 90% Water Aquifer Moving Vertically<br>" *
              "<sub>100m cube, on-axis; porous wet rock above 900m</sub>")

    println("\nPart 2.2b: Moving 90% water aquifer along y at 100m depth")
    println("-" ^ 50)
    scenarios_horiz = Tuple{String, Vector{Float64}, Vector{Float64}}[]
    push!(scenarios_horiz, ("Baseline", flux_baseline, sigma_baseline))
    aquifer_z_center = 100.0

    for y_center in 0.0:50.0:200.0
        center = (0.0, y_center, aquifer_z_center)
        theta_expected = atand(y_center / aquifer_z_center)
        densities, cell_mats = create_cell_config(cube, rock_idx, water_idx,
            porous_idx, porous_density, center, aquifer_hs, 0.9)
        @info "Computing y_center=$(Int(y_center))m (θ≈$(@sprintf("%.0f",theta_expected))°) ..."
        flux, sigma = compute_cube_flux_vs_angle(physics, cube, densities, zeniths;
            n_samples=n_samples, straggling=straggling, scattering=scattering,
            energy_threshold_low=energy_threshold_low, energy_min=energy_min,
            energy_max=energy_max, cell_materials=cell_mats)
        push!(scenarios_horiz, ("y=$(Int(y_center))m (θ≈$(@sprintf("%.0f",theta_expected))°)", flux, sigma))
    end
    create_aquifer_plot(zeniths, scenarios_horiz;
        output_path=joinpath(output_dir, "part2_2b_horizontal_movement.html"),
        title="Part 2.2b: Flux vs Angle - 90% Water Aquifer Moving Along y<br>" *
              "<sub>100m cube at z=[50,150]m; porous wet rock above 900m</sub>")
    println("\nPart 2.2 complete!")
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
    part = -1  # -1 = all

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
    println(" DiffPumas — Gran Sasso LVD Muography")
    println("=" ^ 60)
    println()
    println("Configuration:")
    println("  Dump file:       $(args.dump_path)")
    println("  MC samples:      $(args.n_samples)")
    println("  Threshold:       $(args.energy_threshold_low) GeV")
    println("  Energy range:    $(args.energy_min) - $(args.energy_max) GeV")
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
    water_idx  = get_material_index(physics, "Water")
    porous_idx = get_material_index(physics, "PorousWetRock")
    wet_rock_idx = get_material_index(physics, "WetRock")

    if rock_idx == -1 || air_idx == -1
        println("ERROR: Required materials (StandardRock, Air) not found"); return 1
    end
    water_idx == -1 && println("WARNING: Water material not found; Part 2 will be skipped")

    println("Material indices: rock=$rock_idx, air=$air_idx, water=$water_idx")
    wet_rock_idx != -1 && println("  WetRock (composite): $wet_rock_idx")
    if porous_idx != -1
        println("  PorousWetRock (composite): $porous_idx " *
                "(density=$(round(physics.tables[porous_idx].density; digits=1)) kg/m³)")
    end
    println()

    p_density = porous_idx != -1 ? Float64(physics.tables[porous_idx].density) : 0.0
    shallow_depth = 100.0

    # ── Part 0: 3D topography ───────────────────────────────────────────────
    if args.part < 0 || args.part == 0
        println("Building elevation map from geological section anchors...")
        emap = build_elevation_map()
        info, _ = map_meta(emap)
        println("  Grid: $(info.nx) × $(info.ny)")
        elev_above, _ = map_elevation(emap, 0.0, 0.0)
        println("  Surface above detector: $(round(elev_above; digits=0)) m ASL")
        println("  Rock overburden: $(round(elev_above - DETECTOR_ELEVATION; digits=0)) m")
        println()

        run_part0(physics, rock_idx, air_idx, emap;
            straggling=args.straggling, scattering=args.scattering,
            energy_threshold_low=args.energy_threshold_low,
            porous_material=porous_idx, porous_density=p_density,
            porous_thickness=shallow_depth, output_dir=args.output_dir)
    end

    # ── Part 1: Baseline flux studies ───────────────────────────────────────
    if args.part < 0 || args.part == 1
        println("=" ^ 60)
        println(" Part 1: Baseline Flux vs Zenith Angle")
        println("=" ^ 60)
        println()

        rock_density = Float64(physics.tables[rock_idx].density)
        depths  = collect(0.0:100.0:1000.0)
        zeniths = collect(0.0:2.0:60.0)

        println("Rock density: $(round(rock_density; digits=1)) kg/m³")
        println()

        println("-" ^ 40)
        println(" Part 1.1: Flux vs Zenith Angle")
        println("-" ^ 40)
        flux_grid, sigma_grid = compute_flux_vs_angle(physics, rock_idx, depths, zeniths;
            n_samples=args.n_samples, straggling=args.straggling, scattering=args.scattering,
            energy_threshold_low=args.energy_threshold_low, energy_min=args.energy_min,
            energy_max=args.energy_max, porous_material=porous_idx,
            porous_density=p_density, porous_thickness=shallow_depth)
        create_flux_vs_angle_plot(depths, zeniths, flux_grid, sigma_grid;
            output_path=joinpath(args.output_dir, "part1_1_flux_vs_angle.html"),
            title="Part 1.1: Muon Flux vs Zenith Angle by Rock Depth")
        println()

        println("-" ^ 40)
        println(" Part 1.2: Zenith Angle Scattering Std. Dev.")
        println("-" ^ 40)
        std_grid = compute_zenith_std(physics, rock_idx, air_idx, depths, zeniths;
            n_samples=args.n_samples, straggling=args.straggling, scattering=args.scattering,
            energy_threshold_low=args.energy_threshold_low, energy_min=args.energy_min,
            energy_max=args.energy_max, porous_material=porous_idx,
            porous_density=p_density, porous_thickness=shallow_depth)
        create_zenith_std_plot(depths, zeniths, std_grid;
            output_path=joinpath(args.output_dir, "part1_2_zenith_std.html"))
        println()

        println("-" ^ 40)
        println(" Part 1.3: Flux vs Zenith (fixed energies, 1000m depth)")
        println("-" ^ 40)
        depth_1_3 = 1000.0
        energies_1_3, flux_fe, _ = compute_flux_fixed_energy(
            physics, rock_idx, air_idx, depth_1_3, zeniths;
            n_samples=args.n_samples, n_energies=100,
            straggling=args.straggling, scattering=args.scattering,
            energy_threshold_low=args.energy_threshold_low, energy_min=args.energy_min,
            energy_max=args.energy_max, porous_material=porous_idx,
            porous_density=p_density, porous_thickness=shallow_depth)
        create_flux_fixed_energy_plot(energies_1_3, zeniths, flux_fe;
            depth=depth_1_3,
            output_path=joinpath(args.output_dir, "part1_3_flux_vs_zenith_fixed_energy.html"))
        println()
        println("Part 1 complete!")
    end

    # ── Part 2: Aquifer detection ───────────────────────────────────────────
    if (args.part < 0 || args.part == 2) && water_idx != -1 && porous_idx != -1
        kw = (n_samples=args.n_samples, straggling=args.straggling,
              scattering=args.scattering, energy_threshold_low=args.energy_threshold_low,
              energy_min=args.energy_min, energy_max=args.energy_max,
              output_dir=args.output_dir)

        run_part2_1(physics, rock_idx, air_idx, water_idx, porous_idx, p_density; kw...)
        run_part2_2(physics, rock_idx, air_idx, water_idx, porous_idx, p_density; kw...)
    end

    println()
    println("=" ^ 60)
    println(" All simulations complete!")
    println("=" ^ 60)
    println()
    println("Output files saved to: $(args.output_dir)")
    println()
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
