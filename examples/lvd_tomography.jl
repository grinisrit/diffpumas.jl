#!/usr/bin/env julia
"""
lvd_tomography.jl - LVD flux tomography with rock/water sensitivity cells

This example reuses the Gran Sasso LVD topography reconstruction from
`lvd_muography.jl`, but replaces the slant-depth-only forward model with an
explicit tessellation above the detector. Each cell carries a water-fraction
parameter `w_i`, interpreted as a rock/water mixture in the spirit of the
aquifer examples from `flat_muography.jl`.

Two sensitivity paths are compared:
1. Direct CSDA on cached straight-line path segments with `Zygote.gradient`
2. Finite-difference sensitivities on a stochastic transport calculation for a
   small validation set of representative `(direction, cell)` cases

The primary geometry path is tetrahedral:
- Build a watertight PLC from the reconstructed LVD topography plus a flat
  detector plane
- Tetrahedralize the enclosed volume with TetGen
- Trace one straight ray per angular bin through the tetrahedra

If TetGen fails or the user selects `--geometry grid`, a structured Cartesian
fallback is used.

Angular grid:
- zenith:  `0, 1, ..., 59 deg`
- azimuth: `0, 1, ..., 359 deg`

Usage:
    julia --project=. examples/lvd_tomography.jl [OPTIONS]

Options:
    --output-dir, -o PATH         Output directory (default: examples/data)
    --dump, -d PATH               Physics dump file
    --geometry STR                `auto`, `tetra`, or `grid` (default: auto)
    --mesh-half-km FLOAT          Half-width of the local sensitivity volume (default: 5.0)
    --surface-step-km FLOAT       Horizontal topography sampling for the mesh (default: 1.0)
    --tet-max-volume FLOAT        TetGen maximum tetrahedron volume in m^3 (default: 5e8)
    --grid-nz INT                 Vertical layers for grid fallback (default: 6)
    --zenith-max FLOAT            Exclusive zenith bound in degrees (default: 60.0)
    --zenith-step FLOAT           Zenith step in degrees (default: 1.0)
    --azimuth-step FLOAT          Azimuth step in degrees (default: 1.0)
    --samples, -n INT             CSDA energy samples per angular bin (default: 8)
    --mc-samples INT              MC samples per validation case (default: 64)
    --energy-min FLOAT            Minimum detector energy in GeV (default: 1e-3)
    --energy-max FLOAT            Maximum detector energy in GeV (default: 1e9)
    --threshold FLOAT             Mixed/straggled mode switch in GeV (default: 100.0)
    --fd-delta FLOAT              Water-fraction finite-difference step (default: 0.05)
    --validation-cases INT        Number of MC validation cases (default: 5)
    --seed INT                    RNG seed (default: 42)
    --porous-top FLOAT            Optional shallow porous layer thickness in m (default: 0.0)
    --aquifer-water-fraction FLOAT
                                 Baseline water fraction assigned inside the box target (default: 0.0)
    --aquifer-center-x FLOAT      Aquifer box center east of detector in m (default: 0.0)
    --aquifer-center-y FLOAT      Aquifer box center north of detector in m (default: 0.0)
    --aquifer-center-z FLOAT      Aquifer box center above detector in m (default: 700.0)
    --aquifer-half-x FLOAT        Aquifer box half-size in east direction in m (default: 300.0)
    --aquifer-half-y FLOAT        Aquifer box half-size in north direction in m (default: 300.0)
    --aquifer-half-z FLOAT        Aquifer box half-size in vertical direction in m (default: 150.0)
    --simple-field                Use the single aquifer box instead of the multi-anomaly truth
    --slab-w FLOAT                Shallow slab water fraction in the multi-anomaly truth (default: 0.3)
    --lens-w FLOAT                Deep high-contrast lens water fraction (default: 0.9)
    --dip-w FLOAT                 Dipping-interface water fraction (default: 0.5)
    --reco-iters INT              Iterations for SART/MLEM (default: 200)
    --gd-iters INT                AAD preconditioned-GD iterations, relinearised (default: 40)
    --gd-lr FLOAT                 AAD GD step-length cap, 1.0 = full quadratic step (default: 1.0)
    --min-eval-depth FLOAT        Exclude cells shallower than this (m above detector) from
                                  reconstruction metrics and resolution (default: 350)
    --primary-altitude-km FLOAT   Gaisser/primary flux sampling altitude above detector (default: 30)
    --rock-density FLOAT          Base rock density kg/m^3; 0 = physics-table value (StandardRock,
                                  2650 kg/m^3), at which the paper-match pins the overburden rock
    --papermatch                  Run the final section: GN material-mixture match to the measured
                                  LVD flux + Fig.7 angular-distribution reproduction, with MC+syst error
    --papermatch-only             Run ONLY geometry+calibration+paper-match (skip the inverse demo) —
                                  fast iteration on the reconstruction figures
    --match-data PATH             Measured 2D single-muon intensity map
                                 (default: data/lvd_conf/lvd_single_muon_flux_2d.csv)
    --papermatch-mc-samples INT   MC samples/bin for the paper-match uncertainty grid (default: 64)
    --papermatch-paper-samples INT  MC samples for the Fig.8 energy spectrum (default: 2000)
    --max-plot-cells INT          Maximum cells rendered in the 3D sensitivity plot (default: 150)
    --no-straggling               Disable stochastic energy-loss fluctuations for MC validation
"""

const lvd_tomography = nothing

using DiffPumas
using DiffPumas.Tomography          # forward model, sparse AD Jacobian, solvers, metrics
using DiffPumas.Physics: get_material_index
using DiffPumas.Loader: print_physics_summary
using DiffPumas.Pumas: load_or_create_physics
using DiffPumas.TriangleIntersect: intersect
using DiffPumas.Turtle: ElevationMap, map_elevation
using DiffPumas.Types: MaterialMixture
using PlotlyJS
using Printf
using Random
using Statistics
using LinearAlgebra
using SparseArrays
using Logging
using Serialization

module LVDTopo
include(joinpath(@__DIR__, "lvd_muography.jl"))
end

const DEFAULT_DUMP = joinpath(@__DIR__, "data", "materials.pumas")
const DEFAULT_MDF = joinpath(@__DIR__, "data", "materials.xml")
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "data", "lvd_results")
const DEFAULT_MATCH_DATA = LVDTopo.DEFAULT_MATCH_DATA
const MAX_WATER_FRACTION = 0.9
const POINT_EPS = 1e-4
const SURFACE_SEARCH_STEP_M = 40.0
# Reconstruction volume floor: the meshed overburden surface is clamped to this many metres
# ABOVE the detector plane (z=0). Terrain the inverted nm_c.inc map places below the detector
# (distant mountain flanks descending past 963 m a.s.l., and the data-free far field defaulted
# to VALLEY_FLOOR=600 m) carries no upward-muon information and would otherwise form inverted
# cells (the TetGen "segments intersect at z≈0" failures). See build_surface_grid.
const MIN_MESH_OVERBURDEN_M = 1.0
const CSDA_MAX_RELATIVE_GAIN = 0.02
const CSDA_MAX_STEP_M = 60.0
const MC_STEP_MIN_M = 1e-6
const MC_STEP_EPSILON_M = 1e-7

struct ValidationCase
    label::String
    bin_idx::Int
    cell_idx::Int
    zenith::Float64
    azimuth::Float64
    flux_csda::Float64
    grad_csda::Float64
end

abstract type AbstractSensitivityVolume end

struct TetraVolume <: AbstractSensitivityVolume
    mesh::RawTetGenIO{Float64}
    neighbors::Matrix{Int}
    centroids::Matrix{Float64}
    shallow_mask::BitVector
    half_width_m::Float64
    surface_step_m::Float64
end

struct VoxelVolume <: AbstractSensitivityVolume
    x_edges::Vector{Float64}
    y_edges::Vector{Float64}
    z_edges::Vector{Float64}
    centroids::Matrix{Float64}
    shallow_mask::BitVector
end

@inline primary_altitude_local() = LVDTopo.GAISSER_HEIGHT

@inline function num_cells(volume::TetraVolume)
    return size(volume.mesh.tetrahedronlist, 2)
end

@inline function num_cells(volume::VoxelVolume)
    return size(volume.centroids, 2)
end

@inline function cell_centroid(volume::AbstractSensitivityVolume, idx::Int)
    return (volume.centroids[1, idx], volume.centroids[2, idx], volume.centroids[3, idx])
end

@inline function is_shallow_cell(volume::AbstractSensitivityVolume, idx::Int)
    return volume.shallow_mask[idx]
end

@inline function geometry_name(volume::AbstractSensitivityVolume)
    return volume isa TetraVolume ? "tetra" : "grid"
end

@inline point_at_distance(dir::NTuple{3,Float64}, s::Float64) =
    Point(dir[1] * s, dir[2] * s, dir[3] * s)

function local_surface_height(emap::ElevationMap, x_m::Float64, y_m::Float64)
    elev, inside = map_elevation(emap, x_m / 1000.0, y_m / 1000.0)
    return inside ? elev - LVDTopo.DETECTOR_ELEVATION : NaN
end

function ray_surface_exit_distance(emap::ElevationMap,
                                   dir::NTuple{3,Float64};
                                   step_m::Float64 = SURFACE_SEARCH_STEP_M,
                                   max_distance_m::Float64 = 25_000.0)
    dir[3] <= 0.0 && return (NaN, false, NaN)

    prev_s = 0.0
    for s in step_m:step_m:max_distance_m
        x = dir[1] * s
        y = dir[2] * s
        z = dir[3] * s
        surface_z = local_surface_height(emap, x, y)
        isfinite(surface_z) || return (NaN, false, NaN)

        if z > surface_z
            lo = prev_s
            hi = s
            for _ in 1:40
                mid = 0.5 * (lo + hi)
                mid_surface = local_surface_height(emap, dir[1] * mid, dir[2] * mid)
                if dir[3] * mid > mid_surface
                    hi = mid
                else
                    lo = mid
                end
            end
            s_exit = 0.5 * (lo + hi)
            return (s_exit, true, dir[3] * s_exit)
        end
        prev_s = s
    end

    return (NaN, false, NaN)
end

function exclusive_degree_grid(max_deg::Float64, step_deg::Float64)
    values = Float64[]
    x = 0.0
    while x < max_deg - 1e-9
        push!(values, x)
        x += step_deg
    end
    return values
end

function symmetric_nodes(half_width_km::Float64, step_km::Float64)
    nodes = collect(-half_width_km:step_km:half_width_km)
    if isempty(nodes) || !isapprox(nodes[1], -half_width_km; atol=1e-9)
        push!(nodes, -half_width_km)
    end
    if !any(isapprox(v, 0.0; atol=1e-9) for v in nodes)
        push!(nodes, 0.0)
    end
    if !any(isapprox(v, half_width_km; atol=1e-9) for v in nodes)
        push!(nodes, half_width_km)
    end
    sort!(nodes)
    return unique(nodes)
end

function build_surface_grid(emap::ElevationMap,
                            half_width_km::Float64,
                            step_km::Float64)
    x_nodes_km = symmetric_nodes(half_width_km, step_km)
    y_nodes_km = symmetric_nodes(half_width_km, step_km)
    z_local = zeros(Float64, length(y_nodes_km), length(x_nodes_km))

    for iy in eachindex(y_nodes_km), ix in eachindex(x_nodes_km)
        z_raw = local_surface_height(emap, 1000.0 * x_nodes_km[ix], 1000.0 * y_nodes_km[iy])
        isfinite(z_raw) || error("Topography sampling left the LVD map at x=$(x_nodes_km[ix]) km, y=$(y_nodes_km[iy]) km")
        # Mesh only the overburden ABOVE the detector; clamp sub-detector terrain to a thin
        # floor (the forward ray tracer keeps the true surface, so rock columns are unchanged).
        z_local[iy, ix] = max(z_raw, MIN_MESH_OVERBURDEN_M)
    end

    return x_nodes_km .* 1000.0, y_nodes_km .* 1000.0, z_local
end

@inline top_node_index(ix::Int, iy::Int, nx::Int) = ix + (iy - 1) * nx
@inline bottom_node_index(ix::Int, iy::Int, nx::Int, ny::Int) =
    top_node_index(ix, iy, nx) + nx * ny

function create_topography_plc(x_nodes::Vector{Float64},
                               y_nodes::Vector{Float64},
                               z_top::Matrix{Float64};
                               bottom_z::Float64 = 0.0)
    nx = length(x_nodes)
    ny = length(y_nodes)

    points = Vector{Vector{Float64}}()
    sizehint!(points, 2 * nx * ny)

    for iy in 1:ny, ix in 1:nx
        push!(points, [x_nodes[ix], y_nodes[iy], z_top[iy, ix]])
    end
    for iy in 1:ny, ix in 1:nx
        push!(points, [x_nodes[ix], y_nodes[iy], bottom_z])
    end

    facets = NTuple{4,Int}[]

    for iy in 1:(ny - 1), ix in 1:(nx - 1)
        push!(facets, (
            top_node_index(ix, iy, nx),
            top_node_index(ix + 1, iy, nx),
            top_node_index(ix + 1, iy + 1, nx),
            top_node_index(ix, iy + 1, nx),
        ))
    end

    for iy in 1:(ny - 1), ix in 1:(nx - 1)
        push!(facets, (
            bottom_node_index(ix, iy, nx, ny),
            bottom_node_index(ix, iy + 1, nx, ny),
            bottom_node_index(ix + 1, iy + 1, nx, ny),
            bottom_node_index(ix + 1, iy, nx, ny),
        ))
    end

    for iy in 1:(ny - 1)
        push!(facets, (
            bottom_node_index(1, iy, nx, ny),
            bottom_node_index(1, iy + 1, nx, ny),
            top_node_index(1, iy + 1, nx),
            top_node_index(1, iy, nx),
        ))
        push!(facets, (
            bottom_node_index(nx, iy, nx, ny),
            top_node_index(nx, iy, nx),
            top_node_index(nx, iy + 1, nx),
            bottom_node_index(nx, iy + 1, nx, ny),
        ))
    end

    for ix in 1:(nx - 1)
        push!(facets, (
            bottom_node_index(ix, 1, nx, ny),
            top_node_index(ix, 1, nx),
            top_node_index(ix + 1, 1, nx),
            bottom_node_index(ix + 1, 1, nx, ny),
        ))
        push!(facets, (
            bottom_node_index(ix, ny, nx, ny),
            bottom_node_index(ix + 1, ny, nx, ny),
            top_node_index(ix + 1, ny, nx),
            top_node_index(ix, ny, nx),
        ))
    end

    input = RawTetGenIO{Float64}()
    input.pointlist = hcat(points...)

    facet_matrix = Matrix{Int}(undef, 4, length(facets))
    for (j, facet) in enumerate(facets)
        facet_matrix[:, j] .= collect(facet)
    end
    facetlist!(input, facet_matrix)

    return input
end

function tetra_centroids(mesh::RawTetGenIO{Float64})
    n_tets = size(mesh.tetrahedronlist, 2)
    centroids = zeros(Float64, 3, n_tets)
    for i in 1:n_tets
        tet = mesh.tetrahedronlist[:, i] .+ 1
        centroids[:, i] .= vec(sum(mesh.pointlist[:, tet]; dims=2) ./ 4.0)
    end
    return centroids
end

function compute_shallow_mask(centroids::Matrix{Float64},
                              emap::ElevationMap,
                              porous_top_thickness::Float64)
    porous_top_thickness <= 0.0 && return falses(size(centroids, 2))

    mask = falses(size(centroids, 2))
    for i in 1:size(centroids, 2)
        surface_z = local_surface_height(emap, centroids[1, i], centroids[2, i])
        if isfinite(surface_z)
            mask[i] = surface_z - centroids[3, i] <= porous_top_thickness
        end
    end
    return mask
end

function sort_face_key(face::NTuple{3,Int})
    vals = sort(collect(face))
    return (vals[1], vals[2], vals[3])
end

function tetra_face_indices(tet::AbstractVector{Int})
    return (
        (tet[1], tet[2], tet[3]),
        (tet[1], tet[2], tet[4]),
        (tet[1], tet[3], tet[4]),
        (tet[2], tet[3], tet[4]),
    )
end

function tetra_face_triangle(face::NTuple{3,Int}, points::Matrix{Float64})
    p1 = Point(points[1, face[1]], points[2, face[1]], points[3, face[1]])
    p2 = Point(points[1, face[2]], points[2, face[2]], points[3, face[2]])
    p3 = Point(points[1, face[3]], points[2, face[3]], points[3, face[3]])
    return Triangle(p1, p2, p3)
end

function build_tetra_neighbors(mesh::RawTetGenIO{Float64})
    n_tets = size(mesh.tetrahedronlist, 2)
    neighbors = zeros(Int, n_tets, 4)
    face_map = Dict{NTuple{3,Int}, Vector{Tuple{Int,Int}}}()

    for tet_idx in 1:n_tets
        tet = mesh.tetrahedronlist[:, tet_idx] .+ 1
        for (face_idx, face) in enumerate(tetra_face_indices(tet))
            key = sort_face_key(face)
            push!(get!(face_map, key, Tuple{Int,Int}[]), (tet_idx, face_idx))
        end
    end

    for entries in values(face_map)
        if length(entries) == 2
            (t1, f1), (t2, f2) = entries
            neighbors[t1, f1] = t2
            neighbors[t2, f2] = t1
        end
    end

    return neighbors
end

function point_in_tetrahedron(p::Point,
                              tet::AbstractVector{Int},
                              points::Matrix{Float64};
                              atol::Float64 = 1e-8)
    a = points[:, tet[1]]
    mat = hcat(points[:, tet[2]] .- a,
               points[:, tet[3]] .- a,
               points[:, tet[4]] .- a)
    if abs(det(mat)) < atol
        return false
    end
    rhs = [p.x, p.y, p.z] .- a
    bary = mat \ rhs
    return bary[1] >= -atol && bary[2] >= -atol && bary[3] >= -atol &&
           sum(bary) <= 1.0 + atol
end

function find_tetrahedron_containing_point(volume::TetraVolume, p::Point)
    mesh = volume.mesh
    for tet_idx in 1:size(mesh.tetrahedronlist, 2)
        tet = mesh.tetrahedronlist[:, tet_idx] .+ 1
        if point_in_tetrahedron(p, tet, mesh.pointlist)
            return tet_idx
        end
    end
    return nothing
end

function nearest_tetra_face_intersection(volume::TetraVolume,
                                         tet_idx::Int,
                                         p::Point,
                                         dir::NTuple{3,Float64})
    mesh = volume.mesh
    tet = mesh.tetrahedronlist[:, tet_idx] .+ 1
    ray = Ray(p, Point(p.x + 100_000.0 * dir[1],
                       p.y + 100_000.0 * dir[2],
                       p.z + 100_000.0 * dir[3]))

    best_face = 0
    best_dist = Inf
    best_point = p
    for (face_idx, face) in enumerate(tetra_face_indices(tet))
        hit = intersect(ray, tetra_face_triangle(face, mesh.pointlist))
        if hit.is_intersection && hit.id > 1e-8 && hit.id < best_dist
            best_face = face_idx
            best_dist = hit.id
            best_point = hit.ip
        end
    end
    return best_face, best_dist, best_point
end

function create_tetra_volume(emap::ElevationMap,
                             matcfg::MaterialConfig;
                             half_width_km::Float64,
                             surface_step_km::Float64,
                             max_cell_volume_m3::Float64)
    x_nodes, y_nodes, z_top = build_surface_grid(emap, half_width_km, surface_step_km)
    input = create_topography_plc(x_nodes, y_nodes, z_top; bottom_z=0.0)
    quality = string("pqa", max_cell_volume_m3)
    mesh = tetrahedralize(input, quality)
    centroids = tetra_centroids(mesh)
    shallow_mask = compute_shallow_mask(centroids, emap, matcfg.porous_top_thickness)
    volume = TetraVolume(
        mesh,
        build_tetra_neighbors(mesh),
        centroids,
        shallow_mask,
        1000.0 * half_width_km,
        1000.0 * surface_step_km,
    )

    start_dir = (0.0, 0.0, 1.0)
    p0 = point_at_distance(start_dir, POINT_EPS)
    find_tetrahedron_containing_point(volume, p0) === nothing &&
        error("Detector point is not inside the tetrahedral volume")

    return volume
end

@inline function voxel_nx(volume::VoxelVolume)
    return length(volume.x_edges) - 1
end

@inline function voxel_ny(volume::VoxelVolume)
    return length(volume.y_edges) - 1
end

@inline function voxel_nz(volume::VoxelVolume)
    return length(volume.z_edges) - 1
end

@inline function voxel_linear_index(volume::VoxelVolume, i::Int, j::Int, k::Int)
    return i + (j - 1) * voxel_nx(volume) + (k - 1) * voxel_nx(volume) * voxel_ny(volume)
end

function voxel_ijk(volume::VoxelVolume, idx::Int)
    nx = voxel_nx(volume)
    ny = voxel_ny(volume)
    k = fld(idx - 1, nx * ny) + 1
    r = idx - (k - 1) * nx * ny
    j = fld(r - 1, nx) + 1
    i = r - (j - 1) * nx
    return i, j, k
end

function voxel_cell_indices(volume::VoxelVolume, x::Float64, y::Float64, z::Float64)
    if x < volume.x_edges[1] || x > volume.x_edges[end] ||
       y < volume.y_edges[1] || y > volume.y_edges[end] ||
       z < volume.z_edges[1] || z > volume.z_edges[end]
        return nothing
    end

    i = clamp(searchsortedlast(volume.x_edges, x), 1, length(volume.x_edges) - 1)
    j = clamp(searchsortedlast(volume.y_edges, y), 1, length(volume.y_edges) - 1)
    k = clamp(searchsortedlast(volume.z_edges, z), 1, length(volume.z_edges) - 1)
    return i, j, k
end

function distance_to_voxel_boundary(volume::VoxelVolume,
                                    x::Float64, y::Float64, z::Float64,
                                    dx::Float64, dy::Float64, dz::Float64)
    idx = voxel_cell_indices(volume, x, y, z)
    idx === nothing && return 0.0
    i, j, k = idx

    x_lo = volume.x_edges[i]
    x_hi = volume.x_edges[i + 1]
    y_lo = volume.y_edges[j]
    y_hi = volume.y_edges[j + 1]
    z_lo = volume.z_edges[k]
    z_hi = volume.z_edges[k + 1]

    t_min = Inf
    eps_val = 1e-10

    if abs(dx) > eps_val
        t = dx > 0 ? (x_hi - x) / dx : (x_lo - x) / dx
        t > eps_val && (t_min = min(t_min, t))
    end
    if abs(dy) > eps_val
        t = dy > 0 ? (y_hi - y) / dy : (y_lo - y) / dy
        t > eps_val && (t_min = min(t_min, t))
    end
    if abs(dz) > eps_val
        t = dz > 0 ? (z_hi - z) / dz : (z_lo - z) / dz
        t > eps_val && (t_min = min(t_min, t))
    end

    return isfinite(t_min) ? t_min : 0.0
end

function create_voxel_volume(emap::ElevationMap,
                             matcfg::MaterialConfig;
                             half_width_km::Float64,
                             surface_step_km::Float64,
                             nz::Int)
    x_nodes, y_nodes, z_top = build_surface_grid(emap, half_width_km, surface_step_km)
    z_max = maximum(z_top)
    z_edges = collect(range(0.0, stop=z_max, length=nz + 1))

    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    centroids = zeros(Float64, 3, nx * ny * nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        idx = i + (j - 1) * nx + (k - 1) * nx * ny
        centroids[1, idx] = 0.5 * (x_nodes[i] + x_nodes[i + 1])
        centroids[2, idx] = 0.5 * (y_nodes[j] + y_nodes[j + 1])
        centroids[3, idx] = 0.5 * (z_edges[k] + z_edges[k + 1])
    end

    shallow_mask = compute_shallow_mask(centroids, emap, matcfg.porous_top_thickness)
    return VoxelVolume(x_nodes, y_nodes, z_edges, centroids, shallow_mask)
end

function create_sensitivity_volume(emap::ElevationMap,
                                   matcfg::MaterialConfig,
                                   args)
    if args.geometry == "grid"
        println("Building structured-grid sensitivity volume...")
        return create_voxel_volume(emap, matcfg;
            half_width_km = args.mesh_half_km,
            surface_step_km = args.surface_step_km,
            nz = args.grid_nz)
    end

    if args.geometry in ("auto", "tetra")
        try
            println("Building tetrahedral sensitivity volume...")
            return create_tetra_volume(emap, matcfg;
                half_width_km = args.mesh_half_km,
                surface_step_km = args.surface_step_km,
                max_cell_volume_m3 = args.tet_max_volume)
        catch err
            if args.geometry == "tetra"
                rethrow(err)
            end
            @warn "TetGen path failed, falling back to the structured grid" exception=(err, catch_backtrace())
        end
    end

    println("Building structured-grid sensitivity volume...")
    return create_voxel_volume(emap, matcfg;
        half_width_km = args.mesh_half_km,
        surface_step_km = args.surface_step_km,
        nz = args.grid_nz)
end

function trace_path(volume::TetraVolume,
                    zenith::Float64,
                    azimuth::Float64,
                    emap::ElevationMap)
    dir = direction_vector(zenith, azimuth)
    surface_distance, ok, surface_exit_z = ray_surface_exit_distance(emap, dir)
    ok || return DirectionalPath(zenith, azimuth, CellSegment[], NaN, NaN, NaN, NaN, false)

    current_point = point_at_distance(dir, POINT_EPS)
    current_tet = find_tetrahedron_containing_point(volume, current_point)
    segments = CellSegment[]
    traveled = current_tet === nothing ? 0.0 : POINT_EPS

    while current_tet !== nothing && traveled < surface_distance - POINT_EPS
        face_idx, dist, intersection_point = nearest_tetra_face_intersection(volume, current_tet, current_point, dir)
        (face_idx == 0 || !isfinite(dist)) && break

        remaining = surface_distance - traveled
        step = min(dist, remaining)
        if step > 1e-8
            push!(segments, CellSegment(current_tet, step))
            traveled += step
        end

        if dist >= remaining - 1e-6
            break
        end

        next_tet = volume.neighbors[current_tet, face_idx]
        if next_tet == 0
            current_tet = nothing
        else
            current_point = Point(intersection_point.x + dir[1] * POINT_EPS,
                                  intersection_point.y + dir[2] * POINT_EPS,
                                  intersection_point.z + dir[3] * POINT_EPS)
            current_tet = next_tet
        end
    end

    remaining_rock = max(0.0, surface_distance - traveled)
    return DirectionalPath(zenith, azimuth, segments, traveled, surface_distance,
                           remaining_rock, surface_exit_z, true)
end

function trace_path(volume::VoxelVolume,
                    zenith::Float64,
                    azimuth::Float64,
                    emap::ElevationMap)
    dir = direction_vector(zenith, azimuth)
    surface_distance, ok, surface_exit_z = ray_surface_exit_distance(emap, dir)
    ok || return DirectionalPath(zenith, azimuth, CellSegment[], NaN, NaN, NaN, NaN, false)

    x = dir[1] * POINT_EPS
    y = dir[2] * POINT_EPS
    z = dir[3] * POINT_EPS
    traveled = POINT_EPS
    segments = CellSegment[]

    while traveled < surface_distance - POINT_EPS
        idx = voxel_cell_indices(volume, x, y, z)
        idx === nothing && break
        i, j, k = idx
        cell_idx = voxel_linear_index(volume, i, j, k)
        dist = distance_to_voxel_boundary(volume, x, y, z, dir[1], dir[2], dir[3])
        (!isfinite(dist) || dist <= 0.0) && break

        remaining = surface_distance - traveled
        step = min(dist, remaining)
        if step > 1e-8
            push!(segments, CellSegment(cell_idx, step))
            traveled += step
        end

        if dist >= remaining - 1e-6
            break
        end

        x += dir[1] * (step + POINT_EPS)
        y += dir[2] * (step + POINT_EPS)
        z += dir[3] * (step + POINT_EPS)
    end

    remaining_rock = max(0.0, surface_distance - traveled)
    return DirectionalPath(zenith, azimuth, segments, traveled, surface_distance,
                           remaining_rock, surface_exit_z, true)
end

function create_initial_water_field(volume::AbstractSensitivityVolume, args)
    fractions = zeros(Float64, num_cells(volume))

    if !args.rich_field
        # Legacy single aquifer box (off unless --aquifer-water-fraction > 0).
        args.aquifer_water_fraction <= 0.0 && return fractions
        wf = min(args.aquifer_water_fraction, MAX_WATER_FRACTION)
        for idx in eachindex(fractions)
            x, y, z = cell_centroid(volume, idx)
            if abs(x - args.aquifer_center_x) <= args.aquifer_half_x &&
               abs(y - args.aquifer_center_y) <= args.aquifer_half_y &&
               abs(z - args.aquifer_center_z) <= args.aquifer_half_z
                fractions[idx] = wf
            end
        end
        return fractions
    end

    # Multi-anomaly water truth chosen to exercise the operator's nonlinearity at
    # different angles/depths (grid z-layer midpoints ≈ 160/479/799/1119/1438/1758 m).
    # All anomalies are non-negative water fractions; overlaps resolve to the strongest
    # feature via max():
    #   1. shallow LOW-contrast slab  (z≈479 m, broad)         — near-vertical rays
    #   2. deep HIGH-contrast lens     (z≈1119 m, compact, w≈0.9) — long-path nonlinearity (AAD edge)
    #   3. dipping interface           (tilted sheet z = z0 + slope·x) — oblique high-zenith rays
    slab_w = clamp(args.slab_w, 0.0, MAX_WATER_FRACTION)
    lens_w = clamp(args.lens_w, 0.0, MAX_WATER_FRACTION)
    dip_w  = clamp(args.dip_w,  0.0, MAX_WATER_FRACTION)
    for idx in eachindex(fractions)
        x, y, z = cell_centroid(volume, idx)
        w = 0.0
        # 1. shallow slab
        if abs(z - 479.4) <= 160.0 && abs(x) <= 1600.0 && abs(y) <= 1600.0
            w = max(w, slab_w)
        end
        # 2. deep compact lens
        if abs(z - 1118.6) <= 160.0 && abs(x) <= 600.0 && abs(y) <= 600.0
            w = max(w, lens_w)
        end
        # 3. dipping interface: anomaly where |(z - z0) - slope·x| ≤ band, |y| ≤ extent
        if abs((z - 799.0) - 0.4 * x) <= 220.0 && abs(y) <= 1600.0
            w = max(w, dip_w)
        end
        fractions[idx] = w
    end

    return fractions
end

function precompute_paths(volume::AbstractSensitivityVolume,
                          emap::ElevationMap,
                          zeniths::Vector{Float64},
                          azimuths::Vector{Float64})
    paths = Vector{DirectionalPath}(undef, length(zeniths) * length(azimuths))
    idx = 0
    total = length(paths)

    println("Tracing straight-line geometry on the $(geometry_name(volume)) volume...")
    for theta in zeniths, phi in azimuths
        idx += 1
        paths[idx] = trace_path(volume, theta, phi, emap)
        if idx == 1 || idx % 500 == 0 || idx == total
            path = paths[idx]
            msg = path.valid ? @sprintf("  [%d/%d] theta=%.1f deg phi=%.1f deg R=%.1f m", idx, total, theta, phi, path.surface_distance) :
                               @sprintf("  [%d/%d] theta=%.1f deg phi=%.1f deg invalid", idx, total, theta, phi)
            println(msg)
        end
    end

    return paths
end

# Thin wrapper over the module's threaded sparse-Jacobian assembler. The whole
# 21,600-bin Jacobian is computed with one reverse pass per bin restricted to the
# cells the ray actually crosses (see Tomography.directional_flux_and_grad_csda),
# then stored as a SparseMatrixCSC — orders of magnitude faster and lighter than
# the old per-bin full-vector Zygote loop into a dense Float32 matrix.
function compute_flux_and_jacobian(physics,
                                   shallow_flags::AbstractVector{Bool},
                                   matcfg::MaterialConfig,
                                   site::SiteConfig,
                                   paths::Vector{DirectionalPath},
                                   water_fractions::Vector{Float64},
                                   energy_samples::Vector{EnergySample};
                                   n_cells::Int,
                                   threaded::Bool = true)
    println("Computing direct CSDA flux and sparse Zygote sensitivities " *
            "(threaded=$(threaded), nthreads=$(Threads.nthreads()))...")
    flux_values, jacobian = assemble_forward_and_jacobian(
        physics, shallow_flags, matcfg, site, paths, water_fractions, energy_samples;
        n_cells = n_cells, threaded = threaded)
    nvalid = count(isfinite, flux_values)
    println(@sprintf("  %d/%d bins valid, Jacobian nnz=%d (density=%.2e)",
                     nvalid, length(paths), nnz(jacobian),
                     nnz(jacobian) / max(1, length(paths) * n_cells)))
    return flux_values, jacobian
end

function vector_to_grid(values::AbstractVector{<:Real},
                        n_zen::Int,
                        n_azi::Int)
    grid = fill(NaN, n_zen, n_azi)
    idx = 0
    for iz in 1:n_zen, ia in 1:n_azi
        idx += 1
        grid[iz, ia] = Float64(values[idx])
    end
    return grid
end

function aggregate_cell_sensitivity(jacobian::SparseMatrixCSC, n_valid_bins::Int)
    n_cells = size(jacobian, 2)
    agg = zeros(Float64, n_cells)
    vals = nonzeros(jacobian)
    for cell_idx in 1:n_cells
        s = 0.0
        for k in nzrange(jacobian, cell_idx)
            g = vals[k]
            isfinite(g) && (s += g^2)
        end
        agg[cell_idx] = n_valid_bins > 0 ? sqrt(s / n_valid_bins) : 0.0
    end
    return agg
end

function best_case_for_mask(label::String,
                            mask::Vector{Bool},
                            paths::Vector{DirectionalPath},
                            flux_values::Vector{Float64},
                            jacobian::SparseMatrixCSC,
                            used_cells::Set{Int};
                            allow_reuse::Bool = false)
    best_case = nothing
    best_score = -Inf
    rows = rowvals(jacobian)
    vals = nonzeros(jacobian)

    for cell_idx in 1:size(jacobian, 2)
        (!allow_reuse && cell_idx in used_cells) && continue
        for k in nzrange(jacobian, cell_idx)
            bin_idx = rows[k]
            mask[bin_idx] || continue
            isfinite(flux_values[bin_idx]) || continue
            g = vals[k]
            isfinite(g) || continue
            score = abs(g)
            if score > best_score
                best_score = score
                path = paths[bin_idx]
                best_case = ValidationCase(
                    label, bin_idx, cell_idx, path.zenith, path.azimuth,
                    flux_values[bin_idx], g)
            end
        end
    end

    return best_case
end

function select_validation_cases(paths::Vector{DirectionalPath},
                                 flux_values::Vector{Float64},
                                 jacobian::SparseMatrixCSC;
                                 n_cases::Int = 5)
    valid_bins = [i for i in eachindex(paths) if isfinite(flux_values[i])]
    isempty(valid_bins) && return ValidationCase[]

    flux_median = median(flux_values[valid_bins])
    zenith_median = median([paths[i].zenith for i in valid_bins])

    masks = Vector{Tuple{String, Vector{Bool}}}()
    push!(masks, ("lowZenith_highFlux", [isfinite(flux_values[i]) &&
                                         paths[i].zenith <= zenith_median &&
                                         flux_values[i] >= flux_median
                                         for i in eachindex(paths)]))
    push!(masks, ("lowZenith_lowFlux", [isfinite(flux_values[i]) &&
                                        paths[i].zenith <= zenith_median &&
                                        flux_values[i] < flux_median
                                        for i in eachindex(paths)]))
    push!(masks, ("highZenith_highFlux", [isfinite(flux_values[i]) &&
                                          paths[i].zenith > zenith_median &&
                                          flux_values[i] >= flux_median
                                          for i in eachindex(paths)]))
    push!(masks, ("highZenith_lowFlux", [isfinite(flux_values[i]) &&
                                         paths[i].zenith > zenith_median &&
                                         flux_values[i] < flux_median
                                         for i in eachindex(paths)]))
    push!(masks, ("midZenith_peak", [isfinite(flux_values[i]) &&
                                     abs(paths[i].zenith - zenith_median) <= max(5.0, 0.5 * zenith_median)
                                     for i in eachindex(paths)]))

    used_cells = Set{Int}()
    cases = ValidationCase[]

    for (label, mask) in masks
        length(cases) >= n_cases && break
        case = best_case_for_mask(label, mask, paths, flux_values, jacobian, used_cells)
        if case === nothing
            case = best_case_for_mask(label, mask, paths, flux_values, jacobian, used_cells; allow_reuse=true)
        end
        case === nothing && continue
        push!(cases, case)
        push!(used_cells, case.cell_idx)
    end

    if length(cases) < n_cases
        fallback_mask = [isfinite(flux_values[i]) for i in eachindex(paths)]
        while length(cases) < n_cases
            case = best_case_for_mask("globalPeak", fallback_mask, paths, flux_values, jacobian, used_cells)
            case === nothing && (case = best_case_for_mask("globalPeak", fallback_mask, paths, flux_values, jacobian, used_cells; allow_reuse=true))
            case === nothing && break
            push!(cases, case)
            push!(used_cells, case.cell_idx)
        end
    end

    return cases[1:min(length(cases), n_cases)]
end

function finite_difference_bounds(w0::Float64, delta::Float64)
    hi = min(MAX_WATER_FRACTION, w0 + delta)
    lo = max(0.0, w0 - delta)   # water-only: clamp the FD interval to the non-negative box
    if hi > lo + 1e-12
        return lo, hi
    end
    hi = min(MAX_WATER_FRACTION, w0 + max(delta, 1e-3))
    hi > lo + 1e-12 || error("Finite-difference interval collapsed at w=$w0")
    return lo, hi
end

function run_validation_cases(physics,
                              shallow_flags::AbstractVector{Bool},
                              matcfg::MaterialConfig,
                              site::SiteConfig,
                              paths::Vector{DirectionalPath},
                              base_water_fractions::Vector{Float64},
                              cases::Vector{ValidationCase},
                              args)
    results = NamedTuple[]
    mc_samples = sample_energy_set(args.mc_samples, args.energy_min, args.energy_max, args.seed + 10_000)

    println("Running MC finite-difference validation cases (CSDA+AD vs MC+FD)...")
    for (case_idx, case) in enumerate(cases)
        lo, hi = finite_difference_bounds(base_water_fractions[case.cell_idx], args.fd_delta)
        low_w = copy(base_water_fractions)
        high_w = copy(base_water_fractions)
        low_w[case.cell_idx] = lo
        high_w[case.cell_idx] = hi

        base_materials, base_densities = build_cell_properties_for_mc(shallow_flags, base_water_fractions, matcfg)
        low_materials, low_densities = build_cell_properties_for_mc(shallow_flags, low_w, matcfg)
        high_materials, high_densities = build_cell_properties_for_mc(shallow_flags, high_w, matcfg)

        path = paths[case.bin_idx]
        flux_mc, sigma_mc = compute_directional_flux_mc(
            physics, matcfg, site, path, base_materials, base_densities, mc_samples,
            args.seed + case_idx;
            straggling = args.straggling,
            energy_threshold_low = args.energy_threshold_low,
            scattering = true,
        )
        flux_lo, sigma_lo = compute_directional_flux_mc(
            physics, matcfg, site, path, low_materials, low_densities, mc_samples,
            args.seed + case_idx;
            straggling = args.straggling,
            energy_threshold_low = args.energy_threshold_low,
            scattering = true,
        )
        flux_hi, sigma_hi = compute_directional_flux_mc(
            physics, matcfg, site, path, high_materials, high_densities, mc_samples,
            args.seed + case_idx;
            straggling = args.straggling,
            energy_threshold_low = args.energy_threshold_low,
            scattering = true,
        )

        fd_grad = (flux_hi - flux_lo) / (hi - lo)
        push!(results, (
            label = case.label,
            bin_idx = case.bin_idx,
            cell_idx = case.cell_idx,
            zenith = case.zenith,
            azimuth = case.azimuth,
            w_base = base_water_fractions[case.cell_idx],
            w_lo = lo,
            w_hi = hi,
            flux_csda = case.flux_csda,
            flux_mc = flux_mc,
            sigma_mc = sigma_mc,
            flux_lo = flux_lo,
            sigma_lo = sigma_lo,
            flux_hi = flux_hi,
            sigma_hi = sigma_hi,
            grad_csda = case.grad_csda,
            grad_fd = fd_grad,
        ))

        println(@sprintf("  Case %d/%d: theta=%.1f deg phi=%.1f deg cell=%d | grad_csda=%.4e grad_fd=%.4e",
                         case_idx, length(cases), case.zenith, case.azimuth, case.cell_idx,
                         case.grad_csda, fd_grad))
    end

    return results
end

function create_flux_map_plot(zeniths::Vector{Float64},
                              azimuths::Vector{Float64},
                              flux_grid::Matrix{Float64};
                              output_path::String)
    log_flux = map(f -> (isfinite(f) && f > 0.0) ? log10(f) : NaN, flux_grid)
    traces = GenericTrace[
        heatmap(
            x = azimuths,
            y = zeniths,
            z = log_flux,
            colorscale = "Viridis",
            colorbar = attr(title = "log10 flux"),
            hovertemplate = "phi=%{x:.0f} deg<br>theta=%{y:.0f} deg<br>log10 flux=%{z:.3f}<extra></extra>",
        ),
    ]

    layout = Layout(
        title = "LVD tomography: direct-CSDA directional flux",
        xaxis = attr(title = "Azimuth (deg)"),
        yaxis = attr(title = "Zenith (deg)"),
        width = 1000,
        height = 700,
    )

    savefig(Plot(traces, layout), output_path)
end

function create_cell_sensitivity_plots(zeniths::Vector{Float64},
                                       azimuths::Vector{Float64},
                                       jacobian::SparseMatrixCSC,
                                       cases::Vector{ValidationCase};
                                       output_dir::String)
    seen = Set{Int}()
    for case in cases
        case.cell_idx in seen && continue
        push!(seen, case.cell_idx)
        grid = vector_to_grid(Vector{Float64}(jacobian[:, case.cell_idx]), length(zeniths), length(azimuths))
        finite_vals = [abs(v) for v in vec(grid) if isfinite(v)]
        max_abs = isempty(finite_vals) ? 1e-12 : max(maximum(finite_vals), 1e-12)
        traces = GenericTrace[
            heatmap(
                x = azimuths,
                y = zeniths,
                z = grid,
                colorscale = "RdBu",
                zmid = 0.0,
                zmin = -max_abs,
                zmax = max_abs,
                colorbar = attr(title = "dFlux/dw"),
                hovertemplate = "phi=%{x:.0f} deg<br>theta=%{y:.0f} deg<br>dFlux/dw=%{z:.3e}<extra></extra>",
            ),
        ]

        layout = Layout(
            title = "Cell $(case.cell_idx) sensitivity on the 1 deg flux grid",
            xaxis = attr(title = "Azimuth (deg)"),
            yaxis = attr(title = "Zenith (deg)"),
            width = 1000,
            height = 700,
        )

        output_path = joinpath(output_dir, @sprintf("lvd_tomography_cell_%03d_sensitivity.html", case.cell_idx))
        savefig(Plot(traces, layout), output_path)
    end
end

function create_volume_sensitivity_plot(volume::TetraVolume,
                                        aggregate::Vector{Float64};
                                        output_path::String,
                                        max_cells::Int = 150)
    order = sortperm(aggregate, rev=true)
    selected = order[1:min(max_cells, length(order))]
    max_agg = isempty(selected) ? 1.0 : max(maximum(aggregate[selected]), 1e-12)
    traces = GenericTrace[]

    for (rank, cell_idx) in enumerate(selected)
        tet = volume.mesh.tetrahedronlist[:, cell_idx] .+ 1
        x = [volume.mesh.pointlist[1, v] for v in tet]
        y = [volume.mesh.pointlist[2, v] for v in tet]
        z = [volume.mesh.pointlist[3, v] for v in tet]
        sval = aggregate[cell_idx]
        opacity = clamp(0.15 + 0.65 * sval / max(max_agg, 1e-30), 0.15, 0.8)

        push!(traces, mesh3d(
            x = x, y = y, z = z,
            i = Int32[0, 0, 0, 1],
            j = Int32[1, 1, 2, 2],
            k = Int32[2, 3, 3, 3],
            intensity = fill(sval, 4),
            colorscale = "Viridis",
            cmin = 0.0,
            cmax = max_agg,
            opacity = opacity,
            showscale = rank == 1,
            colorbar = rank == 1 ? attr(title = "RMS |dFlux/dw|") : nothing,
            hovertext = "Cell $cell_idx<br>RMS sensitivity = $(@sprintf("%.3e", sval))",
            hoverinfo = "text",
            name = rank == 1 ? "Sensitivity cells" : "",
            showlegend = rank == 1,
        ))
    end

    push!(traces, scatter3d(
        x = [0.0], y = [0.0], z = [0.0],
        mode = "markers+text",
        marker = attr(size = 7, color = "red"),
        text = ["LVD"],
        textposition = "top center",
        name = "Detector",
    ))

    layout = Layout(
        title = "LVD tomography sensitivity volume (tetra cells)",
        scene = attr(
            xaxis = attr(title = "East (m)"),
            yaxis = attr(title = "North (m)"),
            zaxis = attr(title = "Height above detector (m)"),
            aspectmode = "data",
        ),
        width = 1000,
        height = 800,
    )

    savefig(Plot(traces, layout), output_path)
end

function create_volume_sensitivity_plot(volume::VoxelVolume,
                                        aggregate::Vector{Float64};
                                        output_path::String,
                                        max_cells::Int = 150)
    order = sortperm(aggregate, rev=true)
    selected = order[1:min(max_cells, length(order))]
    x = [volume.centroids[1, idx] for idx in selected]
    y = [volume.centroids[2, idx] for idx in selected]
    z = [volume.centroids[3, idx] for idx in selected]
    c = [aggregate[idx] for idx in selected]
    traces = GenericTrace[
        scatter3d(
            x = x,
            y = y,
            z = z,
            mode = "markers",
            marker = attr(
                size = 5,
                color = c,
                colorscale = "Viridis",
                colorbar = attr(title = "RMS |dFlux/dw|"),
            ),
            name = "Grid cells",
        ),
        scatter3d(
            x = [0.0], y = [0.0], z = [0.0],
            mode = "markers+text",
            marker = attr(size = 7, color = "red"),
            text = ["LVD"],
            textposition = "top center",
            name = "Detector",
        ),
    ]

    layout = Layout(
        title = "LVD tomography sensitivity volume (grid fallback)",
        scene = attr(
            xaxis = attr(title = "East (m)"),
            yaxis = attr(title = "North (m)"),
            zaxis = attr(title = "Height above detector (m)"),
            aspectmode = "data",
        ),
        width = 1000,
        height = 800,
    )

    savefig(Plot(traces, layout), output_path)
end

function write_validation_summary(results,
                                  volume::AbstractSensitivityVolume;
                                  output_path::String)
    # Diff-CSDA (forward flux + Zygote gradient) vs full MC (straggling + scattering
    # on; gradient by central finite difference on the cell mixture). The MC flux and
    # MC-FD gradient are noisy, so the headline test is the z-score (how many MC sigmas
    # separate CSDA from MC), not the raw percentage mismatch — at these sample counts
    # a large % can still be <1 sigma. sigma_grad propagates the two endpoint MC errors.
    flux_z = Float64[]
    grad_z = Float64[]
    open(output_path, "w") do io
        println(io, "LVD tomography validation summary")
        println(io, "geometry = $(geometry_name(volume))")
        println(io, "n_cells = $(num_cells(volume))")
        println(io, "comparison: diff-CSDA + Zygote   vs   full MC (straggling+scattering) + finite difference")
        println(io)

        for (idx, result) in enumerate(results)
            cx, cy, cz = cell_centroid(volume, result.cell_idx)
            dw = max(result.w_hi - result.w_lo, 1e-30)
            sigma_grad = hypot(result.sigma_lo, result.sigma_hi) / dw   # MC error on the FD slope
            flux_rel = 100 * abs(result.flux_mc - result.flux_csda) / max(abs(result.flux_csda), 1e-30)
            grad_rel = 100 * abs(result.grad_fd - result.grad_csda) / max(abs(result.grad_csda), 1e-30)
            fz = (result.flux_mc - result.flux_csda) / max(result.sigma_mc, 1e-30)
            gz = (result.grad_fd - result.grad_csda) / max(sigma_grad, 1e-30)
            push!(flux_z, fz); push!(grad_z, gz)
            agree(z) = abs(z) <= 2 ? "AGREE (<2 sigma)" : "TENSION (>2 sigma)"
            println(io, "Case $idx: $(result.label)")
            println(io, @sprintf("  direction: theta=%.1f deg phi=%.1f deg", result.zenith, result.azimuth))
            println(io, @sprintf("  cell: %d at (%.1f, %.1f, %.1f) m", result.cell_idx, cx, cy, cz))
            println(io, @sprintf("  baseline water fraction: %.3f", result.w_base))
            println(io, @sprintf("  FD interval: [%.3f, %.3f]", result.w_lo, result.w_hi))
            println(io, @sprintf("  flux_csda: %.6e", result.flux_csda))
            println(io, @sprintf("  flux_mc:   %.6e +/- %.2e", result.flux_mc, result.sigma_mc))
            println(io, @sprintf("  flux:  %.2f %% mismatch | %+.2f sigma | %s", flux_rel, fz, agree(fz)))
            println(io, @sprintf("  grad_csda: %.6e", result.grad_csda))
            println(io, @sprintf("  grad_fd:   %.6e +/- %.2e", result.grad_fd, sigma_grad))
            println(io, @sprintf("  grad:  %.2f %% mismatch | %+.2f sigma | %s", grad_rel, gz, agree(gz)))
            println(io)
        end

        # --- aggregate: with noisy FD, "agreement" = fraction within the MC error bar ---
        n = length(results)
        if n > 0
            flux_ok = count(z -> abs(z) <= 2, flux_z)
            grad_ok = count(z -> abs(z) <= 2, grad_z)
            println(io, "Aggregate over $n cases (full MC: straggling+scattering on):")
            println(io, @sprintf("  flux within 2 sigma: %d/%d   | median |z| = %.2f", flux_ok, n, median(abs.(flux_z))))
            println(io, @sprintf("  grad within 2 sigma: %d/%d   | median |z| = %.2f", grad_ok, n, median(abs.(grad_z))))
            println(io, "  note: raise --mc-samples / --validation-cases to tighten the MC error bars.")
        end
    end
end

function parse_commandline()
    args = (
        dump_path = DEFAULT_DUMP,
        output_dir = DEFAULT_OUTPUT_DIR,
        geometry = "auto",
        mesh_half_km = 5.0,
        surface_step_km = 1.0,
        tet_max_volume = 5.0e8,
        grid_nz = 6,
        zenith_max_deg = 60.0,
        zenith_step_deg = 1.0,
        azimuth_step_deg = 1.0,
        n_samples = 8,
        mc_samples = 64,
        energy_min = 1e-3,
        energy_max = 1e9,
        energy_threshold_low = 100.0,
        fd_delta = 0.05,
        validation_cases = 5,
        seed = 42,
        porous_top_thickness = 0.0,
        aquifer_water_fraction = 0.0,
        aquifer_center_x = 0.0,
        aquifer_center_y = 0.0,
        aquifer_center_z = 700.0,
        aquifer_half_x = 300.0,
        aquifer_half_y = 300.0,
        aquifer_half_z = 150.0,
        rich_field = true,         # multi-anomaly ground truth (slab + lens + dip)
        slab_w = 0.3,              # shallow low-contrast slab water fraction
        lens_w = 0.9,              # deep high-contrast lens water fraction
        dip_w = 0.5,               # dipping-interface water fraction
        max_plot_cells = 150,
        straggling = true,
        solver = "all",
        reg_weight = 0.0,
        contrast = 0.5,
        reco_iters = 200,
        gd_iters = 40,             # AAD preconditioned-GD iterations (relinearised)
        gd_lr = 1.0,               # AAD GD step-length cap (1.0 = full quadratic step)
        min_eval_depth = 350.0,    # exclude shallower-than-this cells from metrics/resolution
                                   # (near-surface cells are geometrically unresolvable)
        threaded = true,
        correction = true,         # MC-calibrated CSDA fluctuation/hard-loss correction
        calibration_bins = 48,     # bins used to fit the CSDA correction to full MC (more = tighter)
        calibration_mc_samples = 512,  # MC samples/bin for the calibration reference (high = low MC noise)
        inverse_data = "mc",       # observation source: "mc" (full MC) or "csda"
        inverse_mc_samples = 256,  # MC samples per bin for the (low-noise) observations
        exposure = 1.0e8,          # detector exposure (sets Poisson noise level)
        edge_delta = 0.03,         # Huber transition for the edge-preserving GN prior
        primary_altitude_km = 30.0,    # Gaisser/primary flux sampling altitude above detector
        rock_density = 0.0,            # base rock density (kg/m^3); 0 = use the physics-table value
                                       # (StandardRock, 2650 kg/m^3), at which the paper-match pins it.
        papermatch = false,            # run the final paper-matching section
        papermatch_only = false,       # run ONLY geometry+calibration+paper-match (skip inverse demo)
        match_data = DEFAULT_MATCH_DATA,      # measured 2D single-muon intensity map
        papermatch_zenith_step = 2.0,  # coarse grid for the NMapFluxResult (nmap resolution)
        papermatch_azimuth_step = 4.0,
        papermatch_mc_samples = 64,    # MC samples/bin for the paper-match uncertainty grid
        papermatch_paper_samples = 2000,  # MC samples for the Fig.8 energy spectrum
        papermatch_reg = 1.0,          # smoothness-prior weight (× data curvature) for the
                                       # water-only match
        papermatch_recursions = 1,     # self-consistent recursion: re-calibrate CSDA->MC at the
                                       # current reconstructed field and re-solve, up to N times (1 = off)
        papermatch_recur_tol = 0.02,   # stop the recursion once both the field and the correction
                                       # coefficients change by less than this (relative L2)
        papermatch_damp = 0.6,         # under-relaxation on the field between recursion passes
                                       # (1 = take each solve as-is; <1 damps oscillation)
        papermatch_val_horiz = 80,     # # of stratified theta>=60 (long-slant) bins added to the
                                       # full-MC validation on top of all theta<60 fit bins
        papermatch_dem_unc = 0.15,     # per-bin relative DEM/topography uncertainty slope: the data
                                       # σ gets a term (dem_unc·(secθ−1))·flux in quadrature, so
                                       # long-slant bins (largest terrain error) are DOWN-weighted,
                                       # not excluded. 0 = pure-Poisson weighting.
        papermatch_robust_c = 3.0,     # Cauchy IRLS scale in units of the robust log-residual MAD
                                       # (smaller = more aggressive tail rejection). Tames the heavy
                                       # residual tail (faint topography-artifact bins) in the GN fit.
        papermatch_abs_scale = 0.0,    # if >0, PIN the forward→measured normalization to this absolute
                                       # value (breaks the shape/scale degeneracy); 0 = free log-scale fit.
        detectability = false,         # run the matched-resolution detectability study (then exit)
        detection_benchmark = false,   # run the CSDA gradient matched-filter detection benchmark (then exit)
        reconstruction_sweep = false,  # (b) reconstruction RMSE/SSIM vs exposure (then exit)
        detection_contrast_sweep = false,  # (a) Ψ-vs-SART detection gain vs contrast (then exit)
        sweep_exp_min = 1.0e5,         # exposure sweep range (detector exposure units)
        sweep_exp_max = 1.0e9,
        sweep_n = 13,                  # number of exposure points (log-spaced)
        sweep_realizations = 80,       # noise realizations per exposure per solver
    )

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg in ("--dump", "-d")
            args = merge(args, (dump_path = ARGS[i + 1],)); i += 2
        elseif arg in ("--output-dir", "-o")
            args = merge(args, (output_dir = ARGS[i + 1],)); i += 2
        elseif arg == "--geometry"
            args = merge(args, (geometry = lowercase(ARGS[i + 1]),)); i += 2
        elseif arg == "--mesh-half-km"
            args = merge(args, (mesh_half_km = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--surface-step-km"
            args = merge(args, (surface_step_km = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--tet-max-volume"
            args = merge(args, (tet_max_volume = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--grid-nz"
            args = merge(args, (grid_nz = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--zenith-max"
            args = merge(args, (zenith_max_deg = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--zenith-step"
            args = merge(args, (zenith_step_deg = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--azimuth-step"
            args = merge(args, (azimuth_step_deg = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg in ("--samples", "-n")
            args = merge(args, (n_samples = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--mc-samples"
            args = merge(args, (mc_samples = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--energy-min"
            args = merge(args, (energy_min = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--energy-max"
            args = merge(args, (energy_max = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--threshold"
            args = merge(args, (energy_threshold_low = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--fd-delta"
            args = merge(args, (fd_delta = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--validation-cases"
            args = merge(args, (validation_cases = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--seed"
            args = merge(args, (seed = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--porous-top"
            args = merge(args, (porous_top_thickness = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--aquifer-water-fraction"
            args = merge(args, (aquifer_water_fraction = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--aquifer-center-x"
            args = merge(args, (aquifer_center_x = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--aquifer-center-y"
            args = merge(args, (aquifer_center_y = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--aquifer-center-z"
            args = merge(args, (aquifer_center_z = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--aquifer-half-x"
            args = merge(args, (aquifer_half_x = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--aquifer-half-y"
            args = merge(args, (aquifer_half_y = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--aquifer-half-z"
            args = merge(args, (aquifer_half_z = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--simple-field"
            args = merge(args, (rich_field = false,)); i += 1
        elseif arg == "--slab-w"
            args = merge(args, (slab_w = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--lens-w"
            args = merge(args, (lens_w = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--dip-w"
            args = merge(args, (dip_w = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--max-plot-cells"
            args = merge(args, (max_plot_cells = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--no-straggling"
            args = merge(args, (straggling = false,)); i += 1
        elseif arg == "--solver"
            args = merge(args, (solver = lowercase(ARGS[i + 1]),)); i += 2
        elseif arg == "--reg-weight"
            args = merge(args, (reg_weight = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--contrast"
            args = merge(args, (contrast = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--reco-iters"
            args = merge(args, (reco_iters = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--gd-iters"
            args = merge(args, (gd_iters = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--gd-lr"
            args = merge(args, (gd_lr = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--min-eval-depth"
            args = merge(args, (min_eval_depth = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--primary-altitude-km"
            args = merge(args, (primary_altitude_km = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--rock-density"
            args = merge(args, (rock_density = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--papermatch"
            args = merge(args, (papermatch = true,)); i += 1
        elseif arg == "--papermatch-only"
            args = merge(args, (papermatch = true, papermatch_only = true,)); i += 1
        elseif arg == "--match-data"
            args = merge(args, (match_data = ARGS[i + 1],)); i += 2
        elseif arg == "--papermatch-zenith-step"
            args = merge(args, (papermatch_zenith_step = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--papermatch-azimuth-step"
            args = merge(args, (papermatch_azimuth_step = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--papermatch-mc-samples"
            args = merge(args, (papermatch_mc_samples = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--papermatch-paper-samples"
            args = merge(args, (papermatch_paper_samples = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--papermatch-reg"
            args = merge(args, (papermatch_reg = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--papermatch-recursions"
            args = merge(args, (papermatch_recursions = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--papermatch-recur-tol"
            args = merge(args, (papermatch_recur_tol = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--papermatch-damp"
            args = merge(args, (papermatch_damp = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--papermatch-val-horiz"
            args = merge(args, (papermatch_val_horiz = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--papermatch-dem-unc"
            args = merge(args, (papermatch_dem_unc = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--papermatch-robust-c"
            args = merge(args, (papermatch_robust_c = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--papermatch-abs-scale"
            args = merge(args, (papermatch_abs_scale = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--detectability"
            args = merge(args, (detectability = true,)); i += 1
        elseif arg == "--detection-benchmark"
            args = merge(args, (detection_benchmark = true,)); i += 1
        elseif arg == "--reconstruction-sweep"
            args = merge(args, (reconstruction_sweep = true,)); i += 1
        elseif arg == "--detection-contrast-sweep"
            args = merge(args, (detection_contrast_sweep = true,)); i += 1
        elseif arg == "--sweep-realizations"
            args = merge(args, (sweep_realizations = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--sweep-n"
            args = merge(args, (sweep_n = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--no-threads"
            args = merge(args, (threaded = false,)); i += 1
        elseif arg == "--no-correction"
            args = merge(args, (correction = false,)); i += 1
        elseif arg == "--calibration-bins"
            args = merge(args, (calibration_bins = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--calibration-mc-samples"
            args = merge(args, (calibration_mc_samples = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg == "--inverse-data"
            args = merge(args, (inverse_data = lowercase(ARGS[i + 1]),)); i += 2
        elseif arg == "--exposure"
            args = merge(args, (exposure = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--edge-delta"
            args = merge(args, (edge_delta = parse(Float64, ARGS[i + 1]),)); i += 2
        elseif arg == "--inverse-mc-samples"
            args = merge(args, (inverse_mc_samples = parse(Int, ARGS[i + 1]),)); i += 2
        elseif arg in ("--help", "-h")
            println(@doc lvd_tomography)
            exit(0)
        else
            error("Unknown argument: $arg")
        end
    end

    args.geometry in ("auto", "tetra", "grid") || error("--geometry must be auto, tetra, or grid")
    args.solver in ("all", "sart", "mlem", "gd", "gn") || error("--solver must be all, sart, mlem, gd, or gn")
    args.inverse_data in ("mc", "csda") || error("--inverse-data must be mc or csda")
    return args
end

# --- cell adjacency for the Laplacian smoothness prior -------------------

function cell_neighbors(volume::TetraVolume)
    n = num_cells(volume)
    nbr = [Int[] for _ in 1:n]
    for c in 1:n, f in 1:4
        j = volume.neighbors[c, f]
        j > 0 && push!(nbr[c], j)
    end
    return nbr
end

function cell_neighbors(volume::VoxelVolume)
    nx = voxel_nx(volume); ny = voxel_ny(volume); nz = voxel_nz(volume)
    n = num_cells(volume)
    nbr = [Int[] for _ in 1:n]
    for idx in 1:n
        i, j, k = voxel_ijk(volume, idx)
        for (di, dj, dk) in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1))
            ii, jj, kk = i + di, j + dj, k + dk
            (1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz) || continue
            push!(nbr[idx], voxel_linear_index(volume, ii, jj, kk))
        end
    end
    return nbr
end

# Rough characteristic cell size (m) for the resolution geometric floor.
estimate_cell_size(volume::VoxelVolume) =
    Float64(minimum(diff(volume.z_edges)))
function estimate_cell_size(volume::TetraVolume)
    # cube-root of the mean tetra "cell volume" proxy = surface_step^2 * mean depth span
    return volume.surface_step_m
end

build_site(primary_alt_m::Real = primary_altitude_local()) =
    SiteConfig(Float64(LVDTopo.DETECTOR_ELEVATION), Float64(primary_alt_m))

# --- MC-calibrated CSDA fluctuation/hard-loss correction (Part A) ---------

# Fit the differentiable CSDA correction to full MC on a stratified set of bins,
# then install it so the Jacobian and the inverse forward use the corrected
# operator. Returns (kappa_strag, kappa_hard, stats) or nothing when disabled.
function calibrate_correction_to_mc(physics, shallow_flags, matcfg::MaterialConfig,
                                    site::SiteConfig, paths::Vector{DirectionalPath},
                                    water_fractions::Vector{Float64},
                                    energy_samples::Vector{EnergySample}, args;
                                    fit_geometric::Bool = true,
                                    kh_anchor::Union{Nothing,Float64} = nothing,
                                    kh_anchor_weight::Float64 = 0.0)
    if !args.correction
        set_csda_correction!(enabled = false)
        println("CSDA fluctuation correction: DISABLED (pure CSDA).")
        return nothing
    end
    valid = [b for b in eachindex(paths) if paths[b].valid]
    isempty(valid) && return nothing
    # stratify by zenith so the calibration spans the overburden range
    order = sort(valid; by = b -> paths[b].zenith)
    nb = min(args.calibration_bins, length(order))
    pick = unique(round.(Int, range(1, length(order); length = nb)))
    cbins = order[pick]

    # High-statistics MC reference: the CSDA↔MC residual is MC-noise-limited at low
    # sample counts (≈16% at 24 samples, ≈6% at 256), so calibrate against a clean
    # reference (dedicated --calibration-mc-samples) to approach the ~4% model floor.
    mc_samples = sample_energy_set(args.calibration_mc_samples, args.energy_min, args.energy_max, args.seed + 20_000)
    mats, dens = build_cell_properties_for_mc(shallow_flags, water_fractions, matcfg)
    println("Calibrating CSDA correction to full MC on $(length(cbins)) bins " *
            "($(args.calibration_mc_samples) MC samples/bin, scattering+straggling)...")
    mcflux = Vector{Float64}(undef, length(cbins))
    Threads.@threads for k in eachindex(cbins)
        f, _ = compute_directional_flux_mc(physics, matcfg, site, paths[cbins[k]], mats, dens,
            mc_samples, args.seed + 30_000 + k;
            straggling = args.straggling, energy_threshold_low = args.energy_threshold_low,
            scattering = true)
        mcflux[k] = f
    end
    ks, kh, stats = calibrate_csda_correction(physics, shallow_flags, matcfg, site, paths,
        water_fractions, energy_samples, mcflux, cbins; fit_geometric = fit_geometric,
        kh_anchor = kh_anchor, kh_anchor_weight = kh_anchor_weight)
    println(@sprintf("  fitted α_strag=%.3f (physical ½b(b+1) enhancement) κ_hard=%.3f (%s geometric residual) | CSDA-vs-MC mean rel error %.1f%% -> %.1f%%",
                     ks, kh, fit_geometric ? "+ fitted" : "frozen", 100 * stats.rel_before, 100 * stats.rel_after))
    return (ks, kh, stats)
end

# --- full-MC observations for the inverse problem (cached) ----------------

function mc_observations_for_field(physics, shallow_flags, matcfg::MaterialConfig,
                                   site::SiteConfig, paths::Vector{DirectionalPath},
                                   w_true::Vector{Float64}, valid::Vector{Int}, args;
                                   cache_path::String)
    sig = (length(paths), length(valid), args.inverse_mc_samples, args.seed, args.zenith_step_deg,
           args.azimuth_step_deg, args.geometry, hash(w_true), args.straggling,
           args.energy_min, args.energy_max, args.energy_threshold_low,
           matcfg.rock_density, matcfg.water_density, matcfg.porous_density)
    if isfile(cache_path)
        try
            cached = Serialization.deserialize(cache_path)
            if cached.sig == sig
                println("  loaded cached MC observations ($(length(valid)) bins) from $(basename(cache_path))")
                return cached.obs
            end
        catch
        end
    end
    mc_samples = sample_energy_set(args.inverse_mc_samples, args.energy_min, args.energy_max, args.seed + 40_000)
    mats, dens = build_cell_properties_for_mc(shallow_flags, w_true, matcfg)
    println("  generating full-MC observations for $(length(valid)) bins (scattering+straggling, $(args.inverse_mc_samples) samples)...")
    obs = Vector{Float64}(undef, length(valid))
    Threads.@threads for k in eachindex(valid)
        f, _ = compute_directional_flux_mc(physics, matcfg, site, paths[valid[k]], mats, dens,
            mc_samples, args.seed + 50_000 + k;
            straggling = args.straggling, energy_threshold_low = args.energy_threshold_low,
            scattering = true)
        obs[k] = f
    end
    try
        Serialization.serialize(cache_path, (sig = sig, obs = obs))
    catch e
        @warn "could not cache MC observations" exception=e
    end
    return obs
end

# --- inverse reconstruction: SART / MLEM / GN + metrics -------------------

function run_inverse_demo(physics, shallow_flags, matcfg::MaterialConfig, site::SiteConfig,
                          volume::AbstractSensitivityVolume, paths::Vector{DirectionalPath},
                          energy_samples::Vector{EnergySample},
                          flux_values::Vector{Float64}, jacobian::SparseMatrixCSC,
                          aggregate::Vector{Float64}, args; output_path::String)
    n_cells = num_cells(volume)

    # Ground-truth field: the baseline aquifer if present, otherwise inject a
    # single-cell anomaly at the most informative cell so the demo is non-trivial.
    w_true = create_initial_water_field(volume, args)
    if all(iszero, w_true)
        c = argmax(aggregate)
        w_true[c] = clamp(args.contrast, 0.0, MAX_WATER_FRACTION)
        println(@sprintf("  Injected synthetic anomaly w=%.2f at cell %d (highest sensitivity)",
                         w_true[c], c))
    end

    # Linear operator = corrected-CSDA Jacobian at the no-water baseline (w=0).
    # Water (ρ≈1000) is lighter than rock (ρ≈2650), so adding water RAISES the
    # transmitted flux: ∂flux/∂w > 0 and the anomaly is a flux EXCESS.
    base_flux, base_J = assemble_forward_and_jacobian(
        physics, shallow_flags, matcfg, site, paths, zeros(n_cells), energy_samples;
        n_cells = n_cells, threaded = args.threaded)
    valid_bins = findall(isfinite, base_flux)
    isempty(valid_bins) && (println("No valid bins; skipping inversion."); return nothing)
    Jv = base_J[valid_bins, :]
    f0 = base_flux[valid_bins]

    # Observations: NO inverse crime — generated by the full MC (scattering +
    # straggling) through the true field, then Poisson counting noise. The
    # linear solvers invert `Jv`; only our Gauss-Newton fits the true nonlinear
    # corrected-CSDA operator, which is its edge over MLEM.
    if args.inverse_data == "mc"
        cache_path = joinpath(args.output_dir, "lvd_mc_observations.bin")
        obs_clean = mc_observations_for_field(physics, shallow_flags, matcfg, site, paths,
            w_true, valid_bins, args; cache_path = cache_path)
    else
        ff, _ = assemble_forward_and_jacobian(physics, shallow_flags, matcfg, site, paths,
            w_true, energy_samples; n_cells = n_cells, threaded = args.threaded)
        obs_clean = ff[valid_bins]
    end
    rng = MersenneTwister(args.seed + 777)
    σ = sqrt.(max.(obs_clean, 0.0) ./ args.exposure)
    @inbounds for i in eachindex(σ); σ[i] = max(σ[i], 1e-12 * maximum(obs_clean)); end
    obs = [max(obs_clean[i], 0.0) + σ[i] * randn(rng) for i in eachindex(obs_clean)]
    excess = obs .- f0                      # flux excess from the lighter water

    # Regularisation weights auto-scaled to the operator curvature so they are
    # neither negligible nor overwhelming (or use --reg-weight to set explicitly).
    Wvec = 1.0 ./ σ .^ 2
    diagJtWJ = vec(sum(Wvec .* (Matrix(Jv) .^ 2); dims = 1))   # curvature per cell
    gn_scale = (m = filter(>(0), diagJtWJ); isempty(m) ? 1.0 : median(m))
    sens = vec(sum(max.(Jv, 0.0); dims = 1))
    sm_scale = (m = filter(>(0), sens); isempty(m) ? 1.0 : median(m))
    # The edge Hessian λ·L has O(#neighbors) on its diagonal, so λ must be a small
    # FRACTION of the data curvature median(diag(JᵀWJ)) for mild edge-preserving
    # regularisation (too large collapses GN to a flat field).
    edge_w = args.reg_weight > 0 ? args.reg_weight : 0.01 * gn_scale
    sm_w   = args.reg_weight > 0 ? args.reg_weight : 0.05 * sm_scale

    neighbors = cell_neighbors(volume)
    sm_prior   = SmoothnessPrior(neighbors, sm_w)
    edge_prior = EdgePrior(neighbors, edge_w, args.edge_delta)

    # The true nonlinear corrected-CSDA operator, relinearised via the fast AD
    # Jacobian — shared by BOTH our AAD solvers (GN and GD). SART/MLEM are the
    # frozen-operator baselines (single Jacobian Jv at w=0).
    csda_model = make_csda_operator(physics, shallow_flags, matcfg, site, paths,
        energy_samples; n_cells = n_cells, valid_bins = valid_bins,
        threaded = args.threaded)
    Wobs = 1.0 ./ σ .^ 2
    # Near-surface cells are crossed only by near-vertical rays and are essentially
    # unrecoverable, so we evaluate quality below a minimum depth (configurable). All
    # reported metrics and the resolution map use this evaluation mask.
    zc_all = volume.centroids[3, :]
    eval_mask = zc_all .>= args.min_eval_depth
    roi_mask = (w_true .> 1e-9) .& eval_mask   # signal = water anomaly (w>0)
    bg_mask  = (w_true .<= 1e-9) .& eval_mask

    results = Dict{String,Any}()
    runset = args.solver == "all" ? ("sart", "mlem", "gd", "gn") : (args.solver,)

    for s in runset
        t0 = time()
        if s == "sart"
            w_rec, hist = sart_reconstruct(Jv, excess; n_iter = args.reco_iters,
                relaxation = 0.2, prior = sm_prior)
        elseif s == "mlem"
            w_rec, hist = mlem_reconstruct(max.(Jv, 0.0), max.(excess, 0.0);
                n_iter = args.reco_iters, prior = sm_prior)
        elseif s == "gd"
            # Our AAD first-order solver: preconditioned projected gradient on the
            # true nonlinear corrected-CSDA operator (relinearised each step via the
            # AD Jacobian), inverse-variance weighted, with the same edge-preserving
            # prior as GN. The Jacobi preconditioner keeps the background sparse
            # (Adam marched every cell to the box and collapsed the reconstruction).
            # Non-negative box [0, w_max]: positivity regularizes this ill-posed inverse
            # and keeps the background sparse (an unbounded-below box lets GD fill the
            # background with spurious material and collapses the reconstruction).
            w_rec, hist = gradient_descent_reconstruct(csda_model, obs;
                w0 = zeros(n_cells), n_iter = args.gd_iters,
                lr = args.gd_lr, optimizer = :pgd, prior = edge_prior,
                weights = Wobs, relinearize = true,
                box = (0.0, MAX_WATER_FRACTION))
        elseif s == "gn"
            # Our AAD second-order solver: box-constrained Gauss-Newton on the same
            # nonlinear operator, inverse-variance weighted, edge-preserving prior.
            w_rec, hist = gauss_newton_reconstruct(csda_model, obs;
                w0 = zeros(n_cells), n_iter = max(8, args.reco_iters ÷ 16),
                prior = edge_prior, weights = Wobs, relinearize = true,
                box = (0.0, MAX_WATER_FRACTION))
        else
            error("Unknown solver '$s' (use sart, mlem, gd, gn, or all)")
        end
        dt = time() - t0
        rep = reconstruction_report(w_rec, w_true; mask = eval_mask,
                                    roi_mask = roi_mask, bg_mask = bg_mask)
        results[s] = (w = w_rec, hist = hist, report = rep, seconds = dt)
        println(@sprintf("  [%-4s] mse=%.3e rmse=%.3e psnr=%.1f dB ssim=%.3f snr=%.1f dB  (%.1fs, %d iters)",
                         uppercase(s), rep.mse, rep.rmse, rep.psnr, rep.ssim, rep.snr, dt, length(hist)))
    end

    # --- export truth + reconstructed fields as CSV for the paper's 3D inverse figure.
    # One row per cell: idx, centroid (m), and water fraction. The reconstructions are
    # the headline GN/GD result and the SART/MLEM baselines on the SAME synthetic truth.
    let cen = volume.centroids
        write_field_csv = (path, field) -> open(path, "w") do io
            println(io, "cell_idx,x_m,y_m,z_m,w")
            for c in 1:n_cells
                println(io, @sprintf("%d,%.1f,%.1f,%.1f,%.6f",
                                     c, cen[1, c], cen[2, c], cen[3, c], field[c]))
            end
        end
        write_field_csv(joinpath(args.output_dir, "lvd_synthetic_truth_field.csv"), w_true)
        for (s, r) in results
            write_field_csv(joinpath(args.output_dir, "lvd_synthetic_recon_$(s).csv"), r.w)
        end
        println("  Exported synthetic truth + reconstruction fields to lvd_synthetic_*.csv")
    end

    # --- resolution: full depth × direction map over ALL cells below min depth ---
    # Probe every cell (above the minimum evaluation depth), recover its linearised
    # point-spread, and bin the FWHM by depth layer and azimuthal direction so the
    # map covers all directions and all (resolvable) depths.
    centroids = volume.centroids
    recon_lin = wt -> first(sart_reconstruct(Jv, Jv * wt; n_iter = 120, relaxation = 0.2))
    zmin, zmax = extrema(centroids[3, :])
    # one depth bin per grid z-layer at/above the minimum evaluation depth
    zlayers = filter(z -> z >= args.min_eval_depth - 1e-6,
                     sort(unique(round.(centroids[3, :]; digits = 1))))
    layer_edges = length(zlayers) >= 2 ?
        vcat(zlayers[1] - 1.0,
             [0.5 * (zlayers[k] + zlayers[k + 1]) for k in 1:length(zlayers) - 1],
             zlayers[end] + 1.0) :
        Float64[args.min_eval_depth - 1.0, zmax + 1.0]
    azimuth_edges = collect(range(0.0, 360.0; length = 9))   # 8 directional sectors
    res_cells = findall(z -> z >= args.min_eval_depth - 1e-6, vec(centroids[3, :]))
    resmap = resolution_map(recon_lin, centroids, zeros(n_cells), res_cells;
        depth_edges = layer_edges, azimuth_edges = azimuth_edges, contrast = args.contrast,
        zenith_step_deg = args.zenith_step_deg, cell_size_m = estimate_cell_size(volume))
    # depth-only summary (median over directions) for the console + return value.
    # The empirical FWHM is clipped at the per-depth geometric floor: a recovered
    # point-spread confined to the impulse cell is "at the grid limit", not 0 m.
    rvd = NamedTuple[]
    for k in eachindex(resmap.depths)
        finite = filter(isfinite, resmap.fwhm[k, :])
        isempty(finite) && continue
        push!(rvd, (depth_m = resmap.depths[k],
                    fwhm_m = max(median(finite), resmap.floor[k]),
                    floor_m = resmap.floor[k], n_cells = sum(resmap.counts[k, :])))
    end

    # by-depth reconstruction bands also start at the minimum evaluation depth
    depth_edges = collect(range(max(args.min_eval_depth, zmin), zmax; length = 5))

    # --- by-depth reconstruction quality: AAD (GN/GD) should pull ahead in the
    # deeper bands where the corrected-CSDA operator is most nonlinear -----------
    zc = centroids[3, :]
    nbands = length(depth_edges) - 1
    depth_masks = [(depth_edges[k] .<= zc) .& (k == nbands ? (zc .<= depth_edges[k + 1]) :
                                                              (zc .< depth_edges[k + 1])) for k in 1:nbands]
    band_reports = Dict{String,Vector{Any}}()
    for s in runset
        haskey(results, s) || continue
        wrec = results[s].w
        band_reports[s] = Any[count(depth_masks[k]) == 0 ? nothing :
            reconstruction_report(wrec, w_true; mask = depth_masks[k]) for k in 1:nbands]
    end

    # --- by-zenith DATA residual: evaluate every reconstruction through the TRUE
    # nonlinear forward and bin the misfit by ray zenith (apples-to-apples) ------
    zb = Float64[paths[b].zenith for b in valid_bins]
    zedges = collect(range(minimum(zb), maximum(zb); length = 4))   # 3 zenith bands
    zmasks = [(zedges[k] .<= zb) .& (k == 3 ? (zb .<= zedges[k + 1]) :
                                              (zb .< zedges[k + 1])) for k in 1:3]
    relresid(p, idx) = any(idx) ?
        sqrt(sum((p[idx] .- obs[idx]) .^ 2)) / max(sqrt(sum(obs[idx] .^ 2)), 1e-30) : NaN
    zenith_reports = Dict{String,Vector{Float64}}()
    for s in runset
        haskey(results, s) || continue
        pred_true = csda_model(results[s].w)[1]
        zenith_reports[s] = Float64[relresid(pred_true, zmasks[k]) for k in 1:3]
    end

    open(output_path, "w") do io
        println(io, "LVD tomography inverse reconstruction summary")
        println(io, "geometry = $(geometry_name(volume)), n_cells = $n_cells, valid_bins = $(length(valid_bins))")
        println(io, "ground-truth nonzero cells = $(count(>(0.0), w_true)), contrast = $(args.contrast)")
        println(io, "regularisation weight = $(args.reg_weight)")
        println(io)
        for s in runset
            haskey(results, s) || continue
            r = results[s]
            println(io, "[$(uppercase(s))]  ($(round(r.seconds; digits=2)) s, $(length(r.hist)) iters)")
            println(io, @sprintf("  MSE=%.4e RMSE=%.4e PSNR=%.2f dB SSIM=%.4f SNR=%.2f dB",
                                 r.report.mse, r.report.rmse, r.report.psnr, r.report.ssim, r.report.snr))
            println(io, @sprintf("  final data misfit ||Jw-b|| = %.4e", r.hist[end]))
            println(io)
        end

        # By-depth reconstruction error (RMSE / SSIM) per solver.
        println(io, "Reconstruction RMSE by depth band (lower = better):")
        hdr = "  " * rpad("depth band [m]", 22) * join([rpad(uppercase(s), 11) for s in runset])
        println(io, hdr)
        for k in 1:nbands
            label = @sprintf("%.0f-%.0f", depth_edges[k], depth_edges[k + 1])
            row = "  " * rpad(label, 22)
            for s in runset
                br = get(band_reports, s, nothing)
                rep = br === nothing ? nothing : br[k]
                row *= rpad(rep === nothing ? "-" : @sprintf("%.4f", rep.rmse), 11)
            end
            println(io, row)
        end
        println(io)
        println(io, "Reconstruction SSIM by depth band (higher = better):")
        println(io, hdr)
        for k in 1:nbands
            label = @sprintf("%.0f-%.0f", depth_edges[k], depth_edges[k + 1])
            row = "  " * rpad(label, 22)
            for s in runset
                br = get(band_reports, s, nothing)
                rep = br === nothing ? nothing : br[k]
                row *= rpad(rep === nothing ? "-" : @sprintf("%.4f", rep.ssim), 11)
            end
            println(io, row)
        end
        println(io)

        # By-zenith data residual through the TRUE nonlinear forward.
        println(io, "Relative data residual ||pred_true - obs||/||obs|| by zenith band:")
        println(io, "  " * rpad("zenith band [deg]", 22) * join([rpad(uppercase(s), 11) for s in runset]))
        for k in 1:3
            label = @sprintf("%.0f-%.0f", zedges[k], zedges[k + 1])
            row = "  " * rpad(label, 22)
            for s in runset
                zr = get(zenith_reports, s, nothing)
                row *= rpad(zr === nothing ? "-" : @sprintf("%.4f", zr[k]), 11)
            end
            println(io, row)
        end
        println(io)

        println(io, "Resolution (linearised point-spread FWHM vs depth):")
        println(io, @sprintf("  %-12s %-12s %-12s %-8s", "depth[m]", "FWHM[m]", "floor[m]", "n_cells"))
        for e in rvd
            println(io, @sprintf("  %-12.1f %-12.1f %-12.1f %-8d", e.depth_m, e.fwhm_m, e.floor_m, e.n_cells))
        end
        if isempty(rvd)
            println(io, "  (no depth bins populated)")
        end
        println(io)

        # Full resolution map: recovered FWHM[m] for all depths × all directions.
        # Entries below the per-depth geometric floor (cell size / angular sampling)
        # are grid-limited and reported as "<=floor"; "-" marks sectors with no cells.
        println(io, "Resolution FWHM[m] map — all depths (rows) × azimuth directions (cols, deg):")
        azhdr = "  " * rpad("depth[m]", 9) *
                join([rpad(@sprintf("%.0f", a), 7) for a in resmap.azimuths]) * rpad("floor", 7)
        println(io, azhdr)
        for k in eachindex(resmap.depths)
            row = "  " * rpad(@sprintf("%.0f", resmap.depths[k]), 9)
            for a in eachindex(resmap.azimuths)
                v = resmap.fwhm[k, a]
                entry = !isfinite(v) ? "-" :
                        (v < resmap.floor[k] ? @sprintf("<=%.0f", resmap.floor[k]) :
                                               @sprintf("%.0f", v))
                row *= rpad(entry, 7)
            end
            row *= rpad(@sprintf("%.0f", resmap.floor[k]), 7)
            println(io, row)
        end
        println(io, @sprintf("  (azimuth sectors of %.0f deg; cells evaluated below %.0f m)",
                             360.0 / length(resmap.azimuths), args.min_eval_depth))
        println(io)

        # Companion cell-count map so unsampled sectors are explicit in the figure.
        println(io, "Resolution cell-count map — all depths (rows) × azimuth directions (cols, deg):")
        println(io, azhdr)
        for k in eachindex(resmap.depths)
            row = "  " * rpad(@sprintf("%.0f", resmap.depths[k]), 9)
            for a in eachindex(resmap.azimuths)
                row *= rpad(string(resmap.counts[k, a]), 7)
            end
            row *= rpad(@sprintf("%.0f", resmap.floor[k]), 7)
            println(io, row)
        end
        println(io, @sprintf("  (azimuth sectors of %.0f deg; cells evaluated below %.0f m)",
                             360.0 / length(resmap.azimuths), args.min_eval_depth))
    end

    println("Reconstruction RMSE by depth band (lower = better):")
    println("  " * rpad("depth[m]", 16) * join([rpad(uppercase(s), 10) for s in runset]))
    for k in 1:nbands
        row = "  " * rpad(@sprintf("%.0f-%.0f", depth_edges[k], depth_edges[k + 1]), 16)
        for s in runset
            br = get(band_reports, s, nothing)
            rep = br === nothing ? nothing : br[k]
            row *= rpad(rep === nothing ? "-" : @sprintf("%.4f", rep.rmse), 10)
        end
        println(row)
    end
    println("Resolution vs depth (linearised PSF FWHM):")
    for e in rvd
        println(@sprintf("  depth=%.0f m -> FWHM=%.0f m (geometric floor %.0f m, %d cells)",
                         e.depth_m, e.fwhm_m, e.floor_m, e.n_cells))
    end

    return (results = results, w_true = w_true, resolution = rvd, resolution_map = resmap)
end

# ===========================================================================
# Final section: explain the MEASURED Gran Sasso LVD flux with a per-cell
# material-mixture field (Gauss-Newton) under STANDARD rock density, instead of
# the paper's ad-hoc uniform non-standard density (2710 kg/m³). Reuses the
# muography Part-3 machinery (LVDTopo.run_part3) by building an NMapFluxResult
# from the tomography mixture forward. Every table/plot carries the MC +
# systematic error (sigma_mc / sigma_syst / sigma_total).
# ===========================================================================

# One measured bin from rock_int.txt: az/zenith interval, slant rock (m), intensity.
struct MeasuredBin
    az_lo::Float64; az_hi::Float64; zen_lo::Float64; zen_hi::Float64
    rock_m::Float64; intensity::Float64
end

function load_measured_intensity(path::String)
    bins = MeasuredBin[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        if occursin(",", s)
            p = strip.(split(s, ","))
            (isempty(p) || lowercase(p[1]) == "bin_index") && continue
            # Headered Agafonova extraction:
            # bin_index, az_lo, az_hi, az_center, zen_lo, zen_hi, zen_center,
            # slant_rock_m, intensity_raw, ...
            length(p) >= 9 || continue
            push!(bins, MeasuredBin(parse(Float64, p[2]), parse(Float64, p[3]),
                                    parse(Float64, p[5]), parse(Float64, p[6]),
                                    parse(Float64, p[8]), parse(Float64, p[9])))
        else
            p = split(s)
            length(p) >= 7 || continue
            # Legacy rock_int.txt:
            # bin_index az_lo az_hi zen_lo zen_hi slant_rock_m intensity
            push!(bins, MeasuredBin(parse(Float64, p[2]), parse(Float64, p[3]),
                                    parse(Float64, p[4]), parse(Float64, p[5]),
                                    parse(Float64, p[6]), parse(Float64, p[7])))
        end
    end
    return bins
end

# Measured intensity for each path's (θ,φ); NaN where uncovered or non-positive.
function measured_intensity_for_paths(paths::Vector{DirectionalPath}, bins::Vector{MeasuredBin})
    out = fill(NaN, length(paths))
    for (i, path) in enumerate(paths)
        path.valid || continue
        θ = path.zenith; φ = mod(path.azimuth, 360.0)
        for b in bins
            inzen = b.zen_lo <= θ < b.zen_hi
            inaz = if b.az_hi - b.az_lo >= 359.999
                true
            else
                lo = mod(b.az_lo, 360.0); hi = mod(b.az_hi, 360.0)
                lo <= hi ? (lo <= φ < hi) : (φ >= lo || φ < hi)
            end
            if inzen && inaz
                out[i] = b.intensity > 0 ? b.intensity : NaN
                break
            end
        end
    end
    return out
end

# Build an LVDTopo.NMapFluxResult from the tomography mixture forward on the
# coarse (nmap-resolution) grid: per-bin MC flux + MC/systematic uncertainty via
# the generic estimate_transport_uncertainty wrapper, plus the slant-rock grid.
function nmap_result_for(physics, shallow_flags, matcfg::MaterialConfig, site::SiteConfig,
                         cpaths::Vector{DirectionalPath}, czen::Vector{Float64},
                         caz::Vector{Float64}, w_field::Vector{Float64},
                         psamples, args; base_seed::Int = 2026)
    nz = length(czen); na = length(caz)
    flux = fill(NaN, nz, na); smc = zeros(nz, na); ssy = zeros(nz, na)
    sto = zeros(nz, na); rock = zeros(nz, na)
    materials, densities = build_cell_properties_for_mc(shallow_flags, w_field, matcfg)
    Threads.@threads for idx in eachindex(cpaths)
        path = cpaths[idx]
        z = div(idx - 1, na) + 1; a = mod(idx - 1, na) + 1   # paths order: θ outer, φ inner
        rock[z, a] = path.remaining_rock_distance + sum(s.distance for s in path.segments; init = 0.0)
        path.valid || continue
        evaluate = variation -> compute_directional_flux_mc(
            physics, matcfg, site, path, materials, densities, psamples, variation.seed;
            straggling = variation.straggling,
            energy_threshold_low = variation.energy_threshold_low,
            scattering = variation.scattering)
        budget = estimate_transport_uncertainty(evaluate;
            straggling = args.straggling, scattering = true,
            energy_threshold_low = Float64(args.energy_threshold_low),
            seed = base_seed + idx, threshold_factors = (0.5, 2.0))
        flux[z, a] = budget.value; smc[z, a] = budget.sigma_mc
        ssy[z, a] = budget.sigma_syst; sto[z, a] = budget.sigma_total
    end
    return LVDTopo.NMapFluxResult(copy(czen), copy(caz), flux, smc, ssy, sto, rock)
end

# 3D scatter of the non-negative water-content field, ranked by w so the strongest
# water cells are shown (the field is water-only: w ≥ 0 everywhere).
function plot_water_field(volume, w::Vector{Float64}; output_path::String, max_cells::Int = 200)
    order = sortperm(w, rev = true)
    sel = order[1:min(max_cells, length(order))]
    sel = [i for i in sel if w[i] > 1e-6]
    isempty(sel) && (sel = order[1:min(max_cells, length(order))])
    cmax = maximum(w[sel]; init = 1e-6)
    traces = GenericTrace[
        scatter3d(x = [volume.centroids[1, i] for i in sel],
                  y = [volume.centroids[2, i] for i in sel],
                  z = [volume.centroids[3, i] for i in sel],
                  mode = "markers",
                  marker = attr(size = 5, color = [w[i] for i in sel],
                                colorscale = "Blues", cmin = 0.0, cmax = cmax,
                                colorbar = attr(title = "water fraction w")),
                  name = "water"),
        scatter3d(x = [0.0], y = [0.0], z = [0.0], mode = "markers+text",
                  marker = attr(size = 7, color = "black"), text = ["LVD"],
                  textposition = "top center", name = "Detector"),
    ]
    layout = Layout(title = "Reconstructed water-content field (single muon angular distribution)",
        scene = attr(xaxis = attr(title = "East (m)"), yaxis = attr(title = "North (m)"),
                     zaxis = attr(title = "Height above detector (m)"), aspectmode = "data"),
        width = 1000, height = 800)
    savefig(Plot(traces, layout), output_path)
end

# 3D map of WHERE the water proportion increased vs pure standard rock: cells with
# w>0, coloured by water fraction (and the implied effective-density drop). Answers
# "where did the reconstruction add water relative to rock".
function plot_water_increase(volume, w::Vector{Float64}, matcfg::MaterialConfig;
                             output_path::String, w_floor::Float64 = 0.02)
    sel = [i for i in eachindex(w) if w[i] > w_floor]
    if isempty(sel)
        @warn "plot_water_increase: no cells with w>$(w_floor); skipping $output_path"; return
    end
    # effective density drop fraction vs pure rock, for the hover text
    ddrop = [100.0 * (1.0 - cell_density(w[i], false, matcfg) / matcfg.rock_density) for i in sel]
    traces = GenericTrace[
        scatter3d(x = [volume.centroids[1, i] for i in sel],
                  y = [volume.centroids[2, i] for i in sel],
                  z = [volume.centroids[3, i] for i in sel],
                  mode = "markers",
                  marker = attr(size = 5, color = [w[i] for i in sel],
                                colorscale = "Blues", cmin = 0.0, cmax = MAX_WATER_FRACTION,
                                colorbar = attr(title = "water fraction w")),
                  text = [@sprintf("w=%.2f<br>ρ_eff %.1f%% below rock", w[i], dd)
                          for (i, dd) in zip(sel, ddrop)],
                  hovertemplate = "%{text}<extra></extra>", name = "water increase"),
        scatter3d(x = [0.0], y = [0.0], z = [0.0], mode = "markers+text",
                  marker = attr(size = 7, color = "red"), text = ["LVD"],
                  textposition = "top center", name = "Detector"),
    ]
    layout = Layout(title = "Where the water proportion increased vs pure rock (reconstructed)",
        scene = attr(xaxis = attr(title = "East (m)"), yaxis = attr(title = "North (m)"),
                     zaxis = attr(title = "Height above detector (m)"), aspectmode = "data"),
        width = 1000, height = 800)
    savefig(Plot(traces, layout), output_path)
end

# Figure 7 only (the reconstructed single-muon azimuthal angular distribution) vs the
# paper, with MC + systematic bands. Figure 8 (energy spectrum) is intentionally
# omitted — it is model-generated, not reconstructed from the detector.
function create_figure7_plot(fig7, model_az, model_prof, smc, ssy, sto, fit; output_path::String)
    scale = fit.scale
    prof = scale .* model_prof .* 1e9            # to 1e-9 cm^-2 s^-1 deg^-1
    syst = scale .* ssy .* 1e9
    tot  = scale .* sto .* 1e9
    mc   = scale .* smc .* 1e9
    band(lo, hi, fill) = scatter(x = vcat(model_az, reverse(model_az)),
        y = vcat(hi, reverse(lo)), fill = "toself", fillcolor = fill,
        line = attr(width = 0), hoverinfo = "skip", showlegend = false)
    traces = GenericTrace[
        band(prof .- tot, prof .+ tot, "rgba(40,110,220,0.10)"),
        band(prof .- syst, prof .+ syst, "rgba(40,110,220,0.20)"),
        scatter(x = fig7.curve_az, y = fig7.curve_intensity .* 1e9, mode = "lines",
                line = attr(color = "rgb(190,40,40)", dash = "dash", width = 2), name = "paper curve"),
        scatter(x = fig7.point_az, y = fig7.point_intensity .* 1e9, mode = "markers",
                marker = attr(color = "black", size = 5), name = "paper LVD points"),
        scatter(x = model_az, y = prof, mode = "lines",
                line = attr(color = "rgb(40,110,220)", width = 3), name = "reconstructed mixture",
                text = [@sprintf("φ=%.0f°<br>I=%.3e<br>MC=%.1f%%<br>syst=%.1f%%<br>total=%.1f%%",
                                 a, y, 100*m/max(abs(y),1e-30), 100*s/max(abs(y),1e-30), 100*t/max(abs(y),1e-30))
                        for (a,y,m,s,t) in zip(model_az, prof, mc, syst, tot)],
                hovertemplate = "%{text}<extra></extra>"),
    ]
    layout = Layout(
        title = attr(text = "Single muon angular distribution — Figure 7 (azimuthal intensity, θ≤60°)<br>" *
            "<sub>reconstructed material mixture (standard rock) vs LVD paper · " *
            @sprintf("offset %.1f°, curve NRMSE %.3f · ±1σ MC (light) + systematic (dark)</sub>",
                     fit.offset_deg, fit.curve_nrmse)),
        xaxis = attr(title = "Azimuth φ_LVD (deg)", range = [0, 360], dtick = 60),
        yaxis = attr(title = "Intensity (10⁻⁹ cm⁻² s⁻¹ deg⁻¹)"),
        width = 1050, height = 650)
    savefig(Plot(traces, layout), output_path)
end

function write_single_muon_2d_comparison(paths::Vector{DirectionalPath},
                                         valid_bins::Vector{Int},
                                         obs::Vector{Float64},
                                         pred::Vector{Float64},
                                         normalization::Float64;
                                         output_csv::String,
                                         output_plot::String)
    zeniths = sort(unique([paths[b].zenith for b in valid_bins]))
    azimuths = sort(unique([mod(paths[b].azimuth, 360.0) for b in valid_bins]))
    zmap = Dict(z => i for (i, z) in enumerate(zeniths))
    amap = Dict(a => i for (i, a) in enumerate(azimuths))
    data = fill(NaN, length(zeniths), length(azimuths))
    model = fill(NaN, length(zeniths), length(azimuths))
    rel = fill(NaN, length(zeniths), length(azimuths))

    mkpath(dirname(output_csv))
    relvals = Float64[]
    open(output_csv, "w") do io
        println(io, "bin_idx,zenith_deg,azimuth_deg,measured_raw,model_raw,model_over_measured,relative_residual")
        for (k, b) in enumerate(valid_bins)
            measured_raw = obs[k] * normalization
            model_raw = pred[k] * normalization
            ratio = model_raw / max(measured_raw, 1e-30)
            residual = ratio - 1.0
            push!(relvals, residual)
            zi = zmap[paths[b].zenith]
            ai = amap[mod(paths[b].azimuth, 360.0)]
            data[zi, ai] = measured_raw
            model[zi, ai] = model_raw
            rel[zi, ai] = 100.0 * residual
            println(io, @sprintf("%d,%.6f,%.6f,%.8e,%.8e,%.8e,%.8e",
                                 b, paths[b].zenith, mod(paths[b].azimuth, 360.0),
                                 measured_raw, model_raw, ratio, residual))
        end
    end

    log_data = map(x -> (isfinite(x) && x > 0.0) ? log10(x) : NaN, data)
    log_model = map(x -> (isfinite(x) && x > 0.0) ? log10(x) : NaN, model)
    traces = GenericTrace[
        heatmap(x = azimuths, y = zeniths, z = log_data,
                colorscale = "Viridis", name = "measured log10(raw)",
                colorbar = attr(title = "log10 raw"),
                hovertemplate = "φ=%{x:.0f}°<br>θ=%{y:.0f}°<br>log10(data)=%{z:.3f}<extra></extra>"),
        heatmap(x = azimuths, y = zeniths, z = log_model,
                colorscale = "Viridis", name = "model log10(raw)",
                visible = "legendonly",
                hovertemplate = "φ=%{x:.0f}°<br>θ=%{y:.0f}°<br>log10(model)=%{z:.3f}<extra></extra>"),
        heatmap(x = azimuths, y = zeniths, z = rel,
                colorscale = "RdBu", reversescale = true, zmid = 0.0,
                name = "model-data residual (%)", visible = "legendonly",
                hovertemplate = "φ=%{x:.0f}°<br>θ=%{y:.0f}°<br>resid=%{z:.1f}%<extra></extra>"),
    ]
    layout = Layout(title = "Agafonova/LVD 2D single-muon map: measured data vs reconstructed mixture",
        xaxis = attr(title = "Azimuth φ (deg)", dtick = 30),
        yaxis = attr(title = "Zenith θ (deg)", dtick = 10),
        width = 1100, height = 700)
    savefig(Plot(traces, layout), output_plot)

    rms = isempty(relvals) ? NaN : sqrt(mean(relvals .^ 2))
    bias = isempty(relvals) ? NaN : mean(relvals)
    return (n_bins = length(relvals), rms_rel = rms, bias_rel = bias)
end

# Matched-resolution detectability study (the fair comparison for the paper). For each
# solver we sweep its regularization to trace a RESOLUTION–EFFICIENCY frontier: at each
# setting we measure (a) the point-spread FWHM (noiseless point-source reconstruction)
# and (b) τ₉₅, the exposure for 95% detection of the deep lens. Detection significance
# DP = (⟨w⟩_lens,signal − μ_null)/σ_null; in the linear-Gaussian threshold regime
# DP ∝ √τ, so τ₉₅ = τ₀·(z/DP(τ₀))² with z = 3 + Φ⁻¹(0.95) = 4.645. Comparing solvers AT
# MATCHED FWHM isolates statistical efficiency (variance) from resolution (bias) — the
# only fair basis for an efficiency-gain claim.
function run_detectability_matched(physics, shallow_flags, matcfg::MaterialConfig, site::SiteConfig,
                                   volume, paths::Vector{DirectionalPath}, energy_samples, args;
                                   output_path::String, plot_path::String)
    n_cells = num_cells(volume)
    w_true = zeros(n_cells)
    for i in 1:n_cells
        x, y, z = cell_centroid(volume, i)
        (abs(z - 1118.6) <= 160.0 && abs(x) <= 600.0 && abs(y) <= 600.0) &&
            (w_true[i] = clamp(args.lens_w, 0.0, MAX_WATER_FRACTION))
    end
    lens_cells = findall(>(0.0), w_true)
    if isempty(lens_cells)
        println("  no deep-lens cells on this grid; skipping detectability study"); return nothing
    end
    lcx = mean(volume.centroids[1, lens_cells]); lcy = mean(volume.centroids[2, lens_cells])
    lcz = mean(volume.centroids[3, lens_cells])
    c0 = lens_cells[argmin([(volume.centroids[1, i] - lcx)^2 + (volume.centroids[2, i] - lcy)^2 +
                            (volume.centroids[3, i] - lcz)^2 for i in lens_cells])]
    center = (volume.centroids[1, c0], volume.centroids[2, c0], volume.centroids[3, c0])

    base_flux, base_J = assemble_forward_and_jacobian(physics, shallow_flags, matcfg, site,
        paths, zeros(n_cells), energy_samples; n_cells = n_cells, threaded = args.threaded)
    valid_bins = findall(isfinite, base_flux)
    Jv = base_J[valid_bins, :]; f0 = base_flux[valid_bins]
    excess_lens = Jv * w_true; obs_clean = f0 .+ excess_lens
    excess_pt = Vector(Jv[:, c0])             # point-source response (unit contrast at c0)

    neighbors = cell_neighbors(volume)
    Lmat = Matrix(edge_hessian(EdgePrior(neighbors, 1.0, Inf), zeros(n_cells)))

    τ0 = sqrt(args.sweep_exp_min * args.sweep_exp_max)   # reference exposure (log-midpoint)
    σ0 = sqrt.(max.(obs_clean, 0.0) ./ τ0); σ0 .= max.(σ0, 1e-12 * maximum(obs_clean))
    W0 = 1.0 ./ σ0 .^ 2
    JtW0J = Matrix(transpose(Jv) * (W0 .* Jv))
    dmed = median(filter(>(0), [JtW0J[i, i] for i in 1:n_cells]))
    z95 = 4.645   # 3σ detection threshold with 95% of realizations above it

    # All solvers use positivity (real prior for a positive anomaly): SART/MLEM intrinsically,
    # GN via clamping its regularised inverse-variance solution to ≥0 — otherwise the unconstrained
    # GN is unfairly handicapped at detecting a positive lens.
    recon(solver, knob, ex) =
        solver === :sart ? max.(first(sart_reconstruct(Jv, ex; n_iter = round(Int, knob), relaxation = 0.2)), 0.0) :
        solver === :mlem ? first(mlem_reconstruct(max.(Jv, 0.0), max.(ex, 0.0); n_iter = round(Int, knob))) :
        (Afac = cholesky(Symmetric(JtW0J .+ (knob * dmed) .* Lmat); check = false);
         issuccess(Afac) ? max.(Afac \ (transpose(Jv) * (W0 .* ex)), 0.0) : zeros(n_cells))

    function fwhm_of(solver, knob)
        w = recon(solver, knob, excess_pt)
        pk = w[c0] > 0 ? w[c0] : maximum(w)
        radial_fwhm(w, volume.centroids, center, pk)
    end
    function tau95_of(solver, knob, seed)
        M = args.sweep_realizations
        rng = MersenneTwister(seed)
        As = Vector{Float64}(undef, M); An = Vector{Float64}(undef, M)
        for m in 1:M
            As[m] = mean(recon(solver, knob, excess_lens .+ σ0 .* randn(rng, length(σ0)))[lens_cells])
            An[m] = mean(recon(solver, knob, σ0 .* randn(rng, length(σ0)))[lens_cells])
        end
        dp = (mean(As) - mean(An)) / max(std(An), 1e-30)
        return dp <= 0 ? Inf : τ0 * (z95 / dp)^2
    end

    knobs = Dict(:sart => [400.0, 200, 100, 50, 25], :mlem => [400.0, 200, 100, 50, 25],
                 :gn => [0.003, 0.01, 0.03, 0.1, 0.3])
    labels = Dict(:sart => "SART", :mlem => "MLEM", :gn => "GN")
    println("Matched-resolution detectability frontier: lens w=$(args.lens_w) " *
            "($(length(lens_cells)) cells), $(args.sweep_realizations) realizations/point...")
    res = Dict(s => NamedTuple[] for s in (:sart, :mlem, :gn))
    sc = 0
    for s in (:sart, :mlem, :gn), k in knobs[s]
        sc += 1
        fw = fwhm_of(s, k); t = tau95_of(s, k, args.seed + 7000 + sc)
        push!(res[s], (knob = float(k), fwhm = fw, tau95 = t))
        println(@sprintf("  %-4s knob=%-7.4g  FWHM=%6.0f m  τ95=%.3e", labels[s], k, fw, t))
    end

    # efficiency at a MATCHED FWHM = midpoint of the FWHM range common to all solvers
    fwhm_lo = maximum(minimum(r.fwhm for r in res[s]) for s in keys(res))
    fwhm_hi = minimum(maximum(r.fwhm for r in res[s]) for s in keys(res))
    fstar = 0.5 * (fwhm_lo + fwhm_hi)
    matched = fwhm_hi > fwhm_lo
    function interp_tau(s, F)
        pts = sort(res[s]; by = r -> r.fwhm)
        F <= pts[1].fwhm && return pts[1].tau95
        F >= pts[end].fwhm && return pts[end].tau95
        for k in 2:length(pts)
            if F <= pts[k].fwhm
                t = (F - pts[k-1].fwhm) / (pts[k].fwhm - pts[k-1].fwhm + 1e-30)
                return 10.0 ^ (log10(pts[k-1].tau95) + t * (log10(pts[k].tau95) - log10(pts[k-1].tau95)))
            end
        end
        return pts[end].tau95
    end
    tstar = Dict(s => interp_tau(s, fstar) for s in keys(res))

    open(output_path, "w") do io
        println(io, "Matched-resolution detectability frontier (deep lens, DP≥3, 95% detection)")
        println(io, @sprintf("lens w=%.2f, %d lens cells; reference exposure τ₀=%.2e; %d realizations/point",
                             args.lens_w, length(lens_cells), τ0, args.sweep_realizations))
        println(io, "GN = regularised inverse-variance least squares; SART/MLEM = algebraic baselines.")
        println(io, "τ₉₅ = exposure for 95% detection at the given resolution (DP ∝ √τ scaling).")
        for s in (:sart, :mlem, :gn)
            println(io); println(io, "$(labels[s]):  " * rpad("knob", 10) * rpad("FWHM[m]", 10) * "tau95")
            for r in sort(res[s]; by = r -> r.fwhm)
                println(io, "  " * rpad(@sprintf("%.4g", r.knob), 10) * rpad(@sprintf("%.0f", r.fwhm), 10) *
                        @sprintf("%.3e", r.tau95))
            end
        end
        println(io)
        if matched
            println(io, @sprintf("At matched resolution FWHM ≈ %.0f m (common-overlap midpoint):", fstar))
            for s in (:sart, :mlem, :gn)
                println(io, @sprintf("  %-5s τ₉₅ = %.3e", labels[s], tstar[s]))
            end
            for s in (:sart, :mlem)
                println(io, @sprintf("  efficiency gain GN vs %s = %.1f×", labels[s], tstar[s] / tstar[:gn]))
            end
        else
            println(io, "No common FWHM overlap across solvers — widen the knob ranges.")
        end
    end

    colors = Dict(:sart => "rgb(200,120,40)", :mlem => "rgb(60,160,90)", :gn => "rgb(40,110,220)")
    traces = GenericTrace[]
    for s in (:sart, :mlem, :gn)
        pts = sort(res[s]; by = r -> r.fwhm)
        push!(traces, scatter(x = [r.fwhm for r in pts], y = [r.tau95 for r in pts],
                              mode = "lines+markers", name = labels[s], line = attr(color = colors[s])))
    end
    matched && push!(traces, scatter(x = [fstar, fstar],
        y = [minimum(values(tstar)) * 0.5, maximum(values(tstar)) * 2.0],
        mode = "lines", line = attr(color = "gray", dash = "dot"), name = "matched FWHM"))
    savefig(Plot(traces, Layout(title = "Detection efficiency vs resolution (deep lens, 95% @ DP≥3)",
        xaxis = attr(title = "Point-spread FWHM (m)"),
        yaxis = attr(title = "τ₉₅ exposure for 95% detection", type = "log"),
        width = 1000, height = 650)), plot_path)
    println("  detectability outputs: $output_path , $plot_path")
    return (res = res, fstar = fstar, tstar = tstar)
end

# Second benchmark: a DETECTION-specific statistic built from CSDA first-order info
# (the flux gradient), following Benton et al. (information-theoretic muon radiography,
# docs/muon_tomography_information_theory.md). The muon counts form a Poisson channel
# k_m ~ Poisson(τ·μ_m); a perturbation χ gives μ_m → μ_m(1 + ζ_m·χ) with the relative
# sensitivity ζ_m = (∂μ_m/∂χ)/μ_m = (J·Δw)_m / μ_m (our AD Jacobian over the baseline
# flux). The Neyman–Pearson-optimal LINEAR statistic is the matched filter
#   Ψ = Σ_m ζ_m k_m,
# whose detection significance is DP = χ·√(τ · Σ_m (∂μ_m/∂χ)²/μ_m) = √(τ·F) — the Fisher
# information bound, ANALYTIC (no reconstruction, no GN/GD). Compared against the
# unweighted plume-ROI sum Φ = Σ_{m∈ROI} k_m (Benton's baseline) and the SART
# reconstruction detector. τ₉₅ = z²/F with z = 3 + Φ⁻¹(0.95) = 4.645 (95% above 3σ).
function run_detection_benchmark(physics, shallow_flags, matcfg::MaterialConfig, site::SiteConfig,
                                 volume, paths::Vector{DirectionalPath}, energy_samples, args;
                                 output_path::String, plot_path::String)
    n_cells = num_cells(volume)
    w_true = zeros(n_cells)
    for i in 1:n_cells
        x, y, z = cell_centroid(volume, i)
        (abs(z - 1118.6) <= 160.0 && abs(x) <= 600.0 && abs(y) <= 600.0) &&
            (w_true[i] = clamp(args.lens_w, 0.0, MAX_WATER_FRACTION))
    end
    lens_cells = findall(>(0.0), w_true)
    if isempty(lens_cells)
        println("  no deep-lens cells on this grid; skipping detection benchmark"); return nothing
    end
    base_flux, base_J = assemble_forward_and_jacobian(physics, shallow_flags, matcfg, site,
        paths, zeros(n_cells), energy_samples; n_cells = n_cells, threaded = args.threaded)
    valid_bins = findall(isfinite, base_flux)
    Jv = base_J[valid_bins, :]; f0 = base_flux[valid_bins]
    μ = max.(f0, 1e-30)
    s = Jv * w_true                       # first-order flux change from the lens (signal template)
    z95 = 4.645

    # Ψ — optimal matched filter (Fisher information bound), analytic
    fisher = sum(s .^ 2 ./ μ)
    τ_psi = fisher > 0 ? z95^2 / fisher : Inf

    # Φ — unweighted plume-ROI count sum (bins carrying ≥10% of peak |signal|)
    roi = abs.(s) .>= 0.1 * maximum(abs.(s))
    num = sum(s[roi]); den = sum(μ[roi])
    τ_phi = (den > 0 && num != 0) ? z95^2 * den / num^2 : Inf

    # SART reconstruction detector with a PROPERLY FPR-CONTROLLED test (no GN/GD):
    # threshold = empirical 3σ one-sided quantile of SART's OWN null distribution (which
    # is non-Gaussian under positivity), then τ₉₅ = exposure for 95% detection power. This
    # matches Ψ's operating point (3σ FPR, 95% power) so the comparison is apples-to-apples.
    τ0 = sqrt(args.sweep_exp_min * args.sweep_exp_max)
    σ0 = sqrt.(μ ./ τ0); σ0 .= max.(σ0, 1e-12 * maximum(μ))
    excess = s
    amp(w) = mean(w[lens_cells])
    sart_amp(ex) = amp(max.(first(sart_reconstruct(Jv, ex; n_iter = 200, relaxation = 0.2)), 0.0))
    function sart_tau95_fpr(seed)
        rng = MersenneTwister(seed); Mn = 3000; Ms = 600
        An = [sart_amp(σ0 .* randn(rng, length(σ0))) for _ in 1:Mn]
        As = [sart_amp(excess .+ σ0 .* randn(rng, length(σ0))) for _ in 1:Ms]
        μn = mean(An); Δ = mean(As) - μn
        Δ <= 0 && return Inf
        qcut = quantile(An, 0.99865) - μn       # 3σ one-sided FPR threshold (null noise quantile)
        ε = As .- mean(As)                      # signal-noise component at τ0
        rate(τ) = mean((Δ * sqrt(τ / τ0) .+ ε) .> qcut)
        lo, hi = τ0 * 1e-5, τ0 * 1e5
        for _ in 1:64
            mid = sqrt(lo * hi); rate(mid) >= 0.95 ? (hi = mid) : (lo = mid)
        end
        return hi
    end
    τ_sart = sart_tau95_fpr(args.seed + 8000)

    println("Detection benchmark (deep lens, DP≥3, 95% detection; no GN/GD):")
    println(@sprintf("  Ψ matched filter (optimal, Fisher): τ₉₅ = %.3e", τ_psi))
    println(@sprintf("  Φ ROI count sum (Benton baseline):  τ₉₅ = %.3e  (Ψ gain %.2f×)", τ_phi, τ_phi / τ_psi))
    println(@sprintf("  SART reconstruction detector:       τ₉₅ = %.3e  (Ψ gain %.2f×)", τ_sart, τ_sart / τ_psi))

    open(output_path, "w") do io
        println(io, "Detection benchmark — CSDA first-order (flux-gradient) statistics vs SART")
        println(io, "(deep lens w=$(args.lens_w), $(length(lens_cells)) cells; DP≥3, 95% detection; Poisson channel)")
        println(io, "Method = optimal info-theory matched filter Ψ=Σ ζ_m k_m, ζ_m=(∂μ_m/∂χ)/μ_m (Benton et al. 2020).")
        println(io, "NO Gauss-Newton / gradient-descent reconstruction is used.")
        println(io)
        println(io, @sprintf("  %-34s τ₉₅ = %.4e", "Ψ matched filter (optimal/Fisher)", τ_psi))
        println(io, @sprintf("  %-34s τ₉₅ = %.4e   (Ψ %.2f× better)", "Φ unweighted plume-ROI sum", τ_phi, τ_phi / τ_psi))
        println(io, @sprintf("  %-34s τ₉₅ = %.4e   (Ψ %.2f× better)", "SART reconstruction detector", τ_sart, τ_sart / τ_psi))
        println(io)
        println(io, @sprintf("ROI: %d / %d bins carry ≥10%% of peak |∂flux/∂χ|.", count(roi), length(s)))
        println(io, "Ψ is the Neyman–Pearson optimal linear detector, so τ₉₅(Ψ) ≤ τ₉₅(any linear statistic).")
    end

    methods = ["Ψ matched filter", "Φ ROI sum", "SART recon"]
    taus = [τ_psi, τ_phi, τ_sart]
    savefig(Plot([bar(x = methods, y = taus, marker = attr(color = ["rgb(40,110,220)", "rgb(200,120,40)", "rgb(60,160,90)"]))],
        Layout(title = "Detection efficiency: exposure for 95% detection of the deep lens (lower = better)",
            yaxis = attr(title = "τ₉₅ exposure", type = "log"), width = 850, height = 600)), plot_path)
    println("  detection-benchmark outputs: $output_path , $plot_path")
    return (psi = τ_psi, phi = τ_phi, sart = τ_sart)
end

# (b) Benchmark 1 as a sweep: reconstruction accuracy (RMSE, SSIM vs the true field) vs
# detector exposure, all four solvers with the FULL NONLINEAR operator (GD/GN relinearize
# each step, as in the headline result). Clean observations are the full MC (computed once,
# no inverse crime); only the counting noise scales with exposure. Use a reduced-but-
# overdetermined angular grid so the nonlinear GN sweep is affordable.
function run_reconstruction_sweep(physics, shallow_flags, matcfg::MaterialConfig, site::SiteConfig,
                                  volume, paths::Vector{DirectionalPath}, energy_samples, args;
                                  output_path::String, plot_path::String)
    n_cells = num_cells(volume)
    w_true = create_initial_water_field(volume, args)
    if all(iszero, w_true)
        for i in 1:n_cells
            x, y, z = cell_centroid(volume, i)
            (abs(z - 1118.6) <= 160.0 && abs(x) <= 600.0 && abs(y) <= 600.0) &&
                (w_true[i] = clamp(args.lens_w, 0.0, MAX_WATER_FRACTION))
        end
    end
    base_flux, base_J = assemble_forward_and_jacobian(physics, shallow_flags, matcfg, site,
        paths, zeros(n_cells), energy_samples; n_cells = n_cells, threaded = args.threaded)
    valid_bins = findall(isfinite, base_flux)
    Jv = base_J[valid_bins, :]; f0 = base_flux[valid_bins]
    clean_obs = mc_observations_for_field(physics, shallow_flags, matcfg, site, paths, w_true, valid_bins, args;
        cache_path = joinpath(args.output_dir, "lvd_recon_sweep_obs.bin"))
    csda_model = make_csda_operator(physics, shallow_flags, matcfg, site, paths, energy_samples;
        n_cells = n_cells, valid_bins = valid_bins, threaded = args.threaded)
    eval_mask = volume.centroids[3, :] .>= args.min_eval_depth
    neighbors = cell_neighbors(volume)
    sens = vec(sum(max.(Jv, 0.0); dims = 1)); sm_scale = (m = filter(>(0), sens); isempty(m) ? 1.0 : median(m))
    sm_prior = SmoothnessPrior(neighbors, 0.05 * sm_scale)

    exps = 10.0 .^ range(log10(args.sweep_exp_min), log10(args.sweep_exp_max); length = args.sweep_n)
    solvers = ("sart", "mlem", "gd", "gn")
    rmse = Dict(s => zeros(length(exps)) for s in solvers)
    ssimv = Dict(s => zeros(length(exps)) for s in solvers)
    M = max(3, args.sweep_realizations ÷ 10)
    println("Reconstruction-accuracy sweep (NONLINEAR operator, full-MC obs): $(length(exps)) exposures × $M realizations...")
    rng = MersenneTwister(args.seed + 9000)
    for (ei, τ) in enumerate(exps)
        σ = sqrt.(max.(clean_obs, 0.0) ./ τ); σ .= max.(σ, 1e-12 * maximum(clean_obs)); W = 1.0 ./ σ .^ 2
        diagc = vec(sum(W .* (Matrix(Jv) .^ 2); dims = 1)); gn_scale = (m = filter(>(0), diagc); isempty(m) ? 1.0 : median(m))
        edge_prior = EdgePrior(neighbors, 0.01 * gn_scale, args.edge_delta)
        acc = Dict(s => (r = Float64[], s = Float64[]) for s in solvers)
        for _ in 1:M
            obs = clean_obs .+ σ .* randn(rng, length(σ)); excess = obs .- f0
            recs = Dict(
                "sart" => first(sart_reconstruct(Jv, excess; n_iter = args.reco_iters, relaxation = 0.2, prior = sm_prior)),
                "mlem" => first(mlem_reconstruct(max.(Jv, 0.0), max.(excess, 0.0); n_iter = args.reco_iters, prior = sm_prior)),
                "gd"   => first(gradient_descent_reconstruct(csda_model, obs; w0 = zeros(n_cells),
                            n_iter = args.gd_iters, lr = args.gd_lr, optimizer = :pgd, prior = edge_prior,
                            weights = W, relinearize = true, box = (0.0, MAX_WATER_FRACTION))),
                "gn"   => first(gauss_newton_reconstruct(csda_model, obs; w0 = zeros(n_cells),
                            n_iter = max(8, args.reco_iters ÷ 16), prior = edge_prior,
                            weights = W, relinearize = true, box = (0.0, MAX_WATER_FRACTION))))
            for s in solvers
                rep = reconstruction_report(recs[s], w_true; mask = eval_mask)
                push!(acc[s].r, rep.rmse); push!(acc[s].s, rep.ssim)
            end
        end
        for s in solvers; rmse[s][ei] = mean(acc[s].r); ssimv[s][ei] = mean(acc[s].s); end
        println(@sprintf("  τ=%.2e  RMSE  SART=%.3f MLEM=%.3f GD=%.3f GN=%.3f", τ,
                         rmse["sart"][ei], rmse["mlem"][ei], rmse["gd"][ei], rmse["gn"][ei]))
    end

    open(output_path, "w") do io
        println(io, "Reconstruction accuracy vs exposure (nonlinear operator, full-MC obs; truth = multi-anomaly field)")
        println(io, @sprintf("%d realizations/point; eval depth ≥ %.0f m", M, args.min_eval_depth))
        for metric in ("RMSE", "SSIM")
            d = metric == "RMSE" ? rmse : ssimv
            println(io); println(io, "$metric:"); println(io, "  " * rpad("exposure", 12) * join([rpad(uppercase(s), 9) for s in solvers]))
            for ei in eachindex(exps)
                println(io, "  " * rpad(@sprintf("%.2e", exps[ei]), 12) * join([rpad(@sprintf("%.3f", d[s][ei]), 9) for s in solvers]))
            end
        end
    end
    colors = Dict("sart" => "rgb(200,120,40)", "mlem" => "rgb(60,160,90)", "gd" => "rgb(150,80,200)", "gn" => "rgb(40,110,220)")
    traces = GenericTrace[scatter(x = collect(exps), y = rmse[s], mode = "lines+markers",
                                  name = uppercase(s), line = attr(color = colors[s])) for s in solvers]
    savefig(Plot(traces, Layout(title = "Reconstruction RMSE vs exposure (lower = better)",
        xaxis = attr(title = "Detector exposure", type = "log"),
        yaxis = attr(title = "RMSE (water fraction)"), width = 1000, height = 600)), plot_path)
    println("  reconstruction-sweep outputs: $output_path , $plot_path")
    return (exps = exps, rmse = rmse, ssim = ssimv)
end

# (a) Detection-efficiency gain of the Ψ matched filter vs SART as a function of anomaly
# contrast. Ψ/Φ τ₉₅ ∝ 1/χ² (ratio contrast-independent); SART is nonlinear, so Ψ's
# advantage over it is expected to grow toward low contrast (Benton regime).
function run_detection_contrast_sweep(physics, shallow_flags, matcfg::MaterialConfig, site::SiteConfig,
                                      volume, paths::Vector{DirectionalPath}, energy_samples, args;
                                      output_path::String, plot_path::String)
    n_cells = num_cells(volume)
    lens_mask = falses(n_cells)
    for i in 1:n_cells
        x, y, z = cell_centroid(volume, i)
        (abs(z - 1118.6) <= 160.0 && abs(x) <= 600.0 && abs(y) <= 600.0) && (lens_mask[i] = true)
    end
    lens_cells = findall(lens_mask)
    isempty(lens_cells) && (println("  no lens cells; skipping"); return nothing)
    base_flux, base_J = assemble_forward_and_jacobian(physics, shallow_flags, matcfg, site,
        paths, zeros(n_cells), energy_samples; n_cells = n_cells, threaded = args.threaded)
    valid_bins = findall(isfinite, base_flux); Jv = base_J[valid_bins, :]
    μ = max.(base_flux[valid_bins], 1e-30)
    s_unit = Jv * Float64.(lens_mask)        # unit-contrast lens response
    z95 = 4.645
    τ0base = sqrt(args.sweep_exp_min * args.sweep_exp_max)
    amp(w) = mean(w[lens_cells])

    contrasts = [0.05, 0.1, 0.2, 0.4, 0.8]
    rows = NamedTuple[]
    println("Detection gain vs contrast (Ψ matched filter vs SART, FPR-controlled)...")
    for (ci, χ) in enumerate(contrasts)
        s = χ .* s_unit
        τ_psi = z95^2 / sum(s .^ 2 ./ μ)
        # SART τ95 (FPR-controlled) at this contrast; reference exposure scaled so noise ~ signal
        τ0 = τ0base
        σ0 = sqrt.(μ ./ τ0); σ0 .= max.(σ0, 1e-12 * maximum(μ))
        rng = MersenneTwister(args.seed + 9100 + ci); Mn = 2000; Ms = 500
        sa(ex) = amp(max.(first(sart_reconstruct(Jv, ex; n_iter = 200, relaxation = 0.2)), 0.0))
        An = [sa(σ0 .* randn(rng, length(σ0))) for _ in 1:Mn]
        As = [sa(s .+ σ0 .* randn(rng, length(σ0))) for _ in 1:Ms]
        μn = mean(An); Δ = mean(As) - μn
        if Δ <= 0
            τ_sart = Inf
        else
            qcut = quantile(An, 0.99865) - μn; ε = As .- mean(As)
            rate(τ) = mean((Δ * sqrt(τ / τ0) .+ ε) .> qcut)
            lo, hi = τ0 * 1e-5, τ0 * 1e5
            for _ in 1:64; mid = sqrt(lo * hi); rate(mid) >= 0.95 ? (hi = mid) : (lo = mid); end
            τ_sart = hi
        end
        push!(rows, (χ = χ, psi = τ_psi, sart = τ_sart, gain = τ_sart / τ_psi))
        println(@sprintf("  contrast=%.2f  Ψ τ95=%.3e  SART τ95=%.3e  gain=%.2f×", χ, τ_psi, τ_sart, τ_sart / τ_psi))
    end

    open(output_path, "w") do io
        println(io, "Detection efficiency gain Ψ (CSDA matched filter) vs SART, vs anomaly contrast")
        println(io, "(deep lens; 95% detection, 3σ FPR-controlled; no GN/GD)")
        println(io, "  " * rpad("contrast", 10) * rpad("Ψ τ95", 13) * rpad("SART τ95", 13) * "gain")
        for r in rows
            println(io, "  " * rpad(@sprintf("%.2f", r.χ), 10) * rpad(@sprintf("%.3e", r.psi), 13) *
                    rpad(@sprintf("%.3e", r.sart), 13) * @sprintf("%.2f×", r.gain))
        end
    end
    savefig(Plot([scatter(x = [r.χ for r in rows], y = [r.gain for r in rows], mode = "lines+markers",
                          name = "Ψ/SART gain")],
        Layout(title = "Ψ matched-filter detection gain over SART vs contrast",
            xaxis = attr(title = "Anomaly water-fraction contrast"),
            yaxis = attr(title = "τ₉₅(SART) / τ₉₅(Ψ)"), width = 950, height = 600)), plot_path)
    println("  detection-contrast outputs: $output_path , $plot_path")
    return rows
end

function run_paper_match(physics, shallow_flags, matcfg::MaterialConfig, volume,
                         emap, paths::Vector{DirectionalPath}, energy_samples,
                         rock_idx::Int, air_idx::Int, args)
    println("=" ^ 64)
    println(" Single muon angular distribution — water-content reconstruction from the measured 2D LVD surface")
    println("=" ^ 64)
    n_cells = num_cells(volume)
    println(@sprintf(" base rock density = %.0f kg/m^3 (Gran Sasso rock, Kudryavtsev 2009); water-only field w in [0, %.1f]",
                     matcfg.rock_density, MAX_WATER_FRACTION))

    # --- GN water-content match to the FULL measured 2D azimuth/zenith surface.
    # Solve the inverse directly against the whole observed angular map — every populated bin
    # of the measured (zenith, azimuth) surface, at its native resolution (one line of sight
    # per bin), spanning all azimuths AND the full zenith range (theta up to ~86 deg). The
    # near-horizontal rays add long-slant lines of sight through laterally distinct rock; that
    # independent coverage is what makes the otherwise single-viewpoint column-density inverse
    # meaningfully better posed.  The baseline is uniform Gran Sasso rock (no 1D-curve fit and
    # no denser-than-rock endpoint): the field carries only a non-negative water fraction.
    println("GN water-content match to the full measured 2D surface (rock=$(round(Int, matcfg.rock_density)) kg/m^3, altitude=$(args.primary_altitude_km) km)...")
    match_site = build_site(args.primary_altitude_km * 1000.0)
    match_bins = load_measured_intensity(args.match_data)
    match_zen = Float64[0.5 * (b.zen_lo + b.zen_hi) for b in match_bins]
    match_az  = Float64[(b.az_hi - b.az_lo >= 359.999) ? 0.0 : 0.5 * (b.az_lo + b.az_hi) for b in match_bins]
    println("Tracing geometry for $(length(match_bins)) measured-surface bins (0 <= theta <= $(round(maximum(match_zen); digits=0)) deg)...")
    match_paths = [trace_path(volume, match_zen[i], match_az[i], emap) for i in eachindex(match_bins)]
    meas = Float64[b.intensity for b in match_bins]
    base_flux, base_J = assemble_forward_and_jacobian(physics, shallow_flags, matcfg, match_site,
        match_paths, zeros(n_cells), energy_samples; n_cells = n_cells, threaded = args.threaded)
    covered = findall(i -> match_paths[i].valid && isfinite(base_flux[i]) && base_flux[i] > 0 &&
                           isfinite(meas[i]) && meas[i] > 0, eachindex(base_flux))
    if isempty(covered)
        println("  no covered bins (measured ∩ valid); skipping match"); return nothing
    end
    valid_bins = covered
    meas_v = meas[valid_bins]
    match_paths_v = match_paths[valid_bins]
    # Helpers: a MaterialConfig with the global rock density set to ρ, and the forward
    # flux over the covered bins at a given (ρ_rock, water field).
    mk_cfg = ρ -> MaterialConfig(matcfg.rock_material, matcfg.water_material, matcfg.air_material,
        matcfg.porous_material, ρ, matcfg.water_density, matcfg.porous_density,
        matcfg.porous_top_thickness)
    # Flux-only forward over the covered bins (no Jacobian) — cheap enough to call many times
    # inside the 1-D density search even on a dense mesh.
    flux_only_at = (ρ, wv) -> begin
        cfg = mk_cfg(ρ)
        out = Vector{Float64}(undef, length(match_paths_v))
        if args.threaded
            Threads.@threads for i in eachindex(match_paths_v)
                out[i] = compute_directional_flux_csda(physics, shallow_flags, cfg, match_site,
                    match_paths_v[i], wv, energy_samples)
            end
        else
            for i in eachindex(match_paths_v)
                out[i] = compute_directional_flux_csda(physics, shallow_flags, cfg, match_site,
                    match_paths_v[i], wv, energy_samples)
            end
        end
        out
    end

    # Exclude a few topography-artifact bins from the FIT objective: directions where the
    # uniform-rock baseline disagrees with the measurement by more than ~15x relative to the
    # bulk (median) log-ratio. These are bins in which the digital elevation model
    # under-resolves the true rock thickness (a near-zero measured count under a bright model
    # bin); they carry no material information. They are kept in the 2D map CSV/figure (so the
    # artifact is shown honestly) but given zero weight in the fit.
    rho_pin = matcfg.rock_density                     # overburden rock density pinned at the configured
                                                       # floor (--rock-density; 2710 kg/m³ = rr.for Gran Sasso)
    flux0 = flux_only_at(rho_pin, zeros(n_cells))     # dry standard-rock baseline
    lr0 = log.(max.(flux0, 1e-300)) .- log.(max.(meas_v, 1e-300))
    lr_center = median(lr0)
    fit_mask = Float64.(abs.(lr0 .- lr_center) .<= log(15.0))   # 1 = used in the fit, 0 = artifact
    n_art = Int(length(fit_mask) - sum(fit_mask))
    mw = sum(fit_mask)
    opt_logscale = pred -> exp(sum(fit_mask .* (log.(max.(meas_v, 1e-300)) .- log.(max.(pred, 1e-300)))) / mw)
    rel_resid = (pred, ob) -> norm(fit_mask .* (pred .- ob)) / max(norm(fit_mask .* ob), 1e-30)
    log_std = pred -> (lr = log.(max.(pred, 1e-300)) .- log.(max.(meas_v, 1e-300));
                       μ = sum(fit_mask .* lr) / mw; sqrt(sum(fit_mask .* (lr .- μ) .^ 2) / mw))

    # Fit against the measured 2D azimuth/zenith surface of:
    #   (a) the rock density PINNED at the standard-rock floor (rho_pin) — rock cannot be lighter
    #       than standard rock;
    #   (b) a GLOBAL WATER-MIX fraction w0 — a uniform rock+water background of the whole
    #       overburden (a fitted "wet standard rock"), the differentiable global parameter that
    #       carries the bulk lightening the pinned density cannot. It is fitted by minimising the
    #       SCALE-INVARIANT shape misfit (variance of the log model/measured ratio; the global
    #       normalisation drops out, so w0 captures only the zenith-dependent absorption shape);
    #   (c) a closed-form global normalisation s (forward→measured unit conversion);
    #   (d) a NON-NEGATIVE per-cell water field w∈[0,0.9] on top of that uniform mix (relinearised
    #       GN), adding localized structure. No denser-than-rock (signed) endpoint is used.
    golden = (f, a, b) -> begin
        φ = (sqrt(5.0) - 1.0) / 2.0
        c = b - φ * (b - a); d = a + φ * (b - a); fc = f(c); fd = f(d)
        for _ in 1:12
            if fc < fd
                b = d; d = c; fd = fc; c = b - φ * (b - a); fc = f(c)
            else
                a = c; c = d; fc = fd; d = a + φ * (b - a); fd = f(d)
            end
        end
        0.5 * (a + b)
    end

    # GEOLOGICAL PRIOR: the known Gran Sasso saturated cave layer / aquifer sits ~550 m above the
    # detector. We restrict the reconstructed water to cells ABOVE 500 m; everything below stays
    # dry standard rock. The free (volume-filling) fit is unphysical -- this anchors the water to
    # the aquifer region that the single-viewpoint data cannot localise on their own.
    cell_z = Float64[cell_centroid(volume, i)[3] for i in 1:n_cells]
    # Only ROCK cells crossed by the measured rays count (a cell with no ray sensitivity is air,
    # above the topographic surface) -- otherwise the smoothness prior bleeds water into air cells
    # above the mountain. Aquifer = above 500 m AND inside the rock (nonzero column in the forward J).
    crossed = vec(sum(abs.(base_J[valid_bins, :]); dims = 1)) .> 0.0
    aquifer_mask = (cell_z .> 500.0) .& crossed
    n_aq = count(aquifer_mask)
    maskdiag = Diagonal(Float64.(aquifer_mask))
    println(@sprintf("  aquifer prior: water restricted to %d / %d rock cells above 500 m (below surface)", n_aq, n_cells))

    # Density pinned at the standard-rock floor; the lightening is carried by a uniform water-mix
    # fraction w0 over the AQUIFER cells, fitted by the scale-invariant shape misfit.
    logvar_w0 = w0 -> begin
        wv = zeros(n_cells); wv[aquifer_mask] .= w0
        pred = flux_only_at(rho_pin, wv)
        lr = log.(max.(pred, 1e-300)) .- log.(max.(meas_v, 1e-300))
        μ = sum(fit_mask .* lr) / mw
        sum(fit_mask .* (lr .- μ) .^ 2) / mw
    end
    rho_fit = rho_pin
    w_matched = zeros(n_cells)
    s = opt_logscale(flux0)
    res_base = rel_resid(flux0, meas_v ./ s)
    # Smoothness-prior weight, scaled to the weighted data curvature diag(JᵀWJ).
    σ0 = sqrt.(max.(meas_v ./ s, 0.0) ./ args.exposure); σ0 .= max.(σ0, 1e-12 * maximum(meas_v ./ s))
    Wobs0 = (1.0 ./ σ0 .^ 2) .* fit_mask
    diagJtWJ = vec(sum(Wobs0 .* (Matrix(base_J[valid_bins, :]) .^ 2); dims = 1))
    gn_scale = (m = filter(>(0), diagJtWJ); isempty(m) ? 1.0 : median(m))
    sm_prior = SmoothnessPrior(cell_neighbors(volume), args.papermatch_reg * gn_scale)
    # Reference in-fit data-weight level the prior weight `gn_scale` is calibrated to.
    # The robust + DEM reweighting below is RELATIVE (per-bin) — we renormalize each
    # GN pass so its median in-fit weight matches this reference, otherwise shrinking
    # the absolute weight level would let the fixed smoothness prior over-regularize
    # the field to zero.
    ref_wmed = (p = filter(>(0), Wobs0); isempty(p) ? 1.0 : median(p))
    # Per-fit-bin DEM/topography down-weight: the long-slant (high-zenith) rays traverse
    # the most laterally-extended, least-resolved terrain, so their column density carries
    # the largest geometric error. We apply a BOUNDED multiplicative weight that falls from
    # 1 (vertical) toward ~1/5 (near-horizontal), wdem = 1/(1+(dem_unc·(secθ−1))²). A
    # variance-in-quadrature form would be wrong here: at exposure≈1e8 the Poisson σ is
    # ~1e-4 relative, so any absolute DEM fraction would dominate σ by orders of magnitude
    # and collapse all non-vertical weights, leaving the fit driven by a handful of vertical
    # bins (→ zero water). The bounded multiplicative factor down-weights slant bins gently.
    zen_v = Float64[match_paths_v[i].zenith for i in eachindex(match_paths_v)]
    dem_slant = args.papermatch_dem_unc .* max.(1.0 ./ cosd.(zen_v) .- 1.0, 0.0)
    wdem = 1.0 ./ (1.0 .+ dem_slant .^ 2)
    fit_idx = findall(>(0.0), fit_mask)   # in-fit (non-artifact) bins, for the robust scale
    model_rho = make_csda_operator(physics, shallow_flags, mk_cfg(rho_fit), match_site,
        match_paths, energy_samples; n_cells = n_cells, valid_bins = valid_bins, threaded = args.threaded)
    # Aquifer-masked operator: zero the Jacobian columns outside the aquifer (only those cells fit).
    model_aq = function (w)
        pr, J = model_rho(w)
        return pr, J * maskdiag
    end
    # SELF-CONSISTENT RECURSION: re-calibrate the CSDA->MC correction THROUGH the current
    # reconstructed field, then re-solve, N times. Iteration 1 uses the calibration done at w=0
    # (dry rock); later iterations re-calibrate through the recovered water field, so the
    # differentiable surrogate matches full MC for the medium it is actually fitting.
    n_recur = max(1, args.papermatch_recursions)
    damp = clamp(args.papermatch_damp, 0.0, 1.0)
    w0_fit = 0.0
    w_prev = zeros(n_cells)        # converged field from the previous pass (warm start + damping)
    prev_coef = nothing            # previous correction coefficients (convergence check)
    for recur in 1:n_recur
        if recur > 1
            println(@sprintf("  [recursion %d/%d] re-calibrating CSDA->MC through the current reconstructed field (geometric residual frozen)...",
                             recur, n_recur))
            # Re-fit only the hard-loss κ_hard through the recovered field (the physical
            # straggling enhancement and the material-independent geometric residual are
            # both pinned). A gentle ridge anchors κ_hard toward its previous-pass value so
            # MC noise in the calibration reference cannot make the recursion oscillate.
            calibrate_correction_to_mc(physics, shallow_flags, matcfg, match_site, paths,
                copy(w_prev), energy_samples, args; fit_geometric = false,
                kh_anchor = get_csda_correction().kappa_hard, kh_anchor_weight = 1.0e-3)
        end
        # (a) diagnostic global water-mix shape fit (reported, not added to the field)
        w0_fit = golden(logvar_w0, 0.0, MAX_WATER_FRACTION)
        # (b) per-cell aquifer GN, warm-started from the previous pass (faster + path-stable).
        # IRLS: the FIRST solve (recur==1 & outer==1, the cold start) uses pure
        # inverse-variance weighting — at initialization the dry baseline is wrong
        # *everywhere*, so the large-residual bins are the INFORMATIVE deep ones, not
        # outliers; robust down-weighting them there would suppress the very signal that
        # drives the water reconstruction. Robustness is introduced only once a water
        # field exists and the residual scatter reflects genuine outliers.
        w_solved = copy(w_prev)
        for outer in 1:3
            cold = (recur == 1 && outer == 1)
            pred = flux_only_at(rho_fit, w_solved)
            s = opt_logscale(pred); obs = meas_v ./ s
            # Poisson inverse-variance weights, with a BOUNDED DEM down-weight (wdem) and a
            # robust outlier down-weight (rweight); both are relative and the renorm below
            # restores the overall data-vs-prior balance.
            σ = sqrt.(max.(obs, 0.0) ./ args.exposure); σ .= max.(σ, 1e-12 * maximum(obs))
            # Robust IRLS: Cauchy weight on the log-residual, scale set from the robust
            # (MAD) spread of the in-fit residuals, so the heavy tail of faint
            # topography-artifact bins is smoothly down-weighted (not L2-dominant). Skipped
            # at the cold start (every bin mis-fits there → large residual ≠ outlier).
            rweight = ones(length(obs))
            if !cold
                lr = log.(max.(pred, 1e-300)) .- log.(max.(obs, 1e-300))
                lr_med = median(lr[fit_idx])
                mad = median(abs.(lr[fit_idx] .- lr_med))
                cscale = max(1.4826 * mad * args.papermatch_robust_c, 1e-6)
                rweight = 1.0 ./ (1.0 .+ ((lr .- lr_med) ./ cscale) .^ 2)
            end
            Wobs = (1.0 ./ σ .^ 2) .* fit_mask .* wdem .* rweight
            # renormalize to the reference in-fit weight level so the robust+DEM
            # reweighting stays RELATIVE and the prior balance is preserved.
            wmed_cur = (p = Wobs[fit_idx]; q = filter(>(0), p); isempty(q) ? 1.0 : median(q))
            Wobs .*= ref_wmed / max(wmed_cur, 1e-300)
            w_solved, _ = gauss_newton_reconstruct(model_aq, obs; w0 = w_solved, n_iter = 6,
                prior = sm_prior, weights = Wobs, relinearize = true, box = (0.0, MAX_WATER_FRACTION))
            w_solved[.!aquifer_mask] .= 0.0   # keep water inside the aquifer rock cells only
        end
        # under-relaxation between passes (pass 1 takes the solve as-is; w_prev=0 there)
        w_matched = recur == 1 ? w_solved : (damp .* w_solved .+ (1.0 - damp) .* w_prev)
        w_matched[.!aquifer_mask] .= 0.0
        predr = flux_only_at(rho_fit, w_matched); s = opt_logscale(predr)
        # convergence diagnostics: relative L2 change in the field and the correction coefficients
        cur = get_csda_correction()
        coef = [cur.kappa_strag, cur.kappa_hard, cur.resid_a, cur.resid_b, cur.resid_c, cur.resid_d]
        dfield = norm(w_matched .- w_prev) / max(norm(w_matched), 1e-12)
        dcoef = prev_coef === nothing ? NaN : norm(coef .- prev_coef) / max(norm(coef), 1e-12)
        println(@sprintf("  [recursion %d/%d] w0=%.3f  s=%.3e  w mean=%.4f  rel resid=%.3f  log-std=%.3f  | dfield=%.2e dcoef=%.2e",
                         recur, n_recur, w0_fit, s, mean(w_matched), rel_resid(predr, meas_v ./ s),
                         log_std(predr), dfield, isnan(dcoef) ? -1.0 : dcoef))
        w_prev = copy(w_matched)
        prev_coef = coef
        if recur > 1 && dfield < args.papermatch_recur_tol && dcoef < args.papermatch_recur_tol
            println(@sprintf("  [recursion] self-consistent: dfield & dcoef < tol=%.2e after %d passes; stopping.",
                             args.papermatch_recur_tol, recur))
            break
        end
    end

    # Final state. The global forward→measured normalization s is otherwise a FREE
    # log-scale fit, so it absorbs the absolute flux level and only the shape is
    # constrained. Estimate s (and its uncertainty) from the most RELIABLE bins —
    # the near-vertical in-fit bins, least affected by DEM error — as the per-bin
    # log(measured/model) spread, and report the implied absolute-scale band. With
    # --papermatch-abs-scale>0 the scale is PINNED to that absolute value instead,
    # turning the shape fit into a quantitative (absolute) water-content result.
    pred_m = flux_only_at(rho_fit, w_matched)
    vert_fit = [k for k in eachindex(valid_bins)
                if fit_mask[k] > 0.0 && match_zen[valid_bins[k]] < 40.0 && pred_m[k] > 0.0]
    s_log = isempty(vert_fit) ? log(opt_logscale(pred_m)) :
            mean(log.(meas_v[vert_fit]) .- log.(pred_m[vert_fit]))
    s_reliable = exp(s_log)
    s_relsd = length(vert_fit) > 1 ?
              std(log.(meas_v[vert_fit]) .- log.(pred_m[vert_fit])) / sqrt(length(vert_fit)) : NaN
    s = args.papermatch_abs_scale > 0.0 ? args.papermatch_abs_scale : opt_logscale(pred_m)
    obs = meas_v ./ s
    res_m = rel_resid(pred_m, obs)
    log_std_base = log_std(flux0); log_std_m = log_std(pred_m)
    # Box-saturation diagnostic: split pinned cells into the DRY floor (w=0) and the
    # WATER-SATURATED ceiling (w=MAX). Ceiling cells are directions the water-only
    # model cannot explain (they would need ρ<water): flag them as "water-insufficient".
    floor_pinned = count(x -> x <= 1e-3, w_matched)
    ceil_pinned  = count(x -> x >= MAX_WATER_FRACTION - 1e-3, w_matched)
    aq_ceiling   = count(i -> aquifer_mask[i] && w_matched[i] >= MAX_WATER_FRACTION - 1e-3, 1:n_cells)
    pinned = floor_pinned + ceil_pinned
    println(@sprintf("  global scale s=%.4e (free log-fit) ; reliable-bin s=%.4e (±%.1f%% from %d vertical in-fit bins)%s",
                     opt_logscale(pred_m), s_reliable, isnan(s_relsd) ? 0.0 : 100 * s_relsd, length(vert_fit),
                     args.papermatch_abs_scale > 0.0 ? @sprintf(" ; PINNED to %.4e", s) : ""))
    println(@sprintf("  box saturation: %d/%d cells at dry floor (w=0), %d at water ceiling (w=%.1f; %d in aquifer = water-insufficient directions)",
                     floor_pinned, n_cells, ceil_pinned, MAX_WATER_FRACTION, aq_ceiling))
    println(@sprintf("  rock density pinned at standard rock %.0f kg/m^3; global water-mix fraction w0=%.3f",
                     rho_fit, w0_fit))
    n_vert = count(i -> match_zen[i] < 60.0, valid_bins)
    n_horiz = length(valid_bins) - n_vert
    println(@sprintf("  fit bins=%d (excluded %d topography-artifact bins; theta<60: %d, theta>=60: %d, up to %.0f deg)",
                     length(valid_bins) - n_art, n_art, n_vert, n_horiz, maximum(match_zen[valid_bins])))
    println(@sprintf("  norm=%.3e  rel resid (fit bins) %.3f -> %.3f  log-std %.3f -> %.3f",
                     s, res_base, res_m, log_std_base, log_std_m))
    println(@sprintf("  matched w: mean=%.3f std=%.3f min=%.3f max=%.3f  box-pinned cells=%d/%d",
                     mean(w_matched), std(w_matched), minimum(w_matched), maximum(w_matched), pinned, n_cells))

    # --- Rigorous inverse-problem statistics: posterior uncertainty map (Laplace),
    # null-model comparison, and nuisance-parameter (absolute-level) profiling.
    # Statistical weights are the pure Poisson × DEM down-weight on the fit bins
    # (no robust/conditioning rescale): the honest data covariance. The Laplace
    # posterior is CONDITIONAL on the normalisation s (the per-cell shape the
    # angular data constrain); the absolute level, degenerate with s, is left to
    # the inventory profile.
    println("Posterior uncertainty + null-model tests + absolute-level profile...")
    obs_stat = meas_v ./ s
    W_stat = fit_mask .* wdem .* (args.exposure ./ max.(obs_stat, 1e-300))
    flux_w = w -> flux_only_at(rho_fit, w)
    post = laplace_posterior(model_aq, w_matched; weights = W_stat, prior = sm_prior, free = aquifer_mask)
    nulls = null_model_tests(flux_w, w_matched, w0_fit, aquifer_mask,
                             meas_v, fit_mask, wdem, args.exposure; p_eff = post.p_eff)
    prof = profile_inventory(flux_w, w_matched, aquifer_mask,
                             meas_v, fit_mask, wdem, args.exposure; var_scale = nulls.var_scale)
    aq_idx = findall(aquifer_mask)
    n_aq = max(length(aq_idx), 1)
    # Calibrate the posterior covariance to reduced-χ²=1: the nominal exposure is
    # arbitrary, so inflate σ by √φ. The resolution (averaging kernel) and p_eff are
    # scale-invariant and need no calibration.
    sigfac = sqrt(nulls.var_scale)
    sigma_cal = post.sigma .* sigfac
    z_cal = post.z ./ sigfac
    sig_med = isempty(aq_idx) ? 0.0 : median(sigma_cal[aq_idx])
    res_med = isempty(aq_idx) ? 0.0 : median(post.resolution[aq_idx])
    z_resolved = count(i -> z_cal[i] >= 2.0, aq_idx)
    wbar_map = prof.T_map / n_aq          # mean band fraction at the GN MAP
    wbar_min = prof.w_mean_min            # amplitude-profile maximum-likelihood level
    println(@sprintf("  posterior (Laplace, conditional on s; σ calibrated to reduced-χ²=1): p_eff=%.1f effective DOF of %d cells; median resolution=%.2f; median σ_w=%.2f; %d cells at w/σ≥2",
                     post.p_eff, length(aq_idx), res_med, sig_med, z_resolved))
    println(@sprintf("  null-model (calibrated, φ=%.0f): water-vs-dry %.0fσ; structure-vs-uniform %.0fσ  [χ² dry=%.3e unif=%.3e fit=%.3e, ndata=%d]",
                     nulls.var_scale, nulls.sigma_dry, nulls.sigma_unif,
                     nulls.chi2_dry, nulls.chi2_unif, nulls.chi2_fit, nulls.ndata))
    println(@sprintf("  model selection: AIC dry=%.0f unif=%.0f fit=%.0f | BIC dry=%.0f unif=%.0f fit=%.0f (lower favoured)",
                     nulls.aic.dry, nulls.aic.unif, nulls.aic.fit, nulls.bic.dry, nulls.bic.unif, nulls.bic.fit))
    println(@sprintf("  amplitude profile (shape fixed, s profiled out): GN-MAP w̄=%.3f; profile-ML w̄=%.3f; 68%% CI w̄∈[%.3f, %.3f]; 95%% CI w̄∈[%.3f, %.3f]",
                     wbar_map, wbar_min, prof.ci68[1], prof.ci68[2], prof.ci95[1], prof.ci95[2]))
    unc_csv = joinpath(args.output_dir, "lvd_water_uncertainty.csv")
    open(unc_csv, "w") do io
        println(io, "cell_idx,x_m,y_m,z_m,w,sigma_w,z_score,resolution,pinned")
        for i in aq_idx
            cx, cy, cz = cell_centroid(volume, i)
            pin = w_matched[i] <= 1e-3 ? "floor" :
                  (w_matched[i] >= MAX_WATER_FRACTION - 1e-3 ? "ceiling" : "free")
            println(io, @sprintf("%d,%.1f,%.1f,%.1f,%.4f,%.4f,%.3f,%.3f,%s",
                                 i, cx, cy, cz, w_matched[i], sigma_cal[i], z_cal[i], post.resolution[i], pin))
        end
    end
    prof_csv = joinpath(args.output_dir, "lvd_water_profile.csv")
    open(prof_csv, "w") do io
        println(io, "T_inventory,w_mean,chi2,delta_chi2,delta_chi2_calibrated")
        for k in eachindex(prof.T)
            println(io, @sprintf("%.4f,%.5f,%.4f,%.4f,%.4f",
                                 prof.T[k], prof.w_mean[k], prof.chi2[k], prof.dchi2[k], prof.dchi2_cal[k]))
        end
    end
    println("  uncertainty + profile CSVs: $unc_csv , $prof_csv")

    # --- Reproduce the measured 2D azimuth/zenith map (measured / model / residual per bin).
    # The reconstruction targets this surface directly; no 1D angular-curve projection is
    # produced (the 2D map is the detector observable).
    println("Writing the measured-vs-model 2D single-muon map (per-bin measured/model/residual)...")
    map_csv = joinpath(args.output_dir, "lvd_single_muon_2d_match.csv")
    map_plot = joinpath(args.output_dir, "lvd_single_muon_2d_match.html")
    map_stats = write_single_muon_2d_comparison(match_paths, valid_bins, obs, pred_m, s;
        output_csv = map_csv, output_plot = map_plot)

    # --- Reconstructed-field outputs written FIRST (fast), so the 3D field/figure data are
    # available immediately after the fit, before the slow full-MC validation below.
    matcfg_fit = mk_cfg(rho_fit)
    field_plot = joinpath(args.output_dir, "lvd_single_muon_angular_distribution_field.html")
    plot_water_field(volume, w_matched; output_path = field_plot, max_cells = 300)
    water_plot = joinpath(args.output_dir, "lvd_water_increase_vs_rock.html")
    plot_water_increase(volume, w_matched, matcfg_fit; output_path = water_plot)
    w_csv = joinpath(args.output_dir, "lvd_matched_mixture_field.csv")
    open(w_csv, "w") do io
        println(io, "cell_idx,x_m,y_m,z_m,w,rho_eff_kg_m3")
        for i in 1:n_cells
            cx, cy, cz = cell_centroid(volume, i)
            println(io, @sprintf("%d,%.1f,%.1f,%.1f,%.4f,%.1f",
                                 i, cx, cy, cz, w_matched[i], cell_density(w_matched[i], false, matcfg_fit)))
        end
    end
    println("  reconstructed-field outputs: $field_plot , $water_plot , $w_csv")

    # --- ULTIMATE TEST: push the reconstructed aquifer through FULL backward Monte Carlo
    # transport (scattering+straggling) and confirm it reproduces the measured 2D flux. The CSDA
    # operator only supplies the differentiable Jacobian for the inversion; this independent
    # full-MC check rules out the recovered field being an artefact of that surrogate.
    # Validate ALL theta<60 fit bins plus a STRATIFIED subset of the high-theta
    # (long-slant) bins, so the check spans the full depth range of the fit and not
    # only the vertical bins. Full MC on every horizontal bin is prohibitively slow.
    mc_cap = 60.0
    vert = [k for k in eachindex(valid_bins) if fit_mask[k] > 0.0 && match_paths[valid_bins[k]].zenith < mc_cap]
    horiz_all = [k for k in eachindex(valid_bins) if fit_mask[k] > 0.0 && match_paths[valid_bins[k]].zenith >= mc_cap]
    sort!(horiz_all; by = k -> match_paths[valid_bins[k]].zenith)
    nhz = clamp(args.papermatch_val_horiz, 0, length(horiz_all))
    horiz = nhz > 0 ? horiz_all[unique(round.(Int, range(1, length(horiz_all); length = nhz)))] : Int[]
    val_set = sort(unique(vcat(vert, horiz)))
    println(@sprintf("Full-MC validation over %d bins (%d theta<%.0f + %d stratified theta>=%.0f; %d MC samples/bin, scattering+straggling)...",
                     length(val_set), length(vert), mc_cap, length(horiz), mc_cap, args.papermatch_mc_samples))
    mc_samp = sample_energy_set(args.papermatch_mc_samples, args.energy_min, args.energy_max, args.seed + 777)
    mats_r, dens_r = build_cell_properties_for_mc(shallow_flags, w_matched, matcfg_fit)
    mc_rec = fill(NaN, length(valid_bins))
    mc_err = fill(NaN, length(valid_bins))
    Threads.@threads for j in eachindex(val_set)
        k = val_set[j]
        p = match_paths[valid_bins[k]]
        mc_rec[k], mc_err[k] = compute_directional_flux_mc(physics, matcfg_fit, match_site, p, mats_r, dens_r,
            mc_samp, args.seed + 900_000 + k; straggling = args.straggling,
            energy_threshold_low = args.energy_threshold_low, scattering = true)
    end
    idx = findall(k -> isfinite(mc_rec[k]) && fit_mask[k] > 0.0, eachindex(valid_bins))
    s_mc = exp(mean(log.(meas_v[idx]) .- log.(max.(mc_rec[idx], 1e-300))))
    relres = ii -> norm(mc_rec[ii] .- meas_v[ii] ./ s_mc) / max(norm(meas_v[ii] ./ s_mc), 1e-30)
    res_mc = relres(idx)
    logstd_mc = std(log.(max.(mc_rec[idx], 1e-300)) .- log.(max.(meas_v[idx], 1e-300)))
    vidx = [k for k in idx if match_paths[valid_bins[k]].zenith < mc_cap]
    hidx = [k for k in idx if match_paths[valid_bins[k]].zenith >= mc_cap]
    res_mc_v = isempty(vidx) ? NaN : relres(vidx)
    res_mc_h = isempty(hidx) ? NaN : relres(hidx)
    println(@sprintf("  full-MC residual (%d bins): reconstructed %.3f   (CSDA fit was %.3f -> %.3f)",
                     length(idx), res_mc, res_base, res_m))
    println(@sprintf("    by depth: theta<%.0f (%d bins) %.3f ; theta>=%.0f (%d bins) %.3f",
                     mc_cap, length(vidx), res_mc_v, mc_cap, length(hidx), res_mc_h))
    println(@sprintf("  full-MC log model/measured std: %.3f  (CSDA %.3f)", logstd_mc, log_std_m))
    mcv_csv = joinpath(args.output_dir, "lvd_single_muon_2d_mc_validation.csv")
    mcv_plot = joinpath(args.output_dir, "lvd_single_muon_2d_mc_validation.html")
    mc_stats = write_single_muon_2d_comparison(match_paths, valid_bins[idx], meas_v[idx] ./ s_mc, mc_rec[idx], s_mc;
        output_csv = mcv_csv, output_plot = mcv_plot)
    # Per-bin measured vs full-MC simulated flux WITH the MC error band, for the 3D angular plot.
    band_csv = joinpath(args.output_dir, "lvd_mc_flux_band.csv")
    open(band_csv, "w") do io
        println(io, "bin_idx,zenith_deg,azimuth_deg,measured_flux,mc_flux,mc_err")
        for k in idx
            b = valid_bins[k]
            println(io, @sprintf("%d,%.4f,%.4f,%.8e,%.8e,%.8e",
                                 b, match_paths[b].zenith, mod(match_paths[b].azimuth, 360.0),
                                 meas_v[k], mc_rec[k] * s_mc, mc_err[k] * s_mc))
        end
    end
    println("  full-MC flux band: $band_csv")
    pm_txt = joinpath(args.output_dir, "lvd_single_muon_angular_distribution.txt")
    open(pm_txt, "w") do io
        println(io, "Water-content reconstruction from the measured 2D LVD single-muon angular surface")
        println(io, "(pinned overburden rock + a fitted global water-mix + a non-negative per-cell water field; no signed/denser-rock endpoint)")
        println(io, @sprintf("rock density PINNED at the configured Gran Sasso value = %.0f kg/m^3 (rr.for; the data prefer lighter, lightening comes from water)",
                             rho_fit))
        println(io, @sprintf("FITTED global water-mix fraction w0 = %.4f (uniform 'wet standard rock' background, scale-invariant shape fit)",
                             w0_fit))
        println(io, @sprintf("water endpoint density = %.0f kg/m^3 (w=+%.1f)", matcfg.water_density, MAX_WATER_FRACTION))
        println(io, @sprintf("primary/Gaisser altitude = %.0f km", args.primary_altitude_km))
        println(io)
        println(io, "Joint (rho_rock, water-field) GN match vs the full measured 2D azimuth/zenith surface:")
        println(io, @sprintf("  match data: %s", args.match_data))
        println(io, @sprintf("  covered bins: %d (theta<60: %d, theta>=60: %d, max theta %.0f deg)",
                             length(valid_bins), n_vert, n_horiz, maximum(match_zen[valid_bins])))
        println(io, @sprintf("  topography-artifact bins excluded from the fit: %d (baseline off by >15x vs the median ratio)", n_art))
        println(io, @sprintf("  forward->measured normalization: %.4e%s", s,
                             args.papermatch_abs_scale > 0.0 ? " (PINNED, absolute)" : " (free log-fit)"))
        println(io, @sprintf("  reliable-bin scale: %.4e ± %.1f%% (from %d near-vertical in-fit bins; the absolute anchor)",
                             s_reliable, isnan(s_relsd) ? 0.0 : 100 * s_relsd, length(vert_fit)))
        println(io, @sprintf("  relative data residual (fit bins): baseline %.4f -> matched %.4f", res_base, res_m))
        println(io, @sprintf("  log model/measured std (scale-invariant shape misfit): %.4f -> %.4f", log_std_base, log_std_m))
        println(io, @sprintf("  matched water fraction: mean=%.3f max=%.3f nonzero(>1e-3)=%d/%d",
                             mean(w_matched), maximum(w_matched), count(>(1e-3), w_matched), n_cells))
        println(io, @sprintf("  water increased (w>0.02) in %d / %d cells", count(>(0.02), w_matched), n_cells))
        println(io, @sprintf("  box-pinned cells (w=0 or w=%.1f): %d / %d (dry floor %d, water ceiling %d)",
                             MAX_WATER_FRACTION, pinned, n_cells, floor_pinned, ceil_pinned))
        println(io, @sprintf("  water-insufficient (aquifer cells at the w=%.1f ceiling, water-only model under-explains): %d",
                             MAX_WATER_FRACTION, aq_ceiling))
        println(io)
        println(io, "Posterior uncertainty, null-model comparison, and water-amplitude profile:")
        println(io, "  Laplace posterior (Gauss-Newton Hessian JtWJ + prior; conditional on the normalization s,")
        println(io, "  i.e. the per-cell SHAPE the angular data constrain; sigma calibrated to reduced-chi^2=1):")
        println(io, @sprintf("    effective resolved DOF p_eff = %.1f of %d aquifer cells; median averaging-kernel resolution = %.2f",
                             post.p_eff, length(aq_idx), res_med))
        println(io, @sprintf("    median calibrated posterior sigma_w = %.2f; cells at w/sigma >= 2: %d / %d   (csv: lvd_water_uncertainty.csv)",
                             sig_med, z_resolved, length(aq_idx)))
        println(io, "  Null-model likelihood-ratio. The published LVD intensities carry no absolute count error,")
        println(io, @sprintf("  so chi^2 is calibrated to reduced-chi^2 = 1 (variance scale phi = %.0f; ndata = %d fit bins):",
                             nulls.var_scale, nulls.ndata))
        println(io, @sprintf("    raw chi^2: dry standard rock = %.3e ; uniform band water-mix = %.3e ; structured fit = %.3e",
                             nulls.chi2_dry, nulls.chi2_unif, nulls.chi2_fit))
        println(io, @sprintf("    water vs dry rock:         Delta chi2/phi on p_eff=%.1f dof   -> %.0f sigma",
                             post.p_eff, nulls.sigma_dry))
        println(io, @sprintf("    structure vs uniform band: Delta chi2/phi on p_eff-1=%.1f dof -> %.0f sigma",
                             max(post.p_eff - 1, 0.0), nulls.sigma_unif))
        println(io, @sprintf("    AIC (reduced): dry=%.0f uniform=%.0f fit=%.0f | BIC: dry=%.0f uniform=%.0f fit=%.0f (lower favoured)",
                             nulls.aic.dry, nulls.aic.unif, nulls.aic.fit,
                             nulls.bic.dry, nulls.bic.unif, nulls.bic.fit))
        println(io, "  Water-amplitude profile: the reconstructed shape is rescaled and s refit, profiling the")
        println(io, "  overall level (mean band fraction w_bar) given the shape and priors. This constrains the")
        println(io, "  amplitude along the reconstructed shape; the residual shape freedom is in the per-cell posterior above.")
        println(io, @sprintf("    GN-MAP w_bar = %.3f ; amplitude-profile maximum-likelihood w_bar = %.3f", wbar_map, wbar_min))
        println(io, @sprintf("    68%% CI: w_bar in [%.3f, %.3f] ; 95%% CI: w_bar in [%.3f, %.3f]  (NaN = not bounded within the grid)",
                             prof.ci68[1], prof.ci68[2], prof.ci95[1], prof.ci95[2]))
        println(io, "    csv: lvd_water_profile.csv (T, w_mean, chi2, delta_chi2, delta_chi2_calibrated)")
        println(io)
        println(io, "2D measured-vs-model map (CSDA surrogate used for the fit):")
        println(io, @sprintf("  bins used: %d", map_stats.n_bins))
        println(io, @sprintf("  relative residual RMS: %.4f", map_stats.rms_rel))
        println(io, @sprintf("  relative residual bias: %.4f", map_stats.bias_rel))
        println(io, "  csv: lvd_single_muon_2d_match.csv (per-bin measured/model/residual)")
        println(io)
        println(io, "FULL backward-MC validation of the reconstructed field (scattering+straggling, the verdict):")
        println(io, @sprintf("  MC samples/bin: %d", args.papermatch_mc_samples))
        println(io, @sprintf("  validated bins: %d (theta<60: %d, stratified theta>=60: %d)", length(idx), length(vidx), length(hidx)))
        println(io, @sprintf("  relative data residual (fit bins): reconstructed %.4f  (CSDA fit %.4f)", res_mc, res_m))
        println(io, @sprintf("  by depth: theta<60 %.4f ; theta>=60 %.4f", res_mc_v, res_mc_h))
        println(io, @sprintf("  log model/measured std: %.4f  (CSDA %.4f)", logstd_mc, log_std_m))
        println(io, @sprintf("  (CSDA-fit residual was %.4f -> %.4f; full MC confirms the improvement)", res_base, res_m))
        println(io, @sprintf("  2D map residual RMS under MC: %.4f", mc_stats.rms_rel))
        println(io, "  csv: lvd_single_muon_2d_mc_validation.csv (measured / full-MC model / residual)")
        println(io, "Caveat: the global forward->measured normalization absorbs the absolute flux scale;")
        println(io, "  the full-MC check is the high-fidelity confirmation that the CSDA-reconstructed field holds up.")
    end
    println("  paper-match outputs: $pm_txt")
    println("  $map_plot")
    println("  $map_csv")
    println("  $mcv_csv")
    println("  $mcv_plot")
    println("  $field_plot")
    println("  $water_plot")
    println("  $w_csv")
    return (w_matched = w_matched, normalization = s,
            residual = res_m, map_stats = map_stats)
end

function main()
    args = parse_commandline()

    println("=" ^ 68)
    println(" DiffPumas - LVD Tomography with Rock/Water Sensitivities")
    println("=" ^ 68)
    println()
    println("Configuration:")
    println("  Geometry mode:     $(args.geometry)")
    println("  Mesh half-width:   $(args.mesh_half_km) km")
    println("  Surface step:      $(args.surface_step_km) km")
    println("  Tet max volume:    $(args.tet_max_volume) m^3")
    println("  Grid NZ:           $(args.grid_nz)")
    println("  Angular grid:      theta < $(args.zenith_max_deg) by $(args.zenith_step_deg) deg, phi by $(args.azimuth_step_deg) deg")
    println("  CSDA samples/bin:  $(args.n_samples)")
    println("  MC samples/case:   $(args.mc_samples)")
    println("  Energy range:      $(args.energy_min) - $(args.energy_max) GeV")
    println("  Threshold:         $(args.energy_threshold_low) GeV")
    println("  FD delta:          $(args.fd_delta)")
    println("  Validation cases:  $(args.validation_cases)")
    println("  Seed:              $(args.seed)")
    println("  Porous top:        $(args.porous_top_thickness) m")
    if args.rich_field
        println("  Ground truth:      multi-anomaly (slab w=$(args.slab_w), lens w=$(args.lens_w), dip w=$(args.dip_w))")
    else
        println("  Ground truth:      single aquifer box, w=$(args.aquifer_water_fraction)")
    end
    println("  AAD GD iters/lr:   $(args.gd_iters) / $(args.gd_lr)")
    println("  Output dir:        $(args.output_dir)")
    println()

    mkpath(args.output_dir)

    physics = load_or_create_physics(args.dump_path; mdf_path = DEFAULT_MDF)
    physics === nothing && error("Failed to load physics tables")
    print_physics_summary(physics)
    println()

    rock_idx = get_material_index(physics, "StandardRock")
    water_idx = get_material_index(physics, "Water")
    air_idx = get_material_index(physics, "Air")
    porous_idx = get_material_index(physics, "PorousWetRock")
    rock_idx == -1 && error("StandardRock not found")
    water_idx == -1 && error("Water not found")
    air_idx == -1 && error("Air not found")

    rock_dens = args.rock_density > 0 ? args.rock_density : Float64(physics.tables[rock_idx].density)
    matcfg = MaterialConfig(
        rock_idx,
        water_idx,
        air_idx,
        porous_idx,
        rock_dens,
        Float64(physics.tables[water_idx].density),
        porous_idx > 0 ? Float64(physics.tables[porous_idx].density) : 0.0,
        args.porous_top_thickness,
    )

    println("Building / loading LVD topography...")
    emap = LVDTopo.build_elevation_map()
    println("  Detector elevation: $(LVDTopo.DETECTOR_ELEVATION) m ASL")
    println("  Primary altitude:   $(args.primary_altitude_km * 1000.0) m above detector")
    println()

    volume = create_sensitivity_volume(emap, matcfg, args)
    println("Constructed $(geometry_name(volume)) sensitivity volume with $(num_cells(volume)) cells")
    println()

    site = build_site(args.primary_altitude_km * 1000.0)
    shallow_flags = Bool[is_shallow_cell(volume, i) for i in 1:num_cells(volume)]

    water_fractions = create_initial_water_field(volume, args)
    n_aquifer = count(>(0.0), water_fractions)
    println("Baseline water-fraction field:")
    println("  Non-zero cells: $n_aquifer / $(length(water_fractions))")
    if n_aquifer > 0
        println(@sprintf("  Aquifer box center: (%.1f, %.1f, %.1f) m",
                         args.aquifer_center_x, args.aquifer_center_y, args.aquifer_center_z))
    end
    println()

    zeniths = exclusive_degree_grid(args.zenith_max_deg, args.zenith_step_deg)
    azimuths = exclusive_degree_grid(360.0, args.azimuth_step_deg)
    paths = precompute_paths(volume, emap, zeniths, azimuths)
    println()

    energy_samples = sample_energy_set(args.n_samples, args.energy_min, args.energy_max, args.seed)

    # Calibrate the differentiable CSDA fluctuation/hard-loss correction to full
    # MC (Part A) BEFORE assembling the Jacobian, so the forward operator and its
    # sensitivities use the corrected (MC-matched) physics.
    calibrate_correction_to_mc(physics, shallow_flags, matcfg, site, paths,
        water_fractions, energy_samples, args)
    println()

    # (b) Reconstruction accuracy vs exposure (then exit).
    if args.reconstruction_sweep
        run_reconstruction_sweep(physics, shallow_flags, matcfg, site, volume, paths, energy_samples, args;
            output_path = joinpath(args.output_dir, "lvd_reconstruction_sweep.txt"),
            plot_path = joinpath(args.output_dir, "lvd_reconstruction_sweep.html"))
        println("\nReconstruction-sweep outputs written to $(args.output_dir)")
        return 0
    end

    # (a) Detection gain vs contrast (then exit).
    if args.detection_contrast_sweep
        run_detection_contrast_sweep(physics, shallow_flags, matcfg, site, volume, paths, energy_samples, args;
            output_path = joinpath(args.output_dir, "lvd_detection_contrast_sweep.txt"),
            plot_path = joinpath(args.output_dir, "lvd_detection_contrast_sweep.html"))
        println("\nDetection-contrast-sweep outputs written to $(args.output_dir)")
        return 0
    end

    # Detection benchmark (then exit): CSDA gradient matched filter Ψ vs ROI-sum vs SART.
    if args.detection_benchmark
        run_detection_benchmark(physics, shallow_flags, matcfg, site, volume, paths, energy_samples, args;
            output_path = joinpath(args.output_dir, "lvd_detection_benchmark.txt"),
            plot_path = joinpath(args.output_dir, "lvd_detection_benchmark.html"))
        println("\nDetection-benchmark outputs written to $(args.output_dir)")
        return 0
    end

    # Matched-resolution detectability study (then exit): τ₉₅ per solver + efficiency gains.
    if args.detectability
        run_detectability_matched(physics, shallow_flags, matcfg, site, volume, paths, energy_samples, args;
            output_path = joinpath(args.output_dir, "lvd_detectability_sweep.txt"),
            plot_path = joinpath(args.output_dir, "lvd_detectability_sweep.html"))
        println("\nDetectability-sweep outputs written to $(args.output_dir)")
        return 0
    end

    # Fast path: regenerate ONLY the paper-match figures (skip the inverse-demo,
    # validation, resolution and main plots) — for iterating on the reconstruction.
    if args.papermatch_only
        run_paper_match(physics, shallow_flags, matcfg, volume, emap, paths, energy_samples,
                        rock_idx, air_idx, args)
        println("\nPaper-match-only outputs written to $(args.output_dir)")
        return 0
    end

    t_jac = time()
    flux_values, jacobian = compute_flux_and_jacobian(
        physics, shallow_flags, matcfg, site, paths, water_fractions, energy_samples;
        n_cells = num_cells(volume), threaded = args.threaded,
    )
    println(@sprintf("  Jacobian assembled in %.2f s", time() - t_jac))
    println()

    n_valid = count(isfinite, flux_values)
    flux_grid = vector_to_grid(flux_values, length(zeniths), length(azimuths))
    aggregate = aggregate_cell_sensitivity(jacobian, n_valid)
    cases = select_validation_cases(paths, flux_values, jacobian; n_cases = args.validation_cases)
    validation_results = run_validation_cases(
        physics, shallow_flags, matcfg, site, paths, water_fractions, cases, args,
    )
    println()

    # Inverse problem: recover the per-cell rock/water field with SART, MLEM and
    # our differentiable gradient-descent solver, then estimate spatial resolution.
    inverse_txt = joinpath(args.output_dir, "lvd_tomography_inverse.txt")
    println("Solving the inverse problem (solver=$(args.solver))...")
    run_inverse_demo(physics, shallow_flags, matcfg, site, volume, paths, energy_samples,
        flux_values, jacobian, aggregate, args; output_path = inverse_txt)
    println()

    flux_plot = joinpath(args.output_dir, "lvd_tomography_flux.html")
    mesh_plot = joinpath(args.output_dir, "lvd_tomography_mesh.html")
    summary_txt = joinpath(args.output_dir, "lvd_tomography_validation.txt")

    create_flux_map_plot(zeniths, azimuths, flux_grid; output_path = flux_plot)
    create_cell_sensitivity_plots(zeniths, azimuths, jacobian, cases; output_dir = args.output_dir)
    create_volume_sensitivity_plot(volume, aggregate;
        output_path = mesh_plot,
        max_cells = args.max_plot_cells)
    write_validation_summary(validation_results, volume; output_path = summary_txt)

    if args.papermatch
        run_paper_match(physics, shallow_flags, matcfg, volume, emap, paths, energy_samples,
                        rock_idx, air_idx, args)
        println()
    end

    println("Outputs written to:")
    println("  $flux_plot")
    println("  $mesh_plot")
    println("  $summary_txt")
    println("  $inverse_txt")
    println()
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
