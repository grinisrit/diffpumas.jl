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
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "data")
const MAX_WATER_FRACTION = 0.9
const POINT_EPS = 1e-4
const SURFACE_SEARCH_STEP_M = 40.0
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
        z_local[iy, ix] = local_surface_height(emap, 1000.0 * x_nodes_km[ix], 1000.0 * y_nodes_km[iy])
        isfinite(z_local[iy, ix]) || error("Topography sampling left the LVD map at x=$(x_nodes_km[ix]) km, y=$(y_nodes_km[iy]) km")
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
    args.aquifer_water_fraction <= 0.0 && return fractions

    water_fraction = min(args.aquifer_water_fraction, MAX_WATER_FRACTION)

    for idx in eachindex(fractions)
        x, y, z = cell_centroid(volume, idx)
        if abs(x - args.aquifer_center_x) <= args.aquifer_half_x &&
           abs(y - args.aquifer_center_y) <= args.aquifer_half_y &&
           abs(z - args.aquifer_center_z) <= args.aquifer_half_z
            fractions[idx] = water_fraction
        end
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
    lo = max(0.0, w0 - delta)
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
        )
        flux_lo, sigma_lo = compute_directional_flux_mc(
            physics, matcfg, site, path, low_materials, low_densities, mc_samples,
            args.seed + case_idx;
            straggling = args.straggling,
            energy_threshold_low = args.energy_threshold_low,
        )
        flux_hi, sigma_hi = compute_directional_flux_mc(
            physics, matcfg, site, path, high_materials, high_densities, mc_samples,
            args.seed + case_idx;
            straggling = args.straggling,
            energy_threshold_low = args.energy_threshold_low,
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
    open(output_path, "w") do io
        println(io, "LVD tomography validation summary")
        println(io, "geometry = $(geometry_name(volume))")
        println(io, "n_cells = $(num_cells(volume))")
        println(io)

        for (idx, result) in enumerate(results)
            cx, cy, cz = cell_centroid(volume, result.cell_idx)
            flux_rel = 100 * abs(result.flux_mc - result.flux_csda) / max(abs(result.flux_csda), 1e-30)
            grad_rel = 100 * abs(result.grad_fd - result.grad_csda) / max(abs(result.grad_csda), 1e-30)
            println(io, "Case $idx: $(result.label)")
            println(io, @sprintf("  direction: theta=%.1f deg phi=%.1f deg", result.zenith, result.azimuth))
            println(io, @sprintf("  cell: %d at (%.1f, %.1f, %.1f) m", result.cell_idx, cx, cy, cz))
            println(io, @sprintf("  baseline water fraction: %.3f", result.w_base))
            println(io, @sprintf("  FD interval: [%.3f, %.3f]", result.w_lo, result.w_hi))
            println(io, @sprintf("  flux_csda: %.6e", result.flux_csda))
            println(io, @sprintf("  flux_mc:   %.6e +/- %.2e", result.flux_mc, result.sigma_mc))
            println(io, @sprintf("  flux mismatch: %.2f %%", flux_rel))
            println(io, @sprintf("  grad_csda: %.6e", result.grad_csda))
            println(io, @sprintf("  grad_fd:   %.6e", result.grad_fd))
            println(io, @sprintf("  grad mismatch: %.2f %%", grad_rel))
            println(io)
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
        max_plot_cells = 150,
        straggling = true,
        solver = "all",
        reg_weight = 0.0,
        contrast = 0.5,
        reco_iters = 200,
        threaded = true,
        correction = true,         # MC-calibrated CSDA fluctuation/hard-loss correction
        calibration_bins = 24,     # bins used to fit the correction to MC
        inverse_data = "mc",       # observation source: "mc" (full MC) or "csda"
        inverse_mc_samples = 256,  # MC samples per bin for the (low-noise) observations
        exposure = 1.0e8,          # detector exposure (sets Poisson noise level)
        edge_delta = 0.03,         # Huber transition for the edge-preserving GN prior
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
        elseif arg == "--no-threads"
            args = merge(args, (threaded = false,)); i += 1
        elseif arg == "--no-correction"
            args = merge(args, (correction = false,)); i += 1
        elseif arg == "--calibration-bins"
            args = merge(args, (calibration_bins = parse(Int, ARGS[i + 1]),)); i += 2
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

build_site() = SiteConfig(Float64(LVDTopo.DETECTOR_ELEVATION), Float64(primary_altitude_local()))

# --- MC-calibrated CSDA fluctuation/hard-loss correction (Part A) ---------

# Fit the differentiable CSDA correction to full MC on a stratified set of bins,
# then install it so the Jacobian and the inverse forward use the corrected
# operator. Returns (kappa_strag, kappa_hard, stats) or nothing when disabled.
function calibrate_correction_to_mc(physics, shallow_flags, matcfg::MaterialConfig,
                                    site::SiteConfig, paths::Vector{DirectionalPath},
                                    water_fractions::Vector{Float64},
                                    energy_samples::Vector{EnergySample}, args)
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

    mc_samples = sample_energy_set(args.mc_samples, args.energy_min, args.energy_max, args.seed + 20_000)
    mats, dens = build_cell_properties_for_mc(shallow_flags, water_fractions, matcfg)
    println("Calibrating CSDA correction to full MC on $(length(cbins)) bins (scattering+straggling)...")
    mcflux = Vector{Float64}(undef, length(cbins))
    Threads.@threads for k in eachindex(cbins)
        f, _ = compute_directional_flux_mc(physics, matcfg, site, paths[cbins[k]], mats, dens,
            mc_samples, args.seed + 30_000 + k;
            straggling = args.straggling, energy_threshold_low = args.energy_threshold_low,
            scattering = true)
        mcflux[k] = f
    end
    ks, kh, stats = calibrate_csda_correction(physics, shallow_flags, matcfg, site, paths,
        water_fractions, energy_samples, mcflux, cbins)
    println(@sprintf("  fitted kappa_strag=%.3f kappa_hard=%.3f | CSDA-vs-MC mean rel error %.1f%% -> %.1f%%",
                     ks, kh, 100 * stats.rel_before, 100 * stats.rel_after))
    return (ks, kh, stats)
end

# --- full-MC observations for the inverse problem (cached) ----------------

function mc_observations_for_field(physics, shallow_flags, matcfg::MaterialConfig,
                                   site::SiteConfig, paths::Vector{DirectionalPath},
                                   w_true::Vector{Float64}, valid::Vector{Int}, args;
                                   cache_path::String)
    sig = (length(paths), length(valid), args.inverse_mc_samples, args.seed, args.zenith_step_deg,
           args.azimuth_step_deg, args.geometry, hash(w_true), args.straggling,
           args.energy_min, args.energy_max, args.energy_threshold_low)
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
            model = w -> (Jv * w, Jv)
            w_rec, hist = gradient_descent_reconstruct(model, excess;
                w0 = zeros(n_cells), n_iter = args.reco_iters,
                lr = 0.01, optimizer = :adam, prior = sm_prior, relinearize = false)
        elseif s == "gn"
            # Our DiffPumas-native solver: box-constrained Gauss-Newton on the
            # true nonlinear corrected-CSDA operator (relinearised via the fast
            # AD Jacobian), inverse-variance weighted, with an edge-preserving prior.
            model = make_csda_operator(physics, shallow_flags, matcfg, site, paths,
                energy_samples; n_cells = n_cells, valid_bins = valid_bins,
                threaded = args.threaded)
            w_rec, hist = gauss_newton_reconstruct(model, obs;
                w0 = zeros(n_cells), n_iter = max(8, args.reco_iters ÷ 16),
                prior = edge_prior, weights = 1.0 ./ σ .^ 2, relinearize = true)
        else
            error("Unknown solver '$s' (use sart, mlem, gd, gn, or all)")
        end
        dt = time() - t0
        rep = reconstruction_report(w_rec, w_true)
        results[s] = (w = w_rec, hist = hist, report = rep, seconds = dt)
        println(@sprintf("  [%-4s] mse=%.3e rmse=%.3e psnr=%.1f dB ssim=%.3f  (%.1fs, %d iters)",
                         uppercase(s), rep.mse, rep.rmse, rep.psnr, rep.ssim, dt, length(hist)))
    end

    # --- resolution estimate (point-spread / FWHM vs depth) --------------
    # Linearised PSF using the baseline Jacobian: probe a few cells across depth.
    centroids = volume.centroids
    order = sortperm(aggregate, rev = true)
    probe = order[1:min(8, length(order))]
    recon_lin = wt -> first(sart_reconstruct(Jv, Jv * wt; n_iter = 120, relaxation = 0.2))
    zmin, zmax = extrema(centroids[3, :])
    depth_edges = collect(range(max(0.0, zmin), zmax; length = 5))
    rvd = resolution_vs_depth(recon_lin, centroids, zeros(n_cells), probe;
        depth_edges = depth_edges, contrast = args.contrast,
        zenith_step_deg = args.zenith_step_deg, cell_size_m = estimate_cell_size(volume))

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
        println(io, "Resolution (linearised point-spread FWHM vs depth):")
        println(io, @sprintf("  %-12s %-12s %-12s %-8s", "depth[m]", "FWHM[m]", "floor[m]", "n_cells"))
        for e in rvd
            println(io, @sprintf("  %-12.1f %-12.1f %-12.1f %-8d", e.depth_m, e.fwhm_m, e.floor_m, e.n_cells))
        end
        if isempty(rvd)
            println(io, "  (no depth bins populated)")
        end
    end

    println("Resolution vs depth (linearised PSF FWHM):")
    for e in rvd
        println(@sprintf("  depth=%.0f m -> FWHM=%.0f m (geometric floor %.0f m, %d cells)",
                         e.depth_m, e.fwhm_m, e.floor_m, e.n_cells))
    end

    return (results = results, w_true = w_true, resolution = rvd)
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
    println("  Baseline aquifer:  w=$(args.aquifer_water_fraction)")
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

    matcfg = MaterialConfig(
        rock_idx,
        water_idx,
        air_idx,
        porous_idx,
        Float64(physics.tables[rock_idx].density),
        Float64(physics.tables[water_idx].density),
        porous_idx > 0 ? Float64(physics.tables[porous_idx].density) : 0.0,
        args.porous_top_thickness,
    )

    println("Building / loading LVD topography...")
    emap = LVDTopo.build_elevation_map()
    println("  Detector elevation: $(LVDTopo.DETECTOR_ELEVATION) m ASL")
    println("  Primary altitude:   $(primary_altitude_local()) m above detector")
    println()

    volume = create_sensitivity_volume(emap, matcfg, args)
    println("Constructed $(geometry_name(volume)) sensitivity volume with $(num_cells(volume)) cells")
    println()

    site = build_site()
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
