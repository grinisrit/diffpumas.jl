#!/usr/bin/env julia
"""
export_lvd_topography.jl — dump the Gran Sasso / LVD elevation grid to CSV.

Reuses the topography reconstruction in `lvd_muography.jl` (build_elevation_map,
terrain_elevation_augmented) so the paper's 3D topography figure can be rendered
statically with matplotlib without re-running any transport.  Output:

    examples/data/lvd_results/lvd_topography_grid.csv   (x_km, y_km, elev_m)

The detector sits at the local origin (x=y=0) at DETECTOR_ELEVATION m a.s.l.
Run:  julia --project=. examples/export_lvd_topography.jl
"""

# Pull in the topography helpers exactly the way lvd_tomography.jl does, inside a
# private module so the script's main() guard stays inert on include.
module LVDTopo
include(joinpath(@__DIR__, "lvd_muography.jl"))
end

using Printf

function main()
    println("Building LVD elevation map from nm_c.inc + cross-section geometry…")
    emap = LVDTopo.build_elevation_map()

    out_dir = joinpath(@__DIR__, "data", "lvd_results")
    isdir(out_dir) || mkpath(out_dir)
    out_path = joinpath(out_dir, "lvd_topography_grid.csv")

    open(out_path, "w") do io
        println(io, "x_km,y_km,elev_m")
        for iy in 0:(emap.ny - 1)
            y_km = emap.y0 + iy * emap.dy
            for ix in 0:(emap.nx - 1)
                x_km = emap.x0 + ix * emap.dx
                z_m  = emap.data[iy + 1, ix + 1]
                println(io, @sprintf("%.4f,%.4f,%.2f", x_km, y_km, z_m))
            end
        end
    end

    zmin, zmax = extrema(emap.data)
    println(@sprintf("Wrote %s  (%d×%d nodes, elev %.0f–%.0f m, detector at %.0f m a.s.l.)",
                     out_path, emap.nx, emap.ny, zmin, zmax, LVDTopo.DETECTOR_ELEVATION))
    return 0
end

main()
