"""
    DiffPumas

A Julia package for automatic differentiation using Zygote.jl.
"""
module DiffPumas

using PlotlyJS
using TetGen
using TriangleIntersect
using Zygote

# Export Zygote functionality
export gradient, pullback, jacobian

# Export TriangleIntersect functionality for ray-triangle intersections
export Point, Ray, Triangle, Intersection, intersect

# Export TetGen functionality for tetrahedral meshing
export tetrahedralize, RawTetGenIO, facetlist!

# Re-export commonly used PlotlyJS functions for convenient plotting
export plot, Plot, Layout, scatter, scatter3d, bar, heatmap, surface

# Include submodules
include("Plotting.jl")
using .Plotting
export plot_trajectories

# Package code goes here

end # module

