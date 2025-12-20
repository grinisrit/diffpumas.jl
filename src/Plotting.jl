"""
    Plotting

Visualization utilities for DiffPumas trajectory simulations using PlotlyJS.
"""
module Plotting

using PlotlyJS
using ..Types: Vec3, State

export plot_trajectories, plot_transport_path

# Define Point type for trajectories (reuse Vec3 or define simple struct)
const Point3D = Vec3{Float64}

"""
    plot_trajectories(mesh, cell_values, trajectories, output_file; n_trajectories=10, gradients=nothing)

Create an interactive 3D PlotlyJS plot showing:
- The cube with tetrahedral mesh
- Cell values as a transparent heatmap
- Gradient values as a separate heatmap (if provided)
- Trajectory paths (up to n_trajectories)

Arguments:
- `mesh`: Mesh object with pointlist and tetrahedronlist
- `cell_values`: Vector of cell values for heatmap coloring
- `trajectories`: Vector of trajectory data (list of points)
- `output_file`: Path to save the HTML plot file
- `n_trajectories`: Number of trajectories to display (default: 10)
- `gradients`: Optional vector of gradient values for each cell (default: nothing)
"""
function plot_trajectories(
    mesh,
    cell_values::Vector{Float64},
    trajectories::Vector,
    output_file::String;
    n_trajectories::Int = 10,
    gradients::Union{Vector{Float64}, Nothing} = nothing
)
    traces = PlotlyJS.GenericTrace[]
    
    # Check if mesh has required fields
    if !hasproperty(mesh, :tetrahedronlist) || !hasproperty(mesh, :pointlist)
        @warn "Mesh must have tetrahedronlist and pointlist fields"
        return nothing
    end
    
    # 1. Plot tetrahedral mesh as wireframe
    edges_set = Set{Tuple{Int, Int}}()
    n_tets = size(mesh.tetrahedronlist, 2)
    points = mesh.pointlist
    n_points = size(points, 2)
    
    for i in 1:n_tets
        tet = mesh.tetrahedronlist[:, i] .+ 1
        v1, v2, v3, v4 = tet
        
        if all([v1, v2, v3, v4] .<= n_points) && all([v1, v2, v3, v4] .>= 1)
            for edge in [
                (min(v1,v2), max(v1,v2)), (min(v1,v3), max(v1,v3)), (min(v1,v4), max(v1,v4)),
                (min(v2,v3), max(v2,v3)), (min(v2,v4), max(v2,v4)), (min(v3,v4), max(v3,v4))
            ]
                push!(edges_set, edge)
            end
        end
    end
    
    # Create edge traces
    for (v1, v2) in edges_set
        if v1 <= n_points && v2 <= n_points && v1 >= 1 && v2 >= 1
            x_edge = [points[1, v1], points[1, v2], nothing]
            y_edge = [points[2, v1], points[2, v2], nothing]
            z_edge = [points[3, v1], points[3, v2], nothing]
            
            push!(traces, scatter3d(
                x = x_edge,
                y = y_edge,
                z = z_edge,
                mode = "lines",
                line = attr(width = 1, color = "rgba(100, 100, 100, 0.3)"),
                showlegend = false,
                hoverinfo = "skip"
            ))
        end
    end
    
    # 2. Plot cell values
    valid_cell_values = cell_values[1:min(n_tets, length(cell_values))]
    cell_min = length(valid_cell_values) > 0 ? minimum(valid_cell_values) : 0.0
    cell_max = length(valid_cell_values) > 0 ? maximum(valid_cell_values) : 1.0
    
    for i in 1:min(n_tets, length(cell_values))
        tet = mesh.tetrahedronlist[:, i] .+ 1
        v1, v2, v3, v4 = tet
        
        if !all([v1, v2, v3, v4] .<= n_points) || !all([v1, v2, v3, v4] .>= 1)
            continue
        end
        
        x_coords = [points[1, v1], points[1, v2], points[1, v3], points[1, v4]]
        y_coords = [points[2, v1], points[2, v2], points[2, v3], points[2, v4]]
        z_coords = [points[3, v1], points[3, v2], points[3, v3], points[3, v4]]
        
        i_faces = Int32[0, 0, 0, 1]
        j_faces = Int32[1, 1, 2, 2]
        k_faces = Int32[2, 3, 3, 3]
        
        cell_val = cell_values[i]
        
        push!(traces, mesh3d(
            x = x_coords,
            y = y_coords,
            z = z_coords,
            i = i_faces,
            j = j_faces,
            k = k_faces,
            intensity = [cell_val, cell_val, cell_val, cell_val],
            colorscale = "Viridis",
            opacity = 0.15,
            showscale = (i == 1),
            colorbar = (i == 1) ? attr(title = "Cell Value", len = 0.5, x = 1.02) : nothing,
            cmin = cell_min,
            cmax = cell_max,
            name = i == 1 ? "Cell Values" : "",
            showlegend = (i == 1)
        ))
    end
    
    # 3. Plot trajectories
    colors = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "yellow", "pink", "brown",
        "lime", "navy", "olive", "teal", "maroon"
    ]
    
    n_to_plot = min(n_trajectories, length(trajectories))
    
    for (idx, trajectory) in enumerate(trajectories[1:n_to_plot])
        if length(trajectory) < 2
            continue
        end
        
        # Handle different trajectory types
        x_traj = Float64[]
        y_traj = Float64[]
        z_traj = Float64[]
        
        for p in trajectory
            if isa(p, Vec3)
                push!(x_traj, p[1])
                push!(y_traj, p[2])
                push!(z_traj, p[3])
            elseif hasproperty(p, :x)
                push!(x_traj, p.x)
                push!(y_traj, p.y)
                push!(z_traj, p.z)
            elseif length(p) >= 3
                push!(x_traj, p[1])
                push!(y_traj, p[2])
                push!(z_traj, p[3])
            end
        end
        
        color = colors[mod1(idx, length(colors))]
        
        push!(traces, scatter3d(
            x = x_traj,
            y = y_traj,
            z = z_traj,
            mode = "lines+markers",
            line = attr(width = 2, color = color),
            marker = attr(size = 3, color = color),
            name = "Trajectory $idx"
        ))
    end
    
    # 4. Create layout
    layout = Layout(
        title = attr(text = "Tetrahedral Mesh with Trajectories", font = attr(size = 20)),
        scene = attr(
            xaxis = attr(title = "X"),
            yaxis = attr(title = "Y"),
            zaxis = attr(title = "Z"),
            aspectmode = "cube"
        ),
        width = 1000,
        height = 800
    )
    
    # 5. Create plot and save
    p = Plot(traces, layout)
    savefig(p, output_file)
    
    println("✓ Plot saved to: $output_file")
    return p
end

"""
    plot_transport_path(states, output_file; title="Particle Transport Path")

Plot a particle transport path from a sequence of states.

Arguments:
- `states`: Vector of State objects
- `output_file`: Path to save the HTML plot
- `title`: Plot title
"""
function plot_transport_path(
    states::Vector{State{T}},
    output_file::String;
    title::String = "Particle Transport Path"
) where T<:Real
    
    traces = PlotlyJS.GenericTrace[]
    
    # Extract positions
    x = [s.position[1] for s in states]
    y = [s.position[2] for s in states]
    z = [s.position[3] for s in states]
    
    # Energy for coloring
    energies = [s.energy for s in states]
    
    # Main trajectory
    push!(traces, scatter3d(
        x = x,
        y = y,
        z = z,
        mode = "lines+markers",
        line = attr(width = 4, color = energies, colorscale = "Viridis"),
        marker = attr(size = 4, color = energies, colorscale = "Viridis",
                     colorbar = attr(title = "Energy (GeV)")),
        name = "Transport Path"
    ))
    
    # Start point
    push!(traces, scatter3d(
        x = [x[1]],
        y = [y[1]],
        z = [z[1]],
        mode = "markers",
        marker = attr(size = 10, color = "green", symbol = "diamond"),
        name = "Start"
    ))
    
    # End point
    push!(traces, scatter3d(
        x = [x[end]],
        y = [y[end]],
        z = [z[end]],
        mode = "markers",
        marker = attr(size = 10, color = "red", symbol = "square"),
        name = "End"
    ))
    
    layout = Layout(
        title = attr(text = title, font = attr(size = 20)),
        scene = attr(
            xaxis = attr(title = "X (m)"),
            yaxis = attr(title = "Y (m)"),
            zaxis = attr(title = "Z (m)"),
            aspectmode = "data"
        ),
        width = 1000,
        height = 800
    )
    
    p = Plot(traces, layout)
    savefig(p, output_file)
    
    println("✓ Transport path plot saved to: $output_file")
    return p
end

end # module
