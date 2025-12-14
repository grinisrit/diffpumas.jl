"""
    Plotting

Visualization utilities for DiffPumas trajectory simulations using PlotlyJS.
"""
module Plotting

using DiffPumas: Point, RawTetGenIO
using PlotlyJS

export plot_trajectories

"""
    plot_trajectories(mesh, cell_values, trajectories, output_file; n_trajectories=10, gradients=nothing)

Create an interactive 3D PlotlyJS plot showing:
- The cube with tetrahedral mesh
- Cell values as a transparent heatmap
- Gradient values as a separate heatmap (if provided)
- Trajectory paths (up to n_trajectories)

Arguments:
- `mesh`: RawTetGenIO mesh object
- `cell_values`: Vector of cell values for heatmap coloring
- `trajectories`: Vector of trajectory data (list of points)
- `output_file`: Path to save the HTML plot file
- `n_trajectories`: Number of trajectories to display (default: 10)
- `gradients`: Optional vector of gradient values for each cell (default: nothing)
"""
function plot_trajectories(
    mesh,
    cell_values::Vector{Float64},
    trajectories::Vector{Vector{Point}},
    output_file::String;
    n_trajectories::Int = 10,
    gradients::Union{Vector{Float64}, Nothing} = nothing
)
    traces = PlotlyJS.GenericTrace[]
    
    # 1. Plot tetrahedral mesh as wireframe
    # Extract edges from tetrahedra
    edges_set = Set{Tuple{Int, Int}}()
    n_tets = size(mesh.tetrahedronlist, 2)
    points = mesh.pointlist  # 3×N matrix
    n_points = size(points, 2)
    
    for i in 1:n_tets
        tet = mesh.tetrahedronlist[:, i] .+ 1  # Convert to 1-indexed
        v1, v2, v3, v4 = tet
        
        # Check bounds before adding edges
        if all([v1, v2, v3, v4] .<= n_points) && all([v1, v2, v3, v4] .>= 1)
            # Add edges (each edge only once)
            for edge in [
                (min(v1,v2), max(v1,v2)), (min(v1,v3), max(v1,v3)), (min(v1,v4), max(v1,v4)),
                (min(v2,v3), max(v2,v3)), (min(v2,v4), max(v2,v4)), (min(v3,v4), max(v3,v4))
            ]
                push!(edges_set, edge)
            end
        end
    end
    
    # Create edge traces for mesh visualization
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
    
    # 2. Plot cell values as filled tetrahedral volumes
    # Create individual mesh3d trace for each tetrahedron to properly fill volumes
    
    # First, collect all cell values for colorbar range
    valid_cell_values = Float64[]
    for i in 1:min(n_tets, length(cell_values))
        push!(valid_cell_values, cell_values[i])
    end
    cell_min = length(valid_cell_values) > 0 ? minimum(valid_cell_values) : 0.0
    cell_max = length(valid_cell_values) > 0 ? maximum(valid_cell_values) : 1.0
    
    # Create mesh3d trace for each tetrahedron
    for i in 1:min(n_tets, length(cell_values))
        tet = mesh.tetrahedronlist[:, i] .+ 1
        v1, v2, v3, v4 = tet
        
        # Check bounds
        if !all([v1, v2, v3, v4] .<= n_points) || !all([v1, v2, v3, v4] .>= 1)
            continue
        end
        
        # Get vertex coordinates for this tetrahedron
        x_coords = [points[1, v1], points[1, v2], points[1, v3], points[1, v4]]
        y_coords = [points[2, v1], points[2, v2], points[2, v3], points[2, v4]]
        z_coords = [points[3, v1], points[3, v2], points[3, v3], points[3, v4]]
        
        # Define 4 triangular faces of the tetrahedron (0-indexed)
        # Face 1: vertices 0, 1, 2
        # Face 2: vertices 0, 1, 3
        # Face 3: vertices 0, 2, 3
        # Face 4: vertices 1, 2, 3
        i_faces = Int32[0, 0, 0, 1]
        j_faces = Int32[1, 1, 2, 2]
        k_faces = Int32[2, 3, 3, 3]
        
        # Use intensity to color the entire tetrahedron
        cell_val = cell_values[i]
        
        push!(traces, mesh3d(
            x = x_coords,
            y = y_coords,
            z = z_coords,
            i = i_faces,
            j = j_faces,
            k = k_faces,
            intensity = [cell_val, cell_val, cell_val, cell_val],  # Same intensity for all vertices
            colorscale = "Viridis",
            opacity = 0.15,  # Very transparent for better visibility of trajectories
            showscale = (i == 1),  # Only show colorbar on first trace
            colorbar = (i == 1) ? attr(title = "Cell Value", len = 0.5, x = 1.02) : nothing,
            cmin = cell_min,
            cmax = cell_max,
            name = i == 1 ? "Cell Values" : "",
            showlegend = (i == 1),
            hovertemplate = i == 1 ? "Cell Value: %{intensity}<br><extra></extra>" : "skip"
        ))
    end
    
    # 2b. Plot gradients as filled tetrahedral volumes (if provided)
    if gradients !== nothing
        # Collect gradient values for colorbar range
        valid_grad_values = Float64[]
        for i in 1:min(n_tets, length(gradients))
            push!(valid_grad_values, abs(gradients[i]))
        end
        grad_min = length(valid_grad_values) > 0 ? minimum(valid_grad_values) : 0.0
        grad_max = length(valid_grad_values) > 0 ? maximum(valid_grad_values) : 1.0
        
        # Create mesh3d trace for each tetrahedron with gradient coloring
        for i in 1:min(n_tets, length(gradients))
            tet = mesh.tetrahedronlist[:, i] .+ 1
            v1, v2, v3, v4 = tet
            
            # Check bounds
            if !all([v1, v2, v3, v4] .<= n_points) || !all([v1, v2, v3, v4] .>= 1)
                continue
            end
            
            # Get vertex coordinates for this tetrahedron
            x_coords = [points[1, v1], points[1, v2], points[1, v3], points[1, v4]]
            y_coords = [points[2, v1], points[2, v2], points[2, v3], points[2, v4]]
            z_coords = [points[3, v1], points[3, v2], points[3, v3], points[3, v4]]
            
            # Define 4 triangular faces of the tetrahedron (0-indexed)
            i_faces = Int32[0, 0, 0, 1]
            j_faces = Int32[1, 1, 2, 2]
            k_faces = Int32[2, 3, 3, 3]
            
            # Use absolute gradient value for coloring
            abs_grad = abs(gradients[i])
            
            push!(traces, mesh3d(
                x = x_coords,
                y = y_coords,
                z = z_coords,
                i = i_faces,
                j = j_faces,
                k = k_faces,
                intensity = [abs_grad, abs_grad, abs_grad, abs_grad],
                colorscale = "Hot",
                opacity = 0.25,  # More transparent so cell values and trajectories show through
                showscale = (i == 1),
                colorbar = (i == 1) ? attr(title = "|Gradient|", len = 0.5, x = 1.15, y = 0.5) : nothing,
                cmin = grad_min,
                cmax = grad_max,
                name = i == 1 ? "Gradients" : "",
                showlegend = (i == 1),
                hovertemplate = i == 1 ? "|Gradient|: %{intensity}<br><extra></extra>" : "skip"
            ))
        end
    end
    
    # 3. Plot trajectories
    # Extended color palette for more trajectories
    colors = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "yellow", "pink", "brown",
        "lime", "navy", "olive", "teal", "maroon",
        "aqua", "fuchsia", "silver", "gray", "black",
        "coral", "salmon", "gold", "khaki", "plum",
        "turquoise", "lavender", "tan", "ivory", "crimson",
        "indigo", "chocolate", "darkgreen", "darkblue", "darkred",
        "darkorange", "darkviolet", "deeppink", "deepskyblue", "dodgerblue",
        "forestgreen", "hotpink", "lightblue", "lightgreen", "lightgray",
        "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "mediumblue"
    ]
    
    n_to_plot = min(n_trajectories, length(trajectories))
    
    for (idx, trajectory) in enumerate(trajectories[1:n_to_plot])
        if length(trajectory) < 2
            continue
        end
        
        x_traj = [p.x for p in trajectory]
        y_traj = [p.y for p in trajectory]
        z_traj = [p.z for p in trajectory]
        
        color = colors[mod1(idx, length(colors))]
        
        push!(traces, scatter3d(
            x = x_traj,
            y = y_traj,
            z = z_traj,
            mode = "lines+markers",
            line = attr(width = 2, color = color),  # Slightly thinner for more trajectories
            marker = attr(size = 3, color = color),  # Slightly smaller markers
            name = "Trajectory $idx",
            hovertemplate = "Trajectory $idx<br>" *
                          "X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
            showlegend = idx <= 20  # Only show first 20 in legend to avoid clutter
        ))
    end
    
    # 4. Create layout
    layout = Layout(
        title = attr(
            text = "Tetrahedral Mesh with Trajectories",
            font = attr(size = 20)
        ),
        scene = attr(
            xaxis = attr(title = "X", range = [0, 1]),
            yaxis = attr(title = "Y", range = [0, 1]),
            zaxis = attr(title = "Z", range = [0, 1]),
            aspectmode = "cube",
            camera = attr(
                eye = attr(x = 1.5, y = 1.5, z = 1.5)
            )
        ),
        width = 1000,
        height = 800,
        showlegend = true
    )
    
    # 5. Create plot and save
    p = Plot(traces, layout)
    
    # Save to HTML file
    savefig(p, output_file)
    
    println("✓ Plot saved to: $output_file")
    
    return p
end

end # module

