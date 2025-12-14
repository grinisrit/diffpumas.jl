"""
Monte Carlo Trajectory Simulation with Automatic Differentiation

This example demonstrates:
1. Creating a unit cube and generating a tetrahedral mesh with TetGen.jl
2. Assigning random values to each tetrahedron cell (differentiable parameters)
3. Simulating particle trajectories through the mesh using TriangleIntersect.jl
4. Computing the mean weighted path length over multiple trajectories
5. Using Zygote.jl to compute sensitivities (gradients) of the mean with respect to cell values
"""

using DiffPumas
using Random
using LinearAlgebra
using Statistics
using ZygoteRules
using ArgParse
import Printf: @sprintf

"""
    compute_cell_values_from_source(mesh, source_point)

Compute cell values based on distance from source point.
Cells at the source (x=0) have value 0,
cells further away have larger values (close to 1).
"""
function compute_cell_values_from_source(mesh::RawTetGenIO{Float64}, source_point::Point)
    n_tets = size(mesh.tetrahedronlist, 2)
    points = mesh.pointlist
    n_points = size(points, 2)
    cell_values = zeros(Float64, n_tets)
    
    # Threshold for considering a cell "at the source" (very close to x=0)
    source_threshold = 1e-6
    
    for i in 1:n_tets
        tet = mesh.tetrahedronlist[:, i] .+ 1  # Convert to 1-indexed
        v1, v2, v3, v4 = tet
        
        # Check bounds
        if all([v1, v2, v3, v4] .<= n_points) && all([v1, v2, v3, v4] .>= 1)
            # Compute tetrahedron center
            center = [
                (points[1, v1] + points[1, v2] + points[1, v3] + points[1, v4]) / 4,
                (points[2, v1] + points[2, v2] + points[2, v3] + points[2, v4]) / 4,
                (points[3, v1] + points[3, v2] + points[3, v3] + points[3, v4]) / 4
            ]
            
            # Use x-coordinate to determine value (since source is at x=0)
            # Cells at source (x < threshold) have value 0
            # Value increases linearly from 0 to 1 for x > threshold
            x_coord = center[1]
            if x_coord <= source_threshold
                cell_values[i] = 0.0
            else
                # Normalize: map [source_threshold, 1.0] to [0.0, 1.0]
                cell_values[i] = clamp((x_coord - source_threshold) / (1.0 - source_threshold), 0.0, 1.0)
            end
        end
    end
    
    return cell_values
end

"""
    create_unit_cube_mesh(λ)

Create a tetrahedral mesh of a unit cube using TetGen.jl.
λ controls the mesh density (smaller λ = finer mesh).
"""
function create_unit_cube_mesh(λ::Float64)
    # Create a unit cube with vertices
    # Bottom face (z=0)
    points = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        # Top face (z=1)
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]
    ]
    
    # Define the 6 faces of the cube as facets
    # Each face is defined by 4 vertices (using 1-indexed references)
    facets = [
        [1, 2, 3, 4],  # Bottom face (z=0)
        [5, 6, 7, 8],  # Top face (z=1)
        [1, 2, 6, 5],  # Front face (y=0)
        [2, 3, 7, 6],  # Right face (x=1)
        [3, 4, 8, 7],  # Back face (y=1)
        [4, 1, 5, 8]   # Left face (x=0)
    ]
    
    # Create TetGen input structure
    input = RawTetGenIO{Float64}()
    
    # Convert points to column-major format (3×N matrix)
    input.pointlist = hcat(points...)  # Each column is a point
    
    # Use TetGen's facetlist! helper function
    # facets matrix: each column is a facet, rows are vertex indices (1-indexed)
    facet_matrix = hcat(facets...)
    facetlist!(input, facet_matrix)
    
    # Tetrahedralize with quality constraint controlled by λ
    # 'p' = PLC (piecewise linear complex)
    # 'q' = quality mesh generation
    # 'a' = maximum tetrahedron volume = λ
    # 'v' = verbose
    quality = string("pq", "a", λ)
    mesh_result = tetrahedralize(input, quality)
    
    return mesh_result
end

"""
    get_tetrahedron_faces(tetrahedron, points)

Get the 4 triangular faces of a tetrahedron.
Each face is defined by 3 vertices (as Points for TriangleIntersect).
"""
function get_tetrahedron_faces(tetrahedron, points)
    # tetrahedron is a 4-element vector of point indices (0-indexed from TetGen)
    # points is a 3×N matrix (each column is a point)
    
    v1, v2, v3, v4 = tetrahedron .+ 1  # Convert to 1-indexed
    
    p1 = Point(points[1, v1], points[2, v1], points[3, v1])
    p2 = Point(points[1, v2], points[2, v2], points[3, v2])
    p3 = Point(points[1, v3], points[2, v3], points[3, v3])
    p4 = Point(points[1, v4], points[2, v4], points[3, v4])
    
    # Create 4 triangular faces (each opposite to one vertex)
    faces = [
        Triangle(p1, p2, p3),  # Opposite v4
        Triangle(p1, p2, p4),  # Opposite v3
        Triangle(p1, p3, p4),  # Opposite v2
        Triangle(p2, p3, p4)   # Opposite v1
    ]
    
    return faces
end

"""
    point_in_cube(p)

Check if point p is inside the unit cube [0,1]³.
"""
function point_in_cube(p::Point)
    return 0.0 <= p.x <= 1.0 && 0.0 <= p.y <= 1.0 && 0.0 <= p.z <= 1.0
end

"""
    find_intersection_with_cube_boundary(ray)

Find the intersection of a ray with the unit cube boundary.
Returns the intersection point and distance, or nothing if no intersection.
"""
function find_intersection_with_cube_boundary(ray::Ray)
    # Cube faces as triangles - all 12 triangles (6 faces × 2 triangles each)
    cube_faces = [
        # Bottom face (z=0) - 2 triangles
        Triangle(Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0)),
        Triangle(Point(0, 0, 0), Point(1, 1, 0), Point(0, 1, 0)),
        # Top face (z=1) - 2 triangles
        Triangle(Point(0, 0, 1), Point(1, 0, 1), Point(1, 1, 1)),
        Triangle(Point(0, 0, 1), Point(1, 1, 1), Point(0, 1, 1)),
        # Front face (y=0) - 2 triangles
        Triangle(Point(0, 0, 0), Point(1, 0, 0), Point(1, 0, 1)),
        Triangle(Point(0, 0, 0), Point(1, 0, 1), Point(0, 0, 1)),
        # Back face (y=1) - 2 triangles
        Triangle(Point(0, 1, 0), Point(1, 1, 1), Point(1, 1, 0)),
        Triangle(Point(0, 1, 0), Point(0, 1, 1), Point(1, 1, 1)),
        # Left face (x=0) - 2 triangles
        Triangle(Point(0, 0, 0), Point(0, 1, 0), Point(0, 1, 1)),
        Triangle(Point(0, 0, 0), Point(0, 1, 1), Point(0, 0, 1)),
        # Right face (x=1) - 2 triangles
        Triangle(Point(1, 0, 0), Point(1, 1, 1), Point(1, 1, 0)),
        Triangle(Point(1, 0, 0), Point(1, 0, 1), Point(1, 1, 1)),
    ]
    
    best_intersection = nothing
    min_distance = Inf
    
    for face in cube_faces
        intersection = intersect(ray, face)
        if intersection.is_intersection && intersection.id < min_distance && intersection.id > 1e-10
            min_distance = intersection.id
            best_intersection = intersection
        end
    end
    
    return best_intersection
end

"""
    trace_trajectory(start_point, direction, mesh, cell_values, rng)

Trace a particle trajectory through the tetrahedral mesh.
At each cell boundary, the direction is randomly changed (scattering).
Returns the weighted path length sum: Σ(cell_value × segment_length).
"""
function trace_trajectory(
    start_point::Point,
    direction::Vector{Float64},
    mesh::RawTetGenIO{Float64},
    cell_values::Vector{Float64},
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    current_point = start_point
    direction = direction / norm(direction)  # Normalize
    
    total_weighted_length = 0.0
    
    # Maximum iterations to prevent infinite loops
    max_iterations = 1000
    iteration = 0
    
    while point_in_cube(current_point) && iteration < max_iterations
        iteration += 1
        
        # Create ray from current point
        end_point = Point(
            current_point.x + direction[1] * 100.0,  # Large distance
            current_point.y + direction[2] * 100.0,
            current_point.z + direction[3] * 100.0
        )
        ray = Ray(current_point, end_point)
        
        # Find intersection with cube boundary (exit point)
        boundary_intersection = find_intersection_with_cube_boundary(ray)
        
        if boundary_intersection === nothing
            break  # No intersection found, exit loop
        end
        
        exit_point = boundary_intersection.ip
        exit_distance = boundary_intersection.id
        
            # Find which tetrahedron contains the current point
        current_tet_idx = find_tetrahedron_containing_point(current_point, mesh)
        
        if current_tet_idx === nothing
            # Point not in any tetrahedron, move forward a small amount
            current_point = Point(
                current_point.x + direction[1] * 0.01,
                current_point.y + direction[2] * 0.01,
                current_point.z + direction[3] * 0.01
            )
            continue
        end
        
        # Find next intersection with a tetrahedron face
        tet_faces = get_tetrahedron_faces(
            mesh.tetrahedronlist[:, current_tet_idx],  # current_tet_idx is already 1-indexed
            mesh.pointlist  # Use output pointlist
        )
        
        closest_face = nothing
        closest_distance = Inf
        
        for face in tet_faces
            intersection = intersect(ray, face)
            if intersection.is_intersection && 
               intersection.id > 1e-10 &&  # Avoid self-intersection
               intersection.id < closest_distance &&
               intersection.id < exit_distance
                closest_distance = intersection.id
                closest_face = face
            end
        end
        
        # Calculate segment length - this is the differentiable part
        if closest_face !== nothing && closest_distance < exit_distance
            # Segment is inside current tetrahedron
            segment_length = closest_distance
            # Only this multiplication is differentiable - current_tet_idx is determined by geometry
            total_weighted_length += cell_values[current_tet_idx] * segment_length
            
            # Move to intersection point
            intersection = intersect(ray, closest_face)
            current_point = intersection.ip
            
            # Randomly change direction at cell boundary (scattering)
            # Use more aggressive scattering - random direction on full sphere
            new_direction = randn(rng, 3)
            direction = new_direction / norm(new_direction)  # Normalize to unit vector
        else
            # Segment to exit point
            segment_length = exit_distance
            total_weighted_length += cell_values[current_tet_idx] * segment_length
            break  # Exited the cube
        end
    end
    
    return total_weighted_length
end

"""
    find_tetrahedron_containing_point(p, mesh)

Find which tetrahedron contains point p.
Returns the tetrahedron index (1-indexed) or nothing.
"""
function find_tetrahedron_containing_point(p::Point, mesh::RawTetGenIO{Float64})
    # Simple brute force search - could be optimized with spatial indexing
    # Use output pointlist which has all refined points
    output_points = mesh.pointlist
    n_tets = size(mesh.tetrahedronlist, 2)
    
    for i in 1:n_tets
        tet = mesh.tetrahedronlist[:, i]
        if point_in_tetrahedron(p, tet, output_points)
            return i  # 1-indexed
        end
    end
    
    return nothing
end

"""
    point_in_tetrahedron(p, tet, points)

Check if point p is inside tetrahedron tet using barycentric coordinates.
"""
function point_in_tetrahedron(p::Point, tet, points)
    # Get tetrahedron vertices (TetGen uses 0-indexed, Julia uses 1-indexed)
    v1_idx, v2_idx, v3_idx, v4_idx = tet .+ 1  # Convert to 1-indexed
    
    # Check bounds
    n_points = size(points, 2)
    if any([v1_idx, v2_idx, v3_idx, v4_idx] .> n_points) || any([v1_idx, v2_idx, v3_idx, v4_idx] .< 1)
        return false
    end
    
    v1 = [points[1, v1_idx], points[2, v1_idx], points[3, v1_idx]]
    v2 = [points[1, v2_idx], points[2, v2_idx], points[3, v2_idx]]
    v3 = [points[1, v3_idx], points[2, v3_idx], points[3, v3_idx]]
    v4 = [points[1, v4_idx], points[2, v4_idx], points[3, v4_idx]]
    
    p_vec = [p.x, p.y, p.z]
    
    # Compute barycentric coordinates
    T = hcat(v2 - v1, v3 - v1, v4 - v1)
    b = p_vec - v1
    
    try
        λ = T \ b
        # Point is inside if all barycentric coordinates are positive and sum <= 1
        return all(λ .>= -1e-10) && sum(λ) <= 1.0 + 1e-10
    catch
        return false  # Singular matrix
    end
end

"""
    compute_weighted_path_length(cell_lengths, cell_values)

Compute weighted path length from precomputed segment lengths per cell.
This is the differentiable part.
"""
function compute_weighted_path_length(cell_lengths::Vector{Float64}, cell_values::Vector{Float64})
    return sum(cell_lengths .* cell_values)
end

"""
    simulate_trajectories(N, start_point, mesh, cell_values, rng; scatter_prob=0.7)

Simulate N trajectories and return the mean weighted path length.
For differentiation, we precompute path segments, then compute weighted sum.
"""
function simulate_trajectories(
    N::Int,
    start_point::Point,
    mesh::RawTetGenIO{Float64},
    cell_values::Vector{Float64},
    rng::AbstractRNG = Random.GLOBAL_RNG;
    scatter_prob::Float64 = 0.7
)
    n_tets = size(mesh.tetrahedronlist, 2)
    total_cell_lengths = zeros(Float64, n_tets)
    
    # Precompute all trajectory segments (non-differentiable geometry)
    for _ in 1:N
        # Generate random direction (uniform on sphere)
        direction = randn(rng, 3)
        direction = direction / norm(direction)
        
        # Trace trajectory and accumulate path lengths per cell
        cell_lengths = trace_trajectory_segments(start_point, direction, mesh, rng=rng, scatter_prob=scatter_prob)
        total_cell_lengths .+= cell_lengths
    end
    
    # Average over trajectories
    avg_cell_lengths = total_cell_lengths / N
    
    # Compute weighted sum (this is differentiable w.r.t. cell_values)
    return compute_weighted_path_length(avg_cell_lengths, cell_values)
end

"""
    trace_trajectory_segments(start_point, direction, mesh; collect_points=false, rng, scatter_prob=0.7)

Trace a trajectory and return path length in each cell.
At each cell boundary, the direction is randomly changed (scattering).
Returns a vector of path lengths, one per tetrahedron.
If collect_points=true, also returns a vector of points along the trajectory.
scatter_prob controls the probability of additional scattering events (0.0-1.0, higher = more scattering).
"""
function trace_trajectory_segments(
    start_point::Point,
    direction::Vector{Float64},
    mesh::RawTetGenIO{Float64};
    collect_points::Bool = false,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    scatter_prob::Float64 = 0.7
)
    n_tets = size(mesh.tetrahedronlist, 2)
    cell_lengths = zeros(Float64, n_tets)
    trajectory_points = collect_points ? Point[start_point] : Point[]
    
    current_point = start_point
    direction = direction / norm(direction)  # Normalize
    
    # Maximum iterations to prevent infinite loops
    max_iterations = 1000
    iteration = 0
    
    while point_in_cube(current_point) && iteration < max_iterations
        iteration += 1
        
        # Create ray from current point
        end_point = Point(
            current_point.x + direction[1] * 100.0,  # Large distance
            current_point.y + direction[2] * 100.0,
            current_point.z + direction[3] * 100.0
        )
        ray = Ray(current_point, end_point)
        
        # Find intersection with cube boundary (exit point)
        boundary_intersection = find_intersection_with_cube_boundary(ray)
        
        if boundary_intersection === nothing
            break  # No intersection found, exit loop
        end
        
        exit_point = boundary_intersection.ip
        exit_distance = boundary_intersection.id
        
        # Find which tetrahedron contains the current point
        current_tet_idx = find_tetrahedron_containing_point(current_point, mesh)
        
        if current_tet_idx === nothing
            # Point not in any tetrahedron, move forward a small amount
            current_point = Point(
                current_point.x + direction[1] * 0.01,
                current_point.y + direction[2] * 0.01,
                current_point.z + direction[3] * 0.01
            )
            continue
        end
        
        # Find next intersection with a tetrahedron face
        tet_faces = get_tetrahedron_faces(
            mesh.tetrahedronlist[:, current_tet_idx],  # current_tet_idx is already 1-indexed
            mesh.pointlist  # Use output pointlist
        )
        
        closest_face = nothing
        closest_distance = Inf
        
        for face in tet_faces
            intersection = intersect(ray, face)
            if intersection.is_intersection && 
               intersection.id > 1e-10 &&  # Avoid self-intersection
               intersection.id < closest_distance &&
               intersection.id < exit_distance
                closest_distance = intersection.id
                closest_face = face
            end
        end
        
        # Calculate segment length
        if closest_face !== nothing && closest_distance < exit_distance
            # Segment is inside current tetrahedron
            segment_length = closest_distance
            cell_lengths[current_tet_idx] += segment_length
            
            # Move to intersection point
            intersection = intersect(ray, closest_face)
            current_point = intersection.ip
            if collect_points
                push!(trajectory_points, current_point)
            end
            
            # Randomly change direction at cell boundary (high scattering)
            # Use more aggressive scattering - random direction on full sphere
            new_direction = randn(rng, 3)
            direction = new_direction / norm(new_direction)  # Normalize to unit vector
            
            # Additional scattering: add extra random direction changes for more scattering
            # This makes trajectories more chaotic and increases path lengths
            if rand(rng) < scatter_prob  # Probability controlled by scatter_prob parameter
                scatter_dir = randn(rng, 3)
                scatter_dir = scatter_dir / norm(scatter_dir)
                # Mix current direction with random scatter for high scattering intensity
                # Higher scatter_prob means more weight on the random scatter direction
                scatter_weight = 1.0 + scatter_prob  # Weight increases with scatter_prob
                direction = (direction + scatter_dir * scatter_weight) / (1.0 + scatter_weight)
                direction = direction / norm(direction)
                
                # Occasionally add a third scatter for maximum chaos (increases with scatter_prob)
                if rand(rng) < scatter_prob * 0.5  # Chance increases with scatter_prob
                    scatter_dir2 = randn(rng, 3)
                    scatter_dir2 = scatter_dir2 / norm(scatter_dir2)
                    direction = (direction + scatter_dir2) / 2.0
                    direction = direction / norm(direction)
                end
            end
        else
            # Segment to exit point
            segment_length = exit_distance
            cell_lengths[current_tet_idx] += segment_length
            if collect_points
                # Add exit point
                push!(trajectory_points, exit_point)
            end
            break  # Exited the cube
        end
    end
    
    if collect_points
        return cell_lengths, trajectory_points
    else
        return cell_lengths
    end
end

"""
    parse_commandline()

Parse command line arguments for the simulation.
"""
function parse_commandline()
    s = ArgParseSettings(description = "Monte Carlo Trajectory Simulation with Automatic Differentiation")
    
    @add_arg_table! s begin
        "--lambda", "-λ"
            help = "Mesh density parameter (smaller = finer mesh)"
            arg_type = Float64
            default = 0.1
        "--N", "-n"
            help = "Number of trajectories to simulate"
            arg_type = Int
            default = 100
        "--output", "-o"
            help = "Output file for the plot (HTML format)"
            arg_type = String
            default = "trajectories_plot.html"
        "--seed", "-s"
            help = "Random seed for reproducibility"
            arg_type = Int
            default = 42
        "--scatter-prob", "-p"
            help = "Probability of additional scattering at each boundary (0.0-1.0, higher = more scattering)"
            arg_type = Float64
            default = 0.7
        "--plot-trajectories", "-t"
            help = "Number of trajectories to plot in visualization (default: all, or 50 max)"
            arg_type = Int
            default = -1  # -1 means use all or max 50
    end
    
    return parse_args(s)
end

# Main example
function main(args = nothing)
    if args === nothing
        args = parse_commandline()
    end
    
    println("=== Monte Carlo Trajectory Simulation with AD ===\n")
    
    # Parameters from CLI or defaults
    λ = args["lambda"]
    N = args["N"]
    output_file = args["output"]
    seed = args["seed"]
    
    scatter_prob = haskey(args, "scatter-prob") ? args["scatter-prob"] : 0.7
    n_plot_trajs = haskey(args, "plot-trajectories") ? args["plot-trajectories"] : -1
    if n_plot_trajs == -1
        n_plot_trajs = min(N, 50)  # Default: all trajectories up to 50
    else
        n_plot_trajs = min(n_plot_trajs, N)  # Can't plot more than we have
    end
    
    println("Parameters:")
    println("  λ (mesh density): $λ")
    println("  N (trajectories): $N")
    println("  Output file: $output_file")
    println("  Seed: $seed")
    println("  Scatter probability: $scatter_prob (higher = more scattering)")
    println("  Trajectories to plot: $n_plot_trajs\n")
    
    println("Creating tetrahedral mesh with λ = $λ...")
    mesh = create_unit_cube_mesh(λ)
    n_tets = size(mesh.tetrahedronlist, 2)
    println("Generated mesh with $n_tets tetrahedra\n")
    
    # Initialize random number generator
    rng = MersenneTwister(seed)
    
    # Start point on one side of the cube (left face, center)
    start_point = Point(0.0, 0.5, 0.5)
    println("Starting point: ($(start_point.x), $(start_point.y), $(start_point.z))\n")
    
    # Create cell values based on distance from source
    # Cells closer to source (x=0) have smaller values, cells further (x=1) have larger values
    cell_values = compute_cell_values_from_source(mesh, start_point)
    println("Assigned distance-based values to $n_tets cells (range: [$(minimum(cell_values)), $(maximum(cell_values))])\n")
    println("  (Values increase from ~0 at x=0 to ~1 at x=1)\n")
    
    println("Simulating $N trajectories...")
    
    # Simulate trajectories (non-differentiable version for testing)
    sim_start_time = Base.time()
    mean_result = simulate_trajectories(N, start_point, mesh, cell_values, rng; scatter_prob=scatter_prob)
    sim_time = Base.time() - sim_start_time
    println("Mean weighted path length: $mean_result")
    println("Simulation time: $(round(sim_time, digits=3)) seconds\n")
    
    # Now compute with Zygote for automatic differentiation
    println("Computing gradients with Zygote.jl...")
    
    # Precompute trajectory segments (non-differentiable geometry)
    # Also collect trajectory points for visualization
    rng_fixed = MersenneTwister(seed)
    n_tets = size(mesh.tetrahedronlist, 2)
    total_cell_lengths = zeros(Float64, n_tets)
    all_trajectories = Vector{Point}[]
    
    for _ in 1:N
        direction = randn(rng_fixed, 3)
        direction = direction / norm(direction)
        cell_lengths, traj_points = trace_trajectory_segments(start_point, direction, mesh; 
            collect_points=true, rng=rng_fixed, scatter_prob=scatter_prob)
        total_cell_lengths .+= cell_lengths
        push!(all_trajectories, traj_points)
    end
    avg_cell_lengths = total_cell_lengths / N
    
    # Now only differentiate the weighted sum
    function objective(cell_vals)
        return compute_weighted_path_length(avg_cell_lengths, cell_vals)
    end
    
    # Compute gradient using Zygote
    grad_start_time = Base.time()
    grad_result = gradient(objective, cell_values)
    grad = grad_result[1]  # Extract the gradient array
    grad_time = Base.time() - grad_start_time
    
    println("Gradient computed!")
    println("Gradient computation time: $(round(grad_time, digits=3)) seconds")
    println("Gradient statistics:")
    println("  Min: $(minimum(grad))")
    println("  Max: $(maximum(grad))")
    println("  Mean: $(mean(grad))")
    println("  Std: $(std(grad))")
    
    # Sort gradients by absolute value (descending) with cell indices
    grad_with_indices = [(abs(grad[i]), grad[i], i) for i in 1:length(grad)]
    sort!(grad_with_indices, rev=true, by=x -> x[1])  # Sort by absolute value descending
    
    println("\nTop gradients (sorted by absolute value, descending):")
    println("  Rank | Cell Index | Gradient Value | Abs Value")
    println("  " * "-" ^ 60)
    n_top = min(20, length(grad_with_indices))  # Show top 20 or all if fewer
    for rank in 1:n_top
        abs_val, grad_val, cell_idx = grad_with_indices[rank]
        # Print with 1e-6 precision (scientific notation with 6 decimal places)
        grad_str = @sprintf("%.6e", grad_val)
        abs_str = @sprintf("%.6e", abs_val)
        println("  $(lpad(rank, 4)) | $(lpad(cell_idx, 10)) | $(lpad(grad_str, 13)) | $(lpad(abs_str, 9))")
    end
    if length(grad_with_indices) > n_top
        println("  ... (showing top $n_top of $(length(grad_with_indices)) gradients)")
    end
    println()
    
    # Create and save visualization (without gradients - they are printed above)
    println("\nCreating visualization...")
    plot_trajectories(mesh, cell_values, all_trajectories, output_file; 
        n_trajectories=n_plot_trajs, gradients=nothing)
    
    return mean_result, grad, all_trajectories
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

