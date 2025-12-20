"""
    TriangleIntersect

A minimal implementation for ray-triangle intersection.
"""
module TriangleIntersect

using LinearAlgebra

export Point, Ray, Triangle, Intersection, intersect

"""
    Point(x, y, z)

3D point type.
"""
struct Point
    x::Float64
    y::Float64
    z::Float64
end

"""
    Ray(origin, direction)
    Ray(origin, end_point)

Ray type with origin point and direction vector.
Can be constructed from origin and direction vector, or origin and end point.
"""
struct Ray
    origin::Point
    direction::Vector{Float64}
    
    # Constructor from two points
    Ray(origin::Point, end_point::Point) = new(
        origin,
        [end_point.x - origin.x, end_point.y - origin.y, end_point.z - origin.z]
    )
    
    # Constructor from origin and direction
    Ray(origin::Point, direction::Vector{Float64}) = new(origin, direction)
end

"""
    Triangle(p1, p2, p3)

Triangle type defined by three points.
"""
struct Triangle
    p1::Point
    p2::Point
    p3::Point
end

"""
    Intersection(is_intersection, id)
    Intersection(is_intersection, id, ip)

Intersection result containing whether intersection occurred, the distance, and intersection point.
"""
struct Intersection
    is_intersection::Bool
    id::Float64  # distance along ray
    ip::Point    # intersection point
    
    # Constructor without ip (for backward compatibility)
    Intersection(is_intersection::Bool, id::Float64) = new(is_intersection, id, Point(0.0, 0.0, 0.0))
    # Constructor with ip
    Intersection(is_intersection::Bool, id::Float64, ip::Point) = new(is_intersection, id, ip)
end

"""
    intersect(ray, triangle)

Compute ray-triangle intersection using Möller–Trumbore algorithm.
Returns Intersection(is_intersection, distance).
"""
function intersect(ray::Ray, triangle::Triangle)
    EPSILON = 1e-10
    
    v0 = [triangle.p1.x, triangle.p1.y, triangle.p1.z]
    v1 = [triangle.p2.x, triangle.p2.y, triangle.p2.z]
    v2 = [triangle.p3.x, triangle.p3.y, triangle.p3.z]
    
    orig = [ray.origin.x, ray.origin.y, ray.origin.z]
    dir = ray.direction
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = cross(dir, edge2)
    a = dot(edge1, h)
    
    if abs(a) < EPSILON
        return Intersection(false, Inf, Point(0.0, 0.0, 0.0))
    end
    
    f = 1.0 / a
    s = orig - v0
    u = f * dot(s, h)
    
    if u < 0.0 || u > 1.0
        return Intersection(false, Inf, Point(0.0, 0.0, 0.0))
    end
    
    q = cross(s, edge1)
    v = f * dot(dir, q)
    
    if v < 0.0 || u + v > 1.0
        return Intersection(false, Inf, Point(0.0, 0.0, 0.0))
    end
    
    t = f * dot(edge2, q)
    
    if t > EPSILON
        # Calculate intersection point
        intersection_point = Point(
            orig[1] + t * dir[1],
            orig[2] + t * dir[2],
            orig[3] + t * dir[3]
        )
        return Intersection(true, t, intersection_point)
    else
        return Intersection(false, Inf, Point(0.0, 0.0, 0.0))
    end
end

end # module

