"""
    Turtle

Julia port of the TURTLE library (Topographic Utilities for tRansporting
parTicules over Long rangEs) by Valentin Niess (CNRS/IN2P3, LPC).

Provides:
- ECEF ↔ geodetic ↔ horizontal coordinate transforms (WGS84)
- Geographic projections (Lambert conformal conic, UTM)
- Elevation maps with bilinear interpolation and gradient
- Tile stacks with LRU caching for global topography data
- Adaptive ECEF stepper through layered topography

Reference: https://github.com/niess/turtle
"""
module Turtle

using StaticArrays

export TurtleError, TurtleReturn
export ecef_from_geodetic, ecef_to_geodetic
export ecef_from_horizontal, ecef_to_horizontal
export Projection, LambertProjection, UTMProjection
export projection_project, projection_unproject
export ElevationMap, MapInfo
export map_create, map_load, map_dump, map_fill, map_node
export map_elevation, map_gradient, map_meta, map_projection
export ElevationStack, stack_create, stack_elevation, stack_gradient, stack_load
export Stepper, StepperSample
export stepper_create, stepper_step, stepper_position
export stepper_add_flat, stepper_add_map, stepper_add_stack, stepper_add_layer
export stepper_geoid_set, stepper_geoid_get
export stepper_range_get, stepper_range_set
export stepper_slope_get, stepper_slope_set
export stepper_resolution_get, stepper_resolution_set
export stepper_reset
export load_hgt, load_asc, load_grd

# ═══════════════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════════════

@enum TurtleReturn begin
    TURTLE_RETURN_SUCCESS = 0
    TURTLE_RETURN_BAD_ADDRESS
    TURTLE_RETURN_BAD_EXTENSION
    TURTLE_RETURN_BAD_FORMAT
    TURTLE_RETURN_BAD_PROJECTION
    TURTLE_RETURN_BAD_JSON
    TURTLE_RETURN_DOMAIN_ERROR
    TURTLE_RETURN_LIBRARY_ERROR
    TURTLE_RETURN_LOCK_ERROR
    TURTLE_RETURN_MEMORY_ERROR
    TURTLE_RETURN_PATH_ERROR
    TURTLE_RETURN_UNLOCK_ERROR
end

struct TurtleError <: Exception
    code::TurtleReturn
    message::String
end

Base.showerror(io::IO, e::TurtleError) = print(io, "TurtleError($(e.code)): $(e.message)")

# ═══════════════════════════════════════════════════════════════════════════════
# WGS84 ellipsoid parameters
# ═══════════════════════════════════════════════════════════════════════════════

const WGS84_A = 6378137.0
const WGS84_B = 6356752.3142
const WGS84_E = 0.081819190842622
const WGS84_F = 1.0 / 298.257223563

# ═══════════════════════════════════════════════════════════════════════════════
# ECEF coordinate transforms
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ecef_from_geodetic(latitude, longitude, elevation) -> SVector{3,Float64}

Convert geodetic coordinates to ECEF Cartesian coordinates.
Latitude and longitude in degrees, elevation in meters.
"""
function ecef_from_geodetic(latitude::Real, longitude::Real, elevation::Real)
    a = WGS84_A
    e = WGS84_E

    ϕ = deg2rad(latitude)
    λ = deg2rad(longitude)
    s = sin(ϕ)
    c = cos(ϕ)
    R = a / sqrt(1.0 - e * e * s * s)

    SVector(
        (R + elevation) * c * cos(λ),
        (R + elevation) * c * sin(λ),
        (R * (1.0 - e * e) + elevation) * s
    )
end

"""
    ecef_to_geodetic(ecef) -> (latitude, longitude, altitude)

Convert ECEF Cartesian coordinates to geodetic coordinates.
Returns latitude/longitude in degrees, altitude in meters.
Uses Olson (1996) algorithm.
"""
function ecef_to_geodetic(ecef::AbstractVector{<:Real})
    a = WGS84_A
    e2 = WGS84_E * WGS84_E
    a1 = a * e2
    a2 = a1 * a1
    a3 = 0.5 * a1 * e2
    a4 = 2.5 * a2
    a5 = a1 + a3
    a6 = 1.0 - e2

    x, y, z = Float64(ecef[1]), Float64(ecef[2]), Float64(ecef[3])

    if x == 0.0 && y == 0.0
        lat = z >= 0.0 ? 90.0 : -90.0
        lon = 0.0
        alt = abs(z) - WGS84_B
        return (lat, lon, alt)
    end

    lon = rad2deg(atan(y, x))

    zp = abs(z)
    w2 = x * x + y * y
    w = sqrt(w2)
    z2 = z * z
    r2 = w2 + z2
    r = sqrt(r2)
    s2 = z2 / r2
    c2 = w2 / r2

    if c2 > 0.3
        u = a2 / r
        v = a3 - a4 / r
        s = (zp / r) * (1.0 + c2 * (a1 + u + s2 * v) / r)
        la = asin(s)
        ss = s * s
        c_val = sqrt(1.0 - ss)
    else
        u = a2 / r
        v = a3 - a4 / r
        c_val = (w / r) * (1.0 - s2 * (a5 - u - c2 * v) / r)
        la = acos(c_val)
        ss = 1.0 - c_val * c_val
        s = sqrt(ss)
    end

    g = 1.0 - e2 * ss
    rg = a / sqrt(g)
    rf = a6 * rg
    u_val = w - rg * c_val
    v_val = zp - rf * s
    f = c_val * u_val + s * v_val
    m = c_val * v_val - s * u_val
    p = m / (rf / g + f)

    la += p
    if z < 0.0
        la = -la
    end
    lat = rad2deg(la)
    alt = f + 0.5 * m * p

    return (lat, lon, alt)
end

"""
    compute_enu(latitude, longitude) -> (east, north, up)

Compute the local East, North, Up basis vectors in ECEF coordinates.
"""
function compute_enu(latitude::Real, longitude::Real)
    λ = deg2rad(longitude)
    ϕ = deg2rad(latitude)
    sl = sin(λ); cl = cos(λ)
    sp = sin(ϕ); cp = cos(ϕ)

    east  = SVector(-sl, cl, 0.0)
    north = SVector(-cl * sp, -sl * sp, cp)
    up    = SVector(cl * cp, sl * cp, sp)

    return (east, north, up)
end

"""
    ecef_from_horizontal(latitude, longitude, azimuth, elevation) -> SVector{3,Float64}

Convert horizontal coordinates (azimuth, elevation) to an ECEF direction vector.
All angles in degrees.
"""
function ecef_from_horizontal(latitude::Real, longitude::Real,
                              azimuth::Real, elevation::Real)
    e, n, u = compute_enu(latitude, longitude)

    az = deg2rad(azimuth)
    el = deg2rad(elevation)
    ce = cos(el)
    r = SVector(ce * sin(az), ce * cos(az), sin(el))

    SVector(
        r[1] * e[1] + r[2] * n[1] + r[3] * u[1],
        r[1] * e[2] + r[2] * n[2] + r[3] * u[2],
        r[1] * e[3] + r[2] * n[3] + r[3] * u[3]
    )
end

"""
    ecef_to_horizontal(latitude, longitude, direction) -> (azimuth, elevation)

Convert an ECEF direction vector to horizontal coordinates.
Returns azimuth and elevation in degrees.
"""
function ecef_to_horizontal(latitude::Real, longitude::Real,
                            direction::AbstractVector{<:Real})
    e, n, u = compute_enu(latitude, longitude)

    x = e[1] * direction[1] + e[2] * direction[2] + e[3] * direction[3]
    y = n[1] * direction[1] + n[2] * direction[2] + n[3] * direction[3]
    z = u[1] * direction[1] + u[2] * direction[2] + u[3] * direction[3]

    r = sqrt(direction[1]^2 + direction[2]^2 + direction[3]^2)
    if r <= eps(Float32)
        return (0.0, 0.0)
    end

    az = rad2deg(atan(x, y))
    arg = clamp(z / r, -1.0, 1.0)
    el = rad2deg(asin(arg))

    return (az, el)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Geographic projections
# ═══════════════════════════════════════════════════════════════════════════════

abstract type Projection end

"""Lambert conformal conic projection parameters."""
struct LambertParameters
    e::Float64
    n::Float64
    c::Float64
    lambda_c::Float64
    xs::Float64
    ys::Float64
end

const LAMBERT_PARAMS = (
    LambertParameters(0.08248325676, 0.7604059656, 11603796.98, 0.04079234433, 600000.0, 5657616.674),
    LambertParameters(0.08248325676, 0.7289686274, 11745793.39, 0.04079234433, 600000.0, 6199695.768),
    LambertParameters(0.08248325676, 0.7289686274, 11745793.39, 0.04079234433, 600000.0, 8199695.768),
    LambertParameters(0.08248325676, 0.6959127966, 11947992.52, 0.04079234433, 600000.0, 6791905.085),
    LambertParameters(0.08248325676, 0.6712679322, 12136281.99, 0.04079234433, 234.358, 7239161.542),
    LambertParameters(0.08181919112, 0.7253743710, 11755528.70, 0.05235987756, 700000.0, 12657560.145),
)

"""
    LambertProjection

Lambert conformal conic projection. Tags: I, II, IIe, III, IV, 93.
"""
struct LambertProjection <: Projection
    tag::Int
    name::String

    function LambertProjection(tag_name::AbstractString)
        tags = ("I", "II", "IIe", "III", "IV", "93")
        idx = findfirst(==(tag_name), tags)
        idx === nothing && throw(TurtleError(TURTLE_RETURN_BAD_PROJECTION,
            "invalid Lambert tag: $tag_name"))
        new(idx, "Lambert $tag_name")
    end
end

function lambert_latitude_to_iso(latitude::Real, e::Real)
    ϕ = deg2rad(latitude)
    s = sin(ϕ)
    log(tan(π / 4 + ϕ / 2) * ((1.0 - e * s) / (1.0 + e * s))^(e / 2))
end

function lambert_iso_to_latitude(L::Real, e::Real)
    eL = exp(L)
    ϕ = 2.0 * atan(eL) - π / 2
    for _ in 1:100
        s = sin(ϕ)
        ϕ_new = 2.0 * atan(((1.0 + e * s) / (1.0 - e * s))^(e / 2) * eL) - π / 2
        abs(ϕ_new - ϕ) <= eps(Float32) && return rad2deg(ϕ_new)
        ϕ = ϕ_new
    end
    rad2deg(ϕ)
end

function projection_project(proj::LambertProjection, latitude::Real, longitude::Real)
    p = LAMBERT_PARAMS[proj.tag]
    L = lambert_latitude_to_iso(latitude, p.e)
    cenL = p.c * exp(-p.n * L)
    λ = deg2rad(longitude)
    θ = p.n * (λ - p.lambda_c)
    x = p.xs + cenL * sin(θ)
    y = p.ys - cenL * cos(θ)
    return (x, y)
end

function projection_unproject(proj::LambertProjection, x::Real, y::Real)
    p = LAMBERT_PARAMS[proj.tag]
    dx = x - p.xs
    dy = y - p.ys
    R = sqrt(dx * dx + dy * dy)
    γ = atan(dx, -dy)
    lon = rad2deg(p.lambda_c + γ / p.n)
    L = -log(R / p.c) / p.n
    lat = lambert_iso_to_latitude(L, p.e)
    return (lat, lon)
end

"""
    UTMProjection

Universal Transverse Mercator projection.
"""
struct UTMProjection <: Projection
    longitude_0::Float64
    hemisphere::Int
    name::String

    function UTMProjection(zone::Int, hemisphere::Char)
        h = hemisphere == 'N' ? 1 : hemisphere == 'S' ? -1 :
            throw(TurtleError(TURTLE_RETURN_BAD_PROJECTION,
                "invalid UTM hemisphere: $hemisphere"))
        lon0 = 6.0 * zone - 183.0
        new(lon0, h, "UTM $(zone)$(hemisphere)")
    end

    function UTMProjection(longitude_0::Real, hemisphere::Char)
        h = hemisphere == 'N' ? 1 : hemisphere == 'S' ? -1 :
            throw(TurtleError(TURTLE_RETURN_BAD_PROJECTION,
                "invalid UTM hemisphere: $hemisphere"))
        new(Float64(longitude_0), h, "UTM $(longitude_0)$(hemisphere)")
    end
end

function projection_project(proj::UTMProjection, latitude::Real, longitude::Real)
    a = 6378.137e3
    f = WGS84_F
    E0 = 5e5
    N0 = proj.hemisphere > 0 ? 0.0 : 1e7
    k0 = 0.9996

    n = f / (2.0 - f)
    A = a / (1.0 + n) * (1.0 + n * n * (0.25 + 0.0625 * n * n))
    α = (
        n * (0.5 + n * (-2.0 / 3.0 + 5.0 / 16.0 * n)),
        n * n * (13.0 / 48.0 - 3.0 / 5.0 * n),
        61.0 / 240.0 * n * n * n
    )

    c = 2.0 * sqrt(n) / (1.0 + n)
    s = sin(deg2rad(latitude))
    t = sinh(atanh(s) - c * atanh(c * s))
    dl = deg2rad(longitude - proj.longitude_0)
    ζ = atan(t, cos(dl))
    η = atanh(sin(dl) / sqrt(1.0 + t * t))

    xs = 0.0; ys = 0.0
    for i in 1:3
        xs += α[i] * cos(2.0 * i * ζ) * sinh(2.0 * i * η)
        ys += α[i] * sin(2.0 * i * ζ) * cosh(2.0 * i * η)
    end
    x = E0 + k0 * A * (η + xs)
    y = N0 + k0 * A * (ζ + ys)
    return (x, y)
end

function projection_unproject(proj::UTMProjection, x::Real, y::Real)
    a = 6378.137e3
    f = WGS84_F
    E0 = 5e5
    N0 = proj.hemisphere > 0 ? 0.0 : 1e7
    k0 = 0.9996

    n = f / (2.0 - f)
    A = a / (1.0 + n) * (1.0 + n * n * (0.25 + 0.0625 * n * n))
    β = (
        n * (0.5 + n * (-2.0 / 3.0 + 37.0 / 96.0 * n)),
        n * n * (1.0 / 48.0 + 1.0 / 15.0 * n),
        17.0 / 480.0 * n * n * n
    )
    δ = (
        n * (2.0 + n * (-2.0 / 3.0 - 2.0 * n)),
        n * n * (7.0 / 3.0 - 8.0 / 5.0 * n),
        56.0 / 15.0 * n * n * n
    )

    ζ0 = (y - N0) / (k0 * A)
    η0 = (x - E0) / (k0 * A)
    ζ = ζ0; η = η0
    for i in 1:3
        ζ -= β[i] * sin(2.0 * i * ζ0) * cosh(2.0 * i * η0)
        η -= β[i] * cos(2.0 * i * ζ0) * sinh(2.0 * i * η0)
    end
    χ = asin(sin(ζ) / cosh(η))
    s = 0.0
    for i in 1:3
        s += δ[i] * sin(2.0 * i * χ)
    end
    lat = rad2deg(χ + s)
    lon = proj.longitude_0 + rad2deg(atan(sinh(η), cos(ζ)))
    return (lat, lon)
end

"""
    parse_projection(name::AbstractString) -> Projection

Parse a projection from a name string (e.g. "Lambert 93", "UTM 31N").
"""
function parse_projection(name::AbstractString)
    parts = split(strip(name))
    isempty(parts) && throw(TurtleError(TURTLE_RETURN_BAD_PROJECTION,
        "empty projection name"))

    if parts[1] == "Lambert"
        length(parts) < 2 && throw(TurtleError(TURTLE_RETURN_BAD_PROJECTION,
            "missing Lambert tag"))
        return LambertProjection(String(parts[2]))
    elseif parts[1] == "UTM"
        length(parts) < 2 && throw(TurtleError(TURTLE_RETURN_BAD_PROJECTION,
            "missing UTM specifier"))
        spec = String(parts[2])
        hemisphere = spec[end]
        zone_str = spec[1:end-1]
        if '.' in zone_str
            lon0 = parse(Float64, zone_str)
            return UTMProjection(lon0, hemisphere)
        else
            zone = parse(Int, zone_str)
            return UTMProjection(zone, hemisphere)
        end
    else
        throw(TurtleError(TURTLE_RETURN_BAD_PROJECTION,
            "unknown projection: $(parts[1])"))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Elevation maps
# ═══════════════════════════════════════════════════════════════════════════════

struct MapInfo
    nx::Int
    ny::Int
    x::Tuple{Float64, Float64}
    y::Tuple{Float64, Float64}
    z::Tuple{Float64, Float64}
end

"""
    ElevationMap

A gridded elevation map with optional geographic projection.
Stores elevation as Float64 at regular grid nodes.
"""
mutable struct ElevationMap
    nx::Int
    ny::Int
    x0::Float64
    y0::Float64
    z0::Float64
    dx::Float64
    dy::Float64
    dz::Float64
    projection::Union{Projection, Nothing}
    data::Matrix{Float64}
end

function map_create(info::MapInfo; projection::Union{AbstractString, Projection, Nothing}=nothing)
    (info.nx <= 0 || info.ny <= 0) &&
        throw(TurtleError(TURTLE_RETURN_DOMAIN_ERROR, "invalid map dimensions"))
    info.z[1] == info.z[2] &&
        throw(TurtleError(TURTLE_RETURN_DOMAIN_ERROR, "zero z range"))

    proj = if projection isa AbstractString
        parse_projection(projection)
    else
        projection
    end

    nx = info.nx; ny = info.ny
    dx = nx > 1 ? (info.x[2] - info.x[1]) / (nx - 1) : 0.0
    dy = ny > 1 ? (info.y[2] - info.y[1]) / (ny - 1) : 0.0
    dz = (info.z[2] - info.z[1]) / 65535.0

    ElevationMap(nx, ny, info.x[1], info.y[1], info.z[1], dx, dy, dz, proj,
                 fill(info.z[1], ny, nx))
end

function map_fill(m::ElevationMap, ix::Int, iy::Int, elevation::Real)
    (ix < 0 || ix >= m.nx || iy < 0 || iy >= m.ny) &&
        throw(TurtleError(TURTLE_RETURN_DOMAIN_ERROR, "node indices out of range"))
    m.data[iy + 1, ix + 1] = Float64(elevation)
end

function map_node(m::ElevationMap, ix::Int, iy::Int)
    (ix < 0 || ix >= m.nx || iy < 0 || iy >= m.ny) &&
        throw(TurtleError(TURTLE_RETURN_DOMAIN_ERROR, "node indices out of range"))
    x = m.x0 + ix * m.dx
    y = m.y0 + iy * m.dy
    z = m.data[iy + 1, ix + 1]
    return (x, y, z)
end

"""
    map_elevation(m, x, y) -> (elevation, inside)

Bilinear interpolation of elevation at geographic coordinates (x, y).
"""
function map_elevation(m::ElevationMap, x::Real, y::Real)
    (isnan(x) || isnan(y)) && return (0.0, false)

    hx = (x - m.x0) / m.dx
    hy = (y - m.y0) / m.dy

    (hx > m.nx - 1 || hx < 0 || hy > m.ny - 1 || hy < 0) && return (0.0, false)

    ix = floor(Int, hx)
    iy = floor(Int, hy)
    if ix == m.nx - 1
        ix -= 1; hx = 1.0
    else
        hx -= ix
    end
    if iy == m.ny - 1
        iy -= 1; hy = 1.0
    else
        hy -= iy
    end

    z00 = m.data[iy + 1, ix + 1]
    z10 = m.data[iy + 1, ix + 2]
    z01 = m.data[iy + 2, ix + 1]
    z11 = m.data[iy + 2, ix + 2]
    z = z00 * (1.0 - hx) * (1.0 - hy) + z01 * (1.0 - hx) * hy +
        z10 * hx * (1.0 - hy) + z11 * hx * hy

    return (z, true)
end

"""
    map_gradient(m, x, y) -> (gx, gy, inside)

Compute the terrain gradient at geographic coordinates with adaptive stencil.
"""
function map_gradient(m::ElevationMap, x::Real, y::Real)
    (isnan(x) || isnan(y)) && return (0.0, 0.0, false)

    hx = (x - m.x0) / m.dx
    hy = (y - m.y0) / m.dy

    (hx > m.nx - 1 || hx < 0 || hy > m.ny - 1 || hy < 0) && return (0.0, 0.0, false)

    ix = floor(Int, hx)
    iy = floor(Int, hy)
    if ix == m.nx - 1
        ix -= 1; hx = 1.0
    else
        hx -= ix
    end
    if iy == m.ny - 1
        iy -= 1; hy = 1.0
    else
        hy -= iy
    end

    z00 = m.data[iy + 1, ix + 1]
    z10 = m.data[iy + 1, ix + 2]
    z01 = m.data[iy + 2, ix + 1]
    z11 = m.data[iy + 2, ix + 2]

    gx = if hx <= 0.5
        gx1 = (z10 - z00) * (1.0 - hy) + (z11 - z01) * hy
        if ix == 0
            gx1 / m.dx
        else
            z_10 = m.data[iy + 1, ix]
            z_11 = m.data[iy + 2, ix]
            gx0 = (z00 - z_10) * (1.0 - hy) + (z01 - z_11) * hy
            ax = hx + 0.5
            (gx0 * (1.0 - ax) + gx1 * ax) / m.dx
        end
    else
        gx0 = (z10 - z00) * (1.0 - hy) + (z11 - z01) * hy
        if ix == m.nx - 2
            gx0 / m.dx
        else
            z20 = m.data[iy + 1, ix + 3]
            z21 = m.data[iy + 2, ix + 3]
            gx1 = (z20 - z10) * (1.0 - hy) + (z21 - z11) * hy
            ax = hx - 0.5
            (gx0 * (1.0 - ax) + gx1 * ax) / m.dx
        end
    end

    gy = if hy <= 0.5
        gy1 = (z01 - z00) * (1.0 - hx) + (z11 - z10) * hx
        if iy == 0
            gy1 / m.dy
        else
            z0_1 = m.data[iy, ix + 1]
            z1_1 = m.data[iy, ix + 2]
            gy0 = (z00 - z0_1) * (1.0 - hx) + (z10 - z1_1) * hx
            ay = hy + 0.5
            (gy0 * (1.0 - ay) + gy1 * ay) / m.dy
        end
    else
        gy0 = (z01 - z00) * (1.0 - hx) + (z11 - z10) * hx
        if iy == m.ny - 2
            gy0 / m.dy
        else
            z02 = m.data[iy + 3, ix + 1]
            z12 = m.data[iy + 3, ix + 2]
            gy1 = (z02 - z01) * (1.0 - hx) + (z12 - z11) * hx
            ay = hy - 0.5
            (gy0 * (1.0 - ay) + gy1 * ay) / m.dy
        end
    end

    return (gx, gy, true)
end

function map_meta(m::ElevationMap)
    info = MapInfo(m.nx, m.ny,
        (m.x0, m.x0 + (m.nx - 1) * m.dx),
        (m.y0, m.y0 + (m.ny - 1) * m.dy),
        (m.z0, m.z0 + 65535 * m.dz))
    return (info, m.projection)
end

map_projection(m::ElevationMap) = m.projection

# ═══════════════════════════════════════════════════════════════════════════════
# I/O: file format readers
# ═══════════════════════════════════════════════════════════════════════════════

"""
    load_hgt(path) -> ElevationMap

Load an SRTM HGT file. Grid size inferred from file size.
Filename convention: `N45E002.hgt` encodes the SW corner.
"""
function load_hgt(path::AbstractString)
    fname = basename(path)
    m = match(r"([NS])(\d+)([EW])(\d+)\.hgt"i, fname)
    m === nothing && throw(TurtleError(TURTLE_RETURN_BAD_FORMAT,
        "cannot parse HGT filename: $fname"))

    lat = parse(Int, m[2])
    if uppercase(m[1]) == "S"; lat = -lat; end
    lon = parse(Int, m[4])
    if uppercase(m[3]) == "W"; lon = -lon; end

    fsize = filesize(path)
    n = round(Int, sqrt(fsize / 2))
    n * n * 2 != fsize && throw(TurtleError(TURTLE_RETURN_BAD_FORMAT,
        "unexpected HGT file size: $fsize"))

    raw = read(path)
    data = Matrix{Float64}(undef, n, n)
    for iy in 1:n
        for ix in 1:n
            offset = ((n - iy) * n + (ix - 1)) * 2
            hi = raw[offset + 1]
            lo = raw[offset + 2]
            val = Int16(hi) << 8 | Int16(lo)
            data[iy, ix] = val == -32768 ? 0.0 : Float64(val)
        end
    end

    dx = 1.0 / (n - 1)
    dy = 1.0 / (n - 1)

    ElevationMap(n, n,
        Float64(lon), Float64(lat), 0.0,
        dx, dy, 1.0,
        nothing, data)
end

"""
    load_asc(path) -> ElevationMap

Load an ASCII grid (.asc) elevation file.
"""
function load_asc(path::AbstractString)
    lines = readlines(path)
    ncols = 0; nrows = 0
    xll = 0.0; yll = 0.0; cellsize = 0.0
    nodata = -9999.0
    header_lines = 0

    for line in lines
        parts = split(strip(line))
        isempty(parts) && continue
        key = lowercase(parts[1])
        if key == "ncols"
            ncols = parse(Int, parts[2]); header_lines += 1
        elseif key == "nrows"
            nrows = parse(Int, parts[2]); header_lines += 1
        elseif key == "xllcorner" || key == "xllcenter"
            xll = parse(Float64, parts[2]); header_lines += 1
        elseif key == "yllcorner" || key == "yllcenter"
            yll = parse(Float64, parts[2]); header_lines += 1
        elseif key == "cellsize"
            cellsize = parse(Float64, parts[2]); header_lines += 1
        elseif key == "nodata_value"
            nodata = parse(Float64, parts[2]); header_lines += 1
        else
            break
        end
    end

    data = Matrix{Float64}(undef, nrows, ncols)
    row = 0
    for i in (header_lines + 1):length(lines)
        vals = split(strip(lines[i]))
        isempty(vals) && continue
        row += 1
        row > nrows && break
        for (col, v) in enumerate(vals)
            col > ncols && break
            val = parse(Float64, v)
            data[nrows - row + 1, col] = val == nodata ? 0.0 : val
        end
    end

    ElevationMap(ncols, nrows,
        xll, yll, 0.0,
        cellsize, cellsize, 1.0,
        nothing, data)
end

"""
    load_grd(path) -> ElevationMap

Load a GRD (EGM96 geoid) file.
Header: lat_min lat_max lon_min lon_max dlat dlon
"""
function load_grd(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && throw(TurtleError(TURTLE_RETURN_BAD_FORMAT, "empty GRD file"))

    hdr = split(strip(lines[1]))
    length(hdr) < 6 && throw(TurtleError(TURTLE_RETURN_BAD_FORMAT,
        "invalid GRD header"))

    lat_min = parse(Float64, hdr[1])
    lat_max = parse(Float64, hdr[2])
    lon_min = parse(Float64, hdr[3])
    lon_max = parse(Float64, hdr[4])
    dlat = parse(Float64, hdr[5])
    dlon = parse(Float64, hdr[6])

    nlat = round(Int, (lat_max - lat_min) / dlat) + 1
    nlon = round(Int, (lon_max - lon_min) / dlon) + 1

    data = Matrix{Float64}(undef, nlat, nlon)
    vals = Float64[]
    for i in 2:length(lines)
        for v in split(strip(lines[i]))
            push!(vals, parse(Float64, v))
        end
    end

    idx = 0
    for iy in 1:nlat
        for ix in 1:nlon
            idx += 1
            idx <= length(vals) || break
            data[iy, ix] = vals[idx]
        end
    end

    ElevationMap(nlon, nlat,
        lon_min, lat_min, 0.0,
        dlon, dlat, 1.0,
        nothing, data)
end

"""
    map_load(path) -> ElevationMap

Load an elevation map from file. Format guessed from extension.
Supports: .hgt, .asc, .grd
"""
function map_load(path::AbstractString)
    isfile(path) || throw(TurtleError(TURTLE_RETURN_PATH_ERROR,
        "file not found: $path"))
    ext = lowercase(splitext(path)[2])
    if ext == ".hgt"
        load_hgt(path)
    elseif ext == ".asc"
        load_asc(path)
    elseif ext == ".grd"
        load_grd(path)
    else
        throw(TurtleError(TURTLE_RETURN_BAD_EXTENSION,
            "unsupported file format: $ext"))
    end
end

function map_dump(m::ElevationMap, path::AbstractString)
    throw(TurtleError(TURTLE_RETURN_BAD_EXTENSION,
        "map_dump not yet implemented for Julia port"))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Elevation stack (tile management with LRU cache)
# ═══════════════════════════════════════════════════════════════════════════════

struct TileInfo
    path::String
    lat::Int
    lon::Int
end

"""
    ElevationStack

A managed collection of tiled elevation data with LRU caching.
Scans a directory of HGT/ASC/GRD tiles and loads them on demand.
"""
mutable struct ElevationStack
    root::String
    tiles::Dict{Tuple{Int,Int}, String}
    cache::Vector{Tuple{Tuple{Int,Int}, ElevationMap}}
    max_size::Int
    lat_delta::Float64
    lon_delta::Float64
end

"""
    stack_create(path; max_size=0) -> ElevationStack

Create a stack by scanning a directory for elevation tiles.
`max_size <= 0` means unlimited cache.
"""
function stack_create(path::AbstractString; max_size::Int=0)
    isdir(path) || throw(TurtleError(TURTLE_RETURN_PATH_ERROR,
        "directory not found: $path"))

    tiles = Dict{Tuple{Int,Int}, String}()
    for f in readdir(path)
        fpath = joinpath(path, f)
        isfile(fpath) || continue
        ext = lowercase(splitext(f)[2])
        ext in (".hgt", ".asc", ".grd") || continue

        if ext == ".hgt"
            m = match(r"([NS])(\d+)([EW])(\d+)\.hgt"i, f)
            m === nothing && continue
            lat = parse(Int, m[2])
            if uppercase(m[1]) == "S"; lat = -lat; end
            lon = parse(Int, m[4])
            if uppercase(m[3]) == "W"; lon = -lon; end
            tiles[(lat, lon)] = fpath
        end
    end

    ElevationStack(path, tiles,
        Vector{Tuple{Tuple{Int,Int}, ElevationMap}}(),
        max_size, 1.0, 1.0)
end

function stack_get_tile!(stack::ElevationStack, lat::Int, lon::Int)
    key = (lat, lon)
    for (i, (k, m)) in enumerate(stack.cache)
        if k == key
            if i > 1
                deleteat!(stack.cache, i)
                pushfirst!(stack.cache, (k, m))
            end
            return m
        end
    end

    haskey(stack.tiles, key) || return nothing

    m = map_load(stack.tiles[key])

    if stack.max_size > 0 && length(stack.cache) >= stack.max_size
        pop!(stack.cache)
    end
    pushfirst!(stack.cache, (key, m))
    return m
end

"""
    stack_elevation(stack, latitude, longitude) -> (elevation, inside)

Get elevation at geodetic coordinates with automatic tile loading.
"""
function stack_elevation(stack::ElevationStack, latitude::Real, longitude::Real)
    lat_tile = floor(Int, latitude)
    lon_tile = floor(Int, longitude)

    m = stack_get_tile!(stack, lat_tile, lon_tile)
    m === nothing && return (0.0, false)

    return map_elevation(m, longitude, latitude)
end

"""
    stack_gradient(stack, latitude, longitude) -> (glat, glon, inside)
"""
function stack_gradient(stack::ElevationStack, latitude::Real, longitude::Real)
    lat_tile = floor(Int, latitude)
    lon_tile = floor(Int, longitude)

    m = stack_get_tile!(stack, lat_tile, lon_tile)
    m === nothing && return (0.0, 0.0, false)

    return map_gradient(m, longitude, latitude)
end

"""
    stack_load(stack) -> nothing

Preload all tiles into memory (up to max_size).
"""
function stack_load(stack::ElevationStack)
    for (key, path) in stack.tiles
        (stack.max_size > 0 && length(stack.cache) >= stack.max_size) && break
        found = false
        for (k, _) in stack.cache
            if k == key; found = true; break; end
        end
        found && continue
        m = map_load(path)
        pushfirst!(stack.cache, (key, m))
    end
    nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# ECEF Stepper
# ═══════════════════════════════════════════════════════════════════════════════

abstract type StepperDataSource end

struct FlatSource <: StepperDataSource end

struct MapSource <: StepperDataSource
    map::ElevationMap
end

struct StackSource <: StepperDataSource
    stack::ElevationStack
end

struct StepperLayer
    sources::Vector{Tuple{StepperDataSource, Float64}}
end

mutable struct StepperSample
    position::MVector{3, Float64}
    latitude::Float64
    longitude::Float64
    altitude::Float64
    elevation::NTuple{2, Float64}
    index::NTuple{2, Int}
end

mutable struct LocalTransform
    reference_ecef::MVector{3, Float64}
    reference_geo::MVector{5, Float64}
    jacobian::MMatrix{5, 3, Float64, 15}
    valid::Bool
end

"""
    Stepper

Adaptive stepper through layered topography using ECEF coordinates.
Supports flat surfaces, elevation maps, and tiled stacks as data sources.
"""
mutable struct Stepper
    layers::Vector{StepperLayer}
    geoid::Union{ElevationMap, Nothing}
    local_range::Float64
    slope_factor::Float64
    resolution_factor::Float64
    last::StepperSample
    transform::LocalTransform
end

function stepper_create()
    Stepper(
        StepperLayer[],
        nothing,
        1.0,
        0.4,
        1e-2,
        StepperSample(
            MVector(typemax(Float64), typemax(Float64), typemax(Float64)),
            0.0, 0.0, 0.0, (0.0, 0.0), (-1, -1)),
        LocalTransform(
            MVector(typemax(Float64), typemax(Float64), typemax(Float64)),
            MVector(0.0, 0.0, 0.0, 0.0, 0.0),
            MMatrix{5, 3}(zeros(5, 3)),
            false)
    )
end

function stepper_add_layer(s::Stepper)
    if isempty(s.layers) || !isempty(s.layers[end].sources)
        push!(s.layers, StepperLayer(Tuple{StepperDataSource, Float64}[]))
    end
    nothing
end

function stepper_add_flat(s::Stepper, ground_level::Real=0.0)
    if isempty(s.layers)
        push!(s.layers, StepperLayer(Tuple{StepperDataSource, Float64}[]))
    end
    push!(s.layers[end].sources, (FlatSource(), Float64(ground_level)))
    nothing
end

function stepper_add_map(s::Stepper, m::ElevationMap, offset::Real=0.0)
    if isempty(s.layers)
        push!(s.layers, StepperLayer(Tuple{StepperDataSource, Float64}[]))
    end
    push!(s.layers[end].sources, (MapSource(m), Float64(offset)))
    nothing
end

function stepper_add_stack(s::Stepper, stack::ElevationStack, offset::Real=0.0)
    if isempty(s.layers)
        push!(s.layers, StepperLayer(Tuple{StepperDataSource, Float64}[]))
    end
    push!(s.layers[end].sources, (StackSource(stack), Float64(offset)))
    nothing
end

stepper_geoid_set(s::Stepper, geoid::Union{ElevationMap, Nothing}) = (s.geoid = geoid; stepper_reset(s))
stepper_geoid_get(s::Stepper) = s.geoid

stepper_range_get(s::Stepper) = s.local_range
function stepper_range_set(s::Stepper, range::Real)
    s.local_range = Float64(range)
    stepper_reset(s)
end

stepper_slope_get(s::Stepper) = s.slope_factor
stepper_slope_set(s::Stepper, slope::Real) = (s.slope_factor = Float64(slope))

stepper_resolution_get(s::Stepper) = s.resolution_factor
stepper_resolution_set(s::Stepper, resolution::Real) = (s.resolution_factor = Float64(resolution))

function stepper_reset(s::Stepper)
    s.last.position .= typemax(Float64)
    s.last.index = (-1, -1)
    s.transform.valid = false
    s.transform.reference_ecef .= typemax(Float64)
    nothing
end

function ecef_to_geodetic_corrected(s::Stepper, position)
    lat, lon, alt = ecef_to_geodetic(position)
    if s.geoid !== nothing
        lo = lon >= 0.0 ? lon : lon + 360.0
        undulation, inside = map_elevation(s.geoid, lo, lat)
        if inside
            alt -= undulation
        end
    end
    return (lat, lon, alt)
end

function source_elevation(src::FlatSource, ::Stepper, lat::Float64, lon::Float64)
    return (0.0, true)
end

function source_elevation(src::MapSource, s::Stepper, lat::Float64, lon::Float64)
    m = src.map
    proj = m.projection
    if proj !== nothing
        x, y = projection_project(proj, lat, lon)
        return map_elevation(m, x, y)
    else
        return map_elevation(m, lon, lat)
    end
end

function source_elevation(src::StackSource, ::Stepper, lat::Float64, lon::Float64)
    return stack_elevation(src.stack, lat, lon)
end

function get_geographic_cached(s::Stepper, position)
    tr = s.transform
    if tr.valid && s.local_range > 0.0
        range = 0.0
        for i in 1:3
            r = abs(position[i] - tr.reference_ecef[i])
            r > range && (range = r)
        end
        if range < s.local_range
            geo = MVector(0.0, 0.0, 0.0, 0.0, 0.0)
            local_ = SVector(
                position[1] - tr.reference_ecef[1],
                position[2] - tr.reference_ecef[2],
                position[3] - tr.reference_ecef[3])
            for i in 1:3
                geo[i] = tr.reference_geo[i]
                for j in 1:3
                    geo[i] += tr.jacobian[i, j] * local_[j]
                end
            end
            return (geo[1], geo[2], geo[3])
        end
    end

    lat, lon, alt = ecef_to_geodetic_corrected(s, position)

    step = 0.0
    for i in 1:3
        d = abs(position[i] - s.last.position[i])
        d > step && (step = d)
    end

    if s.local_range > 0.0 && step < 0.33 * s.local_range
        tr.reference_ecef .= position
        tr.reference_geo[1] = lat
        tr.reference_geo[2] = lon
        tr.reference_geo[3] = alt

        for j in 1:3
            pos2 = MVector(position[1], position[2], position[3])
            pos2[j] += 10.0
            lat2, lon2, alt2 = ecef_to_geodetic_corrected(s, pos2)
            tr.jacobian[1, j] = 0.1 * (lat2 - lat)
            tr.jacobian[2, j] = 0.1 * (lon2 - lon)
            tr.jacobian[3, j] = 0.1 * (alt2 - alt)
        end
        tr.valid = true
    end

    return (lat, lon, alt)
end

function stepper_sample!(s::Stepper, position)
    pos_changed = (position[1] != s.last.position[1] ||
                   position[2] != s.last.position[2] ||
                   position[3] != s.last.position[3])

    if !pos_changed
        return s.last
    end

    s.transform.valid && (s.transform.valid = false)

    lat, lon, alt = get_geographic_cached(s, position)

    best_layer = -1
    best_data = -1
    elev_below = -Inf
    elev_above = Inf

    for (li, layer) in enumerate(s.layers)
        for (di, (src, offset)) in Iterators.reverse(enumerate(layer.sources))
            elev, inside = source_elevation(src, s, lat, lon)
            if inside
                elev += offset
                if elev >= alt
                    best_layer = li - 1
                    best_data = di - 1
                    elev_above = elev
                    @goto found
                else
                    if li > 1 || di > 1
                        elev_below = max(elev_below, elev)
                    end
                    best_layer = li
                    best_data = di - 1
                end
                break
            end
        end
    end
    @label found

    s.last.position .= position
    s.last.latitude = lat
    s.last.longitude = lon
    s.last.altitude = alt
    s.last.index = (best_layer, best_data)
    s.last.elevation = (elev_below == -Inf ? 0.0 : elev_below, elev_above == Inf ? 0.0 : elev_above)

    return s.last
end

"""
    stepper_step(s, position, direction=nothing) -> (sample, step_length)

Compute or perform a step through the topography.

If `direction` is `nothing`, samples the geography at `position` and returns a
tentative step length. If `direction` is provided, performs the step (modifying
`position` in-place) and uses binary search to locate medium boundaries.

Returns `(sample::StepperSample, step_length::Float64)`.
"""
function stepper_step(s::Stepper, position::AbstractVector{<:Real};
                      direction::Union{AbstractVector{<:Real}, Nothing}=nothing)
    pos = MVector{3, Float64}(position[1], position[2], position[3])

    sample = stepper_sample!(s, pos)
    if sample.index[1] < 0
        if direction !== nothing
            position .= pos
        end
        return (sample, 0.0)
    end

    ds = 0.0
    for i in 0:1
        if sample.index[1] == 0 && i == 0
            continue
        elseif sample.index[1] == length(s.layers) && i == 1
            break
        end
        dsi = abs(sample.altitude - sample.elevation[i + 1])
        if dsi < ds || ds <= 0.0
            ds = dsi
        end
    end
    ds *= s.slope_factor
    if ds < s.resolution_factor
        ds = s.resolution_factor
    end

    if direction === nothing
        return (sample, ds)
    end

    dir = SVector{3, Float64}(direction[1], direction[2], direction[3])
    pos .+= dir .* ds

    medium0 = sample.index[1]
    sample = stepper_sample!(s, pos)
    medium1 = sample.index[1]

    if medium0 != medium1
        ds0 = -ds; ds1 = 0.0
        while ds1 - ds0 > 1e-8
            ds2 = 0.5 * (ds0 + ds1)
            pos2 = MVector(
                pos[1] + dir[1] * ds2,
                pos[2] + dir[2] * ds2,
                pos[3] + dir[3] * ds2)

            old_pos = MVector(s.last.position...)
            old_sample_index = s.last.index
            old_sample_elev = s.last.elevation
            old_sample_lat = s.last.latitude
            old_sample_lon = s.last.longitude
            old_sample_alt = s.last.altitude

            s.last.position .= typemax(Float64)
            sample2 = stepper_sample!(s, pos2)
            medium2 = sample2.index[1]

            if medium2 == medium0
                ds0 = ds2
                s.last.position .= old_pos
                s.last.index = old_sample_index
                s.last.elevation = old_sample_elev
                s.last.latitude = old_sample_lat
                s.last.longitude = old_sample_lon
                s.last.altitude = old_sample_alt
            else
                medium1 = medium2
                ds1 = ds2
                pos .= pos2
            end
        end
        ds += ds1
        pos .+= dir .* ds1
        stepper_sample!(s, pos)
    end

    position[1] = pos[1]
    position[2] = pos[2]
    position[3] = pos[3]

    return (s.last, ds)
end

"""
    stepper_position(s, latitude, longitude, height, layer_index) -> (position, data_index)

Convert geographic coordinates to an ECEF position on a given layer.
"""
function stepper_position(s::Stepper, latitude::Real, longitude::Real,
                          height::Real, layer_index::Int)
    (layer_index < 0 || layer_index >= length(s.layers)) &&
        throw(TurtleError(TURTLE_RETURN_DOMAIN_ERROR,
            "layer index out of range: $layer_index"))

    layer = s.layers[layer_index + 1]
    for (di, (src, offset)) in Iterators.reverse(enumerate(layer.sources))
        elev, inside = source_elevation(src, s, Float64(latitude), Float64(longitude))
        if inside
            elev += offset
            if s.geoid !== nothing
                lo = longitude >= 0.0 ? Float64(longitude) : Float64(longitude) + 360.0
                undulation, ginside = map_elevation(s.geoid, lo, Float64(latitude))
                if ginside
                    elev += undulation
                end
            end

            pos = ecef_from_geodetic(latitude, longitude, elev + height)
            return (pos, di - 1)
        end
    end

    return (SVector(0.0, 0.0, 0.0), -1)
end

end # module Turtle
