"""
    Types

Core data structures for PUMAS Monte Carlo simulation.
Designed to be compatible with Zygote.jl automatic differentiation.
"""
module Types

using StaticArrays
using ChainRulesCore

export Particle, MUON, TAU
export EnergyLossMode, ENERGY_LOSS_DISABLED, ENERGY_LOSS_CSDA, ENERGY_LOSS_MIXED, ENERGY_LOSS_STRAGGLED
export DecayMode, DECAY_DISABLED, DECAY_WEIGHTED, DECAY_RANDOMISED
export DirectionMode, DIRECTION_FORWARD, DIRECTION_BACKWARD
export ScatteringMode, SCATTERING_DISABLED, SCATTERING_MIXED
export Event, EVENT_NONE, EVENT_LIMIT_ENERGY, EVENT_LIMIT_DISTANCE
export EVENT_LIMIT_GRAMMAGE, EVENT_LIMIT_TIME, EVENT_MEDIUM
export EVENT_VERTEX_BREMSSTRAHLUNG, EVENT_VERTEX_PAIR_CREATION
export EVENT_VERTEX_PHOTONUCLEAR, EVENT_VERTEX_DELTA_RAY
export EVENT_VERTEX_COULOMB, EVENT_VERTEX_DECAY, EVENT_WEIGHT
export EVENT_START, EVENT_STOP
export Process, PROCESS_BREMSSTRAHLUNG, PROCESS_PAIR_PRODUCTION, PROCESS_PHOTONUCLEAR
export Step, STEP_CHECK, STEP_RAW
export Property, PROPERTY_CROSS_SECTION, PROPERTY_ELASTIC_CUTOFF_ANGLE
export PROPERTY_ELASTIC_PATH, PROPERTY_STOPPING_POWER, PROPERTY_RANGE
export PROPERTY_KINETIC_ENERGY, PROPERTY_MAGNETIC_ROTATION
export PROPERTY_TRANSPORT_PATH, PROPERTY_PROPER_TIME
export State, Locals, Medium, ContextMode, ContextLimit
export Vec3, LocalsCallback, update_state, has_event

# Type alias for 3D vectors (static for performance, Zygote-friendly)
const Vec3{T} = SVector{3,T}

"""
    Particle

Supported particle types for transport.
"""
@enum Particle begin
    MUON = 0
    TAU = 1
end

"""
    EnergyLossMode

Modes for energy loss computation.
"""
@enum EnergyLossMode begin
    ENERGY_LOSS_DISABLED = -1
    ENERGY_LOSS_CSDA = 0      # Continuously Slowing Down Approximation
    ENERGY_LOSS_MIXED = 1     # Mixed Monte Carlo (soft + hard)
    ENERGY_LOSS_STRAGGLED = 2 # Mixed + electronic straggling
end

"""
    DecayMode

Modes for handling particle decay.
"""
@enum DecayMode begin
    DECAY_DISABLED = -1
    DECAY_WEIGHTED = 0    # Weight factor for decay
    DECAY_RANDOMISED = 1  # Random decay vertex
end

"""
    DirectionMode

Direction of Monte Carlo flow.
"""
@enum DirectionMode begin
    DIRECTION_FORWARD = 0
    DIRECTION_BACKWARD = 1
end

"""
    ScatteringMode

Algorithm for scattering simulation.
"""
@enum ScatteringMode begin
    SCATTERING_DISABLED = -1
    SCATTERING_MIXED = 1  # Mixed soft/hard
end

"""
    Event

Flags for transport events.
"""
@enum Event begin
    EVENT_NONE = 0
    EVENT_LIMIT_ENERGY = 1
    EVENT_LIMIT_DISTANCE = 2
    EVENT_LIMIT_GRAMMAGE = 4
    EVENT_LIMIT_TIME = 8
    EVENT_MEDIUM = 16
    EVENT_VERTEX_BREMSSTRAHLUNG = 32
    EVENT_VERTEX_PAIR_CREATION = 64
    EVENT_VERTEX_PHOTONUCLEAR = 128
    EVENT_VERTEX_DELTA_RAY = 256
    EVENT_VERTEX_COULOMB = 512
    EVENT_VERTEX_DECAY = 1024
    EVENT_WEIGHT = 2048
    EVENT_START = 4096
    EVENT_STOP = 8192
end

# Allow combining events with bitwise OR (returns Int, not Event)
# Event flags are designed to be combined as integers
combine_events(a::Event, b::Event) = Int(a) | Int(b)
combine_events(a::Int, b::Event) = a | Int(b)
combine_events(a::Event, b::Int) = Int(a) | b

"""Check if event flags contain a specific flag"""
has_event(events::Int, flag::Event) = events & Int(flag) != 0
has_event(events::Event, flag::Event) = Int(events) & Int(flag) != 0

export combine_events

"""
    Process

Radiative processes in PUMAS.
"""
@enum Process begin
    PROCESS_BREMSSTRAHLUNG = 0
    PROCESS_PAIR_PRODUCTION = 1
    PROCESS_PHOTONUCLEAR = 2
end

"""
    Step

Return codes for medium callback.
"""
@enum Step begin
    STEP_CHECK = 0  # Cross-check proposed step
    STEP_RAW = 1    # Use proposed step as-is
end

"""
    Property

Physics properties that can be tabulated.
"""
@enum Property begin
    PROPERTY_CROSS_SECTION = 0
    PROPERTY_ELASTIC_CUTOFF_ANGLE = 1
    PROPERTY_ELASTIC_PATH = 2
    PROPERTY_STOPPING_POWER = 3
    PROPERTY_RANGE = 4
    PROPERTY_KINETIC_ENERGY = 5
    PROPERTY_MAGNETIC_ROTATION = 6
    PROPERTY_TRANSPORT_PATH = 7
    PROPERTY_PROPER_TIME = 8
end

"""
    State{T<:Real}

Monte Carlo state for a particle.
All fields are differentiable with Zygote.

# Fields
- `charge::T`: Electric charge (±1 for physical particles)
- `energy::T`: Kinetic energy in GeV
- `distance::T`: Total travelled distance in m
- `grammage::T`: Total travelled grammage in kg/m²
- `time::T`: Proper time in m/c
- `weight::T`: Monte Carlo weight
- `position::Vec3{T}`: Absolute location in m
- `direction::Vec3{T}`: Unit direction vector
- `decayed::Bool`: Whether particle has decayed
"""
struct State{T<:Real}
    charge::T
    energy::T
    distance::T
    grammage::T
    time::T
    weight::T
    position::Vec3{T}
    direction::Vec3{T}
    decayed::Bool
end

# Constructor with default values
function State{T}(;
    charge::T = one(T),
    energy::T = one(T),
    distance::T = zero(T),
    grammage::T = zero(T),
    time::T = zero(T),
    weight::T = one(T),
    position::Vec3{T} = Vec3{T}(0, 0, 0),
    direction::Vec3{T} = Vec3{T}(0, 0, -1),
    decayed::Bool = false
) where T<:Real
    State{T}(charge, energy, distance, grammage, time, weight, position, direction, decayed)
end

State(; kwargs...) = State{Float64}(; kwargs...)

# Make State Zygote-friendly with ChainRulesCore
function ChainRulesCore.rrule(::Type{State{T}}, charge, energy, distance, grammage, 
                              time, weight, position, direction, decayed) where T
    state = State{T}(charge, energy, distance, grammage, time, weight, position, direction, decayed)
    function State_pullback(Δ)
        return (NoTangent(), Δ.charge, Δ.energy, Δ.distance, Δ.grammage, 
                Δ.time, Δ.weight, Δ.position, Δ.direction, NoTangent())
    end
    return state, State_pullback
end

# Helper to create updated State (immutable update pattern for AD)
function update_state(s::State{T};
    charge = s.charge,
    energy = s.energy,
    distance = s.distance,
    grammage = s.grammage,
    time = s.time,
    weight = s.weight,
    position = s.position,
    direction = s.direction,
    decayed = s.decayed
) where T
    State{T}(charge, energy, distance, grammage, time, weight, position, direction, decayed)
end

"""
    Locals{T<:Real}

Local properties of a propagation medium.
Differentiable with Zygote.

# Fields
- `density::T`: Material density in kg/m³
- `magnet::Vec3{T}`: Local magnetic field in T
"""
struct Locals{T<:Real}
    density::T
    magnet::Vec3{T}
end

Locals{T}(density::T) where T = Locals{T}(density, Vec3{T}(0, 0, 0))
Locals(density::Real) = Locals{typeof(density)}(density)
Locals(density::Real, magnet::Vec3) = Locals(density, magnet)

# ChainRulesCore rrule for Locals
function ChainRulesCore.rrule(::Type{Locals{T}}, density, magnet) where T
    locals = Locals{T}(density, magnet)
    function Locals_pullback(Δ)
        return (NoTangent(), Δ.density, Δ.magnet)
    end
    return locals, Locals_pullback
end

"""
    LocalsCallback{F}

Wrapper for a function that sets local properties.
The callback signature: (medium, state) -> (locals::Locals, step_hint::Real)
"""
struct LocalsCallback{F}
    func::F
end

(cb::LocalsCallback)(medium, state) = cb.func(medium, state)

"""
    Medium{T<:Real}

Description of a propagation medium.

# Fields
- `material::Int`: Material index in physics tables
- `locals_cb::Union{LocalsCallback, Nothing}`: Callback for local properties
"""
struct Medium{T<:Real}
    material::Int
    locals_cb::Union{LocalsCallback, Nothing}
    
    function Medium{T}(material::Int, locals_cb = nothing) where T
        new{T}(material, locals_cb)
    end
end

Medium(material::Int, locals_cb = nothing) = Medium{Float64}(material, locals_cb)

"""
    ContextMode

Mode flags for Monte Carlo transport.
"""
struct ContextMode
    energy_loss::EnergyLossMode
    decay::DecayMode
    direction::DirectionMode
    scattering::ScatteringMode
end

# Default modes
function ContextMode(;
    energy_loss::EnergyLossMode = ENERGY_LOSS_STRAGGLED,
    decay::DecayMode = DECAY_WEIGHTED,
    direction::DirectionMode = DIRECTION_FORWARD,
    scattering::ScatteringMode = SCATTERING_MIXED
)
    ContextMode(energy_loss, decay, direction, scattering)
end

"""
    ContextLimit{T<:Real}

External limits for Monte Carlo transport.
"""
struct ContextLimit{T<:Real}
    energy::T      # Min (forward) or max (backward) kinetic energy in GeV
    distance::T    # Maximum distance in m
    grammage::T    # Maximum grammage in kg/m²
    time::T        # Maximum proper time in m/c
end

function ContextLimit{T}(;
    energy::T = zero(T),
    distance::T = typemax(T),
    grammage::T = typemax(T),
    time::T = typemax(T)
) where T<:Real
    ContextLimit{T}(energy, distance, grammage, time)
end

ContextLimit(; kwargs...) = ContextLimit{Float64}(; kwargs...)

end # module Types

