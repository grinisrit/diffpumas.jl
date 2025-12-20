"""
    Context

Simulation context for Monte Carlo transport.
"""
module Context

using ..Constants
using ..Types
using ..Physics
using ..Transport
using Random

export SimulationContext, create_context, set_seed!, transport!

"""
    SimulationContext{T}

Manages simulation state and settings for Monte Carlo transport.

# Fields
- `physics`: Physics tables
- `mode`: Transport mode settings
- `event_mask`: Events that stop transport
- `limit`: External limits
- `accuracy`: MC integration accuracy
- `medium_cb`: Medium geometry callback
- `rng`: Random number generator
"""
mutable struct SimulationContext{T<:Real}
    physics::PhysicsTables{T}
    mode::ContextMode
    event_mask::Event
    limit::ContextLimit{T}
    accuracy::T
    medium_cb::Union{Function, Nothing}
    rng::AbstractRNG
    
    # Internal state
    random_data::Vector{T}
end

"""
    create_context(physics; kwargs...)

Create a new simulation context.

# Arguments
- `physics`: Physics tables

# Keyword Arguments
- `mode`: Transport mode (default: forward CSDA with scattering)
- `event_mask`: Events to stop on (default: none)
- `limit`: External limits
- `accuracy`: MC accuracy (default: 1%)
- `medium_cb`: Geometry callback
- `seed`: Random seed (nothing for random)
"""
function create_context(physics::PhysicsTables{T};
                        mode::ContextMode = ContextMode(),
                        event_mask::Event = EVENT_NONE,
                        limit::ContextLimit{T} = ContextLimit{T}(),
                        accuracy::T = T(DEFAULT_ACCURACY),
                        medium_cb::Union{Function, Nothing} = nothing,
                        seed::Union{Int, Nothing} = nothing) where T<:Real
    
    rng = if seed === nothing
        Random.default_rng()
    else
        Random.MersenneTwister(seed)
    end
    
    SimulationContext{T}(
        physics, mode, event_mask, limit, accuracy,
        medium_cb, rng, T[]
    )
end

"""
    set_seed!(context, seed)

Set the random seed for the context.
"""
function set_seed!(context::SimulationContext, seed::Int)
    context.rng = Random.MersenneTwister(seed)
    return nothing
end

"""
    get_medium(context, state)

Get medium and step info at current state position.
"""
function get_medium(context::SimulationContext{T}, state::State{T}) where T
    if context.medium_cb === nothing
        error("No medium callback defined")
    end
    
    return context.medium_cb(context, state)
end

"""
    transport!(context, state)

Transport a particle according to context settings.
Returns final state and event.

# Arguments
- `context`: Simulation context
- `state`: Initial particle state

# Returns
- `final_state`: Final particle state
- `event`: Event that stopped transport
- `media`: Initial and final media (if applicable)
"""
function transport!(context::SimulationContext{T}, state::State{T}) where T<:Real
    physics = context.physics
    
    # Check initial state
    if state.decayed
        return state, EVENT_VERTEX_DECAY, nothing
    end
    
    if state.weight <= zero(T)
        return state, EVENT_WEIGHT, nothing
    end
    
    # Get initial medium
    if context.medium_cb === nothing
        error("No medium callback defined for transport")
    end
    
    medium_result = context.medium_cb(context, state)
    if medium_result === nothing
        return state, EVENT_MEDIUM, nothing
    end
    
    medium, step_hint = medium_result
    if medium === nothing
        return state, EVENT_MEDIUM, nothing
    end
    
    initial_medium = medium
    current_state = state
    event = EVENT_NONE
    
    # Main transport loop
    max_steps = 10000
    for step in 1:max_steps
        # Check limits
        if has_event(context.event_mask, EVENT_LIMIT_ENERGY)
            if context.mode.direction == DIRECTION_FORWARD
                if current_state.energy <= context.limit.energy
                    event = EVENT_LIMIT_ENERGY
                    break
                end
            else
                if current_state.energy >= context.limit.energy
                    event = EVENT_LIMIT_ENERGY
                    break
                end
            end
        end
        
        if has_event(context.event_mask, EVENT_LIMIT_DISTANCE)
            if current_state.distance >= context.limit.distance
                event = EVENT_LIMIT_DISTANCE
                break
            end
        end
        
        if has_event(context.event_mask, EVENT_LIMIT_GRAMMAGE)
            if current_state.grammage >= context.limit.grammage
                event = EVENT_LIMIT_GRAMMAGE
                break
            end
        end
        
        if has_event(context.event_mask, EVENT_LIMIT_TIME)
            if current_state.time >= context.limit.time
                event = EVENT_LIMIT_TIME
                break
            end
        end
        
        # Get current medium and step size
        medium_result = context.medium_cb(context, current_state)
        if medium_result === nothing
            event = EVENT_MEDIUM
            break
        end
        
        medium, step_max = medium_result
        if medium === nothing
            event = EVENT_MEDIUM
            break
        end
        
        # Compute step size
        if step_max <= zero(T)
            step_max = T(1e6)  # Effectively infinite
        end
        
        # Apply accuracy limit
        step_kinetic = current_state.energy * context.accuracy / 
                       property_stopping_power(physics, context.mode.energy_loss, 
                                               medium.material, current_state.energy)
        
        step_size = min(step_max, step_kinetic)
        step_size = max(step_size, T(STEP_MIN))
        
        # Transport step
        new_state, step_event = transport_single_medium(
            physics, current_state, medium, step_size;
            mode = context.mode.energy_loss
        )
        
        current_state = new_state
        
        # Check for energy exhaustion
        if step_event == EVENT_LIMIT_ENERGY
            event = EVENT_LIMIT_ENERGY
            break
        end
        
        # Check weight
        if current_state.weight <= zero(T)
            event = EVENT_WEIGHT
            break
        end
    end
    
    return current_state, event, (initial_medium, medium)
end

"""
    random_uniform(context)

Get uniform random number in [0, 1].
"""
function random_uniform(context::SimulationContext{T}) where T
    return T(rand(context.rng))
end

"""
    random_exponential(context, λ)

Get exponential random variate with rate λ.
"""
function random_exponential(context::SimulationContext{T}, λ::T) where T
    u = random_uniform(context)
    return -log(u) / λ
end

end # module Context

