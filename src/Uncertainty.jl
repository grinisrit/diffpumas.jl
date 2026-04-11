"""
    Uncertainty

Shared helpers for estimating Monte Carlo and transport-systematic
uncertainties. The first implementation focuses on transport settings:

- `straggling`
- `scattering`
- `energy_threshold_low`

The helpers are intentionally generic so examples with custom integrators can
reuse the same budget structure instead of duplicating the variation logic.
"""
module Uncertainty

using ..Geometry: compute_flux, PRIMARY_ALTITUDE

export TransportVariation, SystematicSourceResult, TransportUncertaintyResult
export estimate_transport_uncertainty, compute_flux_uncertainty
export relative_uncertainty_percent

"""
    TransportVariation

Transport settings for a single nominal or systematic-variation run.
"""
struct TransportVariation{T<:Real}
    straggling::Bool
    scattering::Bool
    energy_threshold_low::T
    seed::Int
end

"""
    SystematicSourceResult

Result for a single systematic source. `sigma` is the envelope obtained from
the absolute deviations between the nominal value and all source variants.
"""
struct SystematicSourceResult{S}
    sigma::S
    reference_value::S
    variant_values::Vector{S}
    variant_sigmas::Vector{S}
    variant_labels::Vector{String}
end

"""
    TransportUncertaintyResult

Combined MC + transport-systematic uncertainty budget for an observable.
"""
struct TransportUncertaintyResult{S,T<:Real}
    value::S
    sigma_mc::S
    sigma_syst::S
    sigma_total::S
    sources::Dict{Symbol,SystematicSourceResult{S}}
    nominal::TransportVariation{T}
end

_zero_like(x::Number) = zero(x)
_zero_like(x::AbstractArray) = map(zero, x)

function _evaluate_source(evaluate, nominal_value, variants::Vector{TransportVariation{T}},
                          labels::Vector{String}) where T<:Real
    S = typeof(nominal_value)
    source_sigma = _zero_like(nominal_value)
    variant_values = S[]
    variant_sigmas = S[]

    for variation in variants
        value, sigma_mc = evaluate(variation)
        push!(variant_values, value)
        push!(variant_sigmas, sigma_mc)
        source_sigma = max.(source_sigma, abs.(value .- nominal_value))
    end

    return SystematicSourceResult(source_sigma, nominal_value, variant_values, variant_sigmas, labels)
end

function _collect_threshold_variants(nominal::TransportVariation{T},
                                     threshold_factors) where T<:Real
    variants = TransportVariation{T}[]
    labels = String[]

    nominal_threshold = nominal.energy_threshold_low
    nominal_threshold <= zero(T) && return variants, labels

    seen = Set{T}()
    for factor_raw in threshold_factors
        factor = T(factor_raw)
        threshold = factor * nominal_threshold
        threshold <= zero(T) && continue
        isapprox(threshold, nominal_threshold; atol=eps(T), rtol=sqrt(eps(T))) && continue
        threshold in seen && continue
        push!(seen, threshold)
        push!(variants, TransportVariation(nominal.straggling, nominal.scattering, threshold, nominal.seed))
        push!(labels, "threshold=$(threshold)")
    end

    return variants, labels
end

"""
    estimate_transport_uncertainty(evaluate; kwargs...)

Estimate the uncertainty budget for an observable.

`evaluate` must accept a single `TransportVariation` argument and return
`(value, sigma_mc)`, where both entries can be either scalars or arrays with a
shared shape.
"""
function estimate_transport_uncertainty(evaluate;
                                        straggling::Bool = true,
                                        scattering::Bool = true,
                                        energy_threshold_low::T = T(100.0),
                                        seed::Int = 42,
                                        threshold_factors = (0.5, 2.0)) where T<:Real
    nominal = TransportVariation(straggling, scattering, energy_threshold_low, seed)
    nominal_value, sigma_mc = evaluate(nominal)

    S = typeof(nominal_value)
    sources = Dict{Symbol,SystematicSourceResult{S}}()

    straggling_variants = TransportVariation{T}[
        TransportVariation(!nominal.straggling, nominal.scattering,
                           nominal.energy_threshold_low, nominal.seed)
    ]
    sources[:straggling] = _evaluate_source(
        evaluate,
        nominal_value,
        straggling_variants,
        ["straggling=$(!nominal.straggling)"],
    )

    scattering_variants = TransportVariation{T}[
        TransportVariation(nominal.straggling, !nominal.scattering,
                           nominal.energy_threshold_low, nominal.seed)
    ]
    sources[:scattering] = _evaluate_source(
        evaluate,
        nominal_value,
        scattering_variants,
        ["scattering=$(!nominal.scattering)"],
    )

    threshold_variants, threshold_labels = _collect_threshold_variants(nominal, threshold_factors)
    sources[:energy_threshold_low] = _evaluate_source(
        evaluate,
        nominal_value,
        threshold_variants,
        threshold_labels,
    )

    sigma_syst = _zero_like(nominal_value)
    for source in values(sources)
        sigma_syst = sigma_syst .+ source.sigma .^ 2
    end
    sigma_syst = sqrt.(sigma_syst)
    sigma_total = sqrt.(sigma_mc .^ 2 .+ sigma_syst .^ 2)

    return TransportUncertaintyResult(
        nominal_value,
        sigma_mc,
        sigma_syst,
        sigma_total,
        sources,
        nominal,
    )
end

"""
    compute_flux_uncertainty(physics, rock_density, rock_thickness, elevation,
                             energy_min, energy_max; kwargs...)

Convenience wrapper around `Geometry.compute_flux(...)` that evaluates the
nominal MC flux and the first-pass transport systematic variations using common
random numbers (shared seed).
"""
function compute_flux_uncertainty(physics,
                                  rock_density::T,
                                  rock_thickness::T,
                                  elevation::T,
                                  energy_min::T,
                                  energy_max::T;
                                  n_samples::Int = 1000,
                                  seed::Int = 42,
                                  straggling::Bool = true,
                                  scattering::Bool = true,
                                  energy_threshold_low::T = T(100.0),
                                  threshold_factors = (0.5, 2.0),
                                  porous_material::Int = -1,
                                  porous_density::T = zero(T),
                                  porous_thickness::T = zero(T),
                                  primary_altitude::T = T(PRIMARY_ALTITUDE)) where T<:Real
    function evaluate(variation::TransportVariation{T})
        return compute_flux(
            physics,
            rock_density,
            rock_thickness,
            elevation,
            energy_min,
            energy_max;
            n_samples = n_samples,
            seed = variation.seed,
            straggling = variation.straggling,
            scattering = variation.scattering,
            energy_threshold_low = variation.energy_threshold_low,
            porous_material = porous_material,
            porous_density = porous_density,
            porous_thickness = porous_thickness,
            primary_altitude = primary_altitude,
        )
    end

    return estimate_transport_uncertainty(
        evaluate;
        straggling = straggling,
        scattering = scattering,
        energy_threshold_low = energy_threshold_low,
        seed = seed,
        threshold_factors = threshold_factors,
    )
end

"""
    relative_uncertainty_percent(value, sigma; floor = 1e-30)

Convert an absolute uncertainty into a relative percentage while guarding
against divisions by zero.
"""
_safe_abs_denominator(value::Number, floor::Real) = max(abs(value), floor)
_safe_abs_denominator(value::AbstractArray, floor::Real) =
    map(v -> max(abs(v), floor), value)

function relative_uncertainty_percent(value, sigma; floor::Real = 1e-30)
    return 100 .* sigma ./ _safe_abs_denominator(value, floor)
end

end # module Uncertainty
