using Test, Random, DiffPumas
using DiffPumas.Straggling: sample_ehs_event, compute_ehs_mean_free_path

mutable struct EHSFixedRNG <: AbstractRNG
    draws::Vector{Float64}
    index::Int
end
EHSFixedRNG(values::Real...) = EHSFixedRNG(collect(Float64, values), 0)
function Random.rand(rng::EHSFixedRNG)
    rng.index += 1
    return rng.draws[rng.index]
end

@testset "EHS rate-majorant thinning" begin
    # A controlled material has range(K)=K and lambda(K)=scale*(1+K)
    # at its knots. Hence an event at K=2 from reference K=1 must be
    # accepted with rate_event/rate_reference=(1/3)/(1/2)=2/3.
    p = create_physics(MUON; n_energies=4, K_min=1., K_max=8.,
        settings=PhysicsSettings(dedx_path="/nonexistent"))
    for (i, table) in enumerate(p.tables)
        table.mixed_stopping_power .= 1.
        table.csda_stopping_power .= 1.
        table.mixed_range .= table.energies
        table.csda_range .= table.energies
        table.elastic_path .= i .* (1 .+ table.energies) .* table.energies .* (table.energies .+ 2p.mass)
    end
    materials = (1, MaterialMixture(1), MaterialMixture([1, 2], [.3, .7]))
    for material in materials, backward in (true, false)
        ki, kf = backward ? (1., 4.) : (4., 1.)
        initial_range = property_range(p, ENERGY_LOSS_MIXED, material, ki)
        event_range = property_range(p, ENERGY_LOSS_MIXED, material, 2.)
        candidate = abs(event_range - initial_range)
        grammage = abs(property_range(p, ENERGY_LOSS_MIXED, material, kf) - initial_range)
        reference_path = compute_ehs_mean_free_path(p, material, 1.)
        first_draw = exp(-candidate / reference_path)
        sample(u) = sample_ehs_event(p, material, ki, kf, grammage,
            initial_range, 1., EHSFixedRNG(first_draw, u); backward=backward)

        accepted, x, energy = sample(.6)
        @test accepted
        @test x ≈ candidate rtol=1e-12
        @test energy ≈ 2. rtol=1e-12
        @test !first(sample(.8))
        @test first(sample(0.))
        @test !first(sample(1.))

        # Integrate the acceptance CDF over uniform variates: exactly 2/3
        # survive. The inverted ratio would accept every candidate.
        accepted_count = count(j -> first(sample((j-.5)/600)), 1:600)
        @test accepted_count == 400

        # No event at or beyond the segment endpoint, before acceptance draws.
        endpoint_rng = EHSFixedRNG(exp(-(grammage + .01) / reference_path))
        @test !first(sample_ehs_event(p, material, ki, kf, grammage,
            initial_range, 1., endpoint_rng; backward=backward))
        @test endpoint_rng.index == 1
        exact_rng = EHSFixedRNG(.5)
        exact_endpoint = -log(.5) * reference_path
        @test !first(sample_ehs_event(p, material, ki, kf, exact_endpoint,
            initial_range, 1., exact_rng; backward=backward))
        @test exact_rng.index == 1
    end
end
