using Test
using DiffPumas
using DiffPumas.Materials: compute_dcs_integral, dcs_bremsstrahlung_ssr,
    dcs_pair_production_ssr, dcs_photonuclear_drss
using DiffPumas.Constants: AVOGADRO_NUMBER

@testset "Radiative DCS moments and units" begin
    processes = (:bremsstrahlung, :pair_production, :photonuclear)
    # Independent official-C raw-DCS integration; provenance and quadrature
    # refinement are recorded in the fixture and its generator.
    for line in readlines(joinpath(@__DIR__, "data", "pumas_radiative_moments.csv"))
        (startswith(line, "#") || startswith(line, "process")) && continue
        fields = split(line, ',')
        process = processes[parse(Int, fields[1]) + 1]
        energy = parse(Float64, fields[2])
        mass, xlow, xhigh = parse.(Float64, fields[3:5])
        mode = parse(Int, fields[6])
        reference = parse(Float64, fields[7])
        # The full hard band includes physical endpoints and piecewise model
        # transitions: default table quadrature is less accurate there.
        tolerance = xhigh == 1. ? 3e-4 : 1e-6
        @test compute_dcs_integral(process, 11., 22., 149e-9, mass,
            energy, xlow, xhigh, mode) ≈ reference rtol=tolerance
        if xhigh == 1.
            @test compute_dcs_integral(process, 11., 22., 149e-9, mass,
                energy, xlow, xhigh, mode; npoints=1800) ≈ reference rtol=2e-5
        end
    end

    for (process, raw) in zip(processes,
            (dcs_bremsstrahlung_ssr, dcs_pair_production_ssr, dcs_photonuclear_drss))
        energy = 100.
        # A vanishing-width energy band has σ ≈ (dσ/dq) Δq times
        # number of atoms/kg, with no extra energy-dependent normalization.
        x, dx = .2, 1e-5
        sigma = compute_dcs_integral(process, 11., 22., 149e-9, MUON_MASS,
            energy, x - dx / 2, x + dx / 2, 0)
        local_density = raw(11., 22., MUON_MASS, energy, x * energy) *
            AVOGADRO_NUMBER / (22. * .001)
        @test sigma / (energy * dx) ≈ local_density rtol=1e-7

        # Energy-weighted moments of a positive measure obey support bounds.
        moments = [compute_dcs_integral(process, 11., 22., 149e-9, MUON_MASS,
            energy, .1, .5, mode) for mode in 0:2]
        @test all(>(0), moments)
        @test .1energy * moments[1] <= moments[2] <= .5energy * moments[1]
        @test .1energy * moments[2] <= moments[3] <= .5energy * moments[2]
    end
end
