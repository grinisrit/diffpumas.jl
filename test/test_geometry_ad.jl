using Test
using DiffPumas
using Zygote
using Random

@testset "Backward geometry density derivatives" begin
    physics = create_physics(MUON; n_energies=50, K_min=0.01, K_max=1e6)
    rock = get_material_index(physics, "StandardRock")
    air = get_material_index(physics, "Air")
    density = 2650.0
    # Stay away from table knots and discrete transport regime boundaries.
    fd(f, x; h=1e-2) = (f(x + h) - f(x - h)) / (2h)
    state = State{Float64}(energy=10.0)

    @testset "Single step" begin
        for field in (:energy, :weight, :time)
            f(rho) = getproperty(DiffPumas.Geometry.transport_backward_step(
                physics, state, rock, rho, 0.7), field)
            derivative = Zygote.gradient(f, density)[1]
            @test derivative !== nothing
            @test isfinite(derivative)
            @test derivative ≈ fd(f, density) rtol=2e-5 atol=1e-14
        end
    end

    @testset "Adaptive layered transport" begin
        function final_energy(rho)
            geometry = TwoLayerGeometry{Float64}(10.0, rho, rock, air,
                                                -1, 0.0, 0.0, 40.0)
            return DiffPumas.Geometry.transport_backward_through_geometry(
                physics, geometry, state, 1e6).energy
        end
        derivative = Zygote.gradient(final_energy, density)[1]
        @test isfinite(derivative)
        @test derivative > 0
        @test derivative ≈ fd(final_energy, density) rtol=3e-4
    end

    @testset "Flux including documented straggling approximation" begin
        for (energy, thickness) in ((10.0, 10.0), (120.0, 3.0)), straggling in (false, true)
            f(rho) = compute_flux_differentiable(physics, rho, thickness, 45.0,
                                                energy, 1.0; straggling=straggling)
            flux, derivative = compute_flux_gradient(physics, density, thickness,
                                                     45.0, energy, 1.0;
                                                     straggling=straggling)
            @test flux > 0
            @test isfinite(derivative)
            @test derivative < 0
            @test derivative ≈ fd(f, density) rtol=5e-4
            # The public AD helper explicitly uses CSDA when no RNG is supplied.
            @test flux ≈ compute_flux_differentiable(physics, density, thickness,
                                                    45.0, energy, 1.0; straggling=false)
        end
    end

    @testset "MC helper integrates its deterministic gradient surrogate" begin
        seed = 77
        n = 3
        emin, emax = 2.0, 20.0
        result = run_backward_mc(physics; rock_density=density, rock_thickness=1.0,
            elevation=90.0, energy_min=emin, energy_max=emax, n_samples=n,
            compute_gradient=true, seed=seed, straggling=false, scattering=false)
        # With stochastic physics disabled, only energy/charge sampling consumes
        # RNG draws. Reconstruct that quadrature independently of the helper.
        rng = MersenneTwister(seed)
        span = log(emax / emin)
        samples = [begin
            energy = emin * exp(span * rand(rng))
            charge = rand(rng) > 0.5 ? 1.0 : -1.0
            (energy, charge, 2energy * span)
        end for _ in 1:n]
        surrogate(rho) = sum(weight * compute_flux_differentiable(
            physics, rho, 1.0, 90.0, energy, charge)
            for (energy, charge, weight) in samples) / n
        @test result.gradient_method == :deterministic_csda_sample_average
        @test result.gradient_flux ≈ surrogate(density)
        @test result.flux ≈ result.gradient_flux
        @test result.gradient ≈ fd(surrogate, density) rtol=5e-4
    end

    @testset "Zero-rock density sensitivity" begin
        _, derivative = compute_flux_gradient(physics, density, 0.0, 90.0, 10.0, 1.0)
        @test derivative == 0.0
        result = run_backward_mc(physics; rock_thickness=0.0, elevation=90.0,
            energy_min=10.0, n_samples=1, compute_gradient=true,
            straggling=false, scattering=false)
        @test result.gradient == 0.0
        @test isfinite(result.gradient_flux)
        @test result.flux ≈ result.gradient_flux
    end

    @testset "Stochastic straggling remains active" begin
        rng = MersenneTwister(27)
        energies = [DiffPumas.Geometry.transport_backward_step(
            physics, state, rock, density, 0.7; rng=rng, straggling=true).energy
                    for _ in 1:16]
        deterministic = DiffPumas.Geometry.transport_backward_step(
            physics, state, rock, density, 0.7).energy
        @test all(isfinite, energies)
        @test all(>=(state.energy), energies)
        @test length(unique(energies)) > 1
        @test any(!=(deterministic), energies)
    end
end
