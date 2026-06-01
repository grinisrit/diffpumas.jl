using Test
using DiffPumas
using Zygote
using LinearAlgebra
using Random

@testset "DiffPumas.jl" begin
    
    @testset "Constants" begin
        @test MUON_MASS ≈ 0.10565839 rtol=1e-6
        @test TAU_MASS ≈ 1.77682 rtol=1e-4
        @test ELECTRON_MASS ≈ 0.510998910e-3 rtol=1e-6
        @test ALPHA_EM ≈ 7.2973525693e-3 rtol=1e-10
    end
    
    @testset "Types" begin
        # Test State creation
        state = State{Float64}(
            charge = 1.0,
            energy = 10.0,
            weight = 1.0,
            position = Vec3(0.0, 0.0, 0.0),
            direction = Vec3(0.0, 0.0, -1.0)
        )
        @test state.charge == 1.0
        @test state.energy == 10.0
        @test norm(state.direction) ≈ 1.0
        
        # Test Locals
        locals = Locals(2650.0)
        @test locals.density == 2650.0
        @test norm(locals.magnet) == 0.0
        
        # Test Event operations
        e1 = EVENT_LIMIT_ENERGY
        e2 = EVENT_MEDIUM
        combined = combine_events(e1, e2)
        @test has_event(combined, e1)
        @test has_event(combined, e2)
        @test !has_event(combined, EVENT_VERTEX_DECAY)
    end
    
    @testset "Materials" begin
        # Test standard rock
        @test STANDARD_ROCK.density ≈ 2650.0
        @test length(STANDARD_ROCK.elements) == 8
        
        # Test stopping power
        dedx = electronic_stopping_power(STANDARD_ROCK, MUON_MASS, 10.0)
        @test dedx > 0
        @test isfinite(dedx)
        
        # Test elastic DCS
        dcs = elastic_dcs(14.0, 28.0, MUON_MASS, 10.0, 0.1)
        @test dcs >= 0
        @test isfinite(dcs)
    end
    
    @testset "Physics" begin
        # Create physics tables
        physics = create_physics(MUON; n_energies=50, K_min=0.01, K_max=1e6)
        
        @test physics.particle == MUON
        @test physics.mass ≈ MUON_MASS
        @test physics.ctau ≈ MUON_C_TAU
        @test length(physics.tables) == 2  # Rock and Air
        
        # Test material lookup
        rock_idx = get_material_index(physics, "StandardRock")
        @test rock_idx > 0
        
        # Test property interpolation
        range = property_range(physics, ENERGY_LOSS_CSDA, rock_idx, 10.0)
        @test range > 0
        @test isfinite(range)
        
        dedx = property_stopping_power(physics, ENERGY_LOSS_CSDA, rock_idx, 10.0)
        @test dedx > 0
        @test isfinite(dedx)
    end
    
    @testset "Transport" begin
        physics = create_physics(MUON; n_energies=50, K_min=0.01, K_max=1e6)
        rock_idx = get_material_index(physics, "StandardRock")
        
        # Create initial state
        state = State{Float64}(
            charge = 1.0,
            energy = 100.0,
            weight = 1.0,
            position = Vec3(0.0, 0.0, 0.0),
            direction = Vec3(0.0, 0.0, -1.0)
        )
        
        # Test transport with density
        density = 2650.0
        new_state, event = transport_with_density(physics, state, rock_idx, density, 1.0)
        
        @test new_state.energy < state.energy  # Energy lost
        @test new_state.distance > 0  # Distance traveled
        @test new_state.grammage > 0  # Grammage accumulated
        @test new_state.weight <= state.weight  # Weight decreased (decay)
    end
    
    @testset "Differentiability" begin
        physics = create_physics(MUON; n_energies=50, K_min=0.01, K_max=1e6)
        rock_idx = get_material_index(physics, "StandardRock")
        
        # Test gradient of transport w.r.t. density
        state = State{Float64}(
            charge = 1.0,
            energy = 10.0,
            weight = 1.0,
            position = Vec3(0.0, 0.0, 0.0),
            direction = Vec3(0.0, 0.0, -1.0)
        )
        
        # Function to differentiate
        function loss_fn(density)
            new_state, _ = transport_with_density(physics, state, rock_idx, density, 1.0)
            return new_state.grammage
        end
        
        # Compute gradient
        density = 2650.0
        grad = Zygote.gradient(loss_fn, density)[1]
        
        # Gradient should be approximately distance (= 1.0)
        @test grad !== nothing
        @test grad ≈ 1.0 rtol=0.1
    end
    
    @testset "Geometry" begin
        physics = create_physics(MUON; n_energies=50, K_min=0.01, K_max=1e6)
        
        # Test flux model
        flux = flux_gccly(1.0, 10.0, 1.0)
        @test flux > 0
        @test isfinite(flux)
        
        # Test flux gradient (simplified)
        rock_density = 2650.0
        rock_thickness = 10.0
        elevation = 45.0
        energy_final = 10.0
        charge = 1.0
        
        flux_val, grad = compute_flux_gradient(
            physics, rock_density, rock_thickness, elevation, energy_final, charge
        )
        
        @test isfinite(flux_val)
        # Note: gradient might be zero or very small for this simple test
    end

    @testset "Transport Uncertainty" begin
        @testset "Budget combination math" begin
            function vector_budget_evaluator(variation)
                base = (variation.straggling ? 2.0 : 0.0) +
                       (variation.scattering ? 3.0 : 0.0) +
                       variation.energy_threshold_low / 10.0
                value = [100.0 + base, 200.0 + 2.0 * base]
                sigma_mc = [0.25, 0.50]
                return value, sigma_mc
            end

            budget = estimate_transport_uncertainty(
                vector_budget_evaluator;
                straggling = true,
                scattering = true,
                energy_threshold_low = 10.0,
                seed = 7,
                threshold_factors = (0.5, 2.0),
            )

            @test budget.value ≈ [106.0, 212.0]
            @test budget.sigma_mc ≈ [0.25, 0.50]
            @test budget.sources[:straggling].sigma ≈ [2.0, 4.0]
            @test budget.sources[:scattering].sigma ≈ [3.0, 6.0]
            @test budget.sources[:energy_threshold_low].sigma ≈ [1.0, 2.0]
            @test budget.sigma_syst ≈ sqrt.([14.0, 56.0])
            @test budget.sigma_total ≈ sqrt.([14.0 + 0.25^2, 56.0 + 0.50^2])
        end

        @testset "Same-seed evaluation is deterministic" begin
            function seeded_evaluator(variation)
                rng = MersenneTwister(variation.seed)
                value = 1.0 +
                        (variation.straggling ? 0.3 : -0.2) +
                        (variation.scattering ? 0.5 : -0.4) +
                        variation.energy_threshold_low / 100.0 +
                        rand(rng)
                sigma_mc = rand(rng) / 10.0
                return value, sigma_mc
            end

            budget_a = estimate_transport_uncertainty(
                seeded_evaluator;
                straggling = true,
                scattering = true,
                energy_threshold_low = 20.0,
                seed = 17,
                threshold_factors = (0.5, 2.0),
            )
            budget_b = estimate_transport_uncertainty(
                seeded_evaluator;
                straggling = true,
                scattering = true,
                energy_threshold_low = 20.0,
                seed = 17,
                threshold_factors = (0.5, 2.0),
            )

            @test budget_a.value == budget_b.value
            @test budget_a.sigma_mc == budget_b.sigma_mc
            @test budget_a.sigma_syst == budget_b.sigma_syst
            @test budget_a.sigma_total == budget_b.sigma_total
            @test budget_a.sources[:straggling].variant_values == budget_b.sources[:straggling].variant_values
            @test budget_a.sources[:energy_threshold_low].variant_values == budget_b.sources[:energy_threshold_low].variant_values
        end

        @testset "Flux uncertainty smoke test" begin
            physics = create_physics(MUON; n_energies=30, K_min=0.05, K_max=1e4)
            budget = compute_flux_uncertainty(
                physics,
                2650.0,
                25.0,
                55.0,
                1.0,
                50.0;
                n_samples = 8,
                seed = 11,
                straggling = true,
                scattering = true,
                energy_threshold_low = 50.0,
                threshold_factors = (0.5, 2.0),
            )

            @test isfinite(budget.value)
            @test isfinite(budget.sigma_mc)
            @test isfinite(budget.sigma_syst)
            @test isfinite(budget.sigma_total)
            @test budget.sigma_mc >= 0.0
            @test budget.sigma_syst >= 0.0
            @test budget.sigma_total >= budget.sigma_mc
            @test budget.sigma_total >= budget.sigma_syst
            @test length(budget.sources[:energy_threshold_low].variant_values) == 2
        end
    end
    
    @testset "Turtle ECEF" begin
        using DiffPumas.Turtle

        # Round-trip: geodetic -> ECEF -> geodetic
        lat0, lon0, alt0 = 45.764, 2.955, 500.0
        ecef = ecef_from_geodetic(lat0, lon0, alt0)
        lat1, lon1, alt1 = ecef_to_geodetic(ecef)

        @test lat1 ≈ lat0 atol=1e-6
        @test lon1 ≈ lon0 atol=1e-6
        @test alt1 ≈ alt0 atol=0.01

        # Pole test
        ecef_pole = ecef_from_geodetic(90.0, 0.0, 0.0)
        lat_p, lon_p, alt_p = ecef_to_geodetic(ecef_pole)
        @test lat_p ≈ 90.0 atol=1e-6
        @test alt_p ≈ 0.0 atol=0.1

        # Horizontal direction round-trip
        dir = ecef_from_horizontal(lat0, lon0, 26.0, 5.0)
        az, el = ecef_to_horizontal(lat0, lon0, dir)
        @test az ≈ 26.0 atol=1e-4
        @test el ≈ 5.0 atol=1e-4

        # Vertical direction
        dir_up = ecef_from_horizontal(0.0, 0.0, 0.0, 90.0)
        az_u, el_u = ecef_to_horizontal(0.0, 0.0, dir_up)
        @test el_u ≈ 90.0 atol=1e-4
    end

    @testset "Turtle Projections" begin
        using DiffPumas.Turtle

        # UTM round-trip (zone 31N, centre of France)
        proj = UTMProjection(31, 'N')
        lat0, lon0 = 45.764, 2.955
        x, y = projection_project(proj, lat0, lon0)
        lat1, lon1 = projection_unproject(proj, x, y)
        @test lat1 ≈ lat0 atol=1e-6
        @test lon1 ≈ lon0 atol=1e-6

        # Lambert 93 round-trip
        lproj = LambertProjection("93")
        x2, y2 = projection_project(lproj, lat0, lon0)
        lat2, lon2 = projection_unproject(lproj, x2, y2)
        @test lat2 ≈ lat0 atol=1e-4
        @test lon2 ≈ lon0 atol=1e-4

        # Parse projection string
        p1 = parse_projection("UTM 31N")
        @test p1 isa UTMProjection
        p2 = parse_projection("Lambert 93")
        @test p2 isa LambertProjection
    end

    @testset "Turtle ElevationMap" begin
        using DiffPumas.Turtle

        info = MapInfo(11, 11, (0.0, 10.0), (0.0, 10.0), (0.0, 100.0))
        m = map_create(info)

        # Fill with a plane z = x + 2*y
        for ix in 0:10
            for iy in 0:10
                map_fill(m, ix, iy, Float64(ix + 2 * iy))
            end
        end

        # Check node retrieval
        x, y, z = map_node(m, 5, 3)
        @test x ≈ 5.0
        @test y ≈ 3.0
        @test z ≈ 11.0

        # Check interpolation
        elev, inside = map_elevation(m, 5.5, 3.5)
        @test inside
        @test elev ≈ 12.5 atol=0.01

        # Check outside
        _, outside = map_elevation(m, -1.0, 5.0)
        @test !outside

        # Check gradient (should be ~(1, 2) for z = x + 2y)
        gx, gy, gin = map_gradient(m, 5.0, 5.0)
        @test gin
        @test gx ≈ 1.0 atol=0.2
        @test gy ≈ 2.0 atol=0.2
    end

    @testset "Turtle Stepper" begin
        using DiffPumas.Turtle

        s = stepper_create()
        @test stepper_slope_get(s) ≈ 0.4
        @test stepper_resolution_get(s) ≈ 0.01
        @test stepper_range_get(s) ≈ 1.0

        stepper_add_flat(s, 0.0)
        stepper_slope_set(s, 1.0)

        # Position on the flat surface at equator
        pos, didx = stepper_position(s, 0.0, 0.0, 1.0, 0)
        @test didx >= 0

        # Step at the position (no direction = just sample)
        sample, ds = stepper_step(s, pos)
        @test ds >= 0.0
    end

    @testset "Loader" begin
        physics = create_physics(MUON; n_energies=20, K_min=0.1, K_max=1e3)

        # Test save/load (using temp file)
        tmp_path = tempname() * ".pumas"
        @test save_physics(physics, tmp_path)

        loaded = load_physics(tmp_path)
        @test loaded !== nothing
        @test loaded.particle == physics.particle
        @test length(loaded.tables) == length(physics.tables)

        # Clean up
        rm(tmp_path, force=true)
    end

    include("test_tomography.jl")
end

println("\nAll tests passed! ✓")
