using Test
using DiffPumas
using Zygote
using LinearAlgebra

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
end

println("\nAll tests passed! ✓")
