using Test
using DiffPumas

@testset "Loader entry points honor the same MDF definitions" begin
    mktempdir() do dir
        # Three tabulated energies keep this physics-generation regression small.
        # The blank line after units is accepted by the existing DEDX parser.
        root = joinpath(dir, "pumas-materials")
        datadir = joinpath(root, "dedx", "muon")
        mkpath(datadir)
        for (filename, name, zoa) in (("standard_rock", "StandardRock", 0.5),
                                       ("water", "Water", 0.55509),
                                       ("air", "Air", 0.49918))
            write(joinpath(datadir, filename * ".txt"), """
            Incident particle is a Muon with M = 105.65839 MeV
            Index = 0: $name
            Absorber with <Z/A> = $zoa
            T [MeV] p [MeV/c]

            1000 1100 1.8 0.001 0.001 0.001 0.003 1.803 550 0.0 0.99
            10000 10100 2.1 0.03 0.03 0.03 0.09 2.19 4900 0.0 0.999
            100000 100100 2.2 0.2 0.2 0.2 0.6 2.8 40000 0.0 0.9999
            """)
        end
        mdf = joinpath(@__DIR__, "..", "examples", "data", "materials.xml")
        withenv("PUMAS_MATERIALS_PATH" => root) do
            particle = load_or_create_physics(MUON;
                dump_path=joinpath(dir, "particle.pumas"), mdf_path=mdf)
            filename = load_or_create_physics(joinpath(dir, "filename.pumas");
                mdf_path=mdf, verbose=false)
            @test particle.materials == filename.materials
            @test particle.settings.dedx_path == filename.settings.dedx_path == root
            @test particle.tables[get_material_index(particle, "Water")].I ≈ 79.7e-9
            for table in particle.tables
                other = filename.tables[get_material_index(filename, table.name)]
                @test length(table.energies) == 3
                # Includes mixed stopping, straggling, and scattering tables,
                # which previously used different base-material recipes.
                @test all(field -> getfield(table, field) == getfield(other, field),
                          fieldnames(typeof(table)))
            end
            # A custom MDF must reuse its cache without requiring standard
            # material names. Missing custom composites must still be rebuilt.
            custom_mdf = joinpath(dir, "custom.xml")
            write(custom_mdf, """
                <pumas>
                  <element name="Q" Z="11" A="22" I="136.4" />
                  <material name="CustomRock" density="2.65">
                    <component name="Q" fraction="1" />
                  </material>
                  <composite name="CustomMix">
                    <component name="CustomRock" fraction="1" />
                  </composite>
                </pumas>
                """)
            custom_data = joinpath(datadir, "custom_rock.txt")
            write(custom_data, read(joinpath(datadir, "standard_rock.txt"), String))
            custom_dump = joinpath(dir, "custom.pumas")
            custom = load_or_create_physics(custom_dump; mdf_path=custom_mdf, verbose=false)
            original_bytes = read(custom_dump)
            rm(custom_data)
            cached = load_or_create_physics(custom_dump; mdf_path=custom_mdf, verbose=false)
            @test Set(keys(cached.materials)) == Set(["CustomRock", "CustomMix"])
            @test all(table -> length(table.energies) == 3, cached.tables)
            @test read(custom_dump) == original_bytes
            @test cached.tables[1].straggling == custom.tables[1].straggling
            write(custom_data, read(joinpath(datadir, "standard_rock.txt"), String))
            incomplete = DiffPumas.Physics.PhysicsTables(custom.particle, custom.mass,
                custom.ctau, custom.settings, Dict("CustomRock" => 1), [custom.tables[1]])
            @test save_physics(incomplete, custom_dump)
            rebuilt = load_or_create_physics(custom_dump; mdf_path=custom_mdf, verbose=false)
            @test haskey(rebuilt.materials, "CustomMix")
            @test all(table -> length(table.energies) == 3, rebuilt.tables)
        end
    end
end
