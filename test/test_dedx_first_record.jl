using Test
using DiffPumas.DEDXLoader: load_dedx_file

@testset "DEDX units line preserves first physical record" begin
    header=["Incident particle is a Muon with M = 105.65839 MeV",
        "Index = 1: StandardRock", "<Z/A> = 0.5",
        "T [MeV] p [MeV/c] ion brems pair photo rad total range delta beta"]
    firstrow="1.0 14.5 20.0 0.1 0.2 0.3 0.6 20.6 0.04 0.0 0.136"
    secondrow="1.2 16.0 19.0 0.2 0.3 0.4 0.9 19.9 0.05 0.0 0.149"
    # Historical/generated files differ in their optional separators. Both
    # immediately adjacent records and blanks/comments/rules must retain 1 MeV.
    for separators in (String[],[""],["", "# data follows", repeat("-",80)])
        mktempdir() do directory
            file=joinpath(directory,"fixture.txt")
            write(file,join(vcat(header,separators,[firstrow,secondrow]),'\n')*"\n")
            data=load_dedx_file(file)
            @test data.material_name=="StandardRock"
            @test data.particle_mass≈0.10565839
            @test data.kinetic_energy==[0.001,0.0012]
            @test data.momentum≈[0.0145,0.016]
            @test data.ionization≈[20.0,19.0]*1e-4
            @test data.stopping_power≈[20.6,19.9]*1e-4
            @test data.csda_range≈[0.4,0.5]
            @test length(data.beta)==2
        end
    end
end
