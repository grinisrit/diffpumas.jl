using Test, Serialization
using DiffPumas
using DiffPumas.Physics
using DiffPumas.Loader
@testset "Versioned recipes and identity" begin
    p=create_physics(MUON;n_energies=8,K_min=100.,K_max=1e4,
        materials=[STANDARD_ROCK],settings=PhysicsSettings(dedx_path="/nonexistent"))
    md=Physics.material_metadata(p;required=true)
    @test md.table_version == 2
    @test length(md.recipes)==1
    @test_throws ErrorException Physics.material_metadata(deepcopy(p);required=true)
    GC.gc()
    @test Physics.material_metadata(p;required=true)==md
    mktempdir() do dir
        path=joinpath(dir,"new.pumas")
        @test save_physics(p,path)
        q=load_physics(path)
        @test q isa PhysicsTables
        @test Physics.material_metadata(q;required=true)==md
        @test q.tables !== p.tables
        q.tables[1].mixed_stopping_power[1] *= 2
        @test_throws ErrorException Physics.material_metadata(q;required=true)
        @test Physics.material_metadata(p;required=true)==md
        @test !save_physics(q,joinpath(dir,"mutated.pumas"))
        legacy=joinpath(dir,"legacy.pumas")
        open(io->serialize(io,p),legacy,"w")
        r=load_physics(legacy)
        @test r isa PhysicsTables
        @test Physics.material_metadata(r) === nothing
        @test_throws ErrorException Physics.material_metadata(r;required=true)
        data=open(deserialize,path)
        altered=merge(data.material_metadata,(recipe_fingerprint="bad",))
        open(io->serialize(io,merge(data,(material_metadata=altered,))),path,"w")
        @test load_physics(path) === nothing
    end
end
