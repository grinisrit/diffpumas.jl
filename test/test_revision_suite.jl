# Standalone: julia --project=. test/test_revision_suite.jl
using Test
include(joinpath(@__DIR__, "..", "examples", "revision_suite.jl"))
@testset "TetGen Julia connectivity" begin
    mesh=RawTetGenIO{Float64}()
    mesh.pointlist=[0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    mesh.tetrahedronlist=reshape(Cint[1,2,3,4],4,1)
    @test tetra_centroids(mesh) ≈ fill(0.25,3,1)
    @test build_tetra_neighbors(mesh)==zeros(Int,1,4)
    volume=TetraVolume(mesh,zeros(Int,1,4),tetra_centroids(mesh),falses(1),1.0,1.0)
    face,distance,hit=nearest_tetra_face_intersection(volume,1,Point(0.1,0.1,0.1),(0.0,0.0,1.0))
    @test face>0
    @test distance≈0.7
    @test hit.z≈0.8
    triangle=Triangle(Point(0.0,0.0,1.0),Point(1.0,0.0,1.0),Point(0.0,1.0,1.0))
    boundary=tolerant_tetra_face_hit(Point(-1e-13,0.2,0.0),(0.0,0.0,1.0),triangle)
    @test boundary !== nothing && boundary[1]≈1.0
    @test tolerant_tetra_face_hit(Point(-0.01,0.2,0.0),(0.0,0.0,1.0),triangle) === nothing
    @test point_in_tetrahedron(Point(0.1,0.1,0.1),Int.(mesh.tetrahedronlist[:,1]),mesh.pointlist)
    mesh.tetrahedronlist[4,1]=5
    @test_throws ArgumentError tetra_centroids(mesh)
end
@testset "Adjoint finite difference controls" begin
    original=get_adjoint_fd_step()
    try
        @test set_adjoint_fd_step!(1e-5)==1e-5
        @test get_adjoint_fd_step()==1e-5
        for invalid in (0.0,-1.0,Inf,NaN,0.1)
            @test_throws ArgumentError set_adjoint_fd_step!(invalid)
        end
    finally
        set_adjoint_fd_step!(original)
    end
end
@testset "Revision option bounds" begin
    @test_throws ErrorException revision_options(["--samples","0"])
    @test_throws ErrorException revision_options(["--rays-step","0"])
    @test_throws ErrorException revision_options(["--unknown","1"])
end

@testset "Measured robustness controls" begin
    saved=copy(ARGS)
    try
        empty!(ARGS)
        append!(ARGS,["--papermatch-fit-only","--papermatch-prior-cutoff","350",
            "--papermatch-mask-ratio","Inf","--papermatch-calibration-water","0.3"])
        opts=parse_commandline()
        @test opts.papermatch && opts.papermatch_only && opts.papermatch_fit_only
        @test opts.papermatch_prior_cutoff==350
        @test isinf(opts.papermatch_mask_ratio)
        @test opts.papermatch_calibration_water==0.3
        @test opts.csda_quadrature=="midpoint"
        @test opts.n_samples==32
    finally
        empty!(ARGS);append!(ARGS,saved)
    end
end

@testset "Masked forward/Jacobian consistency" begin
    toy=w->([w[1]+2w[2]],sparse(reshape([1.0,2.0],1,2)))
    restricted=masked_material_operator(toy,Bool[true,false])
    w=[0.2,0.8]
    prediction,J=restricted(w)
    @test prediction≈[0.2]
    @test Matrix(J)≈[1.0 0.0]
    for c in eachindex(w)
        h=1e-6;delta=zeros(2);delta[c]=h
        fd=(first(restricted(w+delta))-first(restricted(w-delta)))/(2h)
        @test fd≈Vector(J[:,c]) atol=1e-10
    end
end

@testset "Validation angular configuration" begin
    @test validation_zeniths(revision_options(String[])) == [0,15,30,45,60,75]
    @test validation_zeniths(revision_options(["--validation-zeniths","75,0,60"])) == [0,60,75]
    for invalid in ("0,90","0,NaN","0,0")
        @test_throws ErrorException revision_options(["--validation-zeniths",invalid])
    end
end

@testset "High-zenith validation geometry" begin
    emap=LVDTopo.build_elevation_map()
    mat=MaterialConfig(1,2,3,4,2710.0,1000.0,0.0,0.0)
    vol=create_voxel_volume(emap,mat;half_width_km=3.0,surface_step_km=1.5,nz=8)
    paths=precompute_paths(vol,emap,[60.0,75.0],[0.0,90.0,180.0,270.0])
    @test length(paths)==8
    @test all(p->p.valid && !isempty(p.segments),paths)
    @test all(p->isfinite(p.surface_distance) && p.surface_distance>0,paths)
    @test all(p->p.distance_in_volume>0 && p.remaining_rock_distance>=0,paths)
end

@testset "Seeded selected/population zenith coverage" begin
    paths=[(zenith=t,surface_distance=1000.0+10b) for (b,t) in enumerate([0,15,30,45,60,75])]
    volume=(centroids=[zeros(1,8);zeros(1,8);reshape(1.0:8.0,1,8)],)
    J=sparse([10.0b+c for b in 1:6,c in 1:8])
    one=validation_pairs(paths,volume,J,12,MersenneTwister(42),[0,15,30,45,60,75])
    two=validation_pairs(paths,volume,J,12,MersenneTwister(42),[0,15,30,45,60,75])
    @test one==two
    @test length(unique(one))==12
    @test Set(paths[p.bin].zenith for p in one)==Set([0,15,30,45,60,75])
end

@testset "Fixed-path sensitivity envelope excludes scattering" begin
    variations=NamedTuple[]
    evaluate=v->begin
        push!(variations,v)
        (10.0+2v.straggling+v.energy_threshold_low/1000,0.3)
    end
    budget=fixed_path_energy_uncertainty(evaluate;straggling=true,energy_threshold_low=100.0,seed=42)
    @test budget.value≈12.1
    @test budget.sigma_syst≈hypot(2.0,0.1)
    @test budget.sigma_total≈hypot(0.3,hypot(2.0,0.1))
    @test length(variations)==4
    @test all(v->!v.scattering && v.seed==42,variations)
end

@testset "Strict physics loading and content-bound MC cache" begin
    mktempdir() do directory
        missing=joinpath(directory,"missing.dump")
        @test_throws ErrorException load_tomography_physics(missing)
        @test !ispath(missing)
        dump=joinpath(directory,"physics.dump");write(dump,"invalid physics bytes")
        original=read(dump)
        with_logger(NullLogger()) do
            @test_throws ErrorException load_tomography_physics(dump)
        end
        @test read(dump)==original
        source=joinpath(directory,"source.jl");write(source,"model v1")
        physics=(loss=[1.0,2.0],)
        paths=[(zenith=0.0,segments=[(cell=1,distance=10.0),]),
               (zenith=30.0,segments=[(cell=2,distance=12.0),])]
        config=(dump_path=dump,seed=42,energy_min=.001,energy_max=1e9)
        key=(;p=physics,site=(height=30000.0,),rays=paths,bins=[1,2],options=config)->
            inverse_mc_cache_identity(p,Bool[false,false],:material,site,rays,[.2,.3],bins,options;source_paths=[source])
        identity=key()
        @test identity==key(p=deepcopy(physics),rays=deepcopy(paths))
        @test identity!=key(p=(loss=[1.0,2.1],))
        @test identity!=key(site=(height=31000.0,))
        @test identity!=key(rays=reverse(paths))
        @test identity!=key(bins=[2,1])
        @test identity!=key(options=merge(config,(seed=43,)))
        write(dump,"other physics bytes");@test identity!=key();write(dump,original)
        write(source,"model v2");@test identity!=key();write(source,"model v1")
        cache=joinpath(directory,"observations.bin");observations=[1.0,2.0]
        Serialization.serialize(cache,(sig=:historical,obs=observations))
        @test read_inverse_mc_cache(cache,identity,2)===nothing
        record=(schema=3,identity=identity,obs=observations,sigma_mc=[0.1,0.2],
            observations_sha256=serialized_content_sha256(observations),sigma_sha256=serialized_content_sha256([0.1,0.2]))
        Serialization.serialize(cache,record)
        @test read_inverse_mc_cache(cache,identity,2)==(obs=observations,sigma_mc=[0.1,0.2])
        @test read_inverse_mc_cache(cache,key(bins=[2,1]),2)===nothing
        @test read_inverse_mc_cache(cache,identity,1)===nothing
        diagnostic_paths=[(zenith=0.,azimuth=0.,surface_distance=1000.),(zenith=45.,azimuth=0.,surface_distance=1500.)]
        diagnostic_args=(inverse_mc_samples=256,seed=42)
        verified=read_inverse_mc_cache(cache,identity,2)
        csv=export_inverse_mc_diagnostics(cache,identity,diagnostic_paths,[1,2],verified.obs,verified.sigma_mc,diagnostic_args;cache_hit=true)
        records=readlines(csv)
        @test length(records)==3
        @test split(records[2],',')[7]=="0.1"
        @test split(records[3],',')[7]=="0.2"
        @test endswith(records[2],",true")
        Serialization.serialize(cache,merge(record,(sigma_mc=[0.1,0.3],)))
        @test read_inverse_mc_cache(cache,identity,2)===nothing
        Serialization.serialize(cache,merge(record,(sigma_mc=[0.1,NaN],sigma_sha256=serialized_content_sha256([0.1,NaN]))))
        @test read_inverse_mc_cache(cache,identity,2)===nothing
        Serialization.serialize(cache,merge(record,(obs=[1.0,2.1],)))
        @test read_inverse_mc_cache(cache,identity,2)===nothing
    end
end

@testset "Calibration identities bind frozen correction and energy nodes" begin
    mktempdir() do directory
        dump=joinpath(directory,"physics.dump");write(dump,"identity fixture")
        source=joinpath(directory,"source.jl");write(source,"source fixture")
        args=(dump_path=dump,)
        energies=[EnergySample(10.,0.,1.),EnergySample(20.,0.,2.)]
        initial=(enabled=true,resid_a=1.0)
        context(;coefficients=initial,nodes=energies,pass=1)=calibration_diagnostic_context(args,coefficients,nodes;fit_geometric=false,calibration_pass=pass)
        identity(ctx)=inverse_mc_cache_identity(:physics,Bool[false],:material,:site,:paths,[.3],[1],ctx;source_paths=[source])
        baseline=identity(context())
        @test baseline!=identity(context(coefficients=(enabled=true,resid_a=1.1)))
        @test baseline!=identity(context(nodes=[EnergySample(11.,0.,1.),energies[2]]))
        @test baseline!=identity(context(pass=2))
    end
end

@testset "Production cache hit reexports verified MC sampling errors" begin
    mktempdir() do directory
        dump=joinpath(directory,"physics.dump");write(dump,"cache identity fixture")
        args=(dump_path=dump,inverse_mc_samples=256,seed=42)
        material=MaterialConfig(1,2,3,4,2710.,1000.,0.,0.)
        site=SiteConfig(1000.,30000.)
        paths=[DirectionalPath(0.,0.,[CellSegment(1,100.)],100.,100.,0.,100.,true),
               DirectionalPath(45.,90.,[CellSegment(1,150.)],150.,150.,0.,100.,true)]
        valid=[2,1];water=[.3];shallow=Bool[false]
        identity=inverse_mc_cache_identity(:fixture_physics,shallow,material,site,paths,water,valid,args)
        cache=joinpath(directory,"observations.bin")
        observations=[2.,1.];errors=[.2,.1]
        Serialization.serialize(cache,(schema=3,identity=identity,obs=observations,sigma_mc=errors,
            observations_sha256=serialized_content_sha256(observations),sigma_sha256=serialized_content_sha256(errors)))
        # Fixture physics cannot transport: success proves this is the real hit
        # branch, with the ordered bins and errors restored rather than sampled.
        result=mc_observations_for_field(:fixture_physics,shallow,material,site,paths,water,valid,args;cache_path=cache)
        @test result==observations
        csv=joinpath(directory,"lvd_mc_observations_$(identity[1:16]).csv")
        lines=readlines(csv)
        @test split(lines[2],',')[2]=="2"
        @test split(lines[2],',')[7]=="0.2"
        @test split(lines[3],',')[2]=="1"
        @test split(lines[3],',')[7]=="0.1"
        @test all(line->endswith(line,",true"),lines[2:end])
    end
end
