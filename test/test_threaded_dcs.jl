# Standalone: JULIA_NUM_THREADS=2 julia --project=. test/test_threaded_dcs.jl
using Test
using DiffPumas
using DiffPumas.Materials: dcs_bremsstrahlung_ssr, dcs_pair_production_ssr,
    dcs_photonuclear_drss, compute_dcs_integral

@testset "Independent DCS histories share no mutable workspace" begin
    # Distinct log-integration bounds, nuclei, masses and energy transfers.
    # Compare every repetition, not just the final value of each worker.
    materials=((1.,1.008),(8.,15.999),(11.,22.),(26.,55.845),(82.,207.2))
    cases=[(z=z,a=a,m=mass,k=k,x=x) for (z,a) in materials,
        mass in (MUON_MASS,TAU_MASS), k in (10.,100.,1e4,1e6), x in (.025,.2,.75)]
    cases=vec(cases)
    functions=(dcs_bremsstrahlung_ssr,dcs_pair_production_ssr,dcs_photonuclear_drss)
    serial=[f(c.z,c.a,c.m,c.k,c.x*c.k) for c in cases,f in functions]
    @test all(isfinite,serial)
    @test all(>=(0),serial)
    @test all(j->any(>(0),serial[:,j]),axes(serial,2))
    if Threads.nthreads()<2
        @info "Run JULIA_NUM_THREADS=2 or greater to exercise concurrent DCS calls"
        @test_skip Threads.nthreads()>=2
    else
        repetitions=32
        concurrent=zeros(length(cases),length(functions),repetitions)
        Threads.@threads :static for i in 1:length(cases)*repetitions
            rep=cld(i,length(cases));cindex=mod1(i+17rep,length(cases));c=cases[cindex]
            for (j,f) in enumerate(functions)
                concurrent[cindex,j,rep]=f(c.z,c.a,c.m,c.k,c.x*c.k)
            end
        end
        # Scheduling must not change a deterministic scalar-function result.
        # Bitwise equality is appropriate: each call performs identical operations.
        @test all(rep->concurrent[:,:,rep]==serial,1:repetitions)
        # Public integrated moments also compose the raw functions concurrently.
        processes=(:bremsstrahlung,:pair_production,:photonuclear)
        selected=cases[1:13:end]
        moments=[compute_dcs_integral(p,c.z,c.a,149e-9,c.m,c.k,.05,.6,mode;npoints=24)
            for c in selected,p in processes,mode in 0:2]
        threaded=similar(moments)
        Threads.@threads for i in eachindex(selected)
            c=selected[i]
            for (j,p) in enumerate(processes),mode in 0:2
                threaded[i,j,mode+1]=compute_dcs_integral(p,c.z,c.a,149e-9,c.m,c.k,.05,.6,mode;npoints=24)
            end
        end
        @test threaded==moments
    end
end
