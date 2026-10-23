#!/usr/bin/env julia
# Nominal flat-column MC only: no systematic variants and no archived cache writes.
using DiffPumas, Random, Printf
using DiffPumas.Geometry: compute_flux_single
length(ARGS)==6 || error("Usage: flat_comparison_julia.jl DUMP OUTPUT.csv SAMPLES SEED smoke|full|control|deep nominal|csda")
dump,output=ARGS[1:2];n=parse(Int,ARGS[3]);seed=parse(Int,ARGS[4]);scope=ARGS[5];mode=ARGS[6];stochastic=mode=="nominal"
n>0 || error("Samples must be positive")
scope in ("smoke","full","control","deep") || error("Unknown scope")
mode in ("nominal","csda") || error("Unknown transport mode")
source_max=parse(Float64,get(ENV,"PUMAS_REFERENCE_SOURCE_MAX","1e9"))
physics=load_physics(dump);physics===nothing && error("Unreadable physics dump")
rock=get_material_index(physics,"StandardRock");air=get_material_index(physics,"Air")
contexts=stochastic ? DiffPumas.MixedTransport.prepare_mixed_kernels(physics,eachindex(physics.tables)) : nothing
points=scope=="smoke" ? [(0.,0.),(100.,45.),(800.,75.)] : scope=="control" ? [(100.,45.),(400.,60.),(800.,75.)] : scope=="deep" ? [(800.,75.)] : [(t,z) for t in 0.:50.:800. for z in (0.,15.,30.,45.,60.,75.)]
open(output,"w") do io
    println(io,"thickness_m,zenith_deg,flux,sigma_mc,samples,seed,seconds")
    for (thickness,zenith) in points
        started=time();rng=MersenneTwister(seed);span=log(1e12)
        geometry=TwoLayerGeometry{Float64}(thickness,2650.,rock,air,-1,0.,0.,1000.)
        total=0.;total2=0.
        for _ in 1:n
            energy=1e-3*exp(span*rand(rng));charge=rand(rng)>.5 ? 1. : -1.
            value=2energy*span*compute_flux_single(physics,geometry,energy,90.0 - zenith,charge;
                rng=rng,straggling=stochastic,scattering=stochastic,energy_threshold_low=100.,mixed_contexts=contexts,source_energy_max=source_max)
            total+=value;total2+=value^2
        end
        flux=total/n;sigma=sqrt(max(total2/n-flux^2,0.)/n)
        println(io,join((thickness,zenith,flux,sigma,n,seed,time()-started),','));flush(io)
        @printf("Julia thickness=%g zenith=%g flux=%g seconds=%.2f\n",thickness,zenith,flux,time()-started)
    end
end
