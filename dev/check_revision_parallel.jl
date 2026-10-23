#!/usr/bin/env julia
# Execute under1/4threads and compare parity.csv bytes. No AD/global mutations
# occur during independent replicate evaluation.
include(joinpath(@__DIR__,"..","examples","revision_suite.jl"))
length(ARGS)==2 || error("Usage: check_revision_parallel.jl DUMP NEW_OUTPUT_DIRECTORY")
ispath(ARGS[2]) && error("Output exists")
physics=load_tomography_physics(ARGS[1])
idx(name)=get_material_index(physics,name)
mat=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),idx("PorousWetRock"),2710.,physics.tables[idx("Water")].density,0.,0.)
site=SiteConfig(1000.,2000.)
path=DirectionalPath(0.,0.,[CellSegment(1,200.),CellSegment(2,200.)],400.,400.,0.,400.,true)
shallow=falses(2)
set_csda_correction!(enabled=false)
rows=revision_replicates(4;parallel=true) do replicate
    seed=20261023+100replicate
    energies=sample_energy_set(16,100.,1e5,seed)
    fields=([.275,.3],[.325,.3],zeros(2),fill(.3,2))
    measurements=map(fields) do field
        materials,densities=build_cell_properties_for_mc(shallow,field,mat)
        compute_directional_flux_mc(physics,mat,site,path,materials,densities,energies,seed+1;
            straggling=true,scattering=false,energy_threshold_low=100.)
    end
    low,high,dry,wet=measurements
    secant=(compute_directional_flux_csda(physics,shallow,mat,site,path,fields[2],energies)-
        compute_directional_flux_csda(physics,shallow,mat,site,path,fields[1],energies))/.05
    (replicate=replicate,seed=seed,low=low[1],high=high[1],dry=dry[1],wet=wet[1],
        sigma_low=low[2],sigma_high=high[2],sigma_dry=dry[2],sigma_wet=wet[2],csda_secant=secant)
end
@assert any(row->row.low>0 && row.dry>0,rows)
mkpath(ARGS[2]);csv_rows(joinpath(ARGS[2],"parity.csv"),rows)
open(joinpath(ARGS[2],"config.toml"),"w") do io
    TOML.print(io,Dict("threads"=>Threads.nthreads(),"dump_sha256"=>bytes2hex(sha256(read(ARGS[1]))),
        "suite_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"..","examples","revision_suite.jl"))))))
end
println("Real fixed-path replicate fixture complete on $(Threads.nthreads()) threads")
