#!/usr/bin/env julia
# Independently reconstruct the frozen validation Jacobian and audit the sampled strata.
include(joinpath(@__DIR__,"..","examples","revision_suite.jl"))
length(ARGS)==1 || error("Usage: julia --project=. dev/audit_revision_selection.jl RESULT_DIRECTORY")
out=abspath(ARGS[1]); cfg=TOML.parsefile(joinpath(out,"config.toml"))
physics=load_physics(cfg["physics_dump_path"])
index(name)=get_material_index(physics,name)
mat=MaterialConfig(index("StandardRock"),index("Water"),index("Air"),index("PorousWetRock"),
    parse(Float64,cfg["rock-density"]),Float64(physics.tables[index("Water")].density),0.0,0.0)
coeff=TOML.parsefile(joinpath(out,"calibration.toml"))
set_csda_correction!(;(Symbol(k)=>v for (k,v) in coeff)...)
emap=LVDTopo.build_elevation_map()
volume=create_voxel_volume(emap,mat;half_width_km=3.0,surface_step_km=1.5,nz=8)
step=parse(Float64,cfg["rays-step"])
paths=filter(p->p.valid,precompute_paths(volume,emap,exclusive_degree_grid(60.0,step),exclusive_degree_grid(360.0,step)))
water=fill(0.3,num_cells(volume));shallow=falses(length(water))
es=csda_energy_quadrature(parse(Int,cfg["samples"]),1e-3,1e9)
flux,J=assemble_forward_and_jacobian(physics,shallow,mat,build_site(30000.0),paths,water,es;n_cells=length(water),threaded=false)
observed=Dict{Tuple{Int,Int},Int}()
header=String[]
for (n,line) in enumerate(eachline(joinpath(out,"fd_steps.csv")))
    fields=split(line,',')
    if n==1;global header=fields;continue;end
    row=Dict(zip(header,fields))
    observed[(parse(Int,row["bin"]),parse(Int,row["cell"]))]=parse(Int,row["case"])
end
expected=stratified_pairs(paths,volume,J,parse(Int,cfg["cases"]),MersenneTwister(parse(Int,cfg["seed"])))
@assert Set(keys(observed))==Set((p.bin,p.cell) for p in expected)
bi,ci,vals=findnz(J)
keep=findall(k->isfinite(vals[k]) && vals[k]!=0,eachindex(vals))
features=([paths[bi[k]].zenith for k in keep],[paths[bi[k]].surface_distance for k in keep],
    [volume.centroids[3,ci[k]] for k in keep],abs.(vals[keep]))
edges=[quantile(x,[0.25,0.5,0.75]) for x in features]
rows=NamedTuple[]; selected_strata=Set{NTuple{4,Int}}();population_strata=Set{NTuple{4,Int}}()
for (n,k) in enumerate(keep)
    quartiles=ntuple(j->1+count(<(features[j][n]),edges[j]),4)
    case=get(observed,(bi[k],ci[k]),0)
    push!(population_strata,quartiles)
    case>0 && push!(selected_strata,quartiles)
    push!(rows,(bin=bi[k],cell=ci[k],selected_case=case,zenith_deg=features[1][n],slant_m=features[2][n],cell_z_m=features[3][n],gradient=vals[k],
        zenith_quartile=quartiles[1],slant_quartile=quartiles[2],height_quartile=quartiles[3],sensitivity_quartile=quartiles[4]))
end
csv_rows(joinpath(out,"selection_population_audit.csv"),rows)
open(joinpath(out,"selection_audit.toml"),"w") do io
    TOML.print(io,Dict("selected_pairs"=>length(observed),"selected_joint_strata"=>length(selected_strata),
        "population_joint_strata"=>length(population_strata),"population_nonzero_entries"=>length(keep),
        "exact_seeded_selection_reproduced"=>true,"quartile_edges_zenith_deg"=>edges[1],
        "quartile_edges_slant_m"=>edges[2],"quartile_edges_cell_height_m"=>edges[3],"quartile_edges_abs_sensitivity"=>edges[4],
        "scope"=>"reconstructed frozen calibrated midpoint32 Jacobian; excludes zero sensitivities and near-horizontal measured directions"))
end
println("Selection audit passed: $(length(observed)) pairs from $(length(selected_strata)) joint strata of $(length(population_strata)) populated strata")
