include(joinpath(@__DIR__,"prototype.jl"))
include("/home/roland/devspace/diffpumas.jl/examples/lvd_tomography.jl")
using DelimitedFiles, TOML, SHA
empty!(ARGS);include(joinpath(@__DIR__,"saved_args.jl"));args=parse_commandline()
ph=load_tomography_physics(args.dump_path);idx(n)=get_material_index(ph,n)
porous=idx("PorousWetRock")
mat=T.MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),porous,2710.,Float64(ph.tables[idx("Water")].density),porous>0 ? Float64(ph.tables[porous].density) : 0.,args.porous_top_thickness)
load_csda_corrections!(args.correction_artifact)
emap=LVDTopo.build_elevation_map();volume=create_sensitivity_volume(emap,mat,args)
shallow=Bool[is_shallow_cell(volume,c) for c in 1:num_cells(volume)]
site=build_site(args.primary_altitude_km*1000)
bins=load_measured_intensity(args.match_data);b=bins[916]
zen=.5*(b.zen_lo+b.zen_hi);az=.5*(b.az_lo+b.az_hi)
path=trace_path(volume,zen,az,emap)
fieldfile="/home/roland/devspace/diffpumas.jl/examples/data/stability_v2_dry_start015_20261024/results/lvd_fit_field.csv"
a,h=readdlm(fieldfile,',',Any;header=true);w=Float64.(a[:,findfirst(==("w"),vec(h))])
es=csda_energy_quadrature(args.n_samples,args.energy_min,args.energy_max)
f0,ids,j0=directional_flux_and_grad_csda(ph,shallow,mat,site,path,w,es)
pos=findfirst(==(275),ids);default=j0[pos]
full(x)=compute_directional_flux_csda(ph,shallow,mat,site,path,x,es)
fd=Dict{String,Float64}()
for step in (1e-5,1e-6,1e-7,1e-8)
    hi=copy(w);lo=copy(w);hi[275]+=step;lo[275]-=step
    fd[string(step)]=(full(hi)-full(lo))/(2step)
end
include(joinpath(@__DIR__,"rules.jl"))
f1,ids1,j1=directional_flux_and_grad_csda(ph,shallow,mat,site,path,w,es)
new=j1[findfirst(==(275),ids1)]
@test f1==f0
@test new≈fd["1.0e-7"] rtol=2e-6
@test new≈fd["1.0e-8"] rtol=2e-6
# Valid zero-weight energy cotangent and incoming-weight tangent remain nonzero.
output,pullback=CRC.rrule(T.propagate_csda_cell_segment,ph,.2,false,mat,30.,123.4567,0.)
tangent=pullback((1.,1.,CRC.NoTangent()))
@test tangent[7]!=0.
@test tangent[8]>0.
result=Dict("scope"=>"unapplied process-local dual segment prototype", "bin"=>916,"cell"=>275,
    "water_fraction"=>w[275],"flux_before"=>f0,"flux_after"=>f1,"default_adjoint"=>default,
    "dual_adjoint"=>new,"whole_flux_fd"=>fd,"field_sha256"=>bytes2hex(sha256(read(fieldfile))),
    "physics_sha256"=>bytes2hex(sha256(read(args.dump_path))),
    "prototype_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"prototype.jl")))),
    "rules_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"rules.jl")))))
open(io->TOML.print(io,result),joinpath(@__DIR__,"adversarial_result.toml"),"w")
println(result)
