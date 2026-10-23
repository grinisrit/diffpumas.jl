# External deterministic control; no production or tracked source is modified.
const REPO="/home/roland/devspace/diffpumas.jl"
include(joinpath(REPO,"dev","energy_quadrature_probe.jl"))
using DelimitedFiles, Dates
length(ARGS)==1 || error("Usage: external run.jl NEW_OUTPUT")
function run_final_field_probe(output::String)
ispath(output) && error("Refuse existing output")
main_dir=joinpath(REPO,"examples","data","lvd_revision_corrected_20261023")
dump=joinpath(REPO,"examples","data","revision_physics_v2_final_20261023","materials.pumas")
fieldfile=joinpath(main_dir,"lvd_fit_field.csv");statusfile=joinpath(main_dir,"lvd_recursion_status.txt")
calfile=joinpath(main_dir,"lvd_mc_calibration_de8834f1c7261ba2.toml")
status=Dict{String,String}()
for line in eachline(statusfile)
    occursin('=',line) || continue
    key,value=split(line,'=';limit=2);status[key]=value
end
status["converged"]=="true" && status["passes"]=="5" || error("Final-fit status differs from approved checkpoint")
status["seed"]=="42" && status["csda_quadrature"]=="midpoint" && status["csda_energy_samples"]=="32" || error("Production configuration differs")
cal=TOML.parsefile(calfile);cal["calibration_pass"]==5 || error("Not final-pass calibration")
correction=cal["correction"]
physics=load_physics(dump);physics===nothing && error("Pinned dump invalid")
idx(name)=get_material_index(physics,name)
mat=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),idx("PorousWetRock"),
    parse(Float64,status["rock_density_kg_m3"]),Float64(physics.tables[idx("Water")].density),0.,0.)
site=build_site(30000.)
emap=LVDTopo.build_elevation_map()
volume=create_voxel_volume(emap,mat;half_width_km=5.,surface_step_km=1.,nz=6)
nc=num_cells(volume);nc==600 || error("Production grid is not600cells")
raw,head=readdlm(fieldfile,',';header=true);header=vec(String.(head));col(name)=findfirst(==(name),header)
size(raw,1)==nc || error("Field size differs")
actual=zeros(nc);allowed=falses(nc)
for c in 1:nc
    Int(raw[c,col("cell_idx")])==c || error("Field cell order differs")
    for (axis,name) in enumerate(("x_m","y_m","z_m"))
        isapprox(Float64(raw[c,col(name)]),volume.centroids[axis,c];atol=1e-8,rtol=0) || error("Field coordinate mismatch")
    end
    actual[c]=Float64(raw[c,col("w")]);0<=actual[c]<=MAX_WATER_FRACTION || error("Field outside box")
    flag=string(raw[c,col("aquifer_mask")]);flag in ("true","false") || error("Invalid support flag")
    allowed[c]=flag=="true"
end
all(iszero,actual[.!allowed]) || error("Nonzero field outside allowed support")
ref=Float64.(cal["reference_water_fractions"])
length(ref)==nc && maximum(abs.(ref.-actual))<1e-13 || error("Final calibration reference does not match final field")
rules=[(nodes=n,samples=csda_energy_quadrature(n,1e-3,1e9)) for n in (32,64,128,256,512)]
serialized_content_sha256(first(rules).samples)==cal["csda_energy_samples_sha256"] || error("32-node rule differs from recorded production quadrature")
zeniths=[0.,30.,45.,60.,75.,85.,86.];paths=precompute_paths(volume,emap,zeniths,[0.])
mkpath(output)
inputs=[dump,fieldfile,statusfile,calfile,joinpath(main_dir,"lvd_recursion_history.csv"),@__FILE__,
    joinpath(REPO,"dev","energy_quadrature_probe.jl"),joinpath(REPO,"examples","revision_suite.jl")]
inputhash=Dict(abspath(p)=>bytes2hex(sha256(read(p))) for p in inputs)
config=Dict("started_utc"=>string(now(UTC)),"julia_version"=>string(VERSION),"threads"=>Threads.nthreads(),
    "input_sha256"=>inputhash,"physics_dump"=>dump,"field"=>fieldfile,"calibration"=>calfile,
    "geometry"=>"production voxel half5km step1km nz6;600 coordinates verified", "requested_zeniths_deg"=>zeniths,
    "azimuth_deg"=>0.,"primary_altitude_m"=>site.primary_altitude,"rock_density_kg_m3"=>mat.rock_density,
    "nodes"=>[r.nodes for r in rules],"reference_nodes"=>512,"fields"=>["dry","final_fit"],
    "random_sampling"=>false,"production_seed_record"=>42,"correction"=>correction,
    "cell_selection"=>"longest crossed cell in the recorded allowed support; no allowed crossing =>flux only",
    "reference_scope"=>"512-midpoint convergence comparator, not exact physical truth; fixed source convention; no MC errors",
    "derivative_scope"=>"local allowed-cell finite difference with other fields fixed; valid central or second-order one-sided stencil")
open(joinpath(output,"config.toml"),"w") do io;TOML.print(io,config);end
coverage=NamedTuple[]
for p in paths
    eligible=[s for s in p.segments if allowed[s.cell_idx]]
    selected=isempty(eligible) ? 0 : eligible[argmax([s.distance for s in eligible])].cell_idx
    push!(coverage,(zenith=p.zenith,azimuth=p.azimuth,valid=p.valid,slant_m=p.surface_distance,
        surface_exit_z_m=p.surface_exit_z,distance_in_volume_m=p.distance_in_volume,
        remaining_dry_rock_m=p.remaining_rock_distance,allowed_segments=length(eligible),cell=selected,
        derivative_status=selected==0 ? "no_free_parameter" : "allowed_support_cell"))
end
csv_rows(joinpath(output,"path_coverage.csv"),coverage)
all(p->p.valid && !isempty(p.segments),paths) || error("Invalid requested ray; no measurements produced")
rows=NamedTuple[];shallow=falses(nc);h=1e-4
for corrected in (false,true)
    corrected ? set_csda_correction!(;(Symbol(k)=>v for (k,v) in correction)...) : set_csda_correction!(enabled=false)
    for (j,path) in enumerate(paths), (fieldname,field) in (("dry",zeros(nc)),("final_fit",actual))
        c=coverage[j].cell
        for rule in rules
            w=copy(field)
            flux=compute_directional_flux_csda(physics,shallow,mat,site,path,w,rule.samples)
            gradient=NaN;half_gradient=NaN;method="none";lower=NaN;upper=NaN
            function derivative(step)
                w0=field[c]
                function value(x)
                    w[c]=x
                    compute_directional_flux_csda(physics,shallow,mat,site,path,w,rule.samples)
                end
                if w0-step>=0 && w0+step<=MAX_WATER_FRACTION
                    return (value(w0+step)-value(w0-step))/(2step),"central",w0-step,w0+step
                elseif w0+2step<=MAX_WATER_FRACTION
                    return (-3flux+4value(w0+step)-value(w0+2step))/(2step),"forward_second_order",w0,w0+2step
                elseif w0-2step>=0
                    return (3flux-4value(w0-step)+value(w0-2step))/(2step),"backward_second_order",w0-2step,w0
                end
                error("No valid derivative stencil")
            end
            if c!=0
                gradient,method,lower,upper=derivative(h)
                if rule.nodes==512
                    half_gradient,_,_,_=derivative(h/2)
                end
            end
            push!(rows,(corrected=corrected,zenith=path.zenith,azimuth=path.azimuth,field=fieldname,nodes=rule.nodes,
                flux=flux,cell=c,water_at_cell=c==0 ? NaN : field[c],derivative_status=coverage[j].derivative_status,
                fd_method=method,fd_step=h,fd_lower=lower,fd_upper=upper,gradient=gradient,reference_gradient_half_step=half_gradient))
            csv_rows(joinpath(output,"measurements.csv"),rows)
            println(last(rows));flush(stdout)
        end
    end
end
comparison=NamedTuple[]
for row in rows
    reference=only(filter(r->r.corrected==row.corrected && r.zenith==row.zenith && r.field==row.field && r.nodes==512,rows))
    fdefined=isfinite(reference.flux) && reference.flux!=0
    gdefined=isfinite(reference.gradient) && reference.gradient!=0
    push!(comparison,merge(row,(absolute_flux_difference=row.flux-reference.flux,
        relative_flux_defined=fdefined,relative_flux_error=fdefined ? row.flux/reference.flux-1 : NaN,
        absolute_gradient_difference=row.gradient-reference.gradient,
        relative_gradient_defined=gdefined,relative_gradient_error=gdefined ? row.gradient/reference.gradient-1 : NaN)))
end
csv_rows(joinpath(output,"comparison.csv"),comparison)
changes=[p for (p,digest) in inputhash if bytes2hex(sha256(read(p)))!=digest]
open(joinpath(output,"status.toml"),"w") do io
    TOML.print(io,Dict("finished_utc"=>string(now(UTC)),"source_unchanged"=>isempty(changes),
        "changed_inputs"=>changes,"rows"=>length(rows),"valid_paths"=>count(p->p.valid,paths),
        "scope"=>"Deterministic convergence only;512 nodes not physical truth; no free parameter =>no local fit-gradient claim"))
end
isempty(changes) || error("Input/source changed")

end
run_final_field_probe(abspath(ARGS[1]))
