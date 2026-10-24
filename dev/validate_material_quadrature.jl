#!/usr/bin/env julia
# Seven-ray deterministic protocol preserved from final_field_quadrature/run.jl.
const REPO=normpath(joinpath(@__DIR__,".."))
include(joinpath(REPO,"dev","energy_quadrature_probe.jl"))
using DelimitedFiles,Dates
function material_quadrature_options(argv)
    iseven(length(argv)) || error("expected --name value pairs")
    opts=Dict("fieldcsv"=>"","correctionartifact"=>"","output"=>"","dump"=>"")
    for i in 1:2:length(argv)
        key=replace(argv[i],r"^--"=>"");haskey(opts,key) || error("unknown option $(argv[i])")
        isempty(opts[key]) || error("duplicate option $(argv[i])")
        opts[key]=abspath(argv[i+1])
    end
    all(x->!isempty(x),values(opts)) || error("required: --fieldcsv --correctionartifact --output --dump")
    opts
end
function validate_material_quadrature(opts)
    output=opts["output"];dump=opts["dump"];fieldfile=opts["fieldcsv"];artifact=opts["correctionartifact"]
    isdir(output) && !issubset(Set(readdir(output)),Set(["execution.json","run.log","pid.json"])) && error("refuse existing evidence")
    original_base=deepcopy(get_csda_correction());original_material=deepcopy(get_csda_material_correction())
    try
    metadata=load_csda_corrections!(artifact)
    get(metadata,"physics_dump_sha256","")==bytes2hex(sha256(read(dump))) || error("artifact physics hash mismatch")
    metadata["csda_nodes"]==32 || error("artifact midpoint rule differs")
    metadata["energy_min"]==1e-3 && metadata["energy_max"]==1e9 || error("artifact energy domain differs")
    physics=load_physics(dump);physics===nothing && error("invalid physics dump")
    idx(name)=get_material_index(physics,name)
    mat=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),idx("PorousWetRock"),
        Float64(metadata["rock_density"]),Float64(physics.tables[idx("Water")].density),0.,0.)
    site=build_site(Float64(metadata["primary_altitude_m"]))
    emap=LVDTopo.build_elevation_map()
    volume=create_voxel_volume(emap,mat;half_width_km=5.,surface_step_km=1.,nz=6)
    nc=num_cells(volume);nc==600 || error("production grid must contain600cells")
    raw,head=readdlm(fieldfile,',';header=true);header=vec(String.(head))
    function col(name)
        k=findfirst(==(name),header);k===nothing && error("missing field column $name");k
    end
    size(raw,1)==nc || error("field size mismatch")
    actual=zeros(nc);allowed=falses(nc)
    for c in 1:nc
        Int(raw[c,col("cell_idx")])==c || error("field order mismatch")
        for (axis,name) in enumerate(("x_m","y_m","z_m"))
            isapprox(Float64(raw[c,col(name)]),volume.centroids[axis,c];atol=1e-8,rtol=0) || error("field coordinate mismatch")
        end
        actual[c]=Float64(raw[c,col("w")]);isfinite(actual[c]) && 0<=actual[c]<=MAX_WATER_FRACTION || error("field outside box")
        flag=string(raw[c,col("aquifer_mask")]);flag in ("true","false") || error("invalid support flag")
        allowed[c]=flag=="true"
    end
    all(iszero,actual[.!allowed]) || error("nonzero field outside support")
rules=[(nodes=n,samples=csda_energy_quadrature(n,1e-3,1e9)) for n in (32,64,128,256,512)]
zeniths=[0.,30.,45.,60.,75.,85.,86.];paths=precompute_paths(volume,emap,zeniths,[0.])
mkpath(output)
inputs=[dump,fieldfile,artifact,@__FILE__,joinpath(REPO,"dev","energy_quadrature_probe.jl"),
    joinpath(REPO,"examples","revision_suite.jl"),joinpath(REPO,"examples","lvd_tomography.jl"),
    joinpath(REPO,"Project.toml"),joinpath(REPO,"Manifest.toml")]
append!(inputs,filter(p->endswith(p,".jl"),readdir(joinpath(REPO,"src");join=true)))
inputhash=Dict(abspath(p)=>bytes2hex(sha256(read(p))) for p in inputs)
config=Dict("started_utc"=>string(now(UTC)),"julia_version"=>string(VERSION),"threads"=>Threads.nthreads(),
    "input_sha256"=>inputhash,"physics_dump"=>dump,"field"=>fieldfile,"correction_artifact"=>artifact,
    "geometry"=>"production voxel half5km step1km nz6;600 coordinates verified", "requested_zeniths_deg"=>zeniths,
    "azimuth_deg"=>0.,"primary_altitude_m"=>site.primary_altitude,"rock_density_kg_m3"=>mat.rock_density,
    "nodes"=>[r.nodes for r in rules],"reference_nodes"=>512,"fields"=>["dry","final_fit"],
    "random_sampling"=>false,"correction_metadata"=>metadata,
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
    if corrected
        load_csda_corrections!(artifact)
    else
        set_csda_correction!(enabled=false)
        set_csda_material_correction!(enabled=false)
    end
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

finally
    set_csda_correction!(;(name=>getfield(original_base,name) for name in fieldnames(typeof(original_base)))...)
    set_csda_material_correction!(enabled=original_material.enabled,linear=original_material.linear,quadratic=original_material.quadratic)
end

end
if abspath(PROGRAM_FILE)==@__FILE__
    validate_material_quadrature(material_quadrature_options(ARGS))
end
