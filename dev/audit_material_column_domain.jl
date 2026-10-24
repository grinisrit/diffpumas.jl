#!/usr/bin/env julia
# Read-only final-field composition-domain audit: geometry and density only.
include(joinpath(@__DIR__,"validate_surrogate_v2.jl"))
function audit_column_domain(argv=ARGS)
    iseven(length(argv)) || error("expected --name value pairs")
    opts=Dict("fieldcsv"=>"","raycsv"=>"","artifact"=>"","dump"=>"","output-dir"=>"")
    for i in 1:2:length(argv)
        k=replace(argv[i],r"^--"=>"");haskey(opts,k) || error("unknown option")
        isempty(opts[k]) || error("duplicate option");opts[k]=abspath(argv[i+1])
    end
    all(x->!isempty(x),values(opts)) || error("all options required")
    out=opts["output-dir"]
    isdir(out) && !issubset(Set(readdir(out)),Set(["execution.json","pid.json","run.log"])) && error("refuse existing evidence")
    mkpath(out)
    inputpaths=[@__FILE__,joinpath(@__DIR__,"validate_surrogate_v2.jl"),opts["fieldcsv"],opts["raycsv"],opts["artifact"],opts["dump"]]
    hashes=Dict(abspath(p)=>bytes2hex(sha256(read(p))) for p in inputpaths)
    metadata=load_csda_corrections!(opts["artifact"])
    metadata["physics_dump_sha256"]==bytes2hex(sha256(read(opts["dump"]))) || error("artifact dump mismatch")
    ph=load_physics(opts["dump"]);ph===nothing && error("invalid physics dump")
    idx(n)=get_material_index(ph,n)
    mat=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),idx("PorousWetRock"),Float64(metadata["rock_density"]),Float64(ph.tables[idx("Water")].density),0.,0.)
    em=LVDTopo.build_elevation_map();vol=create_voxel_volume(em,mat;half_width_km=5.,surface_step_km=1.,nz=6)
    nc=num_cells(vol);nc==600 || error("requires full 600-cell field")
    raw=readrows(opts["fieldcsv"]);length(raw)==nc || error("field length mismatch")
    w=zeros(nc);shallow=falses(nc)
    for (c,r) in enumerate(raw)
        Int(r["cell_idx"])==c || error("field order mismatch")
        for (j,k) in enumerate(("x_m","y_m","z_m"))
            isapprox(Float64(r[k]),vol.centroids[j,c];atol=1e-8,rtol=0) || error("field coordinate mismatch")
        end
        w[c]=Float64(r["w"]);isfinite(w[c]) && 0<=w[c]<=MAX_WATER_FRACTION || error("field outside box")
        string(r["aquifer_mask"]) in ("true","false") || error("invalid support flag")
        string(r["aquifer_mask"])=="true" || iszero(w[c]) || error("water outside imposed support")
    end
    qmin=Float64(metadata["training_q_min"]);qmax=Float64(metadata["training_q_max"])
    rows=NamedTuple[]
    for r in readrows(opts["raycsv"])
        theta=Float64(r["zenith_deg"]);phi=Float64(r["azimuth_deg"])
        p=trace_path(vol,theta,phi,em)
        string(p.valid)==string(r["ray_valid"]) || error("ray validity changed")
        (!p.valid && !isfinite(p.surface_distance) && !isfinite(Float64(r["slant_m"]))) ||
            isapprox(p.surface_distance,Float64(r["slant_m"]);rtol=1e-12,atol=1e-8) || error("ray geometry changed")
        q=p.valid ? DiffPumas.Tomography._column_water_fraction(mat,p.remaining_rock_distance,[s.distance for s in p.segments],[s.cell_idx for s in p.segments],[shallow[s.cell_idx] for s in p.segments],w) : NaN
        push!(rows,(bin_idx=Int(r["bin_idx"]),zenith_deg=theta,azimuth_deg=phi,ray_valid=p.valid,covered=string(r["covered"])=="true",fit_mask=Float64(r["fit_mask"]),column_water_fraction=q,
            training_q_min=qmin,training_q_max=qmax,below_training=p.valid && q<qmin-1e-12,above_training=p.valid && q>qmax+1e-12,dry_column=p.valid && q==0))
    end
    csv_rows(joinpath(out,"column_domain.csv"),rows)
    function stats(subset)
        q=[r.column_water_fraction for r in subset]
        isempty(q) && return Dict("count"=>0)
        Dict("count"=>length(q),"minimum"=>minimum(q),"median"=>median(q),"maximum"=>maximum(q),"q10"=>quantile(q,.1),"q90"=>quantile(q,.9),
            "below_training_count"=>count(r->r.below_training,subset),"above_training_count"=>count(r->r.above_training,subset),"dry_column_count"=>count(r->r.dry_column,subset))
    end
    changes=[p for (p,d) in hashes if bytes2hex(sha256(read(p)))!=d]
    summary=Dict("source_inputs_sha256"=>hashes,"changed_inputs"=>changes,"training_q_min"=>qmin,"training_q_max"=>qmax,
        "all_valid"=>stats(filter(r->r.ray_valid,rows)),"covered"=>stats(filter(r->r.covered && r.ray_valid,rows)),"fitted"=>stats(filter(r->r.covered && r.ray_valid && r.fit_mask>0,rows)),
        "scope"=>"Final recovered-field column composition versus empirical training range; extrapolation description only, no MC accuracy, water-recovery accuracy or uncertainty claim")
    open(io->TOML.print(io,summary),joinpath(out,"column_domain_summary.toml"),"w")
    isempty(changes) || error("input changed")
end
if abspath(PROGRAM_FILE)==@__FILE__;audit_column_domain();end
