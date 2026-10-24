#!/usr/bin/env julia
# Separate verification of one original MC training ray; not a ninth replicate.
include(joinpath(@__DIR__,"validate_surrogate_v2.jl"))
function replay_original_calibration(argv=ARGS)
    iseven(length(argv))||error("expected --name value pairs")
    opts=Dict(k=>"" for k in ("dump","artifact","training","output"))
    for i in 1:2:length(argv)
        key=replace(argv[i],r"^--"=>"");haskey(opts,key)&&isempty(opts[key])||error("unknown/repeated option")
        opts[key]=abspath(argv[i+1])
    end
    all(!isempty,values(opts))||error("all inputs required")
    ispath(opts["output"])&&error("fresh output required")
    metadata=load_csda_corrections!(opts["artifact"])
    metadata["physics_dump_sha256"]==bytes2hex(sha256(read(opts["dump"])))||error("dump identity")
    rows=filter(r->Int(r["bin"])==1,readrows(opts["training"]));length(rows)==2||error("original first ray must have two wet contrasts")
    sort!(rows;by=r->Int(r["field"]));Int[r["field"] for r in rows]==[1,2]||error("original contrast identity")
    seed=Int(first(rows)["seed"]);all(r->Int(r["seed"])==seed,rows)||error("paired seed identity")
    seed==2027112401||error("original production seed identity")
    ph,mat,site,em,vol,shallow,_=v2state(opts["dump"])
    paths=filter(p->p.valid,precompute_paths(vol,em,Float64.(metadata["training_zeniths"]),Float64.(metadata["training_azimuths"])))
    length(paths)==12&&num_cells(vol)==128||error("original training geometry")
    path=first(paths)
    path.zenith==Float64(first(rows)["zenith"])&&path.azimuth==Float64(first(rows)["azimuth"])||error("original first ray identity")
    Float64(rows[1]["mc_dry"])==Float64(rows[2]["mc_dry"])||error("saved dry estimate mismatch")
    expected=[Float64(rows[1]["mc_dry"]),Float64(rows[1]["mc_wet"]),Float64(rows[2]["mc_wet"])]
    samples=sample_energy_set(8192,1e-3,1e9,seed);results=NamedTuple[]
    mkpath(opts["output"])
    for (k,fraction) in enumerate((0.,.2,.6))
        w=fill(fraction,num_cells(vol));mats,dens=build_cell_properties_for_mc(shallow,w,mat)
        value,se=compute_directional_flux_mc(ph,mat,site,path,mats,dens,samples,seed+1;straggling=true,scattering=false,energy_threshold_low=100.)
        equal=isequal(value,expected[k])
        push!(results,(water_fraction=fraction,original_mc=expected[k],replayed_mc=value,replayed_sigma=se,bitwise_equal=equal,energy_seed=seed,transport_seed=seed+1,samples=8192))
        csv_rows(joinpath(opts["output"],"original_ray_replay.csv"),results)
        equal||error("original MC replay mismatch; preserve and diagnose, do not resample")
    end
    open(io->TOML.print(io,Dict("all_three_mc_fluxes_bitwise_equal"=>true,"training_ray"=>1,"zenith_deg"=>path.zenith,"azimuth_deg"=>path.azimuth,"energy_seed"=>seed,"transport_seed"=>seed+1,"samples_per_field"=>8192,"scope"=>"Separate verification of original training simulation under current MC implementation. Not a ninth ensemble replicate, retuning, or validation of physical accuracy.")),joinpath(opts["output"],"replay_summary.toml"),"w")
end
if abspath(PROGRAM_FILE)==@__FILE__;replay_original_calibration();end
