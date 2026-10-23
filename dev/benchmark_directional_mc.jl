#!/usr/bin/env julia
# Small, recorded cost/variance diagnostic; no calibration or scientific coverage claim.
include(joinpath(@__DIR__,"..","examples","revision_suite.jl"))
function benchmark_directional_mc(argv=ARGS)
    options=Dict("dump"=>"","output"=>"","samples"=>"256,512,1024","replicates"=>"4","seed"=>"20261023")
    iseven(length(argv)) || error("Options require --name value pairs")
    for i in 1:2:length(argv)
        key=replace(argv[i],r"^--"=>"");haskey(options,key) || error("Unknown option $(argv[i])")
        options[key]=argv[i+1]
    end
    isempty(options["dump"]) && error("--dump required")
    isempty(options["output"]) && error("--output NEW_DIRECTORY required")
    ispath(options["output"]) && error("Output exists; select a new directory")
    counts=parse.(Int,split(options["samples"],','));all(>(0),counts) || error("samples must be positive")
    replicates=parse(Int,options["replicates"]);replicates>1 || error("At least two replicates required")
    seed=parse(Int,options["seed"])
    physics=load_tomography_physics(options["dump"])
    idx(name)=get_material_index(physics,name)
    material=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),idx("PorousWetRock"),2710.,physics.tables[idx("Water")].density,0.,0.)
    volume,paths,coverage=quadrature_timing_geometry(material)
    all(p->p.valid && !isempty(p.segments),paths) || error("Invalid benchmark paths")
    output=abspath(options["output"]);mkpath(output)
    csv_rows(joinpath(output,"coverage.csv"),coverage)
    sourcefiles=vcat(inverse_mc_source_paths(),[@__FILE__])
    hashes=Dict(abspath(p)=>bytes2hex(sha256(read(p))) for p in sourcefiles)
    config=merge(options,Dict("started_utc"=>string(now(UTC)),"julia_version"=>string(VERSION),
        "source_sha256"=>hashes,"physics_sha256"=>bytes2hex(sha256(read(options["dump"]))),
        "cpu_affinity"=>readchomp(`taskset -pc $(getpid())`),"warm_samples_per_direction_field"=>128,
        "scope"=>"Three azimuth-zero directions and two uniform fields in600-cell production geometry; sampling precision and wall time diagnostics only; not angular-domain or uncertainty coverage validation",
        "transport"=>"fixed-path stochastic energy loss, scattering=false,straggling=true,threshold100GeV",
        "energy_min_GeV"=>.001,"energy_max_GeV"=>1e9))
    open(joinpath(output,"config.toml"),"w") do io;TOML.print(io,config);end
    rows=NamedTuple[];site=build_site(30000.0);shallow=falses(num_cells(volume))
    for fraction in (0.,.3), (b,path) in enumerate(paths)
        materials,densities=build_cell_properties_for_mc(shallow,fill(fraction,num_cells(volume)),material)
        warm=sample_energy_set(128,.001,1e9,seed+100b)
        compute_directional_flux_mc(physics,material,site,path,materials,densities,warm,seed+200b;
            straggling=true,scattering=false,energy_threshold_low=100.)
        for n in counts,replicate in 1:replicates
            energyseed=seed+100000b+1000n+replicate
            energies=sample_energy_set(n,.001,1e9,energyseed)
            measurement=@timed compute_directional_flux_mc(physics,material,site,path,materials,densities,energies,energyseed+1;
                straggling=true,scattering=false,energy_threshold_low=100.)
            flux,sigma=measurement.value
            push!(rows,(zenith=path.zenith,azimuth=path.azimuth,slant_m=path.surface_distance,
                fraction=fraction,samples=n,replicate=replicate,energy_seed=energyseed,transport_seed=energyseed+1,
                flux=flux,sigma_mc=sigma,relative_sigma=flux>0 ? sigma/flux : NaN,
                seconds=measurement.time,allocated_bytes=measurement.bytes,
                seconds_per_history=measurement.time/n,process_peak_rss_bytes=Sys.maxrss()))
            csv_rows(joinpath(output,"measurements.csv"),rows)
            println(last(rows));flush(stdout)
        end
    end
    unchanged=all(p->bytes2hex(sha256(read(p)))==hashes[abspath(p)],sourcefiles)
    open(joinpath(output,"status.toml"),"w") do io
        TOML.print(io,Dict("completed_utc"=>string(now(UTC)),"source_unchanged"=>unchanged,"rows"=>length(rows)))
    end
    unchanged || error("Source changed during benchmark; timings mix source provenance")
end
function quadrature_timing_geometry(material)
    emap=LVDTopo.build_elevation_map()
    volume=create_voxel_volume(emap,material;half_width_km=5.,surface_step_km=1.,nz=6)
    paths=precompute_paths(volume,emap,[0.,45.,75.],[0.])
    coverage=[(zenith=p.zenith,azimuth=p.azimuth,valid=p.valid,segments=length(p.segments),
        slant_m=p.surface_distance,in_volume_m=p.distance_in_volume,remaining_rock_m=p.remaining_rock_distance) for p in paths]
    volume,paths,coverage
end
if abspath(PROGRAM_FILE)==@__FILE__
    benchmark_directional_mc()
end
