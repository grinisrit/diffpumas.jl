#!/usr/bin/env julia
# Research-only convergence probe; does not change production energy sampling.
# Historical positional invocation: julia --project=. dev/energy_quadrature_probe.jl NEW_DIRECTORY
# New inputs: --output NEW_DIRECTORY --dump FILE --calibration FILE --zeniths 0,30,45,60,75
include(joinpath(@__DIR__, "..", "examples", "revision_suite.jl"))

const HISTORICAL_QUADRATURE_DUMP=joinpath(@__DIR__,"..","examples","data","revision_physics","materials.pumas")
const HISTORICAL_QUADRATURE_CALIBRATION=joinpath(@__DIR__,"revision_inputs","quadrature_frozen_calibration.toml")

function quadrature_probe_options(argv)
    options=Dict("dump"=>HISTORICAL_QUADRATURE_DUMP,
        "calibration"=>HISTORICAL_QUADRATURE_CALIBRATION,"zeniths"=>"0,30,45","output"=>"")
    i=1
    while i<=length(argv)
        argument=argv[i]
        if startswith(argument,"--")
            key=argument[3:end]
            haskey(options,key) || error("Unknown option $argument")
            i<length(argv) || error("Missing value for $argument")
            key=="output" && !isempty(options["output"]) && error("Output specified twice")
            options[key]=argv[i+1];i+=2
        else
            isempty(options["output"]) || error("Output specified twice")
            options["output"]=argument;i+=1
        end
    end
    isempty(options["output"]) && error("Explicit output required: --output NEW_DIRECTORY (historical positional output is also accepted)")
    zeniths=parse.(Float64,split(options["zeniths"],','))
    !isempty(zeniths) && all(t->isfinite(t) && 0<=t<90,zeniths) && length(unique(zeniths))==length(zeniths) || error("zeniths must be unique finite angles in [0,90)")
    for key in ("dump","calibration")
        isfile(options[key]) || error("Input $key file not found: $(options[key])")
    end
    ispath(options["output"]) && error("Output already exists; select a new directory to preserve existing results")
    options,zeniths
end

function quadrature_probe_geometry(emap,mat,zeniths)
    volume=create_voxel_volume(emap,mat;half_width_km=3.0,surface_step_km=1.5,nz=8)
    paths=precompute_paths(volume,emap,zeniths,[0.0])
    coverage=[(zenith=p.zenith,azimuth=p.azimuth,valid=p.valid,
        nonempty=!isempty(p.segments),slant_m=p.surface_distance,
        in_volume_m=p.distance_in_volume,remaining_rock_m=p.remaining_rock_distance) for p in paths]
    volume,paths,coverage
end

function paired_log_midpoints(n; emin=1e-3, emax=1e9)
    span = log(emax / emin)
    # compute_directional_flux_csda multiplies each sample by two for charge
    # sampling. Equal inclusion of both charges retains that normalization.
    [EnergySample(emin * exp(span * (i - 0.5) / n), charge,
                  emin * exp(span * (i - 0.5) / n) * span)
     for i in 1:n for charge in (-1.0, 1.0)]
end

function quadrature_probe(output;dump=HISTORICAL_QUADRATURE_DUMP,
        calibration_file=HISTORICAL_QUADRATURE_CALIBRATION,zeniths=[0.0,30.0,45.0])
    ispath(output) && error("Output already exists; select a new directory")
    isfile(dump) || error("Physics dump not found: $dump")
    isfile(calibration_file) || error("Frozen calibration not found: $calibration_file")
    calibration = TOML.parsefile(calibration_file)
    mkpath(output)
    physics = load_physics(dump)
    physics === nothing && error("Pinned physics dump cannot be read")
    idx(name) = get_material_index(physics, name)
    mat = MaterialConfig(idx("StandardRock"), idx("Water"), idx("Air"), idx("PorousWetRock"),
                         2710.0, physics.tables[idx("Water")].density, 0.0, 0.0)
    site = build_site(30000.0)
    emap = LVDTopo.build_elevation_map()
    volume,paths,coverage=quadrature_probe_geometry(emap,mat,zeniths)
    csv_rows(joinpath(output,"path_coverage.csv"),coverage)
    nc = num_cells(volume)
    shallow = falses(nc)
    rules = [(name="seeded8_seed42", samples=sample_energy_set(8, 1e-3, 1e9, 42)),
             (name="seeded32_seed20261022", samples=sample_energy_set(32, 1e-3, 1e9, 20261022))]
    for n in (16, 32, 64, 128, 256)
        push!(rules, (name="paired_midpoint$n", samples=paired_log_midpoints(n)))
    end
    config = Dict("julia_version" => string(VERSION), "threads" => Threads.nthreads(),
        "physics_path"=>abspath(dump), "calibration_path"=>abspath(calibration_file),
        "probe_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "tomography_source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"..","src","Tomography.jl")))),
        "requested_zeniths_deg"=>zeniths,"requested_paths"=>length(coverage),
        "valid_nonempty_paths"=>count(p->p.valid && !isempty(p.segments),paths),
        "physics_sha256" => bytes2hex(sha256(read(dump))), "rock_density" => 2710.0,
        "calibration_sha256" => bytes2hex(sha256(read(calibration_file))),
        "derivative_method" => "h=1e-4 central at .3; second-order right difference at 0",
        "geometry" => "voxel nz8 half-width3km surface-step1.5km, azimuth0deg",
        "cell_selection" => "longest intersected segment per ray",
        "reference_scope" => "paired midpoint256 is a convergence comparator, not exact truth")
    open(joinpath(output, "config.toml"), "w") do io; TOML.print(io, config); end
    all(p -> p.valid && !isempty(p.segments), paths) || error("Requested probe path invalid; inspect path_coverage.csv; no convergence measurements produced")
    rows = NamedTuple[]
    for corrected in (false, true)
        if corrected
            set_csda_correction!(; (Symbol(k) => v for (k, v) in calibration)...)
        else
            set_csda_correction!(enabled=false)
        end
        for path in paths, fraction in (0.0, 0.3)
            cell = path.segments[argmax([s.distance for s in path.segments])].cell_idx
            for rule in rules
                function value(x)
                    w = fill(fraction, nc)
                    w[cell] = x
                    compute_directional_flux_csda(physics, shallow, mat, site, path, w, rule.samples)
                end
                flux = value(fraction)
                h = 1e-4
                gradient = fraction == 0.0 ? (-3flux + 4value(h) - value(2h)) / (2h) :
                    (value(fraction + h) - value(fraction - h)) / (2h)
                half_gradient = if rule.name == "paired_midpoint256"
                    hh = h / 2
                    fraction == 0.0 ? (-3flux + 4value(hh) - value(2hh)) / (2hh) :
                        (value(fraction + hh) - value(fraction - hh)) / (2hh)
                else
                    NaN
                end
                push!(rows, (corrected=corrected, zenith=path.zenith, azimuth=path.azimuth,
                    slant_m=path.surface_distance, cell=cell,
                    cell_distance_m=path.segments[argmax([s.distance for s in path.segments])].distance,
                    fraction=fraction, rule=rule.name, particles=length(rule.samples),
                    flux=flux, gradient=gradient, reference_gradient_half_step=half_gradient))
                csv_rows(joinpath(output, "measurements.csv"), rows)
                println(last(rows)); flush(stdout)
            end
        end
    end
    summaries = NamedTuple[]
    for row in rows
        reference = only(filter(r -> r.corrected == row.corrected && r.zenith == row.zenith &&
            r.fraction == row.fraction && r.rule == "paired_midpoint256", rows))
        push!(summaries, merge(row, (relative_flux_error=row.flux/reference.flux-1,
            relative_gradient_error=row.gradient/reference.gradient-1)))
    end
    csv_rows(joinpath(output, "comparison.csv"), summaries)
end

if abspath(PROGRAM_FILE) == @__FILE__
    options,zeniths=quadrature_probe_options(ARGS)
    quadrature_probe(abspath(options["output"]);dump=options["dump"],
        calibration_file=options["calibration"],zeniths=zeniths)
end
