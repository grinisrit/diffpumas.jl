#!/usr/bin/env julia
#=
Same-lens acceptance controls, with explicit integrated exposure E.

Run with one TOML configuration argument. `base_args` must be the audited
detection-stage arguments (starting at --dump), including the frozen material
correction. `output` must be a new directory. This script includes the frozen
example but never invokes or changes its main function.
=#
using TOML, DelimitedFiles
length(ARGS) == 1 || error("Expected one config.toml")
const ACCEPTANCE_CONFIG_PATH = abspath(ARGS[1])
const ACCEPTANCE_CONFIG = TOML.parsefile(ACCEPTANCE_CONFIG_PATH)
empty!(ARGS); append!(ARGS, ACCEPTANCE_CONFIG["base_args"])
include(joinpath(@__DIR__, "..", "examples", "lvd_tomography.jl"))

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows; println(io, join(row, ',')); end
    end
end

function angular_probabilities(paths)
    thetas = sort(unique(p.zenith for p in paths))
    phis = sort(unique(p.azimuth for p in paths))
    length(thetas) == 30 && length(phis) == 90 || error("Require headline 2700-ray grid")
    thetas == collect(0.0:2.0:58.0) || error("Unexpected zenith grid")
    phis == collect(0.0:4.0:356.0) || error("Unexpected azimuth grid")
    edges = vcat(0.0, [(thetas[i]+thetas[i+1])/2 for i in 1:length(thetas)-1], 60.0)
    index = Dict(t => i for (i,t) in enumerate(thetas))
    omega = [deg2rad(4.0)*(cosd(edges[index[p.zenith]])-cosd(edges[index[p.zenith]+1])) for p in paths]
    isapprox(sum(omega), pi; rtol=1e-13) || error("Solid-angle domain mismatch")
    normalize(x) = x ./ sum(x)
    profiles = ["legacy_equal_bin" => fill(1/length(paths),length(paths)),
        "uniform_directional_area" => normalize(omega),
        "cosine" => normalize(omega .* [cosd(p.zenith) for p in paths]),
        "azimuth_modulated" => normalize(omega .* [cosd(p.zenith)*(1+0.5*cosd(2*p.azimuth)) for p in paths])]
    all(all(>(0),p) && isapprox(sum(p),1;atol=1e-14) for (_,p) in profiles) || error("Invalid exposure shares")
    return omega, profiles
end

function exposure_search(rate, reference)
    # Check both endpoints; expand rather than silently returning a search floor.
    lo, hi = reference*1e-5, reference*1e5
    for _ in 1:20
        rate(lo) < .95 && break
        lo /= 10
    end
    for _ in 1:20
        rate(hi) >= .95 && break
        hi *= 10
    end
    flo, fhi = rate(lo), rate(hi)
    if !(flo < .95 <= fhi)
        return (exposure=Inf, lower=lo, upper=hi, rate_lower=flo, rate_upper=fhi, bracketed=false)
    end
    for _ in 1:64
        mid=sqrt(lo*hi)
        rate(mid) >= .95 ? (hi=mid) : (lo=mid)
    end
    return (exposure=hi, lower=lo, upper=hi, rate_lower=rate(lo), rate_upper=rate(hi), bracketed=true)
end

function acceptance_main()
    cfg = ACCEPTANCE_CONFIG
    bytes2hex(sha256(read(cfg["detection_execution"])))==cfg["detection_execution_sha256"] || error("Headline execution record changed")
    bytes2hex(sha256(read(cfg["headline_report"])))==cfg["headline_report_sha256"] || error("Headline report changed")
    out = abspath(cfg["output"])
    ispath(out) && error("Refuse existing output directory")
    mkpath(out)
    cp(ACCEPTANCE_CONFIG_PATH,joinpath(out,"config.toml"))
    open(joinpath(out,"experiment_scope.toml"),"w") do io
        TOML.print(io,Dict(
            "exposure"=>"E=sum tau_m; tau_m=E*p_m; sum p_m=1. Equal-bin legacy conversion E=N*tau_old. Fixed total integrated exposure, not fixed expected event count.",
            "acceptance"=>"Four predeclared assumed profiles, not measured or calibrated LVD acceptance. Solid angle enters the three geometric profiles before normalization.",
            "hypotheses"=>"All detectors and Poisson pilot use the same dry-Jacobian first-order lens signal. Finite nonlinear lens flux is an exported validity audit only, not the pilot signal.",
            "gaussian"=>"Matched/ROI local baseline-variance planning formulas; SART fixed-reference mean-only sqrt(exposure) extrapolation. Shared Gaussian standard-normal streams across profiles are paired common random numbers.",
            "poisson"=>"Fresh independent calibration-null, test-null and signal count populations at each predeclared exposure. All three statistics evaluated on identical counts. Strict statistic>calibration threshold.",
            "intervals"=>"Exact 95% Clopper-Pearson Monte Carlo intervals conditional on the realized calibration threshold; calibration-threshold uncertainty is not propagated. 3000 null draws give about four expected 3sigma-tail events; no certified 3sigma FPR or finite-contrast transport power."))
    end
    args = parse_commandline()
    isempty(args.correction_artifact) && error("Require a frozen correction artifact")
    args.geometry == "grid" && args.csda_quadrature == "midpoint" && args.n_samples == 32 || error("Headline geometry/quadrature required")
    args.lens_w == .9 || error("Same headline lens w=.9 required")
    ph = load_tomography_physics(args.dump_path)
    rock=get_material_index(ph,"StandardRock"); water=get_material_index(ph,"Water")
    air=get_material_index(ph,"Air"); porous=get_material_index(ph,"PorousWetRock")
    mat=MaterialConfig(rock,water,air,porous,args.rock_density,
        Float64(ph.tables[water].density),porous>0 ? Float64(ph.tables[porous].density) : 0.,args.porous_top_thickness)
    args.porous_top_thickness == 0 && args.rock_density == 2710 || error("Unexpected material settings")
    emap=LVDTopo.build_elevation_map(); volume=create_sensitivity_volume(emap,mat,args)
    num_cells(volume)==600 || error("Require headline 600-cell geometry")
    site=build_site(args.primary_altitude_km*1000)
    paths=precompute_paths(volume,emap,exclusive_degree_grid(args.zenith_max_deg,args.zenith_step_deg),exclusive_degree_grid(360.,args.azimuth_step_deg))
    shallow=Bool[is_shallow_cell(volume,i) for i in 1:num_cells(volume)]
    energies=csda_energy_quadrature(args.n_samples,args.energy_min,args.energy_max)
    artifact=TOML.parsefile(args.correction_artifact)
    metadata=get(artifact,"metadata",Dict())
    get(metadata,"physics_dump_sha256","")==bytes2hex(sha256(read(args.dump_path))) || error("Dump/artifact mismatch")
    for (key,expected) in (("primary_altitude_m",args.primary_altitude_km*1000),
            ("rock_density",mat.rock_density),("csda_nodes",args.n_samples),
            ("energy_min",args.energy_min),("energy_max",args.energy_max))
        get(metadata,key,nothing)==expected || error("Artifact metadata mismatch: $key")
    end
    load_csda_corrections!(args.correction_artifact)
    f,J=assemble_forward_and_jacobian(ph,shallow,mat,site,paths,zeros(600),energies;n_cells=600,threaded=args.threaded)
    length(f)==2700 && all(isfinite,f) && all(>(0),f) || error("Require all 2700 finite positive headline bins")
    all(isfinite,nonzeros(J)) || error("Nonfinite Jacobian")
    w=zeros(600)
    for i in eachindex(w)
        x,y,z=cell_centroid(volume,i)
        abs(z-1118.6)<=160 && abs(x)<=600 && abs(y)<=600 && (w[i]=.9)
    end
    lens=findall(>(0),w); length(lens)==4 || error("Headline lens changed")
    s=J*w; roi=abs.(s).>=.1*maximum(abs.(s))
    count(roi)==cfg["headline_roi_bins"] || error("Headline ROI changed")
    all(>=(0),f+s) || error("First-order signal gives negative Poisson means")
    # Save finite-lens forward predictions as a validity audit, not silently
    # substitute a different template into only one of the compared detectors.
    finite_flux=zeros(length(paths))
    Threads.@threads for i in eachindex(paths)
        finite_flux[i]=compute_directional_flux_csda(ph,shallow,mat,site,paths[i],w,energies)
    end
    all(isfinite,finite_flux) && all(>(0),finite_flux) || error("Nonfinite nonlinear audit flux")
    omega,profiles=angular_probabilities(paths)
    write_csv(joinpath(out,"directional_templates.csv"),["bin","theta_deg","phi_deg","omega_sr","dry_flux","linear_signal","finite_lens_flux","roi"],
        [(i,paths[i].zenith,paths[i].azimuth,omega[i],f[i],s[i],finite_flux[i],roi[i]) for i in eachindex(f)])
    write_csv(joinpath(out,"exposure_shares.csv"),vcat(["bin"],[name for (name,_) in profiles]),
        [vcat([i],[p[i] for (_,p) in profiles]) for i in eachindex(f)])
    open(joinpath(out,"frozen_arrays.bin"),"w") do io;serialize(io,(f=f,J=J,signal=s,lens=lens,roi=roi,omega=omega,profiles=profiles));end
    amplitude(A,y)=mean(first(sart_reconstruct(A,y;n_iter=200,relaxation=.2))[lens])
    seed=Int(get(cfg,"seed",20261024)); n=length(f)
    old_tau0=sqrt(args.sweep_exp_min*args.sweep_exp_max); E0=n*old_tau0
    old_sigma=max.(sqrt.(f./old_tau0),1e-12*maximum(f))
    legacy_info=sum(s.^2 ./ f); legacy_roi=sum(f[roi])/sum(s[roi])^2
    abs(4.645^2/legacy_info-cfg["headline_matched_tau"])<=5e-4 || error("Headline matched estimate changed beyond printed precision")
    abs(4.645^2*legacy_roi-cfg["headline_roi_tau"])<=5e-4 || error("Headline ROI estimate changed beyond printed precision")
    p=profiles[1].second; b=p.*f; d=p.*s; B=spdiagm(0=>p)*J
    parity=Dict{String,Any}()
    parity["matched_conversion_relative_error"]=abs((sum(d.^2 ./ b)*n)/legacy_info-1)
    parity["roi_conversion_relative_error"]=abs((sum(b[roi])/sum(d[roi])^2/n)/legacy_roi-1)
    parity["noise_conversion_max_abs"]=maximum(abs.(sqrt.(b./E0).-old_sigma./n))
    z=randn(MersenneTwister(seed),n)
    parity["sart_null_max_abs"]=abs(amplitude(J,old_sigma.*z)-amplitude(B,old_sigma.*z./n))
    parity["sart_signal_max_abs"]=abs(amplitude(J,s.+old_sigma.*z)-amplitude(B,(s.+old_sigma.*z)./n))
    all(v<=1e-10 for v in values(parity)) || error("Legacy-flat parity gate failed: $parity")
    benchmark=Int(get(cfg,"benchmark_draws",32)); benchmark>0 || error("Positive benchmark draw count required")
    trial=zeros(benchmark)
    elapsed=@elapsed Threads.@threads for i in 1:benchmark
        trial[i]=amplitude(B,sqrt.(b./E0).*randn(MersenneTwister(seed+300000+i),n))
    end
    open(joinpath(out,"preflight.toml"),"w") do io;TOML.print(io,Dict("parity"=>parity,"bins"=>n,"cells"=>600,"lens_cells"=>lens,"legacy_tau0"=>old_tau0,"total_E0"=>E0,
        "benchmark_draws"=>benchmark,"benchmark_seconds"=>elapsed,"julia_threads"=>Threads.nthreads(),
        "linear_signal_relative_l2_discrepancy"=>norm(s.-(finite_flux.-f))/norm(finite_flux.-f),
        "scope"=>"E=sum directional exposures; tau_m=E*p_m. Original equal-bin headline is E=N*tau_old. Same first-order signal for all detectors; finite-lens flux exported only as validity audit."));end
    get(cfg,"preflight_only",false) && return
    Mn=Int(get(cfg,"gaussian_null",3000)); Ms=Int(get(cfg,"gaussian_signal",600))
    pilot_rows=Any[]; planning_rows=Any[]
    z95=4.645; alpha=.0013498980316301 # Preserve headline planning convention.
    poisson_helper=joinpath(@__DIR__,"acceptance_poisson_draws.py")
    for (pi,(name,p)) in enumerate(profiles)
        b=p.*f; d=p.*s; B=spdiagm(0=>p)*J
        info=sum(d.^2 ./ b); Epsi=z95^2/info
        Eroi=z95^2*sum(b[roi])/sum(d[roi])^2
        sigma=sqrt.(b./E0)
        An=zeros(Mn); As=zeros(Ms)
        Threads.@threads for i in 1:Mn
            An[i]=amplitude(B,sigma.*randn(MersenneTwister(seed+100000+i),n))
        end
        Threads.@threads for i in 1:Ms
            As[i]=amplitude(B,d.+sigma.*randn(MersenneTwister(seed+200000+i),n))
        end
        nullmean=mean(An); delta=mean(As)-nullmean
        threshold=quantile(An,1-alpha)-nullmean; eps=As.-mean(As)
        crossing=exposure_search(E->mean((delta*sqrt(E/E0).+eps).>threshold),E0)
        push!(planning_rows,(name,sum(p),Epsi,Eroi,crossing.exposure,crossing.bracketed,crossing.lower,crossing.upper,crossing.rate_lower,crossing.rate_upper,E0,sum(b),sum(b+d),sum(d)))
        write_csv(joinpath(out,name*"_gaussian.csv"),["kind","draw","amplitude"],vcat([("null",i,x) for (i,x) in enumerate(An)],[("signal",i,x) for (i,x) in enumerate(As)]))
        get(cfg,"poisson_pilot",false) || continue
        for factor in get(cfg,"pilot_exposure_factors",[.5,1.,2.])
            E=factor*Epsi; folder=joinpath(out,name*"_poisson_"*string(factor));mkpath(folder)
            writedlm(joinpath(folder,"null_rates.txt"),b);writedlm(joinpath(folder,"signal_rates.txt"),b+d)
            ncal=Int(get(cfg,"poisson_calibration_null",3000)); nt=Int(get(cfg,"poisson_test_null",3000)); ns=Int(get(cfg,"poisson_signal",600))
            stats=Dict{String,Matrix{Float64}}()
            for (ki,(kind,draws,rates)) in enumerate((("calibration_null",ncal,"null"),("test_null",nt,"null"),("signal",ns,"signal")))
                counts_path=joinpath(folder,kind*".bin")
                local_seed=seed+1000000*pi+10000*findfirst(==(factor),get(cfg,"pilot_exposure_factors",[.5,1.,2.]))+1000*ki
                run(`python3 $poisson_helper draw --rates $(joinpath(folder,rates*"_rates.txt")) --exposure $E --draws $draws --seed $local_seed --output $counts_path`)
                counts=reshape(reinterpret(Int64,read(counts_path)),n,draws)
                result=zeros(draws,3)
                Threads.@threads for i in 1:draws
                    excess=counts[:,i]./E.-b
                    result[i,1]=dot(d./b,excess)
                    result[i,2]=sum(excess[roi])
                    result[i,3]=amplitude(B,excess)
                end
                stats[kind]=result
                writedlm(joinpath(folder,kind*"_statistics.csv"),result,',')
            end
            for (mi,method) in enumerate(("matched_first_order","roi_count","sart_200"))
                cut=quantile(stats["calibration_null"][:,mi],1-alpha)
                kn=count(>(cut),stats["test_null"][:,mi]);ks=count(>(cut),stats["signal"][:,mi])
                push!(pilot_rows,(name,method,factor,E,E*sum(b),E*sum(b+d),cut,kn,nt,ks,ns))
            end
        end
        # Checkpoint after every completed profile, including raw test outcomes.
        write_csv(joinpath(out,"planning.csv"),["profile","sum_p","matched_E95","roi_E95","sart_E95","sart_bracketed","sart_lower","sart_upper","rate_lower","rate_upper","reference_E","background_rate","signal_hypothesis_rate","excess_rate"],planning_rows)
        if !isempty(pilot_rows)
            raw=joinpath(out,"poisson_pilot_raw.csv");write_csv(raw,["profile","method","exposure_factor","total_E","expected_null_counts","expected_signal_counts","threshold","null_exceedances","null_draws","signal_exceedances","signal_draws"],pilot_rows)
            run(`python3 $poisson_helper intervals --input $raw --output $(joinpath(out,"poisson_pilot_intervals.csv"))`)
        end
    end
    # Gaussian-only jobs also emit the table (the pilot branch checkpoints above).
    write_csv(joinpath(out,"planning.csv"),["profile","sum_p","matched_E95","roi_E95","sart_E95","sart_bracketed","sart_lower","sart_upper","rate_lower","rate_upper","reference_E","background_rate","signal_hypothesis_rate","excess_rate"],planning_rows)
end

if get(ACCEPTANCE_CONFIG,"load_only",false)
    println("Acceptance helper runtime include/load smoke: PASS (no numerical job)")
else
    acceptance_main()
end
