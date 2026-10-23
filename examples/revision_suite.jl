#!/usr/bin/env julia
#=
Bounded reviewer experiments. Run with JULIA_NUM_THREADS=1 (or a resource-limited
remote job): julia --project=. examples/revision_suite.jl --output PATH
Options: --rays-step 30 --samples 32 --mc-samples 256 --cases 24 --replicates 4
All CSV rows are actual measurements, not precomputed manuscript numbers.
Acceptance scenarios are illustrative geometries, not measured LVD acceptance.
=#
include(joinpath(@__DIR__, "lvd_tomography.jl"))
using TOML
using SHA
using Dates

function revision_options(argv)
    o = Dict("output" => joinpath(@__DIR__, "data", "revision"), "rays-step" => "30",
             "samples" => "32", "mc-samples" => "64", "cases" => "12",
             "validation-zeniths" => "0,15,30,45,60,75", "replicates" => "3", "seed" => "20261022", "rock-density" => "2710", "calibrate" => "true", "dump" => joinpath(@__DIR__,"data","revision_physics","materials.pumas"))
    iseven(length(argv)) || error("Options require --name value pairs")
    for k in 1:2:length(argv)
        key = replace(argv[k], r"^--" => "")
        haskey(o,key) || error("Unknown option $(argv[k])")
        o[key] = argv[k+1]
    end
    o["calibrate"] in ("true","false") || error("calibrate must be true or false")
    parse(Float64,o["rock-density"]) > 0 || error("rock-density must be positive")
    for key in ("samples", "mc-samples", "cases", "replicates")
        parse(Int,o[key]) > 0 || error("$key must be positive")
    end
    angularstep=parse(Float64,o["rays-step"])
    0 < angularstep <= 60 || error("rays-step must be in (0,60]")
    isapprox(60/angularstep,round(60/angularstep);atol=1e-8) || error("rays-step must divide 60 degrees")
    parse(Int,o["cases"])>=length(validation_zeniths(o)) || error("cases must cover each requested validation zenith")
    o
end

function validation_zeniths(o)
    values=parse.(Float64,split(o["validation-zeniths"],','))
    !isempty(values) && all(x->isfinite(x) && 0<=x<90,values) && length(unique(values))==length(values) || error("validation-zeniths must be unique finite angles in [0,90)")
    sort(values)
end

function csv_rows(path, rows)
    isempty(rows) && return
    open(path,"w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows
            println(io, join((x isa AbstractString ? "\"" * replace(x,"\""=>"\"\"") * "\"" : string(x) for x in values(row)), ','))
        end
    end
end

function measured_stage(f)
    f()  # compile and warm this exact stage before measurement
    GC.gc()
    t = @timed f()
    (value=t.value, seconds=t.time, allocated_bytes=t.bytes,
     gc_seconds=t.gctime, process_peak_rss_bytes=Sys.maxrss())
end

# Rank strata avoid selecting only large entries. Round-robin over the joint
# zenith/slant/depth/sensitivity quartiles, then seeded sampling within strata.
function stratified_pairs(paths, volume, J, count_cases, rng)
    bi, ci, vv = findnz(J)
    keep = findall(k -> isfinite(vv[k]) && vv[k] != 0, eachindex(vv))
    isempty(keep) && error("No nonzero Jacobian entries")
    features = ([paths[bi[k]].zenith for k in keep],
        [paths[bi[k]].surface_distance for k in keep],
        [volume.centroids[3,ci[k]] for k in keep], abs.(vv[keep]))
    edges = [quantile(x,[0.25,0.5,0.75]) for x in features]
    groups = Dict{NTuple{4,Int},Vector{Int}}()
    for (n,k) in enumerate(keep)
        stratum = ntuple(j -> 1 + count(<(features[j][n]), edges[j]),4)
        push!(get!(groups,stratum,Int[]),k)
    end
    strata = shuffle(rng,sort!(collect(keys(groups))))
    for v in values(groups); shuffle!(rng,v); end
    chosen = Int[]
    while length(chosen) < min(count_cases,length(keep))
        for s in strata
            isempty(groups[s]) || push!(chosen,pop!(groups[s]))
            length(chosen) >= min(count_cases,length(keep)) && break
        end
    end
    [(bin=bi[k],cell=ci[k]) for k in chosen]
end

function validation_pairs(paths,vol,J,ncase,rng,zeniths)
    pairs=stratified_pairs(paths,vol,J,ncase,rng)
    # Guarantee representation of each requested valid zenith when case budget allows.
    represented=Set(paths[p.bin].zenith for p in pairs)
    ncase>=length(zeniths) || error("cases must cover each requested validation zenith")
    for theta in zeniths
        theta in represented && continue
        candidates=[(bin=b,cell=c) for (b,c,v) in zip(findnz(J)...) if paths[b].zenith==theta && isfinite(v) && v!=0]
        isempty(candidates) && continue
        counts=Dict(t=>count(p->paths[p.bin].zenith==t,pairs) for t in unique(p.zenith for p in paths))
        replaceindex=findlast(p->counts[paths[p.bin].zenith]>1,pairs)
        replaceindex===nothing && error("Insufficient pair budget for zenith coverage")
        pairs[replaceindex]=rand(rng,candidates);push!(represented,theta)
    end
    pairs
end

function revision_main(argv=ARGS)
    o=revision_options(argv)
    isfile(o["dump"]) || error("Physics dump not found: $(o["dump"]). Run dev/create_revision_physics.sh before revision experiments.")
    out=abspath(o["output"]); mkpath(out)
    seed=parse(Int,o["seed"]); rng=MersenneTwister(seed)
    nenergy=parse(Int,o["samples"]); nmc=parse(Int,o["mc-samples"])
    nrep=parse(Int,o["replicates"]); ncase=parse(Int,o["cases"])
    step=parse(Float64,o["rays-step"])
    open(joinpath(out,"config.toml"),"w") do io
        TOML.print(io,merge(o,Dict("julia_version"=>string(VERSION),
            "started_utc"=>string(now(UTC)),
            "suite_sha256"=>bytes2hex(sha256(read(@__FILE__))),
            "physics_dump_path"=>abspath(o["dump"]),
            "physics_dump_sha256"=>bytes2hex(sha256(read(o["dump"]))),
            "physics_dump_provenance"=>"input dump loaded without regeneration; inspect adjacent provenance.json for producer and source details",
            "physics_provenance_sha256"=>isfile(joinpath(dirname(o["dump"]),"provenance.json")) ? bytes2hex(sha256(read(joinpath(dirname(o["dump"]),"provenance.json")))) : "unavailable",
            "energy_quadrature"=>"deterministic midpoint log-energy CSDA quadrature; independent random MC energy samples per replicate; paired MC-energy CSDA secants isolate shared quadrature randomness",
            "adjoint_rules_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"..","src","Tomography_rules.jl")))),
            "manifest_sha256"=>isfile(joinpath(@__DIR__,"..","Manifest.toml")) ? bytes2hex(sha256(read(joinpath(@__DIR__,"..","Manifest.toml")))) : "absent",
            "threads"=>Threads.nthreads(),"cpu_threads"=>Sys.CPU_THREADS,
            "selection_scope"=>"seeded round-robin joint quartiles with replacement of overrepresented zenith cases to guarantee each valid requested zenith with nonzero sensitivity; invalid/zero-sensitivity directions recorded, not certified",
            "seed_policy"=>"independent replicate energy and transport seeds; common low/high seeds within replicate",
            "acceptance_scope"=>"illustrative normalized geometric profiles, not measured LVD acceptance",
            "transport_scope"=>"fixed prescribed-ray stochastic energy-loss MC; scattering disabled, no scattered-boundary retracing or detector-acceptance validation",
            "closure_scope"=>"frozen global dry MC calibration predicts unseen spatially varying wet field with independent MC samples; matched independent dry/wet replicates share energy/transport seeds; paired flux-bias change includes dry/wet covariance; calibration uncertainty not propagated; uncalibrated mode separately records per-bin dry scalar comparator",
            "timing_scope"=>"uncorrected CSDA scaling stages; exact stage warmed once; allocated bytes are cumulative allocations; RSS is process cumulative peak",
            "jacobian_validation_scope"=>o["calibrate"]=="true" ? "frozen independent dry MC calibration against midpoint CSDA reference; wet validation fields and independent MC seeds" : "uncorrected pure CSDA",
            "robustness_scope"=>"local linearized wet increments; prior center, ridge strength, angular mask and flux normalization only; not nonlinear measured-data refits")))
    end
    physics=load_physics(o["dump"])
    physics === nothing && error("Could not load physics dump $(o["dump"])")
    idx(name)=get_material_index(physics,name)
    mat=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),idx("PorousWetRock"),
        parse(Float64,o["rock-density"]),Float64(physics.tables[idx("Water")].density),0.0,0.0)
    site=build_site(30000.0); emap=LVDTopo.build_elevation_map()
    es=csda_energy_quadrature(nenergy,1e-3,1e9)
    set_csda_correction!(enabled=false)
    timing=NamedTuple[]; finals=nothing
    for nz in (2,4,8), angular in (2step,step)
        volstage=measured_stage(() -> create_voxel_volume(emap,mat;half_width_km=3.0,surface_step_km=1.5,nz=nz))
        vol=volstage.value; nc=num_cells(vol); shallow=falses(nc); w=fill(0.3,nc)
        trace=measured_stage(() -> precompute_paths(vol,emap,exclusive_degree_grid(60.0,angular),exclusive_degree_grid(360.0,angular)))
        paths=filter(p->p.valid,trace.value); isempty(paths) && error("No valid paths")
        # Warm compilation before measurements on each new shape.
        assemble_forward_and_jacobian(physics,shallow,mat,site,paths[1:1],w,es;n_cells=nc,threaded=false)
        forward=measured_stage(() -> [compute_directional_flux_csda(physics,shallow,mat,site,p,w,es) for p in paths])
        jac=measured_stage(() -> assemble_forward_and_jacobian(physics,shallow,mat,site,paths,w,es;n_cells=nc,threaded=false))
        flux,J=jac.value; W=1 ./ max.(abs.(flux),1e-20)
        normal=measured_stage(() -> Matrix(J' * Diagonal(W) * J))
        H=normal.value; ridge=max(maximum(diag(H))*1e-4,1e-20)
        factor=measured_stage(() -> cholesky(Symmetric(H+ridge*I)))
        solve=measured_stage(() -> factor.value \ ones(nc))
        for (stage,measurement) in (("volume",volstage),("trace",trace),("forward",forward),("jacobian",jac),("normal",normal),("factor",factor),("solve",solve))
            push!(timing,(nz=nz,rays=length(paths),cells=nc,intersections=sum(length(p.segments) for p in paths),nnz=nnz(J),stage=stage,
                seconds=measurement.seconds,allocated_bytes=measurement.allocated_bytes,gc_seconds=measurement.gc_seconds,
                process_peak_rss_bytes=measurement.process_peak_rss_bytes,retained_jacobian_bytes=Base.summarysize(J),dense_normal_bytes=Base.summarysize(H)))
        end
        csv_rows(joinpath(out,"scaling.csv"),timing)
        finals=(vol,paths,shallow,w,flux,J)
    end
    vol,paths,shallow,w,flux,J=finals
    if o["calibrate"]=="true"
        calargs=(correction=true,calibration_bins=min(ncase,length(paths)),calibration_mc_samples=nmc,
            energy_min=1e-3,energy_max=1e9,seed=seed,straggling=true,energy_threshold_low=100.0)
        calibrate_correction_to_mc(physics,shallow,mat,site,paths,zeros(length(w)),es,calargs)
        flux,J=assemble_forward_and_jacobian(physics,shallow,mat,site,paths,w,es;n_cells=length(w),threaded=false)
    end
    # Scaling/acceptance retain their original 0<=theta<60 angular domain.
    scaling_state=(paths,flux,J)
    requested=precompute_paths(vol,emap,validation_zeniths(o),exclusive_degree_grid(360.0,step))
    coverage=[(zenith=theta,requested_rays=count(p->p.zenith==theta,requested),valid_rays=count(p->p.zenith==theta && p.valid,requested)) for theta in validation_zeniths(o)]
    paths=filter(p->p.valid,requested)
    isempty(paths) && error("No valid validation paths")
    flux,J=assemble_forward_and_jacobian(physics,shallow,mat,site,paths,w,es;n_cells=length(w),threaded=false)
    correction=get_csda_correction()
    open(joinpath(out,"calibration.toml"),"w") do io
        TOML.print(io,Dict(string(name)=>getfield(correction,name) for name in fieldnames(typeof(correction))))
    end
    pairs=validation_pairs(paths,vol,J,ncase,rng,validation_zeniths(o)); fdrows=NamedTuple[]; mcrows=NamedTuple[]
    validation_bi,_,validation_values=findnz(J)
    csv_rows(joinpath(out,"validation_coverage.csv"),[(;row...,selected_rays=length(unique(p.bin for p in pairs if paths[p.bin].zenith==row.zenith)),selected_pairs=count(p->paths[p.bin].zenith==row.zenith,pairs),population_nonzero_entries=count(k->paths[validation_bi[k]].zenith==row.zenith && isfinite(validation_values[k]) && validation_values[k]!=0,eachindex(validation_values))) for row in coverage])
    csv_rows(joinpath(out,"selected_pairs.csv"),[(case=k,bin=p.bin,cell=p.cell,zenith=paths[p.bin].zenith,azimuth=paths[p.bin].azimuth,slant_m=paths[p.bin].surface_distance,cell_z_m=vol.centroids[3,p.cell],reference_jacobian=J[p.bin,p.cell]) for (k,p) in enumerate(pairs)])
    for fraction in (0.0,0.3,0.8)
        wf=fill(fraction,length(w)); f,Jf=assemble_forward_and_jacobian(physics,shallow,mat,site,paths,wf,es;n_cells=length(w),threaded=false)
        for (k,pair) in enumerate(pairs)
            b,c=pair.bin,pair.cell; ad=Jf[b,c]
            for delta in (1e-2,1e-3,1e-4,1e-5,1e-6)
                lo,hi=finite_difference_bounds(fraction,delta); wl=copy(wf); wh=copy(wf); wl[c]=lo; wh[c]=hi
                fl=compute_directional_flux_csda(physics,shallow,mat,site,paths[b],wl,es)
                fh=compute_directional_flux_csda(physics,shallow,mat,site,paths[b],wh,es)
                fd=(fh-fl)/(hi-lo)
                push!(fdrows,(case=k,bin=b,cell=c,zenith=paths[b].zenith,azimuth=paths[b].azimuth,slant_m=paths[b].surface_distance,cell_z_m=vol.centroids[3,c],fraction=fraction,step=delta,ad=ad,fd=fd,error=ad-fd,relative_error=(ad-fd)/max(abs(ad),abs(fd),1e-30),boundary_one_sided=fraction==0.0))
            end
            for rep in 1:nrep
                sd=seed+100000+1000*k+100*rep+round(Int,10fraction)
                energies=sample_energy_set(nmc,1e-3,1e9,sd)
                lo,hi=finite_difference_bounds(fraction,0.05); wl=copy(wf); wh=copy(wf); wl[c]=lo; wh[c]=hi
                ml,dl=build_cell_properties_for_mc(shallow,wl,mat); mh,dh=build_cell_properties_for_mc(shallow,wh,mat)
                fl,sl=compute_directional_flux_mc(physics,mat,site,paths[b],ml,dl,energies,sd+1;straggling=true,scattering=false,energy_threshold_low=100.0)
                fh,sh=compute_directional_flux_mc(physics,mat,site,paths[b],mh,dh,energies,sd+1;straggling=true,scattering=false,energy_threshold_low=100.0)
                cslo=compute_directional_flux_csda(physics,shallow,mat,site,paths[b],wl,es)
                cshi=compute_directional_flux_csda(physics,shallow,mat,site,paths[b],wh,es)
                cssecant=(cshi-cslo)/(hi-lo)
                paired_cslo=compute_directional_flux_csda(physics,shallow,mat,site,paths[b],wl,energies)
                paired_cshi=compute_directional_flux_csda(physics,shallow,mat,site,paths[b],wh,energies)
                paired_cssecant=(paired_cshi-paired_cslo)/(hi-lo)
                push!(mcrows,(case=k,bin=b,cell=c,fraction=fraction,replicate=rep,seed=sd,
                    fraction_lo=lo,fraction_hi=hi,interval_width=hi-lo,ad=ad,csda_secant=cssecant,
                    mc_fd=(fh-fl)/(hi-lo),ad_minus_secant=ad-cssecant,mc_minus_csda_secant=(fh-fl)/(hi-lo)-cssecant,
                    csda_secant_paired_energy=paired_cssecant,mc_minus_csda_paired_energy=(fh-fl)/(hi-lo)-paired_cssecant,
                    flux_low=fl,flux_high=fh,sigma_low=sl,sigma_high=sh))
                csv_rows(joinpath(out,"mc_jacobian.csv"),mcrows)
            end
        end
        csv_rows(joinpath(out,"fd_steps.csv"),fdrows)
    end
    # Repeat complete reverse passes after changing the scalar custom-adjoint step.
    internalsteps=NamedTuple[]; original_step=get_adjoint_fd_step()
    try
        for h in (1e-3,1e-4,1e-5,1e-6,1e-7,1e-8)
            set_adjoint_fd_step!(h)
            _,Js=assemble_forward_and_jacobian(physics,shallow,mat,site,paths,w,es;n_cells=length(w),threaded=false)
            for (k,pair) in enumerate(pairs)
                push!(internalsteps,(case=k,bin=pair.bin,cell=pair.cell,adjoint_step=h,gradient=Js[pair.bin,pair.cell],reference_gradient=J[pair.bin,pair.cell],relative_change=(Js[pair.bin,pair.cell]-J[pair.bin,pair.cell])/max(abs(J[pair.bin,pair.cell]),1e-30)))
            end
        end
    finally
        set_adjoint_fd_step!(original_step)
    end
    csv_rows(joinpath(out,"internal_adjoint_steps.csv"),internalsteps)
    # Independent replicate dispersion estimates MC derivative error including CRN covariance.
    summaries=NamedTuple[]
    for fraction in (0.0,0.3,0.8), k in eachindex(pairs)
        rows=filter(r->r.case==k && r.fraction==fraction,mcrows); vals=[r.mc_fd for r in rows]
        se=length(vals)>1 ? std(vals)/sqrt(length(vals)) : NaN
        paireddiff=[r.mc_minus_csda_paired_energy for r in rows]
        pairedse=length(paireddiff)>1 ? std(paireddiff)/sqrt(length(paireddiff)) : NaN
        push!(summaries,(case=k,fraction=fraction,fraction_lo=first(rows).fraction_lo,fraction_hi=first(rows).fraction_hi,
            ad=first(rows).ad,csda_secant=first(rows).csda_secant,mc_mean=mean(vals),mc_standard_error=se,
            ad_minus_secant=first(rows).ad-first(rows).csda_secant,
            mc_minus_csda_secant=mean(vals)-first(rows).csda_secant,
            mc_secant_zscore=(mean(vals)-first(rows).csda_secant)/se,
            paired_energy_csda_mean=mean(r.csda_secant_paired_energy for r in rows),
            paired_energy_discrepancy_mean=mean(paireddiff),paired_energy_discrepancy_standard_error=pairedse,
            paired_energy_discrepancy_zscore=mean(paireddiff)/pairedse))
    end
    csv_rows(joinpath(out,"mc_jacobian_summary.csv"),summaries)
    # Frozen dry scalar calibration; all wet-field MC values use unseen samples.
    closure=NamedTuple[]; dry=zeros(length(w)); wet=[0.15+0.45*(sin(vol.centroids[1,c]/700)^2)*(vol.centroids[3,c]>350) for c in eachindex(w)]
    selected=unique(p.bin for p in pairs)
    for b in selected
        drycs=compute_directional_flux_csda(physics,shallow,mat,site,paths[b],dry,es)
        md,dd=build_cell_properties_for_mc(shallow,dry,mat)
        train=sample_energy_set(nmc,1e-3,1e9,seed+500000+b)
        fm,_=compute_directional_flux_mc(physics,mat,site,paths[b],md,dd,train,seed+510000+b;straggling=true,scattering=false,energy_threshold_low=100.0)
        correction=fm/max(drycs,1e-30)
        frozen_paper_pred=compute_directional_flux_csda(physics,shallow,mat,site,paths[b],wet,es)
        per_bin_scalar_pred=correction*frozen_paper_pred
        pred=o["calibrate"]=="true" ? frozen_paper_pred : per_bin_scalar_pred
        preddry=o["calibrate"]=="true" ? drycs : correction*drycs
        mw,dw=build_cell_properties_for_mc(shallow,wet,mat)
        for rep in 1:nrep
            sd=seed+600000+100*b+rep; test=sample_energy_set(nmc,1e-3,1e9,sd)
            drytruth,drysigma=compute_directional_flux_mc(physics,mat,site,paths[b],md,dd,test,sd+1;straggling=true,scattering=false,energy_threshold_low=100.0)
            truth,sigma=compute_directional_flux_mc(physics,mat,site,paths[b],mw,dw,test,sd+1;straggling=true,scattering=false,energy_threshold_low=100.0)
            push!(closure,(bin=b,zenith=paths[b].zenith,azimuth=paths[b].azimuth,slant_m=paths[b].surface_distance,replicate=rep,seed=sd,predicted_dry=preddry,frozen_global_dry_prediction=drycs,mc_dry=drytruth,sigma_mc_dry=drysigma,dry_relative_residual=(preddry-drytruth)/max(abs(drytruth),1e-30),paired_bias_change=(pred-truth)-(preddry-drytruth),paired_mc_wetting_increment=truth-drytruth,dry_correction=correction,prediction_mode=o["calibrate"]=="true" ? "frozen_global_dry_calibration" : "per_bin_dry_scalar",predicted_wet=pred,frozen_global_prediction=frozen_paper_pred,per_bin_scalar_prediction=per_bin_scalar_pred,mc_wet=truth,sigma_mc=sigma,relative_residual=(pred-truth)/max(abs(truth),1e-30)))
        end
        csv_rows(joinpath(out,"held_out_closure.csv"),closure)
    end
    paths,flux,J=scaling_state
    # Fisher matched-filter versus summed-ROI counting; all profiles normalized
    # to equal mean acceptance, hence equal integrated nominal exposure.
    # Midpoint angular cells integrate the sampled 0<=zenith<60 domain. The
    # vertical cap is split between azimuth wedges; last ring ends at 60 degrees.
    thetas=sort(unique(p.zenith for p in paths))
    theta_edges=vcat(0.0,[(thetas[j]+thetas[j+1])/2 for j in 1:length(thetas)-1],60.0)
    theta_index=Dict(theta=>j for (j,theta) in enumerate(thetas))
    omega=[deg2rad(step)*(cosd(theta_edges[theta_index[p.zenith]])-cosd(theta_edges[theta_index[p.zenith]+1])) for p in paths]
    detection=NamedTuple[]; signal=J*wet; roi=abs.(signal).>maximum(abs.(signal))*0.05
    for (name,raw) in (("uniform",ones(length(paths))),
        ("cosine",[cosd(p.zenith) for p in paths]),
        ("azimuth_modulated",[cosd(p.zenith)*(1+0.5*cosd(2p.azimuth)) for p in paths]))
        a=raw./(sum(omega.*raw)/sum(omega)); exposure_weights=omega.*a
        info=sum(exposure_weights.*signal.^2 ./ max.(flux,1e-30))
        roiinfo=sum((exposure_weights.*signal)[roi])^2/sum((exposure_weights.*flux)[roi])
        # One-sided 3-sigma null threshold and .95 power, matching the paper convention.
        push!(detection,(acceptance=name,total_solid_angle_sr=sum(omega),acceptance_solid_angle_integral=sum(exposure_weights),matched_information=info,roi_information=roiinfo,
            false_positive_probability=0.00134989803163,target_power=0.95,
            tau95_matched=(3+1.64485362695147)^2/info,tau95_roi=(3+1.64485362695147)^2/roiinfo,
            exposure_gain=1-roiinfo/info))
    end
    csv_rows(joinpath(out,"acceptance.csv"),detection)
    # Linearized reconstruction robustness: lambda relative to normal curvature,
    # altered starting/prior field, and explicit angular sector masking.
    sensitivity=NamedTuple[]; obs=flux+signal; nc=length(w)
    for maskname in ("all","exclude_135_180"), lambda in (1e-4,1e-2,1.0), priorlevel in (0.0,0.3), fluxscale in (0.98,1.0,1.02)
        keep=maskname=="all" ? trues(length(paths)) : [!(135<=p.azimuth<=180 && 40<=p.zenith<=55) for p in paths]
        base=flux[keep]*fluxscale; A=J[keep,:]; weights=1 ./ max.(flux[keep],1e-30)
        H=Matrix(A'*Diagonal(weights)*A); reg=lambda*max(maximum(diag(H)),1e-20)
        fit=clamp.((H+reg*I)\(A'*(weights.*(obs[keep]-base))+reg*fill(priorlevel,nc)),0.0,0.9)
        push!(sensitivity,(mask=maskname,lambda_relative=lambda,prior_fraction=priorlevel,flux_scale=fluxscale,bins=count(keep),field_rmse=sqrt(mean((fit-wet).^2)),mean_field=mean(fit),data_relative_rmse=norm(base+A*fit-obs[keep])/norm(obs[keep])))
    end
    csv_rows(joinpath(out,"linearized_robustness.csv"),sensitivity)
    println("Revision suite complete: $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    revision_main()
end
