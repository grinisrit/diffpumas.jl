#!/usr/bin/env julia
# Finite training-MC parameter dispersion, distinct from structural model error.
include(joinpath(@__DIR__,"validate_surrogate_v2.jl"))
using SparseArrays, LinearAlgebra, ForwardDiff

function calibration_ensemble(argv=ARGS)
    iseven(length(argv)) || error("expected --name value pairs")
    opts=Dict(k=>"" for k in ("output","dump","artifact","baseline","coverage","training"))
    for i in 1:2:length(argv)
        k=replace(argv[i],r"^--"=>"");haskey(opts,k)&&isempty(opts[k]) || error("unknown/duplicate option")
        opts[k]=abspath(argv[i+1])
    end
    all(!isempty,values(opts)) || error("all options required")
    out=opts["output"];ispath(out)&&error("fresh data output required")
    inputs=[@__FILE__,joinpath(@__DIR__,"validate_surrogate_v2.jl"),joinpath(@__DIR__,"..","examples","revision_suite.jl"),joinpath(@__DIR__,"..","examples","lvd_tomography.jl"),opts["dump"],opts["artifact"],opts["training"]]
    append!(inputs,[joinpath(opts["baseline"],n) for n in ("lvd_fit_field.csv","lvd_fit_predictions.csv","lvd_solver_configuration.toml","lvd_mc_coverage.csv")])
    append!(inputs,[joinpath(opts["coverage"],n) for n in ("final_field_coverage.csv","final_jacobian_sparse.csv","coverage_summary.toml","end_status.json")])
    append!(inputs,filter(p->endswith(p,".jl"),readdir(joinpath(@__DIR__,"..","src");join=true)))
    hashes=Dict(abspath(p)=>bytes2hex(sha256(read(p))) for p in inputs)
    coverage_audit=TOML.parsefile(joinpath(opts["coverage"],"coverage_summary.toml"))["audit"]
    isempty(coverage_audit["changed_inputs"])&&coverage_audit["prediction_bitwise_parity"]&&coverage_audit["jacobian_forward_bitwise_parity"]||error("coverage provenance gates")
    for name in (joinpath(opts["baseline"],"lvd_fit_field.csv"),joinpath(opts["baseline"],"lvd_fit_predictions.csv"),opts["artifact"],opts["dump"],joinpath(@__DIR__,"..","src","Tomography.jl"),joinpath(@__DIR__,"..","src","Tomography_rules.jl"))
        get(coverage_audit["input_sha256"],abspath(name),"")==bytes2hex(sha256(read(name)))||error("loaded Jacobian source/field binding")
    end
    metadata=load_csda_corrections!(opts["artifact"]);original=get_csda_material_correction();beta0=[original.linear,original.quadratic]
    metadata["physics_dump_sha256"]==hashes[opts["dump"]] || error("dump identity")
    cfg=TOML.parsefile(joinpath(opts["baseline"],"lvd_solver_configuration.toml"))
    cfg["correction_artifact_sha256"]==hashes[opts["artifact"]] || error("artifact identity")
    set_adjoint_derivative_mode!(:forwarddiff)
    ph,mat,site,em,trainvol,trainshallow,es=v2state(opts["dump"])
    train=filter(p->p.valid,precompute_paths(trainvol,em,Float64.(metadata["training_zeniths"]),Float64.(metadata["training_azimuths"])))
    length(train)==12 || error("original 12 training paths required")
    nt=num_cells(trainvol);trainfields=[zeros(nt),fill(.2,nt),fill(.6,nt)]
    dry=readrows(opts["training"]);length(dry)==24 || error("original training records")
    md=zeros(12);mw=zeros(2,12)
    for r in dry
        b=Int(r["bin"]);k=Int(r["field"])
        train[b].zenith==Float64(r["zenith"])&&train[b].azimuth==Float64(r["azimuth"]) || error("training path identity")
        md[b]=Float64(r["mc_dry"]);mw[k,b]=Float64(r["mc_wet"])
    end
    replay=calibrate_csda_material_correction(ph,trainshallow,mat,site,train,trainfields[1],trainfields[2:3],es,md,mw,collect(1:12);ridge=1e-6)
    isequal([replay.linear,replay.quadratic],beta0) || error("original saved-training coefficient replay failed")
    load_csda_corrections!(opts["artifact"])
    vol=create_voxel_volume(em,mat;half_width_km=5.,surface_step_km=1.,nz=6)
    nc=num_cells(vol);nc==600||error("production grid")
    fields=readrows(joinpath(opts["baseline"],"lvd_fit_field.csv"));length(fields)==600||error("field dimension")
    w=zeros(nc);support=falses(nc)
    for (c,r) in enumerate(fields)
        Int(r["cell_idx"])==c&&isequal(Tuple(Float64(r[k]) for k in ("x_m","y_m","z_m")),cell_centroid(vol,c)) || error("production geometry identity")
        w[c]=Float64(r["w"]);support[c]=string(r["aquifer_mask"])=="true"
    end
    all(x->isfinite(x)&&0<=x<=MAX_WATER_FRACTION,w)&&all(iszero,w[.!support])||error("field box/support")
    count(support)==147&&bytes2hex(sha256(join(findall(support),',')))==cfg["support_sha256"]||error("support identity")
    saved=readrows(joinpath(opts["baseline"],"lvd_fit_predictions.csv"));ids=Int[r["bin_idx"] for r in saved]
    length(ids)==1003&&length(unique(ids))==1003||error("prediction dimensions")
    fit=Bool[Float64(r["fit_mask"])>0 for r in saved];count(fit)==690||error("fit dimensions")
    covrows=readrows(joinpath(opts["coverage"],"final_field_coverage.csv"));Int[r["bin_idx"] for r in covrows]==ids||error("coverage order")
    paths=[trace_path(vol,Float64(r["zenith_deg"]),Float64(r["azimuth_deg"]),em) for r in covrows];all(p->p.valid,paths)||error("invalid production path")
    shallow=falses(nc);f0=[compute_directional_flux_csda(ph,shallow,mat,site,p,w,es) for p in paths]
    isequal(reinterpret(UInt64,f0),reinterpret(UInt64,Float64[r["matched_csda_flux"] for r in saved]))||error("final predictions bitwise gate")
    byid=Dict(id=>i for (i,id) in enumerate(ids));jrows=readrows(joinpath(opts["coverage"],"final_jacobian_sparse.csv"))
    J0=sparse([byid[Int(r["bin_idx"])] for r in jrows],Int[r["cell_idx"] for r in jrows],Float64[r["derivative"] for r in jrows],1003,600)
    q=zeros(1003);di=Int[];dc=Int[];dv=Float64[]
    for (i,p) in enumerate(paths)
        mass=p.remaining_rock_distance*mat.rock_density;watermass=0.
        for seg in p.segments
            rho=cell_density(w[seg.cell_idx],false,mat);g=seg.distance*rho
            mass+=g;watermass+=w[seg.cell_idx]*g
        end
        isfinite(mass)&&mass>0||error("positive finite rock-mixture mass required")
        q[i]=watermass/max(mass,1e-30)
        for seg in p.segments
            c=seg.cell_idx;rho=cell_density(w[c],false,mat);g=seg.distance*rho
            dg=-seg.distance*rho^2*(1/mat.water_density-1/mat.rock_density)
            push!(di,i);push!(dc,c);push!(dv,(g+(w[c]-q[i])*dg)/mass)
        end
    end
    Dq=sparse(di,dc,dv,1003,600)
    # Validate the geometry-only derivative independently by dual arithmetic.
    checked=[1,argmin(abs.([p.zenith-62 for p in paths])),argmin(abs.([p.zenith-78 for p in paths]))]
    for i in checked
        p=paths[i];qfun=v->DiffPumas.Tomography._column_water_fraction(mat,p.remaining_rock_distance,[s.distance for s in p.segments],[s.cell_idx for s in p.segments],falses(length(p.segments)),v)
        isapprox(qfun(w),q[i];atol=1e-15,rtol=1e-14)||error("column composition mismatch")
        dual=ForwardDiff.gradient(qfun,w)
        maximum(abs.(dual.-Vector(Dq[i,:])))<=1e-12||error("analytic column derivative mismatch")
    end
    function propagated(beta)
        delta=beta.-beta0;ratio=exp.(delta[1].*q.+delta[2].*q.^2)
        flux=ratio.*f0
        J=spdiagm(0=>ratio)*J0+spdiagm(0=>flux.*(delta[1].+2delta[2].*q))*Dq
        flux,J
    end
    pf,pj=propagated(beta0);isequal(pf,f0)&&pj==J0||error("zero-delta formula identity")
    all(isone,exp.(beta0[1].*zeros(3).+beta0[2].*zeros(3)))||error("dry anchor")
    mkpath(out);mkpath(joinpath(out,"replicates"))
    plan=Dict("input_sha256"=>hashes,"replicates"=>8,"training_rays"=>12,"training_fields"=>[0.,.2,.6],"histories_per_ray_field"=>8192,"total_histories"=>2359296,"ridge"=>1e-6,"seed_formula"=>"3100000000 + replicate*1000000 + training_ray*10000; transport=energy+1; same streams across three fields within ray","dry_correction"=>"frozen; not refitted","coefficient_origin"=>beta0,"original_training_replay_exact"=>true,"final_prediction_bitwise_parity"=>true,"propagation_formula"=>"f_beta=r*f0; J_beta=r*J0 + diag(f_beta*(delta_beta1+2*delta_beta2*q))*Dq; r=exp(delta_beta1*q+delta_beta2*q^2)","column_fraction"=>"water mass / total traversed rock-mixture mass; includes remaining dry rock; excludes air","scope"=>"Empirical repeated finite-MC calibration parameter dispersion only; dry calibration held fixed. Not structural approximation error, data uncertainty, posterior, confidence interval, field refit or water-recovery accuracy. Final field lies below wet training range.")
    open(io->TOML.print(io,plan),joinpath(out,"preregistered_plan.toml"),"w")
    betas=zeros(8,2);coeffrows=NamedTuple[];ensemble=NamedTuple[];formula_checks=NamedTuple[]
    for rep in 1:8
        started=time();mcvalues=zeros(3,12);errors=zeros(3,12)
        Threads.@threads for b in 1:12
            seed=3_100_000_000+rep*1_000_000+b*10000;energy=sample_energy_set(8192,1e-3,1e9,seed)
            for k in 1:3
                mats,dens=build_cell_properties_for_mc(trainshallow,trainfields[k],mat)
                mcvalues[k,b],errors[k,b]=compute_directional_flux_mc(ph,mat,site,train[b],mats,dens,energy,seed+1;straggling=true,scattering=false,energy_threshold_low=100.)
            end
            csv_rows(joinpath(out,"replicates","rep_"*string(rep)*"_ray_"*string(b)*".csv"),[(replicate=rep,training_ray=b,zenith_deg=train[b].zenith,azimuth_deg=train[b].azimuth,water_fraction=trainfields[k][1],mc_flux=mcvalues[k,b],mc_sigma=errors[k,b],energy_seed=seed,transport_seed=seed+1,samples=8192) for k in 1:3])
        end
        all(x->isfinite(x)&&x>0,mcvalues)||error("nonpositive training flux; retain failed replicate, do not resample or omit")
        fitcoef=calibrate_csda_material_correction(ph,trainshallow,mat,site,train,trainfields[1],trainfields[2:3],es,vec(mcvalues[1,:]),mcvalues[2:3,:],collect(1:12);ridge=1e-6)
        betas[rep,:]=[fitcoef.linear,fitcoef.quadratic]
        push!(coeffrows,(replicate=rep,linear=fitcoef.linear,quadratic=fitcoef.quadratic,elapsed_seconds=time()-started,condition_number=fitcoef.condition_number,ridge=1e-6))
        csv_rows(joinpath(out,"coefficients.csv"),coeffrows)
        flux,J=propagated(vec(betas[rep,:]));absj=vec(sum(abs.(J[:,support]);dims=2))
        isequal(flux[q.==0],f0[q.==0])||error("dry-column flux anchor changed")
        for i in eachindex(ids)
            push!(ensemble,(replicate=rep,bin_idx=ids[i],fit_mask=fit[i],column_water_fraction=q[i],final_flux=flux[i],relative_flux_change=flux[i]/f0[i]-1,support_abs_jacobian=absj[i],support_abs_logflux_sensitivity=absj[i]/flux[i]))
        end
        csv_rows(joinpath(out,"fixed_field_ensemble.csv"),ensemble)
        if rep in (1,8)
            set_csda_material_correction!(linear=betas[rep,1],quadratic=betas[rep,2])
            for i in checked
                af,aj=assemble_forward_and_jacobian(ph,shallow,mat,site,[paths[i]],w,es;n_cells=nc,threaded=false)
                ferror=abs(af[1]-flux[i])/max(abs(af[1]),1e-300)
                jerror=maximum(abs.(Vector(aj[1,:]).-Vector(J[i,:])))/max(maximum(abs.(Vector(aj[1,:]))),1e-300)
                push!(formula_checks,(replicate=rep,bin_idx=ids[i],relative_flux_error=ferror,relative_max_jacobian_error=jerror))
                ferror<=1e-12&&jerror<=1e-10||error("closed factor/J formula validation failed")
            end
            csv_rows(joinpath(out,"formula_checks.csv"),formula_checks)
        end
        println("Calibration replicate ",rep,"/8 complete; beta=",betas[rep,:]," seconds=",time()-started);flush(stdout)
    end
    load_csda_corrections!(opts["artifact"])
    changed=[p for (p,h) in hashes if bytes2hex(sha256(read(p)))!=h]
    summary=Dict("replicates"=>8,"coefficient_mean"=>vec(mean(betas;dims=1)),"coefficient_std"=>vec(std(betas;dims=1)),"coefficient_min"=>vec(minimum(betas;dims=1)),"coefficient_max"=>vec(maximum(betas;dims=1)),"coefficient_covariance_rows"=>[collect(row) for row in eachrow(cov(betas))],"source_inputs_sha256"=>hashes,"changed_inputs"=>changed,"dry_anchor_multiplier"=>1.,"scope"=>plan["scope"])
    open(io->TOML.print(io,summary),joinpath(out,"ensemble_summary.toml"),"w")
    isempty(changed)||error("input changed during calibration ensemble")
end
if abspath(PROGRAM_FILE)==@__FILE__;calibration_ensemble();end
