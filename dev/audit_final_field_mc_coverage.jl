#!/usr/bin/env julia
# Deterministic final-field coverage diagnostic; no MC or optimization runs.
include(joinpath(@__DIR__, "validate_surrogate_v2.jl"))
using SparseArrays, LinearAlgebra

function final_coverage_options(argv)
    keys=("data", "artifact", "dump", "output")
    iseven(length(argv)) || error("expected --name value pairs")
    opts=Dict(k=>"" for k in keys)
    for i in 1:2:length(argv)
        k=replace(argv[i],r"^--"=>"")
        haskey(opts,k) && isempty(opts[k]) || error("unknown or repeated option")
        opts[k]=abspath(argv[i+1])
    end
    all(!isempty,values(opts)) || error("required: --data --artifact --dump --output")
    opts
end

function final_coverage(opts)
    dat=opts["data"];out=opts["output"]
    isdir(out) && !issubset(Set(readdir(out)),Set(["execution.json","run.log","pid.json"])) && error("refuse existing evidence")
    files=[joinpath(dat,n) for n in ("lvd_fit_field.csv","lvd_fit_mask.csv","lvd_fit_predictions.csv","lvd_mc_coverage.csv","lvd_solver_configuration.toml")]
    append!(files,[opts["artifact"],opts["dump"],@__FILE__,joinpath(@__DIR__,"validate_surrogate_v2.jl"),joinpath(@__DIR__,"..","examples","revision_suite.jl"),joinpath(@__DIR__,"..","examples","lvd_tomography.jl"),joinpath(@__DIR__,"..","Project.toml"),joinpath(@__DIR__,"..","Manifest.toml")])
    append!(files,filter(p->endswith(p,".jl"),readdir(joinpath(@__DIR__,"..","src");join=true)))
    hashes=Dict(abspath(p)=>bytes2hex(sha256(read(p))) for p in files)
    cfg=TOML.parsefile(joinpath(dat,"lvd_solver_configuration.toml"))
    cfg["adjoint_backend"]=="forwarddiff" || error("requires recorded forwarddiff backend")
    DiffPumas.Tomography.set_adjoint_derivative_mode!(:forwarddiff)
    cfg["correction_artifact_sha256"]==hashes[opts["artifact"]] || error("correction identity mismatch")
    metadata=load_csda_corrections!(opts["artifact"])
    metadata["physics_dump_sha256"]==hashes[opts["dump"]] || error("physics identity mismatch")
    metadata["csda_nodes"]==32 && metadata["energy_min"]==1e-3 && metadata["energy_max"]==1e9 || error("energy rule mismatch")
    ph=load_physics(opts["dump"]);ph===nothing && error("invalid physics")
    idx(n)=get_material_index(ph,n)
    mat=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),idx("PorousWetRock"),Float64(metadata["rock_density"]),Float64(ph.tables[idx("Water")].density),0.,0.)
    site=build_site(Float64(metadata["primary_altitude_m"]))
    em=LVDTopo.build_elevation_map();vol=create_voxel_volume(em,mat;half_width_km=5.,surface_step_km=1.,nz=6)
    nc=num_cells(vol);nc==600 || error("expected 600 cells")
    field=readrows(joinpath(dat,"lvd_fit_field.csv"));length(field)==nc || error("field dimension")
    w=zeros(nc);support=falses(nc)
    for (c,r) in enumerate(field)
        Int(r["cell_idx"])==c || error("cell identity")
        for (axis,key) in enumerate(("x_m","y_m","z_m"))
            isapprox(Float64(r[key]),vol.centroids[axis,c];atol=1e-8,rtol=0) || error("geometry identity")
        end
        w[c]=Float64(r["w"]);isfinite(w[c]) && 0<=w[c]<=MAX_WATER_FRACTION || error("field box")
        string(r["aquifer_mask"]) in ("true","false") || error("support encoding")
        support[c]=string(r["aquifer_mask"])=="true"
    end
    all(iszero,w[.!support]) || error("field outside support")
    count(support)==147 && bytes2hex(sha256(join(findall(support),',')))==cfg["support_sha256"] || error("support identity")
    masks=Dict(Int(r["bin_idx"])=>r for r in readrows(joinpath(dat,"lvd_fit_mask.csv")))
    predicted=readrows(joinpath(dat,"lvd_fit_predictions.csv"))
    coverage=Dict(Int(r["bin_idx"])=>r for r in readrows(joinpath(dat,"lvd_mc_coverage.csv")))
    ids=Int[r["bin_idx"] for r in predicted];length(ids)==1003 && length(unique(ids))==1003 || error("covered dimensions")
    Set(ids)==Set(keys(coverage)) || error("coverage bin identity")
    paths=DirectionalPath[];fit=Bool[];mc=Bool[]
    for r in predicted
        id=Int(r["bin_idx"]);m=masks[id];v=coverage[id]
        string(m["covered"])=="true" || error("uncovered prediction")
        p=trace_path(vol,Float64(m["zenith_deg"]),Float64(m["azimuth_deg"]),em)
        p.valid && isapprox(p.surface_distance,Float64(v["slant_m"]);rtol=1e-12,atol=1e-8) || error("ray geometry")
        Float64(m["fit_mask"])==Float64(v["fit_mask"])==Float64(r["fit_mask"]) || error("mask mismatch")
        push!(paths,p);push!(fit,Float64(r["fit_mask"])>0);push!(mc,string(v["selected_mc"])=="true")
    end
    count(fit)==690 && count(mc)==630 && all(fit[mc]) && count(fit .& .!mc)==60 || error("subset dimensions")
    bytes2hex(sha256(join(ids[fit],',')))==cfg["screen_sha256"] || error("screen identity")
    shallow=falses(nc);es=csda_energy_quadrature(32,1e-3,1e9)
    mkpath(out)
    open(io->TOML.print(io,Dict("input_sha256"=>hashes,"started_utc"=>string(now(UTC)),"threads"=>Threads.nthreads(),"backend"=>string(get_adjoint_derivative_mode()),"scope"=>"Deterministic final-field sensitivity of selected/omitted stochastic-closure rays; no additional MC, accuracy or calibrated uncertainty claim")),joinpath(out,"configuration.toml"),"w")
    # Gate the scalar forward path first, independently of Jacobian assembly.
    flux=[compute_directional_flux_csda(ph,shallow,mat,site,p,w,es) for p in paths]
    saved=Float64[r["matched_csda_flux"] for r in predicted]
    dry=zeros(nc)
    dry_flux=[compute_directional_flux_csda(ph,shallow,mat,site,p,dry,es) for p in paths]
    saved_dry=Float64[r["dry_csda_flux"] for r in predicted]
    isequal(reinterpret(UInt64,dry_flux),reinterpret(UInt64,saved_dry)) || error("saved dry prediction bitwise parity failed")
    all(isfinite,flux) && all(>(0),flux) || error("nonfinite forward")
    isequal(reinterpret(UInt64,flux),reinterpret(UInt64,saved)) || error("saved prediction bitwise parity failed")
    println("Prediction parity PASS: 1003 saved final-field predictions exact");flush(stdout)
    jf,J=assemble_forward_and_jacobian(ph,shallow,mat,site,paths,w,es;n_cells=nc,threaded=true)
    all(isfinite,nonzeros(J)) && size(J)==(1003,600) || error("Jacobian dimension/finite gate")
    isequal(reinterpret(UInt64,jf),reinterpret(UInt64,saved)) || error("Jacobian forward parity failed")
    total_abs=vec(sum(abs.(J);dims=2));supported_abs=vec(sum(abs.(J[:,support]);dims=2))
    rows=NamedTuple[]
    for (i,p) in enumerate(paths)
        old=coverage[ids[i]];crossed=unique(s.cell_idx for s in p.segments if support[s.cell_idx])
        push!(rows,(bin_idx=ids[i],zenith_deg=p.zenith,azimuth_deg=p.azimuth,slant_m=p.surface_distance,fit_mask=fit[i],selected_mc=mc[i],fit_omitted=fit[i]&&!mc[i],support_crossed_cells=length(crossed),final_flux=flux[i],final_all_abs_jacobian=total_abs[i],final_support_abs_jacobian=supported_abs[i],final_support_abs_logflux_sensitivity=supported_abs[i]/flux[i],dry_reference_flux=Float64(old["dry_reference_flux"]),dry_support_abs_jacobian=Float64(old["dry_aquifer_abs_sensitivity"]),dry_support_abs_logflux_sensitivity=Float64(old["dry_logflux_aquifer_abs_sensitivity"])))
    end
    csv_rows(joinpath(out,"final_field_coverage.csv"),rows)
    ri,ci,vi=findnz(J)
    csv_rows(joinpath(out,"final_jacobian_sparse.csv"),[(bin_idx=ids[i],cell_idx=c,derivative=v,in_support=support[c]) for (i,c,v) in zip(ri,ci,vi)])
    groups=Dict("covered"=>trues(length(ids)),"fitted"=>fit,"mc_checked"=>mc,"fit_omitted"=>fit .& .!mc,"excluded_from_fit"=>.!fit)
    low=Bool[p.zenith<60 for p in paths]
    for (label,mask) in collect(groups), (suffix,angular) in (("below60",low),("at_or_above60",.!low))
        groups[label*"_"*suffix]=mask .& angular
    end
    cellrows=NamedTuple[]
    for c in 1:nc
        deriv=abs.(Vector(J[:,c]));logderiv=deriv./flux
        push!(cellrows,(cell_idx=c,x_m=vol.centroids[1,c],y_m=vol.centroids[2,c],z_m=vol.centroids[3,c],in_support=support[c],final_water=w[c],covered_abs_jacobian=sum(deriv),fitted_abs_jacobian=sum(deriv[fit]),mc_checked_abs_jacobian=sum(deriv[mc]),fit_omitted_abs_jacobian=sum(deriv[fit .& .!mc]),covered_abs_logflux_sensitivity=sum(logderiv),fitted_abs_logflux_sensitivity=sum(logderiv[fit]),mc_checked_abs_logflux_sensitivity=sum(logderiv[mc]),fit_omitted_abs_logflux_sensitivity=sum(logderiv[fit .& .!mc]),fitted_has_information=sum(deriv[fit])>0,mc_fraction_fitted_abs_jacobian=sum(deriv[fit])>0 ? sum(deriv[mc])/sum(deriv[fit]) : 0.0))
    end
    csv_rows(joinpath(out,"final_field_cell_coverage.csv"),cellrows)
    stats(v)=Dict("min"=>minimum(v),"q10"=>quantile(v,.1),"median"=>median(v),"q90"=>quantile(v,.9),"max"=>maximum(v),"sum"=>sum(v))
    summary=Dict{String,Any}()
    for (label,mask) in groups
        subset=rows[mask]
        isempty(subset) && (summary[label]=Dict("count"=>0,"information_status"=>"empty_group");continue)
        crossed=Set(s.cell_idx for p in paths[mask] for s in p.segments if support[s.cell_idx])
        summary[label]=Dict("count"=>length(subset),"zenith_deg"=>stats([r.zenith_deg for r in subset]),"slant_m"=>stats([r.slant_m for r in subset]),"support_cells_intersected"=>sort(collect(crossed)),"final_support_abs_jacobian"=>stats([r.final_support_abs_jacobian for r in subset]),"final_support_abs_logflux_sensitivity"=>stats([r.final_support_abs_logflux_sensitivity for r in subset]),"dry_support_abs_jacobian"=>stats([r.dry_support_abs_jacobian for r in subset]),"dry_support_abs_logflux_sensitivity"=>stats([r.dry_support_abs_logflux_sensitivity for r in subset]))
        for (name,v) in (("all_abs_jacobian",total_abs),("support_abs_jacobian",supported_abs),("support_abs_logflux_sensitivity",supported_abs./flux))
            summary[label]["covered_"*name*"_denominator_status"]=sum(v)>0 ? "positive_information" : "no_information"
            summary[label]["fitted_"*name*"_denominator_status"]=sum(v[fit])>0 ? "positive_information" : "no_information"
            summary[label]["fraction_covered_final_"*name]=sum(v)>0 ? sum(v[mask])/sum(v) : 0.0
            summary[label]["fraction_fitted_final_"*name]=sum(v[fit])>0 ? sum(v[mask .& fit])/sum(v[fit]) : 0.0
        end
    end
    # Preregistered deterministic FD subset: omitted rays nearest zenith 10/50/90 percentiles,
    # choosing their largest supported derivative. Zero-support cases are explicit.
    fd=NamedTuple[];omitted=findall(fit .& .!mc)
    for theta in quantile([paths[i].zenith for i in omitted],[.1,.5,.9])
        i=omitted[argmin(abs.([paths[j].zenith-theta for j in omitted]))]
        active=findall(support);c=active[argmax(abs.(Vector(J[i,active])))];ad=J[i,c]
        for h in (1e-4,5e-5)
            f(x)=begin v=copy(w);v[c]=x;compute_directional_flux_csda(ph,shallow,mat,site,paths[i],v,es) end
            x=w[c];deriv= x>=h && x+h<=MAX_WATER_FRACTION ? (f(x+h)-f(x-h))/(2h) : x<h ? (-3f(x)+4f(x+h)-f(x+2h))/(2h) : (3f(x)-4f(x-h)+f(x-2h))/(2h)
            push!(fd,(bin_idx=ids[i],cell_idx=c,step=h,ad_derivative=ad,fd_derivative=deriv,absolute_error=abs(ad-deriv),relative_error=abs(ad-deriv)/max(abs(ad),abs(deriv),1e-30),no_supported_sensitivity=iszero(ad)))
        end
    end
    csv_rows(joinpath(out,"omitted_component_fd.csv"),fd)
    changed=[p for (p,h) in hashes if bytes2hex(sha256(read(p)))!=h]
    summary["audit"]=Dict("input_sha256"=>hashes,"changed_inputs"=>changed,"prediction_bitwise_parity"=>true,"dry_prediction_bitwise_parity"=>true,"jacobian_forward_bitwise_parity"=>true,"jacobian_rows"=>size(J,1),"jacobian_columns"=>size(J,2),"jacobian_nonzeros"=>nnz(J),"scope"=>"Absolute derivative sums are unweighted coverage diagnostics, not resolution, Fisher information, solid-angle/acceptance coverage, new stochastic validation, or uncertainty bounds. Dry columns remain explicitly dry-reference; final columns use the recovered field and frozen material correction.")
    open(io->TOML.print(io,summary),joinpath(out,"coverage_summary.toml"),"w")
    isempty(changed) || error("input changed during evaluation")
    println("Final-field deterministic coverage diagnostic complete; no additional stochastic validation")
end
if abspath(PROGRAM_FILE)==@__FILE__;final_coverage(final_coverage_options(ARGS));end
