#!/usr/bin/env julia
# Add only the 60 omitted fitted rays; original 630 histories remain untouched.
include(joinpath(@__DIR__,"validate_surrogate_v2.jl"))
using LinearAlgebra

function supplement_closure()
    pos=findfirst(==("--baseline-results"),ARGS);pos===nothing && error("baseline required")
    baseline=abspath(ARGS[pos+1]);deleteat!(ARGS,pos:pos+1)
    args=parse_commandline();out=args.output_dir
    ispath(out) && error("fresh data output required")
    args.geometry=="grid" && args.n_samples==32 && args.csda_quadrature=="midpoint" || error("original geometry/energy rule required")
    args.papermatch_mc_samples==8192 && args.seed==42 && args.straggling && args.energy_threshold_low==100. || error("original MC settings required")
    args.energy_min==1e-3 && args.energy_max==1e9 && args.primary_altitude_km==30. || error("original energy/source convention required")
    files=[joinpath(baseline,n) for n in ("lvd_fit_field.csv","lvd_fit_predictions.csv","lvd_fit_mask.csv","lvd_solver_configuration.toml","lvd_mc_coverage.csv","lvd_mc_common_subset.csv")]
    append!(files,[@__FILE__,joinpath(@__DIR__,"validate_surrogate_v2.jl"),joinpath(@__DIR__,"..","examples","revision_suite.jl"),joinpath(@__DIR__,"..","examples","lvd_tomography.jl"),args.dump_path,args.correction_artifact])
    append!(files,filter(p->endswith(p,".jl"),readdir(joinpath(@__DIR__,"..","src");join=true)))
    hashes=Dict(abspath(p)=>bytes2hex(sha256(read(p))) for p in files)
    cfg=TOML.parsefile(joinpath(baseline,"lvd_solver_configuration.toml"))
    cfg["correction_artifact_sha256"]==bytes2hex(sha256(read(args.correction_artifact))) || error("artifact identity")
    metadata=load_csda_corrections!(args.correction_artifact)
    metadata["physics_dump_sha256"]==bytes2hex(sha256(read(args.dump_path))) || error("dump identity")
    ph=load_tomography_physics(args.dump_path);idx(n)=get_material_index(ph,n)
    mat=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),idx("PorousWetRock"),args.rock_density,Float64(ph.tables[idx("Water")].density),0.,0.)
    args.porous_top_thickness==0. && args.rock_density==2710. || error("material settings")
    em=LVDTopo.build_elevation_map();vol=create_sensitivity_volume(em,mat,args)
    field=readrows(joinpath(baseline,"lvd_fit_field.csv"));num_cells(vol)==length(field)==600 || error("600-cell field required")
    w=zeros(600);support=falses(600)
    for (c,r) in enumerate(field)
        Int(r["cell_idx"])==c || error("cell order")
        isequal(Tuple(Float64(r[k]) for k in ("x_m","y_m","z_m")),cell_centroid(vol,c)) || error("cell coordinate identity")
        w[c]=Float64(r["w"]);isfinite(w[c]) && 0<=w[c]<=MAX_WATER_FRACTION || error("field box")
        string(r["aquifer_mask"]) in ("true","false") || error("support encoding")
        support[c]=string(r["aquifer_mask"])=="true"
    end
    all(iszero,w[.!support]) && count(support)==147 || error("support field")
    bytes2hex(sha256(join(findall(support),',')))==cfg["support_sha256"] || error("support identity")
    saved=readrows(joinpath(baseline,"lvd_fit_predictions.csv"));length(saved)==1003 || error("covered dimensions")
    ids=Int[r["bin_idx"] for r in saved];length(unique(ids))==1003 || error("duplicate predictions")
    fit=Bool[Float64(r["fit_mask"])>0 for r in saved];count(fit)==690 || error("fit dimensions")
    bytes2hex(sha256(join(ids[fit],',')))==cfg["screen_sha256"] || error("screen identity")
    coverage=Dict(Int(r["bin_idx"])=>r for r in readrows(joinpath(baseline,"lvd_mc_coverage.csv")))
    Set(keys(coverage))==Set(ids) || error("coverage identity")
    old=readrows(joinpath(baseline,"lvd_mc_common_subset.csv"));length(old)==630 || error("original closure dimensions")
    oldmap=Dict(Int(r["bin_idx"])=>r for r in old);length(oldmap)==630 || error("duplicate original closure")
    oldids=Set(keys(oldmap));newpos=findall(i->fit[i] && !(ids[i] in oldids),eachindex(ids))
    length(newpos)==60 && all(id->id in Set(ids[fit]),oldids) || error("strict complementary subsets")
    for i in eachindex(ids)
        (string(coverage[ids[i]]["selected_mc"])=="true")== (ids[i] in oldids) || error("selected MC identity")
    end
    paths=[trace_path(vol,Float64(coverage[id]["zenith_deg"]),Float64(coverage[id]["azimuth_deg"]),em) for id in ids]
    all(p->p.valid,paths) || error("invalid original ray")
    for (i,p) in enumerate(paths)
        isapprox(p.surface_distance,Float64(coverage[ids[i]]["slant_m"]);rtol=1e-12,atol=1e-8) || error("ray slant identity")
    end
    site=build_site(args.primary_altitude_km*1000);shallow=falses(600);es=csda_energy_quadrature(32,args.energy_min,args.energy_max)
    wet=[compute_directional_flux_csda(ph,shallow,mat,site,p,w,es) for p in paths]
    dry=[compute_directional_flux_csda(ph,shallow,mat,site,p,zeros(600),es) for p in paths]
    isequal(reinterpret(UInt64,wet),reinterpret(UInt64,Float64[r["matched_csda_flux"] for r in saved])) || error("final forward bitwise parity")
    isequal(reinterpret(UInt64,dry),reinterpret(UInt64,Float64[r["dry_csda_flux"] for r in saved])) || error("dry forward bitwise parity")
    scales=Float64[r["normalization"] for r in saved];all(==(first(scales)),scales) || error("nonuniform saved scale")
    scale=first(scales);obs=Float64[r["measured_flux"] for r in saved]
    for (i,id) in enumerate(ids)
        id in oldids || continue;r=oldmap[id]
        isequal((Float64(r["wet_csda_unscaled"]),Float64(r["dry_csda_unscaled"]),Float64(r["measured_raw"]),Float64(r["common_scale"])),(wet[i],dry[i],obs[i],scale)) || error("original closure/model parity")
    end
    mc_samples=sample_energy_set(8192,args.energy_min,args.energy_max,819)
    newseeds=[16_000_000+ids[i] for i in newpos];oldseeds=Set(900_042+i for i in eachindex(ids) if ids[i] in oldids)
    isempty(Base.intersect(Set(newseeds),oldseeds)) && length(unique(newseeds))==60 || error("transport seed collision")
    mkpath(out)
    plan=Dict("source_inputs_sha256"=>hashes,"new_bins"=>ids[newpos],"new_transport_seeds"=>newseeds,"histories_per_ray"=>8192,"energy_seed"=>819,"original_transport_seed_formula"=>"900042 + position in original 1003 covered predictions","new_transport_seed_formula"=>"16000000 + original bin_idx","shared_energy_proposals"=>true,"scattering"=>false,"straggling"=>true,"energy_threshold_low"=>100.,"original_checked_bins"=>630,"new_checked_bins"=>60,"combined_fit_bins"=>690,"total_covered_bins"=>1003,"scope"=>"Supplemental same-field fixed-path stochastic energy-loss closure; shared finite energy proposals induce cross-bin dependence. No optimizer rerun, independent water validation, angular scattering validation, or calibration/systematic uncertainty propagation.")
    open(io->TOML.print(io,plan),joinpath(out,"preregistered_plan.toml"),"w")
    csv_rows(joinpath(out,"energy_proposals.csv"),[(sample_index=i,energy=s.energy,charge=s.charge,weight=s.weight) for (i,s) in enumerate(mc_samples)])
    mats,dens=build_cell_properties_for_mc(shallow,w,mat);newflux=fill(NaN,60);newerr=fill(NaN,60)
    mkpath(joinpath(out,"per_bin"))
    println("Parity gates PASS; supplementing 60 omitted fitted directions, 8192 histories each");flush(stdout)
    Threads.@threads for j in eachindex(newpos)
        i=newpos[j]
        newflux[j],newerr[j]=compute_directional_flux_mc(ph,mat,site,paths[i],mats,dens,mc_samples,newseeds[j];straggling=true,energy_threshold_low=100.,scattering=false)
        open(io->TOML.print(io,Dict("bin_idx"=>ids[i],"wet_mc_unscaled"=>newflux[j],"wet_mc_sigma_unscaled"=>newerr[j],"samples"=>8192,"energy_seed"=>819,"transport_seed"=>newseeds[j])),joinpath(out,"per_bin","bin_"*string(ids[i])*".toml"),"w")
        println("Supplement complete bin=",ids[i]," flux=",newflux[j]," SE=",newerr[j]);flush(stdout)
    end
    all(x->isfinite(x)&&x>=0,newflux) && all(x->isfinite(x)&&x>=0,newerr) || error("invalid supplemental flux/error")
    csv_rows(joinpath(out,"supplemental_mc.csv"),[(bin_idx=ids[i],zenith_deg=paths[i].zenith,azimuth_deg=paths[i].azimuth,slant_m=paths[i].surface_distance,wet_mc_unscaled=newflux[j],wet_mc_sigma_unscaled=newerr[j],zero_mc_estimate=iszero(newflux[j]),samples=8192,energy_seed=819,transport_seed=newseeds[j],origin="supplemental_60") for (j,i) in enumerate(newpos)])
    newmap=Dict(ids[i]=>j for (j,i) in enumerate(newpos));combined=NamedTuple[]
    for i in findall(fit)
        id=ids[i];original=id in oldids
        f=original ? Float64(oldmap[id]["wet_mc_unscaled"]) : newflux[newmap[id]]
        se=original ? Float64(oldmap[id]["wet_mc_sigma_unscaled"]) : newerr[newmap[id]]
        push!(combined,(bin_idx=id,zenith_deg=paths[i].zenith,azimuth_deg=paths[i].azimuth,slant_m=paths[i].surface_distance,measured_raw=obs[i],dry_csda_unscaled=dry[i],wet_csda_unscaled=wet[i],wet_mc_unscaled=f,wet_mc_sigma_unscaled=se,zero_mc_estimate=iszero(f),common_scale=scale,origin=original ? "original_630" : "supplemental_60",samples=8192,energy_seed=819,transport_seed=original ? 900_042+i : 16_000_000+id))
    end
    length(combined)==690 && length(unique(r.bin_idx for r in combined))==690 || error("combined unique fitted dimensions")
    csv_rows(joinpath(out,"combined_690_mc.csv"),combined)
    results=NamedTuple[]
    for (group,rows) in (("all_fitted_690",combined),("original_630",filter(r->r.origin=="original_630",combined)),("supplemental_60",filter(r->r.origin=="supplemental_60",combined)),("fitted_below60",filter(r->r.zenith_deg<60,combined)),("fitted_at_or_above60",filter(r->r.zenith_deg>=60,combined)))
        y=[r.measured_raw for r in rows]
        for (model,key) in (("dry_csda",:dry_csda_unscaled),("wet_csda",:wet_csda_unscaled),("wet_mc",:wet_mc_unscaled))
            p=[getproperty(r,key) for r in rows];profile=exp(mean(log.(max.(y,1e-300)).-log.(max.(p,1e-300))))
            residual(s)=norm(p.-y./s)/max(norm(y./s),1e-30)
            push!(results,(group=group,model=model,bins=length(rows),common_scale=scale,log_profiled_subset_scale=profile,relative_residual_common_scale=residual(scale),relative_residual_log_profiled_scale=residual(profile)))
        end
    end
    csv_rows(joinpath(out,"combined_residual_summary.csv"),results)
    changed=[p for (p,h) in hashes if bytes2hex(sha256(read(p)))!=h]
    open(io->TOML.print(io,Dict("changed_inputs"=>changed,"original630_predictions_preserved"=>true,"dry_and_final_prediction_bitwise_parity"=>true,"combined_count"=>690,"new_count"=>60,"supplemental_zero_mc_estimates"=>count(iszero,newflux),"source_inputs_sha256"=>hashes,"relative_residual_definition"=>"norm(prediction - observation/scale)/norm(observation/scale)","subset_profile_definition"=>"exp(mean(log(max(observation,1e-300))-log(max(prediction,1e-300)))); zero MC estimates retained, never resampled to force positivity","scope"=>plan["scope"])),joinpath(out,"completion_audit.toml"),"w")
    isempty(changed) || error("changed inputs during supplemental closure")
end
if abspath(PROGRAM_FILE)==@__FILE__;supplement_closure();end
