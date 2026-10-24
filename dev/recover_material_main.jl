#!/usr/bin/env julia
# Standalone report/closure recovery; never imported by the frozen example.
using SHA, TOML, DelimitedFiles
const RECOVERY_ROOT=normpath(joinpath(@__DIR__,".."))
const RECOVERY_FAILED=ENV["DIFFPUMAS_RECOVERY_FAILED_DATA"]
const RECOVERY_ORIGINAL=joinpath(RECOVERY_ROOT,"examples","lvd_tomography.jl")
const RECOVERY_EXPECTED="dc4e25dce955c53eddb965f7b97bec2eae78703658e9d5ef530f1f7fddd676c5"
bytes2hex(sha256(read(RECOVERY_ORIGINAL)))==RECOVERY_EXPECTED || error("frozen example changed")
function recovery_rows(path)
    data,head=readdlm(path,',',Any;header=true)
    [Dict(string(head[j])=>data[i,j] for j in axes(data,2)) for i in axes(data,1)]
end
function recovery_equal(label,a,b)
    isequal(a,b) || error("recovery parity failure: "*label)
end
const recovery_original_text=read(RECOVERY_ORIGINAL,String)
const recovery_start="    for recur in 1:n_recur\n"
const recovery_stop="\n    open(joinpath(args.output_dir,\"lvd_recursion_status.txt\"),\"w\") do io"
count(recovery_start,recovery_original_text)==1 || error("recursion anchor changed")
const recovery_i=findfirst(recovery_start,recovery_original_text)
const recovery_j=findnext(recovery_stop,recovery_original_text,last(recovery_i)+1)
const recovery_replay=raw"""
    # Recovery only: restore the audited completed optimizer state, no solve.
    stable_fit && n_recur==1 || error("only frozen one-pass stable fit recovery allowed")
    cfg_saved=TOML.parsefile(joinpath(RECOVERY_FAILED,"lvd_solver_configuration.toml"))
    saved_field=recovery_rows(joinpath(RECOVERY_FAILED,"lvd_fit_field.csv"))
    length(saved_field)==n_cells || error("saved field size mismatch")
    w_matched=Float64[r["w"] for r in saved_field]
    for c in 1:n_cells
        r=saved_field[c]; x,y,z=cell_centroid(volume,c)
        recovery_equal("cell identity",Int(r["cell_idx"]),c)
        recovery_equal("cell coordinate",(Float64(r["x_m"]),Float64(r["y_m"]),Float64(r["z_m"])),(x,y,z))
        recovery_equal("support",string(r["aquifer_mask"]),string(aquifer_mask[c]))
    end
    recovery_equal("support hash",bytes2hex(sha256(join(findall(aquifer_mask),','))),cfg_saved["support_sha256"])
    recovery_equal("screen hash",bytes2hex(sha256(join(valid_bins[fit_idx],','))),cfg_saved["screen_sha256"])
    recovery_equal("artifact",bytes2hex(sha256(read(args.correction_artifact))),cfg_saved["correction_artifact_sha256"])
    recovery_equal("fixed lambda",args.papermatch_absolute_prior_weight,cfg_saved["absolute_prior_weight"])
    recovery_equal("backend",string(get_adjoint_derivative_mode()),cfg_saved["adjoint_backend"])
    robust_model,profile_scale,robust_predict=DiffPumas.Tomography.profiled_log_observation_operator(model_aq,meas_v;
        weights=fit_mask.*wdem,robust_scale=args.papermatch_log_scale,
        normalization=args.papermatch_abs_scale,predict_only=predict_aq,return_predict_only=true)
    opt_logscale=profile_scale
    sm_prior=SmoothnessPrior(cell_neighbors(volume),cfg_saved["absolute_prior_weight"])
    r_final,J_final=robust_model(w_matched)
    objective_final=.5dot(r_final,r_final)+DiffPumas.Tomography.prior_value(sm_prior,w_matched)
    active=findall(aquifer_mask)
    grad_final=transpose(J_final[:,active])*r_final
    grad_final .+= DiffPumas.Tomography.laplacian_gradient(sm_prior,w_matched)[active]
    projected_final=norm(w_matched[active].-clamp.(w_matched[active].-grad_final,0.,MAX_WATER_FRACTION),Inf)
    hist=recovery_rows(joinpath(RECOVERY_FAILED,"lvd_solver_history.csv"))
    recovery_equal("objective",objective_final,Float64(last(hist)["objective"]))
    recovery_equal("projected gradient",projected_final,Float64(last(hist)["projected_gradient"]))
    saved_prediction=recovery_rows(joinpath(RECOVERY_FAILED,"lvd_fit_predictions.csv"))
    prediction=flux_only_at(rho_fit,w_matched)
    recovery_equal("covered bins",valid_bins,Int[r["bin_idx"] for r in saved_prediction])
    recovery_equal("fit mask",fit_mask,Float64[r["fit_mask"] for r in saved_prediction])
    recovery_equal("dry flux",reinterpret(UInt64,flux0),reinterpret(UInt64,Float64[r["dry_csda_flux"] for r in saved_prediction]))
    recovery_equal("matched flux",reinterpret(UInt64,prediction),reinterpret(UInt64,Float64[r["matched_csda_flux"] for r in saved_prediction]))
    s=profile_scale(prediction)
    all(r->isequal(Float64(r["normalization"]),s),saved_prediction) || error("saved scale differs")
    recovery_equal("observations",meas_v,Float64[r["measured_flux"] for r in saved_prediction])
    keys_rec=(:pass,:dfield,:dcoef,:relative_residual,:log_std,:normalization,:mean_water,:kappa_strag,:kappa_hard,:resid_a,:resid_b,:resid_c,:resid_d)
    for row in recovery_rows(joinpath(RECOVERY_FAILED,"lvd_recursion_history.csv"))
        push!(recursion_rows,NamedTuple{keys_rec}(Tuple(key==:pass ? Int(row[string(key)]) : Float64(row[string(key)]) for key in keys_rec)))
    end
    w0_fit=NaN
    open(joinpath(args.output_dir,"recovery_parity.toml"),"w") do io
        TOML.print(io,Dict("passed"=>true,"field_restored_without_optimization"=>true,"prediction_bitwise"=>true,
            "objective_bitwise"=>true,"projected_gradient_bitwise"=>true,"objective"=>objective_final,
            "projected_gradient"=>projected_final,"normalization"=>s,"cells"=>n_cells,"covered_bins"=>length(valid_bins)))
    end
    println("Recovery parity PASS: saved field, masks, predictions, objective and projected gradient exact; no optimizer replay.")
    flush(stdout)
    get(ENV,"DIFFPUMAS_RECOVERY_PARITY_ONLY","0")=="1" && return (recovery_parity_only=true,)
"""
const recovery_text=recovery_original_text[1:first(recovery_i)-1]*recovery_replay*recovery_original_text[first(recovery_j):end]
# Tail is byte-identical: only the complete recursion block is replaced.
Base.include_string(Main,recovery_text,RECOVERY_ORIGINAL)
function recovery_main()
    args=parse_commandline()
    args.papermatch && !args.papermatch_fit_only || error("full paper-match recovery required")
    args.papermatch_fit_mode=="stable" || error("stable recovery required")
    mkpath(args.output_dir)
    open(joinpath(dirname(args.output_dir),"recovery_transformed_example.jl"),"w") do io;write(io,recovery_text);end
    # Exact original geometry/config initialization; no synthetic paths/MC/solvers.
    physics=load_tomography_physics(args.dump_path)
    rock_idx=get_material_index(physics,"StandardRock");water_idx=get_material_index(physics,"Water")
    air_idx=get_material_index(physics,"Air");porous_idx=get_material_index(physics,"PorousWetRock")
    matcfg=MaterialConfig(rock_idx,water_idx,air_idx,porous_idx,args.rock_density,
        Float64(physics.tables[water_idx].density),porous_idx>0 ? Float64(physics.tables[porous_idx].density) : 0.,args.porous_top_thickness)
    emap=LVDTopo.build_elevation_map();volume=create_sensitivity_volume(emap,matcfg,args)
    shallow_flags=Bool[is_shallow_cell(volume,i) for i in 1:num_cells(volume)]
    args.csda_quadrature=="midpoint" || error("frozen midpoint quadrature required")
    energy_samples=csda_energy_quadrature(args.n_samples,args.energy_min,args.energy_max)
    load_csda_corrections!(args.correction_artifact)
    run_paper_match(physics,shallow_flags,matcfg,volume,emap,DirectionalPath[],energy_samples,rock_idx,air_idx,args)
    return 0
end
exit(recovery_main())
