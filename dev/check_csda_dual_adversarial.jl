#!/usr/bin/env julia
# Read-only deterministic regression against the saved branch-crossing case.
# Does not load prototype methods or run reconstruction / stochastic histories.
include(joinpath(@__DIR__,"..","examples","lvd_tomography.jl"))
using DelimitedFiles, SHA, TOML, ChainRulesCore

function dual_adversarial_check()
    raw=copy(ARGS);empty!(ARGS);baseline="";reference=""
    i=1
    while i<=length(raw)
        if raw[i] in ("--baseline-results","--reference-result")
            i<length(raw) || error("missing helper argument")
            if raw[i]=="--baseline-results";baseline=abspath(raw[i+1]);else;reference=abspath(raw[i+1]);end
            i+=2
        else
            push!(ARGS,raw[i]);i+=1
        end
    end
    isempty(baseline) && error("--baseline-results required")
    isempty(reference) && error("--reference-result required")
    args=parse_commandline()
    isdir(args.output_dir) && error("output must be fresh")
    args.geometry=="grid" || error("saved case requires grid geometry")
    ref=TOML.parsefile(reference)
    fieldpath=joinpath(baseline,"lvd_fit_field.csv")
    digest(p)=bytes2hex(sha256(read(p)))
    digest(fieldpath)==ref["field_sha256"] || error("saved field hash differs")
    digest(args.dump_path)==ref["physics_sha256"] || error("physics hash differs")
    data,header=readdlm(fieldpath,',',Any;header=true)
    field=Float64.(data[:,findfirst(==("w"),vec(header))])
    field[275]==ref["water_fraction"] || error("saved cell value differs")
    ph=load_tomography_physics(args.dump_path);idx(name)=get_material_index(ph,name)
    porous=idx("PorousWetRock")
    mat=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),porous,
        args.rock_density>0 ? args.rock_density : Float64(ph.tables[idx("StandardRock")].density),
        Float64(ph.tables[idx("Water")].density),porous>0 ? Float64(ph.tables[porous].density) : 0.,args.porous_top_thickness)
    load_csda_corrections!(args.correction_artifact)
    emap=LVDTopo.build_elevation_map();volume=create_sensitivity_volume(emap,mat,args)
    num_cells(volume)==length(field)==600 || error("saved grid differs")
    shallow=Bool[is_shallow_cell(volume,c) for c in eachindex(field)]
    site=build_site(args.primary_altitude_km*1000)
    bins=load_measured_intensity(args.match_data);bin=bins[916]
    zen=.5*(bin.zen_lo+bin.zen_hi);az=.5*(bin.az_lo+bin.az_hi)
    path=trace_path(volume,zen,az,emap)
    energies=csda_energy_quadrature(args.n_samples,args.energy_min,args.energy_max)
    oldmode=get_adjoint_derivative_mode();oldstep=get_adjoint_fd_step()
    get_adjoint_derivative_mode()==:forwarddiff || error("library default is not forwarddiff")
    TM=DiffPumas.Tomography
    result=Dict{String,Any}()
    try
        flux,ids,jac=directional_flux_and_grad_csda(ph,shallow,mat,site,path,field,energies)
        tangent=jac[findfirst(==(275),ids)]
        flux==ref["flux_before"] || error("Float64 primal changed")
        differences=Dict{String,Float64}()
        for h in (1e-7,1e-8)
            hi=copy(field);lo=copy(field);hi[275]+=h;lo[275]-=h
            f(x)=compute_directional_flux_csda(ph,shallow,mat,site,path,x,energies)
            differences[string(h)]=(f(hi)-f(lo))/(2h)
            isapprox(tangent,differences[string(h)];rtol=2e-6) || error("applied-library tangent disagrees with whole flux FD")
        end
        set_adjoint_derivative_mode!(:finite_difference);set_adjoint_fd_step!(1e-6)
        legacyflux,legacyids,legacyjac=directional_flux_and_grad_csda(ph,shallow,mat,site,path,field,energies)
        legacy=legacyjac[findfirst(==(275),legacyids)]
        legacyflux==flux || error("adjoint mode changed primal")
        isapprox(legacy,ref["default_adjoint"];rtol=1e-10) || error("legacy backend no longer reproduces saved numerical adjoint")
        set_adjoint_derivative_mode!(:forwarddiff)
        _,pb=ChainRulesCore.rrule(TM.propagate_csda_cell_segment,ph,.2,false,mat,30.,123.4567,0.)
        zero_weight=pb((1.,1.,NoTangent()))
        zero_weight[7]!=0 && zero_weight[8]>0 || error("valid zero-weight cotangents lost")
        result=Dict("scope"=>"applied-library deterministic branch-crossing regression; no prototype overloads, no stochastic transport",
            "passed"=>true,"bin"=>916,"cell"=>275,"flux"=>flux,"bitwise_primal_match"=>true,
            "forwarddiff_adjoint"=>tangent,"legacy_adjoint"=>legacy,"whole_flux_fd"=>differences,
            "zero_weight_energy_cotangent"=>zero_weight[7],"zero_weight_input_weight_cotangent"=>zero_weight[8],
            "field_sha256"=>digest(fieldpath),"physics_sha256"=>digest(args.dump_path),
            "reference_sha256"=>digest(reference),"helper_sha256"=>digest(@__FILE__),
            "source_sha256"=>Dict(p=>digest(joinpath(@__DIR__,"..",p)) for p in
                ("Project.toml","src/Physics.jl","src/Geometry.jl","src/Tomography.jl","src/Tomography_rules.jl")))
    finally
        set_adjoint_fd_step!(oldstep);set_adjoint_derivative_mode!(oldmode)
    end
    mkpath(args.output_dir)
    open(io->TOML.print(io,result),joinpath(args.output_dir,"dual_adversarial_check.toml"),"w")
    println("Applied-library adversarial check passed: ",args.output_dir)
end

if abspath(PROGRAM_FILE)==@__FILE__;dual_adversarial_check();end
