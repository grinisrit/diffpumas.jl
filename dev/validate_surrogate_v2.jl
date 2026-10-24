#!/usr/bin/env julia
# Frozen training/test split; no coefficient fitting uses test rows.
include(joinpath(@__DIR__, "..", "examples", "revision_suite.jl"))
using DelimitedFiles

function readrows(path)
    a,h=readdlm(path, ',', Any;header=true)
    [Dict(string(h[j])=>a[i,j] for j in axes(a,2)) for i in axes(a,1)]
end
function install_dry(path)
    c=TOML.parsefile(path)
    set_csda_correction!(; (Symbol(k)=>v for (k,v) in c)...)
end
function v2state(dump)
    ph=load_physics(dump); ph===nothing && error("invalid physics dump")
    idx(n)=get_material_index(ph,n)
    mat=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),idx("PorousWetRock"),2710.,Float64(ph.tables[idx("Water")].density),0.,0.)
    em=LVDTopo.build_elevation_map()
    vol=create_voxel_volume(em,mat;half_width_km=3.,surface_step_km=1.5,nz=8)
    (ph,mat,build_site(30000.),em,vol,falses(num_cells(vol)),csda_energy_quadrature(32,1e-3,1e9))
end

function validation_v2(argv=ARGS)
    if !isempty(argv) && startswith(argv[1], "--")
        iseven(length(argv)) || error("expected option/value pairs")
        opts=Dict(argv[i]=>argv[i+1] for i in 1:2:length(argv))
        mode=get(opts,"--mode","pilot"); mode in ("pilot","production") || error("invalid mode")
        argv=[opts["--output-dir"],opts["--dump"],mode=="pilot" ? "128" : "1024",mode=="pilot" ? "4" : "8",mode=="pilot" ? "2026102401" : "2027102401"]
    end
    length(argv)==5 || error("usage: OUTPUT DUMP MC_SAMPLES REPLICATES SEED_BLOCK")
    out,dump=abspath.(argv[1:2]); nmc,nrep=parse.(Int,argv[3:4]); seedbase=parse(Int,argv[5]); nrep>=2 || error("need >=2 independent replicates")
    ntrain=nmc==128 ? 128 : 8192
    isdir(out) && !issubset(Set(readdir(out)),Set(["execution.json","run.log","pid.json"])) && error("output already has evidence; preserve immutable evidence")
    mkpath(out)
    # Write before evaluating any stochastic result. Pilot is also held out and
    # cannot be used for retuning coefficients before claiming this split final.
    plan=Dict("seed"=>seedbase,"training_zeniths"=>[0.,15.,30.,45.],"training_azimuths"=>[7.5,127.5,247.5],
      "test_zeniths"=>[15.,45.,60.,75.],"test_azimuths"=>[37.5,157.5,277.5],"training_uniform_fields"=>[0.2,0.6],
      "test_fields"=>["uniform_0.35","spatial_sine","spatial_cosine"],"ridge"=>1e-6,"mc_samples"=>nmc,"training_mc_samples"=>ntrain,"replicates"=>nrep,
      "primary_altitude_m"=>30000.,"rock_density"=>2710.,"csda_nodes"=>32,"energy_min"=>1e-3,"energy_max"=>1e9,
      "dry_calibration"=>"unchanged corrected_validation/calibration.toml","scope"=>"fixed straight path, scattering disabled; 30km source boundary held fixed and unresolved",
      "flux_acceptance"=>"median absolute relative residual improves >=30%; maximum absolute residual does not increase",
      "derivative_acceptance"=>"AD/FD relative error <1e-4; paired-energy derivative median absolute bias and count >2 estimated SE must not worsen beyond replicate MC precision",
      "inference_scope"=>"conditional replicate standard errors; no coefficient uncertainty or calibrated confidence coverage; no retuning on test",
      "driver_sha256"=>bytes2hex(sha256(read(@__FILE__))),"source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"..","src","Tomography.jl")))),
      "physics_dump_sha256"=>bytes2hex(sha256(read(dump))))
    open(io->TOML.print(io,plan),joinpath(out,"preregistered_plan.toml"),"w")
    ph,mat,site,em,vol,shallow,es=v2state(dump); nc=num_cells(vol)
    dry=zeros(nc); install_dry(joinpath(@__DIR__,"..","docs","revision_evidence","corrected_validation","calibration.toml"))
    set_csda_material_correction!(enabled=false)
    train=filter(p->p.valid,precompute_paths(vol,em,plan["training_zeniths"],plan["training_azimuths"]))
    test=filter(p->p.valid,precompute_paths(vol,em,plan["test_zeniths"],plan["test_azimuths"]))
    fields=[fill(.2,nc),fill(.6,nc)]
    function mc(path,w,sd;nsamples=nmc)
        mats,dens=build_cell_properties_for_mc(shallow,w,mat)
        en=sample_energy_set(nsamples,1e-3,1e9,sd)
        compute_directional_flux_mc(ph,mat,site,path,mats,dens,en,sd+1;straggling=true,scattering=false,energy_threshold_low=100.)[1]
    end
    tr=NamedTuple[]; md=zeros(length(train)); mw=zeros(2,length(train))
    for (b,p) in enumerate(train)
        sd=seedbase+10000b
        md[b]=mc(p,dry,sd;nsamples=ntrain)
        for k in 1:2
            mw[k,b]=mc(p,fields[k],sd;nsamples=ntrain)
            push!(tr,(bin=b,zenith=p.zenith,azimuth=p.azimuth,field=k,seed=sd,mc_dry=md[b],mc_wet=mw[k,b]))
        end
        csv_rows(joinpath(out,"training.csv"),tr)
    end
    fit=calibrate_csda_material_correction(ph,shallow,mat,site,train,dry,fields,es,md,mw,collect(eachindex(train));ridge=1e-6)
    save_csda_corrections(joinpath(out,"frozen_corrections.toml");metadata=merge(plan,Dict("training_q_min"=>fit.q_min,"training_q_max"=>fit.q_max,"training_design_condition_number"=>fit.condition_number,"training_design_rank"=>2)))
    coeff=get_csda_material_correction(); linear,quadratic=coeff.linear,coeff.quadratic
    open(io->TOML.print(io,Dict("linear"=>linear,"quadratic"=>quadratic,"ridge"=>1e-6,"fit"=>string(fit))),joinpath(out,"material_calibration.toml"),"w")
    testfields=[fill(.35,nc),[.15+.4*sin(vol.centroids[1,c]/900+vol.centroids[3,c]/600)^2 for c in 1:nc],
      [.15+.4*cos(vol.centroids[1,c]/900+vol.centroids[3,c]/600)^2 for c in 1:nc]]
    rows=NamedTuple[]; derivatives=NamedTuple[]
    for (b,p) in enumerate(test), (k,w) in enumerate(testfields)
        set_csda_material_correction!(enabled=false)
        old=compute_directional_flux_csda(ph,shallow,mat,site,p,w,es)
        set_csda_material_correction!(linear=linear,quadratic=quadratic)
        new=compute_directional_flux_csda(ph,shallow,mat,site,p,w,es)
        q=DiffPumas.Tomography._column_water_fraction(mat,p.remaining_rock_distance,[s.distance for s in p.segments],[s.cell_idx for s in p.segments],[shallow[s.cell_idx] for s in p.segments],w)
        _,J=assemble_forward_and_jacobian(ph,shallow,mat,site,[p],w,es;n_cells=nc,threaded=false)
        _,cells,vals=findnz(J); isempty(cells) && error("test path lacks sensitivities")
        c=cells[argmax(abs.(vals))]; delta=.05; wl=copy(w);wh=copy(w);wl[c]-=delta;wh[c]+=delta
        epsfd=1e-6; wa=copy(w);wb=copy(w);wa[c]-=epsfd;wb[c]+=epsfd
        fd=(compute_directional_flux_csda(ph,shallow,mat,site,p,wb,es)-compute_directional_flux_csda(ph,shallow,mat,site,p,wa,es))/(2epsfd)
        mcvalues=revision_replicates(nrep;parallel=true) do rep
            sd=seedbase+1000000+10000b+100rep
            (mc(p,w,sd),mc(p,wl,sd),mc(p,wh,sd))
        end
        for rep in 1:nrep
            sd=seedbase+1000000+10000b+100rep # shared fields retain covariance; not training streams
            truth=mcvalues[rep][1]
            push!(rows,(bin=b,zenith=p.zenith,azimuth=p.azimuth,field=k,column_water_fraction=q,replicate=rep,seed=sd,before=old,after=new,mc=truth))
            # CRN low/high transport plus matched-energy CSDA isolates derivative discrepancy.
            en=sample_energy_set(nmc,1e-3,1e9,sd)
            ml=mcvalues[rep][2];mh=mcvalues[rep][3]
            set_csda_material_correction!(enabled=false)
            csold=(compute_directional_flux_csda(ph,shallow,mat,site,p,wh,en)-compute_directional_flux_csda(ph,shallow,mat,site,p,wl,en))/(2delta)
            set_csda_material_correction!(linear=linear,quadratic=quadratic)
            csnew=(compute_directional_flux_csda(ph,shallow,mat,site,p,wh,en)-compute_directional_flux_csda(ph,shallow,mat,site,p,wl,en))/(2delta)
            push!(derivatives,(bin=b,zenith=p.zenith,azimuth=p.azimuth,field=k,cell=c,replicate=rep,seed=sd,ad=J[1,c],fd=fd,interval=2delta,mc_secant=(mh-ml)/(2delta),before=csold,after=csnew))
            csv_rows(joinpath(out,"heldout_flux.csv"),rows);csv_rows(joinpath(out,"heldout_derivatives.csv"),derivatives)
        end
    end
    println("Validation complete: ",out)
end
if abspath(PROGRAM_FILE)==@__FILE__;validation_v2();end
