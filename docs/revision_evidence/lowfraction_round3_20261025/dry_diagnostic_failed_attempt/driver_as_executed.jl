#!/usr/bin/env julia
# Isolated experiment. No global source/production coefficient files are changed.
include(joinpath(@__DIR__, "validate_surrogate_v2.jl"))
const ROOT3=normpath(joinpath(@__DIR__,".."))
const PLAN3=joinpath(ROOT3,"docs/revision_evidence/lowfraction_round3_20261025/plan.toml")
function main3(args=ARGS)
    length(args)==3 || error("usage: diagnose|pilot OUTPUT PHYSICS_DUMP")
    mode,out,dump=args; mode in ("diagnose","pilot") || error("invalid mode")
    mode=="pilot" && get(ENV,"LOWFRACTION_CONVERGENCE_GATE","")!="independently-validated-root-go" && error("convergence gate not released")
    ispath(out) && !isempty(readdir(out)) && error("immutable output directory must be empty")
    mkpath(out); cfg=TOML.parsefile(PLAN3)
    ph,mat,site,em,vol,shallow,es=v2state(dump); nc=num_cells(vol); dry=zeros(nc)
    install_dry(joinpath(ROOT3,"docs/revision_evidence/corrected_validation/calibration.toml"))
    set_csda_material_correction!(enabled=false)
    audit=NamedTuple[]
    for p in precompute_paths(vol,em,[0.,15.,30.,45.,60.,75.],[37.5,157.5,277.5])
        L=DiffPumas.Tomography.path_rock_length(p)
        for nodes in (32,64,128,256)
            f=p.valid ? compute_directional_flux_csda(ph,shallow,mat,site,p,dry,csda_energy_quadrature(nodes,1e-3,1e9)) : NaN
            push!(audit,(valid=p.valid,zenith=p.zenith,azimuth=p.azimuth,slant_m=L,outside_training_angle=p.zenith>45,outside_training_slant=!(1193.1299812311227<=L<=2297.328362524513),nodes=nodes,flux=f,source_altitude_m=site.primary_altitude))
        end
    end
    csv_rows(joinpath(out,"dry_quadrature_domain.csv"),audit)
    mode=="diagnose" && return
    provenance=Dict("included_sha256"=>Dict(f=>bytes2hex(sha256(read(joinpath(ROOT3,f)))) for f in ["dev/validate_surrogate_v2.jl","examples/revision_suite.jl","dev/lowfraction_round3_summary.py","docs/revision_evidence/corrected_validation/calibration.toml"]),"driver_sha256"=>bytes2hex(sha256(read(@__FILE__))),"plan_sha256"=>bytes2hex(sha256(read(PLAN3))),"dump_sha256"=>bytes2hex(sha256(read(dump))),"source_sha256"=>Dict(f=>bytes2hex(sha256(read(joinpath(ROOT3,"src",f)))) for f in readdir(joinpath(ROOT3,"src")) if endswith(f,".jl")),"control_sha256"=>bytes2hex(sha256(read(joinpath(ROOT3,"docs/revision_evidence/surrogate_v2/production/material_calibration.toml")))))
    open(io->TOML.print(io,provenance),joinpath(out,"predraw_provenance.toml"),"w")
    cp(PLAN3,joinpath(out,"preregistered_plan.toml"))
    train=precompute_paths(vol,em,cfg["train_zenith"],cfg["train_azimuth"])
    test=precompute_paths(vol,em,cfg["test_zenith"],cfg["test_azimuth"])
    all(p->p.valid,train) && all(p->p.valid,test) || error("invalid planned ray: unresolved; no replacement")
    function mc(p,w,seed,n)
        mats,dens=build_cell_properties_for_mc(shallow,w,mat)
        en=sample_energy_set(n,1e-3,1e9,seed)
        compute_directional_flux_mc(ph,mat,site,p,mats,dens,en,seed+1;straggling=true,scattering=false,energy_threshold_low=100.)[1]
    end
    fields=[fill(w,nc) for w in cfg["train_uniform"]]; md=zeros(length(train));mw=zeros(length(fields),length(train)); rows=NamedTuple[]
    training_refs=revision_replicates(length(train);parallel=true) do b
        p=train[b];seed=cfg["train_seed"]+10000b
        (mc(p,dry,seed,cfg["ntrain"]),[mc(p,w,seed,cfg["ntrain"]) for w in fields])
    end
    for b in eachindex(train)
        seed=cfg["train_seed"]+10000b;md[b]=training_refs[b][1]
        for k in eachindex(fields)
            mw[k,b]=training_refs[b][2][k]
            push!(rows,(bin=b,field=k,seed=seed,dry=md[b],wet=mw[k,b]))
        end
    end
    csv_rows(joinpath(out,"training.csv"),rows)
    all(x->isfinite(x)&&x>0,md) && all(x->isfinite(x)&&x>0,mw) || error("nonpositive training reference: unresolved")
    fit=calibrate_csda_material_correction(ph,shallow,mat,site,train,dry,fields,es,md,mw,collect(eachindex(train));ridge=1e-6)
    open(io->TOML.print(io,Dict(string(k)=>v for (k,v) in pairs(fit))),joinpath(out,"training_fit_structured.toml"),"w")
    cand=get_csda_material_correction(); beta=(cand.linear,cand.quadratic)
    save_csda_corrections(joinpath(out,"candidate_frozen_before_validation.toml");metadata=Dict("fit"=>string(fit),"plan_sha256"=>provenance["plan_sha256"]))
    control=TOML.parsefile(joinpath(ROOT3,"docs/revision_evidence/surrogate_v2/production/material_calibration.toml"))
    arms=[("control",control["linear"],control["quadratic"]),("candidate",beta...)]
    testfields=[fill(.04,nc),fill(.11,nc),[.015+.12*sin(vol.centroids[1,c]/1130+vol.centroids[3,c]/710+.37)^2 for c in 1:nc],[vol.centroids[3,c]>650 && hypot(vol.centroids[1,c]-210,vol.centroids[2,c]+170)<1370 ? .14 : .01 for c in 1:nc]]
    fluxrows=NamedTuple[];derows=NamedTuple[];adrows=NamedTuple[]
    for (b,p) in enumerate(test), (k,w) in enumerate(testfields)
        # Coordinate fixed geometrically, independent of candidate Jacobian.
        cells=sort(unique([s.cell_idx for s in p.segments]));isempty(cells)&&error("no crossed cells: unresolved")
        c=cells[cld(length(cells),2)];lo=max(0.,w[c]-.01);hi=w[c]+.01;width=hi-lo
        wl=copy(w);wh=copy(w);wl[c]=lo;wh[c]=hi
        refs=revision_replicates(cfg["replicates"];parallel=true) do rep
            seed=cfg["test_seed"]+100000b+1000k+10rep
            (seed,mc(p,dry,seed,cfg["nmc"]),mc(p,w,seed,cfg["nmc"]),mc(p,wl,seed,cfg["nmc"]),mc(p,wh,seed,cfg["nmc"]))
        end
        for (arm,lin,quad) in arms
            set_csda_material_correction!(linear=lin,quadratic=quad)
            f=compute_directional_flux_csda(ph,shallow,mat,site,p,w,es);fdry=compute_directional_flux_csda(ph,shallow,mat,site,p,dry,es)
            _,J=assemble_forward_and_jacobian(ph,shallow,mat,site,[p],w,es;n_cells=nc,threaded=false)
            for h in (1e-4,1e-5,1e-6)
                wa=copy(w);wb=copy(w);wa[c]-=h;wb[c]+=h
                fd=(compute_directional_flux_csda(ph,shallow,mat,site,p,wb,es)-compute_directional_flux_csda(ph,shallow,mat,site,p,wa,es))/(2h)
                push!(adrows,(bin=b,field=k,arm=arm,cell=c,step=h,ad=J[1,c],fd=fd))
            end
            for (rep,(seed,m0,m,ml,mh)) in enumerate(refs)
                en=sample_energy_set(cfg["nmc"],1e-3,1e9,seed)
                cs=(compute_directional_flux_csda(ph,shallow,mat,site,p,wh,en)-compute_directional_flux_csda(ph,shallow,mat,site,p,wl,en))/width
                push!(fluxrows,(bin=b,zenith=p.zenith,azimuth=p.azimuth,field=k,arm=arm,replicate=rep,seed=seed,csda=f,csda_dry=fdry,mc=m,mc_dry=m0))
                push!(derows,(bin=b,field=k,arm=arm,replicate=rep,seed=seed,cell=c,lo=lo,hi=hi,csda_secant=cs,mc_secant=(mh-ml)/width,mc_lo=ml,mc_hi=mh))
            end
        end
        csv_rows(joinpath(out,"heldout_flux.csv"),fluxrows);csv_rows(joinpath(out,"heldout_derivatives.csv"),derows);csv_rows(joinpath(out,"ad_fd_steps.csv"),adrows)
    end
end
abspath(PROGRAM_FILE)==@__FILE__ && main3()
