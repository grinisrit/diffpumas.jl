#!/usr/bin/env julia
# Deterministic replay on immutable 2026-10-23 MC observations. These previously
# inspected observations are a historical comparison, not the new held-out gate.
include(joinpath(@__DIR__,"validate_surrogate_v2.jl"))
function replay_v2(argv=ARGS)
    length(argv)==3 || error("usage: OUTPUT DUMP MATERIAL_CALIBRATION_TOML")
    out,dump,coeff=abspath.(argv);mkpath(out)
    oldroot=joinpath(@__DIR__,"..","docs","revision_evidence","corrected_validation")
    ph,mat,site,em,vol,shallow,es=v2state(dump);nc=num_cells(vol)
    paths=filter(p->p.valid,precompute_paths(vol,em,[0.,15.,30.,45.,60.,75.],exclusive_degree_grid(360.,15.)))
    install_dry(joinpath(oldroot,"calibration.toml"));c=TOML.parsefile(coeff)
    enable()=set_csda_material_correction!(linear=c["linear"],quadratic=c["quadratic"])
    derivatives=NamedTuple[];closure=NamedTuple[]
    for r in readrows(joinpath(oldroot,"mc_jacobian.csv"))
        b=Int(r["bin"]);cell=Int(r["cell"]);fraction=Float64(r["fraction"])
        w=fill(fraction,nc);wl=copy(w);wh=copy(w);wl[cell]=r["fraction_lo"];wh[cell]=r["fraction_hi"]
        en=sample_energy_set(1024,1e-3,1e9,Int(r["seed"]))
        enable()
        secant=(compute_directional_flux_csda(ph,shallow,mat,site,paths[b],wh,en)-compute_directional_flux_csda(ph,shallow,mat,site,paths[b],wl,en))/r["interval_width"]
        push!(derivatives,(case=Int(r["case"]),bin=b,cell=cell,fraction=fraction,replicate=Int(r["replicate"]),seed=Int(r["seed"]),mc_secant=r["mc_fd"],before=r["csda_secant_paired_energy"],after=secant))
        csv_rows(joinpath(out,"historical_derivatives.csv"),derivatives)
    end
    wet=[.15+.45*sin(vol.centroids[1,c]/700)^2*(vol.centroids[3,c]>350) for c in 1:nc]
    for r in readrows(joinpath(oldroot,"held_out_closure.csv"))
        b=Int(r["bin"]);enable()
        pred=compute_directional_flux_csda(ph,shallow,mat,site,paths[b],wet,es)
        push!(closure,(bin=b,zenith=r["zenith"],replicate=Int(r["replicate"]),seed=Int(r["seed"]),before=r["predicted_wet"],after=pred,mc=r["mc_wet"]))
    end
    csv_rows(joinpath(out,"historical_flux.csv"),closure)
    provenance=Dict("scope"=>"same immutable MC observations; already inspected historical field, not independent blind validation",
        "coefficient_sha256"=>bytes2hex(sha256(read(coeff))),
        "mc_reference_sha256"=>bytes2hex(sha256(read(joinpath(oldroot,"mc_jacobian.csv")))))
    open(joinpath(out,"historical_provenance.toml"),"w") do io
        TOML.print(io,provenance)
    end
end
if abspath(PROGRAM_FILE)==@__FILE__;replay_v2();end
