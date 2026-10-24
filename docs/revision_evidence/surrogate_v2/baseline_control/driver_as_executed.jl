include(joinpath(@__DIR__, "..", "..", "dev", "validate_surrogate_v2.jl"))
out,dump=abspath.(ARGS)
oldroot=joinpath(@__DIR__,"..","..","docs","revision_evidence","corrected_validation")
ph,mat,site,em,vol,shallow,es=v2state(dump)
paths=filter(p->p.valid,precompute_paths(vol,em,[0.,15.,30.,45.,60.,75.],exclusive_degree_grid(360.,15.)))
install_dry(joinpath(oldroot,"calibration.toml"));set_csda_material_correction!(enabled=false)
nc=num_cells(vol);dry=zeros(nc);wet=[.15+.45*sin(vol.centroids[1,c]/700)^2*(vol.centroids[3,c]>350) for c in 1:nc]
rows=NamedTuple[]; seen=Set{Int}()
for r in readrows(joinpath(oldroot,"held_out_closure.csv"))
    b=Int(r["bin"]);b in seen && continue;push!(seen,b)
    for (name,w,key) in (("dry",dry,"predicted_dry"),("wet",wet,"predicted_wet"))
        actual=compute_directional_flux_csda(ph,shallow,mat,site,paths[b],w,es)
        ref=Float64(r[key]);err=(actual-ref)/max(abs(ref),1e-30)
        push!(rows,(bin=b,zenith=paths[b].zenith,field=name,stored_prediction=ref,reproduced_prediction=actual,relative_difference=err))
    end
end
csv_rows(joinpath(out,"baseline_control.csv"),rows)
maxerr=maximum(abs(r.relative_difference) for r in rows)
open(joinpath(out,"baseline_control.toml"),"w") do io
    TOML.print(io,Dict("rays"=>length(seen),"comparisons"=>length(rows),"max_absolute_relative_difference"=>maxerr,"tolerance"=>1e-10,"pass"=>maxerr<1e-10,"scope"=>"material correction disabled; frozen historical dry correction; same geometry, fields, and midpoint nodes"))
end
@assert length(seen)==21
@assert maxerr<1e-10
println("Baseline control max relative difference: ",maxerr)
