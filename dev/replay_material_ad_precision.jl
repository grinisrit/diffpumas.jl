#!/usr/bin/env julia
# Deterministic rule-change gate: never calls MC, refits, or reselects a cell.
include(joinpath(@__DIR__,"validate_surrogate_v2.jl"))
samebits(x,y)=reinterpret(UInt64,Float64(x))==reinterpret(UInt64,Float64(y))

function replay_ad_precision(argv=ARGS)
    iseven(length(argv)) || error("expected --name value pairs")
    opts=Dict("output-dir"=>"","dump"=>"","artifact"=>"","validation-root"=>"","historical-reference"=>"","historical-replay"=>"")
    for i in 1:2:length(argv)
        key=replace(argv[i],r"^--"=>"");haskey(opts,key) || error("unknown option")
        isempty(opts[key]) || error("duplicate option");opts[key]=abspath(argv[i+1])
    end
    all(x->!isempty(x),values(opts)) || error("all six options required")
    out=opts["output-dir"]
    isdir(out) && !issubset(Set(readdir(out)),Set(["execution.json","pid.json","run.log"])) && error("refuse existing evidence")
    mkpath(out)
    vr=opts["validation-root"];hr=opts["historical-reference"];replay=opts["historical-replay"]
    audit_script=raw"""
import json,hashlib,sys
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
vr,hr,replay,out=map(Path,sys.argv[1:])
report={}
for root in (vr,replay):
 end=json.loads((root/'end_status.json').read_text())
 assert all(end[k] is True for k in ('execution_completed','source_unchanged','runtime_unchanged'))
 for name,d in end['output_sha256'].items():assert sha(root/name)==d,(root,name)
 report[str(root)]=dict(end_status_sha256=sha(root/'end_status.json'),source_producer=json.loads((root/'execution.json').read_text())['source_sha256'])
old=dict(line.split(None,1)[::-1] for line in (hr/'SHA256SUMS').read_text().splitlines())
for name in ('mc_jacobian.csv','config.toml','calibration.toml'):assert sha(hr/name)==old[name]
report['historical_manifest_sha256']=sha(hr/'SHA256SUMS')
(out/'reference_audit.json').write_text(json.dumps(report,indent=2)+'\n')
"""
    run(Cmd(["python3","-c",audit_script,vr,hr,replay,out]))
    inputpaths=[@__FILE__,joinpath(@__DIR__,"validate_surrogate_v2.jl"),opts["dump"],opts["artifact"],
        joinpath(vr,"heldout_flux.csv"),joinpath(vr,"heldout_derivatives.csv"),joinpath(vr,"preregistered_plan.toml"),
        joinpath(hr,"mc_jacobian.csv"),joinpath(hr,"config.toml"),joinpath(hr,"SHA256SUMS"),
        joinpath(vr,"end_status.json"),joinpath(replay,"end_status.json"),joinpath(replay,"historical_derivatives.csv")]
    hashes=Dict(abspath(p)=>bytes2hex(sha256(read(p))) for p in inputpaths)
    ph,mat,site,em,vol,shallow,es=v2state(opts["dump"]);nc=num_cells(vol)
    metadata=load_csda_corrections!(opts["artifact"])
    metadata["physics_dump_sha256"]==bytes2hex(sha256(read(opts["dump"]))) || error("artifact dump mismatch")
    coeff=deepcopy(get_csda_material_correction())
    install(enabled)=set_csda_material_correction!(enabled=enabled,linear=coeff.linear,quadratic=coeff.quadratic)
    paths=filter(p->p.valid,precompute_paths(vol,em,[15.,45.,60.,75.],[37.5,157.5,277.5]))
    histpaths=filter(p->p.valid,precompute_paths(vol,em,[0.,15.,30.,45.,60.,75.],exclusive_degree_grid(360.,15.)))
    fields=[fill(.35,nc),[.15+.4*sin(vol.centroids[1,c]/900+vol.centroids[3,c]/600)^2 for c in 1:nc],
        [.15+.4*cos(vol.centroids[1,c]/900+vol.centroids[3,c]/600)^2 for c in 1:nc],
        [vol.centroids[3,c]>500 && hypot(vol.centroids[1,c],vol.centroids[2,c])<1500 ? .6 : 0. for c in 1:nc]]
    fluxrows=readrows(joinpath(vr,"heldout_flux.csv"));testrows=readrows(joinpath(vr,"heldout_derivatives.csv"))
    histrows=readrows(joinpath(hr,"mc_jacobian.csv"));histafter=readrows(joinpath(replay,"historical_derivatives.csv"))
    length(testrows)==384 && length(histrows)==576 && length(histafter)==576 || error("reference coverage differs")
    histlookup=Dict((Int(r["case"]),Float64(r["fraction"]),Int(r["replicate"]))=>r for r in histafter)
    primal=NamedTuple[];paired=NamedTuple[];adrows=NamedTuple[]
    unique_flux=Dict((Int(r["bin"]),Int(r["field"]))=>r for r in fluxrows)
    for enabled in (false,true)
        install(enabled);label=enabled ? "after" : "before"
        for ((b,k),r) in sort!(collect(unique_flux);by=first)
            actual=compute_directional_flux_csda(ph,shallow,mat,site,paths[b],fields[k],es)
            push!(primal,(bin=b,field=k,mode=label,stored=Float64(r[label]),replayed=actual,bitwise_equal=samebits(actual,r[label])))
        end
        # Constant global correction during ordered threaded primal evaluations.
        for (name,rows,pool,nmc) in (("new",testrows,paths,1024),("historical",histrows,histpaths,1024))
            results=revision_replicates(length(rows);parallel=true) do i
                r=rows[i];b=Int(r["bin"]);c=Int(r["cell"])
                w=name=="new" ? copy(fields[Int(r["field"])]) : fill(Float64(r["fraction"]),nc)
                lo=Float64(r["fraction_lo"]);hi=Float64(r["fraction_hi"])
                wl=copy(w);wh=copy(w);wl[c]=lo;wh[c]=hi
                en=sample_energy_set(nmc,1e-3,1e9,Int(r["seed"]))
                value=(compute_directional_flux_csda(ph,shallow,mat,site,pool[b],wh,en)-compute_directional_flux_csda(ph,shallow,mat,site,pool[b],wl,en))/(hi-lo)
                old=name=="new" ? Float64(r[label]) : enabled ? Float64(histlookup[(Int(r["case"]),Float64(r["fraction"]),Int(r["replicate"]))]["after"]) : Float64(r["csda_secant_paired_energy"])
                (population=name,bin=b,cell=c,field=name=="new" ? Float64(r["field"]) : Float64(r["fraction"]),replicate=Int(r["replicate"]),mode=label,stored=old,replayed=value,bitwise_equal=samebits(old,value))
            end
            append!(paired,results);csv_rows(joinpath(out,"paired_primal_bitwise.csv"),paired)
        end
    end
    csv_rows(joinpath(out,"flux_primal_bitwise.csv"),primal)
    install(true)
    for (name,rows,pool) in (("new",testrows,paths),("historical",histrows,histpaths))
        cases=Dict((Int(r["bin"]),Int(r["cell"]),name=="new" ? Float64(r["field"]) : Float64(r["fraction"]))=>r for r in rows)
        for ((b,c,k),r) in sort!(collect(cases);by=first)
            w=name=="new" ? copy(fields[Int(k)]) : fill(k,nc)
            _,J=assemble_forward_and_jacobian(ph,shallow,mat,site,[pool[b]],w,es;n_cells=nc,threaded=false)
            ad=J[1,c];base=compute_directional_flux_csda(ph,shallow,mat,site,pool[b],w,es)
            for h in (1e-4,1e-5,1e-6,1e-7,1e-8)
                wa=copy(w);wb=copy(w)
                if w[c]==0
                    wa[c]=h;wb[c]=2h
                    fd=(-3base+4compute_directional_flux_csda(ph,shallow,mat,site,pool[b],wa,es)-compute_directional_flux_csda(ph,shallow,mat,site,pool[b],wb,es))/(2h)
                else
                    wa[c]-=h;wb[c]+=h
                    fd=(compute_directional_flux_csda(ph,shallow,mat,site,pool[b],wb,es)-compute_directional_flux_csda(ph,shallow,mat,site,pool[b],wa,es))/(2h)
                end
                push!(adrows,(population=name,bin=b,cell=c,field=k,step=h,ad=ad,fd=fd,relative_error=abs(ad-fd)/max(abs(ad),abs(fd),1e-30),one_sided=w[c]==0))
            end
            csv_rows(joinpath(out,"ad_fd_replay.csv"),adrows)
        end
    end
    changes=[p for (p,d) in hashes if bytes2hex(sha256(read(p)))!=d]
    gate=filter(r->r.step==1e-6,adrows)
    status=Dict("source_inputs_sha256"=>hashes,"changed_inputs"=>changes,"flux_checks"=>length(primal),"paired_secant_checks"=>length(paired),
        "all_stored_primal_values_bitwise_equal"=>all(r->r.bitwise_equal,vcat(primal,paired)),
        "ad_fd_cases"=>length(gate),"max_relative_ad_fd_error_at_original_step"=>maximum(r.relative_error for r in gate),
        "ad_fd_threshold"=>1e-4,"internal_adjoint_step"=>get_adjoint_fd_step(),
        "scope"=>"Deterministic same-case replay; no MC, refitting, reselection, or new physical-reference accuracy claim")
    open(io->TOML.print(io,status),joinpath(out,"replay_status.toml"),"w")
    isempty(changes) || error("input changed")
    status["all_stored_primal_values_bitwise_equal"] || error("primal changed")
    length(gate)==120 && status["max_relative_ad_fd_error_at_original_step"]<1e-4 || error("local AD/FD gate failed")
end
if abspath(PROGRAM_FILE)==@__FILE__;replay_ad_precision();end
