include(joinpath(@__DIR__,"..","..","..","examples","revision_suite.jl"))
function current_scaling()
 length(ARGS)==2 || error("usage: DUMP FRESH_OUTPUT")
 dump,out=ARGS;ispath(out)&&error("fresh output required")
 physics=load_physics(dump);idx(n)=get_material_index(physics,n)
 mat=MaterialConfig(idx("StandardRock"),idx("Water"),idx("Air"),idx("PorousWetRock"),2710.,Float64(physics.tables[idx("Water")].density),0.,0.)
 site=build_site(30000.);emap=LVDTopo.build_elevation_map();es=csda_energy_quadrature(32,1e-3,1e9)
 set_adjoint_derivative_mode!(:forwarddiff);set_csda_correction!(enabled=false);set_csda_material_correction!(enabled=false)
 rows=NamedTuple[];mkpath(out)
 # Each stage is compiled/warmed once, then three independent timed evaluations with GC first.
 function measured3(f)
  f();measurements=Any[]
  for _ in 1:3
   GC.gc();t=@timed f();push!(measurements,(value=t.value,seconds=t.time,allocated_bytes=t.bytes,gc_seconds=t.gctime,process_peak_rss_bytes=Sys.maxrss()))
  end
  measurements
 end
 for nz in (2,4,8),angular in (30.,15.)
  volume=measured3(()->create_voxel_volume(emap,mat;half_width_km=3.,surface_step_km=1.5,nz=nz));vol=last(volume).value;nc=num_cells(vol);shallow=falses(nc);w=fill(.3,nc)
  trace=measured3(()->precompute_paths(vol,emap,exclusive_degree_grid(60.,angular),exclusive_degree_grid(360.,angular)));paths=filter(p->p.valid,last(trace).value)
  assemble_forward_and_jacobian(physics,shallow,mat,site,paths[1:1],w,es;n_cells=nc,threaded=false)
  forward=measured3(()->[compute_directional_flux_csda(physics,shallow,mat,site,p,w,es) for p in paths])
  jac=measured3(()->assemble_forward_and_jacobian(physics,shallow,mat,site,paths,w,es;n_cells=nc,threaded=false));flux,J=last(jac).value;W=1 ./max.(abs.(flux),1e-20)
  normal=measured3(()->Matrix(J'*Diagonal(W)*J));H=last(normal).value;ridge=max(maximum(diag(H))*1e-4,1e-20)
  factor=measured3(()->cholesky(Symmetric(H+ridge*I)));solve=measured3(()->last(factor).value\ones(nc))
  for (stage,measurements) in (("volume",volume),("trace",trace),("forward",forward),("jacobian",jac),("normal",normal),("factor",factor),("solve",solve))
   for (rep,m) in enumerate(measurements)
    push!(rows,(nz=nz,rays=length(paths),cells=nc,intersections=sum(length(p.segments) for p in paths),nnz=nnz(J),stage=stage,repeat=rep,seconds=m.seconds,allocated_bytes=m.allocated_bytes,gc_seconds=m.gc_seconds,process_peak_rss_bytes=m.process_peak_rss_bytes,retained_jacobian_bytes=Base.summarysize(J),dense_normal_bytes=Base.summarysize(H)))
   end
  end
  csv_rows(joinpath(out,"scaling_repeats.csv"),rows)
 end
 open(io->TOML.print(io,Dict("julia_version"=>string(VERSION),"machine"=>Sys.MACHINE,"cpu_name"=>Sys.CPU_NAME,"host_logical_cpu_threads"=>Sys.CPU_THREADS,"blas_configuration"=>string(BLAS.get_config()),"backend"=>string(get_adjoint_derivative_mode()),"Julia_threads_available"=>Threads.nthreads(),"BLAS_threads"=>BLAS.get_num_threads(),"forward_and_jacobian_threaded"=>false,"Float64"=>true,"quadrature_nodes"=>32,"corrections"=>"both disabled to reproduce historical benchmark configuration","repetitions"=>3,"scope"=>"Warm individual-stage measurements, noMC/JIT in timed samples, smallgrid scaling only; cumulative process peak RSS not stage incremental memory; no large-scale capacity threshold.")),joinpath(out,"configuration.toml"),"w")
end
current_scaling()
