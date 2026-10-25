using Test,SHA,TOML
root=normpath(joinpath(@__DIR__,"..","..",".."));source=joinpath(root,"docs/revision_evidence/focused_revision_20261025/current_grid_tail_audit_v2.jl")
syntax=Meta.parseall(read(source,String));definition=only(x for x in syntax.args if x isa Expr && x.head==:function && x.args[1] isa Expr && x.args[1].args[1]==:focused_fd_from_values)
eval(definition) # Only the actual arithmetic helper; never include/run transport.
checks=0
@testset "Independent actual stencil polynomial and bound checks" begin
 for (f,derivative) in ((x->1.234,x->0.),(x->.7+1.4x,x->1.4),(x->.3-.7x+2x*x,x->-.7+4x)),h in (1e-2,1e-3,1e-4)
  for (tag,x,a,b) in (("central",.4,.4-h,.4+h),("forward2",0.,h,2h),("backward2",.9,.9-h,.9-2h))
   @test all(0 .<= [x,a,b] .<= .9)
   @test isapprox(focused_fd_from_values(f(x),f(a),f(b),h,tag),derivative(x);rtol=1e-9,atol=2e-9)
   global checks+=1
  end
 end
 @test_throws ErrorException focused_fd_from_values(1.,2.,3.,.1,"invalid")
end
open(io->TOML.print(io,Dict("passed"=>true,"analytic_stencil_cases"=>checks,"source_sha256"=>bytes2hex(sha256(read(source))),"scope"=>"Independent actual-helper constant/linear/quadratic tests, central and both feasible boundary stencils; no transport/MC execution")),joinpath(@__DIR__,"actual_stencil_unit.toml"),"w")
