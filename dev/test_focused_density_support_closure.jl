using DiffPumas,LinearAlgebra,Test
using DiffPumas.Tomography
# Extract the actual data-transform function from the corrected driver, rather
# than maintaining a second implementation of its physics/objective formula.
syntax=Meta.parseall(read(joinpath(@__DIR__,"focused_density_support_v2.jl"),String))
main=only(x for x in syntax.args if x isa Expr && x.head==:function && x.args[1] isa Expr && x.args[1].args[1]==:focused_density_support_main)
parts=only(x for x in main.args[2].args if x isa Expr && x.head==:function && x.args[1] isa Expr && x.args[1].args[1]==:parts)
@eval function actual_parts_scope_regression()
    obs=[2.,3.];W=[1.,.5];prior=SmoothnessPrior([[2],[1]],.2)
    args=(papermatch_log_scale=.3,papermatch_abs_scale=0.)
    ray_cached=(rho,w)->rho.*exp.(w)
    $(parts)
    w=[.1,.2];before=ray_cached(2710.,w)
    transformed=parts(before,w)
    after=ray_cached(2710.,w)
    before,after,transformed
end
@testset "Actual driver data transform preserves ray-cache callback" begin
    before,after,result=actual_parts_scope_regression()
    @test before==after
    @test isfinite(result.objective) && result.objective>=result.prior
    @test isfinite(result.normalization) && result.normalization>0
end
