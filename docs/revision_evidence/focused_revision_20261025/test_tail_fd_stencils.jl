using Test
syntax=Meta.parseall(read(joinpath(@__DIR__,"current_grid_tail_audit_v2.jl"),String))
helper=only(x for x in syntax.args if x isa Expr && x.head==:function && x.args[1].args[1]==:focused_fd_from_values)
Core.eval(Main,helper)
@testset "Actual finite-difference stencil algebra" begin
 @test focused_fd_from_values(2.1,2.1034,2.1068,.001,"forward2")≈3.4 atol=1e-12
 @test focused_fd_from_values(5.16,5.1566,5.1532,.001,"backward2")≈3.4 atol=1e-12
 @test abs(focused_fd_from_values(1.4e-6,1.4e-6,1.4e-6,.001,"forward2"))<1e-17
 @test focused_fd_from_values(2.1,2.0966,2.1034,.001,"central")≈3.4 atol=1e-12
end
