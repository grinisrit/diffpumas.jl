using Test, LinearAlgebra
@testset "Fixed robust profile and constrained KKT" begin
    A=[1.0 0.2; -0.3 1.0; 0.6 -0.7; -0.8 -0.1]
    model=w->(exp.(A*w),exp.(A*w).*A)
    observed=exp.(A*[0.3,0.6]).*2.5
    robust,scale=DiffPumas.Tomography.profiled_log_observation_operator(model,observed;robust_scale=0.3)
    @test_throws ArgumentError DiffPumas.Tomography.profiled_log_observation_operator(model,observed;normalization=-1)
    @test_throws ArgumentError DiffPumas.Tomography.profiled_log_observation_operator(model,observed;robust_scale=Inf)
    w=[0.1,0.4];r,J=robust(w)
    h=1e-6
    fd=hcat([(first(robust(w+h*I(2)[:,k]))-first(robust(w-h*I(2)[:,k])))/(2h) for k=1:2]...)
    @test J≈fd rtol=1e-6 atol=1e-8
    @test scale(first(model([0.3,0.6])))≈2.5
    for initial in ([0.,0.],[0.7,0.7],[0.1,0.8])
        diagnostics=NamedTuple[]
        fit,_=DiffPumas.Tomography.gauss_newton_reconstruct(robust,zeros(4);w0=initial,n_iter=100,
            solver_mode=:active_set,projected_gradient_tol=1e-9,diagnostics=diagnostics)
        @test fit≈[0.3,0.6] atol=1e-5
        @test last(diagnostics).projected_gradient<1e-8
        @test all(diff([x.objective for x in diagnostics]).<=1e-12)
    end
    # Coupled normal matrix: clipping the unconstrained Newton step can stall.
    B=[1.0 0.9;0.0 0.2]
    linear=w->(B*w,B)
    diagnostics=NamedTuple[]
    fit,_=DiffPumas.Tomography.gauss_newton_reconstruct(linear,[-1.0,0.4];w0=[0.,0.5],n_iter=100,
        solver_mode=:active_set,diagnostics=diagnostics,projected_gradient_tol=1e-9)
    @test fit[1]≈0 atol=1e-8
    @test last(diagnostics).projected_gradient<1e-8
end
