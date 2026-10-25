using DiffPumas,LinearAlgebra,Test
using DiffPumas.Tomography
const MAX_WATER_FRACTION=DiffPumas.Tomography.MAX_WATER_FRACTION
include(joinpath(@__DIR__,"focused_density_support_solver.jl"))
@testset "Instrumented bounded GN equals production active-set GN" begin
    model=w->([exp(w[1])+.2w[2],exp(2w[2])+.1w[1]],[exp(w[1]) .2;.1 2exp(2w[2])])
    observation=[1.25,1.5];prior=SmoothnessPrior([[2],[1]],.2);w0=[.01,.02]
    function state(w)
        p,J=model(w);r=p-observation;pv=DiffPumas.Tomography.prior_value(prior,w)
        g=transpose(J)*r+laplacian_gradient(prior,w)
        (water=copy(w),flux=p,J=J,residual=r,data=.5dot(r,r),prior=pv,objective=.5dot(r,r)+pv,
            normalization=1.,branch_pg=norm(w.-clamp.(w.-g,0.,MAX_WATER_FRACTION),Inf))
    end
    for budget in (1,3,10)
        trace=Any[];accepted=Any[]
        out=focused_bounded_gn(state(w0),state,state,prior,trues(2);max_outer=budget,deadline=Inf,
            trace_callback=x->push!(trace,x),state_callback=(s,i,k)->push!(accepted,s))
        expected,_=gauss_newton_reconstruct(model,observation;w0=w0,n_iter=budget,prior=prior,
            solver_mode=:active_set,projected_gradient_tol=1e-6,predict_only=w->first(model(w)))
        @test out.state.water==expected
        @test out.state.objective==state(expected).objective
        @test all(x->x.reason!="accepted_full_confirmation"||x.accepted,trace)
        @test all(s->all(0 .<=s.water.<=MAX_WATER_FRACTION),accepted)
    end
    initial=state(w0);out=focused_bounded_gn(initial,state,state,prior,trues(2);deadline=time()-1)
    @test out.state.water==w0 && out.accepted==0
    @test out.termination=="soft_deadline_before_outer"
    # Deadline after a flux-only trial must not call full confirmation or accept it.
    calls=Ref(0);tick=Ref(0.);full=w->begin;calls[]+=1;state(w);end
    out=focused_bounded_gn(initial,full,w->begin;tick[]=2.;state(w);end,prior,trues(2);deadline=1.,clock=()->tick[])
    @test calls[]==0 && out.accepted==0 && out.state.water==w0
    @test out.termination=="soft_deadline_after_prediction"
    calls[]=0;tick[]=0.
    out=focused_bounded_gn(initial,w->begin;calls[]+=1;tick[]=2.;state(w);end,state,prior,trues(2);deadline=1.,clock=()->tick[])
    @test calls[]==1 && out.accepted==0 && out.state.water==w0
    @test out.termination=="soft_deadline_during_confirmation"
end
