# Standalone: julia --project=. test/test_inverse_solver.jl
using Test, DiffPumas, DiffPumas.Tomography, LinearAlgebra, SparseArrays
const InvTestTomography=DiffPumas.Tomography
@testset "Consistent inverse objectives and constraints" begin
    graph=[[2,2,1],Int[],[2]]
    smooth=SmoothnessPrior(graph,2.3)
    @test smooth.neighbors==[[2],[1,3],[2]]
    @test EdgePrior(graph,2.3,0.2).neighbors==smooth.neighbors
    w=[0.1,0.4,0.8];h=1e-6;basis=Matrix{Float64}(I,3,3)
    for prior in (smooth,EdgePrior(graph,2.3,0.2))
        grad=prior isa SmoothnessPrior ? laplacian_gradient(prior,w) : edge_gradient(prior,w)
        fd=[(prior_value(prior,w+h*basis[:,k])-prior_value(prior,w-h*basis[:,k]))/(2h) for k in 1:3]
        @test grad≈fd rtol=1e-8 atol=1e-9
    end
    H=Matrix(InvTestTomography._prior_hessian(smooth,w))
    Hfd=hcat([(laplacian_gradient(smooth,w+h*basis[:,k])-laplacian_gradient(smooth,w-h*basis[:,k]))/(2h) for k in 1:3]...)
    @test H≈Hfd rtol=1e-8 atol=1e-9
    @test H*ones(3)≈zeros(3) atol=1e-12
    @test minimum(eigvals(Symmetric(H)))>=-1e-10
    edge=EdgePrior(graph,2.3,0.2)
    @test InvTestTomography._prior_hessian(edge,w)*w≈edge_gradient(edge,w)
    model=v->(copy(v),sparse(Matrix{Float64}(I,length(v),length(v))))
    posterior=laplace_posterior(model,w;weights=ones(3),prior=smooth,free=trues(3))
    @test posterior.HR≈H
    edgepost=laplace_posterior(model,w;weights=ones(3),prior=edge,free=trues(3))
    @test edgepost.HR≈zeros(3,3) # Both differences lie outside the Huber quadratic branch.
    @test edgepost.sigma≈ones(3) atol=1e-7

    # Starting at a perfect data fit, regularization should still move the field.
    prior=SmoothnessPrior([[2],[1]],10.0)
    fit,hist=gauss_newton_reconstruct(model,[1.0,0.0];w0=[1.0,0.0],prior=prior,n_iter=12,box=(0.0,1.0))
    @test fit≈[21/41,20/41] atol=1e-6
    @test norm(fit-[1.0,0.0])>0.1
    @test 0.5*norm(fit-[1.0,0.0])^2+prior_value(prior,fit)<prior_value(prior,[1.0,0.0])
    @test hist[1]==0 && hist[end]>0 # Data misfit can rise while MAP cost falls.

    # A nonzero fixed neighbor contributes curvature but cannot itself move.
    boundarymodel=v->([v[1]],sparse(reshape([1.0,0.0],1,2)))
    fixedfit,_=gauss_newton_reconstruct(boundarymodel,[0.0];w0=[0.1,0.8],prior=SmoothnessPrior([[2],[1]],2.0),free=Bool[true,false],n_iter=12)
    @test fixedfit[2]===0.8
    @test fixedfit[1]≈0.64 atol=1e-6
    nonlinear_fixed=v->([exp(v[1])+0.3v[2],0.4v[1]+exp(v[2])],
        sparse([exp(v[1]) 0.3;0.4 exp(v[2])]))
    oraclefit,_=gauss_newton_reconstruct(nonlinear_fixed,[1.3,1.6];w0=[0.1,0.4],
        weights=[2.0,5.0],prior=SmoothnessPrior([[2],[1]],0.7),free=Bool[true,false],n_iter=30)
    @test oraclefit[2]===0.4
    @test oraclefit[1]≈0.24287925833166624 atol=2e-6
    calls=Ref(0)
    counted=v->begin calls[]+=1;model(v) end
    allfixed,allhist=gauss_newton_reconstruct(counted,[0.0,0.0];w0=[0.2,0.8],free=falses(2),n_iter=12)
    @test allfixed==[0.2,0.8] && length(allhist)==1 && calls[]==1
    @test_throws DimensionMismatch gauss_newton_reconstruct(model,zeros(2);w0=zeros(2),free=trues(1))

    affine=v->([2+3v[1]],sparse(reshape([3.0],1,1)))
    frozen,_=gauss_newton_reconstruct(affine,[3.5];w0=[0.1],relinearize=false,n_iter=12)
    @test frozen[1]≈0.5 atol=1e-7
    gd_calls=Ref(0)
    nonlinear=v->begin gd_calls[]+=1;([2+v[1]+v[1]^2],sparse(reshape([1+2v[1]],1,1))) end
    gd,_=gradient_descent_reconstruct(nonlinear,[2.8];w0=[0.2],relinearize=false,optimizer=:gd,n_iter=150,lr=0.1)
    @test gd[1]≈0.6 atol=1e-7
    @test gd_calls[]==1
    step_model=v->(copy(v),sparse(reshape([1.0],1,1)))
    stepfit,stephistory=gradient_descent_reconstruct(step_model,[1.0];w0=[0.0],optimizer=:gd,n_iter=1,lr=0.1)
    @test stephistory[end]≈norm(stepfit-[1.0])
    smooth_calls=Ref(0)
    flatdata=v->begin smooth_calls[]+=1;([0.0],spzeros(1,5)) end
    smoothonly=SmoothnessPrior([[2],[1,3],[2,4],[3,5],[4]],1.0)
    initial=[0.0,0.1,0.6,0.3,0.9]
    smoothed,_=gradient_descent_reconstruct(flatdata,[0.0];w0=initial,prior=smoothonly,
        optimizer=:pgd,n_iter=50,lr=1.0)
    @test smooth_calls[]>3 # A constant data residual is not convergence of the prior.
    @test prior_value(smoothonly,smoothed)<1e-6*prior_value(smoothonly,initial)
    @test abs(last(gradient_descent_reconstruct(affine,[3.5];w0=[-2.0],optimizer=:pgd,n_iter=4,lr=1.0)[2]))<1e-5
end

@testset "GN prediction-only line search" begin
    target=[exp(8*0.65)]
    reference_calls=Ref(0);fast_calls=Ref(0);prediction_calls=Ref(0)
    primal=v->[exp(8v[1])]
    reference=v->begin reference_calls[]+=1;(primal(v),sparse(reshape([8exp(8v[1])],1,1))) end
    fast=v->begin fast_calls[]+=1;(primal(v),sparse(reshape([8exp(8v[1])],1,1))) end
    predict=v->begin prediction_calls[]+=1;primal(v) end
    full,fullhistory=gauss_newton_reconstruct(reference,target;w0=[0.01],n_iter=20,box=(0.0,1.0))
    cheap,cheaphistory=gauss_newton_reconstruct(fast,target;w0=[0.01],n_iter=20,box=(0.0,1.0),predict_only=predict)
    @test cheap≈full atol=1e-12 rtol=1e-12
    @test cheaphistory≈fullhistory atol=1e-12 rtol=1e-12
    @test cheap[1]≈0.65 atol=1e-8
    @test fast_calls[]<reference_calls[]
    @test prediction_calls[]>0
    @info "GN Jacobian evaluation counts" full=reference_calls[] predict_only=fast_calls[] primal_trials=prediction_calls[]
    # An inaccurate prediction callback may not authorize an actual objective increase.
    checked,checkedhist=gauss_newton_reconstruct(fast,target;w0=[0.01],n_iter=8,box=(0.0,1.0),predict_only=v->target)
    @test all(diff(checkedhist).<=1e-10)
    @test norm(primal(checked)-target)<norm(primal([0.01])-target)
    zero_calls=Ref(0)
    zero_model=v->begin zero_calls[]+=1;(copy(v),sparse(reshape([1.0],1,1))) end
    saturated,history=gauss_newton_reconstruct(zero_model,[2.0];w0=[0.9],n_iter=8)
    @test saturated==[0.9] && zero_calls[]==1 && length(history)==2
end
