# Tiny convex simplex QP with exact line minimization along pair transfers.
# Objective is half squared norm after cancellation by active box normals.
# The Frank-Wolfe gap bounds suboptimality for this finite sampled convex problem.
function round3_pairwise_direction(G,w,active;maxsweeps=2000,gap_atol=1e-14,gap_rtol=1e-8)
    lower=w[active].==0;upper=w[active].==MAX_WATER_FRACTION
    function residual(q)
        r=copy(q);r[lower].=min.(r[lower],0.);r[upper].=max.(r[upper],0.);r
    end
    value(q)=sum(abs2,residual(q))/2
    m=size(G,2);alpha=fill(1/m,m);q=G*alpha;sweeps=0
    function line_minimum(q,delta,lo,hi)
        knots=Float64[lo,hi]
        for k in eachindex(q)
            if (lower[k] || upper[k]) && delta[k]!=0
                t=-q[k]/delta[k]
                lo<t<hi && push!(knots,t)
            end
        end
        sort!(unique!(knots));bestt=0.;best=value(q)
        for k in 1:length(knots)-1
            left,right=knots[k],knots[k+1];mid=(left+right)/2
            qm=q.+mid.*delta
            participating=.!lower .& .!upper .| (lower .& (qm.<0)) .| (upper .& (qm.>0))
            denom=sum(abs2,delta[participating])
            t=denom>0 ? clamp(-sum(q[participating].*delta[participating])/denom,left,right) : left
            f=value(q.+t.*delta)
            if f<best;best=f;bestt=t;end
        end
        bestt,best
    end
    gap=Inf;psi=value(q)
    for sweep in 1:maxsweeps
        for i in 1:m-1,j in i+1:m
            delta=G[:,i].-G[:,j]
            t,f=line_minimum(q,delta,-alpha[i],alpha[j])
            if f<psi
                alpha[i]+=t;alpha[j]-=t
                q=G*alpha;psi=value(q)
            end
        end
        r=residual(q);gradient=transpose(G)*r
        gap=max(dot(alpha,gradient)-minimum(gradient),0.)
        sweeps=sweep
        gap<=gap_atol+gap_rtol*psi && break
    end
    @assert minimum(alpha)>=-1e-12 && abs(sum(alpha)-1)<1e-10
    r=residual(q);d=zeros(length(w));d[active].=-r
    d,alpha,norm(r),sweeps,gap,gap<=gap_atol+gap_rtol*psi
end
