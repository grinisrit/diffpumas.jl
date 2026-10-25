# Finite sampled-gradient direction heuristic, without a stationarity claim.
function round3_simplex_projection(v)
    u=sort(v;rev=true);css=cumsum(u).-1
    rho=findlast(i->u[i]>css[i]/i,eachindex(u))
    max.(v.-css[rho]/rho,0.)
end
function round3_sampled_direction(G,w,active; maxiter=10000)
    lower=w[active].==0;upper=w[active].==MAX_WATER_FRACTION
    function residual(q)
        r=copy(q);r[lower].=min.(r[lower],0.);r[upper].=max.(r[upper],0.);r
    end
    alpha=fill(1/size(G,2),size(G,2));L=max(opnorm(G)^2,eps())
    iterations=0;change=Inf
    for k in 1:maxiter
        r=residual(G*alpha)
        next=round3_simplex_projection(alpha.-(transpose(G)*r)./L)
        change=norm(next.-alpha,Inf);alpha=next;iterations=k
        change<1e-12 && break
    end
    q=G*alpha;r=residual(q);d=zeros(length(w));d[active].=-r
    d,alpha,norm(r),iterations,change
end
