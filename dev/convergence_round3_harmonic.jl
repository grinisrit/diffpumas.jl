# Exact conditional minimization of an unchanged nonnegative graph prior.
function round3_harmonic_map(L,support,seen,start)
    n=length(start);U=findall(support .& .!seen);C=findall(support .& seen)
    @assert isapprox(L,transpose(L);atol=1e-12,rtol=1e-12)
    @assert maximum(abs.(vec(sum(L;dims=2))))<=1e-10*max(opnorm(L,Inf),1.)
    @assert all(L[i,j]<=1e-12 for i in 1:n for j in 1:n if i!=j)
    tangent=zeros(n,length(C))
    for (k,c) in enumerate(C);tangent[c,k]=1.;end
    pending=Set(U);blocks=Any[];records=Dict{String,Any}[]
    while !isempty(pending)
        firstcell=minimum(pending);component=Int[];queue=[firstcell];delete!(pending,firstcell)
        while !isempty(queue)
            i=pop!(queue);push!(component,i)
            neighbors=sort([j for j in pending if L[i,j]<0])
            for j in neighbors;delete!(pending,j);push!(queue,j);end
        end
        sort!(component);outside=setdiff(collect(1:n),component)
        B=L[component,outside];anchored=any(B.<0)
        meanvalue=sum(start[component])/length(component)
        factor=anchored ? cholesky(Symmetric(L[component,component]);check=true) : nothing
        if anchored
            tangent[component,:].=-(factor\L[component,C])
        end
        push!(blocks,(cells=component,outside=outside,B=B,factor=factor,meanvalue=meanvalue,anchored=anchored))
        push!(records,Dict("cells"=>component,"anchored"=>anchored,"representative_mean"=>meanvalue))
    end
    function extend(x)
        y=copy(x)
        for block in blocks
            value=block.anchored ? block.factor\(-block.B*y[block.outside]) : fill(block.meanvalue,length(block.cells))
            @assert minimum(value)>=-1e-10 && maximum(value)<=MAX_WATER_FRACTION+1e-10
            y[block.cells].=clamp.(value,0.,MAX_WATER_FRACTION)
        end
        @assert y[.!support]==start[.!support]
        y
    end
    extend,U,C,records,tangent
end
