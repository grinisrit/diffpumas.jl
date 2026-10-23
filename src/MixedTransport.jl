"""Prepared straight-line backward MIXED transport with weighted adjoint hard events.

Contexts own copies of their numeric inputs. Prepare once per batch; physics and
recipe mutations after preparation do not affect that batch. Reprepare to validate
and use changed inputs (in-place table modification is rejected).
"""
module MixedTransport
using ..Physics
using ..Types
using ..Constants
using ..Materials
using Random
export MixedKernel, prepare_mixed_kernel, prepare_mixed_kernels, reverse_del_event, transport_mixed_step

struct MixedKernel
    mass::Float64
    ctau::Float64
    cutoff::Float64
    energies::Vector{Float64}
    stopping::Vector{Float64}
    sigma::Vector{Float64}
    range::Vector{Float64}
    hazard::Vector{Float64}
    proper::Vector{Float64}
    elements::Tuple
    onset::Float64
    owner::WeakRef
    recipe_key::Tuple
end

MixedKernel(mass,ctau,cutoff,energies,stopping,sigma,range,hazard,proper,elements,onset) =
    MixedKernel(mass,ctau,cutoff,energies,stopping,sigma,range,hazard,proper,elements,onset,WeakRef(nothing),())
function check_kernel_owner(c::MixedKernel,p::PhysicsTables,material)
    mix=material isa Integer ? MaterialMixture(Int(material)) : material
    c.owner.value === p.tables && c.mass==p.mass && c.ctau==p.ctau && c.cutoff==p.settings.cutoff &&
        c.recipe_key==(Tuple(mix.materials),Tuple(mix.fractions)) ||
        throw(ArgumentError("Prepared mixed kernel belongs to different physics or material; prepare a new batch context"))
    return c
end

# Linear interpolation in the natural variable, including exactly flat hazard
# intervals. All inversions are monotone; no table extrapolation is implicit.
function lookup(x, xs, ys)
    x < first(xs) && throw(DomainError(x,"Below prepared mixed kernel domain"))
    x > last(xs) && throw(DomainError(x,"Above prepared mixed kernel domain; regenerate tables with a larger K_max"))
    i = clamp(searchsortedlast(xs,x),1,length(xs)-1)
    d = xs[i+1]-xs[i]
    return d == 0 ? ys[i] : ys[i]+(x-xs[i])/d*(ys[i+1]-ys[i])
end

prepare_mixed_kernel(p::PhysicsTables,material;refinement=8) =
    only(prepare_mixed_kernels(p,[material];refinement=refinement))
function prepare_mixed_kernels(p::PhysicsTables,materials;refinement=8)
    metadata=Physics.material_metadata(p;required=true)
    return [_prepare_mixed_kernel(p,m,metadata;refinement=refinement) for m in materials]
end
function _prepare_mixed_kernel(p::PhysicsTables, material, metadata; refinement=8)
    metadata === nothing && error("Elemental recipes required")
    (p.settings.bremsstrahlung,p.settings.pair_production,p.settings.photonuclear)==("SSR","SSR","DRSS") || error("Mixed kernel supports SSR/SSR/DRSS models only")
    mix = material isa Integer ? MaterialMixture(Int(material)) : material
    indices, fractions = Tuple(mix.materials),Tuple(mix.fractions)
    all(i -> 1<=i<=length(p.tables),indices) || error("Invalid mixture material")
    all(f -> isfinite(f)&&f>=0,fractions) && isapprox(sum(fractions),1;atol=1e-12) || error("Invalid mixture fractions")
    lo = maximum(first(p.tables[i].energies) for i in indices)
    hi = minimum(last(p.tables[i].energies) for i in indices)
    grid = sort!(unique!(vcat([Float64[e for e in p.tables[i].energies if lo<=e<=hi] for i in indices]...)))
    energies = Float64[]
    for j in 1:length(grid)-1
        append!(energies,exp.(range(log(grid[j]),log(grid[j+1]);length=refinement+1))[1:end-1])
    end
    push!(energies,hi)
    energies[1]=lo
    # Snapshot arrays: interpolation delegates to existing table convention only
    # during preparation, never retains mutable user tables in inner loops.
    stop = [sum(f*property_stopping_power(p,ENERGY_LOSS_MIXED,i,k) for (i,f) in zip(indices,fractions)) for k in energies]
    sigma = [sum(f*property_cross_section(p,i,k) for (i,f) in zip(indices,fractions)) for k in energies]
    all(x->isfinite(x)&&x>0,stop) || error("Invalid restricted stopping power")
    all(x->isfinite(x)&&x>=0,sigma) || error("Invalid hard cross section")
    # Official PUMAS regularises the zero-rate tail with a do-nothing process.
    # Retain the first positive table knot as Kt and its rate below onset.
    original_rates=[sum(f*property_cross_section(p,i,k) for (i,f) in zip(indices,fractions)) for k in grid]
    onset_index=findfirst(>(0),original_rates)
    onset=onset_index===nothing ? Inf : grid[onset_index]
    if onset_index!==nothing
        sigma[energies .< onset] .= original_rates[onset_index]
    end
    r=zeros(length(energies));h=copy(r);t=copy(r)
    for j in 2:length(energies)
        a,b=energies[j-1],energies[j]; d=b-a
        r[j]=r[j-1]+d/2*(1/stop[j-1]+1/stop[j])
        h[j]=h[j-1]+d/2*(sigma[j-1]/stop[j-1]+sigma[j]/stop[j])
        t[j]=t[j-1]+d/2*p.mass*(1/(sqrt(a*(a+2p.mass))*stop[j-1])+1/(sqrt(b*(b+2p.mass))*stop[j]))
    end
    elements=Tuple((Z=e.Z,A=e.A,I=e.I,fraction=f*e.fraction)
                   for (i,f) in zip(indices,fractions) for e in metadata.recipes[i].elements)
    return MixedKernel(p.mass,p.ctau,p.settings.cutoff,energies,stop,sigma,r,h,t,elements,onset,WeakRef(p.tables),(indices,fractions))
end

function differential_components(c::MixedKernel,k,q)
    return Tuple((element=e,process=process,value=e.fraction*AVOGADRO_NUMBER/(e.A*.001)*k*raw)
        for e in c.elements for (process,raw) in (
            (1,Materials.dcs_bremsstrahlung_ssr(e.Z,e.A,c.mass,k,q)),
            (2,Materials.dcs_pair_production_ssr(e.Z,e.A,c.mass,k,q)),
            (3,Materials.dcs_photonuclear_drss(e.Z,e.A,c.mass,k,q)),
            (4,Materials.electronic_dcs(e.Z,e.I,c.mass,k,q))))
end
function differential_kernel(c::MixedKernel,k,q)
    return sum(x.value for x in differential_components(c,k,q);init=0.)
end

function reverse_del_event(c::MixedKernel,k::Float64,rng::AbstractRNG;details=false)
    sigma=lookup(k,c.energies,c.sigma)
    sigma>0 || error("Cannot sample a zero-rate hard event")
    # Below Kt, the regularised hazard includes a null CEL process.
    pcel = k>=c.onset ? 0. : -log(c.cutoff)/(-log(c.cutoff)+log(c.onset/(c.onset-k)))
    if pcel>0 && rand(rng)<pcel
        return details ? (k,1/pcel,0,nothing) : (k,1/pcel)
    end
    xmax = k < .5*(c.mass-ELECTRON_MASS)^2/ELECTRON_MASS ?
        min(1.,2ELECTRON_MASS*(k+2c.mass)/(c.mass+ELECTRON_MASS)^2) : 1.
    xmin=k>=c.onset*(1-c.cutoff) ? c.cutoff : 1-k/c.onset
    xmax>xmin || return details ? (k,1.,0,nothing) : (k,1.)
    invnorm=1/xmin-1/xmax
    nu=1/(1/xmin-rand(rng)*invnorm)
    r=1-nu
    ancestor=k/r
    pdf=1/(invnorm*nu^2)
    components=differential_components(c,ancestor,ancestor-k)
    total=sum(x.value for x in components;init=0.)
    weight=total/(pdf*r*sigma*(1-pcel))
    if details
        total<=0 && return ancestor,weight,0,nothing
        u=rand(rng)*total
        for x in components
            u-=x.value
            u<=0 && return ancestor,weight,x.process,x.element
        end
        x=last(components)
        return ancestor,weight,x.process,x.element
    end
    return ancestor,weight
end

function transport_mixed_step(c::MixedKernel,state::State{Float64},density::Float64,
                              distance::Float64,rng::AbstractRNG;energy_limit=-1.)
    density>0 && distance>=0 || throw(ArgumentError("Positive density and nonnegative distance required"))
    k=state.energy
    energy_limit>last(c.energies) && throw(ArgumentError("Requested mixed energy limit exceeds prepared tables; use source cutoff ≤ $(last(c.energies)) GeV or regenerate larger tables"))
    # This implementation deliberately consumes the entire requested segment,
    # including multiple events. A mode/energy boundary can shorten the result.
    remaining=distance*density; travelled=0.;proper=0.;weight=state.weight
    ceiling=energy_limit>0 ? min(energy_limit,last(c.energies)) : last(c.energies)
    # The caller's termination ceiling may exceed the data domain. Stop only
    # at its actual ceiling; never silently report a partial path as complete.
    state.energy>last(c.energies) && throw(DomainError(state.energy,"Ancestor energy exceeds mixed tables; regenerate with larger K_max"))
    exhausted=true
    for event in 1:100000
        if remaining<=eps(max(1.,distance*density));exhausted=false;break;end
        if k>=ceiling;exhausted=false;break;end
        ri=lookup(k,c.energies,c.range);hi=lookup(k,c.energies,c.hazard)
        range_to_ceiling=max(0.,lookup(ceiling,c.energies,c.range)-ri)
        endpoint_grammage=min(remaining,range_to_ceiling)
        targetr=ri+endpoint_grammage
        # Range -> energy -> range is not an exact floating-point roundtrip.
        # At a tiny final geometry remainder, retain the known path increment.
        endpoint=endpoint_grammage==range_to_ceiling ? ceiling :
            max(k,lookup(targetr,c.range,c.energies))
        optical_depth=randexp(rng)
        # Backward hazard increases with energy. An exponential optical depth
        # is independent of arbitrary geometry subdivision (memorylessness).
        eventh=hi+optical_depth
        occurs=eventh<lookup(endpoint,c.energies,c.hazard)
        pre=occurs ? max(k,lookup(eventh,c.hazard,c.energies)) : endpoint
        dx=occurs ? max(0.,lookup(pre,c.energies,c.range)-ri) : endpoint_grammage
        # Subtracting two large cumulative proper integrals also loses tiny
        # increments. Local endpoint quadrature is stable at that resolution.
        dt=if dx<=8eps(ri) || pre-k<=8eps(k)
            dx/density*c.mass/2*(inv(sqrt(k*(k+2c.mass)))+inv(sqrt(pre*(pre+2c.mass))))
        else
            (lookup(pre,c.energies,c.proper)-lookup(k,c.energies,c.proper))/density
        end
        weight*=lookup(pre,c.energies,c.stopping)/lookup(k,c.energies,c.stopping)*exp(-dt/c.ctau)
        travelled+=dx;remaining=max(0.,remaining-dx);proper+=dt;k=pre
        if occurs
            k,w=reverse_del_event(c,k,rng);weight*=w
            isfinite(weight)&&weight>=0 || error("Invalid adjoint hard-event weight")
            if weight==0;exhausted=false;break;end
        else
            exhausted=false
            break
        end
    end
    exhausted && error("Mixed transport exceeded 100000 hard events; segment incomplete")
    if weight>0 && k>=last(c.energies) && remaining>eps(max(1.,distance*density)) && (energy_limit<0 || k<energy_limit)
        throw(DomainError(k,"Mixed transport exhausted energy tables before segment completion; regenerate with larger K_max"))
    end
    d=travelled/density
    return State{Float64}(state.charge,k,state.distance+d,state.grammage+travelled,
        state.time+proper,weight,state.position-d*state.direction,state.direction,state.decayed)
end
end
