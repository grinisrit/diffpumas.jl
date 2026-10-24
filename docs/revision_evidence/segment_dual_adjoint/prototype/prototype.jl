# Unapplied, process-local prototype: no Float64 method or physical model changes.
using DiffPumas, Zygote, Test, LinearAlgebra, Statistics
const FD = Zygote.ForwardDiff
const T = DiffPumas.Tomography
const P = DiffPumas.Physics
const G = DiffPumas.Geometry
const _pv(x::FD.Dual) = FD.value(x)
const _pv(x::Real) = x

@inline function P.interpolate_table_fast(x::FD.Dual, lx::FD.Dual,
        xs::Vector{Float64}, lxs::Vector{Float64}, ys::Vector{Float64})
    n=length(xs)
    _pv(x)<=xs[1] && return zero(x)+ys[1]
    _pv(x)>=xs[n] && return zero(x)+ys[n]
    lo,hi=1,n
    while lo<hi-1
        mid=(lo+hi)>>1
        if _pv(lx)<lxs[mid];hi=mid;else;lo=mid;end
    end
    a=(lx-lxs[lo])/(lxs[lo+1]-lxs[lo])
    return ys[lo]+a*(ys[lo+1]-ys[lo])
end
@inline function P.property_stopping_power(ph::P.PhysicsTables{Float64}, mode::P.EnergyLossMode,
        material::Int, energy::FD.Dual)
    tab=ph.tables[material]
    ys=mode==P.ENERGY_LOSS_MIXED ? tab.mixed_stopping_power : tab.csda_stopping_power
    return P.interpolate_table_fast(energy,log(energy),tab.energies,tab.log_energies,ys)
end
@inline function G.compute_decay_weight_from_path(ph::P.PhysicsTables{Float64}, ds::Real, energy::FD.Dual)
    gamma=one(energy)+energy/ph.mass
    beta=sqrt(max(one(energy)-one(energy)/gamma^2,1e-12))
    return exp(-ds/(beta*gamma*ph.ctau))
end
@inline G.compute_decay_weight_from_path(ph::P.PhysicsTables{Float64},ds::FD.Dual,energy::Float64) =
    G.compute_decay_weight_from_path(ph,ds,energy+zero(ds))

function cell_tangent(ph,w,sh,mat,ds,e)
    ed,gd,ok=T.propagate_csda_cell_segment(ph,FD.Dual(w,1.,0.),sh,mat,ds,FD.Dual(e,0.,1.),1.)
    ok || error("invalid transport in prototype")
    return (_pv(ed),_pv(gd),FD.partials(ed)[1],FD.partials(ed)[2],FD.partials(gd)[1],FD.partials(gd)[2])
end
function cell_fd(ph,w,sh,mat,ds,e;h=1e-6)
    ep,gp,_=T.propagate_csda_cell_segment(ph,w+h,sh,mat,ds,e,1.)
    em,gm,_=T.propagate_csda_cell_segment(ph,w-h,sh,mat,ds,e,1.)
    he=h*max(abs(e),1.)
    ee,ge,_=T.propagate_csda_cell_segment(ph,w,sh,mat,ds,e+he,1.)
    ef,gf,_=T.propagate_csda_cell_segment(ph,w,sh,mat,ds,e-he,1.)
    return ((ep-em)/(2h),(ee-ef)/(2he),(gp-gm)/(2h),(ge-gf)/(2he))
end

if abspath(PROGRAM_FILE)==@__FILE__
    ph=load_physics(ARGS[1]);ph===nothing && error("physics dump failed")
    rock=get_material_index(ph,"StandardRock");rock<1 && error("StandardRock absent")
    water=get_material_index(ph,"Water"); air=get_material_index(ph,"Air")
    mat=MaterialConfig(rock,water,air,-1,2710.,1000.,0.,0.)
    println("ForwardDiff version ",pkgversion(FD))
    cases=[(w,e,ds) for w in (0.,.1147192654,.8), e in (1.,100.,10000.),ds in (30.,333.)]
    largest=zeros(4)
    @testset "Dual segment primal/derivative feasibility" begin
        for (w,e,ds) in cases
            f=cell_tangent(ph,w,false,mat,ds,e)
            primal=T.propagate_csda_cell_segment(ph,w,false,mat,ds,e,1.)
            @test f[1]≈primal[1] rtol=2e-14
            @test f[2]≈primal[2] rtol=2e-14
            fd=cell_fd(ph,w,false,mat,ds,e;h=1e-7)
            for k in 1:4
                rel=abs(f[k+2]-fd[k])/max(abs(f[k+2]),abs(fd[k]),1e-20)
                largest[k]=max(largest[k],rel)
            end
        end
        # Uniform and atmospheric walks exercise both Float64/Dual path-length dispatch.
        for fun in (e->T.propagate_csda_uniform_segment(ph,rock,2710.,333.,e,1.),
                    e->T.propagate_csda_air_segment(ph,air,SiteConfig(963.,30000.),1000.,.8,10000.,e,1.))
            e,g,ok=fun(FD.Dual(123.4567,1.)); a,b,valid=fun(123.4567)
            @test ok==valid
            @test _pv(e)≈a rtol=2e-14
            @test _pv(g)≈b rtol=2e-14
            h=1e-6;hi=fun(123.4567 + h);lo=fun(123.4567 - h)
            @test FD.partials(e)[1]≈(hi[1]-lo[1])/(2h) rtol=1e-5
            @test FD.partials(g)[1]≈(hi[2]-lo[2])/(2h) rtol=1e-4 atol=1e-12
        end
    end
    println("largest relative local tangent/FD differences (dE/dw,dE/de,dG/dw,dG/de): ",largest)
    w,e,ds=.1147192654,100.,333.
    cell_tangent(ph,w,false,mat,ds,e);cell_fd(ph,w,false,mat,ds,e)
    n=500
    dual_seconds=@elapsed for _ in 1:n;cell_tangent(ph,w,false,mat,ds,e);end
    fd_seconds=@elapsed for _ in 1:n;cell_fd(ph,w,false,mat,ds,e);end
    dual_bytes=@allocated cell_tangent(ph,w,false,mat,ds,e)
    fd_bytes=@allocated cell_fd(ph,w,false,mat,ds,e)
    println("benchmark n=",n," dual_seconds=",dual_seconds," fd4walk_seconds=",fd_seconds,
        " dual_bytes=",dual_bytes," fd_bytes=",fd_bytes," speedup=",fd_seconds/dual_seconds)
end
