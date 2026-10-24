# Process-local replacement rules for adversarial test. Never included by tracked code.
import ChainRulesCore
const CRC=ChainRulesCore
function energy_tangent(fun,e)
    ed,gd,ok=fun(FD.Dual(e,1.))
    ok || error("invalid unit-weight tangent transport")
    return _pv(ed),_pv(gd),FD.partials(ed)[1],FD.partials(gd)[1]
end
function CRC.rrule(::typeof(T.propagate_csda_cell_segment),ph,w,sh::Bool,mat::T.MaterialConfig,ds::Float64,e,wt)
    out=T.propagate_csda_cell_segment(ph,w,sh,mat,ds,e,wt)
    function pb(delta)
        de=T._as_real(delta[1]);dg=T._as_real(delta[2])
        if !out[3] || !(isfinite(de)&&isfinite(dg))
            return (CRC.NoTangent(),CRC.NoTangent(),CRC.ZeroTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.ZeroTangent(),CRC.ZeroTangent())
        end
        _,g,ew,ee,gw,ge=cell_tangent(ph,Float64(w),sh,mat,ds,Float64(e))
        return (CRC.NoTangent(),CRC.NoTangent(),de*ew+dg*wt*gw,CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),de*ee+dg*wt*ge,dg*g)
    end
    return out,pb
end
function CRC.rrule(::typeof(T.propagate_csda_uniform_segment),ph,mat::Int,rho::Float64,ds::Float64,e,wt)
    out=T.propagate_csda_uniform_segment(ph,mat,rho,ds,e,wt)
    function pb(delta)
        de=T._as_real(delta[1]);dg=T._as_real(delta[2])
        if !out[3] || !(isfinite(de)&&isfinite(dg))
            return (CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.ZeroTangent(),CRC.ZeroTangent())
        end
        _,g,ee,ge=energy_tangent(x->T.propagate_csda_uniform_segment(ph,mat,rho,ds,x,1.),Float64(e))
        return (CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),de*ee+dg*wt*ge,dg*g)
    end
    return out,pb
end
function CRC.rrule(::typeof(T.propagate_csda_air_segment),ph,mat::Int,site::T.SiteConfig,z::Float64,cos::Float64,ds::Float64,e,wt)
    out=T.propagate_csda_air_segment(ph,mat,site,z,cos,ds,e,wt)
    function pb(delta)
        de=T._as_real(delta[1]);dg=T._as_real(delta[2])
        if !out[3] || !(isfinite(de)&&isfinite(dg))
            return (CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.ZeroTangent(),CRC.ZeroTangent())
        end
        _,g,ee,ge=energy_tangent(x->T.propagate_csda_air_segment(ph,mat,site,z,cos,ds,x,1.),Float64(e))
        return (CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),CRC.NoTangent(),de*ee+dg*wt*ge,dg*g)
    end
    return out,pb
end
