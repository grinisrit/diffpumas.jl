# Segment pullbacks collapse adaptive transport to one reverse-mode node. The
# default computes exact local forward-mode tangents through the executed
# piecewise-smooth walk, without a large Zygote tape or a finite-difference step.
# This differentiates the discretised deterministic model, not stochastic MC.
# At interpolation/adaptive knots a classical derivative need not exist; branch
# choices use primal values only and deterministic ties (see _csda_min/max).
import ChainRulesCore
using ChainRulesCore: NoTangent, ZeroTangent, AbstractZero

@inline _as_real(x) = x isa AbstractZero ? 0.0 : Float64(x)
const _RRULE_FD_REL = Ref(1.0e-6)
const _ADJOINT_DERIVATIVE_MODE = Ref(:forwarddiff)

"""Local segment adjoint mode: `:forwarddiff` (default) or legacy `:finite_difference`."""
get_adjoint_derivative_mode() = _ADJOINT_DERIVATIVE_MODE[]
"""Select local segment adjoints. Set only between evaluations, never concurrently."""
function set_adjoint_derivative_mode!(mode::Symbol)
    mode in (:forwarddiff, :finite_difference) || throw(ArgumentError("unknown adjoint derivative mode"))
    _ADJOINT_DERIVATIVE_MODE[] = mode
end
"""Legacy central-difference relative step; unused in the default forward-mode adjoint."""
get_adjoint_fd_step() = _RRULE_FD_REL[]
"""Set the legacy FD step without changing the adjoint mode. Not safe during evaluations."""
function set_adjoint_fd_step!(step::Real)
    isfinite(step) && 0 < step < 0.1 || throw(ArgumentError("adjoint FD step must be finite and in (0, 0.1)"))
    _RRULE_FD_REL[] = Float64(step)
end
export get_adjoint_fd_step, set_adjoint_fd_step!
export get_adjoint_derivative_mode, set_adjoint_derivative_mode!

# Unit incoming weight isolates G, so d(weight_out)/d(weight_in)=G even when
# the actual incoming weight is zero. Both input partials share one dual walk.
function _cell_segment_partials(physics, w, shallow, matcfg, distance, energy)
    if get_adjoint_derivative_mode() === :forwarddiff
        wd = ForwardDiff.Dual(w, 1.0, 0.0)
        ed = ForwardDiff.Dual(energy, 0.0, 1.0)
        eo, g, _ = propagate_csda_cell_segment(physics, wd, shallow, matcfg, distance, ed, one(ed))
        return ForwardDiff.value(g), ForwardDiff.partials(eo)[1], ForwardDiff.partials(eo)[2],
               ForwardDiff.partials(g)[1], ForwardDiff.partials(g)[2]
    end
    hw = _RRULE_FD_REL[]; he = hw * max(abs(energy), 1.0)
    _, g, _ = propagate_csda_cell_segment(physics, w, shallow, matcfg, distance, energy, 1.0)
    epw, gpw, _ = propagate_csda_cell_segment(physics, w+hw, shallow, matcfg, distance, energy, 1.0)
    emw, gmw, _ = propagate_csda_cell_segment(physics, w-hw, shallow, matcfg, distance, energy, 1.0)
    epe, gpe, _ = propagate_csda_cell_segment(physics, w, shallow, matcfg, distance, energy+he, 1.0)
    eme, gme, _ = propagate_csda_cell_segment(physics, w, shallow, matcfg, distance, energy-he, 1.0)
    return g, (epw-emw)/(2hw), (epe-eme)/(2he), (gpw-gmw)/(2hw), (gpe-gme)/(2he)
end
function _energy_segment_partials(propagate, energy)
    if get_adjoint_derivative_mode() === :forwarddiff
        eo, g, _ = propagate(ForwardDiff.Dual(energy, 1.0))
        return ForwardDiff.value(g), ForwardDiff.partials(eo)[1], ForwardDiff.partials(g)[1]
    end
    he = _RRULE_FD_REL[] * max(abs(energy), 1.0)
    _, g, _ = propagate(energy)
    ep, gp, _ = propagate(energy+he)
    em, gm, _ = propagate(energy-he)
    return g, (ep-em)/(2he), (gp-gm)/(2he)
end

# Invalid transport retains the existing stopped-path convention: every input
# cotangent is zero. A valid zero-weight input is NOT an invalid transport.
function ChainRulesCore.rrule(::typeof(propagate_csda_cell_segment),
        physics, w, shallow::Bool, matcfg::MaterialConfig,
        distance::Float64, energy, weight)
    out = propagate_csda_cell_segment(physics, w, shallow, matcfg, distance, energy, weight)
    wf, ef, wt = Float64(w), Float64(energy), Float64(weight)
    function cell_segment_pullback(delta)
        de, dg = _as_real(delta[1]), _as_real(delta[2])
        if !out[3] || !(isfinite(de) && isfinite(dg))
            return (NoTangent(),NoTangent(),ZeroTangent(),NoTangent(),NoTangent(),NoTangent(),ZeroTangent(),ZeroTangent())
        end
        g, ew, ee, gw, ge = _cell_segment_partials(physics,wf,shallow,matcfg,distance,ef)
        return (NoTangent(),NoTangent(),de*ew+dg*wt*gw,NoTangent(),NoTangent(),NoTangent(),de*ee+dg*wt*ge,dg*g)
    end
    return out, cell_segment_pullback
end
function ChainRulesCore.rrule(::typeof(propagate_csda_uniform_segment),
        physics, material::Int, density::Float64, distance::Float64, energy, weight)
    out = propagate_csda_uniform_segment(physics,material,density,distance,energy,weight)
    ef, wt = Float64(energy), Float64(weight)
    function uniform_segment_pullback(delta)
        de, dg = _as_real(delta[1]), _as_real(delta[2])
        if !out[3] || !(isfinite(de) && isfinite(dg))
            return (NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),ZeroTangent(),ZeroTangent())
        end
        g, ee, ge = _energy_segment_partials(e->propagate_csda_uniform_segment(physics,material,density,distance,e,one(e)),ef)
        return (NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),de*ee+dg*wt*ge,dg*g)
    end
    return out, uniform_segment_pullback
end
function ChainRulesCore.rrule(::typeof(propagate_csda_air_segment),
        physics, air_material::Int, site::SiteConfig, start_z::Float64,
        cos_theta::Float64, distance::Float64, energy, weight)
    out = propagate_csda_air_segment(physics,air_material,site,start_z,cos_theta,distance,energy,weight)
    ef, wt = Float64(energy), Float64(weight)
    function air_segment_pullback(delta)
        de, dg = _as_real(delta[1]), _as_real(delta[2])
        if !out[3] || !(isfinite(de) && isfinite(dg))
            return (NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),ZeroTangent(),ZeroTangent())
        end
        g, ee, ge = _energy_segment_partials(e->propagate_csda_air_segment(physics,air_material,site,start_z,cos_theta,distance,e,one(e)),ef)
        return (NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),de*ee+dg*wt*ge,dg*g)
    end
    return out, air_segment_pullback
end
