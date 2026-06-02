# ===========================================================================
# Custom reverse-mode rules for the CSDA segment propagators.
#
# Each `propagate_csda_*_segment` advances (energy, weight) through an adaptive
# stepping loop with data-dependent control flow. Letting Zygote trace those
# loops node-by-node builds a huge per-bin tape (~100 MB/bin) whose allocation
# serialises the threads through GC. These `rrule`s collapse each segment to a
# single tape node: the pullback runs a few extra *primal* (no-tape) walks to
# finite-difference the smooth scalar map, and treats the weight exactly (the
# segment multiplies the incoming weight by an energy-only factor `G`, so
# `∂weight_out/∂weight_in = G` and `∂energy_out/∂weight_in = 0`).
#
# This keeps reverse-mode AD (Zygote) and the full physics — we only hand the
# differentiator an analytic-quality adjoint for the inner loop instead of
# making it trace every step. Parity with the traced path is covered in
# test/test_tomography.jl (sparse adjoint vs full Zygote vs finite difference).
# ===========================================================================

import ChainRulesCore
using ChainRulesCore: NoTangent, ZeroTangent, AbstractZero

@inline _as_real(x) = x isa AbstractZero ? 0.0 : Float64(x)

# Relative finite-difference step for the internal scalar derivatives. The maps
# are smooth (table interpolation is piecewise-linear in log-energy and the
# mixtures are linear in w), so a central difference is accurate to ~1e-9.
const _RRULE_FD_REL = 1.0e-6

# --- rock/water cell segment: differentiable in (w, energy, weight) ---------
function ChainRulesCore.rrule(::typeof(propagate_csda_cell_segment),
                              physics, w, shallow::Bool, matcfg::MaterialConfig,
                              distance::Float64, energy, weight)
    e_out, w_out, ok = propagate_csda_cell_segment(physics, w, shallow, matcfg, distance, energy, weight)
    wf = Float64(w); ef = Float64(energy); wt = Float64(weight)
    function cell_segment_pullback(Δ)
        Δe = _as_real(Δ[1]); Δw = _as_real(Δ[2])   # cotangents on (energy_out, weight_out); ok carries none
        if !ok || wt == 0.0 || !(isfinite(Δe) && isfinite(Δw))
            return (NoTangent(), NoTangent(), ZeroTangent(), NoTangent(), NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent())
        end
        hw = _RRULE_FD_REL
        he = _RRULE_FD_REL * max(abs(ef), 1.0)
        # Re-run the primal with weight = 1 to isolate the energy-only weight
        # factor G and the energy map; partials by central finite difference.
        ePw, GPw, _ = propagate_csda_cell_segment(physics, wf + hw, shallow, matcfg, distance, ef, 1.0)
        eMw, GMw, _ = propagate_csda_cell_segment(physics, wf - hw, shallow, matcfg, distance, ef, 1.0)
        ePe, GPe, _ = propagate_csda_cell_segment(physics, wf, shallow, matcfg, distance, ef + he, 1.0)
        eMe, GMe, _ = propagate_csda_cell_segment(physics, wf, shallow, matcfg, distance, ef - he, 1.0)
        deout_dw = (ePw - eMw) / (2hw); dG_dw = (GPw - GMw) / (2hw)
        deout_de = (ePe - eMe) / (2he); dG_de = (GPe - GMe) / (2he)
        G = w_out / wt
        w̄ = Δe * deout_dw + Δw * wt * dG_dw
        ē = Δe * deout_de + Δw * wt * dG_de
        w̄eight = Δw * G
        return (NoTangent(), NoTangent(), w̄, NoTangent(), NoTangent(), NoTangent(), ē, w̄eight)
    end
    return (e_out, w_out, ok), cell_segment_pullback
end

# --- fixed-material segment (rock remainder): diff in (energy, weight) -------
function ChainRulesCore.rrule(::typeof(propagate_csda_uniform_segment),
                              physics, material::Int, density::Float64,
                              distance::Float64, energy, weight)
    e_out, w_out, ok = propagate_csda_uniform_segment(physics, material, density, distance, energy, weight)
    ef = Float64(energy); wt = Float64(weight)
    function uniform_segment_pullback(Δ)
        Δe = _as_real(Δ[1]); Δw = _as_real(Δ[2])
        if !ok || wt == 0.0 || !(isfinite(Δe) && isfinite(Δw))
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent())
        end
        he = _RRULE_FD_REL * max(abs(ef), 1.0)
        ePe, GPe, _ = propagate_csda_uniform_segment(physics, material, density, distance, ef + he, 1.0)
        eMe, GMe, _ = propagate_csda_uniform_segment(physics, material, density, distance, ef - he, 1.0)
        deout_de = (ePe - eMe) / (2he); dG_de = (GPe - GMe) / (2he)
        G = w_out / wt
        ē = Δe * deout_de + Δw * wt * dG_de
        w̄eight = Δw * G
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), ē, w̄eight)
    end
    return (e_out, w_out, ok), uniform_segment_pullback
end

# --- atmosphere segment: diff in (energy, weight) ---------------------------
function ChainRulesCore.rrule(::typeof(propagate_csda_air_segment),
                              physics, air_material::Int, site::SiteConfig,
                              start_z::Float64, cos_theta::Float64,
                              distance::Float64, energy, weight)
    e_out, w_out, ok = propagate_csda_air_segment(physics, air_material, site, start_z, cos_theta, distance, energy, weight)
    ef = Float64(energy); wt = Float64(weight)
    function air_segment_pullback(Δ)
        Δe = _as_real(Δ[1]); Δw = _as_real(Δ[2])
        if !ok || wt == 0.0 || !(isfinite(Δe) && isfinite(Δw))
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent())
        end
        he = _RRULE_FD_REL * max(abs(ef), 1.0)
        ePe, GPe, _ = propagate_csda_air_segment(physics, air_material, site, start_z, cos_theta, distance, ef + he, 1.0)
        eMe, GMe, _ = propagate_csda_air_segment(physics, air_material, site, start_z, cos_theta, distance, ef - he, 1.0)
        deout_de = (ePe - eMe) / (2he); dG_de = (GPe - GMe) / (2he)
        G = w_out / wt
        ē = Δe * deout_de + Δw * wt * dG_de
        w̄eight = Δw * G
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), ē, w̄eight)
    end
    return (e_out, w_out, ok), air_segment_pullback
end
