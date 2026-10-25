# Instrumented arithmetic copy of src/Tomography.jl CSDA propagation.
# Adds new names only; production functions remain untouched. Serial trace only.
# Every instrumented flux must match production value to <=1e-13 relative.
@eval DiffPumas.Tomography begin
const ROUND3_EVENTS=Ref(Any[])
const ROUND3_SAMPLE=Ref(0)
const ROUND3_SEGMENT=Ref(0)
function round3_trace_cell_segment(physics, w, shallow::Bool, matcfg::MaterialConfig,
                                     distance::Float64, energy, weight)
    density = cell_density(w, shallow, matcfg)
    remaining = distance
    trace_step=0
    while _csda_value(remaining) > 1e-8
        dedx0 = cell_stopping_power(physics, w, shallow, matcfg, energy)
        (!isfinite(dedx0) || _csda_value(dedx0) <= 0.0) && return (energy, zero(weight), false)

        limiter_density=density
        grammage_rel = CSDA_MAX_RELATIVE_GAIN * _csda_max(energy, 1e-6) / dedx0
        grammage_step = _csda_min(remaining * density, _csda_max(grammage_rel, 1e-6), CSDA_MAX_STEP_M * density)
        ds = _csda_min(remaining, grammage_step / _csda_max(density, 1e-12))
        _csda_value(ds) <= 0.0 && return (energy, zero(weight), false)

        grammage = ds * density
        energy_predict = energy + dedx0 * grammage
        dedx1 = cell_stopping_power(physics, w, shallow, matcfg, energy_predict)
        dE = 0.5 * (dedx0 + dedx1) * grammage
        energy_next = energy + dE

        (!isfinite(energy_next) || _csda_value(energy_next) >= 1e12) && return (energy_next, zero(weight), false)

        dedx_next = cell_stopping_power(physics, w, shallow, matcfg, energy_next)
        weight = weight * (dedx_next / _csda_max(dedx0, 1e-30))
        weight = weight * compute_decay_weight_from_path(physics, ds, 0.5 * (energy + energy_next))

        trace_step+=1
        idxs=Int[searchsortedlast(table.log_energies,log(Float64(energy))) for table in physics.tables]
        idxp=Int[searchsortedlast(table.log_energies,log(Float64(energy_predict))) for table in physics.tables]
        idxn=Int[searchsortedlast(table.log_energies,log(Float64(energy_next))) for table in physics.tables]
        candidates = ROUND3_SEGMENT[]==-2 ? [remaining,CSDA_MAX_STEP_M,grammage_rel/_csda_max(limiter_density,1e-12)] :
            [remaining*density,_csda_max(grammage_rel,1e-6),CSDA_MAX_STEP_M*density]
        branch=argmin(candidates)
        ties=findall(==(minimum(candidates)),candidates)
        push!(ROUND3_EVENTS[],(sample=ROUND3_SAMPLE[],segment=ROUND3_SEGMENT[],step=trace_step,
            energy=Float64(energy),predict=Float64(energy_predict),next=Float64(energy_next),
            remaining=Float64(remaining),ds=Float64(ds),intervals=vcat(idxs,idxp,idxn),
            limiter=string(branch),limiter_density=Float64(limiter_density),step_candidates=Float64.(candidates),tied_branches=ties,
            final_step=ds>=remaining,air_floor=ROUND3_SEGMENT[]==-2 && ds==1e-3))
        energy = energy_next
        remaining -= ds
    end
    return energy, weight, true
end


function round3_trace_uniform_segment(physics, material::Int, density::Float64,
                                        distance::Float64, energy, weight)
    remaining = distance
    trace_step=0
    while _csda_value(remaining) > 1e-8
        dedx0 = property_stopping_power(physics, ENERGY_LOSS_CSDA, material, energy)
        (!isfinite(dedx0) || _csda_value(dedx0) <= 0.0) && return (energy, zero(weight), false)

        limiter_density=density
        grammage_rel = CSDA_MAX_RELATIVE_GAIN * _csda_max(energy, 1e-6) / dedx0
        grammage_step = _csda_min(remaining * density, _csda_max(grammage_rel, 1e-6), CSDA_MAX_STEP_M * density)
        ds = _csda_min(remaining, grammage_step / _csda_max(density, 1e-12))
        _csda_value(ds) <= 0.0 && return (energy, zero(weight), false)

        grammage = ds * density
        energy_predict = energy + dedx0 * grammage
        dedx1 = property_stopping_power(physics, ENERGY_LOSS_CSDA, material, energy_predict)
        dE = 0.5 * (dedx0 + dedx1) * grammage
        energy_next = energy + dE

        (!isfinite(energy_next) || _csda_value(energy_next) >= 1e12) && return (energy_next, zero(weight), false)

        dedx_next = property_stopping_power(physics, ENERGY_LOSS_CSDA, material, energy_next)
        weight = weight * (dedx_next / _csda_max(dedx0, 1e-30))
        weight = weight * compute_decay_weight_from_path(physics, ds, 0.5 * (energy + energy_next))

        trace_step+=1
        idxs=Int[searchsortedlast(table.log_energies,log(Float64(energy))) for table in physics.tables]
        idxp=Int[searchsortedlast(table.log_energies,log(Float64(energy_predict))) for table in physics.tables]
        idxn=Int[searchsortedlast(table.log_energies,log(Float64(energy_next))) for table in physics.tables]
        candidates = ROUND3_SEGMENT[]==-2 ? [remaining,CSDA_MAX_STEP_M,grammage_rel/_csda_max(limiter_density,1e-12)] :
            [remaining*density,_csda_max(grammage_rel,1e-6),CSDA_MAX_STEP_M*density]
        branch=argmin(candidates)
        ties=findall(==(minimum(candidates)),candidates)
        push!(ROUND3_EVENTS[],(sample=ROUND3_SAMPLE[],segment=ROUND3_SEGMENT[],step=trace_step,
            energy=Float64(energy),predict=Float64(energy_predict),next=Float64(energy_next),
            remaining=Float64(remaining),ds=Float64(ds),intervals=vcat(idxs,idxp,idxn),
            limiter=string(branch),limiter_density=Float64(limiter_density),step_candidates=Float64.(candidates),tied_branches=ties,
            final_step=ds>=remaining,air_floor=ROUND3_SEGMENT[]==-2 && ds==1e-3))
        energy = energy_next
        remaining -= ds
    end
    return energy, weight, true
end


function round3_trace_air_segment(physics, air_material::Int, site::SiteConfig,
                                    start_z::Float64, cos_theta::Float64,
                                    distance::Float64, energy, weight)
    remaining = distance
    traversed = 0.0
    trace_step=0
    while _csda_value(remaining) > 1e-8
        altitude_mid = site.detector_elevation + start_z +
                       cos_theta * (traversed + 0.5 * _csda_min(remaining, CSDA_MAX_STEP_M))
        density = locals_air_at_altitude(altitude_mid).density
        dedx0 = property_stopping_power(physics, ENERGY_LOSS_CSDA, air_material, energy)
        (!isfinite(dedx0) || _csda_value(dedx0) <= 0.0) && return (energy, zero(weight), false)

        limiter_density=density
        grammage_rel = CSDA_MAX_RELATIVE_GAIN * _csda_max(energy, 1e-6) / dedx0
        ds = _csda_min(remaining, CSDA_MAX_STEP_M, grammage_rel / _csda_max(density, 1e-12))
        ds = _csda_max(ds, 1e-3)

        altitude_mid = site.detector_elevation + start_z + cos_theta * (traversed + 0.5 * ds)
        density = locals_air_at_altitude(altitude_mid).density
        grammage = ds * density
        energy_predict = energy + dedx0 * grammage
        dedx1 = property_stopping_power(physics, ENERGY_LOSS_CSDA, air_material, energy_predict)
        dE = 0.5 * (dedx0 + dedx1) * grammage
        energy_next = energy + dE

        (!isfinite(energy_next) || _csda_value(energy_next) >= 1e12) && return (energy_next, zero(weight), false)

        dedx_next = property_stopping_power(physics, ENERGY_LOSS_CSDA, air_material, energy_next)
        weight = weight * (dedx_next / _csda_max(dedx0, 1e-30))
        weight = weight * compute_decay_weight_from_path(physics, ds, 0.5 * (energy + energy_next))

        trace_step+=1
        idxs=Int[searchsortedlast(table.log_energies,log(Float64(energy))) for table in physics.tables]
        idxp=Int[searchsortedlast(table.log_energies,log(Float64(energy_predict))) for table in physics.tables]
        idxn=Int[searchsortedlast(table.log_energies,log(Float64(energy_next))) for table in physics.tables]
        candidates = ROUND3_SEGMENT[]==-2 ? [remaining,CSDA_MAX_STEP_M,grammage_rel/_csda_max(limiter_density,1e-12)] :
            [remaining*density,_csda_max(grammage_rel,1e-6),CSDA_MAX_STEP_M*density]
        branch=argmin(candidates)
        ties=findall(==(minimum(candidates)),candidates)
        push!(ROUND3_EVENTS[],(sample=ROUND3_SAMPLE[],segment=ROUND3_SEGMENT[],step=trace_step,
            energy=Float64(energy),predict=Float64(energy_predict),next=Float64(energy_next),
            remaining=Float64(remaining),ds=Float64(ds),intervals=vcat(idxs,idxp,idxn),
            limiter=string(branch),limiter_density=Float64(limiter_density),step_candidates=Float64.(candidates),tied_branches=ties,
            final_step=ds>=remaining,air_floor=ROUND3_SEGMENT[]==-2 && ds==1e-3))
        energy = energy_next
        traversed += ds
        remaining -= ds
    end
    return energy, weight, true
end

function round3_trace_flux(physics, shallow_flags::AbstractVector{Bool},
                                       matcfg::MaterialConfig, site::SiteConfig,
                                       path::DirectionalPath,
                                       water_fractions::AbstractVector{<:Real},
                                       energy_samples::Vector{EnergySample})
    path.valid || return NaN
    cos_theta = cosd(path.zenith)
    cos_theta <= 0.0 && return NaN
    # Direct scalar indexing of the full field (no array mutation), so this is
    # itself Zygote-differentiable over the full vector; used as the forward
    # value and as an independent full-vector AD reference in tests.
    flux_sum = zero(eltype(water_fractions)) + 0.0
    @inbounds for (sample_index,sample) in enumerate(energy_samples)
        ROUND3_SAMPLE[]=sample_index
        energy = sample.energy
        weight = 1.0
        Vacc = zero(eltype(water_fractions)) + 0.0
        Hacc = zero(eltype(water_fractions)) + 0.0
        ok = true
        for (seg_index,seg) in enumerate(path.segments)
            ROUND3_SEGMENT[]=seg_index
            e_in = energy
            energy, weight, ok = round3_trace_cell_segment(
                physics, water_fractions[seg.cell_idx], shallow_flags[seg.cell_idx],
                matcfg, seg.distance, energy, weight)
            ok || break
            e_rep = 0.5 * (e_in + energy)
            wseg = water_fractions[seg.cell_idx]
            shseg = shallow_flags[seg.cell_idx]
            gx = seg.distance * cell_density(wseg, shseg, matcfg)
            Vacc += cell_straggling(physics, wseg, shseg, matcfg, e_rep) * gx
            Hacc += cell_hardloss(physics, wseg, shseg, matcfg, e_rep) * gx
        end
        if ok && path.remaining_rock_distance > 1e-8
            e_in = energy
            ROUND3_SEGMENT[]=-1
            energy, weight, ok = round3_trace_uniform_segment(
                physics, matcfg.rock_material, matcfg.rock_density,
                path.remaining_rock_distance, energy, weight)
            if ok
                e_rep = 0.5 * (e_in + energy)
                gx = path.remaining_rock_distance * matcfg.rock_density
                Vacc += property_straggling(physics, matcfg.rock_material, e_rep) * gx
                Hacc += rock_hardloss(physics, matcfg, e_rep) * gx
            end
        end
        if ok
            air_distance = max(0.0, site.primary_altitude - path.surface_exit_z) / max(cos_theta, 1e-3)
            ROUND3_SEGMENT[]=-2
            energy, weight, ok = round3_trace_air_segment(
                physics, matcfg.air_material, site, path.surface_exit_z, cos_theta,
                air_distance, energy, weight)
        end
        if ok
            corr = csda_var_correction(energy, Vacc, Hacc, cos_theta)
            flux_single = weight * corr * flux_gccly(cos_theta, energy, sample.charge)
        else
            flux_single = zero(eltype(water_fractions))
        end
        flux_sum += 2.0 * sample.weight * flux_single
    end
    return csda_geom_multiplier(path) * csda_material_multiplier(path, shallow_flags, matcfg,
        water_fractions) * flux_sum / length(energy_samples)
end

end
