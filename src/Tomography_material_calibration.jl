"""
    calibrate_csda_material_correction(physics, shallow, matcfg, site, paths,
        baseline, fields, samples, mc_baseline, mc_fields, bins; ridge=1e-6)

Fit two material coefficients to paired log-flux contrasts at fixed direction.
`mc_fields[j,k]` is the MC flux for `fields[j]` and `bins[k]`;
`mc_baseline[k]` uses the same direction and the supplied baseline field.
Only explicitly supplied training fields/bins enter the fit. Independent fields,
directions and MC seeds must be retained by the caller for validation. The dry
CSDA correction is held fixed throughout. No geometric feature or intercept is
fitted. Returns coefficients and training diagnostics, then installs the fit.
This empirical correction supplies no claim about stochastic MC derivatives.
"""
function calibrate_csda_material_correction(physics, shallow, matcfg, site, paths,
        baseline, fields, samples, mc_baseline, mc_fields, bins; ridge::Real=1e-6)
    length(fields) >= 2 || throw(ArgumentError("at least two material training fields required"))
    size(mc_fields) == (length(fields), length(bins)) || throw(DimensionMismatch("MC field/bin matrix"))
    length(mc_baseline) == length(bins) || throw(DimensionMismatch("MC baseline"))
    ridge >= 0 && isfinite(ridge) || throw(ArgumentError("invalid ridge"))
    for w in (baseline, fields...)
        length(w) == length(shallow) || throw(DimensionMismatch("field length"))
        all(x -> isfinite(x) && 0 <= x <= 1, w) || throw(ArgumentError("mass fractions outside [0,1]"))
    end
    qcolumn(path, w) = _column_water_fraction(matcfg, path.remaining_rock_distance,
        [s.distance for s in path.segments], [s.cell_idx for s in path.segments],
        [shallow[s.cell_idx] for s in path.segments], w)
    old = get_csda_material_correction()
    rows = Vector{Float64}[]; targets = Float64[]; qs = Float64[]
    try
        set_csda_material_correction!(enabled=false)
        for (k,b) in enumerate(bins)
            path = paths[b]
            f0 = compute_directional_flux_csda(physics, shallow, matcfg, site, path, baseline, samples)
            m0 = mc_baseline[k]
            isfinite(f0) && f0 > 0 && isfinite(m0) && m0 > 0 || throw(ArgumentError("invalid baseline flux"))
            q0 = qcolumn(path, baseline)
            for (j,w) in enumerate(fields)
                f = compute_directional_flux_csda(physics, shallow, matcfg, site, path, w, samples)
                m = mc_fields[j,k]
                isfinite(f) && f > 0 && isfinite(m) && m > 0 || throw(ArgumentError("invalid training flux"))
                q = qcolumn(path,w)
                push!(rows, [q-q0, q*q-q0*q0]); push!(qs,q)
                push!(targets, log(m/m0)-log(f/f0))
            end
        end
        A = reduce(vcat, transpose.(rows))
        rank(A) == 2 || throw(ArgumentError("material training contrasts do not identify two coefficients"))
        coef = (A' * A + ridge*I) \ (A' * targets)
        prediction = A*coef
        set_csda_material_correction!(linear=coef[1], quadratic=coef[2])
        return (linear=coef[1], quadratic=coef[2], ridge=Float64(ridge),
            condition_number=cond(A), q_min=minimum(qs), q_max=maximum(qs),
            n_fields=length(fields), bins=collect(bins), n_pairs=length(targets),
            rms_log_before=sqrt(sum(abs2,targets)/length(targets)),
            rms_log_after=sqrt(sum(abs2,prediction-targets)/length(targets)),
            baseline=collect(baseline), training_fields=[collect(w) for w in fields])
    catch
        CSDA_MATERIAL_CORRECTION[] = old
        rethrow()
    end
end

"""Save the complete frozen dry and material correction with user-supplied provenance."""
function save_csda_corrections(path::AbstractString; metadata::AbstractDict=Dict{String,Any}())
    base = get_csda_correction(); material = get_csda_material_correction()
    data = Dict("schema_version"=>1,
        "csda"=>Dict(string(k)=>getfield(base,k) for k in fieldnames(CsdaCorrection)),
        "material"=>Dict(string(k)=>getfield(material,k) for k in fieldnames(CsdaMaterialCorrection)),
        "metadata"=>Dict(string(k)=>v for (k,v) in metadata))
    open(path,"w") do io
        TOML.print(io,data)
    end
    return path
end

"""Load and atomically install a version-1 complete correction artifact; return provenance."""
function load_csda_corrections!(path::AbstractString)
    data = TOML.parsefile(path)
    get(data,"schema_version",nothing) == 1 || throw(ArgumentError("unsupported correction schema"))
    b = data["csda"]; m = data["material"]
    # Validate all fields before changing either global correction.
    b["enabled"] isa Bool && m["enabled"] isa Bool || throw(ArgumentError("enabled must be Boolean"))
    bk = (:kappa_strag,:kappa_hard,:resid_a,:resid_b,:resid_c,:resid_d)
    bv = [Float64(b[string(k)]) for k in bk]
    mv = [Float64(m[k]) for k in ("linear","quadratic")]
    all(isfinite,bv) && all(isfinite,mv) || throw(ArgumentError("nonfinite correction coefficients"))
    newbase = CsdaCorrection(b["enabled"],bv...)
    newmaterial = CsdaMaterialCorrection(m["enabled"],mv...)
    CSDA_CORRECTION[] = newbase
    CSDA_MATERIAL_CORRECTION[] = newmaterial
    return get(data,"metadata",Dict{String,Any}())
end
