# Exact replacement closure extracted from frozen run_cached_stability_fit.py.
# Only wrapper parameters/return are added for isolated parity testing.
function build_exact_cached_flux_factory(physics, shallow_flags, matcfg, match_site, match_paths_v, energy_samples, args, mk_cfg)
    # Audited isolated implementation variant: exact last-value ray cache.
    args.papermatch_fit_mode == "stable" || error("cache variant requires stable fit")
    isempty(args.correction_artifact) && error("cache variant requires frozen artifact")
    cache_scalars(x) = ntuple(i -> getfield(x, i), fieldcount(typeof(x)))
    cache_guard = () -> (
        physics_identity=objectid(physics),
        base=cache_scalars(get_csda_correction()),
        material=cache_scalars(get_csda_material_correction()),
        config=cache_scalars(matcfg), site=cache_scalars(match_site),
        shallow=collect(shallow_flags),
        nodes=[(e.energy,e.charge,e.weight) for e in energy_samples],
        paths=[(p.zenith,p.azimuth,[(s.cell_idx,s.distance) for s in p.segments],
            p.distance_in_volume,p.surface_distance,p.remaining_rock_distance,
            p.surface_exit_z,p.valid) for p in match_paths_v])
    # Physics tables are immutable for this closure lifetime and identity-bound
    # by the outer audited physics-dump/source/runtime certificate.
    ray_cache = GuardedStabilityRayCachePrototype.GuardedRayCache(match_paths_v, cache_guard)
    flux_only_at = (ρ, wv) -> begin
        isequal(ρ,matcfg.rock_density) || error("cache variant requires pinned rock density")
        cfg = mk_cfg(ρ)
        forward = (i,w) -> compute_directional_flux_csda(physics, shallow_flags,
            cfg, match_site, match_paths_v[i], w, energy_samples)
        first(GuardedStabilityRayCachePrototype.evaluate!(ray_cache,wv,forward;
            threaded=args.threaded))
    end
    return flux_only_at,ray_cache
end
