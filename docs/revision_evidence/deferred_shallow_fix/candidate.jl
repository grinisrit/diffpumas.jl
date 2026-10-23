function mc_cell_mixture_and_density_candidate(w::Float64, shallow::Bool, matcfg::MaterialConfig)
    if is_shallow_porous(shallow, matcfg)
        porous_frac = 0.5 * (1.0 - w)
        rock_frac = 0.5 * (1.0 - w)
        mix = MaterialMixture(
            [matcfg.porous_material, matcfg.rock_material, matcfg.water_material],
            [porous_frac, rock_frac, w])
        # Mass fractions combine the specific volumes (harmonic density), matching
        # `cell_density` and the PUMAS composite convention (Eq. rho-mixture).
        density = 1.0 / (porous_frac / matcfg.porous_density +
                         rock_frac / matcfg.rock_density + w / matcfg.water_density)
        return mix, density
    end
    if w <= 1e-12
        return MaterialMixture(matcfg.rock_material), matcfg.rock_density
    end
    mix = MaterialMixture([matcfg.rock_material, matcfg.water_material], [1.0 - w, w])
    density = 1.0 / ((1.0 - w) / matcfg.rock_density + w / matcfg.water_density)
    return mix, density
end
