# Tomography module tests: CSDA forward model, sparse AD Jacobian (vs Zygote,
# vs finite-difference, vs stochastic MC finite-difference), inverse solvers,
# image-quality metrics, and the resolution estimator.
#
# Uses small hand-built `DirectionalPath`s (no TetGen / topography) so the suite
# is fully deterministic and runs in a few seconds.

using DiffPumas
using DiffPumas.Tomography
import Zygote
using SparseArrays
using Statistics
using Random
using LinearAlgebra

@testset "Tomography" begin

    # --- shared fixture -----------------------------------------------------
    physics = create_physics(MUON; n_energies=80, K_min=0.01, K_max=1e6,
                             materials=[STANDARD_ROCK, AIR, WATER])
    rock = get_material_index(physics, "StandardRock")
    water = get_material_index(physics, "Water")
    air = get_material_index(physics, "Air")
    matcfg = MaterialConfig(rock, water, air, -1,
        Float64(physics.tables[rock].density),
        Float64(physics.tables[water].density),
        0.0, 0.0)
    site = SiteConfig(1000.0, 12000.0)

    n_cells = 4
    shallow = falses(n_cells)
    # A moderate-overburden path (good flux + finite gradients)
    path = DirectionalPath(20.0, 0.0,
        [CellSegment(1, 30.0), CellSegment(2, 40.0), CellSegment(3, 30.0)],
        100.0, 100.0, 0.0, 60.0, true)
    # A second path crossing a different cell subset
    path2 = DirectionalPath(35.0, 90.0,
        [CellSegment(2, 50.0), CellSegment(4, 35.0)],
        85.0, 85.0, 0.0, 55.0, true)
    paths = [path, path2]
    wfield = [0.10, 0.20, 0.05, 0.15]
    samples = sample_energy_set(6, 1e-3, 1e6, 42)

    @testset "1. sparse adjoint vs Zygote full-vector" begin
        for p in paths
            fv, cells, grad = directional_flux_and_grad_csda(physics, shallow, matcfg, site, p, wfield, samples)
            gz = Zygote.gradient(ww -> compute_directional_flux_csda(physics, shallow, matcfg, site, p, ww, samples), wfield)[1]
            @test gz !== nothing
            # values match on-path cells tightly
            for (k, c) in enumerate(cells)
                @test grad[k] ≈ gz[c] rtol=1e-6
            end
            # sparsity: off-path cells have zero gradient
            onpath = Set(cells)
            for c in 1:n_cells
                if !(c in onpath)
                    @test abs(gz[c]) < 1e-12
                end
            end
        end
    end

    @testset "2. CSDA AD vs forward finite-difference" begin
        fv, cells, grad = directional_flux_and_grad_csda(physics, shallow, matcfg, site, path, wfield, samples)
        h = 1e-5
        for (k, c) in enumerate(cells)
            wp = copy(wfield); wp[c] += h
            wm = copy(wfield); wm[c] -= h
            fp = compute_directional_flux_csda(physics, shallow, matcfg, site, path, wp, samples)
            fm = compute_directional_flux_csda(physics, shallow, matcfg, site, path, wm, samples)
            fd = (fp - fm) / (2h)
            @test grad[k] ≈ fd rtol=1e-4
        end
    end

    @testset "2b. CSDA AD vs forward FD with MC correction ENABLED" begin
        # The default fixture runs with the correction disabled, so it never
        # exercises the physical-straggling enhancement term (which depends on w
        # through the surface energy / spectral index). Enable it here and confirm
        # the sparse AD Jacobian still matches the forward finite-difference: this
        # locks in the differentiability of `local_spectral_index`/`csda_var_correction`.
        set_csda_correction!(enabled=true, kappa_strag=1.0, kappa_hard=7.0,
                             resid_a=0.1, resid_b=-0.05, resid_c=0.01, resid_d=0.2)
        try
            fv, cells, grad = directional_flux_and_grad_csda(physics, shallow, matcfg, site, path, wfield, samples)
            @test isfinite(fv) && fv > 0
            h = 1e-5
            for (k, c) in enumerate(cells)
                wp = copy(wfield); wp[c] += h
                wm = copy(wfield); wm[c] -= h
                fp = compute_directional_flux_csda(physics, shallow, matcfg, site, path, wp, samples)
                fm = compute_directional_flux_csda(physics, shallow, matcfg, site, path, wm, samples)
                fd = (fp - fm) / (2h)
                @test grad[k] ≈ fd rtol=1e-4
            end
        finally
            set_csda_correction!(enabled=false)   # restore default for later testsets
        end
    end

    @testset "3. CSDA AD vs stochastic MC finite-difference" begin
        # FD on the full Monte-Carlo transport must agree with the CSDA AD
        # gradient within the MC statistical error.
        fv, cells, grad = directional_flux_and_grad_csda(physics, shallow, matcfg, site, path, wfield, samples)
        mc_samples = sample_energy_set(4000, 1e-3, 1e6, 7)
        δ = 0.05
        target_cell = cells[2]                 # a well-illuminated interior cell
        k = findfirst(==(target_cell), cells)
        lo = max(0.0, wfield[target_cell] - δ)
        hi = min(0.9, wfield[target_cell] + δ)
        w_lo = copy(wfield); w_lo[target_cell] = lo
        w_hi = copy(wfield); w_hi[target_cell] = hi
        mats_lo, dens_lo = build_cell_properties_for_mc(shallow, w_lo, matcfg)
        mats_hi, dens_hi = build_cell_properties_for_mc(shallow, w_hi, matcfg)
        flo, slo = compute_directional_flux_mc(physics, matcfg, site, path, mats_lo, dens_lo, mc_samples, 100;
            straggling=true, energy_threshold_low=100.0)
        fhi, shi = compute_directional_flux_mc(physics, matcfg, site, path, mats_hi, dens_hi, mc_samples, 100;
            straggling=true, energy_threshold_low=100.0)
        grad_fd = (fhi - flo) / (hi - lo)
        sigma = sqrt(slo^2 + shi^2) / (hi - lo)
        @test isfinite(grad_fd)
        @test isfinite(grad[k])
        # agreement within 4 MC sigma (sign + magnitude)
        @test abs(grad[k] - grad_fd) <= 4.0 * sigma + 0.05 * abs(grad[k])
    end

    @testset "4. assembly correctness + threaded == serial" begin
        f_ser, J_ser = assemble_forward_and_jacobian(physics, shallow, matcfg, site, paths, wfield, samples; n_cells=n_cells, threaded=false)
        f_thr, J_thr = assemble_forward_and_jacobian(physics, shallow, matcfg, site, paths, wfield, samples; n_cells=n_cells, threaded=true)
        @test f_ser == f_thr
        @test Matrix(J_ser) == Matrix(J_thr)
        # rows match per-bin adjoint
        for (b, p) in enumerate(paths)
            fv, cells, grad = directional_flux_and_grad_csda(physics, shallow, matcfg, site, p, wfield, samples)
            @test f_ser[b] ≈ fv rtol=1e-12
            for (k, c) in enumerate(cells)
                @test J_ser[b, c] ≈ grad[k] rtol=1e-10
            end
        end
    end

    @testset "5. inverse solvers recover a known field" begin
        # Well-conditioned, nonnegative linear absorption system A w ≈ b:
        # direct per-cell measurements (identity rows) plus mixed projections.
        n = 6
        A = zeros(12, n)
        for j in 1:n; A[j, j] = 1.0; end
        A[7, :]  = [1, 1, 0, 0, 0, 0]
        A[8, :]  = [0, 1, 1, 0, 0, 0]
        A[9, :]  = [0, 0, 1, 1, 0, 0]
        A[10, :] = [0, 0, 0, 1, 1, 0]
        A[11, :] = [0, 0, 0, 0, 1, 1]
        A[12, :] = [1, 0, 0, 0, 0, 1]
        w_true = [0.30, 0.10, 0.50, 0.20, 0.40, 0.05]
        b = A * w_true

        w_sart, h_sart = sart_reconstruct(A, b; n_iter=400, relaxation=0.3)
        @test mse(w_sart, w_true) < 1e-3
        @test h_sart[end] < h_sart[1]
        @test all(0.0 .<= w_sart .<= 0.9)

        # MLEM maximises a Poisson/KL objective (not least squares), so it
        # converges to the true field more slowly; allow more iterations + slack.
        w_mlem, h_mlem = mlem_reconstruct(A, b; n_iter=3000)
        @test mse(w_mlem, w_true) < 5e-3
        @test h_mlem[end] < h_mlem[1]
        @test all(0.0 .<= w_mlem .<= 0.9)

        model = w -> (A * w, A)
        w_gd, h_gd = gradient_descent_reconstruct(model, b; w0=fill(0.2, n),
            n_iter=4000, lr=0.05, optimizer=:adam)
        @test mse(w_gd, w_true) < 1e-4
        @test h_gd[end] < 1e-2
        @test all(0.0 .<= w_gd .<= 0.9)

        # smoothness prior is usable and reduces roughness on a noisy problem
        neighbors = [filter(!=(i), [max(1, i - 1), min(n, i + 1)]) for i in 1:n]
        prior = SmoothnessPrior(neighbors, 0.01)
        w_reg, _ = sart_reconstruct(A, b; n_iter=200, relaxation=0.2, prior=prior)
        @test all(0.0 .<= w_reg .<= 0.9)
    end

    @testset "6. metric sanity" begin
        x = [0.1, 0.4, 0.2, 0.7, 0.3]
        y = [0.12, 0.38, 0.25, 0.65, 0.28]
        @test mse(x, x) == 0.0
        @test rmse(x, x) == 0.0
        @test ssim_metric(x, x) ≈ 1.0 rtol=1e-10
        @test isinf(psnr(x, x))
        @test mse(x, y) > 0
        @test psnr(x, y) > 0
        @test 0.0 <= ssim_metric(x, y) <= 1.0 + 1e-9
        roi = [true, false, false, true, false]
        bg = [false, true, true, false, true]
        @test cnr(x, roi, bg) >= 0
        # hand-checked MSE
        @test mse([0.0, 1.0], [1.0, 1.0]) ≈ 0.5
        rep = reconstruction_report(x, y; roi_mask=roi, bg_mask=bg)
        @test rep.mse ≈ mse(x, y)
        @test isfinite(rep.psnr)
    end

    @testset "7. resolution estimator" begin
        # radial_fwhm on a known PSF: anomaly + ring of cells at radius 5 m
        centroids = [0.0 5.0 -5.0 0.0  0.0;
                     0.0 0.0  0.0 5.0 -5.0;
                     0.0 0.0  0.0 0.0  0.0]
        psf = [1.0, 0.6, 0.6, 0.4, 0.6]   # cells 2,3,5 exceed half-max (0.5) at r=5
        fw = radial_fwhm(psf, centroids, (0.0, 0.0, 0.0), 1.0)
        @test fw ≈ 10.0 rtol=1e-9   # 2 * r_half, r_half = 5

        # point_spread_recovery with a near-perfect reconstruct closure
        base = zeros(5)
        recon = w_true -> begin
            # pretend solver: recover anomaly with slight blur into neighbours
            out = copy(w_true)
            out[2] += 0.1 * w_true[1]
            out
        end
        res = point_spread_recovery(recon, centroids, base, 1; contrast=0.5)
        @test res.peak_cell == 1
        @test isfinite(res.fwhm_m)
        @test res.fwhm_m >= 0.0
        @test res.center == (0.0, 0.0, 0.0)

        rvd = resolution_vs_depth(recon, centroids, base, [1, 4];
            depth_edges=[-1.0, 1.0, 10.0], contrast=0.5, zenith_step_deg=1.0)
        @test rvd isa Vector
    end

    @testset "8. edge prior + Gauss-Newton beats MLEM (noisy, blocky)" begin
        n = 20
        neighbors = [filter(j -> 1 <= j <= n && j != i, [i - 1, i + 1]) for i in 1:n]

        # EdgePrior gradient matches a finite-difference of R(w)
        prior = EdgePrior(neighbors, 1.0, 0.05)
        Rfun(w) = begin
            s = 0.0
            for i in 1:n, j in neighbors[i]
                i < j || continue
                t = w[i] - w[j]
                s += abs(t) <= prior.delta ? 0.5 * t^2 : prior.delta * (abs(t) - 0.5 * prior.delta)
            end
            prior.weight * s
        end
        wprobe = collect(range(0.0, 0.6; length = n)); wprobe[10:14] .= 0.7
        g = edge_gradient(prior, wprobe)
        for k in (3, 10, 17)
            h = 1e-6
            wp = copy(wprobe); wp[k] += h; wm = copy(wprobe); wm[k] -= h
            @test isapprox(g[k], (Rfun(wp) - Rfun(wm)) / (2h); rtol = 1e-4, atol = 1e-8)
        end
        H = edge_hessian(prior, wprobe)
        @test size(H) == (n, n)
        @test isapprox(H, transpose(H); atol = 1e-12)          # symmetric
        @test minimum(eigvals(Matrix(H) + 1e-9I)) > 0           # SPD

        # Blocky truth; data from a NONLINEAR forward (no inverse crime).
        # MLEM/SART invert the linearised operator A (mismatched); Gauss-Newton
        # relinearises the true nonlinear model — the physics setup in miniature,
        # where our method's edge is fitting the real operator under noise.
        rng = MersenneTwister(2024)
        w_true = zeros(n); w_true[9:13] .= 0.5
        rows = Vector{Vector{Float64}}()
        for _ in 1:60
            a = zeros(n); st = rand(rng, 1:(n - 4)); L = rand(rng, 3:5)
            a[st:st+L-1] .= 0.5 .+ rand(rng, L)
            push!(rows, a)
        end
        A = permutedims(hcat(rows...))
        m0 = maximum(A * w_true)
        c = 0.6 / m0                          # quadratic gain ~20–30% at the block
        forward(w) = (u = A * w; u .+ c .* u .^ 2)
        nl_jac(w)  = (u = A * w; (Diagonal(1 .+ 2 .* c .* u) * A, u .+ c .* u .^ 2))
        b = forward(w_true) .+ 0.02 .* m0 .* randn(rng, size(A, 1))

        sm = SmoothnessPrior(neighbors, 1e-3)
        ep = EdgePrior(neighbors, 1e-3, 0.03)
        # Baselines see only the linearised operator A and the (nonlinear) data b.
        w_mlem, _ = mlem_reconstruct(max.(A, 0.0), max.(b, 0.0); n_iter = 1500, prior = sm)
        model = w -> (p = nl_jac(w); (p[2], p[1]))
        w_gn, hist = gauss_newton_reconstruct(model, b; w0 = fill(0.1, n), n_iter = 40,
            prior = ep, relinearize = true)

        @test all(0.0 .<= w_gn .<= 0.9)
        @test hist[end] <= hist[1]
        rmse_gn = sqrt(mse(w_gn, w_true)); rmse_mlem = sqrt(mse(w_mlem, w_true))
        @test rmse_gn < rmse_mlem            # our method beats MLEM under model mismatch
        @test rmse_gn < 0.7 * rmse_mlem      # …by a clear margin
        @test ssim_metric(w_gn, w_true) > ssim_metric(w_mlem, w_true)
    end

    @testset "9. posterior / null-model / inventory-profile statistics" begin
        Random.seed!(7)
        n, m = 6, 40
        A = abs.(randn(m, n)) .+ 0.2
        wtrue = [0.0, 0.4, 0.0, 0.6, 0.1, 0.0]
        free = trues(n)

        # (a) Laplace posterior σ and p_eff vs closed form, on a linear model
        lin = w -> (A * w, sparse(A))
        Wv = fill(50.0, m)
        post = laplace_posterior(lin, wtrue; weights = Wv,
            prior = SmoothnessPrior([Int[] for _ in 1:n], 0.0), free = free)
        Σ = inv(Symmetric(transpose(A) * Diagonal(Wv) * A))
        @test isapprox(post.sigma, sqrt.(diag(Σ)); rtol = 1e-5)
        @test isapprox(post.p_eff, Float64(n); atol = 1e-5)        # no prior → p_eff = #params
        # a smoothness prior shrinks the effective DOF
        post_s = laplace_posterior(lin, wtrue; weights = Wv,
            prior = SmoothnessPrior([[mod1(i + 1, n)] for i in 1:n], 5.0), free = free)
        @test post_s.p_eff < post.p_eff

        # (b)+(c) null-model and profile on a NONLINEAR forward exp(-A·w): the global
        # log-scale s cannot absorb a uniform water scaling, so the tests are informative
        f = w -> exp.(-(A * w))
        meas = f(wtrue)
        fitm = ones(m); wdem = ones(m); expo = 1e8
        nulls = null_model_tests(f, wtrue, 0.3, free, meas, fitm, wdem, expo; p_eff = post.p_eff)
        @test nulls.chi2_fit < nulls.chi2_dry
        @test nulls.chi2_fit < nulls.chi2_unif
        @test nulls.sigma_dry > 3.0                                # water strongly favoured over dry rock
        @test nulls.aic.fit < nulls.aic.dry                        # AIC favours the fit even with penalty

        prof = profile_inventory(f, wtrue, free, meas, fitm, wdem, expo; n_grid = 41)
        @test isapprox(prof.T_map, sum(wtrue); rtol = 1e-9)
        @test all(prof.dchi2 .>= -1e-9)                            # Δχ² ≥ 0 (min subtracted)
        @test abs(prof.T[argmin(prof.chi2)] - sum(wtrue)) < 0.2 * sum(wtrue)  # min at MAP inventory
    end

end
