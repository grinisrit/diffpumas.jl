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

end
