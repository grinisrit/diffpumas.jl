# Standalone: julia --project=. test/test_s7_protocol.jl
using Test, DiffPumas, DiffPumas.Tomography, LinearAlgebra, Random, Statistics, TOML

# --- toy 10 x 10 x 6 grid with the real centroid layout and a toy nonlinear operator --------
function s7_toy_centroids()
    zs = [159.796, 479.388, 798.98, 1118.572, 1438.164, 1757.756]
    xs = [-4500.0, -3500, -2500, -1500, -500, 500, 1500, 2500, 3500, 4500]
    cols = Vector{Vector{Float64}}()
    for z in zs, y in xs, x in xs
        push!(cols, [x, y, z])
    end
    return reduce(hcat, cols)
end

function s7_toy_neighbors(C)
    n = size(C, 2)
    nb = [Int[] for _ in 1:n]
    for i in 1:n, j in 1:n
        i == j && continue
        d = abs.(C[:, i] .- C[:, j])
        steps = (d[1] ≈ 1000.0 && d[2] < 1 && d[3] < 1) || (d[2] ≈ 1000.0 && d[1] < 1 && d[3] < 1) ||
                (d[3] ≈ 319.592 && d[1] < 1 && d[2] < 1)
        steps && push!(nb[i], j)
    end
    return nb
end

function s7_toy_operator(C; seed = 7, nbins = 150, nfree = 60)
    rng = MersenneTwister(seed)
    n = size(C, 2)
    freeidx = sort(randperm(rng, n)[1:nfree])
    J0 = zeros(nbins, n)
    for j in freeidx, i in 1:nbins
        rand(rng) < 0.25 && (J0[i, j] = 1e-4 * (0.2 + rand(rng)))
    end
    for j in freeidx
        J0[:, j] .= max.(J0[:, j], 0.0)
        iszero(J0[:, j]) && (J0[rand(rng, 1:nbins), j] = 1e-4)
    end
    f0 = 1e-3 .* (0.5 .+ rand(rng, nbins))
    umax = 1e-3
    pred(w) = (u = J0 * w; f0 .+ u .- 0.2 .* u .^ 2 ./ umax)
    model(w) = (u = J0 * w; (pred(w), Diagonal(1 .- 0.4 .* u ./ umax) * J0))
    free = Tomography.sensed_cells(J0)
    return ProtocolOperator(J0, f0, model, pred, s7_toy_neighbors(C), free)
end

@testset "S7 protocol" begin
    C = s7_toy_centroids()

    @testset "exposure list and seeds" begin
        @test parse_exposure_list("1e3, 3.16e3,1e8") == [1e3, 3.16e3, 1e8]
        @test_throws ArgumentError parse_exposure_list("1e3,1e3")
        @test_throws ArgumentError parse_exposure_list("-1")
        @test_throws ArgumentError parse_exposure_list("")
        @test PROTOCOL_EXPOSURES == [1e3, 3.16e3, 1e4, 3.16e4, 1e5, 1e6, 1e8]
        @test protocol_noise_seed(:test, "X", 3, 5) == 42 + 9000 + 100000 + 10000 + 300 + 5
        @test protocol_noise_seed(:dev, "M", 1, 1) == 42 + 9000 + 100 + 1
        @test_throws ArgumentError protocol_noise_seed(:other, "M", 1, 1)
    end

    @testset "Poisson counts with paired seeds" begin
        # quantile against the explicit CDF
        for λ in (0.4, 2.5, 30.0), u in (0.01, 0.3, 0.5, 0.97)
            pmf = [exp(-λ + k * log(λ) - sum(log, 1:k; init = 0.0)) for k in 0:200]
            k = poisson_quantile(u, λ)
            @test sum(pmf[1:k+1]) >= u - 1e-12
            k > 0 && @test sum(pmf[1:k]) < u + 1e-12
        end
        @test poisson_quantile(0.7, 0.0) == 0
        tmpl = fill(2.0e-4, 6000)
        a = draw_poisson_counts(tmpl, 1.6e4, 119347)       # λ = 3.2
        b = draw_poisson_counts(tmpl, 1.6e4, 119347)
        c = draw_poisson_counts(tmpl, 1.6e4, 119348)
        @test a == b
        @test a != c
        @test abs(mean(a) - 3.2) < 0.15 && abs(var(a) - 3.2) < 0.3
        big = draw_poisson_counts(fill(5.0e-4, 6000), 1.0e8, 5)  # λ = 5e4
        @test abs(mean(big) / 5.0e4 - 1) < 2e-3 && abs(var(big) / 5.0e4 - 1) < 0.08
        # a fixed seed is monotonically coupled across exposures
        lo = draw_poisson_counts(tmpl, 1e4, 9); hi = draw_poisson_counts(tmpl, 1e6, 9)
        @test all(hi .>= lo)
        @test all(iszero, draw_poisson_counts(zeros(10), 1e8, 3))
        @test all(iszero, draw_poisson_counts(-ones(10), 1e8, 3))
        u1 = make_unit(tmpl, :test, "M", 2, 1e4, 3); u2 = make_unit(tmpl, :test, "M", 2, 1e4, 3)
        @test u1.counts == u2.counts && u1.obs == u1.counts ./ 1e4 && u1.seed == protocol_noise_seed(:test, "M", 2, 3)
    end

    @testset "weights come from the observed counts only" begin
        @test protocol_weights([0, 1, 4, 100], 2.0) == [4.0, 4.0, 1.0, 0.04]
        # the weight function takes counts and tau, nothing else
        @test all(m -> length(m.sig.parameters) == 3, methods(protocol_weights))
        op = s7_toy_operator(C)
        tmpl = op.predict(0.4 .* op.free .* (C[3, :] .> 1000))
        unit = make_unit(tmpl, :dev, "M", 1, 1e5, 1)
        W = protocol_weights(unit.counts, unit.tau)
        run1 = run_gn(op, unit.obs, W, 0.1, [3, :final], :relin, zeros(size(C, 2)))
        # replace the noise-free template by garbage after the counts are drawn
        fill!(tmpl, 123.0)
        run2 = run_gn(op, unit.obs, W, 0.1, [3, :final], :relin, zeros(size(C, 2)))
        @test [v for (_, v) in run1.vectors] == [v for (_, v) in run2.vectors]
        # the oracle weights are a separate, explicit diagnostic
        @test oracle_weights(op.f0, 1e5) ≈ 1e5 ./ op.f0
    end

    @testset "test field and gate G1 counts" begin
        w = protocol_test_field(C)
        names, masks, vals = field_structures(:test, C)
        @test count(>(0), w) == 29
        @test count(masks[1]) == 12 && count(masks[2]) == 14 && count(masks[3]) == 4
        @test count(masks[2] .& (C[3, :] .< 350)) == 2
        @test count(masks[1] .& masks[2]) == 1
        i = findfirst(k -> C[1, k] == -500 && C[2, k] == -1500 && abs(C[3, k] - 1438.164) < 1, 1:size(C, 2))
        @test masks[1][i] && masks[2][i] && w[i] == 0.6
        geo = make_geometry(:test, w, C, trues(size(C, 2)))
        rep = gate_g1_report(geo)
        @test rep.nonzero_in_Ds == 27 && rep.per_structure == [12, 12, 4] && rep.passed
        # dev field: 42 nonzero cells with the rich-field structure rules
        dn, dm, dv = field_structures(:dev, C)
        wd = zeros(size(C, 2)); for (m, v) in zip(dm, dv); wd[m] .= max.(wd[m], v); end
        @test count(>(0), wd) == 42
        gd = make_geometry(:dev, wd, C, trues(size(C, 2)))
        @test gate_g1_report(gd).nonzero_in_Ds == 38
        @test gate_g1_report(gd).zero_field_rmse_s ≈ sqrt(sum(abs2, wd[C[3, :] .>= 350]) / count(C[3, :] .>= 350))
    end

    @testset "crossed-cell evaluation metric" begin
        op = s7_toy_operator(C)
        w = protocol_test_field(C)
        w[.!op.free] .= 0.0                     # make some truth cells fixed-dry
        geo = make_geometry(:test, w, C, op.free)
        @test geo.Ds == (geo.eval_mask .& BitVector(op.free))
        wr = clamp.(w .+ 0.05 .* sin.(1:length(w)), 0, 0.9) .* op.free
        m = protocol_metrics(wr, geo)
        idx = findall(geo.Ds)
        @test m.rmse_s ≈ sqrt(mean((wr[idx] .- w[idx]) .^ 2))
        # legacy relation for a reconstruction that is exactly 0 outside the free set
        ev = findall(geo.eval_mask)
        c = sum(abs2, w[ev[.!geo.Ds[ev]]])
        @test m.rmse500^2 ≈ (length(idx) * m.rmse_s^2 + c) / length(ev)
        zero_m = protocol_metrics(zeros(length(w)), geo)
        @test zero_m.rmse_s ≈ gate_g1_report(geo).zero_field_rmse_s
        @test isnan(protocol_metrics(zeros(length(w)), geo).corr)
    end

    @testset "free = sensed for every solver" begin
        op = s7_toy_operator(C)
        n = size(C, 2)
        @test count(op.free) < n
        wt = zeros(n); wt[findall(op.free)[1:8]] .= 0.5
        tmpl = op.predict(wt)
        unit = make_unit(tmpl, :dev, "M", 1, 1e5, 1)
        W = protocol_weights(unit.counts, unit.tau)
        runs = [run_sart(op, unit.obs, 0.3, [5, 50]), run_mlem(op, unit.obs, 0.5, [10, 100]),
                run_gn(op, unit.obs, W, 0.1, [1, :final], :lin, zeros(n)),
                run_gn(op, unit.obs, W, 0.1, [1, :final], :relin, zeros(n)),
                run_gd(op, unit.obs, W, 0.1, [5, :final], zeros(n))]
        wm = only(run_mlem(op, unit.obs, 0.5, [50]).vectors).second
        push!(runs, run_gn(op, unit.obs, W, 0.1, [2, :final], :relin, wm))
        push!(runs, run_gd(op, unit.obs, W, 0.1, [5, :final], wm))
        for r in runs, (_, v) in r.vectors
            @test all(iszero, v[.!op.free])
            @test all(0 .<= v .<= 0.9)
        end
        # SART smoothing really acts on the free cells (it leaks into the neighbours if not masked)
        r = only(run_sart(op, unit.obs, 0.3, [30]).vectors).second
        @test any(r[op.free] .> 0)
        nof = Tomography.sart_reconstruct(op.J0, unit.obs .- op.f0; n_iter = 30,
            prior = SmoothnessPrior(op.neighbors, 1.0), smooth_strength = 0.3)[1]
        @test any(nof[.!op.free] .!= 0)
        # GN-lin predicts f0 + J0 w: its objective equals the linear model's
        lin = run_gn(op, unit.obs, W, 0.1, [:final], :lin, zeros(n))
        @test lin.jac == 1 && lin.fwd == 0
        # explicit K iterates are consistent with the update count
        g = run_gn(op, unit.obs, W, 0.1, [1, 2, 3, 5, 8, :final], :relin, zeros(n))
        @test g.updates >= 1 && g.stop in ("max_updates", "projected_gradient", "no_step", "objective_decrease")
        @test last(g.vectors[end]) == last(g.vectors[min(g.updates, 5)]) || g.updates > 8
    end

    @testset "development grid, selection freeze and test stage" begin
        mktempdir() do dir
            op = s7_toy_operator(C)
            n = size(C, 2)
            w_dev = zeros(n)
            _, dm, dv = field_structures(:dev, C)
            for (m, v) in zip(dm, dv); w_dev[m] .= max.(w_dev[m], v); end
            w_dev .*= op.free
            w_test = protocol_test_field(C) .* op.free
            exposures = [1e3, 1e4]            # arm M runs at 1e4 only (reduced grid), X and Mc at both
            s_dry = 0.8
            op_cal = scale_operator(op, s_dry)
            geo_d = make_geometry(:dev, w_dev, C, op.free)
            geo_t = make_geometry(:test, w_test, C, op.free)
            tm_d = s_dry .* op.predict(w_dev); tm_t = s_dry .* op.predict(w_test)     # MC data 20 % below the published operator
            templates_d = Dict("M" => tm_d, "Mc" => tm_d, "X" => op.predict(w_dev) .* 0.9)
            templates_t = Dict("M" => tm_t, "Mc" => tm_t, "X" => op.predict(w_test) .* 0.9)
            devdir = joinpath(dir, "dev")
            ctx = ProtocolContext(op, geo_d, :dev, templates_d, exposures, 2, 42, devdir, op_cal)
            arms = ["M", "Mc", "X"]
            for phase in 1:3
                mlem_sel = phase >= 2 ? Tomography.read_phase_selection(devdir, 1) : nothing
                gn_sel = phase == 3 ? Tomography.read_phase_selection(devdir, 2) : nothing
                for round in 1:4
                    for shard in 1:2   # two shards, as two processes would run
                        dev_phase!(ctx, phase, load_grids(devdir, arms, 2); mlem_selection = mlem_sel,
                                   gn_selection = gn_sel, shard = (shard, 2))
                    end
                    dev_select!(devdir, phase, arms, exposures, 2) == 0 && break
                end
            end
            # per-realisation logging: one row per (arm, exposure, rep, solver, hyper, K)
            rows = read_rows(rows_in(devdir, "rows_phase"))
            @test !isempty(rows) && all(r -> all(c -> haskey(r, c), PROTOCOL_COLUMNS), rows)
            @test length(unique(Tomography.row_key(r) for r in rows)) == length(rows)       # no duplicates
            @test Set(r["rep"] for r in rows) == Set(["1", "2"])
            @test Set(r["solver"] for r in rows) == Set(PROTOCOL_SOLVERS)
            # reduced arm M: exposure 1e4 only, no GD
            @test Set(r["exp_index"] for r in rows if r["arm"] == "M") == Set(["2"])
            @test isempty([r for r in rows if r["arm"] == "M" && r["solver"] in ("GD-0", "MLEM->GD")])
            @test Set(r["exp_index"] for r in rows if r["arm"] == "Mc") == Set(["1", "2"])
            @test !isempty([r for r in rows if r["arm"] == "Mc" && r["solver"] == "GD-0"])
            # arms M and Mc read identical observation vectors (same seeds, same template) at the same exposure
            @test [r["seed"] for r in rows if r["arm"] == "M" && r["rep"] == "1" && r["exp_index"] == "2"][1] ==
                  [r["seed"] for r in rows if r["arm"] == "Mc" && r["rep"] == "1" && r["exp_index"] == "2"][1]
            # GN rows record the cap and the K = 13 iterate
            @test any(r -> r["solver"] == "GN-0" && r["iter"] == "K=13" && r["cap"] == "20", rows)
            @test all(r -> r["seed"] == string(protocol_noise_seed(:dev, r["arm"], parse(Int, r["exp_index"]), parse(Int, r["rep"]))), rows)
            r1 = first(r for r in rows if r["solver"] == "GN-0")
            @test parse(Float64, r1["secs"]) > 0 && parse(Int, r1["jac"]) >= 1 && r1["stop"] != ""
            # every reconstruction is saved and matches the logged metric
            f, key = split(r1["rec"], '#')
            saved = Tomography.Serialization.deserialize(joinpath(devdir, f))
            @test parse(Float64, r1["rmse_s"]) ≈ protocol_metrics(saved[key], geo_d).rmse_s
            # resume: a second pass computes nothing new
            @test dev_phase!(ctx, 1, load_grids(devdir, arms, 2); shard = (1, 2)) == 0
            # freeze
            selpath = joinpath(dir, "selection.toml")
            @test_throws ErrorException load_selection(selpath)
            sha = Tomography.dev_finalize!(devdir, selpath, Dict("field" => "dev"))
            sel, sha2, _ = load_selection(selpath)
            @test sha == sha2 && length(sel) == (7 * 2) * 2 + 5          # X and Mc: 7 solvers x 2 exposures; M: 5 solvers x 1
            @test all(k -> get(sel[k], "cap", 0) in (20, 32, 40), [k for k in keys(sel) if sel[k]["solver"] in ("GN-0", "GN-lin", "MLEM->GN", "GD-0", "MLEM->GD")])
            # the test stage refuses a missing or altered selection
            testdir = joinpath(dir, "test")
            tctx = ProtocolContext(op, geo_t, :test, templates_t, exposures, 2, 42, testdir, op_cal)
            @test_throws ErrorException (s = load_selection(joinpath(dir, "absent.toml")); test_stage!(tctx, s[1], s[2]))
            txt = read(selpath, String)
            write(joinpath(dir, "tampered.toml"), replace(txt, "param = " => "param = 9"; count = 1))
            @test_throws ErrorException load_selection(joinpath(dir, "tampered.toml"))
            # test stage on the frozen file, two shards, with the hash recorded
            for shard in 1:2
                test_stage!(tctx, sel, sha; shard = (shard, 2))
            end
            trows = read_rows(rows_in(testdir, "rows_"))
            @test all(r -> r["selection_sha256"] == sha && r["field"] == "test", trows)
            @test Set(r["solver"] for r in trows) == Set(["ZERO"; collect(PROTOCOL_SOLVERS)])
            @test count(r -> r["selected"] == "1" && r["solver"] == "GN-0", trows) == (2 + 2 + 1) * 2     # X, Mc: 2 exposures; M: 1; 2 reps
            @test isempty([r for r in trows if r["arm"] == "M" && (r["solver"] in ("GD-0", "MLEM->GD") || r["exp_index"] == "1")])
            # the amended GN K grid is logged at the test stage (K = 13 from the saved iterates)
            @test any(r -> r["solver"] == "GN-0" && r["iter"] == "K=13", trows)
            # test-stage summary: cell statistics and the paired contrasts C1, C2, C3
            sm = test_summary(trows)
            c3 = [r for r in sm if r["kind"] == "C3"]
            @test Set(r["solver"] for r in c3) == Set(["MLEM", "GN-0", "MLEM->GN"]) && all(r -> r["exp_index"] == "2" && r["arm"] == "M-Mc", c3)
            tab = Tomography.test_rmse_table(trows)
            for r in c3
                d = [tab[("M", r["solver"], 2)][k] - tab[("Mc", r["solver"], 2)][k] for k in 1:2]
                @test parse(Float64, r["mean"]) ≈ mean(d) && parse(Int, r["n_negative"]) == count(<(0), d)
            end
            @test Set((r["kind"], r["arm"]) for r in sm if r["kind"] in ("C1", "C2")) == Set((k, a) for k in ("C1", "C2") for a in arms)
            @test count(r -> r["kind"] == "C1" && r["arm"] == "M", sm) == 1 && count(r -> r["kind"] == "C1" && r["arm"] == "Mc", sm) == 2
            sfile = Tomography.write_test_summary(joinpath(dir, "summary.tsv"), sm)
            @test countlines(sfile) == length(sm) + 1
            @test_throws ArgumentError test_stage!(ProtocolContext(op, geo_d, :dev, templates_d, exposures, 1, 42, testdir, op_cal), sel, sha)
            # oracle-weight diagnostic goes to its own directory and is labelled
            diagdir = joinpath(dir, "diag")
            dctx = ProtocolContext(op, geo_t, :test, templates_t, exposures, 2, 42, diagdir, op_cal)
            test_stage!(dctx, sel, sha; weights = :oracle, solvers = ("GN-0",), arms = ["Mc"], stage = "diag")
            drows = read_rows(rows_in(diagdir, "rows_"))
            @test !isempty(drows) && all(r -> r["weights"] == "oracle" && r["arm"] == "Mc" && r["solver"] == "GN-0", drows)
            # arm Mc refuses to run without the dry-normalised operator
            nocal = ProtocolContext(op, geo_t, :test, templates_t, exposures, 2, 42, joinpath(dir, "nocal"))
            @test_throws ErrorException test_stage!(nocal, sel, sha; arms = ["Mc"])
            @test_throws ErrorException dev_phase!(ProtocolContext(op, geo_d, :dev, templates_d, exposures, 2, 42, joinpath(dir, "nocal2")),
                                                   1, default_grids(arms, 2); shard = (1, 1))
        end
    end

    @testset "template cache refuses a mismatching key" begin
        mktempdir() do dir
            w = [0.0, 0.3]; bins = [1, 2, 3]
            k = template_key(:dev, "M", 16384, 42, w, bins)
            path = joinpath(dir, "t.jls")
            save_template(path, k, [1.0, 2.0, 3.0])
            @test load_template(path, k).template == [1.0, 2.0, 3.0]
            @test_throws ErrorException load_template(path, template_key(:dev, "M", 1024, 42, w, bins))
            @test_throws ErrorException load_template(path, template_key(:test, "M", 16384, 43, w, bins))
            @test_throws ErrorException load_template(joinpath(dir, "none.jls"), k)
        end
    end

    @testset "arm Mc: shared noise, dry-normalised operator, dry calibration" begin
        @test protocol_noise_seed(:test, "Mc", 4, 2) == protocol_noise_seed(:test, "M", 4, 2)
        @test protocol_noise_seed(:test, "Mc", 4, 2) != protocol_noise_seed(:test, "X", 4, 2)
        tmpl = fill(3.0e-4, 500)
        um = make_unit(tmpl, :dev, "M", 3, 1e5, 2); uc = make_unit(tmpl, :dev, "Mc", 3, 1e5, 2)
        @test um.counts == uc.counts && um.obs == uc.obs && um.seed == uc.seed
        @test template_arm("Mc") == "M" && template_arm("X") == "X"
        # reduced arm M
        @test arm_solvers("M") == ("SART", "MLEM", "GN-lin", "GN-0", "MLEM->GN")
        @test arm_solvers("Mc") == PROTOCOL_SOLVERS && arm_solvers("X") == PROTOCOL_SOLVERS
        @test [t for t in PROTOCOL_EXPOSURES if arm_exposure_active("M", t)] == [1e4, 1e5, 1e8]
        @test all(t -> arm_exposure_active("Mc", t) && arm_exposure_active("X", t), PROTOCOL_EXPOSURES)

        op = s7_toy_operator(C)
        n = size(C, 2)
        ops = scale_operator(op, 0.8)
        w = 0.3 .* op.free .* (C[3, :] .> 1000)
        @test ops.f0 == 0.8 .* op.f0 && ops.J0 == 0.8 .* op.J0 && ops.free == op.free
        @test ops.predict(w) == 0.8 .* op.predict(w)
        @test all(ops.model(w) .≈ (0.8 .* op.model(w)[1], 0.8 .* op.model(w)[2]))
        @test ops.s_plus ≈ 0.8 * op.s_plus                        # recomputed from the scaled J0
        @test_throws ArgumentError scale_operator(op, 0.0)
        @test_throws ArgumentError scale_operator(op, NaN)
        # s = 1 reproduces arm M to machine precision (here: bit for bit)
        op1 = scale_operator(op, 1.0)
        wt = zeros(n); wt[findall(op.free)[1:8]] .= 0.5
        unit = make_unit(op.predict(wt), :dev, "Mc", 1, 1e5, 1)
        W = protocol_weights(unit.counts, unit.tau)
        for (a, b) in ((run_mlem(op, unit.obs, 0.5, [20, 60]), run_mlem(op1, unit.obs, 0.5, [20, 60])),
                       (run_sart(op, unit.obs, 0.1, [5, 20]), run_sart(op1, unit.obs, 0.1, [5, 20])),
                       (run_gn(op, unit.obs, W, 0.1, [2, :final], :relin, zeros(n)), run_gn(op1, unit.obs, W, 0.1, [2, :final], :relin, zeros(n))),
                       (run_gn(op, unit.obs, W, 0.1, [2, :final], :lin, zeros(n)), run_gn(op1, unit.obs, W, 0.1, [2, :final], :lin, zeros(n))),
                       (run_gd(op, unit.obs, W, 0.1, [5, :final], zeros(n)), run_gd(op1, unit.obs, W, 0.1, [5, :final], zeros(n))))
            @test maximum(maximum(abs.(va .- vb)) for ((_, va), (_, vb)) in zip(a.vectors, b.vectors)) <= 1e-14
        end
        # the weights do not depend on the scale (counts only); h_+ does
        @test curvature_scale(W, ops.J0, ops.free) ≈ 0.64 * curvature_scale(W, op.J0, op.free)

        mktempdir() do dir
            # G0b: s = median of the ratios over the bins with a positive dry CSDA flux
            csda = [1.0, 2.0, 0.0, 4.0, 5.0, NaN]; mc = [0.8, 1.7, 0.3, 3.2, 4.4, 1.0]
            bins, r = dry_ratios(mc, csda)
            @test bins == [1, 2, 4, 5] && r ≈ [0.8, 0.85, 0.8, 0.88]
            @test dry_scale(r) ≈ 0.825
            zen = collect(range(0, 58; length = 6))
            rep = dry_report(mc, csda, 0.01 .* mc, zen)
            @test rep["s"] ≈ 0.825 && rep["n_bins"] == 4 && rep["ratio_of_sums"] ≈ sum(mc[[1,2,4,5]]) / sum(csda[[1,2,4,5]])
            @test rep["median_rel_sigma_mc"] ≈ 0.01 && length(rep["band_median_r"]) == length(rep["band_counts"])
            @test check_g0b(0.801)[1] && check_g0b(0.83)[1] && check_g0b(0.772)[1]
            @test !check_g0b(0.8311)[1] && !check_g0b(0.7689)[1] && !check_g0b(NaN)[1]
            # dry_calibration.toml: frozen with a hash; arm Mc refuses without a valid file
            path = joinpath(dir, "dry_calibration.toml")
            @test_throws ErrorException load_dry_calibration(path)
            sha = write_dry_calibration(path, 0.8012, Dict("n_bins" => 2700, "seed" => 44, "histories" => 16384, "git_commit" => "abc"))
            s_, sha2, cal = load_dry_calibration(path)
            @test s_ == 0.8012 && sha == sha2 && cal["seed"] == 44 && isfile(path * ".sha256") && strip(read(path * ".sha256", String)) == sha
            write(joinpath(dir, "tampered.toml"), replace(read(path, String), "0.8012" => "0.9012"))
            @test_throws ErrorException load_dry_calibration(joinpath(dir, "tampered.toml"))
            write(joinpath(dir, "bad.toml"), "[meta]\ncontent_sha256 = \"x\"\n")
            @test_throws ErrorException load_dry_calibration(joinpath(dir, "bad.toml"))
            @test_throws ArgumentError write_dry_calibration(joinpath(dir, "z.toml"), -1.0, Dict{String,Any}())
            # contexts without the calibrated operator refuse arm Mc
            geo = make_geometry(:dev, zeros(n), C, op.free)
            ctx = ProtocolContext(op, geo, :dev, Dict("Mc" => op.predict(wt)), [1e4], 1, 42, joinpath(dir, "o"))
            @test_throws ErrorException Tomography.arm_operator(ctx, "Mc")
            @test Tomography.arm_operator(ctx, "M") === op
            @test Tomography.arm_operator(ProtocolContext(op, geo, :dev, Dict("Mc" => op.predict(wt)), [1e4], 1, 42, joinpath(dir, "o"), ops), "Mc") === ops
        end
    end

    @testset "GN cap and K grid, GD fixed budget" begin
        @test Tomography.default_grid("GN-0")["iter"] == [1, 2, 3, 5, 8, 13]
        @test Tomography.default_grid("GD-0")["iter"] == [5, 10, 20]
        @test PROTOCOL_GN_CAP == 20 && PROTOCOL_GN_CAP_EXT == 32 && PROTOCOL_GD_CAP == 40
        g = default_grids(["X"], 1)
        @test g["GN-0|X|1"]["cap"] == 20 && g["GN-lin|X|1"]["cap"] == 20 && g["MLEM->GN|X|1"]["cap"] == 20
        @test g["GD-0|X|1"]["cap"] == 40 && !haskey(g["MLEM|X|1"], "cap")
        mkrow(solver, rep, c, it, rm, stop; cap = "20") = Dict("solver" => solver, "arm" => "X", "exp_index" => "1", "weights" => "obs",
            "param" => string(c), "iter" => it, "rmse_s" => string(rm), "stop" => stop, "cap" => cap, "rep" => string(rep))
        function gnrows(; stops, best = "final", cap = "20")
            out = Dict{String,String}[]
            for rep in 1:2, c in (0.01, 0.1), it in ("K=1", "K=5", "K=13", "final")
                rm = (it == best && c == 0.1) ? 0.1 : 0.2
                push!(out, mkrow("GN-0", rep, c, it, rm, it == "final" ? stops[rep] : "-"; cap = cap))
                # the other GN solvers of the phase converge early (never on the cap)
                for other in ("GN-lin", "MLEM->GN")
                    push!(out, mkrow(other, rep, c, it, it == "K=5" && c == 0.1 ? 0.1 : 0.2, "objective_decrease"; cap = cap))
                end
            end
            return out
        end
        # (a) final selected and both realisations stopped on the cap -> one rerun with cap 32 and K up to 20
        grids = default_grids(["X"], 1)
        sel, pending = Tomography.select_phase!(gnrows(stops = ["max_updates", "max_updates"]), 2, grids, ["X"], [1e4], 2)
        @test pending == 1 && grids["GN-0|X|1"]["cap"] == 32 && grids["GN-0|X|1"]["iter"] == [1, 2, 3, 5, 8, 13, 20]
        @test grids["GN-0|X|1"]["ext"]["cap"] == 1 && sel["GN-0|X|1"]["extensions_pending"]
        @test grids["GN-lin|X|1"]["cap"] == 20 && grids["MLEM->GN|X|1"]["cap"] == 20     # other cells untouched
        # (b) selected K not final, or one realisation converged: no extension
        grids = default_grids(["X"], 1)
        sel, pending = Tomography.select_phase!(gnrows(stops = ["max_updates", "max_updates"], best = "K=13"), 2, grids, ["X"], [1e4], 2)
        @test pending == 0 && grids["GN-0|X|1"]["cap"] == 20 && sel["GN-0|X|1"]["budget_note"] == ""
        grids = default_grids(["X"], 1)
        sel, pending = Tomography.select_phase!(gnrows(stops = ["max_updates", "objective_decrease"]), 2, grids, ["X"], [1e4], 2)
        @test pending == 0 && grids["GN-0|X|1"]["cap"] == 20
        # (c) after the extension only the cap-32 rows count; final at the cap again: kept and reported once
        grids = default_grids(["X"], 1)
        rows = gnrows(stops = ["max_updates", "max_updates"])
        Tomography.select_phase!(rows, 2, grids, ["X"], [1e4], 2)
        rows32 = gnrows(stops = ["max_updates", "max_updates"], cap = "32")
        sel, pending = Tomography.select_phase!([rows; rows32], 2, grids, ["X"], [1e4], 2)
        @test pending == 0 && grids["GN-0|X|1"]["cap"] == 32 && sel["GN-0|X|1"]["cap"] == 32
        @test sel["GN-0|X|1"]["budget_note"] == "not converged at 32 updates"
        # ... and when the cap-32 rows prefer a smaller K, nothing is reported
        grids = default_grids(["X"], 1)
        Tomography.select_phase!(rows, 2, grids, ["X"], [1e4], 2)
        sel, pending = Tomography.select_phase!([rows; gnrows(stops = ["max_updates", "max_updates"], cap = "32", best = "K=13")], 2, grids, ["X"], [1e4], 2)
        @test pending == 0 && sel["GN-0|X|1"]["iter"] == "K=13" && sel["GN-0|X|1"]["budget_note"] == ""
        # GD: the cap of 40 is never extended; a final on the cap is a fixed budget
        grids = default_grids(["X"], 1)
        gdrows = [mkrow("GD-0", rep, 0.1, it, it == "final" ? 0.1 : 0.2, it == "final" ? "max_iter" : "-"; cap = "40") for rep in 1:2, it in ("K=5", "K=20", "final")]
        gdother = [mkrow("MLEM->GD", rep, 0.1, it, it == "K=20" ? 0.1 : 0.2, "objective_change"; cap = "40") for rep in 1:2, it in ("K=5", "K=20", "final")]
        sel, pending = Tomography.select_phase!([vec(gdrows); vec(gdother)], 3, grids, ["X"], [1e4], 2)
        @test pending == 0 && grids["GD-0|X|1"]["cap"] == 40 && sel["GD-0|X|1"]["budget_note"] == "fixed budget of 40 iterations"
        # labels of the logged final iterate
        @test Tomography.final_budget_note("GN-0", "final", "max_updates", 20) == "K = 20 (cap)"
        @test Tomography.final_budget_note("MLEM->GN", "final", "max_updates", 32) == "K = 32 (cap)"
        @test Tomography.final_budget_note("GN-0", "final", "objective_decrease", 20) == ""
        @test Tomography.final_budget_note("GN-0", "K=5", "max_updates", 20) == ""
        @test Tomography.final_budget_note("GD-0", "final", "max_iter", 40) == "fixed budget of 40 iterations"
        @test Tomography.final_budget_note("GD-0", "final", "objective_change", 40) == ""
        @test Tomography.final_budget_note("MLEM", "final", "n_iter", nothing) == ""
        # real runs: the run labels the cap-limited GN and GD solves
        op = s7_toy_operator(C); n = size(C, 2)
        wt = zeros(n); wt[findall(op.free)[1:8]] .= 0.5
        unit = make_unit(op.predict(wt), :dev, "X", 1, 1e8, 1)
        W = protocol_weights(unit.counts, unit.tau)
        geo = make_geometry(:dev, wt, C, op.free)
        g2 = run_gn(op, unit.obs, W, 1e-3, [1, 2, :final], :relin, zeros(n); max_updates = 2)
        @test g2.stop == "max_updates" && g2.updates == 2
        row = Tomography.protocol_log_row("dev", geo, op, unit, W, "GN-0", "zero", "obs", 1e-3, "final", last(g2.vectors[end]), g2; cap = 2)
        @test row["final_note"] == "K = 2 (cap)" && row["budget_limited"] == "1" && row["cap"] == "2"
        d2 = run_gd(op, unit.obs, W, 1e-3, [5, :final], zeros(n); max_iter = 3)
        @test d2.stop == "max_iter"
        rowd = Tomography.protocol_log_row("dev", geo, op, unit, W, "GD-0", "zero", "obs", 1e-3, "final", last(d2.vectors[end]), d2; cap = 3)
        @test rowd["final_note"] == "fixed budget of 3 iterations" && rowd["budget_limited"] == "1"
        rowk = Tomography.protocol_log_row("dev", geo, op, unit, W, "GN-0", "zero", "obs", 1e-3, "K=1", last(g2.vectors[1]), g2; cap = 2)
        @test rowk["final_note"] == "" && rowk["budget_limited"] == "0"
        # a development phase honours the cap of the grid and records it; resume keys include the cap
        mktempdir() do dir
            tm = op.predict(wt)
            ctx = ProtocolContext(op, geo, :dev, Dict("X" => tm), [1e5], 2, 42, dir)
            grids = default_grids(["X"], 1)
            grids["GN-0|X|1"]["param"] = [0.1]; grids["GN-lin|X|1"]["param"] = [0.1]; grids["MLEM->GN|X|1"]["param"] = [0.1]
            grids["GN-0|X|1"]["cap"] = 32; grids["GN-0|X|1"]["iter"] = copy(PROTOCOL_GN_K_EXT)
            sel1 = Dict(k => Dict("param" => 0.5, "iter" => "n=50") for k in ("MLEM|X|1",))
            n1 = dev_phase!(ctx, 2, grids; mlem_selection = sel1)
            rows = read_rows(rows_in(dir, "rows_phase"))
            gn0 = [r for r in rows if r["solver"] == "GN-0"]
            @test !isempty(gn0) && all(r -> r["cap"] == "32", gn0) && any(r -> r["iter"] == "K=20", gn0)
            @test dev_phase!(ctx, 2, grids; mlem_selection = sel1) == 0        # resume: nothing missing
            grids["GN-0|X|1"]["cap"] = 20; grids["GN-0|X|1"]["iter"] = copy(PROTOCOL_GN_K)
            @test dev_phase!(ctx, 2, grids; mlem_selection = sel1) == 2        # the cap-20 cell (2 realisations) is a different cell
        end
    end

    @testset "summary statistics" begin
        @test t_crit95(7) ≈ 2.364624 atol = 1e-5
        @test t_crit95(1) ≈ 12.7062 atol = 1e-3
        ci = Tomography.mean_ci95([1.0, 2.0, 3.0, 4.0])
        @test ci.mean == 2.5
        @test isapprox(ci.lo, 2.5 - 3.182446 * std([1.0, 2.0, 3.0, 4.0]) / 2; atol = 1e-4)
        mk(arm, solver, rep, v) = Dict("stage" => "test", "selected" => "1", "weights" => "obs", "arm" => arm, "solver" => solver,
            "exp_index" => "1", "rep" => string(rep), "rmse_s" => string(v), "budget_limited" => "0")
        rows = Dict{String,String}[]
        for rep in 1:8
            push!(rows, mk("Mc", "MLEM", rep, 0.20 + 0.001rep)); push!(rows, mk("Mc", "MLEM->GN", rep, 0.10 + 0.001rep))
            push!(rows, mk("Mc", "GN-0", rep, 0.19 + 0.001rep))
            push!(rows, mk("M", "MLEM", rep, 0.25 + 0.001rep)); push!(rows, mk("M", "GN-0", rep, 0.30 + 0.001rep))
            push!(rows, mk("M", "MLEM->GN", rep, 0.27 + 0.001rep))
        end
        sm = test_summary(rows)
        lab(kind, arm) = only(r["label"] for r in sm if r["kind"] == kind && r["arm"] == arm && r["exp_index"] == "1")
        @test lab("C1", "Mc") == "clearly better"       # 50 % reduction, interval below 0
        @test lab("C2", "Mc") == "not distinguished" || lab("C2", "Mc") == "better"
        @test lab("C2", "M") == "MLEM better"
        c3 = Dict(r["solver"] => r for r in sm if r["kind"] == "C3")
        @test parse(Float64, c3["MLEM"]["mean"]) ≈ 0.05 && parse(Float64, c3["GN-0"]["mean"]) ≈ 0.11
        @test c3["MLEM"]["n_negative"] == "0" && parse(Float64, c3["MLEM"]["lo"]) ≈ 0.05   # constant difference
    end

    @testset "claim-based work sharing" begin
        mktempdir() do dir
            claims = joinpath(dir, "claims")
            mkpath(claims)
            @test claim_unit!(claims, "X_e1_r1") && !claim_unit!(claims, "X_e1_r1") && claim_unit!(claims, "X_e1_r2")
            mark_unit_done!(claims, "X_e1_r1"); @test isfile(joinpath(claims, "X_e1_r1.done"))
            announce_units!(claims, 7); @test strip(read(joinpath(claims, "TOTAL"), String)) == "7"
            @test unit_name("Mc", 3, 2) == "Mc_e3_r2"
            # the pause file blocks a worker until it disappears
            pf = joinpath(dir, "PAUSE"); touch(pf)
            t = @async (wait_unpaused(pf; poll = 0.05); :resumed)
            sleep(0.3); @test !istaskdone(t)
            rm(pf); @test fetch(t) === :resumed
            wait_unpaused(nothing)
            # two workers on one claim dir share the units of a phase without overlap; the result equals a serial run
            C = s7_toy_centroids(); op = s7_toy_operator(C); n = size(C, 2)
            wd = zeros(n); wd[findall(op.free)[1:10]] .= 0.4
            geo = make_geometry(:dev, wd, C, op.free)
            tm = Dict("X" => op.predict(wd))
            exps = [1e4, 1e5]
            serial = ProtocolContext(op, geo, :dev, tm, exps, 2, 42, joinpath(dir, "serial"))
            dev_phase!(serial, 1, default_grids(["X"], 2))
            shared = ProtocolContext(op, geo, :dev, tm, exps, 2, 42, joinpath(dir, "shared"))
            cdir = joinpath(dir, "cl")
            n1 = dev_phase!(shared, 1, default_grids(["X"], 2); claim_dir = cdir, worker = 1)
            n2 = dev_phase!(shared, 1, default_grids(["X"], 2); claim_dir = cdir, worker = 2)
            @test n1 > 0 && n2 == 0                                   # everything was claimed by the first worker
            @test strip(read(joinpath(cdir, "TOTAL"), String)) == "4" && length(filter(endswith(".done"), readdir(cdir))) == 4
            key(r) = (r["arm"], r["exp_index"], r["rep"], r["solver"], r["param"], r["iter"], r["rmse_s"])
            @test sort(key.(read_rows(rows_in(joinpath(dir, "serial"), "rows_phase")))) == sort(key.(read_rows(rows_in(joinpath(dir, "shared"), "rows_phase"))))
            @test isfile(joinpath(dir, "shared", "rows_phase1_shard1.tsv"))
            # a claimed unit of a dead worker is skipped until its claim is removed
            rm(joinpath(cdir, "X_e1_r1")); rm(joinpath(cdir, "X_e1_r1.done"))
            rm(joinpath(dir, "shared"); recursive = true)
            @test dev_phase!(shared, 1, default_grids(["X"], 2); claim_dir = cdir, worker = 3) > 0
            @test length(filter(endswith(".done"), readdir(cdir))) == 4
        end
    end

    @testset "budget cuts and the 36 h trigger" begin
        # cut semantics
        @test !cut_excludes(Set(), "X", 3.16e3, "GN-0")
        @test cut_excludes(Set([1]), "Mc", 1e4, "MLEM->GD") && !cut_excludes(Set([1]), "Mc", 1e4, "GD-0")
        @test cut_excludes(Set([2]), "X", 3.16e3, "GD-0") && cut_excludes(Set([2]), "X", 3.16e4, "GD-0") && !cut_excludes(Set([2]), "X", 1e5, "GD-0")
        @test !cut_excludes(Set([2]), "X", 3.16e3, "GN-0")
        @test cut_excludes_unit(Set([3]), "M", 1e5) && !cut_excludes_unit(Set([3]), "Mc", 1e5) && !cut_excludes_unit(Set([3]), "M", 1e4)
        @test cut_excludes_unit(Set([4]), "X", 3.16e3) && cut_excludes(Set([4]), "Mc", 3.16e3, "SART") && !cut_excludes_unit(Set([4]), "X", 1e3)
        @test !cut_excludes_unit(Set([0]), "X", 3.16e3)               # cut 0 removes no unit
        @test PROTOCOL_CUT_ORDER == [0, 1, 2, 3, 4]
        # projection from the measured development times
        exps = [1e4]
        drow(solver, rep, secs, upd; arm = "X") = Dict("solver" => solver, "arm" => arm, "exp_index" => "1", "weights" => "obs", "param" => "0.1",
            "cap" => "20", "rep" => string(rep), "secs" => string(secs), "updates" => string(upd), "iter" => "K=2")
        rows = [drow("GN-0", r, 3600.0, 10) for r in 1:2]
        sel = Dict("GN-0|X|1" => Dict("solver" => "GN-0", "arm" => "X", "exp_index" => 1, "param" => 0.1, "iter" => "K=2", "cap" => 20))
        @test project_test_cost(rows, sel, exps, 8, Set{Int}()) ≈ 8 * 3600
        @test project_test_cost(rows, sel, exps, 8, Set([0])) ≈ 2 * 3600 + 6 * 720    # reps 3..8 stop at K = 2 of 10 updates
        @test project_test_cost(rows, sel, exps, 8, Set([0, 1])) ≈ 2 * 3600 + 6 * 720
        seln = merge(sel, Dict("GN-0|X|1" => merge(sel["GN-0|X|1"], Dict("iter" => "final"))))
        @test project_test_cost(rows, seln, exps, 8, Set([0])) ≈ 8 * 3600                 # final: nothing to save
        # trigger: projected total = elapsed + test/16 processes
        p = plan_cuts(rows, sel, exps, 8; elapsed_h = 10.0)
        @test p.cuts == Int[] && p.met && p.final.total_h ≈ 10 + 28800 / 3600 / 16
        p = plan_cuts(rows, sel, exps, 8; elapsed_h = 35.7)
        @test p.cuts == [0] && p.met && p.final.total_h ≈ 35.7 + 11520 / 3600 / 16 && length(p.history) == 2
        p = plan_cuts(rows, sel, exps, 8; elapsed_h = 36.5)
        @test p.cuts == [0, 1, 2, 3, 4] && !p.met                                            # cannot be met: all cuts, reported
        # cuts.toml belongs to one selection
        mktempdir() do dir
            path = joinpath(dir, "cuts.toml")
            @test_throws ErrorException load_cuts(path, "abc")
            write_cuts(path, plan_cuts(rows, sel, exps, 8; elapsed_h = 35.7), "abc")
            @test load_cuts(path, "abc") == Set([0])
            @test_throws ErrorException load_cuts(path, "other")
        end
        # cut 0 and cut 1 in the test stage
        C = s7_toy_centroids(); op = s7_toy_operator(C); n = size(C, 2)
        wt = zeros(n); wt[findall(op.free)[1:10]] .= 0.4
        geo = make_geometry(:test, wt, C, op.free)
        mset = Dict("param" => 0.5, "iter" => "n=25")
        selt = Dict{String,Any}()
        for sname in PROTOCOL_SOLVERS
            e = Dict{String,Any}("solver" => sname, "arm" => "X", "exp_index" => 1, "tau" => 1e4, "param" => sname in ("SART",) ? 0.1 : 0.5,
                                 "iter" => sname in ("SART", "MLEM") ? "n=25" : "K=2")
            sname in PROTOCOL_GN_SOLVERS && (e["cap"] = 20; e["param"] = 0.1)
            sname in PROTOCOL_GD_SOLVERS && (e["cap"] = 40; e["param"] = 0.1)
            selt[group_key(sname, "X", 1)] = e
        end
        mktempdir() do dir
            ctx = ProtocolContext(op, geo, :test, Dict("X" => op.predict(wt)), [1e4], 3, 42, joinpath(dir, "t"))
            test_stage!(ctx, selt, "sha"; cuts = Set([0, 1]))
            trows = read_rows(rows_in(joinpath(dir, "t"), "rows_"))
            @test isempty([r for r in trows if r["solver"] == "MLEM->GD"])
            gn = [r for r in trows if r["solver"] == "GN-0"]
            @test any(r -> r["rep"] == "1" && r["iter"] == "final", gn) && any(r -> r["rep"] == "2" && r["iter"] == "final", gn)
            @test !any(r -> r["rep"] == "3" && r["iter"] == "final", gn)                 # realisation 3 stops at the selected K
            @test !any(r -> r["rep"] == "3" && r["iter"] in ("K=3", "K=5", "K=8", "K=13"), gn)
            @test count(r -> r["rep"] == "3" && r["selected"] == "1", gn) == 1 && any(r -> r["rep"] == "3" && r["iter"] == "K=2" && r["selected"] == "1", gn)
            # without cuts the same selection logs every realisation to the end
            ctx2 = ProtocolContext(op, geo, :test, Dict("X" => op.predict(wt)), [1e4], 3, 42, joinpath(dir, "t2"))
            test_stage!(ctx2, selt, "sha")
            t2 = read_rows(rows_in(joinpath(dir, "t2"), "rows_"))
            @test any(r -> r["solver"] == "MLEM->GD", t2) && any(r -> r["rep"] == "3" && r["solver"] == "GN-0" && r["iter"] == "final", t2)
            # the selected K is always among the logged iterates (also for a GD K outside the default list)
            selg = deepcopy(selt); selg[group_key("GD-0", "X", 1)]["iter"] = "K=3"
            ctx3 = ProtocolContext(op, geo, :test, Dict("X" => op.predict(wt)), [1e4], 1, 42, joinpath(dir, "t3"))
            test_stage!(ctx3, selg, "sha"; solvers = ("GD-0",))
            @test any(r -> r["solver"] == "GD-0" && r["iter"] == "K=3" && r["selected"] == "1", read_rows(rows_in(joinpath(dir, "t3"), "rows_")))
        end
    end

    @testset "process count of the driver script" begin
        script = joinpath(@__DIR__, "..", "examples", "s7_protocol.sh")
        mktempdir() do dir
            env = Dict("OUT" => dir, "REPO" => joinpath(@__DIR__, ".."))
            p = run(pipeline(ignorestatus(setenv(`bash $script test`, merge(env, Dict("NPROC" => "65"))));
                             stderr = joinpath(dir, "err.txt"), stdout = devnull))
            @test p.exitcode != 0
            @test occursin("NPROC=65 exceeds", read(joinpath(dir, "err.txt"), String))
            @test !isdir(joinpath(dir, "test"))                                 # refused before anything ran
            @test occursin("JULIA_NUM_THREADS=1", read(script, String))
        end
    end

    @testset "selection tie and edge rules" begin
        mk(solver, p, it, v) = Dict("solver" => solver, "arm" => "M", "exp_index" => "1", "weights" => "obs",
                                    "param" => string(p), "iter" => it, "rmse_s" => string(v))
        rows = Dict{String,String}[]
        for rep in 1:2
            push!(rows, mk("MLEM", 0.5, "n=25", 0.2)); push!(rows, mk("MLEM", 1.6, "n=25", 0.2))   # exact tie in the weight
            push!(rows, mk("MLEM", 1.6, "n=50", 0.2)); push!(rows, mk("MLEM", 0.05, "n=25", 0.3))
        end
        (p, lbl), v = Tomography.best_candidate(Tomography.candidate_means(rows, "MLEM", "M", 1, 2))
        @test p == 1.6 && lbl == "n=25"                      # larger weight, then smaller count
        # incomplete candidates (fewer repetitions than expected) are ignored
        push!(rows, mk("MLEM", 5.0, "n=100", 0.01))
        @test Tomography.candidate_means(rows, "MLEM", "M", 1, 2) |> keys |> collect |> k -> !((5.0, "n=100") in k)
        # edge rule
        @test Tomography.extend_value([0.0, 0.05, 0.16, 0.5, 1.6, 5.0], :lo) === nothing
        @test Tomography.extend_value([0.0, 0.05, 0.16, 0.5, 1.6, 5.0], :hi) ≈ 5 * 5 / 1.6
        @test Tomography.extend_value([10, 25, 50, 100, 200, 400], :lo; integer = true) == 4
        @test Tomography.extend_value([10, 25, 50, 100, 200, 400], :hi; integer = true) == 800
        @test Tomography.extend_value([1, 2, 5], :lo; integer = true) === nothing
        @test Tomography.extend_value([0.0, 0.03, 0.1, 0.3], :hi; cap = 1.0) ≈ 0.9
        @test Tomography.extend_value([0.0, 0.03, 0.1, 0.3, 0.9], :hi; cap = 1.0) == 1.0
        @test Tomography.extend_value([0.0, 0.03, 0.1, 0.3, 0.9, 1.0], :hi; cap = 1.0) === nothing
        rows = Dict{String,String}[]
        for rep in 1:2
            push!(rows, mk("SART", 0.1, "n=10", 0.1)); push!(rows, mk("SART", 0.03, "n=10", 0.2))
            push!(rows, mk("MLEM", 5.0, "n=100", 0.1)); push!(rows, mk("MLEM", 1.6, "n=100", 0.2))
        end
        grids = default_grids(["M"], 1)
        sel, pending = Tomography.select_phase!(rows, 1, grids, ["M"], [1e4], 2)
        @test pending == 1                                    # MLEM sits at the top weight; SART is interior
        @test sel["SART|M|1"]["param"] == 0.1 && sel["SART|M|1"]["extensions_pending"] == false
        @test sel["MLEM|M|1"]["extensions_pending"] == true && 5 * 5 / 1.6 in grids["MLEM|M|1"]["param"]
        @test grids["MLEM|M|1"]["ext"]["param_hi"] == 1
        # after two extensions the edge is kept and reported
        grids["MLEM|M|1"]["ext"]["param_hi"] = 2
        grids["MLEM|M|1"]["param"] = filter(<=(5.0), grids["MLEM|M|1"]["param"])
        sel, pending = Tomography.select_phase!(rows, 1, grids, ["M"], [1e4], 2)
        @test pending == 0 && occursin("param_hi", sel["MLEM|M|1"]["at_edge"])
    end
end
