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
            exposures = [1e4, 1e5]
            geo_d = make_geometry(:dev, w_dev, C, op.free)
            geo_t = make_geometry(:test, w_test, C, op.free)
            templates_d = Dict("M" => op.predict(w_dev), "X" => op.predict(w_dev) .* 0.9)
            templates_t = Dict("M" => op.predict(w_test), "X" => op.predict(w_test) .* 0.9)
            devdir = joinpath(dir, "dev")
            ctx = ProtocolContext(op, geo_d, :dev, templates_d, exposures, 2, 42, devdir)
            arms = ["M", "X"]
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
            @test sha == sha2 && length(sel) == length(PROTOCOL_SOLVERS) * 2 * 2
            # the test stage refuses a missing or altered selection
            testdir = joinpath(dir, "test")
            tctx = ProtocolContext(op, geo_t, :test, templates_t, exposures, 2, 42, testdir)
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
            @test count(r -> r["selected"] == "1" && r["solver"] == "GN-0", trows) == 2 * 2 * 2
            @test_throws ArgumentError test_stage!(ProtocolContext(op, geo_d, :dev, templates_d, exposures, 1, 42, testdir), sel, sha)
            # oracle-weight diagnostic goes to its own directory and is labelled
            diagdir = joinpath(dir, "diag")
            dctx = ProtocolContext(op, geo_t, :test, templates_t, exposures, 2, 42, diagdir)
            test_stage!(dctx, sel, sha; weights = :oracle, solvers = ("GN-0",), arms = ["M"], stage = "diag")
            drows = read_rows(rows_in(diagdir, "rows_"))
            @test !isempty(drows) && all(r -> r["weights"] == "oracle" && r["arm"] == "M" && r["solver"] == "GN-0", drows)
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
