# ===========================================================================
# Pre-registered synthetic reconstruction protocol ("S7").
#
# Included into `module Tomography` (shares its imports/namespace).
#
# Everything that does not need the physics set-up lives here so that it can be unit
# tested on small toy operators: Poisson observations with paired seeds, weights built
# from the observed counts only, the ray-crossed free set, the crossed-cell metrics, the
# truth fields, the per-solver runners, the development stage (grids, selection, tie and
# edge rules), the frozen `selection.toml` and the test / diagnostic stages.
# `examples/lvd_tomography.jl` only assembles the real operator and calls these functions.
# ===========================================================================

using SHA
using Serialization
using SpecialFunctions: gamma_inc

# --------------------------------------------------------------------------
# Exposure grid, seeds, Poisson observations
# --------------------------------------------------------------------------

"Per-bin exposure grid of the protocol (7 values)."
const PROTOCOL_EXPOSURES = [1.0e3, 3.16e3, 1.0e4, 3.16e4, 1.0e5, 1.0e6, 1.0e8]
"Data arms: `M` = fixed-path Monte Carlo template, `X` = exact (corrected CSDA) template."
const PROTOCOL_ARMS = ("M", "X")

"""
    parse_exposure_list(text) -> Vector{Float64}

Explicit exposure list of `--sweep-exposures` (comma separated, positive, no repeats,
kept in the given order).
"""
function parse_exposure_list(text::AbstractString)
    vals = Float64[]
    for token in split(text, ',')
        t = strip(token)
        isempty(t) && continue
        v = tryparse(Float64, t)
        (v === nothing || !(v > 0) || !isfinite(v)) && throw(ArgumentError("bad exposure '$t'"))
        push!(vals, v)
    end
    isempty(vals) && throw(ArgumentError("empty exposure list"))
    length(unique(vals)) == length(vals) || throw(ArgumentError("repeated exposure in list"))
    return vals
end

"""
    protocol_noise_seed(field, arm, exp_index, rep; base=42) -> Int

`base + 9000 + 100000 f + 10000 a + 100 i + r` with `f = 0` for the development field
(`:dev`) and `1` for the test field (`:test`), `a = 0` for arm `"M"` and `1` for `"X"`,
`i` the exposure index (1-based) and `r` the realisation index (1-based).
"""
function protocol_noise_seed(field::Symbol, arm::AbstractString, exp_index::Integer, rep::Integer; base::Integer = 42)
    f = field === :dev ? 0 : field === :test ? 1 : throw(ArgumentError("field must be :dev or :test"))
    a = arm == "M" ? 0 : arm == "X" ? 1 : throw(ArgumentError("arm must be \"M\" or \"X\""))
    return Int(base) + 9000 + 100000f + 10000a + 100 * Int(exp_index) + Int(rep)
end

"Smallest integer `k >= 0` with `P(Poisson(λ) <= k) >= u` (inverse transform sampling)."
function poisson_quantile(u::Real, λ::Real)
    (λ <= 0 || u <= 0) && return 0
    cdf(k) = gamma_inc(k + 1.0, Float64(λ))[2]          # P(K <= k) = Q(k+1, λ)
    hi = ceil(Int, λ + 8sqrt(λ) + 10)
    while cdf(hi) < u
        hi *= 2
    end
    lo = -1
    while hi - lo > 1
        mid = (lo + hi) >> 1
        cdf(mid) >= u ? (hi = mid) : (lo = mid)
    end
    return hi
end

"""
    draw_poisson_counts(template, tau, seed) -> Vector{Int}

Counts `k_m ~ Poisson(tau * max(template_m, 0))`. One vector of `length(template)`
uniform draws is taken from `MersenneTwister(seed)` and inverted through the Poisson
quantile, so the stream consumed per seed is fixed and the draws are monotonically
coupled across exposures for a fixed seed.
"""
function draw_poisson_counts(template::AbstractVector{<:Real}, tau::Real, seed::Integer)
    u = rand(MersenneTwister(seed), length(template))
    return [poisson_quantile(u[m], tau * max(template[m], 0.0)) for m in eachindex(template)]
end

"""
One realisation of the protocol. The solver path only ever sees `counts` and `obs`
(`obs = counts / tau`); the noise-free template is deliberately not part of it.
"""
struct ProtocolUnit
    field::Symbol
    arm::String
    exp_index::Int
    tau::Float64
    rep::Int
    seed::Int
    counts::Vector{Int}
    obs::Vector{Float64}
end

function make_unit(template::AbstractVector{<:Real}, field::Symbol, arm::AbstractString,
                   exp_index::Integer, tau::Real, rep::Integer; base_seed::Integer = 42)
    seed = protocol_noise_seed(field, arm, exp_index, rep; base = base_seed)
    counts = draw_poisson_counts(template, tau, seed)
    return ProtocolUnit(field, String(arm), Int(exp_index), Float64(tau), Int(rep), seed,
                        counts, counts ./ tau)
end

# --------------------------------------------------------------------------
# Weights, free set, prior scales
# --------------------------------------------------------------------------

"""
    protocol_weights(counts, tau) -> Vector

`W = 1/σ²`, `σ² = max(k, 1)/tau²`: computed from the observed counts only. With a
one-count floor this weighted least-squares objective is not the Poisson likelihood
below about 10 counts per bin.
"""
protocol_weights(counts::AbstractVector{<:Real}, tau::Real) = Float64(tau)^2 ./ max.(counts, 1)

"""
    oracle_weights(template, tau) -> Vector

The pre-protocol rule (`σ = sqrt(max(template,0)/tau)`, floor `1e-12 max(template)`),
built from the noise-free template. Diagnostic only: a real fit has no such template.
"""
function oracle_weights(template::AbstractVector{<:Real}, tau::Real)
    σ = sqrt.(max.(template, 0.0) ./ tau)
    σ .= max.(σ, 1e-12 * maximum(template))
    return 1.0 ./ σ .^ 2
end

"Boolean mask of the cells with a nonzero column in the (dry) Jacobian."
sensed_cells(J::AbstractMatrix) = Bool[any(!iszero, view(J, :, j)) for j in axes(J, 2)]
sensed_cells(J::SparseMatrixCSC) = Bool[J.colptr[j+1] > J.colptr[j] && any(!iszero, view(J.nzval, J.colptr[j]:J.colptr[j+1]-1)) for j in 1:size(J, 2)]

"`h_+`: median over the free columns of `Σ_m W_m J_mj²` (positive entries)."
function curvature_scale(W::AbstractVector, J::AbstractMatrix, free::AbstractVector{Bool})
    d = vec(sum(W .* (J .^ 2); dims = 1))[free]
    m = filter(>(0), d)
    return isempty(m) ? 1.0 : median(m)
end

"`s_+`: median positive column sum of `max(J, 0)`."
function positive_sensitivity_scale(J::AbstractMatrix)
    m = filter(>(0), vec(sum(max.(J, 0.0); dims = 1)))
    return isempty(m) ? 1.0 : median(m)
end

# --------------------------------------------------------------------------
# Truth fields
# --------------------------------------------------------------------------

"""
    field_structures(which, centroids) -> (names, masks, values)

Structure rules on cell centroids (3 × n, metres). `:dev` = the existing rich field
(slab, sheet, lens: 0.3, 0.5, 0.9). `:test` = the held-out field (a: upper slab 0.4,
b: sheet dipping along y 0.6, c: off-axis body 0.8). Overlaps resolve by `max`.
"""
function field_structures(which::Symbol, centroids::AbstractMatrix)
    x = centroids[1, :]; y = centroids[2, :]; z = centroids[3, :]
    if which === :dev
        return (["slab", "sheet", "lens"],
                [BitVector(@. (abs(z - 479.4) <= 160.0) & (abs(x) <= 1600.0) & (abs(y) <= 1600.0)),
                 BitVector(@. (abs((z - 799.0) - 0.4x) <= 220.0) & (abs(y) <= 1600.0)),
                 BitVector(@. (abs(z - 1118.6) <= 160.0) & (abs(x) <= 600.0) & (abs(y) <= 600.0))],
                [0.3, 0.5, 0.9])
    elseif which === :test
        return (["a", "b", "c"],
                [BitVector(@. (abs(z - 1438.2) <= 160.0) & (-2600.0 <= x <= -400.0) & (abs(y) <= 1600.0)),
                 BitVector(@. (abs((z - 799.0) + 0.4y) <= 220.0) & (abs(x) <= 600.0)),
                 BitVector(@. (abs(z - 1118.6) <= 160.0) & (-1600.0 <= x <= -400.0) & (400.0 <= y <= 1600.0))],
                [0.4, 0.6, 0.8])
    end
    throw(ArgumentError("field must be :dev or :test"))
end

"Held-out test field `T` on the given centroids."
function protocol_test_field(centroids::AbstractMatrix)
    _, masks, values = field_structures(:test, centroids)
    w = zeros(Float64, size(centroids, 2))
    for (m, v) in zip(masks, values)
        @. w = ifelse(m, max(w, v), w)
    end
    return w
end

"""
Truth field, evaluation domain and structure masks of one field.
`Ds` is the primary evaluation domain: free (ray-crossed) cells above the depth mask.
"""
struct ProtocolGeometry
    which::Symbol
    w_true::Vector{Float64}
    eval_mask::BitVector
    free::Vector{Bool}
    Ds::BitVector
    anom::BitVector
    bg::BitVector
    struct_names::Vector{String}
    struct_masks::Vector{BitVector}
end

function make_geometry(which::Symbol, w_true::AbstractVector{<:Real}, centroids::AbstractMatrix,
                       free::AbstractVector{Bool}; min_eval_depth::Real = 350.0)
    names, masks, _ = field_structures(which, centroids)
    em = BitVector(centroids[3, :] .>= min_eval_depth)
    Ds = em .& BitVector(free)
    w = collect(Float64, w_true)
    return ProtocolGeometry(which, w, em, collect(Bool, free), Ds, Ds .& (w .> 0), Ds .& .!(w .> 0), names, masks)
end

"""
    gate_g1_report(geo) -> NamedTuple

Truth-only report: nonzero cells of each structure inside `Ds`. Gate G1 passes when
every structure has at least 3 and the total lies between 12 and 40.
"""
function gate_g1_report(geo::ProtocolGeometry)
    per = [count(geo.struct_masks[k] .& geo.Ds .& (geo.w_true .> 0)) for k in eachindex(geo.struct_names)]
    total = count(geo.Ds .& (geo.w_true .> 0))
    passed = all(>=(3), per) && 12 <= total <= 40
    zero_rmse = sqrt(sum(abs2, geo.w_true[geo.Ds]) / count(geo.Ds))
    return (field = geo.which, structures = geo.struct_names, per_structure = per,
            per_structure_all_cells = [count(m .& (geo.w_true .> 0)) for m in geo.struct_masks],
            nonzero_in_Ds = total, nonzero_total = count(>(0), geo.w_true),
            n_free = count(geo.free), n_Ds = count(geo.Ds), n_eval = count(geo.eval_mask),
            zero_field_rmse_s = zero_rmse, passed = passed)
end

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

_rms(v) = isempty(v) ? NaN : sqrt(sum(abs2, v) / length(v))

"""
    protocol_metrics(w, geo) -> NamedTuple

Primary `rmse_s` over `Ds`, plus anomaly RMSE, background mean and RMSE, mean recovered
`w` per structure (inside `Ds`), Pearson correlation over `Ds` and the legacy
500-cell RMSE (`rmse500`, evaluation mask).
"""
function protocol_metrics(w::AbstractVector, geo::ProtocolGeometry)
    e = w .- geo.w_true
    wd = w[geo.Ds]; td = geo.w_true[geo.Ds]
    corr = (std(wd) > 0 && std(td) > 0) ? cor(wd, td) : NaN
    return (rmse_s = _rms(e[geo.Ds]), rmse500 = _rms(e[geo.eval_mask]),
            anom_rmse = _rms(e[geo.anom]),
            bg_mean = any(geo.bg) ? mean(w[geo.bg]) : NaN, bg_rmse = _rms(e[geo.bg]),
            struct_means = [any(m .& geo.Ds) ? mean(w[m .& geo.Ds]) : NaN for m in geo.struct_masks],
            corr = corr)
end

"Data residual `relres = ‖pred − obs‖/‖obs‖` and `chi2/M` with weights `W`."
function protocol_fit_metrics(pred::AbstractVector, obs::AbstractVector, W::AbstractVector)
    r = pred .- obs
    return (relres = norm(r) / norm(obs), chi2m = sum(W .* r .^ 2) / length(obs))
end

# --------------------------------------------------------------------------
# Operator bundle and solver runners
# --------------------------------------------------------------------------

"""
    ProtocolOperator(J0, f0, model, predict, neighbors, free; delta=0.03, box=(0,0.9))

`J0`/`f0`: dry Jacobian and dry flux on the valid bins; `model(w) -> (pred, J)`,
`predict(w) -> pred` the nonlinear operator (full and flux-only); `free` the set `S`.
"""
struct ProtocolOperator{M,P}
    J0::Matrix{Float64}
    f0::Vector{Float64}
    model::M
    predict::P
    neighbors::Vector{Vector{Int}}
    free::Vector{Bool}
    delta::Float64
    box::Tuple{Float64,Float64}
    s_plus::Float64
    A_pos::Matrix{Float64}
end

function ProtocolOperator(J0::AbstractMatrix, f0::AbstractVector, model, predict,
                          neighbors::Vector{Vector{Int}}, free::AbstractVector{Bool};
                          delta::Real = 0.03, box::Tuple{Float64,Float64} = (0.0, MAX_WATER_FRACTION))
    J = Matrix{Float64}(J0)
    return ProtocolOperator(J, collect(Float64, f0), model, predict, neighbors, collect(Bool, free),
                            Float64(delta), box, positive_sensitivity_scale(J), max.(J, 0.0))
end

mutable struct EvalCounter
    fwd::Int
    jac::Int
end

"Result of one solver run: `vectors` maps a label (`n=25`, `K=3`, `final`) to a field."
struct SolveRun
    vectors::Vector{Pair{String,Vector{Float64}}}
    secs::Float64
    fwd::Int
    jac::Int
    updates::Int
    stop::String
    update_log::Vector{Any}
end

_iter_label(prefix, k) = k === :final ? "final" : "$prefix=$k"

"SART with Laplacian smoothing strength `q` (free cells only); fields at every `n_list` count."
function run_sart(op::ProtocolOperator, obs::AbstractVector, q::Real, n_list::AbstractVector{<:Integer})
    t0 = time()
    snaps = Dict{Int,Vector{Float64}}()
    prior = q > 0 ? SmoothnessPrior(op.neighbors, 1.0) : nothing
    sart_reconstruct(op.J0, obs .- op.f0; n_iter = maximum(n_list), relaxation = 0.2, box = op.box,
                     prior = prior, smooth_strength = Float64(q), free = op.free,
                     snapshots = snaps, snapshot_iters = n_list)
    return SolveRun([_iter_label("n", n) => snaps[n] for n in n_list], time() - t0, 0, 0,
                    maximum(n_list), "n_iter", Any[])
end

"MLEM on `max(J0,0)` / `max(obs − f0, 0)`, penalty weight `c_M · s_+`, start 1e-3 on the free cells."
function run_mlem(op::ProtocolOperator, obs::AbstractVector, cM::Real, n_list::AbstractVector{<:Integer})
    t0 = time()
    snaps = Dict{Int,Vector{Float64}}()
    prior = cM > 0 ? SmoothnessPrior(op.neighbors, cM * op.s_plus) : nothing
    mlem_reconstruct(op.A_pos, max.(obs .- op.f0, 0.0); n_iter = maximum(n_list), box = op.box,
                     prior = prior, free = op.free, snapshots = snaps, snapshot_iters = n_list)
    return SolveRun([_iter_label("n", n) => snaps[n] for n in n_list], time() - t0, 0, 0,
                    maximum(n_list), "n_iter", Any[])
end

"""
Gauss-Newton run (at most `max_updates` outer updates; stop on projected gradient
<= 1e-6, relative objective decrease < `rel_tol`, or no accepted step). `mode` is
`:lin` (frozen operator, prediction `f0 + J0 w`), or `:relin` (relinearised; start `w_start`).
"""
function run_gn(op::ProtocolOperator, obs::AbstractVector, W::AbstractVector, c::Real,
                K_list, mode::Symbol, w_start::AbstractVector; max_updates::Int = 20, rel_tol::Float64 = 1e-5)
    mode in (:lin, :relin) || throw(ArgumentError("mode must be :lin or :relin"))
    ctr = EvalCounter(0, 0)
    model = w -> (ctr.jac += 1; op.model(w))
    predict = w -> (ctr.fwd += 1; op.predict(w))
    prior = EdgePrior(op.neighbors, c * curvature_scale(W, op.J0, op.free), op.delta)
    iterates = Vector{Vector{Float64}}(); log = Any[]; stop = Ref(:none)
    t0 = time()
    gauss_newton_reconstruct(model, obs; w0 = w_start, n_iter = max_updates, prior = prior,
        solver_mode = :active_set, weights = W, relinearize = (mode === :relin), box = op.box,
        predict_only = mode === :relin ? predict : nothing, free = op.free,
        rel_objective_tol = rel_tol, iterates = iterates, update_log = log, stop = stop)
    secs = time() - t0
    nup = length(iterates) - 1
    pick(K) = K === :final ? iterates[end] : iterates[min(Int(K), nup) + 1]
    return SolveRun([_iter_label("K", K) => copy(pick(K)) for K in K_list], secs, ctr.fwd, ctr.jac,
                    nup, String(stop[]), log)
end

"Projected, Jacobi-preconditioned gradient descent (at most `max_iter` iterations)."
function run_gd(op::ProtocolOperator, obs::AbstractVector, W::AbstractVector, c::Real,
                K_list, w_start::AbstractVector; max_iter::Int = 40)
    ctr = EvalCounter(0, 0)
    model = w -> (ctr.jac += 1; op.model(w))
    prior = EdgePrior(op.neighbors, c * curvature_scale(W, op.J0, op.free), op.delta)
    snaps = Dict{Int,Vector{Float64}}()
    t0 = time()
    w, hist = gradient_descent_reconstruct(model, obs; w0 = w_start, n_iter = max_iter, lr = 1.0,
        optimizer = :pgd, prior = prior, weights = W, relinearize = true, box = op.box,
        free = op.free, snapshots = snaps)
    secs = time() - t0
    niter = length(hist)
    pick(K) = K === :final ? w : snaps[min(Int(K), niter)]
    return SolveRun([_iter_label("K", K) => copy(pick(K)) for K in K_list], secs, ctr.fwd, ctr.jac,
                    niter, niter < max_iter ? "objective_change" : "max_iter", Any[])
end

# --------------------------------------------------------------------------
# Grids, TSV logging
# --------------------------------------------------------------------------

const PROTOCOL_SOLVERS = ("SART", "MLEM", "GN-lin", "GN-0", "MLEM->GN", "GD-0", "MLEM->GD")
const PROTOCOL_PHASES = Dict(1 => ("SART", "MLEM"), 2 => ("GN-lin", "GN-0", "MLEM->GN"), 3 => ("GD-0", "MLEM->GD"))

"Initial hyperparameter grids (Q2.7) of one group `solver|arm|exp_index`."
function default_grid(solver::AbstractString)
    solver == "SART" && return Dict{String,Any}("param" => [0.0, 0.03, 0.1, 0.3], "iter" => [1, 2, 5, 10, 25, 50, 100, 200])
    solver == "MLEM" && return Dict{String,Any}("param" => [0.0, 0.05, 0.16, 0.5, 1.6, 5.0], "iter" => [10, 25, 50, 100, 200, 400])
    solver in ("GN-lin", "GN-0", "MLEM->GN") && return Dict{String,Any}("param" => [0.001, 0.01, 0.1, 1.0], "iter" => [1, 2, 3, 5, 8])
    solver in ("GD-0", "MLEM->GD") && return Dict{String,Any}("param" => Float64[], "iter" => [5, 10, 20])
    throw(ArgumentError("unknown solver $solver"))
end

group_key(solver, arm, exp_index) = "$solver|$arm|$exp_index"

function default_grids(arms, n_exposures::Integer)
    g = Dict{String,Any}()
    for s in PROTOCOL_SOLVERS, a in arms, i in 1:n_exposures
        d = default_grid(s)
        d["ext"] = Dict{String,Any}("param_lo" => 0, "param_hi" => 0, "iter_lo" => 0, "iter_hi" => 0)
        g[group_key(s, a, i)] = d
    end
    return g
end

const PROTOCOL_COLUMNS = ["stage", "field", "arm", "exp_index", "tau", "rep", "seed", "solver", "start",
    "weights", "param", "iter", "selected", "rmse_s", "rmse500", "anom_rmse", "bg_mean", "bg_rmse",
    "s1_mean", "s2_mean", "s3_mean", "corr", "relres", "chi2m", "secs", "fwd", "jac", "updates", "stop",
    "rec", "selection_sha256"]

function append_rows(path::AbstractString, rows::Vector{Dict{String,String}})
    isempty(rows) && return
    mkpath(dirname(abspath(path)))
    new = !isfile(path)
    open(path, "a") do io
        new && println(io, join(PROTOCOL_COLUMNS, '\t'))
        for r in rows
            println(io, join((get(r, c, "") for c in PROTOCOL_COLUMNS), '\t'))
        end
    end
end

function read_rows(files)
    rows = Dict{String,String}[]
    for f in files
        isfile(f) || continue
        lines = readlines(f)
        isempty(lines) && continue
        cols = split(lines[1], '\t')
        for l in lines[2:end]
            isempty(l) && continue
            push!(rows, Dict{String,String}(zip(cols, split(l, '\t'; keepempty = true))))
        end
    end
    return rows
end

rows_in(dir, stem) = isdir(dir) ? sort([joinpath(dir, f) for f in readdir(dir) if startswith(f, stem) && endswith(f, ".tsv")]) : String[]

"Identity of a logged row (used for resume)."
row_key(r) = join((r["field"], r["arm"], r["exp_index"], r["rep"], r["solver"], r["start"], r["weights"], r["param"], r["iter"]), '|')

_s(x) = x isa AbstractString ? String(x) : string(x)

"""
Build one log row for field `w` of a run (`label` = `n=..`, `K=..` or `final`).
`rec` names the file holding the saved vector.
"""
function protocol_log_row(stage, geo::ProtocolGeometry, op::ProtocolOperator, unit::ProtocolUnit, W,
                          solver, start, weights, param, label::AbstractString, w, run::SolveRun;
                          selected = true, rec = "", selection_sha256 = "")
    m = protocol_metrics(w, geo)
    f = protocol_fit_metrics(op.predict(w), unit.obs, W)
    sm = [k <= length(m.struct_means) ? m.struct_means[k] : NaN for k in 1:3]
    return Dict{String,String}(
        "stage" => stage, "field" => String(unit.field), "arm" => unit.arm, "exp_index" => _s(unit.exp_index),
        "tau" => _s(unit.tau), "rep" => _s(unit.rep), "seed" => _s(unit.seed), "solver" => solver,
        "start" => start, "weights" => weights, "param" => _s(param), "iter" => label,
        "selected" => selected ? "1" : "0", "rmse_s" => _s(m.rmse_s), "rmse500" => _s(m.rmse500),
        "anom_rmse" => _s(m.anom_rmse), "bg_mean" => _s(m.bg_mean), "bg_rmse" => _s(m.bg_rmse),
        "s1_mean" => _s(sm[1]), "s2_mean" => _s(sm[2]), "s3_mean" => _s(sm[3]), "corr" => _s(m.corr),
        "relres" => _s(f.relres), "chi2m" => _s(f.chi2m), "secs" => _s(run.secs), "fwd" => _s(run.fwd),
        "jac" => _s(run.jac), "updates" => _s(run.updates), "stop" => run.stop, "rec" => rec,
        "selection_sha256" => selection_sha256)
end

# --------------------------------------------------------------------------
# Templates (explicit cache key; a mismatching cache is refused, never reused)
# --------------------------------------------------------------------------

"""
    template_key(field, arm, histories, seed, w_true, valid_bins) -> NamedTuple

Cache key of a template: field, arm, MC histories, base seed, and content hashes of the
truth field and of the valid-bin list.
"""
template_key(field::Symbol, arm::AbstractString, histories::Integer, seed::Integer, w_true, valid_bins) =
    (schema = 1, field = String(field), arm = String(arm), histories = Int(histories), seed = Int(seed),
     w_true_sha256 = bytes2hex(sha256(reinterpret(UInt8, collect(Float64, w_true)))),
     valid_bins_sha256 = bytes2hex(sha256(reinterpret(UInt8, collect(Int, valid_bins)))))

function save_template(path::AbstractString, key, template::AbstractVector, extra = nothing)
    mkpath(dirname(abspath(path)))
    tmp = path * ".tmp"
    Serialization.serialize(tmp, (key = key, template = collect(Float64, template), extra = extra))
    mv(tmp, path; force = true)
    return path
end

"Load a template; error if the file is missing or its stored key differs from `key`."
function load_template(path::AbstractString, key)
    isfile(path) || error("template cache $path is missing")
    c = Serialization.deserialize(path)
    c.key == key || error("template cache $path does not match the requested key (field, arm, histories, seed, truth field or bins differ); refusing to reuse it")
    return c
end

# --------------------------------------------------------------------------
# Selection (development stage)
# --------------------------------------------------------------------------

_iterkey(lbl::AbstractString) = lbl == "final" ? typemax(Int) : parse(Int, split(lbl, '=')[end])

"Strictly-keyed means of `rmse_s` over the repetitions for each (param, iter) candidate of a group."
function candidate_means(rows, solver, arm, exp_index, nreps)
    acc = Dict{Tuple{Float64,String},Vector{Float64}}()
    for r in rows
        r["solver"] == solver && r["arm"] == arm && r["exp_index"] == string(exp_index) && r["weights"] == "obs" || continue
        push!(get!(acc, (parse(Float64, r["param"]), r["iter"]), Float64[]), parse(Float64, r["rmse_s"]))
    end
    return Dict(k => mean(v) for (k, v) in acc if length(v) == nreps)
end

"Lowest mean `rmse_s`; exact ties go to the larger prior weight, then the smaller iteration count."
function best_candidate(cands::Dict)
    isempty(cands) && error("no complete candidate in group")
    ks = collect(keys(cands))
    sort!(ks; by = k -> (cands[k], -k[1], _iterkey(k[2])))
    return ks[1], cands[ks[1]]
end

_round_half_even_int(x) = max(1, round(Int, x))

"""
Next grid value beyond the lower or upper end of `vals` with the same ratio as the two
edge values, or `nothing` where the grid cannot be extended (zero, `q > 1`, 1).
"""
function extend_value(vals::Vector, dir::Symbol; integer::Bool = false, cap::Real = Inf)
    v = sort(vals)
    length(v) >= 2 || return nothing
    if dir === :lo
        v[1] > 0 || return nothing
        integer && v[1] <= 1 && return nothing
        new = v[1] / (v[2] / v[1])
        return integer ? _round_half_even_int(new) : new
    else
        v[end] >= cap && return nothing
        new = v[end] * (v[end] / v[end-1])
        return integer ? _round_half_even_int(new) : min(new, cap)
    end
end

"""
    select_phase!(rows, phase, grids, arms, exposures, nreps; q_cap=1.0) -> (selection, pending)

Takes all development rows, applies the selection, tie and edge rules for the solvers
of `phase`, extends `grids` in place where the selection sits at a grid edge (at most
twice per direction; zero, `q > 1`, `K = 1` and `final` are hard bounds) and returns the
provisional selection plus the number of grid extensions made.
"""
function select_phase!(rows, phase::Int, grids::Dict, arms, exposures, nreps::Int; q_cap::Real = 1.0)
    selection = Dict{String,Any}()
    pending = 0
    for solver in PROTOCOL_PHASES[phase], arm in arms, i in eachindex(exposures)
        key = group_key(solver, arm, i)
        g = grids[key]
        cands = candidate_means(rows, solver, arm, i, nreps)
        isempty(cands) && error("no complete development rows for group $key")
        (p, lbl), val = best_candidate(cands)
        iterval = lbl == "final" ? nothing : _iterkey(lbl)
        ext = g["ext"]
        edge = String[]
        params = Float64.(g["param"]); iters = Int.(g["iter"])
        iter_integer_extendable = solver in ("SART", "MLEM", "GD-0", "MLEM->GD")
        extended = false
        if !isempty(params)
            cap = solver == "SART" ? q_cap : Inf
            if p == minimum(params)
                nv = extend_value(params, :lo)
                if nv !== nothing && ext["param_lo"] < 2
                    push!(g["param"], nv); ext["param_lo"] += 1; extended = true
                else
                    push!(edge, "param_lo")
                end
            elseif p == maximum(params)
                nv = extend_value(params, :hi; cap = cap)
                if nv !== nothing && nv > maximum(params) && ext["param_hi"] < 2
                    push!(g["param"], nv); ext["param_hi"] += 1; extended = true
                else
                    push!(edge, "param_hi")
                end
            end
        end
        if iterval !== nothing && iter_integer_extendable
            if iterval == minimum(iters)
                nv = extend_value(iters, :lo; integer = true)
                if nv !== nothing && nv < minimum(iters) && ext["iter_lo"] < 2
                    push!(g["iter"], nv); ext["iter_lo"] += 1; extended = true
                else
                    push!(edge, "iter_lo")
                end
            elseif iterval == maximum(iters) && solver in ("SART", "MLEM")
                nv = extend_value(iters, :hi; integer = true)
                if nv !== nothing && ext["iter_hi"] < 2
                    push!(g["iter"], nv); ext["iter_hi"] += 1; extended = true
                else
                    push!(edge, "iter_hi")
                end
            end
        end
        extended && (pending += 1)
        selection[key] = Dict{String,Any}("solver" => solver, "arm" => arm, "exp_index" => i,
            "tau" => exposures[i], "param" => p, "iter" => lbl, "dev_rmse_s_mean" => val,
            "at_edge" => join(edge, ","), "extensions_pending" => extended)
    end
    return selection, pending
end

# --------------------------------------------------------------------------
# selection.toml (frozen; the test stage refuses to run without it)
# --------------------------------------------------------------------------

function _toml_text(d::Dict)
    io = IOBuffer(); TOML.print(io, d; sorted = true); String(take!(io))
end

"Write `selection.toml` (+ `selection.toml.sha256`) from the final per-group selections."
function write_selection(path::AbstractString, selection::Dict, meta::Dict)
    body = Dict{String,Any}("selection" => selection)
    sha = bytes2hex(sha256(_toml_text(body)))
    full = Dict{String,Any}("meta" => merge(Dict{String,Any}(meta), Dict("content_sha256" => sha)), "selection" => selection)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io; TOML.print(io, full; sorted = true); end
    write(path * ".sha256", sha * "\n")
    return sha
end

"""
    load_selection(path) -> (selection, sha256, meta)

The test and diagnostic stages call this first. It errors if the file is missing or if
its content hash does not match.
"""
function load_selection(path::AbstractString)
    isfile(path) || error("selection.toml not found at $path: the test stage refuses to run before the development selection is frozen")
    d = TOML.parsefile(path)
    haskey(d, "selection") && haskey(d, "meta") || error("$path is not a selection file")
    sha = bytes2hex(sha256(_toml_text(Dict{String,Any}("selection" => d["selection"]))))
    sha == get(d["meta"], "content_sha256", "") || error("selection file $path fails its content hash")
    return d["selection"], sha, d["meta"]
end

"Selected settings of a group: `(param, iter)` with `iter` an `Int` or `:final`."
function selected_setting(selection::Dict, solver, arm, exp_index)
    s = get(selection, group_key(solver, arm, exp_index), nothing)
    s === nothing && error("selection has no entry for $(group_key(solver, arm, exp_index))")
    lbl = s["iter"]
    return (param = Float64(s["param"]), iter = lbl == "final" ? :final : _iterkey(lbl), label = lbl)
end

# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

"""
Context of a stage run. `templates` maps arm => noise-free template (used for the
Poisson draw and the oracle diagnostic only); `units` are drawn once.
"""
struct ProtocolContext
    op::ProtocolOperator
    geo::ProtocolGeometry
    field::Symbol
    templates::Dict{String,Vector{Float64}}
    exposures::Vector{Float64}
    nreps::Int
    base_seed::Int
    outdir::String
end

function enumerate_units(ctx::ProtocolContext; arms = collect(keys(ctx.templates)), shard = (1, 1))
    units = Tuple{String,Int,Int}[]
    for arm in sort(collect(arms)), i in eachindex(ctx.exposures), r in 1:ctx.nreps
        push!(units, (arm, i, r))
    end
    return [u for (n, u) in enumerate(units) if (n - 1) % shard[2] == shard[1] - 1]
end

draw_unit(ctx::ProtocolContext, arm, i, r) =
    make_unit(ctx.templates[arm], ctx.field, arm, i, ctx.exposures[i], r; base_seed = ctx.base_seed)

function _unit_stem(stage, u::ProtocolUnit, solver, param)
    s = replace(solver, "->" => "to")
    return "$(stage)_$(u.field)_$(u.arm)_e$(u.exp_index)_r$(u.rep)_$(s)_$(replace(string(param), "." => "p"))"
end

function _save_run(dir, stage, u, solver, param, run::SolveRun)
    mkpath(joinpath(dir, "recs"))
    f = joinpath("recs", _unit_stem(stage, u, solver, param) * ".jls")
    Serialization.serialize(joinpath(dir, f), Dict(k => v for (k, v) in run.vectors))
    return f
end

function _mlem_vector(op, unit, setting)
    n = setting.iter === :final ? error("MLEM iteration count cannot be final") : setting.iter
    return only(run_mlem(op, unit.obs, setting.param, [n]).vectors).second
end

_iter_values(g) = [Int.(g["iter"]); :final]

"""
    dev_phase!(ctx, phase, grids; mlem_selection=nothing, gn_selection=nothing, shard=(1,1)) -> Int

Run (or resume) the development phase `phase` on this process's share of the units;
rows go to `<outdir>/rows_phase<phase>_shard<i>.tsv`, vectors to `<outdir>/recs/`.
Only combinations missing from the existing rows are computed. Returns the number of
solver runs done. `mlem_selection` (phase 1 selection) feeds `MLEM->GN`; `gn_selection`
(phase 2 selection) gives `c` for the GD solvers.
"""
function dev_phase!(ctx::ProtocolContext, phase::Int, grids::Dict;
                    mlem_selection = nothing, gn_selection = nothing, shard = (1, 1))
    haskey(PROTOCOL_PHASES, phase) || throw(ArgumentError("phase must be 1, 2 or 3"))
    ctx.field === :dev || throw(ArgumentError("the development stage runs on the development field"))
    done = Set(row_key(r) for r in read_rows(rows_in(ctx.outdir, "rows_phase")))
    out = joinpath(ctx.outdir, "rows_phase$(phase)_shard$(shard[1]).tsv")
    nruns = 0
    for (arm, i, r) in enumerate_units(ctx; shard = shard)
        u = draw_unit(ctx, arm, i, r)
        W = protocol_weights(u.counts, u.tau)
        zero = zeros(length(ctx.geo.w_true))
        for solver in PROTOCOL_PHASES[phase]
            g = grids[group_key(solver, arm, i)]
            start = "-"; w_start = zero
            if solver == "MLEM->GN"
                s = selected_setting(mlem_selection, "MLEM", arm, i)
                w_start = _mlem_vector(ctx.op, u, s); start = "mlem(c=$(s.param),n=$(s.label))"
            elseif solver == "MLEM->GD"
                s = selected_setting(mlem_selection, "MLEM", arm, i)
                w_start = _mlem_vector(ctx.op, u, s); start = "mlem(c=$(s.param),n=$(s.label))"
            elseif solver in ("GN-0", "GN-lin", "GD-0")
                start = "zero"
            end
            params = solver in ("GD-0", "MLEM->GD") ?
                [selected_setting(gn_selection, solver == "GD-0" ? "GN-0" : "MLEM->GN", arm, i).param] : Float64.(g["param"])
            for p in params
                labels = solver in ("SART", "MLEM") ? [_iter_label("n", n) for n in Int.(g["iter"])] :
                         [_iter_label("K", K) for K in _iter_values(g)]
                key(l) = join((String(ctx.field), arm, string(i), string(r), solver, start, "obs", string(p), l), '|')
                missing_labels = [l for l in labels if !(key(l) in done)]
                isempty(missing_labels) && continue
                run = if solver == "SART"
                    run_sart(ctx.op, u.obs, p, Int.(g["iter"]))
                elseif solver == "MLEM"
                    run_mlem(ctx.op, u.obs, p, Int.(g["iter"]))
                elseif solver in ("GN-lin", "GN-0", "MLEM->GN")
                    run_gn(ctx.op, u.obs, W, p, _iter_values(g), solver == "GN-lin" ? :lin : :relin, w_start)
                else
                    run_gd(ctx.op, u.obs, W, p, _iter_values(g), w_start)
                end
                nruns += 1
                rec = _save_run(ctx.outdir, "dev", u, solver, p, run)
                rows = [protocol_log_row("dev", ctx.geo, ctx.op, u, W, solver, start, "obs", p, lbl, w, run;
                                         rec = rec * "#" * lbl) for (lbl, w) in run.vectors if lbl in missing_labels]
                append_rows(out, rows)
                println("  dev phase $phase $arm e$i r$r $solver c/q=$p ", "$(round(run.secs; digits = 1))s"); flush(stdout)
            end
        end
    end
    return nruns
end

"""
    test_stage!(ctx, selection, sha; shard=(1,1), weights=:obs, solvers=("ZERO", PROTOCOL_SOLVERS...)) -> Int

Test (or diagnostic) stage with the frozen selection. For every unit it logs the zero
field and each solver at its selected setting; GN/GD runs also log their other K values
with `selected=0` (error-against-update diagnostics, never used for selection).
`weights=:oracle` draws the weights from the template (diagnostic only, labelled
`oracle` in the `weights` column). Rows go to `<outdir>/rows_shard<i>.tsv`; combinations
already present are skipped.
"""
function test_stage!(ctx::ProtocolContext, selection::Dict, sha::AbstractString;
                     shard = (1, 1), weights::Symbol = :obs, solvers = ("ZERO", PROTOCOL_SOLVERS...),
                     exp_indices = eachindex(ctx.exposures), arms = collect(keys(ctx.templates)),
                     stage::String = "test")
    ctx.field === :test || throw(ArgumentError("the test stage runs on the test field"))
    weights in (:obs, :oracle) || throw(ArgumentError("weights must be :obs or :oracle"))
    wlabel = String(weights)
    done = Set(row_key(r) for r in read_rows(rows_in(ctx.outdir, "rows_")))
    out = joinpath(ctx.outdir, "rows_shard$(shard[1]).tsv")
    n = length(ctx.geo.w_true)
    nruns = 0
    for (arm, i, r) in enumerate_units(ctx; arms = arms, shard = shard)
        i in exp_indices || continue
        u = draw_unit(ctx, arm, i, r)
        W = weights === :oracle ? oracle_weights(ctx.templates[arm], u.tau) : protocol_weights(u.counts, u.tau)
        if "ZERO" in solvers && weights === :obs &&
           !(join((String(ctx.field), arm, string(i), string(r), "ZERO", "-", wlabel, "0.0", "final"), '|') in done)
            z = SolveRun(["final" => zeros(n)], 0.0, 0, 0, 0, "-", Any[])
            append_rows(out, [protocol_log_row(stage, ctx.geo, ctx.op, u, W, "ZERO", "-", wlabel, 0.0, "final",
                                               zeros(n), z; selection_sha256 = sha)])
        end
        mset = selected_setting(selection, "MLEM", arm, i)
        mlem_vec = nothing
        for solver in PROTOCOL_SOLVERS
            solver in solvers || continue
            s = selected_setting(selection, solver, arm, i)
            warm = solver in ("MLEM->GN", "MLEM->GD")
            start = warm ? "mlem(c=$(mset.param),n=$(mset.label))" : solver in ("GN-lin", "GN-0", "GD-0") ? "zero" : "-"
            c = solver in ("GD-0", "MLEM->GD") ?
                selected_setting(selection, solver == "GD-0" ? "GN-0" : "MLEM->GN", arm, i).param : s.param
            labels = solver in ("SART", "MLEM") ? [s.label] :
                     solver in ("GD-0", "MLEM->GD") ? [_iter_label("K", K) for K in [5, 10, 20, :final]] :
                     [_iter_label("K", K) for K in [1, 2, 3, 5, 8, :final]]
            keyf(l) = join((String(ctx.field), arm, string(i), string(r), solver, start, wlabel, string(c), l), '|')
            all(l -> keyf(l) in done, labels) && continue
            warm && mlem_vec === nothing && (mlem_vec = _mlem_vector(ctx.op, u, mset))
            run = if solver == "SART"
                run_sart(ctx.op, u.obs, c, [s.iter])
            elseif solver == "MLEM"
                run_mlem(ctx.op, u.obs, c, [s.iter])
            elseif solver in ("GN-lin", "GN-0", "MLEM->GN")
                run_gn(ctx.op, u.obs, W, c, [1, 2, 3, 5, 8, :final], solver == "GN-lin" ? :lin : :relin,
                       warm ? mlem_vec : zeros(n))
            else
                run_gd(ctx.op, u.obs, W, c, [5, 10, 20, :final], warm ? mlem_vec : zeros(n))
            end
            nruns += 1
            rec = _save_run(ctx.outdir, stage * "_" * wlabel, u, solver, c, run)
            rows = [protocol_log_row(stage, ctx.geo, ctx.op, u, W, solver, start, wlabel, c, lbl, w, run;
                                     selected = lbl == s.label, rec = rec * "#" * lbl, selection_sha256 = sha)
                    for (lbl, w) in run.vectors if !(keyf(lbl) in done)]
            append_rows(out, rows)
            println("  $stage $arm e$i r$r $solver ", "$(round(run.secs; digits = 1))s"); flush(stdout)
        end
    end
    return nruns
end

# --------------------------------------------------------------------------
# Development driver helpers: grid state, per-phase selection, final freeze
# --------------------------------------------------------------------------

function load_grids(outdir::AbstractString, arms, n_exposures::Integer)
    path = joinpath(outdir, "grids.toml")
    isfile(path) || return default_grids(arms, n_exposures)
    d = TOML.parsefile(path)
    for (_, g) in d
        g["param"] = Float64.(g["param"]); g["iter"] = Int.(g["iter"])
    end
    return d
end

function save_grids(outdir::AbstractString, grids::Dict)
    mkpath(outdir)
    clean = Dict{String,Any}(k => Dict{String,Any}("param" => sort(Float64.(g["param"])), "iter" => sort(Int.(g["iter"])),
                                                   "ext" => g["ext"]) for (k, g) in grids)
    open(joinpath(outdir, "grids.toml"), "w") do io; TOML.print(io, clean; sorted = true); end
end

phase_selection_path(outdir, phase) = joinpath(outdir, "sel_phase$(phase).toml")

function read_phase_selection(outdir, phase)
    path = phase_selection_path(outdir, phase)
    isfile(path) || error("phase $phase selection not found at $path; run the earlier phase and its selection first")
    return TOML.parsefile(path)
end

"""
    dev_select!(outdir, phase, arms, exposures, nreps) -> Int

Selection step after a development phase: reads the rows, applies the selection, tie
and edge rules, writes `sel_phase<phase>.toml` and the (possibly extended)
`grids.toml`. Returns the number of grid extensions made; while it is positive the
phase must be run again (the missing runs are computed) and selected again.
"""
function dev_select!(outdir::AbstractString, phase::Int, arms, exposures, nreps::Int)
    grids = load_grids(outdir, arms, length(exposures))
    rows = read_rows(rows_in(outdir, "rows_phase"))
    sel, pending = select_phase!(rows, phase, grids, arms, exposures, nreps)
    save_grids(outdir, grids)
    open(phase_selection_path(outdir, phase), "w") do io; TOML.print(io, sel; sorted = true); end
    return pending
end

"""
    dev_finalize!(outdir, path, meta) -> sha256

Merge the three phase selections into the frozen `selection.toml` (with its content
hash). Refuses while a phase is missing or has pending grid extensions.
"""
function dev_finalize!(outdir::AbstractString, path::AbstractString, meta::Dict)
    merged = Dict{String,Any}()
    for phase in 1:3
        for (k, v) in read_phase_selection(outdir, phase)
            get(v, "extensions_pending", false) && error("phase $phase still has pending grid extensions ($k)")
            merged[k] = v
        end
    end
    return write_selection(path, merged, meta)
end
