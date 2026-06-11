# ===========================================================================
# Inverse solvers, image-quality metrics, and resolution estimator.
#
# Included into `module Tomography` (shares its imports/namespace).
#
# The unknown is the per-cell rock/water proportion `w ∈ [0, MAX_WATER_FRACTION]`.
# SART and MLEM operate on a (locally linearised) system `A w ≈ b`;
# `gradient_descent_reconstruct` minimises the nonlinear data misfit using the
# AD Jacobian. All solvers box-constrain `w` and return `(w, history)` where
# `history` is the per-iteration data-misfit norm `‖A w − b‖` (or `‖pred − obs‖`).
# ===========================================================================

# --------------------------------------------------------------------------
# Smoothness (Bayesian / Tikhonov) prior on a cell adjacency graph
# --------------------------------------------------------------------------

"""
    SmoothnessPrior(neighbors, weight)

Graph-Laplacian smoothness prior `R(w) = weight * Σ_i Σ_{j∈N(i)} (w_i - w_j)^2`.
`neighbors[i]` lists the cell indices adjacent to cell `i`. Acts as a Bayesian
MAP prior (MLEM one-step-late) or Tikhonov regulariser (SART/GD).
"""
struct SmoothnessPrior
    neighbors::Vector{Vector{Int}}
    weight::Float64
end

"""
    laplacian_gradient(prior, w) -> Vector

Gradient `∂R/∂w` of the smoothness prior: `g_k = 2*weight*Σ_{j∈N(k)} (w_k - w_j)`.
"""
function laplacian_gradient(prior::SmoothnessPrior, w::AbstractVector{<:Real})
    g = zeros(Float64, length(w))
    λ = prior.weight
    @inbounds for k in eachindex(w)
        acc = 0.0
        for j in prior.neighbors[k]
            acc += (w[k] - w[j])
        end
        g[k] = 2.0 * λ * acc
    end
    return g
end

"""
    apply_laplacian_smoothing(prior, w, strength) -> Vector

One explicit smoothing sweep `w_k ← (1-strength) w_k + strength * mean(neighbors)`.
"""
function apply_laplacian_smoothing(prior::SmoothnessPrior, w::AbstractVector{<:Real}, strength::Float64)
    out = copy(w)
    @inbounds for k in eachindex(w)
        nb = prior.neighbors[k]
        isempty(nb) && continue
        m = 0.0
        for j in nb
            m += w[j]
        end
        m /= length(nb)
        out[k] = (1.0 - strength) * w[k] + strength * m
    end
    return out
end

# --------------------------------------------------------------------------
# SART — Simultaneous Algebraic Reconstruction Technique (absorption)
# --------------------------------------------------------------------------

"""
    sart_reconstruct(A, b; w0, n_iter=50, relaxation=0.2, box=(0,0.9),
                     prior=nothing, smooth_strength=0.1) -> (w, history)

Solve `A w ≈ b` with the SART update
`w ← clamp(w + relaxation · D_col · Aᵀ · D_row · (b − A w), box)`,
where `D_row = 1/Σ_j|A_ij|` and `D_col = 1/Σ_i|A_ij|`. With `prior`, a Laplacian
smoothing sweep is applied each iteration (Tikhonov-style regularisation).
"""
function sart_reconstruct(A::AbstractMatrix, b::AbstractVector;
                          w0::Union{Nothing,AbstractVector} = nothing,
                          n_iter::Int = 50, relaxation::Float64 = 0.2,
                          box::Tuple{Float64,Float64} = (0.0, MAX_WATER_FRACTION),
                          prior::Union{Nothing,SmoothnessPrior} = nothing,
                          smooth_strength::Float64 = 0.1)
    n = size(A, 2)
    w = w0 === nothing ? fill(0.0, n) : collect(Float64, w0)
    Aabs = abs.(A)
    rowsum = vec(sum(Aabs, dims = 2))
    colsum = vec(sum(Aabs, dims = 1))
    @inbounds for i in eachindex(rowsum); rowsum[i] == 0 && (rowsum[i] = 1.0); end
    @inbounds for j in eachindex(colsum); colsum[j] == 0 && (colsum[j] = 1.0); end
    history = Float64[]
    for _ in 1:n_iter
        r = b .- A * w
        upd = (transpose(A) * (r ./ rowsum)) ./ colsum
        w = clamp.(w .+ relaxation .* upd, box[1], box[2])
        if prior !== nothing
            w = clamp.(apply_laplacian_smoothing(prior, w, smooth_strength), box[1], box[2])
        end
        push!(history, norm(b .- A * w))
    end
    return w, history
end

# --------------------------------------------------------------------------
# MLEM / Bayesian-MAP — multiplicative EM (nonnegative system)
# --------------------------------------------------------------------------

"""
    mlem_reconstruct(A, b; w0, n_iter=100, box=(0,0.9), prior=nothing, eps=1e-12)
        -> (w, history)

Maximum-likelihood (Poisson) EM for a **nonnegative** system `A ≥ 0, b ≥ 0`:
`w ← w .* (Aᵀ (b ./ (A w))) ./ (Aᵀ 1 + λ ∂R/∂w)` (one-step-late MAP when `prior`
is given), then clamp to `box`. For absorption muography pass the transmission
**deficit** (`b = baseline − flux`, `A = −J`) so both are nonnegative.
"""
function mlem_reconstruct(A::AbstractMatrix, b::AbstractVector;
                          w0::Union{Nothing,AbstractVector} = nothing,
                          n_iter::Int = 100,
                          box::Tuple{Float64,Float64} = (0.0, MAX_WATER_FRACTION),
                          prior::Union{Nothing,SmoothnessPrior} = nothing,
                          eps::Float64 = 1e-12)
    n = size(A, 2)
    w = w0 === nothing ? fill(max(box[1], 1e-3), n) : collect(Float64, w0)
    # ensure strictly positive start (multiplicative updates cannot leave 0)
    @inbounds for i in eachindex(w); w[i] = max(w[i], 1e-6); end
    sens = vec(sum(A, dims = 1))
    @inbounds for j in eachindex(sens); sens[j] <= 0 && (sens[j] = 1.0); end
    history = Float64[]
    for _ in 1:n_iter
        Aw = A * w
        @inbounds for i in eachindex(Aw); Aw[i] = max(Aw[i], eps); end
        ratio = transpose(A) * (b ./ Aw)
        denom = copy(sens)
        if prior !== nothing
            denom = denom .+ max.(laplacian_gradient(prior, w), 0.0)
        end
        w = clamp.(w .* ratio ./ denom, box[1], box[2])
        push!(history, norm(b .- A * w))
    end
    return w, history
end

# --------------------------------------------------------------------------
# Edge-preserving (Huber / TV) prior on the cell adjacency graph
# --------------------------------------------------------------------------

"""
    EdgePrior(neighbors, weight, delta)

Edge-preserving prior `R(w) = weight · Σ_{i<j∈N} ρ_δ(w_i − w_j)` with the Huber
loss `ρ_δ` (quadratic for `|t|≤δ`, linear beyond). Unlike the quadratic
[`SmoothnessPrior`], it penalises a sharp boundary only linearly, so it
preserves blocky anomalies (e.g. an aquifer box) instead of blurring them.
`delta → 0` approaches total variation. Provides the gradient `∂R/∂w` and an
SPD lagged-diffusivity (IRLS) Hessian `weight·Lᵂ` for Gauss-Newton.
"""
struct EdgePrior
    neighbors::Vector{Vector{Int}}
    weight::Float64
    delta::Float64
end

@inline _huber_grad(t, δ) = abs(t) <= δ ? t : δ * sign(t)
@inline _huber_irls(t, δ) = abs(t) <= δ ? 1.0 : δ / abs(t)   # ψ(t) = ρ'(t)/t

"""
    edge_gradient(prior, w) -> Vector
"""
function edge_gradient(prior::EdgePrior, w::AbstractVector{<:Real})
    g = zeros(Float64, length(w))
    λ = prior.weight; δ = prior.delta
    @inbounds for k in eachindex(w)
        acc = 0.0
        for j in prior.neighbors[k]
            acc += _huber_grad(w[k] - w[j], δ)
        end
        g[k] = λ * acc
    end
    return g
end

"""
    edge_hessian(prior, w) -> SparseMatrixCSC

Lagged-diffusivity (IRLS) Gauss-Newton Hessian `weight · Lᵂ`, a weighted graph
Laplacian with edge weights `ψ(w_i−w_j) = ρ'_δ/(w_i−w_j)`. SPD, so the
normal-equation matrix stays positive definite.
"""
function edge_hessian(prior::EdgePrior, w::AbstractVector{<:Real})
    n = length(w); λ = prior.weight; δ = prior.delta
    I = Int[]; J = Int[]; V = Float64[]
    @inbounds for k in 1:n
        diag = 0.0
        for j in prior.neighbors[k]
            ψ = _huber_irls(w[k] - w[j], δ)
            diag += ψ
            push!(I, k); push!(J, j); push!(V, -λ * ψ)
        end
        push!(I, k); push!(J, k); push!(V, λ * diag)
    end
    return sparse(I, J, V, n, n)
end

# --------------------------------------------------------------------------
# Gradient descent on the nonlinear misfit using the AD Jacobian (our method)
# --------------------------------------------------------------------------

"""
    gradient_descent_reconstruct(model, obs; w0, n_iter=200, lr=1e-2,
                                 box=(0,0.9), prior=nothing, optimizer=:adam,
                                 relinearize=true, weights=nothing) -> (w, history)

Minimise `½‖pred(w) − obs‖²_W + R(w)` by projected gradient descent. `model(w)`
returns `(pred::Vector, J)` with `J = ∂pred/∂w` (the AD sparse Jacobian); the
gradient is `Jᵀ(W ⊙ (pred − obs)) + ∂R/∂w`. With `relinearize=false` the Jacobian
is computed once at `w0` and reused (cheaper, linearised solve). `optimizer` is
`:pgd` (recommended), `:adam`, or `:gd`. `:pgd` is a diagonally-preconditioned
projected gradient: it scales the gradient by the Gauss-Newton + prior curvature
diagonal and takes the exact quadratic-model step length, so uninformative cells
(≈0 gradient AND ≈0 curvature) get ≈0 step and the background stays sparse — where
`:adam`, being per-coordinate scale-free, marches every cell to the box and
destroys the reconstruction. For `:pgd`, `lr` is the step-length cap (use `1.0`).
`prior` may be a quadratic [`SmoothnessPrior`] or an
edge-preserving [`EdgePrior`] (same regulariser as Gauss-Newton), and `weights`
is an optional per-bin `1/σ²` for inverse-variance parity with `gauss_newton_reconstruct`.
This is a DiffPumas-native solver: the same differentiable CSDA forward model used
for sensitivities drives the inversion (with `relinearize=true`).
"""
function gradient_descent_reconstruct(model, obs::AbstractVector;
                                      w0::AbstractVector,
                                      n_iter::Int = 200, lr::Float64 = 1e-2,
                                      box::Tuple{Float64,Float64} = (0.0, MAX_WATER_FRACTION),
                                      prior::Union{Nothing,SmoothnessPrior,EdgePrior} = nothing,
                                      optimizer::Symbol = :adam,
                                      relinearize::Bool = true,
                                      weights::Union{Nothing,AbstractVector} = nothing)
    w = collect(Float64, w0)
    history = Float64[]
    W = weights === nothing ? nothing : collect(Float64, weights)
    prior_grad(wv) = prior === nothing ? nothing :
        (prior isa EdgePrior ? edge_gradient(prior, wv) : laplacian_gradient(prior, wv))
    # Adam state
    m = zeros(Float64, length(w))
    v = zeros(Float64, length(w))
    β1 = 0.9; β2 = 0.999; ϵ = 1e-8
    Jfixed = nothing
    for t in 1:n_iter
        local pred, J
        if relinearize
            pred, J = model(w)
        elseif Jfixed === nothing
            pred, Jfixed = model(w)
            J = Jfixed
        else
            pred, _ = model(w)
            J = Jfixed
        end
        r = pred .- obs
        g = W === nothing ? transpose(J) * r : transpose(J) * (W .* r)
        if prior !== nothing
            g = g .+ prior_grad(w)
        end
        if optimizer === :pgd
            # Jacobi (diagonal Gauss-Newton + prior) preconditioner, then the exact
            # step length that minimises the local quadratic model along p = D⁻¹g.
            dgJ = W === nothing ? vec(sum(J .^ 2; dims = 1)) : vec(sum(W .* (J .^ 2); dims = 1))
            Hr = prior === nothing ? nothing :
                 (prior isa EdgePrior ? edge_hessian(prior, w) :
                  edge_hessian(EdgePrior(prior.neighbors, prior.weight, Inf), w))
            dvec = collect(Float64, dgJ)
            if Hr !== nothing
                @inbounds for i in eachindex(dvec); dvec[i] += Hr[i, i]; end
            end
            dmax = maximum(dvec); dmax = dmax > 0 ? dmax : 1.0
            p = g ./ max.(dvec, 1e-9 * dmax)                 # preconditioned descent dir
            Jp = J * p
            denom = W === nothing ? dot(Jp, Jp) : dot(Jp, W .* Jp)
            Hr === nothing || (denom += dot(p, Hr * p))      # curvature along p
            num = dot(g, p)
            α = (denom > 0 && isfinite(num)) ? clamp(num / denom, 0.0, lr) : 0.0
            @inbounds for i in eachindex(w)
                w[i] = clamp(w[i] - α * p[i], box[1], box[2])
            end
        elseif optimizer === :adam
            @inbounds for i in eachindex(w)
                m[i] = β1 * m[i] + (1 - β1) * g[i]
                v[i] = β2 * v[i] + (1 - β2) * g[i]^2
                mhat = m[i] / (1 - β1^t)
                vhat = v[i] / (1 - β2^t)
                w[i] = clamp(w[i] - lr * mhat / (sqrt(vhat) + ϵ), box[1], box[2])
            end
        else
            @inbounds for i in eachindex(w)
                w[i] = clamp(w[i] - lr * g[i], box[1], box[2])
            end
        end
        push!(history, norm(r))
        # converged (preconditioned mode takes near-optimal steps, so the misfit
        # plateaus quickly): stop early to avoid wasted relinearisations.
        if optimizer === :pgd && length(history) > 1 &&
           abs(history[end - 1] - history[end]) <= 1e-5 * max(history[end - 1], eps())
            break
        end
    end
    return w, history
end

# --------------------------------------------------------------------------
# Gauss-Newton / Levenberg-Marquardt on the AD Jacobian (our method)
# --------------------------------------------------------------------------

"""
    gauss_newton_reconstruct(model, obs; w0, n_iter=12, box=(0,0.9),
                             prior=nothing, lm_damping=1e-3, lm_factor=3.0,
                             relinearize=true, weights=nothing) -> (w, history)

Box-constrained Gauss-Newton (Levenberg-Marquardt) for the nonlinear data misfit
`½‖pred(w) − obs‖²_W + R(w)`. `model(w) → (pred, J)` supplies the forward flux and
its sparse AD Jacobian; each step solves the regularised normal equations
`(JᵀWJ + H_R + μ·diag(JᵀWJ)) δ = −(JᵀW r + ∇R)` and projects `w+δ` onto `box`,
adapting the LM damping `μ` on accept/reject. With `relinearize=true` the Jacobian
is recomputed each iteration (cheap with the fast assembly), so this fits the true
nonlinear corrected-CSDA operator — the DiffPumas-native solver, and the one meant
to beat MLEM on full-MC data. `weights` is an optional per-bin `1/σ²`.
"""
function gauss_newton_reconstruct(model, obs::AbstractVector;
                                  w0::AbstractVector,
                                  n_iter::Int = 12,
                                  box::Tuple{Float64,Float64} = (0.0, MAX_WATER_FRACTION),
                                  prior::Union{Nothing,SmoothnessPrior,EdgePrior} = nothing,
                                  lm_damping::Float64 = 1e-3,
                                  lm_factor::Float64 = 3.0,
                                  relinearize::Bool = true,
                                  weights::Union{Nothing,AbstractVector} = nothing)
    w = clamp.(collect(Float64, w0), box[1], box[2])
    history = Float64[]
    W = weights === nothing ? nothing : collect(Float64, weights)

    prior_grad(wv) = prior === nothing ? nothing :
        (prior isa EdgePrior ? edge_gradient(prior, wv) : laplacian_gradient(prior, wv))
    prior_hess(wv) = prior === nothing ? nothing :
        (prior isa EdgePrior ? edge_hessian(prior, wv) :
         # quadratic prior: constant weighted Laplacian (weights = 1)
         edge_hessian(EdgePrior(prior.neighbors, prior.weight, Inf), wv))

    cost(r) = W === nothing ? 0.5 * dot(r, r) : 0.5 * sum(W .* r .^ 2)

    pred, J = model(w)
    r = pred .- obs
    f_cur = cost(r)
    μ = lm_damping
    push!(history, norm(r))

    for _ in 1:n_iter
        Jt = transpose(J)
        # Normal-equation matrix and RHS (optionally weighted)
        if W === nothing
            JtJ = Matrix(Jt * J)
            grad = Jt * r
        else
            WJ = (W .* J)
            JtJ = Matrix(Jt * WJ)
            grad = Jt * (W .* r)
        end
        if prior !== nothing
            grad = grad .+ prior_grad(w)
            JtJ = JtJ .+ Matrix(prior_hess(w))
        end
        d = [JtJ[i, i] for i in axes(JtJ, 1)]
        dmax = maximum(d); dmax = dmax > 0 ? dmax : 1.0

        # For a strongly nonlinear operator the full Gauss-Newton step overshoots,
        # so we damp (LM) AND backtrack along the step direction: try the LM step,
        # then halve it until the true (nonlinear) cost actually decreases. A few
        # LM levels guard against a poorly-scaled direction.
        accepted = false
        for _try in 1:6
            A = JtJ + Diagonal(μ .* max.(d, 1e-12 * dmax))
            δ = try
                -(A \ grad)
            catch
                fill(0.0, length(w))
            end
            if !all(isfinite, δ)
                μ = min(μ * lm_factor, 1e12); continue
            end
            α = 1.0
            for _ls in 1:14
                w_new = clamp.(w .+ α .* δ, box[1], box[2])
                pred_new, J_new = relinearize ? model(w_new) : (J * w_new, J)
                f_new = cost(pred_new .- obs)
                if isfinite(f_new) && f_new < f_cur - 1e-10 * abs(f_cur)
                    w = w_new; r = pred_new .- obs; J = J_new; f_cur = f_new
                    accepted = true
                    break
                end
                α *= 0.5
            end
            if accepted
                μ = max(μ / lm_factor, 1e-12)
                break
            else
                μ = min(μ * lm_factor, 1e12)
            end
        end
        push!(history, norm(r))
        accepted || break
    end
    return w, history
end

"""
    make_csda_operator(physics, shallow_flags, matcfg, site, paths, energy_samples;
                       n_cells, valid_bins, threaded=true) -> model

Build a `model(w) -> (pred, J)` closure over the CSDA geometry, restricted to
`valid_bins` (rows). Suitable as the `model` argument of
`gradient_descent_reconstruct`. `pred` is the forward flux on the valid bins and
`J` the corresponding sparse Jacobian rows.
"""
function make_csda_operator(physics, shallow_flags::AbstractVector{Bool},
                            matcfg::MaterialConfig, site::SiteConfig,
                            paths::Vector{DirectionalPath},
                            energy_samples::Vector{EnergySample};
                            n_cells::Int, valid_bins::AbstractVector{Int},
                            threaded::Bool = true)
    sub_paths = paths[valid_bins]
    return function (w)
        flux, J = assemble_forward_and_jacobian(physics, shallow_flags, matcfg, site,
            sub_paths, w, energy_samples; n_cells = n_cells, threaded = threaded)
        return flux, J
    end
end

# ===========================================================================
# Image-quality metrics (docs/muon_tomography_overview.md §4)
# ===========================================================================

_select(x, mask) = mask === nothing ? x : x[mask]

"""
    mse(recon, truth; mask=nothing) -> Float64
"""
function mse(recon::AbstractVector, truth::AbstractVector; mask = nothing)
    a = _select(recon, mask); b = _select(truth, mask)
    return mean((a .- b) .^ 2)
end

rmse(recon, truth; mask = nothing) = sqrt(mse(recon, truth; mask = mask))

"""
    snr_metric(recon; mask=nothing) -> Float64

Signal-to-noise ratio in dB, `10·log10(mean² / var)` over the (masked) field.
"""
function snr_metric(recon::AbstractVector; mask = nothing)
    a = _select(recon, mask)
    μ = mean(a); σ2 = var(a)
    σ2 <= 0 && return Inf
    return 10.0 * log10(μ^2 / σ2)
end

"""
    recon_snr(recon, truth; mask=nothing) -> Float64

Truth-referenced reconstruction SNR in dB, `10·log10(‖truth‖² / ‖truth − recon‖²)`
over the (masked) field. Unlike [`snr_metric`] (which only measures a field's own
mean²/var and rewards flat/diverged solutions), this rises only as the
reconstruction approaches the ground truth, so it ranks solvers correctly.
"""
function recon_snr(recon::AbstractVector, truth::AbstractVector; mask = nothing)
    a = _select(recon, mask); b = _select(truth, mask)
    sig = dot(b, b)
    err = sum((a .- b) .^ 2)
    err <= 0 && return Inf
    sig <= 0 && return -Inf
    return 10.0 * log10(sig / err)
end

"""
    cnr(field, roi_mask, bg_mask) -> Float64

Contrast-to-noise ratio `|μ_roi − μ_bg| / sqrt(σ_roi² + σ_bg²)`.
"""
function cnr(field::AbstractVector, roi_mask::AbstractVector{Bool}, bg_mask::AbstractVector{Bool})
    roi = field[roi_mask]; bg = field[bg_mask]
    (isempty(roi) || isempty(bg)) && return NaN
    σ2 = var(roi) + var(bg)
    denom = sqrt(σ2)
    denom <= 0 && return Inf
    return abs(mean(roi) - mean(bg)) / denom
end

"""
    psnr(recon, truth; peak=MAX_WATER_FRACTION, mask=nothing) -> Float64

Peak signal-to-noise ratio in dB, `10·log10(peak² / MSE)`.
"""
function psnr(recon::AbstractVector, truth::AbstractVector;
              peak::Float64 = MAX_WATER_FRACTION, mask = nothing)
    m = mse(recon, truth; mask = mask)
    m <= 0 && return Inf
    return 10.0 * log10(peak^2 / m)
end

"""
    ssim_metric(recon, truth; peak=MAX_WATER_FRACTION, mask=nothing) -> Float64

Global (single-window) structural similarity over the (masked) cell values —
appropriate for unstructured meshes where there is no regular pixel grid.
"""
function ssim_metric(recon::AbstractVector, truth::AbstractVector;
                     peak::Float64 = MAX_WATER_FRACTION, mask = nothing)
    a = _select(recon, mask); b = _select(truth, mask)
    μx = mean(a); μy = mean(b)
    σx2 = var(a); σy2 = var(b)
    σxy = length(a) > 1 ? cov(a, b) : 0.0
    c1 = (0.01 * peak)^2
    c2 = (0.03 * peak)^2
    return ((2μx * μy + c1) * (2σxy + c2)) / ((μx^2 + μy^2 + c1) * (σx2 + σy2 + c2))
end

"""
    reconstruction_report(recon, truth; mask=nothing, roi_mask=nothing, bg_mask=nothing,
                          peak=MAX_WATER_FRACTION) -> NamedTuple

Bundle of all comparison metrics between a reconstructed and a ground-truth field.
`mask` restricts mse/rmse/psnr/ssim/snr to a subset of cells (e.g. a depth band);
the reported `snr` is the truth-referenced [`recon_snr`], not the field-only
[`snr_metric`].
"""
function reconstruction_report(recon::AbstractVector, truth::AbstractVector;
                               mask = nothing, roi_mask = nothing, bg_mask = nothing,
                               peak::Float64 = MAX_WATER_FRACTION)
    c = (roi_mask !== nothing && bg_mask !== nothing) ? cnr(recon, roi_mask, bg_mask) : NaN
    return (mse = mse(recon, truth; mask = mask), rmse = rmse(recon, truth; mask = mask),
            psnr = psnr(recon, truth; peak = peak, mask = mask),
            ssim = ssim_metric(recon, truth; peak = peak, mask = mask),
            snr = recon_snr(recon, truth; mask = mask), cnr = c)
end

# ===========================================================================
# Resolution estimator (point-spread / FWHM vs depth)
# ===========================================================================

"""
    point_spread_recovery(reconstruct, centroids, baseline, anomaly_cell;
                          contrast=0.5) -> (psf, fwhm_m, peak_cell, center)

Inject a single-cell water-fraction anomaly of size `contrast` at `anomaly_cell`,
forward-simulate noiseless observations, reconstruct, and characterise the
recovered point-spread function `psf = w_rec − baseline`.

`reconstruct(w_true) -> w_rec` is a caller-supplied closure that forward-simulates
observations from `w_true` and runs a solver. `centroids` is `3 × n_cells`
(metres). Returns the PSF, its FWHM in metres (from a radial half-maximum about
the anomaly centroid), the peak cell, and the anomaly centroid.
"""
function point_spread_recovery(reconstruct, centroids::AbstractMatrix{<:Real},
                               baseline::AbstractVector{<:Real}, anomaly_cell::Int;
                               contrast::Float64 = 0.5)
    w_true = collect(Float64, baseline)
    w_true[anomaly_cell] = clamp(baseline[anomaly_cell] + contrast, 0.0, MAX_WATER_FRACTION)
    w_rec = reconstruct(w_true)
    psf = w_rec .- baseline
    cx = centroids[1, anomaly_cell]; cy = centroids[2, anomaly_cell]; cz = centroids[3, anomaly_cell]
    peak_cell = argmax(abs.(psf))
    peak_val = psf[peak_cell]
    fwhm_m = radial_fwhm(psf, centroids, (cx, cy, cz), peak_val)
    return (psf = psf, fwhm_m = fwhm_m, peak_cell = peak_cell, center = (cx, cy, cz))
end

"""
    radial_fwhm(psf, centroids, center, peak_val) -> Float64

Estimate full-width-at-half-maximum (metres) of a recovered anomaly: the largest
radius from `center` at which the PSF (sign of `peak_val`) still exceeds half its
peak, doubled. Returns `NaN` if the peak is degenerate.
"""
function radial_fwhm(psf::AbstractVector{<:Real}, centroids::AbstractMatrix{<:Real},
                     center::NTuple{3,<:Real}, peak_val::Real)
    abs(peak_val) <= 0 && return NaN
    half = 0.5 * peak_val
    s = sign(peak_val)
    r_half = 0.0
    @inbounds for i in eachindex(psf)
        # cell still "in" the anomaly if it has the same sign and exceeds half-max
        if s * psf[i] >= s * half && s * psf[i] > 0
            dx = centroids[1, i] - center[1]
            dy = centroids[2, i] - center[2]
            dz = centroids[3, i] - center[3]
            r = sqrt(dx^2 + dy^2 + dz^2)
            r > r_half && (r_half = r)
        end
    end
    return 2.0 * r_half
end

"""
    resolution_vs_depth(reconstruct, centroids, baseline, candidate_cells;
                        depth_edges, contrast=0.5, zenith_step_deg=1.0) -> Vector{NamedTuple}

Estimate recovered resolution as a function of depth (height above detector).
For each cell in `candidate_cells`, run `point_spread_recovery` and bin the FWHM
by the cell's `z` centroid into the intervals defined by `depth_edges`. Each
bin reports the median empirical FWHM and a geometric floor
`max(local_cell_size, depth · Δθ)` with `Δθ = deg2rad(zenith_step_deg)`, so the
number is interpretable against the angular-sampling limit.
"""
function resolution_vs_depth(reconstruct, centroids::AbstractMatrix{<:Real},
                             baseline::AbstractVector{<:Real},
                             candidate_cells::AbstractVector{Int};
                             depth_edges::AbstractVector{<:Real},
                             contrast::Float64 = 0.5,
                             zenith_step_deg::Float64 = 1.0,
                             cell_size_m::Float64 = 0.0)
    Δθ = deg2rad(zenith_step_deg)
    nb = length(depth_edges) - 1
    fwhms = [Float64[] for _ in 1:nb]
    depths = [Float64[] for _ in 1:nb]
    for c in candidate_cells
        z = centroids[3, c]
        bin = searchsortedlast(depth_edges, z)
        (bin < 1 || bin > nb) && continue
        res = point_spread_recovery(reconstruct, centroids, baseline, c; contrast = contrast)
        isfinite(res.fwhm_m) || continue
        push!(fwhms[bin], res.fwhm_m)
        push!(depths[bin], z)
    end
    out = NamedTuple[]
    for bin in 1:nb
        isempty(fwhms[bin]) && continue
        d = median(depths[bin])
        emp = median(fwhms[bin])
        floor_m = max(cell_size_m, d * Δθ)
        push!(out, (depth_m = d, fwhm_m = emp, floor_m = floor_m, n_cells = length(fwhms[bin])))
    end
    return out
end

"""
    resolution_map(reconstruct, centroids, baseline, cells;
                   depth_edges, azimuth_edges, contrast=0.5,
                   zenith_step_deg=1.0, cell_size_m=0.0)
        -> (depths, azimuths, fwhm, counts, floor)

Recovered FWHM resolution over a full **depth × direction** grid. For every cell in
`cells` run [`point_spread_recovery`], then bin its FWHM by depth (`z` centroid,
via `depth_edges`) and by azimuthal direction about the detector axis
(`atand(y, x) ∈ [0,360)`, via `azimuth_edges`). Returns the per-bin median FWHM in
`fwhm[ndepth, nazim]` (NaN where empty), the cell `counts`, the depth-bin midpoints
`depths`, azimuth-bin midpoints `azimuths`, and the per-depth geometric floor
`max(cell_size_m, depth · Δθ)`. Pass all cells (above a minimum depth) to cover
all directions and all depths.
"""
function resolution_map(reconstruct, centroids::AbstractMatrix{<:Real},
                        baseline::AbstractVector{<:Real},
                        cells::AbstractVector{Int};
                        depth_edges::AbstractVector{<:Real},
                        azimuth_edges::AbstractVector{<:Real},
                        contrast::Float64 = 0.5,
                        zenith_step_deg::Float64 = 1.0,
                        cell_size_m::Float64 = 0.0)
    Δθ = deg2rad(zenith_step_deg)
    nd = length(depth_edges) - 1
    na = length(azimuth_edges) - 1
    acc = [Float64[] for _ in 1:nd, _ in 1:na]
    for c in cells
        z = centroids[3, c]
        db = searchsortedlast(depth_edges, z)
        (db < 1 || db > nd) && continue
        az = atand(centroids[2, c], centroids[1, c]); az < 0 && (az += 360.0)
        ab = searchsortedlast(azimuth_edges, az)
        (ab < 1 || ab > na) && continue
        res = point_spread_recovery(reconstruct, centroids, baseline, c; contrast = contrast)
        isfinite(res.fwhm_m) || continue
        push!(acc[db, ab], res.fwhm_m)
    end
    fwhm = fill(NaN, nd, na); counts = zeros(Int, nd, na)
    for db in 1:nd, ab in 1:na
        isempty(acc[db, ab]) && continue
        fwhm[db, ab] = median(acc[db, ab]); counts[db, ab] = length(acc[db, ab])
    end
    depths = [0.5 * (depth_edges[k] + depth_edges[k + 1]) for k in 1:nd]
    azimuths = [0.5 * (azimuth_edges[k] + azimuth_edges[k + 1]) for k in 1:na]
    floors = [max(cell_size_m, depths[k] * Δθ) for k in 1:nd]
    return (depths = depths, azimuths = azimuths, fwhm = fwhm, counts = counts, floor = floors)
end

# ===========================================================================
# Rigorous inverse-problem statistics: Laplace posterior, null-model tests,
# and nuisance-parameter (absolute-level) profiling.
#
# These turn a point reconstruction into an uncertainty-aware result. The key
# physical separation, made explicit below, is:
#   * the Laplace posterior gives the CONDITIONAL (given the global normalisation
#     s) per-cell SHAPE uncertainty — what the angular data constrain;
#   * the inventory profile gives the ABSOLUTE-LEVEL uncertainty — the direction
#     that is degenerate with s and is therefore left to a 1-D profile likelihood.
# ===========================================================================

"""
    laplace_posterior(model, w, weights; prior, free) -> NamedTuple

Gaussian-linear (Laplace) posterior at the MAP field `w`. Rebuilds the AD
Jacobian `J = ∂pred/∂w` at `w`, forms the posterior precision
`Λ = JᵀWJ + H_R` (the Gauss–Newton / penalised-fit Hessian, with `H_R` the
quadratic smoothness-prior Laplacian), and inverts it on the free index set
`free` (the cells that may carry signal — e.g. the aquifer cells).

This is the CONDITIONAL posterior given the global normalisation `s`: it
quantifies the per-cell shape that the angular data constrain. The absolute
level (degenerate with `s`) is handled by [`profile_inventory`](@ref).

Returns `(sigma, z, p_eff, resolution, JtWJ, HR)` where `sigma` (per-cell
posterior std), `z = w/σ`, and `resolution` (averaging-kernel diagonal,
`R = ΣΛ_data`) are length-`N` and zero outside `free`; `p_eff = tr(R)` is the
effective number of resolved parameters.
"""
function laplace_posterior(model, w::AbstractVector;
                           weights::AbstractVector,
                           prior::Union{Nothing,SmoothnessPrior,EdgePrior} = nothing,
                           free::AbstractVector{Bool},
                           ridge::Float64 = 1e-9)
    _, J = model(w)
    W = collect(Float64, weights)
    WJ = W .* J
    JtWJ = Matrix(transpose(J) * WJ)                 # data Fisher information (N×N)
    HR = prior === nothing ? zeros(size(JtWJ)) :
         (prior isa EdgePrior ? Matrix(edge_hessian(prior, w)) :
          Matrix(edge_hessian(EdgePrior(prior.neighbors, prior.weight, Inf), w)))
    Λ = JtWJ .+ HR
    n = length(w)
    idx = findall(free)
    sigma = zeros(n); zsc = zeros(n); res = zeros(n); p_eff = 0.0
    if !isempty(idx)
        Λff = Λ[idx, idx]
        rg = ridge * (tr(Λff) / length(idx) + 1.0)
        Σff = inv(Symmetric(Λff + rg * I))            # ≤ |free|³, trivial here
        dvar = diag(Σff)
        Rff = Σff * JtWJ[idx, idx]                    # averaging kernel (resolution)
        p_eff = tr(Rff)
        for (a, i) in enumerate(idx)
            sigma[i] = sqrt(max(dvar[a], 0.0))
            zsc[i] = sigma[i] > 0 ? w[i] / sigma[i] : 0.0
            res[i] = Rff[a, a]
        end
    end
    return (sigma = sigma, z = zsc, p_eff = p_eff, resolution = res, JtWJ = JtWJ, HR = HR)
end

# χ²(field) under statistical (Poisson × DEM, fit-mask) weights, with the global
# log-scale s refit to the field so the comparison is over shape only.
function _weighted_chi2(flux_only, w, meas, fit_mask, wdem, exposure)
    pred = flux_only(w)
    mw = sum(fit_mask)
    s = exp(sum(fit_mask .* (log.(max.(meas, 1e-300)) .- log.(max.(pred, 1e-300)))) / max(mw, 1e-30))
    obs = meas ./ s
    σ2 = max.(obs, 0.0) ./ exposure
    Wv = fit_mask .* wdem ./ max.(σ2, 1e-300)
    return sum(Wv .* (pred .- obs) .^ 2), s
end

# Wilson–Hilferty: number of Gaussian σ for an observed χ² value with k DOF
# (no external dependency). For k≤0 returns 0.
function _chi2_sigma(chi2::Float64, k::Float64)
    k <= 0 && return 0.0
    t = (chi2 / k)^(1.0 / 3.0)
    m = 1.0 - 2.0 / (9.0 * k)
    sd = sqrt(2.0 / (9.0 * k))
    return (t - m) / sd
end

"""
    null_model_tests(flux_only, w_map, w0_uniform, free, meas, fit_mask, wdem,
                     exposure; p_eff) -> NamedTuple

Likelihood-ratio comparison of the structured reconstruction against two nulls:
`N0` dry standard rock (`w≡0`) and `N1` a uniform band water-mix (`w≡w0` on
`free`). Reports weighted χ² for each, `Δχ²` against the effective added DOF
(`p_eff` for dry→fit, `p_eff−1` for uniform→fit), a Wilson–Hilferty Gaussian
significance, and AIC/BIC (with effective parameter counts) so the extra water
cells are penalised, not merely rewarded.

The published LVD intensities carry no absolute count uncertainty, so `exposure`
only sets an arbitrary overall χ² scale. We therefore CALIBRATE to a reduced χ²
of unity: the variance scale `φ = χ²_fit/(ndata−p_eff)` is estimated from the
fit residuals and the significances use `Δχ²/φ` (an F-test), making them
independent of the nominal exposure. `φ` is returned as `var_scale`.
"""
function null_model_tests(flux_only, w_map::AbstractVector, w0_uniform::Float64,
                          free::AbstractVector{Bool}, meas, fit_mask, wdem, exposure;
                          p_eff::Float64)
    w_dry  = zeros(length(w_map))
    w_unif = [free[i] ? w0_uniform : 0.0 for i in eachindex(w_map)]
    c_dry,  _ = _weighted_chi2(flux_only, w_dry,  meas, fit_mask, wdem, exposure)
    c_unif, _ = _weighted_chi2(flux_only, w_unif, meas, fit_mask, wdem, exposure)
    c_fit,  _ = _weighted_chi2(flux_only, w_map,  meas, fit_mask, wdem, exposure)
    ndata = count(>(0.0), fit_mask)
    # reduced-χ²=1 calibration: estimate the per-bin variance scale from the fit
    var_scale = max(c_fit / max(ndata - p_eff, 1.0), 1e-30)
    dchi_dry  = c_dry  - c_fit
    dchi_unif = c_unif - c_fit
    dp_dry  = max(p_eff, 1e-6)
    dp_unif = max(p_eff - 1.0, 1e-6)
    # AIC/BIC on the calibrated (reduced) deviance so the penalty is on the same scale
    aic(c, p) = c / var_scale + 2p
    bic(c, p) = c / var_scale + p * log(max(ndata, 1))
    return (chi2_dry = c_dry, chi2_unif = c_unif, chi2_fit = c_fit, ndata = ndata,
            var_scale = var_scale,
            dchi2_dry = dchi_dry, dchi2_unif = dchi_unif, p_eff = p_eff,
            sigma_dry = _chi2_sigma(max(dchi_dry, 0.0) / var_scale, dp_dry),
            sigma_unif = _chi2_sigma(max(dchi_unif, 0.0) / var_scale, dp_unif),
            aic = (dry = aic(c_dry, 0.0), unif = aic(c_unif, 1.0), fit = aic(c_fit, p_eff)),
            bic = (dry = bic(c_dry, 0.0), unif = bic(c_unif, 1.0), fit = bic(c_fit, p_eff)))
end

"""
    profile_inventory(flux_only, w_map, free, meas, fit_mask, wdem, exposure;
                      var_scale=1.0, n_grid=21, span=(0.2,3.0), w_max=MAX_WATER_FRACTION)
        -> NamedTuple

Profile likelihood for the water amplitude: the reconstructed shape `w_map` is
rescaled to a grid of overall levels (clamped to the box) and the global
normalisation `s` is refit at each, tracing `χ²(T)` for the inventory
`T = Σ_free w` (equivalently the mean band fraction `w̄ = T/|free|`).

This profiles the AMPLITUDE along the reconstructed shape — it constrains the
overall water level given that shape and the priors; it is not a full
shape-marginalised profile (the residual shape freedom is carried by the per-cell
[`laplace_posterior`](@ref)). `Δχ²` is calibrated by `var_scale` (the reduced-χ²
variance scale `φ` from [`null_model_tests`](@ref)) so the confidence thresholds
(1 → 68%, 3.84 → 95%) are exposure-independent.

Returns `(T, w_mean, chi2, dchi2, dchi2_cal, T_map, T_min, w_mean_min, ci68, ci95,
var_scale)`; `dchi2_cal = dchi2/var_scale` and the CIs are `w̄` bounds (NaN where
the profile does not cross the threshold within the grid).
"""
function profile_inventory(flux_only, w_map::AbstractVector, free::AbstractVector{Bool},
                           meas, fit_mask, wdem, exposure;
                           var_scale::Float64 = 1.0,
                           n_grid::Int = 21, span::Tuple{Float64,Float64} = (0.2, 3.0),
                           w_max::Float64 = MAX_WATER_FRACTION)
    nfree = max(count(free), 1)
    T_map = sum(w_map)
    scales = collect(range(span[1], span[2], length = n_grid))
    Ts = Float64[]; wbar = Float64[]; chis = Float64[]
    for sc in scales
        w = clamp.(sc .* w_map, 0.0, w_max)
        T = sum(w)
        c, _ = _weighted_chi2(flux_only, w, meas, fit_mask, wdem, exposure)
        push!(Ts, T); push!(wbar, T / nfree); push!(chis, c)
    end
    kmin = argmin(chis)
    dchi = chis .- chis[kmin]
    dchi_cal = dchi ./ max(var_scale, 1e-30)
    # crossing of (calibrated) Δχ²=thr on each side of the minimum, in w̄ units
    function crossings(thr)
        lo = NaN; hi = NaN
        for k in kmin:-1:2
            if dchi_cal[k] <= thr <= dchi_cal[k-1]
                f = (thr - dchi_cal[k]) / (dchi_cal[k-1] - dchi_cal[k] + 1e-30)
                lo = wbar[k] + f * (wbar[k-1] - wbar[k]); break
            end
        end
        for k in kmin:(length(dchi_cal)-1)
            if dchi_cal[k] <= thr <= dchi_cal[k+1]
                f = (thr - dchi_cal[k]) / (dchi_cal[k+1] - dchi_cal[k] + 1e-30)
                hi = wbar[k] + f * (wbar[k+1] - wbar[k]); break
            end
        end
        return (lo, hi)
    end
    return (T = Ts, w_mean = wbar, chi2 = chis, dchi2 = dchi, dchi2_cal = dchi_cal,
            T_map = T_map, T_min = Ts[kmin], w_mean_min = wbar[kmin],
            ci68 = crossings(1.0), ci95 = crossings(3.84), var_scale = var_scale)
end
