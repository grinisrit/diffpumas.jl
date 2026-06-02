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
# Gradient descent on the nonlinear misfit using the AD Jacobian (our method)
# --------------------------------------------------------------------------

"""
    gradient_descent_reconstruct(model, obs; w0, n_iter=200, lr=1e-2,
                                 box=(0,0.9), prior=nothing, optimizer=:adam,
                                 relinearize=true) -> (w, history)

Minimise `½‖pred(w) − obs‖² + R(w)` by projected gradient descent. `model(w)`
returns `(pred::Vector, J)` with `J = ∂pred/∂w` (the AD sparse Jacobian); the
gradient is `Jᵀ(pred − obs) + ∂R/∂w`. With `relinearize=false` the Jacobian is
computed once at `w0` and reused (cheaper, linearised solve). `optimizer` is
`:adam` or `:gd`. This is the DiffPumas-native solver: the same differentiable
CSDA forward model used for sensitivities drives the inversion.
"""
function gradient_descent_reconstruct(model, obs::AbstractVector;
                                      w0::AbstractVector,
                                      n_iter::Int = 200, lr::Float64 = 1e-2,
                                      box::Tuple{Float64,Float64} = (0.0, MAX_WATER_FRACTION),
                                      prior::Union{Nothing,SmoothnessPrior} = nothing,
                                      optimizer::Symbol = :adam,
                                      relinearize::Bool = true)
    w = collect(Float64, w0)
    history = Float64[]
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
        g = transpose(J) * r
        if prior !== nothing
            g = g .+ laplacian_gradient(prior, w)
        end
        if optimizer === :adam
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

        accepted = false
        for _try in 1:8
            A = JtJ + Diagonal(μ .* max.(d, 1e-12 * dmax))
            δ = try
                -(A \ grad)
            catch
                fill(0.0, length(w))
            end
            w_new = clamp.(w .+ δ, box[1], box[2])
            pred_new, J_new = relinearize ? model(w_new) : (J * w_new, J)
            r_new = pred_new .- obs
            f_new = cost(r_new)
            if isfinite(f_new) && f_new < f_cur
                w = w_new; r = r_new; J = J_new; f_cur = f_new
                μ = max(μ / lm_factor, 1e-12)
                accepted = true
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
    reconstruction_report(recon, truth; roi_mask=nothing, bg_mask=nothing, peak=MAX_WATER_FRACTION)
        -> NamedTuple

Bundle of all comparison metrics between a reconstructed and a ground-truth field.
"""
function reconstruction_report(recon::AbstractVector, truth::AbstractVector;
                               roi_mask = nothing, bg_mask = nothing,
                               peak::Float64 = MAX_WATER_FRACTION)
    c = (roi_mask !== nothing && bg_mask !== nothing) ? cnr(recon, roi_mask, bg_mask) : NaN
    return (mse = mse(recon, truth), rmse = rmse(recon, truth),
            psnr = psnr(recon, truth; peak = peak),
            ssim = ssim_metric(recon, truth; peak = peak),
            snr = snr_metric(recon), cnr = c)
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
