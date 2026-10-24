# Unreferenced isolated instrumented copy, scientific source SHA256 1c708b159294c2eab4e7b3a6359284661883669adcc68c5e28253cafab4865c8
# Only changes: unique function name, trial_trace keyword, trace append.
@eval DiffPumas.Tomography begin
function replay_cache_gauss_newton_reconstruct(model, obs::AbstractVector;
                                  w0::AbstractVector,
                                  n_iter::Int = 12,
                                  box::Tuple{Float64,Float64} = (0.0, MAX_WATER_FRACTION),
                                  prior::Union{Nothing,SmoothnessPrior,EdgePrior} = nothing,
                                  lm_damping::Float64 = 1e-3,
                                  lm_factor::Float64 = 3.0,
                                  relinearize::Bool = true,
                                  weights::Union{Nothing,AbstractVector} = nothing,
                                  free::Union{Nothing,AbstractVector{Bool}} = nothing,
                                  predict_only = nothing,
                                  solver_mode::Symbol = :legacy,
                                  projected_gradient_tol::Float64 = 1e-6,
                                  objective_tol::Float64 = 0.0,
                                  diagnostics = nothing,
                                  diagnostic_callback = nothing,
                                  trial_trace = Any[])
    record_diagnostic(row) = begin
        diagnostics === nothing || push!(diagnostics,row)
        diagnostic_callback === nothing || diagnostic_callback(row)
    end
    solver_mode in (:active_set, :legacy) || throw(ArgumentError("unknown solver_mode"))
    w = clamp.(collect(Float64,w0),box[1],box[2])
    free === nothing || length(free)==length(w) || throw(DimensionMismatch("free mask must match w0"))
    active=free===nothing ? collect(eachindex(w)) : findall(free)
    history=Float64[]
    W=weights===nothing ? nothing : collect(Float64,weights)
    data_cost(r)=W===nothing ? 0.5*dot(r,r) : 0.5*sum(W.*r.^2)
    objective(pred,wv)=data_cost(pred.-obs)+prior_value(prior,wv)
    pred,J=model(w)
    linear_origin=copy(w);linear_pred=copy(pred);linear_J=J
    r=pred.-obs;f_cur=objective(pred,w);mu=lm_damping
    push!(history,norm(r))
    isempty(active) && return w,history

    for _ in 1:n_iter
        Ja=J[:,active]
        Jt=transpose(Ja)
        if W===nothing
            normal=Matrix(Jt*Ja);grad=Jt*r
        else
            normal=Matrix(Jt*(W.*Ja));grad=Jt*(W.*r)
        end
        if prior !== nothing
            pg=prior isa EdgePrior ? edge_gradient(prior,w) : laplacian_gradient(prior,w)
            grad=grad.+pg[active]
            normal=normal.+Matrix(_prior_hessian(prior,w)[active,active])
        end
        # KKT residual excludes outward gradients at active box constraints.
        pg = w[active] .- clamp.(w[active] .- grad, box[1], box[2])
        pg_norm = norm(pg, Inf)
        record_diagnostic((iteration=length(history)-1, objective=f_cur, projected_gradient=pg_norm))
        solver_mode === :active_set && pg_norm <= projected_gradient_tol && break
        movable = solver_mode === :legacy ? trues(length(active)) :
            .!(((w[active] .<= box[1]) .& (grad .> 0)) .|
               ((w[active] .>= box[2]) .& (grad .< 0)))
        d=diag(normal);dmax=maximum(d);dmax=dmax>0 ? dmax : 1.0
        accepted=false
        f_before=f_cur
        for _try in 1:(solver_mode === :active_set ? 7 : 6)
            A=normal+Diagonal(mu.*max.(d,1e-12*dmax))
            # If LM directions exhaust their searches, a bounded diagonally
            # scaled projected gradient supplies an independent descent safeguard.
            rescue = solver_mode === :active_set && _try == 7
            delta=rescue ? -grad ./ max.(d,maximum(abs.(grad)),1e-30) : try
                step = zeros(length(active))
                step[movable] .= -(A[movable,movable]\grad[movable])
                # Projection of a coupled Newton direction need not be descent.
                # Fall back to a diagonally scaled projected gradient direction.
                projected = clamp.(w[active] .+ step, box[1], box[2]) .- w[active]
                if solver_mode === :active_set && dot(grad,projected) >= 0
                    step .= -grad ./ max.(d,1e-12*dmax)
                end
                step
            catch
                solver_mode === :active_set ? -grad ./ max.(d,1e-12*dmax) : fill(0.0,length(active))
            end
            if !all(isfinite,delta)
                mu=min(mu*lm_factor,1e12);continue
            end
            alpha=1.0
            cached_w=nothing;cached_pred=nothing;cached_J=nothing
            for _ls in 1:(rescue ? 40 : 14)
                w_new=copy(w)
                w_new[active].=clamp.(w[active].+alpha.*delta,box[1],box[2])
                # Halving an exactly zero projected step cannot change the candidate.
                # Other LM levels may still propose a different direction.
                w_new==w && break
                if cached_w !== nothing && w_new==cached_w
                    pred_new=cached_pred;J_new=cached_J
                elseif relinearize && predict_only !== nothing
                    pred_new=predict_only(w_new);J_new=nothing
                    cached_w=w_new;cached_pred=pred_new;cached_J=nothing
                else
                    pred_new,J_new=relinearize ? model(w_new) : (linear_pred+linear_J*(w_new-linear_origin),linear_J)
                    cached_w=w_new;cached_pred=pred_new;cached_J=J_new
                end
                f_new=objective(pred_new,w_new)
                push!(trial_trace,(iteration=length(history)-1,lm_try=_try,halving=_ls,alpha=alpha,mu=mu,water=copy(w_new),prediction=copy(pred_new),objective=f_new,directional=dot(grad,w_new[active].-w[active])))
                if _gn_accept_trial(f_new,f_cur,grad,w_new[active].-w[active],solver_mode)
                    # The optional prediction callback must equal model(w)[1].
                    # Build a fresh derivative only for an accepted candidate.
                    if relinearize && predict_only !== nothing && J_new === nothing
                        pred_new,J_new=model(w_new)
                        f_new=objective(pred_new,w_new)
                        cached_pred=pred_new;cached_J=J_new
                    end
                    # Confirm the true objective after any callback/full-model
                    # discrepancy; an approximate callback cannot authorize an increase.
                    if _gn_accept_trial(f_new,f_cur,grad,w_new[active].-w[active],solver_mode)
                        w=w_new;r=pred_new.-obs;J=J_new;pred=pred_new;f_cur=f_new
                        accepted=true;break
                    end
                end
                alpha*=0.5
            end
            if accepted
                mu=max(mu/lm_factor,1e-12);break
            else
                mu=min(mu*lm_factor,1e12)
            end
        end
        push!(history,norm(r))
        accepted || break
        # Objective stagnation is recorded, but never called KKT convergence.
        solver_mode === :active_set && objective_tol > 0 && abs(f_before-f_cur) <= objective_tol*max(1.0,abs(f_before)) && break
    end
    if diagnostics !== nothing || diagnostic_callback !== nothing
        final_grad = transpose(J[:,active]) * (W === nothing ? r : W.*r)
        if prior !== nothing
            final_grad .+= (prior isa EdgePrior ? edge_gradient(prior,w) : laplacian_gradient(prior,w))[active]
        end
        final_pg = w[active] .- clamp.(w[active].-final_grad,box[1],box[2])
        record_diagnostic((iteration=length(history)-1,objective=f_cur,projected_gradient=norm(final_pg,Inf)))
    end
    return w,history
end

end
